#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

from tune_xgrammar_masked_sampler import (
    MASK_WORDS,
    VOCAB,
    check_status,
    configure,
    expected_sample,
    stream_pointer,
)
from tune_xgrammar_sparse_lm_head import (
    HIDDEN,
    allowed_ids,
    launch_dense,
    launch_sparse,
    parse_int_list,
)

REFERENCE_ROOT = Path(__file__).resolve().parents[1] / "reference"
sys.path.insert(0, str(REFERENCE_ROOT))

from glmrt_reference.triton_sampling_capture import (  # noqa: E402
    _sample_from_candidates,
)


MAX_BLOCK_VOCAB = 1_024


@triton.jit
def _compact_block_topk_mapped(
    compact_logits,
    allowed_token_ids,
    candidate_scores,
    candidate_indices,
    ALLOWED_TOKENS: tl.constexpr,
    TOP_K: tl.constexpr,
    NUM_VOCAB_BLOCKS: tl.constexpr,
    BLOCK_VOCAB: tl.constexpr,
) -> None:
    block = tl.program_id(0)
    offsets = block * BLOCK_VOCAB + tl.arange(0, BLOCK_VOCAB)
    valid = offsets < ALLOWED_TOKENS
    values = tl.load(compact_logits + offsets, mask=valid, other=-float("inf"))
    token_ids = tl.load(allowed_token_ids + offsets, mask=valid, other=0)
    selected = tl.full((BLOCK_VOCAB,), False, tl.int1)
    out_base = block * TOP_K

    # IDs are sorted, so compact-position tie breaking matches vocabulary-ID order.
    for rank in range(0, TOP_K):
        candidates = tl.where(selected, -float("inf"), values)
        best_score = tl.max(candidates, axis=0)
        best_mask = valid & (candidates == best_score)
        best_offset = tl.min(
            tl.where(best_mask, offsets, ALLOWED_TOKENS), axis=0
        )
        best_token = tl.sum(tl.where(offsets == best_offset, token_ids, 0), axis=0)
        tl.store(candidate_scores + out_base + rank, best_score)
        tl.store(candidate_indices + out_base + rank, best_token)
        selected = selected | (offsets == best_offset)


def capture(operation) -> torch.cuda.CUDAGraph:
    operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()
    return graph


def measure(
    graphs: list[torch.cuda.CUDAGraph],
    warmup: int,
    iterations: int,
    repeats: int,
) -> list[float]:
    for iteration in range(warmup):
        graphs[iteration % len(graphs)].replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for iteration in range(iterations):
            graphs[iteration % len(graphs)].replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return samples


def summarize(samples: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def packed_mask(host_ids: torch.Tensor) -> torch.Tensor:
    token_ids = host_ids.to(torch.int64)
    words = token_ids // 32
    bits = torch.bitwise_left_shift(torch.ones_like(token_ids), token_ids.remainder(32))
    packed = torch.zeros(MASK_WORDS, dtype=torch.int64)
    packed.index_add_(0, words, bits)
    host_mask = torch.empty(MASK_WORDS, dtype=torch.int32, pin_memory=True)
    host_mask.copy_(packed.to(torch.int32))
    return host_mask


def launch_compact_sample(
    compact_logits: torch.Tensor,
    allowed_token_ids: torch.Tensor,
    candidate_scores: torch.Tensor,
    candidate_indices: torch.Tensor,
    random_uniforms: torch.Tensor,
    out_argmax_indices: torch.Tensor,
    out_argmax_scores: torch.Tensor,
    out_indices: torch.Tensor,
    out_scores: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> None:
    count = compact_logits.numel()
    block_vocab = min(MAX_BLOCK_VOCAB, triton.next_power_of_2(count))
    num_vocab_blocks = triton.cdiv(count, block_vocab)
    block_candidates = triton.next_power_of_2(num_vocab_blocks * top_k)
    top_k_block = triton.next_power_of_2(top_k)
    _compact_block_topk_mapped[(num_vocab_blocks,)](
        compact_logits,
        allowed_token_ids,
        candidate_scores,
        candidate_indices,
        ALLOWED_TOKENS=count,
        TOP_K=top_k,
        NUM_VOCAB_BLOCKS=num_vocab_blocks,
        BLOCK_VOCAB=block_vocab,
    )
    _sample_from_candidates[(1,)](
        candidate_scores,
        candidate_indices,
        random_uniforms,
        out_argmax_indices,
        out_argmax_scores,
        out_indices,
        out_scores,
        TEMPERATURE=temperature,
        TOP_K=top_k,
        TOP_P=top_p,
        NUM_VOCAB_BLOCKS=num_vocab_blocks,
        BLOCK_CANDIDATES=block_candidates,
        TOP_K_BLOCK=top_k_block,
    )


def validate_outputs(
    label: str,
    actual_indices: torch.Tensor,
    actual_scores: torch.Tensor,
    expected_indices: torch.Tensor,
    expected_scores: torch.Tensor,
) -> None:
    if not torch.equal(actual_indices, expected_indices):
        raise RuntimeError(
            f"{label} indices differ: got={actual_indices.cpu().tolist()} "
            f"expected={expected_indices.cpu().tolist()}"
        )
    if not torch.allclose(actual_scores, expected_scores, rtol=2e-5, atol=2e-6):
        max_abs = (actual_scores - expected_scores).abs().max().item()
        raise RuntimeError(f"{label} scores differ, max_abs={max_abs}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare dense masked sampling with a compact allowed-row LM-head graph."
        )
    )
    parser.add_argument("--native-lib", type=Path, required=True)
    parser.add_argument(
        "--allowed", default="8,256,4096,16384,38720,77440,116160,154880"
    )
    parser.add_argument("--top-k", default="1,8")
    parser.add_argument("--sets", type=int, default=4)
    parser.add_argument("--native-grid-blocks", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    counts = parse_int_list(args.allowed, "allowed")
    top_ks = parse_int_list(args.top_k, "top-k")
    if any(count > VOCAB for count in counts):
        parser.error(f"allowed counts must not exceed {VOCAB}")
    if any(top_k > 8 for top_k in top_ks):
        parser.error("top-k values must not exceed 8")
    if min(args.sets, args.iterations, args.repeats) < 1 or args.warmup < 0:
        parser.error("sets/iterations/repeats must be positive and warmup nonnegative")
    if not 1 <= args.native_grid_blocks <= 256:
        parser.error("native-grid-blocks must be between 1 and 256")
    if args.temperature <= 0.0 or not 0.0 < args.top_p <= 1.0:
        parser.error("temperature must be positive and top-p must be in (0, 1]")

    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    torch.manual_seed(args.seed)
    lib = ctypes.CDLL(str(args.native_lib.resolve()))
    lib.glmrt_last_error.argtypes = (ctypes.c_char_p, ctypes.c_size_t)
    lib.glmrt_last_error.restype = ctypes.c_int
    native_sampler = configure(
        lib,
        "glmrt_cuda_logits_masked_sample_topk_topp_f32_grid_candidate_async",
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_void_p,
        ),
    )

    weight = torch.empty((VOCAB, HIDDEN), dtype=torch.bfloat16, device=device)
    weight.uniform_(-0.02, 0.02)
    hidden_sets = [
        torch.randn((1, HIDDEN), dtype=torch.bfloat16, device=device)
        for _ in range(args.sets)
    ]
    random_uniforms = torch.tensor([0.37], dtype=torch.float32, device=device)
    cases = []

    for count in counts:
        for top_k in top_ks:
            if top_k > count:
                continue
            block_vocab = min(MAX_BLOCK_VOCAB, triton.next_power_of_2(count))
            num_vocab_blocks = triton.cdiv(count, block_vocab)
            dense_graphs = []
            sparse_graphs = []
            graph_lifetimes = []
            max_logit_abs = 0.0
            bitwise_logits = True

            for set_index in range(args.sets):
                hidden = hidden_sets[set_index]
                host_ids = allowed_ids(count, set_index)
                device_ids = host_ids.cuda(non_blocking=False)
                host_mask = packed_mask(host_ids)
                device_mask = host_mask.cuda(non_blocking=False)
                dense_logits = torch.empty((1, VOCAB), dtype=torch.float32, device=device)
                sparse_logits = torch.empty((1, count), dtype=torch.float32, device=device)
                candidate_scores = torch.empty(
                    num_vocab_blocks * top_k, dtype=torch.float32, device=device
                )
                candidate_indices = torch.empty(
                    num_vocab_blocks * top_k, dtype=torch.uint32, device=device
                )
                partial_keys = torch.empty(
                    args.native_grid_blocks * 8, dtype=torch.uint64, device=device
                )
                dense_indices = torch.empty(1, dtype=torch.uint32, device=device)
                dense_scores = torch.empty(1, dtype=torch.float32, device=device)
                sparse_argmax_indices = torch.empty(
                    1, dtype=torch.uint32, device=device
                )
                sparse_argmax_scores = torch.empty(
                    1, dtype=torch.float32, device=device
                )
                sparse_indices = torch.empty(1, dtype=torch.uint32, device=device)
                sparse_scores = torch.empty(1, dtype=torch.float32, device=device)

                def dense_operation(
                    hidden=hidden,
                    host_mask=host_mask,
                    device_mask=device_mask,
                    dense_logits=dense_logits,
                    partial_keys=partial_keys,
                    dense_indices=dense_indices,
                    dense_scores=dense_scores,
                ) -> None:
                    device_mask.copy_(host_mask, non_blocking=True)
                    launch_dense(hidden, weight, dense_logits)
                    check_status(
                        lib,
                        native_sampler(
                            ctypes.c_void_p(dense_logits.data_ptr()),
                            ctypes.c_void_p(device_mask.data_ptr()),
                            ctypes.c_void_p(random_uniforms.data_ptr()),
                            ctypes.c_void_p(partial_keys.data_ptr()),
                            ctypes.c_void_p(dense_indices.data_ptr()),
                            ctypes.c_void_p(dense_scores.data_ptr()),
                            1,
                            VOCAB,
                            MASK_WORDS,
                            args.native_grid_blocks,
                            args.temperature,
                            top_k,
                            args.top_p,
                            stream_pointer(),
                        ),
                        "launch dense masked sampler",
                    )

                def sparse_operation(
                    hidden=hidden,
                    host_ids=host_ids,
                    device_ids=device_ids,
                    sparse_logits=sparse_logits,
                    candidate_scores=candidate_scores,
                    candidate_indices=candidate_indices,
                    sparse_argmax_indices=sparse_argmax_indices,
                    sparse_argmax_scores=sparse_argmax_scores,
                    sparse_indices=sparse_indices,
                    sparse_scores=sparse_scores,
                ) -> None:
                    device_ids.copy_(host_ids, non_blocking=True)
                    launch_sparse(hidden, weight, device_ids, sparse_logits)
                    launch_compact_sample(
                        sparse_logits,
                        device_ids,
                        candidate_scores,
                        candidate_indices,
                        random_uniforms,
                        sparse_argmax_indices,
                        sparse_argmax_scores,
                        sparse_indices,
                        sparse_scores,
                        temperature=args.temperature,
                        top_k=top_k,
                        top_p=args.top_p,
                    )

                dense_operation()
                sparse_operation()
                torch.cuda.synchronize()
                selected_dense = dense_logits.index_select(1, device_ids.to(torch.int64))
                bitwise_logits = bitwise_logits and torch.equal(
                    sparse_logits, selected_dense
                )
                max_logit_abs = max(
                    max_logit_abs,
                    (sparse_logits - selected_dense).abs().max().item(),
                )
                allowed = torch.zeros((1, VOCAB), dtype=torch.bool, device=device)
                allowed[0, device_ids.to(torch.int64)] = True
                expected_indices, expected_scores = expected_sample(
                    dense_logits,
                    allowed,
                    random_uniforms,
                    top_k,
                    args.temperature,
                    args.top_p,
                )
                validate_outputs(
                    "dense native",
                    dense_indices,
                    dense_scores,
                    expected_indices,
                    expected_scores,
                )
                validate_outputs(
                    "sparse compact",
                    sparse_indices,
                    sparse_scores,
                    expected_indices,
                    expected_scores,
                )
                dense_graphs.append(capture(dense_operation))
                sparse_graphs.append(capture(sparse_operation))
                graph_lifetimes.append(
                    (
                        host_ids,
                        device_ids,
                        host_mask,
                        device_mask,
                        dense_logits,
                        sparse_logits,
                        candidate_scores,
                        candidate_indices,
                        partial_keys,
                        dense_indices,
                        dense_scores,
                        sparse_argmax_indices,
                        sparse_argmax_scores,
                        sparse_indices,
                        sparse_scores,
                    )
                )

            dense_timing = summarize(
                measure(dense_graphs, args.warmup, args.iterations, args.repeats)
            )
            sparse_timing = summarize(
                measure(sparse_graphs, args.warmup, args.iterations, args.repeats)
            )
            cases.append(
                {
                    "allowed_tokens": count,
                    "allowed_fraction": count / VOCAB,
                    "top_k": top_k,
                    "block_vocab": block_vocab,
                    "num_vocab_blocks": num_vocab_blocks,
                    "bitwise_dense_selected_logits": bitwise_logits,
                    "max_abs_dense_selected_logits": max_logit_abs,
                    "exact_sample_indices": True,
                    "dense_masked_graph": dense_timing,
                    "sparse_compact_graph": sparse_timing,
                    "speedup": dense_timing["median_ms"]
                    / sparse_timing["median_ms"],
                }
            )

    print(
        json.dumps(
            {
                "benchmark": "xgrammar_sparse_lm_head_compact_sample_candidate",
                "status": "ok",
                "device": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "rows": 1,
                "hidden": HIDDEN,
                "vocab": VOCAB,
                "weight_bytes": VOCAB * HIDDEN * 2,
                "sets": args.sets,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "native_grid_blocks": args.native_grid_blocks,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "repeats": args.repeats,
                "cases": cases,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
