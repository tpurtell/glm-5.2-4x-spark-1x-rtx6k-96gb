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
    BLOCK_K,
    BLOCK_M,
    BLOCK_N,
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


BLOCK_VOCAB = 1_024


@triton.jit
def _independent_sparse_lm_head_logits_bf16(
    hidden,
    lm_head,
    allowed_token_ids,
    logits,
    ROWS: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    ALLOWED_TOKENS: tl.constexpr,
    HIDDEN_STRIDE_ROW: tl.constexpr,
    IDS_STRIDE_ROW: tl.constexpr,
    LOGITS_STRIDE_ROW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_n = offs_n < ALLOWED_TOKENS
    token_ids = tl.load(
        allowed_token_ids + row * IDS_STRIDE_ROW + offs_n,
        mask=valid_n,
        other=0,
    )
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, HIDDEN_DIM, BLOCK_K):
        k = k0 + offs_k
        hidden_values = tl.load(
            hidden
            + row * HIDDEN_STRIDE_ROW
            + offs_m[:, None] * 0
            + k[None, :],
            mask=(offs_m[:, None] == 0) & (k[None, :] < HIDDEN_DIM),
            other=0.0,
        )
        weight_values = tl.load(
            lm_head + token_ids[None, :] * HIDDEN_DIM + k[:, None],
            mask=valid_n[None, :] & (k[:, None] < HIDDEN_DIM),
            other=0.0,
        )
        acc += tl.dot(hidden_values, weight_values)
    tl.store(
        logits
        + row * LOGITS_STRIDE_ROW
        + offs_m[:, None] * 0
        + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] == 0) & valid_n[None, :],
    )


@triton.jit
def _compact_block_topk_mapped_rows(
    compact_logits,
    allowed_token_ids,
    candidate_scores,
    candidate_indices,
    ALLOWED_TOKENS: tl.constexpr,
    TOP_K: tl.constexpr,
    NUM_VOCAB_BLOCKS: tl.constexpr,
    IDS_STRIDE_ROW: tl.constexpr,
    LOGITS_STRIDE_ROW: tl.constexpr,
    BLOCK_VOCAB: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK_VOCAB + tl.arange(0, BLOCK_VOCAB)
    valid = offsets < ALLOWED_TOKENS
    values = tl.load(
        compact_logits + row * LOGITS_STRIDE_ROW + offsets,
        mask=valid,
        other=-float("inf"),
    )
    token_ids = tl.load(
        allowed_token_ids + row * IDS_STRIDE_ROW + offsets,
        mask=valid,
        other=0,
    )
    selected = tl.full((BLOCK_VOCAB,), False, tl.int1)
    out_base = (row * NUM_VOCAB_BLOCKS + block) * TOP_K

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


def make_host_ids(rows: int, count: int, set_index: int) -> torch.Tensor:
    host_ids = torch.empty((rows, count), dtype=torch.int32, pin_memory=True)
    for row in range(rows):
        host_ids[row].copy_(allowed_ids(count, set_index * rows + row))
    return host_ids


def make_host_masks(host_ids: torch.Tensor) -> torch.Tensor:
    rows = host_ids.shape[0]
    host_masks = torch.empty((rows, MASK_WORDS), dtype=torch.int32, pin_memory=True)
    for row in range(rows):
        token_ids = host_ids[row].to(torch.int64)
        words = token_ids // 32
        bits = torch.bitwise_left_shift(
            torch.ones_like(token_ids), token_ids.remainder(32)
        )
        packed = torch.zeros(MASK_WORDS, dtype=torch.int64)
        packed.index_add_(0, words, bits)
        host_masks[row].copy_(packed.to(torch.int32))
    return host_masks


def launch_independent_projection(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    token_ids: torch.Tensor,
    output: torch.Tensor,
) -> None:
    rows, count = token_ids.shape
    _independent_sparse_lm_head_logits_bf16[
        (rows, triton.cdiv(count, BLOCK_N))
    ](
        hidden,
        weight,
        token_ids,
        output,
        ROWS=rows,
        HIDDEN_DIM=HIDDEN,
        ALLOWED_TOKENS=count,
        HIDDEN_STRIDE_ROW=HIDDEN,
        IDS_STRIDE_ROW=count,
        LOGITS_STRIDE_ROW=count,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )


def launch_compact_sample_rows(
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
    rows: int,
    count: int,
    ids_stride_row: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> None:
    block_vocab = min(BLOCK_VOCAB, triton.next_power_of_2(count))
    num_vocab_blocks = triton.cdiv(count, block_vocab)
    _compact_block_topk_mapped_rows[(rows, num_vocab_blocks)](
        compact_logits,
        allowed_token_ids,
        candidate_scores,
        candidate_indices,
        ALLOWED_TOKENS=count,
        TOP_K=top_k,
        NUM_VOCAB_BLOCKS=num_vocab_blocks,
        IDS_STRIDE_ROW=ids_stride_row,
        LOGITS_STRIDE_ROW=count,
        BLOCK_VOCAB=block_vocab,
    )
    _sample_from_candidates[(rows,)](
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
        BLOCK_CANDIDATES=triton.next_power_of_2(num_vocab_blocks * top_k),
        TOP_K_BLOCK=triton.next_power_of_2(top_k),
    )


def validate_sample(
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
        description="Measure distinct-mask sparse constrained sampling at MTP widths."
    )
    parser.add_argument("--native-lib", type=Path, required=True)
    parser.add_argument("--rows", default="1,2,4,6")
    parser.add_argument("--allowed", default="8,256,1024,4096,38720,116160")
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

    row_counts = parse_int_list(args.rows, "rows")
    allowed_counts = parse_int_list(args.allowed, "allowed")
    top_ks = parse_int_list(args.top_k, "top-k")
    if any(rows > 16 for rows in row_counts):
        parser.error("rows must not exceed 16")
    if any(count > VOCAB for count in allowed_counts):
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
    cases = []

    for rows in row_counts:
        hidden_sets = [
            torch.randn((rows, HIDDEN), dtype=torch.bfloat16, device=device)
            for _ in range(args.sets)
        ]
        random_uniforms = torch.linspace(
            0.17, 0.83, rows, dtype=torch.float32, device=device
        )
        for count in allowed_counts:
            for top_k in top_ks:
                if top_k > count:
                    continue
                block_vocab = min(BLOCK_VOCAB, triton.next_power_of_2(count))
                num_vocab_blocks = triton.cdiv(count, block_vocab)
                dense_graphs = []
                serial_graphs = []
                independent_graphs = []
                shared_graphs = []
                graph_lifetimes = []
                exact_logits = True
                max_logit_abs = 0.0

                for set_index in range(args.sets):
                    hidden = hidden_sets[set_index]
                    host_ids = make_host_ids(rows, count, set_index)
                    device_ids = host_ids.cuda(non_blocking=False)
                    host_shared_ids = host_ids[0].clone().pin_memory()
                    device_shared_ids = host_shared_ids.cuda(non_blocking=False)
                    host_masks = make_host_masks(host_ids)
                    device_masks = host_masks.cuda(non_blocking=False)
                    dense_logits = torch.empty(
                        (rows, VOCAB), dtype=torch.float32, device=device
                    )
                    serial_logits = torch.empty(
                        (rows, count), dtype=torch.float32, device=device
                    )
                    independent_logits = torch.empty_like(serial_logits)
                    shared_logits = torch.empty_like(serial_logits)
                    partial_keys = torch.empty(
                        rows * args.native_grid_blocks * 8,
                        dtype=torch.uint64,
                        device=device,
                    )

                    def sampler_buffers():
                        return (
                            torch.empty(
                                (rows, num_vocab_blocks, top_k),
                                dtype=torch.float32,
                                device=device,
                            ),
                            torch.empty(
                                (rows, num_vocab_blocks, top_k),
                                dtype=torch.uint32,
                                device=device,
                            ),
                            torch.empty(rows, dtype=torch.uint32, device=device),
                            torch.empty(rows, dtype=torch.float32, device=device),
                            torch.empty(rows, dtype=torch.uint32, device=device),
                            torch.empty(rows, dtype=torch.float32, device=device),
                        )

                    serial_buffers = sampler_buffers()
                    independent_buffers = sampler_buffers()
                    shared_buffers = sampler_buffers()
                    dense_indices = torch.empty(
                        rows, dtype=torch.uint32, device=device
                    )
                    dense_scores = torch.empty(
                        rows, dtype=torch.float32, device=device
                    )

                    def dense_operation() -> None:
                        device_masks.copy_(host_masks, non_blocking=True)
                        launch_dense(hidden, weight, dense_logits)
                        check_status(
                            lib,
                            native_sampler(
                                ctypes.c_void_p(dense_logits.data_ptr()),
                                ctypes.c_void_p(device_masks.data_ptr()),
                                ctypes.c_void_p(random_uniforms.data_ptr()),
                                ctypes.c_void_p(partial_keys.data_ptr()),
                                ctypes.c_void_p(dense_indices.data_ptr()),
                                ctypes.c_void_p(dense_scores.data_ptr()),
                                rows,
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

                    def serial_operation() -> None:
                        device_ids.copy_(host_ids, non_blocking=True)
                        for row in range(rows):
                            launch_sparse(
                                hidden[row : row + 1],
                                weight,
                                device_ids[row],
                                serial_logits[row : row + 1],
                            )
                            launch_compact_sample_rows(
                                serial_logits[row : row + 1],
                                device_ids[row : row + 1],
                                serial_buffers[0][row : row + 1],
                                serial_buffers[1][row : row + 1],
                                random_uniforms[row : row + 1],
                                serial_buffers[2][row : row + 1],
                                serial_buffers[3][row : row + 1],
                                serial_buffers[4][row : row + 1],
                                serial_buffers[5][row : row + 1],
                                rows=1,
                                count=count,
                                ids_stride_row=count,
                                temperature=args.temperature,
                                top_k=top_k,
                                top_p=args.top_p,
                            )

                    def independent_operation() -> None:
                        device_ids.copy_(host_ids, non_blocking=True)
                        launch_independent_projection(
                            hidden, weight, device_ids, independent_logits
                        )
                        launch_compact_sample_rows(
                            independent_logits,
                            device_ids,
                            independent_buffers[0],
                            independent_buffers[1],
                            random_uniforms,
                            independent_buffers[2],
                            independent_buffers[3],
                            independent_buffers[4],
                            independent_buffers[5],
                            rows=rows,
                            count=count,
                            ids_stride_row=count,
                            temperature=args.temperature,
                            top_k=top_k,
                            top_p=args.top_p,
                        )

                    def shared_operation() -> None:
                        device_shared_ids.copy_(host_shared_ids, non_blocking=True)
                        launch_sparse(
                            hidden, weight, device_shared_ids, shared_logits
                        )
                        launch_compact_sample_rows(
                            shared_logits,
                            device_shared_ids,
                            shared_buffers[0],
                            shared_buffers[1],
                            random_uniforms,
                            shared_buffers[2],
                            shared_buffers[3],
                            shared_buffers[4],
                            shared_buffers[5],
                            rows=rows,
                            count=count,
                            ids_stride_row=0,
                            temperature=args.temperature,
                            top_k=top_k,
                            top_p=args.top_p,
                        )

                    dense_operation()
                    serial_operation()
                    independent_operation()
                    shared_operation()
                    torch.cuda.synchronize()
                    selected_dense = torch.stack(
                        [
                            dense_logits[row].index_select(
                                0, device_ids[row].to(torch.int64)
                            )
                            for row in range(rows)
                        ]
                    )
                    for label, logits in (
                        ("serial", serial_logits),
                        ("independent", independent_logits),
                    ):
                        exact_logits = exact_logits and torch.equal(
                            logits, selected_dense
                        )
                        max_logit_abs = max(
                            max_logit_abs,
                            (logits - selected_dense).abs().max().item(),
                        )
                    allowed = torch.zeros(
                        (rows, VOCAB), dtype=torch.bool, device=device
                    )
                    allowed.scatter_(1, device_ids.to(torch.int64), True)
                    expected_indices, expected_scores = expected_sample(
                        dense_logits,
                        allowed,
                        random_uniforms,
                        top_k,
                        args.temperature,
                        args.top_p,
                    )
                    validate_sample(
                        "dense native",
                        dense_indices,
                        dense_scores,
                        expected_indices,
                        expected_scores,
                    )
                    validate_sample(
                        "serial sparse",
                        serial_buffers[4],
                        serial_buffers[5],
                        expected_indices,
                        expected_scores,
                    )
                    validate_sample(
                        "independent sparse",
                        independent_buffers[4],
                        independent_buffers[5],
                        expected_indices,
                        expected_scores,
                    )
                    shared_allowed = torch.zeros_like(allowed)
                    shared_allowed[:, device_shared_ids.to(torch.int64)] = True
                    shared_expected_indices, shared_expected_scores = expected_sample(
                        dense_logits,
                        shared_allowed,
                        random_uniforms,
                        top_k,
                        args.temperature,
                        args.top_p,
                    )
                    validate_sample(
                        "shared sparse",
                        shared_buffers[4],
                        shared_buffers[5],
                        shared_expected_indices,
                        shared_expected_scores,
                    )

                    dense_graphs.append(capture(dense_operation))
                    serial_graphs.append(capture(serial_operation))
                    independent_graphs.append(capture(independent_operation))
                    shared_graphs.append(capture(shared_operation))
                    graph_lifetimes.append(
                        (
                            host_ids,
                            device_ids,
                            host_shared_ids,
                            device_shared_ids,
                            host_masks,
                            device_masks,
                            dense_logits,
                            serial_logits,
                            independent_logits,
                            shared_logits,
                            partial_keys,
                            serial_buffers,
                            independent_buffers,
                            shared_buffers,
                            dense_indices,
                            dense_scores,
                        )
                    )

                timings = {
                    "dense_masked": summarize(
                        measure(
                            dense_graphs,
                            args.warmup,
                            args.iterations,
                            args.repeats,
                        )
                    ),
                    "serial_distinct": summarize(
                        measure(
                            serial_graphs,
                            args.warmup,
                            args.iterations,
                            args.repeats,
                        )
                    ),
                    "fused_distinct": summarize(
                        measure(
                            independent_graphs,
                            args.warmup,
                            args.iterations,
                            args.repeats,
                        )
                    ),
                    "shared_mask": summarize(
                        measure(
                            shared_graphs,
                            args.warmup,
                            args.iterations,
                            args.repeats,
                        )
                    ),
                }
                dense_ms = timings["dense_masked"]["median_ms"]
                cases.append(
                    {
                        "rows": rows,
                        "allowed_tokens_per_row": count,
                        "allowed_fraction": count / VOCAB,
                        "top_k": top_k,
                        "bitwise_dense_selected_logits": exact_logits,
                        "max_abs_dense_selected_logits": max_logit_abs,
                        "exact_sample_indices": True,
                        "timings": timings,
                        "speedup_fused_distinct_vs_dense": dense_ms
                        / timings["fused_distinct"]["median_ms"],
                        "speedup_fused_vs_serial_distinct": timings[
                            "serial_distinct"
                        ]["median_ms"]
                        / timings["fused_distinct"]["median_ms"],
                        "speedup_shared_vs_dense": dense_ms
                        / timings["shared_mask"]["median_ms"],
                    }
                )

    print(
        json.dumps(
            {
                "benchmark": "xgrammar_sparse_lm_head_mtp_candidate",
                "status": "ok",
                "device": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
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
