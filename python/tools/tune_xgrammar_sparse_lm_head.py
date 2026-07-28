#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REFERENCE_ROOT = Path(__file__).resolve().parents[1] / "reference"
sys.path.insert(0, str(REFERENCE_ROOT))

from glmrt_reference.triton_sampling_capture import (  # noqa: E402
    _lm_head_logits_bf16,
)


HIDDEN = 6_144
VOCAB = 154_880
BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 64


@triton.jit
def _sparse_lm_head_logits_bf16(
    hidden,
    lm_head,
    allowed_token_ids,
    logits,
    ROWS: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    ALLOWED_TOKENS: tl.constexpr,
    HIDDEN_STRIDE_ROW: tl.constexpr,
    LM_HEAD_STRIDE_TOKEN: tl.constexpr,
    LOGITS_STRIDE_ROW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_n = offs_n < ALLOWED_TOKENS
    token_ids = tl.load(allowed_token_ids + offs_n, mask=valid_n, other=0)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, HIDDEN_DIM, BLOCK_K):
        k = k0 + offs_k
        hidden_values = tl.load(
            hidden + offs_m[:, None] * HIDDEN_STRIDE_ROW + k[None, :],
            mask=(offs_m[:, None] < ROWS) & (k[None, :] < HIDDEN_DIM),
            other=0.0,
        )
        weight_values = tl.load(
            lm_head + token_ids[None, :] * LM_HEAD_STRIDE_TOKEN + k[:, None],
            mask=valid_n[None, :] & (k[:, None] < HIDDEN_DIM),
            other=0.0,
        )
        acc += tl.dot(hidden_values, weight_values)
    tl.store(
        logits + offs_m[:, None] * LOGITS_STRIDE_ROW + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < ROWS) & valid_n[None, :],
    )


def parse_int_list(value: str, label: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",") if item]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{label} must be comma-separated integers") from error
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError(f"{label} values must be positive")
    return values


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


def allowed_ids(count: int, set_index: int) -> torch.Tensor:
    if count > VOCAB:
        raise ValueError(f"allowed token count {count} exceeds vocabulary {VOCAB}")
    permutation = (torch.arange(VOCAB, dtype=torch.int64) * 7_919 + set_index * 1_009 + 17) % VOCAB
    selected = permutation[:count].sort().values.to(torch.int32)
    pinned = torch.empty(count, dtype=torch.int32, pin_memory=True)
    pinned.copy_(selected)
    return pinned


def launch_dense(hidden: torch.Tensor, weight: torch.Tensor, output: torch.Tensor) -> None:
    _lm_head_logits_bf16[
        (triton.cdiv(hidden.shape[0], BLOCK_M), triton.cdiv(VOCAB, BLOCK_N))
    ](
        hidden,
        weight,
        output,
        ROWS=hidden.shape[0],
        HIDDEN_DIM=HIDDEN,
        VOCAB=VOCAB,
        HIDDEN_STRIDE_ROW=HIDDEN,
        LM_HEAD_STRIDE_TOKEN=HIDDEN,
        LOGITS_STRIDE_ROW=VOCAB,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )


def launch_sparse(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    token_ids: torch.Tensor,
    output: torch.Tensor,
) -> None:
    count = token_ids.numel()
    _sparse_lm_head_logits_bf16[
        (triton.cdiv(hidden.shape[0], BLOCK_M), triton.cdiv(count, BLOCK_N))
    ](
        hidden,
        weight,
        token_ids,
        output,
        ROWS=hidden.shape[0],
        HIDDEN_DIM=HIDDEN,
        ALLOWED_TOKENS=count,
        HIDDEN_STRIDE_ROW=HIDDEN,
        LM_HEAD_STRIDE_TOKEN=HIDDEN,
        LOGITS_STRIDE_ROW=count,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the production dense BF16 LM head with sparse allowed-row gathers."
    )
    parser.add_argument(
        "--allowed",
        default="8,256,4096,16384,38720,77440,154880",
        help="Comma-separated allowed-token counts.",
    )
    parser.add_argument("--sets", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    counts = parse_int_list(args.allowed, "allowed")
    if any(count > VOCAB for count in counts):
        parser.error(f"allowed counts must not exceed {VOCAB}")
    if min(args.sets, args.iterations, args.repeats) < 1 or args.warmup < 0:
        parser.error("sets/iterations/repeats must be positive and warmup nonnegative")

    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    torch.manual_seed(args.seed)
    weight = torch.empty((VOCAB, HIDDEN), dtype=torch.bfloat16, device=device)
    weight.uniform_(-0.02, 0.02)
    hidden_sets = [
        torch.randn((1, HIDDEN), dtype=torch.bfloat16, device=device)
        for _ in range(args.sets)
    ]
    dense_outputs = [
        torch.empty((1, VOCAB), dtype=torch.float32, device=device)
        for _ in range(args.sets)
    ]
    dense_graphs = [
        capture(lambda hidden=hidden, output=output: launch_dense(hidden, weight, output))
        for hidden, output in zip(hidden_sets, dense_outputs, strict=True)
    ]
    dense_hot = summarize(
        measure([dense_graphs[0]], args.warmup, args.iterations, args.repeats)
    )
    dense_rotating = summarize(
        measure(dense_graphs, args.warmup, args.iterations, args.repeats)
    )

    cases = []
    for count in counts:
        host_id_sets = [allowed_ids(count, set_index) for set_index in range(args.sets)]
        device_id_sets = [host_ids.cuda(non_blocking=False) for host_ids in host_id_sets]
        sparse_outputs = [
            torch.empty((1, count), dtype=torch.float32, device=device)
            for _ in range(args.sets)
        ]
        sparse_graphs = []
        copy_sparse_graphs = []
        max_abs = 0.0
        bitwise = True
        for set_index in range(args.sets):
            hidden = hidden_sets[set_index]
            host_ids = host_id_sets[set_index]
            device_ids = device_id_sets[set_index]
            output = sparse_outputs[set_index]

            def sparse_operation(
                hidden=hidden,
                device_ids=device_ids,
                output=output,
            ) -> None:
                launch_sparse(hidden, weight, device_ids, output)

            def copy_sparse_operation(
                hidden=hidden,
                host_ids=host_ids,
                device_ids=device_ids,
                output=output,
            ) -> None:
                device_ids.copy_(host_ids, non_blocking=True)
                launch_sparse(hidden, weight, device_ids, output)

            sparse_operation()
            torch.cuda.synchronize()
            reference = dense_outputs[set_index].index_select(
                1, device_ids.to(torch.int64)
            )
            bitwise = bitwise and torch.equal(output, reference)
            max_abs = max(max_abs, (output - reference).abs().max().item())
            sparse_graphs.append(capture(sparse_operation))
            copy_sparse_graphs.append(capture(copy_sparse_operation))

        sparse_hot = summarize(
            measure([sparse_graphs[0]], args.warmup, args.iterations, args.repeats)
        )
        sparse_rotating = summarize(
            measure(sparse_graphs, args.warmup, args.iterations, args.repeats)
        )
        copy_sparse_rotating = summarize(
            measure(copy_sparse_graphs, args.warmup, args.iterations, args.repeats)
        )
        cases.append(
            {
                "allowed_tokens": count,
                "allowed_fraction": count / VOCAB,
                "allowed_id_bytes": count * 4,
                "weight_bytes_selected": count * HIDDEN * 2,
                "bitwise_dense_selected_logits": bitwise,
                "max_abs_dense_selected_logits": max_abs,
                "sparse_hot": sparse_hot,
                "sparse_rotating": sparse_rotating,
                "pinned_copy_sparse_rotating": copy_sparse_rotating,
                "speedup_vs_dense_rotating": (
                    dense_rotating["median_ms"] / sparse_rotating["median_ms"]
                ),
                "speedup_including_id_copy": (
                    dense_rotating["median_ms"]
                    / copy_sparse_rotating["median_ms"]
                ),
            }
        )

    print(
        json.dumps(
            {
                "benchmark": "xgrammar_sparse_lm_head_candidate",
                "status": "ok",
                "device": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "rows": 1,
                "hidden": HIDDEN,
                "vocab": VOCAB,
                "weight_bytes": VOCAB * HIDDEN * 2,
                "sets": args.sets,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "repeats": args.repeats,
                "dense_hot": dense_hot,
                "dense_rotating": dense_rotating,
                "cases": cases,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
