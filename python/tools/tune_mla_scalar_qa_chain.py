#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import statistics
from pathlib import Path

import torch


HIDDEN = 6144
Q_LORA_RANK = 2048


def check_status(lib: ctypes.CDLL, status: int, action: str) -> None:
    if status == 0:
        return
    error = ctypes.create_string_buffer(512)
    lib.glmrt_last_error(error, len(error))
    raise RuntimeError(f"{action} failed with status {status}: {error.value.decode()}")


def stream_pointer() -> ctypes.c_void_p:
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


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


def parse_rows(raw: str) -> tuple[int, ...]:
    try:
        rows = tuple(int(value) for value in raw.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("rows must be comma-separated integers") from error
    if not rows or any(value not in range(2, 9) for value in rows):
        raise argparse.ArgumentTypeError("rows must be in 2..8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure exact scalar-Q-A graph scheduling candidates."
    )
    parser.add_argument("--native-lib", type=Path, required=True)
    parser.add_argument("--rows", type=parse_rows, default=(2, 4, 8))
    parser.add_argument("--weight-sets", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if min(args.weight_sets, args.warmup, args.iterations, args.repeats) < 1:
        parser.error("weight-sets, warmup, iterations, and repeats must be positive")
    if not args.eps > 0:
        parser.error("eps must be positive")

    torch.manual_seed(args.seed)
    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    lib = ctypes.CDLL(str(args.native_lib.resolve()))
    lib.glmrt_last_error.argtypes = (ctypes.c_char_p, ctypes.c_size_t)
    lib.glmrt_last_error.restype = ctypes.c_int
    rmsnorm = lib.glmrt_cuda_rmsnorm_bf16_async
    rmsnorm.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_void_p,
    )
    rmsnorm.restype = ctypes.c_int
    linear = lib.glmrt_cuda_linear_bf16_cublas_async
    linear.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    linear.restype = ctypes.c_int
    candidate = lib.glmrt_cuda_mla_scalar_qa_batched_norm_candidate_async
    candidate.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_void_p,
    )
    candidate.restype = ctypes.c_int

    input_norm_weight = torch.randn(HIDDEN, dtype=torch.bfloat16, device=device)
    q_a_norm_weight = torch.randn(
        Q_LORA_RANK, dtype=torch.bfloat16, device=device
    )
    q_a_weights = torch.randn(
        (args.weight_sets, Q_LORA_RANK, HIDDEN),
        dtype=torch.bfloat16,
        device=device,
    ) * 0.02
    results = []

    for rows in args.rows:
        hidden = torch.randn((rows, HIDDEN), dtype=torch.bfloat16, device=device)
        baseline_normalized = torch.empty_like(hidden)
        baseline_projected = torch.empty(
            (rows, Q_LORA_RANK), dtype=torch.bfloat16, device=device
        )
        baseline_output = torch.empty_like(baseline_projected)
        candidate_normalized = torch.empty_like(hidden)
        candidate_projected = torch.empty_like(baseline_projected)
        candidate_output = torch.empty_like(baseline_projected)

        def launch_rmsnorm(
            x: torch.Tensor, weight: torch.Tensor, output: torch.Tensor, size: int
        ) -> None:
            check_status(
                lib,
                rmsnorm(
                    ctypes.c_void_p(x.data_ptr()),
                    ctypes.c_void_p(weight.data_ptr()),
                    ctypes.c_void_p(output.data_ptr()),
                    1,
                    size,
                    args.eps,
                    stream_pointer(),
                ),
                "baseline RMSNorm",
            )

        def launch_baseline(weight: torch.Tensor) -> None:
            for row in range(rows):
                launch_rmsnorm(
                    hidden[row], input_norm_weight, baseline_normalized[row], HIDDEN
                )
                check_status(
                    lib,
                    linear(
                        ctypes.c_void_p(baseline_normalized[row].data_ptr()),
                        ctypes.c_void_p(weight.data_ptr()),
                        None,
                        ctypes.c_void_p(baseline_projected[row].data_ptr()),
                        1,
                        HIDDEN,
                        Q_LORA_RANK,
                        stream_pointer(),
                    ),
                    "baseline scalar Q-A",
                )
                launch_rmsnorm(
                    baseline_projected[row],
                    q_a_norm_weight,
                    baseline_output[row],
                    Q_LORA_RANK,
                )

        def launch_candidate(weight: torch.Tensor) -> None:
            check_status(
                lib,
                candidate(
                    ctypes.c_void_p(hidden.data_ptr()),
                    ctypes.c_void_p(input_norm_weight.data_ptr()),
                    ctypes.c_void_p(candidate_normalized.data_ptr()),
                    ctypes.c_void_p(weight.data_ptr()),
                    ctypes.c_void_p(candidate_projected.data_ptr()),
                    ctypes.c_void_p(q_a_norm_weight.data_ptr()),
                    ctypes.c_void_p(candidate_output.data_ptr()),
                    rows,
                    HIDDEN,
                    Q_LORA_RANK,
                    args.eps,
                    stream_pointer(),
                ),
                "batched-norm scalar Q-A candidate",
            )

        launch_baseline(q_a_weights[0])
        launch_candidate(q_a_weights[0])
        torch.cuda.synchronize()
        exact_projected = torch.equal(baseline_projected, candidate_projected)
        exact_output = torch.equal(baseline_output, candidate_output)
        max_abs = float(
            (baseline_output.float() - candidate_output.float()).abs().max()
        )
        baseline_graphs = [
            capture(lambda weight=weight: launch_baseline(weight))
            for weight in q_a_weights
        ]
        candidate_graphs = [
            capture(lambda weight=weight: launch_candidate(weight))
            for weight in q_a_weights
        ]
        baseline_before = measure(
            baseline_graphs, args.warmup, args.iterations, args.repeats
        )
        candidate_samples = measure(
            candidate_graphs, args.warmup, args.iterations, args.repeats
        )
        baseline_after = measure(
            baseline_graphs, args.warmup, args.iterations, args.repeats
        )
        baseline_samples = baseline_before + baseline_after
        baseline_median = statistics.median(baseline_samples)
        candidate_median = statistics.median(candidate_samples)
        result = {
            "baseline_median_ms": baseline_median,
            "baseline_nodes": rows * 3,
            "baseline_samples_ms": baseline_samples,
            "benchmark": "mla_scalar_qa_batched_norm_candidate",
            "candidate_median_ms": candidate_median,
            "candidate_nodes": rows + 2,
            "candidate_samples_ms": candidate_samples,
            "exact_output": exact_output,
            "exact_projected": exact_projected,
            "gpu": properties.name,
            "max_abs_error": max_abs,
            "rows": rows,
            "speedup": baseline_median / candidate_median,
            "weight_sets": args.weight_sets,
            "weight_working_set_bytes": q_a_weights.numel()
            * q_a_weights.element_size(),
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    print(
        json.dumps(
            {
                "benchmark": "mla_scalar_qa_batched_norm_candidate_summary",
                "exact": all(
                    result["exact_output"] and result["exact_projected"]
                    for result in results
                ),
                "max_speedup": max(result["speedup"] for result in results),
                "min_speedup": min(result["speedup"] for result in results),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
