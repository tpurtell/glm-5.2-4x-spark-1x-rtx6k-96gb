#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import statistics
from pathlib import Path

import torch


KV_LORA_RANK = 512
ROPE_DIM = 64
KV_WIDTH = KV_LORA_RANK + ROPE_DIM


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
    for iteration in range(warmup * len(graphs)):
        graphs[iteration % len(graphs)].replay()
    torch.cuda.synchronize()
    samples = []
    launches = iterations * len(graphs)
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            for graph in graphs:
                graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / launches)
    return samples


def parse_positions(raw: str) -> tuple[int, ...]:
    try:
        positions = tuple(int(value) for value in raw.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "positions must be comma-separated integers"
        ) from error
    if not positions or min(positions) < 0 or max(positions) > 0xFFFFFFFF:
        raise argparse.ArgumentTypeError("positions must fit uint32")
    return positions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure exact cross-layer reuse of decode MLA RoPE factors."
    )
    parser.add_argument("--native-lib", type=Path, required=True)
    parser.add_argument("--positions", type=parse_positions, default=(1, 1024, 131071))
    parser.add_argument("--layers", type=int, default=78)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--theta", type=float, default=1_000_000.0)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if min(args.layers, args.warmup, args.iterations, args.repeats) < 1:
        parser.error("layers, warmup, iterations, and repeats must be positive")
    if not args.eps > 0 or not args.theta > 0:
        parser.error("eps and theta must be positive")

    torch.manual_seed(args.seed)
    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    lib = ctypes.CDLL(str(args.native_lib.resolve()))
    lib.glmrt_last_error.argtypes = (ctypes.c_char_p, ctypes.c_size_t)
    lib.glmrt_last_error.restype = ctypes.c_int
    baseline = lib.glmrt_cuda_mla_kv_prepare_bf16_async
    baseline.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_void_p,
    )
    baseline.restype = ctypes.c_int
    factor_launch = lib.glmrt_cuda_mla_rope_factors_f32_candidate_async
    factor_launch.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_void_p,
    )
    factor_launch.restype = ctypes.c_int
    candidate = lib.glmrt_cuda_mla_kv_prepare_bf16_precomputed_rope_candidate_async
    candidate.argtypes = (
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

    projected = torch.randn(
        (args.layers, KV_WIDTH), dtype=torch.bfloat16, device=device
    )
    norm_weights = torch.randn(
        (args.layers, KV_LORA_RANK), dtype=torch.bfloat16, device=device
    )
    results = []

    for position_value in args.positions:
        positions = torch.tensor([position_value], dtype=torch.uint32, device=device)
        factors = torch.empty(ROPE_DIM, dtype=torch.float32, device=device)
        baseline_output = torch.empty_like(projected)
        candidate_output = torch.empty_like(projected)

        def launch_factors() -> None:
            check_status(
                lib,
                factor_launch(
                    ctypes.c_void_p(positions.data_ptr()),
                    ctypes.c_void_p(factors.data_ptr()),
                    1,
                    args.theta,
                    stream_pointer(),
                ),
                "RoPE factor candidate",
            )

        def launch_baseline(layer: int) -> None:
            check_status(
                lib,
                baseline(
                    ctypes.c_void_p(projected[layer].data_ptr()),
                    ctypes.c_void_p(positions.data_ptr()),
                    ctypes.c_void_p(norm_weights[layer].data_ptr()),
                    ctypes.c_void_p(baseline_output[layer].data_ptr()),
                    1,
                    KV_WIDTH * 2,
                    KV_WIDTH * 2,
                    args.eps,
                    args.theta,
                    stream_pointer(),
                ),
                "baseline MLA KV prepare",
            )

        def launch_candidate(layer: int) -> None:
            check_status(
                lib,
                candidate(
                    ctypes.c_void_p(projected[layer].data_ptr()),
                    ctypes.c_void_p(factors.data_ptr()),
                    ctypes.c_void_p(norm_weights[layer].data_ptr()),
                    ctypes.c_void_p(candidate_output[layer].data_ptr()),
                    1,
                    KV_WIDTH * 2,
                    KV_WIDTH * 2,
                    args.eps,
                    stream_pointer(),
                ),
                "precomputed-RoPE MLA KV prepare",
            )

        launch_factors()
        for layer in range(args.layers):
            launch_baseline(layer)
            launch_candidate(layer)
        torch.cuda.synchronize()
        exact = torch.equal(baseline_output, candidate_output)
        max_abs = float(
            (baseline_output.float() - candidate_output.float()).abs().max()
        )
        baseline_graphs = [
            capture(lambda layer=layer: launch_baseline(layer))
            for layer in range(args.layers)
        ]
        candidate_graphs = [
            capture(lambda layer=layer: launch_candidate(layer))
            for layer in range(args.layers)
        ]
        factor_graph = capture(launch_factors)
        baseline_before = measure(
            baseline_graphs, args.warmup, args.iterations, args.repeats
        )
        candidate_samples = measure(
            candidate_graphs, args.warmup, args.iterations, args.repeats
        )
        baseline_after = measure(
            baseline_graphs, args.warmup, args.iterations, args.repeats
        )
        factor_samples = measure(
            [factor_graph], args.warmup * args.layers,
            args.iterations * args.layers, args.repeats
        )
        baseline_samples = baseline_before + baseline_after
        baseline_median = statistics.median(baseline_samples)
        candidate_median = statistics.median(candidate_samples)
        factor_median = statistics.median(factor_samples)
        amortized_candidate = candidate_median + factor_median / args.layers
        result = {
            "amortized_candidate_ms_per_layer": amortized_candidate,
            "baseline_median_ms_per_layer": baseline_median,
            "baseline_samples_ms_per_layer": baseline_samples,
            "benchmark": "mla_kv_rope_factor_reuse_candidate",
            "candidate_median_ms_per_layer": candidate_median,
            "candidate_samples_ms_per_layer": candidate_samples,
            "exact": exact,
            "factor_median_ms_per_token": factor_median,
            "factor_samples_ms_per_token": factor_samples,
            "gpu": properties.name,
            "layers": args.layers,
            "max_abs_error": max_abs,
            "position": position_value,
            "speedup_amortized": baseline_median / amortized_candidate,
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    print(
        json.dumps(
            {
                "benchmark": "mla_kv_rope_factor_reuse_candidate_summary",
                "exact": all(result["exact"] for result in results),
                "max_speedup_amortized": max(
                    result["speedup_amortized"] for result in results
                ),
                "min_speedup_amortized": min(
                    result["speedup_amortized"] for result in results
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
