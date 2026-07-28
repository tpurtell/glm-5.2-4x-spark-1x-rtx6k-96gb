#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import statistics
from pathlib import Path

import torch


HIDDEN = 6144
FP8_ROW_BYTES = HIDDEN + ctypes.sizeof(ctypes.c_float)


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


def configure(lib: ctypes.CDLL, symbol: str, argtypes) -> ctypes._CFuncPtr:
    function = getattr(lib, symbol)
    function.argtypes = argtypes
    function.restype = ctypes.c_int
    return function


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure fused coordinator FP8 decode combine and residual add."
    )
    parser.add_argument("--native-lib", type=Path, required=True)
    parser.add_argument("--partial-rows", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--sets", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if min(args.sets, args.warmup, args.iterations, args.repeats) < 1:
        parser.error("sets, warmup, iterations, and repeats must be positive")

    torch.manual_seed(args.seed)
    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    lib = ctypes.CDLL(str(args.native_lib.resolve()))
    lib.glmrt_last_error.argtypes = (ctypes.c_char_p, ctypes.c_size_t)
    lib.glmrt_last_error.restype = ctypes.c_int
    pack = configure(
        lib,
        "glmrt_cuda_gather_rows_f32_to_fp8_e4m3_row_scaled_async",
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ),
    )
    zero = configure(
        lib,
        "glmrt_cuda_zero_f32_async",
        (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p),
    )
    scatter = configure(
        lib,
        "glmrt_cuda_scatter_add_rows_fp8_e4m3_row_scaled_to_f32_async",
        (
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ),
    )
    residual_add = configure(
        lib,
        "glmrt_cuda_residual_add_shared_f32_delta_bf16_async",
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ),
    )
    candidate = configure(
        lib,
        "glmrt_cuda_fp8_decode_combine_residual_async",
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ),
    )

    identity_indices = torch.tensor(
        list(range(args.partial_rows)), dtype=torch.uint32, device=device
    )
    destination_indices = torch.zeros(
        args.partial_rows, dtype=torch.uint32, device=device
    )
    graph_pairs = []
    exact = True
    max_abs = 0.0

    for set_index in range(args.sets):
        partial_f32 = torch.randn(
            (args.partial_rows, HIDDEN), dtype=torch.float32, device=device
        ) * (0.03 + set_index * 0.002)
        packed = torch.empty(
            args.partial_rows * FP8_ROW_BYTES, dtype=torch.uint8, device=device
        )
        residual = torch.randn(HIDDEN, dtype=torch.bfloat16, device=device)
        shared = torch.randn(HIDDEN, dtype=torch.bfloat16, device=device) * 0.03
        accumulator = torch.empty(HIDDEN, dtype=torch.float32, device=device)
        baseline_output = torch.empty(HIDDEN, dtype=torch.bfloat16, device=device)
        candidate_output = torch.empty_like(baseline_output)

        check_status(
            lib,
            pack(
                ctypes.c_void_p(partial_f32.data_ptr()),
                ctypes.c_void_p(identity_indices.data_ptr()),
                ctypes.c_void_p(packed.data_ptr()),
                args.partial_rows,
                HIDDEN,
                FP8_ROW_BYTES,
                stream_pointer(),
            ),
            "pack FP8 host partials",
        )

        def launch_baseline() -> None:
            check_status(
                lib,
                zero(
                    ctypes.c_void_p(accumulator.data_ptr()),
                    HIDDEN,
                    stream_pointer(),
                ),
                "zero FP8 combine accumulator",
            )
            check_status(
                lib,
                scatter(
                    ctypes.c_void_p(packed.data_ptr()),
                    FP8_ROW_BYTES,
                    ctypes.c_void_p(destination_indices.data_ptr()),
                    ctypes.c_void_p(accumulator.data_ptr()),
                    args.partial_rows,
                    HIDDEN,
                    stream_pointer(),
                ),
                "scatter FP8 host partials",
            )
            check_status(
                lib,
                residual_add(
                    ctypes.c_void_p(residual.data_ptr()),
                    ctypes.c_void_p(shared.data_ptr()),
                    ctypes.c_void_p(accumulator.data_ptr()),
                    ctypes.c_void_p(baseline_output.data_ptr()),
                    HIDDEN,
                    stream_pointer(),
                ),
                "baseline residual add",
            )

        def launch_candidate() -> None:
            check_status(
                lib,
                candidate(
                    ctypes.c_void_p(residual.data_ptr()),
                    ctypes.c_void_p(shared.data_ptr()),
                    ctypes.c_void_p(packed.data_ptr()),
                    FP8_ROW_BYTES,
                    ctypes.c_void_p(candidate_output.data_ptr()),
                    args.partial_rows,
                    HIDDEN,
                    stream_pointer(),
                ),
                "fused FP8 decode combine",
            )

        launch_baseline()
        launch_candidate()
        torch.cuda.synchronize()
        exact = exact and torch.equal(baseline_output, candidate_output)
        max_abs = max(
            max_abs,
            float((baseline_output.float() - candidate_output.float()).abs().max()),
        )
        graph_pairs.append((capture(launch_baseline), capture(launch_candidate)))

    baseline_graphs = [pair[0] for pair in graph_pairs]
    candidate_graphs = [pair[1] for pair in graph_pairs]
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
    print(
        json.dumps(
            {
                "baseline_median_ms": baseline_median,
                "baseline_nodes": 3,
                "baseline_samples_ms": baseline_samples,
                "benchmark": "coordinator_fp8_decode_combine",
                "candidate_median_ms": candidate_median,
                "candidate_nodes": 1,
                "candidate_samples_ms": candidate_samples,
                "exact": exact,
                "gpu": properties.name,
                "max_abs_error": max_abs,
                "partial_rows": args.partial_rows,
                "sets": args.sets,
                "speedup": baseline_median / candidate_median,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
