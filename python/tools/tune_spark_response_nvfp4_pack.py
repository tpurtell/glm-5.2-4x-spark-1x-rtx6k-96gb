#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import struct
from pathlib import Path

from tune_b12x_spark_top1_native import (
    Allocation,
    CUDA_MEMCPY_HOST_TO_DEVICE,
    CudaRuntime,
    NativeLibrary,
    copy_from_device,
    measure,
)


HIDDEN = 6_144
ROW_BYTES = HIDDEN // 2 + HIDDEN // 16


def parse_ints(raw: str, label: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in raw.split(",") if item)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{label} must be comma-separated integers"
        ) from error
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError(f"{label} values must be positive")
    return values


def configure_current(lib: ctypes.CDLL):
    function = lib.glmrt_cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3_async
    function.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    function.restype = ctypes.c_int
    return function


def configure_policy(lib: ctypes.CDLL):
    function = (
        lib.glmrt_cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3_policy_candidate_async
    )
    function.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    function.restype = ctypes.c_int
    return function


def configure_candidate(lib: ctypes.CDLL):
    function = (
        lib.glmrt_cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3_grouped_candidate_async
    )
    function.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    function.restype = ctypes.c_int
    return function


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tune the benchmark-only grouped Spark F32-to-NVFP4 response pack."
        )
    )
    parser.add_argument("--native-lib", type=Path, required=True)
    parser.add_argument("--rows", default="1,4,16,64,256")
    parser.add_argument("--blocks-per-row", default="1,2,4,8,16,24,32,48")
    parser.add_argument("--source-sets", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows_values = parse_ints(args.rows, "rows")
    block_values = parse_ints(args.blocks_per_row, "blocks-per-row")
    if max(rows_values) > 256:
        parser.error("rows must not exceed 256")
    if max(block_values) > HIDDEN // (16 * 8):
        parser.error("blocks-per-row must not exceed 48")
    if min(args.warmup, args.iterations, args.repeats) < 1:
        parser.error("warmup, iterations, and repeats must be positive")

    runtime = CudaRuntime()
    native = NativeLibrary(args.native_lib)
    current = configure_current(native.lib)
    policy = configure_policy(native.lib)
    candidate = configure_candidate(native.lib)
    stream = ctypes.c_void_p()
    runtime.check(
        runtime.lib.cudaStreamCreateWithFlags(ctypes.byref(stream), 1),
        "create benchmark stream",
    )
    max_rows = max(rows_values)
    source_set_bytes = max_rows * HIDDEN * ctypes.sizeof(ctypes.c_float)
    source = Allocation(runtime, args.source_sets * source_set_bytes)
    indices = Allocation(runtime, max_rows * ctypes.sizeof(ctypes.c_uint32))
    current_output = Allocation(runtime, max_rows * ROW_BYTES)
    candidate_output = Allocation(runtime, max_rows * ROW_BYTES)
    graph_execs: list[ctypes.c_void_p] = []
    try:
        for source_set in range(args.source_sets):
            value = 0.5 + source_set * 0.25
            host = ctypes.create_string_buffer(
                struct.pack("<f", value) * (max_rows * HIDDEN)
            )
            runtime.check(
                runtime.lib.cudaMemcpy(
                    source.offset(source_set * source_set_bytes),
                    ctypes.cast(host, ctypes.c_void_p),
                    source_set_bytes,
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                ),
                f"copy source set {source_set}",
            )

        def launch_current(source_set: int, rows: int) -> None:
            native.check(
                current(
                    source.offset(source_set * source_set_bytes),
                    indices.ptr,
                    current_output.ptr,
                    rows,
                    HIDDEN,
                    ROW_BYTES,
                    stream,
                ),
                "launch current NVFP4 response pack",
            )

        def launch_candidate(source_set: int, rows: int, blocks: int) -> None:
            native.check(
                candidate(
                    source.offset(source_set * source_set_bytes),
                    indices.ptr,
                    candidate_output.ptr,
                    rows,
                    HIDDEN,
                    ROW_BYTES,
                    blocks,
                    stream,
                ),
                "launch grouped NVFP4 response pack",
            )

        def launch_policy(source_set: int, rows: int) -> None:
            native.check(
                policy(
                    source.offset(source_set * source_set_bytes),
                    indices.ptr,
                    candidate_output.ptr,
                    rows,
                    HIDDEN,
                    ROW_BYTES,
                    stream,
                ),
                "launch NVFP4 response policy candidate",
            )

        results = []
        for rows in rows_values:
            host_indices = (ctypes.c_uint32 * rows)(
                *(rows - row - 1 for row in range(rows))
            )
            runtime.check(
                runtime.lib.cudaMemcpy(
                    indices.ptr,
                    ctypes.cast(host_indices, ctypes.c_void_p),
                    ctypes.sizeof(host_indices),
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                ),
                f"copy M{rows} row indices",
            )
            output_bytes = rows * ROW_BYTES
            launch_current(0, rows)
            runtime.check(
                runtime.lib.cudaStreamSynchronize(stream),
                f"finish M{rows} current validation",
            )
            expected = copy_from_device(runtime, current_output, output_bytes)
            current_graphs = [
                native.capture(
                    stream,
                    lambda source_set=source_set, rows=rows: launch_current(
                        source_set, rows
                    ),
                )
                for source_set in range(args.source_sets)
            ]
            graph_execs.extend(current_graphs)
            current_samples = measure(
                runtime,
                native,
                current_graphs,
                stream,
                args.warmup,
                args.iterations,
                args.repeats,
            )
            current_median = statistics.median(current_samples)
            launch_policy(0, rows)
            runtime.check(
                runtime.lib.cudaStreamSynchronize(stream),
                f"finish M{rows} policy validation",
            )
            policy_exact = (
                copy_from_device(runtime, candidate_output, output_bytes) == expected
            )
            policy_graphs = [
                native.capture(
                    stream,
                    lambda source_set=source_set, rows=rows: launch_policy(
                        source_set, rows
                    ),
                )
                for source_set in range(args.source_sets)
            ]
            graph_execs.extend(policy_graphs)
            policy_samples = measure(
                runtime,
                native,
                policy_graphs,
                stream,
                args.warmup,
                args.iterations,
                args.repeats,
            )
            policy_median = statistics.median(policy_samples)
            candidates = []
            for blocks in block_values:
                launch_candidate(0, rows, blocks)
                runtime.check(
                    runtime.lib.cudaStreamSynchronize(stream),
                    f"finish M{rows} blocks={blocks} validation",
                )
                actual = copy_from_device(runtime, candidate_output, output_bytes)
                candidate_graphs = [
                    native.capture(
                        stream,
                        lambda source_set=source_set, rows=rows, blocks=blocks: (
                            launch_candidate(source_set, rows, blocks)
                        ),
                    )
                    for source_set in range(args.source_sets)
                ]
                graph_execs.extend(candidate_graphs)
                samples = measure(
                    runtime,
                    native,
                    candidate_graphs,
                    stream,
                    args.warmup,
                    args.iterations,
                    args.repeats,
                )
                candidates.append(
                    {
                        "bitwise_equal": actual == expected,
                        "blocks_per_row": blocks,
                        "median_ms": statistics.median(samples),
                        "samples_ms": samples,
                    }
                )
            best = min(candidates, key=lambda item: float(item["median_ms"]))
            result = {
                "rows": rows,
                "current_median_ms": current_median,
                "current_samples_ms": current_samples,
                "policy_bitwise_equal": policy_exact,
                "policy_median_ms": policy_median,
                "policy_samples_ms": policy_samples,
                "policy_speedup": current_median / policy_median,
                "candidates": candidates,
                "best_blocks_per_row": best["blocks_per_row"],
                "best_median_ms": best["median_ms"],
                "best_speedup": current_median / float(best["median_ms"]),
                "all_bitwise_equal": all(
                    bool(item["bitwise_equal"]) for item in candidates
                ),
            }
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

        report = {
            "benchmark": "spark_response_nvfp4_grouped_pack",
            "device_working_set_bytes": args.source_sets * source_set_bytes,
            "hidden": HIDDEN,
            "results": results,
            "serving_path_changed": False,
            "source_sets": args.source_sets,
        }
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.write_text(payload, encoding="ascii")
        print(payload, end="")
    finally:
        for graph_exec in graph_execs:
            native.lib.glmrt_cuda_graph_exec_destroy(graph_exec)
        runtime.lib.cudaStreamDestroy(stream)


if __name__ == "__main__":
    main()
