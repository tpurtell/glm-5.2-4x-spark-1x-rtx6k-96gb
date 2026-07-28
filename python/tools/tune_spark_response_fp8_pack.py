#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import statistics
from pathlib import Path

from tune_b12x_spark_top1_native import (
    Allocation,
    CUDA_MEMCPY_HOST_TO_DEVICE,
    CudaRuntime,
    NativeLibrary,
    copy_from_device,
    measure,
)


HIDDEN = 6144
FP8_ROW_BYTES = HIDDEN + ctypes.sizeof(ctypes.c_float)


def configure_pack(lib: ctypes.CDLL, symbol: str):
    function = getattr(lib, symbol)
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


def configure_gather(lib: ctypes.CDLL, symbol: str):
    function = getattr(lib, symbol)
    function.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    function.restype = ctypes.c_int
    return function


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure exact Spark F32-accumulator to row-scaled FP8 packing."
    )
    parser.add_argument("--native-lib", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if min(args.rows, args.warmup, args.iterations, args.repeats) < 1:
        parser.error("rows, warmup, iterations, and repeats must be positive")
    if args.rows > 256:
        parser.error("rows must be at most 256")

    runtime = CudaRuntime()
    native = NativeLibrary(args.native_lib)
    current = configure_pack(
        native.lib, "glmrt_cuda_gather_rows_f32_to_fp8_e4m3_row_scaled_async"
    )
    candidate = configure_pack(
        native.lib,
        "glmrt_cuda_gather_rows_f32_to_fp8_e4m3_row_scaled_register_candidate_async",
    )
    gather_f32 = configure_gather(native.lib, "glmrt_cuda_gather_rows_f32_async")
    f32_to_bf16 = native.lib.glmrt_cuda_f32_to_bf16_async
    f32_to_bf16.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    f32_to_bf16.restype = ctypes.c_int
    bf16_candidate = configure_gather(
        native.lib, "glmrt_cuda_gather_rows_f32_to_bf16_candidate_async"
    )
    stream = ctypes.c_void_p()
    runtime.check(
        runtime.lib.cudaStreamCreateWithFlags(ctypes.byref(stream), 1),
        "create benchmark stream",
    )

    values = args.rows * HIDDEN
    source_bytes = values * ctypes.sizeof(ctypes.c_float)
    output_bytes = args.rows * FP8_ROW_BYTES
    source = Allocation(runtime, source_bytes)
    indices = Allocation(runtime, args.rows * ctypes.sizeof(ctypes.c_uint32))
    current_output = Allocation(runtime, output_bytes)
    candidate_output = Allocation(runtime, output_bytes)
    bf16_scratch = Allocation(runtime, source_bytes)
    current_bf16_output = Allocation(runtime, values * ctypes.sizeof(ctypes.c_uint16))
    candidate_bf16_output = Allocation(runtime, values * ctypes.sizeof(ctypes.c_uint16))
    host_source = (ctypes.c_float * values)()
    for index in range(values):
        host_source[index] = ((index * 17 + index // HIDDEN * 29) % 1021 - 510) / 127.0
    host_indices = (ctypes.c_uint32 * args.rows)(
        *(args.rows - row - 1 for row in range(args.rows))
    )
    runtime.check(
        runtime.lib.cudaMemcpy(
            source.ptr,
            ctypes.cast(host_source, ctypes.c_void_p),
            source_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        ),
        "copy accumulator rows",
    )
    runtime.check(
        runtime.lib.cudaMemcpy(
            indices.ptr,
            ctypes.cast(host_indices, ctypes.c_void_p),
            ctypes.sizeof(host_indices),
            CUDA_MEMCPY_HOST_TO_DEVICE,
        ),
        "copy completion indices",
    )

    def launch(function, output: Allocation) -> None:
        native.check(
            function(
                source.ptr,
                indices.ptr,
                output.ptr,
                args.rows,
                HIDDEN,
                FP8_ROW_BYTES,
                stream,
            ),
            "FP8 response pack",
        )

    def launch_current_bf16() -> None:
        native.check(
            gather_f32(
                source.ptr,
                indices.ptr,
                bf16_scratch.ptr,
                args.rows,
                HIDDEN,
                stream,
            ),
            "BF16 response gather",
        )
        native.check(
            f32_to_bf16(
                bf16_scratch.ptr,
                current_bf16_output.ptr,
                values,
                stream,
            ),
            "BF16 response conversion",
        )

    def launch_candidate_bf16() -> None:
        native.check(
            bf16_candidate(
                source.ptr,
                indices.ptr,
                candidate_bf16_output.ptr,
                args.rows,
                HIDDEN,
                stream,
            ),
            "fused BF16 response pack",
        )

    launch(current, current_output)
    launch(candidate, candidate_output)
    launch_current_bf16()
    launch_candidate_bf16()
    runtime.check(runtime.lib.cudaStreamSynchronize(stream), "validate response pack")
    current_bytes = copy_from_device(runtime, current_output, output_bytes)
    candidate_bytes = copy_from_device(runtime, candidate_output, output_bytes)
    exact = current_bytes == candidate_bytes
    bf16_bytes = values * ctypes.sizeof(ctypes.c_uint16)
    bf16_exact = copy_from_device(
        runtime, current_bf16_output, bf16_bytes
    ) == copy_from_device(runtime, candidate_bf16_output, bf16_bytes)

    current_graph = native.capture(stream, lambda: launch(current, current_output))
    candidate_graph = native.capture(stream, lambda: launch(candidate, candidate_output))
    current_bf16_graph = native.capture(stream, launch_current_bf16)
    candidate_bf16_graph = native.capture(stream, launch_candidate_bf16)
    current_before = measure(
        runtime,
        native,
        [current_graph],
        stream,
        args.warmup,
        args.iterations,
        args.repeats,
    )
    candidate_samples = measure(
        runtime,
        native,
        [candidate_graph],
        stream,
        args.warmup,
        args.iterations,
        args.repeats,
    )
    current_after = measure(
        runtime,
        native,
        [current_graph],
        stream,
        args.warmup,
        args.iterations,
        args.repeats,
    )
    current_samples = current_before + current_after
    current_median = statistics.median(current_samples)
    candidate_median = statistics.median(candidate_samples)
    current_bf16_before = measure(
        runtime,
        native,
        [current_bf16_graph],
        stream,
        args.warmup,
        args.iterations,
        args.repeats,
    )
    candidate_bf16_samples = measure(
        runtime,
        native,
        [candidate_bf16_graph],
        stream,
        args.warmup,
        args.iterations,
        args.repeats,
    )
    current_bf16_after = measure(
        runtime,
        native,
        [current_bf16_graph],
        stream,
        args.warmup,
        args.iterations,
        args.repeats,
    )
    current_bf16_samples = current_bf16_before + current_bf16_after
    current_bf16_median = statistics.median(current_bf16_samples)
    candidate_bf16_median = statistics.median(candidate_bf16_samples)
    print(
        json.dumps(
            {
                "benchmark": "spark_response_fp8_register_pack",
                "bf16_candidate_median_ms": candidate_bf16_median,
                "bf16_candidate_samples_ms": candidate_bf16_samples,
                "bf16_current_median_ms": current_bf16_median,
                "bf16_current_samples_ms": current_bf16_samples,
                "bf16_exact": bf16_exact,
                "bf16_speedup": current_bf16_median / candidate_bf16_median,
                "candidate_median_ms": candidate_median,
                "candidate_samples_ms": candidate_samples,
                "current_median_ms": current_median,
                "current_samples_ms": current_samples,
                "exact": exact,
                "fp8_payload_bytes": output_bytes,
                "rows": args.rows,
                "speedup": current_median / candidate_median,
                "wire_reduction_vs_bf16": 1.0
                - output_bytes / (args.rows * HIDDEN * 2),
            },
            sort_keys=True,
        )
    )
    native.lib.glmrt_cuda_graph_exec_destroy(current_graph)
    native.lib.glmrt_cuda_graph_exec_destroy(candidate_graph)
    native.lib.glmrt_cuda_graph_exec_destroy(current_bf16_graph)
    native.lib.glmrt_cuda_graph_exec_destroy(candidate_bf16_graph)
    runtime.lib.cudaStreamDestroy(stream)


if __name__ == "__main__":
    main()
