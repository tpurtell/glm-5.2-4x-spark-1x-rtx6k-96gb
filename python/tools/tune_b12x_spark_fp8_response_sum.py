#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import time
from pathlib import Path

import torch


HIDDEN = 6144
TOP_K = 8
FP8_ROW_BYTES = HIDDEN + 4


class DeviceBuffer(ctypes.Structure):
    _fields_ = (
        ("ptr", ctypes.c_void_p),
        ("bytes", ctypes.c_size_t),
        ("device_id", ctypes.c_int),
        ("flags", ctypes.c_uint64),
    )


def device_buffer(tensor: torch.Tensor) -> DeviceBuffer:
    return DeviceBuffer(
        tensor.data_ptr(),
        tensor.numel() * tensor.element_size(),
        tensor.device.index or 0,
        0,
    )


def check_status(library: ctypes.CDLL, status: int, action: str) -> None:
    if status == 0:
        return
    error = ctypes.create_string_buffer(512)
    library.glmrt_last_error_message(error, len(error))
    raise RuntimeError(
        f"{action} failed with status {status}: {error.value.decode()}"
    )


def gpu_samples(operation, warmup: int, iterations: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return samples


def synchronized_wall_samples(
    operation, warmup: int, iterations: int, repeats: int
) -> list[float]:
    for _ in range(warmup):
        operation()
        torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        for _ in range(iterations):
            operation()
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        samples.append(elapsed_ms / iterations)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the current B12X top-k BF16 sum plus FP8 codec with a "
            "single exact-rounding fused response kernel."
        )
    )
    parser.add_argument("--native-lib", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--sync-iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()
    if min(
        *args.rows,
        args.warmup,
        args.iterations,
        args.sync_iterations,
        args.repeats,
    ) < 1:
        parser.error("rows and iteration counts must be positive")

    torch.manual_seed(args.seed)
    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    library = ctypes.CDLL(str(args.native_lib.resolve()))
    sum_bf16 = library.glmrt_cuda_b12x_spark_sum_topk8_bf16_async
    sum_bf16.argtypes = (
        DeviceBuffer,
        DeviceBuffer,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    sum_bf16.restype = ctypes.c_int
    pack_fp8 = library.glmrt_cuda_bf16_rows_to_fp8_e4m3_row_scaled_async
    pack_fp8.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    pack_fp8.restype = ctypes.c_int
    fused = library.glmrt_cuda_b12x_spark_sum_topk8_bf16_to_fp8_async
    fused.argtypes = (
        DeviceBuffer,
        DeviceBuffer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    fused.restype = ctypes.c_int

    results = []
    for rows in args.rows:
        routed = torch.randn(
            (rows * TOP_K, HIDDEN), dtype=torch.bfloat16, device=device
        )
        summed = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
        current_output = torch.empty(
            (rows, FP8_ROW_BYTES), dtype=torch.uint8, device=device
        )
        fused_output = torch.empty_like(current_output)
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

        def current_operation() -> None:
            check_status(
                library,
                sum_bf16(
                    device_buffer(routed), device_buffer(summed), rows, stream
                ),
                "summing B12X top-k rows as BF16",
            )
            check_status(
                library,
                pack_fp8(
                    summed.data_ptr(),
                    current_output.data_ptr(),
                    rows,
                    HIDDEN,
                    FP8_ROW_BYTES,
                    stream,
                ),
                "packing the summed BF16 rows as FP8",
            )

        def fused_operation() -> None:
            check_status(
                library,
                fused(
                    device_buffer(routed),
                    device_buffer(fused_output),
                    rows,
                    FP8_ROW_BYTES,
                    stream,
                ),
                "summing and packing B12X top-k rows as FP8",
            )

        current_operation()
        fused_operation()
        torch.cuda.synchronize()
        bitwise_equal = torch.equal(current_output, fused_output)
        differing_values = torch.count_nonzero(current_output != fused_output).item()
        current_gpu = gpu_samples(
            current_operation, args.warmup, args.iterations, args.repeats
        )
        fused_gpu = gpu_samples(
            fused_operation, args.warmup, args.iterations, args.repeats
        )
        current_wall = synchronized_wall_samples(
            current_operation, args.warmup, args.sync_iterations, args.repeats
        )
        fused_wall = synchronized_wall_samples(
            fused_operation, args.warmup, args.sync_iterations, args.repeats
        )
        current_gpu_median = statistics.median(current_gpu)
        fused_gpu_median = statistics.median(fused_gpu)
        current_wall_median = statistics.median(current_wall)
        fused_wall_median = statistics.median(fused_wall)
        result = {
            "rows": rows,
            "bitwise_equal": bitwise_equal,
            "differing_values": differing_values,
            "current_gpu_ms": current_gpu_median,
            "fused_gpu_ms": fused_gpu_median,
            "gpu_change_pct": 100.0
            * (fused_gpu_median / current_gpu_median - 1.0),
            "current_sync_wall_ms": current_wall_median,
            "fused_sync_wall_ms": fused_wall_median,
            "sync_wall_change_pct": 100.0
            * (fused_wall_median / current_wall_median - 1.0),
            "current_gpu_samples_ms": current_gpu,
            "fused_gpu_samples_ms": fused_gpu,
            "current_sync_wall_samples_ms": current_wall,
            "fused_sync_wall_samples_ms": fused_wall,
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    if not all(result["bitwise_equal"] for result in results):
        raise SystemExit("fused response output differs from the current closure")


if __name__ == "__main__":
    main()
