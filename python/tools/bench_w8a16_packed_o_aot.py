#!/usr/bin/env python3
"""Validate and time the native single-resident packed W8A16 O kernels."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import torch

from tune_w8a16_projection import (
    CATALOG_PATH,
    DEFAULT_TENSORS,
    bench,
    check_status,
    load_bf16_weight,
)


SIZE_K = 16_384
SIZE_N = 6_144
GROUP_SIZE = 256


class DeviceBuffer(ctypes.Structure):
    _fields_ = (
        ("ptr", ctypes.c_void_p),
        ("bytes", ctypes.c_size_t),
        ("device_id", ctypes.c_int),
        ("flags", ctypes.c_uint64),
    )


class CoordinatorBuffers(ctypes.Structure):
    _fields_ = tuple(
        (name, DeviceBuffer)
        for name in (
            "input",
            "weight",
            "output",
            "scale",
            "global_scale",
            "packed_route_indices",
            "block_expert_ids",
            "packed_route_count",
            "topk_weights",
            "c_tmp",
            "locks",
        )
    )


def device_buffer(tensor: torch.Tensor) -> DeviceBuffer:
    return DeviceBuffer(
        tensor.data_ptr(), tensor.numel() * tensor.element_size(), 0, 0
    )


def device_buffer_view(
    tensor: torch.Tensor, byte_offset: int, byte_count: int
) -> DeviceBuffer:
    total_bytes = tensor.numel() * tensor.element_size()
    if byte_offset < 0 or byte_count < 0 or byte_offset + byte_count > total_bytes:
        raise ValueError("device buffer view is out of bounds")
    return DeviceBuffer(tensor.data_ptr() + byte_offset, byte_count, 0, 0)


def configure_native(path: Path) -> ctypes.CDLL:
    native = ctypes.CDLL(str(path.resolve()))
    native.glmrt_cuda_quantize_bf16_w8a16_group256_packed_async.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    native.glmrt_cuda_quantize_bf16_w8a16_group256_packed_async.restype = ctypes.c_int
    native.glmrt_cuda_linear_w8a16_group256_m1_warp_packed_async.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    native.glmrt_cuda_linear_w8a16_group256_m1_warp_packed_async.restype = ctypes.c_int
    native.glmrt_cuda_linear_w8a16_group256_m1_warp_packed_parity_batched_async.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    native.glmrt_cuda_linear_w8a16_group256_m1_warp_packed_parity_batched_async.restype = (
        ctypes.c_int
    )
    native.glmrt_cuda_w8a16_packed_o_aot_init.restype = ctypes.c_int
    native.glmrt_cuda_w8a16_packed_o_initialize_launch_buffers_async.argtypes = (
        ctypes.POINTER(CoordinatorBuffers),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    native.glmrt_cuda_w8a16_packed_o_initialize_launch_buffers_async.restype = (
        ctypes.c_int
    )
    native.glmrt_cuda_w8a16_packed_o_async.argtypes = (
        ctypes.POINTER(CoordinatorBuffers),
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    native.glmrt_cuda_w8a16_packed_o_async.restype = ctypes.c_int
    return native


def bucket_block_m(rows: int) -> int:
    if rows <= 16:
        return 16
    if rows <= 64:
        return 32
    return 48


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--native-library",
        type=Path,
        default=Path("native/build-cuda-rdma-coordinator-aot/libglmrt_native.so"),
    )
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument(
        "--kernel",
        choices=("auto", "parity", "aot"),
        default="auto",
        help="Force the recurrent-parity or grouped AOT path for crossover sweeps.",
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if not 1 <= args.rows <= 512:
        parser.error("--rows must be in 1..=512")
    if args.kernel == "parity" and not 2 <= args.rows <= 16:
        parser.error("--kernel parity requires --rows in 2..=16")

    with CATALOG_PATH.open() as handle:
        catalog = json.load(handle)
    weight = load_bf16_weight(catalog, DEFAULT_TENSORS[1])
    if tuple(weight.shape) != (SIZE_N, SIZE_K):
        raise RuntimeError(f"unexpected O weight shape {tuple(weight.shape)}")

    native = configure_native(args.native_library)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    packed_weight = torch.empty(SIZE_N * SIZE_K, device="cuda", dtype=torch.int8)
    scales = torch.empty(
        (SIZE_K // GROUP_SIZE, SIZE_N), device="cuda", dtype=torch.float32
    )
    check_status(
        native.glmrt_cuda_quantize_bf16_w8a16_group256_packed_async(
            weight.data_ptr(),
            packed_weight.data_ptr(),
            scales.data_ptr(),
            SIZE_K,
            SIZE_N,
            stream,
        ),
        "direct packed W8 quantization",
    )

    generator = torch.Generator(device="cuda")
    generator.manual_seed(17)
    inputs = torch.randn(
        (args.rows, SIZE_K),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    output = torch.empty(
        (args.rows, SIZE_N), device="cuda", dtype=torch.bfloat16
    )

    recurrent_output = None
    recurrent_launch = None
    selected_kernel = args.kernel
    if selected_kernel == "auto":
        selected_kernel = (
            "recurrent"
            if args.rows == 1
            else "parity"
            if args.rows <= 8
            else "aot"
        )
    if args.rows == 1 and selected_kernel != "aot":
        def launch(_index: int) -> None:
            check_status(
                native.glmrt_cuda_linear_w8a16_group256_m1_warp_packed_async(
                    inputs.data_ptr(),
                    packed_weight.data_ptr(),
                    scales.data_ptr(),
                    output.data_ptr(),
                    SIZE_K,
                    SIZE_N,
                    stream,
                ),
                "packed W8 M=1 launch",
            )
    elif selected_kernel == "parity":
        def launch(_index: int) -> None:
            check_status(
                native.glmrt_cuda_linear_w8a16_group256_m1_warp_packed_parity_batched_async(
                    inputs.data_ptr(),
                    packed_weight.data_ptr(),
                    scales.data_ptr(),
                    output.data_ptr(),
                    args.rows,
                    SIZE_K,
                    SIZE_N,
                    stream,
                ),
                "packed W8 parity-batched launch",
            )

        recurrent_output = torch.empty_like(output)

        def recurrent_launch(_index: int) -> None:
            for row in range(args.rows):
                check_status(
                    native.glmrt_cuda_linear_w8a16_group256_m1_warp_packed_async(
                        inputs[row].data_ptr(),
                        packed_weight.data_ptr(),
                        scales.data_ptr(),
                        recurrent_output[row].data_ptr(),
                        SIZE_K,
                        SIZE_N,
                        stream,
                    ),
                    "packed W8 recurrent reference launch",
                )
    else:
        check_status(native.glmrt_cuda_w8a16_packed_o_aot_init(), "packed O AOT init")
        max_chunk_rows = min(args.rows, 256)
        block_m = bucket_block_m(max_chunk_rows)
        route_slots = (
            (max_chunk_rows + block_m - 1) // block_m
        ) * block_m
        route_blocks = route_slots // block_m
        global_scale = torch.empty(1, device="cuda", dtype=torch.float32)
        routes = torch.empty(route_slots, device="cuda", dtype=torch.int32)
        block_experts = torch.empty(route_blocks, device="cuda", dtype=torch.int32)
        route_count = torch.empty(1, device="cuda", dtype=torch.int32)
        topk_weights = torch.empty(route_slots, device="cuda", dtype=torch.float32)
        scratch_elements = max(
            SIZE_N * route_slots,
            4 * 256 * block_m * 256,
        )
        scratch = torch.empty(scratch_elements, device="cuda", dtype=torch.float32)
        locks = torch.empty(1_024, device="cuda", dtype=torch.int32)
        def launch(_index: int) -> None:
            row_offset = 0
            while row_offset < args.rows:
                chunk_rows = min(args.rows - row_offset, 256)
                chunk_block_m = bucket_block_m(chunk_rows)
                buffers = CoordinatorBuffers(
                    device_buffer_view(
                        inputs,
                        row_offset * SIZE_K * torch.bfloat16.itemsize,
                        chunk_rows * SIZE_K * torch.bfloat16.itemsize,
                    ),
                    device_buffer(packed_weight),
                    device_buffer_view(
                        output,
                        row_offset * SIZE_N * torch.bfloat16.itemsize,
                        chunk_rows * SIZE_N * torch.bfloat16.itemsize,
                    ),
                    device_buffer(scales),
                    device_buffer(global_scale),
                    device_buffer(routes),
                    device_buffer(block_experts),
                    device_buffer(route_count),
                    device_buffer(topk_weights),
                    device_buffer(scratch),
                    device_buffer(locks),
                )
                check_status(
                    native.glmrt_cuda_w8a16_packed_o_initialize_launch_buffers_async(
                        ctypes.byref(buffers), chunk_rows, chunk_block_m, stream
                    ),
                    "packed O metadata initialization",
                )
                check_status(
                    native.glmrt_cuda_w8a16_packed_o_async(
                        ctypes.byref(buffers), chunk_rows, stream
                    ),
                    "packed O AOT launch",
                )
                row_offset += chunk_rows

    if selected_kernel == "aot" and args.rows <= 16:
        recurrent_output = torch.empty_like(output)

        def recurrent_launch(_index: int) -> None:
            for row in range(args.rows):
                check_status(
                    native.glmrt_cuda_linear_w8a16_group256_m1_warp_packed_async(
                        inputs[row].data_ptr(),
                        packed_weight.data_ptr(),
                        scales.data_ptr(),
                        recurrent_output[row].data_ptr(),
                        SIZE_K,
                        SIZE_N,
                        stream,
                    ),
                    "packed W8 recurrent reference launch",
                )

    launch(0)
    if recurrent_launch is not None:
        recurrent_launch(0)
    torch.cuda.synchronize()
    mismatch_count = (
        torch.count_nonzero(output != recurrent_output).item()
        if recurrent_output is not None
        else 0
    )
    if selected_kernel == "parity" and mismatch_count:
        raise RuntimeError(
            f"packed parity output differs from recurrent packed M=1: "
            f"mismatches={mismatch_count}"
        )
    actual = output.float()
    reference = inputs.float() @ weight.float().T
    relative_l2 = ((actual - reference).norm() / reference.norm()).item()
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(), reference.flatten(), dim=0
    ).item()
    if not torch.isfinite(actual).all() or relative_l2 > 0.02 or cosine < 0.999:
        raise RuntimeError(
            f"packed O numerical validation failed: relative_l2={relative_l2:.9f} "
            f"cosine={cosine:.9f}"
        )
    timing = bench(
        launch,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    recurrent_timing = (
        bench(
            recurrent_launch,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        if recurrent_launch is not None
        else None
    )
    recurrent_text = (
        f" recurrent_median_ms={recurrent_timing.median_ms:.6f}"
        if recurrent_timing is not None
        else ""
    )
    print(
        f"rows={args.rows} kernel={selected_kernel} "
        f"relative_l2={relative_l2:.9f} cosine={cosine:.9f} "
        f"recurrent_mismatches={mismatch_count} "
        f"median_ms={timing.median_ms:.6f} "
        f"range_ms={timing.minimum_ms:.6f}-{timing.maximum_ms:.6f}"
        f"{recurrent_text}"
    )


if __name__ == "__main__":
    main()
