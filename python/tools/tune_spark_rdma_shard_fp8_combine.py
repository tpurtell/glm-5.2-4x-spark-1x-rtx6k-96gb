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
    DeviceBuffer,
    NativeLibrary,
    copy_from_device,
    measure,
)


HIDDEN = 6144
BF16_ROW_BYTES = HIDDEN * ctypes.sizeof(ctypes.c_uint16)
F32_ROW_BYTES = HIDDEN * ctypes.sizeof(ctypes.c_float)
FP8_ROW_BYTES = HIDDEN + ctypes.sizeof(ctypes.c_float)
LOCAL_BF16 = 2
WIRE_FP8 = 2


class HostBuffer(ctypes.Structure):
    _fields_ = (
        ("ptr", ctypes.c_void_p),
        ("bytes", ctypes.c_size_t),
        ("flags", ctypes.c_uint64),
    )


class RouteShardBuffers(ctypes.Structure):
    _fields_ = (
        ("local", DeviceBuffer),
        ("peers", DeviceBuffer * 3),
        ("output_f32", DeviceBuffer),
    )


class FusedRailBuffers(ctypes.Structure):
    _fields_ = (
        ("local_bf16", DeviceBuffer),
        ("peer_rail0", DeviceBuffer * 3),
        ("peer_rail1", DeviceBuffer * 3),
        ("output_fp8", DeviceBuffer),
    )


class MappedAllocation:
    def __init__(self, native: NativeLibrary, nbytes: int) -> None:
        self.native = native
        self.host = HostBuffer()
        self.device = DeviceBuffer()
        native.check(
            native.lib.glmrt_alloc_host_buffer(nbytes, ctypes.byref(self.host)),
            f"allocate {nbytes} mapped host bytes",
        )
        try:
            native.check(
                native.lib.glmrt_cuda_host_buffer_device_alias(
                    self.host, ctypes.byref(self.device)
                ),
                "map host allocation into CUDA",
            )
        except Exception:
            native.lib.glmrt_free_host_buffer(ctypes.byref(self.host))
            raise

    def write(self, payload: bytes) -> None:
        if len(payload) != self.host.bytes:
            raise ValueError("mapped payload size mismatch")
        ctypes.memmove(self.host.ptr, payload, len(payload))

    def close(self) -> None:
        if self.host.ptr:
            self.native.check(
                self.native.lib.glmrt_free_host_buffer(ctypes.byref(self.host)),
                "free mapped host allocation",
            )
            self.host = HostBuffer()
            self.device = DeviceBuffer()


def device_slice(
    allocation: Allocation, offset: int, nbytes: int
) -> DeviceBuffer:
    if offset < 0 or nbytes < 0 or offset + nbytes > allocation.nbytes:
        raise ValueError("device slice is out of range")
    return DeviceBuffer(allocation.offset(offset), nbytes, 0, 0)


def peer_payload(rows: int, peer: int, rail: int) -> bytes:
    finite_codes = bytes((0x00, 0x28, 0x30, 0x38, 0x40, 0xB0, 0xB8, 0xC0))
    values = (finite_codes * (HIDDEN // len(finite_codes) + 1))[:HIDDEN]
    payload = bytearray(rows * FP8_ROW_BYTES)
    for row in range(rows):
        offset = row * FP8_ROW_BYTES
        rotation = (peer * 3 + rail * 5 + row) % len(finite_codes)
        rotated = values[rotation:] + values[:rotation]
        payload[offset : offset + HIDDEN] = rotated
        struct.pack_into("f", payload, offset + HIDDEN, 0.003 + 0.0001 * (peer + row))
    return bytes(payload)


def configure_native(native: NativeLibrary) -> tuple[ctypes._CFuncPtr, ...]:
    lib = native.lib
    lib.glmrt_alloc_host_buffer.argtypes = (
        ctypes.c_size_t,
        ctypes.POINTER(HostBuffer),
    )
    lib.glmrt_cuda_host_buffer_device_alias.argtypes = (
        HostBuffer,
        ctypes.POINTER(DeviceBuffer),
    )
    lib.glmrt_free_host_buffer.argtypes = (ctypes.POINTER(HostBuffer),)
    current = lib.glmrt_cuda_reduce_route_shards_to_f32_async
    current.argtypes = (
        ctypes.POINTER(RouteShardBuffers),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    )
    pack = lib.glmrt_cuda_gather_rows_f32_to_fp8_e4m3_row_scaled_async
    pack.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    fused = lib.glmrt_cuda_reduce_route_shards_bf16_fp8_to_fp8_rail_candidate_async
    fused.argtypes = (
        ctypes.POINTER(FusedRailBuffers),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    f32_to_bf16 = lib.glmrt_cuda_f32_to_bf16_async
    f32_to_bf16.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    for function in (
        lib.glmrt_alloc_host_buffer,
        lib.glmrt_cuda_host_buffer_device_alias,
        lib.glmrt_free_host_buffer,
        current,
        pack,
        fused,
        f32_to_bf16,
    ):
        function.restype = ctypes.c_int
    return current, pack, fused, f32_to_bf16


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure fused mapped-FP8 Spark shard reduction and response packing."
    )
    parser.add_argument("--native-lib", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--rail0-rows", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if not 1 <= args.rail0_rows <= args.rows <= 256:
        parser.error("require 1 <= rail0-rows <= rows <= 256")
    if min(args.warmup, args.iterations, args.repeats) < 1:
        parser.error("warmup, iterations, and repeats must be positive")

    runtime = CudaRuntime()
    native = NativeLibrary(args.native_lib)
    current, pack, fused, f32_to_bf16 = configure_native(native)
    stream = ctypes.c_void_p()
    runtime.check(
        runtime.lib.cudaStreamCreateWithFlags(ctypes.byref(stream), 1),
        "create benchmark stream",
    )

    local_f32 = Allocation(runtime, args.rows * F32_ROW_BYTES)
    local_bf16 = Allocation(runtime, args.rows * BF16_ROW_BYTES)
    output_f32 = Allocation(runtime, args.rows * F32_ROW_BYTES)
    current_output = Allocation(runtime, args.rows * FP8_ROW_BYTES)
    fused_output = Allocation(runtime, args.rows * FP8_ROW_BYTES)
    indices = Allocation(runtime, args.rows * ctypes.sizeof(ctypes.c_uint32))
    mapped: list[MappedAllocation] = []
    graphs: list[ctypes.c_void_p] = []
    try:
        host_local = (ctypes.c_float * (args.rows * HIDDEN))()
        for index in range(len(host_local)):
            host_local[index] = ((index * 17 + index // HIDDEN * 29) % 1021 - 510) / 4096.0
        runtime.check(
            runtime.lib.cudaMemcpy(
                local_f32.ptr,
                ctypes.cast(host_local, ctypes.c_void_p),
                ctypes.sizeof(host_local),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            ),
            "copy local shard source",
        )
        host_indices = (ctypes.c_uint32 * args.rows)(*range(args.rows))
        runtime.check(
            runtime.lib.cudaMemcpy(
                indices.ptr,
                ctypes.cast(host_indices, ctypes.c_void_p),
                ctypes.sizeof(host_indices),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            ),
            "copy identity row indices",
        )
        native.check(
            f32_to_bf16(
                local_f32.ptr,
                local_bf16.ptr,
                args.rows * HIDDEN,
                stream,
            ),
            "convert local shard to BF16",
        )

        rail_rows = (args.rail0_rows, args.rows - args.rail0_rows)
        peer_rails: list[list[DeviceBuffer]] = [[], []]
        for rail, rows in enumerate(rail_rows):
            for peer in range(3):
                if rows == 0:
                    peer_rails[rail].append(DeviceBuffer())
                    continue
                allocation = MappedAllocation(native, rows * FP8_ROW_BYTES)
                allocation.write(peer_payload(rows, peer, rail))
                mapped.append(allocation)
                peer_rails[rail].append(allocation.device)

        current_buffers = []
        row_offset = 0
        for rail, rows in enumerate(rail_rows):
            if rows == 0:
                continue
            current_buffers.append(
                (
                    rows,
                    RouteShardBuffers(
                        device_slice(local_bf16, row_offset * BF16_ROW_BYTES, rows * BF16_ROW_BYTES),
                        (DeviceBuffer * 3)(*peer_rails[rail]),
                        device_slice(output_f32, row_offset * F32_ROW_BYTES, rows * F32_ROW_BYTES),
                    ),
                )
            )
            row_offset += rows
        fused_buffers = FusedRailBuffers(
            local_bf16.buffer(),
            (DeviceBuffer * 3)(*peer_rails[0]),
            (DeviceBuffer * 3)(*peer_rails[1]),
            fused_output.buffer(),
        )

        def launch_current() -> None:
            for rows, buffers in current_buffers:
                native.check(
                    current(
                        ctypes.byref(buffers),
                        rows,
                        HIDDEN,
                        FP8_ROW_BYTES,
                        LOCAL_BF16,
                        WIRE_FP8,
                        3,
                        stream,
                    ),
                    "reduce route shard rail to F32",
                )
            native.check(
                pack(
                    output_f32.ptr,
                    indices.ptr,
                    current_output.ptr,
                    args.rows,
                    HIDDEN,
                    FP8_ROW_BYTES,
                    stream,
                ),
                "pack current FP8 response",
            )

        def launch_fused() -> None:
            native.check(
                fused(
                    ctypes.byref(fused_buffers),
                    args.rows,
                    args.rail0_rows,
                    HIDDEN,
                    FP8_ROW_BYTES,
                    FP8_ROW_BYTES,
                    stream,
                ),
                "launch fused route shard response",
            )

        launch_current()
        launch_fused()
        runtime.check(runtime.lib.cudaStreamSynchronize(stream), "validate fused reduction")
        output_bytes = args.rows * FP8_ROW_BYTES
        exact = copy_from_device(runtime, current_output, output_bytes) == copy_from_device(
            runtime, fused_output, output_bytes
        )

        current_graph = native.capture(stream, launch_current)
        fused_graph = native.capture(stream, launch_fused)
        graphs.extend((current_graph, fused_graph))
        current_before = measure(
            runtime,
            native,
            [current_graph],
            stream,
            args.warmup,
            args.iterations,
            args.repeats,
        )
        fused_samples = measure(
            runtime,
            native,
            [fused_graph],
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
        fused_median = statistics.median(fused_samples)
        print(
            json.dumps(
                {
                    "benchmark": "spark_rdma_shard_fp8_combine",
                    "rows": args.rows,
                    "rail0_rows": args.rail0_rows,
                    "threads": 384,
                    "mapped_peer_inputs": True,
                    "exact": exact,
                    "current_median_ms": current_median,
                    "fused_median_ms": fused_median,
                    "saved_us": (current_median - fused_median) * 1000.0,
                    "speedup": current_median / fused_median,
                    "current_samples_ms": current_samples,
                    "fused_samples_ms": fused_samples,
                },
                indent=2,
                sort_keys=True,
            )
        )
        if not exact:
            raise SystemExit("fused output differs from the current path")
    finally:
        for graph in graphs:
            native.lib.glmrt_cuda_graph_exec_destroy(graph)
        for allocation in reversed(mapped):
            allocation.close()
        runtime.lib.cudaStreamDestroy(stream)


if __name__ == "__main__":
    main()
