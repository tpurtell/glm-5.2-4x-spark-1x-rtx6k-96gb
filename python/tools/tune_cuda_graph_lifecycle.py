#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import ctypes
import json
import statistics
import time
from pathlib import Path
from typing import Callable

Q_B_ROWS = 8
Q_B_SIZE_K = 2_048
Q_B_SIZE_N = 16_384
Q_B_ROUTE_SLOTS = 8
Q_B_SCRATCH_ELEMENTS = 2_097_152
Q_B_LOCK_ELEMENTS = 1_024
RMS_HIDDEN = 256
RMS_BUCKETS = (1, 2, 4, 8)
CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2
CUDA_STREAM_NON_BLOCKING = 1


class DeviceBuffer(ctypes.Structure):
    _fields_ = (
        ("ptr", ctypes.c_void_p),
        ("bytes", ctypes.c_size_t),
        ("device_id", ctypes.c_int),
        ("flags", ctypes.c_uint64),
    )


class CoordinatorW4A16Buffers(ctypes.Structure):
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


class GraphCaptureInfo(ctypes.Structure):
    _fields_ = (
        ("graph", ctypes.c_void_p),
        ("graph_exec", ctypes.c_void_p),
        ("node_count", ctypes.c_size_t),
        ("kernel_node_count", ctypes.c_size_t),
        ("memcpy_node_count", ctypes.c_size_t),
        ("memset_node_count", ctypes.c_size_t),
    )


class CudaRuntime:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcudart.so")
        self.lib.cudaGetErrorString.argtypes = (ctypes.c_int,)
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p
        self.lib.cudaGetDevice.argtypes = (ctypes.POINTER(ctypes.c_int),)
        self.lib.cudaGetDevice.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = (ctypes.c_void_p,)
        self.lib.cudaFree.restype = ctypes.c_int
        self.lib.cudaMalloc.argtypes = (
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        )
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaMemsetAsync.argtypes = (
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_void_p,
        )
        self.lib.cudaMemsetAsync.restype = ctypes.c_int
        self.lib.cudaMemcpy.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        )
        self.lib.cudaMemcpy.restype = ctypes.c_int
        self.lib.cudaStreamCreateWithFlags.argtypes = (
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
        )
        self.lib.cudaStreamCreateWithFlags.restype = ctypes.c_int
        self.lib.cudaStreamSynchronize.argtypes = (ctypes.c_void_p,)
        self.lib.cudaStreamSynchronize.restype = ctypes.c_int
        self.lib.cudaStreamDestroy.argtypes = (ctypes.c_void_p,)
        self.lib.cudaStreamDestroy.restype = ctypes.c_int
        self.lib.cudaEventCreate.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
        self.lib.cudaEventCreate.restype = ctypes.c_int
        self.lib.cudaEventRecord.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        self.lib.cudaEventRecord.restype = ctypes.c_int
        self.lib.cudaEventSynchronize.argtypes = (ctypes.c_void_p,)
        self.lib.cudaEventSynchronize.restype = ctypes.c_int
        self.lib.cudaEventElapsedTime.argtypes = (
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        self.lib.cudaEventElapsedTime.restype = ctypes.c_int
        self.lib.cudaEventDestroy.argtypes = (ctypes.c_void_p,)
        self.lib.cudaEventDestroy.restype = ctypes.c_int
        self.lib.cudaMemGetInfo.argtypes = (
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        )
        self.lib.cudaMemGetInfo.restype = ctypes.c_int
        self.graph_mem_trim = getattr(self.lib, "cudaDeviceGraphMemTrim", None)
        if self.graph_mem_trim is not None:
            self.graph_mem_trim.argtypes = (ctypes.c_int,)
            self.graph_mem_trim.restype = ctypes.c_int
        self.check(self.lib.cudaFree(None), "initialize CUDA")

    def check(self, status: int, action: str) -> None:
        if status == 0:
            return
        message = self.lib.cudaGetErrorString(status)
        detail = message.decode() if message else f"CUDA status {status}"
        raise RuntimeError(f"{action} failed: {detail}")

    def create_stream(self) -> ctypes.c_void_p:
        stream = ctypes.c_void_p()
        self.check(
            self.lib.cudaStreamCreateWithFlags(
                ctypes.byref(stream), CUDA_STREAM_NON_BLOCKING
            ),
            "create CUDA stream",
        )
        if stream.value is None:
            raise RuntimeError("CUDA stream creation returned null")
        return stream

    def synchronize(self, stream: ctypes.c_void_p) -> None:
        self.check(self.lib.cudaStreamSynchronize(stream), "synchronize CUDA stream")

    def destroy_stream(self, stream: ctypes.c_void_p) -> None:
        if stream.value is None:
            return
        self.check(self.lib.cudaStreamDestroy(stream), "destroy CUDA stream")
        stream.value = None

    def memset_async(
        self,
        allocation: Allocation,
        value: int,
        stream: ctypes.c_void_p,
    ) -> None:
        self.check(
            self.lib.cudaMemsetAsync(allocation.ptr, value, allocation.nbytes, stream),
            "memset CUDA allocation",
        )

    def copy_h2d(self, allocation: Allocation, source: ctypes.Array) -> None:
        nbytes = ctypes.sizeof(source)
        if nbytes > allocation.nbytes:
            raise ValueError("host source exceeds CUDA allocation")
        self.check(
            self.lib.cudaMemcpy(
                allocation.ptr,
                ctypes.cast(source, ctypes.c_void_p),
                nbytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            ),
            "copy host data to CUDA",
        )

    def copy_d2h(self, allocation: Allocation, nbytes: int) -> bytes:
        if nbytes > allocation.nbytes:
            raise ValueError("CUDA source exceeds allocation")
        output = (ctypes.c_ubyte * nbytes)()
        self.check(
            self.lib.cudaMemcpy(
                ctypes.cast(output, ctypes.c_void_p),
                allocation.ptr,
                nbytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            ),
            "copy CUDA data to host",
        )
        return bytes(output)

    def mem_info(self) -> tuple[int, int]:
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        self.check(
            self.lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)),
            "query CUDA memory",
        )
        return free.value, total.value

    def trim_graph_memory(self) -> bool:
        if self.graph_mem_trim is None:
            return False
        device = ctypes.c_int()
        self.check(self.lib.cudaGetDevice(ctypes.byref(device)), "query CUDA device")
        self.check(self.graph_mem_trim(device.value), "trim CUDA graph memory")
        return True

    def elapsed_graph_ms(
        self,
        stream: ctypes.c_void_p,
        iterations: int,
        operation: Callable[[], None],
    ) -> float:
        start = ctypes.c_void_p()
        end = ctypes.c_void_p()
        self.check(self.lib.cudaEventCreate(ctypes.byref(start)), "create start event")
        self.check(self.lib.cudaEventCreate(ctypes.byref(end)), "create end event")
        try:
            self.check(self.lib.cudaEventRecord(start, stream), "record start event")
            for _ in range(iterations):
                operation()
            self.check(self.lib.cudaEventRecord(end, stream), "record end event")
            self.check(self.lib.cudaEventSynchronize(end), "synchronize end event")
            elapsed = ctypes.c_float()
            self.check(
                self.lib.cudaEventElapsedTime(ctypes.byref(elapsed), start, end),
                "measure CUDA events",
            )
            return elapsed.value / iterations
        finally:
            self.lib.cudaEventDestroy(end)
            self.lib.cudaEventDestroy(start)


class Allocation:
    def __init__(self, runtime: CudaRuntime, nbytes: int) -> None:
        self.runtime = runtime
        self.nbytes = nbytes
        self.ptr = ctypes.c_void_p()
        runtime.check(
            runtime.lib.cudaMalloc(ctypes.byref(self.ptr), nbytes),
            f"allocate {nbytes} CUDA bytes",
        )
        atexit.register(self.close)

    def close(self) -> None:
        if self.ptr.value is None:
            return
        self.runtime.lib.cudaFree(self.ptr)
        self.ptr = ctypes.c_void_p()

    def buffer(self) -> DeviceBuffer:
        return DeviceBuffer(self.ptr, self.nbytes, 0, 0)


class NativeLibrary:
    def __init__(self, path: Path) -> None:
        self.lib = ctypes.CDLL(str(path.resolve()))
        self.lib.glmrt_last_error.argtypes = (ctypes.c_char_p, ctypes.c_size_t)
        self.lib.glmrt_last_error.restype = ctypes.c_int
        self.lib.glmrt_cuda_b12x_coordinator_aot_init.argtypes = ()
        self.lib.glmrt_cuda_b12x_coordinator_aot_init.restype = ctypes.c_int
        self.lib.glmrt_cuda_b12x_coordinator_w4a16_initialize_launch_buffers_async.argtypes = (
            ctypes.POINTER(CoordinatorW4A16Buffers),
            ctypes.c_void_p,
        )
        self.lib.glmrt_cuda_b12x_coordinator_w4a16_initialize_launch_buffers_async.restype = (
            ctypes.c_int
        )
        self.lib.glmrt_cuda_b12x_coordinator_w4a16_q_b_m8_async.argtypes = (
            ctypes.POINTER(CoordinatorW4A16Buffers),
            ctypes.c_size_t,
            ctypes.c_void_p,
        )
        self.lib.glmrt_cuda_b12x_coordinator_w4a16_q_b_m8_async.restype = ctypes.c_int
        self.lib.glmrt_cuda_rmsnorm_bf16_async.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_void_p,
        )
        self.lib.glmrt_cuda_rmsnorm_bf16_async.restype = ctypes.c_int
        self.lib.glmrt_cuda_graph_begin_capture.argtypes = (ctypes.c_void_p,)
        self.lib.glmrt_cuda_graph_begin_capture.restype = ctypes.c_int
        self.lib.glmrt_cuda_graph_end_capture_retained.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(GraphCaptureInfo),
        )
        self.lib.glmrt_cuda_graph_end_capture_retained.restype = ctypes.c_int
        self.lib.glmrt_cuda_graph_launch.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        self.lib.glmrt_cuda_graph_launch.restype = ctypes.c_int
        self.lib.glmrt_cuda_graph_exec_update.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        self.lib.glmrt_cuda_graph_exec_update.restype = ctypes.c_int
        self.lib.glmrt_cuda_graph_destroy.argtypes = (ctypes.c_void_p,)
        self.lib.glmrt_cuda_graph_destroy.restype = ctypes.c_int
        self.lib.glmrt_cuda_graph_exec_destroy.argtypes = (ctypes.c_void_p,)
        self.lib.glmrt_cuda_graph_exec_destroy.restype = ctypes.c_int

    def error(self) -> str:
        buffer = ctypes.create_string_buffer(1_024)
        self.lib.glmrt_last_error(buffer, len(buffer))
        return buffer.value.decode(errors="replace")

    def check(self, status: int, action: str) -> None:
        if status != 0:
            raise RuntimeError(f"{action} failed with status {status}: {self.error()}")

    def aot_init(self) -> None:
        self.check(
            self.lib.glmrt_cuda_b12x_coordinator_aot_init(),
            "initialize coordinator B12X AOT modules",
        )

    def begin_capture(self, stream: ctypes.c_void_p) -> None:
        self.check(
            self.lib.glmrt_cuda_graph_begin_capture(stream),
            "begin CUDA graph capture",
        )

    def end_capture(self, stream: ctypes.c_void_p) -> CapturedGraph:
        info = GraphCaptureInfo()
        self.check(
            self.lib.glmrt_cuda_graph_end_capture_retained(stream, ctypes.byref(info)),
            "end CUDA graph capture",
        )
        if info.graph is None or info.graph_exec is None:
            raise RuntimeError("retained capture returned null handles")
        return CapturedGraph(self, info)

    def launch(self, graph_exec: ctypes.c_void_p, stream: ctypes.c_void_p) -> None:
        self.check(
            self.lib.glmrt_cuda_graph_launch(graph_exec, stream),
            "launch CUDA graph",
        )

    def update(self, graph_exec: ctypes.c_void_p, graph: ctypes.c_void_p) -> None:
        self.check(
            self.lib.glmrt_cuda_graph_exec_update(graph_exec, graph),
            "update CUDA graph exec",
        )

    def destroy_graph(self, graph: ctypes.c_void_p) -> None:
        self.check(self.lib.glmrt_cuda_graph_destroy(graph), "destroy CUDA graph")

    def destroy_exec(self, graph_exec: ctypes.c_void_p) -> None:
        self.check(
            self.lib.glmrt_cuda_graph_exec_destroy(graph_exec),
            "destroy CUDA graph exec",
        )


class CapturedGraph:
    def __init__(self, native: NativeLibrary, info: GraphCaptureInfo) -> None:
        self.native = native
        self.graph = ctypes.c_void_p(info.graph)
        self.graph_exec = ctypes.c_void_p(info.graph_exec)
        self.node_count = info.node_count
        self.kernel_node_count = info.kernel_node_count
        self.memcpy_node_count = info.memcpy_node_count
        self.memset_node_count = info.memset_node_count

    def launch(self, stream: ctypes.c_void_p) -> None:
        self.native.launch(self.graph_exec, stream)

    def destroy_graph(self) -> None:
        if self.graph.value is None:
            return
        self.native.destroy_graph(self.graph)
        self.graph = ctypes.c_void_p()

    def destroy_exec(self) -> None:
        if self.graph_exec.value is None:
            return
        self.native.destroy_exec(self.graph_exec)
        self.graph_exec = ctypes.c_void_p()

    def close(self, *, exec_first: bool = True) -> None:
        if exec_first:
            self.destroy_exec()
            self.destroy_graph()
        else:
            self.destroy_graph()
            self.destroy_exec()


class QBFixture:
    def __init__(self, runtime: CudaRuntime) -> None:
        self.runtime = runtime
        self.allocations = {
            "input": Allocation(runtime, Q_B_ROWS * Q_B_SIZE_K * 2),
            "weight": Allocation(runtime, Q_B_SIZE_N * Q_B_SIZE_K // 2),
            "output": Allocation(runtime, Q_B_ROWS * Q_B_SIZE_N * 2),
            "scale": Allocation(runtime, Q_B_SIZE_N * Q_B_SIZE_K // 16),
            "global_scale": Allocation(runtime, 4),
            "packed_route_indices": Allocation(runtime, Q_B_ROUTE_SLOTS * 4),
            "block_expert_ids": Allocation(runtime, 4),
            "packed_route_count": Allocation(runtime, 4),
            "topk_weights": Allocation(runtime, Q_B_ROUTE_SLOTS * 4),
            "c_tmp": Allocation(runtime, Q_B_SCRATCH_ELEMENTS * 4),
            "locks": Allocation(runtime, Q_B_LOCK_ELEMENTS * 4),
        }
        self.buffers = CoordinatorW4A16Buffers(
            *(
                self.allocations[name].buffer()
                for name, _ in CoordinatorW4A16Buffers._fields_
            )
        )

    def initialize(self, native: NativeLibrary, stream: ctypes.c_void_p) -> None:
        for allocation in self.allocations.values():
            self.runtime.memset_async(allocation, 0, stream)
        self.runtime.synchronize(stream)
        one = (ctypes.c_float * 1)(1.0)
        self.runtime.copy_h2d(self.allocations["global_scale"], one)
        native.check(
            native.lib.glmrt_cuda_b12x_coordinator_w4a16_initialize_launch_buffers_async(
                ctypes.byref(self.buffers), stream
            ),
            "initialize coordinator Q-B launch buffers",
        )
        self.runtime.synchronize(stream)

    def launch(self, native: NativeLibrary, stream: ctypes.c_void_p) -> None:
        native.check(
            native.lib.glmrt_cuda_b12x_coordinator_w4a16_q_b_m8_async(
                ctypes.byref(self.buffers), Q_B_ROWS, stream
            ),
            "launch coordinator Q-B M8",
        )

    def output_is_zero(self) -> bool:
        sample = self.runtime.copy_d2h(self.allocations["output"], 4_096)
        return not any(sample)

    def close(self) -> None:
        for allocation in reversed(tuple(self.allocations.values())):
            allocation.close()


class RmsFixture:
    def __init__(self, runtime: CudaRuntime) -> None:
        self.runtime = runtime
        self.x = Allocation(runtime, max(RMS_BUCKETS) * RMS_HIDDEN * 2)
        self.weight = Allocation(runtime, RMS_HIDDEN * 2)
        self.output = Allocation(runtime, max(RMS_BUCKETS) * RMS_HIDDEN * 2)
        weight = (ctypes.c_uint16 * RMS_HIDDEN)(*([0x3F80] * RMS_HIDDEN))
        runtime.copy_h2d(self.weight, weight)

    def initialize(self, stream: ctypes.c_void_p) -> None:
        self.runtime.memset_async(self.x, 0, stream)
        self.runtime.memset_async(self.output, 0xA5, stream)
        self.runtime.synchronize(stream)

    def launch(
        self,
        native: NativeLibrary,
        stream: ctypes.c_void_p,
        rows: int,
    ) -> None:
        native.check(
            native.lib.glmrt_cuda_rmsnorm_bf16_async(
                self.x.ptr,
                self.weight.ptr,
                self.output.ptr,
                rows,
                RMS_HIDDEN,
                ctypes.c_float(1.0e-5),
                stream,
            ),
            f"launch RMSNorm M{rows}",
        )

    def close(self) -> None:
        self.output.close()
        self.weight.close()
        self.x.close()


def capture(
    native: NativeLibrary,
    stream: ctypes.c_void_p,
    operation: Callable[[], None],
) -> tuple[CapturedGraph, float]:
    start = time.perf_counter_ns()
    native.begin_capture(stream)
    operation()
    graph = native.end_capture(stream)
    elapsed_us = (time.perf_counter_ns() - start) / 1_000.0
    if graph.node_count == 0 or graph.kernel_node_count == 0:
        graph.close()
        raise RuntimeError("captured graph has no kernel nodes")
    return graph, elapsed_us


def median_us(operation: Callable[[], None], iterations: int) -> float:
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        operation()
        samples.append((time.perf_counter_ns() - start) / 1_000.0)
    return statistics.median(samples)


def run(args: argparse.Namespace) -> dict[str, object]:
    runtime = CudaRuntime()
    native = NativeLibrary(args.native_lib)
    stream = runtime.create_stream()
    q_b = QBFixture(runtime)
    rms = RmsFixture(runtime)
    graphs: list[CapturedGraph] = []
    try:
        q_b.initialize(native, stream)
        rms.initialize(stream)

        free_before_init, total_bytes = runtime.mem_info()
        start = time.perf_counter_ns()
        native.aot_init()
        cold_init_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        runtime.synchronize(stream)
        free_after_init, _ = runtime.mem_info()
        hot_init_median_us = median_us(native.aot_init, 100)

        q_b.launch(native, stream)
        runtime.synchronize(stream)
        if not q_b.output_is_zero():
            raise RuntimeError("zero-weight eager Q-B output was nonzero")

        primary, primary_capture_us = capture(
            native, stream, lambda: q_b.launch(native, stream)
        )
        graphs.append(primary)
        if primary.memset_node_count == 0:
            raise RuntimeError("Q-B graph did not retain its lock memset node")
        for _ in range(10):
            primary.launch(stream)
        runtime.synchronize(stream)
        q_b_replay_ms = runtime.elapsed_graph_ms(
            stream, args.replays, lambda: primary.launch(stream)
        )
        if not q_b.output_is_zero():
            raise RuntimeError("zero-weight Q-B graph output was nonzero")

        replacement, replacement_capture_us = capture(
            native, stream, lambda: q_b.launch(native, stream)
        )
        graphs.append(replacement)
        start = time.perf_counter_ns()
        native.update(primary.graph_exec, replacement.graph)
        update_us = (time.perf_counter_ns() - start) / 1_000.0
        replacement.destroy_exec()
        primary.destroy_graph()
        primary.graph = replacement.graph
        replacement.graph = ctypes.c_void_p()
        primary.node_count = replacement.node_count
        primary.kernel_node_count = replacement.kernel_node_count
        primary.memcpy_node_count = replacement.memcpy_node_count
        primary.memset_node_count = replacement.memset_node_count
        primary.launch(stream)
        runtime.synchronize(stream)

        incompatible, _ = capture(native, stream, lambda: rms.launch(native, stream, 1))
        graphs.append(incompatible)
        incompatible_update_error = ""
        try:
            native.update(primary.graph_exec, incompatible.graph)
        except RuntimeError as error:
            incompatible_update_error = str(error)
        if not incompatible_update_error:
            raise RuntimeError("incompatible graph exec update unexpectedly succeeded")
        primary.launch(stream)
        runtime.synchronize(stream)
        post_failed_update_launch = True
        incompatible.close()

        bucket_graphs: list[tuple[int, CapturedGraph]] = []
        bucket_capture_us: dict[str, float] = {}
        bucket_node_counts: dict[str, dict[str, int]] = {}
        for rows in RMS_BUCKETS:
            graph, elapsed_us = capture(
                native, stream, lambda rows=rows: rms.launch(native, stream, rows)
            )
            bucket_graphs.append((rows, graph))
            bucket_capture_us[str(rows)] = elapsed_us
            bucket_node_counts[str(rows)] = {
                "total": graph.node_count,
                "kernel": graph.kernel_node_count,
                "memcpy": graph.memcpy_node_count,
                "memset": graph.memset_node_count,
            }
        for _ in range(args.bucket_replay_rounds):
            for _, graph in bucket_graphs:
                graph.launch(stream)
        runtime.synchronize(stream)
        bucket_replay_ms = runtime.elapsed_graph_ms(
            stream,
            args.replays,
            lambda: [graph.launch(stream) for _, graph in bucket_graphs],
        ) / len(bucket_graphs)

        graph_first_exec_survives = True
        for index, (_, graph) in enumerate(bucket_graphs):
            if index % 2 == 0:
                graph.destroy_graph()
                graph.launch(stream)
                runtime.synchronize(stream)
                graph.destroy_exec()
            else:
                graph.destroy_exec()
                graph.destroy_graph()

        warm, _ = capture(native, stream, lambda: rms.launch(native, stream, 1))
        warm.launch(stream)
        runtime.synchronize(stream)
        warm.close()
        runtime.trim_graph_memory()
        free_before_churn, _ = runtime.mem_info()

        churn_capture_us = []
        churn_destroy_us = []
        for cycle in range(args.cycles):
            rows = RMS_BUCKETS[cycle % len(RMS_BUCKETS)]
            graph, elapsed_us = capture(
                native, stream, lambda rows=rows: rms.launch(native, stream, rows)
            )
            churn_capture_us.append(elapsed_us)
            graph.launch(stream)
            runtime.synchronize(stream)
            start = time.perf_counter_ns()
            graph.close(exec_first=cycle % 2 == 0)
            churn_destroy_us.append((time.perf_counter_ns() - start) / 1_000.0)
        runtime.trim_graph_memory()
        free_after_churn, _ = runtime.mem_info()
        retained_bytes_after_trim = max(0, free_before_churn - free_after_churn)
        if retained_bytes_after_trim > args.max_retained_bytes:
            raise RuntimeError(
                "CUDA graph churn retained "
                f"{retained_bytes_after_trim} bytes after trim, limit is "
                f"{args.max_retained_bytes}"
            )

        primary.close()
        runtime.synchronize(stream)
        return {
            "benchmark": "cuda_graph_aot_lifecycle",
            "native_lib": str(args.native_lib.resolve()),
            "configuration": {
                "q_b_shape": [Q_B_ROWS, Q_B_SIZE_N, Q_B_SIZE_K],
                "rms_buckets": list(RMS_BUCKETS),
                "replays": args.replays,
                "bucket_replay_rounds": args.bucket_replay_rounds,
                "churn_cycles": args.cycles,
            },
            "aot": {
                "cold_init_ms": cold_init_ms,
                "hot_init_median_us": hot_init_median_us,
                "free_bytes_before_init": free_before_init,
                "free_bytes_after_init": free_after_init,
                "persistent_module_bytes": free_before_init - free_after_init,
            },
            "q_b_graph": {
                "capture_us": primary_capture_us,
                "replacement_capture_us": replacement_capture_us,
                "node_count": primary.node_count,
                "kernel_node_count": primary.kernel_node_count,
                "memcpy_node_count": primary.memcpy_node_count,
                "memset_node_count": primary.memset_node_count,
                "replay_ms": q_b_replay_ms,
                "identical_exec_update_us": update_us,
                "incompatible_update_rejected": True,
                "incompatible_update_error": incompatible_update_error,
                "post_failed_update_launch": post_failed_update_launch,
                "zero_output_validated": True,
            },
            "bucket_graphs": {
                "capture_us": bucket_capture_us,
                "node_counts": bucket_node_counts,
                "replay_ms_per_bucket": bucket_replay_ms,
                "graph_first_exec_survives": graph_first_exec_survives,
                "both_destroy_orders_passed": True,
            },
            "churn": {
                "capture_median_us": statistics.median(churn_capture_us),
                "capture_max_us": max(churn_capture_us),
                "destroy_median_us": statistics.median(churn_destroy_us),
                "destroy_max_us": max(churn_destroy_us),
                "free_bytes_before": free_before_churn,
                "free_bytes_after": free_after_churn,
                "retained_bytes_after_trim": retained_bytes_after_trim,
                "max_retained_bytes": args.max_retained_bytes,
                "graph_mem_trim_available": runtime.graph_mem_trim is not None,
            },
            "device_total_bytes": total_bytes,
        }
    finally:
        for graph in reversed(graphs):
            try:
                graph.close()
            except RuntimeError:
                pass
        try:
            runtime.synchronize(stream)
        except RuntimeError:
            pass
        rms.close()
        q_b.close()
        runtime.destroy_stream(stream)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate B12X AOT and retained CUDA graph lifecycle behavior."
    )
    parser.add_argument("--native-lib", type=Path, required=True)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--bucket-replay-rounds", type=int, default=25)
    parser.add_argument("--cycles", type=int, default=32)
    parser.add_argument("--max-retained-bytes", type=int, default=1 << 20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.replays, args.bucket_replay_rounds, args.cycles) <= 0:
        parser.error("replays, bucket-replay-rounds, and cycles must be positive")
    if args.max_retained_bytes < 0:
        parser.error("max-retained-bytes cannot be negative")
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii"
        )


if __name__ == "__main__":
    main()
