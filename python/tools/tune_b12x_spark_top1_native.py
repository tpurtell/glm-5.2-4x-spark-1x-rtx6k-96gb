#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import ctypes
import json
import random
import statistics
from pathlib import Path


HIDDEN = 6144
INTERMEDIATE = 512
EXPERTS = 256
MAX_PACKED_ROUTE_SLOTS = 20_224
MAX_ROUTE_BLOCKS = 422
SCRATCH_ELEMENTS = 1_572_864
LOCK_ELEMENTS = 194
CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2
CUDA_MEMCPY_DEVICE_TO_DEVICE = 3
CUDA_STREAM_NON_BLOCKING = 1


class DeviceBuffer(ctypes.Structure):
    _fields_ = (
        ("ptr", ctypes.c_void_p),
        ("bytes", ctypes.c_size_t),
        ("device_id", ctypes.c_int),
        ("flags", ctypes.c_uint64),
    )


class SparkW4A16Buffers(ctypes.Structure):
    _fields_ = tuple(
        (name, DeviceBuffer)
        for name in (
            "input",
            "w13_weight",
            "w2_weight",
            "fc1_output",
            "activated",
            "output",
            "w13_scale",
            "w2_scale",
            "w13_global_scale",
            "w2_global_scale",
            "packed_route_indices",
            "block_expert_ids",
            "packed_route_count",
            "topk_weights",
            "fc1_scratch",
            "fc2_scratch",
            "locks",
        )
    )


class CudaRuntime:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcudart.so")
        self.lib.cudaGetErrorString.argtypes = (ctypes.c_int,)
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p
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
        self.check(self.lib.cudaFree(None), "initialize CUDA")

    def check(self, status: int, action: str) -> None:
        if status == 0:
            return
        message = self.lib.cudaGetErrorString(status)
        detail = message.decode() if message else f"CUDA status {status}"
        raise RuntimeError(f"{action} failed: {detail}")


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

    def offset(self, offset: int) -> ctypes.c_void_p:
        if offset < 0 or offset > self.nbytes:
            raise ValueError("allocation offset is out of range")
        assert self.ptr.value is not None
        return ctypes.c_void_p(self.ptr.value + offset)

    def buffer(
        self, *, offset: int = 0, advertised_bytes: int | None = None
    ) -> DeviceBuffer:
        return DeviceBuffer(
            self.offset(offset),
            self.nbytes - offset if advertised_bytes is None else advertised_bytes,
            0,
            0,
        )


class NativeLibrary:
    def __init__(self, path: Path) -> None:
        self.lib = ctypes.CDLL(str(path.resolve()))
        self.lib.glmrt_last_error.argtypes = (
            ctypes.c_char_p,
            ctypes.c_size_t,
        )
        self.lib.glmrt_last_error.restype = ctypes.c_int
        self.lib.glmrt_cuda_b12x_w4a16_pack_weight_async.argtypes = (
            DeviceBuffer,
            DeviceBuffer,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        )
        self.lib.glmrt_cuda_b12x_w4a16_pack_scale_async.argtypes = (
            DeviceBuffer,
            DeviceBuffer,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_void_p,
        )
        top1_args = (
            ctypes.POINTER(SparkW4A16Buffers),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_void_p,
        )
        self.lib.glmrt_cuda_b12x_spark_w4a16_top1_async.argtypes = top1_args
        self.lib.glmrt_cuda_b12x_spark_w4a16_top1_grid_candidate_async.argtypes = (
            ctypes.POINTER(SparkW4A16Buffers),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_void_p,
        )
        self.lib.glmrt_cuda_graph_begin_capture.argtypes = (ctypes.c_void_p,)
        self.lib.glmrt_cuda_graph_end_capture.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )
        self.lib.glmrt_cuda_graph_launch.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        self.lib.glmrt_cuda_graph_exec_destroy.argtypes = (ctypes.c_void_p,)
        for name in (
            "glmrt_cuda_b12x_w4a16_pack_weight_async",
            "glmrt_cuda_b12x_w4a16_pack_scale_async",
            "glmrt_cuda_b12x_spark_w4a16_top1_async",
            "glmrt_cuda_b12x_spark_w4a16_top1_grid_candidate_async",
            "glmrt_cuda_graph_begin_capture",
            "glmrt_cuda_graph_end_capture",
            "glmrt_cuda_graph_launch",
            "glmrt_cuda_graph_exec_destroy",
        ):
            getattr(self.lib, name).restype = ctypes.c_int

    def check(self, status: int, action: str) -> None:
        if status == 0:
            return
        error = ctypes.create_string_buffer(512)
        self.lib.glmrt_last_error(error, len(error))
        raise RuntimeError(
            f"{action} failed with status {status}: {error.value.decode()}"
        )

    def capture(self, stream: ctypes.c_void_p, operation) -> ctypes.c_void_p:
        self.check(
            self.lib.glmrt_cuda_graph_begin_capture(stream), "begin graph capture"
        )
        operation()
        graph_exec = ctypes.c_void_p()
        self.check(
            self.lib.glmrt_cuda_graph_end_capture(stream, ctypes.byref(graph_exec)),
            "end graph capture",
        )
        if graph_exec.value is None:
            raise RuntimeError("graph capture returned a null executable")
        return graph_exec


def copy_to_device(
    runtime: CudaRuntime, destination: Allocation, source, nbytes: int
) -> None:
    runtime.check(
        runtime.lib.cudaMemcpy(
            destination.ptr,
            ctypes.cast(source, ctypes.c_void_p),
            nbytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        ),
        "copy host data to CUDA",
    )


def copy_from_device(
    runtime: CudaRuntime, source: Allocation, nbytes: int
) -> bytes:
    output = (ctypes.c_ubyte * nbytes)()
    runtime.check(
        runtime.lib.cudaMemcpy(
            ctypes.cast(output, ctypes.c_void_p),
            source.ptr,
            nbytes,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        ),
        "copy CUDA data to host",
    )
    return bytes(output)


def measure(
    runtime: CudaRuntime,
    native: NativeLibrary,
    graph_execs: list[ctypes.c_void_p],
    stream: ctypes.c_void_p,
    warmup: int,
    iterations: int,
    repeats: int,
) -> list[float]:
    for iteration in range(warmup):
        native.check(
            native.lib.glmrt_cuda_graph_launch(
                graph_execs[iteration % len(graph_execs)], stream
            ),
            "warmup graph launch",
        )
    runtime.check(runtime.lib.cudaStreamSynchronize(stream), "synchronize warmup")
    start = ctypes.c_void_p()
    end = ctypes.c_void_p()
    runtime.check(runtime.lib.cudaEventCreate(ctypes.byref(start)), "create start event")
    runtime.check(runtime.lib.cudaEventCreate(ctypes.byref(end)), "create end event")
    try:
        samples = []
        for _ in range(repeats):
            runtime.check(runtime.lib.cudaEventRecord(start, stream), "record start")
            for iteration in range(iterations):
                native.check(
                    native.lib.glmrt_cuda_graph_launch(
                        graph_execs[iteration % len(graph_execs)], stream
                    ),
                    "measured graph launch",
                )
            runtime.check(runtime.lib.cudaEventRecord(end, stream), "record end")
            runtime.check(runtime.lib.cudaEventSynchronize(end), "wait for end event")
            elapsed = ctypes.c_float()
            runtime.check(
                runtime.lib.cudaEventElapsedTime(
                    ctypes.byref(elapsed), start, end
                ),
                "read elapsed event time",
            )
            samples.append(elapsed.value / iterations)
        return samples
    finally:
        runtime.lib.cudaEventDestroy(start)
        runtime.lib.cudaEventDestroy(end)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Spark top-1 AOT grids with raw CUDA allocations and no PyTorch."
        )
    )
    parser.add_argument("--native-lib", type=Path, required=True)
    parser.add_argument(
        "--rows",
        type=int,
        choices=(1, 2, 4, 8, 16, 32, 64, 128, 256),
        required=True,
    )
    parser.add_argument("--grids", type=int, nargs="+", required=True)
    parser.add_argument("--weight-sets", type=int, choices=range(1, 17), default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if min(args.grids) < 1 or max(args.grids) > 48:
        parser.error("grids must be in 1..48")
    if min(args.warmup, args.iterations, args.repeats) < 1:
        parser.error("warmup, iterations, and repeats must be positive")

    runtime = CudaRuntime()
    native = NativeLibrary(args.native_lib)
    rng = random.Random(args.seed)
    stream = ctypes.c_void_p()
    runtime.check(
        runtime.lib.cudaStreamCreateWithFlags(
            ctypes.byref(stream), CUDA_STREAM_NON_BLOCKING
        ),
        "create CUDA stream",
    )
    graph_execs: list[ctypes.c_void_p] = []
    try:
        w13_weight_bytes = 2 * INTERMEDIATE * HIDDEN // 2
        w2_weight_bytes = HIDDEN * INTERMEDIATE // 2
        w13_scale_bytes = 2 * INTERMEDIATE * HIDDEN // 16
        w2_scale_bytes = HIDDEN * INTERMEDIATE // 16
        w13_weight = Allocation(runtime, args.weight_sets * w13_weight_bytes)
        w2_weight = Allocation(runtime, args.weight_sets * w2_weight_bytes)
        w13_scale = Allocation(runtime, args.weight_sets * w13_scale_bytes)
        w2_scale = Allocation(runtime, args.weight_sets * w2_scale_bytes)
        pack_source = Allocation(runtime, w13_weight_bytes)

        def memset(allocation: Allocation, value: int, nbytes: int) -> None:
            runtime.check(
                runtime.lib.cudaMemsetAsync(
                    allocation.ptr, value, nbytes, stream
                ),
                "initialize CUDA allocation",
            )

        def copy_pack_source(data: bytes) -> None:
            host_data = ctypes.create_string_buffer(data)
            runtime.check(
                runtime.lib.cudaMemcpy(
                    pack_source.ptr,
                    ctypes.cast(host_data, ctypes.c_void_p),
                    len(data),
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                ),
                "copy synthetic source weights",
            )

        for expert_id in range(args.weight_sets):
            copy_pack_source(rng.randbytes(w13_weight_bytes))
            native.check(
                native.lib.glmrt_cuda_b12x_w4a16_pack_weight_async(
                    pack_source.buffer(advertised_bytes=w13_weight_bytes),
                    w13_weight.buffer(
                        offset=expert_id * w13_weight_bytes,
                        advertised_bytes=w13_weight_bytes,
                    ),
                    HIDDEN,
                    2 * INTERMEDIATE,
                    INTERMEDIATE,
                    stream,
                ),
                f"pack synthetic W13 weight {expert_id}",
            )
            runtime.check(
                runtime.lib.cudaStreamSynchronize(stream),
                f"finish synthetic W13 weight {expert_id}",
            )
            copy_pack_source(rng.randbytes(w2_weight_bytes))
            native.check(
                native.lib.glmrt_cuda_b12x_w4a16_pack_weight_async(
                    pack_source.buffer(advertised_bytes=w2_weight_bytes),
                    w2_weight.buffer(
                        offset=expert_id * w2_weight_bytes,
                        advertised_bytes=w2_weight_bytes,
                    ),
                    INTERMEDIATE,
                    HIDDEN,
                    0,
                    stream,
                ),
                f"pack synthetic W2 weight {expert_id}",
            )
            runtime.check(
                runtime.lib.cudaStreamSynchronize(stream),
                f"finish synthetic W2 weight {expert_id}",
            )
        memset(pack_source, 0x38, w13_scale_bytes)
        native.check(
            native.lib.glmrt_cuda_b12x_w4a16_pack_scale_async(
                pack_source.buffer(advertised_bytes=w13_scale_bytes),
                w13_scale.buffer(advertised_bytes=w13_scale_bytes),
                HIDDEN,
                2 * INTERMEDIATE,
                INTERMEDIATE,
                ctypes.c_float(1.0),
                stream,
            ),
            "pack synthetic W13 scale",
        )
        memset(pack_source, 0x38, w2_scale_bytes)
        native.check(
            native.lib.glmrt_cuda_b12x_w4a16_pack_scale_async(
                pack_source.buffer(advertised_bytes=w2_scale_bytes),
                w2_scale.buffer(advertised_bytes=w2_scale_bytes),
                INTERMEDIATE,
                HIDDEN,
                0,
                ctypes.c_float(1.0),
                stream,
            ),
            "pack synthetic W2 scale",
        )
        runtime.check(runtime.lib.cudaStreamSynchronize(stream), "finish weight packing")

        for expert_id in range(1, args.weight_sets):
            for allocation, stride in (
                (w13_scale, w13_scale_bytes),
                (w2_scale, w2_scale_bytes),
            ):
                runtime.check(
                    runtime.lib.cudaMemcpy(
                        allocation.offset(expert_id * stride),
                        allocation.ptr,
                        stride,
                        CUDA_MEMCPY_DEVICE_TO_DEVICE,
                    ),
                    "replicate packed expert weights",
                )

        global_scale = Allocation(runtime, args.weight_sets * ctypes.sizeof(ctypes.c_float))
        host_global = (ctypes.c_float * args.weight_sets)(
            *([1.0] * args.weight_sets)
        )
        copy_to_device(runtime, global_scale, host_global, ctypes.sizeof(host_global))

        input_ = Allocation(runtime, args.rows * HIDDEN * 2)
        host_input = (ctypes.c_uint16 * (args.rows * HIDDEN))(
            *([0x3E80] * (args.rows * HIDDEN))
        )
        copy_to_device(runtime, input_, host_input, ctypes.sizeof(host_input))
        fc1 = Allocation(runtime, args.rows * 2 * INTERMEDIATE * 2)
        activated = Allocation(runtime, args.rows * INTERMEDIATE * 2)
        output = Allocation(runtime, args.rows * HIDDEN * 2)
        packed_routes = Allocation(runtime, MAX_PACKED_ROUTE_SLOTS * 4)
        block_experts = Allocation(runtime, MAX_ROUTE_BLOCKS * 4)
        route_count = Allocation(runtime, 4)
        topk_weights = Allocation(runtime, args.rows * 4)
        fc1_scratch = Allocation(runtime, SCRATCH_ELEMENTS * 4)
        fc2_scratch = Allocation(runtime, SCRATCH_ELEMENTS * 4)
        locks = Allocation(runtime, LOCK_ELEMENTS * 4)

        buffers = SparkW4A16Buffers(
            input_.buffer(),
            w13_weight.buffer(advertised_bytes=EXPERTS * w13_weight_bytes),
            w2_weight.buffer(advertised_bytes=EXPERTS * w2_weight_bytes),
            fc1.buffer(),
            activated.buffer(),
            output.buffer(),
            w13_scale.buffer(advertised_bytes=EXPERTS * w13_scale_bytes),
            w2_scale.buffer(advertised_bytes=EXPERTS * w2_scale_bytes),
            global_scale.buffer(advertised_bytes=EXPERTS * 4),
            global_scale.buffer(advertised_bytes=EXPERTS * 4),
            packed_routes.buffer(),
            block_experts.buffer(),
            route_count.buffer(),
            topk_weights.buffer(),
            fc1_scratch.buffer(),
            fc2_scratch.buffer(),
            locks.buffer(),
        )

        def launch_current(expert_id: int) -> None:
            native.check(
                native.lib.glmrt_cuda_b12x_spark_w4a16_top1_async(
                    ctypes.byref(buffers),
                    args.rows,
                    args.rows,
                    expert_id,
                    stream,
                ),
                "launch current top-1 AOT",
            )

        def launch_candidate(expert_id: int, grid: int) -> None:
            native.check(
                native.lib.glmrt_cuda_b12x_spark_w4a16_top1_grid_candidate_async(
                    ctypes.byref(buffers),
                    args.rows,
                    args.rows,
                    expert_id,
                    grid,
                    stream,
                ),
                "launch candidate top-1 AOT grid",
            )

        launch_current(0)
        runtime.check(runtime.lib.cudaStreamSynchronize(stream), "run current AOT")
        expected = copy_from_device(runtime, output, args.rows * HIDDEN * 2)
        current_graphs = [
            native.capture(stream, lambda expert_id=expert_id: launch_current(expert_id))
            for expert_id in range(args.weight_sets)
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
        current_grid = {
            1: 8,
            2: 16,
            4: 32,
            8: 32,
            16: 48,
            32: 48,
            64: 48,
            128: 48,
            256: 48,
        }[args.rows]
        print(
            json.dumps(
                {
                    "benchmark": "b12x_spark_top1_native_raw",
                    "grid_x": current_grid,
                    "median_ms": statistics.median(current_samples),
                    "min_ms": min(current_samples),
                    "rows": args.rows,
                    "samples_ms": current_samples,
                    "weight_sets": args.weight_sets,
                    "weight_working_set_bytes": args.weight_sets
                    * (
                        w13_weight_bytes
                        + w2_weight_bytes
                        + w13_scale_bytes
                        + w2_scale_bytes
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )

        results = []
        for grid in args.grids:
            launch_candidate(0, grid)
            runtime.check(
                runtime.lib.cudaStreamSynchronize(stream), "validate candidate AOT"
            )
            actual = copy_from_device(runtime, output, args.rows * HIDDEN * 2)
            candidate_graphs = [
                native.capture(
                    stream,
                    lambda expert_id=expert_id, grid=grid: launch_candidate(
                        expert_id, grid
                    ),
                )
                for expert_id in range(args.weight_sets)
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
            result = {
                "benchmark": "b12x_spark_top1_native_grid_candidate_raw",
                "bitwise_equal": actual == expected,
                "grid_x": grid,
                "median_ms": statistics.median(samples),
                "min_ms": min(samples),
                "rows": args.rows,
                "samples_ms": samples,
                "weight_sets": args.weight_sets,
            }
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

        print(
            json.dumps(
                {
                    "benchmark": "b12x_spark_top1_native_raw_summary",
                    "best": min(results, key=lambda item: item["median_ms"]),
                    "current_median_ms": statistics.median(current_samples),
                    "rows": args.rows,
                    "weight_sets": args.weight_sets,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        for graph_exec in graph_execs:
            native.lib.glmrt_cuda_graph_exec_destroy(graph_exec)
        runtime.lib.cudaStreamDestroy(stream)


if __name__ == "__main__":
    main()
