#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import random
import statistics
from pathlib import Path

from tune_b12x_spark_top1_native import (
    Allocation,
    CUDA_MEMCPY_DEVICE_TO_DEVICE,
    CUDA_MEMCPY_HOST_TO_DEVICE,
    CUDA_STREAM_NON_BLOCKING,
    CudaRuntime,
    DeviceBuffer,
    NativeLibrary,
    SparkW4A16Buffers,
    copy_from_device,
    copy_to_device,
    measure,
)


HIDDEN = 6144
INTERMEDIATE = 512
EXPERTS = 256
TOP_K = 8
INPUT_PAYLOAD_BYTES = HIDDEN // 2 + HIDDEN // 16
MAX_PACKED_ROUTE_SLOTS = 24_320
MAX_ROUTE_BLOCKS = 760
SCRATCH_ELEMENTS = 1_572_864
LOCK_ELEMENTS = 194


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep the production Spark packed W4A16 M1/top-8 decode grid."
    )
    parser.add_argument("--native-lib", type=Path, required=True)
    parser.add_argument(
        "--grids", type=int, nargs="+", default=(16, 24, 32, 40, 48)
    )
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    if min(args.grids) < 1 or max(args.grids) > 48:
        parser.error("grids must be in 1..48 on the 48-SM Spark")
    if min(args.warmup, args.iterations, args.repeats) < 1:
        parser.error("warmup, iterations, and repeats must be positive")

    runtime = CudaRuntime()
    native = NativeLibrary(args.native_lib)
    decode_args = (
        ctypes.POINTER(SparkW4A16Buffers),
        DeviceBuffer,
        ctypes.c_size_t,
        DeviceBuffer,
        ctypes.c_void_p,
    )
    native.lib.glmrt_cuda_b12x_spark_w4a16_decode_m1_nvfp4_async.argtypes = decode_args
    native.lib.glmrt_cuda_b12x_spark_w4a16_decode_m1_nvfp4_async.restype = ctypes.c_int
    native.lib.glmrt_cuda_b12x_spark_w4a16_decode_m1_nvfp4_grid_candidate_async.argtypes = (
        *decode_args[:-1],
        ctypes.c_int,
        ctypes.c_void_p,
    )
    native.lib.glmrt_cuda_b12x_spark_w4a16_decode_m1_nvfp4_grid_candidate_async.restype = (
        ctypes.c_int
    )

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
        w13_weight = Allocation(runtime, TOP_K * w13_weight_bytes)
        w2_weight = Allocation(runtime, TOP_K * w2_weight_bytes)
        w13_scale = Allocation(runtime, TOP_K * w13_scale_bytes)
        w2_scale = Allocation(runtime, TOP_K * w2_scale_bytes)
        pack_source = Allocation(runtime, w13_weight_bytes)

        def copy_source(data: bytes) -> None:
            host = ctypes.create_string_buffer(data)
            runtime.check(
                runtime.lib.cudaMemcpy(
                    pack_source.ptr,
                    ctypes.cast(host, ctypes.c_void_p),
                    len(data),
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                ),
                "copy synthetic packed-weight source",
            )

        for expert_id in range(TOP_K):
            copy_source(rng.randbytes(w13_weight_bytes))
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
                f"pack W13 expert {expert_id}",
            )
            runtime.check(runtime.lib.cudaStreamSynchronize(stream), "finish W13 pack")
            copy_source(rng.randbytes(w2_weight_bytes))
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
                f"pack W2 expert {expert_id}",
            )
            runtime.check(runtime.lib.cudaStreamSynchronize(stream), "finish W2 pack")

        runtime.check(
            runtime.lib.cudaMemsetAsync(
                pack_source.ptr, 0x38, w13_scale_bytes, stream
            ),
            "initialize W13 scales",
        )
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
            "pack W13 scales",
        )
        runtime.check(
            runtime.lib.cudaMemsetAsync(
                pack_source.ptr, 0x38, w2_scale_bytes, stream
            ),
            "initialize W2 scales",
        )
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
            "pack W2 scales",
        )
        runtime.check(runtime.lib.cudaStreamSynchronize(stream), "finish scale packing")
        for expert_id in range(1, TOP_K):
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
                    "replicate packed scales",
                )

        global_scale = Allocation(runtime, TOP_K * ctypes.sizeof(ctypes.c_float))
        host_global = (ctypes.c_float * TOP_K)(*([1.0] * TOP_K))
        copy_to_device(runtime, global_scale, host_global, ctypes.sizeof(host_global))
        input_payload = Allocation(runtime, INPUT_PAYLOAD_BYTES)
        host_payload = (ctypes.c_ubyte * INPUT_PAYLOAD_BYTES)(
            *([0x22] * (HIDDEN // 2) + [0x38] * (HIDDEN // 16))
        )
        copy_to_device(runtime, input_payload, host_payload, INPUT_PAYLOAD_BYTES)
        topk_ids = Allocation(runtime, TOP_K * ctypes.sizeof(ctypes.c_int32))
        host_ids = (ctypes.c_int32 * TOP_K)(*range(TOP_K))
        copy_to_device(runtime, topk_ids, host_ids, ctypes.sizeof(host_ids))
        topk_weights = Allocation(runtime, TOP_K * ctypes.sizeof(ctypes.c_float))
        host_weights = (ctypes.c_float * TOP_K)(*([1.0 / TOP_K] * TOP_K))
        copy_to_device(runtime, topk_weights, host_weights, ctypes.sizeof(host_weights))

        input_bf16 = Allocation(runtime, HIDDEN * 2)
        fc1 = Allocation(runtime, TOP_K * 2 * INTERMEDIATE * 2)
        activated = Allocation(runtime, TOP_K * INTERMEDIATE * 2)
        output = Allocation(runtime, TOP_K * HIDDEN * 2)
        packed_routes = Allocation(runtime, MAX_PACKED_ROUTE_SLOTS * 4)
        block_experts = Allocation(runtime, MAX_ROUTE_BLOCKS * 4)
        route_count = Allocation(runtime, 4)
        fc1_scratch = Allocation(runtime, SCRATCH_ELEMENTS * 4)
        fc2_scratch = Allocation(runtime, SCRATCH_ELEMENTS * 4)
        locks = Allocation(runtime, LOCK_ELEMENTS * 4)
        buffers = SparkW4A16Buffers(
            input_bf16.buffer(),
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

        def launch_current() -> None:
            native.check(
                native.lib.glmrt_cuda_b12x_spark_w4a16_decode_m1_nvfp4_async(
                    ctypes.byref(buffers),
                    input_payload.buffer(),
                    INPUT_PAYLOAD_BYTES,
                    topk_ids.buffer(),
                    stream,
                ),
                "launch current packed M1/top-8 decode",
            )

        def launch_grid(grid: int) -> None:
            native.check(
                native.lib.glmrt_cuda_b12x_spark_w4a16_decode_m1_nvfp4_grid_candidate_async(
                    ctypes.byref(buffers),
                    input_payload.buffer(),
                    INPUT_PAYLOAD_BYTES,
                    topk_ids.buffer(),
                    grid,
                    stream,
                ),
                f"launch packed M1/top-8 grid {grid}",
            )

        launch_current()
        runtime.check(runtime.lib.cudaStreamSynchronize(stream), "validate current decode")
        expected = copy_from_device(runtime, output, HIDDEN * 2)
        current_graph = native.capture(stream, launch_current)
        graph_execs.append(current_graph)
        current_samples = measure(
            runtime,
            native,
            [current_graph],
            stream,
            args.warmup,
            args.iterations,
            args.repeats,
        )
        results = []
        for grid in args.grids:
            launch_grid(grid)
            runtime.check(runtime.lib.cudaStreamSynchronize(stream), "validate grid")
            actual = copy_from_device(runtime, output, HIDDEN * 2)
            graph = native.capture(stream, lambda grid=grid: launch_grid(grid))
            graph_execs.append(graph)
            samples = measure(
                runtime,
                native,
                [graph],
                stream,
                args.warmup,
                args.iterations,
                args.repeats,
            )
            result = {
                "bitwise_equal": actual == expected,
                "grid_x": grid,
                "median_ms": statistics.median(samples),
                "samples_ms": samples,
            }
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
        print(
            json.dumps(
                {
                    "benchmark": "b12x_spark_packed_w4a16_decode_m1_top8_grid",
                    "current_grid_x": 32,
                    "current_median_ms": statistics.median(current_samples),
                    "current_samples_ms": current_samples,
                    "best": min(results, key=lambda item: item["median_ms"]),
                    "expert_weight_working_set_bytes": TOP_K
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
    finally:
        for graph_exec in graph_execs:
            native.lib.glmrt_cuda_graph_exec_destroy(graph_exec)
        runtime.lib.cudaStreamDestroy(stream)


if __name__ == "__main__":
    main()
