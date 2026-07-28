#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import math
import random
import statistics
import struct
from pathlib import Path

from tune_b12x_spark_top1_native import (
    CUDA_MEMCPY_DEVICE_TO_DEVICE,
    CUDA_MEMCPY_HOST_TO_DEVICE,
    CUDA_STREAM_NON_BLOCKING,
    EXPERTS,
    HIDDEN,
    INTERMEDIATE,
    MAX_PACKED_ROUTE_SLOTS,
    MAX_ROUTE_BLOCKS,
    SCRATCH_ELEMENTS,
    Allocation,
    CudaRuntime,
    DeviceBuffer,
    SparkW4A16Buffers,
    copy_from_device,
    copy_to_device,
    measure,
)

REGIMES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
LOCK_ELEMENTS = 1_024


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class MixedW4A4Buffers(ctypes.Structure):
    _fields_ = tuple(
        (name, DeviceBuffer)
        for name in (
            "input_packed",
            "input_scale",
            "w13_weight_source",
            "w13_scale_source",
            "w13_global_scale",
            "fc1_output",
            "fc1_reordered",
            "activated",
            "w2_weight_packed",
            "w2_scale_packed",
            "w2_global_scale",
            "output",
            "packed_route_indices",
            "block_expert_ids",
            "packed_route_count",
            "topk_weights",
            "fc2_scratch",
            "locks",
        )
    )


def bf16_sample(data: bytes, count: int = 8) -> list[float]:
    values = struct.unpack_from(f"<{min(count, len(data) // 2)}H", data)
    return [struct.unpack("<f", struct.pack("<I", value << 16))[0] for value in values]


def compare_bf16(reference: bytes, candidate: bytes) -> dict[str, float | int | bool]:
    if len(reference) != len(candidate) or len(reference) % 2 != 0:
        raise ValueError("BF16 buffers must have equal even byte lengths")
    count = len(reference) // 2
    reference_bits = struct.unpack(f"<{count}H", reference)
    candidate_bits = struct.unpack(f"<{count}H", candidate)
    difference_squared = 0.0
    reference_squared = 0.0
    candidate_squared = 0.0
    dot = 0.0
    max_abs_error = 0.0
    finite = True
    mismatches = 0
    for reference_bit, candidate_bit in zip(
        reference_bits, candidate_bits, strict=True
    ):
        reference_value = struct.unpack("<f", struct.pack("<I", reference_bit << 16))[0]
        candidate_value = struct.unpack("<f", struct.pack("<I", candidate_bit << 16))[0]
        difference = candidate_value - reference_value
        difference_squared += difference * difference
        reference_squared += reference_value * reference_value
        candidate_squared += candidate_value * candidate_value
        dot += reference_value * candidate_value
        max_abs_error = max(max_abs_error, abs(difference))
        finite = finite and math.isfinite(candidate_value)
        mismatches += reference_bit != candidate_bit
    norm_product = math.sqrt(reference_squared * candidate_squared)
    return {
        "bitwise_equal": mismatches == 0,
        "different_elements": mismatches,
        "finite": finite,
        "max_abs_error": max_abs_error,
        "relative_l2_error": (
            math.sqrt(difference_squared / reference_squared)
            if reference_squared != 0.0
            else math.sqrt(difference_squared)
        ),
        "cosine_similarity": dot / norm_product if norm_product != 0.0 else 1.0,
    }


class MixedNativeLibrary:
    def __init__(self, path: Path) -> None:
        self.lib = ctypes.CDLL(str(path.resolve()))
        self.lib.glmrt_last_error.argtypes = (
            ctypes.c_char_p,
            ctypes.c_size_t,
        )
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
        self.lib.glmrt_cuda_b12x_spark_w4a16_top1_async.argtypes = (
            ctypes.POINTER(SparkW4A16Buffers),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_void_p,
        )
        self.lib.glmrt_cuda_b12x_spark_mixed_w4a4_candidate_async.argtypes = (
            ctypes.POINTER(MixedW4A4Buffers),
            ctypes.c_size_t,
            ctypes.c_void_p,
        )
        self.lib.glmrt_cuda_b12x_spark_mixed_w4a4_grid_candidate_async.argtypes = (
            ctypes.POINTER(MixedW4A4Buffers),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        )
        self.lib.glmrt_cuda_b12x_spark_mixed_w4a4_candidate_requirements.argtypes = (
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
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
            "glmrt_cuda_b12x_spark_mixed_w4a4_candidate_async",
            "glmrt_cuda_b12x_spark_mixed_w4a4_grid_candidate_async",
            "glmrt_cuda_b12x_spark_mixed_w4a4_candidate_requirements",
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the off-path mixed W4A4-FC1/BF16-FC2 exact AOT buckets "
            "against production W4A16 while rotating expert weights."
        )
    )
    parser.add_argument("--rows", default=",".join(str(value) for value in REGIMES))
    parser.add_argument("--grid-x", type=int, choices=range(1, 97))
    parser.add_argument("--native-lib", type=Path, required=True)
    parser.add_argument("--weight-sets", type=int, choices=range(1, 17), default=8)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--constant-weights", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.repeats) < 1:
        parser.error("warmup, iterations, and repeats must be positive")
    try:
        row_counts = tuple(int(value) for value in args.rows.split(","))
    except ValueError as error:
        parser.error(f"invalid --rows: {error}")
    if not row_counts or len(set(row_counts)) != len(row_counts):
        parser.error("--rows must contain unique exact bucket sizes")
    if any(rows not in REGIMES for rows in row_counts):
        parser.error(f"--rows must be drawn from {REGIMES}")
    max_rows = max(row_counts)

    runtime = CudaRuntime()
    native = MixedNativeLibrary(args.native_lib)
    candidate = native.lib.glmrt_cuda_b12x_spark_mixed_w4a4_grid_candidate_async
    candidate_requirements = {}
    for rows in row_counts:
        values = [ctypes.c_size_t() for _ in range(5)]
        native.check(
            native.lib.glmrt_cuda_b12x_spark_mixed_w4a4_candidate_requirements(
                rows, *(ctypes.byref(value) for value in values)
            ),
            f"query M{rows} mixed W4A4 requirements",
        )
        candidate_requirements[rows] = {
            "block_size": values[0].value,
            "route_slots": values[1].value,
            "route_blocks": values[2].value,
            "scratch_elements": values[3].value,
            "default_grid_x": values[4].value,
        }
    max_route_slots = max(
        MAX_PACKED_ROUTE_SLOTS,
        *(item["route_slots"] for item in candidate_requirements.values()),
    )
    max_route_blocks = max(
        MAX_ROUTE_BLOCKS,
        *(item["route_blocks"] for item in candidate_requirements.values()),
    )
    max_scratch_elements = max(
        SCRATCH_ELEMENTS,
        *(item["scratch_elements"] for item in candidate_requirements.values()),
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
        source_w13 = Allocation(runtime, args.weight_sets * w13_weight_bytes)
        packed_w13 = Allocation(runtime, args.weight_sets * w13_weight_bytes)
        packed_w2 = Allocation(runtime, args.weight_sets * w2_weight_bytes)
        source_w13_scale = Allocation(runtime, args.weight_sets * w13_scale_bytes)
        packed_w13_scale = Allocation(runtime, args.weight_sets * w13_scale_bytes)
        packed_w2_scale = Allocation(runtime, args.weight_sets * w2_scale_bytes)
        pack_source = Allocation(runtime, w13_weight_bytes)

        def memset(allocation: Allocation, value: int, nbytes: int) -> None:
            runtime.check(
                runtime.lib.cudaMemsetAsync(allocation.ptr, value, nbytes, stream),
                "initialize CUDA allocation",
            )

        def copy_host(destination: ctypes.c_void_p, data: bytes) -> None:
            host = ctypes.create_string_buffer(data)
            runtime.check(
                runtime.lib.cudaMemcpy(
                    destination,
                    ctypes.cast(host, ctypes.c_void_p),
                    len(data),
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                ),
                "copy synthetic weights",
            )

        for expert_id in range(args.weight_sets):
            source = (
                bytes([0x22]) * w13_weight_bytes
                if args.constant_weights
                else rng.randbytes(w13_weight_bytes)
            )
            copy_host(source_w13.offset(expert_id * w13_weight_bytes), source)
            copy_host(pack_source.ptr, source)
            native.check(
                native.lib.glmrt_cuda_b12x_w4a16_pack_weight_async(
                    pack_source.buffer(advertised_bytes=w13_weight_bytes),
                    packed_w13.buffer(
                        offset=expert_id * w13_weight_bytes,
                        advertised_bytes=w13_weight_bytes,
                    ),
                    HIDDEN,
                    2 * INTERMEDIATE,
                    INTERMEDIATE,
                    stream,
                ),
                f"pack W13 weight {expert_id}",
            )
            runtime.check(
                runtime.lib.cudaStreamSynchronize(stream),
                f"finish W13 weight {expert_id}",
            )

            source = (
                bytes([0x22]) * w2_weight_bytes
                if args.constant_weights
                else rng.randbytes(w2_weight_bytes)
            )
            copy_host(pack_source.ptr, source)
            native.check(
                native.lib.glmrt_cuda_b12x_w4a16_pack_weight_async(
                    pack_source.buffer(advertised_bytes=w2_weight_bytes),
                    packed_w2.buffer(
                        offset=expert_id * w2_weight_bytes,
                        advertised_bytes=w2_weight_bytes,
                    ),
                    INTERMEDIATE,
                    HIDDEN,
                    0,
                    stream,
                ),
                f"pack W2 weight {expert_id}",
            )
            runtime.check(
                runtime.lib.cudaStreamSynchronize(stream),
                f"finish W2 weight {expert_id}",
            )

        memset(source_w13_scale, 0x38, source_w13_scale.nbytes)
        memset(pack_source, 0x38, w13_scale_bytes)
        native.check(
            native.lib.glmrt_cuda_b12x_w4a16_pack_scale_async(
                pack_source.buffer(advertised_bytes=w13_scale_bytes),
                packed_w13_scale.buffer(advertised_bytes=w13_scale_bytes),
                HIDDEN,
                2 * INTERMEDIATE,
                INTERMEDIATE,
                ctypes.c_float(1.0),
                stream,
            ),
            "pack W13 scale",
        )
        memset(pack_source, 0x38, w2_scale_bytes)
        native.check(
            native.lib.glmrt_cuda_b12x_w4a16_pack_scale_async(
                pack_source.buffer(advertised_bytes=w2_scale_bytes),
                packed_w2_scale.buffer(advertised_bytes=w2_scale_bytes),
                INTERMEDIATE,
                HIDDEN,
                0,
                ctypes.c_float(1.0),
                stream,
            ),
            "pack W2 scale",
        )
        runtime.check(runtime.lib.cudaStreamSynchronize(stream), "finish scales")
        for expert_id in range(1, args.weight_sets):
            for allocation, stride in (
                (packed_w13_scale, w13_scale_bytes),
                (packed_w2_scale, w2_scale_bytes),
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

        source_global_scale = Allocation(runtime, args.weight_sets * 4)
        packed_global_scale = Allocation(runtime, args.weight_sets * 4)
        host_source_global = (ctypes.c_float * args.weight_sets)(
            *([1.0] * args.weight_sets)
        )
        host_packed_global = (ctypes.c_float * args.weight_sets)(
            *([2.0**119] * args.weight_sets)
        )
        copy_to_device(
            runtime,
            source_global_scale,
            host_source_global,
            ctypes.sizeof(host_source_global),
        )
        copy_to_device(
            runtime,
            packed_global_scale,
            host_packed_global,
            ctypes.sizeof(host_packed_global),
        )

        production_input = Allocation(runtime, max_rows * HIDDEN * 2)
        host_input = (ctypes.c_uint16 * (max_rows * HIDDEN))(
            *([0x3F80] * (max_rows * HIDDEN))
        )
        copy_to_device(runtime, production_input, host_input, ctypes.sizeof(host_input))
        candidate_input = Allocation(runtime, max_rows * HIDDEN // 2)
        input_scale = Allocation(
            runtime, align_up(max_rows, 128) * align_up(HIDDEN // 16, 4)
        )
        memset(candidate_input, 0x22, candidate_input.nbytes)
        memset(input_scale, 0x38, input_scale.nbytes)

        fc1 = Allocation(runtime, max_rows * 2 * INTERMEDIATE * 2)
        fc1_reordered = Allocation(runtime, max_rows * 2 * INTERMEDIATE * 2)
        activated = Allocation(runtime, max_rows * INTERMEDIATE * 2)
        output = Allocation(runtime, max_rows * HIDDEN * 2)
        packed_routes = Allocation(runtime, max_route_slots * 4)
        block_experts = Allocation(runtime, max_route_blocks * 4)
        route_count = Allocation(runtime, 4)
        production_topk = Allocation(runtime, max_rows * 4)
        candidate_topk = Allocation(runtime, max_rows * 4)
        fc1_scratch = Allocation(runtime, max_scratch_elements * 4)
        fc2_scratch = Allocation(runtime, max_scratch_elements * 4)
        locks = Allocation(runtime, LOCK_ELEMENTS * 4)

        production_buffers = SparkW4A16Buffers(
            production_input.buffer(),
            packed_w13.buffer(advertised_bytes=EXPERTS * w13_weight_bytes),
            packed_w2.buffer(advertised_bytes=EXPERTS * w2_weight_bytes),
            fc1.buffer(),
            activated.buffer(),
            output.buffer(),
            packed_w13_scale.buffer(advertised_bytes=EXPERTS * w13_scale_bytes),
            packed_w2_scale.buffer(advertised_bytes=EXPERTS * w2_scale_bytes),
            packed_global_scale.buffer(advertised_bytes=EXPERTS * 4),
            packed_global_scale.buffer(advertised_bytes=EXPERTS * 4),
            packed_routes.buffer(),
            block_experts.buffer(),
            route_count.buffer(),
            production_topk.buffer(),
            fc1_scratch.buffer(),
            fc2_scratch.buffer(),
            locks.buffer(),
        )
        candidate_buffers = [
            MixedW4A4Buffers(
                candidate_input.buffer(),
                input_scale.buffer(),
                source_w13.buffer(
                    offset=expert_id * w13_weight_bytes,
                    advertised_bytes=w13_weight_bytes,
                ),
                source_w13_scale.buffer(
                    offset=expert_id * w13_scale_bytes,
                    advertised_bytes=w13_scale_bytes,
                ),
                source_global_scale.buffer(offset=expert_id * 4, advertised_bytes=4),
                fc1.buffer(),
                fc1_reordered.buffer(),
                activated.buffer(),
                packed_w2.buffer(
                    offset=expert_id * w2_weight_bytes,
                    advertised_bytes=w2_weight_bytes,
                ),
                packed_w2_scale.buffer(
                    offset=expert_id * w2_scale_bytes,
                    advertised_bytes=w2_scale_bytes,
                ),
                packed_global_scale.buffer(offset=expert_id * 4, advertised_bytes=4),
                output.buffer(),
                packed_routes.buffer(),
                block_experts.buffer(),
                route_count.buffer(),
                candidate_topk.buffer(),
                fc2_scratch.buffer(),
                locks.buffer(),
            )
            for expert_id in range(args.weight_sets)
        ]

        def launch_production(expert_id: int, rows: int) -> None:
            native.check(
                native.lib.glmrt_cuda_b12x_spark_w4a16_top1_async(
                    ctypes.byref(production_buffers),
                    rows,
                    rows,
                    expert_id,
                    stream,
                ),
                "launch production W4A16",
            )

        def launch_candidate(buffers: MixedW4A4Buffers, rows: int) -> None:
            native.check(
                candidate(
                    ctypes.byref(buffers), rows, args.grid_x or 0, stream
                ),
                "launch mixed W4A4 candidate",
            )

        working_set = args.weight_sets * (
            w13_weight_bytes + w2_weight_bytes + w13_scale_bytes + w2_scale_bytes
        )
        results = []
        for rows in row_counts:
            launch_production(0, rows)
            runtime.check(
                runtime.lib.cudaStreamSynchronize(stream), "run production"
            )
            expected_fc1 = copy_from_device(
                runtime, fc1, rows * 2 * INTERMEDIATE * 2
            )
            expected_activated = copy_from_device(
                runtime, activated, rows * INTERMEDIATE * 2
            )
            expected = copy_from_device(runtime, output, rows * HIDDEN * 2)
            launch_candidate(candidate_buffers[0], rows)
            runtime.check(runtime.lib.cudaStreamSynchronize(stream), "run candidate")
            actual_fc1 = copy_from_device(
                runtime, fc1, rows * 2 * INTERMEDIATE * 2
            )
            actual_activated = copy_from_device(
                runtime, activated, rows * INTERMEDIATE * 2
            )
            actual = copy_from_device(runtime, output, rows * HIDDEN * 2)
            output_comparisons = [compare_bf16(expected, actual)]
            for expert_id, buffers in enumerate(candidate_buffers[1:], start=1):
                launch_production(expert_id, rows)
                runtime.check(
                    runtime.lib.cudaStreamSynchronize(stream),
                    f"run production expert {expert_id}",
                )
                expert_expected = copy_from_device(
                    runtime, output, rows * HIDDEN * 2
                )
                launch_candidate(buffers, rows)
                runtime.check(
                    runtime.lib.cudaStreamSynchronize(stream),
                    f"run candidate expert {expert_id}",
                )
                expert_actual = copy_from_device(
                    runtime, output, rows * HIDDEN * 2
                )
                output_comparisons.append(
                    compare_bf16(expert_expected, expert_actual)
                )

            row_bytes = 2 * INTERMEDIATE * 2
            half_bytes = INTERMEDIATE * 2
            swapped_fc1 = b"".join(
                actual_fc1[offset + half_bytes : offset + row_bytes]
                + actual_fc1[offset : offset + half_bytes]
                for offset in range(0, len(actual_fc1), row_bytes)
            )

            production_graphs = [
                native.capture(
                    stream,
                    lambda expert_id=expert_id, rows=rows: launch_production(
                        expert_id, rows
                    ),
                )
                for expert_id in range(args.weight_sets)
            ]
            candidate_graphs = [
                native.capture(
                    stream,
                    lambda buffers=buffers, rows=rows: launch_candidate(
                        buffers, rows
                    ),
                )
                for buffers in candidate_buffers
            ]
            graph_execs.extend(production_graphs)
            graph_execs.extend(candidate_graphs)
            production_samples = measure(
                runtime,
                native,
                production_graphs,
                stream,
                args.warmup,
                args.iterations,
                args.repeats,
            )
            candidate_samples = measure(
                runtime,
                native,
                candidate_graphs,
                stream,
                args.warmup,
                args.iterations,
                args.repeats,
            )
            production_median = statistics.median(production_samples)
            candidate_median = statistics.median(candidate_samples)
            result = {
                "rows": rows,
                "candidate_requirements": candidate_requirements[rows],
                "selected_grid_x": (
                    args.grid_x or candidate_requirements[rows]["default_grid_x"]
                ),
                "fc1_bitwise_equal": actual_fc1 == expected_fc1,
                "fc1_bitwise_equal_after_half_swap": swapped_fc1 == expected_fc1,
                "fc1_candidate_sample": bf16_sample(actual_fc1),
                "fc1_production_sample": bf16_sample(expected_fc1),
                "activated_bitwise_equal": actual_activated == expected_activated,
                "activated_candidate_sample": bf16_sample(actual_activated),
                "activated_production_sample": bf16_sample(expected_activated),
                "bitwise_equal": all(
                    bool(item["bitwise_equal"]) for item in output_comparisons
                ),
                "output_finite": all(
                    bool(item["finite"]) for item in output_comparisons
                ),
                "output_max_abs_error": max(
                    float(item["max_abs_error"]) for item in output_comparisons
                ),
                "output_max_different_elements": max(
                    int(item["different_elements"]) for item in output_comparisons
                ),
                "output_max_relative_l2_error": max(
                    float(item["relative_l2_error"])
                    for item in output_comparisons
                ),
                "output_min_cosine_similarity": min(
                    float(item["cosine_similarity"])
                    for item in output_comparisons
                ),
                "output_comparisons": output_comparisons,
                "output_candidate_sample": bf16_sample(actual),
                "output_production_sample": bf16_sample(expected),
                "candidate_median_ms": candidate_median,
                "candidate_samples_ms": candidate_samples,
                "candidate_speedup_vs_production_w4a16": (
                    production_median / candidate_median
                ),
                "production_w4a16_median_ms": production_median,
                "production_w4a16_samples_ms": production_samples,
            }
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

        report = {
            "benchmark": "b12x_spark_mixed_w4a4_exact_buckets_native_raw",
            "constant_weights": args.constant_weights,
            "results": results,
            "weight_sets": args.weight_sets,
            "weight_working_set_bytes_per_path": working_set,
            "serving_path_changed": False,
        }
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            Path(args.output).write_text(payload, encoding="ascii")
        print(payload, end="")
    finally:
        for graph_exec in graph_execs:
            native.lib.glmrt_cuda_graph_exec_destroy(graph_exec)
        runtime.lib.cudaStreamDestroy(stream)


if __name__ == "__main__":
    main()
