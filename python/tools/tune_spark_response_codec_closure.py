#!/usr/bin/env python3
from __future__ import annotations

import argparse
import array
import ctypes
import json
import math
import statistics
import struct
from pathlib import Path

from tune_b12x_spark_top1_native import (
    Allocation,
    CUDA_MEMCPY_HOST_TO_DEVICE,
    CudaRuntime,
    copy_from_device,
    measure,
)


HIDDEN = 6_144
SPARKS = 4
RESPONSE_HEADER_BYTES = 96
BF16_ROW_BYTES = HIDDEN * 2
FP8_ROW_BYTES = HIDDEN + 4
NVFP4_ROW_BYTES = HIDDEN // 2 + HIDDEN // 16
BF16_PATTERN = (
    0x0000,
    0x3D00,
    0x3D80,
    0xBE00,
    0x3E80,
    0xBF00,
    0x3F40,
    0xBF80,
    0x3FC0,
    0xC000,
    0x4040,
    0xC080,
    0x40A0,
    0xC0C0,
    0x40E0,
    0xC100,
)


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


def parse_floats(raw: str, label: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item) for item in raw.split(",") if item)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{label} must be comma-separated numbers"
        ) from error
    if not values or any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError(f"{label} values must be positive")
    return values


def bf16_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def compare_f32(reference: bytes, candidate: bytes) -> dict[str, float | bool]:
    reference_values = array.array("f")
    reference_values.frombytes(reference)
    candidate_values = array.array("f")
    candidate_values.frombytes(candidate)
    if len(reference_values) != len(candidate_values):
        raise ValueError("F32 comparison buffers differ in length")
    difference_squared = 0.0
    reference_squared = 0.0
    candidate_squared = 0.0
    dot = 0.0
    maximum = 0.0
    finite = True
    for reference_value, candidate_value in zip(
        reference_values, candidate_values, strict=True
    ):
        difference = candidate_value - reference_value
        difference_squared += difference * difference
        reference_squared += reference_value * reference_value
        candidate_squared += candidate_value * candidate_value
        dot += reference_value * candidate_value
        maximum = max(maximum, abs(difference))
        finite = finite and math.isfinite(candidate_value)
    norm_product = math.sqrt(reference_squared * candidate_squared)
    return {
        "bitwise_equal": reference == candidate,
        "cosine_similarity": dot / norm_product if norm_product else 1.0,
        "finite": finite,
        "max_abs_error": maximum,
        "relative_l2_error": (
            math.sqrt(difference_squared / reference_squared)
            if reference_squared
            else math.sqrt(difference_squared)
        ),
    }


def configure(lib: ctypes.CDLL, symbol: str, argtypes: tuple[object, ...]):
    function = getattr(lib, symbol)
    function.argtypes = argtypes
    function.restype = ctypes.c_int
    return function


class NativeRuntime:
    def __init__(self, path: Path) -> None:
        self.lib = ctypes.CDLL(str(path.resolve()))
        self.lib.glmrt_last_error.argtypes = (ctypes.c_char_p, ctypes.c_size_t)
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
        for symbol in (
            "glmrt_cuda_graph_begin_capture",
            "glmrt_cuda_graph_end_capture",
            "glmrt_cuda_graph_launch",
            "glmrt_cuda_graph_exec_destroy",
        ):
            getattr(self.lib, symbol).restype = ctypes.c_int

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
            self.lib.glmrt_cuda_graph_begin_capture(stream),
            "begin graph capture",
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
            "Measure Spark token aggregation, response codecs, coordinator "
            "combine, and an isolated RDMA wire model."
        )
    )
    parser.add_argument("--native-lib", type=Path, required=True)
    parser.add_argument("--rows", default="1,4,6,7,8,16,32,64,128,256")
    parser.add_argument("--source-sets", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--link-gbps", default="116,190")
    parser.add_argument("--fixed-one-way-us", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows_values = parse_ints(args.rows, "rows")
    link_rates = parse_floats(args.link_gbps, "link-gbps")
    if max(rows_values) > 256:
        parser.error("rows must not exceed 256")
    if min(args.warmup, args.iterations, args.repeats) < 1:
        parser.error("warmup, iterations, and repeats must be positive")
    if args.fixed_one_way_us < 0.0:
        parser.error("fixed-one-way-us must be nonnegative")

    runtime = CudaRuntime()
    native = NativeRuntime(args.native_lib)
    pointer = ctypes.c_void_p
    size = ctypes.c_size_t
    zero = configure(
        native.lib,
        "glmrt_cuda_zero_f32_async",
        (pointer, size, pointer),
    )
    aggregate = configure(
        native.lib,
        "glmrt_cuda_scatter_add_rows_bf16_weighted_to_f32_async",
        (pointer, pointer, pointer, pointer, size, size, pointer),
    )
    pack_bf16 = configure(
        native.lib,
        "glmrt_cuda_gather_rows_f32_to_bf16_candidate_async",
        (pointer, pointer, pointer, size, size, pointer),
    )
    pack_fp8 = configure(
        native.lib,
        "glmrt_cuda_gather_rows_f32_to_fp8_e4m3_row_scaled_register_candidate_async",
        (pointer, pointer, pointer, size, size, size, pointer),
    )
    pack_nvfp4 = configure(
        native.lib,
        "glmrt_cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3_policy_candidate_async",
        (pointer, pointer, pointer, size, size, size, pointer),
    )
    decode_bf16 = configure(
        native.lib,
        "glmrt_cuda_scatter_add_rows_bf16_to_f32_async",
        (pointer, pointer, pointer, size, size, pointer),
    )
    decode_fp8 = configure(
        native.lib,
        "glmrt_cuda_scatter_add_rows_fp8_e4m3_row_scaled_to_f32_async",
        (pointer, size, pointer, pointer, size, size, pointer),
    )
    decode_nvfp4 = configure(
        native.lib,
        "glmrt_cuda_scatter_add_rows_nvfp4_e2m1_fp8_e4m3_to_f32_async",
        (pointer, size, pointer, pointer, size, size, pointer),
    )

    codecs = {
        "bf16": (BF16_ROW_BYTES, pack_bf16, decode_bf16),
        "fp8": (FP8_ROW_BYTES, pack_fp8, decode_fp8),
        "nvfp4": (NVFP4_ROW_BYTES, pack_nvfp4, decode_nvfp4),
    }
    max_rows = max(rows_values)
    max_route_rows = sum(1 + ((token * 5 + 3) % 8) for token in range(max_rows))
    route_set_bytes = max_route_rows * HIDDEN * 2
    accumulator_set_bytes = max_rows * HIDDEN * 4
    route_source = Allocation(runtime, args.source_sets * route_set_bytes)
    accumulators = Allocation(runtime, args.source_sets * accumulator_set_bytes)
    route_indices = Allocation(runtime, max_route_rows * 4)
    route_weights = Allocation(runtime, max_route_rows * 4)
    completion_indices = Allocation(runtime, max_rows * 4)
    coordinator_output = Allocation(runtime, accumulator_set_bytes)
    payloads = {
        name: Allocation(runtime, args.source_sets * max_rows * row_bytes)
        for name, (row_bytes, _, _) in codecs.items()
    }
    stream = ctypes.c_void_p()
    runtime.check(
        runtime.lib.cudaStreamCreateWithFlags(ctypes.byref(stream), 1),
        "create response closure stream",
    )

    try:
        for source_set in range(args.source_sets):
            pattern = BF16_PATTERN[source_set:] + BF16_PATTERN[:source_set]
            row = struct.pack(f"<{len(pattern)}H", *pattern) * (HIDDEN // len(pattern))
            host = ctypes.create_string_buffer(row * max_route_rows)
            runtime.check(
                runtime.lib.cudaMemcpy(
                    route_source.offset(source_set * route_set_bytes),
                    ctypes.cast(host, ctypes.c_void_p),
                    route_set_bytes,
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                ),
                f"copy route source set {source_set}",
            )

        def check(status: int, action: str) -> None:
            native.check(status, action)

        def zero_accumulator(destination: ctypes.c_void_p, rows: int) -> None:
            check(zero(destination, rows * HIDDEN, stream), "zero F32 accumulator")

        def launch_aggregate(source_set: int, rows: int, route_rows: int) -> None:
            destination = accumulators.offset(source_set * accumulator_set_bytes)
            zero_accumulator(destination, rows)
            check(
                aggregate(
                    route_source.offset(source_set * route_set_bytes),
                    route_indices.ptr,
                    route_weights.ptr,
                    destination,
                    route_rows,
                    HIDDEN,
                    stream,
                ),
                "aggregate local expert rows",
            )

        def payload_pointer(name: str, source_set: int) -> ctypes.c_void_p:
            row_bytes = codecs[name][0]
            return payloads[name].offset(source_set * max_rows * row_bytes)

        def launch_pack(name: str, source_set: int, rows: int) -> None:
            row_bytes, function, _ = codecs[name]
            source = accumulators.offset(source_set * accumulator_set_bytes)
            destination = payload_pointer(name, source_set)
            if name == "bf16":
                status = function(
                    source,
                    completion_indices.ptr,
                    destination,
                    rows,
                    HIDDEN,
                    stream,
                )
            else:
                status = function(
                    source,
                    completion_indices.ptr,
                    destination,
                    rows,
                    HIDDEN,
                    row_bytes,
                    stream,
                )
            check(status, f"pack {name} Spark response")

        def launch_spark_closure(
            name: str, source_set: int, rows: int, route_rows: int
        ) -> None:
            launch_aggregate(source_set, rows, route_rows)
            launch_pack(name, source_set, rows)

        def launch_decode(name: str, source_set: int, rows: int) -> None:
            row_bytes, _, function = codecs[name]
            zero_accumulator(coordinator_output.ptr, rows)
            for _ in range(SPARKS):
                if name == "bf16":
                    status = function(
                        payload_pointer(name, source_set),
                        completion_indices.ptr,
                        coordinator_output.ptr,
                        rows,
                        HIDDEN,
                        stream,
                    )
                else:
                    status = function(
                        payload_pointer(name, source_set),
                        row_bytes,
                        completion_indices.ptr,
                        coordinator_output.ptr,
                        rows,
                        HIDDEN,
                        stream,
                    )
                check(status, f"decode {name} coordinator response")

        results = []
        for rows in rows_values:
            route_counts = tuple(1 + ((token * 5 + 3) % 8) for token in range(rows))
            route_ids = []
            weights = []
            for token, route_count in enumerate(route_counts):
                route_ids.extend([token] * route_count)
                weights.extend([1.0 / route_count] * route_count)
            route_rows = len(route_ids)
            completion_order = tuple(
                sorted(range(rows), key=lambda token: (route_counts[token], token))
            )
            host_route_indices = (ctypes.c_uint32 * route_rows)(*route_ids)
            host_route_weights = (ctypes.c_float * route_rows)(*weights)
            host_completion = (ctypes.c_uint32 * rows)(*completion_order)
            for destination, host, label in (
                (route_indices, host_route_indices, "route indices"),
                (route_weights, host_route_weights, "route weights"),
                (completion_indices, host_completion, "completion indices"),
            ):
                runtime.check(
                    runtime.lib.cudaMemcpy(
                        destination.ptr,
                        ctypes.cast(host, ctypes.c_void_p),
                        ctypes.sizeof(host),
                        CUDA_MEMCPY_HOST_TO_DEVICE,
                    ),
                    f"copy M{rows} {label}",
                )

            for source_set in range(args.source_sets):
                launch_aggregate(source_set, rows, route_rows)
            runtime.check(
                runtime.lib.cudaStreamSynchronize(stream),
                f"prepare M{rows} accumulators",
            )
            accumulator_bytes = rows * HIDDEN * 4
            actual_aggregate = copy_from_device(
                runtime, accumulators, accumulator_bytes
            )
            expected_values = array.array(
                "f",
                (
                    bf16_to_f32(BF16_PATTERN[col % len(BF16_PATTERN)])
                    for _token in range(rows)
                    for col in range(HIDDEN)
                ),
            ).tobytes()
            aggregation_error = compare_f32(expected_values, actual_aggregate)

            row_graphs: list[ctypes.c_void_p] = []
            aggregation_graphs = [
                native.capture(
                    stream,
                    lambda source_set=source_set, rows=rows, route_rows=route_rows: (
                        launch_aggregate(source_set, rows, route_rows)
                    ),
                )
                for source_set in range(args.source_sets)
            ]
            row_graphs.extend(aggregation_graphs)
            aggregation_samples = measure(
                runtime,
                native,
                aggregation_graphs,
                stream,
                args.warmup,
                args.iterations,
                args.repeats,
            )
            codec_results = {}
            decoded_outputs = {}
            for name, (row_bytes, _, _) in codecs.items():
                for source_set in range(args.source_sets):
                    launch_pack(name, source_set, rows)
                runtime.check(
                    runtime.lib.cudaStreamSynchronize(stream),
                    f"prepare M{rows} {name} payloads",
                )
                payload_bytes = rows * row_bytes
                first_payload = copy_from_device(
                    runtime, payloads[name], payload_bytes
                )
                launch_decode(name, 0, rows)
                runtime.check(
                    runtime.lib.cudaStreamSynchronize(stream),
                    f"validate M{rows} {name} decode",
                )
                first_decode = copy_from_device(
                    runtime, coordinator_output, accumulator_bytes
                )
                launch_spark_closure(name, 0, rows, route_rows)
                launch_decode(name, 0, rows)
                runtime.check(
                    runtime.lib.cudaStreamSynchronize(stream),
                    f"repeat M{rows} {name} closure",
                )
                replay_stable = (
                    copy_from_device(runtime, payloads[name], payload_bytes)
                    == first_payload
                    and copy_from_device(
                        runtime, coordinator_output, accumulator_bytes
                    )
                    == first_decode
                )
                decoded_outputs[name] = first_decode

                pack_graphs = [
                    native.capture(
                        stream,
                        lambda name=name, source_set=source_set, rows=rows: (
                            launch_pack(name, source_set, rows)
                        ),
                    )
                    for source_set in range(args.source_sets)
                ]
                closure_graphs = [
                    native.capture(
                        stream,
                        lambda name=name, source_set=source_set, rows=rows,
                        route_rows=route_rows: launch_spark_closure(
                            name, source_set, rows, route_rows
                        ),
                    )
                    for source_set in range(args.source_sets)
                ]
                decode_graphs = [
                    native.capture(
                        stream,
                        lambda name=name, source_set=source_set, rows=rows: (
                            launch_decode(name, source_set, rows)
                        ),
                    )
                    for source_set in range(args.source_sets)
                ]
                row_graphs.extend(pack_graphs)
                row_graphs.extend(closure_graphs)
                row_graphs.extend(decode_graphs)
                pack_samples = measure(
                    runtime,
                    native,
                    pack_graphs,
                    stream,
                    args.warmup,
                    args.iterations,
                    args.repeats,
                )
                closure_samples = measure(
                    runtime,
                    native,
                    closure_graphs,
                    stream,
                    args.warmup,
                    args.iterations,
                    args.repeats,
                )
                decode_samples = measure(
                    runtime,
                    native,
                    decode_graphs,
                    stream,
                    args.warmup,
                    args.iterations,
                    args.repeats,
                )
                pack_median = statistics.median(pack_samples)
                closure_median = statistics.median(closure_samples)
                decode_median = statistics.median(decode_samples)
                wire_bytes = RESPONSE_HEADER_BYTES + rows * 4 + payload_bytes
                wire_models = {}
                for link_gbps in link_rates:
                    serialization_ms = wire_bytes * 8.0 / (link_gbps * 1e6)
                    wire_models[f"{link_gbps:g}"] = {
                        "critical_path_ms": (
                            closure_median
                            + args.fixed_one_way_us / 1_000.0
                            + serialization_ms
                            + decode_median
                        ),
                        "serialization_ms": serialization_ms,
                    }
                codec_results[name] = {
                    "coordinator_decode_median_ms": decode_median,
                    "coordinator_decode_samples_ms": decode_samples,
                    "pack_median_ms": pack_median,
                    "pack_samples_ms": pack_samples,
                    "payload_bytes_per_spark": payload_bytes,
                    "replay_stable": replay_stable,
                    "spark_closure_median_ms": closure_median,
                    "spark_closure_samples_ms": closure_samples,
                    "total_wire_bytes_four_sparks": SPARKS * wire_bytes,
                    "wire_bytes_per_spark": wire_bytes,
                    "wire_models": wire_models,
                }

            reference = decoded_outputs["bf16"]
            for name in codecs:
                codec_results[name]["error_vs_bf16"] = compare_f32(
                    reference, decoded_outputs[name]
                )
            result = {
                "aggregation_error_vs_weighted_fixture": aggregation_error,
                "aggregation_median_ms": statistics.median(aggregation_samples),
                "aggregation_samples_ms": aggregation_samples,
                "codecs": codec_results,
                "completion_order": completion_order,
                "route_count_max": max(route_counts),
                "route_count_min": min(route_counts),
                "route_rows": route_rows,
                "rows": rows,
            }
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
            for graph_exec in row_graphs:
                native.lib.glmrt_cuda_graph_exec_destroy(graph_exec)

        report = {
            "benchmark": "spark_response_codec_closure",
            "fixed_one_way_us_assumption": args.fixed_one_way_us,
            "hidden": HIDDEN,
            "link_gbps": link_rates,
            "response_header_bytes": RESPONSE_HEADER_BYTES,
            "results": results,
            "serving_path_changed": False,
            "source_sets": args.source_sets,
            "sparks": SPARKS,
        }
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.write_text(payload, encoding="ascii")
        print(payload, end="")
    finally:
        runtime.lib.cudaStreamDestroy(stream)


if __name__ == "__main__":
    main()
