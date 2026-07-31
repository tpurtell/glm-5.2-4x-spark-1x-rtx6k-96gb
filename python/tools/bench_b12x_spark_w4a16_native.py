#!/usr/bin/env python3
from __future__ import annotations

import _pinned_sparkinfer  # noqa: F401

import argparse
import ctypes
import json
import statistics
from pathlib import Path

import torch


HIDDEN = 6144
INTERMEDIATE = 512
EXPERTS = 256
BENCH_EXPERTS = 8
TOP_K = 8
MAX_PACKED_ROUTE_SLOTS = 32_512
MAX_ROUTE_BLOCKS = 760
SCRATCH_ELEMENTS = 3_145_728
SPARK_LOCK_ELEMENTS = 194
# The local reference M1 object may have been exported on the 188-SM
# coordinator before the target-SM guard existed.  Give the numerical harness
# enough backing storage for that object's baked-in barrier slots; the new
# candidate itself still targets and validates the 194-element Spark ABI.
LOCK_ELEMENTS = 1_026
DEFAULT_DECODE_GRID_X = 32
MAX_DECODE_GRID_X = (SPARK_LOCK_ELEMENTS - 2) // 2
E4M3_ONE = 0x38


class DeviceBuffer(ctypes.Structure):
    _fields_ = (
        ("ptr", ctypes.c_void_p),
        ("bytes", ctypes.c_size_t),
        ("device_id", ctypes.c_int),
        ("flags", ctypes.c_uint64),
    )


class HostBuffer(ctypes.Structure):
    _fields_ = (
        ("ptr", ctypes.c_void_p),
        ("bytes", ctypes.c_size_t),
        ("flags", ctypes.c_uint64),
    )


class CudaGraphCaptureInfo(ctypes.Structure):
    _fields_ = (
        ("graph", ctypes.c_void_p),
        ("graph_exec", ctypes.c_void_p),
        ("node_count", ctypes.c_size_t),
        ("kernel_node_count", ctypes.c_size_t),
        ("memcpy_node_count", ctypes.c_size_t),
        ("memset_node_count", ctypes.c_size_t),
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
            "micro_w13_global_scale",
            "micro_w2_global_scale",
            "barrier_count",
            "barrier_epoch",
        )
    )


def device_buffer(tensor: torch.Tensor, *, advertised_bytes: int | None = None) -> DeviceBuffer:
    return DeviceBuffer(
        tensor.data_ptr(),
        tensor.numel() * tensor.element_size()
        if advertised_bytes is None
        else advertised_bytes,
        tensor.device.index or 0,
        0,
    )


def check_status(lib: ctypes.CDLL, status: int, action: str) -> None:
    if status == 0:
        return
    message = ""
    last_error = getattr(lib, "glmrt_last_error_message", None)
    if last_error is not None:
        error = ctypes.create_string_buffer(512)
        last_error(error, len(error))
        message = f": {error.value.decode()}"
    raise RuntimeError(f"{action} failed with status {status}{message}")


def measure(operation, warmup: int, iterations: int, repeats: int) -> list[float]:
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


def prefill_route_metadata(
    device: torch.device,
    rows: int,
    active_experts: int,
    routes_per_row: int,
    expert_route_counts: list[int] | None,
    topk_expert_ids: list[list[int]] | None = None,
    route_block_rows: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    logical_route_slots = rows * TOP_K
    routes_by_expert: list[list[int]] = [[] for _ in range(EXPERTS)]
    logical_route_indices = [
        row_index * TOP_K + route_slot
        for row_index in range(rows)
        for route_slot in range(routes_per_row)
    ]
    if topk_expert_ids is not None:
        if len(topk_expert_ids) != rows or any(
            len(row) != routes_per_row for row in topk_expert_ids
        ):
            raise RuntimeError("explicit prefill top-k expert IDs have the wrong shape")
        for row_index, row_experts in enumerate(topk_expert_ids):
            for route_slot, expert_id in enumerate(row_experts):
                routes_by_expert[expert_id].append(row_index * TOP_K + route_slot)
    elif expert_route_counts is None:
        for active_route_index, logical_route_index in enumerate(logical_route_indices):
            routes_by_expert[active_route_index % active_experts].append(
                logical_route_index
            )
    else:
        cursor = 0
        for expert_id, count in enumerate(expert_route_counts):
            routes_by_expert[expert_id].extend(
                logical_route_indices[cursor : cursor + count]
            )
            cursor += count
        if cursor != len(logical_route_indices):
            raise RuntimeError(
                f"expert route counts cover {cursor} routes, expected "
                f"{len(logical_route_indices)}"
            )

    packed_routes: list[int] = []
    block_experts: list[int] = []
    for expert_id, routes in enumerate(routes_by_expert):
        for start in range(0, len(routes), route_block_rows):
            block = routes[start : start + route_block_rows]
            packed_routes.extend(block)
            packed_routes.extend(
                [logical_route_slots] * (route_block_rows - len(block))
            )
            block_experts.append(expert_id)
    if len(packed_routes) > MAX_PACKED_ROUTE_SLOTS:
        raise RuntimeError(
            f"packed prefill routes {len(packed_routes)} exceed {MAX_PACKED_ROUTE_SLOTS}"
        )
    if len(block_experts) > MAX_ROUTE_BLOCKS:
        raise RuntimeError(
            f"prefill route blocks {len(block_experts)} exceed {MAX_ROUTE_BLOCKS}"
        )
    return (
        torch.tensor(packed_routes, dtype=torch.int32, device=device),
        torch.tensor(block_experts, dtype=torch.int32, device=device),
        len(packed_routes),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark exact native Spark W4A16 decode or prefill entry points."
    )
    parser.add_argument("--native-lib", type=Path, required=True)
    parser.add_argument("--scenario", choices=("decode", "prefill"), default="decode")
    parser.add_argument("--prefill-rows", type=int, default=512)
    parser.add_argument(
        "--prefill-grid-x",
        type=int,
        help="Use the benchmark-only packed-prefill persistent-grid override.",
    )
    parser.add_argument(
        "--fused-fp8-output",
        action="store_true",
        help="Pack the combined prefill result directly as row-scaled FP8.",
    )
    parser.add_argument("--active-experts", type=int, default=EXPERTS)
    parser.add_argument(
        "--weight-experts",
        type=int,
        help=(
            "allocate only this many leading expert weights for bounded prefill "
            "sweeps; defaults to all 256 experts"
        ),
    )
    parser.add_argument(
        "--expert-route-counts",
        help="Comma-separated counts for experts 0..N-1; overrides uniform routing.",
    )
    parser.add_argument(
        "--routes-per-row",
        choices=range(1, TOP_K + 1),
        type=int,
        default=TOP_K,
        help="Pack this many real routes into each logical top-k=8 prefill row.",
    )
    parser.add_argument(
        "--empty-routes",
        action="store_true",
        help="Leave packed prefill route count at zero to measure fixed overhead.",
    )
    parser.add_argument(
        "--input-memory", choices=("device", "mapped"), default="device"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument(
        "--concurrent-streams",
        choices=(1, 2, 3, 4),
        type=int,
        default=1,
        help="Replay independent workspaces concurrently against shared weights.",
    )
    parser.add_argument(
        "--concurrent-route-overlap",
        choices=range(TOP_K + 1),
        type=int,
        default=0,
        help="Leading expert IDs shared by every distinct concurrent decode row.",
    )
    parser.add_argument(
        "--weight-sets",
        type=int,
        default=1,
        help="Rotate graphs across independent weight sets to defeat hot-weight caching.",
    )
    parser.add_argument(
        "--production-staging",
        action="store_true",
        help="Capture the production pinned-H2D inputs and retained-output D2D copy.",
    )
    parser.add_argument(
        "--copy-token-output",
        action="store_true",
        help="Copy each decode lane's combined BF16 row into a shared row-major target.",
    )
    parser.add_argument(
        "--cold-capture",
        action="store_true",
        help="Capture before the first eager launch to match a fresh daemon workspace.",
    )
    parser.add_argument(
        "--native-graph-api",
        action="store_true",
        help="Use glmrt's graph capture API instead of torch.cuda.CUDAGraph.",
    )
    parser.add_argument(
        "--eager",
        action="store_true",
        help="Benchmark direct native launches without CUDA graph capture.",
    )
    parser.add_argument(
        "--poison-locks",
        action="store_true",
        help="Initialize split-K locks nonzero to verify the native launch resets them.",
    )
    parser.add_argument(
        "--verify-exact-replays",
        type=int,
        default=0,
        help="Require this many graph/eager replays to produce bit-identical token rows.",
    )
    parser.add_argument(
        "--m1-parity-candidate",
        action="store_true",
        help="Benchmark the ordered direct-top-k M=2..8 candidate.",
    )
    parser.add_argument(
        "--m1-fused-sum-candidate",
        action="store_true",
        help="Benchmark atomic fused top-k accumulation for packed M=1 decode.",
    )
    parser.add_argument(
        "--grouped-m1-parity-candidate",
        action="store_true",
        help=(
            "Benchmark grouped-by-expert block-8 M=2..8 arithmetic with "
            "fixed-order top-k reduction."
        ),
    )
    parser.add_argument(
        "--grouped-wide-m1-parity-candidate",
        action="store_true",
        help=(
            "Benchmark grouped-by-expert M=2..8 arithmetic with the selected "
            "wide FC2 tile and fixed-order top-k reduction."
        ),
    )
    parser.add_argument(
        "--verify-m1-parity",
        action="store_true",
        help="Compare every candidate row bit-for-bit with repeated M=1 decode.",
    )
    parser.add_argument(
        "--weight-memory",
        choices=("device", "managed"),
        default="device",
        help="Allocate packed weights like the benchmark or the production Spark slabs.",
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    expert_route_counts = None
    if args.expert_route_counts is not None:
        try:
            expert_route_counts = [
                int(value) for value in args.expert_route_counts.split(",")
            ]
        except ValueError as error:
            parser.error(f"invalid expert-route-counts: {error}")
        if (
            not expert_route_counts
            or len(expert_route_counts) > EXPERTS
            or any(count < 0 for count in expert_route_counts)
        ):
            parser.error(
                f"expert-route-counts must contain 1..{EXPERTS} nonnegative counts"
            )
    if min(
        args.warmup,
        args.iterations,
        args.repeats,
        args.weight_sets,
        args.concurrent_streams,
    ) < 1:
        parser.error(
            "warmup, iterations, repeats, weight-sets, and concurrent-streams must be positive"
        )
    if args.verify_exact_replays < 0:
        parser.error("verify-exact-replays must be nonnegative")
    if args.verify_exact_replays and (
        args.weight_sets != 1 or args.concurrent_streams != 1
    ):
        parser.error("exact replay verification requires one weight set and one stream")
    if args.empty_routes and args.scenario != "prefill":
        parser.error("empty-routes is only valid for the prefill scenario")
    if args.scenario == "prefill" and not 1 <= args.prefill_rows <= 2048:
        parser.error("prefill-rows must be between 1 and 2048")
    if args.prefill_grid_x is not None and (
        args.scenario != "prefill" or not 1 <= args.prefill_grid_x <= 96
    ):
        parser.error("prefill-grid-x requires prefill and must be in 1..96")
    if args.fused_fp8_output and (
        args.scenario != "prefill"
        or args.prefill_grid_x is not None
        or args.m1_parity_candidate
        or args.grouped_m1_parity_candidate
        or args.grouped_wide_m1_parity_candidate
        or args.verify_m1_parity
    ):
        parser.error(
            "fused-fp8-output requires ordinary prefill without a grid or parity check"
        )
    if args.routes_per_row != TOP_K and args.scenario != "prefill":
        parser.error("routes-per-row is only configurable for the prefill scenario")
    if args.empty_routes and args.routes_per_row != TOP_K:
        parser.error("empty-routes and partial routes-per-row are mutually exclusive")
    if args.routes_per_row != TOP_K and args.concurrent_streams != 1:
        parser.error("partial routes-per-row requires one concurrent stream")
    if args.concurrent_streams > 1 and (
        args.weight_sets != 1
        or args.input_memory != "device"
        or args.production_staging
    ):
        parser.error(
            "concurrent-streams > 1 requires one weight set, device input, "
            "and no production staging"
        )
    if args.native_graph_api and args.concurrent_streams > 1:
        parser.error("native-graph-api currently supports one stream")
    if args.concurrent_route_overlap and (
        not (
            (args.scenario == "decode" and args.concurrent_streams > 1)
            or (args.scenario == "prefill" and 2 <= args.prefill_rows <= 8)
        )
    ):
        parser.error(
            "concurrent-route-overlap requires multi-stream decode or M=2..8 prefill"
        )
    if args.eager and (args.native_graph_api or args.concurrent_streams > 1):
        parser.error("eager mode requires one stream and cannot use native-graph-api")
    if args.production_staging and (
        args.scenario != "decode" or args.input_memory != "mapped"
    ):
        parser.error("production-staging requires decode with mapped input memory")
    if args.copy_token_output and args.scenario != "decode":
        parser.error("copy-token-output requires the decode scenario")
    if args.m1_fused_sum_candidate and args.scenario != "decode":
        parser.error("m1-fused-sum-candidate requires packed decode")
    parity_candidate_count = sum(
        (
            args.m1_parity_candidate,
            args.grouped_m1_parity_candidate,
            args.grouped_wide_m1_parity_candidate,
        )
    )
    if parity_candidate_count > 1:
        parser.error(
            "direct and grouped M1 parity candidates are mutually exclusive"
        )
    parity_candidate = (
        args.m1_parity_candidate
        or args.grouped_m1_parity_candidate
        or args.grouped_wide_m1_parity_candidate
    )
    if parity_candidate and (
        args.scenario != "prefill"
        or not 2 <= args.prefill_rows <= 8
        or args.routes_per_row != TOP_K
        or args.empty_routes
    ):
        parser.error(
            "M1 parity candidates require packed prefill M=2..8 with eight routes"
        )
    if args.verify_m1_parity and not (
        parity_candidate or (args.scenario == "prefill" and args.prefill_rows == 1)
    ):
        parser.error(
            "verify-m1-parity requires an M1 parity candidate or prefill M=1"
        )
    if not TOP_K <= args.active_experts <= EXPERTS:
        parser.error(
            f"active-experts must be between {TOP_K} and {EXPERTS}"
        )
    if args.weight_experts is not None:
        if not TOP_K <= args.weight_experts <= EXPERTS:
            parser.error(f"weight-experts must be between {TOP_K} and {EXPERTS}")
        if args.active_experts > args.weight_experts:
            parser.error("active-experts cannot exceed weight-experts")
        if expert_route_counts is not None and any(
            count > 0
            for count in expert_route_counts[args.weight_experts :]
        ):
            parser.error("expert-route-counts activates an unallocated expert")

    from sparkinfer.moe._shared.kernels.w4a16.prepare import (
        prepare_w4a16_modelopt_nvfp4_weights,
    )

    torch.manual_seed(args.seed)
    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    rows = 1 if args.scenario == "decode" else args.prefill_rows
    capacity_rows = rows
    if args.scenario == "prefill":
        capacity_rows = max(2, 1 << (rows - 1).bit_length())
    weight_experts = (
        args.weight_experts or BENCH_EXPERTS
        if args.scenario == "decode"
        else args.weight_experts or EXPERTS
    )
    prepared_sets = []
    for _ in range(args.weight_sets):
        w13 = torch.randint(
            0,
            256,
            (weight_experts, 2 * INTERMEDIATE, HIDDEN // 2),
            dtype=torch.uint8,
            device=device,
        )
        w2 = torch.randint(
            0,
            256,
            (weight_experts, HIDDEN, INTERMEDIATE // 2),
            dtype=torch.uint8,
            device=device,
        )
        w13_scale = torch.full(
            (weight_experts, 2 * INTERMEDIATE, HIDDEN // 16),
            E4M3_ONE,
            dtype=torch.uint8,
            device=device,
        ).view(torch.float8_e4m3fn)
        w2_scale = torch.full(
            (weight_experts, HIDDEN, INTERMEDIATE // 16),
            E4M3_ONE,
            dtype=torch.uint8,
            device=device,
        ).view(torch.float8_e4m3fn)
        global_scale = torch.ones(weight_experts, dtype=torch.float32, device=device)
        prepared_sets.append(
            prepare_w4a16_modelopt_nvfp4_weights(
                w13,
                w13_scale,
                global_scale,
                w2,
                w2_scale,
                global_scale,
                activation="silu",
                params_dtype=torch.bfloat16,
                w13_layout="w13",
                reuse_input_storage=True,
            )
        )

    input_bf16 = torch.empty(
        (capacity_rows, HIDDEN), dtype=torch.bfloat16, device=device
    )
    fc1_output = torch.empty(
        capacity_rows * TOP_K * 2 * INTERMEDIATE,
        dtype=torch.bfloat16,
        device=device,
    )
    activated = torch.empty(
        capacity_rows * TOP_K * INTERMEDIATE,
        dtype=torch.bfloat16,
        device=device,
    )
    # Every top-k=8 object emits route rows, so the native wrapper needs the
    # full route workspace before reducing to token rows.
    output_rows = capacity_rows * TOP_K
    output = torch.empty((output_rows, HIDDEN), dtype=torch.bfloat16, device=device)
    packed_routes = torch.zeros(MAX_PACKED_ROUTE_SLOTS, dtype=torch.int32, device=device)
    block_experts = torch.zeros(MAX_ROUTE_BLOCKS, dtype=torch.int32, device=device)
    packed_route_count = torch.zeros(1, dtype=torch.int32, device=device)
    metadata_rows = (
        max(capacity_rows, args.concurrent_streams)
        if args.scenario == "decode"
        else capacity_rows
    )
    topk_weights = torch.zeros(
        (metadata_rows, TOP_K), dtype=torch.float32, device=device
    )
    topk_weights[:rows, : args.routes_per_row].fill_(1.0 / args.routes_per_row)
    if args.scenario == "decode":
        topk_weights[:, : args.routes_per_row].fill_(1.0 / args.routes_per_row)
    retained_output = torch.empty_like(output)
    fc1_scratch = torch.zeros(SCRATCH_ELEMENTS, dtype=torch.float32, device=device)
    fc2_scratch = torch.zeros(SCRATCH_ELEMENTS, dtype=torch.float32, device=device)
    lock_initial_value = 37 if args.poison_locks else 0
    locks = torch.full(
        (LOCK_ELEMENTS,), lock_initial_value, dtype=torch.int32, device=device
    )
    input_payload_row_bytes = HIDDEN // 2 + HIDDEN // 16
    payload_rows = (
        max(rows, args.concurrent_streams) if args.scenario == "decode" else rows
    )
    input_payload = torch.empty(
        (payload_rows, input_payload_row_bytes), dtype=torch.uint8, device=device
    )
    output_fp8_row_bytes = HIDDEN + ctypes.sizeof(ctypes.c_float)
    output_fp8 = torch.empty(
        (capacity_rows, output_fp8_row_bytes), dtype=torch.uint8, device=device
    )
    input_payload[:, : HIDDEN // 2].random_(0, 256)
    input_payload[:, HIDDEN // 2 :].fill_(E4M3_ONE)
    topk_id_values = [
        [
            (row_index * TOP_K + route_slot) % args.active_experts
            for route_slot in range(TOP_K)
        ]
        for row_index in range(metadata_rows)
    ]
    overlap_rows = (
        args.concurrent_streams
        if args.scenario == "decode" and args.concurrent_streams > 1
        else rows
        if args.scenario == "prefill" and rows >= 2
        else 1
    )
    if overlap_rows > 1:
        overlap = args.concurrent_route_overlap
        distinct = TOP_K - overlap
        for lane in range(1, overlap_rows):
            topk_id_values[lane][:overlap] = topk_id_values[0][:overlap]
            topk_id_values[lane][overlap:] = [
                expert_id % args.active_experts
                for expert_id in range(
                    TOP_K + (lane - 1) * distinct,
                    TOP_K + lane * distinct,
                )
            ]
    topk_ids_rows = torch.tensor(
        topk_id_values, dtype=torch.int32, device=device
    )
    topk_ids = topk_ids_rows[0]
    if args.scenario == "prefill" and not args.empty_routes:
        if rows == 1:
            packed_routes[:TOP_K].copy_(topk_ids)
            packed_route_count.fill_(TOP_K)
        else:
            route_indices, route_experts, packed_count = prefill_route_metadata(
                device,
                rows,
                args.active_experts,
                args.routes_per_row,
                expert_route_counts,
                topk_id_values[:rows]
                if rows <= 8 and expert_route_counts is None
                else None,
                route_block_rows=(
                    8
                    if (
                        args.grouped_m1_parity_candidate
                        or args.grouped_wide_m1_parity_candidate
                    )
                    else 32
                ),
            )
            packed_routes[: route_indices.numel()].copy_(route_indices)
            block_experts[: route_experts.numel()].copy_(route_experts)
            packed_route_count.fill_(packed_count)

    buffers_sets = [
        SparkW4A16Buffers(
            device_buffer(input_bf16),
            device_buffer(
                prepared.w13,
                advertised_bytes=EXPERTS * 2 * INTERMEDIATE * HIDDEN // 2,
            ),
            device_buffer(
                prepared.w2,
                advertised_bytes=EXPERTS * HIDDEN * INTERMEDIATE // 2,
            ),
            device_buffer(fc1_output),
            device_buffer(activated),
            device_buffer(output),
            device_buffer(
                prepared.w13_scale,
                advertised_bytes=EXPERTS * 2 * INTERMEDIATE * HIDDEN // 16,
            ),
            device_buffer(
                prepared.w2_scale,
                advertised_bytes=EXPERTS * HIDDEN * INTERMEDIATE // 16,
            ),
            device_buffer(prepared.w13_global_scale, advertised_bytes=EXPERTS * 4),
            device_buffer(prepared.w2_global_scale, advertised_bytes=EXPERTS * 4),
            device_buffer(packed_routes),
            device_buffer(block_experts),
            device_buffer(packed_route_count),
            device_buffer(topk_weights),
            device_buffer(fc1_scratch),
            device_buffer(fc2_scratch),
            device_buffer(locks),
        )
        for prepared in prepared_sets
    ]
    lane_outputs = [output]
    concurrent_tensors: list[torch.Tensor] = []
    if args.concurrent_streams > 1:
        prepared = prepared_sets[0]
        for lane in range(1, args.concurrent_streams):
            lane_input = torch.empty_like(input_bf16)
            lane_fc1 = torch.empty_like(fc1_output)
            lane_activated = torch.empty_like(activated)
            lane_output = torch.empty_like(output)
            lane_fc1_scratch = torch.zeros_like(fc1_scratch)
            lane_fc2_scratch = torch.zeros_like(fc2_scratch)
            lane_locks = torch.full_like(locks, lock_initial_value)
            concurrent_tensors.extend(
                (
                    lane_input,
                    lane_fc1,
                    lane_activated,
                    lane_output,
                    lane_fc1_scratch,
                    lane_fc2_scratch,
                    lane_locks,
                )
            )
            lane_outputs.append(lane_output)
            buffers_sets.append(
                SparkW4A16Buffers(
                    device_buffer(lane_input),
                    device_buffer(
                        prepared.w13,
                        advertised_bytes=EXPERTS * 2 * INTERMEDIATE * HIDDEN // 2,
                    ),
                    device_buffer(
                        prepared.w2,
                        advertised_bytes=EXPERTS * HIDDEN * INTERMEDIATE // 2,
                    ),
                    device_buffer(lane_fc1),
                    device_buffer(lane_activated),
                    device_buffer(lane_output),
                    device_buffer(
                        prepared.w13_scale,
                        advertised_bytes=EXPERTS * 2 * INTERMEDIATE * HIDDEN // 16,
                    ),
                    device_buffer(
                        prepared.w2_scale,
                        advertised_bytes=EXPERTS * HIDDEN * INTERMEDIATE // 16,
                    ),
                    device_buffer(prepared.w13_global_scale, advertised_bytes=EXPERTS * 4),
                    device_buffer(prepared.w2_global_scale, advertised_bytes=EXPERTS * 4),
                    device_buffer(packed_routes),
                    device_buffer(block_experts),
                    device_buffer(packed_route_count),
                    device_buffer(
                        topk_weights[lane : lane + 1]
                        if args.scenario == "decode"
                        else topk_weights
                    ),
                    device_buffer(lane_fc1_scratch),
                    device_buffer(lane_fc2_scratch),
                    device_buffer(lane_locks),
                )
            )
    joined_token_output = torch.empty(
        (args.concurrent_streams, HIDDEN), dtype=torch.bfloat16, device=device
    )

    lib = ctypes.CDLL(str(args.native_lib.resolve()))
    lib.glmrt_cuda_b12x_spark_aot_init.restype = ctypes.c_int
    decode_symbol = (
        "glmrt_cuda_b12x_spark_w4a16_decode_m1_fused_sum_nvfp4_async"
        if args.m1_fused_sum_candidate
        else "glmrt_cuda_b12x_spark_w4a16_decode_m1_nvfp4_async"
    )
    decode_launch = getattr(lib, decode_symbol)
    decode_launch.argtypes = (
        ctypes.POINTER(SparkW4A16Buffers),
        DeviceBuffer,
        ctypes.c_size_t,
        DeviceBuffer,
        ctypes.c_void_p,
    )
    decode_launch.restype = ctypes.c_int
    prefill_launch = lib.glmrt_cuda_b12x_spark_w4a16_prefill_topk8_nvfp4_async
    prefill_launch.argtypes = (
        ctypes.POINTER(SparkW4A16Buffers),
        DeviceBuffer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    prefill_launch.restype = ctypes.c_int
    prefill_fp8_launch = (
        lib.glmrt_cuda_b12x_spark_w4a16_prefill_topk8_nvfp4_fp8_async
    )
    prefill_fp8_launch.argtypes = (
        ctypes.POINTER(SparkW4A16Buffers),
        DeviceBuffer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        DeviceBuffer,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    prefill_fp8_launch.restype = ctypes.c_int
    prefill_grid_launch = (
        lib.glmrt_cuda_b12x_spark_w4a16_prefill_topk8_nvfp4_grid_candidate_async
    )
    prefill_grid_launch.argtypes = (
        ctypes.POINTER(SparkW4A16Buffers),
        DeviceBuffer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    )
    prefill_grid_launch.restype = ctypes.c_int
    m1_parity_launch = (
        lib.glmrt_cuda_b12x_spark_w4a16_m1_parity_m2_8_nvfp4_async
    )
    m1_parity_launch.argtypes = (
        ctypes.POINTER(SparkW4A16Buffers),
        DeviceBuffer,
        ctypes.c_size_t,
        DeviceBuffer,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    m1_parity_launch.restype = ctypes.c_int
    grouped_m1_parity_launch = (
        lib.glmrt_cuda_b12x_spark_w4a16_m1_parity_grouped_m2_8_nvfp4_async
    )
    grouped_m1_parity_launch.argtypes = (
        ctypes.POINTER(SparkW4A16Buffers),
        DeviceBuffer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    grouped_m1_parity_launch.restype = ctypes.c_int
    grouped_wide_m1_parity_launch = (
        lib.glmrt_cuda_b12x_spark_w4a16_m1_parity_grouped_wide_m2_8_nvfp4_async
    )
    grouped_wide_m1_parity_launch.argtypes = (
        ctypes.POINTER(SparkW4A16Buffers),
        DeviceBuffer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    grouped_wide_m1_parity_launch.restype = ctypes.c_int
    copy_h2d = lib.glmrt_copy_h2d_async
    copy_h2d.argtypes = (
        DeviceBuffer,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    copy_h2d.restype = ctypes.c_int
    copy_d2d = lib.glmrt_copy_d2d_async
    copy_d2d.argtypes = (
        DeviceBuffer,
        DeviceBuffer,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    copy_d2d.restype = ctypes.c_int
    lib.glmrt_alloc_managed_device_buffer.argtypes = (
        ctypes.c_size_t,
        ctypes.POINTER(DeviceBuffer),
    )
    lib.glmrt_alloc_managed_device_buffer.restype = ctypes.c_int
    lib.glmrt_free_device_buffer.argtypes = (ctypes.POINTER(DeviceBuffer),)
    lib.glmrt_free_device_buffer.restype = ctypes.c_int
    graph_begin = lib.glmrt_cuda_graph_begin_capture
    graph_begin.argtypes = (ctypes.c_void_p,)
    graph_begin.restype = ctypes.c_int
    graph_end = lib.glmrt_cuda_graph_end_capture_retained
    graph_end.argtypes = (ctypes.c_void_p, ctypes.POINTER(CudaGraphCaptureInfo))
    graph_end.restype = ctypes.c_int
    graph_launch = lib.glmrt_cuda_graph_launch
    graph_launch.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    graph_launch.restype = ctypes.c_int
    graph_exec_destroy = lib.glmrt_cuda_graph_exec_destroy
    graph_exec_destroy.argtypes = (ctypes.c_void_p,)
    graph_exec_destroy.restype = ctypes.c_int
    graph_destroy = lib.glmrt_cuda_graph_destroy
    graph_destroy.argtypes = (ctypes.c_void_p,)
    graph_destroy.restype = ctypes.c_int

    managed_weight_allocations: list[DeviceBuffer] = []
    if args.weight_memory == "managed":
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

        def managed_clone(tensor: torch.Tensor, advertised_bytes: int) -> DeviceBuffer:
            allocation = DeviceBuffer()
            actual_bytes = tensor.numel() * tensor.element_size()
            check_status(
                lib,
                lib.glmrt_alloc_managed_device_buffer(
                    actual_bytes, ctypes.byref(allocation)
                ),
                "allocating managed packed weights",
            )
            check_status(
                lib,
                copy_d2d(allocation, device_buffer(tensor), actual_bytes, stream),
                "copying packed weights into managed memory",
            )
            managed_weight_allocations.append(allocation)
            return DeviceBuffer(
                allocation.ptr,
                advertised_bytes,
                allocation.device_id,
                allocation.flags,
            )

        for buffers, prepared in zip(buffers_sets, prepared_sets, strict=True):
            buffers.w13_weight = managed_clone(
                prepared.w13, EXPERTS * 2 * INTERMEDIATE * HIDDEN // 2
            )
            buffers.w2_weight = managed_clone(
                prepared.w2, EXPERTS * HIDDEN * INTERMEDIATE // 2
            )
            buffers.w13_scale = managed_clone(
                prepared.w13_scale, EXPERTS * 2 * INTERMEDIATE * HIDDEN // 16
            )
            buffers.w2_scale = managed_clone(
                prepared.w2_scale, EXPERTS * HIDDEN * INTERMEDIATE // 16
            )
            buffers.w13_global_scale = managed_clone(
                prepared.w13_global_scale, EXPERTS * 4
            )
            buffers.w2_global_scale = managed_clone(
                prepared.w2_global_scale, EXPERTS * 4
            )
        torch.cuda.synchronize()
    check_status(
        lib,
        lib.glmrt_cuda_b12x_spark_aot_init(),
        "initializing SparkInfer AOT",
    )

    mapped_input = HostBuffer()
    mapped_topk_ids = HostBuffer()
    mapped_topk_weights = HostBuffer()
    input_payload_buffer = device_buffer(input_payload)
    if args.input_memory == "mapped":
        lib.glmrt_alloc_host_buffer.argtypes = (
            ctypes.c_size_t,
            ctypes.POINTER(HostBuffer),
        )
        lib.glmrt_alloc_host_buffer.restype = ctypes.c_int
        lib.glmrt_cuda_host_buffer_device_alias.argtypes = (
            HostBuffer,
            ctypes.POINTER(DeviceBuffer),
        )
        lib.glmrt_cuda_host_buffer_device_alias.restype = ctypes.c_int
        lib.glmrt_free_host_buffer.argtypes = (ctypes.POINTER(HostBuffer),)
        lib.glmrt_free_host_buffer.restype = ctypes.c_int
        check_status(
            lib,
            lib.glmrt_alloc_host_buffer(input_payload.numel(), ctypes.byref(mapped_input)),
            "allocating mapped input payload",
        )
        host_payload = torch.empty(input_payload.shape, dtype=torch.uint8)
        host_payload[:, : HIDDEN // 2].random_(0, 256)
        host_payload[:, HIDDEN // 2 :].fill_(E4M3_ONE)
        ctypes.memmove(mapped_input.ptr, host_payload.data_ptr(), input_payload.numel())
        input_payload_buffer = DeviceBuffer()
        check_status(
            lib,
            lib.glmrt_cuda_host_buffer_device_alias(
                mapped_input, ctypes.byref(input_payload_buffer)
            ),
            "mapping input payload into CUDA",
        )
        if args.production_staging:
            input_payload_buffer = device_buffer(input_payload)
            for host_buffer, tensor, label in (
                (mapped_topk_ids, torch.arange(TOP_K, dtype=torch.int32), "expert IDs"),
                (
                    mapped_topk_weights,
                    torch.full((TOP_K,), 1.0 / TOP_K, dtype=torch.float32),
                    "route weights",
                ),
            ):
                check_status(
                    lib,
                    lib.glmrt_alloc_host_buffer(
                        tensor.numel() * tensor.element_size(), ctypes.byref(host_buffer)
                    ),
                    f"allocating mapped {label}",
                )
                ctypes.memmove(
                    host_buffer.ptr,
                    tensor.data_ptr(),
                    tensor.numel() * tensor.element_size(),
                )

    def launch(buffers: SparkW4A16Buffers, lane: int = 0) -> None:
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        if args.scenario == "prefill" and args.routes_per_row < TOP_K:
            # The AOT kernel writes only packed real routes. Clear untouched
            # logical lanes before its fixed top-k=8 reduction consumes them.
            output.zero_()
        if args.production_staging:
            check_status(
                lib,
                copy_h2d(
                    device_buffer(input_payload),
                    mapped_input.ptr,
                    input_payload.numel(),
                    stream,
                ),
                "copying staged NVFP4 input",
            )
            check_status(
                lib,
                copy_h2d(
                    device_buffer(topk_weights),
                    mapped_topk_weights.ptr,
                    TOP_K * ctypes.sizeof(ctypes.c_float),
                    stream,
                ),
                "copying staged route weights",
            )
            check_status(
                lib,
                copy_h2d(
                    device_buffer(topk_ids),
                    mapped_topk_ids.ptr,
                    TOP_K * ctypes.sizeof(ctypes.c_int32),
                    stream,
                ),
                "copying staged expert IDs",
            )
        if args.scenario == "decode":
            lane_input_payload = (
                input_payload_buffer
                if args.input_memory == "mapped"
                else device_buffer(input_payload[lane : lane + 1])
            )
            status = decode_launch(
                ctypes.byref(buffers),
                lane_input_payload,
                input_payload_row_bytes,
                device_buffer(topk_ids_rows[lane]),
                stream,
            )
        elif args.m1_parity_candidate:
            status = m1_parity_launch(
                ctypes.byref(buffers),
                input_payload_buffer,
                input_payload_row_bytes,
                device_buffer(topk_ids_rows),
                rows,
                stream,
            )
        elif args.grouped_wide_m1_parity_candidate:
            status = grouped_wide_m1_parity_launch(
                ctypes.byref(buffers),
                input_payload_buffer,
                input_payload_row_bytes,
                rows,
                stream,
            )
        elif args.grouped_m1_parity_candidate:
            status = grouped_m1_parity_launch(
                ctypes.byref(buffers),
                input_payload_buffer,
                input_payload_row_bytes,
                rows,
                stream,
            )
        elif args.fused_fp8_output:
            status = prefill_fp8_launch(
                ctypes.byref(buffers),
                input_payload_buffer,
                input_payload_row_bytes,
                rows,
                device_buffer(output_fp8),
                output_fp8_row_bytes,
                stream,
            )
        else:
            if args.prefill_grid_x is None:
                status = prefill_launch(
                    ctypes.byref(buffers),
                    input_payload_buffer,
                    input_payload_row_bytes,
                    rows,
                    stream,
                )
            else:
                status = prefill_grid_launch(
                    ctypes.byref(buffers),
                    input_payload_buffer,
                    input_payload_row_bytes,
                    rows,
                    args.prefill_grid_x,
                    stream,
                )
        kernel_kind = (
            "grouped-wide-m1-parity"
            if args.grouped_wide_m1_parity_candidate
            else "grouped-m1-parity"
            if args.grouped_m1_parity_candidate
            else "m1-parity"
            if args.m1_parity_candidate
            else args.scenario
        )
        check_status(lib, status, f"launching native W4A16 {kernel_kind}")
        if args.copy_token_output:
            check_status(
                lib,
                copy_d2d(
                    device_buffer(joined_token_output[lane : lane + 1]),
                    device_buffer(lane_outputs[lane][0:1]),
                    HIDDEN * torch.bfloat16.itemsize,
                    stream,
                ),
                "copying combined BF16 token output",
            )
        if args.production_staging:
            check_status(
                lib,
                copy_d2d(
                    device_buffer(retained_output),
                    device_buffer(output),
                    output.numel() * output.element_size(),
                    stream,
                ),
                "copying retained BF16 output",
            )

    if args.verify_m1_parity:
        launch(buffers_sets[0])
        candidate = input_bf16[:rows].clone()
        # The retained local M1 oracle object may predate the target-SM export
        # guard and therefore address the coordinator-sized lock tail.  Clear
        # the full numerical-harness allocation after preserving the candidate;
        # the candidate's own poison-lock reset was already exercised above.
        locks.zero_()
        reference = torch.empty_like(candidate)
        for row_index in range(rows):
            row_buffers = SparkW4A16Buffers()
            ctypes.memmove(
                ctypes.byref(row_buffers),
                ctypes.byref(buffers_sets[0]),
                ctypes.sizeof(SparkW4A16Buffers),
            )
            row_buffers.input = device_buffer(input_bf16[row_index : row_index + 1])
            row_buffers.topk_weights = device_buffer(
                topk_weights[row_index : row_index + 1]
            )
            check_status(
                lib,
                decode_launch(
                    ctypes.byref(row_buffers),
                    device_buffer(input_payload[row_index : row_index + 1]),
                    input_payload_row_bytes,
                    device_buffer(topk_ids_rows[row_index]),
                    ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
                ),
                f"launching M1 parity reference row {row_index}",
            )
            reference[row_index].copy_(output[0])
        torch.cuda.synchronize()
        if not torch.equal(candidate, reference):
            mismatch = candidate != reference
            differing = torch.count_nonzero(mismatch).item()
            maximum = (candidate.float() - reference.float()).abs().max().item()
            mismatch_indices = (
                torch.nonzero(mismatch, as_tuple=False)[:8].cpu().tolist()
            )
            mismatch_values = [
                {
                    "index": index,
                    "candidate": float(candidate[tuple(index)].float().item()),
                    "reference": float(reference[tuple(index)].float().item()),
                }
                for index in mismatch_indices
            ]
            raise RuntimeError(
                "M1 parity candidate did not match repeated M1: "
                f"rows={rows} differing={differing} max_abs={maximum} "
                f"mismatches={mismatch_values}"
            )

    if args.concurrent_streams == 1:
        replay_stream = torch.cuda.Stream()
        replay_streams = [replay_stream] * len(buffers_sets)
    else:
        replay_streams = [torch.cuda.Stream() for _ in buffers_sets]
    if not args.cold_capture:
        for buffer_index, (buffers, replay_stream) in enumerate(
            zip(buffers_sets, replay_streams, strict=True)
        ):
            lane = buffer_index if args.concurrent_streams > 1 else 0
            with torch.cuda.stream(replay_stream):
                launch(buffers, lane)
        torch.cuda.synchronize()
    graphs: list[torch.cuda.CUDAGraph | CudaGraphCaptureInfo] = []
    if not args.eager:
        for buffer_index, (buffers, replay_stream) in enumerate(
            zip(buffers_sets, replay_streams, strict=True)
        ):
            lane = buffer_index if args.concurrent_streams > 1 else 0
            if args.native_graph_api:
                capture = CudaGraphCaptureInfo()
                with torch.cuda.stream(replay_stream):
                    stream = ctypes.c_void_p(replay_stream.cuda_stream)
                    check_status(lib, graph_begin(stream), "beginning native graph capture")
                    launch(buffers, lane)
                    check_status(
                        lib,
                        graph_end(stream, ctypes.byref(capture)),
                        "ending native graph capture",
                    )
                if capture.kernel_node_count < 1:
                    raise RuntimeError("native graph capture did not retain a kernel node")
                graphs.append(capture)
            else:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=replay_stream):
                    launch(buffers, lane)
                graphs.append(graph)
    graph_index = 0

    def replay() -> None:
        nonlocal graph_index
        if args.eager:
            launch(buffers_sets[0])
            return
        if args.native_graph_api:
            current_stream = torch.cuda.current_stream()
            replay_stream = replay_streams[0]
            replay_stream.wait_stream(current_stream)
            capture = graphs[0]
            assert isinstance(capture, CudaGraphCaptureInfo)
            check_status(
                lib,
                graph_launch(
                    capture.graph_exec,
                    ctypes.c_void_p(replay_stream.cuda_stream),
                ),
                "launching native CUDA graph",
            )
            current_stream.wait_stream(replay_stream)
            return
        if args.concurrent_streams == 1:
            graph = graphs[graph_index]
            assert isinstance(graph, torch.cuda.CUDAGraph)
            graph.replay()
            graph_index = (graph_index + 1) % len(graphs)
            return
        current_stream = torch.cuda.current_stream()
        for graph, replay_stream in zip(graphs, replay_streams, strict=True):
            replay_stream.wait_stream(current_stream)
            with torch.cuda.stream(replay_stream):
                graph.replay()
        for replay_stream in replay_streams:
            current_stream.wait_stream(replay_stream)

    if args.verify_exact_replays:
        replay()
        torch.cuda.synchronize()
        token_output = (
            output[:rows]
            if args.scenario == "decode"
            else output_fp8[:rows]
            if args.fused_fp8_output
            else input_bf16[:rows]
        )
        expected = token_output.clone()
        for replay_index in range(1, args.verify_exact_replays):
            replay()
            torch.cuda.synchronize()
            if not torch.equal(token_output, expected):
                differing = torch.count_nonzero(token_output != expected).item()
                maximum = (
                    (token_output.float() - expected.float()).abs().max().item()
                )
                raise RuntimeError(
                    "native W4A16 output changed on exact replay "
                    f"{replay_index + 1}/{args.verify_exact_replays}: "
                    f"differing={differing} max_abs={maximum}"
                )

    samples = measure(replay, args.warmup, args.iterations, args.repeats)
    if args.native_graph_api and not args.eager:
        torch.cuda.synchronize()
        for graph in graphs:
            assert isinstance(graph, CudaGraphCaptureInfo)
            check_status(
                lib,
                graph_exec_destroy(graph.graph_exec),
                "destroying native CUDA graph executable",
            )
            check_status(
                lib,
                graph_destroy(graph.graph),
                "destroying native CUDA graph",
            )
    requested_grid_x_text = __import__("os").environ.get(
        "GLMRT_B12X_SPARK_W4A16_DECODE_GRID_X", ""
    )
    requested_grid_x = int(requested_grid_x_text or DEFAULT_DECODE_GRID_X)
    effective_grid_x = (
        requested_grid_x
        if 0 < requested_grid_x <= MAX_DECODE_GRID_X
        else DEFAULT_DECODE_GRID_X
    )
    if args.scenario == "decode":
        active_experts = TOP_K
    elif args.empty_routes:
        active_experts = 0
    elif expert_route_counts is not None:
        active_experts = sum(count > 0 for count in expert_route_counts)
    else:
        active_experts = args.active_experts
    print(
        json.dumps(
            {
                "benchmark": f"b12x_spark_w4a16_native_{args.scenario}",
                "kernel": (
                    "grouped-wide-fixed-order-m1-parity"
                    if args.grouped_wide_m1_parity_candidate
                    else "grouped-block8-m1-parity"
                    if args.grouped_m1_parity_candidate
                    else
                    "ordered-direct-topk-m1-parity"
                    if args.m1_parity_candidate
                    else "atomic-fused-sum-m1"
                    if args.m1_fused_sum_candidate
                    else args.scenario
                ),
                "scenario": args.scenario,
                "rows": rows,
                "capacity_rows": capacity_rows,
                "routes": 0 if args.empty_routes else rows * args.routes_per_row,
                "routes_per_row": 0 if args.empty_routes else args.routes_per_row,
                "logical_route_slots": rows * TOP_K,
                "active_experts": active_experts,
                "expert_route_counts": expert_route_counts,
                "empty_routes": args.empty_routes,
                "input_encoding": "nvfp4_dequantized_to_bf16",
                "input_memory": args.input_memory,
                "production_staging": args.production_staging,
                "cold_capture": args.cold_capture,
                "graph_api": (
                    "eager"
                    if args.eager
                    else "native"
                    if args.native_graph_api
                    else "torch"
                ),
                "poison_locks": args.poison_locks,
                "verified_exact_replays": args.verify_exact_replays,
                "weight_sets": args.weight_sets,
                "weight_experts": weight_experts,
                "concurrent_streams": args.concurrent_streams,
                "aggregate_rows": rows * args.concurrent_streams,
                "concurrent_inputs": (
                    "distinct-row-payloads-and-routes"
                    if args.scenario == "decode" and args.concurrent_streams > 1
                    else "single"
                ),
                "concurrent_route_overlap": args.concurrent_route_overlap,
                "copy_token_output": args.copy_token_output,
                "weight_memory": args.weight_memory,
                "weight_layout": "packed",
                "effective_grid_x": effective_grid_x if args.scenario == "decode" else None,
                "requested_grid_x": requested_grid_x if args.scenario == "decode" else None,
                "prefill_grid_x": args.prefill_grid_x,
                "fused_fp8_output": args.fused_fp8_output,
                "weight_payload": "random",
                "median_ms": statistics.median(samples),
                "min_ms": min(samples),
                "samples_ms": samples,
            },
            sort_keys=True,
        )
    )
    if args.input_memory == "mapped":
        for host_buffer in (mapped_topk_weights, mapped_topk_ids):
            if host_buffer.ptr:
                check_status(
                    lib,
                    lib.glmrt_free_host_buffer(ctypes.byref(host_buffer)),
                    "freeing mapped route metadata",
                )
        check_status(
            lib,
            lib.glmrt_free_host_buffer(ctypes.byref(mapped_input)),
            "freeing mapped input payload",
        )
    for allocation in managed_weight_allocations:
        check_status(
            lib,
            lib.glmrt_free_device_buffer(ctypes.byref(allocation)),
            "freeing managed packed weights",
        )


if __name__ == "__main__":
    main()
