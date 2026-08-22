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
E4M3_ONE = 0x38
MAX_ROWS = 256
MAX_PACKED_ROUTE_SLOTS = 20_224
MAX_ROUTE_BLOCKS = 422
SCRATCH_ELEMENTS = 1_572_864
LOCK_ELEMENTS = 194


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


def device_buffer(
    tensor: torch.Tensor, *, advertised_bytes: int | None = None
) -> DeviceBuffer:
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
    error = ctypes.create_string_buffer(512)
    lib.glmrt_last_error(error, len(error))
    raise RuntimeError(
        f"{action} failed with status {status}: {error.value.decode()}"
    )


def measure(
    graph: torch.cuda.CUDAGraph | list[torch.cuda.CUDAGraph],
    stream: torch.cuda.Stream,
    warmup: int,
    iterations: int,
    repeats: int,
) -> list[float]:
    graphs = graph if isinstance(graph, list) else [graph]
    with torch.cuda.stream(stream):
        for iteration in range(warmup):
            graphs[iteration % len(graphs)].replay()
    stream.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record(stream)
            for iteration in range(iterations):
                graphs[iteration % len(graphs)].replay()
            end.record(stream)
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return samples


def route_metadata(
    rows: int, capacity_rows: int, block_size: int, direct_topk: bool
) -> tuple[list[int], list[int], int]:
    if direct_topk:
        return [0] * rows, [0] * rows, rows
    padded_rows = ((rows + block_size - 1) // block_size) * block_size
    packed_routes = list(range(rows)) + [rows] * (padded_rows - rows)
    block_experts = [0] * (padded_rows // block_size)
    if len(packed_routes) > capacity_rows * 8:
        raise RuntimeError("route metadata exceeds the production top-1 capacity")
    return packed_routes, block_experts, padded_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tune one production-specialized SparkInfer top-1 W4A16 kernel "
            "without allocating all expert weights."
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
    parser.add_argument("--fc1-tile-n", type=int)
    parser.add_argument("--fc1-tile-k", type=int)
    parser.add_argument("--fc2-tile-n", type=int)
    parser.add_argument("--fc2-tile-k", type=int)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--weight-sets",
        type=int,
        choices=range(1, 17),
        default=1,
        help="round-robin this many physical experts in native measurements",
    )
    parser.add_argument(
        "--native-only",
        action="store_true",
        help="skip CuTe compilation and measure only current/candidate native grids",
    )
    args = parser.parse_args()
    for prefix in ("fc1", "fc2"):
        tile_n = getattr(args, f"{prefix}_tile_n")
        tile_k = getattr(args, f"{prefix}_tile_k")
        if (tile_n is None) != (tile_k is None):
            parser.error(f"--{prefix}-tile-n and --{prefix}-tile-k must be paired")
    if min(args.grids) < 1:
        parser.error("grid sizes must be positive")
    if min(args.warmup, args.iterations, args.repeats) < 1:
        parser.error("warmup, iterations, and repeats must be positive")
    if args.native_only and any(
        getattr(args, name) is not None
        for name in ("fc1_tile_n", "fc1_tile_k", "fc2_tile_n", "fc2_tile_k")
    ):
        parser.error("tile overrides cannot be used with --native-only")
    if max(args.grids) > 48:
        parser.error("SM121 top-1 grids cannot exceed the cooperative cap 48")

    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    sms = int(properties.multi_processor_count)
    max_shared_mem = int(properties.shared_memory_per_block_optin)

    # Reserve the bounded synthetic weights before importing compiler modules.
    # This avoids intermittent context-start failures on nearly-full Spark UMA.
    w13 = torch.full(
        (args.weight_sets, 2 * INTERMEDIATE, HIDDEN // 2),
        0x71,
        dtype=torch.uint8,
        device=device,
    )
    w2 = torch.full(
        (args.weight_sets, HIDDEN, INTERMEDIATE // 2),
        0x83,
        dtype=torch.uint8,
        device=device,
    )
    w13_scale = torch.full(
        (args.weight_sets, 2 * INTERMEDIATE, HIDDEN // 16),
        E4M3_ONE,
        dtype=torch.uint8,
        device=device,
    ).view(torch.float8_e4m3fn)
    w2_scale = torch.full(
        (args.weight_sets, HIDDEN, INTERMEDIATE // 16),
        E4M3_ONE,
        dtype=torch.uint8,
        device=device,
    ).view(torch.float8_e4m3fn)
    global_scale = torch.ones(args.weight_sets, dtype=torch.float32, device=device)

    from b12x.moe._shared.kernels.w4a16.prepare import (
        prepare_w4a16_modelopt_nvfp4_weights,
    )

    prepared = prepare_w4a16_modelopt_nvfp4_weights(
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

    rows = args.rows
    capacity_rows = rows
    block_size = 8
    direct_topk = rows <= 8
    tc_decode_fused_sum = direct_topk
    max_m_blocks = capacity_rows
    fused = None
    if not args.native_only:
        from b12x.moe._shared.kernels.w4a16 import kernel as w4a16_kernel
        from b12x.moe._shared.kernels.w4a16.kernel import (
            compile_w4a16_fused_moe,
        )

        original_init = w4a16_kernel.W4A16FusedMoeKernel.__init__

        def override_tiles(self, *init_args, **init_kwargs):
            for prefix in ("fc1", "fc2"):
                tile_n = getattr(args, f"{prefix}_tile_n")
                tile_k = getattr(args, f"{prefix}_tile_k")
                if tile_n is not None:
                    init_kwargs[f"{prefix}_tile_n"] = tile_n
                    init_kwargs[f"{prefix}_tile_k"] = tile_k
            original_init(self, *init_args, **init_kwargs)

        w4a16_kernel.W4A16FusedMoeKernel.__init__ = override_tiles
        fused = compile_w4a16_fused_moe(
            size_m=capacity_rows,
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            num_experts=EXPERTS,
            top_k=1,
            activation="silu",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            moe_block_size=block_size,
            max_m_blocks=max_m_blocks,
            element_dtype="bf16",
            fast_math=True,
            sms=sms,
            max_shared_mem=max_shared_mem,
            weight_layout="packed",
            scale_format="e4m3_k16",
            direct_topk_routes=direct_topk,
            tc_decode_fused_sum=tc_decode_fused_sum,
        )

    hidden = torch.full(
        (capacity_rows, HIDDEN),
        0.25,
        dtype=torch.bfloat16,
        device=device,
    )
    native_fc1 = torch.empty(
        capacity_rows * 2 * INTERMEDIATE, dtype=torch.bfloat16, device=device
    )
    native_activated = torch.empty(
        capacity_rows * INTERMEDIATE, dtype=torch.bfloat16, device=device
    )
    native_output = torch.empty(
        (capacity_rows, HIDDEN), dtype=torch.bfloat16, device=device
    )
    native_fc1_scratch = torch.empty(
        SCRATCH_ELEMENTS, dtype=torch.float32, device=device
    )
    native_fc2_scratch = torch.empty_like(native_fc1_scratch)
    native_locks = torch.zeros(LOCK_ELEMENTS, dtype=torch.int32, device=device)
    native_packed_routes = torch.zeros(
        MAX_PACKED_ROUTE_SLOTS, dtype=torch.int32, device=device
    )
    native_block_experts = torch.zeros(
        MAX_ROUTE_BLOCKS, dtype=torch.int32, device=device
    )
    native_packed_route_count = torch.zeros(1, dtype=torch.int32, device=device)
    native_topk_weights = torch.zeros(
        (capacity_rows, 1), dtype=torch.float32, device=device
    )
    native_buffers = SparkW4A16Buffers(
        device_buffer(hidden),
        device_buffer(
            prepared.w13,
            advertised_bytes=EXPERTS * 2 * INTERMEDIATE * HIDDEN // 2,
        ),
        device_buffer(
            prepared.w2,
            advertised_bytes=EXPERTS * HIDDEN * INTERMEDIATE // 2,
        ),
        device_buffer(native_fc1),
        device_buffer(native_activated),
        device_buffer(native_output),
        device_buffer(
            prepared.w13_scale,
            advertised_bytes=EXPERTS * 2 * INTERMEDIATE * HIDDEN // 16,
        ),
        device_buffer(
            prepared.w2_scale,
            advertised_bytes=EXPERTS * HIDDEN * INTERMEDIATE // 16,
        ),
        device_buffer(
            prepared.w13_global_scale, advertised_bytes=EXPERTS * 4
        ),
        device_buffer(
            prepared.w2_global_scale, advertised_bytes=EXPERTS * 4
        ),
        device_buffer(native_packed_routes),
        device_buffer(native_block_experts),
        device_buffer(native_packed_route_count),
        device_buffer(native_topk_weights),
        device_buffer(native_fc1_scratch),
        device_buffer(native_fc2_scratch),
        device_buffer(native_locks),
    )

    native_lib = ctypes.CDLL(str(args.native_lib.resolve()))
    native_lib.glmrt_last_error.argtypes = (ctypes.c_char_p, ctypes.c_size_t)
    native_lib.glmrt_last_error.restype = ctypes.c_int
    native_lib.glmrt_cuda_b12x_spark_w4a16_top1_async.argtypes = (
        ctypes.POINTER(SparkW4A16Buffers),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_void_p,
    )
    native_lib.glmrt_cuda_b12x_spark_w4a16_top1_async.restype = ctypes.c_int
    native_lib.glmrt_cuda_b12x_spark_w4a16_top1_grid_candidate_async.argtypes = (
        ctypes.POINTER(SparkW4A16Buffers),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_void_p,
    )
    native_lib.glmrt_cuda_b12x_spark_w4a16_top1_grid_candidate_async.restype = (
        ctypes.c_int
    )
    native_stream = torch.cuda.Stream()

    def launch_native(expert_id: int) -> None:
        check_status(
            native_lib,
            native_lib.glmrt_cuda_b12x_spark_w4a16_top1_async(
                ctypes.byref(native_buffers),
                rows,
                capacity_rows,
                expert_id,
                ctypes.c_void_p(native_stream.cuda_stream),
            ),
            "native top-1 launch",
        )

    with torch.cuda.stream(native_stream):
        launch_native(0)
    native_stream.synchronize()
    expected = native_output.clone()
    native_graphs = []
    for expert_id in range(args.weight_sets):
        native_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(native_graph, stream=native_stream):
            launch_native(expert_id)
        native_graphs.append(native_graph)
    native_samples = measure(
        native_graphs,
        native_stream,
        args.warmup,
        args.iterations,
        args.repeats,
    )
    print(
        json.dumps(
            {
                "benchmark": "b12x_spark_top1_native",
                "grid_x": {
                    1: 8,
                    2: 16,
                    4: 32,
                    8: 32,
                    16: 48,
                    32: 48,
                    64: 48,
                    128: 48,
                    256: 48,
                }[capacity_rows],
                "median_ms": statistics.median(native_samples),
                "min_ms": min(native_samples),
                "rows": rows,
                "samples_ms": native_samples,
                "weight_sets": args.weight_sets,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    def launch_native_grid_candidate(grid: int, expert_id: int) -> None:
        check_status(
            native_lib,
            native_lib.glmrt_cuda_b12x_spark_w4a16_top1_grid_candidate_async(
                ctypes.byref(native_buffers),
                rows,
                capacity_rows,
                expert_id,
                grid,
                ctypes.c_void_p(native_stream.cuda_stream),
            ),
            "native top-1 grid candidate launch",
        )

    candidate_stream = None
    launch_candidate = None
    candidate_output = None
    if not args.native_only:
        from b12x.moe._shared.kernels.w4a16.kernel import (
            _cutlass_element_dtype,
            cuda,
            cute,
            make_ptr,
        )
        import cutlass

        assert fused is not None
        candidate_stream = torch.cuda.Stream()
        candidate_fc1 = torch.empty_like(native_fc1)
        candidate_activated = torch.empty_like(native_activated)
        candidate_output = torch.empty_like(native_output)
        candidate_fc1_scratch = torch.empty_like(native_fc1_scratch)
        candidate_fc2_scratch = torch.empty_like(native_fc2_scratch)
        candidate_locks = torch.zeros_like(native_locks)
        packed_route_values, block_expert_values, packed_count = route_metadata(
            rows, capacity_rows, block_size, direct_topk
        )
        packed_routes = torch.full(
            (MAX_PACKED_ROUTE_SLOTS,), rows, dtype=torch.int32, device=device
        )
        packed_routes[: len(packed_route_values)] = torch.tensor(
            packed_route_values, dtype=torch.int32, device=device
        )
        block_experts = torch.zeros(
            MAX_ROUTE_BLOCKS, dtype=torch.int32, device=device
        )
        block_experts[: len(block_expert_values)] = torch.tensor(
            block_expert_values, dtype=torch.int32, device=device
        )
        packed_route_count = torch.tensor(
            [packed_count], dtype=torch.int32, device=device
        )
        topk_weights = torch.ones(
            (capacity_rows, 1), dtype=torch.float32, device=device
        )
        rotation_placeholder = torch.zeros(1, dtype=torch.float16, device=device)

        def launch_candidate_impl(grid: int) -> None:
            # Resolve the stream under capture. Caching the pointer before capture
            # silently records an empty graph with current CuTe/PyTorch builds.
            stream = torch.cuda.current_stream()
            candidate_locks.zero_()
            hidden_pointer = make_ptr(
                _cutlass_element_dtype("bf16"),
                hidden.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            fused.compiled(
                hidden_pointer,
                hidden_pointer,
                hidden_pointer,
                prepared.w13.view(torch.int32).view(-1),
                prepared.w2.view(torch.int32).view(-1),
                candidate_fc1,
                candidate_activated,
                candidate_output.view(-1),
                prepared.w13_scale.view(torch.uint8).view(torch.int32).view(-1),
                prepared.w2_scale.view(torch.uint8).view(torch.int32).view(-1),
                prepared.w13_global_scale,
                prepared.w2_global_scale,
                packed_routes,
                block_experts,
                packed_route_count,
                prepared.w13_global_scale,
                0,
                make_ptr(
                    cutlass.Float32,
                    topk_weights.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                ),
                candidate_fc1_scratch,
                candidate_fc2_scratch,
                candidate_locks,
                rotation_placeholder,
                rotation_placeholder,
                rotation_placeholder,
                rows,
                grid,
                cuda.CUstream(stream.cuda_stream),
            )

        launch_candidate = launch_candidate_impl

    results = []
    native_candidate_results = []
    for grid in args.grids:
        with torch.cuda.stream(native_stream):
            launch_native_grid_candidate(grid, 0)
        native_stream.synchronize()
        native_candidate_actual = native_output.clone()
        native_candidate_difference = (
            native_candidate_actual.float() - expected.float()
        ).abs()
        native_candidate_graphs = []
        for expert_id in range(args.weight_sets):
            native_candidate_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(native_candidate_graph, stream=native_stream):
                launch_native_grid_candidate(grid, expert_id)
            native_candidate_graphs.append(native_candidate_graph)
        native_candidate_samples = measure(
            native_candidate_graphs,
            native_stream,
            args.warmup,
            args.iterations,
            args.repeats,
        )
        native_candidate_result = {
            "benchmark": "b12x_spark_top1_native_grid_candidate",
            "bitwise_equal": bool(torch.equal(native_candidate_actual, expected)),
            "grid_x": grid,
            "max_abs_error": float(native_candidate_difference.max()),
            "median_ms": statistics.median(native_candidate_samples),
            "min_ms": min(native_candidate_samples),
            "rows": rows,
            "samples_ms": native_candidate_samples,
            "weight_sets": args.weight_sets,
        }
        native_candidate_results.append(native_candidate_result)
        print(json.dumps(native_candidate_result, sort_keys=True), flush=True)

        if args.native_only:
            continue
        assert candidate_stream is not None
        assert launch_candidate is not None
        assert candidate_output is not None
        assert fused is not None
        with torch.cuda.stream(candidate_stream):
            launch_candidate(grid)
        candidate_stream.synchronize()
        actual = candidate_output.clone()
        difference = (actual.float() - expected.float()).abs()
        expected_norm = torch.linalg.vector_norm(expected.float())
        difference_norm = torch.linalg.vector_norm(difference)
        validation = {
            "bitwise_equal": bool(torch.equal(actual, expected)),
            "cosine_similarity": float(
                torch.nn.functional.cosine_similarity(
                    actual.float().reshape(1, -1),
                    expected.float().reshape(1, -1),
                )
            ),
            "l2_relative_error": float(difference_norm / expected_norm),
            "max_abs_error": float(difference.max()),
        }
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=candidate_stream):
            launch_candidate(grid)
        samples = measure(
            graph,
            candidate_stream,
            args.warmup,
            args.iterations,
            args.repeats,
        )
        result = {
            "benchmark": "b12x_spark_top1_candidate",
            "blocks_per_sm": int(fused.blocks_per_sm),
            "direct_topk": direct_topk,
            "fc1_tile_k": int(fused.fc1_tile_k),
            "fc1_tile_n": int(fused.fc1_tile_n),
            "fc2_tile_k": int(fused.fc2_tile_k),
            "fc2_tile_n": int(fused.fc2_tile_n),
            "grid_x": grid,
            "median_ms": statistics.median(samples),
            "min_ms": min(samples),
            "rows": rows,
            "samples_ms": samples,
            "tc_decode_fused_sum": tc_decode_fused_sum,
            "weight_sets": 1,
            **validation,
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    print(
        json.dumps(
            {
                "benchmark": "b12x_spark_top1_tune_summary",
                "best": (
                    min(results, key=lambda item: item["median_ms"])
                    if results
                    else None
                ),
                "best_native_grid_candidate": min(
                    native_candidate_results,
                    key=lambda item: item["median_ms"],
                ),
                "native_median_ms": statistics.median(native_samples),
                "rows": rows,
                "sms": sms,
                "weight_sets": args.weight_sets,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
