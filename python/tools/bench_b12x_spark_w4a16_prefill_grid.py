#!/usr/bin/env python3
from __future__ import annotations

import _pinned_sparkinfer  # noqa: F401

import argparse
import json
import random
import statistics
from pathlib import Path

import torch


HIDDEN = 6144
INTERMEDIATE = 512
EXPERTS = 256
TOP_K = 8
E4M3_ONE = 0x38


def route_metadata(
    rows: int,
    active_experts: int,
    block_size: int,
    route_counts: list[int] | None = None,
    seed: int = 17,
) -> tuple[list[int], list[int]]:
    routes_by_expert: list[list[int]] = [[] for _ in range(EXPERTS)]
    if route_counts is None:
        for route_index in range(rows * TOP_K):
            routes_by_expert[route_index % active_experts].append(route_index)
    else:
        route_indices = list(range(rows * TOP_K))
        random.Random(seed).shuffle(route_indices)
        offset = 0
        for expert_id, count in enumerate(route_counts):
            routes_by_expert[expert_id].extend(route_indices[offset : offset + count])
            offset += count

    packed_routes: list[int] = []
    block_experts: list[int] = []
    sentinel = rows * TOP_K
    for expert_id, routes in enumerate(routes_by_expert):
        for start in range(0, len(routes), block_size):
            block = routes[start : start + block_size]
            packed_routes.extend(block)
            packed_routes.extend([sentinel] * (block_size - len(block)))
            block_experts.append(expert_id)
    return packed_routes, block_experts


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep production-style SparkInfer W4A16 route blocks, tiles, and "
            "persistent grids."
        )
    )
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--active-experts", type=int, default=EXPERTS)
    parser.add_argument(
        "--weight-experts",
        type=int,
        default=EXPERTS,
        help="allocate only this many leading expert weights for bounded sweeps",
    )
    parser.add_argument(
        "--weight-sets",
        type=int,
        default=1,
        help="rotate graphs across independent weight sets to defeat hot-weight caching",
    )
    parser.add_argument("--block-sizes", type=int, nargs="+", default=(16, 32))
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=(48, 64, 80, 96))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--route-counts-json",
        type=Path,
        help=(
            "optional JSON array with one route count per expert; counts must "
            "sum to rows * top-k and replace the balanced synthetic routes"
        ),
    )
    parser.add_argument(
        "--fused-topk-sum",
        action="store_true",
        help=(
            "benchmark SparkInfer's BF16 atomic top-k epilogue with route-packed "
            "metadata (an unsupported diagnostic extension of the decode path)"
        ),
    )
    parser.add_argument(
        "--tc-prefill-tiles",
        action="store_true",
        help="apply SparkInfer's wider TC-decode FC2 tile selection without fused summation",
    )
    parser.add_argument("--fc1-tile-n", type=int, default=0)
    parser.add_argument("--fc1-tile-k", type=int, default=0)
    parser.add_argument("--fc2-tile-n", type=int, default=0)
    parser.add_argument("--fc2-tile-k", type=int, default=0)
    parser.add_argument(
        "--pipeline-stages",
        type=int,
        choices=(2, 3, 4),
        default=4,
        help="override SparkInfer's compile-time async staging depth",
    )
    parser.add_argument(
        "--target-blocks-per-sm",
        type=int,
        choices=(0, 1, 2, 3, 4),
        default=0,
        help=(
            "override the generated launch bound; values above the natural "
            "residency may force register spilling"
        ),
    )
    parser.add_argument(
        "--direct-topk-routes",
        action="store_true",
        help="use the production small-M direct top-k route ABI",
    )
    parser.add_argument(
        "--decomposed",
        action="store_true",
        help=(
            "also benchmark separate FC1, activation, and FC2 kernels against "
            "the fused persistent kernel"
        ),
    )
    parser.add_argument("--decomposed-fc1-tile-n", type=int, default=0)
    parser.add_argument("--decomposed-fc1-tile-k", type=int, default=0)
    parser.add_argument("--decomposed-fc2-tile-n", type=int, default=0)
    parser.add_argument("--decomposed-fc2-tile-k", type=int, default=0)
    parser.add_argument("--decomposed-fc1-grid", type=int, default=0)
    parser.add_argument("--decomposed-fc2-grid", type=int, default=0)
    args = parser.parse_args()
    if args.rows < 1 or min(
        args.warmup, args.iterations, args.repeats, args.weight_sets
    ) < 1:
        parser.error(
            "rows, warmup, iterations, repeats, and weight-sets must be positive"
        )
    if not TOP_K <= args.active_experts <= EXPERTS:
        parser.error(f"active-experts must be in {TOP_K}..={EXPERTS}")
    if not args.active_experts <= args.weight_experts <= EXPERTS:
        parser.error(
            "weight-experts must be between active-experts and " f"{EXPERTS}"
        )
    if args.direct_topk_routes and args.rows > 6:
        parser.error("direct-topk-routes requires rows <= 6")
    if args.direct_topk_routes and (args.fused_topk_sum or args.tc_prefill_tiles):
        parser.error(
            "direct-topk-routes benchmarks the production separate-route output"
        )
    if args.decomposed and args.direct_topk_routes:
        parser.error("--decomposed requires route-packed packed weights")
    if any(block not in (8, 16, 32, 48, 64) for block in args.block_sizes):
        parser.error("block sizes must be selected from 8, 16, 32, 48, 64")
    if any(grid < 1 for grid in args.grid_sizes):
        parser.error("grid sizes must be positive")
    if bool(args.fc1_tile_n) != bool(args.fc1_tile_k):
        parser.error("fc1-tile-n and fc1-tile-k must be specified together")
    if bool(args.fc2_tile_n) != bool(args.fc2_tile_k):
        parser.error("fc2-tile-n and fc2-tile-k must be specified together")
    if bool(args.decomposed_fc1_tile_n) != bool(args.decomposed_fc1_tile_k):
        parser.error(
            "decomposed-fc1-tile-n and decomposed-fc1-tile-k "
            "must be specified together"
        )
    if bool(args.decomposed_fc2_tile_n) != bool(args.decomposed_fc2_tile_k):
        parser.error(
            "decomposed-fc2-tile-n and decomposed-fc2-tile-k "
            "must be specified together"
        )
    route_counts = None
    if args.route_counts_json is not None:
        route_counts = json.loads(args.route_counts_json.read_text(encoding="utf-8"))
        if (
            not isinstance(route_counts, list)
            or len(route_counts) != EXPERTS
            or any(not isinstance(count, int) or count < 0 for count in route_counts)
        ):
            parser.error(f"route-counts-json must contain {EXPERTS} non-negative integers")
        if sum(route_counts) != args.rows * TOP_K:
            parser.error(
                "route-counts-json counts must sum to "
                f"{args.rows * TOP_K}, got {sum(route_counts)}"
            )
        args.active_experts = sum(count > 0 for count in route_counts)
        if args.direct_topk_routes:
            parser.error("route-counts-json is incompatible with direct-topk-routes")
        if args.active_experts > args.weight_experts:
            parser.error("route-counts-json activates an unallocated expert")

    from b12x.moe._shared.kernels.w4a16.host import (
        max_packed_route_slots,
        packed_gemm_scratch_elements,
    )
    from b12x.moe._shared.kernels.w4a16 import kernel as w4a16_kernel
    from b12x.moe._shared.kernels.w4a16.kernel import (
        _cutlass_element_dtype,
        compile_w4a16_activation,
        compile_w4a16_fused_moe,
        compile_w4a16_gemm,
        cuda,
        cute,
        make_ptr,
    )
    from b12x.moe._shared.kernels.w4a16.prepare import (
        prepare_w4a16_modelopt_nvfp4_weights,
    )

    w4a16_kernel._STAGES = args.pipeline_stages
    if args.target_blocks_per_sm:
        original_gemm_init = w4a16_kernel.W4A16GemmKernel.__init__

        def override_gemm_residency(self, *init_args, **init_kwargs):
            original_gemm_init(self, *init_args, **init_kwargs)
            self.blocks_per_sm = args.target_blocks_per_sm

        w4a16_kernel.W4A16GemmKernel.__init__ = override_gemm_residency
    torch.manual_seed(args.seed)
    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    sms = int(properties.multi_processor_count)
    max_shared_mem = int(properties.shared_memory_per_block_optin)
    max_grid = (
        sms
        if args.direct_topk_routes
        else sms * (args.target_blocks_per_sm or 2)
    )
    if any(grid > max_grid for grid in args.grid_sizes):
        parser.error(f"grid sizes cannot exceed the cooperative cap {max_grid}")
    if any(
        grid < 0 or grid > sms * 2
        for grid in (args.decomposed_fc1_grid, args.decomposed_fc2_grid)
    ):
        parser.error(f"decomposed grids must be in 1..={sms * 2} when specified")

    reference_fused = None
    if args.direct_topk_routes:
        reference_fused = compile_w4a16_fused_moe(
            size_m=args.rows,
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            num_experts=EXPERTS,
            top_k=TOP_K,
            activation="silu",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            moe_block_size=8,
            max_m_blocks=args.rows * TOP_K,
            element_dtype="bf16",
            fast_math=True,
            sms=sms,
            max_shared_mem=max_shared_mem,
            weight_layout="packed",
            scale_format="e4m3_k16",
            direct_topk_routes=True,
            tc_decode_fused_sum=False,
        )

    if (
        args.fused_topk_sum
        or args.tc_prefill_tiles
        or args.fc1_tile_n
        or args.fc2_tile_n
    ):
        original_init = w4a16_kernel.W4A16FusedMoeKernel.__init__

        def allow_route_packed_fused_sum(self, *init_args, **init_kwargs):
            requested = bool(init_kwargs.get("tc_decode_fused_sum"))
            route_packed = not bool(init_kwargs.get("direct_topk_routes"))
            if requested and route_packed:
                init_kwargs["tc_decode_fused_sum"] = False
            if args.fc1_tile_n:
                init_kwargs["fc1_tile_n"] = args.fc1_tile_n
                init_kwargs["fc1_tile_k"] = args.fc1_tile_k
            if args.fc2_tile_n:
                init_kwargs["fc2_tile_n"] = args.fc2_tile_n
                init_kwargs["fc2_tile_k"] = args.fc2_tile_k
            original_init(self, *init_args, **init_kwargs)
            if requested and route_packed and args.fused_topk_sum:
                self.tc_decode_fused_sum = True
                self.fc2.fused_topk_sum = True
                self.fc2.fused_sum_topk = int(init_kwargs["top_k"])

        w4a16_kernel.W4A16FusedMoeKernel.__init__ = allow_route_packed_fused_sum

    prepared_sets = []
    for _ in range(args.weight_sets):
        w13 = torch.randint(
            0,
            256,
            (args.weight_experts, 2 * INTERMEDIATE, HIDDEN // 2),
            dtype=torch.uint8,
            device=device,
        )
        w2 = torch.randint(
            0,
            256,
            (args.weight_experts, HIDDEN, INTERMEDIATE // 2),
            dtype=torch.uint8,
            device=device,
        )
        w13_scale = torch.full(
            (args.weight_experts, 2 * INTERMEDIATE, HIDDEN // 16),
            E4M3_ONE,
            dtype=torch.uint8,
            device=device,
        ).view(torch.float8_e4m3fn)
        w2_scale = torch.full(
            (args.weight_experts, HIDDEN, INTERMEDIATE // 16),
            E4M3_ONE,
            dtype=torch.uint8,
            device=device,
        ).view(torch.float8_e4m3fn)
        global_scale = torch.ones(
            args.weight_experts, dtype=torch.float32, device=device
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
        prepared_sets.append(prepared)

    routed_rows = args.rows * TOP_K
    hidden = torch.randn((args.rows, HIDDEN), dtype=torch.bfloat16, device=device)
    fc1_out = torch.empty(
        routed_rows * max(2 * INTERMEDIATE, HIDDEN),
        dtype=torch.bfloat16,
        device=device,
    )
    activated = torch.empty(
        routed_rows * INTERMEDIATE,
        dtype=torch.bfloat16,
        device=device,
    )
    topk_weights = torch.full(
        (args.rows, TOP_K),
        1.0 / TOP_K,
        dtype=torch.float32,
        device=device,
    )
    workspace = torch.zeros(sms * 4 + 2, dtype=torch.int32, device=device)
    stream = torch.cuda.Stream()
    results = []
    reference_output = None
    block_sizes = list(args.block_sizes)
    if args.direct_topk_routes:
        block_sizes = [8, *(block for block in block_sizes if block != 8)]

    for block_size in block_sizes:
        if args.direct_topk_routes:
            route_capacity = routed_rows
            scratch_route_slots = routed_rows * block_size
            max_m_blocks = routed_rows
            packed_route_values = [
                route_index % args.active_experts
                for route_index in range(routed_rows)
            ]
            block_expert_values = [0] * routed_rows
        else:
            route_capacity = max_packed_route_slots(
                routed_rows,
                block_size,
                EXPERTS,
            )
            scratch_route_slots = route_capacity
            max_m_blocks = (route_capacity + block_size - 1) // block_size
            packed_route_values, block_expert_values = route_metadata(
                args.rows,
                args.active_experts,
                block_size,
                route_counts,
                args.seed,
            )
        packed_routes = torch.full(
            (route_capacity,),
            routed_rows,
            dtype=torch.int32,
            device=device,
        )
        packed_routes[: len(packed_route_values)] = torch.tensor(
            packed_route_values,
            dtype=torch.int32,
            device=device,
        )
        block_experts = torch.zeros(
            max_m_blocks,
            dtype=torch.int32,
            device=device,
        )
        block_experts[: len(block_expert_values)] = torch.tensor(
            block_expert_values,
            dtype=torch.int32,
            device=device,
        )
        packed_route_count = torch.tensor(
            [len(packed_route_values)],
            dtype=torch.int32,
            device=device,
        )
        scratch_elements = max(
            packed_gemm_scratch_elements(
                size_n=2 * INTERMEDIATE,
                route_slots=scratch_route_slots,
                moe_block_size=block_size,
                sms=sms,
            ),
            packed_gemm_scratch_elements(
                size_n=HIDDEN,
                route_slots=scratch_route_slots,
                moe_block_size=block_size,
                sms=sms,
            ),
        )
        fc1_scratch = torch.empty(scratch_elements, dtype=torch.float32, device=device)
        fc2_scratch = torch.empty(scratch_elements, dtype=torch.float32, device=device)
        fused = compile_w4a16_fused_moe(
            size_m=args.rows,
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            num_experts=EXPERTS,
            top_k=TOP_K,
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
            direct_topk_routes=args.direct_topk_routes,
            tc_decode_fused_sum=args.fused_topk_sum or args.tc_prefill_tiles,
        )
        decomposed_fc1 = None
        decomposed_fc2 = None
        decomposed_activation = None
        decomposed_fc1_locks = None
        decomposed_fc2_locks = None
        if args.decomposed:
            if args.fused_topk_sum:
                parser.error("--decomposed is incompatible with --fused-topk-sum")
            decomposed_fc1 = compile_w4a16_gemm(
                size_m=args.rows,
                size_n=2 * INTERMEDIATE,
                size_k=HIDDEN,
                num_experts=EXPERTS,
                top_k=TOP_K,
                mul_topk_weights=False,
                tile_n=int(
                    args.decomposed_fc1_tile_n
                    or args.fc1_tile_n
                    or fused.fc1_tile_n
                ),
                tile_k=int(
                    args.decomposed_fc1_tile_k
                    or args.fc1_tile_k
                    or fused.fc1_tile_k
                ),
                moe_block_size=block_size,
                max_m_blocks=max_m_blocks,
                element_dtype="bf16",
                scale_format="e4m3_k16",
            )
            decomposed_activation = compile_w4a16_activation(
                rows=routed_rows,
                intermediate_size=INTERMEDIATE,
                activation="silu",
                element_dtype="bf16",
                fast_math=True,
            )
            decomposed_fc2 = compile_w4a16_gemm(
                size_m=routed_rows,
                size_n=HIDDEN,
                size_k=INTERMEDIATE,
                num_experts=EXPERTS,
                top_k=1,
                mul_topk_weights=True,
                tile_n=int(
                    args.decomposed_fc2_tile_n
                    or args.fc2_tile_n
                    or fused.fc2_tile_n
                ),
                tile_k=int(
                    args.decomposed_fc2_tile_k
                    or args.fc2_tile_k
                    or fused.fc2_tile_k
                ),
                moe_block_size=block_size,
                max_m_blocks=max_m_blocks,
                element_dtype="bf16",
                scale_format="e4m3_k16",
            )
            decomposed_fc1_locks = torch.zeros(
                sms * 4, dtype=torch.int32, device=device
            )
            decomposed_fc2_locks = torch.zeros_like(decomposed_fc1_locks)

        rotation_placeholder = torch.zeros(1, dtype=torch.float16, device=device)

        def bf16_ptr(tensor: torch.Tensor):
            return make_ptr(
                _cutlass_element_dtype("bf16"),
                tensor.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )

        def launch_compiled(compiled, grid: int, prepared) -> None:
            workspace.zero_()
            hidden_ptr = bf16_ptr(hidden)
            compiled.compiled(
                hidden_ptr,
                hidden_ptr,
                hidden_ptr,
                prepared.w13.view(torch.int32).view(-1),
                prepared.w2.view(torch.int32).view(-1),
                fc1_out[: routed_rows * 2 * INTERMEDIATE],
                activated,
                fc1_out[: routed_rows * HIDDEN],
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
                    __import__("cutlass").Float32,
                    topk_weights.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                ),
                fc1_scratch,
                fc2_scratch,
                workspace,
                rotation_placeholder,
                rotation_placeholder,
                rotation_placeholder,
                # The pinned SparkInfer ABI carries expert-map and two
                # trellis-LUT pointers even for packed W4A16. They are
                # compile-time inactive here but must still occupy their
                # launch slots.
                rotation_placeholder,
                rotation_placeholder,
                rotation_placeholder,
                EXPERTS,
                0,
                args.rows,
                grid,
                cuda.CUstream(stream.cuda_stream),
            )

        def launch(grid: int, prepared) -> None:
            launch_compiled(fused, grid, prepared)

        def launch_decomposed(grid: int, prepared) -> None:
            assert decomposed_fc1 is not None
            assert decomposed_fc2 is not None
            assert decomposed_activation is not None
            assert decomposed_fc1_locks is not None
            assert decomposed_fc2_locks is not None
            fc1_grid = args.decomposed_fc1_grid or grid
            fc2_grid = args.decomposed_fc2_grid or grid
            stream_arg = cuda.CUstream(stream.cuda_stream)
            topk_ptr = make_ptr(
                __import__("cutlass").Float32,
                topk_weights.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=4,
            )
            decomposed_fc1_locks.zero_()
            hidden_ptr = bf16_ptr(hidden)
            decomposed_fc1.compiled(
                hidden_ptr,
                hidden_ptr,
                prepared.w13.view(torch.int32).view(-1),
                bf16_ptr(fc1_out[: routed_rows * 2 * INTERMEDIATE]),
                prepared.w13_scale.view(torch.uint8).view(torch.int32).view(-1),
                prepared.w13_global_scale,
                packed_routes,
                block_experts,
                packed_route_count,
                topk_ptr,
                fc1_scratch,
                decomposed_fc1_locks,
                args.rows,
                fc1_grid,
                stream_arg,
            )
            decomposed_activation.compiled(
                fc1_out[: routed_rows * 2 * INTERMEDIATE],
                activated,
                routed_rows,
                stream_arg,
            )
            decomposed_fc2_locks.zero_()
            activated_ptr = bf16_ptr(activated)
            decomposed_fc2.compiled(
                activated_ptr,
                activated_ptr,
                prepared.w2.view(torch.int32).view(-1),
                bf16_ptr(fc1_out[: routed_rows * HIDDEN]),
                prepared.w2_scale.view(torch.uint8).view(torch.int32).view(-1),
                prepared.w2_global_scale,
                packed_routes,
                block_experts,
                packed_route_count,
                topk_ptr,
                fc2_scratch,
                decomposed_fc2_locks,
                routed_rows,
                fc2_grid,
                stream_arg,
            )

        if args.direct_topk_routes and block_size == 8:
            assert reference_fused is not None
            with torch.cuda.stream(stream):
                launch_compiled(reference_fused, 32, prepared_sets[0])
            stream.synchronize()
            reference_output = fc1_out[: routed_rows * HIDDEN].clone()

        for grid in args.grid_sizes:
            cooperative_cap = sms * int(fused.blocks_per_sm)
            if grid > cooperative_cap:
                print(
                    json.dumps(
                        {
                            "benchmark": "b12x_spark_w4a16_prefill_grid_skipped",
                            "block_size": block_size,
                            "blocks_per_sm": int(fused.blocks_per_sm),
                            "cooperative_cap": cooperative_cap,
                            "grid_x": grid,
                            "reason": "cooperative_launch_too_large",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue
            if args.fused_topk_sum:
                baseline = compile_w4a16_fused_moe(
                    size_m=args.rows,
                    hidden_size=HIDDEN,
                    intermediate_size=INTERMEDIATE,
                    num_experts=EXPERTS,
                    top_k=TOP_K,
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
                    direct_topk_routes=False,
                    tc_decode_fused_sum=False,
                )
                with torch.cuda.stream(stream):
                    launch_compiled(baseline, grid, prepared_sets[0])
                torch.cuda.synchronize()
                expected = (
                    fc1_out[: routed_rows * HIDDEN]
                    .view(args.rows, TOP_K, HIDDEN)
                    .float()
                    .sum(dim=1)
                    .to(torch.bfloat16)
                )
                with torch.cuda.stream(stream):
                    launch(grid, prepared_sets[0])
                torch.cuda.synchronize()
                actual = fc1_out[: args.rows * HIDDEN].view(args.rows, HIDDEN)
                difference = (actual.float() - expected.float()).abs()
                expected_f32 = expected.float()
                actual_f32 = actual.float()
                expected_norm = torch.linalg.vector_norm(expected_f32)
                difference_norm = torch.linalg.vector_norm(difference)
                print(
                    json.dumps(
                        {
                            "benchmark": "b12x_spark_w4a16_prefill_fused_sum_validation",
                            "block_size": block_size,
                            "cosine_similarity": float(
                                torch.nn.functional.cosine_similarity(
                                    actual_f32.reshape(1, -1),
                                    expected_f32.reshape(1, -1),
                                )
                            ),
                            "exact_fraction": float((actual == expected).float().mean()),
                            "expected_max_abs": float(expected_f32.abs().max()),
                            "expected_mean_abs": float(expected_f32.abs().mean()),
                            "finite": bool(torch.isfinite(actual.float()).all()),
                            "grid_x": grid,
                            "l2_relative_error": float(difference_norm / expected_norm),
                            "max_abs_error": float(difference.max()),
                            "mean_abs_error": float(difference.mean()),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            with torch.cuda.stream(stream):
                launch(grid, prepared_sets[0])
            stream.synchronize()
            validation = {}
            if reference_output is not None:
                actual = fc1_out[: routed_rows * HIDDEN]
                difference = (actual.float() - reference_output.float()).abs()
                reference_norm = torch.linalg.vector_norm(reference_output.float())
                validation = {
                    "bitwise_equal": bool(torch.equal(actual, reference_output)),
                    "l2_relative_error": float(
                        torch.linalg.vector_norm(difference) / reference_norm
                    ),
                    "max_abs_error": float(difference.max()),
                }
            graphs = []
            for prepared in prepared_sets:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    launch(grid, prepared)
                graphs.append(graph)
            graph_index = 0

            def replay_fused() -> None:
                nonlocal graph_index
                graphs[graph_index].replay()
                graph_index = (graph_index + 1) % len(graphs)

            samples = measure(
                replay_fused,
                args.warmup,
                args.iterations,
                args.repeats,
            )
            result = {
                "active_experts": args.active_experts,
                "block_size": block_size,
                "blocks_per_sm": int(fused.blocks_per_sm),
                "direct_topk_routes": args.direct_topk_routes,
                "executed_route_slots": len(packed_route_values),
                "fc1_tile_k": int(args.fc1_tile_k or fused.fc1_tile_k),
                "fc1_tile_n": int(args.fc1_tile_n or fused.fc1_tile_n),
                "fc2_tile_k": int(args.fc2_tile_k or fused.fc2_tile_k),
                "fc2_tile_n": int(args.fc2_tile_n or fused.fc2_tile_n),
                "fused_topk_sum": args.fused_topk_sum,
                "grid_x": grid,
                "logical_routes": routed_rows,
                "max_m_blocks": max_m_blocks,
                "median_ms": statistics.median(samples),
                "min_ms": min(samples),
                "implementation": "fused",
                "padding_factor": len(packed_route_values) / routed_rows,
                "pipeline_stages": args.pipeline_stages,
                "route_blocks": len(block_expert_values),
                "route_capacity": route_capacity,
                "samples_ms": samples,
                "tc_prefill_tiles": args.tc_prefill_tiles,
                "target_blocks_per_sm": args.target_blocks_per_sm or None,
                "weight_experts": args.weight_experts,
                "weight_layout": "packed",
                "weight_sets": args.weight_sets,
                **validation,
            }
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
            if args.decomposed:
                with torch.cuda.stream(stream):
                    launch(grid, prepared_sets[0])
                stream.synchronize()
                fused_reference = fc1_out[: routed_rows * HIDDEN].clone()
                with torch.cuda.stream(stream):
                    launch_decomposed(grid, prepared_sets[0])
                stream.synchronize()
                decomposed_output = fc1_out[: routed_rows * HIDDEN]
                decomposed_difference = (
                    decomposed_output.float() - fused_reference.float()
                ).abs()
                decomposed_graphs = []
                for prepared in prepared_sets:
                    decomposed_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(decomposed_graph, stream=stream):
                        launch_decomposed(grid, prepared)
                    decomposed_graphs.append(decomposed_graph)
                decomposed_graph_index = 0

                def replay_decomposed() -> None:
                    nonlocal decomposed_graph_index
                    decomposed_graphs[decomposed_graph_index].replay()
                    decomposed_graph_index = (
                        decomposed_graph_index + 1
                    ) % len(decomposed_graphs)

                decomposed_samples = measure(
                    replay_decomposed,
                    args.warmup,
                    args.iterations,
                    args.repeats,
                )
                decomposed_result = {
                    **result,
                    "blocks_per_sm": None,
                    "fc1_blocks_per_sm": int(decomposed_fc1.blocks_per_sm),
                    "fc1_grid_x": args.decomposed_fc1_grid or grid,
                    "fc1_tile_k": int(decomposed_fc1.tile_k),
                    "fc1_tile_n": int(decomposed_fc1.tile_n),
                    "fc2_blocks_per_sm": int(decomposed_fc2.blocks_per_sm),
                    "fc2_grid_x": args.decomposed_fc2_grid or grid,
                    "fc2_tile_k": int(decomposed_fc2.tile_k),
                    "fc2_tile_n": int(decomposed_fc2.tile_n),
                    "implementation": "decomposed",
                    "median_ms": statistics.median(decomposed_samples),
                    "min_ms": min(decomposed_samples),
                    "samples_ms": decomposed_samples,
                    "bitwise_equal_fused": bool(
                        torch.equal(decomposed_output, fused_reference)
                    ),
                    "l2_relative_error_fused": float(
                        torch.linalg.vector_norm(decomposed_difference)
                        / torch.linalg.vector_norm(fused_reference.float())
                    ),
                    "max_abs_error_fused": float(decomposed_difference.max()),
                }
                results.append(decomposed_result)
                print(json.dumps(decomposed_result, sort_keys=True), flush=True)

    print(
        json.dumps(
            {
                "benchmark": (
                    "b12x_spark_w4a16_direct_topk_grid"
                    if args.direct_topk_routes
                    else "b12x_spark_w4a16_prefill_grid"
                ),
                "best": min(results, key=lambda result: result["median_ms"]),
                "rows": args.rows,
                "sms": sms,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
