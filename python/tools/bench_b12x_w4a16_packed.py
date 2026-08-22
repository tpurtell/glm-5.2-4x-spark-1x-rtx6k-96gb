#!/usr/bin/env python3
from __future__ import annotations

import _pinned_sparkinfer  # noqa: F401

import argparse
import heapq
import json
import statistics
from dataclasses import replace
from pathlib import Path

import torch


HIDDEN = 6144
INTERMEDIATE = 512
MODEL_TOP_K = 8
EXPERTS = 256
BENCH_EXPERTS = 8
MAX_ROWS = 512
E4M3_ONE = 0x38


def parse_rows(value: str) -> list[int]:
    rows = [int(item) for item in value.split(",") if item.strip()]
    if not rows or any(row < 1 or row > MAX_ROWS for row in rows):
        raise argparse.ArgumentTypeError(
            f"rows must be comma-separated values in [1, {MAX_ROWS}]"
        )
    return rows


def route_ids_from_counts(
    counts: list[int], rows: int, top_k: int
) -> list[list[int]]:
    """Realize expert degrees without assigning one expert twice to a row."""
    row_heap = [(0, row_id) for row_id in range(rows)]
    heapq.heapify(row_heap)
    assignments = [[] for _ in range(rows)]
    for expert_id, count in sorted(
        enumerate(counts), key=lambda item: (-item[1], item[0])
    ):
        selected = [heapq.heappop(row_heap) for _ in range(count)]
        for degree, row_id in selected:
            assignments[row_id].append(expert_id)
            heapq.heappush(row_heap, (degree + 1, row_id))
    if any(len(row) != top_k for row in assignments):
        raise ValueError("expert route counts do not realize a balanced top-k plan")
    return assignments


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


def report(label: str, samples: list[float]) -> None:
    print(
        f"{label} median_ms={statistics.median(samples):.6f} "
        f"min_ms={min(samples):.6f} "
        f"samples={','.join(f'{sample:.6f}' for sample in samples)}",
        flush=True,
    )


def prepare_weights(device: torch.device, bench_experts: int):
    from b12x.moe._shared.kernels.w4a16.prepare import (
        prepare_w4a16_modelopt_nvfp4_weights,
    )

    w13 = torch.zeros(
        (bench_experts, 2 * INTERMEDIATE, HIDDEN // 2),
        dtype=torch.uint8,
        device=device,
    )
    w2 = torch.zeros(
        (bench_experts, HIDDEN, INTERMEDIATE // 2),
        dtype=torch.uint8,
        device=device,
    )
    w13_scale = torch.full(
        (bench_experts, 2 * INTERMEDIATE, HIDDEN // 16),
        E4M3_ONE,
        dtype=torch.uint8,
        device=device,
    ).view(torch.float8_e4m3fn)
    w2_scale = torch.full(
        (bench_experts, HIDDEN, INTERMEDIATE // 16),
        E4M3_ONE,
        dtype=torch.uint8,
        device=device,
    ).view(torch.float8_e4m3fn)
    globals_ = torch.ones(bench_experts, dtype=torch.float32, device=device)
    prepared = prepare_w4a16_modelopt_nvfp4_weights(
        w13,
        w13_scale,
        globals_,
        w2,
        w2_scale,
        globals_,
        activation="silu",
        params_dtype=torch.bfloat16,
        w13_layout="w13",
        reuse_input_storage=True,
    )
    # The selected expert IDs stay in [0, BENCH_EXPERTS), but compiling with
    # the production expert count preserves the real pointer strides and route plan.
    return replace(prepared, num_experts=EXPERTS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the exact E4M3/K16 packed W4A16 Spark MoE path."
    )
    parser.add_argument("--rows", type=parse_rows, default=parse_rows("1,8,16,32,64,128,256"))
    parser.add_argument("--top-k", type=int, choices=(1, MODEL_TOP_K), default=MODEL_TOP_K)
    parser.add_argument("--bench-experts", type=int, default=BENCH_EXPERTS)
    parser.add_argument(
        "--expert-route-counts-json",
        type=Path,
        help=(
            "JSON array of per-expert route counts; requires one row bucket and "
            "overrides the synthetic route pattern"
        ),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--block-sizes",
        type=parse_rows,
        default=[],
        help=(
            "also compile deterministic expert-packed candidates with these "
            "route block sizes (for example 8,16,32)"
        ),
    )
    args = parser.parse_args()
    if not 1 <= args.bench_experts <= EXPERTS:
        parser.error(f"--bench-experts must be in [1, {EXPERTS}]")
    expert_route_counts = None
    if args.expert_route_counts_json is not None:
        if len(args.rows) != 1:
            parser.error("--expert-route-counts-json requires exactly one row bucket")
        try:
            expert_route_counts = json.loads(
                args.expert_route_counts_json.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"invalid expert route-count JSON: {error}")
        if (
            not isinstance(expert_route_counts, list)
            or len(expert_route_counts) != EXPERTS
            or any(
                not isinstance(count, int) or count < 0
                for count in expert_route_counts
            )
            or sum(expert_route_counts) != args.rows[0] * args.top_k
            or any(count > args.rows[0] for count in expert_route_counts)
        ):
            parser.error(
                "expert route counts must contain 256 non-negative integers "
                "not exceeding rows and summing to rows * top-k"
            )
        if any(
            count > 0
            for count in expert_route_counts[args.bench_experts :]
        ):
            parser.error("expert route counts activate an unallocated expert")

    from b12x.moe._shared.kernels.w4a16.host import (
        make_w4a16_packed_buffers,
        max_packed_route_slots,
        packed_gemm_scratch_elements,
    )
    from b12x.moe._shared.kernels.w4a16.kernel import (
        compile_w4a16_fused_moe,
        run_w4a16_moe,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    sms = int(properties.multi_processor_count)
    max_shared_mem = int(properties.shared_memory_per_block_optin)
    prepared = prepare_weights(device, args.bench_experts)
    top_k = args.top_k
    for rows in args.rows:
        x = torch.randn((rows, HIDDEN), dtype=torch.bfloat16, device=device)
        if expert_route_counts is not None:
            try:
                route_ids = route_ids_from_counts(
                    expert_route_counts, rows, top_k
                )
            except ValueError as error:
                parser.error(str(error))
            topk_ids = torch.tensor(
                route_ids, dtype=torch.int32, device=device
            )
        elif top_k == 1:
            topk_ids = torch.zeros((rows, 1), dtype=torch.int32, device=device)
        else:
            token_ids = torch.arange(rows, dtype=torch.int32, device=device).view(rows, 1)
            route_ids = torch.arange(top_k, dtype=torch.int32, device=device).view(1, top_k)
            topk_ids = (token_ids + route_ids) % args.bench_experts
        topk_weights = torch.full(
            (rows, top_k), 1.0 / top_k, dtype=torch.float32, device=device
        )
        buffers = make_w4a16_packed_buffers(
            prepared,
            m=rows,
            topk=top_k,
            dtype=torch.bfloat16,
            device=device,
            route_num_experts=EXPERTS,
        )
        if args.block_sizes:
            maximum_route_slots = max(
                max_packed_route_slots(rows * top_k, block_size, EXPERTS)
                for block_size in args.block_sizes
            )
            maximum_route_blocks = max(
                (max_packed_route_slots(rows * top_k, block_size, EXPERTS)
                 + block_size - 1)
                // block_size
                for block_size in args.block_sizes
            )
            maximum_fc1_scratch = max(
                packed_gemm_scratch_elements(
                    size_n=2 * INTERMEDIATE,
                    route_slots=max_packed_route_slots(
                        rows * top_k, block_size, EXPERTS
                    ),
                    moe_block_size=block_size,
                    sms=sms,
                )
                for block_size in args.block_sizes
            )
            maximum_fc2_scratch = max(
                packed_gemm_scratch_elements(
                    size_n=HIDDEN,
                    route_slots=max_packed_route_slots(
                        rows * top_k, block_size, EXPERTS
                    ),
                    moe_block_size=block_size,
                    sms=sms,
                )
                for block_size in args.block_sizes
            )
            buffers = replace(
                buffers,
                packed_route_indices=torch.empty(
                    maximum_route_slots, dtype=torch.int32, device=device
                ),
                block_expert_ids=torch.empty(
                    maximum_route_blocks, dtype=torch.int32, device=device
                ),
                fc1_c_tmp=torch.empty(
                    maximum_fc1_scratch, dtype=torch.float32, device=device
                ),
                fc2_c_tmp=torch.empty(
                    maximum_fc2_scratch, dtype=torch.float32, device=device
                ),
            )

        def launch() -> None:
            run_w4a16_moe(
                x,
                prepared,
                topk_weights,
                topk_ids,
                activation="silu",
                intermediate_cache13=buffers.intermediate_cache13,
                intermediate_cache2=buffers.intermediate_cache2,
                output=buffers.output,
                fc1_c_tmp=buffers.fc1_c_tmp,
                fc2_c_tmp=buffers.fc2_c_tmp,
                packed_route_indices=buffers.packed_route_indices,
                block_expert_ids=buffers.block_expert_ids,
                packed_route_count=buffers.packed_route_count,
                expert_offsets=buffers.expert_offsets,
                apply_router_weight_on_input=False,
                fast_math=True,
            )

        launch()
        torch.cuda.synchronize()
        report(
            f"packed_w4a16_eager rows={rows} topk={top_k} routes={rows * top_k}",
            measure(launch, args.warmup, args.iterations, args.repeats),
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        report(
            f"packed_w4a16_graph rows={rows} topk={top_k} routes={rows * top_k}",
            measure(graph.replay, args.warmup, args.iterations, args.repeats),
        )

        deterministic_reference = None
        for block_size in args.block_sizes:
            if block_size != 8 and block_size % 16 != 0:
                raise ValueError("candidate block sizes must be 8 or multiples of 16")
            route_slots = max_packed_route_slots(
                rows * top_k, block_size, EXPERTS
            )
            max_m_blocks = (route_slots + block_size - 1) // block_size
            candidate = compile_w4a16_fused_moe(
                size_m=rows,
                hidden_size=HIDDEN,
                intermediate_size=INTERMEDIATE,
                num_experts=EXPERTS,
                top_k=top_k,
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

            def launch_candidate() -> None:
                run_w4a16_moe(
                    x,
                    prepared,
                    topk_weights,
                    topk_ids,
                    activation="silu",
                    intermediate_cache13=buffers.intermediate_cache13,
                    intermediate_cache2=buffers.intermediate_cache2,
                    output=buffers.output,
                    fc1_c_tmp=buffers.fc1_c_tmp,
                    fc2_c_tmp=buffers.fc2_c_tmp,
                    packed_route_indices=buffers.packed_route_indices,
                    block_expert_ids=buffers.block_expert_ids,
                    packed_route_count=buffers.packed_route_count,
                    expert_offsets=buffers.expert_offsets,
                    apply_router_weight_on_input=False,
                    fast_math=True,
                    fused_launch=candidate,
                )

            launch_candidate()
            torch.cuda.synchronize()
            actual = buffers.output.clone()
            if deterministic_reference is None:
                deterministic_reference = actual
            difference = (actual.float() - deterministic_reference.float()).abs()
            candidate_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(candidate_graph):
                launch_candidate()
            samples = measure(
                candidate_graph.replay,
                args.warmup,
                args.iterations,
                args.repeats,
            )
            print(
                f"packed_w4a16_candidate rows={rows} topk={top_k} "
                f"routes={rows * top_k} block={block_size} "
                f"blocks_per_sm={candidate.blocks_per_sm} "
                f"tiles={candidate.fc1_tile_n}x{candidate.fc1_tile_k}/"
                f"{candidate.fc2_tile_n}x{candidate.fc2_tile_k} "
                f"bitwise_reference={torch.equal(actual, deterministic_reference)} "
                f"max_abs_reference={float(difference.max()):.9f} "
                f"median_ms={statistics.median(samples):.6f} "
                f"min_ms={min(samples):.6f} "
                f"samples={','.join(f'{sample:.6f}' for sample in samples)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
