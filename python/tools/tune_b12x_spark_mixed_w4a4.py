#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass

import torch

HIDDEN = 6_144
INTERMEDIATE = 512
FC1_COLS = 2 * INTERMEDIATE
E4M3_ONE = 0x38


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def grouped_scale_storage(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    return torch.full(
        (1, align_up(rows, 128), align_up(cols // 16, 4)),
        E4M3_ONE,
        dtype=torch.uint8,
        device=device,
    )


def unpack_unit_scale_fp4(packed: torch.Tensor) -> torch.Tensor:
    if packed.ndim != 3 or packed.shape[-1] != 1:
        raise ValueError("expected grouped FP4 tensor [rows, cols/2, 1]")
    table = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.bfloat16,
        device=packed.device,
    )
    values = packed[..., 0]
    low = table[(values & 0x0F).long()]
    high = table[(values >> 4).long()]
    return torch.stack((low, high), dim=-1).reshape(values.shape[0], -1)


def comparison(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float | bool]:
    reference_f32 = reference.float()
    candidate_f32 = candidate.float()
    difference = candidate_f32 - reference_f32
    reference_norm = torch.linalg.vector_norm(reference_f32)
    candidate_norm = torch.linalg.vector_norm(candidate_f32)
    difference_norm = torch.linalg.vector_norm(difference)
    norm_product = reference_norm * candidate_norm
    return {
        "finite": bool(torch.isfinite(candidate_f32).all().item()),
        "max_abs_error": float(difference.abs().max().item()),
        "relative_l2_error": float(
            (difference_norm / reference_norm).item()
            if reference_norm.item() != 0.0
            else difference_norm.item()
        ),
        "cosine_similarity": float(
            (torch.sum(reference_f32 * candidate_f32) / norm_product).item()
            if norm_product.item() != 0.0
            else 1.0
        ),
        "bitwise_equal": bool(torch.equal(reference, candidate)),
    }


def capture_graph(stream: torch.cuda.Stream, operation) -> torch.cuda.CUDAGraph:
    with torch.cuda.stream(stream):
        operation()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        operation()
    graph.replay()
    stream.synchronize()
    return graph


def measure_graph(
    graph: torch.cuda.CUDAGraph,
    stream: torch.cuda.Stream,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> list[float]:
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            graph.replay()
    stream.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record(stream)
            for _ in range(iterations):
                graph.replay()
            end.record(stream)
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return samples


@dataclass
class PreparedWeights:
    source_w13: torch.Tensor
    source_w13_scale: torch.Tensor
    source_w2: torch.Tensor
    source_w2_scale: torch.Tensor
    packed: object


def prepare_weights(device: torch.device, seed: int) -> PreparedWeights:
    from b12x.cute.fp4 import as_grouped_scale_view
    from b12x.moe.fused.w4a16.prepare import (
        prepare_w4a16_modelopt_nvfp4_weights,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    source_w13 = torch.randint(
        0,
        256,
        (FC1_COLS, HIDDEN // 2, 1),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    source_w2 = torch.randint(
        0,
        256,
        (HIDDEN, INTERMEDIATE // 2, 1),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    w13_scale_storage = grouped_scale_storage(FC1_COLS, HIDDEN, device)
    w2_scale_storage = grouped_scale_storage(HIDDEN, INTERMEDIATE, device)
    global_scale = torch.ones(1, dtype=torch.float32, device=device)
    packed = prepare_w4a16_modelopt_nvfp4_weights(
        source_w13.permute(2, 0, 1).contiguous(),
        w13_scale_storage[:, :FC1_COLS, : HIDDEN // 16].view(torch.float8_e4m3fn),
        global_scale,
        source_w2.permute(2, 0, 1).contiguous(),
        w2_scale_storage[:, :HIDDEN, : INTERMEDIATE // 16].view(torch.float8_e4m3fn),
        global_scale,
        activation="silu",
        params_dtype=torch.bfloat16,
        # The benchmark source is already logical gate/up. Production W13 can
        # retain up/gate storage by swapping the two reads in the activation.
        w13_layout="w31",
        reuse_input_storage=False,
    )
    return PreparedWeights(
        source_w13=source_w13,
        source_w13_scale=as_grouped_scale_view(w13_scale_storage, FC1_COLS, HIDDEN),
        source_w2=source_w2,
        source_w2_scale=as_grouped_scale_view(w2_scale_storage, HIDDEN, INTERMEDIATE),
        packed=packed,
    )


def benchmark_rows(
    *,
    rows: int,
    weights: PreparedWeights,
    device: torch.device,
    stream: torch.cuda.Stream,
    grid_x: int,
    warmup: int,
    iterations: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    import cutlass
    from b12x.cute.fp4 import as_grouped_scale_view
    from b12x.gemm.dense import dense_gemm
    from b12x.moe.fused.w4a16.host import (
        max_packed_route_slots,
        packed_gemm_scratch_elements,
        select_route_block_size_m,
    )
    from b12x.moe.fused.w4a16.kernel import (
        _cutlass_element_dtype,
        _select_tile_config,
        compile_w4a16_activation,
        compile_w4a16_fused_moe,
        compile_w4a16_gemm,
        cuda,
        cute,
        make_ptr,
    )

    properties = torch.cuda.get_device_properties(device)
    sms = int(properties.multi_processor_count)
    max_shared_mem = int(properties.shared_memory_per_block_optin)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + rows)
    input_packed = torch.randint(
        0,
        256,
        (rows, HIDDEN // 2, 1),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    input_scale_storage = grouped_scale_storage(rows, HIDDEN, device)
    input_scale = as_grouped_scale_view(input_scale_storage, rows, HIDDEN)
    input_bf16 = unpack_unit_scale_fp4(input_packed).contiguous()
    alpha = torch.ones(1, dtype=torch.float32, device=device)

    block_size = select_route_block_size_m(rows, 1, 1)
    route_slots = max_packed_route_slots(rows, block_size, 1)
    route_blocks = (route_slots + block_size - 1) // block_size
    padded_rows = align_up(rows, block_size)
    packed_routes = torch.full((route_slots,), rows, dtype=torch.int32, device=device)
    packed_routes[:rows] = torch.arange(rows, dtype=torch.int32, device=device)
    block_experts = torch.zeros(route_blocks, dtype=torch.int32, device=device)
    packed_route_count = torch.tensor([padded_rows], dtype=torch.int32, device=device)
    topk_weights = torch.ones(rows, dtype=torch.float32, device=device)

    fc1_tile_k, fc1_tile_n, _, _ = _select_tile_config(
        problem_m=rows,
        problem_n=FC1_COLS,
        problem_k=HIDDEN,
        top_k=1,
        moe_block_size=block_size,
        sms=sms,
        max_shared_mem=max_shared_mem,
        scale_format="e4m3_k16",
    )
    fc2_tile_k, fc2_tile_n, _, _ = _select_tile_config(
        problem_m=rows,
        problem_n=HIDDEN,
        problem_k=INTERMEDIATE,
        top_k=1,
        moe_block_size=block_size,
        sms=sms,
        max_shared_mem=max_shared_mem,
        scale_format="e4m3_k16",
    )
    fc1_kernel = compile_w4a16_gemm(
        size_m=rows,
        size_n=FC1_COLS,
        size_k=HIDDEN,
        num_experts=1,
        top_k=1,
        mul_topk_weights=False,
        tile_n=fc1_tile_n,
        tile_k=fc1_tile_k,
        moe_block_size=block_size,
        max_m_blocks=route_blocks,
        element_dtype="bf16",
        scale_format="e4m3_k16",
    )
    fc2_kernel = compile_w4a16_gemm(
        size_m=rows,
        size_n=HIDDEN,
        size_k=INTERMEDIATE,
        num_experts=1,
        top_k=1,
        mul_topk_weights=False,
        tile_n=fc2_tile_n,
        tile_k=fc2_tile_k,
        moe_block_size=block_size,
        max_m_blocks=route_blocks,
        element_dtype="bf16",
        scale_format="e4m3_k16",
    )
    activation_kernel = compile_w4a16_activation(
        rows=rows,
        intermediate_size=INTERMEDIATE,
        activation="silu",
        element_dtype="bf16",
        fast_math=True,
    )
    direct_topk_routes = rows <= 8
    production_kernel = compile_w4a16_fused_moe(
        size_m=rows,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_experts=1,
        top_k=1,
        activation="silu",
        apply_router_weight_on_input=False,
        zero_fc2_output=False,
        moe_block_size=block_size,
        max_m_blocks=rows if direct_topk_routes else route_blocks,
        element_dtype="bf16",
        fast_math=True,
        sms=sms,
        max_shared_mem=max_shared_mem,
        weight_layout="packed",
        scale_format="e4m3_k16",
        direct_topk_routes=direct_topk_routes,
        tc_decode_fused_sum=direct_topk_routes,
    )

    candidate_fc1 = torch.empty(
        (rows, FC1_COLS, 1), dtype=torch.bfloat16, device=device
    )
    baseline_fc1 = torch.empty((rows, FC1_COLS), dtype=torch.bfloat16, device=device)
    candidate_activated = torch.empty(
        (rows, INTERMEDIATE), dtype=torch.bfloat16, device=device
    )
    baseline_activated = torch.empty_like(candidate_activated)
    candidate_output = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
    baseline_output = torch.empty_like(candidate_output)
    production_fc1 = torch.empty(
        rows * max(FC1_COLS, HIDDEN), dtype=torch.bfloat16, device=device
    )
    production_activated = torch.empty_like(candidate_activated)
    production_output = torch.empty_like(candidate_output)

    fc1_scratch = torch.empty(
        packed_gemm_scratch_elements(
            size_n=FC1_COLS,
            route_slots=route_slots,
            moe_block_size=block_size,
            sms=sms,
        ),
        dtype=torch.float32,
        device=device,
    )
    fc2_scratch = torch.empty(
        packed_gemm_scratch_elements(
            size_n=HIDDEN,
            route_slots=route_slots,
            moe_block_size=block_size,
            sms=sms,
        ),
        dtype=torch.float32,
        device=device,
    )
    production_route_slots = rows * block_size if direct_topk_routes else route_slots
    production_fc1_scratch = torch.empty(
        packed_gemm_scratch_elements(
            size_n=FC1_COLS,
            route_slots=production_route_slots,
            moe_block_size=block_size,
            sms=sms,
        ),
        dtype=torch.float32,
        device=device,
    )
    production_fc2_scratch = torch.empty(
        packed_gemm_scratch_elements(
            size_n=HIDDEN,
            route_slots=production_route_slots,
            moe_block_size=block_size,
            sms=sms,
        ),
        dtype=torch.float32,
        device=device,
    )
    fc1_locks = torch.zeros(4 * 256, dtype=torch.int32, device=device)
    fc2_locks = torch.zeros_like(fc1_locks)
    production_workspace = torch.zeros(sms * 4 + 2, dtype=torch.int32, device=device)
    production_routes = (
        torch.zeros_like(topk_weights, dtype=torch.int32)
        if direct_topk_routes
        else packed_routes
    )
    production_blocks = production_routes if direct_topk_routes else block_experts
    production_count = production_routes if direct_topk_routes else packed_route_count
    stream_arg = cuda.CUstream(stream.cuda_stream)
    topk_ptr = make_ptr(
        cutlass.Float32,
        topk_weights.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )

    def launch_w4a16(
        kernel,
        a: torch.Tensor,
        b: torch.Tensor,
        output: torch.Tensor,
        scale: torch.Tensor,
        global_scale: torch.Tensor,
        scratch: torch.Tensor,
        locks: torch.Tensor,
    ) -> None:
        locks.zero_()
        kernel.compiled(
            make_ptr(
                _cutlass_element_dtype("bf16"),
                a.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            b.view(torch.int32).view(-1),
            output.view(-1),
            scale.view(torch.uint8).view(torch.int32).view(-1),
            global_scale,
            packed_routes,
            block_experts,
            packed_route_count,
            topk_ptr,
            scratch,
            locks,
            rows,
            grid_x,
            stream_arg,
        )

    def launch_a4_fc1() -> None:
        dense_gemm(
            (input_packed, input_scale),
            (weights.source_w13, weights.source_w13_scale),
            out=candidate_fc1,
            alpha=alpha,
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            c_dtype="bfloat16",
            sf_vec_size=16,
            expected_m=rows,
        )

    def launch_a16_fc1() -> None:
        launch_w4a16(
            fc1_kernel,
            input_bf16,
            weights.packed.w13,
            baseline_fc1,
            weights.packed.w13_scale,
            weights.packed.w13_global_scale,
            fc1_scratch,
            fc1_locks,
        )

    def launch_candidate_activation() -> None:
        activation_kernel.compiled(
            candidate_fc1.view(-1), candidate_activated.view(-1), rows, stream_arg
        )

    def launch_baseline_activation() -> None:
        activation_kernel.compiled(
            baseline_fc1.view(-1), baseline_activated.view(-1), rows, stream_arg
        )

    def launch_candidate_fc2() -> None:
        launch_w4a16(
            fc2_kernel,
            candidate_activated,
            weights.packed.w2,
            candidate_output,
            weights.packed.w2_scale,
            weights.packed.w2_global_scale,
            fc2_scratch,
            fc2_locks,
        )

    def launch_baseline_fc2() -> None:
        launch_w4a16(
            fc2_kernel,
            baseline_activated,
            weights.packed.w2,
            baseline_output,
            weights.packed.w2_scale,
            weights.packed.w2_global_scale,
            fc2_scratch,
            fc2_locks,
        )

    def launch_candidate() -> None:
        launch_a4_fc1()
        launch_candidate_activation()
        launch_candidate_fc2()

    def launch_baseline() -> None:
        launch_a16_fc1()
        launch_baseline_activation()
        launch_baseline_fc2()

    def launch_production() -> None:
        production_workspace.zero_()
        production_kernel.compiled(
            make_ptr(
                _cutlass_element_dtype("bf16"),
                input_bf16.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            weights.packed.w13.view(torch.int32).view(-1),
            weights.packed.w2.view(torch.int32).view(-1),
            production_fc1[: rows * FC1_COLS],
            production_activated,
            production_output,
            weights.packed.w13_scale.view(torch.uint8).view(torch.int32).view(-1),
            weights.packed.w2_scale.view(torch.uint8).view(torch.int32).view(-1),
            weights.packed.w13_global_scale,
            weights.packed.w2_global_scale,
            production_routes,
            production_blocks,
            production_count,
            weights.packed.w13_global_scale,
            0,
            topk_ptr,
            production_fc1_scratch,
            production_fc2_scratch,
            production_workspace,
            rows,
            grid_x,
            stream_arg,
        )

    candidate_graph = capture_graph(stream, launch_candidate)
    candidate_reference = candidate_output.clone()
    baseline_graph = capture_graph(stream, launch_baseline)
    baseline_reference = baseline_output.clone()
    production_graph = capture_graph(stream, launch_production)
    production_reference = production_output.clone()
    candidate_graph.replay()
    stream.synchronize()
    replay_stable = torch.equal(candidate_reference, candidate_output)

    phase_operations = {
        "a4_fc1": launch_a4_fc1,
        "a16_fc1": launch_a16_fc1,
        "activation": launch_candidate_activation,
        "a16_fc2": launch_candidate_fc2,
    }
    phase_samples = {}
    for name, operation in phase_operations.items():
        graph = capture_graph(stream, operation)
        samples = measure_graph(
            graph,
            stream,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        )
        phase_samples[name] = {
            "median_ms": statistics.median(samples),
            "samples_ms": samples,
        }

    candidate_samples = measure_graph(
        candidate_graph,
        stream,
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )
    baseline_samples = measure_graph(
        baseline_graph,
        stream,
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )
    production_samples = measure_graph(
        production_graph,
        stream,
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )
    candidate_median = statistics.median(candidate_samples)
    baseline_median = statistics.median(baseline_samples)
    production_median = statistics.median(production_samples)
    return {
        "rows": rows,
        "block_size": block_size,
        "route_slots": route_slots,
        "grid_x": grid_x,
        "fc1_tile": [fc1_tile_k, fc1_tile_n],
        "fc2_tile": [fc2_tile_k, fc2_tile_n],
        "candidate_median_ms": candidate_median,
        "candidate_samples_ms": candidate_samples,
        "baseline_median_ms": baseline_median,
        "baseline_samples_ms": baseline_samples,
        "candidate_speedup_vs_production_w4a16": production_median / candidate_median,
        "production_w4a16_median_ms": production_median,
        "production_w4a16_samples_ms": production_samples,
        "candidate_replay_bitwise_stable": replay_stable,
        "candidate_vs_decomposed_w4a16": comparison(
            baseline_reference, candidate_reference
        ),
        "candidate_vs_production_w4a16": comparison(
            production_reference, candidate_reference
        ),
        "phases": phase_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark an off-path Spark W4A4 FC1 + BF16 activation + "
            "packed-W4A16 FC2 composition."
        )
    )
    parser.add_argument("--rows", default="1,2,4,8,16,32,64,128,256")
    parser.add_argument("--grid-x", type=int, default=48)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        row_counts = [int(value) for value in args.rows.split(",")]
    except ValueError as error:
        parser.error(f"invalid rows: {error}")
    if (
        not row_counts
        or any(rows < 1 or rows > 256 for rows in row_counts)
        or len(set(row_counts)) != len(row_counts)
    ):
        parser.error("rows must be unique integers in 1..256")
    if min(args.grid_x, args.warmup, args.iterations, args.repeats) < 1:
        parser.error("grid-x, warmup, iterations, and repeats must be positive")

    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    max_grid = int(properties.multi_processor_count) * 2
    if args.grid_x > max_grid:
        parser.error(f"grid-x cannot exceed cooperative cap {max_grid}")
    stream = torch.cuda.Stream()
    weights = prepare_weights(device, args.seed)
    results = []
    for rows in row_counts:
        result = benchmark_rows(
            rows=rows,
            weights=weights,
            device=device,
            stream=stream,
            grid_x=32 if rows == 1 and args.grid_x == 48 else args.grid_x,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
            seed=args.seed,
        )
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    report = {
        "benchmark": "b12x_spark_mixed_w4a4_fc1_bf16_fc2",
        "device": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "shape": {
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "experts": 1,
            "top_k": 1,
        },
        "input": "prequantized NVFP4 with E4M3 K16 unit scales",
        "weights": "random NVFP4 with E4M3 K16 unit scales",
        "resident_layout": "source W13 plus packed W2; packed W13 retained only as timing oracle",
        "serving_path_changed": False,
        "results": results,
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        with open(args.output, "w", encoding="ascii") as output:
            output.write(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
