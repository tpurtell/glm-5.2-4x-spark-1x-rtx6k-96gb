#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import statistics
from pathlib import Path

import torch


HIDDEN = 6144
INTERMEDIATE = 512
EXPERTS = 256
TOP_K = 8
E4M3_ONE = 0x38
W4A4_SCRATCH_BYTES = 46_459_260


class DeviceBuffer(ctypes.Structure):
    _fields_ = (
        ("ptr", ctypes.c_void_p),
        ("bytes", ctypes.c_size_t),
        ("device_id", ctypes.c_int),
        ("flags", ctypes.c_uint64),
    )


class SparkW4A4Buffers(ctypes.Structure):
    _fields_ = tuple(
        (name, DeviceBuffer)
        for name in (
            "input",
            "topk_ids",
            "topk_weights",
            "w13_weight",
            "w13_scale",
            "w1_alphas",
            "a1_gscale",
            "w2_weight",
            "w2_scale",
            "w2_alphas",
            "a2_gscale",
            "output",
            "scratch",
        )
    )


class SparkMoeTp4M1Buffers(ctypes.Structure):
    _fields_ = tuple(
        (name, DeviceBuffer)
        for name in (
            "input_payload",
            "input_bf16",
            "w13_weight",
            "w13_scale",
            "w1_alphas",
            "a1_gscale",
            "a2_gscale",
            "intermediate",
            "w2_weight",
            "w2_scale",
            "w2_alphas",
            "topk_ids",
            "topk_weights",
            "output",
            "barrier_count",
            "barrier_epoch",
        )
    )


def device_buffer(tensor: torch.Tensor) -> DeviceBuffer:
    return DeviceBuffer(
        tensor.data_ptr(),
        tensor.numel() * tensor.element_size(),
        tensor.device.index or 0,
        0,
    )


def check_status(lib: ctypes.CDLL, status: int, action: str) -> None:
    if status == 0:
        return
    error = ctypes.create_string_buffer(512)
    lib.glmrt_last_error_message(error, len(error))
    raise RuntimeError(f"{action} failed with status {status}: {error.value.decode()}")


def comparison_metrics(
    prefix: str, reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float | bool]:
    reference_f64 = reference.double()
    candidate_f64 = candidate.double()
    finite = torch.isfinite(reference_f64) & torch.isfinite(candidate_f64)
    difference = reference_f64[finite] - candidate_f64[finite]
    reference_finite = reference_f64[finite]
    candidate_finite = candidate_f64[finite]
    reference_norm = torch.linalg.vector_norm(reference_finite)
    candidate_norm = torch.linalg.vector_norm(candidate_finite)
    difference_norm = torch.linalg.vector_norm(difference)
    norm_product = reference_norm * candidate_norm
    return {
        f"{prefix}_max_abs_error": (
            difference.abs().max().item() if difference.numel() else 0.0
        ),
        f"{prefix}_relative_l2_error": (
            (difference_norm / reference_norm).item()
            if reference_norm.item() != 0.0
            else difference_norm.item()
        ),
        f"{prefix}_cosine_similarity": (
            (reference_finite @ candidate_finite / norm_product).item()
            if norm_product.item() != 0.0
            else 1.0
        ),
        f"{prefix}_nonfinite_match": torch.equal(
            torch.isnan(reference_f64), torch.isnan(candidate_f64)
        )
        and torch.equal(torch.isinf(reference_f64), torch.isinf(candidate_f64)),
    }


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


def route_ids(
    rows: int,
    active_experts: int,
    device: torch.device,
    expert_route_counts: list[int] | None,
) -> torch.Tensor:
    if expert_route_counts is not None:
        ids = [
            expert_id
            for expert_id, count in enumerate(expert_route_counts)
            for _ in range(count)
        ]
        return torch.tensor(ids, dtype=torch.int32, device=device).view(rows, TOP_K)
    routes = torch.arange(rows * TOP_K, dtype=torch.int32, device=device)
    return (routes % active_experts).view(rows, TOP_K)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark B12X's grouped dynamic W4A4 MoE kernel on a Spark shape."
    )
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--active-experts", type=int, default=EXPERTS)
    parser.add_argument(
        "--quant-mode",
        choices=("nvfp4", "w4a8_nvfp4"),
        default="nvfp4",
    )
    parser.add_argument(
        "--expert-route-counts",
        help="Comma-separated counts for experts 0..N-1; overrides uniform routing.",
    )
    parser.add_argument(
        "--expert-route-counts-json",
        type=Path,
        help="JSON array of counts for experts 0..255; overrides uniform routing.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--native-lib",
        type=Path,
        help="also validate and time the native W4A4 AOT entry point",
    )
    parser.add_argument(
        "--native-only",
        action="store_true",
        help="skip the Python/CuTe launch and benchmark only the native AOT entry point",
    )
    args = parser.parse_args()
    if args.rows < 1 or min(args.warmup, args.iterations, args.repeats) < 1:
        parser.error("rows, warmup, iterations, and repeats must be positive")
    if not TOP_K <= args.active_experts <= EXPERTS:
        parser.error(f"active-experts must be between {TOP_K} and {EXPERTS}")
    if args.native_only and args.native_lib is None:
        parser.error("--native-only requires --native-lib")
    if args.native_lib is not None and args.quant_mode != "nvfp4":
        parser.error("native W4A4 AOT comparison requires quant-mode=nvfp4")
    if (
        args.expert_route_counts is not None
        and args.expert_route_counts_json is not None
    ):
        parser.error(
            "expert-route-counts and expert-route-counts-json are mutually exclusive"
        )
    expert_route_counts = None
    if args.expert_route_counts_json is not None:
        try:
            expert_route_counts = json.loads(
                args.expert_route_counts_json.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"invalid expert-route-counts-json: {error}")
    elif args.expert_route_counts is not None:
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
            or sum(expert_route_counts) != args.rows * TOP_K
        ):
            parser.error(
                "expert-route-counts must contain at most 256 non-negative values "
                f"summing to {args.rows * TOP_K}"
            )
        args.active_experts = sum(count > 0 for count in expert_route_counts)

    from b12x.integration.tp_moe import (
        TPMoEScratchCaps,
        plan_b12x_fp4_moe_weights,
        plan_tp_moe_scratch,
        prepare_b12x_fp4_moe_weights,
    )

    torch.manual_seed(args.seed)
    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    weight_plan = plan_b12x_fp4_moe_weights(
        quant_modes=args.quant_mode,
        source_format="modelopt_nvfp4",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        w13_layout="w13",
    )
    w13 = torch.randint(
        0,
        256,
        (EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2),
        dtype=torch.uint8,
        device=device,
    )
    w2 = torch.randint(
        0,
        256,
        (EXPERTS, HIDDEN, INTERMEDIATE // 2),
        dtype=torch.uint8,
        device=device,
    )
    w13_scale = torch.full(
        (EXPERTS, 2 * INTERMEDIATE, HIDDEN // 16),
        E4M3_ONE,
        dtype=torch.uint8,
        device=device,
    ).view(torch.float8_e4m3fn)
    w2_scale = torch.full(
        (EXPERTS, HIDDEN, INTERMEDIATE // 16),
        E4M3_ONE,
        dtype=torch.uint8,
        device=device,
    ).view(torch.float8_e4m3fn)
    global_scale = torch.ones(EXPERTS, dtype=torch.float32, device=device)
    activation_gscale = torch.ones(EXPERTS, dtype=torch.float32, device=device)
    experts = prepare_b12x_fp4_moe_weights(
        plan=weight_plan,
        w1_global_scale=global_scale,
        w2_global_scale=global_scale,
        w1_fp4=w13,
        w1_blockscale=w13_scale,
        w2_fp4=w2,
        w2_blockscale=w2_scale,
        a1_gscale=activation_gscale,
        a2_gscale=activation_gscale,
        params_dtype=torch.bfloat16,
    )
    hidden = torch.randn((args.rows, HIDDEN), dtype=torch.bfloat16, device=device)
    topk_ids = route_ids(args.rows, args.active_experts, device, expert_route_counts)
    topk_weights = torch.full(
        (args.rows, TOP_K),
        1.0 / TOP_K,
        dtype=torch.float32,
        device=device,
    )
    python_result = {}
    output = None
    graph = None
    if not args.native_only:
        caps = TPMoEScratchCaps(
            max_tokens=args.rows,
            num_topk=TOP_K,
            device=device,
            weight_plan=weight_plan,
            quant_mode=args.quant_mode,
            core_token_counts=(args.rows,),
            route_num_experts=0,
            frozen=True,
        )
        scratch_plan = plan_tp_moe_scratch(caps)
        scratch = torch.empty(
            scratch_plan.layout.total_nbytes,
            dtype=torch.uint8,
            device=device,
        )
        output = torch.empty_like(hidden)
        binding = scratch_plan.bind(
            scratch=scratch,
            a=hidden,
            experts=experts,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            output=output,
            fast_math=True,
        )

        binding.run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            binding.run()
        graph.replay()
        torch.cuda.synchronize()
        replay_reference = output.clone()
        graph.replay()
        torch.cuda.synchronize()
        replay_metrics = comparison_metrics(
            "replay", replay_reference.float(), output.float()
        )
        samples = measure(graph.replay, args.warmup, args.iterations, args.repeats)
        python_result = {
            "implementation": binding.implementation,
            "max_rows": binding.max_rows,
            "physical_tiles_capacity": binding.physical_tiles_capacity,
            "scratch_bytes": scratch.numel(),
            "task_capacity": binding.task_capacity,
            "median_ms": statistics.median(samples),
            "min_ms": min(samples),
            "samples_ms": samples,
            "replay_bitwise_equal": bool(torch.equal(replay_reference, output)),
            **replay_metrics,
        }
    native_result = {}
    if args.native_lib is not None:
        if args.rows > 256:
            parser.error("native W4A4 AOT currently supports at most 256 rows")
        lib = ctypes.CDLL(str(args.native_lib.resolve()))
        lib.glmrt_cuda_b12x_spark_aot_init.restype = ctypes.c_int
        launch = lib.glmrt_cuda_b12x_spark_w4a4_prefill_topk8_bf16_async
        launch.argtypes = (
            ctypes.POINTER(SparkW4A4Buffers),
            ctypes.c_size_t,
            ctypes.c_void_p,
        )
        launch.restype = ctypes.c_int
        launch_m1 = lib.glmrt_cuda_b12x_spark_moe_tp4_m1_nvfp4_async
        launch_m1.argtypes = (
            ctypes.POINTER(SparkMoeTp4M1Buffers),
            ctypes.c_size_t,
            ctypes.c_void_p,
        )
        launch_m1.restype = ctypes.c_int
        check_status(
            lib,
            lib.glmrt_cuda_b12x_spark_aot_init(),
            "initializing native B12X AOT modules",
        )
        native_output = torch.empty_like(hidden)
        native_scratch = torch.empty(
            W4A4_SCRATCH_BYTES, dtype=torch.uint8, device=device
        )
        native_buffers = SparkW4A4Buffers(
            device_buffer(hidden),
            device_buffer(topk_ids),
            device_buffer(topk_weights),
            device_buffer(experts.w1_fp4),
            device_buffer(experts.w1_blockscale),
            device_buffer(experts.w1_alphas),
            device_buffer(experts.a1_gscale),
            device_buffer(experts.w2_fp4),
            device_buffer(experts.w2_blockscale),
            device_buffer(experts.w2_alphas),
            device_buffer(experts.a2_gscale),
            device_buffer(native_output),
            device_buffer(native_scratch),
        )

        def native_launch() -> None:
            status = launch(
                ctypes.byref(native_buffers),
                args.rows,
                ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
            )
            check_status(lib, status, "launching native W4A4 prefill")

        native_launch()
        torch.cuda.synchronize()
        if output is None:
            reference = native_output.float().clone()
            native_launch()
            torch.cuda.synchronize()
            comparison = comparison_metrics(
                "native_repeat", reference, native_output.float()
            )
        else:
            reference = output.float().clone()
            comparison = comparison_metrics("native", reference, native_output.float())
        if graph is not None:
            graph.replay()
            torch.cuda.synchronize()
            comparison.update(
                comparison_metrics("python_replay", reference, output.float())
            )
        native_eager_samples = measure(
            native_launch, args.warmup, args.iterations, args.repeats
        )
        native_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(native_graph):
            native_launch()
        native_graph_samples = measure(
            native_graph.replay, args.warmup, args.iterations, args.repeats
        )
        native_result = {
            "native_eager_median_ms": statistics.median(native_eager_samples),
            "native_eager_samples_ms": native_eager_samples,
            "native_graph_median_ms": statistics.median(native_graph_samples),
            "native_graph_samples_ms": native_graph_samples,
            "native_scratch_bytes": native_scratch.numel(),
            **comparison,
        }
        if args.rows == 1:
            payload_stride = HIDDEN // 2 + HIDDEN // 16
            input_payload = torch.empty(
                payload_stride, dtype=torch.uint8, device=device
            )
            input_payload[: HIDDEN // 2] = 0x22
            input_payload[HIDDEN // 2 :] = E4M3_ONE
            m1_input_bf16 = torch.empty_like(hidden)
            m1_intermediate = torch.empty(
                (TOP_K, INTERMEDIATE), dtype=torch.bfloat16, device=device
            )
            m1_output = torch.empty_like(hidden)
            barrier_count = torch.empty(1, dtype=torch.int32, device=device)
            barrier_epoch = torch.empty(1, dtype=torch.int32, device=device)
            m1_buffers = SparkMoeTp4M1Buffers(
                device_buffer(input_payload),
                device_buffer(m1_input_bf16),
                device_buffer(experts.w1_fp4),
                device_buffer(experts.w1_blockscale),
                device_buffer(experts.w1_alphas),
                device_buffer(experts.a1_gscale),
                device_buffer(experts.a2_gscale),
                device_buffer(m1_intermediate),
                device_buffer(experts.w2_fp4),
                device_buffer(experts.w2_blockscale),
                device_buffer(experts.w2_alphas),
                device_buffer(topk_ids),
                device_buffer(topk_weights),
                device_buffer(m1_output),
                device_buffer(barrier_count),
                device_buffer(barrier_epoch),
            )

            def native_m1_launch() -> None:
                status = launch_m1(
                    ctypes.byref(m1_buffers),
                    payload_stride,
                    ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
                )
                check_status(lib, status, "launching native W4A4 M1 micro kernel")

            native_m1_launch()
            torch.cuda.synchronize()
            m1_reference = m1_output.float().clone()
            native_m1_launch()
            torch.cuda.synchronize()
            m1_comparison = comparison_metrics(
                "native_m1_repeat", m1_reference, m1_output.float()
            )
            native_m1_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(native_m1_graph):
                native_m1_launch()
            native_m1_graph_samples = measure(
                native_m1_graph.replay,
                args.warmup,
                args.iterations,
                args.repeats,
            )
            native_result.update(
                {
                    "native_m1_graph_median_ms": statistics.median(
                        native_m1_graph_samples
                    ),
                    "native_m1_graph_samples_ms": native_m1_graph_samples,
                    **m1_comparison,
                }
            )
    counts = torch.bincount(topk_ids.reshape(-1), minlength=EXPERTS).cpu().tolist()
    active_counts = [count for count in counts if count]
    print(
        json.dumps(
            {
                "benchmark": "b12x_spark_w4a4_grouped",
                "rows": args.rows,
                "routes": args.rows * TOP_K,
                "experts": EXPERTS,
                "active_experts": len(active_counts),
                "input_encoding": (
                    "bf16_requantized_to_mxfp8"
                    if args.quant_mode == "w4a8_nvfp4"
                    else "bf16_requantized_to_nvfp4"
                ),
                "quant_mode": args.quant_mode,
                "route_rows_min": min(active_counts),
                "route_rows_p50": statistics.median(active_counts),
                "route_rows_max": max(active_counts),
                "weight_payload": "random",
                **python_result,
                **native_result,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
