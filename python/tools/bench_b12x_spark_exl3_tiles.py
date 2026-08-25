#!/usr/bin/env python3
"""Tune SparkInfer EXL3 tiles at GLM-5.2's exact TP4 geometry."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics

os.environ["B12X_COMPILE_DISK_CACHE"] = "0"
os.environ["B12X_COMPILE_MEMORY_CACHE"] = "0"

import _pinned_sparkinfer  # noqa: E402,F401
import torch  # noqa: E402

from _b12x_exl3_k3_profile import (  # noqa: E402
    K128_N128_FC1,
    K64_N128,
    K64_N256,
)

from b12x.moe import fused_moe  # noqa: E402
from b12x.moe._shared.kernels.w4a16.host import (  # noqa: E402
    select_route_block_size_m,
)
from validate_b12x_exl3_native import (  # noqa: E402
    _canonical_json,
    _load_route_profile_sample,
    _route_ids_from_counts,
    _validate_route_counts,
)


EXPERTS = 256
HIDDEN = 6144
INTERMEDIATE = 512
TOP_K = 8
BITS = 3
TILE_CANDIDATES = (
    K64_N256,
    K128_N128_FC1,
    (64, 256, 128, 128),
    (128, 128, 128, 128),
    K64_N128,
    (128, 64, 128, 64),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--rows",
        default="1",
        help=(
            "comma-separated candidate M regimes (every exact M through 32, "
            "then powers of two plus the 2064-row suffix bucket)"
        ),
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--route-block-rows",
        type=int,
        choices=(8, 16, 32, 48, 64),
        help=(
            "override SparkInfer's automatic same-expert route block for an "
            "isolated scheduling sweep"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--route-profile",
        type=Path,
        help="accepted live GLM route profile; requires one exact --rows value",
    )
    parser.add_argument(
        "--route-profile-sample",
        type=int,
        help="zero-based fixture index from --route-profile",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional content-bound JSON result; refuses to overwrite",
    )
    return parser.parse_args()


def _time_graph(
    graph: torch.cuda.CUDAGraph,
    *,
    iterations: int,
    rounds: int,
) -> list[float]:
    samples: list[float] = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return samples


def _make_source_tensors(device: torch.device) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device)
    generator.manual_seed(20260823)
    trellis_words = 16 * BITS
    w13 = torch.randint(
        -32768,
        32767,
        (
            2,
            EXPERTS,
            HIDDEN // 16,
            INTERMEDIATE // 16,
            trellis_words,
        ),
        dtype=torch.int16,
        device=device,
        generator=generator,
    )
    w2 = torch.randint(
        -32768,
        32767,
        (
            EXPERTS,
            INTERMEDIATE // 16,
            HIDDEN // 16,
            trellis_words,
        ),
        dtype=torch.int16,
        device=device,
        generator=generator,
    )
    hidden_rotations = torch.ones(
        (EXPERTS, HIDDEN), dtype=torch.float16, device=device
    )
    intermediate_rotations = torch.ones(
        (EXPERTS, 3 * INTERMEDIATE), dtype=torch.float16, device=device
    )
    return w13, w2, hidden_rotations, intermediate_rotations


def _prepare_weights(
    source_tensors: tuple[torch.Tensor, ...],
    *,
    tile_config: tuple[int, int, int, int],
) -> fused_moe.ExpertWeights:
    w13, w2, hidden_rotations, intermediate_rotations = source_tensors
    plan = fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format="exl3_trellis_mcg",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        w13_layout="w13",
        trellis_bits=BITS,
        trellis_tile_config=tile_config,
    )
    return fused_moe.prepare_weights(
        plan=plan,
        params_dtype=torch.bfloat16,
        w1_fp4=w13,
        w2_fp4=w2,
        gate_suh=hidden_rotations,
        up_suh=hidden_rotations,
        intermediate_rotations=intermediate_rotations,
        down_svh=hidden_rotations,
        trellis_mcg=0xCBAC1FED,
    )


def _benchmark_case(
    *,
    experts: fused_moe.ExpertWeights,
    rows: int,
    device: torch.device,
    iterations: int,
    rounds: int,
    warmup: int,
    reference: torch.Tensor | None,
    expert_route_counts: list[int] | None,
    route_block_rows: int | None,
) -> tuple[dict[str, object], torch.Tensor]:
    block_size = (
        select_route_block_size_m(rows, TOP_K, EXPERTS)
        if route_block_rows is None
        else int(route_block_rows)
    )
    plan = fused_moe.plan(
        fused_moe.Caps(
            max_tokens=rows,
            num_topk=TOP_K,
            route_num_experts=EXPERTS,
            device=device,
            weight_plan=experts.plan,
            quant_mode="w4a16",
            w4a16_block_size_m=block_size,
        )
    )
    scratch_spec = plan.scratch_specs()[0]
    scratch = torch.empty(scratch_spec.shape, dtype=scratch_spec.dtype, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(20260823 + rows)
    source = (
        torch.randn((rows, HIDDEN), device=device, generator=generator) * 0.002
    ).to(torch.bfloat16)
    if expert_route_counts is None:
        row_ids = torch.arange(rows, device=device, dtype=torch.int32).view(-1, 1)
        route_offsets = torch.arange(TOP_K, device=device, dtype=torch.int32).view(1, -1)
        topk_ids = (row_ids * 17 + route_offsets * 29) % EXPERTS
    else:
        topk_ids = torch.tensor(
            _route_ids_from_counts(expert_route_counts, rows, TOP_K),
            dtype=torch.int32,
            device=device,
        )
    topk_weights = torch.rand(
        (rows, TOP_K), dtype=torch.float32, device=device, generator=generator
    )
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)
    binding = plan.bind(
        scratch=scratch,
        a=source,
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    eager = binding.run().clone()
    torch.cuda.synchronize(device)
    if not torch.isfinite(eager).all() or float(eager.norm()) <= 1.0e-9:
        raise RuntimeError(f"non-finite or zero EXL3 output at rows={rows}")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = binding.run()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize(device)
    if not torch.equal(captured, eager):
        raise RuntimeError(f"CUDA graph differs from eager EXL3 output at rows={rows}")

    if reference is None:
        reference = eager.clone()
    difference = eager - reference
    relative_l2 = float(difference.norm() / reference.norm().clamp_min(1.0e-9))
    cosine = float(
        torch.nn.functional.cosine_similarity(
            eager.flatten(), reference.flatten(), dim=0
        )
    )
    samples = _time_graph(graph, iterations=iterations, rounds=rounds)
    launch = dict(plan._prewarmed_fused_launches)[rows]
    result: dict[str, object] = {
        "rows": rows,
        "route_block_rows": block_size,
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "samples_ms": samples,
        "fc1_tile": [launch.fc1_tile_k, launch.fc1_tile_n],
        "fc2_tile": [launch.fc2_tile_k, launch.fc2_tile_n],
        "blocks_per_sm": launch.blocks_per_sm,
        "registers_per_thread": launch.registers_per_thread,
        "local_memory_bytes": launch.local_memory_bytes,
        "relative_l2": relative_l2,
        "cosine": cosine,
    }
    del captured, graph, binding, scratch, plan, eager
    return result, reference


def main() -> None:
    args = _parse_args()
    rows = tuple(int(value) for value in args.rows.split(",") if value.strip())
    if not rows or any(value < 1 or value > 2064 for value in rows):
        raise SystemExit("--rows must contain exact M values in 1..2064")
    if args.iterations < 1 or args.rounds < 1 or args.warmup < 1:
        raise SystemExit("iterations, rounds, and warmup must be positive")
    if (args.route_profile is None) != (args.route_profile_sample is None):
        raise SystemExit("--route-profile and --route-profile-sample are required together")
    if args.route_profile is not None and len(rows) != 1:
        raise SystemExit("--route-profile requires exactly one --rows value")
    if args.route_profile_sample is not None and args.route_profile_sample < 0:
        raise SystemExit("--route-profile-sample must be non-negative")
    if args.output is not None and args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("--device must select CUDA")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    major, minor = torch.cuda.get_device_capability(device)
    if major != 12 or minor not in (0, 1):
        raise SystemExit(f"benchmark requires SM120/SM121, got SM{major}{minor}")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    expert_route_counts = None
    route_fixture = None
    if args.route_profile is not None:
        expert_route_counts, route_fixture = _load_route_profile_sample(
            args.route_profile, args.route_profile_sample, rows[0]
        )
        expert_route_counts = _validate_route_counts(expert_route_counts, rows[0])

    source_tensors = _make_source_tensors(device)
    references: dict[int, torch.Tensor] = {}
    results: list[dict[str, object]] = []
    for tile_config in TILE_CANDIDATES:
        experts = _prepare_weights(source_tensors, tile_config=tile_config)
        for row_count in rows:
            try:
                result, reference = _benchmark_case(
                    experts=experts,
                    rows=row_count,
                    device=device,
                    iterations=args.iterations,
                    rounds=args.rounds,
                    warmup=args.warmup,
                    reference=references.get(row_count),
                    expert_route_counts=expert_route_counts,
                    route_block_rows=args.route_block_rows,
                )
            except ValueError as exc:
                if "force_tile_config" not in str(exc) or "does not fit" not in str(exc):
                    raise
                results.append(
                    {
                        "rows": row_count,
                        "tile_config": list(tile_config),
                        "skipped": str(exc),
                    }
                )
                continue
            references.setdefault(row_count, reference)
            result["tile_config"] = list(tile_config)
            results.append(result)
        del experts
        gc.collect()
        torch.cuda.empty_cache()

    winners = {
        str(row_count): min(
            (
                result
                for result in results
                if result["rows"] == row_count and "median_ms" in result
            ),
            key=lambda result: float(result["median_ms"]),
        )["tile_config"]
        for row_count in rows
    }
    body = {
                "device": torch.cuda.get_device_name(device),
                "geometry": {
                    "experts": EXPERTS,
                    "hidden": HIDDEN,
                    "intermediate_tp4": INTERMEDIATE,
                    "top_k": TOP_K,
                    "bits": BITS,
                },
                "iterations": args.iterations,
                "rounds": args.rounds,
                "route_block_rows_override": args.route_block_rows,
                "route_fixture": route_fixture,
                "results": results,
                "winners": winners,
    }
    report = {
        **body,
        "report_sha256": hashlib.sha256(_canonical_json(body)).hexdigest(),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
