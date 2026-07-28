#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


ROWS = [1, 2, 4, 8, 16]
BASELINE_DECODE_TPS = 1.66
BASELINE_DECODE_MS = 1000.0 / BASELINE_DECODE_TPS


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def latest_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no artifact matched {directory / pattern}")
    return matches[-1]


def latest_optional(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern))
    if not matches:
        return None
    return matches[-1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL row") from exc
    return rows


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON") from exc
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected JSON object")
    return data


def ok_kernel_rows(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in load_jsonl(path):
        if row.get("benchmark") != "cuda_kernel_microbench" or row.get("status") != "ok":
            continue
        kernel = row.get("kernel")
        rows = row.get("rows")
        if isinstance(kernel, str) and isinstance(rows, int):
            indexed[(kernel, rows)] = row
    return indexed


def require_row(
    indexes: list[dict[tuple[str, int], dict[str, Any]]],
    kernel: str,
    rows: int,
) -> dict[str, Any]:
    for index in indexes:
        row = index.get((kernel, rows))
        if row is not None:
            return row
    raise KeyError(f"missing benchmark row kernel={kernel} rows={rows}")


def avg_ms(row: dict[str, Any]) -> float:
    value = row.get("avg_ms")
    if not isinstance(value, (int, float)):
        raise TypeError(f"benchmark row missing numeric avg_ms: {row}")
    return float(value)


def scheduler_overhead_ms(path: Path) -> float:
    data = load_json(path)
    value = data.get("scheduler_overhead_per_call_us")
    if not isinstance(value, (int, float)):
        raise TypeError(f"scheduler artifact missing numeric scheduler_overhead_per_call_us: {path}")
    return float(value) / 1000.0


def replay_rows(path: Path | None) -> dict[tuple[str, int], dict[str, Any]]:
    if path is None:
        return {}
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in load_jsonl(path):
        if row.get("benchmark") != "phase0_layer_sweep_replay" or row.get("status") != "ok":
            continue
        kernel = row.get("kernel")
        rows = row.get("rows")
        if isinstance(kernel, str) and isinstance(rows, int):
            indexed[(kernel, rows)] = row
    return indexed


def source_name(row: dict[str, Any], source_by_kernel: dict[str, Path]) -> str:
    kernel = row["kernel"]
    return source_by_kernel[kernel].name


def component_row(
    source_by_kernel: dict[str, Path],
    kernel: str,
    label: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "benchmark": "phase0_layer_sweep_component",
        "component": label,
        "kernel": kernel,
        "rows": row["rows"],
        "avg_ms": avg_ms(row),
        "source_artifact": source_name(row, source_by_kernel),
        "status": "ok",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Phase0a Step 7 layer-sweep artifact from primitive benchmark rows."
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=repo_root() / "reports/phase0_artifacts/benchmarks",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        help="Output prefix without extension. Defaults to layer_sweep_synthetic_<timestamp>.",
    )
    args = parser.parse_args()

    benchmark_dir = args.benchmark_dir
    triton_path = latest_one(benchmark_dir, "triton_swap_full_dims_*.jsonl")
    cublas_path = latest_one(benchmark_dir, "cublas_linear_full_dims_*.jsonl")
    cub_router_path = latest_one(benchmark_dir, "cub_router_topk_full_dims_*.jsonl")
    mxfp4_path = latest_one(benchmark_dir, "mxfp4_kv_pack_*.jsonl")
    primitive_path = latest_one(benchmark_dir, "layer_sweep_primitives_*.jsonl")
    scheduler_path = latest_one(benchmark_dir, "scheduler_smoke_*.json")
    replay_path = latest_optional(benchmark_dir, "phase0_layer_sweep_replay_20*.jsonl")
    scheduler_ms = scheduler_overhead_ms(scheduler_path)
    replay_by_kernel = replay_rows(replay_path)

    indexes = [
        ok_kernel_rows(path)
        for path in [primitive_path, triton_path, cublas_path, cub_router_path, mxfp4_path]
    ]
    source_by_kernel = {
        "rmsnorm_bf16": primitive_path,
        "residual_add_bf16": primitive_path,
        "linear_bf16_cublas": primitive_path,
        "mla_rope_attention_bf16": primitive_path,
        "triton_silu_gated_mlp_rows_bf16_graph": triton_path,
        "triton_router_topk_bf16_graph": cub_router_path,
        "triton_lm_head_sample_topk_topp_bf16_graph": triton_path,
        "mla_kv_pack_mxfp4_ds_mla": mxfp4_path,
        "nvfp4_route_bf16_staged_accumulate_pack": primitive_path,
        "scheduler_admission_overhead": scheduler_path,
    }

    rows_out: list[dict[str, Any]] = []
    dense_totals: dict[int, float] = {}
    sparse_totals: dict[int, float] = {}
    lm_head_totals: dict[int, float] = {}

    for rows in ROWS:
        rmsnorm = require_row(indexes, "rmsnorm_bf16", rows)
        residual = require_row(indexes, "residual_add_bf16", rows)
        cublas_linear = require_row(indexes, "linear_bf16_cublas", rows)
        mla_attention = require_row(indexes, "mla_rope_attention_bf16", rows)
        dense_mlp = require_row(indexes, "triton_silu_gated_mlp_rows_bf16_graph", rows)
        router = require_row(indexes, "triton_router_topk_bf16_graph", rows)
        mxfp4_pack = require_row(indexes, "mla_kv_pack_mxfp4_ds_mla", rows)
        sparse_expert_mlp = require_row(
            indexes, "nvfp4_route_bf16_staged_accumulate_pack", rows
        )
        lm_head = require_row(indexes, "triton_lm_head_sample_topk_topp_bf16_graph", rows)

        for kernel, label, row in [
            ("rmsnorm_bf16", "rmsnorm", rmsnorm),
            ("residual_add_bf16", "residual_add", residual),
            ("linear_bf16_cublas", "cublas_linear_6144x6144", cublas_linear),
            ("mla_rope_attention_bf16", "mla_rope_attention", mla_attention),
            ("triton_silu_gated_mlp_rows_bf16_graph", "triton_dense_mlp", dense_mlp),
            ("triton_router_topk_bf16_graph", "triton_router_topk", router),
            ("mla_kv_pack_mxfp4_ds_mla", "mxfp4_kv_pack", mxfp4_pack),
            (
                "nvfp4_route_bf16_staged_accumulate_pack",
                "local_nvfp4_sparse_expert_mlp",
                sparse_expert_mlp,
            ),
            (
                "scheduler_admission_overhead",
                "scheduler_admission",
                {
                    "kernel": "scheduler_admission_overhead",
                    "rows": rows,
                    "avg_ms": scheduler_ms,
                },
            ),
            ("triton_lm_head_sample_topk_topp_bf16_graph", "triton_lm_head_sampling", lm_head),
        ]:
            rows_out.append(component_row(source_by_kernel, kernel, label, row))

        dense_components = {
            "rmsnorm_x2_ms": 2.0 * avg_ms(rmsnorm),
            "attention_linear_cublas_x4_ms": 4.0 * avg_ms(cublas_linear),
            "mla_rope_attention_ms": avg_ms(mla_attention),
            "mxfp4_kv_pack_ms": avg_ms(mxfp4_pack),
            "triton_dense_mlp_ms": avg_ms(dense_mlp),
            "residual_add_x2_ms": 2.0 * avg_ms(residual),
            "scheduler_admission_overhead_ms": scheduler_ms,
        }
        dense_total = sum(dense_components.values())
        dense_totals[rows] = dense_total
        rows_out.append(
            {
                "benchmark": "phase0_layer_sweep_dense_layer_synthetic",
                "rows": rows,
                "avg_ms": dense_total,
                "layer_count": 1,
                "glm52_layer_range": "0..2",
                "coverage": "synthetic lower-bound over new local primitives",
                "missing_components": [],
                "components": dense_components,
                "status": "partial_lower_bound",
            }
        )

        sparse_components = {
            "rmsnorm_x2_ms": 2.0 * avg_ms(rmsnorm),
            "mla_rope_attention_ms": avg_ms(mla_attention),
            "triton_router_topk_ms": avg_ms(router),
            "mxfp4_kv_pack_ms": avg_ms(mxfp4_pack),
            "local_nvfp4_sparse_expert_mlp_ms": avg_ms(sparse_expert_mlp),
            "residual_add_x2_ms": 2.0 * avg_ms(residual),
            "scheduler_admission_overhead_ms": scheduler_ms,
        }
        sparse_total = sum(sparse_components.values())
        sparse_totals[rows] = sparse_total
        rows_out.append(
            {
                "benchmark": "phase0_layer_sweep_sparse_layer_synthetic",
                "rows": rows,
                "avg_ms": sparse_total,
                "layer_count": 1,
                "glm52_layer_range": "3..77",
                "coverage": "synthetic lower-bound over new local primitives",
                "missing_components": [],
                "components": sparse_components,
                "status": "partial_lower_bound",
            }
        )

        lm_head_totals[rows] = avg_ms(lm_head)
        full_components = {
            "dense_layers_0_2_ms": 3.0 * dense_total,
            "sparse_layers_3_77_ms": 75.0 * sparse_total,
            "terminal_lm_head_sampling_ms": avg_ms(lm_head),
        }
        full_total = sum(full_components.values())
        rows_out.append(
            {
                "benchmark": "phase0_layer_sweep_full_78_layer_coordinator_local_synthetic",
                "rows": rows,
                "avg_ms": full_total,
                "dense_layers": 3,
                "sparse_layers": 75,
                "baseline_decode_tps": BASELINE_DECODE_TPS,
                "baseline_decode_ms": BASELINE_DECODE_MS,
                "lower_bound_speedup_vs_phase0_baseline": BASELINE_DECODE_MS / full_total,
                "coverage": "coordinator-local synthetic lower-bound; not a live full-model decode",
                "missing_components": ["real_weight_residency_effects"],
                "components": full_components,
                "status": "partial_lower_bound",
            }
        )

    replay_complete = all(
        replay_by_kernel.get((kernel, rows)) is not None
        for rows in ROWS
        for kernel in [
            "phase0_dense_layer_replay",
            "phase0_sparse_layer_replay",
            "phase0_full_78_layer_coordinator_local_replay",
        ]
    )
    if replay_complete:
        for rows in ROWS:
            dense_replay = replay_by_kernel[("phase0_dense_layer_replay", rows)]
            sparse_replay = replay_by_kernel[("phase0_sparse_layer_replay", rows)]
            full_replay = replay_by_kernel[
                ("phase0_full_78_layer_coordinator_local_replay", rows)
            ]
            rows_out.extend(
                [
                    {
                        "benchmark": "phase0_layer_sweep_dense_layer_actual_replay",
                        "rows": rows,
                        "avg_ms": avg_ms(dense_replay),
                        "source_artifact": replay_path.name if replay_path else None,
                        "status": "ok",
                        "coverage": dense_replay.get("scope"),
                        "components": dense_replay.get("components", {}),
                    },
                    {
                        "benchmark": "phase0_layer_sweep_sparse_layer_actual_replay",
                        "rows": rows,
                        "avg_ms": avg_ms(sparse_replay),
                        "source_artifact": replay_path.name if replay_path else None,
                        "status": "ok",
                        "coverage": sparse_replay.get("scope"),
                        "components": sparse_replay.get("components", {}),
                    },
                    {
                        "benchmark": "phase0_layer_sweep_full_78_layer_coordinator_local_actual_replay",
                        "rows": rows,
                        "avg_ms": avg_ms(full_replay),
                        "source_artifact": replay_path.name if replay_path else None,
                        "status": "ok",
                        "coverage": full_replay.get("scope"),
                        "baseline_decode_tps": full_replay.get("baseline_decode_tps"),
                        "baseline_decode_ms": full_replay.get("baseline_decode_ms"),
                        "speedup_vs_phase0_baseline": full_replay.get(
                            "speedup_vs_phase0_baseline"
                        ),
                        "decode_tokens_per_second_equivalent": full_replay.get(
                            "decode_tokens_per_second_equivalent"
                        ),
                        "components": full_replay.get("components", {}),
                    },
                ]
            )
        rows_out.append(
            {
                "benchmark": "phase0_layer_sweep_step7_status",
                "status": "ok",
                "source_artifact": replay_path.name if replay_path else None,
                "coverage": [
                    "single dense layer actual coordinator-local synthetic replay",
                    "single sparse layer actual coordinator-local synthetic replay",
                    "full 78-layer coordinator-local synthetic replay",
                    "per-component lower-bound breakdown vs phase0 1.66 TPS baseline",
                ],
                "remaining_caveat": "Replay uses synthetic resident buffers and does not load real checkpoint weights.",
            }
        )
    else:
        rows_out.append(
            {
                "benchmark": "phase0_layer_sweep_gap",
                "status": "incomplete",
                "reason": "The sweep includes synthetic MLA/RoPE attention-body timing, local synthetic NVFP4 routed sparse-MLP replacement timing, and measured scheduler-admission overhead, but no complete actual full coordinator-local 78-layer replay artifact was found.",
                "required_next_evidence": [
                    "single dense layer actual replay",
                    "single sparse layer actual replay including attention plus router plus residual plus local expert replacement",
                    "full 78-layer coordinator-local replay with synthetic weights",
                ],
            }
        )

    if args.out_prefix is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_prefix = benchmark_dir / f"layer_sweep_synthetic_{timestamp}"
    else:
        out_prefix = args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_prefix.with_suffix(".jsonl")
    md_path = out_prefix.with_suffix(".md")

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    write_markdown(
        md_path,
        rows_out,
        [primitive_path, triton_path, cublas_path, cub_router_path, mxfp4_path, scheduler_path],
        replay_path,
    )
    print(jsonl_path)
    print(md_path)
    return 0


def write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    sources: list[Path],
    replay_path: Path | None,
) -> None:
    def selected(benchmark: str) -> list[dict[str, Any]]:
        return [row for row in rows if row.get("benchmark") == benchmark]

    with path.open("w", encoding="utf-8") as f:
        f.write("# Phase0a Layer Sweep\n\n")
        f.write("This artifact aggregates saved full-dimension CUDA primitive benchmarks and, when present, the opt-in Step 7 coordinator-local replay artifact. ")
        f.write("Actual replay rows use synthetic resident buffers and do not load real checkpoint weights or use Spark.\n\n")
        f.write("## Source Artifacts\n\n")
        for source in sources:
            f.write(f"- `{source.name}`\n")
        if replay_path is not None:
            f.write(f"- `{replay_path.name}`\n")

        f.write("\n## Per-Component Rows\n\n")
        f.write("| rows | rmsnorm | cublas linear | MLA/RoPE attention | Triton dense MLP | Triton router | MXFP4 KV pack | local NVFP4 sparse MLP | scheduler admission | LM head sampling |\n")
        f.write("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        components = selected("phase0_layer_sweep_component")
        for row_count in ROWS:
            by_component = {
                row["component"]: row["avg_ms"]
                for row in components
                if row.get("rows") == row_count
            }
            f.write(
                f"| {row_count} | {by_component['rmsnorm']:.6f} | "
                f"{by_component['cublas_linear_6144x6144']:.6f} | "
                f"{by_component['mla_rope_attention']:.6f} | "
                f"{by_component['triton_dense_mlp']:.6f} | "
                f"{by_component['triton_router_topk']:.6f} | "
                f"{by_component['mxfp4_kv_pack']:.6f} | "
                f"{by_component['local_nvfp4_sparse_expert_mlp']:.6f} | "
                f"{by_component['scheduler_admission']:.6f} | "
                f"{by_component['triton_lm_head_sampling']:.6f} |\n"
            )

        f.write("\n## Synthetic Layer Totals\n\n")
        f.write("| rows | dense layer lower-bound ms | sparse layer lower-bound ms | full 78-layer lower-bound ms | lower-bound speedup vs 1.66 TPS baseline |\n")
        f.write("| ---: | ---: | ---: | ---: | ---: |\n")
        dense = {row["rows"]: row for row in selected("phase0_layer_sweep_dense_layer_synthetic")}
        sparse = {row["rows"]: row for row in selected("phase0_layer_sweep_sparse_layer_synthetic")}
        full = {row["rows"]: row for row in selected("phase0_layer_sweep_full_78_layer_coordinator_local_synthetic")}
        for row_count in ROWS:
            f.write(
                f"| {row_count} | {dense[row_count]['avg_ms']:.6f} | "
                f"{sparse[row_count]['avg_ms']:.6f} | "
                f"{full[row_count]['avg_ms']:.6f} | "
                f"{full[row_count]['lower_bound_speedup_vs_phase0_baseline']:.1f}x |\n"
            )

        replay_status = selected("phase0_layer_sweep_step7_status")
        if replay_status:
            f.write("\n## Actual Coordinator-Local Replay\n\n")
            f.write("| rows | dense actual replay ms | sparse actual replay ms | full 78-layer actual replay ms | equivalent decode TPS | speedup vs 1.66 TPS baseline |\n")
            f.write("| ---: | ---: | ---: | ---: | ---: | ---: |\n")
            dense_actual = {
                row["rows"]: row
                for row in selected("phase0_layer_sweep_dense_layer_actual_replay")
            }
            sparse_actual = {
                row["rows"]: row
                for row in selected("phase0_layer_sweep_sparse_layer_actual_replay")
            }
            full_actual = {
                row["rows"]: row
                for row in selected(
                    "phase0_layer_sweep_full_78_layer_coordinator_local_actual_replay"
                )
            }
            for row_count in ROWS:
                full_row = full_actual[row_count]
                f.write(
                    f"| {row_count} | {dense_actual[row_count]['avg_ms']:.6f} | "
                    f"{sparse_actual[row_count]['avg_ms']:.6f} | "
                    f"{full_row['avg_ms']:.6f} | "
                    f"{full_row['decode_tokens_per_second_equivalent']:.6f} | "
                    f"{full_row['speedup_vs_phase0_baseline']:.2f}x |\n"
                )

        f.write("\n## Coverage\n\n")
        f.write("- Dense layer total includes two RMSNorms, four 6144x6144 cuBLAS linears, synthetic MLA/RoPE attention body, MXFP4 KV pack, Triton dense MLP, two residual adds, and measured scheduler-admission overhead.\n")
        f.write("- Sparse layer total includes two RMSNorms, synthetic MLA/RoPE attention body, Triton router top-k, MXFP4 KV pack, local synthetic NVFP4 routed sparse MLP, two residual adds, and measured scheduler-admission overhead.\n")
        if replay_status:
            f.write("- Actual replay rows cover one dense layer, one sparse layer, and full 78-layer coordinator-local replay with the same synthetic resident buffers.\n")
            f.write("- Treat both full 78-layer tables as synthetic coordinator-local evidence, not a live full-model decode result with real checkpoint weights.\n")
        else:
            f.write("- Missing from this artifact: real weight residency effects and the actual full 78-layer coordinator-local replay.\n")
            f.write("- Treat the full 78-layer number as a local synthetic lower bound, not a live full-model decode result.\n")


if __name__ == "__main__":
    raise SystemExit(main())
