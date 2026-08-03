#!/usr/bin/env python3
"""Join hybrid MoE replay arms and project the concurrent critical path."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TOP_K = 8
SPARSE_LAYERS = 75
CURRENT_SPARK_SHARD_BYTES = 101_921_587_200
EXPERTS = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--spark", type=Path, action="append", required=True)
    parser.add_argument("--local", type=Path, action="append", required=True)
    parser.add_argument("--spark-upper-bound", type=Path)
    parser.add_argument("--route-bank", type=Path, required=True)
    parser.add_argument("--live-benchmark", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--remote-fixed-ms", type=float, default=0.128)
    parser.add_argument("--local-path-ms", type=float, default=0.025)
    parser.add_argument("--merge-ms", type=float, default=0.010)
    return parser.parse_args()


def load_artifact(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = None
    measurements = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record.get("record") == "manifest":
                manifest = record
            elif record.get("record") == "measurement":
                measurements[str(record["case_id"])] = record
    if manifest is None or not measurements:
        raise ValueError(f"artifact is missing manifest or measurements: {path}")
    return manifest, measurements


def median_by_m(measurements: dict[str, dict[str, Any]]) -> dict[int, float]:
    rows: dict[int, list[float]] = defaultdict(list)
    for record in measurements.values():
        rows[int(record["physical_m"])].append(float(record["median_ms"]))
    return {physical_m: statistics.median(values) for physical_m, values in rows.items()}


def select_schedule_by_m(
    artifacts: list[tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]],
) -> tuple[dict[int, tuple[Path, dict[str, dict[str, Any]]]], dict[int, dict[str, Any]]]:
    candidates: dict[int, list[tuple[float, Path, dict[str, Any], dict[str, dict[str, Any]]]]] = defaultdict(list)
    for path, manifest, measurements in artifacts:
        for physical_m, median_ms in median_by_m(measurements).items():
            candidates[physical_m].append((median_ms, path, manifest, measurements))
    selected = {}
    policy = {}
    for physical_m, options in candidates.items():
        median_ms, path, manifest, measurements = min(options, key=lambda item: item[0])
        selected[physical_m] = (path, measurements)
        policy[physical_m] = {
            "source": str(path),
            "schedule": manifest.get("small_m_kernel"),
            "median_ms": median_ms,
            "alternatives": [
                {
                    "source": str(option[1]),
                    "schedule": option[2].get("small_m_kernel"),
                    "median_ms": option[0],
                }
                for option in sorted(options, key=lambda item: item[0])
            ],
        }
    return selected, policy


def route_weights(path: Path, supported_ms: set[int]) -> tuple[dict[int, float], Counter[int]]:
    counts: Counter[int] = Counter()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if (
                record.get("record") == "fragment"
                and int(record.get("layer_id", -1)) == 3
                and record.get("case") not in ("count", "repeat")
            ):
                physical_m = int(record["physical_m"])
                if physical_m in supported_ms:
                    counts[physical_m] += 1
    total = sum(counts.values())
    if not total:
        raise ValueError("route bank contains no supported semantic cycles")
    return {physical_m: count / total for physical_m, count in counts.items()}, counts


def load_live_benchmark(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record.get("case") and record.get("verify_cycles"):
                rows.append(record)
    if not rows:
        raise ValueError(f"live benchmark contains no decode samples: {path}")
    cycles = sum(int(row["verify_cycles"]) for row in rows)
    decode_ms = sum(float(row["decode_ms"]) for row in rows)
    tokens = sum(int(row["completion_tokens"]) for row in rows)
    return {
        "path": str(path),
        "records": len(rows),
        "verify_cycles": cycles,
        "decode_ms": decode_ms,
        "completion_tokens": tokens,
        "mean_cycle_ms": decode_ms / cycles,
        "aggregate_tps": tokens / (decode_ms / 1000.0),
    }


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def weighted_sum(rows_by_m: dict[int, dict[str, Any]], weights: dict[int, float], key: str) -> float:
    return sum(weights[physical_m] * float(rows_by_m[physical_m][key]) for physical_m in weights)


def main() -> None:
    args = parse_args()
    paths = [args.baseline, *args.spark, *args.local, args.route_bank]
    if args.spark_upper_bound is not None:
        paths.append(args.spark_upper_bound)
    if args.live_benchmark is not None:
        paths.append(args.live_benchmark)
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"required artifact does not exist: {path}")
    if args.output_json.exists() or args.output_markdown.exists():
        raise SystemExit("refusing to overwrite an output artifact")
    if min(args.remote_fixed_ms, args.local_path_ms, args.merge_ms) < 0:
        raise SystemExit("overhead charges must be nonnegative")

    baseline_manifest, baseline = load_artifact(args.baseline)
    spark_artifacts = []
    local_artifacts = []
    for path in args.spark:
        manifest, measurements = load_artifact(path)
        if not manifest.get("valid_output", False):
            raise ValueError(f"primary Spark artifact has invalid output: {path}")
        spark_artifacts.append((path, manifest, measurements))
    for path in args.local:
        manifest, measurements = load_artifact(path)
        if not manifest.get("valid_output", False):
            raise ValueError(f"primary local artifact has invalid output: {path}")
        local_artifacts.append((path, manifest, measurements))

    placement_hashes = {
        manifest.get("placement", {}).get("placement_sha256")
        for _, manifest, _ in [*spark_artifacts, *local_artifacts]
    }
    if len(placement_hashes) != 1:
        raise ValueError(f"candidate placement hashes disagree: {placement_hashes}")
    revisions = {
        manifest.get("sparkinfer_revision")
        for _, manifest, _ in [*spark_artifacts, *local_artifacts]
    }
    revisions.add(baseline_manifest.get("sparkinfer_revision"))
    if len(revisions) != 1:
        raise ValueError(f"SparkInfer revisions disagree: {revisions}")

    spark_selected, spark_policy = select_schedule_by_m(spark_artifacts)
    local_selected, local_policy = select_schedule_by_m(local_artifacts)
    supported_ms = set(median_by_m(baseline)) & set(spark_selected) & set(local_selected)
    weights, route_counts = route_weights(args.route_bank, supported_ms)
    supported_ms = set(weights)

    case_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for case_id, base in baseline.items():
        physical_m = int(base["physical_m"])
        if physical_m not in supported_ms:
            continue
        spark = spark_selected[physical_m][1].get(case_id)
        local = local_selected[physical_m][1].get(case_id)
        if spark is None or local is None:
            raise ValueError(f"candidate artifact is missing {case_id}")
        baseline_ms = float(base["median_ms"])
        spark_ms = float(spark["median_ms"])
        local_ms = float(local["median_ms"])
        kernel_critical_ms = max(spark_ms, local_ms)
        baseline_dispatch_ms = baseline_ms + args.remote_fixed_ms
        spark_dispatch_ms = spark_ms + args.remote_fixed_ms
        local_branch_ms = local_ms + args.local_path_ms
        hybrid_dispatch_ms = max(spark_dispatch_ms, local_branch_ms) + args.merge_ms
        case_rows[physical_m].append(
            {
                "baseline_kernel_ms": baseline_ms,
                "spark_kernel_ms": spark_ms,
                "local_kernel_ms": local_ms,
                "kernel_critical_ms": kernel_critical_ms,
                "kernel_speedup": baseline_ms / kernel_critical_ms,
                "baseline_dispatch_ms": baseline_dispatch_ms,
                "hybrid_dispatch_ms": hybrid_dispatch_ms,
                "dispatch_speedup": baseline_dispatch_ms / hybrid_dispatch_ms,
                "long_arm": "spark" if spark_dispatch_ms >= local_branch_ms else "local",
                "spark_route_fraction": float(spark["logical_routes"])
                / (physical_m * TOP_K),
                "local_route_fraction": float(local["logical_routes"])
                / (physical_m * TOP_K),
            }
        )

    summaries = {}
    for physical_m in sorted(case_rows):
        rows = case_rows[physical_m]
        summaries[physical_m] = {
            "physical_m": physical_m,
            "route_cycles": route_counts[physical_m],
            "route_weight": weights[physical_m],
            "cases": len(rows),
            "baseline_kernel_ms": statistics.median(row["baseline_kernel_ms"] for row in rows),
            "spark_kernel_ms": statistics.median(row["spark_kernel_ms"] for row in rows),
            "local_kernel_ms": statistics.median(row["local_kernel_ms"] for row in rows),
            "critical_kernel_ms": statistics.median(row["kernel_critical_ms"] for row in rows),
            "kernel_speedup_median": statistics.median(row["kernel_speedup"] for row in rows),
            "kernel_speedup_p05": percentile([row["kernel_speedup"] for row in rows], 0.05),
            "baseline_dispatch_ms": statistics.median(row["baseline_dispatch_ms"] for row in rows),
            "hybrid_dispatch_ms": statistics.median(row["hybrid_dispatch_ms"] for row in rows),
            "dispatch_speedup_median": statistics.median(row["dispatch_speedup"] for row in rows),
            "dispatch_speedup_p05": percentile([row["dispatch_speedup"] for row in rows], 0.05),
            "spark_long_cases": sum(row["long_arm"] == "spark" for row in rows),
            "local_long_cases": sum(row["long_arm"] == "local" for row in rows),
            "mean_spark_route_fraction": statistics.mean(row["spark_route_fraction"] for row in rows),
            "mean_local_route_fraction": statistics.mean(row["local_route_fraction"] for row in rows),
            "spark_schedule": spark_policy[physical_m]["schedule"],
            "local_schedule": local_policy[physical_m]["schedule"],
        }

    weighted = {
        "baseline_kernel_ms_per_layer": weighted_sum(summaries, weights, "baseline_kernel_ms"),
        "critical_kernel_ms_per_layer": weighted_sum(summaries, weights, "critical_kernel_ms"),
        "baseline_dispatch_ms_per_layer": weighted_sum(summaries, weights, "baseline_dispatch_ms"),
        "hybrid_dispatch_ms_per_layer": weighted_sum(summaries, weights, "hybrid_dispatch_ms"),
        "spark_route_fraction": weighted_sum(summaries, weights, "mean_spark_route_fraction"),
        "local_route_fraction": weighted_sum(summaries, weights, "mean_local_route_fraction"),
    }
    weighted["kernel_speedup"] = (
        weighted["baseline_kernel_ms_per_layer"]
        / weighted["critical_kernel_ms_per_layer"]
    )
    weighted["dispatch_speedup"] = (
        weighted["baseline_dispatch_ms_per_layer"]
        / weighted["hybrid_dispatch_ms_per_layer"]
    )
    weighted["saved_ms_per_sparse_layer"] = (
        weighted["baseline_dispatch_ms_per_layer"]
        - weighted["hybrid_dispatch_ms_per_layer"]
    )
    weighted["saved_ms_per_verify_cycle"] = (
        SPARSE_LAYERS * weighted["saved_ms_per_sparse_layer"]
    )

    live = load_live_benchmark(args.live_benchmark)
    if live is not None:
        projected_cycle_ms = live["mean_cycle_ms"] - weighted["saved_ms_per_verify_cycle"]
        live["projected_cycle_ms"] = projected_cycle_ms
        live["projected_tps"] = live["aggregate_tps"] * live["mean_cycle_ms"] / projected_cycle_ms
        live["projected_tps_gain"] = live["projected_tps"] / live["aggregate_tps"] - 1.0

    upper_bound = None
    if args.spark_upper_bound is not None:
        upper_manifest, upper = load_artifact(args.spark_upper_bound)
        if upper_manifest.get("valid_output", True):
            raise ValueError("Spark upper-bound artifact unexpectedly claims valid output")
        deltas = []
        for case_id, record in upper.items():
            physical_m = int(record["physical_m"])
            if physical_m not in supported_ms:
                continue
            valid = spark_selected[physical_m][1][case_id]
            deltas.append(float(valid["median_ms"]) - float(record["median_ms"]))
        upper_bound = {
            "source": str(args.spark_upper_bound),
            "median_saved_ms": statistics.median(deltas),
            "mean_saved_ms": statistics.mean(deltas),
            "note": "invalid-output ceiling for a future skip-aware epilogue",
        }

    local_experts = int(next(iter(spark_artifacts))[1]["placement"]["local_experts_per_layer"])
    resident_bytes = {
        "local_gpu_full_experts": CURRENT_SPARK_SHARD_BYTES * 4 * local_experts // EXPERTS,
        "spark_tp4_complement_per_node": CURRENT_SPARK_SHARD_BYTES * (EXPERTS - local_experts) // EXPERTS,
        "current_spark_tp4_per_node": CURRENT_SPARK_SHARD_BYTES,
    }
    report = {
        "schema": "glmrt-local-spark-hybrid-critical-path-analysis-v1",
        "decision": "proceed-to-transport-prototype",
        "scope": "held-out semantic route replay; kernel timing plus conservative fixed overhead model",
        "sparkinfer_revision": next(iter(revisions)),
        "placement_sha256": next(iter(placement_hashes)),
        "charges_ms": {
            "remote_fixed_per_layer": args.remote_fixed_ms,
            "local_p2p_and_path_per_layer": args.local_path_ms,
            "final_merge_per_layer": args.merge_ms,
        },
        "route_counts": dict(sorted(route_counts.items())),
        "spark_policy": spark_policy,
        "local_policy": local_policy,
        "summaries": [summaries[m] for m in sorted(summaries)],
        "weighted": weighted,
        "live_projection": live,
        "skip_aware_epilogue_upper_bound": upper_bound,
        "resident_bytes": resident_bytes,
        "limitations": [
            "No production route split, RDMA payload reduction, P2P handoff, or numerical partial merge is implemented yet.",
            "Remote fixed cost is charged unchanged, so no credit is taken for smaller requests or responses.",
            "Kernel arms ran concurrently on separate physical GPUs during qualification, but end-to-end transport was modeled rather than replayed through serving.",
            "The live TPS projection assumes unchanged model outputs, draft acceptance, and non-expert work.",
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Local-expert / Spark-complement critical-path replay",
        "",
        "## Decision",
        "",
        "Proceed to a bounded transport prototype. The compact Spark complement remains the long branch, but it contracts enough to produce a material transport-conservative decode projection. This is not yet an integration result.",
        "",
        "## Weighted decode result",
        "",
        "| Metric | Current | Hybrid | Speedup |",
        "| --- | ---: | ---: | ---: |",
        f"| Kernel critical path per sparse layer | {weighted['baseline_kernel_ms_per_layer']:.3f} ms | {weighted['critical_kernel_ms_per_layer']:.3f} ms | {weighted['kernel_speedup']:.2f}x |",
        f"| Transport-conservative dispatch per sparse layer | {weighted['baseline_dispatch_ms_per_layer']:.3f} ms | {weighted['hybrid_dispatch_ms_per_layer']:.3f} ms | {weighted['dispatch_speedup']:.2f}x |",
        f"| Saved time per 75-layer verify cycle | — | {weighted['saved_ms_per_verify_cycle']:.2f} ms | — |",
    ]
    if live is not None:
        lines.extend(
            [
                f"| 400 W weighted decode projection | {live['aggregate_tps']:.3f} TPS | {live['projected_tps']:.3f} TPS | +{live['projected_tps_gain'] * 100:.1f}% |",
            ]
        )
    lines.extend(
        [
            "",
            f"The retained semantic mix sends {weighted['local_route_fraction'] * 100:.1f}% of routes to GPU1 and {weighted['spark_route_fraction'] * 100:.1f}% to the four-Spark TP4 complement. The model charges the full historical 0.128 ms/layer remote fixed cost to both current and hybrid paths, 0.025 ms/layer for GPU P2P/path overhead, and 0.010 ms/layer for the final merge.",
            "",
            "The GB10 and RTX arms use four held-out chains and three rotated sparse layers per M, nine CUDA-event graph replays per case, and SparkInfer 6920e89. The RTX measurements ran on otherwise-idle GPU1 at the current 400 W cap while the paired Spark arm ran concurrently on ostrich. The selected complement policy is direct at M=2--3 and grouped-wide from M=4 upward; GPU1 uses direct at M=2 and grouped-wide above it.",
            "",
            "## Physical-M curve",
            "",
            "| M | Mix | Current kernel | Spark compact | RTX local | Conservative dispatch speedup | Long arm |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for physical_m in sorted(summaries):
        row = summaries[physical_m]
        long_arm = "Spark" if row["spark_long_cases"] >= row["local_long_cases"] else "RTX"
        lines.append(
            f"| {physical_m} | {row['route_weight'] * 100:.1f}% | {row['baseline_kernel_ms']:.3f} | {row['spark_kernel_ms']:.3f} | {row['local_kernel_ms']:.3f} | {(row['dispatch_speedup_median'] - 1) * 100:.1f}% | {long_arm} ({row['spark_long_cases']}/{row['cases']} Spark-long) |"
        )
    lines.extend(
        [
            "",
            "## Residency and implementation gate",
            "",
            f"The proposed non-duplicated layout uses {resident_bytes['local_gpu_full_experts'] / 2**30:.2f} GiB on GPU1 and {resident_bytes['spark_tp4_complement_per_node'] / 2**30:.2f} GiB per Spark, versus {resident_bytes['current_spark_tp4_per_node'] / 2**30:.2f} GiB per Spark today.",
            "",
            "The next prototype must split real router output before dispatch, send only the compact complement to Spark, execute complete selected experts on GPU1, and numerically merge the two weighted partials. It must measure real RDMA/P2P overlap and preserve output parity before any resident-layout or fast-loader migration is accepted.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    if upper_bound is not None:
        lines.extend(
            [
                "",
                f"The invalid no-clear epilogue ceiling saves only {upper_bound['median_saved_ms'] * 1000:.1f} us at the median, so the primary result does not depend on an unimplemented epilogue optimization.",
            ]
        )
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
