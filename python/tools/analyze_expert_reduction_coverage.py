#!/usr/bin/env python3
"""Measure reduction-cutoff exposure from real adaptive expert-route traces."""

from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path
from typing import Iterable


DEFAULT_THRESHOLDS = (8, 12, 16, 20, 24, 28, 32, 36, 39, 40, 41)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-bank", type=Path, required=True)
    parser.add_argument(
        "--benchmark-jsonl",
        type=Path,
        help="Optional concurrency benchmark whose lane draft trajectories are analyzed.",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def read_route_bank(path: Path) -> tuple[dict, dict[str, list[tuple[int, ...]]]]:
    manifest = None
    outputs: dict[str, list[tuple[int, ...]]] = collections.defaultdict(list)
    with open_text(path) as source:
        for line in source:
            record = json.loads(line)
            if record["record"] == "manifest":
                manifest = record
            elif record["record"] == "output":
                outputs[record["case"]].append(tuple(record["physical_ms"]))
    if manifest is None:
        raise ValueError("route bank has no manifest")
    if not outputs:
        raise ValueError("route bank has no output records")
    return manifest, dict(outputs)


def normalized(distribution: dict[int, float]) -> dict[int, float]:
    total = sum(distribution.values())
    if total <= 0:
        raise ValueError("empty row distribution")
    return {rows: weight / total for rows, weight in distribution.items()}


def independent_cycle_distribution(
    outputs: dict[str, list[tuple[int, ...]]],
    case_weights: dict[str, float],
) -> dict[int, float]:
    singles: dict[int, float] = collections.defaultdict(float)
    for case, case_weight in case_weights.items():
        trajectories = outputs[case]
        output_weight = case_weight / len(trajectories)
        for trajectory in trajectories:
            for rows in trajectory:
                singles[rows] += output_weight
    singles = normalized(singles)
    pairs: dict[int, float] = collections.defaultdict(float)
    for left_rows, left_weight in singles.items():
        for right_rows, right_weight in singles.items():
            pairs[left_rows + right_rows] += left_weight * right_weight
    return normalized(pairs)


def aligned_trajectory_distribution(
    weighted_trajectories: Iterable[tuple[tuple[int, ...], float]],
) -> dict[int, float]:
    trajectories = list(weighted_trajectories)
    pairs: dict[int, float] = collections.defaultdict(float)
    for left, left_weight in trajectories:
        for right, right_weight in trajectories:
            event_weight = left_weight * right_weight
            for left_rows, right_rows in zip(left, right):
                pairs[left_rows + right_rows] += event_weight
    return normalized(pairs)


def weighted_route_trajectories(
    outputs: dict[str, list[tuple[int, ...]]],
    case_weights: dict[str, float],
) -> list[tuple[tuple[int, ...], float]]:
    weighted = []
    for case, case_weight in case_weights.items():
        trajectories = outputs[case]
        output_weight = case_weight / len(trajectories)
        weighted.extend((trajectory, output_weight) for trajectory in trajectories)
    return weighted


def summarize(distribution: dict[int, float], thresholds: list[int]) -> dict:
    mean_rows = sum(rows * weight for rows, weight in distribution.items())
    threshold_summaries = []
    for threshold in thresholds:
        decision_weight = sum(
            weight for rows, weight in distribution.items() if rows >= threshold
        )
        row_weight = (
            sum(
                rows * weight
                for rows, weight in distribution.items()
                if rows >= threshold
            )
            / mean_rows
        )
        threshold_summaries.append(
            {
                "threshold": threshold,
                "spark_decision_fraction": decision_weight,
                "coordinator_decision_fraction": 1.0 - decision_weight,
                "spark_row_fraction": row_weight,
                "coordinator_row_fraction": 1.0 - row_weight,
            }
        )
    return {
        "min_pair_rows": min(distribution),
        "max_pair_rows": max(distribution),
        "mean_pair_rows": mean_rows,
        "pair_row_distribution": {
            str(rows): distribution[rows] for rows in sorted(distribution)
        },
        "thresholds": threshold_summaries,
    }


def read_benchmark_trajectories(path: Path) -> list[tuple[int, ...]]:
    trajectories = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if "repeat" not in record:
                continue
            for lane in record["lanes"]:
                drafts = lane.get("draft_lengths")
                if drafts:
                    trajectories.append(tuple(int(draft) + 1 for draft in drafts))
    if not trajectories:
        raise ValueError("benchmark has no measured lane draft trajectories")
    return trajectories


def main() -> None:
    args = parse_args()
    manifest, outputs = read_route_bank(args.route_bank)
    normal_cases = manifest["normal_weighted_case_ids"]
    multipliers = manifest.get("collection_multipliers", {})
    semantic_weights = {
        case: float(multipliers.get(case, 1)) for case in normal_cases
    }

    report = {
        "schema": "glmrt-expert-reduction-coverage-v1",
        "route_bank": str(args.route_bank.resolve()),
        "thresholds": args.thresholds,
        "normal_case_weights": semantic_weights,
        "semantic_independent_cycles": summarize(
            independent_cycle_distribution(outputs, semantic_weights),
            args.thresholds,
        ),
        "semantic_aligned_outputs": summarize(
            aligned_trajectory_distribution(
                weighted_route_trajectories(outputs, semantic_weights)
            ),
            args.thresholds,
        ),
        "homogeneous_aligned_cases": {
            case: summarize(
                aligned_trajectory_distribution(
                    weighted_route_trajectories(outputs, {case: 1.0})
                ),
                args.thresholds,
            )
            for case in normal_cases
        },
    }

    diagnostics = manifest.get("diagnostic_case_ids", [])
    if diagnostics:
        diagnostic_weights = {case: 1.0 for case in diagnostics}
        report["diagnostic_independent_cycles"] = summarize(
            independent_cycle_distribution(outputs, diagnostic_weights),
            args.thresholds,
        )

    if args.benchmark_jsonl:
        trajectories = read_benchmark_trajectories(args.benchmark_jsonl)
        trajectory_counts = collections.Counter(trajectories)
        total = sum(trajectory_counts.values())
        report["benchmark_aligned_trajectories"] = {
            "benchmark_jsonl": str(args.benchmark_jsonl.resolve()),
            "measured_lanes": total,
            **summarize(
                aligned_trajectory_distribution(
                    [
                        (trajectory, count / total)
                        for trajectory, count in trajectory_counts.items()
                    ]
                ),
                args.thresholds,
            ),
        }

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
