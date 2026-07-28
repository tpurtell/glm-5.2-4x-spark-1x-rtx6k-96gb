#!/usr/bin/env python3
"""Analyze paired coordinator-vs-Spark expert-reduction replay results."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReplayPair:
    chain_id: str
    coordinator_ms: float
    spark_ms: float
    coordinator_first: bool
    coordinator_layer_ms: tuple[float, ...]
    spark_layer_ms: tuple[float, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_726)
    return parser.parse_args()


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a percentile of an empty sample")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def aggregate_speedup(pairs: list[tuple[float, float]]) -> float:
    coordinator = sum(pair[0] for pair in pairs)
    spark = sum(pair[1] for pair in pairs)
    return 100.0 * (coordinator / spark - 1.0)


def aggregate_speedup_or_none(
    pairs: list[tuple[float, float]],
) -> float | None:
    return aggregate_speedup(pairs) if pairs else None


def bootstrap_speedup_ci(
    pairs: list[tuple[float, float]],
    samples: int,
    rng: random.Random,
) -> tuple[float, float]:
    if samples < 1:
        raise ValueError("--bootstrap-samples must be positive")
    estimates = []
    for _ in range(samples):
        resampled = [rng.choice(pairs) for _ in pairs]
        estimates.append(aggregate_speedup(resampled))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def load_measurements(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = None
    complete = False
    measurements = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            record = json.loads(line)
            kind = record.get("record")
            if kind == "manifest":
                if manifest is not None:
                    raise ValueError("result contains multiple manifests")
                manifest = record
            elif kind == "measurement":
                measurements.append(record)
            elif kind == "complete":
                complete = record.get("status") == "complete"
    if manifest is None:
        raise ValueError("result has no manifest")
    if not complete:
        raise ValueError("result is incomplete (missing complete footer)")
    if not measurements:
        raise ValueError("result has no measurements")
    return manifest, measurements


def paired_by_m(
    measurements: list[dict[str, Any]],
) -> dict[int, list[ReplayPair]]:
    by_chain: dict[tuple[int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in measurements:
        key = (int(record["physical_m"]), record["chain_id"])
        path = record["path"]
        if path in by_chain[key]:
            raise ValueError(f"duplicate {path} measurement for M={key[0]} {key[1]}")
        by_chain[key][path] = record

    result: dict[int, list[ReplayPair]] = defaultdict(list)
    for (physical_m, chain_id), paths in by_chain.items():
        expected = {"coordinator", "spark-row-sharded"}
        if set(paths) != expected:
            raise ValueError(
                f"M={physical_m} {chain_id} has paths {sorted(paths)}, "
                f"expected {sorted(expected)}"
            )
        coordinator = paths["coordinator"]
        spark = paths["spark-row-sharded"]
        result[physical_m].append(
            ReplayPair(
                chain_id=chain_id,
                coordinator_ms=float(coordinator["dispatch_ms"]),
                spark_ms=float(spark["dispatch_ms"]),
                coordinator_first=int(coordinator["path_order"]) == 0,
                coordinator_layer_ms=tuple(
                    float(value) for value in coordinator["layer_ms"]
                ),
                spark_layer_ms=tuple(float(value) for value in spark["layer_ms"]),
            )
        )
    return dict(sorted(result.items()))


def summarize(
    manifest: dict[str, Any],
    pairs_by_m: dict[int, list[ReplayPair]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    rows = []
    rng = random.Random(seed)
    first_confident_spark_m = None
    for physical_m, replay_pairs in pairs_by_m.items():
        pairs = [(pair.coordinator_ms, pair.spark_ms) for pair in replay_pairs]
        speedup = aggregate_speedup(pairs)
        ci_low, ci_high = bootstrap_speedup_ci(pairs, bootstrap_samples, rng)
        spark_wins = sum(spark < coordinator for coordinator, spark in pairs)
        paired_speedups = [
            100.0 * (coordinator / spark - 1.0)
            for coordinator, spark in pairs
        ]
        coordinator_first_pairs = [
            (pair.coordinator_ms, pair.spark_ms)
            for pair in replay_pairs
            if pair.coordinator_first
        ]
        spark_first_pairs = [
            (pair.coordinator_ms, pair.spark_ms)
            for pair in replay_pairs
            if not pair.coordinator_first
        ]
        coordinator_layers = [
            value for pair in replay_pairs for value in pair.coordinator_layer_ms
        ]
        spark_layers = [
            value for pair in replay_pairs for value in pair.spark_layer_ms
        ]
        if first_confident_spark_m is None and ci_low > 0.0:
            first_confident_spark_m = physical_m
        rows.append(
            {
                "physical_m": physical_m,
                "pairs": len(pairs),
                "coordinator_chain_ms_median": statistics.median(
                    coordinator for coordinator, _ in pairs
                ),
                "spark_chain_ms_median": statistics.median(
                    spark for _, spark in pairs
                ),
                "coordinator_layer_ms_median": statistics.median(
                    coordinator / 75.0 for coordinator, _ in pairs
                ),
                "spark_layer_ms_median": statistics.median(
                    spark / 75.0 for _, spark in pairs
                ),
                "coordinator_layer_ms_mean": statistics.mean(
                    coordinator / 75.0 for coordinator, _ in pairs
                ),
                "spark_layer_ms_mean": statistics.mean(
                    spark / 75.0 for _, spark in pairs
                ),
                "aggregate_spark_speedup_pct": speedup,
                "paired_speedup_stddev_pct": (
                    statistics.stdev(paired_speedups)
                    if len(paired_speedups) > 1
                    else 0.0
                ),
                "bootstrap_95_ci_pct": [ci_low, ci_high],
                "spark_pair_wins": spark_wins,
                "coordinator_first_spark_speedup_pct": aggregate_speedup_or_none(
                    coordinator_first_pairs
                ),
                "spark_first_spark_speedup_pct": aggregate_speedup_or_none(
                    spark_first_pairs
                ),
                "coordinator_layer_ms_p95": percentile(coordinator_layers, 0.95),
                "spark_layer_ms_p95": percentile(spark_layers, 0.95),
            }
        )
    return {
        "schema": "glmrt-expert-reduction-replay-summary-v1",
        "input_schema": manifest["schema"],
        "cohort": manifest["cohort"],
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "first_tested_confident_spark_m": first_confident_spark_m,
        "rows": rows,
    }


def print_table(summary: dict[str, Any]) -> None:
    print(
        "| M | pairs | coordinator mean ms/layer | Spark mean ms/layer | "
        "Spark speedup | 95% paired bootstrap CI | wins |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary["rows"]:
        ci_low, ci_high = row["bootstrap_95_ci_pct"]
        print(
            f"| {row['physical_m']} | {row['pairs']} | "
            f"{row['coordinator_layer_ms_mean']:.4f} | "
            f"{row['spark_layer_ms_mean']:.4f} | "
            f"{row['aggregate_spark_speedup_pct']:+.2f}% | "
            f"[{ci_low:+.2f}%, {ci_high:+.2f}%] | "
            f"{row['spark_pair_wins']}/{row['pairs']} |"
        )
    first = summary["first_tested_confident_spark_m"]
    print(
        "\nFirst tested M with a positive lower 95% confidence bound: "
        f"{first if first is not None else 'none'}"
    )


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"result does not exist: {args.input}")
    manifest, measurements = load_measurements(args.input)
    pairs = paired_by_m(measurements)
    summary = summarize(
        manifest,
        pairs,
        args.bootstrap_samples,
        args.seed,
    )
    print_table(summary)
    if args.summary is not None:
        if args.summary.exists():
            raise SystemExit(f"refusing to overwrite summary: {args.summary}")
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.summary.with_name(f".{args.summary.name}.tmp")
        temporary.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.summary)


if __name__ == "__main__":
    main()
