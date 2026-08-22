#!/usr/bin/env python3
"""Build the causal route-shape reference used by dSpark cost calibration.

The prefix scheduler must choose M before the target routers run.  This tool
therefore does not try to expose current-wave routes to serving.  It samples
complete real decode trajectories offline and records the route distribution
that the one-dimensional E[T | C,M] capacity curve must average over.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bench_real_full_mtp_acceptance import WEIGHTED_CASE_IDS
from plan_expert_reduction_replay import (
    SPARSE_LAYERS,
    CyclePiece,
    NaturalCycle,
    load_cycles,
    select_pieces,
    weighted_case_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-bank", type=Path, required=True)
    parser.add_argument(
        "--width-corpus",
        type=Path,
        help=(
            "C1 dSpark qualification JSONL used to estimate the joint semantic "
            "case/verification-width distribution"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--max-drafts", type=int, default=7)
    parser.add_argument("--samples-per-cell", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20_260_813)
    return parser.parse_args()


def load_width_distribution(
    path: Path | None, semantic_cases: list[str]
) -> tuple[dict[int, float], dict[int, dict[str, float]], dict[str, Any]]:
    widths = Counter()
    cases_by_width: dict[int, Counter[str]] = defaultdict(Counter)
    if path is not None:
        with path.open(encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                lanes = record.get("lanes")
                observations = lanes if isinstance(lanes, list) else [record]
                for observation in observations:
                    case = observation.get("case")
                    if case not in semantic_cases:
                        continue
                    for draft_length in observation.get("draft_lengths", []):
                        width = 1 + int(draft_length)
                        if 1 <= width <= 8:
                            widths[width] += 1
                            cases_by_width[width][case] += 1

    # A pseudocount makes widths absent from the observed dSpark decisions
    # (most notably the no-draft M=1 case) representable without allowing
    # them to distort widths which have real coverage.
    width_weights = {width: float(widths[width] + 1) for width in range(1, 9)}
    case_weights = {
        width: {
            case: float(cases_by_width[width][case] + 1)
            for case in semantic_cases
        }
        for width in range(1, 9)
    }
    evidence = {
        "path": str(path) if path is not None else None,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
        if path is not None
        else None,
        "observations": sum(widths.values()),
        "width_counts": {str(width): widths[width] for width in range(1, 9)},
        "case_width_counts": {
            case: {
                str(width): cases_by_width[width][case]
                for width in range(1, 9)
            }
            for case in semantic_cases
        },
        "laplace_pseudocount_per_case_width": 1,
    }
    return width_weights, case_weights, evidence


def sample_request_shapes(
    rng: random.Random,
    requests: int,
    target_rows: int,
    semantic_cases: list[str],
    width_weights: dict[int, float],
    case_weights: dict[int, dict[str, float]],
) -> list[tuple[str, int]]:
    width_tuples = [
        widths
        for widths in itertools.product(range(1, 9), repeat=requests)
        if sum(widths) == target_rows
    ]
    if not width_tuples:
        raise ValueError(f"no legal request-width partition for C={requests}, M={target_rows}")
    tuple_weights = [
        math.prod(width_weights[width] for width in widths)
        for widths in width_tuples
    ]
    widths = rng.choices(width_tuples, weights=tuple_weights, k=1)[0]
    return [
        (
            rng.choices(
                semantic_cases,
                weights=[case_weights[width][case] for case in semantic_cases],
                k=1,
            )[0],
            width,
        )
        for width in widths
    ]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: list[int]) -> dict[str, float | int]:
    return {
        "minimum": min(values),
        "p05": percentile([float(value) for value in values], 0.05),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "p95": percentile([float(value) for value in values], 0.95),
        "maximum": max(values),
    }


def rows_for_pieces(pieces: list[CyclePiece], layer_id: int) -> list[tuple[int, ...]]:
    return [
        row
        for piece in pieces
        for row in piece.cycle.routes_by_layer[layer_id][
            piece.row_start : piece.row_start + piece.row_count
        ]
    ]


def wire_cohorts(requests: int) -> list[tuple[int, ...]]:
    # This mirrors real_full_scheduler_execution_for_batched_shapes...:
    # C4/C8 merge adjacent pairs, while C1/C2/C3 dispatch independently.
    if requests in (4, 8):
        return [tuple(range(start, start + 2)) for start in range(0, requests, 2)]
    return [(index,) for index in range(requests)]


def sample_shape(
    rng: random.Random,
    requests: int,
    target_rows: int,
    cycles_by_case: dict[str, list[NaturalCycle]],
    semantic_cases: list[str],
    width_weights: dict[int, float],
    case_weights: dict[int, dict[str, float]],
) -> dict[str, int]:
    request_shapes = sample_request_shapes(
        rng,
        requests,
        target_rows,
        semantic_cases,
        width_weights,
        case_weights,
    )
    request_pieces = []
    for case, row_count in request_shapes:
        request_pieces.append(select_pieces(rng, row_count, cycles_by_case, [case]))

    total_unique = 0
    total_reused = 0
    total_assignments = 0
    total_square_sum = 0
    hottest_expert = 0
    cohort_unique = [0 for _ in wire_cohorts(requests)]
    for layer_id in SPARSE_LAYERS:
        request_rows = [
            rows_for_pieces(pieces, layer_id) for pieces in request_pieces
        ]
        for cohort_index, members in enumerate(wire_cohorts(requests)):
            loads = Counter(
                expert
                for member in members
                for row in request_rows[member]
                for expert in row
            )
            assignments = sum(loads.values())
            unique = len(loads)
            total_assignments += assignments
            total_unique += unique
            total_reused += assignments - unique
            total_square_sum += sum(load * load for load in loads.values())
            hottest_expert = max(hottest_expert, max(loads.values()))
            cohort_unique[cohort_index] += unique
    return {
        "wire_batches": len(wire_cohorts(requests)) * len(SPARSE_LAYERS),
        "route_assignments": total_assignments,
        "unique_experts": total_unique,
        "critical_unique_experts": max(cohort_unique),
        "reused_assignments": total_reused,
        "max_expert_load": hottest_expert,
        "load_square_sum": total_square_sum,
        "request_widths": [width for _, width in request_shapes],
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    bank_manifest, cycles = load_cycles(args.route_bank)
    cycles_by_case: dict[str, list[NaturalCycle]] = defaultdict(list)
    for cycle in cycles:
        cycles_by_case[cycle.case].append(cycle)
    semantic_cases = weighted_case_ids(cycles, WEIGHTED_CASE_IDS)
    width_weights, case_weights, width_evidence = load_width_distribution(
        args.width_corpus, semantic_cases
    )
    rng = random.Random(args.seed)
    cells = {}
    for requests in range(1, args.max_concurrency + 1):
        for target_rows in range(
            requests, requests * (args.max_drafts + 1) + 1
        ):
            samples = [
                sample_shape(
                    rng,
                    requests,
                    target_rows,
                    cycles_by_case,
                    semantic_cases,
                    width_weights,
                    case_weights,
                )
                for _ in range(args.samples_per_cell)
            ]
            cells[f"{requests}:{target_rows}"] = {
                "requests": requests,
                "target_rows": target_rows,
                "wire_cohorts": [list(cohort) for cohort in wire_cohorts(requests)],
                "samples": len(samples),
                "route_shape": {
                    name: summarize([sample[name] for sample in samples])
                    for name in samples[0]
                    if name != "request_widths"
                },
                "request_width_distribution": {
                    str(index): summarize(
                        [sample["request_widths"][index] for sample in samples]
                    )
                    for index in range(requests)
                },
            }
    result = {
        "schema": "glmrt-dspark-route-reference-v1",
        "route_bank": str(args.route_bank),
        "route_bank_schema": bank_manifest["schema"],
        "route_bank_source_trace_log": bank_manifest.get("source_trace_log"),
        "seed": args.seed,
        "samples_per_cell": args.samples_per_cell,
        "max_concurrency": args.max_concurrency,
        "max_drafts": args.max_drafts,
        "semantic_case_weights": semantic_cases,
        "verification_width_evidence": width_evidence,
        "sparse_layers": list(SPARSE_LAYERS),
        "cells": cells,
    }
    canonical = json.dumps(result, ensure_ascii=False, sort_keys=True).encode()
    result["source_sha256"] = hashlib.sha256(canonical).hexdigest()
    return result


def main() -> None:
    args = parse_args()
    if args.max_concurrency < 1 or args.max_drafts < 1:
        raise SystemExit("--max-concurrency and --max-drafts must be positive")
    if args.samples_per_cell < 8:
        raise SystemExit("--samples-per-cell must be at least 8")
    if not args.route_bank.is_file():
        raise SystemExit(f"route bank does not exist: {args.route_bank}")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite route reference: {args.output}")
    result = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_sha256": result["source_sha256"],
                "cells": len(result["cells"]),
                "samples_per_cell": args.samples_per_cell,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
