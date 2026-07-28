#!/usr/bin/env python3
"""Sweep packed W4A16 AOT bucket boundaries with retained real expert routes."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_ROWS = (
    15,
    16,
    17,
    31,
    32,
    33,
    63,
    64,
    65,
    127,
    128,
    129,
    255,
    256,
    257,
    511,
    512,
    513,
    1023,
    1024,
    1025,
    2047,
    2048,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--native-lib", type=Path, required=True)
    parser.add_argument("--replay-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rows",
        default=",".join(str(rows) for rows in DEFAULT_ROWS),
        help="Comma-separated exact row counts.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=2_026_072_601)
    return parser.parse_args()


def load_chains(path: Path) -> list[dict[str, Any]]:
    chains = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record.get("record") == "chain":
                chains.append(record)
    if not chains:
        raise ValueError(f"replay plan contains no chains: {path}")
    return chains


def layer_for_chain(chain: dict[str, Any], chain_index: int) -> dict[str, Any]:
    layers = chain["layers"]
    # Spread a small retained sample across the beginning, middle, and end of
    # the sparse stack instead of benchmarking layer 3 repeatedly.
    position = (0, len(layers) // 2, len(layers) - 1)[chain_index % 3]
    return layers[position]


def route_counts(layer: dict[str, Any], rows: int) -> list[int]:
    routes = layer["routes"][:rows]
    if len(routes) != rows or any(len(row) != 8 for row in routes):
        raise ValueError(
            f"layer {layer['layer_id']} does not contain {rows} top-8 rows"
        )
    counts = Counter(expert for row in routes for expert in row)
    # Expert identity does not affect the kernel timing. Remap active experts
    # contiguously so the native harness can allocate only the weights touched
    # by this retained route-count distribution.
    return [count for _, count in sorted(counts.items())]


def benchmark_order(rows: list[int], chain_index: int, seed: int) -> list[int]:
    if chain_index % 3 == 0:
        return rows
    if chain_index % 3 == 1:
        return list(reversed(rows))
    shuffled = list(rows)
    random.Random(seed + chain_index).shuffle(shuffled)
    return shuffled


def main() -> None:
    args = parse_args()
    rows = list(dict.fromkeys(int(value) for value in args.rows.split(",") if value))
    if (
        not rows
        or min(rows) < 1
        or max(rows) > 2048
        or min(args.warmup, args.iterations, args.repeats) < 1
    ):
        raise SystemExit("rows must be in 1..2048 and sample counts must be positive")
    for path in (args.harness, args.native_lib, args.replay_plan):
        if not path.is_file():
            raise SystemExit(f"required file does not exist: {path}")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")

    chains = load_chains(args.replay_plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        manifest = {
            "record": "manifest",
            "schema": "glmrt-b12x-w4a16-packed-bucket-sweep-v1",
            "replay_plan": str(args.replay_plan),
            "rows": rows,
            "chains": len(chains),
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "graph_api": "torch",
            "timing": "cuda-event",
            "network_involved": False,
        }
        output.write(json.dumps(manifest, separators=(",", ":")) + "\n")
        output.flush()
        print(json.dumps(manifest, sort_keys=True), flush=True)

        for chain_index, chain in enumerate(chains):
            layer = layer_for_chain(chain, chain_index)
            for active_rows in benchmark_order(rows, chain_index, args.seed):
                counts = route_counts(layer, active_rows)
                command = [
                    sys.executable,
                    str(args.harness),
                    "--native-lib",
                    str(args.native_lib),
                    "--scenario",
                    "prefill",
                    "--prefill-rows",
                    str(active_rows),
                    "--active-experts",
                    str(len(counts)),
                    "--weight-experts",
                    str(len(counts)),
                    "--expert-route-counts",
                    ",".join(str(count) for count in counts),
                    "--warmup",
                    str(args.warmup),
                    "--iterations",
                    str(args.iterations),
                    "--repeats",
                    str(args.repeats),
                    "--seed",
                    str(args.seed + chain_index),
                ]
                completed = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                benchmark = json.loads(completed.stdout.strip().splitlines()[-1])
                record = {
                    "record": "measurement",
                    "chain_id": chain["chain_id"],
                    "chain_index": chain_index,
                    "layer_id": layer["layer_id"],
                    "rows": active_rows,
                    "capacity_rows": benchmark["capacity_rows"],
                    "active_experts": len(counts),
                    "logical_routes": active_rows * 8,
                    "padded_routes": sum(
                        ((count + 31) // 32) * 32 for count in counts
                    ),
                    "median_ms": benchmark["median_ms"],
                    "min_ms": benchmark["min_ms"],
                    "samples_ms": benchmark["samples_ms"],
                    "useful_rows_per_ms": active_rows / benchmark["median_ms"],
                }
                output.write(json.dumps(record, separators=(",", ":")) + "\n")
                output.flush()
                print(json.dumps(record, sort_keys=True), flush=True)
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
