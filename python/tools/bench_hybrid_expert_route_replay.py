#!/usr/bin/env python3
"""Replay one arm of a local-full-expert / remote-TP4 hybrid MoE split.

The retained route plan is split with one static, per-layer residency map:

* ``baseline`` executes all routes with the current GB10 TP4 shard (I=512);
* ``spark`` executes only the non-local routes with a compact GB10 TP4 shard;
* ``local`` executes the selected routes as complete experts (I=2048) on RTX.

Run the GB10 arms on a Spark and the local arm on the otherwise idle second
coordinator GPU.  The companion analyzer joins records by case ID and charges
``max(spark, local)`` rather than adding the two branches.

This is deliberately a kernel gate.  Route packing, RDMA, P2P, and the final
partial-result merge are outside CUDA graph capture and must be charged by the
analyzer before making an end-to-end projection.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

import bench_tp2_ep2_route_replay as replay


LOCAL_EXPERTS = 56
REMOTE_EXPERTS = replay.EXPERTS - LOCAL_EXPERTS
DEFAULT_MS = tuple(range(1, 17))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-plan", type=Path, required=True)
    parser.add_argument(
        "--placement-route-bank",
        type=Path,
        required=True,
        help=(
            "Natural route bank used only for its semantic physical-M histogram; "
            "expert identities still come exclusively from held-out training chains."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("baseline", "spark", "local"), required=True)
    parser.add_argument(
        "--ms",
        default=",".join(str(value) for value in DEFAULT_MS),
        help="Comma-separated physical M values from the replay plan.",
    )
    parser.add_argument("--local-experts", type=int, default=LOCAL_EXPERTS)
    parser.add_argument("--placement-training-max-m", type=int, default=16)
    parser.add_argument("--chains-per-m", type=int, default=4)
    parser.add_argument("--layers-per-chain", type=int, default=3)
    parser.add_argument("--warmup-rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20_260_801)
    parser.add_argument(
        "--small-m-kernel",
        choices=("direct", "grouped", "grouped-wide"),
        default="grouped-wide",
        help="Compact candidate schedule at M=2..8; M=1 always uses direct fused sum.",
    )
    parser.add_argument(
        "--large-m-block-size",
        choices=(8, 16, 32),
        type=int,
        default=32,
    )
    parser.add_argument(
        "--no-zero-output-upper-bound",
        action="store_true",
        help=(
            "Omit partial-route output clearing. This is an invalid-output upper "
            "bound for a future skip-aware epilogue, not a shippable result."
        ),
    )
    return parser.parse_args()


def make_residency(
    training_chains: list[dict[str, Any]],
    *,
    local_experts: int,
    training_max_m: int,
    natural_m_counts: Counter[int],
) -> tuple[dict[int, tuple[tuple[int, ...], tuple[int, ...]]], dict[str, Any]]:
    if not 1 <= local_experts < replay.EXPERTS:
        raise ValueError(f"local expert count must be in 1..{replay.EXPERTS - 1}")
    eligible = [
        chain
        for chain in training_chains
        if int(chain["physical_m"]) <= training_max_m
    ]
    if not eligible:
        raise ValueError("placement training filter selected no chains")

    chains_by_m = Counter(int(chain["physical_m"]) for chain in eligible)
    effective = [
        chain for chain in eligible if natural_m_counts[int(chain["physical_m"])] > 0
    ]
    if not effective:
        raise ValueError("placement histogram has no overlap with training chains")
    frequency = {layer_id: Counter() for layer_id in replay.SPARSE_LAYERS}
    for chain in effective:
        physical_m = int(chain["physical_m"])
        # Give the complete set of training chains at M a total weight equal to
        # the observed number of natural cycles at M. Dividing by M prevents a
        # rare wide synthetic chain from dominating merely because it has more
        # route rows.
        route_weight = natural_m_counts[physical_m] / (
            chains_by_m[physical_m] * physical_m * replay.TOP_K
        )
        for layer in chain["layers"]:
            counts = frequency[int(layer["layer_id"])]
            for row in layer["routes"]:
                for expert in row:
                    counts[int(expert)] += route_weight

    residents = {}
    training_coverage = []
    placement_digest = hashlib.sha256()
    for layer_id in replay.SPARSE_LAYERS:
        counts = frequency[layer_id]
        ordered = sorted(
            range(replay.EXPERTS),
            key=lambda expert: (-counts[expert], expert),
        )
        local = tuple(sorted(ordered[:local_experts]))
        local_set = set(local)
        remote = tuple(
            expert for expert in range(replay.EXPERTS) if expert not in local_set
        )
        residents[layer_id] = (local, remote)
        local_routes = sum(counts[expert] for expert in local)
        total_routes = sum(counts.values())
        training_coverage.append(local_routes / max(total_routes, 1))
        placement_digest.update(layer_id.to_bytes(2, "little"))
        placement_digest.update(bytes(local))

    return residents, {
        "method": "held-out per-layer top-frequency local residency",
        "training_chains": len(effective),
        "training_max_m": training_max_m,
        "training_weight": "natural semantic cycle histogram, normalized per M row",
        "natural_m_counts": dict(sorted(natural_m_counts.items())),
        "local_experts_per_layer": local_experts,
        "remote_experts_per_layer": replay.EXPERTS - local_experts,
        "mean_training_route_coverage": statistics.mean(training_coverage),
        "min_layer_training_route_coverage": min(training_coverage),
        "max_layer_training_route_coverage": max(training_coverage),
        "placement_sha256": placement_digest.hexdigest(),
    }


def natural_semantic_m_counts(path: Path, training_max_m: int) -> Counter[int]:
    counts: Counter[int] = Counter()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if (
                record.get("record") == "fragment"
                and int(record.get("layer_id", -1)) == replay.SPARSE_LAYERS[0]
                and record.get("case") not in ("count", "repeat")
            ):
                physical_m = int(record["physical_m"])
                if physical_m <= training_max_m:
                    counts[physical_m] += 1
    if not counts:
        raise ValueError(f"no natural semantic cycles found in {path}")
    return counts


def rows_for_arm(
    replay_case: replay.ReplayCase,
    arm: str,
    residents: dict[int, tuple[tuple[int, ...], tuple[int, ...]]],
) -> tuple[tuple[int, ...], ...]:
    if arm == "baseline":
        return replay_case.routes
    resident_index = 0 if arm == "local" else 1
    arm_residents = residents[replay_case.layer_id][resident_index]
    local_id = {expert: index for index, expert in enumerate(arm_residents)}
    return tuple(
        tuple(local_id[expert] for expert in row if expert in local_id)
        for row in replay_case.routes
    )


def summarize(
    cases: list[replay.ReplayCase],
    measurements: list[replay.GraphMeasurement],
) -> list[dict[str, Any]]:
    by_m: dict[int, list[replay.GraphMeasurement]] = {}
    for measurement in measurements:
        by_m.setdefault(measurement.replay_case.physical_m, []).append(measurement)
    records = []
    for physical_m in sorted(by_m):
        rows = by_m[physical_m]
        medians = [row.median_ms for row in rows]
        logical = [
            0 if row.metadata is None else row.metadata.logical_routes for row in rows
        ]
        padded = [
            0 if row.metadata is None else row.metadata.padded_routes for row in rows
        ]
        active = [
            0 if row.metadata is None else row.metadata.active_experts for row in rows
        ]
        full_routes = physical_m * replay.TOP_K
        records.append(
            {
                "record": "summary",
                "physical_m": physical_m,
                "cases": len(rows),
                "median_ms": statistics.median(medians),
                "p05_ms": replay.percentile(medians, 0.05),
                "p95_ms": replay.percentile(medians, 0.95),
                "mean_logical_routes": statistics.mean(logical),
                "mean_route_fraction": statistics.mean(logical) / full_routes,
                "mean_padded_routes": statistics.mean(padded),
                "mean_active_experts": statistics.mean(active),
                "empty_cases": sum(row.graph is None for row in rows),
            }
        )
    return records


def main() -> None:
    args = parse_args()
    requested_ms = replay.parse_ms(args.ms)
    if min(
        args.chains_per_m,
        args.layers_per_chain,
        args.warmup_rounds,
        args.repeats,
        args.placement_training_max_m,
    ) < 1:
        raise SystemExit("sample counts and placement training M must be positive")
    if not args.replay_plan.is_file():
        raise SystemExit(f"replay plan does not exist: {args.replay_plan}")
    if not args.placement_route_bank.is_file():
        raise SystemExit(
            f"placement route bank does not exist: {args.placement_route_bank}"
        )
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    capability = tuple(torch.cuda.get_device_capability())
    if args.arm in ("baseline", "spark") and capability != (12, 1):
        raise SystemExit(
            f"{args.arm} must run on GB10 sm_121, got {props.name} {capability}"
        )
    if args.arm == "local" and capability != (12, 0):
        raise SystemExit(
            f"local must run on coordinator Blackwell sm_120, got {props.name} {capability}"
        )

    replay_manifest, cases, training_chains = replay.load_cases(
        args.replay_plan,
        requested_ms,
        args.chains_per_m,
        args.layers_per_chain,
    )
    residents, placement_metadata = make_residency(
        training_chains,
        local_experts=args.local_experts,
        training_max_m=args.placement_training_max_m,
        natural_m_counts=natural_semantic_m_counts(
            args.placement_route_bank, args.placement_training_max_m
        ),
    )
    del training_chains
    device = torch.device("cuda", torch.cuda.current_device())
    sms = int(props.multi_processor_count)
    max_shared_mem = int(props.shared_memory_per_block_optin)
    started = time.time()

    if args.arm == "baseline":
        num_experts = replay.EXPERTS
        intermediate_size = replay.TP4_INTERMEDIATE
        baseline = True
    elif args.arm == "spark":
        num_experts = replay.EXPERTS - args.local_experts
        intermediate_size = replay.TP4_INTERMEDIATE
        baseline = False
    else:
        num_experts = args.local_experts
        intermediate_size = replay.FULL_INTERMEDIATE
        baseline = False

    prepared = replay.make_prepared_weights(
        num_experts=num_experts,
        intermediate_size=intermediate_size,
        seed=args.seed,
        device=device,
    )
    torch.cuda.synchronize()

    plans: dict[tuple[int, int], replay.KernelPlan] = {}

    def resolve_plan(
        replay_case: replay.ReplayCase,
        metadata: replay.RouteMetadata,
    ) -> replay.KernelPlan:
        key = (replay_case.physical_m, metadata.topk)
        plan = plans.get(key)
        if plan is not None:
            return plan
        print(
            json.dumps(
                {
                    "record": "compile",
                    "arm": args.arm,
                    "physical_m": replay_case.physical_m,
                    "topk": metadata.topk,
                    "intermediate_size": intermediate_size,
                    "num_experts": num_experts,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        plan = replay.make_plan(
            label=args.arm,
            prepared=prepared,
            m=replay_case.physical_m,
            topk=metadata.topk,
            baseline=baseline,
            sms=sms,
            max_shared_mem=max_shared_mem,
            device=device,
            tp2_small_m_kernel=args.small_m_kernel,
            tp2_large_m_block_size=args.large_m_block_size,
            tp2_no_zero_output_upper_bound=args.no_zero_output_upper_bound,
        )
        plans[key] = plan
        return plan

    measurements = []
    for case_index, replay_case in enumerate(cases, start=1):
        rows = rows_for_arm(replay_case, args.arm, residents)
        direct = replay_case.physical_m == 1 or (
            not baseline
            and replay_case.physical_m <= 8
            and args.small_m_kernel == "direct"
        )
        metadata = replay.build_route_metadata(
            rows,
            num_experts=num_experts,
            block_size=(
                8
                if replay_case.physical_m <= 8
                else args.large_m_block_size
            ),
            direct_topk=direct,
            device=device,
        )
        plan = None if metadata is None else resolve_plan(replay_case, metadata)
        measurements.append(
            replay.capture_measurement(
                arm=args.arm,
                replay_case=replay_case,
                metadata=metadata,
                plan=plan,
            )
        )
        if case_index % max(1, args.chains_per_m * args.layers_per_chain) == 0:
            print(
                json.dumps(
                    {
                        "record": "capture_progress",
                        "arm": args.arm,
                        "physical_m": replay_case.physical_m,
                        "cases_captured": case_index,
                        "total_cases": len(cases),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    replay.time_graphs(
        measurements,
        warmup_rounds=args.warmup_rounds,
        repeats=args.repeats,
        seed=args.seed,
    )
    if any(not measurement.finite for measurement in measurements):
        raise RuntimeError("one or more graph outputs were non-finite")
    if any(
        measurement.graph is not None and not measurement.nonzero
        for measurement in measurements
    ):
        raise RuntimeError("one or more nonempty graph outputs were all zero")

    manifest = {
        "record": "manifest",
        "schema": "glmrt-local-spark-hybrid-route-replay-arm-v1",
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "root_revision": replay.git_revision(Path(__file__).resolve().parents[2]),
        "sparkinfer_revision": replay._pinned_sparkinfer.REVISION,
        "sparkinfer_version": replay._pinned_sparkinfer.VERSION,
        "replay_plan": str(args.replay_plan.resolve()),
        "replay_plan_schema": replay_manifest.get("schema"),
        "arm": args.arm,
        "requested_ms": requested_ms,
        "chains_per_m": args.chains_per_m,
        "layers_per_chain": args.layers_per_chain,
        "warmup_rounds": args.warmup_rounds,
        "repeats": args.repeats,
        "seed": args.seed,
        "placement": placement_metadata,
        "num_experts": num_experts,
        "intermediate_size": intermediate_size,
        "small_m_kernel": "production" if baseline else args.small_m_kernel,
        "large_m_block_size": args.large_m_block_size,
        "zero_partial_route_output": not args.no_zero_output_upper_bound,
        "valid_output": not args.no_zero_output_upper_bound,
        "network_involved": False,
        "route_pack_timed": False,
        "timing": "CUDA graph replay with CUDA events; route cases interleaved",
        "gpu": {
            "name": props.name,
            "capability": list(capability),
            "sms": sms,
            "max_shared_mem": max_shared_mem,
        },
    }
    records = [manifest]
    records.extend(replay.measurement_record(item) for item in measurements)
    records.extend(summarize(cases, measurements))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
    temporary.replace(args.output)
    for record in records:
        if record["record"] in ("manifest", "summary"):
            print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
