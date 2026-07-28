#!/usr/bin/env python3
"""Collect a compact, self-validating bank of real dSpark expert routes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable

from bench_real_full_mtp_acceptance import (
    CASES,
    WEIGHTED_CASE_IDS,
    completion_payload,
    request_completion,
    summarize_case,
)


TRACE_MARKER = "protocol_v2_expert_queue_row_routes "
FIRST_SPARSE_LAYER = 3
LAST_SPARSE_LAYER = 77
SPARSE_LAYERS = tuple(range(FIRST_SPARSE_LAYER, LAST_SPARSE_LAYER + 1))
DEFAULT_CASE_IDS = (*WEIGHTED_CASE_IDS, "count", "repeat")
COLLECTION_MULTIPLIERS = {"code": 2}


@dataclass(frozen=True)
class RouteFragment:
    request_id_base: int
    layer_id: int
    physical_m: int
    source_kinds: tuple[str, ...]
    routes: tuple[tuple[int, ...], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default="http://127.0.0.1:8000/v1/chat/completions"
    )
    parser.add_argument("--model", default="lukealonso/GLM-5.2-NVFP4")
    parser.add_argument(
        "--trace-log",
        type=Path,
        required=True,
        help=(
            "Coordinator stderr log produced with "
            "GLMRT_PROTOCOL_V2_EXPERT_QUEUE_STATS=1 and "
            "GLMRT_PROTOCOL_V2_EXPERT_QUEUE_ROW_ROUTES=1."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output route bank; .gz produces compressed JSONL.",
    )
    parser.add_argument(
        "--outputs-per-case",
        type=int,
        default=8,
        help=(
            "Complete outputs collected per case before collection multipliers; "
            "code is collected twice."
        ),
    )
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        choices=sorted(CASES),
        help=(
            "Collect only this case; may be repeated to set an explicit "
            "multiplicity. By default collects the semantic suite plus count/repeat."
        ),
    )
    parser.add_argument(
        "--include-syntax",
        action="store_true",
        help="Also collect the two long syntax reachability cases.",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


def parse_field(fields: str, key: str) -> str:
    prefix = f"{key}="
    for field in fields.split():
        if field.startswith(prefix):
            return field[len(prefix) :]
    raise ValueError(f"route trace has no {key} field")


def parse_route_line(line: str) -> RouteFragment | None:
    marker_offset = line.find(TRACE_MARKER)
    if marker_offset < 0:
        return None
    fields = line[marker_offset + len(TRACE_MARKER) :]
    request_id_base = int(parse_field(fields, "request_id_base"))
    layer_id = int(parse_field(fields, "layer_id"))
    physical_m = int(parse_field(fields, "rows"))
    source_kinds = tuple(
        part for part in parse_field(fields, "source_kinds").split("+") if part
    )
    try:
        raw_routes = fields.split("row_routes=", 1)[1].strip()
    except IndexError as error:
        raise ValueError("route trace has no row_routes field") from error

    indexed_routes: dict[int, tuple[int, ...]] = {}
    for raw_row in raw_routes.split(","):
        raw_index, raw_experts = raw_row.split(":", 1)
        row_index = int(raw_index)
        experts = tuple(int(expert) for expert in raw_experts.split("+"))
        if len(experts) != 8:
            raise ValueError(
                f"row {row_index} has {len(experts)} routes instead of top-8"
            )
        if len(set(experts)) != len(experts):
            raise ValueError(f"row {row_index} repeats an expert ID")
        indexed_routes[row_index] = experts
    expected_indices = set(range(physical_m))
    if set(indexed_routes) != expected_indices:
        raise ValueError(
            f"row indices {sorted(indexed_routes)} do not match 0..{physical_m - 1}"
        )
    return RouteFragment(
        request_id_base=request_id_base,
        layer_id=layer_id,
        physical_m=physical_m,
        source_kinds=source_kinds,
        routes=tuple(indexed_routes[index] for index in range(physical_m)),
    )


def read_trace_delta(trace: BinaryIO, start_offset: int) -> tuple[list[RouteFragment], int]:
    trace.seek(0, os.SEEK_END)
    end_offset = trace.tell()
    trace.seek(start_offset)
    delta = trace.read(end_offset - start_offset).decode("utf-8", errors="replace")
    fragments = []
    for line_number, line in enumerate(delta.splitlines(), start=1):
        try:
            fragment = parse_route_line(line)
        except ValueError as error:
            raise ValueError(f"trace delta line {line_number}: {error}") from error
        if fragment is not None:
            fragments.append(fragment)
    return fragments, end_offset


def verify_cycles(
    fragments: Iterable[RouteFragment], expected_physical_ms: list[int]
) -> list[list[RouteFragment]]:
    verify_fragments = [
        fragment
        for fragment in fragments
        if "MtpVerifyBlock" in fragment.source_kinds
        and fragment.layer_id in SPARSE_LAYERS
    ]
    expected_fragments = len(expected_physical_ms) * len(SPARSE_LAYERS)
    if len(verify_fragments) != expected_fragments:
        raise ValueError(
            f"captured {len(verify_fragments)} verify layer fragments; "
            f"expected {expected_fragments} for {len(expected_physical_ms)} cycles"
        )

    cycles = []
    for cycle_index, expected_m in enumerate(expected_physical_ms):
        start = cycle_index * len(SPARSE_LAYERS)
        cycle = verify_fragments[start : start + len(SPARSE_LAYERS)]
        actual_layers = tuple(fragment.layer_id for fragment in cycle)
        if actual_layers != SPARSE_LAYERS:
            raise ValueError(
                f"verify cycle {cycle_index} layer order/coverage is incomplete: "
                f"{actual_layers[:4]}...{actual_layers[-4:]}"
            )
        actual_ms = {fragment.physical_m for fragment in cycle}
        if actual_ms != {expected_m}:
            raise ValueError(
                f"verify cycle {cycle_index} captured physical M {sorted(actual_ms)}, "
                f"API metrics report M={expected_m}"
            )
        cycles.append(cycle)
    return cycles


def output_writer(path: Path, compressed: bool):
    if compressed:
        return gzip.open(path, "wt", encoding="utf-8", compresslevel=6)
    return path.open("w", encoding="utf-8")


def write_record(output: Any, record: dict[str, Any]) -> None:
    output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    output.write("\n")


def selected_cases(args: argparse.Namespace) -> list[str]:
    if args.cases:
        return list(args.cases)
    selected = list(DEFAULT_CASE_IDS)
    if args.include_syntax:
        selected.extend(("syntax-rust", "syntax-python"))
    return selected


def collection_schedule(args: argparse.Namespace) -> list[tuple[str, int]]:
    selected = selected_cases(args)
    if args.cases:
        occurrences = Counter(selected)
        return [
            (case_id, output_id)
            for case_id in dict.fromkeys(selected)
            for output_id in range(args.outputs_per_case * occurrences[case_id])
        ]
    return [
        (case_id, output_id)
        for case_id in selected
        for output_id in range(
            args.outputs_per_case * COLLECTION_MULTIPLIERS.get(case_id, 1)
        )
    ]


def main() -> None:
    args = parse_args()
    if args.outputs_per_case < 1:
        raise SystemExit("--outputs-per-case must be positive")
    if not args.trace_log.is_file():
        raise SystemExit(f"trace log does not exist: {args.trace_log}")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing route bank: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    schedule = collection_schedule(args)
    prompt_manifest = {
        case_id: {
            "category": CASES[case_id].category,
            "max_tokens": CASES[case_id].max_tokens,
            "prompt_sha256": hashlib.sha256(
                CASES[case_id].prompt.encode()
            ).hexdigest(),
            "prompt": CASES[case_id].prompt,
        }
        for case_id in dict.fromkeys(case_id for case_id, _ in schedule)
    }
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    if temporary.exists():
        temporary.unlink()

    collected_outputs = Counter()
    collected_cycles = Counter()
    collected_fragments = Counter()
    try:
        with args.trace_log.open("rb") as trace, output_writer(
            temporary, args.output.suffix == ".gz"
        ) as output:
            trace.seek(0, os.SEEK_END)
            trace_offset = trace.tell()
            write_record(
                output,
                {
                    "record": "manifest",
                    "schema": "glmrt-expert-route-bank-v1",
                    "created_unix": time.time(),
                    "model": args.model,
                    "source_trace_log": str(args.trace_log.resolve()),
                    "first_sparse_layer": FIRST_SPARSE_LAYER,
                    "last_sparse_layer": LAST_SPARSE_LAYER,
                    "top_k": 8,
                    "collection_multipliers": COLLECTION_MULTIPLIERS,
                    "normal_weighted_case_ids": list(WEIGHTED_CASE_IDS),
                    "diagnostic_case_ids": ["count", "repeat"],
                    "prompts": prompt_manifest,
                    "scheduled_outputs": len(schedule),
                },
            )

            for schedule_index, (case_id, output_id) in enumerate(schedule, start=1):
                case = CASES[case_id]
                payload = completion_payload(args.model, case)
                result = request_completion(args.url, payload, args.timeout)
                summary = summarize_case(case_id, result)
                fragments, trace_offset = read_trace_delta(trace, trace_offset)
                expected_physical_ms = [
                    draft_length + 1 for draft_length in summary["draft_lengths"]
                ]
                cycles = verify_cycles(fragments, expected_physical_ms)
                response_id = result.get("id")
                bank_output_id = f"{case_id}-{output_id:03d}"
                content = result["choices"][0]["message"]["content"]
                write_record(
                    output,
                    {
                        "record": "output",
                        "output_id": bank_output_id,
                        "response_id": response_id,
                        "case": case_id,
                        "category": case.category,
                        "prompt_tokens": summary["prompt_tokens"],
                        "completion_tokens": summary["completion_tokens"],
                        "finish_reason": summary["finish_reason"],
                        "content_sha256": summary["content_sha256"],
                        "content": content,
                        "verify_cycles": len(cycles),
                        "physical_ms": expected_physical_ms,
                    },
                )
                for cycle_index, cycle in enumerate(cycles):
                    for fragment in cycle:
                        write_record(
                            output,
                            {
                                "record": "fragment",
                                "output_id": bank_output_id,
                                "case": case_id,
                                "category": case.category,
                                "cycle": cycle_index,
                                "physical_m": fragment.physical_m,
                                "layer_id": fragment.layer_id,
                                "request_id_base": fragment.request_id_base,
                                "routes": fragment.routes,
                            },
                        )
                output.flush()
                collected_outputs[case_id] += 1
                collected_cycles[case_id] += len(cycles)
                collected_fragments[case_id] += sum(len(cycle) for cycle in cycles)
                print(
                    json.dumps(
                        {
                            "progress": schedule_index,
                            "scheduled": len(schedule),
                            "case": case_id,
                            "output_id": bank_output_id,
                            "cycles": len(cycles),
                            "fragments": sum(len(cycle) for cycle in cycles),
                            "physical_m_histogram": dict(
                                sorted(Counter(expected_physical_ms).items())
                            ),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

            write_record(
                output,
                {
                    "record": "summary",
                    "outputs_by_case": dict(collected_outputs),
                    "cycles_by_case": dict(collected_cycles),
                    "fragments_by_case": dict(collected_fragments),
                    "outputs": sum(collected_outputs.values()),
                    "cycles": sum(collected_cycles.values()),
                    "fragments": sum(collected_fragments.values()),
                },
            )
        temporary.replace(args.output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    print(
        json.dumps(
            {
                "route_bank": str(args.output),
                "bytes": args.output.stat().st_size,
                "outputs": sum(collected_outputs.values()),
                "cycles": sum(collected_cycles.values()),
                "fragments": sum(collected_fragments.values()),
            }
        )
    )


if __name__ == "__main__":
    main()
