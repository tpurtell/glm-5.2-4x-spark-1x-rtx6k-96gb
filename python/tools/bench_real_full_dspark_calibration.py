#!/usr/bin/env python3
"""Run deterministic, routing-diverse concurrent dSpark calibration cohorts."""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from bench_real_full_mtp_acceptance import (
    CASES,
    WEIGHTED_CASE_IDS,
    completion_payload,
    request_completion,
    summarize_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default="http://127.0.0.1:8000/v1/chat/completions"
    )
    parser.add_argument("--model", default="lukealonso/GLM-5.2-NVFP4")
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20_260_813)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


def request_lane(
    *,
    lane: int,
    case_id: str,
    prompt_prefix: str,
    model: str,
    url: str,
    timeout: float,
    barrier: threading.Barrier,
) -> dict[str, Any]:
    case = CASES[case_id]
    payload = json.loads(completion_payload(model, case))
    payload["messages"][0]["content"] = prompt_prefix + payload["messages"][0]["content"]
    barrier.wait()
    request_start = time.perf_counter()
    result = request_completion(
        url,
        json.dumps(payload, ensure_ascii=False).encode(),
        timeout,
    )
    response_end = time.perf_counter()
    summary = summarize_case(case_id, result)
    summary.update(
        {
            "lane": lane,
            "request_start": request_start,
            "response_end": response_end,
            "first_token": request_start
            + float(result["metrics"]["time_to_first_token_ms"]) / 1_000.0,
            "runtime_captures": int(
                result["metrics"]["real_full"]["request_coordinator_graph_captures"]
            ),
        }
    )
    return summary


def run_batch(args: argparse.Namespace, batch_index: int, warmup: bool) -> dict[str, Any]:
    corpus = list(WEIGHTED_CASE_IDS)
    start = (args.seed + batch_index * args.concurrency) % len(corpus)
    case_ids = [corpus[(start + lane) % len(corpus)] for lane in range(args.concurrency)]
    barrier = threading.Barrier(args.concurrency)
    batch_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                request_lane,
                lane=lane,
                case_id=case_id,
                prompt_prefix=(
                    f"Calibration nonce {args.seed}-{batch_index}-{lane}. "
                    "Treat this identifier as irrelevant.\n"
                ),
                model=args.model,
                url=args.url,
                timeout=args.timeout,
                barrier=barrier,
            )
            for lane, case_id in enumerate(case_ids)
        ]
        lanes = [future.result() for future in futures]
    first_token_times = [lane["first_token"] for lane in lanes]
    decode_end_times = [
        first_token + lane["decode_ms"] / 1_000.0
        for first_token, lane in zip(first_token_times, lanes, strict=True)
    ]
    decode_window_seconds = max(decode_end_times) - min(first_token_times)
    timed_tokens = sum(lane["completion_tokens"] - 1 for lane in lanes)
    for lane in lanes:
        lane["request_start_ms"] = (lane.pop("request_start") - batch_start) * 1_000.0
        lane["response_end_ms"] = (lane.pop("response_end") - batch_start) * 1_000.0
        lane["first_token_ms"] = (lane.pop("first_token") - batch_start) * 1_000.0
    return {
        "schema": "glmrt-dspark-calibration-batch-v1",
        "warmup": warmup,
        "batch": batch_index,
        "seed": args.seed,
        "concurrency": args.concurrency,
        "case_ids": case_ids,
        "timed_tokens": timed_tokens,
        "decode_window_ms": decode_window_seconds * 1_000.0,
        "aggregate_decode_tps": (
            timed_tokens / decode_window_seconds if decode_window_seconds > 0.0 else 0.0
        ),
        "all_zero_runtime_captures": all(
            lane["runtime_captures"] == 0 for lane in lanes
        ),
        "lanes": sorted(lanes, key=lambda lane: lane["lane"]),
    }


def main() -> None:
    args = parse_args()
    if not 1 <= args.concurrency <= 4:
        raise SystemExit("--concurrency must be in 1..4")
    if args.warmups < 0 or args.repeats < 1:
        raise SystemExit("--warmups must be non-negative and --repeats positive")
    batches = []
    for batch_index in range(args.warmups + args.repeats):
        batch = run_batch(args, batch_index, batch_index < args.warmups)
        batches.append(batch)
        print(json.dumps(batch, ensure_ascii=False), flush=True)
    measured = [batch for batch in batches if not batch["warmup"]]
    tps = [batch["aggregate_decode_tps"] for batch in measured]
    summary = {
        "schema": "glmrt-dspark-calibration-summary-v1",
        "seed": args.seed,
        "concurrency": args.concurrency,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "median_aggregate_decode_tps": statistics.median(tps),
        "mean_aggregate_decode_tps": statistics.mean(tps),
        "minimum_aggregate_decode_tps": min(tps),
        "maximum_aggregate_decode_tps": max(tps),
        "all_zero_runtime_captures": all(
            batch["all_zero_runtime_captures"] for batch in batches
        ),
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
