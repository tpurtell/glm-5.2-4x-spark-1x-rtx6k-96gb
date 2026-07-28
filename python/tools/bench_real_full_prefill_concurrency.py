#!/usr/bin/env python3
"""Measure simultaneous cache-cold prefill throughput and TTFT fairness."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from real_full_matrix import (
    MODEL_ID,
    default_tokenizer_path,
    git_commit,
    load_corpus,
    render_messages,
    repo_root,
)
from bench_real_full_concurrency import token_zero_nonces


def comma_separated_positive_ints(value: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "plan values must be comma-separated integers"
        ) from error
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("plan values must be positive")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, type=Path)
    source_shape = parser.add_mutually_exclusive_group(required=True)
    source_shape.add_argument("--source-tokens", type=int)
    source_shape.add_argument(
        "--source-token-plan",
        type=comma_separated_positive_ints,
        help="comma-separated source-token count for each lane",
    )
    parser.add_argument("--concurrency", required=True, type=int)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument(
        "--max-token-plan",
        type=comma_separated_positive_ints,
        help="comma-separated maximum output tokens for each lane",
    )
    parser.add_argument(
        "--cache-state",
        choices=("token-zero-nonce", "exact-repeat"),
        default="token-zero-nonce",
        help=(
            "token-zero-nonce makes the first user-content token unique; "
            "the shared chat-template prefix may still hit"
        ),
    )
    parser.add_argument(
        "--nonce-seed",
        type=int,
        default=time.time_ns(),
        help=(
            "reproducible first-content-token nonce bank; never reuse against "
            "a live cache"
        ),
    )
    parser.add_argument("--max-context-tokens", type=int, default=400_000)
    parser.add_argument("--pool-tokens", type=int, default=600_000)
    parser.add_argument("--page-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:8000/v1/chat/completions"
    )
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--instruction",
        default="\n\nAnalyze the material above and explain its main technical design.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.concurrency, args.repeats, args.max_tokens) < 1:
        parser.error("concurrency, repeats, and max tokens must be positive")
    if args.source_tokens is not None and args.source_tokens < 1:
        parser.error("source tokens must be positive")
    if args.warmups < 0 or args.timeout_seconds <= 0:
        parser.error("warmups must be non-negative and timeout must be positive")
    if min(args.max_context_tokens, args.pool_tokens, args.page_tokens) < 1:
        parser.error("context, pool, and page token limits must be positive")
    if (
        args.source_token_plan is not None
        and len(args.source_token_plan) != args.concurrency
    ):
        parser.error("--source-token-plan length must equal --concurrency")
    if args.max_token_plan is not None and len(args.max_token_plan) != args.concurrency:
        parser.error("--max-token-plan length must equal --concurrency")
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    return args


def request_one(
    endpoint: str,
    body: bytes,
    timeout_seconds: float,
    barrier: threading.Barrier,
    lane: int,
) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    barrier.wait()
    request_start = time.perf_counter()
    first_token_at = None
    metrics = None
    output_parts = []
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                continue
            event = json.loads(payload)
            for choice in event.get("choices") or []:
                content = (choice.get("delta") or {}).get("content")
                if content:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    output_parts.append(content)
            if "metrics" in event:
                metrics = event["metrics"]
    response_end = time.perf_counter()
    if metrics is None or first_token_at is None:
        raise RuntimeError(f"lane {lane} completed without token/metrics")
    content = "".join(output_parts)
    return {
        "lane": lane,
        "request_start": request_start,
        "first_token_at": first_token_at,
        "response_end": response_end,
        "metrics": metrics,
        "content": content,
        "server_error": (
            content
            if content.startswith("real-full streaming executor error:")
            else None
        ),
    }


def execute_batch(
    args: argparse.Namespace,
    bodies: list[bytes],
    planned_prompt_tokens: list[int],
    max_tokens: list[int],
    prompt_nonces: list[dict[str, Any] | None],
) -> dict[str, Any]:
    barrier = threading.Barrier(args.concurrency)
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                request_one,
                args.endpoint,
                bodies[lane],
                args.timeout_seconds,
                barrier,
                lane,
            )
            for lane in range(args.concurrency)
        ]
        raw_lanes = [future.result() for future in futures]

    window_start = min(lane["request_start"] for lane in raw_lanes)
    window_end = max(lane["first_token_at"] for lane in raw_lanes)
    response_end = max(lane["response_end"] for lane in raw_lanes)
    lanes = []
    for raw in sorted(raw_lanes, key=lambda lane: lane["lane"]):
        metrics = raw["metrics"]
        real_full = metrics.get("real_full") or {}
        lane = raw["lane"]
        server_error = raw["server_error"]
        reported_prompt_tokens = metrics.get("prompt_tokens")
        if (
            server_error is None
            and reported_prompt_tokens is not None
            and int(reported_prompt_tokens) != planned_prompt_tokens[lane]
        ):
            raise RuntimeError(
                f"lane {lane} reported {metrics['prompt_tokens']} prompt "
                f"tokens, expected {planned_prompt_tokens[lane]}"
            )
        completion_tokens = int(metrics.get("output_tokens") or 0)
        decode_ms = float(metrics.get("decode_ms") or 0.0)
        nonce = prompt_nonces[lane]
        lanes.append(
            {
                "lane": lane,
                "planned_prompt_tokens": planned_prompt_tokens[lane],
                "max_output_tokens": max_tokens[lane],
                "required_context_tokens": (
                    planned_prompt_tokens[lane] + max_tokens[lane]
                ),
                "reserved_page_tokens": (
                    (
                        planned_prompt_tokens[lane]
                        + max_tokens[lane]
                        + args.page_tokens
                        - 1
                    )
                    // args.page_tokens
                    * args.page_tokens
                ),
                "request_start_ms": (raw["request_start"] - window_start) * 1_000.0,
                "external_ttft_ms": (
                    raw["first_token_at"] - raw["request_start"]
                )
                * 1_000.0,
                "response_end_ms": (raw["response_end"] - window_start) * 1_000.0,
                "successful": server_error is None,
                "server_error": server_error,
                "server_ttft_ms": (
                    float(metrics["time_to_first_token_ms"])
                    if metrics.get("time_to_first_token_ms") is not None
                    else None
                ),
                "prefill_ms": float(metrics.get("prefill_ms") or 0.0),
                "prefill_rows": int(metrics.get("layerwave_prefill_rows") or 0),
                "reported_prefill_tps": (
                    float(metrics["prefill_tokens_per_sec"])
                    if metrics.get("prefill_tokens_per_sec") is not None
                    else None
                ),
                "prefill_chunks": int(metrics.get("prefill_chunk_count") or 0),
                "completion_tokens": completion_tokens,
                "decode_ms": decode_ms,
                "decode_tps": (
                    (completion_tokens - 1) * 1_000.0 / decode_ms
                    if completion_tokens > 1 and decode_ms > 0.0
                    else None
                ),
                "runtime_captures": int(
                    real_full.get("request_coordinator_graph_captures") or 0
                ),
                "verify_cycles": int(real_full.get("mtp_verify_cycles") or 0),
                "draft_tokens": int(real_full.get("mtp_draft_tokens") or 0),
                "accepted_draft_tokens": int(
                    real_full.get("mtp_accepted_draft_tokens") or 0
                ),
                "attention_complete": bool(
                    real_full.get("scheduler_full_context_device_attention_complete")
                    or False
                ),
                "numeric_progression_passed": bool(
                    real_full.get("request_numeric_progression_passed") or False
                ),
                "content_sha256": hashlib.sha256(raw["content"].encode()).hexdigest(),
                "content_utf8_bytes": len(raw["content"].encode()),
                "prompt_nonce": (
                    None
                    if nonce is None
                    else {
                        "marker": nonce["marker"],
                        "first_content_token_id": nonce["first_content_token_id"],
                    }
                ),
            }
        )
    total_rows = sum(lane["prefill_rows"] for lane in lanes)
    prefill_window_ms = (window_end - window_start) * 1_000.0
    all_successful = all(lane["successful"] for lane in lanes)
    return {
        "concurrency": args.concurrency,
        "planned_prompt_tokens": planned_prompt_tokens,
        "max_output_tokens": max_tokens,
        "required_context_tokens": [
            prompt_tokens + output_tokens
            for prompt_tokens, output_tokens in zip(
                planned_prompt_tokens, max_tokens, strict=True
            )
        ],
        "reserved_page_tokens": sum(
            lane["reserved_page_tokens"] for lane in lanes
        ),
        "total_prefill_rows": total_rows,
        "prefill_window_ms": prefill_window_ms,
        "response_window_ms": (response_end - window_start) * 1_000.0,
        "aggregate_prefill_tps": (
            total_rows * 1_000.0 / prefill_window_ms
            if all_successful and prefill_window_ms > 0.0
            else None
        ),
        "all_successful": all_successful,
        "all_attention_complete": all(
            lane["attention_complete"] for lane in lanes
        ),
        "all_numeric_progression_passed": all(
            lane["numeric_progression_passed"] for lane in lanes
        ),
        "all_zero_runtime_captures": all(
            lane["runtime_captures"] == 0 for lane in lanes
        ) and all_successful,
        "lanes": lanes,
    }


def main() -> None:
    args = parse_args()
    root = repo_root()
    tokenizer_path = (args.tokenizer or default_tokenizer_path()).resolve()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    corpus, corpus_sha256 = load_corpus(args.source)
    corpus_ids = tokenizer.encode(corpus, add_special_tokens=False).ids
    source_token_plan = args.source_token_plan or [args.source_tokens] * args.concurrency
    max_token_plan = args.max_token_plan or [args.max_tokens] * args.concurrency
    if max(source_token_plan) > len(corpus_ids):
        raise SystemExit(
            f"requested {max(source_token_plan)} source tokens, "
            f"corpus has {len(corpus_ids)}"
        )
    request_count = (args.warmups + args.repeats) * args.concurrency
    if args.cache_state == "token-zero-nonce":
        nonce_bank: list[dict[str, Any] | None] = token_zero_nonces(
            count=request_count,
            seed=args.nonce_seed,
            tokenizer_path=tokenizer_path,
        )
    else:
        nonce_bank = [None] * request_count
    nonce_offset = 0

    def build_batch_inputs(
    ) -> tuple[list[bytes], list[int], list[dict[str, Any] | None]]:
        nonlocal nonce_offset
        prompt_nonces = nonce_bank[nonce_offset : nonce_offset + args.concurrency]
        nonce_offset += args.concurrency
        bodies = []
        planned_prompt_tokens = []
        for lane, source_tokens in enumerate(source_token_plan):
            source_text = tokenizer.decode(
                corpus_ids[:source_tokens],
                skip_special_tokens=False,
            )
            nonce = prompt_nonces[lane]
            prompt_prefix = "" if nonce is None else str(nonce["prefix"])
            messages = [
                {
                    "role": "user",
                    "content": (
                        prompt_prefix + source_text + args.instruction
                    ),
                }
            ]
            prompt_tokens = len(
                tokenizer.encode(
                    render_messages(messages),
                    add_special_tokens=False,
                ).ids
            )
            required_tokens = prompt_tokens + max_token_plan[lane]
            if required_tokens > args.max_context_tokens:
                raise SystemExit(
                    f"lane {lane} requires {required_tokens} tokens, exceeding "
                    f"--max-context-tokens {args.max_context_tokens}"
                )
            planned_prompt_tokens.append(prompt_tokens)
            bodies.append(
                json.dumps(
                    {
                        "model": MODEL_ID,
                        "messages": messages,
                        "stream": True,
                        "max_tokens": max_token_plan[lane],
                    }
                ).encode()
            )
        reserved_page_tokens = sum(
            (
                (prompt_tokens + output_tokens + args.page_tokens - 1)
                // args.page_tokens
                * args.page_tokens
            )
            for prompt_tokens, output_tokens in zip(
                planned_prompt_tokens,
                max_token_plan,
                strict=True,
            )
        )
        if reserved_page_tokens > args.pool_tokens:
            raise SystemExit(
                f"batch reserves {reserved_page_tokens} page-rounded tokens, "
                f"exceeding --pool-tokens {args.pool_tokens}"
            )
        return bodies, planned_prompt_tokens, prompt_nonces

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "record": "manifest",
        "schema": "glmrt-prefill-concurrency-v2",
        "commit": git_commit(root),
        "model": MODEL_ID,
        "endpoint": args.endpoint,
        "sources": [str(source.resolve()) for source in args.source],
        "corpus_sha256": corpus_sha256,
        "source_token_plan": source_token_plan,
        "max_token_plan": max_token_plan,
        "cache_state": args.cache_state,
        "nonce_seed": args.nonce_seed,
        "max_context_tokens": args.max_context_tokens,
        "pool_tokens": args.pool_tokens,
        "page_tokens": args.page_tokens,
        "instruction": args.instruction,
        "concurrency": args.concurrency,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "timeout_seconds": args.timeout_seconds,
        "tokenizer": str(tokenizer_path),
    }
    if args.dry_run:
        _, planned_prompt_tokens, prompt_nonces = build_batch_inputs()
        required_context_tokens = [
            prompt_tokens + output_tokens
            for prompt_tokens, output_tokens in zip(
                planned_prompt_tokens,
                max_token_plan,
                strict=True,
            )
        ]
        reserved_page_tokens = [
            (
                (required_tokens + args.page_tokens - 1)
                // args.page_tokens
                * args.page_tokens
            )
            for required_tokens in required_context_tokens
        ]
        print(
            json.dumps(
                {
                    **manifest,
                    "planned_prompt_tokens": planned_prompt_tokens,
                    "required_context_tokens": required_context_tokens,
                    "reserved_page_tokens": reserved_page_tokens,
                    "total_reserved_page_tokens": sum(reserved_page_tokens),
                    "nonce_token_ids": [
                        None
                        if nonce is None
                        else nonce["first_content_token_id"]
                        for nonce in prompt_nonces
                    ],
                },
                indent=2,
            )
        )
        return
    measurements = []
    with args.output.open("w", encoding="utf-8") as output:
        output.write(json.dumps(manifest, separators=(",", ":")) + "\n")
        print(json.dumps(manifest, sort_keys=True), flush=True)
        for sample in range(args.warmups + args.repeats):
            bodies, planned_prompt_tokens, prompt_nonces = build_batch_inputs()
            measurement = execute_batch(
                args,
                bodies,
                planned_prompt_tokens,
                max_token_plan,
                prompt_nonces,
            )
            measurement["record"] = (
                "warmup" if sample < args.warmups else "measurement"
            )
            measurement["sample"] = (
                sample + 1
                if sample < args.warmups
                else sample - args.warmups + 1
            )
            output.write(json.dumps(measurement, separators=(",", ":")) + "\n")
            output.flush()
            print(json.dumps(measurement, sort_keys=True), flush=True)
            if measurement["record"] == "measurement":
                measurements.append(measurement)
        successful_measurements = [
            item for item in measurements if item["all_successful"]
        ]
        samples = [
            item["aggregate_prefill_tps"] for item in successful_measurements
        ]
        slowest_ttft_samples = [
            max(lane["external_ttft_ms"] for lane in item["lanes"])
            for item in successful_measurements
        ]
        ttft_spread_samples = [
            max(lane["external_ttft_ms"] for lane in item["lanes"])
            - min(lane["external_ttft_ms"] for lane in item["lanes"])
            for item in successful_measurements
        ]
        summary = {
            "record": "summary",
            "successful_measurements": len(successful_measurements),
            "failed_measurements": len(measurements) - len(successful_measurements),
            "median_aggregate_prefill_tps": (
                statistics.median(samples) if samples else None
            ),
            "mean_aggregate_prefill_tps": (
                statistics.mean(samples) if samples else None
            ),
            "min_aggregate_prefill_tps": min(samples) if samples else None,
            "max_aggregate_prefill_tps": max(samples) if samples else None,
            "stdev_aggregate_prefill_tps": (
                statistics.stdev(samples) if len(samples) > 1 else 0.0
            ),
            "median_slowest_external_ttft_ms": (
                statistics.median(slowest_ttft_samples)
                if slowest_ttft_samples
                else None
            ),
            "median_external_ttft_spread_ms": (
                statistics.median(ttft_spread_samples)
                if ttft_spread_samples
                else None
            ),
            "all_successful": len(successful_measurements) == len(measurements),
            "all_attention_complete": all(
                item["all_attention_complete"] for item in measurements
            ) and len(successful_measurements) == len(measurements),
            "all_numeric_progression_passed": all(
                item["all_numeric_progression_passed"] for item in measurements
            ) and len(successful_measurements) == len(measurements),
            "all_zero_runtime_captures": all(
                item["all_zero_runtime_captures"] for item in measurements
            ) and len(successful_measurements) == len(measurements),
        }
        output.write(json.dumps(summary, separators=(",", ":")) + "\n")
        print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
