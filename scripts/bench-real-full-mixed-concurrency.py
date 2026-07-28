#!/usr/bin/env python3
"""Measure steady and membership-changing real-full streaming workloads."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python" / "tools"))
from bench_real_full_concurrency import FIXTURES, judge_code  # noqa: E402

CODE_PROMPT = FIXTURES["code"].prompt


@dataclass
class StreamResult:
    lane: str
    max_tokens: int
    start_s: float
    first_content_s: float | None
    end_s: float
    status_code: int
    content: str
    finish_reason: str | None
    metrics: dict[str, Any] | None
    usage: dict[str, Any] | None
    cancelled: bool
    content_chunks: int

    @property
    def server_error(self) -> bool:
        return self.content.startswith("real-full streaming executor error:")

    def as_dict(self, origin_s: float) -> dict[str, Any]:
        real_full = (self.metrics or {}).get("real_full") or {}
        correct, verdict = judge_code(self.content)
        server_ttft_ms = (self.metrics or {}).get("time_to_first_token_ms")
        decode_ms = (self.metrics or {}).get("decode_ms")
        completion_tokens = (
            (self.usage or {}).get("completion_tokens")
            or (self.metrics or {}).get("output_tokens")
            or self.content_chunks
        )
        return {
            "lane": self.lane,
            "max_tokens": self.max_tokens,
            "status_code": self.status_code,
            "cancelled": self.cancelled,
            "server_error": self.server_error,
            "request_start_ms": (self.start_s - origin_s) * 1_000.0,
            "first_content_ms": (
                None
                if self.first_content_s is None
                else (self.first_content_s - origin_s) * 1_000.0
            ),
            "response_end_ms": (self.end_s - origin_s) * 1_000.0,
            "ttft_ms": (
                None
                if self.first_content_s is None
                else (self.first_content_s - self.start_s) * 1_000.0
            ),
            "server_ttft_ms": server_ttft_ms,
            "prefill_ms": (self.metrics or {}).get("prefill_ms"),
            "decode_ms": decode_ms,
            "decode_tps": (
                (completion_tokens - 1) / (decode_ms / 1_000.0)
                if completion_tokens > 1 and decode_ms
                else None
            ),
            "response_ms": (self.end_s - self.start_s) * 1_000.0,
            "completion_tokens": completion_tokens,
            "finish_reason": self.finish_reason,
            "runtime_captures": real_full.get(
                "request_coordinator_graph_captures"
            ),
            "verify_cycles": real_full.get("mtp_verify_cycles"),
            "draft_tokens": real_full.get("mtp_draft_tokens"),
            "accepted_draft_tokens": real_full.get(
                "mtp_accepted_draft_tokens"
            ),
            "draft_lengths": real_full.get("mtp_draft_lengths"),
            "accepted_draft_lengths": real_full.get(
                "mtp_accepted_draft_lengths"
            ),
            "content_chunks": self.content_chunks,
            "content_sha256": hashlib.sha256(self.content.encode()).hexdigest(),
            "code_contract": correct,
            "code_verdict": verdict,
            "content_preview": self.content[:160],
        }


async def stream_request(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    lane: str,
    max_tokens: int,
    delay_s: float = 0.0,
    cancel_after_chunks: int | None = None,
) -> StreamResult:
    await asyncio.sleep(delay_s)
    start_s = time.perf_counter()
    first_content_s: float | None = None
    end_s = start_s
    content_parts: list[str] = []
    finish_reason: str | None = None
    metrics: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    cancelled = False
    content_chunks = 0
    status_code = 0
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": CODE_PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    async with client.stream(
        "POST",
        f"{base_url.rstrip('/')}/chat/completions",
        json=payload,
    ) as response:
        status_code = response.status_code
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            if event.get("metrics"):
                metrics = event["metrics"]
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if text is not None:
                if first_content_s is None:
                    first_content_s = time.perf_counter()
                content_parts.append(text)
                content_chunks += 1
                if (
                    cancel_after_chunks is not None
                    and content_chunks >= cancel_after_chunks
                ):
                    cancelled = True
                    break
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
    end_s = time.perf_counter()
    return StreamResult(
        lane=lane,
        max_tokens=max_tokens,
        start_s=start_s,
        first_content_s=first_content_s,
        end_s=end_s,
        status_code=status_code,
        content="".join(content_parts),
        finish_reason=finish_reason,
        metrics=metrics,
        usage=usage,
        cancelled=cancelled,
        content_chunks=content_chunks,
    )


def summarize(
    scenario: str,
    repeat: int,
    origin_s: float,
    results: list[StreamResult],
) -> dict[str, Any]:
    lanes = [result.as_dict(origin_s) for result in results]
    completed = [
        result
        for result in results
        if not result.cancelled and not result.server_error
    ]
    failed = [
        result
        for result in results
        if not result.cancelled and result.server_error
    ]
    timed_tokens = sum(
        max(
            0,
            int(
                (result.usage or {}).get("completion_tokens")
                or (result.metrics or {}).get("output_tokens")
                or result.content_chunks
            )
            - 1,
        )
        for result in completed
    )
    first_content = [
        result.first_content_s
        for result in completed
        if result.first_content_s is not None
    ]
    response_window_s = (
        max(result.end_s for result in completed) - min(first_content)
        if completed and first_content
        else 0.0
    )
    decode_starts = [
        result.start_s
        + float((result.metrics or {})["time_to_first_token_ms"]) / 1_000.0
        for result in completed
        if (result.metrics or {}).get("time_to_first_token_ms") is not None
        and (result.metrics or {}).get("decode_ms") is not None
    ]
    decode_ends = [
        result.start_s
        + (
            float((result.metrics or {})["time_to_first_token_ms"])
            + float((result.metrics or {})["decode_ms"])
        )
        / 1_000.0
        for result in completed
        if (result.metrics or {}).get("time_to_first_token_ms") is not None
        and (result.metrics or {}).get("decode_ms") is not None
    ]
    decode_window_s = (
        max(decode_ends) - min(decode_starts)
        if decode_starts and decode_ends
        else 0.0
    )
    completed_captures = [
        lane["runtime_captures"]
        for lane in lanes
        if not lane["cancelled"] and lane["runtime_captures"] is not None
    ]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario,
        "repeat": repeat,
        "timed_tokens": timed_tokens,
        "decode_window_ms": decode_window_s * 1_000.0,
        "aggregate_decode_tps": (
            timed_tokens / decode_window_s if decode_window_s > 0.0 else None
        ),
        "response_window_ms": response_window_s * 1_000.0,
        "aggregate_response_window_tps": (
            timed_tokens / response_window_s if response_window_s > 0.0 else None
        ),
        "completed_requests": len(completed),
        "failed_requests": len(failed),
        "cancelled_requests": sum(result.cancelled for result in results),
        "all_http_200": all(result.status_code == 200 for result in results),
        "all_non_cancelled_successful": not failed,
        "all_zero_runtime_captures": bool(completed_captures)
        and all(capture == 0 for capture in completed_captures),
        "lanes": lanes,
    }


async def run_batch(
    args: argparse.Namespace,
    repeat: int,
) -> dict[str, Any]:
    limits = [int(value) for value in args.max_tokens.split(",")]
    delays = [float(value) for value in args.delays.split(",")]
    timeout = httpx.Timeout(args.timeout)
    limits = (limits * args.concurrency)[: args.concurrency]
    delays = (delays * args.concurrency)[: args.concurrency]
    origin_s = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        if args.scenario in {"steady", "staggered-drain"}:
            tasks = [
                asyncio.create_task(
                    stream_request(
                        client,
                        base_url=args.base_url,
                        model=args.model,
                        lane=f"lane-{lane}",
                        max_tokens=limits[lane],
                        delay_s=0.0 if args.scenario == "steady" else delays[lane],
                    )
                )
                for lane in range(args.concurrency)
            ]
            results = list(await asyncio.gather(*tasks))
        else:
            if args.concurrency < 2:
                raise ValueError("cancel-replace requires concurrency >= 2")
            cancel_task = asyncio.create_task(
                stream_request(
                    client,
                    base_url=args.base_url,
                    model=args.model,
                    lane="cancelled",
                    max_tokens=max(limits),
                    cancel_after_chunks=args.cancel_after_chunks,
                )
            )
            peer_tasks = [
                asyncio.create_task(
                    stream_request(
                        client,
                        base_url=args.base_url,
                        model=args.model,
                        lane=f"peer-{lane}",
                        max_tokens=limits[lane],
                    )
                )
                for lane in range(args.concurrency - 1)
            ]
            cancelled = await cancel_task
            replacement = asyncio.create_task(
                stream_request(
                    client,
                    base_url=args.base_url,
                    model=args.model,
                    lane="replacement",
                    max_tokens=args.replacement_max_tokens,
                )
            )
            results = [cancelled, *(await asyncio.gather(*peer_tasks)), await replacement]
    return summarize(args.scenario, repeat, origin_s, results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:8000/v1"
    )
    parser.add_argument(
        "--model", default="lukealonso/GLM-5.2-NVFP4-full"
    )
    parser.add_argument(
        "--scenario",
        choices=("steady", "staggered-drain", "cancel-replace"),
        default="staggered-drain",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", default="64,128,192,320")
    parser.add_argument("--delays", default="0,0.25,0.5,0.75")
    parser.add_argument("--cancel-after-chunks", type=int, default=12)
    parser.add_argument("--replacement-max-tokens", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 16:
        parser.error("--concurrency must be in 1..16")
    return args


async def main() -> None:
    args = parse_args()
    records: list[dict[str, Any]] = []
    for repeat in range(1 - args.warmups, args.repeats + 1):
        record = await run_batch(args, repeat)
        record["warmup"] = repeat <= 0
        print(json.dumps(record, sort_keys=True), flush=True)
        if repeat > 0:
            records.append(record)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record, sort_keys=True) + "\n")
    tps = [
        record["aggregate_response_window_tps"]
        for record in records
        if record["aggregate_response_window_tps"] is not None
    ]
    decode_tps = [
        record["aggregate_decode_tps"]
        for record in records
        if record["aggregate_decode_tps"] is not None
    ]
    if tps:
        print(
            json.dumps(
                {
                    "samples": len(tps),
                    "median_aggregate_decode_tps": (
                        statistics.median(decode_tps) if decode_tps else None
                    ),
                    "median_aggregate_response_window_tps": statistics.median(
                        tps
                    ),
                    "min_aggregate_response_window_tps": min(tps),
                    "max_aggregate_response_window_tps": max(tps),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
