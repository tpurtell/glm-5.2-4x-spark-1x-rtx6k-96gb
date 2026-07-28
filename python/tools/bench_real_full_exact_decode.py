#!/usr/bin/env python3
"""Run pinned exact-output real-full decode gates against a serving coordinator."""

from __future__ import annotations

import argparse
import json
import statistics
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class Fixture:
    beta_repetitions: int
    prompt_tokens: int
    max_tokens: int
    final_count: int


FIXTURES = {
    "mtp-794-25": Fixture(761, 794, 25, 13),
    "no-mtp-979-99": Fixture(946, 979, 99, 50),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", choices=FIXTURES)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default="lukealonso/GLM-5.2-NVFP4")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")

    fixture = FIXTURES[args.fixture]
    content = (
        "Background words below are irrelevant. "
        + "beta " * fixture.beta_repetitions
        + "\nCount from 1 to 50, one number per line. Do not add any other text."
    )
    expected = "\n".join(str(value) for value in range(1, fixture.final_count + 1))
    payload = json.dumps(
        {
            "model": args.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": fixture.max_tokens,
        }
    ).encode()

    samples = []
    for sample_index in range(args.repeats):
        request = urllib.request.Request(
            args.url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result = json.load(response)

        usage = result["usage"]
        metrics = result["metrics"]
        real_full = metrics["real_full"]
        output = result["choices"][0]["message"]["content"]
        decode_ms = metrics["decode_ms"]
        decode_tps = (
            (usage["completion_tokens"] - 1) / (decode_ms / 1_000.0)
            if usage["completion_tokens"] > 1 and decode_ms > 0.0
            else 0.0
        )
        sample = {
            "sample": sample_index + 1,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "prefill_ms": metrics["prefill_ms"],
            "decode_ms": decode_ms,
            "decode_tps": decode_tps,
            "exact": output == expected,
            "runtime_captures": real_full["request_coordinator_graph_captures"],
            "mtp_verify_cycles": real_full["mtp_verify_cycles"],
            "mtp_draft_tokens": real_full["mtp_draft_tokens"],
            "mtp_accepted_draft_tokens": real_full["mtp_accepted_draft_tokens"],
            "mtp_accepted_draft_lengths": real_full["mtp_accepted_draft_lengths"],
            "mtp_full_match_cycles": real_full["mtp_full_match_cycles"],
            "mtp_total_verify_cycle_ms": real_full["mtp_total_verify_cycle_ms"],
        }
        samples.append(sample)
        print(json.dumps(sample), flush=True)

    summary = {
        "fixture": args.fixture,
        "samples": len(samples),
        "mean_decode_tps": statistics.mean(
            sample["decode_tps"] for sample in samples
        ),
        "median_decode_tps": statistics.median(
            sample["decode_tps"] for sample in samples
        ),
        "min_decode_tps": min(sample["decode_tps"] for sample in samples),
        "max_decode_tps": max(sample["decode_tps"] for sample in samples),
        "stdev_decode_tps": (
            statistics.stdev(sample["decode_tps"] for sample in samples)
            if len(samples) > 1
            else 0.0
        ),
        "all_exact": all(sample["exact"] for sample in samples),
        "all_zero_runtime_captures": all(
            sample["runtime_captures"] == 0 for sample in samples
        ),
        "expected_prompt_tokens": fixture.prompt_tokens,
        "all_prompt_token_counts_match": all(
            sample["prompt_tokens"] == fixture.prompt_tokens for sample in samples
        ),
        "expected_completion_tokens": fixture.max_tokens,
        "all_completion_token_counts_match": all(
            sample["completion_tokens"] == fixture.max_tokens for sample in samples
        ),
    }
    print(json.dumps(summary))
    if not (
        summary["all_exact"]
        and summary["all_zero_runtime_captures"]
        and summary["all_prompt_token_counts_match"]
        and summary["all_completion_token_counts_match"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
