#!/usr/bin/env python3
"""Measure native-MTP acceptance on varied semantic generation tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromptCase:
    category: str
    prompt: str
    max_tokens: int


CASES = {
    "count": PromptCase(
        "low-entropy",
        "Count from 1 to 64, one number per line. Do not add any other text.",
        160,
    ),
    "repeat": PromptCase(
        "repetition",
        'Repeat the exact text "red green blue" 24 times, one repetition per line. '
        "Do not number the lines.",
        160,
    ),
    "code": PromptCase(
        "code",
        "Write a Python function merge_intervals(intervals) that merges overlapping "
        "integer intervals. Include type hints, a short docstring, and three assert-based "
        "examples. Return only one Python code block.",
        320,
    ),
    "math": PromptCase(
        "reasoning",
        "A shop discounts a $240 jacket by 25%, then applies 8% sales tax to the "
        "discounted price. What is the final price? Show the calculation briefly.",
        128,
    ),
    "fable": PromptCase(
        "creative-prose",
        "Write a self-contained fable of 140 to 170 words about two parrots who disagree "
        "about sharing credit. End with a one-sentence moral.",
        256,
    ),
    "hello": PromptCase("short-response", "hi", 32),
    "topic": PromptCase(
        "exposition",
        "Explain virtual memory to a junior programmer in five concise bullet points, "
        "including paging, page faults, and the role of the TLB.",
        224,
    ),
    "structured-json": PromptCase(
        "structured-output",
        "Return only a JSON object describing a file edit with keys path, operation, "
        "line_start, line_end, and rationale. Use path src/cache.rs, operation replace, "
        "lines 41 through 47, and a one-sentence rationale about removing a redundant copy.",
        128,
    ),
    "multilingual": PromptCase(
        "multilingual",
        "請用繁體中文，以四個簡短條列解釋什麼是寫入時複製（copy-on-write），"
        "並包含一個行程 fork 後修改記憶體頁面的例子。",
        192,
    ),
}

WEIGHTED_CASE_IDS = tuple(
    case_id for case_id in CASES if case_id not in {"count", "repeat"}
)

# Explicit rare-width diagnostics. They are selectable with --case but are
# excluded from the default weighted corpus because their repetitive syntax is
# deliberately favorable to long speculative windows.
CASES.update(
    {
        "syntax-rust": PromptCase(
            "diagnostic-syntax",
            "Return only a Rust code block declaring enum Op with exactly 128 "
            "variants named Op000 through Op127, one variant per line.",
            512,
        ),
        "syntax-python": PromptCase(
            "diagnostic-syntax",
            "Return only Python code defining POWERS_OF_TWO as a parenthesized "
            "tuple containing 2**0 through 2**127, one expression per line.",
            512,
        ),
    }
)
REACHABILITY_CASE_IDS = ("count", "repeat", "syntax-rust", "syntax-python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default="http://127.0.0.1:8000/v1/chat/completions"
    )
    parser.add_argument("--reference-url")
    parser.add_argument("--model", default="lukealonso/GLM-5.2-NVFP4")
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Override the selected corpus cases' completion-token budgets.",
    )
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        choices=sorted(CASES),
        help="Run only this case; may be repeated. Overrides --suite.",
    )
    parser.add_argument(
        "--suite",
        choices=("weighted", "reachability", "all"),
        default="weighted",
        help=(
            "Corpus used when --case is omitted. The weighted suite excludes "
            "deliberately easy counting/repetition/syntax reachability probes."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Run the complete selected corpus this many times.",
    )
    parser.add_argument(
        "--nonce-seed",
        type=int,
        help=(
            "Prefix every prompt with a deterministic unique nonce, preventing "
            "prompt-cache reuse in paired performance qualification."
        ),
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


def completion_payload(
    model: str,
    case: PromptCase,
    max_tokens: int | None = None,
    prompt_prefix: str = "",
) -> bytes:
    return json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt_prefix + case.prompt}],
            "temperature": 0,
            "max_tokens": case.max_tokens if max_tokens is None else max_tokens,
        }
    ).encode()


def request_completion(url: str, payload: bytes, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def summarize_case(case_id: str, result: dict[str, Any]) -> dict[str, Any]:
    case = CASES[case_id]
    usage = result["usage"]
    metrics = result["metrics"]
    mtp = metrics["real_full"]
    draft_lengths = list(mtp.get("mtp_draft_lengths", []))
    accepted = list(mtp["mtp_accepted_draft_lengths"])
    cycle_ms = list(mtp["mtp_verify_cycle_ms"])
    verify_cycles = int(mtp["mtp_verify_cycles"])
    drafts = int(mtp["mtp_draft_tokens"])
    accepted_total = int(mtp["mtp_accepted_draft_tokens"])
    emitted = int(mtp["mtp_emitted_tokens_from_verify"])
    total_cycle_ms = float(mtp["mtp_total_verify_cycle_ms"])
    decode_ms = float(metrics["decode_ms"])
    completion_tokens = int(usage["completion_tokens"])
    content = result["choices"][0]["message"]["content"]
    return {
        "case": case_id,
        "category": case.category,
        "prompt_tokens": int(usage["prompt_tokens"]),
        "completion_tokens": completion_tokens,
        "finish_reason": result["choices"][0]["finish_reason"],
        "decode_ms": decode_ms,
        "decode_tps": (
            (completion_tokens - 1) / (decode_ms / 1_000.0)
            if completion_tokens > 1 and decode_ms > 0.0
            else 0.0
        ),
        "verify_cycles": verify_cycles,
        "draft_tokens": drafts,
        "accepted_draft_tokens": accepted_total,
        "accepted_draft_rate": accepted_total / drafts if drafts else 0.0,
        "draft_lengths": draft_lengths,
        "mean_accepted_draft_length": statistics.mean(accepted) if accepted else 0.0,
        "accepted_draft_lengths": accepted,
        "verify_cycle_ms": cycle_ms,
        "full_match_cycles": int(mtp["mtp_full_match_cycles"]),
        "emitted_tokens_from_verify": emitted,
        "emitted_tokens_per_verify_cycle": emitted / verify_cycles
        if verify_cycles
        else 0.0,
        "emitted_tokens_per_verify_cycle_second": emitted
        / (total_cycle_ms / 1_000.0)
        if total_cycle_ms > 0.0
        else 0.0,
        "runtime_captures": int(mtp["request_coordinator_graph_captures"]),
        "content_chars": len(content),
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "content_preview": content[:160].replace("\n", "\\n"),
    }


def main() -> None:
    args = parse_args()
    if args.max_tokens is not None and args.max_tokens < 1:
        raise SystemExit("--max-tokens must be positive")
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    if args.cases:
        selected = args.cases
        selected_suite = "explicit"
    elif args.suite == "weighted":
        selected = list(WEIGHTED_CASE_IDS)
        selected_suite = "weighted"
    elif args.suite == "reachability":
        selected = list(REACHABILITY_CASE_IDS)
        selected_suite = "reachability"
    else:
        selected = list(CASES)
        selected_suite = "all"
    summaries = []
    repeat_summaries = []
    for repeat_index in range(args.repeats):
        repeat_cases = []
        for case_id in selected:
            prompt_prefix = (
                f"Qualification nonce {args.nonce_seed}-{repeat_index}-{case_id}. "
                "Treat this identifier as irrelevant.\n"
                if args.nonce_seed is not None
                else ""
            )
            payload = completion_payload(
                args.model,
                CASES[case_id],
                args.max_tokens,
                prompt_prefix,
            )
            result = request_completion(args.url, payload, args.timeout)
            summary = summarize_case(case_id, result)
            summary["repeat"] = repeat_index + 1
            if args.reference_url:
                reference = request_completion(args.reference_url, payload, args.timeout)
                summary["reference_content_match"] = (
                    reference["choices"][0]["message"]["content"]
                    == result["choices"][0]["message"]["content"]
                )
                summary["reference_finish_reason_match"] = (
                    reference["choices"][0]["finish_reason"]
                    == summary["finish_reason"]
                )
                summary["reference_completion_tokens_match"] = (
                    int(reference["usage"]["completion_tokens"])
                    == summary["completion_tokens"]
                )
            repeat_cases.append(summary)
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)

        repeat_timed_tokens = sum(
            summary["completion_tokens"] - 1 for summary in repeat_cases
        )
        repeat_decode_ms = sum(summary["decode_ms"] for summary in repeat_cases)
        repeat_emitted = sum(
            summary["emitted_tokens_from_verify"] for summary in repeat_cases
        )
        repeat_verify_ms = sum(
            sum(summary["verify_cycle_ms"]) for summary in repeat_cases
        )
        repeat_summaries.append(
            {
                "repeat": repeat_index + 1,
                "wall_decode_tps": (
                    repeat_timed_tokens / (repeat_decode_ms / 1_000.0)
                    if repeat_decode_ms > 0.0
                    else 0.0
                ),
                "emitted_tokens_per_verify_cycle_second": (
                    repeat_emitted / (repeat_verify_ms / 1_000.0)
                    if repeat_verify_ms > 0.0
                    else 0.0
                ),
            }
        )

    accepted_histogram = Counter(
        length for summary in summaries for length in summary["accepted_draft_lengths"]
    )
    draft_histogram = Counter(
        length for summary in summaries for length in summary["draft_lengths"]
    )
    physical_m_histogram = Counter(
        length + 1 for summary in summaries for length in summary["draft_lengths"]
    )
    emitted_length_histogram = Counter(
        length + 1
        for summary in summaries
        for length in summary["accepted_draft_lengths"]
    )
    accepted_by_physical_m: dict[int, Counter[int]] = {}
    full_matches_by_physical_m = Counter()
    scalar_cycles = sum(
        max(
            0,
            summary["completion_tokens"]
            - summary["emitted_tokens_from_verify"],
        )
        for summary in summaries
    )
    physical_m_histogram[1] += scalar_cycles
    emitted_length_histogram[1] += scalar_cycles
    accepted_by_physical_m[1] = Counter({0: scalar_cycles})
    for summary in summaries:
        for drafts, accepted in zip(
            summary["draft_lengths"], summary["accepted_draft_lengths"], strict=True
        ):
            physical_m = drafts + 1
            accepted_by_physical_m.setdefault(physical_m, Counter())[accepted] += 1
            if accepted == drafts:
                full_matches_by_physical_m[physical_m] += 1
    total_drafts = sum(summary["draft_tokens"] for summary in summaries)
    total_accepted = sum(summary["accepted_draft_tokens"] for summary in summaries)
    total_emitted = sum(summary["emitted_tokens_from_verify"] for summary in summaries)
    total_cycles = sum(summary["verify_cycles"] for summary in summaries)
    total_cycle_ms = sum(sum(summary["verify_cycle_ms"]) for summary in summaries)
    total_timed_tokens = sum(summary["completion_tokens"] - 1 for summary in summaries)
    total_decode_ms = sum(summary["decode_ms"] for summary in summaries)
    wall_samples = [summary["wall_decode_tps"] for summary in repeat_summaries]
    verifier_samples = [
        summary["emitted_tokens_per_verify_cycle_second"]
        for summary in repeat_summaries
    ]
    aggregate = {
        "suite": selected_suite,
        "selected_case_ids": selected,
        "cases": len(summaries),
        "cases_per_repeat": len(selected),
        "corpus_repeats": args.repeats,
        "repeat_summaries": repeat_summaries,
        "wall_decode_tps": (
            total_timed_tokens / (total_decode_ms / 1_000.0)
            if total_decode_ms > 0.0
            else 0.0
        ),
        "median_repeat_wall_decode_tps": statistics.median(wall_samples),
        "min_repeat_wall_decode_tps": min(wall_samples),
        "max_repeat_wall_decode_tps": max(wall_samples),
        "stdev_repeat_wall_decode_tps": (
            statistics.stdev(wall_samples) if len(wall_samples) > 1 else 0.0
        ),
        "verify_cycles": total_cycles,
        "scalar_cycles": scalar_cycles,
        "target_cycles": scalar_cycles + total_cycles,
        "draft_tokens": total_drafts,
        "accepted_draft_tokens": total_accepted,
        "accepted_draft_rate": total_accepted / total_drafts if total_drafts else 0.0,
        "draft_length_histogram": dict(sorted(draft_histogram.items())),
        "physical_m_histogram": dict(sorted(physical_m_histogram.items())),
        "accepted_draft_length_histogram": dict(sorted(accepted_histogram.items())),
        "emitted_length_histogram": dict(sorted(emitted_length_histogram.items())),
        "accepted_drafts_by_physical_m": {
            physical_m: dict(sorted(histogram.items()))
            for physical_m, histogram in sorted(accepted_by_physical_m.items())
        },
        "full_matches_by_physical_m": dict(
            sorted(full_matches_by_physical_m.items())
        ),
        "max_selected_physical_m": max(physical_m_histogram, default=1),
        "max_emitted_tokens_in_cycle": max(emitted_length_histogram, default=1),
        "emitted_tokens_from_verify": total_emitted,
        "emitted_tokens_per_verify_cycle": total_emitted / total_cycles
        if total_cycles
        else 0.0,
        "emitted_tokens_per_verify_cycle_second": total_emitted
        / (total_cycle_ms / 1_000.0)
        if total_cycle_ms > 0.0
        else 0.0,
        "median_repeat_emitted_tokens_per_verify_cycle_second": statistics.median(
            verifier_samples
        ),
        "min_repeat_emitted_tokens_per_verify_cycle_second": min(verifier_samples),
        "max_repeat_emitted_tokens_per_verify_cycle_second": max(verifier_samples),
        "stdev_repeat_emitted_tokens_per_verify_cycle_second": (
            statistics.stdev(verifier_samples)
            if len(verifier_samples) > 1
            else 0.0
        ),
        "all_zero_runtime_captures": all(
            summary["runtime_captures"] == 0 for summary in summaries
        ),
    }
    if args.reference_url:
        aggregate["all_reference_outputs_match"] = all(
            summary["reference_content_match"]
            and summary["reference_finish_reason_match"]
            and summary["reference_completion_tokens_match"]
            for summary in summaries
        )
    print(json.dumps({"aggregate": aggregate}, ensure_ascii=False))


if __name__ == "__main__":
    main()
