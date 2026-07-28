#!/usr/bin/env python3
"""Normalize dSpark shadow traces and replay cheap confidence policies offline.

Shadow serving advances the target model one token at a time while dSpark
records the complete proposal/confidence chain at every target frontier.  A
policy replay can therefore select D drafts, observe the known accepted prefix,
and jump by ``1 + accepted`` frontiers without changing the target trajectory.
Cycle time is estimated from the maintained balanced-profile prior; this tool
compares controller policy, not kernel timing.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


TRACE_PREFIX = "real_full_dspark_shadow_policy_trace "
TRACE_SCHEMA = "glmrt-dspark-shadow-policy-trace-v1"
NORMALIZED_SCHEMA = "glmrt-dspark-policy-trace-bank-v1"
SUMMARY_SCHEMA = "glmrt-dspark-policy-replay-summary-v1"
INTERNAL_SEQUENCE_PREFIX = "real-full-startup-"
CONTEXT_BUCKET_TOKENS = 32 * 1024
CONTEXT_BUCKET_PRIOR_MS = 2.0
PHYSICAL_M_CYCLE_MS = (
    51.02,
    79.05,
    97.98,
    108.90,
    120.47,
    131.21,
    142.90,
    149.43,
    159.10,
    164.54,
    173.30,
    187.73,
    203.16,
    213.94,
    218.58,
    220.62,
)


@dataclass(frozen=True)
class TraceRecord:
    sequence_id: str
    origin_context: int
    proposal_token_ids: tuple[int, ...]
    conditional_confidence: tuple[float, ...]
    accepted_prefix: int
    observed_positions: int
    resolution: str
    target_token: int | None
    sequence_ordinal: int = -1
    label: str = "unlabeled"


@dataclass
class Observation:
    confidence: tuple[float, ...]
    accepted: int


@dataclass
class Fit:
    bias: float
    variance: float


class ConfidencePolicy:
    name = "base"

    def prepare_context(self, context_tokens: int) -> None:
        del context_tokens

    def adjusted(self, confidence: tuple[float, ...]) -> list[float]:
        raise NotImplementedError

    def force_probe(self) -> bool:
        return False

    def record_selection(self, drafts: int) -> None:
        del drafts

    def observe(
        self,
        raw_confidence: tuple[float, ...],
        accepted: int,
    ) -> None:
        del raw_confidence, accepted


class RawPolicy(ConfidencePolicy):
    name = "raw"

    def adjusted(self, confidence: tuple[float, ...]) -> list[float]:
        return list(confidence)


class ScalarLogitPolicy(ConfidencePolicy):
    def __init__(
        self,
        name: str,
        *,
        window: int,
        prior_precision: float,
        bias_limit: float,
        recency: bool,
        probe: bool,
    ) -> None:
        self.name = name
        self.window = window
        self.prior_precision = prior_precision
        self.bias_limit = bias_limit
        self.recency = recency
        self.probe = probe
        self.observations: deque[Observation] = deque()
        self.bias = 0.0
        self.variance = 1.0 / prior_precision
        self.zero_draft_plans = 0

    def adjusted(self, confidence: tuple[float, ...]) -> list[float]:
        return [apply_logit_bias(value, self.bias) for value in confidence]

    def force_probe(self) -> bool:
        if not self.probe:
            return False
        uncertainty = min(max(self.variance / 4.0, 0.0), 1.0)
        interval = round(16.0 - uncertainty * 12.0)
        return self.zero_draft_plans >= max(interval, 1)

    def record_selection(self, drafts: int) -> None:
        self.zero_draft_plans = self.zero_draft_plans + 1 if drafts == 0 else 0

    def observe(
        self,
        raw_confidence: tuple[float, ...],
        accepted: int,
    ) -> None:
        if not raw_confidence or accepted > len(raw_confidence):
            return
        self.observations.append(Observation(raw_confidence, accepted))
        while len(self.observations) > self.window:
            self.observations.popleft()
        if self.recency:
            fit = fit_current_scalar_bias(
                self.observations,
                previous_bias=self.bias,
                prior_precision=self.prior_precision,
                bias_limit=self.bias_limit,
            )
        else:
            fit = fit_legacy_scalar_bias(
                self.observations,
                prior_precision=self.prior_precision,
                bias_limit=self.bias_limit,
            )
        self.bias = fit.bias
        self.variance = fit.variance


class PositionGradientPolicy(ConfidencePolicy):
    """One shared and 15 position-local online logistic residuals.

    This is deliberately tiny: each observed draft position performs two
    multiply-add updates and two clamps.  The shared term handles request-wide
    confidence drift; the local term can learn that later dSpark positions have
    a different calibration error.
    """

    def __init__(
        self,
        global_rate: float,
        position_rate: float,
        shrink: float,
        bias_limit: float,
    ) -> None:
        self.name = (
            f"position-gradient-g{global_rate:g}-p{position_rate:g}"
            f"-s{shrink:g}-l{bias_limit:g}"
        )
        self.global_rate = global_rate
        self.position_rate = position_rate
        self.shrink = shrink
        self.bias_limit = bias_limit
        self.global_bias = 0.0
        self.position_bias = [0.0] * 15

    def adjusted(self, confidence: tuple[float, ...]) -> list[float]:
        return [
            apply_logit_bias(value, self.global_bias + self.position_bias[index])
            for index, value in enumerate(confidence)
        ]

    def observe(
        self,
        raw_confidence: tuple[float, ...],
        accepted: int,
    ) -> None:
        observed = (
            accepted + 1 if accepted < len(raw_confidence) else accepted
        )
        for position, raw in enumerate(raw_confidence[:observed]):
            bias = self.global_bias + self.position_bias[position]
            predicted = apply_logit_bias(raw, bias)
            outcome = 1.0 if position < accepted else 0.0
            error = outcome - predicted
            self.global_bias = clamp(
                self.global_bias + self.global_rate * error,
                -self.bias_limit,
                self.bias_limit,
            )
            self.position_bias[position] = clamp(
                self.position_bias[position] * self.shrink
                + self.position_rate * error,
                -self.bias_limit,
                self.bias_limit,
            )


class AdaptiveResidualPolicy(ConfidencePolicy):
    """Replay the production candidate selected from the trace-bank sweep."""

    def __init__(
        self,
        *,
        long_context_bias: float,
        context_shape: str,
        context_start: int,
        context_tau: float,
        global_rate: float,
        global_decay: float,
        position_rate: float,
        position_decay: float,
        bias_limit: float,
    ) -> None:
        self.name = (
            f"adaptive-residual-l{long_context_bias:g}"
            f"-c{context_shape}-s{context_start}-t{context_tau:g}"
            f"-g{global_rate:g}x{global_decay:g}"
            f"-p{position_rate:g}x{position_decay:g}"
        )
        self.long_context_bias = long_context_bias
        self.context_shape = context_shape
        self.context_start = context_start
        self.context_tau = context_tau
        self.global_rate = global_rate
        self.global_decay = global_decay
        self.position_rate = position_rate
        self.position_decay = position_decay
        self.bias_limit = bias_limit
        self.context_bias = 0.0
        self.dynamic_bias = 0.0
        self.position_bias = [0.0] * 15

    def prepare_context(self, context_tokens: int) -> None:
        if self.context_shape == "step":
            fraction = float(context_tokens >= CONTEXT_BUCKET_TOKENS)
        elif self.context_shape == "linear":
            fraction = min(
                max(context_tokens - self.context_start, 0) / self.context_tau,
                1.0,
            )
        else:
            excess = max(context_tokens - self.context_start, 0)
            fraction = 1.0 - math.exp(-excess / self.context_tau)
        self.context_bias = self.long_context_bias * fraction

    def adjusted(self, confidence: tuple[float, ...]) -> list[float]:
        return [
            apply_logit_bias(
                value,
                self.context_bias + self.dynamic_bias + self.position_bias[index],
            )
            for index, value in enumerate(confidence)
        ]

    def record_selection(self, drafts: int) -> None:
        if drafts == 0:
            self.dynamic_bias *= self.global_decay

    def observe(
        self,
        raw_confidence: tuple[float, ...],
        accepted: int,
    ) -> None:
        if not raw_confidence or accepted > len(raw_confidence):
            return
        observed = accepted + 1 if accepted < len(raw_confidence) else accepted
        errors = []
        global_bias = self.context_bias + self.dynamic_bias
        for position, raw in enumerate(raw_confidence[:observed]):
            predicted = apply_logit_bias(
                raw,
                global_bias + self.position_bias[position],
            )
            outcome = 1.0 if position < accepted else 0.0
            error = outcome - predicted
            errors.append(error)
            self.position_bias[position] = clamp(
                self.position_decay * self.position_bias[position]
                + self.position_rate * error,
                -self.bias_limit,
                self.bias_limit,
            )
        if errors:
            self.dynamic_bias = clamp(
                self.global_decay * self.dynamic_bias
                + self.global_rate * (sum(errors) / len(errors)),
                -self.bias_limit,
                self.bias_limit,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        required=True,
        help="server log or normalized trace JSONL; repeat to merge banks",
    )
    parser.add_argument(
        "--label-cycle",
        help=(
            "comma-separated labels assigned to sequences in first-seen order "
            "and repeated as a cycle"
        ),
    )
    parser.add_argument(
        "--include-internal",
        action="store_true",
        help="retain startup/internal sequences; excluded by default",
    )
    parser.add_argument(
        "--skip-sequences",
        type=int,
        default=0,
        help="skip this many non-internal sequences in first-seen order",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        help="retain at most this many sequences after --skip-sequences",
    )
    parser.add_argument(
        "--normalized-output",
        type=Path,
        help="write validated, label-augmented trace-bank JSONL",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="write replay summary JSON",
    )
    parser.add_argument(
        "--policy",
        action="append",
        choices=("raw", "legacy", "current", "position-gradient", "residual"),
        help="policy to replay; defaults to all five",
    )
    parser.add_argument(
        "--max-drafts",
        type=int,
        default=15,
        help=(
            "maximum production verify drafts per request; use 7 for the "
            "RedHat checkpoint and 15 for Siro"
        ),
    )
    parser.add_argument("--position-global-rate", type=float, default=0.10)
    parser.add_argument("--position-rate", type=float, default=0.35)
    parser.add_argument("--position-shrink", type=float, default=0.98)
    parser.add_argument("--position-bias-limit", type=float, default=4.0)
    parser.add_argument("--residual-long-context-bias", type=float, default=-0.40)
    parser.add_argument(
        "--residual-context-shape",
        choices=("step", "linear", "exponential"),
        default="linear",
    )
    parser.add_argument("--residual-context-start", type=int, default=2 * 1024)
    parser.add_argument("--residual-context-tau", type=float, default=64 * 1024)
    parser.add_argument("--residual-global-rate", type=float, default=0.40)
    parser.add_argument("--residual-global-decay", type=float, default=0.90)
    parser.add_argument("--residual-position-rate", type=float, default=0.01)
    parser.add_argument("--residual-position-decay", type=float, default=0.98)
    parser.add_argument("--residual-bias-limit", type=float, default=4.0)
    return parser.parse_args()


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def probability(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be a finite probability, got {value}")
    return result


def apply_logit_bias(raw_probability: float, bias: float) -> float:
    raw_probability = clamp(raw_probability, 1.0e-6, 1.0 - 1.0e-6)
    logit = math.log(raw_probability / (1.0 - raw_probability)) + bias
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def observed_positions(observation: Observation) -> int:
    return (
        observation.accepted + 1
        if observation.accepted < len(observation.confidence)
        else observation.accepted
    )


def fit_legacy_scalar_bias(
    observations: Iterable[Observation],
    *,
    prior_precision: float,
    bias_limit: float,
) -> Fit:
    observations = tuple(observations)
    bias = 0.0
    curvature = prior_precision
    for _ in range(12):
        gradient = prior_precision * bias
        curvature = prior_precision
        for observation in observations:
            for position, raw in enumerate(
                observation.confidence[: observed_positions(observation)]
            ):
                calibrated = apply_logit_bias(raw, bias)
                outcome = 1.0 if position < observation.accepted else 0.0
                gradient += calibrated - outcome
                curvature += calibrated * (1.0 - calibrated)
        if curvature <= 0.0:
            break
        next_bias = clamp(
            bias - gradient / curvature,
            -bias_limit,
            bias_limit,
        )
        if abs(next_bias - bias) < 1.0e-6:
            bias = next_bias
            break
        bias = next_bias
    return Fit(bias, 1.0 / max(curvature, prior_precision))


def fit_current_scalar_bias(
    observations: Iterable[Observation],
    *,
    previous_bias: float,
    prior_precision: float,
    bias_limit: float,
) -> Fit:
    observations = tuple(observations)
    newest_surprise = 0.0
    if observations:
        newest = observations[-1]
        errors = []
        for position, raw in enumerate(
            newest.confidence[: observed_positions(newest)]
        ):
            predicted = apply_logit_bias(raw, previous_bias)
            outcome = 1.0 if position < newest.accepted else 0.0
            errors.append(abs(predicted - outcome))
        if errors:
            newest_surprise = sum(errors) / len(errors)
    recency_decay = clamp(0.96 - newest_surprise * (0.96 - 0.70), 0.70, 0.96)

    def evaluate(bias: float) -> tuple[float, float]:
        gradient = prior_precision * bias
        curvature = prior_precision
        for age, observation in enumerate(reversed(observations)):
            weight = recency_decay**age
            for position, raw in enumerate(
                observation.confidence[: observed_positions(observation)]
            ):
                calibrated = apply_logit_bias(raw, bias)
                outcome = 1.0 if position < observation.accepted else 0.0
                gradient += weight * (calibrated - outcome)
                curvature += weight * calibrated * (1.0 - calibrated)
        return gradient, curvature

    lower = -bias_limit
    upper = bias_limit
    lower_gradient = evaluate(lower)[0]
    upper_gradient = evaluate(upper)[0]
    if lower_gradient >= 0.0:
        bias = lower
    elif upper_gradient <= 0.0:
        bias = upper
    else:
        for _ in range(40):
            midpoint = 0.5 * (lower + upper)
            if evaluate(midpoint)[0] < 0.0:
                lower = midpoint
            else:
                upper = midpoint
        bias = 0.5 * (lower + upper)
    curvature = evaluate(bias)[1]
    return Fit(bias, 1.0 / max(curvature, prior_precision))


def cycle_ms(physical_m: int, context_tokens: int) -> float:
    if not 1 <= physical_m <= len(PHYSICAL_M_CYCLE_MS):
        raise ValueError(f"physical M must be in 1..16, got {physical_m}")
    return (
        PHYSICAL_M_CYCLE_MS[physical_m - 1]
        + (context_tokens // CONTEXT_BUCKET_TOKENS) * CONTEXT_BUCKET_PRIOR_MS
    )


def select_drafts(
    adjusted_confidence: list[float],
    context_tokens: int,
    force_probe: bool,
) -> int:
    best_drafts = 0
    expected_tokens = 1.0
    best_tps = 1_000.0 / cycle_ms(1, context_tokens)
    survival = 1.0
    for drafts, confidence in enumerate(adjusted_confidence, start=1):
        survival *= confidence
        expected_tokens += survival
        candidate_tps = (
            expected_tokens * 1_000.0 / cycle_ms(drafts + 1, context_tokens)
        )
        if candidate_tps > best_tps:
            best_tps = candidate_tps
            best_drafts = drafts
    if force_probe and best_drafts == 0 and adjusted_confidence:
        return 1
    return best_drafts


def parse_trace_object(value: dict[str, Any]) -> TraceRecord:
    if value.get("schema") != TRACE_SCHEMA:
        raise ValueError(f"unexpected trace schema {value.get('schema')!r}")
    proposal = tuple(int(token) for token in value["proposal_token_ids"])
    confidence = tuple(
        probability(item, "conditional confidence")
        for item in value["conditional_confidence"]
    )
    if not proposal or len(proposal) != len(confidence):
        raise ValueError("proposal IDs and confidence must have equal nonzero length")
    accepted = int(value["accepted_prefix"])
    observed = int(value["observed_positions"])
    resolution = str(value["resolution"])
    if resolution not in {"full_match", "mismatch", "request_end"}:
        raise ValueError(f"unknown trace resolution {resolution!r}")
    if not 0 <= accepted <= len(proposal):
        raise ValueError(f"accepted prefix {accepted} is outside proposal")
    expected_observed = {
        "full_match": len(proposal),
        "mismatch": accepted + 1,
        "request_end": accepted,
    }[resolution]
    if observed != expected_observed:
        raise ValueError(
            f"{resolution} observed {observed}, expected {expected_observed}"
        )
    if resolution == "full_match" and accepted != len(proposal):
        raise ValueError("full-match trace did not accept the full proposal")
    target = value.get("target_token")
    return TraceRecord(
        sequence_id=str(value["sequence_id"]),
        origin_context=int(value["origin_context"]),
        proposal_token_ids=proposal,
        conditional_confidence=confidence,
        accepted_prefix=accepted,
        observed_positions=observed,
        resolution=resolution,
        target_token=None if target is None else int(target),
        sequence_ordinal=int(value.get("sequence_ordinal", -1)),
        label=str(value.get("label", "unlabeled")),
    )


def trace_objects(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8", errors="replace") as source:
        for line_number, line in enumerate(source, start=1):
            marker = line.find(TRACE_PREFIX)
            if marker >= 0:
                encoded = line[marker + len(TRACE_PREFIX) :].strip()
            elif line.lstrip().startswith("{"):
                encoded = line.strip()
            else:
                continue
            try:
                value = json.loads(encoded)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if value.get("schema") == NORMALIZED_SCHEMA:
                continue
            if value.get("schema") == TRACE_SCHEMA:
                yield value


def load_traces(
    paths: list[Path],
    label_cycle: list[str],
    include_internal: bool,
    skip_sequences: int,
    max_sequences: int | None,
) -> tuple[list[TraceRecord], list[str]]:
    raw = []
    first_seen: dict[str, int] = {}
    sources = []
    for path in paths:
        resolved = path.resolve()
        sources.append(str(resolved))
        for value in trace_objects(resolved):
            sequence_id = str(value["sequence_id"])
            if (
                not include_internal
                and sequence_id.startswith(INTERNAL_SEQUENCE_PREFIX)
            ):
                continue
            if sequence_id not in first_seen:
                first_seen[sequence_id] = len(first_seen)
            ordinal = first_seen[sequence_id]
            selected_ordinal = ordinal - skip_sequences
            if selected_ordinal < 0 or (
                max_sequences is not None
                and selected_ordinal >= max_sequences
            ):
                continue
            label = (
                label_cycle[selected_ordinal % len(label_cycle)]
                if label_cycle
                else str(value.get("label", "unlabeled"))
            )
            value = dict(value)
            value["sequence_ordinal"] = selected_ordinal
            value["label"] = label
            raw.append(parse_trace_object(value))
    if not raw:
        raise ValueError("no dSpark shadow policy traces found")
    unique: dict[tuple[str, int], TraceRecord] = {}
    for record in raw:
        key = (record.sequence_id, record.origin_context)
        if key in unique:
            raise ValueError(
                f"duplicate trace for {record.sequence_id} at {record.origin_context}"
            )
        unique[key] = record
    records = sorted(
        unique.values(),
        key=lambda item: (item.sequence_ordinal, item.origin_context),
    )
    return records, sources


def make_policy(name: str, args: argparse.Namespace) -> ConfidencePolicy:
    if name == "raw":
        return RawPolicy()
    if name == "legacy":
        return ScalarLogitPolicy(
            "legacy-scalar-logit",
            window=16,
            prior_precision=2.0,
            bias_limit=2.0,
            recency=False,
            probe=False,
        )
    if name == "current":
        return ScalarLogitPolicy(
            "current-scalar-logit",
            window=16,
            prior_precision=0.25,
            bias_limit=13.0,
            recency=True,
            probe=True,
        )
    if name == "position-gradient":
        return PositionGradientPolicy(
            args.position_global_rate,
            args.position_rate,
            args.position_shrink,
            args.position_bias_limit,
        )
    if name == "residual":
        return AdaptiveResidualPolicy(
            long_context_bias=args.residual_long_context_bias,
            context_shape=args.residual_context_shape,
            context_start=args.residual_context_start,
            context_tau=args.residual_context_tau,
            global_rate=args.residual_global_rate,
            global_decay=args.residual_global_decay,
            position_rate=args.residual_position_rate,
            position_decay=args.residual_position_decay,
            bias_limit=args.residual_bias_limit,
        )
    raise ValueError(f"unknown policy {name}")


def replay_sequence(
    records: list[TraceRecord],
    policy: ConfidencePolicy,
    max_drafts: int,
) -> dict[str, Any]:
    ordered_contexts = sorted(record.origin_context for record in records)
    runs: list[list[int]] = []
    for origin_context in ordered_contexts:
        if not runs or origin_context != runs[-1][-1] + 1:
            runs.append([origin_context])
        else:
            runs[-1].append(origin_context)
    decode_contexts = max(runs, key=lambda run: (len(run), run[0]))
    by_context = {
        record.origin_context: record
        for record in records
        if decode_contexts[0] <= record.origin_context <= decode_contexts[-1]
    }
    context = min(by_context)
    maximum_context = max(by_context)
    decisions = 0
    emitted = 0
    draft_tokens = 0
    accepted_tokens = 0
    predicted_ms = 0.0
    censored_cycles = 0
    missing_frontiers = 0
    physical_m: Counter[int] = Counter()
    while context in by_context:
        record = by_context[context]
        # This window was pending after the request had already emitted its
        # terminal token. It carries useful censoring metadata for calibration
        # analysis, but there is no subsequent target cycle to schedule.
        if record.resolution == "request_end":
            censored_cycles += 1
            break
        policy.prepare_context(context)
        adjusted = policy.adjusted(record.conditional_confidence)[:max_drafts]
        drafts = select_drafts(adjusted, context, policy.force_probe())
        accepted = min(drafts, record.accepted_prefix)
        decisions += 1
        draft_tokens += drafts
        accepted_tokens += accepted
        cycle_emitted = 1 + accepted
        emitted += cycle_emitted
        predicted_ms += cycle_ms(drafts + 1, context)
        physical_m[drafts + 1] += 1
        policy.record_selection(drafts)
        if drafts:
            policy.observe(record.conditional_confidence[:drafts], accepted)
        next_context = context + cycle_emitted
        if next_context not in by_context:
            if next_context <= maximum_context:
                missing_frontiers += 1
            break
        context = next_context
    return {
        "decisions": decisions,
        "emitted_tokens": emitted,
        "draft_tokens": draft_tokens,
        "accepted_draft_tokens": accepted_tokens,
        "predicted_ms": predicted_ms,
        "censored_cycles": censored_cycles,
        "missing_frontiers": missing_frontiers,
        "physical_m_histogram": dict(sorted(physical_m.items())),
    }


def replay(
    records: list[TraceRecord],
    policy_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    sequences: dict[str, list[TraceRecord]] = {}
    for record in records:
        sequences.setdefault(record.sequence_id, []).append(record)
    totals: Counter[str] = Counter()
    physical_m: Counter[int] = Counter()
    labels: Counter[str] = Counter()
    per_sequence = []
    for sequence_records in sequences.values():
        policy = make_policy(policy_name, args)
        result = replay_sequence(sequence_records, policy, args.max_drafts)
        for key in (
            "decisions",
            "emitted_tokens",
            "draft_tokens",
            "accepted_draft_tokens",
            "censored_cycles",
            "missing_frontiers",
        ):
            totals[key] += result[key]
        predicted_ms = float(result["predicted_ms"])
        totals["predicted_us"] += round(predicted_ms * 1_000.0)
        physical_m.update(
            {
                int(key): value
                for key, value in result["physical_m_histogram"].items()
            }
        )
        label = sequence_records[0].label
        labels[label] += 1
        per_sequence.append(
            {
                "sequence_ordinal": sequence_records[0].sequence_ordinal,
                "label": label,
                **result,
                "predicted_tps": (
                    result["emitted_tokens"] * 1_000.0 / predicted_ms
                    if predicted_ms
                    else 0.0
                ),
            }
        )
    predicted_ms = totals["predicted_us"] / 1_000.0
    return {
        "policy": make_policy(policy_name, args).name,
        "sequences": len(sequences),
        "labels": dict(sorted(labels.items())),
        "decisions": totals["decisions"],
        "emitted_tokens": totals["emitted_tokens"],
        "draft_tokens": totals["draft_tokens"],
        "accepted_draft_tokens": totals["accepted_draft_tokens"],
        "accepted_draft_rate": (
            totals["accepted_draft_tokens"] / totals["draft_tokens"]
            if totals["draft_tokens"]
            else 0.0
        ),
        "predicted_ms": predicted_ms,
        "predicted_tps": (
            totals["emitted_tokens"] * 1_000.0 / predicted_ms
            if predicted_ms
            else 0.0
        ),
        "physical_m_histogram": dict(sorted(physical_m.items())),
        "censored_cycles": totals["censored_cycles"],
        "missing_frontiers": totals["missing_frontiers"],
        "per_sequence": per_sequence,
    }


def calibration_group(records: Iterable[TraceRecord]) -> dict[str, Any]:
    observations = []
    position_observations: dict[int, list[Observation]] = {}
    raw_sum = 0.0
    accepted = 0
    observed = 0
    for record in records:
        confidence = record.conditional_confidence[: record.observed_positions]
        if not confidence:
            continue
        accepted_prefix = min(record.accepted_prefix, len(confidence))
        observation = Observation(confidence, accepted_prefix)
        observations.append(observation)
        for position, raw in enumerate(confidence):
            outcome = int(position < accepted_prefix)
            raw_sum += raw
            accepted += outcome
            observed += 1
            position_observations.setdefault(position, []).append(
                Observation((raw,), outcome)
            )
    fit = fit_legacy_scalar_bias(
        observations,
        prior_precision=1.0e-3,
        bias_limit=13.0,
    )
    return {
        "records": len(observations),
        "observed_positions": observed,
        "accepted_positions": accepted,
        "raw_probability_mean": raw_sum / observed if observed else 0.0,
        "observed_acceptance_rate": accepted / observed if observed else 0.0,
        "fitted_logit_bias": fit.bias,
        "by_position": {
            str(position + 1): {
                "samples": len(items),
                "fitted_logit_bias": fit_legacy_scalar_bias(
                    items,
                    prior_precision=1.0e-3,
                    bias_limit=13.0,
                ).bias,
            }
            for position, items in sorted(position_observations.items())
        },
    }


def calibration_summary(records: list[TraceRecord]) -> dict[str, Any]:
    labels: dict[str, list[TraceRecord]] = {}
    for record in records:
        labels.setdefault(record.label, []).append(record)
    return {
        "overall": calibration_group(records),
        "by_label": {
            label: calibration_group(items)
            for label, items in sorted(labels.items())
        },
    }


def write_normalized(
    path: Path,
    records: list[TraceRecord],
    sources: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sequences = len({record.sequence_id for record in records})
    manifest = {
        "schema": NORMALIZED_SCHEMA,
        "sources": sources,
        "sequences": sequences,
        "records": len(records),
        "labels": dict(Counter(record.label for record in records)),
    }
    with path.open("w", encoding="utf-8") as destination:
        destination.write(json.dumps(manifest, sort_keys=True) + "\n")
        for record in records:
            value = asdict(record)
            value["schema"] = TRACE_SCHEMA
            destination.write(json.dumps(value, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    labels = (
        [label.strip() for label in args.label_cycle.split(",") if label.strip()]
        if args.label_cycle
        else []
    )
    if args.label_cycle and not labels:
        raise SystemExit("--label-cycle must contain at least one label")
    if args.position_global_rate < 0.0 or args.position_rate < 0.0:
        raise SystemExit("position learning rates must be nonnegative")
    if not 0.0 <= args.position_shrink <= 1.0:
        raise SystemExit("--position-shrink must be in 0..1")
    if args.position_bias_limit <= 0.0:
        raise SystemExit("--position-bias-limit must be positive")
    for name in ("residual_global_rate", "residual_position_rate"):
        if getattr(args, name) < 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be nonnegative")
    for name in ("residual_global_decay", "residual_position_decay"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be in 0..1")
    if args.residual_bias_limit <= 0.0:
        raise SystemExit("--residual-bias-limit must be positive")
    if args.residual_context_start < 0:
        raise SystemExit("--residual-context-start must be nonnegative")
    if args.residual_context_tau <= 0.0:
        raise SystemExit("--residual-context-tau must be positive")
    if args.skip_sequences < 0:
        raise SystemExit("--skip-sequences must be nonnegative")
    if args.max_sequences is not None and args.max_sequences < 1:
        raise SystemExit("--max-sequences must be positive")
    if not 1 <= args.max_drafts <= 15:
        raise SystemExit("--max-drafts must be in 1..15")
    records, sources = load_traces(
        args.input,
        labels,
        args.include_internal,
        args.skip_sequences,
        args.max_sequences,
    )
    policies = args.policy or [
        "raw",
        "legacy",
        "current",
        "position-gradient",
        "residual",
    ]
    results = [replay(records, policy, args) for policy in policies]
    summary = {
        "schema": SUMMARY_SCHEMA,
        "sources": sources,
        "trace_records": len(records),
        "sequences": len({record.sequence_id for record in records}),
        "calibration": calibration_summary(records),
        "results": results,
    }
    if args.normalized_output:
        write_normalized(args.normalized_output.resolve(), records, sources)
    if args.summary_output:
        output = args.summary_output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
