#!/usr/bin/env python3
"""Build source-disjoint GLM-5.2 EXL3 calibration and held-out corpora."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable

from tokenizers import Tokenizer

# The production corpus was selected with this exact lightweight chat wrapper.
# The JSONL stores the unwrapped prompt; this renderer is used only for stable
# token budgeting and is embedded here so downstream calibration is standalone.
_BOS = "<｜begin▁of▁sentence｜>"
_USER = "<｜User｜>"
_ASSISTANT = "<｜Assistant｜>"
_THINK_CLOSE = "</think>"


def render_nonthinking_messages(messages: list[dict[str, Any]]) -> str:
    if len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError("calibration renderer requires one user message")
    content = messages[0].get("content")
    if not isinstance(content, str):
        raise ValueError("calibration renderer requires text content")
    return f"{_BOS}{_USER}{content}{_ASSISTANT}{_THINK_CLOSE}"


SCHEMA = "ds4rt-flash-exl3-calibration-corpus-v3"
DEFAULT_CALIBRATION_TOKENS = 1_080_000
DEFAULT_HELDOUT_TOKENS = 65_536
DEFAULT_RECORD_TARGET_TOKENS = 768
DEFAULT_MIN_RECORD_TOKENS = 512
DEFAULT_MAX_RECORD_TOKENS = 1_024
HELDOUT_BUCKETS = 4
SCREENING_FRACTION = 0.10
WIKI_FOCUS_LANGUAGES = ("en", "zh")
CODE_PRIORITY_LANGUAGES = ("Python", "C++", "CUDA", "C", "Rust")
AXIS_WEIGHTS = {
    "code_agentic": 360_000,
    "general": 260_000,
    "math_reasoning": 220_000,
    "reasoning_termination": 130_000,
    "structured_output": 110_000,
}
MATH_CHINESE_TOKEN_FRACTION = 0.20
LEGAL_TOPICS = {"law", "politics", "econ"}
REASONING_TOPICS = {"math", "physics", "chem", "cs", "phil", "ai"}
REASONING_FAMILIES = {"mc", "relate"}


@dataclass(frozen=True)
class SourceRecord:
    domain: str
    family: str
    stratum: tuple[str, ...]
    group: str
    identity: str
    prompt: str
    metadata: dict[str, str]
    article_path: Path | None = None


@dataclass(frozen=True)
class PreparedRecord:
    source: SourceRecord
    prompt: str
    prompt_tokens: int
    axis: str = "general"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data", type=Path, default=Path("../training-data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help=(
            "tokenizer.json used for deterministic selection; use the manifest's "
            "recorded tokenizer to reproduce an existing corpus exactly"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--calibration-tokens", type=int, default=DEFAULT_CALIBRATION_TOKENS
    )
    parser.add_argument("--heldout-tokens", type=int, default=DEFAULT_HELDOUT_TOKENS)
    parser.add_argument(
        "--record-target-tokens", type=int, default=DEFAULT_RECORD_TARGET_TOKENS
    )
    parser.add_argument(
        "--min-record-tokens", type=int, default=DEFAULT_MIN_RECORD_TOKENS
    )
    parser.add_argument(
        "--max-record-tokens", type=int, default=DEFAULT_MAX_RECORD_TOKENS
    )
    parser.add_argument("--max-output-tokens", type=int, default=8)
    args = parser.parse_args()
    if args.calibration_tokens < 1 or args.heldout_tokens < 1:
        parser.error("calibration and held-out token targets must be positive")
    if not (
        1
        <= args.min_record_tokens
        <= args.record_target_tokens
        <= args.max_record_tokens
    ):
        parser.error(
            "record token bounds must satisfy 1 <= minimum <= target <= maximum"
        )
    if not 2 <= args.max_output_tokens <= 256:
        parser.error("max output tokens must be in 2..256")
    return args


def stable_digest(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def git_snapshot(repo: Path, *, input_paths: tuple[str, ...]) -> dict[str, Any]:
    def run(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )

    revision_result = run("rev-parse", "HEAD")
    if revision_result.returncode != 0:
        raise ValueError(f"cannot resolve Git revision for {repo}")
    revision = revision_result.stdout.strip()
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError(f"invalid Git revision for {repo}: {revision!r}")
    dirty_result = run("status", "--porcelain", "--", *input_paths)
    if dirty_result.returncode != 0:
        raise ValueError(f"cannot inspect source cleanliness for {repo}")
    dirty = [line for line in dirty_result.stdout.splitlines() if line]
    if dirty:
        raise ValueError(
            f"corpus inputs are not committed under {repo}: {', '.join(dirty[:8])}"
        )
    return {"repository": str(repo), "revision": revision, "input_paths": input_paths}


def load_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"record at {path}:{line_number} is not an object")
            yield line_number, value


def required_text(record: dict[str, Any], key: str, *, location: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} has invalid {key!r}")
    return value


def load_source_records(training_data: Path) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    wiki_root = training_data / "banks" / "wiki"
    for path in sorted(wiki_root.glob("*/*/*.jsonl")):
        for line_number, raw in load_jsonl(path):
            location = f"{path}:{line_number}"
            family = required_text(raw, "type", location=location)
            language = required_text(raw, "language", location=location)
            topic = required_text(raw, "topic", location=location)
            prompt = required_text(raw, "prompt", location=location)
            article = required_text(raw, "article_path", location=location)
            article_path = training_data / article
            identity = stable_digest("wiki", article, family, prompt)
            records.append(
                SourceRecord(
                    domain="wiki",
                    family=family,
                    stratum=(family, language),
                    group=f"wiki:{article}",
                    identity=identity,
                    prompt=prompt,
                    metadata={
                        "language": language,
                        "topic": topic,
                        "article_path": article,
                    },
                    article_path=article_path,
                )
            )

    code_root = training_data / "banks" / "code"
    for path in sorted(code_root.glob("*/*.jsonl")):
        for line_number, raw in load_jsonl(path):
            location = f"{path}:{line_number}"
            family = required_text(raw, "type", location=location)
            repo = required_text(raw, "repo", location=location)
            language = required_text(raw, "language", location=location)
            file_name = required_text(raw, "file", location=location)
            prompt = required_text(raw, "prompt", location=location)
            identity = stable_digest("code", repo, family, file_name, prompt)
            records.append(
                SourceRecord(
                    domain="code",
                    family=family,
                    stratum=(family, language),
                    group=f"code:{repo}",
                    identity=identity,
                    prompt=prompt,
                    metadata={
                        "repo": repo,
                        "language": language,
                        "file": file_name,
                    },
                )
            )

    math_root = training_data / "banks" / "math"
    for path in sorted(math_root.glob("*/*.jsonl")):
        for line_number, raw in load_jsonl(path):
            location = f"{path}:{line_number}"
            family = required_text(raw, "type", location=location)
            language = required_text(raw, "language", location=location)
            topic = required_text(raw, "topic", location=location)
            problem_id = required_text(raw, "problem_id", location=location)
            prompt = required_text(raw, "prompt", location=location)
            source = str(raw.get("source") or path.stem)
            subset = str(raw.get("subset") or path.stem)
            identity = stable_digest(
                "math", problem_id, family, language, source, subset, prompt
            )
            records.append(
                SourceRecord(
                    domain="math",
                    family=family,
                    stratum=(source, subset, language),
                    group=f"math:{problem_id}",
                    identity=identity,
                    prompt=prompt,
                    metadata={
                        "language": language,
                        "topic": topic,
                        "problem_id": problem_id,
                        "source": source,
                        "subset": subset,
                    },
                )
            )

    structured_root = training_data / "banks" / "structured"
    for path in sorted(structured_root.glob("*/*.jsonl")):
        for line_number, raw in load_jsonl(path):
            location = f"{path}:{line_number}"
            family = required_text(raw, "type", location=location)
            language = required_text(raw, "language", location=location)
            sample_id = raw.get("sample_id")
            if not isinstance(sample_id, (str, int)) or str(sample_id).strip() == "":
                raise ValueError(f"{location} has invalid 'sample_id'")
            prompt = required_text(raw, "prompt", location=location)
            source = str(raw.get("source") or path.stem)
            sample_id_text = str(sample_id)
            schema_id = str(raw.get("schema_id") or "")
            identity = stable_digest(
                "structured", sample_id_text, family, source, schema_id, prompt
            )
            metadata = {
                "language": language,
                "sample_id": sample_id_text,
                "source": source,
            }
            if schema_id:
                metadata["schema_id"] = schema_id
            records.append(
                SourceRecord(
                    domain="structured",
                    family=family,
                    stratum=(source, family, language),
                    group=f"struct:{sample_id_text}",
                    identity=identity,
                    prompt=prompt,
                    metadata=metadata,
                )
            )
    if not records:
        raise ValueError(f"no source prompt banks found under {training_data}")
    return records


def split_for_group(group: str, *, seed: int) -> str:
    bucket = int(stable_digest(seed, group)[:16], 16) % HELDOUT_BUCKETS
    return "heldout" if bucket == 0 else "calibration"


def rendered_token_count(tokenizer: Tokenizer, prompt: str) -> int:
    rendered = render_nonthinking_messages([{"role": "user", "content": prompt}])
    return len(tokenizer.encode(rendered, add_special_tokens=False).ids)


def wiki_prompt(
    source: SourceRecord,
    *,
    tokenizer: Tokenizer,
    article: str,
    target_tokens: int,
    max_tokens: int,
) -> tuple[str, int]:
    article_name = source.metadata["article_path"]
    prefix = (
        "Use the reference article below to complete the task in the task's "
        "language.\n\n"
        f"Reference article ({article_name}):\n"
    )
    suffix = f"\n\nTask:\n{source.prompt}"
    article_ids = tokenizer.encode(article, add_special_tokens=False).ids
    low = 0
    high = len(article_ids)
    best: tuple[int, int, str] | None = None
    while low <= high:
        middle = (low + high) // 2
        excerpt = tokenizer.decode(article_ids[:middle], skip_special_tokens=False)
        prompt = prefix + excerpt + suffix
        count = rendered_token_count(tokenizer, prompt)
        candidate = (abs(count - target_tokens), count, prompt)
        if count <= max_tokens and (best is None or candidate[:2] < best[:2]):
            best = candidate
        if count < target_tokens:
            low = middle + 1
        elif count > target_tokens:
            high = middle - 1
        else:
            return prompt, count
    if best is None:
        prompt = prefix + suffix
        return prompt, rendered_token_count(tokenizer, prompt)
    return best[2], best[1]


def bounded_prompt_prefix(
    prompt: str,
    *,
    tokenizer: Tokenizer,
    target_tokens: int,
    max_tokens: int,
) -> tuple[str, int]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    low = 0
    high = len(prompt_ids)
    best: tuple[int, int, str] | None = None
    while low <= high:
        middle = (low + high) // 2
        candidate_prompt = tokenizer.decode(
            prompt_ids[:middle], skip_special_tokens=False
        )
        count = rendered_token_count(tokenizer, candidate_prompt)
        candidate = (abs(count - target_tokens), count, candidate_prompt)
        if count <= max_tokens and (best is None or candidate[:2] < best[:2]):
            best = candidate
        if count < target_tokens:
            low = middle + 1
        elif count > target_tokens:
            high = middle - 1
        else:
            return candidate_prompt, count
    if best is None:
        return "", rendered_token_count(tokenizer, "")
    return best[2], best[1]


def code_prompt(
    source: SourceRecord,
    *,
    tokenizer: Tokenizer,
    target_tokens: int,
    max_tokens: int,
) -> tuple[str, int]:
    """Keep the agentic task header and a bounded, real source-code window."""
    opening = source.prompt.find("```")
    opening_end = source.prompt.find("\n", opening + 3) if opening >= 0 else -1
    closing = source.prompt.rfind("```")
    if opening < 0 or opening_end < 0 or closing <= opening_end:
        return bounded_prompt_prefix(
            source.prompt,
            tokenizer=tokenizer,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
        )
    prefix = source.prompt[: opening_end + 1]
    body = source.prompt[opening_end + 1 : closing]
    suffix = source.prompt[closing:]
    body_ids = tokenizer.encode(body, add_special_tokens=False).ids
    low = 0
    high = len(body_ids)
    best: tuple[int, int, str] | None = None
    while low <= high:
        middle = (low + high) // 2
        excerpt = tokenizer.decode(body_ids[:middle], skip_special_tokens=False)
        prompt = prefix + excerpt + suffix
        count = rendered_token_count(tokenizer, prompt)
        candidate = (abs(count - target_tokens), count, prompt)
        if count <= max_tokens and (best is None or candidate[:2] < best[:2]):
            best = candidate
        if count < target_tokens:
            low = middle + 1
        elif count > target_tokens:
            high = middle - 1
        else:
            return prompt, count
    if best is not None:
        return best[2], best[1]
    return bounded_prompt_prefix(
        source.prompt,
        tokenizer=tokenizer,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
    )


def prepare_record(
    source: SourceRecord,
    *,
    tokenizer: Tokenizer,
    article_cache: dict[Path, str],
    target_tokens: int,
    min_tokens: int,
    max_tokens: int,
) -> PreparedRecord | None:
    if source.domain == "code":
        prompt, count = code_prompt(
            source,
            tokenizer=tokenizer,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
        )
    elif source.article_path is None:
        prompt = source.prompt
        count = rendered_token_count(tokenizer, prompt)
    else:
        article_path = source.article_path
        try:
            article = article_cache[article_path]
        except KeyError:
            article = article_path.read_text(encoding="utf-8")
            article_cache[article_path] = article
        prompt, count = wiki_prompt(
            source,
            tokenizer=tokenizer,
            article=article,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
        )
    if not min_tokens <= count <= max_tokens:
        return None
    return PreparedRecord(source=source, prompt=prompt, prompt_tokens=count)


def take_round_robin(
    records: list[SourceRecord],
    *,
    split: str,
    domain: str,
    tokenizer: Tokenizer,
    seed: int,
    target_tokens: int,
    min_tokens: int,
    max_tokens: int,
    selected_identities: set[str],
    selected_groups: set[str],
    article_cache: dict[Path, str],
    languages: set[str] | None = None,
    excluded_languages: set[str] | None = None,
    topics: set[str] | None = None,
    excluded_topics: set[str] | None = None,
    families: set[str] | None = None,
    unique_groups: bool = True,
    require_limit: bool = True,
    count: int | None = None,
    token_budget: int | None = None,
) -> list[PreparedRecord]:
    if (count is None) == (token_budget is None):
        raise ValueError("round-robin selection requires exactly one limit")
    strata: dict[tuple[str, ...], deque[SourceRecord]] = {}
    grouped: dict[tuple[str, ...], list[SourceRecord]] = defaultdict(list)
    for record in records:
        if record.domain != domain or split_for_group(record.group, seed=seed) != split:
            continue
        language = record.metadata["language"]
        if languages is not None and language not in languages:
            continue
        if excluded_languages is not None and language in excluded_languages:
            continue
        topic = record.metadata.get("topic")
        if topics is not None and topic not in topics:
            continue
        if excluded_topics is not None and topic in excluded_topics:
            continue
        if families is not None and record.family not in families:
            continue
        grouped[record.stratum].append(record)
    for stratum, members in grouped.items():
        members.sort(key=lambda item: stable_digest(seed, split, item.identity))
        strata[stratum] = deque(members)
    stratum_order = sorted(
        strata, key=lambda item: stable_digest(seed, split, domain, *item)
    )
    selected: list[PreparedRecord] = []
    total_tokens = 0
    while stratum_order and (
        (count is not None and len(selected) < count)
        or (token_budget is not None and total_tokens < token_budget)
    ):
        made_progress = False
        for stratum in list(stratum_order):
            queue = strata[stratum]
            prepared = None
            while queue and prepared is None:
                source = queue.popleft()
                if (
                    source.identity in selected_identities
                    or (unique_groups and source.group in selected_groups)
                ):
                    continue
                prepared = prepare_record(
                    source,
                    tokenizer=tokenizer,
                    article_cache=article_cache,
                    target_tokens=target_tokens,
                    min_tokens=min_tokens,
                    max_tokens=max_tokens,
                )
            if not queue:
                stratum_order.remove(stratum)
            if prepared is None:
                continue
            selected.append(prepared)
            selected_identities.add(prepared.source.identity)
            selected_groups.add(prepared.source.group)
            total_tokens += prepared.prompt_tokens
            made_progress = True
            if (count is not None and len(selected) >= count) or (
                token_budget is not None and total_tokens >= token_budget
            ):
                break
        if not made_progress:
            break
    if require_limit and count is not None and len(selected) < count:
        raise ValueError(
            f"{split} {domain} source banks supplied {len(selected)} eligible "
            f"source-disjoint records; need {count}"
        )
    if require_limit and token_budget is not None and total_tokens < token_budget:
        raise ValueError(
            f"{split} {domain} source banks supplied {total_tokens} eligible tokens; "
            f"need {token_budget}"
        )
    return selected


def select_wiki_records(
    records: list[SourceRecord],
    *,
    split: str,
    token_budget: int,
    tokenizer: Tokenizer,
    seed: int,
    target_tokens: int,
    min_tokens: int,
    max_tokens: int,
    axis: str = "general",
    topics: set[str] | None = None,
    excluded_topics: set[str] | None = None,
    families: set[str] | None = None,
    selected_identities: set[str] | None = None,
    selected_groups: set[str] | None = None,
    unique_groups: bool = True,
) -> list[PreparedRecord]:
    record_count = (token_budget + target_tokens - 1) // target_tokens
    english_count = max(1, round(record_count * 0.40))
    chinese_count = max(1, round(record_count * 0.35))
    other_count = record_count - english_count - chinese_count
    if other_count < 1:
        other_count = 1
        if english_count >= chinese_count:
            english_count -= 1
        else:
            chinese_count -= 1
    quotas = (({"en"}, english_count), ({"zh"}, chinese_count))
    selected: list[PreparedRecord] = []
    if selected_identities is None:
        selected_identities = set()
    if selected_groups is None:
        selected_groups = set()
    article_cache: dict[Path, str] = {}
    for languages, count in quotas:
        selected.extend(
            take_round_robin(
                records,
                split=split,
                domain="wiki",
                tokenizer=tokenizer,
                seed=seed,
                target_tokens=target_tokens,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                selected_identities=selected_identities,
                selected_groups=selected_groups,
                article_cache=article_cache,
                languages=languages,
                topics=topics,
                excluded_topics=excluded_topics,
                families=families,
                unique_groups=unique_groups,
                count=count,
            )
        )
    selected.extend(
        take_round_robin(
            records,
            split=split,
            domain="wiki",
            tokenizer=tokenizer,
            seed=seed,
            target_tokens=target_tokens,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            selected_identities=selected_identities,
            selected_groups=selected_groups,
            article_cache=article_cache,
            excluded_languages=set(WIKI_FOCUS_LANGUAGES),
            topics=topics,
            excluded_topics=excluded_topics,
            families=families,
            unique_groups=unique_groups,
            count=other_count,
        )
    )
    return [replace(item, axis=axis) for item in selected]


def select_code_records(
    records: list[SourceRecord],
    *,
    split: str,
    token_budget: int,
    tokenizer: Tokenizer,
    seed: int,
    target_tokens: int,
    min_tokens: int,
    max_tokens: int,
    axis: str = "code_agentic",
) -> list[PreparedRecord]:
    selected: list[PreparedRecord] = []
    selected_identities: set[str] = set()
    selected_groups: set[str] = set()
    article_cache: dict[Path, str] = {}
    priority_count = min(
        len(CODE_PRIORITY_LANGUAGES),
        (token_budget + target_tokens - 1) // target_tokens,
    )
    for language in CODE_PRIORITY_LANGUAGES[:priority_count]:
        selected.extend(
            take_round_robin(
                records,
                split=split,
                domain="code",
                tokenizer=tokenizer,
                seed=seed,
                target_tokens=target_tokens,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                selected_identities=selected_identities,
                selected_groups=selected_groups,
                article_cache=article_cache,
                languages={language},
                unique_groups=False,
                require_limit=False,
                count=1,
            )
        )
    remaining_tokens = token_budget - sum(item.prompt_tokens for item in selected)
    if remaining_tokens > 0:
        selected.extend(
            take_round_robin(
                records,
                split=split,
                domain="code",
                tokenizer=tokenizer,
                seed=seed,
                target_tokens=target_tokens,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                selected_identities=selected_identities,
                selected_groups=selected_groups,
                article_cache=article_cache,
                unique_groups=False,
                token_budget=remaining_tokens,
            )
        )
    return [replace(item, axis=axis) for item in selected]


def select_math_records(
    records: list[SourceRecord],
    *,
    split: str,
    token_budget: int,
    tokenizer: Tokenizer,
    seed: int,
    target_tokens: int,
    min_tokens: int,
    max_tokens: int,
    axis: str = "math_reasoning",
) -> list[PreparedRecord]:
    selected_identities: set[str] = set()
    selected_groups: set[str] = set()
    article_cache: dict[Path, str] = {}
    chinese_budget = round(token_budget * MATH_CHINESE_TOKEN_FRACTION)
    selected = take_round_robin(
        records,
        split=split,
        domain="math",
        tokenizer=tokenizer,
        seed=seed,
        target_tokens=target_tokens,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        selected_identities=selected_identities,
        selected_groups=selected_groups,
        article_cache=article_cache,
        languages={"zh"},
        require_limit=False,
        token_budget=chinese_budget,
    )
    remaining_tokens = token_budget - sum(item.prompt_tokens for item in selected)
    if remaining_tokens > 0:
        selected.extend(
            take_round_robin(
                records,
                split=split,
                domain="math",
                tokenizer=tokenizer,
                seed=seed,
                target_tokens=target_tokens,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                selected_identities=selected_identities,
                selected_groups=selected_groups,
                article_cache=article_cache,
                excluded_languages={"zh"},
                token_budget=remaining_tokens,
            )
        )
    return [replace(item, axis=axis) for item in selected]


def select_structured_records(
    records: list[SourceRecord],
    *,
    split: str,
    token_budget: int,
    tokenizer: Tokenizer,
    seed: int,
    target_tokens: int,
    min_tokens: int,
    max_tokens: int,
    axis: str = "structured_output",
) -> list[PreparedRecord]:
    selected = take_round_robin(
        records,
        split=split,
        domain="structured",
        tokenizer=tokenizer,
        seed=seed,
        target_tokens=target_tokens,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        selected_identities=set(),
        selected_groups=set(),
        article_cache={},
        token_budget=token_budget,
    )
    return [replace(item, axis=axis) for item in selected]


def select_split(
    records: list[SourceRecord],
    *,
    split: str,
    token_budget: int,
    tokenizer: Tokenizer,
    seed: int,
    target_tokens: int,
    min_tokens: int,
    max_tokens: int,
) -> list[PreparedRecord]:
    total_axis_weight = sum(AXIS_WEIGHTS.values())
    axis_budgets: dict[str, int] = {}
    remaining = token_budget
    axes = list(AXIS_WEIGHTS)
    for axis in axes[:-1]:
        budget = round(token_budget * AXIS_WEIGHTS[axis] / total_axis_weight)
        axis_budgets[axis] = budget
        remaining -= budget
    axis_budgets[axes[-1]] = remaining

    selected_identities: set[str] = set()
    selected_groups: set[str] = set()
    selected = (
        select_wiki_records(
            records,
            split=split,
            token_budget=axis_budgets["reasoning_termination"],
            tokenizer=tokenizer,
            seed=seed,
            target_tokens=target_tokens,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            axis="reasoning_termination",
            topics=REASONING_TOPICS,
            families=REASONING_FAMILIES,
            selected_identities=selected_identities,
            selected_groups=selected_groups,
            unique_groups=False,
        )
    )
    selected.extend(
        select_wiki_records(
            records,
            split=split,
            token_budget=axis_budgets["general"],
            tokenizer=tokenizer,
            seed=seed,
            target_tokens=target_tokens,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            axis="general",
            excluded_topics=LEGAL_TOPICS | REASONING_TOPICS,
            selected_identities=selected_identities,
            selected_groups=selected_groups,
        )
    )
    selected.extend(
        select_code_records(
            records,
            split=split,
            token_budget=axis_budgets["code_agentic"],
            tokenizer=tokenizer,
            seed=seed,
            target_tokens=target_tokens,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            axis="code_agentic",
        )
    )
    selected.extend(
        select_math_records(
            records,
            split=split,
            token_budget=axis_budgets["math_reasoning"],
            tokenizer=tokenizer,
            seed=seed,
            target_tokens=target_tokens,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )
    )
    selected.extend(
        select_structured_records(
            records,
            split=split,
            token_budget=axis_budgets["structured_output"],
            tokenizer=tokenizer,
            seed=seed,
            target_tokens=target_tokens,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )
    )
    selected.sort(
        key=lambda item: stable_digest(seed, split, "output", item.source.identity)
    )
    return selected


def select_screening_records(
    calibration: list[PreparedRecord], *, seed: int
) -> list[PreparedRecord]:
    by_axis: dict[str, list[PreparedRecord]] = defaultdict(list)
    for record in calibration:
        by_axis[record.axis].append(record)
    if set(by_axis) != set(AXIS_WEIGHTS):
        raise ValueError(
            "calibration axes differ from the screening policy: "
            f"found={sorted(by_axis)} expected={sorted(AXIS_WEIGHTS)}"
        )
    selected: list[PreparedRecord] = []
    for axis in AXIS_WEIGHTS:
        records = sorted(
            by_axis[axis],
            key=lambda item: stable_digest(
                seed, "screening", axis, item.source.identity
            ),
        )
        target = round(
            sum(item.prompt_tokens for item in records) * SCREENING_FRACTION
        )
        total = 0
        for record in records:
            selected.append(record)
            total += record.prompt_tokens
            if total >= target:
                break
        if total < target:
            raise ValueError(
                f"screening selection for {axis} supplied {total} tokens; need {target}"
            )
    selected.sort(
        key=lambda item: stable_digest(
            seed, "screening", "output", item.source.identity
        )
    )
    return selected


def jsonl_bytes(
    records: list[PreparedRecord],
    *,
    split: str,
    max_output_tokens: int,
    tokenizer: Tokenizer,
) -> tuple[bytes, list[dict[str, Any]]]:
    lines: list[str] = []
    provenance: list[dict[str, Any]] = []
    for index, item in enumerate(records):
        prefix = {"calibration": "cal", "heldout": "hold", "screening": "screen"}[
            split
        ]
        identifier = f"{prefix}-{index:03d}"
        value = {
            "id": identifier,
            "prompt": item.prompt,
            "max_tokens": max_output_tokens,
        }
        lines.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
        rendered = render_nonthinking_messages(
            [{"role": "user", "content": item.prompt}]
        )
        token_ids = tokenizer.encode(rendered, add_special_tokens=False).ids
        token_payload = json.dumps(token_ids, separators=(",", ":")).encode()
        provenance.append(
            {
                "id": identifier,
                "axis": item.axis,
                "domain": item.source.domain,
                "family": item.source.family,
                "group": item.source.group,
                "source_identity": item.source.identity,
                "prompt_tokens": item.prompt_tokens,
                "prompt_sha256": hashlib.sha256(item.prompt.encode()).hexdigest(),
                "token_ids_sha256": hashlib.sha256(token_payload).hexdigest(),
                **item.source.metadata,
            }
        )
    return ("\n".join(lines) + "\n").encode("utf-8"), provenance


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def summarize(provenance: list[dict[str, Any]]) -> dict[str, Any]:
    languages = sorted(
        {record["language"] for record in provenance if "language" in record}
    )
    return {
        "records": len(provenance),
        "prompt_tokens": sum(record["prompt_tokens"] for record in provenance),
        "domains": {
            domain: sum(record["prompt_tokens"] for record in provenance if record["domain"] == domain)
            for domain in sorted({record["domain"] for record in provenance})
        },
        "axes": {
            axis: sum(
                record["prompt_tokens"]
                for record in provenance
                if record["axis"] == axis
            )
            for axis in sorted({record["axis"] for record in provenance})
        },
        "families": sorted({record["family"] for record in provenance}),
        "languages": languages,
        "language_records": {
            language: sum(record.get("language") == language for record in provenance)
            for language in languages
        },
        "language_prompt_tokens": {
            language: sum(
                record["prompt_tokens"]
                for record in provenance
                if record.get("language") == language
            )
            for language in languages
        },
        "source_groups": len({record["group"] for record in provenance}),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    training_data = args.training_data.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"corpus output is not a directory: {output}")
        if any(output.iterdir()):
            raise ValueError(f"corpus output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.tokenizer.expanduser().resolve(strict=True)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    sources = load_source_records(training_data)
    builder_path = Path(__file__).resolve()
    builder_repo = builder_path.parents[2]
    source_snapshot = git_snapshot(
        training_data,
        input_paths=("banks", "processed/wiki"),
    )
    builder_snapshot = git_snapshot(
        builder_repo,
        input_paths=(str(builder_path.relative_to(builder_repo)),),
    )
    selected = {
        "calibration": select_split(
            sources,
            split="calibration",
            token_budget=args.calibration_tokens,
            tokenizer=tokenizer,
            seed=args.seed,
            target_tokens=args.record_target_tokens,
            min_tokens=args.min_record_tokens,
            max_tokens=args.max_record_tokens,
        ),
        "heldout": select_split(
            sources,
            split="heldout",
            token_budget=args.heldout_tokens,
            tokenizer=tokenizer,
            seed=args.seed,
            target_tokens=args.record_target_tokens,
            min_tokens=args.min_record_tokens,
            max_tokens=args.max_record_tokens,
        ),
    }
    selected["screening"] = select_screening_records(
        selected["calibration"], seed=args.seed
    )
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "training_data": str(training_data),
        "training_data_snapshot": source_snapshot,
        "builder": {
            **builder_snapshot,
            "path": str(builder_path),
            "sha256": hashlib.sha256(builder_path.read_bytes()).hexdigest(),
        },
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "seed": args.seed,
        "source_split": {
            "method": "sha256_group_bucket",
            "heldout_buckets": 1,
            "total_buckets": HELDOUT_BUCKETS,
            "wiki_group": "article_path",
            "code_group": "repository",
            "math_group": "problem_id",
            "structured_group": "sample_id",
        },
        "selection": {
            "axis_token_weights": AXIS_WEIGHTS,
            "axis_sources": {
                "general": "wikipedia topics excluding legal and reasoning topic sets",
                "code_agentic": "review/rewrite/ablation prompts with bounded real code windows",
                "math_reasoning": "NuminaMath-CoT plus a 20% Chinese CMATH token slice",
                "reasoning_termination": {
                    "topics": sorted(REASONING_TOPICS),
                    "families": sorted(REASONING_FAMILIES),
                },
                "structured_output": "packed xLAM function-calling schemas, queries, and expected calls",
            },
            "wiki_language_token_fraction": {
                "en": 0.40,
                "zh": 0.35,
                "other": 0.25,
            },
            "code_priority_languages": list(CODE_PRIORITY_LANGUAGES),
            "math_chinese_token_fraction": MATH_CHINESE_TOKEN_FRACTION,
            "record_target_tokens": args.record_target_tokens,
            "min_record_tokens": args.min_record_tokens,
            "max_record_tokens": args.max_record_tokens,
            "max_output_tokens": args.max_output_tokens,
            "screening_fraction": SCREENING_FRACTION,
            "answers_included": False,
        },
        "splits": {},
    }
    split_groups: dict[str, set[str]] = {}
    for split, records in selected.items():
        payload, provenance = jsonl_bytes(
            records,
            split=split,
            max_output_tokens=args.max_output_tokens,
            tokenizer=tokenizer,
        )
        file_name = f"{split}.jsonl"
        write_atomic(output / file_name, payload)
        split_groups[split] = {record["group"] for record in provenance}
        target_prompt_tokens = {
            "calibration": args.calibration_tokens,
            "heldout": args.heldout_tokens,
            "screening": round(args.calibration_tokens * SCREENING_FRACTION),
        }[split]
        manifest["splits"][split] = {
            "file": file_name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "target_prompt_tokens": target_prompt_tokens,
            "summary": summarize(provenance),
            "records": provenance,
        }
        if split == "screening":
            manifest["splits"][split]["derived_from"] = "calibration"
    overlap = split_groups["calibration"] & split_groups["heldout"]
    if overlap:
        raise RuntimeError(f"calibration/held-out source group overlap: {sorted(overlap)}")
    calibration_identities = {
        record.source.identity for record in selected["calibration"]
    }
    screening_identities = {record.source.identity for record in selected["screening"]}
    if not screening_identities <= calibration_identities:
        raise RuntimeError("screening rows are not a subset of calibration")
    manifest["source_group_overlap"] = []
    manifest["screening_calibration_identity_subset"] = True
    manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    write_atomic(output / "manifest.json", manifest_payload)
    return manifest


def main() -> None:
    try:
        manifest = build(parse_args())
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({"schema": manifest["schema"], "splits": {
        split: value["summary"] for split, value in manifest["splits"].items()
    }}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
