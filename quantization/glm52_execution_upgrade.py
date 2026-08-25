#!/usr/bin/env python3
"""Independent validator for GLM-5.2 content-bound execution upgrades."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


EXECUTION_UPGRADE_FILENAME = "glmrt-execution-upgrade.json"
EXECUTION_UPGRADE_HISTORY_DIRNAME = "execution-upgrade-history"
EXECUTION_UPGRADE_SCHEMA = "glmrt-glm52-execution-upgrade-v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class ExecutionUpgradeError(RuntimeError):
    """An upgrade record or its history is incomplete or inconsistent."""


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ExecutionUpgradeError(
            "execution upgrade contains a non-canonical value"
        ) from error


def _json_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ExecutionUpgradeError(f"not a regular execution-upgrade file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecutionUpgradeError(f"cannot read execution upgrade: {path}") from error
    if not isinstance(value, dict):
        raise ExecutionUpgradeError("execution upgrade is not a JSON object")
    return value


def _validate_record(
    record: dict[str, Any],
    *,
    parent_plan_sha256: str,
    label: str,
) -> None:
    digest = record.get("upgrade_sha256")
    body = {
        key: value for key, value in record.items() if key != "upgrade_sha256"
    }
    if (
        record.get("schema") != EXECUTION_UPGRADE_SCHEMA
        or record.get("parent_plan_sha256") != parent_plan_sha256
        or not isinstance(digest, str)
        or SHA256_RE.fullmatch(digest) is None
        or hashlib.sha256(canonical_json(body)).hexdigest() != digest
    ):
        raise ExecutionUpgradeError(f"{label} is invalid")


def _previous_link(record: dict[str, Any], *, label: str) -> str | None:
    fields = (
        "previous_upgrade_sha256",
        "previous_failed_upgrade_sha256",
    )
    present = [(field, record[field]) for field in fields if field in record]
    if len(present) > 1:
        raise ExecutionUpgradeError(f"{label} has ambiguous ancestry")
    if not present:
        return None
    _field, value = present[0]
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ExecutionUpgradeError(f"{label} has an invalid ancestry link")
    return value


def read_execution_upgrade(
    root: Path,
    *,
    parent_plan_sha256: str,
) -> dict[str, Any]:
    """Validate and return the active upgrade plus its entire ancestry."""

    if SHA256_RE.fullmatch(parent_plan_sha256) is None:
        raise ExecutionUpgradeError("parent plan digest is invalid")
    active = _json_object(root / EXECUTION_UPGRADE_FILENAME)
    _validate_record(
        active,
        parent_plan_sha256=parent_plan_sha256,
        label="active execution upgrade",
    )

    history_root = root / EXECUTION_UPGRADE_HISTORY_DIRNAME
    history: dict[str, dict[str, Any]] = {}
    if history_root.exists():
        if not history_root.is_dir() or history_root.is_symlink():
            raise ExecutionUpgradeError(
                "execution-upgrade history is not a regular directory"
            )
        for path in history_root.iterdir():
            match = re.fullmatch(r"([0-9a-f]{64})\.json", path.name)
            if match is None:
                raise ExecutionUpgradeError(
                    "execution-upgrade history contains an unsafe entry"
                )
            record = _json_object(path)
            _validate_record(
                record,
                parent_plan_sha256=parent_plan_sha256,
                label="archived execution upgrade",
            )
            digest = match.group(1)
            if record["upgrade_sha256"] != digest:
                raise ExecutionUpgradeError(
                    "archived execution-upgrade filename differs from its digest"
                )
            history[digest] = record

    cursor = _previous_link(active, label="execution upgrade")
    visited: set[str] = set()
    while cursor is not None:
        if cursor in visited or cursor not in history:
            raise ExecutionUpgradeError(
                "execution-upgrade ancestry is incomplete or cyclic"
            )
        visited.add(cursor)
        record = history[cursor]
        cursor = _previous_link(record, label="archived execution upgrade")
    if set(history) != visited:
        raise ExecutionUpgradeError(
            "execution-upgrade history contains an unlinked record"
        )
    return active


def read_execution_upgrade_chain(
    root: Path,
    *,
    parent_plan_sha256: str,
) -> tuple[dict[str, Any], ...]:
    """Return the active record followed by its validated ancestry."""

    active = read_execution_upgrade(
        root,
        parent_plan_sha256=parent_plan_sha256,
    )
    records = [active]
    cursor = _previous_link(active, label="execution upgrade")
    while cursor is not None:
        record = _json_object(
            root / EXECUTION_UPGRADE_HISTORY_DIRNAME / f"{cursor}.json"
        )
        _validate_record(
            record,
            parent_plan_sha256=parent_plan_sha256,
            label="archived execution upgrade",
        )
        if record["upgrade_sha256"] != cursor:
            raise ExecutionUpgradeError(
                "archived execution-upgrade filename differs from its digest"
            )
        records.append(record)
        cursor = _previous_link(record, label="archived execution upgrade")
    return tuple(records)


__all__ = [
    "EXECUTION_UPGRADE_FILENAME",
    "EXECUTION_UPGRADE_HISTORY_DIRNAME",
    "EXECUTION_UPGRADE_SCHEMA",
    "ExecutionUpgradeError",
    "canonical_json",
    "read_execution_upgrade",
    "read_execution_upgrade_chain",
]
