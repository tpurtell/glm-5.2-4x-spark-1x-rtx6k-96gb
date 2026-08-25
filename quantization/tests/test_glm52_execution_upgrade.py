import hashlib
import json

import pytest

from glm52_execution_upgrade import (
    EXECUTION_UPGRADE_FILENAME,
    EXECUTION_UPGRADE_HISTORY_DIRNAME,
    EXECUTION_UPGRADE_SCHEMA,
    ExecutionUpgradeError,
    canonical_json,
    read_execution_upgrade,
    read_execution_upgrade_chain,
)


def _record(parent: str, **extra):
    body = {
        "schema": EXECUTION_UPGRADE_SCHEMA,
        "parent_plan_sha256": parent,
        "parent_execution": {"image_digest": "old"},
        "upgraded_execution": {"image_digest": "new"},
        "change_contract": {"purpose": "test"},
        **extra,
    }
    return {
        **body,
        "upgrade_sha256": hashlib.sha256(canonical_json(body)).hexdigest(),
    }


def test_execution_upgrade_chain_round_trip(tmp_path):
    parent = "a" * 64
    first = _record(parent)
    second = _record(parent, previous_upgrade_sha256=first["upgrade_sha256"])
    history = tmp_path / EXECUTION_UPGRADE_HISTORY_DIRNAME
    history.mkdir()
    (history / f"{first['upgrade_sha256']}.json").write_text(
        json.dumps(first), encoding="utf-8"
    )
    (tmp_path / EXECUTION_UPGRADE_FILENAME).write_text(
        json.dumps(second), encoding="utf-8"
    )

    assert read_execution_upgrade(
        tmp_path, parent_plan_sha256=parent
    ) == second
    assert read_execution_upgrade_chain(
        tmp_path, parent_plan_sha256=parent
    ) == (second, first)


def test_execution_upgrade_rejects_unlinked_history(tmp_path):
    parent = "a" * 64
    active = _record(parent)
    orphan = _record(parent, marker="orphan")
    history = tmp_path / EXECUTION_UPGRADE_HISTORY_DIRNAME
    history.mkdir()
    (history / f"{orphan['upgrade_sha256']}.json").write_text(
        json.dumps(orphan), encoding="utf-8"
    )
    (tmp_path / EXECUTION_UPGRADE_FILENAME).write_text(
        json.dumps(active), encoding="utf-8"
    )

    with pytest.raises(ExecutionUpgradeError, match="unlinked"):
        read_execution_upgrade(tmp_path, parent_plan_sha256=parent)


def test_execution_upgrade_rejects_null_link_before_real_link(tmp_path):
    parent = "a" * 64
    first = _record(parent)
    active = _record(
        parent,
        previous_upgrade_sha256=None,
        previous_failed_upgrade_sha256=first["upgrade_sha256"],
    )
    history = tmp_path / EXECUTION_UPGRADE_HISTORY_DIRNAME
    history.mkdir()
    (history / f"{first['upgrade_sha256']}.json").write_text(
        json.dumps(first), encoding="utf-8"
    )
    (tmp_path / EXECUTION_UPGRADE_FILENAME).write_text(
        json.dumps(active), encoding="utf-8"
    )

    with pytest.raises(ExecutionUpgradeError, match="ambiguous ancestry"):
        read_execution_upgrade_chain(tmp_path, parent_plan_sha256=parent)
