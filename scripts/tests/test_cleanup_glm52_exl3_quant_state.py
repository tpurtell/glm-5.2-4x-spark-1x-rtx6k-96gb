from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[2]
TOOLS = ROOT / "python" / "tools"
sys.path.insert(0, str(TOOLS))
TOOL_PATH = TOOLS / "cleanup_glm52_exl3_quant_state.py"
SPEC = importlib.util.spec_from_file_location("_glm52_exl3_cleanup", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


def _bound(value: dict, field: str) -> dict:
    return value | {field: hashlib.sha256(TOOL._canonical_json(value)).hexdigest()}


def test_owned_tree_cleanup_unlinks_but_never_follows_symlinks(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    keep = outside / "keep"
    keep.write_bytes(b"user")
    target = tmp_path / "owned" / "nested"
    target.mkdir(parents=True)
    (target / "payload").write_bytes(b"scratch")
    (target / "outside-link").symlink_to(outside, target_is_directory=True)

    TOOL._remove_owned_tree(tmp_path / "owned")

    assert not (tmp_path / "owned").exists()
    assert keep.read_bytes() == b"user"


def test_cleanup_target_guard_rejects_broad_and_protected_roots(tmp_path: Path) -> None:
    artifact = tmp_path / "state" / "artifact"
    artifact.mkdir(parents=True)
    with pytest.raises(TOOL.CleanupError, match="too broad"):
        TOOL._assert_safe_target(Path("/tmp"), protected=(artifact,))
    with pytest.raises(TOOL.CleanupError, match="contains protected"):
        TOOL._assert_safe_target(tmp_path / "state", protected=(artifact,))


def test_hub_release_must_bind_the_exact_publication(tmp_path: Path) -> None:
    publication = tmp_path / "publication"
    publication.mkdir()
    model = publication / "model.safetensors"
    model.write_bytes(b"packed")
    file_entry = {
        "path": model.name,
        "bytes": model.stat().st_size,
        "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
    }
    validation_sha256 = "a" * 64
    quant_sha256 = "b" * 64
    plan_sha256 = "c" * 64
    publication_body = {
        "schema": "glmrt-hf-standard-publication-v3",
        "model_id": TOOL.MODEL_ID,
        "source_artifact_manifest_sha256": "d" * 64,
        "source_validation_sha256": validation_sha256,
        "source_quant_evidence_sha256": quant_sha256,
        "source_serving_qualification_sha256": "e" * 64,
        "plan_sha256": plan_sha256,
        "files": [file_entry],
    }
    publication_report = tmp_path / "publication.json"
    publication_record = {
        **publication_body,
        "publication_sha256": hashlib.sha256(
            TOOL._canonical_json(publication_body)
        ).hexdigest(),
        "status": "ready",
        "output": str(publication),
    }
    publication_report.write_text(json.dumps(publication_record), encoding="utf-8")
    publication_file_sha256 = hashlib.sha256(publication_report.read_bytes()).hexdigest()
    hub_body = {
        "schema": TOOL.HUB_SCHEMA,
        "status": "accepted",
        "model_id": TOOL.MODEL_ID,
        "requested_revision": "main",
        "resolved_revision": "f" * 40,
        "visibility": "public",
        "gated": False,
        "publication": {
            "path": "standard-publication.json",
            "bytes": publication_report.stat().st_size,
            "sha256": publication_file_sha256,
            "schema": "glmrt-hf-standard-publication-v3",
        },
        "publication_sha256": publication_record["publication_sha256"],
        "files": [{**file_entry, "method": "lfs-sha256"}],
        "file_bytes": file_entry["bytes"],
        "freshly_downloaded": [],
        "fresh_download_limit": 64 * 1024 * 1024,
    }
    hub_report = tmp_path / "hub.json"
    hub_report.write_text(
        json.dumps(_bound(hub_body, "report_sha256")), encoding="utf-8"
    )

    identity = TOOL._validate_hub_release(
        publication_report,
        hub_report,
        validation_sha256=validation_sha256,
        quant_sha256=quant_sha256,
        plan_sha256=plan_sha256,
    )
    assert identity["resolved_revision"] == "f" * 40

    changed = json.loads(hub_report.read_text(encoding="utf-8"))
    changed["publication_sha256"] = "0" * 64
    changed.pop("report_sha256")
    hub_report.write_text(
        json.dumps(_bound(changed, "report_sha256")), encoding="utf-8"
    )
    with pytest.raises(TOOL.CleanupError, match="does not prove"):
        TOOL._validate_hub_release(
            publication_report,
            hub_report,
            validation_sha256=validation_sha256,
            quant_sha256=quant_sha256,
            plan_sha256=plan_sha256,
        )
