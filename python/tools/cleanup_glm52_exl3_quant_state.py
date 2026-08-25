#!/usr/bin/env python3
"""Safely release regenerable GLM-5.2 EXL3 quantization state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage_glm52_exl3_hf_snapshot import (  # noqa: E402
    MODEL_ID,
    _canonical_json,
    _json_object,
    _publication_evidence,
    _quant_evidence,
    _validation_evidence,
)
from verify_glm52_exl3_hub_publication import (  # noqa: E402
    SCHEMA as HUB_SCHEMA,
)


SCHEMA = "glmrt-glm52-exl3-state-cleanup-v1"
PLAN_SCHEMAS = {
    "glmrt-glm52-gptqmodel-plan-v1",
    "glmrt-glm52-gptqmodel-plan-v2",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
REVISION_RE = re.compile(r"[0-9a-f]{40,64}\Z")
TRANSIENT_RUN_DIRECTORIES = (
    ("layer-boundary", "layer-boundary"),
    ("capture-frontier", "layer-capture-frontier"),
    ("capture-batch-journal", "capture-batch-journal"),
    ("post-quant-replay", "post-quant-replay"),
    ("jit-cache", "jit"),
)


class CleanupError(RuntimeError):
    """The requested cleanup is not proven safe."""


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _planned_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CleanupError(f"plan has no {label} path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise CleanupError(f"plan {label} path is not absolute")
    return Path(os.path.abspath(path))


def _bound_json(path: Path, digest_field: str, label: str) -> dict[str, Any]:
    resolved = path.expanduser()
    if resolved.is_symlink():
        raise CleanupError(f"{label} is a symbolic link")
    resolved = resolved.resolve(strict=True)
    value = _json_object(resolved)
    claimed = value.get(digest_field)
    body = {key: item for key, item in value.items() if key != digest_field}
    if (
        SHA256_RE.fullmatch(str(claimed or "")) is None
        or hashlib.sha256(_canonical_json(body)).hexdigest() != claimed
    ):
        raise CleanupError(f"{label} content binding is invalid")
    return value


def _validate_plan(path: Path, expected_sha256: str) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser()
    if resolved.is_symlink():
        raise CleanupError("quantization plan is a symbolic link")
    resolved = resolved.resolve(strict=True)
    plan = _json_object(resolved)
    claimed = plan.get("plan_sha256")
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if (
        plan.get("schema") not in PLAN_SCHEMAS
        or claimed != expected_sha256
        or hashlib.sha256(_canonical_json(body)).hexdigest() != claimed
    ):
        raise CleanupError("quantization plan identity is invalid")
    return resolved, plan


def _validate_hub_release(
    publication_report_path: Path,
    hub_report_path: Path,
    *,
    validation_sha256: str,
    quant_sha256: str,
    plan_sha256: str,
) -> dict[str, Any]:
    raw_publication = _json_object(publication_report_path.expanduser().resolve(strict=True))
    publication = Path(str(raw_publication.get("output", ""))).expanduser().resolve(
        strict=True
    )
    try:
        entries, publication_identity, publication_report = _publication_evidence(
            publication_report_path,
            publication=publication,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise CleanupError("standard publication evidence is invalid") from error
    if (
        publication_report.get("plan_sha256") != plan_sha256
        or publication_report.get("source_validation_sha256") != validation_sha256
        or publication_report.get("source_quant_evidence_sha256") != quant_sha256
    ):
        raise CleanupError("publication evidence belongs to another quantization")

    hub = _bound_json(hub_report_path, "report_sha256", "Hub verification")
    remote_files = hub.get("files")
    expected = {
        entry["path"]: (entry["bytes"], entry["sha256"]) for entry in entries
    }
    actual = (
        {
            entry.get("path"): (entry.get("bytes"), entry.get("sha256"))
            for entry in remote_files
            if isinstance(entry, dict)
        }
        if isinstance(remote_files, list)
        else {}
    )
    hub_publication = hub.get("publication")
    if (
        hub.get("schema") != HUB_SCHEMA
        or hub.get("status") != "accepted"
        or hub.get("model_id") != MODEL_ID
        or hub.get("visibility") != "public"
        or hub.get("gated") is not False
        or REVISION_RE.fullmatch(str(hub.get("resolved_revision", ""))) is None
        or hub.get("publication_sha256")
        != publication_report.get("publication_sha256")
        or not isinstance(hub_publication, dict)
        or hub_publication.get("sha256")
        != publication_identity["sha256"]
        or not isinstance(remote_files, list)
        or len(actual) != len(remote_files)
        or actual != expected
    ):
        raise CleanupError("Hub verification does not prove this publication")
    return {
        "path": os.fspath(hub_report_path.expanduser().resolve(strict=True)),
        "sha256": _hash_file(hub_report_path.expanduser().resolve(strict=True)),
        "resolved_revision": hub["resolved_revision"],
        "publication_sha256": hub["publication_sha256"],
    }


def _assert_safe_target(path: Path, *, protected: tuple[Path, ...]) -> None:
    if not path.is_absolute() or len(path.parts) < 4:
        raise CleanupError(f"cleanup target is too broad: {path}")
    for item in protected:
        if item == path or item.is_relative_to(path):
            raise CleanupError(f"cleanup target contains protected state: {path}")
    if path.is_symlink():
        raise CleanupError(f"cleanup root is a symbolic link: {path}")
    if path.exists() and not path.is_dir():
        raise CleanupError(f"cleanup target is not a directory: {path}")


def _remove_owned_tree(path: Path) -> None:
    """Remove one exact directory without following an entry symlink."""

    if not path.exists():
        return
    if path.is_symlink() or not path.is_dir():
        raise CleanupError(f"cleanup root is not a regular directory: {path}")
    with os.scandir(path) as entries:
        children = list(entries)
    for entry in children:
        child = Path(entry.path)
        if entry.is_symlink():
            child.unlink()
        elif entry.is_dir(follow_symlinks=False):
            _remove_owned_tree(child)
        elif entry.is_file(follow_symlinks=False):
            child.unlink()
        else:
            raise CleanupError(f"cleanup tree contains an unsupported entry: {child}")
    path.rmdir()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as target:
            target.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def cleanup(
    *,
    artifact_validation_path: Path,
    quant_evidence_path: Path,
    publication_report_path: Path | None,
    hub_verification_path: Path | None,
    release_projection_checkpoints: bool,
    execute: bool,
) -> dict[str, Any]:
    raw_validation = _json_object(
        artifact_validation_path.expanduser().resolve(strict=True)
    )
    artifact = Path(str(raw_validation.get("artifact", ""))).expanduser().resolve(
        strict=True
    )
    artifact_manifest = _json_object(artifact / "glmrt-gptqmodel-artifact.json")
    validation_identity, validation = _validation_evidence(
        artifact_validation_path,
        artifact=artifact,
        artifact_manifest_sha256=artifact_manifest["manifest_sha256"],
    )
    quant_identity, quant = _quant_evidence(
        quant_evidence_path,
        plan_sha256=validation["plan_sha256"],
    )
    if (
        validation["projection_checkpoint"]["checkpoint_inventory_sha256"]
        != quant["integrity"]["checkpoint_inventory_sha256"]
    ):
        raise CleanupError(
            "artifact and quant evidence bind different projection inventories"
        )
    plan_path = Path(str(quant.get("plan", {}).get("path", "")))
    resolved_plan_path, plan = _validate_plan(
        plan_path,
        validation["plan_sha256"],
    )
    raw_output = _planned_path(plan.get("output"), "output")
    raw_run_state = _planned_path(plan.get("run_state_dir"), "run-state")
    source_record = plan.get("source")
    raw_source = _planned_path(
        source_record.get("path") if isinstance(source_record, dict) else None,
        "source",
    )
    if raw_output.is_symlink() or raw_run_state.is_symlink() or raw_source.is_symlink():
        raise CleanupError("plan output, run-state, and source roots must not be symlinks")
    output = raw_output.resolve(strict=True)
    run_state = raw_run_state.resolve(strict=True)
    source = raw_source.resolve(strict=True)
    if output != artifact or not run_state.is_dir() or run_state.is_symlink():
        raise CleanupError("plan output or run-state identity differs")
    if (run_state / "export-stage").exists() or (run_state / "export-stage").is_symlink():
        raise CleanupError("completed output still has an export stage")

    hub_identity: dict[str, Any] | None = None
    if release_projection_checkpoints:
        if publication_report_path is None or hub_verification_path is None:
            raise CleanupError(
                "checkpoint release requires publication and Hub verification reports"
            )
        hub_identity = _validate_hub_release(
            publication_report_path,
            hub_verification_path,
            validation_sha256=validation_identity["sha256"],
            quant_sha256=quant_identity["sha256"],
            plan_sha256=validation["plan_sha256"],
        )
    elif publication_report_path is not None or hub_verification_path is not None:
        raise CleanupError(
            "publication evidence is accepted only with --release-projection-checkpoints"
        )

    targets: list[tuple[str, Path]] = [
        (
            "active-layer-source",
            _planned_path(plan.get("active_layer_source_dir"), "active-source"),
        ),
        ("offload", _planned_path(plan.get("offload_dir"), "offload")),
    ]
    targets.extend((role, run_state / name) for role, name in TRANSIENT_RUN_DIRECTORIES)
    if release_projection_checkpoints:
        targets.append(
            (
                "projection-checkpoints",
                _planned_path(
                    plan.get("projection_checkpoint", {}).get("root"),
                    "projection-checkpoint",
                ),
            )
        )
    paths = [path for _role, path in targets]
    if len(set(paths)) != len(paths) or any(
        left != right and (left.is_relative_to(right) or right.is_relative_to(left))
        for index, left in enumerate(paths)
        for right in paths[index + 1 :]
    ):
        raise CleanupError("cleanup targets overlap")
    protected = (artifact, source, run_state, resolved_plan_path)
    for _role, path in targets:
        _assert_safe_target(path, protected=protected)

    target_records = [
        {
            "role": role,
            "path": os.fspath(path),
            "existed": path.exists(),
        }
        for role, path in targets
    ]
    if execute:
        for _role, path in targets:
            _remove_owned_tree(path)
            if path.parent.is_dir() and not path.parent.is_symlink():
                descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
    body = {
        "schema": SCHEMA,
        "status": "executed" if execute else "planned",
        "model_id": MODEL_ID,
        "plan": {
            "path": os.fspath(resolved_plan_path),
            "plan_sha256": validation["plan_sha256"],
        },
        "artifact_validation": validation_identity,
        "quant_evidence": quant_identity,
        "hub_verification": hub_identity,
        "release_projection_checkpoints": release_projection_checkpoints,
        "targets": target_records,
    }
    return {
        **body,
        "report_sha256": hashlib.sha256(_canonical_json(body)).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-validation", type=Path, required=True)
    parser.add_argument("--quant-evidence", type=Path, required=True)
    parser.add_argument("--publication-report", type=Path)
    parser.add_argument("--hub-verification", type=Path)
    parser.add_argument("--release-projection-checkpoints", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = cleanup(
        artifact_validation_path=args.artifact_validation,
        quant_evidence_path=args.quant_evidence,
        publication_report_path=args.publication_report,
        hub_verification_path=args.hub_verification,
        release_projection_checkpoints=args.release_projection_checkpoints,
        execute=args.execute,
    )
    if args.output is not None:
        _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (CleanupError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"cleanup-glm52-exl3-state: {error}", file=sys.stderr)
        raise SystemExit(2) from error
