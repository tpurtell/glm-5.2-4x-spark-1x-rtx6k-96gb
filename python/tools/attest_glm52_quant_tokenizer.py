#!/usr/bin/env python3
"""Attest tokenizer inputs omitted by an already-running GLM-5.2 K3 plan.

This is a narrow recovery tool for the production v1 run whose immutable plan
predates tokenizer source binding.  It must execute from the exact pinned
quantization image while that original container is still running.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "glmrt-glm52-legacy-tokenizer-attestation-v1"
PLAN_SCHEMA = "glmrt-glm52-gptqmodel-plan-v1"
TOKENIZATION_CONTRACT = (
    "gptqmodel-raw-text-add-special-tokens-return-pt-v1"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
HF_BLOB_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")


class AttestationError(RuntimeError):
    """The legacy run's tokenizer identity cannot be proven."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def json_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AttestationError(f"{label} must be one regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AttestationError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AttestationError(f"{label} is not a JSON object: {path}")
    return value


def bound_record(value: dict[str, Any], digest_field: str) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != digest_field}
    digest = value.get(digest_field)
    if (
        not isinstance(digest, str)
        or SHA256_RE.fullmatch(digest) is None
        or hashlib.sha256(canonical_json(body)).hexdigest() != digest
    ):
        raise AttestationError(f"{digest_field} is invalid")
    return value


def tokenizer_file_identity(snapshot: Path, name: str) -> dict[str, Any]:
    """Hash one canonical HF tokenizer file and retain forensic timestamps."""

    if Path(name).name != name:
        raise AttestationError(f"unsafe tokenizer file name: {name}")
    entry = snapshot / name
    if not entry.is_symlink():
        raise AttestationError(
            f"legacy tokenizer input is not a canonical HF symlink: {entry}"
        )
    link = Path(os.readlink(entry))
    if (
        link.is_absolute()
        or len(link.parts) != 4
        or link.parts[:3] != ("..", "..", "blobs")
        or HF_BLOB_RE.fullmatch(link.parts[3]) is None
    ):
        raise AttestationError(f"invalid HF tokenizer blob link: {entry}")
    blob_root = (snapshot.parent.parent / "blobs").resolve(strict=True)
    try:
        blob = entry.resolve(strict=True)
    except OSError as exc:
        raise AttestationError(f"broken tokenizer blob link: {entry}") from exc
    expected = blob_root / link.parts[3]
    if blob != expected or blob.is_symlink() or not blob.is_file():
        raise AttestationError(f"tokenizer blob escapes its HF store: {entry}")
    digest = sha256_file(blob)
    if len(link.parts[3]) == 64 and digest != link.parts[3]:
        raise AttestationError(f"tokenizer SHA-256 blob content changed: {entry}")
    entry_stat = entry.lstat()
    blob_stat = blob.stat()
    return {
        "name": name,
        "bytes": blob_stat.st_size,
        "sha256": digest,
        "hf_blob_id": link.parts[3],
        "snapshot_entry_ctime_ns": entry_stat.st_ctime_ns,
        "blob_ctime_ns": blob_stat.st_ctime_ns,
    }


def tokenizer_identity_core(record: dict[str, Any]) -> dict[str, Any]:
    """Return fields shared with tokenizer identities in new immutable plans."""

    return {
        key: record[key]
        for key in ("name", "bytes", "sha256", "hf_blob_id")
    }


def calibration_stream(
    path: Path,
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    path = path.expanduser().resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise AttestationError("calibration JSONL must be one regular file")
    rows: list[tuple[str, str]] = []
    text_field: str | None = None
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise AttestationError(
                        f"calibration line {line_number} is not an object"
                    )
                fields = [name for name in ("text", "prompt") if name in value]
                if len(fields) != 1:
                    raise AttestationError(
                        f"calibration line {line_number} has an invalid text schema"
                    )
                if text_field is None:
                    text_field = fields[0]
                elif text_field != fields[0]:
                    raise AttestationError("calibration text schema changes by row")
                text = value[fields[0]]
                identifier = value.get("id", f"line-{line_number:08d}")
                if not isinstance(text, str) or not text.strip():
                    raise AttestationError(
                        f"calibration line {line_number} has invalid text"
                    )
                if not isinstance(identifier, str) or not identifier:
                    raise AttestationError(
                        f"calibration line {line_number} has invalid id"
                    )
                rows.append((identifier, text))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AttestationError(f"cannot read calibration JSONL: {exc}") from exc
    identifiers = [identifier for identifier, _ in rows]
    if not rows or len(identifiers) != len(set(identifiers)):
        raise AttestationError("calibration JSONL is empty or has duplicate ids")
    normalized = canonical_json(
        [{"id": identifier, "text": text} for identifier, text in rows]
    )
    return rows, {
        "path": os.fspath(path),
        "file_sha256": sha256_file(path),
        "normalized_stream_sha256": hashlib.sha256(normalized).hexdigest(),
        "text_field": text_field,
        "examples": len(rows),
        "utf8_bytes": sum(len(text.encode()) for _, text in rows),
    }


def _one_token_row(value: Any, *, label: str) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        not isinstance(value, list)
        or len(value) != 1
        or not isinstance(value[0], list)
        or any(isinstance(token, bool) or not isinstance(token, int) for token in value[0])
    ):
        raise AttestationError(f"tokenizer returned invalid {label} geometry")
    return value[0]


def token_stream_identity(
    tokenizer: Any,
    rows: Iterable[tuple[str, str]],
) -> dict[str, Any]:
    digest = hashlib.sha256()
    lengths: list[int] = []
    record_count = 0
    for identifier, text in rows:
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = _one_token_row(encoded["input_ids"], label="input_ids")
        raw_mask = encoded.get("attention_mask")
        attention_mask = (
            [1] * len(input_ids)
            if raw_mask is None
            else _one_token_row(raw_mask, label="attention_mask")
        )
        if len(attention_mask) != len(input_ids):
            raise AttestationError("tokenizer attention mask length differs")
        digest.update(
            canonical_json(
                {
                    "id": identifier,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
            )
            + b"\n"
        )
        lengths.append(len(input_ids))
        record_count += 1
    if not lengths:
        raise AttestationError("token stream is empty")
    return {
        "contract": TOKENIZATION_CONTRACT,
        "add_special_tokens": True,
        "return_tensors": "pt",
        "records": record_count,
        "total_tokens": sum(lengths),
        "minimum_tokens": min(lengths),
        "maximum_tokens": max(lengths),
        "prepared_token_stream_sha256": digest.hexdigest(),
    }


def _parse_started_at(value: Any) -> tuple[datetime, int]:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise AttestationError("container StartedAt is invalid")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise AttestationError("container StartedAt is invalid") from exc
    return parsed, int(parsed.timestamp() * 1_000_000_000)


def _mount_identity(container: dict[str, Any], path: Path) -> dict[str, Any]:
    matches: list[tuple[int, dict[str, Any], Path]] = []
    for mount in container.get("Mounts", []):
        if not isinstance(mount, dict) or mount.get("Type") != "bind":
            continue
        try:
            destination = Path(str(mount["Destination"]))
            relative = path.relative_to(destination)
        except (KeyError, ValueError):
            continue
        matches.append((len(destination.parts), mount, relative))
    if not matches:
        raise AttestationError(f"container has no bind mount for {path}")
    _, mount, relative = max(matches, key=lambda item: item[0])
    host_path = (Path(str(mount.get("Source"))) / relative).resolve(strict=True)
    if host_path != path:
        raise AttestationError(f"container bind mapping differs for {path}")
    return {
        "source": str(mount["Source"]),
        "destination": str(mount["Destination"]),
        "read_write": mount.get("RW") is True,
    }


def inspect_identity(
    path: Path,
    *,
    plan: dict[str, Any],
    source: Path,
    corpus: Path,
    tokenizer_files: list[dict[str, Any]],
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AttestationError("docker inspect input must be one regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AttestationError(f"cannot read docker inspect input: {exc}") from exc
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise AttestationError("docker inspect input must contain exactly one container")
    container = value[0]
    state = container.get("State")
    preflight = plan.get("preflight")
    if (
        not isinstance(state, dict)
        or not isinstance(preflight, dict)
        or state.get("Running") is not True
        or state.get("Paused") is not False
        or state.get("Restarting") is not False
        or state.get("OOMKilled") is not False
        or state.get("Dead") is not False
        or state.get("ExitCode") != 0
        or state.get("Error") not in (None, "")
        or container.get("RestartCount") != 0
        or container.get("Image") != preflight.get("image_digest")
    ):
        raise AttestationError("docker inspect does not describe the live pinned run")
    started, started_ns = _parse_started_at(state.get("StartedAt"))
    for record in tokenizer_files:
        if (
            record["snapshot_entry_ctime_ns"] > started_ns
            or record["blob_ctime_ns"] > started_ns
        ):
            raise AttestationError(
                f"tokenizer input postdates original container start: {record['name']}"
            )
    identifier = str(container.get("Id", ""))
    if SHA256_RE.fullmatch(identifier) is None:
        raise AttestationError("docker inspect container id is invalid")
    return {
        "id": identifier,
        "name": str(container.get("Name", "")).removeprefix("/"),
        "image_digest": container["Image"],
        "started_at": started.astimezone(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "restart_count": 0,
        "source_mount": _mount_identity(container, source),
        "corpus_mount": _mount_identity(container, corpus),
        "tokenizer_inputs_predate_start": True,
    }


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("tokenicer", "tokenizers", "torch", "transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise AttestationError(f"required package is absent: {name}") from exc
    return versions


def build_attestation(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = args.plan.expanduser().resolve(strict=True)
    plan = bound_record(json_object(plan_path, label="plan"), "plan_sha256")
    source = args.snapshot.expanduser().resolve(strict=True)
    if not source.is_dir() or source.is_symlink():
        raise AttestationError("source snapshot must be one regular directory")
    rows, corpus = calibration_stream(args.calibration_jsonl)
    plan_source = plan.get("source")
    if (
        plan.get("schema") != PLAN_SCHEMA
        or not isinstance(plan_source, dict)
        or "tokenizer_files" in plan_source
        or Path(str(plan_source.get("path", ""))).resolve() != source
        or plan_source.get("revision") != source.name
        or plan_source.get("config_sha256") != sha256_file(source / "config.json")
        or plan_source.get("index_sha256")
        != sha256_file(source / "model.safetensors.index.json")
        or plan.get("corpus") != corpus
    ):
        raise AttestationError("legacy plan does not bind the supplied source/corpus")
    tokenizer_files = [
        tokenizer_file_identity(source, name) for name in TOKENIZER_FILES
    ]
    container = inspect_identity(
        args.container_inspect.expanduser().resolve(strict=True),
        plan=plan,
        source=source,
        corpus=Path(corpus["path"]),
        tokenizer_files=tokenizer_files,
    )

    from tokenicer import Tokenicer
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(
        source,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer = Tokenicer.load(
        os.fspath(source),
        model_config=model_config,
        local_files_only=True,
        trust_remote_code=False,
    ).tokenizer
    tokenization = token_stream_identity(tokenizer, rows)
    if tokenization["records"] != corpus["examples"]:
        raise AttestationError("tokenized record count differs from the corpus")

    body = {
        "schema": SCHEMA,
        "status": "accepted",
        "scope": "legacy-plan-omitted-tokenizer-source-identity",
        "plan": {
            "path": os.fspath(plan_path),
            "plan_sha256": plan["plan_sha256"],
        },
        "source": {
            "path": os.fspath(source),
            "revision": source.name,
            "tokenizer_files": tokenizer_files,
        },
        "corpus": corpus,
        "container": container,
        "environment": {
            "python": platform.python_version(),
            "packages": _package_versions(),
            "tokenizer_class": (
                f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
            ),
        },
        "tokenization": tokenization,
    }
    return body | {
        "attestation_sha256": hashlib.sha256(canonical_json(body)).hexdigest()
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as target:
            target.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--calibration-jsonl", type=Path, required=True)
    parser.add_argument("--container-inspect", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_attestation(args)
    atomic_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except AttestationError as exc:
        raise SystemExit(f"glm52-tokenizer-attestation: {exc}") from exc
