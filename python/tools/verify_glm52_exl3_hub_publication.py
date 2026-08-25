#!/usr/bin/env python3
"""Verify the exact public Hub revision of the calibrated GLM-5.2 EXL3 model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable

from stage_glm52_exl3_hf_snapshot import (
    MODEL_ID,
    _canonical_json,
    _json_object,
    _publication_evidence,
)


SCHEMA = "glmrt-glm52-exl3-hub-verification-v1"
REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_FRESH_DOWNLOAD_LIMIT = 64 * 1024 * 1024


class HubVerificationError(RuntimeError):
    """The remote revision does not exactly match the accepted publication."""


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def verify(
    *,
    publication_report_path: Path,
    revision: str,
    api: Any,
    downloader: Callable[..., str],
    token: bool | str | None = None,
    fresh_download_limit: int = DEFAULT_FRESH_DOWNLOAD_LIMIT,
) -> dict[str, Any]:
    if isinstance(fresh_download_limit, bool) or fresh_download_limit < 0:
        raise HubVerificationError("fresh-download limit must be nonnegative")
    report_path = publication_report_path.expanduser()
    if report_path.is_symlink():
        raise HubVerificationError("publication report is a symbolic link")
    report_path = report_path.resolve(strict=True)
    publication_report = _json_object(report_path)
    publication = Path(str(publication_report.get("output", ""))).expanduser().resolve(
        strict=True
    )
    try:
        entries, publication_identity, publication_report = _publication_evidence(
            report_path,
            publication=publication,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise HubVerificationError("local publication evidence is invalid") from error
    expected = {entry["path"]: entry for entry in entries}

    info = api.model_info(
        MODEL_ID,
        revision=revision,
        files_metadata=True,
        token=token,
    )
    resolved_revision = field(info, "sha")
    siblings = field(info, "siblings")
    if (
        field(info, "id") != MODEL_ID
        or field(info, "private") is not False
        or field(info, "gated") not in {None, False}
        or not isinstance(resolved_revision, str)
        or REVISION_RE.fullmatch(resolved_revision) is None
        or not isinstance(siblings, list)
    ):
        raise HubVerificationError(
            "Hub model identity, visibility, or resolved revision is invalid"
        )

    remote: dict[str, Any] = {}
    for sibling in siblings:
        path = field(sibling, "path", field(sibling, "rfilename"))
        size = field(sibling, "size")
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or ".." in Path(path).parts
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or path in remote
        ):
            raise HubVerificationError("Hub file metadata is malformed")
        remote[path] = sibling
    if set(remote) != set(expected):
        raise HubVerificationError(
            "Hub file inventory differs: "
            f"missing={sorted(set(expected) - set(remote))} "
            f"unexpected={sorted(set(remote) - set(expected))}"
        )

    verified: list[dict[str, Any]] = []
    freshly_downloaded: list[str] = []
    with tempfile.TemporaryDirectory(prefix="glmrt-exl3-hub-verify-") as cache_root:
        for path, expected_entry in sorted(expected.items()):
            sibling = remote[path]
            size = field(sibling, "size")
            if size != expected_entry["bytes"]:
                raise HubVerificationError(f"Hub file size differs: {path}")
            lfs = field(sibling, "lfs")
            lfs_sha256 = field(lfs, "sha256") if lfs is not None else None
            if lfs_sha256 is not None and (
                not isinstance(lfs_sha256, str)
                or SHA256_RE.fullmatch(lfs_sha256) is None
                or lfs_sha256 != expected_entry["sha256"]
                or field(lfs, "size") != size
            ):
                raise HubVerificationError(f"Hub LFS identity differs: {path}")
            method = "lfs-sha256"
            if size <= fresh_download_limit:
                downloaded = Path(
                    downloader(
                        repo_id=MODEL_ID,
                        filename=path,
                        revision=resolved_revision,
                        repo_type="model",
                        cache_dir=cache_root,
                        force_download=True,
                        token=token,
                    )
                ).resolve(strict=True)
                if (
                    not downloaded.is_file()
                    or downloaded.stat().st_size != size
                    or hash_file(downloaded) != expected_entry["sha256"]
                ):
                    raise HubVerificationError(f"fresh Hub download differs: {path}")
                freshly_downloaded.append(path)
                method = "fresh-download-sha256"
            elif lfs_sha256 is None:
                raise HubVerificationError(
                    f"large Hub file has no remotely verifiable SHA-256: {path}"
                )
            verified.append(
                {
                    "path": path,
                    "bytes": size,
                    "sha256": expected_entry["sha256"],
                    "method": method,
                }
            )

    body = {
        "schema": SCHEMA,
        "status": "accepted",
        "model_id": MODEL_ID,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "visibility": "public",
        "gated": False,
        "publication": publication_identity,
        "publication_sha256": publication_report["publication_sha256"],
        "files": verified,
        "file_bytes": sum(entry["bytes"] for entry in verified),
        "freshly_downloaded": freshly_downloaded,
        "fresh_download_limit": fresh_download_limit,
    }
    if not math.isfinite(float(body["file_bytes"])):
        raise HubVerificationError("remote byte total is invalid")
    return {
        **body,
        "report_sha256": hashlib.sha256(_canonical_json(body)).hexdigest(),
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as target:
            target.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication-report", type=Path, required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--fresh-download-limit", type=int, default=DEFAULT_FRESH_DOWNLOAD_LIMIT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    report = verify(
        publication_report_path=args.publication_report,
        revision=args.revision,
        api=HfApi(),
        downloader=hf_hub_download,
        fresh_download_limit=args.fresh_download_limit,
    )
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
