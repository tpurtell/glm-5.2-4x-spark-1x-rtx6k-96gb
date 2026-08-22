#!/usr/bin/env python3
"""Materialize and verify SparkInfer release provenance and legal material."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tomllib
from types import ModuleType


sys.dont_write_bytecode = True

PROVENANCE_SCHEMA = 1
SPARKINFER_LICENSE_SPDX = "Apache-2.0"
PROVENANCE_FIELDS = {
    "schema",
    "component",
    "repository",
    "revision",
    "source_tree_sha256",
    "license",
    "notices",
}


class ReleaseProvenanceError(RuntimeError):
    """Raised when SparkInfer release material is incomplete or inconsistent."""


def _load_source_verifier() -> ModuleType:
    path = Path(__file__).with_name("verify-sparkinfer-source.py")
    spec = importlib.util.spec_from_file_location("glmrt_sparkinfer_verifier", path)
    if spec is None or spec.loader is None:
        raise ReleaseProvenanceError(f"cannot load SparkInfer verifier: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERIFIER = _load_source_verifier()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_package_license(source: Path, distributed_license: Path) -> None:
    source_license = source / "LICENSE"
    try:
        with (source / "pyproject.toml").open("rb") as stream:
            project = tomllib.load(stream)["project"]
    except (KeyError, OSError, tomllib.TOMLDecodeError) as exc:
        raise ReleaseProvenanceError(
            f"cannot read SparkInfer license metadata: {exc}"
        ) from exc

    if project.get("license") != SPARKINFER_LICENSE_SPDX:
        raise ReleaseProvenanceError(
            "SparkInfer package metadata must declare license=Apache-2.0"
        )
    license_files = project.get("license-files")
    if not isinstance(license_files, list) or "LICENSE" not in license_files:
        raise ReleaseProvenanceError(
            "SparkInfer package metadata must distribute the root LICENSE"
        )
    try:
        source_bytes = source_license.read_bytes()
        distributed_bytes = distributed_license.read_bytes()
    except OSError as exc:
        raise ReleaseProvenanceError(
            f"cannot read SparkInfer license material: {exc}"
        ) from exc
    required_license_text = (
        b"Apache License",
        b"Version 2.0, January 2004",
        b"TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION",
    )
    if any(marker not in source_bytes for marker in required_license_text):
        raise ReleaseProvenanceError(
            f"verified SparkInfer source license {source_license} is not the "
            "Apache License, Version 2.0 text"
        )
    if source_bytes != distributed_bytes:
        raise ReleaseProvenanceError(
            f"distributed SparkInfer license {distributed_license} differs from "
            f"the verified source license {source_license}"
        )


def _validate_notices(source: Path, notices: Path, repository: str) -> None:
    try:
        text = notices.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReleaseProvenanceError(
            f"cannot read third-party notices {notices}: {exc}"
        ) from exc
    repository_url = repository.rstrip("/").removesuffix(".git")
    required = (
        "## SparkInfer",
        repository_url,
        "SPDX-License-Identifier: Apache-2.0",
        "SPARKINFER_LICENSE",
        "SPARKINFER_PROVENANCE.json",
        "SPARKINFER_SHA256SUMS",
    )
    missing = [value for value in required if value not in text]
    if missing:
        raise ReleaseProvenanceError(
            f"third-party notices {notices} omit SparkInfer release material: "
            + ", ".join(missing)
        )
    derived_components = (
        (
            source / "b12x" / "_lib" / "dense_gemm.py",
            (
                "### NVIDIA dense GEMM component in SparkInfer",
                "b12x/_lib/dense_gemm.py",
                "Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.",
            ),
        ),
        (
            source
            / "b12x"
            / "attention"
            / "_shared"
            / "contiguous"
            / "forward.py",
            (
                "### FlashAttention-derived contiguous attention component "
                "in SparkInfer",
                "b12x/attention/_shared/contiguous/forward.py",
                "Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang,",
            ),
        ),
    )
    for component, component_notices in derived_components:
        if not component.is_file():
            continue
        missing = [value for value in component_notices if value not in text]
        if missing:
            raise ReleaseProvenanceError(
                f"third-party notices {notices} omit notices required by "
                f"{component.relative_to(source)}: "
                + ", ".join(missing)
            )


def expected_provenance(
    source: Path,
    lock: Path,
    distributed_license: Path,
    notices: Path,
) -> dict[str, object]:
    source = source.resolve()
    lock_data = VERIFIER.verify(source, lock)
    _validate_package_license(source, distributed_license)
    repository = str(lock_data["repository"])
    _validate_notices(source, notices, repository)
    return {
        "schema": PROVENANCE_SCHEMA,
        "component": "sparkinfer",
        "repository": repository,
        "revision": lock_data["revision"],
        "source_tree_sha256": lock_data["source_tree_sha256"],
        "license": {
            "spdx": SPARKINFER_LICENSE_SPDX,
            "artifact": "SPARKINFER_LICENSE",
            "sha256": file_sha256(distributed_license),
        },
        "notices": {
            "artifact": "THIRD_PARTY_NOTICES.md",
            "sha256": file_sha256(notices),
        },
    }


def write_provenance(path: Path, provenance: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify_provenance(path: Path, expected: dict[str, object]) -> None:
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReleaseProvenanceError(
            f"SparkInfer release provenance not found: {path}"
        ) from exc
    except OSError as exc:
        raise ReleaseProvenanceError(
            f"cannot read SparkInfer release provenance {path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ReleaseProvenanceError(
            f"invalid SparkInfer release provenance {path}: {exc}"
        ) from exc
    if not isinstance(actual, dict) or set(actual) != PROVENANCE_FIELDS:
        fields = sorted(actual) if isinstance(actual, dict) else type(actual).__name__
        raise ReleaseProvenanceError(
            f"SparkInfer release provenance fields are invalid: {fields}"
        )
    if actual != expected:
        raise ReleaseProvenanceError(
            f"SparkInfer release provenance {path} does not match the verified "
            "source, lock, license, and notices"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--notices", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", type=Path, metavar="PATH")
    action.add_argument("--verify", type=Path, metavar="PATH")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        expected = expected_provenance(
            args.source,
            args.lock,
            args.license,
            args.notices,
        )
        if args.write is not None:
            write_provenance(args.write, expected)
            print(
                f"wrote SparkInfer release provenance: {args.write} "
                f"revision={expected['revision']}"
            )
        else:
            verify_provenance(args.verify, expected)
            print(
                f"verified SparkInfer release provenance: {args.verify} "
                f"revision={expected['revision']}"
            )
    except (OSError, ReleaseProvenanceError, VERIFIER.VerificationError) as exc:
        print(
            f"SparkInfer release provenance verification failed: {exc}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
