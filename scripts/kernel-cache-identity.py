#!/usr/bin/env python3
"""Materialize a fail-closed runtime kernel-cache namespace.

FlashInfer's own cache directory is version/architecture scoped and Ninja
tracks dependency mtimes.  That is not sufficient when a cache volume
survives an image replacement whose same-version sources have older mtimes.
This helper adds a content/environment namespace outside FlashInfer's cache.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any


SCHEMA = 1
IGNORED_DIRECTORY_NAMES = {"__pycache__", ".pytest_cache", ".ruff_cache"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}
COMPILE_ENVIRONMENT_NAMES = (
    "CC",
    "CXX",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_TOOLKIT_PATH",
    "CUDACXX",
    "FLASHINFER_JIT_DEBUG",
    "FLASHINFER_JIT_LINEINFO",
    "NVCC_APPEND_FLAGS",
    "NVCC_PREPEND_FLAGS",
    "TORCH_CUDA_ARCH_LIST",
)
TOOLCHAIN_DISTRIBUTIONS = (
    "flashinfer-python",
    "torch",
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-base",
    "nvidia-cutlass-dsl-libs-core",
    "nvidia-cutlass-dsl-libs-cu12",
    "nvidia-cutlass-dsl-libs-cu13",
    "cuda-python",
    "cuda-bindings",
)


class CacheIdentityError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tree_sha256(root: Path) -> str:
    root = root.resolve()
    if not root.is_dir():
        raise CacheIdentityError(f"kernel source tree does not exist: {root}")
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in root.rglob("*")
        if not any(part in IGNORED_DIRECTORY_NAMES for part in path.parts)
        and path.suffix not in IGNORED_SUFFIXES
        and (path.is_file() or path.is_symlink())
    )
    for path in paths:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        if path.is_symlink():
            digest.update(b"symlink\0")
            digest.update(os.readlink(path).encode("utf-8"))
        else:
            digest.update(b"file\0")
            digest.update(f"{path.stat().st_mode & 0o777:o}".encode("ascii"))
            digest.update(b"\0")
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def distribution_facts(name: str) -> dict[str, str]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"version": "missing", "record_sha256": "missing"}
    record = distribution.read_text("RECORD")
    return {
        "version": distribution.version,
        "record_sha256": "missing" if record is None else sha256_bytes(record.encode()),
    }


def find_flashinfer_root(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    spec = importlib.util.find_spec("flashinfer")
    if spec is None or spec.submodule_search_locations is None:
        raise CacheIdentityError("installed flashinfer package was not found")
    locations = list(spec.submodule_search_locations)
    if len(locations) != 1:
        raise CacheIdentityError(
            f"expected one flashinfer package root, found {locations!r}"
        )
    return Path(locations[0]).resolve()


def command_output(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()


def gpu_facts() -> dict[str, list[str] | str]:
    output = command_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if output == "unavailable":
        return {"driver_versions": ["unavailable"], "compute_capabilities": ["unavailable"]}
    drivers: set[str] = set()
    capabilities: set[str] = set()
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2:
            drivers.add(fields[0])
            capabilities.add(fields[1])
    if not drivers or not capabilities:
        raise CacheIdentityError(f"could not parse nvidia-smi GPU identity: {output!r}")
    return {
        "driver_versions": sorted(drivers),
        "compute_capabilities": sorted(capabilities),
    }


def identity_payload(
    flashinfer_root: Path,
    environment_id: str,
) -> dict[str, Any]:
    if not environment_id or "\n" in environment_id or len(environment_id) > 512:
        raise CacheIdentityError("kernel-cache environment identity is missing or invalid")
    return {
        "schema": SCHEMA,
        "environment_id": environment_id,
        "flashinfer": {
            "root": os.fspath(flashinfer_root),
            "source_tree_sha256": tree_sha256(flashinfer_root),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": list(sys.version_info[:3]),
        },
        "platform": {
            "machine": platform.machine(),
            "system": platform.system(),
        },
        "gpu": gpu_facts(),
        "distributions": {
            name: distribution_facts(name) for name in TOOLCHAIN_DISTRIBUTIONS
        },
        "compile_environment": {
            name: os.environ.get(name, "") for name in COMPILE_ENVIRONMENT_NAMES
        },
    }


def identity_key(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def manifest_for(payload: dict[str, Any], key: str) -> dict[str, Any]:
    return {
        "schema": "glmrt.kernel_cache_identity.v1",
        "identity": key,
        "payload": payload,
    }


def materialize(cache_root: Path, payload: dict[str, Any], key: str) -> Path:
    identity_root = cache_root / key
    identity_root.mkdir(parents=True, exist_ok=True)
    manifest_path = identity_root / "IDENTITY.json"
    expected = manifest_for(payload, key)
    if manifest_path.exists():
        try:
            observed = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CacheIdentityError(
                f"kernel-cache identity manifest is unreadable: {manifest_path}"
            ) from error
        if observed != expected:
            raise CacheIdentityError(
                f"kernel-cache identity manifest mismatch: {manifest_path}"
            )
        return identity_root

    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=identity_root,
            prefix=".IDENTITY.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = output.name
            json.dump(expected, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, manifest_path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return identity_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--environment-id",
        default=os.environ.get("GLMRT_KERNEL_CACHE_ENVIRONMENT_ID", ""),
    )
    parser.add_argument("--flashinfer-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        root = find_flashinfer_root(args.flashinfer_root)
        payload = identity_payload(root, args.environment_id)
        key = identity_key(payload)
        materialize(args.cache_root.resolve(), payload, key)
    except CacheIdentityError as error:
        print(f"kernel cache identity: {error}", file=sys.stderr)
        return 2
    print(key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
