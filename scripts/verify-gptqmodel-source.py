#!/usr/bin/env python3
"""Verify the exact GPTQModel fork used by GLMRT quantization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


IGNORED_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}
IGNORED_TOP_LEVEL = {".venv", "build", "dist"}
LOCK_FIELDS = {"schema", "repository", "revision", "source_tree_sha256"}
REVISION_RE = re.compile(r"[0-9a-f]{40}")
DIGEST_RE = re.compile(r"[0-9a-f]{64}")


class VerificationError(RuntimeError):
    """The GPTQModel checkout does not match the committed source lock."""


def _run_git(source: Path, *args: str) -> str:
    source = source.resolve()
    command = [
        "git",
        "-c",
        f"safe.directory={os.fspath(source)}",
        "-C",
        os.fspath(source),
        *args,
    ]
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise VerificationError(f"{' '.join(command)} failed: {detail}")
    return result.stdout.strip()


def _normalized_repository(url: str) -> str:
    value = url.strip()
    ssh_match = re.fullmatch(r"git@([^:]+):(.+)", value)
    if ssh_match:
        value = f"https://{ssh_match.group(1)}/{ssh_match.group(2)}"
    if value.endswith(".git"):
        value = value[:-4]
    return value.rstrip("/")


def _ignored(relative: Path) -> bool:
    return any(part in IGNORED_NAMES for part in relative.parts) or (
        bool(relative.parts)
        and (
            relative.parts[0] in IGNORED_TOP_LEVEL
            or relative.parts[0].endswith(".egg-info")
        )
    )


def source_tree_sha256(source: Path) -> str:
    """Hash paths, Git-relevant modes, symlink targets, and file contents."""

    source = source.resolve()
    entries: list[tuple[str, Path]] = []
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if _ignored(relative):
            continue
        if path.is_file() or path.is_symlink():
            entries.append((relative.as_posix(), path))

    digest = hashlib.sha256()
    for relative, path in sorted(entries):
        if path.is_symlink():
            mode = "120000"
            target = os.readlink(path)
            if not (path.parent / target).resolve(strict=False).is_relative_to(source):
                raise VerificationError(
                    f"GPTQModel source symlink escapes the source tree: "
                    f"{relative} -> {target}"
                )
            content = target.encode()
        else:
            mode = "100755" if path.stat().st_mode & 0o111 else "100644"
            content = path.read_bytes()
        digest.update(mode.encode())
        digest.update(b" ")
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def _verify_source_identity(source: Path) -> None:
    required = (
        source / "LICENSE",
        source / "pyproject.toml",
        source / "gptqmodel" / "__init__.py",
        source / "gptqmodel" / "models" / "definitions" / "glm_moe_dsa.py",
        source / "gptqmodel" / "looper" / "exllamav3_processor.py",
        source
        / "gptqmodel"
        / "exllamav3"
        / "modules"
        / "quant"
        / "exl3_lib"
        / "quantize.py",
    )
    missing = [os.fspath(path) for path in required if not path.is_file()]
    if missing:
        raise VerificationError(
            "GPTQModel source is incomplete; missing " + ", ".join(missing)
        )


def _load_lock(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise VerificationError(
            f"GPTQModel lock not found: {path}; initialize the pinned submodule"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read GPTQModel lock {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"GPTQModel lock must contain one object: {path}")
    if set(value) != LOCK_FIELDS:
        raise VerificationError("GPTQModel lock has unexpected or missing fields")
    if value["schema"] != 1:
        raise VerificationError(f"unsupported GPTQModel lock schema: {value['schema']!r}")
    if not isinstance(value["repository"], str) or not value["repository"].startswith(
        "https://"
    ):
        raise VerificationError("GPTQModel repository must be an HTTPS URL")
    if not isinstance(value["revision"], str) or not REVISION_RE.fullmatch(
        value["revision"]
    ):
        raise VerificationError("GPTQModel revision must be a lowercase 40-hex commit")
    if not isinstance(value["source_tree_sha256"], str) or not DIGEST_RE.fullmatch(
        value["source_tree_sha256"]
    ):
        raise VerificationError(
            "GPTQModel source_tree_sha256 must be lowercase 64-hex"
        )
    return value


def verify(source: Path, lock_path: Path) -> dict[str, object]:
    source = source.resolve()
    lock = _load_lock(lock_path.resolve())
    _verify_source_identity(source)
    actual_digest = source_tree_sha256(source)
    if actual_digest != lock["source_tree_sha256"]:
        raise VerificationError(
            "GPTQModel source content does not match the lock: "
            f"expected {lock['source_tree_sha256']}, found {actual_digest}"
        )
    if (source / ".git").exists():
        actual_revision = _run_git(source, "rev-parse", "HEAD")
        if actual_revision != lock["revision"]:
            raise VerificationError(
                "GPTQModel checkout revision does not match the lock: "
                f"expected {lock['revision']}, found {actual_revision}"
            )
        status = _run_git(source, "status", "--porcelain", "--untracked-files=all")
        if status:
            raise VerificationError("GPTQModel checkout has source changes:\n" + status)
        actual_repository = _run_git(source, "remote", "get-url", "origin")
        if _normalized_repository(actual_repository) != _normalized_repository(
            str(lock["repository"])
        ):
            raise VerificationError(
                "GPTQModel origin does not match the lock: "
                f"expected {lock['repository']}, found {actual_repository}"
            )
    return lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--lock", type=Path)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--print-revision", action="store_true")
    output.add_argument("--print-tree-sha256", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    if args.print_tree_sha256:
        _verify_source_identity(source)
        print(source_tree_sha256(source))
        return 0
    if args.lock is None:
        raise VerificationError("--lock is required unless --print-tree-sha256 is used")
    lock = verify(source, args.lock)
    if args.print_revision:
        print(lock["revision"])
    else:
        print(
            "verified GPTQModel "
            f"revision={lock['revision']} "
            f"source_tree_sha256={lock['source_tree_sha256']} source={source}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(f"verify-gptqmodel-source: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
