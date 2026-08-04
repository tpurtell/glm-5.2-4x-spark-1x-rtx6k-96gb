#!/usr/bin/env python3
"""Verify the exact XGrammar source embedded in the native coordinator library."""

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
    "build",
    "dist",
}
IGNORED_SOURCE_TREES = {
    Path("3rdparty/cpptrace"),
    Path("3rdparty/googletest"),
}
LOCK_FIELDS = {
    "schema",
    "repository",
    "revision",
    "dlpack_revision",
    "source_tree_sha256",
}
REVISION_RE = re.compile(r"[0-9a-f]{40}")
DIGEST_RE = re.compile(r"[0-9a-f]{64}")


class VerificationError(RuntimeError):
    pass


def run_git(source: Path, *args: str) -> str:
    source = source.resolve()
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={os.fspath(source)}",
            "-C",
            os.fspath(source),
            *args,
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise VerificationError(f"git {' '.join(args)} failed for {source}: {detail}")
    return result.stdout.strip()


def normalized_repository(url: str) -> str:
    value = url.strip()
    match = re.fullmatch(r"git@([^:]+):(.+)", value)
    if match:
        value = f"https://{match.group(1)}/{match.group(2)}"
    if value.endswith(".git"):
        value = value[:-4]
    return value.rstrip("/")


def ignored(relative: Path) -> bool:
    return (
        any(part in IGNORED_NAMES for part in relative.parts)
        or any(
            relative == source_tree or source_tree in relative.parents
            for source_tree in IGNORED_SOURCE_TREES
        )
        or relative.suffix in {".pyc", ".pyo"}
    )


def source_tree_sha256(source: Path) -> str:
    source = source.resolve()
    entries: list[tuple[str, Path]] = []
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if ignored(relative):
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
                    f"XGrammar source symlink escapes the tree: {relative} -> {target}"
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


def load_lock(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read XGrammar lock {path}: {error}") from error
    if not isinstance(value, dict) or set(value) != LOCK_FIELDS:
        raise VerificationError(
            f"XGrammar lock fields must be exactly {sorted(LOCK_FIELDS)}"
        )
    if value["schema"] != 1:
        raise VerificationError(f"unsupported XGrammar lock schema {value['schema']!r}")
    if not isinstance(value["repository"], str) or not str(value["repository"]).startswith(
        "https://"
    ):
        raise VerificationError("XGrammar repository must be an HTTPS URL")
    for field in ("revision", "dlpack_revision"):
        if not isinstance(value[field], str) or not REVISION_RE.fullmatch(str(value[field])):
            raise VerificationError(f"XGrammar {field} must be a lowercase 40-hex commit")
    if not isinstance(value["source_tree_sha256"], str) or not DIGEST_RE.fullmatch(
        str(value["source_tree_sha256"])
    ):
        raise VerificationError("XGrammar source_tree_sha256 must be lowercase 64-hex")
    return value


def verify(source: Path, lock_path: Path) -> dict[str, object]:
    source = source.resolve()
    required = (
        source / "LICENSE",
        source / "include/xgrammar/compiler.h",
        source / "cpp/grammar_compiler.cc",
        source / "3rdparty/picojson/picojson.h",
        source / "3rdparty/dlpack/include/dlpack/dlpack.h",
    )
    missing = [os.fspath(path) for path in required if not path.is_file()]
    if missing:
        raise VerificationError("XGrammar source is incomplete; missing " + ", ".join(missing))
    lock = load_lock(lock_path.resolve())
    actual_digest = source_tree_sha256(source)
    if actual_digest != lock["source_tree_sha256"]:
        raise VerificationError(
            "XGrammar source content does not match the lock: "
            f"expected {lock['source_tree_sha256']}, found {actual_digest}"
        )
    if (source / ".git").exists():
        if run_git(source, "rev-parse", "HEAD") != lock["revision"]:
            raise VerificationError("XGrammar checkout revision does not match the lock")
        if run_git(source, "status", "--porcelain", "--untracked-files=all"):
            raise VerificationError("XGrammar checkout has tracked or untracked changes")
        repository = run_git(source, "remote", "get-url", "origin")
        if normalized_repository(repository) != normalized_repository(str(lock["repository"])):
            raise VerificationError("XGrammar checkout repository does not match the lock")
        dlpack = source / "3rdparty/dlpack"
        if run_git(dlpack, "rev-parse", "HEAD") != lock["dlpack_revision"]:
            raise VerificationError("XGrammar dlpack revision does not match the lock")
        if run_git(dlpack, "status", "--porcelain", "--untracked-files=all"):
            raise VerificationError("XGrammar dlpack checkout has local changes")
    return lock


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--print-source-digest", action="store_true")
    args = parser.parse_args()
    if args.print_source_digest:
        print(source_tree_sha256(args.source))
        return 0
    lock = verify(args.source, args.lock)
    print(
        f"verified XGrammar revision={lock['revision']} "
        f"source_tree_sha256={lock['source_tree_sha256']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
