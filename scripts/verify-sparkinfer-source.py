#!/usr/bin/env python3
"""Verify GLMRT's pinned SparkInfer source and report its provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib


IGNORED_ANYWHERE = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}
IGNORED_TOP_LEVEL = {
    ".deps",
    ".venv",
    "build",
    "dist",
}
PYTHON_CACHE_DIRECTORIES = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}
PYTHON_BYTECODE_SUFFIXES = {
    ".pyc",
    ".pyo",
}
LOCK_FIELDS = {
    "schema",
    "repository",
    "revision",
    "source_tree_sha256",
}
REVISION_RE = re.compile(r"[0-9a-f]{40}")
DIGEST_RE = re.compile(r"[0-9a-f]{64}")


class VerificationError(RuntimeError):
    """Raised when the checkout does not match the committed source lock."""


def _run_git(source: Path, *args: str) -> str:
    source = source.resolve()
    # Release builds mount the checkout read-only into a root-owned container.
    # Git's dubious-ownership guard is correct by default, but the verifier has
    # already resolved the exact source path supplied by the caller. Trust only
    # that path for this invocation instead of mutating global Git config.
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
    parts = relative.parts
    return (
        any(part in IGNORED_ANYWHERE for part in parts)
        or (
            bool(parts)
            and (
                parts[0] in IGNORED_TOP_LEVEL
                or parts[0].endswith(".egg-info")
                or parts[0].startswith(".sm120port")
            )
        )
    )


def source_tree_sha256(source: Path) -> str:
    """Hash source paths, Git-relevant modes, symlink targets, and contents."""

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
            resolved_target = (path.parent / target).resolve(strict=False)
            if not resolved_target.is_relative_to(source):
                raise VerificationError(
                    "SparkInfer source symlink escapes the source tree: "
                    f"{relative} -> {target}"
                )
            content = target.encode("utf-8")
        else:
            mode = "100755" if path.stat().st_mode & 0o111 else "100644"
            content = path.read_bytes()
        digest.update(mode.encode("ascii"))
        digest.update(b" ")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def _load_lock(path: Path) -> dict[str, object]:
    display = "<stdin>" if os.fspath(path) == "-" else os.fspath(path)
    try:
        text = sys.stdin.read() if os.fspath(path) == "-" else path.read_text(
            encoding="utf-8"
        )
        value = json.loads(text)
    except FileNotFoundError as exc:
        raise VerificationError(
            f"SparkInfer lock not found: {path}; initialize the pinned submodule"
        ) from exc
    except OSError as exc:
        raise VerificationError(f"cannot read SparkInfer lock {display}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise VerificationError(f"invalid SparkInfer lock {display}: {exc}") from exc

    if not isinstance(value, dict):
        raise VerificationError(
            f"SparkInfer lock must contain one JSON object: {display}"
        )
    unexpected = set(value) - LOCK_FIELDS
    missing = LOCK_FIELDS - set(value)
    if unexpected or missing:
        raise VerificationError(
            f"SparkInfer lock fields mismatch: missing={sorted(missing)} "
            f"unexpected={sorted(unexpected)}"
        )
    if value["schema"] != 1:
        raise VerificationError(
            f"unsupported SparkInfer lock schema: {value['schema']!r}"
        )
    repository = value["repository"]
    revision = value["revision"]
    tree_digest = value["source_tree_sha256"]
    if not isinstance(repository, str) or not repository.startswith("https://"):
        raise VerificationError("SparkInfer repository must be an HTTPS URL")
    if not isinstance(revision, str) or not REVISION_RE.fullmatch(revision):
        raise VerificationError("SparkInfer revision must be a lowercase 40-hex commit")
    if not isinstance(tree_digest, str) or not DIGEST_RE.fullmatch(tree_digest):
        raise VerificationError(
            "SparkInfer source_tree_sha256 must be a lowercase 64-hex digest"
        )
    return value


def _verify_package_identity(source: Path) -> None:
    required = (
        source / "LICENSE",
        source / "pyproject.toml",
        source / "sparkinfer" / "__init__.py",
    )
    missing = [os.fspath(path) for path in required if not path.is_file()]
    if missing:
        raise VerificationError(
            "SparkInfer source is incomplete; missing " + ", ".join(missing)
        )
    try:
        with (source / "pyproject.toml").open("rb") as stream:
            project = tomllib.load(stream).get("project", {})
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise VerificationError(
            f"cannot read SparkInfer package metadata {source / 'pyproject.toml'}: "
            f"{exc}"
        ) from exc
    if project.get("name") != "sparkinfer":
        raise VerificationError(
            f"{source / 'pyproject.toml'} does not declare project.name=sparkinfer"
        )


def require_no_python_cache(source: Path) -> None:
    """Reject generated Python caches in a metadata-free release source tree."""

    source = source.resolve()
    for path in source.rglob("*"):
        if (
            path.name in PYTHON_CACHE_DIRECTORIES
            or path.suffix in PYTHON_BYTECODE_SUFFIXES
        ):
            relative = path.relative_to(source)
            raise VerificationError(
                "SparkInfer metadata-free source contains a Python cache: "
                f"{relative}"
            )


def verify(source: Path, lock_path: Path) -> dict[str, object]:
    source = source.resolve()
    if os.fspath(lock_path) != "-":
        lock_path = lock_path.resolve()
    _verify_package_identity(source)
    lock = _load_lock(lock_path)

    actual_digest = source_tree_sha256(source)
    if actual_digest != lock["source_tree_sha256"]:
        raise VerificationError(
            "SparkInfer source content does not match the lock: "
            f"expected {lock['source_tree_sha256']}, found {actual_digest}"
        )

    # A submodule checkout has its own .git directory or gitfile. Release
    # archives deliberately omit it, so the content digest remains the
    # authoritative verification in that case.
    if (source / ".git").exists():
        actual_revision = _run_git(source, "rev-parse", "HEAD")
        if actual_revision != lock["revision"]:
            raise VerificationError(
                "SparkInfer checkout revision does not match the lock: "
                f"expected {lock['revision']}, found {actual_revision}"
            )
        status = _run_git(source, "status", "--porcelain", "--untracked-files=all")
        if status:
            raise VerificationError(
                "SparkInfer checkout has tracked or untracked source changes:\n"
                + status
            )
        actual_repository = _run_git(source, "remote", "get-url", "origin")
        if _normalized_repository(actual_repository) != _normalized_repository(
            str(lock["repository"])
        ):
            raise VerificationError(
                "SparkInfer origin does not match the lock: "
                f"expected {lock['repository']}, found {actual_repository}"
            )

    return lock


def verify_import_source(source: Path) -> Path:
    """Require the runtime ``sparkinfer`` import to come from ``source``."""

    source = source.resolve()
    try:
        import sparkinfer
    except Exception as exc:
        raise VerificationError(f"cannot import sparkinfer: {exc}") from exc

    module_file = getattr(sparkinfer, "__file__", None)
    if not module_file:
        raise VerificationError("imported sparkinfer has no __file__")
    imported_path = Path(module_file).resolve()
    try:
        imported_path.relative_to(source)
    except ValueError as exc:
        raise VerificationError(
            "imported sparkinfer resolves outside the verified source tree: "
            f"module={imported_path}, source={source}"
        ) from exc
    return imported_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--lock", type=Path)
    parser.add_argument(
        "--assert-import-source",
        action="store_true",
        help="also require the runtime sparkinfer import to resolve inside --source",
    )
    parser.add_argument(
        "--require-no-python-cache",
        action="store_true",
        help=(
            "reject __pycache__, Python bytecode, and tool cache directories; "
            "use for metadata-free release source copies"
        ),
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--print-revision", action="store_true")
    output.add_argument("--print-tree-sha256", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    if args.require_no_python_cache:
        require_no_python_cache(source)
    if args.print_tree_sha256:
        _verify_package_identity(source)
        print(source_tree_sha256(source))
        return 0
    if args.lock is None:
        raise VerificationError("--lock is required unless --print-tree-sha256 is used")
    lock = verify(source, args.lock)
    imported_path = None
    if args.assert_import_source:
        imported_path = verify_import_source(source)
    if args.print_revision:
        print(lock["revision"])
    else:
        print(
            "verified SparkInfer "
            f"revision={lock['revision']} "
            f"source_tree_sha256={lock['source_tree_sha256']} "
            f"source={source}"
            + (f" import={imported_path}" if imported_path is not None else "")
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(f"verify-sparkinfer-source: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
