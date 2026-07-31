#!/usr/bin/env python3
"""Verify the complete build-relevant inventory of a frozen GLMRT source tree."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys


sys.dont_write_bytecode = True

MANIFEST_LINE_RE = re.compile(r"^([0-9a-f]{64})  (\./.+)$")
IGNORED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        ".glmrt-cache",
        ".glmrt-release",
        ".glmrt-release-image",
        "dist",
    }
)
IGNORED_FILE_SUFFIXES = (".pyc", ".pyo")


class SourceManifestError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ignored_directory(relative: PurePosixPath) -> bool:
    parts = relative.parts
    if any(part in IGNORED_DIRECTORY_NAMES for part in parts):
        return True
    if len(parts) >= 2 and parts[:2] == ("rust", "target"):
        return True
    return len(parts) >= 2 and parts[0] == "native" and parts[1].startswith(
        "build"
    )


def ignored_file(relative: PurePosixPath) -> bool:
    return (
        relative.name == ".git"
        or any(
            part in IGNORED_DIRECTORY_NAMES for part in relative.parts[:-1]
        )
        or relative.name.endswith(IGNORED_FILE_SUFFIXES)
    )


def source_inventory(source: Path) -> set[str]:
    inventory: set[str] = set()
    unsupported: list[str] = []
    for current_root, directory_names, file_names in os.walk(
        source, topdown=True, followlinks=False
    ):
        current = Path(current_root)
        relative_root = current.relative_to(source)
        kept_directories: list[str] = []
        for name in directory_names:
            path = current / name
            relative = PurePosixPath(*(relative_root / name).parts)
            if ignored_directory(relative):
                continue
            if path.is_symlink():
                unsupported.append(f"./{relative.as_posix()} (symlink directory)")
                continue
            kept_directories.append(name)
        directory_names[:] = kept_directories

        for name in file_names:
            path = current / name
            relative = PurePosixPath(*(relative_root / name).parts)
            if ignored_file(relative):
                continue
            mode = path.lstat().st_mode
            display = f"./{relative.as_posix()}"
            if stat.S_ISLNK(mode):
                unsupported.append(f"{display} (symlink)")
            elif stat.S_ISREG(mode):
                inventory.add(display)
            else:
                unsupported.append(f"{display} (non-regular file)")
    if unsupported:
        details = "\n  ".join(unsupported[:20])
        raise SourceManifestError(
            "release source contains unsupported build-relevant entries:\n  "
            f"{details}"
        )
    return inventory


def read_manifest(path: str) -> tuple[dict[str, str], bytes]:
    if path == "-":
        manifest_bytes = sys.stdin.buffer.read()
    else:
        manifest_bytes = Path(path).read_bytes()
    try:
        lines = manifest_bytes.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise SourceManifestError("source manifest is not UTF-8") from error

    expected: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        match = MANIFEST_LINE_RE.fullmatch(line)
        if match is None:
            raise SourceManifestError(
                f"invalid source manifest line {line_number}: {line!r}"
            )
        digest, relative = match.groups()
        if relative in expected:
            raise SourceManifestError(
                f"duplicate source manifest path on line {line_number}: {relative}"
            )
        expected[relative] = digest
    if not expected:
        raise SourceManifestError("source manifest is empty")
    return expected, manifest_bytes


def verify(source: Path, manifest_path: str) -> tuple[str, int]:
    source = source.resolve()
    if not source.is_dir():
        raise SourceManifestError(f"source directory not found: {source}")

    expected, manifest_bytes = read_manifest(manifest_path)
    actual = source_inventory(source)
    expected_paths = set(expected)
    if actual != expected_paths:
        missing = sorted(expected_paths - actual)
        unlisted = sorted(actual - expected_paths)
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing[:20]))
        if unlisted:
            details.append("unlisted: " + ", ".join(unlisted[:20]))
        raise SourceManifestError(
            "release source inventory differs from the manifest ("
            + "; ".join(details)
            + ")"
        )

    mismatched: list[str] = []
    for relative, expected_digest in expected.items():
        actual_digest = file_sha256(source / relative.removeprefix("./"))
        if actual_digest != expected_digest:
            mismatched.append(
                f"{relative}: expected {expected_digest}, found {actual_digest}"
            )
    if mismatched:
        raise SourceManifestError(
            "release source content differs from the manifest:\n  "
            + "\n  ".join(mismatched[:20])
        )
    return hashlib.sha256(manifest_bytes).hexdigest(), len(expected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        required=True,
        help="SOURCE_SHA256SUMS path, or - to read it from stdin",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest_sha256, file_count = verify(args.source, args.manifest)
    except (OSError, SourceManifestError) as error:
        print(f"release source verification failed: {error}", file=sys.stderr)
        return 2
    print(
        "release source verified: "
        f"files={file_count} manifest_sha256={manifest_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
