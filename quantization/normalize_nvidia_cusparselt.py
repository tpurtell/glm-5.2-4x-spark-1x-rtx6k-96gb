#!/usr/bin/env python3
"""Audit and normalize NVIDIA's arm64 cuSPARSELt wheel platform metadata.

NVIDIA publishes the 0.8.1 arm64 artifact with an aarch64 filename and AArch64
ELF payload, but its internal WHEEL tag says ``manylinux2014_sbsa``. Python's
standard platform tag is ``manylinux2014_aarch64``, so uv installs the correct
artifact and then reports it as incompatible. This narrowly rewrites that one
metadata tag, updates RECORD, and leaves durable evidence in the dist-info
directory. It never modifies the CUDA library.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import struct
import sys
import sysconfig


DIST_NAME = "nvidia_cusparselt_cu13"
DIST_VERSION = "0.8.1"
SOURCE_WHEEL_SHA256 = "4dca476c50bf4780d46cd0bfbd82e2bc10a08e4fef7950917ce8d7578d22a23f"
ORIGINAL_TAG = "Tag: py3-none-manylinux2014_sbsa"
NORMALIZED_TAG = "Tag: py3-none-manylinux2014_aarch64"
MARKER_NAME = "DS4RT_SBSA_NORMALIZATION.json"
ELF_MACHINE_AARCH64 = 183


class NormalizationError(RuntimeError):
    """The installed cuSPARSELt wheel does not match the audited exception."""


def _hash_record(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return "sha256=" + encoded.decode("ascii")


def _distribution(site_packages: Path) -> Path:
    matches = sorted(site_packages.glob(f"{DIST_NAME}-{DIST_VERSION}.dist-info"))
    if len(matches) != 1:
        raise NormalizationError(
            f"expected one {DIST_NAME}-{DIST_VERSION}.dist-info directory, found {matches}"
        )
    return matches[0]


def _elf_machine(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            header = handle.read(20)
    except OSError as exc:
        raise NormalizationError(f"cannot read cuSPARSELt payload: {path}: {exc}") from exc
    if len(header) != 20 or header[:4] != b"\x7fELF":
        raise NormalizationError(f"cuSPARSELt payload is not ELF: {path}")
    if header[4] != 2 or header[5] != 1:
        raise NormalizationError(f"cuSPARSELt payload is not ELF64 little-endian: {path}")
    return struct.unpack("<H", header[18:20])[0]


def _expected_marker(library_relative: str) -> dict[str, object]:
    return {
        "schema": 1,
        "distribution": f"nvidia-cusparselt-cu13=={DIST_VERSION}",
        "source_wheel_sha256": SOURCE_WHEEL_SHA256,
        "reason": "NVIDIA arm64 filename/payload uses aarch64 but WHEEL tag uses nonstandard sbsa",
        "original_tag": ORIGINAL_TAG.removeprefix("Tag: "),
        "normalized_tag": NORMALIZED_TAG.removeprefix("Tag: "),
        "library": library_relative,
        "elf_machine": ELF_MACHINE_AARCH64,
    }


def _load_record(path: Path) -> list[list[str]]:
    try:
        return list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"))))
    except OSError as exc:
        raise NormalizationError(f"cannot read wheel RECORD: {path}: {exc}") from exc


def _replace_record_entry(
    rows: list[list[str]], relative: str, data: bytes, *, allow_add: bool
) -> None:
    replacement = [relative, _hash_record(data), str(len(data))]
    for index, row in enumerate(rows):
        if row and row[0] == relative:
            rows[index] = replacement
            return
    if not allow_add:
        raise NormalizationError(f"wheel RECORD has no entry for {relative}")
    rows.append(replacement)


def _record_bytes(rows: list[list[str]]) -> bytes:
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    return output.getvalue().encode()


def normalize(site_packages: Path, *, apply: bool, machine: str) -> dict[str, object]:
    if machine.lower() not in {"aarch64", "arm64"}:
        return {"status": "not-applicable", "machine": machine.lower()}

    site_packages = site_packages.resolve()
    dist_info = _distribution(site_packages)
    wheel_path = dist_info / "WHEEL"
    record_path = dist_info / "RECORD"
    marker_path = dist_info / MARKER_NAME
    library = site_packages / "nvidia" / "cusparselt" / "lib" / "libcusparseLt.so.0"
    machine_id = _elf_machine(library)
    if machine_id != ELF_MACHINE_AARCH64:
        raise NormalizationError(
            f"cuSPARSELt payload e_machine={machine_id}, expected AArch64 {ELF_MACHINE_AARCH64}"
        )

    library_relative = library.relative_to(site_packages).as_posix()
    expected_marker = _expected_marker(library_relative)
    marker_bytes = (json.dumps(expected_marker, indent=2, sort_keys=True) + "\n").encode()
    wheel_text = wheel_path.read_text(encoding="utf-8")

    if apply:
        if NORMALIZED_TAG in wheel_text and marker_path.is_file():
            return normalize(site_packages, apply=False, machine=machine)
        if wheel_text.count(ORIGINAL_TAG) != 1 or NORMALIZED_TAG in wheel_text:
            raise NormalizationError("cuSPARSELt WHEEL tag is not the audited NVIDIA sbsa tag")
        normalized_bytes = wheel_text.replace(ORIGINAL_TAG, NORMALIZED_TAG).encode()
        rows = _load_record(record_path)
        wheel_relative = wheel_path.relative_to(site_packages).as_posix()
        marker_relative = marker_path.relative_to(site_packages).as_posix()
        _replace_record_entry(rows, wheel_relative, normalized_bytes, allow_add=False)
        _replace_record_entry(rows, marker_relative, marker_bytes, allow_add=True)
        wheel_path.write_bytes(normalized_bytes)
        marker_path.write_bytes(marker_bytes)
        record_path.write_bytes(_record_bytes(rows))

    if wheel_path.read_text(encoding="utf-8").count(NORMALIZED_TAG) != 1:
        raise NormalizationError("normalized cuSPARSELt aarch64 WHEEL tag is missing")
    try:
        actual_marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NormalizationError(f"cannot read cuSPARSELt normalization marker: {exc}") from exc
    if actual_marker != expected_marker:
        raise NormalizationError("cuSPARSELt normalization marker does not match the contract")

    rows = _load_record(record_path)
    by_path = {row[0]: row for row in rows if row}
    for path, data in ((wheel_path, wheel_path.read_bytes()), (marker_path, marker_bytes)):
        relative = path.relative_to(site_packages).as_posix()
        if by_path.get(relative) != [relative, _hash_record(data), str(len(data))]:
            raise NormalizationError(f"cuSPARSELt RECORD mismatch for {relative}")

    return {"status": "normalized-and-verified", **expected_marker}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-packages", type=Path, default=Path(sysconfig.get_paths()["purelib"]))
    parser.add_argument("--machine", default=platform.machine(), help=argparse.SUPPRESS)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = normalize(args.site_packages, apply=args.apply, machine=args.machine)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except NormalizationError as exc:
        print(f"normalize-nvidia-cusparselt: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
