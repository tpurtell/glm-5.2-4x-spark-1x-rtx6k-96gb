#!/usr/bin/env python3
"""Measure whether packed NVFP4 tensors have useful lossless structure.

This reads safetensors directly with the Python standard library, so it can run
on Spark nodes that do not have torch or safetensors installed. Tensor samples
are spread across the complete payload instead of reading only its first bytes.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


DEFAULT_PATTERN = (
    r"^model\.layers\.(?:3|20|40|60)\.mlp\.experts\."
    r"(?:0|1|17|63|127|191|255)\."
    r"(?:gate_proj|up_proj|down_proj)\.weight$"
)


@dataclass(frozen=True)
class TensorLocation:
    name: str
    shard: Path
    offset: int
    length: int
    dtype: str
    shape: tuple[int, ...]


def parse_size(value: str) -> int:
    units = {
        "": 1,
        "k": 1 << 10,
        "ki": 1 << 10,
        "m": 1 << 20,
        "mi": 1 << 20,
        "g": 1 << 30,
        "gi": 1 << 30,
    }
    match = re.fullmatch(r"(?i)\s*(\d+)\s*([kmgi]*)b?\s*", value)
    if match is None or match.group(2).lower() not in units:
        raise argparse.ArgumentTypeError(f"invalid byte size: {value}")
    return int(match.group(1)) * units[match.group(2).lower()]


def resolve_snapshot(path: Path) -> Path:
    if (path / "model.safetensors.index.json").is_file():
        return path
    snapshots = path / "snapshots"
    candidates = sorted(
        candidate
        for candidate in snapshots.iterdir()
        if (candidate / "model.safetensors.index.json").is_file()
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one safetensors snapshot below {path}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def read_header(handle: BinaryIO) -> tuple[int, dict[str, object]]:
    length_bytes = handle.read(8)
    if len(length_bytes) != 8:
        raise RuntimeError("truncated safetensors header length")
    header_length = struct.unpack("<Q", length_bytes)[0]
    header = json.loads(handle.read(header_length))
    return 8 + header_length, header


def tensor_locations(
    snapshot: Path, pattern: re.Pattern[str]
) -> list[TensorLocation]:
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    selected = sorted(name for name in weight_map if pattern.search(name))
    headers: dict[str, tuple[int, dict[str, object]]] = {}
    locations = []
    for name in selected:
        shard_name = weight_map[name]
        if shard_name not in headers:
            with (snapshot / shard_name).open("rb") as handle:
                headers[shard_name] = read_header(handle)
        data_start, header = headers[shard_name]
        metadata = header[name]
        start, end = metadata["data_offsets"]
        locations.append(
            TensorLocation(
                name=name,
                shard=snapshot / shard_name,
                offset=data_start + start,
                length=end - start,
                dtype=metadata["dtype"],
                shape=tuple(metadata["shape"]),
            )
        )
    return locations


def sample_ranges(length: int, maximum: int, chunk_bytes: int) -> list[tuple[int, int]]:
    if length <= maximum:
        return [(0, length)]
    sample_count = max(1, maximum // chunk_bytes)
    sample_length = min(chunk_bytes, length)
    if sample_count == 1:
        return [((length - sample_length) // 2, sample_length)]
    maximum_start = length - sample_length
    return [
        (round(index * maximum_start / (sample_count - 1)), sample_length)
        for index in range(sample_count)
    ]


def entropy(histogram: list[int]) -> float:
    total = sum(histogram)
    if total == 0:
        return 0.0
    return -sum(
        (count / total) * math.log2(count / total)
        for count in histogram
        if count
    )


def analyze_tensor(
    location: TensorLocation, maximum: int, chunk_bytes: int
) -> dict[str, object]:
    byte_histogram: collections.Counter[int] = collections.Counter()
    compressor = zlib.compressobj(level=1)
    compressed_bytes = 0
    sampled_bytes = 0
    with location.shard.open("rb") as handle:
        for relative_offset, length in sample_ranges(
            location.length, maximum, chunk_bytes
        ):
            handle.seek(location.offset + relative_offset)
            payload = handle.read(length)
            if len(payload) != length:
                raise RuntimeError(f"truncated tensor data for {location.name}")
            byte_histogram.update(payload)
            compressed_bytes += len(compressor.compress(payload))
            sampled_bytes += len(payload)
    compressed_bytes += len(compressor.flush())

    byte_counts = [byte_histogram[index] for index in range(256)]
    nibble_counts = [0] * 16
    for byte, count in enumerate(byte_counts):
        nibble_counts[byte & 0xF] += count
        nibble_counts[byte >> 4] += count
    byte_entropy = entropy(byte_counts)
    nibble_entropy = entropy(nibble_counts)
    return {
        "name": location.name,
        "shard": location.shard.name,
        "dtype": location.dtype,
        "shape": location.shape,
        "tensor_bytes": location.length,
        "sampled_bytes": sampled_bytes,
        "byte_entropy_bits": byte_entropy,
        "independent_nibble_entropy_bits": nibble_entropy,
        "byte_entropy_lower_bound_ratio": byte_entropy / 8.0,
        "independent_nibble_lower_bound_ratio": nibble_entropy / 4.0,
        "zlib_level1_ratio": compressed_bytes / sampled_bytes,
        "zero_nibble_fraction": nibble_counts[0] / sum(nibble_counts),
        "nibble_histogram": nibble_counts,
        "_byte_histogram": byte_counts,
        "_compressed_bytes": compressed_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument(
        "--max-bytes-per-tensor", type=parse_size, default=parse_size("16MiB")
    )
    parser.add_argument("--chunk-bytes", type=parse_size, default=parse_size("1MiB"))
    parser.add_argument("--per-tensor", action="store_true")
    args = parser.parse_args()

    snapshot = resolve_snapshot(args.model)
    pattern = re.compile(args.pattern)
    locations = tensor_locations(snapshot, pattern)
    if not locations:
        raise RuntimeError(f"no tensors matched {args.pattern!r}")

    combined_byte_histogram = [0] * 256
    combined_nibble_histogram = [0] * 16
    total_tensor_bytes = 0
    total_sampled_bytes = 0
    total_compressed_bytes = 0
    results = []
    for location in locations:
        result = analyze_tensor(
            location, args.max_bytes_per_tensor, args.chunk_bytes
        )
        total_tensor_bytes += int(result["tensor_bytes"])
        total_sampled_bytes += int(result["sampled_bytes"])
        total_compressed_bytes += int(result.pop("_compressed_bytes"))
        byte_histogram = result.pop("_byte_histogram")
        for index, count in enumerate(byte_histogram):
            combined_byte_histogram[index] += count
            combined_nibble_histogram[index & 0xF] += count
            combined_nibble_histogram[index >> 4] += count
        if args.per_tensor:
            print(json.dumps(result, separators=(",", ":")))
        results.append(result)

    byte_entropy = entropy(combined_byte_histogram)
    nibble_entropy = entropy(combined_nibble_histogram)
    summary = {
        "snapshot": os.fspath(snapshot),
        "pattern": args.pattern,
        "tensor_count": len(results),
        "tensor_bytes": total_tensor_bytes,
        "sampled_bytes": total_sampled_bytes,
        "byte_entropy_bits": byte_entropy,
        "independent_nibble_entropy_bits": nibble_entropy,
        "byte_entropy_lower_bound_ratio": byte_entropy / 8.0,
        "independent_nibble_lower_bound_ratio": nibble_entropy / 4.0,
        "zlib_level1_ratio": total_compressed_bytes / total_sampled_bytes,
        "zero_nibble_fraction": (
            combined_nibble_histogram[0] / sum(combined_nibble_histogram)
        ),
        "nibble_histogram": combined_nibble_histogram,
    }
    print(json.dumps(summary, separators=(",", ":")))


if __name__ == "__main__":
    main()
