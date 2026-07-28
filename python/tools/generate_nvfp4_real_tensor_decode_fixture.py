"""Generate the maintained raw-safetensors NVFP4 decode fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

from glmrt_reference.quant_ref import (
    decode_packed_nvfp4_values,
    unpack_low_first_nibbles_bytes,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_snapshot(snapshot_path: str) -> Path:
    snapshot = Path(snapshot_path)
    if snapshot.exists():
        return snapshot
    hf_home = Path(
        os.environ.get("HF_HOME") or Path.home() / ".cache/huggingface"
    )
    hub_cache = Path(os.environ.get("HF_HUB_CACHE") or hf_home / "hub")
    try:
        hub_index = snapshot.parts.index("hub")
        remapped = hub_cache.joinpath(*snapshot.parts[hub_index + 1 :])
    except ValueError:
        remapped = snapshot
    if not remapped.exists():
        raise FileNotFoundError(f"GLM-5.2 snapshot is not available: {snapshot}")
    return remapped


def tensor_by_name(catalog: dict, name: str) -> dict:
    for tensor in catalog["tensors"]:
        if tensor["name"] == name:
            return tensor
    raise KeyError(f"missing tensor {name}")


def read_at(path: Path, offset: int, byte_count: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(byte_count)
    if len(data) != byte_count:
        raise IOError(f"read {len(data)} bytes from {path}, expected {byte_count}")
    return data


def scalar_f32(snapshot: Path, tensor: dict) -> float:
    data = read_at(snapshot / tensor["file"], tensor["byte_offset"], 4)
    return struct.unpack("<f", data)[0]


def build_fixture(args: argparse.Namespace) -> dict:
    catalog_path = Path(args.catalog)
    if not catalog_path.is_absolute():
        catalog_path = repo_root() / catalog_path
    catalog_bytes = catalog_path.read_bytes()
    catalog = json.loads(catalog_bytes.decode("utf-8"))
    snapshot = resolve_snapshot(catalog["snapshot_path"])

    base_name = f"model.layers.{args.layer}.mlp.experts.{args.expert}.{args.projection}"
    weight = tensor_by_name(catalog, f"{base_name}.weight")
    weight_scale = tensor_by_name(catalog, f"{base_name}.weight_scale")
    input_scale = tensor_by_name(catalog, f"{base_name}.input_scale")
    weight_scale_2 = tensor_by_name(catalog, f"{base_name}.weight_scale_2")

    if weight["dtype"] != "u8" or weight_scale["dtype"] != "f8e4m3":
        raise ValueError(
            f"unexpected tensor dtypes: {weight['dtype']}, {weight_scale['dtype']}"
        )
    if input_scale["shape"] or weight_scale_2["shape"]:
        raise ValueError("input_scale and weight_scale_2 must be scalar tensors")

    packed_byte_count = (args.value_count + 1) // 2
    scale_byte_count = (args.value_count + 15) // 16
    packed_offset = weight["byte_offset"] + args.row * weight["shape"][1]
    scale_offset = weight_scale["byte_offset"] + args.row * weight_scale["shape"][1]
    packed = read_at(snapshot / weight["file"], packed_offset, packed_byte_count)
    scales = read_at(snapshot / weight_scale["file"], scale_offset, scale_byte_count)
    full_row_value_count = weight["shape"][1] * 2
    full_row_packed_byte_count = weight["shape"][1]
    full_row_scale_byte_count = weight_scale["shape"][1]
    full_row_packed = read_at(
        snapshot / weight["file"],
        packed_offset,
        full_row_packed_byte_count,
    )
    full_row_scales = read_at(
        snapshot / weight_scale["file"],
        scale_offset,
        full_row_scale_byte_count,
    )
    scale_2 = scalar_f32(snapshot, weight_scale_2)
    input_scale_value = scalar_f32(snapshot, input_scale)
    decoded = decode_packed_nvfp4_values(packed, scales, scale_2, args.value_count)
    full_row_decoded = decode_packed_nvfp4_values(
        full_row_packed,
        full_row_scales,
        scale_2,
        full_row_value_count,
    )

    return {
        "format_version": 1,
        "source": "python-reference-raw-safetensors",
        "model_id": catalog["model_id"],
        "snapshot_path": catalog["snapshot_path"],
        "catalog_path": str(catalog_path.relative_to(repo_root())),
        "catalog_sha256": hashlib.sha256(catalog_bytes).hexdigest(),
        "quant_recipe": "nvfp4-e2m1-f8e4m3",
        "packing_order": "low-nibble-first",
        "value_formula": "e2m1_code * f8e4m3(weight_scale[value_index / 16]) * weight_scale_2",
        "tolerance_abs": 1.0e-6,
        "layer_id": args.layer,
        "expert_id": args.expert,
        "projection": args.projection,
        "row_index": args.row,
        "value_count": args.value_count,
        "packed_byte_count": packed_byte_count,
        "scale_byte_count": scale_byte_count,
        "packed_bytes_hex": packed.hex(),
        "scale_bytes_hex": scales.hex(),
        "nibble_codes": unpack_low_first_nibbles_bytes(packed, args.value_count),
        "input_scale": input_scale_value,
        "weight_scale_2": scale_2,
        "decoded_values": decoded,
        "decoded_checksum": sum(decoded),
        "full_row": {
            "value_count": full_row_value_count,
            "packed_byte_count": full_row_packed_byte_count,
            "scale_byte_count": full_row_scale_byte_count,
            "packed_bytes_hex": full_row_packed.hex(),
            "scale_bytes_hex": full_row_scales.hex(),
            "decoded_checksum": sum(full_row_decoded),
            "decoded_l2_norm": sum(value * value for value in full_row_decoded),
            "first_decoded": full_row_decoded[0],
            "last_decoded": full_row_decoded[-1],
        },
        "tensors": {
            "weight": weight,
            "weight_scale": weight_scale,
            "input_scale": input_scale,
            "weight_scale_2": weight_scale_2,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        required=True,
        help="diagnostic catalog generated by `just inspect-model`",
    )
    parser.add_argument(
        "--output",
        default="tests/fixtures/nvfp4/real_tensor_decode.json",
    )
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--projection", default="gate_proj")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--value-count", type=int, default=64)
    args = parser.parse_args()

    output = Path(args.output)
    if not output.is_absolute():
        output = repo_root() / output
    output.parent.mkdir(parents=True, exist_ok=True)
    fixture = build_fixture(args)
    output.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output.relative_to(repo_root())}")
    print(
        "decoded_checksum="
        f"{fixture['decoded_checksum']:.12g} "
        f"packed_bytes={fixture['packed_byte_count']} scale_bytes={fixture['scale_byte_count']}"
    )


if __name__ == "__main__":
    main()
