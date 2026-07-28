from __future__ import annotations

import json
from pathlib import Path

import pytest

from glmrt_reference.quant_ref import decode_packed_nvfp4_values, f8e4m3_byte_to_float


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _fixture() -> dict:
    path = _repo_root() / "tests/fixtures/nvfp4/real_tensor_decode.json"
    if not path.exists():
        pytest.skip("NVFP4 real tensor decode fixture is not available")
    return json.loads(path.read_text(encoding="utf-8"))


def test_real_checkpoint_nvfp4_decode_fixture_matches_reference_math():
    fixture = _fixture()
    packed = bytes.fromhex(fixture["packed_bytes_hex"])
    scales = bytes.fromhex(fixture["scale_bytes_hex"])

    decoded = decode_packed_nvfp4_values(
        packed,
        scales,
        fixture["weight_scale_2"],
        fixture["value_count"],
    )

    assert fixture["source"] == "python-reference-raw-safetensors"
    assert fixture["quant_recipe"] == "nvfp4-e2m1-f8e4m3"
    assert fixture["packing_order"] == "low-nibble-first"
    assert fixture["projection"] == "gate_proj"
    assert fixture["tensors"]["weight"]["name"] == "model.layers.3.mlp.experts.0.gate_proj.weight"
    assert fixture["tensors"]["weight_scale"]["dtype"] == "f8e4m3"
    assert f8e4m3_byte_to_float(scales[0]) == pytest.approx(1.25, abs=1.0e-12)
    assert decoded == pytest.approx(fixture["decoded_values"], abs=fixture["tolerance_abs"])
    assert sum(decoded) == pytest.approx(fixture["decoded_checksum"], abs=fixture["tolerance_abs"])

    full_row = fixture["full_row"]
    full_packed = bytes.fromhex(full_row["packed_bytes_hex"])
    full_scales = bytes.fromhex(full_row["scale_bytes_hex"])
    full_decoded = decode_packed_nvfp4_values(
        full_packed,
        full_scales,
        fixture["weight_scale_2"],
        full_row["value_count"],
    )
    full_l2 = sum(value * value for value in full_decoded)

    assert full_row["value_count"] == 6144
    assert len(full_packed) == full_row["packed_byte_count"]
    assert len(full_scales) == full_row["scale_byte_count"]
    assert full_decoded[0] == pytest.approx(full_row["first_decoded"], abs=fixture["tolerance_abs"])
    assert full_decoded[-1] == pytest.approx(full_row["last_decoded"], abs=fixture["tolerance_abs"])
    assert sum(full_decoded) == pytest.approx(
        full_row["decoded_checksum"], abs=fixture["tolerance_abs"]
    )
    assert full_l2 == pytest.approx(full_row["decoded_l2_norm"], abs=fixture["tolerance_abs"])
