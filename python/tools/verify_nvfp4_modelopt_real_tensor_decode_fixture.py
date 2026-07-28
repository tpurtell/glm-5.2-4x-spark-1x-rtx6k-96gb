#!/usr/bin/env python3
"""Verify the real NVFP4 decode fixture against NVIDIA ModelOpt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import modelopt
import torch
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else repo_root() / path


def dequantize_modelopt(
    *,
    packed_bytes_hex: str,
    scale_bytes_hex: str,
    value_count: int,
    weight_scale_2: float,
    device: torch.device,
) -> torch.Tensor:
    packed = torch.tensor(
        list(bytes.fromhex(packed_bytes_hex)),
        dtype=torch.uint8,
        device=device,
    ).reshape(1, -1)
    scale_u8 = torch.tensor(
        list(bytes.fromhex(scale_bytes_hex)),
        dtype=torch.uint8,
        device=device,
    )
    scale = scale_u8.view(torch.float8_e4m3fn).reshape(1, -1)
    tensor = NVFP4QTensor(torch.Size([1, value_count]), torch.float32, packed)
    decoded = tensor.dequantize(
        dtype=torch.float32,
        scale=scale,
        double_scale=torch.tensor(weight_scale_2, dtype=torch.float32, device=device),
        block_sizes={-1: 16},
    ).reshape(-1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return decoded


def compare_sequence(actual: torch.Tensor, expected: list[float], tolerance: float) -> dict[str, Any]:
    expected_tensor = torch.tensor(expected, dtype=torch.float32)
    diff = torch.abs(actual.cpu() - expected_tensor)
    max_abs_diff = float(diff.max().item()) if diff.numel() else 0.0
    return {
        "value_count": int(actual.numel()),
        "max_abs_diff": max_abs_diff,
        "passed": max_abs_diff <= tolerance,
    }


def build_verification(fixture_path: Path) -> dict[str, Any]:
    fixture_bytes = fixture_path.read_bytes()
    fixture = json.loads(fixture_bytes.decode("utf-8"))
    tolerance = float(fixture["tolerance_abs"])
    weight_scale_2 = float(fixture["weight_scale_2"])
    device = resolve_device(os.environ.get("GLMRT_NVFP4_MODEL_OPT_DEVICE", "auto"))

    window = dequantize_modelopt(
        packed_bytes_hex=fixture["packed_bytes_hex"],
        scale_bytes_hex=fixture["scale_bytes_hex"],
        value_count=int(fixture["value_count"]),
        weight_scale_2=weight_scale_2,
        device=device,
    )
    window_comparison = compare_sequence(window, fixture["decoded_values"], tolerance)
    window_checksum = float(window.sum().item())

    full_fixture = fixture["full_row"]
    full = dequantize_modelopt(
        packed_bytes_hex=full_fixture["packed_bytes_hex"],
        scale_bytes_hex=full_fixture["scale_bytes_hex"],
        value_count=int(full_fixture["value_count"]),
        weight_scale_2=weight_scale_2,
        device=device,
    )
    full_checksum = float(full.sum().item())
    full_l2_norm = float((full * full).sum().item())
    full_first = float(full[0].item())
    full_last = float(full[-1].item())
    full_checks = {
        "value_count": int(full.numel()),
        "checksum": full_checksum,
        "checksum_expected": float(full_fixture["decoded_checksum"]),
        "checksum_abs_diff": abs(full_checksum - float(full_fixture["decoded_checksum"])),
        "l2_norm": full_l2_norm,
        "l2_norm_expected": float(full_fixture["decoded_l2_norm"]),
        "l2_norm_abs_diff": abs(full_l2_norm - float(full_fixture["decoded_l2_norm"])),
        "first_decoded": full_first,
        "first_decoded_expected": float(full_fixture["first_decoded"]),
        "last_decoded": full_last,
        "last_decoded_expected": float(full_fixture["last_decoded"]),
    }
    full_checks["passed"] = (
        full_checks["checksum_abs_diff"] <= tolerance
        and full_checks["l2_norm_abs_diff"] <= tolerance
        and abs(full_first - float(full_fixture["first_decoded"])) <= tolerance
        and abs(full_last - float(full_fixture["last_decoded"])) <= tolerance
    )
    comparison_passed = bool(window_comparison["passed"] and full_checks["passed"])

    return {
        "format_version": 1,
        "schema": "glmrt.phase0.nvfp4_modelopt_reference.v1",
        "source": "python/tools/verify_nvfp4_modelopt_real_tensor_decode_fixture.py",
        "status": "passed" if comparison_passed else "failed",
        "comparison_executed": True,
        "comparison_passed": comparison_passed,
        "independent_reference": "modelopt.torch.quantization.qtensor.nvfp4_tensor.NVFP4QTensor",
        "container_image": os.environ.get("GLMRT_NVFP4_MODEL_OPT_IMAGE", "unknown"),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "cuda_device_capability": (
            list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None
        ),
        "torch_version": torch.__version__,
        "modelopt_version": getattr(modelopt, "__version__", "unknown"),
        "torch_float4_dtype": str(torch.float4_e2m1fn_x2),
        "modelopt_e2m1_codebook": [
            float(value)
            for value in NVFP4QTensor.get_e2m1_values(torch.device("cpu")).tolist()
        ],
        "fixture_path": _repo_relative(fixture_path),
        "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "fixture_model_id": fixture["model_id"],
        "tensor": fixture["tensors"]["weight"]["name"],
        "layer_id": fixture["layer_id"],
        "expert_id": fixture["expert_id"],
        "projection": fixture["projection"],
        "row_index": fixture["row_index"],
        "quant_recipe": fixture["quant_recipe"],
        "packing_order": fixture["packing_order"],
        "tolerance_abs": tolerance,
        "window": {
            **window_comparison,
            "checksum": window_checksum,
            "checksum_expected": float(fixture["decoded_checksum"]),
            "checksum_abs_diff": abs(window_checksum - float(fixture["decoded_checksum"])),
        },
        "full_row": full_checks,
    }


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root()))
    except ValueError:
        return str(path)


def resolve_device(requested: str) -> torch.device:
    requested = requested.strip().lower()
    if requested in ("", "auto"):
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA NVFP4 verification requested, but torch.cuda.is_available() is false"
        )
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("tests/fixtures/nvfp4/real_tensor_decode.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/nvfp4/modelopt_reference.json"),
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("GLMRT_NVFP4_MODEL_OPT_DEVICE", "auto"),
        help="torch device for ModelOpt dequantization (default: env or auto)",
    )
    args = parser.parse_args()

    os.environ["GLMRT_NVFP4_MODEL_OPT_DEVICE"] = args.device
    fixture_path = resolve_repo_path(args.fixture)
    output_path = resolve_repo_path(args.output)
    verification = build_verification(fixture_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {_repo_relative(output_path)}")
    print(
        "status="
        f"{verification['status']} "
        f"window_max_abs_diff={verification['window']['max_abs_diff']:.6g} "
        f"full_checksum_abs_diff={verification['full_row']['checksum_abs_diff']:.6g}"
    )
    if not verification["comparison_passed"]:
        raise SystemExit("ModelOpt NVFP4 fixture comparison failed")


if __name__ == "__main__":
    main()
