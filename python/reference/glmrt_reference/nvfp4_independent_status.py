from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_ORACLE_MODULES = ("torch", "modelopt", "modelopt.torch")
SUPPORT_MODULES = ("safetensors",)


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def build_independent_reference_status(
    repo_root: Path,
    *,
    fixture_path: Path | None = None,
    summary_path: Path | None = None,
    verification_path: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    fixture_path = fixture_path or (
        repo_root / "tests/fixtures/nvfp4/real_tensor_decode.json"
    )
    summary_path = summary_path or repo_root / "reports/phase0_summary.json"
    verification_path = verification_path or (
        repo_root / "tests/fixtures/nvfp4/modelopt_reference.json"
    )

    fixture = _read_json(fixture_path, "NVFP4 real tensor decode fixture")
    summary = _read_optional_json(summary_path)
    verification = _read_optional_json(verification_path)
    verification_passed = _verification_passed(verification, fixture_path, repo_root)
    available = {
        name: module_available(name)
        for name in (*REQUIRED_ORACLE_MODULES, *SUPPORT_MODULES)
    }
    missing_oracle_modules = [
        name for name in REQUIRED_ORACLE_MODULES if not available[name]
    ]

    if verification_passed:
        status = "independent_modelopt_reference_comparison_passed"
        reason = (
            "committed NVIDIA ModelOpt NVFP4 reference comparison artifact "
            "matches the real checkpoint packed row/window fixture"
        )
    elif missing_oracle_modules:
        status = "blocked_missing_independent_oracle_dependency"
        reason = (
            "independent PyTorch/ModelOpt NVFP4 oracle cannot run because required "
            f"module(s) are unavailable: {', '.join(missing_oracle_modules)}"
        )
    else:
        status = "blocked_independent_oracle_adapter_not_implemented"
        reason = (
            "torch and modelopt are importable, but GLMRT has not implemented a "
            "ModelOpt-backed real checkpoint routed-expert row comparison adapter"
        )

    comparison_claimed = bool(
        summary.get(
            "nvfp4_recipe_independent_real_reference_comparison",
            summary.get("nvfp4_recipe_independent_validation", False),
        )
    )
    comparison_executed = verification_passed

    return {
        "format_version": 1,
        "status": status,
        "independent_real_reference_comparison": verification_passed,
        "comparison_executed": comparison_executed,
        "reason": reason,
        "required_oracle_modules": list(REQUIRED_ORACLE_MODULES),
        "missing_oracle_modules": missing_oracle_modules,
        "available_modules": available,
        "verification_artifact": (
            {
                "path": _repo_relative(verification_path, repo_root),
                "source": verification.get("source"),
                "independent_reference": verification.get("independent_reference"),
                "container_image": verification.get("container_image"),
                "device": verification.get("device"),
                "cuda_device_name": verification.get("cuda_device_name"),
                "cuda_device_capability": verification.get("cuda_device_capability"),
                "torch_version": verification.get("torch_version"),
                "modelopt_version": verification.get("modelopt_version"),
                "fixture_sha256": verification.get("fixture_sha256"),
                "window_max_abs_diff": verification.get("window", {}).get("max_abs_diff"),
                "full_row_checksum_abs_diff": verification.get("full_row", {}).get(
                    "checksum_abs_diff"
                ),
            }
            if verification_passed
            else None
        ),
        "checked_fixture": {
            "path": _repo_relative(fixture_path, repo_root),
            "source": fixture["source"],
            "model_id": fixture["model_id"],
            "tensor": fixture["tensors"]["weight"]["name"],
            "layer_id": fixture["layer_id"],
            "expert_id": fixture["expert_id"],
            "projection": fixture["projection"],
            "row_index": fixture["row_index"],
            "value_count": fixture["value_count"],
            "decoded_checksum": fixture["decoded_checksum"],
            "tolerance_abs": fixture["tolerance_abs"],
        },
        "phase0_summary_path": _repo_relative(summary_path, repo_root),
        "phase0_summary_claims_independent_comparison": comparison_claimed,
        "phase0_summary_consistent": comparison_claimed == comparison_executed,
        "next_required_step": (
            "Use scripts/verify-nvfp4-modelopt-container.sh to refresh the "
            "ModelOpt comparison artifact after any NVFP4 decode recipe or fixture change."
            if verification_passed
            else "Install a real independent NVFP4 oracle environment with torch and "
            "ModelOpt, then compare the same checkpoint row/window against ModelOpt "
            "output before changing nvfp4_recipe_independent_real_reference_comparison."
        ),
    }


def _read_json(path: Path, description: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{description} is not available: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _verification_passed(verification: dict[str, Any], fixture_path: Path, repo_root: Path) -> bool:
    if not verification:
        return False
    if not verification.get("comparison_passed"):
        return False
    if not verification.get("comparison_executed"):
        return False
    expected_fixture_path = _repo_relative(fixture_path, repo_root)
    if verification.get("fixture_path") != expected_fixture_path:
        return False
    return verification.get("fixture_sha256") == _sha256_file(fixture_path)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root))
    except ValueError:
        return str(path)
