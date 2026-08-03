#!/usr/bin/env python3
"""Build the content identity for a resident WIP Spark expert runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXACT_EXPERT_ENVIRONMENT_KEYS = {
    "GLMRT_MODEL_ID",
    "GLMRT_MTP_BF16_EXPERTS",
    "GLMRT_PROTOCOL_V2_TCP_TIMING",
    "GLMRT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES",
    "GLMRT_REAL_FULL_CUDA_ROUTE_VALIDATE",
    "GLMRT_SERVE_PROFILE",
    "GLMRT_SPARK_BUILD_PROFILE",
    "GLMRT_SPARK_EXPERT_REAL_LAYER",
    "GLMRT_SPARK_EXPERT_TRANSPORT",
    "GLMRT_SPARK_GPU_RUNTIME",
    "GLMRT_SPARK_INCLUDE_MTP_LAYER",
    "GLMRT_VERBS_APP_IB_PORT_NUM",
}
EXPERT_ENVIRONMENT_PREFIXES = (
    "GLMRT_B12X_SPARK_",
    "GLMRT_EXPERT_",
    "GLMRT_REAL_FULL_NVFP4_ROUTE_",
    "GLMRT_REAL_FULL_PROTOCOL_V2_",
    "GLMRT_SPARK_LAYER_",
    "GLMRT_SPARK_NCCL_",
    "GLMRT_SPARK_TRANSFORMER_TP_",
)


def is_expert_environment_key(key: str) -> bool:
    return key in EXACT_EXPERT_ENVIRONMENT_KEYS or key.startswith(
        EXPERT_ENVIRONMENT_PREFIXES
    )


def parse_settings(values: list[str]) -> dict[str, str]:
    settings: dict[str, str] = {}
    for value in values:
        key, separator, setting = value.partition("=")
        if not separator or not key or key in settings:
            raise ValueError(f"invalid or duplicate expert runtime setting: {value!r}")
        settings[key] = setting
    return settings


def build_payload(
    resolved_profile: dict[str, object],
    expert_slot_fingerprint: str,
    settings: dict[str, str],
    ambient_environment: dict[str, str],
) -> dict[str, object]:
    profile_environment = resolved_profile.get("environment")
    if not isinstance(profile_environment, dict) or not all(
        isinstance(key, str) and isinstance(value, (str, int, float, bool))
        for key, value in profile_environment.items()
    ):
        raise ValueError("resolved profile has no scalar environment object")

    # Profile resolution has the same last-writer-wins precedence used by the
    # launcher before phase0-spark-tcp-bench.sh is invoked.
    expert_environment = {
        key: value
        for key, value in ambient_environment.items()
        if is_expert_environment_key(key)
    }
    expert_environment.update(
        {
            key: str(value)
            for key, value in profile_environment.items()
            if is_expert_environment_key(key)
        }
    )
    return {
        "schema": "glmrt-wip-expert-runtime-v2",
        "expert_slot_fingerprint": expert_slot_fingerprint,
        "environment": expert_environment,
        "settings": settings,
    }


def canonical_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-profile", type=Path, required=True)
    parser.add_argument("--expert-slot-fingerprint", required=True)
    parser.add_argument("--setting", action="append", default=[])
    parser.add_argument("--print-payload", action="store_true")
    args = parser.parse_args()

    if not SHA256_RE.fullmatch(args.expert_slot_fingerprint):
        parser.error("--expert-slot-fingerprint must be a lowercase SHA-256 value")
    try:
        resolved_profile = json.loads(args.resolved_profile.read_text(encoding="utf-8"))
        if not isinstance(resolved_profile, dict):
            raise ValueError("resolved profile root is not an object")
        payload = build_payload(
            resolved_profile,
            args.expert_slot_fingerprint,
            parse_settings(args.setting),
            dict(os.environ),
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))

    encoded = canonical_bytes(payload)
    if args.print_payload:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(hashlib.sha256(encoded).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
