#!/usr/bin/env python3
"""Cache and validate the pinned GLM-5.2 DSpark implementation fixtures."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import prod
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open


@dataclass(frozen=True)
class DsparkFixture:
    repo_id: str
    revision: str
    block_size: int
    speculative_tokens: int
    verifier: str
    verifier_revision: str | None
    aux_layers: tuple[int, ...]
    draft_layers: int
    sample_from_anchor: bool
    sliding_window: int | None
    model_bytes: int
    tensor_count: int
    parameter_count: int


FIXTURES = (
    DsparkFixture(
        repo_id="RedHatAI/GLM-5.2-speculator.dspark",
        revision="8bc9ac46fbf507f3ee3ad82304116a1f63e9edb4",
        block_size=8,
        speculative_tokens=8,
        verifier="RedHatAI/GLM-5.2-NVFP4-FP8",
        verifier_revision=None,
        aux_layers=(2, 20, 39, 58, 75),
        draft_layers=3,
        sample_from_anchor=True,
        sliding_window=2_048,
        model_bytes=6_305_465_978,
        tensor_count=42,
        parameter_count=3_152_730_753,
    ),
    DsparkFixture(
        repo_id="siro1/glm-5.2-dspark-preview",
        revision="7ff03018b3a443bfb9fca166739bd5f37ee5908b",
        block_size=16,
        speculative_tokens=15,
        verifier="nvidia/GLM-5.2-NVFP4",
        verifier_revision="aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa",
        aux_layers=(8, 23, 39, 55, 70),
        draft_layers=5,
        sample_from_anchor=False,
        sliding_window=None,
        model_bytes=7_614_140_882,
        tensor_count=64,
        parameter_count=3_807_067_009,
    ),
)

EXPECTED_VOCAB_SIZE = 154_880
EXPECTED_HIDDEN_SIZE = 6_144
EXPECTED_MARKOV_RANK = 256
EXPECTED_MASK_TOKEN_ID = 154_856


def expected_tensor_shapes(fixture: DsparkFixture) -> dict[str, tuple[int, ...]]:
    shapes = {
        "confidence_head.proj.bias": (1,),
        "confidence_head.proj.weight": (1, 6_400),
        "embed_tokens.weight": (EXPECTED_VOCAB_SIZE, EXPECTED_HIDDEN_SIZE),
        "fc.weight": (EXPECTED_HIDDEN_SIZE, 5 * EXPECTED_HIDDEN_SIZE),
        "hidden_norm.weight": (EXPECTED_HIDDEN_SIZE,),
        "lm_head.weight": (EXPECTED_VOCAB_SIZE, EXPECTED_HIDDEN_SIZE),
        "markov_head.markov_w1.weight": (EXPECTED_VOCAB_SIZE, EXPECTED_MARKOV_RANK),
        "markov_head.markov_w2.weight": (EXPECTED_VOCAB_SIZE, EXPECTED_MARKOV_RANK),
        "norm.weight": (EXPECTED_HIDDEN_SIZE,),
    }
    for layer in range(fixture.draft_layers):
        prefix = f"layers.{layer}"
        shapes.update(
            {
                f"{prefix}.input_layernorm.weight": (EXPECTED_HIDDEN_SIZE,),
                f"{prefix}.mlp.down_proj.weight": (
                    EXPECTED_HIDDEN_SIZE,
                    12_288,
                ),
                f"{prefix}.mlp.gate_proj.weight": (
                    12_288,
                    EXPECTED_HIDDEN_SIZE,
                ),
                f"{prefix}.mlp.up_proj.weight": (
                    12_288,
                    EXPECTED_HIDDEN_SIZE,
                ),
                f"{prefix}.post_attention_layernorm.weight": (
                    EXPECTED_HIDDEN_SIZE,
                ),
                f"{prefix}.self_attn.k_norm.weight": (64,),
                f"{prefix}.self_attn.k_proj.weight": (4_096, EXPECTED_HIDDEN_SIZE),
                f"{prefix}.self_attn.o_proj.weight": (EXPECTED_HIDDEN_SIZE, 4_096),
                f"{prefix}.self_attn.q_norm.weight": (64,),
                f"{prefix}.self_attn.q_proj.weight": (4_096, EXPECTED_HIDDEN_SIZE),
                f"{prefix}.self_attn.v_proj.weight": (4_096, EXPECTED_HIDDEN_SIZE),
            }
        )
    return shapes


def validate_weight_manifest(
    fixture: DsparkFixture, weights_path: Path
) -> tuple[int, int]:
    expected = expected_tensor_shapes(fixture)
    with safe_open(weights_path, framework="pt", device="cpu") as weights:
        actual_names = set(weights.keys())
        expected_names = set(expected)
        mismatches: dict[str, object] = {}
        if actual_names != expected_names:
            mismatches["tensor_names"] = {
                "missing": sorted(expected_names - actual_names),
                "unexpected": sorted(actual_names - expected_names),
            }
        parameter_count = 0
        for name in sorted(actual_names & expected_names):
            tensor = weights.get_slice(name)
            shape = tuple(tensor.get_shape())
            dtype = str(tensor.get_dtype())
            if shape != expected[name] or dtype != "BF16":
                mismatches[name] = {
                    "expected_shape": expected[name],
                    "actual_shape": shape,
                    "expected_dtype": "BF16",
                    "actual_dtype": dtype,
                }
            parameter_count += prod(shape)
    if len(actual_names) != fixture.tensor_count:
        mismatches["tensor_count"] = {
            "expected": fixture.tensor_count,
            "actual": len(actual_names),
        }
    if parameter_count != fixture.parameter_count:
        mismatches["parameter_count"] = {
            "expected": fixture.parameter_count,
            "actual": parameter_count,
        }
    if mismatches:
        raise RuntimeError(
            "DSpark safetensors manifest failed validation: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )
    return len(actual_names), parameter_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache both pinned GLM-5.2 DSpark checkpoints and validate their contracts."
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="validate an existing cache without making network requests",
    )
    return parser.parse_args()


def validate_snapshot(fixture: DsparkFixture, snapshot: Path) -> dict[str, object]:
    config_path = snapshot / "config.json"
    weights_path = snapshot / "model.safetensors"
    config = json.loads(config_path.read_text())
    transformer = config["transformer_layer_config"]
    proposal = config["speculators_config"]["proposal_methods"][0]

    expected = {
        "architectures": ["DSparkDraftModel"],
        "speculators_model_type": "dspark",
        "aux_hidden_state_layer_ids": list(fixture.aux_layers),
        "block_size": fixture.block_size,
        "draft_vocab_size": EXPECTED_VOCAB_SIZE,
        "markov_rank": EXPECTED_MARKOV_RANK,
        "mask_token_id": EXPECTED_MASK_TOKEN_ID,
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if bool(config.get("sample_from_anchor", False)) != fixture.sample_from_anchor:
        mismatches["sample_from_anchor"] = {
            "expected": fixture.sample_from_anchor,
            "actual": config.get("sample_from_anchor"),
        }
    transformer_expected = {
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "num_hidden_layers": fixture.draft_layers,
        "vocab_size": EXPECTED_VOCAB_SIZE,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "intermediate_size": 12_288,
        "layer_types": [
            (
                "sliding_attention"
                if fixture.sliding_window is not None
                else "full_attention"
            )
        ]
        * fixture.draft_layers,
        "sliding_window": fixture.sliding_window,
    }
    mismatches.update(
        {
            f"transformer_layer_config.{key}": {
                "expected": value,
                "actual": transformer.get(key),
            }
            for key, value in transformer_expected.items()
            if transformer.get(key) != value
        }
    )
    if proposal.get("speculative_tokens") != fixture.speculative_tokens:
        mismatches["proposal.speculative_tokens"] = {
            "expected": fixture.speculative_tokens,
            "actual": proposal.get("speculative_tokens"),
        }
    verifier = config["speculators_config"]["verifier"].get("name_or_path")
    verifier_matches = verifier == fixture.verifier
    if not verifier_matches and fixture.verifier_revision is not None:
        cache_repo = f"models--{fixture.verifier.replace('/', '--')}"
        verifier_matches = Path(verifier).as_posix().endswith(
            f"{cache_repo}/snapshots/{fixture.verifier_revision}"
        )
    if not verifier_matches:
        mismatches["verifier.name_or_path"] = {
            "expected": fixture.verifier,
            "actual": verifier,
        }
    if not weights_path.is_file() or weights_path.stat().st_size != fixture.model_bytes:
        mismatches["model.safetensors.size"] = {
            "expected": fixture.model_bytes,
            "actual": weights_path.stat().st_size if weights_path.exists() else None,
        }
    if snapshot.name != fixture.revision:
        mismatches["resolved_revision"] = {
            "expected": fixture.revision,
            "actual": snapshot.name,
        }
    if mismatches:
        raise RuntimeError(
            f"DSpark fixture {fixture.repo_id} failed validation: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )
    tensor_count, parameter_count = validate_weight_manifest(fixture, weights_path)

    return {
        "repo_id": fixture.repo_id,
        "revision": fixture.revision,
        "snapshot": str(snapshot),
        "model_bytes": weights_path.stat().st_size,
        "parameter_count": parameter_count,
        "tensor_count": tensor_count,
        "block_size": fixture.block_size,
        "speculative_tokens": fixture.speculative_tokens,
        "verifier": fixture.verifier,
        "aux_hidden_state_layer_ids": list(fixture.aux_layers),
        "checkpoint_convention": (
            "anchor_first"
            if fixture.sample_from_anchor
            else "speculators_bonus_anchor"
        ),
        "sliding_window": fixture.sliding_window,
    }


def main() -> None:
    args = parse_args()
    records = []
    for fixture in FIXTURES:
        snapshot = Path(
            snapshot_download(
                repo_id=fixture.repo_id,
                revision=fixture.revision,
                allow_patterns=[
                    "config.json",
                    "model.safetensors",
                    "README.md",
                ],
                local_files_only=args.local_files_only,
                max_workers=8,
            )
        )
        records.append(validate_snapshot(fixture, snapshot))
    print(json.dumps({"dspark_fixtures": records}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
