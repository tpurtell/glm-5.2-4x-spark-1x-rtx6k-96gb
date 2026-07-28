#!/usr/bin/env python3
"""Offline safetensors catalog inspection for GLMRT phase0."""

from __future__ import annotations

import argparse
import json
import os
import struct
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "lukealonso/GLM-5.2-NVFP4"
DEFAULT_HOSTS = ["ostrich", "dodo", "emu", "kiwi"]


def hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def model_cache(model_id: str) -> Path:
    return hf_home() / "hub" / f"models--{model_id.replace('/', '--')}"


def resolve_snapshot(model_id: str) -> Path:
    snapshots = sorted((model_cache(model_id) / "snapshots").glob("*"))
    snapshots = [path for path in snapshots if path.is_dir()]
    if not snapshots:
        raise SystemExit(f"no local snapshot found for {model_id}")
    return snapshots[-1]


def read_config(snapshot: Path, model_id: str) -> dict[str, Any]:
    with (snapshot / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    quant = config.get("quantization_config", {})
    algo = str(quant.get("quant_algo", "NVFP4")).lower()
    recipe = "glm52_nvfp4_lukealonso_v1" if algo == "nvfp4" else f"unknown_{algo}"
    return {
        "model_id": model_id,
        "hidden_size": config.get("hidden_size", 6144),
        "num_hidden_layers": config.get("num_hidden_layers", 78),
        "first_k_dense_replace": config.get("first_k_dense_replace", 3),
        "routed_experts": config.get("n_routed_experts", 256),
        "top_k": config.get("num_experts_per_tok", 8),
        "quantization_recipe": recipe,
    }


def parse_safetensors_header(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    data_start = 8 + header_len
    out: dict[str, dict[str, Any]] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        out[name] = {
            "dtype": dtype_from_safetensors(meta["dtype"]),
            "shape": meta["shape"],
            "byte_offset": data_start + start,
            "byte_length": end - start,
        }
    return out


def dtype_from_safetensors(dtype: str) -> str:
    return {
        "BF16": "bf16",
        "F16": "f16",
        "F32": "f32",
        "F8_E4M3": "f8e4m3",
        "F8_E5M2": "f8e5m2",
        "I8": "i8",
        "I16": "i16",
        "I32": "i32",
        "U8": "u8",
        "F4": "f4",
    }.get(dtype, f"unknown:{dtype}")


def extract_number(name: str, marker: str) -> int | None:
    if marker not in name:
        return None
    tail = name.split(marker, 1)[1]
    digits = []
    for ch in tail:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    return int("".join(digits)) if digits else None


def is_quantization_tensor(name: str) -> bool:
    return (
        name.endswith(".input_scale")
        or name.endswith(".weight_scale")
        or name.endswith(".weight_scale_2")
    )


def classify(name: str, layer_id: int | None, expert_id: int | None, facts: dict[str, Any]) -> str:
    if layer_id is not None and layer_id >= int(facts["num_hidden_layers"]):
        return "mtp"
    if expert_id is not None and ".mlp.experts." in name:
        return "routed-expert"
    if ".mlp.shared_experts." in name:
        return "shared-expert"
    if ".mlp.gate." in name:
        return "router"
    if name == "model.embed_tokens.weight":
        return "embedding"
    if name == "lm_head.weight":
        return "lm-head"
    if ".self_attn." in name:
        return "attention"
    if "layernorm" in name or name.endswith(".norm.weight") or ".norm." in name:
        return "norm"
    if ".mlp." in name:
        return "dense-mlp"
    return "other"


def build_catalog(model_id: str) -> dict[str, Any]:
    snapshot = resolve_snapshot(model_id)
    facts = read_config(snapshot, model_id)
    with (snapshot / "model.safetensors.index.json").open("r", encoding="utf-8") as f:
        index = json.load(f)["weight_map"]
    files = sorted(set(index.values()))
    headers: dict[str, tuple[str, dict[str, Any]]] = {}
    for file_name in files:
        for name, meta in parse_safetensors_header(snapshot / file_name).items():
            headers[name] = (file_name, meta)

    tensors = []
    for name, file_name in index.items():
        header_file, meta = headers[name]
        if header_file != file_name:
            raise SystemExit(f"index/header mismatch for {name}: {file_name} != {header_file}")
        layer_id = extract_number(name, "model.layers.")
        expert_id = extract_number(name, ".mlp.experts.")
        quant = is_quantization_tensor(name)
        tensors.append(
            {
                "name": name,
                "file": file_name,
                "dtype": meta["dtype"],
                "shape": meta["shape"],
                "byte_offset": meta["byte_offset"],
                "byte_length": meta["byte_length"],
                "role": classify(name, layer_id, expert_id, facts),
                "layer_id": layer_id,
                "expert_id": expert_id,
                "is_quantization_metadata": quant,
            }
        )
    tensors.sort(key=lambda item: item["name"])
    return {
        "model_id": model_id,
        "snapshot_path": str(snapshot),
        "facts": facts,
        "tensors": tensors,
    }


def write_summary(catalog: dict[str, Any], path: Path) -> None:
    counts = Counter(t["role"] for t in catalog["tensors"])
    routed_quant = sum(
        1
        for t in catalog["tensors"]
        if t["role"] == "routed-expert" and t["is_quantization_metadata"]
    )
    lines = [
        "# Tensor Classification Summary",
        "",
        f"- Model: `{catalog['model_id']}`",
        f"- Snapshot: `{catalog['snapshot_path']}`",
        f"- Tensor count: `{len(catalog['tensors'])}`",
        f"- Hidden size: `{catalog['facts']['hidden_size']}`",
        f"- Hidden layers: `{catalog['facts']['num_hidden_layers']}`",
        f"- Routed experts per MoE layer: `{catalog['facts']['routed_experts']}`",
        f"- Top-k experts per token: `{catalog['facts']['top_k']}`",
        f"- Quantization recipe: `{catalog['facts']['quantization_recipe']}`",
        "",
        "## Role Counts",
        "",
        "| Role | Tensors |",
        "| --- | ---: |",
    ]
    for role, count in sorted(counts.items()):
        lines.append(f"| {role} | {count} |")
    lines.extend(
        [
            "",
            "## Routed Expert Detail",
            "",
            f"- Routed expert quantization tensors: `{routed_quant}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    catalog = build_catalog(args.model_id)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(catalog, separators=(",", ":")), encoding="utf-8")
    write_summary(catalog, args.summary)
    print(f"wrote catalog tensors={len(catalog['tensors'])} out={args.out}")


if __name__ == "__main__":
    main()

