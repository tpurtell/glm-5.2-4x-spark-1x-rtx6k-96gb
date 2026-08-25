from __future__ import annotations

import importlib.util
import hashlib
import json
import struct
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
TOOL_PATH = ROOT / "python" / "tools" / "validate_glm52_exl3_artifact.py"
SPEC = importlib.util.spec_from_file_location("_glmrt_exl3_artifact_validator", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


def test_reports_exact_direct_tp4_resident_geometry() -> None:
    assert TOOL.EXPECTED_TP4_RESIDENT_BYTES == 68_714_572_800


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    payload = bytearray()
    for name, (dtype, shape, value) in tensors.items():
        start = len(payload)
        payload.extend(value)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def test_complete_glm52_module_contract_has_every_base_expert_projection() -> None:
    modules = TOOL._module_contract()

    assert len(modules) == 75 * 256 * 3
    assert modules["model.layers.3.mlp.experts.0.gate_proj"] == (6144, 2048)
    assert modules["model.layers.77.mlp.experts.255.down_proj"] == (2048, 6144)
    assert "model.layers.78.mlp.experts.0.gate_proj" not in modules


def test_snapshot_inventory_checks_index_header_and_payload_geometry(tmp_path: Path) -> None:
    shard = tmp_path / "model-00001-of-00001.safetensors"
    write_safetensors(
        shard,
        {
            "a": ("I16", [2], b"\x01\x00\x02\x00"),
            "b": ("I32", [], b"\xed\x1f\xac\xcb"),
        },
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": shard.name, "b": shard.name}}),
        encoding="utf-8",
    )

    inventory = TOOL._snapshot_inventory(tmp_path, reject_symlinks=True)

    assert set(inventory.tensors) == {"a", "b"}
    assert inventory.tensor_bytes == 8
    assert inventory.tensors["a"].shape == (2,)
    assert inventory.tensors["b"].shape == ()


def test_retained_native_comparison_rejects_payload_drift(tmp_path: Path) -> None:
    source_path = tmp_path / "source.safetensors"
    artifact_path = tmp_path / "artifact.safetensors"
    write_safetensors(source_path, {"x": ("I16", [2], b"\x01\x00\x02\x00")})
    write_safetensors(artifact_path, {"x": ("I16", [2], b"\x01\x00\x03\x00")})
    source = TOOL._parse_safetensors(source_path, reject_symlink=True)["x"]
    artifact = TOOL._parse_safetensors(artifact_path, reject_symlink=True)["x"]

    with pytest.raises(TOOL.ArtifactValidationError, match="payload differs"):
        TOOL._compare_retained_tensor("x", source, artifact)


def test_checkpoint_artifact_join_proves_exact_packed_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    module = "model.layers.3.mlp.experts.0.gate_proj"
    contract = TOOL._tensor_contract(module, 16, 16)
    suffix_tensors: dict[str, tuple[str, list[int], bytes]] = {}
    artifact_tensors: dict[str, tuple[str, list[int], bytes]] = {}
    for index, (full_name, (dtype, shape)) in enumerate(contract.items(), start=1):
        suffix = full_name.removeprefix(f"{module}.")
        payload = bytes([index]) * (TOOL.math.prod(shape) * TOOL.DTYPE_BYTES[dtype])
        suffix_tensors[suffix] = (dtype, list(shape), payload)
        artifact_tensors[full_name] = (dtype, list(shape), payload)

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    artifact_shard = artifact / "model-00001-of-00001.safetensors"
    write_safetensors(artifact_shard, artifact_tensors)
    (artifact / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {name: artifact_shard.name for name in artifact_tensors}}
        ),
        encoding="utf-8",
    )
    artifact_inventory = TOOL._snapshot_inventory(artifact, reject_symlinks=True)

    request = {
        "schema": TOOL.CHECKPOINT_SCHEMA,
        "schema_version": TOOL.CHECKPOINT_SCHEMA_VERSION,
        "module": module,
        "processor_layer_index": 3,
    }
    request["request_sha256"] = hashlib.sha256(
        TOOL._canonical_json(request)
    ).hexdigest()
    digest = request["request_sha256"]
    checkpoint_dir = tmp_path / "checkpoints" / digest[:2] / digest[2:4]
    checkpoint_dir.mkdir(parents=True)
    tensor_path = checkpoint_dir / f"{digest}.safetensors"
    write_safetensors(tensor_path, suffix_tensors)
    ledger = {"module": module, "processor_layer_index": 3}
    specs = {
        suffix: {
            "dtype": TOOL.TORCH_DTYPE[dtype],
            "shape": shape,
            "numel": TOOL.math.prod(shape),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for suffix, (dtype, shape, payload) in suffix_tensors.items()
    }
    manifest = {
        "schema": TOOL.CHECKPOINT_SCHEMA,
        "schema_version": TOOL.CHECKPOINT_SCHEMA_VERSION,
        "request": request,
        "request_sha256": digest,
        "tensor_file": tensor_path.name,
        "tensor_sha256": hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
        "tensors": specs,
        "result": {"ledger_record": ledger},
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        TOOL._canonical_json(manifest)
    ).hexdigest()
    (checkpoint_dir / f"{digest}.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    monkeypatch.setattr(TOOL, "EXPECTED_MODULES", 1)
    monkeypatch.setattr(TOOL, "EXPECTED_EXL3_TENSORS", 4)
    report = TOOL._validate_checkpoint_artifact_join(
        checkpoint_root=tmp_path / "checkpoints",
        plan={"projection_checkpoint": {"root": str(tmp_path / "checkpoints")}},
        artifact_inventory=artifact_inventory,
        modules={module: (16, 16)},
    )

    assert report["projection_count"] == 1
    assert report["tensor_count"] == 4
    assert report["tensor_bytes"] == sum(len(value[2]) for value in suffix_tensors.values())
    assert TOOL.SHA256_RE.fullmatch(report["checkpoint_inventory_sha256"])

    drifted = dict(artifact_tensors)
    dtype, shape, payload = drifted[f"{module}.mcg"]
    drifted[f"{module}.mcg"] = (dtype, shape, bytes([payload[0] ^ 0xFF]) + payload[1:])
    write_safetensors(artifact_shard, drifted)
    with pytest.raises(
        TOOL.ArtifactValidationError,
        match="artifact packed tensor differs",
    ):
        TOOL._validate_checkpoint_artifact_join(
            checkpoint_root=tmp_path / "checkpoints",
            plan={"projection_checkpoint": {"root": str(tmp_path / "checkpoints")}},
            artifact_inventory=artifact_inventory,
            modules={module: (16, 16)},
        )


def test_quantization_config_requires_exact_k3_mcg_storage(tmp_path: Path) -> None:
    module = "model.layers.3.mlp.experts.0.gate_proj"
    tensor_contract = TOOL._tensor_contract(module, 6144, 2048)
    storage = {
        module: {
            "stored_tensors": {
                name: {
                    "shape": list(shape),
                    "torch_dtype": {
                        "I16": "int16",
                        "F16": "float16",
                        "I32": "int32",
                    }[dtype],
                }
                for name, (dtype, shape) in tensor_contract.items()
            },
            "quant_format": "exl3",
            "bits_per_weight": 3,
            "mcg_multiplier": TOOL.MCG_MULTIPLIER,
        }
    }
    quantization_config = {
        "method": "exl3",
        "quant_method": "exl3",
        "format": "exl3",
        "checkpoint_format": "exl3",
        "bits": 3.0,
        "codebook": "mcg",
        "out_scales": "auto",
        "group_size": -1,
        "desc_act": False,
        "module_include": [TOOL.MODULE_INCLUDE],
    }
    config = {
        "model_type": "glm_moe_dsa",
        "hidden_size": 6144,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 78,
        "first_k_dense_replace": 3,
        "num_nextn_predict_layers": 1,
        "quantization_config": quantization_config,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "quantize_config.json").write_text(
        json.dumps({**quantization_config, "tensor_storage": storage}),
        encoding="utf-8",
    )

    value = TOOL._validate_quantization_config(tmp_path, {module: (6144, 2048)})
    assert value["tensor_storage"] == storage

    value["tensor_storage"][module]["bits_per_weight"] = 2
    (tmp_path / "quantize_config.json").write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(TOOL.ArtifactValidationError, match="tensor_storage entry"):
        TOOL._validate_quantization_config(tmp_path, {module: (6144, 2048)})


def _tokenizer_snapshot(tmp_path: Path) -> tuple[Path, list[dict[str, object]]]:
    model = tmp_path / "models--zai-org--GLM-5.2"
    snapshot = model / "snapshots" / ("a" * 40)
    blobs = model / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    for name, payload in (
        ("tokenizer.json", b'{"tokenizer": true}\n'),
        ("tokenizer_config.json", b'{"bos_token": "x"}\n'),
    ):
        digest = hashlib.sha256(payload).hexdigest()
        (blobs / digest).write_bytes(payload)
        (snapshot / name).symlink_to(Path("../../blobs") / digest)
    identities = [
        TOOL._source_tokenizer_identity(snapshot, name)
        for name in TOOL.TOKENIZER_FILES
    ]
    return snapshot, identities


def test_tokenizer_evidence_accepts_directly_bound_plan(tmp_path: Path) -> None:
    snapshot, identities = _tokenizer_snapshot(tmp_path)
    plan = {"source": {"tokenizer_files": identities}}

    assert TOOL._validate_tokenizer_evidence(
        plan=plan,
        source=snapshot,
        attestation_path=None,
    ) == {"mode": "plan-bound", "tokenizer_files": identities}

    identities[0]["bytes"] += 1
    with pytest.raises(TOOL.ArtifactValidationError, match="immutable plan"):
        TOOL._validate_tokenizer_evidence(
            plan=plan,
            source=snapshot,
            attestation_path=None,
        )


def test_legacy_tokenizer_attestation_is_content_and_plan_bound(
    tmp_path: Path,
) -> None:
    snapshot, identities = _tokenizer_snapshot(tmp_path)
    corpus = {"examples": 2, "file_sha256": "b" * 64}
    plan = {
        "plan_sha256": "c" * 64,
        "source": {"revision": snapshot.name},
        "corpus": corpus,
        "preflight": {"image_digest": "sha256:" + "d" * 64},
    }
    body = {
        "schema": TOOL.TOKENIZER_ATTESTATION_SCHEMA,
        "status": "accepted",
        "scope": "legacy-plan-omitted-tokenizer-source-identity",
        "plan": {"plan_sha256": plan["plan_sha256"]},
        "source": {
            "path": str(snapshot),
            "revision": snapshot.name,
            "tokenizer_files": [
                identity
                | {"snapshot_entry_ctime_ns": 1, "blob_ctime_ns": 1}
                for identity in identities
            ],
        },
        "corpus": corpus,
        "container": {
            "image_digest": plan["preflight"]["image_digest"],
            "restart_count": 0,
            "tokenizer_inputs_predate_start": True,
        },
        "tokenization": {
            "contract": TOOL.TOKENIZATION_CONTRACT,
            "add_special_tokens": True,
            "return_tensors": "pt",
            "records": 2,
            "total_tokens": 7,
            "minimum_tokens": 3,
            "maximum_tokens": 4,
            "prepared_token_stream_sha256": "e" * 64,
        },
    }
    report = body | {
        "attestation_sha256": hashlib.sha256(
            TOOL._canonical_json(body)
        ).hexdigest()
    }
    path = tmp_path / "attestation.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    evidence = TOOL._validate_tokenizer_evidence(
        plan=plan,
        source=snapshot,
        attestation_path=path,
    )
    assert evidence["mode"] == "legacy-live-container-attestation"
    assert evidence["total_tokens"] == 7

    report["tokenization"]["total_tokens"] = 8
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(TOOL.ArtifactValidationError, match="attestation is invalid"):
        TOOL._validate_tokenizer_evidence(
            plan=plan,
            source=snapshot,
            attestation_path=path,
        )
