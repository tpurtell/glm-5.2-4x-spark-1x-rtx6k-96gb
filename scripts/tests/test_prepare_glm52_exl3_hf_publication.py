from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
TOOLS = ROOT / "python" / "tools"
sys.path.insert(0, str(TOOLS))
import stage_glm52_exl3_hf_snapshot as STAGE  # noqa: E402

TOOL_PATH = TOOLS / "prepare_glm52_exl3_hf_publication.py"
SPEC = importlib.util.spec_from_file_location("_glmrt_exl3_publication", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bound(value: dict, field: str) -> dict:
    return value | {field: hashlib.sha256(STAGE._canonical_json(value)).hexdigest()}


def quant_evidence(path: Path, plan_sha256: str) -> Path:
    report = bound(
        {
            "schema": STAGE.QUANT_EVIDENCE_SCHEMA,
            "status": "accepted",
            "quality_scope": (
                "projection-quantizer-evidence-not-end-to-end-model-quality"
            ),
            "plan": {"plan_sha256": plan_sha256},
            "coverage": {
                "expected_projection_count": STAGE.EXPECTED_PROJECTIONS,
                "projection_count": STAGE.EXPECTED_PROJECTIONS,
                "expected_expert_count": 75 * 256,
                "observed_expert_count": 75 * 256,
                "complete_expert_count": 75 * 256,
                "recovered_expert_count": 0,
                "layers": list(range(3, 78)),
            },
            "integrity": {
                "tensor_payload_hashes_verified": True,
                "journal_record_count": STAGE.EXPECTED_PROJECTIONS,
                "checkpoint_inventory_sha256": "e" * 64,
            },
            "metrics": {
                "global": {"aggregate_hessian_weighted_relative_error": 0.003}
            },
        },
        "report_sha256",
    )
    path.write_bytes(STAGE._canonical_json(report) + b"\n")
    return path


def serving_qualification(
    path: Path,
    *,
    artifact: Path,
    validation: Path,
    quant: Path,
    artifact_manifest_sha256: str,
    plan_sha256: str,
) -> Path:
    library = path.parent / "libglmrt_native.so"
    library.write_bytes(b"test-native-library")
    library_identity = {
        "path": str(library.resolve()),
        "bytes": library.stat().st_size,
        "sha256": digest(library),
    }
    rows = [1, 3, 9, 10, 129, 257, 513, 1025, 2049, 2064]
    native_paths: list[Path] = []
    for tp_rank in range(4):
        native = bound(
            {
                "schema": "glmrt-b12x-exl3-native-validation-v1",
                "status": "accepted",
                "sparkinfer_revision": "3" * 40,
                "native_library": library_identity,
                "device": {
                    "name": "NVIDIA GB10",
                    "compute_capability": "12.1",
                },
                "weight_source": {
                    "kind": "calibrated-projection-checkpoints",
                    "root": str((path.parent / "projection-checkpoints").resolve()),
                    "layer_id": 3,
                    "tp_rank": tp_rank,
                    "tp_world_size": 4,
                    "projection_count": 768,
                    "tensor_bytes": 3_636_464_640,
                    "inventory_sha256": "f" * 64,
                },
                "cases": [
                    {
                        "rows": row,
                        "capacity_rows": (
                            row
                            if row in (9, 257)
                            else 2064
                            if row > 2048
                            else 1 << (row - 1).bit_length()
                        ),
                        "route_block_rows": (
                            8
                            if row <= 128
                            else 16
                            if row <= 257
                            else 32
                            if row <= 512
                            else 48
                            if row <= 1024
                            else 64
                        ),
                        "packed_route_count": row * 8,
                        "fc1_tile": [64, 256],
                        "fc2_tile": [64, 256],
                        "blocks_per_sm": 1,
                        "registers_per_thread": 200,
                        "local_memory_bytes": 0,
                        "relative_l2": 0.0,
                        "cosine": 1.0,
                        "max_abs": 0.0,
                    }
                    for row in rows
                ],
            },
            "report_sha256",
        )
        native_path = path.parent / f"native-tp{tp_rank}.json"
        native_path.write_bytes(STAGE._canonical_json(native) + b"\n")
        native_paths.append(native_path)
    report = bound(
        {
            "schema": TOOL.SERVING_QUALIFICATION_SCHEMA,
            "status": "accepted",
            "model_id": TOOL.MODEL_ID,
            "artifact": str(artifact.resolve()),
            "artifact_manifest_sha256": artifact_manifest_sha256,
            "plan_sha256": plan_sha256,
            "artifact_validation": {"sha256": digest(validation)},
            "quant_evidence": {"sha256": digest(quant)},
            "runtime": {
                "engine_identity": "wip-exl3-qualified-111111111111-222222222222",
                "coordinator_slot_fingerprint": "1" * 64,
                "expert_slot_fingerprint": "2" * 64,
                "sparkinfer_revision": "3" * 40,
                "profile": "balanced",
                "power_limit_w": 400,
                "speculation": "dspark",
            },
            "gates": {
                name: True for name in sorted(TOOL.REQUIRED_SERVING_GATES)
            },
            "failed_gates": [],
            "evidence": {
                "candidate_native_validations": [
                    {
                        "path": str(native_path.resolve()),
                        "bytes": native_path.stat().st_size,
                        "sha256": digest(native_path),
                        "schema": "glmrt-b12x-exl3-native-validation-v1",
                    }
                    for native_path in native_paths
                ]
            },
            "results": {
                "native_kernel": {
                    "tp_ranks": [0, 1, 2, 3],
                    "layer_id": 3,
                    "checkpoint_inventory_sha256": "f" * 64,
                    "native_library": library_identity,
                    "required_rows": rows,
                }
            },
        },
        "report_sha256",
    )
    path.write_bytes(STAGE._canonical_json(report) + b"\n")
    return path


def test_publication_is_standard_only_and_hardlinks_only_weight_shards(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    source = tmp_path / "source"
    artifact.mkdir()
    source.mkdir()
    for name in (".gitattributes", "LICENSE", "chat_template.jinja"):
        (source / name).write_text(f"source {name}\n", encoding="utf-8")
    for name in ("generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        (artifact / name).write_text("{}\n", encoding="utf-8")
    shard = artifact / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"published-exl3-weights")
    (artifact / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": shard.stat().st_size},
                "weight_map": {"tensor": shard.name},
            }
        ),
        encoding="utf-8",
    )
    declaration = {
        "method": "exl3",
        "quant_method": "exl3",
        "format": "exl3",
        "checkpoint_format": "exl3",
        "bits": 3.0,
        "meta": {
            "ds4rt_error_ledger": {"local": "/private/path"},
            "offload_to_disk": True,
            "offload_to_disk_path": "/private/offload",
            "moe_vram_strategy_devices": ["cuda:0", "cuda:1"],
            "quantizer": "pinned",
        },
    }
    (artifact / "config.json").write_text(
        json.dumps({"quantization_config": declaration}), encoding="utf-8"
    )
    (artifact / "quantize_config.json").write_text(
        json.dumps({**declaration, "tensor_storage": {"module": {}}}),
        encoding="utf-8",
    )
    plan = artifact / "glmrt-gptqmodel-plan.json"
    plan.write_text('{"private":"local plan"}\n', encoding="utf-8")
    records = {
        path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
        for path in artifact.iterdir()
        if path.name not in {
            "glmrt-gptqmodel-artifact.json",
            "glmrt-gptqmodel-run.json",
        }
    }
    manifest = {
        "schema": TOOL.ARTIFACT_SCHEMA,
        "manifest_sha256": "a" * 64,
        "files": records,
    }
    (artifact / "glmrt-gptqmodel-artifact.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (artifact / "glmrt-gptqmodel-run.json").write_text("{}\n", encoding="utf-8")
    validation = tmp_path / "validation.json"
    validation_body = {
        "schema": TOOL._validation_evidence.__globals__["VALIDATION_SCHEMA"],
        "status": "accepted",
        "model_id": TOOL.MODEL_ID,
        "artifact": str(artifact.resolve()),
        "source_snapshot": str(source.resolve()),
        "artifact_manifest_sha256": "a" * 64,
        "plan_sha256": "b" * 64,
        "retained_native_bytes_verified": True,
        "artifact_manifest_file_hashes_verified": True,
        "projection_checkpoint_bytes_verified": True,
        "projection_checkpoint": {
            "root": str(tmp_path / "projection-checkpoints"),
            "projection_count": STAGE.EXPECTED_PROJECTIONS,
            "tensor_count": STAGE.EXPECTED_PROJECTIONS * 4,
            "tensor_bytes": 272_734_848_000,
            "checkpoint_inventory_sha256": "e" * 64,
        },
        "tokenizer_evidence": {
            "mode": "plan-bound",
            "tokenizer_files": [
                {"name": "tokenizer.json", "bytes": 1, "sha256": "c" * 64},
                {
                    "name": "tokenizer_config.json",
                    "bytes": 1,
                    "sha256": "d" * 64,
                },
            ],
        },
    }
    validation.write_bytes(
        STAGE._canonical_json(bound(validation_body, "report_sha256")) + b"\n"
    )
    quant = quant_evidence(tmp_path / "quant-evidence.json", "b" * 64)
    serving = serving_qualification(
        tmp_path / "serving-qualification.json",
        artifact=artifact,
        validation=validation,
        quant=quant,
        artifact_manifest_sha256="a" * 64,
        plan_sha256="b" * 64,
    )
    readme = tmp_path / "README.md"
    serving_report_sha256 = json.loads(serving.read_text(encoding="utf-8"))[
        "report_sha256"
    ]
    readme.write_text(
        "---\nlicense: mit\n---\n\n# Calibrated K3\n\n"
        f"Qualification evidence SHA-256: `{serving_report_sha256}`\n",
        encoding="utf-8",
    )
    output = tmp_path / "public"

    report = TOOL.prepare(
        artifact,
        source,
        validation,
        quant,
        serving,
        readme,
        output,
        link_mode="hardlink",
    )

    assert report["status"] == "ready"
    assert {path.name for path in output.iterdir()} == set(TOOL.PUBLIC_METADATA) | {
        shard.name
    }
    attributes = (output / ".gitattributes").read_text(encoding="utf-8")
    for name in TOOL.HUB_LFS_ATTRIBUTE_PATHS:
        assert f"{name} filter=lfs diff=lfs merge=lfs -text" in attributes
    assert (output / shard.name).stat().st_ino == shard.stat().st_ino
    assert not (output / plan.name).exists()
    public_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    public_external = json.loads(
        (output / "quantize_config.json").read_text(encoding="utf-8")
    )
    assert "tensor_storage" not in public_config["quantization_config"]
    assert public_external["tensor_storage"] == {"module": {}}
    assert "ds4rt_error_ledger" not in public_external["meta"]
    assert not TOOL.PRIVATE_EXECUTION_META.intersection(public_external["meta"])
    assert public_external["meta"] == {"quantizer": "pinned"}
    assert report["plan_sha256"] == "b" * 64
    assert report["source_quant_evidence_sha256"] == digest(quant)
    assert report["source_serving_qualification_sha256"] == digest(serving)

    forged_native = json.loads(serving.read_text(encoding="utf-8"))
    forged_native.pop("report_sha256")
    forged_native["results"]["native_kernel"]["tp_ranks"] = [0, 1, 2, 2]
    forged_path = tmp_path / "serving-forged-native.json"
    forged_path.write_bytes(
        STAGE._canonical_json(bound(forged_native, "report_sha256")) + b"\n"
    )
    with pytest.raises(TOOL.PublicationError, match="unverifiable native EXL3"):
        TOOL._serving_qualification(
            forged_path,
            artifact=artifact.resolve(),
            artifact_manifest_sha256="a" * 64,
            plan_sha256="b" * 64,
            validation_sha256=digest(validation),
            quant_evidence_sha256=digest(quant),
            projection_checkpoint_root=(tmp_path / "projection-checkpoints").resolve(),
        )

    with pytest.raises(TOOL.PublicationError, match="source snapshot differs"):
        TOOL._validated_source_snapshot(
            json.loads(validation.read_text(encoding="utf-8")),
            tmp_path / "different-source",
        )

    publication_report = tmp_path / "publication.json"
    publication_report.write_text(json.dumps(report), encoding="utf-8")
    staged = STAGE.stage(
        output,
        None,
        publication_report_path=publication_report,
        model_id=TOOL.MODEL_ID,
        hf_home=tmp_path / "hf",
        link_mode="hardlink",
        update_ref=False,
    )
    assert (
        Path(staged["snapshot"]).joinpath(shard.name).resolve().stat().st_ino
        == shard.stat().st_ino
    )

    tampered = dict(report)
    tampered["plan_sha256"] = "c" * 64
    tampered_report = tmp_path / "publication-tampered.json"
    tampered_report.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(STAGE.StagingError, match="does not bind"):
        STAGE.stage(
            output,
            None,
            publication_report_path=tampered_report,
            model_id=TOOL.MODEL_ID,
            hf_home=tmp_path / "tampered-hf",
            link_mode="hardlink",
            update_ref=False,
        )

    rejected = json.loads(serving.read_text(encoding="utf-8"))
    rejected.pop("report_sha256")
    rejected["gates"]["tool_eval_points"] = False
    serving.write_bytes(
        STAGE._canonical_json(bound(rejected, "report_sha256")) + b"\n"
    )
    with pytest.raises(TOOL.PublicationError, match="does not accept"):
        TOOL._serving_qualification(
            serving,
            artifact=artifact.resolve(),
            artifact_manifest_sha256="a" * 64,
            plan_sha256="b" * 64,
            validation_sha256=digest(validation),
            quant_evidence_sha256=digest(quant),
            projection_checkpoint_root=(tmp_path / "projection-checkpoints").resolve(),
        )

    incomplete = json.loads(serving.read_text(encoding="utf-8"))
    incomplete.pop("report_sha256")
    incomplete["gates"] = {"blended_decode": True}
    incomplete["failed_gates"] = []
    serving.write_bytes(
        STAGE._canonical_json(bound(incomplete, "report_sha256")) + b"\n"
    )
    with pytest.raises(TOOL.PublicationError, match="does not accept"):
        TOOL._serving_qualification(
            serving,
            artifact=artifact.resolve(),
            artifact_manifest_sha256="a" * 64,
            plan_sha256="b" * 64,
            validation_sha256=digest(validation),
            quant_evidence_sha256=digest(quant),
            projection_checkpoint_root=(tmp_path / "projection-checkpoints").resolve(),
        )
