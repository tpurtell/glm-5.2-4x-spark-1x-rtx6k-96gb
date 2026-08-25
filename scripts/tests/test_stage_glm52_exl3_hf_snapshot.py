from __future__ import annotations

import hashlib
import importlib.util
import json
import shlex
import subprocess
import sys
import threading
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
TOOLS = ROOT / "python" / "tools"
sys.path.insert(0, str(TOOLS))
TOOL_PATH = TOOLS / "stage_glm52_exl3_hf_snapshot.py"
SPEC = importlib.util.spec_from_file_location("_glmrt_exl3_snapshot_stager", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)
SYNC_PATH = TOOLS / "sync_glm52_exl3_hf_snapshot.py"
SYNC_SPEC = importlib.util.spec_from_file_location("_glmrt_exl3_snapshot_sync", SYNC_PATH)
assert SYNC_SPEC is not None and SYNC_SPEC.loader is not None
SYNC = importlib.util.module_from_spec(SYNC_SPEC)
sys.modules[SYNC_SPEC.name] = SYNC
SYNC_SPEC.loader.exec_module(SYNC)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bound(value: dict, field: str) -> dict:
    return value | {field: hashlib.sha256(TOOL._canonical_json(value)).hexdigest()}


def make_quant_evidence(tmp_path: Path, plan_sha256: str) -> Path:
    path = tmp_path / "quant-evidence.json"
    report = bound(
        {
            "schema": TOOL.QUANT_EVIDENCE_SCHEMA,
            "status": "accepted",
            "quality_scope": (
                "projection-quantizer-evidence-not-end-to-end-model-quality"
            ),
            "plan": {"plan_sha256": plan_sha256},
            "coverage": {
                "expected_projection_count": TOOL.EXPECTED_PROJECTIONS,
                "projection_count": TOOL.EXPECTED_PROJECTIONS,
                "expected_expert_count": 75 * 256,
                "observed_expert_count": 75 * 256,
                "complete_expert_count": 75 * 256,
                "recovered_expert_count": 0,
                "layers": list(range(3, 78)),
            },
            "integrity": {
                "tensor_payload_hashes_verified": True,
                "journal_record_count": TOOL.EXPECTED_PROJECTIONS,
                "checkpoint_inventory_sha256": "e" * 64,
            },
            "metrics": {
                "global": {"aggregate_hessian_weighted_relative_error": 0.003}
            },
        },
        "report_sha256",
    )
    path.write_bytes(TOOL._canonical_json(report) + b"\n")
    return path


def make_candidate(tmp_path: Path) -> tuple[Path, Path, Path]:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    tensor = artifact / "model.safetensors"
    tensor.write_bytes(b"accepted-exl3-tensors")
    plan = artifact / "glmrt-gptqmodel-plan.json"
    plan.write_text("{}\n", encoding="utf-8")
    records = {
        plan.name: {"bytes": plan.stat().st_size, "sha256": sha256(plan)},
        tensor.name: {"bytes": tensor.stat().st_size, "sha256": sha256(tensor)},
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
    report = tmp_path / "validation.json"
    report_body = {
        "schema": TOOL.VALIDATION_SCHEMA,
        "status": "accepted",
        "model_id": TOOL.MODEL_ID,
        "artifact": str(artifact.resolve()),
        "artifact_manifest_sha256": "a" * 64,
        "plan_sha256": "b" * 64,
        "retained_native_bytes_verified": True,
        "artifact_manifest_file_hashes_verified": True,
        "projection_checkpoint_bytes_verified": True,
        "projection_checkpoint": {
            "root": str(tmp_path / "projection-checkpoints"),
            "projection_count": TOOL.EXPECTED_PROJECTIONS,
            "tensor_count": TOOL.EXPECTED_PROJECTIONS * 4,
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
    report.write_bytes(
        TOOL._canonical_json(bound(report_body, "report_sha256")) + b"\n"
    )
    return artifact, report, make_quant_evidence(tmp_path, "b" * 64)


def test_hardlink_stage_uses_standard_blob_snapshot_and_plain_ref(tmp_path: Path) -> None:
    artifact, report, quant_evidence = make_candidate(tmp_path)
    hf_home = tmp_path / "hf"

    staged = TOOL.stage(
        artifact,
        report,
        quant_evidence_report_path=quant_evidence,
        model_id=TOOL.MODEL_ID,
        hf_home=hf_home,
        link_mode="hardlink",
        update_ref=False,
    )

    snapshot = Path(staged["snapshot"])
    tensor_link = snapshot / "model.safetensors"
    tensor_blob = tensor_link.resolve(strict=True)
    assert tensor_link.is_symlink()
    assert tensor_blob.stat().st_ino == (artifact / "model.safetensors").stat().st_ino
    ref = Path(staged["cache_root"]) / "refs" / "main"
    assert ref.read_text(encoding="utf-8") == staged["revision"] + "\n"
    contract = SYNC._local_contract(hf_home, TOOL.MODEL_ID)
    assert contract.revision == staged["revision"]
    assert contract.files == staged["files"]
    assert contract.bytes == staged["bytes"]


def test_stage_ref_move_requires_explicit_permission(tmp_path: Path) -> None:
    artifact, report, quant_evidence = make_candidate(tmp_path)
    hf_home = tmp_path / "hf"
    cache = TOOL._model_cache_root(hf_home, TOOL.MODEL_ID)
    (cache / "refs").mkdir(parents=True)
    (cache / "refs" / "main").write_text("old-revision\n", encoding="utf-8")

    with pytest.raises(TOOL.StagingError, match="--update-ref"):
        TOOL.stage(
            artifact,
            report,
            quant_evidence_report_path=quant_evidence,
            model_id=TOOL.MODEL_ID,
            hf_home=hf_home,
            link_mode="hardlink",
            update_ref=False,
        )


def test_stage_rejects_validation_without_tokenizer_evidence(tmp_path: Path) -> None:
    artifact, report, quant_evidence = make_candidate(tmp_path)
    value = json.loads(report.read_text(encoding="utf-8"))
    value.pop("tokenizer_evidence")
    report.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(TOOL.StagingError, match="does not accept"):
        TOOL.stage(
            artifact,
            report,
            quant_evidence_report_path=quant_evidence,
            model_id=TOOL.MODEL_ID,
            hf_home=tmp_path / "hf",
            link_mode="hardlink",
            update_ref=False,
        )


def test_stage_rejects_tampered_quant_evidence(tmp_path: Path) -> None:
    artifact, report, quant_evidence = make_candidate(tmp_path)
    value = json.loads(quant_evidence.read_text(encoding="utf-8"))
    value["coverage"]["projection_count"] -= 1
    quant_evidence.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(TOOL.StagingError, match="quant-evidence"):
        TOOL.stage(
            artifact,
            report,
            quant_evidence_report_path=quant_evidence,
            model_id=TOOL.MODEL_ID,
            hf_home=tmp_path / "hf",
            link_mode="hardlink",
            update_ref=False,
        )


def test_sync_host_list_is_unique_and_safe() -> None:
    assert SYNC._hosts("ostrich,dodo,emu,kiwi") == (
        "ostrich",
        "dodo",
        "emu",
        "kiwi",
    )
    with pytest.raises(ValueError, match="unique"):
        SYNC._hosts("ostrich,ostrich")
    with pytest.raises(ValueError, match="unsafe"):
        SYNC._hosts("ostrich,bad/host")


def test_remote_hf_home_preserves_python_source_as_one_ssh_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []

    def fake_run(command, **_kwargs):
        observed.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="/home/tj/.cache/huggingface\n",
        )

    monkeypatch.setattr(SYNC.subprocess, "run", fake_run)

    assert SYNC._remote_hf_home("ostrich") == Path(
        "/home/tj/.cache/huggingface"
    )
    assert observed == [
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "ostrich",
            shlex.join(
                [
                    "python3",
                    "-c",
                    "import os,pathlib; print(pathlib.Path(os.environ.get('HF_HOME', pathlib.Path.home()/'.cache'/'huggingface')).expanduser().resolve())",
                ]
            ),
        ]
    ]


def test_remote_verifier_hashes_the_exact_staged_payload(tmp_path: Path) -> None:
    artifact, report, quant_evidence = make_candidate(tmp_path)
    hf_home = tmp_path / "hf"
    staged = TOOL.stage(
        artifact,
        report,
        quant_evidence_report_path=quant_evidence,
        model_id=TOOL.MODEL_ID,
        hf_home=hf_home,
        link_mode="hardlink",
        update_ref=False,
    )
    command = [
        sys.executable,
        "-c",
        SYNC.REMOTE_VERIFY,
        staged["cache_root"],
        staged["revision"],
        TOOL.MODEL_ID,
        "1",
    ]

    accepted = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert accepted.returncode == 0, accepted.stderr
    verified = json.loads(accepted.stdout)
    assert verified["revision"] == staged["revision"]
    assert verified["bytes"] == staged["bytes"]

    tensor_blob = (Path(staged["snapshot"]) / "model.safetensors").resolve()
    original = tensor_blob.read_bytes()
    corrupted = original.replace(b"accepted", b"rejected", 1)
    assert len(corrupted) == len(original) and corrupted != original
    tensor_blob.write_bytes(corrupted)
    rejected = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert rejected.returncode != 0
    assert "remote blob hash mismatch" in rejected.stderr


def test_sync_fans_out_all_hosts_concurrently_and_sorts_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hosts = ("ostrich", "dodo", "emu", "kiwi")
    contract = SYNC.LocalContract(
        root=tmp_path,
        revision="a" * 64,
        files=17,
        bytes=123_456,
    )
    barrier = threading.Barrier(len(hosts))
    observed: list[str] = []
    observed_lock = threading.Lock()

    monkeypatch.setattr(SYNC, "_local_contract", lambda *_args: contract)

    def fake_sync_host(host, actual_contract, *, model_id, verify_hashes):
        assert actual_contract == contract
        assert model_id == TOOL.MODEL_ID
        assert verify_hashes is True
        barrier.wait(timeout=2.0)
        with observed_lock:
            observed.append(host)
        return {
            "host": host,
            "revision": contract.revision,
            "files": contract.files,
            "bytes": contract.bytes,
            "verified_blobs": contract.files,
        }

    monkeypatch.setattr(SYNC, "_sync_host", fake_sync_host)
    result = SYNC.sync(
        model_id=TOOL.MODEL_ID,
        hf_home=tmp_path / "hf",
        hosts=hosts,
        verify_hashes=True,
    )

    assert set(observed) == set(hosts)
    assert [entry["host"] for entry in result["hosts"]] == sorted(hosts)
    assert result["remote_payload_hashes_verified"] is True
