from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[2]
TOOLS = ROOT / "python" / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "_verify_glm52_exl3_hub",
    TOOLS / "verify_glm52_exl3_hub_publication.py",
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fixture(tmp_path: Path):
    publication = tmp_path / "publication"
    publication.mkdir()
    payloads = {"README.md": b"abc", "model-00001-of-00001.safetensors": b"12345"}
    for name, data in payloads.items():
        (publication / name).write_bytes(data)
    body = {
        "schema": "glmrt-hf-standard-publication-v3",
        "model_id": TOOL.MODEL_ID,
        "source_artifact_manifest_sha256": "1" * 64,
        "source_validation_sha256": "2" * 64,
        "source_quant_evidence_sha256": "3" * 64,
        "source_serving_qualification_sha256": "4" * 64,
        "plan_sha256": "5" * 64,
        "files": [
            {"path": name, "bytes": len(data), "sha256": digest(data)}
            for name, data in sorted(payloads.items())
        ],
    }
    report = {
        **body,
        "publication_sha256": hashlib.sha256(TOOL._canonical_json(body)).hexdigest(),
        "status": "ready",
        "output": str(publication),
    }
    report_path = tmp_path / "publication.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    revision = "a" * 40
    siblings = [
        SimpleNamespace(path="README.md", size=3, lfs=None),
        SimpleNamespace(
            rfilename="model-00001-of-00001.safetensors",
            size=5,
            lfs=SimpleNamespace(size=5, sha256=digest(payloads["model-00001-of-00001.safetensors"])),
        ),
    ]
    info = SimpleNamespace(
        id=TOOL.MODEL_ID,
        sha=revision,
        private=False,
        gated=False,
        siblings=siblings,
    )
    return report_path, payloads, info


class FakeApi:
    def __init__(self, info):
        self.info = info

    def model_info(self, repo_id, **kwargs):
        assert repo_id == TOOL.MODEL_ID
        assert kwargs["files_metadata"] is True
        return self.info


def downloader(payloads: dict[str, bytes], tmp_path: Path, *, corrupt: bool = False):
    def download(**kwargs):
        data = payloads[kwargs["filename"]]
        if corrupt:
            data = b"bad"
        destination = tmp_path / f"download-{Path(kwargs['filename']).name}"
        destination.write_bytes(data)
        return str(destination)

    return download


def test_accepts_exact_public_revision_with_fresh_metadata_download(tmp_path: Path) -> None:
    report_path, payloads, info = fixture(tmp_path)
    report = TOOL.verify(
        publication_report_path=report_path,
        revision="main",
        api=FakeApi(info),
        downloader=downloader(payloads, tmp_path),
        fresh_download_limit=3,
    )

    assert report["status"] == "accepted"
    assert report["resolved_revision"] == "a" * 40
    assert report["freshly_downloaded"] == ["README.md"]
    assert report["files"][1]["method"] == "lfs-sha256"


def test_rejects_unexpected_remote_file(tmp_path: Path) -> None:
    report_path, payloads, info = fixture(tmp_path)
    info.siblings.append(SimpleNamespace(path="junk.txt", size=1, lfs=None))

    with pytest.raises(TOOL.HubVerificationError, match="inventory differs"):
        TOOL.verify(
            publication_report_path=report_path,
            revision="main",
            api=FakeApi(info),
            downloader=downloader(payloads, tmp_path),
            fresh_download_limit=3,
        )


def test_rejects_corrupt_fresh_metadata_download(tmp_path: Path) -> None:
    report_path, payloads, info = fixture(tmp_path)

    with pytest.raises(TOOL.HubVerificationError, match="fresh Hub download differs"):
        TOOL.verify(
            publication_report_path=report_path,
            revision="main",
            api=FakeApi(info),
            downloader=downloader(payloads, tmp_path, corrupt=True),
            fresh_download_limit=3,
        )


def test_rejects_private_remote_model(tmp_path: Path) -> None:
    report_path, payloads, info = fixture(tmp_path)
    info.private = True

    with pytest.raises(TOOL.HubVerificationError, match="visibility"):
        TOOL.verify(
            publication_report_path=report_path,
            revision="main",
            api=FakeApi(info),
            downloader=downloader(payloads, tmp_path),
            fresh_download_limit=3,
        )
