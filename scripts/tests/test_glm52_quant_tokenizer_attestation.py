from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
TOOL_PATH = ROOT / "python" / "tools" / "attest_glm52_quant_tokenizer.py"
SPEC = importlib.util.spec_from_file_location("_glmrt_tokenizer_attestation", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


class FakeTokenizer:
    def __call__(self, text, *, add_special_tokens, return_tensors):
        assert add_special_tokens is True
        assert return_tensors == "pt"
        return {
            "input_ids": [[1, len(text), 2]],
            "attention_mask": [[1, 1, 1]],
        }


def test_token_stream_identity_binds_order_ids_tokens_and_masks() -> None:
    rows = [("a", "one"), ("b", "three")]
    identity = TOOL.token_stream_identity(FakeTokenizer(), rows)

    digest = hashlib.sha256()
    for identifier, text in rows:
        digest.update(
            TOOL.canonical_json(
                {
                    "id": identifier,
                    "input_ids": [1, len(text), 2],
                    "attention_mask": [1, 1, 1],
                }
            )
            + b"\n"
        )
    assert identity == {
        "contract": TOOL.TOKENIZATION_CONTRACT,
        "add_special_tokens": True,
        "return_tensors": "pt",
        "records": 2,
        "total_tokens": 6,
        "minimum_tokens": 3,
        "maximum_tokens": 3,
        "prepared_token_stream_sha256": digest.hexdigest(),
    }


def test_tokenizer_file_identity_rejects_mutated_sha_blob(tmp_path: Path) -> None:
    model = tmp_path / "models--zai-org--GLM-5.2"
    snapshot = model / "snapshots" / ("a" * 40)
    blobs = model / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    payload = b'{"tokenizer": true}\n'
    digest = hashlib.sha256(payload).hexdigest()
    blob = blobs / digest
    blob.write_bytes(payload)
    (snapshot / "tokenizer.json").symlink_to(Path("../../blobs") / digest)

    identity = TOOL.tokenizer_file_identity(snapshot, "tokenizer.json")
    assert TOOL.tokenizer_identity_core(identity) == {
        "name": "tokenizer.json",
        "bytes": len(payload),
        "sha256": digest,
        "hf_blob_id": digest,
    }

    blob.write_bytes(b"changed\n")
    with pytest.raises(TOOL.AttestationError, match="content changed"):
        TOOL.tokenizer_file_identity(snapshot, "tokenizer.json")


def test_bound_record_rejects_tampering() -> None:
    body = {"schema": "test", "value": 1}
    record = body | {
        "attestation_sha256": hashlib.sha256(TOOL.canonical_json(body)).hexdigest()
    }
    assert TOOL.bound_record(record, "attestation_sha256") == record
    record["value"] = 2
    with pytest.raises(TOOL.AttestationError, match="invalid"):
        TOOL.bound_record(record, "attestation_sha256")
