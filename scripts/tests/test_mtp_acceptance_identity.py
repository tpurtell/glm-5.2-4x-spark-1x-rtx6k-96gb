from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).parents[2]
TOOL_PATH = ROOT / "python" / "tools" / "bench_real_full_mtp_acceptance.py"
SPEC = importlib.util.spec_from_file_location("_glmrt_mtp_acceptance", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


def test_prompt_contract_is_model_independent_and_content_complete() -> None:
    contract = TOOL.prompt_contract(
        ["code", "multilingual"],
        suite="explicit",
        repeats=5,
        nonce_seed=2026082201,
        max_tokens=None,
    )
    digest = hashlib.sha256(
        json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()

    assert contract == {
        "suite": "explicit",
        "cases": [
            {
                "id": "code",
                "category": "code",
                "prompt": TOOL.CASES["code"].prompt,
                "max_tokens": TOOL.CASES["code"].max_tokens,
            },
            {
                "id": "multilingual",
                "category": "multilingual",
                "prompt": TOOL.CASES["multilingual"].prompt,
                "max_tokens": TOOL.CASES["multilingual"].max_tokens,
            },
        ],
        "repeats": 5,
        "nonce_seed": 2026082201,
        "temperature": 0,
        "quality_contract_version": TOOL.QUALITY_CONTRACT_VERSION,
    }
    assert len(digest) == 64


def test_completion_payload_changes_only_model_for_matched_arms() -> None:
    case = TOOL.CASES["code"]
    prefix = "Qualification nonce 1-0-code. Treat this identifier as irrelevant.\n"
    baseline = json.loads(
        TOOL.completion_payload("baseline/nvfp4", case, prompt_prefix=prefix)
    )
    candidate = json.loads(
        TOOL.completion_payload("candidate/exl3", case, prompt_prefix=prefix)
    )

    assert baseline | {"model": candidate["model"]} == candidate


def test_semantic_contracts_accept_valid_code_json_and_math() -> None:
    code = '''```python
def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping integer intervals."""
    return intervals

assert merge_intervals([]) == []
assert merge_intervals([(1, 2)]) == [(1, 2)]
assert merge_intervals([(1, 2), (2, 3)])
```'''
    structured = json.dumps(
        {
            "path": "src/cache.rs",
            "operation": "replace",
            "line_start": 41,
            "line_end": 47,
            "rationale": "Remove a redundant copy.",
        }
    )

    assert TOOL.validate_case_content("code", code)["quality_contract_passed"]
    assert TOOL.validate_case_content("structured-json", structured)[
        "quality_contract_passed"
    ]
    assert TOOL.validate_case_content(
        "math", "240 × 0.75 = 180; 180 × 1.08 = $194.40."
    )["quality_contract_passed"]


def test_semantic_contract_rejects_non_json_structured_output() -> None:
    result = TOOL.validate_case_content(
        "structured-json", '```json\n{"path":"src/cache.rs"}\n```'
    )

    assert result["quality_contract_passed"] is False
    assert result["quality_contract_issues"]


def test_fable_contract_accepts_an_unlabelled_final_moral() -> None:
    story_words = " ".join(["feather"] * 140)
    result = TOOL.validate_case_content(
        "fable",
        f"{story_words}. Sharing credit lets every teammate shine.",
    )

    assert result["quality_contract_passed"] is True
