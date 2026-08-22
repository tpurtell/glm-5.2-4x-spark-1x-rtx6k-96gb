from __future__ import annotations

import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/tools"))

from build_dspark_route_reference import (  # noqa: E402
    load_width_distribution,
    sample_request_shapes,
)


def test_joint_width_reference_conditions_on_total_m_and_semantics(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "c1.jsonl"
    corpus.write_text(
        "\n".join(
            [
                json.dumps({"case": "code", "draft_lengths": [1] * 100}),
                json.dumps({"case": "math", "draft_lengths": [7] * 100}),
            ]
        )
        + "\n"
    )
    semantic_cases = ["code", "math"]
    width_weights, case_weights, evidence = load_width_distribution(
        corpus, semantic_cases
    )

    shapes = sample_request_shapes(
        random.Random(7),
        requests=2,
        target_rows=10,
        semantic_cases=semantic_cases,
        width_weights=width_weights,
        case_weights=case_weights,
    )
    assert sorted(width for _, width in shapes) == [2, 8]
    assert {width: case for case, width in shapes} == {2: "code", 8: "math"}
    assert evidence["observations"] == 200
    assert evidence["width_counts"]["1"] == 0

    no_draft = sample_request_shapes(
        random.Random(11),
        requests=4,
        target_rows=4,
        semantic_cases=semantic_cases,
        width_weights=width_weights,
        case_weights=case_weights,
    )
    assert [width for _, width in no_draft] == [1, 1, 1, 1]
