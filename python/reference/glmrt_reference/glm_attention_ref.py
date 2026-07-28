from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .glm_dsa_metadata_ref import GlmDsaIndexerMetadata, glm52_dsa_indexer_metadata

Matrix = tuple[tuple[float, ...], ...]
CandidateScores = tuple[tuple[int, float], ...]

PHASE0_ATTENTION_ORACLE_STATUS = (
    "executable-bounded: no-torch tiny GLM attention math covers main MLA and "
    "DSA indexer modes; real-checkpoint full MLA/RoPE comparison pending"
)


@dataclass(frozen=True)
class BoundedGlmAttentionResult:
    mode: str
    rope_mode: str
    rope_theta: float
    layer_id: int
    positions: tuple[int, ...]
    query: Matrix
    key: Matrix
    value: Matrix
    rope_query: Matrix
    rope_key: Matrix
    scores: Matrix
    weights: Matrix
    output: Matrix
    output_checksum: float
    kv_cache_rows: int
    kv_cache_width: int
    selected_indices: tuple[int, ...] = ()
    dsa_selection_query: tuple[float, ...] = ()
    dsa_candidate_scores: CandidateScores = ()
    dsa_score_order: tuple[int, ...] = ()
    phase0_status: str = PHASE0_ATTENTION_ORACLE_STATUS


def _validate_matrix(name: str, rows: Sequence[Sequence[float]]) -> Matrix:
    if not rows:
        raise ValueError(f"{name} must contain at least one row")
    width = len(rows[0])
    if width == 0:
        raise ValueError(f"{name} rows must not be empty")
    converted = tuple(tuple(float(value) for value in row) for row in rows)
    if any(len(row) != width for row in converted):
        raise ValueError(f"{name} rows must have a stable width")
    return converted


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("dot-product inputs must have equal width")
    return sum(a * b for a, b in zip(left, right, strict=True))


def _softmax(row: Sequence[float]) -> tuple[float, ...]:
    finite = [value for value in row if math.isfinite(value)]
    if not finite:
        raise ValueError("softmax row must contain at least one finite score")
    shift = max(finite)
    exponentials = [0.0 if not math.isfinite(value) else math.exp(value - shift) for value in row]
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)


def apply_rope_rows(
    rows: Sequence[Sequence[float]],
    positions: Sequence[int],
    *,
    theta: float = 10_000.0,
) -> Matrix:
    matrix = _validate_matrix("rows", rows)
    if len(matrix) != len(positions):
        raise ValueError("positions must match row count")
    width = len(matrix[0])
    if width % 2 != 0:
        raise ValueError("RoPE rows must have an even width")

    rotated: list[tuple[float, ...]] = []
    for row, position in zip(matrix, positions, strict=True):
        values: list[float] = []
        for pair in range(width // 2):
            angle = float(position) * (theta ** (-2.0 * pair / width))
            cos = math.cos(angle)
            sin = math.sin(angle)
            even = row[pair * 2]
            odd = row[pair * 2 + 1]
            values.extend((even * cos - odd * sin, even * sin + odd * cos))
        rotated.append(tuple(values))
    return tuple(rotated)


def causal_attention(
    query: Sequence[Sequence[float]],
    key: Sequence[Sequence[float]],
    value: Sequence[Sequence[float]],
    *,
    causal: bool = True,
) -> tuple[Matrix, Matrix, Matrix]:
    query_matrix = _validate_matrix("query", query)
    key_matrix = _validate_matrix("key", key)
    value_matrix = _validate_matrix("value", value)
    if len(key_matrix) != len(value_matrix):
        raise ValueError("key and value row counts must match")
    if len(query_matrix[0]) != len(key_matrix[0]):
        raise ValueError("query and key widths must match")

    scale = 1.0 / math.sqrt(len(query_matrix[0]))
    scores: list[tuple[float, ...]] = []
    weights: list[tuple[float, ...]] = []
    output: list[tuple[float, ...]] = []
    for query_index, query_row in enumerate(query_matrix):
        score_row = tuple(
            float("-inf") if causal and key_index > query_index else _dot(query_row, key_row) * scale
            for key_index, key_row in enumerate(key_matrix)
        )
        weight_row = _softmax(score_row)
        output_row = tuple(
            sum(weight * value_row[column] for weight, value_row in zip(weight_row, value_matrix, strict=True))
            for column in range(len(value_matrix[0]))
        )
        scores.append(score_row)
        weights.append(weight_row)
        output.append(output_row)
    return tuple(scores), tuple(weights), tuple(output)


def select_dsa_indices(
    query: Sequence[Sequence[float]],
    candidates: Sequence[Sequence[float]],
    candidate_ids: Sequence[int],
    *,
    top_k: int,
) -> tuple[int, ...]:
    query_matrix = _validate_matrix("query", query)
    candidate_matrix = _validate_matrix("candidates", candidates)
    if len(candidate_matrix) != len(candidate_ids):
        raise ValueError("candidate_ids must match candidate row count")
    if len(query_matrix[0]) != len(candidate_matrix[0]):
        raise ValueError("query and candidate widths must match")
    if top_k <= 0 or top_k > len(candidate_matrix):
        raise ValueError("top_k must fit inside candidate count")

    scored = score_dsa_candidates(query_matrix, candidate_matrix, candidate_ids)
    return tuple(candidate_id for candidate_id, _ in sort_dsa_candidate_scores(scored)[:top_k])


def score_dsa_candidates(
    query: Sequence[Sequence[float]],
    candidates: Sequence[Sequence[float]],
    candidate_ids: Sequence[int],
) -> CandidateScores:
    query_matrix = _validate_matrix("query", query)
    candidate_matrix = _validate_matrix("candidates", candidates)
    if len(candidate_matrix) != len(candidate_ids):
        raise ValueError("candidate_ids must match candidate row count")
    if len(query_matrix[0]) != len(candidate_matrix[0]):
        raise ValueError("query and candidate widths must match")
    return tuple(
        (
            int(candidate_id),
            sum(_dot(query_row, candidate) for query_row in query_matrix),
        )
        for candidate_id, candidate in zip(candidate_ids, candidate_matrix, strict=True)
    )


def sort_dsa_candidate_scores(scored: Sequence[tuple[int, float]]) -> CandidateScores:
    return tuple(sorted(scored, key=lambda item: (-item[1], item[0])))


def _build_attention_result(
    *,
    mode: str,
    rope_mode: str,
    rope_theta: float,
    layer_id: int,
    positions: tuple[int, ...],
    query: Matrix,
    key: Matrix,
    value: Matrix,
    selected_indices: tuple[int, ...] = (),
    dsa_selection_query: tuple[float, ...] = (),
    dsa_candidate_scores: CandidateScores = (),
    dsa_score_order: tuple[int, ...] = (),
) -> BoundedGlmAttentionResult:
    rope_query = apply_rope_rows(query, positions, theta=rope_theta)
    rope_key = apply_rope_rows(key, positions, theta=rope_theta)
    scores, weights, output = causal_attention(rope_query, rope_key, value)
    output_checksum = sum(sum(row) for row in output)
    return BoundedGlmAttentionResult(
        mode=mode,
        rope_mode=rope_mode,
        rope_theta=rope_theta,
        layer_id=layer_id,
        positions=positions,
        query=query,
        key=key,
        value=value,
        rope_query=rope_query,
        rope_key=rope_key,
        scores=scores,
        weights=weights,
        output=output,
        output_checksum=output_checksum,
        kv_cache_rows=len(key),
        kv_cache_width=len(key[0]),
        selected_indices=selected_indices,
        dsa_selection_query=dsa_selection_query,
        dsa_candidate_scores=dsa_candidate_scores,
        dsa_score_order=dsa_score_order,
    )


def bounded_main_mla_attention_fixture() -> BoundedGlmAttentionResult:
    query = _validate_matrix(
        "main_query",
        (
            (-0.20, 0.45, 0.70, -0.10),
            (0.30, -0.15, 0.40, 0.25),
            (0.55, 0.20, -0.35, 0.60),
        ),
    )
    key = _validate_matrix(
        "main_key",
        (
            (0.10, 0.50, -0.20, 0.30),
            (0.35, -0.25, 0.45, 0.15),
            (-0.40, 0.30, 0.20, -0.55),
        ),
    )
    value = _validate_matrix(
        "main_value",
        (
            (0.25, -0.10, 0.50, 0.00),
            (-0.35, 0.45, 0.15, -0.20),
            (0.10, 0.30, -0.25, 0.40),
        ),
    )
    return _build_attention_result(
        mode="main_mla_rope",
        rope_mode="main_mla",
        rope_theta=10_000.0,
        layer_id=0,
        positions=(0, 1, 2),
        query=query,
        key=key,
        value=value,
    )


def bounded_dsa_indexer_attention_fixture(
    *,
    metadata: GlmDsaIndexerMetadata | None = None,
    layer_id: int = 22,
) -> BoundedGlmAttentionResult:
    metadata = metadata or glm52_dsa_indexer_metadata()
    if not metadata.layer_has_dsa_indexer(layer_id):
        raise ValueError(f"layer {layer_id} does not have a DSA indexer")

    query = _validate_matrix(
        "dsa_query",
        (
            (0.42, -0.18, 0.12, 0.36),
            (-0.25, 0.55, 0.33, -0.08),
            (0.60, 0.10, -0.28, 0.45),
        ),
    )
    key = _validate_matrix(
        "dsa_key",
        (
            (0.30, -0.20, 0.50, 0.05),
            (-0.45, 0.15, 0.25, 0.35),
            (0.20, 0.40, -0.30, 0.10),
        ),
    )
    value = _validate_matrix(
        "dsa_value",
        (
            (0.05, 0.40, -0.15),
            (0.55, -0.20, 0.25),
            (-0.35, 0.10, 0.45),
        ),
    )
    positions = (4, 5, 6)
    rope_theta = 250_000.0
    rope_query = apply_rope_rows(query, positions, theta=rope_theta)
    candidates = _validate_matrix(
        "dsa_candidates",
        (
            (0.35, 0.10, -0.20, 0.45),
            (-0.10, 0.30, 0.55, -0.05),
            (0.60, -0.15, 0.05, 0.20),
            (0.20, 0.50, -0.35, 0.15),
            (0.70, 0.05, -0.25, 0.40),
            (-0.30, 0.45, 0.15, 0.25),
        ),
    )
    candidate_ids = (6, 10, 14, 18, 22, 26)
    dsa_selection_query = rope_query[-1]
    candidate_scores = score_dsa_candidates(
        (dsa_selection_query,),
        candidates,
        candidate_ids,
    )
    score_order = tuple(
        candidate_id
        for candidate_id, _ in sort_dsa_candidate_scores(candidate_scores)
    )
    selected_indices = score_order[:3]
    return _build_attention_result(
        mode="dsa_indexer_rope",
        rope_mode="dsa_indexer",
        rope_theta=rope_theta,
        layer_id=layer_id,
        positions=positions,
        query=query,
        key=key,
        value=value,
        selected_indices=selected_indices,
        dsa_selection_query=dsa_selection_query,
        dsa_candidate_scores=candidate_scores,
        dsa_score_order=score_order,
    )


def bounded_attention_oracle_status() -> str:
    return PHASE0_ATTENTION_ORACLE_STATUS


def bounded_attention_oracle_summary() -> dict[str, object]:
    main = bounded_main_mla_attention_fixture()
    dsa = bounded_dsa_indexer_attention_fixture()
    return {
        "status": PHASE0_ATTENTION_ORACLE_STATUS,
        "main_mla_output_checksum": main.output_checksum,
        "dsa_indexer_layer": dsa.layer_id,
        "dsa_output_checksum": dsa.output_checksum,
        "dsa_selected_indices": dsa.selected_indices,
        "covers_real_checkpoint": False,
        "remaining_gap": "bounded synthetic fixtures only; real checkpoint full MLA/RoPE/Rust comparison pending",
    }
