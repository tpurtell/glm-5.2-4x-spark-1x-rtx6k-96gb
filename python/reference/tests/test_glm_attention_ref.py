import pytest

from glmrt_reference.glm_attention_ref import (
    apply_rope_rows,
    bounded_attention_oracle_status,
    bounded_dsa_indexer_attention_fixture,
    bounded_main_mla_attention_fixture,
    causal_attention,
    score_dsa_candidates,
    select_dsa_indices,
    sort_dsa_candidate_scores,
)


def _round_matrix(matrix, digits=9):
    return tuple(tuple(round(value, digits) for value in row) for row in matrix)


def test_bounded_main_mla_attention_fixture_has_executable_causal_math():
    result = bounded_main_mla_attention_fixture()

    assert result.mode == "main_mla_rope"
    assert result.rope_mode == "main_mla"
    assert result.layer_id == 0
    assert result.positions == (0, 1, 2)
    assert result.kv_cache_rows == 3
    assert result.kv_cache_width == 4
    assert result.weights[0] == (1.0, 0.0, 0.0)
    assert result.weights[1][2] == 0.0
    assert _round_matrix(result.output) == (
        (0.25, -0.1, 0.5, 0.0),
        (-0.068633782, 0.192080967, 0.314130294, -0.106211261),
        (0.021285782, 0.177895058, 0.194357707, 0.040745399),
    )
    assert result.output_checksum == pytest.approx(1.4156501637709846, abs=1.0e-12)


def test_bounded_dsa_indexer_attention_fixture_selects_indexer_layers_and_indices():
    result = bounded_dsa_indexer_attention_fixture(layer_id=22)

    assert result.mode == "dsa_indexer_rope"
    assert result.rope_mode == "dsa_indexer"
    assert result.layer_id == 22
    assert result.dsa_selection_query == result.rope_query[-1]
    assert result.dsa_candidate_scores == (
        (6, 0.4623014741993756),
        (10, -0.26118327816228637),
        (14, 0.448223624297716),
        (18, 0.25186666011409375),
        (22, 0.6692369918263625),
        (26, -0.14460267449685055),
    )
    assert result.dsa_score_order == (22, 6, 14, 18, 26, 10)
    assert result.selected_indices == (22, 6, 14)
    assert result.weights[0] == (1.0, 0.0, 0.0)
    assert result.weights[1][2] == 0.0
    for row in result.weights:
        assert sum(row) == pytest.approx(1.0, abs=1.0e-12)
    assert _round_matrix(result.output) == (
        (0.05, 0.4, -0.15),
        (0.317789409, 0.078652709, 0.064231527),
        (0.067700318, 0.082189553, 0.213960845),
    )
    assert result.output_checksum == pytest.approx(1.124524360893057, abs=1.0e-12)


def test_rope_modes_are_separate_for_main_mla_and_dsa_indexer():
    main = bounded_main_mla_attention_fixture()
    dsa = bounded_dsa_indexer_attention_fixture()

    assert main.rope_theta == 10_000.0
    assert dsa.rope_theta == 250_000.0
    assert main.rope_mode != dsa.rope_mode
    assert main.rope_query != dsa.rope_query
    assert main.output_checksum != pytest.approx(dsa.output_checksum)
    assert bounded_attention_oracle_status().startswith("executable-bounded:")


def test_attention_helpers_reject_shape_mismatches():
    with pytest.raises(ValueError, match="positions"):
        apply_rope_rows(((1.0, 2.0),), ())
    with pytest.raises(ValueError, match="query and key widths"):
        causal_attention(((1.0, 2.0),), ((1.0,),), ((0.0,),))
    with pytest.raises(ValueError, match="top_k"):
        select_dsa_indices(((1.0, 0.0),), ((1.0, 0.0),), (22,), top_k=2)


def test_dsa_candidate_scoring_is_stable_and_tie_breaks_by_candidate_id():
    scored = score_dsa_candidates(
        ((1.0, 0.0),),
        ((0.5, 0.0), (0.5, 0.0), (0.25, 0.0)),
        (18, 6, 10),
    )
    assert scored == ((18, 0.5), (6, 0.5), (10, 0.25))
    assert sort_dsa_candidate_scores(scored) == ((6, 0.5), (18, 0.5), (10, 0.25))
    assert select_dsa_indices(
        ((1.0, 0.0),),
        ((0.5, 0.0), (0.5, 0.0), (0.25, 0.0)),
        (18, 6, 10),
        top_k=2,
    ) == (6, 18)
