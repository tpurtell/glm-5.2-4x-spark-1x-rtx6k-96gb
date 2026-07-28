import pytest

torch = pytest.importorskip("torch")

from glmrt_reference.prefill_ref import (
    PrefillRow,
    TinyPrefillWeights,
    mix_prefill_rows,
    tiny_prefill,
    tiny_prefill_chunked,
)


def make_weights() -> TinyPrefillWeights:
    torch.manual_seed(17)
    vocab = 19
    max_positions = 32
    hidden = 8
    return TinyPrefillWeights(
        token_embedding=torch.randn(vocab, hidden),
        position_embedding=torch.randn(max_positions, hidden) * 0.1,
        attention_norm=torch.ones(hidden),
        q_proj=torch.randn(hidden, hidden) * 0.2,
        k_proj=torch.randn(hidden, hidden) * 0.2,
        v_proj=torch.randn(hidden, hidden) * 0.2,
        o_proj=torch.randn(hidden, hidden) * 0.2,
        final_norm=torch.ones(hidden),
        lm_head=torch.randn(vocab, hidden) * 0.2,
    )


def test_chunked_prefill_matches_unchunked_tiny_logits_and_kv():
    weights = make_weights()
    token_ids = torch.tensor([1, 4, 7, 3, 5, 9, 2, 8], dtype=torch.long)
    unchunked = tiny_prefill(token_ids, weights)

    for chunk_size in [1, 2, 3, 4, 8]:
        chunked = tiny_prefill_chunked(token_ids, weights, chunk_size=chunk_size)
        torch.testing.assert_close(chunked.logits, unchunked.logits, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(chunked.hidden, unchunked.hidden, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(chunked.key_cache, unchunked.key_cache, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(chunked.value_cache, unchunked.value_cache, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(chunked.positions, unchunked.positions)


def test_prefill_rows_mix_for_same_layer_and_bucket_only():
    rows = [
        PrefillRow("req-b", layer_id=3, graph_bucket=16, position=4, token_id=7),
        PrefillRow("req-a", layer_id=3, graph_bucket=16, position=0, token_id=1),
        PrefillRow("req-a", layer_id=3, graph_bucket=16, position=1, token_id=4),
    ]

    mixed = mix_prefill_rows(rows)
    assert [(row.request_id, row.position) for row in mixed] == [
        ("req-a", 0),
        ("req-a", 1),
        ("req-b", 4),
    ]

    with pytest.raises(ValueError, match="different layers"):
        mix_prefill_rows([rows[0], PrefillRow("req-c", 4, 16, 0, 2)])

    with pytest.raises(ValueError, match="different graph buckets"):
        mix_prefill_rows([rows[0], PrefillRow("req-c", 3, 32, 0, 2)])
