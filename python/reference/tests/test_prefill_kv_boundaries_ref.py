import pytest

torch = pytest.importorskip("torch")

from glmrt_reference.prefill_ref import TinyPrefillWeights, tiny_prefill_chunked


def make_identityish_weights() -> TinyPrefillWeights:
    vocab = 16
    max_positions = 32
    hidden = 6
    token_embedding = torch.arange(vocab * hidden, dtype=torch.float32).reshape(vocab, hidden) / 100.0
    position_embedding = torch.arange(max_positions * hidden, dtype=torch.float32).reshape(max_positions, hidden) / 1000.0
    eye = torch.eye(hidden)
    return TinyPrefillWeights(
        token_embedding=token_embedding,
        position_embedding=position_embedding,
        attention_norm=torch.ones(hidden),
        q_proj=eye,
        k_proj=eye,
        v_proj=eye,
        o_proj=eye,
        final_norm=torch.ones(hidden),
        lm_head=token_embedding,
    )


def test_chunk_boundaries_do_not_change_kv_positions():
    weights = make_identityish_weights()
    token_ids = torch.tensor([2, 3, 4, 5, 6, 7, 8], dtype=torch.long)

    one_chunk = tiny_prefill_chunked(token_ids, weights, chunk_size=7, position_offset=5)
    three_chunks = tiny_prefill_chunked(token_ids, weights, chunk_size=3, position_offset=5)

    expected_positions = torch.arange(5, 12, dtype=torch.long)
    torch.testing.assert_close(three_chunks.positions, expected_positions)
    torch.testing.assert_close(three_chunks.positions, one_chunk.positions)
    torch.testing.assert_close(three_chunks.key_cache, one_chunk.key_cache, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(three_chunks.value_cache, one_chunk.value_cache, atol=1e-5, rtol=1e-5)


def test_chunk_size_must_be_positive():
    weights = make_identityish_weights()
    token_ids = torch.tensor([1, 2, 3], dtype=torch.long)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        tiny_prefill_chunked(token_ids, weights, chunk_size=0)
