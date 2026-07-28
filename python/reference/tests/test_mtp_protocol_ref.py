import pytest

torch = pytest.importorskip("torch")

from glmrt_reference.mtp_protocol_ref import committed_kv_indices, verify_draft_tokens


def test_verify_draft_tokens_accepts_prefix_until_mismatch():
    logits = torch.tensor(
        [
            [0.0, 2.0, 1.0],
            [3.0, 1.0, 0.0],
            [0.0, 1.0, 4.0],
        ]
    )
    accepted, complete = verify_draft_tokens(logits, torch.tensor([1, 0, 1]))
    assert accepted == 2
    assert complete is False


def test_committed_kv_indices_tracks_accepted_prefix():
    indices = committed_kv_indices(start_pos=10, accepted=3)
    torch.testing.assert_close(indices, torch.tensor([10, 11, 12]))
