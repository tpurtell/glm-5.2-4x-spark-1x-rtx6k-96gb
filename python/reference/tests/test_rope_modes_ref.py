import pytest

torch = pytest.importorskip("torch")

from glmrt_reference.rope_ref import apply_rope


def test_rope_position_zero_is_identity():
    x = torch.randn(1, 1, 8)
    out = apply_rope(x, torch.tensor([0]))
    torch.testing.assert_close(out, x)


def test_rope_preserves_pairwise_norms():
    x = torch.randn(2, 4, 8)
    out = apply_rope(x, torch.arange(4))
    before = x.reshape(2, 4, 4, 2).float().norm(dim=-1)
    after = out.reshape(2, 4, 4, 2).float().norm(dim=-1)
    torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-5)
