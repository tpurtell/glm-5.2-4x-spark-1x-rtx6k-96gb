import pytest

torch = pytest.importorskip("torch")

from glmrt_reference.rmsnorm_ref import rmsnorm


def test_rmsnorm_matches_manual_formula():
    x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 0.0, 2.0]], dtype=torch.float32)
    weight = torch.tensor([1.0, 0.5, 2.0], dtype=torch.float32)
    out = rmsnorm(x, weight, eps=1e-6)
    expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * weight
    torch.testing.assert_close(out, expected)
