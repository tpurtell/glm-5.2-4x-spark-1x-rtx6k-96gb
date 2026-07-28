import pytest

torch = pytest.importorskip("torch")

from glmrt_reference.router_ref import route_topk


def test_router_returns_normalized_topk_weights():
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    router = torch.tensor([[2.0, 0.0], [0.0, 2.0], [-1.0, -1.0]])
    indices, weights = route_topk(hidden, router, top_k=2)
    assert indices.shape == (2, 2)
    assert weights.shape == (2, 2)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2))
