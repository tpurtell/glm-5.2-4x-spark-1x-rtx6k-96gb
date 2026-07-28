import pytest

torch = pytest.importorskip("torch")

from glmrt_reference.layer_ref import tiny_attention_block, tiny_moe_layer
from glmrt_reference.moe_ref import ExpertWeights


def test_tiny_attention_block_preserves_hidden_shape():
    torch.manual_seed(1)
    hidden = torch.randn(3, 8)
    weight = torch.ones(8)
    proj = torch.eye(8)
    out = tiny_attention_block(hidden, weight, proj, proj, proj, proj)
    assert out.shape == hidden.shape


def test_tiny_moe_layer_returns_routes_and_hidden_shape():
    torch.manual_seed(2)
    hidden = torch.randn(4, 6)
    norm_weight = torch.ones(6)
    router = torch.randn(3, 6)
    experts = {
        idx: ExpertWeights(
            gate_proj=torch.randn(5, 6),
            up_proj=torch.randn(5, 6),
            down_proj=torch.randn(6, 5),
        )
        for idx in range(3)
    }
    out, indices, weights = tiny_moe_layer(hidden, norm_weight, router, experts, top_k=2)
    assert out.shape == hidden.shape
    assert indices.shape == (4, 2)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(4))
