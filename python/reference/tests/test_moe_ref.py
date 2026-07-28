import pytest

torch = pytest.importorskip("torch")

from glmrt_reference.moe_ref import ExpertWeights, expert_forward, moe_forward


def make_expert(scale: float) -> ExpertWeights:
    eye = torch.eye(4)
    return ExpertWeights(gate_proj=eye * scale, up_proj=eye, down_proj=eye)


def test_moe_single_expert_matches_expert_forward():
    hidden = torch.randn(2, 4)
    experts = {0: make_expert(1.0)}
    route_indices = torch.zeros(2, 1, dtype=torch.long)
    route_weights = torch.ones(2, 1)
    out = moe_forward(hidden, experts, route_indices, route_weights)
    expected = torch.cat([expert_forward(hidden[i : i + 1], experts[0]) for i in range(2)], dim=0)
    torch.testing.assert_close(out, expected)
