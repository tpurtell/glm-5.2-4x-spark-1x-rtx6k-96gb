import pytest

torch = pytest.importorskip("torch")

from glmrt_reference.moe_ref import ExpertWeights, moe_forward, partitioned_moe_forward


def make_experts() -> dict[int, ExpertWeights]:
    experts = {}
    for expert_id in range(4):
        torch.manual_seed(expert_id)
        experts[expert_id] = ExpertWeights(
            gate_proj=torch.randn(3, 6),
            up_proj=torch.randn(3, 6),
            down_proj=torch.randn(6, 3),
        )
    return experts


def test_partitioned_moe_partials_sum_to_full_output():
    torch.manual_seed(7)
    hidden = torch.randn(5, 6)
    experts = make_experts()
    route_indices = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 3], [1, 3]])
    route_weights = torch.full((5, 2), 0.5)
    full = moe_forward(hidden, experts, route_indices, route_weights)
    partitioned, partials = partitioned_moe_forward(
        hidden,
        experts,
        route_indices,
        route_weights,
        owner_fn=lambda expert_id: ["ostrich", "dodo", "emu", "kiwi"][expert_id],
    )
    assert set(partials) == {"ostrich", "dodo", "emu", "kiwi"}
    torch.testing.assert_close(partitioned, full)
