from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from glmrt_reference.glm_moe_dsa_layer_ref import (
    GlmMoeDsaConfig,
    glm52_expert_owner,
    glm_moe_dsa_generated_layer_forward,
    glm_router_topk,
    glm_shared_expert_forward,
    glm52_sparse_layer_config,
)
from glmrt_reference.moe_ref import ExpertWeights
from glmrt_reference.rmsnorm_ref import rmsnorm
from glmrt_reference.serve_profiles import find_hf_snapshot

def _tensor_checksum(tensor: torch.Tensor) -> float:
    return float(tensor.float().sum().item())


def _sparse_router_weight(config: GlmMoeDsaConfig) -> torch.Tensor:
    weight = torch.zeros(config.routed_experts, config.hidden_size)
    for expert_id in range(config.routed_experts):
        weight[expert_id, expert_id % config.hidden_size] = 0.03125 + expert_id / 8192.0
        weight[expert_id, (expert_id * 17 + 13) % config.hidden_size] = -0.015625
    return weight


def test_glm52_router_uses_correction_bias_for_selection_and_raw_scores_for_weights():
    config = GlmMoeDsaConfig(
        hidden_size=4,
        routed_experts=4,
        experts_per_token=2,
        moe_intermediate_size=2,
        routed_scaling_factor=2.5,
    )
    hidden = torch.tensor([[1.0, 0.25, -0.5, 0.125]])
    router_weight = torch.tensor(
        [
            [-2.0, 0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
        ]
    )
    correction_bias = torch.tensor([2.0, 0.0, 0.0, 0.0])

    route_indices, route_weights, scores, selection_scores = glm_router_topk(
        hidden,
        router_weight,
        config,
        correction_bias=correction_bias,
    )

    assert route_indices.tolist() == [[0, 1]]
    assert selection_scores[0, 0] > selection_scores[0, 1]
    raw = torch.tensor([scores[0, 0], scores[0, 1]])
    expected = raw / raw.sum() * config.routed_scaling_factor
    torch.testing.assert_close(route_weights[0], expected)
    torch.testing.assert_close(route_weights.sum(dim=-1), torch.tensor([2.5]))


def test_glm52_generated_single_layer_golden_partitions_routed_and_shared_branches():
    config = glm52_sparse_layer_config()
    layer_id = 3
    hidden = torch.linspace(-0.75, 0.75, steps=2 * config.hidden_size).reshape(2, config.hidden_size)
    norm_weight = torch.linspace(0.875, 1.125, steps=config.hidden_size)
    router_weight = _sparse_router_weight(config)
    correction_bias = torch.linspace(-0.125, 0.125, steps=config.routed_experts)

    result = glm_moe_dsa_generated_layer_forward(
        hidden,
        norm_weight,
        router_weight,
        correction_bias,
        config,
        layer_id=layer_id,
    )

    assert result.hidden.shape == (2, 6144)
    assert result.route_indices.shape == (2, 8)
    assert result.route_weights.shape == (2, 8)
    torch.testing.assert_close(result.route_weights.sum(dim=-1), torch.full((2,), 2.5))
    assert set(result.owner_partials) == {"spark-0", "spark-1", "spark-2", "spark-3"}
    torch.testing.assert_close(sum(result.owner_partials.values()), result.routed_moe)
    torch.testing.assert_close(result.hidden, hidden + result.routed_moe + result.shared_mlp)
    assert glm52_expert_owner(layer_id, int(result.route_indices[0, 0])) in result.owner_partials
    assert result.route_indices[0].tolist() == [255, 254, 253, 252, 251, 250, 249, 248]
    assert result.route_indices[1].tolist() == [255, 254, 253, 252, 251, 250, 249, 248]
    assert _tensor_checksum(result.hidden) == pytest.approx(0.17001724243164062, abs=1.0e-6)
    assert _tensor_checksum(result.routed_moe) == pytest.approx(0.12115321308374405, abs=1.0e-6)
    assert _tensor_checksum(result.shared_mlp) == pytest.approx(0.04888710379600525, abs=1.0e-6)


def _load_real_layer3_tensors():
    safetensors = pytest.importorskip("safetensors")
    snapshot = find_hf_snapshot("lukealonso/GLM-5.2-NVFP4")
    if snapshot is None:
        pytest.skip("GLM-5.2 snapshot is not available")
    index_path = snapshot / "model.safetensors.index.json"
    if not index_path.is_file():
        pytest.skip("GLM-5.2 safetensors index is not available")
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]

    def tensor_name(name: str) -> tuple[Path, str]:
        filename = weight_map.get(name)
        if filename is None:
            pytest.skip(f"missing tensor {name}")
        return snapshot / filename, name

    def load(name: str) -> torch.Tensor:
        file_path, tensor = tensor_name(name)
        with safetensors.safe_open(file_path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(tensor)

    return {
        "norm": load("model.layers.3.post_attention_layernorm.weight"),
        "router": load("model.layers.3.mlp.gate.weight"),
        "router_bias": load("model.layers.3.mlp.gate.e_score_correction_bias"),
        "shared_gate": load("model.layers.3.mlp.shared_experts.gate_proj.weight"),
        "shared_up": load("model.layers.3.mlp.shared_experts.up_proj.weight"),
        "shared_down": load("model.layers.3.mlp.shared_experts.down_proj.weight"),
    }


def test_real_glm52_layer3_router_and_shared_expert_golden_from_checkpoint():
    config = glm52_sparse_layer_config()
    tensors = _load_real_layer3_tensors()
    hidden = torch.linspace(-0.01, 0.01, steps=config.hidden_size).reshape(1, config.hidden_size)
    normalized = rmsnorm(hidden, tensors["norm"])

    route_indices, route_weights, _, _ = glm_router_topk(
        normalized,
        tensors["router"],
        config,
        correction_bias=tensors["router_bias"],
    )
    shared = glm_shared_expert_forward(
        normalized,
        ExpertWeights(
            gate_proj=tensors["shared_gate"],
            up_proj=tensors["shared_up"],
            down_proj=tensors["shared_down"],
        ),
    )

    assert route_indices.shape == (1, 8)
    assert route_weights.shape == (1, 8)
    assert route_indices.max() < config.routed_experts
    torch.testing.assert_close(route_weights.sum(dim=-1), torch.tensor([2.5]), rtol=1.0e-6, atol=1.0e-6)
    assert shared.shape == hidden.shape
    assert config.routed_scaling_factor == pytest.approx(2.5)
    assert route_indices.shape[-1] == config.experts_per_token
    assert torch.isfinite(route_weights).all()
    assert torch.isfinite(shared).all()
