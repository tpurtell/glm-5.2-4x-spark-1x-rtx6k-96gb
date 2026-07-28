from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShapeConfig:
    hidden_size: int
    num_hidden_layers: int
    first_k_dense_replace: int
    routed_experts: int
    experts_per_token: int
    expert_intermediate_size: int
    attention_heads: int
    head_dim: int


def tiny_config() -> ShapeConfig:
    return ShapeConfig(
        hidden_size=16,
        num_hidden_layers=4,
        first_k_dense_replace=1,
        routed_experts=4,
        experts_per_token=2,
        expert_intermediate_size=8,
        attention_heads=2,
        head_dim=8,
    )


def glm_shape_config() -> ShapeConfig:
    return ShapeConfig(
        hidden_size=6144,
        num_hidden_layers=78,
        first_k_dense_replace=3,
        routed_experts=256,
        experts_per_token=8,
        expert_intermediate_size=2048,
        attention_heads=64,
        head_dim=192,
    )
