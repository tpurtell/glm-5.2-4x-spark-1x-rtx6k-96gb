from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .moe_ref import ExpertWeights, expert_forward
from .rmsnorm_ref import rmsnorm

DEFAULT_EXPERT_HOSTS = ("spark-0", "spark-1", "spark-2", "spark-3")


@dataclass(frozen=True)
class GlmMoeDsaConfig:
    hidden_size: int = 6144
    routed_experts: int = 256
    experts_per_token: int = 8
    moe_intermediate_size: int = 2048
    scoring_func: str = "sigmoid"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 2.5


@dataclass(frozen=True)
class GeneratedExpertWeights:
    expert_id: int
    hidden_size: int = 6144
    intermediate_size: int = 2048
    gate_scale: float = 0.03125
    up_scale: float = 0.015625
    down_scale: float = 0.0625


@dataclass(frozen=True)
class GlmMoeDsaLayerResult:
    hidden: torch.Tensor
    normalized: torch.Tensor
    route_indices: torch.Tensor
    route_weights: torch.Tensor
    routed_moe: torch.Tensor
    shared_mlp: torch.Tensor
    owner_partials: dict[str, torch.Tensor]


def glm52_sparse_layer_config() -> GlmMoeDsaConfig:
    return GlmMoeDsaConfig()


def glm52_expert_owner(
    layer_id: int,
    expert_id: int,
    hosts: Sequence[str] = DEFAULT_EXPERT_HOSTS,
    routed_experts: int = 256,
) -> str:
    if not hosts:
        raise ValueError("hosts must not be empty")
    return hosts[(layer_id * routed_experts + expert_id) % len(hosts)]


def glm_router_topk(
    hidden: torch.Tensor,
    router_weight: torch.Tensor,
    config: GlmMoeDsaConfig,
    correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = hidden.float() @ router_weight.float().T
    if config.scoring_func == "sigmoid":
        scores = torch.sigmoid(logits)
    else:
        raise ValueError(f"unsupported GLM router scoring function: {config.scoring_func}")

    selection_scores = scores
    if correction_bias is not None:
        selection_scores = selection_scores + correction_bias.float().reshape(1, -1)

    _, route_indices = torch.topk(selection_scores, k=config.experts_per_token, dim=-1)
    route_weights = torch.gather(scores, dim=-1, index=route_indices)
    if config.norm_topk_prob:
        route_weights = route_weights / route_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    route_weights = route_weights * config.routed_scaling_factor
    return route_indices, route_weights, scores, selection_scores


def generated_expert_forward(hidden: torch.Tensor, weights: GeneratedExpertWeights) -> torch.Tensor:
    if hidden.shape[-1] != weights.hidden_size:
        raise ValueError(f"hidden size {hidden.shape[-1]} does not match {weights.hidden_size}")
    device = hidden.device
    idx = torch.arange(weights.intermediate_size, device=device)
    hidden_size = weights.hidden_size
    expert_offset = weights.expert_id % hidden_size
    gate_cols = (idx * 3 + expert_offset) % hidden_size
    up_cols = (idx * 5 + expert_offset * 7) % hidden_size
    out_cols = (idx * 7 + expert_offset * 11) % hidden_size

    gate = hidden.float().index_select(dim=-1, index=gate_cols) * weights.gate_scale
    up = hidden.float().index_select(dim=-1, index=up_cols) * weights.up_scale
    activated = F.silu(gate) * up
    output = hidden.new_zeros(hidden.shape).float()
    output.index_add_(dim=-1, index=out_cols, source=activated * weights.down_scale)
    return output.to(dtype=hidden.dtype)


def generated_moe_forward(
    hidden: torch.Tensor,
    route_indices: torch.Tensor,
    route_weights: torch.Tensor,
    config: GlmMoeDsaConfig,
) -> torch.Tensor:
    output = torch.zeros_like(hidden)
    rows, top_k = route_indices.shape
    for row in range(rows):
        for slot in range(top_k):
            expert_id = int(route_indices[row, slot])
            expert = GeneratedExpertWeights(
                expert_id=expert_id,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
            )
            output[row] += route_weights[row, slot].to(dtype=hidden.dtype) * generated_expert_forward(
                hidden[row : row + 1], expert
            ).squeeze(0)
    return output


def partitioned_generated_moe_forward(
    hidden: torch.Tensor,
    route_indices: torch.Tensor,
    route_weights: torch.Tensor,
    config: GlmMoeDsaConfig,
    owner_fn: Callable[[int], str],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    partials: dict[str, torch.Tensor] = {}
    rows, top_k = route_indices.shape
    for row in range(rows):
        for slot in range(top_k):
            expert_id = int(route_indices[row, slot])
            owner = owner_fn(expert_id)
            if owner not in partials:
                partials[owner] = torch.zeros_like(hidden)
            expert = GeneratedExpertWeights(
                expert_id=expert_id,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
            )
            partials[owner][row] += route_weights[row, slot].to(dtype=hidden.dtype) * generated_expert_forward(
                hidden[row : row + 1], expert
            ).squeeze(0)

    total = torch.zeros_like(hidden)
    for partial in partials.values():
        total += partial
    return total, partials


def generated_shared_expert_forward(hidden: torch.Tensor, config: GlmMoeDsaConfig) -> torch.Tensor:
    return generated_expert_forward(
        hidden,
        GeneratedExpertWeights(
            expert_id=-1,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
        ),
    )


def glm_moe_dsa_generated_layer_forward(
    hidden: torch.Tensor,
    post_attention_norm_weight: torch.Tensor,
    router_weight: torch.Tensor,
    correction_bias: torch.Tensor | None,
    config: GlmMoeDsaConfig,
    layer_id: int,
    hosts: Sequence[str] = DEFAULT_EXPERT_HOSTS,
    include_shared: bool = True,
) -> GlmMoeDsaLayerResult:
    normalized = rmsnorm(hidden, post_attention_norm_weight)
    route_indices, route_weights, _, _ = glm_router_topk(
        normalized,
        router_weight,
        config,
        correction_bias=correction_bias,
    )
    routed_moe, partials = partitioned_generated_moe_forward(
        normalized,
        route_indices,
        route_weights,
        config,
        owner_fn=lambda expert_id: glm52_expert_owner(
            layer_id,
            expert_id,
            hosts=hosts,
            routed_experts=config.routed_experts,
        ),
    )
    shared_mlp = generated_shared_expert_forward(normalized, config) if include_shared else torch.zeros_like(hidden)
    return GlmMoeDsaLayerResult(
        hidden=hidden + routed_moe + shared_mlp,
        normalized=normalized,
        route_indices=route_indices,
        route_weights=route_weights,
        routed_moe=routed_moe,
        shared_mlp=shared_mlp,
        owner_partials=partials,
    )


def glm_shared_expert_forward(hidden: torch.Tensor, weights: ExpertWeights) -> torch.Tensor:
    return expert_forward(hidden, weights)
