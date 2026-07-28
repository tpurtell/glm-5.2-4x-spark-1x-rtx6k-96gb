from __future__ import annotations

import torch

from .attention_ref import scaled_dot_product_attention
from .moe_ref import ExpertWeights, moe_forward
from .rmsnorm_ref import rmsnorm
from .router_ref import route_topk


def tiny_attention_block(
    hidden: torch.Tensor,
    norm_weight: torch.Tensor,
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
    o_proj: torch.Tensor,
) -> torch.Tensor:
    x = rmsnorm(hidden, norm_weight)
    q = x.float() @ q_proj.float().T
    k = x.float() @ k_proj.float().T
    v = x.float() @ v_proj.float().T
    attended = scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), causal=True).squeeze(0)
    return hidden + (attended @ o_proj.float().T).to(dtype=hidden.dtype)


def tiny_moe_layer(
    hidden: torch.Tensor,
    norm_weight: torch.Tensor,
    router_weight: torch.Tensor,
    expert_weights: dict[int, ExpertWeights],
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = rmsnorm(hidden, norm_weight)
    route_indices, route_weights = route_topk(x, router_weight, top_k=top_k)
    return hidden + moe_forward(x, expert_weights, route_indices, route_weights), route_indices, route_weights
