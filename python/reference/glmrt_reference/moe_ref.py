from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ExpertWeights:
    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor


def expert_forward(hidden: torch.Tensor, weights: ExpertWeights) -> torch.Tensor:
    gate = hidden.float() @ weights.gate_proj.float().T
    up = hidden.float() @ weights.up_proj.float().T
    activated = F.silu(gate) * up
    return (activated @ weights.down_proj.float().T).to(dtype=hidden.dtype)


def moe_forward(
    hidden: torch.Tensor,
    expert_weights: dict[int, ExpertWeights],
    route_indices: torch.Tensor,
    route_weights: torch.Tensor,
) -> torch.Tensor:
    output = torch.zeros_like(hidden)
    rows, top_k = route_indices.shape
    for row in range(rows):
        for slot in range(top_k):
            expert_id = int(route_indices[row, slot])
            gate = route_weights[row, slot].to(dtype=hidden.dtype)
            output[row] += gate * expert_forward(hidden[row : row + 1], expert_weights[expert_id]).squeeze(0)
    return output


def partitioned_moe_forward(
    hidden: torch.Tensor,
    expert_weights: dict[int, ExpertWeights],
    route_indices: torch.Tensor,
    route_weights: torch.Tensor,
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
            gate = route_weights[row, slot].to(dtype=hidden.dtype)
            partials[owner][row] += gate * expert_forward(
                hidden[row : row + 1], expert_weights[expert_id]
            ).squeeze(0)
    total = torch.zeros_like(hidden)
    for partial in partials.values():
        total += partial
    return total, partials
