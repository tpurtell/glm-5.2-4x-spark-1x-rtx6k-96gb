from __future__ import annotations

import math

import torch


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(query.shape[-1])
    scores = query.float() @ key.float().transpose(-2, -1) * scale
    if causal:
        q_len, k_len = scores.shape[-2:]
        mask = torch.ones(q_len, k_len, dtype=torch.bool, device=scores.device).triu(1 + k_len - q_len)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return (probs @ value.float()).to(dtype=query.dtype)
