from __future__ import annotations

import torch


def route_topk(
    hidden: torch.Tensor,
    router_weight: torch.Tensor,
    top_k: int,
    scoring: str = "sigmoid",
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = hidden.float() @ router_weight.float().T
    if scoring == "sigmoid":
        scores = torch.sigmoid(logits)
    elif scoring == "softmax":
        scores = torch.softmax(logits, dim=-1)
    else:
        raise ValueError(f"unsupported scoring function: {scoring}")
    weights, indices = torch.topk(scores, k=top_k, dim=-1)
    if normalize:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return indices, weights
