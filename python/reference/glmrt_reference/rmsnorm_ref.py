from __future__ import annotations

import torch


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    y = x.float() * torch.rsqrt(variance + eps)
    return (y * weight.float()).to(dtype=x.dtype)
