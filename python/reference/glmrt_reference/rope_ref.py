from __future__ import annotations

import torch


def rope_frequencies(dim: int, positions: torch.Tensor, theta: float = 10_000.0) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError("RoPE dimension must be even")
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=positions.device).float() / dim))
    return torch.outer(positions.float(), inv_freq)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, theta: float = 10_000.0) -> torch.Tensor:
    dim = x.shape[-1]
    freqs = rope_frequencies(dim, positions, theta=theta)
    cos = freqs.cos().unsqueeze(0)
    sin = freqs.sin().unsqueeze(0)
    x_even = x[..., 0::2].float()
    x_odd = x[..., 1::2].float()
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    out = torch.empty_like(x.float())
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out.to(dtype=x.dtype)
