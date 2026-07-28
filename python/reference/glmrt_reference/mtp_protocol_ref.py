from __future__ import annotations

import torch


def verify_draft_tokens(target_logits: torch.Tensor, draft_tokens: torch.Tensor) -> tuple[int, bool]:
    """Return accepted prefix length and whether all draft tokens were accepted."""

    predictions = target_logits.argmax(dim=-1)
    accepted = 0
    for pred, draft in zip(predictions.tolist(), draft_tokens.tolist(), strict=False):
        if pred != draft:
            return accepted, False
        accepted += 1
    return accepted, True


def committed_kv_indices(start_pos: int, accepted: int) -> torch.Tensor:
    return torch.arange(start_pos, start_pos + accepted, dtype=torch.long)
