from __future__ import annotations

from dataclasses import dataclass

import torch

from .attention_ref import scaled_dot_product_attention
from .rmsnorm_ref import rmsnorm


@dataclass(frozen=True)
class TinyPrefillWeights:
    token_embedding: torch.Tensor
    position_embedding: torch.Tensor
    attention_norm: torch.Tensor
    q_proj: torch.Tensor
    k_proj: torch.Tensor
    v_proj: torch.Tensor
    o_proj: torch.Tensor
    final_norm: torch.Tensor
    lm_head: torch.Tensor


@dataclass(frozen=True)
class PrefillResult:
    logits: torch.Tensor
    hidden: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    positions: torch.Tensor


@dataclass(frozen=True)
class PrefillRow:
    request_id: str
    layer_id: int
    graph_bucket: int
    position: int
    token_id: int


def tiny_prefill(
    token_ids: torch.Tensor,
    weights: TinyPrefillWeights,
    position_offset: int = 0,
) -> PrefillResult:
    empty_kv = torch.empty(0, weights.k_proj.shape[0], dtype=torch.float32, device=token_ids.device)
    return _tiny_prefill_chunk(
        token_ids=token_ids,
        weights=weights,
        position_offset=position_offset,
        past_key=empty_kv,
        past_value=empty_kv,
        past_positions=torch.empty(0, dtype=torch.long, device=token_ids.device),
    )


def tiny_prefill_chunked(
    token_ids: torch.Tensor,
    weights: TinyPrefillWeights,
    chunk_size: int,
    position_offset: int = 0,
) -> PrefillResult:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    chunks: list[PrefillResult] = []
    past_key = torch.empty(0, weights.k_proj.shape[0], dtype=torch.float32, device=token_ids.device)
    past_value = torch.empty_like(past_key)
    past_positions = torch.empty(0, dtype=torch.long, device=token_ids.device)
    for start in range(0, token_ids.numel(), chunk_size):
        chunk = token_ids[start : start + chunk_size]
        result = _tiny_prefill_chunk(
            token_ids=chunk,
            weights=weights,
            position_offset=position_offset + start,
            past_key=past_key,
            past_value=past_value,
            past_positions=past_positions,
        )
        chunks.append(result)
        past_key = result.key_cache
        past_value = result.value_cache
        past_positions = result.positions
    return PrefillResult(
        logits=torch.cat([chunk.logits for chunk in chunks], dim=0),
        hidden=torch.cat([chunk.hidden for chunk in chunks], dim=0),
        key_cache=past_key,
        value_cache=past_value,
        positions=past_positions,
    )


def mix_prefill_rows(rows: list[PrefillRow]) -> list[PrefillRow]:
    if not rows:
        return []
    layer_id = rows[0].layer_id
    graph_bucket = rows[0].graph_bucket
    for row in rows:
        if row.layer_id != layer_id:
            raise ValueError("cannot mix prefill rows from different layers")
        if row.graph_bucket != graph_bucket:
            raise ValueError("cannot mix prefill rows from different graph buckets")
    return sorted(rows, key=lambda row: (row.layer_id, row.graph_bucket, row.request_id, row.position))


def _tiny_prefill_chunk(
    token_ids: torch.Tensor,
    weights: TinyPrefillWeights,
    position_offset: int,
    past_key: torch.Tensor,
    past_value: torch.Tensor,
    past_positions: torch.Tensor,
) -> PrefillResult:
    positions = torch.arange(
        position_offset,
        position_offset + token_ids.numel(),
        dtype=torch.long,
        device=token_ids.device,
    )
    hidden = weights.token_embedding[token_ids] + weights.position_embedding[positions]
    normalized = rmsnorm(hidden, weights.attention_norm)
    query = normalized.float() @ weights.q_proj.float().T
    key = normalized.float() @ weights.k_proj.float().T
    value = normalized.float() @ weights.v_proj.float().T
    all_key = torch.cat([past_key, key], dim=0)
    all_value = torch.cat([past_value, value], dim=0)
    attended = scaled_dot_product_attention(
        query.unsqueeze(0),
        all_key.unsqueeze(0),
        all_value.unsqueeze(0),
        causal=True,
    ).squeeze(0)
    hidden = hidden + (attended @ weights.o_proj.float().T).to(dtype=hidden.dtype)
    logits = rmsnorm(hidden, weights.final_norm).float() @ weights.lm_head.float().T
    return PrefillResult(
        logits=logits,
        hidden=hidden,
        key_cache=all_key,
        value_cache=all_value,
        positions=torch.cat([past_positions, positions], dim=0),
    )
