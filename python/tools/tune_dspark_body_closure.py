#!/usr/bin/env python3
"""Benchmark the real-weight dSpark context update and five-layer draft body."""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import flashinfer
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open

from tune_dspark_head_closure import FIXTURES, capture, measure, parse_int_list

HIDDEN = 6_144
HEADS = 64
HEAD_DIM = 64
Q_SIZE = HEADS * HEAD_DIM
LAYERS = 5
TARGET_FEATURES = 5
EPSILON = 1.0e-5


@dataclass
class LayerWeights:
    input_norm: torch.Tensor
    post_norm: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    qkv: torch.Tensor
    output: torch.Tensor
    gate_up: torch.Tensor
    down: torch.Tensor


@dataclass
class BodyWeights:
    target_fusion: torch.Tensor
    hidden_norm: torch.Tensor
    final_norm: torch.Tensor
    layers: list[LayerWeights]
    fused_context_kv: torch.Tensor
    stacked_k_norm: torch.Tensor


@dataclass
class ContextState:
    target_features: torch.Tensor
    fused_hidden: torch.Tensor
    key_output: torch.Tensor
    value_output: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor


@dataclass
class QueryState:
    input: torch.Tensor
    output: torch.Tensor
    context_keys: list[torch.Tensor]
    context_values: list[torch.Tensor]
    attention_queries: list[torch.Tensor]
    attention_output: torch.Tensor
    ragged_keys: list[torch.Tensor]
    ragged_values: list[torch.Tensor]
    flashinfer_output: torch.Tensor
    flashinfer_wrapper: object
    cos: torch.Tensor
    sin: torch.Tensor


def rms_norm(
    source: torch.Tensor, weight: torch.Tensor, epsilon: float = EPSILON
) -> torch.Tensor:
    variance = source.float().pow(2).mean(dim=-1, keepdim=True)
    return source * torch.rsqrt(variance + epsilon).to(source.dtype) * weight


def apply_rope(
    source: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    first, second = source.chunk(2, dim=-1)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


def rope_factors(
    positions: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    exponent = torch.arange(0, HEAD_DIM, 2, device=device, dtype=torch.float32)
    inverse_frequency = 1.0 / (8_000_000.0 ** (exponent / HEAD_DIM))
    angles = positions.float().unsqueeze(-1) * inverse_frequency
    return angles.cos().to(torch.bfloat16), angles.sin().to(torch.bfloat16)


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def load_body_weights(snapshot: Path, device: torch.device) -> BodyWeights:
    with safe_open(
        snapshot / "model.safetensors", framework="pt", device=str(device)
    ) as checkpoint:
        target_fusion = checkpoint.get_tensor("fc.weight")
        hidden_norm = checkpoint.get_tensor("hidden_norm.weight")
        final_norm = checkpoint.get_tensor("norm.weight")
        layers = []
        for layer_index in range(LAYERS):
            prefix = f"layers.{layer_index}"
            qkv = torch.cat(
                [
                    checkpoint.get_tensor(f"{prefix}.self_attn.{name}_proj.weight")
                    for name in ("q", "k", "v")
                ],
                dim=0,
            ).contiguous()
            gate_up = torch.cat(
                [
                    checkpoint.get_tensor(f"{prefix}.mlp.{name}_proj.weight")
                    for name in ("gate", "up")
                ],
                dim=0,
            ).contiguous()
            layers.append(
                LayerWeights(
                    input_norm=checkpoint.get_tensor(
                        f"{prefix}.input_layernorm.weight"
                    ),
                    post_norm=checkpoint.get_tensor(
                        f"{prefix}.post_attention_layernorm.weight"
                    ),
                    q_norm=checkpoint.get_tensor(f"{prefix}.self_attn.q_norm.weight"),
                    k_norm=checkpoint.get_tensor(f"{prefix}.self_attn.k_norm.weight"),
                    qkv=qkv,
                    output=checkpoint.get_tensor(f"{prefix}.self_attn.o_proj.weight"),
                    gate_up=gate_up,
                    down=checkpoint.get_tensor(f"{prefix}.mlp.down_proj.weight"),
                )
            )
    fused_context_kv = torch.cat(
        [layer.qkv[Q_SIZE:] for layer in layers], dim=0
    ).contiguous()
    stacked_k_norm = torch.stack([layer.k_norm for layer in layers], dim=0).contiguous()
    return BodyWeights(
        target_fusion=target_fusion,
        hidden_norm=hidden_norm,
        final_norm=final_norm,
        layers=layers,
        fused_context_kv=fused_context_kv,
        stacked_k_norm=stacked_k_norm,
    )


def body_weight_bytes(weights: BodyWeights) -> int:
    tensors = [weights.target_fusion, weights.hidden_norm, weights.final_norm]
    for layer in weights.layers:
        tensors.extend(
            [
                layer.input_norm,
                layer.post_norm,
                layer.q_norm,
                layer.k_norm,
                layer.qkv,
                layer.output,
                layer.gate_up,
                layer.down,
            ]
        )
    return sum(tensor_bytes(tensor) for tensor in tensors)


def make_context_state(rows: int, device: torch.device, seed: int) -> ContextState:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    positions = torch.arange(rows, device=device, dtype=torch.int64) + 1_024
    cos, sin = rope_factors(positions, device)
    return ContextState(
        target_features=torch.randn(
            rows,
            TARGET_FEATURES * HIDDEN,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        ),
        fused_hidden=torch.empty(rows, HIDDEN, device=device, dtype=torch.bfloat16),
        key_output=torch.empty(
            LAYERS, rows, HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16
        ),
        value_output=torch.empty(
            LAYERS, rows, HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16
        ),
        cos=cos.view(1, rows, 1, HEAD_DIM // 2),
        sin=sin.view(1, rows, 1, HEAD_DIM // 2),
    )


def context_operations(
    state: ContextState, weights: BodyWeights
) -> tuple[
    Callable[[], None], Callable[[], None], Callable[[], None], Callable[[], None]
]:
    rows = state.target_features.shape[0]

    def target_fusion() -> None:
        state.fused_hidden.copy_(
            rms_norm(
                F.linear(state.target_features, weights.target_fusion),
                weights.hidden_norm,
            )
        )

    def fused_kv() -> None:
        projected = F.linear(state.fused_hidden, weights.fused_context_kv)
        all_kv = (
            projected.view(rows, LAYERS, 2, HEADS, HEAD_DIM)
            .permute(2, 1, 0, 3, 4)
            .contiguous()
        )
        keys = all_kv[0]
        values = all_kv[1]
        k_norm_weights = weights.stacked_k_norm.view(LAYERS, 1, 1, HEAD_DIM)
        keys = rms_norm(keys, k_norm_weights)
        keys = apply_rope(keys, state.cos, state.sin)
        state.key_output.copy_(keys)
        state.value_output.copy_(values)

    def split_kv() -> None:
        for layer_index, layer in enumerate(weights.layers):
            projected = F.linear(state.fused_hidden, layer.qkv[Q_SIZE:])
            keys, values = projected.chunk(2, dim=-1)
            keys = keys.view(rows, HEADS, HEAD_DIM)
            values = values.view(rows, HEADS, HEAD_DIM)
            keys = rms_norm(keys, layer.k_norm)
            keys = apply_rope(keys, state.cos[0], state.sin[0])
            state.key_output[layer_index].copy_(keys)
            state.value_output[layer_index].copy_(values)

    def complete_fused_update() -> None:
        target_fusion()
        fused_kv()

    return target_fusion, fused_kv, split_kv, complete_fused_update


def make_query_state(
    concurrency: int,
    block_size: int,
    context_tokens: int,
    device: torch.device,
    seed: int,
    flashinfer_backend: str,
) -> QueryState:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    positions = torch.arange(block_size, device=device, dtype=torch.int64)
    positions += context_tokens
    cos, sin = rope_factors(positions, device)
    total_kv_tokens = context_tokens + block_size

    def random_ragged_cache() -> torch.Tensor:
        return torch.randn(
            concurrency * total_kv_tokens,
            HEADS,
            HEAD_DIM,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )

    ragged_keys = [random_ragged_cache() for _ in range(LAYERS)]
    ragged_values = [random_ragged_cache() for _ in range(LAYERS)]
    context_keys = [
        cache.view(concurrency, total_kv_tokens, HEADS, HEAD_DIM)[:, :context_tokens]
        for cache in ragged_keys
    ]
    context_values = [
        cache.view(concurrency, total_kv_tokens, HEADS, HEAD_DIM)[:, :context_tokens]
        for cache in ragged_values
    ]
    attention_queries = [
        torch.randn(
            concurrency,
            block_size,
            HEADS,
            HEAD_DIM,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        for _ in range(LAYERS)
    ]
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    qo_indptr = (
        torch.arange(concurrency + 1, device=device, dtype=torch.int32) * block_size
    )
    kv_indptr = (
        torch.arange(concurrency + 1, device=device, dtype=torch.int32)
        * total_kv_tokens
    )
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace,
        kv_layout="NHD",
        use_cuda_graph=True,
        qo_indptr_buf=torch.empty_like(qo_indptr),
        kv_indptr_buf=torch.empty_like(kv_indptr),
        backend=flashinfer_backend,
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads=HEADS,
        num_kv_heads=HEADS,
        head_dim_qk=HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        causal=False,
        sm_scale=1.0 / math.sqrt(HEAD_DIM),
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
    )
    return QueryState(
        input=torch.randn(
            concurrency,
            block_size,
            HIDDEN,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        ),
        output=torch.empty(
            concurrency,
            block_size,
            HIDDEN,
            device=device,
            dtype=torch.bfloat16,
        ),
        context_keys=context_keys,
        context_values=context_values,
        attention_queries=attention_queries,
        attention_output=torch.empty(
            LAYERS,
            concurrency,
            block_size,
            HEADS,
            HEAD_DIM,
            device=device,
            dtype=torch.bfloat16,
        ),
        ragged_keys=ragged_keys,
        ragged_values=ragged_values,
        flashinfer_output=torch.empty(
            concurrency * block_size,
            HEADS,
            HEAD_DIM,
            device=device,
            dtype=torch.bfloat16,
        ),
        flashinfer_wrapper=wrapper,
        cos=cos.view(1, block_size, 1, HEAD_DIM // 2),
        sin=sin.view(1, block_size, 1, HEAD_DIM // 2),
    )


def query_operations(state: QueryState, weights: BodyWeights) -> tuple[
    Callable[[], None],
    Callable[[], None],
    Callable[[], None],
    Callable[[], None],
    Callable[[], None],
]:
    concurrency, block_size, _ = state.input.shape
    scale = 1.0 / math.sqrt(HEAD_DIM)
    context_tokens = state.context_keys[0].shape[1]
    total_kv_tokens = context_tokens + block_size

    def execute_body(use_flashinfer: bool) -> None:
        hidden = state.input
        for layer_index, layer in enumerate(weights.layers):
            normalized = rms_norm(hidden, layer.input_norm)
            qkv = F.linear(normalized, layer.qkv)
            query, key, value = qkv.split((Q_SIZE, Q_SIZE, Q_SIZE), dim=-1)
            query = query.view(concurrency, block_size, HEADS, HEAD_DIM)
            key = key.view(concurrency, block_size, HEADS, HEAD_DIM)
            value = value.view(concurrency, block_size, HEADS, HEAD_DIM)
            query = rms_norm(query, layer.q_norm)
            key = rms_norm(key, layer.k_norm)
            query = apply_rope(query, state.cos, state.sin)
            key = apply_rope(key, state.cos, state.sin)
            if use_flashinfer:
                key_cache = state.ragged_keys[layer_index]
                value_cache = state.ragged_values[layer_index]
                key_cache.view(concurrency, total_kv_tokens, HEADS, HEAD_DIM)[
                    :, context_tokens:
                ].copy_(key)
                value_cache.view(concurrency, total_kv_tokens, HEADS, HEAD_DIM)[
                    :, context_tokens:
                ].copy_(value)
                state.flashinfer_wrapper.run(
                    query.reshape(-1, HEADS, HEAD_DIM),
                    key_cache,
                    value_cache,
                    out=state.flashinfer_output,
                )
                attended = state.flashinfer_output.view(concurrency, block_size, Q_SIZE)
            else:
                all_keys = torch.cat((state.context_keys[layer_index], key), dim=1)
                all_values = torch.cat(
                    (state.context_values[layer_index], value), dim=1
                )
                attended = F.scaled_dot_product_attention(
                    query.transpose(1, 2),
                    all_keys.transpose(1, 2),
                    all_values.transpose(1, 2),
                    is_causal=False,
                    scale=scale,
                )
                attended = attended.transpose(1, 2).reshape(
                    concurrency, block_size, Q_SIZE
                )
            hidden = hidden + F.linear(attended, layer.output)
            normalized = rms_norm(hidden, layer.post_norm)
            gate, up = F.linear(normalized, layer.gate_up).chunk(2, dim=-1)
            hidden = hidden + F.linear(F.silu(gate) * up, layer.down)
        state.output.copy_(rms_norm(hidden, weights.final_norm))

    def full_body_torch_sdpa() -> None:
        execute_body(False)

    def full_body_flashinfer() -> None:
        execute_body(True)

    def dense_projection_closure() -> None:
        hidden = state.input
        for layer in weights.layers:
            normalized = rms_norm(hidden, layer.input_norm)
            query = F.linear(normalized, layer.qkv)[..., :Q_SIZE]
            hidden = hidden + F.linear(query, layer.output)
            normalized = rms_norm(hidden, layer.post_norm)
            gate, up = F.linear(normalized, layer.gate_up).chunk(2, dim=-1)
            hidden = hidden + F.linear(F.silu(gate) * up, layer.down)
        state.output.copy_(rms_norm(hidden, weights.final_norm))

    def torch_attention_closure() -> None:
        for layer_index in range(LAYERS):
            keys = state.ragged_keys[layer_index].view(
                concurrency, total_kv_tokens, HEADS, HEAD_DIM
            )
            values = state.ragged_values[layer_index].view(
                concurrency, total_kv_tokens, HEADS, HEAD_DIM
            )
            state.attention_output[layer_index].copy_(
                F.scaled_dot_product_attention(
                    state.attention_queries[layer_index].transpose(1, 2),
                    keys.transpose(1, 2),
                    values.transpose(1, 2),
                    is_causal=False,
                    scale=scale,
                ).transpose(1, 2)
            )

    def flashinfer_attention_closure() -> None:
        for layer_index in range(LAYERS):
            state.flashinfer_wrapper.run(
                state.attention_queries[layer_index].reshape(-1, HEADS, HEAD_DIM),
                state.ragged_keys[layer_index],
                state.ragged_values[layer_index],
                out=state.attention_output[layer_index].view(-1, HEADS, HEAD_DIM),
            )

    return (
        full_body_torch_sdpa,
        full_body_flashinfer,
        dense_projection_closure,
        torch_attention_closure,
        flashinfer_attention_closure,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the off-path real-weight dSpark hidden-fusion/context-KV "
            "update and five-layer 1+N query body."
        )
    )
    parser.add_argument("--fixture", choices=sorted(FIXTURES), default="siro")
    parser.add_argument("--context-rows", default="1,4,8,16")
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--context-tokens", type=int, default=1_024)
    parser.add_argument(
        "--flashinfer-backend",
        choices=("auto", "fa2", "fa3", "cudnn", "cutlass", "cute-dsl"),
        default="auto",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    context_rows = parse_int_list(args.context_rows, "context-rows")
    concurrency_values = parse_int_list(args.concurrency, "concurrency")
    if min(args.context_tokens, args.iterations, args.repeats) < 1 or args.warmup < 0:
        parser.error(
            "context-tokens/iterations/repeats must be positive and warmup nonnegative"
        )

    fixture = FIXTURES[args.fixture]
    snapshot = Path(
        snapshot_download(
            fixture["repo_id"],
            revision=fixture["revision"],
            local_files_only=True,
        )
    )
    config = json.loads((snapshot / "config.json").read_text())
    block_size = int(config["block_size"])
    if block_size != fixture["proposal_tokens"] + 1:
        raise RuntimeError("fixture does not use the expected 1+N bonus-anchor layout")

    torch.manual_seed(args.seed)
    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    weights = load_body_weights(snapshot, device)
    torch.cuda.synchronize()

    context_results = []
    for rows in context_rows:
        state = make_context_state(rows, device, args.seed + rows * 1_000)
        target_fusion, fused_kv, split_kv, complete_fused = context_operations(
            state, weights
        )
        target_fusion()
        fused_kv()
        torch.cuda.synchronize()
        fused_keys = state.key_output.clone()
        fused_values = state.value_output.clone()
        split_kv()
        torch.cuda.synchronize()
        key_difference = (state.key_output - fused_keys).float()
        value_difference = (state.value_output - fused_values).float()
        validation = {
            "key_max_abs": float(key_difference.abs().max()),
            "key_relative_l2": float(
                torch.linalg.vector_norm(key_difference)
                / torch.linalg.vector_norm(fused_keys.float())
            ),
            "value_max_abs": float(value_difference.abs().max()),
            "value_relative_l2": float(
                torch.linalg.vector_norm(value_difference)
                / torch.linalg.vector_norm(fused_values.float())
            ),
        }
        target_graph = capture(target_fusion)
        fused_graph = capture(fused_kv)
        split_graph = capture(split_kv)
        complete_graph = capture(complete_fused)
        timings = {
            "target_hidden_fusion": measure(
                [target_graph.replay], args.warmup, args.iterations, args.repeats
            ),
            "fused_context_kv_from_hidden": measure(
                [fused_graph.replay], args.warmup, args.iterations, args.repeats
            ),
            "split_context_kv_from_hidden": measure(
                [split_graph.replay], args.warmup, args.iterations, args.repeats
            ),
            "two_graph_complete_update": measure(
                [target_graph.replay, fused_graph.replay],
                args.warmup,
                args.iterations,
                args.repeats,
            ),
            "single_graph_complete_update": measure(
                [complete_graph.replay], args.warmup, args.iterations, args.repeats
            ),
        }
        context_result = {
            "benchmark": "dspark_context_update",
            "rows": rows,
            "validation": validation,
            "timings": timings,
            "fused_over_split_speedup": (
                timings["split_context_kv_from_hidden"]["gpu_ms"]["median"]
                / timings["fused_context_kv_from_hidden"]["gpu_ms"]["median"]
            ),
        }
        context_results.append(context_result)
        print(json.dumps(context_result, sort_keys=True), flush=True)
        del (
            state,
            fused_keys,
            fused_values,
            target_graph,
            fused_graph,
            split_graph,
            complete_graph,
        )
        gc.collect()
        torch.cuda.empty_cache()

    query_results = []
    for concurrency in concurrency_values:
        state = make_query_state(
            concurrency,
            block_size,
            args.context_tokens,
            device,
            args.seed + concurrency * 10_000,
            args.flashinfer_backend,
        )
        (
            torch_body,
            flashinfer_body,
            dense_closure,
            torch_attention,
            flashinfer_attention,
        ) = query_operations(state, weights)
        torch_attention()
        torch.cuda.synchronize()
        torch_attention_reference = state.attention_output.clone()
        flashinfer_attention()
        torch.cuda.synchronize()
        attention_difference = (
            state.attention_output - torch_attention_reference
        ).float()
        torch_body()
        torch.cuda.synchronize()
        torch_reference = state.output.clone()
        flashinfer_body()
        torch.cuda.synchronize()
        flashinfer_reference = state.output.clone()
        backend_difference = (flashinfer_reference - torch_reference).float()
        torch_graph = capture(torch_body)
        torch_graph.replay()
        torch.cuda.synchronize()
        torch_graph_difference = (state.output - torch_reference).float()
        flashinfer_graph = capture(flashinfer_body)
        flashinfer_graph.replay()
        torch.cuda.synchronize()
        flashinfer_graph_difference = (state.output - flashinfer_reference).float()
        torch_graph_max_abs = float(torch_graph_difference.abs().max())
        flashinfer_graph_max_abs = float(flashinfer_graph_difference.abs().max())
        validation = {
            "torch_graph_exact": torch_graph_max_abs == 0.0,
            "torch_graph_max_abs": torch_graph_max_abs,
            "flashinfer_graph_exact": bool(
                torch.equal(state.output, flashinfer_reference)
            ),
            "flashinfer_graph_max_abs": flashinfer_graph_max_abs,
            "flashinfer_vs_torch_max_abs": float(backend_difference.abs().max()),
            "flashinfer_vs_torch_relative_l2": float(
                torch.linalg.vector_norm(backend_difference)
                / torch.linalg.vector_norm(torch_reference.float())
            ),
            "flashinfer_attention_vs_torch_max_abs": float(
                attention_difference.abs().max()
            ),
            "flashinfer_attention_vs_torch_relative_l2": float(
                torch.linalg.vector_norm(attention_difference)
                / torch.linalg.vector_norm(torch_attention_reference.float())
            ),
        }
        dense_graph = capture(dense_closure)
        torch_attention_graph = capture(torch_attention)
        flashinfer_attention_graph = capture(flashinfer_attention)
        timings = {
            "torch_sdpa_copy_full_five_layer_body": measure(
                [torch_graph.replay], args.warmup, args.iterations, args.repeats
            ),
            "flashinfer_ragged_full_five_layer_body": measure(
                [flashinfer_graph.replay],
                args.warmup,
                args.iterations,
                args.repeats,
            ),
            "dense_projection_closure": measure(
                [dense_graph.replay], args.warmup, args.iterations, args.repeats
            ),
            "torch_sdpa_five_attention_calls": measure(
                [torch_attention_graph.replay],
                args.warmup,
                args.iterations,
                args.repeats,
            ),
            "flashinfer_ragged_five_attention_calls": measure(
                [flashinfer_attention_graph.replay],
                args.warmup,
                args.iterations,
                args.repeats,
            ),
        }
        query_result = {
            "benchmark": "dspark_query_body",
            "concurrency": concurrency,
            "query_slots_per_request": block_size,
            "proposal_tokens_per_request": block_size - 1,
            "context_tokens_per_request": args.context_tokens,
            "validation": validation,
            "timings": timings,
        }
        query_results.append(query_result)
        print(json.dumps(query_result, sort_keys=True), flush=True)
        del (
            state,
            torch_attention_reference,
            torch_reference,
            flashinfer_reference,
            torch_graph,
            flashinfer_graph,
            dense_graph,
            torch_attention_graph,
            flashinfer_attention_graph,
        )
        gc.collect()
        torch.cuda.empty_cache()

    report = {
        "benchmark": "dspark_body_closure_summary",
        "checkpoint_convention": "speculators_bonus_anchor_1_plus_n",
        "fixture": args.fixture,
        "repo_id": fixture["repo_id"],
        "revision": fixture["revision"],
        "gpu": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "flashinfer": flashinfer.__version__,
        "flashinfer_backend": args.flashinfer_backend,
        "cuda": torch.version.cuda,
        "body_weight_bytes": body_weight_bytes(weights),
        "fused_context_kv_duplicate_bytes": tensor_bytes(weights.fused_context_kv),
        "protocol": {
            "seed": args.seed,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
        },
        "note": (
            "Benchmark-only real BF16 draft-body weights with synthetic RMS-like "
            "target/query states and persistent BF16 context KV. The embedding, "
            "LM/Markov/confidence heads, target verification, and serving path are "
            "excluded. Planned FlashInfer ragged NHD attention uses persistent KV "
            "storage and in-place query-slot writes; copy-heavy Torch SDPA is retained "
            "only as a numerical and timing reference."
        ),
        "context_results": context_results,
        "query_results": query_results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
