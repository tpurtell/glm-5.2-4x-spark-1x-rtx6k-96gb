#!/usr/bin/env python3
"""Measure low-precision dSpark Markov weights inside the real head trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from torchao.prototype.mx_formats.inference_workflow import (
    NVFP4DynamicActivationNVFP4WeightConfig,
)
from torchao.quantization import quantize_

from tune_dspark_head_closure import (
    FIXTURES,
    HIDDEN,
    MARKOV_RANK,
    VOCAB,
    capture,
    load_weights,
    make_state,
    measure,
    operations,
    parse_int_list,
)


FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def common_prefix_lengths(reference: torch.Tensor, candidate: torch.Tensor) -> list[int]:
    matches = reference == candidate
    return matches.to(torch.int64).cumprod(dim=1).sum(dim=1).cpu().tolist()


def storage_bytes(tensor: torch.Tensor) -> int:
    names = getattr(tensor, "tensor_data_names", None)
    if names is None:
        return tensor.numel() * tensor.element_size()
    return sum(
        getattr(tensor, name).numel() * getattr(tensor, name).element_size()
        for name in names
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare BF16, FP8, and NVFP4 Markov W2 inside the benchmark-only "
            "real-weight dSpark autoregressive head closure."
        )
    )
    parser.add_argument("--fixture", choices=sorted(FIXTURES), default="siro")
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    concurrency_values = parse_int_list(args.concurrency, "concurrency")
    if min(args.iterations, args.repeats) < 1 or args.warmup < 0:
        parser.error("iterations/repeats must be positive and warmup nonnegative")

    fixture = FIXTURES[args.fixture]
    proposal_tokens = fixture["proposal_tokens"]
    snapshot = Path(
        snapshot_download(
            fixture["repo_id"],
            revision=fixture["revision"],
            local_files_only=True,
        )
    )
    config = json.loads((snapshot / "config.json").read_text())
    if config["block_size"] != proposal_tokens + 1:
        raise RuntimeError("fixture does not use the expected 1+N bonus-anchor layout")
    torch.manual_seed(args.seed)
    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    weights = load_weights(snapshot, device)

    weight_scale = (
        weights.markov_w2.float().abs().amax() / FP8_MAX
    ).clamp_min(1.0e-12)
    markov_w2_fp8_t = (
        (weights.markov_w2 / weight_scale)
        .to(torch.float8_e4m3fn)
        .t()
        .contiguous()
    )
    nvfp4_linear = torch.nn.Linear(
        MARKOV_RANK,
        VOCAB,
        bias=False,
        device=device,
        dtype=torch.bfloat16,
    )
    nvfp4_linear.weight = torch.nn.Parameter(weights.markov_w2, requires_grad=False)
    quantize_(
        nvfp4_linear,
        NVFP4DynamicActivationNVFP4WeightConfig(
            use_dynamic_per_tensor_scale=True,
            use_triton_kernel=False,
        ),
    )
    torch.cuda.synchronize()

    results = []
    for concurrency in concurrency_values:
        state = make_state(
            concurrency,
            proposal_tokens,
            device,
            args.seed + concurrency * 10_000,
        )
        base_logits, _, bf16_sequence = operations(state, weights)

        def deferred_confidence(previous_embeddings: list[torch.Tensor]) -> None:
            markov_features = torch.stack(previous_embeddings, dim=1)
            features = torch.cat((state.hidden, markov_features), dim=-1)
            confidence = torch.sigmoid(
                F.linear(
                    features.view(-1, HIDDEN + MARKOV_RANK),
                    weights.confidence_weight,
                    weights.confidence_bias,
                ).view(concurrency, proposal_tokens)
            )
            state.output_confidence.copy_(confidence)

        def fp8_sequence() -> None:
            previous = state.anchor_tokens
            previous_embeddings = []
            for position in range(proposal_tokens):
                previous_embedding = F.embedding(previous, weights.markov_w1)
                previous_embeddings.append(previous_embedding)
                activation_scale = (
                    previous_embedding.float().abs().amax() / FP8_MAX
                ).clamp_min(1.0e-12)
                embedding_fp8 = (previous_embedding / activation_scale).to(
                    torch.float8_e4m3fn
                )
                markov_bias = torch._scaled_mm(
                    embedding_fp8,
                    markov_w2_fp8_t,
                    activation_scale,
                    weight_scale,
                    out_dtype=torch.bfloat16,
                )
                next_token = torch.argmax(
                    state.base_logits[:, position] + markov_bias, dim=-1
                )
                state.output_tokens[:, position].copy_(next_token)
                previous = next_token
            deferred_confidence(previous_embeddings)

        def nvfp4_sequence() -> None:
            previous = state.anchor_tokens
            previous_embeddings = []
            for position in range(proposal_tokens):
                previous_embedding = F.embedding(previous, weights.markov_w1)
                previous_embeddings.append(previous_embedding)
                markov_bias = nvfp4_linear(previous_embedding)
                next_token = torch.argmax(
                    state.base_logits[:, position] + markov_bias, dim=-1
                )
                state.output_tokens[:, position].copy_(next_token)
                previous = next_token
            deferred_confidence(previous_embeddings)

        def closure(sequence: Callable[[], None]) -> None:
            base_logits()
            sequence()

        base_logits()
        bf16_sequence()
        torch.cuda.synchronize()
        reference_tokens = state.output_tokens.clone()
        reference_confidence = state.output_confidence.clone()

        candidates = {}
        for name, sequence in (("fp8", fp8_sequence), ("nvfp4", nvfp4_sequence)):
            sequence()
            torch.cuda.synchronize()
            candidate_tokens = state.output_tokens.clone()
            candidate_confidence = state.output_confidence.clone()
            prefixes = common_prefix_lengths(reference_tokens, candidate_tokens)
            candidates[name] = {
                "token_agreement": float(
                    (candidate_tokens == reference_tokens).float().mean()
                ),
                "common_prefix_lengths": prefixes,
                "mean_common_prefix": sum(prefixes) / len(prefixes),
                "full_trajectory_agreement": float(
                    (candidate_tokens == reference_tokens).all(dim=1).float().mean()
                ),
                "confidence_max_abs": float(
                    (candidate_confidence - reference_confidence).abs().max()
                ),
            }

        bf16_graph = capture(lambda: closure(bf16_sequence))
        fp8_graph = capture(lambda: closure(fp8_sequence))
        nvfp4_graph = capture(lambda: closure(nvfp4_sequence))
        timing = {
            "bf16": measure(
                [bf16_graph.replay], args.warmup, args.iterations, args.repeats
            ),
            "fp8": measure(
                [fp8_graph.replay], args.warmup, args.iterations, args.repeats
            ),
            "nvfp4": measure(
                [nvfp4_graph.replay], args.warmup, args.iterations, args.repeats
            ),
        }
        bf16_ms = timing["bf16"]["gpu_ms"]["median"]
        result = {
            "benchmark": "dspark_head_lowp_trajectory",
            "concurrency": concurrency,
            "proposal_tokens": proposal_tokens,
            "quality": candidates,
            "timing": timing,
            "speedup": {
                dtype: bf16_ms / timing[dtype]["gpu_ms"]["median"]
                for dtype in ("fp8", "nvfp4")
            },
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        del state
        torch.cuda.empty_cache()

    report = {
        "benchmark": "dspark_head_lowp_trajectory_summary",
        "fixture": args.fixture,
        "repo_id": fixture["repo_id"],
        "revision": fixture["revision"],
        "checkpoint_convention": "speculators_bonus_anchor_1_plus_n",
        "gpu": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "protocol": {
            "seed": args.seed,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
        },
        "markov_weight_storage_bytes": {
            "bf16": storage_bytes(weights.markov_w2),
            "fp8": storage_bytes(markov_w2_fp8_t) + weight_scale.numel() * 4,
            "nvfp4": storage_bytes(nvfp4_linear.weight),
        },
        "note": (
            "Benchmark-only real-weight head trajectory over RMS-like synthetic "
            "draft hidden rows. Common-prefix agreement is not target acceptance."
        ),
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
