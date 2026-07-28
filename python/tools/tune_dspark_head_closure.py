#!/usr/bin/env python3
"""Benchmark the real-weight dSpark LM/Markov/confidence head closure off path."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open


HIDDEN = 6_144
MARKOV_RANK = 256
VOCAB = 154_880

FIXTURES = {
    "redhat": {
        "repo_id": "RedHatAI/GLM-5.2-speculator.dspark",
        "revision": "8bc9ac46fbf507f3ee3ad82304116a1f63e9edb4",
        "proposal_tokens": 8,
    },
    "siro": {
        "repo_id": "siro1/glm-5.2-dspark-preview",
        "revision": "7ff03018b3a443bfb9fca166739bd5f37ee5908b",
        "proposal_tokens": 15,
    },
}


def parse_int_list(value: str, label: str) -> list[int]:
    try:
        values = [int(part) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be comma-separated integers") from exc
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError(f"{label} must contain positive integers")
    return values


def quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]

    return {
        "min": min(values),
        "median": statistics.median(values),
        "p90": percentile(0.90),
        "max": max(values),
    }


def measure(
    launches: list[Callable[[], None]], warmup: int, iterations: int, repeats: int
) -> dict[str, dict[str, float]]:
    for _ in range(warmup):
        for launch in launches:
            launch()
    torch.cuda.synchronize()
    gpu_samples = []
    host_samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter_ns()
        start.record()
        for _ in range(iterations):
            for launch in launches:
                launch()
        end.record()
        end.synchronize()
        wall_end = time.perf_counter_ns()
        gpu_samples.append(start.elapsed_time(end) / iterations)
        host_samples.append((wall_end - wall_start) / 1.0e6 / iterations)
    return {
        "gpu_ms": quantiles(gpu_samples),
        "host_ms": quantiles(host_samples),
    }


def capture(launch: Callable[[], None]) -> torch.cuda.CUDAGraph:
    current = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    side.wait_stream(current)
    with torch.cuda.stream(side):
        launch()
    current.wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    torch.cuda.synchronize()
    return graph


@dataclass
class HeadWeights:
    lm_head: torch.Tensor
    markov_w1: torch.Tensor
    markov_w2: torch.Tensor
    confidence_weight: torch.Tensor
    confidence_bias: torch.Tensor


@dataclass
class HeadState:
    hidden: torch.Tensor
    anchor_tokens: torch.Tensor
    base_logits: torch.Tensor
    output_tokens: torch.Tensor
    output_confidence: torch.Tensor


def load_weights(snapshot: Path, device: torch.device) -> HeadWeights:
    path = snapshot / "model.safetensors"
    names = {
        "lm_head": "lm_head.weight",
        "markov_w1": "markov_head.markov_w1.weight",
        "markov_w2": "markov_head.markov_w2.weight",
        "confidence_weight": "confidence_head.proj.weight",
        "confidence_bias": "confidence_head.proj.bias",
    }
    loaded = {}
    with safe_open(path, framework="pt", device=str(device)) as weights:
        for field, name in names.items():
            loaded[field] = weights.get_tensor(name)
    result = HeadWeights(**loaded)
    expected = {
        "lm_head": (VOCAB, HIDDEN),
        "markov_w1": (VOCAB, MARKOV_RANK),
        "markov_w2": (VOCAB, MARKOV_RANK),
        "confidence_weight": (1, HIDDEN + MARKOV_RANK),
        "confidence_bias": (1,),
    }
    for name, shape in expected.items():
        tensor = getattr(result, name)
        if tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16:
            raise RuntimeError(
                f"dSpark {name} expected BF16 {shape}, got {tensor.dtype} {tuple(tensor.shape)}"
            )
    return result


def make_state(
    concurrency: int, proposal_tokens: int, device: torch.device, seed: int
) -> HeadState:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    hidden = torch.randn(
        concurrency,
        proposal_tokens,
        HIDDEN,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    anchor_tokens = torch.randint(
        VOCAB,
        (concurrency,),
        generator=generator,
        device=device,
        dtype=torch.int64,
    )
    return HeadState(
        hidden=hidden,
        anchor_tokens=anchor_tokens,
        base_logits=torch.empty(
            concurrency,
            proposal_tokens,
            VOCAB,
            device=device,
            dtype=torch.bfloat16,
        ),
        output_tokens=torch.empty(
            concurrency, proposal_tokens, device=device, dtype=torch.int64
        ),
        output_confidence=torch.empty(
            concurrency, proposal_tokens, device=device, dtype=torch.float32
        ),
    )


def operations(
    state: HeadState, weights: HeadWeights
) -> tuple[Callable[[], None], Callable[[], None], Callable[[], None]]:
    concurrency, proposal_tokens, _ = state.hidden.shape
    hidden_flat = state.hidden.view(concurrency * proposal_tokens, HIDDEN)

    def base_logits() -> None:
        torch.mm(hidden_flat, weights.lm_head.t(), out=state.base_logits.view(-1, VOCAB))

    def sequential_inline_confidence() -> None:
        previous = state.anchor_tokens
        for position in range(proposal_tokens):
            previous_embedding = F.embedding(previous, weights.markov_w1)
            markov_bias = F.linear(previous_embedding, weights.markov_w2)
            next_token = torch.argmax(
                state.base_logits[:, position] + markov_bias, dim=-1
            )
            confidence_features = torch.cat(
                (state.hidden[:, position], previous_embedding), dim=-1
            )
            confidence = torch.sigmoid(
                F.linear(
                    confidence_features,
                    weights.confidence_weight,
                    weights.confidence_bias,
                ).squeeze(-1)
            )
            state.output_tokens[:, position].copy_(next_token)
            state.output_confidence[:, position].copy_(confidence)
            previous = next_token

    def sequential_deferred_confidence() -> None:
        previous = state.anchor_tokens
        previous_embeddings = []
        for position in range(proposal_tokens):
            previous_embedding = F.embedding(previous, weights.markov_w1)
            previous_embeddings.append(previous_embedding)
            markov_bias = F.linear(previous_embedding, weights.markov_w2)
            next_token = torch.argmax(
                state.base_logits[:, position] + markov_bias, dim=-1
            )
            state.output_tokens[:, position].copy_(next_token)
            previous = next_token
        markov_features = torch.stack(previous_embeddings, dim=1)
        confidence_features = torch.cat((state.hidden, markov_features), dim=-1)
        confidence = torch.sigmoid(
            F.linear(
                confidence_features.view(-1, HIDDEN + MARKOV_RANK),
                weights.confidence_weight,
                weights.confidence_bias,
            ).view(concurrency, proposal_tokens)
        )
        state.output_confidence.copy_(confidence)

    return base_logits, sequential_inline_confidence, sequential_deferred_confidence


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the benchmark-only dSpark batched LM-head plus sequential "
            "Markov/confidence closure using pinned real weights."
        )
    )
    parser.add_argument("--fixture", choices=sorted(FIXTURES), default="siro")
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    concurrency_values = parse_int_list(args.concurrency, "concurrency")
    if min(args.iterations, args.repeats) < 1 or args.warmup < 0:
        parser.error("iterations/repeats must be positive and warmup nonnegative")

    fixture = FIXTURES[args.fixture]
    snapshot = Path(
        snapshot_download(
            fixture["repo_id"],
            revision=fixture["revision"],
            local_files_only=True,
        )
    )
    config = json.loads((snapshot / "config.json").read_text())
    proposal_tokens = fixture["proposal_tokens"]
    if config["block_size"] != proposal_tokens + 1:
        raise RuntimeError("fixture does not use the expected 1+N bonus-anchor layout")

    torch.manual_seed(args.seed)
    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    weights = load_weights(snapshot, device)
    torch.cuda.synchronize()
    resident_weight_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            weights.lm_head,
            weights.markov_w1,
            weights.markov_w2,
            weights.confidence_weight,
            weights.confidence_bias,
        )
    )

    results = []
    for concurrency in concurrency_values:
        state = make_state(
            concurrency,
            proposal_tokens,
            device,
            args.seed + concurrency * 10_000,
        )
        base_logits, inline_confidence, deferred_confidence = operations(state, weights)

        base_logits()
        inline_confidence()
        torch.cuda.synchronize()
        reference_tokens = state.output_tokens.clone()
        reference_confidence = state.output_confidence.clone()

        deferred_confidence()
        torch.cuda.synchronize()
        deferred_tokens = state.output_tokens.clone()
        deferred_confidence_values = state.output_confidence.clone()
        token_exact = bool(torch.equal(reference_tokens, deferred_tokens))
        confidence_max_abs = float(
            (reference_confidence - deferred_confidence_values).abs().max()
        )
        if not token_exact:
            raise RuntimeError(
                f"deferred confidence changed dSpark tokens at concurrency {concurrency}"
            )

        base_graph = capture(base_logits)
        inline_graph = capture(inline_confidence)
        deferred_graph = capture(deferred_confidence)

        def full_inline() -> None:
            base_logits()
            inline_confidence()

        def full_deferred() -> None:
            base_logits()
            deferred_confidence()

        full_inline_graph = capture(full_inline)
        full_deferred_graph = capture(full_deferred)
        full_deferred_graph.replay()
        torch.cuda.synchronize()
        graph_token_exact = bool(torch.equal(reference_tokens, state.output_tokens))
        graph_confidence_max_abs = float(
            (reference_confidence - state.output_confidence).abs().max()
        )
        if not graph_token_exact:
            raise RuntimeError(
                f"captured dSpark closure changed tokens at concurrency {concurrency}"
            )

        timings = {
            "base_lm_head_graph": measure(
                [base_graph.replay], args.warmup, args.iterations, args.repeats
            ),
            "sequential_inline_confidence_graph": measure(
                [inline_graph.replay], args.warmup, args.iterations, args.repeats
            ),
            "sequential_deferred_confidence_graph": measure(
                [deferred_graph.replay], args.warmup, args.iterations, args.repeats
            ),
            "split_base_deferred_graphs": measure(
                [base_graph.replay, deferred_graph.replay],
                args.warmup,
                args.iterations,
                args.repeats,
            ),
            "single_graph_inline_confidence": measure(
                [full_inline_graph.replay],
                args.warmup,
                args.iterations,
                args.repeats,
            ),
            "single_graph_deferred_confidence": measure(
                [full_deferred_graph.replay],
                args.warmup,
                args.iterations,
                args.repeats,
            ),
            "eager_deferred_confidence": measure(
                [full_deferred], args.warmup, args.iterations, args.repeats
            ),
        }
        inline_ms = timings["single_graph_inline_confidence"]["gpu_ms"]["median"]
        deferred_ms = timings["single_graph_deferred_confidence"]["gpu_ms"]["median"]
        result = {
            "benchmark": "dspark_head_closure",
            "concurrency": concurrency,
            "proposal_tokens": proposal_tokens,
            "target_verification_rows": concurrency * (proposal_tokens + 1),
            "token_exact": token_exact and graph_token_exact,
            "confidence_max_abs": max(confidence_max_abs, graph_confidence_max_abs),
            "timings": timings,
            "deferred_confidence_speedup": inline_ms / deferred_ms,
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        del state
        torch.cuda.empty_cache()

    report = {
        "benchmark": "dspark_head_closure_summary",
        "checkpoint_convention": "speculators_bonus_anchor_1_plus_n",
        "fixture": args.fixture,
        "repo_id": fixture["repo_id"],
        "revision": fixture["revision"],
        "gpu": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "resident_weight_bytes": resident_weight_bytes,
        "note": (
            "Benchmark-only real-weight head closure; no serving module imports this tool. "
            "The LM head is byte-identical to the current target and should be aliased live."
        ),
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
