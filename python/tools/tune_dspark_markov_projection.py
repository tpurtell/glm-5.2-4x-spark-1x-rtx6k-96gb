#!/usr/bin/env python3
"""Tune benchmark-only low-precision dSpark Markov projections on real weights."""

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
from torchao.prototype.mx_formats.inference_workflow import (
    NVFP4DynamicActivationNVFP4WeightConfig,
)
from torchao.quantization import quantize_


MARKOV_RANK = 256
VOCAB = 154_880
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
FIXTURE = {
    "repo_id": "siro1/glm-5.2-dspark-preview",
    "revision": "7ff03018b3a443bfb9fca166739bd5f37ee5908b",
}


def parse_int_list(value: str, label: str) -> list[int]:
    try:
        result = [int(part) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be comma-separated integers") from exc
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError(f"{label} must contain positive integers")
    return result


def summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "max": ordered[-1],
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


def measure(
    launch: Callable[[], None], warmup: int, iterations: int, repeats: int
) -> dict[str, dict[str, float]]:
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    gpu = []
    host = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter_ns()
        start.record()
        for _ in range(iterations):
            launch()
        end.record()
        end.synchronize()
        wall_end = time.perf_counter_ns()
        gpu.append(start.elapsed_time(end) / iterations)
        host.append((wall_end - wall_start) / 1.0e6 / iterations)
    return {"gpu_ms": summary(gpu), "host_ms": summary(host)}


def tensor_storage_bytes(tensor: torch.Tensor) -> int:
    names = getattr(tensor, "tensor_data_names", None)
    if names is None:
        return tensor.numel() * tensor.element_size()
    return sum(
        getattr(tensor, name).numel() * getattr(tensor, name).element_size()
        for name in names
    )


@dataclass
class ProjectionState:
    inputs: list[torch.Tensor]
    bf16_outputs: list[torch.Tensor]
    fp8_outputs: list[torch.Tensor]
    nvfp4_outputs: list[torch.Tensor]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare real-weight BF16, FP8, and NVFP4 dSpark Markov projections "
            "without changing serving."
        )
    )
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--proposal-tokens", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    concurrency_values = parse_int_list(args.concurrency, "concurrency")
    if min(args.proposal_tokens, args.iterations, args.repeats) < 1 or args.warmup < 0:
        parser.error(
            "proposal-tokens/iterations/repeats must be positive and warmup nonnegative"
        )

    snapshot = Path(
        snapshot_download(
            FIXTURE["repo_id"],
            revision=FIXTURE["revision"],
            local_files_only=True,
        )
    )
    torch.manual_seed(args.seed)
    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    with safe_open(
        snapshot / "model.safetensors", framework="pt", device=str(device)
    ) as weights:
        markov_w1 = weights.get_tensor("markov_head.markov_w1.weight")
        markov_w2 = weights.get_tensor("markov_head.markov_w2.weight")
    if tuple(markov_w1.shape) != (VOCAB, MARKOV_RANK) or tuple(markov_w2.shape) != (
        VOCAB,
        MARKOV_RANK,
    ):
        raise RuntimeError("unexpected dSpark Markov weight geometry")

    weight_scale = (markov_w2.float().abs().amax() / FP8_MAX).clamp_min(1.0e-12)
    markov_w2_fp8_t = (markov_w2 / weight_scale).to(torch.float8_e4m3fn).t().contiguous()

    nvfp4_linear = torch.nn.Linear(
        MARKOV_RANK,
        VOCAB,
        bias=False,
        device=device,
        dtype=torch.bfloat16,
    )
    nvfp4_linear.weight = torch.nn.Parameter(markov_w2, requires_grad=False)
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
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + concurrency * 10_000)
        token_ids = torch.randint(
            VOCAB,
            (args.proposal_tokens, concurrency),
            generator=generator,
            device=device,
            dtype=torch.int64,
        )
        inputs = [F.embedding(row, markov_w1) for row in token_ids]
        state = ProjectionState(
            inputs=inputs,
            bf16_outputs=[
                torch.empty(concurrency, VOCAB, device=device, dtype=torch.bfloat16)
                for _ in inputs
            ],
            fp8_outputs=[
                torch.empty(concurrency, VOCAB, device=device, dtype=torch.bfloat16)
                for _ in inputs
            ],
            nvfp4_outputs=[
                torch.empty(concurrency, VOCAB, device=device, dtype=torch.bfloat16)
                for _ in inputs
            ],
        )

        def bf16_closure() -> None:
            for source, output in zip(
                state.inputs, state.bf16_outputs, strict=True
            ):
                output.copy_(F.linear(source, markov_w2))

        def fp8_closure() -> None:
            for source, output in zip(state.inputs, state.fp8_outputs, strict=True):
                activation_scale = (
                    source.float().abs().amax() / FP8_MAX
                ).clamp_min(1.0e-12)
                source_fp8 = (source / activation_scale).to(torch.float8_e4m3fn)
                output.copy_(
                    torch._scaled_mm(
                        source_fp8,
                        markov_w2_fp8_t,
                        activation_scale,
                        weight_scale,
                        out_dtype=torch.bfloat16,
                    )
                )

        def nvfp4_closure() -> None:
            for source, output in zip(
                state.inputs, state.nvfp4_outputs, strict=True
            ):
                output.copy_(nvfp4_linear(source))

        bf16_closure()
        fp8_closure()
        nvfp4_closure()
        torch.cuda.synchronize()
        reference = torch.stack(state.bf16_outputs).float()
        fp8 = torch.stack(state.fp8_outputs).float()
        nvfp4 = torch.stack(state.nvfp4_outputs).float()

        def error(candidate: torch.Tensor) -> dict[str, float]:
            difference = candidate - reference
            return {
                "max_abs": float(difference.abs().max()),
                "relative_l2": float(
                    torch.linalg.vector_norm(difference)
                    / torch.linalg.vector_norm(reference)
                ),
                "argmax_agreement": float(
                    (candidate.argmax(dim=-1) == reference.argmax(dim=-1))
                    .float()
                    .mean()
                ),
            }

        bf16_graph = capture(bf16_closure)
        fp8_graph = capture(fp8_closure)
        nvfp4_graph = capture(nvfp4_closure)
        timing = {
            "bf16": measure(
                bf16_graph.replay, args.warmup, args.iterations, args.repeats
            ),
            "fp8": measure(
                fp8_graph.replay, args.warmup, args.iterations, args.repeats
            ),
            "nvfp4": measure(
                nvfp4_graph.replay, args.warmup, args.iterations, args.repeats
            ),
        }
        bf16_ms = timing["bf16"]["gpu_ms"]["median"]
        result = {
            "benchmark": "dspark_markov_projection",
            "concurrency": concurrency,
            "proposal_tokens": args.proposal_tokens,
            "error": {"fp8": error(fp8), "nvfp4": error(nvfp4)},
            "timing": timing,
            "speedup": {
                dtype: bf16_ms / timing[dtype]["gpu_ms"]["median"]
                for dtype in ("fp8", "nvfp4")
            },
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        del state, reference, fp8, nvfp4
        torch.cuda.empty_cache()

    report = {
        "benchmark": "dspark_markov_projection_summary",
        "repo_id": FIXTURE["repo_id"],
        "revision": FIXTURE["revision"],
        "gpu": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "weight_storage_bytes": {
            "bf16": tensor_storage_bytes(markov_w2),
            "fp8": tensor_storage_bytes(markov_w2_fp8_t) + weight_scale.numel() * 4,
            "nvfp4": tensor_storage_bytes(nvfp4_linear.weight),
        },
        "note": (
            "Benchmark-only draft-head precision experiment. Quantized Markov "
            "weights can change acceptance and require live model-quality gates."
        ),
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
