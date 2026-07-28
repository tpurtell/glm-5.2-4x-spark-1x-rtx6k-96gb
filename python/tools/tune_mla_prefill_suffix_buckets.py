#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


REFERENCE_ROOT = Path(__file__).resolve().parents[1] / "reference"
if str(REFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFERENCE_ROOT))

from glmrt_reference.b12x_mla_capture import (  # noqa: E402
    capture_flashinfer_mla_rope_attention,
    prepare_flashinfer_mla_rope_attention,
)


HEADS = 64
NOPE_DIM = 192
ROPE_DIM = 64
V_DIM = 256
QK_DIM = NOPE_DIM + ROPE_DIM
SCALE = QK_DIM**-0.5


def parse_int_list(raw: str, label: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in raw.split(",") if item)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{label} must be comma-separated integers"
        ) from error
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError(f"{label} values must be positive")
    return values


def descriptor(tensor: torch.Tensor) -> dict[str, int]:
    return {
        "ptr": tensor.data_ptr(),
        "bytes": tensor.numel() * tensor.element_size(),
        "device_id": tensor.device.index or 0,
    }


def capture(operation: Callable[[], None]) -> torch.cuda.CUDAGraph:
    operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()
    return graph


def measure(
    graph: torch.cuda.CUDAGraph,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, float | list[float]]:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return {
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "samples_ms": samples,
    }


@dataclass
class Chunk:
    context: dict
    kwargs: dict[str, int | float]


def make_chunks(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    values: torch.Tensor,
    q_staging: torch.Tensor,
    k_staging: torch.Tensor,
    output: torch.Tensor,
    workspace: torch.Tensor,
    prefix_rows: int,
    suffix_rows: int,
    chunk_rows: int,
) -> list[Chunk]:
    if suffix_rows % chunk_rows != 0:
        raise ValueError(
            f"suffix rows {suffix_rows} are not divisible by chunk rows {chunk_rows}"
        )
    chunks = []
    for query_offset in range(0, suffix_rows, chunk_rows):
        total_rows = prefix_rows + query_offset + chunk_rows
        context = {
            "cuda_stream": torch.cuda.current_stream().cuda_stream,
            "buffers": {
                "q_nope": descriptor(
                    q_nope[query_offset : query_offset + chunk_rows]
                ),
                "q_rope": descriptor(
                    q_rope[query_offset : query_offset + chunk_rows]
                ),
                "k_nope": descriptor(k_nope[:total_rows]),
                "k_rope": descriptor(k_rope[:total_rows]),
                "values": descriptor(values[:total_rows]),
                "q": descriptor(q_staging[:chunk_rows]),
                "k": descriptor(k_staging[:total_rows]),
                "output": descriptor(
                    output[query_offset : query_offset + chunk_rows]
                ),
                "workspace": descriptor(workspace),
            },
        }
        kwargs = {
            "rows": total_rows,
            "query_row_offset": prefix_rows + query_offset,
            "query_rows": chunk_rows,
            "heads": HEADS,
            "nope_dim": NOPE_DIM,
            "rope_dim": ROPE_DIM,
            "v_dim": V_DIM,
            "scale": SCALE,
        }
        chunks.append(Chunk(context=context, kwargs=kwargs))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare exact expanded-BF16 FlashInfer prefill suffix bucket policies "
            "above an existing cached prefix."
        )
    )
    parser.add_argument("--prefix-rows", type=int, default=1_024)
    parser.add_argument("--suffix-rows", default="256,512,1024")
    parser.add_argument("--candidate-chunks", default="256,512,1024")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    suffix_values = parse_int_list(args.suffix_rows, "suffix-rows")
    candidate_chunks = parse_int_list(args.candidate_chunks, "candidate-chunks")
    if args.prefix_rows < 1:
        parser.error("prefix-rows must be positive")
    if min(args.iterations, args.repeats) < 1 or args.warmup < 0:
        parser.error(
            "iterations/repeats must be positive and warmup nonnegative"
        )

    torch.manual_seed(args.seed)
    torch.cuda.init()
    from flashinfer.prefill import SINGLE_KERNEL_TMP_SIZE

    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    max_suffix = max(suffix_values)
    max_total = args.prefix_rows + max_suffix
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)
    q_nope = torch.randn(
        (max_suffix, HEADS, NOPE_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ) * 0.05
    q_rope = torch.randn(
        (max_suffix, HEADS, ROPE_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ) * 0.05
    k_nope = torch.randn(
        (max_total, HEADS, NOPE_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ) * 0.05
    k_rope = torch.randn(
        (max_total, ROPE_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ) * 0.05
    values = torch.randn(
        (max_total, HEADS, V_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ) * 0.05
    q_staging = torch.empty(
        (max_suffix, HEADS, QK_DIM), dtype=torch.bfloat16, device=device
    )
    k_staging = torch.empty(
        (max_total, HEADS, QK_DIM), dtype=torch.bfloat16, device=device
    )
    workspace = torch.empty(SINGLE_KERNEL_TMP_SIZE, dtype=torch.uint8, device=device)
    results = []

    for suffix_rows in suffix_values:
        policies = tuple(
            chunk_rows
            for chunk_rows in candidate_chunks
            if chunk_rows <= suffix_rows and suffix_rows % chunk_rows == 0
        )
        if suffix_rows not in policies:
            policies = (*policies, suffix_rows)
        outputs = {
            chunk_rows: torch.empty(
                (suffix_rows, HEADS, V_DIM),
                dtype=torch.bfloat16,
                device=device,
            )
            for chunk_rows in policies
        }
        chunks_by_policy = {
            chunk_rows: make_chunks(
                q_nope,
                q_rope,
                k_nope,
                k_rope,
                values,
                q_staging,
                k_staging,
                outputs[chunk_rows],
                workspace,
                args.prefix_rows,
                suffix_rows,
                chunk_rows,
            )
            for chunk_rows in policies
        }
        for chunks in chunks_by_policy.values():
            for chunk in chunks:
                prepare_flashinfer_mla_rope_attention(
                    chunk.context, **chunk.kwargs
                )
        torch.cuda.synchronize()

        graphs = {}
        for chunk_rows, chunks in chunks_by_policy.items():

            def launch(chunks=chunks) -> None:
                for chunk in chunks:
                    chunk.context["cuda_stream"] = (
                        torch.cuda.current_stream().cuda_stream
                    )
                    capture_flashinfer_mla_rope_attention(
                        chunk.context, **chunk.kwargs
                    )

            graphs[chunk_rows] = capture(launch)
            graphs[chunk_rows].replay()
        torch.cuda.synchronize()
        reference = outputs[suffix_rows].clone()

        for chunk_rows in policies:
            exact = bool(torch.equal(outputs[chunk_rows], reference))
            actual_f32 = outputs[chunk_rows].float()
            reference_f32 = reference.float()
            delta = actual_f32 - reference_f32
            max_abs = float(delta.abs().max())
            relative_l2 = float(delta.norm()) / max(
                float(reference_f32.norm()), 1.0e-12
            )
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    actual_f32.flatten(), reference_f32.flatten(), dim=0
                )
            )
            numerically_close = max_abs <= 1.25e-4 and cosine >= 0.99998
            timing = measure(
                graphs[chunk_rows],
                args.warmup,
                args.iterations,
                args.repeats,
            )
            result = {
                "benchmark": "mla_prefill_suffix_buckets",
                "chunk_count": suffix_rows // chunk_rows,
                "chunk_rows": chunk_rows,
                "exact_vs_monolithic": exact,
                "cosine_vs_monolithic": cosine,
                "max_abs_vs_monolithic": max_abs,
                "numerically_close_vs_monolithic": numerically_close,
                "relative_l2_vs_monolithic": relative_l2,
                "prefix_rows": args.prefix_rows,
                "suffix_rows": suffix_rows,
                "timing": timing,
            }
            if not numerically_close:
                raise RuntimeError(
                    f"split suffix policy diverged: suffix={suffix_rows} chunk={chunk_rows} "
                    f"max_abs={max_abs} relative_l2={relative_l2} cosine={cosine}"
                )
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

    report = {
        "benchmark": "mla_prefill_suffix_buckets_summary",
        "gpu": properties.name,
        "note": (
            "Benchmark-only expanded-BF16 FlashInfer path. No serving bucket or "
            "dispatch policy is changed."
        ),
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
