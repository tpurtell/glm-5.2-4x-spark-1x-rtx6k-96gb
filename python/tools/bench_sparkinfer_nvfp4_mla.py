#!/usr/bin/env python3
"""Benchmark Sparkinfer's native packed-NVFP4 GLM sparse-MLA kernels.

Run against a Sparkinfer source checkout, for example:

  PYTHONPATH=.glmrt-cache/external/sparkinfer-master \
    .venv/bin/python python/tools/bench_sparkinfer_nvfp4_mla.py

The 432-byte cache record is the production BF16-RoPE ABI:
256 bytes packed E2M1 latent, 32 bytes E4M3 group-16 scales, 16 bytes
padding, and 128 bytes BF16 RoPE.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Callable

import torch

from sparkinfer.attention import sparse_mla
from sparkinfer.attention._shared.mla.prefill import run_unified_prefill
from sparkinfer.attention._shared.mla.traits import ScaleFormat


PAGE_SIZE = 64
HEAD_DIM = 576
V_HEAD_DIM = 512
RECORD_BYTES = 432


def comma_ints(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def comma_strings(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one mode")
    return values


def make_cache(tokens: int, generator: torch.Generator) -> torch.Tensor:
    if tokens % PAGE_SIZE:
        raise ValueError(f"tokens must be divisible by {PAGE_SIZE}")
    packed = torch.randint(
        0,
        256,
        (tokens, 256),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    # 0x38 is E4M3 1.0. Random packed values plus unit scales are sufficient
    # to exercise every native load/dequant path without exceptional scales.
    scales = torch.full((tokens, 32), 0x38, dtype=torch.uint8, device="cuda")
    padding = torch.zeros((tokens, 16), dtype=torch.uint8, device="cuda")
    rope = torch.randn(
        (tokens, 64),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    ).view(torch.uint8).reshape(tokens, 128)
    return torch.cat((packed, scales, padding, rope), dim=1).view(
        tokens // PAGE_SIZE,
        PAGE_SIZE,
        RECORD_BYTES,
    )


def capture(launch: Callable[[], object]) -> torch.cuda.CUDAGraph:
    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    torch.cuda.synchronize()
    return graph


def warm_samples(
    graph: torch.cuda.CUDAGraph,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> list[float]:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return samples


def cold_samples(
    graph: torch.cuda.CUDAGraph,
    *,
    l2_flush: torch.Tensor,
    repeats: int,
) -> list[float]:
    samples: list[float] = []
    for _ in range(repeats):
        l2_flush.zero_()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=comma_ints, default=[1, 2, 3, 4, 5, 6, 7, 8, 12, 16])
    parser.add_argument("--modes", type=comma_strings, default=["prefill", "decode"])
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--cold-repeats", type=int, default=25)
    parser.add_argument("--l2-flush-mib", type=int, default=192)
    args = parser.parse_args()

    if any(mode not in ("prefill", "decode") for mode in args.modes):
        parser.error("--modes accepts prefill and decode")
    if any(rows < 1 or rows > 2048 for rows in args.rows):
        parser.error("--rows values must be in 1..=2048")
    if args.heads % 8:
        parser.error("--heads must be divisible by 8")
    if args.topk <= 0 or args.topk % PAGE_SIZE:
        parser.error(f"--topk must be a positive multiple of {PAGE_SIZE}")
    if not sparse_mla.is_supported(torch.device("cuda")):
        parser.error("Sparkinfer sparse MLA is not supported on this GPU")

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260727)
    max_rows = max(args.rows)
    q = torch.randn(
        (max_rows, args.heads, HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    kv = make_cache(args.topk, generator)
    indices = torch.arange(args.topk, dtype=torch.int32, device="cuda").repeat(
        max_rows, 1
    )
    cache_seqlens = torch.full(
        (max_rows,), args.topk, dtype=torch.int32, device="cuda"
    )
    active = cache_seqlens.clone()
    l2_flush = torch.empty(
        args.l2_flush_mib * 1024 * 1024,
        dtype=torch.uint8,
        device="cuda",
    )
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    for mode in args.modes:
        for rows in args.rows:
            q_view = q[:rows]
            indices_view = indices[:rows]
            cache_seqlens_view = cache_seqlens[:rows]
            active_view = active[:rows]
            keepalive: list[torch.Tensor] = []

            if mode == "prefill":
                output = torch.empty(
                    (rows, args.heads, V_HEAD_DIM),
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                lse = torch.empty(
                    (rows, args.heads), dtype=torch.float32, device="cuda"
                )
                keepalive.extend((output, lse))

                def launch() -> object:
                    return run_unified_prefill(
                        q=q_view,
                        kv_cache=kv,
                        topk_indices=indices_view,
                        topk_length=active_view,
                        sm_scale=sm_scale,
                        latent_scale=1.0,
                        page_block_size=PAGE_SIZE,
                        output=output,
                        lse_out=lse,
                        scale_format=int(ScaleFormat.NVFP4_E4M3),
                        fp8_rope=False,
                    )

            else:
                plan = sparse_mla.plan(
                    sparse_mla.Caps(
                        device="cuda",
                        num_q_heads=args.heads,
                        max_q_rows=rows,
                        max_width=args.topk,
                        kv_dtype=torch.uint8,
                        head_dim=HEAD_DIM,
                        v_head_dim=V_HEAD_DIM,
                        mode="decode",
                        max_batch=rows,
                        max_chunks_per_row=64,
                        page_size=PAGE_SIZE,
                    )
                )
                spec = plan.scratch_specs()[0]
                scratch = torch.empty(
                    spec.shape, dtype=spec.dtype, device=spec.device
                )
                binding = sparse_mla.bind(
                    plan,
                    scratch=scratch,
                    q=q_view,
                    selected_indices=indices_view,
                    cache_seqlens_int32=cache_seqlens_view,
                    nsa_cache_seqlens_int32=active_view,
                )
                keepalive.append(scratch)

                def launch() -> object:
                    return sparse_mla.run_decode(
                        binding=binding,
                        kv_cache=kv,
                        sm_scale=sm_scale,
                        latent_scale=1.0,
                        v_head_dim=V_HEAD_DIM,
                        scale_format=int(ScaleFormat.NVFP4_E4M3),
                        fp8_rope=False,
                    )

            probe = launch()
            probe_output = probe[0] if isinstance(probe, tuple) else probe
            torch.cuda.synchronize()
            if tuple(probe_output.shape) != (rows, args.heads, V_HEAD_DIM):
                raise RuntimeError(
                    f"{mode} returned unexpected shape {tuple(probe_output.shape)}"
                )
            if not bool(torch.isfinite(probe_output).all().item()):
                raise RuntimeError(f"{mode} returned non-finite output")
            if int(torch.count_nonzero(probe_output).item()) == 0:
                raise RuntimeError(f"{mode} returned an all-zero output")

            graph = capture(launch)
            warm = warm_samples(
                graph,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            cold = cold_samples(
                graph,
                l2_flush=l2_flush,
                repeats=args.cold_repeats,
            )
            result = {
                "mode": mode,
                "rows": rows,
                "heads": args.heads,
                "topk": args.topk,
                "record_bytes": RECORD_BYTES,
                "warm_ms_median": statistics.median(warm),
                "warm_ms_min": min(warm),
                "cold_ms_median": statistics.median(cold),
                "cold_ms_min": min(cold),
            }
            print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
