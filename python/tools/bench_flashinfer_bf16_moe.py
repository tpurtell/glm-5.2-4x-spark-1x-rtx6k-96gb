#!/usr/bin/env python3
"""Probe FlashInfer's grouped BF16 routed-MoE kernel at GLM-5.2 TP4 geometry.

This is an implementation-selection probe, not the canonical end-to-end
benchmark.  It deliberately uses a small expert population by default while
preserving the real hidden/intermediate widths, top-k, and dispersed routing
shape.  Increase ``--experts`` to 256 to exercise the full resident footprint.
"""

from __future__ import annotations

import argparse
import json

import torch
from flashinfer import shuffle_matrix_a
from flashinfer.fused_moe import cutlass_fused_moe, trtllm_bf16_routed_moe
from flashinfer.fused_moe.core import convert_to_block_layout
from flashinfer.tllm_enums import WeightLayout


def block_major_k(weights: torch.Tensor) -> torch.Tensor:
    """Apply the layout used by FlashInfer's routed BF16 correctness test."""

    packed = []
    for expert_weights in weights:
        shuffled = shuffle_matrix_a(expert_weights.view(torch.uint8), 64)
        packed.append(convert_to_block_layout(shuffled, 128))
    return torch.stack(packed).view(torch.bfloat16).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument(
        "--backend", choices=("cutlass", "trtllm"), default="cutlass"
    )
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.experts < args.top_k:
        parser.error("--experts must be at least --top-k")

    device = torch.device("cuda")
    torch.manual_seed(7)
    w13 = torch.randn(
        args.experts,
        2 * args.intermediate,
        args.hidden,
        device=device,
        dtype=torch.bfloat16,
    )
    w2 = torch.randn(
        args.experts,
        args.hidden,
        args.intermediate,
        device=device,
        dtype=torch.bfloat16,
    )
    if args.backend == "trtllm":
        w13 = block_major_k(w13)
        w2 = block_major_k(w2)

    for rows in args.rows:
        hidden = torch.randn(rows, args.hidden, device=device, dtype=torch.bfloat16)
        expert_ids = (
            torch.arange(rows * args.top_k, device=device, dtype=torch.int32)
            .reshape(rows, args.top_k)
            .remainder(args.experts)
        )
        route_weights = torch.full(
            (rows, args.top_k),
            1.0 / args.top_k,
            device=device,
            dtype=torch.float32,
        )
        packed_route_weights = route_weights.to(torch.bfloat16)
        packed_routes = (expert_ids << 16) | packed_route_weights.view(
            torch.int16
        ).to(torch.int32)
        output = torch.empty_like(hidden)

        def invoke() -> None:
            if args.backend == "cutlass":
                cutlass_fused_moe(
                    hidden,
                    expert_ids,
                    route_weights,
                    w13,
                    w2,
                    torch.bfloat16,
                    quant_scales=None,
                    output=output,
                    enable_pdl=True,
                    tune_max_num_tokens=max(args.rows),
                    use_fused_finalize=False,
                )
            else:
                trtllm_bf16_routed_moe(
                    topk_ids=packed_routes,
                    hidden_states=hidden,
                    gemm1_weights=w13,
                    gemm2_weights=w2,
                    num_experts=args.experts,
                    top_k=args.top_k,
                    n_group=None,
                    topk_group=None,
                    intermediate_size=args.intermediate,
                    local_expert_offset=0,
                    local_num_experts=args.experts,
                    routed_scaling_factor=None,
                    routing_method_type=5,
                    use_shuffled_weight=True,
                    weight_layout=WeightLayout.BlockMajorK,
                    do_finalize=True,
                    enable_pdl=True,
                    tune_max_num_tokens=max(args.rows),
                    output=output,
                )

        for _ in range(args.warmup):
            invoke()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            invoke()
        end.record()
        end.synchronize()
        elapsed_ms = start.elapsed_time(end) / args.iterations
        print(
            json.dumps(
                {
                    "rows": rows,
                    "routes": rows * args.top_k,
                    "experts": args.experts,
                    "backend": args.backend,
                    "elapsed_ms": elapsed_ms,
                    "rows_per_second": rows * 1000.0 / elapsed_ms,
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
