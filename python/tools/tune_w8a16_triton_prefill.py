#!/usr/bin/env python3
"""Sweep register-dequantized Triton W8A16 prefill GEMM tiles."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import torch
import triton
import triton.language as tl

from tune_w8a16_projection import (
    CATALOG_PATH,
    PROJECTION_TENSORS,
    Timing,
    bench,
    check_status,
    load_bf16_weight,
    metrics,
)


CONFIGS = (
    (16, 64, 64, 4, 3),
    (32, 64, 64, 4, 3),
    (32, 128, 64, 8, 3),
    (64, 64, 64, 8, 3),
    (64, 128, 64, 8, 3),
    (128, 64, 64, 8, 3),
    (128, 128, 64, 8, 3),
    (64, 64, 128, 8, 3),
    (128, 128, 32, 8, 3),
    (128, 128, 64, 4, 3),
    (128, 128, 64, 8, 2),
    (128, 128, 64, 8, 4),
    (128, 128, 128, 8, 3),
    (256, 64, 64, 8, 3),
    (64, 256, 64, 8, 3),
    (64, 128, 128, 8, 3),
    (64, 256, 128, 8, 3),
    (128, 64, 128, 8, 3),
    (128, 256, 64, 8, 3),
    (128, 128, 128, 4, 3),
    (128, 128, 128, 8, 2),
    (64, 64, 256, 8, 2),
    (32, 256, 128, 8, 3),
    (64, 256, 128, 4, 3),
    (64, 256, 128, 16, 3),
    (64, 256, 128, 8, 2),
    (32, 512, 64, 8, 3),
    (32, 512, 64, 16, 3),
    (64, 512, 64, 16, 3),
)


@triton.jit
def w8a16_group256_gemm(
    a,
    weight,
    scales,
    output,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DEQUANT_BF16: tl.constexpr,
    POST_SCALE_GROUP: tl.constexpr,
    ROW_MAJOR_WEIGHT: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    programs_per_group = GROUP_M * num_pid_n
    group_id = pid // programs_per_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % programs_per_group) % group_size_m)
    pid_n = (pid % programs_per_group) // group_size_m

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if POST_SCALE_GROUP:
        for group_start in range(0, K, 256):
            group_accumulator = tl.zeros(
                (BLOCK_M, BLOCK_N), dtype=tl.float32
            )
            for group_k in range(0, 256, BLOCK_K):
                k_start = group_start + group_k
                a_tile = tl.load(
                    a
                    + offsets_m[:, None] * K
                    + k_start
                    + offsets_k[None, :],
                    mask=offsets_m[:, None] < M,
                    other=0.0,
                )
                if ROW_MAJOR_WEIGHT:
                    quantized = tl.trans(
                        tl.load(
                            weight
                            + offsets_n[:, None] * K
                            + k_start
                            + offsets_k[None, :]
                        )
                    )
                else:
                    quantized = tl.load(
                        weight
                        + (k_start + offsets_k[:, None]) * N
                        + offsets_n[None, :]
                    )
                group_accumulator += tl.dot(
                    a_tile, quantized.to(tl.bfloat16)
                )
            if ROW_MAJOR_WEIGHT:
                scale = tl.load(
                    scales
                    + offsets_n * (K // 256)
                    + group_start // 256
                )
            else:
                scale = tl.load(
                    scales + (group_start // 256) * N + offsets_n
                )
            accumulator += group_accumulator * scale[None, :]
    else:
        for k_start in range(0, K, BLOCK_K):
            a_tile = tl.load(
                a + offsets_m[:, None] * K + k_start + offsets_k[None, :],
                mask=offsets_m[:, None] < M,
                other=0.0,
            )
            if ROW_MAJOR_WEIGHT:
                quantized = tl.trans(
                    tl.load(
                        weight
                        + offsets_n[:, None] * K
                        + k_start
                        + offsets_k[None, :]
                    )
                )
                scale = tl.load(
                    scales + offsets_n * (K // 256) + k_start // 256
                )
            else:
                quantized = tl.load(
                    weight
                    + (k_start + offsets_k[:, None]) * N
                    + offsets_n[None, :]
                )
                scale = tl.load(
                    scales + (k_start // 256) * N + offsets_n
                )
            if DEQUANT_BF16:
                dequantized = quantized.to(tl.bfloat16) * scale[None, :]
            else:
                dequantized = (quantized.to(tl.float32) * scale[None, :]).to(
                    tl.bfloat16
                )
            accumulator += tl.dot(a_tile, dequantized)

    tl.store(
        output + offsets_m[:, None] * N + offsets_n[None, :],
        accumulator.to(tl.bfloat16),
        mask=offsets_m[:, None] < M,
    )


def configure_native(path: Path):
    native = ctypes.CDLL(str(path.resolve()))
    quantize = native.glmrt_cuda_quantize_bf16_w8a16_group256_async
    quantize.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    )
    quantize.restype = ctypes.c_int
    dequantize = native.glmrt_cuda_dequantize_w8a16_group256_bf16_async
    dequantize.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    dequantize.restype = ctypes.c_int
    return quantize, dequantize


def label(config: tuple[int, int, int, int, int]) -> str:
    block_m, block_n, block_k, warps, stages = config
    return f"triton-m{block_m}-n{block_n}-k{block_k}-w{warps}-s{stages}"


def report(kernel: str, rows: int, timing: Timing) -> None:
    print(
        f"timing rows={rows} kernel={kernel} "
        f"median_ms={timing.median_ms:.6f} "
        f"range_ms={timing.minimum_ms:.6f}-{timing.maximum_ms:.6f} "
        f"projection_tokens_per_second={rows * 1000.0 / timing.median_ms:.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--native-library",
        type=Path,
        default=Path("native/build-w8a16/libglmrt_native.so"),
    )
    parser.add_argument(
        "--tensor", choices=tuple(PROJECTION_TENSORS), default="o"
    )
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument(
        "--weight-layout", choices=("k-major", "row-major"), default="k-major"
    )
    parser.add_argument(
        "--config",
        action="append",
        help="benchmark one BLOCK_M,BLOCK_N,BLOCK_K,warps,stages tuple",
    )
    parser.add_argument(
        "--dequant-mode",
        choices=("pre-fp32", "pre-bf16", "post-fp32", "all"),
        default="pre-fp32",
        help="direct-kernel group-scale placement to test",
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    configs = CONFIGS
    if args.config:
        configs = tuple(
            tuple(int(value) for value in selected.split(","))
            for selected in args.config
        )
        if any(len(selected) != 5 for selected in configs):
            raise ValueError("--config requires five comma-separated integers")

    with CATALOG_PATH.open() as handle:
        catalog = json.load(handle)
    name = PROJECTION_TENSORS[args.tensor]
    weight = load_bf16_weight(catalog, name)
    output_rows, hidden = weight.shape
    groups = hidden // 256
    quantize, dequantize = configure_native(args.native_library)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    weight_k = torch.empty((hidden, output_rows), device="cuda", dtype=torch.int8)
    scales = torch.empty((groups, output_rows), device="cuda", dtype=torch.float32)
    check_status(
        quantize(
            weight.data_ptr(),
            weight_k.data_ptr(),
            scales.data_ptr(),
            hidden,
            output_rows,
            1,
            stream,
        ),
        "K-major W8 quantization",
    )
    torch.cuda.synchronize()
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260721 + args.rows)
    activation = torch.randn(
        (args.rows, hidden),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    output = torch.empty(
        (args.rows, output_rows), device="cuda", dtype=torch.bfloat16
    )
    reference = torch.mm(activation, weight.T)
    bf16_weights = [weight.clone() for _ in range(4)]
    w8_weights = [weight_k.clone() for _ in range(4)]
    w8_scales = [scales.clone() for _ in range(4)]
    w8_scales_bf16 = [scales.to(torch.bfloat16) for _ in range(4)]
    direct_weights = w8_weights
    direct_scales = w8_scales
    row_major_weight = args.weight_layout == "row-major"
    if row_major_weight:
        weight_row = torch.empty(
            (output_rows, hidden), device="cuda", dtype=torch.int8
        )
        scales_row = torch.empty(
            (output_rows, groups), device="cuda", dtype=torch.float32
        )
        check_status(
            quantize(
                weight.data_ptr(),
                weight_row.data_ptr(),
                scales_row.data_ptr(),
                hidden,
                output_rows,
                0,
                stream,
            ),
            "row-major W8 quantization",
        )
        torch.cuda.synchronize()
        direct_weights = [weight_row.clone() for _ in range(4)]
        direct_scales = [scales_row.clone() for _ in range(4)]
    bf16_scratch = torch.empty_like(weight)

    def expand(slot: int = 0) -> None:
        check_status(
            dequantize(
                w8_weights[slot].data_ptr(),
                w8_scales[slot].data_ptr(),
                bf16_scratch.data_ptr(),
                hidden,
                output_rows,
                stream,
            ),
            "K-major W8 to row-major BF16 dequantization",
        )

    expand()
    torch.cuda.synchronize()
    expanded_result = torch.mm(activation, bf16_scratch.T)
    result = metrics(expanded_result, reference)
    print(
        f"quality rows={args.rows} kernel=dequant-bf16-cublas "
        f"relative_l2={result['relative_l2']:.9f} "
        f"cosine={result['cosine']:.9f} "
        f"max_abs={result['max_abs']:.6f}"
    )

    def launch(
        config: tuple[int, int, int, int, int],
        slot: int = 0,
        *,
        bf16_scale: bool = False,
        post_scale: bool = False,
    ) -> None:
        block_m, block_n, block_k, warps, stages = config
        grid = (triton.cdiv(args.rows, block_m) * triton.cdiv(output_rows, block_n),)
        w8a16_group256_gemm[grid](
            activation,
            direct_weights[slot],
            direct_scales[slot].to(torch.bfloat16)
            if bf16_scale
            else direct_scales[slot],
            output,
            M=args.rows,
            N=output_rows,
            K=hidden,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            DEQUANT_BF16=bf16_scale,
            POST_SCALE_GROUP=post_scale,
            ROW_MAJOR_WEIGHT=row_major_weight,
            num_warps=warps,
            num_stages=stages,
        )

    dequant_modes = (
        (False, False, "pre-fp32-scale"),
        (True, False, "pre-bf16-scale"),
        (False, True, "post-fp32-scale"),
    )
    if args.dequant_mode != "all":
        dequant_modes = tuple(
            mode
            for mode in dequant_modes
            if mode[2].startswith(args.dequant_mode)
        )
    for bf16_scale, post_scale, scale_label in dequant_modes:
        for config in configs:
            launch(
                config,
                bf16_scale=bf16_scale,
                post_scale=post_scale,
            )
            torch.cuda.synchronize()
            result = metrics(output, reference)
            print(
                f"quality rows={args.rows} "
                f"kernel={label(config)}-{scale_label} "
                f"relative_l2={result['relative_l2']:.9f} "
                f"cosine={result['cosine']:.9f} "
                f"max_abs={result['max_abs']:.6f}"
            )

    bf16_warm = bench(
        lambda _: torch.mm(activation, bf16_weights[0].T, out=output),
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    bf16_rotating = bench(
        lambda index: torch.mm(
            activation, bf16_weights[index & 3].T, out=output
        ),
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    report("bf16-cublas-warm", args.rows, bf16_warm)
    report("bf16-cublas-rotating", args.rows, bf16_rotating)

    dequant_warm = bench(
        lambda _: expand(0),
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    dequant_rotating = bench(
        lambda index: expand(index & 3),
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    report("dequant-only-warm", args.rows, dequant_warm)
    report("dequant-only-rotating", args.rows, dequant_rotating)

    def expand_and_project(slot: int) -> None:
        expand(slot)
        torch.mm(activation, bf16_scratch.T, out=output)

    expanded_warm = bench(
        lambda _: expand_and_project(0),
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    expanded_rotating = bench(
        lambda index: expand_and_project(index & 3),
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    report("dequant-bf16-cublas-warm", args.rows, expanded_warm)
    report("dequant-bf16-cublas-rotating", args.rows, expanded_rotating)

    for bf16_scale, post_scale, scale_label in dequant_modes:
        for config in configs:
            warm = bench(
                lambda _, selected=config, use_bf16=bf16_scale,
                use_post=post_scale: launch(
                    selected,
                    0,
                    bf16_scale=use_bf16,
                    post_scale=use_post,
                ),
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            rotating = bench(
                lambda index, selected=config, use_bf16=bf16_scale,
                use_post=post_scale: launch(
                    selected,
                    index & 3,
                    bf16_scale=use_bf16,
                    post_scale=use_post,
                ),
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            report(f"{label(config)}-{scale_label}-warm", args.rows, warm)
            report(
                f"{label(config)}-{scale_label}-rotating", args.rows, rotating
            )


if __name__ == "__main__":
    main()
