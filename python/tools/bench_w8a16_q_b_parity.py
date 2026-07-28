#!/usr/bin/env python3
"""Benchmark recurrent-exact batched W8A16 Q-A/Q-B projections."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
from statistics import median
from typing import Callable

import torch

from tune_w8a16_projection import (
    CATALOG_PATH,
    GROUP_SIZE,
    PROJECTION_TENSORS,
    check_status,
    load_bf16_weight,
    metrics,
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
    simt = native.glmrt_cuda_linear_w8a16_group256_m1_simt_async
    simt.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    )
    simt.restype = ctypes.c_int
    parity = native.glmrt_cuda_linear_w8a16_group256_m1_parity_batched_async
    parity.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    parity.restype = ctypes.c_int
    return quantize, simt, parity


def bench(
    launch: Callable[[int], None], *, warmup: int, iterations: int, repeats: int
) -> list[float]:
    samples: list[float] = []
    for _ in range(repeats):
        for index in range(warmup):
            launch(index)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for index in range(iterations):
            launch(index)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--native-library",
        type=Path,
        default=Path("native/build-cuda-rdma-coordinator-aot/libglmrt_native.so"),
    )
    parser.add_argument("--tensor", choices=("q-a", "q-b"), default="q-b")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if not 2 <= args.rows <= 16:
        parser.error("rows must be in 2..=16")

    with CATALOG_PATH.open() as handle:
        catalog = json.load(handle)
    weight = load_bf16_weight(catalog, PROJECTION_TENSORS[args.tensor])
    output_dim, input_dim = weight.shape
    assert input_dim % GROUP_SIZE == 0
    groups = input_dim // GROUP_SIZE
    quantize, simt, parity = configure_native(args.native_library)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    packed_weight = torch.empty_like(weight, dtype=torch.int8)
    scales = torch.empty((output_dim, groups), device="cuda", dtype=torch.float32)
    check_status(
        quantize(
            weight.data_ptr(),
            packed_weight.data_ptr(),
            scales.data_ptr(),
            input_dim,
            output_dim,
            0,
            stream,
        ),
        f"{args.tensor} W8 quantization",
    )
    torch.cuda.synchronize()

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260722)
    activation = torch.randn(
        (args.rows, input_dim),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    recurrent_output = torch.empty(
        (args.rows, output_dim), device="cuda", dtype=torch.bfloat16
    )
    parity_output = torch.empty_like(recurrent_output)
    bf16_output = torch.empty_like(recurrent_output)

    def launch_recurrent(weight_slot: torch.Tensor, scale_slot: torch.Tensor) -> None:
        for row in range(args.rows):
            check_status(
                simt(
                    activation[row].data_ptr(),
                    weight_slot.data_ptr(),
                    scale_slot.data_ptr(),
                    recurrent_output[row].data_ptr(),
                    input_dim,
                    output_dim,
                    3,
                    stream,
                ),
                f"recurrent {args.tensor} W8 projection",
            )

    def launch_parity(weight_slot: torch.Tensor, scale_slot: torch.Tensor) -> None:
        check_status(
            parity(
                activation.data_ptr(),
                weight_slot.data_ptr(),
                scale_slot.data_ptr(),
                parity_output.data_ptr(),
                args.rows,
                input_dim,
                output_dim,
                stream,
            ),
            f"batched-parity {args.tensor} W8 projection",
        )

    launch_recurrent(packed_weight, scales)
    launch_parity(packed_weight, scales)
    torch.cuda.synchronize()
    mismatch = int((recurrent_output != parity_output).sum())
    quality = metrics(parity_output, activation @ weight.T)
    print(
        f"quality rows={args.rows} mismatch={mismatch} "
        f"relative_l2={quality['relative_l2']:.9f} "
        f"cosine={quality['cosine']:.9f} max_abs={quality['max_abs']:.6f}"
    )
    if mismatch:
        raise RuntimeError(f"batched-parity output differs at {mismatch} elements")

    # Four copies rotate more than the GPU L2 capacity between revisits.
    packed_weights = [packed_weight.clone() for _ in range(4)]
    scale_copies = [scales.clone() for _ in range(4)]
    bf16_weights = [weight.clone() for _ in range(4)]
    cases = {
        "bf16-warm": lambda _: torch.mm(
            activation, bf16_weights[0].T, out=bf16_output
        ),
        "bf16-rotating": lambda index: torch.mm(
            activation, bf16_weights[index & 3].T, out=bf16_output
        ),
        "recurrent-warm": lambda _: launch_recurrent(packed_weights[0], scale_copies[0]),
        "parity-warm": lambda _: launch_parity(packed_weights[0], scale_copies[0]),
        "recurrent-rotating": lambda index: launch_recurrent(
            packed_weights[index & 3], scale_copies[index & 3]
        ),
        "parity-rotating": lambda index: launch_parity(
            packed_weights[index & 3], scale_copies[index & 3]
        ),
    }
    for label, launch in cases.items():
        samples = bench(
            launch,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        print(
            f"timing kernel={label} rows={args.rows} "
            f"median_ms={median(samples):.6f} "
            f"range_ms={min(samples):.6f}-{max(samples):.6f}"
        )


if __name__ == "__main__":
    main()
