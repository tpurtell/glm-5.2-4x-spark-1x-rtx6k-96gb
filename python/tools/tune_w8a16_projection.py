#!/usr/bin/env python3
"""Tune native group-256 W8A16 M=1 kernels on real GLM projection weights."""

from __future__ import annotations

import argparse
import ctypes
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Callable

import numpy as np
import torch


GROUP_SIZE = 256
CATALOG_PATH = Path(".glmrt-cache/model-artifacts/diagnostic/model_catalog.json")
DEFAULT_TENSORS = (
    "model.layers.0.self_attn.q_b_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
)
PROJECTION_TENSORS = {
    "q-a": "model.layers.0.self_attn.q_a_proj.weight",
    "q-b": DEFAULT_TENSORS[0],
    "o": DEFAULT_TENSORS[1],
}
EXTRA_TENSORS = {
    "dsa-weights": "model.layers.0.self_attn.indexer.weights_proj.weight",
    "dsa-wq-b": "model.layers.0.self_attn.indexer.wq_b.weight",
    "lm-head": "lm_head.weight",
    "mtp-eh": "model.layers.78.eh_proj.weight",
    "router": "model.layers.3.mlp.gate.weight",
    "shared-down": "model.layers.3.mlp.shared_experts.down_proj.weight",
    "shared-gate": "model.layers.3.mlp.shared_experts.gate_proj.weight",
    "shared-up": "model.layers.3.mlp.shared_experts.up_proj.weight",
}
SIMT_VARIANTS = {
    0: "simt-r1-w4-cache",
    1: "simt-r2-w4-cache",
    2: "simt-r4-w4-cache",
    3: "simt-r1-w4-nc",
    4: "simt-r2-w4-nc",
    5: "simt-r4-w4-nc",
    6: "simt-r2-w8-cache",
    7: "simt-r4-w8-cache",
    8: "simt-r2-w8-nc",
    9: "simt-r4-w8-nc",
    10: "simt-r1-w8-cache",
    11: "simt-r1-w8-nc",
    12: "simt-r1-w4-shared-cache",
    13: "simt-r1-w4-shared-nc",
    14: "simt-r1-w8-shared-cache",
    15: "simt-r1-w8-shared-nc",
}


@dataclass(frozen=True)
class Timing:
    median_ms: float
    minimum_ms: float
    maximum_ms: float


def tensor_memmap(catalog: dict, tensor: dict) -> np.memmap:
    return np.memmap(
        Path(catalog["snapshot_path"]) / tensor["file"],
        dtype=np.uint16,
        mode="r",
        offset=tensor["byte_offset"],
        shape=tuple(tensor["shape"]),
    )


def load_bf16_weight(catalog: dict, name: str) -> torch.Tensor:
    tensor = next(item for item in catalog["tensors"] if item["name"] == name)
    raw = tensor_memmap(catalog, tensor)
    # NumPy cannot construct a BF16 tensor directly. Preserve the checkpoint
    # bits through int16 and reinterpret them after the CPU-to-GPU copy.
    return (
        torch.from_numpy(np.array(raw, copy=True).view(np.int16))
        .cuda()
        .view(torch.bfloat16)
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
    return quantize, simt


def check_status(status: int, operation: str) -> None:
    if status != 0:
        raise RuntimeError(f"{operation} failed with native status {status}")


def elapsed_ms(operation: Callable[[], None], *, repetitions: int = 1) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repetitions


def bench(
    launch: Callable[[int], None],
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> Timing:
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
    return Timing(median(samples), min(samples), max(samples))


def quality_inputs(hidden: int) -> list[tuple[str, torch.Tensor]]:
    inputs: list[tuple[str, torch.Tensor]] = []
    for seed in range(4):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        value = torch.randn(
            hidden, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        inputs.append((f"normal-{seed}", value))

    generator = torch.Generator(device="cuda")
    generator.manual_seed(17)
    uniform = (
        torch.rand(
            hidden, generator=generator, device="cuda", dtype=torch.float32
        )
        * 2.0
        - 1.0
    ).to(torch.bfloat16)
    inputs.append(("uniform", uniform))

    generator.manual_seed(29)
    outliers = torch.randn(
        hidden, generator=generator, device="cuda", dtype=torch.float32
    )
    outliers[::257] *= 16.0
    inputs.append(("outlier", outliers.to(torch.bfloat16)))

    one_hot = torch.zeros(hidden, device="cuda", dtype=torch.bfloat16)
    one_hot[min(977, hidden - 1)] = 1.0
    inputs.append(("one-hot", one_hot))
    return inputs


def metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    candidate_f32 = candidate.float().reshape(-1)
    reference_f32 = reference.float().reshape(-1)
    difference = candidate_f32 - reference_f32
    relative_l2 = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(
        reference_f32
    )
    cosine = torch.nn.functional.cosine_similarity(
        candidate_f32, reference_f32, dim=0
    )
    return {
        "relative_l2": float(relative_l2),
        "cosine": float(cosine),
        "max_abs": float(difference.abs().max()),
    }


def summarize_quality(
    label: str,
    launch: Callable[[torch.Tensor], torch.Tensor],
    weight: torch.Tensor,
) -> dict[str, float]:
    results = []
    deterministic = True
    for input_label, activation in quality_inputs(weight.shape[1]):
        reference = torch.mv(weight, activation)
        candidate = launch(activation).clone()
        repeated = launch(activation).clone()
        torch.cuda.synchronize()
        deterministic &= torch.equal(candidate, repeated)
        result = metrics(candidate, reference)
        results.append(result)
        print(
            "quality "
            f"kernel={label} input={input_label} "
            f"relative_l2={result['relative_l2']:.9f} "
            f"cosine={result['cosine']:.9f} "
            f"max_abs={result['max_abs']:.6f}"
        )
    summary = {
        "max_relative_l2": max(item["relative_l2"] for item in results),
        "min_cosine": min(item["cosine"] for item in results),
        "max_abs": max(item["max_abs"] for item in results),
        "deterministic": float(deterministic),
    }
    print(
        "quality_summary "
        f"kernel={label} "
        f"max_relative_l2={summary['max_relative_l2']:.9f} "
        f"min_cosine={summary['min_cosine']:.9f} "
        f"max_abs={summary['max_abs']:.6f} "
        f"deterministic={deterministic}"
    )
    return summary


def run_tensor(
    *,
    name: str,
    weight: torch.Tensor,
    quantize,
    simt,
    warmup: int,
    iterations: int,
    repeats: int,
) -> None:
    output_rows, hidden = weight.shape
    if hidden % GROUP_SIZE != 0 or output_rows % 32 != 0:
        raise ValueError(f"unsupported W8 shape {tuple(weight.shape)}")
    groups = hidden // GROUP_SIZE
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    weight_row = torch.empty(
        (output_rows, hidden), device="cuda", dtype=torch.int8
    )
    scale_row = torch.empty(
        (output_rows, groups), device="cuda", dtype=torch.float32
    )
    def quantize_row() -> None:
        check_status(
            quantize(
                weight.data_ptr(),
                weight_row.data_ptr(),
                scale_row.data_ptr(),
                hidden,
                output_rows,
                0,
                stream,
            ),
            "row-major W8 quantization",
        )

    row_quantize_ms = elapsed_ms(quantize_row)
    torch.cuda.synchronize()
    print(
        f"tensor={name} shape={tuple(weight.shape)} "
        f"bf16_MB={weight.nbytes / 1e6:.3f} "
        f"w8_MB={(weight_row.nbytes + scale_row.nbytes) / 1e6:.3f} "
        f"row_quantize_ms={row_quantize_ms:.3f}"
    )

    output = torch.empty(output_rows, device="cuda", dtype=torch.bfloat16)

    def launch_simt(variant: int, activation: torch.Tensor) -> torch.Tensor:
        check_status(
            simt(
                activation.data_ptr(),
                weight_row.data_ptr(),
                scale_row.data_ptr(),
                output.data_ptr(),
                hidden,
                output_rows,
                variant,
                stream,
            ),
            f"SIMT W8A16 variant {variant}",
        )
        return output

    # Every launch geometry must agree before timing.
    validation_input = quality_inputs(hidden)[0][1]
    simt_reference = launch_simt(0, validation_input).clone()
    torch.cuda.synchronize()
    for variant, label in SIMT_VARIANTS.items():
        candidate = launch_simt(variant, validation_input).clone()
        torch.cuda.synchronize()
        if not torch.equal(candidate, simt_reference):
            mismatch = int((candidate != simt_reference).sum())
            raise RuntimeError(f"{label} differs from variant 0 at {mismatch} outputs")

    summarize_quality(
        "simt-group256", lambda activation: launch_simt(0, activation), weight
    )
    # Four copies exceed the coordinator's L2 residency for both real shapes.
    # Warm results repeatedly use copy zero; rotating results revisit a copy
    # only after the other three matrices have streamed.
    bf16_weights = [weight.clone() for _ in range(4)]
    row_weights = [weight_row.clone() for _ in range(4)]
    row_scales = [scale_row.clone() for _ in range(4)]
    activation = validation_input

    def report(label: str, timing: Timing, weight_bytes: int) -> None:
        effective_gbps = weight_bytes / timing.median_ms / 1e6
        print(
            f"timing tensor={name} kernel={label} "
            f"median_ms={timing.median_ms:.6f} "
            f"range_ms={timing.minimum_ms:.6f}-{timing.maximum_ms:.6f} "
            f"effective_weight_GBps={effective_gbps:.1f}"
        )

    bf16_warm = bench(
        lambda _: torch.mv(bf16_weights[0], activation, out=output),
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )
    bf16_cold = bench(
        lambda index: torch.mv(bf16_weights[index & 3], activation, out=output),
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )
    report("bf16-cublas-warm", bf16_warm, weight.nbytes)
    report("bf16-cublas-rotating", bf16_cold, weight.nbytes)

    for variant, label in SIMT_VARIANTS.items():
        def launch_warm(_: int, selected: int = variant) -> None:
            check_status(
                simt(
                    activation.data_ptr(),
                    row_weights[0].data_ptr(),
                    row_scales[0].data_ptr(),
                    output.data_ptr(),
                    hidden,
                    output_rows,
                    selected,
                    stream,
                ),
                label,
            )

        def launch_rotating(index: int, selected: int = variant) -> None:
            slot = index & 3
            check_status(
                simt(
                    activation.data_ptr(),
                    row_weights[slot].data_ptr(),
                    row_scales[slot].data_ptr(),
                    output.data_ptr(),
                    hidden,
                    output_rows,
                    selected,
                    stream,
                ),
                label,
            )

        warm = bench(
            launch_warm,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        )
        cold = bench(
            launch_rotating,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        )
        report(f"{label}-warm", warm, weight_row.nbytes + scale_row.nbytes)
        report(f"{label}-rotating", cold, weight_row.nbytes + scale_row.nbytes)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--native-library",
        type=Path,
        default=Path("native/build-w8a16/libglmrt_native.so"),
    )
    parser.add_argument(
        "--tensor",
        action="append",
        choices=(*PROJECTION_TENSORS, *EXTRA_TENSORS),
        help="projection to test; omit to run both",
    )
    parser.add_argument("--warmup", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=240)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.repeats <= 0:
        parser.error("warmup must be nonnegative; iterations/repeats must be positive")

    with CATALOG_PATH.open() as handle:
        catalog = json.load(handle)
    quantize, simt = configure_native(args.native_library)
    selected = args.tensor or ["q-b", "o"]
    names = {**PROJECTION_TENSORS, **EXTRA_TENSORS}
    for projection in selected:
        name = names[projection]
        weight = load_bf16_weight(catalog, name)
        run_tensor(
            name=name,
            weight=weight,
            quantize=quantize,
            simt=simt,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        del weight
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
