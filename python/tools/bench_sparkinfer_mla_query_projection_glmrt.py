#!/usr/bin/env python3
"""Benchmark SparkInfer MLA query projection against GLMRT's native routes.

The candidate uses four non-copying H=16 views of GLM-5's H=64 tensors.  The
two controls reproduce the native launch structures used by GLMRT:

* ``decode``: one 64-way strided-batched cuBLAS GEMM per query row followed by
  the two cudaMemcpy2D assembly nodes used by packed-FP8 MLA decode.
* ``prefill``: token/head transpose, one 64-way strided-batched cuBLAS GEMM,
  and the native absorbed-query composition kernel.

All arms use the same token-major BF16 query, the same sole resident
``[H,448,512]`` KV-B allocation, caller-owned outputs, and CUDA-graph replay.
This is a performance-gating tool; it does not change serving policy.
"""

from __future__ import annotations

import _pinned_sparkinfer

import argparse
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
from typing import Callable, Literal

import torch

from sparkinfer.gemm import mla_query_projection


HEADS = 64
HEAD_GROUP = 16
NOPE_DIM = 192
KV_B_ROWS = 448
LATENT_DIM = 512
ROPE_DIM = 64
QUERY_DIM = LATENT_DIM + ROPE_DIM
BF16_BYTES = 2
DEFAULT_M_VALUES = (1, 2, 4, 8, 16, 32)


class DeviceBuffer(ctypes.Structure):
    _fields_ = (
        ("ptr", ctypes.c_void_p),
        ("bytes", ctypes.c_size_t),
        ("device_id", ctypes.c_int),
        ("flags", ctypes.c_uint64),
    )


@dataclass(frozen=True)
class NativeApi:
    library: ctypes.CDLL
    matmul: object
    copy_2d: object
    transpose: object
    compose: object


@dataclass(frozen=True)
class Timing:
    samples_us: tuple[float, ...]

    def as_dict(self) -> dict[str, object]:
        ordered = sorted(self.samples_us)
        return {
            "median_us": statistics.median(ordered),
            "mean_us": statistics.mean(ordered),
            "min_us": ordered[0],
            "max_us": ordered[-1],
            "samples_us": list(self.samples_us),
        }


def parse_int_csv(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(dict.fromkeys(int(part) for part in value.split(",") if part))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--native-library",
        type=Path,
        default=Path("native/build-cuda-sparkinfer-routed/libglmrt_native.so"),
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--m-values", type=parse_int_csv, default=DEFAULT_M_VALUES)
    parser.add_argument(
        "--routes",
        choices=("decode", "prefill", "both"),
        default="both",
        help="native launch structure(s) to compare",
    )
    parser.add_argument(
        "--weight-count",
        type=int,
        default=1,
        help="resident KV-B tensors cycled inside each graph (use 79 for all layers)",
    )
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(m < 1 or m > 32 for m in args.m_values):
        parser.error("--m-values must be in [1,32]")
    if args.weight_count < 1 or args.weight_count > 79:
        parser.error("--weight-count must be in [1,79]")
    if args.warmup < 1 or args.iterations < 1 or args.repeats < 3:
        parser.error("--warmup/--iterations must be positive and --repeats >= 3")
    if not args.native_library.is_file():
        parser.error(f"native library does not exist: {args.native_library}")
    return args


def configure_native(path: Path) -> NativeApi:
    library = ctypes.CDLL(str(path.resolve()))
    pointer = ctypes.c_void_p
    size = ctypes.c_size_t

    matmul = library.glmrt_cuda_matmul_bf16_strided_batched_cublas_async
    matmul.argtypes = (
        pointer,
        pointer,
        pointer,
        size,
        size,
        size,
        size,
        size,
        size,
        size,
        pointer,
    )
    matmul.restype = ctypes.c_int

    copy_2d = library.glmrt_copy_d2d_2d_async
    copy_2d.argtypes = (
        DeviceBuffer,
        size,
        DeviceBuffer,
        size,
        size,
        size,
        pointer,
    )
    copy_2d.restype = ctypes.c_int

    transpose = library.glmrt_cuda_transpose_rows_heads_bf16_async
    transpose.argtypes = (pointer, pointer, size, size, size, pointer)
    transpose.restype = ctypes.c_int

    compose = library.glmrt_cuda_mla_compose_absorbed_query_bf16_async
    compose.argtypes = (
        pointer,
        pointer,
        pointer,
        size,
        size,
        size,
        size,
        pointer,
    )
    compose.restype = ctypes.c_int
    return NativeApi(library, matmul, copy_2d, transpose, compose)


def check_status(api: NativeApi, status: int, action: str) -> None:
    if status == 0:
        return
    message = ctypes.create_string_buffer(1024)
    api.library.glmrt_last_error(message, len(message))
    raise RuntimeError(
        f"{action} failed with status={status}: "
        f"{message.value.decode(errors='replace')}"
    )


def pointer(tensor: torch.Tensor, element_offset: int = 0) -> ctypes.c_void_p:
    return ctypes.c_void_p(
        tensor.data_ptr() + element_offset * tensor.element_size()
    )


def device_buffer_span(
    tensor: torch.Tensor, *, byte_offset: int = 0, byte_span: int | None = None
) -> DeviceBuffer:
    total_bytes = tensor.numel() * tensor.element_size()
    if byte_offset < 0 or byte_offset > total_bytes:
        raise ValueError(f"invalid byte offset {byte_offset} for {total_bytes}-byte tensor")
    remaining = total_bytes - byte_offset
    span = remaining if byte_span is None else byte_span
    if span < 0 or span > remaining:
        raise ValueError(
            f"invalid byte span {span} at offset {byte_offset} "
            f"for {total_bytes}-byte tensor"
        )
    return DeviceBuffer(
        tensor.data_ptr() + byte_offset,
        span,
        tensor.device.index or 0,
        0,
    )


def stream_pointer() -> ctypes.c_void_p:
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def make_decode_launch(
    api: NativeApi,
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    weight: torch.Tensor,
    projected: torch.Tensor,
    out: torch.Tensor,
) -> Callable[[], None]:
    m = int(q_nope.shape[0])
    weight_head_stride = KV_B_ROWS * LATENT_DIM
    q_row_elements = HEADS * NOPE_DIM
    projected_row_elements = HEADS * LATENT_DIM
    output_pitch_bytes = QUERY_DIM * BF16_BYTES
    latent_pitch_bytes = LATENT_DIM * BF16_BYTES
    rope_pitch_bytes = ROPE_DIM * BF16_BYTES
    copy_rows = m * HEADS
    out_span = (copy_rows - 1) * output_pitch_bytes + latent_pitch_bytes
    rope_out_offset = LATENT_DIM * BF16_BYTES
    rope_out_span = (copy_rows - 1) * output_pitch_bytes + rope_pitch_bytes
    projected_span = (copy_rows - 1) * latent_pitch_bytes + latent_pitch_bytes
    rope_span = (copy_rows - 1) * rope_pitch_bytes + rope_pitch_bytes
    latent_dst = device_buffer_span(out, byte_span=out_span)
    latent_src = device_buffer_span(projected, byte_span=projected_span)
    rope_dst = device_buffer_span(
        out, byte_offset=rope_out_offset, byte_span=rope_out_span
    )
    rope_src = device_buffer_span(q_rope, byte_span=rope_span)

    def launch() -> None:
        stream = stream_pointer()
        for row in range(m):
            check_status(
                api,
                api.matmul(
                    pointer(q_nope, row * q_row_elements),
                    pointer(weight),
                    pointer(projected, row * projected_row_elements),
                    HEADS,
                    1,
                    NOPE_DIM,
                    LATENT_DIM,
                    NOPE_DIM,
                    weight_head_stride,
                    LATENT_DIM,
                    stream,
                ),
                f"decode q projection row={row}",
            )
        check_status(
            api,
            api.copy_2d(
                latent_dst,
                output_pitch_bytes,
                latent_src,
                latent_pitch_bytes,
                latent_pitch_bytes,
                copy_rows,
                stream,
            ),
            "decode latent cudaMemcpy2D assembly",
        )
        check_status(
            api,
            api.copy_2d(
                rope_dst,
                output_pitch_bytes,
                rope_src,
                rope_pitch_bytes,
                rope_pitch_bytes,
                copy_rows,
                stream,
            ),
            "decode RoPE cudaMemcpy2D assembly",
        )

    return launch


def make_prefill_launch(
    api: NativeApi,
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    weight: torch.Tensor,
    transposed: torch.Tensor,
    projected_head_major: torch.Tensor,
    out: torch.Tensor,
) -> Callable[[], None]:
    m = int(q_nope.shape[0])
    weight_head_stride = KV_B_ROWS * LATENT_DIM

    def launch() -> None:
        stream = stream_pointer()
        check_status(
            api,
            api.transpose(
                pointer(q_nope),
                pointer(transposed),
                m,
                HEADS,
                NOPE_DIM,
                stream,
            ),
            "prefill q transpose",
        )
        check_status(
            api,
            api.matmul(
                pointer(transposed),
                pointer(weight),
                pointer(projected_head_major),
                HEADS,
                m,
                NOPE_DIM,
                LATENT_DIM,
                m * NOPE_DIM,
                weight_head_stride,
                m * LATENT_DIM,
                stream,
            ),
            "prefill q projection",
        )
        check_status(
            api,
            api.compose(
                pointer(projected_head_major),
                pointer(q_rope),
                pointer(out),
                m,
                HEADS,
                LATENT_DIM,
                ROPE_DIM,
                stream,
            ),
            "prefill absorbed-query composition",
        )

    return launch


def make_candidate_launch(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
) -> Callable[[], None]:
    # These are metadata-only views.  No tensor is cloned or materialized.
    q_head_major = q_nope.permute(1, 0, 2)
    k_weight = weight[:, :NOPE_DIM, :]
    groups = tuple(
        (
            q_head_major[head : head + HEAD_GROUP],
            k_weight[head : head + HEAD_GROUP],
            q_rope[:, head : head + HEAD_GROUP, :],
            out[:, head : head + HEAD_GROUP, :],
        )
        for head in range(0, HEADS, HEAD_GROUP)
    )

    def launch() -> None:
        for q_group, weight_group, rope_group, out_group in groups:
            mla_query_projection.run(q_group, weight_group, rope_group, out_group)

    return launch


def capture_graph(launch: Callable[[], None]) -> torch.cuda.CUDAGraph:
    for _ in range(3):
        launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def time_graphs(
    control: torch.cuda.CUDAGraph,
    candidate: torch.cuda.CUDAGraph,
    *,
    operations_per_replay: int,
    warmup: int,
    iterations: int,
    repeats: int,
) -> tuple[Timing, Timing]:
    for index in range(warmup):
        (control if index % 2 == 0 else candidate).replay()
        (candidate if index % 2 == 0 else control).replay()
    torch.cuda.synchronize()
    samples: dict[str, list[float]] = {"control": [], "candidate": []}

    def one(label: str, graph: torch.cuda.CUDAGraph) -> None:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        per_operation_us = (
            start.elapsed_time(end) * 1000.0 / iterations / operations_per_replay
        )
        samples[label].append(per_operation_us)

    for repeat in range(repeats):
        order = (
            (("control", control), ("candidate", candidate))
            if repeat % 2 == 0
            else (("candidate", candidate), ("control", control))
        )
        for label, graph in order:
            one(label, graph)
    return Timing(tuple(samples["control"])), Timing(tuple(samples["candidate"]))


def tensor_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, object]:
    ref = reference.float()
    got = actual.float()
    difference = got - ref
    ref_flat = ref.reshape(-1)
    got_flat = got.reshape(-1)
    denominator = float(torch.linalg.vector_norm(ref_flat).item()) * float(
        torch.linalg.vector_norm(got_flat).item()
    )
    cosine = (
        float(torch.dot(ref_flat, got_flat).item()) / denominator
        if denominator
        else float("nan")
    )
    return {
        "bitwise_equal": bool(
            torch.equal(reference.view(torch.uint8), actual.view(torch.uint8))
        ),
        "allclose_rtol_2e-2_atol_2e-2": bool(
            torch.allclose(ref, got, rtol=2e-2, atol=2e-2)
        ),
        "cosine": cosine,
        "rmse": float(torch.sqrt(torch.mean(difference.square())).item()),
        "mean_abs": float(difference.abs().mean().item()),
        "max_abs": float(difference.abs().max().item()),
        "exact_fraction": float((ref == got).float().mean().item()),
        "finite": bool(torch.isfinite(got).all().item()),
        "nonzero": bool(torch.count_nonzero(actual).item()),
    }


def graph_replay_checks(
    graph: torch.cuda.CUDAGraph,
    q_nope: torch.Tensor,
    out: torch.Tensor,
) -> dict[str, object]:
    graph.replay()
    torch.cuda.synchronize()
    first = out.clone()
    q_nope.add_(torch.tensor(0.03125, dtype=q_nope.dtype, device=q_nope.device))
    graph.replay()
    torch.cuda.synchronize()
    second = out.clone()
    q_nope.sub_(torch.tensor(0.03125, dtype=q_nope.dtype, device=q_nope.device))
    before = torch.cuda.memory_allocated(q_nope.device)
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated(q_nope.device)
    return {
        "fresh_input_observed": not torch.equal(
            first.view(torch.uint8), second.view(torch.uint8)
        ),
        "torch_allocation_delta_bytes_two_replays": after - before,
        "fixed_output_pointer": out.data_ptr(),
    }


def benchmark_case(
    api: NativeApi,
    *,
    route: Literal["decode", "prefill"],
    m: int,
    weights: torch.Tensor,
    generator: torch.Generator,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, object]:
    q_nope = torch.randn(
        (m, HEADS, NOPE_DIM),
        dtype=torch.bfloat16,
        device=weights.device,
        generator=generator,
    )
    q_rope = torch.randn(
        (m, HEADS, ROPE_DIM),
        dtype=torch.bfloat16,
        device=weights.device,
        generator=generator,
    )
    control_out = torch.empty(
        (m, HEADS, QUERY_DIM), dtype=torch.bfloat16, device=weights.device
    )
    candidate_out = torch.empty_like(control_out)
    projected = torch.empty(
        (m, HEADS, LATENT_DIM), dtype=torch.bfloat16, device=weights.device
    )
    transposed = torch.empty(
        (HEADS, m, NOPE_DIM), dtype=torch.bfloat16, device=weights.device
    )
    projected_head_major = torch.empty(
        (HEADS, m, LATENT_DIM), dtype=torch.bfloat16, device=weights.device
    )

    control_launches: list[Callable[[], None]] = []
    candidate_launches: list[Callable[[], None]] = []
    per_weight_metrics: list[dict[str, object]] = []
    for index, weight in enumerate(weights):
        control = (
            make_decode_launch(api, q_nope, q_rope, weight, projected, control_out)
            if route == "decode"
            else make_prefill_launch(
                api,
                q_nope,
                q_rope,
                weight,
                transposed,
                projected_head_major,
                control_out,
            )
        )
        candidate = make_candidate_launch(q_nope, q_rope, weight, candidate_out)
        control()
        candidate()
        torch.cuda.synchronize()
        full_metrics = tensor_metrics(control_out, candidate_out)
        rope_equal = bool(
            torch.equal(
                control_out[..., LATENT_DIM:].view(torch.uint8),
                candidate_out[..., LATENT_DIM:].view(torch.uint8),
            )
        )
        latent_metrics = tensor_metrics(
            control_out[..., :LATENT_DIM], candidate_out[..., :LATENT_DIM]
        )
        per_weight_metrics.append(
            {
                "weight_index": index,
                "full": full_metrics,
                "latent": latent_metrics,
                "rope_suffix_bitwise_equal": rope_equal,
            }
        )
        control_launches.append(control)
        candidate_launches.append(candidate)

    def control_cycle() -> None:
        for launch in control_launches:
            launch()

    def candidate_cycle() -> None:
        for launch in candidate_launches:
            launch()

    control_graph = capture_graph(control_cycle)
    candidate_graph = capture_graph(candidate_cycle)
    allocation_before = torch.cuda.memory_allocated(weights.device)
    control_graph.replay()
    candidate_graph.replay()
    torch.cuda.synchronize()
    allocation_after = torch.cuda.memory_allocated(weights.device)
    replay_checks = graph_replay_checks(candidate_graph, q_nope, candidate_out)
    control_timing, candidate_timing = time_graphs(
        control_graph,
        candidate_graph,
        operations_per_replay=len(weights),
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )
    control_median = statistics.median(control_timing.samples_us)
    candidate_median = statistics.median(candidate_timing.samples_us)
    all_micro_close = all(
        bool(item["full"]["allclose_rtol_2e-2_atol_2e-2"])
        and bool(item["full"]["finite"])
        and bool(item["full"]["nonzero"])
        and bool(item["rope_suffix_bitwise_equal"])
        for item in per_weight_metrics
    )
    return {
        "route": route,
        "m": m,
        "weight_count": len(weights),
        "layout": {
            "q_nope_shape": list(q_nope.shape),
            "q_nope_stride": list(q_nope.stride()),
            "candidate_q_head_major_stride": list(
                q_nope.permute(1, 0, 2).stride()
            ),
            "resident_kv_b_shape": list(weights.shape[1:]),
            "resident_kv_b_stride": list(weights[0].stride()),
            "candidate_k_weight_stride": list(
                weights[0, :, :NOPE_DIM, :].stride()
            ),
            "candidate_output_stride": list(candidate_out.stride()),
            "materialized_candidate_views": False,
        },
        "correctness": {
            "micro_gate_passed": all_micro_close,
            "trajectory_quality_gate_still_required": True,
            "per_weight": per_weight_metrics,
        },
        "graph": {
            "captured": True,
            "torch_allocation_delta_bytes_both_replays": allocation_after
            - allocation_before,
            "candidate": replay_checks,
        },
        "control": control_timing.as_dict(),
        "candidate": candidate_timing.as_dict(),
        "candidate_over_control": candidate_median / control_median,
        "speedup_percent": (control_median / candidate_median - 1.0) * 100.0,
        "ratio_direction": "candidate_median_us / control_median_us; lower is better",
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_snapshot() -> list[str]:
    fields = (
        "index,name,uuid,pstate,clocks.current.sm,clocks.current.memory,"
        "power.limit,clocks_throttle_reasons.active"
    )
    try:
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return output.strip().splitlines()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    if torch.cuda.get_device_capability(device) not in ((12, 0), (12, 1)):
        raise RuntimeError("SparkInfer MLA query projection requires SM120/SM121")
    api = configure_native(args.native_library)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # One tensor is one resident allocation containing all requested layer
    # weights.  The unused V rows remain in place, exactly as in GLMRT.
    weights = (
        torch.randn(
            (args.weight_count, HEADS, KV_B_ROWS, LATENT_DIM),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        * 0.02
    )
    warm_weight = weights[0, :HEAD_GROUP, :NOPE_DIM, :]
    mla_query_projection.prewarm(
        warm_weight,
        args.m_values,
        output_dtype=torch.bfloat16,
    )

    routes: tuple[Literal["decode", "prefill"], ...]
    if args.routes == "both":
        routes = ("decode", "prefill")
    else:
        routes = (args.routes,)
    cases: list[dict[str, object]] = []
    for route in routes:
        for m in args.m_values:
            if route == "decode" and m > 16:
                continue
            cases.append(
                benchmark_case(
                    api,
                    route=route,
                    m=m,
                    weights=weights,
                    generator=generator,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    repeats=args.repeats,
                )
            )
            gc.collect()
            torch.cuda.empty_cache()

    native_path = args.native_library.resolve()
    result = {
        "schema": "glmrt-sparkinfer-mla-query-projection-benchmark-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "worktree": str(Path(__file__).resolve().parents[2]),
        "sparkinfer_source": str(_pinned_sparkinfer.SOURCE),
        "sparkinfer_revision": _pinned_sparkinfer.REVISION,
        "sparkinfer_version": _pinned_sparkinfer.VERSION,
        "sparkinfer_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=_pinned_sparkinfer.SOURCE,
                text=True,
            ).strip()
        ),
        "native_library": str(native_path),
        "native_library_sha256": sha256(native_path),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "gpu_snapshot": gpu_snapshot(),
        "weight_count": args.weight_count,
        "resident_weight_bytes": weights.numel() * weights.element_size(),
        "single_resident_kv_b_allocation": True,
        "cases": cases,
        "decision_contract": {
            "micro_correctness_is_not_quality_acceptance": True,
            "promote_only_winning_m_buckets": True,
            "requires_live_trajectory_and_prefill_non_regression": True,
            "requires_final_clean_locked_sparkinfer_source": True,
        },
    }
    encoded = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
