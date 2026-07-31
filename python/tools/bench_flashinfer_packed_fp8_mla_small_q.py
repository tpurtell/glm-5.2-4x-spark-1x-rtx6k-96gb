#!/usr/bin/env python3
"""Compare recurrent and exact batched FlashInfer packed-FP8 MLA decode."""

from __future__ import annotations

import _pinned_sparkinfer  # noqa: F401

import argparse
import ctypes
import math
import statistics
from collections.abc import Callable
from pathlib import Path

import torch
from sparkinfer.attention._shared.mla.reference import (
    pack_mla_kv_cache_reference,
)
from flashinfer.mla._sparse_mla_sm120 import (
    sparse_mla_sm120_decode_dsv3_2,
)


RANK = 512
ROPE_DIM = 64
PAGE_ROWS = 64
GLM_NSA_MODEL_TYPE = 2


def exact_grouped_chunks(rows: int, bucket_rows: int) -> int | None:
    if bucket_rows == 1024:
        if 3 <= rows <= 5:
            return 2
        if 6 <= rows <= 8:
            return 3
    if bucket_rows == 2048:
        if 2 <= rows <= 3:
            return 2
        if rows == 4 or 7 <= rows <= 8:
            return 3
        if 5 <= rows <= 6:
            return 4
    return None


def load_exact_grouped(path: Path):
    if not path.is_file():
        return None
    library = ctypes.CDLL(str(path.resolve()))
    function = library.glmrt_cuda_packed_fp8_mla_exact_grouped_async
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    function.restype = ctypes.c_int
    return function


def comma_ints(value: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",") if item]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def capture(launch: Callable[[], None]) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    return graph


def measure(
    graph: torch.cuda.CUDAGraph,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=comma_ints, default=[2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--buckets", type=comma_ints, default=[128, 512, 1024, 2048])
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument(
        "--active-rows",
        type=int,
        default=800,
        help="largest causal length (clamped to each bucket)",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--native-library",
        type=Path,
        default=Path("native/build-cuda-rdma-coordinator-aot/libglmrt_native.so"),
    )
    args = parser.parse_args()
    exact_grouped = load_exact_grouped(args.native_library)
    if any(rows < 2 or rows > 16 for rows in args.rows):
        parser.error("--rows values must be in 2..=16")
    if any(bucket not in (128, 512, 1024, 2048) for bucket in args.buckets):
        parser.error("--buckets values must be 128, 512, 1024, or 2048")
    if args.heads not in (16, 64):
        parser.error("--heads must be 16 or 64")

    generator = torch.Generator(device="cuda")
    generator.manual_seed(123)
    scale = 1.0 / math.sqrt(RANK + ROPE_DIM)
    max_rows = max(args.rows)
    for bucket_rows in args.buckets:
        splits = bucket_rows // PAGE_ROWS
        q = torch.randn(
            (max_rows, args.heads, RANK + ROPE_DIM),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        k_nope = torch.randn(
            (bucket_rows, RANK),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        k_rope = torch.randn(
            (bucket_rows, ROPE_DIM),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        kv = pack_mla_kv_cache_reference(k_nope, k_rope).view(
            bucket_rows // PAGE_ROWS, PAGE_ROWS, 656
        )
        indices = torch.arange(
            bucket_rows, dtype=torch.int32, device="cuda"
        ).repeat(max_rows, 1)
        active_rows = min(bucket_rows, max(max_rows, args.active_rows))
        base_length = active_rows - max_rows
        lengths = torch.arange(
            base_length + 1,
            base_length + max_rows + 1,
            dtype=torch.int32,
            device="cuda",
        ).clamp(max=bucket_rows)

        for rows in args.rows:
            q_view = q[:rows]
            indices_view = indices[:rows]
            lengths_view = lengths[:rows]

            def allocate() -> tuple[torch.Tensor, ...]:
                return (
                    torch.empty(
                        (rows, args.heads, RANK),
                        dtype=torch.bfloat16,
                        device="cuda",
                    ),
                    torch.empty(
                        (rows, args.heads), dtype=torch.float32, device="cuda"
                    ),
                    torch.empty(
                        (rows, args.heads, splits, RANK),
                        dtype=torch.bfloat16,
                        device="cuda",
                    ),
                    torch.empty(
                        (rows, args.heads, splits),
                        dtype=torch.float32,
                        device="cuda",
                    ),
                )

            recurrent_out, recurrent_lse, recurrent_mid, recurrent_mid_lse = allocate()
            parity_out, parity_lse, parity_mid, parity_mid_lse = allocate()
            automatic_out, automatic_lse, automatic_mid, automatic_mid_lse = allocate()
            grouped_chunks = (
                exact_grouped_chunks(rows, bucket_rows)
                if args.heads == 64
                else None
            )
            grouped_buffers = allocate() if exact_grouped is not None and grouped_chunks else None

            def recurrent() -> None:
                for row in range(rows):
                    sparse_mla_sm120_decode_dsv3_2(
                        q_view[row : row + 1],
                        kv,
                        indices_view[row : row + 1],
                        recurrent_mid[row : row + 1],
                        recurrent_mid_lse[row : row + 1],
                        recurrent_out[row : row + 1],
                        recurrent_lse[row : row + 1],
                        scale,
                        topk_length=lengths_view[row : row + 1],
                        model_type=GLM_NSA_MODEL_TYPE,
                        chunks_per_block=1,
                    )

            def parity() -> None:
                sparse_mla_sm120_decode_dsv3_2(
                    q_view,
                    kv,
                    indices_view,
                    parity_mid,
                    parity_mid_lse,
                    parity_out,
                    parity_lse,
                    scale,
                    topk_length=lengths_view,
                    model_type=GLM_NSA_MODEL_TYPE,
                    chunks_per_block=1,
                )

            def automatic() -> None:
                sparse_mla_sm120_decode_dsv3_2(
                    q_view,
                    kv,
                    indices_view,
                    automatic_mid,
                    automatic_mid_lse,
                    automatic_out,
                    automatic_lse,
                    scale,
                    topk_length=lengths_view,
                    model_type=GLM_NSA_MODEL_TYPE,
                )

            def grouped() -> None:
                assert exact_grouped is not None
                assert grouped_chunks is not None
                assert grouped_buffers is not None
                grouped_out, grouped_lse, grouped_mid, grouped_mid_lse = grouped_buffers
                status = exact_grouped(
                    ctypes.c_void_p(q_view.data_ptr()),
                    ctypes.c_void_p(kv.data_ptr()),
                    ctypes.c_void_p(indices_view.data_ptr()),
                    ctypes.c_void_p(grouped_mid.data_ptr()),
                    ctypes.c_void_p(grouped_mid_lse.data_ptr()),
                    ctypes.c_void_p(lengths_view.data_ptr()),
                    ctypes.c_void_p(grouped_out.data_ptr()),
                    ctypes.c_void_p(grouped_lse.data_ptr()),
                    rows,
                    args.heads,
                    bucket_rows,
                    grouped_chunks,
                    ctypes.c_float(scale),
                    kv.stride(0),
                    ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
                )
                if status != 0:
                    raise RuntimeError(
                        f"exact grouped packed-FP8 MLA failed with status {status}"
                    )

            recurrent()
            parity()
            automatic()
            if grouped_buffers is not None:
                grouped()
            torch.cuda.synchronize()
            recurrent_graph = capture(recurrent)
            parity_graph = capture(parity)
            automatic_graph = capture(automatic)
            grouped_graph = capture(grouped) if grouped_buffers is not None else None
            recurrent_samples = measure(
                recurrent_graph, args.warmup, args.iterations, args.repeats
            )
            parity_samples = measure(
                parity_graph, args.warmup, args.iterations, args.repeats
            )
            automatic_samples = measure(
                automatic_graph, args.warmup, args.iterations, args.repeats
            )
            grouped_samples = (
                measure(grouped_graph, args.warmup, args.iterations, args.repeats)
                if grouped_graph is not None
                else None
            )
            recurrent_ms = statistics.median(recurrent_samples)
            parity_ms = statistics.median(parity_samples)
            automatic_ms = statistics.median(automatic_samples)
            grouped_suffix = ""
            if grouped_samples is not None and grouped_buffers is not None:
                grouped_out, grouped_lse, grouped_mid, grouped_mid_lse = grouped_buffers
                grouped_ms = statistics.median(grouped_samples)
                active_splits = (int(lengths_view.max().item()) + PAGE_ROWS - 1) // PAGE_ROWS
                grouped_suffix = (
                    f" grouped_cpb={grouped_chunks}"
                    f" grouped_exact={torch.equal(grouped_out, recurrent_out)}"
                    f" grouped_lse_exact={torch.equal(grouped_lse, recurrent_lse)}"
                    f" grouped_mid_exact="
                    f"{torch.equal(grouped_mid[:, :, :active_splits], recurrent_mid[:, :, :active_splits])}"
                    f" grouped_mid_lse_exact="
                    f"{torch.equal(grouped_mid_lse, recurrent_mid_lse)}"
                    f" grouped_ms={grouped_ms:.6f}"
                    f" grouped_vs_parity_speedup={parity_ms / grouped_ms:.3f}"
                )
            print(
                f"rows={rows} bucket={bucket_rows} heads={args.heads} "
                f"parity_exact={torch.equal(parity_out, recurrent_out)} "
                f"automatic_exact={torch.equal(automatic_out, recurrent_out)} "
                f"recurrent_ms={recurrent_ms:.6f} parity_ms={parity_ms:.6f} "
                f"automatic_ms={automatic_ms:.6f} "
                f"parity_speedup={recurrent_ms / parity_ms:.3f}"
                f"{grouped_suffix}",
                flush=True,
            )


if __name__ == "__main__":
    main()
