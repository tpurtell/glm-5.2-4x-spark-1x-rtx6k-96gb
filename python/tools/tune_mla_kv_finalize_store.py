#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import statistics
from pathlib import Path

import torch


HIDDEN = 6_144
KV_LORA_RANK = 512
ROPE_DIM = 64
KV_WIDTH = KV_LORA_RANK + ROPE_DIM
KV_BYTES = KV_WIDTH * 2
DSA_VALUES = 128
DSA_BYTES = DSA_VALUES * 2
FP8_MAIN_BYTES = 512 + 4 * 4 + 64 * 2
MXFP4_MAIN_BYTES = 512 // 2 + 512 // 32 + 64 * 2
FORMAT_IDS = {"bf16": 0, "fp8": 1, "nvfp4": 2}
FORMAT_MAIN_BYTES = {
    "bf16": KV_BYTES,
    "fp8": FP8_MAIN_BYTES,
    "nvfp4": MXFP4_MAIN_BYTES,
}
CUDA_MEMCPY_DEVICE_TO_DEVICE = 3


def parse_int_list(value: str, label: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",") if item]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{label} must be comma-separated integers"
        ) from error
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError(f"{label} values must be positive")
    return values


def parse_formats(value: str) -> list[str]:
    formats = [item for item in value.split(",") if item]
    unknown = sorted(set(formats) - FORMAT_IDS.keys())
    if not formats or unknown:
        raise argparse.ArgumentTypeError(
            f"formats must be drawn from {sorted(FORMAT_IDS)}, got {unknown}"
        )
    return formats


def parse_bools(value: str, label: str) -> list[bool]:
    values = []
    for item in value.split(","):
        if item not in ("0", "1"):
            raise argparse.ArgumentTypeError(f"{label} values must be 0 or 1")
        values.append(item == "1")
    if not values:
        raise argparse.ArgumentTypeError(f"{label} cannot be empty")
    return values


def check_status(lib: ctypes.CDLL, status: int, action: str) -> None:
    if status == 0:
        return
    error = ctypes.create_string_buffer(512)
    lib.glmrt_last_error(error, len(error))
    raise RuntimeError(f"{action} failed with status {status}: {error.value.decode()}")


def configure(lib: ctypes.CDLL, symbol: str, argtypes):
    function = getattr(lib, symbol)
    function.argtypes = argtypes
    function.restype = ctypes.c_int
    return function


def stream_pointer() -> ctypes.c_void_p:
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def pointer(tensor: torch.Tensor | None, byte_offset: int = 0) -> ctypes.c_void_p:
    if tensor is None:
        return ctypes.c_void_p()
    return ctypes.c_void_p(tensor.data_ptr() + byte_offset)


def capture(operation) -> torch.cuda.CUDAGraph:
    operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()
    return graph


def measure(
    graphs: list[torch.cuda.CUDAGraph],
    warmup: int,
    iterations: int,
    repeats: int,
) -> list[float]:
    for iteration in range(warmup * len(graphs)):
        graphs[iteration % len(graphs)].replay()
    torch.cuda.synchronize()
    launches = iterations * len(graphs)
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            for graph in graphs:
                graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / launches)
    return samples


def summarize(samples: list[float]) -> dict[str, float | list[float]]:
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark exact fused MLA KV finalize, pack, attention handoff, "
            "cache store, and DSA-tail copy."
        )
    )
    parser.add_argument("--native-lib", type=Path, required=True)
    parser.add_argument("--rows", default="1,2,4,6")
    parser.add_argument("--contexts", default="1024")
    parser.add_argument("--formats", type=parse_formats, default=list(FORMAT_IDS))
    parser.add_argument("--dsa", default="0,1")
    parser.add_argument("--sets", type=int, default=4)
    parser.add_argument("--layers", type=int, default=78)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--theta", type=float, default=1_000_000.0)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    rows_values = parse_int_list(args.rows, "rows")
    contexts = parse_int_list(args.contexts, "contexts")
    formats = args.formats if isinstance(args.formats, list) else parse_formats(args.formats)
    dsa_values = parse_bools(args.dsa, "dsa")
    if max(rows_values) > 4096:
        parser.error("rows must not exceed 4096")
    if max(contexts) + max(rows_values) > 0xFFFFFFFF:
        parser.error("context plus rows must fit uint32")
    if min(args.sets, args.layers, args.iterations, args.repeats) < 1 or args.warmup < 0:
        parser.error(
            "sets/layers/iterations/repeats must be positive and warmup nonnegative"
        )
    if args.eps <= 0.0 or args.theta <= 0.0:
        parser.error("eps and theta must be positive")

    torch.manual_seed(args.seed)
    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    lib = ctypes.CDLL(str(args.native_lib.resolve()))
    lib.glmrt_last_error.argtypes = (ctypes.c_char_p, ctypes.c_size_t)
    lib.glmrt_last_error.restype = ctypes.c_int
    rmsnorm = configure(
        lib,
        "glmrt_cuda_rmsnorm_bf16_async",
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_void_p,
        ),
    )
    linear = configure(
        lib,
        "glmrt_cuda_linear_bf16_cublas_async",
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ),
    )
    layernorm = configure(
        lib,
        "glmrt_cuda_layernorm_affine_bf16_async",
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_void_p,
        ),
    )
    prepare_precomputed = configure(
        lib,
        "glmrt_cuda_mla_kv_prepare_bf16_precomputed_rope_candidate_async",
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_void_p,
        ),
    )
    factor_launch = configure(
        lib,
        "glmrt_cuda_mla_rope_factors_f32_candidate_async",
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_void_p,
        ),
    )
    pack_fp8 = configure(
        lib,
        "glmrt_cuda_mla_kv_pack_fp8_ds_mla_async",
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ),
    )
    pack_nvfp4 = configure(
        lib,
        "glmrt_cuda_mla_kv_pack_mxfp4_ds_mla_async",
        pack_fp8.argtypes,
    )
    write_blocks = configure(
        lib,
        "glmrt_cuda_kv_cache_write_blocks_async",
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ),
    )
    candidate = configure(
        lib,
        "glmrt_cuda_mla_kv_finalize_store_candidate_async",
        (
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
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_void_p,
        ),
    )
    cudart = ctypes.CDLL("libcudart.so.13")
    cuda_memcpy_async = cudart.cudaMemcpyAsync
    cuda_memcpy_async.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    )
    cuda_memcpy_async.restype = ctypes.c_int

    input_norm_weight = torch.randn(HIDDEN, dtype=torch.bfloat16, device=device)
    kv_norm_weight = torch.randn(
        KV_LORA_RANK, dtype=torch.bfloat16, device=device
    )
    dsa_norm_weight = torch.randn(DSA_VALUES, dtype=torch.bfloat16, device=device)
    dsa_norm_bias = torch.randn(DSA_VALUES, dtype=torch.bfloat16, device=device)
    kv_weights = [
        torch.randn((KV_WIDTH, HIDDEN), dtype=torch.bfloat16, device=device)
        for _ in range(args.sets)
    ]
    dsa_weights = [
        torch.randn((DSA_VALUES, HIDDEN), dtype=torch.bfloat16, device=device)
        for _ in range(args.sets)
    ]
    results = []

    for rows in rows_values:
        hidden_sets = [
            torch.randn((rows, HIDDEN), dtype=torch.bfloat16, device=device)
            for _ in range(args.sets)
        ]
        seed_projected_sets = [
            torch.randn((rows, KV_WIDTH), dtype=torch.bfloat16, device=device)
            for _ in range(args.sets)
        ]
        seed_dsa_sets = [
            torch.randn((rows, DSA_VALUES), dtype=torch.bfloat16, device=device)
            for _ in range(args.sets)
        ]
        for context in contexts:
            positions = torch.arange(
                context,
                context + rows,
                dtype=torch.int64,
                device=device,
            ).to(torch.uint32)
            rope_factors = torch.empty(
                (rows, ROPE_DIM), dtype=torch.float32, device=device
            )
            def launch_factors() -> None:
                check_status(
                    lib,
                    factor_launch(
                        pointer(positions),
                        pointer(rope_factors),
                        rows,
                        args.theta,
                        stream_pointer(),
                    ),
                    "prepare RoPE factors",
                )

            launch_factors()
            torch.cuda.synchronize()
            factor_graph = capture(launch_factors)
            factor_timing = summarize(
                measure(
                    [factor_graph],
                    args.warmup,
                    args.iterations,
                    args.repeats,
                )
            )
            for format_name in formats:
                format_id = FORMAT_IDS[format_name]
                main_bytes = FORMAT_MAIN_BYTES[format_name]
                for with_dsa in dsa_values:
                    dsa_count = DSA_VALUES if with_dsa else 0
                    dsa_bytes = DSA_BYTES if with_dsa else 0
                    cache_stride = main_bytes + dsa_bytes
                    cache_rows = context + rows
                    baseline_cache = torch.full(
                        (cache_rows * cache_stride,),
                        0xA5,
                        dtype=torch.uint8,
                        device=device,
                    )
                    candidate_cache = torch.full_like(baseline_cache, 0x5A)
                    cache_offset = context * cache_stride
                    main_src_offsets = torch.arange(
                        0,
                        rows * main_bytes,
                        main_bytes,
                        dtype=torch.int64,
                        device=device,
                    )
                    main_cache_offsets = torch.arange(
                        cache_offset,
                        cache_offset + rows * cache_stride,
                        cache_stride,
                        dtype=torch.int64,
                        device=device,
                    )
                    main_block_bytes = torch.full(
                        (rows,), main_bytes, dtype=torch.int64, device=device
                    )
                    dsa_src_offsets = torch.arange(
                        0,
                        rows * max(dsa_bytes, 1),
                        max(dsa_bytes, 1),
                        dtype=torch.int64,
                        device=device,
                    )
                    dsa_cache_offsets = main_cache_offsets + main_bytes
                    dsa_block_bytes = torch.full(
                        (rows,), dsa_bytes, dtype=torch.int64, device=device
                    )
                    states = []
                    final_baseline_graphs = []
                    final_candidate_graphs = []
                    pipeline_baseline_graphs = []
                    pipeline_candidate_graphs = []
                    exact_cache = True
                    exact_attention = True
                    exact_pipeline_cache = True
                    exact_pipeline_attention = True
                    finalize_cache_mismatch_bytes = 0
                    finalize_attention_mismatch_values = 0
                    pipeline_cache_mismatch_bytes = 0
                    pipeline_attention_mismatch_values = 0
                    pipeline_projection_mismatch_values = 0
                    pipeline_dsa_mismatch_values = 0
                    pipeline_attention_latent_mismatch_values = 0
                    pipeline_attention_rope_mismatch_values = 0
                    pipeline_attention_max_abs = 0.0

                    for set_index in range(args.sets):
                        seed_projected = seed_projected_sets[set_index]
                        seed_dsa = seed_dsa_sets[set_index] if with_dsa else None
                        baseline_prepared = torch.empty_like(seed_projected)
                        candidate_attention = torch.empty_like(seed_projected)
                        baseline_attention = torch.empty_like(seed_projected)
                        baseline_packed = (
                            None
                            if format_name == "bf16"
                            else torch.empty(
                                rows * main_bytes, dtype=torch.uint8, device=device
                            )
                        )

                        def launch_baseline_finalize(
                            projected: torch.Tensor,
                            dsa: torch.Tensor | None,
                            prepared: torch.Tensor,
                            attention: torch.Tensor,
                            packed: torch.Tensor | None,
                        ) -> None:
                            status = prepare_precomputed(
                                pointer(projected),
                                pointer(rope_factors),
                                pointer(kv_norm_weight),
                                pointer(prepared),
                                rows,
                                KV_BYTES,
                                KV_BYTES,
                                args.eps,
                                stream_pointer(),
                            )
                            check_status(lib, status, "baseline KV prepare")
                            write_source = prepared
                            if format_name == "fp8":
                                check_status(
                                    lib,
                                    pack_fp8(
                                        pointer(prepared),
                                        pointer(packed),
                                        rows,
                                        KV_BYTES,
                                        main_bytes,
                                        stream_pointer(),
                                    ),
                                    "baseline FP8 pack",
                                )
                                write_source = packed
                            elif format_name == "nvfp4":
                                check_status(
                                    lib,
                                    pack_nvfp4(
                                        pointer(prepared),
                                        pointer(packed),
                                        rows,
                                        KV_BYTES,
                                        main_bytes,
                                        stream_pointer(),
                                    ),
                                    "baseline NVFP4 pack",
                                )
                                write_source = packed
                            check_status(
                                lib,
                                write_blocks(
                                    pointer(write_source),
                                    pointer(baseline_cache),
                                    pointer(main_src_offsets),
                                    pointer(main_cache_offsets),
                                    pointer(main_block_bytes),
                                    rows,
                                    stream_pointer(),
                                ),
                                "baseline main cache write",
                            )
                            cuda_status = cuda_memcpy_async(
                                pointer(attention),
                                pointer(prepared),
                                rows * KV_BYTES,
                                CUDA_MEMCPY_DEVICE_TO_DEVICE,
                                stream_pointer(),
                            )
                            if cuda_status != 0:
                                raise RuntimeError(
                                    f"attention-ready cudaMemcpyAsync failed: {cuda_status}"
                                )
                            if dsa is not None:
                                check_status(
                                    lib,
                                    write_blocks(
                                        pointer(dsa),
                                        pointer(baseline_cache),
                                        pointer(dsa_src_offsets),
                                        pointer(dsa_cache_offsets),
                                        pointer(dsa_block_bytes),
                                        rows,
                                        stream_pointer(),
                                    ),
                                    "baseline DSA cache write",
                                )

                        def launch_candidate_finalize(
                            projected: torch.Tensor,
                            dsa: torch.Tensor | None,
                            attention: torch.Tensor,
                        ) -> None:
                            check_status(
                                lib,
                                candidate(
                                    pointer(projected),
                                    pointer(rope_factors),
                                    pointer(kv_norm_weight),
                                    pointer(candidate_cache, cache_offset),
                                    pointer(attention),
                                    pointer(dsa),
                                    rows,
                                    KV_BYTES,
                                    cache_stride,
                                    KV_BYTES,
                                    dsa_bytes,
                                    dsa_count,
                                    format_id,
                                    args.eps,
                                    stream_pointer(),
                                ),
                                "fused KV finalize/store candidate",
                            )

                        def baseline_finalize() -> None:
                            launch_baseline_finalize(
                                seed_projected,
                                seed_dsa,
                                baseline_prepared,
                                baseline_attention,
                                baseline_packed,
                            )

                        def candidate_finalize() -> None:
                            launch_candidate_finalize(
                                seed_projected,
                                seed_dsa,
                                candidate_attention,
                            )

                        hidden = hidden_sets[set_index]
                        kv_weight = kv_weights[set_index]
                        dsa_weight = dsa_weights[set_index]
                        baseline_normalized = torch.empty_like(hidden)
                        candidate_normalized = torch.empty_like(hidden)
                        baseline_projected = torch.empty_like(seed_projected)
                        candidate_projected = torch.empty_like(seed_projected)
                        baseline_dsa_projected = (
                            torch.empty_like(seed_dsa) if with_dsa else None
                        )
                        candidate_dsa_projected = (
                            torch.empty_like(seed_dsa) if with_dsa else None
                        )
                        baseline_dsa = torch.empty_like(seed_dsa) if with_dsa else None
                        candidate_dsa = torch.empty_like(seed_dsa) if with_dsa else None
                        baseline_pipeline_prepared = torch.empty_like(seed_projected)
                        baseline_pipeline_attention = torch.empty_like(seed_projected)
                        candidate_pipeline_attention = torch.empty_like(seed_projected)
                        baseline_pipeline_packed = (
                            None
                            if format_name == "bf16"
                            else torch.empty(
                                rows * main_bytes, dtype=torch.uint8, device=device
                            )
                        )

                        def launch_prefix(
                            normalized: torch.Tensor,
                            projected: torch.Tensor,
                            dsa_projected: torch.Tensor | None,
                            dsa: torch.Tensor | None,
                        ) -> None:
                            check_status(
                                lib,
                                rmsnorm(
                                    pointer(hidden),
                                    pointer(input_norm_weight),
                                    pointer(normalized),
                                    rows,
                                    HIDDEN,
                                    args.eps,
                                    stream_pointer(),
                                ),
                                "input RMSNorm",
                            )
                            check_status(
                                lib,
                                linear(
                                    pointer(normalized),
                                    pointer(kv_weight),
                                    ctypes.c_void_p(),
                                    pointer(projected),
                                    rows,
                                    HIDDEN,
                                    KV_WIDTH,
                                    stream_pointer(),
                                ),
                                "KV-A projection",
                            )
                            if dsa is not None and dsa_projected is not None:
                                check_status(
                                    lib,
                                    linear(
                                        pointer(normalized),
                                        pointer(dsa_weight),
                                        ctypes.c_void_p(),
                                        pointer(dsa_projected),
                                        rows,
                                        HIDDEN,
                                        DSA_VALUES,
                                        stream_pointer(),
                                    ),
                                    "DSA projection",
                                )
                                check_status(
                                    lib,
                                    layernorm(
                                        pointer(dsa_projected),
                                        pointer(dsa_norm_weight),
                                        pointer(dsa_norm_bias),
                                        pointer(dsa),
                                        rows,
                                        DSA_VALUES,
                                        args.eps,
                                        stream_pointer(),
                                    ),
                                    "DSA layernorm",
                                )

                        def baseline_pipeline() -> None:
                            launch_prefix(
                                baseline_normalized,
                                baseline_projected,
                                baseline_dsa_projected,
                                baseline_dsa,
                            )
                            launch_baseline_finalize(
                                baseline_projected,
                                baseline_dsa,
                                baseline_pipeline_prepared,
                                baseline_pipeline_attention,
                                baseline_pipeline_packed,
                            )

                        def candidate_pipeline() -> None:
                            launch_prefix(
                                candidate_normalized,
                                candidate_projected,
                                candidate_dsa_projected,
                                candidate_dsa,
                            )
                            launch_candidate_finalize(
                                candidate_projected,
                                candidate_dsa,
                                candidate_pipeline_attention,
                            )

                        baseline_finalize()
                        candidate_finalize()
                        torch.cuda.synchronize()
                        region = slice(
                            cache_offset, cache_offset + rows * cache_stride
                        )
                        finalize_cache_equal = torch.equal(
                            baseline_cache[region], candidate_cache[region]
                        )
                        finalize_attention_equal = torch.equal(
                            baseline_attention, candidate_attention
                        )
                        exact_cache = exact_cache and finalize_cache_equal
                        exact_attention = exact_attention and finalize_attention_equal
                        finalize_cache_mismatch_bytes += int(
                            (baseline_cache[region] != candidate_cache[region])
                            .sum()
                            .item()
                        )
                        finalize_attention_mismatch_values += int(
                            (baseline_attention != candidate_attention).sum().item()
                        )
                        baseline_pipeline()
                        candidate_pipeline()
                        torch.cuda.synchronize()
                        pipeline_cache_equal = torch.equal(
                            baseline_cache[region], candidate_cache[region]
                        )
                        pipeline_attention_equal = torch.equal(
                            baseline_pipeline_attention,
                            candidate_pipeline_attention,
                        )
                        exact_pipeline_cache = (
                            exact_pipeline_cache and pipeline_cache_equal
                        )
                        exact_pipeline_attention = (
                            exact_pipeline_attention
                            and pipeline_attention_equal
                        )
                        pipeline_cache_mismatch_bytes += int(
                            (baseline_cache[region] != candidate_cache[region])
                            .sum()
                            .item()
                        )
                        pipeline_attention_mismatch_values += int(
                            (
                                baseline_pipeline_attention
                                != candidate_pipeline_attention
                            )
                            .sum()
                            .item()
                        )
                        pipeline_attention_latent_mismatch_values += int(
                            (
                                baseline_pipeline_attention[:, :KV_LORA_RANK]
                                != candidate_pipeline_attention[:, :KV_LORA_RANK]
                            )
                            .sum()
                            .item()
                        )
                        pipeline_attention_rope_mismatch_values += int(
                            (
                                baseline_pipeline_attention[:, KV_LORA_RANK:]
                                != candidate_pipeline_attention[:, KV_LORA_RANK:]
                            )
                            .sum()
                            .item()
                        )
                        pipeline_attention_max_abs = max(
                            pipeline_attention_max_abs,
                            float(
                                (
                                    baseline_pipeline_attention.float()
                                    - candidate_pipeline_attention.float()
                                )
                                .abs()
                                .max()
                                .item()
                            ),
                        )
                        pipeline_projection_mismatch_values += int(
                            (baseline_projected != candidate_projected).sum().item()
                        )
                        if baseline_dsa is not None and candidate_dsa is not None:
                            pipeline_dsa_mismatch_values += int(
                                (baseline_dsa != candidate_dsa).sum().item()
                            )
                        final_baseline_graphs.append(capture(baseline_finalize))
                        final_candidate_graphs.append(capture(candidate_finalize))
                        pipeline_baseline_graphs.append(capture(baseline_pipeline))
                        pipeline_candidate_graphs.append(capture(candidate_pipeline))
                        states.append(
                            (
                                seed_projected,
                                seed_dsa,
                                baseline_prepared,
                                baseline_attention,
                                candidate_attention,
                                baseline_packed,
                                hidden,
                                kv_weight,
                                dsa_weight,
                                baseline_normalized,
                                candidate_normalized,
                                baseline_projected,
                                candidate_projected,
                                baseline_dsa_projected,
                                candidate_dsa_projected,
                                baseline_dsa,
                                candidate_dsa,
                                baseline_pipeline_prepared,
                                baseline_pipeline_attention,
                                candidate_pipeline_attention,
                                baseline_pipeline_packed,
                            )
                        )

                    timings = {
                        "baseline_finalize": summarize(
                            measure(
                                final_baseline_graphs,
                                args.warmup,
                                args.iterations,
                                args.repeats,
                            )
                        ),
                        "candidate_finalize": summarize(
                            measure(
                                final_candidate_graphs,
                                args.warmup,
                                args.iterations,
                                args.repeats,
                            )
                        ),
                        "baseline_pipeline": summarize(
                            measure(
                                pipeline_baseline_graphs,
                                args.warmup,
                                args.iterations,
                                args.repeats,
                            )
                        ),
                        "candidate_pipeline": summarize(
                            measure(
                                pipeline_candidate_graphs,
                                args.warmup,
                                args.iterations,
                                args.repeats,
                            )
                        ),
                    }
                    baseline_finalize_ms = timings["baseline_finalize"]["median_ms"]
                    candidate_finalize_ms = timings["candidate_finalize"]["median_ms"]
                    baseline_pipeline_ms = timings["baseline_pipeline"]["median_ms"]
                    candidate_pipeline_ms = timings["candidate_pipeline"]["median_ms"]
                    results.append(
                        {
                            "rows": rows,
                            "context": context,
                            "format": format_name,
                            "dsa": with_dsa,
                            "precomputed_rope": True,
                            "rope_factor_timing": factor_timing,
                            "rope_factor_amortized_ms_per_layer": (
                                factor_timing["median_ms"] / args.layers
                            ),
                            "cache_stride_bytes": cache_stride,
                            "exact_finalize_cache": exact_cache,
                            "exact_finalize_attention_ready": exact_attention,
                            "exact_pipeline_cache": exact_pipeline_cache,
                            "exact_pipeline_attention_ready": (
                                exact_pipeline_attention
                            ),
                            "finalize_cache_mismatch_bytes": (
                                finalize_cache_mismatch_bytes
                            ),
                            "finalize_attention_mismatch_values": (
                                finalize_attention_mismatch_values
                            ),
                            "pipeline_cache_mismatch_bytes": (
                                pipeline_cache_mismatch_bytes
                            ),
                            "pipeline_attention_mismatch_values": (
                                pipeline_attention_mismatch_values
                            ),
                            "pipeline_attention_latent_mismatch_values": (
                                pipeline_attention_latent_mismatch_values
                            ),
                            "pipeline_attention_rope_mismatch_values": (
                                pipeline_attention_rope_mismatch_values
                            ),
                            "pipeline_attention_max_abs": (
                                pipeline_attention_max_abs
                            ),
                            "pipeline_projection_mismatch_values": (
                                pipeline_projection_mismatch_values
                            ),
                            "pipeline_dsa_mismatch_values": (
                                pipeline_dsa_mismatch_values
                            ),
                            "timings": timings,
                            "finalize_speedup": (
                                baseline_finalize_ms / candidate_finalize_ms
                            ),
                            "pipeline_speedup": (
                                baseline_pipeline_ms / candidate_pipeline_ms
                            ),
                            "pipeline_saved_ms": (
                                baseline_pipeline_ms - candidate_pipeline_ms
                            ),
                        }
                    )

    print(
        json.dumps(
            {
                "benchmark": "mla_kv_finalize_store_candidate",
                "status": "ok",
                "device": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "sets": args.sets,
                "layers": args.layers,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "repeats": args.repeats,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
