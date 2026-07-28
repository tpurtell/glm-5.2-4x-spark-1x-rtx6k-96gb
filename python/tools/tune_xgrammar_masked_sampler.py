#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import math
import statistics
from pathlib import Path

import torch


VOCAB = 154_880
MASK_WORDS = (VOCAB + 31) // 32


def check_status(lib: ctypes.CDLL, status: int, action: str) -> None:
    if status == 0:
        return
    error = ctypes.create_string_buffer(512)
    lib.glmrt_last_error(error, len(error))
    raise RuntimeError(f"{action} failed with status {status}: {error.value.decode()}")


def stream_pointer() -> ctypes.c_void_p:
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def configure(lib: ctypes.CDLL, symbol: str, argtypes) -> ctypes._CFuncPtr:
    function = getattr(lib, symbol)
    function.argtypes = argtypes
    function.restype = ctypes.c_int
    return function


def capture(operation) -> torch.cuda.CUDAGraph:
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
) -> list[float]:
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
    return samples


def parse_int_list(value: str, label: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",") if item]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{label} must be comma-separated integers") from error
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError(f"{label} values must be positive")
    return values


def make_masks(rows: int, allowed_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    if allowed_count > VOCAB:
        raise ValueError(f"allowed_count {allowed_count} exceeds vocabulary {VOCAB}")
    step = 7_919
    if math.gcd(step, VOCAB) != 1:
        raise RuntimeError("mask permutation step must be coprime with vocabulary")

    packed = torch.zeros((rows, MASK_WORDS), dtype=torch.int64)
    allowed = torch.zeros((rows, VOCAB), dtype=torch.bool)
    base = torch.arange(allowed_count, dtype=torch.int64)
    for row in range(rows):
        token_ids = (base * step + row * 1_009 + 17) % VOCAB
        words = token_ids // 32
        bits = torch.bitwise_left_shift(
            torch.ones_like(token_ids), token_ids.remainder(32)
        )
        packed[row].index_add_(0, words, bits)
        allowed[row, token_ids] = True

    host_mask = torch.empty(
        (rows, MASK_WORDS), dtype=torch.int32, pin_memory=True
    )
    host_mask.copy_(packed.to(torch.int32))
    return host_mask, allowed


def expected_sample(
    logits: torch.Tensor,
    allowed: torch.Tensor,
    random_uniforms: torch.Tensor,
    top_k: int,
    temperature: float,
    top_p: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    masked = logits.masked_fill(~allowed, -torch.inf)
    best_logits, best_indices = torch.topk(
        masked, top_k, dim=1, largest=True, sorted=True
    )
    probabilities = torch.softmax(best_logits / temperature, dim=1)
    cumulative = probabilities.cumsum(dim=1)
    nucleus_count = (cumulative < top_p).sum(dim=1) + 1
    nucleus_count.clamp_(max=top_k)
    nucleus_mass = cumulative.gather(1, (nucleus_count - 1).unsqueeze(1)).squeeze(1)
    target = random_uniforms * nucleus_mass
    selected_rank = (cumulative < target.unsqueeze(1)).sum(dim=1)
    selected_rank = torch.minimum(selected_rank, nucleus_count - 1)
    indices = best_indices.gather(1, selected_rank.unsqueeze(1)).squeeze(1)
    scores = probabilities.gather(1, selected_rank.unsqueeze(1)).squeeze(1) / nucleus_mass
    return indices.to(torch.uint32), scores


def summarize(samples: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def validate_grid_tie_break(lib: ctypes.CDLL, grid_fused, grid_blocks: list[int]) -> int:
    host_mask, allowed = make_masks(1, 8)
    expected_token = int(allowed[0].nonzero().min().item())
    device_mask = host_mask.cuda(non_blocking=False)
    logits = torch.zeros((1, VOCAB), dtype=torch.float32, device="cuda")
    random_uniforms = torch.full((1,), 0.5, dtype=torch.float32, device="cuda")
    partial_keys = torch.empty(max(grid_blocks) * 8, dtype=torch.uint64, device="cuda")
    out_indices = torch.empty(1, dtype=torch.uint32, device="cuda")
    out_scores = torch.empty(1, dtype=torch.float32, device="cuda")
    for blocks_per_row in grid_blocks:
        check_status(
            lib,
            grid_fused(
                ctypes.c_void_p(logits.data_ptr()),
                ctypes.c_void_p(device_mask.data_ptr()),
                ctypes.c_void_p(random_uniforms.data_ptr()),
                ctypes.c_void_p(partial_keys.data_ptr()),
                ctypes.c_void_p(out_indices.data_ptr()),
                ctypes.c_void_p(out_scores.data_ptr()),
                1,
                VOCAB,
                MASK_WORDS,
                blocks_per_row,
                1.0,
                1,
                1.0,
                stream_pointer(),
            ),
            f"validate {blocks_per_row}-block tie break",
        )
        torch.cuda.synchronize()
        actual_token = int(out_indices.item())
        if actual_token != expected_token or out_scores.item() != 1.0:
            raise RuntimeError(
                f"grid b{blocks_per_row} tie break selected {actual_token}, "
                f"expected {expected_token}"
            )
    return expected_token


def benchmark_case(
    lib: ctypes.CDLL,
    serial,
    apply_serial,
    grid_fused,
    grid_blocks: list[int],
    rows: int,
    allowed_count: int,
    top_k: int,
    temperature: float,
    top_p: float,
    warmup: int,
    iterations: int,
    repeats: int,
    seed: int,
) -> dict:
    if top_k > allowed_count:
        raise ValueError(
            f"top_k {top_k} exceeds allowed token count {allowed_count}; "
            "this benchmark requires a full top-k set"
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + rows * 100_003 + allowed_count * 17 + top_k)
    logits = torch.randn((rows, VOCAB), generator=generator, dtype=torch.float32).cuda()
    random_uniforms = torch.linspace(
        0.17, 0.83, rows, dtype=torch.float32, device="cuda"
    )
    host_mask, allowed_cpu = make_masks(rows, allowed_count)
    device_mask = host_mask.cuda(non_blocking=False)
    allowed = allowed_cpu.cuda(non_blocking=False)
    masked_logits = logits.masked_fill(~allowed, -torch.inf)
    workspace = torch.empty_like(logits)
    serial_indices = torch.empty(rows, dtype=torch.uint32, device="cuda")
    serial_scores = torch.empty(rows, dtype=torch.float32, device="cuda")
    apply_indices = torch.empty_like(serial_indices)
    apply_scores = torch.empty_like(serial_scores)
    fused_indices = torch.empty_like(serial_indices)
    fused_scores = torch.empty_like(serial_scores)
    partial_keys = torch.empty(
        rows * max(grid_blocks) * 8, dtype=torch.uint64, device="cuda"
    )

    expected_indices, expected_scores = expected_sample(
        logits, allowed, random_uniforms, top_k, temperature, top_p
    )

    def launch_serial() -> None:
        check_status(
            lib,
            serial(
                ctypes.c_void_p(masked_logits.data_ptr()),
                ctypes.c_void_p(random_uniforms.data_ptr()),
                ctypes.c_void_p(serial_indices.data_ptr()),
                ctypes.c_void_p(serial_scores.data_ptr()),
                rows,
                VOCAB,
                temperature,
                top_k,
                top_p,
                stream_pointer(),
            ),
            "launch pre-masked serial sampler",
        )

    def launch_apply_serial() -> None:
        check_status(
            lib,
            apply_serial(
                ctypes.c_void_p(logits.data_ptr()),
                ctypes.c_void_p(device_mask.data_ptr()),
                ctypes.c_void_p(random_uniforms.data_ptr()),
                ctypes.c_void_p(workspace.data_ptr()),
                ctypes.c_void_p(apply_indices.data_ptr()),
                ctypes.c_void_p(apply_scores.data_ptr()),
                rows,
                VOCAB,
                MASK_WORDS,
                temperature,
                top_k,
                top_p,
                stream_pointer(),
            ),
            "launch bitmask apply plus serial sampler",
        )

    def launch_grid(blocks_per_row: int) -> None:
        check_status(
            lib,
            grid_fused(
                ctypes.c_void_p(logits.data_ptr()),
                ctypes.c_void_p(device_mask.data_ptr()),
                ctypes.c_void_p(random_uniforms.data_ptr()),
                ctypes.c_void_p(partial_keys.data_ptr()),
                ctypes.c_void_p(fused_indices.data_ptr()),
                ctypes.c_void_p(fused_scores.data_ptr()),
                rows,
                VOCAB,
                MASK_WORDS,
                blocks_per_row,
                temperature,
                top_k,
                top_p,
                stream_pointer(),
            ),
            f"launch {blocks_per_row}-block fused masked sampler",
        )

    def launch_copy_grid(blocks_per_row: int) -> None:
        device_mask.copy_(host_mask, non_blocking=True)
        launch_grid(blocks_per_row)

    launch_serial()
    launch_apply_serial()
    torch.cuda.synchronize()
    for label, indices, scores in (
        ("serial", serial_indices, serial_scores),
        ("apply_serial", apply_indices, apply_scores),
    ):
        if not torch.equal(indices, expected_indices):
            raise RuntimeError(
                f"{label} indices differ: got={indices.cpu().tolist()} "
                f"expected={expected_indices.cpu().tolist()}"
            )
        if not torch.allclose(scores, expected_scores, rtol=2e-6, atol=2e-7):
            max_abs = (scores - expected_scores).abs().max().item()
            raise RuntimeError(f"{label} scores differ, max_abs={max_abs}")

    for blocks_per_row in grid_blocks:
        launch_grid(blocks_per_row)
        torch.cuda.synchronize()
        if not torch.equal(fused_indices, expected_indices):
            raise RuntimeError(
                f"grid b{blocks_per_row} indices differ: "
                f"got={fused_indices.cpu().tolist()} "
                f"expected={expected_indices.cpu().tolist()}"
            )
        if not torch.allclose(fused_scores, expected_scores, rtol=2e-6, atol=2e-7):
            max_abs = (fused_scores - expected_scores).abs().max().item()
            raise RuntimeError(
                f"grid b{blocks_per_row} scores differ, max_abs={max_abs}"
            )

    baseline_graphs = {
        "pre_masked_serial": capture(launch_serial),
        "apply_then_serial": capture(launch_apply_serial),
    }
    candidate_graphs = {}
    for blocks_per_row in grid_blocks:
        candidate_graphs[f"grid_b{blocks_per_row}"] = capture(
            lambda blocks=blocks_per_row: launch_grid(blocks)
        )
        candidate_graphs[f"pinned_copy_grid_b{blocks_per_row}"] = capture(
            lambda blocks=blocks_per_row: launch_copy_grid(blocks)
        )
    timings = {
        name: summarize(measure(graph, min(warmup, 1), min(iterations, 3), 1))
        for name, graph in baseline_graphs.items()
    }
    timings.update(
        {
            name: summarize(measure(graph, warmup, iterations, repeats))
            for name, graph in candidate_graphs.items()
        }
    )
    best_grid = min(
        grid_blocks,
        key=lambda blocks: timings[f"grid_b{blocks}"]["median_ms"],
    )
    fused_ms = timings[f"grid_b{best_grid}"]["median_ms"]
    copy_fused_ms = timings[f"pinned_copy_grid_b{best_grid}"]["median_ms"]
    apply_serial_ms = timings["apply_then_serial"]["median_ms"]
    return {
        "rows": rows,
        "allowed_tokens": allowed_count,
        "allowed_fraction": allowed_count / VOCAB,
        "top_k": top_k,
        "mask_bytes": rows * MASK_WORDS * 4,
        "exact_indices": True,
        "timings": timings,
        "best_grid_blocks_per_row": best_grid,
        "speedup_vs_apply_serial": apply_serial_ms / fused_ms,
        "speedup_including_mask_copy": apply_serial_ms / copy_fused_ms,
        "mask_copy_overhead_ms": copy_fused_ms - fused_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark a native XGrammar bitmask-aware CUDA logits sampler."
    )
    parser.add_argument("--native-lib", type=Path, required=True)
    parser.add_argument("--rows", default="1,4,6")
    parser.add_argument("--allowed", default="8,4096,38720")
    parser.add_argument("--top-k", default="1,8")
    parser.add_argument("--grid-blocks", default="16,32,48,64")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    rows = parse_int_list(args.rows, "rows")
    allowed_counts = parse_int_list(args.allowed, "allowed")
    top_ks = parse_int_list(args.top_k, "top-k")
    grid_blocks = parse_int_list(args.grid_blocks, "grid-blocks")
    if any(value > 8 for value in top_ks):
        parser.error("candidate supports top-k values from 1 through 8")
    if any(value > 256 for value in grid_blocks):
        parser.error("grid-blocks values must not exceed 256")
    if min(args.iterations, args.repeats) < 1 or args.warmup < 0:
        parser.error("iterations/repeats must be positive and warmup nonnegative")
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        parser.error("temperature must be positive and top-p must be in (0, 1]")

    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    lib = ctypes.CDLL(str(args.native_lib.resolve()))
    lib.glmrt_last_error.argtypes = (ctypes.c_char_p, ctypes.c_size_t)
    lib.glmrt_last_error.restype = ctypes.c_int
    serial = configure(
        lib,
        "glmrt_cuda_logits_sample_topk_topp_f32_async",
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_void_p,
        ),
    )
    apply_serial = configure(
        lib,
        "glmrt_cuda_logits_apply_bitmask_sample_topk_topp_f32_candidate_async",
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
            ctypes.c_float,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_void_p,
        ),
    )
    grid_fused = configure(
        lib,
        "glmrt_cuda_logits_masked_sample_topk_topp_f32_grid_candidate_async",
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
            ctypes.c_float,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_void_p,
        ),
    )
    tie_break_token = validate_grid_tie_break(lib, grid_fused, grid_blocks)

    cases = []
    for row_count in rows:
        for allowed_count in allowed_counts:
            for top_k in top_ks:
                if top_k > allowed_count:
                    continue
                cases.append(
                    benchmark_case(
                        lib,
                        serial,
                        apply_serial,
                        grid_fused,
                        grid_blocks,
                        row_count,
                        allowed_count,
                        top_k,
                        args.temperature,
                        args.top_p,
                        args.warmup,
                        args.iterations,
                        args.repeats,
                        args.seed,
                    )
                )

    print(
        json.dumps(
            {
                "benchmark": "xgrammar_masked_logits_sampler_candidate",
                "status": "ok",
                "device": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "vocab": VOCAB,
                "mask_words": MASK_WORDS,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "repeats": args.repeats,
                "baseline_warmup": min(args.warmup, 1),
                "baseline_iterations": min(args.iterations, 3),
                "baseline_repeats": 1,
                "tie_break_exact": True,
                "tie_break_expected_token": tie_break_token,
                "cases": cases,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
