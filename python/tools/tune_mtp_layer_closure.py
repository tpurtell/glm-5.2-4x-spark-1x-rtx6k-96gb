#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
import time
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

from bench_b12x_coordinator_aot import (  # noqa: E402
    CoordinatorBuffers,
    DeviceBuffer,
    device_buffer,
)

HIDDEN = 6_144
Q_LORA_RANK = 2_048
HEADS = 64
NOPE_DIM = 192
ROPE_DIM = 64
V_DIM = 256
Q_B_WIDTH = HEADS * (NOPE_DIM + ROPE_DIM)
KV_WIDTH = 512 + ROPE_DIM
BF16_CACHE_STRIDE = KV_WIDTH * 2
EPS = 1.0e-6
THETA = 1_000_000.0


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


def pointer(tensor: torch.Tensor | None) -> ctypes.c_void_p:
    return ctypes.c_void_p() if tensor is None else ctypes.c_void_p(tensor.data_ptr())


def stream_pointer() -> ctypes.c_void_p:
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


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


def summarize(samples: list[float]) -> dict[str, float | list[float]]:
    return {
        "median": statistics.median(samples),
        "minimum": min(samples),
        "maximum": max(samples),
        "samples": samples,
    }


def compare_tensors(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float | bool]:
    reference_f32 = reference.float()
    difference = candidate.float() - reference_f32
    reference_norm = torch.linalg.vector_norm(reference_f32)
    difference_norm = torch.linalg.vector_norm(difference)
    return {
        "exact": bool(torch.equal(reference, candidate)),
        "max_abs": float(difference.abs().max().item()),
        "relative_l2": float(
            (difference_norm / reference_norm).item()
            if reference_norm.item() != 0.0
            else difference_norm.item()
        ),
    }


def measure_sequences(
    sequences: list[list[torch.cuda.CUDAGraph]],
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, dict[str, float | list[float]]]:
    for iteration in range(warmup * len(sequences)):
        for graph in sequences[iteration % len(sequences)]:
            graph.replay()
    torch.cuda.synchronize()

    gpu_samples = []
    submit_samples = []
    wall_samples = []
    closures = iterations * len(sequences)
    for _ in range(repeats):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        wall_start = time.perf_counter_ns()
        for _ in range(iterations):
            for sequence in sequences:
                for graph in sequence:
                    graph.replay()
        submit_end = time.perf_counter_ns()
        end_event.record()
        end_event.synchronize()
        wall_end = time.perf_counter_ns()
        gpu_samples.append(start_event.elapsed_time(end_event) / closures)
        submit_samples.append((submit_end - wall_start) / closures / 1_000.0)
        wall_samples.append((wall_end - wall_start) / closures / 1_000.0)
    idle_submit_samples = []
    for iteration in range(max(31, repeats * 7)):
        sequence = sequences[iteration % len(sequences)]
        torch.cuda.synchronize()
        submit_start = time.perf_counter_ns()
        for graph in sequence:
            graph.replay()
        submit_end = time.perf_counter_ns()
        torch.cuda.synchronize()
        idle_submit_samples.append((submit_end - submit_start) / 1_000.0)

    return {
        "gpu_ms_per_closure": summarize(gpu_samples),
        "idle_submit_us_per_closure": summarize(idle_submit_samples),
        "queued_submit_us_per_closure": summarize(submit_samples),
        "wall_us_per_closure": summarize(wall_samples),
    }


@dataclass
class PackedProjection:
    size_k: int
    size_n: int
    weight: torch.Tensor
    scale: torch.Tensor
    global_scale: torch.Tensor
    packed_routes: torch.Tensor
    block_experts: torch.Tensor
    route_count: torch.Tensor
    topk_weights: torch.Tensor
    scratch: torch.Tensor
    locks: torch.Tensor

    def buffers(
        self, input_rows: torch.Tensor, output_rows: torch.Tensor
    ) -> CoordinatorBuffers:
        return CoordinatorBuffers(
            device_buffer(input_rows),
            device_buffer(self.weight),
            device_buffer(output_rows),
            device_buffer(self.scale),
            device_buffer(self.global_scale),
            device_buffer(self.packed_routes),
            device_buffer(self.block_experts),
            device_buffer(self.route_count),
            device_buffer(self.topk_weights),
            device_buffer(self.scratch),
            device_buffer(self.locks),
        )

    @property
    def weight_bytes(self) -> int:
        return (
            self.weight.numel() * self.weight.element_size()
            + self.scale.numel() * self.scale.element_size()
        )


@dataclass
class ClosureState:
    hidden: torch.Tensor
    input_norm_weight: torch.Tensor
    q_a_weight: torch.Tensor
    q_a_norm_weight: torch.Tensor
    normalized_hidden: torch.Tensor
    q_a_projected: torch.Tensor
    q_a_normalized: torch.Tensor
    q_b_output: torch.Tensor
    q_nope: torch.Tensor
    q_rope: torch.Tensor
    k_nope: torch.Tensor
    k_rope: torch.Tensor
    values: torch.Tensor
    attention_q: torch.Tensor
    attention_k: torch.Tensor
    attention_output: torch.Tensor
    attention_workspace: torch.Tensor
    hidden_delta: torch.Tensor
    kv_projected: torch.Tensor
    kv_rope_factors: torch.Tensor
    kv_norm_weight: torch.Tensor
    kv_cache: torch.Tensor
    kv_attention_ready: torch.Tensor
    q_b: PackedProjection
    q_b_m16: PackedProjection | None
    o_projection: PackedProjection
    o_projection_m16: PackedProjection | None
    q_b_split_buffers: list[CoordinatorBuffers]
    q_b_m16_buffers: CoordinatorBuffers | None
    o_buffers: list[CoordinatorBuffers]
    o_m16_buffers: CoordinatorBuffers | None
    attention_contexts: list[dict]
    batched_attention_runner: object


def make_m16_launch_projection(projection: PackedProjection) -> PackedProjection:
    return PackedProjection(
        size_k=projection.size_k,
        size_n=projection.size_n,
        weight=projection.weight,
        scale=projection.scale,
        global_scale=projection.global_scale,
        packed_routes=torch.arange(16, dtype=torch.int32, device="cuda"),
        block_experts=torch.zeros(2, dtype=torch.int32, device="cuda"),
        route_count=torch.full((1,), 16, dtype=torch.int32, device="cuda"),
        topk_weights=torch.ones(16, dtype=torch.float32, device="cuda"),
        scratch=torch.empty(2_097_152, dtype=torch.float32, device="cuda"),
        locks=torch.empty(1_024, dtype=torch.int32, device="cuda"),
    )


def make_packed_projection(
    lib: ctypes.CDLL,
    pack_weight,
    size_k: int,
    size_n: int,
    seed: int,
) -> PackedProjection:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    source = (
        torch.randn(
            (size_n, size_k),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.02
    )
    payload = torch.empty(
        size_n * (size_k // 2 + size_k // 16),
        dtype=torch.uint8,
        device="cuda",
    )
    packed = PackedProjection(
        size_k=size_k,
        size_n=size_n,
        weight=torch.empty(size_n * size_k // 2, dtype=torch.uint8, device="cuda"),
        scale=torch.empty(size_n * size_k // 16, dtype=torch.uint8, device="cuda"),
        global_scale=torch.empty(1, dtype=torch.float32, device="cuda"),
        packed_routes=torch.empty(8, dtype=torch.int32, device="cuda"),
        block_experts=torch.empty(1, dtype=torch.int32, device="cuda"),
        route_count=torch.empty(1, dtype=torch.int32, device="cuda"),
        topk_weights=torch.empty(8, dtype=torch.float32, device="cuda"),
        scratch=torch.empty(2_097_152, dtype=torch.float32, device="cuda"),
        locks=torch.empty(1_024, dtype=torch.int32, device="cuda"),
    )
    check_status(
        lib,
        pack_weight(
            device_buffer(source),
            device_buffer(payload),
            device_buffer(packed.weight),
            device_buffer(packed.scale),
            device_buffer(packed.global_scale),
            size_k,
            size_n,
            stream_pointer(),
        ),
        f"pack W4A16 projection K={size_k} N={size_n}",
    )
    torch.cuda.synchronize()
    return packed


def make_state(
    lib: ctypes.CDLL,
    initialize_buffers,
    pack_weight,
    rows: int,
    context_rows: int,
    workspace_bytes: int,
    seed: int,
) -> ClosureState:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    total_rows = context_rows + rows
    hidden = torch.randn(
        (rows, HIDDEN), dtype=torch.bfloat16, device="cuda", generator=generator
    )
    input_norm_weight = torch.randn(
        HIDDEN, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    q_a_weight = (
        torch.randn(
            (Q_LORA_RANK, HIDDEN),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.02
    )
    q_a_norm_weight = torch.randn(
        Q_LORA_RANK, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    normalized_hidden = torch.empty_like(hidden)
    q_a_projected = torch.empty(
        (rows, Q_LORA_RANK), dtype=torch.bfloat16, device="cuda"
    )
    q_a_normalized = torch.empty_like(q_a_projected)
    q_b_output = torch.empty((rows, Q_B_WIDTH), dtype=torch.bfloat16, device="cuda")
    q_nope = torch.empty((rows, HEADS, NOPE_DIM), dtype=torch.bfloat16, device="cuda")
    q_rope = torch.empty((rows, HEADS, ROPE_DIM), dtype=torch.bfloat16, device="cuda")
    k_nope = torch.randn(
        (total_rows, HEADS, NOPE_DIM),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    k_rope = torch.randn(
        (total_rows, ROPE_DIM),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    values = torch.randn(
        (total_rows, HEADS, V_DIM),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    attention_q = torch.empty(
        (rows, HEADS, NOPE_DIM + ROPE_DIM), dtype=torch.bfloat16, device="cuda"
    )
    attention_k = torch.empty(
        (total_rows, HEADS, NOPE_DIM + ROPE_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    attention_output = torch.empty(
        (rows, HEADS, V_DIM), dtype=torch.bfloat16, device="cuda"
    )
    attention_workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")
    hidden_delta = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device="cuda")
    kv_projected = torch.randn(
        (rows, KV_WIDTH),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    positions = torch.arange(
        context_rows, context_rows + rows, dtype=torch.int64, device="cuda"
    ).to(torch.uint32)
    frequency = torch.arange(ROPE_DIM // 2, dtype=torch.float32, device="cuda")
    inverse = torch.pow(THETA, -2.0 * frequency / ROPE_DIM)
    angles = positions.float().unsqueeze(1) * inverse.unsqueeze(0)
    kv_rope_factors = torch.stack(
        (torch.cos(angles), torch.sin(angles)), dim=-1
    ).reshape(rows, ROPE_DIM)
    kv_norm_weight = torch.randn(
        512, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    kv_cache = torch.empty((rows, BF16_CACHE_STRIDE), dtype=torch.uint8, device="cuda")
    kv_attention_ready = torch.empty(
        (rows, KV_WIDTH), dtype=torch.bfloat16, device="cuda"
    )
    q_b = make_packed_projection(lib, pack_weight, Q_LORA_RANK, Q_B_WIDTH, seed + 1_000)
    o_projection = make_packed_projection(
        lib, pack_weight, Q_B_WIDTH, HIDDEN, seed + 2_000
    )
    q_b_split_buffers = [
        q_b.buffers(
            q_a_normalized[offset : offset + 8], q_b_output[offset : offset + 8]
        )
        for offset in range(0, rows, 8)
    ]
    q_b_m16 = make_m16_launch_projection(q_b) if rows > 8 else None
    q_b_m16_buffers = (
        q_b_m16.buffers(q_a_normalized, q_b_output) if q_b_m16 is not None else None
    )
    o_buffers = [
        o_projection.buffers(attention_output[row].reshape(-1), hidden_delta[row])
        for row in range(rows)
    ]
    o_projection_m16 = make_m16_launch_projection(o_projection) if rows > 1 else None
    o_m16_buffers = (
        o_projection_m16.buffers(attention_output, hidden_delta)
        if o_projection_m16 is not None
        else None
    )
    check_status(
        lib,
        initialize_buffers(ctypes.byref(q_b_split_buffers[0]), stream_pointer()),
        "initialize Q-B launch buffers",
    )
    check_status(
        lib,
        initialize_buffers(ctypes.byref(o_buffers[0]), stream_pointer()),
        "initialize O-projection launch buffers",
    )
    for label, projection, buffers in (
        ("Q-B M16", q_b_m16, q_b_m16_buffers),
        ("O-projection M16", o_projection_m16, o_m16_buffers),
    ):
        if projection is not None and buffers is not None:
            check_status(
                lib,
                initialize_buffers(ctypes.byref(buffers), stream_pointer()),
                f"initialize {label} launch buffers",
            )
            projection.packed_routes.copy_(
                torch.arange(16, dtype=torch.int32, device="cuda")
            )
            projection.block_experts.zero_()
            projection.route_count.fill_(16)
            projection.topk_weights.fill_(1.0)
    torch.cuda.synchronize()
    attention_contexts = []
    for row in range(rows):
        visible_rows = context_rows + row + 1
        attention_contexts.append(
            {
                "cuda_stream": torch.cuda.current_stream().cuda_stream,
                "buffers": {
                    "q_nope": descriptor(q_nope[row : row + 1]),
                    "q_rope": descriptor(q_rope[row : row + 1]),
                    "k_nope": descriptor(k_nope[:visible_rows]),
                    "k_rope": descriptor(k_rope[:visible_rows]),
                    "values": descriptor(values[:visible_rows]),
                    "q": descriptor(attention_q[row : row + 1]),
                    "k": descriptor(attention_k[:visible_rows]),
                    "output": descriptor(attention_output[row : row + 1]),
                    "workspace": descriptor(attention_workspace),
                },
            }
        )
    import flashinfer

    qo_indptr = torch.tensor([0, rows], dtype=torch.int32, device="cuda")
    kv_indptr = torch.tensor([0, total_rows], dtype=torch.int32, device="cuda")
    batched_attention_runner = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        attention_workspace,
        kv_layout="NHD",
        use_cuda_graph=True,
        qo_indptr_buf=torch.empty_like(qo_indptr),
        kv_indptr_buf=torch.empty_like(kv_indptr),
        backend="fa2",
    )
    batched_attention_runner.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads=HEADS,
        num_kv_heads=HEADS,
        head_dim_qk=NOPE_DIM + ROPE_DIM,
        head_dim_vo=V_DIM,
        causal=True,
        sm_scale=(NOPE_DIM + ROPE_DIM) ** -0.5,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
    )
    return ClosureState(
        hidden=hidden,
        input_norm_weight=input_norm_weight,
        q_a_weight=q_a_weight,
        q_a_norm_weight=q_a_norm_weight,
        normalized_hidden=normalized_hidden,
        q_a_projected=q_a_projected,
        q_a_normalized=q_a_normalized,
        q_b_output=q_b_output,
        q_nope=q_nope,
        q_rope=q_rope,
        k_nope=k_nope,
        k_rope=k_rope,
        values=values,
        attention_q=attention_q,
        attention_k=attention_k,
        attention_output=attention_output,
        attention_workspace=attention_workspace,
        hidden_delta=hidden_delta,
        kv_projected=kv_projected,
        kv_rope_factors=kv_rope_factors,
        kv_norm_weight=kv_norm_weight,
        kv_cache=kv_cache,
        kv_attention_ready=kv_attention_ready,
        q_b=q_b,
        q_b_m16=q_b_m16,
        o_projection=o_projection,
        o_projection_m16=o_projection_m16,
        q_b_split_buffers=q_b_split_buffers,
        q_b_m16_buffers=q_b_m16_buffers,
        o_buffers=o_buffers,
        o_m16_buffers=o_m16_buffers,
        attention_contexts=attention_contexts,
        batched_attention_runner=batched_attention_runner,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a shared speculative target-layer closure at MTP and dSpark "
            "verification widths."
        )
    )
    parser.add_argument("--native-lib", type=Path, required=True)
    parser.add_argument("--rows", default="1,4,6,8")
    parser.add_argument("--context", type=int, default=1_024)
    parser.add_argument("--weight-sets", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows_values = parse_int_list(args.rows, "rows")
    if any(rows > 16 for rows in rows_values):
        parser.error("rows must not exceed the benchmark-only M16 capacity")
    if args.context < 1:
        parser.error("context must be positive")
    if min(args.weight_sets, args.iterations, args.repeats) < 1 or args.warmup < 0:
        parser.error(
            "weight-sets/iterations/repeats must be positive and warmup nonnegative"
        )

    torch.manual_seed(args.seed)
    torch.cuda.init()
    from flashinfer.prefill import SINGLE_KERNEL_TMP_SIZE

    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    lib = ctypes.CDLL(str(args.native_lib.resolve()))
    lib.glmrt_last_error.argtypes = (ctypes.c_char_p, ctypes.c_size_t)
    lib.glmrt_last_error.restype = ctypes.c_int
    lib.glmrt_cuda_b12x_coordinator_aot_init.restype = ctypes.c_int
    check_status(
        lib,
        lib.glmrt_cuda_b12x_coordinator_aot_init(),
        "coordinator AOT initialization",
    )
    pack_weight = configure(
        lib,
        "glmrt_cuda_b12x_coordinator_w4a16_quantize_pack_weight_async",
        (
            DeviceBuffer,
            DeviceBuffer,
            DeviceBuffer,
            DeviceBuffer,
            DeviceBuffer,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ),
    )
    initialize_buffers = configure(
        lib,
        "glmrt_cuda_b12x_coordinator_w4a16_initialize_launch_buffers_async",
        (ctypes.POINTER(CoordinatorBuffers), ctypes.c_void_p),
    )
    q_b_launch = configure(
        lib,
        "glmrt_cuda_b12x_coordinator_w4a16_q_b_m8_async",
        (ctypes.POINTER(CoordinatorBuffers), ctypes.c_size_t, ctypes.c_void_p),
    )
    q_b_m16_launch = configure(
        lib,
        "glmrt_cuda_b12x_coordinator_w4a16_q_b_m16_candidate_async",
        (ctypes.POINTER(CoordinatorBuffers), ctypes.c_size_t, ctypes.c_void_p),
    )
    o_launch = configure(
        lib,
        "glmrt_cuda_b12x_coordinator_w4a16_o_proj_m1_async",
        (ctypes.POINTER(CoordinatorBuffers), ctypes.c_void_p),
    )
    o_m16_launch = configure(
        lib,
        "glmrt_cuda_b12x_coordinator_w4a16_o_proj_m16_candidate_async",
        (ctypes.POINTER(CoordinatorBuffers), ctypes.c_size_t, ctypes.c_void_p),
    )
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
    q_a_candidate = configure(
        lib,
        "glmrt_cuda_mla_scalar_qa_batched_norm_candidate_async",
        (
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
            ctypes.c_float,
            ctypes.c_void_p,
        ),
    )
    kv_finalize = configure(
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

    results = []
    for rows in rows_values:
        states = [
            make_state(
                lib,
                initialize_buffers,
                pack_weight,
                rows,
                args.context,
                SINGLE_KERNEL_TMP_SIZE,
                args.seed + rows * 10_000 + index * 100,
            )
            for index in range(args.weight_sets)
        ]
        attention_kwargs = [
            {
                "rows": args.context + row + 1,
                "query_row_offset": args.context + row,
                "query_rows": 1,
                "heads": HEADS,
                "nope_dim": NOPE_DIM,
                "rope_dim": ROPE_DIM,
                "v_dim": V_DIM,
                "scale": (NOPE_DIM + ROPE_DIM) ** -0.5,
            }
            for row in range(rows)
        ]

        def operations(state: ClosureState):
            def qa_reference() -> None:
                for row in range(rows):
                    check_status(
                        lib,
                        rmsnorm(
                            pointer(state.hidden[row]),
                            pointer(state.input_norm_weight),
                            pointer(state.normalized_hidden[row]),
                            1,
                            HIDDEN,
                            EPS,
                            stream_pointer(),
                        ),
                        "reference input RMSNorm",
                    )
                    check_status(
                        lib,
                        linear(
                            pointer(state.normalized_hidden[row]),
                            pointer(state.q_a_weight),
                            ctypes.c_void_p(),
                            pointer(state.q_a_projected[row]),
                            1,
                            HIDDEN,
                            Q_LORA_RANK,
                            stream_pointer(),
                        ),
                        "reference scalar Q-A",
                    )
                    check_status(
                        lib,
                        rmsnorm(
                            pointer(state.q_a_projected[row]),
                            pointer(state.q_a_norm_weight),
                            pointer(state.q_a_normalized[row]),
                            1,
                            Q_LORA_RANK,
                            EPS,
                            stream_pointer(),
                        ),
                        "reference Q-A RMSNorm",
                    )

            def qa_candidate_launch() -> None:
                if rows == 1:
                    qa_reference()
                    return
                check_status(
                    lib,
                    q_a_candidate(
                        pointer(state.hidden),
                        pointer(state.input_norm_weight),
                        pointer(state.normalized_hidden),
                        pointer(state.q_a_weight),
                        pointer(state.q_a_projected),
                        pointer(state.q_a_norm_weight),
                        pointer(state.q_a_normalized),
                        rows,
                        HIDDEN,
                        Q_LORA_RANK,
                        EPS,
                        stream_pointer(),
                    ),
                    "batched-norm Q-A candidate",
                )

            def split_q_b_output() -> None:
                q_view = state.q_b_output[:rows].view(rows, HEADS, NOPE_DIM + ROPE_DIM)
                state.q_nope.copy_(q_view[..., :NOPE_DIM])
                state.q_rope.copy_(q_view[..., NOPE_DIM:])

            def q_b_reference_and_split() -> None:
                for index, buffers in enumerate(state.q_b_split_buffers):
                    active_rows = min(8, rows - index * 8)
                    check_status(
                        lib,
                        q_b_launch(
                            ctypes.byref(buffers), active_rows, stream_pointer()
                        ),
                        "split-M8 Q-B reference",
                    )
                split_q_b_output()

            def q_b_candidate_and_split() -> None:
                if state.q_b_m16_buffers is None:
                    q_b_reference_and_split()
                    return
                check_status(
                    lib,
                    q_b_m16_launch(
                        ctypes.byref(state.q_b_m16_buffers), rows, stream_pointer()
                    ),
                    "M16 Q-B candidate",
                )
                split_q_b_output()

            def kv_finalize_launch() -> None:
                check_status(
                    lib,
                    kv_finalize(
                        pointer(state.kv_projected),
                        pointer(state.kv_rope_factors),
                        pointer(state.kv_norm_weight),
                        pointer(state.kv_cache),
                        pointer(state.kv_attention_ready),
                        ctypes.c_void_p(),
                        rows,
                        KV_WIDTH * 2,
                        BF16_CACHE_STRIDE,
                        KV_WIDTH * 2,
                        0,
                        0,
                        0,
                        EPS,
                        stream_pointer(),
                    ),
                    "BF16 KV finalize",
                )

            def attention_reference_launch() -> None:
                for context, kwargs in zip(
                    state.attention_contexts, attention_kwargs, strict=True
                ):
                    context["cuda_stream"] = torch.cuda.current_stream().cuda_stream
                    capture_flashinfer_mla_rope_attention(context, **kwargs)

            def prepare_batched_attention_inputs() -> None:
                torch.cat((state.q_nope, state.q_rope), dim=-1, out=state.attention_q)
                torch.cat(
                    (
                        state.k_nope,
                        state.k_rope[:, None, :].expand(-1, HEADS, -1),
                    ),
                    dim=-1,
                    out=state.attention_k,
                )

            def batched_attention_launch() -> None:
                prepare_batched_attention_inputs()
                state.batched_attention_runner.run(
                    state.attention_q,
                    state.attention_k,
                    state.values,
                    out=state.attention_output,
                )

            def o_projection_launch(row: int) -> None:
                check_status(
                    lib,
                    o_launch(ctypes.byref(state.o_buffers[row]), stream_pointer()),
                    "O-projection",
                )

            def o_projection_candidate_launch() -> None:
                if state.o_m16_buffers is None:
                    o_projection_launch(0)
                    return
                check_status(
                    lib,
                    o_m16_launch(
                        ctypes.byref(state.o_m16_buffers), rows, stream_pointer()
                    ),
                    "M16 O-projection candidate",
                )

            def closure(
                qa_launch: Callable[[], None],
                q_b_launch_closure: Callable[[], None],
                attention_launch_closure: Callable[[], None],
                o_launch_closure: Callable[[], None],
            ) -> None:
                qa_launch()
                q_b_launch_closure()
                kv_finalize_launch()
                attention_launch_closure()
                o_launch_closure()

            def o_projection_reference_launch() -> None:
                for row in range(rows):
                    o_projection_launch(row)

            return (
                qa_reference,
                qa_candidate_launch,
                q_b_reference_and_split,
                q_b_candidate_and_split,
                kv_finalize_launch,
                attention_reference_launch,
                batched_attention_launch,
                o_projection_launch,
                o_projection_reference_launch,
                o_projection_candidate_launch,
                closure,
            )

        for state in states:
            for context, kwargs in zip(
                state.attention_contexts, attention_kwargs, strict=True
            ):
                prepare_flashinfer_mla_rope_attention(context, **kwargs)
        torch.cuda.synchronize()

        stage_sequences = []
        closure_reference_sequences = []
        closure_candidate_sequences = []
        closure_batched_attention_sequences = []
        stage_buckets: dict[str, list[list[torch.cuda.CUDAGraph]]] = {
            "qa_reference": [],
            "q_b_split_m8_reference": [],
            "q_b_m16_candidate": [],
            "kv_finalize": [],
            "attention_progressive_reference": [],
            "attention_batched_causal": [],
            "o_projection_scalar_reference": [],
            "o_projection_m16_candidate": [],
        }
        exact_checks = []
        q_b_m16_checks = []
        batched_attention_checks = []
        for state in states:
            (
                qa_reference,
                qa_candidate_launch,
                q_b_reference_and_split,
                q_b_candidate_and_split,
                kv_finalize_launch,
                attention_reference_launch,
                batched_attention_launch,
                o_projection_launch,
                o_projection_reference_launch,
                o_projection_candidate_launch,
                closure,
            ) = operations(state)
            qa_graph = capture(qa_reference)
            q_b_reference_graph = capture(q_b_reference_and_split)
            q_b_candidate_graph = capture(q_b_candidate_and_split)
            kv_graph = capture(kv_finalize_launch)
            attention_reference_graph = capture(attention_reference_launch)
            batched_attention_graph = capture(batched_attention_launch)
            o_graphs = [
                capture(lambda row=row: o_projection_launch(row)) for row in range(rows)
            ]
            o_candidate_graph = capture(o_projection_candidate_launch)
            stage_sequences.append(
                [
                    qa_graph,
                    q_b_reference_graph,
                    kv_graph,
                    attention_reference_graph,
                    *o_graphs,
                ]
            )
            stage_buckets["qa_reference"].append([qa_graph])
            stage_buckets["q_b_split_m8_reference"].append([q_b_reference_graph])
            stage_buckets["q_b_m16_candidate"].append([q_b_candidate_graph])
            stage_buckets["kv_finalize"].append([kv_graph])
            stage_buckets["attention_progressive_reference"].append(
                [attention_reference_graph]
            )
            stage_buckets["attention_batched_causal"].append([batched_attention_graph])
            stage_buckets["o_projection_scalar_reference"].append(o_graphs)
            stage_buckets["o_projection_m16_candidate"].append([o_candidate_graph])
            reference_graph = capture(
                lambda: closure(
                    qa_reference,
                    q_b_reference_and_split,
                    attention_reference_launch,
                    o_projection_reference_launch,
                )
            )
            reference_graph.replay()
            torch.cuda.synchronize()
            reference = {
                "q_a_projected": state.q_a_projected.clone(),
                "q_a_normalized": state.q_a_normalized.clone(),
                "q_b_output": state.q_b_output[:rows].clone(),
                "q_nope": state.q_nope.clone(),
                "q_rope": state.q_rope.clone(),
                "kv_cache": state.kv_cache.clone(),
                "kv_attention_ready": state.kv_attention_ready.clone(),
                "attention_output": state.attention_output.clone(),
                "hidden_delta": state.hidden_delta.clone(),
            }
            q_b_candidate_graph.replay()
            torch.cuda.synchronize()
            q_b_m16_checks.append(
                compare_tensors(reference["q_b_output"], state.q_b_output[:rows])
            )
            candidate_graph = capture(
                lambda: closure(
                    qa_candidate_launch,
                    q_b_reference_and_split,
                    attention_reference_launch,
                    o_projection_candidate_launch,
                )
            )
            candidate_graph.replay()
            torch.cuda.synchronize()
            exact = {
                name: bool(torch.equal(expected, getattr(state, name)[:rows]))
                for name, expected in reference.items()
            }
            exact_checks.append(exact)
            batched_attention_candidate_graph = capture(
                lambda: closure(
                    qa_candidate_launch,
                    q_b_reference_and_split,
                    batched_attention_launch,
                    o_projection_candidate_launch,
                )
            )
            batched_attention_candidate_graph.replay()
            torch.cuda.synchronize()
            batched_attention_checks.append(
                {
                    "attention_output": compare_tensors(
                        reference["attention_output"], state.attention_output
                    ),
                    "hidden_delta": compare_tensors(
                        reference["hidden_delta"], state.hidden_delta
                    ),
                }
            )
            closure_reference_sequences.append([reference_graph])
            closure_candidate_sequences.append([candidate_graph])
            closure_batched_attention_sequences.append(
                [batched_attention_candidate_graph]
            )

        stage_timing = measure_sequences(
            stage_sequences, args.warmup, args.iterations, args.repeats
        )
        closure_reference_timing = measure_sequences(
            closure_reference_sequences,
            args.warmup,
            args.iterations,
            args.repeats,
        )
        closure_candidate_timing = measure_sequences(
            closure_candidate_sequences,
            args.warmup,
            args.iterations,
            args.repeats,
        )
        closure_batched_attention_timing = measure_sequences(
            closure_batched_attention_sequences,
            args.warmup,
            args.iterations,
            args.repeats,
        )
        component_timing = {
            label: measure_sequences(
                sequences, args.warmup, args.iterations, args.repeats
            )["gpu_ms_per_closure"]
            for label, sequences in stage_buckets.items()
        }
        stage_gpu = stage_timing["gpu_ms_per_closure"]["median"]
        reference_gpu = closure_reference_timing["gpu_ms_per_closure"]["median"]
        candidate_gpu = closure_candidate_timing["gpu_ms_per_closure"]["median"]
        batched_attention_gpu = closure_batched_attention_timing["gpu_ms_per_closure"][
            "median"
        ]
        result = {
            "benchmark": "mtp_layer_closure",
            "context_rows": args.context,
            "exact": all(all(check.values()) for check in exact_checks),
            "exact_checks": exact_checks,
            "gpu": properties.name,
            "note": (
                "Benchmark-only expanded-BF16 target layer. The exact candidate keeps "
                "split-M8 Q-B and progressive attention; causal batched attention and "
                "M16 Q-B remain numerical acceptance gates. Serving dispatch is unchanged."
            ),
            "rows": rows,
            "stage_replay": stage_timing,
            "single_graph_reference": closure_reference_timing,
            "single_graph_exact_batched_candidate": closure_candidate_timing,
            "single_graph_batched_attention_candidate": (
                closure_batched_attention_timing
            ),
            "q_b_m16_checks": q_b_m16_checks,
            "batched_attention_checks": batched_attention_checks,
            "component_gpu_ms": component_timing,
            "stage_to_reference_graph_speedup": stage_gpu / reference_gpu,
            "reference_to_exact_batched_speedup": reference_gpu / candidate_gpu,
            "reference_to_batched_attention_speedup": (
                reference_gpu / batched_attention_gpu
            ),
            "stage_to_candidate_graph_speedup": stage_gpu / candidate_gpu,
            "weight_sets": args.weight_sets,
            "weight_working_set_bytes": sum(
                state.q_a_weight.numel() * state.q_a_weight.element_size()
                + state.q_b.weight_bytes
                + state.o_projection.weight_bytes
                for state in states
            ),
        }
        if not result["exact"]:
            raise RuntimeError(
                f"exact verifier candidate diverged at rows={rows}: {exact_checks}"
            )
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

        del states
        torch.cuda.empty_cache()

    report = {
        "benchmark": "mtp_layer_closure_summary",
        "context_rows": args.context,
        "exact": all(result["exact"] for result in results),
        "gpu": properties.name,
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
