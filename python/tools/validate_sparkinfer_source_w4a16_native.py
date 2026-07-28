#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import statistics
from pathlib import Path

import torch


HIDDEN = 6144
INTERMEDIATE = 512
EXPERTS = 256
TEST_EXPERTS = 8
TOP_K = 8
MAX_ROUTE_SLOTS = 32_512
MAX_ROUTE_BLOCKS = 508
MAX_SCRATCH_ELEMENTS = 3_145_728
LOCK_ELEMENTS = 1_026
E4M3_ONE = 0x38


class DeviceBuffer(ctypes.Structure):
    _fields_ = (
        ("ptr", ctypes.c_void_p),
        ("bytes", ctypes.c_size_t),
        ("device_id", ctypes.c_int),
        ("flags", ctypes.c_uint64),
    )


class SparkW4A16Buffers(ctypes.Structure):
    _fields_ = tuple(
        (name, DeviceBuffer)
        for name in (
            "input",
            "w13_weight",
            "w2_weight",
            "fc1_output",
            "activated",
            "output",
            "w13_scale",
            "w2_scale",
            "w13_global_scale",
            "w2_global_scale",
            "packed_route_indices",
            "block_expert_ids",
            "packed_route_count",
            "topk_weights",
            "fc1_scratch",
            "fc2_scratch",
            "locks",
            "micro_w13_global_scale",
            "micro_w2_global_scale",
            "barrier_count",
            "barrier_epoch",
        )
    )


def device_buffer(
    tensor: torch.Tensor, *, advertised_bytes: int | None = None
) -> DeviceBuffer:
    return DeviceBuffer(
        tensor.data_ptr(),
        tensor.numel() * tensor.element_size()
        if advertised_bytes is None
        else advertised_bytes,
        tensor.device.index or 0,
        0,
    )


def check_status(library: ctypes.CDLL, status: int, action: str) -> None:
    if status == 0:
        return
    error = ctypes.create_string_buffer(512)
    library.glmrt_last_error(error, len(error))
    raise RuntimeError(
        f"{action} failed with status {status}: {error.value.decode()}"
    )


def decode_nvfp4_payload(payload: torch.Tensor) -> torch.Tensor:
    packed = payload[:, : HIDDEN // 2]
    low = packed & 0x0F
    high = packed >> 4
    codes = torch.stack((low, high), dim=-1).reshape(payload.shape[0], HIDDEN)
    values = torch.tensor(
        (
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ),
        dtype=torch.float32,
        device=payload.device,
    )
    scales = (
        payload[:, HIDDEN // 2 :]
        .view(torch.float8_e4m3fn)
        .float()
        .repeat_interleave(16, dim=1)
    )
    return (values[codes.long()] * scales).to(torch.bfloat16)


def swizzle_modelopt_scales(logical: torch.Tensor) -> torch.Tensor:
    if logical.ndim != 3:
        raise ValueError(f"expected [experts, rows, cols] scales, got {logical.shape}")
    experts, rows, cols = logical.shape
    if rows % 128 != 0 or cols % 4 != 0:
        raise ValueError(
            f"ModelOpt scale shape must align to 128x4, got {rows}x{cols}"
        )
    return (
        logical.reshape(experts, rows // 128, 4, 32, cols // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .reshape(experts, rows, cols)
    )


def pack_routes(
    topk_ids: torch.Tensor, route_indices: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = topk_ids.shape[0]
    sentinel = rows * TOP_K
    ids = topk_ids.cpu().tolist()
    routes = route_indices.cpu().tolist()
    by_expert: list[list[int]] = [[] for _ in range(EXPERTS)]
    for row in range(rows):
        for slot in range(TOP_K):
            by_expert[ids[row][slot]].append(routes[row][slot])
    packed: list[int] = []
    block_experts: list[int] = []
    for expert, expert_routes in enumerate(by_expert):
        for start in range(0, len(expert_routes), block_size):
            block = expert_routes[start : start + block_size]
            packed.extend(block)
            packed.extend([sentinel] * (block_size - len(block)))
            block_experts.append(expert)
    return (
        torch.tensor(packed, dtype=torch.int32, device=topk_ids.device),
        torch.tensor(
            block_experts, dtype=torch.int32, device=topk_ids.device
        ),
    )


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    difference = actual_f32 - expected_f32
    reference_rms = expected_f32.square().mean().sqrt().item()
    error_rms = difference.square().mean().sqrt().item()
    cosine = torch.nn.functional.cosine_similarity(
        actual_f32.reshape(1, -1), expected_f32.reshape(1, -1)
    ).item()
    return {
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
        "relative_rmse": error_rms / max(reference_rms, 1e-12),
        "cosine": cosine,
        "exact_fraction": actual.eq(expected).float().mean().item(),
        "actual_finite_fraction": actual_f32.isfinite().float().mean().item(),
        "expected_finite_fraction": expected_f32.isfinite().float().mean().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate native latest-SparkInfer source-layout W4A16 for every "
            "physical M=1..16 against upstream and repeated M=1."
        )
    )
    parser.add_argument("--native-lib", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.repeats) < 1:
        parser.error("warmup, iterations, and repeats must be positive")
    direct_max_rows = int(
        os.environ.get("GLMRT_SPARKINFER_SOURCE_W4A16_DIRECT_MAX_ROWS", "2")
    )
    if direct_max_rows < 0 or direct_max_rows > 8:
        parser.error(
            "GLMRT_SPARKINFER_SOURCE_W4A16_DIRECT_MAX_ROWS must be in 0..8"
        )

    from sparkinfer.moe._shared.kernels.w4a16.kernel import run_w4a16_moe
    from sparkinfer.moe._shared.kernels.w4a16.prepare import (
        make_w4a16_packed_buffers,
        prepare_w4a16_modelopt_native_weights,
    )

    torch.manual_seed(args.seed)
    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    w13 = torch.randint(
        0,
        256,
        (TEST_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2),
        dtype=torch.uint8,
        device=device,
    )
    w2 = torch.randint(
        0,
        256,
        (TEST_EXPERTS, HIDDEN, INTERMEDIATE // 2),
        dtype=torch.uint8,
        device=device,
    )
    w13_scale_logical = torch.randint(
        0x20,
        0x41,
        (TEST_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 16),
        dtype=torch.uint8,
        device=device,
    ).view(torch.float8_e4m3fn)
    w2_scale_logical = torch.randint(
        0x20,
        0x41,
        (TEST_EXPERTS, HIDDEN, INTERMEDIATE // 16),
        dtype=torch.uint8,
        device=device,
    ).view(torch.float8_e4m3fn)
    w13_scale = swizzle_modelopt_scales(w13_scale_logical)
    w2_scale = swizzle_modelopt_scales(w2_scale_logical)
    global_scale = torch.full(
        (TEST_EXPERTS,), 1.0e-3, dtype=torch.float32, device=device
    )
    prepared = prepare_w4a16_modelopt_native_weights(
        w13,
        w13_scale,
        global_scale,
        w2,
        w2_scale,
        global_scale,
        activation="silu",
        params_dtype=torch.bfloat16,
        w13_layout="w13",
    )

    capacity = 16
    payload_row_bytes = HIDDEN // 2 + HIDDEN // 16
    payload = torch.empty(
        (capacity, payload_row_bytes), dtype=torch.uint8, device=device
    )
    payload[:, : HIDDEN // 2].random_(0, 256)
    payload[:, HIDDEN // 2 :].fill_(E4M3_ONE)
    decoded_input = decode_nvfp4_payload(payload)
    topk_ids = torch.stack(
        [
            (torch.arange(TOP_K, device=device, dtype=torch.int32) + row)
            % TEST_EXPERTS
            for row in range(capacity)
        ]
    )
    topk_weights = torch.softmax(
        torch.randn(capacity, TOP_K, device=device), dim=-1
    )

    input_bf16 = torch.empty(
        (capacity, HIDDEN), dtype=torch.bfloat16, device=device
    )
    fc1_output = torch.empty(
        capacity * TOP_K * 2 * INTERMEDIATE,
        dtype=torch.bfloat16,
        device=device,
    )
    activated = torch.empty(
        capacity * TOP_K * INTERMEDIATE,
        dtype=torch.bfloat16,
        device=device,
    )
    routed_output = torch.empty(
        (capacity * TOP_K, HIDDEN), dtype=torch.bfloat16, device=device
    )
    packed_routes = torch.zeros(
        MAX_ROUTE_SLOTS, dtype=torch.int32, device=device
    )
    block_experts = torch.zeros(
        MAX_ROUTE_BLOCKS, dtype=torch.int32, device=device
    )
    packed_route_count = torch.zeros(1, dtype=torch.int32, device=device)
    fc1_scratch = torch.empty(
        MAX_SCRATCH_ELEMENTS, dtype=torch.float32, device=device
    )
    fc2_scratch = torch.empty_like(fc1_scratch)
    locks = torch.zeros(LOCK_ELEMENTS, dtype=torch.int32, device=device)
    barrier_count = torch.zeros(1, dtype=torch.int32, device=device)
    barrier_epoch = torch.zeros(1, dtype=torch.int32, device=device)

    buffers = SparkW4A16Buffers(
        device_buffer(input_bf16),
        device_buffer(
            prepared.w13,
            advertised_bytes=EXPERTS
            * 2
            * INTERMEDIATE
            * HIDDEN
            // 2,
        ),
        device_buffer(
            prepared.w2,
            advertised_bytes=EXPERTS * HIDDEN * INTERMEDIATE // 2,
        ),
        device_buffer(fc1_output),
        device_buffer(activated),
        device_buffer(routed_output),
        device_buffer(
            prepared.w13_scale,
            advertised_bytes=EXPERTS
            * 2
            * INTERMEDIATE
            * HIDDEN
            // 16,
        ),
        device_buffer(
            prepared.w2_scale,
            advertised_bytes=EXPERTS * HIDDEN * INTERMEDIATE // 16,
        ),
        device_buffer(
            prepared.w13_global_scale, advertised_bytes=EXPERTS * 4
        ),
        device_buffer(
            prepared.w2_global_scale, advertised_bytes=EXPERTS * 4
        ),
        device_buffer(packed_routes),
        device_buffer(block_experts),
        device_buffer(packed_route_count),
        device_buffer(topk_weights),
        device_buffer(fc1_scratch),
        device_buffer(fc2_scratch),
        device_buffer(locks),
        device_buffer(
            prepared.micro_w13_global_scale,
            advertised_bytes=EXPERTS * 4,
        ),
        device_buffer(
            prepared.micro_w2_global_scale,
            advertised_bytes=EXPERTS * 4,
        ),
        device_buffer(barrier_count),
        device_buffer(barrier_epoch),
    )

    library = ctypes.CDLL(str(args.native_lib.resolve()))
    library.glmrt_last_error.argtypes = (
        ctypes.c_char_p,
        ctypes.c_size_t,
    )
    library.glmrt_last_error.restype = ctypes.c_int
    launch = library.glmrt_cuda_sparkinfer_source_w4a16_topk8_nvfp4_async
    launch.argtypes = (
        ctypes.POINTER(SparkW4A16Buffers),
        DeviceBuffer,
        ctypes.c_size_t,
        DeviceBuffer,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    launch.restype = ctypes.c_int
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    route_indices = torch.arange(
        capacity * TOP_K, dtype=torch.int32, device=device
    ).reshape(capacity, TOP_K)

    def stage_routes(
        rows: int,
        ids: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
    ) -> None:
        if rows <= direct_max_rows:
            return
        ids = topk_ids[:rows] if ids is None else ids
        indices = route_indices[:rows] if indices is None else indices
        packed, experts = pack_routes(
            ids, indices, block_size=8
        )
        packed_routes.zero_()
        block_experts.zero_()
        packed_routes[: packed.numel()].copy_(packed)
        block_experts[: experts.numel()].copy_(experts)
        packed_route_count.fill_(packed.numel())

    def native_launch(rows: int) -> None:
        stage_routes(rows)
        status = launch(
            ctypes.byref(buffers),
            device_buffer(payload[:rows]),
            payload_row_bytes,
            device_buffer(topk_ids[:rows]),
            rows,
            stream,
        )
        check_status(library, status, f"native source W4A16 M={rows}")

    baseline_rows = []
    baseline_topk_weights = torch.empty_like(topk_weights)
    for row in range(capacity):
        row_buffers = SparkW4A16Buffers.from_buffer_copy(buffers)
        if direct_max_rows == 0:
            baseline_topk_weights.zero_()
            baseline_topk_weights[0].copy_(topk_weights[row])
            row_buffers.topk_weights = device_buffer(baseline_topk_weights)
            stage_routes(
                1,
                topk_ids[row : row + 1],
                torch.arange(TOP_K, dtype=torch.int32, device=device).reshape(1, TOP_K),
            )
        else:
            row_buffers.topk_weights = device_buffer(topk_weights[row : row + 1])
        status = launch(
            ctypes.byref(row_buffers),
            device_buffer(payload[row : row + 1]),
            payload_row_bytes,
            device_buffer(topk_ids[row : row + 1]),
            1,
            stream,
        )
        check_status(library, status, f"native source W4A16 baseline row {row}")
        torch.cuda.synchronize()
        baseline_rows.append(input_bf16[0].clone())
    repeated_m1 = torch.stack(baseline_rows)

    upstream_outputs: dict[int, torch.Tensor] = {}
    for rows in (1, 16):
        upstream_buffers = make_w4a16_packed_buffers(
            prepared,
            m=rows,
            topk=TOP_K,
            dtype=torch.bfloat16,
            device=device,
        )
        upstream_outputs[rows] = run_w4a16_moe(
            decoded_input[:rows].contiguous(),
            prepared,
            topk_weights[:rows].contiguous(),
            topk_ids[:rows].contiguous(),
            activation="silu",
            intermediate_cache13=upstream_buffers.intermediate_cache13,
            intermediate_cache2=upstream_buffers.intermediate_cache2,
            output=upstream_buffers.output,
            fc1_c_tmp=upstream_buffers.fc1_c_tmp,
            fc2_c_tmp=upstream_buffers.fc2_c_tmp,
            packed_route_indices=upstream_buffers.packed_route_indices,
            block_expert_ids=upstream_buffers.block_expert_ids,
            packed_route_count=upstream_buffers.packed_route_count,
            expert_offsets=upstream_buffers.expert_offsets,
        ).clone()
        torch.cuda.synchronize()

    results = []
    for rows in range(1, capacity + 1):
        native_launch(rows)
        torch.cuda.synchronize()
        actual = input_bf16[:rows].clone()
        parity = error_metrics(actual, repeated_m1[:rows])
        upstream = (
            error_metrics(actual, upstream_outputs[rows])
            if rows in upstream_outputs
            else None
        )

        for _ in range(args.warmup):
            native_launch(rows)
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iterations):
                native_launch(rows)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / args.iterations)
        results.append(
            {
                "m": rows,
                "kernel": "direct" if rows <= direct_max_rows else "generic_m16",
                "median_ms": statistics.median(samples),
                "samples_ms": samples,
                "vs_repeated_m1": parity,
                "vs_upstream": upstream,
            }
        )

    failures = [
        result
        for result in results
        if not math.isfinite(result["vs_repeated_m1"]["relative_rmse"])
        or result["vs_repeated_m1"]["relative_rmse"] > 0.02
        or result["vs_repeated_m1"]["cosine"] < 0.999
        or (
            result["vs_upstream"] is not None
            and (
                not math.isfinite(result["vs_upstream"]["relative_rmse"])
                or result["vs_upstream"]["relative_rmse"] > 0.02
                or result["vs_upstream"]["cosine"] < 0.999
            )
        )
    ]
    report = {
        "seed": args.seed,
        "direct_max_rows": direct_max_rows,
        "native_library": str(args.native_lib.resolve()),
        "sparkinfer_source": str(
            Path(__import__("sparkinfer").__file__).resolve()
        ),
        "rows": results,
        "passed": not failures,
        "failed_m": [result["m"] for result in failures],
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
