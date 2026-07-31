#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
from pathlib import Path

import torch


class DeviceBuffer(ctypes.Structure):
    _fields_ = (
        ("ptr", ctypes.c_void_p),
        ("bytes", ctypes.c_size_t),
        ("device_id", ctypes.c_int),
        ("flags", ctypes.c_uint64),
    )


class CoordinatorBuffers(ctypes.Structure):
    _fields_ = tuple(
        (name, DeviceBuffer)
        for name in (
            "input",
            "weight",
            "output",
            "scale",
            "global_scale",
            "packed_route_indices",
            "block_expert_ids",
            "packed_route_count",
            "topk_weights",
            "c_tmp",
            "locks",
        )
    )


@dataclass(frozen=True)
class Projection:
    label: str
    size_n: int
    size_k: int
    launch_symbol: str
    max_rows: int


PROJECTIONS = {
    "q_b": Projection(
        "Q-B", 16_384, 2_048, "glmrt_cuda_b12x_coordinator_w4a16_q_b_m8_async", 8
    ),
    "q_b_m16_candidate": Projection(
        "Q-B M16 candidate",
        16_384,
        2_048,
        "glmrt_cuda_b12x_coordinator_w4a16_q_b_m16_candidate_async",
        16,
    ),
    "o_proj": Projection(
        "O-projection",
        6_144,
        16_384,
        "glmrt_cuda_b12x_coordinator_w4a16_o_proj_m1_async",
        1,
    ),
    "o_proj_m16_candidate": Projection(
        "O-projection M16 candidate",
        6_144,
        16_384,
        "glmrt_cuda_b12x_coordinator_w4a16_o_proj_m16_candidate_async",
        16,
    ),
    "o_proj_tn64_candidate": Projection(
        "O-projection TN64 candidate",
        6_144,
        16_384,
        "glmrt_cuda_b12x_coordinator_w4a16_o_proj_m1_tn64_candidate_async",
        1,
    ),
}


def device_buffer(tensor: torch.Tensor) -> DeviceBuffer:
    return DeviceBuffer(tensor.data_ptr(), tensor.numel() * tensor.element_size(), 0, 0)


def check_status(lib: ctypes.CDLL, status: int, action: str) -> None:
    if status == 0:
        return
    error = ctypes.create_string_buffer(512)
    lib.glmrt_last_error_message(error, len(error))
    raise RuntimeError(f"{action} failed with status {status}: {error.value.decode()}")


def configure_abi(lib: ctypes.CDLL, projection: Projection) -> ctypes._CFuncPtr:
    lib.glmrt_cuda_b12x_coordinator_aot_init.restype = ctypes.c_int
    lib.glmrt_cuda_b12x_coordinator_w4a16_quantize_pack_weight_async.argtypes = (
        DeviceBuffer,
        DeviceBuffer,
        DeviceBuffer,
        DeviceBuffer,
        DeviceBuffer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    lib.glmrt_cuda_b12x_coordinator_w4a16_quantize_pack_weight_async.restype = ctypes.c_int
    lib.glmrt_cuda_b12x_coordinator_w4a16_initialize_launch_buffers_async.argtypes = (
        ctypes.POINTER(CoordinatorBuffers),
        ctypes.c_void_p,
    )
    lib.glmrt_cuda_b12x_coordinator_w4a16_initialize_launch_buffers_async.restype = ctypes.c_int
    launch = getattr(lib, projection.launch_symbol)
    if projection.max_rows > 1:
        launch.argtypes = (
            ctypes.POINTER(CoordinatorBuffers),
            ctypes.c_size_t,
            ctypes.c_void_p,
        )
    else:
        launch.argtypes = (ctypes.POINTER(CoordinatorBuffers), ctypes.c_void_p)
    launch.restype = ctypes.c_int
    return launch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and time coordinator decode-only SparkInfer W4A16 AOT kernels."
    )
    parser.add_argument("--native-lib", type=Path, required=True)
    parser.add_argument("--projection", choices=tuple(PROJECTIONS), required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--graph",
        action="store_true",
        help="measure production-like CUDA graph replay instead of eager launches",
    )
    args = parser.parse_args()

    if args.warmup < 1 or args.iterations < 1:
        parser.error("warmup and iterations must be positive")

    torch.manual_seed(args.seed)
    torch.cuda.init()
    projection = PROJECTIONS[args.projection]
    if not 1 <= args.rows <= projection.max_rows:
        parser.error(
            f"--rows must be between 1 and {projection.max_rows} for {args.projection}"
        )
    lib = ctypes.CDLL(str(args.native_lib.resolve()))
    launch = configure_abi(lib, projection)
    check_status(lib, lib.glmrt_cuda_b12x_coordinator_aot_init(), "AOT initialization")

    weight = torch.randn(
        (projection.size_n, projection.size_k), device="cuda", dtype=torch.bfloat16
    ) * 0.02
    input_rows = torch.randn(
        (args.rows, projection.size_k), device="cuda", dtype=torch.bfloat16
    )
    output = torch.empty(
        (args.rows, projection.size_n), device="cuda", dtype=torch.bfloat16
    )
    payload = torch.empty(
        projection.size_n * (projection.size_k // 2 + projection.size_k // 16),
        device="cuda",
        dtype=torch.uint8,
    )
    packed_weight = torch.empty(
        projection.size_n * projection.size_k // 2, device="cuda", dtype=torch.uint8
    )
    packed_scale = torch.empty(
        projection.size_n * projection.size_k // 16, device="cuda", dtype=torch.uint8
    )
    global_scale = torch.empty(1, device="cuda", dtype=torch.float32)
    route_slots = max(8, projection.max_rows)
    route_blocks = max(1, (route_slots + 7) // 8)
    packed_routes = torch.empty(route_slots, device="cuda", dtype=torch.int32)
    block_experts = torch.empty(route_blocks, device="cuda", dtype=torch.int32)
    route_count = torch.empty(1, device="cuda", dtype=torch.int32)
    topk_weights = torch.empty(route_slots, device="cuda", dtype=torch.float32)
    scratch = torch.empty(2_097_152, device="cuda", dtype=torch.float32)
    locks = torch.empty(1_024, device="cuda", dtype=torch.int32)
    buffers = CoordinatorBuffers(
        device_buffer(input_rows),
        device_buffer(packed_weight),
        device_buffer(output),
        device_buffer(packed_scale),
        device_buffer(global_scale),
        device_buffer(packed_routes),
        device_buffer(block_experts),
        device_buffer(route_count),
        device_buffer(topk_weights),
        device_buffer(scratch),
        device_buffer(locks),
    )
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    def launch_once(cuda_stream: ctypes.c_void_p = stream) -> int:
        if projection.max_rows > 1:
            return launch(ctypes.byref(buffers), args.rows, cuda_stream)
        return launch(ctypes.byref(buffers), cuda_stream)

    check_status(
        lib,
        lib.glmrt_cuda_b12x_coordinator_w4a16_quantize_pack_weight_async(
            device_buffer(weight),
            device_buffer(payload),
            device_buffer(packed_weight),
            device_buffer(packed_scale),
            device_buffer(global_scale),
            projection.size_k,
            projection.size_n,
            stream,
        ),
        "weight quantize/pack",
    )
    check_status(
        lib,
        lib.glmrt_cuda_b12x_coordinator_w4a16_initialize_launch_buffers_async(
            ctypes.byref(buffers), stream
        ),
        "launch-buffer initialization",
    )
    if route_slots > 8:
        packed_routes.copy_(torch.arange(route_slots, device="cuda", dtype=torch.int32))
        block_experts.zero_()
        route_count.fill_(route_slots)
        topk_weights.fill_(1.0)
    torch.cuda.synchronize()

    check_status(lib, launch_once(), "W4A16 validation launch")
    torch.cuda.synchronize()
    actual = output.float()
    reference = input_rows.float() @ weight.float().T
    relative_l2 = ((actual - reference).norm() / reference.norm()).item()
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(), reference.flatten(), dim=0
    ).item()
    finite = bool(torch.isfinite(actual).all().item())
    if not finite or cosine < 0.97 or relative_l2 > 0.30:
        scale_nonzero = int(torch.count_nonzero(packed_scale).item())
        raise RuntimeError(
            f"numerical validation failed: finite={finite} cosine={cosine:.6f} "
            f"relative_l2={relative_l2:.6f} output_max={actual.abs().max().item():.6g} "
            f"global_scale={global_scale.item():.6g} "
            f"packed_scale_nonzero={scale_nonzero}/{packed_scale.numel()} "
            f"packed_scale_range={packed_scale.min().item()}..{packed_scale.max().item()} "
            f"routes={packed_routes.cpu().tolist()} route_count={route_count.item()}"
        )

    row0_equal = True
    row0_max_abs_diff = 0.0
    if projection.max_rows > 1 and args.rows > 1:
        batched_row0 = output[0].clone()
        check_status(
            lib,
            launch(ctypes.byref(buffers), 1, stream),
            "W4A16 single-row invariance launch",
        )
        torch.cuda.synchronize()
        row0_equal = bool(torch.equal(batched_row0, output[0]))
        row0_max_abs_diff = float(
            (batched_row0.float() - output[0].float()).abs().max().item()
        )
        if not row0_equal:
            raise RuntimeError(
                "row-0 output changed with active_m: "
                f"max_abs_diff={row0_max_abs_diff:.6g}"
            )

    if args.graph:
        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        with torch.cuda.graph(graph, stream=capture_stream):
            check_status(
                lib,
                launch_once(ctypes.c_void_p(capture_stream.cuda_stream)),
                "W4A16 graph capture launch",
            )
        operation = graph.replay
        execution_mode = "graph"
    else:
        def operation() -> None:
            check_status(lib, launch_once(), "W4A16 measured launch")

        execution_mode = "eager"

    for _ in range(args.warmup):
        operation()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iterations):
        operation()
    end.record()
    end.synchronize()
    milliseconds = start.elapsed_time(end) / args.iterations
    print(
        f"projection={args.projection} mode={execution_mode} rows={args.rows} "
        f"n={projection.size_n} k={projection.size_k} "
        f"finite={str(finite).lower()} cosine={cosine:.6f} "
        f"relative_l2={relative_l2:.6f} row0_equal={str(row0_equal).lower()} "
        f"row0_max_abs_diff={row0_max_abs_diff:.6g} kernel_ms={milliseconds:.6f}"
    )


if __name__ == "__main__":
    main()
