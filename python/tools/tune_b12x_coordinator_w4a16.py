#!/usr/bin/env python3
from __future__ import annotations

import _pinned_sparkinfer  # noqa: F401

import argparse
import ctypes
import json
import statistics
from dataclasses import asdict, dataclass
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
    rows: int
    native_symbol: str
    native_kind: str


@dataclass(frozen=True)
class Tile:
    tile_k: int
    tile_n: int


@dataclass(frozen=True)
class TimingResult:
    projection: str
    implementation: str
    execution_mode: str
    rows: int
    size_n: int
    size_k: int
    tile_k: int | None
    tile_n: int | None
    blocks_per_sm: int | None
    grid_x: int | None
    warmup: int
    iterations: int
    repeats: int
    median_ms: float
    min_ms: float
    max_ms: float
    exact_native: bool
    max_abs_diff_native: float
    relative_l2_native: float
    cosine_native: float


PROJECTIONS = {
    "q_b": Projection(
        label="Q-B M8",
        size_n=16_384,
        size_k=2_048,
        rows=8,
        native_symbol="glmrt_cuda_b12x_coordinator_w4a16_q_b_m8_async",
        native_kind="w4a16",
    ),
    "o_proj": Projection(
        label="O-projection M1",
        size_n=6_144,
        size_k=16_384,
        rows=1,
        native_symbol="glmrt_cuda_b12x_coordinator_w4a16_o_proj_m1_async",
        native_kind="w4a16",
    ),
    "o_proj_m16": Projection(
        label="O-projection M16",
        size_n=6_144,
        size_k=16_384,
        rows=16,
        native_symbol="glmrt_cuda_b12x_coordinator_w4a16_o_proj_m16_candidate_async",
        native_kind="w4a16",
    ),
    "q_a": Projection(
        label="Q-A M1",
        size_n=2_048,
        size_k=6_144,
        rows=1,
        native_symbol="glmrt_cuda_linear_bf16_cublas_async",
        native_kind="bf16-cublas",
    ),
}

DEFAULT_TILES = (
    Tile(tile_k=128, tile_n=128),
    Tile(tile_k=64, tile_n=128),
    Tile(tile_k=128, tile_n=64),
)


def device_buffer(tensor: torch.Tensor) -> DeviceBuffer:
    return DeviceBuffer(
        tensor.data_ptr(),
        tensor.numel() * tensor.element_size(),
        tensor.device.index or 0,
        0,
    )


def current_stream_pointer() -> ctypes.c_void_p:
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def check_status(lib: ctypes.CDLL, status: int, action: str) -> None:
    if status == 0:
        return
    error = ctypes.create_string_buffer(512)
    lib.glmrt_last_error_message(error, len(error))
    raise RuntimeError(f"{action} failed with status {status}: {error.value.decode()}")


def parse_tiles(raw: str) -> tuple[Tile, ...]:
    tiles = []
    for item in raw.split(","):
        try:
            tile_k, tile_n = (int(value) for value in item.split("x", maxsplit=1))
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"invalid tile {item!r}; expected KxN"
            ) from error
        tiles.append(Tile(tile_k=tile_k, tile_n=tile_n))
    if not tiles:
        raise argparse.ArgumentTypeError("at least one tile is required")
    return tuple(tiles)


def parse_grid_multipliers(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in raw.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("grid multipliers must be numbers") from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("grid multipliers must be positive")
    return values


def configure_native(
    lib: ctypes.CDLL, projection: Projection
) -> ctypes._CFuncPtr:
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
    lib.glmrt_cuda_b12x_coordinator_w4a16_quantize_pack_weight_async.restype = (
        ctypes.c_int
    )
    lib.glmrt_cuda_b12x_coordinator_w4a16_initialize_launch_buffers_async.argtypes = (
        ctypes.POINTER(CoordinatorBuffers),
        ctypes.c_void_p,
    )
    lib.glmrt_cuda_b12x_coordinator_w4a16_initialize_launch_buffers_async.restype = (
        ctypes.c_int
    )
    launch = getattr(lib, projection.native_symbol)
    if projection.native_kind == "bf16-cublas":
        launch.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        )
    elif projection.rows > 1:
        launch.argtypes = (
            ctypes.POINTER(CoordinatorBuffers),
            ctypes.c_size_t,
            ctypes.c_void_p,
        )
    else:
        launch.argtypes = (ctypes.POINTER(CoordinatorBuffers), ctypes.c_void_p)
    launch.restype = ctypes.c_int
    return launch


def configure_cudart() -> ctypes.CDLL:
    cudart = ctypes.CDLL("libcudart.so")
    cudart.cudaMemsetAsync.argtypes = (
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    cudart.cudaMemsetAsync.restype = ctypes.c_int
    return cudart


def measure(operation, warmup: int, iterations: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return samples


def capture_graph(operation) -> torch.cuda.CUDAGraph:
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()
    torch.cuda.synchronize()
    return graph


def comparison(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[bool, float, float, float]:
    reference_f32 = reference.float()
    candidate_f32 = candidate.float()
    difference = candidate_f32 - reference_f32
    reference_norm = torch.linalg.vector_norm(reference_f32)
    difference_norm = torch.linalg.vector_norm(difference)
    norm_product = reference_norm * torch.linalg.vector_norm(candidate_f32)
    relative_l2 = (
        (difference_norm / reference_norm).item()
        if reference_norm.item() != 0.0
        else difference_norm.item()
    )
    cosine = (
        ((reference_f32.flatten() @ candidate_f32.flatten()) / norm_product).item()
        if norm_product.item() != 0.0
        else 1.0
    )
    return (
        torch.equal(reference, candidate),
        difference.abs().max().item(),
        relative_l2,
        cosine,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep off-path SparkInfer W4A16 coordinator tiles and persistent grids "
            "against the production native AOT result."
        )
    )
    parser.add_argument("--native-lib", type=Path, required=True)
    parser.add_argument("--projection", choices=tuple(PROJECTIONS), required=True)
    parser.add_argument(
        "--tiles",
        type=parse_tiles,
        default=DEFAULT_TILES,
        help="comma-separated KxN tile pairs",
    )
    parser.add_argument(
        "--grid-multipliers",
        type=parse_grid_multipliers,
        default=(0.5, 1.0, 1.5, 2.0),
        help="comma-separated SM multipliers, clipped to the compiled occupancy cap",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--eager",
        action="store_true",
        help="measure direct Python/CuTe dispatch instead of production-like graph replay",
    )
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.repeats) < 1:
        parser.error("warmup, iterations, and repeats must be positive")

    from b12x.moe._shared.kernels.w4a16.kernel import (
        _cutlass_element_dtype,
        compile_w4a16_gemm,
        cuda,
        cute,
        make_ptr,
    )

    torch.manual_seed(args.seed)
    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    sms = int(properties.multi_processor_count)
    projection = PROJECTIONS[args.projection]
    native_lib = ctypes.CDLL(str(args.native_lib.resolve()))
    native_launch = configure_native(native_lib, projection)
    cudart = configure_cudart()
    check_status(
        native_lib,
        native_lib.glmrt_cuda_b12x_coordinator_aot_init(),
        "native AOT initialization",
    )

    weight = torch.randn(
        (projection.size_n, projection.size_k),
        dtype=torch.bfloat16,
        device=device,
    ) * 0.02
    input_rows = torch.randn(
        (projection.rows, projection.size_k),
        dtype=torch.bfloat16,
        device=device,
    )
    native_output = torch.empty(
        (projection.rows, projection.size_n),
        dtype=torch.bfloat16,
        device=device,
    )
    candidate_output = torch.empty_like(native_output)
    payload = torch.empty(
        projection.size_n
        * (projection.size_k // 2 + projection.size_k // 16),
        dtype=torch.uint8,
        device=device,
    )
    packed_weight = torch.empty(
        projection.size_n * projection.size_k // 2,
        dtype=torch.uint8,
        device=device,
    )
    packed_scale = torch.empty(
        projection.size_n * projection.size_k // 16,
        dtype=torch.uint8,
        device=device,
    )
    global_scale = torch.empty(1, dtype=torch.float32, device=device)
    route_slots = max(8, projection.rows)
    route_blocks = max(1, (route_slots + 7) // 8)
    packed_routes = torch.empty(route_slots, dtype=torch.int32, device=device)
    block_experts = torch.empty(route_blocks, dtype=torch.int32, device=device)
    route_count = torch.empty(1, dtype=torch.int32, device=device)
    topk_weights = torch.empty(route_slots, dtype=torch.float32, device=device)
    scratch = torch.empty(2_097_152, dtype=torch.float32, device=device)
    locks = torch.empty(1_024, dtype=torch.int32, device=device)
    buffers = CoordinatorBuffers(
        device_buffer(input_rows),
        device_buffer(packed_weight),
        device_buffer(native_output),
        device_buffer(packed_scale),
        device_buffer(global_scale),
        device_buffer(packed_routes),
        device_buffer(block_experts),
        device_buffer(route_count),
        device_buffer(topk_weights),
        device_buffer(scratch),
        device_buffer(locks),
    )
    stream_pointer = current_stream_pointer()
    check_status(
        native_lib,
        native_lib.glmrt_cuda_b12x_coordinator_w4a16_quantize_pack_weight_async(
            device_buffer(weight),
            device_buffer(payload),
            device_buffer(packed_weight),
            device_buffer(packed_scale),
            device_buffer(global_scale),
            projection.size_k,
            projection.size_n,
            stream_pointer,
        ),
        "production weight quantize/pack",
    )
    check_status(
        native_lib,
        native_lib.glmrt_cuda_b12x_coordinator_w4a16_initialize_launch_buffers_async(
            ctypes.byref(buffers), stream_pointer
        ),
        "production launch-buffer initialization",
    )
    if route_slots > 8:
        packed_routes.copy_(torch.arange(route_slots, dtype=torch.int32, device=device))
        block_experts.zero_()
        route_count.fill_(route_slots)
        topk_weights.fill_(1.0)

    def launch_native() -> None:
        launch_stream = current_stream_pointer()
        if projection.native_kind == "bf16-cublas":
            status = native_launch(
                ctypes.c_void_p(input_rows.data_ptr()),
                ctypes.c_void_p(weight.data_ptr()),
                None,
                ctypes.c_void_p(native_output.data_ptr()),
                projection.rows,
                projection.size_k,
                projection.size_n,
                launch_stream,
            )
        elif projection.rows > 1:
            status = native_launch(
                ctypes.byref(buffers), projection.rows, launch_stream
            )
        else:
            status = native_launch(ctypes.byref(buffers), launch_stream)
        check_status(native_lib, status, "production native launch")

    launch_native()
    torch.cuda.synchronize()
    native_reference = native_output.clone()
    bf16_reference = input_rows.float() @ weight.float().T
    native_quality = comparison(
        bf16_reference.to(torch.bfloat16), native_reference
    )
    native_operation = launch_native
    if not args.eager:
        native_graph = capture_graph(launch_native)
        native_operation = native_graph.replay
    native_samples = measure(
        native_operation, args.warmup, args.iterations, args.repeats
    )
    results = [
        TimingResult(
            projection=args.projection,
            implementation="native-current",
            execution_mode="eager" if args.eager else "graph",
            rows=projection.rows,
            size_n=projection.size_n,
            size_k=projection.size_k,
            tile_k=None,
            tile_n=None,
            blocks_per_sm=None,
            grid_x=None,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
            median_ms=statistics.median(native_samples),
            min_ms=min(native_samples),
            max_ms=max(native_samples),
            exact_native=True,
            max_abs_diff_native=0.0,
            relative_l2_native=0.0,
            cosine_native=1.0,
        )
    ]
    print(
        json.dumps(
            {
                **asdict(results[0]),
                "bf16_reference_max_abs_diff": native_quality[1],
                "bf16_reference_relative_l2": native_quality[2],
                "bf16_reference_cosine": native_quality[3],
                "gpu": properties.name,
                "sms": sms,
            },
            sort_keys=True,
        )
    )

    input_pointer = make_ptr(
        _cutlass_element_dtype("bf16"),
        input_rows.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    candidate_output_pointer = make_ptr(
        _cutlass_element_dtype("bf16"),
        candidate_output.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    for tile in args.tiles:
        compiled = compile_w4a16_gemm(
            size_m=projection.rows,
            size_n=projection.size_n,
            size_k=projection.size_k,
            num_experts=1,
            top_k=1,
            mul_topk_weights=False,
            tile_n=tile.tile_n,
            tile_k=tile.tile_k,
            moe_block_size=8,
            max_m_blocks=route_blocks,
            element_dtype="bf16",
            scale_format="e4m3_k16",
        )
        max_grid = sms * int(compiled.blocks_per_sm)
        grids = sorted(
            {
                max(1, min(max_grid, int(round(sms * multiplier))))
                for multiplier in args.grid_multipliers
            }
        )
        for grid_x in grids:

            def launch_candidate() -> None:
                launch_stream_pointer = current_stream_pointer()
                cuda_status = cudart.cudaMemsetAsync(
                    ctypes.c_void_p(locks.data_ptr()),
                    0,
                    locks.numel() * locks.element_size(),
                    launch_stream_pointer,
                )
                if cuda_status != 0:
                    raise RuntimeError(
                        f"cudaMemsetAsync for candidate locks failed: {cuda_status}"
                    )
                compiled.compiled(
                    input_pointer,
                    input_pointer,
                    packed_weight.view(torch.int32),
                    candidate_output_pointer,
                    packed_scale.view(torch.int32),
                    global_scale,
                    packed_routes,
                    block_experts,
                    route_count,
                    topk_weights,
                    scratch,
                    locks,
                    projection.rows,
                    grid_x,
                    cuda.CUstream(torch.cuda.current_stream().cuda_stream),
                )

            launch_candidate()
            torch.cuda.synchronize()
            exact, max_abs, relative_l2, cosine = comparison(
                native_reference, candidate_output
            )
            candidate_operation = launch_candidate
            if not args.eager:
                candidate_graph = capture_graph(launch_candidate)
                candidate_operation = candidate_graph.replay
            samples = measure(
                candidate_operation, args.warmup, args.iterations, args.repeats
            )
            result = TimingResult(
                projection=args.projection,
                implementation="candidate",
                execution_mode="eager" if args.eager else "graph",
                rows=projection.rows,
                size_n=projection.size_n,
                size_k=projection.size_k,
                tile_k=tile.tile_k,
                tile_n=tile.tile_n,
                blocks_per_sm=int(compiled.blocks_per_sm),
                grid_x=grid_x,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
                median_ms=statistics.median(samples),
                min_ms=min(samples),
                max_ms=max(samples),
                exact_native=exact,
                max_abs_diff_native=max_abs,
                relative_l2_native=relative_l2,
                cosine_native=cosine,
            )
            results.append(result)
            print(json.dumps(asdict(result), sort_keys=True))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps([asdict(result) for result in results], indent=2) + "\n",
            encoding="ascii",
        )


if __name__ == "__main__":
    main()
