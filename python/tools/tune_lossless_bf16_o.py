"""Measure direct lossless packed-BF16 projection GEMV."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
from statistics import median

import numpy as np
import torch
import triton
import triton.language as tl

TILE_WIDTH = 1024


def tensor_memmap(catalog: dict, tensor: dict) -> np.memmap:
    return np.memmap(
        Path(catalog["snapshot_path"]) / tensor["file"],
        dtype=np.uint16,
        mode="r",
        offset=tensor["byte_offset"],
        shape=tuple(tensor["shape"]),
    )


def optimal_tile_windows(
    exponent_tiles: np.ndarray, *, batch_tiles: int = 8192
) -> tuple[np.ndarray, np.ndarray]:
    tile_bases = np.empty(exponent_tiles.shape[0], dtype=np.uint8)
    escape_counts = np.empty(exponent_tiles.shape[0], dtype=np.uint16)
    for start in range(0, exponent_tiles.shape[0], batch_tiles):
        stop = min(start + batch_tiles, exponent_tiles.shape[0])
        batch = exponent_tiles[start:stop]
        tile_offsets = (
            np.arange(stop - start, dtype=np.uint32)[:, None] * 256
        )
        encoded = batch.astype(np.uint32) + tile_offsets
        histogram = np.bincount(
            encoded.reshape(-1), minlength=(stop - start) * 256
        ).reshape(stop - start, 256)
        prefix = np.pad(
            np.cumsum(histogram, axis=1), ((0, 0), (1, 0)), constant_values=0
        )
        coverage = prefix[:, 15:] - prefix[:, :-15]
        best_bases = np.argmax(coverage, axis=1)
        best_coverage = coverage[np.arange(stop - start), best_bases]
        tile_bases[start:stop] = best_bases.astype(np.uint8)
        escape_counts[start:stop] = TILE_WIDTH - best_coverage
    return tile_bases, escape_counts


def scan_all_projection_layers(catalog: dict) -> None:
    for projection in ("q_b_proj", "o_proj"):
        suffix = f"self_attn.{projection}.weight"
        tensors = sorted(
            (
                item
                for item in catalog["tensors"]
                if item["name"].endswith(suffix) and item["layer_id"] is not None
            ),
            key=lambda item: item["layer_id"],
        )
        global_max = -1
        global_layer = -1
        global_tile = -1
        for tensor in tensors:
            raw = tensor_memmap(catalog, tensor)
            exponent_tiles = (
                ((np.asarray(raw) >> 7) & 0xFF)
                .astype(np.uint8)
                .reshape(-1, TILE_WIDTH)
            )
            _, escape_counts = optimal_tile_windows(exponent_tiles)
            layer_tile = int(np.argmax(escape_counts))
            layer_max = int(escape_counts[layer_tile])
            if layer_max > global_max:
                global_max = layer_max
                global_layer = int(tensor["layer_id"])
                global_tile = layer_tile
            print(
                f"scan projection={projection} layer={tensor['layer_id']} "
                f"max_escapes={layer_max} "
                f"nonzero_fraction={float(np.mean(escape_counts != 0)):.8f}"
            )
        print(
            f"scan_summary projection={projection} layers={len(tensors)} "
            f"max_escapes={global_max} layer={global_layer} tile={global_tile}"
        )


@triton.jit
def packed_bf16_gemv_sparse_escapes(
    x,
    low,
    codes,
    metadata,
    y,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ESCAPE_CAPACITY: tl.constexpr,
):
    row = tl.program_id(0)
    acc = tl.zeros((), tl.float32)
    for base in range(0, K, BLOCK_K):
        k = base + tl.arange(0, BLOCK_K)
        idx = row * K + k
        lo = tl.load(low + idx).to(tl.uint32)
        code_idx = row * (K // 2) + (k // 2)
        packed_code = tl.load(codes + code_idx).to(tl.uint32)
        code = (packed_code >> ((k & 1) * 4)) & 15
        tile = row * (K // BLOCK_K) + base // BLOCK_K
        header = tl.load(metadata + tile * (ESCAPE_CAPACITY + 1)).to(tl.uint32)
        exponent = code + (header & 0xFF)
        escape_count = (header >> 8) & 0xFF
        if escape_count != 0:
            for slot in range(ESCAPE_CAPACITY):
                entry = tl.load(
                    metadata + tile * (ESCAPE_CAPACITY + 1) + slot + 1,
                    mask=slot < escape_count,
                    other=0,
                ).to(tl.uint32)
                escape_position = entry & 0xFFFF
                escape_exponent = entry >> 16
                exponent = tl.where(
                    (slot < escape_count)
                    & (code == 15)
                    & ((k - base) == escape_position),
                    escape_exponent,
                    exponent,
                )
        bits = (lo & 0x7F) | ((lo & 0x80) << 8) | (exponent << 7)
        weight = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
        activation = tl.load(x + k).to(tl.float32)
        acc += tl.sum(weight.to(tl.float32) * activation)
    tl.store(y + row, acc.to(tl.bfloat16))


@triton.jit
def unpack_bf16_sparse_escapes(
    low,
    codes,
    metadata,
    output,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ESCAPE_CAPACITY: tl.constexpr,
):
    tile = tl.program_id(0)
    tiles_per_row = K // BLOCK_K
    row = tile // tiles_per_row
    base = (tile % tiles_per_row) * BLOCK_K
    k = base + tl.arange(0, BLOCK_K)
    idx = row * K + k
    lo = tl.load(low + idx).to(tl.uint32)
    packed_code = tl.load(codes + row * (K // 2) + k // 2).to(tl.uint32)
    code = (packed_code >> ((k & 1) * 4)) & 15
    header = tl.load(metadata + tile * (ESCAPE_CAPACITY + 1)).to(tl.uint32)
    exponent = code + (header & 0xFF)
    escape_count = (header >> 8) & 0xFF
    for slot in range(ESCAPE_CAPACITY):
        entry = tl.load(
            metadata + tile * (ESCAPE_CAPACITY + 1) + slot + 1,
            mask=slot < escape_count,
            other=0,
        ).to(tl.uint32)
        exponent = tl.where(
            (slot < escape_count)
            & (code == 15)
            & ((k - base) == (entry & 0xFFFF)),
            entry >> 16,
            exponent,
        )
    bits = (lo & 0x7F) | ((lo & 0x80) << 8) | (exponent << 7)
    tl.store(output + idx, bits.to(tl.uint16))


def bench(
    launch, *, warmup: int = 24, iterations: int = 240, repeats: int = 5
) -> tuple[float, float, float]:
    samples = []
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
    return median(samples), min(samples), max(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tensor",
        default="model.layers.0.self_attn.o_proj.weight",
        help="BF16 layer-0 projection to benchmark",
    )
    parser.add_argument(
        "--native-library",
        type=Path,
        help="also benchmark glmrt's native SM120 lossless GEMV",
    )
    parser.add_argument(
        "--scan-all-layers",
        action="store_true",
        help="scan every Q-B/O tensor for the required per-tile escape capacity",
    )
    args = parser.parse_args()
    with Path(".glmrt-cache/model-artifacts/diagnostic/model_catalog.json").open() as handle:
        catalog = json.load(handle)
    if args.scan_all_layers:
        scan_all_projection_layers(catalog)
        return
    tensor = next(
        item for item in catalog["tensors"] if item["name"] == args.tensor
    )
    raw = tensor_memmap(catalog, tensor)
    weight = (
        torch.from_numpy(np.array(raw, copy=True).view(np.int16))
        .view(torch.bfloat16)
        .cuda()
    )
    bits = weight.view(torch.int16).to(torch.int32) & 0xFFFF
    low = ((bits & 0x7F) | ((bits >> 8) & 0x80)).to(torch.uint8)
    exponent = ((bits >> 7) & 0xFF).to(torch.uint8)
    escape_capacity = 5
    exponent_cpu = (
        ((np.asarray(raw) >> 7) & 0xFF)
        .astype(np.uint8)
        .reshape(-1, TILE_WIDTH)
    )
    tile_bases, escape_counts = optimal_tile_windows(exponent_cpu)
    tile_metadata = np.zeros(
        (exponent_cpu.shape[0], escape_capacity + 1), dtype=np.uint32
    )
    for tile_index, tile_exponents in enumerate(exponent_cpu):
        base = int(tile_bases[tile_index])
        escape_positions = np.flatnonzero(
            (tile_exponents < base) | (tile_exponents >= base + 15)
        )
        if escape_positions.size != escape_counts[tile_index]:
            raise RuntimeError("vectorized escape count does not match tile encoding")
        if escape_positions.size > escape_capacity:
            raise RuntimeError(
                f"tile {tile_index} requires {escape_positions.size} escapes, "
                f"capacity is {escape_capacity}"
            )
        tile_bases[tile_index] = base
        tile_metadata[tile_index, 0] = base | (escape_positions.size << 8)
        for slot, position in enumerate(escape_positions):
            tile_metadata[tile_index, slot + 1] = int(position) | (
                int(tile_exponents[position]) << 16
            )
    tile_base_device = torch.from_numpy(tile_bases).cuda().reshape(
        weight.shape[0], weight.shape[1] // TILE_WIDTH, 1
    )
    tile_code = torch.where(
        (exponent.reshape(weight.shape[0], -1, TILE_WIDTH) >= tile_base_device)
        & (
            exponent.reshape(weight.shape[0], -1, TILE_WIDTH)
            < tile_base_device + 15
        ),
        exponent.reshape(weight.shape[0], -1, TILE_WIDTH) - tile_base_device,
        torch.full_like(exponent.reshape(weight.shape[0], -1, TILE_WIDTH), 15),
    ).reshape_as(exponent)
    tile_codes = (
        tile_code[:, 0::2] | (tile_code[:, 1::2] << 4)
    ).contiguous()
    metadata = torch.from_numpy(tile_metadata.reshape(-1)).cuda()
    print(
        "shape={} raw_MB={:.3f} packed_main_MB={:.3f} "
        "sparse_escape_metadata_MB={:.3f} sparse_total_MB={:.3f} "
        "max_escapes_per_tile={}".format(
            tuple(weight.shape),
            weight.nbytes / 1e6,
            (low.nbytes + tile_codes.nbytes) / 1e6,
            metadata.nbytes / 1e6,
            (low.nbytes + tile_codes.nbytes + metadata.nbytes) / 1e6,
            int((tile_metadata[:, 0] >> 8).max()),
        )
    )
    unpacked_bits = torch.empty_like(weight, dtype=torch.int16)
    unpack_bf16_sparse_escapes[(tile_metadata.shape[0],)](
        low,
        tile_codes,
        metadata,
        unpacked_bits,
        K=weight.shape[1],
        BLOCK_K=TILE_WIDTH,
        ESCAPE_CAPACITY=escape_capacity,
        num_warps=4,
    )
    torch.cuda.synchronize()
    mismatch_mask = unpacked_bits != weight.view(torch.int16)
    mismatch_count = int(mismatch_mask.sum())
    print(f"unpack_bit_mismatches={mismatch_count}")
    if mismatch_count:
        mismatch_indices = torch.nonzero(mismatch_mask)[:8].cpu().tolist()
        print(
            "unpack_mismatch_examples={}".format(
                [
                    (
                        index,
                        int(weight.view(torch.int16)[tuple(index)]) & 0xFFFF,
                        int(unpacked_bits[tuple(index)]) & 0xFFFF,
                    )
                    for index in mismatch_indices
                ]
            )
        )

    output_rows, hidden = weight.shape
    torch.manual_seed(0)
    activation = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(output_rows, device="cuda", dtype=torch.bfloat16)
    reference = torch.mv(weight, activation)

    packed_bf16_gemv_sparse_escapes[(output_rows,)](
        activation,
        low,
        tile_codes,
        metadata,
        output,
        K=hidden,
        BLOCK_K=TILE_WIDTH,
        ESCAPE_CAPACITY=escape_capacity,
        num_warps=4,
    )
    torch.cuda.synchronize()
    sparse_candidate = output.clone()
    sparse_difference = sparse_candidate.float() - reference.float()
    print(
        "sparse_validation relative_l2={:.9f} bitwise={} max_abs={:.6f}".format(
            float(
                torch.linalg.vector_norm(sparse_difference)
                / torch.linalg.vector_norm(reference.float())
            ),
            torch.equal(sparse_candidate, reference),
            float(sparse_difference.abs().max()),
        )
    )

    raw_weights = [weight.clone() for _ in range(4)]
    low_weights = [low.clone() for _ in range(4)]
    tile_code_weights = [tile_codes.clone() for _ in range(4)]
    metadata_weights = [metadata.clone() for _ in range(4)]
    bf16_ms, bf16_min_ms, bf16_max_ms = bench(
        lambda index: torch.mv(raw_weights[index & 3], activation, out=output)
    )
    print(
        f"bf16_rotating_ms={bf16_ms:.6f} "
        f"range={bf16_min_ms:.6f}-{bf16_max_ms:.6f}"
    )
    sparse_ms, sparse_min_ms, sparse_max_ms = bench(
        lambda index: packed_bf16_gemv_sparse_escapes[(output_rows,)](
            activation,
            low_weights[index & 3],
            tile_code_weights[index & 3],
            metadata_weights[index & 3],
            output,
            K=hidden,
            BLOCK_K=TILE_WIDTH,
            ESCAPE_CAPACITY=escape_capacity,
            num_warps=4,
        )
    )
    print(
        f"sparse_packed_rotating_ms={sparse_ms:.6f} "
        f"range={sparse_min_ms:.6f}-{sparse_max_ms:.6f} "
        f"speedup={bf16_ms / sparse_ms:.4f}"
    )

    if args.native_library is not None:
        native = ctypes.CDLL(str(args.native_library.resolve()))
        native_launch = native.glmrt_cuda_linear_lossless_bf16_m1_async
        native_launch.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        )
        native_launch.restype = ctypes.c_int
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

        def launch_native(index: int) -> None:
            status = native_launch(
                activation.data_ptr(),
                low_weights[index & 3].data_ptr(),
                tile_code_weights[index & 3].data_ptr(),
                metadata_weights[index & 3].data_ptr(),
                output.data_ptr(),
                hidden,
                output_rows,
                escape_capacity + 1,
                stream,
            )
            if status != 0:
                raise RuntimeError(f"native lossless GEMV failed with status {status}")

        launch_native(0)
        torch.cuda.synchronize()
        native_candidate = output.clone()
        native_difference = native_candidate.float() - reference.float()
        print(
            "native_validation relative_l2={:.9f} bitwise={} max_abs={:.6f}".format(
                float(
                    torch.linalg.vector_norm(native_difference)
                    / torch.linalg.vector_norm(reference.float())
                ),
                torch.equal(native_candidate, reference),
                float(native_difference.abs().max()),
            )
        )
        native_ms, native_min_ms, native_max_ms = bench(launch_native)
        print(
            f"native_packed_rotating_ms={native_ms:.6f} "
            f"range={native_min_ms:.6f}-{native_max_ms:.6f} "
            f"speedup={bf16_ms / native_ms:.4f}"
        )


if __name__ == "__main__":
    main()
