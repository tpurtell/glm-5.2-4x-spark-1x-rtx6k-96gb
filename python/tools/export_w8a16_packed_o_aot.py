#!/usr/bin/env python3
"""Export single-resident packed W8A16 O-projection kernels for SM120."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ["B12X_CUTE_COMPILE_DISK_CACHE"] = "0"
os.environ["B12X_CUTE_COMPILE_MEMORY_CACHE"] = "0"

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import Int32

from tune_w8a16_cute_packed_prefill import (
    W8A16PackedGemmKernel,
    as_cute,
)


SIZE_N = 6_144
SIZE_K = 16_384


@dataclass(frozen=True)
class KernelSpec:
    max_rows: int
    block_m: int
    tile_n: int
    tile_k: int
    stages: int
    grid: int

    @property
    def name(self) -> str:
        return f"w8a16_packed_o_m{self.max_rows}"


SPECS = (
    KernelSpec(16, 16, 128, 64, 3, 188),
    KernelSpec(32, 32, 128, 64, 3, 188),
    KernelSpec(64, 32, 128, 64, 3, 188),
    KernelSpec(128, 48, 128, 64, 3, 376),
    KernelSpec(256, 48, 256, 64, 3, 188),
)


def compile_spec(spec: KernelSpec):
    route_slots = ((spec.max_rows + spec.block_m - 1) // spec.block_m) * spec.block_m
    route_blocks = route_slots // spec.block_m
    scratch_elements = max(
        SIZE_N * route_slots,
        4 * 256 * spec.block_m * 256,
    )
    device = torch.device("cuda", torch.cuda.current_device())
    stream = torch.cuda.current_stream(device)
    tensors = {
        "input": torch.empty((spec.max_rows, SIZE_K), device=device, dtype=torch.bfloat16),
        "weight": torch.empty(SIZE_N * SIZE_K // 4, device=device, dtype=torch.int32),
        "output": torch.empty((spec.max_rows, SIZE_N), device=device, dtype=torch.bfloat16),
        "scales": torch.empty(SIZE_N * (SIZE_K // 256), device=device, dtype=torch.float32),
        "global_scale": torch.ones(1, device=device, dtype=torch.float32),
        "routes": torch.arange(route_slots, device=device, dtype=torch.int32),
        "block_experts": torch.zeros(route_blocks, device=device, dtype=torch.int32),
        "route_count": torch.tensor([route_slots], device=device, dtype=torch.int32),
        "topk": torch.ones(route_slots, device=device, dtype=torch.float32),
        "scratch": torch.empty(scratch_elements, device=device, dtype=torch.float32),
        "locks": torch.zeros(1_024, device=device, dtype=torch.int32),
    }
    kernel = W8A16PackedGemmKernel(
        size_m=spec.max_rows,
        size_n=SIZE_N,
        size_k=SIZE_K,
        block_m=spec.block_m,
        tile_n=spec.tile_n,
        tile_k=spec.tile_k,
        stages=spec.stages,
        bf16_scale_mul=False,
        post_scale_groups=True,
    )
    args = (
        as_cute(tensors["input"].flatten()),
        as_cute(tensors["weight"].flatten()),
        as_cute(tensors["output"].flatten()),
        as_cute(tensors["scales"].flatten()),
        as_cute(tensors["global_scale"]),
        as_cute(tensors["routes"]),
        as_cute(tensors["block_experts"]),
        as_cute(tensors["route_count"]),
        as_cute(tensors["topk"]),
        as_cute(tensors["scratch"]),
        as_cute(tensors["locks"]),
        Int32(spec.max_rows),
        Int32(spec.grid),
        cuda.CUstream(stream.cuda_stream),
    )
    return cute.compile(kernel, *args), kernel.shared_words * 4, route_slots, route_blocks


def export_kernels(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.init()
    config = ["#pragma once", ""]
    metadata = []
    for spec in SPECS:
        compiled, shared_bytes, route_slots, route_blocks = compile_spec(spec)
        compiled.export_to_c(
            str(output_dir),
            spec.name,
            f"glmrt_{spec.name}",
        )
        macro = f"GLMRT_W8A16_PACKED_O_M{spec.max_rows}"
        config.extend(
            (
                f"#define {macro}_MAX_ROWS {spec.max_rows}",
                f"#define {macro}_BLOCK_M {spec.block_m}",
                f"#define {macro}_GRID_X {spec.grid}",
                f"#define {macro}_ROUTE_SLOTS {route_slots}",
                f"#define {macro}_ROUTE_BLOCKS {route_blocks}",
                "",
            )
        )
        metadata.append(
            f"m<={spec.max_rows},block_m:{spec.block_m},tile_n:{spec.tile_n},"
            f"tile_k:{spec.tile_k},stages:{spec.stages},grid:{spec.grid},"
            f"shared:{shared_bytes}"
        )
        print(f"exported {spec.name}: {metadata[-1]}")
    (output_dir / "w8a16_packed_o_aot_config.h").write_text(
        "\n".join(config), encoding="ascii"
    )
    (output_dir / "w8a16_packed_o_aot.meta").write_text(
        "\n".join(metadata) + "\n", encoding="ascii"
    )
    (output_dir / "w8a16_packed_o_aot.stamp").touch()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export_kernels(args.output_dir.resolve())


if __name__ == "__main__":
    main()
