#!/usr/bin/env python3
"""Export bucketed SM120 row-major W8A16 Triton cubins as C++ arrays."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from tune_w8a16_triton_prefill import w8a16_group256_gemm


@dataclass(frozen=True)
class KernelSpec:
    projection: str
    max_rows: int
    n: int
    k: int
    block_m: int
    block_n: int
    block_k: int
    warps: int
    stages: int

    @property
    def symbol(self) -> str:
        return f"glmrt_w8a16_{self.projection}_m{self.max_rows}"


SPECS = (
    KernelSpec("qa", 16, 2048, 6144, 16, 64, 64, 4, 3),
    KernelSpec("qa", 32, 2048, 6144, 16, 64, 64, 4, 3),
    KernelSpec("qa", 64, 2048, 6144, 16, 64, 64, 4, 3),
    KernelSpec("qa", 128, 2048, 6144, 32, 64, 64, 4, 3),
    KernelSpec("qa", 256, 2048, 6144, 32, 64, 64, 4, 3),
    KernelSpec("qa", 512, 2048, 6144, 64, 128, 128, 8, 3),
    KernelSpec("qa", 1024, 2048, 6144, 64, 128, 128, 8, 3),
    KernelSpec("qa", 2048, 2048, 6144, 64, 128, 128, 8, 3),
    KernelSpec("qb", 16, 16384, 2048, 16, 64, 64, 4, 3),
    KernelSpec("qb", 32, 16384, 2048, 32, 64, 64, 4, 3),
    KernelSpec("qb", 64, 16384, 2048, 64, 128, 64, 8, 3),
    KernelSpec("qb", 128, 16384, 2048, 128, 128, 64, 8, 3),
    KernelSpec("qb", 256, 16384, 2048, 64, 128, 64, 8, 3),
    KernelSpec("o", 16, 6144, 16384, 16, 64, 64, 4, 3),
    KernelSpec("o", 32, 6144, 16384, 32, 64, 64, 4, 3),
    KernelSpec("o", 64, 6144, 16384, 64, 64, 128, 8, 3),
    KernelSpec("o", 128, 6144, 16384, 128, 64, 64, 4, 3),
    KernelSpec("o", 256, 6144, 16384, 64, 256, 128, 8, 3),
)


def compile_spec(spec: KernelSpec, tensors: dict[str, torch.Tensor]):
    return w8a16_group256_gemm.warmup(
        tensors["a"],
        tensors["weight"],
        tensors["scales"],
        tensors["output"],
        M=spec.max_rows,
        N=spec.n,
        K=spec.k,
        BLOCK_M=spec.block_m,
        BLOCK_N=spec.block_n,
        BLOCK_K=spec.block_k,
        GROUP_M=8,
        DEQUANT_BF16=False,
        POST_SCALE_GROUP=False,
        ROW_MAJOR_WEIGHT=True,
        num_warps=spec.warps,
        num_stages=spec.stages,
        grid=(1,),
    )


def byte_initializer(data: bytes) -> str:
    rows = []
    for offset in range(0, len(data), 16):
        rows.append(
            "  " + ", ".join(f"0x{value:02x}" for value in data[offset : offset + 16])
        )
    return ",\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    header = [
        "#pragma once",
        "#include <cstddef>",
        "namespace glmrt_w8a16_aot {",
        "struct KernelConfig {",
        "  const unsigned char* cubin;",
        "  std::size_t cubin_size;",
        "  const char* symbol;",
        "  std::size_t max_rows;",
        "  std::size_t input_dim;",
        "  std::size_t output_dim;",
        "  std::size_t block_m;",
        "  std::size_t block_n;",
        "  std::size_t threads;",
        "  std::size_t shared_bytes;",
        "};",
        "extern const KernelConfig kernels[];",
        "extern const std::size_t kernel_count;",
        "}  // namespace glmrt_w8a16_aot",
        "",
    ]
    source = [
        '#include "w8a16_row_major_aot.h"',
        "namespace glmrt_w8a16_aot {",
    ]
    entries = []
    tensor_cache: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    for spec in SPECS:
        shape = (spec.n, spec.k)
        if shape not in tensor_cache:
            tensor_cache[shape] = {
                "a": torch.empty(
                    (256, spec.k), device="cuda", dtype=torch.bfloat16
                ),
                "weight": torch.empty(
                    (spec.n, spec.k), device="cuda", dtype=torch.int8
                ),
                "scales": torch.empty(
                    (spec.n, spec.k // 256), device="cuda", dtype=torch.float32
                ),
                "output": torch.empty(
                    (256, spec.n), device="cuda", dtype=torch.bfloat16
                ),
            }
        compiled = compile_spec(spec, tensor_cache[shape])
        cubin = compiled.asm["cubin"]
        if compiled.name != "w8a16_group256_gemm":
            raise RuntimeError(f"unexpected Triton symbol {compiled.name}")
        source.extend(
            (
                f"alignas(16) static const unsigned char {spec.symbol}[] = {{",
                byte_initializer(cubin),
                "};",
            )
        )
        entries.append(
            "  {"
            f"{spec.symbol}, sizeof({spec.symbol}), \"{compiled.name}\", "
            f"{spec.max_rows}, {spec.k}, {spec.n}, {spec.block_m}, {spec.block_n}, "
            f"{compiled.metadata.num_warps * 32}, {compiled.metadata.shared}"
            "}"
        )
        print(
            f"exported {spec.projection} rows<={spec.max_rows} "
            f"tile={spec.block_m}x{spec.block_n}x{spec.block_k} "
            f"bytes={len(cubin)} shared={compiled.metadata.shared}"
        )

    source.extend(
        (
            "const KernelConfig kernels[] = {",
            ",\n".join(entries),
            "};",
            "const std::size_t kernel_count = sizeof(kernels) / sizeof(kernels[0]);",
            "}  // namespace glmrt_w8a16_aot",
            "",
        )
    )
    (args.output_dir / "w8a16_row_major_aot.h").write_text("\n".join(header))
    (args.output_dir / "w8a16_row_major_aot.cc").write_text("\n".join(source))
    (args.output_dir / "w8a16_row_major_aot.stamp").write_text("ok\n")


if __name__ == "__main__":
    main()
