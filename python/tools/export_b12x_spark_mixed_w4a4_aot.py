#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

REGIMES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
FC1_N = 1_024
FC1_K = 6_144
FC2_N = 6_144
FC2_K = 512


def export_kernels(output_dir: Path) -> None:
    os.environ["B12X_CUTE_COMPILE_DISK_CACHE"] = "0"
    os.environ["B12X_CUTE_COMPILE_MEMORY_CACHE"] = "0"

    import torch
    from b12x.moe.fused.w4a16.host import (
        max_packed_route_slots,
        packed_gemm_scratch_elements,
        select_route_block_size_m,
    )
    from b12x.moe.fused.w4a16.kernel import (
        _select_tile_config,
        compile_w4a16_activation,
        compile_w4a16_gemm,
    )
    from export_b12x_spark_aot import _compile_dense_kernel

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    sms = int(properties.multi_processor_count)
    max_shared_mem = int(properties.shared_memory_per_block_optin)

    config_lines = ["#pragma once"]
    metadata_lines = []
    for rows in REGIMES:
        fc1_name = f"mixed_w4a4_fc1_m{rows}"
        fc1 = _compile_dense_kernel(rows=rows, n=FC1_N, k=FC1_K)
        fc1.export_to_c(
            output_dir.as_posix(), fc1_name, f"glmrt_b12x_{fc1_name}"
        )

        activation_name = f"mixed_w4a16_activation_m{rows}"
        activation = compile_w4a16_activation(
            rows=rows,
            intermediate_size=FC2_K,
            activation="silu",
            element_dtype="bf16",
            fast_math=True,
        )
        activation.compiled.export_to_c(
            output_dir.as_posix(),
            activation_name,
            f"glmrt_b12x_{activation_name}",
        )

        block_size = select_route_block_size_m(rows, 1, 1)
        route_slots = max_packed_route_slots(rows, block_size, 1)
        route_blocks = (route_slots + block_size - 1) // block_size
        tile_k, tile_n, _, _ = _select_tile_config(
            problem_m=rows,
            problem_n=FC2_N,
            problem_k=FC2_K,
            top_k=1,
            moe_block_size=block_size,
            sms=sms,
            max_shared_mem=max_shared_mem,
            scale_format="e4m3_k16",
        )
        fc2 = compile_w4a16_gemm(
            size_m=rows,
            size_n=FC2_N,
            size_k=FC2_K,
            num_experts=1,
            top_k=1,
            mul_topk_weights=False,
            tile_n=tile_n,
            tile_k=tile_k,
            moe_block_size=block_size,
            max_m_blocks=route_blocks,
            element_dtype="bf16",
            scale_format="e4m3_k16",
        )
        fc2_name = f"mixed_w4a16_fc2_m{rows}"
        fc2.compiled.export_to_c(
            output_dir.as_posix(), fc2_name, f"glmrt_b12x_{fc2_name}"
        )
        # A second resident FC2 block lost at M8/M16/M32 on GB10. Keep the
        # exported default at one block per SM; the benchmark ABI can sweep it.
        grid_x = sms
        scratch_elements = packed_gemm_scratch_elements(
            size_n=FC2_N,
            route_slots=route_slots,
            moe_block_size=block_size,
            sms=sms,
        )

        prefix = f"GLMRT_B12X_MIXED_W4A4_M{rows}"
        config_lines.extend(
            (
                f"#define {prefix}_GRID_X {grid_x}",
                f"#define {prefix}_BLOCK_SIZE {block_size}",
                f"#define {prefix}_ROUTE_SLOTS {route_slots}",
                f"#define {prefix}_ROUTE_BLOCKS {route_blocks}",
                f"#define {prefix}_SCRATCH_ELEMENTS {scratch_elements}",
            )
        )
        metadata_lines.append(
            f"m:{rows},fc1_n:{FC1_N},fc1_k:{FC1_K},fc2_n:{FC2_N},"
            f"fc2_k:{FC2_K},fc2_tile_n:{tile_n},fc2_tile_k:{tile_k},"
            f"grid:{grid_x},block:{block_size},route_slots:{route_slots},"
            f"route_blocks:{route_blocks},scratch_elements:{scratch_elements}"
        )

    (output_dir / "b12x_spark_mixed_w4a4_aot_config.h").write_text(
        "\n".join(config_lines) + "\n", encoding="ascii"
    )
    (output_dir / "b12x_spark_mixed_w4a4_aot.meta").write_text(
        "\n".join(metadata_lines) + "\n", encoding="ascii"
    )
    (output_dir / "b12x_spark_mixed_w4a4_aot.stamp").touch()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the off-path Spark mixed W4A4 exact-bucket candidates."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export_kernels(args.output_dir.resolve())


if __name__ == "__main__":
    main()
