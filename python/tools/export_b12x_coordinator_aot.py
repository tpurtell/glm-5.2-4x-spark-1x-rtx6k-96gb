#!/usr/bin/env python3
from __future__ import annotations

import _pinned_sparkinfer  # noqa: F401

import argparse
import os
from pathlib import Path


PROJECTIONS = (
    ("q_b_m8", 16_384, 2_048, 8, None, None),
    ("q_b_m16_candidate", 16_384, 2_048, 16, None, None),
    ("o_proj_m1", 6_144, 16_384, 1, None, None),
    ("o_proj_m16_candidate", 6_144, 16_384, 16, None, None),
    ("o_proj_m1_tn64_candidate", 6_144, 16_384, 1, 128, 64),
)


def export_kernels(output_dir: Path) -> None:
    os.environ["SPARKINFER_COMPILE_DISK_CACHE"] = "0"
    os.environ["SPARKINFER_COMPILE_MEMORY_CACHE"] = "0"

    import torch
    from sparkinfer.moe._shared.kernels.w4a16.kernel import (
        _select_tile_config,
        compile_w4a16_gemm,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    sms = int(properties.multi_processor_count)
    max_shared_mem = int(properties.shared_memory_per_block_optin)
    metadata = []
    config = ["#pragma once", ""]

    for label, size_n, size_k, size_m, forced_tile_k, forced_tile_n in PROJECTIONS:
        if forced_tile_k is None or forced_tile_n is None:
            tile_k, tile_n, _, _ = _select_tile_config(
                problem_m=size_m,
                problem_n=size_n,
                problem_k=size_k,
                top_k=1,
                moe_block_size=8,
                sms=sms,
                max_shared_mem=max_shared_mem,
                scale_format="e4m3_k16",
            )
        else:
            tile_k = forced_tile_k
            tile_n = forced_tile_n
        kernel = compile_w4a16_gemm(
            size_m=size_m,
            size_n=size_n,
            size_k=size_k,
            num_experts=1,
            top_k=1,
            mul_topk_weights=False,
            tile_n=tile_n,
            tile_k=tile_k,
            moe_block_size=8,
            max_m_blocks=(size_m + 7) // 8,
            element_dtype="bf16",
            scale_format="e4m3_k16",
        )
        name = f"coordinator_w4a16_{label}"
        kernel.compiled.export_to_c(
            str(output_dir),
            name,
            f"glmrt_b12x_{name}",
        )
        grid_x = sms * int(kernel.blocks_per_sm)
        macro_label = label.upper()
        config.extend(
            (
                f"#define GLMRT_B12X_COORDINATOR_{macro_label}_SIZE_N {size_n}",
                f"#define GLMRT_B12X_COORDINATOR_{macro_label}_SIZE_K {size_k}",
                f"#define GLMRT_B12X_COORDINATOR_{macro_label}_GRID_X {grid_x}",
                "",
            )
        )
        metadata.append(
            f"{label}=m:{size_m},n:{size_n},k:{size_k},tile_n:{tile_n},tile_k:{tile_k},grid:{grid_x}"
        )

    (output_dir / "b12x_coordinator_aot.meta").write_text(
        "\n".join(metadata) + "\n",
        encoding="ascii",
    )
    (output_dir / "b12x_coordinator_aot_config.h").write_text(
        "\n".join(config),
        encoding="ascii",
    )
    (output_dir / "b12x_coordinator_aot.stamp").touch()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export coordinator decode-only SparkInfer W4A16 projections."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export_kernels(args.output_dir.resolve())


if __name__ == "__main__":
    main()
