#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path


ROWS = tuple(range(2, 9))
TOP_K = 8


def export_kernels(output_dir: Path, target_sms: int) -> None:
    # CuTe's C exporter needs the compiled IR, which executable-only cache hits
    # do not retain.
    os.environ["B12X_CUTE_COMPILE_DISK_CACHE"] = "0"
    os.environ["B12X_CUTE_COMPILE_MEMORY_CACHE"] = "0"

    import cuda.bindings.driver as cuda
    import torch
    import b12x.moe.fused.w4a16.kernel as w4a16_kernel
    from b12x.moe.fused.w4a16.host import select_route_block_size_m

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    torch.empty(1, dtype=torch.uint8, device=device)
    properties = torch.cuda.get_device_properties(device)
    physical_sms = int(properties.multi_processor_count)
    if target_sms <= 0 or target_sms > physical_sms:
        raise ValueError(
            f"target_sms must be in 1..{physical_sms} on this export host, got {target_sms}"
        )
    # These objects execute on 48-SM GB10 Sparks.  The SM count is part of the
    # kernel ABI: it fixes the cooperative-grid barrier offsets in `locks`.
    # Keeping it explicit also lets the coordinator GPU build a numerically
    # representative Spark object without silently compiling 188-SM offsets.
    sms = target_sms
    max_shared_mem = int(properties.shared_memory_per_block_optin)

    # b12x's public compile helper uses its `sms` argument for tile/grid
    # planning, but W4A16GemmKernel.__init__ independently rereads the export
    # device and uses that physical SM count for its internal barrier offsets.
    # The exporter is a dedicated process, so present a target-property proxy
    # for the duration of compilation and keep both halves of the ABI aligned.
    physical_get_device_properties = torch.cuda.get_device_properties

    class TargetDeviceProperties:
        def __init__(self, base: object) -> None:
            self._base = base

        @property
        def multi_processor_count(self) -> int:
            return target_sms

        def __getattr__(self, name: str) -> object:
            return getattr(self._base, name)

    def target_get_device_properties(device_arg: object = None) -> object:
        return TargetDeviceProperties(physical_get_device_properties(device_arg))

    torch.cuda.get_device_properties = target_get_device_properties

    # b12x 0.30.2 limits ordered direct-top-k to M<=6 only because its public
    # M=7/8 policy selects the atomic fused-sum TC-decode epilogue. The direct
    # route GEMM itself supports all 64 M=8 route rows. Export that same kernel
    # with separate route outputs so glmrt can reduce top-k in fixed order.
    if w4a16_kernel._MAX_DIRECT_TOPK_ROUTE_M < max(ROWS):
        w4a16_kernel._MAX_DIRECT_TOPK_ROUTE_M = max(ROWS)

    # Request the wider FC2 tile through the upstream TC-decode planner while
    # retaining grouped execution and its fixed-order reduction. Upstream
    # couples that tile selection to the direct-route atomic epilogue, so
    # suppress the epilogue during construction for grouped objects.
    original_fused_init = w4a16_kernel.W4A16FusedMoeKernel.__init__

    def allow_grouped_wide_fixed_order(self, *args, **kwargs):
        requested = bool(kwargs.get("tc_decode_fused_sum"))
        grouped = not bool(kwargs.get("direct_topk_routes"))
        if requested and grouped:
            kwargs["tc_decode_fused_sum"] = False
        original_fused_init(self, *args, **kwargs)

    w4a16_kernel.W4A16FusedMoeKernel.__init__ = allow_grouped_wide_fixed_order

    launch = w4a16_kernel.W4A16FusedMoeKernel.__call__
    launch.__annotations__["stream"] = cuda.CUstream
    wrapped = getattr(launch, "__wrapped__", None)
    if wrapped is not None:
        wrapped.__annotations__["stream"] = cuda.CUstream

    config_lines = ["#pragma once"]
    metadata_lines = [
        f"b12x_version={importlib.metadata.version('b12x')}",
        f"target_sms={sms}",
    ]
    for rows in ROWS:
        block_size = select_route_block_size_m(rows, TOP_K, 256)
        fused = w4a16_kernel.compile_w4a16_fused_moe(
            size_m=rows,
            hidden_size=6144,
            intermediate_size=512,
            num_experts=256,
            top_k=TOP_K,
            activation="silu",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            moe_block_size=block_size,
            max_m_blocks=rows * TOP_K,
            element_dtype="bf16",
            fast_math=True,
            sms=sms,
            max_shared_mem=max_shared_mem,
            weight_layout="packed",
            scale_format="e4m3_k16",
            direct_topk_routes=True,
            # Atomic BF16 route accumulation is not scalar deterministic.
            # Emit per-route BF16 rows and reduce them later in route order.
            tc_decode_fused_sum=False,
        )
        name = f"moe_tp4_w4a16_m1_parity_m{rows}_topk8"
        fused.compiled.export_to_c(
            str(output_dir),
            name,
            f"glmrt_b12x_{name}",
        )
        grid_x = w4a16_kernel._w4a16_fused_persistent_grid_x(
            fused=fused,
            m=rows,
            topk=TOP_K,
            intermediate_size=512,
            activation="silu",
            direct_topk_routes=True,
            sms=sms,
        )
        macro = f"GLMRT_B12X_W4A16_M1_PARITY_M{rows}_TOPK8"
        config_lines.append(f"#define {macro}_GRID_X {grid_x}")
        metadata_lines.append(
            f"m{rows}=grid:{grid_x},block:{block_size},routes:{rows * TOP_K},"
            "direct_topk:1,ordered_route_output:1"
        )

        # Preserve the same block-8 GEMM arithmetic as scalar M=1 while
        # grouping logical routes by expert.  The grouped kernel scatters each
        # route result back to its original logical route row; glmrt then folds
        # top-k in fixed order.  This isolates route execution order from the
        # numerical contract and recovers weight reuse when several target
        # rows select the same expert.
        grouped_block_size = 8
        grouped = w4a16_kernel.compile_w4a16_fused_moe(
            size_m=rows,
            hidden_size=6144,
            intermediate_size=512,
            num_experts=256,
            top_k=TOP_K,
            activation="silu",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            moe_block_size=grouped_block_size,
            max_m_blocks=rows * TOP_K,
            element_dtype="bf16",
            fast_math=True,
            sms=sms,
            max_shared_mem=max_shared_mem,
            weight_layout="packed",
            scale_format="e4m3_k16",
            direct_topk_routes=False,
            tc_decode_fused_sum=False,
        )
        grouped_name = (
            f"moe_tp4_w4a16_m1_parity_grouped_m{rows}_topk8"
        )
        grouped.compiled.export_to_c(
            str(output_dir),
            grouped_name,
            f"glmrt_b12x_{grouped_name}",
        )
        grouped_grid_x = w4a16_kernel._w4a16_fused_persistent_grid_x(
            fused=grouped,
            m=rows,
            topk=TOP_K,
            intermediate_size=512,
            activation="silu",
            direct_topk_routes=False,
            sms=sms,
        )
        grouped_macro = (
            f"GLMRT_B12X_W4A16_M1_PARITY_GROUPED_M{rows}_TOPK8"
        )
        config_lines.append(
            f"#define {grouped_macro}_GRID_X {grouped_grid_x}"
        )
        metadata_lines.append(
            f"grouped_m{rows}=grid:{grouped_grid_x},"
            f"block:{grouped_block_size},max_blocks:{rows * TOP_K},"
            "direct_topk:0,ordered_route_output:1"
        )

        # Same grouped route execution and fixed-order output as the original
        # object, with the wider FC2 tile chosen by the TC-decode planner.
        grouped_wide = w4a16_kernel.compile_w4a16_fused_moe(
            size_m=rows,
            hidden_size=6144,
            intermediate_size=512,
            num_experts=256,
            top_k=TOP_K,
            activation="silu",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            moe_block_size=grouped_block_size,
            max_m_blocks=rows * TOP_K,
            element_dtype="bf16",
            fast_math=True,
            sms=sms,
            max_shared_mem=max_shared_mem,
            weight_layout="packed",
            scale_format="e4m3_k16",
            direct_topk_routes=False,
            tc_decode_fused_sum=True,
        )
        grouped_wide_name = (
            f"moe_tp4_w4a16_m1_parity_grouped_wide_m{rows}_topk8"
        )
        grouped_wide.compiled.export_to_c(
            str(output_dir),
            grouped_wide_name,
            f"glmrt_b12x_{grouped_wide_name}",
        )
        grouped_wide_grid_x = w4a16_kernel._w4a16_fused_persistent_grid_x(
            fused=grouped_wide,
            m=rows,
            topk=TOP_K,
            intermediate_size=512,
            activation="silu",
            direct_topk_routes=False,
            sms=sms,
        )
        grouped_wide_macro = (
            f"GLMRT_B12X_W4A16_M1_PARITY_GROUPED_WIDE_M{rows}_TOPK8"
        )
        config_lines.append(
            f"#define {grouped_wide_macro}_GRID_X {grouped_wide_grid_x}"
        )
        metadata_lines.append(
            f"grouped_wide_m{rows}=grid:{grouped_wide_grid_x},"
            f"block:{grouped_block_size},max_blocks:{rows * TOP_K},"
            f"fc2_tile:{grouped_wide.fc2_tile_n}x{grouped_wide.fc2_tile_k},"
            "direct_topk:0,ordered_route_output:1"
        )

    (output_dir / "b12x_spark_w4a16_m1_parity_aot_config.h").write_text(
        "\n".join(config_lines) + "\n", encoding="ascii"
    )
    (output_dir / "b12x_spark_w4a16_m1_parity_aot.meta").write_text(
        "\n".join(metadata_lines) + "\n", encoding="ascii"
    )
    (output_dir / "b12x_spark_w4a16_m1_parity_aot.stamp").write_text(
        "ready\n", encoding="ascii"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export ordered direct-top-k Spark W4A16 M=2..8 candidates."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-sms", type=int, default=48)
    args = parser.parse_args()
    export_kernels(args.output_dir.resolve(), args.target_sms)


if __name__ == "__main__":
    main()
