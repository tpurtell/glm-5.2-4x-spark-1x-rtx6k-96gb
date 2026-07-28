#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path


DIRECT_ROWS = tuple(range(1, 9))
PREFILL_CAPACITIES = (16, 32, 64, 128, 256, 512, 1024, 2048)
HIDDEN = 6144
INTERMEDIATE = 512
EXPERTS = 256
TOP_K = 8


def annotate_stream(backend: type, cuda: object) -> None:
    launch = backend.__call__
    launch.__annotations__["stream"] = cuda.CUstream
    wrapped = getattr(launch, "__wrapped__", None)
    if wrapped is not None:
        wrapped.__annotations__["stream"] = cuda.CUstream


def export_kernels(output_dir: Path, target_sms: int) -> None:
    # C export needs live compiler IR rather than an executable-only cache hit.
    os.environ["SPARKINFER_COMPILE_DISK_CACHE"] = "0"
    os.environ["SPARKINFER_COMPILE_MEMORY_CACHE"] = "0"

    import cuda.bindings.driver as cuda
    import torch
    import sparkinfer.moe._shared.kernels.w4a16.kernel as w4a16
    from sparkinfer.moe._shared.kernels.micro import MoEMicroKernelBackend
    from sparkinfer.moe._shared.kernels.w4a16.host import (
        max_packed_route_slots,
        packed_gemm_scratch_elements,
        select_route_block_size_m,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    torch.empty(1, dtype=torch.uint8, device=device)
    physical_properties = torch.cuda.get_device_properties(device)
    physical_sms = int(physical_properties.multi_processor_count)
    if target_sms <= 0 or target_sms > physical_sms:
        raise ValueError(
            f"target_sms must be in 1..{physical_sms} on this export host, "
            f"got {target_sms}"
        )
    max_shared_mem = int(physical_properties.shared_memory_per_block_optin)

    # The SM count is part of both cooperative launch planning and barrier
    # addressing. Present the 48-SM Spark target even when exporting on the
    # 188-SM coordinator GPU.
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
    annotate_stream(MoEMicroKernelBackend, cuda)
    annotate_stream(w4a16.W4A16FusedMoeKernel, cuda)

    try:
        sparkinfer_version = importlib.metadata.version("sparkinfer")
    except importlib.metadata.PackageNotFoundError:
        # A pinned source checkout supplied through PYTHONPATH is a supported
        # exporter input and does not necessarily have installed wheel metadata.
        sparkinfer_version = "source-checkout"

    config_lines = ["#pragma once"]
    metadata_lines = [
        f"sparkinfer_version={sparkinfer_version}",
        f"sparkinfer_module={Path(w4a16.__file__).resolve()}",
        f"target_sms={target_sms}",
        f"max_shared_mem={max_shared_mem}",
        "weight_layout=modelopt",
        "scale_format=e4m3_k16",
        "w13_layout=w13",
    ]

    for rows in DIRECT_ROWS:
        launch = w4a16._compile_w4a16_small_m_direct(
            m=rows,
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            num_experts=EXPERTS,
            topk=TOP_K,
            activation="silu",
            fast_math=True,
            topk_ids_dtype=torch.int32,
            device=device,
            scale_format="e4m3_k16",
            w13_layout="w13",
        )
        name = f"sparkinfer_source_w4a16_direct_m{rows}_topk8"
        launch.compiled.export_to_c(
            str(output_dir),
            name,
            f"glmrt_{name}",
        )
        macro = f"GLMRT_SPARKINFER_SOURCE_W4A16_DIRECT_M{rows}_TOPK8"
        config_lines.append(f"#define {macro}_GRID_X {launch.grid_x}")
        metadata_lines.append(
            f"direct_m{rows}=grid:{launch.grid_x},combined_output:1"
        )

    max_route_slots = 1
    max_route_blocks = 1
    max_scratch_elements = 1
    for rows in PREFILL_CAPACITIES:
        block_size = select_route_block_size_m(rows, TOP_K, EXPERTS)
        route_slots = max_packed_route_slots(
            rows * TOP_K,
            block_size,
            EXPERTS,
        )
        route_blocks = (route_slots + block_size - 1) // block_size
        fused = w4a16.compile_w4a16_fused_moe(
            size_m=rows,
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            num_experts=EXPERTS,
            top_k=TOP_K,
            activation="silu",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            moe_block_size=block_size,
            max_m_blocks=route_blocks,
            element_dtype="bf16",
            fast_math=True,
            sms=target_sms,
            max_shared_mem=max_shared_mem,
            weight_layout="modelopt",
            scale_format="e4m3_k16",
            w13_layout="w13",
            direct_topk_routes=False,
            tc_decode_fused_sum=False,
        )
        name = f"sparkinfer_source_w4a16_prefill_m{rows}_topk8"
        fused.compiled.export_to_c(
            str(output_dir),
            name,
            f"glmrt_{name}",
        )
        grid_x = w4a16._w4a16_fused_persistent_grid_x(
            fused=fused,
            m=rows,
            topk=TOP_K,
            intermediate_size=INTERMEDIATE,
            activation="silu",
            direct_topk_routes=False,
            sms=target_sms,
        )
        fc1_scratch = packed_gemm_scratch_elements(
            size_n=2 * INTERMEDIATE,
            route_slots=route_slots,
            moe_block_size=block_size,
            sms=target_sms,
        )
        fc2_scratch = packed_gemm_scratch_elements(
            size_n=HIDDEN,
            route_slots=route_slots,
            moe_block_size=block_size,
            sms=target_sms,
        )
        max_route_slots = max(max_route_slots, route_slots)
        max_route_blocks = max(max_route_blocks, route_blocks)
        max_scratch_elements = max(
            max_scratch_elements,
            fc1_scratch,
            fc2_scratch,
        )
        macro = f"GLMRT_SPARKINFER_SOURCE_W4A16_PREFILL_M{rows}_TOPK8"
        config_lines.extend(
            [
                f"#define {macro}_GRID_X {grid_x}",
                f"#define {macro}_BLOCK_SIZE {block_size}",
                f"#define {macro}_ROUTE_SLOTS {route_slots}",
                f"#define {macro}_ROUTE_BLOCKS {route_blocks}",
                f"#define {macro}_FC1_SCRATCH_ELEMENTS {fc1_scratch}",
                f"#define {macro}_FC2_SCRATCH_ELEMENTS {fc2_scratch}",
            ]
        )
        metadata_lines.append(
            f"prefill_m{rows}=grid:{grid_x},block:{block_size},"
            f"route_slots:{route_slots},route_blocks:{route_blocks},"
            f"fc1_tile:{fused.fc1_tile_n}x{fused.fc1_tile_k},"
            f"fc2_tile:{fused.fc2_tile_n}x{fused.fc2_tile_k},"
            f"blocks_per_sm:{fused.blocks_per_sm},"
            f"fc1_scratch:{fc1_scratch},fc2_scratch:{fc2_scratch}"
        )

    config_lines.extend(
        [
            f"#define GLMRT_SPARKINFER_SOURCE_W4A16_MAX_ROUTE_SLOTS {max_route_slots}",
            f"#define GLMRT_SPARKINFER_SOURCE_W4A16_MAX_ROUTE_BLOCKS {max_route_blocks}",
            f"#define GLMRT_SPARKINFER_SOURCE_W4A16_MAX_SCRATCH_ELEMENTS {max_scratch_elements}",
            "#define GLMRT_SPARKINFER_SOURCE_W4A16_LOCK_ELEMENTS 1026",
        ]
    )
    metadata_lines.extend(
        [
            f"max_route_slots={max_route_slots}",
            f"max_route_blocks={max_route_blocks}",
            f"max_scratch_elements={max_scratch_elements}",
            "lock_elements=1026",
        ]
    )
    (output_dir / "sparkinfer_source_w4a16_aot_config.h").write_text(
        "\n".join(config_lines) + "\n",
        encoding="ascii",
    )
    (output_dir / "sparkinfer_source_w4a16_aot.meta").write_text(
        "\n".join(metadata_lines) + "\n",
        encoding="utf-8",
    )
    (output_dir / "sparkinfer_source_w4a16_aot.stamp").write_text(
        "ready\n",
        encoding="ascii",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export latest SparkInfer source-layout W4A16 kernels for the "
            "48-SM GLM-5.2 TP4 expert host."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-sms", type=int, default=48)
    args = parser.parse_args()
    export_kernels(args.output_dir.resolve(), args.target_sms)


if __name__ == "__main__":
    main()
