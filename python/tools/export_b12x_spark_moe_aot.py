#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path


PREFILL_REGIMES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
DECODE_GRID_X = 32
TOP1_M1_GRID_X = 32
TOP1_MULTIROW_GRID_X = 48


def prepare_export(output_dir: Path):
    # A disk-cache hit is executable-only and has no IR for export_to_c().
    os.environ["B12X_CUTE_COMPILE_DISK_CACHE"] = "0"
    os.environ["B12X_CUTE_COMPILE_MEMORY_CACHE"] = "0"

    import cuda.bindings.driver as cuda
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    torch.empty(1, dtype=torch.uint8, device=device)
    return cuda, torch, device


def annotate_stream(backend, cuda) -> None:
    # B12X 0.30 omits the stream annotation required by CuTe's C exporter.
    launch = backend.__call__
    launch.__annotations__["stream"] = cuda.CUstream
    wrapped = getattr(launch, "__wrapped__", None)
    if wrapped is not None:
        wrapped.__annotations__["stream"] = cuda.CUstream


def w4a4_prefill_workspace_config(torch, device) -> tuple[list[str], str]:
    from b12x.integration.tp_moe import (
        TPMoEScratchCaps,
        plan_b12x_fp4_moe_weights,
        plan_tp_moe_scratch,
    )

    weight_plan = plan_b12x_fp4_moe_weights(
        quant_modes="nvfp4",
        source_format="modelopt_nvfp4",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=256,
        hidden_size=6144,
        intermediate_size=512,
        w13_layout="w13",
    )
    scratch_plan = plan_tp_moe_scratch(
        TPMoEScratchCaps(
            max_tokens=256,
            num_topk=8,
            device=device,
            weight_plan=weight_plan,
            quant_mode="nvfp4",
            core_token_counts=(256,),
            route_num_experts=0,
            frozen=True,
        )
    )
    core_plan = scratch_plan._core_workspace_plan
    if core_plan.implementation != "dynamic":
        raise RuntimeError(
            f"M256 W4A4 unexpectedly planned {core_plan.implementation!r}"
        )
    if (
        core_plan.dynamic_physical_tiles is None
        or core_plan.dynamic_task_capacity is None
    ):
        raise RuntimeError("M256 W4A4 dynamic workspace has no task geometry")

    prefix = "GLMRT_B12X_W4A4_PREFILL_M256_TOPK8"
    config_lines = [
        f"#define {prefix}_MAX_ROUTED_ROWS {core_plan.routed_rows}",
        f"#define {prefix}_PHYSICAL_TILES {core_plan.dynamic_physical_tiles}",
        f"#define {prefix}_TASK_CAPACITY {core_plan.dynamic_task_capacity}",
    ]
    offset = 0
    rows_padded = None
    for spec in core_plan.tensor_specs:
        element_bytes = torch.empty((), dtype=spec.dtype).element_size()
        alignment = max(16, element_bytes)
        offset = (offset + alignment - 1) // alignment * alignment
        nbytes = math.prod(spec.shape) * element_bytes
        macro = spec.name.upper()
        config_lines.extend(
            [
                f"#define {prefix}_{macro}_OFFSET {offset}",
                f"#define {prefix}_{macro}_BYTES {nbytes}",
            ]
        )
        if spec.name == "packed_input":
            rows_padded = int(spec.shape[1])
        offset += nbytes
    if offset != scratch_plan.layout.core_workspace_nbytes:
        raise RuntimeError(
            "M256 W4A4 workspace layout drifted: "
            f"mapped {offset} bytes, planned "
            f"{scratch_plan.layout.core_workspace_nbytes}"
        )
    if rows_padded is None:
        raise RuntimeError("M256 W4A4 workspace has no packed input")
    config_lines.extend(
        [
            f"#define {prefix}_ROWS_PADDED {rows_padded}",
            f"#define {prefix}_SCRATCH_BYTES {scratch_plan.layout.total_nbytes}",
        ]
    )
    metadata = (
        f"scratch:{scratch_plan.layout.total_nbytes},rows_padded:{rows_padded},"
        f"physical_tiles:{core_plan.dynamic_physical_tiles},"
        f"tasks:{core_plan.dynamic_task_capacity}"
    )
    return config_lines, metadata


def export_w4a4_prefill(output_dir: Path, cuda, torch, device) -> tuple[int, int]:
    from b12x.integration.tp_moe import _get_dynamic_kernel, _get_impl_mac
    from b12x.moe.fused.dynamic import MoEDynamicKernelBackend

    rows = 256
    top_k = 8
    routed_rows = rows * top_k
    grid_x = _get_impl_mac("dynamic", routed_rows=routed_rows)
    annotate_stream(MoEDynamicKernelBackend, cuda)
    compiled, effective_grid_x = _get_dynamic_kernel(
        256,
        rows,
        6144,
        512,
        top_k,
        routed_rows,
        topk_ids_dtype=torch.int32,
        fast_math=True,
        mac_override=grid_x,
        activation="silu",
        quant_mode="nvfp4",
        direct_routing=False,
        share_input_across_experts=False,
        deterministic_output=False,
        swiglu_limit=None,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
    )
    name = "moe_tp4_w4a4_prefill_m256_topk8"
    compiled.export_to_c(str(output_dir), name, f"glmrt_b12x_{name}")
    return effective_grid_x, routed_rows


def export_w4a4_prefill_only(output_dir: Path) -> None:
    cuda, torch, device = prepare_export(output_dir)
    grid_x, max_rows = export_w4a4_prefill(output_dir, cuda, torch, device)
    workspace_lines, workspace_metadata = w4a4_prefill_workspace_config(
        torch, device
    )
    (output_dir / "b12x_spark_moe_aot_config.h").write_text(
        "#pragma once\n"
        f"#define GLMRT_B12X_W4A4_PREFILL_M256_TOPK8_GRID_X {grid_x}\n"
        + "\n".join(workspace_lines)
        + "\n",
        encoding="ascii",
    )
    (output_dir / "b12x_spark_moe_aot.meta").write_text(
        "w4a4_prefill_m256_topk8="
        f"grid:{grid_x},max_rows:{max_rows},{workspace_metadata}\n",
        encoding="ascii",
    )


def export_kernels(output_dir: Path, target_sms: int) -> None:
    cuda, torch, device = prepare_export(output_dir)
    from b12x.integration.tp_moe import _get_micro_kernel
    from b12x.moe.fused.micro import MoEMicroKernelBackend

    annotate_stream(MoEMicroKernelBackend, cuda)
    compiled, grid_x = _get_micro_kernel(
        256,
        1,
        6144,
        512,
        8,
        topk_ids_dtype=torch.int32,
        fast_math=True,
        share_input_across_experts=True,
        share_expert_scales=False,
        single_token=True,
        activation="silu",
        device=device,
        quant_mode="nvfp4",
        swiglu_limit=None,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
    )
    name = "moe_tp4_m1"
    compiled.export_to_c(
        str(output_dir),
        name,
        "glmrt_b12x_moe_tp4_m1",
    )

    from b12x.moe.fused.w4a16.host import (
        max_packed_route_slots,
        select_route_block_size_m,
    )
    from b12x.moe.fused.w4a16.kernel import (
        W4A16FusedMoeKernel,
        _w4a16_fused_persistent_grid_x,
        compile_w4a16_fused_moe,
    )

    w4a16_launch = W4A16FusedMoeKernel.__call__
    w4a16_launch.__annotations__["stream"] = cuda.CUstream
    w4a16_wrapped = getattr(w4a16_launch, "__wrapped__", None)
    if w4a16_wrapped is not None:
        w4a16_wrapped.__annotations__["stream"] = cuda.CUstream

    properties = torch.cuda.get_device_properties(device)
    physical_sms = int(properties.multi_processor_count)
    if target_sms <= 0 or target_sms > physical_sms:
        raise ValueError(
            f"target_sms must be in 1..{physical_sms} on this export host, got {target_sms}"
        )
    sms = target_sms
    max_shared_mem = int(properties.shared_memory_per_block_optin)
    config_lines = [
        "#pragma once",
        f"#define GLMRT_B12X_MOE_TP4_M1_GRID_X {grid_x}",
    ]
    metadata_lines = [f"micro_grid_x={grid_x}", f"w4a16_target_sms={sms}"]

    dynamic_grid_x, dynamic_routed_rows = export_w4a4_prefill(
        output_dir, cuda, torch, device
    )
    config_lines.append(
        "#define GLMRT_B12X_W4A4_PREFILL_M256_TOPK8_GRID_X "
        f"{dynamic_grid_x}"
    )
    workspace_lines, workspace_metadata = w4a4_prefill_workspace_config(
        torch, device
    )
    config_lines.extend(workspace_lines)
    metadata_lines.append(
        "w4a4_prefill_m256_topk8="
        f"grid:{dynamic_grid_x},max_rows:{dynamic_routed_rows},"
        f"{workspace_metadata}"
    )

    # compile_w4a16_fused_moe() uses its `sms` argument for planning, but the
    # underlying W4A16 kernel independently rereads the export GPU and bakes
    # that physical SM count into cooperative-barrier offsets.  Present the
    # 48-SM GB10 target while exporting W4A16 so its lock ABI remains valid
    # even when this script is run on the 188-SM coordinator.
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

    def export_w4a16(
        *,
        rows: int,
        top_k: int,
        label: str,
        weight_layout: str = "packed",
        direct_topk: bool | None = None,
        block_size: int | None = None,
        tc_decode_fused_sum: bool | None = None,
    ) -> None:
        block_size_overridden = block_size is not None
        if block_size is None:
            block_size = select_route_block_size_m(rows, top_k, 256)
        if direct_topk is None:
            direct_topk = rows <= (8 if top_k == 1 else 4)
        if tc_decode_fused_sum is None:
            tc_decode_fused_sum = direct_topk and top_k == 1
        if top_k == 8 and not direct_topk and not block_size_overridden:
            # This is part of the native route-metadata ABI, not just a kernel
            # tuning choice. Keep it aligned with route.rs and the benchmark.
            block_size = 32 if rows <= 2048 else 48
        packed_route_slots = max_packed_route_slots(rows * top_k, block_size, 256)
        max_m_blocks = (
            rows * top_k
            if direct_topk
            else (packed_route_slots + block_size - 1) // block_size
        )
        fused = compile_w4a16_fused_moe(
            size_m=rows,
            hidden_size=6144,
            intermediate_size=512,
            num_experts=256,
            top_k=top_k,
            activation="silu",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            moe_block_size=block_size,
            max_m_blocks=max_m_blocks,
            element_dtype="bf16",
            fast_math=True,
            sms=sms,
            max_shared_mem=max_shared_mem,
            weight_layout=weight_layout,
            scale_format="e4m3_k16",
            direct_topk_routes=direct_topk,
            # Keep route outputs separate. The native wrapper reduces them in
            # fixed route order, avoiding atomic top-k accumulation.
            tc_decode_fused_sum=tc_decode_fused_sum,
        )
        export_name = f"moe_tp4_w4a16_{label}"
        fused.compiled.export_to_c(
            str(output_dir),
            export_name,
            f"glmrt_b12x_{export_name}",
        )
        persistent_grid = _w4a16_fused_persistent_grid_x(
            fused=fused,
            m=rows,
            topk=top_k,
            intermediate_size=512,
            activation="silu",
            direct_topk_routes=direct_topk,
            sms=sms,
        )
        if label.endswith("decode_m1"):
            persistent_grid = DECODE_GRID_X
        elif label == "prefill_m2048_topk8":
            persistent_grid = 80
        elif top_k == 1:
            persistent_grid = (
                TOP1_M1_GRID_X if rows == 1 else TOP1_MULTIROW_GRID_X
            )
        macro = label.upper()
        config_lines.extend(
            [
                f"#define GLMRT_B12X_W4A16_{macro}_GRID_X {persistent_grid}",
                f"#define GLMRT_B12X_W4A16_{macro}_BLOCK_SIZE {block_size}",
                f"#define GLMRT_B12X_W4A16_{macro}_PACKED_ROUTE_SLOTS {packed_route_slots}",
                f"#define GLMRT_B12X_W4A16_{macro}_MAX_M_BLOCKS {max_m_blocks}",
            ]
        )
        metadata_lines.append(
            f"{label}=grid:{persistent_grid},block:{block_size},"
            f"route_slots:{packed_route_slots},max_m_blocks:{max_m_blocks},"
            f"layout:{weight_layout},direct_topk:{int(direct_topk)},"
            f"tc_decode_fused_sum:{int(tc_decode_fused_sum)}"
        )

    export_w4a16(rows=1, top_k=8, label="decode_m1")
    export_w4a16(
        rows=1,
        top_k=8,
        label="decode_m1_fused_sum",
        direct_topk=True,
        tc_decode_fused_sum=True,
    )
    export_w4a16(
        rows=1,
        top_k=8,
        label="modelopt_decode_m1",
        weight_layout="modelopt",
        direct_topk=False,
        block_size=8,
    )
    for rows in PREFILL_REGIMES[1:]:
        export_w4a16(rows=rows, top_k=8, label=f"prefill_m{rows}_topk8")
    export_w4a16(rows=512, top_k=8, label="prefill_m512_topk8")
    export_w4a16(rows=1024, top_k=8, label="prefill_m1024_topk8")
    export_w4a16(rows=2048, top_k=8, label="prefill_m2048_topk8")
    for rows in PREFILL_REGIMES:
        export_w4a16(rows=rows, top_k=1, label=f"top1_m{rows}")

    (output_dir / "b12x_spark_moe_aot_config.h").write_text(
        "\n".join(config_lines) + "\n",
        encoding="ascii",
    )
    (output_dir / "b12x_spark_moe_aot.meta").write_text(
        "\n".join(metadata_lines) + "\n",
        encoding="ascii",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the Spark B12X MoE AOT kernels."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-sms", type=int, default=48)
    parser.add_argument(
        "--only-w4a4-prefill",
        action="store_true",
        help="export only the M256 dynamic W4A4 prefill kernel",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if args.only_w4a4_prefill:
        export_w4a4_prefill_only(output_dir)
    else:
        export_kernels(output_dir, args.target_sms)


if __name__ == "__main__":
    main()
