"""Qualified GLM-5.2 TP4 EXL3 K3 Spark kernel profile."""

from __future__ import annotations


EXL3_K3_AOT_REGIMES = (
    1,
    2,
    4,
    8,
    9,
    16,
    32,
    64,
    128,
    256,
    257,
    512,
    1024,
    2048,
    2064,
)

K64_N256 = (64, 256, 64, 256)
K128_N128_FC1 = (128, 128, 64, 256)
K64_N128 = (64, 128, 64, 128)

# Measured on the exact GB10 H6144/I512/E256/top-k=8 full-rotation geometry.
# A per-regime profile avoids trading decode wins for larger-M throughput.
EXL3_K3_TILE_CONFIG_BY_M = {
    1: K64_N256,
    2: K128_N128_FC1,
    4: K64_N256,
    8: K128_N128_FC1,
    9: K128_N128_FC1,
    16: K64_N256,
    32: K64_N256,
    64: K64_N256,
    128: K128_N128_FC1,
    256: K64_N128,
    # Balanced serving emits a 256-row prefill chunk together with the first
    # decode row. Route replay shows M=257 is common enough that rounding it
    # to the geometrically different M=512 kernel leaves material throughput
    # on the table. Its exact tile sweep selected the M=256 K64/N128 shape.
    257: K64_N128,
    512: K128_N128_FC1,
    1024: K64_N128,
    2048: K64_N256,
    # A full 2,048-row prefill wave can carry one target row plus at most 15
    # dSpark draft rows. Keep that rare combined tail in one dispatch. It is
    # geometrically only 0.8% wider than M=2048, so inherit the M=2048 tile
    # until the coordinator is available for a dedicated tail sweep.
    2064: K64_N256,
}

# Persistent grids emitted by the qualified source-time GB10 sweep. Keep this
# table explicit so the completed-model route replay can change a grid without
# modifying the exporter or the native dispatcher. The exporter independently
# verifies that each value is no larger than the compiled tile's cooperative
# residency limit.
EXL3_K3_GRID_X_BY_M = {
    # The complete native call (NVFP4 input decode, fused MoE, and rotated
    # top-k sum) favored 44 over 48 by 2.11% and 1.37% in independent
    # alternating M=1 runs. M=1 has only one possible expert-reuse degree.
    1: 44,
    # These grids are the static, no-tail-regression compromises from exact
    # calibrated layer-40 weights replaying the accepted production route
    # capture. They replace the earlier source-weight provisional grids.
    2: 44,
    4: 48,
    8: 44,
    9: 40,
    16: 44,
    32: 48,
    64: 48,
    128: 48,
    256: 144,
    # M=257 uses the same winning tile as M=256. Start from its safe
    # three-block-per-SM grid; the compiled exact kernel is re-swept during
    # completed-model qualification before publication.
    257: 144,
    512: 44,
    1024: 96,
    # The M=2048/2064 kernel uses one 244-register CTA per SM. Completion work
    # is scheduled on a high-priority stream so this grid can retain all 48
    # SMs without starving the short FP8/RDMA staging kernels between CTAs.
    2048: 48,
    2064: 48,
}

if tuple(EXL3_K3_TILE_CONFIG_BY_M) != EXL3_K3_AOT_REGIMES:
    raise RuntimeError("EXL3 K3 tile profile does not cover every AOT regime in order")
if tuple(EXL3_K3_GRID_X_BY_M) != EXL3_K3_AOT_REGIMES:
    raise RuntimeError("EXL3 K3 grid profile does not cover every AOT regime in order")


def exl3_k3_tile_config(capacity_rows: int) -> tuple[int, int, int, int]:
    try:
        return EXL3_K3_TILE_CONFIG_BY_M[int(capacity_rows)]
    except KeyError as exc:
        raise ValueError(f"unsupported EXL3 K3 AOT capacity {capacity_rows}") from exc


def exl3_k3_grid_x(capacity_rows: int) -> int:
    try:
        return EXL3_K3_GRID_X_BY_M[int(capacity_rows)]
    except KeyError as exc:
        raise ValueError(f"unsupported EXL3 K3 AOT capacity {capacity_rows}") from exc


def exl3_k3_route_block_rows(capacity_rows: int) -> int:
    """Return SparkInfer's packed top-8 route-block ABI for one AOT bucket."""

    capacity_rows = int(capacity_rows)
    if capacity_rows not in EXL3_K3_AOT_REGIMES:
        raise ValueError(f"unsupported EXL3 K3 AOT capacity {capacity_rows}")
    route_count = capacity_rows * 8
    for block_rows in (8, 16, 32, 48, 64):
        if 10 * route_count < 9 * 256 * block_rows:
            return block_rows
    return 64


def exl3_k3_capacity_rows(live_rows: int) -> int:
    live_rows = int(live_rows)
    if live_rows <= 0:
        raise ValueError("EXL3 K3 live rows must be positive")
    for capacity_rows in EXL3_K3_AOT_REGIMES:
        if live_rows <= capacity_rows:
            return capacity_rows
    raise ValueError(
        f"EXL3 K3 live rows {live_rows} exceed maximum {EXL3_K3_AOT_REGIMES[-1]}"
    )
