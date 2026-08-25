from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).parents[2]
PROFILE_PATH = ROOT / "python" / "tools" / "_b12x_exl3_k3_profile.py"
SPEC = importlib.util.spec_from_file_location("_glmrt_exl3_k3_profile", PROFILE_PATH)
assert SPEC is not None and SPEC.loader is not None
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)


def test_exl3_k3_profile_covers_every_aot_bucket() -> None:
    assert tuple(PROFILE.EXL3_K3_TILE_CONFIG_BY_M) == PROFILE.EXL3_K3_AOT_REGIMES
    assert tuple(PROFILE.EXL3_K3_GRID_X_BY_M) == PROFILE.EXL3_K3_AOT_REGIMES
    assert PROFILE.EXL3_K3_AOT_REGIMES == (
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
    for rows, tile_config in PROFILE.EXL3_K3_TILE_CONFIG_BY_M.items():
        assert PROFILE.exl3_k3_tile_config(rows) == tile_config
        assert len(tile_config) == 4
        assert all(value >= 64 and value % 64 == 0 for value in tile_config)
        assert PROFILE.exl3_k3_grid_x(rows) == PROFILE.EXL3_K3_GRID_X_BY_M[rows]
        assert PROFILE.exl3_k3_grid_x(rows) > 0
        assert PROFILE.exl3_k3_route_block_rows(rows) in (8, 16, 32, 48, 64)


def test_exl3_k3_profile_keeps_decode_and_large_prefill_winners_distinct() -> None:
    assert PROFILE.exl3_k3_tile_config(1) == PROFILE.K64_N256
    assert PROFILE.exl3_k3_tile_config(2) == PROFILE.K128_N128_FC1
    assert PROFILE.exl3_k3_tile_config(8) == PROFILE.K128_N128_FC1
    assert PROFILE.exl3_k3_tile_config(256) == PROFILE.K64_N128
    assert PROFILE.exl3_k3_tile_config(257) == PROFILE.K64_N128
    assert PROFILE.exl3_k3_tile_config(1024) == PROFILE.K64_N128
    assert PROFILE.exl3_k3_tile_config(2048) == PROFILE.K64_N256
    assert PROFILE.exl3_k3_tile_config(2064) == PROFILE.K64_N256
    assert PROFILE.exl3_k3_grid_x(1) == 44
    assert PROFILE.exl3_k3_grid_x(2) == 44
    assert PROFILE.exl3_k3_grid_x(8) == 44
    assert PROFILE.exl3_k3_grid_x(9) == 40
    assert PROFILE.exl3_k3_grid_x(16) == 44
    assert PROFILE.exl3_k3_grid_x(256) == 144
    assert PROFILE.exl3_k3_grid_x(257) == 144
    assert PROFILE.exl3_k3_grid_x(512) == 44
    assert PROFILE.exl3_k3_grid_x(1024) == 96
    assert PROFILE.exl3_k3_grid_x(2064) == 48


def test_exl3_k3_capacity_selection_uses_exact_specializations() -> None:
    assert PROFILE.exl3_k3_capacity_rows(1) == 1
    assert PROFILE.exl3_k3_capacity_rows(3) == 4
    assert PROFILE.exl3_k3_capacity_rows(8) == 8
    assert PROFILE.exl3_k3_capacity_rows(9) == 9
    assert PROFILE.exl3_k3_capacity_rows(10) == 16
    assert PROFILE.exl3_k3_capacity_rows(256) == 256
    assert PROFILE.exl3_k3_capacity_rows(257) == 257
    assert PROFILE.exl3_k3_capacity_rows(258) == 512
    assert PROFILE.exl3_k3_capacity_rows(2048) == 2048
    assert PROFILE.exl3_k3_capacity_rows(2049) == 2064
    assert PROFILE.exl3_k3_capacity_rows(2064) == 2064


def test_exl3_k3_route_block_policy_covers_exact_m257_abi() -> None:
    assert PROFILE.exl3_k3_route_block_rows(128) == 8
    assert PROFILE.exl3_k3_route_block_rows(256) == 16
    assert PROFILE.exl3_k3_route_block_rows(257) == 16
    assert PROFILE.exl3_k3_route_block_rows(512) == 32
    assert PROFILE.exl3_k3_route_block_rows(1024) == 48
    assert PROFILE.exl3_k3_route_block_rows(2048) == 64
    assert PROFILE.exl3_k3_route_block_rows(2064) == 64


def test_source_grid_evidence_supports_the_provisional_profile() -> None:
    evidence = json.loads(
        (
            ROOT
            / "scripts"
            / "tests"
            / "fixtures"
            / "glm52-exl3-k3-source-grid-evidence.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    assert evidence["schema"] == "glmrt-glm52-exl3-k3-source-grid-evidence-v1"
    cases = {item["label"]: item["case"] for item in evidence["cases"]}

    m1 = cases["m1_grid44_vs48_final"]
    assert (m1["grid_x"], m1["candidate_comparison"]["grid_x"]) == (44, 48)
    assert m1["candidate_comparison"]["delta_percent"] > 1.0

    m9_random = cases["m9_random_grid40_repeat"]["candidate_comparison"]
    m9_reuse = cases["m9_high_reuse_grid40"]["candidate_comparison"]
    assert m9_random["grid_x"] == m9_reuse["grid_x"] == 40
    assert m9_random["delta_percent"] < 0.0
    assert m9_reuse["delta_percent"] > 10.0
    # The synthetic source-weight sweep was provisional. Completed-model
    # calibrated route replay subsequently selected the static M=9 grid 40.
    assert PROFILE.exl3_k3_grid_x(9) == 40

    tail = cases["m2048_vs_capacity2064_high_reuse"]["candidate_comparison"]
    assert tail["capacity_rows"] == 2064
    assert abs(tail["delta_percent"]) < 1.0
    assert cases["m2049_boundary"]["capacity_rows"] == 2064
