from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
TOOL_PATH = ROOT / "python" / "tools" / "analyze_glm52_expert_startup.py"
SPEC = importlib.util.spec_from_file_location("_glmrt_expert_startup", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)
RUNTIME_FINGERPRINT = "a" * 64


def log_text(
    *,
    exl3: bool,
    missing_layer: int | None = None,
    exit_status: int | None = None,
    cooperative: bool = False,
    resident_bytes: int = TOOL.EXL3_RESIDENT_BYTES_PER_RANK_LAYER,
    role: str = "spark-0",
    runtime_fingerprint: str = RUNTIME_FINGERPRINT,
) -> str:
    model = TOOL.EXL3_MODEL if exl3 else "lukealonso/GLM-5.2-NVFP4"
    lines = [
        "== 2026-08-22T00:00:00+00:00 starting expert-9100: stale ==",
        "expertd_startup_phase stage=loadplan elapsed_ms=999.000 total_ms=999.000",
        "== 2026-08-23T00:00:00+00:00 starting expert-9100: current ==",
        "starting expertd synthetic_weights=false transport=verbs-host "
        f"listen=0.0.0.0:9100 model_id={model} "
        f"runtime_identity={runtime_fingerprint} loadplan=None "
        f'catalog_source=Some("hf://fixture") real_layer=None role=Some("{role}")',
    ]
    stages = [
        "loadplan",
        "python-capture",
        "catalog-owner-config",
        "catalog-filter-validation",
        "executor-configuration",
    ]
    for index, stage in enumerate(stages, 1):
        lines.append(
            f"expertd_startup_phase stage={stage} elapsed_ms={index}.000 "
            f"total_ms={index}.000"
        )
    if exl3:
        for layer in range(3, 78):
            if layer == missing_layer:
                continue
            if cooperative:
                lines.append(
                    "real_exl3_cuda_layer_preload "
                    f"layer_id={layer} experts=256 source_experts=64 cooperative=true "
                    "packed_exchange=true source_bytes=909116160 source_requests=768 "
                    "source_spans=1 direct_io=true source_gbps=12.000 load_ms=75.000 "
                    "pack_ms=65.000 allocation_ms=20.000 upload_ms=35.000 "
                    "exchange_ms=40.000 resident_bytes=916194304"
                )
            else:
                lines.append(
                    "real_exl3_cuda_layer_preload "
                    f"layer_id={layer} experts=256 cooperative=false direct_resident=true "
                    f"source_bytes={TOOL.EXL3_DIRECT_SOURCE_BYTES_PER_RANK_LAYER} "
                    "source_gbps=2.900 allocation_ms=37.000 direct_ms=321.000 "
                    f"resident_bytes={resident_bytes}"
                )
    lines.extend(
        [
            "expertd_startup_phase stage=resident-preload elapsed_ms=30000.000 total_ms=30005.000",
            "expertd_real_weight_resident_preload "
            "projection_groups=57600 layers=75 experts=19200 weight_bytes=0 "
            "quant_metadata_bytes=0 route_cache_entries=0 route_cache_loads=0 "
            "route_cache_hits=0 projection_row_entries=57600 projection_row_loads=0 "
            "projection_row_hits=0 cuda_reference_enabled=true "
            "cuda_projection_groups=57600 cuda_weight_bytes=68714572800 "
            "cuda_weight_scale_bytes=0 cuda_projection_entries=57600 "
            "cuda_projection_uploads=57600 cuda_cache_hits=0",
            "expertd_startup_phase stage=service-handoff elapsed_ms=1.000 total_ms=30006.000",
        ]
    )
    if exit_status is not None:
        lines.append(
            f"== 2026-08-23T01:00:00+00:00 expert-9100 exited status={exit_status} =="
        )
    return "\n".join(lines) + "\n"


def four_logs(
    tmp_path: Path,
    *,
    exl3: bool,
    missing_layer: int | None = None,
    exit_status: int | None = None,
    cooperative: bool = False,
    resident_bytes: int = TOOL.EXL3_RESIDENT_BYTES_PER_RANK_LAYER,
    runtime_fingerprint: str = RUNTIME_FINGERPRINT,
):
    logs = []
    for index, host in enumerate(("ostrich", "dodo", "emu", "kiwi")):
        path = tmp_path / f"{host}.log"
        path.write_text(
            log_text(
                exl3=exl3,
                missing_layer=missing_layer if index == 0 else None,
                exit_status=exit_status,
                cooperative=cooperative,
                resident_bytes=resident_bytes,
                role=f"spark-{index}",
                runtime_fingerprint=runtime_fingerprint,
            ),
            encoding="utf-8",
        )
        logs.append((host, path))
    return logs


def test_accepts_complete_four_host_direct_exl3_startup(tmp_path: Path) -> None:
    report = TOOL.analyze(
        model=TOOL.EXL3_MODEL,
        weight_format="exl3",
        cache_state="cold",
        include_mtp=False,
        expert_runtime_fingerprint=RUNTIME_FINGERPRINT,
        logs=four_logs(tmp_path, exl3=True),
    )

    assert report["status"] == "accepted"
    assert report["summary"]["maximum_resident_preload_ms"] == 30000.0
    assert report["hosts"][0]["log"]["selected_start_line"] == 3
    assert report["hosts"][0]["process"]["model_id"] == TOOL.EXL3_MODEL
    assert report["hosts"][0]["exl3"]["layers"] == 75
    assert report["preload_mode"] == "direct-resident"
    assert report["hosts"][0]["exl3"]["preload_mode"] == "direct-resident"
    assert report["hosts"][0]["resident"]["projection_groups"] == 57600


def test_accepts_coalesced_cooperative_exl3_startup(tmp_path: Path) -> None:
    report = TOOL.analyze(
        model=TOOL.EXL3_MODEL,
        weight_format="exl3",
        cache_state="cold",
        include_mtp=False,
        expert_runtime_fingerprint=RUNTIME_FINGERPRINT,
        logs=four_logs(tmp_path, exl3=True, cooperative=True),
    )

    assert report["preload_mode"] == "cooperative-coalesced"
    assert report["hosts"][0]["exl3"]["source_requests"] == 75 * 768
    assert report["hosts"][0]["exl3"]["pack_ms"] == 75 * 65.0


def test_rejects_wrong_direct_resident_geometry(tmp_path: Path) -> None:
    with pytest.raises(TOOL.StartupError, match="resident geometry"):
        TOOL.analyze(
            model=TOOL.EXL3_MODEL,
            weight_format="exl3",
            cache_state="cold",
            include_mtp=False,
            expert_runtime_fingerprint=RUNTIME_FINGERPRINT,
            logs=four_logs(tmp_path, exl3=True, resident_bytes=123),
        )


def test_rejects_incomplete_exl3_layer_coverage(tmp_path: Path) -> None:
    with pytest.raises(TOOL.StartupError, match="layer coverage"):
        TOOL.analyze(
            model=TOOL.EXL3_MODEL,
            weight_format="exl3",
            cache_state="cold",
            include_mtp=False,
            expert_runtime_fingerprint=RUNTIME_FINGERPRINT,
            logs=four_logs(tmp_path, exl3=True, missing_layer=37),
        )


def test_accepts_nvfp4_startup_without_exl3_layer_lines(tmp_path: Path) -> None:
    report = TOOL.analyze(
        model="lukealonso/GLM-5.2-NVFP4",
        weight_format="nvfp4",
        cache_state="cold",
        include_mtp=False,
        expert_runtime_fingerprint=RUNTIME_FINGERPRINT,
        logs=four_logs(tmp_path, exl3=False),
    )

    assert all(host["exl3"] is None for host in report["hosts"])


def test_accepts_orderly_container_stop_after_complete_startup(tmp_path: Path) -> None:
    report = TOOL.analyze(
        model=TOOL.EXL3_MODEL,
        weight_format="exl3",
        cache_state="warm",
        include_mtp=False,
        expert_runtime_fingerprint=RUNTIME_FINGERPRINT,
        logs=four_logs(tmp_path, exl3=True, exit_status=143),
    )

    assert all(host["process"]["exit_status"] == 143 for host in report["hosts"])


def test_rejects_failed_process_after_startup(tmp_path: Path) -> None:
    with pytest.raises(TOOL.StartupError, match="exited unsuccessfully"):
        TOOL.analyze(
            model=TOOL.EXL3_MODEL,
            weight_format="exl3",
            cache_state="warm",
            include_mtp=False,
            expert_runtime_fingerprint=RUNTIME_FINGERPRINT,
            logs=four_logs(tmp_path, exl3=True, exit_status=1),
        )


def test_rejects_model_label_that_differs_from_launched_process(tmp_path: Path) -> None:
    with pytest.raises(TOOL.StartupError, match="launched model"):
        TOOL.analyze(
            model=TOOL.EXL3_MODEL,
            weight_format="exl3",
            cache_state="cold",
            include_mtp=False,
            expert_runtime_fingerprint=RUNTIME_FINGERPRINT,
            logs=four_logs(tmp_path, exl3=False),
        )


def test_rejects_logs_from_another_expert_runtime(tmp_path: Path) -> None:
    with pytest.raises(TOOL.StartupError, match="runtime identity differs"):
        TOOL.analyze(
            model=TOOL.EXL3_MODEL,
            weight_format="exl3",
            cache_state="cold",
            include_mtp=False,
            expert_runtime_fingerprint=RUNTIME_FINGERPRINT,
            logs=four_logs(tmp_path, exl3=True, runtime_fingerprint="b" * 64),
        )
