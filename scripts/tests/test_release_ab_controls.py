from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RELEASE_COMMON = ROOT / "scripts" / "release-common.sh"

BASE_CONFIG = """\
PROFILE=balanced
MODEL=luke
SPECULATION=dspark
SPARK_0_HOST=ostrich
SPARK_1_HOST=dodo
SPARK_2_HOST=emu
SPARK_3_HOST=kiwi
SPARK_0_LANE_A=10.55.0.1
SPARK_1_LANE_A=10.55.0.2
SPARK_2_LANE_A=10.55.0.3
SPARK_3_LANE_A=10.55.0.4
"""


def load_config(tmp_path: Path, extra: str = "") -> subprocess.CompletedProcess[str]:
    config = tmp_path / "glmrt.config"
    config.write_text(BASE_CONFIG + extra, encoding="utf-8")
    return subprocess.run(
        [
            "bash",
            "-c",
            (
                'source "$1"; release_load_config "$2"; '
                'printf "%s\\n%s\\n" '
                '"$SPARKINFER_GLM_H64_QUERY_PROJECTION" '
                '"${DSPARK_FIXED_DRAFTS-unset}"'
            ),
            "bash",
            str(RELEASE_COMMON),
            str(config),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_release_ab_controls_have_safe_defaults(tmp_path: Path) -> None:
    result = load_config(tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["auto", ""]


def test_release_ab_controls_accept_explicit_qualification_values(
    tmp_path: Path,
) -> None:
    result = load_config(
        tmp_path,
        "SPARKINFER_GLM_H64_QUERY_PROJECTION=disable\nDSPARK_FIXED_DRAFTS=7\n",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["disable", "7"]


def test_fixed_dspark_depth_rejects_non_dspark_profile(tmp_path: Path) -> None:
    result = load_config(
        tmp_path,
        "SPECULATION=plain\nDSPARK_FIXED_DRAFTS=2\n",
    )

    assert result.returncode == 2
    assert "DSPARK_FIXED_DRAFTS requires SPECULATION=dspark" in result.stderr


def test_release_ab_controls_reject_invalid_values(tmp_path: Path) -> None:
    invalid_h64 = load_config(
        tmp_path,
        "SPARKINFER_GLM_H64_QUERY_PROJECTION=enabled\n",
    )
    assert invalid_h64.returncode == 2
    assert (
        "SPARKINFER_GLM_H64_QUERY_PROJECTION must be auto, disable, or force"
        in invalid_h64.stderr
    )

    invalid_depth = load_config(tmp_path, "DSPARK_FIXED_DRAFTS=8\n")
    assert invalid_depth.returncode == 2
    assert "DSPARK_FIXED_DRAFTS must be empty or in 0..7" in invalid_depth.stderr


def test_run_fingerprints_and_explicitly_sets_both_ab_controls() -> None:
    launcher = (ROOT / "run.sh").read_text(encoding="utf-8")
    fingerprint = launcher.split(
        'deployment_fingerprint="$(', maxsplit=1
    )[1].split("check_model_cache_local()", maxsplit=1)[0]
    env_file = launcher.split('env_file="$state_dir/coordinator.env"', maxsplit=1)[
        1
    ].split("mkdir -p", maxsplit=1)[0]

    assert '"$SPARKINFER_GLM_H64_QUERY_PROJECTION"' in fingerprint
    assert '"$DSPARK_FIXED_DRAFTS"' in fingerprint
    assert (
        "GLMRT_SPARKINFER_GLM_H64_BF16_QUERY_PROJECTION="
        "$SPARKINFER_GLM_H64_QUERY_PROJECTION"
    ) in env_file
    assert (
        "GLMRT_REAL_FULL_DSPARK_FIXED_DRAFTS=$DSPARK_FIXED_DRAFTS"
        in env_file
    )
    assert "GLMRT_SPARKINFER_COMMIT=$coordinator_sparkinfer_commit" in env_file
    assert (
        "GLMRT_COORDINATOR_POWER_LIMIT_WATTS=$coordinator_power_limit_watts"
        in env_file
    )


def test_release_and_wip_containers_never_auto_start_on_boot() -> None:
    release_launcher = (ROOT / "run.sh").read_text(encoding="utf-8")
    wip_builder = (ROOT / "wip.sh").read_text(encoding="utf-8")

    assert "--restart unless-stopped" not in release_launcher
    assert "--restart unless-stopped" not in wip_builder
    assert release_launcher.count("--restart no") == 1
    assert wip_builder.count("--restart no") == 2


def test_parallel_wip_builds_fail_before_finalizing_stale_outputs() -> None:
    wip_builder = (ROOT / "wip.sh").read_text(encoding="utf-8")

    assert wip_builder.count("|| return $?") == 2
    assert (
        "/wip/source coordinator 120 /wip/build/coordinator "
        "/wip/output/coordinator \\\n    || return $?"
    ) in wip_builder
    assert (
        "/wip/source expert 121 /wip/build/expert /wip/output/expert \\\n    || return $?"
    ) in wip_builder
