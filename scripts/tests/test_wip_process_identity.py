from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WIP_PROCESS = ROOT / "scripts" / "wip-process.sh"
FINGERPRINT = "a" * 64


def wait_for(path: Path, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {path}")


def invoke(runtime: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(WIP_PROCESS), *args],
        env={**os.environ, "GLMRT_WIP_RUNTIME_ROOT": str(runtime)},
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_identity_is_bound_to_live_pid_and_start_time(tmp_path: Path) -> None:
    runtime = tmp_path / "run"
    process = subprocess.Popen(
        [str(WIP_PROCESS), "run", "test-process", "sleep", "60"],
        env={**os.environ, "GLMRT_WIP_RUNTIME_ROOT": str(runtime)},
    )
    try:
        wait_for(runtime / "test-process.pid")
        bound = invoke(runtime, "bind-identity", "test-process", FINGERPRINT)
        assert bound.returncode == 0, bound.stderr

        identity = invoke(runtime, "identity", "test-process")
        assert identity.returncode == 0, identity.stderr
        assert identity.stdout.strip() == FINGERPRINT

        identity_file = runtime / "test-process.identity"
        contents = identity_file.read_text(encoding="utf-8")
        identity_file.write_text(
            contents.replace("start_ticks=", "start_ticks=0#"), encoding="utf-8"
        )
        stale = invoke(runtime, "identity", "test-process")
        assert stale.returncode != 0
        assert stale.stdout == ""
    finally:
        os.kill(process.pid, signal.SIGTERM)
        process.wait(timeout=5)

    assert not (runtime / "test-process.pid").exists()
    assert not (runtime / "test-process.identity").exists()


def test_bind_identity_rejects_non_sha256_value(tmp_path: Path) -> None:
    result = invoke(tmp_path / "run", "bind-identity", "test-process", "not-a-hash")

    assert result.returncode == 2
    assert "invalid WIP process fingerprint" in result.stderr


def test_wip_launcher_has_separate_expert_and_deployment_identities() -> None:
    launcher = (ROOT / "scripts" / "run-wip.sh").read_text(encoding="utf-8")

    assert "wip-expert-runtime-identity.py" in launcher
    assert "glmrt-wip-deployment-v2" in launcher
    assert 'GLMRT_RELEASE_CONFIG_SHA256="$expert_runtime_fingerprint"' in launcher
    assert "bind-identity" in launcher
    assert "'$expert_process' '$expert_runtime_fingerprint'" in launcher
    assert "reusing four fingerprint-matched resident WIP Spark experts" in launcher


def test_wip_builder_streams_every_local_heredoc_into_docker() -> None:
    builder = (ROOT / "wip.sh").read_text(encoding="utf-8")
    local_heredocs = re.findall(
        r'^\s*docker exec (?P<options>.*?)"\$coordinator_container" bash -s .*<<',
        builder,
        flags=re.MULTILINE,
    )

    assert len(local_heredocs) == 4
    assert all("-i" in options.split() for options in local_heredocs)
