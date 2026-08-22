from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "kernel-cache-identity.py"
ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location("kernel_cache_identity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
IDENTITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IDENTITY)


def fixture_payload(source: Path, environment_id: str = "sha256:image") -> dict:
    return {
        "schema": IDENTITY.SCHEMA,
        "environment_id": environment_id,
        "flashinfer": {
            "root": str(source.resolve()),
            "source_tree_sha256": IDENTITY.tree_sha256(source),
        },
    }


def test_tree_digest_tracks_source_and_ignores_python_cache(tmp_path: Path) -> None:
    source = tmp_path / "flashinfer"
    source.mkdir()
    kernel = source / "kernel.py"
    kernel.write_text("VALUE = 1\n", encoding="utf-8")
    initial = IDENTITY.tree_sha256(source)

    pycache = source / "__pycache__"
    pycache.mkdir()
    (pycache / "kernel.cpython-312.pyc").write_bytes(b"generated")
    assert IDENTITY.tree_sha256(source) == initial

    kernel.write_text("VALUE = 2\n", encoding="utf-8")
    assert IDENTITY.tree_sha256(source) != initial


def test_identity_changes_with_environment_or_source(tmp_path: Path) -> None:
    source = tmp_path / "flashinfer"
    source.mkdir()
    kernel = source / "kernel.py"
    kernel.write_text("VALUE = 1\n", encoding="utf-8")
    first = IDENTITY.identity_key(fixture_payload(source))
    assert first != IDENTITY.identity_key(fixture_payload(source, "sha256:new-image"))
    kernel.write_text("VALUE = 2\n", encoding="utf-8")
    assert first != IDENTITY.identity_key(fixture_payload(source))


def test_materialize_is_deterministic_and_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "flashinfer"
    source.mkdir()
    (source / "kernel.py").write_text("VALUE = 1\n", encoding="utf-8")
    payload = fixture_payload(source)
    key = IDENTITY.identity_key(payload)
    first = IDENTITY.materialize(tmp_path / "cache", payload, key)
    assert IDENTITY.materialize(tmp_path / "cache", payload, key) == first
    manifest_path = first / "IDENTITY.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["identity"] == key

    manifest["payload"]["environment_id"] = "corrupt"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(IDENTITY.CacheIdentityError, match="manifest mismatch"):
        IDENTITY.materialize(tmp_path / "cache", payload, key)


def test_missing_environment_identity_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "flashinfer"
    source.mkdir()
    with pytest.raises(IDENTITY.CacheIdentityError, match="missing or invalid"):
        IDENTITY.identity_payload(source, "")


def test_release_image_packages_cache_identity_helper() -> None:
    dockerfile = (ROOT / "docker" / "Dockerfile.release").read_text(encoding="utf-8")
    assert (
        "COPY scripts/kernel-cache-identity.py "
        "/opt/glmrt/scripts/kernel-cache-identity.py"
    ) in dockerfile


def test_b12x_compile_cache_uses_current_environment_contract() -> None:
    launcher = (ROOT / "scripts" / "real-full-tcp-serve.sh").read_text(
        encoding="utf-8"
    )
    assert "export B12X_COMPILE_CACHE_DIR=" in launcher
    assert "export SPARKINFER_COMPILE_CACHE_DIR=" not in launcher

    exporters = (
        "export_b12x_coordinator_aot.py",
        "export_b12x_spark_moe_aot.py",
        "export_b12x_spark_w4a16_m1_parity_aot.py",
        "export_w8a16_packed_o_aot.py",
    )
    for name in exporters:
        source = (ROOT / "python" / "tools" / name).read_text(encoding="utf-8")
        assert 'B12X_COMPILE_DISK_CACHE"] = "0"' in source
        assert 'B12X_COMPILE_MEMORY_CACHE"] = "0"' in source
        assert "SPARKINFER_COMPILE_" not in source


def test_remote_release_launch_preserves_empty_optional_arguments() -> None:
    launcher = (ROOT / "scripts" / "phase0-spark-tcp-bench.sh").read_text(
        encoding="utf-8"
    )
    assert 'existing_container_arg="${existing_container:-__unset__}"' in launcher
    assert 'runtime_cache_dir_arg="${runtime_cache_dir:-__unset__}"' in launcher
    assert '"$existing_container_arg" "$prebuilt_bin"' in launcher
    assert '"$prebuilt_native_lib" "$runtime_cache_dir_arg"' in launcher
    assert 'if [ "$existing_container" = "__unset__" ]; then' in launcher
    assert 'if [ "$runtime_cache_dir" = "__unset__" ]; then' in launcher
