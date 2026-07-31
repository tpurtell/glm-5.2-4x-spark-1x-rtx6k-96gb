from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts" / "sparkinfer-release-provenance.py"
SPEC = importlib.util.spec_from_file_location("sparkinfer_release_provenance", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PROVENANCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROVENANCE)


def make_release_fixture(
    root: Path,
    *,
    repository: str = "https://example.invalid/owner/sparkinfer-fork.git",
    revision: str = "2" * 40,
) -> tuple[Path, Path, Path, Path]:
    source = root / "sparkinfer"
    package = source / "sparkinfer"
    package.mkdir(parents=True)
    license_text = """Apache License
Version 2.0, January 2004
TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
fixture remainder
"""
    (source / "LICENSE").write_text(license_text, encoding="utf-8")
    (source / "pyproject.toml").write_text(
        """[project]
name = "sparkinfer"
version = "9.9.9"
license = "Apache-2.0"
license-files = ["LICENSE"]
""",
        encoding="utf-8",
    )
    (package / "__init__.py").write_text(
        '__version__ = "9.9.9"\n', encoding="utf-8"
    )
    (package / "kernel.py").write_text("VALUE = 1\n", encoding="utf-8")
    lock = root / "sparkinfer.lock.json"
    lock.write_text(
        json.dumps(
            {
                "schema": 1,
                "repository": repository,
                "revision": revision,
                "source_tree_sha256": PROVENANCE.VERIFIER.source_tree_sha256(source),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    distributed_license = root / "SPARKINFER_LICENSE"
    distributed_license.write_bytes((source / "LICENSE").read_bytes())
    repository_for_notice = repository.removesuffix(".git")
    notices = root / "THIRD_PARTY_NOTICES.md"
    notices.write_text(
        f"""# Third-Party Notices

## SparkInfer

Source: <{repository_for_notice}>

SPDX-License-Identifier: Apache-2.0

Release records: SPARKINFER_LICENSE, SPARKINFER_PROVENANCE.json, and
SPARKINFER_SHA256SUMS.
""",
        encoding="utf-8",
    )
    return source, lock, distributed_license, notices


def test_manifest_records_generic_locked_source_and_material_hashes(
    tmp_path: Path,
) -> None:
    repository = "https://code.example.test/team/custom-sparkinfer.git"
    revision = "a" * 40
    source, lock, license_path, notices = make_release_fixture(
        tmp_path,
        repository=repository,
        revision=revision,
    )

    manifest = PROVENANCE.expected_provenance(
        source, lock, license_path, notices
    )

    assert manifest["repository"] == repository
    assert manifest["revision"] == revision
    assert manifest["source_tree_sha256"] == json.loads(
        lock.read_text(encoding="utf-8")
    )["source_tree_sha256"]
    assert manifest["license"] == {
        "spdx": "Apache-2.0",
        "artifact": "SPARKINFER_LICENSE",
        "sha256": PROVENANCE.file_sha256(license_path),
    }
    assert manifest["notices"] == {
        "artifact": "THIRD_PARTY_NOTICES.md",
        "sha256": PROVENANCE.file_sha256(notices),
    }


def test_cli_writes_and_verifies_canonical_provenance(tmp_path: Path) -> None:
    source, lock, license_path, notices = make_release_fixture(tmp_path)
    output = tmp_path / "SPARKINFER_PROVENANCE.json"
    common = [
        sys.executable,
        str(SCRIPT),
        "--source",
        str(source),
        "--lock",
        str(lock),
        "--license",
        str(license_path),
        "--notices",
        str(notices),
    ]

    written = subprocess.run(
        [*common, "--write", str(output)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    verified = subprocess.run(
        [*common, "--verify", str(output)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert written.returncode == 0, written.stderr
    assert verified.returncode == 0, verified.stderr
    assert output.read_text(encoding="utf-8").endswith("\n")


def test_changed_license_or_notices_are_rejected(tmp_path: Path) -> None:
    source, lock, license_path, notices = make_release_fixture(tmp_path)

    license_path.write_text("different license\n", encoding="utf-8")
    with pytest.raises(
        PROVENANCE.ReleaseProvenanceError,
        match="differs from the verified source license",
    ):
        PROVENANCE.expected_provenance(source, lock, license_path, notices)

    license_path.write_bytes((source / "LICENSE").read_bytes())
    notices.write_text(
        notices.read_text(encoding="utf-8").replace(
            "https://example.invalid/owner/sparkinfer-fork",
            "https://example.invalid/wrong/fork",
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        PROVENANCE.ReleaseProvenanceError,
        match="omit SparkInfer release material",
    ):
        PROVENANCE.expected_provenance(source, lock, license_path, notices)


def test_release_builds_materialize_and_verify_all_records() -> None:
    artifact_builder = (
        ROOT / "scripts" / "build-release-artifacts.sh"
    ).read_text(encoding="utf-8")
    release_builder = (ROOT / "build.sh").read_text(encoding="utf-8")
    image = (ROOT / "docker" / "Dockerfile.release").read_text(encoding="utf-8")

    assert "sparkinfer-release-provenance.py" in artifact_builder
    assert '--write "$output_dir/SPARKINFER_PROVENANCE.json"' in artifact_builder
    assert "SPARKINFER_SHA256SUMS" in artifact_builder
    assert "sha256sum -c SPARKINFER_SHA256SUMS" in artifact_builder

    assert "sparkinfer-release-provenance.py" in image
    assert "--verify /opt/glmrt/share/SPARKINFER_PROVENANCE.json" in image
    assert "sha256sum -c SPARKINFER_SHA256SUMS" in image
    assert "spark-moe-mode-common.sh" not in image

    assert release_builder.count(
        ':/opt/glmrt/share/SPARKINFER_SHA256SUMS"'
    ) == 2
    assert '--verify "$repo_root/dist/$role/SPARKINFER_PROVENANCE.json"' in (
        release_builder
    )
    assert "coordinator/SPARKINFER_SHA256SUMS" in release_builder
    assert "spark-expert/SPARKINFER_SHA256SUMS" in release_builder
