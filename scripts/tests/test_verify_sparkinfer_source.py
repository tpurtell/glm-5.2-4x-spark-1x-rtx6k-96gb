from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "verify-sparkinfer-source.py"
BOOTSTRAP = Path(__file__).parents[2] / "python" / "tools" / "_pinned_sparkinfer.py"
SPEC = importlib.util.spec_from_file_location("verify_sparkinfer_source", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def make_source(root: Path) -> Path:
    source = root / "sparkinfer"
    package = source / "b12x"
    package.mkdir(parents=True)
    (source / "LICENSE").write_text("Apache-2.0 fixture\n", encoding="utf-8")
    (source / "pyproject.toml").write_text(
        '[project]\nname = "b12x"\nversion = "1.1.0"\n',
        encoding="utf-8",
    )
    (package / "__init__.py").write_text(
        '__version__ = "1.1.0"\n', encoding="utf-8"
    )
    (package / "kernel.py").write_text("VALUE = 1\n", encoding="utf-8")
    return source


def write_lock(
    source: Path,
    path: Path,
    *,
    revision: str = "1" * 40,
    repository: str = "https://github.com/tpurtell/sparkinfer-glmrt",
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "repository": repository,
                "revision": revision,
                "source_tree_sha256": VERIFIER.source_tree_sha256(source),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def git(source: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", os.fspath(source), *args],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def test_run_git_trusts_only_the_exact_resolved_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "parent" / ".." / "sparkinfer"
    resolved = source.resolve()
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="revision\n", stderr="")

    monkeypatch.setattr(VERIFIER.subprocess, "run", fake_run)

    assert VERIFIER._run_git(source, "rev-parse", "HEAD") == "revision"
    assert calls == [
        [
            "git",
            "-c",
            f"safe.directory={resolved}",
            "-C",
            os.fspath(resolved),
            "rev-parse",
            "HEAD",
        ]
    ]


def test_metadata_free_tree_matches_lock_and_rejects_mutation(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    lock = tmp_path / "sparkinfer.lock.json"
    write_lock(source, lock)

    verified = VERIFIER.verify(source, lock)
    assert verified["revision"] == "1" * 40

    (source / "b12x" / "kernel.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )
    with pytest.raises(VERIFIER.VerificationError, match="does not match"):
        VERIFIER.verify(source, lock)


def test_native_extension_cannot_hide_from_archive_digest(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    before = VERIFIER.source_tree_sha256(source)
    (source / "b12x" / "kernel.so").write_bytes(b"untrusted extension")
    assert VERIFIER.source_tree_sha256(source) != before


def test_generated_directory_names_are_ignored_only_at_the_root(
    tmp_path: Path,
) -> None:
    source = make_source(tmp_path)
    before = VERIFIER.source_tree_sha256(source)
    nested = source / "b12x" / "build"
    nested.mkdir()
    (nested / "kernel.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert VERIFIER.source_tree_sha256(source) != before


def test_generated_cache_directories_do_not_change_digest(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    before = VERIFIER.source_tree_sha256(source)
    cache = source / "b12x" / "__pycache__"
    cache.mkdir()
    (cache / "kernel.cpython-312.pyc").write_bytes(b"generated")
    egg_info = source / "sparkinfer.egg-info"
    egg_info.mkdir()
    (egg_info / "PKG-INFO").write_text("generated\n", encoding="utf-8")
    assert VERIFIER.source_tree_sha256(source) == before


@pytest.mark.parametrize(
    "relative",
    (
        Path("b12x/__pycache__/kernel.cpython-312.pyc"),
        Path("b12x/.mypy_cache/kernel.json"),
        Path("b12x/.pytest_cache/nodeids"),
        Path("b12x/.ruff_cache/cache"),
        Path("b12x/kernel.pyo"),
    ),
)
def test_metadata_free_source_rejects_python_caches(
    tmp_path: Path, relative: Path
) -> None:
    source = make_source(tmp_path)
    path = source / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"generated")

    with pytest.raises(
        VERIFIER.VerificationError,
        match="metadata-free source contains a Python cache",
    ):
        VERIFIER.require_no_python_cache(source)


def test_digest_tracks_executable_mode_and_symlink_target(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    kernel = source / "b12x" / "kernel.py"
    link = source / "selected.py"
    link.symlink_to("b12x/kernel.py")
    initial = VERIFIER.source_tree_sha256(source)

    kernel.chmod(0o755)
    executable = VERIFIER.source_tree_sha256(source)
    assert executable != initial

    link.unlink()
    link.symlink_to("b12x/__init__.py")
    assert VERIFIER.source_tree_sha256(source) != executable


def test_source_symlink_cannot_escape_archive_root(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    (source / "outside.py").symlink_to("../outside.py")
    with pytest.raises(VERIFIER.VerificationError, match="escapes"):
        VERIFIER.source_tree_sha256(source)


def test_git_checkout_requires_locked_revision_cleanliness_and_origin(
    tmp_path: Path,
) -> None:
    source = make_source(tmp_path)
    git(source, "init", "-q")
    git(source, "add", ".")
    git(
        source,
        "-c",
        "user.name=GLMRT test",
        "-c",
        "user.email=glmrt-test@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    git(
        source,
        "remote",
        "add",
        "origin",
        "git@github.com:tpurtell/sparkinfer-glmrt.git",
    )
    revision = git(source, "rev-parse", "HEAD")
    lock = tmp_path / "sparkinfer.lock.json"
    write_lock(source, lock, revision=revision)
    VERIFIER.verify(source, lock)

    nested_gitfile = source / "b12x" / "nested" / ".git"
    nested_gitfile.parent.mkdir()
    nested_gitfile.write_text(
        "gitdir: /host-only/submodule/metadata\n", encoding="utf-8"
    )
    archive = tmp_path / "archive"
    archive.mkdir()
    tar_payload = subprocess.run(
        [
            "tar",
            "-C",
            os.fspath(source),
            "--exclude=.git",
            "--exclude=*/.git",
            "-cf",
            "-",
            ".",
        ],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    subprocess.run(
        ["tar", "-C", os.fspath(archive), "-xf", "-"],
        check=True,
        input=tar_payload,
    )
    assert not (archive / ".git").exists()
    assert not (archive / "b12x" / "nested" / ".git").exists()
    VERIFIER.verify(archive, lock)
    nested_gitfile.unlink()
    nested_gitfile.parent.rmdir()

    cache = source / "b12x" / "__pycache__"
    cache.mkdir()
    (cache / "untracked.pyc").write_bytes(b"dirty")
    with pytest.raises(VERIFIER.VerificationError, match="source changes"):
        VERIFIER.verify(source, lock)
    (cache / "untracked.pyc").unlink()
    cache.rmdir()

    git(source, "remote", "set-url", "origin", "https://example.invalid/fork.git")
    with pytest.raises(VERIFIER.VerificationError, match="origin"):
        VERIFIER.verify(source, lock)


def test_invalid_package_metadata_is_reported_without_a_traceback(
    tmp_path: Path,
) -> None:
    source = make_source(tmp_path)
    (source / "pyproject.toml").write_text("[project\n", encoding="utf-8")
    lock = tmp_path / "sparkinfer.lock.json"
    write_lock(source, lock)
    with pytest.raises(VERIFIER.VerificationError, match="package metadata"):
        VERIFIER.verify(source, lock)


def test_cli_can_require_import_from_verified_source(tmp_path: Path) -> None:
    source = make_source(tmp_path / "verified")
    lock = tmp_path / "sparkinfer.lock.json"
    write_lock(source, lock)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.fspath(source)

    result = subprocess.run(
        [
            os.fspath(SCRIPT),
            "--source",
            os.fspath(source),
            "--lock",
            os.fspath(lock),
            "--assert-import-source",
        ],
        check=False,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"import={source / 'b12x' / '__init__.py'}" in result.stdout


def test_cli_can_require_cache_free_metadata_source(tmp_path: Path) -> None:
    source = make_source(tmp_path / "verified")
    lock = tmp_path / "sparkinfer.lock.json"
    write_lock(source, lock)
    cache = source / "b12x" / "__pycache__"
    cache.mkdir()
    (cache / "kernel.cpython-312.pyc").write_bytes(b"ignored bytecode")

    normal = subprocess.run(
        [
            os.fspath(SCRIPT),
            "--source",
            os.fspath(source),
            "--lock",
            os.fspath(lock),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    guarded = subprocess.run(
        [
            os.fspath(SCRIPT),
            "--source",
            os.fspath(source),
            "--lock",
            os.fspath(lock),
            "--require-no-python-cache",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert normal.returncode == 0, normal.stderr
    assert guarded.returncode == 2
    assert "metadata-free source contains a Python cache" in guarded.stderr


def test_cli_rejects_import_outside_verified_source(tmp_path: Path) -> None:
    source = make_source(tmp_path / "verified")
    other_source = make_source(tmp_path / "other")
    lock = tmp_path / "sparkinfer.lock.json"
    write_lock(source, lock)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.fspath(other_source)

    result = subprocess.run(
        [
            os.fspath(SCRIPT),
            "--source",
            os.fspath(source),
            "--lock",
            os.fspath(lock),
            "--assert-import-source",
        ],
        check=False,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 2
    assert "resolves outside the verified source tree" in result.stderr


def test_standalone_bootstrap_honors_explicit_verified_source(
    tmp_path: Path,
) -> None:
    source = make_source(tmp_path / "verified")
    lock = tmp_path / "sparkinfer.lock.json"
    write_lock(source, lock)
    env = os.environ.copy()
    env["GLMRT_SPARKINFER_SOURCE_DIR"] = os.fspath(source)
    env["GLMRT_SPARKINFER_LOCK_FILE"] = os.fspath(lock)
    env["PYTHONPATH"] = os.fspath(BOOTSTRAP.parent)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import _pinned_sparkinfer as pinned; "
                "print(pinned.SOURCE); print(pinned.LOCK); "
                "print(pinned.IMPORTED_MODULE); print(pinned.REVISION)"
            ),
        ],
        check=False,
        cwd=BOOTSTRAP.parents[2],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    reported_source, reported_lock, imported_module, revision = (
        result.stdout.strip().splitlines()
    )
    assert Path(reported_source) == source.resolve()
    assert Path(reported_lock) == lock.resolve()
    assert Path(imported_module).is_relative_to(source.resolve())
    assert revision == "1" * 40
