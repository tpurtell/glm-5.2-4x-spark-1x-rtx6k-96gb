"""Make standalone tools use GLMRT's verified SparkInfer source tree."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tomllib
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(
    os.environ.get(
        "GLMRT_SPARKINFER_SOURCE_DIR",
        ROOT / "third_party" / "sparkinfer",
    )
).expanduser().resolve()
LOCK = Path(
    os.environ.get(
        "GLMRT_SPARKINFER_LOCK_FILE",
        ROOT / "third_party" / "sparkinfer.lock.json",
    )
).expanduser().resolve()
VERIFIER_PATH = ROOT / "scripts" / "verify-sparkinfer-source.py"


def _load_verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_glmrt_verify_sparkinfer_source",
        VERIFIER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load SparkInfer source verifier: {VERIFIER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepend_source() -> None:
    source = os.fspath(SOURCE)
    retained_paths: list[str] = []
    for entry in sys.path:
        if entry and Path(entry).resolve() == SOURCE:
            continue
        retained_paths.append(entry)
    sys.path[:] = [source, *retained_paths]


try:
    _verifier = _load_verifier()
    LOCK_DATA = _verifier.verify(SOURCE, LOCK)
    _prepend_source()
    IMPORTED_MODULE = _verifier.verify_import_source(SOURCE)
except Exception as exc:
    raise RuntimeError(
        "standalone SparkInfer tools require the verified "
        f"{SOURCE} tree locked by {LOCK}: {exc}"
    ) from exc

REVISION = str(LOCK_DATA["revision"])
VERSION = str(
    tomllib.loads((SOURCE / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
)
