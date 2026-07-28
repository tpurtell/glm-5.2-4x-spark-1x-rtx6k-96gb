#!/usr/bin/env python3
"""Probe whether the native library can build and test with RDMA enabled."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


MISSING_IBVERBS_MESSAGE = "GLMRT_ENABLE_RDMA=ON requires libibverbs headers and library"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_command(command: list[str], cwd: Path) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "output": completed.stdout,
    }


def classify_status(steps: list[dict[str, Any]]) -> str:
    configure = steps[0]
    if configure["returncode"] != 0:
        if MISSING_IBVERBS_MESSAGE in configure["output"]:
            return "configure_failed_missing_libibverbs"
        return "configure_failed"
    if len(steps) > 1 and steps[1]["returncode"] != 0:
        return "build_failed"
    if len(steps) > 2 and steps[2]["returncode"] != 0:
        return "test_failed"
    return "configured_built_tested"


def build_status(args: argparse.Namespace) -> dict[str, Any]:
    root = repo_root()
    build_dir = args.build_dir if args.build_dir.is_absolute() else root / args.build_dir
    if args.clean and build_dir.exists():
        shutil.rmtree(build_dir)

    steps: list[dict[str, Any]] = []
    configure = [
        "cmake",
        "-S",
        "native",
        "-B",
        str(build_dir),
        "-G",
        args.generator,
        "-DGLMRT_ENABLE_CUDA=OFF",
        "-DGLMRT_ENABLE_RDMA=ON",
    ]
    steps.append(run_command(configure, root))
    if steps[-1]["returncode"] == 0:
        steps.append(run_command(["cmake", "--build", str(build_dir)], root))
    if len(steps) > 1 and steps[-1]["returncode"] == 0:
        steps.append(
            run_command(
                ["ctest", "--test-dir", str(build_dir), "--output-on-failure"],
                root,
            )
        )

    status = classify_status(steps)
    passed = status == "configured_built_tested"
    if passed:
        next_required_step = (
            "Use this RDMA-enabled native build for verbs-host app transport "
            "preflight and benchmarks."
        )
    else:
        next_required_step = (
            "Install libibverbs development headers/library and expose an RDMA device, "
            "then rerun this probe before claiming verbs-host app transport support."
        )
    return {
        "format_version": 1,
        "status": status,
        "rdma_enabled_requested": True,
        "cuda_enabled_requested": False,
        "passed": passed,
        "build_dir": str(build_dir),
        "source_dir": "native",
        "generator": args.generator,
        "missing_libibverbs": status == "configure_failed_missing_libibverbs",
        "commands": steps,
        "next_required_step": next_required_step,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("/tmp/glmrt-native-rdma-build"))
    parser.add_argument("--generator", default="Ninja")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()

    status = build_status(args)
    output = args.output if args.output.is_absolute() else repo_root() / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output.relative_to(repo_root())}")
    print(f"status={status['status']} passed={status['passed']}")
    for step in status["commands"]:
        print(f"$ {' '.join(step['command'])}")
        print(f"returncode={step['returncode']} elapsed_seconds={step['elapsed_seconds']}")
        step_output = step["output"].rstrip()
        if step_output:
            print(step_output)
    if args.require_pass and not status["passed"]:
        raise SystemExit(status["status"])


if __name__ == "__main__":
    main()
