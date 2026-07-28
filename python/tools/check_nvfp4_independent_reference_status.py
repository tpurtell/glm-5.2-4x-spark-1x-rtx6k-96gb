#!/usr/bin/env python3
"""Write structured status for the independent NVFP4 oracle requirement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from glmrt_reference.nvfp4_independent_status import build_independent_reference_status


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root() / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(
            "tests/fixtures/nvfp4/real_tensor_decode.json"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/phase0_summary.json"),
    )
    parser.add_argument(
        "--verification",
        type=Path,
        default=Path("tests/fixtures/nvfp4/modelopt_reference.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/phase0_artifacts/nvfp4_independent_reference_status_step396.json"
        ),
    )
    parser.add_argument(
        "--require-comparison",
        action="store_true",
        help="exit non-zero unless an independent comparison has actually run",
    )
    args = parser.parse_args()

    status = build_independent_reference_status(
        repo_root(),
        fixture_path=resolve_repo_path(args.fixture),
        summary_path=resolve_repo_path(args.summary),
        verification_path=resolve_repo_path(args.verification),
    )
    output = resolve_repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {output.relative_to(repo_root())}")
    print(
        "status="
        f"{status['status']} "
        f"comparison={status['independent_real_reference_comparison']} "
        f"missing={','.join(status['missing_oracle_modules']) or 'none'}"
    )
    if args.require_comparison and not status["independent_real_reference_comparison"]:
        raise SystemExit(status["reason"])


if __name__ == "__main__":
    main()
