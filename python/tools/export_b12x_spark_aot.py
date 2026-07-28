#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


REGIMES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
PROJECTIONS = {
    "gate": (2048, 6144),
    "down": (6144, 2048),
    "gate_tp4": (512, 6144),
    "down_tp4": (6144, 512),
}


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _compile_dense_kernel(
    *,
    rows: int,
    n: int,
    k: int,
) -> Any:
    import torch
    import b12x.gemm.dense as dense
    from b12x.cute.fp4 import as_grouped_scale_view

    compiled: list[Any] = []
    original_compile = dense.b12x_compile

    def capture_compile(*args: Any, **kwargs: Any) -> Any:
        result = original_compile(*args, **kwargs)
        compiled.append(result)
        return result

    dense.b12x_compile = capture_compile
    dense._get_compiled_dense_gemm.cache_clear()
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        a = torch.zeros((rows, k // 2, 1), dtype=torch.uint8, device=device)
        b = torch.zeros((n, k // 2, 1), dtype=torch.uint8, device=device)
        a_scale_storage = torch.zeros(
            (1, _align_up(rows, 128), _align_up(k // 16, 4)),
            dtype=torch.uint8,
            device=device,
        )
        b_scale_storage = torch.zeros(
            (1, _align_up(n, 128), _align_up(k // 16, 4)),
            dtype=torch.uint8,
            device=device,
        )
        a_scale = as_grouped_scale_view(a_scale_storage, rows, k)
        b_scale = as_grouped_scale_view(b_scale_storage, n, k)
        alpha = torch.ones((1,), dtype=torch.float32, device=device)
        output = torch.empty((rows, n, 1), dtype=torch.bfloat16, device=device)
        dense.dense_gemm(
            (a, a_scale),
            (b, b_scale),
            alpha=alpha,
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            c_dtype="bfloat16",
            sf_vec_size=16,
            out=output,
            expected_m=rows,
        )
        torch.cuda.synchronize()
    finally:
        dense.b12x_compile = original_compile

    if len(compiled) != 1:
        raise RuntimeError(
            f"expected one B12X dense kernel compile for n={n} k={k} rows={rows}, "
            f"got {len(compiled)}"
        )
    return compiled[0]


def export_kernels(output_dir: Path) -> None:
    # Disk-cache hits are executable-only and do not retain the MLIR required
    # by export_to_c(). Force real compilation for build-time AOT export.
    os.environ["B12X_CUTE_COMPILE_DISK_CACHE"] = "0"
    os.environ["B12X_CUTE_COMPILE_MEMORY_CACHE"] = "0"
    output_dir.mkdir(parents=True, exist_ok=True)
    for projection, (n, k) in PROJECTIONS.items():
        for rows in REGIMES:
            name = f"{projection}_m{rows}"
            prefix = f"glmrt_b12x_{name}"
            compiled = _compile_dense_kernel(rows=rows, n=n, k=k)
            compiled.export_to_c(str(output_dir), name, prefix)
    (output_dir / "b12x_spark_aot.stamp").write_text("ready\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Spark GLM B12X dense kernels as C-compatible objects."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export_kernels(args.output_dir.resolve())


if __name__ == "__main__":
    main()
