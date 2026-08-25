#!/usr/bin/env python3
"""Summarize one matched four-Spark GLM-5.2 expert startup from daemon logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
from typing import Any


SCHEMA = "glmrt-glm52-expert-startup-v2"
EXL3_MODEL = "wrldsuksgo2mars/GLM-5.2-EXL3-K3-calibrated-v1"
NVFP4_MODELS = {
    "lukealonso/GLM-5.2-NVFP4",
    "lukealonso/GLM-5.2-NVFP4-full",
    "nvidia/GLM-5.2-NVFP4",
    "nvidia/GLM-5.2-NVFP4-full",
}
EXPECTED_HOSTS = 4
EXPECTED_EXL3_LAYERS = set(range(3, 78))
EXL3_DIRECT_SOURCE_BYTES_PER_RANK_LAYER = 916_196_352
EXL3_COOPERATIVE_SOURCE_BYTES_PER_RANK_LAYER = 909_116_160
EXL3_RESIDENT_BYTES_PER_RANK_LAYER = 916_194_304
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
START_RE = re.compile(r"^== .* starting expert-[0-9]+:")
EXIT_RE = re.compile(r"^== .* expert-[0-9]+ exited status=(?P<status>-?[0-9]+) ==$")
PHASE_RE = re.compile(
    r"^expertd_startup_phase stage=(?P<stage>[A-Za-z0-9-]+) "
    r"elapsed_ms=(?P<elapsed>[0-9]+(?:\.[0-9]+)?) "
    r"total_ms=(?P<total>[0-9]+(?:\.[0-9]+)?)$"
)
FIELD_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[^ ]+)")
PROCESS_PREFIX = "starting expertd "


class StartupError(RuntimeError):
    """The logs do not prove one complete, production expert startup."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def finite_nonnegative(value: str, label: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise StartupError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(number) or number < 0.0:
        raise StartupError(f"{label} is not finite and nonnegative")
    return number


def finite_positive(value: str, label: str) -> float:
    number = finite_nonnegative(value, label)
    if number <= 0.0:
        raise StartupError(f"{label} is not positive")
    return number


def integer_field(fields: dict[str, str], name: str, label: str) -> int:
    try:
        value = int(fields[name])
    except (KeyError, ValueError) as error:
        raise StartupError(f"{label} has no integer {name}") from error
    if value < 0:
        raise StartupError(f"{label} has a negative {name}")
    return value


def fields(line: str, prefix: str) -> dict[str, str]:
    if not line.startswith(prefix):
        raise StartupError(f"line does not begin with {prefix!r}")
    parsed = {match.group("key"): match.group("value") for match in FIELD_RE.finditer(line)}
    if not parsed:
        raise StartupError(f"{prefix} line has no key/value fields")
    return parsed


def final_start_segment(path: Path) -> tuple[Path, list[str], int]:
    source = path.expanduser()
    if source.is_symlink():
        raise StartupError(f"startup log is a symbolic link: {source}")
    resolved = source.resolve(strict=True)
    if not resolved.is_file():
        raise StartupError(f"startup log is not one regular file: {resolved}")
    lines = resolved.read_text(encoding="utf-8").splitlines()
    starts = [index for index, line in enumerate(lines) if START_RE.match(line)]
    if not starts:
        raise StartupError(f"startup log has no expert process boundary: {resolved}")
    segment = lines[starts[-1] :]
    return resolved, segment, starts[-1] + 1


def summarize_log(
    host: str,
    path: Path,
    *,
    model: str,
    weight_format: str,
    include_mtp: bool,
    expert_runtime_fingerprint: str,
) -> dict[str, Any]:
    resolved, lines, start_line = final_start_segment(path)
    phases: dict[str, dict[str, float]] = {}
    process_lines: list[str] = []
    exit_statuses: list[int] = []
    resident_lines: list[str] = []
    exl3_lines: list[str] = []
    for line in lines:
        if line.startswith(PROCESS_PREFIX):
            process_lines.append(line)
        elif match := EXIT_RE.match(line):
            exit_statuses.append(int(match.group("status")))
        elif match := PHASE_RE.match(line):
            stage = match.group("stage")
            if stage in phases:
                raise StartupError(f"{resolved} duplicates startup phase {stage}")
            phases[stage] = {
                "elapsed_ms": finite_nonnegative(
                    match.group("elapsed"), f"{resolved}:{stage}.elapsed_ms"
                ),
                "total_ms": finite_nonnegative(
                    match.group("total"), f"{resolved}:{stage}.total_ms"
                ),
            }
        elif line.startswith("expertd_real_weight_resident_preload "):
            resident_lines.append(line)
        elif line.startswith("real_exl3_cuda_layer_preload "):
            exl3_lines.append(line)
    if len(process_lines) != 1:
        raise StartupError(f"{resolved} must report one expert process configuration")
    process = fields(process_lines[0], PROCESS_PREFIX)
    if process.get("synthetic_weights") != "false":
        raise StartupError(f"{resolved} did not launch with real weights")
    if process.get("model_id") != model:
        raise StartupError(
            f"{resolved} launched model {process.get('model_id')!r}, expected {model!r}"
        )
    if process.get("runtime_identity") != expert_runtime_fingerprint:
        raise StartupError(
            f"{resolved} runtime identity differs from the deployment fingerprint"
        )
    if process.get("transport") != "verbs-host":
        raise StartupError(f"{resolved} did not use the production verbs-host transport")
    if process.get("role") not in {f'Some("spark-{rank}")' for rank in range(4)}:
        raise StartupError(f"{resolved} has an invalid Spark role identity")
    if len(exit_statuses) > 1:
        raise StartupError(f"{resolved} reports more than one process exit")
    if exit_statuses and exit_statuses[0] not in {0, 143}:
        raise StartupError(
            f"{resolved} expert process exited unsuccessfully: {exit_statuses[0]}"
        )
    required_phases = {
        "loadplan",
        "python-capture",
        "catalog-owner-config",
        "catalog-filter-validation",
        "executor-configuration",
        "resident-preload",
        "service-handoff",
    }
    missing = required_phases - set(phases)
    if missing:
        raise StartupError(f"{resolved} is missing startup phases: {sorted(missing)}")
    if len(resident_lines) != 1:
        raise StartupError(f"{resolved} must report one resident preload summary")
    resident = fields(resident_lines[0], "expertd_real_weight_resident_preload ")
    expected_layers = len(EXPECTED_EXL3_LAYERS) + int(include_mtp)
    expected_projection_groups = expected_layers * 256 * 3
    expected = {
        "projection_groups": expected_projection_groups,
        "layers": expected_layers,
        "experts": expected_layers * 256,
        "cuda_projection_groups": expected_projection_groups,
        # RouteTensorCacheStats reports logical expert projections resident
        # inside its per-layer slabs, not the number of slab allocations.
        "cuda_projection_entries": expected_projection_groups,
        "cuda_projection_uploads": expected_projection_groups,
    }
    for name, value in expected.items():
        if integer_field(resident, name, os.fspath(resolved)) != value:
            raise StartupError(f"{resolved} has unexpected {name}")
    if resident.get("cuda_reference_enabled") != "true":
        raise StartupError(f"{resolved} did not preload CUDA expert weights")

    exl3: dict[str, Any] | None = None
    if weight_format == "exl3":
        by_layer: dict[int, dict[str, Any]] = {}
        for line in exl3_lines:
            parsed = fields(line, "real_exl3_cuda_layer_preload ")
            layer_id = integer_field(parsed, "layer_id", os.fspath(resolved))
            if layer_id in by_layer:
                raise StartupError(f"{resolved} duplicates EXL3 layer {layer_id}")
            if integer_field(parsed, "experts", os.fspath(resolved)) != 256:
                raise StartupError(f"{resolved} layer {layer_id} did not preload 256 experts")
            source_bytes = integer_field(parsed, "source_bytes", os.fspath(resolved))
            resident_bytes = integer_field(
                parsed, "resident_bytes", os.fspath(resolved)
            )
            if resident_bytes != EXL3_RESIDENT_BYTES_PER_RANK_LAYER:
                raise StartupError(
                    f"{resolved} layer {layer_id} resident geometry differs"
                )
            common = {
                "source_bytes": source_bytes,
                "source_gbps": finite_positive(
                    parsed.get("source_gbps", ""),
                    f"{resolved}:layer {layer_id} source_gbps",
                ),
                "resident_bytes": resident_bytes,
            }
            if parsed.get("cooperative") == "false":
                if (
                    parsed.get("direct_resident") != "true"
                    or source_bytes != EXL3_DIRECT_SOURCE_BYTES_PER_RANK_LAYER
                ):
                    raise StartupError(
                        f"{resolved} layer {layer_id} direct source geometry differs"
                    )
                by_layer[layer_id] = {
                    **common,
                    "preload_mode": "direct-resident",
                    "allocation_ms": finite_nonnegative(
                        parsed.get("allocation_ms", ""),
                        f"{resolved}:layer {layer_id} allocation_ms",
                    ),
                    "direct_ms": finite_positive(
                        parsed.get("direct_ms", ""),
                        f"{resolved}:layer {layer_id} direct_ms",
                    ),
                }
            elif parsed.get("cooperative") == "true":
                if (
                    parsed.get("packed_exchange") != "true"
                    or parsed.get("direct_io") != "true"
                    or source_bytes
                    != EXL3_COOPERATIVE_SOURCE_BYTES_PER_RANK_LAYER
                    or integer_field(parsed, "source_experts", os.fspath(resolved))
                    != 64
                    or integer_field(parsed, "source_requests", os.fspath(resolved))
                    != 768
                    or not 1
                    <= integer_field(parsed, "source_spans", os.fspath(resolved))
                    <= 8
                ):
                    raise StartupError(
                        f"{resolved} layer {layer_id} was not coalesced cooperative TP4"
                    )
                by_layer[layer_id] = {
                    **common,
                    "preload_mode": "cooperative-coalesced",
                    "source_requests": 768,
                    "source_spans": integer_field(
                        parsed, "source_spans", os.fspath(resolved)
                    ),
                    **{
                        f"{name}_ms": finite_nonnegative(
                            parsed.get(f"{name}_ms", ""),
                            f"{resolved}:layer {layer_id} {name}_ms",
                        )
                        for name in ("load", "pack", "allocation", "upload", "exchange")
                    },
                }
            else:
                raise StartupError(
                    f"{resolved} layer {layer_id} has an unsupported EXL3 preload mode"
                )
        if set(by_layer) != EXPECTED_EXL3_LAYERS:
            raise StartupError(
                f"{resolved} EXL3 layer coverage differs: "
                f"missing={sorted(EXPECTED_EXL3_LAYERS - set(by_layer))} "
                f"unexpected={sorted(set(by_layer) - EXPECTED_EXL3_LAYERS)}"
            )
        resident_sizes = {record["resident_bytes"] for record in by_layer.values()}
        if len(resident_sizes) != 1:
            raise StartupError(f"{resolved} EXL3 layer resident geometry changes")
        preload_modes = {record["preload_mode"] for record in by_layer.values()}
        if len(preload_modes) != 1:
            raise StartupError(f"{resolved} mixes EXL3 preload modes")
        preload_mode = preload_modes.pop()
        exl3 = {
            "preload_mode": preload_mode,
            "layers": len(by_layer),
            "source_bytes": sum(record["source_bytes"] for record in by_layer.values()),
            "resident_bytes": sum(record["resident_bytes"] for record in by_layer.values()),
            "minimum_source_gbps": min(
                record["source_gbps"] for record in by_layer.values()
            ),
        }
        if preload_mode == "direct-resident":
            exl3.update(
                allocation_ms=sum(
                    record["allocation_ms"] for record in by_layer.values()
                ),
                direct_ms=sum(record["direct_ms"] for record in by_layer.values()),
            )
            minimum_resident_ms = exl3["allocation_ms"] + exl3["direct_ms"]
        else:
            for name in ("load", "pack", "allocation", "upload", "exchange"):
                exl3[f"{name}_ms"] = sum(
                    record[f"{name}_ms"] for record in by_layer.values()
                )
            exl3["source_requests"] = sum(
                record["source_requests"] for record in by_layer.values()
            )
            exl3["source_spans"] = sum(
                record["source_spans"] for record in by_layer.values()
            )
            # Disk load may overlap all current-layer store work. The store
            # phases themselves still execute in order on one rank.
            minimum_resident_ms = max(
                exl3["load_ms"],
                sum(
                    exl3[f"{name}_ms"]
                    for name in ("pack", "allocation", "upload", "exchange")
                ),
            )
        if (
            integer_field(resident, "cuda_weight_bytes", os.fspath(resolved))
            != exl3["resident_bytes"]
            or integer_field(
                resident, "cuda_weight_scale_bytes", os.fspath(resolved)
            )
            != 0
        ):
            raise StartupError(
                f"{resolved} resident summary differs from direct EXL3 layers"
            )
        if phases["resident-preload"]["elapsed_ms"] + 1.0 < minimum_resident_ms:
            raise StartupError(
                f"{resolved} resident phase is shorter than its EXL3 layer timings"
            )
    elif exl3_lines:
        raise StartupError(f"{resolved} NVFP4 startup unexpectedly executed EXL3 layers")

    return {
        "host": host,
        "log": {
            "path": os.fspath(resolved),
            "bytes": resolved.stat().st_size,
            "sha256": hash_file(resolved),
            "selected_start_line": start_line,
        },
        "process": {
            "synthetic_weights": False,
            "model_id": process["model_id"],
            "transport": process.get("transport"),
            "listen": process.get("listen"),
            "role": process.get("role"),
            "runtime_identity": process["runtime_identity"],
            "exit_status": exit_statuses[0] if exit_statuses else None,
        },
        "phases": phases,
        "resident": {
            name: (
                resident[name]
                if name == "cuda_reference_enabled"
                else integer_field(resident, name, os.fspath(resolved))
            )
            for name in (
                "projection_groups",
                "layers",
                "experts",
                "cuda_reference_enabled",
                "cuda_projection_groups",
                "cuda_weight_bytes",
                "cuda_weight_scale_bytes",
                "cuda_projection_entries",
                "cuda_projection_uploads",
            )
        },
        "exl3": exl3,
    }


def analyze(
    *,
    model: str,
    weight_format: str,
    cache_state: str,
    include_mtp: bool,
    expert_runtime_fingerprint: str,
    logs: list[tuple[str, Path]],
) -> dict[str, Any]:
    if weight_format == "exl3" and model != EXL3_MODEL:
        raise StartupError("EXL3 startup must use the calibrated EXL3 model ID")
    if weight_format == "exl3" and include_mtp:
        raise StartupError("EXL3 dSpark qualification must not load the native MTP layer")
    if weight_format == "nvfp4" and model not in NVFP4_MODELS:
        raise StartupError("NVFP4 startup must use a supported NVFP4 model ID")
    if SHA256_RE.fullmatch(expert_runtime_fingerprint) is None:
        raise StartupError("expert runtime fingerprint must be lowercase SHA-256")
    if len(logs) != EXPECTED_HOSTS or len({host for host, _ in logs}) != EXPECTED_HOSTS:
        raise StartupError("startup evidence requires four unique Spark host logs")
    hosts = [
        summarize_log(
            host,
            path,
            model=model,
            weight_format=weight_format,
            include_mtp=include_mtp,
            expert_runtime_fingerprint=expert_runtime_fingerprint,
        )
        for host, path in logs
    ]
    roles = {host["process"]["role"] for host in hosts}
    if roles != {f'Some("spark-{rank}")' for rank in range(4)}:
        raise StartupError("startup evidence does not cover four unique Spark roles")
    resident_ms = [host["phases"]["resident-preload"]["elapsed_ms"] for host in hosts]
    total_ms = [host["phases"]["service-handoff"]["total_ms"] for host in hosts]
    if weight_format == "exl3":
        preload_modes = {host["exl3"]["preload_mode"] for host in hosts}
        if len(preload_modes) != 1:
            raise StartupError("startup evidence mixes EXL3 preload modes across hosts")
        preload_mode = preload_modes.pop()
    else:
        preload_mode = "nvfp4-production"
    body = {
        "schema": SCHEMA,
        "status": "accepted",
        "model": model,
        "expert_runtime_fingerprint": expert_runtime_fingerprint,
        "weight_format": weight_format,
        "preload_mode": preload_mode,
        "cache_state": cache_state,
        "include_mtp": include_mtp,
        "hosts": hosts,
        "summary": {
            "host_count": len(hosts),
            "maximum_resident_preload_ms": max(resident_ms),
            "median_resident_preload_ms": statistics.median(resident_ms),
            "maximum_service_handoff_total_ms": max(total_ms),
            "median_service_handoff_total_ms": statistics.median(total_ms),
        },
    }
    return {**body, "report_sha256": hashlib.sha256(canonical_json(body)).hexdigest()}


def parse_log(value: str) -> tuple[str, Path]:
    host, separator, raw_path = value.partition("=")
    if not separator or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", host):
        raise argparse.ArgumentTypeError("log must be HOST=PATH")
    return host, Path(raw_path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as target:
            target.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--weight-format", choices=("nvfp4", "exl3"), required=True)
    parser.add_argument("--cache-state", choices=("cold", "warm"), required=True)
    parser.add_argument("--include-mtp", action="store_true")
    parser.add_argument("--expert-runtime-fingerprint", required=True)
    parser.add_argument("--log", type=parse_log, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        model=args.model,
        weight_format=args.weight_format,
        cache_state=args.cache_state,
        include_mtp=args.include_mtp,
        expert_runtime_fingerprint=args.expert_runtime_fingerprint,
        logs=args.log,
    )
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
