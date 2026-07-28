#!/usr/bin/env python3
"""Check that differently sized requests do not accumulate coordinator VRAM."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default="http://127.0.0.1:8000/v1/chat/completions"
    )
    parser.add_argument("--model", default="lukealonso/GLM-5.2-NVFP4-full")
    parser.add_argument(
        "--prompt-repetitions",
        type=int,
        nargs="+",
        default=[8_000, 10_000, 12_000, 14_000, 16_000, 18_000, 21_000],
        help="Counts of the single-token 'x ' probe string.",
    )
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-growth-mib", type=int, default=512)
    parser.add_argument("--plateau-tolerance-mib", type=int, default=32)
    return parser.parse_args()


def glmrt_vram_mib(server_pid: int | None) -> tuple[int, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory,name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    processes = []
    for line in output.splitlines():
        if not line.strip():
            continue
        pid_text, memory_text, name = (part.strip() for part in line.split(",", 2))
        processes.append((int(pid_text), int(memory_text), name))

    if server_pid is not None:
        matches = [process for process in processes if process[0] == server_pid]
    else:
        matches = [
            process for process in processes if os.path.basename(process[2]) == "glmrt"
        ]
    if len(matches) != 1:
        target = f"PID {server_pid}" if server_pid is not None else "one glmrt process"
        raise RuntimeError(f"expected {target} in nvidia-smi, found {matches!r}")
    pid, memory_mib, _ = matches[0]
    return pid, memory_mib


def execute_probe(args: argparse.Namespace, repetitions: int) -> dict[str, object]:
    payload = json.dumps(
        {
            "model": args.model,
            "messages": [{"role": "user", "content": "x " * repetitions}],
            "max_tokens": 1,
            "temperature": 0,
        }
    ).encode()
    request = urllib.request.Request(
        args.url, data=payload, headers={"Content-Type": "application/json"}
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        result = json.load(response)
    elapsed_seconds = time.monotonic() - started
    pid, memory_mib = glmrt_vram_mib(args.server_pid)
    usage = result["usage"]
    real_full = result.get("metrics", {}).get("real_full", {})
    return {
        "prompt_repetitions": repetitions,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "elapsed_seconds": elapsed_seconds,
        "server_pid": pid,
        "vram_mib": memory_mib,
        "runtime_captures": real_full.get("request_coordinator_graph_captures"),
    }


def main() -> None:
    args = parse_args()
    if not args.prompt_repetitions or any(
        repetitions < 1 for repetitions in args.prompt_repetitions
    ):
        raise SystemExit("--prompt-repetitions must contain positive values")

    server_pid, baseline_mib = glmrt_vram_mib(args.server_pid)
    args.server_pid = server_pid
    targets = list(args.prompt_repetitions)
    sequence = targets + [min(targets), max(targets)]
    samples = []
    for sample_index, repetitions in enumerate(sequence, start=1):
        sample = execute_probe(args, repetitions)
        sample["sample"] = sample_index
        samples.append(sample)
        print(json.dumps(sample), flush=True)

    first_max_index = targets.index(max(targets))
    plateau_mib = samples[first_max_index]["vram_mib"]
    retained_growth_mib = max(sample["vram_mib"] for sample in samples) - baseline_mib
    final_probe_deltas_mib = [
        abs(sample["vram_mib"] - plateau_mib) for sample in samples[-2:]
    ]
    captures = [
        sample["runtime_captures"]
        for sample in samples
        if sample["runtime_captures"] is not None
    ]
    summary = {
        "server_pid": server_pid,
        "baseline_vram_mib": baseline_mib,
        "plateau_vram_mib": plateau_mib,
        "retained_growth_mib": retained_growth_mib,
        "max_growth_mib": args.max_growth_mib,
        "final_probe_deltas_mib": final_probe_deltas_mib,
        "plateau_tolerance_mib": args.plateau_tolerance_mib,
        "runtime_capture_counts": captures,
        "all_zero_runtime_captures": all(capture == 0 for capture in captures),
        "passed": (
            retained_growth_mib <= args.max_growth_mib
            and max(final_probe_deltas_mib) <= args.plateau_tolerance_mib
        ),
    }
    print(json.dumps({"summary": summary}), flush=True)
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
