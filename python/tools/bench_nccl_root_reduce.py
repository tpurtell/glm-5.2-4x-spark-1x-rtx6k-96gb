#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.distributed as dist


def fixture(rank: int, rows: int, row_width: int, device: torch.device) -> torch.Tensor:
    row = torch.arange(rows, dtype=torch.int32, device=device).view(-1, 1)
    column = torch.arange(row_width, dtype=torch.int32, device=device).view(1, -1)
    values = torch.remainder(rank * 37 + row * 17 + column * 13, 127)
    return ((values.to(torch.float32) - 63.0) / 16.0).to(torch.bfloat16)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def summary(values: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "mean_us": statistics.fmean(values),
        "p50_us": percentile(values, 0.50),
        "p95_us": percentile(values, 0.95),
        "p99_us": percentile(values, 0.99),
        "max_us": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the exact NCCL BF16 root-reduce shape used by Spark partials."
    )
    parser.add_argument("--rows", type=int, nargs="+", default=(1, 16, 256))
    parser.add_argument("--row-width", type=int, default=6144)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--serialized-iterations", type=int, default=30)
    parser.add_argument("--network-label", default="unspecified")
    args = parser.parse_args()
    if (
        any(rows < 1 for rows in args.rows)
        or args.row_width < 1
        or min(
            args.warmup,
            args.iterations,
            args.repeats,
            args.serialized_iterations,
        )
        < 1
    ):
        parser.error(
            "rows, row-width, warmup, iterations, repeats, and serialized-iterations "
            "must be positive"
        )

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)

    for rows in args.rows:
        source = fixture(rank, rows, args.row_width, device)
        working = torch.empty_like(source)

        for _ in range(args.warmup):
            working.copy_(source)
            dist.reduce(working, dst=0, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        local_samples: list[float] = []
        local_host_samples: list[float] = []
        for _ in range(args.repeats):
            dist.barrier()
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
            host_started = time.perf_counter_ns()
            for start, end in zip(starts, ends, strict=True):
                working.copy_(source)
                start.record()
                dist.reduce(working, dst=0, op=dist.ReduceOp.SUM)
                end.record()
            ends[-1].synchronize()
            host_elapsed_us = (time.perf_counter_ns() - host_started) / 1_000.0
            local_host_samples.append(host_elapsed_us / args.iterations)
            local_samples.extend(
                start.elapsed_time(end) * 1_000.0
                for start, end in zip(starts, ends, strict=True)
            )

        dist.barrier()
        serialized_samples: list[float] = []
        serialized_host_samples: list[float] = []
        for _ in range(args.serialized_iterations):
            working.copy_(source)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            host_started = time.perf_counter_ns()
            start.record()
            dist.reduce(working, dst=0, op=dist.ReduceOp.SUM)
            end.record()
            end.synchronize()
            serialized_host_samples.append(
                (time.perf_counter_ns() - host_started) / 1_000.0
            )
            serialized_samples.append(start.elapsed_time(end) * 1_000.0)

        working.copy_(source)
        dist.reduce(working, dst=0, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        quality = None
        if rank == 0:
            expected = sum(
                fixture(peer_rank, rows, args.row_width, device).to(torch.float32)
                for peer_rank in range(world_size)
            ).to(torch.bfloat16)
            exact = torch.equal(working, expected)
            if not exact:
                difference = working.to(torch.float32) - expected.to(torch.float32)
                relative_l2 = (
                    torch.linalg.vector_norm(difference)
                    / torch.linalg.vector_norm(expected.to(torch.float32)).clamp_min(1.0e-30)
                ).item()
                raise RuntimeError(f"NCCL BF16 root reduction was not exact: {relative_l2=:.6e}")
            quality = {
                "exact": True,
                "output_checksum": working.to(torch.float32).sum().item(),
                "expected_checksum": expected.to(torch.float32).sum().item(),
            }

        gathered_samples: list[list[float] | None] = [None] * world_size
        gathered_host: list[list[float] | None] = [None] * world_size
        gathered_serialized: list[list[float] | None] = [None] * world_size
        gathered_serialized_host: list[list[float] | None] = [None] * world_size
        dist.all_gather_object(gathered_samples, local_samples)
        dist.all_gather_object(gathered_host, local_host_samples)
        dist.all_gather_object(gathered_serialized, serialized_samples)
        dist.all_gather_object(gathered_serialized_host, serialized_host_samples)
        if rank == 0:
            peer_samples = [samples or [] for samples in gathered_samples]
            critical_samples = [
                max(peer[index] for peer in peer_samples)
                for index in range(len(peer_samples[0]))
            ]
            peer_host = [samples or [] for samples in gathered_host]
            critical_host = [
                max(peer[index] for peer in peer_host)
                for index in range(len(peer_host[0]))
            ]
            peer_serialized = [samples or [] for samples in gathered_serialized]
            critical_serialized = [
                max(peer[index] for peer in peer_serialized)
                for index in range(len(peer_serialized[0]))
            ]
            peer_serialized_host = [
                samples or [] for samples in gathered_serialized_host
            ]
            critical_serialized_host = [
                max(peer[index] for peer in peer_serialized_host)
                for index in range(len(peer_serialized_host[0]))
            ]
            print(
                json.dumps(
                    {
                        "benchmark": "nccl-bf16-root-reduction",
                        "network_label": args.network_label,
                        "pre_firmware": args.network_label == "pre-firmware",
                        "world_size": world_size,
                        "root_rank": 0,
                        "rows": rows,
                        "row_width": args.row_width,
                        "payload_bytes_per_rank": rows * args.row_width * 2,
                        "warmup_iterations": args.warmup,
                        "iterations": args.iterations,
                        "repeats": args.repeats,
                        "serialized_iterations": args.serialized_iterations,
                        "queued_critical_gpu": summary(critical_samples),
                        "queued_critical_host_step": summary(critical_host),
                        "serialized_critical_gpu": summary(critical_serialized),
                        "serialized_critical_host": summary(
                            critical_serialized_host
                        ),
                        "peer_queued_gpu": [summary(peer) for peer in peer_samples],
                        "peer_queued_host_step": [
                            summary(peer) for peer in peer_host
                        ],
                        "peer_serialized_gpu": [
                            summary(peer) for peer in peer_serialized
                        ],
                        "peer_serialized_host": [
                            summary(peer) for peer in peer_serialized_host
                        ],
                        "quality": quality,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
