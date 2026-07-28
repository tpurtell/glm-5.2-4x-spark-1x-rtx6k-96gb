#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist


def row_partition(rows: int, world_size: int, rank: int) -> tuple[int, int]:
    base_rows, extra_rows = divmod(rows, world_size)
    local_rows = base_rows + int(rank < extra_rows)
    row_start = rank * base_rows + min(rank, extra_rows)
    return row_start, local_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the Spark FP8 row-sharded all-to-all payload shape."
    )
    parser.add_argument("--rows", type=int, nargs="+", default=(512, 1024))
    parser.add_argument("--row-bytes", type=int, default=6148)
    parser.add_argument("--hidden-dim", type=int, default=6144)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if (
        any(rows < 1 for rows in args.rows)
        or min(args.row_bytes, args.hidden_dim) < 1
        or min(args.warmup, args.iterations, args.repeats) < 1
    ):
        parser.error("rows, row-bytes, warmup, iterations, and repeats must be positive")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    device = torch.device("cuda", torch.cuda.current_device())

    for rows in args.rows:
        if rows < world_size:
            parser.error("each case needs at least one row per rank")
        send_splits = [
            row_partition(rows, world_size, peer)[1] * args.row_bytes
            for peer in range(world_size)
        ]
        local_bytes = send_splits[rank]
        recv_splits = [local_bytes] * world_size
        send = torch.zeros(sum(send_splits), dtype=torch.uint8, device=device)
        recv = torch.empty(sum(recv_splits), dtype=torch.uint8, device=device)

        def operation() -> None:
            dist.all_to_all_single(
                recv,
                send,
                output_split_sizes=recv_splits,
                input_split_sizes=send_splits,
            )

        for _ in range(args.warmup):
            operation()
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.repeats):
            dist.barrier()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iterations):
                operation()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / args.iterations)
        gathered: list[list[float] | None] = [None] * world_size
        dist.all_gather_object(gathered, samples)
        if rank == 0:
            flat_samples = [sample for peer in gathered for sample in (peer or [])]
            print(
                json.dumps(
                    {
                        "benchmark": "nccl_row_all_to_all",
                        "local_payload_bytes": local_bytes * (world_size - 1),
                        "median_ms": statistics.median(flat_samples),
                        "peer_samples_ms": gathered,
                        "row_bytes": args.row_bytes,
                        "rows": rows,
                        "world_size": world_size,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        if rows % world_size != 0:
            continue
        local_rows = rows // world_size
        reduce_send = torch.zeros(
            (rows, args.hidden_dim), dtype=torch.bfloat16, device=device
        )
        reduce_recv = torch.empty(
            (local_rows, args.hidden_dim), dtype=torch.bfloat16, device=device
        )

        def reduce_scatter_operation() -> None:
            dist.reduce_scatter_tensor(reduce_recv, reduce_send)

        for _ in range(args.warmup):
            reduce_scatter_operation()
        torch.cuda.synchronize()
        reduce_samples = []
        for _ in range(args.repeats):
            dist.barrier()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iterations):
                reduce_scatter_operation()
            end.record()
            end.synchronize()
            reduce_samples.append(start.elapsed_time(end) / args.iterations)
        gathered_reduce: list[list[float] | None] = [None] * world_size
        dist.all_gather_object(gathered_reduce, reduce_samples)
        if rank == 0:
            flat_reduce = [
                sample for peer in gathered_reduce for sample in (peer or [])
            ]
            print(
                json.dumps(
                    {
                        "benchmark": "nccl_reduce_scatter_bf16",
                        "hidden_dim": args.hidden_dim,
                        "input_bytes": reduce_send.numel() * reduce_send.element_size(),
                        "median_ms": statistics.median(flat_reduce),
                        "output_bytes": reduce_recv.numel()
                        * reduce_recv.element_size(),
                        "peer_samples_ms": gathered_reduce,
                        "rows": rows,
                        "world_size": world_size,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
