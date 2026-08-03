#!/usr/bin/env python3
"""Compare the current TP4 W4A16 Spark kernel with a TP2/EP2 replay.

The benchmark is deliberately network-free.  It replays retained, real GLM-5.2
top-8 routes on one GB10 and compares:

* the production TP4 shard (all experts, I_tp=512); and
* the critical TP2 pair (half the experts per pair, I_tp=1024).

TP2 expert placement can use expert parity or a held-out, per-layer frequency
partition trained from the replay plan.  Both pair kernels are timed and the
slower one is charged to the candidate, matching the distributed critical
path.  Small-M pair routes are compacted per row before selecting a direct-
top-k specialization; larger M uses the ordinary expert-grouped path.  Route
packing itself is outside the timed CUDA graph in both arms, as it is in the
native Spark executor.
"""

from __future__ import annotations

import _pinned_sparkinfer

import argparse
import dataclasses
import json
import random
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from sparkinfer._lib.utils import make_ptr
from sparkinfer.moe._shared.kernels.w4a16.host import (
    W4A16PackedBuffers,
    make_w4a16_packed_buffers,
)
import sparkinfer.moe._shared.kernels.w4a16.kernel as w4a16_kernel
from sparkinfer.moe._shared.kernels.w4a16.prepare import (
    PreparedW4A16MoeWeights,
    prepare_w4a16_modelopt_nvfp4_weights,
)


HIDDEN = 6144
FULL_INTERMEDIATE = 2048
TP4_INTERMEDIATE = FULL_INTERMEDIATE // 4
TP2_INTERMEDIATE = FULL_INTERMEDIATE // 2
EXPERTS = 256
TP2_EXPERTS = EXPERTS // 2
TOP_K = 8
E4M3_ONE = 0x38
SPARSE_LAYERS = tuple(range(3, 78))
DEFAULT_MS = (*range(1, 17), 32, 64, 128, 256, 512)


@dataclass(frozen=True)
class ReplayCase:
    physical_m: int
    chain_id: str
    layer_id: int
    routes: tuple[tuple[int, ...], ...]

    @property
    def case_id(self) -> str:
        return f"{self.chain_id}-l{self.layer_id}"


@dataclass(frozen=True)
class RouteMetadata:
    topk: int
    local_routes: tuple[tuple[int, ...], ...]
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    packed_route_indices: torch.Tensor
    block_expert_ids: torch.Tensor
    packed_route_count: torch.Tensor
    logical_routes: int
    padded_routes: int
    active_experts: int


@dataclass
class KernelPlan:
    label: str
    prepared: PreparedW4A16MoeWeights
    buffers: W4A16PackedBuffers
    fused: w4a16_kernel.W4A16FusedMoeCompileResult
    capacity_m: int
    topk: int
    intermediate_size: int
    num_experts: int
    block_size: int
    direct_topk: bool
    fused_sum: bool
    zero_fc2_output: bool
    sms: int
    max_shared_mem: int
    input_bf16: torch.Tensor
    grid_x: int


@dataclass
class GraphMeasurement:
    arm: str
    replay_case: ReplayCase
    metadata: RouteMetadata | None
    plan: KernelPlan | None
    graph: torch.cuda.CUDAGraph | None
    samples_ms: list[float] = field(default_factory=list)
    deterministic: bool | None = None
    finite: bool = True
    nonzero: bool = True

    @property
    def median_ms(self) -> float:
        if self.graph is None:
            return 0.0
        return statistics.median(self.samples_ms)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ms",
        default=",".join(str(value) for value in DEFAULT_MS),
        help="Comma-separated physical M values from the replay plan.",
    )
    parser.add_argument("--chains-per-m", type=int, default=4)
    parser.add_argument("--layers-per-chain", type=int, default=3)
    parser.add_argument("--warmup-rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20_260_731)
    parser.add_argument(
        "--tp2-placement",
        choices=("parity", "trained-frequency"),
        default="parity",
        help=(
            "Assign expert_id%%2, or learn an exactly 128/128 per-layer "
            "frequency partition from replay chains disjoint from the held-out run"
        ),
    )
    parser.add_argument(
        "--tp2-small-m-routing",
        choices=("compact", "masked"),
        default="compact",
        help=(
            "compact selects the smallest per-pair top-k covering every row; "
            "masked keeps the original top-k=8 with nonresident routes set to -1"
        ),
    )
    parser.add_argument(
        "--tp2-small-m-kernel",
        choices=("direct", "grouped", "grouped-wide"),
        default="direct",
        help="TP2 M<=8 execution schedule; M1 always uses direct fused-sum.",
    )
    parser.add_argument(
        "--tp2-large-m-block-size",
        choices=(8, 16, 32),
        type=int,
        default=32,
        help="Expert route-block size for the TP2 M>8 arm.",
    )
    parser.add_argument(
        "--tp2-no-zero-output-upper-bound",
        action="store_true",
        help=(
            "Omit the partial-route FC2 zero phase to measure an invalid-output "
            "upper bound for a future skip-aware pair epilogue."
        ),
    )
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Skip the independent full-I TP4-vs-TP2 numerical gate.",
    )
    return parser.parse_args()


def parse_ms(raw: str) -> list[int]:
    values = list(dict.fromkeys(int(value) for value in raw.split(",") if value))
    if not values or any(value < 1 or value > 512 for value in values):
        raise ValueError("--ms must contain values in 1..512")
    return values


def git_revision(path: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=path,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() or "unavailable"


def load_cases(
    path: Path,
    requested_ms: list[int],
    chains_per_m: int,
    layers_per_chain: int,
) -> tuple[dict[str, Any], list[ReplayCase], list[dict[str, Any]]]:
    manifest = None
    by_m: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record.get("record") == "manifest":
                manifest = record
            elif record.get("record") == "chain" and record.get("cohort") == "semantic":
                by_m[int(record["physical_m"])].append(record)
    if manifest is None:
        raise ValueError(f"replay plan has no manifest: {path}")

    # The final chains at each M are the held-out benchmark cohort.  All prior
    # chains remain available for an optional static placement fit.  This split
    # is independent of --ms, so separate small/large runs learn one map.
    training_chains = []
    for physical_m in sorted(by_m):
        chains = sorted(by_m[physical_m], key=lambda item: item["chain_id"])
        training_chains.extend(chains[:-chains_per_m])

    cases = []
    for physical_m in requested_ms:
        chains = sorted(by_m.get(physical_m, []), key=lambda item: item["chain_id"])
        if len(chains) < chains_per_m:
            raise ValueError(
                f"M={physical_m} has {len(chains)} semantic chains, need {chains_per_m}"
            )
        selected = chains[-chains_per_m:]
        layer_count = len(selected[0]["layers"])
        if layers_per_chain > layer_count:
            raise ValueError(
                f"requested {layers_per_chain} layers from a {layer_count}-layer chain"
            )
        if layers_per_chain == 1:
            positions = (layer_count // 2,)
        else:
            positions = tuple(
                round(index * (layer_count - 1) / (layers_per_chain - 1))
                for index in range(layers_per_chain)
            )
        for chain_offset, chain in enumerate(selected):
            # Rotate the evenly spaced layer sample so every chain does not hit
            # precisely the same three layers.
            for position in positions:
                layer = chain["layers"][(position + chain_offset * 7) % layer_count]
                routes = tuple(
                    tuple(int(expert) for expert in row) for row in layer["routes"]
                )
                if len(routes) != physical_m or any(len(row) != TOP_K for row in routes):
                    raise ValueError(f"invalid routes in {chain['chain_id']}")
                cases.append(
                    ReplayCase(
                        physical_m=physical_m,
                        chain_id=str(chain["chain_id"]),
                        layer_id=int(layer["layer_id"]),
                        routes=routes,
                    )
                )
    return manifest, cases, training_chains


def make_pair_residency(
    placement: str,
    training_chains: list[dict[str, Any]],
) -> tuple[dict[int, tuple[tuple[int, ...], tuple[int, ...]]], dict[str, Any]]:
    if placement == "parity":
        residents = {
            layer_id: (
                tuple(expert for expert in range(EXPERTS) if expert % 2 == 0),
                tuple(expert for expert in range(EXPERTS) if expert % 2 == 1),
            )
            for layer_id in SPARSE_LAYERS
        }
        return residents, {
            "method": "expert_id % 2",
            "training_chains": 0,
            "mean_training_load_ratio": None,
            "max_training_load_ratio": None,
        }
    if not training_chains:
        raise ValueError(
            "trained-frequency placement needs replay chains outside the held-out set"
        )

    frequency = {layer_id: Counter() for layer_id in SPARSE_LAYERS}
    for chain in training_chains:
        for layer in chain["layers"]:
            counts = frequency[int(layer["layer_id"])]
            counts.update(int(expert) for row in layer["routes"] for expert in row)

    residents = {}
    load_ratios = []
    for layer_id in SPARSE_LAYERS:
        counts = frequency[layer_id]
        pair_experts: list[list[int]] = [[], []]
        pair_loads = [0, 0]
        ordered = sorted(
            range(EXPERTS),
            key=lambda expert: (-counts[expert], expert),
        )
        for expert in ordered:
            if len(pair_experts[0]) == TP2_EXPERTS:
                pair = 1
            elif len(pair_experts[1]) == TP2_EXPERTS:
                pair = 0
            else:
                pair = 0 if pair_loads[0] <= pair_loads[1] else 1
            pair_experts[pair].append(expert)
            pair_loads[pair] += counts[expert]
        if any(len(experts) != TP2_EXPERTS for experts in pair_experts):
            raise AssertionError("trained placement did not produce a 128/128 split")
        residents[layer_id] = (
            tuple(sorted(pair_experts[0])),
            tuple(sorted(pair_experts[1])),
        )
        mean_load = sum(pair_loads) / 2.0
        load_ratios.append(max(pair_loads) / max(mean_load, 1.0))
    return residents, {
        "method": "held-out per-layer greedy frequency balance",
        "training_chains": len(training_chains),
        "mean_training_load_ratio": statistics.mean(load_ratios),
        "max_training_load_ratio": max(load_ratios),
    }


def local_pair_routes(
    replay_case: ReplayCase,
    pair: int,
    *,
    compact: bool,
    residents_by_layer: dict[
        int, tuple[tuple[int, ...], tuple[int, ...]]
    ],
) -> tuple[tuple[int, ...], ...]:
    residents = residents_by_layer[replay_case.layer_id][pair]
    local_id = {expert: index for index, expert in enumerate(residents)}
    rows = []
    for row in replay_case.routes:
        if compact:
            rows.append(tuple(local_id[expert] for expert in row if expert in local_id))
        else:
            rows.append(tuple(local_id.get(expert, -1) for expert in row))
    return tuple(rows)


def build_route_metadata(
    rows: tuple[tuple[int, ...], ...],
    *,
    num_experts: int,
    block_size: int,
    direct_topk: bool,
    device: torch.device,
) -> RouteMetadata | None:
    if not rows:
        raise ValueError("route metadata requires at least one row")
    topk = max((len(row) for row in rows), default=0)
    if topk == 0:
        return None
    if topk > TOP_K:
        raise ValueError(f"local top-k {topk} exceeds {TOP_K}")
    padded_rows = tuple(row + (-1,) * (topk - len(row)) for row in rows)
    topk_ids = torch.tensor(padded_rows, dtype=torch.int32, device=device)
    # Keep original top-8 router weighting.  These values affect arithmetic but
    # not scheduling; 1/8 also makes TP2 pair partials sum to the TP4 result in
    # the independent split-weight correctness gate.
    topk_weights = torch.full(
        (len(rows), topk), 1.0 / TOP_K, dtype=torch.float32, device=device
    )
    logical_routes = sum(expert >= 0 for row in padded_rows for expert in row)
    active_experts = len({expert for row in padded_rows for expert in row if expert >= 0})

    if direct_topk:
        flat = topk_ids.reshape(-1)
        return RouteMetadata(
            topk=topk,
            local_routes=padded_rows,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            packed_route_indices=flat,
            block_expert_ids=flat,
            packed_route_count=flat[:1],
            logical_routes=logical_routes,
            padded_routes=len(rows) * topk,
            active_experts=active_experts,
        )

    routes_by_expert: list[list[int]] = [[] for _ in range(num_experts)]
    for row_index, row in enumerate(padded_rows):
        for route_slot, expert in enumerate(row):
            if expert >= 0:
                routes_by_expert[expert].append(row_index * topk + route_slot)
    sentinel = len(rows) * topk
    packed_routes: list[int] = []
    block_experts: list[int] = []
    for expert, route_indices in enumerate(routes_by_expert):
        for start in range(0, len(route_indices), block_size):
            block = route_indices[start : start + block_size]
            packed_routes.extend(block)
            packed_routes.extend([sentinel] * (block_size - len(block)))
            block_experts.append(expert)
    if not packed_routes:
        return None
    return RouteMetadata(
        topk=topk,
        local_routes=padded_rows,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        packed_route_indices=torch.tensor(
            packed_routes, dtype=torch.int32, device=device
        ),
        block_expert_ids=torch.tensor(
            block_experts, dtype=torch.int32, device=device
        ),
        packed_route_count=torch.tensor(
            [len(packed_routes)], dtype=torch.int32, device=device
        ),
        logical_routes=logical_routes,
        padded_routes=len(packed_routes),
        active_experts=active_experts,
    )


def make_prepared_weights(
    *,
    num_experts: int,
    intermediate_size: int,
    seed: int,
    device: torch.device,
) -> PreparedW4A16MoeWeights:
    generator = torch.Generator(device=device).manual_seed(seed)
    w13 = torch.randint(
        0,
        256,
        (num_experts, 2 * intermediate_size, HIDDEN // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    w2 = torch.randint(
        0,
        256,
        (num_experts, HIDDEN, intermediate_size // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    w13_scale = torch.full(
        (num_experts, 2 * intermediate_size, HIDDEN // 16),
        E4M3_ONE,
        dtype=torch.uint8,
        device=device,
    ).view(torch.float8_e4m3fn)
    w2_scale = torch.full(
        (num_experts, HIDDEN, intermediate_size // 16),
        E4M3_ONE,
        dtype=torch.uint8,
        device=device,
    ).view(torch.float8_e4m3fn)
    global_scale = torch.ones(num_experts, dtype=torch.float32, device=device)
    prepared = prepare_w4a16_modelopt_nvfp4_weights(
        w13,
        w13_scale,
        global_scale,
        w2,
        w2_scale,
        global_scale,
        activation="silu",
        params_dtype=torch.bfloat16,
        w13_layout="w13",
        reuse_input_storage=True,
    )
    del w13, w2, w13_scale, w2_scale, global_scale
    return prepared


def capacity_for_m(m: int) -> int:
    if m <= 8:
        return m
    return 1 << (m - 1).bit_length()


def compile_grouped_wide(
    compile_call: Callable[..., w4a16_kernel.W4A16FusedMoeCompileResult],
    **kwargs: Any,
) -> w4a16_kernel.W4A16FusedMoeCompileResult:
    """Select the production M2-8 wide tile without enabling atomic output."""

    original_init = w4a16_kernel.W4A16FusedMoeKernel.__init__

    def allow_grouped_wide_fixed_order(self, *args, **init_kwargs):
        requested = bool(init_kwargs.get("tc_decode_fused_sum"))
        grouped = not bool(init_kwargs.get("direct_topk_routes"))
        if requested and grouped:
            init_kwargs["tc_decode_fused_sum"] = False
        original_init(self, *args, **init_kwargs)

    w4a16_kernel.W4A16FusedMoeKernel.__init__ = allow_grouped_wide_fixed_order
    try:
        fused = compile_call(tc_decode_fused_sum=True, **kwargs)
    finally:
        w4a16_kernel.W4A16FusedMoeKernel.__init__ = original_init
    # The compiled kernel has fixed-order route output; make the host-side
    # descriptor reflect that instead of the planner-only wide-tile request.
    return dataclasses.replace(fused, tc_decode_fused_sum=False)


def make_plan(
    *,
    label: str,
    prepared: PreparedW4A16MoeWeights,
    m: int,
    topk: int,
    baseline: bool,
    sms: int,
    max_shared_mem: int,
    device: torch.device,
    tp2_small_m_kernel: str,
    tp2_large_m_block_size: int,
    tp2_no_zero_output_upper_bound: bool,
) -> KernelPlan:
    capacity_m = capacity_for_m(m)
    direct_topk = (
        m == 1
        if baseline
        else m == 1 or (m <= 8 and tp2_small_m_kernel == "direct")
    )
    fused_sum = direct_topk and (m == 1 if baseline else True)
    block_size = (
        8
        if m <= 8
        else 32
        if baseline
        else int(tp2_large_m_block_size)
    )
    zero_fc2_output = (
        (not baseline)
        and not direct_topk
        and not tp2_no_zero_output_upper_bound
    )
    common = dict(
        size_m=capacity_m,
        hidden_size=HIDDEN,
        intermediate_size=int(prepared.intermediate_size),
        num_experts=int(prepared.num_experts),
        top_k=topk,
        activation="silu",
        apply_router_weight_on_input=False,
        zero_fc2_output=zero_fc2_output,
        moe_block_size=block_size,
        max_m_blocks=capacity_m * topk,
        element_dtype="bf16",
        fast_math=True,
        sms=sms,
        max_shared_mem=max_shared_mem,
        weight_layout=prepared.weight_layout,
        scale_format=prepared.scale_format,
        w13_layout=getattr(prepared, "w13_layout", "packed"),
        direct_topk_routes=direct_topk,
    )
    use_grouped_wide = (
        2 <= m <= 8
        and (baseline or tp2_small_m_kernel == "grouped-wide")
    )
    if use_grouped_wide:
        fused = compile_grouped_wide(w4a16_kernel.compile_w4a16_fused_moe, **common)
    else:
        fused = w4a16_kernel.compile_w4a16_fused_moe(
            tc_decode_fused_sum=fused_sum,
            **common,
        )
    buffers = make_w4a16_packed_buffers(
        prepared,
        m=capacity_m,
        topk=topk,
        dtype=torch.bfloat16,
        device=device,
        block_size_m=block_size,
    )
    input_bf16 = torch.randn(
        (capacity_m, HIDDEN), dtype=torch.bfloat16, device=device
    ) * 0.05
    grid_x = w4a16_kernel._w4a16_fused_persistent_grid_x(
        fused=fused,
        m=m,
        topk=topk,
        intermediate_size=int(prepared.intermediate_size),
        activation="silu",
        direct_topk_routes=direct_topk,
        sms=sms,
    )
    return KernelPlan(
        label=label,
        prepared=prepared,
        buffers=buffers,
        fused=fused,
        capacity_m=capacity_m,
        topk=topk,
        intermediate_size=int(prepared.intermediate_size),
        num_experts=int(prepared.num_experts),
        block_size=block_size,
        direct_topk=direct_topk,
        fused_sum=fused_sum,
        zero_fc2_output=zero_fc2_output,
        sms=sms,
        max_shared_mem=max_shared_mem,
        input_bf16=input_bf16,
        grid_x=grid_x,
    )


def launch_plan(plan: KernelPlan, metadata: RouteMetadata, active_m: int) -> torch.Tensor:
    if metadata.topk != plan.topk:
        raise ValueError("metadata and kernel top-k disagree")
    buffers = plan.buffers
    prepared = plan.prepared
    capacity_routed_rows = plan.capacity_m * plan.topk
    fc1_cols = 2 * plan.intermediate_size
    intermediate = buffers.intermediate_cache13.view(-1)
    fc1_out = intermediate[: capacity_routed_rows * fc1_cols]
    activated = buffers.intermediate_cache2.view(-1)[
        : capacity_routed_rows * plan.intermediate_size
    ]
    fc2_out = (
        buffers.output.view(-1)
        if plan.fused_sum
        else intermediate[: capacity_routed_rows * HIDDEN]
    )
    stream = int(torch.cuda.current_stream().cuda_stream)
    # Invoke the already compiled object directly.  Besides excluding Python
    # dispatcher work from capture, this is required for the TC-decode
    # 512x32 FC2 specialization: SparkInfer's generic force-tile validation
    # intentionally rejects tile_k<64 even though the TC planner admits this
    # one scale-group-aligned specialization.
    rot_dummy = w4a16_kernel._rot_scales_dummy(plan.input_bf16.device)
    plan.fused.compiled(
        make_ptr(
            w4a16_kernel._cutlass_element_dtype("bf16"),
            plan.input_bf16.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            w4a16_kernel._cutlass_element_dtype("bf16"),
            plan.input_bf16.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            w4a16_kernel._cutlass_element_dtype("bf16"),
            plan.input_bf16.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        prepared.w13.view(torch.int32).view(-1),
        prepared.w2.view(torch.int32).view(-1),
        fc1_out,
        activated,
        fc2_out,
        prepared.w13_scale.view(torch.uint8).view(torch.int32).view(-1),
        prepared.w2_scale.view(torch.uint8).view(torch.int32).view(-1),
        prepared.w13_global_scale,
        prepared.w2_global_scale,
        metadata.packed_route_indices,
        metadata.block_expert_ids,
        metadata.packed_route_count,
        prepared.w13_global_scale,
        0,
        make_ptr(
            cutlass.Float32,
            metadata.topk_weights.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        buffers.fc1_c_tmp,
        buffers.fc2_c_tmp,
        prepared.workspace,
        rot_dummy,
        rot_dummy,
        rot_dummy,
        active_m,
        plan.grid_x,
        cuda.CUstream(stream),
    )
    if not plan.fused_sum:
        torch.ops.sparkinfer.w4a16_topk_sum_launch(
            fc2_out,
            buffers.output,
            active_m,
            plan.topk,
            HIDDEN,
            "bf16",
            stream,
        )
    return buffers.output[:active_m]


def capture_measurement(
    *,
    arm: str,
    replay_case: ReplayCase,
    metadata: RouteMetadata | None,
    plan: KernelPlan | None,
) -> GraphMeasurement:
    if metadata is None:
        return GraphMeasurement(
            arm=arm,
            replay_case=replay_case,
            metadata=None,
            plan=None,
            graph=None,
        )
    assert plan is not None
    launch_plan(plan, metadata, replay_case.physical_m)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch_plan(plan, metadata, replay_case.physical_m)
    graph.replay()
    torch.cuda.synchronize()
    first = plan.buffers.output[: replay_case.physical_m].clone()
    graph.replay()
    torch.cuda.synchronize()
    second = plan.buffers.output[: replay_case.physical_m].clone()
    finite = bool(torch.isfinite(second).all().item())
    nonzero = bool(torch.count_nonzero(second).item())
    return GraphMeasurement(
        arm=arm,
        replay_case=replay_case,
        metadata=metadata,
        plan=plan,
        graph=graph,
        deterministic=bool(torch.equal(first, second)),
        finite=finite,
        nonzero=nonzero,
    )


def time_graphs(
    measurements: list[GraphMeasurement],
    *,
    warmup_rounds: int,
    repeats: int,
    seed: int,
) -> None:
    active = [measurement for measurement in measurements if measurement.graph is not None]
    rng = random.Random(seed)
    for _ in range(warmup_rounds):
        order = list(active)
        rng.shuffle(order)
        for measurement in order:
            assert measurement.graph is not None
            measurement.graph.replay()
    torch.cuda.synchronize()

    for _ in range(repeats):
        order = list(active)
        rng.shuffle(order)
        timed = []
        for measurement in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            assert measurement.graph is not None
            measurement.graph.replay()
            end.record()
            timed.append((measurement, start, end))
        torch.cuda.synchronize()
        for measurement, start, end in timed:
            measurement.samples_ms.append(float(start.elapsed_time(end)))


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def measurement_record(measurement: GraphMeasurement) -> dict[str, Any]:
    metadata = measurement.metadata
    plan = measurement.plan
    record = {
        "record": "measurement",
        "arm": measurement.arm,
        "case_id": measurement.replay_case.case_id,
        "chain_id": measurement.replay_case.chain_id,
        "layer_id": measurement.replay_case.layer_id,
        "physical_m": measurement.replay_case.physical_m,
        "empty": measurement.graph is None,
        "median_ms": measurement.median_ms,
        "samples_ms": measurement.samples_ms,
        "deterministic": measurement.deterministic,
        "finite": measurement.finite,
        "nonzero": measurement.nonzero,
    }
    if metadata is not None and plan is not None:
        record.update(
            {
                "topk_capacity": metadata.topk,
                "logical_routes": metadata.logical_routes,
                "padded_routes": metadata.padded_routes,
                "active_experts": metadata.active_experts,
                "capacity_m": plan.capacity_m,
                "intermediate_size": plan.intermediate_size,
                "num_experts": plan.num_experts,
                "route_block_size": plan.block_size,
                "direct_topk": plan.direct_topk,
                "fused_sum": plan.fused_sum,
                "zero_fc2_output": plan.zero_fc2_output,
                "grid_x": plan.grid_x,
                "blocks_per_sm": plan.fused.blocks_per_sm,
                "fc1_tile_k": plan.fused.fc1_tile_k,
                "fc1_tile_n": plan.fused.fc1_tile_n,
                "fc2_tile_k": plan.fused.fc2_tile_k,
                "fc2_tile_n": plan.fused.fc2_tile_n,
            }
        )
    return record


def summarize(
    cases: list[ReplayCase], measurements: list[GraphMeasurement]
) -> list[dict[str, Any]]:
    indexed = {
        (measurement.replay_case.case_id, measurement.arm): measurement
        for measurement in measurements
    }
    rows_by_m: dict[int, list[dict[str, float]]] = defaultdict(list)
    for replay_case in cases:
        baseline_measurement = indexed[(replay_case.case_id, "tp4")]
        pair0_measurement = indexed[(replay_case.case_id, "tp2_pair0")]
        pair1_measurement = indexed[(replay_case.case_id, "tp2_pair1")]
        baseline = baseline_measurement.median_ms
        pair0 = pair0_measurement.median_ms
        pair1 = pair1_measurement.median_ms
        critical = max(pair0, pair1)
        baseline_padded = (
            0
            if baseline_measurement.metadata is None
            else baseline_measurement.metadata.padded_routes
        )
        pair_logical = tuple(
            0 if measurement.metadata is None else measurement.metadata.logical_routes
            for measurement in (pair0_measurement, pair1_measurement)
        )
        pair_padded = tuple(
            0 if measurement.metadata is None else measurement.metadata.padded_routes
            for measurement in (pair0_measurement, pair1_measurement)
        )
        comparable_padded_work = all(
            measurement.plan is None
            or baseline_measurement.plan is None
            or measurement.plan.direct_topk == baseline_measurement.plan.direct_topk
            for measurement in (pair0_measurement, pair1_measurement)
        )
        rows_by_m[replay_case.physical_m].append(
            {
                "baseline": baseline,
                "pair0": pair0,
                "pair1": pair1,
                "critical": critical,
                "speedup": baseline / critical,
                "perfect_balance_speedup": baseline
                / max((pair0 + pair1) / 2.0, 1e-12),
                "imbalance": max(pair0, pair1) / max((pair0 + pair1) / 2.0, 1e-12),
                "logical_route_imbalance": max(pair_logical)
                / max(sum(pair_logical) / 2.0, 1e-12),
                "padded_route_imbalance": max(pair_padded)
                / max(sum(pair_padded) / 2.0, 1e-12),
                "critical_padded_work_ratio": (
                    2.0 * max(pair_padded) / max(baseline_padded, 1)
                    if comparable_padded_work
                    else None
                ),
            }
        )
    records = []
    for physical_m in sorted(rows_by_m):
        rows = rows_by_m[physical_m]
        speedups = [row["speedup"] for row in rows]
        records.append(
            {
                "record": "summary",
                "physical_m": physical_m,
                "cases": len(rows),
                "tp4_median_ms": statistics.median(row["baseline"] for row in rows),
                "tp2_pair0_median_ms": statistics.median(row["pair0"] for row in rows),
                "tp2_pair1_median_ms": statistics.median(row["pair1"] for row in rows),
                "tp2_critical_median_ms": statistics.median(
                    row["critical"] for row in rows
                ),
                "speedup_median": statistics.median(speedups),
                "speedup_p05": percentile(speedups, 0.05),
                "speedup_p95": percentile(speedups, 0.95),
                "perfect_balance_speedup_median": statistics.median(
                    row["perfect_balance_speedup"] for row in rows
                ),
                "pair_timing_imbalance_mean": statistics.mean(
                    row["imbalance"] for row in rows
                ),
                "logical_route_imbalance_mean": statistics.mean(
                    row["logical_route_imbalance"] for row in rows
                ),
                "padded_route_imbalance_mean": statistics.mean(
                    row["padded_route_imbalance"] for row in rows
                ),
                "critical_padded_work_ratio_mean": (
                    statistics.mean(
                        row["critical_padded_work_ratio"]
                        for row in rows
                        if row["critical_padded_work_ratio"] is not None
                    )
                    if any(
                        row["critical_padded_work_ratio"] is not None for row in rows
                    )
                    else None
                ),
            }
        )
    return records


def run_split_correctness(device: torch.device) -> dict[str, Any]:
    """Check that two TP2 shards cover the same full-I math as four TP4 shards.

    This small gate uses one expert and M=1 so source slicing is unambiguous and
    does not consume material benchmark memory.  It validates the W13 gate/up
    and W2 intermediate axes rather than the unrelated EP routing policy.
    """

    # The performance run uses independently randomized shards; a reference
    # check over a full 2048-wide source would retain several large temporary
    # packs alongside the benchmark weights.  Use a pure BF16 identity here to
    # validate the partition algebra and leave exact packed-kernel arithmetic to
    # SparkInfer's existing W4A16 oracle suite.
    generator = torch.Generator(device=device).manual_seed(20_260_731)
    gate = torch.randn(
        (1, FULL_INTERMEDIATE), dtype=torch.bfloat16, device=device, generator=generator
    )
    up = torch.randn(
        (1, FULL_INTERMEDIATE), dtype=torch.bfloat16, device=device, generator=generator
    )
    activated = torch.nn.functional.silu(gate.float()).to(torch.bfloat16) * up
    w2 = torch.randn(
        (FULL_INTERMEDIATE, 64), dtype=torch.bfloat16, device=device, generator=generator
    )
    full = activated.float() @ w2.float()
    tp4 = sum(
        activated[:, rank * TP4_INTERMEDIATE : (rank + 1) * TP4_INTERMEDIATE].float()
        @ w2[rank * TP4_INTERMEDIATE : (rank + 1) * TP4_INTERMEDIATE].float()
        for rank in range(4)
    )
    tp2 = sum(
        activated[:, rank * TP2_INTERMEDIATE : (rank + 1) * TP2_INTERMEDIATE].float()
        @ w2[rank * TP2_INTERMEDIATE : (rank + 1) * TP2_INTERMEDIATE].float()
        for rank in range(2)
    )
    return {
        "record": "correctness",
        "gate": "full-intermediate-partition",
        "tp4_max_abs": float((full - tp4).abs().max().item()),
        "tp2_max_abs": float((full - tp2).abs().max().item()),
        "tp4_tp2_max_abs": float((tp4 - tp2).abs().max().item()),
        "passed": bool(torch.allclose(tp4, tp2, rtol=1e-5, atol=1e-4)),
        "scope": "partition algebra; packed W4A16 covered by SparkInfer oracle tests",
    }


def main() -> None:
    args = parse_args()
    requested_ms = parse_ms(args.ms)
    if min(
        args.chains_per_m,
        args.layers_per_chain,
        args.warmup_rounds,
        args.repeats,
    ) < 1:
        raise SystemExit("sample counts must be positive")
    if not args.replay_plan.is_file():
        raise SystemExit(f"replay plan does not exist: {args.replay_plan}")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if tuple(torch.cuda.get_device_capability()) != (12, 1):
        raise SystemExit(
            f"TP2/EP2 qualification must run on GB10 sm_121, got "
            f"{props.name} capability={torch.cuda.get_device_capability()}"
        )

    replay_manifest, cases, training_chains = load_cases(
        args.replay_plan,
        requested_ms,
        args.chains_per_m,
        args.layers_per_chain,
    )
    residents_by_layer, placement_metadata = make_pair_residency(
        args.tp2_placement,
        training_chains,
    )
    del training_chains
    device = torch.device("cuda", torch.cuda.current_device())
    sms = int(props.multi_processor_count)
    max_shared_mem = int(props.shared_memory_per_block_optin)
    started = time.time()

    correctness = None if args.skip_correctness else run_split_correctness(device)
    if correctness is not None and not correctness["passed"]:
        raise RuntimeError(f"TP partition correctness gate failed: {correctness}")

    # Three resident sets prevent artificial pair-to-pair cache sharing on the
    # single benchmark GPU.  Each TP2 pair receives its own physical weights,
    # just as it would on separate nodes.
    tp4_weights = make_prepared_weights(
        num_experts=EXPERTS,
        intermediate_size=TP4_INTERMEDIATE,
        seed=args.seed,
        device=device,
    )
    tp2_weights = (
        make_prepared_weights(
            num_experts=TP2_EXPERTS,
            intermediate_size=TP2_INTERMEDIATE,
            seed=args.seed + 1,
            device=device,
        ),
        make_prepared_weights(
            num_experts=TP2_EXPERTS,
            intermediate_size=TP2_INTERMEDIATE,
            seed=args.seed + 2,
            device=device,
        ),
    )
    torch.cuda.synchronize()

    plans: dict[tuple[str, int, int], KernelPlan] = {}

    def resolve_plan(
        arm: str,
        replay_case: ReplayCase,
        metadata: RouteMetadata,
    ) -> KernelPlan:
        key = (arm, replay_case.physical_m, metadata.topk)
        plan = plans.get(key)
        if plan is not None:
            return plan
        if arm == "tp4":
            prepared = tp4_weights
            baseline = True
        else:
            pair = int(arm[-1])
            prepared = tp2_weights[pair]
            baseline = False
        print(
            json.dumps(
                {
                    "record": "compile",
                    "arm": arm,
                    "physical_m": replay_case.physical_m,
                    "capacity_m": capacity_for_m(replay_case.physical_m),
                    "topk": metadata.topk,
                    "intermediate_size": int(prepared.intermediate_size),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        plan = make_plan(
            label=arm,
            prepared=prepared,
            m=replay_case.physical_m,
            topk=metadata.topk,
            baseline=baseline,
            sms=sms,
            max_shared_mem=max_shared_mem,
            device=device,
            tp2_small_m_kernel=args.tp2_small_m_kernel,
            tp2_large_m_block_size=args.tp2_large_m_block_size,
            tp2_no_zero_output_upper_bound=args.tp2_no_zero_output_upper_bound,
        )
        plans[key] = plan
        return plan

    measurements: list[GraphMeasurement] = []
    compact = args.tp2_small_m_routing == "compact"
    for case_index, replay_case in enumerate(cases, start=1):
        baseline_metadata = build_route_metadata(
            replay_case.routes,
            num_experts=EXPERTS,
            block_size=8 if replay_case.physical_m <= 8 else 32,
            direct_topk=replay_case.physical_m == 1,
            device=device,
        )
        assert baseline_metadata is not None
        baseline_plan = resolve_plan("tp4", replay_case, baseline_metadata)
        measurements.append(
            capture_measurement(
                arm="tp4",
                replay_case=replay_case,
                metadata=baseline_metadata,
                plan=baseline_plan,
            )
        )
        for pair in (0, 1):
            pair_rows = local_pair_routes(
                replay_case,
                pair,
                compact=compact and replay_case.physical_m <= 8,
                residents_by_layer=residents_by_layer,
            )
            pair_direct = replay_case.physical_m == 1 or (
                replay_case.physical_m <= 8
                and args.tp2_small_m_kernel == "direct"
            )
            pair_metadata = build_route_metadata(
                pair_rows,
                num_experts=TP2_EXPERTS,
                block_size=(
                    8
                    if replay_case.physical_m <= 8
                    else args.tp2_large_m_block_size
                ),
                direct_topk=pair_direct,
                device=device,
            )
            arm = f"tp2_pair{pair}"
            pair_plan = (
                None
                if pair_metadata is None
                else resolve_plan(arm, replay_case, pair_metadata)
            )
            measurements.append(
                capture_measurement(
                    arm=arm,
                    replay_case=replay_case,
                    metadata=pair_metadata,
                    plan=pair_plan,
                )
            )
        if case_index % max(1, args.layers_per_chain * args.chains_per_m) == 0:
            print(
                json.dumps(
                    {
                        "record": "capture_progress",
                        "physical_m": replay_case.physical_m,
                        "cases_captured": case_index,
                        "total_cases": len(cases),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    time_graphs(
        measurements,
        warmup_rounds=args.warmup_rounds,
        repeats=args.repeats,
        seed=args.seed,
    )
    summaries = summarize(cases, measurements)
    if any(not measurement.finite for measurement in measurements):
        raise RuntimeError("one or more graph outputs were non-finite")
    if any(
        measurement.graph is not None and not measurement.nonzero
        for measurement in measurements
    ):
        raise RuntimeError("one or more nonempty graph outputs were all zero")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    manifest = {
        "record": "manifest",
        "schema": "glmrt-tp2-ep2-w4a16-route-replay-v1",
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "root_revision": git_revision(Path(__file__).resolve().parents[2]),
        "sparkinfer_revision": _pinned_sparkinfer.REVISION,
        "sparkinfer_version": _pinned_sparkinfer.VERSION,
        "replay_plan": str(args.replay_plan.resolve()),
        "replay_plan_schema": replay_manifest.get("schema"),
        "requested_ms": requested_ms,
        "chains_per_m": args.chains_per_m,
        "layers_per_chain": args.layers_per_chain,
        "warmup_rounds": args.warmup_rounds,
        "repeats": args.repeats,
        "seed": args.seed,
        "network_involved": False,
        "route_pack_timed": False,
        "timing": "CUDA graph replay with CUDA events; arms interleaved",
        "tp4": {
            "experts_per_node": EXPERTS,
            "intermediate_size": TP4_INTERMEDIATE,
            "small_m": "M1 fused direct; M2-8 production grouped-wide fixed-order",
        },
        "tp2_ep2": {
            "experts_per_node": TP2_EXPERTS,
            "intermediate_size": TP2_INTERMEDIATE,
            "placement": args.tp2_placement,
            "placement_metadata": placement_metadata,
            "small_m_routing": args.tp2_small_m_routing,
            "small_m_kernel": args.tp2_small_m_kernel,
            "large_m_block_size": args.tp2_large_m_block_size,
            "zero_partial_route_output": not args.tp2_no_zero_output_upper_bound,
            "valid_output": not args.tp2_no_zero_output_upper_bound,
            "charged_latency": "max(pair0,pair1)",
        },
        "gpu": {
            "name": props.name,
            "capability": list(torch.cuda.get_device_capability()),
            "sms": sms,
            "max_shared_mem": max_shared_mem,
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    with temporary.open("w", encoding="utf-8") as output:
        output.write(json.dumps(manifest, separators=(",", ":")) + "\n")
        if correctness is not None:
            output.write(json.dumps(correctness, separators=(",", ":")) + "\n")
        for measurement in measurements:
            output.write(
                json.dumps(measurement_record(measurement), separators=(",", ":"))
                + "\n"
            )
        for record in summaries:
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
    temporary.replace(args.output)
    for record in summaries:
        print(json.dumps(record, sort_keys=True), flush=True)
    print(
        json.dumps(
            {
                "record": "complete",
                "output": str(args.output),
                "measurements": len(measurements),
                "elapsed_seconds": time.time() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
