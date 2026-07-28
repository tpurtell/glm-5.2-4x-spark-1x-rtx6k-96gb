from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import triton
import triton.language as tl

from b12x_mla_capture import _bf16_tensor, _i32_tensor, _u8_tensor
from dspark_capture import _fp8_tensor, _i64_tensor


_DSPARK_BODY_STATES: dict[tuple[Any, ...], "_DsparkBodyState"] = {}


@dataclass(frozen=True)
class _DsparkBodyLayerWeights:
    input_norm: Any
    post_norm: Any
    q_norm: Any
    k_norm: Any
    qkv_t: Any
    output_t: Any
    gate_up_t: Any
    down_t: Any


@dataclass(frozen=True)
class _DsparkBodyState:
    device_id: int
    cuda_stream: int
    layers: int
    active_requests: int
    query_rows: int
    total_rows: int
    total_pages: int
    page_size: int
    max_pages_per_request: int
    hidden_size: int
    intermediate_size: int
    heads: int
    head_dim: int
    cache_dtype: str
    input: Any
    output: Any
    reference_output: Any
    hidden_attention: Any
    hidden_mlp: Any
    normalized: Any
    qkv: Any
    q: Any
    attention: Any
    attention_flat: Any
    delta: Any
    gate_up: Any
    activation: Any
    k_cache: Any
    v_cache: Any
    workspace: Any
    query_lengths: Any
    kv_lengths: Any
    query_positions: Any
    block_tables: Any
    query_offsets: Any
    output_offsets: Any
    query_indptr: Any
    kv_indptr: Any
    page_indices: Any
    last_page_len: Any
    flashinfer_wrapper: Any | None
    final_norm: Any
    layer_weights: tuple[_DsparkBodyLayerWeights, ...]


def prepare_dspark_cudnn_paged_body(ctx: dict[str, Any], **kwargs: Any) -> None:
    """Bind, initialize, compile, and warm one fixed-address dSpark body."""

    state = _body_state(ctx, kwargs, create=True)
    _run_body(state)
    import torch

    stream = torch.cuda.ExternalStream(state.cuda_stream, device=state.device_id)
    with torch.cuda.device(state.device_id), torch.cuda.stream(stream), torch.no_grad():
        state.reference_output.copy_(state.output)


def capture_dspark_cudnn_paged_body(ctx: dict[str, Any], **kwargs: Any) -> None:
    """Launch the allocation-free active draft-layer body during capture."""

    _run_body(_body_state(ctx, kwargs, create=False))


def _body_state(
    ctx: dict[str, Any], kwargs: dict[str, Any], *, create: bool
) -> _DsparkBodyState:
    layers = int(kwargs["layers"])
    active_requests = int(kwargs["active_requests"])
    query_rows = int(kwargs["query_rows"])
    total_pages = int(kwargs["total_pages"])
    page_size = int(kwargs["page_size"])
    max_pages_per_request = int(kwargs["max_pages_per_request"])
    hidden_size = int(kwargs["hidden_size"])
    intermediate_size = int(kwargs["intermediate_size"])
    heads = int(kwargs["heads"])
    head_dim = int(kwargs["head_dim"])
    seed = int(kwargs["seed"])
    initialize_input = bool(kwargs.get("initialize_input", True))
    initialize_kv = bool(kwargs.get("initialize_kv", True))
    cache_dtype = str(kwargs.get("cache_dtype", "bf16")).lower()

    if layers not in (3, 5):
        raise ValueError(
            f"supported GLM-5.2 dSpark checkpoints require three or five layers, got {layers}"
        )
    if active_requests not in (1, 2, 4):
        raise ValueError(
            "dSpark body active_requests must be one of 1, 2, or 4, "
            f"got {active_requests}"
        )
    if query_rows not in (8, 16):
        raise ValueError(f"dSpark body query_rows must be 8 or 16, got {query_rows}")
    if page_size not in (16, 32, 64, 128):
        raise ValueError(f"unsupported dSpark body page size {page_size}")
    if hidden_size != 6144 or intermediate_size != 12288:
        raise ValueError(
            "GLM-5.2 dSpark body requires hidden/intermediate 6144/12288, "
            f"got {hidden_size}/{intermediate_size}"
        )
    if heads != 64 or head_dim != 64:
        raise ValueError(
            "GLM-5.2 dSpark body requires 64 heads of width 64, "
            f"got {heads}x{head_dim}"
        )
    if total_pages < active_requests or max_pages_per_request < 1:
        raise ValueError("dSpark body page counts are invalid")
    if cache_dtype not in ("bf16", "fp8"):
        raise ValueError(f"unsupported dSpark body cache dtype {cache_dtype}")

    buffers = ctx["buffers"]
    mutable_names = (
        "input",
        "output",
        "reference_output",
        "hidden_attention",
        "hidden_mlp",
        "normalized",
        "qkv",
        "q",
        "attention",
        "delta",
        "gate_up",
        "activation",
        "k_cache",
        "v_cache",
        "workspace",
        "query_lengths",
        "kv_lengths",
        "query_positions",
        "block_tables",
        "query_offsets",
        "output_offsets",
        "query_indptr",
        "kv_indptr",
        "page_indices",
        "last_page_len",
    )
    weight_names = ["final_norm"]
    for layer in range(layers):
        weight_names.extend(
            (
                f"layer_{layer}_input_norm",
                f"layer_{layer}_post_norm",
                f"layer_{layer}_q_norm",
                f"layer_{layer}_k_norm",
                f"layer_{layer}_qkv",
                f"layer_{layer}_output",
                f"layer_{layer}_gate_up",
                f"layer_{layer}_down",
            )
        )
    required_names = (*mutable_names, *weight_names)
    missing = [name for name in required_names if name not in buffers]
    if missing:
        raise ValueError(f"dSpark body is missing buffers: {missing}")

    device_id = int(buffers["input"]["device_id"])
    for name in required_names:
        if int(buffers[name]["device_id"]) != device_id:
            raise ValueError(f"dSpark body buffer {name} is on another CUDA device")

    total_rows = active_requests * query_rows
    attention_width = heads * head_dim
    shapes = {
        "input": (total_rows, hidden_size),
        "output": (total_rows, hidden_size),
        "reference_output": (total_rows, hidden_size),
        "hidden_attention": (total_rows, hidden_size),
        "hidden_mlp": (total_rows, hidden_size),
        "normalized": (total_rows, hidden_size),
        "qkv": (total_rows, 3 * attention_width),
        "q": (total_rows, heads, head_dim),
        "attention": (total_rows, heads, head_dim),
        "delta": (total_rows, hidden_size),
        "gate_up": (total_rows, 2 * intermediate_size),
        "activation": (total_rows, intermediate_size),
        "k_cache": (layers, total_pages, heads, page_size, head_dim),
        "v_cache": (layers, total_pages, heads, page_size, head_dim),
        "workspace": (int(buffers["workspace"]["bytes"]),),
        "query_lengths": (active_requests,),
        "kv_lengths": (active_requests,),
        "query_positions": (total_rows,),
        "block_tables": (active_requests, max_pages_per_request),
        "query_offsets": (active_requests + 1,),
        "output_offsets": (active_requests + 1,),
        "query_indptr": (active_requests + 1,),
        "kv_indptr": (active_requests + 1,),
        "page_indices": (total_pages,),
        "last_page_len": (active_requests,),
        "final_norm": (hidden_size,),
    }
    for name, shape in shapes.items():
        element_bytes = 1 if name == "workspace" else 2
        if name in (
            "query_lengths",
            "kv_lengths",
            "query_positions",
            "block_tables",
            "query_indptr",
            "kv_indptr",
            "page_indices",
            "last_page_len",
        ):
            element_bytes = 4
        elif name in ("query_offsets", "output_offsets"):
            element_bytes = 8
        elif name in ("k_cache", "v_cache") and cache_dtype == "fp8":
            element_bytes = 1
        required_bytes = _tensor_bytes(shape, element_bytes)
        if int(buffers[name]["bytes"]) < required_bytes:
            raise ValueError(
                f"dSpark body buffer {name} has {buffers[name]['bytes']} bytes, "
                f"requires {required_bytes}"
            )

    key = (
        int(ctx["cuda_stream"]),
        layers,
        active_requests,
        query_rows,
        total_pages,
        page_size,
        max_pages_per_request,
        hidden_size,
        intermediate_size,
        heads,
        head_dim,
        cache_dtype,
        initialize_input,
        initialize_kv,
        *((name, int(buffers[name]["ptr"])) for name in required_names),
    )
    state = _DSPARK_BODY_STATES.get(key)
    if state is not None:
        return state
    if not create:
        raise RuntimeError("dSpark body capture requires a matching startup prepare call")

    import torch

    cuda_stream = int(ctx["cuda_stream"])
    stream = torch.cuda.ExternalStream(cuda_stream, device=device_id)
    with torch.cuda.device(device_id), torch.cuda.stream(stream), torch.no_grad():
        tensors: dict[str, Any] = {}
        for name in (
            "input",
            "output",
            "reference_output",
            "hidden_attention",
            "hidden_mlp",
            "normalized",
            "qkv",
            "q",
            "attention",
            "delta",
            "gate_up",
            "activation",
            "final_norm",
        ):
            tensors[name] = _bf16_tensor(buffers[name], shapes[name])
        cache_tensor = _bf16_tensor if cache_dtype == "bf16" else _fp8_tensor
        tensors["k_cache"] = cache_tensor(buffers["k_cache"], shapes["k_cache"])
        tensors["v_cache"] = cache_tensor(buffers["v_cache"], shapes["v_cache"])
        workspace = _u8_tensor(buffers["workspace"], shapes["workspace"])
        query_lengths = _i32_tensor(
            buffers["query_lengths"], shapes["query_lengths"]
        )
        kv_lengths = _i32_tensor(buffers["kv_lengths"], shapes["kv_lengths"])
        query_positions = _i32_tensor(
            buffers["query_positions"], shapes["query_positions"]
        )
        block_tables = _i32_tensor(
            buffers["block_tables"], shapes["block_tables"]
        )
        query_offsets = _i64_tensor(
            buffers["query_offsets"], shapes["query_offsets"]
        )
        output_offsets = _i64_tensor(
            buffers["output_offsets"], shapes["output_offsets"]
        )
        query_indptr = _i32_tensor(buffers["query_indptr"], shapes["query_indptr"])
        kv_indptr = _i32_tensor(buffers["kv_indptr"], shapes["kv_indptr"])
        page_indices = _i32_tensor(
            buffers["page_indices"], shapes["page_indices"]
        )
        last_page_len = _i32_tensor(
            buffers["last_page_len"], shapes["last_page_len"]
        )

        layer_weights = []
        for layer in range(layers):
            prefix = f"layer_{layer}"
            input_norm = _bf16_tensor(buffers[f"{prefix}_input_norm"], (hidden_size,))
            post_norm = _bf16_tensor(buffers[f"{prefix}_post_norm"], (hidden_size,))
            q_norm = _bf16_tensor(buffers[f"{prefix}_q_norm"], (head_dim,))
            k_norm = _bf16_tensor(buffers[f"{prefix}_k_norm"], (head_dim,))
            qkv = _bf16_tensor(
                buffers[f"{prefix}_qkv"], (3 * attention_width, hidden_size)
            )
            output = _bf16_tensor(
                buffers[f"{prefix}_output"], (hidden_size, attention_width)
            )
            gate_up = _bf16_tensor(
                buffers[f"{prefix}_gate_up"],
                (2 * intermediate_size, hidden_size),
            )
            down = _bf16_tensor(
                buffers[f"{prefix}_down"], (hidden_size, intermediate_size)
            )
            layer_weights.append(
                _DsparkBodyLayerWeights(
                    input_norm=input_norm,
                    post_norm=post_norm,
                    q_norm=q_norm,
                    k_norm=k_norm,
                    qkv_t=qkv.t(),
                    output_t=output.t(),
                    gate_up_t=gate_up.t(),
                    down_t=down.t(),
                )
            )

        query_lengths.fill_(query_rows)
        element_stride = query_rows * heads * head_dim
        offsets = torch.arange(
            active_requests + 1, dtype=torch.int64, device=device_id
        ) * element_stride
        query_offsets.copy_(offsets)
        output_offsets.copy_(offsets)

        if initialize_input or initialize_kv:
            generator = torch.Generator(device=device_id)
            generator.manual_seed(seed)
            if initialize_input:
                tensors["input"].normal_(generator=generator)
            if initialize_kv:
                if cache_dtype == "bf16":
                    tensors["k_cache"].normal_(generator=generator)
                    tensors["v_cache"].normal_(generator=generator)
                else:
                    for cache in (tensors["k_cache"], tensors["v_cache"]):
                        source = torch.empty(
                            cache.shape,
                            dtype=torch.bfloat16,
                            device=cache.device,
                        )
                        source.normal_(generator=generator)
                        cache.copy_(source.to(torch.float8_e4m3fn))

        flashinfer_wrapper = None
        if cache_dtype == "fp8":
            from flashinfer import BatchPrefillWithPagedKVCacheWrapper

            runtime_query_indptr = query_indptr.clone()
            runtime_kv_indptr = kv_indptr.clone()
            runtime_page_indices = page_indices.clone()
            runtime_last_page_len = last_page_len.clone()
            physical_pages_per_request = total_pages // active_requests
            planning_query_indptr = torch.arange(
                active_requests + 1, dtype=torch.int32, device=device_id
            ) * query_rows
            planning_kv_indptr = torch.arange(
                active_requests + 1, dtype=torch.int32, device=device_id
            ) * physical_pages_per_request
            planning_page_indices = torch.arange(
                total_pages, dtype=torch.int32, device=device_id
            )
            planning_last_page_len = torch.full(
                (active_requests,), page_size, dtype=torch.int32, device=device_id
            )
            flashinfer_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                workspace,
                kv_layout="HND",
                use_cuda_graph=True,
                qo_indptr_buf=query_indptr,
                paged_kv_indptr_buf=kv_indptr,
                paged_kv_indices_buf=page_indices,
                paged_kv_last_page_len_buf=last_page_len,
                backend="auto",
            )
            flashinfer_wrapper.plan(
                planning_query_indptr,
                planning_kv_indptr,
                planning_page_indices,
                planning_last_page_len,
                heads,
                heads,
                head_dim,
                page_size,
                causal=False,
                sm_scale=1.0 / math.sqrt(head_dim),
                q_data_type=torch.bfloat16,
                kv_data_type=torch.float8_e4m3fn,
                o_data_type=torch.bfloat16,
            )
            query_indptr.copy_(runtime_query_indptr)
            kv_indptr.copy_(runtime_kv_indptr)
            page_indices.copy_(runtime_page_indices)
            last_page_len.copy_(runtime_last_page_len)

    state = _DsparkBodyState(
        device_id=device_id,
        cuda_stream=cuda_stream,
        layers=layers,
        active_requests=active_requests,
        query_rows=query_rows,
        total_rows=total_rows,
        total_pages=total_pages,
        page_size=page_size,
        max_pages_per_request=max_pages_per_request,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        heads=heads,
        head_dim=head_dim,
        cache_dtype=cache_dtype,
        input=tensors["input"],
        output=tensors["output"],
        reference_output=tensors["reference_output"],
        hidden_attention=tensors["hidden_attention"],
        hidden_mlp=tensors["hidden_mlp"],
        normalized=tensors["normalized"],
        qkv=tensors["qkv"],
        q=tensors["q"],
        attention=tensors["attention"],
        attention_flat=tensors["attention"].view(total_rows, attention_width),
        delta=tensors["delta"],
        gate_up=tensors["gate_up"],
        activation=tensors["activation"],
        k_cache=tensors["k_cache"],
        v_cache=tensors["v_cache"],
        workspace=workspace,
        query_lengths=query_lengths,
        kv_lengths=kv_lengths,
        query_positions=query_positions,
        block_tables=block_tables,
        query_offsets=query_offsets,
        output_offsets=output_offsets,
        query_indptr=query_indptr,
        kv_indptr=kv_indptr,
        page_indices=page_indices,
        last_page_len=last_page_len,
        flashinfer_wrapper=flashinfer_wrapper,
        final_norm=tensors["final_norm"],
        layer_weights=tuple(layer_weights),
    )
    _DSPARK_BODY_STATES[key] = state
    return state


def _run_body(state: _DsparkBodyState) -> None:
    import torch

    stream = torch.cuda.ExternalStream(state.cuda_stream, device=state.device_id)
    max_sequence_kv = state.max_pages_per_request * state.page_size
    scale = 1.0 / math.sqrt(state.head_dim)
    hidden = state.input
    with torch.cuda.device(state.device_id), torch.cuda.stream(stream), torch.no_grad():
        for layer_index, weights in enumerate(state.layer_weights):
            _dspark_rms_norm[(state.total_rows,)](
                hidden,
                weights.input_norm,
                state.normalized,
                WIDTH=state.hidden_size,
                BLOCK=triton.next_power_of_2(state.hidden_size),
                EPSILON=1.0e-5,
                num_warps=8,
            )
            torch.mm(state.normalized, weights.qkv_t, out=state.qkv)
            _dspark_qkv_rope_append[(state.total_rows * state.heads,)](
                state.qkv,
                weights.q_norm,
                weights.k_norm,
                state.kv_lengths,
                state.query_positions,
                state.block_tables,
                state.q,
                state.k_cache[layer_index],
                state.v_cache[layer_index],
                QUERY_ROWS=state.query_rows,
                HEADS=state.heads,
                HEAD_DIM=state.head_dim,
                PAGE_SIZE=state.page_size,
                MAX_PAGES=state.max_pages_per_request,
                QKV_WIDTH=3 * state.heads * state.head_dim,
                THETA=8_000_000.0,
                EPSILON=1.0e-5,
                BLOCK=state.head_dim,
                num_warps=4,
            )
            if state.cache_dtype == "fp8":
                if state.flashinfer_wrapper is None:
                    raise RuntimeError("dSpark FP8 body wrapper is not initialized")
                state.flashinfer_wrapper.run(
                    state.q,
                    (state.k_cache[layer_index], state.v_cache[layer_index]),
                    out=state.attention,
                )
            else:
                from flashinfer.cudnn.prefill import (
                    cudnn_batch_prefill_with_kv_cache,
                )

                cudnn_batch_prefill_with_kv_cache(
                    state.q,
                    state.k_cache[layer_index],
                    state.v_cache[layer_index],
                    scale,
                    state.workspace,
                    max_token_per_sequence=state.query_rows,
                    max_sequence_kv=max_sequence_kv,
                    actual_seq_lens_q=state.query_lengths,
                    actual_seq_lens_kv=state.kv_lengths,
                    block_tables=state.block_tables,
                    causal=False,
                    return_lse=False,
                    batch_offsets_q=state.query_offsets,
                    batch_offsets_o=state.output_offsets,
                    out=state.attention,
                    is_cuda_graph_compatible=True,
                )
            torch.mm(state.attention_flat, weights.output_t, out=state.delta)
            _dspark_add[(triton.cdiv(state.total_rows * state.hidden_size, 256),)](
                hidden,
                state.delta,
                state.hidden_attention,
                TOTAL=state.total_rows * state.hidden_size,
                BLOCK=256,
            )
            _dspark_rms_norm[(state.total_rows,)](
                state.hidden_attention,
                weights.post_norm,
                state.normalized,
                WIDTH=state.hidden_size,
                BLOCK=triton.next_power_of_2(state.hidden_size),
                EPSILON=1.0e-5,
                num_warps=8,
            )
            torch.mm(state.normalized, weights.gate_up_t, out=state.gate_up)
            _dspark_silu_mul[(triton.cdiv(state.total_rows * state.intermediate_size, 256),)](
                state.gate_up,
                state.activation,
                ROWS=state.total_rows,
                INTERMEDIATE=state.intermediate_size,
                BLOCK=256,
            )
            torch.mm(state.activation, weights.down_t, out=state.delta)
            _dspark_add[(triton.cdiv(state.total_rows * state.hidden_size, 256),)](
                state.hidden_attention,
                state.delta,
                state.hidden_mlp,
                TOTAL=state.total_rows * state.hidden_size,
                BLOCK=256,
            )
            hidden = state.hidden_mlp

        _dspark_rms_norm[(state.total_rows,)](
            hidden,
            state.final_norm,
            state.output,
            WIDTH=state.hidden_size,
            BLOCK=triton.next_power_of_2(state.hidden_size),
            EPSILON=1.0e-5,
            num_warps=8,
        )


@triton.jit
def _dspark_rms_norm(
    source,
    weight,
    output,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    EPSILON: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < WIDTH
    values = tl.load(source + row * WIDTH + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / WIDTH
    inverse_rms = tl.rsqrt(variance + EPSILON)
    scales = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(output + row * WIDTH + offsets, values * inverse_rms * scales, mask=mask)


@triton.jit
def _dspark_add(
    left,
    right,
    output,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    left_values = tl.load(left + offsets, mask=mask, other=0.0)
    right_values = tl.load(right + offsets, mask=mask, other=0.0)
    tl.store(output + offsets, left_values + right_values, mask=mask)


@triton.jit
def _dspark_silu_mul(
    gate_up,
    activation,
    ROWS: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = ROWS * INTERMEDIATE
    mask = offsets < total
    row = offsets // INTERMEDIATE
    column = offsets - row * INTERMEDIATE
    gate_index = row * (2 * INTERMEDIATE) + column
    gate = tl.load(gate_up + gate_index, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(
        gate_up + gate_index + INTERMEDIATE, mask=mask, other=0.0
    ).to(tl.float32)
    tl.store(activation + offsets, gate * tl.sigmoid(gate) * up, mask=mask)


@triton.jit
def _dspark_qkv_rope_append(
    qkv,
    q_norm,
    k_norm,
    kv_lengths,
    query_positions,
    block_tables,
    q_output,
    k_cache,
    v_cache,
    QUERY_ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    QKV_WIDTH: tl.constexpr,
    THETA: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    program = tl.program_id(0)
    token_row = program // HEADS
    head = program - token_row * HEADS
    request = token_row // QUERY_ROWS
    query_row = token_row - request * QUERY_ROWS
    offsets = tl.arange(0, BLOCK)
    mask = offsets < HEAD_DIM

    row_base = token_row * QKV_WIDTH
    head_base = head * HEAD_DIM
    q_values = tl.load(qkv + row_base + head_base + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    k_values = tl.load(
        qkv + row_base + HEADS * HEAD_DIM + head_base + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    q_variance = tl.sum(q_values * q_values, axis=0) / HEAD_DIM
    k_variance = tl.sum(k_values * k_values, axis=0) / HEAD_DIM
    q_scales = tl.load(q_norm + offsets, mask=mask, other=0.0).to(tl.float32)
    k_scales = tl.load(k_norm + offsets, mask=mask, other=0.0).to(tl.float32)
    q_values = q_values * tl.rsqrt(q_variance + EPSILON) * q_scales
    k_values = k_values * tl.rsqrt(k_variance + EPSILON) * k_scales

    half = HEAD_DIM // 2
    pair = offsets % half
    paired_offsets = tl.where(offsets < half, offsets + half, offsets - half)
    q_pair = tl.load(
        qkv + row_base + head_base + paired_offsets, mask=mask, other=0.0
    ).to(tl.float32)
    k_pair = tl.load(
        qkv + row_base + HEADS * HEAD_DIM + head_base + paired_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    q_pair_scales = tl.load(q_norm + paired_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    k_pair_scales = tl.load(k_norm + paired_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    q_pair = q_pair * tl.rsqrt(q_variance + EPSILON) * q_pair_scales
    k_pair = k_pair * tl.rsqrt(k_variance + EPSILON) * k_pair_scales

    kv_length = tl.load(kv_lengths + request)
    position = tl.load(query_positions + token_row)
    frequency = tl.exp((-math.log(THETA) * (2.0 * pair)) / HEAD_DIM)
    angle = position.to(tl.float32) * frequency
    cosine = tl.cos(angle)
    sine = tl.sin(angle)
    q_rotated = tl.where(
        offsets < half,
        q_values * cosine - q_pair * sine,
        q_values * cosine + q_pair * sine,
    )
    k_rotated = tl.where(
        offsets < half,
        k_values * cosine - k_pair * sine,
        k_values * cosine + k_pair * sine,
    )
    tl.store(q_output + token_row * HEADS * HEAD_DIM + head_base + offsets, q_rotated)

    cache_position = kv_length - QUERY_ROWS + query_row
    logical_page = cache_position // PAGE_SIZE
    page_offset = cache_position - logical_page * PAGE_SIZE
    physical_page = tl.load(block_tables + request * MAX_PAGES + logical_page)
    cache_base = (
        ((physical_page * HEADS + head) * PAGE_SIZE + page_offset) * HEAD_DIM
    )
    tl.store(k_cache + cache_base + offsets, k_rotated)
    v_values = tl.load(
        qkv + row_base + 2 * HEADS * HEAD_DIM + head_base + offsets,
        mask=mask,
        other=0.0,
    )
    tl.store(v_cache + cache_base + offsets, v_values)


def _tensor_bytes(shape: tuple[int, ...], element_bytes: int) -> int:
    values = 1
    for dim in shape:
        values *= int(dim)
    return values * element_bytes
