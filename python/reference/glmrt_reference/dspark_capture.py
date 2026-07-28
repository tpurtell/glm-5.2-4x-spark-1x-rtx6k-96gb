from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from b12x_mla_capture import _bf16_tensor, _dlpack_bridge, _i32_tensor, _u8_tensor


_DLPACK_CODE_INT = 0
_DSPARK_ATTENTION_STATES: dict[tuple[Any, ...], "_DsparkAttentionState"] = {}


@dataclass(frozen=True)
class _DsparkAttentionState:
    device_id: int
    cuda_stream: int
    layers: int
    active_requests: int
    query_rows: int
    total_pages: int
    page_size: int
    max_pages_per_request: int
    heads: int
    head_dim: int
    cache_dtype: str
    q: Any
    k_cache: Any
    v_cache: Any
    output: Any
    workspace: Any
    query_lengths: Any
    kv_lengths: Any
    block_tables: Any
    query_offsets: Any
    output_offsets: Any
    query_indptr: Any
    kv_indptr: Any
    page_indices: Any
    last_page_len: Any
    flashinfer_wrapper: Any | None


def prepare_dspark_cudnn_paged_attention(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    """Compile and warm dSpark's dynamic-length paged attention graph."""

    state = _attention_state(ctx, kwargs, create=True)
    _run_attention(state)


def capture_dspark_cudnn_paged_attention(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    """Launch allocation-free dSpark attention during external graph capture."""

    state = _attention_state(ctx, kwargs, create=False)
    _run_attention(state)


def _attention_state(
    ctx: dict[str, Any], kwargs: dict[str, Any], *, create: bool
) -> _DsparkAttentionState:
    layers = int(kwargs["layers"])
    active_requests = int(kwargs["active_requests"])
    query_rows = int(kwargs["query_rows"])
    total_pages = int(kwargs["total_pages"])
    page_size = int(kwargs["page_size"])
    max_pages_per_request = int(kwargs["max_pages_per_request"])
    heads = int(kwargs["heads"])
    head_dim = int(kwargs["head_dim"])
    cache_dtype = str(kwargs.get("cache_dtype", "bf16")).lower()

    if layers < 1 or layers > 5:
        raise ValueError(f"dSpark attention layers must be in [1, 5], got {layers}")
    if active_requests not in (1, 2, 4):
        raise ValueError(
            "dSpark attention active_requests must be one of 1, 2, or 4, "
            f"got {active_requests}"
        )
    if query_rows not in (8, 16):
        raise ValueError(f"dSpark attention query_rows must be 8 or 16, got {query_rows}")
    if page_size not in (16, 32, 64, 128):
        raise ValueError(
            "dSpark attention page_size must be one of 16, 32, 64, or 128, "
            f"got {page_size}"
        )
    if total_pages < 1 or max_pages_per_request < 1:
        raise ValueError("dSpark attention page counts must be positive")
    if heads != 64 or head_dim != 64:
        raise ValueError(
            "GLM-5.2 dSpark attention requires heads=64 and head_dim=64, "
            f"got heads={heads} head_dim={head_dim}"
        )
    if cache_dtype not in ("bf16", "fp8"):
        raise ValueError(f"unsupported dSpark attention cache dtype {cache_dtype}")

    buffers = ctx["buffers"]
    required_buffers = (
        "q",
        "k_cache",
        "v_cache",
        "output",
        "workspace",
        "query_lengths",
        "kv_lengths",
        "block_tables",
        "query_offsets",
        "output_offsets",
        "query_indptr",
        "kv_indptr",
        "page_indices",
        "last_page_len",
    )
    missing = [name for name in required_buffers if name not in buffers]
    if missing:
        raise ValueError(f"dSpark attention is missing buffers: {missing}")

    device_id = int(buffers["q"]["device_id"])
    for name in required_buffers:
        if int(buffers[name]["device_id"]) != device_id:
            raise ValueError(
                f"dSpark attention buffer {name} is on a different CUDA device"
            )

    total_query_rows = active_requests * query_rows
    q_shape = (layers, total_query_rows, heads, head_dim)
    kv_shape = (layers, total_pages, heads, page_size, head_dim)
    output_shape = q_shape
    workspace_shape = (int(buffers["workspace"]["bytes"]),)
    lengths_shape = (active_requests,)
    block_tables_shape = (active_requests, max_pages_per_request)
    offsets_shape = (active_requests + 1,)
    indptr_shape = (active_requests + 1,)
    page_indices_shape = (total_pages,)
    last_page_len_shape = (active_requests,)
    cache_element_bytes = 2 if cache_dtype == "bf16" else 1
    expected_bytes = {
        "q": _tensor_bytes(q_shape, 2),
        "k_cache": _tensor_bytes(kv_shape, cache_element_bytes),
        "v_cache": _tensor_bytes(kv_shape, cache_element_bytes),
        "output": _tensor_bytes(output_shape, 2),
        "workspace": _tensor_bytes(workspace_shape, 1),
        "query_lengths": _tensor_bytes(lengths_shape, 4),
        "kv_lengths": _tensor_bytes(lengths_shape, 4),
        "block_tables": _tensor_bytes(block_tables_shape, 4),
        "query_offsets": _tensor_bytes(offsets_shape, 8),
        "output_offsets": _tensor_bytes(offsets_shape, 8),
        "query_indptr": _tensor_bytes(indptr_shape, 4),
        "kv_indptr": _tensor_bytes(indptr_shape, 4),
        "page_indices": _tensor_bytes(page_indices_shape, 4),
        "last_page_len": _tensor_bytes(last_page_len_shape, 4),
    }
    for name, minimum in expected_bytes.items():
        if int(buffers[name]["bytes"]) < minimum:
            raise ValueError(
                f"dSpark attention buffer {name} has {buffers[name]['bytes']} bytes, "
                f"requires at least {minimum}"
            )

    cuda_stream = int(ctx["cuda_stream"])
    key = (
        cuda_stream,
        layers,
        active_requests,
        query_rows,
        total_pages,
        page_size,
        max_pages_per_request,
        heads,
        head_dim,
        cache_dtype,
        *(
            (name, int(buffers[name]["ptr"]), int(buffers[name]["bytes"]))
            for name in required_buffers
        ),
    )
    state = _DSPARK_ATTENTION_STATES.get(key)
    if state is not None:
        return state
    if not create:
        raise RuntimeError(
            "dSpark attention capture requires a matching prepare call during startup"
        )

    import torch

    stream = torch.cuda.ExternalStream(cuda_stream, device=device_id)
    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        q = _bf16_tensor(buffers["q"], q_shape)
        cache_tensor = _bf16_tensor if cache_dtype == "bf16" else _fp8_tensor
        k_cache = cache_tensor(buffers["k_cache"], kv_shape)
        v_cache = cache_tensor(buffers["v_cache"], kv_shape)
        output = _bf16_tensor(buffers["output"], output_shape)
        workspace = _u8_tensor(buffers["workspace"], workspace_shape)
        query_lengths = _i32_tensor(buffers["query_lengths"], lengths_shape)
        kv_lengths = _i32_tensor(buffers["kv_lengths"], lengths_shape)
        block_tables = _i32_tensor(buffers["block_tables"], block_tables_shape)
        query_offsets = _i64_tensor(buffers["query_offsets"], offsets_shape)
        output_offsets = _i64_tensor(buffers["output_offsets"], offsets_shape)
        query_indptr = _i32_tensor(buffers["query_indptr"], indptr_shape)
        kv_indptr = _i32_tensor(buffers["kv_indptr"], indptr_shape)
        page_indices = _i32_tensor(buffers["page_indices"], page_indices_shape)
        last_page_len = _i32_tensor(
            buffers["last_page_len"], last_page_len_shape
        )

        query_lengths.fill_(query_rows)
        element_stride = query_rows * heads * head_dim
        fixed_offsets = torch.arange(
            active_requests + 1, dtype=torch.int64, device=q.device
        ) * element_stride
        query_offsets.copy_(fixed_offsets)
        output_offsets.copy_(fixed_offsets)

        flashinfer_wrapper = None
        if cache_dtype == "fp8":
            from flashinfer import BatchPrefillWithPagedKVCacheWrapper

            runtime_query_indptr = query_indptr.clone()
            runtime_kv_indptr = kv_indptr.clone()
            runtime_page_indices = page_indices.clone()
            runtime_last_page_len = last_page_len.clone()
            physical_pages_per_request = total_pages // active_requests
            planning_query_indptr = torch.arange(
                active_requests + 1, dtype=torch.int32, device=q.device
            ) * query_rows
            planning_kv_indptr = torch.arange(
                active_requests + 1, dtype=torch.int32, device=q.device
            ) * physical_pages_per_request
            planning_page_indices = torch.arange(
                total_pages, dtype=torch.int32, device=q.device
            )
            planning_last_page_len = torch.full(
                (active_requests,), page_size, dtype=torch.int32, device=q.device
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

    state = _DsparkAttentionState(
        device_id=device_id,
        cuda_stream=cuda_stream,
        layers=layers,
        active_requests=active_requests,
        query_rows=query_rows,
        total_pages=total_pages,
        page_size=page_size,
        max_pages_per_request=max_pages_per_request,
        heads=heads,
        head_dim=head_dim,
        cache_dtype=cache_dtype,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        workspace=workspace,
        query_lengths=query_lengths,
        kv_lengths=kv_lengths,
        block_tables=block_tables,
        query_offsets=query_offsets,
        output_offsets=output_offsets,
        query_indptr=query_indptr,
        kv_indptr=kv_indptr,
        page_indices=page_indices,
        last_page_len=last_page_len,
        flashinfer_wrapper=flashinfer_wrapper,
    )
    _DSPARK_ATTENTION_STATES[key] = state
    return state


def _run_attention(state: _DsparkAttentionState) -> None:
    import torch

    stream = torch.cuda.ExternalStream(state.cuda_stream, device=state.device_id)
    max_sequence_kv = state.max_pages_per_request * state.page_size
    scale = 1.0 / math.sqrt(state.head_dim)
    with torch.cuda.device(state.device_id), torch.cuda.stream(stream):
        for layer in range(state.layers):
            if state.cache_dtype == "fp8":
                if state.flashinfer_wrapper is None:
                    raise RuntimeError("dSpark FP8 attention wrapper is not initialized")
                state.flashinfer_wrapper.run(
                    state.q[layer],
                    (state.k_cache[layer], state.v_cache[layer]),
                    out=state.output[layer],
                )
            else:
                from flashinfer.cudnn.prefill import (
                    cudnn_batch_prefill_with_kv_cache,
                )

                cudnn_batch_prefill_with_kv_cache(
                    state.q[layer],
                    state.k_cache[layer],
                    state.v_cache[layer],
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
                    out=state.output[layer],
                    is_cuda_graph_compatible=True,
                )


def _fp8_tensor(buffer: dict[str, Any], shape: tuple[int, ...]):
    import torch

    return _u8_tensor(buffer, shape).view(torch.float8_e4m3fn)


def _i64_tensor(buffer: dict[str, Any], shape: tuple[int, ...]):
    import torch

    dl_data_type, raw_dlpack_tensor = _dlpack_bridge()
    return torch.utils.dlpack.from_dlpack(
        raw_dlpack_tensor(
            ptr=int(buffer["ptr"]),
            shape=shape,
            dtype=dl_data_type(_DLPACK_CODE_INT, 64, 1),
            device_id=int(buffer["device_id"]),
        )
    )


def _tensor_bytes(shape: tuple[int, ...], element_bytes: int) -> int:
    values = 1
    for dim in shape:
        values *= dim
    return values * element_bytes
