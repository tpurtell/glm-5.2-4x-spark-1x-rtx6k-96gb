from __future__ import annotations

import ctypes
import os
from importlib import import_module
from typing import Any

import triton
import triton.language as tl


_DLPACK_DEVICE_CUDA = 2
_DLPACK_CODE_UINT = 1
_DLPACK_CODE_FLOAT = 2
_DLPACK_CODE_BFLOAT = 4
_DLPACK_OWNERS: dict[int, Any] = {}
_DLPACK_BRIDGE: tuple[Any, Any] | None = None
_TARGET_ENV = "GLMRT_TRITON_SAMPLING_CAPTURE_TARGET"
_BLOCK_VOCAB = 1024


def capture_lm_head_sample_topk_topp(ctx: dict[str, Any], **kwargs: Any) -> None:
    """Launch graph-capturable BF16 LM-head top-k/top-p sampling through Triton."""

    target = os.environ.get(_TARGET_ENV)
    if target:
        module_name, _, function_name = target.partition(":")
        if not module_name or not function_name:
            raise ValueError(f"{_TARGET_ENV} must be formatted as module:function")
        getattr(import_module(module_name), function_name)(ctx, **kwargs)
        return

    import torch

    rows = int(kwargs["rows"])
    hidden_dim = int(kwargs["hidden_dim"])
    vocab = int(kwargs["vocab"])
    temperature = float(kwargs["temperature"])
    top_k = int(kwargs["top_k"])
    top_p = float(kwargs["top_p"])
    if rows <= 0 or hidden_dim <= 0 or vocab <= 0:
        raise ValueError(
            "Triton LM-head sampler requires positive shape, "
            f"got rows={rows}, hidden_dim={hidden_dim}, vocab={vocab}"
        )
    if top_k <= 0 or top_k > vocab:
        raise ValueError(f"Triton LM-head sampler invalid top_k={top_k} for vocab={vocab}")
    if temperature <= 0.0:
        raise ValueError(f"Triton LM-head sampler requires positive temperature, got {temperature}")
    if top_p <= 0.0 or top_p > 1.0:
        raise ValueError(f"Triton LM-head sampler requires top_p in (0, 1], got {top_p}")

    buffers = ctx["buffers"]
    hidden_buffer = buffers["hidden"]
    device_id = int(hidden_buffer["device_id"])
    stream = torch.cuda.ExternalStream(int(ctx["cuda_stream"]), device=device_id)

    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        hidden = _bf16_tensor(hidden_buffer, (rows, hidden_dim))
        lm_head = _bf16_tensor(buffers["lm_head"], (vocab, hidden_dim))
        random_uniforms = _f32_tensor(buffers["random_uniforms"], (rows,))
        logits = _f32_tensor(buffers["logits"], (rows, vocab))
        candidate_scores = _f32_tensor(
            buffers["candidate_scores"],
            (rows, triton.cdiv(vocab, _BLOCK_VOCAB), top_k),
        )
        candidate_indices = _u32_tensor(
            buffers["candidate_indices"],
            (rows, triton.cdiv(vocab, _BLOCK_VOCAB), top_k),
        )
        out_indices = _u32_tensor(buffers["out_indices"], (rows,))
        out_scores = _f32_tensor(buffers["out_scores"], (rows,))
        out_argmax_indices = _u32_tensor(
            buffers.get("out_argmax_indices", buffers["out_indices"]), (rows,)
        )
        out_argmax_scores = _f32_tensor(
            buffers.get("out_argmax_scores", buffers["out_scores"]), (rows,)
        )

        block_m = 16
        block_n = 16
        block_k = 64
        num_vocab_blocks = triton.cdiv(vocab, _BLOCK_VOCAB)
        block_candidates = triton.next_power_of_2(num_vocab_blocks * top_k)
        top_k_block = triton.next_power_of_2(top_k)
        logits_grid = (triton.cdiv(rows, block_m), triton.cdiv(vocab, block_n))
        _lm_head_logits_bf16[logits_grid](
            hidden,
            lm_head,
            logits,
            ROWS=rows,
            HIDDEN_DIM=hidden_dim,
            VOCAB=vocab,
            HIDDEN_STRIDE_ROW=hidden_dim,
            LM_HEAD_STRIDE_TOKEN=hidden_dim,
            LOGITS_STRIDE_ROW=vocab,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
        )
        _block_topk[(rows, num_vocab_blocks)](
            logits,
            candidate_scores,
            candidate_indices,
            VOCAB=vocab,
            TOP_K=top_k,
            NUM_VOCAB_BLOCKS=num_vocab_blocks,
            BLOCK_VOCAB=_BLOCK_VOCAB,
        )
        _sample_from_candidates[(rows,)](
            candidate_scores,
            candidate_indices,
            random_uniforms,
            out_argmax_indices,
            out_argmax_scores,
            out_indices,
            out_scores,
            TEMPERATURE=temperature,
            TOP_K=top_k,
            TOP_P=top_p,
            NUM_VOCAB_BLOCKS=num_vocab_blocks,
            BLOCK_CANDIDATES=block_candidates,
            TOP_K_BLOCK=top_k_block,
        )


def capture_lm_head_constrained_sample_topk_topp(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    """Launch graph-capturable BF16 LM-head sampling with token bitmasks."""

    target = os.environ.get(_TARGET_ENV)
    if target:
        module_name, _, function_name = target.partition(":")
        if not module_name or not function_name:
            raise ValueError(f"{_TARGET_ENV} must be formatted as module:function")
        getattr(import_module(module_name), function_name)(ctx, **kwargs)
        return

    import torch

    rows = int(kwargs["rows"])
    hidden_dim = int(kwargs["hidden_dim"])
    vocab = int(kwargs["vocab"])
    temperature = float(kwargs["temperature"])
    top_k = int(kwargs["top_k"])
    top_p = float(kwargs["top_p"])
    if rows <= 0 or hidden_dim <= 0 or vocab <= 0:
        raise ValueError(
            "constrained Triton LM-head sampler requires positive shape, "
            f"got rows={rows}, hidden_dim={hidden_dim}, vocab={vocab}"
        )
    if top_k <= 0 or top_k > vocab:
        raise ValueError(
            f"constrained Triton LM-head sampler invalid top_k={top_k} for vocab={vocab}"
        )
    if temperature <= 0.0:
        raise ValueError(
            "constrained Triton LM-head sampler requires positive temperature, "
            f"got {temperature}"
        )
    if top_p <= 0.0 or top_p > 1.0:
        raise ValueError(
            f"constrained Triton LM-head sampler requires top_p in (0, 1], got {top_p}"
        )

    buffers = ctx["buffers"]
    hidden_buffer = buffers["hidden"]
    device_id = int(hidden_buffer["device_id"])
    stream = torch.cuda.ExternalStream(int(ctx["cuda_stream"]), device=device_id)

    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        hidden = _bf16_tensor(hidden_buffer, (rows, hidden_dim))
        lm_head = _bf16_tensor(buffers["lm_head"], (vocab, hidden_dim))
        random_uniforms = _f32_tensor(buffers["random_uniforms"], (rows,))
        mask_words = triton.cdiv(vocab, 32)
        token_bitmask = _u32_tensor(buffers["token_bitmask"], (rows, mask_words))
        logits = _f32_tensor(buffers["logits"], (rows, vocab))
        candidate_scores = _f32_tensor(
            buffers["candidate_scores"],
            (rows, triton.cdiv(vocab, _BLOCK_VOCAB), top_k),
        )
        candidate_indices = _u32_tensor(
            buffers["candidate_indices"],
            (rows, triton.cdiv(vocab, _BLOCK_VOCAB), top_k),
        )
        out_indices = _u32_tensor(buffers["out_indices"], (rows,))
        out_scores = _f32_tensor(buffers["out_scores"], (rows,))
        out_argmax_indices = _u32_tensor(
            buffers.get("out_argmax_indices", buffers["out_indices"]), (rows,)
        )
        out_argmax_scores = _f32_tensor(
            buffers.get("out_argmax_scores", buffers["out_scores"]), (rows,)
        )

        block_m = 16
        block_n = 16
        block_k = 64
        num_vocab_blocks = triton.cdiv(vocab, _BLOCK_VOCAB)
        block_candidates = triton.next_power_of_2(num_vocab_blocks * top_k)
        top_k_block = triton.next_power_of_2(top_k)
        logits_grid = (triton.cdiv(rows, block_m), triton.cdiv(vocab, block_n))
        _lm_head_logits_bf16[logits_grid](
            hidden,
            lm_head,
            logits,
            ROWS=rows,
            HIDDEN_DIM=hidden_dim,
            VOCAB=vocab,
            HIDDEN_STRIDE_ROW=hidden_dim,
            LM_HEAD_STRIDE_TOKEN=hidden_dim,
            LOGITS_STRIDE_ROW=vocab,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
        )
        _block_topk_masked[(rows, num_vocab_blocks)](
            logits,
            token_bitmask,
            candidate_scores,
            candidate_indices,
            VOCAB=vocab,
            TOP_K=top_k,
            NUM_VOCAB_BLOCKS=num_vocab_blocks,
            MASK_WORDS=mask_words,
            BLOCK_VOCAB=_BLOCK_VOCAB,
        )
        _sample_from_candidates[(rows,)](
            candidate_scores,
            candidate_indices,
            random_uniforms,
            out_argmax_indices,
            out_argmax_scores,
            out_indices,
            out_scores,
            TEMPERATURE=temperature,
            TOP_K=top_k,
            TOP_P=top_p,
            NUM_VOCAB_BLOCKS=num_vocab_blocks,
            BLOCK_CANDIDATES=block_candidates,
            TOP_K_BLOCK=top_k_block,
        )


@triton.jit
def _lm_head_logits_bf16(
    hidden,
    lm_head,
    logits,
    ROWS: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    VOCAB: tl.constexpr,
    HIDDEN_STRIDE_ROW: tl.constexpr,
    LM_HEAD_STRIDE_TOKEN: tl.constexpr,
    LOGITS_STRIDE_ROW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, HIDDEN_DIM, BLOCK_K):
        k = k0 + offs_k
        hidden_values = tl.load(
            hidden + offs_m[:, None] * HIDDEN_STRIDE_ROW + k[None, :],
            mask=(offs_m[:, None] < ROWS) & (k[None, :] < HIDDEN_DIM),
            other=0.0,
        )
        weight_values = tl.load(
            lm_head + offs_n[None, :] * LM_HEAD_STRIDE_TOKEN + k[:, None],
            mask=(offs_n[None, :] < VOCAB) & (k[:, None] < HIDDEN_DIM),
            other=0.0,
        )
        acc += tl.dot(hidden_values, weight_values)
    tl.store(
        logits + offs_m[:, None] * LOGITS_STRIDE_ROW + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < ROWS) & (offs_n[None, :] < VOCAB),
    )


@triton.jit
def _block_topk(
    logits,
    candidate_scores,
    candidate_indices,
    VOCAB: tl.constexpr,
    TOP_K: tl.constexpr,
    NUM_VOCAB_BLOCKS: tl.constexpr,
    BLOCK_VOCAB: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK_VOCAB + tl.arange(0, BLOCK_VOCAB)
    valid = offsets < VOCAB
    values = tl.load(logits + row * VOCAB + offsets, mask=valid, other=-float("inf"))
    selected = tl.full((BLOCK_VOCAB,), False, tl.int1)
    out_base = (row * NUM_VOCAB_BLOCKS + block) * TOP_K

    for rank in range(0, TOP_K):
        candidates = tl.where(selected, -float("inf"), values)
        best_score = tl.max(candidates, axis=0)
        best_mask = candidates == best_score
        best_token = tl.min(tl.where(best_mask, offsets, VOCAB), axis=0)
        tl.store(candidate_scores + out_base + rank, best_score)
        tl.store(candidate_indices + out_base + rank, best_token)
        selected = selected | (offsets == best_token)


@triton.jit
def _block_topk_masked(
    logits,
    token_bitmask,
    candidate_scores,
    candidate_indices,
    VOCAB: tl.constexpr,
    TOP_K: tl.constexpr,
    NUM_VOCAB_BLOCKS: tl.constexpr,
    MASK_WORDS: tl.constexpr,
    BLOCK_VOCAB: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK_VOCAB + tl.arange(0, BLOCK_VOCAB)
    valid = offsets < VOCAB
    mask_word = tl.load(
        token_bitmask + row * MASK_WORDS + offsets // 32,
        mask=valid,
        other=0,
    )
    allowed = ((mask_word >> (offsets % 32)) & 1) != 0
    values = tl.load(
        logits + row * VOCAB + offsets,
        mask=valid & allowed,
        other=-float("inf"),
    )
    selected = tl.full((BLOCK_VOCAB,), False, tl.int1)
    out_base = (row * NUM_VOCAB_BLOCKS + block) * TOP_K

    for rank in range(0, TOP_K):
        candidates = tl.where(selected, -float("inf"), values)
        best_score = tl.max(candidates, axis=0)
        best_mask = candidates == best_score
        best_token = tl.min(tl.where(best_mask, offsets, VOCAB), axis=0)
        tl.store(candidate_scores + out_base + rank, best_score)
        tl.store(candidate_indices + out_base + rank, best_token)
        selected = selected | (offsets == best_token)


@triton.jit
def _sample_from_candidates(
    candidate_scores,
    candidate_indices,
    random_uniforms,
    out_argmax_indices,
    out_argmax_scores,
    out_indices,
    out_scores,
    TEMPERATURE: tl.constexpr,
    TOP_K: tl.constexpr,
    TOP_P: tl.constexpr,
    NUM_VOCAB_BLOCKS: tl.constexpr,
    BLOCK_CANDIDATES: tl.constexpr,
    TOP_K_BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    candidate_count: tl.constexpr = NUM_VOCAB_BLOCKS * TOP_K
    candidate_offsets = tl.arange(0, BLOCK_CANDIDATES)
    valid_candidates = candidate_offsets < candidate_count
    candidate_base = row * candidate_count
    scores = tl.load(
        candidate_scores + candidate_base + candidate_offsets,
        mask=valid_candidates,
        other=-float("inf"),
    )
    indices = tl.load(
        candidate_indices + candidate_base + candidate_offsets,
        mask=valid_candidates,
        other=0,
    )
    selected_candidates = tl.full((BLOCK_CANDIDATES,), False, tl.int1)
    rank_offsets = tl.arange(0, TOP_K_BLOCK)
    valid_ranks = rank_offsets < TOP_K
    top_logits = tl.full((TOP_K_BLOCK,), -float("inf"), tl.float32)
    top_indices = tl.full((TOP_K_BLOCK,), 0, tl.uint32)

    for rank in range(0, TOP_K):
        masked_scores = tl.where(selected_candidates, -float("inf"), scores)
        best_score = tl.max(masked_scores, axis=0)
        best_mask = masked_scores == best_score
        best_token = tl.min(tl.where(best_mask, indices, 0xFFFFFFFF), axis=0)
        top_logits = tl.where(rank_offsets == rank, best_score, top_logits)
        top_indices = tl.where(rank_offsets == rank, best_token, top_indices)
        selected_candidates = selected_candidates | (
            (scores == best_score) & (indices == best_token)
        )

    argmax_index = tl.sum(tl.where(rank_offsets == 0, top_indices, 0), axis=0)
    argmax_score = tl.sum(tl.where(rank_offsets == 0, top_logits, 0.0), axis=0)
    tl.store(out_argmax_indices + row, argmax_index)
    tl.store(out_argmax_scores + row, argmax_score)

    scaled = top_logits / TEMPERATURE
    max_scaled = tl.max(tl.where(valid_ranks, scaled, -float("inf")), axis=0)
    exp_values = tl.exp(tl.where(valid_ranks, scaled - max_scaled, -float("inf")))
    total = tl.maximum(tl.sum(exp_values, axis=0), 1.0e-20)
    probs = exp_values / total
    cumulative = tl.cumsum(tl.where(valid_ranks, probs, 0.0), 0)
    top_p_clamped = tl.minimum(tl.maximum(TOP_P, 1.0e-6), 1.0)
    nucleus_rank = tl.min(
        tl.where((cumulative >= top_p_clamped) & valid_ranks, rank_offsets, TOP_K - 1),
        axis=0,
    )
    nucleus_mass = tl.sum(tl.where(rank_offsets == nucleus_rank, cumulative, 0.0), axis=0)
    nucleus_mass = tl.maximum(nucleus_mass, 1.0e-20)
    random_value = tl.load(random_uniforms + row)
    random_value = tl.minimum(tl.maximum(random_value, 0.0), 0.99999994)
    target = random_value * nucleus_mass
    selected_rank = tl.min(
        tl.where(
            (cumulative >= target) & (rank_offsets <= nucleus_rank) & valid_ranks,
            rank_offsets,
            TOP_K - 1,
        ),
        axis=0,
    )
    selected_index = tl.sum(tl.where(rank_offsets == selected_rank, top_indices, 0), axis=0)
    selected_score = tl.sum(
        tl.where(rank_offsets == selected_rank, probs / nucleus_mass, 0.0),
        axis=0,
    )
    selected_score = tl.minimum(selected_score, 1.0)
    tl.store(out_indices + row, selected_index)
    tl.store(out_scores + row, selected_score)


def _bf16_tensor(buffer: dict[str, Any], shape: tuple[int, ...]):
    return _typed_tensor(buffer, shape, _DLPACK_CODE_BFLOAT, 16)


def _f32_tensor(buffer: dict[str, Any], shape: tuple[int, ...]):
    return _typed_tensor(buffer, shape, _DLPACK_CODE_FLOAT, 32)


def _u32_tensor(buffer: dict[str, Any], shape: tuple[int, ...]):
    return _typed_tensor(buffer, shape, _DLPACK_CODE_UINT, 32)


def _typed_tensor(buffer: dict[str, Any], shape: tuple[int, ...], code: int, bits: int):
    import torch

    element_bytes = bits // 8
    required = element_bytes
    for dim in shape:
        required *= int(dim)
    if int(buffer["bytes"]) < required:
        raise ValueError(
            f"raw tensor buffer is too small for shape {shape}: "
            f"{buffer['bytes']} < {required}"
        )

    dl_data_type, raw_dlpack_tensor = _dlpack_bridge()
    return torch.utils.dlpack.from_dlpack(
        raw_dlpack_tensor(
            ptr=int(buffer["ptr"]),
            shape=shape,
            dtype=dl_data_type(code, bits, 1),
            device_id=int(buffer["device_id"]),
        )
    )


def _dlpack_bridge() -> tuple[Any, Any]:
    global _DLPACK_BRIDGE
    if _DLPACK_BRIDGE is not None:
        return _DLPACK_BRIDGE

    class DLDevice(ctypes.Structure):
        _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]

    class DLDataType(ctypes.Structure):
        _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]

    class DLTensor(ctypes.Structure):
        _fields_ = [
            ("data", ctypes.c_void_p),
            ("device", DLDevice),
            ("ndim", ctypes.c_int),
            ("dtype", DLDataType),
            ("shape", ctypes.POINTER(ctypes.c_int64)),
            ("strides", ctypes.POINTER(ctypes.c_int64)),
            ("byte_offset", ctypes.c_uint64),
        ]

    class DLManagedTensor(ctypes.Structure):
        pass

    DLManagedTensorPtr = ctypes.POINTER(DLManagedTensor)
    DLManagedTensorDeleter = ctypes.CFUNCTYPE(None, DLManagedTensorPtr)

    @DLManagedTensorDeleter
    def delete_dlpack_tensor(ptr: DLManagedTensorPtr) -> None:
        if bool(ptr):
            _DLPACK_OWNERS.pop(ctypes.addressof(ptr.contents), None)

    DLManagedTensor._fields_ = [
        ("dl_tensor", DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", DLManagedTensorDeleter),
    ]

    py_capsule_new = ctypes.pythonapi.PyCapsule_New
    py_capsule_new.restype = ctypes.py_object
    py_capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

    class RawDlpackTensor:
        def __init__(
            self,
            *,
            ptr: int,
            shape: tuple[int, ...],
            dtype: DLDataType,
            device_id: int,
        ) -> None:
            if ptr == 0:
                raise ValueError("raw DLPack tensor pointer is null")
            if any(dim < 0 for dim in shape):
                raise ValueError(f"raw DLPack tensor shape has a negative dimension: {shape}")
            self.ptr = int(ptr)
            self.shape = shape
            self.dtype = dtype
            self.device_id = int(device_id)
            self._shape = (ctypes.c_int64 * len(shape))(*shape)
            self._strides = _contiguous_strides(shape)
            self._strides_array = (ctypes.c_int64 * len(shape))(*self._strides)
            self._managed = DLManagedTensor()
            self._managed.dl_tensor = DLTensor(
                ctypes.c_void_p(self.ptr),
                DLDevice(_DLPACK_DEVICE_CUDA, self.device_id),
                len(shape),
                self.dtype,
                self._shape,
                self._strides_array,
                0,
            )
            self._managed.manager_ctx = None
            self._managed.deleter = delete_dlpack_tensor

        def __dlpack_device__(self) -> tuple[int, int]:
            return (_DLPACK_DEVICE_CUDA, self.device_id)

        def __dlpack__(self, stream: int | None = None) -> object:
            del stream
            address = ctypes.addressof(self._managed)
            _DLPACK_OWNERS[address] = self
            return py_capsule_new(ctypes.c_void_p(address), b"dltensor", None)

    _DLPACK_BRIDGE = (DLDataType, RawDlpackTensor)
    return _DLPACK_BRIDGE


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    strides = [1] * len(shape)
    running = 1
    for idx in range(len(shape) - 1, -1, -1):
        strides[idx] = running
        running *= int(shape[idx])
    return tuple(strides)
