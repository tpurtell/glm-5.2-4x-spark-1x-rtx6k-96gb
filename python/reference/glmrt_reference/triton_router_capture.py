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
_TARGET_ENV = "GLMRT_TRITON_ROUTER_CAPTURE_TARGET"


def capture_router_topk(ctx: dict[str, Any], **kwargs: Any) -> None:
    """Launch a graph-capturable BF16 router top-k through Triton."""

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
    experts = int(kwargs["experts"])
    top_k = int(kwargs["top_k"])
    routed_scaling_factor = float(kwargs["routed_scaling_factor"])
    if rows <= 0 or hidden_dim <= 0 or experts <= 0:
        raise ValueError(
            "Triton router top-k requires positive shape, "
            f"got rows={rows}, hidden_dim={hidden_dim}, experts={experts}"
        )
    if top_k <= 0 or top_k > experts:
        raise ValueError(
            f"Triton router top-k invalid top_k={top_k} for experts={experts}"
        )

    buffers = ctx["buffers"]
    hidden_buffer = buffers["hidden"]
    device_id = int(hidden_buffer["device_id"])
    stream = torch.cuda.ExternalStream(int(ctx["cuda_stream"]), device=device_id)

    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        hidden = _bf16_tensor(hidden_buffer, (rows, hidden_dim))
        router_weight = _bf16_tensor(buffers["router_weight"], (experts, hidden_dim))
        correction_bias = _f32_tensor(buffers["correction_bias"], (experts,))
        score_scratch = _f32_tensor(buffers["score_scratch"], (rows, experts))
        topk_indices = _u32_tensor(buffers["topk_indices"], (rows, top_k))
        topk_scores = _f32_tensor(buffers["topk_scores"], (rows, top_k))
        topk_weights = _f32_tensor(buffers["topk_weights"], (rows, top_k))

        block_m = 16
        block_n = 16
        block_k = 64
        score_grid = (triton.cdiv(rows, block_m), triton.cdiv(experts, block_n))
        block_experts = triton.next_power_of_2(experts)
        _router_scores_bf16[score_grid](
            hidden,
            router_weight,
            score_scratch,
            ROWS=rows,
            HIDDEN_DIM=hidden_dim,
            EXPERTS=experts,
            HIDDEN_STRIDE_ROW=hidden_dim,
            WEIGHT_STRIDE_EXPERT=hidden_dim,
            SCORE_STRIDE_ROW=experts,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
        )
        _router_topk[(rows,)](
            score_scratch,
            correction_bias,
            topk_indices,
            topk_scores,
            topk_weights,
            EXPERTS=experts,
            TOP_K=top_k,
            ROUTED_SCALING_FACTOR=routed_scaling_factor,
            BLOCK_EXPERTS=block_experts,
        )


@triton.jit
def _router_scores_bf16(
    hidden,
    router_weight,
    scores,
    ROWS: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    EXPERTS: tl.constexpr,
    HIDDEN_STRIDE_ROW: tl.constexpr,
    WEIGHT_STRIDE_EXPERT: tl.constexpr,
    SCORE_STRIDE_ROW: tl.constexpr,
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
            router_weight + offs_n[None, :] * WEIGHT_STRIDE_EXPERT + k[:, None],
            mask=(offs_n[None, :] < EXPERTS) & (k[:, None] < HIDDEN_DIM),
            other=0.0,
        )
        acc += tl.dot(hidden_values, weight_values)
    tl.store(
        scores + offs_m[:, None] * SCORE_STRIDE_ROW + offs_n[None, :],
        tl.sigmoid(acc),
        mask=(offs_m[:, None] < ROWS) & (offs_n[None, :] < EXPERTS),
    )


@triton.jit
def _router_topk(
    scores,
    correction_bias,
    topk_indices,
    topk_scores,
    topk_weights,
    EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    ROUTED_SCALING_FACTOR: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_EXPERTS)
    expert_mask = offsets < EXPERTS
    row_scores = tl.load(
        scores + row * EXPERTS + offsets,
        mask=expert_mask,
        other=0.0,
    )
    score_is_finite = row_scores == row_scores
    corrected = row_scores + tl.load(
        correction_bias + offsets,
        mask=expert_mask,
        other=-float("inf"),
    )
    corrected = tl.where(
        expert_mask & score_is_finite & (corrected == corrected),
        corrected,
        -float("inf"),
    )
    selected = tl.full((BLOCK_EXPERTS,), False, tl.int1)
    score_sum = tl.full((), 0.0, tl.float32)

    for rank in range(0, TOP_K):
        candidates = tl.where(selected, -float("inf"), corrected)
        best_corrected = tl.max(candidates, axis=0)
        best_mask = candidates == best_corrected
        best_expert = tl.min(tl.where(best_mask, offsets, BLOCK_EXPERTS), axis=0)
        best_valid = best_expert < EXPERTS
        safe_expert = tl.minimum(best_expert, EXPERTS - 1)
        best_score = tl.load(
            scores + row * EXPERTS + safe_expert,
            mask=best_valid,
            other=0.0,
        )
        best_score = tl.where(best_valid & (best_score == best_score), best_score, 0.0)
        tl.store(topk_indices + row * TOP_K + rank, tl.where(best_valid, safe_expert, 0))
        tl.store(topk_scores + row * TOP_K + rank, best_score)
        score_sum += best_score
        selected = selected | (best_valid & (offsets == safe_expert))

    score_sum = tl.maximum(score_sum, 1.0e-12)
    for rank in range(0, TOP_K):
        score = tl.load(topk_scores + row * TOP_K + rank)
        score = tl.where(score == score, score, 0.0)
        tl.store(topk_weights + row * TOP_K + rank, score / score_sum * ROUTED_SCALING_FACTOR)


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
