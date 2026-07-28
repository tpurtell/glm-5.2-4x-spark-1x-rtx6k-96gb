from __future__ import annotations

import ctypes
import os
from importlib import import_module
from typing import Any

import triton
import triton.language as tl


_DLPACK_DEVICE_CUDA = 2
_DLPACK_CODE_FLOAT = 2
_DLPACK_CODE_BFLOAT = 4
_DLPACK_OWNERS: dict[int, Any] = {}
_DLPACK_BRIDGE: tuple[Any, Any] | None = None
_TARGET_ENV = "GLMRT_TRITON_MLP_CAPTURE_TARGET"


def capture_dense_mlp(ctx: dict[str, Any], **kwargs: Any) -> None:
    """Launch a graph-capturable BF16 SiLU-gated dense MLP through Triton."""

    target = os.environ.get(_TARGET_ENV)
    if target:
        module_name, _, function_name = target.partition(":")
        if not module_name or not function_name:
            raise ValueError(f"{_TARGET_ENV} must be formatted as module:function")
        getattr(import_module(module_name), function_name)(ctx, **kwargs)
        return

    import torch

    rows = int(kwargs["rows"])
    hidden = int(kwargs["hidden"])
    intermediate = int(kwargs["intermediate"])
    down_stride = int(kwargs["down_stride"])
    if rows <= 0:
        raise ValueError(f"Triton dense MLP requires positive rows, got {rows}")
    if hidden <= 0 or intermediate <= 0:
        raise ValueError(
            "Triton dense MLP requires positive dimensions, "
            f"got hidden={hidden}, intermediate={intermediate}"
        )
    if down_stride < intermediate:
        raise ValueError(
            "Triton dense MLP down stride must cover intermediate columns, "
            f"got down_stride={down_stride}, intermediate={intermediate}"
        )

    buffers = ctx["buffers"]
    input_buffer = buffers["input"]
    device_id = int(input_buffer["device_id"])
    stream = torch.cuda.ExternalStream(int(ctx["cuda_stream"]), device=device_id)

    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        input_tensor = _bf16_tensor(input_buffer, (rows, hidden))
        gate_weight = _bf16_tensor(buffers["gate_weight"], (intermediate, hidden))
        up_weight = _bf16_tensor(buffers["up_weight"], (intermediate, hidden))
        down_weight = _bf16_tensor(buffers["down_weight"], (hidden, down_stride))
        gate_output = _f32_tensor(buffers["gate_output"], (rows, intermediate))
        up_output = _f32_tensor(buffers["up_output"], (rows, intermediate))
        activation = _bf16_tensor(buffers["activation"], (rows, intermediate))
        output = _bf16_tensor(buffers["output"], (rows, hidden))

        if rows <= 16:
            block_m = 16
            block_n = 16
            hidden_block_k = 256
            intermediate_block_k = 256
        else:
            block_m = 64
            block_n = 128
            hidden_block_k = 64
            intermediate_block_k = 64
        gate_grid = (triton.cdiv(rows, block_m), triton.cdiv(intermediate, block_n))
        output_grid = (triton.cdiv(rows, block_m), triton.cdiv(hidden, block_n))
        total_activation = rows * intermediate

        _matmul_bf16_bf16_to_f32[gate_grid](
            input_tensor,
            gate_weight,
            gate_output,
            M=rows,
            N=intermediate,
            K=hidden,
            A_STRIDE_M=hidden,
            A_STRIDE_K=1,
            B_STRIDE_N=hidden,
            B_STRIDE_K=1,
            C_STRIDE_M=intermediate,
            C_STRIDE_N=1,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=hidden_block_k,
        )
        _matmul_bf16_bf16_to_f32[gate_grid](
            input_tensor,
            up_weight,
            up_output,
            M=rows,
            N=intermediate,
            K=hidden,
            A_STRIDE_M=hidden,
            A_STRIDE_K=1,
            B_STRIDE_N=hidden,
            B_STRIDE_K=1,
            C_STRIDE_M=intermediate,
            C_STRIDE_N=1,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=hidden_block_k,
        )
        _silu_mul_f32_to_bf16[(triton.cdiv(total_activation, 256),)](
            gate_output,
            up_output,
            activation,
            TOTAL=total_activation,
            BLOCK=256,
        )
        _matmul_bf16_bf16_to_bf16[output_grid](
            activation,
            down_weight,
            output,
            M=rows,
            N=hidden,
            K=intermediate,
            A_STRIDE_M=intermediate,
            A_STRIDE_K=1,
            B_STRIDE_N=down_stride,
            B_STRIDE_K=1,
            C_STRIDE_M=hidden,
            C_STRIDE_N=1,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=intermediate_block_k,
        )


@triton.jit
def _matmul_bf16_bf16_to_f32(
    a,
    b,
    c,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    A_STRIDE_M: tl.constexpr,
    A_STRIDE_K: tl.constexpr,
    B_STRIDE_N: tl.constexpr,
    B_STRIDE_K: tl.constexpr,
    C_STRIDE_M: tl.constexpr,
    C_STRIDE_N: tl.constexpr,
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
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a_values = tl.load(
            a + offs_m[:, None] * A_STRIDE_M + k[None, :] * A_STRIDE_K,
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        b_values = tl.load(
            b + offs_n[None, :] * B_STRIDE_N + k[:, None] * B_STRIDE_K,
            mask=(offs_n[None, :] < N) & (k[:, None] < K),
            other=0.0,
        )
        acc += tl.dot(a_values, b_values)
    tl.store(
        c + offs_m[:, None] * C_STRIDE_M + offs_n[None, :] * C_STRIDE_N,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _silu_mul_f32_to_bf16(
    gate,
    up,
    activation,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    gate_values = tl.load(gate + offsets, mask=mask, other=0.0)
    up_values = tl.load(up + offsets, mask=mask, other=0.0)
    tl.store(
        activation + offsets,
        gate_values * tl.sigmoid(gate_values) * up_values,
        mask=mask,
    )


@triton.jit
def _matmul_bf16_bf16_to_bf16(
    a,
    b,
    c,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    A_STRIDE_M: tl.constexpr,
    A_STRIDE_K: tl.constexpr,
    B_STRIDE_N: tl.constexpr,
    B_STRIDE_K: tl.constexpr,
    C_STRIDE_M: tl.constexpr,
    C_STRIDE_N: tl.constexpr,
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
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a_values = tl.load(
            a + offs_m[:, None] * A_STRIDE_M + k[None, :] * A_STRIDE_K,
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        b_values = tl.load(
            b + offs_n[None, :] * B_STRIDE_N + k[:, None] * B_STRIDE_K,
            mask=(offs_n[None, :] < N) & (k[:, None] < K),
            other=0.0,
        )
        acc += tl.dot(a_values, b_values)
    tl.store(
        c + offs_m[:, None] * C_STRIDE_M + offs_n[None, :] * C_STRIDE_N,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _bf16_tensor(buffer: dict[str, Any], shape: tuple[int, ...]):
    return _typed_tensor(buffer, shape, _DLPACK_CODE_BFLOAT, 16)


def _f32_tensor(buffer: dict[str, Any], shape: tuple[int, ...]):
    return _typed_tensor(buffer, shape, _DLPACK_CODE_FLOAT, 32)


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
    stride = 1
    strides = []
    for dim in reversed(shape):
        strides.append(stride)
        stride *= max(int(dim), 1)
    return tuple(reversed(strides))
