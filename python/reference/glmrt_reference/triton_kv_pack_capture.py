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
_TARGET_ENV = "GLMRT_TRITON_KV_PACK_CAPTURE_TARGET"
_NOPE_VALUES = 512
_ROPE_VALUES = 64
_SCALE_OFFSET_BYTES = 512
_ROPE_OFFSET_BYTES = 528
_PACKED_BYTES = 656
_GROUPS = 4
_GROUP_SIZE = 128
_E4M3_MAX = 448.0


def capture_mla_kv_pack_fp8_ds_mla(ctx: dict[str, Any], **kwargs: Any) -> None:
    """Launch graph-capturable MLA FP8 DS KV packing through Triton."""

    target = os.environ.get(_TARGET_ENV)
    if target:
        module_name, _, function_name = target.partition(":")
        if not module_name or not function_name:
            raise ValueError(f"{_TARGET_ENV} must be formatted as module:function")
        getattr(import_module(module_name), function_name)(ctx, **kwargs)
        return

    import torch

    rows = int(kwargs["rows"])
    projected_stride_bytes = int(kwargs["projected_stride_bytes"])
    packed_stride_bytes = int(kwargs["packed_stride_bytes"])
    if rows <= 0:
        raise ValueError(f"Triton MLA FP8 KV pack requires rows > 0, got {rows}")
    if projected_stride_bytes <= 0 or projected_stride_bytes % 2 != 0:
        raise ValueError(
            "Triton MLA FP8 KV pack requires a positive BF16-aligned projected stride, "
            f"got {projected_stride_bytes}"
        )
    projected_stride_bf16 = projected_stride_bytes // 2
    if projected_stride_bf16 < _NOPE_VALUES + _ROPE_VALUES:
        raise ValueError(
            "Triton MLA FP8 KV pack projected stride is too small: "
            f"{projected_stride_bf16} < {_NOPE_VALUES + _ROPE_VALUES}"
        )
    if packed_stride_bytes < _PACKED_BYTES or packed_stride_bytes % 4 != 0:
        raise ValueError(
            "Triton MLA FP8 KV pack requires a packed stride >= 656 and 4-byte aligned, "
            f"got {packed_stride_bytes}"
        )

    buffers = ctx["buffers"]
    projected_buffer = buffers["projected"]
    device_id = int(projected_buffer["device_id"])
    stream = torch.cuda.ExternalStream(int(ctx["cuda_stream"]), device=device_id)

    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        projected_bf16 = _bf16_tensor(projected_buffer, (rows, projected_stride_bf16))
        projected_u16 = _u16_tensor(projected_buffer, (rows, projected_stride_bf16))
        packed_buffer = buffers["packed"]
        packed_u8 = _u8_tensor(packed_buffer, (rows, packed_stride_bytes))
        packed_u16 = _u16_tensor(packed_buffer, (rows, packed_stride_bytes // 2))
        packed_f32 = _f32_tensor(packed_buffer, (rows, packed_stride_bytes // 4))
        _mla_kv_pack_fp8_ds_mla[(rows,)](
            projected_bf16,
            projected_u16,
            packed_u8,
            packed_u16,
            packed_f32,
            PROJECTED_STRIDE_BF16=projected_stride_bf16,
            PACKED_STRIDE_BYTES=packed_stride_bytes,
            PACKED_STRIDE_U16=packed_stride_bytes // 2,
            PACKED_STRIDE_F32=packed_stride_bytes // 4,
            BLOCK_GROUP=_GROUP_SIZE,
            BLOCK_ROPE=_ROPE_VALUES,
        )


@triton.jit
def _mla_kv_pack_fp8_ds_mla(
    projected_bf16,
    projected_u16,
    packed_u8,
    packed_u16,
    packed_f32,
    PROJECTED_STRIDE_BF16: tl.constexpr,
    PACKED_STRIDE_BYTES: tl.constexpr,
    PACKED_STRIDE_U16: tl.constexpr,
    PACKED_STRIDE_F32: tl.constexpr,
    BLOCK_GROUP: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    group_offsets = tl.arange(0, BLOCK_GROUP)

    for group in tl.static_range(0, 4):
        cols = group * BLOCK_GROUP + group_offsets
        values = tl.load(projected_bf16 + row * PROJECTED_STRIDE_BF16 + cols).to(tl.float32)
        max_abs = tl.max(tl.abs(values), axis=0)
        scale = tl.where(max_abs > 0.0, max_abs / 448.0, 1.0)
        encoded = _f32_to_e4m3(values / scale)
        tl.store(packed_u8 + row * PACKED_STRIDE_BYTES + cols, encoded)
        tl.store(
            packed_f32 + row * PACKED_STRIDE_F32 + 128 + group,
            scale,
        )

    rope_offsets = tl.arange(0, BLOCK_ROPE)
    rope = tl.load(
        projected_u16 + row * PROJECTED_STRIDE_BF16 + 512 + rope_offsets
    )
    tl.store(
        packed_u16 + row * PACKED_STRIDE_U16 + 264 + rope_offsets,
        rope,
    )


@triton.jit
def _f32_to_e4m3(values):
    clipped = tl.minimum(tl.maximum(values, -448.0), 448.0)
    abs_values = tl.abs(clipped)
    sign = tl.where(clipped < 0.0, 0x80, 0).to(tl.int32)
    safe_abs = tl.maximum(abs_values, 1.0e-30)

    exp_unbiased = tl.floor(tl.log2(safe_abs))
    exp_unbiased = tl.minimum(tl.maximum(exp_unbiased, -6.0), 8.0)
    exp_scale = tl.exp2(exp_unbiased)
    mantissa_unrounded = (abs_values / exp_scale - 1.0) * 8.0
    mantissa_floor = tl.floor(mantissa_unrounded)
    mantissa_frac = mantissa_unrounded - mantissa_floor
    mantissa_odd = (mantissa_floor - 2.0 * tl.floor(mantissa_floor / 2.0)) != 0.0
    mantissa_round_up = (mantissa_frac > 0.5) | ((mantissa_frac == 0.5) & mantissa_odd)
    mantissa = (mantissa_floor + tl.where(mantissa_round_up, 1.0, 0.0)).to(tl.int32)
    exponent = (exp_unbiased + 7.0).to(tl.int32)
    carry = mantissa >= 8
    exponent = exponent + tl.where(carry, 1, 0)
    mantissa = tl.where(carry, 0, mantissa)
    exponent = tl.minimum(exponent, 15)
    mantissa = tl.where(exponent == 15, tl.minimum(mantissa, 6), mantissa)
    normal = (exponent << 3) | mantissa

    sub_unrounded = abs_values * 512.0
    sub_floor = tl.floor(sub_unrounded)
    sub_frac = sub_unrounded - sub_floor
    sub_odd = (sub_floor - 2.0 * tl.floor(sub_floor / 2.0)) != 0.0
    sub_round_up = (sub_frac > 0.5) | ((sub_frac == 0.5) & sub_odd)
    sub_mantissa = (sub_floor + tl.where(sub_round_up, 1.0, 0.0)).to(tl.int32)
    sub_mantissa = tl.minimum(tl.maximum(sub_mantissa, 0), 7)
    code = tl.where(abs_values < 0.015625, sub_mantissa, normal)
    code = tl.where(abs_values == 0.0, 0, code)
    return (code | sign).to(tl.uint8)


def _bf16_tensor(buffer: dict[str, Any], shape: tuple[int, ...]):
    return _typed_tensor(buffer, shape, _DLPACK_CODE_BFLOAT, 16)


def _f32_tensor(buffer: dict[str, Any], shape: tuple[int, ...]):
    return _typed_tensor(buffer, shape, _DLPACK_CODE_FLOAT, 32)


def _u8_tensor(buffer: dict[str, Any], shape: tuple[int, ...]):
    return _typed_tensor(buffer, shape, _DLPACK_CODE_UINT, 8)


def _u16_tensor(buffer: dict[str, Any], shape: tuple[int, ...]):
    return _typed_tensor(buffer, shape, _DLPACK_CODE_UINT, 16)


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
