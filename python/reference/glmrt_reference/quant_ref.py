from __future__ import annotations

from collections.abc import Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised by no-torch hosts
    torch = None


NVFP4_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

NVFP4_E2M1_CODEBOOK = (
    torch.tensor(NVFP4_E2M1_VALUES, dtype=torch.float32) if torch is not None else None
)


def _require_torch():
    if torch is None:
        raise ModuleNotFoundError("torch is required for tensor quantize/dequantize helpers")
    return torch


def nvfp4_quantize(x: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
    torch = _require_torch()
    codebook = NVFP4_E2M1_CODEBOOK.to(device=x.device)
    scaled = x.float() / torch.as_tensor(scale, dtype=torch.float32, device=x.device)
    distances = (scaled.unsqueeze(-1) - codebook).abs()
    return distances.argmin(dim=-1).to(torch.uint8)


def nvfp4_dequantize(codes: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
    torch = _require_torch()
    codebook = NVFP4_E2M1_CODEBOOK.to(device=codes.device)
    values = codebook[codes.long()]
    return values * torch.as_tensor(scale, dtype=torch.float32, device=codes.device)


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    torch = _require_torch()
    flat = codes.flatten().to(torch.uint8)
    if flat.numel() % 2:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8, device=flat.device)])
    low = flat[0::2] & 0x0F
    high = (flat[1::2] & 0x0F) << 4
    return low | high


def unpack_nibbles(packed: torch.Tensor, count: int) -> torch.Tensor:
    torch = _require_torch()
    packed = packed.flatten().to(torch.uint8)
    out = torch.empty(packed.numel() * 2, dtype=torch.uint8, device=packed.device)
    out[0::2] = packed & 0x0F
    out[1::2] = (packed >> 4) & 0x0F
    return out[:count]


def nvfp4_e2m1_code_value(code: int) -> float:
    return NVFP4_E2M1_VALUES[code & 0x0F]


def f8e4m3_byte_to_float(byte: int) -> float:
    byte &= 0xFF
    if byte in (0, 0x80):
        return 0.0
    sign = 1.0 if byte & 0x80 == 0 else -1.0
    exponent = (byte >> 3) & 0x0F
    mantissa = byte & 0x07
    significand = mantissa / 8.0 if exponent == 0 else 1.0 + mantissa / 8.0
    exponent_power = -6 if exponent == 0 else exponent - 7
    return sign * significand * (2.0**exponent_power)


def unpack_low_first_nibbles_bytes(
    packed: bytes | bytearray | Sequence[int],
    count: int,
) -> list[int]:
    values: list[int] = []
    for byte in packed:
        values.append(byte & 0x0F)
        if len(values) == count:
            return values
        values.append((byte >> 4) & 0x0F)
        if len(values) == count:
            return values
    if len(values) < count:
        raise ValueError(f"packed bytes contain {len(values)} nibbles, need {count}")
    return values


def decode_packed_nvfp4_values(
    packed: bytes | bytearray | Sequence[int],
    scale: bytes | bytearray | Sequence[int],
    scale_2: float,
    count: int,
) -> list[float]:
    if count < 0:
        raise ValueError("count must be non-negative")
    required_scale_bytes = (count + 15) // 16
    if len(scale) < required_scale_bytes:
        raise ValueError(f"scale bytes contain {len(scale)} values, need {required_scale_bytes}")
    codes = unpack_low_first_nibbles_bytes(packed, count)
    return [
        nvfp4_e2m1_code_value(code)
        * f8e4m3_byte_to_float(scale[value_index // 16])
        * scale_2
        for value_index, code in enumerate(codes)
    ]
