import pytest

torch = pytest.importorskip("torch")

from glmrt_reference.quant_ref import (
    nvfp4_dequantize,
    nvfp4_quantize,
    pack_nibbles,
    unpack_nibbles,
)


def test_pack_unpack_nibbles_roundtrip_odd_count():
    codes = torch.tensor([0, 1, 2, 15, 8], dtype=torch.uint8)
    packed = pack_nibbles(codes)
    unpacked = unpack_nibbles(packed, count=codes.numel())
    torch.testing.assert_close(unpacked, codes)


def test_nvfp4_quantize_dequantize_uses_codebook_values():
    x = torch.tensor([-6.1, -1.4, 0.2, 2.9, 5.8])
    codes = nvfp4_quantize(x, scale=1.0)
    deq = nvfp4_dequantize(codes, scale=1.0)
    assert deq.shape == x.shape
    assert torch.max(torch.abs(deq - x)) <= 0.3
