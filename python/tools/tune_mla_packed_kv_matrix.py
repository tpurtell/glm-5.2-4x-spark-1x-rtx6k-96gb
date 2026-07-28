#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


REFERENCE_ROOT = Path(__file__).resolve().parents[1] / "reference"
if str(REFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFERENCE_ROOT))

from glmrt_reference.b12x_mla_capture import (  # noqa: E402
    capture_flashinfer_compressed_mla_decode_chunk,
    capture_flashinfer_packed_fp8_mla_decode,
    prepare_flashinfer_compressed_mla_decode_chunk,
    prepare_flashinfer_packed_fp8_mla_decode,
)


HEADS = 64
RANK = 512
ROPE_DIM = 64
KV_WIDTH = RANK + ROPE_DIM
DSA_VALUES = 128
DSA_BYTES = DSA_VALUES * 2
BF16_MAIN_BYTES = KV_WIDTH * 2
FP8_MAIN_BYTES = RANK + 4 * 4 + ROPE_DIM * 2
NVFP4_MAIN_BYTES = RANK // 2 + RANK // 32 + ROPE_DIM * 2
MAIN_BYTES = {
    "bf16": BF16_MAIN_BYTES,
    "fp8": FP8_MAIN_BYTES,
    "nvfp4": NVFP4_MAIN_BYTES,
}
DIRECT_FP8_BUCKETS = (128, 512, 1_024, 2_048)
MAX_CHUNK_ROWS = 2_048
EXACT_TAIL_ROWS = 32
DSA_LAYERS = 21
TOTAL_LAYERS = 78
SCALE = KV_WIDTH**-0.5


def parse_int_list(raw: str, label: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in raw.split(",") if item)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{label} must be comma-separated integers"
        ) from error
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError(f"{label} values must be positive")
    return values


def parse_formats(raw: str) -> tuple[str, ...]:
    values = tuple(item for item in raw.split(",") if item)
    unknown = sorted(set(values) - MAIN_BYTES.keys())
    if not values or unknown:
        raise argparse.ArgumentTypeError(
            f"formats must be drawn from {sorted(MAIN_BYTES)}, got {unknown}"
        )
    return values


def check_status(lib: ctypes.CDLL, status: int, action: str) -> None:
    if status == 0:
        return
    error = ctypes.create_string_buffer(512)
    lib.glmrt_last_error(error, len(error))
    raise RuntimeError(
        f"{action} failed with status {status}: {error.value.decode()}"
    )


def configure(lib: ctypes.CDLL, symbol: str, argtypes):
    function = getattr(lib, symbol)
    function.argtypes = argtypes
    function.restype = ctypes.c_int
    return function


def pointer(tensor: torch.Tensor, byte_offset: int = 0) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr() + byte_offset)


def stream_pointer() -> ctypes.c_void_p:
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def descriptor(tensor: torch.Tensor) -> dict[str, int]:
    return {
        "ptr": tensor.data_ptr(),
        "bytes": tensor.numel() * tensor.element_size(),
        "device_id": tensor.device.index or 0,
    }


def capture(operation: Callable[[], None]) -> torch.cuda.CUDAGraph:
    operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()
    return graph


def measure(
    graph: torch.cuda.CUDAGraph,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, float | list[float]]:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return {
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "samples_ms": samples,
    }


def chunk_plan(rows: int) -> tuple[tuple[int, int], ...]:
    chunks = []
    offset = 0
    while offset < rows:
        remaining = rows - offset
        if remaining <= EXACT_TAIL_ROWS:
            chunk_rows = remaining
        else:
            chunk_rows = min(1 << (remaining.bit_length() - 1), MAX_CHUNK_ROWS)
        chunks.append((offset, chunk_rows))
        offset += chunk_rows
    return tuple(chunks)


def direct_fp8_bucket(rows: int) -> int:
    for bucket in DIRECT_FP8_BUCKETS:
        if rows <= bucket:
            return bucket
    raise ValueError(f"direct packed FP8 does not support {rows} rows")


def bf16_view(cache: torch.Tensor, main_bytes: int) -> torch.Tensor:
    return cache[:, :main_bytes].view(torch.bfloat16)


@dataclass
class CacheFixture:
    projected: torch.Tensor
    compact: dict[str, torch.Tensor]
    with_dsa: dict[str, torch.Tensor]

    def source(self, fmt: str, dsa: bool) -> torch.Tensor:
        return self.with_dsa[fmt] if dsa else self.compact[fmt]

    def row_stride(self, fmt: str, dsa: bool) -> int:
        return MAIN_BYTES[fmt] + (DSA_BYTES if dsa else 0)


@dataclass
class Workspace:
    q_nope: torch.Tensor
    q_rope: torch.Tensor
    q_combined: torch.Tensor
    bf16_chunk: torch.Tensor
    fp8_chunk: torch.Tensor
    partial: torch.Tensor
    partial_lse: torch.Tensor
    output: torch.Tensor
    output_lse: torch.Tensor
    compressed_workspace: torch.Tensor
    fp8_indices: torch.Tensor
    fp8_lengths: torch.Tensor
    fp8_out_lse: torch.Tensor
    fp8_mid_out: torch.Tensor
    fp8_mid_lse: torch.Tensor


def make_fixture(
    lib: ctypes.CDLL,
    pack_fp8,
    pack_nvfp4,
    rows: int,
    seed: int,
) -> CacheFixture:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    projected = torch.randn(
        (rows, KV_WIDTH),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    ) * 0.05
    compact = {
        "bf16": torch.empty((rows, BF16_MAIN_BYTES), dtype=torch.uint8, device="cuda"),
        "fp8": torch.empty((rows, FP8_MAIN_BYTES), dtype=torch.uint8, device="cuda"),
        "nvfp4": torch.empty((rows, NVFP4_MAIN_BYTES), dtype=torch.uint8, device="cuda"),
    }
    bf16_view(compact["bf16"], BF16_MAIN_BYTES).copy_(projected)
    check_status(
        lib,
        pack_fp8(
            pointer(projected),
            pointer(compact["fp8"]),
            rows,
            BF16_MAIN_BYTES,
            FP8_MAIN_BYTES,
            stream_pointer(),
        ),
        "pack FP8 KV fixture",
    )
    check_status(
        lib,
        pack_nvfp4(
            pointer(projected),
            pointer(compact["nvfp4"]),
            rows,
            BF16_MAIN_BYTES,
            NVFP4_MAIN_BYTES,
            stream_pointer(),
        ),
        "pack NVFP4 KV fixture",
    )
    with_dsa = {}
    for fmt, main in MAIN_BYTES.items():
        cache = torch.zeros(
            (rows, main + DSA_BYTES), dtype=torch.uint8, device="cuda"
        )
        cache[:, :main].copy_(compact[fmt])
        cache[:, main:].copy_(
            torch.randint(
                0,
                256,
                (rows, DSA_BYTES),
                dtype=torch.uint8,
                device="cuda",
                generator=generator,
            )
        )
        with_dsa[fmt] = cache
    torch.cuda.synchronize()
    return CacheFixture(projected=projected, compact=compact, with_dsa=with_dsa)


def make_workspace(target_rows: int, seed: int) -> Workspace:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    q_nope = torch.randn(
        (target_rows, HEADS, RANK),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    ) * 0.05
    q_rope = torch.randn(
        (target_rows, HEADS, ROPE_DIM),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    ) * 0.05
    return Workspace(
        q_nope=q_nope,
        q_rope=q_rope,
        q_combined=torch.empty(
            (target_rows, HEADS, KV_WIDTH), dtype=torch.bfloat16, device="cuda"
        ),
        bf16_chunk=torch.empty(
            (MAX_CHUNK_ROWS, KV_WIDTH), dtype=torch.bfloat16, device="cuda"
        ),
        fp8_chunk=torch.empty(
            (MAX_CHUNK_ROWS, FP8_MAIN_BYTES), dtype=torch.uint8, device="cuda"
        ),
        partial=torch.empty((HEADS, RANK), dtype=torch.bfloat16, device="cuda"),
        partial_lse=torch.empty(HEADS, dtype=torch.float32, device="cuda"),
        output=torch.empty(
            (target_rows, HEADS, RANK), dtype=torch.bfloat16, device="cuda"
        ),
        output_lse=torch.empty(
            (target_rows, HEADS), dtype=torch.float32, device="cuda"
        ),
        compressed_workspace=torch.empty(
            32 * 1024 * 1024, dtype=torch.uint8, device="cuda"
        ),
        fp8_indices=torch.arange(
            MAX_CHUNK_ROWS, dtype=torch.int32, device="cuda"
        ).view(1, -1),
        fp8_lengths=torch.empty(target_rows, dtype=torch.int32, device="cuda"),
        fp8_out_lse=torch.empty(
            (target_rows, HEADS), dtype=torch.float32, device="cuda"
        ),
        fp8_mid_out=torch.empty(
            (target_rows, HEADS, MAX_CHUNK_ROWS // 64, RANK),
            dtype=torch.bfloat16,
            device="cuda",
        ),
        fp8_mid_lse=torch.empty(
            (target_rows, HEADS, MAX_CHUNK_ROWS // 64),
            dtype=torch.float32,
            device="cuda",
        ),
    )


class DecodeCase:
    def __init__(
        self,
        lib: ctypes.CDLL,
        unpack_fp8,
        unpack_nvfp4,
        merge,
        fixture: CacheFixture,
        workspace: Workspace,
        fmt: str,
        dsa: bool,
        context_rows: int,
        target_rows: int,
    ) -> None:
        self.lib = lib
        self.unpack_fp8 = unpack_fp8
        self.unpack_nvfp4 = unpack_nvfp4
        self.merge = merge
        self.fixture = fixture
        self.workspace = workspace
        self.fmt = fmt
        self.dsa = dsa
        self.context_rows = context_rows
        self.target_rows = target_rows
        self.source = fixture.source(fmt, dsa)
        self.source_stride = fixture.row_stride(fmt, dsa)
        self.direct_fp8 = fmt == "fp8" and context_rows + target_rows <= MAX_CHUNK_ROWS
        self.compressed_contexts: dict[tuple[int, int], dict] = {}
        self.direct_contexts: list[dict] = []
        self.direct_bucket = (
            direct_fp8_bucket(context_rows + target_rows) if self.direct_fp8 else None
        )
        self.workspace.fp8_lengths[:target_rows].copy_(
            torch.arange(
                context_rows + 1,
                context_rows + target_rows + 1,
                dtype=torch.int32,
                device="cuda",
            )
        )
        self._build_contexts()

    def _build_contexts(self) -> None:
        if self.direct_fp8:
            assert self.direct_bucket is not None
            for row in range(self.target_rows):
                splits = self.direct_bucket // 64
                self.direct_contexts.append(
                    {
                        "cuda_stream": torch.cuda.current_stream().cuda_stream,
                        "buffers": {
                            "q": descriptor(self.workspace.q_combined[row : row + 1]),
                            "kv": descriptor(
                                self.workspace.fp8_chunk[: self.direct_bucket]
                            ),
                            "indices": descriptor(
                                self.workspace.fp8_indices[:, : self.direct_bucket]
                            ),
                            "topk_length": descriptor(
                                self.workspace.fp8_lengths[row : row + 1]
                            ),
                            "output": descriptor(
                                self.workspace.output[row : row + 1]
                            ),
                            "out_lse": descriptor(
                                self.workspace.fp8_out_lse[row : row + 1]
                            ),
                            "mid_out": descriptor(
                                self.workspace.fp8_mid_out[
                                    row : row + 1, :, :splits
                                ]
                            ),
                            "mid_lse": descriptor(
                                self.workspace.fp8_mid_lse[
                                    row : row + 1, :, :splits
                                ]
                            ),
                        },
                    }
                )
            return

        chunk_sizes = {
            chunk_rows
            for row in range(self.target_rows)
            for _, chunk_rows in chunk_plan(self.context_rows + row + 1)
        }
        for row in range(self.target_rows):
            for chunk_rows in chunk_sizes:
                self.compressed_contexts[(row, chunk_rows)] = {
                    "cuda_stream": torch.cuda.current_stream().cuda_stream,
                    "buffers": {
                        "q_nope": descriptor(
                            self.workspace.q_nope[row : row + 1]
                        ),
                        "q_rope": descriptor(
                            self.workspace.q_rope[row : row + 1]
                        ),
                        "kv": descriptor(self.workspace.bf16_chunk[:chunk_rows]),
                        "partial": descriptor(self.workspace.partial),
                        "partial_lse": descriptor(self.workspace.partial_lse),
                        "workspace": descriptor(
                            self.workspace.compressed_workspace
                        ),
                    },
                }

    def prepare(self) -> None:
        if self.direct_fp8:
            assert self.direct_bucket is not None
            kwargs = {
                "bucket_rows": self.direct_bucket,
                "heads": HEADS,
                "nope_dim": RANK,
                "rope_dim": ROPE_DIM,
                "scale": SCALE,
            }
            for context in self.direct_contexts:
                prepare_flashinfer_packed_fp8_mla_decode(context, **kwargs)
            self.workspace.fp8_lengths[: self.target_rows].copy_(
                torch.arange(
                    self.context_rows + 1,
                    self.context_rows + self.target_rows + 1,
                    dtype=torch.int32,
                    device="cuda",
                )
            )
            return
        prepared = set()
        for (_, chunk_rows), context in self.compressed_contexts.items():
            if chunk_rows in prepared:
                continue
            prepare_flashinfer_compressed_mla_decode_chunk(
                context,
                rows=chunk_rows,
                heads=HEADS,
                nope_dim=RANK,
                rope_dim=ROPE_DIM,
                scale=SCALE,
            )
            prepared.add(chunk_rows)

    def _stage_compressed(self, offset: int, rows: int) -> None:
        destination = self.workspace.bf16_chunk[:rows]
        if self.fmt == "bf16":
            source = bf16_view(self.source, BF16_MAIN_BYTES)
            destination.copy_(source[offset : offset + rows])
            return
        unpack = self.unpack_fp8 if self.fmt == "fp8" else self.unpack_nvfp4
        check_status(
            self.lib,
            unpack(
                pointer(self.source, offset * self.source_stride),
                pointer(destination),
                rows,
                self.source_stride,
                BF16_MAIN_BYTES,
                stream_pointer(),
            ),
            f"unpack {self.fmt} compressed attention chunk",
        )

    def _compressed_attention(self, row: int, chunk_rows: int) -> None:
        context = self.compressed_contexts[(row, chunk_rows)]
        context["cuda_stream"] = torch.cuda.current_stream().cuda_stream
        capture_flashinfer_compressed_mla_decode_chunk(
            context,
            rows=chunk_rows,
            heads=HEADS,
            nope_dim=RANK,
            rope_dim=ROPE_DIM,
            scale=SCALE,
        )

    def _copy_first_partial(self, row: int) -> None:
        self.workspace.output[row].copy_(self.workspace.partial)
        self.workspace.output_lse[row].copy_(self.workspace.partial_lse)

    def _merge_partial(self, row: int) -> None:
        check_status(
            self.lib,
            self.merge(
                pointer(self.workspace.output[row]),
                pointer(self.workspace.output_lse[row]),
                pointer(self.workspace.partial),
                pointer(self.workspace.partial_lse),
                HEADS,
                RANK,
                stream_pointer(),
            ),
            "merge compressed MLA state",
        )

    def _launch_direct_fp8(self) -> None:
        assert self.direct_bucket is not None
        visible_rows = self.context_rows + self.target_rows
        self.workspace.q_combined[: self.target_rows, :, :RANK].copy_(
            self.workspace.q_nope[: self.target_rows]
        )
        self.workspace.q_combined[: self.target_rows, :, RANK:].copy_(
            self.workspace.q_rope[: self.target_rows]
        )
        self.workspace.fp8_chunk[:visible_rows].copy_(
            self.source[:visible_rows, :FP8_MAIN_BYTES]
        )
        kwargs = {
            "bucket_rows": self.direct_bucket,
            "heads": HEADS,
            "nope_dim": RANK,
            "rope_dim": ROPE_DIM,
            "scale": SCALE,
        }
        for context in self.direct_contexts:
            context["cuda_stream"] = torch.cuda.current_stream().cuda_stream
            capture_flashinfer_packed_fp8_mla_decode(context, **kwargs)

    def stage_probe(self) -> None:
        if self.direct_fp8:
            visible_rows = self.context_rows + self.target_rows
            self.workspace.fp8_chunk[:visible_rows].copy_(
                self.source[:visible_rows, :FP8_MAIN_BYTES]
            )
            return
        chunk_rows = chunk_plan(self.context_rows + 1)[0][1]
        self._stage_compressed(0, chunk_rows)

    def attention_probe(self) -> None:
        if self.direct_fp8:
            assert self.direct_bucket is not None
            self.workspace.q_combined[0:1, :, :RANK].copy_(
                self.workspace.q_nope[0:1]
            )
            self.workspace.q_combined[0:1, :, RANK:].copy_(
                self.workspace.q_rope[0:1]
            )
            context = self.direct_contexts[0]
            context["cuda_stream"] = torch.cuda.current_stream().cuda_stream
            capture_flashinfer_packed_fp8_mla_decode(
                context,
                bucket_rows=self.direct_bucket,
                heads=HEADS,
                nope_dim=RANK,
                rope_dim=ROPE_DIM,
                scale=SCALE,
            )
            return
        chunk_rows = chunk_plan(self.context_rows + 1)[0][1]
        self._compressed_attention(0, chunk_rows)

    def merge_probe(self) -> None:
        self._merge_partial(0)

    @property
    def component_shape(self) -> dict[str, int | str]:
        if self.direct_fp8:
            assert self.direct_bucket is not None
            return {
                "attention_rows": self.direct_bucket,
                "cache_rows_staged": self.context_rows + self.target_rows,
                "mode": "direct-packed",
            }
        chunk_rows = chunk_plan(self.context_rows + 1)[0][1]
        return {
            "attention_rows": chunk_rows,
            "cache_rows_staged": chunk_rows,
            "mode": "compressed-chunk",
        }

    def launch(self) -> None:
        if self.direct_fp8:
            self._launch_direct_fp8()
            return
        for row in range(self.target_rows):
            visible_rows = self.context_rows + row + 1
            for index, (offset, chunk_rows) in enumerate(chunk_plan(visible_rows)):
                self._stage_compressed(offset, chunk_rows)
                self._compressed_attention(row, chunk_rows)
                if index == 0:
                    self._copy_first_partial(row)
                else:
                    self._merge_partial(row)

    @property
    def backend(self) -> str:
        if self.direct_fp8:
            return "flashinfer-packed-fp8-sm120"
        if self.fmt == "bf16":
            return "flashinfer-compressed-bf16"
        return f"{self.fmt}-unpack-plus-flashinfer-compressed-bf16"

    @property
    def graph_operations(self) -> dict[str, int]:
        if self.direct_fp8:
            return {
                "cache_stages": 1,
                "codec_launches": 0,
                "attention_launches": self.target_rows,
                "merge_launches": 0,
            }
        chunks = sum(
            len(chunk_plan(self.context_rows + row + 1))
            for row in range(self.target_rows)
        )
        return {
            "cache_stages": chunks,
            "codec_launches": chunks if self.fmt != "bf16" else 0,
            "attention_launches": chunks,
            "merge_launches": chunks - self.target_rows,
        }


def error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual_f32 = actual.float()
    reference_f32 = reference.float()
    delta = actual_f32 - reference_f32
    reference_norm = float(reference_f32.norm())
    relative_l2 = float(delta.norm()) / max(reference_norm, 1.0e-12)
    cosine = float(
        torch.nn.functional.cosine_similarity(
            actual_f32.flatten(), reference_f32.flatten(), dim=0
        )
    )
    return {
        "cosine": cosine,
        "max_abs": float(delta.abs().max()),
        "relative_l2": relative_l2,
    }


def memory_accounting(
    contexts: tuple[int, ...],
    gpu_gib: float,
    resident_gib: float,
    reserve_gib: float,
) -> list[dict[str, float | int | str]]:
    results = []
    available_bytes = max(0.0, gpu_gib - resident_gib - reserve_gib) * 2**30
    for fmt, main_bytes in MAIN_BYTES.items():
        bytes_per_token = TOTAL_LAYERS * main_bytes + DSA_LAYERS * DSA_BYTES
        max_tokens = int(available_bytes // bytes_per_token)
        for context in contexts:
            cache_bytes = context * bytes_per_token
            results.append(
                {
                    "cache_gib": cache_bytes / 2**30,
                    "context_rows": context,
                    "dsa_layers": DSA_LAYERS,
                    "format": fmt,
                    "main_bytes_per_layer_token": main_bytes,
                    "total_bytes_per_token": bytes_per_token,
                    "estimated_max_tokens_after_reserve": max_tokens,
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark packed/compressed MLA decode across cache dtypes, context "
            "lengths, MTP widths, and optional DSA row tails."
        )
    )
    parser.add_argument("--native-lib", type=Path, required=True)
    parser.add_argument("--contexts", default="1024,16384,131072,262144")
    parser.add_argument("--target-rows", default="1,4,6,8")
    parser.add_argument("--formats", type=parse_formats, default=tuple(MAIN_BYTES))
    parser.add_argument("--dsa", default="0,1")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--work-rows", type=int, default=65_536)
    parser.add_argument("--max-iterations", type=int, default=32)
    parser.add_argument("--component-iterations", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--gpu-gib", type=float, default=96.0)
    parser.add_argument("--resident-gib", type=float, default=43.0)
    parser.add_argument("--reserve-gib", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    contexts = parse_int_list(args.contexts, "contexts")
    target_rows_values = parse_int_list(args.target_rows, "target-rows")
    formats = args.formats if isinstance(args.formats, tuple) else parse_formats(args.formats)
    if formats[0] != "bf16":
        parser.error("--formats must list bf16 first as the validation reference")
    try:
        dsa_values = tuple(bool(int(item)) for item in args.dsa.split(","))
    except ValueError as error:
        raise SystemExit("--dsa values must be 0 or 1") from error
    if not dsa_values or any(item not in ("0", "1") for item in args.dsa.split(",")):
        parser.error("--dsa values must be 0 or 1")
    if max(target_rows_values) > 8:
        parser.error("target-rows must not exceed 8")
    if (
        min(
            args.work_rows,
            args.max_iterations,
            args.component_iterations,
            args.repeats,
        )
        < 1
        or args.warmup < 0
    ):
        parser.error(
            "work-rows/max-iterations/repeats must be positive and warmup nonnegative"
        )
    if min(args.gpu_gib, args.resident_gib, args.reserve_gib) < 0:
        parser.error("memory GiB values must be nonnegative")

    torch.manual_seed(args.seed)
    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    lib = ctypes.CDLL(str(args.native_lib.resolve()))
    lib.glmrt_last_error.argtypes = (ctypes.c_char_p, ctypes.c_size_t)
    lib.glmrt_last_error.restype = ctypes.c_int
    pack_args = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    pack_fp8 = configure(
        lib, "glmrt_cuda_mla_kv_pack_fp8_ds_mla_async", pack_args
    )
    pack_nvfp4 = configure(
        lib, "glmrt_cuda_mla_kv_pack_mxfp4_ds_mla_async", pack_args
    )
    unpack_fp8 = configure(
        lib, "glmrt_cuda_mla_kv_unpack_fp8_ds_mla_async", pack_args
    )
    unpack_nvfp4 = configure(
        lib, "glmrt_cuda_mla_kv_unpack_mxfp4_ds_mla_async", pack_args
    )
    merge = configure(
        lib,
        "glmrt_cuda_mla_merge_state_bf16_async",
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ),
    )

    max_rows = max(contexts) + max(target_rows_values)
    fixture = make_fixture(
        lib, pack_fp8, pack_nvfp4, max_rows, args.seed + 1_000
    )
    results = []
    component_timings: dict[str, dict] = {}
    references: dict[tuple[int, int, bool], torch.Tensor] = {}
    no_dsa_outputs: dict[tuple[str, int, int], torch.Tensor] = {}

    for context_rows in contexts:
        for target_rows in target_rows_values:
            for dsa in dsa_values:
                for fmt in formats:
                    workspace = make_workspace(
                        target_rows,
                        args.seed + context_rows + target_rows * 100,
                    )
                    case = DecodeCase(
                        lib,
                        unpack_fp8,
                        unpack_nvfp4,
                        merge,
                        fixture,
                        workspace,
                        fmt,
                        dsa,
                        context_rows,
                        target_rows,
                    )
                    case.prepare()
                    torch.cuda.synchronize()
                    component_shape = case.component_shape
                    component_key = ":".join(
                        (
                            fmt,
                            f"dsa={int(dsa)}",
                            str(component_shape["mode"]),
                            f"stage={component_shape['cache_rows_staged']}",
                            f"attention={component_shape['attention_rows']}",
                        )
                    )
                    if component_key not in component_timings:
                        stage_graph = capture(case.stage_probe)
                        stage_graph.replay()
                        torch.cuda.synchronize()
                        attention_graph = capture(case.attention_probe)
                        attention_graph.replay()
                        torch.cuda.synchronize()
                        component = {
                            "shape": component_shape,
                            "stage_or_unpack": measure(
                                stage_graph,
                                args.warmup,
                                args.component_iterations,
                                args.repeats,
                            ),
                            "attention_core": measure(
                                attention_graph,
                                args.warmup,
                                args.component_iterations,
                                args.repeats,
                            ),
                        }
                        if not case.direct_fp8:
                            workspace.output[0].copy_(workspace.partial)
                            workspace.output_lse[0].copy_(workspace.partial_lse)
                            merge_graph = capture(case.merge_probe)
                            component["merge"] = measure(
                                merge_graph,
                                args.warmup,
                                args.component_iterations,
                                args.repeats,
                            )
                            del merge_graph
                        component_timings[component_key] = component
                        del stage_graph, attention_graph
                    graph = capture(case.launch)
                    graph.replay()
                    torch.cuda.synchronize()
                    output = workspace.output[:target_rows].clone()
                    key = (context_rows, target_rows, dsa)
                    if fmt == "bf16":
                        references[key] = output
                    reference = references.get(key)
                    if reference is None:
                        raise RuntimeError(
                            "BF16 must be listed before compressed formats for validation"
                        )
                    metrics = error_metrics(output, reference)
                    dsa_key = (fmt, context_rows, target_rows)
                    if not dsa:
                        no_dsa_outputs[dsa_key] = output
                        dsa_exact = True
                    else:
                        dsa_exact = bool(
                            torch.equal(output, no_dsa_outputs[dsa_key])
                        )
                    iterations = max(
                        1,
                        min(
                            args.max_iterations,
                            args.work_rows
                            // max(1, context_rows * target_rows),
                        ),
                    )
                    timing = measure(
                        graph,
                        args.warmup,
                        iterations,
                        args.repeats,
                    )
                    result = {
                        "backend": case.backend,
                        "benchmark": "mla_packed_kv_decode_matrix",
                        "cache_bytes": (
                            context_rows + target_rows
                        )
                        * case.source_stride,
                        "cache_format": fmt,
                        "cache_row_stride_bytes": case.source_stride,
                        "component_timing_key": component_key,
                        "cached_context_rows": context_rows,
                        "dsa_tail": dsa,
                        "dsa_tail_output_exact": dsa_exact,
                        "error_vs_bf16": metrics,
                        "graph_operations": case.graph_operations,
                        "iterations": iterations,
                        "target_rows": target_rows,
                        "timing": timing,
                    }
                    if not dsa_exact:
                        raise RuntimeError(
                            f"DSA tail changed main attention output for {dsa_key}"
                        )
                    results.append(result)
                    print(json.dumps(result, sort_keys=True), flush=True)
                    del graph, case, workspace
                    torch.cuda.empty_cache()

    accounting = memory_accounting(
        tuple(sorted(set(contexts))),
        args.gpu_gib,
        args.resident_gib,
        args.reserve_gib,
    )
    report = {
        "benchmark": "mla_packed_kv_decode_matrix_summary",
        "formats": formats,
        "gpu": properties.name,
        "memory_accounting": accounting,
        "component_timings": component_timings,
        "memory_assumptions": {
            "dsa_layers": DSA_LAYERS,
            "gpu_gib": args.gpu_gib,
            "resident_model_and_static_gib": args.resident_gib,
            "runtime_reserve_gib": args.reserve_gib,
            "total_layers": TOTAL_LAYERS,
        },
        "note": (
            "Benchmark-only latent-attention matrix. It excludes absorbed-query "
            "and value-projection GEMMs and does not alter serving dispatch."
        ),
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
