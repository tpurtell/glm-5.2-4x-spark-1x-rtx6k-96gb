#!/usr/bin/env python3
"""Compare one combined packed-FP8 MLA launch with request-local launches.

The inputs are self-contained trace directories emitted by
GLMRT_REAL_FULL_PACKED_MLA_TRACE_DIR.  This deliberately uses production Q,
KV, KV-B, and packed-W8 O tensors while placing each request at a distinct
physical offset in one synthetic layer plane.
"""

from __future__ import annotations

import argparse
import ctypes
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from flashinfer.mla._sparse_mla_sm120 import sparse_mla_sm120_decode_dsv3_2


PAGE_ROWS = 64
PACKED_KV_ROW_BYTES = 656
W8_GROUP_SIZE = 256
MODEL_TYPE_GLM_NSA = 2
DECODE_BUCKETS = (128, 512, 1024, 2048)


def read_tensor(path: Path, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
    expected = 1
    for dimension in shape:
        expected *= dimension
    values = torch.from_file(str(path), shared=False, size=expected, dtype=dtype)
    if values.numel() != expected:
        raise ValueError(
            f"{path} has {values.numel()} {dtype} values, expected {expected} for {shape}"
        )
    return values.reshape(shape).to(device="cuda")


@dataclass
class Trace:
    directory: Path
    meta: dict[str, object]
    q_projected: torch.Tensor
    q_nope: torch.Tensor
    q_rope: torch.Tensor
    packed_kv: torch.Tensor
    kv_b: torch.Tensor
    o_weight: torch.Tensor
    o_scales: torch.Tensor

    @classmethod
    def load(cls, directory: Path) -> Trace:
        with (directory / "meta.json").open() as handle:
            meta = json.load(handle)
        if meta["format"] != "glmrt-packed-mla-trace-v1":
            raise ValueError(f"{directory} has unsupported format {meta['format']!r}")
        rows = int(meta["rows"])
        query_rows = int(meta["query_rows"])
        heads = int(meta["heads"])
        rank = int(meta["rank"])
        nope_dim = int(meta["nope_dim"])
        rope_dim = int(meta["rope_dim"])
        value_dim = int(meta["value_dim"])
        hidden_dim = int(meta["hidden_dim"])
        row_stride = int(meta["kv_row_stride_bytes"])
        if row_stride != PACKED_KV_ROW_BYTES:
            raise ValueError(
                f"{directory} row stride is {row_stride}, expected {PACKED_KV_ROW_BYTES}"
            )
        input_dim = heads * value_dim
        files = meta["files"]
        assert isinstance(files, dict)
        return cls(
            directory=directory,
            meta=meta,
            q_projected=read_tensor(
                directory / str(files["q_projected"]),
                torch.bfloat16,
                (query_rows, heads, nope_dim + rope_dim),
            ),
            q_nope=read_tensor(
                directory / str(files["q_nope"]),
                torch.bfloat16,
                (query_rows, heads, nope_dim),
            ),
            q_rope=read_tensor(
                directory / str(files["q_rope_rotated"]),
                torch.bfloat16,
                (query_rows, heads, rope_dim),
            ),
            packed_kv=read_tensor(
                directory / str(files["packed_kv"]),
                torch.uint8,
                (rows, row_stride),
            ),
            kv_b=read_tensor(
                directory / str(files["kv_b_weight"]),
                torch.bfloat16,
                (heads, nope_dim + value_dim, rank),
            ),
            o_weight=read_tensor(
                directory / str(files["o_weight_w8_packed"]),
                torch.int8,
                (hidden_dim, input_dim),
            ),
            o_scales=read_tensor(
                directory / str(files["o_weight_w8_scales"]),
                torch.float32,
                (input_dim // W8_GROUP_SIZE, hidden_dim),
            ),
        )


def configure_native(path: Path) -> ctypes.CDLL:
    native = ctypes.CDLL(str(path.resolve()))
    size = ctypes.c_size_t
    pointer = ctypes.c_void_p

    native.glmrt_cuda_matmul_bf16_strided_batched_cublas_async.argtypes = (
        pointer,
        pointer,
        pointer,
        size,
        size,
        size,
        size,
        size,
        size,
        size,
        pointer,
    )
    native.glmrt_cuda_matmul_bf16_strided_batched_cublas_async.restype = ctypes.c_int
    native.glmrt_cuda_linear_bf16_strided_batched_cublas_async.argtypes = (
        pointer,
        pointer,
        pointer,
        size,
        size,
        size,
        size,
        size,
        size,
        size,
        pointer,
    )
    native.glmrt_cuda_linear_bf16_strided_batched_cublas_async.restype = ctypes.c_int
    for name in (
        "glmrt_cuda_transpose_rows_heads_bf16_async",
        "glmrt_cuda_transpose_heads_rows_bf16_async",
    ):
        function = getattr(native, name)
        function.argtypes = (pointer, pointer, size, size, size, pointer)
        function.restype = ctypes.c_int
    native.glmrt_cuda_linear_w8a16_group256_m1_warp_packed_async.argtypes = (
        pointer,
        pointer,
        pointer,
        pointer,
        size,
        size,
        pointer,
    )
    native.glmrt_cuda_linear_w8a16_group256_m1_warp_packed_async.restype = ctypes.c_int
    native.glmrt_cuda_linear_w8a16_group256_m1_warp_packed_parity_batched_async.argtypes = (
        pointer,
        pointer,
        pointer,
        pointer,
        size,
        size,
        size,
        pointer,
    )
    native.glmrt_cuda_linear_w8a16_group256_m1_warp_packed_parity_batched_async.restype = (
        ctypes.c_int
    )
    return native


def check_status(status: int, operation: str) -> None:
    if status != 0:
        raise RuntimeError(f"{operation} failed with native status {status}")


def pointer(tensor: torch.Tensor, element_offset: int = 0) -> ctypes.c_void_p:
    return ctypes.c_void_p(
        tensor.data_ptr() + element_offset * tensor.element_size()
    )


def rounded_capacity(rows: int) -> int:
    return ((rows + PAGE_ROWS - 1) // PAGE_ROWS) * PAGE_ROWS


def decode_bucket(rows: int) -> int:
    for bucket in DECODE_BUCKETS:
        if rows <= bucket:
            return bucket
    raise ValueError(f"causal length {rows} exceeds packed decode buckets")


class Pipeline:
    def __init__(
        self,
        traces: list[Trace],
        bases: list[int],
        kv_plane: torch.Tensor,
        native: ctypes.CDLL,
    ) -> None:
        self.native = native
        self.traces = traces
        self.bases = bases
        meta = traces[0].meta
        self.heads = int(meta["heads"])
        self.rank = int(meta["rank"])
        self.nope_dim = int(meta["nope_dim"])
        self.rope_dim = int(meta["rope_dim"])
        self.value_dim = int(meta["value_dim"])
        self.hidden_dim = int(meta["hidden_dim"])
        self.scale = float(meta["scale"])
        self.rows = sum(int(trace.meta["query_rows"]) for trace in traces)
        self.bucket = decode_bucket(max(int(trace.meta["rows"]) for trace in traces))
        self.kv = kv_plane.view(
            kv_plane.shape[0] // PAGE_ROWS, PAGE_ROWS, PACKED_KV_ROW_BYTES
        )
        self.q_nope = torch.cat([trace.q_nope for trace in traces], dim=0).contiguous()
        self.q_rope = torch.cat([trace.q_rope for trace in traces], dim=0).contiguous()
        self.kv_b = traces[0].kv_b
        self.o_weight = traces[0].o_weight
        self.o_scales = traces[0].o_scales
        self.q_absorbed = torch.empty(
            (self.rows, self.heads, self.rank),
            dtype=torch.bfloat16,
            device="cuda",
        )
        self.q = torch.empty(
            (self.rows, self.heads, self.rank + self.rope_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )
        self.indices = torch.zeros(
            (self.rows, self.bucket), dtype=torch.int32, device="cuda"
        )
        lengths: list[int] = []
        query_offset = 0
        for trace, base in zip(traces, bases, strict=True):
            total_rows = int(trace.meta["rows"])
            query_rows = int(trace.meta["query_rows"])
            prefix_rows = total_rows - query_rows
            for query_index in range(query_rows):
                causal_rows = prefix_rows + query_index + 1
                self.indices[query_offset, :causal_rows] = (
                    torch.arange(causal_rows, dtype=torch.int32, device="cuda") + base
                )
                lengths.append(causal_rows)
                query_offset += 1
        self.lengths = torch.tensor(lengths, dtype=torch.int32, device="cuda")
        splits = self.bucket // PAGE_ROWS
        self.attention = torch.empty(
            (self.rows, self.heads, self.rank),
            dtype=torch.bfloat16,
            device="cuda",
        )
        self.lse = torch.empty(
            (self.rows, self.heads), dtype=torch.float32, device="cuda"
        )
        self.mid = torch.empty(
            (self.rows, self.heads, splits, self.rank),
            dtype=torch.bfloat16,
            device="cuda",
        )
        self.mid_lse = torch.empty(
            (self.rows, self.heads, splits),
            dtype=torch.float32,
            device="cuda",
        )
        self.attention_head_major = torch.empty_like(self.attention)
        self.values_head_major = torch.empty(
            (self.heads, self.rows, self.value_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )
        self.values = torch.empty(
            (self.rows, self.heads, self.value_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )
        self.hidden = torch.empty(
            (self.rows, self.hidden_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )

    def launch(self) -> None:
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        weight_head_stride = (self.nope_dim + self.value_dim) * self.rank
        q_nope_row = self.heads * self.nope_dim
        q_absorbed_row = self.heads * self.rank
        for row in range(self.rows):
            check_status(
                self.native.glmrt_cuda_matmul_bf16_strided_batched_cublas_async(
                    pointer(self.q_nope, row * q_nope_row),
                    pointer(self.kv_b),
                    pointer(self.q_absorbed, row * q_absorbed_row),
                    self.heads,
                    1,
                    self.nope_dim,
                    self.rank,
                    self.nope_dim,
                    weight_head_stride,
                    self.rank,
                    stream,
                ),
                f"absorbed Q row {row}",
            )
        self.q[:, :, : self.rank].copy_(self.q_absorbed)
        self.q[:, :, self.rank :].copy_(self.q_rope)
        sparse_mla_sm120_decode_dsv3_2(
            self.q,
            self.kv,
            self.indices,
            self.mid,
            self.mid_lse,
            self.attention,
            self.lse,
            self.scale,
            topk_length=self.lengths,
            model_type=MODEL_TYPE_GLM_NSA,
            chunks_per_block=1,
        )
        check_status(
            self.native.glmrt_cuda_transpose_rows_heads_bf16_async(
                pointer(self.attention),
                pointer(self.attention_head_major),
                self.rows,
                self.heads,
                self.rank,
                stream,
            ),
            "latent rows-to-heads transpose",
        )
        value_weight_offset = self.nope_dim * self.rank
        check_status(
            self.native.glmrt_cuda_linear_bf16_strided_batched_cublas_async(
                pointer(self.attention_head_major),
                pointer(self.kv_b, value_weight_offset),
                pointer(self.values_head_major),
                self.heads,
                self.rows,
                self.rank,
                self.value_dim,
                self.rows * self.rank,
                weight_head_stride,
                self.rows * self.value_dim,
                stream,
            ),
            "value expansion",
        )
        check_status(
            self.native.glmrt_cuda_transpose_heads_rows_bf16_async(
                pointer(self.values_head_major),
                pointer(self.values),
                self.rows,
                self.heads,
                self.value_dim,
                stream,
            ),
            "value heads-to-rows transpose",
        )
        input_dim = self.heads * self.value_dim
        if self.rows >= 4:
            check_status(
                self.native.glmrt_cuda_linear_w8a16_group256_m1_warp_packed_parity_batched_async(
                    pointer(self.values),
                    pointer(self.o_weight),
                    pointer(self.o_scales),
                    pointer(self.hidden),
                    self.rows,
                    input_dim,
                    self.hidden_dim,
                    stream,
                ),
                "packed W8 parity O projection",
            )
        else:
            for row in range(self.rows):
                check_status(
                    self.native.glmrt_cuda_linear_w8a16_group256_m1_warp_packed_async(
                        pointer(self.values, row * input_dim),
                        pointer(self.o_weight),
                        pointer(self.o_scales),
                        pointer(self.hidden, row * self.hidden_dim),
                        input_dim,
                        self.hidden_dim,
                        stream,
                    ),
                    f"packed W8 recurrent O projection row {row}",
                )

    def outputs(self) -> dict[str, torch.Tensor]:
        return {
            "absorbed_q": self.q_absorbed,
            "latent_attention": self.attention,
            "expanded_value": self.values,
            "hidden": self.hidden,
        }


def validate_compatible(traces: list[Trace]) -> None:
    keys = (
        "layer",
        "heads",
        "rank",
        "nope_dim",
        "rope_dim",
        "value_dim",
        "hidden_dim",
        "scale",
    )
    reference = traces[0].meta
    for trace in traces[1:]:
        for key in keys:
            if trace.meta[key] != reference[key]:
                raise ValueError(
                    f"trace mismatch for {key}: {reference[key]!r} != {trace.meta[key]!r}"
                )
        for name in ("kv_b", "o_weight", "o_scales"):
            if not torch.equal(getattr(trace, name), getattr(traces[0], name)):
                raise ValueError(f"{trace.directory} has a different {name} tensor")


def compare(name: str, combined: torch.Tensor, separate: torch.Tensor) -> bool:
    combined = combined.detach()
    separate = separate.detach()
    mismatch = combined.view(torch.int16) != separate.view(torch.int16)
    mismatch_count = int(mismatch.sum().item())
    if mismatch_count == 0:
        print(f"{name:20s} exact")
        return True
    first = int(mismatch.flatten().nonzero()[0].item())
    max_abs = float(
        (combined.float() - separate.float()).abs().max().item()
    )
    print(
        f"{name:20s} FAIL mismatches={mismatch_count}/{mismatch.numel()} "
        f"first_element={first} max_abs={max_abs:.8g}"
    )
    return False


def capture(pipeline: Pipeline) -> torch.cuda.CUDAGraph:
    pipeline.launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pipeline.launch()
    return graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs=2, type=Path)
    parser.add_argument(
        "--native-library",
        type=Path,
        default=Path("native/build-cuda-rdma-coordinator-aot/libglmrt_native.so"),
    )
    parser.add_argument(
        "--physical-bases",
        default="64,256",
        help="comma-separated, page-aligned physical row bases",
    )
    parser.add_argument("--skip-graphs", action="store_true")
    args = parser.parse_args()
    bases = [int(value) for value in args.physical_bases.split(",")]
    if len(bases) != len(args.traces) or any(base % PAGE_ROWS for base in bases):
        parser.error("--physical-bases must provide two page-aligned values")

    traces = [Trace.load(directory) for directory in args.traces]
    validate_compatible(traces)
    ranges = [
        (base, base + int(trace.meta["rows"]))
        for base, trace in zip(bases, traces, strict=True)
    ]
    if max(ranges[0][0], ranges[1][0]) < min(ranges[0][1], ranges[1][1]):
        parser.error("physical trace ranges overlap")
    capacity = rounded_capacity(max(end for _, end in ranges))
    kv_plane = torch.zeros(
        (capacity, PACKED_KV_ROW_BYTES), dtype=torch.uint8, device="cuda"
    )
    for trace, base in zip(traces, bases, strict=True):
        rows = int(trace.meta["rows"])
        kv_plane[base : base + rows].copy_(trace.packed_kv)

    native = configure_native(args.native_library)
    combined = Pipeline(traces, bases, kv_plane, native)
    singles = [
        Pipeline([trace], [base], kv_plane, native)
        for trace, base in zip(traces, bases, strict=True)
    ]
    combined.launch()
    for pipeline in singles:
        pipeline.launch()
    torch.cuda.synchronize()

    passed = True
    for name, tensor in combined.outputs().items():
        separate = torch.cat(
            [pipeline.outputs()[name] for pipeline in singles], dim=0
        )
        passed &= compare(name, tensor, separate)

    if not args.skip_graphs:
        combined_graph = capture(combined)
        single_graphs = [capture(pipeline) for pipeline in singles]
        combined_graph.replay()
        for graph in single_graphs:
            graph.replay()
        torch.cuda.synchronize()
        for name, tensor in combined.outputs().items():
            separate = torch.cat(
                [pipeline.outputs()[name] for pipeline in singles], dim=0
            )
            passed &= compare(f"graph_{name}", tensor, separate)

    print(
        f"geometry combined_M={combined.rows} bucket={combined.bucket} "
        f"capacity={capacity} bases={bases}"
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
