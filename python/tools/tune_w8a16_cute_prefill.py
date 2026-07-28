#!/usr/bin/env python3
"""Benchmark a CuTe W8A16 GEMM that dequantizes each weight tile in shared memory.

This is the first tensor-core W8A16 prefill prototype.  It consumes GLMRT's
single K-major group-256 W8 representation directly and never materializes a
global BF16 weight matrix.  The compressed weight and its scale are converted
while staging each CTA tile; BF16 warp MMA then consumes the staged tile.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import math
from pathlib import Path
from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import flashinfer
import torch
from cutlass.cute.runtime import from_dlpack

from tune_w8a16_projection import (
    CATALOG_PATH,
    DEFAULT_TENSORS,
    bench,
    check_status,
    load_bf16_weight,
    metrics,
)


GROUP_SIZE = 256


def _load_tensorop_base():
    example = (
        Path(flashinfer.__file__).resolve().parent
        / "data/cutlass/examples/python/CuTeDSL/ampere/tensorop_gemm.py"
    )
    spec = importlib.util.spec_from_file_location("glmrt_cute_tensorop_gemm", example)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load CuTe tensor-op example from {example}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TensorOpGemm


TensorOpGemm = _load_tensorop_base()


class W8A16SharedDequantGemm(TensorOpGemm):
    """Warp-MMA GEMM with K-major W8 -> BF16 conversion during CTA staging."""

    def __init__(
        self,
        tile_m: int,
        tile_n: int,
        tile_k: int,
        stages: int,
        atom_layout_mnk: Tuple[int, int, int],
        row_major_weight: bool = False,
    ):
        super().__init__(
            cutlass.BFloat16,
            cutlass.BFloat16,
            cutlass.Float32,
            atom_layout_mnk,
        )
        self.cta_tiler = (tile_m, tile_n, tile_k)
        self.num_stages = stages
        self.bM, self.bN, self.bK = self.cta_tiler
        self.num_threads = math.prod(atom_layout_mnk) * 32
        self.row_major_weight = row_major_weight
        mma_m, mma_n, mma_k = self.mma_inst_shape
        assert self.bM % (atom_layout_mnk[0] * mma_m) == 0
        assert self.bN % (atom_layout_mnk[1] * mma_n) == 0
        assert atom_layout_mnk[2] == 1
        assert self.bK % mma_k == 0
        assert self.bK <= GROUP_SIZE and GROUP_SIZE % self.bK == 0
        assert self.num_stages >= 2
        assert self.bN * self.bK % self.num_threads == 0
        assert self.num_threads % self.bN == 0
        if self.row_major_weight:
            assert self.num_threads % self.bK == 0

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mQ: cute.Tensor,
        mS: cute.Tensor,
        mC: cute.Tensor,
    ):
        self.a_major_mode = utils.LayoutEnum.from_tensor(mA)
        self.b_major_mode = utils.LayoutEnum.from_tensor(mQ)
        self.c_major_mode = utils.LayoutEnum.from_tensor(mC)

        copy_bits = 128
        sA_layout = self._make_smem_layout_AB(
            cutlass.BFloat16,
            self.a_major_mode,
            copy_bits,
            (self.bM, self.bK, self.num_stages),
        )
        # Q is int8 in global memory, but each CTA owns only a BF16 shared tile.
        sB_layout = self._make_smem_layout_AB(
            cutlass.BFloat16,
            self.b_major_mode,
            copy_bits,
            (self.bN, self.bK, self.num_stages),
        )
        sC_layout = self._make_smem_layout_C(
            cutlass.BFloat16,
            self.c_major_mode,
            copy_bits,
            (self.bM, self.bN),
        )

        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            cutlass.BFloat16,
            num_bits_per_copy=copy_bits,
        )
        tiled_copy_A = self._make_gmem_tiled_copy_AB(
            atom_async_copy,
            cutlass.BFloat16,
            self.a_major_mode,
            copy_bits,
        )
        atom_sync_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.BFloat16,
            num_bits_per_copy=copy_bits,
        )
        tiled_copy_C = self._make_gmem_tiled_copy_C(
            atom_sync_copy,
            cutlass.BFloat16,
            self.c_major_mode,
            copy_bits,
        )

        op = cute.nvgpu.warp.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            self.mma_inst_shape,
        )
        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )
        tiled_mma = cute.make_tiled_mma(
            op,
            cute.make_layout(self.atom_layout_mnk),
            permutation_mnk=permutation_mnk,
        )

        grid_dim = cute.ceil_div(mC.shape, (self.bM, self.bN, 1))
        self.kernel(
            mA,
            mQ,
            mS,
            mC,
            sA_layout,
            sB_layout,
            sC_layout,
            tiled_copy_A,
            tiled_copy_C,
            tiled_mma,
        ).launch(
            grid=(cute.size(grid_dim[0]), cute.size(grid_dim[1]), 1),
            block=[self.num_threads, 1, 1],
        )

    @cute.jit
    def _stage_b(
        self,
        mQ: cute.Tensor,
        mS: cute.Tensor,
        sB: cute.Tensor,
        n_tile: cutlass.Int32,
        k_tile: cutlass.Int32,
        pipe: cutlass.Int32,
        tid: cutlass.Int32,
    ):
        # Linearize along the contiguous axis of the resident layout.  The
        # row-major path is the final-layout candidate shared with M=1 decode.
        values_per_thread = self.bN * self.bK // self.num_threads
        # tile_k divides the 256-wide quantization group, so every value owned
        # by this thread in this stage uses the same per-output-channel scale.
        scale_group = (
            k_tile * cutlass.Int32(self.bK) // cutlass.Int32(GROUP_SIZE)
        )
        for iteration in cutlass.range_constexpr(values_per_thread):
            linear = tid + cutlass.Int32(iteration * self.num_threads)
            if cutlass.const_expr(self.row_major_weight):
                local_k = linear % cutlass.Int32(self.bK)
                local_n = linear // cutlass.Int32(self.bK)
            else:
                local_n = linear % cutlass.Int32(self.bN)
                local_k = linear // cutlass.Int32(self.bN)
            global_n = n_tile * cutlass.Int32(self.bN) + local_n
            global_k = k_tile * cutlass.Int32(self.bK) + local_k
            scale = mS[global_n, scale_group, 0].to(cutlass.Float32)
            q = mQ[global_n, global_k, 0].to(cutlass.Float32)
            sB[local_n, local_k, pipe] = (q * scale).to(cutlass.BFloat16)

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mQ: cute.Tensor,
        mS: cute.Tensor,
        mC: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sC_layout: cute.ComposedLayout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_C: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        tid = cutlass.Int32(tidx)
        tiler_coord = (bidx, bidy, None)

        gA = cute.local_tile(
            mA[None, None, 0],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(1, None, 1),
        )
        gC = cute.local_tile(
            mC[None, None, 0],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(1, 1, None),
        )
        gA = cute.make_tensor(gA.iterator.align(16), gA.layout)

        mcA = cute.make_identity_tensor(mA.layout.shape)
        cA = cute.local_tile(
            mcA[None, None, 0],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(1, None, 1),
        )

        @cute.struct
        class SharedStorageAB:
            a: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(sA_layout)],
                16,
            ]
            b: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(sB_layout)],
                16,
            ]

        @cute.struct
        class SharedStorageC:
            c: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(sC_layout)],
                16,
            ]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(
            max(SharedStorageAB.size_in_bytes(), SharedStorageC.size_in_bytes()),
            byte_alignment=16,
        )
        sA = SharedStorageAB(storage).a.get_tensor(sA_layout)
        sB = SharedStorageAB(storage).b.get_tensor(sB_layout)
        sC = SharedStorageC(storage).c.get_tensor(sC_layout)

        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_C = tiled_copy_C.get_slice(tidx)
        tAgA = thr_copy_A.partition_S(gA)
        tAsA = thr_copy_A.partition_D(sA)
        tCsC_epilogue = thr_copy_C.partition_S(sC)
        tCgC_epilogue = thr_copy_C.partition_D(gC)
        tAcA = thr_copy_A.partition_S(cA)

        tApA = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tAgA.shape[0][1],
                    cute.size(tAgA, mode=[1]),
                    cute.size(tAgA, mode=[2]),
                ),
                stride=(cute.size(tAgA, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        for rest_v in range(tApA.shape[0]):
            for row in range(tApA.shape[1]):
                tApA[rest_v, row, 0] = cute.elem_less(
                    tAcA[(0, rest_v), row, 0, 0][0], mA.shape[0]
                )

        tAsA.fill(0)
        cute.arch.sync_threads()
        num_smem_stages = cute.size(tAsA, mode=[3])
        k_tile_count = cute.size(tAgA, mode=[3])
        k_tile_index = cutlass.Int32(0)

        # Fill stages-1 buffers.  A uses cp.async; Q is converted directly into
        # the corresponding BF16 shared tile before the same stage is consumed.
        for pipe in range(num_smem_stages - 1):
            cute.copy(
                tiled_copy_A,
                tAgA[None, None, None, k_tile_index],
                tAsA[None, None, None, pipe],
                pred=tApA,
            )
            self._stage_b(mQ, mS, sB, bidy, k_tile_index, cutlass.Int32(pipe), tid)
            k_tile_index += 1
            cute.arch.cp_async_commit_group()

        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCsC = thr_mma.partition_C(sC)
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrC = tiled_mma.make_fragment_C(tCgC)
        tCrC.fill(0.0)

        atom_copy_s2r_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                self.a_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
            ),
            cutlass.BFloat16,
        )
        atom_copy_s2r_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                self.b_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
            ),
            cutlass.BFloat16,
        )
        tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma)
        tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_copy_s2r_B, tiled_mma)
        thr_copy_ldmatrix_A = tiled_copy_s2r_A.get_slice(tidx)
        thr_copy_ldmatrix_B = tiled_copy_s2r_B.get_slice(tidx)
        tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
        tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
        tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
        tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

        smem_pipe_read = 0
        smem_pipe_write = num_smem_stages - 1
        tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
        tCsB_p = tCsB_copy_view[None, None, None, smem_pipe_read]
        num_k_block = cute.size(tCrA, mode=[2])
        if num_k_block > 1:
            cute.arch.cp_async_wait_group(num_smem_stages - 2)
            cute.arch.sync_threads()
            cute.copy(
                tiled_copy_s2r_A,
                tCsA_p[None, None, 0],
                tCrA_copy_view[None, None, 0],
            )
            cute.copy(
                tiled_copy_s2r_B,
                tCsB_p[None, None, 0],
                tCrB_copy_view[None, None, 0],
            )

        for k_tile in range(k_tile_count):
            for k_block in cutlass.range(num_k_block, unroll_full=True):
                if k_block == num_k_block - 1:
                    tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
                    tCsB_p = tCsB_copy_view[None, None, None, smem_pipe_read]
                    cute.arch.cp_async_wait_group(num_smem_stages - 2)
                    cute.arch.sync_threads()

                k_block_next = (k_block + 1) % num_k_block
                cute.copy(
                    tiled_copy_s2r_A,
                    tCsA_p[None, None, k_block_next],
                    tCrA_copy_view[None, None, k_block_next],
                )
                cute.copy(
                    tiled_copy_s2r_B,
                    tCsB_p[None, None, k_block_next],
                    tCrB_copy_view[None, None, k_block_next],
                )

                if k_block == 0 and k_tile + num_smem_stages - 1 < k_tile_count:
                    cute.copy(
                        tiled_copy_A,
                        tAgA[None, None, None, k_tile_index],
                        tAsA[None, None, None, smem_pipe_write],
                        pred=tApA,
                    )
                    self._stage_b(
                        mQ,
                        mS,
                        sB,
                        bidy,
                        k_tile_index,
                        cutlass.Int32(smem_pipe_write),
                        tid,
                    )

                cute.gemm(
                    tiled_mma,
                    tCrC,
                    tCrA[None, None, k_block],
                    tCrB[None, None, k_block],
                    tCrC,
                )

                if k_block == 0:
                    k_tile_index += 1
                    cute.arch.cp_async_commit_group()
                    smem_pipe_write = smem_pipe_read
                    smem_pipe_read += 1
                    if smem_pipe_read == num_smem_stages:
                        smem_pipe_read = 0

        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        tCrD = cute.make_fragment_like(tCrC, cutlass.BFloat16)
        tCrD[None] = tCrC.load().to(cutlass.BFloat16)
        cute.autovec_copy(tCrD, tCsC)

        ceil_m, ceil_n, _ = cute.ceil_div(mC.shape, (self.bM, self.bN, 1))
        mcC = cute.make_identity_tensor(
            (
                cute.size(ceil_m) * self.bM,
                cute.size(ceil_n) * self.bN,
                1,
            )
        )
        cC = cute.local_tile(
            mcC[None, None, 0],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(1, 1, None),
        )
        tCcC = thr_copy_C.partition_S(cC)
        tCrC_epilogue = cute.make_fragment_like(tCsC_epilogue)
        cute.arch.sync_threads()
        cute.autovec_copy(tCsC_epilogue, tCrC_epilogue)

        tCpC = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tCgC_epilogue.shape[0][1],
                    cute.size(tCgC_epilogue, mode=[1]),
                    cute.size(tCgC_epilogue, mode=[2]),
                ),
                stride=(cute.size(tCgC_epilogue, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        for rest_v in range(tCpC.shape[0]):
            for row in range(tCpC.shape[1]):
                tCpC[rest_v, row, 0] = cute.elem_less(
                    tCcC[(0, rest_v), row, 0][0], mC.shape[0]
                )
        for rest_v in range(tCpC.shape[0]):
            for col in range(tCpC.shape[2]):
                cute.copy(
                    tiled_copy_C,
                    tCrC_epilogue[None, None, col],
                    tCgC_epilogue[None, None, col],
                    pred=tCpC[None, None, col],
                )


def configure_native(path: Path):
    native = ctypes.CDLL(str(path.resolve()))
    quantize = native.glmrt_cuda_quantize_bf16_w8a16_group256_async
    quantize.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    )
    quantize.restype = ctypes.c_int
    return quantize


def as_cute(tensor: torch.Tensor):
    return from_dlpack(tensor, assumed_align=16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--native-library",
        type=Path,
        default=Path("native/build-w8a16/libglmrt_native.so"),
    )
    parser.add_argument("--tensor", choices=("q-b", "o"), default="o")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument(
        "--weight-layout", choices=("k-major", "row-major"), default="k-major"
    )
    parser.add_argument("--tile-m", type=int, default=64)
    parser.add_argument("--tile-n", type=int, default=128)
    parser.add_argument("--tile-k", type=int, default=32)
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument(
        "--atom-layout",
        type=str,
        default="2,2,1",
        help="warp MMA atom layout as M,N,K",
    )
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    atom_layout = tuple(int(value) for value in args.atom_layout.split(","))
    if len(atom_layout) != 3:
        raise ValueError("--atom-layout must contain M,N,K")

    with CATALOG_PATH.open() as handle:
        catalog = json.load(handle)
    name = DEFAULT_TENSORS[0 if args.tensor == "q-b" else 1]
    weight = load_bf16_weight(catalog, name)
    output_rows, hidden = weight.shape
    if output_rows % args.tile_n != 0 or hidden % args.tile_k != 0:
        raise ValueError("N and K must be exact multiples of the selected CTA tile")

    quantize = configure_native(args.native_library)
    row_major_weight = args.weight_layout == "row-major"
    if row_major_weight:
        weight_k = torch.empty(
            (output_rows, hidden), device="cuda", dtype=torch.int8
        )
        scales = torch.empty(
            (output_rows, hidden // GROUP_SIZE),
            device="cuda",
            dtype=torch.float32,
        )
    else:
        weight_k = torch.empty(
            (hidden, output_rows), device="cuda", dtype=torch.int8
        )
        scales = torch.empty(
            (hidden // GROUP_SIZE, output_rows),
            device="cuda",
            dtype=torch.float32,
        )
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    check_status(
        quantize(
            weight.data_ptr(),
            weight_k.data_ptr(),
            scales.data_ptr(),
            hidden,
            output_rows,
            0 if row_major_weight else 1,
            stream,
        ),
        f"{args.weight_layout} W8 quantization",
    )
    torch.cuda.synchronize()

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260721 + args.rows)
    activation = torch.randn(
        (args.rows, hidden),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    output = torch.empty(
        (args.rows, output_rows), device="cuda", dtype=torch.bfloat16
    )

    mA = as_cute(activation.unsqueeze(-1))
    mQ = as_cute(
        weight_k.unsqueeze(-1) if row_major_weight else weight_k.T.unsqueeze(-1)
    )
    mS = as_cute(
        scales.unsqueeze(-1) if row_major_weight else scales.T.unsqueeze(-1)
    )
    mC = as_cute(output.unsqueeze(-1))

    kernel = W8A16SharedDequantGemm(
        args.tile_m,
        args.tile_n,
        args.tile_k,
        args.stages,
        atom_layout,  # type: ignore[arg-type]
        row_major_weight=row_major_weight,
    )
    print(
        "compile "
        f"tensor={args.tensor} rows={args.rows} shape={output_rows}x{hidden} "
        f"tile={args.tile_m}x{args.tile_n}x{args.tile_k} "
        f"stages={args.stages} atom={atom_layout} layout={args.weight_layout}"
    )
    compiled = cute.compile(kernel, mA, mQ, mS, mC)
    compiled(mA, mQ, mS, mC)
    torch.cuda.synchronize()

    reference = torch.mm(activation, weight.T)
    quality = metrics(output, reference)
    print(
        "quality "
        f"relative_l2={quality['relative_l2']:.9f} "
        f"cosine={quality['cosine']:.9f} "
        f"max_abs={quality['max_abs']:.6f}"
    )

    timing = bench(
        lambda _: compiled(mA, mQ, mS, mC),
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    bf16_timing = bench(
        lambda _: torch.mm(activation, weight.T, out=output),
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    print(
        "timing "
        f"kernel=cute-shared-dequant median_ms={timing.median_ms:.6f} "
        f"range_ms={timing.minimum_ms:.6f}-{timing.maximum_ms:.6f}"
    )
    print(
        "timing "
        f"kernel=bf16-cublas median_ms={bf16_timing.median_ms:.6f} "
        f"range_ms={bf16_timing.minimum_ms:.6f}-{bf16_timing.maximum_ms:.6f}"
    )


if __name__ == "__main__":
    main()
