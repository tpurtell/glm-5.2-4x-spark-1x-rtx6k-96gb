# Third-Party Notices

## XGrammar

GLMRT statically links a pinned XGrammar release for JSON Schema and strict
tool-call constrained decoding: <https://github.com/mlc-ai/xgrammar>. The
exact revision, nested DLPack revision, and source-tree digest are recorded in
`third_party/xgrammar.lock.json` and distributed as
`XGRAMMAR_PROVENANCE.json`.

Copyright (c) the XGrammar authors and other per-file copyright holders.

SPDX-License-Identifier: Apache-2.0

XGrammar is distributed under the Apache License, Version 2.0. The complete
root license text is distributed as `XGRAMMAR_LICENSE` in standalone
artifacts and as `/opt/glmrt/share/licenses/xgrammar/LICENSE` in inference
images. XGrammar's vendored DLPack headers retain their own Apache-2.0
copyright and license notices in the pinned source tree.

## NVIDIA / FlashInfer SM120 sparse MLA decode

`native/cuda/kernels/packed_fp8_mla_exact.cu` is adapted from the NVIDIA
SM120 sparse MLA decode implementation distributed with FlashInfer 0.6.14,
including `decode_dsv3_2_kernel.cuh` and `decode_dsv4_kernel.cuh`.
The upstream project is <https://github.com/flashinfer-ai/flashinfer>.

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

SPDX-License-Identifier: BSD-3-Clause

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

## SparkInfer

GLMRT builds CuTe kernels from a pinned fork of SparkInfer, formerly `b12x`:
<https://github.com/tpurtell/sparkinfer-glmrt>. The pinned fork revision and
source digest are recorded in `third_party/sparkinfer.lock.json`.
`SPARKINFER_PROVENANCE.json` carries those exact locked values plus hashes of
the distributed license and notices. `SPARKINFER_SHA256SUMS` provides a
directly verifiable checksum list for all three SparkInfer release records.

Copyright (c) 2025 by the SparkInfer authors and other per-file copyright
holders.

SPDX-License-Identifier: Apache-2.0

The SparkInfer project is distributed under the Apache License, Version 2.0,
subject to the per-file notices retained in its source. You may obtain the
Apache License at <https://www.apache.org/licenses/LICENSE-2.0>. The complete
root license text is distributed as `SPARKINFER_LICENSE` in standalone
artifacts and as `/opt/glmrt/share/licenses/sparkinfer/LICENSE` in inference
images.

### NVIDIA dense GEMM component in SparkInfer

SparkInfer's `b12x/_lib/dense_gemm.py`, which GLMRT uses to generate
Spark-side AOT kernels, is adapted from an NVIDIA CUTLASS dense block-scaled
GEMM example and carries this notice:

Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

SPDX-License-Identifier: BSD-3-Clause

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

### FlashAttention-derived contiguous attention component in SparkInfer

SparkInfer's
`b12x/attention/_shared/contiguous/forward.py` is adapted from the
FlashAttention CuTe forward implementation and carries this notice:

Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
Pradeep Ramani, Tri Dao. All rights reserved.

SPDX-License-Identifier: BSD-3-Clause

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
