---
base_model: zai-org/GLM-5.2
language:
- en
- zh
library_name: transformers
license: mit
pipeline_tag: text-generation
tags:
- glm
- mixture-of-experts
- exl3
- 3-bit
---

# GLM-5.2 EXL3 K3 calibrated v1

This repository contains a calibrated EXL3 K3 variant of
[`zai-org/GLM-5.2`](https://huggingface.co/zai-org/GLM-5.2) for GLMRT. The
three dense decoder layers, attention, routers, shared experts, embeddings,
head, and native MTP layer retain their source tensors. Only the 256 routed
experts in base decoder layers 3 through 77 are replaced by EXL3 K3 MCG
`trellis/suh/svh/mcg` tensors.

The artifact uses a single resident expert representation on each of four
Spark TP ranks. Every rank retains its 512-wide intermediate slice of every
expert; the runtime does not keep native and EXL3 copies resident together.

## Quantization

- Source: `zai-org/GLM-5.2`, revision `b4734de4facf877f85769a911abafc5283eab3d9`
- Format: EXL3, 3 physical trellis bits per routed-expert weight
- Codebook: MCG
- Output-scale search: automatic, folded into the stored rotations
- Calibration: 1,080,625 GLM-5.2 tokenizer tokens spanning general text,
  English and Chinese, code/agentic work, mathematics/reasoning termination,
  and structured output
- Sparse coverage: natural top-8 routes, deterministic adjacent-router
  recovery for deficient experts, then an explicitly recorded isotropic
  Hessian residual only for any remaining deficit
- Quantizer: the content-pinned GLMRT GPTQModel fork

The calibration, held-out, and screening splits are source-disjoint. The
complete immutable plan, projection evidence, retained-native proof, and
artifact manifest are maintained by the GLMRT project rather than shipped as
large private recovery files in this standard model repository.

## Runtime

The artifact is intended for GLMRT's generated SparkInfer SM121 EXL3 K3 TP4
path. Generic Transformers metadata is included, but compatibility with other
EXL3 runtimes has not been claimed.

## Qualification

GLMRT_PUBLICATION_RESULTS_PENDING

Before publication this marker is replaced mechanically from the signed final
structural, quantizer, and serving reports. The publication builder rejects a
card that retains the marker or does not cite the exact serving-report hash.

## License and attribution

This derivative follows the source model's MIT license. See `LICENSE` and the
original [`GLM-5.2` model card](https://huggingface.co/zai-org/GLM-5.2) for
upstream architecture, intended-use, and limitation details.
