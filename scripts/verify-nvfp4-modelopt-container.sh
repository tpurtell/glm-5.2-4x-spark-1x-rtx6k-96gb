#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${GLMRT_NVFP4_MODEL_OPT_IMAGE:-nvcr.io/nvidia/pytorch:26.05-py3}"
fixture="${GLMRT_NVFP4_MODEL_OPT_FIXTURE:-tests/fixtures/nvfp4/real_tensor_decode.json}"
output="${GLMRT_NVFP4_MODEL_OPT_OUTPUT:-tests/fixtures/nvfp4/modelopt_reference.json}"
gpus="${GLMRT_NVFP4_MODEL_OPT_DOCKER_GPUS:-all}"
device="${GLMRT_NVFP4_MODEL_OPT_DEVICE:-cuda}"

docker_args=(
  run --rm -i
  --ipc=host
  --ulimit memlock=-1
  --ulimit stack=67108864
  -v "$repo_root:/workspace/glmrt"
  -w /workspace/glmrt
  -e "GLMRT_NVFP4_MODEL_OPT_IMAGE=$image"
  -e "GLMRT_NVFP4_MODEL_OPT_DEVICE=$device"
  --entrypoint python
)
if [ -n "$gpus" ] && [ "$gpus" != "none" ]; then
  docker_args+=(--gpus "$gpus")
fi

docker "${docker_args[@]}" "$image" \
  python/tools/verify_nvfp4_modelopt_real_tensor_decode_fixture.py \
    --fixture "$fixture" \
    --output "$output" \
    --device "$device"
