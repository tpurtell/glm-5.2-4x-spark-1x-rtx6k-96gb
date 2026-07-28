#!/usr/bin/env bash
set -euo pipefail

role="${1:-coordinator}"
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ "${1:-}" == "--" ]]; then
  shift
fi

case "$role" in
  coordinator|oliver)
    image="glmrt-dev:oliver"
    ;;
  expert|spark)
    image="glmrt-dev:spark"
    ;;
  dev)
    image="glmrt-dev:oliver"
    ;;
  *)
    echo "unknown role: $role" >&2
    exit 2
    ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"

docker_args=(
  run --rm
  --gpus all
  -v "$repo_root:/workspace/glmrt"
  -v "$hf_home:$hf_home:ro"
  -v "$hf_home:/root/.cache/huggingface:ro"
  -e HF_HOME="$hf_home"
  -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  -e TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  -e GLMRT_MODEL_ID="${GLMRT_MODEL_ID:-lukealonso/GLM-5.2-NVFP4}"
)

if [[ "$repo_root" != "/workspace/glmrt" ]]; then
  docker_args+=(-v "$repo_root:$repo_root")
fi

if [[ -t 0 && -t 1 ]]; then
  docker_args+=(-it)
fi

if [[ "$role" == "expert" || "$role" == "spark" ]]; then
  docker_args+=(--net=host --ipc=host --ulimit memlock=-1:-1 --cap-add IPC_LOCK)
  if [[ -e /dev/infiniband ]]; then
    docker_args+=(--device=/dev/infiniband)
  fi
fi

if [[ "$role" == "coordinator" || "$role" == "oliver" ]]; then
  if [[ "${GLMRT_DOCKER_HOST_NETWORK:-0}" == "1" ]]; then
    docker_args+=(--net=host --ipc=host --ulimit memlock=-1:-1 --cap-add IPC_LOCK)
    if [[ -e /dev/infiniband ]]; then
      docker_args+=(--device=/dev/infiniband)
    fi
  fi
fi

if [[ $# -eq 0 ]]; then
  set -- bash
fi

exec docker "${docker_args[@]}" "$image" "$@"
