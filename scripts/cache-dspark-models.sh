#!/usr/bin/env bash
set -euo pipefail

role="${1:-coordinator}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$role" in
  coordinator|oliver)
    image="glmrt-dev:oliver"
    ;;
  expert|spark)
    image="glmrt-dev:spark"
    ;;
  *)
    echo "unknown role: $role" >&2
    exit 2
    ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$hf_home"

docker run --rm \
  -v "$repo_root:/workspace/glmrt:ro" \
  -v "$hf_home:/root/.cache/huggingface" \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=0 \
  -e TRANSFORMERS_OFFLINE=0 \
  -e HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}" \
  -e HF_HUB_DISABLE_TELEMETRY=1 \
  -e HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}" \
  "$image" \
  bash -lc 'uv run --no-project python /workspace/glmrt/python/tools/cache_dspark_models.py "$@"' \
  bash "$@"
