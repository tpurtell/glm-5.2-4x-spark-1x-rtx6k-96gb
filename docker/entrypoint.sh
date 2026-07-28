#!/usr/bin/env bash
set -euo pipefail

if [[ -d /workspace/glmrt ]]; then
  export PATH="/workspace/glmrt/scripts:$PATH"
  cd /workspace/glmrt
fi
exec "$@"
