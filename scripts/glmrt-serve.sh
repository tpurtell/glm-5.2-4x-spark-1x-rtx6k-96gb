#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${GLMRT_PYTHON:-$repo_root/.venv/bin/python}"
if [ ! -x "$python" ]; then
  python="$(command -v python3)"
fi
export PYTHONPATH="$repo_root/python/reference${PYTHONPATH:+:$PYTHONPATH}"
exec "$python" "$repo_root/python/tools/resolve_serve_profile.py" "$@"
