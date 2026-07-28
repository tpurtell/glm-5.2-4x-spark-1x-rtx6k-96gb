#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED_COMMIT="557becfb64c503ae9c04344b0047661f43f44320"
SOURCE_DIR="${GLMRT_XGRAMMAR_SOURCE_DIR:-}"
BUILD_DIR="${GLMRT_XGRAMMAR_BUILD_DIR:-/tmp/glmrt-xgrammar-v0.2.3-build}"
FIXTURE="$ROOT/native/tools/fixtures/glm_required_tool_calls.json"
BENCH_SOURCE="$ROOT/native/tools/xgrammar_glm_bench.cc"
BENCH_BINARY="$BUILD_DIR/xgrammar_glm_bench"

if [[ -z "$SOURCE_DIR" ]]; then
  echo "GLMRT_XGRAMMAR_SOURCE_DIR must point to XGrammar v0.2.3" >&2
  exit 2
fi
if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  echo "XGrammar source is not a git checkout: $SOURCE_DIR" >&2
  exit 2
fi

ACTUAL_COMMIT="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  echo "expected XGrammar $EXPECTED_COMMIT, found $ACTUAL_COMMIT" >&2
  exit 2
fi
if [[ ! -f "$SOURCE_DIR/3rdparty/dlpack/include/dlpack/dlpack.h" ]]; then
  echo "XGrammar dlpack submodule is missing; initialize 3rdparty/dlpack" >&2
  exit 2
fi

mkdir -p "$BUILD_DIR"
printf '%s\n' \
  'set(XGRAMMAR_BUILD_PYTHON_BINDINGS OFF)' \
  'set(XGRAMMAR_BUILD_CXX_TESTS OFF)' \
  >"$BUILD_DIR/config.cmake"
cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$BUILD_DIR" --target xgrammar -j "${GLMRT_BUILD_JOBS:-8}" >/dev/null

"${CXX:-c++}" \
  -O3 -DNDEBUG -std=c++17 -flto=auto -Wall -Wextra -Werror \
  -I"$SOURCE_DIR/include" \
  -I"$SOURCE_DIR/3rdparty/picojson" \
  -I"$SOURCE_DIR/3rdparty/dlpack/include" \
  "$BENCH_SOURCE" "$BUILD_DIR/libxgrammar.a" \
  -pthread -o "$BENCH_BINARY"

exec "$BENCH_BINARY" \
  --fixture "$FIXTURE" \
  --xgrammar-commit "$ACTUAL_COMMIT" \
  "$@"
