#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: build-release-artifacts.sh SOURCE_DIR ROLE CUDA_ARCH OUTPUT_DIR" >&2
  exit 2
}

[[ $# -eq 4 ]] || usage
source_dir="$(realpath "$1")"
role="$2"
cuda_arch="$3"
output_dir="$(realpath -m "$4")"

case "$role" in
  coordinator)
    sparkinfer_aot=OFF
    coordinator_aot=ON
    w8a16_aot=ON
    nccl=OFF
    ;;
  expert)
    sparkinfer_aot=ON
    coordinator_aot=OFF
    w8a16_aot=OFF
    nccl=ON
    ;;
  *)
    echo "ROLE must be coordinator or expert" >&2
    exit 2
    ;;
esac
[[ "$cuda_arch" =~ ^[0-9]+$ ]] || {
  echo "CUDA_ARCH must be numeric" >&2
  exit 2
}
[[ -f "$source_dir/rust/Cargo.toml" && -f "$source_dir/native/CMakeLists.txt" ]] || {
  echo "SOURCE_DIR is not a GLMRT source tree: $source_dir" >&2
  exit 2
}
[[ -f "$source_dir/THIRD_PARTY_NOTICES.md" ]] || {
  echo "SOURCE_DIR is missing THIRD_PARTY_NOTICES.md" >&2
  exit 2
}
python3 "$source_dir/scripts/verify-sparkinfer-source.py" \
  --source "$source_dir/third_party/sparkinfer" \
  --lock "$source_dir/third_party/sparkinfer.lock.json"

build_root="$(mktemp -d /tmp/glmrt-release-build.XXXXXX)"
trap 'rm -rf "$build_root"' EXIT
mkdir -p "$build_root/source"
tar \
  -C "$source_dir" \
  --exclude=.git \
  --exclude='*/.git' \
  --exclude=.venv \
  --exclude=.mypy_cache \
  --exclude='*/.mypy_cache' \
  --exclude=.pytest_cache \
  --exclude='*/.pytest_cache' \
  --exclude=.ruff_cache \
  --exclude='*/.ruff_cache' \
  --exclude=__pycache__ \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude=.glmrt-cache \
  --exclude=.glmrt-release \
  --exclude=.glmrt-release-image \
  --exclude=.glmrt-wip \
  --exclude=dist \
  --exclude=rust/target \
  --exclude='native/build*' \
  -cf - . |
  tar -C "$build_root/source" -xf -

python3 "$build_root/source/scripts/verify-sparkinfer-source.py" \
  --source "$build_root/source/third_party/sparkinfer" \
  --lock "$build_root/source/third_party/sparkinfer.lock.json" \
  --require-no-python-cache

export PYO3_PYTHON=python3
cargo build \
  --manifest-path "$build_root/source/rust/Cargo.toml" \
  -p glmrt-daemon \
  --release

cmake \
  -S "$build_root/source/native" \
  -B "$build_root/native" \
  -G Ninja \
  -DGLMRT_ENABLE_CUDA=ON \
  -DGLMRT_ENABLE_RDMA=ON \
  -DGLMRT_ENABLE_SPARKINFER_AOT="$sparkinfer_aot" \
  -DGLMRT_ENABLE_SPARKINFER_COORDINATOR_AOT="$coordinator_aot" \
  -DGLMRT_ENABLE_W8A16_AOT="$w8a16_aot" \
  -DGLMRT_SPARKINFER_SOURCE_DIR="$build_root/source/third_party/sparkinfer" \
  -DGLMRT_SPARKINFER_LOCK_FILE="$build_root/source/third_party/sparkinfer.lock.json" \
  -DGLMRT_ENABLE_NCCL="$nccl" \
  -DPython3_EXECUTABLE="$(command -v python3)" \
  -DGLMRT_CUDA_ARCHITECTURES="$cuda_arch"
cmake --build "$build_root/native"

install -d "$output_dir"
install -m 0755 "$build_root/source/rust/target/release/glmrt" "$output_dir/glmrt"
install -m 0755 "$build_root/native/libglmrt_native.so" "$output_dir/libglmrt_native.so"
install -m 0644 \
  "$build_root/source/THIRD_PARTY_NOTICES.md" \
  "$output_dir/THIRD_PARTY_NOTICES.md"
install -m 0644 \
  "$build_root/source/third_party/sparkinfer/LICENSE" \
  "$output_dir/SPARKINFER_LICENSE"
python3 "$build_root/source/scripts/sparkinfer-release-provenance.py" \
  --source "$build_root/source/third_party/sparkinfer" \
  --lock "$build_root/source/third_party/sparkinfer.lock.json" \
  --license "$output_dir/SPARKINFER_LICENSE" \
  --notices "$output_dir/THIRD_PARTY_NOTICES.md" \
  --write "$output_dir/SPARKINFER_PROVENANCE.json"
(
  cd "$output_dir"
  sha256sum \
    THIRD_PARTY_NOTICES.md \
    SPARKINFER_PROVENANCE.json \
    SPARKINFER_LICENSE >SPARKINFER_SHA256SUMS
  sha256sum -c SPARKINFER_SHA256SUMS
)
test -x "$output_dir/glmrt"
test -s "$output_dir/libglmrt_native.so"
test -s "$output_dir/THIRD_PARTY_NOTICES.md"
test -s "$output_dir/SPARKINFER_PROVENANCE.json"
test -s "$output_dir/SPARKINFER_LICENSE"
test -s "$output_dir/SPARKINFER_SHA256SUMS"
