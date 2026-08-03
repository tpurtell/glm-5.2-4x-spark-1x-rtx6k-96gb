#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: build-wip-artifacts.sh SOURCE_DIR ROLE CUDA_ARCH BUILD_DIR OUTPUT_DIR" >&2
  exit 2
}

[[ $# -eq 5 ]] || usage
source_dir="$(realpath "$1")"
role="$2"
cuda_arch="$3"
build_dir="$(realpath -m "$4")"
output_dir="$(realpath -m "$5")"

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

python3 "$source_dir/scripts/verify-sparkinfer-source.py" \
  --source "$source_dir/third_party/sparkinfer" \
  --lock "$source_dir/third_party/sparkinfer.lock.json"

mkdir -p "$build_dir" "$output_dir"
export PYO3_PYTHON=python3
export PYTHONPATH="$source_dir/third_party/sparkinfer:$source_dir/python/reference/glmrt_reference:$source_dir/python/reference${PYTHONPATH:+:$PYTHONPATH}"
export CARGO_TARGET_DIR="$build_dir/cargo-target"

cargo build \
  --quiet \
  --manifest-path "$source_dir/rust/Cargo.toml" \
  -p glmrt-daemon \
  --release

cmake \
  -S "$source_dir/native" \
  -B "$build_dir/native" \
  -G Ninja \
  -DGLMRT_ENABLE_CUDA=ON \
  -DGLMRT_ENABLE_RDMA=ON \
  -DGLMRT_ENABLE_SPARKINFER_AOT="$sparkinfer_aot" \
  -DGLMRT_ENABLE_SPARKINFER_COORDINATOR_AOT="$coordinator_aot" \
  -DGLMRT_ENABLE_W8A16_AOT="$w8a16_aot" \
  -DGLMRT_SPARKINFER_SOURCE_DIR="$source_dir/third_party/sparkinfer" \
  -DGLMRT_SPARKINFER_LOCK_FILE="$source_dir/third_party/sparkinfer.lock.json" \
  -DGLMRT_ENABLE_NCCL="$nccl" \
  -DPython3_EXECUTABLE="$(command -v python3)" \
  -DGLMRT_CUDA_ARCHITECTURES="$cuda_arch"
cmake --build "$build_dir/native"

install -m 0755 "$CARGO_TARGET_DIR/release/glmrt" "$output_dir/glmrt"
install -m 0755 "$build_dir/native/libglmrt_native.so" "$output_dir/libglmrt_native.so"
(
  cd "$output_dir"
  sha256sum glmrt libglmrt_native.so >ARTIFACT_SHA256SUMS
  sha256sum -c ARTIFACT_SHA256SUMS
)
