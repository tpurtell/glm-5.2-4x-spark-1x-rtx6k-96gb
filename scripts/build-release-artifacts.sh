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
    b12x_aot=OFF
    coordinator_aot=ON
    w8a16_aot=ON
    ;;
  expert)
    b12x_aot=ON
    coordinator_aot=OFF
    w8a16_aot=OFF
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

build_root="$(mktemp -d /tmp/glmrt-release-build.XXXXXX)"
trap 'rm -rf "$build_root"' EXIT
mkdir -p "$build_root/source"
tar \
  -C "$source_dir" \
  --exclude=.git \
  --exclude=.venv \
  --exclude=.glmrt-release \
  --exclude=.glmrt-release-image \
  --exclude=dist \
  --exclude=rust/target \
  --exclude='native/build*' \
  -cf - . |
  tar -C "$build_root/source" -xf -

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
  -DGLMRT_ENABLE_B12X_AOT="$b12x_aot" \
  -DGLMRT_ENABLE_B12X_COORDINATOR_AOT="$coordinator_aot" \
  -DGLMRT_ENABLE_W8A16_AOT="$w8a16_aot" \
  -DGLMRT_ENABLE_SPARKINFER_SOURCE_W4A16_AOT=OFF \
  -DGLMRT_ENABLE_NCCL=OFF \
  -DPython3_EXECUTABLE="$(command -v python3)" \
  -DGLMRT_CUDA_ARCHITECTURES="$cuda_arch"
cmake --build "$build_root/native"

install -d "$output_dir"
install -m 0755 "$build_root/source/rust/target/release/glmrt" "$output_dir/glmrt"
install -m 0755 "$build_root/native/libglmrt_native.so" "$output_dir/libglmrt_native.so"
install -m 0644 \
  "$build_root/source/THIRD_PARTY_NOTICES.md" \
  "$output_dir/THIRD_PARTY_NOTICES.md"
test -x "$output_dir/glmrt"
test -s "$output_dir/libglmrt_native.so"
test -s "$output_dir/THIRD_PARTY_NOTICES.md"
