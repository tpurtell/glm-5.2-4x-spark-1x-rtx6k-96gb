#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$repo_root/scripts/release-common.sh"

usage() {
  cat <<'EOF'
Usage: ./build.sh [--profile FILE]

Builds the coordinator image locally and the Spark image natively over SSH on
the first configured Spark. It exports both release artifact sets to dist/
and distributes the Spark inference image to all configured Spark hosts.

Exact dirty-tree releases must set GLMRT_RELEASE_SOURCE_MANIFEST to a
SOURCE_SHA256SUMS generated from a frozen source tree. Builds from a source
snapshot without .git must also set GLMRT_RELEASE_ENGINE_REVISION to the exact
40-hex revision, or REVISION-dirty-MANIFEST12 for a dirty source snapshot.
EOF
}

config="$repo_root/glmrt.config"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile|--config)
      config="${2:?$1 requires a configuration file}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      release_die "unknown build argument: $1"
      ;;
  esac
done

release_load_config "$config"
release_need docker
release_need ssh
release_need rsync
release_need sha256sum
release_need python3
release_need install

prepare_pinned_source_dependencies() {
  local git_root=""
  if command -v git >/dev/null 2>&1; then
    git_root="$(git -C "$repo_root" rev-parse --show-toplevel 2>/dev/null || true)"
  fi

  if [[ -n "$git_root" &&
    "$(cd "$git_root" && pwd -P)" == "$(cd "$repo_root" && pwd -P)" ]]; then
    echo "== preparing pinned source dependencies =="
    git -C "$repo_root" submodule sync -- \
      third_party/sparkinfer third_party/xgrammar third_party/gptqmodel
    git -C "$repo_root" submodule update --init --checkout -- \
      third_party/sparkinfer third_party/xgrammar third_party/gptqmodel
    git -C "$repo_root/third_party/xgrammar" submodule sync -- \
      3rdparty/dlpack
    git -C "$repo_root/third_party/xgrammar" submodule update --init --checkout -- \
      3rdparty/dlpack
  fi

  [[ -f "$repo_root/third_party/sparkinfer/b12x/__init__.py" ]] ||
    release_die "SparkInfer source is missing; initialize third_party/sparkinfer"
  [[ -f "$repo_root/third_party/xgrammar/include/xgrammar/compiler.h" ]] ||
    release_die "XGrammar source is missing; initialize third_party/xgrammar"
  [[ -f "$repo_root/third_party/xgrammar/3rdparty/dlpack/include/dlpack/dlpack.h" ]] ||
    release_die "XGrammar DLPack source is missing; initialize third_party/xgrammar/3rdparty/dlpack"
  [[ -f "$repo_root/third_party/gptqmodel/gptqmodel/models/definitions/glm_moe_dsa.py" ]] ||
    release_die "GPTQModel source is missing; initialize third_party/gptqmodel"
}

prepare_pinned_source_dependencies

docker info >/dev/null 2>&1 || release_die "local Docker daemon is unavailable"
source_manifest="${GLMRT_RELEASE_SOURCE_MANIFEST:-}"
source_manifest_sha256=""
if [[ -n "$source_manifest" ]]; then
  source_manifest="$(
    python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' \
      "$source_manifest"
  )"
  [[ -f "$source_manifest" ]] ||
    release_die "release source manifest not found: $source_manifest"
  source_manifest_sha256="$(sha256sum "$source_manifest" | awk '{print $1}')"
fi

verify_local_source_manifest() {
  [[ -n "$source_manifest" ]] || return 0
  local current_source_manifest_sha256
  current_source_manifest_sha256="$(
    sha256sum "$source_manifest" | awk '{print $1}'
  )"
  [[ "$current_source_manifest_sha256" == "$source_manifest_sha256" ]] ||
    release_die "release source manifest changed during the build"
  python3 "$repo_root/scripts/verify-release-source-manifest.py" \
    --source "$repo_root" \
    --manifest "$source_manifest" ||
    release_die "release source differs from $source_manifest"
}

verify_remote_source_manifest() {
  [[ -n "$source_manifest" ]] || return 0
  local current_source_manifest_sha256
  current_source_manifest_sha256="$(
    sha256sum "$source_manifest" | awk '{print $1}'
  )"
  [[ "$current_source_manifest_sha256" == "$source_manifest_sha256" ]] ||
    release_die "release source manifest changed during the build"
  local remote_dir_quoted
  printf -v remote_dir_quoted '%q' "$remote_dir"
  ssh -o BatchMode=yes "$seed_host" \
    "cd $remote_dir_quoted && python3 scripts/verify-release-source-manifest.py --source . --manifest -" \
    <"$source_manifest" ||
    release_die "$seed_host staged source differs from $source_manifest"
}

verify_local_source_manifest
sparkinfer_commit="$(
  python3 "$repo_root/scripts/verify-sparkinfer-source.py" \
    --source "$repo_root/third_party/sparkinfer" \
    --lock "$repo_root/third_party/sparkinfer.lock.json" \
    --print-revision
)"
python3 "$repo_root/scripts/verify-xgrammar-source.py" \
  --source "$repo_root/third_party/xgrammar" \
  --lock "$repo_root/third_party/xgrammar.lock.json"
python3 "$repo_root/scripts/verify-gptqmodel-source.py" \
  --source "$repo_root/third_party/gptqmodel" \
  --lock "$repo_root/third_party/gptqmodel.lock.json"
detected_engine_commit="$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || true)"
engine_revision_override="${GLMRT_RELEASE_ENGINE_REVISION:-}"
engine_source_dirty=0
if [[ -n "$detected_engine_commit" &&
  -n "$(git -C "$repo_root" status --porcelain 2>/dev/null || true)" ]]; then
  engine_source_dirty=1
fi
if [[ -n "$engine_revision_override" ]]; then
  [[ "$engine_revision_override" =~ ^[0-9a-f]{40}(-dirty-[0-9a-f]{12})?$ ]] ||
    release_die "GLMRT_RELEASE_ENGINE_REVISION must be REVISION or REVISION-dirty-MANIFEST12"
  [[ -n "$source_manifest_sha256" ]] ||
    release_die "GLMRT_RELEASE_ENGINE_REVISION requires GLMRT_RELEASE_SOURCE_MANIFEST"
  engine_commit="$engine_revision_override"
elif [[ -z "$detected_engine_commit" ]]; then
  release_die "source snapshot has no Git metadata; set GLMRT_RELEASE_ENGINE_REVISION and GLMRT_RELEASE_SOURCE_MANIFEST"
elif ((engine_source_dirty)); then
  [[ -n "$source_manifest_sha256" ]] ||
    release_die "dirty release source requires GLMRT_RELEASE_SOURCE_MANIFEST"
  engine_commit="${detected_engine_commit}-dirty-${source_manifest_sha256:0:12}"
else
  engine_commit="$detected_engine_commit"
fi
if [[ "$engine_commit" == *-dirty-* ]]; then
  [[ -n "$source_manifest_sha256" ]] ||
    release_die "dirty engine revision requires GLMRT_RELEASE_SOURCE_MANIFEST"
  [[ "$engine_commit" == *"-dirty-${source_manifest_sha256:0:12}" ]] ||
    release_die "dirty engine revision suffix does not match the source manifest"
fi
release_source_label_args=()
if [[ -n "$source_manifest_sha256" ]]; then
  release_source_label_args+=(
    --label "io.glmrt.source-manifest.sha256=$source_manifest_sha256"
  )
fi

hosts_csv="$(release_hosts_csv)"
seed_host="$SPARK_0_HOST"
remote_dir="${GLMRT_RELEASE_REMOTE_BUILD_DIR:-}"
artifact_dir="$repo_root/.glmrt-release-image"

echo "== validating native Spark build hosts =="
for host in "$SPARK_0_HOST" "$SPARK_1_HOST" "$SPARK_2_HOST" "$SPARK_3_HOST"; do
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" bash -s <<'REMOTE'
set -euo pipefail
command -v docker >/dev/null
docker info >/dev/null
test "$(uname -m)" = "aarch64"
REMOTE
  echo "  $host: ssh/docker/aarch64 ready"
done

release_sync_program=rsync
if command -v rdmasync >/dev/null 2>&1 &&
  ssh -o BatchMode=yes "$seed_host" "command -v rdmasync >/dev/null 2>&1"; then
  release_sync_program=rdmasync
  echo "== using RDMA source/artifact synchronization =="
fi

release_sync() {
  if [[ "$release_sync_program" == rdmasync ]]; then
    rdmasync -a --rdma=required --rdma-show-config "$@"
  else
    rsync -a "$@"
  fi
}

if [[ -z "$remote_dir" ]]; then
  remote_dir="$(
    ssh -o BatchMode=yes "$seed_host" \
      'printf "%s/glmrt-release-build" "$HOME"'
  )"
fi

local_free_kib="$(df -Pk "$repo_root" | awk 'NR==2 {print $4}')"
((local_free_kib >= 60 * 1024 * 1024)) || release_die "local build needs at least 60 GiB free"
remote_free_kib="$(
  ssh -o BatchMode=yes "$seed_host" bash -s <<'REMOTE'
df -Pk "$HOME" | awk 'NR == 2 { print $4 }'
REMOTE
)"
((remote_free_kib >= 60 * 1024 * 1024)) || release_die "$seed_host build needs at least 60 GiB free"

echo "== building coordinator development image: $COORDINATOR_DOCKER_DEV =="
docker build \
  --build-arg GLMRT_ROLE=coordinator \
  --build-arg CUDA_ARCH=120 \
  --build-arg TARGET_PLATFORM=linux/amd64 \
  --build-arg GLMRT_SPARKINFER_COMMIT="$sparkinfer_commit" \
  -f "$repo_root/docker/Dockerfile.dev" \
  -t "$COORDINATOR_DOCKER_DEV" \
  "$repo_root"

echo "== compiling coordinator release artifacts in GPU-enabled development container =="
mkdir -p "$artifact_dir"
docker run --rm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1:-1 \
  -v "$repo_root:/source:ro" \
  -v "$artifact_dir:/output" \
  "$COORDINATOR_DOCKER_DEV" \
  /source/scripts/build-release-artifacts.sh /source coordinator 120 /output

echo "== building coordinator inference image: $COORDINATOR_DOCKER_INFERENCE =="
docker build \
  "${release_source_label_args[@]}" \
  --build-arg GLMRT_ROLE=coordinator \
  --build-arg CUDA_ARCH=120 \
  --build-arg GLMRT_ENGINE_COMMIT="$engine_commit" \
  --build-arg GLMRT_SPARKINFER_COMMIT="$sparkinfer_commit" \
  -f "$repo_root/docker/Dockerfile.release" \
  -t "$COORDINATOR_DOCKER_INFERENCE" \
  "$repo_root"

echo "== staging native Spark build on $seed_host:$remote_dir =="
ssh -o BatchMode=yes "$seed_host" "mkdir -p '$remote_dir'"
release_sync --delete \
  --exclude '.git' \
  --exclude '.venv/' \
  --exclude '.mypy_cache/' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.pyo' \
  --exclude '.glmrt-cache/' \
  --exclude '.glmrt-release/' \
  --exclude '.glmrt-release-image/' \
  --exclude '.glmrt-wip/' \
  --exclude 'dist/' \
  --exclude 'rust/target/' \
  --exclude 'native/build*/' \
  "$repo_root/" "$seed_host:$remote_dir/"
# The broad staging sync protects excluded paths from deletion. Reconcile the
# pinned source separately so bytecode left by an earlier build cannot survive
# merely because it is now excluded.
release_sync --delete --delete-excluded \
  --exclude '.git' \
  --exclude '.venv/' \
  --exclude '.mypy_cache/' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.pyo' \
  "$repo_root/third_party/sparkinfer/" \
  "$seed_host:$remote_dir/third_party/sparkinfer/"
release_sync --delete --delete-excluded \
  --exclude '.git' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.pyo' \
  "$repo_root/third_party/xgrammar/" \
  "$seed_host:$remote_dir/third_party/xgrammar/"
verify_remote_source_manifest

echo "== building Spark development and inference images natively on $seed_host =="
ssh -o BatchMode=yes "$seed_host" bash -s -- \
  "$remote_dir" "$SPARK_EXPERT_DOCKER_DEV" "$SPARK_EXPERT_DOCKER_INFERENCE" \
  "$engine_commit" "$sparkinfer_commit" "$source_manifest_sha256" <<'REMOTE'
set -euo pipefail
remote_dir="$1"
dev_image="$2"
inference_image="$3"
engine_commit="$4"
sparkinfer_commit="$5"
source_manifest_sha256="${6-}"
release_source_label_args=()
if [[ -n "$source_manifest_sha256" ]]; then
  release_source_label_args+=(
    --label "io.glmrt.source-manifest.sha256=$source_manifest_sha256"
  )
fi
cd "$remote_dir"
python3 scripts/verify-sparkinfer-source.py \
  --source third_party/sparkinfer \
  --lock third_party/sparkinfer.lock.json \
  --require-no-python-cache
docker build \
  --build-arg GLMRT_ROLE=expert \
  --build-arg CUDA_ARCH=121 \
  --build-arg TARGET_PLATFORM=linux/arm64 \
  --build-arg GLMRT_SPARKINFER_COMMIT="$sparkinfer_commit" \
  -f docker/Dockerfile.dev \
  -t "$dev_image" .
mkdir -p .glmrt-release-image
docker run --rm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1:-1 \
  -v "$remote_dir:/source:ro" \
  -v "$remote_dir/.glmrt-release-image:/output" \
  "$dev_image" \
  /source/scripts/build-release-artifacts.sh /source expert 121 /output
docker build \
  "${release_source_label_args[@]}" \
  --build-arg GLMRT_ROLE=expert \
  --build-arg CUDA_ARCH=121 \
  --build-arg GLMRT_ENGINE_COMMIT="$engine_commit" \
  --build-arg GLMRT_SPARKINFER_COMMIT="$sparkinfer_commit" \
  -f docker/Dockerfile.release \
  -t "$inference_image" .
REMOTE
verify_remote_source_manifest

echo "== exporting release binaries =="
mkdir -p "$repo_root/dist/coordinator" "$repo_root/dist/spark-expert"
find "$repo_root/dist/coordinator" "$repo_root/dist/spark-expert" \
  -mindepth 1 -delete
coordinator_container="$(docker create "$COORDINATOR_DOCKER_INFERENCE")"
trap 'docker rm -f "$coordinator_container" >/dev/null 2>&1 || true' EXIT
docker cp "$coordinator_container:/opt/glmrt/bin/glmrt" "$repo_root/dist/coordinator/glmrt"
docker cp "$coordinator_container:/opt/glmrt/lib/libglmrt_native.so" "$repo_root/dist/coordinator/libglmrt_native.so"
docker cp \
  "$coordinator_container:/opt/glmrt/share/THIRD_PARTY_NOTICES.md" \
  "$repo_root/dist/coordinator/THIRD_PARTY_NOTICES.md"
docker cp \
  "$coordinator_container:/opt/glmrt/share/SPARKINFER_PROVENANCE.json" \
  "$repo_root/dist/coordinator/SPARKINFER_PROVENANCE.json"
docker cp \
  "$coordinator_container:/opt/glmrt/share/licenses/sparkinfer/LICENSE" \
  "$repo_root/dist/coordinator/SPARKINFER_LICENSE"
docker cp \
  "$coordinator_container:/opt/glmrt/share/SPARKINFER_SHA256SUMS" \
  "$repo_root/dist/coordinator/SPARKINFER_SHA256SUMS"
docker cp \
  "$coordinator_container:/opt/glmrt/share/XGRAMMAR_PROVENANCE.json" \
  "$repo_root/dist/coordinator/XGRAMMAR_PROVENANCE.json"
docker cp \
  "$coordinator_container:/opt/glmrt/share/licenses/xgrammar/LICENSE" \
  "$repo_root/dist/coordinator/XGRAMMAR_LICENSE"
docker cp \
  "$coordinator_container:/opt/glmrt/share/XGRAMMAR_SHA256SUMS" \
  "$repo_root/dist/coordinator/XGRAMMAR_SHA256SUMS"
docker rm "$coordinator_container" >/dev/null
trap - EXIT

ssh -o BatchMode=yes "$seed_host" bash -s -- \
  "$SPARK_EXPERT_DOCKER_INFERENCE" "$remote_dir/dist/spark-expert" <<'REMOTE'
set -euo pipefail
image="$1"
destination="$2"
mkdir -p "$destination"
find "$destination" -mindepth 1 -delete
container="$(docker create "$image")"
trap 'docker rm -f "$container" >/dev/null 2>&1 || true' EXIT
docker cp "$container:/opt/glmrt/bin/glmrt" "$destination/glmrt"
docker cp "$container:/opt/glmrt/lib/libglmrt_native.so" "$destination/libglmrt_native.so"
docker cp \
  "$container:/opt/glmrt/share/THIRD_PARTY_NOTICES.md" \
  "$destination/THIRD_PARTY_NOTICES.md"
docker cp \
  "$container:/opt/glmrt/share/SPARKINFER_PROVENANCE.json" \
  "$destination/SPARKINFER_PROVENANCE.json"
docker cp \
  "$container:/opt/glmrt/share/licenses/sparkinfer/LICENSE" \
  "$destination/SPARKINFER_LICENSE"
docker cp \
  "$container:/opt/glmrt/share/SPARKINFER_SHA256SUMS" \
  "$destination/SPARKINFER_SHA256SUMS"
docker cp \
  "$container:/opt/glmrt/share/XGRAMMAR_PROVENANCE.json" \
  "$destination/XGRAMMAR_PROVENANCE.json"
docker cp \
  "$container:/opt/glmrt/share/licenses/xgrammar/LICENSE" \
  "$destination/XGRAMMAR_LICENSE"
docker cp \
  "$container:/opt/glmrt/share/XGRAMMAR_SHA256SUMS" \
  "$destination/XGRAMMAR_SHA256SUMS"
docker rm "$container" >/dev/null
trap - EXIT
REMOTE
release_sync --delete \
  "$seed_host:$remote_dir/dist/spark-expert/" \
  "$repo_root/dist/spark-expert/"
dist_source_manifest=()
if [[ -n "$source_manifest" ]]; then
  install -m 0644 "$source_manifest" "$repo_root/dist/SOURCE_SHA256SUMS"
  dist_source_manifest+=(SOURCE_SHA256SUMS)
fi
for role in coordinator spark-expert; do
  python3 "$repo_root/scripts/sparkinfer-release-provenance.py" \
    --source "$repo_root/third_party/sparkinfer" \
    --lock "$repo_root/third_party/sparkinfer.lock.json" \
    --license "$repo_root/dist/$role/SPARKINFER_LICENSE" \
    --notices "$repo_root/dist/$role/THIRD_PARTY_NOTICES.md" \
    --verify "$repo_root/dist/$role/SPARKINFER_PROVENANCE.json"
  (
    cd "$repo_root/dist/$role"
    sha256sum -c SPARKINFER_SHA256SUMS
    sha256sum -c XGRAMMAR_SHA256SUMS
  )
done
(
  cd "$repo_root/dist"
  sha256sum \
    coordinator/glmrt coordinator/libglmrt_native.so \
    coordinator/THIRD_PARTY_NOTICES.md \
    coordinator/SPARKINFER_PROVENANCE.json \
    coordinator/SPARKINFER_LICENSE \
    coordinator/SPARKINFER_SHA256SUMS \
    coordinator/XGRAMMAR_PROVENANCE.json \
    coordinator/XGRAMMAR_LICENSE \
    coordinator/XGRAMMAR_SHA256SUMS \
    spark-expert/glmrt spark-expert/libglmrt_native.so \
    spark-expert/THIRD_PARTY_NOTICES.md \
    spark-expert/SPARKINFER_PROVENANCE.json \
    spark-expert/SPARKINFER_LICENSE \
    spark-expert/SPARKINFER_SHA256SUMS \
    spark-expert/XGRAMMAR_PROVENANCE.json \
    spark-expert/XGRAMMAR_LICENSE \
    spark-expert/XGRAMMAR_SHA256SUMS \
    "${dist_source_manifest[@]}" >SHA256SUMS
  sha256sum -c SHA256SUMS
)
verify_local_source_manifest
verify_remote_source_manifest

echo "== distributing fresh Spark inference image =="
for host in "$SPARK_1_HOST" "$SPARK_2_HOST" "$SPARK_3_HOST"; do
  # A stopped expert container can still reference the prior image ID. Force
  # removal only untags that image while preserving the referenced layers, so
  # ensure_image cannot mistake the stale tag for the fresh seed image.
  ssh -o BatchMode=yes "$host" "docker image rm --force '$SPARK_EXPERT_DOCKER_INFERENCE' >/dev/null 2>&1 || true"
done
rdmapipe_ready=1
for host in "$seed_host" "$SPARK_1_HOST" "$SPARK_2_HOST" "$SPARK_3_HOST"; do
  if ! ssh -o BatchMode=yes "$host" "command -v rdmapipe >/dev/null 2>&1"; then
    rdmapipe_ready=0
    break
  fi
done
if ((rdmapipe_ready)); then
  echo "== concurrently distributing Spark image over RDMA =="
  image_copy_pids=()
  image_copy_hosts=()
  printf -v spark_image_quoted '%q' "$SPARK_EXPERT_DOCKER_INFERENCE"
  for host in "$SPARK_1_HOST" "$SPARK_2_HOST" "$SPARK_3_HOST"; do
    (
      set -o pipefail
      echo "== RDMA image copy $seed_host -> $host =="
      ssh -o BatchMode=yes "$seed_host" \
        "set -o pipefail; docker image save $spark_image_quoted | rdmapipe --send" |
        ssh -o BatchMode=yes "$host" \
          "set -o pipefail; rdmapipe --recv | docker image load"
      echo "== RDMA image copy $seed_host -> $host complete =="
    ) &
    image_copy_pids+=("$!")
    image_copy_hosts+=("$host")
  done
  image_copy_failed=0
  for index in "${!image_copy_pids[@]}"; do
    if ! wait "${image_copy_pids[$index]}"; then
      echo "RDMA image copy to ${image_copy_hosts[$index]} failed" >&2
      image_copy_failed=1
    fi
  done
  ((image_copy_failed == 0)) || release_die "concurrent RDMA Spark image distribution failed"
else
  echo "== rdmapipe unavailable on one or more Sparks; using serial netcat image distribution =="
  GLMRT_SPARK_HOSTS="$hosts_csv" \
  GLMRT_SPARK_IMAGE="$SPARK_EXPERT_DOCKER_INFERENCE" \
  GLMRT_SPARK_IMAGE_SEED_HOST="$seed_host" \
  GLMRT_SPARK_IMAGE_COPY_METHOD=spark-netcat \
  GLMRT_SPARK_IMAGE_ONLY=1 \
  GLMRT_SPARK_SKIP_STAGE=1 \
  "$repo_root/scripts/phase0-spark-tcp-bench.sh"
fi

coordinator_revision="$(
  docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' \
    "$COORDINATOR_DOCKER_INFERENCE"
)"
[[ "$coordinator_revision" == "$engine_commit" ]] || release_die "coordinator image revision mismatch"
coordinator_sparkinfer_revision="$(
  docker image inspect -f '{{index .Config.Labels "io.glmrt.sparkinfer.revision"}}' \
    "$COORDINATOR_DOCKER_INFERENCE"
)"
[[ "$coordinator_sparkinfer_revision" == "$sparkinfer_commit" ]] ||
  release_die "coordinator image SparkInfer revision mismatch"
if [[ -n "$source_manifest_sha256" ]]; then
  coordinator_source_manifest="$(
    docker image inspect -f '{{index .Config.Labels "io.glmrt.source-manifest.sha256"}}' \
      "$COORDINATOR_DOCKER_INFERENCE"
  )"
  [[ "$coordinator_source_manifest" == "$source_manifest_sha256" ]] ||
    release_die "coordinator image source manifest mismatch"
fi
for host in "$SPARK_0_HOST" "$SPARK_1_HOST" "$SPARK_2_HOST" "$SPARK_3_HOST"; do
  revision="$(
    ssh -o BatchMode=yes "$host" \
      "docker image inspect -f '{{index .Config.Labels \"org.opencontainers.image.revision\"}}' '$SPARK_EXPERT_DOCKER_INFERENCE'"
  )"
  [[ "$revision" == "$engine_commit" ]] || release_die "$host Spark image revision mismatch: $revision"
  spark_sparkinfer_revision="$(
    ssh -o BatchMode=yes "$host" \
      "docker image inspect -f '{{index .Config.Labels \"io.glmrt.sparkinfer.revision\"}}' '$SPARK_EXPERT_DOCKER_INFERENCE'"
  )"
  [[ "$spark_sparkinfer_revision" == "$sparkinfer_commit" ]] ||
    release_die "$host Spark image SparkInfer revision mismatch: $spark_sparkinfer_revision"
  if [[ -n "$source_manifest_sha256" ]]; then
    spark_source_manifest="$(
      ssh -o BatchMode=yes "$host" \
        "docker image inspect -f '{{index .Config.Labels \"io.glmrt.source-manifest.sha256\"}}' '$SPARK_EXPERT_DOCKER_INFERENCE'"
    )"
    [[ "$spark_source_manifest" == "$source_manifest_sha256" ]] ||
      release_die "$host Spark image source manifest mismatch: $spark_source_manifest"
  fi
done

echo "Build complete."
echo "  revision:    $engine_commit"
echo "  SparkInfer:  $sparkinfer_commit"
if [[ -n "$source_manifest_sha256" ]]; then
  echo "  source:      $source_manifest_sha256"
fi
echo "  coordinator: $COORDINATOR_DOCKER_INFERENCE"
echo "  spark:       $SPARK_EXPERT_DOCKER_INFERENCE"
echo "  artifacts:   $repo_root/dist"
