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

docker info >/dev/null 2>&1 || release_die "local Docker daemon is unavailable"
engine_commit="$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || echo unknown)"
if [[ -n "$(git -C "$repo_root" status --porcelain 2>/dev/null || true)" ]]; then
  engine_commit="${engine_commit}-dirty"
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
  --build-arg GLMRT_ROLE=coordinator \
  --build-arg CUDA_ARCH=120 \
  --build-arg GLMRT_ENGINE_COMMIT="$engine_commit" \
  -f "$repo_root/docker/Dockerfile.release" \
  -t "$COORDINATOR_DOCKER_INFERENCE" \
  "$repo_root"

echo "== staging native Spark build on $seed_host:$remote_dir =="
ssh -o BatchMode=yes "$seed_host" "mkdir -p '$remote_dir'"
rsync -az --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '.glmrt-cache/' \
  --exclude '.glmrt-release-image/' \
  --exclude 'dist/' \
  --exclude 'rust/target/' \
  --exclude 'native/build*/' \
  --exclude 'reports/phase0_artifacts/benchmarks/' \
  --exclude 'reports/phase0_artifacts/logs/' \
  "$repo_root/" "$seed_host:$remote_dir/"

echo "== building Spark development and inference images natively on $seed_host =="
ssh -o BatchMode=yes "$seed_host" bash -s -- \
  "$remote_dir" "$SPARK_EXPERT_DOCKER_DEV" "$SPARK_EXPERT_DOCKER_INFERENCE" "$engine_commit" <<'REMOTE'
set -euo pipefail
remote_dir="$1"
dev_image="$2"
inference_image="$3"
engine_commit="$4"
cd "$remote_dir"
docker build \
  --build-arg GLMRT_ROLE=expert \
  --build-arg CUDA_ARCH=121 \
  --build-arg TARGET_PLATFORM=linux/arm64 \
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
  --build-arg GLMRT_ROLE=expert \
  --build-arg CUDA_ARCH=121 \
  --build-arg GLMRT_ENGINE_COMMIT="$engine_commit" \
  -f docker/Dockerfile.release \
  -t "$inference_image" .
REMOTE

echo "== exporting release binaries =="
mkdir -p "$repo_root/dist/coordinator" "$repo_root/dist/spark-expert"
coordinator_container="$(docker create "$COORDINATOR_DOCKER_INFERENCE")"
trap 'docker rm -f "$coordinator_container" >/dev/null 2>&1 || true' EXIT
docker cp "$coordinator_container:/opt/glmrt/bin/glmrt" "$repo_root/dist/coordinator/glmrt"
docker cp "$coordinator_container:/opt/glmrt/lib/libglmrt_native.so" "$repo_root/dist/coordinator/libglmrt_native.so"
docker cp \
  "$coordinator_container:/opt/glmrt/share/THIRD_PARTY_NOTICES.md" \
  "$repo_root/dist/coordinator/THIRD_PARTY_NOTICES.md"
docker rm "$coordinator_container" >/dev/null
trap - EXIT

ssh -o BatchMode=yes "$seed_host" bash -s -- \
  "$SPARK_EXPERT_DOCKER_INFERENCE" "$remote_dir/dist/spark-expert" <<'REMOTE'
set -euo pipefail
image="$1"
destination="$2"
mkdir -p "$destination"
container="$(docker create "$image")"
trap 'docker rm -f "$container" >/dev/null 2>&1 || true' EXIT
docker cp "$container:/opt/glmrt/bin/glmrt" "$destination/glmrt"
docker cp "$container:/opt/glmrt/lib/libglmrt_native.so" "$destination/libglmrt_native.so"
docker cp \
  "$container:/opt/glmrt/share/THIRD_PARTY_NOTICES.md" \
  "$destination/THIRD_PARTY_NOTICES.md"
docker rm "$container" >/dev/null
trap - EXIT
REMOTE
rsync -az "$seed_host:$remote_dir/dist/spark-expert/" "$repo_root/dist/spark-expert/"
(
  cd "$repo_root/dist"
  sha256sum \
    coordinator/glmrt coordinator/libglmrt_native.so \
    coordinator/THIRD_PARTY_NOTICES.md \
    spark-expert/glmrt spark-expert/libglmrt_native.so \
    spark-expert/THIRD_PARTY_NOTICES.md >SHA256SUMS
)

echo "== distributing fresh Spark inference image =="
for host in "$SPARK_1_HOST" "$SPARK_2_HOST" "$SPARK_3_HOST"; do
  ssh -o BatchMode=yes "$host" "docker image rm '$SPARK_EXPERT_DOCKER_INFERENCE' >/dev/null 2>&1 || true"
done
GLMRT_SPARK_HOSTS="$hosts_csv" \
GLMRT_SPARK_IMAGE="$SPARK_EXPERT_DOCKER_INFERENCE" \
GLMRT_SPARK_IMAGE_SEED_HOST="$seed_host" \
GLMRT_SPARK_IMAGE_COPY_METHOD=spark-netcat \
GLMRT_SPARK_IMAGE_ONLY=1 \
GLMRT_SPARK_SKIP_STAGE=1 \
"$repo_root/scripts/phase0-spark-tcp-bench.sh"

coordinator_revision="$(
  docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' \
    "$COORDINATOR_DOCKER_INFERENCE"
)"
[[ "$coordinator_revision" == "$engine_commit" ]] || release_die "coordinator image revision mismatch"
for host in "$SPARK_0_HOST" "$SPARK_1_HOST" "$SPARK_2_HOST" "$SPARK_3_HOST"; do
  revision="$(
    ssh -o BatchMode=yes "$host" \
      "docker image inspect -f '{{index .Config.Labels \"org.opencontainers.image.revision\"}}' '$SPARK_EXPERT_DOCKER_INFERENCE'"
  )"
  [[ "$revision" == "$engine_commit" ]] || release_die "$host Spark image revision mismatch: $revision"
done

echo "Build complete."
echo "  revision:    $engine_commit"
echo "  coordinator: $COORDINATOR_DOCKER_INFERENCE"
echo "  spark:       $SPARK_EXPERT_DOCKER_INFERENCE"
echo "  artifacts:   $repo_root/dist"
