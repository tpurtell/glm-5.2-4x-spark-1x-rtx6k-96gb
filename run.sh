#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$repo_root/scripts/release-common.sh"

usage() {
  cat <<'EOF'
Usage: ./run.sh [--profile FILE] [--restart] [--dry-run]

Uses glmrt.config beside this script by default. Despite the option name,
--profile FILE selects an entire alternate configuration file.

--restart  gracefully exits the current API and recreates all five containers
--dry-run  performs configuration/image/model/resource checks without mutation
EOF
}

config="$repo_root/glmrt.config"
restart=0
dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile|--config)
      config="${2:?$1 requires a configuration file}"
      shift 2
      ;;
    --restart)
      restart=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      release_die "unknown run argument: $1"
      ;;
  esac
done

release_load_config "$config"
release_need docker
release_need ssh
release_need rsync
release_need jq
release_need curl
release_need ss
release_need nvidia-smi
release_need sha256sum

if ((restart && dry_run)); then
  release_die "--restart and --dry-run are mutually exclusive"
fi

docker info >/dev/null 2>&1 || release_die "local Docker daemon is unavailable"
docker image inspect "$COORDINATOR_DOCKER_INFERENCE" >/dev/null 2>&1 ||
  release_die "coordinator image is missing: $COORDINATOR_DOCKER_INFERENCE (run ./build.sh)"

hosts_csv="$(release_hosts_csv)"
lane_a_csv="$(release_lane_a_csv)"
lane_b_csv="$(release_lane_b_csv)"
expert_hosts_csv="$(release_expert_hosts_csv)"
coordinator_container="$RELEASE_COORDINATOR_CONTAINER_NAME"
spark_container_prefix="$RELEASE_SPARK_CONTAINER_PREFIX"
state_dir="$repo_root/.glmrt-release"
mkdir -p "$state_dir"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$hf_home"

echo "== checking SSH, Docker images, and current containers =="
running_release_sparks=0
stale_release_sparks=0
running_legacy_sparks=0
for host in "$SPARK_0_HOST" "$SPARK_1_HOST" "$SPARK_2_HOST" "$SPARK_3_HOST"; do
  release_container="${spark_container_prefix}-${host}-${EXPERT_PORT}"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" \
    "docker info >/dev/null && docker image inspect '$SPARK_EXPERT_DOCKER_INFERENCE' >/dev/null" ||
    release_die "$host is unreachable or lacks image $SPARK_EXPERT_DOCKER_INFERENCE"
  if ssh -o BatchMode=yes "$host" "docker ps --format '{{.Names}}' | grep -Fx '$release_container'" >/dev/null; then
    image_state="$(
      ssh -o BatchMode=yes "$host" bash -s -- \
        "$release_container" "$SPARK_EXPERT_DOCKER_INFERENCE" <<'REMOTE'
container="$1"
image="$2"
container_image="$(docker inspect -f '{{.Image}}' "$container")"
selected_image="$(docker image inspect -f '{{.Id}}' "$image")"
if [ "$container_image" = "$selected_image" ]; then
  echo current
else
  echo stale
fi
REMOTE
    )"
    if [[ "$image_state" == current ]]; then
      echo "  $host: current release expert container already running ($release_container)"
      ((running_release_sparks += 1))
    else
      echo "  $host: release expert container uses a stale image ($release_container)"
      ((stale_release_sparks += 1))
    fi
  else
    echo "  $host: image ready; no current release expert container"
  fi
  if ssh -o BatchMode=yes "$host" "docker ps --format '{{.Names}}' | grep -Eq '^glmrt-phase0-tcp-expertd-${host}-${EXPERT_PORT}$'" >/dev/null; then
    echo "  $host: legacy expert container currently occupies the GPU"
    ((running_legacy_sparks += 1))
  fi
done

release_coordinator_running=0
stale_release_coordinator=0
host_api_running=0
if docker ps --format '{{.Names}}' | grep -Fx "$coordinator_container" >/dev/null; then
  coordinator_image="$(docker inspect -f '{{.Image}}' "$coordinator_container")"
  selected_coordinator_image="$(docker image inspect -f '{{.Id}}' "$COORDINATOR_DOCKER_INFERENCE")"
  if [[ "$coordinator_image" == "$selected_coordinator_image" ]]; then
    echo "  coordinator: current release API container already running ($coordinator_container)"
    release_coordinator_running=1
  else
    echo "  coordinator: release API container uses a stale image ($coordinator_container)"
    stale_release_coordinator=1
  fi
fi
if ss -ltnp "sport = :${ADDR##*:}" 2>/dev/null | grep -q glmrt &&
  ! docker ps --format '{{.Names}}' | grep -Fx "$coordinator_container" >/dev/null; then
  echo "  coordinator: host glmrt API process currently occupies ${ADDR##*:}"
  host_api_running=1
elif ((release_coordinator_running == 0 && stale_release_coordinator == 0)); then
  echo "  coordinator: image ready; no API process running"
fi

profile_args=(
  --repo-root "$repo_root"
  --profile "$PROFILE"
  --model "$MODEL"
  --speculation "$SPECULATION"
  --vision "$RELEASE_VISION"
  --mtp-bf16-experts "$MTP_BF16_EXPERTS"
  --headroom-gib "$COORDINATOR_GPU_HEADROOM_GIB"
  --concurrency "$CONCURRENCY"
  --dry-run
)
[[ -z "$KV_POOL_TOKENS" ]] || profile_args+=(--kv-pool-tokens "$KV_POOL_TOKENS")
[[ -z "$MAX_CONTEXT_TOKENS" ]] || profile_args+=(--max-context-tokens "$MAX_CONTEXT_TOKENS")
[[ -z "$MAX_OUTPUT_TOKENS" ]] || profile_args+=(--max-output-tokens "$MAX_OUTPUT_TOKENS")

resolve_profile() {
  docker run --rm \
    --gpus all \
    --net=host \
    -v "$repo_root:$repo_root:ro" \
    -v "$hf_home:$hf_home:ro" \
    -v "$hf_home:/root/.cache/huggingface:ro" \
    -e HF_HOME="$hf_home" \
    -e GLMRT_DSPARK_MODEL_ID="$RELEASE_DSPARK_MODEL_ID" \
    -e GLMRT_DSPARK_REVISION="$RELEASE_DSPARK_REVISION" \
    "$COORDINATOR_DOCKER_INFERENCE" \
    python3 /opt/glmrt/python/tools/resolve_serve_profile.py "${profile_args[@]}"
}

resolved_json="$state_dir/resolved-profile.json"
resolve_profile >"$resolved_json"
jq -e . "$resolved_json" >/dev/null || release_die "profile resolver returned invalid JSON"

blockers="$(jq -r '.blockers[]?' "$resolved_json")"
[[ -z "$blockers" ]] || release_die "profile blockers:\n$blockers"
deployment_fingerprint="$(
  {
    jq -S . "$resolved_json"
    printf '%s\n' \
      "$ADDR" "$EXPERT_PORT" \
      "$SPARK_0_HOST" "$SPARK_0_LANE_A" "$SPARK_0_LANE_B" \
      "$SPARK_1_HOST" "$SPARK_1_LANE_A" "$SPARK_1_LANE_B" \
      "$SPARK_2_HOST" "$SPARK_2_LANE_A" "$SPARK_2_LANE_B" \
      "$SPARK_3_HOST" "$SPARK_3_LANE_A" "$SPARK_3_LANE_B"
  } | sha256sum | awk '{print $1}'
)"

check_model_cache_local() {
  local model_id="$1"
  local revision="${2:-}"
  local root="$hf_home/hub/models--${model_id//\//--}"
  if [[ -z "$revision" && -f "$root/refs/main" ]]; then
    revision="$(<"$root/refs/main")"
  fi
  [[ -n "$revision" && -d "$root/snapshots/$revision" ]] ||
    release_die "coordinator model snapshot is missing: $model_id${revision:+@$revision} under $hf_home"
  if find "$root/snapshots/$revision" -xtype l -print -quit | grep -q .; then
    release_die "coordinator model snapshot has unresolved blobs: $model_id@$revision"
  fi
}

check_model_cache_remote() {
  local host="$1"
  local model_id="$2"
  ssh -o BatchMode=yes "$host" bash -s -- "$model_id" <<'REMOTE'
set -euo pipefail
model_id="$1"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
root="$hf_home/hub/models--${model_id//\//--}"
test -f "$root/refs/main"
revision="$(<"$root/refs/main")"
test -n "$revision"
test -d "$root/snapshots/$revision"
if find "$root/snapshots/$revision" -xtype l -print -quit | grep -q .; then
  exit 1
fi
REMOTE
}

echo "== checking model snapshots =="
check_model_cache_local "$RELEASE_MODEL_ID"
if [[ "$SPECULATION" == "dspark" ]]; then
  check_model_cache_local "$RELEASE_DSPARK_MODEL_ID" "$RELEASE_DSPARK_REVISION"
fi
for host in "$SPARK_0_HOST" "$SPARK_1_HOST" "$SPARK_2_HOST" "$SPARK_3_HOST"; do
  check_model_cache_remote "$host" "$RELEASE_MODEL_ID" ||
    release_die "$host is missing the selected text model: $RELEASE_MODEL_ID"
done
echo "  selected model snapshots are complete"

services_active=$((
  release_coordinator_running + stale_release_coordinator + host_api_running +
    running_release_sparks + stale_release_sparks + running_legacy_sparks
))
if ((services_active)); then
  if ((!restart)); then
    deployment_matches=0
    if ((release_coordinator_running == 1 &&
      running_release_sparks == 4 &&
      stale_release_coordinator == 0 &&
      stale_release_sparks == 0 &&
      running_legacy_sparks == 0 &&
      host_api_running == 0)); then
      coordinator_fingerprint="$(
        docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$coordinator_container" |
          sed -n 's/^GLMRT_RELEASE_CONFIG_SHA256=//p'
      )"
      spark_fingerprint_matches=1
      for host in "$SPARK_0_HOST" "$SPARK_1_HOST" "$SPARK_2_HOST" "$SPARK_3_HOST"; do
        release_container="${spark_container_prefix}-${host}-${EXPERT_PORT}"
        remote_fingerprint="$(
          ssh -o BatchMode=yes "$host" \
            "docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' '$release_container'" |
            sed -n 's/^GLMRT_RELEASE_CONFIG_SHA256=//p'
        )"
        [[ "$remote_fingerprint" == "$deployment_fingerprint" ]] ||
          spark_fingerprint_matches=0
      done
      if [[ "$coordinator_fingerprint" == "$deployment_fingerprint" ]] &&
        ((spark_fingerprint_matches)); then
        deployment_matches=1
      fi
    fi
    if ((deployment_matches)) &&
      curl -fsS "http://127.0.0.1:${ADDR##*:}/v1/models" >/dev/null; then
      echo "All five release services already match the selected images and configuration."
      exit 0
    fi
    release_die "partial, legacy, stale, or configuration-mismatched service state is active; use --restart"
  fi
  echo "== stopping existing API and expert containers =="
  release_stop_services "$coordinator_container" "$spark_container_prefix"
fi

check_local_resources() {
  local available_kib gpu_line total_mib free_mib
  available_kib="$(awk '/MemAvailable:/{print $2}' /proc/meminfo)"
  ((available_kib >= 8 * 1024 * 1024)) ||
    release_die "coordinator has less than 8 GiB available system memory"
  gpu_line="$(nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits | head -1)"
  IFS=, read -r total_mib free_mib <<<"$gpu_line"
  total_mib="$(release_trim "$total_mib")"
  free_mib="$(release_trim "$free_mib")"
  ((free_mib >= 80 * 1024)) ||
    release_die "coordinator GPU has only ${free_mib} MiB free of ${total_mib} MiB"
  echo "  coordinator: RAM $((available_kib / 1024)) MiB available; GPU ${free_mib}/${total_mib} MiB free"
}

check_remote_resources() {
  local host="$1"
  ssh -o BatchMode=yes "$host" bash -s -- "$SPARK_EXPERT_DOCKER_INFERENCE" <<'REMOTE'
set -euo pipefail
image="$1"
available_kib="$(awk '/MemAvailable:/{print $2}' /proc/meminfo)"
if ((available_kib < 64 * 1024 * 1024)); then
  echo "only $((available_kib / 1024)) MiB unified system/GPU memory is available" >&2
  exit 2
fi
gpu_line="$(
  docker run --rm --gpus all "$image" \
    nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits |
    head -1
)"
IFS=, read -r total_mib free_mib <<<"$gpu_line"
total_mib="${total_mib//[[:space:]]/}"
free_mib="${free_mib//[[:space:]]/}"
if [[ "$total_mib" =~ ^[0-9]+$ && "$free_mib" =~ ^[0-9]+$ ]] &&
  ((free_mib < 64 * 1024)); then
  echo "GPU has only ${free_mib} MiB free of ${total_mib} MiB" >&2
  exit 2
fi
if [[ "$total_mib" =~ ^[0-9]+$ && "$free_mib" =~ ^[0-9]+$ ]]; then
  echo "RAM $((available_kib / 1024)) MiB available; GPU ${free_mib}/${total_mib} MiB free"
else
  echo "unified system/GPU memory $((available_kib / 1024)) MiB available"
fi
REMOTE
}

echo "== checking launch headroom =="
check_local_resources
for host in "$SPARK_0_HOST" "$SPARK_1_HOST" "$SPARK_2_HOST" "$SPARK_3_HOST"; do
  echo "  $host: $(check_remote_resources "$host")"
done
if ((dry_run)); then
  echo "Dry-run checks passed."
  exit 0
fi

echo "== starting fresh prebuilt Spark expert containers =="
eval "$(
  jq -r '.environment | to_entries[] | "\(.key)=\(.value | @sh); export \(.key)"' \
    "$resolved_json"
)"
export GLMRT_SPARK_HOSTS="$hosts_csv"
export GLMRT_REAL_FULL_SERVE_EXPERT_HOSTS="$expert_hosts_csv"
export GLMRT_SPARK_IMAGE="$SPARK_EXPERT_DOCKER_INFERENCE"
export GLMRT_SPARK_PREBUILT=1
export GLMRT_RELEASE_CONFIG_SHA256="$deployment_fingerprint"
export GLMRT_SPARK_CONTAINER_PREFIX="$spark_container_prefix"
export GLMRT_SPARK_EXPERT_PORT="$EXPERT_PORT"
export GLMRT_SPARK_EXPERT_TRANSPORT=verbs-host
export GLMRT_SPARK_KEEP_EXPERTS=1
export GLMRT_SPARK_EXPERT_REAL_LAYER=all
export GLMRT_PHASE0_SPARK_SKIP_BENCH=1
export GLMRT_EXPERT_INTERMEDIATE_RDMA_PEERS="$lane_a_csv"
if [[ -n "$lane_b_csv" ]]; then
  export GLMRT_EXPERT_INTERMEDIATE_RDMA_ADDITIONAL_PEERS="$lane_b_csv"
else
  unset GLMRT_EXPERT_INTERMEDIATE_RDMA_ADDITIONAL_PEERS || true
fi
"$repo_root/scripts/phase0-spark-tcp-bench.sh"

env_file="$state_dir/coordinator.env"
jq -r '.environment | to_entries[] | "\(.key)=\(.value)"' "$resolved_json" >"$env_file"
{
  echo "ADDR=$ADDR"
  echo "GLMRT_REAL_FULL_SERVE_EXPERT_HOSTS=$expert_hosts_csv"
  echo "GLMRT_SPARK_HOSTS=$hosts_csv"
  echo "GLMRT_SPARK_EXPERT_PORT=$EXPERT_PORT"
  echo "GLMRT_REAL_FULL_SERVE_START_EXPERTS=0"
  echo "GLMRT_REAL_FULL_SERVE_BUILD_DAEMON=0"
  echo "GLMRT_REAL_FULL_SERVE_BUILD_NATIVE=0"
  echo "GLMRT_REAL_FULL_SERVE_REQUIRE_CUDA=1"
  echo "GLMRT_BIN=/opt/glmrt/bin/glmrt"
  echo "GLMRT_NATIVE_LIB=/opt/glmrt/lib/libglmrt_native.so"
  echo "GLMRT_ENGINE_COMMIT=$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$COORDINATOR_DOCKER_INFERENCE")"
  echo "GLMRT_RELEASE_CONFIG_SHA256=$deployment_fingerprint"
  echo "FLASHINFER_WORKSPACE_BASE=/var/cache/glmrt/flashinfer"
} >>"$env_file"

mkdir -p "$state_dir/flashinfer"
docker_args=(
  run -d
  --name "$coordinator_container"
  --restart unless-stopped
  --gpus all
  --net=host
  --ipc=host
  --ulimit memlock=-1:-1
  --cap-add IPC_LOCK
  --env-file "$env_file"
  --workdir "$repo_root"
  -v "$repo_root:$repo_root:ro"
  -v "$hf_home:$hf_home:ro"
  -v "$hf_home:/root/.cache/huggingface:ro"
  -v "$state_dir/flashinfer:/var/cache/glmrt/flashinfer"
  -e HF_HOME="$hf_home"
)
if [[ -e /dev/infiniband ]]; then
  docker_args+=(--device=/dev/infiniband)
fi
docker_args+=(
  "$COORDINATOR_DOCKER_INFERENCE"
  /opt/glmrt/scripts/real-full-tcp-serve.sh
)

echo "== starting coordinator container =="
docker "${docker_args[@]}" >/dev/null

deadline=$((SECONDS + 900))
until curl -fsS "http://127.0.0.1:${ADDR##*:}/v1/models" >/dev/null 2>&1; do
  coordinator_state="$(
    docker inspect -f '{{.State.Status}} {{.RestartCount}}' \
      "$coordinator_container" 2>/dev/null || true
  )"
  if [[ "$coordinator_state" != "running 0" ]]; then
    docker logs --tail 200 "$coordinator_container" >&2 || true
    release_die "coordinator container became unhealthy during startup (state: ${coordinator_state:-missing})"
  fi
  ((SECONDS < deadline)) || {
    docker logs --tail 200 "$coordinator_container" >&2 || true
    release_die "API did not become ready within 900 seconds"
  }
  sleep 2
done

curl -fsS "http://127.0.0.1:${ADDR##*:}/v1/models" >"$state_dir/models.json"
echo "GLMRT release server is ready at http://127.0.0.1:${ADDR##*:}/v1/"
echo "  profile:     $PROFILE"
echo "  model:       $RELEASE_MODEL_ID"
echo "  speculation: $SPECULATION"
echo "  concurrency: $CONCURRENCY"
echo "  containers:  $coordinator_container + four $spark_container_prefix experts"
