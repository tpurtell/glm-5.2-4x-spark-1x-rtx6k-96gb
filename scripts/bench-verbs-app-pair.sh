#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: scripts/bench-verbs-app-pair.sh HOST_A HOST_B

Stages this repo on two Spark hosts, ensures the arm64 Spark image is present,
builds the ARM glmrt daemon and RDMA-enabled native library remotely, then runs
the app-level verbs ProtocolV2 benchmark with HOST_B as server and HOST_A as
client.

Environment:
  GLMRT_SPARK_IMAGE                 default: glmrt-dev:spark
  GLMRT_SPARK_IMAGE_COPY_METHOD     spark-netcat, ssh-relay, or none; default: spark-netcat
  GLMRT_SPARK_IMAGE_SEED_HOST       default: first host with GLMRT_SPARK_IMAGE
  GLMRT_SPARK_IMAGE_LINK_SUFFIX     default: .200gb for spark-netcat data path
  GLMRT_SPARK_IMAGE_COPY_PORT       default: 29421
  GLMRT_SPARK_BUILD_IMAGE           set 1 to build remotely when no seed image exists
  GLMRT_VERBS_APP_WORKDIR           default: $HOME/glmrt-phase0-verbs-app
  GLMRT_VERBS_APP_PORT              default: 18615
  GLMRT_VERBS_APP_DURATION_SECS     default: 2
  GLMRT_VERBS_APP_PAYLOAD_BYTES     default: GLM 1/2/4/8/16/64/256/512 BF16 rows
  GLMRT_VERBS_APP_ROUNDTRIP_ITERATIONS optional daemon benchmark override
  GLMRT_VERBS_APP_CHAIN_ITERATIONS      optional daemon benchmark override
  GLMRT_VERBS_APP_CHAIN_HOPS            optional daemon benchmark override
  GLMRT_VERBS_APP_CONTROL_TIMEOUT_SECS  optional daemon benchmark override
  GLMRT_VERBS_APP_KEEP_REMOTE       set 1 to keep staged remote workdirs
EOF
  exit 0
fi

host_a="${1:?HOST_A required}"
host_b="${2:?HOST_B required}"

remote_dir="${GLMRT_VERBS_APP_WORKDIR:-}"
image="${GLMRT_SPARK_IMAGE:-glmrt-dev:spark}"
image_copy_method="${GLMRT_SPARK_IMAGE_COPY_METHOD:-spark-netcat}"
image_seed_host="${GLMRT_SPARK_IMAGE_SEED_HOST:-}"
image_link_suffix="${GLMRT_SPARK_IMAGE_LINK_SUFFIX:-.200gb}"
image_copy_port="${GLMRT_SPARK_IMAGE_COPY_PORT:-29421}"
build_missing_image="${GLMRT_SPARK_BUILD_IMAGE:-0}"
port="${GLMRT_VERBS_APP_PORT:-18615}"
duration="${GLMRT_VERBS_APP_DURATION_SECS:-2}"
payloads="${GLMRT_VERBS_APP_PAYLOAD_BYTES:-12288,24576,49152,98304,196608,786432,3145728,6291456}"
keep_remote="${GLMRT_VERBS_APP_KEEP_REMOTE:-0}"
benchmark_dir="$repo_root/reports/phase0_artifacts/benchmarks"
log_dir="$repo_root/reports/phase0_artifacts/logs"
mkdir -p "$benchmark_dir" "$log_dir"

safe_pair="${host_a//[^A-Za-z0-9_.-]/_}_${host_b//[^A-Za-z0-9_.-]/_}"
server_json="$benchmark_dir/verbs_app_${safe_pair}_server.json"
client_json="$benchmark_dir/verbs_app_${safe_pair}_client.json"
server_log="$log_dir/bench-verbs-app-${safe_pair}-server.log"
client_log="$log_dir/bench-verbs-app-${safe_pair}-client.log"

case "$image_copy_method" in
  spark-netcat|ssh-relay|none) ;;
  *)
    echo "GLMRT_SPARK_IMAGE_COPY_METHOD must be spark-netcat, ssh-relay, or none, got: $image_copy_method" >&2
    exit 2
    ;;
esac

for value_name in image_copy_port port duration; do
  value="${!value_name}"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || [ "$value" -lt 1 ] || [ "$value" -gt 65535 ]; then
    echo "$value_name must be an integer in 1..65535, got: $value" >&2
    exit 2
  fi
done

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "required command not found: $1" >&2
    exit 2
  fi
}

need rsync
need ssh

if [ -z "$remote_dir" ]; then
  remote_dir="$(
    ssh -o BatchMode=yes "$host_a" \
      'printf "%s/glmrt-phase0-verbs-app" "$HOME"'
  )"
fi

stage_repo() {
  local host="$1"
  echo "== staging repo on $host:$remote_dir =="
  ssh -o BatchMode=yes "$host" "mkdir -p '$remote_dir'"
  rsync -az --delete \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '.glmrt-cache/' \
    --exclude '.pytest_cache/' \
    --exclude '.ruff_cache/' \
    --exclude '__pycache__/' \
    --exclude 'rust/target/' \
    --exclude 'native/build/' \
    --exclude 'native/build-cuda/' \
    --exclude 'native/build-rdma/' \
    --exclude '.cargo-registry/' \
    --exclude '.cargo-git/' \
    --exclude 'reports/phase0_artifacts/benchmarks/' \
    --exclude 'reports/phase0_artifacts/logs/' \
    "$repo_root"/ "$host:$remote_dir"/
}

image_exists() {
  local host="$1"
  ssh -o BatchMode=yes "$host" "docker image inspect '$image' >/dev/null 2>&1"
}

select_image_seed() {
  local candidate
  if [ -n "$image_seed_host" ]; then
    image_exists "$image_seed_host" || {
      echo "GLMRT_SPARK_IMAGE_SEED_HOST does not have image '$image': $image_seed_host" >&2
      return 1
    }
    echo "$image_seed_host"
    return 0
  fi
  for candidate in "$host_a" "$host_b"; do
    if image_exists "$candidate"; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

wait_for_remote_listen() {
  local host="$1"
  local listen_port="$2"
  local deadline=$((SECONDS + 30))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if ssh -o BatchMode=yes "$host" "ss -ltn sport = :$listen_port 2>/dev/null | tail -n +2 | grep -q ." >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "remote listener on ${host}:${listen_port} did not become ready" >&2
  return 1
}

copy_image_spark_netcat() {
  local src="$1"
  local dest="$2"
  local dest_link="${dest}${image_link_suffix}"
  local state_dir="/tmp/glmrt-image-copy-${USER:-tj}-${dest}-${image_copy_port}"
  echo "== copying $image from $src to $dest over ${dest_link}:${image_copy_port} =="
  ssh -o BatchMode=yes "$dest" bash -s -- "$image_copy_port" "$state_dir" <<'REMOTE'
set -euo pipefail
listen_port="$1"
state_dir="$2"
rm -rf "$state_dir"
mkdir -p "$state_dir"
nohup bash -c '
set -o pipefail
listen_port="$1"
state_dir="$2"
if nc -l -p "$listen_port" | docker load >"$state_dir/docker-load.log" 2>&1; then
  touch "$state_dir/success"
else
  status=$?
  echo "$status" >"$state_dir/status"
  exit "$status"
fi
' bash "$listen_port" "$state_dir" >"$state_dir/listener.log" 2>&1 < /dev/null &
echo $! >"$state_dir/pid"
REMOTE
  wait_for_remote_listen "$dest" "$image_copy_port"
  if ! ssh -o BatchMode=yes "$src" bash -s -- "$image" "$dest_link" "$image_copy_port" <<'REMOTE'
set -euo pipefail
image="$1"
dest_link="$2"
dest_port="$3"
docker save "$image" | nc -N "$dest_link" "$dest_port"
REMOTE
  then
    ssh -o BatchMode=yes "$dest" "test -f '$state_dir/pid' && kill \"\$(cat '$state_dir/pid')\" >/dev/null 2>&1 || true" || true
    return 1
  fi
  ssh -o BatchMode=yes "$dest" bash -s -- "$state_dir" "$image" <<'REMOTE'
set -euo pipefail
state_dir="$1"
image="$2"
pid="$(cat "$state_dir/pid")"
deadline=$((SECONDS + 1800))
while kill -0 "$pid" >/dev/null 2>&1; do
  if [ "$SECONDS" -ge "$deadline" ]; then
    kill "$pid" >/dev/null 2>&1 || true
    echo "timed out waiting for docker load" >&2
    exit 1
  fi
  sleep 1
done
if [ -f "$state_dir/success" ] && docker image inspect "$image" >/dev/null 2>&1; then
  exit 0
fi
cat "$state_dir/listener.log" >&2 || true
cat "$state_dir/docker-load.log" >&2 || true
echo "docker load did not produce image $image" >&2
exit 1
REMOTE
}

copy_image_ssh_relay() {
  local src="$1"
  local dest="$2"
  echo "== copying $image from $src to $dest through local SSH relay =="
  ssh -o BatchMode=yes "$src" "docker save '$image'" | ssh -o BatchMode=yes "$dest" "docker load"
}

copy_image() {
  local src="$1"
  local dest="$2"
  case "$image_copy_method" in
    spark-netcat) copy_image_spark_netcat "$src" "$dest" ;;
    ssh-relay) copy_image_ssh_relay "$src" "$dest" ;;
    none) return 1 ;;
  esac
}

ensure_image() {
  local host="$1"
  if image_exists "$host"; then
    if [ -z "$image_seed_host" ]; then
      image_seed_host="$host"
    fi
    return
  fi
  if [ "$image_copy_method" != "none" ]; then
    if [ -z "$image_seed_host" ]; then
      if ! image_seed_host="$(select_image_seed)"; then
        if [ -n "${GLMRT_SPARK_IMAGE_SEED_HOST:-}" ]; then
          exit 2
        fi
        image_seed_host=""
      fi
    fi
    if [ -n "$image_seed_host" ] && [ "$image_seed_host" != "$host" ]; then
      copy_image "$image_seed_host" "$host"
      image_exists "$host" && return
      echo "copying Spark image '$image' to $host did not make the image available" >&2
      exit 1
    fi
  fi
  if [ "$build_missing_image" != "1" ]; then
    cat >&2 <<EOF
Spark image '$image' is missing on $host.
Seed it from another Spark, rerun with GLMRT_SPARK_IMAGE_COPY_METHOD=ssh-relay,
or explicitly rerun with GLMRT_SPARK_IMAGE_COPY_METHOD=none GLMRT_SPARK_BUILD_IMAGE=1.
EOF
    exit 2
  fi
  echo "== building $image on $host =="
  ssh -o BatchMode=yes "$host" bash -s -- "$remote_dir" "$image" <<'REMOTE'
set -euo pipefail
remote_dir="$1"
image="$2"
cd "$remote_dir"
docker build \
  --platform linux/arm64 \
  --build-arg GLMRT_ROLE=expert \
  --build-arg CUDA_ARCH=121 \
  --build-arg TARGET_PLATFORM=linux/arm64 \
  -f docker/Dockerfile.dev \
  -t "$image" .
REMOTE
  if [ -z "$image_seed_host" ]; then
    image_seed_host="$host"
  fi
}

build_remote() {
  local host="$1"
  echo "== building ARM glmrt and RDMA native library on $host =="
  ssh -o BatchMode=yes "$host" bash -s -- "$remote_dir" "$image" "$host" <<'REMOTE'
set -euo pipefail
remote_dir="$1"
image="$2"
host="$3"
mkdir -p "$remote_dir/.cargo-registry" "$remote_dir/.cargo-git"
docker_args=(
  run --rm
  --net=host
  --ipc=host
  --ulimit memlock=-1:-1
  --cap-add IPC_LOCK
  -v "$remote_dir:/workspace/glmrt"
  -v "$remote_dir/.cargo-registry:/opt/cargo/registry"
  -v "$remote_dir/.cargo-git:/opt/cargo/git"
  -w /workspace/glmrt
)
if [ -e /dev/infiniband ]; then
  docker_args+=(--device=/dev/infiniband)
fi
docker "${docker_args[@]}" \
  -e GLMRT_REMOTE_HOST="$host" \
  "$image" \
  bash -lc '
set -euo pipefail
cargo build --manifest-path rust/Cargo.toml -p glmrt-daemon
python3 python/tools/check_native_rdma_build.py \
  --clean \
  --build-dir native/build-rdma \
  --output "reports/phase0_artifacts/benchmarks/native_rdma_build_${GLMRT_REMOTE_HOST}.json" \
  --require-pass
'
REMOTE
}

run_remote_bench() {
  local host="$1"
  local mode="$2"
  local peer="$3"
  local output="$4"
  local unset_sentinel="__GLMRT_UNSET__"
  local roundtrip_iterations="${GLMRT_VERBS_APP_ROUNDTRIP_ITERATIONS:-$unset_sentinel}"
  local chain_iterations="${GLMRT_VERBS_APP_CHAIN_ITERATIONS:-$unset_sentinel}"
  local chain_hops="${GLMRT_VERBS_APP_CHAIN_HOPS:-75}"
  local control_timeout="${GLMRT_VERBS_APP_CONTROL_TIMEOUT_SECS:-$unset_sentinel}"
  local ib_port="${GLMRT_VERBS_APP_IB_PORT_NUM:-$unset_sentinel}"
  ssh -o BatchMode=yes "$host" bash -s -- \
    "$remote_dir" "$image" "$mode" "$peer" "$port" "$payloads" "$duration" \
    "$roundtrip_iterations" \
    "$chain_iterations" \
    "$chain_hops" \
    "$control_timeout" \
    "$ib_port" >"$output" <<'REMOTE'
set -euo pipefail
remote_dir="$1"
image="$2"
mode="$3"
peer="$4"
port="$5"
payloads="$6"
duration="$7"
shift 7
roundtrip_iterations="${1-}"
chain_iterations="${2-}"
chain_hops="${3-}"
control_timeout="${4-}"
ib_port="${5-}"
for name in roundtrip_iterations chain_iterations control_timeout ib_port; do
  if [ "${!name}" = "__GLMRT_UNSET__" ]; then
    printf -v "$name" ''
  fi
done
mkdir -p "$remote_dir/.cargo-registry" "$remote_dir/.cargo-git"
docker_args=(
  run --rm
  --net=host
  --ipc=host
  --ulimit memlock=-1:-1
  --cap-add IPC_LOCK
  -v "$remote_dir:/workspace/glmrt"
  -v "$remote_dir/.cargo-registry:/opt/cargo/registry"
  -v "$remote_dir/.cargo-git:/opt/cargo/git"
  -w /workspace/glmrt
)
if [ -e /dev/infiniband ]; then
  docker_args+=(--device=/dev/infiniband)
fi
docker "${docker_args[@]}" \
  -e GLMRT_NATIVE_LIB=/workspace/glmrt/native/build-rdma/libglmrt_native.so \
  -e GLMRT_VERBS_APP_ROUNDTRIP_ITERATIONS="$roundtrip_iterations" \
  -e GLMRT_VERBS_APP_CHAIN_ITERATIONS="$chain_iterations" \
  -e GLMRT_VERBS_APP_CHAIN_HOPS="$chain_hops" \
  -e GLMRT_VERBS_APP_CONTROL_TIMEOUT_SECS="$control_timeout" \
  -e GLMRT_VERBS_APP_IB_PORT_NUM="$ib_port" \
  -e GLMRT_BENCH_MODE="$mode" \
  -e GLMRT_BENCH_PEER="$peer" \
  -e GLMRT_BENCH_PORT="$port" \
  -e GLMRT_BENCH_DURATION="$duration" \
  -e GLMRT_BENCH_PAYLOADS="$payloads" \
  "$image" \
  bash -lc '
set -euo pipefail
exec rust/target/debug/glmrt bench-rdma \
  --mode "$GLMRT_BENCH_MODE" \
  --peer "$GLMRT_BENCH_PEER" \
  --port "$GLMRT_BENCH_PORT" \
  --duration-secs "$GLMRT_BENCH_DURATION" \
  --payload-bytes "$GLMRT_BENCH_PAYLOADS"
'
REMOTE
}

pull_remote_artifacts() {
  local host="$1"
  rsync -az "$host:$remote_dir/reports/phase0_artifacts/benchmarks/native_rdma_build_"*.json \
    "$benchmark_dir"/ >/dev/null 2>&1 || true
}

for host in "$host_a" "$host_b"; do
  stage_repo "$host"
  ensure_image "$host"
  build_remote "$host"
done

server_status=0
client_status=0
run_remote_bench "$host_b" app-server "$host_a" "$server_json" >"$server_log" 2>&1 &
server_pid=$!
sleep 3
run_remote_bench "$host_a" app-client "$host_b" "$client_json" >"$client_log" 2>&1 || client_status=$?
wait "$server_pid" || server_status=$?

cat "$server_log"
cat "$client_log"

pull_remote_artifacts "$host_a"
pull_remote_artifacts "$host_b"

if [[ "$client_status" != 0 || "$server_status" != 0 ]]; then
  echo "bench-verbs-app pair failed client_status=$client_status server_status=$server_status" >&2
  exit 1
fi

python3 - <<'PY' "$client_json" "$server_json"
import json
import sys
from pathlib import Path

for path_arg in sys.argv[1:]:
    path = Path(path_arg)
    data = json.loads(path.read_text(encoding="utf-8"))
    runs = data["glmrt_app_benchmark"]["app_transport_runs"]
    print(f"{path.name}: skipped={data['skipped']} runs={len(runs)} all_ok={all(run['ok'] for run in runs)}")
    for run in runs:
        if run["role"] == "client":
            print(
                f"  {run['run_kind']} rows={run['row_count']} hops={run['hops_per_iteration']} "
                f"avg_us={run['roundtrip_latency_micros_avg']} gbps={run['effective_payload_gbps']}"
            )
PY

if [ "$keep_remote" != "1" ]; then
  for host in "$host_a" "$host_b"; do
    ssh -o BatchMode=yes "$host" "rm -rf '$remote_dir/.tmp-verbs-app'" >/dev/null 2>&1 || true
  done
fi
