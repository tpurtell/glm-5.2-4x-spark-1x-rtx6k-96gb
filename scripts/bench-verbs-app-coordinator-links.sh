#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: scripts/bench-verbs-app-coordinator-links.sh [HOSTS]

Benchmarks the coordinator-to-Spark verbs-host ProtocolV2 app transport.
Spark hosts run app-server remotely in the Spark container; the coordinator
runs app-client locally. This does not build Spark images locally.

Environment:
  GLMRT_EXPERT_HOSTS                         default: ostrich,dodo,emu,kiwi
  GLMRT_SPARK_IMAGE                          default: glmrt-dev:spark
  GLMRT_SPARK_BUILD_IMAGE                    set 1 to build missing image on each Spark
  GLMRT_VERBS_APP_WORKDIR                    default: $HOME/glmrt-phase0-verbs-app
  GLMRT_VERBS_APP_COORDINATOR_HOST           default: local short hostname
  GLMRT_VERBS_APP_COORDINATOR_LINK_SUFFIX    default: .200gb
  GLMRT_VERBS_APP_SPARK_LINK_SUFFIX          default: .200gb
  GLMRT_VERBS_APP_COORDINATOR_RUNS           single, fanout, or single,fanout; default: single,fanout
  GLMRT_VERBS_APP_PORT                       default: 18615
  GLMRT_VERBS_APP_DURATION_SECS              default: 2
  GLMRT_VERBS_APP_PAYLOAD_BYTES              default: GLM 1/2/4/8/16/64/256/512 BF16 rows
  GLMRT_NATIVE_LIB                           default: native/build-rdma/libglmrt_native.so
  GLMRT_VERBS_APP_KEEP_REMOTE                set 1 to keep staged remote workdirs
EOF
  exit 0
fi

hosts_csv="${1:-${GLMRT_EXPERT_HOSTS:-ostrich,dodo,emu,kiwi}}"
remote_dir="${GLMRT_VERBS_APP_WORKDIR:-}"
image="${GLMRT_SPARK_IMAGE:-glmrt-dev:spark}"
build_missing_image="${GLMRT_SPARK_BUILD_IMAGE:-0}"
coordinator_host="${GLMRT_VERBS_APP_COORDINATOR_HOST:-$(hostname -s)}"
coordinator_link_suffix="${GLMRT_VERBS_APP_COORDINATOR_LINK_SUFFIX:-.200gb}"
spark_link_suffix="${GLMRT_VERBS_APP_SPARK_LINK_SUFFIX:-.200gb}"
runs_csv="${GLMRT_VERBS_APP_COORDINATOR_RUNS:-single,fanout}"
base_port="${GLMRT_VERBS_APP_PORT:-18615}"
duration="${GLMRT_VERBS_APP_DURATION_SECS:-2}"
payloads="${GLMRT_VERBS_APP_PAYLOAD_BYTES:-12288,24576,49152,98304,196608,786432,3145728,6291456}"
keep_remote="${GLMRT_VERBS_APP_KEEP_REMOTE:-0}"
native_lib="${GLMRT_NATIVE_LIB:-$repo_root/native/build-rdma/libglmrt_native.so}"
benchmark_dir="$repo_root/reports/phase0_artifacts/benchmarks"
log_dir="$repo_root/reports/phase0_artifacts/logs"
mkdir -p "$benchmark_dir" "$log_dir"

IFS=',' read -r -a hosts_raw <<<"$hosts_csv"
hosts=()
for host in "${hosts_raw[@]}"; do
  host="$(printf '%s' "$host" | xargs)"
  [ -n "$host" ] && hosts+=("$host")
done
if [ "${#hosts[@]}" -eq 0 ]; then
  echo "no Spark hosts configured" >&2
  exit 2
fi

for value_name in base_port duration; do
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

safe_name() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9_.-' '_'
}

need rsync
need ssh
need python3

if [ -z "$remote_dir" ]; then
  remote_dir="$(
    ssh -o BatchMode=yes "${hosts[0]}" \
      'printf "%s/glmrt-phase0-verbs-app" "$HOME"'
  )"
fi

python_libs=()
if [ -n "${GLMRT_PYTHON_LIBDIR:-}" ]; then
  python_libs+=("$GLMRT_PYTHON_LIBDIR")
fi
python_sysconfig_lib="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"
if [ -n "$python_sysconfig_lib" ]; then
  python_libs+=("$python_sysconfig_lib")
fi
uv_python_lib="$HOME/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib"
if [ -d "$uv_python_lib" ]; then
  python_libs+=("$uv_python_lib")
fi
if [ "${#python_libs[@]}" -gt 0 ]; then
  python_path="$(IFS=:; printf '%s' "${python_libs[*]}")"
  export LD_LIBRARY_PATH="$python_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
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

ensure_image() {
  local host="$1"
  if image_exists "$host"; then
    return
  fi
  if [ "$build_missing_image" != "1" ]; then
    cat >&2 <<EOF
Spark image '$image' is missing on $host.
Build it on the Spark by rerunning with GLMRT_SPARK_BUILD_IMAGE=1.
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

ensure_local_native() {
  if [ -f "$native_lib" ]; then
    return
  fi
  echo "== building local RDMA native library at $native_lib =="
  python3 python/tools/check_native_rdma_build.py \
    --clean \
    --build-dir "$(dirname "$native_lib")" \
    --output "$benchmark_dir/native_rdma_build_${coordinator_host}.json" \
    --require-pass
}

wait_for_remote_listen() {
  local host="$1"
  local port="$2"
  local label="$3"
  local deadline=$((SECONDS + 60))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if ssh -o BatchMode=yes "$host" "ss -ltn sport = :$port 2>/dev/null | tail -n +2 | grep -q ." >/dev/null 2>&1; then
      return
    fi
    sleep 1
  done
  echo "$label did not start listening on ${host}:${port}" >&2
  return 1
}

run_remote_server() {
  local host="$1"
  local peer="$2"
  local port="$3"
  local output="$4"
  local log="$5"
  ssh -o BatchMode=yes "$host" bash -s -- \
    "$remote_dir" "$image" "$peer" "$port" "$payloads" "$duration" \
    "${GLMRT_VERBS_APP_ROUNDTRIP_ITERATIONS:-__GLMRT_UNSET__}" \
    "${GLMRT_VERBS_APP_CHAIN_ITERATIONS:-__GLMRT_UNSET__}" \
    "${GLMRT_VERBS_APP_CHAIN_HOPS:-75}" \
    "${GLMRT_VERBS_APP_CONTROL_TIMEOUT_SECS:-__GLMRT_UNSET__}" \
    "${GLMRT_VERBS_APP_IB_PORT_NUM:-__GLMRT_UNSET__}" >"$output" 2>"$log" <<'REMOTE'
set -euo pipefail
remote_dir="$1"
image="$2"
peer="$3"
port="$4"
payloads="$5"
duration="$6"
shift 6
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
  -e GLMRT_BENCH_PEER="$peer" \
  -e GLMRT_BENCH_PORT="$port" \
  -e GLMRT_BENCH_DURATION="$duration" \
  -e GLMRT_BENCH_PAYLOADS="$payloads" \
  "$image" \
  bash -lc '
set -euo pipefail
exec rust/target/debug/glmrt bench-rdma \
  --mode app-server \
  --peer "$GLMRT_BENCH_PEER" \
  --port "$GLMRT_BENCH_PORT" \
  --duration-secs "$GLMRT_BENCH_DURATION" \
  --payload-bytes "$GLMRT_BENCH_PAYLOADS"
'
REMOTE
}

run_local_client() {
  local peer="$1"
  local port="$2"
  local output="$3"
  local log="$4"
  GLMRT_NATIVE_LIB="$native_lib" \
  GLMRT_VERBS_APP_ROUNDTRIP_ITERATIONS="${GLMRT_VERBS_APP_ROUNDTRIP_ITERATIONS:-}" \
  GLMRT_VERBS_APP_CHAIN_ITERATIONS="${GLMRT_VERBS_APP_CHAIN_ITERATIONS:-}" \
  GLMRT_VERBS_APP_CHAIN_HOPS="${GLMRT_VERBS_APP_CHAIN_HOPS:-75}" \
  GLMRT_VERBS_APP_CONTROL_TIMEOUT_SECS="${GLMRT_VERBS_APP_CONTROL_TIMEOUT_SECS:-}" \
  GLMRT_VERBS_APP_IB_PORT_NUM="${GLMRT_VERBS_APP_IB_PORT_NUM:-}" \
  RUSTFLAGS="${RUSTFLAGS:--Awarnings}" \
  scripts/glmrt bench-rdma \
    --mode app-client \
    --peer "$peer" \
    --port "$port" \
    --duration-secs "$duration" \
    --payload-bytes "$payloads" >"$output" 2>"$log"
}

pull_remote_artifacts() {
  local host="$1"
  rsync -az "$host:$remote_dir/reports/phase0_artifacts/benchmarks/native_rdma_build_"*.json \
    "$benchmark_dir"/ >/dev/null 2>&1 || true
}

remote_cleanup() {
  local host="$1"
  local port="$2"
  ssh -o BatchMode=yes "$host" "pkill -f 'glmrt bench-rdma --mode app-server .* --port $port' >/dev/null 2>&1 || true" >/dev/null 2>&1 || true
}

run_single() {
  local host="$1"
  local peer="${host}${spark_link_suffix}"
  local safe_host
  safe_host="$(safe_name "$host")"
  local server_json="$benchmark_dir/verbs_app_${safe_host}_${coordinator_host}_server.json"
  local client_json="$benchmark_dir/verbs_app_${coordinator_host}_${safe_host}_client.json"
  local server_log="$log_dir/bench-verbs-app-${safe_host}-${coordinator_host}-server.log"
  local client_log="$log_dir/bench-verbs-app-${coordinator_host}-${safe_host}-client.log"
  local coordinator_peer="${coordinator_host}${coordinator_link_suffix}"
  local port="$base_port"

  echo "== coordinator verbs app single link ${coordinator_host}->${host} port=${port} =="
  run_remote_server "$host" "$coordinator_peer" "$port" "$server_json" "$server_log" &
  local server_pid=$!
  wait_for_remote_listen "$host" "$port" "verbs app server on $host" || {
    remote_cleanup "$host" "$port"
    wait "$server_pid" || true
    return 1
  }
  local client_status=0
  run_local_client "$peer" "$port" "$client_json" "$client_log" || client_status=$?
  local server_status=0
  wait "$server_pid" || server_status=$?
  if [[ "$client_status" != 0 || "$server_status" != 0 ]]; then
    echo "single link failed host=$host client_status=$client_status server_status=$server_status" >&2
    return 1
  fi
  python3 - <<'PY' "$client_json" "$server_json"
import json
import sys
from pathlib import Path

for path_arg in sys.argv[1:]:
    path = Path(path_arg)
    data = json.loads(path.read_text(encoding="utf-8"))
    runs = data["glmrt_app_benchmark"]["app_transport_runs"]
    all_ok = bool(runs) and all(run["ok"] for run in runs)
    print(f"{path.name}: skipped={data['skipped']} runs={len(runs)} all_ok={all_ok}")
    for run in runs:
        if run["role"] == "client":
            print(
                f"  {run['run_kind']} rows={run['row_count']} hops={run['hops_per_iteration']} "
                f"avg_us={run['roundtrip_latency_micros_avg']} gbps={run['effective_payload_gbps']}"
            )
    if data.get("skipped") or not all_ok:
        raise SystemExit(1)
PY
}

run_fanout() {
  local coordinator_peer="${coordinator_host}${coordinator_link_suffix}"
  local pids=()
  local server_hosts=()
  local server_ports=()
  local client_jsons=()
  local server_jsons=()
  local client_logs=()
  local server_logs=()

  echo "== coordinator verbs app fanout/gather hosts=${hosts[*]} base_port=${base_port} =="
  for idx in "${!hosts[@]}"; do
    local host="${hosts[$idx]}"
    local safe_host
    safe_host="$(safe_name "$host")"
    local port=$((base_port + idx + 1))
    local server_json="$benchmark_dir/verbs_app_${safe_host}_${coordinator_host}_fanout_server.json"
    local client_json="$benchmark_dir/verbs_app_${coordinator_host}_${safe_host}_fanout_client.json"
    local server_log="$log_dir/bench-verbs-app-${safe_host}-${coordinator_host}-fanout-server.log"
    local client_log="$log_dir/bench-verbs-app-${coordinator_host}-${safe_host}-fanout-client.log"
    run_remote_server "$host" "$coordinator_peer" "$port" "$server_json" "$server_log" &
    pids+=("$!")
    server_hosts+=("$host")
    server_ports+=("$port")
    client_jsons+=("$client_json")
    server_jsons+=("$server_json")
    client_logs+=("$client_log")
    server_logs+=("$server_log")
  done

  for idx in "${!server_hosts[@]}"; do
    wait_for_remote_listen "${server_hosts[$idx]}" "${server_ports[$idx]}" "fanout server ${server_hosts[$idx]}" || return 1
  done

  local started_ns
  started_ns="$(date +%s%N)"
  local client_pids=()
  for idx in "${!hosts[@]}"; do
    run_local_client \
      "${hosts[$idx]}${spark_link_suffix}" \
      "${server_ports[$idx]}" \
      "${client_jsons[$idx]}" \
      "${client_logs[$idx]}" &
    client_pids+=("$!")
  done

  local client_status=0
  for pid in "${client_pids[@]}"; do
    wait "$pid" || client_status=1
  done
  local server_status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || server_status=1
  done
  local ended_ns
  ended_ns="$(date +%s%N)"
  local summary_json="$benchmark_dir/verbs_app_${coordinator_host}_fanout_gather_summary.json"
  python3 - <<'PY' "$summary_json" "$started_ns" "$ended_ns" "${client_jsons[@]}" -- "${server_jsons[@]}"
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
started_ns = int(sys.argv[2])
ended_ns = int(sys.argv[3])
sep = sys.argv.index("--")
client_paths = [Path(value) for value in sys.argv[4:sep]]
server_paths = [Path(value) for value in sys.argv[sep + 1:]]

clients = []
for path in client_paths:
    data = json.loads(path.read_text(encoding="utf-8"))
    app = data.get("glmrt_app_benchmark") or {}
    runs = app.get("app_transport_runs") or []
    clients.append(
        {
            "artifact": path.name,
            "peer": data.get("peer"),
            "hostname": data.get("hostname"),
            "skipped": data.get("skipped"),
            "run_count": len(runs),
            "all_ok": bool(runs) and all(run.get("ok") for run in runs),
            "roundtrip_runs": sum(1 for run in runs if run.get("run_kind") == "roundtrip"),
            "chain_75hop_runs": sum(1 for run in runs if run.get("run_kind") == "chain_75hop"),
        }
    )

servers = []
for path in server_paths:
    data = json.loads(path.read_text(encoding="utf-8"))
    app = data.get("glmrt_app_benchmark") or {}
    runs = app.get("app_transport_runs") or []
    servers.append(
        {
            "artifact": path.name,
            "peer": data.get("peer"),
            "hostname": data.get("hostname"),
            "skipped": data.get("skipped"),
            "run_count": len(runs),
            "all_ok": bool(runs) and all(run.get("ok") for run in runs),
        }
    )

summary = {
    "benchmark": "spark_verbs_app_protocol_v2_fanout_gather",
    "coordinator": clients[0]["hostname"] if clients else None,
    "spark_count": len(clients),
    "elapsed_micros": (ended_ns - started_ns) // 1000,
    "all_clients_ok": bool(clients) and all(client["all_ok"] for client in clients),
    "all_servers_ok": bool(servers) and all(server["all_ok"] for server in servers),
    "clients": clients,
    "servers": servers,
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(f"{summary_path.name}: spark_count={summary['spark_count']} elapsed_micros={summary['elapsed_micros']} all_clients_ok={summary['all_clients_ok']} all_servers_ok={summary['all_servers_ok']}")
if not summary["all_clients_ok"] or not summary["all_servers_ok"]:
    raise SystemExit(1)
PY

  if [[ "$client_status" != 0 || "$server_status" != 0 ]]; then
    echo "fanout/gather failed client_status=$client_status server_status=$server_status" >&2
    return 1
  fi
}

for host in "${hosts[@]}"; do
  stage_repo "$host"
  ensure_image "$host"
  build_remote "$host"
done
ensure_local_native

case ",$runs_csv," in
  *,single,*) run_single "${hosts[0]}" ;;
esac
case ",$runs_csv," in
  *,fanout,*) run_fanout ;;
esac

for host in "${hosts[@]}"; do
  pull_remote_artifacts "$host"
done

if [ "$keep_remote" != "1" ]; then
  for host in "${hosts[@]}"; do
    ssh -o BatchMode=yes "$host" "rm -rf '$remote_dir/.tmp-verbs-app'" >/dev/null 2>&1 || true
  done
fi
