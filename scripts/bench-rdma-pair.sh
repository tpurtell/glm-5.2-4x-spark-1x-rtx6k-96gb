#!/usr/bin/env bash
set -euo pipefail

host_a="${1:?HOST_A required}"
host_b="${2:?HOST_B required}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
remote_repo="${GLMRT_REMOTE_REPO:-$repo_root}"
port="${GLMRT_RDMA_PORT:-18515}"
duration="${GLMRT_RDMA_DURATION_SECS:-2}"
payloads="${GLMRT_RDMA_PAYLOAD_BYTES:-4096,8192,12288,16384,32768,65536}"
log_dir="$repo_root/reports/phase0_artifacts/logs"
mkdir -p "$log_dir"

server_log="$log_dir/bench-rdma-${host_a}-${host_b}-server.log"
client_log="$log_dir/bench-rdma-${host_a}-${host_b}-client.log"

server_cmd=(
  "cd '$remote_repo' &&"
  "scripts/glmrt bench-rdma"
  "--mode server"
  "--port '$port'"
  "--duration-secs '$duration'"
  "--payload-bytes '$payloads'"
)
client_cmd=(
  "cd '$remote_repo' &&"
  "scripts/glmrt bench-rdma"
  "--mode client"
  "--peer '$host_b'"
  "--port '$port'"
  "--duration-secs '$duration'"
  "--payload-bytes '$payloads'"
)

ssh -o BatchMode=yes "$host_b" "${server_cmd[*]}" >"$server_log" 2>&1 &
server_pid=$!
sleep 3
client_status=0
ssh -o BatchMode=yes "$host_a" "${client_cmd[*]}" >"$client_log" 2>&1 || client_status=$?
wait "$server_pid" || server_status=$?
server_status="${server_status:-0}"

cat "$server_log"
cat "$client_log"

if [[ "$client_status" != 0 || "$server_status" != 0 ]]; then
  echo "bench-rdma pair failed client_status=$client_status server_status=$server_status" >&2
  exit 1
fi
