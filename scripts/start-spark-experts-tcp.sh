#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: scripts/start-spark-experts-tcp.sh [--hosts CSV] [--mode real|synthetic] [--catalog PATH] [--loadplan-dir DIR] [--dry-run]

Starts persistent Spark-hosted ProtocolV2 TCP expert daemons. Source/dev real
mode uses the native NVFP4 binary protocol path with CUDA, model catalog, and
per-host loadplans. Prebuilt release mode infers the catalog and placement from
the selected model and stable spark-0..3 role. The underlying Spark launcher
stages the repo and distributes the existing Spark image over the configured
Spark image copy path; it only rebuilds the image when
GLMRT_SPARK_BUILD_IMAGE=1 is set.

Environment:
  GLMRT_SPARK_HOSTS                 default: ostrich,dodo,emu,kiwi
  GLMRT_PHASE0_SPARK_EXPERT_MODE    real or synthetic; default: real
  GLMRT_SPARK_PREBUILT              use release artifacts and inferred placement; default: 0
  GLMRT_PHASE0_SPARK_CATALOG        default: .glmrt-cache/model-artifacts/diagnostic/model_catalog.json
  GLMRT_PHASE0_SPARK_LOADPLAN_DIR   default: .glmrt-cache/model-artifacts/diagnostic
  GLMRT_SPARK_EXPERT_REAL_LAYER     default: all for real start-only serving
  GLMRT_REAL_FULL_NVFP4_ROUTE_CUDA_GRAPHS
                                      forward to Spark expert containers; default: 1 in real mode
  GLMRT_B12X_SPARK_AOT               build direct SparkInfer AOT kernels; default: 1 in real mode
  GLMRT_B12X_SPARK_GROUPED_DECODE    grouped TP4 M=1/topk=8 decode; default: 1
  GLMRT_B12X_SPARK_W4A16_M1_FUSED_SUM
                                      atomic M=1 top-k accumulation; default: 1 in real mode
  GLMRT_REAL_FULL_NVFP4_ROUTE_CUDA_EVENT_TIMING
                                      forward to Spark expert containers; default: 0
  GLMRT_SPARK_GPU_RUNTIME           nvidia or manual; default: nvidia
  GLMRT_SPARK_EXPERT_TRANSPORT      tcp or verbs-host; default: tcp
  GLMRT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES
                                      concurrent RDMA/NCCL request lanes in 1..8; default: 2
  GLMRT_REAL_FULL_PROTOCOL_V2_PACKED_DIRECT_MAX_ROWS
                                      non-reduced packed expert rows in 8..16; default: 16
  GLMRT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP
                                      auto-discovered per Spark from RDMA netdev IPs
EOF
  exit 0
fi

hosts="${GLMRT_SPARK_HOSTS:-ostrich,dodo,emu,kiwi}"
mode="${GLMRT_PHASE0_SPARK_EXPERT_MODE:-real}"
catalog="${GLMRT_PHASE0_SPARK_CATALOG:-${CATALOG:-.glmrt-cache/model-artifacts/diagnostic/model_catalog.json}}"
loadplan_dir="${GLMRT_PHASE0_SPARK_LOADPLAN_DIR:-.glmrt-cache/model-artifacts/diagnostic}"
prebuilt="${GLMRT_SPARK_PREBUILT:-0}"
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hosts)
      hosts="${2:?--hosts requires a comma-separated host list}"
      shift 2
      ;;
    --mode)
      mode="${2:?--mode requires real or synthetic}"
      shift 2
      ;;
    --catalog)
      catalog="${2:?--catalog requires a path}"
      shift 2
      ;;
    --loadplan-dir)
      loadplan_dir="${2:?--loadplan-dir requires a path}"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

case "$mode" in
  real|synthetic) ;;
  *)
    echo "--mode must be real or synthetic, got: $mode" >&2
    exit 2
    ;;
esac

if [[ -z "$hosts" ]]; then
  echo "--hosts must not be empty" >&2
  exit 2
fi

if [[ "$mode" == "real" && "$prebuilt" != "1" ]]; then
  test -f "$catalog" || {
    echo "catalog not found: $catalog" >&2
    exit 2
  }
  IFS=',' read -r -a host_items <<< "$hosts"
  for host in "${host_items[@]}"; do
    host="$(echo "$host" | xargs)"
    [[ -n "$host" ]] || continue
    test -f "${loadplan_dir}/loadplan.${host}.json" || {
      echo "loadplan not found for $host: ${loadplan_dir}/loadplan.${host}.json" >&2
      exit 2
    }
  done
  export GLMRT_SPARK_EXPERT_REAL_LAYER="${GLMRT_SPARK_EXPERT_REAL_LAYER:-all}"
fi

export GLMRT_SPARK_HOSTS="$hosts"
export GLMRT_PHASE0_SPARK_EXPERT_MODE="$mode"
export GLMRT_PHASE0_SPARK_CATALOG="$catalog"
export GLMRT_PHASE0_SPARK_LOADPLAN_DIR="$loadplan_dir"
export GLMRT_SPARK_KEEP_EXPERTS="${GLMRT_SPARK_KEEP_EXPERTS:-1}"
export GLMRT_PHASE0_SPARK_SKIP_BENCH=1

if [[ "$dry_run" == "1" ]]; then
  printf 'GLMRT_SPARK_HOSTS=%s\n' "$GLMRT_SPARK_HOSTS"
  printf 'GLMRT_PHASE0_SPARK_EXPERT_MODE=%s\n' "$GLMRT_PHASE0_SPARK_EXPERT_MODE"
  printf 'GLMRT_SPARK_PREBUILT=%s\n' "$prebuilt"
  printf 'GLMRT_PHASE0_SPARK_CATALOG=%s\n' "$GLMRT_PHASE0_SPARK_CATALOG"
  printf 'GLMRT_PHASE0_SPARK_LOADPLAN_DIR=%s\n' "$GLMRT_PHASE0_SPARK_LOADPLAN_DIR"
  printf 'GLMRT_SPARK_KEEP_EXPERTS=%s\n' "$GLMRT_SPARK_KEEP_EXPERTS"
  printf 'GLMRT_PHASE0_SPARK_SKIP_BENCH=%s\n' "$GLMRT_PHASE0_SPARK_SKIP_BENCH"
  if [[ "${GLMRT_SPARK_EXPERT_REAL_LAYER+x}" ]]; then
    printf 'GLMRT_SPARK_EXPERT_REAL_LAYER=%s\n' "$GLMRT_SPARK_EXPERT_REAL_LAYER"
  fi
  if [[ "${GLMRT_REAL_FULL_NVFP4_ROUTE_CUDA_GRAPHS+x}" ]]; then
    printf 'GLMRT_REAL_FULL_NVFP4_ROUTE_CUDA_GRAPHS=%s\n' "$GLMRT_REAL_FULL_NVFP4_ROUTE_CUDA_GRAPHS"
  elif [[ "$mode" == "real" ]]; then
    printf 'GLMRT_REAL_FULL_NVFP4_ROUTE_CUDA_GRAPHS=1\n'
  else
    printf 'GLMRT_REAL_FULL_NVFP4_ROUTE_CUDA_GRAPHS=0\n'
  fi
  if [[ "${GLMRT_B12X_SPARK_AOT+x}" ]]; then
    printf 'GLMRT_B12X_SPARK_AOT=%s\n' "$GLMRT_B12X_SPARK_AOT"
  elif [[ "$mode" == "real" ]]; then
    printf 'GLMRT_B12X_SPARK_AOT=1\n'
  else
    printf 'GLMRT_B12X_SPARK_AOT=0\n'
  fi
  printf 'GLMRT_B12X_SPARK_GROUPED_DECODE=%s\n' "${GLMRT_B12X_SPARK_GROUPED_DECODE:-1}"
  if [[ "${GLMRT_SERVE_PROFILE+x}" ]]; then
    printf 'GLMRT_SERVE_PROFILE=%s\n' "$GLMRT_SERVE_PROFILE"
  fi
  if [[ "${GLMRT_B12X_SPARK_W4A16_M1_FUSED_SUM+x}" ]]; then
    printf 'GLMRT_B12X_SPARK_W4A16_M1_FUSED_SUM=%s\n' "$GLMRT_B12X_SPARK_W4A16_M1_FUSED_SUM"
  elif [[ "$mode" == "real" ]]; then
    printf 'GLMRT_B12X_SPARK_W4A16_M1_FUSED_SUM=1\n'
  else
    printf 'GLMRT_B12X_SPARK_W4A16_M1_FUSED_SUM=0\n'
  fi
  printf 'GLMRT_REAL_FULL_NVFP4_ROUTE_CUDA_EVENT_TIMING=%s\n' "${GLMRT_REAL_FULL_NVFP4_ROUTE_CUDA_EVENT_TIMING:-0}"
  printf 'GLMRT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES=%s\n' "${GLMRT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES:-2}"
  printf 'GLMRT_REAL_FULL_PROTOCOL_V2_PACKED_DIRECT_MAX_ROWS=%s\n' "${GLMRT_REAL_FULL_PROTOCOL_V2_PACKED_DIRECT_MAX_ROWS:-2064}"
  printf 'GLMRT_SPARK_GPU_RUNTIME=%s\n' "${GLMRT_SPARK_GPU_RUNTIME:-nvidia}"
  if [[ "${GLMRT_SPARK_EXPERT_TRANSPORT:-tcp}" == "verbs-host" ]]; then
    printf 'GLMRT_SPARK_EXPERT_TRANSPORT=%s\n' "$GLMRT_SPARK_EXPERT_TRANSPORT"
  elif [[ "${GLMRT_SPARK_EXPERT_TRANSPORT+x}" ]]; then
    printf 'GLMRT_SPARK_EXPERT_TRANSPORT=%s\n' "$GLMRT_SPARK_EXPERT_TRANSPORT"
  fi
  printf 'exec scripts/phase0-spark-tcp-bench.sh\n'
  exit 0
fi

exec scripts/phase0-spark-tcp-bench.sh
