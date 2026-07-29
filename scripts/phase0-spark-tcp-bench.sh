#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: scripts/phase0-spark-tcp-bench.sh

Stages this repo on the Spark hosts, starts Spark-hosted binary ProtocolV2
glmrt expertd containers, then runs benchmarks/phase0_bench.py with explicit
Spark targets and an expected ProtocolV2 executor id.

Environment:
  GLMRT_SPARK_HOSTS                 default: ostrich,dodo,emu,kiwi
  GLMRT_PHASE0_SPARK_EXPERT_MODE    real or synthetic; default: real
  GLMRT_SPARK_IMAGE                 default: glmrt-dev:spark
  GLMRT_SPARK_IMAGE_COPY_METHOD     spark-netcat, ssh-relay, or none; default: spark-netcat
  GLMRT_SPARK_IMAGE_SEED_HOST       default: first host with GLMRT_SPARK_IMAGE
  GLMRT_SPARK_IMAGE_LINK_SUFFIX     default: .200gb for spark-netcat data path
  GLMRT_SPARK_IMAGE_COPY_PORT       default: 29420
  GLMRT_SPARK_BUILD_IMAGE           set 1 to build remotely when no seed image exists
  GLMRT_SPARK_FORCE_BUILD_IMAGE     set 1 to rebuild remotely even when image exists
  GLMRT_SPARK_IMAGE_ONLY            set 1 to stage/ensure images and exit before starting experts
  GLMRT_SPARK_BUILD_PROFILE         debug or release; default: release
  GLMRT_SPARK_PREBUILT              use /opt/glmrt release artifacts from the
                                      image instead of building mounted source
  GLMRT_SPARK_GPU_RUNTIME           nvidia or manual; default: nvidia
  GLMRT_SPARK_WORKDIR               default: $HOME/glmrt-phase0-spark-bench
  GLMRT_SPARK_SKIP_STAGE            set 1 when the committed source is already staged remotely
  GLMRT_SPARK_KEEP_EXPERTS          set 1 to leave remote containers running
  GLMRT_SPARK_EXPERT_PORT           default: 9100
  GLMRT_SPARK_EXPERT_TRANSPORT      tcp or verbs-host; default: tcp
  GLMRT_VERBS_APP_IB_PORT_NUM       optional verbs-host IB port override
  GLMRT_PHASE0_TCP_LAYER_ID         default: 3
  GLMRT_SPARK_EXPERT_REAL_LAYER     layer id, all, none, or empty; default: GLMRT_PHASE0_TCP_LAYER_ID
  GLMRT_MTP_BF16_EXPERTS           retain checkpoint BF16 layer-78 expert slabs; default: 0
  GLMRT_REAL_FULL_NVFP4_ROUTE_MANAGED_PROJECTIONS
                                      default: 1 for all-layer real daemon startup, else 0
  GLMRT_REAL_FULL_NVFP4_ROUTE_GROUPED_MULTIROW
                                      default: 1 for real daemon startup
  GLMRT_REAL_FULL_NVFP4_ROUTE_CUDA_GRAPHS
                                      forward to real daemon; default: 1 for real mode, else 0
  GLMRT_B12X_SPARK_AOT               build direct B12X AOT kernels; default: 1 for real mode
  GLMRT_B12X_SPARK_ROUTE_LANES       concurrent direct-B12X lanes in 1..4; default: 4
  GLMRT_B12X_SPARK_GROUPED_DECODE    grouped TP4 M=1/topk=8 decode; default: 1
  GLMRT_B12X_SPARK_W4A16_PACKED      single-layout packed W4A16 backend; default: 1
  GLMRT_SPARKINFER_SOURCE_W4A16      latest SparkInfer source-layout W4A16 backend; default: 0
  GLMRT_SPARKINFER_SOURCE_W4A16_AOT_BUILD
                                      build source-layout kernels; defaults to runtime setting
  GLMRT_SPARKINFER_SOURCE_W4A16_DIRECT_MAX_ROWS
                                      last physical M using direct kernels in 0..8; default: 8
  GLMRT_B12X_SPARK_W4A16_DEVICE_WEIGHTS
                                      cudaMalloc weight/scale slabs; default: 1
  GLMRT_B12X_SPARK_W4A16_DECODE_GRID_X process-start decode kernel grid override in 1..96; default: 32
  GLMRT_B12X_SPARK_W4A16_M1_FUSED_SUM  atomic top-k accumulation for M=1; default: 1 in real mode
  GLMRT_B12X_SPARK_W4A16_SMALL_M_MODE  split-m1, wide, or ordered for M=2..8; default: wide
  GLMRT_EXPERT_INTERMEDIATE_SHARDS   1 or 4; default: 1
  GLMRT_EXPERT_INTERMEDIATE_REDUCTION coordinator, spark, spark-owner, spark-hybrid, spark-rdma, or spark-rdma-hybrid; default: coordinator
  GLMRT_EXPERT_INTERMEDIATE_REDUCTION_DTYPE bf16, fp8, or nvfp4; default: fp8
  GLMRT_EXPERT_INTERMEDIATE_OWNER_REDUCTION_DTYPE
                                      small-row owner dtype; default: reduction dtype
  GLMRT_EXPERT_INTERMEDIATE_REDUCTION_ROOT default: first Spark on the image link
  GLMRT_EXPERT_INTERMEDIATE_REDUCTION_PORT default: 9200
  GLMRT_EXPERT_INTERMEDIATE_REDUCTION_MIN_ROWS default: 16 (spark/spark-hybrid) or 1 (spark-owner)
  GLMRT_EXPERT_INTERMEDIATE_ROW_SHARDED_REDUCTION
                                      partition rows across Spark ranks before reduction; default: 0
  GLMRT_EXPERT_FUSED_FP8_REDUCTION fuse full-row FP8 root combine; default: 1
  GLMRT_EXPERT_NCCL_BF16_REDUCE     experimental root-only BF16 reduce before FP8 response; default: 0
  GLMRT_EXPERT_INTERMEDIATE_OWNER_MAX_ROWS fused small-batch default: 8
  GLMRT_EXPERT_INTERMEDIATE_OWNER_PORT default: GLMRT_SPARK_EXPERT_PORT
  GLMRT_EXPERT_INTERMEDIATE_OWNER_PEERS rank-ordered host:port CSV; default: Spark image-link hosts
  GLMRT_EXPERT_INTERMEDIATE_RDMA_PEERS rank-ordered host CSV; default: Spark image-link hosts
  GLMRT_EXPERT_INTERMEDIATE_RDMA_ADDITIONAL_PEERS
                                      rank-ordered secondary-rail host/IP CSV; defaults to
                                      10.55.0.5,.6,.7,.8 for the four known Sparks
  GLMRT_EXPERT_INTERMEDIATE_RDMA_DEVICES
                                      local ibverbs devices by rail; default: rocep1s0f0,roceP2p1s0f0
  GLMRT_EXPERT_INTERMEDIATE_RDMA_PORT  base for pair/rail ports; default: 9400
  GLMRT_EXPERT_INTERMEDIATE_RDMA_SLOT_BYTES mapped ring slot capacity; default: 4194304
  GLMRT_EXPERT_INTERMEDIATE_RDMA_RING_DEPTH mapped ring slots per peer; default: 4
  GLMRT_EXPERT_INTERMEDIATE_RDMA_STRIPE_MIN_BYTES
                                      minimum per-peer payload to split; default: 262144
  GLMRT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES
                                      concurrent RDMA/NCCL request lanes in 1..8; default: 4
  GLMRT_REAL_FULL_PROTOCOL_V2_PACKED_DIRECT_MAX_ROWS
                                      full-top8 requests bypassing the general CPU
                                      row/group planner, in 8..2048; default: 2048
  GLMRT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP
                                      generated per Spark from RDMA netdev IPs; maps each
                                      accepted control-plane destination IP to its ibverbs device
  GLMRT_SPARK_TRANSFORMER_TP          enable all-rank transformer TP residency; default: 0
  GLMRT_SPARK_TRANSFORMER_TP_RANGE    start:end layer range; default: 0:78
  GLMRT_SPARK_TRANSFORMER_TP_ROOT     NCCL bootstrap root; default: first Spark
  GLMRT_SPARK_TRANSFORMER_TP_PORT     NCCL bootstrap port; default: 9300
  GLMRT_SPARK_TRANSFORMER_TP_COLLECTIVE_PROBE_ITERS optional startup BF16 all-reduce iterations; default: 0
  GLMRT_SPARK_NCCL_SOCKET_IFNAME     default: enp1s0f0np0
  GLMRT_SPARK_NCCL_IB_HCA            default: =rocep1s0f0,roceP2p1s0f0
  GLMRT_SPARK_NCCL_CROSS_NIC         0, 1, or topology-aware 2; default: 2
  GLMRT_SPARK_NCCL_NETDEVS_POLICY    AUTO, ALL, or MAX:N; default: ALL
  GLMRT_SPARK_NCCL_IB_MERGE_NICS     0 or 1; default: 1
  GLMRT_SPARK_NCCL_P2P_NET_CHUNKSIZE grouped send/recv chunk bytes; default: 131072
  GLMRT_SPARK_NCCL_DEBUG             default: WARN
  GLMRT_SPARK_NCCL_LAUNCH_ORDER_IMPLICIT
                                      order concurrent communicators; default: 1
  GLMRT_REAL_FULL_NVFP4_ROUTE_CUDA_EVENT_TIMING
                                      forward to real daemon; default: 0
  GLMRT_REAL_FULL_NVFP4_ROUTE_TIMING
                                      forward to real daemon; default: 0
  GLMRT_REAL_FULL_CUDA_ROUTE_VALIDATE forward CPU route validation; default: 0
  GLMRT_SPARK_SYNC_MODEL_CACHE       set 1 to copy the selected local Hugging
                                      Face model cache to missing Spark hosts;
                                      default: 0
  GLMRT_SPARK_MODEL_SYNC_ONLY        set 1 to stop after model-cache sync
  GLMRT_PHASE0_SPARK_SKIP_BENCH     set 1 to start/wait for experts without running the benchmark
EOF
  exit 0
fi

hosts_csv="${GLMRT_SPARK_HOSTS:-ostrich,dodo,emu,kiwi}"
model_id="${GLMRT_MODEL_ID:-lukealonso/GLM-5.2-NVFP4}"
sync_model_cache="${GLMRT_SPARK_SYNC_MODEL_CACHE:-0}"
model_sync_only="${GLMRT_SPARK_MODEL_SYNC_ONLY:-0}"
mode="${GLMRT_PHASE0_SPARK_EXPERT_MODE:-real}"
remote_dir="${GLMRT_SPARK_WORKDIR:-}"
skip_stage="${GLMRT_SPARK_SKIP_STAGE:-0}"
image="${GLMRT_SPARK_IMAGE:-glmrt-dev:spark}"
image_copy_method="${GLMRT_SPARK_IMAGE_COPY_METHOD:-spark-netcat}"
image_seed_host="${GLMRT_SPARK_IMAGE_SEED_HOST:-}"
image_link_suffix="${GLMRT_SPARK_IMAGE_LINK_SUFFIX:-.200gb}"
image_copy_port="${GLMRT_SPARK_IMAGE_COPY_PORT:-29420}"
build_missing_image="${GLMRT_SPARK_BUILD_IMAGE:-0}"
force_build_image="${GLMRT_SPARK_FORCE_BUILD_IMAGE:-0}"
image_only="${GLMRT_SPARK_IMAGE_ONLY:-0}"
build_profile="${GLMRT_SPARK_BUILD_PROFILE:-release}"
prebuilt="${GLMRT_SPARK_PREBUILT:-0}"
gpu_runtime="${GLMRT_SPARK_GPU_RUNTIME:-nvidia}"
port="${GLMRT_SPARK_EXPERT_PORT:-9100}"
expert_transport="${GLMRT_SPARK_EXPERT_TRANSPORT:-tcp}"
verbs_ib_port_num="${GLMRT_VERBS_APP_IB_PORT_NUM:-}"
if [ "$prebuilt" = "1" ]; then
  # Production resolves tensor metadata directly from the selected HF snapshot.
  # An explicit catalog remains available for diagnostic/repro benchmark runs.
  catalog="${GLMRT_PHASE0_SPARK_CATALOG:-}"
else
  catalog="${GLMRT_PHASE0_SPARK_CATALOG:-.glmrt-cache/model-artifacts/diagnostic/model_catalog.json}"
fi
loadplan_dir="${GLMRT_PHASE0_SPARK_LOADPLAN_DIR:-.glmrt-cache/model-artifacts/diagnostic}"
layer_id="${GLMRT_PHASE0_TCP_LAYER_ID:-3}"
expert_real_layer="${GLMRT_SPARK_EXPERT_REAL_LAYER:-$layer_id}"
mtp_bf16_experts="${GLMRT_MTP_BF16_EXPERTS:-0}"
keep_experts="${GLMRT_SPARK_KEEP_EXPERTS:-0}"
skip_bench="${GLMRT_PHASE0_SPARK_SKIP_BENCH:-0}"
if [ "${GLMRT_REAL_FULL_NVFP4_ROUTE_MANAGED_PROJECTIONS+x}" ]; then
  managed_route_projections="$GLMRT_REAL_FULL_NVFP4_ROUTE_MANAGED_PROJECTIONS"
else
  case "$expert_real_layer" in
    ""|all|none) managed_route_projections=1 ;;
    *) managed_route_projections=0 ;;
  esac
fi
if [ "${GLMRT_REAL_FULL_NVFP4_ROUTE_GROUPED_MULTIROW+x}" ]; then
  grouped_multirow="$GLMRT_REAL_FULL_NVFP4_ROUTE_GROUPED_MULTIROW"
elif [ "$mode" = "real" ]; then
  grouped_multirow=1
else
  grouped_multirow=0
fi
if [ "${GLMRT_REAL_FULL_NVFP4_ROUTE_CUDA_GRAPHS+x}" ]; then
  route_cuda_graphs="$GLMRT_REAL_FULL_NVFP4_ROUTE_CUDA_GRAPHS"
elif [ "$mode" = "real" ]; then
  route_cuda_graphs=1
else
  route_cuda_graphs=0
fi
if [ "${GLMRT_B12X_SPARK_AOT+x}" ]; then
  b12x_spark_aot="$GLMRT_B12X_SPARK_AOT"
elif [ "$mode" = "real" ]; then
  b12x_spark_aot=1
else
  b12x_spark_aot=0
fi
route_timing="${GLMRT_REAL_FULL_NVFP4_ROUTE_TIMING:-0}"
route_cuda_event_timing="${GLMRT_REAL_FULL_NVFP4_ROUTE_CUDA_EVENT_TIMING:-0}"
route_validate="${GLMRT_REAL_FULL_CUDA_ROUTE_VALIDATE:-0}"
protocol_v2_verbs_host_execution_lanes="${GLMRT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES:-4}"
protocol_v2_packed_direct_max_rows="${GLMRT_REAL_FULL_PROTOCOL_V2_PACKED_DIRECT_MAX_ROWS:-2048}"
route_preload_io_workers="${GLMRT_REAL_FULL_NVFP4_ROUTE_PRELOAD_IO_WORKERS:-128}"
route_preload_cooperative="${GLMRT_REAL_FULL_NVFP4_ROUTE_PRELOAD_COOPERATIVE:-1}"
weight_preload_nccl_port="${GLMRT_EXPERT_WEIGHT_PRELOAD_NCCL_PORT:-9350}"
b12x_route_lanes="${GLMRT_B12X_SPARK_ROUTE_LANES:-4}"
b12x_grouped_decode="${GLMRT_B12X_SPARK_GROUPED_DECODE:-1}"
b12x_w4a16_packed="${GLMRT_B12X_SPARK_W4A16_PACKED:-1}"
sparkinfer_source_w4a16="${GLMRT_SPARKINFER_SOURCE_W4A16:-0}"
sparkinfer_source_w4a16_aot_build="${GLMRT_SPARKINFER_SOURCE_W4A16_AOT_BUILD:-$sparkinfer_source_w4a16}"
sparkinfer_source_w4a16_direct_max_rows="${GLMRT_SPARKINFER_SOURCE_W4A16_DIRECT_MAX_ROWS:-2}"
b12x_w4a16_device_weights="${GLMRT_B12X_SPARK_W4A16_DEVICE_WEIGHTS:-1}"
b12x_w4a16_decode_grid_x="${GLMRT_B12X_SPARK_W4A16_DECODE_GRID_X:-}"
b12x_w4a16_small_m_mode="${GLMRT_B12X_SPARK_W4A16_SMALL_M_MODE:-wide}"
if [ "${GLMRT_B12X_SPARK_W4A16_M1_FUSED_SUM+x}" ]; then
  b12x_w4a16_m1_fused_sum="$GLMRT_B12X_SPARK_W4A16_M1_FUSED_SUM"
elif [ "$mode" = "real" ]; then
  b12x_w4a16_m1_fused_sum=1
else
  b12x_w4a16_m1_fused_sum=0
fi
intermediate_shards="${GLMRT_EXPERT_INTERMEDIATE_SHARDS:-1}"
intermediate_reduction="${GLMRT_EXPERT_INTERMEDIATE_REDUCTION:-coordinator}"
intermediate_reduction_dtype="${GLMRT_EXPERT_INTERMEDIATE_REDUCTION_DTYPE:-fp8}"
intermediate_owner_reduction_dtype="${GLMRT_EXPERT_INTERMEDIATE_OWNER_REDUCTION_DTYPE:-$intermediate_reduction_dtype}"
intermediate_reduction_port="${GLMRT_EXPERT_INTERMEDIATE_REDUCTION_PORT:-9200}"
if [ "${GLMRT_EXPERT_INTERMEDIATE_ROW_SHARDED_REDUCTION+x}" ]; then
  intermediate_row_sharded_reduction="$GLMRT_EXPERT_INTERMEDIATE_ROW_SHARDED_REDUCTION"
else
  case "$intermediate_reduction" in
    spark-rdma|spark-rdma-hybrid) intermediate_row_sharded_reduction=1 ;;
    *) intermediate_row_sharded_reduction=0 ;;
  esac
fi
fused_fp8_reduction="${GLMRT_EXPERT_FUSED_FP8_REDUCTION:-1}"
nccl_bf16_reduce="${GLMRT_EXPERT_NCCL_BF16_REDUCE:-0}"
if [ "${GLMRT_EXPERT_INTERMEDIATE_REDUCTION_MIN_ROWS+x}" ]; then
  intermediate_reduction_min_rows="$GLMRT_EXPERT_INTERMEDIATE_REDUCTION_MIN_ROWS"
elif [ "$intermediate_reduction" = "spark-owner" ]; then
  intermediate_reduction_min_rows=1
else
  intermediate_reduction_min_rows=16
fi
intermediate_owner_max_rows="${GLMRT_EXPERT_INTERMEDIATE_OWNER_MAX_ROWS:-8}"
intermediate_owner_port="${GLMRT_EXPERT_INTERMEDIATE_OWNER_PORT:-$port}"
intermediate_rdma_port="${GLMRT_EXPERT_INTERMEDIATE_RDMA_PORT:-9400}"
intermediate_rdma_slot_bytes="${GLMRT_EXPERT_INTERMEDIATE_RDMA_SLOT_BYTES:-4194304}"
intermediate_rdma_ring_depth="${GLMRT_EXPERT_INTERMEDIATE_RDMA_RING_DEPTH:-4}"
intermediate_rdma_additional_peers="${GLMRT_EXPERT_INTERMEDIATE_RDMA_ADDITIONAL_PEERS:-}"
intermediate_rdma_devices="${GLMRT_EXPERT_INTERMEDIATE_RDMA_DEVICES:-}"
intermediate_rdma_stripe_min_bytes="${GLMRT_EXPERT_INTERMEDIATE_RDMA_STRIPE_MIN_BYTES:-262144}"
spark_layer_blocks="${GLMRT_SPARK_LAYER_BLOCKS:-0}"
spark_layer_block_kv_dtype="${GLMRT_SPARK_LAYER_BLOCK_KV_DTYPE:-${GLMRT_KV_CACHE_DTYPE:-fp8}}"
spark_transformer_tp="${GLMRT_SPARK_TRANSFORMER_TP:-0}"
spark_transformer_tp_range="${GLMRT_SPARK_TRANSFORMER_TP_RANGE:-0:78}"
spark_transformer_tp_root="${GLMRT_SPARK_TRANSFORMER_TP_ROOT:-}"
spark_transformer_tp_port="${GLMRT_SPARK_TRANSFORMER_TP_PORT:-9300}"
spark_transformer_tp_collective_probe_iters="${GLMRT_SPARK_TRANSFORMER_TP_COLLECTIVE_PROBE_ITERS:-0}"
nccl_socket_ifname="${GLMRT_SPARK_NCCL_SOCKET_IFNAME:-enp1s0f0np0}"
nccl_ib_hca="${GLMRT_SPARK_NCCL_IB_HCA:-=rocep1s0f0,roceP2p1s0f0}"
nccl_cross_nic="${GLMRT_SPARK_NCCL_CROSS_NIC:-2}"
nccl_netdevs_policy="${GLMRT_SPARK_NCCL_NETDEVS_POLICY:-ALL}"
nccl_ib_merge_nics="${GLMRT_SPARK_NCCL_IB_MERGE_NICS:-1}"
nccl_p2p_net_chunksize="${GLMRT_SPARK_NCCL_P2P_NET_CHUNKSIZE:-131072}"
nccl_debug="${GLMRT_SPARK_NCCL_DEBUG:-WARN}"
nccl_launch_order_implicit="${GLMRT_SPARK_NCCL_LAUNCH_ORDER_IMPLICIT:-1}"
container_prefix="${GLMRT_SPARK_CONTAINER_PREFIX:-glmrt-phase0-tcp-expertd}"

case "$mode" in
  real|synthetic) ;;
  *)
    echo "GLMRT_PHASE0_SPARK_EXPERT_MODE must be real or synthetic, got: $mode" >&2
    exit 2
    ;;
esac
case "$nccl_cross_nic" in
  0|1|2) ;;
  *)
    echo "GLMRT_SPARK_NCCL_CROSS_NIC must be 0, 1, or 2, got: $nccl_cross_nic" >&2
    exit 2
    ;;
esac
if ! [[ "$nccl_netdevs_policy" = "AUTO" || "$nccl_netdevs_policy" = "ALL" || "$nccl_netdevs_policy" =~ ^MAX:[1-9][0-9]*$ ]]; then
  echo "GLMRT_SPARK_NCCL_NETDEVS_POLICY must be AUTO, ALL, or MAX:N, got: $nccl_netdevs_policy" >&2
  exit 2
fi
case "$nccl_ib_merge_nics" in
  0|1) ;;
  *)
    echo "GLMRT_SPARK_NCCL_IB_MERGE_NICS must be 0 or 1, got: $nccl_ib_merge_nics" >&2
    exit 2
    ;;
esac
if ! [[ "$nccl_p2p_net_chunksize" =~ ^[1-9][0-9]*$ ]]; then
  echo "GLMRT_SPARK_NCCL_P2P_NET_CHUNKSIZE must be a positive integer, got: $nccl_p2p_net_chunksize" >&2
  exit 2
fi

if [ "$intermediate_shards" != "1" ] && [ "$intermediate_shards" != "4" ]; then
  echo "GLMRT_EXPERT_INTERMEDIATE_SHARDS must be 1 or 4, got: $intermediate_shards" >&2
  exit 2
fi
if ! [[ "$sparkinfer_source_w4a16_direct_max_rows" =~ ^[0-8]$ ]]; then
  echo "GLMRT_SPARKINFER_SOURCE_W4A16_DIRECT_MAX_ROWS must be in 0..8, got: $sparkinfer_source_w4a16_direct_max_rows" >&2
  exit 2
fi
if [ "$spark_transformer_tp" = "1" ]; then
  if [ "$mode" != "real" ] || [ "$intermediate_shards" != "4" ]; then
    echo "GLMRT_SPARK_TRANSFORMER_TP=1 requires real mode with four intermediate shards" >&2
    exit 2
  fi
  if [ "$spark_layer_blocks" = "1" ]; then
    echo "GLMRT_SPARK_TRANSFORMER_TP and GLMRT_SPARK_LAYER_BLOCKS cannot both be enabled" >&2
    exit 2
  fi
fi
case "$intermediate_reduction" in
  coordinator) ;;
  spark|spark-owner|spark-hybrid|spark-rdma|spark-rdma-hybrid)
    if [ "$mode" != "real" ] || [ "$intermediate_shards" != "4" ]; then
      echo "GLMRT_EXPERT_INTERMEDIATE_REDUCTION=$intermediate_reduction requires real mode with four intermediate shards" >&2
      exit 2
    fi
    if { [ "$intermediate_reduction" = "spark-owner" ] || [ "$intermediate_reduction" = "spark-hybrid" ] || [ "$intermediate_reduction" = "spark-rdma" ] || [ "$intermediate_reduction" = "spark-rdma-hybrid" ]; } && [ "$expert_transport" != "verbs-host" ]; then
      echo "GLMRT_EXPERT_INTERMEDIATE_REDUCTION=$intermediate_reduction requires GLMRT_SPARK_EXPERT_TRANSPORT=verbs-host" >&2
      exit 2
    fi
    if { [ "$intermediate_reduction" = "spark-rdma" ] || [ "$intermediate_reduction" = "spark-rdma-hybrid" ]; } && [ "$intermediate_row_sharded_reduction" != "1" ]; then
      echo "GLMRT_EXPERT_INTERMEDIATE_REDUCTION=$intermediate_reduction requires GLMRT_EXPERT_INTERMEDIATE_ROW_SHARDED_REDUCTION=1" >&2
      exit 2
    fi
    ;;
  *)
    echo "GLMRT_EXPERT_INTERMEDIATE_REDUCTION must be coordinator, spark, spark-owner, spark-hybrid, spark-rdma, or spark-rdma-hybrid, got: $intermediate_reduction" >&2
    exit 2
    ;;
esac
case "${spark_layer_blocks,,}" in
  0|false|no|off) spark_layer_blocks=0 ;;
  1|true|yes|on)
    spark_layer_blocks=1
    if [ "$mode" != "real" ] || [ "$intermediate_shards" != "4" ]; then
      echo "GLMRT_SPARK_LAYER_BLOCKS=1 requires real mode with four intermediate shards" >&2
      exit 2
    fi
    ;;
  *)
    echo "GLMRT_SPARK_LAYER_BLOCKS must be boolean-like, got: $spark_layer_blocks" >&2
    exit 2
    ;;
esac
case "$intermediate_reduction_dtype" in
  bf16|fp8|nvfp4) ;;
  *)
    echo "GLMRT_EXPERT_INTERMEDIATE_REDUCTION_DTYPE must be bf16, fp8, or nvfp4, got: $intermediate_reduction_dtype" >&2
    exit 2
    ;;
esac
case "$intermediate_owner_reduction_dtype" in
  bf16|fp8|nvfp4) ;;
  *)
    echo "GLMRT_EXPERT_INTERMEDIATE_OWNER_REDUCTION_DTYPE must be bf16, fp8, or nvfp4, got: $intermediate_owner_reduction_dtype" >&2
    exit 2
    ;;
esac
if ! [[ "$intermediate_reduction_port" =~ ^[0-9]+$ ]] || [ "$intermediate_reduction_port" -lt 1 ] || [ "$intermediate_reduction_port" -gt 65535 ]; then
  echo "GLMRT_EXPERT_INTERMEDIATE_REDUCTION_PORT must be an integer in 1..65535" >&2
  exit 2
fi
if ! [[ "$intermediate_reduction_min_rows" =~ ^[0-9]+$ ]] || [ "$intermediate_reduction_min_rows" -lt 1 ]; then
  echo "GLMRT_EXPERT_INTERMEDIATE_REDUCTION_MIN_ROWS must be positive" >&2
  exit 2
fi
if [ "$intermediate_row_sharded_reduction" != "0" ] && [ "$intermediate_row_sharded_reduction" != "1" ]; then
  echo "GLMRT_EXPERT_INTERMEDIATE_ROW_SHARDED_REDUCTION must be 0 or 1" >&2
  exit 2
fi
if ! [[ "$intermediate_owner_max_rows" =~ ^[0-9]+$ ]] || [ "$intermediate_owner_max_rows" -lt 1 ]; then
  echo "GLMRT_EXPERT_INTERMEDIATE_OWNER_MAX_ROWS must be positive" >&2
  exit 2
fi
if ! [[ "$intermediate_owner_port" =~ ^[0-9]+$ ]] || [ "$intermediate_owner_port" -lt 1 ] || [ "$intermediate_owner_port" -gt 65535 ]; then
  echo "GLMRT_EXPERT_INTERMEDIATE_OWNER_PORT must be an integer in 1..65535" >&2
  exit 2
fi
if ! [[ "$intermediate_rdma_port" =~ ^[0-9]+$ ]] || [ "$intermediate_rdma_port" -lt 1 ] || [ "$intermediate_rdma_port" -gt 65535 ]; then
  echo "GLMRT_EXPERT_INTERMEDIATE_RDMA_PORT must be an integer in 1..65535" >&2
  exit 2
fi
if ! [[ "$intermediate_rdma_slot_bytes" =~ ^[0-9]+$ ]] || [ "$intermediate_rdma_slot_bytes" -lt 1 ]; then
  echo "GLMRT_EXPERT_INTERMEDIATE_RDMA_SLOT_BYTES must be positive" >&2
  exit 2
fi
if ! [[ "$intermediate_rdma_ring_depth" =~ ^[0-9]+$ ]] || [ "$intermediate_rdma_ring_depth" -lt 1 ]; then
  echo "GLMRT_EXPERT_INTERMEDIATE_RDMA_RING_DEPTH must be positive" >&2
  exit 2
fi
if ! [[ "$intermediate_rdma_stripe_min_bytes" =~ ^[0-9]+$ ]] || [ "$intermediate_rdma_stripe_min_bytes" -lt 1 ]; then
  echo "GLMRT_EXPERT_INTERMEDIATE_RDMA_STRIPE_MIN_BYTES must be positive" >&2
  exit 2
fi

case "$image_copy_method" in
  spark-netcat|ssh-relay|none) ;;
  *)
    echo "GLMRT_SPARK_IMAGE_COPY_METHOD must be spark-netcat, ssh-relay, or none, got: $image_copy_method" >&2
    exit 2
    ;;
esac

if ! [[ "$image_copy_port" =~ ^[0-9]+$ ]] || [ "$image_copy_port" -lt 1 ] || [ "$image_copy_port" -gt 65535 ]; then
  echo "GLMRT_SPARK_IMAGE_COPY_PORT must be an integer in 1..65535" >&2
  exit 2
fi

if ! [[ "$port" =~ ^[0-9]+$ ]] || [ "$port" -lt 1 ] || [ "$port" -gt 65535 ]; then
  echo "GLMRT_SPARK_EXPERT_PORT must be an integer in 1..65535" >&2
  exit 2
fi

case "$skip_bench" in
  0|1) ;;
  *)
    echo "GLMRT_PHASE0_SPARK_SKIP_BENCH must be 0 or 1, got: $skip_bench" >&2
    exit 2
    ;;
esac
case "$mtp_bf16_experts" in
  0|1) ;;
  *)
    echo "GLMRT_MTP_BF16_EXPERTS must be 0 or 1, got: $mtp_bf16_experts" >&2
    exit 2
    ;;
esac
case "$force_build_image" in
  0|1) ;;
  *)
    echo "GLMRT_SPARK_FORCE_BUILD_IMAGE must be 0 or 1, got: $force_build_image" >&2
    exit 2
    ;;
esac
case "$image_only" in
  0|1) ;;
  *)
    echo "GLMRT_SPARK_IMAGE_ONLY must be 0 or 1, got: $image_only" >&2
    exit 2
    ;;
esac

case "$build_profile" in
  debug|release) ;;
  *)
    echo "GLMRT_SPARK_BUILD_PROFILE must be debug or release, got: $build_profile" >&2
    exit 2
    ;;
esac
case "$prebuilt" in
  0|1) ;;
  *)
    echo "GLMRT_SPARK_PREBUILT must be 0 or 1, got: $prebuilt" >&2
    exit 2
    ;;
esac
case "$gpu_runtime" in
  nvidia|manual) ;;
  *)
    echo "GLMRT_SPARK_GPU_RUNTIME must be nvidia or manual, got: $gpu_runtime" >&2
    exit 2
    ;;
esac
case "$expert_transport" in
  tcp|verbs-host) ;;
  *)
    echo "GLMRT_SPARK_EXPERT_TRANSPORT must be tcp or verbs-host, got: $expert_transport" >&2
    exit 2
    ;;
esac

case "$expert_real_layer" in
  ""|all|none) ;;
  *)
    if ! [[ "$expert_real_layer" =~ ^[0-9]+$ ]]; then
      echo "GLMRT_SPARK_EXPERT_REAL_LAYER must be a non-negative integer, all, none, or empty; got: $expert_real_layer" >&2
      exit 2
    fi
    ;;
esac

case "${managed_route_projections,,}" in
  ""|0|1|true|false|yes|no|managed|uma) ;;
  *)
    echo "GLMRT_REAL_FULL_NVFP4_ROUTE_MANAGED_PROJECTIONS must be boolean-like, managed, uma, or empty; got: $managed_route_projections" >&2
    exit 2
    ;;
esac
case "${grouped_multirow,,}" in
  0|1|true|false|yes|no|on|off) ;;
  *)
    echo "GLMRT_REAL_FULL_NVFP4_ROUTE_GROUPED_MULTIROW must be boolean-like, got: $grouped_multirow" >&2
    exit 2
    ;;
esac
case "${route_cuda_graphs,,}" in
  0|1|true|false|yes|no|on|off) ;;
  *)
    echo "GLMRT_REAL_FULL_NVFP4_ROUTE_CUDA_GRAPHS must be boolean-like, got: $route_cuda_graphs" >&2
    exit 2
    ;;
esac
case "${b12x_spark_aot,,}" in
  0|1|true|false|yes|no|on|off) ;;
  *)
    echo "GLMRT_B12X_SPARK_AOT must be boolean-like, got: $b12x_spark_aot" >&2
    exit 2
    ;;
esac
if ! [[ "$b12x_route_lanes" =~ ^[1-4]$ ]]; then
  echo "GLMRT_B12X_SPARK_ROUTE_LANES must be an integer in 1..4, got: $b12x_route_lanes" >&2
  exit 2
fi
if ! [[ "$protocol_v2_verbs_host_execution_lanes" =~ ^[1-8]$ ]]; then
  echo "GLMRT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES must be an integer in 1..8, got: $protocol_v2_verbs_host_execution_lanes" >&2
  exit 2
fi
if ! [[ "$protocol_v2_packed_direct_max_rows" =~ ^[0-9]+$ ]] \
  || [ "$protocol_v2_packed_direct_max_rows" -lt 8 ] \
  || [ "$protocol_v2_packed_direct_max_rows" -gt 2048 ]; then
  echo "GLMRT_REAL_FULL_PROTOCOL_V2_PACKED_DIRECT_MAX_ROWS must be an integer in 8..2048, got: $protocol_v2_packed_direct_max_rows" >&2
  exit 2
fi
rdma_last_port=$((intermediate_rdma_port + protocol_v2_verbs_host_execution_lanes * 6 - 1))
if [ "$rdma_last_port" -gt 65535 ]; then
  echo "Spark RDMA pair/lane ports end at $rdma_last_port, above 65535" >&2
  exit 2
fi
case "${b12x_grouped_decode,,}" in
  0|1|true|false|yes|no|on|off) ;;
  *)
    echo "GLMRT_B12X_SPARK_GROUPED_DECODE must be boolean-like, got: $b12x_grouped_decode" >&2
    exit 2
    ;;
esac
case "${b12x_w4a16_device_weights,,}" in
  0|1|true|false|yes|no|on|off|device|cuda) ;;
  *)
    echo "GLMRT_B12X_SPARK_W4A16_DEVICE_WEIGHTS must be boolean-like, device, or cuda; got: $b12x_w4a16_device_weights" >&2
    exit 2
    ;;
esac
if [ -n "$b12x_w4a16_decode_grid_x" ] &&
  { ! [[ "$b12x_w4a16_decode_grid_x" =~ ^[0-9]+$ ]] ||
    [ "$b12x_w4a16_decode_grid_x" -lt 1 ] ||
    [ "$b12x_w4a16_decode_grid_x" -gt 96 ]; }; then
  echo "GLMRT_B12X_SPARK_W4A16_DECODE_GRID_X must be an integer in 1..96, got: $b12x_w4a16_decode_grid_x" >&2
  exit 2
fi
case "${b12x_w4a16_m1_fused_sum,,}" in
  0|false|no|off) b12x_w4a16_m1_fused_sum=0 ;;
  1|true|yes|on) b12x_w4a16_m1_fused_sum=1 ;;
  *)
    echo "GLMRT_B12X_SPARK_W4A16_M1_FUSED_SUM must be boolean-like, got: $b12x_w4a16_m1_fused_sum" >&2
    exit 2
    ;;
esac
case "${b12x_w4a16_small_m_mode,,}" in
  ordered|wide|wide-ordered|split-m1) ;;
  *)
    echo "GLMRT_B12X_SPARK_W4A16_SMALL_M_MODE must be wide, ordered, or split-m1; got: $b12x_w4a16_small_m_mode" >&2
    exit 2
    ;;
esac
case "${route_timing,,}" in
  0|1|true|false|yes|no|on|off) ;;
  *)
    echo "GLMRT_REAL_FULL_NVFP4_ROUTE_TIMING must be boolean-like, got: $route_timing" >&2
    exit 2
    ;;
esac
case "${route_validate,,}" in
  0|1|true|false|yes|no|on|off) ;;
  *)
    echo "GLMRT_REAL_FULL_CUDA_ROUTE_VALIDATE must be boolean-like, got: $route_validate" >&2
    exit 2
    ;;
esac
case "${route_cuda_event_timing,,}" in
  0|1|true|false|yes|no|on|off) ;;
  *)
    echo "GLMRT_REAL_FULL_NVFP4_ROUTE_CUDA_EVENT_TIMING must be boolean-like, got: $route_cuda_event_timing" >&2
    exit 2
    ;;
esac

spark_secondary_rail_ip() {
  case "$1" in
    ostrich) echo "10.55.0.5" ;;
    dodo) echo "10.55.0.6" ;;
    emu) echo "10.55.0.7" ;;
    kiwi) echo "10.55.0.8" ;;
    *) return 1 ;;
  esac
}

IFS=',' read -r -a hosts <<< "$hosts_csv"
if [ "${#hosts[@]}" -eq 0 ]; then
  echo "GLMRT_SPARK_HOSTS did not contain any hosts" >&2
  exit 2
fi
if [ "$intermediate_shards" = "4" ] && [ "${#hosts[@]}" -ne 4 ]; then
  echo "four-way intermediate sharding requires exactly four Spark hosts" >&2
  exit 2
fi
intermediate_reduction_root="${GLMRT_EXPERT_INTERMEDIATE_REDUCTION_ROOT:-${hosts[0]}${image_link_suffix}}"
spark_transformer_tp_root="${spark_transformer_tp_root:-$intermediate_reduction_root}"
intermediate_owner_peers="${GLMRT_EXPERT_INTERMEDIATE_OWNER_PEERS:-}"
if [ -z "$intermediate_owner_peers" ]; then
  for host in "${hosts[@]}"; do
    if [ -n "$intermediate_owner_peers" ]; then
      intermediate_owner_peers+=","
    fi
    intermediate_owner_peers+="${host}${image_link_suffix}:${intermediate_owner_port}"
  done
fi
intermediate_rdma_peers="${GLMRT_EXPERT_INTERMEDIATE_RDMA_PEERS:-}"
if [ -z "$intermediate_rdma_peers" ]; then
  for host in "${hosts[@]}"; do
    if [ -n "$intermediate_rdma_peers" ]; then
      intermediate_rdma_peers+=","
    fi
    intermediate_rdma_peers+="${host}${image_link_suffix}"
  done
fi
if [ -z "$intermediate_rdma_additional_peers" ]; then
  discovered_additional_peers=""
  all_secondary_rails_known=1
  for host in "${hosts[@]}"; do
    host="$(echo "$host" | xargs)"
    if ! secondary_ip="$(spark_secondary_rail_ip "$host")"; then
      all_secondary_rails_known=0
      break
    fi
    if [ -n "$discovered_additional_peers" ]; then
      discovered_additional_peers+=","
    fi
    discovered_additional_peers+="$secondary_ip"
  done
  if [ "$all_secondary_rails_known" = "1" ]; then
    intermediate_rdma_additional_peers="$discovered_additional_peers"
  fi
fi
if [ -n "$intermediate_rdma_additional_peers" ] && [ -z "$intermediate_rdma_devices" ]; then
  intermediate_rdma_devices="rocep1s0f0,roceP2p1s0f0"
fi

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
    ssh -o BatchMode=yes "${hosts[0]}" \
      'printf "%s/glmrt-phase0-spark-bench" "$HOME"'
  )"
fi
if [ "$mode" = "real" ] && [ "$image_only" != "1" ] && [ "$prebuilt" != "1" ]; then
  need jq
  test -f "$catalog" || {
    echo "catalog not found: $catalog" >&2
    exit 2
  }
fi

container_name_for_host() {
  local host="$1"
  echo "${container_prefix}-${host}-${port}"
}

cleanup() {
  # Cache-only staging must be observational with respect to already-running
  # expert containers. The generic benchmark cleanup otherwise removes the
  # production-named workers even though this invocation never started them.
  if [ "$keep_experts" = "1" ] || [ "$model_sync_only" = "1" ]; then
    return
  fi
  for host in "${hosts[@]}"; do
    local container
    container="$(container_name_for_host "$host")"
    ssh -o BatchMode=yes "$host" bash -s -- "$container" "$remote_dir" <<'REMOTE' >/dev/null 2>&1 || true
set -euo pipefail
container="$1"
remote_dir="$2"
log_dir="$remote_dir/.glmrt-cache/model-artifacts/diagnostic/logs"
mkdir -p "$log_dir"
docker logs "$container" >"$log_dir/${container}.log" 2>&1 || true
docker rm -f "$container" >/dev/null 2>&1 || true
REMOTE
  done
}
trap cleanup EXIT

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
    --exclude 'native/build*/' \
    --exclude 'reports/phase0_artifacts/benchmarks/' \
    --exclude 'reports/phase0_artifacts/logs/' \
    --exclude 'reports/phase0_artifacts/smoke/' \
    --exclude 'reports/phase0_artifacts/tests/' \
    --exclude '.glmrt-cache/model-artifacts/diagnostic/benchmarks/' \
    --exclude '.glmrt-cache/model-artifacts/diagnostic/logs/' \
    --exclude '.glmrt-cache/model-artifacts/diagnostic/smoke/' \
    --exclude '.glmrt-cache/model-artifacts/diagnostic/tests/' \
    "$repo_root"/ "$host:$remote_dir"/
  if [ "$prebuilt" != "1" ] \
    && { [[ "$catalog" == .glmrt-cache/* ]] || [[ "$loadplan_dir" == .glmrt-cache/* ]]; }; then
    ssh -o BatchMode=yes "$host" \
      "mkdir -p '$remote_dir/$(dirname "$catalog")' '$remote_dir/$loadplan_dir'"
    rsync -az "$repo_root/$catalog" "$host:$remote_dir/$catalog"
    rsync -az "$repo_root/$loadplan_dir/" "$host:$remote_dir/$loadplan_dir/"
  fi
  case "$sparkinfer_source_w4a16_aot_build" in
    1|true|yes|on)
      ssh -o BatchMode=yes "$host" \
        "mkdir -p '$remote_dir/.glmrt-cache/external/sparkinfer'"
      rsync -az --delete \
        --exclude '__pycache__/' \
        "$repo_root/.glmrt-cache/external/sparkinfer/sparkinfer/" \
        "$host:$remote_dir/.glmrt-cache/external/sparkinfer/sparkinfer/"
      ;;
  esac
}

sync_model_cache_to_host() {
  local host="$1"
  local model_cache_key="models--${model_id//\//--}"
  local local_hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
  local local_model_root="$local_hf_home/hub/$model_cache_key"
  local revision
  local remote_hf_home

  test -f "$local_model_root/refs/main" || {
    echo "local model cache is missing refs/main: $local_model_root" >&2
    return 1
  }
  revision="$(<"$local_model_root/refs/main")"
  test -n "$revision" && test -d "$local_model_root/snapshots/$revision" || {
    echo "local model snapshot is incomplete: $local_model_root revision=$revision" >&2
    return 1
  }
  if find "$local_model_root/snapshots/$revision" -xtype l -print -quit | grep -q .; then
    echo "local model snapshot contains unresolved blobs: $local_model_root revision=$revision" >&2
    return 1
  fi

  remote_hf_home="$(
    ssh -o BatchMode=yes "$host" \
      'printf "%s" "${HF_HOME:-$HOME/.cache/huggingface}"'
  )"
  if ssh -o BatchMode=yes "$host" bash -s -- \
    "$remote_hf_home/hub/$model_cache_key" "$revision" <<'REMOTE'
set -euo pipefail
model_root="$1"
revision="$2"
test -f "$model_root/refs/main"
test "$(<"$model_root/refs/main")" = "$revision"
test -d "$model_root/snapshots/$revision"
if find "$model_root/snapshots/$revision" -xtype l -print -quit | grep -q .; then
  exit 1
fi
REMOTE
  then
    echo "== model cache already complete on $host: $model_id@$revision =="
    return 0
  fi

  echo "== syncing model cache to $host: $model_id@$revision =="
  ssh -o BatchMode=yes "$host" "mkdir -p '$remote_hf_home/hub/$model_cache_key'"
  rsync -a --partial \
    "$local_model_root/" "$host:$remote_hf_home/hub/$model_cache_key/"
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
  for candidate in "${hosts[@]}"; do
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
    spark-netcat)
      copy_image_spark_netcat "$src" "$dest"
      ;;
    ssh-relay)
      copy_image_ssh_relay "$src" "$dest"
      ;;
    none)
      return 1
      ;;
  esac
}

ensure_image() {
  local host="$1"
  if [ "$force_build_image" != "1" ] && image_exists "$host"; then
    if [ -z "$image_seed_host" ]; then
      image_seed_host="$host"
    fi
    return
  fi
  if [ "$force_build_image" != "1" ] && [ "$image_copy_method" != "none" ]; then
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
  if [ "$force_build_image" != "1" ] && [ "$build_missing_image" != "1" ]; then
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

loadplan_for_host() {
  local host="$1"
  echo "${loadplan_dir}/loadplan.${host}.json"
}

expert_id_for_host() {
  local host="$1"
  local loadplan
  loadplan="$(loadplan_for_host "$host")"
  test -f "$loadplan" || {
    echo "loadplan not found for $host: $loadplan" >&2
    exit 2
  }
  local expert_id
  expert_id="$(
    jq -r --arg host "$host" --argjson layer "$layer_id" '
      .assignments[]
      | select(.owner == $host and .layer_id == $layer and (.tensor_name | endswith(".gate_proj.weight")))
      | .expert_id
    ' "$loadplan" | head -1
  )"
  if [ -z "$expert_id" ] || [ "$expert_id" = "null" ]; then
    echo "no owned layer $layer_id gate projection expert found in $loadplan" >&2
    exit 2
  fi
  echo "$expert_id"
}

start_expertd() {
  local host="$1"
  local container="$2"
  local loadplan="$3"
  local intermediate_shard_rank="$4"
  local catalog_arg="${catalog:-__none__}"
  local loadplan_arg="${loadplan:-__none__}"
  local verbs_ib_port_num_arg="${verbs_ib_port_num:-__unset__}"
  local b12x_w4a16_decode_grid_x_arg="${b12x_w4a16_decode_grid_x:-__unset__}"
  local release_config_sha256_arg="${GLMRT_RELEASE_CONFIG_SHA256:-__unset__}"
  local intermediate_owner_peers_arg="${intermediate_owner_peers:-__unset__}"
  local intermediate_rdma_additional_peers_arg="${intermediate_rdma_additional_peers:-__unset__}"
  local intermediate_rdma_devices_arg="${intermediate_rdma_devices:-__unset__}"
  echo "== starting $mode ProtocolV2 expertd on $host:$port transport=$expert_transport real_layer=${expert_real_layer:-all} mtp_bf16_experts=$mtp_bf16_experts intermediate_shard=${intermediate_shard_rank}/${intermediate_shards} intermediate_reduction=$intermediate_reduction reduction_dtype=$intermediate_reduction_dtype owner_reduction_dtype=$intermediate_owner_reduction_dtype fused_fp8_reduction=$fused_fp8_reduction protocol_v2_execution_lanes=$protocol_v2_verbs_host_execution_lanes packed_direct_max_rows=$protocol_v2_packed_direct_max_rows spark_layer_blocks=$spark_layer_blocks transformer_tp=$spark_transformer_tp transformer_tp_range=$spark_transformer_tp_range managed_route_projections=${managed_route_projections:-0} grouped_multirow=${grouped_multirow:-0} route_cuda_graphs=${route_cuda_graphs:-0} b12x_spark_aot=${b12x_spark_aot:-0} b12x_route_lanes=$b12x_route_lanes b12x_grouped_decode=$b12x_grouped_decode b12x_w4a16_packed=$b12x_w4a16_packed sparkinfer_source_w4a16=$sparkinfer_source_w4a16 sparkinfer_source_direct_max_rows=$sparkinfer_source_w4a16_direct_max_rows b12x_w4a16_device_weights=$b12x_w4a16_device_weights b12x_w4a16_m1_fused_sum=$b12x_w4a16_m1_fused_sum b12x_w4a16_small_m_mode=$b12x_w4a16_small_m_mode nccl_ib_hca=$nccl_ib_hca nccl_cross_nic=$nccl_cross_nic nccl_netdevs_policy=$nccl_netdevs_policy nccl_ib_merge_nics=$nccl_ib_merge_nics nccl_p2p_net_chunksize=$nccl_p2p_net_chunksize nccl_launch_order_implicit=$nccl_launch_order_implicit route_cuda_event_timing=${route_cuda_event_timing:-0} route_timing=${route_timing:-0} build_profile=$build_profile gpu_runtime=$gpu_runtime =="
  ssh -o BatchMode=yes "$host" bash -s -- \
    "$remote_dir" "$image" "$container" "$mode" "$port" "$catalog_arg" "$loadplan_arg" "$host" "$layer_id" "$expert_real_layer" "$managed_route_projections" "$build_profile" "$gpu_runtime" "${GLMRT_PROTOCOL_V2_TCP_TIMING:-0}" "${GLMRT_REAL_FULL_PROTOCOL_V2_EXECUTOR_TIMING:-0}" "$grouped_multirow" "$route_cuda_graphs" "$route_timing" "$expert_transport" "$verbs_ib_port_num_arg" "$route_cuda_event_timing" "$b12x_spark_aot" "$route_validate" "$b12x_route_lanes" "$intermediate_shards" "$intermediate_shard_rank" "$intermediate_reduction" "$intermediate_reduction_dtype" "$intermediate_reduction_root" "$intermediate_reduction_port" "$intermediate_reduction_min_rows" "$nccl_socket_ifname" "$nccl_ib_hca" "$nccl_debug" "$b12x_grouped_decode" "$intermediate_owner_max_rows" "$intermediate_owner_port" "$intermediate_owner_peers_arg" "$spark_layer_blocks" "$spark_layer_block_kv_dtype" "$spark_transformer_tp" "$spark_transformer_tp_range" "$spark_transformer_tp_root" "$spark_transformer_tp_port" "$spark_transformer_tp_collective_probe_iters" "$b12x_w4a16_packed" "$fused_fp8_reduction" "$nccl_bf16_reduce" "$b12x_w4a16_decode_grid_x_arg" "$b12x_w4a16_device_weights" "$intermediate_row_sharded_reduction" "$nccl_launch_order_implicit" "$protocol_v2_verbs_host_execution_lanes" "$intermediate_rdma_peers" "$intermediate_rdma_port" "$intermediate_rdma_slot_bytes" "$intermediate_rdma_ring_depth" "$intermediate_owner_reduction_dtype" "$nccl_cross_nic" "$intermediate_rdma_additional_peers_arg" "$intermediate_rdma_devices_arg" "$intermediate_rdma_stripe_min_bytes" "$nccl_netdevs_policy" "$nccl_ib_merge_nics" "$nccl_p2p_net_chunksize" "$protocol_v2_packed_direct_max_rows" "$b12x_w4a16_m1_fused_sum" "$b12x_w4a16_small_m_mode" "$sparkinfer_source_w4a16" "$sparkinfer_source_w4a16_aot_build" "$sparkinfer_source_w4a16_direct_max_rows" "$model_id" "${GLMRT_SPARK_INCLUDE_MTP_LAYER:-1}" "$mtp_bf16_experts" "$release_config_sha256_arg" "$prebuilt" "$route_preload_io_workers" "$weight_preload_nccl_port" "$route_preload_cooperative" <<'REMOTE'
set -euo pipefail
remote_dir="$1"
image="$2"
container="$3"
mode="$4"
port="$5"
catalog="$6"
if [ "$catalog" = "__none__" ]; then
  catalog=""
fi
loadplan="$7"
if [ "$loadplan" = "__none__" ]; then
  loadplan=""
fi
role_hostname="$8"
layer_id="$9"
expert_real_layer="${10:-}"
managed_route_projections="${11:-0}"
build_profile="${12:-release}"
gpu_runtime="${13:-nvidia}"
protocol_v2_tcp_timing="${14:-0}"
protocol_v2_executor_timing="${15:-0}"
grouped_multirow="${16:-0}"
route_cuda_graphs="${17:-0}"
route_timing="${18:-0}"
expert_transport="${19:-tcp}"
verbs_ib_port_num="${20:-}"
if [ "$verbs_ib_port_num" = "__unset__" ]; then
  verbs_ib_port_num=""
fi
route_cuda_event_timing="${21:-0}"
b12x_spark_aot="${22:-0}"
route_validate="${23:-0}"
b12x_route_lanes="${24:-4}"
intermediate_shards="${25:-1}"
intermediate_shard_rank="${26:-0}"
runtime_role="spark-${intermediate_shard_rank}"
intermediate_reduction="${27:-coordinator}"
intermediate_reduction_dtype="${28:-fp8}"
intermediate_reduction_root="${29:-ostrich.200gb}"
intermediate_reduction_port="${30:-9200}"
intermediate_reduction_min_rows="${31:-16}"
nccl_socket_ifname="${32:-enp1s0f0np0}"
nccl_ib_hca="${33:-=rocep1s0f0,roceP2p1s0f0}"
nccl_debug="${34:-WARN}"
b12x_grouped_decode="${35:-1}"
intermediate_owner_max_rows="${36:-8}"
intermediate_owner_port="${37:-9100}"
intermediate_owner_peers="${38:-}"
if [ "$intermediate_owner_peers" = "__unset__" ]; then
  intermediate_owner_peers=""
fi
spark_layer_blocks="${39:-0}"
spark_layer_block_kv_dtype="${40:-fp8}"
spark_transformer_tp="${41:-0}"
spark_transformer_tp_range="${42:-0:78}"
spark_transformer_tp_root="${43:-ostrich.200gb}"
spark_transformer_tp_port="${44:-9300}"
spark_transformer_tp_collective_probe_iters="${45:-0}"
b12x_w4a16_packed="${46:-1}"
fused_fp8_reduction="${47:-1}"
nccl_bf16_reduce="${48:-0}"
b12x_w4a16_decode_grid_x="${49:-}"
if [ "$b12x_w4a16_decode_grid_x" = "__unset__" ]; then
  b12x_w4a16_decode_grid_x=""
fi
b12x_w4a16_device_weights="${50:-1}"
intermediate_row_sharded_reduction="${51:-0}"
nccl_launch_order_implicit="${52:-1}"
protocol_v2_verbs_host_execution_lanes="${53:-4}"
intermediate_rdma_peers="${54:-}"
intermediate_rdma_port="${55:-9400}"
intermediate_rdma_slot_bytes="${56:-4194304}"
intermediate_rdma_ring_depth="${57:-4}"
intermediate_owner_reduction_dtype="${58:-$intermediate_reduction_dtype}"
nccl_cross_nic="${59:-2}"
intermediate_rdma_additional_peers="${60:-}"
intermediate_rdma_devices="${61:-}"
if [ "$intermediate_rdma_additional_peers" = "__unset__" ]; then
  intermediate_rdma_additional_peers=""
fi
if [ "$intermediate_rdma_devices" = "__unset__" ]; then
  intermediate_rdma_devices=""
fi
intermediate_rdma_stripe_min_bytes="${62:-262144}"
nccl_netdevs_policy="${63:-ALL}"
nccl_ib_merge_nics="${64:-1}"
nccl_p2p_net_chunksize="${65:-131072}"
protocol_v2_packed_direct_max_rows="${66:-2048}"
b12x_w4a16_m1_fused_sum="${67:-0}"
b12x_w4a16_small_m_mode="${68:-wide}"
sparkinfer_source_w4a16="${69:-0}"
sparkinfer_source_w4a16_aot_build="${70:-$sparkinfer_source_w4a16}"
sparkinfer_source_w4a16_direct_max_rows="${71:-2}"
model_id="${72:-lukealonso/GLM-5.2-NVFP4}"
include_mtp_layer="${73:-1}"
mtp_bf16_experts="${74:-0}"
release_config_sha256="${75:-}"
if [ "$release_config_sha256" = "__unset__" ]; then
  release_config_sha256=""
fi
prebuilt="${76:-0}"
route_preload_io_workers="${77:-128}"
weight_preload_nccl_port="${78:-9350}"
route_preload_cooperative="${79:-1}"

discover_rdma_device_map() {
  local entries=()
  local rdma_path
  for rdma_path in /sys/class/infiniband/*; do
    [ -e "$rdma_path" ] || continue
    local rdma_device
    rdma_device="$(basename "$rdma_path")"
    local net_path
    for net_path in "$rdma_path"/device/net/*; do
      [ -e "$net_path" ] || continue
      local net_device
      net_device="$(basename "$net_path")"
      local cidr
      while read -r cidr; do
        [ -n "$cidr" ] || continue
        entries+=("${cidr%/*}=${rdma_device}")
      done < <(ip -4 -o addr show dev "$net_device" 2>/dev/null | awk '{print $4}')
    done
  done
  local IFS=,
  echo "${entries[*]}"
}

rdma_device_map=""
if [ "$expert_transport" = "verbs-host" ]; then
  rdma_device_map="$(discover_rdma_device_map)"
  if [ -z "$rdma_device_map" ]; then
    echo "unable to map Spark RDMA device IPs from /sys/class/infiniband" >&2
    exit 2
  fi
  echo "verbs-host RDMA device map: $rdma_device_map" >&2
fi

docker rm -f "$container" >/dev/null 2>&1 || true
docker_args=(
  run -d
  --name "$container"
  --net=host
  --ipc=host
  # Docker's default seccomp profile blocks io_uring_setup. The expert image
  # runs only trusted, prebuilt glmrt code and needs io_uring for cold HF reads.
  --security-opt seccomp=unconfined
  --ulimit memlock=-1:-1
  --cap-add IPC_LOCK
  -v "$remote_dir:/workspace/glmrt"
  -v "${HF_HOME:-$HOME/.cache/huggingface}:${HF_HOME:-$HOME/.cache/huggingface}:ro"
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface:ro"
  -e HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
  -e HF_HUB_OFFLINE=1
  -e TRANSFORMERS_OFFLINE=1
  -e GLMRT_MODEL_ID="$model_id"
  -e GLMRT_SPARK_INCLUDE_MTP_LAYER="$include_mtp_layer"
  -e GLMRT_MTP_BF16_EXPERTS="$mtp_bf16_experts"
  -e GLMRT_RELEASE_CONFIG_SHA256="$release_config_sha256"
  -e GLMRT_BENCH_MODE="$mode"
  -e GLMRT_BENCH_PORT="$port"
  -e GLMRT_BENCH_CATALOG="$catalog"
  -e GLMRT_BENCH_LOADPLAN="$loadplan"
  -e GLMRT_BENCH_ROLE_HOSTNAME="$runtime_role"
  -e GLMRT_BENCH_LAYER_ID="$layer_id"
  -e GLMRT_BENCH_REAL_LAYER="$expert_real_layer"
  -e GLMRT_BENCH_TRANSPORT="$expert_transport"
  -e GLMRT_SPARK_BUILD_PROFILE="$build_profile"
  -e GLMRT_SPARK_PREBUILT="$prebuilt"
  -e GLMRT_REAL_FULL_NVFP4_ROUTE_MANAGED_PROJECTIONS="$managed_route_projections"
  -e GLMRT_REAL_FULL_NVFP4_ROUTE_PRELOAD_IO_WORKERS="$route_preload_io_workers"
  -e GLMRT_REAL_FULL_NVFP4_ROUTE_PRELOAD_COOPERATIVE="$route_preload_cooperative"
  -e GLMRT_EXPERT_WEIGHT_PRELOAD_NCCL_PORT="$weight_preload_nccl_port"
  -e GLMRT_REAL_FULL_NVFP4_ROUTE_GROUPED_MULTIROW="$grouped_multirow"
  -e GLMRT_REAL_FULL_NVFP4_ROUTE_CUDA_GRAPHS="$route_cuda_graphs"
  -e GLMRT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES="$protocol_v2_verbs_host_execution_lanes"
  -e GLMRT_B12X_SPARK_AOT_BUILD="$b12x_spark_aot"
  -e GLMRT_B12X_SPARK_ROUTE_LANES="$b12x_route_lanes"
  -e GLMRT_B12X_SPARK_GROUPED_DECODE="$b12x_grouped_decode"
  -e GLMRT_B12X_SPARK_W4A16_PACKED="$b12x_w4a16_packed"
  -e GLMRT_SPARKINFER_SOURCE_W4A16="$sparkinfer_source_w4a16"
  -e GLMRT_SPARKINFER_SOURCE_W4A16_AOT_BUILD="$sparkinfer_source_w4a16_aot_build"
  -e GLMRT_SPARKINFER_SOURCE_W4A16_DIRECT_MAX_ROWS="$sparkinfer_source_w4a16_direct_max_rows"
  -e GLMRT_B12X_SPARK_W4A16_DEVICE_WEIGHTS="$b12x_w4a16_device_weights"
  -e GLMRT_B12X_SPARK_W4A16_DECODE_GRID_X="$b12x_w4a16_decode_grid_x"
  -e GLMRT_B12X_SPARK_W4A16_M1_FUSED_SUM="$b12x_w4a16_m1_fused_sum"
  -e GLMRT_B12X_SPARK_W4A16_SMALL_M_MODE="$b12x_w4a16_small_m_mode"
  -e GLMRT_EXPERT_INTERMEDIATE_SHARDS="$intermediate_shards"
  -e GLMRT_EXPERT_INTERMEDIATE_SHARD_RANK="$intermediate_shard_rank"
  -e GLMRT_EXPERT_INTERMEDIATE_REDUCTION="$intermediate_reduction"
  -e GLMRT_EXPERT_INTERMEDIATE_REDUCTION_DTYPE="$intermediate_reduction_dtype"
  -e GLMRT_EXPERT_INTERMEDIATE_OWNER_REDUCTION_DTYPE="$intermediate_owner_reduction_dtype"
  -e GLMRT_EXPERT_INTERMEDIATE_REDUCTION_ROOT="$intermediate_reduction_root"
  -e GLMRT_EXPERT_INTERMEDIATE_REDUCTION_PORT="$intermediate_reduction_port"
  -e GLMRT_EXPERT_INTERMEDIATE_REDUCTION_MIN_ROWS="$intermediate_reduction_min_rows"
  -e GLMRT_EXPERT_INTERMEDIATE_ROW_SHARDED_REDUCTION="$intermediate_row_sharded_reduction"
  -e GLMRT_EXPERT_FUSED_FP8_REDUCTION="$fused_fp8_reduction"
  -e GLMRT_EXPERT_NCCL_BF16_REDUCE="$nccl_bf16_reduce"
  -e GLMRT_EXPERT_INTERMEDIATE_OWNER_MAX_ROWS="$intermediate_owner_max_rows"
  -e GLMRT_EXPERT_INTERMEDIATE_OWNER_PORT="$intermediate_owner_port"
  -e GLMRT_EXPERT_INTERMEDIATE_OWNER_PEERS="$intermediate_owner_peers"
  -e GLMRT_EXPERT_INTERMEDIATE_RDMA_PEERS="$intermediate_rdma_peers"
  -e GLMRT_EXPERT_INTERMEDIATE_RDMA_ADDITIONAL_PEERS="$intermediate_rdma_additional_peers"
  -e GLMRT_EXPERT_INTERMEDIATE_RDMA_DEVICES="$intermediate_rdma_devices"
  -e GLMRT_EXPERT_INTERMEDIATE_RDMA_PORT="$intermediate_rdma_port"
  -e GLMRT_EXPERT_INTERMEDIATE_RDMA_SLOT_BYTES="$intermediate_rdma_slot_bytes"
  -e GLMRT_EXPERT_INTERMEDIATE_RDMA_RING_DEPTH="$intermediate_rdma_ring_depth"
  -e GLMRT_EXPERT_INTERMEDIATE_RDMA_STRIPE_MIN_BYTES="$intermediate_rdma_stripe_min_bytes"
  -e GLMRT_SPARK_LAYER_BLOCKS="$spark_layer_blocks"
  -e GLMRT_SPARK_LAYER_BLOCK_OWNER_ENDPOINT="127.0.0.1:${port}"
  -e GLMRT_SPARK_LAYER_BLOCK_KV_DTYPE="$spark_layer_block_kv_dtype"
  -e GLMRT_SPARK_LAYER_BLOCK_ATTENTION_PYTHON_CAPTURE="$spark_layer_blocks"
  -e GLMRT_SPARK_TRANSFORMER_TP="$spark_transformer_tp"
  -e GLMRT_SPARK_TRANSFORMER_TP_RANGE="$spark_transformer_tp_range"
  -e GLMRT_SPARK_TRANSFORMER_TP_ROOT="$spark_transformer_tp_root"
  -e GLMRT_SPARK_TRANSFORMER_TP_PORT="$spark_transformer_tp_port"
  -e GLMRT_SPARK_TRANSFORMER_TP_COLLECTIVE_PROBE_ITERS="$spark_transformer_tp_collective_probe_iters"
  -e NCCL_SOCKET_IFNAME="$nccl_socket_ifname"
  -e NCCL_IB_HCA="$nccl_ib_hca"
  -e NCCL_CROSS_NIC="$nccl_cross_nic"
  -e NCCL_NETDEVS_POLICY="$nccl_netdevs_policy"
  -e NCCL_IB_MERGE_NICS="$nccl_ib_merge_nics"
  -e NCCL_P2P_NET_CHUNKSIZE="$nccl_p2p_net_chunksize"
  -e NCCL_DEBUG="$nccl_debug"
  -e NCCL_LAUNCH_ORDER_IMPLICIT="$nccl_launch_order_implicit"
  -e GLMRT_REAL_FULL_NVFP4_ROUTE_CUDA_EVENT_TIMING="$route_cuda_event_timing"
  -e GLMRT_REAL_FULL_NVFP4_ROUTE_TIMING="$route_timing"
  -e GLMRT_REAL_FULL_CUDA_ROUTE_VALIDATE="$route_validate"
  -e GLMRT_PROTOCOL_V2_TCP_TIMING="$protocol_v2_tcp_timing"
  -e GLMRT_REAL_FULL_PROTOCOL_V2_EXECUTOR_TIMING="$protocol_v2_executor_timing"
  -e GLMRT_REAL_FULL_PROTOCOL_V2_PACKED_DIRECT_MAX_ROWS="$protocol_v2_packed_direct_max_rows"
)
if [ -n "$rdma_device_map" ]; then
  docker_args+=(-e GLMRT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP="$rdma_device_map")
fi
case "$gpu_runtime" in
  nvidia)
    docker_args+=(--gpus all)
    nvml_library="$(readlink -f /usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1 2>/dev/null || true)"
    if [ -n "$nvml_library" ] && [ -s "$nvml_library" ]; then
      docker_args+=(-v "$nvml_library:/usr/local/nvidia/lib64/libnvidia-ml.so.1:ro")
    fi
    ;;
  manual)
    for device in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset; do
      if [ -e "$device" ]; then
        docker_args+=(--device="$device")
      fi
    done
    if [ -d /dev/nvidia-caps ]; then
      for device in /dev/nvidia-caps/*; do
        if [ -e "$device" ]; then
          docker_args+=(--device="$device")
        fi
      done
    fi
    nvml_library="$(readlink -f /usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1 2>/dev/null || true)"
    if [ -n "$nvml_library" ] && [ -s "$nvml_library" ]; then
      docker_args+=(-v "$nvml_library:/usr/local/nvidia/lib64/libnvidia-ml.so.1:ro")
    fi
    ;;
  *)
    echo "unsupported GLMRT_SPARK_GPU_RUNTIME=$gpu_runtime" >&2
    exit 2
    ;;
esac
if [ -n "$verbs_ib_port_num" ]; then
  docker_args+=(-e GLMRT_VERBS_APP_IB_PORT_NUM="$verbs_ib_port_num")
fi
if [ -e /dev/infiniband ]; then
  docker_args+=(--device=/dev/infiniband)
fi

docker "${docker_args[@]}" "$image" bash -lc '
set -euo pipefail
cd /workspace/glmrt
if [ "${GLMRT_SPARK_PREBUILT:-0}" = "1" ]; then
  test -x /opt/glmrt/bin/glmrt
  test -s /opt/glmrt/lib/libglmrt_native.so
  export GLMRT_REAL_FULL_CUDA_REFERENCE_KERNELS=1
  export GLMRT_NATIVE_LIB=/opt/glmrt/lib/libglmrt_native.so
  real_layer_args=()
  case "${GLMRT_BENCH_REAL_LAYER:-}" in
    ""|all|none) ;;
    *) real_layer_args=(--real-layer "$GLMRT_BENCH_REAL_LAYER") ;;
  esac
  exec /opt/glmrt/bin/glmrt expertd \
    --transport "${GLMRT_BENCH_TRANSPORT:-tcp}" \
    --listen "0.0.0.0:${GLMRT_BENCH_PORT}" \
    --model-id "$GLMRT_MODEL_ID" \
    "${real_layer_args[@]}" \
    --role "$GLMRT_BENCH_ROLE_HOSTNAME"
fi
cargo_args=()
bin_profile=debug
if [ "${GLMRT_SPARK_BUILD_PROFILE:-release}" = "release" ]; then
  cargo_args+=(--release)
  bin_profile=release
fi
cargo build --manifest-path rust/Cargo.toml -p glmrt-daemon "${cargo_args[@]}"
clean_stale_cmake_build_dir() {
  local build_dir="$1"
  local expected_source="$2"
  local cache="${build_dir}/CMakeCache.txt"
  local cached_source=""
  if [ ! -f "$cache" ]; then
    return
  fi
  cached_source="$(sed -n "s/^CMAKE_HOME_DIRECTORY:INTERNAL=//p" "$cache" | tail -n 1)"
  if [ -n "$cached_source" ] && [ "$cached_source" != "$expected_source" ]; then
    echo "removing stale CMake build dir ${build_dir} cached_source=${cached_source} expected_source=${expected_source}" >&2
    rm -rf "$build_dir"
  fi
}
if [ "${GLMRT_BENCH_TRANSPORT:-tcp}" = "verbs-host" ] && [ "$GLMRT_BENCH_MODE" != "real" ]; then
  clean_stale_cmake_build_dir native/build-rdma "$(pwd)/native"
  python3 python/tools/check_native_rdma_build.py \
    --build-dir native/build-rdma \
    --output ".glmrt-cache/model-artifacts/diagnostic/benchmarks/native_rdma_build_${GLMRT_BENCH_ROLE_HOSTNAME}.json" \
    --require-pass
  export GLMRT_NATIVE_LIB=/workspace/glmrt/native/build-rdma/libglmrt_native.so
fi
if [ "$GLMRT_BENCH_MODE" = "real" ]; then
  rdma_enabled=OFF
  native_build_dir=native/build-cuda
  if [ "${GLMRT_BENCH_TRANSPORT:-tcp}" = "verbs-host" ]; then
    rdma_enabled=ON
    native_build_dir=native/build-cuda-rdma
  fi
  b12x_aot=OFF
  case "${GLMRT_B12X_SPARK_AOT_BUILD:-0}" in
    1|true|yes|on) b12x_aot=ON ;;
  esac
  sparkinfer_source_w4a16_aot=OFF
  case "${GLMRT_SPARKINFER_SOURCE_W4A16_AOT_BUILD:-0}" in
    1|true|yes|on) sparkinfer_source_w4a16_aot=ON ;;
  esac
  nccl_enabled=OFF
  case "${GLMRT_EXPERT_INTERMEDIATE_REDUCTION:-coordinator}" in
    spark|spark-hybrid|spark-rdma|spark-rdma-hybrid) nccl_enabled=ON ;;
  esac
  clean_stale_cmake_build_dir "$native_build_dir" "$(pwd)/native"
  cmake -S native -B "$native_build_dir" -G Ninja \
    -DGLMRT_ENABLE_CUDA=ON \
    -DGLMRT_ENABLE_RDMA="$rdma_enabled" \
    -DGLMRT_ENABLE_B12X_AOT="$b12x_aot" \
    -DGLMRT_ENABLE_SPARKINFER_SOURCE_W4A16_AOT="$sparkinfer_source_w4a16_aot" \
    -DGLMRT_SPARKINFER_SOURCE_DIR="${GLMRT_SPARKINFER_SOURCE_DIR:-$(pwd)/.glmrt-cache/external/sparkinfer}" \
    -DGLMRT_ENABLE_NCCL="$nccl_enabled" \
    -DGLMRT_CUDA_ARCHITECTURES=121
  cmake --build "$native_build_dir"
  export GLMRT_REAL_FULL_CUDA_REFERENCE_KERNELS=1
  export GLMRT_NATIVE_LIB="/workspace/glmrt/${native_build_dir}/libglmrt_native.so"
  real_layer_args=()
  case "${GLMRT_BENCH_REAL_LAYER:-}" in
    ""|all|none) ;;
    *) real_layer_args=(--real-layer "$GLMRT_BENCH_REAL_LAYER") ;;
  esac
  catalog_args=()
  loadplan_args=()
  [ -z "$GLMRT_BENCH_CATALOG" ] || catalog_args=(--catalog "$GLMRT_BENCH_CATALOG")
  [ -z "$GLMRT_BENCH_LOADPLAN" ] || loadplan_args=(--loadplan "$GLMRT_BENCH_LOADPLAN")
  exec "rust/target/${bin_profile}/glmrt" expertd \
    --transport "${GLMRT_BENCH_TRANSPORT:-tcp}" \
    --listen "0.0.0.0:${GLMRT_BENCH_PORT}" \
    "${catalog_args[@]}" \
    "${loadplan_args[@]}" \
    "${real_layer_args[@]}" \
    --role "$GLMRT_BENCH_ROLE_HOSTNAME"
fi
exec "rust/target/${bin_profile}/glmrt" expertd \
  --synthetic-weights \
  --transport "${GLMRT_BENCH_TRANSPORT:-tcp}" \
  --listen "0.0.0.0:${GLMRT_BENCH_PORT}"
'
REMOTE
}

wait_for_port() {
  local host="$1"
  local check_host
  check_host="$(
    ssh -G "$host" 2>/dev/null \
      | awk '$1 == "hostname" { print $2; exit }'
  )"
  check_host="${check_host:-$host}"
  local timeout_s="${GLMRT_SPARK_EXPERT_READY_TIMEOUT:-}"
  if [ -z "$timeout_s" ]; then
    if [ "$mode" = "real" ]; then
      timeout_s=900
    else
      timeout_s=180
    fi
  fi
  local deadline=$((SECONDS + timeout_s))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if timeout 2 bash -c ":</dev/tcp/${check_host}/${port}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "expert daemon on ${host}:${port} did not become ready within ${timeout_s}s" >&2
  ssh -o BatchMode=yes "$host" "docker logs '$(container_name_for_host "$host")' 2>&1 | tail -120" >&2 || true
  exit 1
}

targets=()
expert_ids=()
if [ "$skip_stage" != "1" ] && [ "$mode" = "real" ] && [ "$sync_model_cache" = "1" ]; then
  model_sync_pids=()
  for host in "${hosts[@]}"; do
    sync_model_cache_to_host "$host" &
    model_sync_pids+=("$!")
  done
  model_sync_failed=0
  for pid in "${model_sync_pids[@]}"; do
    wait "$pid" || model_sync_failed=1
  done
  if [ "$model_sync_failed" != "0" ]; then
    echo "one or more Spark model-cache syncs failed" >&2
    exit 1
  fi
fi
if [ "$model_sync_only" = "1" ]; then
  if [ "$sync_model_cache" != "1" ]; then
    echo "GLMRT_SPARK_MODEL_SYNC_ONLY=1 requires GLMRT_SPARK_SYNC_MODEL_CACHE=1" >&2
    exit 2
  fi
  echo "Spark model-cache sync complete: $model_id"
  exit 0
fi
for host_index in "${!hosts[@]}"; do
  host="${hosts[$host_index]}"
  if [ "$skip_stage" != "1" ]; then
    stage_repo "$host"
  fi
  ensure_image "$host"
done

if [ "$image_only" = "1" ]; then
  echo "Spark image '$image' is ready on: ${hosts[*]}"
  exit 0
fi

for host_index in "${!hosts[@]}"; do
  host="${hosts[$host_index]}"
  loadplan=""
  if [ "$mode" = "real" ]; then
    if [ "$prebuilt" = "1" ]; then
      expert_ids+=("${host}=${host_index}")
    else
      loadplan="$(loadplan_for_host "$host")"
      expert_ids+=("${host}=$(expert_id_for_host "$host")")
    fi
  fi
  start_expertd "$host" "$(container_name_for_host "$host")" "$loadplan" "$host_index"
  targets+=("${host}:${port}")
done

for host in "${hosts[@]}"; do
  wait_for_port "$host"
done

if [ "$mode" = "real" ]; then
  expected_executor="protocol-v2-real-nvfp4-checkpoint-executor"
else
  expected_executor="protocol-v2-synthetic-route-dependent-executor"
fi

export GLMRT_PHASE0_TCP_EXPERT_ADDRS="${GLMRT_PHASE0_TCP_EXPERT_ADDRS:-$(IFS=,; echo "${targets[*]}")}"
export GLMRT_PHASE0_TCP_EXPECTED_EXECUTOR="${GLMRT_PHASE0_TCP_EXPECTED_EXECUTOR:-$expected_executor}"
export GLMRT_PHASE0_TCP_REQUIRE_EXPECTED_EXECUTOR=1
export GLMRT_PHASE0_TCP_LAYER_ID="$layer_id"
if [ "$mode" = "real" ]; then
  export GLMRT_PHASE0_TCP_EXPERT_IDS="${GLMRT_PHASE0_TCP_EXPERT_IDS:-$(IFS=,; echo "${expert_ids[*]}")}"
fi

echo "== running phase0 binary ProtocolV2 TCP benchmark =="
echo "targets=$GLMRT_PHASE0_TCP_EXPERT_ADDRS"
echo "expected_executor=$GLMRT_PHASE0_TCP_EXPECTED_EXECUTOR"
echo "measured_timeout_ms=${GLMRT_PHASE0_TCP_TIMEOUT_MS:-5000}"
if [ "$mode" = "real" ]; then
  echo "expert_real_layer=${expert_real_layer:-all}"
  echo "managed_route_projections=${managed_route_projections:-0}"
  echo "grouped_multirow=${grouped_multirow:-0}"
  echo "route_cuda_graphs=${route_cuda_graphs:-0}"
  echo "b12x_spark_aot=${b12x_spark_aot:-0}"
  echo "b12x_route_lanes=$b12x_route_lanes"
  echo "b12x_grouped_decode=$b12x_grouped_decode"
  echo "b12x_w4a16_device_weights=$b12x_w4a16_device_weights"
  echo "protocol_v2_packed_direct_max_rows=$protocol_v2_packed_direct_max_rows"
  echo "route_cuda_event_timing=${route_cuda_event_timing:-0}"
  echo "gpu_runtime=${gpu_runtime}"
  echo "real_precompile_warmup=${GLMRT_PHASE0_TCP_WARMUP:-1} warmup_timeout_ms=${GLMRT_PHASE0_TCP_WARMUP_TIMEOUT_MS:-120000}"
fi
if [ "${GLMRT_PHASE0_TCP_EXPERT_IDS:-}" ]; then
  echo "expert_ids=$GLMRT_PHASE0_TCP_EXPERT_IDS"
fi

if [ "$skip_bench" = "1" ]; then
  echo "Skipping phase0 benchmark because GLMRT_PHASE0_SPARK_SKIP_BENCH=1"
else
  python3 benchmarks/phase0_bench.py
fi

if [ "$keep_experts" = "1" ]; then
  echo "Spark expert containers left running because GLMRT_SPARK_KEEP_EXPERTS=1"
fi
