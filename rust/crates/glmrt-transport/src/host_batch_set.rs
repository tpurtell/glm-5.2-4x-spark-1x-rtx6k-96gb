use anyhow::{bail, Context, Result};
use futures::future::try_join_all;
use glmrt_core::{
    plan_completion_first_routes, CompletionRoutePlanEntry, ExpertGraphHostBatchSetLease,
    ExpertGraphInstancePool, ExpertHostBatch, ExpertHostBatchSet, ExpertHostBatchSetAccumulation,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    net::{SocketAddr, ToSocketAddrs},
    path::Path,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        mpsc as std_mpsc, Arc,
    },
    time::Instant,
};

use crate::verbs::{
    VerbsHostProtocolV2CqHarvester, VerbsHostProtocolV2LaneFanoutClient,
    VerbsHostProtocolV2LaneFanoutPending,
};
use crate::{
    tcp_protocol_v2_roundtrip, verbs_host_protocol_v2_roundtrip, ExpertProtocolV2Request,
    ExpertProtocolV2Response, ExpertProtocolV2ResponseHeader, ExpertProtocolV2ResponseView,
    ExpertProtocolV2Status, ExpertProtocolV2StreamPlan, ExpertV2Dtype,
    TcpProtocolV2PersistentClient, TcpTransportConfig, VerbsHostProtocolV2PersistentClient,
    VerbsHostProtocolV2ResponseChunk, VerbsHostProtocolV2ResponsePayload,
    VerbsHostProtocolV2ResponseStreamStats,
};

const PROTOCOL_V2_TCP_TIMING_ENV: &str = "GLMRT_PROTOCOL_V2_TCP_TIMING";
const PROTOCOL_V2_EXPERT_QUEUE_STATS_ENV: &str = "GLMRT_PROTOCOL_V2_EXPERT_QUEUE_STATS";
const PROTOCOL_V2_EXPERT_QUEUE_ROW_ROUTES_ENV: &str = "GLMRT_PROTOCOL_V2_EXPERT_QUEUE_ROW_ROUTES";
const PROTOCOL_V2_EXPERT_QUEUE_ROW_ROUTES_GATE_FILE_ENV: &str =
    "GLMRT_PROTOCOL_V2_EXPERT_QUEUE_ROW_ROUTES_GATE_FILE";
const PROTOCOL_V2_STREAM_INGRESS_ROWS_ENV: &str = "GLMRT_PROTOCOL_V2_STREAM_INGRESS_ROWS";
const PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES_ENV: &str =
    "GLMRT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES";
const PROTOCOL_V2_VERBS_HOST_SHARED_CQ_HARVESTER_ENV: &str =
    "GLMRT_PROTOCOL_V2_VERBS_HOST_SHARED_CQ_HARVESTER";
const PROTOCOL_V2_VERBS_HOST_ADDITIONAL_RAILS_ENV: &str =
    "GLMRT_PROTOCOL_V2_VERBS_HOST_ADDITIONAL_RAILS";
const PROTOCOL_V2_VERBS_HOST_STRIPE_MIN_ROWS_ENV: &str =
    "GLMRT_PROTOCOL_V2_VERBS_HOST_STRIPE_MIN_ROWS";
const PROTOCOL_V2_VERBS_HOST_STRIPE_SPARK_REDUCTION_ENV: &str =
    "GLMRT_PROTOCOL_V2_VERBS_HOST_STRIPE_SPARK_REDUCTION";
const PROTOCOL_V2_STREAM_MAX_ROUTE_GROUP_ROWS: usize = 256;
const DEFAULT_VERBS_HOST_EXECUTION_LANES: usize = 4;
const DEFAULT_VERBS_HOST_STRIPE_MIN_ROWS: usize = 64;
const MAX_VERBS_HOST_RAILS_PER_HOST: usize = 4;
pub const MAX_VERBS_HOST_EXECUTION_LANES: usize = 8;
const SPARK_COLLECTIVE_REQUEST_ID_NAMESPACE: u64 = 1 << 63;
const SPARK_COLLECTIVE_REQUEST_ID_STRIDE: u64 = 65_536;
// NCCL requires one launch order across communicators. API request ID ranges
// overlap, so full-rank reductions use a separate process-wide sequence.
static NEXT_SPARK_COLLECTIVE_REQUEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn reserve_spark_collective_request_ids_from(
    sequence: &AtomicU64,
    count: usize,
) -> Result<Vec<u64>> {
    anyhow::ensure!(
        count > 0,
        "Spark collective request ID reservation is empty"
    );
    let count = u64::try_from(count).context("Spark collective request ID count exceeds u64")?;
    let max_sequence =
        (u64::MAX - SPARK_COLLECTIVE_REQUEST_ID_NAMESPACE) / SPARK_COLLECTIVE_REQUEST_ID_STRIDE;
    let sequence = sequence
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |sequence| {
            sequence
                .checked_add(count)
                .filter(|next| *next <= max_sequence + 1)
        })
        .map_err(|_| anyhow::anyhow!("Spark collective request ID sequence is exhausted"))?;
    (0..count)
        .map(|offset| {
            SPARK_COLLECTIVE_REQUEST_ID_NAMESPACE
                .checked_add((sequence + offset) * SPARK_COLLECTIVE_REQUEST_ID_STRIDE)
                .context("Spark collective request ID overflow")
        })
        .collect()
}

fn reserve_spark_collective_request_id_from(sequence: &AtomicU64) -> Result<u64> {
    reserve_spark_collective_request_ids_from(sequence, 1).map(|mut ids| ids.remove(0))
}

fn reserve_spark_collective_request_id() -> Result<u64> {
    reserve_spark_collective_request_id_from(&NEXT_SPARK_COLLECTIVE_REQUEST_SEQUENCE)
}

fn reserve_spark_collective_request_ids(count: usize) -> Result<Vec<u64>> {
    reserve_spark_collective_request_ids_from(&NEXT_SPARK_COLLECTIVE_REQUEST_SEQUENCE, count)
}

fn parse_verbs_host_execution_lanes(value: Option<&str>) -> Result<usize> {
    let lanes = value
        .map(str::parse::<usize>)
        .transpose()
        .context("parsing verbs-host execution lane count")?
        .unwrap_or(DEFAULT_VERBS_HOST_EXECUTION_LANES);
    anyhow::ensure!(
        (1..=MAX_VERBS_HOST_EXECUTION_LANES).contains(&lanes),
        "{PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES_ENV} must be between 1 and {MAX_VERBS_HOST_EXECUTION_LANES}, got {lanes}"
    );
    Ok(lanes)
}

pub fn protocol_v2_verbs_host_execution_lanes() -> Result<usize> {
    parse_verbs_host_execution_lanes(
        env::var(PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES_ENV)
            .ok()
            .as_deref(),
    )
}

fn protocol_v2_verbs_host_shared_cq_harvester_enabled() -> bool {
    protocol_v2_bool_env(PROTOCOL_V2_VERBS_HOST_SHARED_CQ_HARVESTER_ENV)
}

fn parse_verbs_host_stripe_min_rows(value: Option<&str>) -> Result<usize> {
    let rows = value
        .map(str::parse::<usize>)
        .transpose()
        .context("parsing verbs-host stripe minimum rows")?
        .unwrap_or(DEFAULT_VERBS_HOST_STRIPE_MIN_ROWS);
    anyhow::ensure!(
        rows > 0,
        "{PROTOCOL_V2_VERBS_HOST_STRIPE_MIN_ROWS_ENV} must be positive"
    );
    Ok(rows)
}

fn protocol_v2_verbs_host_stripe_min_rows() -> Result<usize> {
    parse_verbs_host_stripe_min_rows(
        env::var(PROTOCOL_V2_VERBS_HOST_STRIPE_MIN_ROWS_ENV)
            .ok()
            .as_deref(),
    )
}

fn protocol_v2_verbs_host_stripe_spark_reduction_enabled() -> bool {
    protocol_v2_bool_env(PROTOCOL_V2_VERBS_HOST_STRIPE_SPARK_REDUCTION_ENV)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TcpProtocolV2HostBatchTarget {
    pub host: String,
    pub addr: SocketAddr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VerbsHostProtocolV2TargetRails {
    host: String,
    addrs: Vec<SocketAddr>,
}

fn verbs_host_target_rails(
    targets: &[TcpProtocolV2HostBatchTarget],
) -> Result<Vec<VerbsHostProtocolV2TargetRails>> {
    let raw = env::var(PROTOCOL_V2_VERBS_HOST_ADDITIONAL_RAILS_ENV).unwrap_or_default();
    parse_verbs_host_target_rails(targets, &raw)
}

fn parse_verbs_host_target_rails(
    targets: &[TcpProtocolV2HostBatchTarget],
    raw_additional_rails: &str,
) -> Result<Vec<VerbsHostProtocolV2TargetRails>> {
    let mut rails = Vec::with_capacity(targets.len());
    for target in targets {
        anyhow::ensure!(
            !rails
                .iter()
                .any(|candidate: &VerbsHostProtocolV2TargetRails| candidate.host == target.host),
            "duplicate verbs-host target for logical host {}",
            target.host
        );
        rails.push(VerbsHostProtocolV2TargetRails {
            host: target.host.clone(),
            addrs: vec![target.addr],
        });
    }

    for entry in raw_additional_rails
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
    {
        let (host, raw_addrs) = entry.split_once('=').with_context(|| {
            format!(
                "invalid {PROTOCOL_V2_VERBS_HOST_ADDITIONAL_RAILS_ENV} entry {entry:?}; expected host=addr[+addr]"
            )
        })?;
        let host = host.trim();
        let target = rails
            .iter_mut()
            .find(|target| target.host == host)
            .with_context(|| {
                format!("{PROTOCOL_V2_VERBS_HOST_ADDITIONAL_RAILS_ENV} names unknown host {host:?}")
            })?;
        for raw_addr in raw_addrs
            .split('+')
            .map(str::trim)
            .filter(|addr| !addr.is_empty())
        {
            let addr = resolve_verbs_host_rail_addr(raw_addr, target.addrs[0].port())?;
            anyhow::ensure!(
                !target.addrs.contains(&addr),
                "duplicate verbs-host rail address {addr} for host {host}"
            );
            target.addrs.push(addr);
        }
        anyhow::ensure!(
            target.addrs.len() <= MAX_VERBS_HOST_RAILS_PER_HOST,
            "verbs-host host {host} has {} rails; maximum is {MAX_VERBS_HOST_RAILS_PER_HOST}",
            target.addrs.len()
        );
    }
    Ok(rails)
}

fn resolve_verbs_host_rail_addr(raw_addr: &str, default_port: u16) -> Result<SocketAddr> {
    let with_port = if raw_addr.contains(':') {
        raw_addr.to_owned()
    } else {
        format!("{raw_addr}:{default_port}")
    };
    with_port
        .to_socket_addrs()
        .with_context(|| format!("resolving verbs-host rail address {with_port}"))?
        .next()
        .with_context(|| format!("verbs-host rail address {with_port} resolved to no addresses"))
}

#[derive(Debug, Clone, PartialEq)]
struct VerbsHostProtocolV2RailRequest {
    request: ExpertProtocolV2Request,
    local_row_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VerbsHostProtocolV2ResponseStreamMap {
    host_index: usize,
    local_row_indices: Vec<usize>,
}

fn partition_protocol_v2_request_for_rails(
    request: ExpertProtocolV2Request,
    rail_count: usize,
    stripe_min_rows: usize,
) -> Result<Vec<VerbsHostProtocolV2RailRequest>> {
    anyhow::ensure!(
        rail_count > 0,
        "verbs-host request requires at least one rail"
    );
    anyhow::ensure!(
        stripe_min_rows > 0,
        "verbs-host stripe minimum rows must be positive"
    );
    let row_count = request.rows.len();
    if rail_count == 1
        || row_count < stripe_min_rows
        || row_count < 2
        || request.stream_plan_enabled()
        || request.stream_data_enabled()
        || request.precompile_warmup_enabled()
    {
        return Ok(vec![VerbsHostProtocolV2RailRequest {
            local_row_indices: (0..row_count).collect(),
            request,
        }]);
    }

    let part_count = rail_count
        .min(row_count)
        .max(row_count.div_ceil(PROTOCOL_V2_STREAM_MAX_ROUTE_GROUP_ROWS));
    let rows_per_part = row_count.div_ceil(part_count);
    let hidden_stride = request.header.hidden_row_stride_bytes as usize;
    anyhow::ensure!(
        request.hidden_payload.len() == row_count * hidden_stride,
        "verbs-host striped request hidden payload bytes {} did not match rows {row_count} * stride {hidden_stride}",
        request.hidden_payload.len()
    );
    let flags = request.header.flags;
    let mut partitions = Vec::with_capacity(part_count);
    for row_start in (0..row_count).step_by(rows_per_part) {
        let row_end = (row_start + rows_per_part).min(row_count);
        let mut rows = Vec::with_capacity(row_end - row_start);
        let mut routes = Vec::new();
        for (new_row_index, row) in request.rows[row_start..row_end].iter().enumerate() {
            let route_start = row.route_offset as usize;
            let route_end = route_start
                .checked_add(row.route_count as usize)
                .context("verbs-host striped request route range overflow")?;
            let row_routes = request.routes.get(route_start..route_end).with_context(|| {
                format!(
                    "verbs-host striped request row {} route range {route_start}..{route_end} is out of bounds",
                    row_start + new_row_index
                )
            })?;
            let mut partition_row = row.clone();
            partition_row.route_offset = u32::try_from(routes.len())
                .context("verbs-host striped request route offset exceeds u32")?;
            rows.push(partition_row);
            for route in row_routes {
                let mut partition_route = route.clone();
                partition_route.row_index = u32::try_from(new_row_index)
                    .context("verbs-host striped request row index exceeds u32")?;
                routes.push(partition_route);
            }
        }
        let hidden_start = row_start * hidden_stride;
        let hidden_end = row_end * hidden_stride;
        let mut partition = ExpertProtocolV2Request::new_with_hidden_stride(
            request.header.request_id,
            request.header.placement_version,
            request.header.layer_id,
            request.header.hidden_dim,
            request.header.hidden_dtype,
            request.header.hidden_row_stride_bytes,
            rows,
            routes,
            request.hidden_payload[hidden_start..hidden_end].to_vec(),
        )?;
        partition.header.flags = flags;
        partition.validate()?;
        partitions.push(VerbsHostProtocolV2RailRequest {
            request: partition,
            local_row_indices: (row_start..row_end).collect(),
        });
    }
    Ok(partitions)
}

fn assign_spark_collective_request_ids(
    partitions: &mut [VerbsHostProtocolV2RailRequest],
    host_index: usize,
) -> Result<()> {
    let request_ids = reserve_spark_collective_request_ids(partitions.len())?;
    assign_shared_spark_collective_request_ids(partitions, host_index, &request_ids)
}

fn assign_shared_spark_collective_request_ids(
    partitions: &mut [VerbsHostProtocolV2RailRequest],
    host_index: usize,
    request_ids: &[u64],
) -> Result<()> {
    let host_offset =
        u64::try_from(host_index).context("Spark collective host index exceeds u64")?;
    anyhow::ensure!(
        host_offset < 16,
        "Spark collective host index {host_index} exceeds the 16-ID host window"
    );
    anyhow::ensure!(
        partitions.len() == request_ids.len(),
        "Spark collective partition count {} did not match request ID count {}",
        partitions.len(),
        request_ids.len()
    );
    for (partition, request_id) in partitions.iter_mut().zip(request_ids.iter().copied()) {
        anyhow::ensure!(
            request_id % SPARK_COLLECTIVE_REQUEST_ID_STRIDE == 0,
            "Spark collective request ID {request_id} is not stride aligned"
        );
        partition.request.header.request_id = request_id
            .checked_add(host_offset)
            .context("Spark collective host request ID overflow")?;
    }
    Ok(())
}

fn mark_striped_spark_collective_parts(
    partitions: &mut [VerbsHostProtocolV2RailRequest],
    rail_count: usize,
) -> Result<()> {
    anyhow::ensure!(
        partitions.len() <= rail_count,
        "row-sharded Spark collective produced {} partitions for {rail_count} rails; reduce dispatch rows",
        partitions.len()
    );
    if partitions.len() <= 1 {
        return Ok(());
    }
    let part_count = partitions.len();
    for partition in partitions {
        partition
            .request
            .set_spark_collective_part_count(part_count)?;
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub struct TcpProtocolV2HostBatchSetDispatch {
    pub accumulation: ExpertHostBatchSetAccumulation,
    pub partial_outputs_bf16_by_host: Vec<Vec<u8>>,
    pub stats: TcpProtocolV2HostBatchSetDispatchStats,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TcpProtocolV2HostBatchSetBf16PayloadDispatch {
    pub partial_outputs_bf16_by_host: Vec<Vec<u8>>,
    pub global_row_indices_by_host: Vec<Vec<usize>>,
    pub completed_global_row_slices: Vec<Vec<usize>>,
    /// Structural dispatch stats. `output_checksum` is a placeholder until the
    /// coordinator-side GPU reconstruction path fills it from the routed output.
    pub stats: TcpProtocolV2HostBatchSetDispatchStats,
}

#[derive(Debug, PartialEq, Eq)]
pub struct VerbsHostProtocolV2HostBatchSetBf16PayloadChunk {
    pub host_index: usize,
    pub partial_output: VerbsHostProtocolV2ResponsePayload,
    pub output_dtype: ExpertV2Dtype,
    pub output_row_stride_bytes: usize,
    pub global_row_indices: Vec<usize>,
    pub completed_global_row_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TcpProtocolV2HostBatchSetDispatchStats {
    pub hosts: usize,
    pub global_rows: usize,
    pub host_rows: usize,
    pub routes: usize,
    pub output_dim: usize,
    pub output_values: usize,
    pub request_wire_bytes: usize,
    pub response_wire_bytes: usize,
    pub response_executor_ids: Vec<u64>,
    pub contribution_counts: Vec<usize>,
    pub output_checksum: f64,
    pub graph_pool_leases: usize,
    pub graph_pool_fixed_buffer_bytes: usize,
    pub graph_pool_active_rows: usize,
    pub graph_pool_active_routes: usize,
    pub graph_pool_active_expert_tiles: usize,
    pub graph_pool_bucket_rows: Vec<usize>,
}

pub struct TcpProtocolV2HostBatchSetPersistentClient {
    targets: Vec<TcpProtocolV2HostBatchTarget>,
    clients: Vec<TcpProtocolV2PersistentClient>,
}

impl TcpProtocolV2HostBatchSetPersistentClient {
    pub fn new(targets: Vec<TcpProtocolV2HostBatchTarget>, config: TcpTransportConfig) -> Self {
        let clients = targets
            .iter()
            .map(|target| TcpProtocolV2PersistentClient::new(target.addr, config.clone()))
            .collect();
        Self { targets, clients }
    }

    pub async fn dispatch_bf16(
        &mut self,
        set: &ExpertHostBatchSet,
        global_hidden_payload: &[u8],
        request_id_base: u64,
    ) -> Result<TcpProtocolV2HostBatchSetDispatch> {
        let result = tcp_protocol_v2_host_batch_set_bf16_dispatch_persistent_inner(
            set,
            global_hidden_payload,
            request_id_base,
            &self.targets,
            &mut self.clients,
        )
        .await;
        if result.is_err() {
            self.reset_clients();
        }
        result
    }

    pub async fn dispatch_bf16_payload(
        &mut self,
        set: &ExpertHostBatchSet,
        global_hidden_payload: &[u8],
        request_id_base: u64,
    ) -> Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch> {
        let result = tcp_protocol_v2_host_batch_set_bf16_payload_dispatch_persistent_inner(
            set,
            global_hidden_payload,
            request_id_base,
            &self.targets,
            &mut self.clients,
            PayloadContributionCounts::Include,
        )
        .await;
        if result.is_err() {
            self.reset_clients();
        }
        result
    }

    pub async fn dispatch_bf16_payload_structural_stats(
        &mut self,
        set: &ExpertHostBatchSet,
        global_hidden_payload: &[u8],
        request_id_base: u64,
    ) -> Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch> {
        let result = tcp_protocol_v2_host_batch_set_bf16_payload_dispatch_persistent_inner(
            set,
            global_hidden_payload,
            request_id_base,
            &self.targets,
            &mut self.clients,
            PayloadContributionCounts::Omit,
        )
        .await;
        if result.is_err() {
            self.reset_clients();
        }
        result
    }

    fn reset_clients(&mut self) {
        for client in &mut self.clients {
            client.reset();
        }
    }
}

#[derive(Clone)]
pub struct VerbsHostProtocolV2HostBatchSetPersistentClient {
    targets: Vec<TcpProtocolV2HostBatchTarget>,
    clients_by_lane: Vec<Vec<VerbsHostProtocolV2PersistentClient>>,
    additional_clients_by_lane: Vec<Vec<Vec<VerbsHostProtocolV2PersistentClient>>>,
    lane_fanout_clients: Vec<VerbsHostProtocolV2LaneFanoutClient>,
    stripe_min_rows: usize,
    lane_locks: Vec<Arc<tokio::sync::Mutex<()>>>,
    next_lane: Arc<AtomicUsize>,
}

pub struct VerbsHostProtocolV2ReducedIdentityPayloadPending {
    client: VerbsHostProtocolV2HostBatchSetPersistentClient,
    response_dtype: ExpertV2Dtype,
    kind: VerbsHostProtocolV2DirectPayloadKind,
    global_rows: usize,
    output_dim: usize,
    request_wire_bytes: usize,
    response_stream_maps: Vec<VerbsHostProtocolV2ResponseStreamMap>,
    response_chunk_rx:
        Option<tokio::sync::mpsc::UnboundedReceiver<VerbsHostProtocolV2ResponseChunk>>,
    response_receivers:
        Option<Vec<tokio::sync::oneshot::Receiver<Result<VerbsHostProtocolV2ResponseStreamStats>>>>,
    lane_fanout_pending: Option<VerbsHostProtocolV2LaneFanoutPending>,
    _lane_guard: tokio::sync::OwnedMutexGuard<()>,
}

enum VerbsHostProtocolV2DirectPayloadKind {
    ReducedIdentity {
        logical_routes: usize,
    },
    ReplicatedOneRow {
        host_count: usize,
        logical_routes: usize,
    },
    HostBatchSet {
        set: ExpertHostBatchSet,
    },
}

pub enum VerbsHostProtocolV2ReducedIdentityPayloadStart {
    Started(VerbsHostProtocolV2ReducedIdentityPayloadPending),
    Busy(ExpertProtocolV2Request),
}

pub enum VerbsHostProtocolV2HostBatchSetPayloadStart {
    Started(VerbsHostProtocolV2ReducedIdentityPayloadPending),
    Busy(Vec<u8>),
}

impl VerbsHostProtocolV2HostBatchSetPersistentClient {
    pub fn new(
        targets: Vec<TcpProtocolV2HostBatchTarget>,
        config: TcpTransportConfig,
    ) -> Result<Self> {
        let execution_lanes = protocol_v2_verbs_host_execution_lanes()?;
        let target_rails = verbs_host_target_rails(&targets)?;
        let stripe_min_rows = protocol_v2_verbs_host_stripe_min_rows()?;
        let mut clients_by_lane = Vec::with_capacity(execution_lanes);
        let mut additional_clients_by_lane = Vec::with_capacity(execution_lanes);
        let mut lane_fanout_clients = Vec::with_capacity(execution_lanes);
        for lane in 0..execution_lanes {
            let execution_lane =
                u32::try_from(lane).context("verbs-host ProtocolV2 execution lane exceeds u32")?;
            let cq_harvester = if protocol_v2_verbs_host_shared_cq_harvester_enabled() {
                Some(VerbsHostProtocolV2CqHarvester::new(execution_lane)?)
            } else {
                None
            };
            let mut primary_clients = Vec::with_capacity(target_rails.len());
            let mut additional_clients = Vec::with_capacity(target_rails.len());
            for target in &target_rails {
                primary_clients.push(
                    VerbsHostProtocolV2PersistentClient::new_with_execution_lane_and_cq_harvester(
                        target.addrs[0],
                        config.clone(),
                        execution_lane,
                        cq_harvester.clone(),
                    )?,
                );
                additional_clients.push(
                    target.addrs[1..]
                        .iter()
                        .map(|addr| {
                            VerbsHostProtocolV2PersistentClient::new_with_execution_lane_and_cq_harvester(
                                *addr,
                                config.clone(),
                                execution_lane,
                                cq_harvester.clone(),
                            )
                        })
                        .collect::<Result<Vec<_>>>()?,
                );
            }
            lane_fanout_clients.push(VerbsHostProtocolV2LaneFanoutClient::new(
                target_rails.iter().map(|target| target.addrs[0]).collect(),
                config.clone(),
                execution_lane,
            )?);
            clients_by_lane.push(primary_clients);
            additional_clients_by_lane.push(additional_clients);
        }
        let lane_locks = (0..execution_lanes)
            .map(|_| Arc::new(tokio::sync::Mutex::new(())))
            .collect();
        Ok(Self {
            targets,
            clients_by_lane,
            additional_clients_by_lane,
            lane_fanout_clients,
            stripe_min_rows,
            lane_locks,
            next_lane: Arc::new(AtomicUsize::new(0)),
        })
    }

    pub async fn dispatch_bf16(
        &self,
        set: &ExpertHostBatchSet,
        global_hidden_payload: &[u8],
        request_id_base: u64,
    ) -> Result<TcpProtocolV2HostBatchSetDispatch> {
        let (lane, _lane_guard) = self.acquire_execution_lane().await;
        let result = verbs_host_protocol_v2_host_batch_set_bf16_dispatch_persistent_inner(
            set,
            global_hidden_payload,
            request_id_base,
            &self.targets,
            &self.clients_by_lane[lane],
        )
        .await;
        if result.is_err() {
            self.reset_clients();
        }
        result
    }

    pub async fn dispatch_bf16_payload(
        &self,
        set: &ExpertHostBatchSet,
        global_hidden_payload: &[u8],
        request_id_base: u64,
    ) -> Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch> {
        let (lane, _lane_guard) = self.acquire_execution_lane().await;
        let result = verbs_host_protocol_v2_host_batch_set_bf16_payload_dispatch_persistent_inner(
            set,
            global_hidden_payload,
            request_id_base,
            &self.targets,
            &self.clients_by_lane[lane],
            &self.additional_clients_by_lane[lane],
            self.stripe_min_rows,
            PayloadContributionCounts::Include,
        )
        .await;
        if result.is_err() {
            self.reset_clients();
        }
        result
    }

    pub async fn dispatch_bf16_payload_structural_stats(
        &self,
        set: &ExpertHostBatchSet,
        global_hidden_payload: &[u8],
        request_id_base: u64,
    ) -> Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch> {
        let (lane, _lane_guard) = self.acquire_execution_lane().await;
        let result = verbs_host_protocol_v2_host_batch_set_bf16_payload_dispatch_persistent_inner(
            set,
            global_hidden_payload,
            request_id_base,
            &self.targets,
            &self.clients_by_lane[lane],
            &self.additional_clients_by_lane[lane],
            self.stripe_min_rows,
            PayloadContributionCounts::Omit,
        )
        .await;
        if result.is_err() {
            self.reset_clients();
        }
        result
    }

    pub async fn dispatch_bf16_payload_streaming(
        &self,
        set: &ExpertHostBatchSet,
        global_hidden_payload: &[u8],
        request_id_base: u64,
        response_dtype: ExpertV2Dtype,
        reduced_root_host_index: Option<usize>,
        owner_fanout: bool,
        row_sharded_reduction: bool,
        chunk_tx: std_mpsc::Sender<VerbsHostProtocolV2HostBatchSetBf16PayloadChunk>,
    ) -> Result<TcpProtocolV2HostBatchSetDispatchStats> {
        let (lane, _lane_guard) = self.acquire_execution_lane().await;
        let striped_row_sharded_reduction =
            row_sharded_reduction && protocol_v2_verbs_host_stripe_spark_reduction_enabled();
        let request_id_base =
            if reduced_root_host_index.is_some() && !owner_fanout && !striped_row_sharded_reduction
            {
                reserve_spark_collective_request_id()?
            } else {
                request_id_base
            };
        let result =
            verbs_host_protocol_v2_host_batch_set_bf16_payload_dispatch_persistent_streaming_inner(
                set,
                global_hidden_payload,
                request_id_base,
                &self.targets,
                &self.clients_by_lane[lane],
                &self.additional_clients_by_lane[lane],
                self.stripe_min_rows,
                PayloadContributionCounts::Omit,
                response_dtype,
                reduced_root_host_index,
                owner_fanout,
                row_sharded_reduction,
                &chunk_tx,
            )
            .await;
        if result.is_err() {
            self.reset_clients();
        }
        result
    }

    pub async fn dispatch_reduced_identity_payload_streaming(
        &self,
        request: ExpertProtocolV2Request,
        response_dtype: ExpertV2Dtype,
        reduced_root_host_index: usize,
        logical_routes: usize,
        chunk_tx: std_mpsc::Sender<VerbsHostProtocolV2HostBatchSetBf16PayloadChunk>,
    ) -> Result<TcpProtocolV2HostBatchSetDispatchStats> {
        anyhow::ensure!(
            request.spark_reduction_enabled(),
            "direct Spark-owner dispatch requires Spark reduction"
        );
        anyhow::ensure!(
            reduced_root_host_index < self.targets.len(),
            "direct Spark-owner root index {reduced_root_host_index} exceeds {} targets",
            self.targets.len()
        );
        anyhow::ensure!(
            logical_routes >= request.header.route_count as usize,
            "direct Spark-owner logical route count {logical_routes} is smaller than request routes {}",
            request.header.route_count
        );
        let (lane, _lane_guard) = self.acquire_execution_lane().await;
        let rail_clients = self.rail_clients(lane, reduced_root_host_index);
        let result = verbs_host_protocol_v2_reduced_identity_payload_dispatch_persistent(
            request,
            response_dtype,
            reduced_root_host_index,
            logical_routes,
            &rail_clients,
            self.stripe_min_rows,
            chunk_tx,
        )
        .await;
        if result.is_err() {
            self.reset_clients();
        }
        result
    }

    pub fn try_start_reduced_identity_payload(
        &self,
        request: ExpertProtocolV2Request,
        response_dtype: ExpertV2Dtype,
        reduced_root_host_index: usize,
        logical_routes: usize,
    ) -> Result<VerbsHostProtocolV2ReducedIdentityPayloadStart> {
        anyhow::ensure!(
            matches!(
                response_dtype,
                ExpertV2Dtype::Bf16
                    | ExpertV2Dtype::Fp8E4m3RowScaled
                    | ExpertV2Dtype::Nvfp4E2m1Fp8E4m3
            ),
            "direct pending Spark-owner response dtype {response_dtype:?} is unsupported"
        );
        anyhow::ensure!(
            request.spark_reduction_enabled(),
            "direct Spark-owner dispatch requires Spark reduction"
        );
        anyhow::ensure!(
            reduced_root_host_index < self.targets.len(),
            "direct Spark-owner root index {reduced_root_host_index} exceeds {} targets",
            self.targets.len()
        );
        anyhow::ensure!(
            logical_routes >= request.header.route_count as usize,
            "direct Spark-owner logical route count {logical_routes} is smaller than request routes {}",
            request.header.route_count
        );
        let Some((lane, lane_guard)) =
            self.lane_locks.iter().enumerate().find_map(|(lane, lock)| {
                Arc::clone(lock)
                    .try_lock_owned()
                    .ok()
                    .map(|guard| (lane, guard))
            })
        else {
            return Ok(VerbsHostProtocolV2ReducedIdentityPayloadStart::Busy(
                request,
            ));
        };
        let global_rows = request.header.row_count as usize;
        let output_dim = request.header.hidden_dim as usize;
        let rail_clients = self.rail_clients(lane, reduced_root_host_index);
        let mut partitions = partition_protocol_v2_request_for_rails(
            request,
            rail_clients.len(),
            self.stripe_min_rows,
        )?;
        assign_spark_collective_request_ids(&mut partitions, reduced_root_host_index)?;
        let request_wire_bytes = partitions
            .iter()
            .map(|partition| partition.request.wire_stats().wire_bytes)
            .sum();
        let (response_chunk_tx, response_chunk_rx) = tokio::sync::mpsc::unbounded_channel();
        let mut response_stream_maps = Vec::with_capacity(partitions.len());
        let mut response_receivers = Vec::with_capacity(partitions.len());
        for (stream_id, partition) in partitions.into_iter().enumerate() {
            let client = rail_clients[stream_id % rail_clients.len()];
            response_stream_maps.push(VerbsHostProtocolV2ResponseStreamMap {
                host_index: reduced_root_host_index,
                local_row_indices: partition.local_row_indices,
            });
            response_receivers.push(client.enqueue_response_chunks(
                partition.request,
                stream_id,
                response_chunk_tx.clone(),
            )?);
        }
        Ok(VerbsHostProtocolV2ReducedIdentityPayloadStart::Started(
            VerbsHostProtocolV2ReducedIdentityPayloadPending {
                client: self.clone(),
                response_dtype,
                kind: VerbsHostProtocolV2DirectPayloadKind::ReducedIdentity { logical_routes },
                global_rows,
                output_dim,
                request_wire_bytes,
                response_stream_maps,
                response_chunk_rx: Some(response_chunk_rx),
                response_receivers: Some(response_receivers),
                lane_fanout_pending: None,
                _lane_guard: lane_guard,
            },
        ))
    }

    pub fn try_start_host_batch_set_payload(
        &self,
        set: ExpertHostBatchSet,
        global_hidden_payload: Vec<u8>,
        request_id_base: u64,
        response_dtype: ExpertV2Dtype,
    ) -> Result<VerbsHostProtocolV2HostBatchSetPayloadStart> {
        anyhow::ensure!(
            matches!(
                response_dtype,
                ExpertV2Dtype::Bf16
                    | ExpertV2Dtype::Fp8E4m3RowScaled
                    | ExpertV2Dtype::Nvfp4E2m1Fp8E4m3
            ),
            "direct host-batch-set response dtype {response_dtype:?} is unsupported"
        );
        anyhow::ensure!(
            set.global_row_count == 1,
            "direct host-batch-set payload currently requires one global row, got {}",
            set.global_row_count
        );
        let Some((lane, lane_guard)) =
            self.lane_locks.iter().enumerate().find_map(|(lane, lock)| {
                Arc::clone(lock)
                    .try_lock_owned()
                    .ok()
                    .map(|guard| (lane, guard))
            })
        else {
            return Ok(VerbsHostProtocolV2HostBatchSetPayloadStart::Busy(
                global_hidden_payload,
            ));
        };

        anyhow::ensure!(
            self.clients_by_lane[lane].len() == self.targets.len(),
            "direct host-batch-set client count {} did not match target count {}",
            self.clients_by_lane[lane].len(),
            self.targets.len()
        );
        let output_dim = set
            .batches
            .first()
            .map(|batch| batch.hidden_dim)
            .context("direct host-batch-set payload requires at least one host batch")?;
        let hidden_bytes_per_row = set
            .batches
            .first()
            .map(|batch| batch.hidden_bytes_per_row)
            .expect("non-empty direct host-batch set has a first batch");
        anyhow::ensure!(
            global_hidden_payload.len() == set.global_row_count * hidden_bytes_per_row,
            "direct host-batch-set hidden payload bytes {} did not match {} rows of {} bytes",
            global_hidden_payload.len(),
            set.global_row_count,
            hidden_bytes_per_row
        );
        set.reconstruction_plan
            .validate_for_batches(&set.batches, set.global_row_count)
            .context("validating direct host-batch-set reconstruction plan")?;

        let (response_chunk_tx, response_chunk_rx) = tokio::sync::mpsc::unbounded_channel();
        let mut response_stream_maps = Vec::with_capacity(set.batches.len());
        let mut response_receivers = Vec::with_capacity(set.batches.len());
        let mut request_wire_bytes = 0_usize;
        let mut used_clients = vec![false; self.targets.len()];
        for (host_index, host_batch) in set.batches.iter().enumerate() {
            let client_index = target_index(&self.targets, &host_batch.host)?;
            anyhow::ensure!(
                !used_clients[client_index],
                "direct host-batch-set payload has duplicate target batch for host {}",
                host_batch.host
            );
            used_clients[client_index] = true;
            let compact_hidden =
                host_batch.compact_hidden_payload(&global_hidden_payload, set.global_row_count)?;
            let request = ExpertProtocolV2Request::from_expert_host_batch(
                request_id_base + host_index as u64,
                host_batch,
                compact_hidden,
            )?;
            let request = request_with_response_dtype(request, response_dtype);
            request_wire_bytes = request_wire_bytes
                .checked_add(request.wire_stats().wire_bytes)
                .context("direct host-batch-set request bytes overflow usize")?;
            let stream_id = response_stream_maps.len();
            response_stream_maps.push(VerbsHostProtocolV2ResponseStreamMap {
                host_index,
                local_row_indices: host_batch.global_row_indices().collect(),
            });
            response_receivers.push(
                self.clients_by_lane[lane][client_index].enqueue_response_chunks(
                    request,
                    stream_id,
                    response_chunk_tx.clone(),
                )?,
            );
        }

        Ok(VerbsHostProtocolV2HostBatchSetPayloadStart::Started(
            VerbsHostProtocolV2ReducedIdentityPayloadPending {
                client: self.clone(),
                response_dtype,
                kind: VerbsHostProtocolV2DirectPayloadKind::HostBatchSet { set },
                global_rows: 1,
                output_dim,
                request_wire_bytes,
                response_stream_maps,
                response_chunk_rx: Some(response_chunk_rx),
                response_receivers: Some(response_receivers),
                lane_fanout_pending: None,
                _lane_guard: lane_guard,
            },
        ))
    }

    pub fn try_start_replicated_one_row_payload(
        &self,
        request: ExpertProtocolV2Request,
        response_dtype: ExpertV2Dtype,
    ) -> Result<VerbsHostProtocolV2HostBatchSetPayloadStart> {
        anyhow::ensure!(
            matches!(
                response_dtype,
                ExpertV2Dtype::Bf16
                    | ExpertV2Dtype::Fp8E4m3RowScaled
                    | ExpertV2Dtype::Nvfp4E2m1Fp8E4m3
            ),
            "direct replicated one-row response dtype {response_dtype:?} is unsupported"
        );
        anyhow::ensure!(
            request.header.row_count == 1 && request.rows.len() == 1,
            "direct replicated one-row payload requires exactly one row"
        );
        anyhow::ensure!(
            !request.spark_reduction_enabled(),
            "direct replicated one-row payload cannot request Spark reduction"
        );
        let Some((lane, lane_guard)) =
            self.lane_locks.iter().enumerate().find_map(|(lane, lock)| {
                Arc::clone(lock)
                    .try_lock_owned()
                    .ok()
                    .map(|guard| (lane, guard))
            })
        else {
            return Ok(VerbsHostProtocolV2HostBatchSetPayloadStart::Busy(
                request.hidden_payload,
            ));
        };

        anyhow::ensure!(
            self.lane_fanout_clients.len() == self.lane_locks.len() && !self.targets.is_empty(),
            "direct replicated one-row fanout has {} lane clients for {} lanes and {} targets",
            self.lane_fanout_clients.len(),
            self.lane_locks.len(),
            self.targets.len(),
        );
        let output_dim = request.header.hidden_dim as usize;
        let hidden_bytes_per_row = request.header.hidden_row_stride_bytes as usize;
        anyhow::ensure!(
            request.hidden_payload.len() == hidden_bytes_per_row,
            "direct replicated one-row hidden payload bytes {} did not match row stride {hidden_bytes_per_row}",
            request.hidden_payload.len()
        );
        let host_count = self.targets.len();
        let logical_routes = request.header.route_count as usize;
        let request_id_base = request.header.request_id;
        let mut requests = Vec::with_capacity(host_count);
        let mut request_wire_bytes = 0_usize;
        let mut request = Some(request);
        for host_index in 0..host_count {
            let mut host_request = if host_index + 1 == host_count {
                request
                    .take()
                    .expect("direct replicated request is present for the final host")
            } else {
                request
                    .as_ref()
                    .expect("direct replicated request is present")
                    .clone()
            };
            host_request.header.request_id = request_id_base
                .checked_add(host_index as u64)
                .context("direct replicated one-row request ID overflow")?;
            host_request = request_with_response_dtype(host_request, response_dtype);
            request_wire_bytes = request_wire_bytes
                .checked_add(host_request.wire_stats().wire_bytes)
                .context("direct replicated one-row request bytes overflow")?;
            requests.push(host_request);
        }
        let lane_fanout_pending = self.lane_fanout_clients[lane].enqueue(requests)?;

        Ok(VerbsHostProtocolV2HostBatchSetPayloadStart::Started(
            VerbsHostProtocolV2ReducedIdentityPayloadPending {
                client: self.clone(),
                response_dtype,
                kind: VerbsHostProtocolV2DirectPayloadKind::ReplicatedOneRow {
                    host_count,
                    logical_routes,
                },
                global_rows: 1,
                output_dim,
                request_wire_bytes,
                response_stream_maps: Vec::new(),
                response_chunk_rx: None,
                response_receivers: None,
                lane_fanout_pending: Some(lane_fanout_pending),
                _lane_guard: lane_guard,
            },
        ))
    }

    fn reset_clients(&self) {
        for client in self.clients_by_lane.iter().flatten() {
            client.reset();
        }
        for client in self.additional_clients_by_lane.iter().flatten().flatten() {
            client.reset();
        }
        for client in &self.lane_fanout_clients {
            client.reset();
        }
    }

    fn rail_clients(
        &self,
        lane: usize,
        host_index: usize,
    ) -> Vec<&VerbsHostProtocolV2PersistentClient> {
        std::iter::once(&self.clients_by_lane[lane][host_index])
            .chain(self.additional_clients_by_lane[lane][host_index].iter())
            .collect()
    }

    async fn acquire_execution_lane(&self) -> (usize, Option<tokio::sync::OwnedMutexGuard<()>>) {
        debug_assert!(!self.lane_locks.is_empty());
        for lane in 0..self.lane_locks.len() {
            if let Ok(guard) = Arc::clone(&self.lane_locks[lane]).try_lock_owned() {
                return (lane, Some(guard));
            }
        }
        let preferred = self.next_lane.fetch_add(1, Ordering::Relaxed) % self.lane_locks.len();
        let guard = Arc::clone(&self.lane_locks[preferred]).lock_owned().await;
        (preferred, Some(guard))
    }
}

impl VerbsHostProtocolV2ReducedIdentityPayloadPending {
    pub fn finish(
        mut self,
    ) -> Result<(
        TcpProtocolV2HostBatchSetDispatchStats,
        Vec<VerbsHostProtocolV2HostBatchSetBf16PayloadChunk>,
    )> {
        let result = self.finish_inner();
        if result.is_err() {
            self.client.reset_clients();
        }
        result
    }

    fn finish_inner(
        &mut self,
    ) -> Result<(
        TcpProtocolV2HostBatchSetDispatchStats,
        Vec<VerbsHostProtocolV2HostBatchSetBf16PayloadChunk>,
    )> {
        let (response_stats_by_stream, mut lane_fanout_chunks) =
            if let Some(pending) = self.lane_fanout_pending.take() {
                anyhow::ensure!(
                    self.response_receivers.is_none() && self.response_chunk_rx.is_none(),
                    "verbs-host lane fanout also carried per-stream response channels"
                );
                let response = pending.wait()?;
                (response.response_stats_by_stream, Some(response.chunks))
            } else {
                let response_stats = self
                    .response_receivers
                    .take()
                    .context("direct verbs-host pending response was already consumed")?
                    .into_iter()
                    .map(|response_rx| {
                        response_rx
                            .blocking_recv()
                            .context("receiving direct verbs-host response completion")?
                    })
                    .collect::<Result<Vec<_>>>()?;
                (response_stats, None)
            };
        let response_stats = merge_verbs_host_response_stream_stats(&response_stats_by_stream)?;
        match &self.kind {
            VerbsHostProtocolV2DirectPayloadKind::ReducedIdentity { logical_routes } => {
                let mut completed_rows = vec![false; self.global_rows];
                let mut chunks = Vec::with_capacity(response_stats.response_frames);
                for _ in 0..response_stats.response_frames {
                    let response_chunk = self
                        .response_chunk_rx
                        .as_mut()
                        .context("direct Spark-owner response chunk channel is missing")?
                        .try_recv()
                        .context(
                            "direct Spark-owner response completion arrived without its response frame",
                        )?;
                    if let Some(chunk) = reduced_identity_payload_chunk(
                        response_chunk,
                        self.response_dtype,
                        &self.response_stream_maps,
                        self.output_dim,
                        &mut completed_rows,
                    )? {
                        chunks.push(chunk);
                    }
                }
                anyhow::ensure!(
                    completed_rows.iter().all(|completed| *completed),
                    "direct Spark-owner response did not complete every request row"
                );
                let stats = reduced_identity_payload_dispatch_stats(
                    self.global_rows,
                    self.output_dim,
                    *logical_routes,
                    self.request_wire_bytes,
                    response_stats,
                )?;
                Ok((stats, chunks))
            }
            VerbsHostProtocolV2DirectPayloadKind::ReplicatedOneRow {
                host_count,
                logical_routes,
            } => {
                anyhow::ensure!(
                    response_stats.response_frames == *host_count,
                    "direct replicated one-row dispatch received {} frames from {host_count} hosts",
                    response_stats.response_frames
                );
                let mut seen_hosts = vec![false; *host_count];
                let mut remaining_contributions = *host_count;
                let mut chunks = Vec::with_capacity(*host_count);
                let response_chunks = lane_fanout_chunks
                    .take()
                    .context("direct replicated one-row lane fanout response is missing")?;
                anyhow::ensure!(
                    response_chunks.len() == *host_count,
                    "direct replicated one-row lane fanout returned {} chunks for {host_count} hosts",
                    response_chunks.len()
                );
                for response_chunk in response_chunks {
                    chunks.push(replicated_one_row_payload_chunk_from_response(
                        *host_count,
                        self.output_dim,
                        self.response_dtype,
                        &mut seen_hosts,
                        &mut remaining_contributions,
                        response_chunk,
                    )?);
                }
                anyhow::ensure!(
                    remaining_contributions == 0 && seen_hosts.iter().all(|seen| *seen),
                    "direct replicated one-row dispatch did not complete every host"
                );
                let routes = logical_routes
                    .checked_mul(*host_count)
                    .context("direct replicated one-row route count overflow")?;
                let stats = TcpProtocolV2HostBatchSetDispatchStats {
                    hosts: *host_count,
                    global_rows: 1,
                    host_rows: *host_count,
                    routes,
                    output_dim: self.output_dim,
                    output_values: self.output_dim,
                    request_wire_bytes: self.request_wire_bytes,
                    response_wire_bytes: response_stats.response_wire_bytes,
                    response_executor_ids: response_stats_by_stream
                        .iter()
                        .map(|stats| stats.response_executor_id)
                        .collect(),
                    contribution_counts: Vec::new(),
                    output_checksum: 0.0,
                    graph_pool_leases: 0,
                    graph_pool_fixed_buffer_bytes: 0,
                    graph_pool_active_rows: 0,
                    graph_pool_active_routes: 0,
                    graph_pool_active_expert_tiles: 0,
                    graph_pool_bucket_rows: Vec::new(),
                };
                Ok((stats, chunks))
            }
            VerbsHostProtocolV2DirectPayloadKind::HostBatchSet { set } => {
                let mut completion_tracker =
                    HostBatchResponseCompletionTracker::new(set, None, false)?;
                let mut chunks = Vec::with_capacity(response_stats.response_frames);
                for _ in 0..response_stats.response_frames {
                    let response_chunk = self
                        .response_chunk_rx
                        .as_mut()
                        .context("direct host-batch-set response chunk channel is missing")?
                        .try_recv()
                        .context(
                            "direct host-batch-set completion arrived without its response frame",
                        )?;
                    if let Some(chunk) = host_batch_payload_chunk_from_response(
                        set,
                        &self.response_stream_maps,
                        self.output_dim,
                        self.response_dtype,
                        &mut completion_tracker,
                        response_chunk,
                    )? {
                        chunks.push(chunk);
                    }
                }
                completion_tracker.finish()?;
                let stats = TcpProtocolV2HostBatchSetDispatchStats {
                    hosts: set.num_hosts(),
                    global_rows: set.global_row_count,
                    host_rows: set.host_row_count(),
                    routes: set.route_count(),
                    output_dim: self.output_dim,
                    output_values: set
                        .global_row_count
                        .checked_mul(self.output_dim)
                        .context("direct host-batch-set output values overflow usize")?,
                    request_wire_bytes: self.request_wire_bytes,
                    response_wire_bytes: response_stats.response_wire_bytes,
                    response_executor_ids: response_stats_by_stream
                        .iter()
                        .map(|stats| stats.response_executor_id)
                        .collect(),
                    contribution_counts: Vec::new(),
                    output_checksum: 0.0,
                    graph_pool_leases: 0,
                    graph_pool_fixed_buffer_bytes: 0,
                    graph_pool_active_rows: 0,
                    graph_pool_active_routes: 0,
                    graph_pool_active_expert_tiles: 0,
                    graph_pool_bucket_rows: Vec::new(),
                };
                Ok((stats, chunks))
            }
        }
    }
}

fn replicated_one_row_payload_chunk_from_response(
    host_count: usize,
    output_dim: usize,
    expected_output_dtype: ExpertV2Dtype,
    seen_hosts: &mut [bool],
    remaining_contributions: &mut usize,
    response_chunk: VerbsHostProtocolV2ResponseChunk,
) -> Result<VerbsHostProtocolV2HostBatchSetBf16PayloadChunk> {
    let VerbsHostProtocolV2ResponseChunk {
        stream_id,
        header,
        row_indices,
        partial_output_payload,
        wire_bytes: _,
    } = response_chunk;
    anyhow::ensure!(
        stream_id < host_count && seen_hosts.len() == host_count,
        "direct replicated one-row response stream {stream_id} exceeds {host_count} hosts"
    );
    anyhow::ensure!(
        !seen_hosts[stream_id],
        "direct replicated one-row host {stream_id} completed twice"
    );
    anyhow::ensure!(
        header.output_dtype == expected_output_dtype,
        "direct replicated one-row response dtype {:?} did not match requested {:?}",
        header.output_dtype,
        expected_output_dtype
    );
    anyhow::ensure!(
        header.row_count == 1,
        "direct replicated one-row response returned {} rows",
        header.row_count
    );
    if let Some(row_indices) = row_indices.as_ref() {
        anyhow::ensure!(
            row_indices.as_slice() == [0],
            "direct replicated one-row response returned nonzero row indices {row_indices:?}"
        );
    }
    let partial_output =
        response_compact_stream_payload(&header, partial_output_payload, 1, output_dim)?;
    let output_row_stride_bytes = header.output_dtype.row_bytes(output_dim)?;
    anyhow::ensure!(
        *remaining_contributions > 0,
        "direct replicated one-row response completed after the row was already ready"
    );
    seen_hosts[stream_id] = true;
    *remaining_contributions -= 1;
    Ok(VerbsHostProtocolV2HostBatchSetBf16PayloadChunk {
        host_index: stream_id,
        partial_output,
        output_dtype: header.output_dtype,
        output_row_stride_bytes,
        global_row_indices: vec![0],
        completed_global_row_indices: (*remaining_contributions == 0)
            .then_some(vec![0])
            .unwrap_or_default(),
    })
}

pub async fn tcp_protocol_v2_host_batch_set_bf16_dispatch(
    set: &ExpertHostBatchSet,
    global_hidden_payload: &[u8],
    targets: &[TcpProtocolV2HostBatchTarget],
    request_id_base: u64,
    config: TcpTransportConfig,
) -> Result<TcpProtocolV2HostBatchSetDispatch> {
    protocol_v2_host_batch_set_bf16_dispatch_inner(
        HostBatchRoundtripTransport::Tcp,
        set,
        global_hidden_payload,
        targets,
        request_id_base,
        config,
        None,
    )
    .await
}

pub async fn tcp_protocol_v2_host_batch_set_bf16_payload_dispatch(
    set: &ExpertHostBatchSet,
    global_hidden_payload: &[u8],
    targets: &[TcpProtocolV2HostBatchTarget],
    request_id_base: u64,
    config: TcpTransportConfig,
) -> Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch> {
    protocol_v2_host_batch_set_bf16_payload_dispatch_inner(
        HostBatchRoundtripTransport::Tcp,
        set,
        global_hidden_payload,
        targets,
        request_id_base,
        config,
        None,
        PayloadContributionCounts::Include,
    )
    .await
}

pub async fn verbs_host_protocol_v2_host_batch_set_bf16_dispatch(
    set: &ExpertHostBatchSet,
    global_hidden_payload: &[u8],
    targets: &[TcpProtocolV2HostBatchTarget],
    request_id_base: u64,
    config: TcpTransportConfig,
) -> Result<TcpProtocolV2HostBatchSetDispatch> {
    protocol_v2_host_batch_set_bf16_dispatch_inner(
        HostBatchRoundtripTransport::VerbsHost,
        set,
        global_hidden_payload,
        targets,
        request_id_base,
        config,
        None,
    )
    .await
}

pub async fn verbs_host_protocol_v2_host_batch_set_bf16_payload_dispatch(
    set: &ExpertHostBatchSet,
    global_hidden_payload: &[u8],
    targets: &[TcpProtocolV2HostBatchTarget],
    request_id_base: u64,
    config: TcpTransportConfig,
) -> Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch> {
    protocol_v2_host_batch_set_bf16_payload_dispatch_inner(
        HostBatchRoundtripTransport::VerbsHost,
        set,
        global_hidden_payload,
        targets,
        request_id_base,
        config,
        None,
        PayloadContributionCounts::Include,
    )
    .await
}

pub async fn verbs_host_protocol_v2_host_batch_set_bf16_payload_dispatch_structural_stats(
    set: &ExpertHostBatchSet,
    global_hidden_payload: &[u8],
    targets: &[TcpProtocolV2HostBatchTarget],
    request_id_base: u64,
    config: TcpTransportConfig,
) -> Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch> {
    protocol_v2_host_batch_set_bf16_payload_dispatch_inner(
        HostBatchRoundtripTransport::VerbsHost,
        set,
        global_hidden_payload,
        targets,
        request_id_base,
        config,
        None,
        PayloadContributionCounts::Omit,
    )
    .await
}

pub async fn tcp_protocol_v2_host_batch_set_bf16_payload_dispatch_with_graph_pool(
    set: &ExpertHostBatchSet,
    global_hidden_payload: &[u8],
    targets: &[TcpProtocolV2HostBatchTarget],
    request_id_base: u64,
    config: TcpTransportConfig,
    graph_pool: &mut ExpertGraphInstancePool,
) -> Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch> {
    let graph_lease = graph_pool
        .acquire_for_host_batch_set(set)
        .context("acquiring ProtocolV2 host-batch-set payload graph pool leases")?;
    let graph_stats = TcpProtocolV2HostBatchSetGraphStats::from(&graph_lease);
    let dispatch = protocol_v2_host_batch_set_bf16_payload_dispatch_inner(
        HostBatchRoundtripTransport::Tcp,
        set,
        global_hidden_payload,
        targets,
        request_id_base,
        config,
        Some(graph_stats),
        PayloadContributionCounts::Include,
    )
    .await;
    graph_pool
        .release_host_batch_set(graph_lease)
        .context("releasing ProtocolV2 host-batch-set payload graph pool leases")?;
    dispatch
}

pub async fn tcp_protocol_v2_host_batch_set_bf16_dispatch_with_graph_pool(
    set: &ExpertHostBatchSet,
    global_hidden_payload: &[u8],
    targets: &[TcpProtocolV2HostBatchTarget],
    request_id_base: u64,
    config: TcpTransportConfig,
    graph_pool: &mut ExpertGraphInstancePool,
) -> Result<TcpProtocolV2HostBatchSetDispatch> {
    let graph_lease = graph_pool
        .acquire_for_host_batch_set(set)
        .context("acquiring ProtocolV2 host-batch-set graph pool leases")?;
    let graph_stats = TcpProtocolV2HostBatchSetGraphStats::from(&graph_lease);
    let dispatch = protocol_v2_host_batch_set_bf16_dispatch_inner(
        HostBatchRoundtripTransport::Tcp,
        set,
        global_hidden_payload,
        targets,
        request_id_base,
        config,
        Some(graph_stats),
    )
    .await;
    graph_pool
        .release_host_batch_set(graph_lease)
        .context("releasing ProtocolV2 host-batch-set graph pool leases")?;
    dispatch
}

#[derive(Clone, Debug, Default)]
struct TcpProtocolV2HostBatchSetGraphStats {
    graph_pool_leases: usize,
    graph_pool_fixed_buffer_bytes: usize,
    graph_pool_active_rows: usize,
    graph_pool_active_routes: usize,
    graph_pool_active_expert_tiles: usize,
    graph_pool_bucket_rows: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HostBatchRoundtripTransport {
    Tcp,
    VerbsHost,
}

impl HostBatchRoundtripTransport {
    async fn roundtrip(
        self,
        addr: SocketAddr,
        request: &ExpertProtocolV2Request,
        config: TcpTransportConfig,
    ) -> Result<ExpertProtocolV2Response> {
        match self {
            Self::Tcp => tcp_protocol_v2_roundtrip(addr, request, config).await,
            Self::VerbsHost => verbs_host_protocol_v2_roundtrip(addr, request, config).await,
        }
    }
}

impl From<&ExpertGraphHostBatchSetLease> for TcpProtocolV2HostBatchSetGraphStats {
    fn from(lease: &ExpertGraphHostBatchSetLease) -> Self {
        Self {
            graph_pool_leases: lease.num_hosts(),
            graph_pool_fixed_buffer_bytes: lease.total_fixed_buffer_bytes,
            graph_pool_active_rows: lease.active_counts.rows,
            graph_pool_active_routes: lease.active_counts.routes,
            graph_pool_active_expert_tiles: lease.active_counts.expert_tiles,
            graph_pool_bucket_rows: lease.bucket_rows(),
        }
    }
}

async fn protocol_v2_host_batch_set_bf16_dispatch_inner(
    transport: HostBatchRoundtripTransport,
    set: &ExpertHostBatchSet,
    global_hidden_payload: &[u8],
    targets: &[TcpProtocolV2HostBatchTarget],
    request_id_base: u64,
    config: TcpTransportConfig,
    graph_stats: Option<TcpProtocolV2HostBatchSetGraphStats>,
) -> Result<TcpProtocolV2HostBatchSetDispatch> {
    let output_dim = set
        .batches
        .first()
        .map(|batch| batch.hidden_dim)
        .context("ProtocolV2 host-batch-set dispatch requires at least one host batch")?;

    let dispatches = set
        .batches
        .iter()
        .enumerate()
        .map(|(host_index, host_batch)| {
            let addr = target_addr(targets, &host_batch.host)?;
            let compact_hidden =
                host_batch.compact_hidden_payload(global_hidden_payload, set.global_row_count)?;
            let request = ExpertProtocolV2Request::from_expert_host_batch(
                request_id_base + host_index as u64,
                host_batch,
                compact_hidden,
            )?;
            let request_wire_bytes = request.wire_stats().wire_bytes;
            let config = config.clone();
            Ok(async move {
                let response = transport.roundtrip(addr, &request, config).await?;
                let response_wire_bytes = response.wire_stats().wire_bytes;
                let response_executor_id = response.header.executor_id;
                let rows = response_bf16_rows(&response, host_batch.num_rows(), output_dim)?;
                Ok::<_, anyhow::Error>((
                    request_wire_bytes,
                    response_wire_bytes,
                    response_executor_id,
                    rows,
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let dispatches = try_join_all(dispatches).await?;
    let request_wire_bytes = dispatches
        .iter()
        .map(|(request_wire_bytes, _, _, _)| *request_wire_bytes)
        .sum::<usize>();
    let response_wire_bytes = dispatches
        .iter()
        .map(|(_, response_wire_bytes, _, _)| *response_wire_bytes)
        .sum::<usize>();
    let response_executor_ids = dispatches
        .iter()
        .map(|(_, _, response_executor_id, _)| *response_executor_id)
        .collect::<Vec<_>>();
    let mut partial_outputs_bf16_by_host = Vec::with_capacity(dispatches.len());
    let mut partial_outputs_by_host = Vec::with_capacity(dispatches.len());
    for (_, _, _, rows) in dispatches {
        partial_outputs_bf16_by_host.push(rows.compact_payload);
        partial_outputs_by_host.push(rows.rows);
    }

    let accumulation = set.accumulate_partial_outputs_f32(&partial_outputs_by_host, output_dim)?;
    let output_checksum = accumulation
        .values
        .iter()
        .map(|value| *value as f64)
        .sum::<f64>();
    let graph_stats = graph_stats.unwrap_or_default();
    let stats = TcpProtocolV2HostBatchSetDispatchStats {
        hosts: set.num_hosts(),
        global_rows: set.global_row_count,
        host_rows: set.host_row_count(),
        routes: set.route_count(),
        output_dim,
        output_values: accumulation.values.len(),
        request_wire_bytes,
        response_wire_bytes,
        response_executor_ids,
        contribution_counts: accumulation.contribution_counts.clone(),
        output_checksum,
        graph_pool_leases: graph_stats.graph_pool_leases,
        graph_pool_fixed_buffer_bytes: graph_stats.graph_pool_fixed_buffer_bytes,
        graph_pool_active_rows: graph_stats.graph_pool_active_rows,
        graph_pool_active_routes: graph_stats.graph_pool_active_routes,
        graph_pool_active_expert_tiles: graph_stats.graph_pool_active_expert_tiles,
        graph_pool_bucket_rows: graph_stats.graph_pool_bucket_rows,
    };

    Ok(TcpProtocolV2HostBatchSetDispatch {
        accumulation,
        partial_outputs_bf16_by_host,
        stats,
    })
}

struct PersistentHostBatchDispatchJob {
    host_index: usize,
    request_wire_bytes: usize,
    expected_rows: usize,
    output_dim: usize,
    request: ExpertProtocolV2Request,
}

struct PersistentHostBatchStreamDispatchJob {
    host_index: usize,
    rail_index: usize,
    stream_id: usize,
    request_wire_bytes: usize,
    requests: Vec<ExpertProtocolV2Request>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PayloadContributionCounts {
    Include,
    Omit,
}

async fn tcp_protocol_v2_host_batch_set_bf16_dispatch_persistent_inner(
    set: &ExpertHostBatchSet,
    global_hidden_payload: &[u8],
    request_id_base: u64,
    targets: &[TcpProtocolV2HostBatchTarget],
    clients: &mut [TcpProtocolV2PersistentClient],
) -> Result<TcpProtocolV2HostBatchSetDispatch> {
    anyhow::ensure!(
        clients.len() == targets.len(),
        "ProtocolV2 persistent client count {} did not match target count {}",
        clients.len(),
        targets.len()
    );
    let output_dim = set.batches.first().map(|batch| batch.hidden_dim).context(
        "persistent ProtocolV2 host-batch-set dispatch requires at least one host batch",
    )?;
    let hidden_bytes_per_row = set
        .batches
        .first()
        .map(|batch| batch.hidden_bytes_per_row)
        .expect("non-empty set has first batch");
    anyhow::ensure!(
        global_hidden_payload.len() == set.global_row_count * hidden_bytes_per_row,
        "persistent ProtocolV2 host-batch-set hidden payload bytes {} did not match global rows {} * row bytes {}",
        global_hidden_payload.len(),
        set.global_row_count,
        hidden_bytes_per_row
    );
    let mut jobs_by_client = (0..clients.len()).map(|_| None).collect::<Vec<_>>();
    for (host_index, host_batch) in set.batches.iter().enumerate() {
        let client_index = target_index(targets, &host_batch.host)?;
        if jobs_by_client[client_index].is_some() {
            bail!(
                "persistent ProtocolV2 host-batch-set has duplicate target batch for host {}",
                host_batch.host
            );
        }
        let compact_hidden =
            host_batch.compact_hidden_payload(global_hidden_payload, set.global_row_count)?;
        let request = ExpertProtocolV2Request::from_expert_host_batch(
            request_id_base + host_index as u64,
            host_batch,
            compact_hidden,
        )?;
        let request_wire_bytes = request.wire_stats().wire_bytes;
        jobs_by_client[client_index] = Some(PersistentHostBatchDispatchJob {
            host_index,
            request_wire_bytes,
            expected_rows: host_batch.num_rows(),
            output_dim,
            request,
        });
    }

    let dispatches = clients
        .iter_mut()
        .enumerate()
        .filter_map(|(client_index, client)| {
            let job = jobs_by_client[client_index].take()?;
            Some(async move {
                let response = client.roundtrip(&job.request).await?;
                let response_wire_bytes = response.wire_stats().wire_bytes;
                let response_executor_id = response.header.executor_id;
                let rows = response_bf16_rows(&response, job.expected_rows, job.output_dim)?;
                Ok::<_, anyhow::Error>((
                    job.host_index,
                    job.request_wire_bytes,
                    response_wire_bytes,
                    response_executor_id,
                    rows,
                ))
            })
        })
        .collect::<Vec<_>>();
    let mut dispatches = try_join_all(dispatches).await?;
    dispatches.sort_by_key(|(host_index, _, _, _, _)| *host_index);
    let request_wire_bytes = dispatches
        .iter()
        .map(|(_, request_wire_bytes, _, _, _)| *request_wire_bytes)
        .sum::<usize>();
    let response_wire_bytes = dispatches
        .iter()
        .map(|(_, _, response_wire_bytes, _, _)| *response_wire_bytes)
        .sum::<usize>();
    let response_executor_ids = dispatches
        .iter()
        .map(|(_, _, _, response_executor_id, _)| *response_executor_id)
        .collect::<Vec<_>>();
    let mut partial_outputs_bf16_by_host = Vec::with_capacity(dispatches.len());
    let mut partial_outputs_by_host = Vec::with_capacity(dispatches.len());
    for (_, _, _, _, rows) in dispatches {
        partial_outputs_bf16_by_host.push(rows.compact_payload);
        partial_outputs_by_host.push(rows.rows);
    }

    let accumulation = set.accumulate_partial_outputs_f32(&partial_outputs_by_host, output_dim)?;
    let output_checksum = accumulation
        .values
        .iter()
        .map(|value| *value as f64)
        .sum::<f64>();
    let stats = TcpProtocolV2HostBatchSetDispatchStats {
        hosts: set.num_hosts(),
        global_rows: set.global_row_count,
        host_rows: set.host_row_count(),
        routes: set.route_count(),
        output_dim,
        output_values: accumulation.values.len(),
        request_wire_bytes,
        response_wire_bytes,
        response_executor_ids,
        contribution_counts: accumulation.contribution_counts.clone(),
        output_checksum,
        graph_pool_leases: 0,
        graph_pool_fixed_buffer_bytes: 0,
        graph_pool_active_rows: 0,
        graph_pool_active_routes: 0,
        graph_pool_active_expert_tiles: 0,
        graph_pool_bucket_rows: Vec::new(),
    };

    Ok(TcpProtocolV2HostBatchSetDispatch {
        accumulation,
        partial_outputs_bf16_by_host,
        stats,
    })
}

async fn tcp_protocol_v2_host_batch_set_bf16_payload_dispatch_persistent_inner(
    set: &ExpertHostBatchSet,
    global_hidden_payload: &[u8],
    request_id_base: u64,
    targets: &[TcpProtocolV2HostBatchTarget],
    clients: &mut [TcpProtocolV2PersistentClient],
    contribution_counts: PayloadContributionCounts,
) -> Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch> {
    anyhow::ensure!(
        clients.len() == targets.len(),
        "ProtocolV2 persistent payload client count {} did not match target count {}",
        clients.len(),
        targets.len()
    );
    let output_dim = set.batches.first().map(|batch| batch.hidden_dim).context(
        "persistent ProtocolV2 host-batch-set payload dispatch requires at least one host batch",
    )?;
    let hidden_bytes_per_row = set
        .batches
        .first()
        .map(|batch| batch.hidden_bytes_per_row)
        .expect("non-empty set has first batch");
    anyhow::ensure!(
        global_hidden_payload.len() == set.global_row_count * hidden_bytes_per_row,
        "persistent ProtocolV2 host-batch-set payload hidden payload bytes {} did not match global rows {} * row bytes {}",
        global_hidden_payload.len(),
        set.global_row_count,
        hidden_bytes_per_row
    );
    set.reconstruction_plan
        .validate_for_batches(&set.batches, set.global_row_count)
        .context("validating persistent ProtocolV2 host-batch-set payload reconstruction plan")?;
    let mut jobs_by_client = (0..clients.len()).map(|_| None).collect::<Vec<_>>();
    for (host_index, host_batch) in set.batches.iter().enumerate() {
        let client_index = target_index(targets, &host_batch.host)?;
        if jobs_by_client[client_index].is_some() {
            bail!(
                "persistent ProtocolV2 host-batch-set payload has duplicate target batch for host {}",
                host_batch.host
            );
        }
        maybe_log_expert_queue_plan("tcp", request_id_base, host_index, host_batch);
        let compact_hidden =
            host_batch.compact_hidden_payload(global_hidden_payload, set.global_row_count)?;
        let request = ExpertProtocolV2Request::from_expert_host_batch(
            request_id_base + host_index as u64,
            host_batch,
            compact_hidden,
        )?;
        let request_wire_bytes = request.wire_stats().wire_bytes;
        jobs_by_client[client_index] = Some(PersistentHostBatchDispatchJob {
            host_index,
            request_wire_bytes,
            expected_rows: host_batch.num_rows(),
            output_dim,
            request,
        });
    }

    let dispatches = clients
        .iter_mut()
        .enumerate()
        .filter_map(|(client_index, client)| {
            let job = jobs_by_client[client_index].take()?;
            Some(async move {
                let response = client.roundtrip_response_view(&job.request).await?;
                let response_wire_bytes = response.wire_stats().wire_bytes;
                let response_executor_id = response.header.executor_id;
                let compact_payload = response_bf16_compact_payload_from_view(
                    &response,
                    job.expected_rows,
                    job.output_dim,
                )?;
                Ok::<_, anyhow::Error>((
                    job.host_index,
                    job.request_wire_bytes,
                    response_wire_bytes,
                    response_executor_id,
                    compact_payload,
                ))
            })
        })
        .collect::<Vec<_>>();
    let mut dispatches = try_join_all(dispatches).await?;
    dispatches.sort_by_key(|(host_index, _, _, _, _)| *host_index);
    let request_wire_bytes = dispatches
        .iter()
        .map(|(_, request_wire_bytes, _, _, _)| *request_wire_bytes)
        .sum::<usize>();
    let response_wire_bytes = dispatches
        .iter()
        .map(|(_, _, response_wire_bytes, _, _)| *response_wire_bytes)
        .sum::<usize>();
    let response_executor_ids = dispatches
        .iter()
        .map(|(_, _, _, response_executor_id, _)| *response_executor_id)
        .collect::<Vec<_>>();
    let partial_outputs_bf16_by_host = dispatches
        .into_iter()
        .map(|(_, _, _, _, compact_payload)| compact_payload)
        .collect::<Vec<_>>();
    let contribution_counts = match contribution_counts {
        PayloadContributionCounts::Include => reconstruction_contribution_counts(set)?,
        PayloadContributionCounts::Omit => Vec::new(),
    };
    let output_values = set
        .global_row_count
        .checked_mul(output_dim)
        .context("persistent ProtocolV2 host-batch-set payload output value count overflow")?;
    let stats = TcpProtocolV2HostBatchSetDispatchStats {
        hosts: set.num_hosts(),
        global_rows: set.global_row_count,
        host_rows: set.host_row_count(),
        routes: set.route_count(),
        output_dim,
        output_values,
        request_wire_bytes,
        response_wire_bytes,
        response_executor_ids,
        contribution_counts,
        output_checksum: 0.0,
        graph_pool_leases: 0,
        graph_pool_fixed_buffer_bytes: 0,
        graph_pool_active_rows: 0,
        graph_pool_active_routes: 0,
        graph_pool_active_expert_tiles: 0,
        graph_pool_bucket_rows: Vec::new(),
    };

    Ok(TcpProtocolV2HostBatchSetBf16PayloadDispatch {
        partial_outputs_bf16_by_host,
        global_row_indices_by_host: global_row_indices_by_host(set),
        completed_global_row_slices: vec![(0..set.global_row_count).collect()],
        stats,
    })
}

async fn verbs_host_protocol_v2_host_batch_set_bf16_dispatch_persistent_inner(
    set: &ExpertHostBatchSet,
    global_hidden_payload: &[u8],
    request_id_base: u64,
    targets: &[TcpProtocolV2HostBatchTarget],
    clients: &[VerbsHostProtocolV2PersistentClient],
) -> Result<TcpProtocolV2HostBatchSetDispatch> {
    anyhow::ensure!(
        clients.len() == targets.len(),
        "ProtocolV2 persistent verbs-host client count {} did not match target count {}",
        clients.len(),
        targets.len()
    );
    let output_dim = set.batches.first().map(|batch| batch.hidden_dim).context(
        "persistent verbs-host ProtocolV2 host-batch-set dispatch requires at least one host batch",
    )?;
    let hidden_bytes_per_row = set
        .batches
        .first()
        .map(|batch| batch.hidden_bytes_per_row)
        .expect("non-empty set has first batch");
    anyhow::ensure!(
        global_hidden_payload.len() == set.global_row_count * hidden_bytes_per_row,
        "persistent verbs-host ProtocolV2 host-batch-set hidden payload bytes {} did not match global rows {} * row bytes {}",
        global_hidden_payload.len(),
        set.global_row_count,
        hidden_bytes_per_row
    );
    let mut jobs_by_client = (0..clients.len()).map(|_| None).collect::<Vec<_>>();
    for (host_index, host_batch) in set.batches.iter().enumerate() {
        let client_index = target_index(targets, &host_batch.host)?;
        if jobs_by_client[client_index].is_some() {
            bail!(
                "persistent verbs-host ProtocolV2 host-batch-set has duplicate target batch for host {}",
                host_batch.host
            );
        }
        let compact_hidden =
            host_batch.compact_hidden_payload(global_hidden_payload, set.global_row_count)?;
        let request = ExpertProtocolV2Request::from_expert_host_batch(
            request_id_base + host_index as u64,
            host_batch,
            compact_hidden,
        )?;
        let request_wire_bytes = request.wire_stats().wire_bytes;
        jobs_by_client[client_index] = Some(PersistentHostBatchDispatchJob {
            host_index,
            request_wire_bytes,
            expected_rows: host_batch.num_rows(),
            output_dim,
            request,
        });
    }

    let dispatches = clients
        .iter()
        .enumerate()
        .filter_map(|(client_index, client)| {
            let job = jobs_by_client[client_index].take()?;
            Some(async move {
                let response = client.roundtrip(&job.request).await?;
                let response_wire_bytes = response.wire_stats().wire_bytes;
                let response_executor_id = response.header.executor_id;
                let rows = response_bf16_rows(&response, job.expected_rows, job.output_dim)?;
                Ok::<_, anyhow::Error>((
                    job.host_index,
                    job.request_wire_bytes,
                    response_wire_bytes,
                    response_executor_id,
                    rows,
                ))
            })
        })
        .collect::<Vec<_>>();
    let mut dispatches = try_join_all(dispatches).await?;
    dispatches.sort_by_key(|(host_index, _, _, _, _)| *host_index);
    let request_wire_bytes = dispatches
        .iter()
        .map(|(_, request_wire_bytes, _, _, _)| *request_wire_bytes)
        .sum::<usize>();
    let response_wire_bytes = dispatches
        .iter()
        .map(|(_, _, response_wire_bytes, _, _)| *response_wire_bytes)
        .sum::<usize>();
    let response_executor_ids = dispatches
        .iter()
        .map(|(_, _, _, response_executor_id, _)| *response_executor_id)
        .collect::<Vec<_>>();
    let mut partial_outputs_bf16_by_host = Vec::with_capacity(dispatches.len());
    let mut partial_outputs_by_host = Vec::with_capacity(dispatches.len());
    for (_, _, _, _, rows) in dispatches {
        partial_outputs_bf16_by_host.push(rows.compact_payload);
        partial_outputs_by_host.push(rows.rows);
    }

    let accumulation = set.accumulate_partial_outputs_f32(&partial_outputs_by_host, output_dim)?;
    let output_checksum = accumulation
        .values
        .iter()
        .map(|value| *value as f64)
        .sum::<f64>();
    let stats = TcpProtocolV2HostBatchSetDispatchStats {
        hosts: set.num_hosts(),
        global_rows: set.global_row_count,
        host_rows: set.host_row_count(),
        routes: set.route_count(),
        output_dim,
        output_values: accumulation.values.len(),
        request_wire_bytes,
        response_wire_bytes,
        response_executor_ids,
        contribution_counts: accumulation.contribution_counts.clone(),
        output_checksum,
        graph_pool_leases: 0,
        graph_pool_fixed_buffer_bytes: 0,
        graph_pool_active_rows: 0,
        graph_pool_active_routes: 0,
        graph_pool_active_expert_tiles: 0,
        graph_pool_bucket_rows: Vec::new(),
    };

    Ok(TcpProtocolV2HostBatchSetDispatch {
        accumulation,
        partial_outputs_bf16_by_host,
        stats,
    })
}

async fn verbs_host_protocol_v2_host_batch_set_bf16_payload_dispatch_persistent_inner(
    set: &ExpertHostBatchSet,
    global_hidden_payload: &[u8],
    request_id_base: u64,
    targets: &[TcpProtocolV2HostBatchTarget],
    clients: &[VerbsHostProtocolV2PersistentClient],
    additional_clients: &[Vec<VerbsHostProtocolV2PersistentClient>],
    stripe_min_rows: usize,
    contribution_counts: PayloadContributionCounts,
) -> Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch> {
    let stream = verbs_host_protocol_v2_host_batch_set_bf16_payload_dispatch_persistent_stream(
        set,
        global_hidden_payload,
        request_id_base,
        targets,
        clients,
        additional_clients,
        stripe_min_rows,
        contribution_counts,
        ExpertV2Dtype::Bf16,
        None,
        false,
        false,
        None,
    )
    .await?;
    let VerbsHostProtocolV2HostBatchSetPayloadStreamDispatch { chunks, stats } = stream;
    let completed_global_row_slices = chunks
        .iter()
        .filter_map(|chunk| {
            (!chunk.completed_global_row_indices.is_empty())
                .then(|| chunk.completed_global_row_indices.clone())
        })
        .collect();
    let (partial_outputs_bf16_by_host, global_row_indices_by_host) = chunks
        .into_iter()
        .map(|chunk| (chunk.partial_output.into_vec(), chunk.global_row_indices))
        .unzip();
    Ok(TcpProtocolV2HostBatchSetBf16PayloadDispatch {
        partial_outputs_bf16_by_host,
        global_row_indices_by_host,
        completed_global_row_slices,
        stats,
    })
}

async fn verbs_host_protocol_v2_host_batch_set_bf16_payload_dispatch_persistent_streaming_inner(
    set: &ExpertHostBatchSet,
    global_hidden_payload: &[u8],
    request_id_base: u64,
    targets: &[TcpProtocolV2HostBatchTarget],
    clients: &[VerbsHostProtocolV2PersistentClient],
    additional_clients: &[Vec<VerbsHostProtocolV2PersistentClient>],
    stripe_min_rows: usize,
    contribution_counts: PayloadContributionCounts,
    response_dtype: ExpertV2Dtype,
    reduced_root_host_index: Option<usize>,
    owner_fanout: bool,
    row_sharded_reduction: bool,
    chunk_tx: &std_mpsc::Sender<VerbsHostProtocolV2HostBatchSetBf16PayloadChunk>,
) -> Result<TcpProtocolV2HostBatchSetDispatchStats> {
    let stream = verbs_host_protocol_v2_host_batch_set_bf16_payload_dispatch_persistent_stream(
        set,
        global_hidden_payload,
        request_id_base,
        targets,
        clients,
        additional_clients,
        stripe_min_rows,
        contribution_counts,
        response_dtype,
        reduced_root_host_index,
        owner_fanout,
        row_sharded_reduction,
        Some(chunk_tx),
    )
    .await?;
    debug_assert!(stream.chunks.is_empty());
    Ok(stream.stats)
}

async fn verbs_host_protocol_v2_reduced_identity_payload_dispatch_persistent(
    request: ExpertProtocolV2Request,
    response_dtype: ExpertV2Dtype,
    reduced_root_host_index: usize,
    logical_routes: usize,
    clients: &[&VerbsHostProtocolV2PersistentClient],
    stripe_min_rows: usize,
    chunk_tx: std_mpsc::Sender<VerbsHostProtocolV2HostBatchSetBf16PayloadChunk>,
) -> Result<TcpProtocolV2HostBatchSetDispatchStats> {
    anyhow::ensure!(
        matches!(
            response_dtype,
            ExpertV2Dtype::Bf16 | ExpertV2Dtype::Fp8E4m3RowScaled | ExpertV2Dtype::Nvfp4E2m1Fp8E4m3
        ),
        "direct Spark-owner response dtype {response_dtype:?} is not supported"
    );
    let global_rows = request.header.row_count as usize;
    let output_dim = request.header.hidden_dim as usize;
    let mut partitions =
        partition_protocol_v2_request_for_rails(request, clients.len(), stripe_min_rows)?;
    assign_spark_collective_request_ids(&mut partitions, reduced_root_host_index)?;
    let request_wire_bytes = partitions
        .iter()
        .map(|partition| partition.request.wire_stats().wire_bytes)
        .sum();
    let response_stream_maps = partitions
        .iter()
        .map(|partition| VerbsHostProtocolV2ResponseStreamMap {
            host_index: reduced_root_host_index,
            local_row_indices: partition.local_row_indices.clone(),
        })
        .collect::<Vec<_>>();
    let (response_chunk_tx, mut response_chunk_rx) = tokio::sync::mpsc::unbounded_channel();
    let dispatches = partitions
        .into_iter()
        .enumerate()
        .map(|(stream_id, partition)| {
            let client = clients[stream_id % clients.len()];
            let response_chunk_tx = response_chunk_tx.clone();
            async move {
                client
                    .roundtrip_response_chunks(partition.request, stream_id, response_chunk_tx)
                    .await
            }
        })
        .collect::<Vec<_>>();
    drop(response_chunk_tx);
    let response_stats = try_join_all(dispatches)
        .await
        .context("dispatching striped direct Spark-owner reduced response")?;
    let response_stats = merge_verbs_host_response_stream_stats(&response_stats)?;

    let mut completed_rows = vec![false; global_rows];
    while let Some(response_chunk) = response_chunk_rx.recv().await {
        if let Some(chunk) = reduced_identity_payload_chunk(
            response_chunk,
            response_dtype,
            &response_stream_maps,
            output_dim,
            &mut completed_rows,
        )? {
            chunk_tx
                .send(chunk)
                .context("forwarding direct Spark-owner reduced response")?;
        }
    }
    anyhow::ensure!(
        completed_rows.iter().all(|completed| *completed),
        "direct Spark-owner response did not complete every request row"
    );
    reduced_identity_payload_dispatch_stats(
        global_rows,
        output_dim,
        logical_routes,
        request_wire_bytes,
        response_stats,
    )
}

fn reduced_identity_payload_chunk(
    response_chunk: VerbsHostProtocolV2ResponseChunk,
    response_dtype: ExpertV2Dtype,
    response_stream_maps: &[VerbsHostProtocolV2ResponseStreamMap],
    output_dim: usize,
    completed_rows: &mut [bool],
) -> Result<Option<VerbsHostProtocolV2HostBatchSetBf16PayloadChunk>> {
    let VerbsHostProtocolV2ResponseChunk {
        stream_id,
        header,
        row_indices,
        partial_output_payload,
        wire_bytes: _,
    } = response_chunk;
    anyhow::ensure!(
        header.output_dtype == response_dtype,
        "direct Spark-owner response dtype {:?} did not match requested {response_dtype:?}",
        header.output_dtype
    );
    let chunk_rows = header.row_count as usize;
    let local_row_indices = row_indices.unwrap_or_else(|| (0..chunk_rows as u32).collect());
    anyhow::ensure!(
        local_row_indices.len() == chunk_rows,
        "direct Spark-owner response row index count {} did not match chunk rows {chunk_rows}",
        local_row_indices.len()
    );
    let partial_output =
        response_compact_stream_payload(&header, partial_output_payload, chunk_rows, output_dim)?;
    if chunk_rows == 0 {
        return Ok(None);
    }
    let (host_index, global_row_indices) =
        map_verbs_host_response_stream_rows(response_stream_maps, stream_id, &local_row_indices)?;
    let global_rows = completed_rows.len();
    for global_row_index in &global_row_indices {
        let completed = completed_rows.get_mut(*global_row_index).with_context(|| {
            format!(
                "direct Spark-owner response row {global_row_index} exceeds {} request rows",
                global_rows
            )
        })?;
        anyhow::ensure!(
            !*completed,
            "direct Spark-owner response row {global_row_index} completed twice"
        );
        *completed = true;
    }
    let output_row_stride_bytes = header.output_dtype.row_bytes(output_dim)?;
    Ok(Some(VerbsHostProtocolV2HostBatchSetBf16PayloadChunk {
        host_index,
        partial_output,
        output_dtype: header.output_dtype,
        output_row_stride_bytes,
        completed_global_row_indices: global_row_indices.clone(),
        global_row_indices,
    }))
}

fn map_verbs_host_response_stream_rows(
    response_stream_maps: &[VerbsHostProtocolV2ResponseStreamMap],
    stream_id: usize,
    local_row_indices: &[u32],
) -> Result<(usize, Vec<usize>)> {
    let stream_map = response_stream_maps
        .get(stream_id)
        .with_context(|| format!("verbs-host response stream {stream_id} is out of bounds"))?;
    let rows = local_row_indices
        .iter()
        .map(|local_row_index| {
            stream_map
                .local_row_indices
                .get(*local_row_index as usize)
                .copied()
                .with_context(|| {
                    format!(
                        "verbs-host response local row {local_row_index} is out of bounds for stream {stream_id}"
                    )
                })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((stream_map.host_index, rows))
}

fn merge_verbs_host_response_stream_stats(
    stats: &[VerbsHostProtocolV2ResponseStreamStats],
) -> Result<VerbsHostProtocolV2ResponseStreamStats> {
    let first = stats
        .first()
        .context("verbs-host striped dispatch produced no response stats")?;
    let mut merged = VerbsHostProtocolV2ResponseStreamStats {
        response_frames: 0,
        response_wire_bytes: 0,
        response_executor_id: first.response_executor_id,
    };
    for stats in stats {
        anyhow::ensure!(
            stats.response_executor_id == merged.response_executor_id,
            "verbs-host striped response executor changed from {} to {}",
            merged.response_executor_id,
            stats.response_executor_id
        );
        merged.response_frames = merged
            .response_frames
            .checked_add(stats.response_frames)
            .context("verbs-host striped response frame count overflow")?;
        merged.response_wire_bytes = merged
            .response_wire_bytes
            .checked_add(stats.response_wire_bytes)
            .context("verbs-host striped response byte count overflow")?;
    }
    Ok(merged)
}

fn reduced_identity_payload_dispatch_stats(
    global_rows: usize,
    output_dim: usize,
    logical_routes: usize,
    request_wire_bytes: usize,
    response_stats: VerbsHostProtocolV2ResponseStreamStats,
) -> Result<TcpProtocolV2HostBatchSetDispatchStats> {
    let output_values = global_rows
        .checked_mul(output_dim)
        .context("direct Spark-owner output value count overflow")?;
    Ok(TcpProtocolV2HostBatchSetDispatchStats {
        hosts: 1,
        global_rows,
        host_rows: global_rows,
        routes: logical_routes,
        output_dim,
        output_values,
        request_wire_bytes,
        response_wire_bytes: response_stats.response_wire_bytes,
        response_executor_ids: vec![response_stats.response_executor_id],
        contribution_counts: Vec::new(),
        output_checksum: 0.0,
        graph_pool_leases: 0,
        graph_pool_fixed_buffer_bytes: 0,
        graph_pool_active_rows: 0,
        graph_pool_active_routes: 0,
        graph_pool_active_expert_tiles: 0,
        graph_pool_bucket_rows: Vec::new(),
    })
}

struct VerbsHostProtocolV2HostBatchSetPayloadStreamDispatch {
    chunks: Vec<VerbsHostProtocolV2HostBatchSetBf16PayloadChunk>,
    stats: TcpProtocolV2HostBatchSetDispatchStats,
}

async fn verbs_host_protocol_v2_host_batch_set_bf16_payload_dispatch_persistent_stream(
    set: &ExpertHostBatchSet,
    global_hidden_payload: &[u8],
    request_id_base: u64,
    targets: &[TcpProtocolV2HostBatchTarget],
    clients: &[VerbsHostProtocolV2PersistentClient],
    additional_clients: &[Vec<VerbsHostProtocolV2PersistentClient>],
    stripe_min_rows: usize,
    contribution_count_policy: PayloadContributionCounts,
    response_dtype: ExpertV2Dtype,
    reduced_root_host_index: Option<usize>,
    owner_fanout: bool,
    row_sharded_reduction: bool,
    streaming_chunk_tx: Option<&std_mpsc::Sender<VerbsHostProtocolV2HostBatchSetBf16PayloadChunk>>,
) -> Result<VerbsHostProtocolV2HostBatchSetPayloadStreamDispatch> {
    let timing_enabled = protocol_v2_host_batch_set_timing_enabled();
    let total_started = timing_enabled.then(Instant::now);
    let validate_started = timing_enabled.then(Instant::now);
    anyhow::ensure!(
        clients.len() == targets.len(),
        "ProtocolV2 persistent verbs-host payload client count {} did not match target count {}",
        clients.len(),
        targets.len()
    );
    anyhow::ensure!(
        additional_clients.len() == targets.len(),
        "ProtocolV2 persistent verbs-host additional client count {} did not match target count {}",
        additional_clients.len(),
        targets.len()
    );
    let output_dim = set.batches.first().map(|batch| batch.hidden_dim).context(
        "persistent verbs-host ProtocolV2 host-batch-set payload dispatch requires at least one host batch",
    )?;
    anyhow::ensure!(
        matches!(
            response_dtype,
            ExpertV2Dtype::Bf16 | ExpertV2Dtype::Fp8E4m3RowScaled | ExpertV2Dtype::Nvfp4E2m1Fp8E4m3
        ),
        "persistent verbs-host ProtocolV2 response dtype {response_dtype:?} is not supported"
    );
    anyhow::ensure!(
        response_dtype == ExpertV2Dtype::Bf16 || streaming_chunk_tx.is_some(),
        "low-precision responses require streamed verbs-host dispatch"
    );
    if let Some(root_host_index) = reduced_root_host_index {
        anyhow::ensure!(
            root_host_index < set.num_hosts(),
            "Spark-reduced root host index {root_host_index} exceeds {} hosts",
            set.num_hosts()
        );
        anyhow::ensure!(
            streaming_chunk_tx.is_some(),
            "Spark-reduced responses require streamed verbs-host dispatch"
        );
    }
    anyhow::ensure!(
        !owner_fanout || reduced_root_host_index.is_some(),
        "Spark owner fanout requires a reduced root host"
    );
    anyhow::ensure!(
        !row_sharded_reduction || (reduced_root_host_index.is_some() && !owner_fanout),
        "row-sharded Spark reduction requires non-owner Spark reduction"
    );
    let hidden_bytes_per_row = set
        .batches
        .first()
        .map(|batch| batch.hidden_bytes_per_row)
        .expect("non-empty set has first batch");
    anyhow::ensure!(
        global_hidden_payload.len() == set.global_row_count * hidden_bytes_per_row,
        "persistent verbs-host ProtocolV2 host-batch-set payload hidden payload bytes {} did not match global rows {} * row bytes {}",
        global_hidden_payload.len(),
        set.global_row_count,
        hidden_bytes_per_row
    );
    set.reconstruction_plan
        .validate_for_batches(&set.batches, set.global_row_count)
        .context("validating persistent verbs-host ProtocolV2 host-batch-set payload reconstruction plan")?;
    let validate_ms = elapsed_ms_optional(validate_started);
    let prep_started = timing_enabled.then(Instant::now);
    let mut target_lookup_ms = 0.0_f64;
    let mut compact_hidden_ms = 0.0_f64;
    let mut request_build_ms = 0.0_f64;
    let stream_ingress_rows = protocol_v2_stream_ingress_rows()?;
    let striped_row_sharded_reduction =
        row_sharded_reduction && protocol_v2_verbs_host_stripe_spark_reduction_enabled();
    let can_stripe_requests =
        reduced_root_host_index.is_none() || owner_fanout || striped_row_sharded_reduction;
    let mut shared_collective_request_ids = None::<Vec<u64>>;
    let mut jobs_by_client = (0..clients.len()).map(|_| Vec::new()).collect::<Vec<_>>();
    let mut response_stream_maps = Vec::new();
    for (host_index, host_batch) in set.batches.iter().enumerate() {
        if owner_fanout && Some(host_index) != reduced_root_host_index {
            continue;
        }
        let target_started = timing_enabled.then(Instant::now);
        let client_index = target_index(targets, &host_batch.host)?;
        target_lookup_ms += elapsed_ms_optional(target_started);
        if !jobs_by_client[client_index].is_empty() {
            bail!(
                "persistent verbs-host ProtocolV2 host-batch-set payload has duplicate target batch for host {}",
                host_batch.host
            );
        }
        maybe_log_expert_queue_plan("verbs-host", request_id_base, host_index, host_batch);
        let compact_started = timing_enabled.then(Instant::now);
        let compact_hidden =
            host_batch.compact_hidden_payload(global_hidden_payload, set.global_row_count)?;
        compact_hidden_ms += elapsed_ms_optional(compact_started);
        let request_started = timing_enabled.then(Instant::now);
        let request = ExpertProtocolV2Request::from_expert_host_batch(
            request_id_base + host_index as u64,
            host_batch,
            compact_hidden,
        )?;
        let request = if row_sharded_reduction {
            request.with_spark_row_sharded_reduction()
        } else {
            request
        };
        let rail_count = if can_stripe_requests {
            1 + additional_clients[client_index].len()
        } else {
            1
        };
        let mut partitions =
            partition_protocol_v2_request_for_rails(request, rail_count, stripe_min_rows)?;
        if owner_fanout {
            assign_spark_collective_request_ids(&mut partitions, host_index)?;
        } else if striped_row_sharded_reduction {
            mark_striped_spark_collective_parts(&mut partitions, rail_count)?;
            if let Some(request_ids) = &shared_collective_request_ids {
                anyhow::ensure!(
                    request_ids.len() == partitions.len(),
                    "row-sharded Spark collective host {host_index} produced {} rail partitions; expected {}",
                    partitions.len(),
                    request_ids.len()
                );
            } else {
                shared_collective_request_ids =
                    Some(reserve_spark_collective_request_ids(partitions.len())?);
            }
            assign_shared_spark_collective_request_ids(
                &mut partitions,
                host_index,
                shared_collective_request_ids
                    .as_deref()
                    .expect("row-sharded collective request IDs were reserved"),
            )?;
        }
        let striped = rail_count > 1 && partitions.len() > 1;
        for (partition_index, partition) in partitions.into_iter().enumerate() {
            let rail_index = partition_index % rail_count;
            let stream_id = response_stream_maps.len();
            response_stream_maps.push(VerbsHostProtocolV2ResponseStreamMap {
                host_index,
                local_row_indices: partition.local_row_indices,
            });
            let partition_rows = partition.request.header.row_count as usize;
            let requests = match stream_ingress_rows {
                Some(chunk_rows) if !striped && partition_rows > chunk_rows => {
                    streamed_ingress_requests(
                        partition.request,
                        response_dtype,
                        chunk_rows,
                        reduced_root_host_index.is_some(),
                        row_sharded_reduction,
                    )?
                }
                _ => {
                    let request = request_with_response_dtype(partition.request, response_dtype);
                    vec![if row_sharded_reduction {
                        request.with_spark_row_sharded_reduction()
                    } else if reduced_root_host_index.is_some() {
                        request.with_spark_reduction()
                    } else {
                        request
                    }]
                }
            };
            if reduced_root_host_index.is_some() {
                debug_assert!(requests
                    .iter()
                    .all(ExpertProtocolV2Request::spark_reduction_enabled));
            }
            if row_sharded_reduction {
                debug_assert!(requests
                    .iter()
                    .all(ExpertProtocolV2Request::spark_row_sharded_reduction_enabled));
            }
            let request_wire_bytes = requests.iter().try_fold(0_usize, |bytes, request| {
                bytes
                    .checked_add(request.wire_stats().wire_bytes)
                    .context("streamed-ingress request byte count overflow")
            })?;
            jobs_by_client[client_index].push(PersistentHostBatchStreamDispatchJob {
                host_index,
                rail_index,
                stream_id,
                request_wire_bytes,
                requests,
            });
        }
        request_build_ms += elapsed_ms_optional(request_started);
    }
    let prep_ms = elapsed_ms_optional(prep_started);

    let (response_chunk_tx, mut response_chunk_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut dispatches = Vec::new();
    for (client_index, jobs) in jobs_by_client.into_iter().enumerate() {
        for job in jobs {
            let client = if job.rail_index == 0 {
                &clients[client_index]
            } else {
                &additional_clients[client_index][job.rail_index - 1]
            };
            let response_chunk_tx = response_chunk_tx.clone();
            dispatches.push(async move {
                let stats = if job.requests.len() == 1 {
                    client
                        .roundtrip_response_chunks(
                            job.requests
                                .into_iter()
                                .next()
                                .expect("single request exists"),
                            job.stream_id,
                            response_chunk_tx,
                        )
                        .await?
                } else {
                    client
                        .roundtrip_response_chunk_sequence(
                            job.requests,
                            job.stream_id,
                            response_chunk_tx,
                        )
                        .await?
                };
                Ok::<_, anyhow::Error>((
                    job.host_index,
                    job.rail_index,
                    job.request_wire_bytes,
                    stats,
                ))
            });
        }
    }
    drop(response_chunk_tx);
    let join_started = timing_enabled.then(Instant::now);
    let dispatch_future = try_join_all(dispatches);
    tokio::pin!(dispatch_future);
    let mut dispatches = None;
    let mut completion_tracker = HostBatchResponseCompletionTracker::new(
        set,
        reduced_root_host_index,
        row_sharded_reduction,
    )?;
    let mut chunks = Vec::new();
    let mut response_parse_payload_ms = 0.0_f64;
    let mut response_chunk_channel_open = true;
    while dispatches.is_none() {
        tokio::select! {
            response_chunk = response_chunk_rx.recv(), if response_chunk_channel_open => {
                if let Some(response_chunk) = response_chunk {
                    let parse_started = timing_enabled.then(Instant::now);
                    let chunk = host_batch_payload_chunk_from_response(
                        set,
                        &response_stream_maps,
                        output_dim,
                        response_dtype,
                        &mut completion_tracker,
                        response_chunk,
                    )?;
                    response_parse_payload_ms += elapsed_ms_optional(parse_started);
                    if let Some(chunk) = chunk {
                        forward_or_collect_host_batch_payload_chunk(
                            streaming_chunk_tx,
                            &mut chunks,
                            chunk,
                        )?;
                    }
                } else {
                    response_chunk_channel_open = false;
                }
            }
            result = &mut dispatch_future => {
                dispatches = Some(result?);
            }
        }
    }
    while let Some(response_chunk) = response_chunk_rx.recv().await {
        let parse_started = timing_enabled.then(Instant::now);
        let chunk = host_batch_payload_chunk_from_response(
            set,
            &response_stream_maps,
            output_dim,
            response_dtype,
            &mut completion_tracker,
            response_chunk,
        )?;
        response_parse_payload_ms += elapsed_ms_optional(parse_started);
        if let Some(chunk) = chunk {
            forward_or_collect_host_batch_payload_chunk(streaming_chunk_tx, &mut chunks, chunk)?;
        }
    }
    completion_tracker.finish()?;
    let mut dispatches = dispatches.expect("dispatch future completed above");
    let join_ms = elapsed_ms_optional(join_started);
    let sort_started = timing_enabled.then(Instant::now);
    dispatches.sort_by_key(|(host_index, rail_index, _, _)| (*host_index, *rail_index));
    let sort_ms = elapsed_ms_optional(sort_started);
    let stats_started = timing_enabled.then(Instant::now);
    let request_wire_bytes = dispatches
        .iter()
        .map(|(_, _, request_wire_bytes, _)| *request_wire_bytes)
        .sum::<usize>();
    let response_wire_bytes = dispatches
        .iter()
        .map(|(_, _, _, stats)| stats.response_wire_bytes)
        .sum::<usize>();
    let mut response_executor_ids_by_host = BTreeMap::new();
    for (host_index, _, _, stats) in &dispatches {
        if let Some(expected) =
            response_executor_ids_by_host.insert(*host_index, stats.response_executor_id)
        {
            anyhow::ensure!(
                stats.response_executor_id == expected,
                "verbs-host striped response executor for host {host_index} changed from {expected} to {}",
                stats.response_executor_id
            );
        }
    }
    let response_executor_ids = response_executor_ids_by_host
        .values()
        .copied()
        .collect::<Vec<_>>();
    let logical_hosts = response_executor_ids_by_host.len();
    let contribution_counts = match contribution_count_policy {
        PayloadContributionCounts::Include if owner_fanout => {
            let root_host_index = reduced_root_host_index
                .expect("owner fanout validation requires a reduced root host");
            let mut counts = vec![0_usize; set.global_row_count];
            for row in &set.reconstruction_plan.host_row_maps[root_host_index].global_row_indices {
                counts[*row] += 1;
            }
            counts
        }
        PayloadContributionCounts::Include => reconstruction_contribution_counts(set)?,
        PayloadContributionCounts::Omit => Vec::new(),
    };
    let output_values = set.global_row_count.checked_mul(output_dim).context(
        "persistent verbs-host ProtocolV2 host-batch-set payload output value count overflow",
    )?;
    let host_rows = if owner_fanout {
        let root = &set.batches[reduced_root_host_index
            .expect("owner fanout validation requires a reduced root host")];
        root.num_rows()
    } else {
        set.host_row_count()
    };
    // Keep the scheduler's logical TP4 route accounting when one owner
    // proxies the three physical peer submissions.
    let routes = set.route_count();
    let stats = TcpProtocolV2HostBatchSetDispatchStats {
        hosts: logical_hosts,
        global_rows: set.global_row_count,
        host_rows,
        routes,
        output_dim,
        output_values,
        request_wire_bytes,
        response_wire_bytes,
        response_executor_ids,
        contribution_counts,
        output_checksum: 0.0,
        graph_pool_leases: 0,
        graph_pool_fixed_buffer_bytes: 0,
        graph_pool_active_rows: 0,
        graph_pool_active_routes: 0,
        graph_pool_active_expert_tiles: 0,
        graph_pool_bucket_rows: Vec::new(),
    };
    let stats_ms = elapsed_ms_optional(stats_started);
    if timing_enabled {
        eprintln!(
            "protocol_v2_verbs_host_batch_set_payload_timing request_id_base={} hosts={} global_rows={} host_rows={} routes={} request_wire_bytes={} response_wire_bytes={} validate_ms={:.3} prep_ms={:.3} target_lookup_ms={:.3} compact_hidden_ms={:.3} request_build_ms={:.3} join_ms={:.3} response_parse_payload_ms={:.3} sort_ms={:.3} stats_ms={:.3} total_ms={:.3}",
            request_id_base,
            logical_hosts,
            set.global_row_count,
            host_rows,
            routes,
            request_wire_bytes,
            response_wire_bytes,
            validate_ms,
            prep_ms,
            target_lookup_ms,
            compact_hidden_ms,
            request_build_ms,
            join_ms,
            response_parse_payload_ms,
            sort_ms,
            stats_ms,
            elapsed_ms_optional(total_started)
        );
    }

    Ok(VerbsHostProtocolV2HostBatchSetPayloadStreamDispatch { chunks, stats })
}

struct HostBatchResponseCompletionTracker {
    remaining_host_contributions: Vec<usize>,
}

fn streamed_ingress_requests(
    request: ExpertProtocolV2Request,
    response_dtype: ExpertV2Dtype,
    chunk_rows: usize,
    spark_reduction: bool,
    row_sharded_reduction: bool,
) -> Result<Vec<ExpertProtocolV2Request>> {
    anyhow::ensure!(
        chunk_rows > 0 && chunk_rows <= PROTOCOL_V2_STREAM_MAX_ROUTE_GROUP_ROWS,
        "ProtocolV2 streamed-ingress rows must be in 1..={PROTOCOL_V2_STREAM_MAX_ROUTE_GROUP_ROWS}, got {chunk_rows}"
    );
    anyhow::ensure!(
        !request.precompile_warmup_enabled(),
        "ProtocolV2 precompile request cannot use streamed ingress"
    );
    let debug_checksum = request.debug_checksum_enabled();
    let header = request.header.clone();
    let rows = request.rows;
    let routes = request.routes;
    let hidden_payload = request.hidden_payload;
    let completion_entries = routes
        .iter()
        .map(|route| CompletionRoutePlanEntry {
            row_index: route.row_index as usize,
            expert_id: route.expert_id as usize,
            // Routed GLM experts have one intermediate shape. Keeping this
            // key neutral lets the Spark verify its resident projection shape.
            intermediate_rows: 0,
        })
        .collect::<Vec<_>>();
    let completion = plan_completion_first_routes(
        &completion_entries,
        rows.len(),
        PROTOCOL_V2_STREAM_MAX_ROUTE_GROUP_ROWS,
    )?;
    let stream_plan =
        ExpertProtocolV2StreamPlan::from_completion_first(rows.len(), routes.len(), &completion)?;
    stream_plan.validate_against_request(&rows, &routes)?;
    let plan_payload = stream_plan.encode()?;
    let mut plan = ExpertProtocolV2Request::new_stream_plan_with_hidden_stride(
        header.request_id,
        header.placement_version,
        header.layer_id,
        header.hidden_dim,
        header.hidden_dtype,
        header.hidden_row_stride_bytes,
        rows,
        routes,
        plan_payload,
    )?;
    plan = request_with_response_dtype(plan, response_dtype);
    if row_sharded_reduction {
        plan = plan.with_spark_row_sharded_reduction();
    } else if spark_reduction {
        plan = plan.with_spark_reduction();
    }
    if debug_checksum {
        plan = plan.with_debug_checksum();
    }

    let stride = header.hidden_row_stride_bytes as usize;
    let mut requests =
        Vec::with_capacity(1 + completion.activation_row_order.len().div_ceil(chunk_rows));
    requests.push(plan);
    for (chunk_index, activation_rows) in completion
        .activation_row_order
        .chunks(chunk_rows)
        .enumerate()
    {
        let mut payload = Vec::with_capacity(activation_rows.len() * stride);
        for row_index in activation_rows {
            let start = row_index
                .checked_mul(stride)
                .context("ProtocolV2 streamed-ingress hidden row offset overflow")?;
            let end = start
                .checked_add(stride)
                .context("ProtocolV2 streamed-ingress hidden row range overflow")?;
            payload.extend_from_slice(hidden_payload.get(start..end).with_context(|| {
                format!("ProtocolV2 streamed-ingress hidden row {row_index} is out of range")
            })?);
        }
        let row_offset = chunk_index
            .checked_mul(chunk_rows)
            .context("ProtocolV2 streamed-ingress row offset overflow")?;
        let final_frame =
            row_offset + activation_rows.len() == completion.activation_row_order.len();
        let mut data = ExpertProtocolV2Request::new_stream_data(
            header.request_id,
            header.placement_version,
            header.layer_id,
            header.hidden_dim,
            header.hidden_dtype,
            header.hidden_row_stride_bytes,
            u32::try_from(row_offset)
                .context("ProtocolV2 streamed-ingress row offset exceeds u32")?,
            u32::try_from(activation_rows.len())
                .context("ProtocolV2 streamed-ingress chunk row count exceeds u32")?,
            payload,
            final_frame,
        )?;
        data = request_with_response_dtype(data, response_dtype);
        if row_sharded_reduction {
            data = data.with_spark_row_sharded_reduction();
        } else if spark_reduction {
            data = data.with_spark_reduction();
        }
        if debug_checksum {
            data = data.with_debug_checksum();
        }
        requests.push(data);
    }
    anyhow::ensure!(
        requests
            .last()
            .is_some_and(ExpertProtocolV2Request::stream_final_enabled),
        "ProtocolV2 streamed-ingress request sequence has no final data frame"
    );
    Ok(requests)
}

fn request_with_response_dtype(
    request: ExpertProtocolV2Request,
    response_dtype: ExpertV2Dtype,
) -> ExpertProtocolV2Request {
    match response_dtype {
        ExpertV2Dtype::Fp8E4m3RowScaled => request.with_fp8_e4m3_row_scaled_response(),
        ExpertV2Dtype::Nvfp4E2m1Fp8E4m3 => request.with_nvfp4_e2m1_fp8_e4m3_response(),
        _ => request,
    }
}

pub fn protocol_v2_stream_ingress_rows() -> Result<Option<usize>> {
    let Ok(value) = env::var(PROTOCOL_V2_STREAM_INGRESS_ROWS_ENV) else {
        return Ok(None);
    };
    let value = value.trim();
    if value.is_empty() || value == "0" {
        return Ok(None);
    }
    let rows = value
        .parse::<usize>()
        .with_context(|| format!("invalid {PROTOCOL_V2_STREAM_INGRESS_ROWS_ENV} value {value}"))?;
    anyhow::ensure!(
        rows <= PROTOCOL_V2_STREAM_MAX_ROUTE_GROUP_ROWS,
        "{PROTOCOL_V2_STREAM_INGRESS_ROWS_ENV} must be at most {PROTOCOL_V2_STREAM_MAX_ROUTE_GROUP_ROWS}, got {rows}"
    );
    Ok(Some(rows))
}

impl HostBatchResponseCompletionTracker {
    fn new(
        set: &ExpertHostBatchSet,
        reduced_root_host_index: Option<usize>,
        row_sharded_reduction: bool,
    ) -> Result<Self> {
        let remaining_host_contributions = if row_sharded_reduction {
            anyhow::ensure!(
                reduced_root_host_index.is_some(),
                "row-sharded reduction requires a reduction root marker"
            );
            vec![1_usize; set.global_row_count]
        } else if let Some(root_host_index) = reduced_root_host_index {
            let root_map = set
                .reconstruction_plan
                .host_row_maps
                .get(root_host_index)
                .with_context(|| {
                    format!("Spark-reduced root host index {root_host_index} is out of bounds")
                })?;
            let mut counts = vec![0_usize; set.global_row_count];
            for row in &root_map.global_row_indices {
                let count = counts.get_mut(*row).with_context(|| {
                    format!("Spark-reduced root global row {row} is out of bounds")
                })?;
                *count += 1;
            }
            counts
        } else {
            reconstruction_contribution_counts(set)?
        };
        if let Some(row_index) = remaining_host_contributions
            .iter()
            .position(|count| *count == 0)
        {
            bail!("ProtocolV2 host-batch reconstruction row {row_index} has no host contributions");
        }
        Ok(Self {
            remaining_host_contributions,
        })
    }

    fn accept(&mut self, global_row_indices: &[usize]) -> Result<Vec<usize>> {
        let mut completed = Vec::new();
        for global_row_index in global_row_indices {
            let remaining = self
                .remaining_host_contributions
                .get_mut(*global_row_index)
                .with_context(|| {
                    format!(
                        "ProtocolV2 streamed response global row {global_row_index} out of bounds"
                    )
                })?;
            if *remaining == 0 {
                bail!("ProtocolV2 streamed response global row {global_row_index} completed twice");
            }
            *remaining -= 1;
            if *remaining == 0 {
                completed.push(*global_row_index);
            }
        }
        Ok(completed)
    }

    fn finish(self) -> Result<()> {
        if let Some((row_index, remaining)) = self
            .remaining_host_contributions
            .iter()
            .enumerate()
            .find(|(_, remaining)| **remaining != 0)
        {
            bail!(
                "ProtocolV2 streamed response row {row_index} still expected {remaining} host contributions"
            );
        }
        Ok(())
    }
}

fn host_batch_payload_chunk_from_response(
    set: &ExpertHostBatchSet,
    response_stream_maps: &[VerbsHostProtocolV2ResponseStreamMap],
    output_dim: usize,
    expected_output_dtype: ExpertV2Dtype,
    completion_tracker: &mut HostBatchResponseCompletionTracker,
    response_chunk: VerbsHostProtocolV2ResponseChunk,
) -> Result<Option<VerbsHostProtocolV2HostBatchSetBf16PayloadChunk>> {
    let VerbsHostProtocolV2ResponseChunk {
        stream_id,
        header,
        row_indices,
        partial_output_payload,
        wire_bytes: _,
    } = response_chunk;
    anyhow::ensure!(
        header.output_dtype == expected_output_dtype,
        "ProtocolV2 streamed response dtype {:?} did not match requested {:?}",
        header.output_dtype,
        expected_output_dtype
    );
    let chunk_rows = header.row_count as usize;
    let local_row_indices = row_indices.unwrap_or_else(|| (0..chunk_rows as u32).collect());
    anyhow::ensure!(
        local_row_indices.len() == chunk_rows,
        "ProtocolV2 streamed response row index count {} did not match chunk rows {chunk_rows}",
        local_row_indices.len()
    );
    if chunk_rows == 0 {
        response_compact_stream_payload(&header, partial_output_payload, 0, output_dim)?;
        return Ok(None);
    }
    let (host_index, host_local_row_indices) =
        map_verbs_host_response_stream_rows(response_stream_maps, stream_id, &local_row_indices)?;
    let host_row_map = set
        .reconstruction_plan
        .host_row_maps
        .get(host_index)
        .with_context(|| {
            format!("ProtocolV2 streamed response host index {host_index} out of bounds")
        })?;
    let mut global_row_indices = Vec::with_capacity(chunk_rows);
    for host_local_row_index in host_local_row_indices {
        let global_row_index = host_row_map
            .global_row_indices
            .get(host_local_row_index)
            .with_context(|| {
                format!(
                    "ProtocolV2 streamed response host-local row {host_local_row_index} out of bounds for host {}",
                    host_row_map.host
                )
            })?;
        global_row_indices.push(*global_row_index);
    }
    let partial_output =
        response_compact_stream_payload(&header, partial_output_payload, chunk_rows, output_dim)?;
    let compact_output_row_stride_bytes = header.output_dtype.row_bytes(output_dim)?;
    let completed_global_row_indices = completion_tracker.accept(&global_row_indices)?;
    Ok(Some(VerbsHostProtocolV2HostBatchSetBf16PayloadChunk {
        host_index,
        partial_output,
        output_dtype: header.output_dtype,
        output_row_stride_bytes: compact_output_row_stride_bytes,
        global_row_indices,
        completed_global_row_indices,
    }))
}

fn forward_or_collect_host_batch_payload_chunk(
    streaming_chunk_tx: Option<&std_mpsc::Sender<VerbsHostProtocolV2HostBatchSetBf16PayloadChunk>>,
    chunks: &mut Vec<VerbsHostProtocolV2HostBatchSetBf16PayloadChunk>,
    chunk: VerbsHostProtocolV2HostBatchSetBf16PayloadChunk,
) -> Result<()> {
    if let Some(chunk_tx) = streaming_chunk_tx {
        chunk_tx
            .send(chunk)
            .context("forwarding completed ProtocolV2 host-batch response chunk")?;
    } else {
        chunks.push(chunk);
    }
    Ok(())
}

async fn protocol_v2_host_batch_set_bf16_payload_dispatch_inner(
    transport: HostBatchRoundtripTransport,
    set: &ExpertHostBatchSet,
    global_hidden_payload: &[u8],
    targets: &[TcpProtocolV2HostBatchTarget],
    request_id_base: u64,
    config: TcpTransportConfig,
    graph_stats: Option<TcpProtocolV2HostBatchSetGraphStats>,
    contribution_counts: PayloadContributionCounts,
) -> Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch> {
    let output_dim =
        set.batches.first().map(|batch| batch.hidden_dim).context(
            "ProtocolV2 host-batch-set payload dispatch requires at least one host batch",
        )?;
    set.reconstruction_plan
        .validate_for_batches(&set.batches, set.global_row_count)
        .context("validating ProtocolV2 host-batch-set payload reconstruction plan")?;

    let dispatches = set
        .batches
        .iter()
        .enumerate()
        .map(|(host_index, host_batch)| {
            let addr = target_addr(targets, &host_batch.host)?;
            let compact_hidden =
                host_batch.compact_hidden_payload(global_hidden_payload, set.global_row_count)?;
            let request = ExpertProtocolV2Request::from_expert_host_batch(
                request_id_base + host_index as u64,
                host_batch,
                compact_hidden,
            )?;
            let request_wire_bytes = request.wire_stats().wire_bytes;
            let config = config.clone();
            Ok(async move {
                let response = transport.roundtrip(addr, &request, config).await?;
                let response_wire_bytes = response.wire_stats().wire_bytes;
                let response_executor_id = response.header.executor_id;
                let compact_payload = response_bf16_compact_payload_owned(
                    response,
                    host_batch.num_rows(),
                    output_dim,
                )?;
                Ok::<_, anyhow::Error>((
                    request_wire_bytes,
                    response_wire_bytes,
                    response_executor_id,
                    compact_payload,
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let dispatches = try_join_all(dispatches).await?;
    let request_wire_bytes = dispatches
        .iter()
        .map(|(request_wire_bytes, _, _, _)| *request_wire_bytes)
        .sum::<usize>();
    let response_wire_bytes = dispatches
        .iter()
        .map(|(_, response_wire_bytes, _, _)| *response_wire_bytes)
        .sum::<usize>();
    let response_executor_ids = dispatches
        .iter()
        .map(|(_, _, response_executor_id, _)| *response_executor_id)
        .collect::<Vec<_>>();
    let partial_outputs_bf16_by_host = dispatches
        .into_iter()
        .map(|(_, _, _, compact_payload)| compact_payload)
        .collect::<Vec<_>>();
    let contribution_counts = match contribution_counts {
        PayloadContributionCounts::Include => reconstruction_contribution_counts(set)?,
        PayloadContributionCounts::Omit => Vec::new(),
    };
    let output_values = set
        .global_row_count
        .checked_mul(output_dim)
        .context("ProtocolV2 host-batch-set payload output value count overflow")?;
    let graph_stats = graph_stats.unwrap_or_default();
    let stats = TcpProtocolV2HostBatchSetDispatchStats {
        hosts: set.num_hosts(),
        global_rows: set.global_row_count,
        host_rows: set.host_row_count(),
        routes: set.route_count(),
        output_dim,
        output_values,
        request_wire_bytes,
        response_wire_bytes,
        response_executor_ids,
        contribution_counts,
        output_checksum: 0.0,
        graph_pool_leases: graph_stats.graph_pool_leases,
        graph_pool_fixed_buffer_bytes: graph_stats.graph_pool_fixed_buffer_bytes,
        graph_pool_active_rows: graph_stats.graph_pool_active_rows,
        graph_pool_active_routes: graph_stats.graph_pool_active_routes,
        graph_pool_active_expert_tiles: graph_stats.graph_pool_active_expert_tiles,
        graph_pool_bucket_rows: graph_stats.graph_pool_bucket_rows,
    };

    Ok(TcpProtocolV2HostBatchSetBf16PayloadDispatch {
        partial_outputs_bf16_by_host,
        global_row_indices_by_host: global_row_indices_by_host(set),
        completed_global_row_slices: vec![(0..set.global_row_count).collect()],
        stats,
    })
}

fn global_row_indices_by_host(set: &ExpertHostBatchSet) -> Vec<Vec<usize>> {
    set.reconstruction_plan
        .host_row_maps
        .iter()
        .map(|host_map| host_map.global_row_indices.clone())
        .collect()
}

#[derive(Debug, Clone, PartialEq)]
struct ResponseBf16Rows {
    rows: Vec<Vec<f32>>,
    compact_payload: Vec<u8>,
}

fn target_addr(targets: &[TcpProtocolV2HostBatchTarget], host: &str) -> Result<SocketAddr> {
    targets
        .iter()
        .find(|target| target.host == host)
        .map(|target| target.addr)
        .with_context(|| format!("missing ProtocolV2 host-batch target for host {host}"))
}

fn target_index(targets: &[TcpProtocolV2HostBatchTarget], host: &str) -> Result<usize> {
    targets
        .iter()
        .position(|target| target.host == host)
        .with_context(|| format!("missing ProtocolV2 host-batch target for host {host}"))
}

fn response_bf16_rows(
    response: &ExpertProtocolV2Response,
    expected_rows: usize,
    expected_output_dim: usize,
) -> Result<ResponseBf16Rows> {
    let (logical_row_bytes, stride) =
        validate_bf16_response_payload(response, expected_rows, expected_output_dim)?;
    let mut rows = Vec::with_capacity(expected_rows);
    let mut compact_payload = Vec::with_capacity(expected_rows * logical_row_bytes);
    for row_index in 0..expected_rows {
        let row = bf16_response_row(response, row_index, stride, logical_row_bytes)?;
        compact_payload.extend_from_slice(row);
        rows.push(bf16_values(row));
    }
    Ok(ResponseBf16Rows {
        rows,
        compact_payload,
    })
}

fn response_bf16_compact_payload(
    response: &ExpertProtocolV2Response,
    expected_rows: usize,
    expected_output_dim: usize,
) -> Result<Vec<u8>> {
    let (logical_row_bytes, stride) =
        validate_bf16_response_payload(response, expected_rows, expected_output_dim)?;
    let mut compact_payload = Vec::with_capacity(expected_rows * logical_row_bytes);
    for row_index in 0..expected_rows {
        compact_payload.extend_from_slice(bf16_response_row(
            response,
            row_index,
            stride,
            logical_row_bytes,
        )?);
    }
    Ok(compact_payload)
}

fn response_bf16_compact_payload_owned(
    response: ExpertProtocolV2Response,
    expected_rows: usize,
    expected_output_dim: usize,
) -> Result<Vec<u8>> {
    let (logical_row_bytes, stride) =
        validate_bf16_response_payload(&response, expected_rows, expected_output_dim)?;
    let compact_bytes = expected_rows
        .checked_mul(logical_row_bytes)
        .context("ProtocolV2 host-batch compact response byte count overflow")?;
    if stride == logical_row_bytes && response.partial_output_payload.len() == compact_bytes {
        return Ok(response.partial_output_payload);
    }
    response_bf16_compact_payload(&response, expected_rows, expected_output_dim)
}

fn response_compact_stream_payload(
    header: &ExpertProtocolV2ResponseHeader,
    payload: VerbsHostProtocolV2ResponsePayload,
    expected_rows: usize,
    expected_output_dim: usize,
) -> Result<VerbsHostProtocolV2ResponsePayload> {
    if header.status != ExpertProtocolV2Status::Ok {
        bail!(
            "ProtocolV2 streamed host-batch response status {:?} is not ok",
            header.status
        );
    }
    if !matches!(
        header.output_dtype,
        ExpertV2Dtype::Bf16 | ExpertV2Dtype::Fp8E4m3RowScaled | ExpertV2Dtype::Nvfp4E2m1Fp8E4m3
    ) {
        bail!(
            "ProtocolV2 streamed host-batch response dtype {:?} is not BF16, row-scaled FP8 E4M3, or NVFP4",
            header.output_dtype
        );
    }
    if header.row_count as usize != expected_rows {
        bail!(
            "ProtocolV2 streamed host-batch response rows {} did not match expected {expected_rows}",
            header.row_count
        );
    }
    if header.output_dim as usize != expected_output_dim {
        bail!(
            "ProtocolV2 streamed host-batch response output dim {} did not match expected {expected_output_dim}",
            header.output_dim
        );
    }
    let logical_row_bytes = header
        .output_dtype
        .row_bytes(expected_output_dim)
        .context("ProtocolV2 streamed host-batch response row byte count overflow")?;
    let stride = header.output_row_stride_bytes as usize;
    if stride < logical_row_bytes {
        bail!(
            "ProtocolV2 streamed host-batch response row stride {stride} is smaller than logical row bytes {logical_row_bytes}"
        );
    }
    let strided_bytes = expected_rows
        .checked_mul(stride)
        .context("ProtocolV2 streamed host-batch response byte count overflow")?;
    anyhow::ensure!(
        payload.as_ref().len() == strided_bytes,
        "ProtocolV2 streamed host-batch response payload bytes {} did not match expected {strided_bytes}",
        payload.as_ref().len()
    );
    if stride == logical_row_bytes {
        return Ok(payload);
    }

    let compact_bytes = expected_rows
        .checked_mul(logical_row_bytes)
        .context("ProtocolV2 streamed host-batch compact response byte count overflow")?;
    let mut compact_payload = Vec::with_capacity(compact_bytes);
    for row_index in 0..expected_rows {
        let row_start = row_index
            .checked_mul(stride)
            .context("ProtocolV2 streamed host-batch row offset overflow")?;
        compact_payload
            .extend_from_slice(&payload.as_ref()[row_start..row_start + logical_row_bytes]);
    }
    Ok(VerbsHostProtocolV2ResponsePayload::from_owned(
        compact_payload,
    ))
}

fn response_bf16_compact_payload_from_view(
    response: &ExpertProtocolV2ResponseView<'_>,
    expected_rows: usize,
    expected_output_dim: usize,
) -> Result<Vec<u8>> {
    let (logical_row_bytes, stride) =
        validate_bf16_response_view_payload(response, expected_rows, expected_output_dim)?;
    let compact_bytes = expected_rows
        .checked_mul(logical_row_bytes)
        .context("ProtocolV2 host-batch compact response byte count overflow")?;
    if stride == logical_row_bytes && response.partial_output_payload().len() == compact_bytes {
        return Ok(response.partial_output_payload().to_vec());
    }
    let mut compact_payload = Vec::with_capacity(compact_bytes);
    for row_index in 0..expected_rows {
        let row = response
            .partial_output_row_payload(row_index)
            .with_context(|| format!("ProtocolV2 host-batch response row {row_index} missing"))?;
        compact_payload.extend_from_slice(&row[..logical_row_bytes]);
    }
    Ok(compact_payload)
}

fn validate_bf16_response_payload(
    response: &ExpertProtocolV2Response,
    expected_rows: usize,
    expected_output_dim: usize,
) -> Result<(usize, usize)> {
    if response.header.status != ExpertProtocolV2Status::Ok {
        bail!(
            "ProtocolV2 host-batch response status {:?} is not ok",
            response.header.status
        );
    }
    if response.header.output_dtype != ExpertV2Dtype::Bf16 {
        bail!(
            "ProtocolV2 host-batch response dtype {:?} is not BF16",
            response.header.output_dtype
        );
    }
    if response.header.row_count as usize != expected_rows {
        bail!(
            "ProtocolV2 host-batch response rows {} did not match expected {expected_rows}",
            response.header.row_count
        );
    }
    if response.header.output_dim as usize != expected_output_dim {
        bail!(
            "ProtocolV2 host-batch response output dim {} did not match expected {expected_output_dim}",
            response.header.output_dim
        );
    }

    let output_dim = response.header.output_dim as usize;
    let logical_row_bytes = output_dim
        .checked_mul(ExpertV2Dtype::Bf16.bytes_per_element())
        .context("ProtocolV2 host-batch BF16 row byte count overflow")?;
    let stride = response.header.output_row_stride_bytes as usize;
    if stride < logical_row_bytes {
        bail!(
            "ProtocolV2 host-batch response row stride {stride} is smaller than logical row bytes {logical_row_bytes}"
        );
    }
    Ok((logical_row_bytes, stride))
}

fn validate_bf16_response_view_payload(
    response: &ExpertProtocolV2ResponseView<'_>,
    expected_rows: usize,
    expected_output_dim: usize,
) -> Result<(usize, usize)> {
    if response.header.status != ExpertProtocolV2Status::Ok {
        bail!(
            "ProtocolV2 host-batch response status {:?} is not ok",
            response.header.status
        );
    }
    if response.header.output_dtype != ExpertV2Dtype::Bf16 {
        bail!(
            "ProtocolV2 host-batch response dtype {:?} is not BF16",
            response.header.output_dtype
        );
    }
    if response.header.row_count as usize != expected_rows {
        bail!(
            "ProtocolV2 host-batch response rows {} did not match expected {expected_rows}",
            response.header.row_count
        );
    }
    if response.header.output_dim as usize != expected_output_dim {
        bail!(
            "ProtocolV2 host-batch response output dim {} did not match expected {expected_output_dim}",
            response.header.output_dim
        );
    }

    let output_dim = response.header.output_dim as usize;
    let logical_row_bytes = output_dim
        .checked_mul(ExpertV2Dtype::Bf16.bytes_per_element())
        .context("ProtocolV2 host-batch BF16 row byte count overflow")?;
    let stride = response.header.output_row_stride_bytes as usize;
    if stride < logical_row_bytes {
        bail!(
            "ProtocolV2 host-batch response row stride {stride} is smaller than logical row bytes {logical_row_bytes}"
        );
    }
    Ok((logical_row_bytes, stride))
}

fn bf16_response_row(
    response: &ExpertProtocolV2Response,
    row_index: usize,
    stride: usize,
    logical_row_bytes: usize,
) -> Result<&[u8]> {
    let start = row_index
        .checked_mul(stride)
        .context("ProtocolV2 host-batch response row offset overflow")?;
    let end = start
        .checked_add(logical_row_bytes)
        .context("ProtocolV2 host-batch response row end overflow")?;
    response
        .partial_output_payload
        .get(start..end)
        .with_context(|| format!("ProtocolV2 host-batch response row {row_index} missing"))
}

fn reconstruction_contribution_counts(set: &ExpertHostBatchSet) -> Result<Vec<usize>> {
    let mut contribution_counts = vec![0_usize; set.global_row_count];
    for host_map in &set.reconstruction_plan.host_row_maps {
        for row_index in &host_map.global_row_indices {
            let count = contribution_counts.get_mut(*row_index).with_context(|| {
                format!("ProtocolV2 host-batch reconstruction row index {row_index} out of bounds")
            })?;
            *count += 1;
        }
    }
    Ok(contribution_counts)
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct ExpertQueueGroupStats {
    rows: usize,
    routes: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct ExpertQueuePlanStats {
    rows: usize,
    routes: usize,
    experts: usize,
    single_expert_rows: usize,
    multi_expert_rows: usize,
    empty_rows: usize,
    expert_only_rows: usize,
    expert_only_extra_rows: usize,
    min_expert_rows: usize,
    p50_expert_rows: usize,
    max_expert_rows: usize,
    min_expert_routes: usize,
    p50_expert_routes: usize,
    max_expert_routes: usize,
    least_hot_expert: Option<usize>,
    least_hot_expert_rows: usize,
    hottest_expert: Option<usize>,
    hottest_expert_rows: usize,
    expert_route_counts_by_id: Vec<(usize, usize)>,
}

fn maybe_log_expert_queue_plan(
    transport: &str,
    request_id_base: u64,
    host_index: usize,
    host_batch: &ExpertHostBatch,
) {
    if !protocol_v2_expert_queue_stats_enabled() {
        return;
    }
    let stats = expert_queue_plan_stats(host_batch);
    let expert_only_row_factor = if stats.rows == 0 {
        0.0
    } else {
        stats.expert_only_rows as f64 / stats.rows as f64
    };
    let expert_route_counts = stats
        .expert_route_counts_by_id
        .iter()
        .map(|(expert_id, routes)| format!("{expert_id}:{routes}"))
        .collect::<Vec<_>>()
        .join(",");
    eprintln!(
        "protocol_v2_expert_queue_plan transport={} request_id_base={} request_id={} host={} host_index={} rows={} routes={} experts={} single_expert_rows={} multi_expert_rows={} empty_rows={} expert_only_rows={} expert_only_extra_rows={} expert_only_row_factor={:.3} expert_row_min={} expert_row_p50={} expert_row_max={} expert_route_min={} expert_route_p50={} expert_route_max={} least_hot_expert={} least_hot_rows={} hottest_expert={} hottest_rows={} expert_route_counts={}",
        transport,
        request_id_base,
        request_id_base + host_index as u64,
        host_batch.host,
        host_index,
        stats.rows,
        stats.routes,
        stats.experts,
        stats.single_expert_rows,
        stats.multi_expert_rows,
        stats.empty_rows,
        stats.expert_only_rows,
        stats.expert_only_extra_rows,
        expert_only_row_factor,
        stats.min_expert_rows,
        stats.p50_expert_rows,
        stats.max_expert_rows,
        stats.min_expert_routes,
        stats.p50_expert_routes,
        stats.max_expert_routes,
        stats
            .least_hot_expert
            .map(|expert_id| expert_id.to_string())
            .unwrap_or_else(|| "none".to_owned()),
        stats.least_hot_expert_rows,
        stats
            .hottest_expert
            .map(|expert_id| expert_id.to_string())
            .unwrap_or_else(|| "none".to_owned()),
        stats.hottest_expert_rows,
        expert_route_counts,
    );
    if host_index == 0 && protocol_v2_expert_queue_row_routes_enabled() {
        let source_kinds = host_batch
            .rows
            .iter()
            .map(|row| format!("{:?}", row.source_kind))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            .join("+");
        let row_routes = host_batch
            .rows
            .iter()
            .map(|row| {
                let route_end = row.route_offset.saturating_add(row.route_count);
                let experts = host_batch
                    .routes
                    .get(row.route_offset..route_end)
                    .unwrap_or(&[])
                    .iter()
                    .map(|route| route.expert_id.to_string())
                    .collect::<Vec<_>>()
                    .join("+");
                format!("{}:{experts}", row.global_row_index)
            })
            .collect::<Vec<_>>()
            .join(",");
        eprintln!(
            "protocol_v2_expert_queue_row_routes request_id_base={} layer_id={} rows={} source_kinds={} row_routes={}",
            request_id_base,
            host_batch.layer_id.0,
            host_batch.rows.len(),
            source_kinds,
            row_routes
        );
    }
}

fn expert_queue_plan_stats(host_batch: &ExpertHostBatch) -> ExpertQueuePlanStats {
    let mut stats = ExpertQueuePlanStats {
        rows: host_batch.rows.len(),
        routes: host_batch.routes.len(),
        ..ExpertQueuePlanStats::default()
    };
    let mut expert_groups = BTreeMap::<usize, ExpertQueueGroupStats>::new();

    for row in &host_batch.rows {
        let route_end = row.route_offset.saturating_add(row.route_count);
        let mut row_experts = Vec::<usize>::new();
        for route in host_batch
            .routes
            .get(row.route_offset..route_end)
            .unwrap_or(&[])
        {
            let group = expert_groups.entry(route.expert_id).or_default();
            group.routes += 1;
            if !row_experts.contains(&route.expert_id) {
                row_experts.push(route.expert_id);
            }
        }

        match row_experts.len() {
            0 => stats.empty_rows += 1,
            1 => stats.single_expert_rows += 1,
            _ => stats.multi_expert_rows += 1,
        }
        for expert_id in row_experts {
            expert_groups.entry(expert_id).or_default().rows += 1;
        }
    }

    stats.experts = expert_groups.len();
    stats.expert_only_rows = expert_groups.values().map(|group| group.rows).sum();
    stats.expert_only_extra_rows = stats.expert_only_rows.saturating_sub(stats.rows);
    stats.expert_route_counts_by_id = expert_groups
        .iter()
        .map(|(expert_id, group)| (*expert_id, group.routes))
        .collect();

    let mut expert_row_counts = expert_groups
        .values()
        .map(|group| group.rows)
        .collect::<Vec<_>>();
    expert_row_counts.sort_unstable();
    let mut expert_route_counts = expert_groups
        .values()
        .map(|group| group.routes)
        .collect::<Vec<_>>();
    expert_route_counts.sort_unstable();
    stats.min_expert_rows = *expert_row_counts.first().unwrap_or(&0);
    stats.p50_expert_rows = percentile_mid(&expert_row_counts);
    stats.max_expert_rows = *expert_row_counts.last().unwrap_or(&0);
    stats.min_expert_routes = *expert_route_counts.first().unwrap_or(&0);
    stats.p50_expert_routes = percentile_mid(&expert_route_counts);
    stats.max_expert_routes = *expert_route_counts.last().unwrap_or(&0);

    if let Some((expert_id, group)) = expert_groups
        .iter()
        .min_by_key(|(expert_id, group)| (group.rows, group.routes, **expert_id))
    {
        stats.least_hot_expert = Some(*expert_id);
        stats.least_hot_expert_rows = group.rows;
    }
    if let Some((expert_id, group)) = expert_groups
        .iter()
        .max_by_key(|(expert_id, group)| (group.rows, group.routes, **expert_id))
    {
        stats.hottest_expert = Some(*expert_id);
        stats.hottest_expert_rows = group.rows;
    }

    stats
}

fn percentile_mid(sorted_values: &[usize]) -> usize {
    sorted_values
        .get(sorted_values.len().saturating_sub(1) / 2)
        .copied()
        .unwrap_or(0)
}

fn protocol_v2_expert_queue_stats_enabled() -> bool {
    protocol_v2_bool_env(PROTOCOL_V2_EXPERT_QUEUE_STATS_ENV)
        && protocol_v2_expert_queue_trace_gate_open()
}

fn protocol_v2_expert_queue_row_routes_enabled() -> bool {
    protocol_v2_bool_env(PROTOCOL_V2_EXPERT_QUEUE_ROW_ROUTES_ENV)
        && protocol_v2_expert_queue_trace_gate_open()
}

fn protocol_v2_expert_queue_trace_gate_open() -> bool {
    match env::var(PROTOCOL_V2_EXPERT_QUEUE_ROW_ROUTES_GATE_FILE_ENV) {
        Ok(path) => !path.trim().is_empty() && Path::new(path.trim()).is_file(),
        Err(_) => true,
    }
}

fn protocol_v2_host_batch_set_timing_enabled() -> bool {
    protocol_v2_bool_env(PROTOCOL_V2_TCP_TIMING_ENV)
}

fn protocol_v2_bool_env(name: &str) -> bool {
    env::var(name)
        .map(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn elapsed_ms_optional(started: Option<Instant>) -> f64 {
    started.map(elapsed_ms).unwrap_or(0.0)
}

fn bf16_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            f32::from_bits((bits as u32) << 16)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glmrt_core::{
        DType, ExpertBatchRoute, ExpertHostBatchRow, GraphBucket, HostRowToGlobalRowMap, LayerId,
        PartialReconstructionPlan, PlacementVersion, PositionId, RequestId, RowSourceKind,
    };

    #[test]
    fn verbs_host_execution_lane_count_is_bounded() {
        assert_eq!(parse_verbs_host_execution_lanes(None).unwrap(), 4);
        assert_eq!(parse_verbs_host_execution_lanes(Some("1")).unwrap(), 1);
        assert_eq!(parse_verbs_host_execution_lanes(Some("4")).unwrap(), 4);
        assert_eq!(parse_verbs_host_execution_lanes(Some("8")).unwrap(), 8);
        assert!(parse_verbs_host_execution_lanes(Some("0")).is_err());
        assert!(parse_verbs_host_execution_lanes(Some("9")).is_err());
        assert!(parse_verbs_host_execution_lanes(Some("many")).is_err());
    }

    #[test]
    fn verbs_host_target_rails_keep_one_logical_host_with_two_addresses() {
        let targets = vec![TcpProtocolV2HostBatchTarget {
            host: "dodo".to_owned(),
            addr: "10.55.0.2:9100".parse().unwrap(),
        }];

        let rails = parse_verbs_host_target_rails(&targets, "dodo=10.55.0.252").unwrap();

        assert_eq!(rails.len(), 1);
        assert_eq!(rails[0].host, "dodo");
        assert_eq!(
            rails[0].addrs,
            [
                "10.55.0.2:9100".parse().unwrap(),
                "10.55.0.252:9100".parse().unwrap()
            ]
        );
    }

    #[test]
    fn verbs_host_target_rails_reject_unknown_hosts() {
        let targets = vec![TcpProtocolV2HostBatchTarget {
            host: "dodo".to_owned(),
            addr: "10.55.0.2:9100".parse().unwrap(),
        }];

        let error = parse_verbs_host_target_rails(&targets, "emu=10.55.0.251")
            .unwrap_err()
            .to_string();

        assert!(error.contains("unknown host"));
    }

    #[test]
    fn verbs_host_stripe_min_rows_is_positive() {
        assert_eq!(parse_verbs_host_stripe_min_rows(None).unwrap(), 64);
        assert_eq!(parse_verbs_host_stripe_min_rows(Some("128")).unwrap(), 128);
        assert!(parse_verbs_host_stripe_min_rows(Some("0")).is_err());
    }

    #[test]
    fn spark_collective_request_ids_are_contiguous_and_host_aligned() {
        let sequence = AtomicU64::new(7);
        let ids = reserve_spark_collective_request_ids_from(&sequence, 3).unwrap();
        let next = reserve_spark_collective_request_id_from(&sequence).unwrap();

        assert_eq!(ids.len(), 3);
        assert_eq!(ids[0] % SPARK_COLLECTIVE_REQUEST_ID_STRIDE, 0);
        assert_eq!(ids[1] - ids[0], SPARK_COLLECTIVE_REQUEST_ID_STRIDE);
        assert_eq!(ids[2] - ids[1], SPARK_COLLECTIVE_REQUEST_ID_STRIDE);
        assert_eq!(next - ids[2], SPARK_COLLECTIVE_REQUEST_ID_STRIDE);
        assert!(ids[0] >= SPARK_COLLECTIVE_REQUEST_ID_NAMESPACE);
    }

    #[test]
    fn striped_spark_collective_partitions_share_ids_across_hosts() {
        let host_batch = ExpertHostBatch {
            host: "spark0".to_owned(),
            layer_id: LayerId(3),
            placement_version: PlacementVersion::from("test"),
            hidden_dim: 4,
            hidden_bytes_per_row: 8,
            hidden_dtype: DType::Bf16,
            graph_bucket: GraphBucket::new(4),
            quantization_recipe: "test".to_owned(),
            rows: (0..4).map(|index| row(index, index, 1)).collect(),
            routes: (0..4).map(|index| route(index, index)).collect(),
        };
        let request =
            ExpertProtocolV2Request::from_expert_host_batch(99, &host_batch, (0_u8..32).collect())
                .unwrap()
                .with_spark_row_sharded_reduction();
        let mut host0 = partition_protocol_v2_request_for_rails(request.clone(), 2, 2).unwrap();
        let mut host3 = partition_protocol_v2_request_for_rails(request, 2, 2).unwrap();
        let ids = [
            SPARK_COLLECTIVE_REQUEST_ID_NAMESPACE + 11 * SPARK_COLLECTIVE_REQUEST_ID_STRIDE,
            SPARK_COLLECTIVE_REQUEST_ID_NAMESPACE + 12 * SPARK_COLLECTIVE_REQUEST_ID_STRIDE,
        ];

        mark_striped_spark_collective_parts(&mut host0, 2).unwrap();
        mark_striped_spark_collective_parts(&mut host3, 2).unwrap();
        assign_shared_spark_collective_request_ids(&mut host0, 0, &ids).unwrap();
        assign_shared_spark_collective_request_ids(&mut host3, 3, &ids).unwrap();

        assert_eq!(host0.len(), 2);
        assert_eq!(host3.len(), 2);
        for partition_index in 0..2 {
            assert_eq!(
                host0[partition_index].request.spark_collective_part_count(),
                2
            );
            assert_eq!(
                host3[partition_index].request.spark_collective_part_count(),
                2
            );
            assert_eq!(
                host0[partition_index].request.header.request_id,
                ids[partition_index]
            );
            assert_eq!(
                host3[partition_index].request.header.request_id,
                ids[partition_index] + 3
            );
            assert_eq!(
                host0[partition_index].request.header.request_id
                    - host0[partition_index].request.header.request_id % 16,
                host3[partition_index].request.header.request_id
                    - host3[partition_index].request.header.request_id % 16
            );
        }
    }

    #[test]
    fn striped_spark_collective_rejects_more_partitions_than_rails() {
        let host_batch = ExpertHostBatch {
            host: "spark0".to_owned(),
            layer_id: LayerId(3),
            placement_version: PlacementVersion::from("test"),
            hidden_dim: 4,
            hidden_bytes_per_row: 8,
            hidden_dtype: DType::Bf16,
            graph_bucket: GraphBucket::new(4),
            quantization_recipe: "test".to_owned(),
            rows: (0..4).map(|index| row(index, index, 1)).collect(),
            routes: (0..4).map(|index| route(index, index)).collect(),
        };
        let request =
            ExpertProtocolV2Request::from_expert_host_batch(99, &host_batch, (0_u8..32).collect())
                .unwrap()
                .with_spark_row_sharded_reduction();
        let mut partitions = partition_protocol_v2_request_for_rails(request, 2, 1).unwrap();
        assert_eq!(partitions.len(), 2);

        let error = mark_striped_spark_collective_parts(&mut partitions, 1)
            .unwrap_err()
            .to_string();

        assert!(error.contains("2 partitions for 1 rails"));
    }

    #[tokio::test]
    async fn execution_lanes_use_aux_for_overlap_and_reuse_primary_after_release() {
        let client = VerbsHostProtocolV2HostBatchSetPersistentClient {
            targets: Vec::new(),
            clients_by_lane: vec![Vec::new(), Vec::new()],
            additional_clients_by_lane: vec![Vec::new(), Vec::new()],
            lane_fanout_clients: Vec::new(),
            stripe_min_rows: DEFAULT_VERBS_HOST_STRIPE_MIN_ROWS,
            lane_locks: vec![
                Arc::new(tokio::sync::Mutex::new(())),
                Arc::new(tokio::sync::Mutex::new(())),
            ],
            next_lane: Arc::new(AtomicUsize::new(0)),
        };

        let (primary, primary_guard) = client.acquire_execution_lane().await;
        let (auxiliary, auxiliary_guard) = client.acquire_execution_lane().await;
        assert_eq!((primary, auxiliary), (0, 1));
        drop(primary_guard);
        drop(auxiliary_guard);

        let (serial, _serial_guard) = client.acquire_execution_lane().await;
        assert_eq!(serial, 0);
    }

    #[test]
    fn replicated_one_row_chunks_complete_after_every_host_in_arrival_order() {
        let host_count = 4;
        let output_dim = 4;
        let mut seen_hosts = vec![false; host_count];
        let mut remaining_contributions = host_count;

        for (arrival_index, host_index) in [2, 0, 3, 1].into_iter().enumerate() {
            let chunk = replicated_one_row_payload_chunk_from_response(
                host_count,
                output_dim,
                ExpertV2Dtype::Bf16,
                &mut seen_hosts,
                &mut remaining_contributions,
                VerbsHostProtocolV2ResponseChunk {
                    stream_id: host_index,
                    header: ExpertProtocolV2ResponseHeader {
                        request_id: 100 + host_index as u64,
                        placement_version: 1,
                        layer_id: 3,
                        row_count: 1,
                        output_dim: output_dim as u32,
                        output_dtype: ExpertV2Dtype::Bf16,
                        output_row_stride_bytes: (output_dim * 2) as u32,
                        output_payload_bytes: (output_dim * 2) as u64,
                        status: ExpertProtocolV2Status::Ok,
                        flags: 0,
                        executor_id: 7,
                    },
                    row_indices: None,
                    partial_output_payload: VerbsHostProtocolV2ResponsePayload::from_owned(vec![
                        host_index as u8;
                        output_dim * 2
                    ]),
                    wire_bytes: 0,
                },
            )
            .unwrap();

            assert_eq!(chunk.host_index, host_index);
            assert_eq!(chunk.global_row_indices, vec![0]);
            assert_eq!(chunk.partial_output.as_ref(), &[host_index as u8; 8]);
            assert_eq!(
                chunk.completed_global_row_indices,
                if arrival_index + 1 == host_count {
                    vec![0]
                } else {
                    Vec::new()
                }
            );
        }

        assert_eq!(remaining_contributions, 0);
        assert!(seen_hosts.into_iter().all(|seen| seen));
    }

    #[test]
    fn streamed_host_chunks_complete_rows_at_their_last_host_contribution() {
        let set = ExpertHostBatchSet {
            global_row_count: 3,
            batches: Vec::new(),
            reconstruction_plan: PartialReconstructionPlan {
                global_row_count: 3,
                host_row_maps: vec![
                    HostRowToGlobalRowMap {
                        host: "spark0".to_owned(),
                        global_row_indices: vec![0, 1, 2],
                    },
                    HostRowToGlobalRowMap {
                        host: "spark1".to_owned(),
                        global_row_indices: vec![1, 2],
                    },
                    HostRowToGlobalRowMap {
                        host: "spark2".to_owned(),
                        global_row_indices: vec![2],
                    },
                ],
            },
        };
        let mut tracker = HostBatchResponseCompletionTracker::new(&set, None, false).unwrap();

        assert!(tracker.accept(&[2]).unwrap().is_empty());
        assert_eq!(tracker.accept(&[0, 2]).unwrap(), vec![0]);
        assert_eq!(tracker.accept(&[1, 2]).unwrap(), vec![2]);
        assert_eq!(tracker.accept(&[1]).unwrap(), vec![1]);
        tracker.finish().unwrap();
    }

    #[test]
    fn streamed_host_chunk_rejects_a_contribution_after_row_completion() {
        let set = ExpertHostBatchSet {
            global_row_count: 1,
            batches: Vec::new(),
            reconstruction_plan: PartialReconstructionPlan {
                global_row_count: 1,
                host_row_maps: vec![HostRowToGlobalRowMap {
                    host: "spark0".to_owned(),
                    global_row_indices: vec![0],
                }],
            },
        };
        let mut tracker = HostBatchResponseCompletionTracker::new(&set, None, false).unwrap();

        assert_eq!(tracker.accept(&[0]).unwrap(), vec![0]);
        let error = tracker.accept(&[0]).unwrap_err().to_string();

        assert!(error.contains("completed twice"));
    }

    #[test]
    fn spark_reduced_chunks_complete_from_the_root_host_only() {
        let host_row_maps = (0..4)
            .map(|host| HostRowToGlobalRowMap {
                host: format!("spark{host}"),
                global_row_indices: vec![0, 1],
            })
            .collect();
        let set = ExpertHostBatchSet {
            global_row_count: 2,
            batches: Vec::new(),
            reconstruction_plan: PartialReconstructionPlan {
                global_row_count: 2,
                host_row_maps,
            },
        };
        let mut tracker = HostBatchResponseCompletionTracker::new(&set, Some(0), false).unwrap();

        assert_eq!(tracker.accept(&[1]).unwrap(), vec![1]);
        assert_eq!(tracker.accept(&[0]).unwrap(), vec![0]);
        tracker.finish().unwrap();
    }

    #[test]
    fn row_sharded_reduction_completes_each_uneven_partition_once() {
        let host_row_maps = (0..4)
            .map(|host| HostRowToGlobalRowMap {
                host: format!("spark{host}"),
                global_row_indices: (0..10).collect(),
            })
            .collect();
        let set = ExpertHostBatchSet {
            global_row_count: 10,
            batches: Vec::new(),
            reconstruction_plan: PartialReconstructionPlan {
                global_row_count: 10,
                host_row_maps,
            },
        };
        let mut tracker = HostBatchResponseCompletionTracker::new(&set, Some(0), true).unwrap();

        assert_eq!(tracker.accept(&[0, 1, 2]).unwrap(), vec![0, 1, 2]);
        assert_eq!(tracker.accept(&[3, 4, 5]).unwrap(), vec![3, 4, 5]);
        assert_eq!(tracker.accept(&[6, 7]).unwrap(), vec![6, 7]);
        assert_eq!(tracker.accept(&[8, 9]).unwrap(), vec![8, 9]);
        tracker.finish().unwrap();
    }

    #[test]
    fn expert_queue_plan_stats_counts_expert_group_duplication() {
        let host_batch = ExpertHostBatch {
            host: "spark0".to_owned(),
            layer_id: LayerId(3),
            placement_version: PlacementVersion::from("test"),
            hidden_dim: 4,
            hidden_bytes_per_row: 8,
            hidden_dtype: DType::Bf16,
            graph_bucket: GraphBucket::new(3),
            quantization_recipe: "test".to_owned(),
            rows: vec![row(0, 0, 2), row(1, 2, 1), row(2, 3, 3), row(3, 6, 0)],
            routes: vec![
                route(0, 7),
                route(0, 9),
                route(1, 7),
                route(2, 9),
                route(2, 11),
                route(2, 11),
            ],
        };

        let stats = expert_queue_plan_stats(&host_batch);

        assert_eq!(stats.rows, 4);
        assert_eq!(stats.routes, 6);
        assert_eq!(stats.experts, 3);
        assert_eq!(stats.single_expert_rows, 1);
        assert_eq!(stats.multi_expert_rows, 2);
        assert_eq!(stats.empty_rows, 1);
        assert_eq!(stats.expert_only_rows, 5);
        assert_eq!(stats.expert_only_extra_rows, 1);
        assert_eq!(stats.min_expert_rows, 1);
        assert_eq!(stats.p50_expert_rows, 2);
        assert_eq!(stats.max_expert_rows, 2);
        assert_eq!(stats.min_expert_routes, 2);
        assert_eq!(stats.p50_expert_routes, 2);
        assert_eq!(stats.max_expert_routes, 2);
        assert_eq!(stats.least_hot_expert, Some(11));
        assert_eq!(stats.least_hot_expert_rows, 1);
        assert_eq!(
            stats.expert_route_counts_by_id,
            vec![(7, 2), (9, 2), (11, 2)]
        );
    }

    #[test]
    fn streamed_ingress_requests_send_plan_then_completion_ordered_rows() {
        let host_batch = ExpertHostBatch {
            host: "spark0".to_owned(),
            layer_id: LayerId(3),
            placement_version: PlacementVersion::from("test"),
            hidden_dim: 4,
            hidden_bytes_per_row: 8,
            hidden_dtype: DType::Bf16,
            graph_bucket: GraphBucket::new(3),
            quantization_recipe: "test".to_owned(),
            rows: vec![row(0, 0, 2), row(1, 2, 1), row(2, 3, 1)],
            routes: vec![route(0, 5), route(0, 9), route(1, 5), route(2, 9)],
        };
        let hidden_payload = (0_u8..24).collect::<Vec<_>>();
        let request = ExpertProtocolV2Request::from_expert_host_batch(
            81,
            &host_batch,
            hidden_payload.clone(),
        )
        .unwrap();

        let requests =
            streamed_ingress_requests(request, ExpertV2Dtype::Fp8E4m3RowScaled, 2, true, false)
                .unwrap();

        assert_eq!(requests.len(), 3);
        assert!(requests[0].stream_plan_enabled());
        assert!(requests[0].spark_reduction_enabled());
        assert!(requests[0].fp8_e4m3_row_scaled_response_enabled());
        let plan = ExpertProtocolV2StreamPlan::decode(&requests[0].hidden_payload).unwrap();
        assert_eq!(plan.activation_row_order, vec![1, 0, 2]);
        assert!(requests[1].stream_data_enabled());
        assert!(requests[1].spark_reduction_enabled());
        assert_eq!(requests[1].stream_data_row_offset(), Some(0));
        assert!(!requests[1].stream_final_enabled());
        assert_eq!(
            requests[1].hidden_payload,
            [
                hidden_payload[8..16].to_vec(),
                hidden_payload[0..8].to_vec()
            ]
            .concat()
        );
        assert_eq!(requests[2].stream_data_row_offset(), Some(2));
        assert!(requests[2].stream_final_enabled());
        assert_eq!(requests[2].hidden_payload, hidden_payload[16..24]);
    }

    #[test]
    fn rail_partitions_preserve_rows_routes_payload_and_flags() {
        let host_batch = ExpertHostBatch {
            host: "spark0".to_owned(),
            layer_id: LayerId(3),
            placement_version: PlacementVersion::from("test"),
            hidden_dim: 4,
            hidden_bytes_per_row: 8,
            hidden_dtype: DType::Bf16,
            graph_bucket: GraphBucket::new(3),
            quantization_recipe: "test".to_owned(),
            rows: vec![row(0, 0, 2), row(1, 2, 1), row(2, 3, 2)],
            routes: vec![
                route(0, 5),
                route(0, 9),
                route(1, 7),
                route(2, 11),
                route(2, 13),
            ],
        };
        let hidden_payload = (0_u8..24).collect::<Vec<_>>();
        let request = ExpertProtocolV2Request::from_expert_host_batch(
            91,
            &host_batch,
            hidden_payload.clone(),
        )
        .unwrap()
        .with_fp8_e4m3_row_scaled_response()
        .with_spark_reduction()
        .with_debug_checksum();

        let partitions = partition_protocol_v2_request_for_rails(request, 2, 2).unwrap();

        assert_eq!(partitions.len(), 2);
        assert_eq!(partitions[0].local_row_indices, vec![0, 1]);
        assert_eq!(partitions[1].local_row_indices, vec![2]);
        assert_eq!(partitions[0].request.hidden_payload, hidden_payload[..16]);
        assert_eq!(partitions[1].request.hidden_payload, hidden_payload[16..]);
        assert_eq!(partitions[0].request.rows[0].route_offset, 0);
        assert_eq!(partitions[0].request.rows[1].route_offset, 2);
        assert_eq!(partitions[1].request.rows[0].route_offset, 0);
        assert_eq!(
            partitions[1]
                .request
                .routes
                .iter()
                .map(|route| route.row_index)
                .collect::<Vec<_>>(),
            vec![0, 0]
        );
        for partition in partitions {
            assert!(partition.request.fp8_e4m3_row_scaled_response_enabled());
            assert!(partition.request.spark_reduction_enabled());
            assert!(partition.request.debug_checksum_enabled());
            partition.request.validate().unwrap();
        }
    }

    #[test]
    fn rail_partition_threshold_keeps_decode_on_one_request() {
        let host_batch = ExpertHostBatch {
            host: "spark0".to_owned(),
            layer_id: LayerId(3),
            placement_version: PlacementVersion::from("test"),
            hidden_dim: 4,
            hidden_bytes_per_row: 8,
            hidden_dtype: DType::Bf16,
            graph_bucket: GraphBucket::new(1),
            quantization_recipe: "test".to_owned(),
            rows: vec![row(0, 0, 1)],
            routes: vec![route(0, 5)],
        };
        let request =
            ExpertProtocolV2Request::from_expert_host_batch(92, &host_batch, vec![0; 8]).unwrap();

        let partitions = partition_protocol_v2_request_for_rails(request, 2, 64).unwrap();

        assert_eq!(partitions.len(), 1);
        assert_eq!(partitions[0].local_row_indices, vec![0]);
    }

    #[test]
    fn rail_partitions_keep_large_requests_within_compute_bucket() {
        let row_count = 600;
        let host_batch = ExpertHostBatch {
            host: "spark0".to_owned(),
            layer_id: LayerId(3),
            placement_version: PlacementVersion::from("test"),
            hidden_dim: 4,
            hidden_bytes_per_row: 8,
            hidden_dtype: DType::Bf16,
            graph_bucket: GraphBucket::new(row_count),
            quantization_recipe: "test".to_owned(),
            rows: (0..row_count).map(|index| row(index, index, 1)).collect(),
            routes: (0..row_count)
                .map(|index| route(index, index % 16))
                .collect(),
        };
        let request = ExpertProtocolV2Request::from_expert_host_batch(
            93,
            &host_batch,
            vec![0; row_count * 8],
        )
        .unwrap();

        let partitions = partition_protocol_v2_request_for_rails(request, 2, 64).unwrap();

        assert_eq!(partitions.len(), 3);
        assert!(partitions
            .iter()
            .all(|partition| partition.request.rows.len() <= 256));
        assert_eq!(
            partitions
                .iter()
                .flat_map(|partition| partition.local_row_indices.iter().copied())
                .collect::<Vec<_>>(),
            (0..row_count).collect::<Vec<_>>()
        );
    }

    #[test]
    fn response_stream_map_restores_original_host_rows() {
        let maps = vec![
            VerbsHostProtocolV2ResponseStreamMap {
                host_index: 1,
                local_row_indices: vec![0, 2, 4],
            },
            VerbsHostProtocolV2ResponseStreamMap {
                host_index: 1,
                local_row_indices: vec![1, 3, 5],
            },
        ];

        let mapped = map_verbs_host_response_stream_rows(&maps, 1, &[2, 0]).unwrap();

        assert_eq!(mapped, (1, vec![5, 1]));
    }

    fn row(global_row_index: usize, route_offset: usize, route_count: usize) -> ExpertHostBatchRow {
        ExpertHostBatchRow {
            global_row_index,
            row_id: global_row_index as u64,
            source_kind: RowSourceKind::Benchmark,
            request_id: RequestId::from("test"),
            sequence_id: "test".to_owned(),
            token_position: PositionId(global_row_index as u64),
            route_offset,
            route_count,
        }
    }

    fn route(row_index: usize, expert_id: usize) -> ExpertBatchRoute {
        ExpertBatchRoute {
            row_index,
            expert_id,
            gate_weight: 1.0,
        }
    }
}
