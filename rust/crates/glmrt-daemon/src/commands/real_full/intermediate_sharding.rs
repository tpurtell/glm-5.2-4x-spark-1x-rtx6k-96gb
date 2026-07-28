use super::rdma_reduction::{SparkExpertRdmaReduction, SparkExpertRdmaReductionConfig};
use anyhow::{Context, Result};
use glmrt_ffi::{GlmrtNcclComm, NativeLibrary};
use std::{
    collections::BTreeSet,
    env,
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs},
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

pub(crate) const EXPERT_INTERMEDIATE_SHARDS_ENV: &str = "GLMRT_EXPERT_INTERMEDIATE_SHARDS";
pub(crate) const EXPERT_INTERMEDIATE_SHARD_RANK_ENV: &str = "GLMRT_EXPERT_INTERMEDIATE_SHARD_RANK";
pub(crate) const EXPERT_INTERMEDIATE_REDUCTION_ENV: &str = "GLMRT_EXPERT_INTERMEDIATE_REDUCTION";
pub(crate) const EXPERT_INTERMEDIATE_REDUCTION_DTYPE_ENV: &str =
    "GLMRT_EXPERT_INTERMEDIATE_REDUCTION_DTYPE";
pub(crate) const EXPERT_INTERMEDIATE_REDUCTION_ROOT_ENV: &str =
    "GLMRT_EXPERT_INTERMEDIATE_REDUCTION_ROOT";
pub(crate) const EXPERT_INTERMEDIATE_REDUCTION_PORT_ENV: &str =
    "GLMRT_EXPERT_INTERMEDIATE_REDUCTION_PORT";
pub(crate) const EXPERT_INTERMEDIATE_REDUCTION_MIN_ROWS_ENV: &str =
    "GLMRT_EXPERT_INTERMEDIATE_REDUCTION_MIN_ROWS";
pub(crate) const EXPERT_INTERMEDIATE_ROW_SHARDED_REDUCTION_ENV: &str =
    "GLMRT_EXPERT_INTERMEDIATE_ROW_SHARDED_REDUCTION";
pub(crate) const EXPERT_INTERMEDIATE_OWNER_MAX_ROWS_ENV: &str =
    "GLMRT_EXPERT_INTERMEDIATE_OWNER_MAX_ROWS";
pub(crate) const EXPERT_INTERMEDIATE_OWNER_REDUCTION_DTYPE_ENV: &str =
    "GLMRT_EXPERT_INTERMEDIATE_OWNER_REDUCTION_DTYPE";
pub(crate) const EXPERT_INTERMEDIATE_OWNER_PEERS_ENV: &str =
    "GLMRT_EXPERT_INTERMEDIATE_OWNER_PEERS";
pub(crate) const EXPERT_INTERMEDIATE_OWNER_PORT_ENV: &str = "GLMRT_EXPERT_INTERMEDIATE_OWNER_PORT";
pub(crate) const EXPERT_INTERMEDIATE_RDMA_PEERS_ENV: &str = "GLMRT_EXPERT_INTERMEDIATE_RDMA_PEERS";
pub(crate) const EXPERT_INTERMEDIATE_RDMA_ADDITIONAL_PEERS_ENV: &str =
    "GLMRT_EXPERT_INTERMEDIATE_RDMA_ADDITIONAL_PEERS";
pub(crate) const EXPERT_INTERMEDIATE_RDMA_DEVICES_ENV: &str =
    "GLMRT_EXPERT_INTERMEDIATE_RDMA_DEVICES";
pub(crate) const EXPERT_INTERMEDIATE_RDMA_PORT_ENV: &str = "GLMRT_EXPERT_INTERMEDIATE_RDMA_PORT";
pub(crate) const EXPERT_INTERMEDIATE_RDMA_SLOT_BYTES_ENV: &str =
    "GLMRT_EXPERT_INTERMEDIATE_RDMA_SLOT_BYTES";
pub(crate) const EXPERT_INTERMEDIATE_RDMA_RING_DEPTH_ENV: &str =
    "GLMRT_EXPERT_INTERMEDIATE_RDMA_RING_DEPTH";
pub(crate) const EXPERT_INTERMEDIATE_RDMA_STRIPE_MIN_BYTES_ENV: &str =
    "GLMRT_EXPERT_INTERMEDIATE_RDMA_STRIPE_MIN_BYTES";

const DEFAULT_REDUCTION_ROOT: &str = "ostrich.200gb";
const DEFAULT_REDUCTION_PORT: u16 = 9200;
const DEFAULT_REDUCTION_MIN_ROWS: usize = 16;
const DEFAULT_OWNER_REDUCTION_MIN_ROWS: usize = 1;
const DEFAULT_OWNER_REDUCTION_MAX_ROWS: usize = 8;
const DEFAULT_OWNER_PORT: u16 = 9100;
const DEFAULT_RDMA_PORT: u16 = 9400;
const DEFAULT_RDMA_SLOT_BYTES: usize = 4 * 1024 * 1024;
const DEFAULT_RDMA_RING_DEPTH: usize = 4;
const DEFAULT_RDMA_STRIPE_MIN_BYTES: usize = 256 * 1024;
const REDUCTION_ROOT_RANK: usize = 0;
const REDUCTION_BOOTSTRAP_MAGIC: &[u8; 8] = b"GLMNCCL1";
const REDUCTION_BOOTSTRAP_TIMEOUT: Duration = Duration::from_secs(300);
const REDUCTION_BOOTSTRAP_RETRY: Duration = Duration::from_millis(100);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ExpertIntermediateShard {
    pub(crate) count: usize,
    pub(crate) rank: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExpertIntermediateReductionDtype {
    Bf16,
    Fp8,
    Nvfp4,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExpertIntermediateReductionMode {
    Coordinator,
    SparkNccl,
    SparkOwner,
    SparkHybrid,
    SparkRdma,
    SparkRdmaHybrid,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SparkExpertReductionDispatch {
    pub(crate) root_rank: usize,
    pub(crate) owner_fanout: bool,
    pub(crate) row_sharded: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SparkExpertOwnerReductionConfig {
    pub(crate) shard: ExpertIntermediateShard,
    pub(crate) dtype: ExpertIntermediateReductionDtype,
    pub(crate) max_rows: usize,
    pub(crate) peers: Vec<(usize, SocketAddr)>,
}

pub(crate) struct SparkExpertReduction {
    communicator: GlmrtNcclComm,
    pub(crate) dtype: ExpertIntermediateReductionDtype,
    pub(crate) min_rows: usize,
    pub(crate) root_rank: usize,
}

impl SparkExpertReduction {
    pub(crate) fn communicator(&self) -> &GlmrtNcclComm {
        &self.communicator
    }

    pub(crate) fn enabled_for_rows(&self, rows: usize) -> bool {
        rows >= self.min_rows
    }

    pub(crate) fn is_root(&self) -> bool {
        self.communicator.rank() == self.root_rank
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SparkExpertReductionConfig {
    dtype: ExpertIntermediateReductionDtype,
    root_host: String,
    port: u16,
    min_rows: usize,
}

impl ExpertIntermediateShard {
    pub(crate) fn new(count: usize, rank: usize) -> Result<Self> {
        anyhow::ensure!(
            count == 4,
            "intermediate sharding currently supports exactly four shards, got {count}"
        );
        anyhow::ensure!(
            rank < count,
            "intermediate shard rank {rank} is outside 0..{count}"
        );
        Ok(Self { count, rank })
    }

    pub(crate) fn local_rows(self, full_rows: usize) -> Result<usize> {
        anyhow::ensure!(
            full_rows > 0 && full_rows % self.count == 0,
            "intermediate rows {full_rows} are not divisible by {} shards",
            self.count
        );
        Ok(full_rows / self.count)
    }

    pub(crate) fn row_start(self, full_rows: usize) -> Result<usize> {
        self.local_rows(full_rows)?
            .checked_mul(self.rank)
            .context("intermediate shard row start overflow")
    }
}

pub(crate) fn expert_intermediate_shard_count_from_env() -> Result<usize> {
    parse_shard_count(env::var(EXPERT_INTERMEDIATE_SHARDS_ENV).ok().as_deref())
}

pub(crate) fn spark_expert_intermediate_shard_from_env() -> Result<Option<ExpertIntermediateShard>>
{
    let count = expert_intermediate_shard_count_from_env()?;
    if count == 1 {
        return Ok(None);
    }
    let rank = env::var(EXPERT_INTERMEDIATE_SHARD_RANK_ENV)
        .with_context(|| {
            format!(
                "{EXPERT_INTERMEDIATE_SHARDS_ENV}={count} requires {EXPERT_INTERMEDIATE_SHARD_RANK_ENV}"
            )
        })?
        .parse::<usize>()
        .with_context(|| format!("parsing {EXPERT_INTERMEDIATE_SHARD_RANK_ENV}"))?;
    ExpertIntermediateShard::new(count, rank).map(Some)
}

pub(crate) fn initialize_spark_expert_reduction_lane(
    library: Arc<NativeLibrary>,
    shard: Option<ExpertIntermediateShard>,
    execution_lane: u32,
) -> Result<Option<SparkExpertReduction>> {
    let Some(config) = spark_expert_reduction_config_from_env(shard)? else {
        return Ok(None);
    };
    let shard = shard.context("Spark expert reduction requires an intermediate shard")?;
    let port = spark_expert_reduction_port_for_lane(config.port, execution_lane)?;
    let communicator = initialize_spark_nccl_communicator(
        &library,
        shard,
        &config.root_host,
        port,
        "expert reduction",
    )?;
    eprintln!(
        "spark_expert_reduction_ready execution_lane={} rank={} world_size={} root_rank={} dtype={:?} min_rows={} bootstrap={}:{}",
        execution_lane,
        shard.rank,
        shard.count,
        REDUCTION_ROOT_RANK,
        config.dtype,
        config.min_rows,
        config.root_host,
        port
    );
    Ok(Some(SparkExpertReduction {
        communicator,
        dtype: config.dtype,
        min_rows: config.min_rows,
        root_rank: REDUCTION_ROOT_RANK,
    }))
}

pub(crate) fn initialize_spark_expert_rdma_reduction_lane(
    shard: Option<ExpertIntermediateShard>,
    execution_lane: u32,
) -> Result<Option<SparkExpertRdmaReduction>> {
    let Some(config) = spark_expert_rdma_reduction_config_from_env(shard)? else {
        return Ok(None);
    };
    SparkExpertRdmaReduction::connect(config, execution_lane).map(Some)
}

fn spark_expert_reduction_port_for_lane(base_port: u16, execution_lane: u32) -> Result<u16> {
    let offset = u16::try_from(execution_lane).context("Spark execution lane exceeds u16")?;
    base_port
        .checked_add(offset)
        .context("Spark execution-lane NCCL bootstrap port overflow")
}

pub(crate) fn initialize_spark_nccl_communicator(
    library: &Arc<NativeLibrary>,
    shard: ExpertIntermediateShard,
    root_host: &str,
    port: u16,
    purpose: &str,
) -> Result<GlmrtNcclComm> {
    let unique_id = exchange_nccl_unique_id(library, shard, root_host, port)?;
    library
        .nccl_comm_init_rank(&unique_id, shard.count, shard.rank)
        .with_context(|| {
            format!(
                "initializing Spark NCCL {purpose} communicator rank {}/{}",
                shard.rank, shard.count
            )
        })
}

pub(crate) fn spark_expert_reduction_dispatch_for_rows(
    rows: usize,
) -> Result<Option<SparkExpertReductionDispatch>> {
    let mode = expert_intermediate_reduction_mode_from_env()?;
    if mode == ExpertIntermediateReductionMode::Coordinator {
        return Ok(None);
    }
    anyhow::ensure!(
        expert_intermediate_shard_count_from_env()? == 4,
        "Spark expert reduction requires four intermediate shards"
    );
    let min_rows = parse_positive(
        EXPERT_INTERMEDIATE_REDUCTION_MIN_ROWS_ENV,
        env::var(EXPERT_INTERMEDIATE_REDUCTION_MIN_ROWS_ENV)
            .ok()
            .as_deref(),
        if mode == ExpertIntermediateReductionMode::SparkOwner {
            DEFAULT_OWNER_REDUCTION_MIN_ROWS
        } else {
            DEFAULT_REDUCTION_MIN_ROWS
        },
    )?;
    let owner_max_rows = if matches!(
        mode,
        ExpertIntermediateReductionMode::SparkOwner
            | ExpertIntermediateReductionMode::SparkHybrid
            | ExpertIntermediateReductionMode::SparkRdmaHybrid
    ) {
        parse_positive(
            EXPERT_INTERMEDIATE_OWNER_MAX_ROWS_ENV,
            env::var(EXPERT_INTERMEDIATE_OWNER_MAX_ROWS_ENV)
                .ok()
                .as_deref(),
            DEFAULT_OWNER_REDUCTION_MAX_ROWS,
        )?
    } else {
        0
    };
    let row_sharded = parse_boolean(
        EXPERT_INTERMEDIATE_ROW_SHARDED_REDUCTION_ENV,
        env::var(EXPERT_INTERMEDIATE_ROW_SHARDED_REDUCTION_ENV)
            .ok()
            .as_deref(),
        false,
    )?;
    if matches!(
        mode,
        ExpertIntermediateReductionMode::SparkRdma
            | ExpertIntermediateReductionMode::SparkRdmaHybrid
    ) {
        anyhow::ensure!(
            row_sharded,
            "Spark RDMA reduction requires {EXPERT_INTERMEDIATE_ROW_SHARDED_REDUCTION_ENV}=1"
        );
    }
    Ok(reduction_dispatch_for_config(
        mode,
        rows,
        min_rows,
        owner_max_rows,
        row_sharded,
    ))
}

fn reduction_dispatch_for_config(
    mode: ExpertIntermediateReductionMode,
    rows: usize,
    reduction_min_rows: usize,
    owner_max_rows: usize,
    row_sharded: bool,
) -> Option<SparkExpertReductionDispatch> {
    let owner_fanout = match mode {
        ExpertIntermediateReductionMode::SparkOwner => {
            rows >= reduction_min_rows && rows <= owner_max_rows
        }
        ExpertIntermediateReductionMode::SparkHybrid => rows <= owner_max_rows,
        ExpertIntermediateReductionMode::SparkRdmaHybrid => rows <= owner_max_rows,
        ExpertIntermediateReductionMode::Coordinator
        | ExpertIntermediateReductionMode::SparkNccl
        | ExpertIntermediateReductionMode::SparkRdma => false,
    };
    if owner_fanout {
        return Some(SparkExpertReductionDispatch {
            root_rank: REDUCTION_ROOT_RANK,
            owner_fanout: true,
            row_sharded: false,
        });
    }
    let distributed = matches!(
        mode,
        ExpertIntermediateReductionMode::SparkNccl
            | ExpertIntermediateReductionMode::SparkHybrid
            | ExpertIntermediateReductionMode::SparkRdma
            | ExpertIntermediateReductionMode::SparkRdmaHybrid
    ) && rows >= reduction_min_rows;
    distributed.then_some(SparkExpertReductionDispatch {
        root_rank: REDUCTION_ROOT_RANK,
        owner_fanout: false,
        row_sharded,
    })
}

pub(crate) fn balanced_row_partition(
    rows: usize,
    world_size: usize,
    rank: usize,
) -> Result<(usize, usize)> {
    anyhow::ensure!(
        world_size > 0 && rank < world_size,
        "invalid row partition rank"
    );
    let base_rows = rows / world_size;
    let extra_rows = rows % world_size;
    let row_count = base_rows + usize::from(rank < extra_rows);
    let row_start = rank
        .checked_mul(base_rows)
        .and_then(|start| start.checked_add(rank.min(extra_rows)))
        .context("row partition start overflow")?;
    Ok((row_start, row_count))
}

fn expert_intermediate_reduction_mode_from_env() -> Result<ExpertIntermediateReductionMode> {
    let mode =
        env::var(EXPERT_INTERMEDIATE_REDUCTION_ENV).unwrap_or_else(|_| "coordinator".to_owned());
    parse_reduction_mode(&mode)
}

fn parse_reduction_mode(raw: &str) -> Result<ExpertIntermediateReductionMode> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "coordinator" | "off" | "none" => Ok(ExpertIntermediateReductionMode::Coordinator),
        "spark" | "spark-nccl" | "nccl" => Ok(ExpertIntermediateReductionMode::SparkNccl),
        "spark-owner" | "owner" | "verbs-owner" => {
            Ok(ExpertIntermediateReductionMode::SparkOwner)
        }
        "spark-hybrid" | "hybrid" | "owner-nccl" => {
            Ok(ExpertIntermediateReductionMode::SparkHybrid)
        }
        "spark-rdma" | "rdma" | "verbs" => Ok(ExpertIntermediateReductionMode::SparkRdma),
        "spark-rdma-hybrid" | "rdma-hybrid" | "owner-rdma" => {
            Ok(ExpertIntermediateReductionMode::SparkRdmaHybrid)
        }
        value => anyhow::bail!(
            "{EXPERT_INTERMEDIATE_REDUCTION_ENV} must be coordinator, spark, spark-owner, spark-hybrid, spark-rdma, or spark-rdma-hybrid, got {value}"
        ),
    }
}

fn spark_expert_rdma_reduction_config_from_env(
    shard: Option<ExpertIntermediateShard>,
) -> Result<Option<SparkExpertRdmaReductionConfig>> {
    if !matches!(
        expert_intermediate_reduction_mode_from_env()?,
        ExpertIntermediateReductionMode::SparkRdma
            | ExpertIntermediateReductionMode::SparkRdmaHybrid
    ) {
        return Ok(None);
    }
    let shard = shard.context("Spark RDMA reduction requires intermediate sharding")?;
    anyhow::ensure!(
        shard.count == 4,
        "Spark RDMA reduction currently requires four shards"
    );
    let dtype = parse_reduction_dtype(
        env::var(EXPERT_INTERMEDIATE_REDUCTION_DTYPE_ENV)
            .ok()
            .as_deref(),
    )?;
    let min_rows = parse_positive(
        EXPERT_INTERMEDIATE_REDUCTION_MIN_ROWS_ENV,
        env::var(EXPERT_INTERMEDIATE_REDUCTION_MIN_ROWS_ENV)
            .ok()
            .as_deref(),
        DEFAULT_REDUCTION_MIN_ROWS,
    )?;
    let base_port = parse_positive(
        EXPERT_INTERMEDIATE_RDMA_PORT_ENV,
        env::var(EXPERT_INTERMEDIATE_RDMA_PORT_ENV).ok().as_deref(),
        DEFAULT_RDMA_PORT as usize,
    )?;
    let base_port = u16::try_from(base_port)
        .with_context(|| format!("{EXPERT_INTERMEDIATE_RDMA_PORT_ENV} exceeds u16"))?;
    let slot_capacity_bytes = parse_positive(
        EXPERT_INTERMEDIATE_RDMA_SLOT_BYTES_ENV,
        env::var(EXPERT_INTERMEDIATE_RDMA_SLOT_BYTES_ENV)
            .ok()
            .as_deref(),
        DEFAULT_RDMA_SLOT_BYTES,
    )?;
    let ring_depth = parse_positive(
        EXPERT_INTERMEDIATE_RDMA_RING_DEPTH_ENV,
        env::var(EXPERT_INTERMEDIATE_RDMA_RING_DEPTH_ENV)
            .ok()
            .as_deref(),
        DEFAULT_RDMA_RING_DEPTH,
    )?;
    let rank_hosts = env::var(EXPERT_INTERMEDIATE_RDMA_PEERS_ENV)
        .ok()
        .map(|raw| {
            raw.split(',')
                .map(str::trim)
                .filter(|host| !host.is_empty())
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| {
            glmrt_core::EXPERT_HOSTS
                .iter()
                .map(|host| format!("{host}.200gb"))
                .collect()
        });
    anyhow::ensure!(
        rank_hosts.len() == shard.count,
        "{EXPERT_INTERMEDIATE_RDMA_PEERS_ENV} must contain {} rank-ordered hosts, got {}",
        shard.count,
        rank_hosts.len()
    );
    anyhow::ensure!(
        rank_hosts.iter().all(|host| !host.contains(':')),
        "{EXPERT_INTERMEDIATE_RDMA_PEERS_ENV} entries must be host names without ports"
    );
    let additional_rank_hosts = env::var(EXPERT_INTERMEDIATE_RDMA_ADDITIONAL_PEERS_ENV)
        .ok()
        .map(|raw| {
            raw.split(',')
                .map(str::trim)
                .filter(|host| !host.is_empty())
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if !additional_rank_hosts.is_empty() {
        anyhow::ensure!(
            additional_rank_hosts.len() == shard.count,
            "{EXPERT_INTERMEDIATE_RDMA_ADDITIONAL_PEERS_ENV} must contain {} rank-ordered hosts, got {}",
            shard.count,
            additional_rank_hosts.len()
        );
        anyhow::ensure!(
            additional_rank_hosts.iter().all(|host| !host.contains(':')),
            "{EXPERT_INTERMEDIATE_RDMA_ADDITIONAL_PEERS_ENV} entries must be host names without ports"
        );
    }
    let mut rank_hosts_by_rail = vec![rank_hosts];
    if !additional_rank_hosts.is_empty() {
        rank_hosts_by_rail.push(additional_rank_hosts);
    }
    let rdma_devices = match env::var(EXPERT_INTERMEDIATE_RDMA_DEVICES_ENV).ok() {
        Some(raw) => raw
            .split(',')
            .map(str::trim)
            .filter(|device| !device.is_empty())
            .map(str::to_owned)
            .collect::<Vec<_>>(),
        None if rank_hosts_by_rail.len() == 2 => {
            vec!["rocep1s0f0".to_owned(), "roceP2p1s0f0".to_owned()]
        }
        None => Vec::new(),
    };
    anyhow::ensure!(
        rdma_devices.is_empty() || rdma_devices.len() == rank_hosts_by_rail.len(),
        "{EXPERT_INTERMEDIATE_RDMA_DEVICES_ENV} must contain one device per rail ({}), got {}",
        rank_hosts_by_rail.len(),
        rdma_devices.len()
    );
    let stripe_min_bytes = parse_positive(
        EXPERT_INTERMEDIATE_RDMA_STRIPE_MIN_BYTES_ENV,
        env::var(EXPERT_INTERMEDIATE_RDMA_STRIPE_MIN_BYTES_ENV)
            .ok()
            .as_deref(),
        DEFAULT_RDMA_STRIPE_MIN_BYTES,
    )?;
    Ok(Some(SparkExpertRdmaReductionConfig {
        shard,
        dtype,
        min_rows,
        base_port,
        rank_hosts_by_rail,
        rdma_devices,
        slot_capacity_bytes,
        ring_depth,
        stripe_min_bytes,
    }))
}

fn spark_expert_reduction_config_from_env(
    shard: Option<ExpertIntermediateShard>,
) -> Result<Option<SparkExpertReductionConfig>> {
    if !matches!(
        expert_intermediate_reduction_mode_from_env()?,
        ExpertIntermediateReductionMode::SparkNccl | ExpertIntermediateReductionMode::SparkHybrid
    ) {
        return Ok(None);
    }
    let shard = shard.context("Spark expert reduction requires intermediate sharding")?;
    anyhow::ensure!(
        shard.count == 4,
        "Spark expert reduction currently requires four shards"
    );
    let dtype = parse_reduction_dtype(
        env::var(EXPERT_INTERMEDIATE_REDUCTION_DTYPE_ENV)
            .ok()
            .as_deref(),
    )?;
    let root_host = env::var(EXPERT_INTERMEDIATE_REDUCTION_ROOT_ENV)
        .unwrap_or_else(|_| DEFAULT_REDUCTION_ROOT.to_owned());
    anyhow::ensure!(
        !root_host.trim().is_empty(),
        "{EXPERT_INTERMEDIATE_REDUCTION_ROOT_ENV} must not be empty"
    );
    let port = parse_positive(
        EXPERT_INTERMEDIATE_REDUCTION_PORT_ENV,
        env::var(EXPERT_INTERMEDIATE_REDUCTION_PORT_ENV)
            .ok()
            .as_deref(),
        DEFAULT_REDUCTION_PORT as usize,
    )?;
    let port = u16::try_from(port)
        .with_context(|| format!("{EXPERT_INTERMEDIATE_REDUCTION_PORT_ENV} exceeds u16"))?;
    let min_rows = parse_positive(
        EXPERT_INTERMEDIATE_REDUCTION_MIN_ROWS_ENV,
        env::var(EXPERT_INTERMEDIATE_REDUCTION_MIN_ROWS_ENV)
            .ok()
            .as_deref(),
        DEFAULT_REDUCTION_MIN_ROWS,
    )?;
    Ok(Some(SparkExpertReductionConfig {
        dtype,
        root_host,
        port,
        min_rows,
    }))
}

pub(crate) fn spark_expert_owner_reduction_config_from_env(
    shard: Option<ExpertIntermediateShard>,
) -> Result<Option<SparkExpertOwnerReductionConfig>> {
    if !matches!(
        expert_intermediate_reduction_mode_from_env()?,
        ExpertIntermediateReductionMode::SparkOwner
            | ExpertIntermediateReductionMode::SparkHybrid
            | ExpertIntermediateReductionMode::SparkRdmaHybrid
    ) {
        return Ok(None);
    }
    let shard = shard.context("Spark owner reduction requires intermediate sharding")?;
    anyhow::ensure!(
        shard.count == 4,
        "Spark owner reduction currently requires four shards"
    );
    let owner_dtype = env::var(EXPERT_INTERMEDIATE_OWNER_REDUCTION_DTYPE_ENV).ok();
    let reduction_dtype = env::var(EXPERT_INTERMEDIATE_REDUCTION_DTYPE_ENV).ok();
    let dtype = parse_owner_reduction_dtype(owner_dtype.as_deref(), reduction_dtype.as_deref())?;
    let max_rows = parse_positive(
        EXPERT_INTERMEDIATE_OWNER_MAX_ROWS_ENV,
        env::var(EXPERT_INTERMEDIATE_OWNER_MAX_ROWS_ENV)
            .ok()
            .as_deref(),
        DEFAULT_OWNER_REDUCTION_MAX_ROWS,
    )?;
    let endpoints = if let Ok(raw) = env::var(EXPERT_INTERMEDIATE_OWNER_PEERS_ENV) {
        let endpoints = raw
            .split(',')
            .map(str::trim)
            .filter(|endpoint| !endpoint.is_empty())
            .map(str::to_owned)
            .collect::<Vec<_>>();
        anyhow::ensure!(
            endpoints.len() == shard.count,
            "{EXPERT_INTERMEDIATE_OWNER_PEERS_ENV} must contain {} rank-ordered endpoints, got {}",
            shard.count,
            endpoints.len()
        );
        endpoints
    } else {
        let port = parse_positive(
            EXPERT_INTERMEDIATE_OWNER_PORT_ENV,
            env::var(EXPERT_INTERMEDIATE_OWNER_PORT_ENV).ok().as_deref(),
            DEFAULT_OWNER_PORT as usize,
        )?;
        let port = u16::try_from(port)
            .with_context(|| format!("{EXPERT_INTERMEDIATE_OWNER_PORT_ENV} exceeds u16"))?;
        glmrt_core::EXPERT_HOSTS
            .iter()
            .map(|host| format!("{host}.200gb:{port}"))
            .collect()
    };
    let peers = endpoints
        .into_iter()
        .enumerate()
        .filter(|(rank, _)| *rank != shard.rank)
        .map(|(rank, endpoint)| {
            let addr = endpoint
                .to_socket_addrs()
                .with_context(|| format!("resolving Spark owner peer rank {rank} {endpoint}"))?
                .next()
                .with_context(|| {
                    format!("Spark owner peer rank {rank} {endpoint} resolved no addresses")
                })?;
            Ok((rank, addr))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(SparkExpertOwnerReductionConfig {
        shard,
        dtype,
        max_rows,
        peers,
    }))
}

fn parse_reduction_dtype(raw: Option<&str>) -> Result<ExpertIntermediateReductionDtype> {
    match raw.unwrap_or("fp8").trim().to_ascii_lowercase().as_str() {
        "bf16" => Ok(ExpertIntermediateReductionDtype::Bf16),
        "fp8" | "fp8-e4m3" | "fp8-e4m3-row-scaled" => Ok(ExpertIntermediateReductionDtype::Fp8),
        "nvfp4" | "nvfp4-e2m1-fp8-e4m3" => Ok(ExpertIntermediateReductionDtype::Nvfp4),
        value => anyhow::bail!(
            "{EXPERT_INTERMEDIATE_REDUCTION_DTYPE_ENV} must be bf16, fp8, or nvfp4, got {value}"
        ),
    }
}

fn parse_owner_reduction_dtype(
    owner_raw: Option<&str>,
    reduction_raw: Option<&str>,
) -> Result<ExpertIntermediateReductionDtype> {
    parse_reduction_dtype(owner_raw.or(reduction_raw))
}

fn parse_positive(name: &str, raw: Option<&str>, default: usize) -> Result<usize> {
    let value = raw
        .filter(|value| !value.trim().is_empty())
        .map(str::parse::<usize>)
        .transpose()
        .with_context(|| format!("parsing {name}"))?
        .unwrap_or(default);
    anyhow::ensure!(value > 0, "{name} must be positive");
    Ok(value)
}

fn parse_boolean(name: &str, raw: Option<&str>, default: bool) -> Result<bool> {
    match raw.map(str::trim).filter(|value| !value.is_empty()) {
        None => Ok(default),
        Some("1" | "true" | "yes" | "on") => Ok(true),
        Some("0" | "false" | "no" | "off") => Ok(false),
        Some(value) => anyhow::bail!("{name} must be boolean-like, got {value}"),
    }
}

fn exchange_nccl_unique_id(
    library: &NativeLibrary,
    shard: ExpertIntermediateShard,
    root_host: &str,
    port: u16,
) -> Result<Vec<u8>> {
    if shard.rank == REDUCTION_ROOT_RANK {
        let unique_id = library
            .nccl_get_unique_id()
            .context("generating Spark NCCL unique ID")?;
        distribute_nccl_unique_id(&unique_id, shard, port)?;
        Ok(unique_id)
    } else {
        receive_nccl_unique_id(shard, root_host, port)
    }
}

fn distribute_nccl_unique_id(
    unique_id: &[u8],
    shard: ExpertIntermediateShard,
    port: u16,
) -> Result<()> {
    let listener = TcpListener::bind(("0.0.0.0", port))
        .with_context(|| format!("binding Spark NCCL bootstrap listener on port {port}"))?;
    listener
        .set_nonblocking(true)
        .context("making Spark NCCL bootstrap listener nonblocking")?;
    let mut ranks = BTreeSet::new();
    let started = Instant::now();
    while ranks.len() + 1 < shard.count {
        anyhow::ensure!(
            started.elapsed() < REDUCTION_BOOTSTRAP_TIMEOUT,
            "timed out waiting for Spark NCCL bootstrap peers: received {}/{}",
            ranks.len(),
            shard.count - 1
        );
        let (mut stream, peer) = match listener.accept() {
            Ok(accepted) => accepted,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(REDUCTION_BOOTSTRAP_RETRY);
                continue;
            }
            Err(error) => return Err(error).context("accepting Spark NCCL bootstrap peer"),
        };
        configure_bootstrap_stream(&stream)?;
        let rank = read_bootstrap_hello(&mut stream)
            .with_context(|| format!("reading Spark NCCL bootstrap peer {peer}"))?;
        anyhow::ensure!(
            rank < shard.count && rank != REDUCTION_ROOT_RANK && ranks.insert(rank),
            "invalid or duplicate Spark NCCL bootstrap rank {rank}"
        );
        write_bootstrap_unique_id(&mut stream, unique_id)
            .with_context(|| format!("writing Spark NCCL unique ID to rank {rank}"))?;
    }
    Ok(())
}

fn receive_nccl_unique_id(
    shard: ExpertIntermediateShard,
    root_host: &str,
    port: u16,
) -> Result<Vec<u8>> {
    let started = Instant::now();
    let mut last_error = None;
    while started.elapsed() < REDUCTION_BOOTSTRAP_TIMEOUT {
        match TcpStream::connect((root_host, port)) {
            Ok(mut stream) => {
                configure_bootstrap_stream(&stream)?;
                stream.write_all(REDUCTION_BOOTSTRAP_MAGIC)?;
                stream.write_all(&(shard.rank as u32).to_le_bytes())?;
                return read_bootstrap_unique_id(&mut stream);
            }
            Err(error) => {
                last_error = Some(error);
                thread::sleep(REDUCTION_BOOTSTRAP_RETRY);
            }
        }
    }
    anyhow::bail!(
        "timed out connecting Spark NCCL rank {} to {}:{}: {}",
        shard.rank,
        root_host,
        port,
        last_error
            .map(|error| error.to_string())
            .unwrap_or_else(|| "no connection attempt completed".to_owned())
    )
}

fn configure_bootstrap_stream(stream: &TcpStream) -> Result<()> {
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(REDUCTION_BOOTSTRAP_TIMEOUT))?;
    stream.set_write_timeout(Some(REDUCTION_BOOTSTRAP_TIMEOUT))?;
    Ok(())
}

fn read_bootstrap_hello(stream: &mut TcpStream) -> Result<usize> {
    let mut magic = [0_u8; REDUCTION_BOOTSTRAP_MAGIC.len()];
    stream.read_exact(&mut magic)?;
    anyhow::ensure!(
        &magic == REDUCTION_BOOTSTRAP_MAGIC,
        "Spark NCCL bootstrap magic mismatch"
    );
    let mut rank = [0_u8; std::mem::size_of::<u32>()];
    stream.read_exact(&mut rank)?;
    Ok(u32::from_le_bytes(rank) as usize)
}

fn write_bootstrap_unique_id(stream: &mut TcpStream, unique_id: &[u8]) -> Result<()> {
    let bytes = u32::try_from(unique_id.len()).context("Spark NCCL unique ID exceeds u32")?;
    stream.write_all(REDUCTION_BOOTSTRAP_MAGIC)?;
    stream.write_all(&bytes.to_le_bytes())?;
    stream.write_all(unique_id)?;
    Ok(())
}

fn read_bootstrap_unique_id(stream: &mut TcpStream) -> Result<Vec<u8>> {
    let mut magic = [0_u8; REDUCTION_BOOTSTRAP_MAGIC.len()];
    stream.read_exact(&mut magic)?;
    anyhow::ensure!(
        &magic == REDUCTION_BOOTSTRAP_MAGIC,
        "Spark NCCL bootstrap response magic mismatch"
    );
    let mut bytes = [0_u8; std::mem::size_of::<u32>()];
    stream.read_exact(&mut bytes)?;
    let bytes = u32::from_le_bytes(bytes) as usize;
    anyhow::ensure!(
        bytes > 0 && bytes <= 4096,
        "invalid Spark NCCL unique ID bytes {bytes}"
    );
    let mut unique_id = vec![0_u8; bytes];
    stream.read_exact(&mut unique_id)?;
    Ok(unique_id)
}

fn parse_shard_count(raw: Option<&str>) -> Result<usize> {
    let count = raw
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("1")
        .parse::<usize>()
        .with_context(|| format!("parsing {EXPERT_INTERMEDIATE_SHARDS_ENV}"))?;
    anyhow::ensure!(
        matches!(count, 1 | 4),
        "{EXPERT_INTERMEDIATE_SHARDS_ENV} must be 1 or 4, got {count}"
    );
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::{
        balanced_row_partition, parse_owner_reduction_dtype, parse_positive, parse_reduction_dtype,
        parse_reduction_mode, parse_shard_count, reduction_dispatch_for_config,
        spark_expert_reduction_port_for_lane, ExpertIntermediateReductionDtype,
        ExpertIntermediateReductionMode, ExpertIntermediateShard,
    };

    #[test]
    fn execution_lanes_use_distinct_nccl_bootstrap_ports() {
        assert_eq!(
            spark_expert_reduction_port_for_lane(9_200, 0).unwrap(),
            9_200
        );
        assert_eq!(
            spark_expert_reduction_port_for_lane(9_200, 1).unwrap(),
            9_201
        );
        assert!(spark_expert_reduction_port_for_lane(u16::MAX, 1).is_err());
    }

    #[test]
    fn intermediate_shard_config_requires_supported_even_partition() {
        assert_eq!(parse_shard_count(None).unwrap(), 1);
        assert_eq!(parse_shard_count(Some("4")).unwrap(), 4);
        assert!(parse_shard_count(Some("2")).is_err());

        let shard = ExpertIntermediateShard::new(4, 3).unwrap();
        assert_eq!(shard.local_rows(2_048).unwrap(), 512);
        assert_eq!(shard.row_start(2_048).unwrap(), 1_536);
        assert!(shard.local_rows(2_050).is_err());
        assert!(ExpertIntermediateShard::new(4, 4).is_err());
    }

    #[test]
    fn intermediate_reduction_config_parses_codecs_and_positive_values() {
        assert_eq!(
            parse_reduction_mode("coordinator").unwrap(),
            ExpertIntermediateReductionMode::Coordinator
        );
        assert_eq!(
            parse_reduction_mode("spark").unwrap(),
            ExpertIntermediateReductionMode::SparkNccl
        );
        assert_eq!(
            parse_reduction_mode("spark-owner").unwrap(),
            ExpertIntermediateReductionMode::SparkOwner
        );
        assert_eq!(
            parse_reduction_mode("spark-hybrid").unwrap(),
            ExpertIntermediateReductionMode::SparkHybrid
        );
        assert_eq!(
            parse_reduction_mode("spark-rdma").unwrap(),
            ExpertIntermediateReductionMode::SparkRdma
        );
        assert_eq!(
            parse_reduction_mode("spark-rdma-hybrid").unwrap(),
            ExpertIntermediateReductionMode::SparkRdmaHybrid
        );
        assert!(parse_reduction_mode("magic").is_err());
        assert_eq!(
            parse_reduction_dtype(None).unwrap(),
            ExpertIntermediateReductionDtype::Fp8
        );
        assert_eq!(
            parse_reduction_dtype(Some("bf16")).unwrap(),
            ExpertIntermediateReductionDtype::Bf16
        );
        assert_eq!(
            parse_reduction_dtype(Some("nvfp4")).unwrap(),
            ExpertIntermediateReductionDtype::Nvfp4
        );
        assert_eq!(
            parse_owner_reduction_dtype(Some("bf16"), Some("fp8")).unwrap(),
            ExpertIntermediateReductionDtype::Bf16
        );
        assert_eq!(
            parse_owner_reduction_dtype(None, Some("nvfp4")).unwrap(),
            ExpertIntermediateReductionDtype::Nvfp4
        );
        assert!(parse_reduction_dtype(Some("f32")).is_err());
        assert_eq!(parse_positive("test", None, 16).unwrap(), 16);
        assert_eq!(parse_positive("test", Some("64"), 16).unwrap(), 64);
        assert!(parse_positive("test", Some("0"), 16).is_err());
    }

    #[test]
    fn hybrid_reduction_keeps_decode_on_owner_and_prefill_on_nccl() {
        let mode = ExpertIntermediateReductionMode::SparkHybrid;
        assert!(
            reduction_dispatch_for_config(mode, 1, 16, 8, true)
                .unwrap()
                .owner_fanout
        );
        assert!(
            reduction_dispatch_for_config(mode, 8, 16, 8, true)
                .unwrap()
                .owner_fanout
        );
        assert!(reduction_dispatch_for_config(mode, 9, 16, 8, true).is_none());
        assert!(
            !reduction_dispatch_for_config(mode, 16, 16, 8, true)
                .unwrap()
                .owner_fanout
        );
        let prefill = reduction_dispatch_for_config(mode, 1_024, 16, 8, true).unwrap();
        assert!(!prefill.owner_fanout);
        assert!(prefill.row_sharded);
    }

    #[test]
    fn rdma_hybrid_keeps_decode_on_owner_and_prefill_on_rdma() {
        let mode = ExpertIntermediateReductionMode::SparkRdmaHybrid;
        assert!(
            reduction_dispatch_for_config(mode, 8, 16, 8, true)
                .unwrap()
                .owner_fanout
        );
        let prefill = reduction_dispatch_for_config(mode, 256, 16, 8, true).unwrap();
        assert!(!prefill.owner_fanout);
        assert!(prefill.row_sharded);
    }

    #[test]
    fn balanced_row_partitions_cover_uneven_batches() {
        let partitions = (0..4)
            .map(|rank| balanced_row_partition(1_006, 4, rank).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            partitions,
            vec![(0, 252), (252, 252), (504, 251), (755, 251)]
        );
    }
}
