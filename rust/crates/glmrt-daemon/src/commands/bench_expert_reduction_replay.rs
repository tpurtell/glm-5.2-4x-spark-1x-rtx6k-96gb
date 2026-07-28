use crate::cli::BenchExpertReductionReplayArgs;
use anyhow::{bail, Context, Result};
use glmrt_core::{
    DType, ExpertBatch, ExpertBatchRoute, ExpertBatchRow, ExpertHostBatchSet, GraphBucket, LayerId,
    ModelFacts, PlacementVersion, PositionId, RequestId, RowSourceKind, GLM52_HIDDEN_SIZE,
    GLM52_TOP_K,
};
use glmrt_transport::{
    protocol_v2_verbs_host_execution_lanes, verbs_host_preflight, ExpertV2Dtype,
    TcpProtocolV2HostBatchSetDispatchStats, TcpProtocolV2HostBatchTarget, TcpTransportConfig,
    VerbsHostProtocolV2HostBatchSetPersistentClient,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    net::ToSocketAddrs,
    path::Path,
    sync::mpsc,
    time::{Duration, Instant},
};

const REPLAY_REQUEST_ID_START: u64 = 80_000_000_000;

#[derive(Debug, Deserialize)]
struct ReplayChain {
    chain_id: String,
    cohort: String,
    physical_m: usize,
    layers: Vec<ReplayLayer>,
}

#[derive(Debug, Deserialize)]
struct ReplayLayer {
    layer_id: u32,
    routes: Vec<Vec<usize>>,
}

#[derive(Clone, Copy, Debug)]
enum ReductionPath {
    Coordinator,
    SparkRowSharded,
}

impl ReductionPath {
    fn label(self) -> &'static str {
        match self {
            Self::Coordinator => "coordinator",
            Self::SparkRowSharded => "spark-row-sharded",
        }
    }
}

struct ReplayMeasurement {
    dispatch_ms: f64,
    wall_ms: f64,
    layer_ms: Vec<f64>,
    request_wire_bytes: usize,
    response_wire_bytes: usize,
    response_chunks: usize,
    executor_ids: BTreeSet<u64>,
}

pub(crate) async fn run_bench_expert_reduction_replay(
    args: BenchExpertReductionReplayArgs,
) -> Result<()> {
    anyhow::ensure!(args.timeout_ms > 0, "--timeout-ms must be positive");
    anyhow::ensure!(
        args.plan.is_file(),
        "replay plan does not exist: {}",
        args.plan.display()
    );
    anyhow::ensure!(
        !args.output.exists(),
        "refusing to overwrite replay output: {}",
        args.output.display()
    );
    let mut chains = load_replay_chains(&args.plan, &args.cohort)?;
    chains.sort_by(|left, right| {
        (left.physical_m, left.chain_id.as_str()).cmp(&(right.physical_m, right.chain_id.as_str()))
    });
    validate_replay_chains(&chains)?;

    let targets = parse_targets(&args.expert_hosts)?;
    let hosts = targets
        .iter()
        .map(|target| target.host.clone())
        .collect::<Vec<_>>();
    verbs_host_preflight().context("verbs-host replay preflight failed")?;
    let config = TcpTransportConfig {
        timeout: Duration::from_millis(args.timeout_ms),
        ..TcpTransportConfig::default()
    };
    let client = VerbsHostProtocolV2HostBatchSetPersistentClient::new(targets, config)?;

    if let Some(parent) = args
        .output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating replay output directory {}", parent.display()))?;
    }
    let output = File::create(&args.output)
        .with_context(|| format!("creating replay output {}", args.output.display()))?;
    let mut output = BufWriter::new(output);
    write_record(
        &mut output,
        &json!({
            "record": "manifest",
            "schema": "glmrt-expert-reduction-replay-result-v1",
            "status": "started",
            "plan": args.plan.canonicalize()?.display().to_string(),
            "cohort": args.cohort,
            "chains": chains.len(),
            "physical_ms": chains.iter().map(|chain| chain.physical_m).collect::<BTreeSet<_>>(),
            "warmup_chains_per_m": args.warmup_chains_per_m,
            "expert_hosts": hosts,
            "execution_lanes": protocol_v2_verbs_host_execution_lanes()?,
            "ingress_dtype": "nvfp4-e2m1-fp8-e4m3",
            "response_dtype": "bf16",
            "spark_reduction_mode": env_value("GLMRT_EXPERT_INTERMEDIATE_REDUCTION"),
            "spark_reduction_dtype": env_value("GLMRT_EXPERT_INTERMEDIATE_REDUCTION_DTYPE"),
            "spark_row_sharded": env_value("GLMRT_EXPERT_INTERMEDIATE_ROW_SHARDED_REDUCTION"),
            "stripe_spark_reduction": env_value("GLMRT_PROTOCOL_V2_VERBS_HOST_STRIPE_SPARK_REDUCTION"),
            "stripe_min_rows": env_value("GLMRT_PROTOCOL_V2_VERBS_HOST_STRIPE_MIN_ROWS"),
        }),
    )?;
    output.flush()?;

    let mut request_id = REPLAY_REQUEST_ID_START;
    let grouped = chains_by_m(&chains);
    for (physical_m, group) in &grouped {
        for chain in group.iter().take(args.warmup_chains_per_m) {
            for path in [ReductionPath::Coordinator, ReductionPath::SparkRowSharded] {
                let _ = replay_chain(&client, &hosts, chain, path, 0, &mut request_id)
                    .await
                    .with_context(|| {
                        format!(
                            "warming M={} chain={} path={}",
                            physical_m,
                            chain.chain_id,
                            path.label()
                        )
                    })?;
            }
        }
    }

    let benchmark_started = Instant::now();
    let mut measurements = 0_usize;
    for (physical_m, group) in grouped {
        for (chain_index, chain) in group.into_iter().enumerate() {
            let first = if chain_index % 2 == 0 {
                ReductionPath::Coordinator
            } else {
                ReductionPath::SparkRowSharded
            };
            let second = match first {
                ReductionPath::Coordinator => ReductionPath::SparkRowSharded,
                ReductionPath::SparkRowSharded => ReductionPath::Coordinator,
            };
            let layer_rotation = chain_index % chain.layers.len();
            for (path_order, path) in [first, second].into_iter().enumerate() {
                let measurement = replay_chain(
                    &client,
                    &hosts,
                    chain,
                    path,
                    layer_rotation,
                    &mut request_id,
                )
                .await
                .with_context(|| {
                    format!(
                        "timing M={} chain={} path={}",
                        physical_m,
                        chain.chain_id,
                        path.label()
                    )
                })?;
                write_measurement(
                    &mut output,
                    chain,
                    path,
                    path_order,
                    layer_rotation,
                    &measurement,
                )?;
                output.flush()?;
                measurements += 1;
            }
        }
    }
    write_record(
        &mut output,
        &json!({
            "record": "complete",
            "status": "complete",
            "measurements": measurements,
            "elapsed_ms": benchmark_started.elapsed().as_secs_f64() * 1_000.0,
        }),
    )?;
    output.flush()?;
    eprintln!(
        "expert reduction replay complete output={} measurements={} elapsed_ms={:.3}",
        args.output.display(),
        measurements,
        benchmark_started.elapsed().as_secs_f64() * 1_000.0
    );
    Ok(())
}

fn load_replay_chains(path: &Path, cohort: &str) -> Result<Vec<ReplayChain>> {
    let input = File::open(path)
        .with_context(|| format!("opening expert reduction replay plan {}", path.display()))?;
    let mut chains = Vec::new();
    for (line_index, line) in BufReader::new(input).lines().enumerate() {
        let line = line.with_context(|| format!("reading replay plan line {}", line_index + 1))?;
        let value: Value = serde_json::from_str(&line)
            .with_context(|| format!("parsing replay plan line {}", line_index + 1))?;
        if value.get("record").and_then(Value::as_str) != Some("chain") {
            continue;
        }
        let chain: ReplayChain = serde_json::from_value(value)
            .with_context(|| format!("decoding replay chain on line {}", line_index + 1))?;
        if chain.cohort == cohort {
            chains.push(chain);
        }
    }
    anyhow::ensure!(
        !chains.is_empty(),
        "replay plan contains no chains for cohort {cohort:?}"
    );
    Ok(chains)
}

fn validate_replay_chains(chains: &[ReplayChain]) -> Result<()> {
    let mut chain_ids = BTreeSet::new();
    for chain in chains {
        anyhow::ensure!(
            chain_ids.insert(chain.chain_id.as_str()),
            "duplicate replay chain ID {}",
            chain.chain_id
        );
        anyhow::ensure!(
            chain.physical_m > 0,
            "replay chain {} has zero physical M",
            chain.chain_id
        );
        anyhow::ensure!(
            !chain.layers.is_empty(),
            "replay chain {} has no layers",
            chain.chain_id
        );
        let mut layer_ids = BTreeSet::new();
        for layer in &chain.layers {
            anyhow::ensure!(
                layer_ids.insert(layer.layer_id),
                "replay chain {} duplicates layer {}",
                chain.chain_id,
                layer.layer_id
            );
            anyhow::ensure!(
                layer.routes.len() == chain.physical_m,
                "replay chain {} layer {} has {} rows, expected {}",
                chain.chain_id,
                layer.layer_id,
                layer.routes.len(),
                chain.physical_m
            );
            anyhow::ensure!(
                layer
                    .routes
                    .iter()
                    .all(|routes| routes.len() == GLM52_TOP_K),
                "replay chain {} layer {} is not top-{}",
                chain.chain_id,
                layer.layer_id,
                GLM52_TOP_K
            );
        }
    }
    Ok(())
}

fn chains_by_m(chains: &[ReplayChain]) -> BTreeMap<usize, Vec<&ReplayChain>> {
    let mut grouped = BTreeMap::<usize, Vec<&ReplayChain>>::new();
    for chain in chains {
        grouped.entry(chain.physical_m).or_default().push(chain);
    }
    grouped
}

async fn replay_chain(
    client: &VerbsHostProtocolV2HostBatchSetPersistentClient,
    hosts: &[String],
    chain: &ReplayChain,
    path: ReductionPath,
    layer_rotation: usize,
    request_id: &mut u64,
) -> Result<ReplayMeasurement> {
    let wall_started = Instant::now();
    let payload = deterministic_nvfp4_payload(chain.physical_m)?;
    let mut dispatch_ms = 0.0_f64;
    let mut layer_ms = Vec::with_capacity(chain.layers.len());
    let mut request_wire_bytes = 0_usize;
    let mut response_wire_bytes = 0_usize;
    let mut response_chunks = 0_usize;
    let mut executor_ids = BTreeSet::new();

    for offset in 0..chain.layers.len() {
        let layer = &chain.layers[(layer_rotation + offset) % chain.layers.len()];
        let (batch, routes) = expert_batch_for_layer(chain, layer)?;
        let set = ExpertHostBatchSet::replicated_from_expert_batch(&batch, &routes, hosts)?;
        let (chunk_tx, chunk_rx) = mpsc::channel();
        let started = Instant::now();
        let stats = match path {
            ReductionPath::Coordinator => {
                client
                    .dispatch_bf16_payload_streaming(
                        &set,
                        &payload,
                        *request_id,
                        ExpertV2Dtype::Bf16,
                        None,
                        false,
                        false,
                        chunk_tx,
                    )
                    .await
            }
            ReductionPath::SparkRowSharded => {
                client
                    .dispatch_bf16_payload_streaming(
                        &set,
                        &payload,
                        *request_id,
                        ExpertV2Dtype::Bf16,
                        Some(0),
                        false,
                        true,
                        chunk_tx,
                    )
                    .await
            }
        }?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
        *request_id = request_id
            .checked_add(4)
            .context("expert reduction replay request ID overflow")?;
        let chunks = chunk_rx.try_iter().count();
        validate_dispatch_stats(&stats, chain.physical_m, path)?;
        dispatch_ms += elapsed_ms;
        layer_ms.push(elapsed_ms);
        request_wire_bytes = request_wire_bytes
            .checked_add(stats.request_wire_bytes)
            .context("replay request wire byte count overflow")?;
        response_wire_bytes = response_wire_bytes
            .checked_add(stats.response_wire_bytes)
            .context("replay response wire byte count overflow")?;
        response_chunks = response_chunks
            .checked_add(chunks)
            .context("replay response chunk count overflow")?;
        executor_ids.extend(stats.response_executor_ids);
    }
    Ok(ReplayMeasurement {
        dispatch_ms,
        wall_ms: wall_started.elapsed().as_secs_f64() * 1_000.0,
        layer_ms,
        request_wire_bytes,
        response_wire_bytes,
        response_chunks,
        executor_ids,
    })
}

fn expert_batch_for_layer(
    chain: &ReplayChain,
    layer: &ReplayLayer,
) -> Result<(ExpertBatch, Vec<ExpertBatchRoute>)> {
    let rows = (0..chain.physical_m)
        .map(|row_index| ExpertBatchRow {
            row_id: row_index as u64,
            source_kind: if row_index == 0 {
                RowSourceKind::DecodeStep
            } else {
                RowSourceKind::MtpVerifyBlock
            },
            request_id: RequestId(chain.chain_id.clone()),
            sequence_id: chain.chain_id.clone(),
            token_position: PositionId(row_index as u64),
            route_offset: row_index * GLM52_TOP_K,
            route_count: GLM52_TOP_K,
        })
        .collect();
    let routes = layer
        .routes
        .iter()
        .enumerate()
        .flat_map(|(row_index, experts)| {
            experts.iter().map(move |expert_id| ExpertBatchRoute {
                row_index,
                expert_id: *expert_id,
                gate_weight: 1.0 / GLM52_TOP_K as f32,
            })
        })
        .collect();
    Ok((
        ExpertBatch {
            layer_id: LayerId(layer.layer_id),
            placement_version: PlacementVersion("expert-reduction-replay-v1".to_owned()),
            hidden_dim: GLM52_HIDDEN_SIZE,
            hidden_bytes_per_row: nvfp4_row_bytes()?,
            hidden_dtype: DType::F4,
            graph_bucket: GraphBucket::new(chain.physical_m),
            quantization_recipe: ModelFacts::default().quantization_recipe,
            rows,
        },
        routes,
    ))
}

fn nvfp4_row_bytes() -> Result<usize> {
    GLM52_HIDDEN_SIZE
        .checked_div(2)
        .and_then(|packed| packed.checked_add(GLM52_HIDDEN_SIZE / 16))
        .context("replay NVFP4 row byte count overflow")
}

fn deterministic_nvfp4_payload(rows: usize) -> Result<Vec<u8>> {
    let packed_values_per_row = GLM52_HIDDEN_SIZE / 2;
    let scales_per_row = GLM52_HIDDEN_SIZE / 16;
    let row_bytes = nvfp4_row_bytes()?;
    let mut payload = Vec::with_capacity(
        rows.checked_mul(row_bytes)
            .context("replay NVFP4 payload byte count overflow")?,
    );
    for row in 0..rows {
        for packed_index in 0..packed_values_per_row {
            // Two finite, nonzero E2M1 values per byte with a deterministic
            // row/value spread. Expert timing does not depend on their exact
            // numeric values, but avoiding all-zero input exercises real math.
            let low = 1 + ((row + packed_index) % 6) as u8;
            let high = 1 + ((row * 3 + packed_index) % 6) as u8;
            payload.push(low | (high << 4));
        }
        // Positive FP8 E4M3 scale near 1.0 (0x38), with a tiny deterministic
        // spread across scale groups.
        for scale_index in 0..scales_per_row {
            payload.push(0x38 + ((row + scale_index) % 3) as u8);
        }
    }
    Ok(payload)
}

fn validate_dispatch_stats(
    stats: &TcpProtocolV2HostBatchSetDispatchStats,
    physical_m: usize,
    path: ReductionPath,
) -> Result<()> {
    anyhow::ensure!(
        stats.global_rows == physical_m,
        "{} replay returned {} global rows, expected {}",
        path.label(),
        stats.global_rows,
        physical_m
    );
    anyhow::ensure!(
        stats.output_dim == GLM52_HIDDEN_SIZE,
        "{} replay returned output width {}, expected {}",
        path.label(),
        stats.output_dim,
        GLM52_HIDDEN_SIZE
    );
    anyhow::ensure!(
        !stats.response_executor_ids.is_empty(),
        "{} replay returned no executor IDs",
        path.label()
    );
    Ok(())
}

fn write_measurement(
    output: &mut impl Write,
    chain: &ReplayChain,
    path: ReductionPath,
    path_order: usize,
    layer_rotation: usize,
    measurement: &ReplayMeasurement,
) -> Result<()> {
    write_record(
        output,
        &json!({
            "record": "measurement",
            "chain_id": chain.chain_id,
            "cohort": chain.cohort,
            "physical_m": chain.physical_m,
            "path": path.label(),
            "path_order": path_order,
            "layer_rotation": layer_rotation,
            "layers": chain.layers.len(),
            "dispatch_ms": measurement.dispatch_ms,
            "wall_ms": measurement.wall_ms,
            "layer_ms": measurement.layer_ms,
            "request_wire_bytes": measurement.request_wire_bytes,
            "response_wire_bytes": measurement.response_wire_bytes,
            "response_chunks": measurement.response_chunks,
            "executor_ids": measurement.executor_ids,
        }),
    )
}

fn write_record(output: &mut impl Write, record: &Value) -> Result<()> {
    serde_json::to_writer(&mut *output, record)?;
    output.write_all(b"\n")?;
    Ok(())
}

fn parse_targets(raw: &str) -> Result<Vec<TcpProtocolV2HostBatchTarget>> {
    let mut targets = Vec::new();
    for entry in raw
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
    {
        let (host, raw_addr) = entry
            .split_once('=')
            .with_context(|| format!("invalid expert target {entry:?}; expected host=addr"))?;
        let host = host.trim();
        anyhow::ensure!(
            !host.is_empty(),
            "expert target {entry:?} has an empty host"
        );
        let addr = raw_addr
            .trim()
            .to_socket_addrs()
            .with_context(|| format!("resolving expert target {entry:?}"))?
            .next()
            .with_context(|| format!("expert target {entry:?} resolved to no addresses"))?;
        anyhow::ensure!(
            !targets
                .iter()
                .any(|target: &TcpProtocolV2HostBatchTarget| target.host == host),
            "duplicate expert target host {host}"
        );
        targets.push(TcpProtocolV2HostBatchTarget {
            host: host.to_owned(),
            addr,
        });
    }
    if targets.is_empty() {
        bail!("--expert-hosts must contain at least one host=addr target");
    }
    Ok(targets)
}

fn env_value(name: &str) -> Option<String> {
    env::var(name).ok().filter(|value| !value.trim().is_empty())
}
