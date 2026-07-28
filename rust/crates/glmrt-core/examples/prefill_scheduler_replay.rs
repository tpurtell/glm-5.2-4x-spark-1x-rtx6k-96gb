use std::{collections::BTreeSet, env, fs, path::PathBuf, time::Instant};

use glmrt_core::{
    plan_completion_first_routes, CompletionRoutePlanEntry, RollingExpertRowPackAccumulator,
    RollingExpertRowPackConfig,
};
use serde::Serialize;

const DEFAULT_ROWS: &[usize] = &[255, 512, 1024, 2048, 4096, 8192, 16384];
const DEFAULT_WINDOWS: &[usize] = &[512, 1024, 2048, 4096];
const OUTPUT_PACK_ROWS: usize = 512;
const OLDEST_QUANTUM_ROWS: usize = 64;
const SELECTION_QUANTUM_ROWS: usize = 32;
const EXPERT_TILE_ROWS: usize = 32;
const MAX_EXPERT_CALL_ROWS: usize = 256;
const HIDDEN_DIM: usize = 6144;
const INTERMEDIATE_ROWS_PER_SPARK: usize = 2048;
const SPARKS: usize = 4;
const REQUEST_HEADER_BYTES: usize = 96;
const RESPONSE_HEADER_BYTES: usize = 96;
const ROW_DESCRIPTOR_BYTES: usize = 40;
const ROUTE_DESCRIPTOR_BYTES: usize = 10;
const ROW_INDEX_BYTES: usize = 4;
const NVFP4_REQUEST_ROW_BYTES: usize = HIDDEN_DIM / 2 + HIDDEN_DIM / 16;
const FP8_RESPONSE_ROW_BYTES: usize = HIDDEN_DIM + size_of::<f32>();

// Rotated-weight GB10 measurements from S-MLP-02 in
// learnings/experiments/kernel-tune-history.md.
const SPARK_W4A16_US: &[(usize, f64)] = &[
    (1, 36.689),
    (2, 36.896),
    (4, 41.355),
    (8, 66.736),
    (16, 39.997),
    (32, 46.894),
    (64, 67.362),
    (128, 118.494),
    (256, 217.239),
];

#[derive(Debug)]
struct Args {
    trace_log: PathBuf,
    output: Option<PathBuf>,
    rows: Vec<usize>,
    windows: Vec<usize>,
    layer_limit: usize,
}

#[derive(Clone, Debug)]
struct RouteTrace {
    request_id_base: u64,
    rows: Vec<Vec<usize>>,
}

#[derive(Clone, Debug)]
struct PolicyPlan {
    name: String,
    lookahead_rows: usize,
    packs: Vec<Vec<usize>>,
}

#[derive(Clone, Debug, Default)]
struct LayerMetrics {
    planner_us: f64,
    modeled_us: f64,
    first_completion_us: f64,
    p50_completion_us: f64,
    calls: usize,
    completion_frames: usize,
    route_rows: usize,
    bucket_rows: usize,
    max_reorder_rows: usize,
    mean_reorder_rows: f64,
    max_delay_packs: usize,
    first_pack_required_rows: usize,
}

#[derive(Debug, Serialize)]
struct ReplayReport {
    source: SourceReport,
    model: ModelReport,
    results: Vec<ReplayResult>,
}

#[derive(Debug, Serialize)]
struct SourceReport {
    trace_log: String,
    parsed_full_512_row_traces: usize,
    replayed_traces: usize,
    sampling: &'static str,
    routes_per_row_mean: f64,
    active_experts_per_trace_mean: f64,
    request_id_bases: Vec<u64>,
}

#[derive(Debug, Serialize)]
struct ModelReport {
    output_pack_rows: usize,
    oldest_progress_quantum_rows: usize,
    selection_quantum_rows: usize,
    expert_tile_rows: usize,
    max_expert_call_rows: usize,
    spark_count: usize,
    request_dtype: &'static str,
    response_dtype: &'static str,
    request_row_bytes: usize,
    response_row_bytes: usize,
    spark_w4a16_us_by_bucket: Vec<(usize, f64)>,
    limitations: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
struct ReplayResult {
    rows: usize,
    policy: String,
    lookahead_rows: usize,
    planner_ms_mean: f64,
    modeled_spark_ms_mean: f64,
    modeled_spark_speedup_vs_contiguous: f64,
    first_completion_us_mean: f64,
    p50_completion_us_mean: f64,
    expert_calls_mean: f64,
    expert_call_reduction_vs_contiguous: f64,
    graph_bucket_fill: f64,
    max_reorder_rows: usize,
    mean_reorder_rows: f64,
    max_delay_packs: usize,
    first_pack_required_rows: usize,
    logical_request_bytes_four_sparks: usize,
    logical_response_bytes_four_sparks: usize,
    response_frame_overhead_bytes_mean_four_sparks: f64,
    weight_stream_gib_mean_per_spark: f64,
}

fn main() -> Result<(), String> {
    let args = parse_args(env::args().skip(1))?;
    let input = fs::read_to_string(&args.trace_log)
        .map_err(|error| format!("read {}: {error}", args.trace_log.display()))?;
    let mut traces = parse_route_trace_log(&input)?;
    traces.retain(|trace| trace.rows.len() == OUTPUT_PACK_ROWS);
    if traces.is_empty() {
        return Err(format!(
            "{} contains no complete {OUTPUT_PACK_ROWS}-row route traces",
            args.trace_log.display()
        ));
    }
    let parsed_full_trace_count = traces.len();
    traces = evenly_sample_traces(&traces, args.layer_limit);

    let report = replay(&args, &traces, parsed_full_trace_count)?;
    print_report(&report);
    if let Some(path) = &args.output {
        let bytes = serde_json::to_vec_pretty(&report)
            .map_err(|error| format!("serialize replay report: {error}"))?;
        fs::write(path, bytes).map_err(|error| format!("write {}: {error}", path.display()))?;
    }
    Ok(())
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let mut trace_log = None;
    let mut output = None;
    let mut rows = DEFAULT_ROWS.to_vec();
    let mut windows = DEFAULT_WINDOWS.to_vec();
    let mut layer_limit = 12;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let value = match arg.as_str() {
            "--trace-log" | "--output" | "--rows" | "--windows" | "--layers" => args
                .next()
                .ok_or_else(|| format!("{arg} requires a value"))?,
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run -p glmrt-core --example prefill_scheduler_replay -- \\\n+  --trace-log PATH [--rows 255,512,...] [--windows 512,1024,...] \\\n+  [--layers 12] [--output PATH]"
                );
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument {arg}")),
        };
        match arg.as_str() {
            "--trace-log" => trace_log = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            "--rows" => rows = parse_usize_list(&value, "row count")?,
            "--windows" => windows = parse_usize_list(&value, "lookahead window")?,
            "--layers" => {
                layer_limit = value
                    .parse::<usize>()
                    .map_err(|error| format!("invalid layer count {value}: {error}"))?;
                if layer_limit == 0 {
                    return Err("layer count must be non-zero".to_owned());
                }
            }
            _ => unreachable!(),
        }
    }
    let trace_log = trace_log.ok_or_else(|| "--trace-log is required".to_owned())?;
    if rows.contains(&0) {
        return Err("row counts must be non-zero".to_owned());
    }
    if windows.iter().any(|window| *window < OUTPUT_PACK_ROWS) {
        return Err(format!(
            "lookahead windows must be at least {OUTPUT_PACK_ROWS} rows"
        ));
    }
    rows.sort_unstable();
    rows.dedup();
    windows.sort_unstable();
    windows.dedup();
    Ok(Args {
        trace_log,
        output,
        rows,
        windows,
        layer_limit,
    })
}

fn parse_usize_list(value: &str, label: &str) -> Result<Vec<usize>, String> {
    let parsed = value
        .split(',')
        .map(|item| {
            item.parse::<usize>()
                .map_err(|error| format!("invalid {label} {item}: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if parsed.is_empty() {
        return Err(format!("{label} list must not be empty"));
    }
    Ok(parsed)
}

fn parse_route_trace_log(input: &str) -> Result<Vec<RouteTrace>, String> {
    const MARKER: &str = "protocol_v2_expert_queue_row_routes ";
    let mut traces = Vec::new();
    for (line_number, line) in input.lines().enumerate() {
        let Some(marker_offset) = line.find(MARKER) else {
            continue;
        };
        let fields = &line[marker_offset + MARKER.len()..];
        let request_id_base = parse_field(fields, "request_id_base=")?
            .parse::<u64>()
            .map_err(|error| {
                format!("line {} invalid request_id_base: {error}", line_number + 1)
            })?;
        let declared_rows = parse_field(fields, "rows=")?
            .parse::<usize>()
            .map_err(|error| format!("line {} invalid row count: {error}", line_number + 1))?;
        let routes = fields
            .split_once("row_routes=")
            .map(|(_, value)| value)
            .ok_or_else(|| format!("line {} has no row_routes field", line_number + 1))?;
        let mut rows = vec![None; declared_rows];
        for row in routes.split(',') {
            let (row_index, experts) = row
                .split_once(':')
                .ok_or_else(|| format!("line {} malformed route row {row}", line_number + 1))?;
            let row_index = row_index.parse::<usize>().map_err(|error| {
                format!(
                    "line {} invalid route row {row_index}: {error}",
                    line_number + 1
                )
            })?;
            if row_index >= declared_rows {
                return Err(format!(
                    "line {} route row {row_index} exceeds declared rows {declared_rows}",
                    line_number + 1
                ));
            }
            let experts = experts
                .split('+')
                .map(|expert| {
                    expert.parse::<usize>().map_err(|error| {
                        format!("line {} invalid expert {expert}: {error}", line_number + 1)
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            if experts.is_empty() {
                return Err(format!(
                    "line {} row {row_index} has no routes",
                    line_number + 1
                ));
            }
            if experts.iter().collect::<BTreeSet<_>>().len() != experts.len() {
                return Err(format!(
                    "line {} row {row_index} repeats an expert",
                    line_number + 1
                ));
            }
            if rows[row_index].replace(experts).is_some() {
                return Err(format!(
                    "line {} repeats route row {row_index}",
                    line_number + 1
                ));
            }
        }
        let rows = rows
            .into_iter()
            .enumerate()
            .map(|(row_index, routes)| {
                routes
                    .ok_or_else(|| format!("line {} omits route row {row_index}", line_number + 1))
            })
            .collect::<Result<Vec<_>, _>>()?;
        traces.push(RouteTrace {
            request_id_base,
            rows,
        });
    }
    Ok(traces)
}

fn parse_field<'a>(fields: &'a str, name: &str) -> Result<&'a str, String> {
    let start = fields
        .find(name)
        .ok_or_else(|| format!("trace line has no {name} field"))?
        + name.len();
    Ok(fields[start..]
        .split_ascii_whitespace()
        .next()
        .unwrap_or_default())
}

fn evenly_sample_traces(traces: &[RouteTrace], limit: usize) -> Vec<RouteTrace> {
    if traces.len() <= limit {
        return traces.to_vec();
    }
    (0..limit)
        .map(|sample| {
            let index = sample * traces.len() / limit;
            traces[index].clone()
        })
        .collect()
}

fn replay(
    args: &Args,
    traces: &[RouteTrace],
    parsed_full_trace_count: usize,
) -> Result<ReplayReport, String> {
    let mut results = Vec::new();
    for row_count in &args.rows {
        let mut policy_metrics = Vec::<(PolicyPlan, Vec<LayerMetrics>)>::new();
        for policy in policies(*row_count, &args.windows)? {
            let mut metrics = Vec::with_capacity(traces.len());
            for trace in traces {
                let rows = expand_trace_rows(trace, *row_count);
                metrics.push(evaluate_layer(&rows, &policy)?);
            }
            policy_metrics.push((policy, metrics));
        }
        let baseline = policy_metrics
            .first()
            .ok_or_else(|| "replay produced no policies".to_owned())?;
        let baseline_time = mean(&baseline.1, |metric| metric.modeled_us);
        let baseline_calls = mean(&baseline.1, |metric| metric.calls as f64);
        for (policy, metrics) in policy_metrics {
            let modeled_us = mean(&metrics, |metric| metric.modeled_us);
            let calls = mean(&metrics, |metric| metric.calls as f64);
            let route_rows = mean(&metrics, |metric| metric.route_rows as f64);
            let bucket_rows = mean(&metrics, |metric| metric.bucket_rows as f64);
            let completion_frames = mean(&metrics, |metric| metric.completion_frames as f64);
            let route_count = expand_trace_rows(&traces[0], *row_count)
                .iter()
                .map(Vec::len)
                .sum::<usize>();
            let (request_bytes, response_bytes) = logical_wire_bytes(*row_count, route_count);
            let weight_bytes_per_call = HIDDEN_DIM * INTERMEDIATE_ROWS_PER_SPARK * 3 / 2;
            results.push(ReplayResult {
                rows: *row_count,
                policy: policy.name,
                lookahead_rows: policy.lookahead_rows,
                planner_ms_mean: mean(&metrics, |metric| metric.planner_us) / 1000.0,
                modeled_spark_ms_mean: modeled_us / 1000.0,
                modeled_spark_speedup_vs_contiguous: baseline_time / modeled_us,
                first_completion_us_mean: mean(&metrics, |metric| metric.first_completion_us),
                p50_completion_us_mean: mean(&metrics, |metric| metric.p50_completion_us),
                expert_calls_mean: calls,
                expert_call_reduction_vs_contiguous: 1.0 - calls / baseline_calls,
                graph_bucket_fill: route_rows / bucket_rows,
                max_reorder_rows: metrics
                    .iter()
                    .map(|metric| metric.max_reorder_rows)
                    .max()
                    .unwrap_or(0),
                mean_reorder_rows: mean(&metrics, |metric| metric.mean_reorder_rows),
                max_delay_packs: metrics
                    .iter()
                    .map(|metric| metric.max_delay_packs)
                    .max()
                    .unwrap_or(0),
                first_pack_required_rows: metrics
                    .iter()
                    .map(|metric| metric.first_pack_required_rows)
                    .max()
                    .unwrap_or(0),
                logical_request_bytes_four_sparks: request_bytes,
                logical_response_bytes_four_sparks: response_bytes,
                response_frame_overhead_bytes_mean_four_sparks: completion_frames
                    * RESPONSE_HEADER_BYTES as f64
                    * SPARKS as f64,
                weight_stream_gib_mean_per_spark: calls * weight_bytes_per_call as f64
                    / 1024.0_f64.powi(3),
            });
        }
    }
    Ok(ReplayReport {
        source: SourceReport {
            trace_log: args.trace_log.display().to_string(),
            parsed_full_512_row_traces: parsed_full_trace_count,
            replayed_traces: traces.len(),
            sampling: "evenly spaced over complete 512-row trace lines",
            routes_per_row_mean: traces
                .iter()
                .flat_map(|trace| trace.rows.iter())
                .map(|routes| routes.len() as f64)
                .sum::<f64>()
                / traces.iter().map(|trace| trace.rows.len()).sum::<usize>() as f64,
            active_experts_per_trace_mean: traces
                .iter()
                .map(|trace| {
                    trace
                        .rows
                        .iter()
                        .flatten()
                        .copied()
                        .collect::<BTreeSet<_>>()
                        .len() as f64
                })
                .sum::<f64>()
                / traces.len() as f64,
            request_id_bases: traces.iter().map(|trace| trace.request_id_base).collect(),
        },
        model: ModelReport {
            output_pack_rows: OUTPUT_PACK_ROWS,
            oldest_progress_quantum_rows: OLDEST_QUANTUM_ROWS,
            selection_quantum_rows: SELECTION_QUANTUM_ROWS,
            expert_tile_rows: EXPERT_TILE_ROWS,
            max_expert_call_rows: MAX_EXPERT_CALL_ROWS,
            spark_count: SPARKS,
            request_dtype: "NVFP4 E2M1 + FP8 E4M3 block scales",
            response_dtype: "row-scaled FP8 E4M3",
            request_row_bytes: NVFP4_REQUEST_ROW_BYTES,
            response_row_bytes: FP8_RESPONSE_ROW_BYTES,
            spark_w4a16_us_by_bucket: SPARK_W4A16_US.to_vec(),
            limitations: vec![
                "comparative Spark MLP model, not an end-to-end throughput prediction",
                "uses measured isolated per-expert W4A16 bucket times",
                "assumes current four-way intermediate sharding gives every Spark the same routes",
                "does not model coordinator routing, overlap, RDMA contention, or aggregation kernels",
                "traces longer than 512 rows rotate the measured 512-row route sequence",
            ],
        },
        results,
    })
}

fn policies(row_count: usize, windows: &[usize]) -> Result<Vec<PolicyPlan>, String> {
    let contiguous = (0..row_count)
        .collect::<Vec<_>>()
        .chunks(OUTPUT_PACK_ROWS)
        .map(<[usize]>::to_vec)
        .collect::<Vec<_>>();
    let mut policies = vec![PolicyPlan {
        name: "contiguous".to_owned(),
        lookahead_rows: OUTPUT_PACK_ROWS,
        packs: contiguous,
    }];
    for window in windows {
        if *window == OUTPUT_PACK_ROWS {
            continue;
        }
        policies.push(PolicyPlan {
            name: format!("rolling-{window}"),
            lookahead_rows: *window,
            packs: Vec::new(),
        });
    }
    Ok(policies)
}

fn expand_trace_rows(trace: &RouteTrace, row_count: usize) -> Vec<Vec<usize>> {
    let source_rows = trace.rows.len();
    (0..row_count)
        .map(|row_index| {
            let block = row_index / source_rows;
            let within_block = row_index % source_rows;
            let rotated = (within_block + block * 137) % source_rows;
            trace.rows[rotated].clone()
        })
        .collect()
}

fn evaluate_layer(rows: &[Vec<usize>], policy: &PolicyPlan) -> Result<LayerMetrics, String> {
    let planner_started = Instant::now();
    let (packs, first_pack_required_rows) = if policy.packs.is_empty() {
        rolling_stream_packs(rows, policy.lookahead_rows)?
    } else {
        let first_pack_required_rows = policy
            .packs
            .first()
            .and_then(|pack| pack.iter().max())
            .map(|row| row + 1)
            .unwrap_or(0);
        (policy.packs.clone(), first_pack_required_rows)
    };
    let planner_us = planner_started.elapsed().as_secs_f64() * 1e6;
    validate_packs(&packs, rows.len())?;

    let mut metrics = LayerMetrics {
        planner_us,
        ..LayerMetrics::default()
    };
    let mut completion_times = vec![0.0_f64; rows.len()];
    let mut emitted_position = 0_usize;
    for (pack_index, pack) in packs.iter().enumerate() {
        let pack_rows = pack
            .iter()
            .map(|row_index| rows[*row_index].clone())
            .collect::<Vec<_>>();
        let pack_entries = completion_entries(&pack_rows);
        let completion_plan =
            plan_completion_first_routes(&pack_entries, pack_rows.len(), MAX_EXPERT_CALL_ROWS)
                .map_err(|error| error.to_string())?;
        let mut completion_slices = Vec::new();
        for group in completion_plan.groups {
            let rows_in_call = group.route_indices.len();
            let bucket = graph_bucket(rows_in_call)?;
            metrics.modeled_us += spark_call_us(bucket)?;
            metrics.calls += 1;
            metrics.route_rows += rows_in_call;
            metrics.bucket_rows += bucket;
            if !group.completed_rows.is_empty() {
                completion_slices.push((metrics.modeled_us, group.completed_rows));
            }
        }
        emit_coalesced_completions(
            &completion_slices,
            pack,
            &mut completion_times,
            &mut metrics.completion_frames,
        );
        for row_index in pack {
            let displacement = emitted_position.abs_diff(*row_index);
            metrics.max_reorder_rows = metrics.max_reorder_rows.max(displacement);
            metrics.mean_reorder_rows += displacement as f64;
            let original_pack = row_index / OUTPUT_PACK_ROWS;
            metrics.max_delay_packs = metrics
                .max_delay_packs
                .max(pack_index.saturating_sub(original_pack));
            emitted_position += 1;
        }
    }
    metrics.mean_reorder_rows /= rows.len() as f64;
    metrics.first_pack_required_rows = first_pack_required_rows;
    completion_times.sort_by(f64::total_cmp);
    metrics.first_completion_us = completion_times[0];
    metrics.p50_completion_us = completion_times[(completion_times.len() - 1) / 2];
    Ok(metrics)
}

fn rolling_stream_packs(
    rows: &[Vec<usize>],
    lookahead_rows: usize,
) -> Result<(Vec<Vec<usize>>, usize), String> {
    let mut accumulator = RollingExpertRowPackAccumulator::new(RollingExpertRowPackConfig {
        logical_chunk_rows: OLDEST_QUANTUM_ROWS.min(rows.len()),
        max_pack_rows: OUTPUT_PACK_ROWS,
        lookahead_rows,
        expert_tile_rows: EXPERT_TILE_ROWS,
        selection_quantum_rows: SELECTION_QUANTUM_ROWS,
    })
    .map_err(|error| error.to_string())?;
    let mut emissions = Vec::new();
    for row_start in (0..rows.len()).step_by(OUTPUT_PACK_ROWS) {
        let row_end = (row_start + OUTPUT_PACK_ROWS).min(rows.len());
        let entries = completion_entries_with_offset(&rows[row_start..row_end], row_start);
        emissions.extend(
            accumulator
                .push_chunk(&entries, row_end - row_start)
                .map_err(|error| error.to_string())?,
        );
    }
    emissions.extend(accumulator.finish().map_err(|error| error.to_string())?);
    let first_pack_required_rows = emissions
        .first()
        .map(|emission| emission.admitted_rows)
        .unwrap_or(0);
    Ok((
        emissions
            .into_iter()
            .map(|emission| emission.row_indices)
            .collect(),
        first_pack_required_rows,
    ))
}

fn emit_coalesced_completions(
    slices: &[(f64, Vec<usize>)],
    pack: &[usize],
    completion_times: &mut [f64],
    frame_count: &mut usize,
) {
    let mut pending_rows = Vec::new();
    let mut pending_ready_us = 0.0_f64;
    let mut emitted_frames = 0_usize;
    for (slice_index, (ready_us, rows)) in slices.iter().enumerate() {
        if !pending_rows.is_empty() && pending_rows.len() + rows.len() > MAX_EXPERT_CALL_ROWS {
            record_completion_frame(
                &pending_rows,
                pending_ready_us,
                pack,
                completion_times,
                frame_count,
            );
            pending_rows.clear();
            emitted_frames += 1;
        }
        pending_rows.extend_from_slice(rows);
        pending_ready_us = *ready_us;
        let target_rows = if emitted_frames == 0 {
            32
        } else {
            MAX_EXPERT_CALL_ROWS
        };
        if pending_rows.len() >= target_rows || slice_index + 1 == slices.len() {
            record_completion_frame(
                &pending_rows,
                pending_ready_us,
                pack,
                completion_times,
                frame_count,
            );
            pending_rows.clear();
            emitted_frames += 1;
        }
    }
    debug_assert!(pending_rows.is_empty());
}

fn record_completion_frame(
    local_rows: &[usize],
    ready_us: f64,
    pack: &[usize],
    completion_times: &mut [f64],
    frame_count: &mut usize,
) {
    *frame_count += 1;
    for local_row in local_rows {
        completion_times[pack[*local_row]] = ready_us;
    }
}

fn completion_entries(rows: &[Vec<usize>]) -> Vec<CompletionRoutePlanEntry> {
    completion_entries_with_offset(rows, 0)
}

fn completion_entries_with_offset(
    rows: &[Vec<usize>],
    row_offset: usize,
) -> Vec<CompletionRoutePlanEntry> {
    rows.iter()
        .enumerate()
        .flat_map(|(row_index, experts)| {
            experts
                .iter()
                .map(move |expert_id| CompletionRoutePlanEntry {
                    row_index: row_offset + row_index,
                    expert_id: *expert_id,
                    intermediate_rows: INTERMEDIATE_ROWS_PER_SPARK,
                })
        })
        .collect()
}

fn validate_packs(packs: &[Vec<usize>], row_count: usize) -> Result<(), String> {
    if packs
        .iter()
        .any(|pack| pack.is_empty() || pack.len() > OUTPUT_PACK_ROWS)
    {
        return Err("replay produced an empty or oversized output pack".to_owned());
    }
    let mut emitted = packs.iter().flatten().copied().collect::<Vec<_>>();
    emitted.sort_unstable();
    if emitted != (0..row_count).collect::<Vec<_>>() {
        return Err("replay lost, duplicated, or invented rows".to_owned());
    }
    Ok(())
}

fn graph_bucket(rows: usize) -> Result<usize, String> {
    SPARK_W4A16_US
        .iter()
        .map(|(bucket, _)| *bucket)
        .find(|bucket| *bucket >= rows)
        .ok_or_else(|| format!("expert call with {rows} rows exceeds measured bucket table"))
}

fn logical_wire_bytes(row_count: usize, route_count: usize) -> (usize, usize) {
    let pack_count = row_count.div_ceil(OUTPUT_PACK_ROWS);
    let request_bytes = SPARKS
        * (pack_count * REQUEST_HEADER_BYTES
            + row_count * (ROW_DESCRIPTOR_BYTES + NVFP4_REQUEST_ROW_BYTES)
            + route_count * ROUTE_DESCRIPTOR_BYTES);
    let response_bytes = SPARKS
        * (pack_count * RESPONSE_HEADER_BYTES
            + row_count * (ROW_INDEX_BYTES + FP8_RESPONSE_ROW_BYTES));
    (request_bytes, response_bytes)
}

fn spark_call_us(bucket: usize) -> Result<f64, String> {
    SPARK_W4A16_US
        .iter()
        .find_map(|(candidate, us)| (*candidate == bucket).then_some(*us))
        .ok_or_else(|| format!("no Spark timing for {bucket}-row bucket"))
}

fn mean(metrics: &[LayerMetrics], value: impl Fn(&LayerMetrics) -> f64) -> f64 {
    metrics.iter().map(value).sum::<f64>() / metrics.len() as f64
}

fn print_report(report: &ReplayReport) {
    println!(
        "source={} traces={}/{} output_pack={} rows",
        report.source.trace_log,
        report.source.replayed_traces,
        report.source.parsed_full_512_row_traces,
        report.model.output_pack_rows
    );
    println!(
        "rows policy       lookahead plan_ms modeled_ms speedup calls  call_red fill   first_us p50_us first_need max_delay"
    );
    for result in &report.results {
        println!(
            "{:<5} {:<12} {:<9} {:<7.3} {:<10.3} {:<7.3} {:<6.1} {:>7.1}% {:>5.1}% {:<8.1} {:<7.1} {:<10} {}",
            result.rows,
            result.policy,
            result.lookahead_rows,
            result.planner_ms_mean,
            result.modeled_spark_ms_mean,
            result.modeled_spark_speedup_vs_contiguous,
            result.expert_calls_mean,
            result.expert_call_reduction_vs_contiguous * 100.0,
            result.graph_bucket_fill * 100.0,
            result.first_completion_us_mean,
            result.p50_completion_us_mean,
            result.first_pack_required_rows,
            result.max_delay_packs,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_route_trace_line() {
        let input = "noise\nprotocol_v2_expert_queue_row_routes request_id_base=65536 rows=3 row_routes=0:1+2,1:2+3,2:4\n";
        let traces = parse_route_trace_log(input).unwrap();
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].request_id_base, 65536);
        assert_eq!(traces[0].rows, vec![vec![1, 2], vec![2, 3], vec![4]]);
    }

    #[test]
    fn expanded_trace_rotates_repeated_blocks() {
        let trace = RouteTrace {
            request_id_base: 0,
            rows: (0..4).map(|expert| vec![expert]).collect(),
        };
        let rows = expand_trace_rows(&trace, 8);
        assert_eq!(
            rows,
            vec![
                vec![0],
                vec![1],
                vec![2],
                vec![3],
                vec![1],
                vec![2],
                vec![3],
                vec![0]
            ]
        );
    }

    #[test]
    fn trace_sampling_spans_the_full_log() {
        let traces = (0..12)
            .map(|request_id_base| RouteTrace {
                request_id_base,
                rows: vec![vec![0]],
            })
            .collect::<Vec<_>>();
        let sampled = evenly_sample_traces(&traces, 4);
        assert_eq!(
            sampled
                .iter()
                .map(|trace| trace.request_id_base)
                .collect::<Vec<_>>(),
            vec![0, 3, 6, 9]
        );
    }

    #[test]
    fn rolling_replay_preserves_every_row_and_call_cap() {
        let rows = (0..1024)
            .map(|row| vec![row % 17, (row + 3) % 17])
            .collect::<Vec<_>>();
        let policy = PolicyPlan {
            name: "rolling-1024".to_owned(),
            lookahead_rows: 1024,
            packs: Vec::new(),
        };
        let metrics = evaluate_layer(&rows, &policy).unwrap();
        assert_eq!(metrics.route_rows, 2048);
        assert!(metrics.bucket_rows >= metrics.route_rows);
        assert!(metrics.first_pack_required_rows <= 1024);
        assert!(metrics.max_delay_packs <= 1);
    }

    #[test]
    fn contiguous_replay_uses_exact_pack_boundaries() {
        let policy = policies(1025, &[]).unwrap().remove(0);
        assert_eq!(
            policy.packs.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![512, 512, 1]
        );
        validate_packs(&policy.packs, 1025).unwrap();
    }

    #[test]
    fn wire_row_sizes_match_protocol_v2() {
        assert_eq!(NVFP4_REQUEST_ROW_BYTES, 3456);
        assert_eq!(FP8_RESPONSE_ROW_BYTES, 6148);
        let (request, response) = logical_wire_bytes(1024, 8192);
        assert_eq!(request, SPARKS * (2 * 96 + 1024 * 3496 + 8192 * 10));
        assert_eq!(response, SPARKS * (2 * 96 + 1024 * 6152));
    }
}
