use std::{env, fs, path::PathBuf, time::Instant};

use glmrt_core::{
    KvBlockDescriptor, KvCacheBackingStore, KvCacheConfig, LayerId, PositionId,
    GLM52_NUM_HIDDEN_LAYERS,
};
use serde::Serialize;

const HISTORY_BLOCK_TOKENS: usize = 512;
const DEFAULT_CONTEXTS: &[usize] = &[1_024, 16_384, 131_072, 262_144];
const DEFAULT_DRAFT_TOKENS: usize = 6;
const DEFAULT_REPEATS: usize = 5;
const SEQUENCE_ID: &str = "mtp-kv-transaction-bench";

#[derive(Debug)]
struct Args {
    contexts: Vec<usize>,
    draft_tokens: usize,
    repeats: usize,
    output: Option<PathBuf>,
}

#[derive(Clone)]
struct Fixture {
    store: KvCacheBackingStore,
    reservation_id: u64,
    token_start: usize,
    tentative_write_ids: Vec<Vec<u64>>,
    metadata_records: usize,
}

#[derive(Debug, Serialize)]
struct Report {
    benchmark: &'static str,
    model: ModelReport,
    results: Vec<ResultRow>,
    illustrative_mean_3_4: Vec<WeightedResult>,
}

#[derive(Debug, Serialize)]
struct ModelReport {
    layers: usize,
    history_block_tokens: usize,
    draft_tokens: usize,
    repeats: usize,
    measured_acceptance_prefixes: Vec<usize>,
    illustrative_acceptance_distribution: Option<Vec<f64>>,
    illustrative_acceptance_mean: Option<f64>,
    baseline: &'static str,
    candidate: &'static str,
    timing_scope: &'static str,
}

#[derive(Debug, Serialize)]
struct ResultRow {
    context_tokens: usize,
    history_blocks_per_layer: usize,
    metadata_records: usize,
    accepted_tokens: usize,
    baseline_median_us: f64,
    candidate_median_us: f64,
    speedup: f64,
    baseline_us_per_layer: f64,
    candidate_us_per_layer: f64,
}

#[derive(Debug, Serialize)]
struct WeightedResult {
    context_tokens: usize,
    baseline_weighted_us: f64,
    candidate_weighted_us: f64,
    speedup: f64,
}

fn main() -> Result<(), String> {
    let args = parse_args(env::args().skip(1))?;
    let mut results = Vec::new();
    for &context_tokens in &args.contexts {
        let fixture = build_fixture(context_tokens, args.draft_tokens)?;
        validate_all_acceptance_prefixes(&fixture, args.draft_tokens)?;

        for accepted_tokens in 0..=args.draft_tokens {
            let baseline = measure_us(args.repeats, || {
                let mut store = fixture.store.clone();
                let start = Instant::now();
                resolve_range_baseline(
                    &mut store,
                    fixture.reservation_id,
                    fixture.token_start,
                    args.draft_tokens,
                    accepted_tokens,
                )
                .expect("validated baseline transaction");
                start.elapsed()
            });
            let candidate = measure_us(args.repeats, || {
                let mut store = fixture.store.clone();
                let start = Instant::now();
                resolve_direct_candidate(&mut store, &fixture.tentative_write_ids, accepted_tokens)
                    .expect("validated direct transaction");
                start.elapsed()
            });
            let baseline_median_us = median(&baseline);
            let candidate_median_us = median(&candidate);
            results.push(ResultRow {
                context_tokens,
                history_blocks_per_layer: context_tokens / HISTORY_BLOCK_TOKENS,
                metadata_records: fixture.metadata_records,
                accepted_tokens,
                baseline_median_us,
                candidate_median_us,
                speedup: baseline_median_us / candidate_median_us,
                baseline_us_per_layer: baseline_median_us / GLM52_NUM_HIDDEN_LAYERS as f64,
                candidate_us_per_layer: candidate_median_us / GLM52_NUM_HIDDEN_LAYERS as f64,
            });
        }
    }

    let (distribution, mean) = illustrative_acceptance_distribution(args.draft_tokens);
    let illustrative_mean_3_4 = distribution
        .as_deref()
        .map(|weights| weighted_results(&results, weights))
        .unwrap_or_default();
    let report = Report {
        benchmark: "mtp_kv_transaction",
        model: ModelReport {
            layers: GLM52_NUM_HIDDEN_LAYERS,
            history_block_tokens: HISTORY_BLOCK_TOKENS,
            draft_tokens: args.draft_tokens,
            repeats: args.repeats,
            measured_acceptance_prefixes: (0..=args.draft_tokens).collect(),
            illustrative_acceptance_distribution: distribution,
            illustrative_acceptance_mean: mean,
            baseline: "range resolver plus reservation-wide discarded-payload scan per layer",
            candidate: "retained ordered write IDs plus direct discarded-payload removal",
            timing_scope: "resolution only; fixture construction and clone are excluded",
        },
        results,
        illustrative_mean_3_4,
    };
    print_report(&report);
    if let Some(path) = args.output {
        let json = serde_json::to_vec_pretty(&report)
            .map_err(|error| format!("serialize report: {error}"))?;
        fs::write(&path, json).map_err(|error| format!("write {}: {error}", path.display()))?;
    }
    Ok(())
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let mut contexts = DEFAULT_CONTEXTS.to_vec();
    let mut draft_tokens = DEFAULT_DRAFT_TOKENS;
    let mut repeats = DEFAULT_REPEATS;
    let mut output = None;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        if matches!(arg.as_str(), "--help" | "-h") {
            println!(
                "Usage: cargo run -p glmrt-core --release --example mtp_kv_transaction -- \\\n+  [--contexts 1024,16384,131072,262144] [--draft-tokens 6] \\\n+  [--repeats 5] [--output PATH]"
            );
            std::process::exit(0);
        }
        let value = args
            .next()
            .ok_or_else(|| format!("{arg} requires a value"))?;
        match arg.as_str() {
            "--contexts" => contexts = parse_usize_list(&value, "contexts")?,
            "--draft-tokens" => {
                draft_tokens = parse_positive(&value, "draft-tokens")?;
            }
            "--repeats" => repeats = parse_positive(&value, "repeats")?,
            "--output" => output = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument {arg}")),
        }
    }
    if contexts
        .iter()
        .any(|tokens| *tokens == 0 || tokens % HISTORY_BLOCK_TOKENS != 0)
    {
        return Err(format!(
            "contexts must be positive multiples of {HISTORY_BLOCK_TOKENS}"
        ));
    }
    Ok(Args {
        contexts,
        draft_tokens,
        repeats,
        output,
    })
}

fn parse_usize_list(value: &str, label: &str) -> Result<Vec<usize>, String> {
    let values = value
        .split(',')
        .filter(|item| !item.is_empty())
        .map(|item| parse_positive(item, label))
        .collect::<Result<Vec<_>, _>>()?;
    if values.is_empty() {
        return Err(format!("{label} cannot be empty"));
    }
    Ok(values)
}

fn parse_positive(value: &str, label: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("invalid {label} {value:?}: {error}"))?;
    if parsed == 0 {
        return Err(format!("{label} must be positive"));
    }
    Ok(parsed)
}

fn build_fixture(context_tokens: usize, draft_tokens: usize) -> Result<Fixture, String> {
    let capacity_tokens = context_tokens
        .checked_add(draft_tokens)
        .ok_or_else(|| "context plus draft capacity overflowed usize".to_owned())?;
    let mut store = KvCacheBackingStore::new(KvCacheConfig::glm52_phase0(capacity_tokens));
    let reservation_id = store
        .reserve(SEQUENCE_ID, capacity_tokens)
        .map_err(|error| format!("reserve fixture: {error}"))?;
    let history_blocks = context_tokens / HISTORY_BLOCK_TOKENS;
    let mut tentative_write_ids = Vec::with_capacity(GLM52_NUM_HIDDEN_LAYERS);

    for layer in 0..GLM52_NUM_HIDDEN_LAYERS {
        let layer_id = LayerId(layer as u32);
        for block in 0..history_blocks {
            store
                .write_committed_block_metadata(KvBlockDescriptor {
                    reservation_id,
                    sequence_id: SEQUENCE_ID.to_owned(),
                    layer_id,
                    token_start: PositionId((block * HISTORY_BLOCK_TOKENS) as u64),
                    token_count: HISTORY_BLOCK_TOKENS,
                })
                .map_err(|error| format!("record history block: {error}"))?;
        }
        let ids = (0..draft_tokens)
            .map(|offset| {
                store
                    .write_tentative_block_metadata(KvBlockDescriptor {
                        reservation_id,
                        sequence_id: SEQUENCE_ID.to_owned(),
                        layer_id,
                        token_start: PositionId((context_tokens + offset) as u64),
                        token_count: 1,
                    })
                    .map_err(|error| format!("record tentative block: {error}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        tentative_write_ids.push(ids);
    }
    let metadata_records = GLM52_NUM_HIDDEN_LAYERS * (history_blocks + draft_tokens);
    Ok(Fixture {
        store,
        reservation_id,
        token_start: context_tokens,
        tentative_write_ids,
        metadata_records,
    })
}

fn resolve_range_baseline(
    store: &mut KvCacheBackingStore,
    reservation_id: u64,
    token_start: usize,
    draft_tokens: usize,
    accepted_tokens: usize,
) -> Result<(), String> {
    for layer in 0..GLM52_NUM_HIDDEN_LAYERS {
        store
            .resolve_mtp_tentative_writes(
                reservation_id,
                LayerId(layer as u32),
                PositionId(token_start as u64),
                draft_tokens,
                accepted_tokens,
            )
            .map_err(|error| format!("baseline layer {layer}: {error}"))?;
    }
    Ok(())
}

fn resolve_direct_candidate(
    store: &mut KvCacheBackingStore,
    write_ids_by_layer: &[Vec<u64>],
    accepted_tokens: usize,
) -> Result<(), String> {
    for write_ids in write_ids_by_layer {
        store
            .resolve_mtp_tentative_write_ids(write_ids, accepted_tokens)
            .map_err(|error| format!("direct transaction: {error}"))?;
    }
    Ok(())
}

fn validate_all_acceptance_prefixes(fixture: &Fixture, draft_tokens: usize) -> Result<(), String> {
    for accepted_tokens in 0..=draft_tokens {
        let mut baseline = fixture.store.clone();
        let mut candidate = fixture.store.clone();
        resolve_range_baseline(
            &mut baseline,
            fixture.reservation_id,
            fixture.token_start,
            draft_tokens,
            accepted_tokens,
        )?;
        resolve_direct_candidate(
            &mut candidate,
            &fixture.tentative_write_ids,
            accepted_tokens,
        )?;
        if baseline.snapshot() != candidate.snapshot()
            || baseline.backed_write_count() != candidate.backed_write_count()
            || baseline.backed_write_bytes() != candidate.backed_write_bytes()
        {
            return Err(format!(
                "candidate state differs at accepted prefix {accepted_tokens}"
            ));
        }
        let decode_position = PositionId((fixture.token_start + draft_tokens) as u64);
        let baseline_visible = baseline.read_visible_blocks_for_decode(
            fixture.reservation_id,
            LayerId(0),
            decode_position,
        );
        let candidate_visible = candidate.read_visible_blocks_for_decode(
            fixture.reservation_id,
            LayerId(0),
            decode_position,
        );
        if baseline_visible != candidate_visible {
            return Err(format!(
                "candidate visibility differs at accepted prefix {accepted_tokens}"
            ));
        }
    }
    Ok(())
}

fn measure_us(mut repeats: usize, mut operation: impl FnMut() -> std::time::Duration) -> Vec<f64> {
    let _ = operation();
    let mut samples = Vec::with_capacity(repeats);
    while repeats > 0 {
        samples.push(operation().as_secs_f64() * 1_000_000.0);
        repeats -= 1;
    }
    samples
}

fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

fn illustrative_acceptance_distribution(draft_tokens: usize) -> (Option<Vec<f64>>, Option<f64>) {
    if draft_tokens != 6 {
        return (None, None);
    }
    let weights = vec![0.05, 0.10, 0.15, 0.20, 0.20, 0.20, 0.10];
    let mean = weights
        .iter()
        .enumerate()
        .map(|(accepted, weight)| accepted as f64 * weight)
        .sum();
    (Some(weights), Some(mean))
}

fn weighted_results(results: &[ResultRow], weights: &[f64]) -> Vec<WeightedResult> {
    let mut contexts = results
        .iter()
        .map(|row| row.context_tokens)
        .collect::<Vec<_>>();
    contexts.sort_unstable();
    contexts.dedup();
    contexts
        .into_iter()
        .map(|context_tokens| {
            let rows = results
                .iter()
                .filter(|row| row.context_tokens == context_tokens)
                .collect::<Vec<_>>();
            assert_eq!(rows.len(), weights.len());
            let baseline_weighted_us = rows
                .iter()
                .zip(weights)
                .map(|(row, weight)| row.baseline_median_us * weight)
                .sum();
            let candidate_weighted_us = rows
                .iter()
                .zip(weights)
                .map(|(row, weight)| row.candidate_median_us * weight)
                .sum();
            WeightedResult {
                context_tokens,
                baseline_weighted_us,
                candidate_weighted_us,
                speedup: baseline_weighted_us / candidate_weighted_us,
            }
        })
        .collect()
}

fn print_report(report: &Report) {
    println!("context accepted baseline_us direct_us speedup metadata_records");
    for row in &report.results {
        println!(
            "{} {} {:.3} {:.3} {:.2}x {}",
            row.context_tokens,
            row.accepted_tokens,
            row.baseline_median_us,
            row.candidate_median_us,
            row.speedup,
            row.metadata_records,
        );
    }
    if !report.illustrative_mean_3_4.is_empty() {
        println!("mean_3.4_context baseline_us direct_us speedup");
        for row in &report.illustrative_mean_3_4 {
            println!(
                "{} {:.3} {:.3} {:.2}x",
                row.context_tokens,
                row.baseline_weighted_us,
                row.candidate_weighted_us,
                row.speedup,
            );
        }
    }
}
