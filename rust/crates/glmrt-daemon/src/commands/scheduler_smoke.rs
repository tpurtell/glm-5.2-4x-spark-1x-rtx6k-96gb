use anyhow::Result;
use glmrt_core::{
    admit_layerwaves_for_iteration, plan_prefill_chunks, DecodeStep, LayerWave, PrefillChunkPolicy,
    Priority, GLM52_ROUTED_EXPERTS, GLM52_TOP_K,
};
use serde::Serialize;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::cli::SchedulerSmokeArgs;

#[cfg(test)]
mod tests;

#[derive(Debug, Serialize)]
struct SchedulerSmokeReport {
    benchmark: String,
    prefill_tokens: usize,
    chunk_tokens: usize,
    decode_arrivals: usize,
    decode_period_iterations: usize,
    max_prefill_tokens_per_iteration: usize,
    max_active_prefill_chunks: usize,
    iterations: usize,
    admission_calls: usize,
    scheduler_overhead_us: f64,
    scheduler_overhead_per_call_us: f64,
    selected_decode_rows: usize,
    selected_prefill_rows: usize,
    selected_prefill_chunks: usize,
    deferred_prefill_iterations: usize,
    p50_decode_admission_delay_iterations: usize,
    p99_decode_admission_delay_iterations: usize,
    max_decode_admission_delay_iterations: usize,
    p50_decode_inter_token_iterations: usize,
    p99_decode_inter_token_iterations: usize,
    prefill_completion_iterations: usize,
    prefill_rows_per_expert_avg: f64,
    prefill_rows_per_expert_min: usize,
    prefill_rows_per_expert_max: usize,
    prefill_active_experts: usize,
    ttft_iteration_estimate: usize,
}

pub(crate) fn run_scheduler_smoke(args: SchedulerSmokeArgs) -> Result<()> {
    let report = scheduler_smoke_report(args);
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn scheduler_smoke_report(args: SchedulerSmokeArgs) -> SchedulerSmokeReport {
    let chunk_tokens = args.chunk_tokens.max(1);
    let decode_period_iterations = args.decode_period_iterations.max(1);
    let policy = PrefillChunkPolicy {
        chunk_tokens,
        max_prefill_tokens_per_iteration: args.max_prefill_tokens_per_iteration.max(1),
        max_active_prefill_chunks: args.max_active_prefill_chunks.max(1),
        decode_priority: true,
    };
    let mut pending_prefill = plan_prefill_chunks(
        "scheduler-prefill",
        "background-prefill",
        3_u32,
        args.prefill_tokens,
        1,
        Priority(10),
        &policy,
        "phase0-scheduler-smoke",
    )
    .into_iter()
    .map(LayerWave::prefill)
    .collect::<VecDeque<_>>();
    let mut pending_decodes: VecDeque<(usize, LayerWave)> = VecDeque::new();
    let mut decode_created = 0_usize;
    let mut iteration = 0_usize;
    let mut admission_calls = 0_usize;
    let mut scheduler_overhead = Duration::ZERO;
    let mut selected_decode_rows = 0_usize;
    let mut selected_prefill_rows = 0_usize;
    let mut selected_prefill_chunks = 0_usize;
    let mut deferred_prefill_iterations = 0_usize;
    let mut decode_delays = Vec::new();
    let mut decode_selected_iterations = Vec::new();
    let mut prefill_completion_iterations = 0_usize;
    let mut rows_per_expert = vec![0_usize; GLM52_ROUTED_EXPERTS];

    while !pending_prefill.is_empty()
        || !pending_decodes.is_empty()
        || decode_created < args.decode_arrivals
    {
        if decode_created < args.decode_arrivals && iteration % decode_period_iterations == 0 {
            pending_decodes.push_back((
                iteration,
                LayerWave::decode(DecodeStep::new(
                    format!("decode-{decode_created}"),
                    "foreground-decode",
                    3_u32,
                    decode_created as u64,
                    None,
                    Priority(0),
                    "phase0-scheduler-smoke",
                )),
            ));
            decode_created += 1;
        }

        let candidates = pending_decodes
            .iter()
            .map(|(_, wave)| wave.clone())
            .chain(pending_prefill.iter().cloned())
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            iteration += 1;
            continue;
        }

        let start = Instant::now();
        let admission = admit_layerwaves_for_iteration(candidates, &policy);
        scheduler_overhead += start.elapsed();
        admission_calls += 1;
        if admission
            .deferred
            .iter()
            .any(|wave| matches!(wave.mode, glmrt_core::LayerWaveMode::Prefill))
        {
            deferred_prefill_iterations += 1;
        }

        for selected in admission.selected {
            match selected.mode {
                glmrt_core::LayerWaveMode::Decode => {
                    if let Some((arrival_iteration, wave)) =
                        remove_matching_decode(&mut pending_decodes, &selected)
                    {
                        selected_decode_rows += wave.num_rows();
                        decode_delays.push(iteration.saturating_sub(arrival_iteration));
                        decode_selected_iterations.push(iteration);
                    }
                }
                glmrt_core::LayerWaveMode::Prefill => {
                    if let Some(wave) = remove_matching_wave(&mut pending_prefill, &selected) {
                        selected_prefill_rows += wave.num_rows();
                        selected_prefill_chunks += 1;
                        accumulate_prefill_expert_rows(&wave, &mut rows_per_expert);
                        if selected_prefill_rows >= args.prefill_tokens {
                            prefill_completion_iterations = iteration + 1;
                        }
                    }
                }
                _ => {}
            }
        }

        iteration += 1;
    }

    let decode_intervals = decode_selected_iterations
        .windows(2)
        .map(|window| window[1].saturating_sub(window[0]))
        .collect::<Vec<_>>();
    let active_experts = rows_per_expert.iter().filter(|count| **count > 0).count();
    let total_prefill_expert_rows = rows_per_expert.iter().sum::<usize>();
    let scheduler_overhead_us = scheduler_overhead.as_secs_f64() * 1_000_000.0;
    let scheduler_overhead_per_call_us = if admission_calls > 0 {
        scheduler_overhead_us / admission_calls as f64
    } else {
        0.0
    };
    SchedulerSmokeReport {
        benchmark: "scheduler_prefill_decode_interleaving".to_owned(),
        prefill_tokens: args.prefill_tokens,
        chunk_tokens,
        decode_arrivals: args.decode_arrivals,
        decode_period_iterations,
        max_prefill_tokens_per_iteration: policy.max_prefill_tokens_per_iteration,
        max_active_prefill_chunks: policy.max_active_prefill_chunks,
        iterations: iteration,
        admission_calls,
        scheduler_overhead_us,
        scheduler_overhead_per_call_us,
        selected_decode_rows,
        selected_prefill_rows,
        selected_prefill_chunks,
        deferred_prefill_iterations,
        p50_decode_admission_delay_iterations: percentile_nearest_rank(&decode_delays, 0.50),
        p99_decode_admission_delay_iterations: percentile_nearest_rank(&decode_delays, 0.99),
        max_decode_admission_delay_iterations: decode_delays.iter().copied().max().unwrap_or(0),
        p50_decode_inter_token_iterations: percentile_nearest_rank(&decode_intervals, 0.50),
        p99_decode_inter_token_iterations: percentile_nearest_rank(&decode_intervals, 0.99),
        prefill_completion_iterations,
        prefill_rows_per_expert_avg: total_prefill_expert_rows as f64 / GLM52_ROUTED_EXPERTS as f64,
        prefill_rows_per_expert_min: rows_per_expert.iter().copied().min().unwrap_or(0),
        prefill_rows_per_expert_max: rows_per_expert.iter().copied().max().unwrap_or(0),
        prefill_active_experts: active_experts,
        ttft_iteration_estimate: prefill_completion_iterations,
    }
}

fn remove_matching_decode(
    pending: &mut VecDeque<(usize, LayerWave)>,
    selected: &LayerWave,
) -> Option<(usize, LayerWave)> {
    let index = pending.iter().position(|(_, wave)| wave == selected)?;
    pending.remove(index)
}

fn remove_matching_wave(
    pending: &mut VecDeque<LayerWave>,
    selected: &LayerWave,
) -> Option<LayerWave> {
    let index = pending.iter().position(|wave| wave == selected)?;
    pending.remove(index)
}

fn accumulate_prefill_expert_rows(wave: &LayerWave, rows_per_expert: &mut [usize]) {
    for source in &wave.row_sources {
        for row_offset in 0..source.row_count {
            let token_position = source.token_start.0 as usize + row_offset;
            for slot in 0..GLM52_TOP_K {
                let expert_id = (token_position * GLM52_TOP_K + slot) % GLM52_ROUTED_EXPERTS;
                rows_per_expert[expert_id] += 1;
            }
        }
    }
}

fn percentile_nearest_rank(values: &[usize], percentile: f64) -> usize {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = ((sorted.len() as f64 * percentile).ceil() as usize).max(1);
    sorted[(rank - 1).min(sorted.len() - 1)]
}
