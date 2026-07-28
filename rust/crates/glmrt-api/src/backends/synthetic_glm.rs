use glmrt_core::{
    plan_prefill_chunks, DecodeStep, ExpertRequest, ExpertRow, ExpertWaveMetadata,
    KvCacheAllocator, KvCacheConfig, LayerWave, PrefillChunkPolicy, Priority, RouteEntry,
    GLM52_HIDDEN_SIZE, GLM52_TOP_K,
};
use std::sync::atomic::Ordering;
use std::time::Instant;

use crate::metrics::BackendMetrics;
use crate::{
    dispatch_expert_request, duration_ms, runtime_error, sum_partials, ApiError, ApiState,
    BackendCompletion,
};

pub(crate) async fn synthetic_glm_layer_completion(
    state: &ApiState,
    prompt: &str,
    prompt_tokens: usize,
) -> Result<BackendCompletion, ApiError> {
    let prompt_tokens = prompt_tokens.max(1);
    let prefill_rows = prompt_tokens;
    let mut kv_allocator =
        KvCacheAllocator::new(KvCacheConfig::glm52_phase0(prefill_rows + 1 + GLM52_TOP_K));
    let kv_reservation_id = kv_allocator
        .reserve("synthetic-glm-layer", prefill_rows + 1)
        .map_err(runtime_error)?;
    let policy = PrefillChunkPolicy::latency_smoke(16);
    let prefill_chunks = plan_prefill_chunks(
        "synthetic-prefill",
        "synthetic-glm-layer",
        3,
        prefill_rows,
        kv_reservation_id,
        Priority(10),
        &policy,
        "phase0-synthetic-glm-layer",
    );
    let route_groups = partition_routes(&state.config.expert_targets);
    let total_requests = route_groups.len() * (prefill_chunks.len() + 1);
    let request_id = state
        .next_request_id
        .fetch_add(total_requests as u64, Ordering::Relaxed);
    let mut request_offset = 0_u64;
    let mut prefill_partial_count = 0_usize;
    let mut prefill_checksum = 0.0_f64;

    let prefill_start = Instant::now();
    for chunk in prefill_chunks.iter().cloned() {
        let wave = LayerWave::prefill(chunk);
        for (target, routes) in &route_groups {
            let rows = (0..wave.num_rows())
                .map(|row_idx| {
                    let token_pos = wave.row_sources[0].token_start.0 + row_idx as u64;
                    ExpertRow {
                        row_id: token_pos,
                        hidden: synthetic_hidden(&format!("{prompt}\nprefill:{token_pos}")),
                        routes: routes.clone(),
                    }
                })
                .collect::<Vec<_>>();
            let request = ExpertRequest {
                protocol_version: 1,
                request_id: request_id + request_offset,
                placement_version: "phase0-synthetic-glm-layer".to_owned(),
                layer_id: wave.layer_id.0,
                hidden_dim: GLM52_HIDDEN_SIZE as u32,
                wave: Some(ExpertWaveMetadata::from_wave(&wave)),
                rows,
            };
            request_offset += 1;
            let response = dispatch_expert_request(state, target.as_deref(), &request).await?;
            if response.partial_outputs.len() != wave.num_rows() {
                return Err(runtime_error(format!(
                    "prefill expert response had {} rows, expected {}",
                    response.partial_outputs.len(),
                    wave.num_rows()
                )));
            }
            prefill_partial_count += response.partial_outputs.len();
            prefill_checksum += response
                .partial_outputs
                .iter()
                .flatten()
                .map(|value| *value as f64)
                .sum::<f64>();
        }
    }
    let prefill_ms = duration_ms(prefill_start.elapsed());

    let decode_wave = LayerWave::decode(DecodeStep::new(
        "synthetic-decode",
        "synthetic-glm-layer",
        3,
        prefill_rows as u64,
        Some(kv_reservation_id),
        Priority(0),
        "phase0-synthetic-glm-layer",
    ));
    let decode_start = Instant::now();
    let hidden = synthetic_hidden(prompt);
    let mut partials = Vec::with_capacity(route_groups.len());
    for (target, routes) in &route_groups {
        let request = ExpertRequest {
            protocol_version: 1,
            request_id: request_id + request_offset,
            placement_version: "phase0-synthetic-glm-layer".to_owned(),
            layer_id: decode_wave.layer_id.0,
            hidden_dim: GLM52_HIDDEN_SIZE as u32,
            wave: Some(ExpertWaveMetadata::from_wave(&decode_wave)),
            rows: vec![ExpertRow {
                row_id: decode_wave.row_sources[0].token_start.0,
                hidden: hidden.clone(),
                routes: routes.clone(),
            }],
        };
        request_offset += 1;
        let response = dispatch_expert_request(state, target.as_deref(), &request).await?;
        if response.partial_outputs.len() != 1 {
            return Err(runtime_error(format!(
                "expert response had {} rows, expected 1",
                response.partial_outputs.len()
            )));
        }
        partials.push(response.partial_outputs[0].clone());
    }

    let summed = sum_partials(&partials, GLM52_HIDDEN_SIZE)?;
    let checksum = summed.iter().map(|value| *value as f64).sum::<f64>();
    let first = summed.first().copied().unwrap_or_default();
    let last = summed.last().copied().unwrap_or_default();
    let decode_ms = duration_ms(decode_start.elapsed());
    Ok(BackendCompletion {
        content: format!(
            "synthetic glm layer ok hidden={} top_k={} expert_kernel={} prefill_chunks={} prefill_rows={} prefill_partials={} prefill_checksum={:.6} decode_rows={} decode_partials={} checksum={:.6} first={:.6} last={:.6}",
            GLM52_HIDDEN_SIZE,
            GLM52_TOP_K,
            glmrt_transport::SYNTHETIC_EXPERT_KERNEL,
            prefill_chunks.len(),
            prefill_rows,
            prefill_partial_count,
            prefill_checksum,
            decode_wave.num_rows(),
            partials.len(),
            checksum,
            first,
            last
        ),
        reasoning_content: None,
        completion_tokens: None,
        stream_chunks: None,
        metrics: BackendMetrics {
            cache_load_ms: 0.0,
            prefill_ms,
            time_to_first_token_ms: None,
            decode_ms,
            reasoning_tokens: 0,
            cached_prompt_tokens: 0,
            prefill_tokens: prefill_rows,
            prefill_chunk_count: prefill_chunks.len(),
            layerwave_prefill_rows: prefill_rows,
            layerwave_decode_rows: decode_wave.num_rows(),
            real_full: None,
        },
    })
}

fn synthetic_hidden(prompt: &str) -> Vec<f32> {
    let mut seed = 2_166_136_261_u32;
    for byte in prompt.as_bytes() {
        seed ^= *byte as u32;
        seed = seed.wrapping_mul(16_777_619);
    }
    (0..GLM52_HIDDEN_SIZE)
        .map(|idx| {
            seed = seed
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223)
                .wrapping_add(idx as u32);
            ((seed % 2001) as f32 / 1000.0) - 1.0
        })
        .collect()
}

fn partition_routes(targets: &[String]) -> Vec<(Option<String>, Vec<RouteEntry>)> {
    let group_count = targets.len().max(1).min(GLM52_TOP_K);
    let mut groups = (0..group_count)
        .map(|idx| (targets.get(idx).cloned(), Vec::new()))
        .collect::<Vec<_>>();
    for expert_id in 0..GLM52_TOP_K {
        groups[expert_id % group_count].1.push(RouteEntry {
            expert_id: expert_id as u32,
            gate: 1.0 / GLM52_TOP_K as f32,
        });
    }
    groups
}
