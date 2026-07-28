use glmrt_core::{
    admit_layerwaves_for_iteration, plan_prefill_chunks, DType, ExpertBatch, GraphBucket,
    KvBlockDescriptor, KvCacheBackingStore, KvCacheConfig, LayerWave, PrefillChunkPolicy, Priority,
    GLM52_FIRST_K_DENSE_REPLACE,
};

use crate::{runtime_error, ApiError};

use super::REAL_FULL_API_MTP_ACCEPTED_ROWS;

#[derive(Debug, Default)]
pub(super) struct RealFullRequestTraceCounters {
    pub(super) admitted_iterations: usize,
    pub(super) candidate_layerwaves: usize,
    pub(super) selected_layerwaves: usize,
    pub(super) deferred_layerwaves: usize,
    pub(super) sparse_batches: usize,
    pub(super) expert_batch_rows: usize,
    pub(super) expert_batch_routes: usize,
    pub(super) expert_prefill_rows: usize,
    pub(super) expert_decode_rows: usize,
    pub(super) expert_mtp_verify_rows: usize,
    pub(super) expert_prefill_routes: usize,
    pub(super) expert_decode_routes: usize,
    pub(super) expert_mtp_verify_routes: usize,
    pub(super) kv_read_blocks: usize,
    pub(super) committed_kv_writes: usize,
    pub(super) tentative_kv_writes: usize,
}

pub(super) fn prefill_chunk_count(prompt_tokens: usize, policy: &PrefillChunkPolicy) -> usize {
    plan_prefill_chunks(
        "real-full-api-prefill",
        "real-glm-full-api",
        0,
        prompt_tokens,
        1,
        Priority(10),
        policy,
        "phase0-real-full-api",
    )
    .len()
}

pub(super) fn admit_real_full_request_waves(
    candidates: Vec<LayerWave>,
    policy: &PrefillChunkPolicy,
    sparse_batch_graph_bucket: GraphBucket,
    quantization_recipe: &str,
    sparse_batch: &mut Option<ExpertBatch>,
    counters: &mut RealFullRequestTraceCounters,
    kv_store: &mut KvCacheBackingStore,
) -> Result<Vec<LayerWave>, ApiError> {
    counters.admitted_iterations += 1;
    counters.candidate_layerwaves += candidates.len();
    let admission = admit_layerwaves_for_iteration(candidates, policy);
    counters.selected_layerwaves += admission.selected.len();
    counters.deferred_layerwaves += admission.deferred.len();

    for wave in &admission.selected {
        counters.kv_read_blocks += kv_store.read_visible_blocks_for_wave(wave).len();
        if !wave.kv_writes.is_empty() {
            let payloads =
                real_full_api_kv_payloads_for_descriptors(kv_store.config(), &wave.kv_writes, 0x51);
            counters.committed_kv_writes += kv_store
                .write_committed_blocks_for_wave(wave, payloads)
                .map_err(runtime_error)?
                .len();
        }
        if !wave.tentative_kv_writes.is_empty() {
            let payloads = real_full_api_kv_payloads_for_descriptors(
                kv_store.config(),
                &wave.tentative_kv_writes,
                0xa0,
            );
            counters.tentative_kv_writes += kv_store
                .write_tentative_blocks_for_wave(wave, payloads)
                .map_err(runtime_error)?
                .len();
            let first_tentative = wave
                .tentative_kv_writes
                .first()
                .expect("non-empty tentative KV writes have a first descriptor");
            kv_store
                .resolve_mtp_tentative_writes(
                    first_tentative.reservation_id,
                    wave.layer_id,
                    first_tentative.token_start,
                    wave.tentative_kv_writes.len(),
                    wave.tentative_kv_writes
                        .len()
                        .min(REAL_FULL_API_MTP_ACCEPTED_ROWS),
                )
                .map_err(runtime_error)?;
        }

        if (wave.layer_id.0 as usize) >= GLM52_FIRST_K_DENSE_REPLACE {
            match sparse_batch {
                Some(batch) => batch
                    .try_append_wave(wave, DType::Bf16, quantization_recipe.to_owned())
                    .map_err(runtime_error)?,
                None => {
                    *sparse_batch = Some(
                        ExpertBatch::glm52_bf16_from_wave_with_envelope(
                            wave,
                            sparse_batch_graph_bucket,
                        )
                        .map_err(runtime_error)?,
                    );
                }
            }
        }
    }

    Ok(admission.selected)
}

fn real_full_api_kv_payloads_for_descriptors(
    config: &KvCacheConfig,
    descriptors: &[KvBlockDescriptor],
    salt: u8,
) -> Vec<Vec<u8>> {
    descriptors
        .iter()
        .map(|descriptor| {
            let byte = salt ^ descriptor.layer_id.0 as u8 ^ descriptor.token_start.0 as u8;
            vec![byte; config.layer_payload_bytes(descriptor.layer_id, descriptor.token_count)]
        })
        .collect()
}
