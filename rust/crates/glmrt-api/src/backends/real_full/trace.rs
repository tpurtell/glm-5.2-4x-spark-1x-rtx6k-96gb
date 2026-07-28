use glmrt_core::{
    plan_prefill_chunks, DecodeStep, ExpertBatch, GraphBucket, KvCacheBackingStore, KvCacheConfig,
    LayerWave, ModelFacts, MtpVerifyBlock, PrefillChunkPolicy, Priority, RowSourceKind,
    GLM52_NUM_HIDDEN_LAYERS,
};

use crate::{runtime_error, ApiError};

use admission::{admit_real_full_request_waves, prefill_chunk_count, RealFullRequestTraceCounters};
use progression::RealFullRequestNumericProgression;

mod admission;
mod progression;

const REAL_FULL_API_MAX_MTP_VERIFY_ROWS: usize = 4;
pub(super) const REAL_FULL_API_MTP_ACCEPTED_ROWS: usize = 2;

#[derive(Debug, Default)]
pub(super) struct RealFullRequestTrace {
    pub(super) prefill_tokens: usize,
    pub(super) prefill_chunks: usize,
    pub(super) decode_budget: usize,
    pub(super) mtp_verify_rows: usize,
    pub(super) mtp_accepted_rows: usize,
    pub(super) candidate_layerwaves: usize,
    pub(super) layerwaves: usize,
    pub(super) deferred_layerwaves: usize,
    pub(super) admitted_iterations: usize,
    pub(super) sparse_batches: usize,
    pub(super) expert_batch_rows: usize,
    pub(super) expert_batch_routes: usize,
    pub(super) expert_prefill_rows: usize,
    pub(super) expert_decode_rows: usize,
    pub(super) expert_mtp_verify_rows: usize,
    pub(super) expert_prefill_routes: usize,
    pub(super) expert_decode_routes: usize,
    pub(super) expert_mtp_verify_routes: usize,
    pub(super) expert_source_modes_covered: bool,
    pub(super) expert_route_entries_match_source_rows: bool,
    pub(super) kv_read_blocks: usize,
    pub(super) committed_kv_writes: usize,
    pub(super) tentative_kv_writes: usize,
    pub(super) committed_mtp_writes: usize,
    pub(super) discarded_mtp_writes: usize,
    pub(super) backed_kv_writes: usize,
    pub(super) backed_kv_bytes_after_discard: usize,
    pub(super) kv_reservation_bytes: usize,
    pub(super) byte_backed_scheduler_trace: bool,
    pub(super) request_numeric_progression_passed: bool,
    pub(super) request_numeric_progression_source_rows: usize,
    pub(super) request_numeric_progression_hidden_dim: usize,
    pub(super) request_numeric_progression_selected_prefill_rows: usize,
    pub(super) request_numeric_progression_selected_decode_rows: usize,
    pub(super) request_numeric_progression_selected_mtp_rows: usize,
    pub(super) request_numeric_progression_attention_value_updates: usize,
    pub(super) request_numeric_progression_mlp_value_updates: usize,
    pub(super) request_numeric_progression_visible_checksum: f32,
    pub(super) request_numeric_progression_rejected_mtp_checksum: f32,
}

pub(super) fn real_full_request_trace(
    prompt_tokens: usize,
    max_tokens: usize,
) -> Result<RealFullRequestTrace, ApiError> {
    let prefill_tokens = prompt_tokens.max(1);
    let decode_budget = max_tokens.max(1);
    let mtp_verify_rows = decode_budget.min(REAL_FULL_API_MAX_MTP_VERIFY_ROWS).max(1);
    let mtp_accepted_rows = mtp_verify_rows.min(REAL_FULL_API_MTP_ACCEPTED_ROWS);
    let sequence_tokens = prefill_tokens + decode_budget + mtp_verify_rows;
    let mut kv_store = KvCacheBackingStore::new(KvCacheConfig::glm52_phase0(sequence_tokens));
    let kv_reservation_id = kv_store
        .reserve("real-glm-full-api", sequence_tokens)
        .map_err(runtime_error)?;
    let policy = PrefillChunkPolicy {
        chunk_tokens: 512,
        max_prefill_tokens_per_iteration: 512,
        max_active_prefill_chunks: 1,
        decode_priority: true,
    };
    let prefill_chunks = prefill_chunk_count(prefill_tokens, &policy);
    let sparse_batch_graph_bucket = GraphBucket::new(sequence_tokens);
    let quantization_recipe = ModelFacts::default().quantization_recipe;
    let mut counters = RealFullRequestTraceCounters::default();
    let mut numeric_progression = RealFullRequestNumericProgression::new(
        prefill_tokens,
        decode_budget,
        mtp_verify_rows,
        mtp_accepted_rows,
    );

    for layer_id in 0..GLM52_NUM_HIDDEN_LAYERS {
        let mut sparse_batch = None;
        for chunk in plan_prefill_chunks(
            "real-full-api-prefill",
            "real-glm-full-api",
            layer_id as u32,
            prefill_tokens,
            kv_reservation_id,
            Priority(10),
            &policy,
            "phase0-real-full-api",
        ) {
            let selected = admit_real_full_request_waves(
                vec![LayerWave::prefill(chunk)],
                &policy,
                sparse_batch_graph_bucket,
                &quantization_recipe,
                &mut sparse_batch,
                &mut counters,
                &mut kv_store,
            )?;
            numeric_progression.apply_selected(&selected)?;
        }

        let decode_waves = (0..decode_budget)
            .map(|decode_offset| {
                LayerWave::decode(DecodeStep::new(
                    "real-full-api-decode",
                    "real-glm-full-api",
                    layer_id as u32,
                    (prefill_tokens + decode_offset) as u64,
                    Some(kv_reservation_id),
                    Priority(0),
                    "phase0-real-full-api",
                ))
            })
            .collect::<Vec<_>>();
        let selected = admit_real_full_request_waves(
            decode_waves,
            &policy,
            sparse_batch_graph_bucket,
            &quantization_recipe,
            &mut sparse_batch,
            &mut counters,
            &mut kv_store,
        )?;
        numeric_progression.apply_selected(&selected)?;

        let mtp_verify = LayerWave::mtp_verify(MtpVerifyBlock::new(
            "real-full-api-mtp-verify",
            "real-glm-full-api",
            layer_id as u32,
            (prefill_tokens + decode_budget) as u64,
            mtp_verify_rows,
            Some(kv_reservation_id),
            Priority(0),
            GraphBucket::new(mtp_verify_rows),
            "phase0-real-full-api",
        ));
        let selected = admit_real_full_request_waves(
            vec![mtp_verify],
            &policy,
            sparse_batch_graph_bucket,
            &quantization_recipe,
            &mut sparse_batch,
            &mut counters,
            &mut kv_store,
        )?;
        numeric_progression.apply_selected(&selected)?;

        if let Some(batch) = sparse_batch {
            counters.sparse_batches += 1;
            counters.expert_batch_rows += batch.num_rows();
            counters.expert_batch_routes += batch.route_count();
            accumulate_expert_source_rows(&mut counters, &batch);
        }
    }

    let snapshot = kv_store.snapshot();
    let backed_kv_writes = kv_store.backed_write_count();
    let backed_kv_bytes_after_discard = kv_store.backed_write_bytes();
    let byte_backed_scheduler_trace = backed_kv_writes
        == counters.committed_kv_writes + snapshot.committed_writes
        && backed_kv_bytes_after_discard <= snapshot.resident_bytes
        && snapshot.active_reservations == 1;
    let expert_source_rows = counters.expert_prefill_rows
        + counters.expert_decode_rows
        + counters.expert_mtp_verify_rows;
    let expert_source_routes = counters.expert_prefill_routes
        + counters.expert_decode_routes
        + counters.expert_mtp_verify_routes;
    let expert_source_modes_covered = counters.expert_prefill_rows > 0
        && counters.expert_decode_rows > 0
        && counters.expert_mtp_verify_rows > 0;
    let expert_route_entries_match_source_rows = expert_source_rows == counters.expert_batch_rows
        && expert_source_routes == counters.expert_batch_routes;
    let numeric_progression = numeric_progression.finish();
    Ok(RealFullRequestTrace {
        prefill_tokens,
        prefill_chunks,
        decode_budget,
        mtp_verify_rows,
        mtp_accepted_rows,
        candidate_layerwaves: counters.candidate_layerwaves,
        layerwaves: counters.selected_layerwaves,
        deferred_layerwaves: counters.deferred_layerwaves,
        admitted_iterations: counters.admitted_iterations,
        sparse_batches: counters.sparse_batches,
        expert_batch_rows: counters.expert_batch_rows,
        expert_batch_routes: counters.expert_batch_routes,
        expert_prefill_rows: counters.expert_prefill_rows,
        expert_decode_rows: counters.expert_decode_rows,
        expert_mtp_verify_rows: counters.expert_mtp_verify_rows,
        expert_prefill_routes: counters.expert_prefill_routes,
        expert_decode_routes: counters.expert_decode_routes,
        expert_mtp_verify_routes: counters.expert_mtp_verify_routes,
        expert_source_modes_covered,
        expert_route_entries_match_source_rows,
        kv_read_blocks: counters.kv_read_blocks,
        committed_kv_writes: counters.committed_kv_writes,
        tentative_kv_writes: counters.tentative_kv_writes,
        committed_mtp_writes: snapshot.committed_writes,
        discarded_mtp_writes: snapshot.discarded_writes,
        backed_kv_writes,
        backed_kv_bytes_after_discard,
        kv_reservation_bytes: snapshot.resident_bytes,
        byte_backed_scheduler_trace,
        request_numeric_progression_passed: numeric_progression.passed,
        request_numeric_progression_source_rows: numeric_progression.source_rows,
        request_numeric_progression_hidden_dim: numeric_progression.hidden_dim,
        request_numeric_progression_selected_prefill_rows: numeric_progression
            .selected_prefill_rows,
        request_numeric_progression_selected_decode_rows: numeric_progression.selected_decode_rows,
        request_numeric_progression_selected_mtp_rows: numeric_progression.selected_mtp_rows,
        request_numeric_progression_attention_value_updates: numeric_progression
            .attention_value_updates,
        request_numeric_progression_mlp_value_updates: numeric_progression.mlp_value_updates,
        request_numeric_progression_visible_checksum: numeric_progression.visible_checksum,
        request_numeric_progression_rejected_mtp_checksum: numeric_progression
            .rejected_mtp_checksum,
    })
}

fn accumulate_expert_source_rows(counters: &mut RealFullRequestTraceCounters, batch: &ExpertBatch) {
    for row in &batch.rows {
        match row.source_kind {
            RowSourceKind::PrefillChunk => {
                counters.expert_prefill_rows += 1;
                counters.expert_prefill_routes += row.route_count;
            }
            RowSourceKind::DecodeStep => {
                counters.expert_decode_rows += 1;
                counters.expert_decode_routes += row.route_count;
            }
            RowSourceKind::MtpVerifyBlock => {
                counters.expert_mtp_verify_rows += 1;
                counters.expert_mtp_verify_routes += row.route_count;
            }
            RowSourceKind::Benchmark => {}
        }
    }
}
