use anyhow::{Context, Result};
use glmrt_core::{
    DType, DecodeStep, ExpertBatch, GraphBucket, LayerWave, LayerWaveMode, ModelFacts,
    MtpVerifyBlock, PositionId, PrefillChunk, Priority, GLM52_FIRST_K_DENSE_REPLACE,
    GLM52_NUM_HIDDEN_LAYERS,
};

use super::constants::{
    REAL_FULL_PREFLIGHT_DECODE_POSITION, REAL_FULL_PREFLIGHT_DECODE_ROWS,
    REAL_FULL_PREFLIGHT_KV_RESERVATION_ID, REAL_FULL_PREFLIGHT_MTP_ROWS,
    REAL_FULL_PREFLIGHT_MTP_TOKEN_START, REAL_FULL_PREFLIGHT_PREFILL_ROWS,
    REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START, REAL_FULL_PREFLIGHT_REQUEST_ID,
    REAL_FULL_PREFLIGHT_SEQUENCE_ID,
};
use super::types::{
    RealFullExpertBatchDryRun, RealFullLayerSchedulerDryRun, RealFullSchedulerDryRun,
    RealFullWaveDryRun,
};
pub(super) use execution::real_full_scheduler_execution_dry_run;
#[cfg(test)]
pub(super) use execution::real_full_scheduler_execution_for_shape;
pub(super) use execution::{
    load_real_full_kv_snapshot, real_full_scheduler_execute_decode_layer_block,
    real_full_scheduler_execute_decode_layer_block_device_input,
    real_full_scheduler_execute_prefill_decode_layer_block_device_input,
    real_full_scheduler_execution_for_batched_shapes_with_shared_sparse_tcp_and_state_device_hidden,
    real_full_scheduler_execution_for_shape_with_shared_sparse_tcp_and_state_device_hidden,
    real_full_scheduler_execution_for_shape_with_sparse_tcp,
    real_full_scheduler_execution_for_shape_with_state,
    real_full_scheduler_precapture_layer_block_attention, save_real_full_kv_snapshot,
    RealFullKvSnapshot, RealFullSchedulerBatchedInput, RealFullSchedulerDeviceExecution,
    RealFullSchedulerExecutionShape, RealFullSchedulerExecutionState,
    RealFullSchedulerSparseDispatchTransport, RealFullSchedulerSparseTcpDispatchWorker,
};
use protocol_v2::real_full_protocol_v2_batch_probe;
pub(super) use protocol_v2::RealFullSchedulerSparseTcpDispatchProbe;

mod execution;
mod protocol_v2;

pub(super) fn real_full_scheduler_dry_run(catalog_hash: &str) -> Result<RealFullSchedulerDryRun> {
    let placement_version = format!("catalog-{}", &catalog_hash[..16]);
    let graph_bucket = GraphBucket::new(
        REAL_FULL_PREFLIGHT_PREFILL_ROWS
            + REAL_FULL_PREFLIGHT_MTP_ROWS
            + REAL_FULL_PREFLIGHT_DECODE_ROWS,
    );
    let quantization_recipe = ModelFacts::default().quantization_recipe;
    let mut layer_dry_runs = Vec::with_capacity(GLM52_NUM_HIDDEN_LAYERS);
    let mut decode_prefix_read_layers = 0_usize;
    let mut decode_kv_write_layers = 0_usize;
    let mut prefill_prefix_read_layers = 0_usize;
    let mut prefill_kv_write_layers = 0_usize;
    let mut mtp_prefix_read_layers = 0_usize;
    let mut mtp_tentative_write_records = 0_usize;
    let mut sparse_expert_batches = 0_usize;
    let mut rows_per_sparse_expert_batch = 0_usize;
    let mut routes_per_sparse_expert_batch = 0_usize;
    let mut protocol_v2_batch_probe = None;

    for layer_id in 0..GLM52_NUM_HIDDEN_LAYERS {
        let decode = real_full_decode_wave(layer_id, &placement_version);
        let prefill = real_full_prefill_wave(layer_id, &placement_version);
        let mtp_verify = real_full_mtp_wave(layer_id, &placement_version);

        decode_prefix_read_layers += usize::from(!decode.kv_reads.is_empty());
        decode_kv_write_layers += usize::from(!decode.kv_writes.is_empty());
        prefill_prefix_read_layers += usize::from(!prefill.kv_reads.is_empty());
        prefill_kv_write_layers += usize::from(!prefill.kv_writes.is_empty());
        mtp_prefix_read_layers += usize::from(!mtp_verify.kv_reads.is_empty());
        mtp_tentative_write_records += mtp_verify.tentative_kv_writes.len();

        let expert_batch = if layer_id >= GLM52_FIRST_K_DENSE_REPLACE {
            let mut batch = ExpertBatch::glm52_bf16_from_wave_with_envelope(&prefill, graph_bucket)
                .with_context(|| format!("building prefill ExpertBatch for layer {layer_id}"))?;
            batch
                .try_append_wave(&mtp_verify, DType::Bf16, quantization_recipe.clone())
                .with_context(|| format!("appending MTP ExpertBatch rows for layer {layer_id}"))?;
            batch
                .try_append_wave(&decode, DType::Bf16, quantization_recipe.clone())
                .with_context(|| {
                    format!("appending decode ExpertBatch rows for layer {layer_id}")
                })?;
            sparse_expert_batches += 1;
            rows_per_sparse_expert_batch = batch.num_rows();
            routes_per_sparse_expert_batch = batch.route_count();
            if protocol_v2_batch_probe.is_none() {
                protocol_v2_batch_probe =
                    Some(real_full_protocol_v2_batch_probe(layer_id, &batch)?);
            }
            Some(RealFullExpertBatchDryRun {
                rows: batch.num_rows(),
                routes: batch.route_count(),
                graph_bucket_rows: batch.graph_bucket.row_capacity,
                source_modes: vec!["prefill_chunk", "mtp_verify", "decode_step"],
                hidden_dim: batch.hidden_dim,
                hidden_bytes_per_row: batch.hidden_bytes_per_row,
            })
        } else {
            None
        };

        layer_dry_runs.push(RealFullLayerSchedulerDryRun {
            layer_id,
            layer_kind: real_full_layer_kind(layer_id),
            decode: real_full_wave_dry_run(&decode),
            prefill: real_full_wave_dry_run(&prefill),
            mtp_verify: real_full_wave_dry_run(&mtp_verify),
            expert_batch,
        });
    }

    Ok(RealFullSchedulerDryRun {
        status: "dry-run-only",
        scope: "construct LayerWave and mixed ExpertBatch records for real-glm-full",
        placement_version,
        request_id: REAL_FULL_PREFLIGHT_REQUEST_ID,
        sequence_id: REAL_FULL_PREFLIGHT_SEQUENCE_ID,
        kv_reservation_id: REAL_FULL_PREFLIGHT_KV_RESERVATION_ID,
        total_layerwaves: GLM52_NUM_HIDDEN_LAYERS * 3,
        decode_layerwaves: GLM52_NUM_HIDDEN_LAYERS,
        prefill_layerwaves: GLM52_NUM_HIDDEN_LAYERS,
        mtp_verify_layerwaves: GLM52_NUM_HIDDEN_LAYERS,
        dense_coordinator_layers: GLM52_FIRST_K_DENSE_REPLACE,
        sparse_expert_batches,
        graph_bucket_rows: graph_bucket.row_capacity,
        rows_per_sparse_expert_batch,
        routes_per_sparse_expert_batch,
        decode_prefix_read_layers,
        decode_kv_write_layers,
        prefill_prefix_read_layers,
        prefill_kv_write_layers,
        mtp_prefix_read_layers,
        mtp_tentative_write_records,
        protocol_v2_batch_probe: protocol_v2_batch_probe
            .expect("real-full scheduler dry-run has at least one sparse layer"),
        layer_dry_runs,
    })
}

fn real_full_decode_wave(layer_id: usize, placement_version: &str) -> LayerWave {
    LayerWave::decode(DecodeStep::new(
        REAL_FULL_PREFLIGHT_REQUEST_ID,
        REAL_FULL_PREFLIGHT_SEQUENCE_ID,
        layer_id as u32,
        PositionId(REAL_FULL_PREFLIGHT_DECODE_POSITION),
        Some(REAL_FULL_PREFLIGHT_KV_RESERVATION_ID),
        Priority(0),
        placement_version.to_owned(),
    ))
}

fn real_full_prefill_wave(layer_id: usize, placement_version: &str) -> LayerWave {
    LayerWave::prefill(PrefillChunk::new(
        REAL_FULL_PREFLIGHT_REQUEST_ID,
        REAL_FULL_PREFLIGHT_SEQUENCE_ID,
        layer_id as u32,
        PositionId(REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START),
        REAL_FULL_PREFLIGHT_PREFILL_ROWS,
        REAL_FULL_PREFLIGHT_KV_RESERVATION_ID,
        Priority(1),
        GraphBucket::new(REAL_FULL_PREFLIGHT_PREFILL_ROWS),
        placement_version.to_owned(),
    ))
}

fn real_full_mtp_wave(layer_id: usize, placement_version: &str) -> LayerWave {
    LayerWave::mtp_verify(MtpVerifyBlock::new(
        REAL_FULL_PREFLIGHT_REQUEST_ID,
        REAL_FULL_PREFLIGHT_SEQUENCE_ID,
        layer_id as u32,
        PositionId(REAL_FULL_PREFLIGHT_MTP_TOKEN_START),
        REAL_FULL_PREFLIGHT_MTP_ROWS,
        Some(REAL_FULL_PREFLIGHT_KV_RESERVATION_ID),
        Priority(0),
        GraphBucket::new(REAL_FULL_PREFLIGHT_MTP_ROWS),
        placement_version.to_owned(),
    ))
}

fn real_full_wave_dry_run(wave: &LayerWave) -> RealFullWaveDryRun {
    RealFullWaveDryRun {
        mode: real_full_wave_mode_label(wave.mode),
        rows: wave.num_rows(),
        graph_bucket_rows: wave.graph_bucket.row_capacity,
        payload_bytes: wave.payload_bytes_per_direction(),
        kv_reads: wave.kv_reads.len(),
        kv_writes: wave.kv_writes.len(),
        tentative_kv_writes: wave.tentative_kv_writes.len(),
    }
}

fn real_full_wave_mode_label(mode: LayerWaveMode) -> &'static str {
    match mode {
        LayerWaveMode::Decode => "decode",
        LayerWaveMode::Prefill => "prefill",
        LayerWaveMode::MtpVerify => "mtp_verify",
        LayerWaveMode::Benchmark => "benchmark",
    }
}

fn real_full_layer_kind(layer_id: usize) -> &'static str {
    if layer_id < GLM52_FIRST_K_DENSE_REPLACE {
        "dense-mlp"
    } else {
        "sparse-routed-moe"
    }
}
