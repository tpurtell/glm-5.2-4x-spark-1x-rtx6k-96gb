use anyhow::{Context, Result};
use glmrt_core::{
    admit_layerwaves_for_iteration, DType, ExpertBatch, GraphBucket, KvBackedBlock,
    KvBlockDescriptor, KvCacheBackingStore, KvCacheConfig, KvCacheDType, LayerWave, LayerWaveMode,
    PrefillChunkPolicy, RowSourceKind, TensorCatalog, GLM52_DSA_INDEXER_LAYER_IDS_WITH_MTP,
    GLM52_DSA_INDEX_HEAD_DIM, GLM52_FIRST_K_DENSE_REPLACE, GLM52_HIDDEN_SIZE,
    GLM52_MLA_KV_LORA_RANK, GLM52_MLA_QK_ROPE_HEAD_DIM, GLM52_MLA_ROPE_THETA, GLM52_TOP_K,
};
use glmrt_loader::read_tensor_bytes_into;
use std::collections::BTreeMap;
use std::env;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use crate::commands::real_full::coordinator_kernels::{
    concat_device_bf16_row_batches, coordinator_cuda_reference_kernels_enabled,
    coordinator_w4a16_q_b_decode_enabled, coordinator_w8a16_o_proj_decode_enabled,
    coordinator_w8a16_q_a_decode_enabled, coordinator_w8a16_q_b_decode_enabled,
    copy_mla_decode_query_row_to_attention_stream, device_buffer_byte_view,
    layer_norm_affine_bf16_preloaded_resident_weight_bias_device_input_output,
    linear_rows_bf16_m1_parity_batched_preloaded_resident_weight_device_output,
    linear_rows_bf16_preloaded_resident_weight_device_output,
    linear_rows_w8a16_preloaded_resident_weight_device_output,
    mla_decode_query_dsa_projection_bf16_device_outputs,
    mla_decode_query_projection_bf16_device_output,
    mla_decode_scalar_q_a_batched_q_b_projection_bf16_device_outputs,
    preload_resident_weight_from_host_staging, preloaded_resident_weight_device_buffer,
    resident_weight_is_preloaded,
    rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output, DeviceBf16Output,
    MlaDecodeKvDsaProjectionWeights,
};
use crate::commands::real_full::dense::math::bf16_bytes_to_f32;
use crate::commands::real_full::dense::REAL_FULL_DENSE_RMSNORM_EPS;
use crate::commands::real_full::kv::device::RealFullDeviceKvExecutionMirror;
use crate::commands::real_full::scheduler::protocol_v2::real_full_scheduler_host_batch_partition_probe;
use glmrt_ffi::GlmrtDeviceBuffer;

use super::{
    progression::{RealFullSchedulerDeviceHiddenSource, RealFullSchedulerNumericProgression},
    RealFullAdmittedSchedulerIteration, RealFullSchedulerDeviceAttentionDelta,
    RealFullSchedulerExecutionCounters,
};

const REAL_FULL_SCHEDULER_MLA_NUM_ATTENTION_HEADS: usize = 64;
const REAL_FULL_SCHEDULER_MLA_Q_LORA_RANK: usize = 2048;
const REAL_FULL_SCHEDULER_MLA_QK_NOPE_HEAD_DIM: usize = 192;
const REAL_FULL_SCHEDULER_MLA_V_HEAD_DIM: usize = 256;
const REAL_FULL_DSA_INDEX_HEADS: usize = 32;
const REAL_FULL_DSA_TOP_K: usize = 2_048;
const ADMISSION_STAGE_TIMING_ENV: &str = "GLMRT_REAL_FULL_ADMISSION_STAGE_TIMING";
const MTP_TARGET_ATTENTION_FUSION_ENV: &str = "GLMRT_REAL_FULL_MTP_TARGET_ATTENTION_FUSION";
// One authoritative decode row plus up to fifteen Siro proposal rows.
pub(super) const MTP_TARGET_ATTENTION_FUSION_MAX_ROWS: usize = 16;
const MTP_TARGET_SCALAR_Q_A_BATCHED_Q_B_MAX_ROWS: usize = 16;

pub(super) fn real_full_apply_admitted_scheduler_iteration(
    store: &mut KvCacheBackingStore,
    policy: &PrefillChunkPolicy,
    candidates: Vec<LayerWave>,
    expected_modes: &[LayerWaveMode],
    sparse_batch_graph_bucket: GraphBucket,
    quantization_recipe: &str,
    mtp_accepted_rows: Option<usize>,
    record_sparse_host_partition: bool,
    counters: &mut RealFullSchedulerExecutionCounters,
    device_kv: &mut RealFullDeviceKvExecutionMirror,
    numeric_progression: &mut RealFullSchedulerNumericProgression,
    catalog: &TensorCatalog,
) -> Result<RealFullAdmittedSchedulerIteration> {
    let stage_timing = admission_stage_timing_enabled();
    let total_start = stage_timing.then(Instant::now);
    counters.iterations += 1;
    counters.candidate_layerwaves += candidates.len();
    let admission_start = stage_timing.then(Instant::now);
    let admission = admit_layerwaves_for_iteration(candidates, policy);
    let admission_ms = elapsed_ms_optional(admission_start);
    counters.layer_order_verified &= admission
        .selected
        .iter()
        .map(|wave| wave.mode)
        .eq(expected_modes.iter().copied());
    counters.selected_layerwaves += admission.selected.len();
    counters.deferred_layerwaves += admission.deferred.len();
    counters.selected_decode_rows += admission.selected_decode_rows;
    counters.selected_prefill_rows += admission.selected_prefill_rows;
    counters.selected_mtp_rows += admission.selected_mtp_rows;

    let mut sparse_probe_ms = 0.0_f64;
    if admission
        .selected
        .first()
        .is_some_and(|wave| (wave.layer_id.0 as usize) >= GLM52_FIRST_K_DENSE_REPLACE)
    {
        if record_sparse_host_partition {
            let mut selected = admission.selected.iter();
            let first_wave = selected
                .next()
                .expect("sparse admission selected wave is present");
            let mut batch = ExpertBatch::glm52_bf16_from_wave_with_envelope(
                first_wave,
                sparse_batch_graph_bucket,
            )
            .with_context(|| {
                format!(
                    "building admitted sparse ExpertBatch for layer {}",
                    first_wave.layer_id.0
                )
            })?;
            for wave in selected {
                batch
                    .try_append_wave(wave, DType::Bf16, quantization_recipe.to_owned())
                    .with_context(|| {
                        format!(
                            "appending admitted sparse ExpertBatch rows for layer {}",
                            wave.layer_id.0
                        )
                    })?;
            }
            counters.sparse_expert_batches += 1;
            counters.sparse_expert_batch_rows += batch.num_rows();
            counters.sparse_expert_batch_routes += batch.route_count();
            accumulate_sparse_expert_row_sources(&batch, counters);
            let sparse_probe_start = stage_timing.then(Instant::now);
            let host_partition = real_full_scheduler_host_batch_partition_probe(&batch)
                .with_context(|| {
                    format!(
                        "partitioning admitted sparse ExpertBatch for layer {} into host batches",
                        first_wave.layer_id.0
                    )
                })?;
            sparse_probe_ms += elapsed_ms_optional(sparse_probe_start);
            counters.sparse_expert_host_batch_sets += host_partition.host_batch_sets;
            counters.sparse_expert_host_batches += host_partition.host_batches;
            counters.sparse_expert_host_batch_rows += host_partition.rows;
            counters.sparse_expert_host_batch_routes += host_partition.routes;
            counters.sparse_expert_host_batch_expert_tiles += host_partition.expert_tiles;
            counters.sparse_expert_host_batch_routes_match_global &=
                host_partition.routes_match_global;
            counters.sparse_expert_host_batch_graph_counts_valid &=
                host_partition.graph_counts_valid;
            counters.sparse_expert_host_request_frames += host_partition.request_frames;
            counters.sparse_expert_host_request_rows += host_partition.request_rows;
            counters.sparse_expert_host_request_routes += host_partition.request_routes;
            counters.sparse_expert_host_request_payload_bytes +=
                host_partition.request_payload_bytes;
            counters.sparse_expert_host_request_wire_bytes += host_partition.request_wire_bytes;
            counters.sparse_expert_host_response_frames += host_partition.response_frames;
            counters.sparse_expert_host_response_rows += host_partition.response_rows;
            counters.sparse_expert_host_response_payload_bytes +=
                host_partition.response_payload_bytes;
            counters.sparse_expert_host_response_wire_bytes += host_partition.response_wire_bytes;
            counters.sparse_expert_host_wire_envelopes_valid &= host_partition.wire_envelopes_valid;
        } else {
            counters.sparse_expert_batches += 1;
            accumulate_sparse_expert_row_sources_from_waves(&admission.selected, counters)?;
        }
    }

    let mut device_attention_deltas = Vec::new();
    let mut read_visible_ms = 0.0_f64;
    let mut kv_project_ms = 0.0_f64;
    let mut metadata_write_ms = 0.0_f64;
    let mut attention_ms = 0.0_f64;
    let mut tentative_resolve_ms = 0.0_f64;
    let fuse_mtp_target_attention =
        should_fuse_mtp_target_attention(store.config(), &admission.selected)?;
    let mut prepared_attention = Vec::with_capacity(usize::from(fuse_mtp_target_attention) * 2);
    for (wave_index, wave) in admission.selected.iter().enumerate() {
        let read_visible_start = stage_timing.then(Instant::now);
        let visible_blocks = store.read_attention_blocks_for_wave(wave);
        read_visible_ms += elapsed_ms_optional(read_visible_start);
        counters.kv_read_blocks += visible_blocks.len();
        if scheduler_device_kv_readback_validation_enabled() {
            device_kv
                .read_visible_blocks(&visible_blocks)
                .with_context(|| {
                    format!(
                        "validating admitted device KV blocks for layer {}",
                        wave.layer_id.0
                    )
                })?;
        }
        if !wave.kv_writes.is_empty() {
            let kv_project_start = stage_timing.then(Instant::now);
            let payloads = real_full_current_scheduler_kv_payloads_for_descriptors(
                catalog,
                store.config(),
                wave,
                &wave.kv_writes,
                0x40,
                device_kv,
                numeric_progression,
            )
            .with_context(|| {
                format!(
                    "writing admitted committed device KV blocks for layer {}",
                    wave.layer_id.0
                )
            })?;
            kv_project_ms += elapsed_ms_optional(kv_project_start);
            counters.projected_device_kv_writes += payloads.projected_device_writes;
            counters.projected_device_kv_write_bytes += payloads.projected_device_write_bytes;
            let metadata_write_start = stage_timing.then(Instant::now);
            let write_ids = if let Some(host_payloads) = payloads.host_payloads {
                counters.synthetic_kv_payload_writes += host_payloads.len();
                store.write_committed_blocks_for_wave(wave, host_payloads)
            } else {
                store.write_committed_block_metadata_for_wave(wave)
            }
            .with_context(|| {
                format!(
                    "writing admitted committed KV metadata for layer {}",
                    wave.layer_id.0
                )
            })?;
            metadata_write_ms += elapsed_ms_optional(metadata_write_start);
            counters.committed_kv_writes += write_ids.len();
            if fuse_mtp_target_attention {
                prepared_attention.push(PreparedSchedulerAttention {
                    wave_index,
                    visible_blocks: visible_blocks.clone(),
                });
            } else {
                let attention_start = stage_timing.then(Instant::now);
                if let Some(delta) = real_full_launch_scheduler_attention_from_device_kv(
                    store.config(),
                    wave,
                    &visible_blocks,
                    &wave.kv_writes,
                    counters,
                    device_kv,
                    numeric_progression,
                    catalog,
                    payloads.normalized_hidden.as_ref(),
                )
                .with_context(|| {
                    format!(
                        "launching admitted scheduler device attention for committed layer {}",
                        wave.layer_id.0
                    )
                })? {
                    device_attention_deltas.push(delta);
                }
                attention_ms += elapsed_ms_optional(attention_start);
            }
        }
        if !wave.tentative_kv_writes.is_empty() {
            let kv_project_start = stage_timing.then(Instant::now);
            let payloads = real_full_current_scheduler_kv_payloads_for_descriptors(
                catalog,
                store.config(),
                wave,
                &wave.tentative_kv_writes,
                0x80,
                device_kv,
                numeric_progression,
            )
            .with_context(|| {
                format!(
                    "writing admitted tentative device KV blocks for layer {}",
                    wave.layer_id.0
                )
            })?;
            kv_project_ms += elapsed_ms_optional(kv_project_start);
            counters.projected_device_kv_writes += payloads.projected_device_writes;
            counters.projected_device_kv_write_bytes += payloads.projected_device_write_bytes;
            let metadata_write_start = stage_timing.then(Instant::now);
            let write_ids = if let Some(host_payloads) = payloads.host_payloads {
                counters.synthetic_kv_payload_writes += host_payloads.len();
                store.write_tentative_blocks_for_wave(wave, host_payloads)
            } else {
                store.write_tentative_block_metadata_for_wave(wave)
            }
            .with_context(|| {
                format!(
                    "writing admitted tentative KV metadata for layer {}",
                    wave.layer_id.0
                )
            })?;
            metadata_write_ms += elapsed_ms_optional(metadata_write_start);
            counters.tentative_kv_writes += write_ids.len();
            if fuse_mtp_target_attention {
                prepared_attention.push(PreparedSchedulerAttention {
                    wave_index,
                    visible_blocks: visible_blocks.clone(),
                });
            } else {
                let attention_start = stage_timing.then(Instant::now);
                if let Some(delta) = real_full_launch_sequential_mtp_target_attention(
                    store.config(),
                    wave,
                    &visible_blocks,
                    &wave.tentative_kv_writes,
                    counters,
                    device_kv,
                    numeric_progression,
                    catalog,
                    payloads.normalized_hidden.as_ref(),
                )
                .with_context(|| {
                    format!(
                        "launching admitted scheduler device attention for tentative layer {}",
                        wave.layer_id.0
                    )
                })? {
                    device_attention_deltas.push(delta);
                }
                attention_ms += elapsed_ms_optional(attention_start);
            }
            if !fuse_mtp_target_attention {
                let tentative_resolve_start = stage_timing.then(Instant::now);
                resolve_scheduler_mtp_tentative_writes(store, wave, mtp_accepted_rows)?;
                tentative_resolve_ms += elapsed_ms_optional(tentative_resolve_start);
            }
        }
    }

    if fuse_mtp_target_attention {
        let attention_start = stage_timing.then(Instant::now);
        device_attention_deltas.extend(
            real_full_launch_fused_mtp_target_attention(
                store.config(),
                &admission.selected,
                prepared_attention,
                counters,
                device_kv,
                numeric_progression,
                catalog,
            )
            .context("launching fused decode/MTP scheduler device attention")?,
        );
        attention_ms += elapsed_ms_optional(attention_start);
        let tentative_resolve_start = stage_timing.then(Instant::now);
        resolve_scheduler_mtp_tentative_writes(store, &admission.selected[1], mtp_accepted_rows)?;
        tentative_resolve_ms += elapsed_ms_optional(tentative_resolve_start);
    }

    if stage_timing {
        let layer_id = admission
            .selected
            .first()
            .map(|wave| wave.layer_id.0)
            .unwrap_or(u32::MAX);
        eprintln!(
            "real_full_admission_stage_timing layer_id={} selected={} prefill_rows={} decode_rows={} mtp_rows={} admission_ms={:.3} sparse_probe_ms={:.3} read_visible_ms={:.3} kv_project_ms={:.3} metadata_write_ms={:.3} attention_ms={:.3} tentative_resolve_ms={:.3} total_ms={:.3}",
            layer_id,
            admission.selected.len(),
            admission.selected_prefill_rows,
            admission.selected_decode_rows,
            admission.selected_mtp_rows,
            admission_ms,
            sparse_probe_ms,
            read_visible_ms,
            kv_project_ms,
            metadata_write_ms,
            attention_ms,
            tentative_resolve_ms,
            elapsed_ms_optional(total_start)
        );
    }

    Ok(RealFullAdmittedSchedulerIteration {
        selected: admission.selected,
        device_attention_deltas,
    })
}

fn accumulate_sparse_expert_row_sources(
    batch: &ExpertBatch,
    counters: &mut RealFullSchedulerExecutionCounters,
) {
    for row in &batch.rows {
        match row.source_kind {
            RowSourceKind::PrefillChunk => {
                counters.sparse_expert_prefill_rows += 1;
                counters.sparse_expert_prefill_routes += row.route_count;
            }
            RowSourceKind::DecodeStep => {
                counters.sparse_expert_decode_rows += 1;
                counters.sparse_expert_decode_routes += row.route_count;
            }
            RowSourceKind::MtpVerifyBlock => {
                counters.sparse_expert_mtp_verify_rows += 1;
                counters.sparse_expert_mtp_verify_routes += row.route_count;
            }
            RowSourceKind::Benchmark => {}
        }
    }
}

fn accumulate_sparse_expert_row_sources_from_waves(
    selected: &[LayerWave],
    counters: &mut RealFullSchedulerExecutionCounters,
) -> Result<()> {
    for wave in selected {
        for source in &wave.row_sources {
            counters.sparse_expert_batch_rows += source.row_count;
            let routes = source
                .row_count
                .checked_mul(GLM52_TOP_K)
                .context("scheduler sparse expert direct route count overflow")?;
            counters.sparse_expert_batch_routes += routes;
            match source.kind {
                RowSourceKind::PrefillChunk => {
                    counters.sparse_expert_prefill_rows += source.row_count;
                    counters.sparse_expert_prefill_routes += routes;
                }
                RowSourceKind::DecodeStep => {
                    counters.sparse_expert_decode_rows += source.row_count;
                    counters.sparse_expert_decode_routes += routes;
                }
                RowSourceKind::MtpVerifyBlock => {
                    counters.sparse_expert_mtp_verify_rows += source.row_count;
                    counters.sparse_expert_mtp_verify_routes += routes;
                }
                RowSourceKind::Benchmark => {}
            }
        }
    }
    Ok(())
}

struct RealFullCurrentSchedulerKvPayloads {
    host_payloads: Option<Vec<Vec<u8>>>,
    projected_device_writes: usize,
    projected_device_write_bytes: usize,
    normalized_hidden: Option<DeviceBf16Output>,
}

struct PreparedSchedulerAttention {
    wave_index: usize,
    visible_blocks: Vec<KvBackedBlock>,
}

fn resolve_scheduler_mtp_tentative_writes(
    store: &mut KvCacheBackingStore,
    wave: &LayerWave,
    mtp_accepted_rows: Option<usize>,
) -> Result<()> {
    let Some(mtp_accepted_rows) = mtp_accepted_rows else {
        return Ok(());
    };
    let first_tentative = wave
        .tentative_kv_writes
        .first()
        .context("MTP tentative resolution requires at least one write")?;
    store
        .resolve_mtp_tentative_writes(
            first_tentative.reservation_id,
            wave.layer_id,
            first_tentative.token_start,
            wave.tentative_kv_writes.len(),
            mtp_accepted_rows.min(wave.tentative_kv_writes.len()),
        )
        .with_context(|| {
            format!(
                "resolving admitted tentative KV blocks for layer {}",
                wave.layer_id.0
            )
        })?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn real_full_current_scheduler_kv_payloads_for_descriptors(
    catalog: &TensorCatalog,
    config: &KvCacheConfig,
    wave: &LayerWave,
    descriptors: &[KvBlockDescriptor],
    synthetic_salt: u8,
    device_kv: &mut RealFullDeviceKvExecutionMirror,
    numeric_progression: &mut RealFullSchedulerNumericProgression,
) -> Result<RealFullCurrentSchedulerKvPayloads> {
    if let Some(projected) = real_full_projected_current_scheduler_kv_payloads(
        catalog,
        config,
        wave,
        descriptors,
        device_kv,
        numeric_progression,
    )? {
        return Ok(projected);
    }

    let payloads = real_full_kv_payloads_for_descriptors(config, descriptors, synthetic_salt);
    device_kv
        .write_host_blocks(descriptors, &payloads)
        .context("writing admitted synthetic host KV payloads to device cache")?;
    Ok(RealFullCurrentSchedulerKvPayloads {
        host_payloads: Some(payloads),
        projected_device_writes: 0,
        projected_device_write_bytes: 0,
        normalized_hidden: None,
    })
}

fn real_full_projected_current_scheduler_kv_payloads(
    catalog: &TensorCatalog,
    config: &KvCacheConfig,
    wave: &LayerWave,
    descriptors: &[KvBlockDescriptor],
    device_kv: &mut RealFullDeviceKvExecutionMirror,
    numeric_progression: &mut RealFullSchedulerNumericProgression,
) -> Result<Option<RealFullCurrentSchedulerKvPayloads>> {
    if !matches!(
        config.dtype,
        KvCacheDType::Bf16 | KvCacheDType::Fp8 | KvCacheDType::Nvfp4
    ) || descriptors.is_empty()
        || !coordinator_cuda_reference_kernels_enabled()
    {
        return Ok(None);
    }
    let source = single_wave_source(wave)?;
    let descriptor_rows = descriptor_row_count(descriptors)
        .context("counting projected scheduler KV descriptor rows")?;
    if descriptor_rows != source.row_count {
        anyhow::bail!(
            "projected scheduler KV write row mismatch for layer {}: descriptors={} source_rows={}",
            wave.layer_id.0,
            descriptor_rows,
            source.row_count
        );
    }

    let Some(hidden) = numeric_progression
        .device_hidden_source(source.kind, source.token_start.0 as usize, source.row_count)
        .with_context(|| {
            format!(
                "resolving projected scheduler KV hidden rows for layer {}",
                wave.layer_id.0
            )
        })?
    else {
        return Ok(None);
    };
    if hidden.rows != source.row_count || hidden.values_per_row != GLM52_HIDDEN_SIZE {
        anyhow::bail!(
            "projected scheduler KV hidden shape mismatch for layer {}: expected {}x{} got {}x{}",
            wave.layer_id.0,
            source.row_count,
            GLM52_HIDDEN_SIZE,
            hidden.rows,
            hidden.values_per_row
        );
    }

    let layer_id = wave.layer_id.0 as usize;
    if descriptor_rows == 1 {
        let Some(layer) = scheduler_current_kv_resident_layer(catalog, layer_id)? else {
            return Ok(None);
        };
        let dsa_weights = layer
            .dsa
            .as_ref()
            .map(|dsa| MlaDecodeKvDsaProjectionWeights {
                wk: dsa.wk_weight.device_buffer(),
                norm_weight: dsa.k_norm_weight.device_buffer(),
                norm_bias: dsa.k_norm_bias.device_buffer(),
            });
        if let Some(commit) = device_kv
            .write_mla_decode_kv_device_block_bf16(
                descriptors,
                hidden.buffer,
                layer.input_norm_weight.device_buffer(),
                layer.kv_a_weight.device_buffer(),
                layer.kv_norm_weight.device_buffer(),
                dsa_weights,
                REAL_FULL_DENSE_RMSNORM_EPS,
                GLM52_MLA_ROPE_THETA,
            )
            .context("running fused scheduler decode KV projection and cache commit")?
        {
            if commit.writes.len() != descriptors.len() {
                anyhow::bail!(
                    "fused scheduler decode KV write count mismatch for layer {}: writes={} descriptors={}",
                    wave.layer_id.0,
                    commit.writes.len(),
                    descriptors.len()
                );
            }
            let projected_device_write_bytes = commit
                .writes
                .iter()
                .map(|write| write.payload_bytes)
                .sum::<usize>();
            return Ok(Some(RealFullCurrentSchedulerKvPayloads {
                host_payloads: None,
                projected_device_writes: commit.writes.len(),
                projected_device_write_bytes,
                normalized_hidden: Some(commit.normalized_hidden),
            }));
        }
    }
    let outputs = match project_current_scheduler_kv_outputs(catalog, layer_id, hidden)? {
        Some(outputs) => outputs,
        None => return Ok(None),
    };
    let device_writes = if config.layer_has_dsa_indexer(wave.layer_id) {
        let dsa_key = outputs.dsa_key.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "projected scheduler KV DSA write missing DSA key for layer {}",
                wave.layer_id.0
            )
        })?;
        device_kv
            .write_projected_mla_kv_a_and_dsa_key_device_blocks_bf16(
                descriptors,
                outputs.kv_a_projected.buffer(),
                dsa_key.buffer(),
                Some(outputs.kv_norm_weight),
            )
            .context("writing projected scheduler MLA+DSA KV payloads to device cache")?
    } else {
        device_kv
            .write_projected_mla_kv_a_device_blocks_bf16(
                descriptors,
                outputs.kv_a_projected.buffer(),
                Some(outputs.kv_norm_weight),
            )
            .context("writing projected scheduler MLA KV payloads to device cache")?
    };
    let Some(device_writes) = device_writes else {
        return Ok(None);
    };
    if device_writes.len() != descriptors.len() {
        anyhow::bail!(
            "projected scheduler KV device write count mismatch for layer {}: writes={} descriptors={}",
            wave.layer_id.0,
            device_writes.len(),
            descriptors.len()
        );
    }

    validate_projected_scheduler_kv_outputs(
        config,
        descriptors,
        &outputs.kv_a_projected,
        outputs.dsa_key.as_ref(),
    )?;
    let projected_device_write_bytes = device_writes
        .iter()
        .map(|write| write.payload_bytes)
        .sum::<usize>();
    Ok(Some(RealFullCurrentSchedulerKvPayloads {
        host_payloads: None,
        projected_device_writes: device_writes.len(),
        projected_device_write_bytes,
        normalized_hidden: Some(outputs.normalized_hidden),
    }))
}

struct SchedulerCurrentKvProjectionOutputs {
    normalized_hidden: DeviceBf16Output,
    kv_a_projected: DeviceBf16Output,
    kv_norm_weight: GlmrtDeviceBuffer,
    dsa_key: Option<DeviceBf16Output>,
}

#[derive(Clone)]
struct SchedulerCurrentKvResidentLayer {
    input_norm_name: String,
    input_norm_weight: CachedResidentDeviceBuffer,
    kv_a_name: String,
    kv_a_weight: CachedResidentDeviceBuffer,
    kv_norm_weight: CachedResidentDeviceBuffer,
    dsa: Option<SchedulerCurrentKvDsaResidentLayer>,
}

#[derive(Clone)]
struct SchedulerCurrentKvDsaResidentLayer {
    wk_name: String,
    wk_weight: CachedResidentDeviceBuffer,
    k_norm_weight_name: String,
    k_norm_weight: CachedResidentDeviceBuffer,
    k_norm_bias_name: String,
    k_norm_bias: CachedResidentDeviceBuffer,
}

struct SchedulerRealAttentionQueryOutputs {
    q_projected: DeviceBf16Output,
    dsa: Option<SchedulerRealAttentionDsaQueryOutputs>,
    kv_norm_weight: GlmrtDeviceBuffer,
    kv_b_weight: GlmrtDeviceBuffer,
    output_projection_weight_name: String,
}

struct SchedulerRealAttentionDsaQueryOutputs {
    query_projected: DeviceBf16Output,
    weights_projected: DeviceBf16Output,
}

#[derive(Clone)]
struct SchedulerRealAttentionResidentLayer {
    input_norm_name: String,
    input_norm_weight: CachedResidentDeviceBuffer,
    q_a_name: String,
    q_a_norm_name: String,
    q_b_name: String,
    q_a_weight: Option<CachedResidentDeviceBuffer>,
    q_a_norm_weight: CachedResidentDeviceBuffer,
    q_b_weight: Option<CachedResidentDeviceBuffer>,
    kv_norm_weight: CachedResidentDeviceBuffer,
    kv_b_weight: CachedResidentDeviceBuffer,
    output_projection_weight_name: String,
    q_b_rows: usize,
    dsa: Option<SchedulerRealAttentionDsaResidentLayer>,
}

#[derive(Clone)]
struct SchedulerRealAttentionDsaResidentLayer {
    wq_b_name: String,
    wq_b_weight: CachedResidentDeviceBuffer,
    weights_proj_name: String,
    weights_proj_weight: CachedResidentDeviceBuffer,
}

#[derive(Clone, Copy)]
struct CachedResidentDeviceBuffer {
    ptr: usize,
    bytes: usize,
    device_id: i32,
    flags: u64,
}

impl CachedResidentDeviceBuffer {
    fn device_buffer(self) -> GlmrtDeviceBuffer {
        GlmrtDeviceBuffer {
            ptr: self.ptr as *mut std::ffi::c_void,
            bytes: self.bytes,
            device_id: self.device_id,
            flags: self.flags,
        }
    }
}

impl From<GlmrtDeviceBuffer> for CachedResidentDeviceBuffer {
    fn from(buffer: GlmrtDeviceBuffer) -> Self {
        Self {
            ptr: buffer.ptr as usize,
            bytes: buffer.bytes,
            device_id: buffer.device_id,
            flags: buffer.flags,
        }
    }
}

fn project_current_scheduler_kv_outputs(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden: super::progression::RealFullSchedulerDeviceHiddenSource,
) -> Result<Option<SchedulerCurrentKvProjectionOutputs>> {
    let Some(layer) = scheduler_current_kv_resident_layer(catalog, layer_id)? else {
        return Ok(None);
    };
    let SchedulerCurrentKvResidentLayer {
        input_norm_name,
        input_norm_weight: _,
        kv_a_name,
        kv_a_weight: _,
        kv_norm_weight,
        dsa,
    } = layer;
    let kv_a_rows = GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM;

    let normalized = rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output(
        &input_norm_name,
        hidden.buffer,
        hidden.rows,
        GLM52_HIDDEN_SIZE,
        REAL_FULL_DENSE_RMSNORM_EPS,
    )
    .context("normalizing scheduler current hidden rows for projected KV write")?;
    let kv_a_projected = if (2..=16).contains(&hidden.rows) {
        linear_rows_bf16_m1_parity_batched_preloaded_resident_weight_device_output(
            &kv_a_name,
            normalized.buffer(),
            hidden.rows,
            GLM52_HIDDEN_SIZE,
            kv_a_rows,
            kv_a_rows,
        )
    } else {
        linear_rows_bf16_preloaded_resident_weight_device_output(
            &kv_a_name,
            normalized.buffer(),
            None,
            hidden.rows,
            GLM52_HIDDEN_SIZE,
            kv_a_rows,
            kv_a_rows,
        )
    }
    .context("projecting scheduler current hidden rows to MLA kv_a")?;

    let dsa_key = if let Some(dsa) = dsa {
        let wk_projected = if (2..=16).contains(&hidden.rows) {
            linear_rows_bf16_m1_parity_batched_preloaded_resident_weight_device_output(
                &dsa.wk_name,
                normalized.buffer(),
                hidden.rows,
                GLM52_HIDDEN_SIZE,
                GLM52_DSA_INDEX_HEAD_DIM,
                GLM52_DSA_INDEX_HEAD_DIM,
            )
        } else {
            linear_rows_bf16_preloaded_resident_weight_device_output(
                &dsa.wk_name,
                normalized.buffer(),
                None,
                hidden.rows,
                GLM52_HIDDEN_SIZE,
                GLM52_DSA_INDEX_HEAD_DIM,
                GLM52_DSA_INDEX_HEAD_DIM,
            )
        }
        .context("projecting scheduler current hidden rows to DSA wk")?;
        Some(
            layer_norm_affine_bf16_preloaded_resident_weight_bias_device_input_output(
                &dsa.k_norm_weight_name,
                &dsa.k_norm_bias_name,
                wk_projected.buffer(),
                hidden.rows,
                GLM52_DSA_INDEX_HEAD_DIM,
                REAL_FULL_DENSE_RMSNORM_EPS,
            )
            .context("normalizing scheduler current DSA key rows")?,
        )
    } else {
        None
    };

    Ok(Some(SchedulerCurrentKvProjectionOutputs {
        normalized_hidden: normalized,
        kv_a_projected,
        kv_norm_weight: kv_norm_weight.device_buffer(),
        dsa_key,
    }))
}

fn scheduler_current_kv_resident_layer(
    catalog: &TensorCatalog,
    layer_id: usize,
) -> Result<Option<SchedulerCurrentKvResidentLayer>> {
    let cache_key = scheduler_current_kv_layer_cache_key(catalog, layer_id);
    if let Some(layer) = scheduler_current_kv_resident_layer_cache()
        .lock()
        .map_err(|_| anyhow::anyhow!("scheduler current KV resident layer cache poisoned"))?
        .get(&cache_key)
        .cloned()
    {
        return Ok(Some(layer));
    }

    let input_norm_name = format!("model.layers.{layer_id}.input_layernorm.weight");
    let kv_a_name = format!("model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.weight");
    let kv_a_norm_name = format!("model.layers.{layer_id}.self_attn.kv_a_layernorm.weight");
    let kv_a_rows = GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM;
    if !ensure_scheduler_bf16_tensor_resident(
        catalog,
        &input_norm_name,
        &[GLM52_HIDDEN_SIZE],
        "scheduler current KV input RMSNorm pinned staging",
    )? || !ensure_scheduler_bf16_tensor_resident(
        catalog,
        &kv_a_name,
        &[kv_a_rows, GLM52_HIDDEN_SIZE],
        "scheduler current KV kv_a projection pinned staging",
    )? || !ensure_scheduler_bf16_tensor_resident(
        catalog,
        &kv_a_norm_name,
        &[GLM52_MLA_KV_LORA_RANK],
        "scheduler current KV kv_a RMSNorm pinned staging",
    )? {
        return Ok(None);
    }
    let kv_norm_weight = preloaded_resident_weight_device_buffer(
        &kv_a_norm_name,
        bf16_shape_bytes(&[GLM52_MLA_KV_LORA_RANK])?,
    )
    .context("resolving scheduler current MLA kv_a norm resident buffer")?;
    let input_norm_weight = preloaded_resident_weight_device_buffer(
        &input_norm_name,
        bf16_shape_bytes(&[GLM52_HIDDEN_SIZE])?,
    )
    .context("resolving scheduler current input RMSNorm resident buffer")?;
    let kv_a_weight = preloaded_resident_weight_device_buffer(
        &kv_a_name,
        bf16_shape_bytes(&[kv_a_rows, GLM52_HIDDEN_SIZE])?,
    )
    .context("resolving scheduler current kv_a projection resident buffer")?;

    let dsa = if glmrt_core::GLM52_DSA_INDEXER_LAYER_IDS_WITH_MTP.contains(&layer_id) {
        let wk_name = format!("model.layers.{layer_id}.self_attn.indexer.wk.weight");
        let k_norm_weight_name = format!("model.layers.{layer_id}.self_attn.indexer.k_norm.weight");
        let k_norm_bias_name = format!("model.layers.{layer_id}.self_attn.indexer.k_norm.bias");
        if !ensure_scheduler_bf16_tensor_resident(
            catalog,
            &wk_name,
            &[GLM52_DSA_INDEX_HEAD_DIM, GLM52_HIDDEN_SIZE],
            "scheduler current KV DSA wk pinned staging",
        )? || !ensure_scheduler_bf16_tensor_resident(
            catalog,
            &k_norm_weight_name,
            &[GLM52_DSA_INDEX_HEAD_DIM],
            "scheduler current KV DSA k_norm weight pinned staging",
        )? || !ensure_scheduler_bf16_tensor_resident(
            catalog,
            &k_norm_bias_name,
            &[GLM52_DSA_INDEX_HEAD_DIM],
            "scheduler current KV DSA k_norm bias pinned staging",
        )? {
            return Ok(None);
        }
        let wk_weight = preloaded_resident_weight_device_buffer(
            &wk_name,
            bf16_shape_bytes(&[GLM52_DSA_INDEX_HEAD_DIM, GLM52_HIDDEN_SIZE])?,
        )
        .context("resolving scheduler current DSA wk resident buffer")?;
        let k_norm_weight = preloaded_resident_weight_device_buffer(
            &k_norm_weight_name,
            bf16_shape_bytes(&[GLM52_DSA_INDEX_HEAD_DIM])?,
        )
        .context("resolving scheduler current DSA k_norm weight resident buffer")?;
        let k_norm_bias = preloaded_resident_weight_device_buffer(
            &k_norm_bias_name,
            bf16_shape_bytes(&[GLM52_DSA_INDEX_HEAD_DIM])?,
        )
        .context("resolving scheduler current DSA k_norm bias resident buffer")?;
        Some(SchedulerCurrentKvDsaResidentLayer {
            wk_name,
            wk_weight: wk_weight.into(),
            k_norm_weight_name,
            k_norm_weight: k_norm_weight.into(),
            k_norm_bias_name,
            k_norm_bias: k_norm_bias.into(),
        })
    } else {
        None
    };

    let layer = SchedulerCurrentKvResidentLayer {
        input_norm_name,
        input_norm_weight: input_norm_weight.into(),
        kv_a_name,
        kv_a_weight: kv_a_weight.into(),
        kv_norm_weight: kv_norm_weight.into(),
        dsa,
    };
    scheduler_current_kv_resident_layer_cache()
        .lock()
        .map_err(|_| anyhow::anyhow!("scheduler current KV resident layer cache poisoned"))?
        .insert(cache_key, layer.clone());
    Ok(Some(layer))
}

fn project_real_scheduler_attention_query_outputs(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden: super::progression::RealFullSchedulerDeviceHiddenSource,
    normalized_hidden: Option<&DeviceBf16Output>,
    project_dsa: bool,
) -> Result<Option<SchedulerRealAttentionQueryOutputs>> {
    let Some(layer) = scheduler_real_attention_resident_layer(catalog, layer_id)? else {
        return Ok(None);
    };
    let SchedulerRealAttentionResidentLayer {
        input_norm_name,
        input_norm_weight: _,
        q_a_name,
        q_a_norm_name,
        q_b_name,
        q_a_weight,
        q_a_norm_weight,
        q_b_weight,
        kv_norm_weight,
        kv_b_weight,
        output_projection_weight_name,
        q_b_rows,
        dsa,
    } = layer;

    let normalized_owned;
    let normalized_buffer = if let Some(normalized) = normalized_hidden {
        if normalized.rows != hidden.rows || normalized.values_per_row != GLM52_HIDDEN_SIZE {
            anyhow::bail!(
                "scheduler real MLA query normalized hidden shape mismatch for layer {layer_id}: expected {}x{} got {}x{}",
                hidden.rows,
                GLM52_HIDDEN_SIZE,
                normalized.rows,
                normalized.values_per_row
            );
        }
        normalized.buffer()
    } else {
        normalized_owned = rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output(
            &input_norm_name,
            hidden.buffer,
            hidden.rows,
            GLM52_HIDDEN_SIZE,
            REAL_FULL_DENSE_RMSNORM_EPS,
        )
        .context("normalizing scheduler hidden rows for real MLA query projection")?;
        normalized_owned.buffer()
    };
    let (q_projected, dsa) = if hidden.rows == 1 {
        if let Some(dsa) = dsa.filter(|_| project_dsa) {
            let (q_projected, dsa_query_projected, dsa_weights_projected) =
                mla_decode_query_dsa_projection_bf16_device_outputs(
                    layer_id,
                    normalized_buffer,
                    &q_a_name,
                    q_a_weight.map(CachedResidentDeviceBuffer::device_buffer),
                    q_a_norm_weight.device_buffer(),
                    &q_b_name,
                    q_b_weight.map(CachedResidentDeviceBuffer::device_buffer),
                    dsa.wq_b_weight.device_buffer(),
                    dsa.weights_proj_weight.device_buffer(),
                    hidden.rows,
                    GLM52_HIDDEN_SIZE,
                    REAL_FULL_SCHEDULER_MLA_Q_LORA_RANK,
                    q_b_rows,
                    REAL_FULL_DSA_INDEX_HEADS * GLM52_DSA_INDEX_HEAD_DIM,
                    REAL_FULL_DSA_INDEX_HEADS,
                    REAL_FULL_DENSE_RMSNORM_EPS,
                )
                .context("running fused scheduler real MLA and DSA decode query projections")?;
            (
                q_projected,
                Some(SchedulerRealAttentionDsaQueryOutputs {
                    query_projected: dsa_query_projected,
                    weights_projected: dsa_weights_projected,
                }),
            )
        } else {
            (
                mla_decode_query_projection_bf16_device_output(
                    layer_id,
                    normalized_buffer,
                    &q_a_name,
                    q_a_weight.map(CachedResidentDeviceBuffer::device_buffer),
                    q_a_norm_weight.device_buffer(),
                    &q_b_name,
                    q_b_weight.map(CachedResidentDeviceBuffer::device_buffer),
                    hidden.rows,
                    GLM52_HIDDEN_SIZE,
                    REAL_FULL_SCHEDULER_MLA_Q_LORA_RANK,
                    q_b_rows,
                    REAL_FULL_DENSE_RMSNORM_EPS,
                )
                .context("running fused scheduler real MLA decode query projection")?,
                None,
            )
        }
    } else {
        let q_a_projected = if coordinator_w8a16_q_a_decode_enabled() {
            linear_rows_w8a16_preloaded_resident_weight_device_output(
                &q_a_name,
                normalized_buffer,
                hidden.rows,
                GLM52_HIDDEN_SIZE,
                REAL_FULL_SCHEDULER_MLA_Q_LORA_RANK,
            )
        } else {
            linear_rows_bf16_preloaded_resident_weight_device_output(
                &q_a_name,
                normalized_buffer,
                None,
                hidden.rows,
                GLM52_HIDDEN_SIZE,
                REAL_FULL_SCHEDULER_MLA_Q_LORA_RANK,
                REAL_FULL_SCHEDULER_MLA_Q_LORA_RANK,
            )
        }
        .context("projecting scheduler hidden rows to real MLA q_a")?;
        let q_a_normalized = rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output(
            &q_a_norm_name,
            q_a_projected.buffer(),
            hidden.rows,
            REAL_FULL_SCHEDULER_MLA_Q_LORA_RANK,
            REAL_FULL_DENSE_RMSNORM_EPS,
        )
        .context("normalizing scheduler real MLA q_a rows")?;
        let dsa = dsa
            .filter(|_| project_dsa)
            .map(|dsa| {
                let SchedulerRealAttentionDsaResidentLayer {
                    wq_b_name,
                    wq_b_weight: _,
                    weights_proj_name,
                    weights_proj_weight: _,
                } = dsa;
                let query_projected = linear_rows_bf16_preloaded_resident_weight_device_output(
                    &wq_b_name,
                    q_a_normalized.buffer(),
                    None,
                    hidden.rows,
                    REAL_FULL_SCHEDULER_MLA_Q_LORA_RANK,
                    REAL_FULL_DSA_INDEX_HEADS * GLM52_DSA_INDEX_HEAD_DIM,
                    REAL_FULL_DSA_INDEX_HEADS * GLM52_DSA_INDEX_HEAD_DIM,
                )
                .context("projecting scheduler q_a rows to GLM DSA index queries")?;
                let weights_projected = linear_rows_bf16_preloaded_resident_weight_device_output(
                    &weights_proj_name,
                    normalized_buffer,
                    None,
                    hidden.rows,
                    GLM52_HIDDEN_SIZE,
                    REAL_FULL_DSA_INDEX_HEADS,
                    REAL_FULL_DSA_INDEX_HEADS,
                )
                .context("projecting scheduler normalized hidden rows to GLM DSA head weights")?;
                Ok::<SchedulerRealAttentionDsaQueryOutputs, anyhow::Error>(
                    SchedulerRealAttentionDsaQueryOutputs {
                        query_projected,
                        weights_projected,
                    },
                )
            })
            .transpose()?;
        let q_projected = if coordinator_w8a16_q_b_decode_enabled() {
            linear_rows_w8a16_preloaded_resident_weight_device_output(
                &q_b_name,
                q_a_normalized.buffer(),
                hidden.rows,
                REAL_FULL_SCHEDULER_MLA_Q_LORA_RANK,
                q_b_rows,
            )
        } else {
            linear_rows_bf16_preloaded_resident_weight_device_output(
                &q_b_name,
                q_a_normalized.buffer(),
                None,
                hidden.rows,
                REAL_FULL_SCHEDULER_MLA_Q_LORA_RANK,
                q_b_rows,
                q_b_rows,
            )
        }
        .context("projecting scheduler q_a rows to real MLA q_b")?;
        (q_projected, dsa)
    };

    Ok(Some(SchedulerRealAttentionQueryOutputs {
        q_projected,
        dsa,
        kv_norm_weight: kv_norm_weight.device_buffer(),
        kv_b_weight: kv_b_weight.device_buffer(),
        output_projection_weight_name,
    }))
}

fn project_real_scheduler_attention_queries_scalar_q_a_batched_q_b(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden: RealFullSchedulerDeviceHiddenSource,
    project_dsa: bool,
) -> Result<Option<SchedulerRealAttentionQueryOutputs>> {
    let max_rows = if coordinator_w8a16_q_b_decode_enabled() {
        MTP_TARGET_SCALAR_Q_A_BATCHED_Q_B_MAX_ROWS
    } else {
        8
    };
    if hidden.rows <= 1
        || hidden.rows > max_rows
        || !(coordinator_w4a16_q_b_decode_enabled() || coordinator_w8a16_q_b_decode_enabled())
    {
        return Ok(None);
    }
    let Some(layer) = scheduler_real_attention_resident_layer(catalog, layer_id)? else {
        return Ok(None);
    };
    let SchedulerRealAttentionResidentLayer {
        input_norm_name: _,
        input_norm_weight,
        q_a_name,
        q_a_norm_name: _,
        q_b_name,
        q_a_weight,
        q_a_norm_weight,
        q_b_weight: _,
        kv_norm_weight,
        kv_b_weight,
        output_projection_weight_name,
        q_b_rows,
        dsa,
    } = layer;
    let dsa_weights = dsa.as_ref().filter(|_| project_dsa).map(|dsa| {
        (
            dsa.wq_b_weight.device_buffer(),
            dsa.weights_proj_weight.device_buffer(),
            REAL_FULL_DSA_INDEX_HEADS * GLM52_DSA_INDEX_HEAD_DIM,
            REAL_FULL_DSA_INDEX_HEADS,
        )
    });
    let (q_projected, dsa_query_projected, dsa_weights_projected) =
        mla_decode_scalar_q_a_batched_q_b_projection_bf16_device_outputs(
            layer_id,
            hidden.buffer,
            input_norm_weight.device_buffer(),
            &q_a_name,
            q_a_weight.map(CachedResidentDeviceBuffer::device_buffer),
            q_a_norm_weight.device_buffer(),
            &q_b_name,
            dsa_weights,
            hidden.rows,
            GLM52_HIDDEN_SIZE,
            REAL_FULL_SCHEDULER_MLA_Q_LORA_RANK,
            q_b_rows,
            REAL_FULL_DENSE_RMSNORM_EPS,
        )
        .context("projecting batched MTP query and optional DSA rows")?;
    let dsa = match (dsa_query_projected, dsa_weights_projected) {
        (Some(query_projected), Some(weights_projected)) => {
            Some(SchedulerRealAttentionDsaQueryOutputs {
                query_projected,
                weights_projected,
            })
        }
        (None, None) => None,
        _ => anyhow::bail!("batched MTP DSA projection returned an incomplete output pair"),
    };

    Ok(Some(SchedulerRealAttentionQueryOutputs {
        q_projected,
        dsa,
        kv_norm_weight: kv_norm_weight.device_buffer(),
        kv_b_weight: kv_b_weight.device_buffer(),
        output_projection_weight_name,
    }))
}

fn scheduler_real_attention_resident_layer(
    catalog: &TensorCatalog,
    layer_id: usize,
) -> Result<Option<SchedulerRealAttentionResidentLayer>> {
    let cache_key = scheduler_real_attention_layer_cache_key(catalog, layer_id);
    if let Some(layer) = scheduler_real_attention_resident_layer_cache()
        .lock()
        .map_err(|_| anyhow::anyhow!("scheduler real attention resident layer cache poisoned"))?
        .get(&cache_key)
        .cloned()
    {
        return Ok(Some(layer));
    }

    let heads = REAL_FULL_SCHEDULER_MLA_NUM_ATTENTION_HEADS;
    let q_head_dim = REAL_FULL_SCHEDULER_MLA_QK_NOPE_HEAD_DIM
        .checked_add(GLM52_MLA_QK_ROPE_HEAD_DIM)
        .context("scheduler real MLA q head dim overflow")?;
    let q_b_rows = heads
        .checked_mul(q_head_dim)
        .context("scheduler real MLA q_b rows overflow")?;
    let kv_b_rows = heads
        .checked_mul(
            REAL_FULL_SCHEDULER_MLA_QK_NOPE_HEAD_DIM
                .checked_add(REAL_FULL_SCHEDULER_MLA_V_HEAD_DIM)
                .context("scheduler real MLA kv_b head width overflow")?,
        )
        .context("scheduler real MLA kv_b rows overflow")?;
    let context_width = heads
        .checked_mul(REAL_FULL_SCHEDULER_MLA_V_HEAD_DIM)
        .context("scheduler real MLA context width overflow")?;
    let input_norm_name = format!("model.layers.{layer_id}.input_layernorm.weight");
    let q_a_name = format!("model.layers.{layer_id}.self_attn.q_a_proj.weight");
    let q_a_norm_name = format!("model.layers.{layer_id}.self_attn.q_a_layernorm.weight");
    let q_b_name = format!("model.layers.{layer_id}.self_attn.q_b_proj.weight");
    let kv_a_norm_name = format!("model.layers.{layer_id}.self_attn.kv_a_layernorm.weight");
    let kv_b_name = format!("model.layers.{layer_id}.self_attn.kv_b_proj.weight");
    let o_proj_name = format!("model.layers.{layer_id}.self_attn.o_proj.weight");
    let w8a16_q_b_enabled = coordinator_w8a16_q_b_decode_enabled();
    let w8a16_q_a_enabled = coordinator_w8a16_q_a_decode_enabled();
    let w8a16_o_proj_enabled = coordinator_w8a16_o_proj_decode_enabled();

    let tensor_specs = [
        (
            input_norm_name.as_str(),
            vec![GLM52_HIDDEN_SIZE],
            "scheduler real attention input RMSNorm pinned staging",
        ),
        (
            q_a_name.as_str(),
            vec![REAL_FULL_SCHEDULER_MLA_Q_LORA_RANK, GLM52_HIDDEN_SIZE],
            "scheduler real attention q_a projection pinned staging",
        ),
        (
            q_a_norm_name.as_str(),
            vec![REAL_FULL_SCHEDULER_MLA_Q_LORA_RANK],
            "scheduler real attention q_a RMSNorm pinned staging",
        ),
        (
            q_b_name.as_str(),
            vec![q_b_rows, REAL_FULL_SCHEDULER_MLA_Q_LORA_RANK],
            "scheduler real attention q_b projection pinned staging",
        ),
        (
            kv_a_norm_name.as_str(),
            vec![GLM52_MLA_KV_LORA_RANK],
            "scheduler real attention kv_a RMSNorm pinned staging",
        ),
        (
            kv_b_name.as_str(),
            vec![kv_b_rows, GLM52_MLA_KV_LORA_RANK],
            "scheduler real attention kv_b projection pinned staging",
        ),
        (
            o_proj_name.as_str(),
            vec![GLM52_HIDDEN_SIZE, context_width],
            "scheduler real attention o projection pinned staging",
        ),
    ];
    for (tensor_name, expected_shape, label) in tensor_specs {
        if (w8a16_q_a_enabled && tensor_name == q_a_name)
            || (w8a16_q_b_enabled && tensor_name == q_b_name)
            || (w8a16_o_proj_enabled && tensor_name == o_proj_name)
        {
            continue;
        }
        if !ensure_scheduler_bf16_tensor_resident(
            catalog,
            tensor_name,
            expected_shape.as_slice(),
            label,
        )? {
            return Ok(None);
        }
    }

    let kv_norm_weight = preloaded_resident_weight_device_buffer(
        &kv_a_norm_name,
        bf16_shape_bytes(&[GLM52_MLA_KV_LORA_RANK])?,
    )
    .context("resolving scheduler real MLA kv_a norm resident buffer")?;
    let input_norm_weight = preloaded_resident_weight_device_buffer(
        &input_norm_name,
        bf16_shape_bytes(&[GLM52_HIDDEN_SIZE])?,
    )
    .context("resolving scheduler real attention input norm resident buffer")?;
    let kv_b_weight = preloaded_resident_weight_device_buffer(
        &kv_b_name,
        bf16_shape_bytes(&[kv_b_rows, GLM52_MLA_KV_LORA_RANK])?,
    )
    .context("resolving scheduler real MLA kv_b resident buffer")?;
    let q_a_weight = (!w8a16_q_a_enabled)
        .then(|| {
            preloaded_resident_weight_device_buffer(
                &q_a_name,
                bf16_shape_bytes(&[REAL_FULL_SCHEDULER_MLA_Q_LORA_RANK, GLM52_HIDDEN_SIZE])?,
            )
            .context("resolving scheduler real MLA q_a resident buffer")
        })
        .transpose()?;
    let q_a_norm_weight = preloaded_resident_weight_device_buffer(
        &q_a_norm_name,
        bf16_shape_bytes(&[REAL_FULL_SCHEDULER_MLA_Q_LORA_RANK])?,
    )
    .context("resolving scheduler real MLA q_a norm resident buffer")?;
    let q_b_weight = (!w8a16_q_b_enabled)
        .then(|| {
            preloaded_resident_weight_device_buffer(
                &q_b_name,
                bf16_shape_bytes(&[q_b_rows, REAL_FULL_SCHEDULER_MLA_Q_LORA_RANK])?,
            )
            .context("resolving scheduler real MLA q_b resident buffer")
        })
        .transpose()?;
    let dsa = if GLM52_DSA_INDEXER_LAYER_IDS_WITH_MTP.contains(&layer_id) {
        let wq_b_name = format!("model.layers.{layer_id}.self_attn.indexer.wq_b.weight");
        let weights_proj_name =
            format!("model.layers.{layer_id}.self_attn.indexer.weights_proj.weight");
        let dsa_query_width = REAL_FULL_DSA_INDEX_HEADS
            .checked_mul(GLM52_DSA_INDEX_HEAD_DIM)
            .context("scheduler real DSA query width overflow")?;
        if !ensure_scheduler_bf16_tensor_resident(
            catalog,
            &wq_b_name,
            &[dsa_query_width, REAL_FULL_SCHEDULER_MLA_Q_LORA_RANK],
            "scheduler real attention DSA wq_b pinned staging",
        )? || !ensure_scheduler_bf16_tensor_resident(
            catalog,
            &weights_proj_name,
            &[REAL_FULL_DSA_INDEX_HEADS, GLM52_HIDDEN_SIZE],
            "scheduler real attention DSA weights projection pinned staging",
        )? {
            return Ok(None);
        }
        let wq_b_weight = preloaded_resident_weight_device_buffer(
            &wq_b_name,
            bf16_shape_bytes(&[dsa_query_width, REAL_FULL_SCHEDULER_MLA_Q_LORA_RANK])?,
        )
        .context("resolving scheduler real DSA wq_b resident buffer")?;
        let weights_proj_weight = preloaded_resident_weight_device_buffer(
            &weights_proj_name,
            bf16_shape_bytes(&[REAL_FULL_DSA_INDEX_HEADS, GLM52_HIDDEN_SIZE])?,
        )
        .context("resolving scheduler real DSA weights projection resident buffer")?;
        Some(SchedulerRealAttentionDsaResidentLayer {
            wq_b_name,
            wq_b_weight: wq_b_weight.into(),
            weights_proj_name,
            weights_proj_weight: weights_proj_weight.into(),
        })
    } else {
        None
    };
    let layer = SchedulerRealAttentionResidentLayer {
        input_norm_name,
        input_norm_weight: input_norm_weight.into(),
        q_a_name,
        q_a_norm_name,
        q_b_name,
        q_a_weight: q_a_weight.map(Into::into),
        q_a_norm_weight: q_a_norm_weight.into(),
        q_b_weight: q_b_weight.map(Into::into),
        kv_norm_weight: kv_norm_weight.into(),
        kv_b_weight: kv_b_weight.into(),
        output_projection_weight_name: o_proj_name,
        q_b_rows,
        dsa,
    };
    scheduler_real_attention_resident_layer_cache()
        .lock()
        .map_err(|_| anyhow::anyhow!("scheduler real attention resident layer cache poisoned"))?
        .insert(cache_key, layer.clone());
    Ok(Some(layer))
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct SchedulerCurrentKvLayerCacheKey {
    model_id: String,
    snapshot_path: String,
    layer_id: usize,
}

static SCHEDULER_CURRENT_KV_RESIDENT_LAYER_CACHE: OnceLock<
    Mutex<BTreeMap<SchedulerCurrentKvLayerCacheKey, SchedulerCurrentKvResidentLayer>>,
> = OnceLock::new();

fn scheduler_current_kv_resident_layer_cache(
) -> &'static Mutex<BTreeMap<SchedulerCurrentKvLayerCacheKey, SchedulerCurrentKvResidentLayer>> {
    SCHEDULER_CURRENT_KV_RESIDENT_LAYER_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn scheduler_current_kv_layer_cache_key(
    catalog: &TensorCatalog,
    layer_id: usize,
) -> SchedulerCurrentKvLayerCacheKey {
    SchedulerCurrentKvLayerCacheKey {
        model_id: catalog.model_id.clone(),
        snapshot_path: catalog.snapshot_path.clone(),
        layer_id,
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct SchedulerRealAttentionLayerCacheKey {
    model_id: String,
    snapshot_path: String,
    layer_id: usize,
}

static SCHEDULER_REAL_ATTENTION_RESIDENT_LAYER_CACHE: OnceLock<
    Mutex<BTreeMap<SchedulerRealAttentionLayerCacheKey, SchedulerRealAttentionResidentLayer>>,
> = OnceLock::new();

fn scheduler_real_attention_resident_layer_cache() -> &'static Mutex<
    BTreeMap<SchedulerRealAttentionLayerCacheKey, SchedulerRealAttentionResidentLayer>,
> {
    SCHEDULER_REAL_ATTENTION_RESIDENT_LAYER_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn scheduler_real_attention_layer_cache_key(
    catalog: &TensorCatalog,
    layer_id: usize,
) -> SchedulerRealAttentionLayerCacheKey {
    SchedulerRealAttentionLayerCacheKey {
        model_id: catalog.model_id.clone(),
        snapshot_path: catalog.snapshot_path.clone(),
        layer_id,
    }
}

fn real_full_scheduler_mla_attention_scale() -> f32 {
    let q_head_dim = REAL_FULL_SCHEDULER_MLA_QK_NOPE_HEAD_DIM + GLM52_MLA_QK_ROPE_HEAD_DIM;
    (q_head_dim as f32).sqrt().recip()
}

fn ensure_scheduler_bf16_tensor_resident(
    catalog: &TensorCatalog,
    tensor_name: &str,
    expected_shape: &[usize],
    label: &'static str,
) -> Result<bool> {
    let expected_bytes = bf16_shape_bytes(expected_shape)?;
    let cache_key = scheduler_resident_bf16_tensor_cache_key(catalog, tensor_name, expected_bytes);
    if scheduler_resident_bf16_tensor_cache()
        .lock()
        .map_err(|_| anyhow::anyhow!("scheduler resident BF16 tensor cache poisoned"))?
        .contains_key(&cache_key)
    {
        return Ok(true);
    }
    if resident_weight_is_preloaded(tensor_name, expected_bytes) {
        scheduler_resident_bf16_tensor_cache()
            .lock()
            .map_err(|_| anyhow::anyhow!("scheduler resident BF16 tensor cache poisoned"))?
            .insert(cache_key, ());
        return Ok(true);
    }
    if !std::path::Path::new(catalog.snapshot_path.as_str()).exists() {
        return Ok(false);
    }
    preload_resident_weight_from_host_staging(tensor_name, expected_bytes, label, |staging| {
        let summary = read_tensor_bytes_into(catalog, tensor_name, staging)
            .with_context(|| format!("reading scheduler KV tensor {tensor_name}"))?;
        if summary.dtype != DType::Bf16 {
            anyhow::bail!(
                "scheduler KV tensor {tensor_name} expects BF16, got {:?}",
                summary.dtype
            );
        }
        if summary.shape != expected_shape {
            anyhow::bail!(
                "scheduler KV tensor {tensor_name} shape mismatch: expected {:?} got {:?}",
                expected_shape,
                summary.shape
            );
        }
        if summary.bytes_read as usize != expected_bytes {
            anyhow::bail!(
                "scheduler KV tensor {tensor_name} read {} bytes, expected {expected_bytes}",
                summary.bytes_read
            );
        }
        Ok(())
    })
    .with_context(|| format!("preloading scheduler KV tensor {tensor_name}"))?;
    scheduler_resident_bf16_tensor_cache()
        .lock()
        .map_err(|_| anyhow::anyhow!("scheduler resident BF16 tensor cache poisoned"))?
        .insert(cache_key, ());
    Ok(true)
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct SchedulerResidentBf16TensorCacheKey {
    model_id: String,
    snapshot_path: String,
    tensor_name: String,
    expected_bytes: usize,
}

static SCHEDULER_RESIDENT_BF16_TENSOR_CACHE: OnceLock<
    Mutex<BTreeMap<SchedulerResidentBf16TensorCacheKey, ()>>,
> = OnceLock::new();

fn scheduler_resident_bf16_tensor_cache(
) -> &'static Mutex<BTreeMap<SchedulerResidentBf16TensorCacheKey, ()>> {
    SCHEDULER_RESIDENT_BF16_TENSOR_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn scheduler_resident_bf16_tensor_cache_key(
    catalog: &TensorCatalog,
    tensor_name: &str,
    expected_bytes: usize,
) -> SchedulerResidentBf16TensorCacheKey {
    SchedulerResidentBf16TensorCacheKey {
        model_id: catalog.model_id.clone(),
        snapshot_path: catalog.snapshot_path.clone(),
        tensor_name: tensor_name.to_owned(),
        expected_bytes,
    }
}

fn validate_projected_scheduler_kv_outputs(
    config: &KvCacheConfig,
    descriptors: &[KvBlockDescriptor],
    kv_a_projected: &DeviceBf16Output,
    dsa_key: Option<&DeviceBf16Output>,
) -> Result<()> {
    let layer_id = descriptors
        .first()
        .map(|descriptor| descriptor.layer_id)
        .context("projected scheduler KV output validation requires at least one descriptor")?;
    let rows = descriptor_row_count(descriptors)?;
    let main_values_per_row = GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM;
    if kv_a_projected.rows != rows || kv_a_projected.values_per_row != main_values_per_row {
        anyhow::bail!(
            "projected scheduler kv_a output shape mismatch for layer {}: expected {}x{} got {}x{}",
            layer_id.0,
            rows,
            main_values_per_row,
            kv_a_projected.rows,
            kv_a_projected.values_per_row
        );
    }
    let main_stride_bytes = main_values_per_row
        .checked_mul(std::mem::size_of::<u16>())
        .context("projected scheduler KV main stride overflow")?;
    if config.layer_has_dsa_indexer(layer_id) {
        let dsa_key = dsa_key.context("projected scheduler DSA KV payload missing DSA key")?;
        if dsa_key.rows != rows || dsa_key.values_per_row != GLM52_DSA_INDEX_HEAD_DIM {
            anyhow::bail!(
                "projected scheduler DSA key shape mismatch for layer {}: expected {}x{} got {}x{}",
                layer_id.0,
                rows,
                GLM52_DSA_INDEX_HEAD_DIM,
                dsa_key.rows,
                dsa_key.values_per_row
            );
        }
        let dsa_stride_bytes = GLM52_DSA_INDEX_HEAD_DIM
            .checked_mul(std::mem::size_of::<u16>())
            .context("projected scheduler KV DSA stride overflow")?;
        let payload_stride_bytes = main_stride_bytes
            .checked_add(dsa_stride_bytes)
            .context("projected scheduler KV DSA payload stride overflow")?;
        validate_projected_scheduler_kv_stride(config, layer_id, payload_stride_bytes)?;
    } else {
        if dsa_key.is_some() {
            anyhow::bail!(
                "projected scheduler non-DSA KV payload for layer {} unexpectedly has DSA key rows",
                layer_id.0
            );
        }
        validate_projected_scheduler_kv_stride(config, layer_id, main_stride_bytes)?;
    }
    Ok(())
}

fn validate_projected_scheduler_kv_stride(
    config: &KvCacheConfig,
    layer_id: glmrt_core::LayerId,
    payload_stride_bytes: usize,
) -> Result<()> {
    if !matches!(
        config.dtype,
        KvCacheDType::Bf16 | KvCacheDType::Fp8 | KvCacheDType::Nvfp4
    ) {
        anyhow::bail!(
            "projected scheduler KV does not support {} cache storage",
            config.dtype.label()
        );
    }
    let dsa_dim = if config.layer_has_dsa_indexer(layer_id) {
        GLM52_DSA_INDEX_HEAD_DIM
    } else {
        0
    };
    let expected_projected_stride = (GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM + dsa_dim)
        .checked_mul(std::mem::size_of::<u16>())
        .context("projected scheduler KV expected stride overflow")?;
    if payload_stride_bytes != expected_projected_stride {
        anyhow::bail!(
            "projected scheduler KV stride mismatch for layer {}: expected={} projected={}",
            layer_id.0,
            expected_projected_stride,
            payload_stride_bytes
        );
    }
    Ok(())
}

fn scheduler_device_kv_readback_validation_enabled() -> bool {
    match std::env::var("GLMRT_REAL_FULL_VALIDATE_DEVICE_KV_READBACK") {
        Ok(value) => matches!(
            value.as_str(),
            "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
        ),
        Err(_) => cfg!(test),
    }
}

fn mtp_target_attention_fusion_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(MTP_TARGET_ATTENTION_FUSION_ENV)
            .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
            .unwrap_or(true)
    })
}

fn should_fuse_mtp_target_attention(
    config: &KvCacheConfig,
    selected: &[LayerWave],
) -> Result<bool> {
    if !mtp_target_attention_fusion_enabled()
        || !matches!(config.dtype, KvCacheDType::Fp8 | KvCacheDType::Nvfp4)
        || selected.len() != 2
    {
        return Ok(false);
    }
    let decode_wave = &selected[0];
    let mtp_wave = &selected[1];
    let decode = single_wave_source(decode_wave)?;
    let mtp = single_wave_source(mtp_wave)?;
    let rows = decode
        .row_count
        .checked_add(mtp.row_count)
        .context("fused MTP target attention row count overflow")?;
    let contiguous = decode
        .token_start
        .0
        .checked_add(decode.row_count as u64)
        .is_some_and(|next| next == mtp.token_start.0);
    Ok(decode_wave.layer_id == mtp_wave.layer_id
        && decode.kind == RowSourceKind::DecodeStep
        && mtp.kind == RowSourceKind::MtpVerifyBlock
        && decode.row_count == 1
        && mtp.row_count > 0
        && rows <= MTP_TARGET_ATTENTION_FUSION_MAX_ROWS
        && contiguous
        && !decode_wave.kv_writes.is_empty()
        && decode_wave.tentative_kv_writes.is_empty()
        && mtp_wave.kv_writes.is_empty()
        && !mtp_wave.tentative_kv_writes.is_empty()
        && descriptor_row_count(&decode_wave.kv_writes)? == decode.row_count
        && descriptor_row_count(&mtp_wave.tentative_kv_writes)? == mtp.row_count)
}

fn admission_stage_timing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(ADMISSION_STAGE_TIMING_ENV)
            .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
            .unwrap_or(false)
    })
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn elapsed_ms_optional(start: Option<Instant>) -> f64 {
    start.map(elapsed_ms).unwrap_or(0.0)
}

fn descriptor_row_count(descriptors: &[KvBlockDescriptor]) -> Result<usize> {
    descriptors
        .iter()
        .try_fold(0_usize, |acc, descriptor| {
            acc.checked_add(descriptor.token_count)
        })
        .context("scheduler KV descriptor row count overflow")
}

fn bf16_shape_bytes(shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1_usize, |acc, dim| acc.checked_mul(*dim))
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("scheduler BF16 tensor byte length overflow")
}

#[allow(clippy::too_many_arguments)]
fn real_full_launch_fused_mtp_target_attention(
    config: &KvCacheConfig,
    selected: &[LayerWave],
    prepared: Vec<PreparedSchedulerAttention>,
    counters: &mut RealFullSchedulerExecutionCounters,
    device_kv: &mut RealFullDeviceKvExecutionMirror,
    numeric_progression: &mut RealFullSchedulerNumericProgression,
    catalog: &TensorCatalog,
) -> Result<Vec<RealFullSchedulerDeviceAttentionDelta>> {
    let stage_timing = admission_stage_timing_enabled();
    let total_start = stage_timing.then(Instant::now);
    anyhow::ensure!(
        selected.len() == 2 && prepared.len() == 2,
        "fused decode/MTP attention expected two selected and prepared waves, got selected={} prepared={}",
        selected.len(),
        prepared.len()
    );
    anyhow::ensure!(
        prepared[0].wave_index == 0 && prepared[1].wave_index == 1,
        "fused decode/MTP attention wave order mismatch: {:?}",
        prepared
            .iter()
            .map(|part| part.wave_index)
            .collect::<Vec<_>>()
    );
    let decode_wave = &selected[0];
    let mtp_wave = &selected[1];
    let decode = single_wave_source(decode_wave)?;
    let mtp = single_wave_source(mtp_wave)?;
    let total_rows = decode
        .row_count
        .checked_add(mtp.row_count)
        .context("fused decode/MTP attention row count overflow")?;

    let hidden_fuse_start = stage_timing.then(Instant::now);
    numeric_progression
        .fuse_device_hidden_sources(&[decode, mtp], decode_wave.layer_id.0 as usize)
        .context("fusing resident decode/MTP hidden rows for attention")?;

    let fused_hidden = numeric_progression
        .device_hidden_source(decode.kind, decode.token_start.0 as usize, total_rows)
        .context("resolving fused decode/MTP hidden rows for q1-shaped query projection")?
        .context("fused decode/MTP hidden rows are unavailable after fusion")?;
    let hidden_fuse_ms = elapsed_ms_optional(hidden_fuse_start);

    let prepare_start = stage_timing.then(Instant::now);
    let mut fused_wave = decode_wave.clone();
    fused_wave.hidden_shape = glmrt_core::HiddenShape::glm52_bf16_rows(total_rows);
    fused_wave.row_sources[0].row_count = total_rows;
    fused_wave
        .kv_writes
        .extend(mtp_wave.tentative_kv_writes.iter().cloned());
    fused_wave.graph_bucket = GraphBucket::new(total_rows);
    let attention_context_rows = prepared[0]
        .visible_blocks
        .iter()
        .map(|block| &block.descriptor)
        .chain(fused_wave.kv_writes.iter())
        .try_fold(0_usize, |rows, descriptor| {
            rows.checked_add(descriptor.token_count)
                .context("fused MTP target attention context row count overflow")
        })?;
    let prepare_ms = elapsed_ms_optional(prepare_start);
    let query_project_start = stage_timing.then(Instant::now);
    let projected_query = project_real_scheduler_attention_queries_scalar_q_a_batched_q_b(
        catalog,
        fused_wave.layer_id.0 as usize,
        fused_hidden,
        attention_context_rows > REAL_FULL_DSA_TOP_K,
    )?;
    let query_project_ms = elapsed_ms_optional(query_project_start);
    let device_attention_start = stage_timing.then(Instant::now);
    let combined = real_full_launch_scheduler_attention_from_device_kv_with_query_override(
        config,
        &fused_wave,
        &prepared[0].visible_blocks,
        &fused_wave.kv_writes,
        counters,
        device_kv,
        numeric_progression,
        catalog,
        None,
        Some(fused_hidden),
        None,
        projected_query,
    )?
    .context("fused decode/MTP attention produced no device delta")?;
    let device_attention_ms = elapsed_ms_optional(device_attention_start);
    anyhow::ensure!(
        combined.row_count == total_rows,
        "fused decode/MTP attention output rows {} do not match input rows {total_rows}",
        combined.row_count
    );

    let row_bytes = combined
        .values_per_row
        .checked_mul(std::mem::size_of::<u16>())
        .context("fused decode/MTP attention output row bytes overflow")?;
    let split_host_rows = |row_offset: usize, row_count: usize| -> Result<Option<Vec<u8>>> {
        let Some(bytes) = combined.output_bf16.as_ref() else {
            return Ok(None);
        };
        let byte_start = row_offset
            .checked_mul(row_bytes)
            .context("fused decode/MTP host output byte start overflow")?;
        let byte_end = row_offset
            .checked_add(row_count)
            .and_then(|rows| rows.checked_mul(row_bytes))
            .context("fused decode/MTP host output byte end overflow")?;
        anyhow::ensure!(
            byte_end <= bytes.len(),
            "fused decode/MTP host output range {byte_start}..{byte_end} exceeds {} bytes",
            bytes.len()
        );
        Ok(Some(bytes[byte_start..byte_end].to_vec()))
    };
    let decode_host = split_host_rows(0, decode.row_count)?;
    let mtp_host = split_host_rows(decode.row_count, mtp.row_count)?;
    let checksum = |bytes: Option<&Vec<u8>>| -> Result<f64> {
        bytes
            .map(|bytes| {
                bf16_bytes_to_f32(bytes)
                    .map(|values| values.into_iter().map(f64::from).sum::<f64>())
            })
            .transpose()
            .map(|value| value.unwrap_or(0.0))
    };
    let decode_checksum = checksum(decode_host.as_ref())?;
    let mtp_checksum = checksum(mtp_host.as_ref())?;
    let output_device = combined.output_device;
    let base_row_offset = combined.output_device_row_offset;
    let output = vec![
        RealFullSchedulerDeviceAttentionDelta {
            kind: decode.kind,
            token_start: decode.token_start.0 as usize,
            row_count: decode.row_count,
            values_per_row: combined.values_per_row,
            output_bf16: decode_host,
            output_device: Arc::clone(&output_device),
            output_device_row_offset: base_row_offset,
            checksum: decode_checksum,
            backend: combined.backend,
        },
        RealFullSchedulerDeviceAttentionDelta {
            kind: mtp.kind,
            token_start: mtp.token_start.0 as usize,
            row_count: mtp.row_count,
            values_per_row: combined.values_per_row,
            output_bf16: mtp_host,
            output_device,
            output_device_row_offset: base_row_offset + decode.row_count,
            checksum: mtp_checksum,
            backend: combined.backend,
        },
    ];
    if stage_timing {
        eprintln!(
            "real_full_fused_mtp_attention_timing layer_id={} rows={} hidden_fuse_ms={:.3} prepare_ms={:.3} query_project_ms={:.3} device_attention_ms={:.3} total_ms={:.3}",
            decode_wave.layer_id.0,
            total_rows,
            hidden_fuse_ms,
            prepare_ms,
            query_project_ms,
            device_attention_ms,
            elapsed_ms_optional(total_start),
        );
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn real_full_launch_sequential_mtp_target_attention(
    config: &KvCacheConfig,
    wave: &LayerWave,
    visible_blocks: &[KvBackedBlock],
    current_descriptors: &[KvBlockDescriptor],
    counters: &mut RealFullSchedulerExecutionCounters,
    device_kv: &mut RealFullDeviceKvExecutionMirror,
    numeric_progression: &mut RealFullSchedulerNumericProgression,
    catalog: &TensorCatalog,
    normalized_hidden: Option<&DeviceBf16Output>,
) -> Result<Option<RealFullSchedulerDeviceAttentionDelta>> {
    let source = single_wave_source(wave)?;
    if source.kind != RowSourceKind::MtpVerifyBlock || source.row_count <= 1 {
        return real_full_launch_scheduler_attention_from_device_kv(
            config,
            wave,
            visible_blocks,
            current_descriptors,
            counters,
            device_kv,
            numeric_progression,
            catalog,
            normalized_hidden,
        );
    }
    anyhow::ensure!(
        current_descriptors.len() == source.row_count
            && current_descriptors
                .iter()
                .all(|descriptor| descriptor.token_count == 1),
        "sequential MTP target attention requires one KV descriptor per row: rows={} descriptors={}",
        source.row_count,
        current_descriptors.len()
    );
    let Some(hidden) = numeric_progression
        .device_hidden_source(source.kind, source.token_start.0 as usize, source.row_count)
        .context("resolving sequential MTP target hidden rows")?
    else {
        return real_full_launch_scheduler_attention_from_device_kv(
            config,
            wave,
            visible_blocks,
            current_descriptors,
            counters,
            device_kv,
            numeric_progression,
            catalog,
            normalized_hidden,
        );
    };
    let row_bytes = hidden
        .values_per_row
        .checked_mul(std::mem::size_of::<u16>())
        .context("sequential MTP target hidden row bytes overflow")?;
    let visible_rows = visible_blocks.iter().try_fold(0_usize, |rows, block| {
        rows.checked_add(block.descriptor.token_count)
            .context("sequential MTP target visible row count overflow")
    })?;
    let final_attention_rows = visible_rows
        .checked_add(current_descriptors.len())
        .context("sequential MTP target final attention row count overflow")?;
    let projected = project_real_scheduler_attention_queries_scalar_q_a_batched_q_b(
        catalog,
        wave.layer_id.0 as usize,
        hidden,
        final_attention_rows > REAL_FULL_DSA_TOP_K,
    )
    .with_context(|| {
        format!(
            "projecting batched sequential MTP target queries for layer {}",
            wave.layer_id.0
        )
    })?;
    if let Some(projected) = projected.as_ref() {
        anyhow::ensure!(
            projected.q_projected.rows == source.row_count,
            "sequential MTP target projected query rows {} do not match source rows {}",
            projected.q_projected.rows,
            source.row_count
        );
    }
    let mut row_deltas = Vec::with_capacity(source.row_count);
    for (row, descriptor) in current_descriptors.iter().enumerate() {
        let token_start = source
            .token_start
            .0
            .checked_add(row as u64)
            .context("sequential MTP target token position overflow")?;
        anyhow::ensure!(
            descriptor.token_start.0 == token_start,
            "sequential MTP target descriptor row {row} starts at {}, expected {token_start}",
            descriptor.token_start.0
        );
        let row_buffer = device_buffer_byte_view(
            hidden.buffer,
            row.checked_mul(row_bytes)
                .context("sequential MTP target hidden row offset overflow")?,
            row_bytes,
            "sequential MTP target hidden row",
        )?;
        let mut row_wave = wave.clone();
        row_wave.hidden_shape = glmrt_core::HiddenShape::glm52_bf16_rows(1);
        row_wave.row_sources[0].token_start = glmrt_core::PositionId(token_start);
        row_wave.row_sources[0].row_count = 1;
        row_wave.tentative_kv_writes = vec![descriptor.clone()];
        row_wave.graph_bucket = GraphBucket::decode();
        let row_hidden = RealFullSchedulerDeviceHiddenSource {
            buffer: row_buffer,
            rows: 1,
            values_per_row: hidden.values_per_row,
        };
        let projected_row = projected
            .as_ref()
            .map(|projected| {
                let attention_rows = visible_rows
                    .checked_add(row + 1)
                    .context("sequential MTP target row attention count overflow")?;
                let dsa = if attention_rows > REAL_FULL_DSA_TOP_K {
                    projected
                        .dsa
                        .as_ref()
                        .map(|dsa| {
                            Ok::<_, anyhow::Error>(SchedulerRealAttentionDsaQueryOutputs {
                                query_projected: copy_mla_decode_query_row_to_attention_stream(
                                    wave.layer_id.0 as usize,
                                    &dsa.query_projected,
                                    row,
                                    "sequential MTP target projected DSA query row",
                                )?,
                                weights_projected: copy_mla_decode_query_row_to_attention_stream(
                                    wave.layer_id.0 as usize,
                                    &dsa.weights_projected,
                                    row,
                                    "sequential MTP target projected DSA weights row",
                                )?,
                            })
                        })
                        .transpose()?
                } else {
                    None
                };
                Ok::<_, anyhow::Error>(SchedulerRealAttentionQueryOutputs {
                    q_projected: copy_mla_decode_query_row_to_attention_stream(
                        wave.layer_id.0 as usize,
                        &projected.q_projected,
                        row,
                        "sequential MTP target projected query row",
                    )?,
                    dsa,
                    kv_norm_weight: projected.kv_norm_weight,
                    kv_b_weight: projected.kv_b_weight,
                    output_projection_weight_name: projected.output_projection_weight_name.clone(),
                })
            })
            .transpose()?;
        let delta = real_full_launch_scheduler_attention_from_device_kv_with_query_override(
            config,
            &row_wave,
            visible_blocks,
            &current_descriptors[..=row],
            counters,
            device_kv,
            numeric_progression,
            catalog,
            None,
            Some(row_hidden),
            Some(std::slice::from_ref(descriptor)),
            projected_row,
        )?
        .context("sequential MTP target attention produced no device delta")?;
        row_deltas.push(delta);
    }

    let first = row_deltas
        .first()
        .context("sequential MTP target attention produced no row deltas")?;
    let values_per_row = first.values_per_row;
    let backend = first.backend;
    for (row, delta) in row_deltas.iter().enumerate() {
        anyhow::ensure!(
            delta.kind == source.kind
                && delta.token_start == source.token_start.0 as usize + row
                && delta.row_count == 1
                && delta.values_per_row == values_per_row
                && delta.output_device_row_offset == 0
                && delta.output_device.rows == 1
                && delta.output_bf16.is_none()
                && delta.backend == backend,
            "sequential MTP target attention row {row} has an incompatible output"
        );
    }
    let output_rows = row_deltas
        .iter()
        .map(|delta| delta.output_device.as_ref())
        .collect::<Vec<_>>();
    let output_device =
        concat_device_bf16_row_batches(&output_rows, "sequential MTP target attention rows")?;
    let checksum = row_deltas.iter().map(|delta| delta.checksum).sum::<f64>();
    Ok(Some(RealFullSchedulerDeviceAttentionDelta {
        kind: source.kind,
        token_start: source.token_start.0 as usize,
        row_count: source.row_count,
        values_per_row,
        output_bf16: None,
        output_device: Arc::new(output_device),
        output_device_row_offset: 0,
        checksum,
        backend,
    }))
}

#[allow(clippy::too_many_arguments)]
fn real_full_launch_scheduler_attention_from_device_kv(
    config: &KvCacheConfig,
    wave: &LayerWave,
    visible_blocks: &[KvBackedBlock],
    current_descriptors: &[KvBlockDescriptor],
    counters: &mut RealFullSchedulerExecutionCounters,
    device_kv: &mut RealFullDeviceKvExecutionMirror,
    numeric_progression: &mut RealFullSchedulerNumericProgression,
    catalog: &TensorCatalog,
    normalized_hidden: Option<&DeviceBf16Output>,
) -> Result<Option<RealFullSchedulerDeviceAttentionDelta>> {
    real_full_launch_scheduler_attention_from_device_kv_with_query_override(
        config,
        wave,
        visible_blocks,
        current_descriptors,
        counters,
        device_kv,
        numeric_progression,
        catalog,
        normalized_hidden,
        None,
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn real_full_launch_scheduler_attention_from_device_kv_with_query_override(
    config: &KvCacheConfig,
    wave: &LayerWave,
    visible_blocks: &[KvBackedBlock],
    current_descriptors: &[KvBlockDescriptor],
    counters: &mut RealFullSchedulerExecutionCounters,
    device_kv: &mut RealFullDeviceKvExecutionMirror,
    numeric_progression: &mut RealFullSchedulerNumericProgression,
    catalog: &TensorCatalog,
    normalized_hidden: Option<&DeviceBf16Output>,
    query_hidden_override: Option<RealFullSchedulerDeviceHiddenSource>,
    query_descriptors_override: Option<&[KvBlockDescriptor]>,
    projected_query_override: Option<SchedulerRealAttentionQueryOutputs>,
) -> Result<Option<RealFullSchedulerDeviceAttentionDelta>> {
    if !matches!(
        config.dtype,
        KvCacheDType::Bf16 | KvCacheDType::Fp8 | KvCacheDType::Nvfp4
    ) || current_descriptors.is_empty()
    {
        return Ok(None);
    }
    let stage_timing = admission_stage_timing_enabled();
    let total_start = stage_timing.then(Instant::now);
    let source = single_wave_source(wave)?;
    let hidden_source_start = stage_timing.then(Instant::now);
    let query_hidden = match query_hidden_override {
        Some(hidden) => Some(hidden),
        None => numeric_progression
            .device_hidden_source(source.kind, source.token_start.0 as usize, source.row_count)
            .with_context(|| {
                format!(
                    "resolving scheduler resident hidden query rows for layer {}",
                    wave.layer_id.0
                )
            })?,
    };
    let hidden_source_ms = elapsed_ms_optional(hidden_source_start);
    if let Some(hidden) = query_hidden {
        if hidden.rows != source.row_count || hidden.values_per_row != GLM52_HIDDEN_SIZE {
            anyhow::bail!(
                "scheduler resident hidden query shape mismatch for layer {}: expected {}x{} got {}x{}",
                wave.layer_id.0,
                source.row_count,
                GLM52_HIDDEN_SIZE,
                hidden.rows,
                hidden.values_per_row
            );
        }
    }
    let mut query_project_ms = 0.0_f64;
    let mut device_attention_ms = 0.0_f64;
    let launch = if let Some(hidden) = query_hidden {
        let attention_context_rows = visible_blocks
            .iter()
            .map(|block| &block.descriptor)
            .chain(current_descriptors.iter())
            .try_fold(0_usize, |rows, descriptor| {
                rows.checked_add(descriptor.token_count)
                    .context("scheduler attention context rows overflow usize")
            })?;
        let query_project_start = stage_timing.then(Instant::now);
        let real_attention = match projected_query_override {
            Some(projected) => Some(projected),
            None => project_real_scheduler_attention_query_outputs(
                catalog,
                wave.layer_id.0 as usize,
                hidden,
                normalized_hidden,
                attention_context_rows > REAL_FULL_DSA_TOP_K,
            )
            .with_context(|| {
                format!(
                    "projecting real scheduler MLA attention query rows for layer {}",
                    wave.layer_id.0
                )
            })?,
        };
        query_project_ms += elapsed_ms_optional(query_project_start);
        if let Some(real_attention) = real_attention {
            if env::var_os("GLMRT_REAL_FULL_DIAGNOSTIC_LAYER_DUMP_DIR").is_some()
                && wave.layer_id.0 == 0
            {
                eprintln!(
                    "real_full_attention_branch layer_id={} branch=real heads={} nope_dim={} v_dim={} scale={}",
                    wave.layer_id.0,
                    REAL_FULL_SCHEDULER_MLA_NUM_ATTENTION_HEADS,
                    REAL_FULL_SCHEDULER_MLA_QK_NOPE_HEAD_DIM,
                    REAL_FULL_SCHEDULER_MLA_V_HEAD_DIM,
                    real_full_scheduler_mla_attention_scale(),
                );
            }
            let SchedulerRealAttentionQueryOutputs {
                q_projected,
                dsa,
                kv_norm_weight,
                kv_b_weight,
                output_projection_weight_name,
            } = real_attention;
            let dsa_buffers = dsa
                .as_ref()
                .map(|dsa| (dsa.query_projected.buffer(), dsa.weights_projected.buffer()));
            let device_attention_start = stage_timing.then(Instant::now);
            let launch = device_kv
                .run_scheduler_mla_attention_from_device_kv_descriptor_sets_with_projected_query_bf16(
                    visible_blocks,
                    current_descriptors,
                    query_descriptors_override.unwrap_or(current_descriptors),
                    q_projected,
                    dsa_buffers,
                    kv_norm_weight,
                    kv_b_weight,
                    output_projection_weight_name.as_str(),
                    REAL_FULL_SCHEDULER_MLA_NUM_ATTENTION_HEADS,
                    REAL_FULL_SCHEDULER_MLA_QK_NOPE_HEAD_DIM,
                    REAL_FULL_SCHEDULER_MLA_V_HEAD_DIM,
                    REAL_FULL_DENSE_RMSNORM_EPS,
                    GLM52_MLA_ROPE_THETA,
                    real_full_scheduler_mla_attention_scale(),
                )
                .context("running real scheduler MLA attention from live device KV cache")?;
            device_attention_ms += elapsed_ms_optional(device_attention_start);
            launch
        } else {
            if env::var_os("GLMRT_REAL_FULL_DIAGNOSTIC_LAYER_DUMP_DIR").is_some()
                && wave.layer_id.0 == 0
            {
                eprintln!(
                    "real_full_attention_branch layer_id={} branch=synthetic-fallback heads={} nope_dim={} v_dim={}",
                    wave.layer_id.0,
                    2,
                    3,
                    2,
                );
            }
            let device_attention_start = stage_timing.then(Instant::now);
            let launch = device_kv
                .run_scheduler_mla_attention_from_device_kv_descriptor_sets_bf16(
                    visible_blocks,
                    current_descriptors,
                    Some(hidden.buffer),
                )
                .context("running fallback scheduler MLA attention from live device KV cache")?;
            device_attention_ms += elapsed_ms_optional(device_attention_start);
            launch
        }
    } else {
        let device_attention_start = stage_timing.then(Instant::now);
        let launch = device_kv
            .run_scheduler_mla_attention_from_device_kv_descriptor_sets_bf16(
                visible_blocks,
                current_descriptors,
                None,
            )
            .context("running scheduler MLA attention from live device KV cache")?;
        device_attention_ms += elapsed_ms_optional(device_attention_start);
        launch
    };
    if stage_timing {
        eprintln!(
            "real_full_attention_stage_timing layer_id={} rows={} descriptors={} hidden_source_ms={:.3} query_project_ms={:.3} device_attention_ms={:.3} total_ms={:.3}",
            wave.layer_id.0,
            source.row_count,
            visible_blocks.len() + current_descriptors.len(),
            hidden_source_ms,
            query_project_ms,
            device_attention_ms,
            elapsed_ms_optional(total_start)
        );
    }
    let Some(launch) = launch else {
        return Ok(None);
    };
    if source.row_count != launch.query_rows {
        anyhow::bail!(
            "scheduler device attention query rows {} do not match wave source rows {} for layer {}",
            launch.query_rows,
            source.row_count,
            wave.layer_id.0
        );
    }
    counters.device_attention_launches += 1;
    counters.device_attention_status = Some(launch.status);
    counters.device_attention_rows += launch.rows;
    counters.device_attention_query_rows += launch.query_rows;
    counters.device_attention_kv_descriptors += launch.descriptors;
    counters.device_attention_output_bytes += launch.output_bytes;
    counters.device_attention_output_values += launch.output_values;
    counters.device_attention_output_finite_values += launch.output_finite_values;
    counters.device_attention_output_nonzero_values += launch.output_nonzero_values;
    counters.device_attention_output_checksum += launch.output_checksum;
    counters.device_attention_hidden_projection_launches +=
        usize::from(launch.output_projected_to_hidden);
    let output_values_per_row = launch
        .output_values
        .checked_div(launch.output_rows)
        .filter(|values| *values > 0 && launch.output_values % launch.output_rows == 0)
        .context("scheduler device attention output values are not divisible by output rows")?;
    let suffix_row_offset = launch.output_row_offset;
    let suffix_row_end = suffix_row_offset
        .checked_add(launch.query_rows)
        .context("scheduler device attention suffix row end overflow")?;
    if suffix_row_end > launch.output_rows {
        anyhow::bail!(
            "scheduler device attention suffix rows {suffix_row_offset}..{suffix_row_end} exceed output rows {}",
            launch.output_rows
        );
    }
    let suffix_byte_start = suffix_row_offset
        .checked_mul(output_values_per_row)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("scheduler device attention suffix byte offset overflow")?;
    let suffix_bytes = launch
        .query_rows
        .checked_mul(output_values_per_row)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("scheduler device attention suffix byte count overflow")?;
    let suffix_byte_end = suffix_byte_start
        .checked_add(suffix_bytes)
        .context("scheduler device attention suffix byte end overflow")?;
    if suffix_byte_end > launch.output_bytes {
        anyhow::bail!(
            "scheduler device attention suffix range {suffix_byte_start}..{suffix_byte_end} exceeds output bytes {}",
            launch.output_bytes
        );
    }
    let (output_bf16, suffix_checksum) = if output_values_per_row == GLM52_HIDDEN_SIZE {
        if suffix_row_offset != 0
            || launch.output_rows != launch.query_rows
            || suffix_byte_start != 0
            || suffix_byte_end != launch.output_bytes
        {
            anyhow::bail!(
                "scheduler hidden-width attention delta expected direct suffix output rows={} query_rows={} row_offset={} byte_range={}..{} output_bytes={}",
                launch.output_rows,
                launch.query_rows,
                suffix_row_offset,
                suffix_byte_start,
                suffix_byte_end,
                launch.output_bytes
            );
        }
        (None, 0.0)
    } else {
        let output_host_bf16 = launch
            .output_bf16
            .as_ref()
            .context("scheduler compact attention suffix requires host output bytes")?;
        if suffix_byte_end > output_host_bf16.len() {
            anyhow::bail!(
                "scheduler compact attention suffix range {suffix_byte_start}..{suffix_byte_end} exceeds host output bytes {}",
                output_host_bf16.len()
            );
        }
        let output_bf16 = output_host_bf16[suffix_byte_start..suffix_byte_end].to_vec();
        let suffix_values = bf16_bytes_to_f32(&output_bf16)
            .context("decoding scheduler device attention suffix output")?;
        let suffix_checksum = suffix_values.iter().map(|value| *value as f64).sum::<f64>();
        (Some(output_bf16), suffix_checksum)
    };
    if !suffix_checksum.is_finite() {
        anyhow::bail!("scheduler device attention suffix checksum is non-finite");
    }
    Ok(Some(RealFullSchedulerDeviceAttentionDelta {
        kind: source.kind,
        token_start: source.token_start.0 as usize,
        row_count: source.row_count,
        values_per_row: output_values_per_row,
        output_bf16,
        output_device: Arc::new(launch.output_device),
        output_device_row_offset: suffix_row_offset,
        checksum: suffix_checksum,
        backend: launch.status,
    }))
}

fn single_wave_source(wave: &LayerWave) -> Result<&glmrt_core::RowSource> {
    if wave.row_sources.len() != 1 {
        anyhow::bail!(
            "scheduler device attention delta expected exactly one row source for layer {}, got {}",
            wave.layer_id.0,
            wave.row_sources.len()
        );
    }
    Ok(&wave.row_sources[0])
}

fn real_full_kv_payloads_for_descriptors(
    config: &KvCacheConfig,
    descriptors: &[KvBlockDescriptor],
    salt: u8,
) -> Vec<Vec<u8>> {
    descriptors
        .iter()
        .map(|descriptor| {
            let payload_bytes =
                config.layer_payload_bytes(descriptor.layer_id, descriptor.token_count);
            if config.dtype != KvCacheDType::Bf16 {
                let byte = salt ^ descriptor.layer_id.0 as u8 ^ descriptor.token_start.0 as u8;
                return vec![byte; payload_bytes];
            }
            debug_assert_eq!(payload_bytes % std::mem::size_of::<u16>(), 0);
            let values_per_token =
                config.layer_bytes_per_token(descriptor.layer_id) / std::mem::size_of::<u16>();
            debug_assert!(values_per_token > 0);
            let mut payload = Vec::with_capacity(payload_bytes);
            for index in 0..payload_bytes / std::mem::size_of::<u16>() {
                let token_offset = (index / values_per_token) as u64;
                let token = descriptor.token_start.0 + token_offset;
                let keyed = (index as u64)
                    .wrapping_add(token.wrapping_mul(17))
                    .wrapping_add((descriptor.layer_id.0 as u64).wrapping_mul(31))
                    .wrapping_add((salt as u64).wrapping_mul(43));
                let value = ((keyed % 97) as f32 - 48.0) / 256.0;
                let bf16 = (value.to_bits() >> 16) as u16;
                payload.extend_from_slice(&bf16.to_le_bytes());
            }
            debug_assert_eq!(payload.len(), payload_bytes);
            payload
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glmrt_core::{LayerId, PositionId};

    #[test]
    fn scheduler_kv_payload_generation_writes_bf16_directly() {
        let config = KvCacheConfig::glm52_phase0(8);
        let descriptor = KvBlockDescriptor {
            reservation_id: 7,
            sequence_id: "seq".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(2),
            token_count: 2,
        };
        let payloads =
            real_full_kv_payloads_for_descriptors(&config, std::slice::from_ref(&descriptor), 0x40);

        assert_eq!(payloads.len(), 1);
        let payload = &payloads[0];
        assert_eq!(
            payload.len(),
            config.layer_payload_bytes(descriptor.layer_id, descriptor.token_count)
        );

        let values_per_token =
            config.layer_bytes_per_token(descriptor.layer_id) / std::mem::size_of::<u16>();
        for index in [
            0,
            1,
            values_per_token - 1,
            values_per_token,
            payload.len() / std::mem::size_of::<u16>() - 1,
        ] {
            let actual = u16::from_le_bytes([payload[index * 2], payload[index * 2 + 1]]);
            let token_offset = (index / values_per_token) as u64;
            let token = descriptor.token_start.0 + token_offset;
            let keyed = (index as u64)
                .wrapping_add(token.wrapping_mul(17))
                .wrapping_add((descriptor.layer_id.0 as u64).wrapping_mul(31))
                .wrapping_add((0x40_u64).wrapping_mul(43));
            let expected = (((keyed % 97) as f32 - 48.0) / 256.0).to_bits() >> 16;
            assert_eq!(actual, expected as u16, "bf16 mismatch at value {index}");
        }
    }

    #[test]
    fn scheduler_kv_payload_generation_preserves_packed_fill_for_non_bf16() {
        let config = KvCacheConfig::glm52_compressed_nvfp4(8);
        let descriptor = KvBlockDescriptor {
            reservation_id: 7,
            sequence_id: "seq".to_owned(),
            layer_id: LayerId(5),
            token_start: PositionId(4),
            token_count: 3,
        };
        let payloads =
            real_full_kv_payloads_for_descriptors(&config, std::slice::from_ref(&descriptor), 0x80);

        assert_eq!(payloads.len(), 1);
        let payload = &payloads[0];
        assert_eq!(
            payload.len(),
            config.layer_payload_bytes(descriptor.layer_id, descriptor.token_count)
        );
        let expected = 0x80 ^ descriptor.layer_id.0 as u8 ^ descriptor.token_start.0 as u8;
        assert!(payload.iter().all(|byte| *byte == expected));
    }
}
