use anyhow::{Context, Result};
use glmrt_core::{
    DecodeStep, GraphBucket, KvCacheBackingStore, KvCacheConfig, LayerId, LayerWave,
    MtpVerifyBlock, PositionId, PrefillChunk, Priority, GLM52_NUM_HIDDEN_LAYERS,
};

use super::super::constants::{
    REAL_FULL_PREFLIGHT_DECODE_POSITION, REAL_FULL_PREFLIGHT_MTP_ACCEPTED_ROWS,
    REAL_FULL_PREFLIGHT_MTP_ROWS, REAL_FULL_PREFLIGHT_MTP_TOKEN_START,
    REAL_FULL_PREFLIGHT_PREFILL_ROWS, REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START,
    REAL_FULL_PREFLIGHT_REQUEST_ID, REAL_FULL_PREFLIGHT_SEQUENCE_ID,
};
use super::super::types::RealFullAttentionKvIoDryRun;
use super::device::RealFullDeviceKvExecutionMirror;

pub(in crate::commands::real_full) fn real_full_attention_kv_io_dry_run(
    kv_config: KvCacheConfig,
) -> Result<RealFullAttentionKvIoDryRun> {
    let reservation_tokens =
        REAL_FULL_PREFLIGHT_MTP_TOKEN_START as usize + REAL_FULL_PREFLIGHT_MTP_ROWS + 1;
    let attention_kv_config = KvCacheConfig {
        max_tokens: reservation_tokens,
        ..kv_config
    };
    let mut store = KvCacheBackingStore::new(attention_kv_config.clone());
    let reservation_id = store.reserve(REAL_FULL_PREFLIGHT_SEQUENCE_ID, reservation_tokens)?;
    let placement_version = "real-full-attention-kv-io";
    let mut prefix_prefill_wave_writes = 0_usize;
    let mut later_prefill_prefix_read_blocks = 0_usize;
    let mut later_prefill_wave_writes = 0_usize;
    let mut decode_prefix_read_blocks = 0_usize;
    let mut decode_wave_writes = 0_usize;
    let mut mtp_prefix_read_blocks = 0_usize;
    let mut mtp_tentative_wave_writes = 0_usize;
    let mut layer0_decode_read_blocks = 0_usize;
    let mut layer0_mtp_read_blocks = 0_usize;
    let mut device_kv = RealFullDeviceKvExecutionMirror::new(attention_kv_config.clone())?;

    for layer_id in 0..GLM52_NUM_HIDDEN_LAYERS {
        let layer_id = LayerId(layer_id as u32);
        let layer_bytes = store.config().layer_bytes_per_token(layer_id);
        let prefix_prefill = LayerWave::prefill(PrefillChunk::new(
            REAL_FULL_PREFLIGHT_REQUEST_ID,
            REAL_FULL_PREFLIGHT_SEQUENCE_ID,
            layer_id.0,
            PositionId(0),
            REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START as usize,
            reservation_id,
            Priority(1),
            GraphBucket::new(REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START as usize),
            placement_version,
        ));
        let prefix_prefill_payloads = vec![vec![
            layer_id.0 as u8;
            layer_bytes
                * REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START
                    as usize
        ]];
        device_kv
            .write_host_blocks(&prefix_prefill.kv_writes, &prefix_prefill_payloads)
            .with_context(|| {
                format!(
                    "writing LayerWave prefix device KV block for attention layer {}",
                    layer_id.0
                )
            })?;
        prefix_prefill_wave_writes += store
            .write_committed_blocks_for_wave(&prefix_prefill, prefix_prefill_payloads)
            .with_context(|| {
                format!(
                    "writing LayerWave prefix KV block for attention layer {}",
                    layer_id.0
                )
            })?
            .len();

        let later_prefill = LayerWave::prefill(PrefillChunk::new(
            REAL_FULL_PREFLIGHT_REQUEST_ID,
            REAL_FULL_PREFLIGHT_SEQUENCE_ID,
            layer_id.0,
            PositionId(REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START),
            REAL_FULL_PREFLIGHT_PREFILL_ROWS,
            reservation_id,
            Priority(1),
            GraphBucket::new(REAL_FULL_PREFLIGHT_PREFILL_ROWS),
            placement_version,
        ));
        let later_prefill_reads = store.read_visible_blocks_for_wave(&later_prefill);
        later_prefill_prefix_read_blocks += later_prefill_reads.len();
        device_kv
            .read_visible_blocks(&later_prefill_reads)
            .with_context(|| {
                format!(
                    "reading LayerWave later-prefill device KV blocks for attention layer {}",
                    layer_id.0
                )
            })?;
        let later_prefill_payloads = vec![vec![
            layer_id.0.wrapping_add(1) as u8;
            layer_bytes * REAL_FULL_PREFLIGHT_PREFILL_ROWS
        ]];
        device_kv
            .write_host_blocks(&later_prefill.kv_writes, &later_prefill_payloads)
            .with_context(|| {
                format!(
                    "writing LayerWave later-prefill device KV block for attention layer {}",
                    layer_id.0
                )
            })?;
        later_prefill_wave_writes += store
            .write_committed_blocks_for_wave(&later_prefill, later_prefill_payloads)
            .with_context(|| {
                format!(
                    "writing LayerWave later-prefill KV block for attention layer {}",
                    layer_id.0
                )
            })?
            .len();

        let decode = LayerWave::decode(DecodeStep::new(
            REAL_FULL_PREFLIGHT_REQUEST_ID,
            REAL_FULL_PREFLIGHT_SEQUENCE_ID,
            layer_id.0,
            PositionId(REAL_FULL_PREFLIGHT_DECODE_POSITION),
            Some(reservation_id),
            Priority(0),
            placement_version,
        ));
        let decode_reads = store.read_visible_blocks_for_wave(&decode);
        if layer_id.0 == 0 {
            layer0_decode_read_blocks = decode_reads.len();
        }
        decode_prefix_read_blocks += decode_reads.len();
        device_kv
            .read_visible_blocks(&decode_reads)
            .with_context(|| {
                format!(
                    "reading LayerWave decode device KV blocks for attention layer {}",
                    layer_id.0
                )
            })?;
        let decode_payloads = vec![vec![3_u8; layer_bytes]];
        device_kv
            .write_host_blocks(&decode.kv_writes, &decode_payloads)
            .with_context(|| {
                format!(
                    "writing LayerWave decode device KV block for attention layer {}",
                    layer_id.0
                )
            })?;
        decode_wave_writes += store
            .write_committed_blocks_for_wave(&decode, decode_payloads)
            .with_context(|| {
                format!(
                    "writing LayerWave decode KV block for attention layer {}",
                    layer_id.0
                )
            })?
            .len();

        let mtp = LayerWave::mtp_verify(MtpVerifyBlock::new(
            REAL_FULL_PREFLIGHT_REQUEST_ID,
            REAL_FULL_PREFLIGHT_SEQUENCE_ID,
            layer_id.0,
            PositionId(REAL_FULL_PREFLIGHT_MTP_TOKEN_START),
            REAL_FULL_PREFLIGHT_MTP_ROWS,
            Some(reservation_id),
            Priority(0),
            GraphBucket::new(REAL_FULL_PREFLIGHT_MTP_ROWS),
            placement_version,
        ));
        let mtp_reads = store.read_visible_blocks_for_wave(&mtp);
        if layer_id.0 == 0 {
            layer0_mtp_read_blocks = mtp_reads.len();
        }
        mtp_prefix_read_blocks += mtp_reads.len();
        device_kv.read_visible_blocks(&mtp_reads).with_context(|| {
            format!(
                "reading LayerWave MTP device KV blocks for attention layer {}",
                layer_id.0
            )
        })?;
        let mtp_payloads = (0..REAL_FULL_PREFLIGHT_MTP_ROWS)
            .map(|offset| vec![offset as u8; layer_bytes])
            .collect::<Vec<_>>();
        device_kv
            .write_host_blocks(&mtp.tentative_kv_writes, &mtp_payloads)
            .with_context(|| {
                format!(
                    "writing LayerWave tentative MTP device KV blocks for attention layer {}",
                    layer_id.0
                )
            })?;
        mtp_tentative_wave_writes += store
            .write_tentative_blocks_for_wave(&mtp, mtp_payloads)
            .with_context(|| {
                format!(
                    "writing LayerWave tentative MTP KV blocks for attention layer {}",
                    layer_id.0
                )
            })?
            .len();
        store
            .resolve_mtp_tentative_writes(
                reservation_id,
                layer_id,
                PositionId(REAL_FULL_PREFLIGHT_MTP_TOKEN_START),
                REAL_FULL_PREFLIGHT_MTP_ROWS,
                REAL_FULL_PREFLIGHT_MTP_ACCEPTED_ROWS,
            )
            .with_context(|| {
                format!(
                    "resolving LayerWave tentative MTP KV blocks for attention layer {}",
                    layer_id.0
                )
            })?;
    }

    let guard_store = KvCacheBackingStore::new(KvCacheConfig::glm52_phase0(1));
    let guard_wave = LayerWave::prefill(PrefillChunk::new(
        "guard",
        "guard-seq",
        0,
        0,
        1,
        1,
        Priority(0),
        GraphBucket::new(1),
        "guard-placement",
    ));
    let layerwave_payload_count_mismatch_guard = {
        let mut guard_store = guard_store;
        guard_store
            .write_committed_blocks_for_wave(&guard_wave, Vec::new())
            .is_err()
    };
    let snapshot = store.snapshot();
    let device_kv = device_kv.summary();

    Ok(RealFullAttentionKvIoDryRun {
        status: "layerwave-kv-io-dry-run",
        layer_count: GLM52_NUM_HIDDEN_LAYERS,
        prefix_prefill_wave_writes,
        later_prefill_prefix_read_blocks,
        later_prefill_wave_writes,
        decode_prefix_read_blocks,
        decode_wave_writes,
        mtp_prefix_read_blocks,
        mtp_tentative_wave_writes,
        mtp_committed_writes: snapshot.committed_writes,
        mtp_discarded_writes: snapshot.discarded_writes,
        layerwave_payload_count_mismatch_guard,
        layer0_decode_read_blocks,
        layer0_mtp_read_blocks,
        backed_bytes_after_discard: store.backed_write_bytes(),
        device_kv_status: device_kv.status,
        device_kv_writes: device_kv.writes,
        device_kv_reads: device_kv.reads,
        device_kv_bytes: device_kv.bytes,
        uses_device_kv_cache: device_kv.uses_device_kv_cache,
    })
}
