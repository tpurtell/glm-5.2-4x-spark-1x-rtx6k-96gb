use anyhow::{Context, Result};
use glmrt_core::{
    KvBlockDescriptor, KvCacheBackingStore, KvCacheConfig, LayerId, PositionId,
    GLM52_NUM_HIDDEN_LAYERS,
};

use super::constants::{
    REAL_FULL_PREFLIGHT_DECODE_POSITION, REAL_FULL_PREFLIGHT_DECODE_ROWS,
    REAL_FULL_PREFLIGHT_MTP_ACCEPTED_ROWS, REAL_FULL_PREFLIGHT_MTP_ROWS,
    REAL_FULL_PREFLIGHT_MTP_TOKEN_START, REAL_FULL_PREFLIGHT_PREFILL_ROWS,
    REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START, REAL_FULL_PREFLIGHT_SEQUENCE_ID,
};
use super::types::RealFullKvBackingStoreDryRun;

mod attention_io;
pub(in crate::commands::real_full) mod device;

pub(super) use attention_io::real_full_attention_kv_io_dry_run;
use device::real_full_device_kv_block_ios;

pub(super) fn real_full_kv_backing_store_dry_run(
    kv_config: KvCacheConfig,
) -> Result<RealFullKvBackingStoreDryRun> {
    let reservation_tokens =
        REAL_FULL_PREFLIGHT_MTP_TOKEN_START as usize + REAL_FULL_PREFLIGHT_MTP_ROWS + 1;
    let mut store = KvCacheBackingStore::new(KvCacheConfig {
        max_tokens: reservation_tokens,
        ..kv_config
    });
    let reservation_id = store.reserve(REAL_FULL_PREFLIGHT_SEQUENCE_ID, reservation_tokens)?;
    let mut all_layer_prefill_backed_bytes = 0_usize;

    for layer_id in 0..GLM52_NUM_HIDDEN_LAYERS {
        let layer_id = LayerId(layer_id as u32);
        let layer_bytes = store.config().layer_bytes_per_token(layer_id);

        let prefill_payload =
            vec![layer_id.0 as u8; layer_bytes * REAL_FULL_PREFLIGHT_PREFILL_ROWS];
        all_layer_prefill_backed_bytes += prefill_payload.len();
        let prefill_descriptor = KvBlockDescriptor {
            reservation_id,
            sequence_id: REAL_FULL_PREFLIGHT_SEQUENCE_ID.to_owned(),
            layer_id,
            token_start: PositionId(REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START),
            token_count: REAL_FULL_PREFLIGHT_PREFILL_ROWS,
        };
        let decode_descriptor = KvBlockDescriptor {
            reservation_id,
            sequence_id: REAL_FULL_PREFLIGHT_SEQUENCE_ID.to_owned(),
            layer_id,
            token_start: PositionId(REAL_FULL_PREFLIGHT_DECODE_POSITION),
            token_count: REAL_FULL_PREFLIGHT_DECODE_ROWS,
        };
        let tentative_descriptors = (0..REAL_FULL_PREFLIGHT_MTP_ROWS)
            .map(|draft_offset| KvBlockDescriptor {
                reservation_id,
                sequence_id: REAL_FULL_PREFLIGHT_SEQUENCE_ID.to_owned(),
                layer_id,
                token_start: PositionId(REAL_FULL_PREFLIGHT_MTP_TOKEN_START + draft_offset as u64),
                token_count: 1,
            })
            .collect::<Vec<_>>();
        let mut device_descriptors = Vec::with_capacity(2 + tentative_descriptors.len());
        device_descriptors.push(prefill_descriptor.clone());
        device_descriptors.push(decode_descriptor.clone());
        device_descriptors.extend(tentative_descriptors.iter().cloned());
        real_full_device_kv_block_ios(store.config(), &device_descriptors).with_context(|| {
            format!("planning batched device KV blocks for layer {}", layer_id.0)
        })?;

        store
            .write_committed_block(prefill_descriptor, prefill_payload)
            .with_context(|| {
                format!("writing prefill KV backing block for layer {}", layer_id.0)
            })?;
        store
            .write_committed_block(
                decode_descriptor,
                vec![layer_id.0.wrapping_add(1) as u8; layer_bytes],
            )
            .with_context(|| format!("writing decode KV backing block for layer {}", layer_id.0))?;

        for (draft_offset, tentative_descriptor) in tentative_descriptors.into_iter().enumerate() {
            store
                .write_tentative_block(tentative_descriptor, vec![draft_offset as u8; layer_bytes])
                .with_context(|| {
                    format!(
                        "writing MTP KV backing block for layer {} draft {draft_offset}",
                        layer_id.0
                    )
                })?;
        }
        store
            .resolve_mtp_tentative_writes(
                reservation_id,
                layer_id,
                PositionId(REAL_FULL_PREFLIGHT_MTP_TOKEN_START),
                REAL_FULL_PREFLIGHT_MTP_ROWS,
                REAL_FULL_PREFLIGHT_MTP_ACCEPTED_ROWS,
            )
            .with_context(|| format!("resolving MTP KV backing blocks for layer {}", layer_id.0))?;
    }

    let visible_layer0_at_decode = store.read_visible_blocks_for_decode(
        reservation_id,
        LayerId(0),
        PositionId(REAL_FULL_PREFLIGHT_DECODE_POSITION),
    );
    let visible_layer0_after_mtp = store.read_visible_blocks_for_decode(
        reservation_id,
        LayerId(0),
        PositionId(REAL_FULL_PREFLIGHT_MTP_TOKEN_START + REAL_FULL_PREFLIGHT_MTP_ROWS as u64),
    );
    let snapshot = store.snapshot();

    Ok(RealFullKvBackingStoreDryRun {
        status: "backing-store-dry-run",
        layout: store.config().layout_label(),
        reservation_tokens,
        bytes_per_model_token: store.config().bytes_per_token(),
        capacity_bytes: store.config().capacity_bytes(),
        dsa_layer_bytes_per_token: store.config().layer_bytes_per_token(LayerId(0)),
        non_dsa_layer_bytes_per_token: store.config().layer_bytes_per_token(LayerId(21)),
        layer_count: GLM52_NUM_HIDDEN_LAYERS,
        backed_prefill_writes: GLM52_NUM_HIDDEN_LAYERS,
        backed_decode_writes: GLM52_NUM_HIDDEN_LAYERS,
        backed_tentative_mtp_writes: GLM52_NUM_HIDDEN_LAYERS * REAL_FULL_PREFLIGHT_MTP_ROWS,
        committed_mtp_writes: snapshot.committed_writes,
        discarded_mtp_writes: snapshot.discarded_writes,
        backed_write_count_after_discard: store.backed_write_count(),
        backed_bytes_after_discard: store.backed_write_bytes(),
        all_layer_prefill_backed_bytes,
        visible_layer0_blocks_at_decode: visible_layer0_at_decode.len(),
        visible_layer0_bytes_at_decode: visible_layer0_at_decode
            .iter()
            .map(|block| block.bytes.len())
            .sum(),
        visible_layer0_blocks_after_mtp_commit: visible_layer0_after_mtp.len(),
    })
}
