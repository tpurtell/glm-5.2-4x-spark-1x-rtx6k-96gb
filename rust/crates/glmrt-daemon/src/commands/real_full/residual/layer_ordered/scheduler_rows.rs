use anyhow::{Context, Result};
use glmrt_core::{
    admit_layerwaves_for_iteration, GraphBucket, KvCacheBackingStore, KvCacheConfig, LayerId,
    LayerWave, LayerWaveMode, PositionId, PrefillChunk, PrefillChunkPolicy, Priority,
    GLM52_NUM_HIDDEN_LAYERS,
};

use crate::commands::real_full::constants::{
    REAL_FULL_PREFLIGHT_REQUEST_ID, REAL_FULL_PREFLIGHT_SEQUENCE_ID,
};
use crate::commands::real_full::kv::device::{
    RealFullDeviceKvExecutionMirror, RealFullDeviceKvExecutionSummary,
};
use crate::commands::real_full::types::RealFullLayerOrderedSchedulerRowsProbe;

const LAYER_ORDERED_SCHEDULER_PLACEMENT: &str =
    "real-full-layer-ordered-residual-scheduler-binding";

pub(super) struct LayerOrderedSchedulerRowsBinding {
    pub(super) probe: RealFullLayerOrderedSchedulerRowsProbe,
    selected_layers: Vec<usize>,
}

impl LayerOrderedSchedulerRowsBinding {
    pub(super) fn layer_selected(&self, layer_id: usize) -> bool {
        self.selected_layers.binary_search(&layer_id).is_ok()
    }

    pub(super) fn covers_all_layers(&self) -> bool {
        self.selected_layers.len() == GLM52_NUM_HIDDEN_LAYERS
            && self
                .selected_layers
                .iter()
                .copied()
                .eq(0..GLM52_NUM_HIDDEN_LAYERS)
    }
}

pub(super) fn layer_ordered_scheduler_rows_binding() -> LayerOrderedSchedulerRowsBinding {
    match run_layer_ordered_scheduler_rows_probe() {
        Ok(binding) => binding,
        Err(error) => {
            skipped_layer_ordered_scheduler_rows_binding("error", Some(error.to_string()))
        }
    }
}

fn run_layer_ordered_scheduler_rows_probe() -> Result<LayerOrderedSchedulerRowsBinding> {
    let kv_config = KvCacheConfig::glm52_phase0(2);
    let expected_backed_kv_bytes = kv_config.bytes_per_token() * 2;
    let mut store = KvCacheBackingStore::new(kv_config.clone());
    let reservation_id = store.reserve(REAL_FULL_PREFLIGHT_SEQUENCE_ID, 2)?;
    let policy = PrefillChunkPolicy {
        chunk_tokens: 1,
        max_prefill_tokens_per_iteration: 1,
        max_active_prefill_chunks: 1,
        decode_priority: true,
    };

    let mut selected_layerwaves = 0_usize;
    let mut selected_rows = 0_usize;
    let mut row_sources = 0_usize;
    let mut selected_decode_rows = 0_usize;
    let mut selected_prefill_rows = 0_usize;
    let mut selected_mtp_rows = 0_usize;
    let mut deferred_layerwaves = 0_usize;
    let mut kv_read_blocks = 0_usize;
    let mut committed_kv_writes = 0_usize;
    let mut layer_order_verified = true;
    let mut selected_layers = Vec::with_capacity(GLM52_NUM_HIDDEN_LAYERS);
    let mut device_kv = RealFullDeviceKvExecutionMirror::new(kv_config.clone())?;

    for layer_id in 0..GLM52_NUM_HIDDEN_LAYERS {
        let layer = LayerId(layer_id as u32);
        let prefix_wave = LayerWave::prefill(PrefillChunk::new(
            REAL_FULL_PREFLIGHT_REQUEST_ID,
            REAL_FULL_PREFLIGHT_SEQUENCE_ID,
            layer,
            PositionId(0),
            1,
            reservation_id,
            Priority(0),
            GraphBucket::new(1),
            LAYER_ORDERED_SCHEDULER_PLACEMENT,
        ));
        let prefix_payloads = kv_payloads_for_wave(&store, &prefix_wave, layer_id)?;
        device_kv
            .write_host_blocks(&prefix_wave.kv_writes, &prefix_payloads)
            .with_context(|| {
                format!(
                    "writing layer-ordered scheduler prefix device KV block for layer {}",
                    layer.0
                )
            })?;
        committed_kv_writes += store
            .write_committed_blocks_for_wave(&prefix_wave, prefix_payloads)
            .with_context(|| {
                format!(
                    "writing layer-ordered scheduler prefix KV block for layer {}",
                    layer.0
                )
            })?
            .len();

        let wave = LayerWave::prefill(PrefillChunk::new(
            REAL_FULL_PREFLIGHT_REQUEST_ID,
            REAL_FULL_PREFLIGHT_SEQUENCE_ID,
            layer,
            PositionId(1),
            1,
            reservation_id,
            Priority(0),
            GraphBucket::new(1),
            LAYER_ORDERED_SCHEDULER_PLACEMENT,
        ));
        let admission = admit_layerwaves_for_iteration(vec![wave], &policy);
        deferred_layerwaves += admission.deferred.len();
        selected_decode_rows += admission.selected_decode_rows;
        selected_prefill_rows += admission.selected_prefill_rows;
        selected_mtp_rows += admission.selected_mtp_rows;

        layer_order_verified &= admission.selected.len() == 1
            && admission.selected.first().is_some_and(|selected| {
                selected.layer_id == layer && selected.mode == LayerWaveMode::Prefill
            });

        for selected in &admission.selected {
            selected_layerwaves += 1;
            selected_rows += selected.num_rows();
            row_sources += selected.row_sources.len();
            selected_layers.push(selected.layer_id.0 as usize);
            let visible_kv_reads = store.read_visible_blocks_for_wave(selected);
            kv_read_blocks += visible_kv_reads.len();
            device_kv
                .read_visible_blocks(&visible_kv_reads)
                .with_context(|| {
                    format!(
                        "reading layer-ordered scheduler device KV block for layer {}",
                        selected.layer_id.0
                    )
                })?;
            let payloads = kv_payloads_for_wave(&store, selected, layer_id)?;
            device_kv
                .write_host_blocks(&selected.kv_writes, &payloads)
                .with_context(|| {
                    format!(
                        "writing layer-ordered scheduler selected device KV block for layer {}",
                        selected.layer_id.0
                    )
                })?;
            committed_kv_writes += store
                .write_committed_blocks_for_wave(selected, payloads)
                .with_context(|| {
                    format!(
                        "writing layer-ordered scheduler KV block for layer {}",
                        selected.layer_id.0
                    )
                })?
                .len();
        }
    }

    let backed_kv_bytes = store.backed_write_bytes();
    let device_kv = scheduler_rows_device_kv_io(&device_kv);
    let uses_live_scheduler_rows = selected_layerwaves == GLM52_NUM_HIDDEN_LAYERS
        && selected_rows == GLM52_NUM_HIDDEN_LAYERS
        && row_sources == GLM52_NUM_HIDDEN_LAYERS
        && selected_prefill_rows == GLM52_NUM_HIDDEN_LAYERS
        && selected_decode_rows == 0
        && selected_mtp_rows == 0
        && deferred_layerwaves == 0
        && kv_read_blocks == GLM52_NUM_HIDDEN_LAYERS
        && committed_kv_writes == GLM52_NUM_HIDDEN_LAYERS * 2
        && backed_kv_bytes == expected_backed_kv_bytes
        && layer_order_verified;

    Ok(LayerOrderedSchedulerRowsBinding {
        probe: RealFullLayerOrderedSchedulerRowsProbe {
            status: "layerwave-admitted-later-prefill-row-kv-binding",
            scope: "seed one prefix KV block per layer, admit one later-prefill LayerWave row per layer, and bind the layer-ordered residual execution trace to selected scheduler row sources with compressed KV read/write accounting",
            source_mode: "prefill",
            layer_count: GLM52_NUM_HIDDEN_LAYERS,
            selected_layerwaves,
            selected_rows,
            row_sources,
            selected_decode_rows,
            selected_prefill_rows,
            selected_mtp_rows,
            deferred_layerwaves,
            kv_read_blocks,
            committed_kv_writes,
            backed_kv_bytes,
            device_kv_status: device_kv.status,
            device_kv_writes: device_kv.writes,
            device_kv_reads: device_kv.reads,
            device_kv_bytes: device_kv.bytes,
            uses_device_kv_cache: device_kv.uses_device_kv_cache,
            layer_order_verified,
            uses_live_scheduler_rows,
            passed: uses_live_scheduler_rows,
            skipped_reason: None,
        },
        selected_layers,
    })
}

fn skipped_layer_ordered_scheduler_rows_binding(
    status: &'static str,
    skipped_reason: Option<String>,
) -> LayerOrderedSchedulerRowsBinding {
    LayerOrderedSchedulerRowsBinding {
        probe: RealFullLayerOrderedSchedulerRowsProbe {
            status,
            scope: "seed one prefix KV block per layer, admit one later-prefill LayerWave row per layer, and bind the layer-ordered residual execution trace to selected scheduler row sources with compressed KV read/write accounting",
            source_mode: "prefill",
            layer_count: GLM52_NUM_HIDDEN_LAYERS,
            selected_layerwaves: 0,
            selected_rows: 0,
            row_sources: 0,
            selected_decode_rows: 0,
            selected_prefill_rows: 0,
            selected_mtp_rows: 0,
            deferred_layerwaves: 0,
            kv_read_blocks: 0,
            committed_kv_writes: 0,
            backed_kv_bytes: 0,
            device_kv_status: "not-run",
            device_kv_writes: 0,
            device_kv_reads: 0,
            device_kv_bytes: 0,
            uses_device_kv_cache: false,
            layer_order_verified: false,
            uses_live_scheduler_rows: false,
            passed: false,
            skipped_reason,
        },
        selected_layers: Vec::new(),
    }
}

fn scheduler_rows_device_kv_io(
    device_kv: &RealFullDeviceKvExecutionMirror,
) -> RealFullDeviceKvExecutionSummary {
    device_kv.summary()
}

fn kv_payloads_for_wave(
    store: &KvCacheBackingStore,
    wave: &LayerWave,
    salt: usize,
) -> Result<Vec<Vec<u8>>> {
    wave.kv_writes
        .iter()
        .map(|descriptor| {
            let expected = store
                .config()
                .layer_payload_bytes(descriptor.layer_id, descriptor.token_count);
            let byte = (salt as u8) ^ descriptor.layer_id.0 as u8 ^ descriptor.token_start.0 as u8;
            Ok(vec![byte; expected])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::commands::real_full::coordinator_kernels::coordinator_cuda_reference_kernels_enabled;
    use glmrt_core::GLM52_NUM_HIDDEN_LAYERS;

    #[test]
    fn layer_ordered_scheduler_rows_bind_all_layers_to_prefill_rows() {
        let probe = super::layer_ordered_scheduler_rows_binding().probe;

        assert_eq!(
            probe.status,
            "layerwave-admitted-later-prefill-row-kv-binding"
        );
        assert_eq!(probe.source_mode, "prefill");
        assert_eq!(probe.layer_count, GLM52_NUM_HIDDEN_LAYERS);
        assert_eq!(probe.selected_layerwaves, GLM52_NUM_HIDDEN_LAYERS);
        assert_eq!(probe.selected_rows, GLM52_NUM_HIDDEN_LAYERS);
        assert_eq!(probe.row_sources, GLM52_NUM_HIDDEN_LAYERS);
        assert_eq!(probe.selected_decode_rows, 0);
        assert_eq!(probe.selected_prefill_rows, GLM52_NUM_HIDDEN_LAYERS);
        assert_eq!(probe.selected_mtp_rows, 0);
        assert_eq!(probe.deferred_layerwaves, 0);
        assert_eq!(probe.kv_read_blocks, GLM52_NUM_HIDDEN_LAYERS);
        assert_eq!(probe.committed_kv_writes, GLM52_NUM_HIDDEN_LAYERS * 2);
        assert_eq!(probe.backed_kv_bytes, 95_232 * 2);
        assert_device_kv_status(probe.device_kv_status, probe.uses_device_kv_cache);
        if probe.uses_device_kv_cache {
            assert_eq!(probe.device_kv_status, "cuda-kv-cache-live-scheduler");
            assert_eq!(probe.device_kv_writes, GLM52_NUM_HIDDEN_LAYERS * 2);
            assert_eq!(probe.device_kv_reads, GLM52_NUM_HIDDEN_LAYERS);
            assert_eq!(
                probe.device_kv_bytes,
                probe.backed_kv_bytes + probe.backed_kv_bytes / 2
            );
        } else {
            assert_eq!(probe.device_kv_writes, 0);
            assert_eq!(probe.device_kv_reads, 0);
            assert_eq!(probe.device_kv_bytes, 0);
        }
        assert!(probe.layer_order_verified);
        assert!(probe.uses_live_scheduler_rows);
        assert!(probe.passed);
        assert!(probe.skipped_reason.is_none());
    }

    #[test]
    fn layer_ordered_scheduler_rows_binding_exposes_selected_layers() {
        let binding = super::layer_ordered_scheduler_rows_binding();

        assert!(binding.probe.passed);
        assert!(binding.covers_all_layers());
        assert!(binding.layer_selected(0));
        assert!(binding.layer_selected(GLM52_NUM_HIDDEN_LAYERS - 1));
        assert!(!binding.layer_selected(GLM52_NUM_HIDDEN_LAYERS));
    }

    fn assert_device_kv_status(status: &str, uses_device_kv_cache: bool) {
        if coordinator_cuda_reference_kernels_enabled() {
            assert_eq!(status, "cuda-kv-cache-live-scheduler");
            assert!(uses_device_kv_cache);
        } else {
            assert!(matches!(
                status,
                "cuda-kv-cache-live-scheduler"
                    | "cuda-kv-cache-unavailable"
                    | "cuda-kv-cache-error"
            ));
        }
    }
}
