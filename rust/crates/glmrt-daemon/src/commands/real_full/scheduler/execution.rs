use anyhow::{Context, Result};
use glmrt_core::{
    DType, DecodeStep, ExpertOwnerLookup, GraphBucket, KvBlockDescriptor, KvCacheBackingStore,
    KvCacheConfig, LayerId, LayerWave, ModelFacts, MtpVerifyBlock, PositionId, PrefillChunk,
    PrefillChunkPolicy, Priority, RowSourceKind, TensorCatalog, TensorRole,
    GLM52_FIRST_K_DENSE_REPLACE, GLM52_HIDDEN_SIZE, GLM52_MTP_LAYER_ID, GLM52_NUM_HIDDEN_LAYERS,
};
use glmrt_transport::TcpProtocolV2HostBatchTarget;
use std::collections::VecDeque;
use std::env;
use std::sync::{Arc, OnceLock};
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant};

use crate::commands::real_full::constants::{
    REAL_FULL_PREFLIGHT_DECODE_ROWS, REAL_FULL_PREFLIGHT_MTP_ACCEPTED_ROWS,
    REAL_FULL_PREFLIGHT_MTP_ROWS, REAL_FULL_PREFLIGHT_PREFILL_ROWS,
    REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START, REAL_FULL_PREFLIGHT_REQUEST_ID,
    REAL_FULL_PREFLIGHT_SEQUENCE_ID,
};
use crate::commands::real_full::coordinator_kernels::{
    bf16_values_to_f32, coordinator_cuda_graph_stats,
    rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output,
    with_coordinator_owned_device_buffer_bank, CoordinatorCudaGraphStats, DeviceBf16Output,
    CUDA_REFERENCE_LM_HEAD_ARGMAX_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
    CUDA_REFERENCE_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
    TRITON_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
};
use crate::commands::real_full::kv::device::{
    RealFullDeviceKvExecutionMirror, RealFullDeviceKvStorageHandle,
};
use crate::commands::real_full::layer_blocks::SparkLayerBlock;
use crate::commands::real_full::sampling::{
    score_real_lm_head_full_vocab_for_device_hidden_with_options, RealFullLmHeadSamplingOptions,
    RealLmHeadChunkScoreForHidden,
};
use crate::commands::real_full::types::{
    RealFullSchedulerExecutionDryRun, RealFullSchedulerTerminalLmHeadSample,
};

use super::protocol_v2::RealFullSchedulerSparseTcpDispatchProbe;
use crate::commands::real_full::dspark::dspark_target_hidden_tap_layer_ids;
use admission::real_full_apply_admitted_scheduler_iteration;
use progression::{
    bounded_long_prefill_wavefront_required, rolling_sparse_dispatches_per_layer_for_rows,
    rolling_sparse_packs_supported_for_rows, RealFullSchedulerNumericProgression,
    RealFullSchedulerNumericProgressionFinish, RealFullSchedulerNumericProgressionShape,
    RealFullSchedulerSparseTcpRoutedMlpContext, RealFullSchedulerTargetHiddenTaps,
    SchedulerSparseRollingLayerApply, SchedulerSparseTcpCohortPendingDispatch,
    SchedulerSparseTcpPendingApply, SchedulerSparseTcpPreparedDispatch,
};
pub(in crate::commands::real_full) use progression::{
    RealFullSchedulerSparseDispatchTransport, RealFullSchedulerSparseTcpDispatchWorker,
};

mod admission;
mod progression;
mod snapshot;

pub(in crate::commands::real_full) use snapshot::{
    load_real_full_kv_snapshot, save_real_full_kv_snapshot, RealFullKvSnapshot,
};

const REAL_FULL_FINAL_NORM_WEIGHT_NAME: &str = "model.norm.weight";
const REAL_FULL_FINAL_NORM_EPS: f32 = 1.0e-5;
const SCHEDULER_EXECUTION_TIMING_ENV: &str = "GLMRT_REAL_FULL_SCHEDULER_TIMING";
const SCHEDULER_EXECUTION_SUMMARY_TIMING_ENV: &str = "GLMRT_REAL_FULL_SCHEDULER_SUMMARY_TIMING";
const SCHEDULER_TERMINAL_SAMPLE_VALIDATE_ENV: &str = "GLMRT_REAL_FULL_TERMINAL_SAMPLE_VALIDATE";
const MAX_PENDING_SPARSE_WAVEFRONT_DISPATCHES: usize = 4;
const MAX_ACTIVE_ROLLING_SPARSE_DISPATCHES: usize = 8;
const MAX_BUFFERED_ROLLING_SPARSE_DISPATCHES: usize = 16;
const ROLLING_SPARSE_MAX_ACTIVE_DISPATCHES_ENV: &str =
    "GLMRT_REAL_FULL_ROLLING_SPARSE_MAX_ACTIVE_DISPATCHES";
const SPARSE_WAVEFRONT_BUSY_POLL: Duration = Duration::from_millis(1);
const SPARSE_WAVEFRONT_IDLE_POLL: Duration = Duration::from_micros(20);

pub(super) struct RealFullAdmittedSchedulerIteration {
    selected: Vec<LayerWave>,
    device_attention_deltas: Vec<RealFullSchedulerDeviceAttentionDelta>,
}

pub(super) struct RealFullSchedulerDeviceAttentionDelta {
    pub(super) kind: RowSourceKind,
    pub(super) token_start: usize,
    pub(super) row_count: usize,
    pub(super) values_per_row: usize,
    pub(super) output_bf16: Option<Vec<u8>>,
    pub(super) output_device: Arc<DeviceBf16Output>,
    pub(super) output_device_row_offset: usize,
    pub(super) checksum: f64,
    pub(super) backend: &'static str,
}

#[derive(Default)]
struct SchedulerExecutionStageTiming {
    attention_ms: [f64; 2],
    numeric_ms: [f64; 2],
    sparse_wavefront_ms: f64,
}

impl SchedulerExecutionStageTiming {
    fn record_attention(&mut self, layer_id: usize, start: Option<Instant>) {
        self.record(layer_id, start, true);
    }

    fn record_numeric(&mut self, layer_id: usize, start: Option<Instant>) {
        self.record(layer_id, start, false);
    }

    fn record_sparse_wavefront(&mut self, start: Option<Instant>) {
        if let Some(start) = start {
            self.sparse_wavefront_ms += elapsed_ms(start);
        }
    }

    fn record(&mut self, layer_id: usize, start: Option<Instant>, attention: bool) {
        let Some(start) = start else {
            return;
        };
        let layer_class = usize::from(layer_id >= GLM52_FIRST_K_DENSE_REPLACE);
        let elapsed = elapsed_ms(start);
        if attention {
            self.attention_ms[layer_class] += elapsed;
        } else {
            self.numeric_ms[layer_class] += elapsed;
        }
    }
}

#[derive(Default)]
struct SchedulerRollingWavefrontTiming {
    iterations: usize,
    admissions: usize,
    dispatches_started: usize,
    poll_calls: usize,
    poll_progress_events: usize,
    idle_spins: usize,
    idle_sleeps: usize,
    max_active_dispatches: usize,
    max_buffered_dispatches: usize,
    max_accumulator_pages_per_layer: usize,
    max_accumulator_rows_per_layer: usize,
    max_accumulator_pages_total: usize,
    max_accumulator_rows_total: usize,
    plan_ms: f64,
    prepare_ms: f64,
    prepare_attention_ms: f64,
    prepare_numeric_ms: f64,
    push_ms: f64,
    push_shared_mlp_ms: f64,
    push_planner_ms: f64,
    dispatch_start_ms: f64,
    poll_ms: f64,
    idle_wait_ms: f64,
}

impl SchedulerRollingWavefrontTiming {
    fn observe_dispatches(&mut self, active: usize, buffered: usize) {
        self.max_active_dispatches = self.max_active_dispatches.max(active);
        self.max_buffered_dispatches = self.max_buffered_dispatches.max(buffered);
    }

    fn observe_accumulators(&mut self, pages: usize, rows: usize) {
        self.max_accumulator_pages_total = self.max_accumulator_pages_total.max(pages);
        self.max_accumulator_rows_total = self.max_accumulator_rows_total.max(rows);
    }
}

pub(in crate::commands::real_full) struct RealFullSchedulerExecutionState {
    store: KvCacheBackingStore,
    device_kv: RealFullDeviceKvExecutionMirror,
    reservation_id: u64,
    sequence_id: String,
    capacity_tokens: usize,
    owner_thread: ThreadId,
    pending_mtp_draft_token_ids: Vec<usize>,
    mtp_adaptive_draft: RealFullMtpAdaptiveDraftState,
    processed_token_ids: Vec<usize>,
}

const REAL_FULL_MTP_ADAPTIVE_HISTORY_LIMIT: usize = 16;
const REAL_FULL_MTP_SCALAR_PROBE_INITIAL_INTERVAL: usize = 64;
const REAL_FULL_MTP_SCALAR_PROBE_MAX_INTERVAL: usize = 256;
const REAL_FULL_MTP_SCALAR_PROBE_SUCCESSES_REQUIRED: usize = 3;
const REAL_FULL_MTP_SCALAR_TARGET_US_PER_TOKEN: usize = 50_000;
const REAL_FULL_MTP_PHYSICAL_M2_VERIFY_CYCLE_BASELINE_US: usize = 90_000;
// Short-context physical verify cost by logical D. D=1 includes the temporary
// physical-M=3 correctness bridge. Keep this conservative curve explicit until
// live context-aware timing replaces the calibrated baseline.
const REAL_FULL_MTP_VERIFY_CYCLE_BASELINE_US: [usize; 8] = [
    50_000, 105_000, 115_000, 129_000, 140_000, 147_000, 157_000, 165_000,
];

fn real_full_mtp_observation_meets_scalar_gate(
    proposed: usize,
    accepted: usize,
    physical_m2: bool,
) -> bool {
    let Some(mut cycle_us) = REAL_FULL_MTP_VERIFY_CYCLE_BASELINE_US
        .get(proposed)
        .copied()
    else {
        return false;
    };
    if physical_m2 && proposed == 1 {
        cycle_us = REAL_FULL_MTP_PHYSICAL_M2_VERIFY_CYCLE_BASELINE_US;
    }
    accepted
        .saturating_add(1)
        .saturating_mul(REAL_FULL_MTP_SCALAR_TARGET_US_PER_TOKEN)
        >= cycle_us
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RealFullMtpDraftObservation {
    proposed: usize,
    accepted: usize,
}

#[derive(Debug, Default)]
struct RealFullMtpAdaptiveDraftState {
    current: Option<usize>,
    history: VecDeque<RealFullMtpDraftObservation>,
    scalar_decisions_since_probe: usize,
    scalar_probe_in_flight: bool,
    scalar_probe_successes: usize,
    scalar_probe_interval: usize,
    below_scalar_gate_streak: usize,
}

impl RealFullMtpAdaptiveDraftState {
    fn width(&mut self, min: usize, max: usize, start: usize, adaptive: bool) -> usize {
        debug_assert!(min > 0 && min <= start && start <= max);
        if !adaptive {
            self.current = Some(start);
            self.history.clear();
            self.scalar_decisions_since_probe = 0;
            self.scalar_probe_in_flight = false;
            self.scalar_probe_successes = 0;
            self.scalar_probe_interval = REAL_FULL_MTP_SCALAR_PROBE_INITIAL_INTERVAL;
            self.below_scalar_gate_streak = 0;
            return start;
        }
        let current = *self.current.get_or_insert(start);
        if current > 0 {
            return current;
        }
        if self.scalar_probe_in_flight {
            return min;
        }
        self.scalar_decisions_since_probe += 1;
        let probe_interval = self
            .scalar_probe_interval
            .max(REAL_FULL_MTP_SCALAR_PROBE_INITIAL_INTERVAL);
        if self.scalar_decisions_since_probe >= probe_interval {
            self.scalar_decisions_since_probe = 0;
            self.scalar_probe_in_flight = true;
            min
        } else {
            0
        }
    }

    fn observe(
        &mut self,
        min: usize,
        max: usize,
        start: usize,
        proposed: usize,
        accepted: usize,
        adaptive: bool,
        physical_m2: bool,
    ) {
        debug_assert!(min > 0 && min <= start && start <= max);
        if !adaptive {
            self.width(min, max, start, adaptive);
            return;
        }
        let current = *self.current.get_or_insert(start);
        let scalar_probe = current == 0 && self.scalar_probe_in_flight && proposed == min;
        if proposed != current && !scalar_probe {
            // A budget-limited tail is not evidence that the request prefers a
            // smaller speculative window.
            return;
        }

        self.history.push_back(RealFullMtpDraftObservation {
            proposed,
            accepted: accepted.min(proposed),
        });
        while self.history.len() > REAL_FULL_MTP_ADAPTIVE_HISTORY_LIMIT {
            self.history.pop_front();
        }

        let scalar_probe_profitable =
            real_full_mtp_observation_meets_scalar_gate(proposed, accepted, physical_m2);
        let next = if scalar_probe {
            self.scalar_probe_in_flight = false;
            if accepted == proposed && scalar_probe_profitable {
                self.scalar_probe_successes = 0;
                self.scalar_probe_interval = REAL_FULL_MTP_SCALAR_PROBE_INITIAL_INTERVAL;
                min
            } else if accepted == proposed && min == 1 && !physical_m2 {
                // The deterministic D=1 bridge is physically M=3 and cannot
                // beat scalar decode by itself. Require repeated full probes
                // before jumping over it into profitable logical D=2.
                self.scalar_probe_successes += 1;
                if self.scalar_probe_successes >= REAL_FULL_MTP_SCALAR_PROBE_SUCCESSES_REQUIRED {
                    self.scalar_probe_successes = 0;
                    min.saturating_add(1).min(max)
                } else {
                    0
                }
            } else {
                self.scalar_probe_successes = 0;
                self.scalar_probe_interval = self
                    .scalar_probe_interval
                    .max(REAL_FULL_MTP_SCALAR_PROBE_INITIAL_INTERVAL)
                    .saturating_mul(2)
                    .min(REAL_FULL_MTP_SCALAR_PROBE_MAX_INTERVAL);
                0
            }
        } else if accepted == 0 {
            if physical_m2 && current == 1 && self.below_scalar_gate_streak == 0 {
                // One isolated first-draft miss does not invalidate an
                // otherwise profitable physical-M=2 state.
                1
            } else if current <= min.saturating_add(1) {
                self.scalar_decisions_since_probe = 0;
                self.scalar_probe_successes = 0;
                0
            } else {
                current.saturating_sub(1).max(min)
            }
        } else {
            let at_current = self
                .history
                .iter()
                .rev()
                .take_while(|sample| sample.proposed == current);
            let recent = at_current.take(3).copied().collect::<Vec<_>>();
            let poor = |sample: &RealFullMtpDraftObservation| {
                sample.accepted.saturating_add(1).saturating_mul(2) <= sample.proposed
            };
            let saturated = |sample: &RealFullMtpDraftObservation| {
                sample.accepted.saturating_add(1) >= sample.proposed
            };
            if recent.len() >= 2 && recent[..2].iter().all(poor) {
                current.saturating_sub(1).max(min)
            } else if recent.len() >= 3 && recent.iter().all(saturated) {
                current.saturating_add(1).min(max)
            } else {
                current
            }
        };
        if !scalar_probe {
            if real_full_mtp_observation_meets_scalar_gate(proposed, accepted, physical_m2) {
                self.below_scalar_gate_streak = 0;
            } else {
                self.below_scalar_gate_streak += 1;
            }
        }
        // One weak framing cycle can precede highly structured output. Require
        // a second cost miss before abandoning wider speculation.
        let next = if !scalar_probe && self.below_scalar_gate_streak >= 2 {
            if physical_m2 && proposed > 1 && accepted > 0 {
                // The first draft still succeeded, so physical M=2 may be
                // profitable even though this wider cycle was not.
                1
            } else {
                0
            }
        } else {
            next
        };
        if next == 0 {
            self.scalar_decisions_since_probe = 0;
            self.scalar_probe_in_flight = false;
            self.below_scalar_gate_streak = 0;
            self.scalar_probe_interval = self
                .scalar_probe_interval
                .max(REAL_FULL_MTP_SCALAR_PROBE_INITIAL_INTERVAL);
        }
        self.current = Some(next);
    }

    fn reset(&mut self) {
        self.current = None;
        self.history.clear();
        self.scalar_decisions_since_probe = 0;
        self.scalar_probe_in_flight = false;
        self.scalar_probe_successes = 0;
        self.scalar_probe_interval = REAL_FULL_MTP_SCALAR_PROBE_INITIAL_INTERVAL;
        self.below_scalar_gate_streak = 0;
    }
}

pub(in crate::commands::real_full) struct RealFullSchedulerDeviceExecution {
    pub(in crate::commands::real_full) report: RealFullSchedulerExecutionDryRun,
    pub(in crate::commands::real_full) sparse_tcp_dispatch: RealFullSchedulerSparseTcpDispatchProbe,
    pub(in crate::commands::real_full) final_target_device_hidden: Option<DeviceBf16Output>,
    pub(in crate::commands::real_full) target_device_hidden_taps:
        Option<RealFullSchedulerTargetHiddenTaps>,
}

// The serving executor will hold this state behind a mutex. CUDA and host KV
// buffers are only accessed while that mutable state lock is held.
unsafe impl Send for RealFullSchedulerExecutionState {}

impl RealFullSchedulerExecutionState {
    pub(in crate::commands::real_full) fn new(
        kv_config: KvCacheConfig,
        sequence_id: impl Into<String>,
    ) -> Result<Self> {
        let capacity_tokens = kv_config.max_tokens;
        Self::new_with_capacity(kv_config, sequence_id, capacity_tokens)
    }

    pub(in crate::commands::real_full) fn new_with_capacity(
        kv_config: KvCacheConfig,
        sequence_id: impl Into<String>,
        capacity_tokens: usize,
    ) -> Result<Self> {
        Self::new_with_arena_capacity(kv_config, sequence_id, capacity_tokens, capacity_tokens)
    }

    pub(in crate::commands::real_full) fn new_with_arena_capacity(
        kv_config: KvCacheConfig,
        sequence_id: impl Into<String>,
        capacity_tokens: usize,
        arena_capacity_tokens: usize,
    ) -> Result<Self> {
        let device_kv_storage_config = kv_config.clone();
        Self::new_with_arena_capacity_and_storage(
            kv_config,
            sequence_id,
            capacity_tokens,
            arena_capacity_tokens,
            device_kv_storage_config,
            None,
            0,
        )
    }

    pub(in crate::commands::real_full) fn new_with_arena_capacity_and_storage(
        mut kv_config: KvCacheConfig,
        sequence_id: impl Into<String>,
        capacity_tokens: usize,
        arena_capacity_tokens: usize,
        device_kv_storage_config: KvCacheConfig,
        device_kv_storage: Option<RealFullDeviceKvStorageHandle>,
        device_kv_physical_token_base: usize,
    ) -> Result<Self> {
        let sequence_id = sequence_id.into();
        anyhow::ensure!(
            capacity_tokens > 0,
            "real-full scheduler execution state requires nonzero KV capacity"
        );
        anyhow::ensure!(
            arena_capacity_tokens >= capacity_tokens,
            "real-full scheduler arena capacity {arena_capacity_tokens} is smaller than logical sequence capacity {capacity_tokens}"
        );
        anyhow::ensure!(
            arena_capacity_tokens <= kv_config.max_tokens,
            "real-full scheduler arena capacity {arena_capacity_tokens} exceeds global context budget {}",
            kv_config.max_tokens
        );
        anyhow::ensure!(
            device_kv_storage_config.layout == kv_config.layout
                && device_kv_storage_config.layers == kv_config.layers
                && device_kv_storage_config.key_value_width == kv_config.key_value_width
                && device_kv_storage_config.dtype == kv_config.dtype
                && device_kv_storage_config.mla_representation == kv_config.mla_representation
                && device_kv_storage_config.dsa_indexer_layers == kv_config.dsa_indexer_layers
                && device_kv_storage_config.dsa_index_head_dim == kv_config.dsa_index_head_dim
                && device_kv_storage_config.fp8_scale_metadata_bytes_per_token
                    == kv_config.fp8_scale_metadata_bytes_per_token
                && device_kv_storage_config.max_tokens >= arena_capacity_tokens,
            "real-full shared device KV pool format or capacity does not match the sequence cache"
        );
        kv_config.max_tokens = arena_capacity_tokens;
        let mut store = KvCacheBackingStore::new(kv_config.clone());
        let reservation_id = store
            .reserve(sequence_id.as_str(), capacity_tokens)
            .context("reserving persistent real-full scheduler KV state")?;
        let device_kv = match device_kv_storage {
            Some(storage) => RealFullDeviceKvExecutionMirror::new_with_storage(
                storage,
                device_kv_physical_token_base,
                capacity_tokens,
            ),
            None => RealFullDeviceKvExecutionMirror::new(device_kv_storage_config),
        }
        .context("creating persistent real-full scheduler device KV state")?;
        Ok(Self {
            store,
            device_kv,
            reservation_id,
            sequence_id,
            capacity_tokens,
            owner_thread: thread::current().id(),
            pending_mtp_draft_token_ids: Vec::new(),
            mtp_adaptive_draft: RealFullMtpAdaptiveDraftState::default(),
            processed_token_ids: Vec::new(),
        })
    }

    pub(in crate::commands::real_full) fn arena_capacity_tokens(&self) -> usize {
        self.store.config().max_tokens
    }

    pub(in crate::commands::real_full) fn device_kv_storage_handle(
        &self,
    ) -> Option<RealFullDeviceKvStorageHandle> {
        self.device_kv.storage_handle()
    }

    pub(in crate::commands::real_full) fn owned_by_current_thread(&self) -> bool {
        self.owner_thread == thread::current().id()
    }

    pub(in crate::commands::real_full) fn rebind_sequence(
        &mut self,
        sequence_id: impl Into<String>,
        capacity_tokens: usize,
        physical_token_base: usize,
    ) -> Result<()> {
        let sequence_id = sequence_id.into();
        anyhow::ensure!(
            self.owned_by_current_thread(),
            "real-full scheduler KV arena cannot move between graph-owner threads"
        );
        anyhow::ensure!(
            capacity_tokens > 0 && capacity_tokens <= self.arena_capacity_tokens(),
            "real-full scheduler recycled arena capacity {} cannot admit {capacity_tokens} tokens",
            self.arena_capacity_tokens()
        );
        let mut store = KvCacheBackingStore::new(self.store.config().clone());
        let reservation_id = store
            .reserve(sequence_id.as_str(), capacity_tokens)
            .context("reserving rebound real-full scheduler KV state")?;
        self.device_kv
            .rebind_physical_extent(physical_token_base, capacity_tokens)
            .context("rebinding recycled scheduler device KV extent")?;
        self.store = store;
        self.reservation_id = reservation_id;
        self.sequence_id = sequence_id;
        self.capacity_tokens = capacity_tokens;
        self.pending_mtp_draft_token_ids.clear();
        self.mtp_adaptive_draft.reset();
        self.processed_token_ids.clear();
        Ok(())
    }

    pub(in crate::commands::real_full) fn rebind_sequence_physical_pages(
        &mut self,
        physical_pages: &[u32],
        capacity_tokens: usize,
    ) -> Result<()> {
        self.device_kv
            .rebind_physical_pages(physical_pages, capacity_tokens)
            .context("rebinding scheduler sequence to target KV physical pages")
    }

    pub(in crate::commands::real_full) fn extend_sequence_physical_pages(
        &mut self,
        physical_pages: &[u32],
        capacity_tokens: usize,
    ) -> Result<()> {
        self.device_kv
            .extend_physical_pages(physical_pages, capacity_tokens)
            .context("extending scheduler target KV physical page table")
    }

    pub(in crate::commands::real_full) fn copy_target_kv_boundary_page(
        &mut self,
        source_page: u32,
        destination_page: u32,
        valid_tokens: usize,
    ) -> Result<()> {
        self.device_kv
            .copy_target_kv_boundary_page(source_page, destination_page, valid_tokens)
            .context("copying target KV radix boundary page")
    }

    pub(in crate::commands::real_full) fn seed_processed_token_ids(
        &mut self,
        token_ids: &[usize],
    ) -> Result<()> {
        anyhow::ensure!(
            token_ids.len() <= self.capacity_tokens,
            "processed-token radix prefix {} exceeds sequence capacity {}",
            token_ids.len(),
            self.capacity_tokens
        );
        if !token_ids.is_empty() {
            for layer_index in 0..self.store.config().layers {
                let layer_id =
                    u32::try_from(layer_index).context("radix seed layer exceeds u32")?;
                self.store
                    .write_committed_block_metadata(KvBlockDescriptor {
                        reservation_id: self.reservation_id,
                        sequence_id: self.sequence_id.clone(),
                        layer_id: LayerId(layer_id),
                        token_start: PositionId(0),
                        token_count: token_ids.len(),
                    })
                    .with_context(|| {
                        format!(
                            "seeding target radix KV metadata for layer {layer_index} through {} tokens",
                            token_ids.len()
                        )
                    })?;
            }
        }
        self.processed_token_ids.clear();
        self.processed_token_ids.extend_from_slice(token_ids);
        Ok(())
    }

    fn validate_shape(&self, shape: &RealFullSchedulerExecutionShape) -> Result<()> {
        anyhow::ensure!(
            shape.sequence_id == self.sequence_id,
            "real-full scheduler state sequence id mismatch: state={} shape={}",
            self.sequence_id,
            shape.sequence_id
        );
        anyhow::ensure!(
            shape.reservation_tokens() <= self.capacity_tokens,
            "real-full scheduler state capacity {} is smaller than requested token span {}",
            self.capacity_tokens,
            shape.reservation_tokens()
        );
        Ok(())
    }

    pub(in crate::commands::real_full) fn resolve_mtp_tentative_writes(
        &mut self,
        token_start: usize,
        draft_tokens: usize,
        accepted_tokens: usize,
    ) -> Result<()> {
        for layer_id in 0..GLM52_NUM_HIDDEN_LAYERS {
            self.store
                .resolve_mtp_tentative_writes(
                    self.reservation_id,
                    LayerId(layer_id as u32),
                    PositionId(token_start as u64),
                    draft_tokens,
                    accepted_tokens,
                )
                .with_context(|| {
                    format!(
                        "resolving live MTP target KV writes for layer {layer_id} at {token_start}+{draft_tokens} accepted={accepted_tokens}"
                    )
                })?;
        }
        self.device_kv
            .resolve_mtp_tentative_frontiers(
                self.reservation_id,
                &self.sequence_id,
                token_start,
                draft_tokens,
                accepted_tokens,
            )
            .context("resolving live MTP target attention-ready KV frontiers")?;
        Ok(())
    }

    pub(in crate::commands::real_full) fn rewind_mtp_draft_layer(
        &mut self,
        token_start: usize,
    ) -> Result<usize> {
        let layer_id = LayerId(GLM52_MTP_LAYER_ID as u32);
        let discarded = self.store.discard_writes_from(
            self.reservation_id,
            layer_id,
            PositionId(token_start as u64),
        );
        self.device_kv
            .rewind_attention_ready_frontier(
                self.reservation_id,
                &self.sequence_id,
                layer_id,
                token_start,
            )
            .context("rewinding live MTP draft-layer attention-ready frontier")?;
        Ok(discarded)
    }

    pub(in crate::commands::real_full) fn take_pending_mtp_draft_token_ids(
        &mut self,
    ) -> Vec<usize> {
        std::mem::take(&mut self.pending_mtp_draft_token_ids)
    }

    pub(in crate::commands::real_full) fn set_pending_mtp_draft_token_ids(
        &mut self,
        token_ids: Vec<usize>,
    ) {
        self.pending_mtp_draft_token_ids = token_ids;
    }

    pub(in crate::commands::real_full) fn mtp_draft_width(
        &mut self,
        min: usize,
        max: usize,
        start: usize,
        adaptive: bool,
    ) -> usize {
        self.mtp_adaptive_draft.width(min, max, start, adaptive)
    }

    pub(in crate::commands::real_full) fn observe_mtp_draft_acceptance(
        &mut self,
        min: usize,
        max: usize,
        start: usize,
        proposed: usize,
        accepted: usize,
        adaptive: bool,
        physical_m2: bool,
    ) {
        self.mtp_adaptive_draft
            .observe(min, max, start, proposed, accepted, adaptive, physical_m2);
    }

    pub(in crate::commands::real_full) fn record_processed_token_ids(
        &mut self,
        prefix_tokens: usize,
        token_ids: &[usize],
    ) -> Result<()> {
        anyhow::ensure!(
            self.processed_token_ids.len() == prefix_tokens,
            "processed-token frontier {} does not match scheduler prefix {prefix_tokens}",
            self.processed_token_ids.len()
        );
        let next_len = prefix_tokens
            .checked_add(token_ids.len())
            .context("processed-token frontier overflow usize")?;
        anyhow::ensure!(
            next_len <= self.capacity_tokens,
            "processed-token frontier {next_len} exceeds sequence capacity {}",
            self.capacity_tokens
        );
        self.processed_token_ids.extend_from_slice(token_ids);
        Ok(())
    }

    pub(in crate::commands::real_full) fn processed_token_ids(&self) -> &[usize] {
        &self.processed_token_ids
    }

    fn snapshot_layer_payload(&mut self, layer_id: LayerId, token_count: usize) -> Result<Vec<u8>> {
        let descriptor = KvBlockDescriptor {
            reservation_id: self.reservation_id,
            sequence_id: self.sequence_id.clone(),
            layer_id,
            token_start: PositionId(0),
            token_count,
        };
        self.device_kv.snapshot_layer_payload(&descriptor)
    }

    fn snapshot_dsa_index_prefix(
        &self,
        layer_id: LayerId,
        token_count: usize,
    ) -> Result<Option<Vec<u8>>> {
        self.device_kv
            .snapshot_dsa_index_prefix(layer_id, token_count)
    }

    fn snapshot_layer_has_attention_visible_prefix(
        &self,
        layer_id: LayerId,
        token_count: usize,
    ) -> bool {
        if token_count == 0 {
            return true;
        }
        let blocks = self.store.read_visible_blocks_for_decode(
            self.reservation_id,
            layer_id,
            PositionId(self.capacity_tokens as u64),
        );
        let mut covered_end = 0_usize;
        for block in blocks {
            let Ok(block_start) = usize::try_from(block.descriptor.token_start.0) else {
                return false;
            };
            if block_start > covered_end {
                break;
            }
            covered_end = covered_end.max(block_start.saturating_add(block.descriptor.token_count));
            if covered_end >= token_count {
                return true;
            }
        }
        false
    }

    fn snapshot_mtp_layer_token_count(&self, token_count: usize) -> usize {
        self.snapshot_layer_has_attention_visible_prefix(
            LayerId(GLM52_MTP_LAYER_ID as u32),
            token_count,
        )
        .then_some(token_count)
        .unwrap_or(0)
    }

    fn restore_snapshot_layer(
        &mut self,
        layer_id: LayerId,
        token_count: usize,
        payload: &[u8],
        dsa_index_payload: Option<&[u8]>,
    ) -> Result<()> {
        let descriptor = KvBlockDescriptor {
            reservation_id: self.reservation_id,
            sequence_id: self.sequence_id.clone(),
            layer_id,
            token_start: PositionId(0),
            token_count,
        };
        self.device_kv
            .restore_layer_payload(&descriptor, payload)
            .with_context(|| format!("restoring device KV snapshot layer {}", layer_id.0))?;
        self.store
            .write_committed_block_metadata(descriptor)
            .with_context(|| format!("restoring KV metadata for layer {}", layer_id.0))?;
        if let Some(dsa_index_payload) = dsa_index_payload {
            self.device_kv
                .restore_dsa_index_prefix(layer_id, token_count, dsa_index_payload)
                .with_context(|| format!("restoring packed DSA snapshot layer {}", layer_id.0))?;
        }
        Ok(())
    }

    fn set_restored_token_ids(&mut self, token_ids: Vec<usize>) -> Result<()> {
        anyhow::ensure!(
            token_ids.len() <= self.capacity_tokens,
            "snapshot token count {} exceeds sequence capacity {}",
            token_ids.len(),
            self.capacity_tokens
        );
        self.processed_token_ids = token_ids;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub(in crate::commands::real_full) struct RealFullSchedulerExecutionShape {
    pub(in crate::commands::real_full) request_id: String,
    pub(in crate::commands::real_full) sequence_id: String,
    pub(in crate::commands::real_full) placement_version: String,
    pub(in crate::commands::real_full) prefix_tokens: usize,
    pub(in crate::commands::real_full) prefill_tokens: usize,
    pub(in crate::commands::real_full) prefill_chunk_tokens: usize,
    pub(in crate::commands::real_full) decode_rows: usize,
    pub(in crate::commands::real_full) mtp_rows: usize,
    pub(in crate::commands::real_full) mtp_accepted_rows: usize,
    pub(in crate::commands::real_full) prefill_token_ids: Option<Vec<usize>>,
    pub(in crate::commands::real_full) prefill_vision_embeddings:
        Option<Arc<Vec<glmrt_api::RealFullVisionEmbedding>>>,
    pub(in crate::commands::real_full) decode_token_ids: Option<Vec<usize>>,
    pub(in crate::commands::real_full) lm_head_sampling: RealFullLmHeadSamplingOptions,
}

impl RealFullSchedulerExecutionShape {
    pub(in crate::commands::real_full) fn preflight() -> Self {
        Self {
            request_id: REAL_FULL_PREFLIGHT_REQUEST_ID.to_owned(),
            sequence_id: REAL_FULL_PREFLIGHT_SEQUENCE_ID.to_owned(),
            placement_version: "real-full-admitted-scheduler".to_owned(),
            prefix_tokens: 0,
            prefill_tokens: REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START as usize
                + REAL_FULL_PREFLIGHT_PREFILL_ROWS,
            prefill_chunk_tokens: REAL_FULL_PREFLIGHT_PREFILL_ROWS,
            decode_rows: REAL_FULL_PREFLIGHT_DECODE_ROWS,
            mtp_rows: REAL_FULL_PREFLIGHT_MTP_ROWS,
            mtp_accepted_rows: REAL_FULL_PREFLIGHT_MTP_ACCEPTED_ROWS,
            prefill_token_ids: None,
            prefill_vision_embeddings: None,
            decode_token_ids: None,
            lm_head_sampling: RealFullLmHeadSamplingOptions::diagnostic(),
        }
    }

    fn prefill_token_start(&self) -> usize {
        self.prefix_tokens
    }

    fn decode_token_start(&self) -> usize {
        self.prefix_tokens + self.prefill_tokens
    }

    fn mtp_token_start(&self) -> usize {
        self.decode_token_start() + self.decode_rows
    }

    pub(in crate::commands::real_full) fn reservation_tokens(&self) -> usize {
        self.prefix_tokens + self.prefill_tokens + self.decode_rows + self.mtp_rows
    }

    fn sparse_batch_graph_bucket(&self) -> GraphBucket {
        GraphBucket::new(self.prefill_chunk_tokens + self.decode_rows + self.mtp_rows)
    }

    fn mtp_resolution_accepted_rows(&self) -> Option<usize> {
        let has_real_mtp_tokens = self.mtp_rows > 0
            && self
                .decode_token_ids
                .as_ref()
                .is_some_and(|token_ids| token_ids.len() == self.decode_rows + self.mtp_rows);
        (!has_real_mtp_tokens).then_some(self.mtp_accepted_rows)
    }
}

#[derive(Debug, Default)]
struct RealFullSchedulerExecutionCounters {
    iterations: usize,
    candidate_layerwaves: usize,
    selected_layerwaves: usize,
    deferred_layerwaves: usize,
    selected_decode_rows: usize,
    selected_prefill_rows: usize,
    selected_mtp_rows: usize,
    sparse_expert_batches: usize,
    sparse_expert_batch_rows: usize,
    sparse_expert_batch_routes: usize,
    sparse_expert_prefill_rows: usize,
    sparse_expert_decode_rows: usize,
    sparse_expert_mtp_verify_rows: usize,
    sparse_expert_prefill_routes: usize,
    sparse_expert_decode_routes: usize,
    sparse_expert_mtp_verify_routes: usize,
    sparse_expert_host_batch_sets: usize,
    sparse_expert_host_batches: usize,
    sparse_expert_host_batch_rows: usize,
    sparse_expert_host_batch_routes: usize,
    sparse_expert_host_batch_expert_tiles: usize,
    sparse_expert_host_batch_routes_match_global: bool,
    sparse_expert_host_batch_graph_counts_valid: bool,
    sparse_expert_host_request_frames: usize,
    sparse_expert_host_request_rows: usize,
    sparse_expert_host_request_routes: usize,
    sparse_expert_host_request_payload_bytes: usize,
    sparse_expert_host_request_wire_bytes: usize,
    sparse_expert_host_response_frames: usize,
    sparse_expert_host_response_rows: usize,
    sparse_expert_host_response_payload_bytes: usize,
    sparse_expert_host_response_wire_bytes: usize,
    sparse_expert_host_wire_envelopes_valid: bool,
    kv_read_blocks: usize,
    committed_kv_writes: usize,
    tentative_kv_writes: usize,
    projected_device_kv_writes: usize,
    projected_device_kv_write_bytes: usize,
    synthetic_kv_payload_writes: usize,
    device_attention_status: Option<&'static str>,
    device_attention_launches: usize,
    device_attention_rows: usize,
    device_attention_query_rows: usize,
    device_attention_kv_descriptors: usize,
    device_attention_output_bytes: usize,
    device_attention_output_values: usize,
    device_attention_output_finite_values: usize,
    device_attention_output_nonzero_values: usize,
    device_attention_output_checksum: f64,
    device_attention_hidden_projection_launches: usize,
    layer_order_verified: bool,
}

#[derive(Debug, Clone, Copy)]
struct SchedulerKvReportSummary {
    committed_mtp_writes: usize,
    discarded_mtp_writes: usize,
    backed_kv_writes: usize,
    backed_bytes_after_discard: usize,
    kv_reservation_bytes: usize,
    byte_backed_scheduler_trace: bool,
}

pub(in crate::commands::real_full) fn real_full_scheduler_execution_dry_run(
    kv_config: KvCacheConfig,
    catalog: &TensorCatalog,
) -> Result<RealFullSchedulerExecutionDryRun> {
    real_full_scheduler_execution_for_shape(
        kv_config,
        catalog,
        RealFullSchedulerExecutionShape::preflight(),
    )
}

pub(in crate::commands::real_full) fn real_full_scheduler_execution_for_shape(
    kv_config: KvCacheConfig,
    catalog: &TensorCatalog,
    shape: RealFullSchedulerExecutionShape,
) -> Result<RealFullSchedulerExecutionDryRun> {
    Ok(real_full_scheduler_execution_for_shape_inner(
        kv_config, catalog, shape, None, None, false, false, 0,
    )?
    .0)
}

pub(in crate::commands::real_full) fn real_full_scheduler_execution_for_shape_with_state(
    kv_config: KvCacheConfig,
    catalog: &TensorCatalog,
    shape: RealFullSchedulerExecutionShape,
    state: &mut RealFullSchedulerExecutionState,
) -> Result<RealFullSchedulerExecutionDryRun> {
    Ok(real_full_scheduler_execution_for_shape_inner(
        kv_config,
        catalog,
        shape,
        None,
        Some(state),
        false,
        false,
        0,
    )?
    .0)
}

pub(in crate::commands::real_full) fn real_full_scheduler_execution_for_shape_with_sparse_tcp(
    kv_config: KvCacheConfig,
    catalog: &TensorCatalog,
    shape: RealFullSchedulerExecutionShape,
    targets: Vec<TcpProtocolV2HostBatchTarget>,
    owner_lookup: Option<ExpertOwnerLookup>,
    request_id_base: u64,
) -> Result<(
    RealFullSchedulerExecutionDryRun,
    RealFullSchedulerSparseTcpDispatchProbe,
)> {
    validate_scheduler_execution_shape(&shape)?;
    let scheduler_iterations_per_sparse_layer =
        scheduler_sparse_tcp_iterations_per_sparse_layer(&shape);
    let context = RealFullSchedulerSparseTcpRoutedMlpContext::new(
        scheduler_iterations_per_sparse_layer,
        targets,
        owner_lookup,
        request_id_base,
    )?;
    real_full_scheduler_execution_for_shape_with_sparse_tcp_context(
        kv_config, catalog, shape, context, None,
    )
}

pub(in crate::commands::real_full) fn real_full_scheduler_execution_for_shape_with_shared_sparse_tcp_and_state_device_hidden(
    kv_config: KvCacheConfig,
    catalog: &TensorCatalog,
    shape: RealFullSchedulerExecutionShape,
    dispatch_worker: Arc<RealFullSchedulerSparseTcpDispatchWorker>,
    request_id_base: u64,
    state: &mut RealFullSchedulerExecutionState,
    retain_final_target_device_hidden: bool,
    retain_full_target_device_hidden: bool,
    target_device_hidden_tap_rows: usize,
) -> Result<RealFullSchedulerDeviceExecution> {
    validate_scheduler_execution_shape(&shape)?;
    let scheduler_iterations_per_sparse_layer =
        scheduler_sparse_tcp_iterations_per_sparse_layer(&shape);
    let context = RealFullSchedulerSparseTcpRoutedMlpContext::with_dispatch_worker(
        scheduler_iterations_per_sparse_layer,
        dispatch_worker,
        request_id_base,
    )?;
    let (report, sparse_tcp_dispatch, final_target_device_hidden, target_device_hidden_taps) =
        real_full_scheduler_execution_for_shape_inner(
            kv_config,
            catalog,
            shape,
            Some(context),
            Some(state),
            retain_final_target_device_hidden,
            retain_full_target_device_hidden,
            target_device_hidden_tap_rows,
        )?;
    Ok(RealFullSchedulerDeviceExecution {
        report,
        sparse_tcp_dispatch: sparse_tcp_dispatch.context(
            "scheduler sparse TCP residual dispatch probe missing after device-hidden execution",
        )?,
        final_target_device_hidden,
        target_device_hidden_taps,
    })
}

struct BatchedLiveSchedulerRun<'a> {
    buffer_bank: usize,
    shape: RealFullSchedulerExecutionShape,
    store: &'a mut KvCacheBackingStore,
    reservation_id: u64,
    device_kv: &'a mut RealFullDeviceKvExecutionMirror,
    policy: PrefillChunkPolicy,
    sparse_batch_graph_bucket: GraphBucket,
    quantization_recipe: String,
    counters: RealFullSchedulerExecutionCounters,
    stage_timing: SchedulerExecutionStageTiming,
    numeric_progression: RealFullSchedulerNumericProgression,
    retain_final_target_device_hidden: bool,
    execution_start: Instant,
    coordinator_graph_stats_before: Option<CoordinatorCudaGraphStats>,
}

impl<'a> BatchedLiveSchedulerRun<'a> {
    fn sparse_rows(&self) -> usize {
        self.shape.decode_rows + self.shape.mtp_rows
    }

    fn with_buffer_bank<T>(&mut self, action: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        let buffer_bank = self.buffer_bank;
        with_coordinator_owned_device_buffer_bank(buffer_bank, || action(self))
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        catalog: &TensorCatalog,
        shape: RealFullSchedulerExecutionShape,
        state: &'a mut RealFullSchedulerExecutionState,
        dispatch_worker: Arc<RealFullSchedulerSparseTcpDispatchWorker>,
        request_id_base: u64,
        retain_final_target_device_hidden: bool,
        target_device_hidden_tap_rows: usize,
        buffer_bank: usize,
    ) -> Result<Self> {
        validate_scheduler_execution_shape(&shape)?;
        anyhow::ensure!(
            shape.prefill_tokens == 0 && shape.decode_rows == 1,
            "batched live scheduler currently requires recurrent decode/verify shapes, got prefill={} decode={}",
            shape.prefill_tokens,
            shape.decode_rows
        );
        state.validate_shape(&shape)?;
        let reservation_id = state.reservation_id;
        let numeric_shape = RealFullSchedulerNumericProgressionShape::from_execution_shape(&shape);
        let mut numeric_progression = RealFullSchedulerNumericProgression::new(numeric_shape)
            .with_live_request()
            .with_sparse_tcp_routed_mlp(
                RealFullSchedulerSparseTcpRoutedMlpContext::with_dispatch_worker(
                    scheduler_sparse_tcp_iterations_per_sparse_layer(&shape),
                    dispatch_worker,
                    request_id_base,
                )?,
            );
        if retain_final_target_device_hidden {
            numeric_progression = numeric_progression.with_final_target_device_hidden();
        }
        if target_device_hidden_tap_rows > 0 {
            numeric_progression =
                numeric_progression.with_target_device_hidden_taps(target_device_hidden_tap_rows);
        }
        let decode_token_ids = shape
            .decode_token_ids
            .as_deref()
            .context("batched live scheduler requires decode token ids")?;
        numeric_progression
            .seed_decode_token_embeddings(catalog, &decode_token_ids[..shape.decode_rows])
            .context("seeding batched scheduler decode residual rows from token embeddings")?;
        if decode_token_ids.len() > shape.decode_rows {
            numeric_progression
                .seed_mtp_token_embeddings(catalog, &decode_token_ids[shape.decode_rows..])
                .context("seeding batched scheduler MTP residual rows from draft embeddings")?;
        }
        let sparse_batch_graph_bucket = shape.sparse_batch_graph_bucket();
        let policy = PrefillChunkPolicy {
            chunk_tokens: shape.prefill_chunk_tokens,
            max_prefill_tokens_per_iteration: shape.prefill_chunk_tokens,
            max_active_prefill_chunks: 1,
            decode_priority: true,
        };
        let counters = RealFullSchedulerExecutionCounters {
            layer_order_verified: true,
            sparse_expert_host_batch_routes_match_global: true,
            sparse_expert_host_batch_graph_counts_valid: true,
            sparse_expert_host_wire_envelopes_valid: true,
            ..Default::default()
        };
        let RealFullSchedulerExecutionState {
            store, device_kv, ..
        } = state;
        Ok(Self {
            buffer_bank,
            shape,
            store,
            reservation_id,
            device_kv,
            policy,
            sparse_batch_graph_bucket,
            quantization_recipe: ModelFacts::default().quantization_recipe,
            counters,
            stage_timing: SchedulerExecutionStageTiming::default(),
            numeric_progression,
            retain_final_target_device_hidden,
            execution_start: Instant::now(),
            coordinator_graph_stats_before: coordinator_cuda_graph_stats().ok(),
        })
    }

    fn apply_attention_layer(
        &mut self,
        catalog: &TensorCatalog,
        layer_id: usize,
        scheduler_timing: bool,
    ) -> Result<RealFullAdmittedSchedulerIteration> {
        let buffer_bank = self.buffer_bank;
        with_coordinator_owned_device_buffer_bank(buffer_bank, || {
            self.apply_attention_layer_in_bank(catalog, layer_id, scheduler_timing)
        })
    }

    fn apply_attention_layer_in_bank(
        &mut self,
        catalog: &TensorCatalog,
        layer_id: usize,
        scheduler_timing: bool,
    ) -> Result<RealFullAdmittedSchedulerIteration> {
        let layer = LayerId(layer_id as u32);
        let mut waves = (0..self.shape.decode_rows)
            .map(|decode_offset| {
                LayerWave::decode(DecodeStep::new(
                    self.shape.request_id.as_str(),
                    self.shape.sequence_id.as_str(),
                    layer,
                    PositionId((self.shape.decode_token_start() + decode_offset) as u64),
                    Some(self.reservation_id),
                    Priority(0),
                    self.shape.placement_version.as_str(),
                ))
            })
            .collect::<Vec<_>>();
        if self.shape.mtp_rows > 0 {
            waves.push(LayerWave::mtp_verify(MtpVerifyBlock::new(
                self.shape.request_id.as_str(),
                self.shape.sequence_id.as_str(),
                layer,
                PositionId(self.shape.mtp_token_start() as u64),
                self.shape.mtp_rows,
                Some(self.reservation_id),
                Priority(0),
                GraphBucket::new(self.shape.mtp_rows),
                self.shape.placement_version.as_str(),
            )));
        }
        let expected_modes = waves.iter().map(|wave| wave.mode).collect::<Vec<_>>();
        let attention_start = scheduler_timing.then(Instant::now);
        let iteration = real_full_apply_admitted_scheduler_iteration(
            self.store,
            &self.policy,
            waves,
            &expected_modes,
            self.sparse_batch_graph_bucket,
            self.quantization_recipe.as_str(),
            self.shape.mtp_resolution_accepted_rows(),
            false,
            &mut self.counters,
            self.device_kv,
            &mut self.numeric_progression,
            catalog,
        );
        self.stage_timing
            .record_attention(layer_id, attention_start);
        iteration
    }

    fn apply_dense_layer(
        &mut self,
        catalog: &TensorCatalog,
        layer_id: usize,
        scheduler_timing: bool,
    ) -> Result<()> {
        let iteration = self
            .apply_attention_layer(catalog, layer_id, scheduler_timing)
            .with_context(|| {
                format!(
                    "applying batched scheduler dense attention for request {} layer {layer_id}",
                    self.shape.request_id
                )
            })?;
        anyhow::ensure!(
            !self
                .numeric_progression
                .can_pipeline_selected_sparse_tcp_batched(layer_id, &iteration.selected),
            "batched scheduler dense layer {layer_id} unexpectedly selected sparse pipelining"
        );
        let numeric_start = scheduler_timing.then(Instant::now);
        self.with_buffer_bank(|run| {
            run.numeric_progression.apply_selected(
                layer_id,
                catalog,
                &iteration.selected,
                &iteration.device_attention_deltas,
                run.sparse_batch_graph_bucket,
                run.quantization_recipe.as_str(),
            )
        })?;
        self.stage_timing.record_numeric(layer_id, numeric_start);
        self.capture_target_device_hidden_tap(layer_id + 1)
    }

    fn start_sparse_layer(
        &mut self,
        catalog: &TensorCatalog,
        layer_id: usize,
        scheduler_timing: bool,
    ) -> Result<SchedulerSparseTcpPendingApply> {
        let prepared = self.prepare_sparse_layer(
            catalog,
            layer_id,
            scheduler_timing,
            self.sparse_batch_graph_bucket,
        )?;
        self.start_prepared_sparse_layer(catalog, layer_id, scheduler_timing, prepared, true)
    }

    fn prepare_sparse_layer(
        &mut self,
        catalog: &TensorCatalog,
        layer_id: usize,
        scheduler_timing: bool,
        graph_bucket: GraphBucket,
    ) -> Result<SchedulerSparseTcpPreparedDispatch> {
        let iteration = self
            .apply_attention_layer(catalog, layer_id, scheduler_timing)
            .with_context(|| {
                format!(
                    "applying batched scheduler sparse attention for request {} layer {layer_id}",
                    self.shape.request_id
                )
            })?;
        anyhow::ensure!(
            self.numeric_progression
                .can_pipeline_selected_sparse_tcp_batched(layer_id, &iteration.selected),
            "batched scheduler request {} cannot pipeline sparse layer {layer_id}",
            self.shape.request_id
        );
        let numeric_start = scheduler_timing.then(Instant::now);
        let prepared = self
            .with_buffer_bank(|run| {
                run.numeric_progression
                    .prepare_apply_selected_sparse_tcp_batched(
                        layer_id,
                        catalog,
                        &iteration.selected,
                        &iteration.device_attention_deltas,
                        graph_bucket,
                        run.quantization_recipe.as_str(),
                    )
            })?
            .with_context(|| {
                format!(
                    "batched scheduler request {} produced no prepared sparse dispatch for layer {layer_id}",
                    self.shape.request_id
                )
            })?;
        self.stage_timing.record_numeric(layer_id, numeric_start);
        Ok(prepared)
    }

    fn start_prepared_sparse_layer(
        &mut self,
        catalog: &TensorCatalog,
        layer_id: usize,
        scheduler_timing: bool,
        prepared: SchedulerSparseTcpPreparedDispatch,
        start_dispatch: bool,
    ) -> Result<SchedulerSparseTcpPendingApply> {
        let numeric_start = scheduler_timing.then(Instant::now);
        let pending = self.with_buffer_bank(|run| {
            if start_dispatch {
                run.numeric_progression
                    .start_prepared_sparse_tcp_apply(catalog, prepared)
            } else {
                run.numeric_progression
                    .start_prepared_sparse_tcp_apply_without_dispatch(catalog, prepared)
            }
        })?;
        self.stage_timing.record_numeric(layer_id, numeric_start);
        Ok(pending)
    }

    fn try_start_sparse_cohort_dispatch(
        &mut self,
        prepared: &[&SchedulerSparseTcpPreparedDispatch],
    ) -> Result<Option<SchedulerSparseTcpCohortPendingDispatch>> {
        self.with_buffer_bank(|run| {
            run.numeric_progression
                .try_start_sparse_tcp_cohort_dispatch(prepared)
        })
    }

    fn finish_sparse_cohort_dispatch(
        &mut self,
        pending: SchedulerSparseTcpCohortPendingDispatch,
    ) -> Result<Vec<glmrt_transport::TcpProtocolV2HostBatchSetBf16PayloadDispatch>> {
        self.with_buffer_bank(|run| {
            run.numeric_progression
                .finish_sparse_tcp_cohort_dispatch(pending)
        })
    }

    fn record_sparse_cohort_member_dispatch(
        &mut self,
        pending: &SchedulerSparseTcpPendingApply,
        dispatch: &glmrt_transport::TcpProtocolV2HostBatchSetBf16PayloadDispatch,
    ) -> Result<()> {
        self.with_buffer_bank(|run| {
            run.numeric_progression
                .record_sparse_tcp_cohort_member_dispatch(pending, dispatch)
        })
    }

    fn finish_sparse_layer(
        &mut self,
        layer_id: usize,
        pending: SchedulerSparseTcpPendingApply,
    ) -> Result<()> {
        self.with_buffer_bank(|run| {
            run.numeric_progression
                .finish_apply_selected_sparse_tcp_batched(pending)
        })
        .with_context(|| {
            format!(
                "finishing batched scheduler sparse request {} layer {layer_id}",
                self.shape.request_id
            )
        })?;
        self.capture_target_device_hidden_tap(layer_id + 1)
    }

    fn capture_target_device_hidden_tap(&mut self, layer_id: usize) -> Result<()> {
        self.with_buffer_bank(|run| {
            run.numeric_progression
                .capture_target_device_hidden_tap(layer_id)
        })
    }

    fn finish(
        self,
        catalog: &TensorCatalog,
        kv_bytes_per_token: usize,
        scheduler_verbose_timing: bool,
    ) -> Result<RealFullSchedulerDeviceExecution> {
        let buffer_bank = self.buffer_bank;
        with_coordinator_owned_device_buffer_bank(buffer_bank, || {
            let result = finish_stateful_scheduler_execution(
                catalog,
                &self.shape,
                self.store,
                self.device_kv,
                self.numeric_progression,
                self.counters,
                self.retain_final_target_device_hidden,
                self.execution_start,
                self.coordinator_graph_stats_before,
                scheduler_verbose_timing,
                kv_bytes_per_token,
            )?;
            Ok(RealFullSchedulerDeviceExecution {
                report: result.0,
                sparse_tcp_dispatch: result.1.context(
                    "batched scheduler sparse TCP dispatch probe missing after device-hidden execution",
                )?,
                final_target_device_hidden: result.2,
                target_device_hidden_taps: result.3,
            })
        })
    }
}

enum BatchedSparsePairPending {
    Independent {
        member_a: SchedulerSparseTcpPendingApply,
        member_b: SchedulerSparseTcpPendingApply,
    },
    Cohort {
        member_a: SchedulerSparseTcpPendingApply,
        member_b: SchedulerSparseTcpPendingApply,
        dispatch: SchedulerSparseTcpCohortPendingDispatch,
    },
}

fn start_batched_sparse_pair(
    runs: &mut [BatchedLiveSchedulerRun<'_>],
    catalog: &TensorCatalog,
    layer_id: usize,
    scheduler_timing: bool,
) -> Result<BatchedSparsePairPending> {
    let [run_a, run_b] = runs else {
        anyhow::bail!(
            "batched sparse cohort requires two runs, got {}",
            runs.len()
        );
    };
    let combined_rows = run_a
        .sparse_rows()
        .checked_add(run_b.sparse_rows())
        .context("batched sparse cohort row count overflow")?;
    let graph_bucket = GraphBucket::new(combined_rows);
    let prepared_a =
        run_a.prepare_sparse_layer(catalog, layer_id, scheduler_timing, graph_bucket)?;
    let prepared_b =
        run_b.prepare_sparse_layer(catalog, layer_id, scheduler_timing, graph_bucket)?;
    let dispatch = run_a.try_start_sparse_cohort_dispatch(&[&prepared_a, &prepared_b])?;
    if let Some(dispatch) = dispatch {
        let member_a = run_a.start_prepared_sparse_layer(
            catalog,
            layer_id,
            scheduler_timing,
            prepared_a,
            false,
        )?;
        let member_b = run_b.start_prepared_sparse_layer(
            catalog,
            layer_id,
            scheduler_timing,
            prepared_b,
            false,
        )?;
        Ok(BatchedSparsePairPending::Cohort {
            member_a,
            member_b,
            dispatch,
        })
    } else {
        let member_a = run_a.start_prepared_sparse_layer(
            catalog,
            layer_id,
            scheduler_timing,
            prepared_a,
            true,
        )?;
        let member_b = run_b.start_prepared_sparse_layer(
            catalog,
            layer_id,
            scheduler_timing,
            prepared_b,
            true,
        )?;
        Ok(BatchedSparsePairPending::Independent { member_a, member_b })
    }
}

fn finish_batched_sparse_pair(
    runs: &mut [BatchedLiveSchedulerRun<'_>],
    layer_id: usize,
    pending: BatchedSparsePairPending,
) -> Result<()> {
    let [run_a, run_b] = runs else {
        anyhow::bail!(
            "batched sparse cohort requires two runs, got {}",
            runs.len()
        );
    };
    match pending {
        BatchedSparsePairPending::Independent { member_a, member_b } => {
            run_a.finish_sparse_layer(layer_id, member_a)?;
            run_b.finish_sparse_layer(layer_id, member_b)
        }
        BatchedSparsePairPending::Cohort {
            mut member_a,
            mut member_b,
            dispatch,
        } => {
            let mut dispatches = run_a.finish_sparse_cohort_dispatch(dispatch)?;
            anyhow::ensure!(
                dispatches.len() == 2,
                "batched sparse cohort returned {} member dispatches",
                dispatches.len()
            );
            let dispatch_b = dispatches.pop().expect("cohort member B dispatch exists");
            let dispatch_a = dispatches.pop().expect("cohort member A dispatch exists");
            run_b.record_sparse_cohort_member_dispatch(&member_b, &dispatch_b)?;
            member_a.attach_completed_cohort_dispatch(dispatch_a)?;
            member_b.attach_completed_cohort_dispatch(dispatch_b)?;
            run_a.finish_sparse_layer(layer_id, member_a)?;
            run_b.finish_sparse_layer(layer_id, member_b)
        }
    }
}

pub(in crate::commands::real_full) struct RealFullSchedulerBatchedInput<'a> {
    pub(in crate::commands::real_full) shape: RealFullSchedulerExecutionShape,
    pub(in crate::commands::real_full) request_id_base: u64,
    pub(in crate::commands::real_full) state: &'a mut RealFullSchedulerExecutionState,
    pub(in crate::commands::real_full) buffer_bank: usize,
    pub(in crate::commands::real_full) retain_final_target_device_hidden: bool,
    pub(in crate::commands::real_full) target_device_hidden_tap_rows: usize,
}

pub(in crate::commands::real_full) fn real_full_scheduler_execution_for_batched_shapes_with_shared_sparse_tcp_and_state_device_hidden(
    kv_config: KvCacheConfig,
    catalog: &TensorCatalog,
    inputs: Vec<RealFullSchedulerBatchedInput<'_>>,
    dispatch_worker: Arc<RealFullSchedulerSparseTcpDispatchWorker>,
) -> Result<Vec<RealFullSchedulerDeviceExecution>> {
    anyhow::ensure!(
        (2..=8).contains(&inputs.len()),
        "batched recurrent scheduler requires 2..=8 requests, got {}",
        inputs.len()
    );
    let scheduler_timing = scheduler_execution_timing_enabled();
    let scheduler_verbose_timing = scheduler_execution_verbose_timing_enabled();
    let kv_bytes_per_token = kv_config.bytes_per_token();
    let mut runs = Vec::with_capacity(inputs.len());
    for input in inputs {
        let run = with_coordinator_owned_device_buffer_bank(input.buffer_bank, || {
            BatchedLiveSchedulerRun::new(
                catalog,
                input.shape,
                input.state,
                Arc::clone(&dispatch_worker),
                input.request_id_base,
                input.retain_final_target_device_hidden,
                input.target_device_hidden_tap_rows,
                input.buffer_bank,
            )
        })?;
        runs.push(run);
    }

    for layer_id in 0..GLM52_FIRST_K_DENSE_REPLACE {
        let layer_start = Instant::now();
        for run in &mut runs {
            run.apply_dense_layer(catalog, layer_id, scheduler_timing)?;
        }
        if scheduler_verbose_timing {
            let request_ids = runs
                .iter()
                .map(|run| run.shape.request_id.as_str())
                .collect::<Vec<_>>();
            eprintln!(
                "real_full_scheduler_batched_layer_timing requests={request_ids:?} layer_id={} elapsed_ms={:.3}",
                layer_id,
                elapsed_ms(layer_start),
            );
        }
    }

    let first_sparse_layer = GLM52_FIRST_K_DENSE_REPLACE;
    if matches!(runs.len(), 4 | 8) {
        let mut pending_pairs = Vec::with_capacity(runs.len() / 2);
        for pair_start in (0..runs.len()).step_by(2) {
            pending_pairs.push(Some(start_batched_sparse_pair(
                &mut runs[pair_start..pair_start + 2],
                catalog,
                first_sparse_layer,
                scheduler_timing,
            )?));
        }
        for layer_id in first_sparse_layer..GLM52_NUM_HIDDEN_LAYERS {
            let layer_start = Instant::now();
            let next_layer = layer_id + 1;
            for (pair_index, pending_pair) in pending_pairs.iter_mut().enumerate() {
                let pair_start = pair_index * 2;
                finish_batched_sparse_pair(
                    &mut runs[pair_start..pair_start + 2],
                    layer_id,
                    pending_pair.take().with_context(|| {
                        format!("batched scheduler sparse cohort {pair_index} wavefront is missing")
                    })?,
                )?;
                *pending_pair = if next_layer < GLM52_NUM_HIDDEN_LAYERS {
                    Some(start_batched_sparse_pair(
                        &mut runs[pair_start..pair_start + 2],
                        catalog,
                        next_layer,
                        scheduler_timing,
                    )?)
                } else {
                    None
                };
            }
            if scheduler_verbose_timing
                && (layer_id % 10 == 9 || layer_id + 3 >= GLM52_NUM_HIDDEN_LAYERS)
            {
                let request_ids = runs
                    .iter()
                    .map(|run| run.shape.request_id.as_str())
                    .collect::<Vec<_>>();
                eprintln!(
                    "real_full_scheduler_batched_wavefront_timing requests={request_ids:?} layer_id={} elapsed_ms={:.3}",
                    layer_id,
                    elapsed_ms(layer_start),
                );
            }
        }
    } else {
        let mut pending = Vec::with_capacity(runs.len());
        for run in &mut runs {
            pending.push(Some(run.start_sparse_layer(
                catalog,
                first_sparse_layer,
                scheduler_timing,
            )?));
        }
        for layer_id in first_sparse_layer..GLM52_NUM_HIDDEN_LAYERS {
            let layer_start = Instant::now();
            let next_layer = layer_id + 1;
            for (run, pending_apply) in runs.iter_mut().zip(&mut pending) {
                let request_id = run.shape.request_id.clone();
                run.finish_sparse_layer(
                    layer_id,
                    pending_apply.take().with_context(|| {
                        format!(
                            "batched scheduler request {request_id} sparse wavefront is missing"
                        )
                    })?,
                )?;
                *pending_apply = if next_layer < GLM52_NUM_HIDDEN_LAYERS {
                    Some(run.start_sparse_layer(catalog, next_layer, scheduler_timing)?)
                } else {
                    None
                };
            }
            if scheduler_verbose_timing
                && (layer_id % 10 == 9 || layer_id + 3 >= GLM52_NUM_HIDDEN_LAYERS)
            {
                let request_ids = runs
                    .iter()
                    .map(|run| run.shape.request_id.as_str())
                    .collect::<Vec<_>>();
                eprintln!(
                    "real_full_scheduler_batched_wavefront_timing requests={request_ids:?} layer_id={} elapsed_ms={:.3}",
                    layer_id,
                    elapsed_ms(layer_start),
                );
            }
        }
    }

    if scheduler_timing {
        for run in &runs {
            eprintln!(
                "real_full_scheduler_stage_summary request_id={} dense_attention_ms={:.3} dense_numeric_ms={:.3} sparse_attention_ms={:.3} sparse_numeric_ms={:.3} sparse_wavefront_ms={:.3}",
                run.shape.request_id,
                run.stage_timing.attention_ms[0],
                run.stage_timing.numeric_ms[0],
                run.stage_timing.attention_ms[1],
                run.stage_timing.numeric_ms[1],
                run.stage_timing.sparse_wavefront_ms,
            );
        }
    }
    runs.into_iter()
        .map(|run| run.finish(catalog, kv_bytes_per_token, scheduler_verbose_timing))
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn real_full_scheduler_execute_decode_layer_block(
    catalog: &TensorCatalog,
    block: SparkLayerBlock,
    source_request_id: u64,
    token_position: usize,
    hidden_bf16: &[u8],
    dispatch_worker: Arc<RealFullSchedulerSparseTcpDispatchWorker>,
    request_id_base: u64,
    state: &mut RealFullSchedulerExecutionState,
) -> Result<DeviceBf16Output> {
    real_full_scheduler_execute_decode_layer_block_inner(
        catalog,
        block,
        source_request_id,
        token_position,
        DecodeLayerBlockIngress::Host(hidden_bf16),
        dispatch_worker,
        request_id_base,
        state,
    )
}

#[allow(clippy::too_many_arguments, dead_code)]
pub(in crate::commands::real_full) fn real_full_scheduler_execute_decode_layer_block_device_input(
    catalog: &TensorCatalog,
    block: SparkLayerBlock,
    source_request_id: u64,
    token_position: usize,
    hidden: DeviceBf16Output,
    dispatch_worker: Arc<RealFullSchedulerSparseTcpDispatchWorker>,
    request_id_base: u64,
    state: &mut RealFullSchedulerExecutionState,
) -> Result<DeviceBf16Output> {
    real_full_scheduler_execute_decode_layer_block_inner(
        catalog,
        block,
        source_request_id,
        token_position,
        DecodeLayerBlockIngress::Device(hidden),
        dispatch_worker,
        request_id_base,
        state,
    )
}

enum DecodeLayerBlockIngress<'a> {
    Host(&'a [u8]),
    Device(DeviceBf16Output),
}

#[allow(clippy::too_many_arguments)]
fn real_full_scheduler_execute_decode_layer_block_inner(
    catalog: &TensorCatalog,
    block: SparkLayerBlock,
    source_request_id: u64,
    token_position: usize,
    hidden: DecodeLayerBlockIngress<'_>,
    dispatch_worker: Arc<RealFullSchedulerSparseTcpDispatchWorker>,
    request_id_base: u64,
    state: &mut RealFullSchedulerExecutionState,
) -> Result<DeviceBf16Output> {
    let request_id = format!("spark-layer-block-{source_request_id}");
    let shape = RealFullSchedulerExecutionShape {
        request_id: request_id.clone(),
        sequence_id: state.sequence_id.clone(),
        placement_version: format!(
            "spark-layer-block-{}-{}",
            block.start_layer, block.end_layer
        ),
        prefix_tokens: token_position,
        prefill_tokens: 0,
        prefill_chunk_tokens: 1,
        decode_rows: 1,
        mtp_rows: 0,
        mtp_accepted_rows: 0,
        prefill_token_ids: None,
        prefill_vision_embeddings: None,
        decode_token_ids: None,
        lm_head_sampling: RealFullLmHeadSamplingOptions::diagnostic(),
    };
    validate_scheduler_execution_shape(&shape)?;
    state.validate_shape(&shape)?;
    let policy = PrefillChunkPolicy {
        chunk_tokens: 1,
        max_prefill_tokens_per_iteration: 1,
        max_active_prefill_chunks: 1,
        decode_priority: true,
    };
    let graph_bucket = GraphBucket::decode();
    let quantization_recipe = ModelFacts::default().quantization_recipe;
    let numeric_shape = RealFullSchedulerNumericProgressionShape::from_execution_shape(&shape);
    let sparse_context = RealFullSchedulerSparseTcpRoutedMlpContext::with_dispatch_worker(
        1,
        dispatch_worker,
        request_id_base,
    )?;
    let mut progression = RealFullSchedulerNumericProgression::new(numeric_shape)
        .with_live_request()
        .with_sparse_tcp_routed_mlp(sparse_context);
    match hidden {
        DecodeLayerBlockIngress::Host(hidden_bf16) => progression
            .seed_external_decode_hidden(token_position, hidden_bf16)
            .context("seeding Spark layer-block host ingress hidden")?,
        DecodeLayerBlockIngress::Device(hidden) => progression
            .seed_external_decode_device_hidden(token_position, hidden)
            .context("seeding Spark layer-block device ingress hidden")?,
    }
    let mut counters = RealFullSchedulerExecutionCounters {
        layer_order_verified: true,
        sparse_expert_host_batch_routes_match_global: true,
        sparse_expert_host_batch_graph_counts_valid: true,
        sparse_expert_host_wire_envelopes_valid: true,
        ..Default::default()
    };
    for layer_id in block.start_layer..block.end_layer {
        let wave = LayerWave::decode(DecodeStep::new(
            request_id.as_str(),
            state.sequence_id.as_str(),
            LayerId(layer_id as u32),
            PositionId(token_position as u64),
            Some(state.reservation_id),
            Priority(0),
            shape.placement_version.as_str(),
        ));
        let iteration = real_full_apply_admitted_scheduler_iteration(
            &mut state.store,
            &policy,
            vec![wave],
            &[glmrt_core::LayerWaveMode::Decode],
            graph_bucket,
            &quantization_recipe,
            Some(0),
            false,
            &mut counters,
            &mut state.device_kv,
            &mut progression,
            catalog,
        )
        .with_context(|| format!("executing Spark-owned attention for layer {layer_id}"))?;
        progression
            .apply_selected(
                layer_id,
                catalog,
                &iteration.selected,
                &iteration.device_attention_deltas,
                graph_bucket,
                &quantization_recipe,
            )
            .with_context(|| format!("executing Spark-owned MLP for layer {layer_id}"))?;
    }

    progression
        .finish_live_request()?
        .final_decode_device_hidden
        .context("Spark layer block produced no decode hidden output")
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn real_full_scheduler_execute_prefill_decode_layer_block_device_input(
    catalog: &TensorCatalog,
    block: SparkLayerBlock,
    source_request_id: u64,
    token_start: usize,
    hidden: DeviceBf16Output,
    prefill_chunk_tokens: usize,
    dispatch_worker: Arc<RealFullSchedulerSparseTcpDispatchWorker>,
    request_id_base: u64,
    state: &mut RealFullSchedulerExecutionState,
) -> Result<DeviceBf16Output> {
    anyhow::ensure!(
        hidden.rows > 0 && hidden.values_per_row == GLM52_HIDDEN_SIZE,
        "Spark layer-block prefill/decode ingress must be Nx{}, got {}x{}",
        GLM52_HIDDEN_SIZE,
        hidden.rows,
        hidden.values_per_row
    );
    anyhow::ensure!(
        prefill_chunk_tokens > 0,
        "Spark layer-block prefill chunk size must be nonzero"
    );
    let request_id = format!("spark-layer-block-prefill-{source_request_id}");
    let prefill_tokens = hidden.rows - 1;
    let shape = RealFullSchedulerExecutionShape {
        request_id: request_id.clone(),
        sequence_id: state.sequence_id.clone(),
        placement_version: format!(
            "spark-layer-block-prefill-{}-{}",
            block.start_layer, block.end_layer
        ),
        prefix_tokens: token_start,
        prefill_tokens,
        prefill_chunk_tokens,
        decode_rows: 1,
        mtp_rows: 0,
        mtp_accepted_rows: 0,
        prefill_token_ids: None,
        prefill_vision_embeddings: None,
        decode_token_ids: None,
        lm_head_sampling: RealFullLmHeadSamplingOptions::diagnostic(),
    };
    validate_scheduler_execution_shape(&shape)?;
    state.validate_shape(&shape)?;
    let policy = PrefillChunkPolicy {
        chunk_tokens: prefill_chunk_tokens,
        max_prefill_tokens_per_iteration: prefill_chunk_tokens,
        max_active_prefill_chunks: 1,
        decode_priority: true,
    };
    let graph_bucket = shape.sparse_batch_graph_bucket();
    let quantization_recipe = ModelFacts::default().quantization_recipe;
    let sparse_iterations = prefill_tokens.div_ceil(prefill_chunk_tokens).max(1);
    let sparse_context = RealFullSchedulerSparseTcpRoutedMlpContext::with_dispatch_worker(
        sparse_iterations,
        dispatch_worker,
        request_id_base,
    )?;
    let numeric_shape = RealFullSchedulerNumericProgressionShape::from_execution_shape(&shape);
    let mut progression = RealFullSchedulerNumericProgression::new(numeric_shape)
        .with_live_request()
        .with_sparse_tcp_routed_mlp(sparse_context);
    progression
        .seed_external_prefill_decode_device_hidden(token_start, hidden)
        .context("seeding Spark layer-block prefill/decode device ingress")?;
    let mut counters = RealFullSchedulerExecutionCounters {
        layer_order_verified: true,
        sparse_expert_host_batch_routes_match_global: true,
        sparse_expert_host_batch_graph_counts_valid: true,
        sparse_expert_host_wire_envelopes_valid: true,
        ..Default::default()
    };

    for layer_id in block.start_layer..block.end_layer {
        let layer = LayerId(layer_id as u32);
        let mut prefill_chunks = plan_scheduler_prefill_chunks(
            &shape,
            layer,
            state.reservation_id,
            Priority(0),
            &policy,
        );
        let decode = LayerWave::decode(DecodeStep::new(
            request_id.as_str(),
            state.sequence_id.as_str(),
            layer,
            PositionId((token_start + prefill_tokens) as u64),
            Some(state.reservation_id),
            Priority(0),
            shape.placement_version.as_str(),
        ));
        let mut wave_groups = Vec::with_capacity(prefill_chunks.len().max(1));
        if prefill_chunks.is_empty() {
            wave_groups.push(vec![decode]);
        } else {
            let chunk_count = prefill_chunks.len();
            for (chunk_index, prefill_chunk) in prefill_chunks.drain(..).enumerate() {
                let final_chunk = chunk_index + 1 == chunk_count;
                let mut waves = Vec::with_capacity(1 + usize::from(final_chunk));
                waves.push(LayerWave::prefill(prefill_chunk));
                if final_chunk {
                    waves.push(decode.clone());
                }
                wave_groups.push(waves);
            }
        }
        let chunk_count = wave_groups.len();
        for (chunk_index, waves) in wave_groups.into_iter().enumerate() {
            let final_chunk = chunk_index + 1 == chunk_count;
            let expected_modes = waves.iter().map(|wave| wave.mode).collect::<Vec<_>>();
            let mut iteration_policy = policy.clone();
            iteration_policy.decode_priority = !final_chunk;
            let iteration = real_full_apply_admitted_scheduler_iteration(
                &mut state.store,
                &iteration_policy,
                waves,
                &expected_modes,
                graph_bucket,
                &quantization_recipe,
                Some(0),
                false,
                &mut counters,
                &mut state.device_kv,
                &mut progression,
                catalog,
            )
            .with_context(|| {
                format!(
                    "executing Spark layer-block prefill/decode wave {}/{} for layer {layer_id}",
                    chunk_index + 1,
                    chunk_count
                )
            })?;
            progression
                .apply_selected(
                    layer_id,
                    catalog,
                    &iteration.selected,
                    &iteration.device_attention_deltas,
                    graph_bucket,
                    &quantization_recipe,
                )
                .with_context(|| {
                    format!(
                        "applying Spark layer-block prefill/decode wave {}/{} for layer {layer_id}",
                        chunk_index + 1,
                        chunk_count
                    )
                })?;
        }
    }

    progression
        .finish_live_request()?
        .final_decode_device_hidden
        .context("Spark prefill/decode layer block produced no final device hidden output")
}

pub(in crate::commands::real_full) fn real_full_scheduler_precapture_layer_block_attention(
    catalog: &TensorCatalog,
    block: SparkLayerBlock,
    kv_config: KvCacheConfig,
) -> Result<()> {
    let sequence_id = format!(
        "spark-layer-block-attention-capture-{}-{}",
        block.start_layer, block.end_layer
    );
    let mut state = RealFullSchedulerExecutionState::new(kv_config, sequence_id.clone())?;
    let policy = PrefillChunkPolicy {
        chunk_tokens: 1,
        max_prefill_tokens_per_iteration: 1,
        max_active_prefill_chunks: 1,
        decode_priority: true,
    };
    let graph_bucket = GraphBucket::decode();
    let quantization_recipe = ModelFacts::default().quantization_recipe;
    let hidden_bf16 = vec![0_u8; GLM52_HIDDEN_SIZE * std::mem::size_of::<u16>()];
    let sizing_worker = Arc::new(RealFullSchedulerSparseTcpDispatchWorker::new_zero_output()?);
    let sizing_output = real_full_scheduler_execute_decode_layer_block(
        catalog,
        block,
        0,
        0,
        &hidden_bf16,
        Arc::clone(&sizing_worker),
        0,
        &mut state,
    )
    .context("sizing Spark layer-block decode workspaces before attention capture")?;
    drop(sizing_output);
    drop(sizing_worker);
    let mut counters = RealFullSchedulerExecutionCounters {
        layer_order_verified: true,
        sparse_expert_host_batch_routes_match_global: true,
        sparse_expert_host_batch_graph_counts_valid: true,
        sparse_expert_host_wire_envelopes_valid: true,
        ..Default::default()
    };

    // Full token-zero execution sizes every shared graph workspace and seeds KV.
    // Capture compressed token-one attention last so later startup work cannot
    // invalidate its fixed scratch pointers.
    let token_position = 1;
    for layer_id in block.start_layer..block.end_layer {
        let request_id = format!("spark-layer-block-attention-capture-{layer_id}");
        let shape = RealFullSchedulerExecutionShape {
            request_id: request_id.clone(),
            sequence_id: sequence_id.clone(),
            placement_version: "spark-layer-block-attention-capture".to_owned(),
            prefix_tokens: token_position,
            prefill_tokens: 0,
            prefill_chunk_tokens: 1,
            decode_rows: 1,
            mtp_rows: 0,
            mtp_accepted_rows: 0,
            prefill_token_ids: None,
            prefill_vision_embeddings: None,
            decode_token_ids: None,
            lm_head_sampling: RealFullLmHeadSamplingOptions::diagnostic(),
        };
        validate_scheduler_execution_shape(&shape)?;
        state.validate_shape(&shape)?;
        let numeric_shape = RealFullSchedulerNumericProgressionShape::from_execution_shape(&shape);
        let mut progression =
            RealFullSchedulerNumericProgression::new(numeric_shape).with_live_request();
        progression
            .seed_external_decode_hidden(token_position, &hidden_bf16)
            .with_context(|| {
                format!("seeding Spark layer-block attention capture for layer {layer_id}")
            })?;
        let wave = LayerWave::decode(DecodeStep::new(
            request_id.as_str(),
            sequence_id.as_str(),
            LayerId(layer_id as u32),
            PositionId(token_position as u64),
            Some(state.reservation_id),
            Priority(0),
            shape.placement_version.as_str(),
        ));
        let iteration = real_full_apply_admitted_scheduler_iteration(
            &mut state.store,
            &policy,
            vec![wave],
            &[glmrt_core::LayerWaveMode::Decode],
            graph_bucket,
            &quantization_recipe,
            Some(0),
            false,
            &mut counters,
            &mut state.device_kv,
            &mut progression,
            catalog,
        )
        .with_context(|| format!("capturing Spark-owned attention for layer {layer_id}"))?;
        anyhow::ensure!(
            iteration.device_attention_deltas.len() == 1,
            "Spark layer-block attention capture for layer {layer_id} produced {} device deltas",
            iteration.device_attention_deltas.len()
        );
    }
    Ok(())
}

fn real_full_scheduler_execution_for_shape_with_sparse_tcp_context(
    kv_config: KvCacheConfig,
    catalog: &TensorCatalog,
    shape: RealFullSchedulerExecutionShape,
    context: RealFullSchedulerSparseTcpRoutedMlpContext,
    state: Option<&mut RealFullSchedulerExecutionState>,
) -> Result<(
    RealFullSchedulerExecutionDryRun,
    RealFullSchedulerSparseTcpDispatchProbe,
)> {
    let (report, probe, _, _) = real_full_scheduler_execution_for_shape_inner(
        kv_config,
        catalog,
        shape,
        Some(context),
        state,
        false,
        false,
        0,
    )?;
    let probe = probe
        .context("scheduler sparse TCP residual dispatch probe missing after TCP execution")?;
    Ok((report, probe))
}

fn scheduler_sparse_tcp_iterations_per_sparse_layer(
    shape: &RealFullSchedulerExecutionShape,
) -> usize {
    let prefill_chunks = shape.prefill_tokens.div_ceil(shape.prefill_chunk_tokens);
    let sparse_rows = shape
        .prefill_tokens
        .saturating_add(shape.decode_rows)
        .saturating_add(shape.mtp_rows);
    if prefill_chunks >= 2 {
        // The bounded >8K path dispatches one ordinary Spark batch for each
        // physical prefill chunk while wavefronting those chunks across all
        // target layers. It does not use the 256-row rolling sparse pack
        // planner, so its completion probe must count physical chunks too.
        if bounded_long_prefill_wavefront_required(sparse_rows) {
            return prefill_chunks;
        }
        if let Some(dispatches) = rolling_sparse_dispatches_per_layer_for_rows(sparse_rows) {
            return dispatches;
        }
    }
    if prefill_chunks > 0 {
        prefill_chunks
    } else {
        usize::from(shape.decode_rows > 0 || shape.mtp_rows > 0)
    }
}

fn plan_scheduler_decode_mtp_waves(
    shape: &RealFullSchedulerExecutionShape,
    layer: LayerId,
    reservation_id: u64,
) -> Vec<LayerWave> {
    let mut waves = (0..shape.decode_rows)
        .map(|decode_offset| {
            LayerWave::decode(DecodeStep::new(
                shape.request_id.as_str(),
                shape.sequence_id.as_str(),
                layer,
                PositionId((shape.decode_token_start() + decode_offset) as u64),
                Some(reservation_id),
                Priority(0),
                shape.placement_version.as_str(),
            ))
        })
        .collect::<Vec<_>>();
    if shape.mtp_rows > 0 {
        waves.push(LayerWave::mtp_verify(MtpVerifyBlock::new(
            shape.request_id.as_str(),
            shape.sequence_id.as_str(),
            layer,
            PositionId(shape.mtp_token_start() as u64),
            shape.mtp_rows,
            Some(reservation_id),
            Priority(0),
            GraphBucket::new(shape.mtp_rows),
            shape.placement_version.as_str(),
        )));
    }
    waves
}

fn plan_scheduler_chunk_layer_waves(
    shape: &RealFullSchedulerExecutionShape,
    layer: LayerId,
    reservation_id: u64,
    policy: &PrefillChunkPolicy,
) -> Result<Vec<Vec<LayerWave>>> {
    let prefill_chunks =
        plan_scheduler_prefill_chunks(shape, layer, reservation_id, Priority(0), policy);
    let chunk_count = prefill_chunks.len();
    let mut chunk_waves = Vec::with_capacity(chunk_count);
    for (chunk_index, prefill_chunk) in prefill_chunks.into_iter().enumerate() {
        let final_prefill_chunk = chunk_index + 1 == chunk_count;
        let mut waves = Vec::with_capacity(
            1 + usize::from(final_prefill_chunk)
                * (shape.decode_rows + usize::from(shape.mtp_rows > 0)),
        );
        waves.push(LayerWave::prefill(prefill_chunk));
        if final_prefill_chunk {
            waves.extend(plan_scheduler_decode_mtp_waves(
                shape,
                layer,
                reservation_id,
            ));
        }
        chunk_waves.push(waves);
    }
    Ok(chunk_waves)
}

#[allow(clippy::too_many_arguments)]
fn start_scheduler_sparse_wavefront_dispatch(
    store: &mut KvCacheBackingStore,
    policy: &PrefillChunkPolicy,
    waves: Vec<LayerWave>,
    final_prefill_chunk: bool,
    sparse_batch_graph_bucket: GraphBucket,
    quantization_recipe: &str,
    mtp_accepted_rows: Option<usize>,
    record_sparse_host_partition: bool,
    counters: &mut RealFullSchedulerExecutionCounters,
    device_kv: &mut RealFullDeviceKvExecutionMirror,
    numeric_progression: &mut RealFullSchedulerNumericProgression,
    catalog: &TensorCatalog,
    collect_timing: bool,
) -> Result<SchedulerSparseWavefrontStartedDispatch> {
    let layer_id = waves
        .first()
        .context("sparse wavefront dispatch has no waves")?
        .layer_id
        .0 as usize;
    let expected_modes = waves.iter().map(|wave| wave.mode).collect::<Vec<_>>();
    let mut iteration_policy = policy.clone();
    if final_prefill_chunk {
        iteration_policy.decode_priority = false;
    }
    let attention_started = collect_timing.then(Instant::now);
    let iteration = real_full_apply_admitted_scheduler_iteration(
        store,
        &iteration_policy,
        waves,
        &expected_modes,
        sparse_batch_graph_bucket,
        quantization_recipe,
        mtp_accepted_rows,
        record_sparse_host_partition,
        counters,
        device_kv,
        numeric_progression,
        catalog,
    )
    .with_context(|| {
        format!(
            "applying admitted sparse wavefront chunk final={final_prefill_chunk} for layer {layer_id}"
        )
    })?;
    anyhow::ensure!(
        numeric_progression.can_pipeline_selected_sparse_tcp_batched(layer_id, &iteration.selected),
        "sparse wavefront chunk for layer {layer_id} is not eligible for pipelined dispatch"
    );
    let attention_ms = attention_started.map(elapsed_ms).unwrap_or(0.0);
    let numeric_started = collect_timing.then(Instant::now);
    let apply = numeric_progression
        .start_apply_selected_sparse_tcp_batched(
            layer_id,
            catalog,
            &iteration.selected,
            &iteration.device_attention_deltas,
            sparse_batch_graph_bucket,
            quantization_recipe,
        )
        .with_context(|| {
            format!(
                "starting sparse wavefront chunk final={final_prefill_chunk} for layer {layer_id}"
            )
        })?
        .context("eligible sparse wavefront dispatch did not return a pending apply")?;
    Ok(SchedulerSparseWavefrontStartedDispatch {
        apply,
        attention_ms,
        numeric_ms: numeric_started.map(elapsed_ms).unwrap_or(0.0),
    })
}

#[allow(clippy::too_many_arguments)]
fn prepare_scheduler_sparse_wavefront_dispatch(
    store: &mut KvCacheBackingStore,
    policy: &PrefillChunkPolicy,
    waves: Vec<LayerWave>,
    final_prefill_chunk: bool,
    sparse_batch_graph_bucket: GraphBucket,
    quantization_recipe: &str,
    mtp_accepted_rows: Option<usize>,
    record_sparse_host_partition: bool,
    counters: &mut RealFullSchedulerExecutionCounters,
    device_kv: &mut RealFullDeviceKvExecutionMirror,
    numeric_progression: &mut RealFullSchedulerNumericProgression,
    catalog: &TensorCatalog,
    collect_timing: bool,
) -> Result<SchedulerSparseRollingPreparedDispatch> {
    let layer_id = waves
        .first()
        .context("rolling sparse wavefront dispatch has no waves")?
        .layer_id
        .0 as usize;
    let expected_modes = waves.iter().map(|wave| wave.mode).collect::<Vec<_>>();
    let mut iteration_policy = policy.clone();
    if final_prefill_chunk {
        iteration_policy.decode_priority = false;
    }
    let attention_started = collect_timing.then(Instant::now);
    let iteration = real_full_apply_admitted_scheduler_iteration(
        store,
        &iteration_policy,
        waves,
        &expected_modes,
        sparse_batch_graph_bucket,
        quantization_recipe,
        mtp_accepted_rows,
        record_sparse_host_partition,
        counters,
        device_kv,
        numeric_progression,
        catalog,
    )
    .with_context(|| {
        format!(
            "applying admitted rolling sparse chunk final={final_prefill_chunk} for layer {layer_id}"
        )
    })?;
    anyhow::ensure!(
        numeric_progression.can_pipeline_selected_sparse_tcp_batched(layer_id, &iteration.selected),
        "rolling sparse chunk for layer {layer_id} is not eligible for pipelined dispatch"
    );
    let attention_ms = attention_started.map(elapsed_ms).unwrap_or(0.0);
    let numeric_started = collect_timing.then(Instant::now);
    let dispatch = numeric_progression
        .prepare_apply_selected_sparse_tcp_batched(
            layer_id,
            catalog,
            &iteration.selected,
            &iteration.device_attention_deltas,
            sparse_batch_graph_bucket,
            quantization_recipe,
        )?
        .context("eligible rolling sparse chunk did not return a prepared dispatch")?;
    Ok(SchedulerSparseRollingPreparedDispatch {
        dispatch,
        attention_ms,
        numeric_ms: numeric_started.map(elapsed_ms).unwrap_or(0.0),
    })
}

#[derive(Clone, Copy)]
struct SchedulerSparseWavefrontTask {
    layer_offset: usize,
    chunk_index: usize,
}

struct SchedulerSparseRollingPreparedDispatch {
    dispatch: SchedulerSparseTcpPreparedDispatch,
    attention_ms: f64,
    numeric_ms: f64,
}

struct SchedulerSparseWavefrontPending {
    task: SchedulerSparseWavefrontTask,
    apply: SchedulerSparseTcpPendingApply,
}

struct SchedulerSparseWavefrontStartedDispatch {
    apply: SchedulerSparseTcpPendingApply,
    attention_ms: f64,
    numeric_ms: f64,
}

const MAX_ACTIVE_BOUNDED_LONG_PREFILL_CHUNKS: usize = 16;

#[allow(clippy::too_many_arguments)]
fn execute_scheduler_bounded_long_prefill_wavefront(
    store: &mut KvCacheBackingStore,
    shape: &RealFullSchedulerExecutionShape,
    reservation_id: u64,
    policy: &PrefillChunkPolicy,
    sparse_batch_graph_bucket: GraphBucket,
    quantization_recipe: &str,
    record_sparse_host_partition: bool,
    counters: &mut RealFullSchedulerExecutionCounters,
    device_kv: &mut RealFullDeviceKvExecutionMirror,
    numeric_progression: &mut RealFullSchedulerNumericProgression,
    catalog: &TensorCatalog,
    target_device_hidden_tap_rows: usize,
    summary_timing: bool,
) -> Result<()> {
    let wavefront_start = Instant::now();
    let base_chunks =
        plan_scheduler_prefill_chunks(shape, LayerId(0), reservation_id, Priority(0), policy);
    let chunk_count = base_chunks.len();
    let total_rows = shape
        .prefill_tokens
        .checked_add(shape.decode_rows)
        .and_then(|rows| rows.checked_add(shape.mtp_rows))
        .context("bounded long-prefill total row count overflow")?;
    anyhow::ensure!(
        chunk_count >= 2 && bounded_long_prefill_wavefront_required(total_rows),
        "bounded long-prefill wavefront requires a qualified multi-chunk shape"
    );
    let chunk_rows = base_chunks
        .iter()
        .map(|chunk| {
            let row_start = (chunk.token_start.0 as usize)
                .checked_sub(shape.prefix_tokens)
                .context("bounded long-prefill chunk starts before request suffix")?;
            Ok((row_start, chunk.token_count))
        })
        .collect::<Result<Vec<_>>>()?;
    let prefill_token_ids = shape
        .prefill_token_ids
        .as_deref()
        .context("bounded long-prefill wavefront requires prompt token IDs")?;
    let retained_tap_rows = target_device_hidden_tap_rows.min(total_rows);
    let tap_row_start = total_rows.saturating_sub(retained_tap_rows);
    let tap_suffix_chunk_start = if retained_tap_rows == 0 {
        chunk_count
    } else if tap_row_start >= shape.prefill_tokens {
        chunk_count - 1
    } else {
        chunk_rows
            .iter()
            .position(|(row_start, row_count)| row_start + row_count > tap_row_start)
            .context("bounded long-prefill target tap suffix has no owning chunk")?
    };
    let layer_count = GLM52_NUM_HIDDEN_LAYERS;
    let mut started = vec![vec![false; chunk_count]; layer_count];
    let mut finished = vec![vec![false; chunk_count]; layer_count];
    let mut planned_layer_waves = std::iter::repeat_with(|| None)
        .take(layer_count)
        .collect::<Vec<Option<Vec<Option<Vec<LayerWave>>>>>>();
    let mut active_chunks = vec![false; chunk_count];
    let mut active_chunk_count = 0_usize;
    let mut pending = VecDeque::<SchedulerSparseWavefrontPending>::with_capacity(
        MAX_PENDING_SPARSE_WAVEFRONT_DISPATCHES,
    );
    let task_count = layer_count
        .checked_mul(chunk_count)
        .context("bounded long-prefill wavefront task count overflow")?;
    let mut completed_tasks = 0_usize;
    let mut peak_active_chunks = 0_usize;
    let mut idle_poll_started = None::<Instant>;

    while completed_tasks < task_count {
        while pending.len() < MAX_PENDING_SPARSE_WAVEFRONT_DISPATCHES {
            let mut next_task = None::<SchedulerSparseWavefrontTask>;
            for layer_offset in 0..layer_count {
                for chunk_index in 0..chunk_count {
                    let own_previous_layer_finished =
                        layer_offset == 0 || finished[layer_offset - 1][chunk_index];
                    let tap_checkpoint_layer =
                        dspark_target_hidden_tap_layer_ids().contains(&(layer_offset + 1));
                    let suffix_tap_chunk = chunk_index >= tap_suffix_chunk_start;
                    // A dSpark tap is one exact target-layer boundary, not the
                    // latest hidden state for each row. Keep only the small
                    // retained suffix in lock-step at its five checkpoints:
                    // each suffix chunk finishes the checkpoint in order, and
                    // none advances until the final suffix chunk has been
                    // captured. The rest of the long prompt remains fully
                    // wavefronted.
                    let previous_chunk_ready = if tap_checkpoint_layer && suffix_tap_chunk {
                        chunk_index == tap_suffix_chunk_start
                            || finished[layer_offset][chunk_index - 1]
                    } else {
                        chunk_index == 0 || started[layer_offset][chunk_index - 1]
                    };
                    let prior_tap_captured = if layer_offset > 0
                        && dspark_target_hidden_tap_layer_ids().contains(&layer_offset)
                        && suffix_tap_chunk
                    {
                        finished[layer_offset - 1][tap_suffix_chunk_start..]
                            .iter()
                            .all(|finished| *finished)
                    } else {
                        true
                    };
                    let chunk_slot_available = layer_offset != 0
                        || active_chunks[chunk_index]
                        || active_chunk_count < MAX_ACTIVE_BOUNDED_LONG_PREFILL_CHUNKS;
                    if started[layer_offset][chunk_index]
                        || !own_previous_layer_finished
                        || !previous_chunk_ready
                        || !prior_tap_captured
                        || !chunk_slot_available
                    {
                        continue;
                    }
                    let candidate = SchedulerSparseWavefrontTask {
                        layer_offset,
                        chunk_index,
                    };
                    let candidate_key = (
                        candidate.layer_offset + candidate.chunk_index,
                        candidate.layer_offset,
                        candidate.chunk_index,
                    );
                    if next_task.as_ref().is_none_or(|current| {
                        candidate_key
                            < (
                                current.layer_offset + current.chunk_index,
                                current.layer_offset,
                                current.chunk_index,
                            )
                    }) {
                        next_task = Some(candidate);
                    }
                }
            }
            let Some(task) = next_task else {
                break;
            };
            let layer_id = task.layer_offset;
            if planned_layer_waves[layer_id].is_none() {
                let layer_waves = plan_scheduler_chunk_layer_waves(
                    shape,
                    LayerId(layer_id as u32),
                    reservation_id,
                    policy,
                )?;
                anyhow::ensure!(
                    layer_waves.len() == chunk_count,
                    "bounded long-prefill layer {layer_id} planned {} chunks, expected {chunk_count}",
                    layer_waves.len()
                );
                planned_layer_waves[layer_id] = Some(layer_waves.into_iter().map(Some).collect());
            }
            let waves = planned_layer_waves[layer_id]
                .as_mut()
                .and_then(|layer_waves| layer_waves.get_mut(task.chunk_index))
                .and_then(Option::take)
                .with_context(|| {
                    format!(
                        "bounded long-prefill layer {layer_id} chunk {} was not planned exactly once",
                        task.chunk_index
                    )
                })?;
            if layer_id == 0 && !active_chunks[task.chunk_index] {
                let (row_start, row_count) = chunk_rows[task.chunk_index];
                numeric_progression.seed_bounded_prefill_token_embedding_chunk(
                    catalog,
                    &prefill_token_ids[row_start..row_start + row_count],
                    row_start,
                    shape
                        .prefill_vision_embeddings
                        .as_deref()
                        .map(Vec::as_slice),
                )?;
                active_chunks[task.chunk_index] = true;
                active_chunk_count += 1;
                peak_active_chunks = peak_active_chunks.max(active_chunk_count);
            }
            let final_prefill_chunk = task.chunk_index + 1 == chunk_count;
            started[layer_id][task.chunk_index] = true;
            if layer_id < GLM52_FIRST_K_DENSE_REPLACE {
                let expected_modes = waves.iter().map(|wave| wave.mode).collect::<Vec<_>>();
                let mut iteration_policy = policy.clone();
                if final_prefill_chunk {
                    iteration_policy.decode_priority = false;
                }
                let iteration = real_full_apply_admitted_scheduler_iteration(
                    store,
                    &iteration_policy,
                    waves,
                    &expected_modes,
                    sparse_batch_graph_bucket,
                    quantization_recipe,
                    shape.mtp_resolution_accepted_rows(),
                    record_sparse_host_partition,
                    counters,
                    device_kv,
                    numeric_progression,
                    catalog,
                )
                .with_context(|| {
                    format!(
                        "applying bounded long-prefill dense layer {layer_id} chunk {}",
                        task.chunk_index
                    )
                })?;
                numeric_progression
                    .apply_selected(
                        layer_id,
                        catalog,
                        &iteration.selected,
                        &iteration.device_attention_deltas,
                        sparse_batch_graph_bucket,
                        quantization_recipe,
                    )
                    .with_context(|| {
                        format!(
                            "applying bounded long-prefill dense numeric layer {layer_id} chunk {}",
                            task.chunk_index
                        )
                    })?;
                finished[layer_id][task.chunk_index] = true;
                completed_tasks += 1;
                if final_prefill_chunk {
                    numeric_progression
                        .capture_target_device_hidden_tap(layer_id + 1)
                        .with_context(|| {
                            format!(
                                "capturing bounded long-prefill target tap after layer {layer_id}"
                            )
                        })?;
                }
                continue;
            }
            let started_dispatch = start_scheduler_sparse_wavefront_dispatch(
                store,
                policy,
                waves,
                final_prefill_chunk,
                sparse_batch_graph_bucket,
                quantization_recipe,
                shape.mtp_resolution_accepted_rows(),
                record_sparse_host_partition,
                counters,
                device_kv,
                numeric_progression,
                catalog,
                false,
            )?;
            pending.push_back(SchedulerSparseWavefrontPending {
                task,
                apply: started_dispatch.apply,
            });
        }

        if completed_tasks == task_count {
            break;
        }
        anyhow::ensure!(
            !pending.is_empty(),
            "bounded long-prefill wavefront stalled without pending work"
        );
        if let Some(non_streaming_index) = pending
            .iter()
            .position(|pending_task| !pending_task.apply.supports_incremental_stream())
        {
            let pending_task = pending
                .remove(non_streaming_index)
                .context("bounded non-streaming sparse task index is out of range")?;
            numeric_progression
                .finish_apply_selected_sparse_tcp_batched(pending_task.apply)
                .with_context(|| {
                    format!(
                        "finishing bounded long-prefill layer {} chunk {}",
                        pending_task.task.layer_offset, pending_task.task.chunk_index
                    )
                })?;
            let task = pending_task.task;
            finished[task.layer_offset][task.chunk_index] = true;
            completed_tasks += 1;
            if task.chunk_index + 1 == chunk_count {
                numeric_progression.capture_target_device_hidden_tap(task.layer_offset + 1)?;
            }
            if task.layer_offset + 1 == layer_count {
                let (row_start, row_count) = chunk_rows[task.chunk_index];
                numeric_progression
                    .release_bounded_prefill_device_hidden_chunk(row_start, row_count)?;
                active_chunks[task.chunk_index] = false;
                active_chunk_count -= 1;
            }
            idle_poll_started = None;
            continue;
        }

        let mut completed_pending_index = None;
        let mut made_progress = false;
        for pending_index in 0..pending.len() {
            let pending_task = pending
                .get_mut(pending_index)
                .context("bounded long-prefill pending index is out of range")?;
            let progress = numeric_progression
                .try_progress_apply_selected_sparse_tcp_batched(&mut pending_task.apply)
                .with_context(|| {
                    format!(
                        "polling bounded long-prefill layer {} chunk {}",
                        pending_task.task.layer_offset, pending_task.task.chunk_index
                    )
                })?;
            let Some(progress) = progress else {
                continue;
            };
            made_progress = true;
            if progress.dispatch_complete {
                completed_pending_index = Some(pending_index);
                break;
            }
        }
        if let Some(completed_pending_index) = completed_pending_index {
            let pending_task = pending
                .remove(completed_pending_index)
                .context("completed bounded long-prefill task index is out of range")?;
            let task = pending_task.task;
            finished[task.layer_offset][task.chunk_index] = true;
            completed_tasks += 1;
            if task.chunk_index + 1 == chunk_count {
                numeric_progression.capture_target_device_hidden_tap(task.layer_offset + 1)?;
            }
            if task.layer_offset + 1 == layer_count {
                let (row_start, row_count) = chunk_rows[task.chunk_index];
                numeric_progression
                    .release_bounded_prefill_device_hidden_chunk(row_start, row_count)?;
                active_chunks[task.chunk_index] = false;
                active_chunk_count -= 1;
            }
            idle_poll_started = None;
            continue;
        }
        if made_progress {
            idle_poll_started = None;
            continue;
        }
        let idle_started = idle_poll_started.get_or_insert_with(Instant::now);
        if idle_started.elapsed() < SPARSE_WAVEFRONT_BUSY_POLL {
            std::hint::spin_loop();
        } else {
            std::thread::sleep(SPARSE_WAVEFRONT_IDLE_POLL);
        }
    }

    anyhow::ensure!(
        pending.is_empty()
            && active_chunk_count == 0
            && finished.iter().flatten().all(|finished| *finished),
        "bounded long-prefill wavefront left unfinished state"
    );
    if summary_timing {
        eprintln!(
            "real_full_scheduler_bounded_long_prefill_summary request_id={} rows={} chunks={} layers={} peak_active_chunks={} total_ms={:.3}",
            shape.request_id,
            shape.prefill_tokens,
            chunk_count,
            layer_count,
            peak_active_chunks,
            elapsed_ms(wavefront_start),
        );
    }
    Ok(())
}

fn next_ready_sparse_wavefront_task(
    started: &[Vec<bool>],
    finished: &[Vec<bool>],
) -> Option<SchedulerSparseWavefrontTask> {
    let layer_count = started.len();
    let chunk_count = started.first().map_or(0, Vec::len);
    (0..layer_count)
        .flat_map(|layer_offset| {
            (0..chunk_count).filter_map(move |chunk_index| {
                let own_previous_layer_finished =
                    layer_offset == 0 || finished[layer_offset - 1][chunk_index];
                let previous_chunk_started =
                    chunk_index == 0 || started[layer_offset][chunk_index - 1];
                (!started[layer_offset][chunk_index]
                    && own_previous_layer_finished
                    && previous_chunk_started)
                    .then_some(SchedulerSparseWavefrontTask {
                        layer_offset,
                        chunk_index,
                    })
            })
        })
        .min_by_key(|task| {
            (
                task.layer_offset + task.chunk_index,
                task.layer_offset,
                task.chunk_index,
            )
        })
}

#[allow(clippy::too_many_arguments)]
fn execute_scheduler_sparse_chunk_wavefront(
    store: &mut KvCacheBackingStore,
    shape: &RealFullSchedulerExecutionShape,
    reservation_id: u64,
    policy: &PrefillChunkPolicy,
    sparse_batch_graph_bucket: GraphBucket,
    quantization_recipe: &str,
    record_sparse_host_partition: bool,
    counters: &mut RealFullSchedulerExecutionCounters,
    device_kv: &mut RealFullDeviceKvExecutionMirror,
    numeric_progression: &mut RealFullSchedulerNumericProgression,
    catalog: &TensorCatalog,
    summary_timing: bool,
    verbose_timing: bool,
    execution_start: Instant,
) -> Result<()> {
    let wavefront_start = Instant::now();
    let layer_count = GLM52_NUM_HIDDEN_LAYERS - GLM52_FIRST_K_DENSE_REPLACE;
    let chunk_count = shape.prefill_tokens.div_ceil(shape.prefill_chunk_tokens);
    anyhow::ensure!(
        chunk_count >= 2,
        "sparse chunk wavefront requires at least two prefill chunks"
    );
    let mut started = vec![vec![false; chunk_count]; layer_count];
    let mut finished = vec![vec![false; chunk_count]; layer_count];
    let mut layer_started = vec![None::<Instant>; layer_count];
    let mut planned_layer_waves = std::iter::repeat_with(|| None)
        .take(layer_count)
        .collect::<Vec<Option<Vec<Option<Vec<LayerWave>>>>>>();
    let mut pending = VecDeque::<SchedulerSparseWavefrontPending>::with_capacity(
        MAX_PENDING_SPARSE_WAVEFRONT_DISPATCHES,
    );
    let mut completed_tasks = 0_usize;
    let task_count = layer_count
        .checked_mul(chunk_count)
        .context("sparse wavefront task count overflow")?;
    let mut idle_poll_started = None::<Instant>;
    let mut plan_ms = 0.0_f64;
    let mut start_ms = 0.0_f64;
    let mut start_attention_ms = 0.0_f64;
    let mut start_numeric_ms = 0.0_f64;
    let mut poll_ms = 0.0_f64;
    let mut idle_wait_ms = 0.0_f64;
    let mut poll_calls = 0_usize;
    let mut idle_polls = 0_usize;

    while completed_tasks < task_count {
        while pending.len() < MAX_PENDING_SPARSE_WAVEFRONT_DISPATCHES {
            let Some(task) = next_ready_sparse_wavefront_task(&started, &finished) else {
                break;
            };
            let layer_id = GLM52_FIRST_K_DENSE_REPLACE + task.layer_offset;
            let layer = LayerId(layer_id as u32);
            if planned_layer_waves[task.layer_offset].is_none() {
                let plan_start = summary_timing.then(Instant::now);
                let layer_waves =
                    plan_scheduler_chunk_layer_waves(shape, layer, reservation_id, policy)?;
                if let Some(plan_start) = plan_start {
                    plan_ms += elapsed_ms(plan_start);
                }
                anyhow::ensure!(
                    layer_waves.len() == chunk_count,
                    "sparse wavefront layer {layer_id} planned {} chunks, expected {chunk_count}",
                    layer_waves.len()
                );
                planned_layer_waves[task.layer_offset] =
                    Some(layer_waves.into_iter().map(Some).collect());
            }
            let waves = planned_layer_waves[task.layer_offset]
                .as_mut()
                .and_then(|layer_waves| layer_waves.get_mut(task.chunk_index))
                .and_then(Option::take)
                .with_context(|| {
                    format!(
                        "sparse wavefront layer {layer_id} chunk {} was not planned exactly once",
                        task.chunk_index
                    )
                })?;
            let final_prefill_chunk = task.chunk_index + 1 == chunk_count;
            let start = summary_timing.then(Instant::now);
            let started_dispatch = start_scheduler_sparse_wavefront_dispatch(
                store,
                policy,
                waves,
                final_prefill_chunk,
                sparse_batch_graph_bucket,
                quantization_recipe,
                shape.mtp_resolution_accepted_rows(),
                record_sparse_host_partition,
                counters,
                device_kv,
                numeric_progression,
                catalog,
                summary_timing,
            )?;
            if let Some(start) = start {
                start_ms += elapsed_ms(start);
                start_attention_ms += started_dispatch.attention_ms;
                start_numeric_ms += started_dispatch.numeric_ms;
            }
            started[task.layer_offset][task.chunk_index] = true;
            layer_started[task.layer_offset].get_or_insert_with(Instant::now);
            pending.push_back(SchedulerSparseWavefrontPending {
                task,
                apply: started_dispatch.apply,
            });
        }

        anyhow::ensure!(
            !pending.is_empty(),
            "sparse wavefront stalled without a pending or ready task"
        );
        if let Some(non_streaming_index) = pending
            .iter()
            .position(|pending_task| !pending_task.apply.supports_incremental_stream())
        {
            // A balanced/tail chunk can fall below the Spark-reduction
            // streaming threshold even though the surrounding prefill is
            // wavefront-eligible. Complete that one dispatch through the
            // ordinary path instead of handing it to the incremental poller.
            let pending_task = pending
                .remove(non_streaming_index)
                .context("non-streaming sparse wavefront index is out of range")?;
            numeric_progression
                .finish_apply_selected_sparse_tcp_batched(pending_task.apply)
                .with_context(|| {
                    let layer_id = GLM52_FIRST_K_DENSE_REPLACE + pending_task.task.layer_offset;
                    format!(
                        "finishing non-streaming sparse wavefront layer {layer_id} chunk {}",
                        pending_task.task.chunk_index
                    )
                })?;
            if pending_task.task.chunk_index + 1 == chunk_count {
                let layer_id = GLM52_FIRST_K_DENSE_REPLACE + pending_task.task.layer_offset;
                numeric_progression
                    .capture_target_device_hidden_tap(layer_id + 1)
                    .with_context(|| {
                        format!("capturing dSpark target hidden tap after sparse layer {layer_id}")
                    })?;
            }
            finished[pending_task.task.layer_offset][pending_task.task.chunk_index] = true;
            completed_tasks += 1;
            idle_poll_started = None;
            continue;
        }
        let mut completed_pending_index = None;
        let mut made_progress = false;
        for pending_index in 0..pending.len() {
            let pending_task = pending
                .get_mut(pending_index)
                .context("sparse wavefront pending index is out of range")?;
            let poll_start = summary_timing.then(Instant::now);
            let progress = numeric_progression
                .try_progress_apply_selected_sparse_tcp_batched(&mut pending_task.apply)
                .with_context(|| {
                    let layer_id = GLM52_FIRST_K_DENSE_REPLACE + pending_task.task.layer_offset;
                    format!(
                        "polling sparse wavefront layer {layer_id} chunk {}",
                        pending_task.task.chunk_index
                    )
                })?;
            if let Some(poll_start) = poll_start {
                poll_ms += elapsed_ms(poll_start);
                poll_calls += 1;
            }
            let Some(progress) = progress else {
                continue;
            };
            made_progress = true;
            if progress.dispatch_complete {
                completed_pending_index = Some(pending_index);
                break;
            }
        }

        let Some(completed_pending_index) = completed_pending_index else {
            if made_progress {
                idle_poll_started = None;
            } else {
                let idle_started = idle_poll_started.get_or_insert_with(Instant::now);
                let idle_wait_start = summary_timing.then(Instant::now);
                if idle_started.elapsed() < SPARSE_WAVEFRONT_BUSY_POLL {
                    std::hint::spin_loop();
                } else {
                    std::thread::sleep(SPARSE_WAVEFRONT_IDLE_POLL);
                }
                if let Some(idle_wait_start) = idle_wait_start {
                    idle_wait_ms += elapsed_ms(idle_wait_start);
                    idle_polls += 1;
                }
            }
            continue;
        };
        idle_poll_started = None;
        let pending_task = pending
            .remove(completed_pending_index)
            .context("completed sparse wavefront pending index is out of range")?;
        let layer_id = GLM52_FIRST_K_DENSE_REPLACE + pending_task.task.layer_offset;
        if pending_task.task.chunk_index + 1 == chunk_count {
            numeric_progression
                .capture_target_device_hidden_tap(layer_id + 1)
                .with_context(|| {
                    format!("capturing dSpark target hidden tap after sparse layer {layer_id}")
                })?;
        }
        finished[pending_task.task.layer_offset][pending_task.task.chunk_index] = true;
        completed_tasks += 1;

        if pending_task.task.chunk_index + 1 == chunk_count && verbose_timing {
            let layer_ms = layer_started[pending_task.task.layer_offset]
                .map(elapsed_ms)
                .unwrap_or(0.0);
            if layer_id % 10 == 9 || layer_id + 3 >= GLM52_NUM_HIDDEN_LAYERS {
                eprintln!(
                    "real_full_scheduler_wavefront_layer_timing request_id={} layer_id={} chunks={} elapsed_ms={:.3} total_ms={:.3} completed_tasks={}/{} iterations={} sparse_batches={} pending={}",
                    shape.request_id,
                    layer_id,
                    chunk_count,
                    layer_ms,
                    elapsed_ms(execution_start),
                    completed_tasks,
                    task_count,
                    counters.iterations,
                    counters.sparse_expert_batches,
                    pending.len()
                );
            }
        }
    }
    anyhow::ensure!(
        pending.is_empty(),
        "sparse wavefront left pending dispatches"
    );
    anyhow::ensure!(
        finished.iter().flatten().all(|finished| *finished),
        "sparse wavefront left unfinished tasks"
    );
    if summary_timing {
        let total_ms = elapsed_ms(wavefront_start);
        let accounted_ms = plan_ms + start_ms + poll_ms + idle_wait_ms;
        eprintln!(
            "real_full_scheduler_wavefront_summary request_id={} rows={} chunks={} layers={} tasks={} poll_calls={} idle_polls={} total_ms={:.3} plan_ms={:.3} start_ms={:.3} start_attention_ms={:.3} start_numeric_ms={:.3} start_other_ms={:.3} poll_ms={:.3} idle_wait_ms={:.3} other_ms={:.3}",
            shape.request_id,
            shape
                .prefill_tokens
                .saturating_add(shape.decode_rows)
                .saturating_add(shape.mtp_rows),
            chunk_count,
            layer_count,
            task_count,
            poll_calls,
            idle_polls,
            total_ms,
            plan_ms,
            start_ms,
            start_attention_ms,
            start_numeric_ms,
            (start_ms - start_attention_ms - start_numeric_ms).max(0.0),
            poll_ms,
            idle_wait_ms,
            (total_ms - accounted_ms).max(0.0),
        );
    }
    Ok(())
}

fn rolling_sparse_active_dispatches(layers: &[Option<SchedulerSparseRollingLayerApply>]) -> usize {
    layers
        .iter()
        .flatten()
        .map(SchedulerSparseRollingLayerApply::active_dispatches)
        .sum()
}

fn rolling_sparse_buffered_dispatches(
    layers: &[Option<SchedulerSparseRollingLayerApply>],
) -> usize {
    layers
        .iter()
        .flatten()
        .map(SchedulerSparseRollingLayerApply::buffered_dispatches)
        .sum()
}

fn rolling_sparse_active_accumulators(
    layers: &[Option<SchedulerSparseRollingLayerApply>],
) -> (usize, usize) {
    layers
        .iter()
        .flatten()
        .fold((0, 0), |(pages, rows), layer| {
            (
                pages + layer.accumulator_active_pages(),
                rows + layer.accumulator_active_rows(),
            )
        })
}

fn start_ready_rolling_sparse_dispatches(
    numeric_progression: &mut RealFullSchedulerNumericProgression,
    layers: &mut [Option<SchedulerSparseRollingLayerApply>],
) -> Result<usize> {
    let max_active = rolling_sparse_max_active_dispatches();
    let mut active = rolling_sparse_active_dispatches(layers);
    let mut started = 0_usize;
    while active < max_active {
        let mut round_started = false;
        for layer in layers.iter_mut().flatten() {
            if active >= max_active {
                break;
            }
            if layer.queued_dispatches() == 0 {
                continue;
            }
            let count = numeric_progression.start_queued_sparse_rolling_dispatches(layer, 1)?;
            active += count;
            started += count;
            round_started |= count > 0;
        }
        if !round_started {
            break;
        }
    }
    Ok(started)
}

fn rolling_sparse_max_active_dispatches() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        env::var(ROLLING_SPARSE_MAX_ACTIVE_DISPATCHES_ENV)
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .map(|value| value.min(MAX_ACTIVE_ROLLING_SPARSE_DISPATCHES))
            .unwrap_or(MAX_ACTIVE_ROLLING_SPARSE_DISPATCHES)
    })
}

#[allow(clippy::too_many_arguments)]
fn execute_scheduler_sparse_rolling_chunk_wavefront(
    store: &mut KvCacheBackingStore,
    shape: &RealFullSchedulerExecutionShape,
    reservation_id: u64,
    policy: &PrefillChunkPolicy,
    sparse_batch_graph_bucket: GraphBucket,
    quantization_recipe: &str,
    record_sparse_host_partition: bool,
    counters: &mut RealFullSchedulerExecutionCounters,
    device_kv: &mut RealFullDeviceKvExecutionMirror,
    numeric_progression: &mut RealFullSchedulerNumericProgression,
    catalog: &TensorCatalog,
    summary_timing: bool,
    verbose_timing: bool,
    execution_start: Instant,
) -> Result<()> {
    let rolling_start = Instant::now();
    let layer_count = GLM52_NUM_HIDDEN_LAYERS - GLM52_FIRST_K_DENSE_REPLACE;
    let chunk_count = shape.prefill_tokens.div_ceil(shape.prefill_chunk_tokens);
    let total_rows = shape
        .prefill_tokens
        .checked_add(shape.decode_rows)
        .and_then(|rows| rows.checked_add(shape.mtp_rows))
        .context("rolling sparse total row count overflow")?;
    anyhow::ensure!(
        chunk_count >= 2 && rolling_sparse_packs_supported_for_rows(total_rows),
        "rolling sparse wavefront rows={total_rows} chunks={chunk_count} are not eligible"
    );
    let mut started = vec![vec![false; chunk_count]; layer_count];
    let mut finished = vec![vec![false; chunk_count]; layer_count];
    let mut layer_started = vec![None::<Instant>; layer_count];
    let mut planned_layer_waves = std::iter::repeat_with(|| None)
        .take(layer_count)
        .collect::<Vec<Option<Vec<Option<Vec<LayerWave>>>>>>();
    let mut rolling_layers = std::iter::repeat_with(|| None)
        .take(layer_count)
        .collect::<Vec<Option<SchedulerSparseRollingLayerApply>>>();
    let mut completed_tasks = 0_usize;
    let task_count = layer_count
        .checked_mul(chunk_count)
        .context("rolling sparse wavefront task count overflow")?;
    let mut idle_poll_started = None::<Instant>;
    let mut timing = SchedulerRollingWavefrontTiming::default();

    while completed_tasks < task_count
        || rolling_layers
            .iter()
            .flatten()
            .any(|layer| !layer.is_complete())
    {
        timing.iterations += usize::from(summary_timing);
        // Retire completed network work before doing another synchronous chunk preparation.
        // A preparation can include attention, normalization, routing, and shared-MLP launches;
        // admitting a whole lookahead here can otherwise leave completed dispatches occupying all
        // active slots while every Spark waits for the coordinator to post the next pack.
        let mut made_progress = false;
        let mut completed_layers = Vec::new();
        let poll_started = summary_timing.then(Instant::now);
        for (layer_offset, rolling) in rolling_layers.iter_mut().enumerate() {
            let Some(rolling) = rolling.as_mut() else {
                continue;
            };
            let progress = numeric_progression
                .try_progress_sparse_rolling_layer(rolling)
                .with_context(|| {
                    format!(
                        "polling rolling sparse wavefront layer {}",
                        GLM52_FIRST_K_DENSE_REPLACE + layer_offset
                    )
                })?;
            timing.poll_calls += usize::from(summary_timing);
            timing.poll_progress_events += usize::from(summary_timing && progress.made_progress);
            made_progress |= progress.made_progress;
            for chunk_index in progress.completed_task_indices {
                anyhow::ensure!(
                    chunk_index < chunk_count && !finished[layer_offset][chunk_index],
                    "rolling sparse layer {} completed chunk {chunk_index} twice",
                    GLM52_FIRST_K_DENSE_REPLACE + layer_offset
                );
                finished[layer_offset][chunk_index] = true;
                completed_tasks += 1;
                if chunk_index + 1 == chunk_count {
                    let layer_id = GLM52_FIRST_K_DENSE_REPLACE + layer_offset;
                    numeric_progression
                        .capture_target_device_hidden_tap(layer_id + 1)
                        .with_context(|| {
                            format!(
                                "capturing dSpark target hidden tap after rolling sparse layer {layer_id}"
                            )
                        })?;
                }
            }
            if progress.layer_complete {
                completed_layers.push(layer_offset);
            }
        }
        if let Some(poll_started) = poll_started {
            timing.poll_ms += elapsed_ms(poll_started);
        }
        let (active_accumulator_pages, active_accumulator_rows) =
            rolling_sparse_active_accumulators(&rolling_layers);
        timing.observe_accumulators(active_accumulator_pages, active_accumulator_rows);
        for layer_offset in completed_layers {
            let layer_id = GLM52_FIRST_K_DENSE_REPLACE + layer_offset;
            anyhow::ensure!(
                finished[layer_offset].iter().all(|finished| *finished),
                "rolling sparse layer {layer_id} completed with unfinished chunks"
            );
            if verbose_timing && (layer_id % 10 == 9 || layer_id + 3 >= GLM52_NUM_HIDDEN_LAYERS) {
                eprintln!(
                    "real_full_scheduler_rolling_wavefront_layer_timing request_id={} layer_id={} chunks={} elapsed_ms={:.3} total_ms={:.3} completed_tasks={}/{} iterations={} sparse_batches={} buffered={}",
                    shape.request_id,
                    layer_id,
                    chunk_count,
                    layer_started[layer_offset].map(elapsed_ms).unwrap_or(0.0),
                    elapsed_ms(execution_start),
                    completed_tasks,
                    task_count,
                    counters.iterations,
                    counters.sparse_expert_batches,
                    rolling_sparse_buffered_dispatches(&rolling_layers),
                );
            }
            if let Some(rolling) = rolling_layers[layer_offset].as_ref() {
                timing.max_accumulator_pages_per_layer = timing
                    .max_accumulator_pages_per_layer
                    .max(rolling.accumulator_peak_pages());
                timing.max_accumulator_rows_per_layer = timing
                    .max_accumulator_rows_per_layer
                    .max(rolling.accumulator_peak_rows());
            }
            rolling_layers[layer_offset] = None;
        }

        if completed_tasks >= task_count && rolling_layers.iter().all(Option::is_none) {
            break;
        }

        let dispatch_start = summary_timing.then(Instant::now);
        let started_dispatches =
            start_ready_rolling_sparse_dispatches(numeric_progression, &mut rolling_layers)?;
        if let Some(dispatch_start) = dispatch_start {
            timing.dispatch_start_ms += elapsed_ms(dispatch_start);
            timing.dispatches_started += started_dispatches;
            timing.observe_dispatches(
                rolling_sparse_active_dispatches(&rolling_layers),
                rolling_sparse_buffered_dispatches(&rolling_layers),
            );
        }
        made_progress |= started_dispatches > 0;

        // Admit one source chunk per scheduler turn. This keeps completion harvest and queue
        // refill latency bounded by one 512-row preparation instead of an entire lookahead. Any
        // newly eligible 256-row physical packs are posted before another source preparation.
        if rolling_sparse_buffered_dispatches(&rolling_layers)
            < MAX_BUFFERED_ROLLING_SPARSE_DISPATCHES
        {
            let Some(task) = next_ready_sparse_wavefront_task(&started, &finished) else {
                if made_progress {
                    idle_poll_started = None;
                    continue;
                }
                let has_buffered_dispatch = rolling_sparse_buffered_dispatches(&rolling_layers) > 0;
                anyhow::ensure!(
                    has_buffered_dispatch,
                    "rolling sparse wavefront stalled without ready or buffered work"
                );
                let idle_started = idle_poll_started.get_or_insert_with(Instant::now);
                let idle_wait_started = summary_timing.then(Instant::now);
                if idle_started.elapsed() < SPARSE_WAVEFRONT_BUSY_POLL {
                    std::hint::spin_loop();
                    timing.idle_spins += usize::from(summary_timing);
                } else {
                    std::thread::sleep(SPARSE_WAVEFRONT_IDLE_POLL);
                    timing.idle_sleeps += usize::from(summary_timing);
                }
                if let Some(idle_wait_started) = idle_wait_started {
                    timing.idle_wait_ms += elapsed_ms(idle_wait_started);
                }
                continue;
            };
            let layer_id = GLM52_FIRST_K_DENSE_REPLACE + task.layer_offset;
            let layer = LayerId(layer_id as u32);
            if planned_layer_waves[task.layer_offset].is_none() {
                let plan_started = summary_timing.then(Instant::now);
                let layer_waves =
                    plan_scheduler_chunk_layer_waves(shape, layer, reservation_id, policy)?;
                if let Some(plan_started) = plan_started {
                    timing.plan_ms += elapsed_ms(plan_started);
                }
                anyhow::ensure!(
                    layer_waves.len() == chunk_count,
                    "rolling sparse layer {layer_id} planned {} chunks, expected {chunk_count}",
                    layer_waves.len()
                );
                planned_layer_waves[task.layer_offset] =
                    Some(layer_waves.into_iter().map(Some).collect());
            }
            let waves = planned_layer_waves[task.layer_offset]
                .as_mut()
                .and_then(|layer_waves| layer_waves.get_mut(task.chunk_index))
                .and_then(Option::take)
                .with_context(|| {
                    format!(
                        "rolling sparse layer {layer_id} chunk {} was not planned exactly once",
                        task.chunk_index
                    )
                })?;
            let final_prefill_chunk = task.chunk_index + 1 == chunk_count;
            layer_started[task.layer_offset].get_or_insert_with(Instant::now);
            if rolling_layers[task.layer_offset].is_none() {
                rolling_layers[task.layer_offset] =
                    Some(SchedulerSparseRollingLayerApply::new(layer_id, total_rows)?);
            }
            let admission_started = Instant::now();
            let prepare_started = Instant::now();
            let prepared = prepare_scheduler_sparse_wavefront_dispatch(
                store,
                policy,
                waves,
                final_prefill_chunk,
                sparse_batch_graph_bucket,
                quantization_recipe,
                shape.mtp_resolution_accepted_rows(),
                record_sparse_host_partition,
                counters,
                device_kv,
                numeric_progression,
                catalog,
                summary_timing,
            )?;
            let prepare_ms = elapsed_ms(prepare_started);
            let push_started = Instant::now();
            let rolling = rolling_layers[task.layer_offset]
                .as_mut()
                .expect("rolling sparse layer was initialized before preparation");
            let push_timing = numeric_progression.push_prepared_sparse_rolling_layer(
                catalog,
                rolling,
                task.chunk_index,
                prepared.dispatch,
                final_prefill_chunk,
                summary_timing,
            )?;
            let push_ms = elapsed_ms(push_started);
            if summary_timing {
                timing.admissions += 1;
                timing.prepare_ms += prepare_ms;
                timing.prepare_attention_ms += prepared.attention_ms;
                timing.prepare_numeric_ms += prepared.numeric_ms;
                timing.push_ms += push_ms;
                timing.push_shared_mlp_ms += push_timing.shared_mlp_ms;
                timing.push_planner_ms += push_timing.planner_ms;
            }
            started[task.layer_offset][task.chunk_index] = true;
            made_progress = true;
            let dispatch_start = summary_timing.then(Instant::now);
            let started_dispatches =
                start_ready_rolling_sparse_dispatches(numeric_progression, &mut rolling_layers)?;
            if let Some(dispatch_start) = dispatch_start {
                timing.dispatch_start_ms += elapsed_ms(dispatch_start);
                timing.dispatches_started += started_dispatches;
                timing.observe_dispatches(
                    rolling_sparse_active_dispatches(&rolling_layers),
                    rolling_sparse_buffered_dispatches(&rolling_layers),
                );
            }
            if verbose_timing {
                eprintln!(
                    "real_full_scheduler_rolling_admission_timing request_id={} layer_id={} chunk={} final={} prepare_ms={:.3} push_ms={:.3} total_ms={:.3} started_dispatches={} active={} buffered={}",
                    shape.request_id,
                    layer_id,
                    task.chunk_index,
                    final_prefill_chunk,
                    prepare_ms,
                    push_ms,
                    elapsed_ms(admission_started),
                    started_dispatches,
                    rolling_sparse_active_dispatches(&rolling_layers),
                    rolling_sparse_buffered_dispatches(&rolling_layers),
                );
            }
        }

        if made_progress {
            idle_poll_started = None;
            continue;
        }
        let has_ready_task = next_ready_sparse_wavefront_task(&started, &finished).is_some();
        let has_buffered_dispatch = rolling_sparse_buffered_dispatches(&rolling_layers) > 0;
        anyhow::ensure!(
            has_ready_task || has_buffered_dispatch,
            "rolling sparse wavefront stalled without ready or buffered work"
        );
        let idle_started = idle_poll_started.get_or_insert_with(Instant::now);
        let idle_wait_started = summary_timing.then(Instant::now);
        if idle_started.elapsed() < SPARSE_WAVEFRONT_BUSY_POLL {
            std::hint::spin_loop();
            timing.idle_spins += usize::from(summary_timing);
        } else {
            std::thread::sleep(SPARSE_WAVEFRONT_IDLE_POLL);
            timing.idle_sleeps += usize::from(summary_timing);
        }
        if let Some(idle_wait_started) = idle_wait_started {
            timing.idle_wait_ms += elapsed_ms(idle_wait_started);
        }
    }
    anyhow::ensure!(
        finished.iter().flatten().all(|finished| *finished),
        "rolling sparse wavefront left unfinished tasks"
    );
    if summary_timing {
        let total_ms = elapsed_ms(rolling_start);
        let accounted_ms = timing.plan_ms
            + timing.prepare_ms
            + timing.push_ms
            + timing.dispatch_start_ms
            + timing.poll_ms
            + timing.idle_wait_ms;
        eprintln!(
            "real_full_scheduler_rolling_summary request_id={} rows={} chunks={} layers={} iterations={} admissions={} dispatches_started={} poll_calls={} poll_progress_events={} idle_spins={} idle_sleeps={} max_active={} max_buffered={} max_accumulator_pages_per_layer={} max_accumulator_rows_per_layer={} max_accumulator_pages_total={} max_accumulator_rows_total={} total_ms={:.3} plan_ms={:.3} prepare_ms={:.3} prepare_attention_ms={:.3} prepare_numeric_ms={:.3} push_ms={:.3} push_shared_mlp_ms={:.3} push_planner_ms={:.3} dispatch_start_ms={:.3} poll_ms={:.3} idle_wait_ms={:.3} other_ms={:.3}",
            shape.request_id,
            total_rows,
            chunk_count,
            layer_count,
            timing.iterations,
            timing.admissions,
            timing.dispatches_started,
            timing.poll_calls,
            timing.poll_progress_events,
            timing.idle_spins,
            timing.idle_sleeps,
            timing.max_active_dispatches,
            timing.max_buffered_dispatches,
            timing.max_accumulator_pages_per_layer,
            timing.max_accumulator_rows_per_layer,
            timing.max_accumulator_pages_total,
            timing.max_accumulator_rows_total,
            total_ms,
            timing.plan_ms,
            timing.prepare_ms,
            timing.prepare_attention_ms,
            timing.prepare_numeric_ms,
            timing.push_ms,
            timing.push_shared_mlp_ms,
            timing.push_planner_ms,
            timing.dispatch_start_ms,
            timing.poll_ms,
            timing.idle_wait_ms,
            (total_ms - accounted_ms).max(0.0),
        );
    }
    Ok(())
}

fn real_full_scheduler_execution_for_shape_inner(
    kv_config: KvCacheConfig,
    catalog: &TensorCatalog,
    shape: RealFullSchedulerExecutionShape,
    sparse_tcp_routed_mlp: Option<RealFullSchedulerSparseTcpRoutedMlpContext>,
    state: Option<&mut RealFullSchedulerExecutionState>,
    retain_final_target_device_hidden: bool,
    retain_full_target_device_hidden: bool,
    target_device_hidden_tap_rows: usize,
) -> Result<(
    RealFullSchedulerExecutionDryRun,
    Option<RealFullSchedulerSparseTcpDispatchProbe>,
    Option<DeviceBf16Output>,
    Option<RealFullSchedulerTargetHiddenTaps>,
)> {
    let execution_start = Instant::now();
    let coordinator_graph_stats_before = coordinator_cuda_graph_stats().ok();
    let scheduler_timing = scheduler_execution_timing_enabled();
    let scheduler_verbose_timing = scheduler_execution_verbose_timing_enabled();
    validate_scheduler_execution_shape(&shape)?;
    let record_sparse_host_partition = sparse_tcp_routed_mlp.is_none();
    let stateful_live_request = state.is_some();
    if scheduler_verbose_timing {
        eprintln!(
            "real_full_scheduler_timing request_id={} stage=start prefix_tokens={} prefill_tokens={} prefill_chunk_tokens={} decode_rows={} mtp_rows={} sparse_tcp={}",
            shape.request_id,
            shape.prefix_tokens,
            shape.prefill_tokens,
            shape.prefill_chunk_tokens,
            shape.decode_rows,
        shape.mtp_rows,
        sparse_tcp_routed_mlp.is_some()
    );
    }
    if let Some(prefill_token_ids) = shape.prefill_token_ids.as_ref() {
        anyhow::ensure!(
            prefill_token_ids.len() == shape.prefill_tokens,
            "real-full scheduler execution prompt token id count {} does not match prefill tokens {}",
            prefill_token_ids.len(),
            shape.prefill_tokens
        );
    }
    if let Some(decode_token_ids) = shape.decode_token_ids.as_ref() {
        anyhow::ensure!(
            decode_token_ids.len() == shape.decode_rows
                || decode_token_ids.len() == shape.decode_rows + shape.mtp_rows,
            "real-full scheduler execution token id count {} does not match decode rows {} or decode+MTP rows {}",
            decode_token_ids.len(),
            shape.decode_rows,
            shape.decode_rows + shape.mtp_rows
        );
    }
    let reservation_tokens = shape.reservation_tokens();
    let kv_bytes_per_token = kv_config.bytes_per_token();
    let mut local_store: KvCacheBackingStore;
    let mut local_device_kv: RealFullDeviceKvExecutionMirror;
    let (store, reservation_id, device_kv) = if let Some(state) = state {
        state.validate_shape(&shape)?;
        (&mut state.store, state.reservation_id, &mut state.device_kv)
    } else {
        let scheduler_kv_config = KvCacheConfig {
            max_tokens: reservation_tokens,
            ..kv_config
        };
        local_store = KvCacheBackingStore::new(scheduler_kv_config.clone());
        let reservation_id = local_store.reserve(shape.sequence_id.as_str(), reservation_tokens)?;
        local_device_kv = RealFullDeviceKvExecutionMirror::new(scheduler_kv_config)?;
        (&mut local_store, reservation_id, &mut local_device_kv)
    };
    let policy = PrefillChunkPolicy {
        chunk_tokens: shape.prefill_chunk_tokens,
        max_prefill_tokens_per_iteration: shape.prefill_chunk_tokens,
        max_active_prefill_chunks: 1,
        decode_priority: true,
    };
    let sparse_batch_graph_bucket = shape.sparse_batch_graph_bucket();
    let quantization_recipe = ModelFacts::default().quantization_recipe;
    let mut counters = RealFullSchedulerExecutionCounters {
        layer_order_verified: true,
        sparse_expert_host_batch_routes_match_global: true,
        sparse_expert_host_batch_graph_counts_valid: true,
        sparse_expert_host_wire_envelopes_valid: true,
        ..Default::default()
    };
    let mut stage_timing = SchedulerExecutionStageTiming::default();
    let numeric_shape = RealFullSchedulerNumericProgressionShape::from_execution_shape(&shape);
    let mut numeric_progression = RealFullSchedulerNumericProgression::new(numeric_shape);
    if stateful_live_request {
        numeric_progression = numeric_progression.with_live_request();
    }
    if retain_final_target_device_hidden {
        numeric_progression = numeric_progression.with_final_target_device_hidden();
    }
    if retain_full_target_device_hidden {
        numeric_progression = numeric_progression.with_full_target_device_hidden();
    }
    if target_device_hidden_tap_rows > 0 {
        numeric_progression =
            numeric_progression.with_target_device_hidden_taps(target_device_hidden_tap_rows);
    }
    let sparse_chunk_wavefront = stateful_live_request
        && shape.prefill_tokens.div_ceil(shape.prefill_chunk_tokens) >= 2
        && sparse_tcp_routed_mlp
            .as_ref()
            .is_some_and(RealFullSchedulerSparseTcpRoutedMlpContext::supports_chunk_wavefront);
    let sparse_wavefront_rows = shape
        .prefill_tokens
        .saturating_add(shape.decode_rows)
        .saturating_add(shape.mtp_rows);
    let bounded_long_prefill_required =
        bounded_long_prefill_wavefront_required(sparse_wavefront_rows);
    let bounded_long_prefill_wavefront = sparse_chunk_wavefront && bounded_long_prefill_required;
    anyhow::ensure!(
        !stateful_live_request || !bounded_long_prefill_required || bounded_long_prefill_wavefront,
        "live prefill with {sparse_wavefront_rows} rows requires the bounded sparse wavefront"
    );
    let rolling_sparse_chunk_wavefront =
        sparse_chunk_wavefront && rolling_sparse_packs_supported_for_rows(sparse_wavefront_rows);
    if let Some(context) = sparse_tcp_routed_mlp {
        numeric_progression = numeric_progression.with_sparse_tcp_routed_mlp(context);
    }
    if !bounded_long_prefill_wavefront {
        if let Some(prefill_token_ids) = shape.prefill_token_ids.as_deref() {
            numeric_progression
                .seed_prefill_token_embeddings(
                    catalog,
                    prefill_token_ids,
                    shape
                        .prefill_vision_embeddings
                        .as_deref()
                        .map(Vec::as_slice),
                )
                .context("seeding scheduler prefill residual rows from prompt embeddings")?;
        }
    }
    if let Some(decode_token_ids) = shape.decode_token_ids.as_deref() {
        numeric_progression
            .seed_decode_token_embeddings(catalog, &decode_token_ids[..shape.decode_rows])
            .context("seeding scheduler decode residual rows from token embeddings")?;
        if decode_token_ids.len() > shape.decode_rows {
            numeric_progression
                .seed_mtp_token_embeddings(catalog, &decode_token_ids[shape.decode_rows..])
                .context("seeding scheduler MTP residual rows from draft token embeddings")?;
        }
    }
    if scheduler_verbose_timing {
        eprintln!(
            "real_full_scheduler_timing request_id={} stage=setup elapsed_ms={:.3}",
            shape.request_id,
            elapsed_ms(execution_start)
        );
    }

    if bounded_long_prefill_wavefront {
        let wavefront_start = scheduler_timing.then(Instant::now);
        execute_scheduler_bounded_long_prefill_wavefront(
            &mut *store,
            &shape,
            reservation_id,
            &policy,
            sparse_batch_graph_bucket,
            &quantization_recipe,
            record_sparse_host_partition,
            &mut counters,
            &mut *device_kv,
            &mut numeric_progression,
            catalog,
            target_device_hidden_tap_rows,
            scheduler_timing,
        )?;
        stage_timing.record_sparse_wavefront(wavefront_start);
    }

    let ordinary_layer_count = if bounded_long_prefill_wavefront {
        0
    } else {
        GLM52_NUM_HIDDEN_LAYERS
    };
    for layer_id in 0..ordinary_layer_count {
        let layer_start = Instant::now();
        let layer = LayerId(layer_id as u32);
        if sparse_chunk_wavefront && layer_id == GLM52_FIRST_K_DENSE_REPLACE {
            let wavefront_start = scheduler_timing.then(Instant::now);
            if rolling_sparse_chunk_wavefront {
                execute_scheduler_sparse_rolling_chunk_wavefront(
                    &mut *store,
                    &shape,
                    reservation_id,
                    &policy,
                    sparse_batch_graph_bucket,
                    &quantization_recipe,
                    record_sparse_host_partition,
                    &mut counters,
                    &mut *device_kv,
                    &mut numeric_progression,
                    catalog,
                    scheduler_timing,
                    scheduler_verbose_timing,
                    execution_start,
                )?;
            } else {
                execute_scheduler_sparse_chunk_wavefront(
                    &mut *store,
                    &shape,
                    reservation_id,
                    &policy,
                    sparse_batch_graph_bucket,
                    &quantization_recipe,
                    record_sparse_host_partition,
                    &mut counters,
                    &mut *device_kv,
                    &mut numeric_progression,
                    catalog,
                    scheduler_timing,
                    scheduler_verbose_timing,
                    execution_start,
                )?;
            }
            stage_timing.record_sparse_wavefront(wavefront_start);
            break;
        }
        let prefill_chunks =
            plan_scheduler_prefill_chunks(&shape, layer, reservation_id, Priority(0), &policy);

        let mut decode_mtp = (0..shape.decode_rows)
            .map(|decode_offset| {
                LayerWave::decode(DecodeStep::new(
                    shape.request_id.as_str(),
                    shape.sequence_id.as_str(),
                    layer,
                    PositionId((shape.decode_token_start() + decode_offset) as u64),
                    Some(reservation_id),
                    Priority(0),
                    shape.placement_version.as_str(),
                ))
            })
            .collect::<Vec<_>>();
        if shape.mtp_rows > 0 {
            decode_mtp.push(LayerWave::mtp_verify(MtpVerifyBlock::new(
                shape.request_id.as_str(),
                shape.sequence_id.as_str(),
                layer,
                PositionId(shape.mtp_token_start() as u64),
                shape.mtp_rows,
                Some(reservation_id),
                Priority(0),
                GraphBucket::new(shape.mtp_rows),
                shape.placement_version.as_str(),
            )));
        }
        let prefill_ms = if prefill_chunks.len() == 1 {
            let mut combined_policy = policy.clone();
            combined_policy.decode_priority = false;
            let mut combined = Vec::with_capacity(1 + decode_mtp.len());
            combined.push(LayerWave::prefill(
                prefill_chunks
                    .into_iter()
                    .next()
                    .expect("one prefill chunk is present"),
            ));
            combined.extend(decode_mtp);
            let expected_modes = combined.iter().map(|wave| wave.mode).collect::<Vec<_>>();
            let attention_start = scheduler_timing.then(Instant::now);
            let iteration = real_full_apply_admitted_scheduler_iteration(
                &mut *store,
                &combined_policy,
                combined,
                &expected_modes,
                sparse_batch_graph_bucket,
                &quantization_recipe,
                shape.mtp_resolution_accepted_rows(),
                record_sparse_host_partition,
                &mut counters,
                &mut *device_kv,
                &mut numeric_progression,
                catalog,
            )
            .with_context(|| {
                format!("applying admitted combined prefill/decode/MTP waves for layer {layer_id}")
            })?;
            stage_timing.record_attention(layer_id, attention_start);
            let numeric_start = scheduler_timing.then(Instant::now);
            numeric_progression
                .apply_selected(
                    layer_id,
                    catalog,
                    &iteration.selected,
                    &iteration.device_attention_deltas,
                    sparse_batch_graph_bucket,
                    &quantization_recipe,
                )
                .with_context(|| {
                    format!(
                        "applying numeric combined prefill/decode/MTP rows for layer {layer_id}"
                    )
                })?;
            stage_timing.record_numeric(layer_id, numeric_start);
            elapsed_ms(layer_start)
        } else {
            const MAX_PIPELINED_PREFILL_DISPATCHES: usize = 2;
            let mut pending_prefill_dispatches = VecDeque::new();
            let prefill_chunk_count = prefill_chunks.len();
            for (chunk_index, prefill_chunk) in prefill_chunks.into_iter().enumerate() {
                let token_start = prefill_chunk.token_start.0;
                let token_count = prefill_chunk.token_count;
                let final_prefill_chunk = chunk_index + 1 == prefill_chunk_count;
                let mut waves =
                    Vec::with_capacity(1 + usize::from(final_prefill_chunk) * decode_mtp.len());
                waves.push(LayerWave::prefill(prefill_chunk));
                if final_prefill_chunk {
                    waves.append(&mut decode_mtp);
                }
                let expected_modes = waves.iter().map(|wave| wave.mode).collect::<Vec<_>>();
                let mut iteration_policy = policy.clone();
                if final_prefill_chunk {
                    iteration_policy.decode_priority = false;
                }
                let attention_start = scheduler_timing.then(Instant::now);
                let iteration = real_full_apply_admitted_scheduler_iteration(
                    &mut *store,
                    &iteration_policy,
                    waves,
                    &expected_modes,
                    sparse_batch_graph_bucket,
                    &quantization_recipe,
                    shape.mtp_resolution_accepted_rows(),
                    record_sparse_host_partition,
                    &mut counters,
                    &mut *device_kv,
                    &mut numeric_progression,
                    catalog,
                )
                .with_context(|| {
                    format!(
                    "applying admitted prefill chunk {token_start}+{token_count} final={final_prefill_chunk} for layer {layer_id}"
                )
                })?;
                stage_timing.record_attention(layer_id, attention_start);
                if numeric_progression
                    .can_pipeline_selected_sparse_tcp_batched(layer_id, &iteration.selected)
                {
                    let numeric_start = scheduler_timing.then(Instant::now);
                    if let Some(pending) = numeric_progression
                        .start_apply_selected_sparse_tcp_batched(
                            layer_id,
                            catalog,
                            &iteration.selected,
                            &iteration.device_attention_deltas,
                            sparse_batch_graph_bucket,
                            &quantization_recipe,
                        )
                        .with_context(|| {
                            format!(
                                "starting numeric prefill chunk {token_start}+{token_count} for layer {layer_id}"
                            )
                        })?
                    {
                        pending_prefill_dispatches.push_back(pending);
                    }
                    stage_timing.record_numeric(layer_id, numeric_start);
                } else {
                    while let Some(pending) = pending_prefill_dispatches.pop_front() {
                        let numeric_start = scheduler_timing.then(Instant::now);
                        numeric_progression
                            .finish_apply_selected_sparse_tcp_batched(pending)
                            .with_context(|| {
                                format!(
                                    "finishing pipelined numeric prefill chunks for layer {layer_id}"
                                )
                            })?;
                        stage_timing.record_numeric(layer_id, numeric_start);
                    }
                    let numeric_start = scheduler_timing.then(Instant::now);
                    numeric_progression
                        .apply_selected(
                        layer_id,
                        catalog,
                        &iteration.selected,
                        &iteration.device_attention_deltas,
                        sparse_batch_graph_bucket,
                        &quantization_recipe,
                    )
                        .with_context(|| {
                            format!(
                                "applying numeric prefill chunk {token_start}+{token_count} for layer {layer_id}"
                            )
                        })?;
                    stage_timing.record_numeric(layer_id, numeric_start);
                }
                if pending_prefill_dispatches.len() == MAX_PIPELINED_PREFILL_DISPATCHES {
                    let pending = pending_prefill_dispatches
                        .pop_front()
                        .expect("pipelined prefill dispatch queue is non-empty");
                    let numeric_start = scheduler_timing.then(Instant::now);
                    numeric_progression
                        .finish_apply_selected_sparse_tcp_batched(pending)
                        .with_context(|| {
                            format!(
                                "finishing bounded pipelined numeric prefill for layer {layer_id}"
                            )
                        })?;
                    stage_timing.record_numeric(layer_id, numeric_start);
                }
            }
            while let Some(pending) = pending_prefill_dispatches.pop_front() {
                let numeric_start = scheduler_timing.then(Instant::now);
                numeric_progression
                    .finish_apply_selected_sparse_tcp_batched(pending)
                    .with_context(|| {
                        format!("finishing pipelined numeric prefill for layer {layer_id}")
                    })?;
                stage_timing.record_numeric(layer_id, numeric_start);
            }
            let prefill_ms = elapsed_ms(layer_start);

            if !decode_mtp.is_empty() {
                let expected_modes = decode_mtp.iter().map(|wave| wave.mode).collect::<Vec<_>>();
                let attention_start = scheduler_timing.then(Instant::now);
                let iteration = real_full_apply_admitted_scheduler_iteration(
                    &mut *store,
                    &policy,
                    decode_mtp,
                    &expected_modes,
                    sparse_batch_graph_bucket,
                    &quantization_recipe,
                    shape.mtp_resolution_accepted_rows(),
                    record_sparse_host_partition,
                    &mut counters,
                    &mut *device_kv,
                    &mut numeric_progression,
                    catalog,
                )
                .with_context(|| {
                    format!("applying admitted decode/MTP waves for layer {layer_id}")
                })?;
                stage_timing.record_attention(layer_id, attention_start);
                let numeric_start = scheduler_timing.then(Instant::now);
                numeric_progression
                    .apply_selected(
                        layer_id,
                        catalog,
                        &iteration.selected,
                        &iteration.device_attention_deltas,
                        sparse_batch_graph_bucket,
                        &quantization_recipe,
                    )
                    .with_context(|| {
                        format!("applying numeric decode/MTP rows for layer {layer_id}")
                    })?;
                stage_timing.record_numeric(layer_id, numeric_start);
            }
            prefill_ms
        };
        numeric_progression
            .capture_target_device_hidden_tap(layer_id + 1)
            .with_context(|| {
                format!("capturing dSpark target hidden tap after target layer {layer_id}")
            })?;
        if scheduler_verbose_timing
            && (layer_id < 3 || layer_id % 10 == 9 || layer_id + 3 >= GLM52_NUM_HIDDEN_LAYERS)
        {
            let layer_ms = elapsed_ms(layer_start);
            eprintln!(
                "real_full_scheduler_layer_timing request_id={} layer_id={} elapsed_ms={:.3} prefill_ms={:.3} total_ms={:.3} iterations={} sparse_batches={} host_batches={} device_attention_launches={}",
                shape.request_id,
                layer_id,
                layer_ms,
                prefill_ms,
                elapsed_ms(execution_start),
                counters.iterations,
                counters.sparse_expert_batches,
                counters.sparse_expert_host_batches,
                counters.device_attention_launches
            );
        }
    }

    if scheduler_timing {
        eprintln!(
            "real_full_scheduler_stage_summary request_id={} dense_attention_ms={:.3} dense_numeric_ms={:.3} sparse_attention_ms={:.3} sparse_numeric_ms={:.3} sparse_wavefront_ms={:.3}",
            shape.request_id,
            stage_timing.attention_ms[0],
            stage_timing.numeric_ms[0],
            stage_timing.attention_ms[1],
            stage_timing.numeric_ms[1],
            stage_timing.sparse_wavefront_ms,
        );
    }

    let finish_start = Instant::now();
    let numeric_progression_finish = if stateful_live_request {
        numeric_progression
            .finish_live_request()
            .context("finalizing live BF16 scheduler numeric progression")?
    } else {
        numeric_progression
            .finish()
            .context("finalizing BF16 scheduler numeric progression")?
    };
    if scheduler_verbose_timing {
        eprintln!(
            "real_full_scheduler_timing request_id={} stage=numeric_finish elapsed_ms={:.3} total_ms={:.3}",
            shape.request_id,
            elapsed_ms(finish_start),
            elapsed_ms(execution_start)
        );
    }
    let sample_start = Instant::now();
    // A speculative verify cycle retains the complete target hidden batch and
    // scores every target row immediately after this function returns. Scoring
    // the scalar decode row here would run final_norm + lm_head twice for the
    // same row. Keep scalar cycles self-contained, but defer verify sampling
    // to the required batched path.
    let terminal_lm_head_sample = if scheduler_should_defer_target_lm_head_sample(
        retain_final_target_device_hidden,
        shape.mtp_rows,
    ) {
        scheduler_deferred_target_lm_head_sample(
            numeric_progression_finish
                .final_decode_device_hidden
                .as_ref(),
        )
    } else {
        scheduler_terminal_lm_head_sample_with_options(
            catalog,
            numeric_progression_finish
                .final_decode_device_hidden
                .as_ref(),
            shape.lm_head_sampling,
        )
    };
    if scheduler_verbose_timing {
        eprintln!(
            "real_full_scheduler_timing request_id={} stage=terminal_sample elapsed_ms={:.3} total_ms={:.3} status={} sampled_token_id={:?}",
            shape.request_id,
            elapsed_ms(sample_start),
            elapsed_ms(execution_start),
            terminal_lm_head_sample.status,
            terminal_lm_head_sample.sampled_token_id
        );
    }
    let RealFullSchedulerNumericProgressionFinish {
        self_test: numeric_progression_self_test,
        final_decode_device_hidden,
        final_target_device_hidden,
        target_device_hidden_taps,
        sparse_tcp_dispatch_probe,
    } = numeric_progression_finish;
    let device_kv = device_kv.summary();
    let full_context_device_attention_complete = scheduler_full_context_device_attention_complete(
        &shape,
        &counters,
        numeric_progression_self_test.passed,
        device_kv.uses_device_kv_cache,
    );
    let (expected_attention_launch_min, expected_attention_launch_max) =
        scheduler_device_attention_launch_range(&shape);
    let expected_attention_query_rows =
        GLM52_NUM_HIDDEN_LAYERS * (shape.prefill_tokens + shape.decode_rows + shape.mtp_rows);
    let expected_attention_rows = scheduler_expected_device_attention_rows(&shape);
    let expected_attention_descriptor_min = expected_attention_launch_min;
    let expected_attention_descriptor_max = expected_attention_rows;
    let expected_attention_output_values = expected_attention_query_rows * GLM52_HIDDEN_SIZE;
    if scheduler_verbose_timing {
        eprintln!(
            "real_full_scheduler_attention_gate request_id={} complete={} numeric={} layer_order={} uses_device_kv={} launches={}/{}..{} hidden_projection_launches={}/{} rows={}/{}..{} query_rows={}/{} kv_descriptors={}/{}..{} output_values={}/{}",
            shape.request_id,
            full_context_device_attention_complete,
            numeric_progression_self_test.passed,
            counters.layer_order_verified,
            device_kv.uses_device_kv_cache,
            counters.device_attention_launches,
            expected_attention_launch_min,
            expected_attention_launch_max,
            counters.device_attention_hidden_projection_launches,
            counters.device_attention_launches,
            counters.device_attention_rows,
            expected_attention_query_rows,
            expected_attention_rows,
            counters.device_attention_query_rows,
            expected_attention_query_rows,
            counters.device_attention_kv_descriptors,
            expected_attention_descriptor_min,
            expected_attention_descriptor_max,
            counters.device_attention_output_values,
            expected_attention_output_values
        );
    }
    let status = scheduler_execution_status_for_completion(
        &terminal_lm_head_sample,
        full_context_device_attention_complete,
    );
    let request_prefill_chunks = shape.prefill_tokens.div_ceil(shape.prefill_chunk_tokens);
    let kv_report = scheduler_kv_report_summary(
        store,
        &shape,
        kv_bytes_per_token,
        &counters,
        stateful_live_request,
    );
    let coordinator_graph_stats_after = coordinator_cuda_graph_stats().ok();
    let (graph_slots, captured_graphs, graph_captures, graph_launches) = match (
        coordinator_graph_stats_before,
        coordinator_graph_stats_after,
    ) {
        (Some(before), Some(after)) => (
            after.slots,
            after.captured_graphs,
            after.graph_captures.saturating_sub(before.graph_captures),
            after.graph_launches.saturating_sub(before.graph_launches),
        ),
        (None, Some(after)) => (after.slots, after.captured_graphs, 0, 0),
        _ => (0, 0, 0, 0),
    };
    Ok((RealFullSchedulerExecutionDryRun {
        status,
        scope: "admit full-layer prefix-prefill, later-prefill, decode, and MTP waves, then apply selected host/device KV I/O and sparse ExpertBatch records",
        request_prefill_tokens: shape.prefill_tokens,
        request_prefill_chunks,
        request_decode_rows: shape.decode_rows,
        request_mtp_verify_rows: shape.mtp_rows,
        request_mtp_accepted_rows: shape.mtp_accepted_rows,
        request_coordinator_graph_slots: graph_slots,
        request_coordinator_graph_captured_graphs: captured_graphs,
        request_coordinator_graph_captures: graph_captures,
        request_coordinator_graph_launches: graph_launches,
        iterations: counters.iterations,
        candidate_layerwaves: counters.candidate_layerwaves,
        selected_layerwaves: counters.selected_layerwaves,
        deferred_layerwaves: counters.deferred_layerwaves,
        selected_decode_rows: counters.selected_decode_rows,
        selected_prefill_rows: counters.selected_prefill_rows,
        selected_mtp_rows: counters.selected_mtp_rows,
        sparse_expert_batches: counters.sparse_expert_batches,
        sparse_expert_batch_rows: counters.sparse_expert_batch_rows,
        sparse_expert_batch_routes: counters.sparse_expert_batch_routes,
        sparse_expert_prefill_rows: counters.sparse_expert_prefill_rows,
        sparse_expert_decode_rows: counters.sparse_expert_decode_rows,
        sparse_expert_mtp_verify_rows: counters.sparse_expert_mtp_verify_rows,
        sparse_expert_prefill_routes: counters.sparse_expert_prefill_routes,
        sparse_expert_decode_routes: counters.sparse_expert_decode_routes,
        sparse_expert_mtp_verify_routes: counters.sparse_expert_mtp_verify_routes,
        sparse_expert_host_batch_sets: counters.sparse_expert_host_batch_sets,
        sparse_expert_host_batches: counters.sparse_expert_host_batches,
        sparse_expert_host_batch_rows: counters.sparse_expert_host_batch_rows,
        sparse_expert_host_batch_routes: counters.sparse_expert_host_batch_routes,
        sparse_expert_host_batch_expert_tiles: counters.sparse_expert_host_batch_expert_tiles,
        sparse_expert_host_batch_routes_match_global: counters
            .sparse_expert_host_batch_routes_match_global,
        sparse_expert_host_batch_graph_counts_valid: counters
            .sparse_expert_host_batch_graph_counts_valid,
        sparse_expert_host_request_frames: counters.sparse_expert_host_request_frames,
        sparse_expert_host_request_rows: counters.sparse_expert_host_request_rows,
        sparse_expert_host_request_routes: counters.sparse_expert_host_request_routes,
        sparse_expert_host_request_payload_bytes: counters
            .sparse_expert_host_request_payload_bytes,
        sparse_expert_host_request_wire_bytes: counters.sparse_expert_host_request_wire_bytes,
        sparse_expert_host_response_frames: counters.sparse_expert_host_response_frames,
        sparse_expert_host_response_rows: counters.sparse_expert_host_response_rows,
        sparse_expert_host_response_payload_bytes: counters
            .sparse_expert_host_response_payload_bytes,
        sparse_expert_host_response_wire_bytes: counters.sparse_expert_host_response_wire_bytes,
        sparse_expert_host_wire_envelopes_valid: counters.sparse_expert_host_wire_envelopes_valid,
        kv_read_blocks: counters.kv_read_blocks,
        committed_kv_writes: counters.committed_kv_writes,
        tentative_kv_writes: counters.tentative_kv_writes,
        projected_device_kv_writes: counters.projected_device_kv_writes,
        projected_device_kv_write_bytes: counters.projected_device_kv_write_bytes,
        synthetic_kv_payload_writes: counters.synthetic_kv_payload_writes,
        committed_mtp_writes: kv_report.committed_mtp_writes,
        discarded_mtp_writes: kv_report.discarded_mtp_writes,
        backed_kv_writes: kv_report.backed_kv_writes,
        backed_bytes_after_discard: kv_report.backed_bytes_after_discard,
        kv_reservation_bytes: kv_report.kv_reservation_bytes,
        byte_backed_scheduler_trace: kv_report.byte_backed_scheduler_trace,
        device_kv_status: device_kv.status,
        device_kv_writes: device_kv.writes,
        device_kv_reads: device_kv.reads,
        device_kv_bytes: device_kv.bytes,
        uses_device_kv_cache: device_kv.uses_device_kv_cache,
        device_attention_resident_uploads: device_kv.scheduler_attention_resident_uploads,
        device_attention_resident_buffer_uses: device_kv
            .scheduler_attention_resident_buffer_uses,
        device_attention_resident_query_shapes: device_kv.scheduler_attention_resident_query_shapes,
        device_attention_status: counters.device_attention_status.unwrap_or("not-run"),
        device_attention_launches: counters.device_attention_launches,
        device_attention_rows: counters.device_attention_rows,
        device_attention_query_rows: counters.device_attention_query_rows,
        device_attention_kv_descriptors: counters.device_attention_kv_descriptors,
        device_attention_output_bytes: counters.device_attention_output_bytes,
        device_attention_output_values: counters.device_attention_output_values,
        device_attention_output_finite_values: counters.device_attention_output_finite_values,
        device_attention_output_nonzero_values: counters.device_attention_output_nonzero_values,
        device_attention_output_checksum: counters.device_attention_output_checksum,
        device_attention_hidden_projection_launches: counters
            .device_attention_hidden_projection_launches,
        uses_device_kv_attention: counters.device_attention_launches > 0,
        full_context_device_attention_complete,
        numeric_progression_self_test: numeric_progression_self_test,
        terminal_lm_head_sample,
        layer_order_verified: counters.layer_order_verified,
    },
        sparse_tcp_dispatch_probe,
        final_target_device_hidden,
        target_device_hidden_taps,
    ))
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

#[allow(clippy::too_many_arguments)]
fn finish_stateful_scheduler_execution(
    catalog: &TensorCatalog,
    shape: &RealFullSchedulerExecutionShape,
    store: &KvCacheBackingStore,
    device_kv: &RealFullDeviceKvExecutionMirror,
    numeric_progression: RealFullSchedulerNumericProgression,
    counters: RealFullSchedulerExecutionCounters,
    retain_final_target_device_hidden: bool,
    execution_start: Instant,
    coordinator_graph_stats_before: Option<CoordinatorCudaGraphStats>,
    scheduler_verbose_timing: bool,
    kv_bytes_per_token: usize,
) -> Result<(
    RealFullSchedulerExecutionDryRun,
    Option<RealFullSchedulerSparseTcpDispatchProbe>,
    Option<DeviceBf16Output>,
    Option<RealFullSchedulerTargetHiddenTaps>,
)> {
    let finish_start = Instant::now();
    let numeric_progression_finish = numeric_progression
        .finish_live_request()
        .context("finalizing paired live BF16 scheduler numeric progression")?;
    if scheduler_verbose_timing {
        eprintln!(
            "real_full_scheduler_timing request_id={} stage=numeric_finish elapsed_ms={:.3} total_ms={:.3}",
            shape.request_id,
            elapsed_ms(finish_start),
            elapsed_ms(execution_start)
        );
    }
    let sample_start = Instant::now();
    let terminal_lm_head_sample = if scheduler_should_defer_target_lm_head_sample(
        retain_final_target_device_hidden,
        shape.mtp_rows,
    ) {
        scheduler_deferred_target_lm_head_sample(
            numeric_progression_finish
                .final_decode_device_hidden
                .as_ref(),
        )
    } else {
        scheduler_terminal_lm_head_sample_with_options(
            catalog,
            numeric_progression_finish
                .final_decode_device_hidden
                .as_ref(),
            shape.lm_head_sampling,
        )
    };
    if scheduler_verbose_timing {
        eprintln!(
            "real_full_scheduler_timing request_id={} stage=terminal_sample elapsed_ms={:.3} total_ms={:.3} status={} sampled_token_id={:?}",
            shape.request_id,
            elapsed_ms(sample_start),
            elapsed_ms(execution_start),
            terminal_lm_head_sample.status,
            terminal_lm_head_sample.sampled_token_id
        );
    }
    let RealFullSchedulerNumericProgressionFinish {
        self_test: numeric_progression_self_test,
        final_decode_device_hidden,
        final_target_device_hidden,
        target_device_hidden_taps,
        sparse_tcp_dispatch_probe,
    } = numeric_progression_finish;
    let device_kv = device_kv.summary();
    let full_context_device_attention_complete = scheduler_full_context_device_attention_complete(
        shape,
        &counters,
        numeric_progression_self_test.passed,
        device_kv.uses_device_kv_cache,
    );
    let (expected_attention_launch_min, expected_attention_launch_max) =
        scheduler_device_attention_launch_range(shape);
    let expected_attention_query_rows =
        GLM52_NUM_HIDDEN_LAYERS * (shape.prefill_tokens + shape.decode_rows + shape.mtp_rows);
    let expected_attention_rows = scheduler_expected_device_attention_rows(shape);
    let expected_attention_output_values = expected_attention_query_rows * GLM52_HIDDEN_SIZE;
    if scheduler_verbose_timing {
        eprintln!(
            "real_full_scheduler_attention_gate request_id={} complete={} numeric={} layer_order={} uses_device_kv={} launches={}/{}..{} hidden_projection_launches={}/{} rows={}/{}..{} query_rows={}/{} kv_descriptors={}/{}..{} output_values={}/{}",
            shape.request_id,
            full_context_device_attention_complete,
            numeric_progression_self_test.passed,
            counters.layer_order_verified,
            device_kv.uses_device_kv_cache,
            counters.device_attention_launches,
            expected_attention_launch_min,
            expected_attention_launch_max,
            counters.device_attention_hidden_projection_launches,
            counters.device_attention_launches,
            counters.device_attention_rows,
            expected_attention_query_rows,
            expected_attention_rows,
            counters.device_attention_query_rows,
            expected_attention_query_rows,
            counters.device_attention_kv_descriptors,
            expected_attention_launch_min,
            expected_attention_rows,
            counters.device_attention_output_values,
            expected_attention_output_values
        );
    }
    let status = scheduler_execution_status_for_completion(
        &terminal_lm_head_sample,
        full_context_device_attention_complete,
    );
    let request_prefill_chunks = shape.prefill_tokens.div_ceil(shape.prefill_chunk_tokens);
    let kv_report = scheduler_kv_report_summary(store, shape, kv_bytes_per_token, &counters, true);
    let coordinator_graph_stats_after = coordinator_cuda_graph_stats().ok();
    let (graph_slots, captured_graphs, graph_captures, graph_launches) = match (
        coordinator_graph_stats_before,
        coordinator_graph_stats_after,
    ) {
        (Some(before), Some(after)) => (
            after.slots,
            after.captured_graphs,
            after.graph_captures.saturating_sub(before.graph_captures),
            after.graph_launches.saturating_sub(before.graph_launches),
        ),
        (None, Some(after)) => (after.slots, after.captured_graphs, 0, 0),
        _ => (0, 0, 0, 0),
    };
    Ok((
        RealFullSchedulerExecutionDryRun {
            status,
            scope: "admit full-layer prefix-prefill, later-prefill, decode, and MTP waves, then apply selected host/device KV I/O and sparse ExpertBatch records",
            request_prefill_tokens: shape.prefill_tokens,
            request_prefill_chunks,
            request_decode_rows: shape.decode_rows,
            request_mtp_verify_rows: shape.mtp_rows,
            request_mtp_accepted_rows: shape.mtp_accepted_rows,
            request_coordinator_graph_slots: graph_slots,
            request_coordinator_graph_captured_graphs: captured_graphs,
            request_coordinator_graph_captures: graph_captures,
            request_coordinator_graph_launches: graph_launches,
            iterations: counters.iterations,
            candidate_layerwaves: counters.candidate_layerwaves,
            selected_layerwaves: counters.selected_layerwaves,
            deferred_layerwaves: counters.deferred_layerwaves,
            selected_decode_rows: counters.selected_decode_rows,
            selected_prefill_rows: counters.selected_prefill_rows,
            selected_mtp_rows: counters.selected_mtp_rows,
            sparse_expert_batches: counters.sparse_expert_batches,
            sparse_expert_batch_rows: counters.sparse_expert_batch_rows,
            sparse_expert_batch_routes: counters.sparse_expert_batch_routes,
            sparse_expert_prefill_rows: counters.sparse_expert_prefill_rows,
            sparse_expert_decode_rows: counters.sparse_expert_decode_rows,
            sparse_expert_mtp_verify_rows: counters.sparse_expert_mtp_verify_rows,
            sparse_expert_prefill_routes: counters.sparse_expert_prefill_routes,
            sparse_expert_decode_routes: counters.sparse_expert_decode_routes,
            sparse_expert_mtp_verify_routes: counters.sparse_expert_mtp_verify_routes,
            sparse_expert_host_batch_sets: counters.sparse_expert_host_batch_sets,
            sparse_expert_host_batches: counters.sparse_expert_host_batches,
            sparse_expert_host_batch_rows: counters.sparse_expert_host_batch_rows,
            sparse_expert_host_batch_routes: counters.sparse_expert_host_batch_routes,
            sparse_expert_host_batch_expert_tiles: counters.sparse_expert_host_batch_expert_tiles,
            sparse_expert_host_batch_routes_match_global: counters
                .sparse_expert_host_batch_routes_match_global,
            sparse_expert_host_batch_graph_counts_valid: counters
                .sparse_expert_host_batch_graph_counts_valid,
            sparse_expert_host_request_frames: counters.sparse_expert_host_request_frames,
            sparse_expert_host_request_rows: counters.sparse_expert_host_request_rows,
            sparse_expert_host_request_routes: counters.sparse_expert_host_request_routes,
            sparse_expert_host_request_payload_bytes: counters
                .sparse_expert_host_request_payload_bytes,
            sparse_expert_host_request_wire_bytes: counters.sparse_expert_host_request_wire_bytes,
            sparse_expert_host_response_frames: counters.sparse_expert_host_response_frames,
            sparse_expert_host_response_rows: counters.sparse_expert_host_response_rows,
            sparse_expert_host_response_payload_bytes: counters
                .sparse_expert_host_response_payload_bytes,
            sparse_expert_host_response_wire_bytes: counters
                .sparse_expert_host_response_wire_bytes,
            sparse_expert_host_wire_envelopes_valid: counters
                .sparse_expert_host_wire_envelopes_valid,
            kv_read_blocks: counters.kv_read_blocks,
            committed_kv_writes: counters.committed_kv_writes,
            tentative_kv_writes: counters.tentative_kv_writes,
            projected_device_kv_writes: counters.projected_device_kv_writes,
            projected_device_kv_write_bytes: counters.projected_device_kv_write_bytes,
            synthetic_kv_payload_writes: counters.synthetic_kv_payload_writes,
            committed_mtp_writes: kv_report.committed_mtp_writes,
            discarded_mtp_writes: kv_report.discarded_mtp_writes,
            backed_kv_writes: kv_report.backed_kv_writes,
            backed_bytes_after_discard: kv_report.backed_bytes_after_discard,
            kv_reservation_bytes: kv_report.kv_reservation_bytes,
            byte_backed_scheduler_trace: kv_report.byte_backed_scheduler_trace,
            device_kv_status: device_kv.status,
            device_kv_writes: device_kv.writes,
            device_kv_reads: device_kv.reads,
            device_kv_bytes: device_kv.bytes,
            uses_device_kv_cache: device_kv.uses_device_kv_cache,
            device_attention_resident_uploads: device_kv.scheduler_attention_resident_uploads,
            device_attention_resident_buffer_uses: device_kv
                .scheduler_attention_resident_buffer_uses,
            device_attention_resident_query_shapes: device_kv
                .scheduler_attention_resident_query_shapes,
            device_attention_status: counters.device_attention_status.unwrap_or("not-run"),
            device_attention_launches: counters.device_attention_launches,
            device_attention_rows: counters.device_attention_rows,
            device_attention_query_rows: counters.device_attention_query_rows,
            device_attention_kv_descriptors: counters.device_attention_kv_descriptors,
            device_attention_output_bytes: counters.device_attention_output_bytes,
            device_attention_output_values: counters.device_attention_output_values,
            device_attention_output_finite_values: counters
                .device_attention_output_finite_values,
            device_attention_output_nonzero_values: counters
                .device_attention_output_nonzero_values,
            device_attention_output_checksum: counters.device_attention_output_checksum,
            device_attention_hidden_projection_launches: counters
                .device_attention_hidden_projection_launches,
            uses_device_kv_attention: counters.device_attention_launches > 0,
            full_context_device_attention_complete,
            numeric_progression_self_test,
            terminal_lm_head_sample,
            layer_order_verified: counters.layer_order_verified,
        },
        sparse_tcp_dispatch_probe,
        final_target_device_hidden,
        target_device_hidden_taps,
    ))
}

fn scheduler_kv_report_summary(
    store: &KvCacheBackingStore,
    shape: &RealFullSchedulerExecutionShape,
    kv_bytes_per_token: usize,
    counters: &RealFullSchedulerExecutionCounters,
    stateful_live_request: bool,
) -> SchedulerKvReportSummary {
    if stateful_live_request {
        return SchedulerKvReportSummary {
            committed_mtp_writes: 0,
            discarded_mtp_writes: 0,
            backed_kv_writes: counters.committed_kv_writes,
            backed_bytes_after_discard: 0,
            kv_reservation_bytes: shape.reservation_tokens() * kv_bytes_per_token,
            byte_backed_scheduler_trace: false,
        };
    }

    let snapshot = store.snapshot();
    let backed_kv_writes = store.backed_write_count();
    let backed_bytes_after_discard = store.backed_write_bytes();
    let byte_backed_scheduler_trace = backed_kv_writes
        == counters.committed_kv_writes + snapshot.committed_writes
        && backed_bytes_after_discard <= snapshot.resident_bytes
        && snapshot.active_reservations == 1;
    SchedulerKvReportSummary {
        committed_mtp_writes: snapshot.committed_writes,
        discarded_mtp_writes: snapshot.discarded_writes,
        backed_kv_writes,
        backed_bytes_after_discard,
        kv_reservation_bytes: snapshot.resident_bytes,
        byte_backed_scheduler_trace,
    }
}

fn scheduler_execution_timing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        scheduler_execution_timing_env_enabled(SCHEDULER_EXECUTION_TIMING_ENV)
            || scheduler_execution_timing_env_enabled(SCHEDULER_EXECUTION_SUMMARY_TIMING_ENV)
    })
}

fn scheduler_execution_verbose_timing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| scheduler_execution_timing_env_enabled(SCHEDULER_EXECUTION_TIMING_ENV))
}

fn scheduler_execution_timing_env_enabled(name: &str) -> bool {
    env::var(name)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn validate_scheduler_execution_shape(shape: &RealFullSchedulerExecutionShape) -> Result<()> {
    anyhow::ensure!(
        shape.prefill_chunk_tokens > 0,
        "real-full scheduler execution requires a nonzero prefill chunk size"
    );
    anyhow::ensure!(
        shape.decode_rows > 0,
        "real-full scheduler execution requires at least one decode row"
    );
    anyhow::ensure!(
        shape.mtp_accepted_rows <= shape.mtp_rows,
        "real-full scheduler execution accepted MTP rows exceed MTP rows"
    );
    Ok(())
}

fn plan_scheduler_prefill_chunks(
    shape: &RealFullSchedulerExecutionShape,
    layer: LayerId,
    reservation_id: u64,
    priority: Priority,
    policy: &PrefillChunkPolicy,
) -> Vec<PrefillChunk> {
    let chunk_tokens = policy.chunk_tokens.max(1);
    let graph_bucket = policy.graph_bucket();
    let mut chunks = Vec::new();
    let mut offset = 0_usize;
    while offset < shape.prefill_tokens {
        let token_count = (shape.prefill_tokens - offset).min(chunk_tokens);
        chunks.push(PrefillChunk::new(
            shape.request_id.as_str(),
            shape.sequence_id.as_str(),
            layer,
            PositionId((shape.prefill_token_start() + offset) as u64),
            token_count,
            reservation_id,
            priority,
            graph_bucket,
            shape.placement_version.as_str(),
        ));
        offset += token_count;
    }
    chunks
}

fn scheduler_execution_status_for_completion(
    sample: &RealFullSchedulerTerminalLmHeadSample,
    full_context_device_attention_complete: bool,
) -> &'static str {
    if sample.passed && full_context_device_attention_complete {
        "admitted-scheduler-terminal-lm-head-sampled"
    } else if sample.uses_final_decode_device_hidden {
        "admitted-scheduler-terminal-lm-head-blocked"
    } else {
        "admitted-scheduler-dry-run"
    }
}

fn scheduler_full_context_device_attention_complete(
    shape: &RealFullSchedulerExecutionShape,
    counters: &RealFullSchedulerExecutionCounters,
    numeric_progression_passed: bool,
    uses_device_kv_cache: bool,
) -> bool {
    let (expected_launch_min, expected_launch_max) = scheduler_device_attention_launch_range(shape);
    let expected_rows = scheduler_expected_device_attention_rows(shape);
    let expected_query_rows =
        GLM52_NUM_HIDDEN_LAYERS * (shape.prefill_tokens + shape.decode_rows + shape.mtp_rows);
    let expected_output_values = expected_query_rows * GLM52_HIDDEN_SIZE;
    let attention_rows_complete = counters.device_attention_rows >= expected_query_rows
        && counters.device_attention_rows <= expected_rows;
    numeric_progression_passed
        && counters.layer_order_verified
        && uses_device_kv_cache
        && (expected_launch_min..=expected_launch_max).contains(&counters.device_attention_launches)
        && counters.device_attention_hidden_projection_launches
            == counters.device_attention_launches
        && attention_rows_complete
        && counters.device_attention_query_rows == expected_query_rows
        && counters.device_attention_kv_descriptors >= counters.device_attention_launches
        && counters.device_attention_kv_descriptors <= expected_rows
        && counters.device_attention_output_values == expected_output_values
}

fn scheduler_device_attention_launch_range(
    shape: &RealFullSchedulerExecutionShape,
) -> (usize, usize) {
    let prefill_chunks = shape.prefill_tokens.div_ceil(shape.prefill_chunk_tokens);
    let batched_launches_per_layer =
        prefill_chunks + shape.decode_rows + usize::from(shape.mtp_rows > 0);
    let target_rows = shape.decode_rows.saturating_add(shape.mtp_rows);
    let can_fuse_target = shape.decode_rows == 1
        && shape.mtp_rows > 0
        && target_rows <= admission::MTP_TARGET_ATTENTION_FUSION_MAX_ROWS;
    let launches_per_layer_min = batched_launches_per_layer - usize::from(can_fuse_target);
    let launches_per_layer_max = prefill_chunks + shape.decode_rows + shape.mtp_rows;
    (
        GLM52_NUM_HIDDEN_LAYERS * launches_per_layer_min,
        GLM52_NUM_HIDDEN_LAYERS * launches_per_layer_max,
    )
}

fn scheduler_expected_device_attention_rows(shape: &RealFullSchedulerExecutionShape) -> usize {
    let prefill_chunks = shape.prefill_tokens.div_ceil(shape.prefill_chunk_tokens);
    let mut rows_per_layer = 0_usize;
    for chunk_index in 0..prefill_chunks {
        let chunk_start = shape.prefix_tokens + chunk_index * shape.prefill_chunk_tokens;
        let remaining_prefill = shape
            .prefill_tokens
            .saturating_sub(chunk_index * shape.prefill_chunk_tokens);
        let chunk_rows = remaining_prefill.min(shape.prefill_chunk_tokens);
        rows_per_layer += chunk_start + chunk_rows;
    }
    for decode_offset in 0..shape.decode_rows {
        rows_per_layer += shape.prefix_tokens + shape.prefill_tokens + decode_offset + 1;
    }
    let mtp_context_start = shape.prefix_tokens + shape.prefill_tokens + shape.decode_rows;
    for mtp_offset in 0..shape.mtp_rows {
        rows_per_layer += mtp_context_start + mtp_offset + 1;
    }
    GLM52_NUM_HIDDEN_LAYERS * rows_per_layer
}

fn scheduler_terminal_lm_head_sample(
    catalog: &TensorCatalog,
    final_decode_device_hidden: Option<&DeviceBf16Output>,
) -> RealFullSchedulerTerminalLmHeadSample {
    scheduler_terminal_lm_head_sample_with_options(
        catalog,
        final_decode_device_hidden,
        RealFullLmHeadSamplingOptions::diagnostic(),
    )
}

fn scheduler_terminal_lm_head_sample_with_options(
    catalog: &TensorCatalog,
    final_decode_device_hidden: Option<&DeviceBf16Output>,
    sampler_options: RealFullLmHeadSamplingOptions,
) -> RealFullSchedulerTerminalLmHeadSample {
    let Some(hidden) = final_decode_device_hidden else {
        return RealFullSchedulerTerminalLmHeadSample {
            status: "not-run",
            scope: "sample from the scheduler final decode device hidden row through the full-resident BF16 lm_head sampler",
            uses_final_decode_device_hidden: false,
            covers_full_vocabulary: false,
            hidden_dim: 0,
            vocab_size: 0,
            logits_evaluated: 0,
            top_token_id: None,
            sampled_token_id: None,
            sample_top_k: None,
            sample_top_p: None,
            argmax_kernel_backend: None,
            sampler_kernel_backend: None,
            passed: false,
            blocker: Some(
                "scheduler final decode device hidden is unavailable for terminal sampling"
                    .to_owned(),
            ),
        };
    };

    if scheduler_terminal_sample_validate_enabled() {
        if let Err(error) =
            log_terminal_sample_device_bf16_validation("final_decode_hidden", hidden)
        {
            eprintln!(
                "real_full_terminal_sample_validation stage=final_decode_hidden status=error error={error:#}"
            );
        }
    }

    let normalized_hidden = match scheduler_final_norm_device_hidden(catalog, hidden) {
        Ok(hidden) => hidden,
        Err(error) => {
            return RealFullSchedulerTerminalLmHeadSample {
                status: "blocked",
                scope: "sample from the scheduler final-normed decode device hidden row through the full-resident BF16 lm_head sampler",
                uses_final_decode_device_hidden: true,
                covers_full_vocabulary: false,
                hidden_dim: hidden.values_per_row,
                vocab_size: 0,
                logits_evaluated: 0,
                top_token_id: None,
                sampled_token_id: None,
                sample_top_k: None,
                sample_top_p: None,
                argmax_kernel_backend: None,
                sampler_kernel_backend: None,
                passed: false,
                blocker: Some(format!("{error:#}")),
            };
        }
    };

    if scheduler_terminal_sample_validate_enabled() {
        if let Err(error) =
            log_terminal_sample_device_bf16_validation("final_norm_hidden", &normalized_hidden)
        {
            eprintln!(
                "real_full_terminal_sample_validation stage=final_norm_hidden status=error error={error:#}"
            );
        }
    }

    match score_real_lm_head_full_vocab_for_device_hidden_with_options(
        catalog,
        &normalized_hidden,
        1,
        sampler_options,
    ) {
        Ok(sample) => {
            let passed = scheduler_terminal_lm_head_completion_gate(&sample);
            RealFullSchedulerTerminalLmHeadSample {
                status: if passed { "sampled" } else { "blocked" },
                scope: "sample from the scheduler final-normed decode device hidden row through the full-resident BF16 lm_head sampler",
                uses_final_decode_device_hidden: true,
                covers_full_vocabulary: sample.covers_full_vocabulary,
                hidden_dim: sample.hidden_dim,
                vocab_size: sample.vocab_size,
                logits_evaluated: sample.logits_evaluated,
                top_token_id: Some(sample.top_token_id),
                sampled_token_id: Some(sample.sampled_token_id),
                sample_top_k: Some(sample.sample_top_k),
                sample_top_p: Some(sample.sample_top_p),
                argmax_kernel_backend: Some(sample.argmax_kernel_backend),
                sampler_kernel_backend: Some(sample.sampler_kernel_backend),
                passed,
                blocker: (!passed).then(|| {
                    format!(
                        "scheduler terminal lm_head sample must use full-vocabulary preloaded-resident CUDA argmax plus non-greedy top-k/top-p sampler: full_vocab={} logits={} vocab={} top_k={} top_p={} temperature={} argmax_backend={} sampler_backend={}",
                        sample.covers_full_vocabulary,
                        sample.logits_evaluated,
                        sample.vocab_size,
                        sample.sample_top_k,
                        sample.sample_top_p,
                        sample.sample_temperature,
                        sample.argmax_kernel_backend,
                        sample.sampler_kernel_backend,
                    )
                }),
            }
        }
        Err(error) => RealFullSchedulerTerminalLmHeadSample {
            status: "blocked",
            scope: "sample from the scheduler final-normed decode device hidden row through the full-resident BF16 lm_head sampler",
            uses_final_decode_device_hidden: true,
            covers_full_vocabulary: false,
            hidden_dim: hidden.values_per_row,
            vocab_size: 0,
            logits_evaluated: 0,
            top_token_id: None,
            sampled_token_id: None,
            sample_top_k: None,
            sample_top_p: None,
            argmax_kernel_backend: None,
            sampler_kernel_backend: None,
            passed: false,
            blocker: Some(format!("{error:#}")),
        },
    }
}

fn scheduler_deferred_target_lm_head_sample(
    final_decode_device_hidden: Option<&DeviceBf16Output>,
) -> RealFullSchedulerTerminalLmHeadSample {
    let Some(hidden) = final_decode_device_hidden else {
        return RealFullSchedulerTerminalLmHeadSample {
            status: "not-run",
            scope: "defer speculative terminal sampling to the retained target hidden batch",
            uses_final_decode_device_hidden: false,
            covers_full_vocabulary: false,
            hidden_dim: 0,
            vocab_size: 0,
            logits_evaluated: 0,
            top_token_id: None,
            sampled_token_id: None,
            sample_top_k: None,
            sample_top_p: None,
            argmax_kernel_backend: None,
            sampler_kernel_backend: None,
            passed: false,
            blocker: Some(
                "scheduler final decode device hidden is unavailable for deferred target sampling"
                    .to_owned(),
            ),
        };
    };
    RealFullSchedulerTerminalLmHeadSample {
        status: "deferred-target-batch",
        scope: "defer speculative terminal sampling to the retained target hidden batch",
        uses_final_decode_device_hidden: true,
        covers_full_vocabulary: false,
        hidden_dim: hidden.values_per_row,
        vocab_size: 0,
        logits_evaluated: 0,
        top_token_id: None,
        sampled_token_id: None,
        sample_top_k: None,
        sample_top_p: None,
        argmax_kernel_backend: None,
        sampler_kernel_backend: None,
        // The caller cannot emit this placeholder: a failed batched sample
        // aborts the request, while a successful one replaces these fields.
        passed: true,
        blocker: None,
    }
}

fn scheduler_should_defer_target_lm_head_sample(
    retain_final_target_device_hidden: bool,
    mtp_rows: usize,
) -> bool {
    retain_final_target_device_hidden && mtp_rows > 0
}

fn scheduler_final_norm_device_hidden(
    catalog: &TensorCatalog,
    hidden: &DeviceBf16Output,
) -> Result<DeviceBf16Output> {
    let final_norm = catalog
        .tensors
        .iter()
        .find(|tensor| tensor.name == REAL_FULL_FINAL_NORM_WEIGHT_NAME)
        .with_context(|| {
            format!("scheduler terminal sampling requires {REAL_FULL_FINAL_NORM_WEIGHT_NAME}")
        })?;
    if final_norm.dtype != DType::Bf16
        || final_norm.role != TensorRole::Norm
        || final_norm.shape != vec![hidden.values_per_row]
    {
        anyhow::bail!(
            "scheduler terminal final_norm tensor mismatch: name={} dtype={:?} role={:?} shape={:?} expected BF16 Norm [{}]",
            final_norm.name,
            final_norm.dtype,
            final_norm.role,
            final_norm.shape,
            hidden.values_per_row
        );
    }
    rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output(
        REAL_FULL_FINAL_NORM_WEIGHT_NAME,
        hidden.buffer(),
        hidden.rows,
        hidden.values_per_row,
        REAL_FULL_FINAL_NORM_EPS,
    )
    .context("applying scheduler terminal final RMSNorm before lm_head sampling")
}

fn scheduler_terminal_sample_validate_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(SCHEDULER_TERMINAL_SAMPLE_VALIDATE_ENV)
            .map(|value| {
                let value = value.trim();
                !(value.is_empty()
                    || value == "0"
                    || value.eq_ignore_ascii_case("false")
                    || value.eq_ignore_ascii_case("off"))
            })
            .unwrap_or(false)
    })
}

fn log_terminal_sample_device_bf16_validation(
    stage: &'static str,
    hidden: &DeviceBf16Output,
) -> Result<()> {
    let bytes = hidden.copy_to_host_bytes().with_context(|| {
        format!("copying scheduler terminal sample {stage} device BF16 buffer to host")
    })?;
    let values = bf16_values_to_f32(&bytes);
    let mut finite = 0_usize;
    let mut nonzero = 0_usize;
    let mut checksum = 0.0_f64;
    let mut finite_min = f32::INFINITY;
    let mut finite_max = f32::NEG_INFINITY;
    let mut first_nonfinite: Option<(usize, u16, f32)> = None;
    for (index, value) in values.iter().copied().enumerate() {
        if value.is_finite() {
            finite += 1;
            checksum += value as f64;
            finite_min = finite_min.min(value);
            finite_max = finite_max.max(value);
        } else if first_nonfinite.is_none() {
            let byte_index = index * std::mem::size_of::<u16>();
            let bits = u16::from_le_bytes([bytes[byte_index], bytes[byte_index + 1]]);
            first_nonfinite = Some((index, bits, value));
        }
        if value != 0.0 {
            nonzero += 1;
        }
    }
    if finite == 0 {
        finite_min = 0.0;
        finite_max = 0.0;
    }
    let (first_nonfinite_index, first_nonfinite_bits, first_nonfinite_value) = first_nonfinite
        .map(|(index, bits, value)| (index as isize, bits, value))
        .unwrap_or((-1, 0, 0.0));
    eprintln!(
        "real_full_terminal_sample_validation stage={} backend={} rows={} hidden_dim={} values={} finite={} nonzero={} checksum={:.6e} finite_min={:.6e} finite_max={:.6e} first_nonfinite_index={} first_nonfinite_bits=0x{:04x} first_nonfinite_value={}",
        stage,
        hidden.backend,
        hidden.rows,
        hidden.values_per_row,
        values.len(),
        finite,
        nonzero,
        checksum,
        finite_min,
        finite_max,
        first_nonfinite_index,
        first_nonfinite_bits,
        first_nonfinite_value
    );
    Ok(())
}

fn scheduler_terminal_lm_head_completion_gate(sample: &RealLmHeadChunkScoreForHidden) -> bool {
    sample.covers_full_vocabulary
        && sample.logits_evaluated == sample.vocab_size
        && sample.top_logit.is_finite()
        && sample.sampled_score.is_finite()
        && sample.sampled_score > 0.0
        && sample.sampled_score <= 1.0
        && sample.sample_top_k > 1
        && sample.sample_top_p.is_finite()
        && sample.sample_top_p > 0.0
        && sample.sample_top_p <= 1.0
        && sample.sample_temperature.is_finite()
        && sample.sample_temperature > 0.0
        && scheduler_terminal_lm_head_argmax_backend_allowed(sample.argmax_kernel_backend)
        && scheduler_terminal_lm_head_sampler_backend_allowed(sample.sampler_kernel_backend)
}

fn scheduler_terminal_lm_head_argmax_backend_allowed(backend: &str) -> bool {
    matches!(
        backend,
        CUDA_REFERENCE_LM_HEAD_ARGMAX_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
    )
}

fn scheduler_terminal_lm_head_sampler_backend_allowed(backend: &str) -> bool {
    matches!(
        backend,
        CUDA_REFERENCE_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
            | TRITON_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
    )
}

#[cfg(test)]
mod tests {
    use super::bounded_long_prefill_wavefront_required;
    use super::next_ready_sparse_wavefront_task;
    use super::plan_scheduler_prefill_chunks;
    use super::real_full_scheduler_execution_for_shape_with_sparse_tcp;
    use super::scheduler_device_attention_launch_range;
    use super::scheduler_execution_status_for_completion;
    use super::scheduler_expected_device_attention_rows;
    use super::scheduler_full_context_device_attention_complete;
    use super::scheduler_should_defer_target_lm_head_sample;
    use super::scheduler_sparse_tcp_iterations_per_sparse_layer;
    use super::scheduler_terminal_lm_head_completion_gate;
    use super::scheduler_terminal_lm_head_sample;
    use super::RealFullMtpAdaptiveDraftState;
    use super::RealFullSchedulerExecutionCounters;
    use super::RealFullSchedulerExecutionShape;
    use super::MAX_PENDING_SPARSE_WAVEFRONT_DISPATCHES;
    use super::REAL_FULL_FINAL_NORM_WEIGHT_NAME;
    use crate::commands::real_full::constants::{
        REAL_FULL_PREFLIGHT_DECODE_ROWS, REAL_FULL_PREFLIGHT_MTP_ROWS,
        REAL_FULL_PREFLIGHT_PREFILL_ROWS, REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START,
    };
    use crate::commands::real_full::coordinator_kernels::{
        cuda_native_library, cuda_reference_kernels_test_override,
        device_bf16_output_from_f32_values, preload_resident_weight_from_host_staging,
        CUDA_REFERENCE_LM_HEAD_ARGMAX_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
        CUDA_REFERENCE_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
        TRITON_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
    };
    use crate::commands::real_full::sampling::{
        RealFullLmHeadSamplingOptions, RealLmHeadChunkScoreForHidden,
    };

    #[test]
    fn scheduler_attention_launch_range_accounts_for_fused_mtp_target_rows() {
        let mut shape = RealFullSchedulerExecutionShape::preflight();
        shape.prefill_tokens = 0;
        shape.decode_rows = 1;
        shape.mtp_rows = 6;
        assert_eq!(
            scheduler_device_attention_launch_range(&shape),
            (GLM52_NUM_HIDDEN_LAYERS, GLM52_NUM_HIDDEN_LAYERS * 7)
        );

        shape.mtp_rows = 8;
        assert_eq!(
            scheduler_device_attention_launch_range(&shape),
            (GLM52_NUM_HIDDEN_LAYERS, GLM52_NUM_HIDDEN_LAYERS * 9)
        );

        shape.mtp_rows = 0;
        assert_eq!(
            scheduler_device_attention_launch_range(&shape),
            (GLM52_NUM_HIDDEN_LAYERS, GLM52_NUM_HIDDEN_LAYERS)
        );
    }
    use crate::commands::real_full::types::RealFullSchedulerTerminalLmHeadSample;
    use glmrt_core::{
        DType, KvCacheConfig, LayerId, ModelFacts, PrefillChunkPolicy, Priority, TensorCatalog,
        TensorInfo, TensorRole, GLM52_FIRST_K_DENSE_REPLACE, GLM52_HIDDEN_SIZE,
        GLM52_NUM_HIDDEN_LAYERS,
    };
    use glmrt_transport::TcpProtocolV2HostBatchTarget;

    #[test]
    fn adaptive_mtp_draft_width_expands_after_profitable_saturated_cycles() {
        let mut state = RealFullMtpAdaptiveDraftState::default();
        assert_eq!(state.width(1, 7, 4, true), 4);
        state.observe(1, 7, 4, 4, 2, true, false);
        assert_eq!(state.width(1, 7, 4, true), 4);

        for _ in 0..3 {
            state.observe(1, 7, 4, 4, 3, true, false);
        }
        assert_eq!(state.width(1, 7, 4, true), 5);
    }

    #[test]
    fn adaptive_mtp_tolerates_one_framing_miss_before_profitable_content() {
        let mut state = RealFullMtpAdaptiveDraftState::default();
        assert_eq!(state.width(1, 7, 6, true), 6);
        state.observe(1, 7, 6, 6, 1, true, false);
        assert_eq!(state.width(1, 7, 6, true), 6);
        state.observe(1, 7, 6, 6, 6, true, false);
        assert_eq!(state.width(1, 7, 6, true), 6);
        assert_eq!(state.below_scalar_gate_streak, 0);
    }

    #[test]
    fn adaptive_mtp_falls_back_to_scalar_and_periodically_probes_d1() {
        let mut state = RealFullMtpAdaptiveDraftState::default();
        assert_eq!(state.width(1, 7, 6, true), 6);
        state.observe(1, 7, 6, 6, 2, true, false);
        assert_eq!(state.width(1, 7, 6, true), 6);
        state.observe(1, 7, 6, 6, 0, true, false);
        assert_eq!(state.current, Some(0));

        for _ in 1..super::REAL_FULL_MTP_SCALAR_PROBE_INITIAL_INTERVAL {
            assert_eq!(state.width(1, 7, 6, true), 0);
        }
        assert_eq!(state.width(1, 7, 6, true), 1);
        state.observe(1, 7, 6, 1, 0, true, false);
        assert_eq!(state.current, Some(0));
        assert_eq!(
            state.scalar_probe_interval,
            super::REAL_FULL_MTP_SCALAR_PROBE_INITIAL_INTERVAL * 2
        );

        for success_index in 0..super::REAL_FULL_MTP_SCALAR_PROBE_SUCCESSES_REQUIRED {
            for _ in 0..state.scalar_probe_interval {
                if state.width(1, 7, 6, true) == 1 {
                    break;
                }
            }
            state.observe(1, 7, 6, 1, 1, true, false);
            if success_index + 1 < super::REAL_FULL_MTP_SCALAR_PROBE_SUCCESSES_REQUIRED {
                assert_eq!(state.current, Some(0));
            }
        }
        assert_eq!(state.width(1, 7, 6, true), 2);
    }

    #[test]
    fn mtp_scalar_gate_uses_measured_physical_width_costs() {
        assert!(!super::real_full_mtp_observation_meets_scalar_gate(
            6, 2, false
        ));
        assert!(super::real_full_mtp_observation_meets_scalar_gate(
            6, 3, false
        ));
        assert!(!super::real_full_mtp_observation_meets_scalar_gate(
            4, 1, false
        ));
        assert!(super::real_full_mtp_observation_meets_scalar_gate(
            4, 2, false
        ));
        assert!(!super::real_full_mtp_observation_meets_scalar_gate(
            2, 1, false
        ));
        assert!(super::real_full_mtp_observation_meets_scalar_gate(
            2, 2, false
        ));
        assert!(!super::real_full_mtp_observation_meets_scalar_gate(
            1, 1, false
        ));
        assert!(super::real_full_mtp_observation_meets_scalar_gate(
            1, 1, true
        ));
    }

    #[test]
    fn adaptive_mtp_can_hold_physical_m2_between_scalar_and_wider_speculation() {
        let mut state = RealFullMtpAdaptiveDraftState::default();
        assert_eq!(state.width(1, 7, 2, true), 2);

        state.observe(1, 7, 2, 2, 1, true, true);
        assert_eq!(state.width(1, 7, 2, true), 2);
        state.observe(1, 7, 2, 2, 1, true, true);
        assert_eq!(state.width(1, 7, 2, true), 1);

        for _ in 0..3 {
            state.observe(1, 7, 2, 1, 1, true, true);
        }
        assert_eq!(state.width(1, 7, 2, true), 2);
    }

    #[test]
    fn adaptive_mtp_physical_m2_tolerates_one_miss_then_uses_scalar() {
        let mut state = RealFullMtpAdaptiveDraftState {
            current: Some(1),
            ..RealFullMtpAdaptiveDraftState::default()
        };
        state.observe(1, 7, 2, 1, 0, true, true);
        assert_eq!(state.width(1, 7, 2, true), 1);
        state.observe(1, 7, 2, 1, 0, true, true);
        assert_eq!(state.current, Some(0));
    }

    #[test]
    fn adaptive_mtp_profitable_scalar_probe_enters_lowest_candidate() {
        let mut state = RealFullMtpAdaptiveDraftState {
            current: Some(0),
            scalar_probe_in_flight: true,
            ..RealFullMtpAdaptiveDraftState::default()
        };
        assert_eq!(state.width(2, 7, 6, true), 2);
        state.observe(2, 7, 6, 2, 2, true, false);
        assert_eq!(state.width(2, 7, 6, true), 2);
    }

    #[test]
    fn adaptive_mtp_draft_width_ignores_budget_limited_tail_and_bounds_history() {
        let mut state = RealFullMtpAdaptiveDraftState::default();
        assert_eq!(state.width(2, 8, 7, true), 7);
        for _ in 0..24 {
            state.observe(2, 8, 7, 6, 0, true, false);
        }
        assert_eq!(state.width(2, 8, 7, true), 7);
        assert!(state.history.is_empty());

        for _ in 0..24 {
            state.observe(2, 8, 7, 7, 3, true, false);
        }
        assert!(state.history.len() <= super::REAL_FULL_MTP_ADAPTIVE_HISTORY_LIMIT);
    }

    #[test]
    fn fixed_mtp_draft_width_does_not_react_to_acceptance() {
        let mut state = RealFullMtpAdaptiveDraftState::default();
        assert_eq!(state.width(4, 4, 4, false), 4);
        for _ in 0..4 {
            state.observe(4, 4, 4, 4, 0, false, false);
        }
        assert_eq!(state.width(4, 4, 4, false), 4);
        assert!(state.history.is_empty());
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    #[test]
    fn sparse_wavefront_preserves_chunk_and_layer_dependencies() {
        const LAYERS: usize = 3;
        const CHUNKS: usize = 4;

        let mut started = vec![vec![false; CHUNKS]; LAYERS];
        let mut finished = vec![vec![false; CHUNKS]; LAYERS];
        let mut pending = std::collections::VecDeque::new();
        let mut completed = 0;

        while completed < LAYERS * CHUNKS {
            while pending.len() < MAX_PENDING_SPARSE_WAVEFRONT_DISPATCHES {
                let Some(task) = next_ready_sparse_wavefront_task(&started, &finished) else {
                    break;
                };
                assert!(
                    task.layer_offset == 0 || finished[task.layer_offset - 1][task.chunk_index]
                );
                assert!(task.chunk_index == 0 || started[task.layer_offset][task.chunk_index - 1]);
                started[task.layer_offset][task.chunk_index] = true;
                pending.push_back(task);
            }

            let task = pending.pop_front().expect("wavefront must not stall");
            finished[task.layer_offset][task.chunk_index] = true;
            completed += 1;
        }

        assert!(started.iter().flatten().all(|started| *started));
        assert!(finished.iter().flatten().all(|finished| *finished));
    }

    fn tiny_lm_head_catalog(vocab: usize, hidden_dim: usize, byte_length: usize) -> TensorCatalog {
        TensorCatalog {
            model_id: "test".to_owned(),
            snapshot_path: ".".to_owned(),
            facts: ModelFacts::default(),
            tensors: vec![
                TensorInfo {
                    name: "lm_head.weight".to_owned(),
                    file: "lm_head.safetensors".to_owned(),
                    dtype: DType::Bf16,
                    shape: vec![vocab, hidden_dim],
                    byte_offset: 0,
                    byte_length: byte_length as u64,
                    role: TensorRole::LmHead,
                    layer_id: None,
                    expert_id: None,
                    is_quantization_metadata: false,
                },
                TensorInfo {
                    name: REAL_FULL_FINAL_NORM_WEIGHT_NAME.to_owned(),
                    file: "final_norm.safetensors".to_owned(),
                    dtype: DType::Bf16,
                    shape: vec![hidden_dim],
                    byte_offset: 0,
                    byte_length: (hidden_dim * std::mem::size_of::<u16>()) as u64,
                    role: TensorRole::Norm,
                    layer_id: None,
                    expert_id: None,
                    is_quantization_metadata: false,
                },
            ],
        }
    }

    #[test]
    fn scheduler_execution_shape_places_mtp_after_decode_rows() {
        let shape = RealFullSchedulerExecutionShape {
            request_id: "request-shaped-mtp-placement".to_owned(),
            sequence_id: "request-shaped-mtp-placement-sequence".to_owned(),
            placement_version: "request-shaped-mtp-placement-version".to_owned(),
            prefix_tokens: 0,
            prefill_tokens: 9,
            prefill_chunk_tokens: 4,
            decode_rows: 2,
            mtp_rows: 3,
            mtp_accepted_rows: 1,
            prefill_token_ids: None,
            prefill_vision_embeddings: None,
            decode_token_ids: None,
            lm_head_sampling: RealFullLmHeadSamplingOptions::diagnostic(),
        };

        assert_eq!(shape.decode_token_start(), 9);
        assert_eq!(shape.mtp_token_start(), 11);
        assert_eq!(shape.reservation_tokens(), 14);
    }

    #[test]
    fn scheduler_execution_shape_offsets_active_rows_after_prefix_context() {
        let shape = RealFullSchedulerExecutionShape {
            request_id: "request-shaped-prefix-placement".to_owned(),
            sequence_id: "request-shaped-prefix-placement-sequence".to_owned(),
            placement_version: "request-shaped-prefix-placement-version".to_owned(),
            prefix_tokens: 128,
            prefill_tokens: 3,
            prefill_chunk_tokens: 2,
            decode_rows: 2,
            mtp_rows: 1,
            mtp_accepted_rows: 1,
            prefill_token_ids: None,
            prefill_vision_embeddings: None,
            decode_token_ids: None,
            lm_head_sampling: RealFullLmHeadSamplingOptions::diagnostic(),
        };
        let policy = PrefillChunkPolicy::latency_smoke(2);
        let chunks = plan_scheduler_prefill_chunks(&shape, LayerId(7), 99, Priority(0), &policy);

        assert_eq!(shape.prefill_token_start(), 128);
        assert_eq!(shape.decode_token_start(), 131);
        assert_eq!(shape.mtp_token_start(), 133);
        assert_eq!(shape.reservation_tokens(), 134);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].token_start.0, 128);
        assert_eq!(chunks[0].token_count, 2);
        assert_eq!(chunks[1].token_start.0, 130);
        assert_eq!(chunks[1].token_count, 1);
    }

    #[test]
    fn scheduler_prefill_chunks_cover_64_token_live_target() {
        let shape = RealFullSchedulerExecutionShape {
            request_id: "request-shaped-prefill-64".to_owned(),
            sequence_id: "request-shaped-prefill-64-sequence".to_owned(),
            placement_version: "request-shaped-prefill-64-version".to_owned(),
            prefix_tokens: 0,
            prefill_tokens: 130,
            prefill_chunk_tokens: 64,
            decode_rows: 1,
            mtp_rows: 0,
            mtp_accepted_rows: 0,
            prefill_token_ids: None,
            prefill_vision_embeddings: None,
            decode_token_ids: None,
            lm_head_sampling: RealFullLmHeadSamplingOptions::diagnostic(),
        };
        let policy = PrefillChunkPolicy::latency_smoke(64);
        let chunks = plan_scheduler_prefill_chunks(&shape, LayerId(7), 99, Priority(0), &policy);

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].token_start.0, 0);
        assert_eq!(chunks[0].token_count, 64);
        assert_eq!(chunks[1].token_start.0, 64);
        assert_eq!(chunks[1].token_count, 64);
        assert_eq!(chunks[2].token_start.0, 128);
        assert_eq!(chunks[2].token_count, 2);
    }

    #[test]
    fn scheduler_expected_device_attention_rows_include_persistent_prefix() {
        let shape = RealFullSchedulerExecutionShape {
            request_id: "request-shaped-prefix-decode".to_owned(),
            sequence_id: "request-shaped-prefix-decode-sequence".to_owned(),
            placement_version: "request-shaped-prefix-decode-version".to_owned(),
            prefix_tokens: 4,
            prefill_tokens: 0,
            prefill_chunk_tokens: 512,
            decode_rows: 1,
            mtp_rows: 1,
            mtp_accepted_rows: 1,
            prefill_token_ids: None,
            prefill_vision_embeddings: None,
            decode_token_ids: None,
            lm_head_sampling: RealFullLmHeadSamplingOptions::diagnostic(),
        };

        assert_eq!(
            scheduler_expected_device_attention_rows(&shape),
            GLM52_NUM_HIDDEN_LAYERS * 11
        );
    }

    #[test]
    fn scheduler_expected_device_attention_rows_allow_sequential_mtp_queries() {
        let shape = RealFullSchedulerExecutionShape {
            request_id: "request-shaped-prefix-sequential-mtp".to_owned(),
            sequence_id: "request-shaped-prefix-sequential-mtp-sequence".to_owned(),
            placement_version: "request-shaped-prefix-sequential-mtp-version".to_owned(),
            prefix_tokens: 4,
            prefill_tokens: 0,
            prefill_chunk_tokens: 512,
            decode_rows: 1,
            mtp_rows: 2,
            mtp_accepted_rows: 2,
            prefill_token_ids: None,
            prefill_vision_embeddings: None,
            decode_token_ids: None,
            lm_head_sampling: RealFullLmHeadSamplingOptions::diagnostic(),
        };

        assert_eq!(
            scheduler_expected_device_attention_rows(&shape),
            GLM52_NUM_HIDDEN_LAYERS * 18
        );
    }

    #[test]
    fn scheduler_expected_device_attention_rows_include_decode_only_prefix() {
        let shape = RealFullSchedulerExecutionShape {
            request_id: "request-shaped-prefix-decode-only".to_owned(),
            sequence_id: "request-shaped-prefix-decode-only-sequence".to_owned(),
            placement_version: "request-shaped-prefix-decode-only-version".to_owned(),
            prefix_tokens: 4,
            prefill_tokens: 0,
            prefill_chunk_tokens: 512,
            decode_rows: 1,
            mtp_rows: 0,
            mtp_accepted_rows: 0,
            prefill_token_ids: None,
            prefill_vision_embeddings: None,
            decode_token_ids: None,
            lm_head_sampling: RealFullLmHeadSamplingOptions::diagnostic(),
        };

        assert_eq!(
            scheduler_expected_device_attention_rows(&shape),
            GLM52_NUM_HIDDEN_LAYERS * 5
        );
    }

    #[test]
    fn scheduler_sparse_tcp_execution_returns_residual_probe_when_cuda_disabled(
    ) -> anyhow::Result<()> {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);
        let shape = RealFullSchedulerExecutionShape {
            request_id: "scheduler-sparse-tcp-no-cuda".to_owned(),
            sequence_id: "scheduler-sparse-tcp-no-cuda-sequence".to_owned(),
            placement_version: "scheduler-sparse-tcp-no-cuda-placement".to_owned(),
            prefix_tokens: 0,
            prefill_tokens: 1,
            prefill_chunk_tokens: 1,
            decode_rows: 2,
            mtp_rows: 0,
            mtp_accepted_rows: 0,
            prefill_token_ids: None,
            prefill_vision_embeddings: None,
            decode_token_ids: None,
            lm_head_sampling: RealFullLmHeadSamplingOptions::diagnostic(),
        };
        let catalog = tiny_lm_head_catalog(4, 2, 0);
        let target = TcpProtocolV2HostBatchTarget {
            host: "ostrich".to_owned(),
            addr: "127.0.0.1:1".parse()?,
        };

        let (report, probe) = real_full_scheduler_execution_for_shape_with_sparse_tcp(
            KvCacheConfig::glm52_phase0(shape.reservation_tokens()),
            &catalog,
            shape,
            vec![target],
            None,
            90_000,
        )?;

        assert_eq!(
            probe.status,
            "request-shaped-sparse-tcp-residual-dispatch-blocked"
        );
        assert!(!probe.passed);
        assert_eq!(probe.sparse_batches, 0);
        assert_eq!(probe.scheduler_iterations_per_sparse_layer, 1);
        assert_eq!(
            report.sparse_expert_batches,
            GLM52_NUM_HIDDEN_LAYERS - GLM52_FIRST_K_DENSE_REPLACE
        );
        assert!(report.numeric_progression_self_test.passed);
        assert_eq!(
            report.numeric_progression_self_test.selected_decode_rows,
            GLM52_NUM_HIDDEN_LAYERS * 2
        );
        assert_eq!(
            report
                .numeric_progression_self_test
                .device_real_sparse_routed_mlp_delta_status,
            "not-run"
        );
        Ok(())
    }

    #[test]
    fn scheduler_terminal_lm_head_sample_uses_final_decode_device_hidden_when_cuda_available(
    ) -> anyhow::Result<()> {
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);

        let result = (|| -> anyhow::Result<()> {
            if cuda_native_library().is_err() {
                return Ok(());
            }
            let final_norm = bf16_bytes(&[1.0_f32, 0.1]);
            match preload_resident_weight_from_host_staging(
                REAL_FULL_FINAL_NORM_WEIGHT_NAME,
                final_norm.len(),
                "test scheduler terminal final_norm weight",
                |staging| {
                    staging.copy_from_slice(&final_norm);
                    Ok(())
                },
            ) {
                Ok(()) => {}
                Err(error)
                    if error
                        .to_string()
                        .contains("CUDA native allocation unavailable") =>
                {
                    return Ok(());
                }
                Err(error) => return Err(error),
            }
            let lm_head = bf16_bytes(&[
                0.0_f32, 1.0, //
                1.0, 0.0, //
            ]);
            match preload_resident_weight_from_host_staging(
                "lm_head.weight",
                lm_head.len(),
                "test scheduler terminal lm_head sample weight",
                |staging| {
                    staging.copy_from_slice(&lm_head);
                    Ok(())
                },
            ) {
                Ok(()) => {}
                Err(error)
                    if error
                        .to_string()
                        .contains("CUDA native allocation unavailable") =>
                {
                    return Ok(());
                }
                Err(error) => return Err(error),
            }
            let hidden = match device_bf16_output_from_f32_values(
                &[3.0_f32, 4.0],
                1,
                2,
                "test scheduler terminal lm_head hidden",
            ) {
                Ok(hidden) => hidden,
                Err(error)
                    if error
                        .to_string()
                        .contains("CUDA native allocation unavailable") =>
                {
                    return Ok(());
                }
                Err(error) => return Err(error),
            };
            let catalog = tiny_lm_head_catalog(2, 2, lm_head.len());

            let sample = scheduler_terminal_lm_head_sample(&catalog, Some(&hidden));

            assert_eq!(sample.status, "sampled");
            assert!(sample.uses_final_decode_device_hidden);
            assert!(sample.passed);
            assert!(sample.covers_full_vocabulary);
            assert_eq!(sample.hidden_dim, 2);
            assert_eq!(sample.vocab_size, 2);
            assert_eq!(sample.logits_evaluated, 2);
            assert_eq!(sample.top_token_id, Some(1));
            assert!(sample.sampled_token_id.is_some());
            assert_eq!(sample.sample_top_k, Some(2));
            assert_eq!(sample.sample_top_p, Some(0.95));
            assert_eq!(
                sample.argmax_kernel_backend,
                Some(CUDA_REFERENCE_LM_HEAD_ARGMAX_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND)
            );
            let expected_sampler_backend =
                CUDA_REFERENCE_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND;
            assert_eq!(
                sample.sampler_kernel_backend,
                Some(expected_sampler_backend)
            );
            assert!(sample.blocker.is_none());
            Ok(())
        })();

        result
    }

    #[test]
    fn scheduler_terminal_lm_head_sample_reports_missing_final_device_hidden() {
        let catalog = tiny_lm_head_catalog(4, 2, 16);

        let sample = scheduler_terminal_lm_head_sample(&catalog, None);

        assert_eq!(sample.status, "not-run");
        assert!(!sample.uses_final_decode_device_hidden);
        assert!(!sample.passed);
        assert!(sample.blocker.is_some());
        assert_eq!(
            scheduler_execution_status_for_completion(&sample, false),
            "admitted-scheduler-dry-run"
        );
    }

    #[test]
    fn scheduler_defers_only_retained_speculative_target_sampling() {
        assert!(scheduler_should_defer_target_lm_head_sample(true, 1));
        assert!(scheduler_should_defer_target_lm_head_sample(true, 15));
        assert!(!scheduler_should_defer_target_lm_head_sample(true, 0));
        assert!(!scheduler_should_defer_target_lm_head_sample(false, 15));
    }

    #[test]
    fn scheduler_terminal_lm_head_completion_gate_rejects_greedy_or_non_cuda_sampling() {
        let mut sample = terminal_lm_head_score_for_completion_gate_test();
        assert!(scheduler_terminal_lm_head_completion_gate(&sample));

        sample.sample_top_k = 1;
        assert!(!scheduler_terminal_lm_head_completion_gate(&sample));

        sample = terminal_lm_head_score_for_completion_gate_test();
        sample.sampler_kernel_backend =
            TRITON_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND;
        assert!(scheduler_terminal_lm_head_completion_gate(&sample));

        sample = terminal_lm_head_score_for_completion_gate_test();
        sample.sampler_kernel_backend = "cpu-reference-lm-head-sample-topk-topp-bf16";
        assert!(!scheduler_terminal_lm_head_completion_gate(&sample));
    }

    #[test]
    fn scheduler_execution_status_tracks_terminal_lm_head_sample_gate() {
        let passed = RealFullSchedulerTerminalLmHeadSample {
            status: "sampled",
            scope: "test",
            uses_final_decode_device_hidden: true,
            covers_full_vocabulary: true,
            hidden_dim: 2,
            vocab_size: 4,
            logits_evaluated: 4,
            top_token_id: Some(2),
            sampled_token_id: Some(1),
            sample_top_k: Some(4),
            sample_top_p: Some(0.95),
            argmax_kernel_backend: Some(
                CUDA_REFERENCE_LM_HEAD_ARGMAX_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
            ),
            sampler_kernel_backend: Some(
                CUDA_REFERENCE_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
            ),
            passed: true,
            blocker: None,
        };
        assert_eq!(
            scheduler_execution_status_for_completion(&passed, true),
            "admitted-scheduler-terminal-lm-head-sampled"
        );
        assert_eq!(
            scheduler_execution_status_for_completion(&passed, false),
            "admitted-scheduler-terminal-lm-head-blocked"
        );

        let mut blocked = passed;
        blocked.status = "blocked";
        blocked.passed = false;
        blocked.blocker = Some("full resident lm_head missing".to_owned());
        assert_eq!(
            scheduler_execution_status_for_completion(&blocked, true),
            "admitted-scheduler-terminal-lm-head-blocked"
        );

        let not_run = RealFullSchedulerTerminalLmHeadSample {
            uses_final_decode_device_hidden: false,
            ..blocked
        };
        assert_eq!(
            scheduler_execution_status_for_completion(&not_run, true),
            "admitted-scheduler-dry-run"
        );
    }

    #[test]
    fn scheduler_full_context_device_attention_gate_requires_hidden_projection_launches() {
        let shape = RealFullSchedulerExecutionShape::preflight();
        let expected_launches = GLM52_NUM_HIDDEN_LAYERS * 4;
        let expected_query_rows = GLM52_NUM_HIDDEN_LAYERS
            * (REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START as usize
                + REAL_FULL_PREFLIGHT_PREFILL_ROWS
                + REAL_FULL_PREFLIGHT_DECODE_ROWS
                + REAL_FULL_PREFLIGHT_MTP_ROWS);
        let mut counters = RealFullSchedulerExecutionCounters {
            device_attention_launches: expected_launches,
            device_attention_hidden_projection_launches: expected_launches,
            device_attention_rows: scheduler_expected_device_attention_rows(&shape),
            device_attention_query_rows: expected_query_rows,
            device_attention_kv_descriptors: expected_launches,
            device_attention_output_values: expected_query_rows * GLM52_HIDDEN_SIZE,
            device_attention_output_finite_values: expected_query_rows * GLM52_HIDDEN_SIZE,
            device_attention_output_nonzero_values: 1,
            device_attention_output_checksum: 1.0,
            layer_order_verified: true,
            ..Default::default()
        };

        assert!(scheduler_full_context_device_attention_complete(
            &shape, &counters, true, true
        ));
        counters.device_attention_output_finite_values = 0;
        counters.device_attention_output_nonzero_values = 0;
        counters.device_attention_output_checksum = 0.0;
        assert!(scheduler_full_context_device_attention_complete(
            &shape, &counters, true, true
        ));

        counters.device_attention_hidden_projection_launches = expected_launches - 1;
        assert!(!scheduler_full_context_device_attention_complete(
            &shape, &counters, true, true
        ));
        counters.device_attention_hidden_projection_launches = expected_launches;
        counters.device_attention_kv_descriptors = expected_launches - 1;
        assert!(!scheduler_full_context_device_attention_complete(
            &shape, &counters, true, true
        ));
        counters.device_attention_kv_descriptors =
            scheduler_expected_device_attention_rows(&shape) + 1;
        assert!(!scheduler_full_context_device_attention_complete(
            &shape, &counters, true, true
        ));
    }

    #[test]
    fn scheduler_full_context_device_attention_gate_accepts_compact_suffix_rows() {
        let shape = RealFullSchedulerExecutionShape {
            request_id: "request-shaped-compact-suffix".to_owned(),
            sequence_id: "request-shaped-compact-suffix-sequence".to_owned(),
            placement_version: "request-shaped-compact-suffix-version".to_owned(),
            prefix_tokens: 0,
            prefill_tokens: 9,
            prefill_chunk_tokens: 1024,
            decode_rows: 1,
            mtp_rows: 0,
            mtp_accepted_rows: 0,
            prefill_token_ids: None,
            prefill_vision_embeddings: None,
            decode_token_ids: None,
            lm_head_sampling: RealFullLmHeadSamplingOptions::diagnostic(),
        };
        let expected_launches = GLM52_NUM_HIDDEN_LAYERS * 2;
        let expected_query_rows = GLM52_NUM_HIDDEN_LAYERS * 10;
        let expected_context_rows = GLM52_NUM_HIDDEN_LAYERS * 19;
        let mut counters = RealFullSchedulerExecutionCounters {
            device_attention_launches: expected_launches,
            device_attention_hidden_projection_launches: expected_launches,
            device_attention_rows: expected_query_rows,
            device_attention_query_rows: expected_query_rows,
            device_attention_kv_descriptors: expected_launches,
            device_attention_output_values: expected_query_rows * GLM52_HIDDEN_SIZE,
            device_attention_output_finite_values: expected_query_rows * GLM52_HIDDEN_SIZE,
            device_attention_output_nonzero_values: 1,
            device_attention_output_checksum: 1.0,
            layer_order_verified: true,
            ..Default::default()
        };

        assert_eq!(
            scheduler_expected_device_attention_rows(&shape),
            expected_context_rows
        );
        assert!(scheduler_full_context_device_attention_complete(
            &shape, &counters, true, true
        ));

        counters.device_attention_rows = expected_query_rows - 1;
        assert!(!scheduler_full_context_device_attention_complete(
            &shape, &counters, true, true
        ));
    }

    #[test]
    fn scheduler_full_context_device_attention_gate_accepts_token_granular_decode_prefix() {
        let shape = RealFullSchedulerExecutionShape {
            request_id: "request-shaped-token-granular-prefix".to_owned(),
            sequence_id: "request-shaped-token-granular-prefix-sequence".to_owned(),
            placement_version: "request-shaped-token-granular-prefix-version".to_owned(),
            prefix_tokens: 18,
            prefill_tokens: 0,
            prefill_chunk_tokens: 512,
            decode_rows: 1,
            mtp_rows: 0,
            mtp_accepted_rows: 0,
            prefill_token_ids: None,
            prefill_vision_embeddings: None,
            decode_token_ids: None,
            lm_head_sampling: RealFullLmHeadSamplingOptions::diagnostic(),
        };
        let expected_launches = GLM52_NUM_HIDDEN_LAYERS;
        let expected_query_rows = GLM52_NUM_HIDDEN_LAYERS;
        let expected_rows = GLM52_NUM_HIDDEN_LAYERS * 19;
        let counters = RealFullSchedulerExecutionCounters {
            device_attention_launches: expected_launches,
            device_attention_hidden_projection_launches: expected_launches,
            device_attention_rows: expected_rows,
            device_attention_query_rows: expected_query_rows,
            device_attention_kv_descriptors: expected_rows,
            device_attention_output_values: expected_query_rows * GLM52_HIDDEN_SIZE,
            device_attention_output_finite_values: expected_query_rows * GLM52_HIDDEN_SIZE,
            device_attention_output_nonzero_values: 1,
            device_attention_output_checksum: 1.0,
            layer_order_verified: true,
            ..Default::default()
        };

        assert_eq!(
            scheduler_expected_device_attention_rows(&shape),
            expected_rows
        );
        assert!(scheduler_full_context_device_attention_complete(
            &shape, &counters, true, true
        ));
    }

    #[test]
    fn bounded_long_prefill_probe_counts_physical_chunk_dispatches() {
        let shape = RealFullSchedulerExecutionShape {
            request_id: "bounded-long-prefill-probe".to_owned(),
            sequence_id: "bounded-long-prefill-probe-sequence".to_owned(),
            placement_version: "bounded-long-prefill-probe-version".to_owned(),
            prefix_tokens: 0,
            prefill_tokens: 132_031,
            prefill_chunk_tokens: 2_032,
            decode_rows: 1,
            mtp_rows: 0,
            mtp_accepted_rows: 0,
            prefill_token_ids: None,
            prefill_vision_embeddings: None,
            decode_token_ids: None,
            lm_head_sampling: RealFullLmHeadSamplingOptions::diagnostic(),
        };

        assert!(bounded_long_prefill_wavefront_required(
            shape.prefill_tokens + shape.decode_rows
        ));
        assert_eq!(
            scheduler_sparse_tcp_iterations_per_sparse_layer(&shape),
            shape.prefill_tokens.div_ceil(shape.prefill_chunk_tokens)
        );
    }

    fn terminal_lm_head_score_for_completion_gate_test() -> RealLmHeadChunkScoreForHidden {
        RealLmHeadChunkScoreForHidden {
            lm_head_tensor: "lm_head.weight".to_owned(),
            hidden_dim: 2,
            vocab_size: 4,
            start_token_id: 0,
            chunk_rows: 4,
            rows_scored: 4,
            chunks_scored: 1,
            lm_head_bytes_read: 0,
            hidden_values: 2,
            logits_evaluated: 4,
            multiply_accumulate_ops: 8,
            covers_full_vocabulary: true,
            logits_kernel_backend:
                CUDA_REFERENCE_LM_HEAD_ARGMAX_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
            argmax_kernel_backend:
                CUDA_REFERENCE_LM_HEAD_ARGMAX_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
            sampler_kernel_backend:
                CUDA_REFERENCE_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
            top_token_id: 2,
            top_logit: 3.0,
            sampled_token_id: 1,
            sampled_score: 0.25,
            sample_random_uniform: 0.5,
            sample_temperature: 0.7,
            sample_top_k: 4,
            sample_top_p: 0.95,
        }
    }
}
