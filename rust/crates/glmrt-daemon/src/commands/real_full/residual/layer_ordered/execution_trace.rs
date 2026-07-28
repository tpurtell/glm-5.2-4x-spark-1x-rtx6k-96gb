use anyhow::{Context, Result};
use glmrt_core::{
    DType, ExpertBatch, ExpertBatchRoute, ExpertBatchRow, ExpertGraphInstancePool,
    ExpertHostBatchSet, ExpertOwnerLookup, GraphBucket, KvBackedBlock, KvBlockDescriptor,
    KvCacheBackingStore, KvCacheConfig, LayerId, LayerWaveMode, ModelFacts, PlacementVersion,
    PositionId, RequestId, RowSourceKind, TensorCatalog, TensorInfo, EXPERT_HOSTS,
    GLM52_DSA_INDEXER_LAYER_IDS, GLM52_DSA_INDEX_HEAD_DIM, GLM52_FIRST_K_DENSE_REPLACE,
    GLM52_HIDDEN_BF16_BYTES, GLM52_HIDDEN_SIZE, GLM52_NUM_HIDDEN_LAYERS, GLM52_TOP_K,
};
use std::borrow::Cow;
use std::collections::BTreeMap;

#[cfg(test)]
use super::super::super::attention::real_full_attention_residual_full_output_hidden;
use super::super::super::attention::{
    real_full_attention_residual_full_output_hidden_for_layer_from_initial,
    real_full_attention_residual_prefix_hidden_for_layer_from_initial,
    real_full_mla_rope_attention_full_output_hidden_for_layer_from_initial,
    real_full_mla_rope_attention_full_output_kv_cache_context_hidden_for_layer_from_initial,
    real_full_mla_rope_attention_full_output_kv_cache_context_hidden_for_layer_from_initial_device_input,
    real_full_mla_rope_attention_full_output_prefix_context_hidden_for_layer_from_initial,
    real_full_mla_rope_attention_kv_cache_context_hidden_for_layer_from_initial,
    real_full_mla_rope_attention_kv_cache_context_hidden_for_layer_from_initial_device_input,
    real_full_mla_rope_attention_prefix_context_hidden_for_layer_from_initial,
    real_full_mla_rope_attention_prefix_hidden_for_layer_from_initial,
    real_full_mla_rope_kv_cache_block_for_layer_from_hidden, RealFullAttentionResidualPrefixHidden,
    RealFullMlaRopeKvCacheBlock,
};
use super::super::super::coordinator_kernels::{
    coordinator_cuda_graph_stats, gather_rows_bf16, scatter_add_rows_bf16_to_f32,
    sparse_b_scatter_residual_add_bf16, CoordinatorCudaGraphStats, DeviceBf16Output,
    CPU_REFERENCE_CAUSAL_ATTENTION_BF16_BACKEND, CPU_REFERENCE_LINEAR_BF16_BACKEND,
    CPU_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND, CPU_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
    CPU_REFERENCE_RMSNORM_BF16_BACKEND, CPU_REFERENCE_ROUTER_TOPK_BF16_BACKEND,
    CPU_REFERENCE_SILU_GATED_MLP_BF16_BACKEND, CUDA_REFERENCE_CAUSAL_ATTENTION_BF16_BACKEND,
    CUDA_REFERENCE_LINEAR_BF16_BACKEND,
    CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
    CUDA_REFERENCE_LINEAR_BF16_RESIDENT_WEIGHT_BACKEND,
    CUDA_REFERENCE_LM_HEAD_ARGMAX_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
    CUDA_REFERENCE_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
    CUDA_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND, CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
    CUDA_REFERENCE_RMSNORM_BF16_BACKEND,
    CUDA_REFERENCE_RMSNORM_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
    CUDA_REFERENCE_RMSNORM_BF16_RESIDENT_WEIGHT_BACKEND, CUDA_REFERENCE_ROUTER_TOPK_BF16_BACKEND,
    CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
    CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_BACKEND,
    CUDA_REFERENCE_ROUTER_TOPK_BF16_RESIDENT_WEIGHT_BACKEND,
    CUDA_REFERENCE_SILU_GATED_MLP_BF16_BACKEND,
    CUDA_REFERENCE_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND,
    CUDA_REFERENCE_SILU_GATED_MLP_BF16_RESIDENT_WEIGHT_BACKEND,
    TRITON_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
};
use super::super::super::dense::math::{checksum_f64, deterministic_dense_hidden};
use super::super::super::dense::{
    real_full_dense_layer_full_output_hidden_from_initial,
    real_full_dense_layer_full_output_hidden_from_initial_device_input,
    real_full_dense_layer_prefix_hidden_from_initial, RealFullDenseLayerPrefixHidden,
};
use super::super::super::embedding::real_full_embedding_hidden_for_token;
use super::super::super::sampling::{
    score_real_lm_head_chunk_for_hidden, score_real_lm_head_full_vocab_for_device_hidden,
    score_real_lm_head_full_vocab_for_hidden, RealLmHeadChunkScoreForHidden,
};
use super::super::super::sparse_mlp::{
    real_sparse_mlp_shared_layer_full_output_hidden_from_initial,
    real_sparse_mlp_shared_layer_full_output_hidden_from_initial_device_input,
    real_sparse_mlp_shared_layer_hidden_from_initial, RealFullSparseMlpSharedLayerHidden,
};
use super::super::super::types::{
    RealFullLayerOrderedLmHeadSamplingProbe, RealFullLayerOrderedResidualExecutionProbe,
    RealFullLayerOrderedResidualExecutionStep, RealFullResidualCompletionGates,
    RealFullResidualExecutionStepper, RealFullResidualExecutionTensorArtifact,
    RealFullSparseMoeHostBatchSetEvidence, REAL_FULL_RESIDUAL_COMPLETION_BLOCKER,
};
use super::execution_stepper::{
    RealExecutionStepper, RealExecutionStepperFinish, RealExecutionStepperOutput,
};
use super::oracle_fixture::bounded_attention_oracle_stepper_evidence;
use super::scheduler_rows::{
    layer_ordered_scheduler_rows_binding, LayerOrderedSchedulerRowsBinding,
};
use glmrt_transport::{
    expert_protocol_v2_compact_id, tcp_protocol_v2_host_batch_set_bf16_payload_dispatch,
    tcp_protocol_v2_host_batch_set_bf16_payload_dispatch_with_graph_pool, ExpertProtocolV2Request,
    ExpertProtocolV2RequestView, ExpertProtocolV2Response, ExpertProtocolV2Status, ExpertV2Dtype,
    ProtocolV2ExpertExecutor, SyntheticRouteExecutor, TcpProtocolV2HostBatchSetBf16PayloadDispatch,
    TcpProtocolV2HostBatchSetDispatchStats, TcpProtocolV2HostBatchTarget, TcpTransportConfig,
    PROTOCOL_V2_SYNTHETIC_ROUTE_EXECUTOR,
};

const REAL_FULL_LAYER_ORDERED_EXECUTION_TRACE_ENV: &str =
    "GLMRT_REAL_FULL_LAYER_ORDERED_EXECUTION_TRACE";
const REAL_FULL_LAYER_ORDERED_INPUT_ENV: &str = "GLMRT_REAL_FULL_LAYER_ORDERED_INPUT";
const REAL_FULL_LAYER_ORDERED_INPUT_TOKEN_ID_ENV: &str =
    "GLMRT_REAL_FULL_LAYER_ORDERED_INPUT_TOKEN_ID";
const REAL_FULL_LAYER_ORDERED_LM_HEAD_FULL_VOCAB_ENV: &str =
    "GLMRT_REAL_FULL_LAYER_ORDERED_LM_HEAD_FULL_VOCAB";
const REAL_FULL_LAYER_ORDERED_EXPERT_TCP_TARGETS_ENV: &str =
    "GLMRT_REAL_FULL_LAYER_ORDERED_EXPERT_TCP_TARGETS";
const REAL_FULL_LAYER_ORDERED_DEFAULT_TOKEN_ID: usize = 0;
const REAL_FULL_LAYER_ORDERED_LM_HEAD_CHUNK_ROWS: usize = 1024;
const PROTOCOL_V2_REAL_NVFP4_CHECKPOINT_EXECUTOR: &str =
    "protocol-v2-real-nvfp4-checkpoint-executor";
const PROTOCOL_V2_COMPACT_HIDDEN_GATHER_BACKEND: &str = "expert-host-batch-compact-hidden-payload";
const PROTOCOL_V2_RECONSTRUCT_ACCUMULATE_BACKEND: &str =
    "expert-host-batch-reconstruct-accumulate-f32";

#[derive(Clone, Copy)]
struct LayerOrderedExecutionMode {
    row_mode: &'static str,
    attention_backend: LayerOrderedAttentionBackend,
    attention_full_output: bool,
    dense_full_output: bool,
    sparse_full_output: bool,
    lm_head_full_vocab: bool,
    initial_input: LayerOrderedInitialInputMode,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LayerOrderedAttentionBackend {
    CausalPrefix,
    MlaRopePrefix,
}

#[derive(Clone, Copy)]
struct LayerOrderedInitialInputMode {
    token_id: Option<usize>,
    uses_embedding_residual_input: bool,
}

struct InitialResidualInput {
    token_id: Option<usize>,
    embedding_bytes_read: u64,
    embedding_residual_checksum: Option<f64>,
    embedding_kernel_backend: Option<&'static str>,
    embedding_device_resident: bool,
    uses_embedding_residual_input: bool,
}

#[derive(Default)]
struct MlaDsaAttentionCompletionTracker {
    attention_layers: usize,
    kv_cache_mla_layers: usize,
    dsa_indexer_layers: usize,
}

#[derive(Default)]
struct CoordinatorGraphReplayDelta {
    slots: usize,
    captured_graphs: usize,
    graph_captures: usize,
    graph_launches: usize,
}

fn coordinator_graph_replay_delta(
    before: Option<CoordinatorCudaGraphStats>,
) -> CoordinatorGraphReplayDelta {
    let Some(before) = before else {
        return CoordinatorGraphReplayDelta::default();
    };
    let Ok(after) = coordinator_cuda_graph_stats() else {
        return CoordinatorGraphReplayDelta::default();
    };
    CoordinatorGraphReplayDelta {
        slots: after.slots,
        captured_graphs: after.captured_graphs,
        graph_captures: after.graph_captures.saturating_sub(before.graph_captures),
        graph_launches: after.graph_launches.saturating_sub(before.graph_launches),
    }
}

impl MlaDsaAttentionCompletionTracker {
    fn record(&mut self, attention: &RealFullAttentionResidualPrefixHidden) {
        self.attention_layers += 1;
        if attention_uses_kv_cache_mla_context(attention) {
            self.kv_cache_mla_layers += 1;
        }
        if attention_covers_dsa_indexer(attention) {
            self.dsa_indexer_layers += 1;
        }
    }

    fn uses_full_context_mla_dsa_attention(&self) -> bool {
        self.attention_layers == GLM52_NUM_HIDDEN_LAYERS
            && self.kv_cache_mla_layers == GLM52_NUM_HIDDEN_LAYERS
            && self.dsa_indexer_layers == GLM52_DSA_INDEXER_LAYER_IDS.len()
    }
}

fn attention_uses_kv_cache_mla_context(attention: &RealFullAttentionResidualPrefixHidden) -> bool {
    attention.includes_mla_softmax
        && attention.uses_kv_cache_context
        && attention.kv_cache_context_bytes > 0
        && attention.prefix_context_rows > 0
        && attention.total_context_rows > attention.attention_rows
        && matches!(
            attention.attention_backend,
            CPU_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND
                | CUDA_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND
        )
}

fn attention_covers_dsa_indexer(attention: &RealFullAttentionResidualPrefixHidden) -> bool {
    GLM52_DSA_INDEXER_LAYER_IDS.contains(&attention.layer_id)
        && attention.includes_dsa_candidate_selection
        && attention.includes_dsa_softmax
        && attention.dsa_candidate_rows > 0
}

pub(in crate::commands::real_full::residual) fn real_full_layer_ordered_execution_probe(
    catalog: &TensorCatalog,
) -> RealFullLayerOrderedResidualExecutionProbe {
    let trace_env = std::env::var(REAL_FULL_LAYER_ORDERED_EXECUTION_TRACE_ENV).ok();
    let input_env = std::env::var(REAL_FULL_LAYER_ORDERED_INPUT_ENV).ok();
    let input_token_env = std::env::var(REAL_FULL_LAYER_ORDERED_INPUT_TOKEN_ID_ENV).ok();
    let mode = layer_ordered_execution_mode_with_input(
        trace_env.as_deref(),
        input_env.as_deref(),
        input_token_env.as_deref(),
    );
    match run_real_full_layer_ordered_execution_probe_with_mode(catalog, mode) {
        Ok(probe) => probe,
        Err(error) => {
            skipped_real_full_layer_ordered_execution_probe("error", Some(error.to_string()))
        }
    }
}

#[cfg(test)]
fn layer_ordered_execution_mode(env_setting: Option<&str>) -> LayerOrderedExecutionMode {
    layer_ordered_execution_mode_with_input(env_setting, None, None)
}

fn layer_ordered_execution_mode_with_input(
    env_setting: Option<&str>,
    input_env_setting: Option<&str>,
    input_token_env_setting: Option<&str>,
) -> LayerOrderedExecutionMode {
    let initial_input =
        layer_ordered_initial_input_mode(input_env_setting, input_token_env_setting);
    let normalized = env_setting.map(|value| value.trim().to_ascii_lowercase());
    match normalized.as_deref() {
        Some(
            "full-output-attention-mlp-full-vocab"
            | "full-output-all-full-vocab"
            | "all-full-output-full-vocab"
            | "full-output-residual-full-vocab"
            | "full-vocab",
        ) => LayerOrderedExecutionMode {
            row_mode: "full-output-attention-mlp",
            attention_backend: LayerOrderedAttentionBackend::CausalPrefix,
            attention_full_output: true,
            dense_full_output: true,
            sparse_full_output: true,
            lm_head_full_vocab: true,
            initial_input,
        },
        Some(
            "full-output-attention-mlp"
            | "full-output-all"
            | "all-full-output"
            | "full-output-residual",
        ) => LayerOrderedExecutionMode {
            row_mode: "full-output-attention-mlp",
            attention_backend: LayerOrderedAttentionBackend::CausalPrefix,
            attention_full_output: true,
            dense_full_output: true,
            sparse_full_output: true,
            lm_head_full_vocab: false,
            initial_input,
        },
        Some("full-output-mlp" | "mlp-full-output" | "full-output") => LayerOrderedExecutionMode {
            row_mode: "full-output-mlp",
            attention_backend: LayerOrderedAttentionBackend::CausalPrefix,
            attention_full_output: false,
            dense_full_output: true,
            sparse_full_output: true,
            lm_head_full_vocab: false,
            initial_input,
        },
        Some(
            "full-output-mla-rope-attention-mlp-full-vocab"
            | "full-output-mla-rope-all-full-vocab"
            | "mla-rope-full-output-all-full-vocab",
        ) => LayerOrderedExecutionMode {
            row_mode: "full-output-mla-rope-attention-mlp",
            attention_backend: LayerOrderedAttentionBackend::MlaRopePrefix,
            attention_full_output: true,
            dense_full_output: true,
            sparse_full_output: true,
            lm_head_full_vocab: true,
            initial_input,
        },
        Some(
            "full-output-mla-rope-attention-mlp"
            | "full-output-mla-rope-all"
            | "mla-rope-full-output-all",
        ) => LayerOrderedExecutionMode {
            row_mode: "full-output-mla-rope-attention-mlp",
            attention_backend: LayerOrderedAttentionBackend::MlaRopePrefix,
            attention_full_output: true,
            dense_full_output: true,
            sparse_full_output: true,
            lm_head_full_vocab: false,
            initial_input,
        },
        Some("mla-rope" | "bounded-mla-rope" | "mla-rope-attention") => LayerOrderedExecutionMode {
            row_mode: "mla-rope-attention",
            attention_backend: LayerOrderedAttentionBackend::MlaRopePrefix,
            attention_full_output: false,
            dense_full_output: false,
            sparse_full_output: false,
            lm_head_full_vocab: false,
            initial_input,
        },
        Some(
            "mla-rope-full-output" | "full-output-mla-rope" | "full-output-mla-rope-attention",
        ) => LayerOrderedExecutionMode {
            row_mode: "full-output-mla-rope-attention",
            attention_backend: LayerOrderedAttentionBackend::MlaRopePrefix,
            attention_full_output: true,
            dense_full_output: false,
            sparse_full_output: false,
            lm_head_full_vocab: false,
            initial_input,
        },
        Some("1" | "bounded" | "default") | None => LayerOrderedExecutionMode {
            row_mode: "bounded",
            attention_backend: LayerOrderedAttentionBackend::CausalPrefix,
            attention_full_output: false,
            dense_full_output: false,
            sparse_full_output: false,
            lm_head_full_vocab: false,
            initial_input,
        },
        _ => LayerOrderedExecutionMode {
            row_mode: "bounded",
            attention_backend: LayerOrderedAttentionBackend::CausalPrefix,
            attention_full_output: false,
            dense_full_output: false,
            sparse_full_output: false,
            lm_head_full_vocab: false,
            initial_input,
        },
    }
}

fn layer_ordered_initial_input_mode(
    env_setting: Option<&str>,
    token_env_setting: Option<&str>,
) -> LayerOrderedInitialInputMode {
    let token_id = token_env_setting
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(REAL_FULL_LAYER_ORDERED_DEFAULT_TOKEN_ID);
    let normalized = env_setting.map(|value| value.trim().to_ascii_lowercase());
    match normalized.as_deref() {
        Some("deterministic" | "synthetic" | "probe" | "old") => LayerOrderedInitialInputMode {
            token_id: None,
            uses_embedding_residual_input: false,
        },
        _ => LayerOrderedInitialInputMode {
            token_id: Some(token_id),
            uses_embedding_residual_input: true,
        },
    }
}

fn layer_ordered_expert_tcp_targets_from_env() -> Result<Option<Vec<TcpProtocolV2HostBatchTarget>>>
{
    let Ok(raw_targets) = std::env::var(REAL_FULL_LAYER_ORDERED_EXPERT_TCP_TARGETS_ENV) else {
        return Ok(None);
    };
    let raw_targets = raw_targets.trim();
    if raw_targets.is_empty() || raw_targets == "0" {
        return Ok(None);
    }

    let mut targets = Vec::new();
    for entry in raw_targets.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let (host, addr) = entry.split_once('=').ok_or_else(|| {
            anyhow::anyhow!(
                "{REAL_FULL_LAYER_ORDERED_EXPERT_TCP_TARGETS_ENV} entry {entry:?} must use host=ip:port"
            )
        })?;
        let host = host.trim();
        if host.is_empty() {
            anyhow::bail!(
                "{REAL_FULL_LAYER_ORDERED_EXPERT_TCP_TARGETS_ENV} entry {entry:?} has empty host"
            );
        }
        let addr = addr.trim().parse::<std::net::SocketAddr>().map_err(|error| {
            anyhow::anyhow!(
                "{REAL_FULL_LAYER_ORDERED_EXPERT_TCP_TARGETS_ENV} entry {entry:?} has invalid socket address: {error}"
            )
        })?;
        targets.push(TcpProtocolV2HostBatchTarget {
            host: host.to_owned(),
            addr,
        });
    }
    if targets.is_empty() {
        anyhow::bail!("{REAL_FULL_LAYER_ORDERED_EXPERT_TCP_TARGETS_ENV} did not contain any usable host=ip:port entries");
    }
    Ok(Some(targets))
}

fn block_on_sparse_moe_protocol_v2_residual_step(
    sparse: &RealFullSparseMlpSharedLayerHidden,
    residual_before_hidden: &[f32],
    targets: &[TcpProtocolV2HostBatchTarget],
    request_id_base: u64,
    config: TcpTransportConfig,
    kind: SparseMoeProtocolV2DispatchKind,
) -> Result<SparseMoeProtocolV2ResidualStep> {
    if tokio::runtime::Handle::try_current().is_ok() {
        anyhow::bail!(
            "{REAL_FULL_LAYER_ORDERED_EXPERT_TCP_TARGETS_ENV} live expert dispatch is only supported from the synchronous probe runner when no Tokio runtime is already active"
        );
    }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_io()
        .enable_time()
        .build()?;
    runtime.block_on(sparse_moe_protocol_v2_residual_step(
        sparse,
        residual_before_hidden,
        targets,
        request_id_base,
        config,
        kind,
    ))
}

fn skipped_real_full_layer_ordered_execution_probe(
    status: &'static str,
    skipped_reason: Option<String>,
) -> RealFullLayerOrderedResidualExecutionProbe {
    RealFullLayerOrderedResidualExecutionProbe {
        status,
        scope: "trace available real GLM-5.2 numeric residual execution in layer order and expose missing per-layer attention residual stages",
        row_mode: "bounded",
        hidden_source: "not-run",
        layer_count: GLM52_NUM_HIDDEN_LAYERS,
        traced_layers: 0,
        input_token_id: None,
        embedding_bytes_read: 0,
        embedding_residual_checksum: None,
        embedding_kernel_backend: None,
        embedding_device_resident: false,
        trace_steps: 0,
        attention_steps_executed: 0,
        attention_steps_missing: GLM52_NUM_HIDDEN_LAYERS,
        dense_mlp_steps_executed: 0,
        sparse_mlp_steps_executed: 0,
        shared_expert_steps_executed: 0,
        planned_residual_adds: GLM52_NUM_HIDDEN_LAYERS * 2,
        total_numeric_residual_adds: 0,
        residual_adds_missing: GLM52_NUM_HIDDEN_LAYERS * 2,
        residual_prefix_values: 0,
        routed_routes: 0,
        final_residual_checksum: None,
        covers_all_dense_layers: false,
        covers_all_sparse_layers: false,
        covers_full_top_k: false,
        covers_full_output_rows: false,
        carries_attention_into_dense: false,
        carries_dense_into_sparse: false,
        full_residual_stream_complete: false,
        uses_full_model_residual: false,
        coordinator_graph_slots: 0,
        coordinator_graph_captured_graphs: 0,
        coordinator_graph_captures: 0,
        coordinator_graph_launches: 0,
        uses_graph_captured_coordinator_kernels: false,
        scheduler_rows: skipped_layer_ordered_scheduler_rows(
            "not-run",
            Some("layer-ordered residual execution did not run".to_owned()),
        ),
        terminal_lm_head_sampling: skipped_layer_ordered_lm_head_sampling(
            "not-run",
            "not-run",
            0,
            None,
            Some("layer-ordered residual execution did not run".to_owned()),
        ),
        completion_gates: skipped_completion_gates(),
        full_residual_stream_blocker: Some(REAL_FULL_RESIDUAL_COMPLETION_BLOCKER),
        execution_stepper: skipped_execution_stepper(status),
        step_summaries: Vec::new(),
        passed: false,
        skipped_reason,
    }
}

fn run_real_full_layer_ordered_execution_probe_with_mode(
    catalog: &TensorCatalog,
    mode: LayerOrderedExecutionMode,
) -> Result<RealFullLayerOrderedResidualExecutionProbe> {
    let tensor_metadata = TensorMetadataLookup::new(catalog);
    let live_expert_targets = layer_ordered_expert_tcp_targets_from_env()?;
    let graph_stats_before = coordinator_cuda_graph_stats().ok();
    let scheduler_binding = layer_ordered_scheduler_rows_binding();
    let (initial_hidden, initial_device_hidden, initial_input) =
        if mode.initial_input.uses_embedding_residual_input {
            let token_id = mode
                .initial_input
                .token_id
                .unwrap_or(REAL_FULL_LAYER_ORDERED_DEFAULT_TOKEN_ID);
            let embedding = real_full_embedding_hidden_for_token(catalog, token_id)?;
            let embedding_device_resident = embedding.device_hidden.is_some();
            (
                embedding.hidden,
                embedding.device_hidden,
                InitialResidualInput {
                    token_id: Some(embedding.token_id),
                    embedding_bytes_read: embedding.bytes_read,
                    embedding_residual_checksum: Some(embedding.checksum),
                    embedding_kernel_backend: Some(embedding.kernel_backend),
                    embedding_device_resident,
                    uses_embedding_residual_input: true,
                },
            )
        } else {
            (
                deterministic_dense_hidden(GLM52_HIDDEN_SIZE),
                None,
                InitialResidualInput {
                    token_id: None,
                    embedding_bytes_read: 0,
                    embedding_residual_checksum: None,
                    embedding_kernel_backend: None,
                    embedding_device_resident: false,
                    uses_embedding_residual_input: false,
                },
            )
        };
    let mut prefix_context_hidden = initial_prefix_context_hidden(catalog, mode)?;
    let (attention, next_prefix_context_hidden) =
        attention_hidden_for_layer_from_initial_with_scheduler_prefix(
            catalog,
            0,
            initial_hidden,
            initial_device_hidden.as_ref(),
            mode,
            &scheduler_binding,
            prefix_context_hidden.take(),
        )?;
    prefix_context_hidden = next_prefix_context_hidden;
    let mut stepper = RealExecutionStepper::new(GLM52_NUM_HIDDEN_LAYERS, mode.row_mode);
    let mut attention_completion = MlaDsaAttentionCompletionTracker::default();
    attention_completion.record(&attention);
    stepper.record_attention(attention_step(&attention, &tensor_metadata));

    let mut device_hidden = attention.device_hidden;
    let mut hidden = attention.hidden;
    let mut dense_mlp_steps_executed = 0_usize;
    let mut dense_layers_ordered_and_passed = true;
    let mut dense_covers_full_output_rows = true;
    let mut attention_covers_full_output_rows =
        attention.residual_prefix_values == GLM52_HIDDEN_SIZE;
    let mut attention_into_dense_checksums_match = true;
    let mut previous_attention_final_checksum = attention.final_residual_checksum;
    let mut previous_attention_residual_prefix_values = attention.residual_prefix_values;
    let mut dense_final_checksum = attention.final_residual_checksum;
    let mut scheduler_layers_bound =
        scheduler_binding.probe.passed && scheduler_binding.layer_selected(attention.layer_id);
    for layer_id in 0..GLM52_FIRST_K_DENSE_REPLACE {
        if layer_id > attention.layer_id {
            let (layer_attention, next_prefix_context_hidden) =
                attention_hidden_for_layer_from_initial_with_scheduler_prefix(
                    catalog,
                    layer_id,
                    hidden,
                    device_hidden.as_ref(),
                    mode,
                    &scheduler_binding,
                    prefix_context_hidden.take(),
                )?;
            prefix_context_hidden = next_prefix_context_hidden;
            previous_attention_final_checksum = layer_attention.final_residual_checksum;
            previous_attention_residual_prefix_values = layer_attention.residual_prefix_values;
            attention_covers_full_output_rows = attention_covers_full_output_rows
                && layer_attention.residual_prefix_values == GLM52_HIDDEN_SIZE;
            attention_completion.record(&layer_attention);
            stepper.record_attention(attention_step(&layer_attention, &tensor_metadata));
            device_hidden = layer_attention.device_hidden;
            hidden = layer_attention.hidden;
            scheduler_layers_bound &= scheduler_binding.layer_selected(layer_id);
        }
        let dense_initial_attention_prefix_checksum =
            checksum_f64(&hidden[..previous_attention_residual_prefix_values]);
        let dense = if mode.dense_full_output {
            if let Some(device_hidden) = device_hidden.as_ref() {
                real_full_dense_layer_full_output_hidden_from_initial_device_input(
                    catalog,
                    layer_id,
                    hidden,
                    device_hidden,
                )?
            } else {
                real_full_dense_layer_full_output_hidden_from_initial(catalog, layer_id, hidden)?
            }
        } else {
            real_full_dense_layer_prefix_hidden_from_initial(catalog, layer_id, hidden)?
        };
        dense_mlp_steps_executed += 1;
        dense_layers_ordered_and_passed =
            dense_layers_ordered_and_passed && dense.layer_id == layer_id && dense.passed;
        dense_covers_full_output_rows =
            dense_covers_full_output_rows && dense.output_rows == GLM52_HIDDEN_SIZE;
        attention_into_dense_checksums_match = attention_into_dense_checksums_match
            && approx_eq_f64(
                dense_initial_attention_prefix_checksum,
                previous_attention_final_checksum,
            );
        dense_final_checksum = dense.final_residual_checksum;
        stepper.record_dense_mlp(dense_step(&dense, &tensor_metadata));
        device_hidden = dense.device_hidden;
        hidden = dense.hidden;
        prefix_context_hidden =
            dense_prefix_context_hidden_for_layer(catalog, layer_id, prefix_context_hidden, mode)?;
    }

    let carries_attention_into_dense = dense_mlp_steps_executed == GLM52_FIRST_K_DENSE_REPLACE
        && attention_into_dense_checksums_match;

    let mut sparse_mlp_steps_executed = 0_usize;
    let mut sparse_output_rows = 0_usize;
    let mut shared_expert_steps_executed = 0_usize;
    let mut sparse_layers_ordered_and_passed = true;
    let mut sparse_covers_full_top_k = true;
    let mut attention_into_sparse_checksums_match = true;
    let mut first_sparse_layer_id = None;
    let mut last_sparse_layer_id = None;
    let mut carries_dense_into_sparse = false;
    let mut sparse_final_checksum = dense_final_checksum;
    let mut live_expert_daemon_moe_layers = 0_usize;
    for layer_id in GLM52_FIRST_K_DENSE_REPLACE..GLM52_NUM_HIDDEN_LAYERS {
        scheduler_layers_bound &= scheduler_binding.layer_selected(layer_id);
        let pre_attention_prefix_checksum =
            checksum_f64(&hidden[..previous_attention_residual_prefix_values]);
        let (layer_attention, next_prefix_context_hidden) =
            attention_hidden_for_layer_from_initial_with_scheduler_prefix(
                catalog,
                layer_id,
                hidden,
                device_hidden.as_ref(),
                mode,
                &scheduler_binding,
                prefix_context_hidden.take(),
            )?;
        prefix_context_hidden = next_prefix_context_hidden;
        if layer_id == GLM52_FIRST_K_DENSE_REPLACE {
            carries_dense_into_sparse = approx_eq_f64(
                layer_attention.initial_residual_checksum,
                pre_attention_prefix_checksum,
            );
        }
        let attention_final_checksum = layer_attention.final_residual_checksum;
        previous_attention_residual_prefix_values = layer_attention.residual_prefix_values;
        attention_covers_full_output_rows = attention_covers_full_output_rows
            && layer_attention.residual_prefix_values == GLM52_HIDDEN_SIZE;
        attention_completion.record(&layer_attention);
        stepper.record_attention(attention_step(&layer_attention, &tensor_metadata));
        let sparse_initial_attention_prefix_checksum =
            checksum_f64(&layer_attention.hidden[..layer_attention.residual_prefix_values]);
        let sparse_input_device_hidden = layer_attention.device_hidden;
        let sparse_input_hidden = layer_attention.hidden;

        let sparse = if mode.sparse_full_output {
            if let Some(device_hidden) = sparse_input_device_hidden.as_ref() {
                real_sparse_mlp_shared_layer_full_output_hidden_from_initial_device_input(
                    catalog,
                    layer_id,
                    sparse_input_hidden.clone(),
                    device_hidden,
                )?
            } else {
                real_sparse_mlp_shared_layer_full_output_hidden_from_initial(
                    catalog,
                    layer_id,
                    sparse_input_hidden.clone(),
                )?
            }
        } else {
            real_sparse_mlp_shared_layer_hidden_from_initial(
                catalog,
                layer_id,
                sparse_input_hidden.clone(),
            )?
        };
        attention_into_sparse_checksums_match = attention_into_sparse_checksums_match
            && approx_eq_f64(
                sparse_initial_attention_prefix_checksum,
                attention_final_checksum,
            );
        sparse_mlp_steps_executed += 1;
        sparse_output_rows = sparse.output_rows;
        shared_expert_steps_executed += usize::from(sparse.shared_expert_executed);
        sparse_layers_ordered_and_passed =
            sparse_layers_ordered_and_passed && sparse.layer_id == layer_id && sparse.passed;
        sparse_covers_full_top_k = sparse_covers_full_top_k
            && sparse.covers_full_top_k
            && sparse.route_count == GLM52_TOP_K;
        first_sparse_layer_id.get_or_insert(sparse.layer_id);
        last_sparse_layer_id = Some(sparse.layer_id);
        let mut sparse_step = sparse_step(&sparse, &tensor_metadata);
        if let Some(targets) = live_expert_targets.as_ref() {
            let live_sparse = block_on_sparse_moe_protocol_v2_residual_step(
                &sparse,
                &sparse_input_hidden,
                targets,
                520_000 + layer_id as u64,
                TcpTransportConfig::default(),
                real_expertd_sparse_moe_dispatch_kind(),
            )?;
            if !live_sparse.dispatch_stats.output_checksum.is_finite()
                || !live_sparse.routed_output_checksum.is_finite()
                || !live_sparse.shared_output_checksum.is_finite()
            {
                anyhow::bail!(
                    "live sparse MoE ProtocolV2 dispatch produced non-finite checksum for layer {layer_id}"
                );
            }
            if live_sparse.residual_add_backend != sparse.residual_add_backend {
                anyhow::bail!(
                    "live sparse MoE residual-add backend mismatch for layer {layer_id}: live={} local={}",
                    live_sparse.residual_add_backend,
                    sparse.residual_add_backend
                );
            }
            sparse_final_checksum = live_sparse.residual_after_checksum;
            sparse_step.expert_host_batch_set = Some(live_sparse.host_batch_set_evidence);
            sparse_step.residual_delta_checksum = Some(live_sparse.residual_delta_checksum);
            sparse_step.residual_after_checksum = Some(live_sparse.residual_after_checksum);
            live_expert_daemon_moe_layers += 1;
            device_hidden = live_sparse.device_hidden;
            hidden = live_sparse.hidden_after;
        } else {
            sparse_final_checksum = sparse.final_residual_checksum;
            device_hidden = sparse.device_hidden;
            hidden = sparse.hidden;
        }
        prefix_context_hidden =
            sparse_prefix_context_hidden_for_layer(catalog, layer_id, prefix_context_hidden, mode)?;
        stepper.record_sparse_moe_mlp(sparse_step);
    }
    carries_dense_into_sparse = carries_dense_into_sparse && attention_into_sparse_checksums_match;

    let covers_all_dense_layers =
        dense_mlp_steps_executed == GLM52_FIRST_K_DENSE_REPLACE && dense_layers_ordered_and_passed;
    let covers_all_sparse_layers = sparse_mlp_steps_executed
        == GLM52_NUM_HIDDEN_LAYERS - GLM52_FIRST_K_DENSE_REPLACE
        && shared_expert_steps_executed == GLM52_NUM_HIDDEN_LAYERS - GLM52_FIRST_K_DENSE_REPLACE
        && first_sparse_layer_id == Some(GLM52_FIRST_K_DENSE_REPLACE)
        && last_sparse_layer_id == Some(GLM52_NUM_HIDDEN_LAYERS - 1)
        && sparse_layers_ordered_and_passed;
    let covers_full_output_rows = attention_covers_full_output_rows
        && dense_covers_full_output_rows
        && sparse_output_rows == GLM52_HIDDEN_SIZE;
    let (status, scope, hidden_source) = layer_ordered_probe_labels(mode);
    let terminal_lm_head_sampling = layer_ordered_lm_head_sampling_probe(
        catalog,
        hidden_source,
        &hidden,
        device_hidden.as_ref(),
        covers_full_output_rows,
        Some(sparse_final_checksum),
        mode.lm_head_full_vocab || layer_ordered_lm_head_full_vocab_env_enabled(),
    );
    scheduler_layers_bound &= scheduler_binding.covers_all_layers();
    let uses_live_scheduler_rows =
        scheduler_binding.probe.uses_live_scheduler_rows && scheduler_layers_bound;
    let uses_live_expert_daemon_moe = live_expert_targets.is_some()
        && live_expert_daemon_moe_layers == sparse_mlp_steps_executed
        && sparse_mlp_steps_executed == GLM52_NUM_HIDDEN_LAYERS - GLM52_FIRST_K_DENSE_REPLACE;
    let uses_real_lm_head_sampling_residual =
        terminal_lm_head_sampling_satisfies_completion_gate(&terminal_lm_head_sampling);
    let uses_full_model_residual = covers_all_dense_layers
        && covers_all_sparse_layers
        && sparse_covers_full_top_k
        && covers_full_output_rows
        && initial_input.uses_embedding_residual_input
        && uses_live_scheduler_rows
        && attention_completion.uses_full_context_mla_dsa_attention()
        && uses_live_expert_daemon_moe
        && uses_real_lm_head_sampling_residual;
    let coordinator_graphs = coordinator_graph_replay_delta(graph_stats_before);
    let RealExecutionStepperOutput {
        report: execution_stepper,
        steps,
    } = stepper.finish(RealExecutionStepperFinish {
        covers_all_dense_layers,
        covers_all_sparse_layers,
        covers_full_top_k: sparse_covers_full_top_k,
        covers_full_output_rows,
        uses_embedding_residual_input: initial_input.uses_embedding_residual_input,
        uses_live_scheduler_rows,
        uses_full_context_mla_dsa_attention: attention_completion
            .uses_full_context_mla_dsa_attention(),
        uses_live_expert_daemon_moe,
        uses_real_lm_head_sampling_residual,
        uses_full_model_residual,
        coordinator_graph_slots: coordinator_graphs.slots,
        coordinator_graph_captured_graphs: coordinator_graphs.captured_graphs,
        coordinator_graph_captures: coordinator_graphs.graph_captures,
        coordinator_graph_launches: coordinator_graphs.graph_launches,
        bounded_attention_oracle: bounded_attention_oracle_stepper_evidence(),
        full_residual_stream_blocker: Some(REAL_FULL_RESIDUAL_COMPLETION_BLOCKER),
        final_residual_checksum: Some(sparse_final_checksum),
    });
    let traced_layers = execution_stepper.traced_layers;
    let attention_steps_executed = execution_stepper.attention_steps_executed;
    let attention_steps_missing = execution_stepper.attention_steps_missing;
    let dense_mlp_steps_executed = execution_stepper.dense_mlp_steps_executed;
    let sparse_mlp_steps_executed = execution_stepper.sparse_mlp_steps_executed;
    let shared_expert_steps_executed = execution_stepper.shared_expert_steps_executed;
    let planned_residual_adds = execution_stepper.planned_residual_adds;
    let total_numeric_residual_adds = execution_stepper.total_numeric_residual_adds;
    let residual_adds_missing = execution_stepper.residual_adds_missing;
    let routed_routes = execution_stepper.routed_routes;
    let full_residual_stream_complete = execution_stepper.full_residual_stream_complete;
    let passed = traced_layers == GLM52_NUM_HIDDEN_LAYERS
        && attention_steps_executed == GLM52_NUM_HIDDEN_LAYERS
        && attention_steps_missing == 0
        && covers_all_dense_layers
        && covers_all_sparse_layers
        && sparse_covers_full_top_k
        && carries_attention_into_dense
        && carries_dense_into_sparse
        && residual_adds_missing == 0
        && sparse_final_checksum.is_finite();

    Ok(RealFullLayerOrderedResidualExecutionProbe {
        status,
        scope,
        row_mode: mode.row_mode,
        hidden_source,
        layer_count: GLM52_NUM_HIDDEN_LAYERS,
        traced_layers,
        input_token_id: initial_input.token_id,
        embedding_bytes_read: initial_input.embedding_bytes_read,
        embedding_residual_checksum: initial_input.embedding_residual_checksum,
        embedding_kernel_backend: initial_input.embedding_kernel_backend,
        embedding_device_resident: initial_input.embedding_device_resident,
        trace_steps: execution_stepper.trace_steps,
        attention_steps_executed,
        attention_steps_missing,
        dense_mlp_steps_executed,
        sparse_mlp_steps_executed,
        shared_expert_steps_executed,
        planned_residual_adds,
        total_numeric_residual_adds,
        residual_adds_missing,
        residual_prefix_values: execution_stepper.residual_prefix_values,
        routed_routes,
        final_residual_checksum: Some(sparse_final_checksum),
        covers_all_dense_layers,
        covers_all_sparse_layers,
        covers_full_top_k: sparse_covers_full_top_k,
        covers_full_output_rows,
        carries_attention_into_dense,
        carries_dense_into_sparse,
        full_residual_stream_complete,
        uses_full_model_residual,
        coordinator_graph_slots: execution_stepper.coordinator_graph_slots,
        coordinator_graph_captured_graphs: execution_stepper.coordinator_graph_captured_graphs,
        coordinator_graph_captures: execution_stepper.coordinator_graph_captures,
        coordinator_graph_launches: execution_stepper.coordinator_graph_launches,
        uses_graph_captured_coordinator_kernels: execution_stepper
            .uses_graph_captured_coordinator_kernels,
        scheduler_rows: scheduler_binding.probe,
        terminal_lm_head_sampling,
        completion_gates: execution_stepper.completion_gates.clone(),
        full_residual_stream_blocker: execution_stepper.full_residual_stream_blocker,
        execution_stepper,
        step_summaries: steps,
        passed,
        skipped_reason: None,
    })
}

fn layer_ordered_probe_labels(
    mode: LayerOrderedExecutionMode,
) -> (&'static str, &'static str, &'static str) {
    if mode.row_mode == "full-output-mla-rope-attention-mlp" {
        (
            "numeric-real-layer-ordered-full-output-mla-rope-attention-mlp-residual-trace",
            "runs real numeric residual steps in model order for every layer with hidden-width BF16 main MLA/RoPE attention over scheduler-admitted KV-cache prefix context before full-output dense MLP layers 0..2 and full-output routed NVFP4 plus BF16 shared sparse MLP layers 3..77, including bounded DSA/indexer attention on configured DSA layers; live expert daemons and full-model residual ownership remain separate completion gates",
            if mode.initial_input.uses_embedding_residual_input {
                "real-embedding-token-hidden-carried-through-full-output-mla-rope-attention-mlp-order-for-all-layers"
            } else {
                "deterministic-hidden-carried-through-full-output-mla-rope-attention-mlp-order-for-all-layers"
            },
        )
    } else if mode.row_mode == "full-output-mla-rope-attention" {
        (
            "numeric-real-layer-ordered-full-output-mla-rope-attention-residual-trace",
            "runs real numeric residual steps in model order for every layer with hidden-width BF16 main MLA/RoPE attention over scheduler-admitted KV-cache prefix context before dense MLP layers 0..2 and before routed NVFP4 plus BF16 shared sparse MLP layers 3..77, including bounded DSA/indexer attention on configured DSA layers; MLP completion and full-model residual ownership remain separate completion gates",
            if mode.initial_input.uses_embedding_residual_input {
                "real-embedding-token-hidden-carried-through-full-output-mla-rope-attention-mlp-order-for-all-layers"
            } else {
                "deterministic-hidden-carried-through-full-output-mla-rope-attention-mlp-order-for-all-layers"
            },
        )
    } else if mode.row_mode == "mla-rope-attention" {
        (
            "numeric-real-layer-ordered-bounded-mla-rope-attention-residual-trace",
            "runs real numeric residual steps in model order for every layer with bounded BF16 main MLA/RoPE attention over scheduler-admitted KV-cache prefix context before dense MLP layers 0..2 and before routed NVFP4 plus BF16 shared sparse MLP layers 3..77, including bounded DSA/indexer attention on configured DSA layers; full-output rows and full-model residual ownership remain separate completion gates",
            if mode.initial_input.uses_embedding_residual_input {
                "real-embedding-token-hidden-carried-through-bounded-mla-rope-attention-mlp-order-for-all-layers"
            } else {
                "deterministic-hidden-carried-through-bounded-mla-rope-attention-mlp-order-for-all-layers"
            },
        )
    } else if mode.row_mode == "full-output-attention-mlp" {
        (
            "numeric-real-layer-ordered-full-output-attention-mlp-residual-trace",
            "runs real numeric residual steps in model order for every layer with hidden-width BF16 causal attention output rows before full-output dense MLP layers 0..2 and full-output routed NVFP4 plus BF16 shared sparse MLP layers 3..77; full attention context, full MLA/RoPE attention, and full-model residuals remain omitted",
            if mode.initial_input.uses_embedding_residual_input {
                "real-embedding-token-hidden-carried-through-full-output-attention-mlp-order-for-all-layers"
            } else {
                "deterministic-layer0-full-output-attention-hidden-carried-through-full-output-attention-mlp-order-for-all-layers"
            },
        )
    } else if mode.row_mode == "full-output-mlp" {
        (
            "numeric-real-layer-ordered-full-output-mlp-bounded-attention-residual-trace",
            "runs real numeric residual steps in model order for every layer with bounded BF16 attention before full-output dense MLP layers 0..2 and bounded BF16 attention before full-output routed NVFP4 plus BF16 shared sparse MLP layers 3..77; full-output attention, full MLA/RoPE attention, and full-model residuals remain omitted",
            if mode.initial_input.uses_embedding_residual_input {
                "real-embedding-token-hidden-carried-through-bounded-attention-full-output-mlp-order-for-all-layers"
            } else {
                "deterministic-layer0-attention-hidden-carried-through-bounded-attention-full-output-mlp-order-for-all-layers"
            },
        )
    } else {
        (
            "numeric-real-layer-ordered-bounded-all-stage-residual-trace",
            "runs bounded real numeric residual steps in model order for every layer: BF16 attention before dense MLP layers 0..2 and BF16 attention before routed NVFP4 plus BF16 shared sparse MLP layers 3..77; full-output residual rows and full MLA/RoPE attention remain omitted",
            if mode.initial_input.uses_embedding_residual_input {
                "real-embedding-token-hidden-carried-through-bounded-attention-mlp-order-for-all-layers"
            } else {
                "deterministic-layer0-attention-hidden-carried-through-bounded-attention-mlp-order-for-all-layers"
            },
        )
    }
}

fn layer_ordered_lm_head_sampling_probe(
    catalog: &TensorCatalog,
    hidden_source: &'static str,
    hidden: &[f32],
    device_hidden: Option<&DeviceBf16Output>,
    covers_full_output_rows: bool,
    residual_after_checksum: Option<f64>,
    score_full_vocab: bool,
) -> RealFullLayerOrderedLmHeadSamplingProbe {
    if !covers_full_output_rows {
        return skipped_layer_ordered_lm_head_sampling(
            "not-run",
            hidden_source,
            hidden.len(),
            residual_after_checksum,
            Some("layer-ordered lm_head scoring requires full-output residual rows".to_owned()),
        );
    }
    let score_result = if score_full_vocab {
        if let Some(device_hidden) = device_hidden {
            score_real_lm_head_full_vocab_for_device_hidden(
                catalog,
                device_hidden,
                REAL_FULL_LAYER_ORDERED_LM_HEAD_CHUNK_ROWS,
            )
        } else {
            score_real_lm_head_full_vocab_for_hidden(
                catalog,
                hidden,
                REAL_FULL_LAYER_ORDERED_LM_HEAD_CHUNK_ROWS,
            )
        }
    } else {
        score_real_lm_head_chunk_for_hidden(
            catalog,
            hidden,
            REAL_FULL_LAYER_ORDERED_LM_HEAD_CHUNK_ROWS,
        )
    };
    match score_result {
        Ok(score) => {
            layer_ordered_lm_head_sampling_from_score(hidden_source, score, residual_after_checksum)
        }
        Err(error) => skipped_layer_ordered_lm_head_sampling(
            "error",
            hidden_source,
            hidden.len(),
            residual_after_checksum,
            Some(format!("{error:#}")),
        ),
    }
}

fn layer_ordered_lm_head_full_vocab_env_enabled() -> bool {
    std::env::var(REAL_FULL_LAYER_ORDERED_LM_HEAD_FULL_VOCAB_ENV).as_deref() == Ok("1")
}

fn skipped_layer_ordered_scheduler_rows(
    status: &'static str,
    skipped_reason: Option<String>,
) -> super::super::super::types::RealFullLayerOrderedSchedulerRowsProbe {
    super::super::super::types::RealFullLayerOrderedSchedulerRowsProbe {
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
    }
}

fn layer_ordered_lm_head_sampling_from_score(
    hidden_source: &'static str,
    score: RealLmHeadChunkScoreForHidden,
    residual_after_checksum: Option<f64>,
) -> RealFullLayerOrderedLmHeadSamplingProbe {
    let hidden_bytes = score.hidden_values * std::mem::size_of::<f32>();
    let passed = score.rows_scored > 0
        && score.hidden_dim == GLM52_HIDDEN_SIZE
        && score.logits_evaluated == score.rows_scored
        && score.top_logit.is_finite();
    let covers_full_vocabulary = score.covers_full_vocabulary;
    RealFullLayerOrderedLmHeadSamplingProbe {
        status: if covers_full_vocabulary {
            "numeric-real-layer-ordered-lm-head-full-vocab"
        } else {
            "numeric-real-layer-ordered-lm-head-chunk"
        },
        scope: if covers_full_vocabulary {
            "stream-score the full real lm_head.weight vocabulary against the terminal layer-ordered residual hidden row"
        } else {
            "score a bounded real lm_head.weight chunk against the terminal layer-ordered residual hidden row"
        },
        hidden_source,
        uses_real_lm_head: true,
        uses_layer_ordered_residual: true,
        uses_layer_ordered_full_output_residual: true,
        uses_full_model_residual: false,
        lm_head_tensor: Some(score.lm_head_tensor),
        hidden_dim: score.hidden_dim,
        vocab_size: score.vocab_size,
        start_token_id: score.start_token_id,
        chunk_rows: score.chunk_rows,
        rows_scored: score.rows_scored,
        chunks_scored: score.chunks_scored,
        lm_head_bytes_read: score.lm_head_bytes_read,
        hidden_values: score.hidden_values,
        hidden_bytes,
        logits_evaluated: score.logits_evaluated,
        multiply_accumulate_ops: score.multiply_accumulate_ops,
        logits_kernel_backend: Some(score.logits_kernel_backend),
        argmax_kernel_backend: Some(score.argmax_kernel_backend),
        sampler_kernel_backend: Some(score.sampler_kernel_backend),
        covers_full_vocabulary,
        top_token_id: Some(score.top_token_id),
        top_logit: Some(score.top_logit),
        sampled_token_id: Some(score.sampled_token_id),
        sampled_score: Some(score.sampled_score),
        sample_random_uniform: Some(score.sample_random_uniform),
        sample_temperature: Some(score.sample_temperature),
        sample_top_k: Some(score.sample_top_k),
        sample_top_p: Some(score.sample_top_p),
        residual_after_checksum,
        passed,
        skipped_reason: None,
    }
}

fn terminal_lm_head_sampling_satisfies_completion_gate(
    probe: &RealFullLayerOrderedLmHeadSamplingProbe,
) -> bool {
    probe.passed
        && probe.uses_real_lm_head
        && probe.uses_layer_ordered_residual
        && probe.uses_layer_ordered_full_output_residual
        && probe.hidden_dim == GLM52_HIDDEN_SIZE
        && probe.covers_full_vocabulary
        && probe.rows_scored == probe.vocab_size
        && probe.logits_evaluated == probe.vocab_size
        && probe.top_logit.is_some_and(f32::is_finite)
        && probe
            .sampled_score
            .is_some_and(|score| score.is_finite() && score > 0.0 && score <= 1.0)
        && probe.sample_top_k.is_some_and(|top_k| top_k > 1)
        && probe
            .sample_temperature
            .is_some_and(|temperature| temperature.is_finite() && temperature > 0.0)
        && probe
            .sample_top_p
            .is_some_and(|top_p| top_p.is_finite() && top_p > 0.0 && top_p <= 1.0)
        && probe.argmax_kernel_backend
            == Some(CUDA_REFERENCE_LM_HEAD_ARGMAX_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND)
        && probe
            .sampler_kernel_backend
            .is_some_and(terminal_lm_head_sampler_backend_satisfies_completion_gate)
}

fn terminal_lm_head_sampler_backend_satisfies_completion_gate(backend: &str) -> bool {
    matches!(
        backend,
        CUDA_REFERENCE_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
            | TRITON_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
    )
}

fn skipped_layer_ordered_lm_head_sampling(
    status: &'static str,
    hidden_source: &'static str,
    hidden_values: usize,
    residual_after_checksum: Option<f64>,
    skipped_reason: Option<String>,
) -> RealFullLayerOrderedLmHeadSamplingProbe {
    RealFullLayerOrderedLmHeadSamplingProbe {
        status,
        scope: "score a bounded real lm_head.weight chunk against the terminal layer-ordered residual hidden row",
        hidden_source,
        uses_real_lm_head: false,
        uses_layer_ordered_residual: false,
        uses_layer_ordered_full_output_residual: false,
        uses_full_model_residual: false,
        lm_head_tensor: None,
        hidden_dim: hidden_values,
        vocab_size: 0,
        start_token_id: 0,
        chunk_rows: REAL_FULL_LAYER_ORDERED_LM_HEAD_CHUNK_ROWS,
        rows_scored: 0,
        chunks_scored: 0,
        lm_head_bytes_read: 0,
        hidden_values,
        hidden_bytes: hidden_values * std::mem::size_of::<f32>(),
        logits_evaluated: 0,
        multiply_accumulate_ops: 0,
        logits_kernel_backend: None,
        argmax_kernel_backend: None,
        sampler_kernel_backend: None,
        covers_full_vocabulary: false,
        top_token_id: None,
        top_logit: None,
        sampled_token_id: None,
        sampled_score: None,
        sample_random_uniform: None,
        sample_temperature: None,
        sample_top_k: None,
        sample_top_p: None,
        residual_after_checksum,
        passed: false,
        skipped_reason,
    }
}

fn skipped_execution_stepper(status: &'static str) -> RealFullResidualExecutionStepper {
    RealFullResidualExecutionStepper {
        status,
        scope: "records reusable real GLM-5.2 residual execution stages in model order; current trace is bounded and reports full-output/full-model completion separately",
        row_mode: "bounded",
        layer_count: GLM52_NUM_HIDDEN_LAYERS,
        traced_layers: 0,
        trace_steps: 0,
        attention_steps_executed: 0,
        attention_steps_missing: GLM52_NUM_HIDDEN_LAYERS,
        dense_mlp_steps_executed: 0,
        sparse_mlp_steps_executed: 0,
        shared_expert_steps_executed: 0,
        planned_residual_adds: GLM52_NUM_HIDDEN_LAYERS * 2,
        total_numeric_residual_adds: 0,
        residual_adds_missing: GLM52_NUM_HIDDEN_LAYERS * 2,
        residual_prefix_values: 0,
        routed_routes: 0,
        stage_sources_recorded: 0,
        stage_statuses_recorded: 0,
        real_stage_count: 0,
        synthetic_stage_count: 0,
        provisional_stage_count: 0,
        blocked_stage_count: 0,
        coordinator_stage_count: 0,
        coordinator_cuda_stage_count: 0,
        coordinator_cpu_stage_count: 0,
        coordinator_unknown_stage_count: 0,
        uses_cuda_coordinator_kernels: false,
        coordinator_graph_slots: 0,
        coordinator_graph_captured_graphs: 0,
        coordinator_graph_captures: 0,
        coordinator_graph_launches: 0,
        uses_graph_captured_coordinator_kernels: false,
        stages_with_numeric_checksums: 0,
        total_numeric_checksum_fields: 0,
        numeric_checksum_fields_per_stage: 0,
        stages_with_tensor_artifacts: 0,
        total_tensor_artifacts: 0,
        attention_tensor_artifacts_per_stage: 0,
        dense_mlp_tensor_artifacts_per_stage: 0,
        sparse_mlp_tensor_artifacts_per_stage: 0,
        final_residual_checksum: None,
        covers_all_layers: false,
        covers_all_dense_layers: false,
        covers_all_sparse_layers: false,
        covers_full_top_k: false,
        covers_full_output_rows: false,
        stage_order_verified: false,
        full_residual_stream_complete: false,
        uses_full_model_residual: false,
        bounded_attention_oracle:
            super::super::super::types::RealFullBoundedAttentionOracleStepperEvidence::default(),
        completion_gates: skipped_completion_gates(),
        full_residual_stream_blocker: Some(REAL_FULL_RESIDUAL_COMPLETION_BLOCKER),
    }
}

fn skipped_completion_gates() -> RealFullResidualCompletionGates {
    RealFullResidualCompletionGates {
        numeric_layer_order_complete: false,
        attention_steps_complete: false,
        residual_adds_complete: false,
        covers_full_output_rows: false,
        uses_embedding_residual_input: false,
        uses_live_scheduler_rows: false,
        uses_cuda_coordinator_kernels: false,
        uses_graph_captured_coordinator_kernels: false,
        uses_full_context_mla_dsa_attention: false,
        uses_live_expert_daemon_moe: false,
        uses_real_lm_head_sampling_residual: false,
        uses_full_model_residual: false,
        ready_for_full_residual_stream: false,
        missing_gate_count: 12,
        missing_gate_names: vec![
            "numeric_layer_order_complete",
            "attention_steps_complete",
            "residual_adds_complete",
            "covers_full_output_rows",
            "uses_embedding_residual_input",
            "uses_live_scheduler_rows",
            "uses_cuda_coordinator_kernels",
            "uses_graph_captured_coordinator_kernels",
            "uses_full_context_mla_dsa_attention",
            "uses_live_expert_daemon_moe",
            "uses_real_lm_head_sampling_residual",
            "uses_full_model_residual",
        ],
    }
}

fn attention_step(
    attention: &RealFullAttentionResidualPrefixHidden,
    tensor_metadata: &TensorMetadataLookup<'_>,
) -> RealFullLayerOrderedResidualExecutionStep {
    let stage_source = attention_stage_source(attention);
    RealFullLayerOrderedResidualExecutionStep {
        layer_id: attention.layer_id,
        stage: "attention",
        stage_source,
        stage_status: "real",
        executed: true,
        residual_adds: attention.residual_adds,
        output_rows: attention.residual_prefix_values,
        routes_executed: 0,
        selected_routes: Vec::new(),
        expert_host_batch_set: None,
        includes_shared_expert: false,
        tensor_artifacts: attention_tensor_artifacts(
            tensor_metadata,
            attention.layer_id,
            attention.residual_prefix_values,
        ),
        residual_before_checksum: Some(attention.initial_residual_checksum),
        residual_delta_checksum: Some(attention.residual_delta_checksum),
        residual_after_checksum: Some(attention.final_residual_checksum),
        missing_reason: None,
    }
}

fn attention_stage_source(attention: &RealFullAttentionResidualPrefixHidden) -> &'static str {
    let projection_backend = coordinator_linear_backend_family(attention.projection_backend);
    if attention.includes_mla_softmax {
        if attention.uses_kv_cache_context && attention.kv_cache_context_bytes > 0 {
            return match (
                attention.residual_prefix_values == GLM52_HIDDEN_SIZE,
                projection_backend,
                attention.attention_backend,
                attention.residual_add_backend,
            ) {
                (
                    true,
                    CUDA_REFERENCE_LINEAR_BF16_BACKEND,
                    CUDA_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND,
                    CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
                ) => {
                    "real-checkpoint-bf16-full-output-main-mla-rope-kv-cache-prefix-context-attention-cuda-reference-linear-bf16-cuda-reference-mla-rope-attention-bf16-cuda-reference-residual-add-bf16"
                }
                (
                    false,
                    CUDA_REFERENCE_LINEAR_BF16_BACKEND,
                    CUDA_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND,
                    CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
                ) => {
                    "real-checkpoint-bf16-bounded-main-mla-rope-kv-cache-prefix-context-attention-cuda-reference-linear-bf16-cuda-reference-mla-rope-attention-bf16-cuda-reference-residual-add-bf16"
                }
                (
                    true,
                    CPU_REFERENCE_LINEAR_BF16_BACKEND,
                    CPU_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND,
                    CPU_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
                ) => {
                    "real-checkpoint-bf16-full-output-main-mla-rope-kv-cache-prefix-context-attention-cpu-reference-linear-bf16-cpu-reference-mla-rope-attention-bf16-cpu-reference-residual-add-bf16"
                }
                (
                    false,
                    CPU_REFERENCE_LINEAR_BF16_BACKEND,
                    CPU_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND,
                    CPU_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
                ) => {
                    "real-checkpoint-bf16-bounded-main-mla-rope-kv-cache-prefix-context-attention-cpu-reference-linear-bf16-cpu-reference-mla-rope-attention-bf16-cpu-reference-residual-add-bf16"
                }
                (true, _, _, _) => {
                    "real-checkpoint-bf16-full-output-main-mla-rope-kv-cache-prefix-context-attention"
                }
                (false, _, _, _) => {
                    "real-checkpoint-bf16-bounded-main-mla-rope-kv-cache-prefix-context-attention"
                }
            };
        }
        if attention.prefix_context_rows > 0
            && attention.total_context_rows > attention.attention_rows
        {
            let full_output = attention.residual_prefix_values == GLM52_HIDDEN_SIZE;
            let includes_dsa =
                attention.includes_dsa_candidate_selection && attention.includes_dsa_softmax;
            return match (
                full_output,
                includes_dsa,
                projection_backend,
                attention.attention_backend,
                attention.residual_add_backend,
            ) {
                (
                    true,
                    true,
                    CUDA_REFERENCE_LINEAR_BF16_BACKEND,
                    CUDA_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND,
                    CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
                ) => {
                    "real-checkpoint-bf16-full-output-main-mla-rope-supplied-prefix-context-plus-dsa-indexer-attention-cuda-reference-linear-bf16-cuda-reference-mla-rope-attention-bf16-cuda-reference-residual-add-bf16"
                }
                (
                    false,
                    true,
                    CUDA_REFERENCE_LINEAR_BF16_BACKEND,
                    CUDA_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND,
                    CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
                ) => {
                    "real-checkpoint-bf16-bounded-main-mla-rope-supplied-prefix-context-plus-dsa-indexer-attention-cuda-reference-linear-bf16-cuda-reference-mla-rope-attention-bf16-cuda-reference-residual-add-bf16"
                }
                (
                    true,
                    false,
                    CUDA_REFERENCE_LINEAR_BF16_BACKEND,
                    CUDA_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND,
                    CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
                ) => {
                    "real-checkpoint-bf16-full-output-main-mla-rope-supplied-prefix-context-attention-cuda-reference-linear-bf16-cuda-reference-mla-rope-attention-bf16-cuda-reference-residual-add-bf16"
                }
                (
                    false,
                    false,
                    CUDA_REFERENCE_LINEAR_BF16_BACKEND,
                    CUDA_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND,
                    CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
                ) => {
                    "real-checkpoint-bf16-bounded-main-mla-rope-supplied-prefix-context-attention-cuda-reference-linear-bf16-cuda-reference-mla-rope-attention-bf16-cuda-reference-residual-add-bf16"
                }
                (
                    true,
                    true,
                    CPU_REFERENCE_LINEAR_BF16_BACKEND,
                    CPU_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND,
                    CPU_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
                ) => {
                    "real-checkpoint-bf16-full-output-main-mla-rope-supplied-prefix-context-plus-dsa-indexer-attention-cpu-reference-linear-bf16-cpu-reference-mla-rope-attention-bf16-cpu-reference-residual-add-bf16"
                }
                (
                    false,
                    true,
                    CPU_REFERENCE_LINEAR_BF16_BACKEND,
                    CPU_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND,
                    CPU_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
                ) => {
                    "real-checkpoint-bf16-bounded-main-mla-rope-supplied-prefix-context-plus-dsa-indexer-attention-cpu-reference-linear-bf16-cpu-reference-mla-rope-attention-bf16-cpu-reference-residual-add-bf16"
                }
                (
                    true,
                    false,
                    CPU_REFERENCE_LINEAR_BF16_BACKEND,
                    CPU_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND,
                    CPU_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
                ) => {
                    "real-checkpoint-bf16-full-output-main-mla-rope-supplied-prefix-context-attention-cpu-reference-linear-bf16-cpu-reference-mla-rope-attention-bf16-cpu-reference-residual-add-bf16"
                }
                (
                    false,
                    false,
                    CPU_REFERENCE_LINEAR_BF16_BACKEND,
                    CPU_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND,
                    CPU_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
                ) => {
                    "real-checkpoint-bf16-bounded-main-mla-rope-supplied-prefix-context-attention-cpu-reference-linear-bf16-cpu-reference-mla-rope-attention-bf16-cpu-reference-residual-add-bf16"
                }
                (true, true, _, _, _) => {
                    "real-checkpoint-bf16-full-output-main-mla-rope-supplied-prefix-context-plus-dsa-indexer-attention"
                }
                (false, true, _, _, _) => {
                    "real-checkpoint-bf16-bounded-main-mla-rope-supplied-prefix-context-plus-dsa-indexer-attention"
                }
                (true, false, _, _, _) => {
                    "real-checkpoint-bf16-full-output-main-mla-rope-supplied-prefix-context-attention"
                }
                (false, false, _, _, _) => {
                    "real-checkpoint-bf16-bounded-main-mla-rope-supplied-prefix-context-attention"
                }
            };
        }
        if attention.includes_dsa_candidate_selection && attention.includes_dsa_softmax {
            return match (
                attention.residual_prefix_values == GLM52_HIDDEN_SIZE,
                projection_backend,
                attention.attention_backend,
                attention.residual_add_backend,
            ) {
                (
                    true,
                    CUDA_REFERENCE_LINEAR_BF16_BACKEND,
                    CUDA_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND,
                    CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
                ) => {
                    "real-checkpoint-bf16-full-output-main-mla-rope-plus-dsa-indexer-attention-cuda-reference-linear-bf16-cuda-reference-mla-rope-attention-bf16-cuda-reference-residual-add-bf16"
                }
                (
                    false,
                    CUDA_REFERENCE_LINEAR_BF16_BACKEND,
                    CUDA_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND,
                    CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
                ) => {
                    "real-checkpoint-bf16-bounded-main-mla-rope-plus-dsa-indexer-attention-cuda-reference-linear-bf16-cuda-reference-mla-rope-attention-bf16-cuda-reference-residual-add-bf16"
                }
                (
                    true,
                    CPU_REFERENCE_LINEAR_BF16_BACKEND,
                    CPU_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND,
                    CPU_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
                ) => {
                    "real-checkpoint-bf16-full-output-main-mla-rope-plus-dsa-indexer-attention-cpu-reference-linear-bf16-cpu-reference-mla-rope-attention-bf16-cpu-reference-residual-add-bf16"
                }
                (
                    false,
                    CPU_REFERENCE_LINEAR_BF16_BACKEND,
                    CPU_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND,
                    CPU_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
                ) => {
                    "real-checkpoint-bf16-bounded-main-mla-rope-plus-dsa-indexer-attention-cpu-reference-linear-bf16-cpu-reference-mla-rope-attention-bf16-cpu-reference-residual-add-bf16"
                }
                (true, _, _, _) => {
                    "real-checkpoint-bf16-full-output-main-mla-rope-plus-dsa-indexer-attention"
                }
                (false, _, _, _) => {
                    "real-checkpoint-bf16-bounded-main-mla-rope-plus-dsa-indexer-attention"
                }
            };
        }
        return match (
            attention.residual_prefix_values == GLM52_HIDDEN_SIZE,
            projection_backend,
            attention.attention_backend,
            attention.residual_add_backend,
        ) {
            (
                true,
                CUDA_REFERENCE_LINEAR_BF16_BACKEND,
                CUDA_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND,
                CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
            ) => {
                "real-checkpoint-bf16-full-output-main-mla-rope-attention-cuda-reference-linear-bf16-cuda-reference-mla-rope-attention-bf16-cuda-reference-residual-add-bf16"
            }
            (
                false,
                CUDA_REFERENCE_LINEAR_BF16_BACKEND,
                CUDA_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND,
                CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
            ) => {
                "real-checkpoint-bf16-bounded-main-mla-rope-attention-cuda-reference-linear-bf16-cuda-reference-mla-rope-attention-bf16-cuda-reference-residual-add-bf16"
            }
            (
                true,
                CPU_REFERENCE_LINEAR_BF16_BACKEND,
                CPU_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND,
                CPU_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
            ) => {
                "real-checkpoint-bf16-full-output-main-mla-rope-attention-cpu-reference-linear-bf16-cpu-reference-mla-rope-attention-bf16-cpu-reference-residual-add-bf16"
            }
            (
                false,
                CPU_REFERENCE_LINEAR_BF16_BACKEND,
                CPU_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND,
                CPU_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
            ) => {
                "real-checkpoint-bf16-bounded-main-mla-rope-attention-cpu-reference-linear-bf16-cpu-reference-mla-rope-attention-bf16-cpu-reference-residual-add-bf16"
            }
            (true, _, _, _) => "real-checkpoint-bf16-full-output-main-mla-rope-attention",
            (false, _, _, _) => "real-checkpoint-bf16-bounded-main-mla-rope-attention",
        };
    }
    match (
        attention.residual_prefix_values == GLM52_HIDDEN_SIZE,
        projection_backend,
        attention.attention_backend,
        attention.residual_add_backend,
    ) {
        (
            true,
            CUDA_REFERENCE_LINEAR_BF16_BACKEND,
            CUDA_REFERENCE_CAUSAL_ATTENTION_BF16_BACKEND,
            CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
        ) => {
            "real-checkpoint-bf16-causal-attention-full-output-rows-cuda-reference-linear-bf16-cuda-reference-causal-attention-bf16-cuda-reference-residual-add-bf16"
        }
        (
            false,
            CUDA_REFERENCE_LINEAR_BF16_BACKEND,
            CUDA_REFERENCE_CAUSAL_ATTENTION_BF16_BACKEND,
            CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
        ) => {
            "real-checkpoint-bf16-causal-attention-prefix-cuda-reference-linear-bf16-cuda-reference-causal-attention-bf16-cuda-reference-residual-add-bf16"
        }
        (
            true,
            CPU_REFERENCE_LINEAR_BF16_BACKEND,
            CPU_REFERENCE_CAUSAL_ATTENTION_BF16_BACKEND,
            CPU_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
        ) => {
            "real-checkpoint-bf16-causal-attention-full-output-rows-cpu-reference-linear-bf16-cpu-reference-causal-attention-bf16-cpu-reference-residual-add-bf16"
        }
        (
            false,
            CPU_REFERENCE_LINEAR_BF16_BACKEND,
            CPU_REFERENCE_CAUSAL_ATTENTION_BF16_BACKEND,
            CPU_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
        ) => {
            "real-checkpoint-bf16-causal-attention-prefix-cpu-reference-linear-bf16-cpu-reference-causal-attention-bf16-cpu-reference-residual-add-bf16"
        }
        (true, _, _, _) => "real-checkpoint-bf16-causal-attention-full-output-rows",
        (false, _, _, _) => "real-checkpoint-bf16-causal-attention-prefix",
    }
}

fn coordinator_linear_backend_family(backend: &'static str) -> &'static str {
    match backend {
        CUDA_REFERENCE_LINEAR_BF16_RESIDENT_WEIGHT_BACKEND
        | CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND => {
            CUDA_REFERENCE_LINEAR_BF16_BACKEND
        }
        other => other,
    }
}

fn attention_hidden_for_layer_from_initial(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden: Vec<f32>,
    device_hidden: Option<&DeviceBf16Output>,
    mode: LayerOrderedExecutionMode,
    prefix_context_rows: Vec<Vec<f32>>,
) -> Result<RealFullAttentionResidualPrefixHidden> {
    match mode.attention_backend {
        LayerOrderedAttentionBackend::CausalPrefix => {
            if !prefix_context_rows.is_empty() {
                anyhow::bail!(
                    "layer-ordered causal attention does not accept MLA/RoPE prefix context rows"
                );
            }
            if mode.attention_full_output {
                real_full_attention_residual_full_output_hidden_for_layer_from_initial(
                    catalog, layer_id, hidden,
                )
            } else {
                real_full_attention_residual_prefix_hidden_for_layer_from_initial(
                    catalog, layer_id, hidden,
                )
            }
        }
        LayerOrderedAttentionBackend::MlaRopePrefix => {
            if device_hidden.is_some() && !prefix_context_rows.is_empty() {
                anyhow::bail!(
                    "layer-ordered MLA/RoPE device hidden input is only supported with KV-cache prefix rows, not supplied host prefix rows"
                );
            }
            if mode.attention_full_output {
                if prefix_context_rows.is_empty() {
                    return real_full_mla_rope_attention_full_output_hidden_for_layer_from_initial(
                        catalog, layer_id, hidden,
                    );
                }
                return real_full_mla_rope_attention_full_output_prefix_context_hidden_for_layer_from_initial(
                    catalog,
                    layer_id,
                    prefix_context_rows,
                    hidden,
                );
            }
            if prefix_context_rows.is_empty() {
                return real_full_mla_rope_attention_prefix_hidden_for_layer_from_initial(
                    catalog, layer_id, hidden,
                );
            }
            real_full_mla_rope_attention_prefix_context_hidden_for_layer_from_initial(
                catalog,
                layer_id,
                prefix_context_rows,
                hidden,
            )
        }
    }
}

fn attention_hidden_for_layer_from_initial_with_scheduler_prefix(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden: Vec<f32>,
    device_hidden: Option<&DeviceBf16Output>,
    mode: LayerOrderedExecutionMode,
    scheduler_binding: &LayerOrderedSchedulerRowsBinding,
    prefix_context_hidden: Option<Vec<f32>>,
) -> Result<(RealFullAttentionResidualPrefixHidden, Option<Vec<f32>>)> {
    if mode.attention_backend != LayerOrderedAttentionBackend::MlaRopePrefix {
        let attention = attention_hidden_for_layer_from_initial(
            catalog,
            layer_id,
            hidden,
            device_hidden,
            mode,
            Vec::new(),
        )?;
        return Ok((attention, None));
    }

    let Some(prefix_hidden) = prefix_context_hidden else {
        let attention = attention_hidden_for_layer_from_initial(
            catalog,
            layer_id,
            hidden,
            device_hidden,
            mode,
            Vec::new(),
        )?;
        return Ok((attention, None));
    };

    let use_scheduler_prefix = scheduler_binding.probe.passed
        && scheduler_binding.probe.uses_live_scheduler_rows
        && scheduler_binding.layer_selected(layer_id);
    if !use_scheduler_prefix {
        let attention = attention_hidden_for_layer_from_initial(
            catalog,
            layer_id,
            hidden,
            device_hidden,
            mode,
            Vec::new(),
        )?;
        return Ok((attention, None));
    }

    let prefix_hidden_for_kv = prefix_hidden.clone();
    let prefix_attention = attention_hidden_for_layer_from_initial(
        catalog,
        layer_id,
        prefix_hidden,
        None,
        mode,
        Vec::new(),
    )?;
    let prefix_kv_blocks =
        layer_ordered_prefix_kv_cache_blocks_for_layer(catalog, layer_id, &prefix_hidden_for_kv)?;
    let attention = attention_hidden_for_layer_from_initial_with_kv_cache(
        catalog,
        layer_id,
        hidden,
        device_hidden,
        mode,
        prefix_kv_blocks,
    )?;
    Ok((attention, Some(prefix_attention.hidden)))
}

fn attention_hidden_for_layer_from_initial_with_kv_cache(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden: Vec<f32>,
    device_hidden: Option<&DeviceBf16Output>,
    mode: LayerOrderedExecutionMode,
    prefix_kv_blocks: Vec<RealFullMlaRopeKvCacheBlock>,
) -> Result<RealFullAttentionResidualPrefixHidden> {
    if mode.attention_backend != LayerOrderedAttentionBackend::MlaRopePrefix {
        anyhow::bail!("KV-cache prefix context is only supported for MLA/RoPE attention modes");
    }
    if mode.attention_full_output {
        if let Some(device_hidden) = device_hidden {
            return real_full_mla_rope_attention_full_output_kv_cache_context_hidden_for_layer_from_initial_device_input(
                catalog,
                layer_id,
                prefix_kv_blocks,
                hidden,
                device_hidden,
            );
        }
        return real_full_mla_rope_attention_full_output_kv_cache_context_hidden_for_layer_from_initial(
            catalog,
            layer_id,
            prefix_kv_blocks,
            hidden,
        );
    }
    if let Some(device_hidden) = device_hidden {
        return real_full_mla_rope_attention_kv_cache_context_hidden_for_layer_from_initial_device_input(
            catalog,
            layer_id,
            prefix_kv_blocks,
            hidden,
            device_hidden,
        );
    }
    real_full_mla_rope_attention_kv_cache_context_hidden_for_layer_from_initial(
        catalog,
        layer_id,
        prefix_kv_blocks,
        hidden,
    )
}

fn layer_ordered_prefix_kv_cache_blocks_for_layer(
    catalog: &TensorCatalog,
    layer_id: usize,
    prefix_hidden: &[f32],
) -> Result<Vec<RealFullMlaRopeKvCacheBlock>> {
    let config = KvCacheConfig::glm52_phase0(2);
    let layer = LayerId(layer_id as u32);
    let sequence_id = format!("layer-ordered-prefix-kv-layer-{layer_id}");
    let mut store = KvCacheBackingStore::new(config.clone());
    let reservation_id = store.reserve(sequence_id.clone(), 2)?;
    let descriptor = KvBlockDescriptor {
        reservation_id,
        sequence_id,
        layer_id: layer,
        token_start: PositionId(0),
        token_count: 1,
    };
    let block = real_full_mla_rope_kv_cache_block_for_layer_from_hidden(
        catalog,
        layer_id,
        0,
        prefix_hidden,
    )?;
    let expected_payload_bytes = config.layer_payload_bytes(layer, 1);
    if block.bytes.len() != expected_payload_bytes {
        anyhow::bail!(
            "layer-ordered prefix KV payload size mismatch for layer {layer_id}: expected {} got {}",
            expected_payload_bytes,
            block.bytes.len()
        );
    }
    store.write_committed_block(descriptor.clone(), block.bytes)?;
    let visible = store.read_visible_blocks_for_descriptor(&descriptor);
    if visible.len() != 1 {
        anyhow::bail!(
            "layer-ordered prefix KV expected one visible block for layer {layer_id}, got {}",
            visible.len()
        );
    }
    layer_ordered_visible_blocks_to_mla_rope_kv_blocks(&config, visible)
}

fn layer_ordered_visible_blocks_to_mla_rope_kv_blocks(
    config: &KvCacheConfig,
    visible: Vec<KvBackedBlock>,
) -> Result<Vec<RealFullMlaRopeKvCacheBlock>> {
    visible
        .into_iter()
        .map(|block| {
            let expected_payload_bytes = config
                .descriptor_payload_bytes(&block.descriptor)
                .context("validating layer-ordered visible KV descriptor payload bytes")?;
            if block.bytes.len() != expected_payload_bytes {
                anyhow::bail!(
                    "layer-ordered visible KV payload size mismatch for layer {} token_start {} token_count {}: expected {} got {}",
                    block.descriptor.layer_id.0,
                    block.descriptor.token_start.0,
                    block.descriptor.token_count,
                    expected_payload_bytes,
                    block.bytes.len()
                );
            }
            Ok(RealFullMlaRopeKvCacheBlock {
                token_start: usize::try_from(block.descriptor.token_start.0)
                    .context("layer-ordered visible KV token_start does not fit usize")?,
                token_count: block.descriptor.token_count,
                bytes: block.bytes,
            })
        })
        .collect()
}

fn initial_prefix_context_hidden(
    catalog: &TensorCatalog,
    mode: LayerOrderedExecutionMode,
) -> Result<Option<Vec<f32>>> {
    if mode.attention_backend != LayerOrderedAttentionBackend::MlaRopePrefix {
        return Ok(None);
    }
    if mode.initial_input.uses_embedding_residual_input {
        let current_token_id = mode
            .initial_input
            .token_id
            .unwrap_or(REAL_FULL_LAYER_ORDERED_DEFAULT_TOKEN_ID);
        let prefix_token_id = if current_token_id == 0 {
            1
        } else {
            current_token_id - 1
        };
        return Ok(Some(
            real_full_embedding_hidden_for_token(catalog, prefix_token_id)?.hidden,
        ));
    }
    Ok(Some(deterministic_prefix_context_hidden()))
}

fn deterministic_prefix_context_hidden() -> Vec<f32> {
    let mut hidden = deterministic_dense_hidden(GLM52_HIDDEN_SIZE);
    for (idx, value) in hidden.iter_mut().enumerate() {
        *value += ((idx % 17) as f32 - 8.0) / 4096.0;
    }
    hidden
}

fn dense_prefix_context_hidden_for_layer(
    catalog: &TensorCatalog,
    layer_id: usize,
    prefix_context_hidden: Option<Vec<f32>>,
    mode: LayerOrderedExecutionMode,
) -> Result<Option<Vec<f32>>> {
    let Some(hidden) = prefix_context_hidden else {
        return Ok(None);
    };
    if mode.attention_backend != LayerOrderedAttentionBackend::MlaRopePrefix {
        return Ok(None);
    }
    let dense = if mode.dense_full_output {
        real_full_dense_layer_full_output_hidden_from_initial(catalog, layer_id, hidden)?
    } else {
        real_full_dense_layer_prefix_hidden_from_initial(catalog, layer_id, hidden)?
    };
    Ok(Some(dense.hidden))
}

fn sparse_prefix_context_hidden_for_layer(
    catalog: &TensorCatalog,
    layer_id: usize,
    prefix_context_hidden: Option<Vec<f32>>,
    mode: LayerOrderedExecutionMode,
) -> Result<Option<Vec<f32>>> {
    let Some(hidden) = prefix_context_hidden else {
        return Ok(None);
    };
    if mode.attention_backend != LayerOrderedAttentionBackend::MlaRopePrefix {
        return Ok(None);
    }
    let sparse = if mode.sparse_full_output {
        real_sparse_mlp_shared_layer_full_output_hidden_from_initial(catalog, layer_id, hidden)?
    } else {
        real_sparse_mlp_shared_layer_hidden_from_initial(catalog, layer_id, hidden)?
    };
    Ok(Some(sparse.hidden))
}

fn sparse_step(
    sparse: &RealFullSparseMlpSharedLayerHidden,
    tensor_metadata: &TensorMetadataLookup<'_>,
) -> RealFullLayerOrderedResidualExecutionStep {
    debug_assert_eq!(sparse.routed_outputs.len(), sparse.output_rows);
    debug_assert_eq!(sparse.shared_outputs.len(), sparse.output_rows);
    debug_assert_eq!(sparse.layer_outputs.len(), sparse.output_rows);
    let stage_source = sparse_stage_source(sparse);
    RealFullLayerOrderedResidualExecutionStep {
        layer_id: sparse.layer_id,
        stage: "sparse_moe_mlp",
        stage_source,
        stage_status: sparse_stage_status(sparse),
        executed: true,
        residual_adds: sparse.residual_adds,
        output_rows: sparse.output_rows,
        routes_executed: sparse.route_count,
        selected_routes: sparse.routes.clone(),
        expert_host_batch_set: Some(sparse_host_batch_set_evidence(sparse)),
        includes_shared_expert: sparse.shared_expert_executed,
        tensor_artifacts: sparse_tensor_artifacts(tensor_metadata, sparse),
        residual_before_checksum: Some(sparse.layer_summary.residual_before_checksum),
        residual_delta_checksum: Some(sparse.layer_summary.residual_delta_checksum),
        residual_after_checksum: Some(sparse.layer_summary.residual_after_checksum),
        missing_reason: None,
    }
}

fn sparse_stage_status(sparse: &RealFullSparseMlpSharedLayerHidden) -> &'static str {
    if sparse.passed {
        "real"
    } else {
        "blocked"
    }
}

fn sparse_stage_source(sparse: &RealFullSparseMlpSharedLayerHidden) -> &'static str {
    match (
        sparse.output_rows == GLM52_HIDDEN_SIZE,
        sparse.expert_input_norm_backend,
        sparse.router_backend,
        sparse.shared_mlp_backend,
        sparse.residual_add_backend,
    ) {
        (
            true,
            CUDA_REFERENCE_RMSNORM_BF16_BACKEND
            | CUDA_REFERENCE_RMSNORM_BF16_RESIDENT_WEIGHT_BACKEND
            | CUDA_REFERENCE_RMSNORM_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
            CUDA_REFERENCE_ROUTER_TOPK_BF16_BACKEND
            | CUDA_REFERENCE_ROUTER_TOPK_BF16_RESIDENT_WEIGHT_BACKEND
            | CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
            | CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_BACKEND,
            CUDA_REFERENCE_SILU_GATED_MLP_BF16_BACKEND
            | CUDA_REFERENCE_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND
            | CUDA_REFERENCE_SILU_GATED_MLP_BF16_RESIDENT_WEIGHT_BACKEND,
            CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
        ) => {
            "real-checkpoint-nvfp4-routed-plus-bf16-shared-moe-full-output-cuda-reference-rmsnorm-bf16-cuda-reference-router-topk-bf16-cuda-reference-shared-silu-gated-mlp-bf16-cuda-reference-residual-add-bf16"
        }
        (
            false,
            CUDA_REFERENCE_RMSNORM_BF16_BACKEND
            | CUDA_REFERENCE_RMSNORM_BF16_RESIDENT_WEIGHT_BACKEND
            | CUDA_REFERENCE_RMSNORM_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
            CUDA_REFERENCE_ROUTER_TOPK_BF16_BACKEND
            | CUDA_REFERENCE_ROUTER_TOPK_BF16_RESIDENT_WEIGHT_BACKEND
            | CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
            | CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_BACKEND,
            CUDA_REFERENCE_SILU_GATED_MLP_BF16_BACKEND
            | CUDA_REFERENCE_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND
            | CUDA_REFERENCE_SILU_GATED_MLP_BF16_RESIDENT_WEIGHT_BACKEND,
            CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
        ) => {
            "real-checkpoint-nvfp4-routed-plus-bf16-shared-moe-prefix-cuda-reference-rmsnorm-bf16-cuda-reference-router-topk-bf16-cuda-reference-shared-silu-gated-mlp-bf16-cuda-reference-residual-add-bf16"
        }
        (
            true,
            CPU_REFERENCE_RMSNORM_BF16_BACKEND,
            CPU_REFERENCE_ROUTER_TOPK_BF16_BACKEND,
            CPU_REFERENCE_SILU_GATED_MLP_BF16_BACKEND,
            CPU_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
        ) => {
            "real-checkpoint-nvfp4-routed-plus-bf16-shared-moe-full-output-cpu-reference-rmsnorm-bf16-cpu-reference-router-topk-bf16-cpu-reference-shared-silu-gated-mlp-bf16-cpu-reference-residual-add-bf16"
        }
        (
            false,
            CPU_REFERENCE_RMSNORM_BF16_BACKEND,
            CPU_REFERENCE_ROUTER_TOPK_BF16_BACKEND,
            CPU_REFERENCE_SILU_GATED_MLP_BF16_BACKEND,
            CPU_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
        ) => {
            "real-checkpoint-nvfp4-routed-plus-bf16-shared-moe-prefix-cpu-reference-rmsnorm-bf16-cpu-reference-router-topk-bf16-cpu-reference-shared-silu-gated-mlp-bf16-cpu-reference-residual-add-bf16"
        }
        (true, _, _, _, _) => "real-checkpoint-nvfp4-routed-plus-bf16-shared-moe-full-output",
        (false, _, _, _, _) => "real-checkpoint-nvfp4-routed-plus-bf16-shared-moe-prefix",
    }
}

struct SparseSelectedRoutesHostBatchSet {
    batch: ExpertBatch,
    routes: Vec<ExpertBatchRoute>,
    set: ExpertHostBatchSet,
}

struct SparseProtocolV2RequestFraming {
    request_frames: usize,
    request_rows: usize,
    request_routes: usize,
    hidden_payload_bytes: usize,
    request_wire_bytes: usize,
    views_parse: bool,
    uses_compact_hidden_payloads: bool,
    row_gather_backend: &'static str,
    synthetic_response_frames: usize,
    synthetic_response_rows: usize,
    synthetic_response_wire_bytes: usize,
    row_scatter_backend: &'static str,
    accumulated_output_values: usize,
    accumulated_contribution_counts: Vec<usize>,
    accumulated_output_checksum: f64,
    synthetic_outputs_finite: bool,
}

#[derive(Clone, Copy)]
pub(in crate::commands::real_full::residual) struct SparseMoeProtocolV2DispatchKind {
    status: &'static str,
    source: &'static str,
    executor: &'static str,
    closes_live_expert_daemon_moe_gate: bool,
}

pub(in crate::commands::real_full::residual) struct SparseMoeProtocolV2ResidualStep {
    pub(in crate::commands::real_full::residual) hidden_after: Vec<f32>,
    pub(in crate::commands::real_full::residual) device_hidden: Option<DeviceBf16Output>,
    pub(in crate::commands::real_full::residual) host_batch_set_evidence:
        RealFullSparseMoeHostBatchSetEvidence,
    pub(in crate::commands::real_full::residual) dispatch_stats:
        TcpProtocolV2HostBatchSetDispatchStats,
    pub(in crate::commands::real_full::residual) routed_output_checksum: f64,
    pub(in crate::commands::real_full::residual) shared_output_checksum: f64,
    pub(in crate::commands::real_full::residual) residual_delta_checksum: f64,
    pub(in crate::commands::real_full::residual) residual_after_checksum: f64,
    pub(in crate::commands::real_full::residual) residual_add_backend: &'static str,
}

pub(in crate::commands::real_full::residual) fn real_expertd_sparse_moe_dispatch_kind(
) -> SparseMoeProtocolV2DispatchKind {
    SparseMoeProtocolV2DispatchKind {
        status: "protocol-v2-real-expertd-dispatch",
        source: "layer-ordered-real-router-selected-routes-live-expertd",
        executor: PROTOCOL_V2_REAL_NVFP4_CHECKPOINT_EXECUTOR,
        closes_live_expert_daemon_moe_gate: true,
    }
}

#[cfg(test)]
pub(in crate::commands::real_full::residual) fn synthetic_sparse_moe_dispatch_kind(
) -> SparseMoeProtocolV2DispatchKind {
    SparseMoeProtocolV2DispatchKind {
        status: "protocol-v2-synthetic-route-dispatch",
        source: "layer-ordered-real-router-selected-routes-synthetic-protocol-v2",
        executor: PROTOCOL_V2_SYNTHETIC_ROUTE_EXECUTOR,
        closes_live_expert_daemon_moe_gate: false,
    }
}

fn sparse_host_batch_set_evidence(
    sparse: &RealFullSparseMlpSharedLayerHidden,
) -> RealFullSparseMoeHostBatchSetEvidence {
    match build_sparse_selected_routes_host_batch_set(sparse) {
        Ok(plan) => {
            let framing = sparse_protocol_v2_request_framing(
                &plan.set,
                &sparse.expert_input_hidden_bf16_payload,
            );
            let touched_hosts = plan
                .set
                .touched_hosts()
                .map(str::to_owned)
                .collect::<Vec<_>>();
            match framing {
                Ok(framing) => RealFullSparseMoeHostBatchSetEvidence {
                    status: "protocol-v2-frames-from-selected-routes",
                    source: "layer-ordered-real-router-selected-routes",
                    layer_id: sparse.layer_id,
                    global_rows: plan.batch.num_rows(),
                    host_batches: plan.set.num_hosts(),
                    host_rows: plan.set.host_row_count(),
                    routes: plan.routes.len(),
                    hidden_dim: plan.batch.hidden_dim,
                    hidden_bytes_per_row: plan.batch.hidden_bytes_per_row,
                    hidden_dtype: plan.batch.hidden_dtype.clone(),
                    graph_bucket_rows: plan.batch.graph_bucket.row_capacity,
                    touched_hosts,
                    per_host_route_counts: plan
                        .set
                        .batches
                        .iter()
                        .map(|batch| batch.route_count())
                        .collect(),
                    reconstruction_global_rows: plan.set.reconstruction_plan.global_row_count,
                    reconstruction_host_maps: plan.set.reconstruction_plan.host_row_maps.len(),
                    protocol_v2_request_frames: framing.request_frames,
                    protocol_v2_request_rows: framing.request_rows,
                    protocol_v2_request_routes: framing.request_routes,
                    protocol_v2_hidden_payload_bytes: framing.hidden_payload_bytes,
                    protocol_v2_request_wire_bytes: framing.request_wire_bytes,
                    protocol_v2_views_parse: framing.views_parse,
                    protocol_v2_uses_compact_hidden_payloads: framing.uses_compact_hidden_payloads,
                    protocol_v2_row_gather_backend: framing.row_gather_backend,
                    protocol_v2_synthetic_executor: PROTOCOL_V2_SYNTHETIC_ROUTE_EXECUTOR,
                    protocol_v2_synthetic_response_frames: framing.synthetic_response_frames,
                    protocol_v2_synthetic_response_rows: framing.synthetic_response_rows,
                    protocol_v2_synthetic_response_wire_bytes: framing
                        .synthetic_response_wire_bytes,
                    protocol_v2_row_scatter_backend: framing.row_scatter_backend,
                    protocol_v2_accumulated_output_values: framing.accumulated_output_values,
                    protocol_v2_accumulated_contribution_counts: framing
                        .accumulated_contribution_counts,
                    protocol_v2_accumulated_output_checksum: Some(
                        framing.accumulated_output_checksum,
                    ),
                    protocol_v2_synthetic_outputs_finite: framing.synthetic_outputs_finite,
                    protocol_v2_executes_route_dependent_synthetic: true,
                    uses_expert_input_hidden_payload: true,
                    uses_selected_routes: true,
                    uses_route_owners: true,
                    closes_live_expert_daemon_moe_gate: false,
                    skipped_reason: None,
                },
                Err(error) => RealFullSparseMoeHostBatchSetEvidence {
                    status: "protocol-v2-frame-error",
                    source: "layer-ordered-real-router-selected-routes",
                    layer_id: sparse.layer_id,
                    global_rows: plan.batch.num_rows(),
                    host_batches: plan.set.num_hosts(),
                    host_rows: plan.set.host_row_count(),
                    routes: plan.routes.len(),
                    hidden_dim: plan.batch.hidden_dim,
                    hidden_bytes_per_row: plan.batch.hidden_bytes_per_row,
                    hidden_dtype: plan.batch.hidden_dtype.clone(),
                    graph_bucket_rows: plan.batch.graph_bucket.row_capacity,
                    touched_hosts,
                    per_host_route_counts: plan
                        .set
                        .batches
                        .iter()
                        .map(|batch| batch.route_count())
                        .collect(),
                    reconstruction_global_rows: plan.set.reconstruction_plan.global_row_count,
                    reconstruction_host_maps: plan.set.reconstruction_plan.host_row_maps.len(),
                    protocol_v2_request_frames: 0,
                    protocol_v2_request_rows: 0,
                    protocol_v2_request_routes: 0,
                    protocol_v2_hidden_payload_bytes: 0,
                    protocol_v2_request_wire_bytes: 0,
                    protocol_v2_views_parse: false,
                    protocol_v2_uses_compact_hidden_payloads: false,
                    protocol_v2_row_gather_backend: "not-run",
                    protocol_v2_synthetic_executor: PROTOCOL_V2_SYNTHETIC_ROUTE_EXECUTOR,
                    protocol_v2_synthetic_response_frames: 0,
                    protocol_v2_synthetic_response_rows: 0,
                    protocol_v2_synthetic_response_wire_bytes: 0,
                    protocol_v2_row_scatter_backend: "not-run",
                    protocol_v2_accumulated_output_values: 0,
                    protocol_v2_accumulated_contribution_counts: Vec::new(),
                    protocol_v2_accumulated_output_checksum: None,
                    protocol_v2_synthetic_outputs_finite: false,
                    protocol_v2_executes_route_dependent_synthetic: false,
                    uses_expert_input_hidden_payload: !sparse
                        .expert_input_hidden_bf16_payload
                        .is_empty(),
                    uses_selected_routes: true,
                    uses_route_owners: true,
                    closes_live_expert_daemon_moe_gate: false,
                    skipped_reason: Some(error.to_string()),
                },
            }
        }
        Err(error) => RealFullSparseMoeHostBatchSetEvidence {
            status: "error",
            source: "layer-ordered-real-router-selected-routes",
            layer_id: sparse.layer_id,
            global_rows: 0,
            host_batches: 0,
            host_rows: 0,
            routes: sparse.routes.len(),
            hidden_dim: GLM52_HIDDEN_SIZE,
            hidden_bytes_per_row: GLM52_HIDDEN_BF16_BYTES,
            hidden_dtype: DType::Bf16,
            graph_bucket_rows: 1,
            touched_hosts: Vec::new(),
            per_host_route_counts: Vec::new(),
            reconstruction_global_rows: 0,
            reconstruction_host_maps: 0,
            protocol_v2_request_frames: 0,
            protocol_v2_request_rows: 0,
            protocol_v2_request_routes: 0,
            protocol_v2_hidden_payload_bytes: 0,
            protocol_v2_request_wire_bytes: 0,
            protocol_v2_views_parse: false,
            protocol_v2_uses_compact_hidden_payloads: false,
            protocol_v2_row_gather_backend: "not-run",
            protocol_v2_synthetic_executor: PROTOCOL_V2_SYNTHETIC_ROUTE_EXECUTOR,
            protocol_v2_synthetic_response_frames: 0,
            protocol_v2_synthetic_response_rows: 0,
            protocol_v2_synthetic_response_wire_bytes: 0,
            protocol_v2_row_scatter_backend: "not-run",
            protocol_v2_accumulated_output_values: 0,
            protocol_v2_accumulated_contribution_counts: Vec::new(),
            protocol_v2_accumulated_output_checksum: None,
            protocol_v2_synthetic_outputs_finite: false,
            protocol_v2_executes_route_dependent_synthetic: false,
            uses_expert_input_hidden_payload: !sparse.expert_input_hidden_bf16_payload.is_empty(),
            uses_selected_routes: !sparse.routes.is_empty(),
            uses_route_owners: false,
            closes_live_expert_daemon_moe_gate: false,
            skipped_reason: Some(error.to_string()),
        },
    }
}

fn build_sparse_selected_routes_host_batch_set(
    sparse: &RealFullSparseMlpSharedLayerHidden,
) -> Result<SparseSelectedRoutesHostBatchSet> {
    build_sparse_selected_routes_host_batch_set_for_hidden_dim(sparse, GLM52_HIDDEN_SIZE)
}

fn build_sparse_selected_routes_host_batch_set_for_hidden_dim(
    sparse: &RealFullSparseMlpSharedLayerHidden,
    hidden_dim: usize,
) -> Result<SparseSelectedRoutesHostBatchSet> {
    if sparse.routes.is_empty() {
        anyhow::bail!("sparse MoE selected route metadata is empty");
    }
    if hidden_dim == 0 || hidden_dim > GLM52_HIDDEN_SIZE {
        anyhow::bail!(
            "sparse MoE selected route hidden_dim {hidden_dim} is outside 1..={GLM52_HIDDEN_SIZE}"
        );
    }
    let hidden_bytes_per_row = hidden_dim
        .checked_mul(2)
        .ok_or_else(|| anyhow::anyhow!("sparse MoE selected route hidden byte count overflow"))?;

    let routes = sparse
        .routes
        .iter()
        .map(|route| ExpertBatchRoute {
            row_index: 0,
            expert_id: route.expert_id,
            gate_weight: route.normalized_weight,
        })
        .collect::<Vec<_>>();
    let owner_lookup = ExpertOwnerLookup::from_pairs(
        sparse
            .routes
            .iter()
            .map(|route| ((sparse.layer_id, route.expert_id), route.owner.clone())),
    );
    let batch = ExpertBatch {
        layer_id: LayerId(sparse.layer_id as u32),
        placement_version: PlacementVersion("phase0-layer-ordered-selected-routes".to_owned()),
        hidden_dim,
        hidden_bytes_per_row,
        hidden_dtype: DType::Bf16,
        graph_bucket: GraphBucket::new(1),
        quantization_recipe: ModelFacts::default().quantization_recipe,
        rows: vec![ExpertBatchRow {
            row_id: 0,
            source_kind: RowSourceKind::PrefillChunk,
            request_id: RequestId("phase0-layer-ordered-sparse-row-0".to_owned()),
            sequence_id: "phase0-layer-ordered-sparse".to_owned(),
            token_position: PositionId(0),
            route_offset: 0,
            route_count: routes.len(),
        }],
    };
    let expert_hosts = EXPERT_HOSTS
        .iter()
        .map(|host| (*host).to_owned())
        .collect::<Vec<_>>();
    let set = ExpertHostBatchSet::from_expert_batch_with_owner_lookup(
        &batch,
        &routes,
        &expert_hosts,
        &owner_lookup,
    )?;
    set.reconstruction_plan
        .validate_for_batches(&set.batches, set.global_row_count)?;

    Ok(SparseSelectedRoutesHostBatchSet { batch, routes, set })
}

fn sparse_protocol_v2_request_framing(
    set: &ExpertHostBatchSet,
    global_hidden_payload: &[u8],
) -> Result<SparseProtocolV2RequestFraming> {
    let expected_payload = set
        .global_row_count
        .checked_mul(GLM52_HIDDEN_BF16_BYTES)
        .ok_or_else(|| anyhow::anyhow!("sparse ProtocolV2 hidden payload byte count overflow"))?;
    if global_hidden_payload.len() != expected_payload {
        anyhow::bail!(
            "sparse ProtocolV2 framing expected hidden payload bytes {expected_payload}, got {}",
            global_hidden_payload.len()
        );
    }

    let mut request_frames = 0_usize;
    let mut request_rows = 0_usize;
    let mut request_routes = 0_usize;
    let mut hidden_payload_bytes = 0_usize;
    let mut request_wire_bytes = 0_usize;
    let mut response_frames = 0_usize;
    let mut response_rows = 0_usize;
    let mut response_wire_bytes = 0_usize;
    let mut uses_compact_hidden_payloads = true;
    let mut row_gather_backend = None;
    let mut row_scatter_backend = None;
    let mut accumulated_output_values = vec![0.0_f32; set.global_row_count * GLM52_HIDDEN_SIZE];
    let mut accumulated_contribution_counts = vec![0_usize; set.global_row_count];
    let executor = SyntheticRouteExecutor;
    for (host_index, host_batch) in set.batches.iter().enumerate() {
        let host_row_indices = host_batch.global_row_indices().collect::<Vec<_>>();
        let gathered_hidden = gather_rows_bf16(
            global_hidden_payload,
            &host_row_indices,
            set.global_row_count,
            GLM52_HIDDEN_SIZE,
        )?;
        record_sparse_row_backend(&mut row_gather_backend, gathered_hidden.backend, "gather")?;
        let compact_hidden = gathered_hidden.bytes;
        uses_compact_hidden_payloads = uses_compact_hidden_payloads
            && compact_hidden.len() == host_batch.num_rows() * host_batch.hidden_bytes_per_row;
        let request = ExpertProtocolV2Request::from_expert_host_batch(
            498 + host_index as u64,
            host_batch,
            compact_hidden.clone(),
        )?;
        let encoded = request.encode()?;
        let view = ExpertProtocolV2RequestView::parse(&encoded)?;
        if view.hidden_payload() != compact_hidden.as_slice() {
            anyhow::bail!(
                "sparse ProtocolV2 request view hidden payload did not match compact host payload"
            );
        }
        let response = executor.execute(&view)?;
        response_frames += 1;
        response_rows += response.header.row_count as usize;
        response_wire_bytes += response.wire_stats().wire_bytes;
        let response_payload = response_bf16_payload(&response, host_batch.num_rows())?;
        let scatter = scatter_add_rows_bf16_to_f32(
            &response_payload,
            &host_row_indices,
            set.global_row_count,
            GLM52_HIDDEN_SIZE,
            Some(&accumulated_output_values),
        )?;
        record_sparse_row_backend(&mut row_scatter_backend, scatter.backend, "scatter")?;
        accumulated_output_values = scatter.values;
        for row_index in host_row_indices {
            accumulated_contribution_counts[row_index] += 1;
        }
        request_frames += 1;
        request_rows += request.rows.len();
        request_routes += request.routes.len();
        hidden_payload_bytes += request.hidden_payload.len();
        request_wire_bytes += request.wire_stats().wire_bytes;
    }
    for (row_index, count) in accumulated_contribution_counts.iter().enumerate() {
        if *count == 0 {
            anyhow::bail!("sparse ProtocolV2 row scatter did not receive a contribution for global row {row_index}");
        }
    }
    let accumulated_output_checksum = accumulated_output_values
        .iter()
        .map(|value| *value as f64)
        .sum::<f64>();
    let synthetic_outputs_finite = accumulated_output_values
        .iter()
        .all(|value| value.is_finite());
    let row_gather_backend = row_gather_backend.unwrap_or("not-run");
    let row_scatter_backend = row_scatter_backend.unwrap_or("not-run");

    Ok(SparseProtocolV2RequestFraming {
        request_frames,
        request_rows,
        request_routes,
        hidden_payload_bytes,
        request_wire_bytes,
        views_parse: true,
        uses_compact_hidden_payloads,
        row_gather_backend,
        synthetic_response_frames: response_frames,
        synthetic_response_rows: response_rows,
        synthetic_response_wire_bytes: response_wire_bytes,
        row_scatter_backend,
        accumulated_output_values: accumulated_output_values.len(),
        accumulated_contribution_counts,
        accumulated_output_checksum,
        synthetic_outputs_finite,
    })
}

pub(in crate::commands::real_full::residual) async fn sparse_moe_protocol_v2_residual_step(
    sparse: &RealFullSparseMlpSharedLayerHidden,
    residual_before_hidden: &[f32],
    targets: &[TcpProtocolV2HostBatchTarget],
    request_id_base: u64,
    config: TcpTransportConfig,
    kind: SparseMoeProtocolV2DispatchKind,
) -> Result<SparseMoeProtocolV2ResidualStep> {
    if residual_before_hidden.len() != GLM52_HIDDEN_SIZE {
        anyhow::bail!(
            "sparse ProtocolV2 residual step expected full hidden width {}, got {}",
            GLM52_HIDDEN_SIZE,
            residual_before_hidden.len()
        );
    }
    if sparse.output_rows == 0 || sparse.output_rows > GLM52_HIDDEN_SIZE {
        anyhow::bail!(
            "sparse ProtocolV2 residual step invalid output_rows={} for hidden width {}",
            sparse.output_rows,
            GLM52_HIDDEN_SIZE
        );
    }
    if sparse.shared_outputs.len() != sparse.output_rows {
        anyhow::bail!(
            "sparse ProtocolV2 residual step shared output width mismatch: expected {} got {}",
            sparse.output_rows,
            sparse.shared_outputs.len()
        );
    }

    let plan =
        build_sparse_selected_routes_host_batch_set_for_hidden_dim(sparse, sparse.output_rows)?;
    let hidden_payload = sparse_protocol_v2_hidden_payload(sparse)?;
    let dispatch = sparse_moe_protocol_v2_payload_dispatch(
        sparse,
        &plan,
        hidden_payload.as_ref(),
        targets,
        request_id_base,
        config,
    )
    .await?;
    if plan.set.global_row_count != 1 {
        anyhow::bail!(
            "sparse ProtocolV2 residual step currently supports one global row for Sparse-B residual graph execution, got {}",
            plan.set.global_row_count
        );
    }
    let residual_before = &residual_before_hidden[..sparse.output_rows];
    let global_row_indices_by_host = plan
        .set
        .reconstruction_plan
        .host_row_maps
        .iter()
        .map(|host_map| host_map.global_row_indices.as_slice())
        .collect::<Vec<_>>();
    let sparse_b_residual_after = sparse_b_scatter_residual_add_bf16(
        residual_before,
        &sparse.shared_outputs,
        &dispatch.partial_outputs_bf16_by_host,
        &global_row_indices_by_host,
        plan.set.global_row_count,
        sparse.output_rows,
    )?;
    let layer_outputs = sparse_b_residual_after.delta_values.as_slice();
    if layer_outputs.len() != sparse.output_rows {
        anyhow::bail!(
            "sparse ProtocolV2 residual step output width mismatch: expected {} got {}",
            sparse.output_rows,
            layer_outputs.len()
        );
    }
    if !layer_outputs.iter().all(|value| value.is_finite()) {
        anyhow::bail!(
            "sparse ProtocolV2 residual step produced non-finite routed+shared outputs for layer {}",
            sparse.layer_id
        );
    }
    let routed_accounting =
        sparse_protocol_v2_routed_output_accounting(layer_outputs, &sparse.shared_outputs)?;
    let residual_delta_checksum = checksum_f64(layer_outputs);
    let mut dispatch_stats = dispatch.stats;
    dispatch_stats.output_checksum = routed_accounting.checksum;
    validate_sparse_moe_protocol_v2_dispatch(sparse, &plan, &dispatch_stats)?;
    let executor_identity_matches =
        sparse_moe_protocol_v2_executor_identity_matches(&dispatch_stats, kind);
    if kind.closes_live_expert_daemon_moe_gate && !executor_identity_matches {
        anyhow::bail!(
            "live sparse MoE ProtocolV2 dispatch expected executor {} (id={}) from every host, got {:?}",
            kind.executor,
            expert_protocol_v2_compact_id(kind.executor),
            dispatch_stats.response_executor_ids
        );
    }
    let residual_after_checksum = checksum_f64(&sparse_b_residual_after.values);
    let residual_add_backend = sparse_b_residual_after.backend;
    let device_hidden = if sparse.output_rows == GLM52_HIDDEN_SIZE {
        sparse_b_residual_after.device_output
    } else {
        None
    };
    let hidden_after = if sparse.output_rows == GLM52_HIDDEN_SIZE {
        sparse_b_residual_after.values
    } else {
        let mut hidden_after = residual_before_hidden.to_vec();
        hidden_after[..sparse.output_rows].copy_from_slice(&sparse_b_residual_after.values);
        hidden_after
    };
    let shared_output_checksum = checksum_f64(&sparse.shared_outputs);
    let host_batch_set_evidence = sparse_moe_protocol_v2_dispatch_evidence(
        sparse,
        &plan,
        &dispatch_stats,
        hidden_payload.len(),
        routed_accounting.finite,
        kind,
        executor_identity_matches,
    );

    Ok(SparseMoeProtocolV2ResidualStep {
        hidden_after,
        device_hidden,
        host_batch_set_evidence,
        dispatch_stats,
        routed_output_checksum: routed_accounting.checksum,
        shared_output_checksum,
        residual_delta_checksum,
        residual_after_checksum,
        residual_add_backend,
    })
}

async fn sparse_moe_protocol_v2_payload_dispatch(
    sparse: &RealFullSparseMlpSharedLayerHidden,
    plan: &SparseSelectedRoutesHostBatchSet,
    hidden_payload: &[u8],
    targets: &[TcpProtocolV2HostBatchTarget],
    request_id_base: u64,
    config: TcpTransportConfig,
) -> Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch> {
    if sparse.output_rows != GLM52_HIDDEN_SIZE {
        return tcp_protocol_v2_host_batch_set_bf16_payload_dispatch(
            &plan.set,
            hidden_payload,
            targets,
            request_id_base,
            config,
        )
        .await;
    }

    let mut graph_pool = ExpertGraphInstancePool::new();
    graph_pool.register_glm52_bf16(
        plan.batch.layer_id,
        LayerWaveMode::Decode,
        GraphBucket::decode(),
        plan.batch.quantization_recipe.clone(),
        plan.set.num_hosts(),
    )?;
    tcp_protocol_v2_host_batch_set_bf16_payload_dispatch_with_graph_pool(
        &plan.set,
        hidden_payload,
        targets,
        request_id_base,
        config,
        &mut graph_pool,
    )
    .await
}

fn validate_sparse_moe_protocol_v2_dispatch(
    sparse: &RealFullSparseMlpSharedLayerHidden,
    plan: &SparseSelectedRoutesHostBatchSet,
    stats: &TcpProtocolV2HostBatchSetDispatchStats,
) -> Result<()> {
    if stats.global_rows != plan.set.global_row_count {
        anyhow::bail!(
            "sparse ProtocolV2 residual step global rows mismatch: expected {} got {}",
            plan.set.global_row_count,
            stats.global_rows
        );
    }
    if stats.hosts != plan.set.num_hosts() {
        anyhow::bail!(
            "sparse ProtocolV2 residual step host batch count mismatch: expected {} got {}",
            plan.set.num_hosts(),
            stats.hosts
        );
    }
    if stats.host_rows != plan.set.host_row_count() {
        anyhow::bail!(
            "sparse ProtocolV2 residual step host rows mismatch: expected {} got {}",
            plan.set.host_row_count(),
            stats.host_rows
        );
    }
    if stats.routes != plan.routes.len() || stats.routes != sparse.route_count {
        anyhow::bail!(
            "sparse ProtocolV2 residual step route count mismatch: plan={} sparse={} dispatch={}",
            plan.routes.len(),
            sparse.route_count,
            stats.routes
        );
    }
    if stats.output_dim != sparse.output_rows {
        anyhow::bail!(
            "sparse ProtocolV2 residual step output dim mismatch: expected {} got {}",
            sparse.output_rows,
            stats.output_dim
        );
    }
    if stats.output_values != plan.set.global_row_count * sparse.output_rows {
        anyhow::bail!(
            "sparse ProtocolV2 residual step output values mismatch: expected {} got {}",
            plan.set.global_row_count * sparse.output_rows,
            stats.output_values
        );
    }
    if stats.contribution_counts.len() != plan.set.global_row_count {
        anyhow::bail!(
            "sparse ProtocolV2 residual step contribution count length mismatch: expected {} got {}",
            plan.set.global_row_count,
            stats.contribution_counts.len()
        );
    }
    for (row_index, count) in stats.contribution_counts.iter().enumerate() {
        if *count == 0 {
            anyhow::bail!(
                "sparse ProtocolV2 residual step row {row_index} received no expert host contribution"
            );
        }
    }
    if stats.request_wire_bytes == 0 || stats.response_wire_bytes == 0 {
        anyhow::bail!("sparse ProtocolV2 residual step expected nonzero wire bytes");
    }
    if stats.response_executor_ids.len() != stats.hosts {
        anyhow::bail!(
            "sparse ProtocolV2 residual step executor-id count mismatch: expected {} got {}",
            stats.hosts,
            stats.response_executor_ids.len()
        );
    }
    if stats
        .response_executor_ids
        .iter()
        .any(|response_executor_id| *response_executor_id == 0)
    {
        anyhow::bail!("sparse ProtocolV2 residual step received an unstamped executor id");
    }
    if !stats.output_checksum.is_finite() {
        anyhow::bail!("sparse ProtocolV2 residual step output checksum is not finite");
    }
    Ok(())
}

fn sparse_moe_protocol_v2_executor_identity_matches(
    stats: &TcpProtocolV2HostBatchSetDispatchStats,
    kind: SparseMoeProtocolV2DispatchKind,
) -> bool {
    let expected_executor_id = expert_protocol_v2_compact_id(kind.executor);
    stats.response_executor_ids.len() == stats.hosts
        && stats
            .response_executor_ids
            .iter()
            .all(|response_executor_id| *response_executor_id == expected_executor_id)
}

fn sparse_moe_protocol_v2_dispatch_evidence(
    sparse: &RealFullSparseMlpSharedLayerHidden,
    plan: &SparseSelectedRoutesHostBatchSet,
    stats: &TcpProtocolV2HostBatchSetDispatchStats,
    hidden_payload_bytes: usize,
    routed_outputs_finite: bool,
    kind: SparseMoeProtocolV2DispatchKind,
    executor_identity_matches: bool,
) -> RealFullSparseMoeHostBatchSetEvidence {
    RealFullSparseMoeHostBatchSetEvidence {
        status: kind.status,
        source: kind.source,
        layer_id: sparse.layer_id,
        global_rows: plan.batch.num_rows(),
        host_batches: plan.set.num_hosts(),
        host_rows: plan.set.host_row_count(),
        routes: plan.routes.len(),
        hidden_dim: plan.batch.hidden_dim,
        hidden_bytes_per_row: plan.batch.hidden_bytes_per_row,
        hidden_dtype: plan.batch.hidden_dtype.clone(),
        graph_bucket_rows: plan.batch.graph_bucket.row_capacity,
        touched_hosts: plan.set.touched_hosts().map(str::to_owned).collect(),
        per_host_route_counts: plan
            .set
            .batches
            .iter()
            .map(|batch| batch.route_count())
            .collect(),
        reconstruction_global_rows: plan.set.reconstruction_plan.global_row_count,
        reconstruction_host_maps: plan.set.reconstruction_plan.host_row_maps.len(),
        protocol_v2_request_frames: stats.hosts,
        protocol_v2_request_rows: stats.host_rows,
        protocol_v2_request_routes: stats.routes,
        protocol_v2_hidden_payload_bytes: hidden_payload_bytes,
        protocol_v2_request_wire_bytes: stats.request_wire_bytes,
        protocol_v2_views_parse: true,
        protocol_v2_uses_compact_hidden_payloads: true,
        protocol_v2_row_gather_backend: PROTOCOL_V2_COMPACT_HIDDEN_GATHER_BACKEND,
        protocol_v2_synthetic_executor: kind.executor,
        protocol_v2_synthetic_response_frames: stats.hosts,
        protocol_v2_synthetic_response_rows: stats.host_rows,
        protocol_v2_synthetic_response_wire_bytes: stats.response_wire_bytes,
        protocol_v2_row_scatter_backend: PROTOCOL_V2_RECONSTRUCT_ACCUMULATE_BACKEND,
        protocol_v2_accumulated_output_values: stats.output_values,
        protocol_v2_accumulated_contribution_counts: stats.contribution_counts.clone(),
        protocol_v2_accumulated_output_checksum: Some(stats.output_checksum),
        protocol_v2_synthetic_outputs_finite: routed_outputs_finite,
        protocol_v2_executes_route_dependent_synthetic: false,
        uses_expert_input_hidden_payload: true,
        uses_selected_routes: true,
        uses_route_owners: true,
        closes_live_expert_daemon_moe_gate: kind.closes_live_expert_daemon_moe_gate
            && executor_identity_matches,
        skipped_reason: None,
    }
}

#[derive(Clone, Copy, Debug)]
struct SparseProtocolV2RoutedOutputAccounting {
    checksum: f64,
    finite: bool,
}

fn sparse_protocol_v2_routed_output_accounting(
    layer_outputs: &[f32],
    shared_outputs: &[f32],
) -> Result<SparseProtocolV2RoutedOutputAccounting> {
    if layer_outputs.len() != shared_outputs.len() {
        anyhow::bail!(
            "sparse ProtocolV2 routed output accounting length mismatch: layer_outputs={} shared_outputs={}",
            layer_outputs.len(),
            shared_outputs.len()
        );
    }
    let mut checksum = 0.0_f64;
    let mut finite = true;
    for (delta, shared) in layer_outputs.iter().zip(shared_outputs.iter()) {
        let routed = delta - shared;
        checksum += routed as f64;
        finite &= routed.is_finite();
    }
    Ok(SparseProtocolV2RoutedOutputAccounting { checksum, finite })
}

fn bounded_bf16_hidden_payload(full_payload: &[u8], hidden_dim: usize) -> Result<Vec<u8>> {
    let byte_len = hidden_dim
        .checked_mul(2)
        .ok_or_else(|| anyhow::anyhow!("bounded BF16 hidden payload byte count overflow"))?;
    let payload = full_payload.get(..byte_len).ok_or_else(|| {
        anyhow::anyhow!(
            "bounded BF16 hidden payload expected at least {byte_len} bytes, got {}",
            full_payload.len()
        )
    })?;
    Ok(payload.to_vec())
}

fn response_bf16_payload(
    response: &ExpertProtocolV2Response,
    expected_rows: usize,
) -> Result<Vec<u8>> {
    if response.header.status != ExpertProtocolV2Status::Ok {
        anyhow::bail!(
            "sparse ProtocolV2 response status {:?} is not ok",
            response.header.status
        );
    }
    if response.header.output_dtype != ExpertV2Dtype::Bf16 {
        anyhow::bail!(
            "sparse ProtocolV2 response dtype {:?} is not BF16",
            response.header.output_dtype
        );
    }
    if response.header.row_count as usize != expected_rows {
        anyhow::bail!(
            "sparse ProtocolV2 response row count {} did not match expected {expected_rows}",
            response.header.row_count
        );
    }

    let output_dim = response.header.output_dim as usize;
    if output_dim != GLM52_HIDDEN_SIZE {
        anyhow::bail!(
            "sparse ProtocolV2 response output dim {} did not match expected {}",
            output_dim,
            GLM52_HIDDEN_SIZE
        );
    }
    let logical_row_bytes = output_dim
        .checked_mul(2)
        .ok_or_else(|| anyhow::anyhow!("sparse ProtocolV2 BF16 output row byte overflow"))?;
    let stride = response.header.output_row_stride_bytes as usize;
    let mut payload = Vec::with_capacity(expected_rows * logical_row_bytes);
    for row_index in 0..expected_rows {
        let start = row_index
            .checked_mul(stride)
            .ok_or_else(|| anyhow::anyhow!("sparse ProtocolV2 response row offset overflow"))?;
        let end = start
            .checked_add(logical_row_bytes)
            .ok_or_else(|| anyhow::anyhow!("sparse ProtocolV2 response row end overflow"))?;
        let row = response
            .partial_output_payload
            .get(start..end)
            .ok_or_else(|| anyhow::anyhow!("sparse ProtocolV2 response row {row_index} missing"))?;
        payload.extend_from_slice(row);
    }
    Ok(payload)
}

fn sparse_protocol_v2_hidden_payload<'a>(
    sparse: &'a RealFullSparseMlpSharedLayerHidden,
) -> Result<Cow<'a, [u8]>> {
    if sparse.output_rows == GLM52_HIDDEN_SIZE {
        if sparse.expert_input_hidden_bf16_payload.len() != GLM52_HIDDEN_BF16_BYTES {
            anyhow::bail!(
                "sparse ProtocolV2 full-width hidden payload mismatch: expected {} bytes got {}",
                GLM52_HIDDEN_BF16_BYTES,
                sparse.expert_input_hidden_bf16_payload.len()
            );
        }
        return Ok(Cow::Borrowed(&sparse.expert_input_hidden_bf16_payload));
    }
    Ok(Cow::Owned(bounded_bf16_hidden_payload(
        &sparse.expert_input_hidden_bf16_payload,
        sparse.output_rows,
    )?))
}

fn record_sparse_row_backend(
    backend: &mut Option<&'static str>,
    candidate: &'static str,
    stage: &str,
) -> Result<()> {
    match backend {
        Some(existing) if *existing != candidate => {
            anyhow::bail!(
                "sparse ProtocolV2 row {stage} backend mismatch: {existing} vs {candidate}"
            )
        }
        Some(_) => Ok(()),
        None => {
            *backend = Some(candidate);
            Ok(())
        }
    }
}

fn dense_step(
    dense: &RealFullDenseLayerPrefixHidden,
    tensor_metadata: &TensorMetadataLookup<'_>,
) -> RealFullLayerOrderedResidualExecutionStep {
    let stage_source = dense_stage_source(dense);
    RealFullLayerOrderedResidualExecutionStep {
        layer_id: dense.layer_id,
        stage: "dense_mlp",
        stage_source,
        stage_status: "real",
        executed: true,
        residual_adds: dense.residual_adds,
        output_rows: dense.output_rows,
        routes_executed: 0,
        selected_routes: Vec::new(),
        expert_host_batch_set: None,
        includes_shared_expert: false,
        tensor_artifacts: dense_tensor_artifacts(tensor_metadata, dense),
        residual_before_checksum: Some(dense.initial_residual_checksum),
        residual_delta_checksum: Some(dense.residual_delta_checksum),
        residual_after_checksum: Some(dense.final_residual_checksum),
        missing_reason: None,
    }
}

fn dense_stage_source(dense: &RealFullDenseLayerPrefixHidden) -> &'static str {
    match (
        dense.output_rows == GLM52_HIDDEN_SIZE,
        dense.norm_backend,
        coordinator_linear_backend_family(dense.linear_backend),
        dense.mlp_backend,
        dense.residual_add_backend,
    ) {
        (
            true,
            CUDA_REFERENCE_RMSNORM_BF16_BACKEND
            | CUDA_REFERENCE_RMSNORM_BF16_RESIDENT_WEIGHT_BACKEND
            | CUDA_REFERENCE_RMSNORM_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
            CUDA_REFERENCE_LINEAR_BF16_BACKEND,
            CUDA_REFERENCE_SILU_GATED_MLP_BF16_BACKEND
            | CUDA_REFERENCE_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND
            | CUDA_REFERENCE_SILU_GATED_MLP_BF16_RESIDENT_WEIGHT_BACKEND,
            CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
        ) => {
            "real-checkpoint-bf16-dense-mlp-full-output-cuda-reference-rmsnorm-bf16-cuda-reference-linear-bf16-cuda-reference-silu-gated-mlp-bf16-cuda-reference-residual-add-bf16"
        }
        (
            false,
            CUDA_REFERENCE_RMSNORM_BF16_BACKEND
            | CUDA_REFERENCE_RMSNORM_BF16_RESIDENT_WEIGHT_BACKEND
            | CUDA_REFERENCE_RMSNORM_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
            CUDA_REFERENCE_LINEAR_BF16_BACKEND,
            CUDA_REFERENCE_SILU_GATED_MLP_BF16_BACKEND
            | CUDA_REFERENCE_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND
            | CUDA_REFERENCE_SILU_GATED_MLP_BF16_RESIDENT_WEIGHT_BACKEND,
            CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
        ) => {
            "real-checkpoint-bf16-dense-mlp-prefix-cuda-reference-rmsnorm-bf16-cuda-reference-linear-bf16-cuda-reference-silu-gated-mlp-bf16-cuda-reference-residual-add-bf16"
        }
        (
            true,
            CPU_REFERENCE_RMSNORM_BF16_BACKEND,
            CPU_REFERENCE_LINEAR_BF16_BACKEND,
            CPU_REFERENCE_SILU_GATED_MLP_BF16_BACKEND,
            CPU_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
        ) => {
            "real-checkpoint-bf16-dense-mlp-full-output-cpu-reference-rmsnorm-bf16-cpu-reference-linear-bf16-cpu-reference-silu-gated-mlp-bf16-cpu-reference-residual-add-bf16"
        }
        (
            false,
            CPU_REFERENCE_RMSNORM_BF16_BACKEND,
            CPU_REFERENCE_LINEAR_BF16_BACKEND,
            CPU_REFERENCE_SILU_GATED_MLP_BF16_BACKEND,
            CPU_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
        ) => {
            "real-checkpoint-bf16-dense-mlp-prefix-cpu-reference-rmsnorm-bf16-cpu-reference-linear-bf16-cpu-reference-silu-gated-mlp-bf16-cpu-reference-residual-add-bf16"
        }
        (true, _, _, _, _) => "real-checkpoint-bf16-dense-mlp-full-output",
        (false, _, _, _, _) => "real-checkpoint-bf16-dense-mlp-prefix",
    }
}

struct TensorMetadataLookup<'a> {
    tensors_by_name: BTreeMap<&'a str, &'a TensorInfo>,
}

impl<'a> TensorMetadataLookup<'a> {
    fn new(catalog: &'a TensorCatalog) -> Self {
        let tensors_by_name = catalog
            .tensors
            .iter()
            .map(|tensor| (tensor.name.as_str(), tensor))
            .collect::<BTreeMap<_, _>>();
        Self { tensors_by_name }
    }

    fn artifact(
        &self,
        name: String,
        rows_loaded: Option<usize>,
        source: &'static str,
    ) -> RealFullResidualExecutionTensorArtifact {
        let info = self
            .tensors_by_name
            .get(name.as_str())
            .unwrap_or_else(|| panic!("missing real execution tensor metadata for {name}"));
        RealFullResidualExecutionTensorArtifact {
            name,
            dtype: info.dtype.clone(),
            role: info.role.clone(),
            shape: info.shape.clone(),
            rank: info.shape.len(),
            byte_length: info.byte_length,
            is_quantization_metadata: info.is_quantization_metadata,
            rows_loaded,
            full_tensor_loaded: rows_loaded.is_none(),
            source,
        }
    }
}

fn attention_tensor_artifacts(
    tensor_metadata: &TensorMetadataLookup<'_>,
    layer_id: usize,
    output_rows: usize,
) -> Vec<RealFullResidualExecutionTensorArtifact> {
    let full = "attention-full-tensor";
    let prefix = "attention-prefix-rows";
    let mut artifacts = vec![
        tensor_metadata.artifact(
            format!("model.layers.{layer_id}.input_layernorm.weight"),
            None,
            full,
        ),
        tensor_metadata.artifact(
            format!("model.layers.{layer_id}.self_attn.q_a_proj.weight"),
            None,
            full,
        ),
        tensor_metadata.artifact(
            format!("model.layers.{layer_id}.self_attn.q_a_layernorm.weight"),
            None,
            full,
        ),
        tensor_metadata.artifact(
            format!("model.layers.{layer_id}.self_attn.q_b_proj.weight"),
            Some(output_rows),
            prefix,
        ),
        tensor_metadata.artifact(
            format!("model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.weight"),
            None,
            full,
        ),
        tensor_metadata.artifact(
            format!("model.layers.{layer_id}.self_attn.kv_a_layernorm.weight"),
            None,
            full,
        ),
        tensor_metadata.artifact(
            format!("model.layers.{layer_id}.self_attn.kv_b_proj.weight"),
            Some(output_rows),
            prefix,
        ),
        tensor_metadata.artifact(
            format!("model.layers.{layer_id}.self_attn.o_proj.weight"),
            Some(output_rows),
            prefix,
        ),
    ];
    if GLM52_DSA_INDEXER_LAYER_IDS.contains(&layer_id) {
        artifacts.extend(dsa_indexer_tensor_artifacts(tensor_metadata, layer_id));
    }
    artifacts
}

fn dsa_indexer_tensor_artifacts(
    tensor_metadata: &TensorMetadataLookup<'_>,
    layer_id: usize,
) -> Vec<RealFullResidualExecutionTensorArtifact> {
    let full = "dsa-indexer-probe-full-tensor";
    vec![
        tensor_metadata.artifact(
            format!("model.layers.{layer_id}.self_attn.indexer.k_norm.bias"),
            None,
            full,
        ),
        tensor_metadata.artifact(
            format!("model.layers.{layer_id}.self_attn.indexer.k_norm.weight"),
            None,
            full,
        ),
        tensor_metadata.artifact(
            format!("model.layers.{layer_id}.self_attn.indexer.weights_proj.weight"),
            None,
            full,
        ),
        tensor_metadata.artifact(
            format!("model.layers.{layer_id}.self_attn.indexer.wk.weight"),
            None,
            full,
        ),
        tensor_metadata.artifact(
            format!("model.layers.{layer_id}.self_attn.indexer.wq_b.weight"),
            Some(GLM52_DSA_INDEX_HEAD_DIM),
            "dsa-indexer-probe-prefix-rows",
        ),
    ]
}

fn dense_tensor_artifacts(
    tensor_metadata: &TensorMetadataLookup<'_>,
    dense: &RealFullDenseLayerPrefixHidden,
) -> Vec<RealFullResidualExecutionTensorArtifact> {
    let layer_id = dense.layer_id;
    vec![
        tensor_metadata.artifact(
            format!("model.layers.{layer_id}.post_attention_layernorm.weight"),
            None,
            "dense-full-tensor",
        ),
        tensor_metadata.artifact(
            format!("model.layers.{layer_id}.mlp.gate_proj.weight"),
            Some(dense.intermediate_rows),
            "dense-prefix-rows",
        ),
        tensor_metadata.artifact(
            format!("model.layers.{layer_id}.mlp.up_proj.weight"),
            Some(dense.intermediate_rows),
            "dense-prefix-rows",
        ),
        tensor_metadata.artifact(
            format!("model.layers.{layer_id}.mlp.down_proj.weight"),
            Some(dense.output_rows),
            "dense-prefix-rows",
        ),
    ]
}

fn sparse_tensor_artifacts(
    tensor_metadata: &TensorMetadataLookup<'_>,
    sparse: &RealFullSparseMlpSharedLayerHidden,
) -> Vec<RealFullResidualExecutionTensorArtifact> {
    let layer_id = sparse.layer_id;
    let selected_expert_ids = if sparse.routes.is_empty() {
        vec![sparse.layer_summary.expert_id]
    } else {
        sparse
            .routes
            .iter()
            .map(|route| route.expert_id)
            .collect::<Vec<_>>()
    };
    let mut artifacts = vec![
        tensor_metadata.artifact(
            format!("model.layers.{layer_id}.mlp.gate.weight"),
            None,
            "sparse-router-full-tensor",
        ),
        tensor_metadata.artifact(
            format!("model.layers.{layer_id}.mlp.gate.e_score_correction_bias"),
            None,
            "sparse-router-full-tensor",
        ),
    ];
    for expert_id in selected_expert_ids {
        for projection in ["gate_proj", "up_proj"] {
            push_sparse_projection_artifacts(
                &mut artifacts,
                tensor_metadata,
                layer_id,
                expert_id,
                projection,
                sparse.routed_intermediate_rows,
            );
        }
        push_sparse_projection_artifacts(
            &mut artifacts,
            tensor_metadata,
            layer_id,
            expert_id,
            "down_proj",
            sparse.output_rows,
        );
    }
    for projection in ["gate_proj", "up_proj"] {
        artifacts.push(tensor_metadata.artifact(
            format!("model.layers.{layer_id}.mlp.shared_experts.{projection}.weight"),
            Some(sparse.shared_intermediate_rows),
            "sparse-shared-prefix-rows",
        ));
    }
    artifacts.push(tensor_metadata.artifact(
        format!("model.layers.{layer_id}.mlp.shared_experts.down_proj.weight"),
        Some(sparse.output_rows),
        "sparse-shared-prefix-rows",
    ));
    artifacts
}

fn push_sparse_projection_artifacts(
    artifacts: &mut Vec<RealFullResidualExecutionTensorArtifact>,
    tensor_metadata: &TensorMetadataLookup<'_>,
    layer_id: usize,
    expert_id: usize,
    projection: &str,
    rows_loaded: usize,
) {
    let base_name = format!("model.layers.{layer_id}.mlp.experts.{expert_id}.{projection}");
    artifacts.push(tensor_metadata.artifact(
        format!("{base_name}.weight"),
        Some(rows_loaded),
        "sparse-routed-prefix-rows",
    ));
    artifacts.push(tensor_metadata.artifact(
        format!("{base_name}.weight_scale"),
        Some(rows_loaded),
        "sparse-routed-prefix-rows",
    ));
    artifacts.push(tensor_metadata.artifact(
        format!("{base_name}.weight_scale_2"),
        None,
        "sparse-routed-full-metadata",
    ));
}

fn approx_eq_f64(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1.0e-9
}

#[cfg(test)]
mod tests {
    use crate::cli::ExpertDaemonArgs;
    use crate::commands::expertd::run_expertd;
    use crate::commands::real_full::coordinator_kernels::{
        coordinator_cuda_graph_test_stats, coordinator_cuda_reference_kernels_enabled,
        cuda_native_library, cuda_reference_kernels_test_override,
        device_bf16_output_from_f32_values, preload_resident_weight_from_host_staging,
    };
    use std::fs::File;
    use std::net::{SocketAddr, TcpListener as StdTcpListener};
    use std::path::{Path, PathBuf};

    use crate::commands::real_full::types::{
        RealFullExpertSparseMlpSharedChainLayerProbe, RealFullSparseMoeRoute,
    };
    use glmrt_core::{
        DType, KvBackedBlock, KvBlockDescriptor, KvCacheConfig, KvWriteState, LayerId, ModelFacts,
        PositionId, TensorCatalog, TensorInfo, TensorRole, EXPERT_HOSTS, GLM52_DSA_INDEXER_LAYERS,
        GLM52_DSA_INDEXER_LAYER_IDS, GLM52_DSA_INDEX_HEAD_DIM, GLM52_FIRST_K_DENSE_REPLACE,
        GLM52_HIDDEN_BF16_BYTES, GLM52_HIDDEN_SIZE, GLM52_MLA_KV_LORA_RANK,
        GLM52_MLA_QK_ROPE_HEAD_DIM, GLM52_NUM_HIDDEN_LAYERS, GLM52_TOP_K,
    };

    use super::{
        approx_eq_f64, attention_hidden_for_layer_from_initial, attention_stage_source,
        bounded_bf16_hidden_payload, build_sparse_selected_routes_host_batch_set,
        build_sparse_selected_routes_host_batch_set_for_hidden_dim, dense_stage_source,
        layer_ordered_execution_mode, layer_ordered_execution_mode_with_input,
        layer_ordered_lm_head_sampling_from_score, layer_ordered_lm_head_sampling_probe,
        layer_ordered_visible_blocks_to_mla_rope_kv_blocks, real_expertd_sparse_moe_dispatch_kind,
        real_full_attention_residual_full_output_hidden,
        real_full_dense_layer_full_output_hidden_from_initial,
        real_full_mla_rope_attention_full_output_hidden_for_layer_from_initial,
        real_full_mla_rope_attention_prefix_hidden_for_layer_from_initial,
        real_sparse_mlp_shared_layer_full_output_hidden_from_initial,
        real_sparse_mlp_shared_layer_hidden_from_initial,
        run_real_full_layer_ordered_execution_probe_with_mode, sparse_host_batch_set_evidence,
        sparse_moe_protocol_v2_residual_step, sparse_protocol_v2_hidden_payload,
        sparse_protocol_v2_routed_output_accounting, sparse_stage_source, sparse_stage_status,
        sparse_tensor_artifacts, synthetic_sparse_moe_dispatch_kind,
        terminal_lm_head_sampling_satisfies_completion_gate, LayerOrderedAttentionBackend,
        MlaDsaAttentionCompletionTracker, RealFullResidualExecutionStepper,
        RealFullSparseMlpSharedLayerHidden, TensorMetadataLookup,
        CUDA_REFERENCE_CAUSAL_ATTENTION_BF16_BACKEND,
        CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
        CUDA_REFERENCE_LINEAR_BF16_RESIDENT_WEIGHT_BACKEND,
        CUDA_REFERENCE_LM_HEAD_ARGMAX_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
        CUDA_REFERENCE_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
        CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
        CUDA_REFERENCE_RMSNORM_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
        CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_BACKEND,
        CUDA_REFERENCE_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND,
        PROTOCOL_V2_REAL_NVFP4_CHECKPOINT_EXECUTOR, PROTOCOL_V2_SYNTHETIC_ROUTE_EXECUTOR,
    };
    use crate::commands::real_full::attention::RealFullAttentionResidualPrefixHidden;
    use crate::commands::real_full::dense::math::{
        bf16_bytes_from_f32, checksum_f64, deterministic_dense_hidden,
    };
    use crate::commands::real_full::dense::RealFullDenseLayerPrefixHidden;
    use crate::commands::real_full::sampling::RealLmHeadChunkScoreForHidden;
    use glmrt_transport::{
        expert_protocol_v2_compact_id, serve_protocol_v2_tcp_listener_with_executor,
        tcp_protocol_v2_host_batch_set_bf16_dispatch, SyntheticRouteExecutor,
        TcpProtocolV2HostBatchTarget, TcpTransportConfig,
    };
    use std::sync::Arc;
    use tokio::net::{TcpListener, TcpStream};
    use tokio::time::{sleep, Duration};

    #[test]
    fn layer_ordered_visible_kv_blocks_convert_without_device_roundtrip() {
        let config = KvCacheConfig::glm52_phase0(4);
        let descriptor = KvBlockDescriptor {
            reservation_id: 7,
            sequence_id: "seq-layer-ordered-visible-kv".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(2),
            token_count: 1,
        };
        let payload = vec![
            0x5a_u8;
            config
                .descriptor_payload_bytes(&descriptor)
                .expect("descriptor payload bytes")
        ];
        assert_eq!(
            payload.len(),
            (GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM) * 2
        );
        let visible = KvBackedBlock {
            write_id: 42,
            descriptor,
            state: KvWriteState::Committed,
            bytes: payload.clone(),
        };

        let blocks = layer_ordered_visible_blocks_to_mla_rope_kv_blocks(&config, vec![visible])
            .expect("converting visible KV blocks");

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].token_start, 2);
        assert_eq!(blocks[0].token_count, 1);
        assert_eq!(blocks[0].bytes, payload);
    }

    #[test]
    fn layer_ordered_visible_kv_blocks_reject_payload_mismatch() {
        let config = KvCacheConfig::glm52_phase0(4);
        let descriptor = KvBlockDescriptor {
            reservation_id: 7,
            sequence_id: "seq-layer-ordered-visible-kv-mismatch".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(2),
            token_count: 1,
        };
        let visible = KvBackedBlock {
            write_id: 42,
            descriptor,
            state: KvWriteState::Committed,
            bytes: vec![0x5a_u8; 16],
        };

        let err = match layer_ordered_visible_blocks_to_mla_rope_kv_blocks(&config, vec![visible]) {
            Ok(_) => panic!("mismatched visible KV payload should be rejected"),
            Err(error) => error,
        };

        assert!(err
            .to_string()
            .contains("layer-ordered visible KV payload size mismatch"));
    }

    #[test]
    fn layer_ordered_execution_mode_parses_bounded_and_full_output_modes() {
        let default_mode = layer_ordered_execution_mode(None);
        assert_eq!(default_mode.row_mode, "bounded");
        assert!(matches!(
            default_mode.attention_backend,
            LayerOrderedAttentionBackend::CausalPrefix
        ));
        assert!(!default_mode.attention_full_output);
        assert!(!default_mode.dense_full_output);
        assert!(!default_mode.sparse_full_output);
        assert!(!default_mode.lm_head_full_vocab);
        assert!(default_mode.initial_input.uses_embedding_residual_input);
        assert_eq!(default_mode.initial_input.token_id, Some(0));

        let bounded_mode = layer_ordered_execution_mode(Some("bounded"));
        assert_eq!(bounded_mode.row_mode, "bounded");
        assert!(matches!(
            bounded_mode.attention_backend,
            LayerOrderedAttentionBackend::CausalPrefix
        ));
        assert!(!bounded_mode.attention_full_output);
        assert!(!bounded_mode.dense_full_output);
        assert!(!bounded_mode.sparse_full_output);
        assert!(!bounded_mode.lm_head_full_vocab);
        assert!(bounded_mode.initial_input.uses_embedding_residual_input);

        for value in ["full-output-mlp", "mlp-full-output", "full-output"] {
            let full_output_mlp_mode = layer_ordered_execution_mode(Some(value));
            assert_eq!(full_output_mlp_mode.row_mode, "full-output-mlp");
            assert!(matches!(
                full_output_mlp_mode.attention_backend,
                LayerOrderedAttentionBackend::CausalPrefix
            ));
            assert!(!full_output_mlp_mode.attention_full_output);
            assert!(full_output_mlp_mode.dense_full_output);
            assert!(full_output_mlp_mode.sparse_full_output);
            assert!(!full_output_mlp_mode.lm_head_full_vocab);
        }

        for value in [
            "full-output-attention-mlp",
            "full-output-all",
            "all-full-output",
            "full-output-residual",
        ] {
            let full_output_all_mode = layer_ordered_execution_mode(Some(value));
            assert_eq!(full_output_all_mode.row_mode, "full-output-attention-mlp");
            assert!(matches!(
                full_output_all_mode.attention_backend,
                LayerOrderedAttentionBackend::CausalPrefix
            ));
            assert!(full_output_all_mode.attention_full_output);
            assert!(full_output_all_mode.dense_full_output);
            assert!(full_output_all_mode.sparse_full_output);
            assert!(!full_output_all_mode.lm_head_full_vocab);
        }

        for value in [
            "full-output-attention-mlp-full-vocab",
            "full-output-all-full-vocab",
            "all-full-output-full-vocab",
            "full-output-residual-full-vocab",
            "full-vocab",
        ] {
            let full_vocab_mode = layer_ordered_execution_mode(Some(value));
            assert_eq!(full_vocab_mode.row_mode, "full-output-attention-mlp");
            assert!(matches!(
                full_vocab_mode.attention_backend,
                LayerOrderedAttentionBackend::CausalPrefix
            ));
            assert!(full_vocab_mode.attention_full_output);
            assert!(full_vocab_mode.dense_full_output);
            assert!(full_vocab_mode.sparse_full_output);
            assert!(full_vocab_mode.lm_head_full_vocab);
        }

        for value in ["mla-rope", "bounded-mla-rope", "mla-rope-attention"] {
            let mla_rope_mode = layer_ordered_execution_mode(Some(value));
            assert_eq!(mla_rope_mode.row_mode, "mla-rope-attention");
            assert!(matches!(
                mla_rope_mode.attention_backend,
                LayerOrderedAttentionBackend::MlaRopePrefix
            ));
            assert!(!mla_rope_mode.attention_full_output);
            assert!(!mla_rope_mode.dense_full_output);
            assert!(!mla_rope_mode.sparse_full_output);
            assert!(!mla_rope_mode.lm_head_full_vocab);
        }

        for value in [
            "full-output-mla-rope-attention-mlp",
            "full-output-mla-rope-all",
            "mla-rope-full-output-all",
        ] {
            let mla_rope_full_output_all_mode = layer_ordered_execution_mode(Some(value));
            assert_eq!(
                mla_rope_full_output_all_mode.row_mode,
                "full-output-mla-rope-attention-mlp"
            );
            assert!(matches!(
                mla_rope_full_output_all_mode.attention_backend,
                LayerOrderedAttentionBackend::MlaRopePrefix
            ));
            assert!(mla_rope_full_output_all_mode.attention_full_output);
            assert!(mla_rope_full_output_all_mode.dense_full_output);
            assert!(mla_rope_full_output_all_mode.sparse_full_output);
            assert!(!mla_rope_full_output_all_mode.lm_head_full_vocab);
        }

        for value in [
            "full-output-mla-rope-attention-mlp-full-vocab",
            "full-output-mla-rope-all-full-vocab",
            "mla-rope-full-output-all-full-vocab",
        ] {
            let mla_rope_full_vocab_mode = layer_ordered_execution_mode(Some(value));
            assert_eq!(
                mla_rope_full_vocab_mode.row_mode,
                "full-output-mla-rope-attention-mlp"
            );
            assert!(matches!(
                mla_rope_full_vocab_mode.attention_backend,
                LayerOrderedAttentionBackend::MlaRopePrefix
            ));
            assert!(mla_rope_full_vocab_mode.attention_full_output);
            assert!(mla_rope_full_vocab_mode.dense_full_output);
            assert!(mla_rope_full_vocab_mode.sparse_full_output);
            assert!(mla_rope_full_vocab_mode.lm_head_full_vocab);
        }

        for value in [
            "mla-rope-full-output",
            "full-output-mla-rope",
            "full-output-mla-rope-attention",
        ] {
            let mla_rope_full_output_mode = layer_ordered_execution_mode(Some(value));
            assert_eq!(
                mla_rope_full_output_mode.row_mode,
                "full-output-mla-rope-attention"
            );
            assert!(matches!(
                mla_rope_full_output_mode.attention_backend,
                LayerOrderedAttentionBackend::MlaRopePrefix
            ));
            assert!(mla_rope_full_output_mode.attention_full_output);
            assert!(!mla_rope_full_output_mode.dense_full_output);
            assert!(!mla_rope_full_output_mode.sparse_full_output);
            assert!(!mla_rope_full_output_mode.lm_head_full_vocab);
        }

        let deterministic_mode = layer_ordered_execution_mode_with_input(
            Some("bounded"),
            Some("deterministic"),
            Some("17"),
        );
        assert_eq!(deterministic_mode.row_mode, "bounded");
        assert!(
            !deterministic_mode
                .initial_input
                .uses_embedding_residual_input
        );
        assert_eq!(deterministic_mode.initial_input.token_id, None);

        let embedding_mode =
            layer_ordered_execution_mode_with_input(Some("bounded"), Some("embedding"), Some("17"));
        assert!(embedding_mode.initial_input.uses_embedding_residual_input);
        assert_eq!(embedding_mode.initial_input.token_id, Some(17));
    }

    #[test]
    fn layer_ordered_lm_head_sampling_labels_full_vocab_scores() {
        let probe = layer_ordered_lm_head_sampling_from_score(
            "test-layer-ordered-hidden",
            RealLmHeadChunkScoreForHidden {
                lm_head_tensor: "lm_head.weight".to_owned(),
                hidden_dim: GLM52_HIDDEN_SIZE,
                vocab_size: 5,
                start_token_id: 0,
                chunk_rows: 2,
                rows_scored: 5,
                chunks_scored: 3,
                lm_head_bytes_read: (5 * GLM52_HIDDEN_SIZE * 2) as u64,
                hidden_values: GLM52_HIDDEN_SIZE,
                logits_evaluated: 5,
                multiply_accumulate_ops: 5 * GLM52_HIDDEN_SIZE as u64,
                covers_full_vocabulary: true,
                logits_kernel_backend:
                    CUDA_REFERENCE_LM_HEAD_ARGMAX_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
                argmax_kernel_backend:
                    CUDA_REFERENCE_LM_HEAD_ARGMAX_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
                sampler_kernel_backend:
                    CUDA_REFERENCE_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
                top_token_id: 4,
                top_logit: 3.0,
                sampled_token_id: 3,
                sampled_score: 0.42,
                sample_random_uniform: 0.5,
                sample_temperature: 0.7,
                sample_top_k: 8,
                sample_top_p: 0.95,
            },
            Some(0.25),
        );

        assert_eq!(
            probe.status,
            "numeric-real-layer-ordered-lm-head-full-vocab"
        );
        assert!(probe.passed);
        assert!(probe.uses_real_lm_head);
        assert!(probe.uses_layer_ordered_residual);
        assert!(probe.uses_layer_ordered_full_output_residual);
        assert!(probe.covers_full_vocabulary);
        assert_eq!(probe.rows_scored, 5);
        assert_eq!(probe.chunks_scored, 3);
        assert_eq!(probe.top_token_id, Some(4));
        assert_eq!(probe.top_logit, Some(3.0));
        assert_eq!(
            probe.logits_kernel_backend,
            Some(CUDA_REFERENCE_LM_HEAD_ARGMAX_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND)
        );
        assert_eq!(
            probe.argmax_kernel_backend,
            Some(CUDA_REFERENCE_LM_HEAD_ARGMAX_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND)
        );
        assert_eq!(
            probe.sampler_kernel_backend,
            Some(CUDA_REFERENCE_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND)
        );
        assert_eq!(probe.sampled_token_id, Some(3));
        assert_eq!(probe.sampled_score, Some(0.42));
        assert_eq!(probe.sample_top_k, Some(8));
        assert_eq!(probe.sample_top_p, Some(0.95));
        assert_eq!(probe.sample_temperature, Some(0.7));
        assert_eq!(probe.residual_after_checksum, Some(0.25));
        assert!(terminal_lm_head_sampling_satisfies_completion_gate(&probe));
    }

    #[test]
    fn layer_ordered_lm_head_sampling_prefers_final_device_hidden_when_available() {
        if !coordinator_cuda_reference_kernels_enabled() {
            return;
        }
        let weight_name = format!(
            "test.layer-ordered.lm-head.device-input.{}.{}",
            std::process::id(),
            line!()
        );
        let mut lm_head_values = vec![0.0_f32; 4 * GLM52_HIDDEN_SIZE];
        lm_head_values[0] = 1.0;
        lm_head_values[GLM52_HIDDEN_SIZE + 1] = 2.0;
        lm_head_values[2 * GLM52_HIDDEN_SIZE] = 3.0;
        lm_head_values[3 * GLM52_HIDDEN_SIZE] = 2.0;
        let lm_head = bf16_bytes_from_f32(&lm_head_values);
        preload_resident_weight_from_host_staging(
            &weight_name,
            lm_head.len(),
            "test layer-ordered device-input lm_head sampler",
            |staging| {
                staging.copy_from_slice(&lm_head);
                Ok(())
            },
        )
        .expect("preloading tiny layer-ordered lm_head weight");
        let catalog = TensorCatalog {
            model_id: "test/model".to_owned(),
            snapshot_path: ".".to_owned(),
            facts: ModelFacts::default(),
            tensors: vec![TensorInfo {
                name: weight_name,
                file: "unused.safetensors".to_owned(),
                dtype: DType::Bf16,
                shape: vec![4, GLM52_HIDDEN_SIZE],
                byte_offset: 0,
                byte_length: lm_head.len() as u64,
                role: TensorRole::LmHead,
                layer_id: None,
                expert_id: None,
                is_quantization_metadata: false,
            }],
        };
        let mut device_hidden_values = vec![0.0_f32; GLM52_HIDDEN_SIZE];
        device_hidden_values[0] = 1.0;
        let device_hidden = device_bf16_output_from_f32_values(
            &device_hidden_values,
            1,
            GLM52_HIDDEN_SIZE,
            "test layer-ordered lm_head device hidden",
        )
        .expect("uploading layer-ordered device hidden");
        let mut host_hidden_with_different_argmax = vec![0.0_f32; GLM52_HIDDEN_SIZE];
        host_hidden_with_different_argmax[1] = 2.0;

        let probe = layer_ordered_lm_head_sampling_probe(
            &catalog,
            "test-device-final-hidden",
            &host_hidden_with_different_argmax,
            Some(&device_hidden),
            true,
            Some(0.0),
            true,
        );

        assert_eq!(
            probe.status,
            "numeric-real-layer-ordered-lm-head-full-vocab"
        );
        assert!(probe.passed);
        assert!(probe.covers_full_vocabulary);
        assert_eq!(probe.top_token_id, Some(2));
        assert_eq!(probe.top_logit, Some(3.0));
        assert_eq!(probe.sample_top_k, Some(4));
        assert_eq!(
            probe.argmax_kernel_backend,
            Some(CUDA_REFERENCE_LM_HEAD_ARGMAX_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND)
        );
        assert_eq!(
            probe.sampler_kernel_backend,
            Some(CUDA_REFERENCE_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND)
        );
        assert!(terminal_lm_head_sampling_satisfies_completion_gate(&probe));
    }

    #[test]
    fn layer_ordered_lm_head_completion_gate_requires_full_vocabulary() {
        let bounded_probe = layer_ordered_lm_head_sampling_from_score(
            "test-layer-ordered-hidden",
            RealLmHeadChunkScoreForHidden {
                lm_head_tensor: "lm_head.weight".to_owned(),
                hidden_dim: GLM52_HIDDEN_SIZE,
                vocab_size: 5,
                start_token_id: 0,
                chunk_rows: 2,
                rows_scored: 2,
                chunks_scored: 1,
                lm_head_bytes_read: (2 * GLM52_HIDDEN_SIZE * 2) as u64,
                hidden_values: GLM52_HIDDEN_SIZE,
                logits_evaluated: 2,
                multiply_accumulate_ops: 2 * GLM52_HIDDEN_SIZE as u64,
                covers_full_vocabulary: false,
                logits_kernel_backend: "cpu-reference-lm-head-argmax-bf16",
                argmax_kernel_backend: "cpu-reference-lm-head-argmax-bf16",
                sampler_kernel_backend: "cpu-reference-lm-head-sample-topk-topp-bf16",
                top_token_id: 1,
                top_logit: 1.5,
                sampled_token_id: 1,
                sampled_score: 1.0,
                sample_random_uniform: 0.0,
                sample_temperature: 1.0,
                sample_top_k: 1,
                sample_top_p: 1.0,
            },
            Some(0.25),
        );

        assert!(bounded_probe.passed);
        assert!(!bounded_probe.covers_full_vocabulary);
        assert_eq!(
            bounded_probe.status,
            "numeric-real-layer-ordered-lm-head-chunk"
        );
        assert!(!terminal_lm_head_sampling_satisfies_completion_gate(
            &bounded_probe
        ));
    }

    #[test]
    fn layer_ordered_mlp_stage_sources_label_coordinator_backends() {
        let mut dense = RealFullDenseLayerPrefixHidden {
            hidden: vec![0.0; GLM52_HIDDEN_SIZE],
            device_hidden: None,
            layer_id: 0,
            intermediate_rows: 4,
            output_rows: 4,
            residual_adds: 1,
            norm_bytes_read: 8,
            weight_bytes_read: 16,
            norm_backend: "cpu-reference-rmsnorm-bf16",
            linear_backend: "cpu-reference-linear-bf16",
            mlp_backend: "cpu-reference-silu-gated-mlp-bf16",
            norm_checksum: 0.0,
            activation_checksum: 0.0,
            output_checksum: 0.0,
            output_l2_norm: 0.0,
            initial_residual_checksum: 0.0,
            residual_delta_checksum: 0.0,
            final_residual_checksum: 0.0,
            residual_add_backend: "cpu-reference-residual-add-bf16",
            first_residual_after: 0.0,
            last_residual_after: 0.0,
            passed: true,
        };
        let mut sparse = sparse_selected_route_fixture();

        assert_eq!(
            dense_stage_source(&dense),
            "real-checkpoint-bf16-dense-mlp-prefix-cpu-reference-rmsnorm-bf16-cpu-reference-linear-bf16-cpu-reference-silu-gated-mlp-bf16-cpu-reference-residual-add-bf16"
        );
        assert_eq!(
            sparse_stage_source(&sparse),
            "real-checkpoint-nvfp4-routed-plus-bf16-shared-moe-prefix-cpu-reference-rmsnorm-bf16-cpu-reference-router-topk-bf16-cpu-reference-shared-silu-gated-mlp-bf16-cpu-reference-residual-add-bf16"
        );
        assert_eq!(sparse_stage_status(&sparse), "real");
        sparse.passed = false;
        assert_eq!(sparse_stage_status(&sparse), "blocked");
        sparse.passed = true;

        dense.norm_backend = CUDA_REFERENCE_RMSNORM_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND;
        dense.linear_backend = CUDA_REFERENCE_LINEAR_BF16_RESIDENT_WEIGHT_BACKEND;
        dense.mlp_backend =
            CUDA_REFERENCE_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND;
        dense.residual_add_backend = CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND;
        assert_eq!(
            dense_stage_source(&dense),
            "real-checkpoint-bf16-dense-mlp-prefix-cuda-reference-rmsnorm-bf16-cuda-reference-linear-bf16-cuda-reference-silu-gated-mlp-bf16-cuda-reference-residual-add-bf16"
        );

        sparse.expert_input_norm_backend =
            CUDA_REFERENCE_RMSNORM_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND;
        sparse.router_backend =
            CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_BACKEND;
        sparse.shared_mlp_backend =
            CUDA_REFERENCE_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND;
        sparse.residual_add_backend = CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND;
        assert_eq!(
            sparse_stage_source(&sparse),
            "real-checkpoint-nvfp4-routed-plus-bf16-shared-moe-prefix-cuda-reference-rmsnorm-bf16-cuda-reference-router-topk-bf16-cuda-reference-shared-silu-gated-mlp-bf16-cuda-reference-residual-add-bf16"
        );
    }

    #[test]
    fn layer_ordered_attention_stage_source_labels_coordinator_backends() {
        let attention = RealFullAttentionResidualPrefixHidden {
            hidden: vec![0.0; GLM52_HIDDEN_SIZE],
            device_hidden: None,
            layer_id: 0,
            attention_rows: 2,
            prefix_context_rows: 0,
            total_context_rows: 2,
            uses_kv_cache_context: false,
            kv_cache_context_bytes: 0,
            residual_adds: 2,
            residual_prefix_values: 4,
            input_norm_bytes_read: 8,
            projection_bytes_read: 16,
            o_proj_bytes_read: 8,
            projection_backend: "cpu-reference-linear-bf16",
            attention_backend: "cpu-reference-causal-attention-bf16",
            residual_add_backend: "cpu-reference-residual-add-bf16",
            initial_residual_checksum: 0.0,
            residual_delta_checksum: 0.0,
            final_residual_checksum: 0.0,
            includes_causal_softmax: true,
            includes_mla_softmax: false,
            includes_dsa_candidate_selection: false,
            includes_dsa_softmax: false,
            dsa_candidate_rows: 0,
            dsa_selected_indices: Vec::new(),
            dsa_attention_context_checksum: None,
            dsa_projection_backend: None,
        };

        assert_eq!(
            attention_stage_source(&attention),
            "real-checkpoint-bf16-causal-attention-prefix-cpu-reference-linear-bf16-cpu-reference-causal-attention-bf16-cpu-reference-residual-add-bf16"
        );

        let cuda_resident_attention = RealFullAttentionResidualPrefixHidden {
            projection_backend: CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
            attention_backend: CUDA_REFERENCE_CAUSAL_ATTENTION_BF16_BACKEND,
            residual_add_backend: CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
            ..attention.clone()
        };
        assert_eq!(
            attention_stage_source(&cuda_resident_attention),
            "real-checkpoint-bf16-causal-attention-prefix-cuda-reference-linear-bf16-cuda-reference-causal-attention-bf16-cuda-reference-residual-add-bf16"
        );

        let mla_rope_attention = RealFullAttentionResidualPrefixHidden {
            attention_backend: "cpu-reference-mla-rope-attention-bf16",
            residual_prefix_values: 16,
            includes_mla_softmax: true,
            ..attention
        };
        assert_eq!(
            attention_stage_source(&mla_rope_attention),
            "real-checkpoint-bf16-bounded-main-mla-rope-attention-cpu-reference-linear-bf16-cpu-reference-mla-rope-attention-bf16-cpu-reference-residual-add-bf16"
        );

        let full_output_mla_rope_attention = RealFullAttentionResidualPrefixHidden {
            residual_prefix_values: GLM52_HIDDEN_SIZE,
            ..mla_rope_attention
        };
        assert_eq!(
            attention_stage_source(&full_output_mla_rope_attention),
            "real-checkpoint-bf16-full-output-main-mla-rope-attention-cpu-reference-linear-bf16-cpu-reference-mla-rope-attention-bf16-cpu-reference-residual-add-bf16"
        );

        let supplied_prefix_mla_rope_attention = RealFullAttentionResidualPrefixHidden {
            attention_rows: 1,
            prefix_context_rows: 1,
            total_context_rows: 2,
            includes_dsa_candidate_selection: true,
            includes_dsa_softmax: true,
            ..full_output_mla_rope_attention
        };
        assert_eq!(
            attention_stage_source(&supplied_prefix_mla_rope_attention),
            "real-checkpoint-bf16-full-output-main-mla-rope-supplied-prefix-context-plus-dsa-indexer-attention-cpu-reference-linear-bf16-cpu-reference-mla-rope-attention-bf16-cpu-reference-residual-add-bf16"
        );

        let kv_cache_prefix_mla_rope_attention = RealFullAttentionResidualPrefixHidden {
            uses_kv_cache_context: true,
            kv_cache_context_bytes: 1152,
            includes_dsa_candidate_selection: false,
            includes_dsa_softmax: false,
            ..supplied_prefix_mla_rope_attention
        };
        assert_eq!(
            attention_stage_source(&kv_cache_prefix_mla_rope_attention),
            "real-checkpoint-bf16-full-output-main-mla-rope-kv-cache-prefix-context-attention-cpu-reference-linear-bf16-cpu-reference-mla-rope-attention-bf16-cpu-reference-residual-add-bf16"
        );
    }

    #[test]
    fn mla_dsa_completion_tracker_requires_all_layers_kv_cache_and_dsa_indexers() {
        let mut complete = MlaDsaAttentionCompletionTracker::default();
        for layer_id in 0..GLM52_NUM_HIDDEN_LAYERS {
            complete.record(&mla_dsa_completion_attention_fixture(layer_id, true, true));
        }
        assert!(complete.uses_full_context_mla_dsa_attention());

        let mut missing_kv_cache = MlaDsaAttentionCompletionTracker::default();
        for layer_id in 0..GLM52_NUM_HIDDEN_LAYERS {
            missing_kv_cache.record(&mla_dsa_completion_attention_fixture(
                layer_id,
                layer_id != 0,
                true,
            ));
        }
        assert!(!missing_kv_cache.uses_full_context_mla_dsa_attention());

        let mut missing_dsa = MlaDsaAttentionCompletionTracker::default();
        let skipped_dsa_layer = GLM52_DSA_INDEXER_LAYER_IDS[0];
        for layer_id in 0..GLM52_NUM_HIDDEN_LAYERS {
            missing_dsa.record(&mla_dsa_completion_attention_fixture(
                layer_id,
                true,
                layer_id != skipped_dsa_layer,
            ));
        }
        assert!(!missing_dsa.uses_full_context_mla_dsa_attention());
    }

    fn mla_dsa_completion_attention_fixture(
        layer_id: usize,
        uses_kv_cache_context: bool,
        include_dsa: bool,
    ) -> RealFullAttentionResidualPrefixHidden {
        let dsa_layer = GLM52_DSA_INDEXER_LAYER_IDS.contains(&layer_id);
        RealFullAttentionResidualPrefixHidden {
            hidden: Vec::new(),
            device_hidden: None,
            layer_id,
            attention_rows: 1,
            prefix_context_rows: usize::from(uses_kv_cache_context),
            total_context_rows: if uses_kv_cache_context { 2 } else { 1 },
            uses_kv_cache_context,
            kv_cache_context_bytes: if uses_kv_cache_context { 1152 } else { 0 },
            residual_adds: 1,
            residual_prefix_values: GLM52_HIDDEN_SIZE,
            input_norm_bytes_read: 8,
            projection_bytes_read: 16,
            o_proj_bytes_read: 8,
            projection_backend: "cpu-reference-linear-bf16",
            attention_backend: "cpu-reference-mla-rope-attention-bf16",
            residual_add_backend: "cpu-reference-residual-add-bf16",
            initial_residual_checksum: 0.0,
            residual_delta_checksum: 0.0,
            final_residual_checksum: 0.0,
            includes_causal_softmax: true,
            includes_mla_softmax: true,
            includes_dsa_candidate_selection: dsa_layer && include_dsa,
            includes_dsa_softmax: dsa_layer && include_dsa,
            dsa_candidate_rows: usize::from(dsa_layer && include_dsa),
            dsa_selected_indices: Vec::new(),
            dsa_attention_context_checksum: None,
            dsa_projection_backend: None,
        }
    }

    #[test]
    #[ignore = "loads bounded real MLA/RoPE attention rows from the real checkpoint"]
    fn real_checkpoint_mla_rope_attention_hidden_from_supplied_residual_when_available() {
        let Some(catalog) = load_real_catalog_or_skip() else {
            return;
        };
        let initial_hidden = deterministic_dense_hidden(GLM52_HIDDEN_SIZE);
        let initial_prefix_checksum = checksum_f64(&initial_hidden[..16]);

        let attention = real_full_mla_rope_attention_prefix_hidden_for_layer_from_initial(
            &catalog,
            0,
            initial_hidden,
        )
        .expect("running supplied-hidden bounded MLA/RoPE attention residual");

        assert_eq!(attention.layer_id, 0);
        assert_eq!(attention.attention_rows, 1);
        assert_eq!(attention.residual_adds, 1);
        assert_eq!(attention.residual_prefix_values, 16);
        assert!(attention.includes_causal_softmax);
        assert!(attention.includes_mla_softmax);
        assert!(attention.includes_dsa_candidate_selection);
        assert!(attention.includes_dsa_softmax);
        assert_eq!(attention.dsa_candidate_rows, 1);
        assert_eq!(attention.dsa_selected_indices, vec![0]);
        assert_eq!(
            attention.dsa_projection_backend,
            Some("cpu-reference-linear-bf16")
        );
        assert!(attention
            .dsa_attention_context_checksum
            .expect("DSA attention context checksum")
            .is_finite());
        assert_eq!(
            attention_stage_source(&attention),
            "real-checkpoint-bf16-bounded-main-mla-rope-plus-dsa-indexer-attention-cpu-reference-linear-bf16-cpu-reference-mla-rope-attention-bf16-cpu-reference-residual-add-bf16"
        );
        assert!(attention.initial_residual_checksum.is_finite());
        assert!(attention.residual_delta_checksum.is_finite());
        assert!(attention.final_residual_checksum.is_finite());
        assert!(attention.residual_delta_checksum.abs() > 0.0);
        assert!(!approx_eq_f64(
            initial_prefix_checksum,
            attention.final_residual_checksum
        ));
    }

    #[test]
    #[ignore = "loads hidden-width real MLA/RoPE attention rows from the real checkpoint"]
    fn real_checkpoint_mla_rope_attention_full_output_hidden_from_supplied_residual_when_available()
    {
        let Some(catalog) = load_real_catalog_or_skip() else {
            return;
        };
        let initial_hidden = deterministic_dense_hidden(GLM52_HIDDEN_SIZE);
        let initial_checksum = checksum_f64(&initial_hidden);

        let attention = real_full_mla_rope_attention_full_output_hidden_for_layer_from_initial(
            &catalog,
            0,
            initial_hidden,
        )
        .expect("running supplied-hidden full-output MLA/RoPE attention residual");

        assert_eq!(attention.layer_id, 0);
        assert_eq!(attention.attention_rows, 1);
        assert_eq!(attention.residual_adds, 1);
        assert_eq!(attention.residual_prefix_values, GLM52_HIDDEN_SIZE);
        assert!(attention.includes_causal_softmax);
        assert!(attention.includes_mla_softmax);
        assert!(attention.includes_dsa_candidate_selection);
        assert!(attention.includes_dsa_softmax);
        assert_eq!(
            attention_stage_source(&attention),
            "real-checkpoint-bf16-full-output-main-mla-rope-plus-dsa-indexer-attention-cpu-reference-linear-bf16-cpu-reference-mla-rope-attention-bf16-cpu-reference-residual-add-bf16"
        );
        assert!(attention.initial_residual_checksum.is_finite());
        assert!(attention.residual_delta_checksum.is_finite());
        assert!(attention.final_residual_checksum.is_finite());
        assert!(!approx_eq_f64(
            initial_checksum,
            attention.final_residual_checksum
        ));
    }

    #[test]
    fn sparse_tensor_artifacts_cover_all_selected_topk_routes() {
        let layer_id = GLM52_FIRST_K_DENSE_REPLACE;
        let routes = vec![
            RealFullSparseMoeRoute {
                rank: 0,
                expert_id: 17,
                owner: "spark-0".to_owned(),
                score: 0.75,
                corrected_score: 0.8,
                normalized_weight: 0.6,
            },
            RealFullSparseMoeRoute {
                rank: 1,
                expert_id: 33,
                owner: "spark-1".to_owned(),
                score: 0.5,
                corrected_score: 0.7,
                normalized_weight: 0.4,
            },
        ];
        let catalog = sparse_artifact_catalog(layer_id, &[17, 33]);
        let tensor_metadata = TensorMetadataLookup::new(&catalog);
        let sparse = RealFullSparseMlpSharedLayerHidden {
            hidden: Vec::new(),
            device_hidden: None,
            expert_input_hidden_bf16_payload: vec![0; GLM52_HIDDEN_BF16_BYTES],
            layer_id,
            route_count: routes.len(),
            routes: routes.clone(),
            routed_outputs: vec![0.0; 4],
            shared_outputs: vec![0.0; 4],
            layer_outputs: vec![0.0; 4],
            shared_expert_executed: true,
            routed_intermediate_rows: 4,
            shared_intermediate_rows: 4,
            output_rows: 4,
            residual_adds: 1,
            final_residual_checksum: 0.0,
            expert_input_norm_backend: "cpu-reference-rmsnorm-bf16",
            router_backend: "cpu-reference-router-topk-bf16",
            shared_mlp_backend: "cpu-reference-silu-gated-mlp-bf16",
            residual_add_backend: "cpu-reference-residual-add-bf16",
            layer_summary: RealFullExpertSparseMlpSharedChainLayerProbe {
                layer_id,
                expert_id: routes[0].expert_id,
                owner: routes[0].owner.clone(),
                score: routes[0].score,
                corrected_score: routes[0].corrected_score,
                routed_output_checksum: 0.0,
                shared_output_checksum: 0.0,
                output_checksum: 0.0,
                output_l2_norm: 0.0,
                residual_before_checksum: 0.0,
                residual_delta_checksum: 0.0,
                residual_after_checksum: 0.0,
                expert_input_norm_backend: "cpu-reference-rmsnorm-bf16",
                router_backend: "cpu-reference-router-topk-bf16",
                shared_mlp_backend: "cpu-reference-silu-gated-mlp-bf16",
                residual_add_backend: "cpu-reference-residual-add-bf16",
                first_residual_after: 0.0,
                last_residual_after: 0.0,
            },
            covers_full_top_k: false,
            passed: true,
        };

        let artifacts = sparse_tensor_artifacts(&tensor_metadata, &sparse);

        assert_eq!(artifacts.len(), 2 + routes.len() * 9 + 3);
        for route in routes {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                assert_artifact(
                    &artifacts,
                    &format!(
                        "model.layers.{layer_id}.mlp.experts.{}.{projection}.weight",
                        route.expert_id
                    ),
                    DType::U8,
                    TensorRole::RoutedExpert,
                    Some(4),
                    false,
                );
                assert_artifact(
                    &artifacts,
                    &format!(
                        "model.layers.{layer_id}.mlp.experts.{}.{projection}.weight_scale",
                        route.expert_id
                    ),
                    DType::F8E4M3,
                    TensorRole::RoutedExpert,
                    Some(4),
                    false,
                );
                assert_artifact(
                    &artifacts,
                    &format!(
                        "model.layers.{layer_id}.mlp.experts.{}.{projection}.weight_scale_2",
                        route.expert_id
                    ),
                    DType::F32,
                    TensorRole::RoutedExpert,
                    None,
                    true,
                );
            }
        }
    }

    #[test]
    fn sparse_host_batch_set_evidence_partitions_selected_routes_by_owner() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);
        let layer_id = GLM52_FIRST_K_DENSE_REPLACE;
        let sparse = sparse_selected_route_fixture();

        let evidence = sparse_host_batch_set_evidence(&sparse);

        assert_eq!(evidence.status, "protocol-v2-frames-from-selected-routes");
        assert_eq!(evidence.layer_id, layer_id);
        assert_eq!(evidence.global_rows, 1);
        assert_eq!(evidence.host_batches, 2);
        assert_eq!(evidence.host_rows, 2);
        assert_eq!(evidence.routes, 3);
        assert_eq!(evidence.hidden_dim, GLM52_HIDDEN_SIZE);
        assert_eq!(evidence.hidden_bytes_per_row, 12_288);
        assert_eq!(evidence.hidden_dtype, DType::Bf16);
        assert_eq!(evidence.graph_bucket_rows, 1);
        assert_eq!(evidence.touched_hosts, vec!["spark-0", "spark-1"]);
        assert_eq!(evidence.per_host_route_counts, vec![2, 1]);
        assert_eq!(evidence.reconstruction_global_rows, 1);
        assert_eq!(evidence.reconstruction_host_maps, 2);
        assert_eq!(evidence.protocol_v2_request_frames, 2);
        assert_eq!(evidence.protocol_v2_request_rows, 2);
        assert_eq!(evidence.protocol_v2_request_routes, 3);
        assert_eq!(
            evidence.protocol_v2_hidden_payload_bytes,
            2 * GLM52_HIDDEN_BF16_BYTES
        );
        assert!(
            evidence.protocol_v2_request_wire_bytes > evidence.protocol_v2_hidden_payload_bytes
        );
        assert!(evidence.protocol_v2_views_parse);
        assert!(evidence.protocol_v2_uses_compact_hidden_payloads);
        assert_eq!(
            evidence.protocol_v2_row_gather_backend,
            "cpu-reference-gather-rows-bf16"
        );
        assert_eq!(
            evidence.protocol_v2_synthetic_executor,
            PROTOCOL_V2_SYNTHETIC_ROUTE_EXECUTOR
        );
        assert_eq!(evidence.protocol_v2_synthetic_response_frames, 2);
        assert_eq!(evidence.protocol_v2_synthetic_response_rows, 2);
        assert!(evidence.protocol_v2_synthetic_response_wire_bytes > 0);
        assert_eq!(
            evidence.protocol_v2_row_scatter_backend,
            "cpu-reference-scatter-add-rows-bf16-to-f32"
        );
        assert_eq!(
            evidence.protocol_v2_accumulated_output_values,
            GLM52_HIDDEN_SIZE
        );
        assert_eq!(
            evidence.protocol_v2_accumulated_contribution_counts,
            vec![2]
        );
        assert!(evidence
            .protocol_v2_accumulated_output_checksum
            .unwrap()
            .is_finite());
        assert!(evidence.protocol_v2_synthetic_outputs_finite);
        assert!(evidence.protocol_v2_executes_route_dependent_synthetic);
        assert!(evidence.uses_expert_input_hidden_payload);
        assert!(evidence.uses_selected_routes);
        assert!(evidence.uses_route_owners);
        assert!(!evidence.closes_live_expert_daemon_moe_gate);
        assert!(evidence.skipped_reason.is_none());
    }

    #[tokio::test]
    async fn sparse_selected_routes_dispatch_over_protocol_v2_tcp() {
        let sparse = sparse_selected_route_fixture();
        let plan = build_sparse_selected_routes_host_batch_set(&sparse)
            .expect("building selected-route sparse host-batch set");
        let touched_hosts = plan
            .set
            .touched_hosts()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        assert_eq!(touched_hosts, vec!["spark-0", "spark-1"]);
        let mut targets = Vec::new();
        let mut servers = Vec::new();
        for host in touched_hosts {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            servers.push(tokio::spawn(async move {
                let _ = serve_protocol_v2_tcp_listener_with_executor(
                    listener,
                    Arc::new(SyntheticRouteExecutor),
                )
                .await;
            }));
            targets.push(TcpProtocolV2HostBatchTarget { host, addr });
        }

        let dispatch = tcp_protocol_v2_host_batch_set_bf16_dispatch(
            &plan.set,
            &sparse.expert_input_hidden_bf16_payload,
            &targets,
            500,
            TcpTransportConfig::default(),
        )
        .await
        .expect("dispatching selected-route sparse ProtocolV2 host batches over TCP");

        assert_eq!(dispatch.stats.hosts, 2);
        assert_eq!(dispatch.stats.global_rows, 1);
        assert_eq!(dispatch.stats.host_rows, 2);
        assert_eq!(dispatch.stats.routes, 3);
        assert_eq!(dispatch.stats.output_dim, GLM52_HIDDEN_SIZE);
        assert_eq!(dispatch.stats.output_values, GLM52_HIDDEN_SIZE);
        assert_eq!(
            dispatch.stats.response_executor_ids,
            vec![
                expert_protocol_v2_compact_id(PROTOCOL_V2_SYNTHETIC_ROUTE_EXECUTOR),
                expert_protocol_v2_compact_id(PROTOCOL_V2_SYNTHETIC_ROUTE_EXECUTOR)
            ]
        );
        assert_eq!(dispatch.stats.contribution_counts, vec![2]);
        assert!(dispatch.stats.request_wire_bytes > 0);
        assert!(dispatch.stats.response_wire_bytes > 0);
        assert!(dispatch
            .accumulation
            .values
            .iter()
            .all(|value| value.is_finite()));
        assert!(dispatch
            .accumulation
            .values
            .iter()
            .any(|value| *value != 0.0));
        assert_eq!(
            dispatch.stats.output_checksum,
            dispatch
                .accumulation
                .values
                .iter()
                .map(|value| *value as f64)
                .sum::<f64>()
        );
        eprintln!(
            "sparse_selected_routes_protocol_v2_tcp_dispatch executor={} hosts={} global_rows={} host_rows={} routes={} hidden_dim={} output_values={} contribution_counts={:?} request_wire_bytes={} response_wire_bytes={} output_checksum={}",
            PROTOCOL_V2_SYNTHETIC_ROUTE_EXECUTOR,
            dispatch.stats.hosts,
            dispatch.stats.global_rows,
            dispatch.stats.host_rows,
            dispatch.stats.routes,
            dispatch.stats.output_dim,
            dispatch.stats.output_values,
            dispatch.stats.contribution_counts,
            dispatch.stats.request_wire_bytes,
            dispatch.stats.response_wire_bytes,
            dispatch.stats.output_checksum,
        );
        for server in servers {
            server.abort();
        }
    }

    #[tokio::test]
    async fn sparse_selected_routes_protocol_v2_residual_step_over_tcp() {
        let mut sparse = sparse_selected_route_fixture();
        sparse.shared_outputs = vec![0.125, -0.25, 0.5, -0.75];
        let plan = build_sparse_selected_routes_host_batch_set(&sparse)
            .expect("building selected-route sparse host-batch set");
        let touched_hosts = plan
            .set
            .touched_hosts()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        assert_eq!(touched_hosts, vec!["spark-0", "spark-1"]);
        let mut targets = Vec::new();
        let mut servers = Vec::new();
        for host in touched_hosts {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            servers.push(tokio::spawn(async move {
                let _ = serve_protocol_v2_tcp_listener_with_executor(
                    listener,
                    Arc::new(SyntheticRouteExecutor),
                )
                .await;
            }));
            targets.push(TcpProtocolV2HostBatchTarget { host, addr });
        }

        let residual_before = deterministic_layer_ordered_daemon_hidden(sparse.layer_id);
        let residual_step = sparse_moe_protocol_v2_residual_step(
            &sparse,
            &residual_before,
            &targets,
            502,
            TcpTransportConfig::default(),
            synthetic_sparse_moe_dispatch_kind(),
        )
        .await
        .expect("executing selected-route sparse ProtocolV2 residual step over TCP");

        assert_eq!(residual_step.dispatch_stats.hosts, 2);
        assert_eq!(residual_step.dispatch_stats.global_rows, 1);
        assert_eq!(residual_step.dispatch_stats.host_rows, 2);
        assert_eq!(residual_step.dispatch_stats.routes, 3);
        assert_eq!(residual_step.dispatch_stats.output_dim, sparse.output_rows);
        assert_eq!(
            residual_step.dispatch_stats.output_values,
            sparse.output_rows
        );
        assert_eq!(
            residual_step.dispatch_stats.response_executor_ids,
            vec![
                expert_protocol_v2_compact_id(PROTOCOL_V2_SYNTHETIC_ROUTE_EXECUTOR),
                expert_protocol_v2_compact_id(PROTOCOL_V2_SYNTHETIC_ROUTE_EXECUTOR)
            ]
        );
        assert_eq!(residual_step.dispatch_stats.contribution_counts, vec![2]);
        assert_eq!(residual_step.dispatch_stats.graph_pool_leases, 0);
        assert_eq!(
            residual_step.dispatch_stats.graph_pool_bucket_rows,
            Vec::<usize>::new()
        );
        assert!(residual_step.device_hidden.is_none());
        assert_eq!(residual_step.hidden_after.len(), GLM52_HIDDEN_SIZE);
        assert_ne!(
            &residual_step.hidden_after[..sparse.output_rows],
            &residual_before[..sparse.output_rows]
        );
        assert_eq!(
            &residual_step.hidden_after[sparse.output_rows..],
            &residual_before[sparse.output_rows..]
        );
        assert!(residual_step.routed_output_checksum.is_finite());
        assert!(residual_step.shared_output_checksum.is_finite());
        assert!(residual_step.residual_delta_checksum.is_finite());
        assert!(residual_step.residual_after_checksum.is_finite());
        assert_eq!(
            residual_step
                .host_batch_set_evidence
                .protocol_v2_synthetic_executor,
            PROTOCOL_V2_SYNTHETIC_ROUTE_EXECUTOR
        );
        assert!(
            !residual_step
                .host_batch_set_evidence
                .closes_live_expert_daemon_moe_gate
        );
        assert_eq!(
            residual_step
                .host_batch_set_evidence
                .protocol_v2_accumulated_contribution_counts,
            vec![2]
        );

        let real_gate_error = match sparse_moe_protocol_v2_residual_step(
            &sparse,
            &residual_before,
            &targets,
            503,
            TcpTransportConfig::default(),
            real_expertd_sparse_moe_dispatch_kind(),
        )
        .await
        {
            Ok(_) => panic!("synthetic ProtocolV2 endpoints must not close the real expertd gate"),
            Err(error) => error.to_string(),
        };
        assert!(real_gate_error.contains("expected executor"));
        assert!(real_gate_error.contains(PROTOCOL_V2_REAL_NVFP4_CHECKPOINT_EXECUTOR));

        let mut full_width_sparse = sparse_selected_route_fixture();
        full_width_sparse.output_rows = GLM52_HIDDEN_SIZE;
        full_width_sparse.routed_outputs = vec![0.0; GLM52_HIDDEN_SIZE];
        full_width_sparse.shared_outputs = vec![0.0; GLM52_HIDDEN_SIZE];
        full_width_sparse.layer_outputs = vec![0.0; GLM52_HIDDEN_SIZE];
        full_width_sparse.routed_intermediate_rows = GLM52_HIDDEN_SIZE;
        full_width_sparse.shared_intermediate_rows = GLM52_HIDDEN_SIZE;
        let full_width_residual_step = sparse_moe_protocol_v2_residual_step(
            &full_width_sparse,
            &residual_before,
            &targets,
            504,
            TcpTransportConfig::default(),
            synthetic_sparse_moe_dispatch_kind(),
        )
        .await
        .expect("executing full-width selected-route sparse ProtocolV2 residual step over TCP");
        assert_eq!(
            full_width_residual_step.dispatch_stats.output_dim,
            GLM52_HIDDEN_SIZE
        );
        assert_eq!(
            full_width_residual_step.dispatch_stats.output_values,
            GLM52_HIDDEN_SIZE
        );
        assert_eq!(full_width_residual_step.dispatch_stats.graph_pool_leases, 2);
        assert_eq!(
            full_width_residual_step
                .dispatch_stats
                .graph_pool_active_rows,
            full_width_residual_step.dispatch_stats.host_rows
        );
        assert_eq!(
            full_width_residual_step
                .dispatch_stats
                .graph_pool_active_routes,
            full_width_residual_step.dispatch_stats.routes
        );
        assert_eq!(
            full_width_residual_step
                .dispatch_stats
                .graph_pool_bucket_rows,
            vec![1, 1]
        );
        assert!(
            full_width_residual_step
                .dispatch_stats
                .graph_pool_fixed_buffer_bytes
                >= 2 * GLM52_HIDDEN_BF16_BYTES
        );
        assert_eq!(
            full_width_residual_step.device_hidden.is_some(),
            coordinator_cuda_reference_kernels_enabled()
        );
        assert!(full_width_residual_step.routed_output_checksum.is_finite());

        for server in servers {
            server.abort();
        }
    }

    #[tokio::test]
    async fn sparse_selected_routes_dispatch_through_expertd_tcp_entrypoint() {
        let sparse = sparse_selected_route_fixture();
        let plan = build_sparse_selected_routes_host_batch_set(&sparse)
            .expect("building selected-route sparse host-batch set");
        let touched_hosts = plan
            .set
            .touched_hosts()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        assert_eq!(touched_hosts, vec!["spark-0", "spark-1"]);
        let mut targets = Vec::new();
        let mut servers = Vec::new();
        for host in touched_hosts {
            let addr = unused_loopback_addr();
            let args = ExpertDaemonArgs {
                synthetic_weights: true,
                preflight_only: false,
                transport: "tcp".to_owned(),
                listen: addr.to_string(),
                loadplan: None,
                catalog: None,
                model_id: glmrt_core::DEFAULT_MODEL_ID.to_owned(),
                real_layer: Some(GLM52_FIRST_K_DENSE_REPLACE as u32),
                role_hostname: Some(host.clone()),
            };
            servers.push(tokio::spawn(async move { run_expertd(args).await }));
            wait_for_expertd_tcp_listener(addr).await;
            targets.push(TcpProtocolV2HostBatchTarget { host, addr });
        }

        let dispatch = tcp_protocol_v2_host_batch_set_bf16_dispatch(
            &plan.set,
            &sparse.expert_input_hidden_bf16_payload,
            &targets,
            501,
            TcpTransportConfig::default(),
        )
        .await
        .expect("dispatching selected-route sparse ProtocolV2 host batches through expertd");

        assert_eq!(dispatch.stats.hosts, 2);
        assert_eq!(dispatch.stats.global_rows, 1);
        assert_eq!(dispatch.stats.host_rows, 2);
        assert_eq!(dispatch.stats.routes, 3);
        assert_eq!(dispatch.stats.output_dim, GLM52_HIDDEN_SIZE);
        assert_eq!(dispatch.stats.output_values, GLM52_HIDDEN_SIZE);
        assert_eq!(
            dispatch.stats.response_executor_ids,
            vec![
                expert_protocol_v2_compact_id(PROTOCOL_V2_SYNTHETIC_ROUTE_EXECUTOR),
                expert_protocol_v2_compact_id(PROTOCOL_V2_SYNTHETIC_ROUTE_EXECUTOR)
            ]
        );
        assert_eq!(dispatch.stats.contribution_counts, vec![2]);
        assert!(dispatch.stats.request_wire_bytes > 0);
        assert!(dispatch.stats.response_wire_bytes > 0);
        assert!(dispatch
            .accumulation
            .values
            .iter()
            .all(|value| value.is_finite()));
        assert!(dispatch
            .accumulation
            .values
            .iter()
            .any(|value| *value != 0.0));
        assert_eq!(
            dispatch.stats.output_checksum,
            dispatch
                .accumulation
                .values
                .iter()
                .map(|value| *value as f64)
                .sum::<f64>()
        );
        eprintln!(
            "sparse_selected_routes_expertd_tcp_entrypoint_dispatch daemon=run_expertd executor={} hosts={} global_rows={} host_rows={} routes={} hidden_dim={} output_values={} contribution_counts={:?} request_wire_bytes={} response_wire_bytes={} output_checksum={}",
            PROTOCOL_V2_SYNTHETIC_ROUTE_EXECUTOR,
            dispatch.stats.hosts,
            dispatch.stats.global_rows,
            dispatch.stats.host_rows,
            dispatch.stats.routes,
            dispatch.stats.output_dim,
            dispatch.stats.output_values,
            dispatch.stats.contribution_counts,
            dispatch.stats.request_wire_bytes,
            dispatch.stats.response_wire_bytes,
            dispatch.stats.output_checksum,
        );
        for server in servers {
            server.abort();
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    #[ignore = "starts real-expertd checkpoint daemons; run explicitly for layer-ordered live route dispatch coverage"]
    async fn layer_ordered_sparse_stage_routes_dispatch_through_unpinned_real_expertd_entrypoints_when_available(
    ) {
        let Some(catalog) = load_real_catalog_or_skip() else {
            return;
        };
        let Some(loadplan_path) = load_full_loadplan_path_or_skip() else {
            return;
        };
        let catalog_path = real_catalog_path();
        let hosts = EXPERT_HOSTS
            .iter()
            .map(|host| (*host).to_owned())
            .collect::<Vec<_>>();
        let layer_ids = [GLM52_FIRST_K_DENSE_REPLACE, GLM52_NUM_HIDDEN_LAYERS - 1];
        let hidden_dim = 16;
        let mut total_global_rows = 0;
        let mut total_host_rows = 0;
        let mut total_routes = 0;
        let mut total_output_values = 0;
        let mut total_request_wire_bytes = 0;
        let mut total_response_wire_bytes = 0;
        let mut host_batches_per_layer = Vec::new();
        let mut contribution_counts_per_layer = Vec::new();
        let mut output_checksums = Vec::new();

        for layer_id in layer_ids {
            let (targets, servers) = start_unpinned_real_expertd_targets(
                &catalog_path,
                &loadplan_path,
                &hosts,
                Some(layer_id as u32),
            )
            .await;
            let sparse = real_sparse_mlp_shared_layer_hidden_from_initial(
                &catalog,
                layer_id,
                deterministic_layer_ordered_daemon_hidden(layer_id),
            )
            .expect("running bounded layer-ordered sparse MoE stage");
            assert_eq!(sparse.layer_id, layer_id);
            assert_eq!(sparse.route_count, GLM52_TOP_K);
            assert!(sparse.covers_full_top_k);
            assert!(sparse.passed);

            let plan =
                build_sparse_selected_routes_host_batch_set_for_hidden_dim(&sparse, hidden_dim)
                    .expect("building bounded selected-route sparse host-batch set");
            let hidden_payload =
                bounded_bf16_hidden_payload(&sparse.expert_input_hidden_bf16_payload, hidden_dim)
                    .expect("building bounded BF16 hidden payload");
            let dispatch = tcp_protocol_v2_host_batch_set_bf16_dispatch(
                &plan.set,
                &hidden_payload,
                &targets,
                508 + layer_id as u64,
                TcpTransportConfig::default(),
            )
            .await
            .expect("dispatching layer-ordered sparse stage routes through real expertd");

            assert_eq!(dispatch.stats.global_rows, 1);
            assert_eq!(dispatch.stats.host_rows, plan.set.host_row_count());
            assert_eq!(dispatch.stats.routes, GLM52_TOP_K);
            assert_eq!(dispatch.stats.output_dim, hidden_dim);
            assert_eq!(dispatch.stats.output_values, hidden_dim);
            assert_eq!(dispatch.stats.contribution_counts.len(), 1);
            assert_eq!(
                dispatch.stats.contribution_counts[0],
                plan.set.host_row_count()
            );
            assert!(dispatch.stats.request_wire_bytes > 0);
            assert!(dispatch.stats.response_wire_bytes > 0);
            assert!(dispatch
                .accumulation
                .values
                .iter()
                .all(|value| value.is_finite()));
            assert!(dispatch
                .accumulation
                .values
                .iter()
                .any(|value| *value != 0.0));

            total_global_rows += dispatch.stats.global_rows;
            total_host_rows += dispatch.stats.host_rows;
            total_routes += dispatch.stats.routes;
            total_output_values += dispatch.stats.output_values;
            total_request_wire_bytes += dispatch.stats.request_wire_bytes;
            total_response_wire_bytes += dispatch.stats.response_wire_bytes;
            host_batches_per_layer.push(dispatch.stats.hosts);
            contribution_counts_per_layer.push(dispatch.stats.contribution_counts[0]);
            output_checksums.push(dispatch.stats.output_checksum);

            for server in servers {
                server.abort();
            }
        }

        eprintln!(
            "layer_ordered_sparse_stage_real_nvfp4_unpinned_expertd_dispatch daemon=run_expertd executor=protocol-v2-real-nvfp4-checkpoint-executor serving_layer_filter=per-layer layers={:?} hosts={} host_names={:?} hidden_dim={} total_global_rows={} total_host_rows={} total_routes={} total_output_values={} host_batches_per_layer={:?} contribution_counts_per_layer={:?} total_request_wire_bytes={} total_response_wire_bytes={} output_checksums={:?}",
            layer_ids,
            hosts.len(),
            hosts,
            hidden_dim,
            total_global_rows,
            total_host_rows,
            total_routes,
            total_output_values,
            host_batches_per_layer,
            contribution_counts_per_layer,
            total_request_wire_bytes,
            total_response_wire_bytes,
            output_checksums
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    #[ignore = "requires full assigned real-expertd projection preload for all sparse layers before TCP bind"]
    async fn layer_ordered_sparse_stage_routes_dispatch_all_sparse_layers_through_unpinned_real_expertd_entrypoints_when_available(
    ) {
        let Some(catalog) = load_real_catalog_or_skip() else {
            return;
        };
        let Some(loadplan_path) = load_full_loadplan_path_or_skip() else {
            return;
        };
        let catalog_path = real_catalog_path();
        let hosts = EXPERT_HOSTS
            .iter()
            .map(|host| (*host).to_owned())
            .collect::<Vec<_>>();
        let (targets, servers) =
            start_unpinned_real_expertd_targets(&catalog_path, &loadplan_path, &hosts, None).await;

        let hidden_dim = 16;
        let mut first_layer = None;
        let mut last_layer = None;
        let mut layer_count = 0;
        let mut total_global_rows = 0;
        let mut total_host_rows = 0;
        let mut total_routes = 0;
        let mut total_output_values = 0;
        let mut total_request_wire_bytes = 0;
        let mut total_response_wire_bytes = 0;
        let mut min_host_batches_per_layer = usize::MAX;
        let mut max_host_batches_per_layer = 0usize;
        let mut output_checksum_sum = 0.0f64;

        for layer_id in GLM52_FIRST_K_DENSE_REPLACE..GLM52_NUM_HIDDEN_LAYERS {
            let sparse = real_sparse_mlp_shared_layer_hidden_from_initial(
                &catalog,
                layer_id,
                deterministic_layer_ordered_daemon_hidden(layer_id),
            )
            .expect("running bounded layer-ordered sparse MoE stage");
            assert_eq!(sparse.layer_id, layer_id);
            assert_eq!(sparse.route_count, GLM52_TOP_K);
            assert!(sparse.covers_full_top_k);
            assert!(sparse.passed);

            let plan =
                build_sparse_selected_routes_host_batch_set_for_hidden_dim(&sparse, hidden_dim)
                    .expect("building bounded selected-route sparse host-batch set");
            let hidden_payload =
                bounded_bf16_hidden_payload(&sparse.expert_input_hidden_bf16_payload, hidden_dim)
                    .expect("building bounded BF16 hidden payload");
            let dispatch = tcp_protocol_v2_host_batch_set_bf16_dispatch(
                &plan.set,
                &hidden_payload,
                &targets,
                509_000 + layer_id as u64,
                TcpTransportConfig::default(),
            )
            .await
            .expect("dispatching all layer-ordered sparse stage routes through real expertd");

            assert_eq!(dispatch.stats.global_rows, 1);
            assert_eq!(dispatch.stats.hosts, plan.set.num_hosts());
            assert_eq!(dispatch.stats.host_rows, plan.set.host_row_count());
            assert_eq!(dispatch.stats.routes, GLM52_TOP_K);
            assert_eq!(dispatch.stats.output_dim, hidden_dim);
            assert_eq!(dispatch.stats.output_values, hidden_dim);
            assert_eq!(dispatch.stats.contribution_counts.len(), 1);
            assert_eq!(
                dispatch.stats.contribution_counts[0],
                plan.set.host_row_count()
            );
            assert!(dispatch.stats.hosts > 0);
            assert!(dispatch.stats.hosts <= hosts.len());
            assert!(dispatch.stats.request_wire_bytes > 0);
            assert!(dispatch.stats.response_wire_bytes > 0);
            assert!(dispatch
                .accumulation
                .values
                .iter()
                .all(|value| value.is_finite()));
            assert!(dispatch
                .accumulation
                .values
                .iter()
                .any(|value| *value != 0.0));

            if first_layer.is_none() {
                first_layer = Some(layer_id);
            }
            last_layer = Some(layer_id);
            layer_count += 1;
            total_global_rows += dispatch.stats.global_rows;
            total_host_rows += dispatch.stats.host_rows;
            total_routes += dispatch.stats.routes;
            total_output_values += dispatch.stats.output_values;
            total_request_wire_bytes += dispatch.stats.request_wire_bytes;
            total_response_wire_bytes += dispatch.stats.response_wire_bytes;
            min_host_batches_per_layer = min_host_batches_per_layer.min(dispatch.stats.hosts);
            max_host_batches_per_layer = max_host_batches_per_layer.max(dispatch.stats.hosts);
            output_checksum_sum += dispatch.stats.output_checksum;
        }

        assert_eq!(
            layer_count,
            GLM52_NUM_HIDDEN_LAYERS - GLM52_FIRST_K_DENSE_REPLACE
        );
        assert_eq!(first_layer, Some(GLM52_FIRST_K_DENSE_REPLACE));
        assert_eq!(last_layer, Some(GLM52_NUM_HIDDEN_LAYERS - 1));
        assert_eq!(total_global_rows, layer_count);
        assert_eq!(total_routes, layer_count * GLM52_TOP_K);
        assert_eq!(total_output_values, layer_count * hidden_dim);
        assert!(output_checksum_sum.is_finite());
        assert!(output_checksum_sum != 0.0);

        eprintln!(
            "layer_ordered_sparse_stage_all_layers_real_nvfp4_unpinned_expertd_dispatch daemon=run_expertd executor=protocol-v2-real-nvfp4-checkpoint-executor serving_layer_filter=none layers={} first_layer={} last_layer={} hosts={} host_names={:?} hidden_dim={} total_global_rows={} total_host_rows={} total_routes={} total_output_values={} min_host_batches_per_layer={} max_host_batches_per_layer={} total_request_wire_bytes={} total_response_wire_bytes={} output_checksum_sum={}",
            layer_count,
            first_layer.unwrap(),
            last_layer.unwrap(),
            hosts.len(),
            hosts,
            hidden_dim,
            total_global_rows,
            total_host_rows,
            total_routes,
            total_output_values,
            min_host_batches_per_layer,
            max_host_batches_per_layer,
            total_request_wire_bytes,
            total_response_wire_bytes,
            output_checksum_sum
        );

        for server in servers {
            server.abort();
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    #[ignore = "requires full assigned real-expertd projection preload for all sparse layers before TCP bind"]
    async fn layer_ordered_sparse_stage_live_expertd_outputs_mutate_bounded_residual_chain_when_available(
    ) {
        let Some(catalog) = load_real_catalog_or_skip() else {
            return;
        };
        let Some(loadplan_path) = load_full_loadplan_path_or_skip() else {
            return;
        };
        let catalog_path = real_catalog_path();
        let hosts = EXPERT_HOSTS
            .iter()
            .map(|host| (*host).to_owned())
            .collect::<Vec<_>>();
        let (targets, servers) =
            start_unpinned_real_expertd_targets(&catalog_path, &loadplan_path, &hosts, None).await;

        let mut hidden = deterministic_layer_ordered_daemon_hidden(GLM52_FIRST_K_DENSE_REPLACE);
        let mut output_rows = None;
        let mut first_layer = None;
        let mut last_layer = None;
        let mut layer_count = 0;
        let mut residual_adds = 0;
        let mut shared_layers = 0;
        let mut total_global_rows = 0;
        let mut total_host_rows = 0;
        let mut total_routes = 0;
        let mut total_output_values = 0;
        let mut total_request_wire_bytes = 0;
        let mut total_response_wire_bytes = 0;
        let mut min_host_batches_per_layer = usize::MAX;
        let mut max_host_batches_per_layer = 0usize;
        let mut routed_output_checksum_sum = 0.0f64;
        let mut shared_output_checksum_sum = 0.0f64;
        let mut residual_delta_checksum_sum = 0.0f64;
        let mut final_residual_checksum = 0.0f64;

        for layer_id in GLM52_FIRST_K_DENSE_REPLACE..GLM52_NUM_HIDDEN_LAYERS {
            let sparse = real_sparse_mlp_shared_layer_hidden_from_initial(
                &catalog,
                layer_id,
                hidden.clone(),
            )
            .expect("running bounded layer-ordered sparse MoE stage");
            let hidden_dim = sparse.output_rows;
            let initial_output_rows = *output_rows.get_or_insert(hidden_dim);
            assert_eq!(hidden_dim, initial_output_rows);
            assert_eq!(sparse.layer_id, layer_id);
            assert_eq!(sparse.route_count, GLM52_TOP_K);
            assert_eq!(sparse.routed_outputs.len(), hidden_dim);
            assert_eq!(sparse.shared_outputs.len(), hidden_dim);
            assert_eq!(sparse.layer_outputs.len(), hidden_dim);
            assert!(sparse.covers_full_top_k);
            assert!(sparse.shared_expert_executed);
            assert!(sparse.passed);

            let residual_before = hidden[..hidden_dim].to_vec();
            let residual_before_checksum = checksum_f64(&residual_before);
            let residual_step = sparse_moe_protocol_v2_residual_step(
                &sparse,
                &hidden,
                &targets,
                510_000 + layer_id as u64,
                TcpTransportConfig::default(),
                real_expertd_sparse_moe_dispatch_kind(),
            )
            .await
            .expect("executing live sparse residual step through real expertd");
            let stats = &residual_step.dispatch_stats;

            assert_eq!(stats.global_rows, 1);
            assert_eq!(stats.routes, GLM52_TOP_K);
            assert_eq!(stats.output_dim, hidden_dim);
            assert_eq!(stats.output_values, hidden_dim);
            assert_eq!(stats.contribution_counts.len(), 1);
            assert!(stats.contribution_counts[0] > 0);
            assert!(stats.hosts > 0);
            assert!(stats.hosts <= hosts.len());
            assert!(stats.request_wire_bytes > 0);
            assert!(stats.response_wire_bytes > 0);
            assert!(
                residual_step
                    .host_batch_set_evidence
                    .closes_live_expert_daemon_moe_gate
            );
            assert_eq!(
                residual_step
                    .host_batch_set_evidence
                    .protocol_v2_synthetic_executor,
                PROTOCOL_V2_REAL_NVFP4_CHECKPOINT_EXECUTOR
            );
            assert_eq!(
                residual_step.residual_add_backend,
                sparse.residual_add_backend
            );
            assert_eq!(residual_step.hidden_after.len(), GLM52_HIDDEN_SIZE);
            assert!(residual_step.hidden_after[..hidden_dim]
                .iter()
                .all(|value| value.is_finite()));
            hidden = residual_step.hidden_after;

            final_residual_checksum = residual_step.residual_after_checksum;
            assert!(residual_before_checksum.is_finite());
            assert!(residual_step.shared_output_checksum.is_finite());
            assert!(residual_step.residual_delta_checksum.is_finite());
            assert!(final_residual_checksum.is_finite());

            first_layer.get_or_insert(layer_id);
            last_layer = Some(layer_id);
            layer_count += 1;
            residual_adds += 1;
            shared_layers += 1;
            total_global_rows += stats.global_rows;
            total_host_rows += stats.host_rows;
            total_routes += stats.routes;
            total_output_values += stats.output_values;
            total_request_wire_bytes += stats.request_wire_bytes;
            total_response_wire_bytes += stats.response_wire_bytes;
            min_host_batches_per_layer = min_host_batches_per_layer.min(stats.hosts);
            max_host_batches_per_layer = max_host_batches_per_layer.max(stats.hosts);
            routed_output_checksum_sum += residual_step.routed_output_checksum;
            shared_output_checksum_sum += residual_step.shared_output_checksum;
            residual_delta_checksum_sum += residual_step.residual_delta_checksum;
        }

        assert_eq!(
            layer_count,
            GLM52_NUM_HIDDEN_LAYERS - GLM52_FIRST_K_DENSE_REPLACE
        );
        assert_eq!(first_layer, Some(GLM52_FIRST_K_DENSE_REPLACE));
        assert_eq!(last_layer, Some(GLM52_NUM_HIDDEN_LAYERS - 1));
        assert_eq!(residual_adds, layer_count);
        assert_eq!(shared_layers, layer_count);
        assert_eq!(total_global_rows, layer_count);
        assert_eq!(total_routes, layer_count * GLM52_TOP_K);
        assert_eq!(total_output_values, layer_count * output_rows.unwrap());
        assert!(routed_output_checksum_sum.is_finite());
        assert!(routed_output_checksum_sum != 0.0);
        assert!(shared_output_checksum_sum.is_finite());
        assert!(shared_output_checksum_sum != 0.0);
        assert!(residual_delta_checksum_sum.is_finite());
        assert!(residual_delta_checksum_sum != 0.0);
        assert!(final_residual_checksum.is_finite());

        eprintln!(
            "layer_ordered_sparse_stage_live_expertd_residual_chain daemon=run_expertd executor=protocol-v2-real-nvfp4-checkpoint-executor serving_layer_filter=none layers={} first_layer={} last_layer={} hosts={} host_names={:?} hidden_dim={} total_global_rows={} total_host_rows={} total_routes={} residual_adds={} shared_layers={} total_output_values={} min_host_batches_per_layer={} max_host_batches_per_layer={} total_request_wire_bytes={} total_response_wire_bytes={} routed_output_checksum_sum={} shared_output_checksum_sum={} residual_delta_checksum_sum={} final_residual_checksum={}",
            layer_count,
            first_layer.unwrap(),
            last_layer.unwrap(),
            hosts.len(),
            hosts,
            output_rows.unwrap(),
            total_global_rows,
            total_host_rows,
            total_routes,
            residual_adds,
            shared_layers,
            total_output_values,
            min_host_batches_per_layer,
            max_host_batches_per_layer,
            total_request_wire_bytes,
            total_response_wire_bytes,
            routed_output_checksum_sum,
            shared_output_checksum_sum,
            residual_delta_checksum_sum,
            final_residual_checksum
        );

        for server in servers {
            server.abort();
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    #[ignore = "loads full-width sparse outputs and starts real-expertd projection preload; run explicitly for live full-width chain coverage"]
    async fn layer_ordered_sparse_stage_live_expertd_outputs_mutate_full_width_residual_chain_when_available(
    ) {
        let Some(catalog) = load_real_catalog_or_skip() else {
            return;
        };
        let Some(loadplan_path) = load_full_loadplan_path_or_skip() else {
            return;
        };
        let catalog_path = real_catalog_path();
        let hosts = EXPERT_HOSTS
            .iter()
            .map(|host| (*host).to_owned())
            .collect::<Vec<_>>();
        let layer_ids = [GLM52_FIRST_K_DENSE_REPLACE, GLM52_FIRST_K_DENSE_REPLACE + 1];
        let mut hidden = deterministic_layer_ordered_daemon_hidden(layer_ids[0]);
        let mut total_global_rows = 0;
        let mut total_host_rows = 0;
        let mut total_routes = 0;
        let mut total_output_values = 0;
        let mut total_request_wire_bytes = 0;
        let mut total_response_wire_bytes = 0;
        let mut host_batches_per_layer = Vec::new();
        let mut contribution_counts_per_layer = Vec::new();
        let mut routed_output_checksums = Vec::new();
        let mut shared_output_checksums = Vec::new();
        let mut residual_delta_checksums = Vec::new();
        let mut residual_after_checksums = Vec::new();

        for layer_id in layer_ids {
            let (targets, servers) = start_unpinned_real_expertd_targets(
                &catalog_path,
                &loadplan_path,
                &hosts,
                Some(layer_id as u32),
            )
            .await;
            let sparse = real_sparse_mlp_shared_layer_full_output_hidden_from_initial(
                &catalog,
                layer_id,
                hidden.clone(),
            )
            .expect("running full-width layer-ordered sparse MoE stage");
            assert_eq!(sparse.layer_id, layer_id);
            assert_eq!(sparse.route_count, GLM52_TOP_K);
            assert_eq!(sparse.output_rows, GLM52_HIDDEN_SIZE);
            assert_eq!(sparse.routed_outputs.len(), GLM52_HIDDEN_SIZE);
            assert_eq!(sparse.shared_outputs.len(), GLM52_HIDDEN_SIZE);
            assert_eq!(sparse.layer_outputs.len(), GLM52_HIDDEN_SIZE);
            assert!(sparse.covers_full_top_k);
            assert!(sparse.shared_expert_executed);
            assert!(sparse.passed);

            let residual_before = hidden[..GLM52_HIDDEN_SIZE].to_vec();
            let residual_before_checksum = checksum_f64(&residual_before);
            let residual_step = sparse_moe_protocol_v2_residual_step(
                &sparse,
                &hidden,
                &targets,
                511_000 + layer_id as u64,
                TcpTransportConfig::default(),
                real_expertd_sparse_moe_dispatch_kind(),
            )
            .await
            .expect("executing full-width live sparse residual step through real expertd");
            let stats = &residual_step.dispatch_stats;

            assert_eq!(stats.global_rows, 1);
            assert_eq!(stats.routes, GLM52_TOP_K);
            assert_eq!(stats.output_dim, GLM52_HIDDEN_SIZE);
            assert_eq!(stats.output_values, GLM52_HIDDEN_SIZE);
            assert_eq!(stats.contribution_counts.len(), 1);
            assert!(stats.contribution_counts[0] > 0);
            assert!(stats.hosts > 0);
            assert!(stats.hosts <= hosts.len());
            assert!(stats.request_wire_bytes > GLM52_HIDDEN_BF16_BYTES);
            assert!(stats.response_wire_bytes > GLM52_HIDDEN_BF16_BYTES);
            assert!(
                residual_step
                    .host_batch_set_evidence
                    .closes_live_expert_daemon_moe_gate
            );
            assert_eq!(
                residual_step.residual_add_backend,
                sparse.residual_add_backend
            );
            assert_eq!(residual_step.hidden_after.len(), GLM52_HIDDEN_SIZE);
            assert!(residual_step
                .hidden_after
                .iter()
                .all(|value| value.is_finite()));
            hidden = residual_step.hidden_after;

            assert!(residual_before_checksum.is_finite());
            assert!(residual_step.shared_output_checksum.is_finite());
            assert!(residual_step.residual_delta_checksum.is_finite());
            assert!(residual_step.residual_after_checksum.is_finite());

            total_global_rows += stats.global_rows;
            total_host_rows += stats.host_rows;
            total_routes += stats.routes;
            total_output_values += stats.output_values;
            total_request_wire_bytes += stats.request_wire_bytes;
            total_response_wire_bytes += stats.response_wire_bytes;
            host_batches_per_layer.push(stats.hosts);
            contribution_counts_per_layer.push(stats.contribution_counts[0]);
            routed_output_checksums.push(residual_step.routed_output_checksum);
            shared_output_checksums.push(residual_step.shared_output_checksum);
            residual_delta_checksums.push(residual_step.residual_delta_checksum);
            residual_after_checksums.push(residual_step.residual_after_checksum);

            for server in servers {
                server.abort();
            }
        }

        assert_eq!(total_global_rows, layer_ids.len());
        assert_eq!(total_routes, layer_ids.len() * GLM52_TOP_K);
        assert_eq!(total_output_values, layer_ids.len() * GLM52_HIDDEN_SIZE);
        assert!(routed_output_checksums
            .iter()
            .all(|value| value.is_finite()));
        assert!(shared_output_checksums
            .iter()
            .all(|value| value.is_finite()));
        assert!(residual_delta_checksums
            .iter()
            .all(|value| value.is_finite()));
        assert!(residual_after_checksums
            .iter()
            .all(|value| value.is_finite()));
        assert!(routed_output_checksums.iter().any(|value| *value != 0.0));
        assert!(shared_output_checksums.iter().any(|value| *value != 0.0));
        assert!(residual_delta_checksums.iter().any(|value| *value != 0.0));

        eprintln!(
            "layer_ordered_sparse_stage_full_width_live_expertd_residual_chain daemon=run_expertd executor=protocol-v2-real-nvfp4-checkpoint-executor serving_layer_filter=per-layer layers={:?} hosts={} host_names={:?} hidden_dim={} total_global_rows={} total_host_rows={} total_routes={} residual_adds={} shared_layers={} total_output_values={} host_batches_per_layer={:?} contribution_counts_per_layer={:?} total_request_wire_bytes={} total_response_wire_bytes={} routed_output_checksums={:?} shared_output_checksums={:?} residual_delta_checksums={:?} residual_after_checksums={:?}",
            layer_ids,
            hosts.len(),
            hosts,
            GLM52_HIDDEN_SIZE,
            total_global_rows,
            total_host_rows,
            total_routes,
            layer_ids.len(),
            layer_ids.len(),
            total_output_values,
            host_batches_per_layer,
            contribution_counts_per_layer,
            total_request_wire_bytes,
            total_response_wire_bytes,
            routed_output_checksums,
            shared_output_checksums,
            residual_delta_checksums,
            residual_after_checksums
        );
    }

    #[test]
    fn real_checkpoint_layer_ordered_execution_probe_when_available() {
        let Some(catalog) = load_real_catalog_or_skip() else {
            return;
        };
        let cuda_graph_capture_required = cuda_native_library().is_ok();
        let _cuda_reference_override =
            cuda_graph_capture_required.then(|| cuda_reference_kernels_test_override(true));
        let graph_stats_before = if coordinator_cuda_reference_kernels_enabled() {
            coordinator_cuda_graph_test_stats().ok()
        } else {
            None
        };

        let probe = run_real_full_layer_ordered_execution_probe_with_mode(
            &catalog,
            layer_ordered_execution_mode(None),
        )
        .expect("running real layer-ordered residual execution trace probe");

        println!("{}", serde_json::to_string_pretty(&probe).unwrap());
        assert_eq!(
            probe.status,
            "numeric-real-layer-ordered-bounded-all-stage-residual-trace"
        );
        assert_eq!(probe.row_mode, "bounded");
        assert_eq!(
            probe.hidden_source,
            "real-embedding-token-hidden-carried-through-bounded-attention-mlp-order-for-all-layers"
        );
        assert_eq!(probe.input_token_id, Some(0));
        assert_eq!(probe.embedding_bytes_read, 12_288);
        assert!(probe.embedding_residual_checksum.unwrap().is_finite());
        assert_eq!(probe.layer_count, GLM52_NUM_HIDDEN_LAYERS);
        assert_eq!(probe.traced_layers, GLM52_NUM_HIDDEN_LAYERS);
        assert_eq!(probe.trace_steps, GLM52_NUM_HIDDEN_LAYERS * 2);
        assert_eq!(probe.attention_steps_executed, GLM52_NUM_HIDDEN_LAYERS);
        assert_eq!(probe.attention_steps_missing, 0);
        assert_eq!(probe.dense_mlp_steps_executed, GLM52_FIRST_K_DENSE_REPLACE);
        assert_eq!(
            probe.sparse_mlp_steps_executed,
            GLM52_NUM_HIDDEN_LAYERS - GLM52_FIRST_K_DENSE_REPLACE
        );
        assert_eq!(
            probe.shared_expert_steps_executed,
            GLM52_NUM_HIDDEN_LAYERS - GLM52_FIRST_K_DENSE_REPLACE
        );
        assert_eq!(probe.planned_residual_adds, GLM52_NUM_HIDDEN_LAYERS * 2);
        assert_eq!(
            probe.total_numeric_residual_adds,
            GLM52_NUM_HIDDEN_LAYERS * 2
        );
        assert_eq!(probe.residual_adds_missing, 0);
        assert_eq!(probe.residual_prefix_values, 4);
        assert_eq!(
            probe.routed_routes,
            (GLM52_NUM_HIDDEN_LAYERS - GLM52_FIRST_K_DENSE_REPLACE) * GLM52_TOP_K
        );
        assert!(probe.covers_all_dense_layers);
        assert!(probe.covers_all_sparse_layers);
        assert!(probe.covers_full_top_k);
        assert!(!probe.covers_full_output_rows);
        assert!(probe.carries_attention_into_dense);
        assert!(probe.carries_dense_into_sparse);
        assert!(!probe.full_residual_stream_complete);
        assert!(!probe.uses_full_model_residual);
        assert_eq!(
            probe.scheduler_rows.status,
            "layerwave-admitted-later-prefill-row-kv-binding"
        );
        assert_eq!(probe.scheduler_rows.source_mode, "prefill");
        assert_eq!(
            probe.scheduler_rows.selected_layerwaves,
            GLM52_NUM_HIDDEN_LAYERS
        );
        assert_eq!(probe.scheduler_rows.selected_rows, GLM52_NUM_HIDDEN_LAYERS);
        assert_eq!(
            probe.scheduler_rows.selected_prefill_rows,
            GLM52_NUM_HIDDEN_LAYERS
        );
        assert_eq!(probe.scheduler_rows.kv_read_blocks, GLM52_NUM_HIDDEN_LAYERS);
        assert_eq!(
            probe.scheduler_rows.committed_kv_writes,
            GLM52_NUM_HIDDEN_LAYERS * 2
        );
        assert_device_kv_status(
            probe.scheduler_rows.device_kv_status,
            probe.scheduler_rows.uses_device_kv_cache,
        );
        if probe.scheduler_rows.uses_device_kv_cache {
            assert_eq!(
                probe.scheduler_rows.device_kv_writes,
                GLM52_NUM_HIDDEN_LAYERS * 2
            );
            assert_eq!(
                probe.scheduler_rows.device_kv_reads,
                GLM52_NUM_HIDDEN_LAYERS
            );
            assert_eq!(
                probe.scheduler_rows.device_kv_bytes,
                probe.scheduler_rows.backed_kv_bytes + probe.scheduler_rows.backed_kv_bytes / 2
            );
        }
        assert!(probe.scheduler_rows.layer_order_verified);
        assert!(probe.scheduler_rows.uses_live_scheduler_rows);
        assert!(probe.scheduler_rows.passed);
        let graph_replay_expected = cuda_graph_capture_required;
        if let Some(before) = graph_stats_before {
            let after = coordinator_cuda_graph_test_stats()
                .expect("reading layer-ordered coordinator CUDA graph stats after probe");
            assert_eq!(after.slots, before.slots);
            assert!(after.captured_graphs >= before.captured_graphs);
            assert!(after.graph_captures >= before.graph_captures);
            assert!(
                after.graph_launches > before.graph_launches,
                "layer-ordered CUDA trace should launch retained coordinator graphs: before={before:?} after={after:?}"
            );
        }
        assert_eq!(
            probe.uses_graph_captured_coordinator_kernels,
            graph_replay_expected
        );
        assert_eq!(
            probe
                .execution_stepper
                .uses_graph_captured_coordinator_kernels,
            graph_replay_expected
        );
        assert_eq!(
            probe
                .completion_gates
                .uses_graph_captured_coordinator_kernels,
            graph_replay_expected
        );
        if graph_replay_expected {
            assert!(probe.coordinator_graph_slots > 0);
            assert!(probe.coordinator_graph_captured_graphs > 0);
            assert!(probe.coordinator_graph_captures > 0);
            assert!(probe.coordinator_graph_launches > 0);
        } else {
            assert_eq!(probe.coordinator_graph_launches, 0);
        }
        assert!(probe.completion_gates.uses_embedding_residual_input);
        assert!(probe.completion_gates.uses_live_scheduler_rows);
        assert!(!probe.completion_gates.uses_cuda_coordinator_kernels);
        assert_eq!(
            probe.completion_gates.missing_gate_count,
            if graph_replay_expected { 6 } else { 7 }
        );
        assert!(!probe
            .completion_gates
            .missing_gate_names
            .contains(&"uses_embedding_residual_input"));
        assert!(!probe
            .completion_gates
            .missing_gate_names
            .contains(&"uses_live_scheduler_rows"));
        assert!(probe
            .completion_gates
            .missing_gate_names
            .contains(&"uses_cuda_coordinator_kernels"));
        assert_eq!(
            probe
                .completion_gates
                .missing_gate_names
                .contains(&"uses_graph_captured_coordinator_kernels"),
            !graph_replay_expected
        );
        assert!(probe
            .completion_gates
            .missing_gate_names
            .contains(&"uses_full_context_mla_dsa_attention"));
        assert!(probe.final_residual_checksum.unwrap().is_finite());
        assert_eq!(
            probe.execution_stepper.status,
            "real-execution-stepper-bounded-all-stage-trace"
        );
        assert_eq!(probe.execution_stepper.row_mode, "bounded");
        assert_eq!(probe.execution_stepper.layer_count, GLM52_NUM_HIDDEN_LAYERS);
        assert_eq!(
            probe.execution_stepper.traced_layers,
            GLM52_NUM_HIDDEN_LAYERS
        );
        assert_eq!(
            probe.execution_stepper.trace_steps,
            GLM52_NUM_HIDDEN_LAYERS * 2
        );
        assert_eq!(
            probe.execution_stepper.total_numeric_residual_adds,
            GLM52_NUM_HIDDEN_LAYERS * 2
        );
        assert_eq!(probe.execution_stepper.residual_adds_missing, 0);
        assert_eq!(probe.execution_stepper.residual_prefix_values, 4);
        assert_eq!(
            probe.execution_stepper.stage_sources_recorded,
            GLM52_NUM_HIDDEN_LAYERS * 2
        );
        assert_eq!(
            probe.execution_stepper.stage_statuses_recorded,
            GLM52_NUM_HIDDEN_LAYERS * 2
        );
        assert_eq!(
            probe.execution_stepper.real_stage_count,
            GLM52_NUM_HIDDEN_LAYERS * 2
        );
        assert_eq!(probe.execution_stepper.synthetic_stage_count, 0);
        assert_eq!(probe.execution_stepper.provisional_stage_count, 0);
        assert_eq!(probe.execution_stepper.blocked_stage_count, 0);
        assert_execution_stepper_coordinator_backend_counts(
            &probe.execution_stepper,
            GLM52_NUM_HIDDEN_LAYERS * 2,
            GLM52_NUM_HIDDEN_LAYERS,
        );
        assert_eq!(
            probe.execution_stepper.stages_with_numeric_checksums,
            GLM52_NUM_HIDDEN_LAYERS * 2
        );
        assert_eq!(
            probe.execution_stepper.total_numeric_checksum_fields,
            GLM52_NUM_HIDDEN_LAYERS * 2 * 3
        );
        assert_eq!(probe.execution_stepper.numeric_checksum_fields_per_stage, 3);
        assert_eq!(
            probe.execution_stepper.stages_with_tensor_artifacts,
            GLM52_NUM_HIDDEN_LAYERS * 2
        );
        let sparse_tensor_artifacts_per_stage = 2 + GLM52_TOP_K * 9 + 3;
        assert_eq!(
            probe.execution_stepper.total_tensor_artifacts,
            GLM52_NUM_HIDDEN_LAYERS * 8
                + GLM52_DSA_INDEXER_LAYERS * 5
                + GLM52_FIRST_K_DENSE_REPLACE * 4
                + (GLM52_NUM_HIDDEN_LAYERS - GLM52_FIRST_K_DENSE_REPLACE)
                    * sparse_tensor_artifacts_per_stage
        );
        assert_eq!(
            probe.execution_stepper.attention_tensor_artifacts_per_stage,
            13
        );
        assert_eq!(
            probe.execution_stepper.dense_mlp_tensor_artifacts_per_stage,
            4
        );
        assert_eq!(
            probe
                .execution_stepper
                .sparse_mlp_tensor_artifacts_per_stage,
            sparse_tensor_artifacts_per_stage
        );
        assert!(probe.execution_stepper.covers_all_layers);
        assert!(probe.execution_stepper.covers_all_dense_layers);
        assert!(probe.execution_stepper.covers_all_sparse_layers);
        assert!(probe.execution_stepper.covers_full_top_k);
        assert!(probe.execution_stepper.stage_order_verified);
        assert!(!probe.execution_stepper.covers_full_output_rows);
        assert!(!probe.execution_stepper.full_residual_stream_complete);
        assert!(!probe.execution_stepper.uses_full_model_residual);
        assert_eq!(
            probe.execution_stepper.bounded_attention_oracle.status,
            "retired"
        );
        assert_eq!(
            probe.execution_stepper.bounded_attention_oracle.source,
            "retired-stepper-validation-artifacts"
        );
        assert_eq!(
            probe
                .execution_stepper
                .bounded_attention_oracle
                .stepper_stage,
            "retired"
        );
        assert!(!probe.execution_stepper.bounded_attention_oracle.passed);
        assert_eq!(
            probe
                .execution_stepper
                .bounded_attention_oracle
                .fixture_count,
            0
        );
        assert_eq!(
            probe
                .execution_stepper
                .bounded_attention_oracle
                .main_mla_fixture_count,
            0
        );
        assert_eq!(
            probe
                .execution_stepper
                .bounded_attention_oracle
                .dsa_indexer_fixture_count,
            0
        );
        assert_eq!(
            probe
                .execution_stepper
                .bounded_attention_oracle
                .bounded_mlp_fixture_count,
            0
        );
        assert_eq!(
            probe
                .execution_stepper
                .bounded_attention_oracle
                .bounded_sparse_mlp_fixture_count,
            0
        );
        assert_eq!(
            probe
                .execution_stepper
                .bounded_attention_oracle
                .bounded_terminal_fixture_count,
            0
        );
        assert!(
            !probe
                .execution_stepper
                .bounded_attention_oracle
                .covers_real_checkpoint
        );
        assert!(
            !probe
                .execution_stepper
                .bounded_attention_oracle
                .validates_rope_modes
        );
        assert!(
            !probe
                .execution_stepper
                .bounded_attention_oracle
                .validates_dsa_candidate_selection
        );
        assert!(
            !probe
                .execution_stepper
                .bounded_attention_oracle
                .validates_rmsnorm
        );
        assert!(
            !probe
                .execution_stepper
                .bounded_attention_oracle
                .validates_gated_mlp
        );
        assert!(
            !probe
                .execution_stepper
                .bounded_attention_oracle
                .validates_router_topk
        );
        assert!(
            !probe
                .execution_stepper
                .bounded_attention_oracle
                .validates_sparse_routed_mlp
        );
        assert!(
            !probe
                .execution_stepper
                .bounded_attention_oracle
                .validates_shared_mlp
        );
        assert!(
            !probe
                .execution_stepper
                .bounded_attention_oracle
                .validates_embedding_lookup
        );
        assert!(
            !probe
                .execution_stepper
                .bounded_attention_oracle
                .validates_lm_head_argmax
        );
        assert!(probe
            .execution_stepper
            .bounded_attention_oracle
            .mlp_output_checksum
            .is_none());
        assert!(probe
            .execution_stepper
            .bounded_attention_oracle
            .sparse_mlp_output_checksum
            .is_none());
        assert_eq!(
            probe
                .execution_stepper
                .bounded_attention_oracle
                .lm_head_sampled_token_id,
            None
        );
        assert_eq!(
            probe
                .execution_stepper
                .bounded_attention_oracle
                .dsa_selected_indices,
            Vec::<usize>::new()
        );
        assert_eq!(
            probe
                .execution_stepper
                .bounded_attention_oracle
                .skipped_reason
                .as_deref(),
            Some("stepper validation artifacts were removed after phase0 live token output completed")
        );
        assert!(
            probe
                .execution_stepper
                .completion_gates
                .uses_embedding_residual_input
        );
        assert_eq!(probe.step_summaries[0].stage, "attention");
        assert!(probe.step_summaries[0].executed);
        assert_eq!(
            probe.step_summaries[0].stage_source,
            expected_prefix_attention_stage_source()
        );
        assert_eq!(probe.step_summaries[0].stage_status, "real");
        assert_eq!(probe.step_summaries[0].tensor_artifacts.len(), 13);
        assert_step_tensor(
            &probe.step_summaries[0],
            "model.layers.0.input_layernorm.weight",
            DType::Bf16,
            TensorRole::Norm,
            None,
            true,
        );
        assert_step_tensor(
            &probe.step_summaries[0],
            "model.layers.0.self_attn.q_b_proj.weight",
            DType::Bf16,
            TensorRole::Attention,
            Some(4),
            false,
        );
        assert_step_tensor(
            &probe.step_summaries[0],
            "model.layers.0.self_attn.indexer.k_norm.weight",
            DType::Bf16,
            TensorRole::Attention,
            None,
            true,
        );
        assert_step_tensor(
            &probe.step_summaries[0],
            "model.layers.0.self_attn.indexer.wq_b.weight",
            DType::Bf16,
            TensorRole::Attention,
            Some(GLM52_DSA_INDEX_HEAD_DIM),
            false,
        );
        assert_eq!(probe.step_summaries[1].stage, "dense_mlp");
        assert_eq!(probe.step_summaries[1].layer_id, 0);
        assert_eq!(
            probe.step_summaries[1].stage_source,
            expected_prefix_dense_mlp_stage_source()
        );
        assert_eq!(probe.step_summaries[1].stage_status, "real");
        assert_eq!(probe.step_summaries[1].tensor_artifacts.len(), 4);
        assert_step_tensor(
            &probe.step_summaries[1],
            "model.layers.0.mlp.gate_proj.weight",
            DType::Bf16,
            TensorRole::DenseMlp,
            Some(8),
            false,
        );
        assert_eq!(probe.step_summaries[2].stage, "attention");
        assert_eq!(probe.step_summaries[2].layer_id, 1);
        assert!(probe.step_summaries[2].executed);
        assert!(probe.step_summaries[2].missing_reason.is_none());
        assert_eq!(probe.step_summaries[3].stage, "dense_mlp");
        assert_eq!(probe.step_summaries[3].layer_id, 1);
        assert_eq!(probe.step_summaries[4].stage, "attention");
        assert_eq!(probe.step_summaries[4].layer_id, 2);
        assert!(probe.step_summaries[4].executed);
        assert_eq!(probe.step_summaries[6].stage, "attention");
        assert_eq!(
            probe.step_summaries[6].layer_id,
            GLM52_FIRST_K_DENSE_REPLACE
        );
        assert!(probe.step_summaries[6].executed);
        assert_eq!(probe.step_summaries[6].tensor_artifacts.len(), 8);
        assert_eq!(probe.step_summaries[7].stage, "sparse_moe_mlp");
        assert_eq!(
            probe.step_summaries[7].layer_id,
            GLM52_FIRST_K_DENSE_REPLACE
        );
        assert!(probe.step_summaries[7].includes_shared_expert);
        assert_eq!(
            probe.step_summaries[7].stage_source,
            expected_prefix_sparse_moe_stage_source()
        );
        assert_eq!(probe.step_summaries[7].stage_status, "real");
        assert_eq!(
            probe.step_summaries[7].tensor_artifacts.len(),
            sparse_tensor_artifacts_per_stage
        );
        assert_step_tensor(
            &probe.step_summaries[7],
            "model.layers.3.mlp.gate.weight",
            DType::Bf16,
            TensorRole::Router,
            None,
            true,
        );
        assert_step_tensor(
            &probe.step_summaries[7],
            "model.layers.3.mlp.shared_experts.gate_proj.weight",
            DType::Bf16,
            TensorRole::SharedExpert,
            Some(4),
            false,
        );
        assert!(probe.step_summaries[7]
            .tensor_artifacts
            .iter()
            .any(|tensor| tensor.name.contains(".mlp.experts.")
                && tensor.name.ends_with(".weight_scale")
                && tensor.dtype == DType::F8E4M3
                && tensor.role == TensorRole::RoutedExpert
                && tensor.is_quantization_metadata
                && tensor.rows_loaded == Some(4)
                && !tensor.full_tensor_loaded));
        assert_eq!(
            probe.step_summaries.last().unwrap().layer_id,
            GLM52_NUM_HIDDEN_LAYERS - 1
        );
        assert_eq!(probe.step_summaries.last().unwrap().stage, "sparse_moe_mlp");
        assert!(probe.passed);
    }

    #[test]
    #[ignore = "loads full-output dense and sparse MLP rows across all 78 layers"]
    fn real_checkpoint_layer_ordered_full_output_mlp_probe_when_available() {
        let Some(catalog) = load_real_catalog_or_skip() else {
            return;
        };

        let probe = run_real_full_layer_ordered_execution_probe_with_mode(
            &catalog,
            layer_ordered_execution_mode(Some("full-output-mlp")),
        )
        .expect("running real layer-ordered full-output MLP residual execution trace probe");

        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "status": probe.status,
                "row_mode": probe.row_mode,
                "layers": probe.traced_layers,
                "trace_steps": probe.trace_steps,
                "numeric_residual_adds": probe.total_numeric_residual_adds,
                "planned_residual_adds": probe.planned_residual_adds,
                "residual_prefix_values": probe.residual_prefix_values,
                "covers_all_dense_layers": probe.covers_all_dense_layers,
                "covers_all_sparse_layers": probe.covers_all_sparse_layers,
                "covers_full_top_k": probe.covers_full_top_k,
                "covers_full_output_rows": probe.covers_full_output_rows,
                "full_residual_stream_complete": probe.full_residual_stream_complete,
                "uses_full_model_residual": probe.uses_full_model_residual,
                "execution_stepper_status": probe.execution_stepper.status,
                "execution_stepper_row_mode": probe.execution_stepper.row_mode,
                "execution_stepper_residual_prefix_values": probe.execution_stepper.residual_prefix_values,
                "execution_stepper_covers_full_output_rows": probe.execution_stepper.covers_full_output_rows,
                "first_attention_output_rows": probe.step_summaries[0].output_rows,
                "first_dense_output_rows": probe.step_summaries[1].output_rows,
                "first_sparse_output_rows": probe.step_summaries[7].output_rows,
                "last_stage": probe.step_summaries.last().unwrap().stage,
                "last_output_rows": probe.step_summaries.last().unwrap().output_rows,
                "final_residual_checksum": probe.final_residual_checksum,
            }))
            .unwrap()
        );
        assert_eq!(
            probe.status,
            "numeric-real-layer-ordered-full-output-mlp-bounded-attention-residual-trace"
        );
        assert_eq!(probe.row_mode, "full-output-mlp");
        assert_eq!(probe.layer_count, GLM52_NUM_HIDDEN_LAYERS);
        assert_eq!(probe.traced_layers, GLM52_NUM_HIDDEN_LAYERS);
        assert_eq!(probe.trace_steps, GLM52_NUM_HIDDEN_LAYERS * 2);
        assert_eq!(probe.attention_steps_executed, GLM52_NUM_HIDDEN_LAYERS);
        assert_eq!(probe.attention_steps_missing, 0);
        assert_eq!(probe.dense_mlp_steps_executed, GLM52_FIRST_K_DENSE_REPLACE);
        assert_eq!(
            probe.sparse_mlp_steps_executed,
            GLM52_NUM_HIDDEN_LAYERS - GLM52_FIRST_K_DENSE_REPLACE
        );
        assert_eq!(probe.planned_residual_adds, GLM52_NUM_HIDDEN_LAYERS * 2);
        assert_eq!(
            probe.total_numeric_residual_adds,
            GLM52_NUM_HIDDEN_LAYERS * 2
        );
        assert_eq!(probe.residual_adds_missing, 0);
        assert_eq!(probe.residual_prefix_values, GLM52_HIDDEN_SIZE);
        assert!(probe.covers_all_dense_layers);
        assert!(probe.covers_all_sparse_layers);
        assert!(probe.covers_full_top_k);
        assert!(!probe.covers_full_output_rows);
        assert!(probe.carries_attention_into_dense);
        assert!(probe.carries_dense_into_sparse);
        assert!(!probe.full_residual_stream_complete);
        assert!(!probe.uses_full_model_residual);
        assert!(probe.final_residual_checksum.unwrap().is_finite());
        assert_eq!(
            probe.execution_stepper.status,
            "real-execution-stepper-full-output-mlp-bounded-attention-trace"
        );
        assert_eq!(probe.execution_stepper.row_mode, "full-output-mlp");
        assert_eq!(
            probe.execution_stepper.residual_prefix_values,
            GLM52_HIDDEN_SIZE
        );
        assert!(probe.execution_stepper.covers_all_layers);
        assert!(probe.execution_stepper.covers_all_dense_layers);
        assert!(probe.execution_stepper.covers_all_sparse_layers);
        assert!(probe.execution_stepper.covers_full_top_k);
        assert!(probe.execution_stepper.stage_order_verified);
        assert!(!probe.execution_stepper.covers_full_output_rows);
        assert!(!probe.execution_stepper.full_residual_stream_complete);
        assert!(!probe.execution_stepper.uses_full_model_residual);
        assert_eq!(probe.step_summaries[0].stage, "attention");
        assert_eq!(probe.step_summaries[0].output_rows, 4);
        assert_eq!(probe.step_summaries[1].stage, "dense_mlp");
        assert_eq!(probe.step_summaries[1].output_rows, GLM52_HIDDEN_SIZE);
        assert_eq!(
            probe.step_summaries[1].stage_source,
            "real-checkpoint-bf16-dense-mlp-full-output-cpu-reference-rmsnorm-bf16-cpu-reference-linear-bf16-cpu-reference-silu-gated-mlp-bf16-cpu-reference-residual-add-bf16"
        );
        assert_step_tensor(
            &probe.step_summaries[1],
            "model.layers.0.mlp.down_proj.weight",
            DType::Bf16,
            TensorRole::DenseMlp,
            Some(GLM52_HIDDEN_SIZE),
            false,
        );
        assert_eq!(probe.step_summaries[7].stage, "sparse_moe_mlp");
        assert_eq!(probe.step_summaries[7].output_rows, GLM52_HIDDEN_SIZE);
        assert_eq!(
            probe.step_summaries[7].stage_source,
            "real-checkpoint-nvfp4-routed-plus-bf16-shared-moe-full-output-cpu-reference-rmsnorm-bf16-cpu-reference-router-topk-bf16-cpu-reference-shared-silu-gated-mlp-bf16-cpu-reference-residual-add-bf16"
        );
        assert_step_tensor(
            &probe.step_summaries[7],
            "model.layers.3.mlp.shared_experts.down_proj.weight",
            DType::Bf16,
            TensorRole::SharedExpert,
            Some(GLM52_HIDDEN_SIZE),
            false,
        );
        assert_eq!(
            probe.step_summaries.last().unwrap().layer_id,
            GLM52_NUM_HIDDEN_LAYERS - 1
        );
        assert_eq!(probe.step_summaries.last().unwrap().stage, "sparse_moe_mlp");
        assert_eq!(
            probe.step_summaries.last().unwrap().output_rows,
            GLM52_HIDDEN_SIZE
        );
        assert!(probe.passed);
    }

    #[test]
    #[ignore = "loads hidden-width attention rows across a real layer boundary"]
    fn real_checkpoint_layer_ordered_full_output_attention_dense_boundary_when_available() {
        let Some(catalog) = load_real_catalog_or_skip() else {
            return;
        };

        let attention0 = real_full_attention_residual_full_output_hidden(&catalog)
            .expect("running layer-0 full-output attention residual");
        let dense0 = real_full_dense_layer_full_output_hidden_from_initial(
            &catalog,
            attention0.layer_id,
            attention0.hidden,
        )
        .expect("running layer-0 full-output dense residual from full-output attention hidden");
        let dense0_hidden_prefix_checksum = checksum_f64(&dense0.hidden[..GLM52_HIDDEN_SIZE]);
        let attention1 = attention_hidden_for_layer_from_initial(
            &catalog,
            1,
            dense0.hidden,
            None,
            layer_ordered_execution_mode(Some("full-output-attention-mlp")),
            Vec::new(),
        )
        .expect("running layer-1 full-output attention residual from layer-0 dense hidden");

        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "status": "numeric-real-layer-ordered-full-output-attention-dense-boundary",
                "attention0_layer": attention0.layer_id,
                "attention0_output_rows": attention0.residual_prefix_values,
                "attention0_residual_adds": attention0.residual_adds,
                "dense0_layer": dense0.layer_id,
                "dense0_output_rows": dense0.output_rows,
                "dense0_residual_adds": dense0.residual_adds,
                "attention1_layer": attention1.layer_id,
                "attention1_output_rows": attention1.residual_prefix_values,
                "attention1_residual_adds": attention1.residual_adds,
                "attention1_initial_matches_dense0": approx_eq_f64(
                    attention1.initial_residual_checksum,
                    dense0_hidden_prefix_checksum,
                ),
                "attention0_projection_bytes": attention0.projection_bytes_read,
                "attention0_o_proj_bytes": attention0.o_proj_bytes_read,
                "attention1_projection_bytes": attention1.projection_bytes_read,
                "attention1_o_proj_bytes": attention1.o_proj_bytes_read,
                "dense0_weight_bytes": dense0.weight_bytes_read,
                "attention1_final_checksum": attention1.final_residual_checksum,
                "uses_full_model_residual": false,
                "includes_mla_softmax": attention0.includes_mla_softmax || attention1.includes_mla_softmax,
            }))
            .unwrap()
        );

        assert_eq!(attention0.layer_id, 0);
        assert_eq!(attention0.residual_prefix_values, GLM52_HIDDEN_SIZE);
        assert_eq!(attention0.residual_adds, 1);
        assert!(attention0.includes_causal_softmax);
        assert!(!attention0.includes_mla_softmax);
        assert_eq!(dense0.layer_id, 0);
        assert_eq!(dense0.output_rows, GLM52_HIDDEN_SIZE);
        assert_eq!(dense0.residual_adds, 1);
        assert!(dense0.passed);
        assert_eq!(attention1.layer_id, 1);
        assert_eq!(attention1.residual_prefix_values, GLM52_HIDDEN_SIZE);
        assert_eq!(attention1.residual_adds, 1);
        assert!(attention1.includes_causal_softmax);
        assert!(!attention1.includes_mla_softmax);
        assert!(approx_eq_f64(
            attention1.initial_residual_checksum,
            dense0_hidden_prefix_checksum
        ));
        assert_eq!(attention0.input_norm_bytes_read, 12_288);
        assert_eq!(attention1.input_norm_bytes_read, 12_288);
        assert_eq!(attention0.o_proj_bytes_read, 201_326_592);
        assert_eq!(attention1.o_proj_bytes_read, 201_326_592);
        assert!(attention0.projection_bytes_read > 260_000_000);
        assert!(attention1.projection_bytes_read > 260_000_000);
        assert_eq!(dense0.weight_bytes_read, 151_191_552);
        assert!(attention1.final_residual_checksum.is_finite());
    }

    #[test]
    #[ignore = "loads hidden-width attention, dense, and sparse MLP rows across all 78 layers"]
    fn real_checkpoint_layer_ordered_full_output_attention_mlp_probe_when_available() {
        let Some(catalog) = load_real_catalog_or_skip() else {
            return;
        };

        let probe = run_real_full_layer_ordered_execution_probe_with_mode(
            &catalog,
            layer_ordered_execution_mode(Some("full-output-attention-mlp")),
        )
        .expect(
            "running real layer-ordered full-output attention/MLP residual execution trace probe",
        );

        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "status": probe.status,
                "row_mode": probe.row_mode,
                "hidden_source": probe.hidden_source,
                "layers": probe.traced_layers,
                "input_token_id": probe.input_token_id,
                "embedding_bytes_read": probe.embedding_bytes_read,
                "embedding_residual_checksum": probe.embedding_residual_checksum,
                "trace_steps": probe.trace_steps,
                "numeric_residual_adds": probe.total_numeric_residual_adds,
                "planned_residual_adds": probe.planned_residual_adds,
                "residual_prefix_values": probe.residual_prefix_values,
                "covers_full_output_rows": probe.covers_full_output_rows,
                "full_residual_stream_complete": probe.full_residual_stream_complete,
                "uses_full_model_residual": probe.uses_full_model_residual,
                "terminal_lm_head_sampling": probe.terminal_lm_head_sampling,
                "completion_gates": probe.completion_gates,
                "execution_stepper_status": probe.execution_stepper.status,
                "execution_stepper_completion_gates": probe.execution_stepper.completion_gates,
                "first_attention_output_rows": probe.step_summaries[0].output_rows,
                "first_dense_output_rows": probe.step_summaries[1].output_rows,
                "first_sparse_output_rows": probe.step_summaries[7].output_rows,
                "last_stage": probe.step_summaries.last().unwrap().stage,
                "last_output_rows": probe.step_summaries.last().unwrap().output_rows,
                "final_residual_checksum": probe.final_residual_checksum,
            }))
            .unwrap()
        );

        assert_eq!(
            probe.status,
            "numeric-real-layer-ordered-full-output-attention-mlp-residual-trace"
        );
        assert_eq!(probe.row_mode, "full-output-attention-mlp");
        assert_eq!(
            probe.hidden_source,
            "real-embedding-token-hidden-carried-through-full-output-attention-mlp-order-for-all-layers"
        );
        assert_eq!(probe.input_token_id, Some(0));
        assert_eq!(probe.embedding_bytes_read, 12_288);
        assert!(probe.embedding_residual_checksum.unwrap().is_finite());
        assert_eq!(probe.layer_count, GLM52_NUM_HIDDEN_LAYERS);
        assert_eq!(probe.traced_layers, GLM52_NUM_HIDDEN_LAYERS);
        assert_eq!(probe.trace_steps, GLM52_NUM_HIDDEN_LAYERS * 2);
        assert_eq!(probe.attention_steps_executed, GLM52_NUM_HIDDEN_LAYERS);
        assert_eq!(probe.attention_steps_missing, 0);
        assert_eq!(probe.planned_residual_adds, GLM52_NUM_HIDDEN_LAYERS * 2);
        assert_eq!(
            probe.total_numeric_residual_adds,
            GLM52_NUM_HIDDEN_LAYERS * 2
        );
        assert_eq!(probe.residual_adds_missing, 0);
        assert_eq!(probe.residual_prefix_values, GLM52_HIDDEN_SIZE);
        assert!(probe.covers_full_output_rows);
        assert!(!probe.full_residual_stream_complete);
        assert!(!probe.uses_full_model_residual);
        assert!(probe.completion_gates.covers_full_output_rows);
        assert!(probe.completion_gates.uses_embedding_residual_input);
        assert!(probe.completion_gates.uses_live_scheduler_rows);
        assert!(probe.completion_gates.uses_cuda_coordinator_kernels);
        assert!(!probe.completion_gates.uses_real_lm_head_sampling_residual);
        assert_eq!(probe.completion_gates.missing_gate_count, 4);
        assert_eq!(
            probe.completion_gates.missing_gate_names,
            vec![
                "uses_full_context_mla_dsa_attention",
                "uses_live_expert_daemon_moe",
                "uses_real_lm_head_sampling_residual",
                "uses_full_model_residual"
            ]
        );
        assert_eq!(
            probe.terminal_lm_head_sampling.status,
            "numeric-real-layer-ordered-lm-head-chunk"
        );
        assert!(probe.terminal_lm_head_sampling.passed);
        assert_eq!(
            probe.terminal_lm_head_sampling.hidden_dim,
            GLM52_HIDDEN_SIZE
        );
        assert_eq!(
            probe.terminal_lm_head_sampling.hidden_source,
            "real-embedding-token-hidden-carried-through-full-output-attention-mlp-order-for-all-layers"
        );
        assert!(probe.terminal_lm_head_sampling.uses_real_lm_head);
        assert!(
            probe
                .terminal_lm_head_sampling
                .uses_layer_ordered_full_output_residual
        );
        assert!(!probe.terminal_lm_head_sampling.covers_full_vocabulary);
        assert_eq!(probe.terminal_lm_head_sampling.rows_scored, 1024);
        assert_eq!(
            probe.terminal_lm_head_sampling.logits_evaluated,
            probe.terminal_lm_head_sampling.rows_scored
        );
        assert!(probe
            .terminal_lm_head_sampling
            .top_logit
            .unwrap()
            .is_finite());
        assert_eq!(
            probe.execution_stepper.status,
            "real-execution-stepper-full-output-attention-mlp-trace"
        );
        assert!(probe.execution_stepper.covers_full_output_rows);
        assert_execution_stepper_coordinator_backend_counts(
            &probe.execution_stepper,
            GLM52_NUM_HIDDEN_LAYERS * 2,
            GLM52_NUM_HIDDEN_LAYERS * 2,
        );
        assert!(!probe.execution_stepper.full_residual_stream_complete);
        assert!(
            probe
                .execution_stepper
                .completion_gates
                .covers_full_output_rows
        );
        assert!(
            probe
                .execution_stepper
                .completion_gates
                .uses_embedding_residual_input
        );
        assert!(
            probe
                .execution_stepper
                .completion_gates
                .uses_live_scheduler_rows
        );
        assert!(
            probe
                .execution_stepper
                .completion_gates
                .uses_cuda_coordinator_kernels
        );
        assert_eq!(
            probe.execution_stepper.completion_gates.missing_gate_count,
            4
        );
        assert_eq!(probe.step_summaries[0].stage, "attention");
        assert_eq!(probe.step_summaries[0].output_rows, GLM52_HIDDEN_SIZE);
        assert_eq!(
            probe.step_summaries[0].stage_source,
            expected_full_output_attention_stage_source()
        );
        assert_eq!(probe.step_summaries[1].output_rows, GLM52_HIDDEN_SIZE);
        assert_eq!(probe.step_summaries[7].output_rows, GLM52_HIDDEN_SIZE);
        assert_eq!(
            probe.step_summaries.last().unwrap().output_rows,
            GLM52_HIDDEN_SIZE
        );
        assert!(probe.passed);
    }

    #[test]
    #[ignore = "loads hidden-width MLA/RoPE attention, dense, and sparse MLP rows across all 78 layers"]
    fn real_checkpoint_layer_ordered_full_output_mla_rope_attention_mlp_probe_when_available() {
        let Some(catalog) = load_real_catalog_or_skip() else {
            return;
        };

        let probe = run_real_full_layer_ordered_execution_probe_with_mode(
            &catalog,
            layer_ordered_execution_mode(Some("full-output-mla-rope-attention-mlp")),
        )
        .expect(
            "running real layer-ordered full-output MLA/RoPE attention/MLP residual trace probe",
        );

        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "status": probe.status,
                "row_mode": probe.row_mode,
                "hidden_source": probe.hidden_source,
                "layers": probe.traced_layers,
                "trace_steps": probe.trace_steps,
                "numeric_residual_adds": probe.total_numeric_residual_adds,
                "planned_residual_adds": probe.planned_residual_adds,
                "residual_prefix_values": probe.residual_prefix_values,
                "covers_full_output_rows": probe.covers_full_output_rows,
                "full_residual_stream_complete": probe.full_residual_stream_complete,
                "completion_gates": probe.completion_gates,
                "terminal_lm_head_sampling": probe.terminal_lm_head_sampling.status,
                "execution_stepper_status": probe.execution_stepper.status,
                "first_attention_source": probe.step_summaries[0].stage_source,
                "first_attention_output_rows": probe.step_summaries[0].output_rows,
                "first_dense_output_rows": probe.step_summaries[1].output_rows,
                "first_sparse_output_rows": probe.step_summaries[7].output_rows,
                "last_output_rows": probe.step_summaries.last().unwrap().output_rows,
                "final_residual_checksum": probe.final_residual_checksum,
            }))
            .unwrap()
        );

        assert_eq!(
            probe.status,
            "numeric-real-layer-ordered-full-output-mla-rope-attention-mlp-residual-trace"
        );
        assert_eq!(probe.row_mode, "full-output-mla-rope-attention-mlp");
        assert_eq!(
            probe.hidden_source,
            "real-embedding-token-hidden-carried-through-full-output-mla-rope-attention-mlp-order-for-all-layers"
        );
        assert_eq!(probe.traced_layers, GLM52_NUM_HIDDEN_LAYERS);
        assert_eq!(probe.trace_steps, GLM52_NUM_HIDDEN_LAYERS * 2);
        assert_eq!(
            probe.total_numeric_residual_adds,
            GLM52_NUM_HIDDEN_LAYERS * 2
        );
        assert_eq!(probe.residual_adds_missing, 0);
        assert_eq!(probe.residual_prefix_values, GLM52_HIDDEN_SIZE);
        assert!(probe.covers_full_output_rows);
        assert!(!probe.full_residual_stream_complete);
        assert!(!probe.uses_full_model_residual);
        assert!(probe.completion_gates.covers_full_output_rows);
        assert!(probe.completion_gates.uses_full_context_mla_dsa_attention);
        assert!(!probe.completion_gates.uses_live_expert_daemon_moe);
        assert_eq!(
            probe.execution_stepper.status,
            "real-execution-stepper-full-output-mla-rope-attention-mlp-trace"
        );
        assert_eq!(
            probe.step_summaries[0].stage_source,
            expected_full_output_mla_rope_attention_stage_source()
        );
        assert_eq!(probe.step_summaries[0].output_rows, GLM52_HIDDEN_SIZE);
        assert_eq!(probe.step_summaries[1].output_rows, GLM52_HIDDEN_SIZE);
        assert_eq!(probe.step_summaries[7].output_rows, GLM52_HIDDEN_SIZE);
        assert_eq!(
            probe.step_summaries.last().unwrap().output_rows,
            GLM52_HIDDEN_SIZE
        );
        assert!(probe.passed);
    }

    fn assert_step_tensor(
        step: &super::RealFullLayerOrderedResidualExecutionStep,
        name: &str,
        dtype: DType,
        role: TensorRole,
        rows_loaded: Option<usize>,
        full_tensor_loaded: bool,
    ) {
        assert_artifact(
            &step.tensor_artifacts,
            name,
            dtype,
            role,
            rows_loaded,
            full_tensor_loaded,
        );
    }

    fn assert_artifact(
        artifacts: &[super::RealFullResidualExecutionTensorArtifact],
        name: &str,
        dtype: DType,
        role: TensorRole,
        rows_loaded: Option<usize>,
        full_tensor_loaded: bool,
    ) {
        let tensor = artifacts
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap_or_else(|| panic!("missing tensor artifact {name}"));
        assert_eq!(tensor.dtype, dtype);
        assert_eq!(tensor.role, role);
        assert_eq!(tensor.rows_loaded, rows_loaded);
        assert_eq!(tensor.full_tensor_loaded, full_tensor_loaded);
        assert_eq!(tensor.rank, tensor.shape.len());
        assert!(tensor.byte_length > 0);
    }

    fn sparse_selected_route_fixture() -> RealFullSparseMlpSharedLayerHidden {
        let layer_id = GLM52_FIRST_K_DENSE_REPLACE;
        RealFullSparseMlpSharedLayerHidden {
            hidden: Vec::new(),
            device_hidden: None,
            expert_input_hidden_bf16_payload: patterned_bf16_hidden_payload(),
            layer_id,
            route_count: 3,
            routes: vec![
                RealFullSparseMoeRoute {
                    rank: 0,
                    expert_id: 17,
                    owner: "spark-0".to_owned(),
                    score: 0.75,
                    corrected_score: 0.8,
                    normalized_weight: 0.5,
                },
                RealFullSparseMoeRoute {
                    rank: 1,
                    expert_id: 33,
                    owner: "spark-1".to_owned(),
                    score: 0.5,
                    corrected_score: 0.7,
                    normalized_weight: 0.25,
                },
                RealFullSparseMoeRoute {
                    rank: 2,
                    expert_id: 65,
                    owner: "spark-0".to_owned(),
                    score: 0.25,
                    corrected_score: 0.3,
                    normalized_weight: 0.25,
                },
            ],
            routed_outputs: vec![0.0; 4],
            shared_outputs: vec![0.0; 4],
            layer_outputs: vec![0.0; 4],
            shared_expert_executed: true,
            routed_intermediate_rows: 4,
            shared_intermediate_rows: 4,
            output_rows: 4,
            residual_adds: 1,
            final_residual_checksum: 0.0,
            expert_input_norm_backend: "cpu-reference-rmsnorm-bf16",
            router_backend: "cpu-reference-router-topk-bf16",
            shared_mlp_backend: "cpu-reference-silu-gated-mlp-bf16",
            residual_add_backend: "cpu-reference-residual-add-bf16",
            layer_summary: RealFullExpertSparseMlpSharedChainLayerProbe {
                layer_id,
                expert_id: 17,
                owner: "spark-0".to_owned(),
                score: 0.75,
                corrected_score: 0.8,
                routed_output_checksum: 0.0,
                shared_output_checksum: 0.0,
                output_checksum: 0.0,
                output_l2_norm: 0.0,
                residual_before_checksum: 0.0,
                residual_delta_checksum: 0.0,
                residual_after_checksum: 0.0,
                expert_input_norm_backend: "cpu-reference-rmsnorm-bf16",
                router_backend: "cpu-reference-router-topk-bf16",
                shared_mlp_backend: "cpu-reference-silu-gated-mlp-bf16",
                residual_add_backend: "cpu-reference-residual-add-bf16",
                first_residual_after: 0.0,
                last_residual_after: 0.0,
            },
            covers_full_top_k: false,
            passed: true,
        }
    }

    #[test]
    fn sparse_protocol_v2_hidden_payload_borrows_full_width_payload() {
        let mut sparse = sparse_selected_route_fixture();
        sparse.output_rows = GLM52_HIDDEN_SIZE;

        let payload = sparse_protocol_v2_hidden_payload(&sparse).unwrap();

        assert!(matches!(payload, std::borrow::Cow::Borrowed(_)));
        assert_eq!(payload.len(), GLM52_HIDDEN_BF16_BYTES);
        assert_eq!(
            payload.as_ptr(),
            sparse.expert_input_hidden_bf16_payload.as_ptr()
        );
    }

    #[test]
    fn sparse_protocol_v2_hidden_payload_owns_bounded_compaction() {
        let sparse = sparse_selected_route_fixture();

        let payload = sparse_protocol_v2_hidden_payload(&sparse).unwrap();
        let expected = bounded_bf16_hidden_payload(
            &sparse.expert_input_hidden_bf16_payload,
            sparse.output_rows,
        )
        .unwrap();

        assert!(matches!(payload, std::borrow::Cow::Owned(_)));
        assert_eq!(payload.as_ref(), expected.as_slice());
        assert_eq!(
            payload.len(),
            sparse.output_rows * std::mem::size_of::<u16>()
        );
    }

    #[test]
    fn sparse_protocol_v2_routed_output_accounting_streams_checksum() {
        let accounting =
            sparse_protocol_v2_routed_output_accounting(&[3.0, 5.5, -1.0], &[1.0, 0.5, 2.0])
                .unwrap();

        assert_eq!(accounting.checksum, 4.0);
        assert!(accounting.finite);

        let err = sparse_protocol_v2_routed_output_accounting(&[1.0, 2.0], &[1.0])
            .unwrap_err()
            .to_string();
        assert!(err.contains("length mismatch"));
    }

    fn patterned_bf16_hidden_payload() -> Vec<u8> {
        let mut hidden = Vec::with_capacity(GLM52_HIDDEN_BF16_BYTES);
        for idx in 0..GLM52_HIDDEN_SIZE {
            let value = ((idx % 17) as f32 - 8.0) / 16.0;
            hidden.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
        }
        hidden
    }

    fn deterministic_layer_ordered_daemon_hidden(layer_id: usize) -> Vec<f32> {
        (0..GLM52_HIDDEN_SIZE)
            .map(|idx| (((idx + layer_id * 13) % 67) as f32 - 33.0) / 64.0)
            .collect()
    }

    fn sparse_artifact_catalog(layer_id: usize, expert_ids: &[usize]) -> TensorCatalog {
        let mut tensors = Vec::new();
        push_artifact_tensor(
            &mut tensors,
            format!("model.layers.{layer_id}.mlp.gate.weight"),
            DType::Bf16,
            TensorRole::Router,
            vec![256, GLM52_HIDDEN_SIZE],
            false,
        );
        push_artifact_tensor(
            &mut tensors,
            format!("model.layers.{layer_id}.mlp.gate.e_score_correction_bias"),
            DType::F32,
            TensorRole::Router,
            vec![256],
            false,
        );
        for expert_id in expert_ids {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                let base_name =
                    format!("model.layers.{layer_id}.mlp.experts.{expert_id}.{projection}");
                push_artifact_tensor(
                    &mut tensors,
                    format!("{base_name}.weight"),
                    DType::U8,
                    TensorRole::RoutedExpert,
                    vec![4, 1],
                    false,
                );
                push_artifact_tensor(
                    &mut tensors,
                    format!("{base_name}.weight_scale"),
                    DType::F8E4M3,
                    TensorRole::RoutedExpert,
                    vec![4, 1],
                    true,
                );
                push_artifact_tensor(
                    &mut tensors,
                    format!("{base_name}.weight_scale_2"),
                    DType::F32,
                    TensorRole::RoutedExpert,
                    Vec::new(),
                    true,
                );
            }
        }
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            push_artifact_tensor(
                &mut tensors,
                format!("model.layers.{layer_id}.mlp.shared_experts.{projection}.weight"),
                DType::Bf16,
                TensorRole::SharedExpert,
                vec![4, 1],
                false,
            );
        }
        TensorCatalog {
            model_id: "test/sparse-artifacts".to_owned(),
            snapshot_path: "/tmp/glmrt-sparse-artifacts".to_owned(),
            facts: ModelFacts::default(),
            tensors,
        }
    }

    fn push_artifact_tensor(
        tensors: &mut Vec<TensorInfo>,
        name: String,
        dtype: DType,
        role: TensorRole,
        shape: Vec<usize>,
        is_quantization_metadata: bool,
    ) {
        let byte_offset = tensors.len() as u64;
        tensors.push(TensorInfo {
            name,
            file: "artifact.safetensors".to_owned(),
            dtype,
            shape,
            byte_offset,
            byte_length: 1,
            role,
            layer_id: Some(GLM52_FIRST_K_DENSE_REPLACE as u32),
            expert_id: None,
            is_quantization_metadata,
        });
    }

    fn load_real_catalog_or_skip() -> Option<TensorCatalog> {
        if !crate::commands::real_full::tests::fixture::real_checkpoint_tests_enabled() {
            crate::commands::real_full::tests::fixture::real_checkpoint_tests_skip_message(
                "layer-ordered execution trace",
            );
            return None;
        }
        let catalog_path = real_catalog_path();
        let Ok(file) = File::open(&catalog_path) else {
            eprintln!("skipped: missing {}", catalog_path.display());
            return None;
        };
        let catalog: TensorCatalog =
            serde_json::from_reader(file).expect("parsing real GLM catalog fixture");
        if !Path::new(&catalog.snapshot_path).exists() {
            eprintln!("skipped: missing snapshot {}", catalog.snapshot_path);
            return None;
        }
        Some(catalog)
    }

    fn real_catalog_path() -> PathBuf {
        repo_root().join(".glmrt-cache/model-artifacts/diagnostic/model_catalog.json")
    }

    fn load_full_loadplan_path_or_skip() -> Option<PathBuf> {
        if !crate::commands::real_full::tests::fixture::real_checkpoint_tests_enabled() {
            crate::commands::real_full::tests::fixture::real_checkpoint_tests_skip_message(
                "layer-ordered execution trace full loadplan",
            );
            return None;
        }
        let loadplan_path =
            repo_root().join(".glmrt-cache/model-artifacts/diagnostic/loadplan.json");
        if !loadplan_path.exists() {
            eprintln!("skipped: missing {}", loadplan_path.display());
            return None;
        }
        Some(loadplan_path)
    }

    fn unused_loopback_addr() -> SocketAddr {
        let listener = StdTcpListener::bind("127.0.0.1:0").unwrap();
        listener.local_addr().unwrap()
    }

    async fn start_unpinned_real_expertd_targets(
        catalog_path: &Path,
        loadplan_path: &Path,
        hosts: &[String],
        real_layer: Option<u32>,
    ) -> (
        Vec<TcpProtocolV2HostBatchTarget>,
        Vec<tokio::task::JoinHandle<anyhow::Result<()>>>,
    ) {
        let mut targets = Vec::new();
        let mut servers = Vec::new();

        for host in hosts {
            let addr = unused_loopback_addr();
            let args = ExpertDaemonArgs {
                synthetic_weights: false,
                preflight_only: false,
                transport: "tcp".to_owned(),
                listen: addr.to_string(),
                loadplan: Some(loadplan_path.to_path_buf()),
                catalog: Some(catalog_path.to_path_buf()),
                model_id: glmrt_core::DEFAULT_MODEL_ID.to_owned(),
                real_layer,
                role_hostname: Some(host.clone()),
            };
            servers.push(tokio::spawn(async move { run_expertd(args).await }));
            targets.push(TcpProtocolV2HostBatchTarget {
                host: host.clone(),
                addr,
            });
        }
        for target in &targets {
            wait_for_expertd_tcp_listener(target.addr).await;
        }

        (targets, servers)
    }

    async fn wait_for_expertd_tcp_listener(addr: SocketAddr) {
        for _ in 0..24_000 {
            if TcpStream::connect(addr).await.is_ok() {
                return;
            }
            sleep(Duration::from_millis(10)).await;
        }
        panic!("timed out waiting for expertd TCP listener at {addr}");
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

    fn assert_execution_stepper_coordinator_backend_counts(
        stepper: &RealFullResidualExecutionStepper,
        expected_coordinator_stages: usize,
        expected_cuda_stages_when_enabled: usize,
    ) {
        assert_eq!(stepper.coordinator_stage_count, expected_coordinator_stages);
        if coordinator_cuda_reference_kernels_enabled() {
            assert_eq!(
                stepper.coordinator_cuda_stage_count,
                expected_cuda_stages_when_enabled
            );
            assert_eq!(stepper.coordinator_cpu_stage_count, 0);
            assert_eq!(
                stepper.coordinator_unknown_stage_count,
                expected_coordinator_stages - expected_cuda_stages_when_enabled
            );
            assert_eq!(
                stepper.uses_cuda_coordinator_kernels,
                expected_cuda_stages_when_enabled == expected_coordinator_stages
            );
        } else {
            assert_eq!(stepper.coordinator_cuda_stage_count, 0);
            assert_eq!(
                stepper.coordinator_cpu_stage_count,
                expected_coordinator_stages
            );
            assert_eq!(stepper.coordinator_unknown_stage_count, 0);
            assert!(!stepper.uses_cuda_coordinator_kernels);
        }
    }

    fn expected_prefix_attention_stage_source() -> &'static str {
        if coordinator_cuda_reference_kernels_enabled() {
            "real-checkpoint-bf16-causal-attention-prefix-cuda-reference-linear-bf16-cuda-reference-causal-attention-bf16-cuda-reference-residual-add-bf16"
        } else {
            "real-checkpoint-bf16-causal-attention-prefix-cpu-reference-linear-bf16-cpu-reference-causal-attention-bf16-cpu-reference-residual-add-bf16"
        }
    }

    fn expected_full_output_attention_stage_source() -> &'static str {
        if coordinator_cuda_reference_kernels_enabled() {
            "real-checkpoint-bf16-causal-attention-full-output-rows-cuda-reference-linear-bf16-cuda-reference-causal-attention-bf16-cuda-reference-residual-add-bf16"
        } else {
            "real-checkpoint-bf16-causal-attention-full-output-rows-cpu-reference-linear-bf16-cpu-reference-causal-attention-bf16-cpu-reference-residual-add-bf16"
        }
    }

    fn expected_full_output_mla_rope_attention_stage_source() -> &'static str {
        if coordinator_cuda_reference_kernels_enabled() {
            "real-checkpoint-bf16-full-output-main-mla-rope-kv-cache-prefix-context-attention-cuda-reference-linear-bf16-cuda-reference-mla-rope-attention-bf16-cuda-reference-residual-add-bf16"
        } else {
            "real-checkpoint-bf16-full-output-main-mla-rope-kv-cache-prefix-context-attention-cpu-reference-linear-bf16-cpu-reference-mla-rope-attention-bf16-cpu-reference-residual-add-bf16"
        }
    }

    fn expected_prefix_dense_mlp_stage_source() -> &'static str {
        if coordinator_cuda_reference_kernels_enabled() {
            "real-checkpoint-bf16-dense-mlp-prefix"
        } else {
            "real-checkpoint-bf16-dense-mlp-prefix-cpu-reference-rmsnorm-bf16-cpu-reference-linear-bf16-cpu-reference-silu-gated-mlp-bf16-cpu-reference-residual-add-bf16"
        }
    }

    fn expected_prefix_sparse_moe_stage_source() -> &'static str {
        if coordinator_cuda_reference_kernels_enabled() {
            "real-checkpoint-nvfp4-routed-plus-bf16-shared-moe-prefix"
        } else {
            "real-checkpoint-nvfp4-routed-plus-bf16-shared-moe-prefix-cpu-reference-rmsnorm-bf16-cpu-reference-router-topk-bf16-cpu-reference-shared-silu-gated-mlp-bf16-cpu-reference-residual-add-bf16"
        }
    }

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
    }
}
