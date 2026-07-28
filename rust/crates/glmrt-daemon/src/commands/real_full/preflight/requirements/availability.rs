use glmrt_core::{
    GLM52_FIRST_K_DENSE_REPLACE, GLM52_HIDDEN_BF16_BYTES, GLM52_HIDDEN_SIZE,
    GLM52_NUM_HIDDEN_LAYERS, GLM52_ROUTED_EXPERTS, GLM52_TOP_K,
};

use super::super::super::constants::{
    REAL_FULL_PREFLIGHT_DECODE_ROWS, REAL_FULL_PREFLIGHT_MTP_ACCEPTED_ROWS,
    REAL_FULL_PREFLIGHT_MTP_ROWS, REAL_FULL_PREFLIGHT_PREFILL_ROWS,
    REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START,
};
use super::super::super::coordinator_kernels::coordinator_cuda_reference_kernels_enabled;
use super::super::super::coverage::catalog_supports_default_dense_sparse_shared_lm_head;
use super::super::super::types::*;
use super::RealFullPreflightRequirementInputs;

pub(super) struct RealFullPreflightAvailability<'a> {
    pub(super) sparse_layers: usize,
    pub(super) facts_match: bool,
    pub(super) full_catalog_coverage: bool,
    pub(super) execution_plan_available: bool,
    pub(super) residual_stream_dry_run_available: bool,
    pub(super) residual_kernel: &'a RealFullResidualKernelSelfTest,
    pub(super) residual_numeric_kernel_available: bool,
    pub(super) dense_prefix_probe: &'a RealFullDensePrefixProbe,
    pub(super) attention_residual_prefix_probe: &'a RealFullAttentionResidualPrefixProbe,
    pub(super) mla_rope_attention_probe: &'a RealFullMlaRopeAttentionProbe,
    pub(super) dsa_indexer_attention_probe: &'a RealFullDsaIndexerAttentionProbe,
    pub(super) attention_dense_sparse_prefix_probe: &'a RealFullAttentionDenseSparsePrefixProbe,
    pub(super) sampling_dry_run_available: bool,
    pub(super) sampling_lm_head_default_chunk_available: bool,
    pub(super) sampling_real_lm_head_probe: &'a RealFullSamplingRealLmHeadProbe,
    pub(super) scheduler_dry_run_available: bool,
    pub(super) scheduler_real_tensor_catalog_available: bool,
    pub(super) scheduler_execution_available: bool,
    pub(super) scheduler_numeric_progression: &'a RealFullSchedulerNumericProgressionSelfTest,
    pub(super) scheduler_numeric_progression_available: bool,
    pub(super) expert_execution_dry_run_available: bool,
    pub(super) expert_numeric_probe: &'a RealFullExpertRealNvfp4Probe,
    pub(super) expert_all_layer_probe: &'a RealFullExpertAllLayerNvfp4Probe,
    pub(super) expert_residual_chain_probe: &'a RealFullExpertResidualChainNvfp4Probe,
    pub(super) expert_shared_chain_probe: &'a RealFullExpertSparseMlpSharedChainProbe,
    pub(super) expert_scheduler_rows_probe: &'a RealFullExpertSchedulerRowsNvfp4Probe,
    pub(super) kv_backing_store_available: bool,
    pub(super) attention_kv_io_available: bool,
    pub(super) attention_kv_binding_available: bool,
}

pub(super) fn real_full_preflight_availability<'a>(
    inputs: &RealFullPreflightRequirementInputs<'a>,
) -> RealFullPreflightAvailability<'a> {
    let catalog = inputs.catalog;
    let coverage = inputs.coverage;
    let kv_config = inputs.kv_config;
    let expert_hosts = inputs.expert_hosts;
    let execution_plan = inputs.execution_plan;
    let residual_stream_dry_run = inputs.residual_stream_dry_run;
    let sampling_dry_run = inputs.sampling_dry_run;
    let expert_execution_dry_run = inputs.expert_execution_dry_run;
    let scheduler_dry_run = inputs.scheduler_dry_run;
    let scheduler_execution_dry_run = inputs.scheduler_execution_dry_run;
    let kv_backing_store_dry_run = inputs.kv_backing_store_dry_run;
    let attention_kv_io_dry_run = inputs.attention_kv_io_dry_run;
    let attention_kv_binding_dry_run = inputs.attention_kv_binding_dry_run;

    let sparse_layers = GLM52_NUM_HIDDEN_LAYERS - GLM52_FIRST_K_DENSE_REPLACE;
    let facts_match = catalog.facts.hidden_size == GLM52_HIDDEN_SIZE
        && catalog.facts.num_hidden_layers == GLM52_NUM_HIDDEN_LAYERS
        && catalog.facts.first_k_dense_replace == GLM52_FIRST_K_DENSE_REPLACE
        && catalog.facts.routed_experts == GLM52_ROUTED_EXPERTS
        && catalog.facts.top_k == GLM52_TOP_K;
    let full_catalog_coverage = coverage.embedding_tensors > 0
        && coverage.lm_head_tensors > 0
        && coverage.hidden_layers_with_any_tensor == GLM52_NUM_HIDDEN_LAYERS
        && coverage.sparse_layers_with_routed_experts == sparse_layers
        && coverage.dense_layers_with_dense_mlp == GLM52_FIRST_K_DENSE_REPLACE;
    let execution_plan_available = !expert_hosts.is_empty()
        && execution_plan.layer_count == GLM52_NUM_HIDDEN_LAYERS
        && execution_plan.dense_layer_count == GLM52_FIRST_K_DENSE_REPLACE
        && execution_plan.sparse_layer_count == sparse_layers
        && execution_plan
            .layers
            .iter()
            .filter(|layer| layer.mlp.routed_nvfp4_expert_exchange)
            .count()
            == sparse_layers;
    let residual_stream_dry_run_available = !expert_hosts.is_empty()
        && residual_stream_dry_run.layer_count == GLM52_NUM_HIDDEN_LAYERS
        && residual_stream_dry_run.row_count
            == REAL_FULL_PREFLIGHT_PREFILL_ROWS
                + REAL_FULL_PREFLIGHT_MTP_ROWS
                + REAL_FULL_PREFLIGHT_DECODE_ROWS
        && residual_stream_dry_run.dense_layers == GLM52_FIRST_K_DENSE_REPLACE
        && residual_stream_dry_run.sparse_layers == sparse_layers
        && residual_stream_dry_run.remote_sparse_layers == sparse_layers
        && residual_stream_dry_run.total_residual_adds == GLM52_NUM_HIDDEN_LAYERS * 2
        && residual_stream_dry_run.terminal_stages
            == ["final_norm", "lm_head", "full_vocab_sampling"]
        && residual_stream_dry_run.layer_order_verified;
    let residual_kernel = &residual_stream_dry_run.numeric_kernel_self_test;
    let residual_numeric_kernel_available = residual_kernel.passed
        && residual_kernel.residual_adds == residual_kernel.layers * 2
        && residual_kernel.values_updated
            == residual_kernel.layers * residual_kernel.rows * residual_kernel.hidden_dim * 2;
    let dense_prefix_probe = &residual_stream_dry_run.real_dense_prefix_probe;
    let attention_residual_prefix_probe =
        &residual_stream_dry_run.real_attention_residual_prefix_probe;
    let mla_rope_attention_probe = &residual_stream_dry_run.real_mla_rope_attention_probe;
    let dsa_indexer_attention_probe = &residual_stream_dry_run.real_dsa_indexer_attention_probe;
    let attention_dense_sparse_prefix_probe =
        &residual_stream_dry_run.real_attention_dense_sparse_prefix_probe;
    let sampling_dry_run_available = !expert_hosts.is_empty()
        && sampling_dry_run.lm_head_tensor == "lm_head.weight"
        && sampling_dry_run.hidden_dim == GLM52_HIDDEN_SIZE
        && sampling_dry_run.vocab_size > 0
        && sampling_dry_run.sampled_rows == REAL_FULL_PREFLIGHT_DECODE_ROWS
        && sampling_dry_run.covers_full_vocabulary
        && sampling_dry_run.greedy_reduce_chunks == sampling_dry_run.chunk_count
        && sampling_dry_run.requires_numeric_logits;
    let sampling_lm_head_default_chunk = &sampling_dry_run.real_lm_head_default_chunk_probe;
    let sampling_lm_head_default_chunk_available = sampling_lm_head_default_chunk.passed
        && sampling_lm_head_default_chunk.uses_real_lm_head
        && sampling_lm_head_default_chunk.hidden_dim == GLM52_HIDDEN_SIZE
        && sampling_lm_head_default_chunk.rows_scored > 0
        && sampling_lm_head_default_chunk.logits_evaluated
            == sampling_lm_head_default_chunk.rows_scored
        && sampling_lm_head_default_chunk
            .logits_kernel_backend
            .is_some()
        && sampling_lm_head_default_chunk
            .argmax_kernel_backend
            .is_some()
        && sampling_lm_head_default_chunk
            .sampler_kernel_backend
            .is_some();
    let sampling_real_lm_head_probe = &sampling_dry_run.real_lm_head_full_vocab_probe;
    let scheduler_dry_run_available = !expert_hosts.is_empty()
        && scheduler_dry_run.total_layerwaves == GLM52_NUM_HIDDEN_LAYERS * 3
        && scheduler_dry_run.sparse_expert_batches == sparse_layers
        && scheduler_dry_run.rows_per_sparse_expert_batch
            == REAL_FULL_PREFLIGHT_PREFILL_ROWS
                + REAL_FULL_PREFLIGHT_MTP_ROWS
                + REAL_FULL_PREFLIGHT_DECODE_ROWS
        && scheduler_dry_run.routes_per_sparse_expert_batch
            == scheduler_dry_run.rows_per_sparse_expert_batch * GLM52_TOP_K
        && scheduler_dry_run.protocol_v2_batch_probe.passed;
    let scheduler_real_tensor_catalog_available =
        catalog_supports_default_dense_sparse_shared_lm_head(catalog);
    let scheduler_device_attention_launches = GLM52_NUM_HIDDEN_LAYERS * 4;
    let scheduler_device_attention_residency_available =
        if scheduler_execution_dry_run.uses_device_kv_attention {
            let (expected_uploads, expected_query_shapes) =
                if coordinator_cuda_reference_kernels_enabled() {
                    (4, 0)
                } else {
                    (5, 3)
                };
            let expected_buffer_uses_per_launch = if coordinator_cuda_reference_kernels_enabled() {
                4
            } else {
                3
            };
            scheduler_execution_dry_run.device_attention_resident_uploads == expected_uploads
                && scheduler_execution_dry_run.device_attention_resident_query_shapes
                    == expected_query_shapes
                && scheduler_execution_dry_run.device_attention_resident_buffer_uses
                    == scheduler_device_attention_launches * expected_buffer_uses_per_launch
        } else {
            scheduler_execution_dry_run.device_attention_resident_uploads == 0
                && scheduler_execution_dry_run.device_attention_resident_query_shapes == 0
                && scheduler_execution_dry_run.device_attention_resident_buffer_uses == 0
        };
    // The final prefill chunk carries the decode and MTP waves in one admitted
    // ExpertBatch, so the preflight shape has two scheduler iterations per layer.
    let scheduler_execution_available = !expert_hosts.is_empty()
        && scheduler_execution_dry_run.iterations == GLM52_NUM_HIDDEN_LAYERS * 2
        && scheduler_execution_dry_run.candidate_layerwaves == GLM52_NUM_HIDDEN_LAYERS * 4
        && scheduler_execution_dry_run.selected_layerwaves == GLM52_NUM_HIDDEN_LAYERS * 4
        && scheduler_execution_dry_run.deferred_layerwaves == 0
        && scheduler_execution_dry_run.sparse_expert_batches
            == (GLM52_NUM_HIDDEN_LAYERS - GLM52_FIRST_K_DENSE_REPLACE) * 2
        && scheduler_execution_dry_run.sparse_expert_host_batch_sets
            == scheduler_execution_dry_run.sparse_expert_batches
        && scheduler_execution_dry_run.sparse_expert_host_batch_routes
            == scheduler_execution_dry_run.sparse_expert_batch_routes
        && scheduler_execution_dry_run.sparse_expert_host_batch_routes_match_global
        && scheduler_execution_dry_run.sparse_expert_host_batch_graph_counts_valid
        && scheduler_execution_dry_run.sparse_expert_host_request_frames
            == scheduler_execution_dry_run.sparse_expert_host_batches
        && scheduler_execution_dry_run.sparse_expert_host_response_frames
            == scheduler_execution_dry_run.sparse_expert_host_batches
        && scheduler_execution_dry_run.sparse_expert_host_request_rows
            == scheduler_execution_dry_run.sparse_expert_host_batch_rows
        && scheduler_execution_dry_run.sparse_expert_host_response_rows
            == scheduler_execution_dry_run.sparse_expert_host_batch_rows
        && scheduler_execution_dry_run.sparse_expert_host_request_routes
            == scheduler_execution_dry_run.sparse_expert_host_batch_routes
        && scheduler_execution_dry_run.sparse_expert_host_request_payload_bytes
            == scheduler_execution_dry_run.sparse_expert_host_batch_rows * GLM52_HIDDEN_BF16_BYTES
        && scheduler_execution_dry_run.sparse_expert_host_response_payload_bytes
            == scheduler_execution_dry_run.sparse_expert_host_batch_rows * GLM52_HIDDEN_BF16_BYTES
        && scheduler_execution_dry_run.sparse_expert_host_request_wire_bytes
            > scheduler_execution_dry_run.sparse_expert_host_request_payload_bytes
        && scheduler_execution_dry_run.sparse_expert_host_response_wire_bytes
            > scheduler_execution_dry_run.sparse_expert_host_response_payload_bytes
        && scheduler_execution_dry_run.sparse_expert_host_wire_envelopes_valid
        && scheduler_real_tensor_catalog_available
        && scheduler_device_attention_residency_available
        && if coordinator_cuda_reference_kernels_enabled() {
            scheduler_execution_dry_run.full_context_device_attention_complete
        } else {
            !scheduler_execution_dry_run.full_context_device_attention_complete
        }
        && scheduler_execution_dry_run.layer_order_verified;
    let scheduler_numeric_progression = &scheduler_execution_dry_run.numeric_progression_self_test;
    let scheduler_numeric_unique_rows = REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START as usize
        + REAL_FULL_PREFLIGHT_PREFILL_ROWS
        + REAL_FULL_PREFLIGHT_DECODE_ROWS
        + REAL_FULL_PREFLIGHT_MTP_ROWS;
    let scheduler_numeric_source_segments = GLM52_NUM_HIDDEN_LAYERS * 4;
    let scheduler_numeric_value_updates =
        scheduler_numeric_unique_rows * GLM52_NUM_HIDDEN_LAYERS * GLM52_HIDDEN_SIZE;
    let scheduler_numeric_resident_values = scheduler_numeric_unique_rows * GLM52_HIDDEN_SIZE;
    let scheduler_numeric_real_dense_mlp_rows =
        GLM52_FIRST_K_DENSE_REPLACE * scheduler_numeric_unique_rows;
    let scheduler_numeric_real_dense_mlp_values =
        scheduler_numeric_real_dense_mlp_rows * GLM52_HIDDEN_SIZE;
    let scheduler_numeric_real_dense_mlp_source_segments = GLM52_FIRST_K_DENSE_REPLACE * 4;
    let scheduler_numeric_real_sparse_shared_mlp_rows =
        sparse_layers * scheduler_numeric_unique_rows;
    let scheduler_numeric_real_sparse_shared_mlp_values =
        scheduler_numeric_real_sparse_shared_mlp_rows * GLM52_HIDDEN_SIZE;
    let scheduler_numeric_real_sparse_shared_mlp_source_segments = sparse_layers * 4;
    let scheduler_numeric_real_sparse_routed_mlp_rows =
        scheduler_numeric_real_sparse_shared_mlp_rows;
    let scheduler_numeric_real_sparse_routed_mlp_values =
        scheduler_numeric_real_sparse_shared_mlp_values;
    let scheduler_numeric_real_sparse_routed_mlp_source_segments =
        scheduler_numeric_real_sparse_shared_mlp_source_segments;
    let scheduler_numeric_real_sparse_routed_mlp_routes =
        scheduler_numeric_real_sparse_routed_mlp_rows * GLM52_TOP_K;
    let scheduler_device_attention_values_per_row = scheduler_execution_dry_run
        .device_attention_output_values
        .checked_div(scheduler_execution_dry_run.device_attention_rows)
        .filter(|values_per_row| {
            *values_per_row > 0
                && scheduler_execution_dry_run.device_attention_output_values
                    % scheduler_execution_dry_run.device_attention_rows
                    == 0
        });
    let scheduler_numeric_device_attention_delta_available = if scheduler_execution_dry_run
        .uses_device_kv_attention
    {
        let common = scheduler_numeric_progression.uses_device_attention_output_delta
            && scheduler_numeric_progression.attention_device_output_delta_backend
                == scheduler_execution_dry_run.device_attention_status
            && scheduler_numeric_progression
                .attention_device_output_delta_checksum
                .is_finite();
        if coordinator_cuda_reference_kernels_enabled() {
            common
                && scheduler_numeric_progression.device_attention_output_delta_status
                    == "cuda-device-attention-hidden-delta"
                && scheduler_numeric_progression.attention_device_output_delta_rows
                    == scheduler_execution_dry_run.device_attention_query_rows
                && scheduler_numeric_progression.attention_device_output_delta_values
                    == scheduler_execution_dry_run.device_attention_query_rows * GLM52_HIDDEN_SIZE
                && scheduler_numeric_progression.attention_device_output_delta_values
                    == scheduler_execution_dry_run.device_attention_output_values
        } else {
            common
                && scheduler_numeric_progression.device_attention_output_delta_status
                    == "cuda-device-attention-output-prefix-delta"
                && scheduler_numeric_progression.attention_device_output_delta_rows
                    == scheduler_execution_dry_run.device_attention_query_rows
                && scheduler_device_attention_values_per_row
                    .and_then(|values_per_row| {
                        scheduler_numeric_progression
                            .attention_device_output_delta_rows
                            .checked_mul(values_per_row)
                    })
                    .map(|expected_values| {
                        scheduler_numeric_progression.attention_device_output_delta_values
                            == expected_values
                    })
                    .unwrap_or(false)
                && scheduler_numeric_progression.attention_device_output_delta_values
                    <= scheduler_execution_dry_run.device_attention_output_values
        }
    } else {
        !scheduler_numeric_progression.uses_device_attention_output_delta
            && scheduler_numeric_progression.device_attention_output_delta_status == "not-run"
            && scheduler_numeric_progression.attention_device_output_delta_rows == 0
            && scheduler_numeric_progression.attention_device_output_delta_values == 0
            && scheduler_numeric_progression.attention_device_output_delta_checksum == 0.0
            && scheduler_numeric_progression.attention_device_output_delta_backend == "not-run"
    };
    let scheduler_numeric_device_attention_prefix_overlay_available = if scheduler_execution_dry_run
        .uses_device_kv_attention
        && coordinator_cuda_reference_kernels_enabled()
    {
        scheduler_numeric_progression.attention_device_output_delta_device_prefix_rows == 0
            && scheduler_numeric_progression.attention_device_output_delta_device_prefix_values == 0
            && scheduler_numeric_progression.attention_device_output_delta_device_prefix_backend
                == "not-run"
    } else {
        scheduler_numeric_progression.attention_device_output_delta_device_prefix_rows == 0
            && scheduler_numeric_progression.attention_device_output_delta_device_prefix_values == 0
            && scheduler_numeric_progression.attention_device_output_delta_device_prefix_backend
                == "not-run"
    };
    let scheduler_numeric_device_delta_template_available =
        if coordinator_cuda_reference_kernels_enabled() {
            scheduler_numeric_progression.device_delta_template_status
                == "cuda-device-delta-template-not-needed"
                && scheduler_numeric_progression.device_delta_template_uploads == 0
                && scheduler_numeric_progression.device_delta_template_uses == 0
                && scheduler_numeric_progression.device_delta_template_resident_values == 0
        } else {
            scheduler_numeric_progression.device_delta_template_status == "not-run"
                && scheduler_numeric_progression.device_delta_template_uploads == 0
                && scheduler_numeric_progression.device_delta_template_uses == 0
                && scheduler_numeric_progression.device_delta_template_resident_values == 0
        };
    let scheduler_numeric_device_mlp_delta_available =
        if coordinator_cuda_reference_kernels_enabled() {
            !scheduler_numeric_progression.uses_device_mlp_delta
                && scheduler_numeric_progression.device_mlp_delta_status
                    == "cuda-device-hidden-dependent-mlp-delta-not-needed"
                && scheduler_numeric_progression.device_mlp_delta_rows == 0
                && scheduler_numeric_progression.device_mlp_delta_values == 0
                && scheduler_numeric_progression.device_mlp_delta_backend == "not-run"
                && scheduler_numeric_progression.device_mlp_delta_checksum == 0.0
                && scheduler_numeric_progression.device_mlp_weight_uploads == 0
                && scheduler_numeric_progression.device_mlp_weight_resident_values == 0
        } else {
            !scheduler_numeric_progression.uses_device_mlp_delta
                && scheduler_numeric_progression.device_mlp_delta_status == "not-run"
                && scheduler_numeric_progression.device_mlp_delta_rows == 0
                && scheduler_numeric_progression.device_mlp_delta_values == 0
                && scheduler_numeric_progression.device_mlp_delta_checksum == 0.0
                && scheduler_numeric_progression.device_mlp_delta_backend == "not-run"
                && scheduler_numeric_progression.device_mlp_weight_uploads == 0
                && scheduler_numeric_progression.device_mlp_weight_resident_values == 0
        };
    let scheduler_numeric_device_real_dense_mlp_delta_available =
        if coordinator_cuda_reference_kernels_enabled() {
            scheduler_numeric_progression.uses_device_real_dense_mlp_delta
                && scheduler_numeric_progression.device_real_dense_mlp_delta_status
                    == "cuda-real-dense-checkpoint-mlp-delta"
                && scheduler_numeric_progression.device_real_dense_mlp_delta_rows
                    == scheduler_numeric_real_dense_mlp_rows
                && scheduler_numeric_progression.device_real_dense_mlp_delta_values
                    == scheduler_numeric_real_dense_mlp_values
                && scheduler_numeric_progression
                    .device_real_dense_mlp_delta_backend
                    .contains("silu-gated-mlp-bf16-preloaded-gate-up-down")
                && scheduler_numeric_progression
                    .device_real_dense_mlp_norm_backend
                    .contains("rmsnorm-bf16")
                && scheduler_numeric_progression
                    .device_real_dense_mlp_delta_checksum
                    .is_finite()
                && scheduler_numeric_progression.device_real_dense_mlp_weight_tensors
                    == GLM52_FIRST_K_DENSE_REPLACE * 4
                && scheduler_numeric_progression.device_real_dense_mlp_weight_bytes > 0
                && scheduler_numeric_progression.device_real_dense_mlp_layers
                    == GLM52_FIRST_K_DENSE_REPLACE
                && scheduler_numeric_progression.device_real_dense_mlp_source_segments
                    == scheduler_numeric_real_dense_mlp_source_segments
        } else {
            !scheduler_numeric_progression.uses_device_real_dense_mlp_delta
                && scheduler_numeric_progression.device_real_dense_mlp_delta_status == "not-run"
                && scheduler_numeric_progression.device_real_dense_mlp_delta_rows == 0
                && scheduler_numeric_progression.device_real_dense_mlp_delta_values == 0
                && scheduler_numeric_progression.device_real_dense_mlp_delta_checksum == 0.0
                && scheduler_numeric_progression.device_real_dense_mlp_delta_backend == "not-run"
                && scheduler_numeric_progression.device_real_dense_mlp_norm_backend == "not-run"
                && scheduler_numeric_progression.device_real_dense_mlp_weight_tensors == 0
                && scheduler_numeric_progression.device_real_dense_mlp_weight_bytes == 0
                && scheduler_numeric_progression.device_real_dense_mlp_layers == 0
                && scheduler_numeric_progression.device_real_dense_mlp_source_segments == 0
        };
    let scheduler_numeric_device_real_sparse_shared_mlp_delta_available =
        if coordinator_cuda_reference_kernels_enabled() {
            scheduler_numeric_progression.uses_device_real_sparse_shared_mlp_delta
                && scheduler_numeric_progression.device_real_sparse_shared_mlp_delta_status
                    == "cuda-real-sparse-shared-checkpoint-mlp-delta"
                && scheduler_numeric_progression.device_real_sparse_shared_mlp_delta_rows
                    == scheduler_numeric_real_sparse_shared_mlp_rows
                && scheduler_numeric_progression.device_real_sparse_shared_mlp_delta_values
                    == scheduler_numeric_real_sparse_shared_mlp_values
                && scheduler_numeric_progression
                    .device_real_sparse_shared_mlp_delta_backend
                    .contains("silu-gated-mlp-bf16-preloaded-gate-up-down")
                && scheduler_numeric_progression
                    .device_real_sparse_shared_mlp_norm_backend
                    .contains("rmsnorm-bf16")
                && scheduler_numeric_progression
                    .device_real_sparse_shared_mlp_delta_checksum
                    .is_finite()
                && scheduler_numeric_progression.device_real_sparse_shared_mlp_weight_tensors
                    == sparse_layers * 4
                && scheduler_numeric_progression.device_real_sparse_shared_mlp_weight_bytes > 0
                && scheduler_numeric_progression.device_real_sparse_shared_mlp_layers
                    == sparse_layers
                && scheduler_numeric_progression.device_real_sparse_shared_mlp_source_segments
                    == scheduler_numeric_real_sparse_shared_mlp_source_segments
        } else {
            !scheduler_numeric_progression.uses_device_real_sparse_shared_mlp_delta
                && scheduler_numeric_progression.device_real_sparse_shared_mlp_delta_status
                    == "not-run"
                && scheduler_numeric_progression.device_real_sparse_shared_mlp_delta_rows == 0
                && scheduler_numeric_progression.device_real_sparse_shared_mlp_delta_values == 0
                && scheduler_numeric_progression.device_real_sparse_shared_mlp_delta_checksum == 0.0
                && scheduler_numeric_progression.device_real_sparse_shared_mlp_delta_backend
                    == "not-run"
                && scheduler_numeric_progression.device_real_sparse_shared_mlp_norm_backend
                    == "not-run"
                && scheduler_numeric_progression.device_real_sparse_shared_mlp_weight_tensors == 0
                && scheduler_numeric_progression.device_real_sparse_shared_mlp_weight_bytes == 0
                && scheduler_numeric_progression.device_real_sparse_shared_mlp_layers == 0
                && scheduler_numeric_progression.device_real_sparse_shared_mlp_source_segments == 0
        };
    let scheduler_numeric_device_real_sparse_routed_mlp_delta_available =
        if coordinator_cuda_reference_kernels_enabled() {
            scheduler_numeric_progression.uses_device_real_sparse_routed_mlp_delta
                && scheduler_numeric_progression.device_real_sparse_routed_mlp_delta_status
                    == "cuda-real-sparse-routed-nvfp4-checkpoint-mlp-delta"
                && scheduler_numeric_progression.device_real_sparse_routed_mlp_delta_rows
                    == scheduler_numeric_real_sparse_routed_mlp_rows
                && scheduler_numeric_progression.device_real_sparse_routed_mlp_delta_values
                    == scheduler_numeric_real_sparse_routed_mlp_values
                && scheduler_numeric_progression
                    .device_real_sparse_routed_mlp_delta_backend
                    .contains("nvfp4-route-bf16-accumulated-device-output")
                && scheduler_numeric_progression
                    .device_real_sparse_routed_mlp_route_backend
                    .contains("nvfp4-route-bf16-accumulated-device-input")
                && scheduler_numeric_progression
                    .device_real_sparse_routed_mlp_router_backend
                    .contains("router-topk-bf16")
                && scheduler_numeric_progression
                    .device_real_sparse_routed_mlp_router_backend
                    .contains("device-input")
                && scheduler_numeric_progression
                    .device_real_sparse_routed_mlp_delta_checksum
                    .is_finite()
                && scheduler_numeric_progression.device_real_sparse_routed_mlp_routes
                    == scheduler_numeric_real_sparse_routed_mlp_routes
                && scheduler_numeric_progression.device_real_sparse_routed_mlp_router_weight_bytes
                    > 0
                && scheduler_numeric_progression.device_real_sparse_routed_mlp_router_bias_bytes > 0
                && scheduler_numeric_progression
                    .device_real_sparse_routed_mlp_route_cache_cuda_entries
                    > 0
                && scheduler_numeric_progression
                    .device_real_sparse_routed_mlp_route_cache_cuda_uploads
                    > 0
                && scheduler_numeric_progression.device_real_sparse_routed_mlp_router_cache_entries
                    == sparse_layers
                && scheduler_numeric_progression.device_real_sparse_routed_mlp_layers
                    == sparse_layers
                && scheduler_numeric_progression.device_real_sparse_routed_mlp_source_segments
                    == scheduler_numeric_real_sparse_routed_mlp_source_segments
        } else {
            !scheduler_numeric_progression.uses_device_real_sparse_routed_mlp_delta
                && scheduler_numeric_progression.device_real_sparse_routed_mlp_delta_status
                    == "not-run"
                && scheduler_numeric_progression.device_real_sparse_routed_mlp_delta_rows == 0
                && scheduler_numeric_progression.device_real_sparse_routed_mlp_delta_values == 0
                && scheduler_numeric_progression.device_real_sparse_routed_mlp_delta_checksum == 0.0
                && scheduler_numeric_progression.device_real_sparse_routed_mlp_delta_backend
                    == "not-run"
                && scheduler_numeric_progression.device_real_sparse_routed_mlp_route_backend
                    == "not-run"
                && scheduler_numeric_progression.device_real_sparse_routed_mlp_router_backend
                    == "not-run"
                && scheduler_numeric_progression.device_real_sparse_routed_mlp_routes == 0
                && scheduler_numeric_progression.device_real_sparse_routed_mlp_router_weight_bytes
                    == 0
                && scheduler_numeric_progression.device_real_sparse_routed_mlp_router_bias_bytes
                    == 0
                && scheduler_numeric_progression
                    .device_real_sparse_routed_mlp_route_cache_cuda_entries
                    == 0
                && scheduler_numeric_progression
                    .device_real_sparse_routed_mlp_route_cache_cuda_uploads
                    == 0
                && scheduler_numeric_progression.device_real_sparse_routed_mlp_route_cache_cuda_hits
                    == 0
                && scheduler_numeric_progression.device_real_sparse_routed_mlp_router_cache_entries
                    == 0
                && scheduler_numeric_progression.device_real_sparse_routed_mlp_router_cache_hits
                    == 0
                && scheduler_numeric_progression.device_real_sparse_routed_mlp_layers == 0
                && scheduler_numeric_progression.device_real_sparse_routed_mlp_source_segments == 0
        };
    let scheduler_numeric_device_segments_available =
        if coordinator_cuda_reference_kernels_enabled() {
            scheduler_numeric_progression.uses_device_hidden_segment_residual_add
                && scheduler_numeric_progression.device_hidden_segment_status
                    == "cuda-device-hidden-segment-residual-add"
                && scheduler_numeric_progression.device_hidden_segment_residual_adds
                    == scheduler_numeric_source_segments * 2
                && scheduler_numeric_progression.device_hidden_segment_value_updates
                    == scheduler_numeric_value_updates * 2
                && scheduler_numeric_progression
                    .device_hidden_segment_residual_add_backend
                    .contains("residual-add-bf16")
                && scheduler_numeric_progression.device_hidden_segment_resident_segments == 4
                && scheduler_numeric_progression.device_hidden_segment_resident_values
                    == scheduler_numeric_resident_values
                && scheduler_numeric_progression.device_hidden_segment_final_checksum
                    == scheduler_numeric_progression.expected_device_hidden_segment_final_checksum
        } else {
            !scheduler_numeric_progression.uses_device_hidden_segment_residual_add
                && scheduler_numeric_progression.device_hidden_segment_status == "not-run"
                && scheduler_numeric_progression.device_hidden_segment_residual_adds == 0
                && scheduler_numeric_progression.device_hidden_segment_value_updates == 0
                && scheduler_numeric_progression.device_hidden_segment_residual_add_backend
                    == "not-run"
                && scheduler_numeric_progression.device_hidden_segment_resident_segments == 0
                && scheduler_numeric_progression.device_hidden_segment_resident_values == 0
                && scheduler_numeric_progression.device_hidden_segment_final_checksum == 0.0
                && scheduler_numeric_progression.expected_device_hidden_segment_final_checksum
                    == 0.0
        };
    let scheduler_numeric_progression_available = scheduler_execution_available
        && scheduler_numeric_progression.passed
        && scheduler_numeric_progression.layers == GLM52_NUM_HIDDEN_LAYERS
        && scheduler_numeric_progression.unique_source_rows == scheduler_numeric_unique_rows
        && scheduler_numeric_progression.hidden_dim == GLM52_HIDDEN_SIZE
        && scheduler_numeric_progression.residual_dtype == "bf16"
        && scheduler_numeric_progression.selected_prefill_rows
            == (REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START as usize
                + REAL_FULL_PREFLIGHT_PREFILL_ROWS)
                * GLM52_NUM_HIDDEN_LAYERS
        && scheduler_numeric_progression.selected_decode_rows
            == REAL_FULL_PREFLIGHT_DECODE_ROWS * GLM52_NUM_HIDDEN_LAYERS
        && scheduler_numeric_progression.selected_mtp_rows
            == REAL_FULL_PREFLIGHT_MTP_ROWS * GLM52_NUM_HIDDEN_LAYERS
        && scheduler_numeric_progression.mtp_accepted_rows_per_layer
            == REAL_FULL_PREFLIGHT_MTP_ACCEPTED_ROWS
        && scheduler_numeric_progression.source_segments == scheduler_numeric_source_segments
        && scheduler_numeric_progression.attention_residual_adds
            == scheduler_numeric_source_segments
        && scheduler_numeric_progression.mlp_residual_adds == scheduler_numeric_source_segments
        && scheduler_numeric_progression
            .attention_residual_add_backend
            .contains("residual-add-bf16")
        && scheduler_numeric_progression
            .mlp_residual_add_backend
            .contains("residual-add-bf16")
        && scheduler_numeric_progression.attention_value_updates == scheduler_numeric_value_updates
        && scheduler_numeric_progression.mlp_value_updates == scheduler_numeric_value_updates
        && scheduler_numeric_device_attention_delta_available
        && scheduler_numeric_device_attention_prefix_overlay_available
        && scheduler_numeric_device_delta_template_available
        && scheduler_numeric_device_mlp_delta_available
        && scheduler_numeric_device_real_dense_mlp_delta_available
        && scheduler_numeric_device_real_sparse_shared_mlp_delta_available
        && scheduler_numeric_device_real_sparse_routed_mlp_delta_available
        && scheduler_numeric_device_segments_available;
    let expert_execution_dry_run_available = !expert_hosts.is_empty()
        && expert_execution_dry_run.sparse_layers == sparse_layers
        && expert_execution_dry_run.routed_experts_per_layer == GLM52_ROUTED_EXPERTS
        && expert_execution_dry_run.expected_routed_experts == sparse_layers * GLM52_ROUTED_EXPERTS
        && expert_execution_dry_run.covered_sparse_layers == sparse_layers
        && expert_execution_dry_run.fully_covered_experts
            == expert_execution_dry_run.expected_routed_experts
        && expert_execution_dry_run.missing_weight_experts == 0
        && expert_execution_dry_run.missing_quant_metadata_experts == 0
        && expert_execution_dry_run.owner_partitions.len() == expert_hosts.len()
        && expert_execution_dry_run.planned_sparse_expert_batches
            == (GLM52_NUM_HIDDEN_LAYERS - GLM52_FIRST_K_DENSE_REPLACE) * 2
        && expert_execution_dry_run.planned_expert_batch_rows
            == scheduler_execution_dry_run.sparse_expert_batch_rows
        && expert_execution_dry_run.planned_route_entries
            == scheduler_execution_dry_run.sparse_expert_batch_routes
        && expert_execution_dry_run.all_sparse_layers_have_all_experts
        && expert_execution_dry_run.all_experts_have_weight_tensors
        && expert_execution_dry_run.all_experts_have_quant_metadata
        && !expert_execution_dry_run.numeric_execution_implemented;
    let expert_numeric_probe = &expert_execution_dry_run.real_nvfp4_numeric_probe;
    let expert_all_layer_probe = &expert_execution_dry_run.real_nvfp4_all_layer_probe;
    let expert_residual_chain_probe = &expert_execution_dry_run.real_nvfp4_residual_chain_probe;
    let expert_shared_chain_probe = &expert_execution_dry_run.real_sparse_mlp_shared_chain_probe;
    let expert_scheduler_rows_probe = &expert_execution_dry_run.real_nvfp4_scheduler_rows_probe;
    let kv_backing_store_available = kv_backing_store_dry_run.layer_count
        == GLM52_NUM_HIDDEN_LAYERS
        && kv_backing_store_dry_run.bytes_per_model_token == kv_config.bytes_per_token()
        && kv_backing_store_dry_run.backed_prefill_writes == GLM52_NUM_HIDDEN_LAYERS
        && kv_backing_store_dry_run.backed_decode_writes == GLM52_NUM_HIDDEN_LAYERS
        && kv_backing_store_dry_run.committed_mtp_writes
            == GLM52_NUM_HIDDEN_LAYERS * REAL_FULL_PREFLIGHT_MTP_ACCEPTED_ROWS
        && kv_backing_store_dry_run.discarded_mtp_writes
            == GLM52_NUM_HIDDEN_LAYERS
                * (REAL_FULL_PREFLIGHT_MTP_ROWS - REAL_FULL_PREFLIGHT_MTP_ACCEPTED_ROWS);
    let attention_kv_io_available = attention_kv_io_dry_run.layer_count == GLM52_NUM_HIDDEN_LAYERS
        && attention_kv_io_dry_run.prefix_prefill_wave_writes == GLM52_NUM_HIDDEN_LAYERS
        && attention_kv_io_dry_run.later_prefill_prefix_read_blocks == GLM52_NUM_HIDDEN_LAYERS
        && attention_kv_io_dry_run.decode_prefix_read_blocks == GLM52_NUM_HIDDEN_LAYERS * 2
        && attention_kv_io_dry_run.mtp_tentative_wave_writes
            == GLM52_NUM_HIDDEN_LAYERS * REAL_FULL_PREFLIGHT_MTP_ROWS
        && attention_kv_io_dry_run.mtp_committed_writes
            == GLM52_NUM_HIDDEN_LAYERS * REAL_FULL_PREFLIGHT_MTP_ACCEPTED_ROWS
        && attention_kv_io_dry_run.layerwave_payload_count_mismatch_guard;
    let attention_kv_binding_available = attention_kv_binding_dry_run.attention_layers
        == GLM52_NUM_HIDDEN_LAYERS
        && attention_kv_binding_dry_run.common_layers_with_required_tensors
            == GLM52_NUM_HIDDEN_LAYERS
        && attention_kv_binding_dry_run.catalog_dsa_indexer_layers_match_kv_config
        && attention_kv_binding_dry_run.dsa_indexer_layers_with_required_tensors
            == attention_kv_binding_dry_run.dsa_indexer_layers
        && attention_kv_binding_dry_run.kv_layer_bytes_sum == kv_config.bytes_per_token()
        && attention_kv_binding_dry_run.kv_io_layer_count == GLM52_NUM_HIDDEN_LAYERS
        && attention_kv_binding_dry_run.all_attention_layers_bound_to_kv;
    RealFullPreflightAvailability {
        sparse_layers,
        facts_match,
        full_catalog_coverage,
        execution_plan_available,
        residual_stream_dry_run_available,
        residual_kernel,
        residual_numeric_kernel_available,
        dense_prefix_probe,
        attention_residual_prefix_probe,
        mla_rope_attention_probe,
        dsa_indexer_attention_probe,
        attention_dense_sparse_prefix_probe,
        sampling_dry_run_available,
        sampling_lm_head_default_chunk_available,
        sampling_real_lm_head_probe,
        scheduler_dry_run_available,
        scheduler_real_tensor_catalog_available,
        scheduler_execution_available,
        scheduler_numeric_progression,
        scheduler_numeric_progression_available,
        expert_execution_dry_run_available,
        expert_numeric_probe,
        expert_all_layer_probe,
        expert_residual_chain_probe,
        expert_shared_chain_probe,
        expert_scheduler_rows_probe,
        kv_backing_store_available,
        attention_kv_io_available,
        attention_kv_binding_available,
    }
}
