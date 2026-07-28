use glmrt_core::{
    GLM52_FIRST_K_DENSE_REPLACE, GLM52_HIDDEN_SIZE, GLM52_NUM_HIDDEN_LAYERS, GLM52_TOP_K,
};

use super::super::super::constants::{
    REAL_FULL_PREFLIGHT_DECODE_ROWS, REAL_FULL_PREFLIGHT_MTP_ACCEPTED_ROWS,
    REAL_FULL_PREFLIGHT_MTP_ROWS, REAL_FULL_PREFLIGHT_PREFILL_ROWS,
};
use super::super::super::types::*;
use super::availability::RealFullPreflightAvailability;
use super::RealFullPreflightRequirementInputs;

pub(super) fn available_runtime_requirements(
    inputs: &RealFullPreflightRequirementInputs<'_>,
    availability: &RealFullPreflightAvailability<'_>,
) -> Vec<RealFullRequirement> {
    let catalog = inputs.catalog;
    let coverage = inputs.coverage;
    let kv_config = inputs.kv_config;
    let residual_stream_dry_run = inputs.residual_stream_dry_run;
    let sampling_dry_run = inputs.sampling_dry_run;
    let scheduler_dry_run = inputs.scheduler_dry_run;
    let expert_execution_dry_run = inputs.expert_execution_dry_run;
    let scheduler_execution_dry_run = inputs.scheduler_execution_dry_run;
    let kv_backing_store_dry_run = inputs.kv_backing_store_dry_run;
    let attention_kv_io_dry_run = inputs.attention_kv_io_dry_run;
    let attention_kv_binding_dry_run = inputs.attention_kv_binding_dry_run;
    let coordinator_resident_preload = inputs.coordinator_resident_preload;

    let sparse_layers = availability.sparse_layers;
    let facts_match = availability.facts_match;
    let full_catalog_coverage = availability.full_catalog_coverage;
    let execution_plan_available = availability.execution_plan_available;
    let residual_stream_dry_run_available = availability.residual_stream_dry_run_available;
    let residual_kernel = availability.residual_kernel;
    let residual_numeric_kernel_available = availability.residual_numeric_kernel_available;
    let sampling_dry_run_available = availability.sampling_dry_run_available;
    let sampling_lm_head_default_chunk_available =
        availability.sampling_lm_head_default_chunk_available;
    let sampling_real_lm_head_probe = availability.sampling_real_lm_head_probe;
    let scheduler_dry_run_available = availability.scheduler_dry_run_available;
    let scheduler_real_tensor_catalog_available =
        availability.scheduler_real_tensor_catalog_available;
    let scheduler_execution_available = availability.scheduler_execution_available;
    let scheduler_numeric_progression = availability.scheduler_numeric_progression;
    let scheduler_numeric_progression_available =
        availability.scheduler_numeric_progression_available;
    let expert_execution_dry_run_available = availability.expert_execution_dry_run_available;
    let expert_numeric_probe = availability.expert_numeric_probe;
    let kv_backing_store_available = availability.kv_backing_store_available;
    let attention_kv_io_available = availability.attention_kv_io_available;
    let attention_kv_binding_available = availability.attention_kv_binding_available;
    vec![
        RealFullRequirement {
            name: "offline_catalog_loaded",
            passed: true,
            evidence: format!("catalog tensors={}", catalog.tensors.len()),
            blocker: None,
        },
        RealFullRequirement {
            name: "glm52_model_facts_match",
            passed: facts_match,
            evidence: format!(
                "hidden_size={} layers={} dense_layers={} routed_experts={} top_k={}",
                catalog.facts.hidden_size,
                catalog.facts.num_hidden_layers,
                catalog.facts.first_k_dense_replace,
                catalog.facts.routed_experts,
                catalog.facts.top_k
            ),
            blocker: (!facts_match)
                .then_some("catalog facts do not match GLM-5.2 runtime constants"),
        },
        RealFullRequirement {
            name: "full_tensor_catalog_coverage",
            passed: full_catalog_coverage,
            evidence: format!(
                "layers={} sparse_layers={} dense_layers={} embeddings={} lm_head={}",
                coverage.hidden_layers_with_any_tensor,
                coverage.sparse_layers_with_routed_experts,
                coverage.dense_layers_with_dense_mlp,
                coverage.embedding_tensors,
                coverage.lm_head_tensors
            ),
            blocker: (!full_catalog_coverage)
                .then_some("catalog does not expose all tensors needed for full GLM execution"),
        },
        RealFullRequirement {
            name: "compressed_kv_accounting_available",
            passed: true,
            evidence: format!(
                "bytes_per_token={} capacity_bytes={}",
                kv_config.bytes_per_token(),
                kv_config.capacity_bytes()
            ),
            blocker: None,
        },
        RealFullRequirement {
            name: "full_model_execution_plan_available",
            passed: execution_plan_available,
            evidence: format!(
                "layers={} dense={} sparse={} prefill_rows={} decode_sparse_roundtrips={}",
                GLM52_NUM_HIDDEN_LAYERS,
                GLM52_FIRST_K_DENSE_REPLACE,
                sparse_layers,
                REAL_FULL_PREFLIGHT_PREFILL_ROWS,
                sparse_layers
            ),
            blocker: (!execution_plan_available).then_some(
                "runtime contract does not cover every GLM-5.2 layer and sparse expert exchange",
            ),
        },
        coordinator_resident_preload_requirement(coordinator_resident_preload),
        RealFullRequirement {
            name: "full_model_scheduler_dry_run_available",
            passed: scheduler_dry_run_available,
            evidence: format!(
                "layerwaves={} sparse_expert_batches={} mixed_rows={} mixed_routes={} protocol_v2_batch_status={} protocol_v2_request_wire_bytes={} protocol_v2_response_wire_bytes={} protocol_v2_reconstructed_rows={} protocol_v2_reconstruction_payload_matches={} protocol_v2_host_batches={} protocol_v2_host_batch_routes={} protocol_v2_host_batch_graph_counts_valid={} protocol_v2_host_request_frames={} protocol_v2_host_request_wire_bytes={} protocol_v2_host_response_frames={} protocol_v2_host_response_wire_bytes={} protocol_v2_host_wire_envelopes_valid={}",
                GLM52_NUM_HIDDEN_LAYERS * 3,
                sparse_layers,
                REAL_FULL_PREFLIGHT_PREFILL_ROWS
                    + REAL_FULL_PREFLIGHT_MTP_ROWS
                    + REAL_FULL_PREFLIGHT_DECODE_ROWS,
                (REAL_FULL_PREFLIGHT_PREFILL_ROWS
                    + REAL_FULL_PREFLIGHT_MTP_ROWS
                    + REAL_FULL_PREFLIGHT_DECODE_ROWS)
                    * GLM52_TOP_K,
                scheduler_dry_run.protocol_v2_batch_probe.status,
                scheduler_dry_run.protocol_v2_batch_probe.request_wire_bytes,
                scheduler_dry_run.protocol_v2_batch_probe.response_wire_bytes,
                scheduler_dry_run.protocol_v2_batch_probe.reconstructed_response_rows,
                scheduler_dry_run
                    .protocol_v2_batch_probe
                    .reconstructed_response_payload_matches,
                scheduler_dry_run.protocol_v2_batch_probe.host_batches,
                scheduler_dry_run.protocol_v2_batch_probe.host_batch_routes,
                scheduler_dry_run
                    .protocol_v2_batch_probe
                    .host_batch_graph_counts_valid,
                scheduler_dry_run.protocol_v2_batch_probe.host_request_frames,
                scheduler_dry_run
                    .protocol_v2_batch_probe
                    .host_request_wire_bytes,
                scheduler_dry_run
                    .protocol_v2_batch_probe
                    .host_response_frames,
                scheduler_dry_run
                    .protocol_v2_batch_probe
                    .host_response_wire_bytes,
                scheduler_dry_run
                    .protocol_v2_batch_probe
                    .host_wire_envelopes_valid
            ),
            blocker: (!scheduler_dry_run_available)
                .then_some("full-model LayerWave and ExpertBatch dry-run is not internally complete"),
        },
        RealFullRequirement {
            name: "full_model_admitted_scheduler_execution_dry_run_available",
            passed: scheduler_execution_available,
            evidence: format!(
                "iterations={} selected_layerwaves={} sparse_batches={} host_batch_sets={} host_batches={} host_rows={} host_routes={} host_expert_tiles={} host_routes_match_global={} host_graph_counts_valid={} host_request_frames={} host_request_rows={} host_request_routes={} host_request_payload_bytes={} host_request_wire_bytes={} host_response_frames={} host_response_rows={} host_response_payload_bytes={} host_response_wire_bytes={} host_wire_envelopes_valid={} scheduler_real_tensor_catalog_available={} kv_reads={} device_kv_status={} device_kv_writes={} device_kv_reads={} device_kv_bytes={} projected_device_kv_writes={} projected_device_kv_write_bytes={} synthetic_kv_payload_writes={} uses_device_kv_cache={} device_attention_resident_uploads={} device_attention_resident_query_shapes={} device_attention_resident_buffer_uses={} device_attention_status={} device_attention_launches={} device_attention_hidden_projection_launches={} device_attention_rows={} device_attention_query_rows={} device_attention_kv_descriptors={} device_attention_output_bytes={} device_attention_output_values={} device_attention_output_finite_values={} device_attention_output_nonzero_values={} device_attention_output_checksum={} uses_device_kv_attention={} full_context_device_attention_complete={}",
                scheduler_execution_dry_run.iterations,
                scheduler_execution_dry_run.selected_layerwaves,
                scheduler_execution_dry_run.sparse_expert_batches,
                scheduler_execution_dry_run.sparse_expert_host_batch_sets,
                scheduler_execution_dry_run.sparse_expert_host_batches,
                scheduler_execution_dry_run.sparse_expert_host_batch_rows,
                scheduler_execution_dry_run.sparse_expert_host_batch_routes,
                scheduler_execution_dry_run.sparse_expert_host_batch_expert_tiles,
                scheduler_execution_dry_run.sparse_expert_host_batch_routes_match_global,
                scheduler_execution_dry_run.sparse_expert_host_batch_graph_counts_valid,
                scheduler_execution_dry_run.sparse_expert_host_request_frames,
                scheduler_execution_dry_run.sparse_expert_host_request_rows,
                scheduler_execution_dry_run.sparse_expert_host_request_routes,
                scheduler_execution_dry_run.sparse_expert_host_request_payload_bytes,
                scheduler_execution_dry_run.sparse_expert_host_request_wire_bytes,
                scheduler_execution_dry_run.sparse_expert_host_response_frames,
                scheduler_execution_dry_run.sparse_expert_host_response_rows,
                scheduler_execution_dry_run.sparse_expert_host_response_payload_bytes,
                scheduler_execution_dry_run.sparse_expert_host_response_wire_bytes,
                scheduler_execution_dry_run.sparse_expert_host_wire_envelopes_valid,
                scheduler_real_tensor_catalog_available,
                scheduler_execution_dry_run.kv_read_blocks,
                scheduler_execution_dry_run.device_kv_status,
                scheduler_execution_dry_run.device_kv_writes,
                scheduler_execution_dry_run.device_kv_reads,
                scheduler_execution_dry_run.device_kv_bytes,
                scheduler_execution_dry_run.projected_device_kv_writes,
                scheduler_execution_dry_run.projected_device_kv_write_bytes,
                scheduler_execution_dry_run.synthetic_kv_payload_writes,
                scheduler_execution_dry_run.uses_device_kv_cache,
                scheduler_execution_dry_run.device_attention_resident_uploads,
                scheduler_execution_dry_run.device_attention_resident_query_shapes,
                scheduler_execution_dry_run.device_attention_resident_buffer_uses,
                scheduler_execution_dry_run.device_attention_status,
                scheduler_execution_dry_run.device_attention_launches,
                scheduler_execution_dry_run.device_attention_hidden_projection_launches,
                scheduler_execution_dry_run.device_attention_rows,
                scheduler_execution_dry_run.device_attention_query_rows,
                scheduler_execution_dry_run.device_attention_kv_descriptors,
                scheduler_execution_dry_run.device_attention_output_bytes,
                scheduler_execution_dry_run.device_attention_output_values,
                scheduler_execution_dry_run.device_attention_output_finite_values,
                scheduler_execution_dry_run.device_attention_output_nonzero_values,
                scheduler_execution_dry_run.device_attention_output_checksum,
                scheduler_execution_dry_run.uses_device_kv_attention,
                scheduler_execution_dry_run.full_context_device_attention_complete
            ),
            blocker: (!scheduler_execution_available).then_some(
                "admitted full-model scheduler execution dry-run is not internally complete or lacks hidden-width full-context device attention",
            ),
        },
        RealFullRequirement {
            name: "scheduler_numeric_progression_self_test_available",
            passed: scheduler_numeric_progression_available,
            evidence: format!(
                "source_rows={} source_segments={} hidden_dim={} residual_dtype={} visible_checksum={} expected_visible_checksum={} rejected_mtp_checksum={} attention_residual_adds={} mlp_residual_adds={} attention_backend={} mlp_backend={} attention_updates={} mlp_updates={} device_attention_output_delta_status={} device_attention_output_delta_rows={} device_attention_output_delta_values={} device_attention_output_delta_checksum={} device_attention_output_delta_backend={} device_attention_output_delta_device_prefix_rows={} device_attention_output_delta_device_prefix_values={} device_attention_output_delta_device_prefix_backend={} uses_device_attention_output_delta={} device_delta_template_status={} device_delta_template_uploads={} device_delta_template_uses={} device_delta_template_resident_values={} device_mlp_delta_status={} device_mlp_delta_rows={} device_mlp_delta_values={} device_mlp_delta_checksum={} device_mlp_delta_backend={} device_mlp_weight_uploads={} device_mlp_weight_resident_values={} uses_device_mlp_delta={} device_real_dense_mlp_delta_status={} device_real_dense_mlp_delta_rows={} device_real_dense_mlp_delta_values={} device_real_dense_mlp_delta_checksum={} device_real_dense_mlp_delta_backend={} device_real_dense_mlp_norm_backend={} device_real_dense_mlp_weight_tensors={} device_real_dense_mlp_weight_bytes={} device_real_dense_mlp_layers={} device_real_dense_mlp_source_segments={} uses_device_real_dense_mlp_delta={} device_real_sparse_shared_mlp_delta_status={} device_real_sparse_shared_mlp_delta_rows={} device_real_sparse_shared_mlp_delta_values={} device_real_sparse_shared_mlp_delta_checksum={} device_real_sparse_shared_mlp_delta_backend={} device_real_sparse_shared_mlp_norm_backend={} device_real_sparse_shared_mlp_weight_tensors={} device_real_sparse_shared_mlp_weight_bytes={} device_real_sparse_shared_mlp_layers={} device_real_sparse_shared_mlp_source_segments={} uses_device_real_sparse_shared_mlp_delta={} device_real_sparse_routed_mlp_delta_status={} device_real_sparse_routed_mlp_delta_rows={} device_real_sparse_routed_mlp_delta_values={} device_real_sparse_routed_mlp_delta_checksum={} device_real_sparse_routed_mlp_delta_backend={} device_real_sparse_routed_mlp_route_backend={} device_real_sparse_routed_mlp_router_backend={} device_real_sparse_routed_mlp_routes={} device_real_sparse_routed_mlp_router_weight_bytes={} device_real_sparse_routed_mlp_router_bias_bytes={} device_real_sparse_routed_mlp_route_cache_cuda_entries={} device_real_sparse_routed_mlp_route_cache_cuda_uploads={} device_real_sparse_routed_mlp_route_cache_cuda_hits={} device_real_sparse_routed_mlp_router_cache_entries={} device_real_sparse_routed_mlp_router_cache_hits={} device_real_sparse_routed_mlp_layers={} device_real_sparse_routed_mlp_source_segments={} uses_device_real_sparse_routed_mlp_delta={} device_hidden_segment_status={} device_hidden_segment_residual_adds={} device_hidden_segment_updates={} device_hidden_segment_backend={} device_hidden_segment_resident_segments={} device_hidden_segment_resident_values={} device_hidden_segment_final_checksum={} expected_device_hidden_segment_final_checksum={} uses_device_hidden_segment_residual_add={}",
                scheduler_numeric_progression.unique_source_rows,
                scheduler_numeric_progression.source_segments,
                scheduler_numeric_progression.hidden_dim,
                scheduler_numeric_progression.residual_dtype,
                scheduler_numeric_progression.final_visible_checksum,
                scheduler_numeric_progression.expected_visible_checksum,
                scheduler_numeric_progression.rejected_mtp_checksum,
                scheduler_numeric_progression.attention_residual_adds,
                scheduler_numeric_progression.mlp_residual_adds,
                scheduler_numeric_progression.attention_residual_add_backend,
                scheduler_numeric_progression.mlp_residual_add_backend,
                scheduler_numeric_progression.attention_value_updates,
                scheduler_numeric_progression.mlp_value_updates,
                scheduler_numeric_progression.device_attention_output_delta_status,
                scheduler_numeric_progression.attention_device_output_delta_rows,
                scheduler_numeric_progression.attention_device_output_delta_values,
                scheduler_numeric_progression.attention_device_output_delta_checksum,
                scheduler_numeric_progression.attention_device_output_delta_backend,
                scheduler_numeric_progression.attention_device_output_delta_device_prefix_rows,
                scheduler_numeric_progression.attention_device_output_delta_device_prefix_values,
                scheduler_numeric_progression.attention_device_output_delta_device_prefix_backend,
                scheduler_numeric_progression.uses_device_attention_output_delta,
                scheduler_numeric_progression.device_delta_template_status,
                scheduler_numeric_progression.device_delta_template_uploads,
                scheduler_numeric_progression.device_delta_template_uses,
                scheduler_numeric_progression.device_delta_template_resident_values,
                scheduler_numeric_progression.device_mlp_delta_status,
                scheduler_numeric_progression.device_mlp_delta_rows,
                scheduler_numeric_progression.device_mlp_delta_values,
                scheduler_numeric_progression.device_mlp_delta_checksum,
                scheduler_numeric_progression.device_mlp_delta_backend,
                scheduler_numeric_progression.device_mlp_weight_uploads,
                scheduler_numeric_progression.device_mlp_weight_resident_values,
                scheduler_numeric_progression.uses_device_mlp_delta,
                scheduler_numeric_progression.device_real_dense_mlp_delta_status,
                scheduler_numeric_progression.device_real_dense_mlp_delta_rows,
                scheduler_numeric_progression.device_real_dense_mlp_delta_values,
                scheduler_numeric_progression.device_real_dense_mlp_delta_checksum,
                scheduler_numeric_progression.device_real_dense_mlp_delta_backend,
                scheduler_numeric_progression.device_real_dense_mlp_norm_backend,
                scheduler_numeric_progression.device_real_dense_mlp_weight_tensors,
                scheduler_numeric_progression.device_real_dense_mlp_weight_bytes,
                scheduler_numeric_progression.device_real_dense_mlp_layers,
                scheduler_numeric_progression.device_real_dense_mlp_source_segments,
                scheduler_numeric_progression.uses_device_real_dense_mlp_delta,
                scheduler_numeric_progression.device_real_sparse_shared_mlp_delta_status,
                scheduler_numeric_progression.device_real_sparse_shared_mlp_delta_rows,
                scheduler_numeric_progression.device_real_sparse_shared_mlp_delta_values,
                scheduler_numeric_progression.device_real_sparse_shared_mlp_delta_checksum,
                scheduler_numeric_progression.device_real_sparse_shared_mlp_delta_backend,
                scheduler_numeric_progression.device_real_sparse_shared_mlp_norm_backend,
                scheduler_numeric_progression.device_real_sparse_shared_mlp_weight_tensors,
                scheduler_numeric_progression.device_real_sparse_shared_mlp_weight_bytes,
                scheduler_numeric_progression.device_real_sparse_shared_mlp_layers,
                scheduler_numeric_progression.device_real_sparse_shared_mlp_source_segments,
                scheduler_numeric_progression.uses_device_real_sparse_shared_mlp_delta,
                scheduler_numeric_progression.device_real_sparse_routed_mlp_delta_status,
                scheduler_numeric_progression.device_real_sparse_routed_mlp_delta_rows,
                scheduler_numeric_progression.device_real_sparse_routed_mlp_delta_values,
                scheduler_numeric_progression.device_real_sparse_routed_mlp_delta_checksum,
                scheduler_numeric_progression.device_real_sparse_routed_mlp_delta_backend,
                scheduler_numeric_progression.device_real_sparse_routed_mlp_route_backend,
                scheduler_numeric_progression.device_real_sparse_routed_mlp_router_backend,
                scheduler_numeric_progression.device_real_sparse_routed_mlp_routes,
                scheduler_numeric_progression.device_real_sparse_routed_mlp_router_weight_bytes,
                scheduler_numeric_progression.device_real_sparse_routed_mlp_router_bias_bytes,
                scheduler_numeric_progression.device_real_sparse_routed_mlp_route_cache_cuda_entries,
                scheduler_numeric_progression.device_real_sparse_routed_mlp_route_cache_cuda_uploads,
                scheduler_numeric_progression.device_real_sparse_routed_mlp_route_cache_cuda_hits,
                scheduler_numeric_progression.device_real_sparse_routed_mlp_router_cache_entries,
                scheduler_numeric_progression.device_real_sparse_routed_mlp_router_cache_hits,
                scheduler_numeric_progression.device_real_sparse_routed_mlp_layers,
                scheduler_numeric_progression.device_real_sparse_routed_mlp_source_segments,
                scheduler_numeric_progression.uses_device_real_sparse_routed_mlp_delta,
                scheduler_numeric_progression.device_hidden_segment_status,
                scheduler_numeric_progression.device_hidden_segment_residual_adds,
                scheduler_numeric_progression.device_hidden_segment_value_updates,
                scheduler_numeric_progression.device_hidden_segment_residual_add_backend,
                scheduler_numeric_progression.device_hidden_segment_resident_segments,
                scheduler_numeric_progression.device_hidden_segment_resident_values,
                scheduler_numeric_progression.device_hidden_segment_final_checksum,
                scheduler_numeric_progression.expected_device_hidden_segment_final_checksum,
                scheduler_numeric_progression.uses_device_hidden_segment_residual_add
            ),
            blocker: (!scheduler_numeric_progression_available).then_some(
                "admitted scheduler rows do not pass numeric residual progression self-test",
            ),
        },
        RealFullRequirement {
            name: "full_model_residual_stream_dry_run_available",
            passed: residual_stream_dry_run_available,
            evidence: format!(
                "layers={} rows={} residual_adds={} terminal_stages={} attention_prefix_status={} attention_prefix_passed={} mla_rope_status={} mla_rope_passed={} mla_rope_heads={} mla_rope_scores={} dsa_indexer_status={} dsa_indexer_passed={} dsa_indexer_layer={} dsa_candidate_rows={} dsa_top_k={} dense_prefix_status={} dense_prefix_passed={} layer_ordered_prefix_status={} layer_ordered_prefix_passed={} composed_prefix_status={} composed_prefix_passed={}",
                GLM52_NUM_HIDDEN_LAYERS,
                REAL_FULL_PREFLIGHT_PREFILL_ROWS
                    + REAL_FULL_PREFLIGHT_MTP_ROWS
                    + REAL_FULL_PREFLIGHT_DECODE_ROWS,
                GLM52_NUM_HIDDEN_LAYERS * 2,
                "final_norm,lm_head,full_vocab_sampling",
                residual_stream_dry_run
                    .real_attention_residual_prefix_probe
                    .status,
                residual_stream_dry_run
                    .real_attention_residual_prefix_probe
                    .passed,
                residual_stream_dry_run.real_mla_rope_attention_probe.status,
                residual_stream_dry_run.real_mla_rope_attention_probe.passed,
                residual_stream_dry_run
                    .real_mla_rope_attention_probe
                    .attention_heads,
                residual_stream_dry_run
                    .real_mla_rope_attention_probe
                    .causal_attention_scores,
                residual_stream_dry_run
                    .real_dsa_indexer_attention_probe
                    .status,
                residual_stream_dry_run
                    .real_dsa_indexer_attention_probe
                    .passed,
                residual_stream_dry_run
                    .real_dsa_indexer_attention_probe
                    .layer_id,
                residual_stream_dry_run
                    .real_dsa_indexer_attention_probe
                    .candidate_rows,
                residual_stream_dry_run
                    .real_dsa_indexer_attention_probe
                    .dsa_top_k,
                residual_stream_dry_run.real_dense_prefix_probe.status,
                residual_stream_dry_run.real_dense_prefix_probe.passed,
                residual_stream_dry_run
                    .real_layer_ordered_prefix_probe
                    .status,
                residual_stream_dry_run
                    .real_layer_ordered_prefix_probe
                    .passed,
                residual_stream_dry_run
                    .real_attention_dense_sparse_prefix_probe
                    .status,
                residual_stream_dry_run
                    .real_attention_dense_sparse_prefix_probe
                    .passed
            ),
            blocker: (!residual_stream_dry_run_available)
                .then_some("full-model residual stream dry-run does not cover every layer boundary"),
        },
        RealFullRequirement {
            name: "full_residual_numeric_accumulator_kernel_available",
            passed: residual_numeric_kernel_available,
            evidence: format!(
                "self_test_layers={} residual_adds={} values_updated={} checksum={}",
                residual_kernel.layers,
                residual_kernel.residual_adds,
                residual_kernel.values_updated,
                residual_kernel.final_checksum
            ),
            blocker: (!residual_numeric_kernel_available)
                .then_some("numeric residual accumulator kernel self-test failed"),
        },
        RealFullRequirement {
            name: "full_vocab_sampling_dry_run_available",
            passed: sampling_dry_run_available,
            evidence: format!(
                "lm_head=lm_head.weight vocab={} hidden={} chunks={} chunk_rows={} sampled_rows={} default_chunk_status={} default_chunk_passed={} default_chunk_rows={} default_chunk_bytes={}",
                sampling_dry_run.vocab_size,
                GLM52_HIDDEN_SIZE,
                sampling_dry_run.chunk_count,
                sampling_dry_run.chunk_rows,
                REAL_FULL_PREFLIGHT_DECODE_ROWS,
                sampling_dry_run.real_lm_head_default_chunk_probe.status,
                sampling_dry_run.real_lm_head_default_chunk_probe.passed,
                sampling_dry_run.real_lm_head_default_chunk_probe.rows_scored,
                sampling_dry_run
                    .real_lm_head_default_chunk_probe
                    .lm_head_bytes_read
            ),
            blocker: (!sampling_dry_run_available).then_some(
                "full-vocabulary lm_head sampling dry-run does not cover the real vocabulary",
            ),
        },
        RealFullRequirement {
            name: "bf16_lm_head_sampler_kernel_available",
            passed: sampling_lm_head_default_chunk_available,
            evidence: format!(
                "default_chunk_status={} default_chunk_passed={} rows={} logits={} bytes={} logits_backend={:?} argmax_backend={:?} sampler_backend={:?} sample_top_k={:?} sample_top_p={:?} sample_temperature={:?} top_token={:?} sampled_token={:?} top_logit={:?} sampled_score={:?} full_vocab_probe_status={} full_vocab_rows={} full_vocab_bytes={}",
                sampling_dry_run.real_lm_head_default_chunk_probe.status,
                sampling_dry_run.real_lm_head_default_chunk_probe.passed,
                sampling_dry_run.real_lm_head_default_chunk_probe.rows_scored,
                sampling_dry_run
                    .real_lm_head_default_chunk_probe
                    .logits_evaluated,
                sampling_dry_run
                    .real_lm_head_default_chunk_probe
                    .lm_head_bytes_read,
                sampling_dry_run
                    .real_lm_head_default_chunk_probe
                    .logits_kernel_backend,
                sampling_dry_run
                    .real_lm_head_default_chunk_probe
                    .argmax_kernel_backend,
                sampling_dry_run
                    .real_lm_head_default_chunk_probe
                    .sampler_kernel_backend,
                sampling_dry_run
                    .real_lm_head_default_chunk_probe
                    .sample_top_k,
                sampling_dry_run
                    .real_lm_head_default_chunk_probe
                    .sample_top_p,
                sampling_dry_run
                    .real_lm_head_default_chunk_probe
                    .sample_temperature,
                sampling_dry_run.real_lm_head_default_chunk_probe.top_token_id,
                sampling_dry_run
                    .real_lm_head_default_chunk_probe
                    .sampled_token_id,
                sampling_dry_run.real_lm_head_default_chunk_probe.top_logit,
                sampling_dry_run
                    .real_lm_head_default_chunk_probe
                    .sampled_score,
                sampling_real_lm_head_probe.status,
                sampling_real_lm_head_probe.logits_evaluated,
                sampling_real_lm_head_probe.lm_head_bytes_read
            ),
            blocker: (!sampling_lm_head_default_chunk_available).then_some(
                "BF16 lm_head sampler wrapper has not scored a real lm_head chunk",
            ),
        },
        RealFullRequirement {
            name: "all_layer_real_nvfp4_expert_execution_dry_run_available",
            passed: expert_execution_dry_run_available,
            evidence: format!(
                "covered_experts={}/{} weight_tensors={} quant_metadata_tensors={} planned_batches={} route_entries={} real_nvfp4_probe_status={} real_nvfp4_probe_passed={} real_nvfp4_probe_routes={} real_nvfp4_probe_weight_bytes={}",
                expert_execution_dry_run.fully_covered_experts,
                expert_execution_dry_run.expected_routed_experts,
                expert_execution_dry_run.routed_weight_tensors,
                expert_execution_dry_run.routed_quant_metadata_tensors,
                expert_execution_dry_run.planned_sparse_expert_batches,
                expert_execution_dry_run.planned_route_entries,
                expert_numeric_probe.status,
                expert_numeric_probe.passed,
                expert_numeric_probe.route_count,
                expert_numeric_probe.weight_bytes_read
            ),
            blocker: (!expert_execution_dry_run_available).then_some(
                "full-model routed expert coverage/placement dry-run does not cover every sparse layer expert",
            ),
        },
        RealFullRequirement {
            name: "full_model_kv_backing_store_dry_run_available",
            passed: kv_backing_store_available,
            evidence: format!(
                "prefill_writes={} decode_writes={} committed_mtp={} discarded_mtp={} backed_bytes={}",
                GLM52_NUM_HIDDEN_LAYERS,
                GLM52_NUM_HIDDEN_LAYERS,
                GLM52_NUM_HIDDEN_LAYERS * REAL_FULL_PREFLIGHT_MTP_ACCEPTED_ROWS,
                GLM52_NUM_HIDDEN_LAYERS
                    * (REAL_FULL_PREFLIGHT_MTP_ROWS - REAL_FULL_PREFLIGHT_MTP_ACCEPTED_ROWS),
                kv_backing_store_dry_run.backed_bytes_after_discard
            ),
            blocker: (!kv_backing_store_available)
                .then_some("full-model compressed KV backing-store dry-run is not internally complete"),
        },
        RealFullRequirement {
            name: "full_model_attention_kv_io_dry_run_available",
            passed: attention_kv_io_available,
            evidence: format!(
                "prefill_reads={} decode_reads={} mtp_reads={} tentative_mtp_writes={} device_kv_status={} device_kv_writes={} device_kv_reads={} device_kv_bytes={} uses_device_kv_cache={}",
                GLM52_NUM_HIDDEN_LAYERS,
                GLM52_NUM_HIDDEN_LAYERS * 2,
                GLM52_NUM_HIDDEN_LAYERS * 2,
                GLM52_NUM_HIDDEN_LAYERS * REAL_FULL_PREFLIGHT_MTP_ROWS,
                attention_kv_io_dry_run.device_kv_status,
                attention_kv_io_dry_run.device_kv_writes,
                attention_kv_io_dry_run.device_kv_reads,
                attention_kv_io_dry_run.device_kv_bytes,
                attention_kv_io_dry_run.uses_device_kv_cache
            ),
            blocker: (!attention_kv_io_available)
                .then_some("LayerWave-driven attention KV I/O dry-run is not internally complete"),
        },
        RealFullRequirement {
            name: "real_attention_kv_binding_dry_run_available",
            passed: attention_kv_binding_available,
            evidence: format!(
                "attention_layers={} attention_tensors={} dsa_layers={} kv_layer_bytes_sum={} kv_io_layers={} kv_io_prefix_reads={}",
                attention_kv_binding_dry_run.attention_layers,
                attention_kv_binding_dry_run.attention_tensors,
                attention_kv_binding_dry_run.dsa_indexer_layers,
                attention_kv_binding_dry_run.kv_layer_bytes_sum,
                attention_kv_binding_dry_run.kv_io_layer_count,
                attention_kv_binding_dry_run.kv_io_prefix_read_blocks
            ),
            blocker: (!attention_kv_binding_available).then_some(
                "real attention tensor coverage is not bound to the compressed KV layer-byte plan",
            ),
        },
    ]
}

pub(super) fn coordinator_resident_preload_requirement(
    coordinator_resident_preload: &RealFullCoordinatorResidentPreloadPlan,
) -> RealFullRequirement {
    let coordinator_resident_preload_complete = coordinator_resident_preload.startup_required
        && coordinator_resident_preload.uses_named_resident_buffers
        && coordinator_resident_preload.selected_tensor_count > 0
        && coordinator_resident_preload.selected_tensor_bytes > 0
        && coordinator_resident_preload.required_roles_present
            == coordinator_resident_preload.required_role_count
        && coordinator_resident_preload
            .missing_required_roles
            .is_empty()
        && coordinator_resident_preload.selected_tensor_count_matches_roles
        && coordinator_resident_preload.selected_tensor_bytes_matches_roles;

    RealFullRequirement {
        name: "coordinator_startup_resident_preload_plan_available",
        passed: coordinator_resident_preload_complete,
        evidence: format!(
            "status={} selected_tensors={} selected_bytes={} loaded_bytes={} bf16_tensors={} non_bf16_tensors={} required_roles={}/{} missing_required_roles={:?} count_matches_roles={} bytes_matches_roles={} skipped_routed_expert_tensors={} skipped_quantization_tensors={}",
            coordinator_resident_preload.status,
            coordinator_resident_preload.selected_tensor_count,
            coordinator_resident_preload.selected_tensor_bytes,
            coordinator_resident_preload.loaded_tensor_bytes,
            coordinator_resident_preload.bf16_tensor_count,
            coordinator_resident_preload.non_bf16_tensor_count,
            coordinator_resident_preload.required_roles_present,
            coordinator_resident_preload.required_role_count,
            coordinator_resident_preload.missing_required_roles,
            coordinator_resident_preload.selected_tensor_count_matches_roles,
            coordinator_resident_preload.selected_tensor_bytes_matches_roles,
            coordinator_resident_preload.skipped_routed_expert_tensors,
            coordinator_resident_preload.skipped_quantization_tensors
        ),
        blocker: (!coordinator_resident_preload_complete)
            .then_some("coordinator startup resident preload plan does not cover every named immutable role or has inconsistent role accounting"),
    }
}
