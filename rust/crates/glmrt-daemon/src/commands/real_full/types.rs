use glmrt_core::ModelFacts;
use serde::Serialize;
use std::collections::BTreeMap;

mod expert;
mod residual;
mod sampling;
mod scheduler;

pub(super) use expert::*;
pub(super) use residual::*;
pub(super) use sampling::*;
pub(super) use scheduler::*;

#[derive(Debug, Serialize)]
pub(super) struct RealGlmFullPreflightReport {
    pub(super) backend: &'static str,
    pub(super) status: &'static str,
    pub(super) model_id: String,
    pub(super) catalog_path: String,
    pub(super) snapshot_path: String,
    pub(super) catalog_hash: String,
    pub(super) tensor_count: usize,
    pub(super) listen: String,
    pub(super) transport: String,
    pub(super) sparse_transport: RealFullSparseTransportPlan,
    pub(super) expert_hosts: Vec<String>,
    pub(super) model_facts: ModelFacts,
    pub(super) expected_facts: ModelFacts,
    pub(super) role_counts: BTreeMap<String, usize>,
    pub(super) full_model_tensor_coverage: FullModelTensorCoverage,
    pub(super) kv_plan: RealFullKvPlan,
    pub(super) execution_plan: RealFullExecutionPlan,
    pub(super) residual_stream_dry_run: RealFullResidualStreamDryRun,
    pub(super) sampling_dry_run: RealFullSamplingDryRun,
    pub(super) expert_execution_dry_run: RealFullExpertExecutionDryRun,
    pub(super) scheduler_dry_run: RealFullSchedulerDryRun,
    pub(super) scheduler_execution_dry_run: RealFullSchedulerExecutionDryRun,
    pub(super) kv_backing_store_dry_run: RealFullKvBackingStoreDryRun,
    pub(super) attention_kv_io_dry_run: RealFullAttentionKvIoDryRun,
    pub(super) attention_kv_binding_dry_run: RealFullAttentionKvBindingDryRun,
    pub(super) coordinator_resident_preload: RealFullCoordinatorResidentPreloadPlan,
    pub(super) requirements: Vec<RealFullRequirement>,
    pub(super) blocker: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct RealFullSparseTransportPlan {
    pub(super) transport: String,
    pub(super) status: String,
    pub(super) sparse_dispatch_available: bool,
    pub(super) scheduler_dispatch_backend: Option<String>,
    pub(super) supports_rdma: bool,
    pub(super) supports_host_registered_buffers: bool,
    pub(super) requires_pinned_host_memory: bool,
    pub(super) app_transport_implemented: bool,
    pub(super) app_transport_status: String,
    pub(super) preflight_ok: bool,
    pub(super) preflight_error: Option<String>,
    pub(super) frame_protocol: Option<String>,
    pub(super) blocker: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct RealFullCoordinatorResidentPreloadPlan {
    pub(super) status: &'static str,
    pub(super) scope: &'static str,
    pub(super) startup_required: bool,
    pub(super) selected_tensor_count: usize,
    pub(super) selected_tensor_bytes: u64,
    pub(super) loaded_tensor_bytes: u64,
    pub(super) bf16_tensor_count: usize,
    pub(super) non_bf16_tensor_count: usize,
    pub(super) role_counts: BTreeMap<String, usize>,
    pub(super) role_bytes: BTreeMap<String, u64>,
    pub(super) required_role_count: usize,
    pub(super) required_roles_present: usize,
    pub(super) missing_required_roles: Vec<String>,
    pub(super) selected_tensor_count_matches_roles: bool,
    pub(super) selected_tensor_bytes_matches_roles: bool,
    pub(super) skipped_routed_expert_tensors: usize,
    pub(super) skipped_routed_expert_bytes: u64,
    pub(super) skipped_quantization_tensors: usize,
    pub(super) skipped_quantization_bytes: u64,
    pub(super) skipped_mtp_tensors: usize,
    pub(super) skipped_mtp_bytes: u64,
    pub(super) sample_resident_keys: Vec<String>,
    pub(super) uses_named_resident_buffers: bool,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub(super) struct FullModelTensorCoverage {
    pub(super) hidden_layers_with_any_tensor: usize,
    pub(super) sparse_layers_with_routed_experts: usize,
    pub(super) dense_layers_with_dense_mlp: usize,
    pub(super) routed_expert_tensors: usize,
    pub(super) routed_quant_metadata_tensors: usize,
    pub(super) attention_tensors: usize,
    pub(super) router_tensors: usize,
    pub(super) shared_expert_tensors: usize,
    pub(super) embedding_tensors: usize,
    pub(super) lm_head_tensors: usize,
}

#[derive(Debug, Serialize)]
pub(super) struct RealFullKvPlan {
    pub(super) layout: &'static str,
    pub(super) dtype: &'static str,
    pub(super) max_tokens: usize,
    pub(super) bytes_per_token: usize,
    pub(super) capacity_bytes: usize,
}

#[derive(Debug, Serialize)]
pub(super) struct RealFullExecutionPlan {
    pub(super) status: &'static str,
    pub(super) scope: &'static str,
    pub(super) hidden_dim: usize,
    pub(super) hidden_dtype: &'static str,
    pub(super) hidden_bytes_per_row: usize,
    pub(super) decode_rows: usize,
    pub(super) mtp_verify_rows: usize,
    pub(super) prefill_chunk_rows: usize,
    pub(super) layer_count: usize,
    pub(super) dense_layer_count: usize,
    pub(super) sparse_layer_count: usize,
    pub(super) stage_counts: RealFullExecutionStageCounts,
    pub(super) protocol_payloads: RealFullProtocolPayloadPlan,
    pub(super) kv_semantics: RealFullKvSemanticsPlan,
    pub(super) scheduler_contract: RealFullSchedulerContract,
    pub(super) terminal_stages: Vec<&'static str>,
    pub(super) layers: Vec<RealFullLayerExecutionPlan>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub(super) struct RealFullExecutionStageCounts {
    pub(super) attention_layers: usize,
    pub(super) compressed_kv_layers: usize,
    pub(super) dense_mlp_layers: usize,
    pub(super) sparse_moe_layers: usize,
    pub(super) remote_expert_exchange_layers: usize,
    pub(super) residual_add_boundaries: usize,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub(super) struct RealFullProtocolPayloadPlan {
    pub(super) protocol: &'static str,
    pub(super) request_header_bytes: usize,
    pub(super) response_header_bytes: usize,
    pub(super) row_descriptor_bytes: usize,
    pub(super) route_entry_bytes: usize,
    pub(super) sparse_decode_roundtrips_per_token: usize,
    pub(super) sparse_prefill_roundtrips_per_chunk: usize,
    pub(super) max_touched_expert_hosts: usize,
    pub(super) routes_per_decode_sparse_layer: usize,
    pub(super) routes_per_prefill_sparse_layer: usize,
    pub(super) routes_per_mtp_sparse_layer: usize,
    pub(super) routes_per_decode_touched_host: usize,
    pub(super) routes_per_prefill_touched_host: usize,
    pub(super) routes_per_mtp_touched_host: usize,
    pub(super) decode_logical_request_bytes_per_touched_host: usize,
    pub(super) decode_logical_response_bytes_per_touched_host: usize,
    pub(super) prefill_logical_request_bytes_per_touched_host: usize,
    pub(super) prefill_logical_response_bytes_per_touched_host: usize,
    pub(super) mtp_logical_request_bytes_per_touched_host: usize,
    pub(super) mtp_logical_response_bytes_per_touched_host: usize,
    pub(super) decode_wire_request_bytes_per_touched_host: usize,
    pub(super) decode_wire_response_bytes_per_touched_host: usize,
    pub(super) prefill_wire_request_bytes_per_touched_host: usize,
    pub(super) prefill_wire_response_bytes_per_touched_host: usize,
    pub(super) mtp_wire_request_bytes_per_touched_host: usize,
    pub(super) mtp_wire_response_bytes_per_touched_host: usize,
    pub(super) decode_full_sparse_roundtrip_logical_bytes: usize,
    pub(super) prefill_full_sparse_roundtrip_logical_bytes: usize,
    pub(super) mtp_full_sparse_roundtrip_logical_bytes: usize,
    pub(super) decode_full_sparse_roundtrip_wire_bytes: usize,
    pub(super) prefill_full_sparse_roundtrip_wire_bytes: usize,
    pub(super) mtp_full_sparse_roundtrip_wire_bytes: usize,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub(super) struct RealFullKvSemanticsPlan {
    pub(super) layout: &'static str,
    pub(super) bytes_per_token: usize,
    pub(super) decode_reads_prefix_when_position_gt_zero: bool,
    pub(super) decode_writes_committed_current_token: bool,
    pub(super) prefill_chunk_zero_reads_prefix: bool,
    pub(super) prefill_later_chunks_read_prefix: bool,
    pub(super) prefill_writes_committed_chunk_range: bool,
    pub(super) mtp_reads_accepted_prefix: bool,
    pub(super) mtp_writes_tentative_draft_range: bool,
    pub(super) mtp_commits_only_accepted_prefix: bool,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub(super) struct RealFullSchedulerContract {
    pub(super) layer_order_is_strict: bool,
    pub(super) modes: [&'static str; 3],
    pub(super) dense_layers_run_on_coordinator: usize,
    pub(super) sparse_layers_require_expert_batches: usize,
    pub(super) expert_batch_can_mix_compatible_sources: bool,
    pub(super) sampling_after_layer_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct RealFullLayerExecutionPlan {
    pub(super) layer_id: usize,
    pub(super) layer_kind: &'static str,
    pub(super) attention: RealFullAttentionStagePlan,
    pub(super) mlp: RealFullMlpStagePlan,
    pub(super) residual_adds: usize,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub(super) struct RealFullAttentionStagePlan {
    pub(super) input_norm: bool,
    pub(super) qkv_projection: bool,
    pub(super) compressed_kv_read_write: bool,
    pub(super) attention_output_projection: bool,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub(super) struct RealFullMlpStagePlan {
    pub(super) post_attention_norm: bool,
    pub(super) dense_mlp_on_coordinator: bool,
    pub(super) router_on_coordinator: bool,
    pub(super) routed_nvfp4_expert_exchange: bool,
    pub(super) shared_expert_on_coordinator: bool,
    pub(super) routes_per_row: usize,
}

#[derive(Debug, Serialize)]
pub(super) struct RealFullKvBackingStoreDryRun {
    pub(super) status: &'static str,
    pub(super) layout: &'static str,
    pub(super) reservation_tokens: usize,
    pub(super) bytes_per_model_token: usize,
    pub(super) capacity_bytes: usize,
    pub(super) dsa_layer_bytes_per_token: usize,
    pub(super) non_dsa_layer_bytes_per_token: usize,
    pub(super) layer_count: usize,
    pub(super) backed_prefill_writes: usize,
    pub(super) backed_decode_writes: usize,
    pub(super) backed_tentative_mtp_writes: usize,
    pub(super) committed_mtp_writes: usize,
    pub(super) discarded_mtp_writes: usize,
    pub(super) backed_write_count_after_discard: usize,
    pub(super) backed_bytes_after_discard: usize,
    pub(super) all_layer_prefill_backed_bytes: usize,
    pub(super) visible_layer0_blocks_at_decode: usize,
    pub(super) visible_layer0_bytes_at_decode: usize,
    pub(super) visible_layer0_blocks_after_mtp_commit: usize,
}

#[derive(Debug, Serialize)]
pub(super) struct RealFullAttentionKvIoDryRun {
    pub(super) status: &'static str,
    pub(super) layer_count: usize,
    pub(super) prefix_prefill_wave_writes: usize,
    pub(super) later_prefill_prefix_read_blocks: usize,
    pub(super) later_prefill_wave_writes: usize,
    pub(super) decode_prefix_read_blocks: usize,
    pub(super) decode_wave_writes: usize,
    pub(super) mtp_prefix_read_blocks: usize,
    pub(super) mtp_tentative_wave_writes: usize,
    pub(super) mtp_committed_writes: usize,
    pub(super) mtp_discarded_writes: usize,
    pub(super) layerwave_payload_count_mismatch_guard: bool,
    pub(super) layer0_decode_read_blocks: usize,
    pub(super) layer0_mtp_read_blocks: usize,
    pub(super) backed_bytes_after_discard: usize,
    pub(super) device_kv_status: &'static str,
    pub(super) device_kv_writes: usize,
    pub(super) device_kv_reads: usize,
    pub(super) device_kv_bytes: usize,
    pub(super) uses_device_kv_cache: bool,
}

#[derive(Debug, Serialize)]
pub(super) struct RealFullAttentionKvBindingDryRun {
    pub(super) status: &'static str,
    pub(super) scope: &'static str,
    pub(super) attention_layers: usize,
    pub(super) attention_tensors: usize,
    pub(super) bf16_attention_tensors: usize,
    pub(super) common_attention_tensors: usize,
    pub(super) indexer_attention_tensors: usize,
    pub(super) attention_tensor_bytes: u64,
    pub(super) common_layers_with_required_tensors: usize,
    pub(super) dsa_indexer_layers: usize,
    pub(super) dsa_indexer_layers_with_required_tensors: usize,
    pub(super) non_dsa_layers: usize,
    pub(super) non_dsa_layers_without_indexer_tensors: usize,
    pub(super) catalog_dsa_indexer_layer_ids: Vec<usize>,
    pub(super) config_dsa_indexer_layer_ids: Vec<usize>,
    pub(super) catalog_dsa_indexer_layers_match_kv_config: bool,
    pub(super) dsa_layer_bytes_per_token: usize,
    pub(super) non_dsa_layer_bytes_per_token: usize,
    pub(super) kv_bytes_per_token: usize,
    pub(super) kv_layer_bytes_sum: usize,
    pub(super) kv_io_layer_count: usize,
    pub(super) kv_io_prefill_writes: usize,
    pub(super) kv_io_decode_writes: usize,
    pub(super) kv_io_tentative_mtp_writes: usize,
    pub(super) kv_io_prefix_read_blocks: usize,
    pub(super) kv_io_backed_bytes_after_discard: usize,
    pub(super) all_attention_layers_bound_to_kv: bool,
}

#[derive(Debug, Serialize)]
pub(super) struct RealFullRequirement {
    pub(super) name: &'static str,
    pub(super) passed: bool,
    pub(super) evidence: String,
    pub(super) blocker: Option<&'static str>,
}
