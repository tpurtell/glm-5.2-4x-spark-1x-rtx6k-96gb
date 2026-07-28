use glmrt_core::{
    GLM52_FIRST_K_DENSE_REPLACE, GLM52_HIDDEN_BF16_BYTES, GLM52_HIDDEN_SIZE,
    GLM52_NUM_HIDDEN_LAYERS, GLM52_TOP_K,
};

use super::super::super::constants::{
    REAL_FULL_PREFLIGHT_DECODE_POSITION, REAL_FULL_PREFLIGHT_DECODE_ROWS,
    REAL_FULL_PREFLIGHT_MTP_ACCEPTED_ROWS, REAL_FULL_PREFLIGHT_MTP_ROWS,
    REAL_FULL_PREFLIGHT_MTP_TOKEN_START, REAL_FULL_PREFLIGHT_PREFILL_ROWS,
    REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START,
};
use super::super::super::coordinator_kernels::coordinator_cuda_reference_kernels_enabled;
use super::super::super::types::{
    RealGlmFullPreflightReport, REAL_FULL_RESIDUAL_COMPLETION_BLOCKER,
};

pub(super) fn assert_execution_scheduler_report(report: &RealGlmFullPreflightReport) {
    assert_eq!(report.status, "blocked");
    assert_eq!(
        report
            .full_model_tensor_coverage
            .hidden_layers_with_any_tensor,
        GLM52_NUM_HIDDEN_LAYERS
    );
    assert_eq!(
        report
            .full_model_tensor_coverage
            .sparse_layers_with_routed_experts,
        GLM52_NUM_HIDDEN_LAYERS - GLM52_FIRST_K_DENSE_REPLACE
    );
    assert!(report
        .requirements
        .iter()
        .any(|req| req.name == "full_tensor_catalog_coverage" && req.passed));
    assert!(report
        .requirements
        .iter()
        .any(|req| req.name == "full_model_execution_plan_available" && req.passed));
    assert!(report.requirements.iter().any(|req| req.name
        == "coordinator_startup_resident_preload_plan_available"
        && req.passed));
    let resident_preload = &report.coordinator_resident_preload;
    assert_eq!(resident_preload.status, "planned");
    assert!(resident_preload.startup_required);
    assert!(resident_preload.uses_named_resident_buffers);
    assert_eq!(resident_preload.loaded_tensor_bytes, 0);
    assert!(resident_preload.selected_tensor_count > 0);
    assert!(resident_preload.selected_tensor_bytes > 0);
    assert_eq!(
        resident_preload.role_counts.get("Embedding").copied(),
        Some(1)
    );
    assert_eq!(resident_preload.role_counts.get("LmHead").copied(), Some(1));
    assert!(
        resident_preload
            .role_counts
            .get("Attention")
            .copied()
            .unwrap_or(0)
            > 0
    );
    assert!(
        resident_preload
            .role_counts
            .get("Norm")
            .copied()
            .unwrap_or(0)
            > 0
    );
    assert!(
        resident_preload
            .role_counts
            .get("DenseMlp")
            .copied()
            .unwrap_or(0)
            > 0
    );
    assert_eq!(resident_preload.role_counts.get("RoutedExpert"), None);
    assert!(resident_preload.skipped_routed_expert_tensors > 0);
    assert!(resident_preload.skipped_quantization_tensors > 0);
    assert!(resident_preload
        .sample_resident_keys
        .iter()
        .any(|name| name == "model.embed_tokens.weight"));
    assert!(report
        .requirements
        .iter()
        .any(|req| req.name == "full_model_scheduler_dry_run_available" && req.passed));
    let admitted_scheduler_requirement = report
        .requirements
        .iter()
        .find(|req| req.name == "full_model_admitted_scheduler_execution_dry_run_available")
        .expect("admitted scheduler execution requirement is reported");
    assert!(
        admitted_scheduler_requirement.passed,
        "admitted scheduler execution requirement failed: {}",
        admitted_scheduler_requirement.evidence
    );
    assert!(admitted_scheduler_requirement
        .evidence
        .contains("full_context_device_attention_complete="));
    assert!(admitted_scheduler_requirement
        .evidence
        .contains("device_attention_hidden_projection_launches="));
    assert!(admitted_scheduler_requirement
        .evidence
        .contains("scheduler_real_tensor_catalog_available=true"));
    assert!(admitted_scheduler_requirement
        .evidence
        .contains("host_routes_match_global=true host_graph_counts_valid=true"));
    assert!(admitted_scheduler_requirement
        .evidence
        .contains("host_wire_envelopes_valid=true"));
    assert!(report
        .requirements
        .iter()
        .any(|req| req.name == "scheduler_numeric_progression_self_test_available" && req.passed));
    assert!(report
        .requirements
        .iter()
        .any(|req| req.name == "full_model_kv_backing_store_dry_run_available" && req.passed));
    assert!(report
        .requirements
        .iter()
        .any(|req| req.name == "full_model_attention_kv_io_dry_run_available" && req.passed));
    assert!(report
        .requirements
        .iter()
        .any(|req| req.name == "real_attention_kv_binding_dry_run_available" && req.passed));
    assert!(report
        .requirements
        .iter()
        .any(|req| req.name == "real_attention_kv_backing_storage" && req.passed));
    assert!(report
        .requirements
        .iter()
        .any(|req| req.name == "full_model_residual_stream_dry_run_available" && req.passed));
    assert!(report
        .requirements
        .iter()
        .any(|req| req.name == "full_residual_numeric_accumulator_kernel_available" && req.passed));
    assert!(report
        .requirements
        .iter()
        .any(|req| req.name == "full_vocab_sampling_dry_run_available" && req.passed));
    let bf16_lm_head_sampler = report
        .requirements
        .iter()
        .find(|req| req.name == "bf16_lm_head_sampler_kernel_available")
        .expect("bf16 lm_head sampler requirement is reported");
    assert!(!bf16_lm_head_sampler.passed);
    assert_eq!(
        bf16_lm_head_sampler.blocker,
        Some("BF16 lm_head sampler wrapper has not scored a real lm_head chunk")
    );
    assert!(report.requirements.iter().any(|req| req.name
        == "all_layer_real_nvfp4_expert_execution_dry_run_available"
        && req.passed));
    assert!(report
        .requirements
        .iter()
        .any(|req| req.name == "full_residual_stream_execution" && !req.passed));
    let full_residual_stream = report
        .requirements
        .iter()
        .find(|req| req.name == "full_residual_stream_execution")
        .expect("full residual stream requirement is reported");
    assert_eq!(
        full_residual_stream.blocker,
        Some(REAL_FULL_RESIDUAL_COMPLETION_BLOCKER)
    );
    assert!(full_residual_stream.evidence.contains("stage_statuses="));
    assert!(full_residual_stream
        .evidence
        .contains("stage_status_counts(real,synthetic,provisional,blocked)="));
    assert!(full_residual_stream
        .evidence
        .contains("stages_with_numeric_checksums="));
    assert!(full_residual_stream
        .evidence
        .contains("numeric_checksum_fields="));
    assert!(full_residual_stream
        .evidence
        .contains("completion_gates(numeric,attention,residual,full_rows,embedding,scheduler,cuda,graph,mla_dsa,expert_daemon,lm_head,full_model,ready,missing)="));
    assert!(full_residual_stream
        .evidence
        .contains("coordinator_backend_stages(total,cuda,cpu,unknown,all_cuda)="));
    assert!(full_residual_stream
        .evidence
        .contains("coordinator_graphs(slots,captured,captures,launches,replayed)="));
    assert!(full_residual_stream
        .evidence
        .contains("bounded_attention_oracle(status="));
    assert!(full_residual_stream
        .evidence
        .contains("projection_backend="));
    assert!(full_residual_stream.evidence.contains("rope_backend="));
    assert!(full_residual_stream
        .evidence
        .contains("layer-ordered terminal lm_head sampling status="));
    assert!(full_residual_stream.evidence.contains("gate_satisfied="));
    assert!(full_residual_stream.evidence.contains("input_token_id="));
    assert!(full_residual_stream
        .evidence
        .contains("embedding_bytes_read="));
    assert!(full_residual_stream
        .evidence
        .contains("missing_gate_names="));
    assert!(full_residual_stream
        .evidence
        .contains("blocker=Some(\"full residual stream requires every completion gate"));
    assert!(report
        .requirements
        .iter()
        .any(|req| req.name == "full_vocab_sampling" && !req.passed));
    assert_eq!(report.kv_plan.bytes_per_token, 95_232);
    assert_eq!(
        report.expert_hosts,
        ["spark-0", "spark-1", "spark-2", "spark-3"]
    );
    assert_eq!(report.execution_plan.layer_count, GLM52_NUM_HIDDEN_LAYERS);
    assert_eq!(
        report.execution_plan.dense_layer_count,
        GLM52_FIRST_K_DENSE_REPLACE
    );
    assert_eq!(
        report.execution_plan.sparse_layer_count,
        GLM52_NUM_HIDDEN_LAYERS - GLM52_FIRST_K_DENSE_REPLACE
    );
    assert_eq!(
        report
            .execution_plan
            .protocol_payloads
            .max_touched_expert_hosts,
        4
    );
    assert_eq!(
        report
            .execution_plan
            .protocol_payloads
            .decode_logical_request_bytes_per_touched_host,
        12_288
    );
    assert_eq!(
        report
            .execution_plan
            .protocol_payloads
            .decode_wire_request_bytes_per_touched_host,
        12_444
    );
    assert_eq!(
        report
            .execution_plan
            .protocol_payloads
            .decode_wire_response_bytes_per_touched_host,
        12_384
    );
    assert_eq!(
        report
            .execution_plan
            .protocol_payloads
            .prefill_logical_request_bytes_per_touched_host,
        6_291_456
    );
    assert_eq!(
        report
            .execution_plan
            .protocol_payloads
            .prefill_wire_request_bytes_per_touched_host,
        6_322_272
    );
    assert_eq!(
        report
            .execution_plan
            .protocol_payloads
            .prefill_wire_response_bytes_per_touched_host,
        6_291_552
    );
    assert_eq!(
        report
            .execution_plan
            .protocol_payloads
            .mtp_full_sparse_roundtrip_wire_bytes,
        59_184_000
    );
    assert_eq!(report.execution_plan.layers[0].layer_kind, "dense-mlp");
    assert!(
        report.execution_plan.layers[3]
            .mlp
            .routed_nvfp4_expert_exchange
    );
    assert!(
        !report
            .execution_plan
            .kv_semantics
            .prefill_chunk_zero_reads_prefix
    );
    assert!(
        report
            .execution_plan
            .kv_semantics
            .prefill_later_chunks_read_prefix
    );
    assert_eq!(
        report.scheduler_dry_run.total_layerwaves,
        GLM52_NUM_HIDDEN_LAYERS * 3
    );
    assert_eq!(
        report.scheduler_dry_run.sparse_expert_batches,
        GLM52_NUM_HIDDEN_LAYERS - GLM52_FIRST_K_DENSE_REPLACE
    );
    assert_eq!(report.scheduler_dry_run.rows_per_sparse_expert_batch, 521);
    assert_eq!(
        report.scheduler_dry_run.routes_per_sparse_expert_batch,
        4_168
    );
    assert_eq!(
        report.scheduler_dry_run.prefill_prefix_read_layers,
        GLM52_NUM_HIDDEN_LAYERS
    );
    assert_eq!(
        report.scheduler_dry_run.mtp_tentative_write_records,
        GLM52_NUM_HIDDEN_LAYERS * REAL_FULL_PREFLIGHT_MTP_ROWS
    );
    assert!(report.scheduler_dry_run.layer_dry_runs[0]
        .expert_batch
        .is_none());
    let first_sparse_batch = report.scheduler_dry_run.layer_dry_runs[3]
        .expert_batch
        .as_ref()
        .expect("sparse layer has mixed expert batch");
    assert_eq!(
        first_sparse_batch.source_modes,
        ["prefill_chunk", "mtp_verify", "decode_step"]
    );
    assert_eq!(first_sparse_batch.rows, 521);
    assert_eq!(
        report.scheduler_dry_run.protocol_v2_batch_probe.status,
        "encoded-and-reconstructed-mixed-expert-batch-protocol-v2"
    );
    assert_eq!(report.scheduler_dry_run.protocol_v2_batch_probe.layer_id, 3);
    assert_eq!(report.scheduler_dry_run.protocol_v2_batch_probe.rows, 521);
    assert_eq!(
        report.scheduler_dry_run.protocol_v2_batch_probe.routes,
        4_168
    );
    assert_eq!(
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .hidden_payload_bytes,
        6_402_048
    );
    assert_eq!(
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .request_wire_bytes,
        6_464_664
    );
    assert_eq!(
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .response_wire_bytes,
        6_402_144
    );
    assert!(
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .request_frame_buffer_stable
    );
    assert!(
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .response_frame_buffer_stable
    );
    assert!(
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .decoded_request_matches
    );
    assert!(
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .decoded_response_matches
    );
    assert_eq!(
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .reconstructed_response_rows,
        521
    );
    assert_eq!(
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .reconstructed_response_payload_bytes,
        6_402_048
    );
    assert!(
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .reconstructed_response_row_order_matches
    );
    assert!(
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .reconstructed_response_payload_matches
    );
    assert_eq!(
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .host_batches,
        4
    );
    assert_eq!(
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .host_batch_rows,
        2_084
    );
    assert_eq!(
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .host_batch_routes,
        4_168
    );
    assert_eq!(
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .host_batch_expert_tiles,
        256
    );
    assert!(
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .host_batch_routes_match_global
    );
    assert!(
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .host_batch_graph_counts_valid
    );
    assert_eq!(
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .host_request_frames,
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .host_batches
    );
    assert_eq!(
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .host_request_rows,
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .host_batch_rows
    );
    assert_eq!(
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .host_request_routes,
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .host_batch_routes
    );
    assert_eq!(
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .host_request_payload_bytes,
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .host_batch_rows
            * GLM52_HIDDEN_BF16_BYTES
    );
    assert!(
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .host_request_wire_bytes
            > report
                .scheduler_dry_run
                .protocol_v2_batch_probe
                .host_request_payload_bytes
    );
    assert_eq!(
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .host_response_frames,
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .host_batches
    );
    assert_eq!(
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .host_response_rows,
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .host_batch_rows
    );
    assert_eq!(
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .host_response_payload_bytes,
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .host_batch_rows
            * GLM52_HIDDEN_BF16_BYTES
    );
    assert!(
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .host_response_wire_bytes
            > report
                .scheduler_dry_run
                .protocol_v2_batch_probe
                .host_response_payload_bytes
    );
    assert!(
        report
            .scheduler_dry_run
            .protocol_v2_batch_probe
            .host_wire_envelopes_valid
    );
    assert!(report.scheduler_dry_run.protocol_v2_batch_probe.passed);
    assert_eq!(
        report.scheduler_execution_dry_run.iterations,
        GLM52_NUM_HIDDEN_LAYERS * 2
    );
    assert_eq!(
        report.scheduler_execution_dry_run.candidate_layerwaves,
        GLM52_NUM_HIDDEN_LAYERS * 4
    );
    assert_eq!(
        report.scheduler_execution_dry_run.selected_layerwaves,
        GLM52_NUM_HIDDEN_LAYERS * 4
    );
    assert_eq!(report.scheduler_execution_dry_run.deferred_layerwaves, 0);
    assert_eq!(
        report.scheduler_execution_dry_run.sparse_expert_batches,
        (GLM52_NUM_HIDDEN_LAYERS - GLM52_FIRST_K_DENSE_REPLACE) * 2
    );
    assert_eq!(
        report
            .scheduler_execution_dry_run
            .sparse_expert_host_batch_sets,
        report.scheduler_execution_dry_run.sparse_expert_batches
    );
    assert_eq!(
        report
            .scheduler_execution_dry_run
            .sparse_expert_host_batches,
        report.scheduler_execution_dry_run.sparse_expert_batches * glmrt_core::EXPERT_HOSTS.len()
    );
    assert_eq!(
        report
            .scheduler_execution_dry_run
            .sparse_expert_host_batch_rows,
        report.scheduler_execution_dry_run.sparse_expert_batch_rows
            * glmrt_core::EXPERT_HOSTS.len()
    );
    assert_eq!(
        report
            .scheduler_execution_dry_run
            .sparse_expert_host_batch_routes,
        report
            .scheduler_execution_dry_run
            .sparse_expert_batch_routes
    );
    assert!(
        report
            .scheduler_execution_dry_run
            .sparse_expert_host_batch_expert_tiles
            > 0
    );
    assert!(
        report
            .scheduler_execution_dry_run
            .sparse_expert_host_batch_expert_tiles
            <= report
                .scheduler_execution_dry_run
                .sparse_expert_host_batch_routes
    );
    assert!(
        report
            .scheduler_execution_dry_run
            .sparse_expert_host_batch_routes_match_global
    );
    assert!(
        report
            .scheduler_execution_dry_run
            .sparse_expert_host_batch_graph_counts_valid
    );
    assert_eq!(
        report
            .scheduler_execution_dry_run
            .sparse_expert_host_request_frames,
        report
            .scheduler_execution_dry_run
            .sparse_expert_host_batches
    );
    assert_eq!(
        report
            .scheduler_execution_dry_run
            .sparse_expert_host_request_rows,
        report
            .scheduler_execution_dry_run
            .sparse_expert_host_batch_rows
    );
    assert_eq!(
        report
            .scheduler_execution_dry_run
            .sparse_expert_host_request_routes,
        report
            .scheduler_execution_dry_run
            .sparse_expert_host_batch_routes
    );
    assert_eq!(
        report
            .scheduler_execution_dry_run
            .sparse_expert_host_request_payload_bytes,
        report
            .scheduler_execution_dry_run
            .sparse_expert_host_batch_rows
            * GLM52_HIDDEN_BF16_BYTES
    );
    assert!(
        report
            .scheduler_execution_dry_run
            .sparse_expert_host_request_wire_bytes
            > report
                .scheduler_execution_dry_run
                .sparse_expert_host_request_payload_bytes
    );
    assert_eq!(
        report
            .scheduler_execution_dry_run
            .sparse_expert_host_response_frames,
        report
            .scheduler_execution_dry_run
            .sparse_expert_host_batches
    );
    assert_eq!(
        report
            .scheduler_execution_dry_run
            .sparse_expert_host_response_rows,
        report
            .scheduler_execution_dry_run
            .sparse_expert_host_batch_rows
    );
    assert_eq!(
        report
            .scheduler_execution_dry_run
            .sparse_expert_host_response_payload_bytes,
        report
            .scheduler_execution_dry_run
            .sparse_expert_host_batch_rows
            * GLM52_HIDDEN_BF16_BYTES
    );
    assert!(
        report
            .scheduler_execution_dry_run
            .sparse_expert_host_response_wire_bytes
            > report
                .scheduler_execution_dry_run
                .sparse_expert_host_response_payload_bytes
    );
    assert!(
        report
            .scheduler_execution_dry_run
            .sparse_expert_host_wire_envelopes_valid
    );
    assert_eq!(
        report.scheduler_execution_dry_run.kv_read_blocks,
        GLM52_NUM_HIDDEN_LAYERS * 6
    );
    assert_eq!(
        report.scheduler_execution_dry_run.committed_kv_writes,
        GLM52_NUM_HIDDEN_LAYERS * 3
    );
    assert_eq!(
        report.scheduler_execution_dry_run.tentative_kv_writes,
        GLM52_NUM_HIDDEN_LAYERS * REAL_FULL_PREFLIGHT_MTP_ROWS
    );
    assert_eq!(
        report
            .scheduler_execution_dry_run
            .backed_bytes_after_discard,
        97_993_728
    );
    let expected_scheduler_device_kv_writes =
        GLM52_NUM_HIDDEN_LAYERS * (3 + REAL_FULL_PREFLIGHT_MTP_ROWS);
    let expected_scheduler_device_kv_reads = GLM52_NUM_HIDDEN_LAYERS * 23;
    let expected_scheduler_device_attention_launches = GLM52_NUM_HIDDEN_LAYERS * 4;
    let expected_scheduler_device_attention_rows_per_layer = REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START
        as usize
        + (REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START as usize + REAL_FULL_PREFLIGHT_PREFILL_ROWS)
        + (REAL_FULL_PREFLIGHT_DECODE_POSITION as usize + REAL_FULL_PREFLIGHT_DECODE_ROWS)
        + (REAL_FULL_PREFLIGHT_MTP_TOKEN_START as usize + REAL_FULL_PREFLIGHT_MTP_ROWS);
    let expected_scheduler_device_attention_rows =
        GLM52_NUM_HIDDEN_LAYERS * expected_scheduler_device_attention_rows_per_layer;
    let expected_scheduler_device_attention_query_rows = GLM52_NUM_HIDDEN_LAYERS
        * (REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START as usize
            + REAL_FULL_PREFLIGHT_PREFILL_ROWS
            + REAL_FULL_PREFLIGHT_DECODE_ROWS
            + REAL_FULL_PREFLIGHT_MTP_ROWS);
    let expected_scheduler_device_attention_kv_descriptors = GLM52_NUM_HIDDEN_LAYERS * 17;
    let expected_scheduler_device_attention_output_values_per_row =
        if coordinator_cuda_reference_kernels_enabled() {
            GLM52_HIDDEN_SIZE
        } else {
            2 * 2
        };
    let expected_scheduler_device_attention_output_status =
        if coordinator_cuda_reference_kernels_enabled() {
            "cuda-kv-cache-mla-rope-attention-hidden-projection-device-buffer"
        } else {
            "cuda-kv-cache-mla-rope-attention-device-buffer"
        };
    let expected_scheduler_device_attention_output_rows =
        if coordinator_cuda_reference_kernels_enabled() {
            expected_scheduler_device_attention_query_rows
        } else {
            expected_scheduler_device_attention_rows
        };
    let expected_scheduler_device_attention_output_bytes =
        expected_scheduler_device_attention_output_rows
            * expected_scheduler_device_attention_output_values_per_row
            * std::mem::size_of::<u16>();
    let expected_scheduler_device_attention_output_values =
        expected_scheduler_device_attention_output_bytes / std::mem::size_of::<u16>();
    let (
        expected_scheduler_device_attention_resident_uploads,
        expected_scheduler_device_attention_resident_query_shapes,
    ) = if coordinator_cuda_reference_kernels_enabled() {
        (4, 0)
    } else {
        (5, 3)
    };
    let expected_scheduler_device_attention_resident_buffer_uses =
        expected_scheduler_device_attention_launches
            * if coordinator_cuda_reference_kernels_enabled() {
                4
            } else {
                3
            };
    assert_scheduler_device_kv_status(
        report.scheduler_execution_dry_run.device_kv_status,
        report.scheduler_execution_dry_run.uses_device_kv_cache,
    );
    assert_eq!(
        report
            .scheduler_execution_dry_run
            .projected_device_kv_writes
            + report
                .scheduler_execution_dry_run
                .synthetic_kv_payload_writes,
        expected_scheduler_device_kv_writes
    );
    if report
        .scheduler_execution_dry_run
        .projected_device_kv_writes
        == 0
    {
        assert_eq!(
            report
                .scheduler_execution_dry_run
                .projected_device_kv_write_bytes,
            0
        );
    } else {
        assert!(
            report
                .scheduler_execution_dry_run
                .projected_device_kv_write_bytes
                > 0
        );
    }
    if report.scheduler_execution_dry_run.uses_device_kv_cache {
        assert_eq!(
            report.scheduler_execution_dry_run.device_kv_status,
            "cuda-kv-cache-live-scheduler"
        );
        assert_eq!(
            report.scheduler_execution_dry_run.device_kv_writes,
            expected_scheduler_device_kv_writes
        );
        assert_eq!(
            report.scheduler_execution_dry_run.device_kv_reads,
            expected_scheduler_device_kv_reads
        );
        assert_eq!(
            report.scheduler_execution_dry_run.device_kv_bytes,
            684_527_616
        );
        assert_eq!(
            report.scheduler_execution_dry_run.device_attention_status,
            expected_scheduler_device_attention_output_status
        );
        assert!(report.scheduler_execution_dry_run.uses_device_kv_attention);
        assert_eq!(
            report
                .scheduler_execution_dry_run
                .device_attention_resident_uploads,
            expected_scheduler_device_attention_resident_uploads
        );
        assert_eq!(
            report
                .scheduler_execution_dry_run
                .device_attention_resident_query_shapes,
            expected_scheduler_device_attention_resident_query_shapes
        );
        assert_eq!(
            report
                .scheduler_execution_dry_run
                .device_attention_resident_buffer_uses,
            expected_scheduler_device_attention_resident_buffer_uses
        );
        assert_eq!(
            report.scheduler_execution_dry_run.device_attention_launches,
            expected_scheduler_device_attention_launches
        );
        assert_eq!(
            report
                .scheduler_execution_dry_run
                .device_attention_hidden_projection_launches,
            if coordinator_cuda_reference_kernels_enabled() {
                expected_scheduler_device_attention_launches
            } else {
                0
            }
        );
        assert_eq!(
            report.scheduler_execution_dry_run.device_attention_rows,
            expected_scheduler_device_attention_rows
        );
        assert_eq!(
            report
                .scheduler_execution_dry_run
                .device_attention_query_rows,
            expected_scheduler_device_attention_query_rows
        );
        assert_eq!(
            report
                .scheduler_execution_dry_run
                .device_attention_kv_descriptors,
            expected_scheduler_device_attention_kv_descriptors
        );
        assert_eq!(
            report
                .scheduler_execution_dry_run
                .device_attention_output_bytes,
            expected_scheduler_device_attention_output_bytes
        );
        assert_eq!(
            report
                .scheduler_execution_dry_run
                .device_attention_output_values,
            expected_scheduler_device_attention_output_values
        );
        assert_eq!(
            report
                .scheduler_execution_dry_run
                .device_attention_output_finite_values,
            expected_scheduler_device_attention_output_values
        );
        assert!(
            report
                .scheduler_execution_dry_run
                .device_attention_output_nonzero_values
                > 0
        );
        assert!(report
            .scheduler_execution_dry_run
            .device_attention_output_checksum
            .is_finite());
        if coordinator_cuda_reference_kernels_enabled() {
            assert!(
                report
                    .scheduler_execution_dry_run
                    .full_context_device_attention_complete
            );
        } else {
            assert!(
                !report
                    .scheduler_execution_dry_run
                    .full_context_device_attention_complete
            );
        }
    } else {
        assert_eq!(report.scheduler_execution_dry_run.device_kv_writes, 0);
        assert_eq!(report.scheduler_execution_dry_run.device_kv_reads, 0);
        assert_eq!(report.scheduler_execution_dry_run.device_kv_bytes, 0);
        assert_eq!(
            report.scheduler_execution_dry_run.device_attention_status,
            "not-run"
        );
        assert_eq!(
            report.scheduler_execution_dry_run.device_attention_launches,
            0
        );
        assert_eq!(
            report
                .scheduler_execution_dry_run
                .device_attention_hidden_projection_launches,
            0
        );
        assert_eq!(
            report
                .scheduler_execution_dry_run
                .device_attention_resident_uploads,
            0
        );
        assert_eq!(
            report
                .scheduler_execution_dry_run
                .device_attention_resident_query_shapes,
            0
        );
        assert_eq!(
            report
                .scheduler_execution_dry_run
                .device_attention_resident_buffer_uses,
            0
        );
        assert_eq!(report.scheduler_execution_dry_run.device_attention_rows, 0);
        assert_eq!(
            report
                .scheduler_execution_dry_run
                .device_attention_query_rows,
            0
        );
        assert_eq!(
            report
                .scheduler_execution_dry_run
                .device_attention_kv_descriptors,
            0
        );
        assert_eq!(
            report
                .scheduler_execution_dry_run
                .device_attention_output_bytes,
            0
        );
        assert_eq!(
            report
                .scheduler_execution_dry_run
                .device_attention_output_values,
            0
        );
        assert_eq!(
            report
                .scheduler_execution_dry_run
                .device_attention_output_finite_values,
            0
        );
        assert_eq!(
            report
                .scheduler_execution_dry_run
                .device_attention_output_nonzero_values,
            0
        );
        assert_eq!(
            report
                .scheduler_execution_dry_run
                .device_attention_output_checksum,
            0.0
        );
        assert!(!report.scheduler_execution_dry_run.uses_device_kv_attention);
        assert!(
            !report
                .scheduler_execution_dry_run
                .full_context_device_attention_complete
        );
    }
    let scheduler_progression = &report
        .scheduler_execution_dry_run
        .numeric_progression_self_test;
    assert!(scheduler_progression.passed);
    assert_eq!(scheduler_progression.layers, GLM52_NUM_HIDDEN_LAYERS);
    assert_eq!(
        scheduler_progression.source_modes,
        ["prefill", "decode", "mtp_verify"]
    );
    assert_eq!(
        scheduler_progression.unique_source_rows,
        REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START as usize
            + REAL_FULL_PREFLIGHT_PREFILL_ROWS
            + REAL_FULL_PREFLIGHT_DECODE_ROWS
            + REAL_FULL_PREFLIGHT_MTP_ROWS
    );
    assert_eq!(scheduler_progression.hidden_dim, GLM52_HIDDEN_SIZE);
    assert_eq!(scheduler_progression.residual_dtype, "bf16");
    assert_eq!(
        scheduler_progression.selected_prefill_rows,
        (REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START as usize + REAL_FULL_PREFLIGHT_PREFILL_ROWS)
            * GLM52_NUM_HIDDEN_LAYERS
    );
    assert_eq!(
        scheduler_progression.selected_decode_rows,
        REAL_FULL_PREFLIGHT_DECODE_ROWS * GLM52_NUM_HIDDEN_LAYERS
    );
    assert_eq!(
        scheduler_progression.selected_mtp_rows,
        REAL_FULL_PREFLIGHT_MTP_ROWS * GLM52_NUM_HIDDEN_LAYERS
    );
    assert_eq!(
        scheduler_progression.mtp_accepted_rows_per_layer,
        REAL_FULL_PREFLIGHT_MTP_ACCEPTED_ROWS
    );
    assert_eq!(
        scheduler_progression.mtp_rejected_rows_per_layer,
        REAL_FULL_PREFLIGHT_MTP_ROWS - REAL_FULL_PREFLIGHT_MTP_ACCEPTED_ROWS
    );
    let expected_source_segments = GLM52_NUM_HIDDEN_LAYERS * 4;
    assert_eq!(
        scheduler_progression.source_segments,
        expected_source_segments
    );
    assert_eq!(
        scheduler_progression.attention_residual_adds,
        expected_source_segments
    );
    assert_eq!(
        scheduler_progression.mlp_residual_adds,
        expected_source_segments
    );
    assert!(scheduler_progression
        .attention_residual_add_backend
        .contains("residual-add-bf16"));
    assert!(scheduler_progression
        .mlp_residual_add_backend
        .contains("residual-add-bf16"));
    let expected_value_updates =
        scheduler_progression.unique_source_rows * GLM52_NUM_HIDDEN_LAYERS * GLM52_HIDDEN_SIZE;
    let expected_real_dense_mlp_rows =
        GLM52_FIRST_K_DENSE_REPLACE * scheduler_progression.unique_source_rows;
    let expected_real_dense_mlp_values = expected_real_dense_mlp_rows * GLM52_HIDDEN_SIZE;
    let expected_real_dense_mlp_source_segments = GLM52_FIRST_K_DENSE_REPLACE * 4;
    let sparse_layers = GLM52_NUM_HIDDEN_LAYERS - GLM52_FIRST_K_DENSE_REPLACE;
    let expected_real_sparse_shared_mlp_rows =
        sparse_layers * scheduler_progression.unique_source_rows;
    let expected_real_sparse_shared_mlp_values =
        expected_real_sparse_shared_mlp_rows * GLM52_HIDDEN_SIZE;
    let expected_real_sparse_shared_mlp_source_segments = sparse_layers * 4;
    let expected_real_sparse_routed_mlp_rows = expected_real_sparse_shared_mlp_rows;
    let expected_real_sparse_routed_mlp_values = expected_real_sparse_shared_mlp_values;
    let expected_real_sparse_routed_mlp_source_segments =
        expected_real_sparse_shared_mlp_source_segments;
    let expected_real_sparse_routed_mlp_routes = expected_real_sparse_routed_mlp_rows * GLM52_TOP_K;
    let expected_resident_values = scheduler_progression.unique_source_rows * GLM52_HIDDEN_SIZE;
    assert_eq!(
        scheduler_progression.attention_value_updates,
        expected_value_updates
    );
    assert_eq!(
        scheduler_progression.mlp_value_updates,
        expected_value_updates
    );
    if report.scheduler_execution_dry_run.uses_device_kv_attention {
        assert!(scheduler_progression.uses_device_attention_output_delta);
        assert!(scheduler_progression
            .attention_device_output_delta_checksum
            .is_finite());
        assert_eq!(
            scheduler_progression.attention_device_output_delta_backend,
            report.scheduler_execution_dry_run.device_attention_status
        );
        if coordinator_cuda_reference_kernels_enabled() {
            assert_eq!(
                scheduler_progression.device_attention_output_delta_status,
                "cuda-device-attention-hidden-delta"
            );
            assert_eq!(
                scheduler_progression.attention_device_output_delta_rows,
                report
                    .scheduler_execution_dry_run
                    .device_attention_query_rows
            );
            assert_eq!(
                scheduler_progression.attention_device_output_delta_values,
                report
                    .scheduler_execution_dry_run
                    .device_attention_query_rows
                    * GLM52_HIDDEN_SIZE
            );
            assert_eq!(
                scheduler_progression.attention_device_output_delta_values,
                report
                    .scheduler_execution_dry_run
                    .device_attention_output_values
            );
            assert_eq!(
                scheduler_progression.attention_device_output_delta_device_prefix_rows,
                0
            );
            assert_eq!(
                scheduler_progression.attention_device_output_delta_device_prefix_values,
                0
            );
            assert_eq!(
                scheduler_progression.attention_device_output_delta_device_prefix_backend,
                "not-run"
            );
        } else {
            assert_eq!(
                scheduler_progression.device_attention_output_delta_status,
                "cuda-device-attention-output-prefix-delta"
            );
            assert_eq!(
                scheduler_progression.attention_device_output_delta_rows,
                report
                    .scheduler_execution_dry_run
                    .device_attention_query_rows
            );
            assert_eq!(
                scheduler_progression.attention_device_output_delta_values,
                scheduler_progression.attention_device_output_delta_rows
                    * expected_scheduler_device_attention_output_values_per_row
            );
            assert!(
                scheduler_progression.attention_device_output_delta_values
                    <= report
                        .scheduler_execution_dry_run
                        .device_attention_output_values
            );
            assert_eq!(
                scheduler_progression.attention_device_output_delta_device_prefix_rows,
                0
            );
            assert_eq!(
                scheduler_progression.attention_device_output_delta_device_prefix_values,
                0
            );
            assert_eq!(
                scheduler_progression.attention_device_output_delta_device_prefix_backend,
                "not-run"
            );
        }
    } else {
        assert_eq!(
            scheduler_progression.device_attention_output_delta_status,
            "not-run"
        );
        assert!(!scheduler_progression.uses_device_attention_output_delta);
        assert_eq!(scheduler_progression.attention_device_output_delta_rows, 0);
        assert_eq!(
            scheduler_progression.attention_device_output_delta_values,
            0
        );
        assert_eq!(
            scheduler_progression.attention_device_output_delta_checksum,
            0.0
        );
        assert_eq!(
            scheduler_progression.attention_device_output_delta_backend,
            "not-run"
        );
        assert_eq!(
            scheduler_progression.attention_device_output_delta_device_prefix_rows,
            0
        );
        assert_eq!(
            scheduler_progression.attention_device_output_delta_device_prefix_values,
            0
        );
        assert_eq!(
            scheduler_progression.attention_device_output_delta_device_prefix_backend,
            "not-run"
        );
    }
    if coordinator_cuda_reference_kernels_enabled() {
        assert_eq!(
            scheduler_progression.device_delta_template_status,
            "cuda-device-delta-template-not-needed"
        );
        assert_eq!(scheduler_progression.device_delta_template_uploads, 0);
        assert_eq!(scheduler_progression.device_delta_template_uses, 0);
        assert_eq!(
            scheduler_progression.device_delta_template_resident_values,
            0
        );
    } else {
        assert_eq!(
            scheduler_progression.device_delta_template_status,
            "not-run"
        );
        assert_eq!(scheduler_progression.device_delta_template_uploads, 0);
        assert_eq!(scheduler_progression.device_delta_template_uses, 0);
        assert_eq!(
            scheduler_progression.device_delta_template_resident_values,
            0
        );
    }
    if coordinator_cuda_reference_kernels_enabled() {
        assert_eq!(
            scheduler_progression.device_mlp_delta_status,
            "cuda-device-hidden-dependent-mlp-delta-not-needed"
        );
        assert!(!scheduler_progression.uses_device_mlp_delta);
        assert_eq!(scheduler_progression.device_mlp_delta_rows, 0);
        assert_eq!(scheduler_progression.device_mlp_delta_values, 0);
        assert_eq!(scheduler_progression.device_mlp_delta_checksum, 0.0);
        assert_eq!(scheduler_progression.device_mlp_delta_backend, "not-run");
        assert_eq!(scheduler_progression.device_mlp_weight_uploads, 0);
        assert_eq!(scheduler_progression.device_mlp_weight_resident_values, 0);
    } else {
        assert_eq!(scheduler_progression.device_mlp_delta_status, "not-run");
        assert!(!scheduler_progression.uses_device_mlp_delta);
        assert_eq!(scheduler_progression.device_mlp_delta_rows, 0);
        assert_eq!(scheduler_progression.device_mlp_delta_values, 0);
        assert_eq!(scheduler_progression.device_mlp_delta_checksum, 0.0);
        assert_eq!(scheduler_progression.device_mlp_delta_backend, "not-run");
        assert_eq!(scheduler_progression.device_mlp_weight_uploads, 0);
        assert_eq!(scheduler_progression.device_mlp_weight_resident_values, 0);
    }
    if coordinator_cuda_reference_kernels_enabled() {
        assert_eq!(
            scheduler_progression.device_real_dense_mlp_delta_status,
            "cuda-real-dense-checkpoint-mlp-delta"
        );
        assert!(scheduler_progression.uses_device_real_dense_mlp_delta);
        assert_eq!(
            scheduler_progression.device_real_dense_mlp_delta_rows,
            expected_real_dense_mlp_rows
        );
        assert_eq!(
            scheduler_progression.device_real_dense_mlp_delta_values,
            expected_real_dense_mlp_values
        );
        assert!(scheduler_progression
            .device_real_dense_mlp_delta_checksum
            .is_finite());
        assert_eq!(
            scheduler_progression.device_real_dense_mlp_delta_backend,
            "cuda-reference-silu-gated-mlp-bf16-preloaded-gate-up-down-resident-weight"
        );
        assert!(scheduler_progression
            .device_real_dense_mlp_norm_backend
            .contains("rmsnorm-bf16"));
        assert_eq!(
            scheduler_progression.device_real_dense_mlp_weight_tensors,
            GLM52_FIRST_K_DENSE_REPLACE * 4
        );
        assert!(scheduler_progression.device_real_dense_mlp_weight_bytes > 0);
        assert_eq!(
            scheduler_progression.device_real_dense_mlp_layers,
            GLM52_FIRST_K_DENSE_REPLACE
        );
        assert_eq!(
            scheduler_progression.device_real_dense_mlp_source_segments,
            expected_real_dense_mlp_source_segments
        );
    } else {
        assert_eq!(
            scheduler_progression.device_real_dense_mlp_delta_status,
            "not-run"
        );
        assert!(!scheduler_progression.uses_device_real_dense_mlp_delta);
        assert_eq!(scheduler_progression.device_real_dense_mlp_delta_rows, 0);
        assert_eq!(scheduler_progression.device_real_dense_mlp_delta_values, 0);
        assert_eq!(
            scheduler_progression.device_real_dense_mlp_delta_checksum,
            0.0
        );
        assert_eq!(
            scheduler_progression.device_real_dense_mlp_delta_backend,
            "not-run"
        );
        assert_eq!(
            scheduler_progression.device_real_dense_mlp_norm_backend,
            "not-run"
        );
        assert_eq!(
            scheduler_progression.device_real_dense_mlp_weight_tensors,
            0
        );
        assert_eq!(scheduler_progression.device_real_dense_mlp_weight_bytes, 0);
        assert_eq!(scheduler_progression.device_real_dense_mlp_layers, 0);
        assert_eq!(
            scheduler_progression.device_real_dense_mlp_source_segments,
            0
        );
    }
    if coordinator_cuda_reference_kernels_enabled() {
        assert_eq!(
            scheduler_progression.device_real_sparse_shared_mlp_delta_status,
            "cuda-real-sparse-shared-checkpoint-mlp-delta"
        );
        assert!(scheduler_progression.uses_device_real_sparse_shared_mlp_delta);
        assert_eq!(
            scheduler_progression.device_real_sparse_shared_mlp_delta_rows,
            expected_real_sparse_shared_mlp_rows
        );
        assert_eq!(
            scheduler_progression.device_real_sparse_shared_mlp_delta_values,
            expected_real_sparse_shared_mlp_values
        );
        assert!(scheduler_progression
            .device_real_sparse_shared_mlp_delta_checksum
            .is_finite());
        assert_eq!(
            scheduler_progression.device_real_sparse_shared_mlp_delta_backend,
            "cuda-reference-silu-gated-mlp-bf16-preloaded-gate-up-down-resident-weight"
        );
        assert!(scheduler_progression
            .device_real_sparse_shared_mlp_norm_backend
            .contains("rmsnorm-bf16"));
        assert_eq!(
            scheduler_progression.device_real_sparse_shared_mlp_weight_tensors,
            sparse_layers * 4
        );
        assert!(scheduler_progression.device_real_sparse_shared_mlp_weight_bytes > 0);
        assert_eq!(
            scheduler_progression.device_real_sparse_shared_mlp_layers,
            sparse_layers
        );
        assert_eq!(
            scheduler_progression.device_real_sparse_shared_mlp_source_segments,
            expected_real_sparse_shared_mlp_source_segments
        );
    } else {
        assert_eq!(
            scheduler_progression.device_real_sparse_shared_mlp_delta_status,
            "not-run"
        );
        assert!(!scheduler_progression.uses_device_real_sparse_shared_mlp_delta);
        assert_eq!(
            scheduler_progression.device_real_sparse_shared_mlp_delta_rows,
            0
        );
        assert_eq!(
            scheduler_progression.device_real_sparse_shared_mlp_delta_values,
            0
        );
        assert_eq!(
            scheduler_progression.device_real_sparse_shared_mlp_delta_checksum,
            0.0
        );
        assert_eq!(
            scheduler_progression.device_real_sparse_shared_mlp_delta_backend,
            "not-run"
        );
        assert_eq!(
            scheduler_progression.device_real_sparse_shared_mlp_norm_backend,
            "not-run"
        );
        assert_eq!(
            scheduler_progression.device_real_sparse_shared_mlp_weight_tensors,
            0
        );
        assert_eq!(
            scheduler_progression.device_real_sparse_shared_mlp_weight_bytes,
            0
        );
        assert_eq!(
            scheduler_progression.device_real_sparse_shared_mlp_layers,
            0
        );
        assert_eq!(
            scheduler_progression.device_real_sparse_shared_mlp_source_segments,
            0
        );
    }
    if coordinator_cuda_reference_kernels_enabled() {
        assert_eq!(
            scheduler_progression.device_real_sparse_routed_mlp_delta_status,
            "cuda-real-sparse-routed-nvfp4-checkpoint-mlp-delta"
        );
        assert!(scheduler_progression.uses_device_real_sparse_routed_mlp_delta);
        assert_eq!(
            scheduler_progression.device_real_sparse_routed_mlp_delta_rows,
            expected_real_sparse_routed_mlp_rows
        );
        assert_eq!(
            scheduler_progression.device_real_sparse_routed_mlp_delta_values,
            expected_real_sparse_routed_mlp_values
        );
        assert!(scheduler_progression
            .device_real_sparse_routed_mlp_delta_checksum
            .is_finite());
        assert!(scheduler_progression
            .device_real_sparse_routed_mlp_delta_backend
            .contains("nvfp4-route-bf16-accumulated-device-output"));
        assert!(scheduler_progression
            .device_real_sparse_routed_mlp_route_backend
            .contains("nvfp4-route-bf16-accumulated-device-input"));
        assert!(scheduler_progression
            .device_real_sparse_routed_mlp_router_backend
            .contains("router-topk-bf16"));
        assert!(scheduler_progression
            .device_real_sparse_routed_mlp_router_backend
            .contains("device-input"));
        assert_eq!(
            scheduler_progression.device_real_sparse_routed_mlp_routes,
            expected_real_sparse_routed_mlp_routes
        );
        assert!(scheduler_progression.device_real_sparse_routed_mlp_router_weight_bytes > 0);
        assert!(scheduler_progression.device_real_sparse_routed_mlp_router_bias_bytes > 0);
        assert!(scheduler_progression.device_real_sparse_routed_mlp_route_cache_cuda_entries > 0);
        assert!(scheduler_progression.device_real_sparse_routed_mlp_route_cache_cuda_uploads > 0);
        assert_eq!(
            scheduler_progression.device_real_sparse_routed_mlp_router_cache_entries,
            sparse_layers
        );
        assert_eq!(
            scheduler_progression.device_real_sparse_routed_mlp_layers,
            sparse_layers
        );
        assert_eq!(
            scheduler_progression.device_real_sparse_routed_mlp_source_segments,
            expected_real_sparse_routed_mlp_source_segments
        );
    } else {
        assert_eq!(
            scheduler_progression.device_real_sparse_routed_mlp_delta_status,
            "not-run"
        );
        assert!(!scheduler_progression.uses_device_real_sparse_routed_mlp_delta);
        assert_eq!(
            scheduler_progression.device_real_sparse_routed_mlp_delta_rows,
            0
        );
        assert_eq!(
            scheduler_progression.device_real_sparse_routed_mlp_delta_values,
            0
        );
        assert_eq!(
            scheduler_progression.device_real_sparse_routed_mlp_delta_checksum,
            0.0
        );
        assert_eq!(
            scheduler_progression.device_real_sparse_routed_mlp_delta_backend,
            "not-run"
        );
        assert_eq!(
            scheduler_progression.device_real_sparse_routed_mlp_route_backend,
            "not-run"
        );
        assert_eq!(
            scheduler_progression.device_real_sparse_routed_mlp_router_backend,
            "not-run"
        );
        assert_eq!(
            scheduler_progression.device_real_sparse_routed_mlp_routes,
            0
        );
        assert_eq!(
            scheduler_progression.device_real_sparse_routed_mlp_router_weight_bytes,
            0
        );
        assert_eq!(
            scheduler_progression.device_real_sparse_routed_mlp_router_bias_bytes,
            0
        );
        assert_eq!(
            scheduler_progression.device_real_sparse_routed_mlp_route_cache_cuda_entries,
            0
        );
        assert_eq!(
            scheduler_progression.device_real_sparse_routed_mlp_route_cache_cuda_uploads,
            0
        );
        assert_eq!(
            scheduler_progression.device_real_sparse_routed_mlp_route_cache_cuda_hits,
            0
        );
        assert_eq!(
            scheduler_progression.device_real_sparse_routed_mlp_router_cache_entries,
            0
        );
        assert_eq!(
            scheduler_progression.device_real_sparse_routed_mlp_router_cache_hits,
            0
        );
        assert_eq!(
            scheduler_progression.device_real_sparse_routed_mlp_layers,
            0
        );
        assert_eq!(
            scheduler_progression.device_real_sparse_routed_mlp_source_segments,
            0
        );
    }
    if coordinator_cuda_reference_kernels_enabled() {
        assert_eq!(
            scheduler_progression.device_hidden_segment_status,
            "cuda-device-hidden-segment-residual-add"
        );
        assert!(scheduler_progression.uses_device_hidden_segment_residual_add);
        assert_eq!(
            scheduler_progression.device_hidden_segment_residual_adds,
            expected_source_segments * 2
        );
        assert_eq!(
            scheduler_progression.device_hidden_segment_value_updates,
            expected_value_updates * 2
        );
        assert!(scheduler_progression
            .device_hidden_segment_residual_add_backend
            .contains("residual-add-bf16"));
        assert_eq!(
            scheduler_progression.device_hidden_segment_resident_segments,
            4
        );
        assert_eq!(
            scheduler_progression.device_hidden_segment_resident_values,
            expected_resident_values
        );
        assert_eq!(
            scheduler_progression.device_hidden_segment_final_checksum,
            scheduler_progression.expected_device_hidden_segment_final_checksum
        );
    } else {
        assert_eq!(
            scheduler_progression.device_hidden_segment_status,
            "not-run"
        );
        assert!(!scheduler_progression.uses_device_hidden_segment_residual_add);
        assert_eq!(scheduler_progression.device_hidden_segment_residual_adds, 0);
        assert_eq!(scheduler_progression.device_hidden_segment_value_updates, 0);
        assert_eq!(
            scheduler_progression.device_hidden_segment_residual_add_backend,
            "not-run"
        );
        assert_eq!(
            scheduler_progression.device_hidden_segment_resident_segments,
            0
        );
        assert_eq!(
            scheduler_progression.device_hidden_segment_resident_values,
            0
        );
        assert_eq!(
            scheduler_progression.device_hidden_segment_final_checksum,
            0.0
        );
        assert_eq!(
            scheduler_progression.expected_device_hidden_segment_final_checksum,
            0.0
        );
    }
    assert!(scheduler_progression.final_visible_checksum.is_finite());
    assert_eq!(
        scheduler_progression.final_visible_checksum,
        scheduler_progression.expected_visible_checksum
    );
    assert!(scheduler_progression.rejected_mtp_checksum.is_finite());
    assert_eq!(
        scheduler_progression.rejected_mtp_checksum,
        scheduler_progression.expected_rejected_mtp_checksum
    );
    let terminal_lm_head_sample = &report.scheduler_execution_dry_run.terminal_lm_head_sample;
    if coordinator_cuda_reference_kernels_enabled() {
        assert!(terminal_lm_head_sample.uses_final_decode_device_hidden);
        assert!(terminal_lm_head_sample.passed);
        assert!(
            report
                .scheduler_execution_dry_run
                .full_context_device_attention_complete
        );
        assert_eq!(terminal_lm_head_sample.status, "sampled");
        assert_eq!(
            report.scheduler_execution_dry_run.status,
            "admitted-scheduler-terminal-lm-head-sampled"
        );
        assert!(terminal_lm_head_sample.covers_full_vocabulary);
        assert_eq!(
            terminal_lm_head_sample.logits_evaluated,
            terminal_lm_head_sample.vocab_size
        );
        assert!(terminal_lm_head_sample.argmax_kernel_backend.is_some());
        assert!(terminal_lm_head_sample.sampler_kernel_backend.is_some());
        assert!(terminal_lm_head_sample.blocker.is_none());
    } else {
        assert_eq!(
            report.scheduler_execution_dry_run.status,
            "admitted-scheduler-dry-run"
        );
        assert_eq!(terminal_lm_head_sample.status, "not-run");
        assert!(!terminal_lm_head_sample.uses_final_decode_device_hidden);
        assert!(!terminal_lm_head_sample.passed);
        assert!(terminal_lm_head_sample.blocker.is_some());
    }
    assert!(report.scheduler_execution_dry_run.layer_order_verified);
}

fn assert_scheduler_device_kv_status(status: &str, uses_device_kv_cache: bool) {
    if coordinator_cuda_reference_kernels_enabled() {
        assert_eq!(status, "cuda-kv-cache-live-scheduler");
        assert!(uses_device_kv_cache);
    } else {
        assert!(matches!(
            status,
            "cuda-kv-cache-live-scheduler" | "cuda-kv-cache-unavailable" | "cuda-kv-cache-error"
        ));
    }
}
