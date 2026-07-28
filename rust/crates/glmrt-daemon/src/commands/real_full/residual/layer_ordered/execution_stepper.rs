use glmrt_core::GLM52_HIDDEN_SIZE;
use std::collections::BTreeSet;

use super::super::super::types::{
    RealFullBoundedAttentionOracleStepperEvidence, RealFullLayerOrderedResidualExecutionStep,
    RealFullResidualCompletionGates, RealFullResidualExecutionStepper,
    REAL_FULL_RESIDUAL_COMPLETION_BLOCKER,
};

pub(super) struct RealExecutionStepper {
    layer_count: usize,
    row_mode: &'static str,
    steps: Vec<RealFullLayerOrderedResidualExecutionStep>,
    next_layer_id: usize,
    expected_stage: ExpectedStage,
    stage_order_verified: bool,
}

pub(super) struct RealExecutionStepperFinish {
    pub(super) covers_all_dense_layers: bool,
    pub(super) covers_all_sparse_layers: bool,
    pub(super) covers_full_top_k: bool,
    pub(super) covers_full_output_rows: bool,
    pub(super) uses_embedding_residual_input: bool,
    pub(super) uses_live_scheduler_rows: bool,
    pub(super) uses_full_context_mla_dsa_attention: bool,
    pub(super) uses_live_expert_daemon_moe: bool,
    pub(super) uses_real_lm_head_sampling_residual: bool,
    pub(super) uses_full_model_residual: bool,
    pub(super) coordinator_graph_slots: usize,
    pub(super) coordinator_graph_captured_graphs: usize,
    pub(super) coordinator_graph_captures: usize,
    pub(super) coordinator_graph_launches: usize,
    pub(super) bounded_attention_oracle: RealFullBoundedAttentionOracleStepperEvidence,
    pub(super) full_residual_stream_blocker: Option<&'static str>,
    pub(super) final_residual_checksum: Option<f64>,
}

pub(super) struct RealExecutionStepperOutput {
    pub(super) report: RealFullResidualExecutionStepper,
    pub(super) steps: Vec<RealFullLayerOrderedResidualExecutionStep>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ExpectedStage {
    Attention,
    Mlp,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CoordinatorBackendClass {
    Cuda,
    Cpu,
    Unknown,
}

impl RealExecutionStepper {
    pub(super) fn new(layer_count: usize, row_mode: &'static str) -> Self {
        Self {
            layer_count,
            row_mode,
            steps: Vec::with_capacity(layer_count * 2),
            next_layer_id: 0,
            expected_stage: ExpectedStage::Attention,
            stage_order_verified: true,
        }
    }

    pub(super) fn record_attention(&mut self, step: RealFullLayerOrderedResidualExecutionStep) {
        self.record_stage(ExpectedStage::Attention, step);
    }

    pub(super) fn record_dense_mlp(&mut self, step: RealFullLayerOrderedResidualExecutionStep) {
        self.record_stage(ExpectedStage::Mlp, step);
    }

    pub(super) fn record_sparse_moe_mlp(
        &mut self,
        step: RealFullLayerOrderedResidualExecutionStep,
    ) {
        self.record_stage(ExpectedStage::Mlp, step);
    }

    pub(super) fn finish(self, finish: RealExecutionStepperFinish) -> RealExecutionStepperOutput {
        let traced_layers = self
            .steps
            .iter()
            .map(|step| step.layer_id)
            .collect::<BTreeSet<_>>()
            .len();
        let attention_steps_executed = self
            .steps
            .iter()
            .filter(|step| step.stage == "attention" && step.executed)
            .count();
        let attention_steps_missing = self
            .steps
            .iter()
            .filter(|step| step.stage == "attention" && !step.executed)
            .count();
        let dense_mlp_steps_executed = self
            .steps
            .iter()
            .filter(|step| step.stage == "dense_mlp" && step.executed)
            .count();
        let sparse_mlp_steps_executed = self
            .steps
            .iter()
            .filter(|step| step.stage == "sparse_moe_mlp" && step.executed)
            .count();
        let shared_expert_steps_executed = self
            .steps
            .iter()
            .filter(|step| step.stage == "sparse_moe_mlp" && step.includes_shared_expert)
            .count();
        let planned_residual_adds = self.layer_count * 2;
        let total_numeric_residual_adds = self
            .steps
            .iter()
            .filter(|step| step.executed)
            .map(|step| step.residual_adds)
            .sum::<usize>();
        let residual_adds_missing =
            planned_residual_adds.saturating_sub(total_numeric_residual_adds);
        let routed_routes = self
            .steps
            .iter()
            .map(|step| step.routes_executed)
            .sum::<usize>();
        let stage_sources_recorded = self
            .steps
            .iter()
            .filter(|step| !step.stage_source.is_empty())
            .count();
        let stage_statuses_recorded = self
            .steps
            .iter()
            .filter(|step| !step.stage_status.is_empty())
            .count();
        let real_stage_count = stage_count_by_status(&self.steps, "real");
        let synthetic_stage_count = stage_count_by_status(&self.steps, "synthetic");
        let provisional_stage_count = stage_count_by_status(&self.steps, "provisional");
        let blocked_stage_count = stage_count_by_status(&self.steps, "blocked");
        let coordinator_stage_count = self
            .steps
            .iter()
            .filter(|step| is_coordinator_compute_stage(step))
            .count();
        let coordinator_cuda_stage_count = self
            .steps
            .iter()
            .filter(|step| {
                is_coordinator_compute_stage(step)
                    && coordinator_backend_class(step) == CoordinatorBackendClass::Cuda
            })
            .count();
        let coordinator_cpu_stage_count = self
            .steps
            .iter()
            .filter(|step| {
                is_coordinator_compute_stage(step)
                    && coordinator_backend_class(step) == CoordinatorBackendClass::Cpu
            })
            .count();
        let coordinator_unknown_stage_count = coordinator_stage_count
            .saturating_sub(coordinator_cuda_stage_count + coordinator_cpu_stage_count);
        let uses_cuda_coordinator_kernels =
            coordinator_stage_count > 0 && coordinator_cuda_stage_count == coordinator_stage_count;
        let uses_graph_captured_coordinator_kernels = finish.coordinator_graph_launches > 0;
        let stages_with_numeric_checksums = self
            .steps
            .iter()
            .filter(|step| stage_has_all_numeric_checksums(step))
            .count();
        let total_numeric_checksum_fields = self
            .steps
            .iter()
            .map(numeric_checksum_field_count)
            .sum::<usize>();
        let numeric_checksum_fields_per_stage = self
            .steps
            .iter()
            .find(|step| step.executed)
            .map(numeric_checksum_field_count)
            .unwrap_or_default();
        let stages_with_tensor_artifacts = self
            .steps
            .iter()
            .filter(|step| !step.tensor_artifacts.is_empty())
            .count();
        let total_tensor_artifacts = self
            .steps
            .iter()
            .map(|step| step.tensor_artifacts.len())
            .sum::<usize>();
        let attention_tensor_artifacts_per_stage =
            first_tensor_artifact_count_for_stage(&self.steps, "attention");
        let dense_mlp_tensor_artifacts_per_stage =
            first_tensor_artifact_count_for_stage(&self.steps, "dense_mlp");
        let sparse_mlp_tensor_artifacts_per_stage =
            first_tensor_artifact_count_for_stage(&self.steps, "sparse_moe_mlp");
        let residual_prefix_values = self
            .steps
            .last()
            .map(|step| step.output_rows)
            .unwrap_or_default();
        let stage_order_verified = self.stage_order_verified
            && self.next_layer_id == self.layer_count
            && self.expected_stage == ExpectedStage::Attention;
        let covers_all_layers = traced_layers == self.layer_count
            && self.steps.len() == planned_residual_adds
            && stage_order_verified;
        let output_rows_are_full_width = residual_prefix_values == GLM52_HIDDEN_SIZE;
        let covers_full_output_rows = finish.covers_full_output_rows && output_rows_are_full_width;
        let numeric_layer_order_complete = covers_all_layers
            && finish.covers_all_dense_layers
            && finish.covers_all_sparse_layers
            && finish.covers_full_top_k;
        let attention_steps_complete = attention_steps_missing == 0;
        let residual_adds_complete = residual_adds_missing == 0;
        let full_residual_stream_complete = numeric_layer_order_complete
            && attention_steps_complete
            && residual_adds_complete
            && covers_full_output_rows
            && finish.uses_embedding_residual_input
            && finish.uses_live_scheduler_rows
            && uses_cuda_coordinator_kernels
            && uses_graph_captured_coordinator_kernels
            && finish.uses_full_context_mla_dsa_attention
            && finish.uses_live_expert_daemon_moe
            && finish.uses_real_lm_head_sampling_residual
            && finish.uses_full_model_residual;
        let missing_gate_names = missing_completion_gate_names(&[
            ("numeric_layer_order_complete", numeric_layer_order_complete),
            ("attention_steps_complete", attention_steps_complete),
            ("residual_adds_complete", residual_adds_complete),
            ("covers_full_output_rows", covers_full_output_rows),
            (
                "uses_embedding_residual_input",
                finish.uses_embedding_residual_input,
            ),
            ("uses_live_scheduler_rows", finish.uses_live_scheduler_rows),
            (
                "uses_cuda_coordinator_kernels",
                uses_cuda_coordinator_kernels,
            ),
            (
                "uses_graph_captured_coordinator_kernels",
                uses_graph_captured_coordinator_kernels,
            ),
            (
                "uses_full_context_mla_dsa_attention",
                finish.uses_full_context_mla_dsa_attention,
            ),
            (
                "uses_live_expert_daemon_moe",
                finish.uses_live_expert_daemon_moe,
            ),
            (
                "uses_real_lm_head_sampling_residual",
                finish.uses_real_lm_head_sampling_residual,
            ),
            ("uses_full_model_residual", finish.uses_full_model_residual),
        ]);
        let missing_gate_count = missing_gate_names.len();
        let completion_gates = RealFullResidualCompletionGates {
            numeric_layer_order_complete,
            attention_steps_complete,
            residual_adds_complete,
            covers_full_output_rows,
            uses_embedding_residual_input: finish.uses_embedding_residual_input,
            uses_live_scheduler_rows: finish.uses_live_scheduler_rows,
            uses_cuda_coordinator_kernels,
            uses_graph_captured_coordinator_kernels,
            uses_full_context_mla_dsa_attention: finish.uses_full_context_mla_dsa_attention,
            uses_live_expert_daemon_moe: finish.uses_live_expert_daemon_moe,
            uses_real_lm_head_sampling_residual: finish.uses_real_lm_head_sampling_residual,
            uses_full_model_residual: finish.uses_full_model_residual,
            ready_for_full_residual_stream: full_residual_stream_complete,
            missing_gate_count,
            missing_gate_names,
        };
        let full_residual_stream_blocker = if full_residual_stream_complete {
            None
        } else {
            finish
                .full_residual_stream_blocker
                .or(Some(REAL_FULL_RESIDUAL_COMPLETION_BLOCKER))
        };
        let status = if covers_all_layers
            && residual_adds_missing == 0
            && self.row_mode == "full-output-attention-mlp"
        {
            "real-execution-stepper-full-output-attention-mlp-trace"
        } else if covers_all_layers
            && residual_adds_missing == 0
            && self.row_mode == "full-output-mla-rope-attention-mlp"
        {
            "real-execution-stepper-full-output-mla-rope-attention-mlp-trace"
        } else if covers_all_layers
            && residual_adds_missing == 0
            && self.row_mode == "full-output-mla-rope-attention"
        {
            "real-execution-stepper-full-output-mla-rope-attention-trace"
        } else if covers_all_layers
            && residual_adds_missing == 0
            && self.row_mode == "mla-rope-attention"
        {
            "real-execution-stepper-bounded-mla-rope-attention-trace"
        } else if covers_all_layers
            && residual_adds_missing == 0
            && self.row_mode == "full-output-mlp"
        {
            "real-execution-stepper-full-output-mlp-bounded-attention-trace"
        } else if covers_all_layers && residual_adds_missing == 0 {
            "real-execution-stepper-bounded-all-stage-trace"
        } else {
            "real-execution-stepper-incomplete"
        };

        RealExecutionStepperOutput {
            report: RealFullResidualExecutionStepper {
                status,
                scope: "records reusable real GLM-5.2 residual execution stages in model order; current trace is bounded and reports full-output/full-model completion separately",
                row_mode: self.row_mode,
                layer_count: self.layer_count,
                traced_layers,
                trace_steps: self.steps.len(),
                attention_steps_executed,
                attention_steps_missing,
                dense_mlp_steps_executed,
                sparse_mlp_steps_executed,
                shared_expert_steps_executed,
                planned_residual_adds,
                total_numeric_residual_adds,
                residual_adds_missing,
                residual_prefix_values,
                routed_routes,
                stage_sources_recorded,
                stage_statuses_recorded,
                real_stage_count,
                synthetic_stage_count,
                provisional_stage_count,
                blocked_stage_count,
                coordinator_stage_count,
                coordinator_cuda_stage_count,
                coordinator_cpu_stage_count,
                coordinator_unknown_stage_count,
                uses_cuda_coordinator_kernels,
                coordinator_graph_slots: finish.coordinator_graph_slots,
                coordinator_graph_captured_graphs: finish.coordinator_graph_captured_graphs,
                coordinator_graph_captures: finish.coordinator_graph_captures,
                coordinator_graph_launches: finish.coordinator_graph_launches,
                uses_graph_captured_coordinator_kernels,
                stages_with_numeric_checksums,
                total_numeric_checksum_fields,
                numeric_checksum_fields_per_stage,
                stages_with_tensor_artifacts,
                total_tensor_artifacts,
                attention_tensor_artifacts_per_stage,
                dense_mlp_tensor_artifacts_per_stage,
                sparse_mlp_tensor_artifacts_per_stage,
                final_residual_checksum: finish.final_residual_checksum,
                covers_all_layers,
                covers_all_dense_layers: finish.covers_all_dense_layers,
                covers_all_sparse_layers: finish.covers_all_sparse_layers,
                covers_full_top_k: finish.covers_full_top_k,
                covers_full_output_rows,
                stage_order_verified,
                full_residual_stream_complete,
                uses_full_model_residual: finish.uses_full_model_residual,
                bounded_attention_oracle: finish.bounded_attention_oracle,
                completion_gates,
                full_residual_stream_blocker,
            },
            steps: self.steps,
        }
    }

    fn record_stage(
        &mut self,
        expected_stage: ExpectedStage,
        step: RealFullLayerOrderedResidualExecutionStep,
    ) {
        self.stage_order_verified = self.stage_order_verified
            && self.expected_stage == expected_stage
            && step.layer_id == self.next_layer_id;
        if expected_stage == ExpectedStage::Attention {
            self.expected_stage = ExpectedStage::Mlp;
        } else {
            self.expected_stage = ExpectedStage::Attention;
            self.next_layer_id += 1;
        }
        self.steps.push(step);
    }
}

fn is_coordinator_compute_stage(step: &RealFullLayerOrderedResidualExecutionStep) -> bool {
    step.executed && matches!(step.stage, "attention" | "dense_mlp" | "sparse_moe_mlp")
}

fn coordinator_backend_class(
    step: &RealFullLayerOrderedResidualExecutionStep,
) -> CoordinatorBackendClass {
    if step.stage_source.contains("cpu-reference") {
        CoordinatorBackendClass::Cpu
    } else if step.stage_source.contains("cuda-reference") {
        CoordinatorBackendClass::Cuda
    } else {
        CoordinatorBackendClass::Unknown
    }
}

fn missing_completion_gate_names(gates: &[(&'static str, bool)]) -> Vec<&'static str> {
    gates
        .iter()
        .filter_map(|(name, passed)| (!*passed).then_some(*name))
        .collect()
}

fn first_tensor_artifact_count_for_stage(
    steps: &[RealFullLayerOrderedResidualExecutionStep],
    stage: &'static str,
) -> usize {
    steps
        .iter()
        .find(|step| step.stage == stage)
        .map(|step| step.tensor_artifacts.len())
        .unwrap_or_default()
}

fn stage_count_by_status(
    steps: &[RealFullLayerOrderedResidualExecutionStep],
    status: &'static str,
) -> usize {
    steps
        .iter()
        .filter(|step| step.stage_status == status)
        .count()
}

fn stage_has_all_numeric_checksums(step: &RealFullLayerOrderedResidualExecutionStep) -> bool {
    numeric_checksum_field_count(step) == 3
}

fn numeric_checksum_field_count(step: &RealFullLayerOrderedResidualExecutionStep) -> usize {
    [
        step.residual_before_checksum,
        step.residual_delta_checksum,
        step.residual_after_checksum,
    ]
    .into_iter()
    .filter(|checksum| checksum.is_some_and(f64::is_finite))
    .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glmrt_core::GLM52_TOP_K;

    #[test]
    fn bounded_tiny_stepper_runs_without_full_model_catalog() {
        let mut stepper = RealExecutionStepper::new(2, "tiny-bounded-no-catalog");
        stepper.record_attention(tiny_step(
            0,
            "attention",
            "synthetic-tiny-attention",
            0.0,
            0.5,
        ));
        stepper.record_dense_mlp(tiny_step(
            0,
            "dense_mlp",
            "synthetic-tiny-dense-mlp",
            0.5,
            1.5,
        ));
        stepper.record_attention(tiny_step(
            1,
            "attention",
            "synthetic-tiny-attention",
            1.5,
            3.0,
        ));
        stepper.record_sparse_moe_mlp(RealFullLayerOrderedResidualExecutionStep {
            layer_id: 1,
            stage: "sparse_moe_mlp",
            stage_source: "synthetic-tiny-sparse-moe",
            stage_status: "synthetic",
            executed: true,
            residual_adds: 1,
            output_rows: 2,
            routes_executed: 2,
            selected_routes: Vec::new(),
            expert_host_batch_set: None,
            includes_shared_expert: true,
            tensor_artifacts: Vec::new(),
            residual_before_checksum: Some(3.0),
            residual_delta_checksum: Some(4.0),
            residual_after_checksum: Some(7.0),
            missing_reason: None,
        });

        let output = stepper.finish(RealExecutionStepperFinish {
            covers_all_dense_layers: true,
            covers_all_sparse_layers: true,
            covers_full_top_k: true,
            covers_full_output_rows: false,
            uses_embedding_residual_input: false,
            uses_live_scheduler_rows: false,
            uses_full_context_mla_dsa_attention: false,
            uses_live_expert_daemon_moe: false,
            uses_real_lm_head_sampling_residual: false,
            uses_full_model_residual: false,
            coordinator_graph_slots: 0,
            coordinator_graph_captured_graphs: 0,
            coordinator_graph_captures: 0,
            coordinator_graph_launches: 0,
            bounded_attention_oracle: RealFullBoundedAttentionOracleStepperEvidence::default(),
            full_residual_stream_blocker: Some(REAL_FULL_RESIDUAL_COMPLETION_BLOCKER),
            final_residual_checksum: Some(7.0),
        });

        assert_eq!(output.report.row_mode, "tiny-bounded-no-catalog");
        assert_eq!(output.report.layer_count, 2);
        assert_eq!(output.report.traced_layers, 2);
        assert_eq!(output.report.trace_steps, 4);
        assert_eq!(output.report.attention_steps_executed, 2);
        assert_eq!(output.report.dense_mlp_steps_executed, 1);
        assert_eq!(output.report.sparse_mlp_steps_executed, 1);
        assert_eq!(output.report.shared_expert_steps_executed, 1);
        assert_eq!(output.report.total_numeric_residual_adds, 4);
        assert_eq!(output.report.residual_adds_missing, 0);
        assert_eq!(output.report.residual_prefix_values, 2);
        assert_eq!(output.report.routed_routes, 2);
        assert_eq!(output.report.stage_sources_recorded, 4);
        assert_eq!(output.report.stage_statuses_recorded, 4);
        assert_eq!(output.report.synthetic_stage_count, 4);
        assert_eq!(output.report.real_stage_count, 0);
        assert_eq!(output.report.provisional_stage_count, 0);
        assert_eq!(output.report.blocked_stage_count, 0);
        assert_eq!(output.report.coordinator_stage_count, 4);
        assert_eq!(output.report.coordinator_cuda_stage_count, 0);
        assert_eq!(output.report.coordinator_cpu_stage_count, 0);
        assert_eq!(output.report.coordinator_unknown_stage_count, 4);
        assert!(!output.report.uses_cuda_coordinator_kernels);
        assert_eq!(output.report.coordinator_graph_slots, 0);
        assert_eq!(output.report.coordinator_graph_captured_graphs, 0);
        assert_eq!(output.report.coordinator_graph_captures, 0);
        assert_eq!(output.report.coordinator_graph_launches, 0);
        assert!(!output.report.uses_graph_captured_coordinator_kernels);
        assert_eq!(output.report.stages_with_numeric_checksums, 4);
        assert_eq!(output.report.total_numeric_checksum_fields, 12);
        assert_eq!(output.report.numeric_checksum_fields_per_stage, 3);
        assert_eq!(output.report.stages_with_tensor_artifacts, 0);
        assert_eq!(output.report.total_tensor_artifacts, 0);
        assert!(output.report.covers_all_layers);
        assert!(output.report.stage_order_verified);
        assert!(!output.report.covers_full_output_rows);
        assert!(!output.report.full_residual_stream_complete);
        assert!(!output.report.uses_full_model_residual);
        assert_eq!(output.report.bounded_attention_oracle.status, "not-run");
        assert!(!output.report.bounded_attention_oracle.passed);
        assert!(output.report.completion_gates.numeric_layer_order_complete);
        assert!(output.report.completion_gates.attention_steps_complete);
        assert!(output.report.completion_gates.residual_adds_complete);
        assert!(!output.report.completion_gates.covers_full_output_rows);
        assert!(!output.report.completion_gates.uses_embedding_residual_input);
        assert!(!output.report.completion_gates.uses_live_scheduler_rows);
        assert!(!output.report.completion_gates.uses_cuda_coordinator_kernels);
        assert!(
            !output
                .report
                .completion_gates
                .uses_graph_captured_coordinator_kernels
        );
        assert!(
            !output
                .report
                .completion_gates
                .uses_full_context_mla_dsa_attention
        );
        assert!(!output.report.completion_gates.uses_live_expert_daemon_moe);
        assert!(
            !output
                .report
                .completion_gates
                .uses_real_lm_head_sampling_residual
        );
        assert!(!output.report.completion_gates.uses_full_model_residual);
        assert!(
            !output
                .report
                .completion_gates
                .ready_for_full_residual_stream
        );
        assert_eq!(output.report.completion_gates.missing_gate_count, 9);
        assert_eq!(
            output.report.completion_gates.missing_gate_names,
            vec![
                "covers_full_output_rows",
                "uses_embedding_residual_input",
                "uses_live_scheduler_rows",
                "uses_cuda_coordinator_kernels",
                "uses_graph_captured_coordinator_kernels",
                "uses_full_context_mla_dsa_attention",
                "uses_live_expert_daemon_moe",
                "uses_real_lm_head_sampling_residual",
                "uses_full_model_residual"
            ]
        );
        assert_eq!(
            output.report.full_residual_stream_blocker,
            Some(REAL_FULL_RESIDUAL_COMPLETION_BLOCKER)
        );
        assert_eq!(output.report.final_residual_checksum, Some(7.0));
        assert_eq!(output.steps.len(), 4);
    }

    #[test]
    fn full_output_mla_rope_attention_mlp_stepper_status_is_distinct() {
        let mut stepper = RealExecutionStepper::new(1, "full-output-mla-rope-attention-mlp");
        stepper.record_attention(tiny_step(
            0,
            "attention",
            "synthetic-full-output-mla-rope-attention",
            0.0,
            0.5,
        ));
        stepper.record_dense_mlp(tiny_step(
            0,
            "dense_mlp",
            "synthetic-full-output-dense-mlp",
            0.5,
            1.5,
        ));

        let output = stepper.finish(RealExecutionStepperFinish {
            covers_all_dense_layers: true,
            covers_all_sparse_layers: true,
            covers_full_top_k: true,
            covers_full_output_rows: false,
            uses_embedding_residual_input: false,
            uses_live_scheduler_rows: false,
            uses_full_context_mla_dsa_attention: false,
            uses_live_expert_daemon_moe: false,
            uses_real_lm_head_sampling_residual: false,
            uses_full_model_residual: false,
            coordinator_graph_slots: 0,
            coordinator_graph_captured_graphs: 0,
            coordinator_graph_captures: 0,
            coordinator_graph_launches: 0,
            bounded_attention_oracle: RealFullBoundedAttentionOracleStepperEvidence::default(),
            full_residual_stream_blocker: Some(REAL_FULL_RESIDUAL_COMPLETION_BLOCKER),
            final_residual_checksum: Some(1.5),
        });

        assert_eq!(
            output.report.status,
            "real-execution-stepper-full-output-mla-rope-attention-mlp-trace"
        );
        assert_eq!(output.report.row_mode, "full-output-mla-rope-attention-mlp");
        assert!(output.report.covers_all_layers);
        assert!(output.report.completion_gates.numeric_layer_order_complete);
        assert!(!output.report.full_residual_stream_complete);
    }

    #[test]
    fn full_model_completion_gate_clears_when_all_runtime_evidence_is_present() {
        let mut stepper = RealExecutionStepper::new(2, "full-output-mla-rope-attention-mlp");
        stepper.record_attention(full_width_cuda_step(
            0,
            "attention",
            "cuda-reference-full-context-mla-dsa-attention",
            0.0,
            1.0,
            0,
        ));
        stepper.record_dense_mlp(full_width_cuda_step(
            0,
            "dense_mlp",
            "cuda-reference-dense-mlp",
            1.0,
            2.0,
            0,
        ));
        stepper.record_attention(full_width_cuda_step(
            1,
            "attention",
            "cuda-reference-full-context-mla-dsa-attention",
            2.0,
            3.0,
            0,
        ));
        stepper.record_sparse_moe_mlp(full_width_cuda_step(
            1,
            "sparse_moe_mlp",
            "cuda-reference-live-expert-daemon-moe",
            3.0,
            4.0,
            GLM52_TOP_K,
        ));

        let output = stepper.finish(RealExecutionStepperFinish {
            covers_all_dense_layers: true,
            covers_all_sparse_layers: true,
            covers_full_top_k: true,
            covers_full_output_rows: true,
            uses_embedding_residual_input: true,
            uses_live_scheduler_rows: true,
            uses_full_context_mla_dsa_attention: true,
            uses_live_expert_daemon_moe: true,
            uses_real_lm_head_sampling_residual: true,
            uses_full_model_residual: true,
            coordinator_graph_slots: 21,
            coordinator_graph_captured_graphs: 3,
            coordinator_graph_captures: 1,
            coordinator_graph_launches: 4,
            bounded_attention_oracle: RealFullBoundedAttentionOracleStepperEvidence::default(),
            full_residual_stream_blocker: Some(REAL_FULL_RESIDUAL_COMPLETION_BLOCKER),
            final_residual_checksum: Some(4.0),
        });

        assert!(output.report.covers_all_layers);
        assert!(output.report.covers_full_output_rows);
        assert!(output.report.uses_cuda_coordinator_kernels);
        assert!(output.report.uses_graph_captured_coordinator_kernels);
        assert!(output.report.uses_full_model_residual);
        assert!(output.report.full_residual_stream_complete);
        assert!(
            output
                .report
                .completion_gates
                .ready_for_full_residual_stream
        );
        assert_eq!(output.report.completion_gates.missing_gate_count, 0);
        assert!(output.report.completion_gates.missing_gate_names.is_empty());
        assert_eq!(output.report.full_residual_stream_blocker, None);
    }

    fn tiny_step(
        layer_id: usize,
        stage: &'static str,
        stage_source: &'static str,
        residual_before_checksum: f64,
        residual_after_checksum: f64,
    ) -> RealFullLayerOrderedResidualExecutionStep {
        RealFullLayerOrderedResidualExecutionStep {
            layer_id,
            stage,
            stage_source,
            stage_status: "synthetic",
            executed: true,
            residual_adds: 1,
            output_rows: 2,
            routes_executed: 0,
            selected_routes: Vec::new(),
            expert_host_batch_set: None,
            includes_shared_expert: false,
            tensor_artifacts: Vec::new(),
            residual_before_checksum: Some(residual_before_checksum),
            residual_delta_checksum: Some(residual_after_checksum - residual_before_checksum),
            residual_after_checksum: Some(residual_after_checksum),
            missing_reason: None,
        }
    }

    fn full_width_cuda_step(
        layer_id: usize,
        stage: &'static str,
        stage_source: &'static str,
        residual_before_checksum: f64,
        residual_after_checksum: f64,
        routes_executed: usize,
    ) -> RealFullLayerOrderedResidualExecutionStep {
        RealFullLayerOrderedResidualExecutionStep {
            layer_id,
            stage,
            stage_source,
            stage_status: "real",
            executed: true,
            residual_adds: 1,
            output_rows: GLM52_HIDDEN_SIZE,
            routes_executed,
            selected_routes: Vec::new(),
            expert_host_batch_set: None,
            includes_shared_expert: stage == "sparse_moe_mlp",
            tensor_artifacts: Vec::new(),
            residual_before_checksum: Some(residual_before_checksum),
            residual_delta_checksum: Some(residual_after_checksum - residual_before_checksum),
            residual_after_checksum: Some(residual_after_checksum),
            missing_reason: None,
        }
    }
}
