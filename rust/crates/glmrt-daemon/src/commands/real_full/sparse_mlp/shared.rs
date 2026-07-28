use anyhow::Result;
use glmrt_core::{TensorCatalog, GLM52_HIDDEN_SIZE, GLM52_TOP_K};

use super::super::coordinator_kernels::DeviceBf16Output;
use super::super::types::{RealFullExpertSparseMlpSharedChainLayerProbe, RealFullSparseMoeRoute};
use execution::{
    bounded_sparse_mlp_shared_chain_execution_plan,
    execute_real_sparse_mlp_shared_layer_from_hidden_with_plan,
    execute_real_sparse_mlp_shared_layer_from_hidden_with_plan_and_device_input,
    SparseMlpSharedChainExecutionPlan,
};

mod execution;
mod prefix;

const REAL_FULL_SHARED_CHAIN_TOP_K: usize = GLM52_TOP_K;
const REAL_FULL_SHARED_CHAIN_ROUTED_INTERMEDIATE_ROWS: usize = 4;
const REAL_FULL_SHARED_CHAIN_SHARED_INTERMEDIATE_ROWS: usize = 4;
const REAL_FULL_SHARED_CHAIN_OUTPUT_ROWS: usize = 4;
const REAL_FULL_SHARED_CHAIN_FULL_OUTPUT_ROWS: usize = GLM52_HIDDEN_SIZE;

pub(in crate::commands::real_full) struct RealFullSparseMlpSharedLayerHidden {
    pub(in crate::commands::real_full) hidden: Vec<f32>,
    pub(in crate::commands::real_full) device_hidden: Option<DeviceBf16Output>,
    pub(in crate::commands::real_full) expert_input_hidden_bf16_payload: Vec<u8>,
    pub(in crate::commands::real_full) layer_id: usize,
    pub(in crate::commands::real_full) route_count: usize,
    pub(in crate::commands::real_full) routes: Vec<RealFullSparseMoeRoute>,
    pub(in crate::commands::real_full) routed_outputs: Vec<f32>,
    pub(in crate::commands::real_full) shared_outputs: Vec<f32>,
    pub(in crate::commands::real_full) layer_outputs: Vec<f32>,
    pub(in crate::commands::real_full) shared_expert_executed: bool,
    pub(in crate::commands::real_full) routed_intermediate_rows: usize,
    pub(in crate::commands::real_full) shared_intermediate_rows: usize,
    pub(in crate::commands::real_full) output_rows: usize,
    pub(in crate::commands::real_full) residual_adds: usize,
    pub(in crate::commands::real_full) final_residual_checksum: f64,
    pub(in crate::commands::real_full) expert_input_norm_backend: &'static str,
    pub(in crate::commands::real_full) router_backend: &'static str,
    pub(in crate::commands::real_full) shared_mlp_backend: &'static str,
    pub(in crate::commands::real_full) residual_add_backend: &'static str,
    pub(in crate::commands::real_full) layer_summary: RealFullExpertSparseMlpSharedChainLayerProbe,
    pub(in crate::commands::real_full) covers_full_top_k: bool,
    pub(in crate::commands::real_full) passed: bool,
}

pub(in crate::commands::real_full) fn real_sparse_mlp_shared_layer_hidden_from_initial(
    catalog: &TensorCatalog,
    layer_id: usize,
    initial_hidden: Vec<f32>,
) -> Result<RealFullSparseMlpSharedLayerHidden> {
    let execution = execute_real_sparse_mlp_shared_layer_from_hidden_with_plan(
        catalog,
        layer_id,
        initial_hidden,
        bounded_sparse_mlp_shared_chain_execution_plan(),
    )?;
    Ok(real_sparse_mlp_shared_layer_hidden_from_execution(
        execution,
    ))
}

pub(in crate::commands::real_full) fn real_sparse_mlp_shared_layer_full_output_hidden_from_initial(
    catalog: &TensorCatalog,
    layer_id: usize,
    initial_hidden: Vec<f32>,
) -> Result<RealFullSparseMlpSharedLayerHidden> {
    let execution = execute_real_sparse_mlp_shared_layer_from_hidden_with_plan(
        catalog,
        layer_id,
        initial_hidden,
        full_output_sparse_mlp_shared_chain_execution_plan(),
    )?;
    Ok(real_sparse_mlp_shared_layer_hidden_from_execution(
        execution,
    ))
}

pub(in crate::commands::real_full) fn real_sparse_mlp_shared_layer_full_output_hidden_from_initial_device_input(
    catalog: &TensorCatalog,
    layer_id: usize,
    initial_hidden: Vec<f32>,
    device_hidden: &DeviceBf16Output,
) -> Result<RealFullSparseMlpSharedLayerHidden> {
    let execution = execute_real_sparse_mlp_shared_layer_from_hidden_with_plan_and_device_input(
        catalog,
        layer_id,
        initial_hidden,
        device_hidden,
        full_output_sparse_mlp_shared_chain_execution_plan(),
    )?;
    Ok(real_sparse_mlp_shared_layer_hidden_from_execution(
        execution,
    ))
}

fn real_sparse_mlp_shared_layer_hidden_from_execution(
    execution: execution::RealSparseMlpSharedLayerExecution,
) -> RealFullSparseMlpSharedLayerHidden {
    RealFullSparseMlpSharedLayerHidden {
        hidden: execution.hidden_after_layer,
        device_hidden: execution.device_hidden_after_layer,
        expert_input_hidden_bf16_payload: execution.expert_input_hidden_bf16_payload,
        layer_id: execution.layer_summary.layer_id,
        route_count: execution.routes_executed,
        routes: execution.routes,
        routed_outputs: execution.routed_outputs,
        shared_outputs: execution.shared_outputs,
        layer_outputs: execution.layer_outputs,
        shared_expert_executed: execution.shared_expert_executed,
        routed_intermediate_rows: execution.routed_intermediate_rows,
        shared_intermediate_rows: execution.shared_intermediate_rows,
        output_rows: execution.output_rows,
        residual_adds: 1,
        final_residual_checksum: execution.final_residual_checksum,
        expert_input_norm_backend: execution.expert_input_norm_backend,
        router_backend: execution.router_backend,
        shared_mlp_backend: execution.shared_mlp_backend,
        residual_add_backend: execution.residual_add_backend,
        layer_summary: execution.layer_summary,
        covers_full_top_k: execution.covers_full_top_k,
        passed: execution.passed,
    }
}

fn full_output_sparse_mlp_shared_chain_execution_plan() -> SparseMlpSharedChainExecutionPlan {
    SparseMlpSharedChainExecutionPlan {
        routed_intermediate_rows: REAL_FULL_SHARED_CHAIN_ROUTED_INTERMEDIATE_ROWS,
        shared_intermediate_rows: REAL_FULL_SHARED_CHAIN_SHARED_INTERMEDIATE_ROWS,
        output_rows: REAL_FULL_SHARED_CHAIN_FULL_OUTPUT_ROWS,
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::File, path::PathBuf};

    use crate::commands::real_full::types::{
        RealFullExpertSparseMlpSharedChainLayerProbe, RealFullSparseMoeRoute,
    };
    use glmrt_core::{TensorCatalog, GLM52_FIRST_K_DENSE_REPLACE, GLM52_HIDDEN_SIZE, GLM52_TOP_K};

    use super::{
        execution::RealSparseMlpSharedLayerExecution,
        real_sparse_mlp_shared_layer_full_output_hidden_from_initial,
        real_sparse_mlp_shared_layer_hidden_from_execution,
    };

    #[test]
    fn sparse_mlp_shared_layer_hidden_carries_topk_route_metadata() {
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
        let execution = RealSparseMlpSharedLayerExecution {
            hidden_after_layer: vec![1.0, 2.0, 3.0],
            device_hidden_after_layer: None,
            expert_input_hidden_bf16_payload: vec![0; GLM52_HIDDEN_SIZE * 2],
            layer_summary: RealFullExpertSparseMlpSharedChainLayerProbe {
                layer_id: GLM52_FIRST_K_DENSE_REPLACE,
                expert_id: routes[0].expert_id,
                owner: routes[0].owner.clone(),
                score: routes[0].score,
                corrected_score: routes[0].corrected_score,
                routed_output_checksum: 1.0,
                shared_output_checksum: 2.0,
                output_checksum: 3.0,
                output_l2_norm: 4.0,
                residual_before_checksum: 5.0,
                residual_delta_checksum: 6.0,
                residual_after_checksum: 7.0,
                expert_input_norm_backend: "cpu-reference-rmsnorm-bf16",
                router_backend: "cpu-reference-router-topk-bf16",
                shared_mlp_backend: "cpu-reference-silu-gated-mlp-bf16",
                residual_add_backend: "cpu-reference-residual-add-bf16",
                first_residual_after: 8.0,
                last_residual_after: 9.0,
            },
            routes: routes.clone(),
            routed_outputs: vec![0.1, 0.2, 0.3, 0.4],
            shared_outputs: vec![1.1, 1.2, 1.3, 1.4],
            layer_outputs: vec![1.2, 1.4, 1.6, 1.8],
            routes_executed: routes.len(),
            shared_expert_executed: true,
            routed_intermediate_rows: 4,
            shared_intermediate_rows: 4,
            output_rows: 4,
            final_residual_checksum: 7.0,
            expert_input_norm_backend: "cpu-reference-rmsnorm-bf16",
            router_backend: "cpu-reference-router-topk-bf16",
            shared_mlp_backend: "cpu-reference-silu-gated-mlp-bf16",
            residual_add_backend: "cpu-reference-residual-add-bf16",
            covers_full_top_k: false,
            passed: true,
        };

        let hidden = real_sparse_mlp_shared_layer_hidden_from_execution(execution);

        assert_eq!(hidden.layer_id, GLM52_FIRST_K_DENSE_REPLACE);
        assert_eq!(hidden.route_count, routes.len());
        assert_eq!(hidden.routed_outputs, vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(hidden.shared_outputs, vec![1.1, 1.2, 1.3, 1.4]);
        assert_eq!(hidden.layer_outputs, vec![1.2, 1.4, 1.6, 1.8]);
        assert_eq!(hidden.routes, routes);
        assert_eq!(hidden.routes[0].rank, 0);
        assert_eq!(hidden.routes[1].owner, "spark-1");
        assert_eq!(hidden.final_residual_checksum, 7.0);
    }

    #[test]
    fn real_checkpoint_sparse_mlp_shared_layer_full_output_when_available() {
        let Some(catalog) = load_real_catalog_or_skip() else {
            return;
        };

        let hidden = real_sparse_mlp_shared_layer_full_output_hidden_from_initial(
            &catalog,
            GLM52_FIRST_K_DENSE_REPLACE,
            crate::commands::real_full::sparse_mlp::math::deterministic_probe_hidden(
                GLM52_HIDDEN_SIZE,
            ),
        )
        .expect("running real sparse MLP shared layer full-output helper");

        assert_eq!(hidden.layer_id, GLM52_FIRST_K_DENSE_REPLACE);
        assert_eq!(hidden.route_count, GLM52_TOP_K);
        assert_eq!(hidden.output_rows, GLM52_HIDDEN_SIZE);
        assert_eq!(hidden.routed_intermediate_rows, 4);
        assert_eq!(hidden.shared_intermediate_rows, 4);
        assert!(hidden.shared_expert_executed);
        assert!(hidden.covers_full_top_k);
        assert!(hidden.final_residual_checksum.is_finite());
        assert!(hidden.passed);
    }

    fn load_real_catalog_or_skip() -> Option<TensorCatalog> {
        if !crate::commands::real_full::tests::fixture::real_checkpoint_tests_enabled() {
            crate::commands::real_full::tests::fixture::real_checkpoint_tests_skip_message(
                "sparse shared MLP",
            );
            return None;
        }
        let catalog_path =
            repo_root().join(".glmrt-cache/model-artifacts/diagnostic/model_catalog.json");
        let Ok(file) = File::open(&catalog_path) else {
            eprintln!("skipped: missing {}", catalog_path.display());
            return None;
        };
        let catalog: TensorCatalog =
            serde_json::from_reader(file).expect("parsing real GLM catalog fixture");
        if !std::path::Path::new(&catalog.snapshot_path).exists() {
            eprintln!("skipped: missing snapshot {}", catalog.snapshot_path);
            return None;
        }
        Some(catalog)
    }

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
    }
}
