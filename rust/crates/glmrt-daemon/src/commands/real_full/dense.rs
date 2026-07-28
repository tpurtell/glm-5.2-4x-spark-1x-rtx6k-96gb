use anyhow::Result;
use glmrt_core::{TensorCatalog, GLM52_FIRST_K_DENSE_REPLACE};

use super::coordinator_kernels::DeviceBf16Output;
use super::probe_env;
use super::types::RealFullDensePrefixProbe;
use execution::{
    bounded_dense_prefix_execution_plan, execute_dense_layer_residual_from_hidden,
    execute_dense_layer_residual_from_hidden_with_plan,
    execute_dense_layer_residual_from_hidden_with_plan_and_device_input,
    execute_dense_prefix_with_plan, full_output_dense_prefix_execution_plan,
    DenseLayerResidualExecution, DensePrefixExecutionPlan,
};

mod execution;
pub(in crate::commands::real_full) mod math;

const REAL_FULL_PROBE_DENSE_PREFIX_ENV: &str = "GLMRT_REAL_FULL_PROBE_DENSE_PREFIX";
pub(super) const REAL_FULL_DENSE_PREFIX_INTERMEDIATE_ROWS: usize = 8;
pub(super) const REAL_FULL_DENSE_PREFIX_OUTPUT_ROWS: usize = 4;
pub(super) const REAL_FULL_DENSE_RMSNORM_EPS: f32 = 1.0e-5;

pub(in crate::commands::real_full) struct RealFullDenseLayerPrefixHidden {
    pub(in crate::commands::real_full) hidden: Vec<f32>,
    pub(in crate::commands::real_full) device_hidden: Option<DeviceBf16Output>,
    pub(in crate::commands::real_full) layer_id: usize,
    pub(in crate::commands::real_full) intermediate_rows: usize,
    pub(in crate::commands::real_full) output_rows: usize,
    pub(in crate::commands::real_full) residual_adds: usize,
    pub(in crate::commands::real_full) norm_bytes_read: u64,
    pub(in crate::commands::real_full) weight_bytes_read: u64,
    pub(in crate::commands::real_full) norm_backend: &'static str,
    pub(in crate::commands::real_full) linear_backend: &'static str,
    pub(in crate::commands::real_full) mlp_backend: &'static str,
    pub(in crate::commands::real_full) norm_checksum: f64,
    pub(in crate::commands::real_full) activation_checksum: f64,
    pub(in crate::commands::real_full) output_checksum: f64,
    pub(in crate::commands::real_full) output_l2_norm: f32,
    pub(in crate::commands::real_full) initial_residual_checksum: f64,
    pub(in crate::commands::real_full) residual_delta_checksum: f64,
    pub(in crate::commands::real_full) final_residual_checksum: f64,
    pub(in crate::commands::real_full) residual_add_backend: &'static str,
    pub(in crate::commands::real_full) first_residual_after: f32,
    pub(in crate::commands::real_full) last_residual_after: f32,
    pub(in crate::commands::real_full) passed: bool,
}

pub(super) fn real_full_dense_prefix_probe(catalog: &TensorCatalog) -> RealFullDensePrefixProbe {
    let env_value = probe_env::var_opt(REAL_FULL_PROBE_DENSE_PREFIX_ENV);
    let mode = dense_prefix_probe_mode(env_value.as_deref());
    if mode.disabled {
        return skipped_real_full_dense_prefix_probe(
            "not-run",
            mode,
            Some(format!(
                "{REAL_FULL_PROBE_DENSE_PREFIX_ENV}=0 disables the real dense-prefix probe"
            )),
        );
    }

    match run_real_full_dense_prefix_probe(catalog, mode) {
        Ok(probe) => probe,
        Err(error) => skipped_real_full_dense_prefix_probe("error", mode, Some(error.to_string())),
    }
}

pub(in crate::commands::real_full) fn real_full_dense_layer_prefix_hidden_from_initial(
    catalog: &TensorCatalog,
    layer_id: usize,
    initial_hidden: Vec<f32>,
) -> Result<RealFullDenseLayerPrefixHidden> {
    let execution = execute_dense_layer_residual_from_hidden(catalog, layer_id, initial_hidden)?;
    Ok(real_full_dense_layer_prefix_hidden_from_execution(
        execution,
    ))
}

pub(in crate::commands::real_full) fn real_full_dense_layer_full_output_hidden_from_initial(
    catalog: &TensorCatalog,
    layer_id: usize,
    initial_hidden: Vec<f32>,
) -> Result<RealFullDenseLayerPrefixHidden> {
    let execution = execute_dense_layer_residual_from_hidden_with_plan(
        catalog,
        layer_id,
        initial_hidden,
        full_output_dense_prefix_execution_plan(),
    )?;
    Ok(real_full_dense_layer_prefix_hidden_from_execution(
        execution,
    ))
}

pub(in crate::commands::real_full) fn real_full_dense_layer_full_output_hidden_from_initial_device_input(
    catalog: &TensorCatalog,
    layer_id: usize,
    initial_hidden: Vec<f32>,
    device_hidden: &DeviceBf16Output,
) -> Result<RealFullDenseLayerPrefixHidden> {
    let execution = execute_dense_layer_residual_from_hidden_with_plan_and_device_input(
        catalog,
        layer_id,
        initial_hidden,
        Some(device_hidden),
        full_output_dense_prefix_execution_plan(),
    )?;
    Ok(real_full_dense_layer_prefix_hidden_from_execution(
        execution,
    ))
}

fn real_full_dense_layer_prefix_hidden_from_execution(
    execution: DenseLayerResidualExecution,
) -> RealFullDenseLayerPrefixHidden {
    RealFullDenseLayerPrefixHidden {
        hidden: execution.hidden_after_layer,
        device_hidden: execution.device_hidden_after_layer,
        layer_id: execution.layer_id,
        intermediate_rows: execution.intermediate_rows,
        output_rows: execution.output_rows,
        residual_adds: execution.residual_adds,
        norm_bytes_read: execution.norm_bytes_read,
        weight_bytes_read: execution.weight_bytes_read,
        norm_backend: execution.norm_backend,
        linear_backend: execution.linear_backend,
        mlp_backend: execution.mlp_backend,
        norm_checksum: execution.norm_checksum,
        activation_checksum: execution.activation_checksum,
        output_checksum: execution.output_checksum,
        output_l2_norm: execution.output_l2_norm,
        initial_residual_checksum: execution.initial_residual_checksum,
        residual_delta_checksum: execution.residual_delta_checksum,
        final_residual_checksum: execution.final_residual_checksum,
        residual_add_backend: execution.residual_add_backend,
        first_residual_after: execution.first_residual_after,
        last_residual_after: execution.last_residual_after,
        passed: execution.passed,
    }
}

#[derive(Clone, Copy)]
struct DensePrefixProbeMode {
    row_mode: &'static str,
    disabled: bool,
    plan: DensePrefixExecutionPlan,
}

fn dense_prefix_probe_mode(env_setting: Option<&str>) -> DensePrefixProbeMode {
    let normalized = env_setting.map(|value| value.trim().to_ascii_lowercase());
    match normalized.as_deref() {
        Some("0") => DensePrefixProbeMode {
            row_mode: "bounded",
            disabled: true,
            plan: bounded_dense_prefix_execution_plan(),
        },
        Some("full-output" | "output-full") => DensePrefixProbeMode {
            row_mode: "full-output",
            disabled: false,
            plan: full_output_dense_prefix_execution_plan(),
        },
        Some("1" | "bounded" | "default") => DensePrefixProbeMode {
            row_mode: "bounded",
            disabled: false,
            plan: bounded_dense_prefix_execution_plan(),
        },
        _ => DensePrefixProbeMode {
            row_mode: "bounded",
            disabled: false,
            plan: bounded_dense_prefix_execution_plan(),
        },
    }
}

fn skipped_real_full_dense_prefix_probe(
    status: &'static str,
    mode: DensePrefixProbeMode,
    skipped_reason: Option<String>,
) -> RealFullDensePrefixProbe {
    let scope = if mode.row_mode == "full-output" {
        "execute full-output real GLM-5.2 BF16 dense MLP outputs sequentially into all residual dimensions for dense layers 0..2"
    } else {
        "execute bounded real GLM-5.2 BF16 dense MLP outputs sequentially into a residual prefix for dense layers 0..2"
    };
    RealFullDensePrefixProbe {
        status,
        scope,
        opt_in_env: REAL_FULL_PROBE_DENSE_PREFIX_ENV,
        row_mode: mode.row_mode,
        hidden_source: "not-run",
        dense_layers: GLM52_FIRST_K_DENSE_REPLACE,
        layers_executed: 0,
        intermediate_rows: mode.plan.intermediate_rows,
        output_rows: mode.plan.output_rows,
        residual_prefix_values: 0,
        residual_adds: 0,
        norm_tensors_read: 0,
        weight_tensors_read: 0,
        norm_bytes_read: 0,
        weight_bytes_read: 0,
        norm_checksum: None,
        activation_checksum: None,
        output_checksum: None,
        output_l2_norm: None,
        initial_residual_checksum: None,
        residual_delta_checksum: None,
        final_residual_checksum: None,
        first_layer_id: None,
        last_layer_id: None,
        layer_summaries: Vec::new(),
        uses_real_dense_weights: false,
        applies_dense_mlp_residual_prefix: false,
        includes_attention: false,
        covers_all_dense_layers: false,
        covers_full_output_rows: false,
        uses_full_model_residual: false,
        passed: false,
        skipped_reason,
    }
}

fn run_real_full_dense_prefix_probe(
    catalog: &TensorCatalog,
    mode: DensePrefixProbeMode,
) -> Result<RealFullDensePrefixProbe> {
    let execution = execute_dense_prefix_with_plan(catalog, mode.plan)?;
    let (status, scope, hidden_source) = if mode.row_mode == "full-output" {
        (
            "numeric-real-bf16-dense-prefix-full-output",
            "opt-in real GLM-5.2 BF16 post-attention RMSNorm plus dense MLP outputs sequentially into all residual dimensions for dense layers 0..2; attention is still omitted",
            "deterministic-hidden-shaped-f32-row-mutated-through-full-output-dense-mlp-chain",
        )
    } else {
        (
            "numeric-real-bf16-dense-prefix",
            "default bounded real GLM-5.2 BF16 post-attention RMSNorm plus dense MLP outputs sequentially into a residual prefix for dense layers 0..2; attention is still omitted",
            "deterministic-hidden-shaped-f32-row-mutated-through-dense-mlp-prefix",
        )
    };
    Ok(RealFullDensePrefixProbe {
        status,
        scope,
        opt_in_env: REAL_FULL_PROBE_DENSE_PREFIX_ENV,
        row_mode: mode.row_mode,
        hidden_source,
        dense_layers: GLM52_FIRST_K_DENSE_REPLACE,
        layers_executed: execution.layers_executed,
        intermediate_rows: execution.intermediate_rows,
        output_rows: execution.output_rows,
        residual_prefix_values: execution.output_rows,
        residual_adds: execution.residual_adds,
        norm_tensors_read: execution.layers_executed,
        weight_tensors_read: execution.layers_executed * 3,
        norm_bytes_read: execution.norm_bytes_read,
        weight_bytes_read: execution.weight_bytes_read,
        norm_checksum: Some(execution.norm_checksum),
        activation_checksum: Some(execution.activation_checksum),
        output_checksum: Some(execution.output_checksum),
        output_l2_norm: Some(execution.output_l2_norm),
        initial_residual_checksum: Some(execution.initial_residual_checksum),
        residual_delta_checksum: Some(execution.residual_delta_checksum),
        final_residual_checksum: Some(execution.final_residual_checksum),
        first_layer_id: execution.first_layer_id,
        last_layer_id: execution.last_layer_id,
        layer_summaries: execution.layer_summaries,
        uses_real_dense_weights: true,
        applies_dense_mlp_residual_prefix: true,
        includes_attention: false,
        covers_all_dense_layers: execution.covers_all_dense_layers,
        covers_full_output_rows: execution.covers_full_output_rows,
        uses_full_model_residual: false,
        passed: execution.passed,
        skipped_reason: None,
    })
}

#[cfg(test)]
mod tests {
    use std::{fs::File, path::PathBuf};

    use glmrt_core::{TensorCatalog, GLM52_FIRST_K_DENSE_REPLACE, GLM52_HIDDEN_SIZE};

    use super::{dense_prefix_probe_mode, run_real_full_dense_prefix_probe};
    use crate::commands::real_full::coordinator_kernels::coordinator_cuda_reference_kernels_enabled;

    #[test]
    fn dense_prefix_probe_mode_parses_bounded_and_full_output() {
        let default_mode = dense_prefix_probe_mode(None);
        assert_eq!(default_mode.row_mode, "bounded");
        assert!(!default_mode.disabled);
        assert_eq!(default_mode.plan.output_rows, 4);

        let bounded_mode = dense_prefix_probe_mode(Some("1"));
        assert_eq!(bounded_mode.row_mode, "bounded");
        assert!(!bounded_mode.disabled);
        assert_eq!(bounded_mode.plan.output_rows, 4);

        for value in ["full-output", "output-full"] {
            let full_output_mode = dense_prefix_probe_mode(Some(value));
            assert_eq!(full_output_mode.row_mode, "full-output");
            assert!(!full_output_mode.disabled);
            assert_eq!(full_output_mode.plan.output_rows, GLM52_HIDDEN_SIZE);
        }

        let disabled_mode = dense_prefix_probe_mode(Some("0"));
        assert_eq!(disabled_mode.row_mode, "bounded");
        assert!(disabled_mode.disabled);
    }

    #[test]
    fn real_checkpoint_dense_prefix_full_output_probe_when_available() {
        let Some(catalog) = load_real_catalog_or_skip() else {
            return;
        };

        let probe = run_real_full_dense_prefix_probe(
            &catalog,
            dense_prefix_probe_mode(Some("full-output")),
        )
        .expect("running full-output real dense-prefix probe");

        println!("{}", serde_json::to_string_pretty(&probe).unwrap());
        assert_eq!(probe.status, "numeric-real-bf16-dense-prefix-full-output");
        assert_eq!(probe.row_mode, "full-output");
        assert_eq!(probe.dense_layers, GLM52_FIRST_K_DENSE_REPLACE);
        assert_eq!(probe.layers_executed, GLM52_FIRST_K_DENSE_REPLACE);
        assert_eq!(probe.intermediate_rows, 8);
        assert_eq!(probe.output_rows, GLM52_HIDDEN_SIZE);
        assert_eq!(probe.residual_prefix_values, GLM52_HIDDEN_SIZE);
        assert_eq!(probe.residual_adds, GLM52_FIRST_K_DENSE_REPLACE);
        if coordinator_cuda_reference_kernels_enabled() {
            assert!(probe.layer_summaries.iter().all(|layer| {
                layer.norm_backend.starts_with("cuda-reference-")
                    && layer.linear_backend.starts_with("cuda-reference-")
                    && layer.mlp_backend.starts_with("cuda-reference-")
            }));
        } else {
            assert!(probe.norm_bytes_read > 0);
            assert!(probe.weight_bytes_read > 0);
        }
        assert_eq!(probe.first_layer_id, Some(0));
        assert_eq!(probe.last_layer_id, Some(GLM52_FIRST_K_DENSE_REPLACE - 1));
        assert!(probe.uses_real_dense_weights);
        assert!(probe.applies_dense_mlp_residual_prefix);
        assert!(!probe.includes_attention);
        assert!(probe.covers_all_dense_layers);
        assert!(probe.covers_full_output_rows);
        assert!(!probe.uses_full_model_residual);
        assert!(probe.final_residual_checksum.unwrap().is_finite());
        assert!(probe.passed);
    }

    fn load_real_catalog_or_skip() -> Option<TensorCatalog> {
        if !crate::commands::real_full::tests::fixture::real_checkpoint_tests_enabled() {
            crate::commands::real_full::tests::fixture::real_checkpoint_tests_skip_message(
                "dense prefix",
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
