use anyhow::Result;
use glmrt_core::{TensorCatalog, GLM52_HIDDEN_SIZE};

use super::super::attention::real_full_attention_residual_prefix_rows;
use super::super::dense::real_full_dense_layer_prefix_hidden_from_initial;
use super::super::types::RealFullLayerOrderedResidualPrefixProbe;

mod execution_stepper;
mod execution_trace;
mod oracle_fixture;
mod scheduler_rows;
pub(super) use execution_trace::real_full_layer_ordered_execution_probe;

pub(super) fn real_full_layer_ordered_prefix_probe(
    catalog: &TensorCatalog,
) -> RealFullLayerOrderedResidualPrefixProbe {
    match run_real_full_layer_ordered_prefix_probe(catalog) {
        Ok(probe) => probe,
        Err(error) => {
            skipped_real_full_layer_ordered_prefix_probe("error", Some(error.to_string()))
        }
    }
}

fn skipped_real_full_layer_ordered_prefix_probe(
    status: &'static str,
    skipped_reason: Option<String>,
) -> RealFullLayerOrderedResidualPrefixProbe {
    RealFullLayerOrderedResidualPrefixProbe {
        status,
        scope: "execute real GLM-5.2 layer-0 attention residual before layer-0 dense MLP residual for a bounded residual prefix",
        row_mode: "bounded",
        hidden_source: "not-run",
        layer_id: 0,
        attention_rows: 0,
        attention_residual_adds: 0,
        mlp_residual_adds: 0,
        total_residual_adds: 0,
        dense_rows: 0,
        dense_intermediate_rows: 0,
        dense_output_rows: 0,
        residual_prefix_values: 0,
        input_norm_bytes_read: 0,
        attention_projection_bytes_read: 0,
        attention_o_proj_bytes_read: 0,
        dense_norm_bytes_read: 0,
        dense_weight_bytes_read: 0,
        initial_residual_checksum: None,
        attention_residual_checksum: None,
        dense_residual_checksum: None,
        residual_delta_checksum: None,
        includes_attention: false,
        includes_causal_softmax: false,
        includes_mla_softmax: false,
        includes_dense_mlp: false,
        includes_sparse_mlp: false,
        carries_attention_residual_into_mlp: false,
        layer_order_verified: false,
        uses_full_model_residual: false,
        passed: false,
        skipped_reason,
    }
}

fn run_real_full_layer_ordered_prefix_probe(
    catalog: &TensorCatalog,
) -> Result<RealFullLayerOrderedResidualPrefixProbe> {
    let attention = real_full_attention_residual_prefix_rows(catalog)?;
    let attention_required_weight_evidence = attention.required_weight_evidence();
    let mut dense_rows = Vec::with_capacity(attention.hidden_rows.len());
    for hidden in attention.hidden_rows {
        dense_rows.push(real_full_dense_layer_prefix_hidden_from_initial(
            catalog,
            attention.layer_id,
            hidden,
        )?);
    }
    let dense_row_count = dense_rows.len();
    let dense_initial_checksum = dense_rows
        .iter()
        .map(|dense| dense.initial_residual_checksum)
        .sum::<f64>();
    let dense_residual_checksum = dense_rows
        .iter()
        .map(|dense| dense.final_residual_checksum)
        .sum::<f64>();
    let dense_delta_checksum = dense_rows
        .iter()
        .map(|dense| dense.residual_delta_checksum)
        .sum::<f64>();
    let dense_norm_bytes_read = dense_rows
        .iter()
        .map(|dense| dense.norm_bytes_read)
        .sum::<u64>();
    let dense_weight_bytes_read = dense_rows
        .iter()
        .map(|dense| dense.weight_bytes_read)
        .sum::<u64>();
    let mlp_residual_adds = dense_rows
        .iter()
        .map(|dense| dense.residual_adds)
        .sum::<usize>();
    let dense_intermediate_rows = dense_rows
        .first()
        .map(|dense| dense.intermediate_rows)
        .unwrap_or_default();
    let dense_output_rows = dense_rows
        .first()
        .map(|dense| dense.output_rows)
        .unwrap_or_default();
    let carries_attention_residual_into_mlp =
        approx_eq_f64(dense_initial_checksum, attention.final_residual_checksum);
    let total_residual_adds = attention.residual_adds + mlp_residual_adds;
    let residual_delta_checksum = attention.residual_delta_checksum + dense_delta_checksum;
    let layer_order_verified = attention.layer_id == 0
        && dense_rows
            .iter()
            .all(|dense| dense.layer_id == attention.layer_id)
        && carries_attention_residual_into_mlp
        && dense_row_count == attention.attention_rows
        && attention.residual_prefix_values == dense_output_rows * dense_row_count
        && total_residual_adds == attention.attention_rows * 2;
    let passed = layer_order_verified
        && attention.includes_causal_softmax
        && dense_rows.iter().all(|dense| dense.passed)
        && attention_required_weight_evidence
        && dense_rows
            .iter()
            .all(|dense| dense.hidden.len() == GLM52_HIDDEN_SIZE)
        && dense_rows.iter().all(|dense| {
            dense.norm_checksum.is_finite()
                && dense.activation_checksum.is_finite()
                && dense.output_checksum.is_finite()
                && dense.output_l2_norm.is_finite()
                && dense.first_residual_after.is_finite()
                && dense.last_residual_after.is_finite()
        })
        && attention.initial_residual_checksum.is_finite()
        && attention.final_residual_checksum.is_finite()
        && dense_residual_checksum.is_finite()
        && residual_delta_checksum.is_finite();

    Ok(RealFullLayerOrderedResidualPrefixProbe {
        status: "numeric-real-layer0-attention-dense-residual-prefix",
        scope: "default bounded real GLM-5.2 layer-0 residual execution in model order: BF16 causal attention residual first, then BF16 dense MLP residual from the post-attention hidden prefix; later layers, sparse MLP, full MLA/RoPE attention, and full-model residuals are still omitted",
        row_mode: "bounded",
        hidden_source: "deterministic-layer0-attention-hidden-carried-into-layer0-dense-mlp",
        layer_id: attention.layer_id,
        attention_rows: attention.attention_rows,
        attention_residual_adds: attention.residual_adds,
        mlp_residual_adds,
        total_residual_adds,
        dense_rows: dense_row_count,
        dense_intermediate_rows,
        dense_output_rows,
        residual_prefix_values: dense_output_rows * dense_row_count,
        input_norm_bytes_read: attention.input_norm_bytes_read,
        attention_projection_bytes_read: attention.projection_bytes_read,
        attention_o_proj_bytes_read: attention.o_proj_bytes_read,
        dense_norm_bytes_read,
        dense_weight_bytes_read,
        initial_residual_checksum: Some(attention.initial_residual_checksum),
        attention_residual_checksum: Some(attention.final_residual_checksum),
        dense_residual_checksum: Some(dense_residual_checksum),
        residual_delta_checksum: Some(residual_delta_checksum),
        includes_attention: true,
        includes_causal_softmax: attention.includes_causal_softmax,
        includes_mla_softmax: attention.includes_mla_softmax,
        includes_dense_mlp: true,
        includes_sparse_mlp: false,
        carries_attention_residual_into_mlp,
        layer_order_verified,
        uses_full_model_residual: false,
        passed,
        skipped_reason: None,
    })
}

fn approx_eq_f64(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1.0e-9
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::path::{Path, PathBuf};

    use crate::commands::real_full::coordinator_kernels::coordinator_cuda_reference_kernels_enabled;
    use glmrt_core::TensorCatalog;

    use super::run_real_full_layer_ordered_prefix_probe;

    #[test]
    fn real_checkpoint_layer_ordered_prefix_probe_when_available() {
        let Some(catalog) = load_real_catalog_or_skip() else {
            return;
        };

        let probe = run_real_full_layer_ordered_prefix_probe(&catalog)
            .expect("running real layer-ordered residual prefix probe");

        println!("{}", serde_json::to_string_pretty(&probe).unwrap());
        assert_eq!(
            probe.status,
            "numeric-real-layer0-attention-dense-residual-prefix"
        );
        assert_eq!(probe.row_mode, "bounded");
        assert_eq!(probe.layer_id, 0);
        assert_eq!(probe.attention_residual_adds, 2);
        assert_eq!(probe.mlp_residual_adds, 2);
        assert_eq!(probe.total_residual_adds, 4);
        assert_eq!(probe.dense_rows, 2);
        assert_eq!(probe.dense_intermediate_rows, 8);
        assert_eq!(probe.dense_output_rows, 4);
        assert_eq!(probe.residual_prefix_values, 8);
        if !coordinator_cuda_reference_kernels_enabled() {
            assert!(probe.input_norm_bytes_read > 0);
            assert!(probe.attention_projection_bytes_read > 0);
            assert!(probe.attention_o_proj_bytes_read > 0);
            assert_eq!(probe.dense_norm_bytes_read, 24_576);
        }
        assert_eq!(probe.dense_weight_bytes_read, 589_824);
        assert!(probe.includes_attention);
        assert!(probe.includes_causal_softmax);
        assert!(!probe.includes_mla_softmax);
        assert!(probe.includes_dense_mlp);
        assert!(!probe.includes_sparse_mlp);
        assert!(probe.carries_attention_residual_into_mlp);
        assert!(probe.layer_order_verified);
        assert!(!probe.uses_full_model_residual);
        assert!(probe.initial_residual_checksum.unwrap().is_finite());
        assert!(probe.attention_residual_checksum.unwrap().is_finite());
        assert!(probe.dense_residual_checksum.unwrap().is_finite());
        assert!(probe.residual_delta_checksum.unwrap().is_finite());
        assert!(probe.passed);
    }

    fn load_real_catalog_or_skip() -> Option<TensorCatalog> {
        if !crate::commands::real_full::tests::fixture::real_checkpoint_tests_enabled() {
            crate::commands::real_full::tests::fixture::real_checkpoint_tests_skip_message(
                "layer-ordered prefix",
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
        if !Path::new(&catalog.snapshot_path).exists() {
            eprintln!("skipped: missing snapshot {}", catalog.snapshot_path);
            return None;
        }
        Some(catalog)
    }

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
    }
}
