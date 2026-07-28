use anyhow::Result;
use glmrt_core::{
    TensorCatalog, GLM52_FIRST_K_DENSE_REPLACE, GLM52_HIDDEN_BF16_BYTES, GLM52_HIDDEN_SIZE,
    GLM52_NUM_HIDDEN_LAYERS, GLM52_TOP_K,
};

use super::attention::{
    real_full_attention_residual_prefix_probe, real_full_dsa_indexer_attention_probe,
    real_full_mla_rope_attention_probe,
};
use super::constants::{
    REAL_FULL_PREFLIGHT_DECODE_ROWS, REAL_FULL_PREFLIGHT_MTP_ROWS, REAL_FULL_PREFLIGHT_PREFILL_ROWS,
};
use super::dense::real_full_dense_prefix_probe;
use super::types::{
    RealFullAttentionDenseSparsePrefixProbe, RealFullExecutionPlan, RealFullResidualKernelSelfTest,
    RealFullResidualLayerDryRun, RealFullResidualStreamDryRun,
};
use layer_ordered::{
    real_full_layer_ordered_execution_probe, real_full_layer_ordered_prefix_probe,
};

mod layer_ordered;

pub(super) fn real_full_residual_stream_dry_run(
    execution_plan: &RealFullExecutionPlan,
    catalog: &TensorCatalog,
) -> RealFullResidualStreamDryRun {
    let row_count = REAL_FULL_PREFLIGHT_PREFILL_ROWS
        + REAL_FULL_PREFLIGHT_MTP_ROWS
        + REAL_FULL_PREFLIGHT_DECODE_ROWS;
    let residual_state_bytes = row_count * GLM52_HIDDEN_BF16_BYTES;
    let mut state_hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut layer_dry_runs = Vec::with_capacity(execution_plan.layers.len());
    let mut attention_residual_adds = 0_usize;
    let mut mlp_residual_adds = 0_usize;
    let mut dense_layers = 0_usize;
    let mut sparse_layers = 0_usize;
    let mut remote_sparse_layers = 0_usize;

    for (expected_layer_id, layer) in execution_plan.layers.iter().enumerate() {
        let is_sparse = layer.mlp.routed_nvfp4_expert_exchange;
        dense_layers += usize::from(!is_sparse);
        sparse_layers += usize::from(is_sparse);
        remote_sparse_layers += usize::from(is_sparse);
        attention_residual_adds += usize::from(layer.attention.attention_output_projection);
        mlp_residual_adds += layer.residual_adds.saturating_sub(1);

        state_hash = mix_residual_state(state_hash, layer.layer_id, row_count, false);
        state_hash = mix_residual_state(state_hash, layer.layer_id, row_count, true);
        layer_dry_runs.push(RealFullResidualLayerDryRun {
            layer_id: layer.layer_id,
            layer_kind: layer.layer_kind,
            input_rows: row_count,
            residual_state_bytes,
            attention_input_norm: layer.attention.input_norm,
            attention_output_projection: layer.attention.attention_output_projection,
            attention_residual_boundary: layer.attention.attention_output_projection,
            post_attention_norm: layer.mlp.post_attention_norm,
            mlp_kind: if is_sparse {
                "sparse-routed-moe"
            } else {
                "dense-mlp"
            },
            routed_expert_exchange: is_sparse,
            mlp_residual_boundary: layer.residual_adds > 1,
            output_rows: row_count,
            residual_state_hash: state_hash,
            layer_order_verified: layer.layer_id == expected_layer_id,
        });
    }

    let total_residual_adds = attention_residual_adds + mlp_residual_adds;
    let numeric_kernel_self_test = residual_accumulator_self_test();
    let real_dense_prefix_probe = real_full_dense_prefix_probe(catalog);
    let real_attention_residual_prefix_probe = real_full_attention_residual_prefix_probe(catalog);
    let real_mla_rope_attention_probe = real_full_mla_rope_attention_probe(catalog);
    let real_dsa_indexer_attention_probe = real_full_dsa_indexer_attention_probe(catalog);
    let real_attention_dense_sparse_prefix_probe = superseded_attention_dense_sparse_prefix_probe();
    let real_layer_ordered_prefix_probe = real_full_layer_ordered_prefix_probe(catalog);
    let real_layer_ordered_execution_probe = real_full_layer_ordered_execution_probe(catalog);
    RealFullResidualStreamDryRun {
        status: if real_layer_ordered_execution_probe.passed {
            "dry-run-plus-real-layer-ordered-bounded-all-stage-execution-trace"
        } else if real_layer_ordered_prefix_probe.passed {
            "dry-run-plus-real-layer-ordered-prefix-probe"
        } else if real_dense_prefix_probe.passed && real_attention_residual_prefix_probe.passed {
            "dry-run-plus-real-attention-and-dense-prefix-probes"
        } else if real_dense_prefix_probe.passed {
            "dry-run-plus-real-dense-prefix-probe"
        } else if real_attention_residual_prefix_probe.passed {
            "dry-run-plus-real-attention-prefix-probe"
        } else {
            "dry-run-only"
        },
        scope: "walk full 78-layer residual-stream ordering, residual add boundaries, a numeric residual accumulator self-test, bounded real BF16 attention/dense residual-prefix probes, a bounded real layer-0 attention-then-MLP residual-order probe, and a layer-ordered bounded all-stage numeric residual trace that still reports full-output/full-model residual completion separately",
        layer_count: execution_plan.layer_count,
        row_count,
        hidden_dim: GLM52_HIDDEN_SIZE,
        hidden_bytes_per_row: GLM52_HIDDEN_BF16_BYTES,
        residual_state_bytes,
        dense_layers,
        sparse_layers,
        remote_sparse_layers,
        attention_residual_adds,
        mlp_residual_adds,
        total_residual_adds,
        terminal_rows: row_count,
        terminal_stages: execution_plan.terminal_stages.clone(),
        final_residual_state_hash: state_hash,
        numeric_kernel_self_test,
        real_dense_prefix_probe,
        real_attention_residual_prefix_probe,
        real_mla_rope_attention_probe,
        real_dsa_indexer_attention_probe,
        real_attention_dense_sparse_prefix_probe,
        real_layer_ordered_prefix_probe,
        real_layer_ordered_execution_probe,
        layer_order_verified: layer_dry_runs.len() == execution_plan.layer_count
            && layer_dry_runs.iter().all(|layer| layer.layer_order_verified)
            && dense_layers == GLM52_FIRST_K_DENSE_REPLACE,
        layer_dry_runs,
    }
}

fn superseded_attention_dense_sparse_prefix_probe() -> RealFullAttentionDenseSparsePrefixProbe {
    let sparse_layers = GLM52_NUM_HIDDEN_LAYERS - GLM52_FIRST_K_DENSE_REPLACE;
    RealFullAttentionDenseSparsePrefixProbe {
        status: "not-run",
        scope: "legacy composed attention+dense+sparse residual-chain probe superseded by the layer-ordered execution stepper",
        opt_in_env: "not-applicable",
        row_mode: "superseded",
        hidden_source: "not-run",
        attention_layer: 0,
        attention_rows: 0,
        attention_residual_adds: 0,
        dense_layers: GLM52_FIRST_K_DENSE_REPLACE,
        dense_residual_adds: 0,
        sparse_layers,
        sparse_residual_adds: 0,
        shared_expert_layers: 0,
        total_residual_adds: 0,
        sparse_output_rows: 0,
        residual_prefix_values: 0,
        routed_top_k_per_layer: GLM52_TOP_K,
        routed_routes: 0,
        input_norm_bytes_read: 0,
        attention_projection_bytes_read: 0,
        attention_o_proj_bytes_read: 0,
        dense_norm_bytes_read: 0,
        dense_weight_bytes_read: 0,
        sparse_router_weight_bytes_read: 0,
        sparse_router_bias_bytes_read: 0,
        sparse_routed_weight_bytes_read: 0,
        sparse_routed_quant_metadata_bytes_read: 0,
        sparse_shared_weight_bytes_read: 0,
        initial_residual_checksum: None,
        attention_residual_checksum: None,
        dense_residual_checksum: None,
        sparse_residual_checksum: None,
        residual_delta_checksum: None,
        includes_attention: false,
        includes_causal_softmax: false,
        includes_mla_softmax: false,
        includes_dense_layers: false,
        includes_sparse_mlp: false,
        includes_shared_expert: false,
        covers_all_dense_layers: false,
        covers_all_sparse_layers: false,
        covers_full_top_k: false,
        covers_full_output_rows: false,
        uses_full_model_residual: false,
        passed: false,
        skipped_reason: Some(
            "deleted superseded composed probe; use real_layer_ordered_execution_probe for scheduler-aligned residual evidence"
                .to_owned(),
        ),
    }
}

fn mix_residual_state(previous: u64, layer_id: usize, rows: usize, mlp_boundary: bool) -> u64 {
    let stage = if mlp_boundary { 0x9e37_u64 } else { 0x85eb_u64 };
    previous
        .wrapping_mul(0x0000_0100_0000_01b3)
        .wrapping_add(layer_id as u64)
        .wrapping_mul(0x0000_0100_0000_01b3)
        .wrapping_add(rows as u64)
        ^ stage
}

fn residual_accumulator_self_test() -> RealFullResidualKernelSelfTest {
    let layers = 3_usize;
    let rows = 2_usize;
    let hidden_dim = 4_usize;
    let mut residual = vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0];
    let mut residual_adds = 0_usize;

    for layer_idx in 0..layers {
        let attention_delta = vec![0.5 * (layer_idx + 1) as f32; residual.len()];
        let mlp_delta = vec![-0.25 * (layer_idx + 1) as f32; residual.len()];
        apply_residual_delta(&mut residual, &attention_delta)
            .expect("residual self-test attention delta shape is valid");
        residual_adds += 1;
        apply_residual_delta(&mut residual, &mlp_delta)
            .expect("residual self-test MLP delta shape is valid");
        residual_adds += 1;
    }

    let final_checksum = residual.iter().sum::<f32>();
    let expected_checksum = 12.0_f32;
    let first_value = residual.first().copied().unwrap_or_default();
    let last_value = residual.last().copied().unwrap_or_default();
    let expected_first_value = 2.5_f32;
    let expected_last_value = -2.5_f32;
    let values_updated = residual.len() * residual_adds;
    let passed = approx_eq(final_checksum, expected_checksum)
        && approx_eq(first_value, expected_first_value)
        && approx_eq(last_value, expected_last_value)
        && residual_adds == layers * 2
        && values_updated == layers * rows * hidden_dim * 2;

    RealFullResidualKernelSelfTest {
        status: "numeric-self-test",
        scope: "apply numeric attention and MLP residual deltas in-place across a small multi-layer residual stream",
        layers,
        rows,
        hidden_dim,
        residual_adds,
        values_updated,
        final_checksum,
        expected_checksum,
        first_value,
        expected_first_value,
        last_value,
        expected_last_value,
        passed,
    }
}

fn apply_residual_delta(residual: &mut [f32], delta: &[f32]) -> Result<()> {
    if residual.len() != delta.len() {
        anyhow::bail!(
            "residual delta length mismatch: residual={} delta={}",
            residual.len(),
            delta.len()
        );
    }
    for (target, delta) in residual.iter_mut().zip(delta.iter()) {
        *target += delta;
    }
    Ok(())
}

fn approx_eq(left: f32, right: f32) -> bool {
    (left - right).abs() < 1.0e-6
}

#[cfg(test)]
mod tests {
    use super::{apply_residual_delta, residual_accumulator_self_test};

    #[test]
    fn residual_accumulator_applies_deltas_in_place() {
        let mut residual = vec![1.0, -2.0, 3.5];
        apply_residual_delta(&mut residual, &[0.5, 1.0, -1.5]).unwrap();

        assert_eq!(residual, vec![1.5, -1.0, 2.0]);
    }

    #[test]
    fn residual_accumulator_rejects_shape_mismatch() {
        let mut residual = vec![1.0, 2.0];
        let err = apply_residual_delta(&mut residual, &[1.0]).unwrap_err();

        assert!(err.to_string().contains("residual delta length mismatch"));
    }

    #[test]
    fn residual_accumulator_self_test_passes() {
        let self_test = residual_accumulator_self_test();

        assert!(self_test.passed);
        assert_eq!(self_test.layers, 3);
        assert_eq!(self_test.residual_adds, 6);
        assert_eq!(self_test.values_updated, 48);
        assert!((self_test.final_checksum - 12.0).abs() < 1.0e-6);
    }
}
