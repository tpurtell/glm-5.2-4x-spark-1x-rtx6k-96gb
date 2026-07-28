use crate::{
    RealSliceDenseMlpProbe, RealSlicePrefillDenseMlpProbe, RealSlicePrefillDenseMlpResidualProbe,
};

pub(super) fn dense_mlp_probe() -> RealSliceDenseMlpProbe {
    RealSliceDenseMlpProbe {
        layer_id: 0,
        intermediate_count: 4,
        output_count: 2,
        post_attention_layernorm_bytes: 4,
        gate_proj_rows_bytes: 16,
        up_proj_rows_bytes: 16,
        down_proj_rows_bytes: 16,
        post_attention_layernorm_sha256: "post_norm".to_owned(),
        gate_proj_rows_sha256: "dense_gate".to_owned(),
        up_proj_rows_sha256: "dense_up".to_owned(),
        down_proj_rows_sha256: "dense_down".to_owned(),
        norm_l2: 1.25,
        activation_checksum: 0.5,
        output_checksum: 3.25,
        output_l2_norm: 2.5,
        first_output: 1.25,
        last_output: 2.0,
    }
}

pub(super) fn prefill_dense_mlp_probe() -> RealSlicePrefillDenseMlpProbe {
    RealSlicePrefillDenseMlpProbe {
        layer_id: 0,
        row_count: 2,
        intermediate_count: 4,
        output_count: 2,
        residual_source: "embedding_row_standin".to_owned(),
        norm_checksum: 1.5,
        norm_l2_norm: 2.25,
        activation_checksum: 0.75,
        output_checksum: 4.5,
        output_l2_norm: 3.0,
        first_output: 1.0,
        last_output: 2.0,
        outputs: vec![vec![1.0, 0.5], vec![1.0, 2.0]],
    }
}

pub(super) fn prefill_dense_mlp_residual_probe() -> RealSlicePrefillDenseMlpResidualProbe {
    RealSlicePrefillDenseMlpResidualProbe {
        layer_id: 0,
        row_count: 2,
        output_count: 2,
        residual_source: "embedding_row_standin".to_owned(),
        residual_checksum: 1.25,
        branch_checksum: 4.5,
        output_checksum: 5.75,
        output_l2_norm: 4.0,
        first_output: 1.25,
        last_output: 2.25,
    }
}
