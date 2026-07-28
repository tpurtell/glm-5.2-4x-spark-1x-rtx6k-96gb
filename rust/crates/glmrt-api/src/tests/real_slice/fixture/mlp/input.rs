use crate::{
    RealSliceMlpInputNormProbe, RealSliceMlpResidualProbe, RealSliceMoeBranchProbe,
    RealSlicePrefillMlpInputMoeProbe, RealSliceRoute, RealSliceRoutedExpertProbe,
    RealSliceRouterProbe, RealSliceSharedExpertProbe,
};

pub(crate) fn mlp_input_norm_probe() -> RealSliceMlpInputNormProbe {
    RealSliceMlpInputNormProbe {
        layer_id: 3,
        residual_source: "embedding_row_residual_standin".to_owned(),
        hidden_width: 2,
        post_attention_layernorm_bytes: 4,
        post_attention_layernorm_sha256: "mlp_norm".to_owned(),
        normalized_checksum: 0.75,
        normalized_l2_norm: 1.25,
        first_normalized: 0.5,
        last_normalized: 0.25,
        normalized_hidden: vec![0.5, 0.25],
    }
}

pub(crate) fn mlp_input_router_probe() -> RealSliceRouterProbe {
    RealSliceRouterProbe {
        layer_id: 3,
        top_k: 2,
        router_weight_bytes: 16,
        router_bias_bytes: 8,
        router_weight_sha256: "mlp_router_weight".to_owned(),
        router_bias_sha256: "mlp_router_bias".to_owned(),
        routes: vec![
            RealSliceRoute {
                expert_id: 11,
                owner: "ostrich".to_owned(),
                score: 0.9,
                corrected_score: 1.0,
                normalized_weight: 0.55,
            },
            RealSliceRoute {
                expert_id: 13,
                owner: "kiwi".to_owned(),
                score: 0.8,
                corrected_score: 0.95,
                normalized_weight: 0.45,
            },
        ],
    }
}

pub(crate) fn mlp_input_routed_expert_probe() -> RealSliceRoutedExpertProbe {
    RealSliceRoutedExpertProbe {
        layer_id: 3,
        expert_id: 11,
        owner: "ostrich".to_owned(),
        quant_recipe: "nvfp4-e2m1-f8e4m3".to_owned(),
        intermediate_count: 4,
        output_count: 2,
        gate_proj_rows_bytes: 16,
        gate_proj_scale_rows_bytes: 4,
        gate_proj_scalar_bytes: 8,
        up_proj_rows_bytes: 16,
        up_proj_scale_rows_bytes: 4,
        up_proj_scalar_bytes: 8,
        down_proj_rows_bytes: 16,
        down_proj_scale_rows_bytes: 4,
        down_proj_scalar_bytes: 8,
        gate_proj_rows_sha256: "mlp_routed_gate".to_owned(),
        gate_proj_scale_rows_sha256: "mlp_routed_gate_scale".to_owned(),
        up_proj_rows_sha256: "mlp_routed_up".to_owned(),
        up_proj_scale_rows_sha256: "mlp_routed_up_scale".to_owned(),
        down_proj_rows_sha256: "mlp_routed_down".to_owned(),
        down_proj_scale_rows_sha256: "mlp_routed_down_scale".to_owned(),
        activation_checksum: 2.5,
        output_checksum: 4.5,
        output_l2_norm: 3.25,
        first_output: 2.0,
        last_output: 2.5,
        reduction_route_count: 2,
        reduction_routes: vec!["11@ostrich".to_owned(), "13@kiwi".to_owned()],
        reduction_output_checksum: 4.25,
        reduction_output_l2_norm: 3.0,
        reduction_first_output: 2.0,
        reduction_last_output: 2.25,
        reduction_outputs: vec![2.0, 2.25],
    }
}

pub(crate) fn mlp_input_shared_expert_probe() -> RealSliceSharedExpertProbe {
    RealSliceSharedExpertProbe {
        layer_id: 3,
        intermediate_count: 4,
        output_count: 2,
        gate_proj_rows_bytes: 16,
        up_proj_rows_bytes: 16,
        down_proj_rows_bytes: 16,
        gate_proj_rows_sha256: "mlp_shared_gate".to_owned(),
        up_proj_rows_sha256: "mlp_shared_up".to_owned(),
        down_proj_rows_sha256: "mlp_shared_down".to_owned(),
        activation_checksum: 0.75,
        output_checksum: -1.25,
        output_l2_norm: 0.9,
        first_output: -0.5,
        last_output: -0.75,
        outputs: vec![-0.5, -0.75],
    }
}

pub(crate) fn mlp_input_moe_branch_probe() -> RealSliceMoeBranchProbe {
    RealSliceMoeBranchProbe {
        layer_id: 3,
        output_count: 2,
        routed_route_count: 2,
        routed_output_checksum: 4.25,
        shared_output_checksum: -1.25,
        output_checksum: 3.0,
        output_l2_norm: 2.1213202,
        first_output: 1.5,
        last_output: 1.5,
        outputs: vec![1.5, 1.5],
    }
}

pub(crate) fn mlp_input_residual_probe() -> RealSliceMlpResidualProbe {
    RealSliceMlpResidualProbe {
        layer_id: 3,
        output_count: 2,
        residual_source: "embedding_row_residual_standin".to_owned(),
        residual_checksum: -0.25,
        branch_checksum: 3.0,
        output_checksum: 2.75,
        output_l2_norm: 2.0155644,
        first_output: 1.75,
        last_output: 1.0,
    }
}

pub(crate) fn prefill_mlp_input_moe_probe() -> RealSlicePrefillMlpInputMoeProbe {
    RealSlicePrefillMlpInputMoeProbe {
        layer_id: 3,
        row_count: 2,
        output_count: 2,
        route_count: 3,
        residual_source: "embedding_row_residual_standin".to_owned(),
        routed_output_checksum: 4.5,
        shared_output_checksum: -1.0,
        branch_checksum: 3.5,
        residual_checksum: 1.25,
        output_checksum: 4.75,
        output_l2_norm: 3.5,
        first_output: 2.0,
        last_output: 0.75,
        outputs: vec![vec![2.0, 1.0], vec![1.0, 0.75]],
    }
}
