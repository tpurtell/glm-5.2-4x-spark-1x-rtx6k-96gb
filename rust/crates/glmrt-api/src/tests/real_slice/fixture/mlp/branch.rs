use crate::{
    RealSliceMlpResidualProbe, RealSliceMoeBranchProbe, RealSliceRoute, RealSliceRoutedExpertProbe,
    RealSliceRouterProbe, RealSliceSharedExpertProbe,
};

pub(crate) fn router_probe() -> RealSliceRouterProbe {
    RealSliceRouterProbe {
        layer_id: 3,
        top_k: 2,
        router_weight_bytes: 16,
        router_bias_bytes: 8,
        router_weight_sha256: "router_weight".to_owned(),
        router_bias_sha256: "router_bias".to_owned(),
        routes: vec![
            RealSliceRoute {
                expert_id: 7,
                owner: "kiwi".to_owned(),
                score: 0.8,
                corrected_score: 0.9,
                normalized_weight: 0.6,
            },
            RealSliceRoute {
                expert_id: 9,
                owner: "dodo".to_owned(),
                score: 0.6,
                corrected_score: 0.7,
                normalized_weight: 0.4,
            },
        ],
    }
}

pub(crate) fn routed_expert_probe() -> RealSliceRoutedExpertProbe {
    RealSliceRoutedExpertProbe {
        layer_id: 3,
        expert_id: 7,
        owner: "kiwi".to_owned(),
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
        gate_proj_rows_sha256: "routed_gate".to_owned(),
        gate_proj_scale_rows_sha256: "routed_gate_scale".to_owned(),
        up_proj_rows_sha256: "routed_up".to_owned(),
        up_proj_scale_rows_sha256: "routed_up_scale".to_owned(),
        down_proj_rows_sha256: "routed_down".to_owned(),
        down_proj_scale_rows_sha256: "routed_down_scale".to_owned(),
        activation_checksum: 1.5,
        output_checksum: 2.25,
        output_l2_norm: 1.75,
        first_output: 0.5,
        last_output: 1.75,
        reduction_route_count: 2,
        reduction_routes: vec!["7@kiwi".to_owned(), "9@dodo".to_owned()],
        reduction_output_checksum: 3.0,
        reduction_output_l2_norm: 2.0,
        reduction_first_output: 1.0,
        reduction_last_output: 2.0,
        reduction_outputs: vec![1.0, 2.0],
    }
}

pub(crate) fn shared_expert_probe() -> RealSliceSharedExpertProbe {
    RealSliceSharedExpertProbe {
        layer_id: 3,
        intermediate_count: 4,
        output_count: 2,
        gate_proj_rows_bytes: 16,
        up_proj_rows_bytes: 16,
        down_proj_rows_bytes: 16,
        gate_proj_rows_sha256: "shared_gate".to_owned(),
        up_proj_rows_sha256: "shared_up".to_owned(),
        down_proj_rows_sha256: "shared_down".to_owned(),
        activation_checksum: 1.25,
        output_checksum: -0.75,
        output_l2_norm: 0.8,
        first_output: 0.25,
        last_output: -1.0,
        outputs: vec![0.25, -1.0],
    }
}

pub(crate) fn moe_branch_probe() -> RealSliceMoeBranchProbe {
    RealSliceMoeBranchProbe {
        layer_id: 3,
        output_count: 2,
        routed_route_count: 2,
        routed_output_checksum: 3.0,
        shared_output_checksum: -0.75,
        output_checksum: 2.25,
        output_l2_norm: 1.25,
        first_output: 1.25,
        last_output: 1.0,
        outputs: vec![1.25, 1.0],
    }
}

pub(crate) fn mlp_residual_probe() -> RealSliceMlpResidualProbe {
    RealSliceMlpResidualProbe {
        layer_id: 3,
        output_count: 2,
        residual_source: "embedding_row_prefix".to_owned(),
        residual_checksum: -0.25,
        branch_checksum: 2.25,
        output_checksum: 2.0,
        output_l2_norm: 1.5811388,
        first_output: 1.5,
        last_output: 0.5,
    }
}
