use crate::{
    RealSlicePrefillProbe, RealSlicePrefillRouteRow, RealSliceRoute, RealSliceSamplingProbe,
};

pub(super) fn prefill_probe() -> RealSlicePrefillProbe {
    RealSlicePrefillProbe {
        prompt_token_count: 2,
        chunk_token_count: 2,
        hidden_width: 2,
        router_layer_id: 3,
        router_top_k: 2,
        embedding_rows_bytes: 8,
        input_layernorm_weight_bytes: 4,
        mlp_input_layernorm_weight_bytes: 4,
        embedding_rows_sha256: "prefill_embedding".to_owned(),
        input_layernorm_weight_sha256: "norm".to_owned(),
        mlp_input_layernorm_weight_sha256: "mlp_prefill_norm".to_owned(),
        hidden_checksum: 1.25,
        rmsnorm_checksum: -0.25,
        mlp_input_checksum: 0.75,
        hidden_l2_norm: 1.75,
        rmsnorm_l2_norm: 2.0,
        mlp_input_l2_norm: 1.25,
        first_token_id: 4,
        last_token_id: 5,
        first_hidden_value: 0.25,
        last_hidden_value: -0.75,
        first_rmsnorm_value: 0.5,
        last_rmsnorm_value: -1.25,
        first_mlp_input_value: 0.5,
        last_mlp_input_value: 0.25,
        mlp_input_residual_source: "embedding_row_residual_standin".to_owned(),
        route_rows: vec![
            RealSlicePrefillRouteRow {
                row_id: 0,
                token_id: 4,
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
            },
            RealSlicePrefillRouteRow {
                row_id: 1,
                token_id: 5,
                routes: vec![RealSliceRoute {
                    expert_id: 7,
                    owner: "kiwi".to_owned(),
                    score: 0.7,
                    corrected_score: 0.8,
                    normalized_weight: 1.0,
                }],
            },
        ],
        mlp_input_route_rows: vec![
            RealSlicePrefillRouteRow {
                row_id: 0,
                token_id: 4,
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
            },
            RealSlicePrefillRouteRow {
                row_id: 1,
                token_id: 5,
                routes: vec![RealSliceRoute {
                    expert_id: 13,
                    owner: "kiwi".to_owned(),
                    score: 0.7,
                    corrected_score: 0.85,
                    normalized_weight: 1.0,
                }],
            },
        ],
        hidden_rows: vec![vec![0.25, -0.75], vec![1.5, 0.25]],
        rmsnorm_rows: vec![vec![0.5, -0.25], vec![0.75, -1.25]],
        mlp_input_rows: vec![vec![0.5, -0.25], vec![0.75, 0.25]],
    }
}

pub(super) fn sampling_probe() -> RealSliceSamplingProbe {
    RealSliceSamplingProbe {
        strategy: "greedy".to_owned(),
        candidate_start_token_id: 0,
        candidate_count: 2,
        selected_token_id: 1,
        selected_logit: 2.5,
        skip_special_tokens: false,
        decoded_text: "world".to_owned(),
    }
}
