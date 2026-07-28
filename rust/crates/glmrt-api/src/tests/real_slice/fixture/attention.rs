use crate::{
    RealSliceAttentionProjectionProbe, RealSlicePrefillAttentionKvProbe,
    RealSlicePrefillAttentionMlpInputProbe, RealSlicePrefillAttentionOutputProbe,
    RealSlicePrefillAttentionResidualProbe, RealSlicePrefillMlpInputMoeProbe,
    RealSlicePrefillRouteRow, RealSliceRoute,
};

pub(super) fn attention_probe() -> RealSliceAttentionProjectionProbe {
    RealSliceAttentionProjectionProbe {
        layer_id: 3,
        q_lora_rank: 4,
        kv_lora_rank: 2,
        q_output_count: 2,
        kv_output_count: 2,
        q_a_proj_bytes: 16,
        q_a_layernorm_bytes: 8,
        q_b_proj_rows_bytes: 16,
        kv_a_proj_bytes: 8,
        kv_a_layernorm_bytes: 4,
        kv_b_proj_rows_bytes: 8,
        q_a_proj_sha256: "q_a".to_owned(),
        q_a_layernorm_sha256: "q_norm".to_owned(),
        q_b_proj_rows_sha256: "q_b".to_owned(),
        kv_a_proj_sha256: "kv_a".to_owned(),
        kv_a_layernorm_sha256: "kv_norm".to_owned(),
        kv_b_proj_rows_sha256: "kv_b".to_owned(),
        q_a_norm_l2: 1.5,
        kv_a_norm_l2: 0.75,
        kv_rope_checksum: 0.125,
        q_output_checksum: 2.5,
        kv_output_checksum: -1.25,
        q_first_output: 1.0,
        q_last_output: 1.5,
        kv_first_output: -0.25,
        kv_last_output: -1.0,
    }
}

pub(super) fn prefill_attention_kv_probe() -> RealSlicePrefillAttentionKvProbe {
    RealSlicePrefillAttentionKvProbe {
        layer_id: 3,
        row_count: 2,
        token_start: 0,
        token_count: 2,
        q_lora_rank: 4,
        kv_lora_rank: 2,
        q_output_count: 2,
        kv_output_count: 2,
        kv_write_count: 1,
        kv_written_count: 1,
        q_output_checksum: 4.5,
        kv_output_checksum: -2.5,
        kv_rope_checksum: 0.75,
        q_first_output: 1.0,
        q_last_output: 2.0,
        kv_first_output: -0.5,
        kv_last_output: -1.5,
        kv_outputs: vec![vec![-0.5, -0.5], vec![-0.5, -1.0]],
    }
}

pub(super) fn prefill_attention_output_probe() -> RealSlicePrefillAttentionOutputProbe {
    RealSlicePrefillAttentionOutputProbe {
        layer_id: 3,
        row_count: 2,
        input_count: 2,
        output_count: 2,
        context_source: "kv_b_prefix_standin".to_owned(),
        o_proj_rows_bytes: 16,
        o_proj_rows_sha256: "o_proj".to_owned(),
        input_checksum: -2.5,
        output_checksum: 1.25,
        output_l2_norm: 1.0,
        first_output: 0.25,
        last_output: 1.0,
        outputs: vec![vec![0.25, 0.0], vec![0.0, 1.0]],
    }
}

pub(super) fn prefill_attention_residual_probe() -> RealSlicePrefillAttentionResidualProbe {
    RealSlicePrefillAttentionResidualProbe {
        layer_id: 3,
        row_count: 2,
        output_count: 2,
        residual_source: "embedding_row_prefix".to_owned(),
        attention_source: "kv_b_prefix_standin_o_proj_prefix".to_owned(),
        residual_checksum: 1.25,
        branch_checksum: 1.25,
        output_checksum: 2.5,
        output_l2_norm: 2.1505814,
        first_output: 0.5,
        last_output: 1.25,
        outputs: vec![vec![0.5, -0.75], vec![1.5, 1.25]],
    }
}

pub(super) fn prefill_attention_mlp_input_probe() -> RealSlicePrefillAttentionMlpInputProbe {
    RealSlicePrefillAttentionMlpInputProbe {
        layer_id: 3,
        row_count: 2,
        hidden_width: 2,
        attention_prefix_count: 2,
        router_top_k: 2,
        residual_source: "attention_residual_prefix_spliced".to_owned(),
        post_attention_layernorm_bytes: 4,
        post_attention_layernorm_sha256: "attn_mlp_norm".to_owned(),
        router_weight_sha256: "attn_mlp_router_weight".to_owned(),
        router_bias_sha256: "attn_mlp_router_bias".to_owned(),
        normalized_checksum: 2.25,
        normalized_l2_norm: 1.6,
        first_normalized: 0.5,
        last_normalized: 0.75,
        route_rows: vec![
            RealSlicePrefillRouteRow {
                row_id: 0,
                token_id: 4,
                routes: vec![RealSliceRoute {
                    expert_id: 17,
                    owner: "emu".to_owned(),
                    score: 0.75,
                    corrected_score: 0.85,
                    normalized_weight: 1.0,
                }],
            },
            RealSlicePrefillRouteRow {
                row_id: 1,
                token_id: 5,
                routes: vec![RealSliceRoute {
                    expert_id: 19,
                    owner: "dodo".to_owned(),
                    score: 0.65,
                    corrected_score: 0.75,
                    normalized_weight: 1.0,
                }],
            },
        ],
        normalized_rows: vec![vec![0.5, -0.25], vec![1.25, 0.75]],
    }
}

pub(super) fn prefill_attention_mlp_input_moe_probe() -> RealSlicePrefillMlpInputMoeProbe {
    RealSlicePrefillMlpInputMoeProbe {
        layer_id: 3,
        row_count: 2,
        output_count: 2,
        route_count: 2,
        residual_source: "attention_residual_prefix_spliced".to_owned(),
        routed_output_checksum: 1.5,
        shared_output_checksum: -0.5,
        branch_checksum: 1.0,
        residual_checksum: 2.5,
        output_checksum: 3.5,
        output_l2_norm: 2.5,
        first_output: 1.0,
        last_output: 1.75,
        outputs: vec![vec![1.0, 0.75], vec![0.0, 1.75]],
    }
}
