use super::*;

mod attention;
mod dense;
mod layers;
mod mlp;
mod prefill;

use attention::*;
use dense::*;
use layers::attention_layer_fixture;
use mlp::*;
use prefill::*;

pub(super) fn real_slice_info_fixture() -> RealSliceInfo {
    let layer4 = attention_layer_fixture(4);
    let layer5 = attention_layer_fixture(5);
    let layer6 = attention_layer_fixture(6);
    let layer7 = attention_layer_fixture(7);
    let layer8 = attention_layer_fixture(8);
    let layer9 = attention_layer_fixture(9);
    let layer10 = attention_layer_fixture(10);
    let layer11 = attention_layer_fixture(11);
    let layer12 = attention_layer_fixture(12);
    let layer13 = attention_layer_fixture(13);

    RealSliceInfo {
        tensor_count: 1,
        total_bytes: 4,
        tensors: vec![RealSliceTensorInfo {
            name: "model.layers.0.input_layernorm.weight".to_owned(),
            bytes: 4,
            dtype: "bf16".to_owned(),
            shape: vec![2],
            sha256: "0123456789abcdef".to_owned(),
        }],
        logits_probe: Some(RealSliceLogitsProbe {
            probe_prompt: "hello".to_owned(),
            probe_prompt_token_ids: vec![42],
            hidden_token_id: 1,
            candidate_start_token_id: 0,
            candidate_count: 2,
            embedding_row_bytes: 4,
            lm_head_rows_bytes: 8,
            input_layernorm_weight_bytes: 4,
            embedding_row_sha256: "embedding".to_owned(),
            lm_head_rows_sha256: "lm_head".to_owned(),
            input_layernorm_weight_sha256: "norm".to_owned(),
            hidden_l2_norm: 1.0,
            rmsnorm_eps: 1.0e-5,
            rmsnorm_hidden_l2_norm: 1.5,
            rmsnorm_hidden: vec![0.25, -0.5],
            embedding_top_token_id: 0,
            embedding_top_logit: 0.5,
            top_token_id: 1,
            top_logit: 2.5,
            logits: vec![
                RealSliceLogit {
                    token_id: 0,
                    logit: 0.5,
                },
                RealSliceLogit {
                    token_id: 1,
                    logit: 2.5,
                },
            ],
            prefill_probe: Some(prefill_probe()),
            sampling_probe: Some(sampling_probe()),
            mlp_input_norm_probe: Some(mlp_input_norm_probe()),
            mlp_input_router_probe: Some(mlp_input_router_probe()),
            mlp_input_routed_expert_probe: Some(mlp_input_routed_expert_probe()),
            mlp_input_shared_expert_probe: Some(mlp_input_shared_expert_probe()),
            mlp_input_moe_branch_probe: Some(mlp_input_moe_branch_probe()),
            mlp_input_residual_probe: Some(mlp_input_residual_probe()),
            prefill_mlp_input_moe_probe: Some(prefill_mlp_input_moe_probe()),
            router_probe: Some(router_probe()),
            routed_expert_probe: Some(routed_expert_probe()),
            shared_expert_probe: Some(shared_expert_probe()),
            moe_branch_probe: Some(moe_branch_probe()),
            mlp_residual_probe: Some(mlp_residual_probe()),
            attention_probe: Some(attention_probe()),
            prefill_attention_kv_probe: Some(prefill_attention_kv_probe()),
            prefill_attention_output_probe: Some(prefill_attention_output_probe()),
            prefill_attention_residual_probe: Some(prefill_attention_residual_probe()),
            prefill_attention_mlp_input_probe: Some(prefill_attention_mlp_input_probe()),
            prefill_attention_mlp_input_moe_probe: Some(prefill_attention_mlp_input_moe_probe()),
            prefill_next_layer_attention_kv_probe: Some(layer4.kv_probe),
            prefill_next_layer_attention_output_probe: Some(layer4.output_probe),
            prefill_next_layer_attention_residual_probe: Some(layer4.residual_probe),
            prefill_next_layer_attention_mlp_input_probe: Some(layer4.mlp_input_probe),
            prefill_next_layer_attention_mlp_input_moe_probe: Some(layer4.mlp_input_moe_probe),
            prefill_following_layer_attention_kv_probe: Some(layer5.kv_probe),
            prefill_following_layer_attention_output_probe: Some(layer5.output_probe),
            prefill_following_layer_attention_residual_probe: Some(layer5.residual_probe),
            prefill_following_layer_attention_mlp_input_probe: Some(layer5.mlp_input_probe),
            prefill_following_layer_attention_mlp_input_moe_probe: Some(layer5.mlp_input_moe_probe),
            prefill_subsequent_layer_attention_kv_probe: Some(layer6.kv_probe),
            prefill_subsequent_layer_attention_output_probe: Some(layer6.output_probe),
            prefill_subsequent_layer_attention_residual_probe: Some(layer6.residual_probe),
            prefill_subsequent_layer_attention_mlp_input_probe: Some(layer6.mlp_input_probe),
            prefill_subsequent_layer_attention_mlp_input_moe_probe: Some(
                layer6.mlp_input_moe_probe,
            ),
            prefill_deeper_layer_attention_kv_probe: Some(layer7.kv_probe),
            prefill_deeper_layer_attention_output_probe: Some(layer7.output_probe),
            prefill_deeper_layer_attention_residual_probe: Some(layer7.residual_probe),
            prefill_deeper_layer_attention_mlp_input_probe: Some(layer7.mlp_input_probe),
            prefill_deeper_layer_attention_mlp_input_moe_probe: Some(layer7.mlp_input_moe_probe),
            prefill_further_layer_attention_kv_probe: Some(layer8.kv_probe),
            prefill_further_layer_attention_output_probe: Some(layer8.output_probe),
            prefill_further_layer_attention_residual_probe: Some(layer8.residual_probe),
            prefill_further_layer_attention_mlp_input_probe: Some(layer8.mlp_input_probe),
            prefill_further_layer_attention_mlp_input_moe_probe: Some(layer8.mlp_input_moe_probe),
            prefill_extended_layer_attention_kv_probe: Some(layer9.kv_probe),
            prefill_extended_layer_attention_output_probe: Some(layer9.output_probe),
            prefill_extended_layer_attention_residual_probe: Some(layer9.residual_probe),
            prefill_extended_layer_attention_mlp_input_probe: Some(layer9.mlp_input_probe),
            prefill_extended_layer_attention_mlp_input_moe_probe: Some(layer9.mlp_input_moe_probe),
            prefill_layer10_attention_kv_probe: Some(layer10.kv_probe),
            prefill_layer10_attention_output_probe: Some(layer10.output_probe),
            prefill_layer10_attention_residual_probe: Some(layer10.residual_probe),
            prefill_layer10_attention_mlp_input_probe: Some(layer10.mlp_input_probe),
            prefill_layer10_attention_mlp_input_moe_probe: Some(layer10.mlp_input_moe_probe),
            prefill_layer11_attention_kv_probe: Some(layer11.kv_probe),
            prefill_layer11_attention_output_probe: Some(layer11.output_probe),
            prefill_layer11_attention_residual_probe: Some(layer11.residual_probe),
            prefill_layer11_attention_mlp_input_probe: Some(layer11.mlp_input_probe),
            prefill_layer11_attention_mlp_input_moe_probe: Some(layer11.mlp_input_moe_probe),
            prefill_layer12_attention_kv_probe: Some(layer12.kv_probe),
            prefill_layer12_attention_output_probe: Some(layer12.output_probe),
            prefill_layer12_attention_residual_probe: Some(layer12.residual_probe),
            prefill_layer12_attention_mlp_input_probe: Some(layer12.mlp_input_probe),
            prefill_layer12_attention_mlp_input_moe_probe: Some(layer12.mlp_input_moe_probe),
            prefill_layer13_attention_kv_probe: Some(layer13.kv_probe),
            prefill_layer13_attention_output_probe: Some(layer13.output_probe),
            prefill_layer13_attention_residual_probe: Some(layer13.residual_probe),
            prefill_layer13_attention_mlp_input_probe: Some(layer13.mlp_input_probe),
            prefill_layer13_attention_mlp_input_moe_probe: Some(layer13.mlp_input_moe_probe),
            dense_mlp_probe: Some(dense_mlp_probe()),
            prefill_dense_mlp_probe: Some(prefill_dense_mlp_probe()),
            prefill_dense_mlp_residual_probe: Some(prefill_dense_mlp_residual_probe()),
        }),
    }
}
