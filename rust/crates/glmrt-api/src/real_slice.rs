use serde::{Deserialize, Serialize};

mod attention;
mod dense;
mod mlp;

pub use attention::{
    RealSliceAttentionProjectionProbe, RealSlicePrefillAttentionKvProbe,
    RealSlicePrefillAttentionMlpInputProbe, RealSlicePrefillAttentionOutputProbe,
    RealSlicePrefillAttentionResidualProbe, RealSlicePrefillNextLayerAttentionKvProbe,
};
pub use dense::{
    RealSliceDenseMlpProbe, RealSlicePrefillDenseMlpProbe, RealSlicePrefillDenseMlpResidualProbe,
};
pub use mlp::{
    RealSliceMlpInputNormProbe, RealSliceMlpResidualProbe, RealSliceMoeBranchProbe,
    RealSlicePrefillMlpInputMoeProbe, RealSliceRoutedExpertProbe, RealSliceRouterProbe,
    RealSliceSharedExpertProbe,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealSliceInfo {
    pub tensor_count: usize,
    pub total_bytes: u64,
    pub tensors: Vec<RealSliceTensorInfo>,
    pub logits_probe: Option<RealSliceLogitsProbe>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealSliceTensorInfo {
    pub name: String,
    pub bytes: u64,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealSliceLogitsProbe {
    pub probe_prompt: String,
    pub probe_prompt_token_ids: Vec<u32>,
    pub hidden_token_id: u32,
    pub candidate_start_token_id: u32,
    pub candidate_count: usize,
    pub embedding_row_bytes: u64,
    pub lm_head_rows_bytes: u64,
    pub input_layernorm_weight_bytes: u64,
    pub embedding_row_sha256: String,
    pub lm_head_rows_sha256: String,
    pub input_layernorm_weight_sha256: String,
    pub hidden_l2_norm: f32,
    pub rmsnorm_eps: f32,
    pub rmsnorm_hidden_l2_norm: f32,
    #[serde(skip)]
    pub rmsnorm_hidden: Vec<f32>,
    pub embedding_top_token_id: u32,
    pub embedding_top_logit: f32,
    pub top_token_id: u32,
    pub top_logit: f32,
    pub logits: Vec<RealSliceLogit>,
    pub prefill_probe: Option<RealSlicePrefillProbe>,
    pub sampling_probe: Option<RealSliceSamplingProbe>,
    pub mlp_input_norm_probe: Option<RealSliceMlpInputNormProbe>,
    pub mlp_input_router_probe: Option<RealSliceRouterProbe>,
    pub mlp_input_routed_expert_probe: Option<RealSliceRoutedExpertProbe>,
    pub mlp_input_shared_expert_probe: Option<RealSliceSharedExpertProbe>,
    pub mlp_input_moe_branch_probe: Option<RealSliceMoeBranchProbe>,
    pub mlp_input_residual_probe: Option<RealSliceMlpResidualProbe>,
    pub prefill_mlp_input_moe_probe: Option<RealSlicePrefillMlpInputMoeProbe>,
    pub router_probe: Option<RealSliceRouterProbe>,
    pub routed_expert_probe: Option<RealSliceRoutedExpertProbe>,
    pub shared_expert_probe: Option<RealSliceSharedExpertProbe>,
    pub moe_branch_probe: Option<RealSliceMoeBranchProbe>,
    pub mlp_residual_probe: Option<RealSliceMlpResidualProbe>,
    pub attention_probe: Option<RealSliceAttentionProjectionProbe>,
    pub prefill_attention_kv_probe: Option<RealSlicePrefillAttentionKvProbe>,
    pub prefill_attention_output_probe: Option<RealSlicePrefillAttentionOutputProbe>,
    pub prefill_attention_residual_probe: Option<RealSlicePrefillAttentionResidualProbe>,
    pub prefill_attention_mlp_input_probe: Option<RealSlicePrefillAttentionMlpInputProbe>,
    pub prefill_attention_mlp_input_moe_probe: Option<RealSlicePrefillMlpInputMoeProbe>,
    pub prefill_next_layer_attention_kv_probe: Option<RealSlicePrefillNextLayerAttentionKvProbe>,
    pub prefill_next_layer_attention_output_probe: Option<RealSlicePrefillAttentionOutputProbe>,
    pub prefill_next_layer_attention_residual_probe: Option<RealSlicePrefillAttentionResidualProbe>,
    pub prefill_next_layer_attention_mlp_input_probe:
        Option<RealSlicePrefillAttentionMlpInputProbe>,
    pub prefill_next_layer_attention_mlp_input_moe_probe: Option<RealSlicePrefillMlpInputMoeProbe>,
    pub prefill_following_layer_attention_kv_probe:
        Option<RealSlicePrefillNextLayerAttentionKvProbe>,
    pub prefill_following_layer_attention_output_probe:
        Option<RealSlicePrefillAttentionOutputProbe>,
    pub prefill_following_layer_attention_residual_probe:
        Option<RealSlicePrefillAttentionResidualProbe>,
    pub prefill_following_layer_attention_mlp_input_probe:
        Option<RealSlicePrefillAttentionMlpInputProbe>,
    pub prefill_following_layer_attention_mlp_input_moe_probe:
        Option<RealSlicePrefillMlpInputMoeProbe>,
    pub prefill_subsequent_layer_attention_kv_probe:
        Option<RealSlicePrefillNextLayerAttentionKvProbe>,
    pub prefill_subsequent_layer_attention_output_probe:
        Option<RealSlicePrefillAttentionOutputProbe>,
    pub prefill_subsequent_layer_attention_residual_probe:
        Option<RealSlicePrefillAttentionResidualProbe>,
    pub prefill_subsequent_layer_attention_mlp_input_probe:
        Option<RealSlicePrefillAttentionMlpInputProbe>,
    pub prefill_subsequent_layer_attention_mlp_input_moe_probe:
        Option<RealSlicePrefillMlpInputMoeProbe>,
    pub prefill_deeper_layer_attention_kv_probe: Option<RealSlicePrefillNextLayerAttentionKvProbe>,
    pub prefill_deeper_layer_attention_output_probe: Option<RealSlicePrefillAttentionOutputProbe>,
    pub prefill_deeper_layer_attention_residual_probe:
        Option<RealSlicePrefillAttentionResidualProbe>,
    pub prefill_deeper_layer_attention_mlp_input_probe:
        Option<RealSlicePrefillAttentionMlpInputProbe>,
    pub prefill_deeper_layer_attention_mlp_input_moe_probe:
        Option<RealSlicePrefillMlpInputMoeProbe>,
    pub prefill_further_layer_attention_kv_probe: Option<RealSlicePrefillNextLayerAttentionKvProbe>,
    pub prefill_further_layer_attention_output_probe: Option<RealSlicePrefillAttentionOutputProbe>,
    pub prefill_further_layer_attention_residual_probe:
        Option<RealSlicePrefillAttentionResidualProbe>,
    pub prefill_further_layer_attention_mlp_input_probe:
        Option<RealSlicePrefillAttentionMlpInputProbe>,
    pub prefill_further_layer_attention_mlp_input_moe_probe:
        Option<RealSlicePrefillMlpInputMoeProbe>,
    pub prefill_extended_layer_attention_kv_probe:
        Option<RealSlicePrefillNextLayerAttentionKvProbe>,
    pub prefill_extended_layer_attention_output_probe: Option<RealSlicePrefillAttentionOutputProbe>,
    pub prefill_extended_layer_attention_residual_probe:
        Option<RealSlicePrefillAttentionResidualProbe>,
    pub prefill_extended_layer_attention_mlp_input_probe:
        Option<RealSlicePrefillAttentionMlpInputProbe>,
    pub prefill_extended_layer_attention_mlp_input_moe_probe:
        Option<RealSlicePrefillMlpInputMoeProbe>,
    pub prefill_layer10_attention_kv_probe: Option<RealSlicePrefillNextLayerAttentionKvProbe>,
    pub prefill_layer10_attention_output_probe: Option<RealSlicePrefillAttentionOutputProbe>,
    pub prefill_layer10_attention_residual_probe: Option<RealSlicePrefillAttentionResidualProbe>,
    pub prefill_layer10_attention_mlp_input_probe: Option<RealSlicePrefillAttentionMlpInputProbe>,
    pub prefill_layer10_attention_mlp_input_moe_probe: Option<RealSlicePrefillMlpInputMoeProbe>,
    pub prefill_layer11_attention_kv_probe: Option<RealSlicePrefillNextLayerAttentionKvProbe>,
    pub prefill_layer11_attention_output_probe: Option<RealSlicePrefillAttentionOutputProbe>,
    pub prefill_layer11_attention_residual_probe: Option<RealSlicePrefillAttentionResidualProbe>,
    pub prefill_layer11_attention_mlp_input_probe: Option<RealSlicePrefillAttentionMlpInputProbe>,
    pub prefill_layer11_attention_mlp_input_moe_probe: Option<RealSlicePrefillMlpInputMoeProbe>,
    pub prefill_layer12_attention_kv_probe: Option<RealSlicePrefillNextLayerAttentionKvProbe>,
    pub prefill_layer12_attention_output_probe: Option<RealSlicePrefillAttentionOutputProbe>,
    pub prefill_layer12_attention_residual_probe: Option<RealSlicePrefillAttentionResidualProbe>,
    pub prefill_layer12_attention_mlp_input_probe: Option<RealSlicePrefillAttentionMlpInputProbe>,
    pub prefill_layer12_attention_mlp_input_moe_probe: Option<RealSlicePrefillMlpInputMoeProbe>,
    pub prefill_layer13_attention_kv_probe: Option<RealSlicePrefillNextLayerAttentionKvProbe>,
    pub prefill_layer13_attention_output_probe: Option<RealSlicePrefillAttentionOutputProbe>,
    pub prefill_layer13_attention_residual_probe: Option<RealSlicePrefillAttentionResidualProbe>,
    pub prefill_layer13_attention_mlp_input_probe: Option<RealSlicePrefillAttentionMlpInputProbe>,
    pub prefill_layer13_attention_mlp_input_moe_probe: Option<RealSlicePrefillMlpInputMoeProbe>,
    pub dense_mlp_probe: Option<RealSliceDenseMlpProbe>,
    pub prefill_dense_mlp_probe: Option<RealSlicePrefillDenseMlpProbe>,
    pub prefill_dense_mlp_residual_probe: Option<RealSlicePrefillDenseMlpResidualProbe>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealSliceLogit {
    pub token_id: u32,
    pub logit: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealSlicePrefillProbe {
    pub prompt_token_count: usize,
    pub chunk_token_count: usize,
    pub hidden_width: usize,
    pub router_layer_id: u32,
    pub router_top_k: usize,
    pub embedding_rows_bytes: u64,
    pub input_layernorm_weight_bytes: u64,
    pub mlp_input_layernorm_weight_bytes: u64,
    pub embedding_rows_sha256: String,
    pub input_layernorm_weight_sha256: String,
    pub mlp_input_layernorm_weight_sha256: String,
    pub hidden_checksum: f64,
    pub rmsnorm_checksum: f64,
    pub mlp_input_checksum: f64,
    pub hidden_l2_norm: f32,
    pub rmsnorm_l2_norm: f32,
    pub mlp_input_l2_norm: f32,
    pub first_token_id: u32,
    pub last_token_id: u32,
    pub first_hidden_value: f32,
    pub last_hidden_value: f32,
    pub first_rmsnorm_value: f32,
    pub last_rmsnorm_value: f32,
    pub first_mlp_input_value: f32,
    pub last_mlp_input_value: f32,
    pub mlp_input_residual_source: String,
    pub route_rows: Vec<RealSlicePrefillRouteRow>,
    pub mlp_input_route_rows: Vec<RealSlicePrefillRouteRow>,
    #[serde(skip)]
    pub hidden_rows: Vec<Vec<f32>>,
    #[serde(skip)]
    pub rmsnorm_rows: Vec<Vec<f32>>,
    #[serde(skip)]
    pub mlp_input_rows: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealSlicePrefillRouteRow {
    pub row_id: u64,
    pub token_id: u32,
    pub routes: Vec<RealSliceRoute>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealSliceSamplingProbe {
    pub strategy: String,
    pub candidate_start_token_id: u32,
    pub candidate_count: usize,
    pub selected_token_id: u32,
    pub selected_logit: f32,
    pub skip_special_tokens: bool,
    pub decoded_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealSliceRoute {
    pub expert_id: u32,
    pub owner: String,
    pub score: f32,
    pub corrected_score: f32,
    pub normalized_weight: f32,
}
