use serde::{Deserialize, Serialize};

use super::RealSliceRoute;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealSliceMlpInputNormProbe {
    pub layer_id: u32,
    pub residual_source: String,
    pub hidden_width: usize,
    pub post_attention_layernorm_bytes: u64,
    pub post_attention_layernorm_sha256: String,
    pub normalized_checksum: f64,
    pub normalized_l2_norm: f32,
    pub first_normalized: f32,
    pub last_normalized: f32,
    #[serde(skip)]
    pub normalized_hidden: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealSliceRouterProbe {
    pub layer_id: u32,
    pub top_k: usize,
    pub router_weight_bytes: u64,
    pub router_bias_bytes: u64,
    pub router_weight_sha256: String,
    pub router_bias_sha256: String,
    pub routes: Vec<RealSliceRoute>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealSliceRoutedExpertProbe {
    pub layer_id: u32,
    pub expert_id: u32,
    pub owner: String,
    pub quant_recipe: String,
    pub intermediate_count: usize,
    pub output_count: usize,
    pub gate_proj_rows_bytes: u64,
    pub gate_proj_scale_rows_bytes: u64,
    pub gate_proj_scalar_bytes: u64,
    pub up_proj_rows_bytes: u64,
    pub up_proj_scale_rows_bytes: u64,
    pub up_proj_scalar_bytes: u64,
    pub down_proj_rows_bytes: u64,
    pub down_proj_scale_rows_bytes: u64,
    pub down_proj_scalar_bytes: u64,
    pub gate_proj_rows_sha256: String,
    pub gate_proj_scale_rows_sha256: String,
    pub up_proj_rows_sha256: String,
    pub up_proj_scale_rows_sha256: String,
    pub down_proj_rows_sha256: String,
    pub down_proj_scale_rows_sha256: String,
    pub activation_checksum: f64,
    pub output_checksum: f64,
    pub output_l2_norm: f32,
    pub first_output: f32,
    pub last_output: f32,
    pub reduction_route_count: usize,
    pub reduction_routes: Vec<String>,
    pub reduction_output_checksum: f64,
    pub reduction_output_l2_norm: f32,
    pub reduction_first_output: f32,
    pub reduction_last_output: f32,
    #[serde(skip)]
    pub reduction_outputs: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealSliceSharedExpertProbe {
    pub layer_id: u32,
    pub intermediate_count: usize,
    pub output_count: usize,
    pub gate_proj_rows_bytes: u64,
    pub up_proj_rows_bytes: u64,
    pub down_proj_rows_bytes: u64,
    pub gate_proj_rows_sha256: String,
    pub up_proj_rows_sha256: String,
    pub down_proj_rows_sha256: String,
    pub activation_checksum: f64,
    pub output_checksum: f64,
    pub output_l2_norm: f32,
    pub first_output: f32,
    pub last_output: f32,
    #[serde(skip)]
    pub outputs: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealSliceMoeBranchProbe {
    pub layer_id: u32,
    pub output_count: usize,
    pub routed_route_count: usize,
    pub routed_output_checksum: f64,
    pub shared_output_checksum: f64,
    pub output_checksum: f64,
    pub output_l2_norm: f32,
    pub first_output: f32,
    pub last_output: f32,
    #[serde(skip)]
    pub outputs: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealSliceMlpResidualProbe {
    pub layer_id: u32,
    pub output_count: usize,
    pub residual_source: String,
    pub residual_checksum: f64,
    pub branch_checksum: f64,
    pub output_checksum: f64,
    pub output_l2_norm: f32,
    pub first_output: f32,
    pub last_output: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealSlicePrefillMlpInputMoeProbe {
    pub layer_id: u32,
    pub row_count: usize,
    pub output_count: usize,
    pub route_count: usize,
    pub residual_source: String,
    pub routed_output_checksum: f64,
    pub shared_output_checksum: f64,
    pub branch_checksum: f64,
    pub residual_checksum: f64,
    pub output_checksum: f64,
    pub output_l2_norm: f32,
    pub first_output: f32,
    pub last_output: f32,
    #[serde(skip)]
    pub outputs: Vec<Vec<f32>>,
}
