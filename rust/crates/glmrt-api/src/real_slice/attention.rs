use serde::{Deserialize, Serialize};

use super::RealSlicePrefillRouteRow;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealSlicePrefillNextLayerAttentionKvProbe {
    pub source_layer_id: u32,
    pub layer_id: u32,
    pub row_count: usize,
    pub token_start: u32,
    pub token_count: usize,
    pub hidden_width: usize,
    pub input_prefix_count: usize,
    pub residual_source: String,
    pub input_layernorm_bytes: u64,
    pub input_layernorm_sha256: String,
    pub normalized_checksum: f64,
    pub normalized_l2_norm: f32,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub q_output_count: usize,
    pub kv_output_count: usize,
    pub q_a_proj_bytes: u64,
    pub q_a_layernorm_bytes: u64,
    pub q_b_proj_rows_bytes: u64,
    pub kv_a_proj_bytes: u64,
    pub kv_a_layernorm_bytes: u64,
    pub kv_b_proj_rows_bytes: u64,
    pub q_a_proj_sha256: String,
    pub q_a_layernorm_sha256: String,
    pub q_b_proj_rows_sha256: String,
    pub kv_a_proj_sha256: String,
    pub kv_a_layernorm_sha256: String,
    pub kv_b_proj_rows_sha256: String,
    pub q_output_checksum: f64,
    pub kv_output_checksum: f64,
    pub kv_rope_checksum: f64,
    pub first_normalized: f32,
    pub last_normalized: f32,
    pub q_first_output: f32,
    pub q_last_output: f32,
    pub kv_first_output: f32,
    pub kv_last_output: f32,
    #[serde(skip)]
    pub kv_outputs: Vec<Vec<f32>>,
    #[serde(skip)]
    pub residual_rows: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealSliceAttentionProjectionProbe {
    pub layer_id: u32,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub q_output_count: usize,
    pub kv_output_count: usize,
    pub q_a_proj_bytes: u64,
    pub q_a_layernorm_bytes: u64,
    pub q_b_proj_rows_bytes: u64,
    pub kv_a_proj_bytes: u64,
    pub kv_a_layernorm_bytes: u64,
    pub kv_b_proj_rows_bytes: u64,
    pub q_a_proj_sha256: String,
    pub q_a_layernorm_sha256: String,
    pub q_b_proj_rows_sha256: String,
    pub kv_a_proj_sha256: String,
    pub kv_a_layernorm_sha256: String,
    pub kv_b_proj_rows_sha256: String,
    pub q_a_norm_l2: f32,
    pub kv_a_norm_l2: f32,
    pub kv_rope_checksum: f64,
    pub q_output_checksum: f64,
    pub kv_output_checksum: f64,
    pub q_first_output: f32,
    pub q_last_output: f32,
    pub kv_first_output: f32,
    pub kv_last_output: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealSlicePrefillAttentionOutputProbe {
    pub layer_id: u32,
    pub row_count: usize,
    pub input_count: usize,
    pub output_count: usize,
    pub context_source: String,
    pub o_proj_rows_bytes: u64,
    pub o_proj_rows_sha256: String,
    pub input_checksum: f64,
    pub output_checksum: f64,
    pub output_l2_norm: f32,
    pub first_output: f32,
    pub last_output: f32,
    #[serde(skip)]
    pub outputs: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealSlicePrefillAttentionResidualProbe {
    pub layer_id: u32,
    pub row_count: usize,
    pub output_count: usize,
    pub residual_source: String,
    pub attention_source: String,
    pub residual_checksum: f64,
    pub branch_checksum: f64,
    pub output_checksum: f64,
    pub output_l2_norm: f32,
    pub first_output: f32,
    pub last_output: f32,
    #[serde(skip)]
    pub outputs: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealSlicePrefillAttentionMlpInputProbe {
    pub layer_id: u32,
    pub row_count: usize,
    pub hidden_width: usize,
    pub attention_prefix_count: usize,
    pub router_top_k: usize,
    pub residual_source: String,
    pub post_attention_layernorm_bytes: u64,
    pub post_attention_layernorm_sha256: String,
    pub router_weight_sha256: String,
    pub router_bias_sha256: String,
    pub normalized_checksum: f64,
    pub normalized_l2_norm: f32,
    pub first_normalized: f32,
    pub last_normalized: f32,
    pub route_rows: Vec<RealSlicePrefillRouteRow>,
    #[serde(skip)]
    pub normalized_rows: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealSlicePrefillAttentionKvProbe {
    pub layer_id: u32,
    pub row_count: usize,
    pub token_start: u32,
    pub token_count: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub q_output_count: usize,
    pub kv_output_count: usize,
    pub kv_write_count: usize,
    pub kv_written_count: usize,
    pub q_output_checksum: f64,
    pub kv_output_checksum: f64,
    pub kv_rope_checksum: f64,
    pub q_first_output: f32,
    pub q_last_output: f32,
    pub kv_first_output: f32,
    pub kv_last_output: f32,
    #[serde(skip)]
    pub kv_outputs: Vec<Vec<f32>>,
}
