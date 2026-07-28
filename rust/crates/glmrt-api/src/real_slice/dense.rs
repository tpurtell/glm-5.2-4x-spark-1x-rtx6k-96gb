use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealSliceDenseMlpProbe {
    pub layer_id: u32,
    pub intermediate_count: usize,
    pub output_count: usize,
    pub post_attention_layernorm_bytes: u64,
    pub gate_proj_rows_bytes: u64,
    pub up_proj_rows_bytes: u64,
    pub down_proj_rows_bytes: u64,
    pub post_attention_layernorm_sha256: String,
    pub gate_proj_rows_sha256: String,
    pub up_proj_rows_sha256: String,
    pub down_proj_rows_sha256: String,
    pub norm_l2: f32,
    pub activation_checksum: f64,
    pub output_checksum: f64,
    pub output_l2_norm: f32,
    pub first_output: f32,
    pub last_output: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealSlicePrefillDenseMlpProbe {
    pub layer_id: u32,
    pub row_count: usize,
    pub intermediate_count: usize,
    pub output_count: usize,
    pub residual_source: String,
    pub norm_checksum: f64,
    pub norm_l2_norm: f32,
    pub activation_checksum: f64,
    pub output_checksum: f64,
    pub output_l2_norm: f32,
    pub first_output: f32,
    pub last_output: f32,
    #[serde(skip)]
    pub outputs: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealSlicePrefillDenseMlpResidualProbe {
    pub layer_id: u32,
    pub row_count: usize,
    pub output_count: usize,
    pub residual_source: String,
    pub residual_checksum: f64,
    pub branch_checksum: f64,
    pub output_checksum: f64,
    pub output_l2_norm: f32,
    pub first_output: f32,
    pub last_output: f32,
}
