use crate::{
    DEFAULT_MODEL_ID, GLM52_FIRST_K_DENSE_REPLACE, GLM52_HIDDEN_SIZE, GLM52_NUM_HIDDEN_LAYERS,
    GLM52_ROUTED_EXPERTS, GLM52_TOP_K,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TensorRole {
    RoutedExpert,
    SharedExpert,
    Attention,
    Router,
    Norm,
    Embedding,
    LmHead,
    DenseMlp,
    Mtp,
    Quantization,
    Config,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    Bf16,
    F16,
    F32,
    F8E4M3,
    F8E5M2,
    I8,
    I16,
    I32,
    U8,
    F4,
    Unknown(String),
}

impl DType {
    pub fn from_safetensors(value: &str) -> Self {
        match value {
            "BF16" => DType::Bf16,
            "F16" => DType::F16,
            "F32" => DType::F32,
            "F8_E4M3" => DType::F8E4M3,
            "F8_E5M2" => DType::F8E5M2,
            "I8" => DType::I8,
            "I16" => DType::I16,
            "I32" => DType::I32,
            "U8" => DType::U8,
            "F4" => DType::F4,
            other => DType::Unknown(other.to_owned()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFacts {
    pub model_id: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub first_k_dense_replace: usize,
    pub routed_experts: usize,
    pub top_k: usize,
    pub quantization_recipe: String,
}

impl Default for ModelFacts {
    fn default() -> Self {
        Self {
            model_id: DEFAULT_MODEL_ID.to_owned(),
            hidden_size: GLM52_HIDDEN_SIZE,
            num_hidden_layers: GLM52_NUM_HIDDEN_LAYERS,
            first_k_dense_replace: GLM52_FIRST_K_DENSE_REPLACE,
            routed_experts: GLM52_ROUTED_EXPERTS,
            top_k: GLM52_TOP_K,
            quantization_recipe: "glm52_nvfp4_lukealonso_v1".to_owned(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub file: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub byte_offset: u64,
    pub byte_length: u64,
    pub role: TensorRole,
    pub layer_id: Option<u32>,
    pub expert_id: Option<u32>,
    pub is_quantization_metadata: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorCatalog {
    pub model_id: String,
    pub snapshot_path: String,
    pub facts: ModelFacts,
    pub tensors: Vec<TensorInfo>,
}

impl TensorCatalog {
    pub fn summary_by_role(&self) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for tensor in &self.tensors {
            *counts.entry(format!("{:?}", tensor.role)).or_insert(0) += 1;
        }
        counts
    }

    pub fn content_hash(&self) -> String {
        let encoded = serde_json::to_vec(self).expect("serializing catalog cannot fail");
        let digest = Sha256::digest(encoded);
        format!("{digest:x}")
    }
}
