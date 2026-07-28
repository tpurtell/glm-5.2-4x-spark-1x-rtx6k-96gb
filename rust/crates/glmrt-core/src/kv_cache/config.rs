use serde::{Deserialize, Serialize};

use crate::{
    KvBlockDescriptor, LayerId, GLM52_DSA_INDEXER_LAYERS, GLM52_DSA_INDEXER_LAYER_IDS,
    GLM52_DSA_INDEXER_LAYER_IDS_WITH_MTP, GLM52_DSA_INDEX_HEAD_DIM, GLM52_HIDDEN_SIZE,
    GLM52_MLA_FP8_DS_BYTES_PER_TOKEN, GLM52_MLA_KV_LORA_RANK, GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN,
    GLM52_MLA_QK_ROPE_HEAD_DIM, GLM52_NUM_HIDDEN_LAYERS, GLM52_TOTAL_LAYERS_WITH_MTP,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KvCacheDType {
    Bf16,
    F16,
    Fp8,
    Nvfp4,
    F32,
}

impl KvCacheDType {
    pub fn parse_glm52_cache_dtype(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "bf16" | "bfloat16" => Some(Self::Bf16),
            "f16" | "fp16" | "float16" => Some(Self::F16),
            "fp8" | "f8" => Some(Self::Fp8),
            "nvfp4" | "fp4" | "f4" => Some(Self::Nvfp4),
            "f32" | "fp32" | "float32" => Some(Self::F32),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            KvCacheDType::Bf16 => "bf16",
            KvCacheDType::F16 => "f16",
            KvCacheDType::Fp8 => "fp8",
            KvCacheDType::Nvfp4 => "nvfp4",
            KvCacheDType::F32 => "f32",
        }
    }

    pub fn bits_per_element(self) -> usize {
        match self {
            KvCacheDType::Bf16 | KvCacheDType::F16 => 16,
            KvCacheDType::Fp8 => 8,
            KvCacheDType::Nvfp4 => 4,
            KvCacheDType::F32 => 32,
        }
    }

    pub fn bytes_per_element(self) -> usize {
        let bits = self.bits_per_element();
        assert!(
            bits % 8 == 0,
            "packed KV dtype {} is not byte-addressable per element",
            self.label()
        );
        bits / 8
    }

    pub fn packed_bytes_for_elements(self, elements: usize) -> usize {
        (elements * self.bits_per_element()).div_ceil(8)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KvLayout {
    Glm52CompressedBf16,
    Glm52CompressedFp8,
    Glm52CompressedNvfp4,
    ExpandedDebugOnly,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MlaKvCacheRepresentation {
    #[default]
    RawProjected,
    NormalizedRotated,
}

impl MlaKvCacheRepresentation {
    pub fn label(self) -> &'static str {
        match self {
            Self::RawProjected => "raw-projected",
            Self::NormalizedRotated => "normalized-rotated",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCacheConfig {
    pub layout: KvLayout,
    pub layers: usize,
    pub key_value_width: usize,
    pub dtype: KvCacheDType,
    #[serde(default)]
    pub mla_representation: MlaKvCacheRepresentation,
    pub dsa_indexer_layers: usize,
    pub dsa_index_head_dim: usize,
    pub fp8_scale_metadata_bytes_per_token: usize,
    pub max_tokens: usize,
}

impl KvCacheConfig {
    pub fn glm52_phase0(max_tokens: usize) -> Self {
        Self::glm52_compressed_bf16(max_tokens)
    }

    pub fn glm52_compressed(max_tokens: usize, dtype: KvCacheDType) -> Option<Self> {
        match dtype {
            KvCacheDType::Bf16 => Some(Self::glm52_compressed_bf16(max_tokens)),
            KvCacheDType::Fp8 => Some(Self::glm52_compressed_fp8(max_tokens)),
            KvCacheDType::Nvfp4 => Some(Self::glm52_compressed_nvfp4(max_tokens)),
            KvCacheDType::F16 | KvCacheDType::F32 => None,
        }
    }

    pub fn glm52_compressed_bf16(max_tokens: usize) -> Self {
        Self {
            layout: KvLayout::Glm52CompressedBf16,
            layers: GLM52_NUM_HIDDEN_LAYERS,
            key_value_width: GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM,
            dtype: KvCacheDType::Bf16,
            mla_representation: MlaKvCacheRepresentation::RawProjected,
            dsa_indexer_layers: GLM52_DSA_INDEXER_LAYERS,
            dsa_index_head_dim: GLM52_DSA_INDEX_HEAD_DIM,
            fp8_scale_metadata_bytes_per_token: 0,
            max_tokens,
        }
    }

    pub fn glm52_compressed_fp8(max_tokens: usize) -> Self {
        Self {
            layout: KvLayout::Glm52CompressedFp8,
            layers: GLM52_NUM_HIDDEN_LAYERS,
            key_value_width: GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM,
            dtype: KvCacheDType::Fp8,
            mla_representation: MlaKvCacheRepresentation::RawProjected,
            dsa_indexer_layers: GLM52_DSA_INDEXER_LAYERS,
            dsa_index_head_dim: GLM52_DSA_INDEX_HEAD_DIM,
            fp8_scale_metadata_bytes_per_token: 0,
            max_tokens,
        }
    }

    pub fn glm52_compressed_nvfp4(max_tokens: usize) -> Self {
        Self {
            layout: KvLayout::Glm52CompressedNvfp4,
            layers: GLM52_NUM_HIDDEN_LAYERS,
            key_value_width: GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM,
            dtype: KvCacheDType::Nvfp4,
            mla_representation: MlaKvCacheRepresentation::RawProjected,
            dsa_indexer_layers: GLM52_DSA_INDEXER_LAYERS,
            dsa_index_head_dim: GLM52_DSA_INDEX_HEAD_DIM,
            fp8_scale_metadata_bytes_per_token: 0,
            max_tokens,
        }
    }

    pub fn glm52_expanded_debug_bf16(max_tokens: usize) -> Self {
        Self {
            layout: KvLayout::ExpandedDebugOnly,
            layers: GLM52_NUM_HIDDEN_LAYERS,
            key_value_width: GLM52_HIDDEN_SIZE,
            dtype: KvCacheDType::Bf16,
            mla_representation: MlaKvCacheRepresentation::RawProjected,
            dsa_indexer_layers: 0,
            dsa_index_head_dim: 0,
            fp8_scale_metadata_bytes_per_token: 0,
            max_tokens,
        }
    }

    pub fn with_mla_representation(mut self, representation: MlaKvCacheRepresentation) -> Self {
        self.mla_representation = representation;
        self
    }

    pub fn with_mtp_layer(mut self) -> Self {
        self.layers = GLM52_TOTAL_LAYERS_WITH_MTP;
        if matches!(
            self.layout,
            KvLayout::Glm52CompressedBf16
                | KvLayout::Glm52CompressedFp8
                | KvLayout::Glm52CompressedNvfp4
        ) {
            self.dsa_indexer_layers = GLM52_DSA_INDEXER_LAYER_IDS_WITH_MTP.len();
        }
        self
    }

    pub fn layout_label(&self) -> &'static str {
        match self.layout {
            KvLayout::Glm52CompressedBf16 => "glm52-compressed-bf16",
            KvLayout::Glm52CompressedFp8 => "glm52-compressed-fp8",
            KvLayout::Glm52CompressedNvfp4 => "glm52-compressed-nvfp4",
            KvLayout::ExpandedDebugOnly => "expanded-debug-only",
        }
    }

    pub fn dtype_label(&self) -> &'static str {
        self.dtype.label()
    }

    pub fn main_mla_bytes_per_token(&self) -> usize {
        if self.layout == KvLayout::Glm52CompressedFp8 {
            return self.layers * GLM52_MLA_FP8_DS_BYTES_PER_TOKEN;
        }
        if self.layout == KvLayout::Glm52CompressedNvfp4 {
            return self.layers * GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN;
        }
        self.dtype
            .packed_bytes_for_elements(self.layers * self.key_value_width)
    }

    pub fn dsa_indexer_bytes_per_token(&self) -> usize {
        if matches!(
            self.layout,
            KvLayout::Glm52CompressedFp8 | KvLayout::Glm52CompressedNvfp4
        ) {
            return self.dsa_indexer_layers * self.dsa_index_head_dim * std::mem::size_of::<u16>();
        }
        self.dtype
            .packed_bytes_for_elements(self.dsa_indexer_layers * self.dsa_index_head_dim)
    }

    pub fn dsa_indexer_layer_ids(&self) -> &'static [usize] {
        let layer_ids = if self.dsa_indexer_layers > GLM52_DSA_INDEXER_LAYER_IDS.len() {
            &GLM52_DSA_INDEXER_LAYER_IDS_WITH_MTP[..]
        } else {
            &GLM52_DSA_INDEXER_LAYER_IDS[..]
        };
        &layer_ids[..self.dsa_indexer_layers.min(layer_ids.len())]
    }

    pub fn layer_has_dsa_indexer(&self, layer_id: LayerId) -> bool {
        matches!(
            self.layout,
            KvLayout::Glm52CompressedBf16
                | KvLayout::Glm52CompressedFp8
                | KvLayout::Glm52CompressedNvfp4
        ) && self
            .dsa_indexer_layer_ids()
            .contains(&(layer_id.0 as usize))
    }

    pub fn layer_bytes_per_token(&self, layer_id: LayerId) -> usize {
        match self.layout {
            KvLayout::Glm52CompressedFp8 => {
                GLM52_MLA_FP8_DS_BYTES_PER_TOKEN
                    + if self.layer_has_dsa_indexer(layer_id) {
                        self.dsa_index_head_dim * std::mem::size_of::<u16>()
                    } else {
                        0
                    }
            }
            KvLayout::Glm52CompressedNvfp4 => {
                GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN
                    + if self.layer_has_dsa_indexer(layer_id) {
                        self.dsa_index_head_dim * std::mem::size_of::<u16>()
                    } else {
                        0
                    }
            }
            KvLayout::Glm52CompressedBf16 => {
                let elements = if self.layer_has_dsa_indexer(layer_id) {
                    self.key_value_width + self.dsa_index_head_dim
                } else {
                    self.key_value_width
                };
                self.dtype.packed_bytes_for_elements(elements)
            }
            KvLayout::ExpandedDebugOnly => self
                .dtype
                .packed_bytes_for_elements(2 * self.key_value_width),
        }
    }

    pub fn layer_payload_bytes(&self, layer_id: LayerId, token_count: usize) -> usize {
        self.layer_bytes_per_token(layer_id) * token_count
    }

    pub fn layer_base_offset_bytes(&self, layer_id: LayerId) -> Option<usize> {
        let layer_index = usize::try_from(layer_id.0).ok()?;
        if layer_index >= self.layers {
            return None;
        }
        let mut offset = 0_usize;
        for prior_layer in 0..layer_index {
            let layer_span = self
                .layer_bytes_per_token(LayerId(prior_layer as u32))
                .checked_mul(self.max_tokens)?;
            offset = offset.checked_add(layer_span)?;
        }
        Some(offset)
    }

    pub fn descriptor_payload_bytes(&self, descriptor: &KvBlockDescriptor) -> Option<usize> {
        let layer_index = usize::try_from(descriptor.layer_id.0).ok()?;
        if layer_index >= self.layers {
            return None;
        }
        let token_start = usize::try_from(descriptor.token_start.0).ok()?;
        let token_end = token_start.checked_add(descriptor.token_count)?;
        if token_end > self.max_tokens {
            return None;
        }
        Some(self.layer_payload_bytes(descriptor.layer_id, descriptor.token_count))
    }

    pub fn descriptor_offset_bytes(&self, descriptor: &KvBlockDescriptor) -> Option<usize> {
        self.descriptor_payload_bytes(descriptor)?;
        let layer_bytes = self.layer_bytes_per_token(descriptor.layer_id);
        let token_start = usize::try_from(descriptor.token_start.0).ok()?;
        let token_offset = layer_bytes.checked_mul(token_start)?;
        self.layer_base_offset_bytes(descriptor.layer_id)?
            .checked_add(token_offset)
    }

    pub fn bytes_per_token(&self) -> usize {
        match self.layout {
            KvLayout::Glm52CompressedBf16
            | KvLayout::Glm52CompressedFp8
            | KvLayout::Glm52CompressedNvfp4 => {
                self.main_mla_bytes_per_token()
                    + self.dsa_indexer_bytes_per_token()
                    + self.fp8_scale_metadata_bytes_per_token
            }
            KvLayout::ExpandedDebugOnly => self
                .dtype
                .packed_bytes_for_elements(self.layers * 2 * self.key_value_width),
        }
    }

    pub fn capacity_bytes(&self) -> usize {
        self.bytes_per_token() * self.max_tokens
    }
}
