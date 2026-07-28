use serde::{Deserialize, Serialize};

use crate::{
    PositionId, RequestId, GLM52_HIDDEN_BF16_BYTES, GLM52_HIDDEN_SIZE, GLM52_ROUTED_EXPERTS,
    GLM52_TOP_K,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LayerWaveMode {
    Decode,
    Prefill,
    MtpVerify,
    Benchmark,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GraphBucket {
    pub row_capacity: usize,
}

impl GraphBucket {
    pub fn new(row_capacity: usize) -> Self {
        Self {
            row_capacity: row_capacity.max(1),
        }
    }

    pub fn decode() -> Self {
        Self::new(1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HiddenShape {
    pub rows: usize,
    pub hidden_dim: usize,
    pub bytes_per_row: usize,
}

impl HiddenShape {
    pub fn glm52_bf16_rows(rows: usize) -> Self {
        Self {
            rows,
            hidden_dim: GLM52_HIDDEN_SIZE,
            bytes_per_row: GLM52_HIDDEN_BF16_BYTES,
        }
    }

    pub fn payload_bytes(self) -> usize {
        self.rows * self.bytes_per_row
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RowSourceKind {
    DecodeStep,
    PrefillChunk,
    MtpVerifyBlock,
    Benchmark,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowSource {
    pub kind: RowSourceKind,
    pub request_id: RequestId,
    pub sequence_id: String,
    pub token_start: PositionId,
    pub row_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteMetadataPlaceholder {
    pub top_k: usize,
    pub routed_experts: usize,
}

impl Default for RouteMetadataPlaceholder {
    fn default() -> Self {
        Self {
            top_k: GLM52_TOP_K,
            routed_experts: GLM52_ROUTED_EXPERTS,
        }
    }
}
