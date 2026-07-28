use serde::{Deserialize, Serialize};

use super::shape::GraphBucket;
use super::work::PrefillChunk;
use crate::{LayerId, PlacementVersion, PositionId, Priority, RequestId};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefillChunkPolicy {
    pub chunk_tokens: usize,
    pub max_prefill_tokens_per_iteration: usize,
    pub max_active_prefill_chunks: usize,
    pub decode_priority: bool,
}

impl Default for PrefillChunkPolicy {
    fn default() -> Self {
        Self {
            chunk_tokens: 128,
            max_prefill_tokens_per_iteration: 512,
            max_active_prefill_chunks: 4,
            decode_priority: true,
        }
    }
}

impl PrefillChunkPolicy {
    pub fn latency_smoke(chunk_tokens: usize) -> Self {
        Self {
            chunk_tokens: chunk_tokens.max(1),
            max_prefill_tokens_per_iteration: chunk_tokens.max(1),
            max_active_prefill_chunks: 1,
            decode_priority: true,
        }
    }

    pub fn graph_bucket(&self) -> GraphBucket {
        GraphBucket::new(self.chunk_tokens.max(1))
    }
}

pub fn plan_prefill_chunks(
    request_id: impl Into<RequestId>,
    sequence_id: impl Into<String>,
    layer_id: impl Into<LayerId>,
    prompt_tokens: usize,
    kv_reservation_id: u64,
    priority: Priority,
    policy: &PrefillChunkPolicy,
    placement_version: impl Into<PlacementVersion>,
) -> Vec<PrefillChunk> {
    let request_id = request_id.into();
    let sequence_id = sequence_id.into();
    let layer_id = layer_id.into();
    let placement_version = placement_version.into();
    let chunk_tokens = policy.chunk_tokens.max(1);
    let graph_bucket = policy.graph_bucket();
    let mut chunks = Vec::new();
    let mut token_start = 0_usize;
    while token_start < prompt_tokens {
        let token_count = (prompt_tokens - token_start).min(chunk_tokens);
        chunks.push(PrefillChunk::new(
            request_id.clone(),
            sequence_id.clone(),
            layer_id,
            PositionId(token_start as u64),
            token_count,
            kv_reservation_id,
            priority,
            graph_bucket,
            placement_version.clone(),
        ));
        token_start += token_count;
    }
    chunks
}
