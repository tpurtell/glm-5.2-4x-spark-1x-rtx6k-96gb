use serde::{Deserialize, Serialize};

use super::kv::KvBlockDescriptor;
use super::shape::{GraphBucket, HiddenShape};
use crate::{LayerId, PlacementVersion, PositionId, Priority, RequestId};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecodeStep {
    pub request_id: RequestId,
    pub sequence_id: String,
    pub layer_id: LayerId,
    pub position: PositionId,
    pub kv_reservation_id: Option<u64>,
    pub priority: Priority,
    pub graph_bucket: GraphBucket,
    pub placement_version: PlacementVersion,
}

impl DecodeStep {
    pub fn new(
        request_id: impl Into<RequestId>,
        sequence_id: impl Into<String>,
        layer_id: impl Into<LayerId>,
        position: impl Into<PositionId>,
        kv_reservation_id: Option<u64>,
        priority: Priority,
        placement_version: impl Into<PlacementVersion>,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            sequence_id: sequence_id.into(),
            layer_id: layer_id.into(),
            position: position.into(),
            kv_reservation_id,
            priority,
            graph_bucket: GraphBucket::decode(),
            placement_version: placement_version.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefillChunk {
    pub request_id: RequestId,
    pub sequence_id: String,
    pub layer_id: LayerId,
    pub token_start: PositionId,
    pub token_count: usize,
    pub hidden_shape: HiddenShape,
    pub kv_write: KvBlockDescriptor,
    pub priority: Priority,
    pub graph_bucket: GraphBucket,
    pub placement_version: PlacementVersion,
}

impl PrefillChunk {
    pub fn new(
        request_id: impl Into<RequestId>,
        sequence_id: impl Into<String>,
        layer_id: impl Into<LayerId>,
        token_start: impl Into<PositionId>,
        token_count: usize,
        kv_reservation_id: u64,
        priority: Priority,
        graph_bucket: GraphBucket,
        placement_version: impl Into<PlacementVersion>,
    ) -> Self {
        let request_id = request_id.into();
        let sequence_id = sequence_id.into();
        let layer_id = layer_id.into();
        let token_start = token_start.into();
        Self {
            request_id,
            sequence_id: sequence_id.clone(),
            layer_id,
            token_start,
            token_count,
            hidden_shape: HiddenShape::glm52_bf16_rows(token_count),
            kv_write: KvBlockDescriptor {
                reservation_id: kv_reservation_id,
                sequence_id,
                layer_id,
                token_start,
                token_count,
            },
            priority,
            graph_bucket,
            placement_version: placement_version.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MtpVerifyBlock {
    pub request_id: RequestId,
    pub sequence_id: String,
    pub layer_id: LayerId,
    pub token_start: PositionId,
    pub token_count: usize,
    pub kv_reservation_id: Option<u64>,
    pub priority: Priority,
    pub graph_bucket: GraphBucket,
    pub placement_version: PlacementVersion,
}

impl MtpVerifyBlock {
    pub fn new(
        request_id: impl Into<RequestId>,
        sequence_id: impl Into<String>,
        layer_id: impl Into<LayerId>,
        token_start: impl Into<PositionId>,
        token_count: usize,
        kv_reservation_id: Option<u64>,
        priority: Priority,
        graph_bucket: GraphBucket,
        placement_version: impl Into<PlacementVersion>,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            sequence_id: sequence_id.into(),
            layer_id: layer_id.into(),
            token_start: token_start.into(),
            token_count,
            kv_reservation_id,
            priority,
            graph_bucket,
            placement_version: placement_version.into(),
        }
    }
}
