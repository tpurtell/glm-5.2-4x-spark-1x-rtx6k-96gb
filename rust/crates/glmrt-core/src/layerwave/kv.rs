use serde::{Deserialize, Serialize};

use crate::{LayerId, PositionId};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvBlockDescriptor {
    pub reservation_id: u64,
    pub sequence_id: String,
    pub layer_id: LayerId,
    pub token_start: PositionId,
    pub token_count: usize,
}

pub(super) fn prefix_kv_reads(
    reservation_id: u64,
    sequence_id: &str,
    layer_id: LayerId,
    token_start: PositionId,
) -> Vec<KvBlockDescriptor> {
    if token_start.0 == 0 {
        return Vec::new();
    }
    vec![KvBlockDescriptor {
        reservation_id,
        sequence_id: sequence_id.to_owned(),
        layer_id,
        token_start: PositionId(0),
        token_count: token_start.0 as usize,
    }]
}

pub(super) fn tentative_kv_writes_for_range(
    reservation_id: u64,
    sequence_id: &str,
    layer_id: LayerId,
    token_start: PositionId,
    token_count: usize,
) -> Vec<KvBlockDescriptor> {
    (0..token_count)
        .map(|offset| KvBlockDescriptor {
            reservation_id,
            sequence_id: sequence_id.to_owned(),
            layer_id,
            token_start: PositionId(token_start.0 + offset as u64),
            token_count: 1,
        })
        .collect()
}
