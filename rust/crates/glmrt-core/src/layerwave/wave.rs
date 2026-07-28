use serde::{Deserialize, Serialize};

use super::kv::{prefix_kv_reads, tentative_kv_writes_for_range, KvBlockDescriptor};
use super::shape::{
    GraphBucket, HiddenShape, LayerWaveMode, RouteMetadataPlaceholder, RowSource, RowSourceKind,
};
use super::work::{DecodeStep, MtpVerifyBlock, PrefillChunk};
use crate::{GlmrtError, LayerId, PlacementVersion, PositionId, Priority, RequestId};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerWave {
    pub mode: LayerWaveMode,
    pub layer_id: LayerId,
    pub hidden_shape: HiddenShape,
    pub row_sources: Vec<RowSource>,
    pub kv_reads: Vec<KvBlockDescriptor>,
    pub kv_writes: Vec<KvBlockDescriptor>,
    #[serde(default)]
    pub tentative_kv_writes: Vec<KvBlockDescriptor>,
    pub route_metadata: RouteMetadataPlaceholder,
    pub priority: Priority,
    pub graph_bucket: GraphBucket,
    pub placement_version: PlacementVersion,
}

impl LayerWave {
    pub fn decode(step: DecodeStep) -> Self {
        let mut kv_reads = Vec::new();
        let mut kv_writes = Vec::new();
        if let Some(reservation_id) = step.kv_reservation_id {
            if step.position.0 > 0 {
                kv_reads.push(KvBlockDescriptor {
                    reservation_id,
                    sequence_id: step.sequence_id.clone(),
                    layer_id: step.layer_id,
                    token_start: PositionId(0),
                    token_count: step.position.0 as usize,
                });
            }
            kv_writes.push(KvBlockDescriptor {
                reservation_id,
                sequence_id: step.sequence_id.clone(),
                layer_id: step.layer_id,
                token_start: step.position,
                token_count: 1,
            });
        }
        Self {
            mode: LayerWaveMode::Decode,
            layer_id: step.layer_id,
            hidden_shape: HiddenShape::glm52_bf16_rows(1),
            row_sources: vec![RowSource {
                kind: RowSourceKind::DecodeStep,
                request_id: step.request_id,
                sequence_id: step.sequence_id,
                token_start: step.position,
                row_count: 1,
            }],
            kv_reads,
            kv_writes,
            tentative_kv_writes: Vec::new(),
            route_metadata: RouteMetadataPlaceholder::default(),
            priority: step.priority,
            graph_bucket: step.graph_bucket,
            placement_version: step.placement_version,
        }
    }

    pub fn prefill(chunk: PrefillChunk) -> Self {
        let kv_reads = prefix_kv_reads(
            chunk.kv_write.reservation_id,
            &chunk.sequence_id,
            chunk.layer_id,
            chunk.token_start,
        );
        Self {
            mode: LayerWaveMode::Prefill,
            layer_id: chunk.layer_id,
            hidden_shape: chunk.hidden_shape,
            row_sources: vec![RowSource {
                kind: RowSourceKind::PrefillChunk,
                request_id: chunk.request_id,
                sequence_id: chunk.sequence_id,
                token_start: chunk.token_start,
                row_count: chunk.token_count,
            }],
            kv_reads,
            kv_writes: vec![chunk.kv_write],
            tentative_kv_writes: Vec::new(),
            route_metadata: RouteMetadataPlaceholder::default(),
            priority: chunk.priority,
            graph_bucket: chunk.graph_bucket,
            placement_version: chunk.placement_version,
        }
    }

    pub fn mtp_verify(block: MtpVerifyBlock) -> Self {
        let mut kv_reads = Vec::new();
        let mut tentative_kv_writes = Vec::new();
        if let Some(reservation_id) = block.kv_reservation_id {
            kv_reads.extend(prefix_kv_reads(
                reservation_id,
                &block.sequence_id,
                block.layer_id,
                block.token_start,
            ));
            tentative_kv_writes.extend(tentative_kv_writes_for_range(
                reservation_id,
                &block.sequence_id,
                block.layer_id,
                block.token_start,
                block.token_count,
            ));
        }
        Self {
            mode: LayerWaveMode::MtpVerify,
            layer_id: block.layer_id,
            hidden_shape: HiddenShape::glm52_bf16_rows(block.token_count),
            row_sources: vec![RowSource {
                kind: RowSourceKind::MtpVerifyBlock,
                request_id: block.request_id,
                sequence_id: block.sequence_id,
                token_start: block.token_start,
                row_count: block.token_count,
            }],
            kv_reads,
            kv_writes: Vec::new(),
            tentative_kv_writes,
            route_metadata: RouteMetadataPlaceholder::default(),
            priority: block.priority,
            graph_bucket: block.graph_bucket,
            placement_version: block.placement_version,
        }
    }

    pub fn benchmark(
        layer_id: impl Into<LayerId>,
        rows: usize,
        graph_bucket: GraphBucket,
        placement_version: impl Into<PlacementVersion>,
    ) -> Self {
        Self {
            mode: LayerWaveMode::Benchmark,
            layer_id: layer_id.into(),
            hidden_shape: HiddenShape::glm52_bf16_rows(rows),
            row_sources: vec![RowSource {
                kind: RowSourceKind::Benchmark,
                request_id: RequestId("benchmark".to_owned()),
                sequence_id: "benchmark".to_owned(),
                token_start: PositionId(0),
                row_count: rows,
            }],
            kv_reads: Vec::new(),
            kv_writes: Vec::new(),
            tentative_kv_writes: Vec::new(),
            route_metadata: RouteMetadataPlaceholder::default(),
            priority: Priority(0),
            graph_bucket,
            placement_version: placement_version.into(),
        }
    }

    pub fn num_rows(&self) -> usize {
        self.hidden_shape.rows
    }

    pub fn payload_bytes_per_direction(&self) -> usize {
        self.hidden_shape.payload_bytes()
    }

    pub fn roundtrip_bytes_per_host(&self) -> usize {
        self.payload_bytes_per_direction() * 2
    }

    pub fn routed_expert_assignments(&self) -> usize {
        self.num_rows() * self.route_metadata.top_k
    }

    pub fn average_rows_per_expert(&self) -> f64 {
        self.routed_expert_assignments() as f64 / self.route_metadata.routed_experts as f64
    }

    pub fn can_mix_with(&self, other: &Self) -> bool {
        self.mix_rejection_reason(other).is_none()
    }

    pub fn try_merge(&self, other: &Self) -> Result<Self, GlmrtError> {
        if let Some(reason) = self.mix_rejection_reason(other) {
            return Err(GlmrtError::LayerWaveMixRejected { reason });
        }
        let mut merged = self.clone();
        merged.hidden_shape.rows += other.hidden_shape.rows;
        merged.row_sources.extend(other.row_sources.clone());
        merged.kv_reads.extend(other.kv_reads.clone());
        merged.kv_writes.extend(other.kv_writes.clone());
        merged
            .tentative_kv_writes
            .extend(other.tentative_kv_writes.clone());
        merged.priority = Priority(self.priority.0.min(other.priority.0));
        Ok(merged)
    }

    fn mix_rejection_reason(&self, other: &Self) -> Option<String> {
        if self.mode != other.mode {
            return Some(format!(
                "different modes {:?} and {:?}",
                self.mode, other.mode
            ));
        }
        if self.layer_id != other.layer_id {
            return Some(format!(
                "different layers {} and {}",
                self.layer_id.0, other.layer_id.0
            ));
        }
        if self.graph_bucket != other.graph_bucket {
            return Some(format!(
                "different graph buckets {} and {}",
                self.graph_bucket.row_capacity, other.graph_bucket.row_capacity
            ));
        }
        if self.placement_version != other.placement_version {
            return Some("different placement versions".to_owned());
        }
        if self.hidden_shape.hidden_dim != other.hidden_shape.hidden_dim
            || self.hidden_shape.bytes_per_row != other.hidden_shape.bytes_per_row
        {
            return Some("different hidden shapes".to_owned());
        }
        if self.route_metadata != other.route_metadata {
            return Some("different route metadata".to_owned());
        }
        let merged_rows = self.num_rows() + other.num_rows();
        if merged_rows > self.graph_bucket.row_capacity {
            return Some(format!(
                "merged rows {merged_rows} exceed graph bucket capacity {}",
                self.graph_bucket.row_capacity
            ));
        }
        None
    }
}
