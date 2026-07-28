use crate::{LayerWave, LayerWaveMode};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertRequestHeader {
    pub protocol_version: u32,
    pub request_id: u64,
    pub placement_version: String,
    pub layer_id: u32,
    pub hidden_dim: u32,
    pub row_count: u32,
    pub wave_mode: Option<LayerWaveMode>,
    pub graph_bucket_rows: Option<u32>,
    pub logical_bf16_payload_bytes: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteEntry {
    pub expert_id: u32,
    pub gate: f32,
}

/// Debug/integration expert row with host-side f32 hidden values.
///
/// The production expert wire path should use the binary ProtocolV2 transport
/// representation rather than serializing this serde shape directly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertRow {
    pub row_id: u64,
    pub hidden: Vec<f32>,
    pub routes: Vec<RouteEntry>,
}

/// Debug/integration expert request shape for local tests and compatibility
/// dispatch paths. It is intentionally distinct from the binary ProtocolV2
/// expert transport contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertRequest {
    pub protocol_version: u32,
    pub request_id: u64,
    pub placement_version: String,
    pub layer_id: u32,
    pub hidden_dim: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wave: Option<ExpertWaveMetadata>,
    pub rows: Vec<ExpertRow>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertWaveMetadata {
    pub mode: LayerWaveMode,
    pub graph_bucket_rows: u32,
    pub logical_bf16_payload_bytes: usize,
}

impl ExpertWaveMetadata {
    pub fn from_wave(wave: &LayerWave) -> Self {
        Self {
            mode: wave.mode,
            graph_bucket_rows: wave.graph_bucket.row_capacity as u32,
            logical_bf16_payload_bytes: wave.payload_bytes_per_direction(),
        }
    }
}

impl ExpertRequest {
    pub fn header(&self) -> ExpertRequestHeader {
        ExpertRequestHeader {
            protocol_version: self.protocol_version,
            request_id: self.request_id,
            placement_version: self.placement_version.clone(),
            layer_id: self.layer_id,
            hidden_dim: self.hidden_dim,
            row_count: self.rows.len() as u32,
            wave_mode: self.wave.as_ref().map(|wave| wave.mode),
            graph_bucket_rows: self.wave.as_ref().map(|wave| wave.graph_bucket_rows),
            logical_bf16_payload_bytes: self
                .wave
                .as_ref()
                .map(|wave| wave.logical_bf16_payload_bytes),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertResponseHeader {
    pub request_id: u64,
    pub placement_version: String,
    pub layer_id: u32,
    pub status: String,
    pub row_count: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertResponse {
    pub request_id: u64,
    pub placement_version: String,
    pub layer_id: u32,
    pub status: String,
    pub partial_outputs: Vec<Vec<f32>>,
}

impl ExpertResponse {
    pub fn header(&self) -> ExpertResponseHeader {
        ExpertResponseHeader {
            request_id: self.request_id,
            placement_version: self.placement_version.clone(),
            layer_id: self.layer_id,
            status: self.status.clone(),
            row_count: self.partial_outputs.len() as u32,
        }
    }
}
