use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransportRttMeasurement {
    pub payload_bytes: usize,
    pub avg_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p99_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iterations: Option<usize>,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransportPrefillBandwidthMeasurement {
    pub row_count: usize,
    pub logical_payload_bytes: usize,
    pub hops: usize,
    pub total_ms: f64,
    pub avg_ms: f64,
    pub effective_prefill_tokens_per_sec: f64,
    pub aggregate_logical_gbps: f64,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransportCapabilities {
    pub name: String,
    pub supports_rdma: bool,
    pub supports_gpu_buffers: bool,
    pub supports_host_registered_buffers: bool,
    #[serde(default)]
    pub app_transport_implemented: bool,
    #[serde(default)]
    pub app_transport_status: String,
    pub requires_pinned_host_memory: bool,
    pub max_message_size: usize,
    pub preferred_alignment: usize,
    #[serde(default)]
    pub measured_rtt_by_size: Vec<TransportRttMeasurement>,
    #[serde(default)]
    pub measured_prefill_payload_bandwidth: Vec<TransportPrefillBandwidthMeasurement>,
}
