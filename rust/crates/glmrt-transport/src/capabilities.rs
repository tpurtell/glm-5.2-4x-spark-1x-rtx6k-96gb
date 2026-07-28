use anyhow::{bail, Result};
use glmrt_core::TransportCapabilities;
use std::path::Path;

use crate::protocol_v2::EXPERT_PROTOCOL_V2_FRAME_PROTOCOL;

use crate::DEFAULT_MAX_FRAME_BYTES;

pub const VERBS_HOST_PREFLIGHT_ONLY_PROTOCOL: &str = "glmrt-verbs-host-preflight-only-v1";
pub const VERBS_HOST_APP_TRANSPORT_STATUS: &str =
    "implemented-protocol-v2-rc-qp-send-recv-registered-host-buffers";
pub const VERBS_HOST_APP_TRANSPORT_BLOCKER: &str =
    "verbs-host app transport is implemented; run app-server/app-client on RDMA-exposed Spark hosts";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerbsHostPreflight {
    pub infiniband_path: String,
    pub frame_protocol: String,
    pub requires_pinned_host_memory: bool,
}

pub fn inproc_capabilities() -> TransportCapabilities {
    TransportCapabilities {
        name: "inproc".to_owned(),
        supports_rdma: false,
        supports_gpu_buffers: false,
        supports_host_registered_buffers: false,
        app_transport_implemented: true,
        app_transport_status: "implemented".to_owned(),
        requires_pinned_host_memory: false,
        max_message_size: DEFAULT_MAX_FRAME_BYTES,
        preferred_alignment: 64,
        measured_rtt_by_size: Vec::new(),
        measured_prefill_payload_bandwidth: Vec::new(),
    }
}

pub fn tcp_capabilities() -> TransportCapabilities {
    TransportCapabilities {
        name: "tcp".to_owned(),
        supports_rdma: false,
        supports_gpu_buffers: false,
        supports_host_registered_buffers: false,
        app_transport_implemented: true,
        app_transport_status: "implemented-debug-json-and-protocol-v2".to_owned(),
        requires_pinned_host_memory: false,
        max_message_size: DEFAULT_MAX_FRAME_BYTES,
        preferred_alignment: 64,
        measured_rtt_by_size: Vec::new(),
        measured_prefill_payload_bandwidth: Vec::new(),
    }
}

pub fn verbs_host_capabilities() -> TransportCapabilities {
    TransportCapabilities {
        name: "verbs-host".to_owned(),
        supports_rdma: true,
        supports_gpu_buffers: false,
        supports_host_registered_buffers: true,
        app_transport_implemented: true,
        app_transport_status: VERBS_HOST_APP_TRANSPORT_STATUS.to_owned(),
        requires_pinned_host_memory: true,
        max_message_size: DEFAULT_MAX_FRAME_BYTES,
        preferred_alignment: 4096,
        measured_rtt_by_size: Vec::new(),
        measured_prefill_payload_bandwidth: Vec::new(),
    }
}

pub fn verbs_host_available() -> bool {
    Path::new("/dev/infiniband").is_dir()
}

pub fn verbs_host_preflight() -> Result<VerbsHostPreflight> {
    let infiniband_path = "/dev/infiniband";
    if !Path::new(infiniband_path).is_dir() {
        bail!(
            "verbs-host transport requires RDMA host device directory {infiniband_path}; run on a Spark or native Linux host with RDMA exposed"
        );
    }
    Ok(VerbsHostPreflight {
        infiniband_path: infiniband_path.to_owned(),
        frame_protocol: EXPERT_PROTOCOL_V2_FRAME_PROTOCOL.to_owned(),
        requires_pinned_host_memory: true,
    })
}

pub fn verbs_host_app_transport_blocker() -> &'static str {
    VERBS_HOST_APP_TRANSPORT_BLOCKER
}
