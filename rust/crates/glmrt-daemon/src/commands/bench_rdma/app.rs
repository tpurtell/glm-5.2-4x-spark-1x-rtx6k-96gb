use anyhow::{Context, Result};
use glmrt_core::{GLM52_FIRST_K_DENSE_REPLACE, GLM52_HIDDEN_SIZE, GLM52_NUM_HIDDEN_LAYERS};
use glmrt_ffi::{
    c_char_array_to_string, GlmrtRdmaRcCompletionStats, GlmrtRdmaRcEndpointInfo, NativeLibrary,
    GLMRT_STATUS_RDMA_UNAVAILABLE,
};
use glmrt_transport::{
    protocol_v2_synthetic_response, verbs_host_protocol_v2_endpoint_plan,
    verbs_host_protocol_v2_handshake_contract, verbs_host_protocol_v2_round_trip_plan,
    verbs_host_validate_protocol_v2_handshake, ExpertProtocolV2FrameArena, ExpertProtocolV2Request,
    ExpertProtocolV2RequestView, ExpertProtocolV2Response, ExpertProtocolV2ResponseView,
    ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor, ExpertV2Dtype, ExpertV2SourceKind,
    VerbsHostProtocolV2EndpointPlan, VerbsHostProtocolV2HandshakeContract,
    VerbsHostProtocolV2HandshakeValidation, VerbsHostProtocolV2RoundTripPlan,
    VerbsHostRcEndpointDescriptor,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::env;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

const APP_RECV_WR_ID: u64 = 0x7256_0001;
const APP_SEND_WR_ID: u64 = 0x7256_0002;

#[derive(Debug, Serialize)]
pub(super) struct GlmrtVerbsAppBenchmarkAttempt {
    pub(super) attempted: bool,
    pub(super) transport: String,
    pub(super) protocol: String,
    pub(super) resolved_mode: String,
    pub(super) peer: Option<String>,
    pub(super) control_plane: String,
    pub(super) data_plane: String,
    pub(super) memory: String,
    pub(super) polling: String,
    pub(super) uses_reusable_protocol_v2_frame_buffers: bool,
    pub(super) uses_protocol_v2_frame_arenas: bool,
    pub(super) protocol_v2_endpoint_plan: VerbsHostProtocolV2EndpointPlan,
    pub(super) protocol_v2_handshake_contract: VerbsHostProtocolV2HandshakeContract,
    pub(super) protocol_v2_control_plane_dry_run: GlmrtVerbsAppControlPlaneDryRun,
    pub(super) protocol_v2_round_trip_plan: VerbsHostProtocolV2RoundTripPlan,
    pub(super) app_transport_implemented: bool,
    pub(super) app_transport_blocker: String,
    pub(super) preflight_ok: bool,
    pub(super) preflight_error: Option<String>,
    pub(super) app_transport_runs: Vec<GlmrtVerbsAppRun>,
    pub(super) app_transport_error: Option<String>,
    pub(super) payloads: Vec<GlmrtVerbsAppPayload>,
    pub(super) chain_hops: usize,
    pub(super) protocol_v2_chain_plans: Vec<GlmrtVerbsAppChainPlan>,
    pub(super) native_rdma_probe: GlmrtVerbsNativeRdmaProbe,
}

#[derive(Debug, Serialize)]
pub(super) struct GlmrtVerbsAppPayload {
    pub(super) row_count: usize,
    pub(super) source_kind: String,
    pub(super) request_logical_payload_bytes: usize,
    pub(super) request_wire_bytes: usize,
    pub(super) response_logical_payload_bytes: usize,
    pub(super) response_wire_bytes: usize,
    pub(super) total_logical_payload_bytes: usize,
    pub(super) total_wire_bytes: usize,
    pub(super) request_frame_buffer_capacity_bytes: usize,
    pub(super) response_frame_buffer_capacity_bytes: usize,
    pub(super) request_frame_buffer_stable: bool,
    pub(super) response_frame_buffer_stable: bool,
    pub(super) request_frame_arena_capacity_bytes: usize,
    pub(super) response_frame_arena_capacity_bytes: usize,
    pub(super) request_frame_arena_stable: bool,
    pub(super) response_frame_arena_stable: bool,
    pub(super) frame_arena_registration_alignment_bytes: usize,
    pub(super) request_registered_span_bytes: usize,
    pub(super) response_registered_span_bytes: usize,
    pub(super) total_registered_span_bytes: usize,
    pub(super) request_registration_slack_bytes: usize,
    pub(super) response_registration_slack_bytes: usize,
    pub(super) request_registered_span_aligned: bool,
    pub(super) response_registered_span_aligned: bool,
    pub(super) request_hidden_row_view_count: usize,
    pub(super) response_partial_output_row_view_count: usize,
    pub(super) request_hidden_row_view_payload_bytes: usize,
    pub(super) response_partial_output_row_view_payload_bytes: usize,
    pub(super) request_row_views_cover_payload: bool,
    pub(super) response_row_views_cover_payload: bool,
    pub(super) response_generated_by_route_dependent_executor: bool,
    pub(super) response_differs_from_request_payload: bool,
    pub(super) host_memory_registration_required: bool,
}

#[derive(Debug, Serialize)]
pub(super) struct GlmrtVerbsAppChainPlan {
    pub(super) chain_kind: String,
    pub(super) row_count: usize,
    pub(super) source_kind: String,
    pub(super) hops: usize,
    pub(super) request_logical_payload_bytes_per_hop: usize,
    pub(super) response_logical_payload_bytes_per_hop: usize,
    pub(super) request_wire_bytes_per_hop: usize,
    pub(super) response_wire_bytes_per_hop: usize,
    pub(super) registered_span_bytes_per_hop: usize,
    pub(super) total_request_wire_bytes: usize,
    pub(super) total_response_wire_bytes: usize,
    pub(super) total_wire_bytes: usize,
    pub(super) total_logical_payload_bytes: usize,
    pub(super) total_registered_span_bytes: usize,
    pub(super) uses_registered_frame_spans: bool,
    pub(super) uses_request_response_row_views: bool,
}

#[derive(Debug, Serialize)]
pub(super) struct GlmrtVerbsAppControlPlaneDryRun {
    pub(super) app_role: String,
    pub(super) descriptor_exchange: String,
    pub(super) local_endpoint_role: String,
    pub(super) peer_endpoint_required: bool,
    pub(super) peer_endpoint_host: String,
    pub(super) client_endpoint: VerbsHostRcEndpointDescriptor,
    pub(super) server_endpoint: VerbsHostRcEndpointDescriptor,
    pub(super) validation: VerbsHostProtocolV2HandshakeValidation,
    pub(super) validates_peer_qp_psn_gid: bool,
    pub(super) validates_registered_frame_spans: bool,
    pub(super) data_plane_attempted: bool,
    pub(super) data_plane_blocker: String,
}

#[derive(Debug, Serialize)]
pub(super) struct GlmrtVerbsNativeRdmaProbe {
    pub(super) attempted: bool,
    pub(super) native_library_path: Option<String>,
    pub(super) native_library_loaded: bool,
    pub(super) native_library_error: Option<String>,
    pub(super) rdma_enabled: Option<bool>,
    pub(super) rdma_device_count: Option<i32>,
    pub(super) first_device_openable: Option<bool>,
    pub(super) first_device_name: Option<String>,
    pub(super) first_device_status: Option<String>,
    pub(super) host_buffer_plan_checked: bool,
    pub(super) host_buffer_plan_registered_span_bytes: Option<usize>,
    pub(super) host_buffer_plan_span_aligned: Option<bool>,
    pub(super) rc_qp_probe_ok: bool,
    pub(super) rc_qp_probe_unavailable: bool,
    pub(super) rc_qp_probe_error: Option<String>,
    pub(super) rc_send_recv_loopback_ok: bool,
    pub(super) rc_send_recv_loopback_unavailable: bool,
    pub(super) rc_send_recv_loopback_error: Option<String>,
    pub(super) rc_send_recv_loopback_bytes: usize,
    pub(super) rc_send_recv_payload_matches: Option<bool>,
    pub(super) rc_send_recv_send_completions: Option<u32>,
    pub(super) rc_send_recv_recv_completions: Option<u32>,
    pub(super) rc_send_recv_poll_iterations: Option<u32>,
    pub(super) rc_protocol_v2_loopback_ok: bool,
    pub(super) rc_protocol_v2_loopback_unavailable: bool,
    pub(super) rc_protocol_v2_loopback_error: Option<String>,
    pub(super) rc_protocol_v2_request_frame_bytes: usize,
    pub(super) rc_protocol_v2_response_frame_bytes: usize,
    pub(super) rc_protocol_v2_request_payload_matches: Option<bool>,
    pub(super) rc_protocol_v2_response_payload_matches: Option<bool>,
    pub(super) rc_protocol_v2_send_completions: Option<u32>,
    pub(super) rc_protocol_v2_recv_completions: Option<u32>,
    pub(super) rc_protocol_v2_poll_iterations: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct GlmrtVerbsAppNativeEndpointDescriptor {
    pub(super) port_num: u32,
    pub(super) qp_num: u32,
    pub(super) psn: u32,
    pub(super) lid: u32,
    pub(super) active_mtu: u32,
    pub(super) gid_hex: String,
    pub(super) send_frame_bytes: usize,
    pub(super) recv_frame_bytes: usize,
    pub(super) send_registered_span_bytes: usize,
    pub(super) recv_registered_span_bytes: usize,
    pub(super) max_send_wr: u32,
    pub(super) max_recv_wr: u32,
    pub(super) max_sge: u32,
    pub(super) device_name: String,
    pub(super) status: String,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct GlmrtVerbsAppRun {
    pub(super) run_kind: String,
    pub(super) role: String,
    pub(super) peer: String,
    pub(super) ok: bool,
    pub(super) payload_index: usize,
    pub(super) row_count: usize,
    pub(super) source_kind: String,
    pub(super) request_wire_bytes: usize,
    pub(super) response_wire_bytes: usize,
    pub(super) request_logical_payload_bytes: usize,
    pub(super) response_logical_payload_bytes: usize,
    pub(super) request_registered_span_bytes: usize,
    pub(super) response_registered_span_bytes: usize,
    pub(super) iterations: usize,
    pub(super) hops_per_iteration: usize,
    pub(super) roundtrips: usize,
    pub(super) elapsed_micros: u128,
    pub(super) roundtrip_latency_micros_avg: Option<f64>,
    pub(super) roundtrip_latency_micros_min: Option<f64>,
    pub(super) roundtrip_latency_micros_max: Option<f64>,
    pub(super) sample_latency_micros_avg: Option<f64>,
    pub(super) effective_roundtrips_per_second: f64,
    pub(super) effective_payload_gbps: f64,
    pub(super) request_payload_matches: bool,
    pub(super) response_payload_matches: bool,
    pub(super) send_completions: u64,
    pub(super) recv_completions: u64,
    pub(super) poll_iterations: u64,
    pub(super) local_endpoint: GlmrtVerbsAppNativeEndpointDescriptor,
    pub(super) peer_endpoint: GlmrtVerbsAppNativeEndpointDescriptor,
}

#[derive(Debug, Serialize, Deserialize)]
struct GlmrtVerbsAppRunStart {
    message: String,
    run_kind: String,
    payload_index: usize,
    row_count: usize,
    source_kind: String,
    request_wire_bytes: usize,
    response_wire_bytes: usize,
    request_logical_payload_bytes: usize,
    response_logical_payload_bytes: usize,
    request_registered_span_bytes: usize,
    response_registered_span_bytes: usize,
    iterations: usize,
    hops_per_iteration: usize,
    client_endpoint: VerbsHostRcEndpointDescriptor,
    client_native_endpoint: GlmrtVerbsAppNativeEndpointDescriptor,
}

#[derive(Debug, Serialize, Deserialize)]
struct GlmrtVerbsAppRunReady {
    message: String,
    payload_index: usize,
    server_endpoint: VerbsHostRcEndpointDescriptor,
    server_native_endpoint: GlmrtVerbsAppNativeEndpointDescriptor,
}

#[derive(Debug, Serialize, Deserialize)]
struct GlmrtVerbsAppRecvReady {
    message: String,
    payload_index: usize,
}

pub(super) fn glmrt_verbs_app_benchmark_attempt(
    resolved_mode: &str,
    peer: Option<String>,
    payload_bytes: &[usize],
) -> Result<GlmrtVerbsAppBenchmarkAttempt> {
    let payloads = protocol_v2_payloads_for_benchmark(payload_bytes)?;
    let chain_hops = glm52_sparse_moe_chain_hops();
    let protocol_v2_chain_plans = protocol_v2_chain_plans_for_payloads(&payloads, chain_hops)?;
    let native_probe_rows = payloads
        .first()
        .map(|payload| payload.row_count)
        .unwrap_or(1);
    let (_native_source_kind, native_request, native_response) =
        protocol_v2_request_response_for_row_count(1, native_probe_rows)?;
    let native_request_frame = native_request.encode()?;
    let native_response_frame = native_response.encode()?;
    let capabilities = glmrt_transport::verbs_host_capabilities();
    let protocol_v2_endpoint_plan = verbs_host_protocol_v2_endpoint_plan(
        &native_request_frame,
        &native_response_frame,
        capabilities.preferred_alignment,
    )?;
    let protocol_v2_handshake_contract =
        verbs_host_protocol_v2_handshake_contract(&protocol_v2_endpoint_plan);
    let protocol_v2_control_plane_dry_run = protocol_v2_control_plane_dry_run(
        resolved_mode,
        peer.as_deref(),
        &protocol_v2_endpoint_plan,
    )?;
    let protocol_v2_round_trip_plan = verbs_host_protocol_v2_round_trip_plan(
        &protocol_v2_endpoint_plan,
        &protocol_v2_control_plane_dry_run.validation,
        &native_request_frame,
        &native_response_frame,
    )?;
    let native_probe_bytes = native_request.wire_stats().logical_payload_bytes;
    let native_rdma_probe = native_rdma_probe(
        native_probe_bytes,
        &native_request_frame,
        &native_response_frame,
    );
    let preflight = glmrt_transport::verbs_host_preflight();
    let (preflight_ok, preflight_error) = match preflight {
        Ok(_) => (true, None),
        Err(err) => (false, Some(err.to_string())),
    };
    Ok(GlmrtVerbsAppBenchmarkAttempt {
        attempted: true,
        transport: "verbs-host".to_owned(),
        protocol: "ExpertProtocolV2".to_owned(),
        resolved_mode: resolved_mode.to_owned(),
        peer,
        control_plane: "tcp-qp-gid-psn-handshake".to_owned(),
        data_plane: "rc-qp-send-recv".to_owned(),
        memory: "registered-host-buffers".to_owned(),
        polling: "busy-poll-cq".to_owned(),
        uses_reusable_protocol_v2_frame_buffers: payloads.iter().all(|payload| {
            payload.request_frame_buffer_stable && payload.response_frame_buffer_stable
        }),
        uses_protocol_v2_frame_arenas: payloads.iter().all(|payload| {
            payload.request_frame_arena_stable && payload.response_frame_arena_stable
        }),
        protocol_v2_endpoint_plan,
        protocol_v2_handshake_contract,
        protocol_v2_control_plane_dry_run,
        protocol_v2_round_trip_plan,
        app_transport_implemented: capabilities.app_transport_implemented,
        app_transport_blocker: capabilities.app_transport_status,
        preflight_ok,
        preflight_error,
        app_transport_runs: Vec::new(),
        app_transport_error: None,
        payloads,
        chain_hops,
        protocol_v2_chain_plans,
        native_rdma_probe,
    })
}

fn protocol_v2_control_plane_dry_run(
    resolved_mode: &str,
    peer: Option<&str>,
    endpoint_plan: &VerbsHostProtocolV2EndpointPlan,
) -> Result<GlmrtVerbsAppControlPlaneDryRun> {
    let app_role = app_role_for_resolved_mode(resolved_mode)?;
    let (local_endpoint_role, peer_endpoint_required) = match app_role {
        "server" => ("server", true),
        "client" => ("client", true),
        "capability" => ("none", false),
        other => anyhow::bail!("unsupported GLMRT verbs app role: {other}"),
    };
    let (client_host, server_host) = match app_role {
        "server" => (
            peer.filter(|value| !value.trim().is_empty())
                .unwrap_or("client-peer")
                .to_owned(),
            "local-server".to_owned(),
        ),
        "client" => (
            "local-client".to_owned(),
            peer.filter(|value| !value.trim().is_empty())
                .unwrap_or("server-peer")
                .to_owned(),
        ),
        "capability" => ("client-peer".to_owned(), "server-peer".to_owned()),
        other => anyhow::bail!("unsupported GLMRT verbs app role: {other}"),
    };
    let (client_host, server_host) = distinct_endpoint_hosts(client_host, server_host);
    let client_endpoint = protocol_v2_rc_endpoint_descriptor(
        "client",
        &client_host,
        0x010001,
        0x00abc1,
        "00000000000000000000000000000001",
        endpoint_plan,
    );
    let server_endpoint = protocol_v2_rc_endpoint_descriptor(
        "server",
        &server_host,
        0x020002,
        0x00abc2,
        "00000000000000000000000000000002",
        endpoint_plan,
    );
    let validation = verbs_host_validate_protocol_v2_handshake(
        endpoint_plan,
        &client_endpoint,
        &server_endpoint,
    )?;
    let peer_endpoint_host = match app_role {
        "server" => validation.client_host.clone(),
        "client" => validation.server_host.clone(),
        "capability" => format!("{}<->{}", validation.client_host, validation.server_host),
        other => anyhow::bail!("unsupported GLMRT verbs app role: {other}"),
    };
    let validates_peer_qp_psn_gid = validation.peer_qp_num_present
        && validation.peer_psn_present
        && validation.peer_gid_present;
    let validates_registered_frame_spans = validation.registered_spans_match_endpoint_plan;

    Ok(GlmrtVerbsAppControlPlaneDryRun {
        app_role: app_role.to_owned(),
        descriptor_exchange: "tcp-json-rc-endpoint-descriptor-dry-run".to_owned(),
        local_endpoint_role: local_endpoint_role.to_owned(),
        peer_endpoint_required,
        peer_endpoint_host,
        client_endpoint,
        server_endpoint,
        validation,
        validates_peer_qp_psn_gid,
        validates_registered_frame_spans,
        data_plane_attempted: false,
        data_plane_blocker: glmrt_transport::verbs_host_app_transport_blocker().to_owned(),
    })
}

fn app_role_for_resolved_mode(resolved_mode: &str) -> Result<&'static str> {
    match resolved_mode {
        "app-server" => Ok("server"),
        "app-client" => Ok("client"),
        "app-capability" => Ok("capability"),
        other => anyhow::bail!("unsupported GLMRT verbs app resolved mode: {other}"),
    }
}

fn distinct_endpoint_hosts(mut client_host: String, mut server_host: String) -> (String, String) {
    if client_host == server_host {
        client_host.push_str("-client");
        server_host.push_str("-server");
    }
    (client_host, server_host)
}

fn protocol_v2_rc_endpoint_descriptor(
    role: &str,
    host: &str,
    qp_num: u32,
    psn: u32,
    gid_hex: &str,
    endpoint_plan: &VerbsHostProtocolV2EndpointPlan,
) -> VerbsHostRcEndpointDescriptor {
    let (
        send_frame_bytes,
        recv_frame_bytes,
        send_registered_span_bytes,
        recv_registered_span_bytes,
    ) = match role {
        "client" => (
            endpoint_plan.request_frame_bytes,
            endpoint_plan.response_frame_bytes,
            endpoint_plan.request_registered_span_bytes,
            endpoint_plan.response_registered_span_bytes,
        ),
        "server" => (
            endpoint_plan.response_frame_bytes,
            endpoint_plan.request_frame_bytes,
            endpoint_plan.response_registered_span_bytes,
            endpoint_plan.request_registered_span_bytes,
        ),
        _ => unreachable!("control-plane dry-run only creates client/server descriptors"),
    };
    VerbsHostRcEndpointDescriptor {
        role: role.to_owned(),
        host: host.to_owned(),
        port_num: 1,
        qp_num,
        psn,
        gid_hex: gid_hex.to_owned(),
        send_frame_bytes,
        recv_frame_bytes,
        send_registered_span_bytes,
        recv_registered_span_bytes,
        max_send_wr: endpoint_plan.send_work_requests_per_roundtrip as u32,
        max_recv_wr: endpoint_plan.recv_work_requests_per_roundtrip as u32,
        max_sge: endpoint_plan.scatter_gather_entries_per_message as u32,
    }
}

fn native_rdma_probe(
    loopback_bytes: usize,
    request_frame: &[u8],
    response_frame: &[u8],
) -> GlmrtVerbsNativeRdmaProbe {
    let native_library_path = native_library_path();
    let mut probe = GlmrtVerbsNativeRdmaProbe {
        attempted: true,
        native_library_path: native_library_path
            .as_ref()
            .map(|path| path.display().to_string()),
        native_library_loaded: false,
        native_library_error: None,
        rdma_enabled: None,
        rdma_device_count: None,
        first_device_openable: None,
        first_device_name: None,
        first_device_status: None,
        host_buffer_plan_checked: false,
        host_buffer_plan_registered_span_bytes: None,
        host_buffer_plan_span_aligned: None,
        rc_qp_probe_ok: false,
        rc_qp_probe_unavailable: false,
        rc_qp_probe_error: None,
        rc_send_recv_loopback_ok: false,
        rc_send_recv_loopback_unavailable: false,
        rc_send_recv_loopback_error: None,
        rc_send_recv_loopback_bytes: loopback_bytes,
        rc_send_recv_payload_matches: None,
        rc_send_recv_send_completions: None,
        rc_send_recv_recv_completions: None,
        rc_send_recv_poll_iterations: None,
        rc_protocol_v2_loopback_ok: false,
        rc_protocol_v2_loopback_unavailable: false,
        rc_protocol_v2_loopback_error: None,
        rc_protocol_v2_request_frame_bytes: request_frame.len(),
        rc_protocol_v2_response_frame_bytes: response_frame.len(),
        rc_protocol_v2_request_payload_matches: None,
        rc_protocol_v2_response_payload_matches: None,
        rc_protocol_v2_send_completions: None,
        rc_protocol_v2_recv_completions: None,
        rc_protocol_v2_poll_iterations: None,
    };

    let Some(path) = native_library_path else {
        probe.native_library_error = Some(
            "native library not found; set GLMRT_NATIVE_LIB or run just test-native".to_owned(),
        );
        return probe;
    };

    let library = match unsafe { NativeLibrary::load(&path) } {
        Ok(library) => library,
        Err(err) => {
            probe.native_library_error = Some(err.to_string());
            return probe;
        }
    };
    probe.native_library_loaded = true;

    match library.rdma_device_info() {
        Ok(info) => {
            probe.rdma_enabled = Some(info.rdma_enabled != 0);
            probe.rdma_device_count = Some(info.device_count);
            probe.first_device_openable = Some(info.first_device_openable != 0);
            probe.first_device_name = Some(c_char_array_to_string(&info.first_device_name));
            probe.first_device_status = Some(c_char_array_to_string(&info.status));
        }
        Err(err) => {
            probe.native_library_error = Some(err.to_string());
            return probe;
        }
    }

    let host_buffer = vec![0_u8; loopback_bytes];
    match library.rdma_plan_host_buffer_registration(
        host_buffer.as_ptr().cast(),
        host_buffer.len(),
        glmrt_transport::verbs_host_capabilities().preferred_alignment,
    ) {
        Ok(plan) => {
            probe.host_buffer_plan_checked = true;
            probe.host_buffer_plan_registered_span_bytes = Some(plan.registered_span_bytes);
            probe.host_buffer_plan_span_aligned = Some(plan.span_aligned != 0);
        }
        Err(err) => {
            probe.native_library_error = Some(err.to_string());
            return probe;
        }
    }

    match library.rdma_create_rc_qp_probe(1, 16, 16, 1) {
        Ok(_) => {
            probe.rc_qp_probe_ok = true;
        }
        Err(err) => {
            let err = err.to_string();
            probe.rc_qp_probe_unavailable = is_rdma_unavailable_error(&err);
            probe.rc_qp_probe_error = Some(err);
        }
    }

    match library.rdma_rc_send_recv_loopback_probe(1, loopback_bytes) {
        Ok(loopback) => {
            probe.rc_send_recv_loopback_ok = true;
            probe.rc_send_recv_payload_matches = Some(loopback.payload_matches != 0);
            probe.rc_send_recv_send_completions = Some(loopback.send_completions);
            probe.rc_send_recv_recv_completions = Some(loopback.recv_completions);
            probe.rc_send_recv_poll_iterations = Some(loopback.poll_iterations);
        }
        Err(err) => {
            let err = err.to_string();
            probe.rc_send_recv_loopback_unavailable = is_rdma_unavailable_error(&err);
            probe.rc_send_recv_loopback_error = Some(err);
        }
    }

    match library.rdma_rc_protocol_v2_loopback_probe(1, request_frame, response_frame) {
        Ok(loopback) => {
            probe.rc_protocol_v2_loopback_ok = true;
            probe.rc_protocol_v2_request_payload_matches =
                Some(loopback.request_payload_matches != 0);
            probe.rc_protocol_v2_response_payload_matches =
                Some(loopback.response_payload_matches != 0);
            probe.rc_protocol_v2_send_completions = Some(loopback.send_completions);
            probe.rc_protocol_v2_recv_completions = Some(loopback.recv_completions);
            probe.rc_protocol_v2_poll_iterations = Some(loopback.poll_iterations);
        }
        Err(err) => {
            let err = err.to_string();
            probe.rc_protocol_v2_loopback_unavailable = is_rdma_unavailable_error(&err);
            probe.rc_protocol_v2_loopback_error = Some(err);
        }
    }

    probe
}

fn native_library_path() -> Option<PathBuf> {
    if let Ok(path) = env::var("GLMRT_NATIVE_LIB") {
        return Some(PathBuf::from(path));
    }
    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .join("native/build/libglmrt_native.so");
    if manifest_path.exists() {
        return Some(manifest_path);
    }
    let cwd_path = PathBuf::from("native/build/libglmrt_native.so");
    cwd_path.exists().then_some(cwd_path)
}

fn is_rdma_unavailable_error(err: &str) -> bool {
    err.contains(&format!("status {GLMRT_STATUS_RDMA_UNAVAILABLE}"))
}

pub(super) fn run_glmrt_verbs_app_benchmark(
    resolved_mode: &str,
    peer: Option<&str>,
    port: u16,
    payloads: &[GlmrtVerbsAppPayload],
    duration_secs: u64,
) -> Result<Vec<GlmrtVerbsAppRun>> {
    match app_role_for_resolved_mode(resolved_mode)? {
        "server" => run_glmrt_verbs_app_server(peer, port, payloads),
        "client" => {
            let peer = peer.context("GLMRT verbs app client requires --peer")?;
            run_glmrt_verbs_app_client(peer, port, payloads, duration_secs)
        }
        "capability" => Ok(Vec::new()),
        other => anyhow::bail!("unsupported GLMRT verbs app role: {other}"),
    }
}

struct NativeRdmaEndpoint<'a> {
    library: &'a NativeLibrary,
    info: GlmrtRdmaRcEndpointInfo,
}

impl<'a> NativeRdmaEndpoint<'a> {
    fn create(
        library: &'a NativeLibrary,
        role: &str,
        payload: &GlmrtVerbsAppPayload,
        local_psn: u32,
    ) -> Result<Self> {
        let port_num = verbs_app_ib_port_num()?;
        let (send_frame_bytes, recv_frame_bytes, send_span, recv_span) = match role {
            "client" => (
                payload.request_wire_bytes,
                payload.response_wire_bytes,
                payload.request_registered_span_bytes,
                payload.response_registered_span_bytes,
            ),
            "server" => (
                payload.response_wire_bytes,
                payload.request_wire_bytes,
                payload.response_registered_span_bytes,
                payload.request_registered_span_bytes,
            ),
            other => anyhow::bail!("unsupported RDMA endpoint role: {other}"),
        };
        let info = library.rdma_rc_endpoint_create(
            port_num,
            local_psn,
            send_frame_bytes,
            recv_frame_bytes,
            send_span,
            recv_span,
            8,
            8,
            1,
        )?;
        Ok(Self { library, info })
    }

    fn native_descriptor(&self) -> GlmrtVerbsAppNativeEndpointDescriptor {
        GlmrtVerbsAppNativeEndpointDescriptor {
            port_num: self.info.port_num,
            qp_num: self.info.qp_num,
            psn: self.info.psn,
            lid: self.info.lid,
            active_mtu: self.info.active_mtu,
            gid_hex: c_char_array_to_string(&self.info.gid_hex),
            send_frame_bytes: self.info.send_frame_bytes,
            recv_frame_bytes: self.info.recv_frame_bytes,
            send_registered_span_bytes: self.info.send_registered_span_bytes,
            recv_registered_span_bytes: self.info.recv_registered_span_bytes,
            max_send_wr: self.info.max_send_wr,
            max_recv_wr: self.info.max_recv_wr,
            max_sge: self.info.max_sge,
            device_name: c_char_array_to_string(&self.info.device_name),
            status: c_char_array_to_string(&self.info.status),
        }
    }

    fn verbs_descriptor(&self, role: &str, host: &str) -> VerbsHostRcEndpointDescriptor {
        VerbsHostRcEndpointDescriptor {
            role: role.to_owned(),
            host: host.to_owned(),
            port_num: self.info.port_num,
            qp_num: self.info.qp_num,
            psn: self.info.psn,
            gid_hex: c_char_array_to_string(&self.info.gid_hex),
            send_frame_bytes: self.info.send_frame_bytes,
            recv_frame_bytes: self.info.recv_frame_bytes,
            send_registered_span_bytes: self.info.send_registered_span_bytes,
            recv_registered_span_bytes: self.info.recv_registered_span_bytes,
            max_send_wr: self.info.max_send_wr,
            max_recv_wr: self.info.max_recv_wr,
            max_sge: self.info.max_sge,
        }
    }

    fn connect(&self, peer: &GlmrtVerbsAppNativeEndpointDescriptor) -> Result<()> {
        self.library.rdma_rc_endpoint_connect(
            self.info.handle,
            peer.qp_num,
            peer.psn,
            peer.lid,
            &peer.gid_hex,
        )
    }

    fn post_recv(&self, bytes: usize) -> Result<()> {
        self.library
            .rdma_rc_endpoint_post_recv(self.info.handle, bytes, APP_RECV_WR_ID)
    }

    fn send(&self, frame: &[u8]) -> Result<()> {
        self.library
            .rdma_rc_endpoint_send(self.info.handle, frame, APP_SEND_WR_ID)
    }

    fn poll(&self, send: u32, recv: u32) -> Result<GlmrtRdmaRcCompletionStats> {
        self.library
            .rdma_rc_endpoint_poll(self.info.handle, send, recv, u32::MAX, 30_000)
    }

    fn copy_recv(&self, out: &mut [u8], bytes: usize) -> Result<()> {
        self.library
            .rdma_rc_endpoint_copy_recv(self.info.handle, out, bytes)
    }
}

impl Drop for NativeRdmaEndpoint<'_> {
    fn drop(&mut self) {
        if !self.info.handle.is_null() {
            let _ = self.library.rdma_rc_endpoint_destroy(self.info.handle);
            self.info.handle = std::ptr::null_mut();
        }
    }
}

fn run_glmrt_verbs_app_server(
    peer: Option<&str>,
    port: u16,
    payloads: &[GlmrtVerbsAppPayload],
) -> Result<Vec<GlmrtVerbsAppRun>> {
    let native_path = native_library_path().context(
        "native library not found; set GLMRT_NATIVE_LIB or run a native RDMA-enabled build",
    )?;
    let library = unsafe { NativeLibrary::load(&native_path) }?;
    let bind_addr = format!("0.0.0.0:{port}");
    let listener = TcpListener::bind(&bind_addr).with_context(|| format!("binding {bind_addr}"))?;
    listener
        .set_nonblocking(true)
        .context("setting app verbs server listener nonblocking")?;
    let deadline = Instant::now() + verbs_app_control_timeout();
    let (mut stream, remote_addr) = loop {
        match listener.accept() {
            Ok(value) => break value,
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    anyhow::bail!("timed out waiting for GLMRT verbs app client on {bind_addr}");
                }
                thread::sleep(Duration::from_millis(25));
            }
            Err(err) => return Err(err).context("accepting GLMRT verbs app client"),
        }
    };
    configure_control_stream(&stream)?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let peer_label = peer
        .filter(|value| !value.trim().is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| remote_addr.to_string());
    let mut runs = Vec::new();
    let expected_runs = payloads.len() * 2;
    for _ in 0..expected_runs {
        let start: GlmrtVerbsAppRunStart = read_control(&mut reader)?;
        if start.message != "run_start" {
            anyhow::bail!(
                "GLMRT verbs app server expected run_start, got {}",
                start.message
            );
        }
        let payload = payloads.get(start.payload_index).with_context(|| {
            format!(
                "GLMRT verbs app server payload index {} is out of range",
                start.payload_index
            )
        })?;
        runs.push(execute_server_app_run(
            &library,
            &mut stream,
            &start,
            payload,
            &peer_label,
        )?);
    }
    Ok(runs)
}

fn run_glmrt_verbs_app_client(
    peer: &str,
    port: u16,
    payloads: &[GlmrtVerbsAppPayload],
    duration_secs: u64,
) -> Result<Vec<GlmrtVerbsAppRun>> {
    let native_path = native_library_path().context(
        "native library not found; set GLMRT_NATIVE_LIB or run a native RDMA-enabled build",
    )?;
    let library = unsafe { NativeLibrary::load(&native_path) }?;
    let mut stream = connect_control_stream(peer, port)?;
    configure_control_stream(&stream)?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut runs = Vec::new();
    for (payload_index, payload) in payloads.iter().enumerate() {
        let iterations = verbs_app_roundtrip_iterations(payload, duration_secs)?;
        runs.push(execute_client_app_run(
            &library,
            &mut stream,
            &mut reader,
            peer,
            payload_index,
            payload,
            "roundtrip",
            1,
            iterations,
        )?);
    }
    let chain_hops = verbs_app_chain_hops()?;
    let chain_iterations = verbs_app_chain_iterations()?;
    for (payload_index, payload) in payloads.iter().enumerate() {
        runs.push(execute_client_app_run(
            &library,
            &mut stream,
            &mut reader,
            peer,
            payload_index,
            payload,
            "chain_75hop",
            chain_hops,
            chain_iterations,
        )?);
    }
    Ok(runs)
}

fn execute_client_app_run(
    library: &NativeLibrary,
    stream: &mut TcpStream,
    reader: &mut BufReader<TcpStream>,
    peer: &str,
    payload_index: usize,
    payload: &GlmrtVerbsAppPayload,
    run_kind: &str,
    hops_per_iteration: usize,
    iterations: usize,
) -> Result<GlmrtVerbsAppRun> {
    let (_source_kind, request, response) =
        protocol_v2_request_response_for_row_count(payload_index as u64 + 1, payload.row_count)?;
    let request_frame = request.encode()?;
    let response_frame = response.encode()?;
    validate_payload_matches_frames(payload, &request_frame, &response_frame)?;
    let endpoint = NativeRdmaEndpoint::create(
        library,
        "client",
        payload,
        local_psn_for_run("client", payload_index, run_kind),
    )?;
    let (client_host, _server_host) =
        distinct_endpoint_hosts(local_control_host("client"), peer.to_owned());
    let client_endpoint = endpoint.verbs_descriptor("client", &client_host);
    let start = GlmrtVerbsAppRunStart {
        message: "run_start".to_owned(),
        run_kind: run_kind.to_owned(),
        payload_index,
        row_count: payload.row_count,
        source_kind: payload.source_kind.clone(),
        request_wire_bytes: payload.request_wire_bytes,
        response_wire_bytes: payload.response_wire_bytes,
        request_logical_payload_bytes: payload.request_logical_payload_bytes,
        response_logical_payload_bytes: payload.response_logical_payload_bytes,
        request_registered_span_bytes: payload.request_registered_span_bytes,
        response_registered_span_bytes: payload.response_registered_span_bytes,
        iterations,
        hops_per_iteration,
        client_endpoint,
        client_native_endpoint: endpoint.native_descriptor(),
    };
    write_control(stream, &start)?;
    let ready: GlmrtVerbsAppRunReady = read_control(reader)?;
    if ready.message != "run_ready" || ready.payload_index != payload_index {
        anyhow::bail!("GLMRT verbs app client received invalid run_ready message");
    }
    let endpoint_plan = verbs_host_protocol_v2_endpoint_plan(
        &request_frame,
        &response_frame,
        glmrt_transport::verbs_host_capabilities().preferred_alignment,
    )?;
    let validation = verbs_host_validate_protocol_v2_handshake(
        &endpoint_plan,
        &start.client_endpoint,
        &ready.server_endpoint,
    )?;
    let _roundtrip = verbs_host_protocol_v2_round_trip_plan(
        &endpoint_plan,
        &validation,
        &request_frame,
        &response_frame,
    )?;
    endpoint.connect(&ready.server_native_endpoint)?;
    let recv_ready: GlmrtVerbsAppRecvReady = read_control(reader)?;
    if recv_ready.message != "recv_ready" || recv_ready.payload_index != payload_index {
        anyhow::bail!("GLMRT verbs app client received invalid recv_ready message");
    }

    let mut recv_buf = vec![0_u8; response_frame.len()];
    let mut response_payload_matches = true;
    let mut send_completions = 0_u64;
    let mut recv_completions = 0_u64;
    let mut poll_iterations = 0_u64;
    let mut sample_latencies = Vec::with_capacity(iterations);
    let started = Instant::now();
    for _ in 0..iterations {
        let sample_started = Instant::now();
        for _ in 0..hops_per_iteration {
            endpoint.post_recv(response_frame.len())?;
            endpoint.send(&request_frame)?;
            let stats = endpoint.poll(1, 1)?;
            accumulate_completion_stats(
                &mut send_completions,
                &mut recv_completions,
                &mut poll_iterations,
                &stats,
            );
            endpoint.copy_recv(&mut recv_buf, response_frame.len())?;
            if recv_buf != response_frame {
                response_payload_matches = false;
            }
        }
        sample_latencies.push(sample_started.elapsed().as_secs_f64() * 1_000_000.0);
    }
    let elapsed = started.elapsed();
    let roundtrips = checked_mul(
        iterations,
        hops_per_iteration,
        "verbs app client roundtrips",
    )?;
    Ok(build_app_run_report(
        run_kind,
        "client",
        peer,
        payload_index,
        payload,
        iterations,
        hops_per_iteration,
        roundtrips,
        elapsed,
        &sample_latencies,
        true,
        response_payload_matches,
        send_completions,
        recv_completions,
        poll_iterations,
        endpoint.native_descriptor(),
        ready.server_native_endpoint,
    ))
}

fn execute_server_app_run(
    library: &NativeLibrary,
    stream: &mut TcpStream,
    start: &GlmrtVerbsAppRunStart,
    payload: &GlmrtVerbsAppPayload,
    peer: &str,
) -> Result<GlmrtVerbsAppRun> {
    if start.row_count != payload.row_count
        || start.request_wire_bytes != payload.request_wire_bytes
        || start.response_wire_bytes != payload.response_wire_bytes
        || start.request_registered_span_bytes != payload.request_registered_span_bytes
        || start.response_registered_span_bytes != payload.response_registered_span_bytes
    {
        anyhow::bail!("GLMRT verbs app server run_start payload metadata did not match local plan");
    }
    let (_source_kind, request, response) = protocol_v2_request_response_for_row_count(
        start.payload_index as u64 + 1,
        payload.row_count,
    )?;
    let request_frame = request.encode()?;
    let response_frame = response.encode()?;
    validate_payload_matches_frames(payload, &request_frame, &response_frame)?;
    let endpoint = NativeRdmaEndpoint::create(
        library,
        "server",
        payload,
        local_psn_for_run("server", start.payload_index, &start.run_kind),
    )?;
    let (_client_host, server_host) =
        distinct_endpoint_hosts(peer.to_owned(), local_control_host("server"));
    let server_endpoint = endpoint.verbs_descriptor("server", &server_host);
    let endpoint_plan = verbs_host_protocol_v2_endpoint_plan(
        &request_frame,
        &response_frame,
        glmrt_transport::verbs_host_capabilities().preferred_alignment,
    )?;
    let validation = verbs_host_validate_protocol_v2_handshake(
        &endpoint_plan,
        &start.client_endpoint,
        &server_endpoint,
    )?;
    let _roundtrip = verbs_host_protocol_v2_round_trip_plan(
        &endpoint_plan,
        &validation,
        &request_frame,
        &response_frame,
    )?;
    let ready = GlmrtVerbsAppRunReady {
        message: "run_ready".to_owned(),
        payload_index: start.payload_index,
        server_endpoint,
        server_native_endpoint: endpoint.native_descriptor(),
    };
    write_control(stream, &ready)?;
    endpoint.connect(&start.client_native_endpoint)?;
    endpoint.post_recv(request_frame.len())?;
    write_control(
        stream,
        &GlmrtVerbsAppRecvReady {
            message: "recv_ready".to_owned(),
            payload_index: start.payload_index,
        },
    )?;

    let total_roundtrips = checked_mul(
        start.iterations,
        start.hops_per_iteration,
        "verbs app server roundtrips",
    )?;
    let mut recv_buf = vec![0_u8; request_frame.len()];
    let mut have_recv_completion = false;
    let mut request_payload_matches = true;
    let mut send_completions = 0_u64;
    let mut recv_completions = 0_u64;
    let mut poll_iterations = 0_u64;
    let started = Instant::now();
    for roundtrip_idx in 0..total_roundtrips {
        if !have_recv_completion {
            let stats = endpoint.poll(0, 1)?;
            accumulate_completion_stats(
                &mut send_completions,
                &mut recv_completions,
                &mut poll_iterations,
                &stats,
            );
        }
        endpoint.copy_recv(&mut recv_buf, request_frame.len())?;
        if recv_buf != request_frame {
            request_payload_matches = false;
        }
        let is_last = roundtrip_idx + 1 == total_roundtrips;
        if !is_last {
            endpoint.post_recv(request_frame.len())?;
        }
        endpoint.send(&response_frame)?;
        let stats = if is_last {
            have_recv_completion = false;
            endpoint.poll(1, 0)?
        } else {
            have_recv_completion = true;
            endpoint.poll(1, 1)?
        };
        accumulate_completion_stats(
            &mut send_completions,
            &mut recv_completions,
            &mut poll_iterations,
            &stats,
        );
    }
    let elapsed = started.elapsed();
    Ok(build_app_run_report(
        &start.run_kind,
        "server",
        peer,
        start.payload_index,
        payload,
        start.iterations,
        start.hops_per_iteration,
        total_roundtrips,
        elapsed,
        &[],
        request_payload_matches,
        true,
        send_completions,
        recv_completions,
        poll_iterations,
        endpoint.native_descriptor(),
        start.client_native_endpoint.clone(),
    ))
}

fn build_app_run_report(
    run_kind: &str,
    role: &str,
    peer: &str,
    payload_index: usize,
    payload: &GlmrtVerbsAppPayload,
    iterations: usize,
    hops_per_iteration: usize,
    roundtrips: usize,
    elapsed: Duration,
    sample_latencies_micros: &[f64],
    request_payload_matches: bool,
    response_payload_matches: bool,
    send_completions: u64,
    recv_completions: u64,
    poll_iterations: u64,
    local_endpoint: GlmrtVerbsAppNativeEndpointDescriptor,
    peer_endpoint: GlmrtVerbsAppNativeEndpointDescriptor,
) -> GlmrtVerbsAppRun {
    let elapsed_secs = elapsed.as_secs_f64().max(f64::MIN_POSITIVE);
    let total_wire_bytes =
        roundtrips as f64 * (payload.request_wire_bytes + payload.response_wire_bytes) as f64;
    let sample_avg = mean_f64(sample_latencies_micros);
    let roundtrip_samples = sample_latencies_micros
        .iter()
        .map(|value| *value / hops_per_iteration as f64)
        .collect::<Vec<_>>();
    GlmrtVerbsAppRun {
        run_kind: run_kind.to_owned(),
        role: role.to_owned(),
        peer: peer.to_owned(),
        ok: request_payload_matches && response_payload_matches,
        payload_index,
        row_count: payload.row_count,
        source_kind: payload.source_kind.clone(),
        request_wire_bytes: payload.request_wire_bytes,
        response_wire_bytes: payload.response_wire_bytes,
        request_logical_payload_bytes: payload.request_logical_payload_bytes,
        response_logical_payload_bytes: payload.response_logical_payload_bytes,
        request_registered_span_bytes: payload.request_registered_span_bytes,
        response_registered_span_bytes: payload.response_registered_span_bytes,
        iterations,
        hops_per_iteration,
        roundtrips,
        elapsed_micros: elapsed.as_micros(),
        roundtrip_latency_micros_avg: mean_f64(&roundtrip_samples),
        roundtrip_latency_micros_min: min_f64(&roundtrip_samples),
        roundtrip_latency_micros_max: max_f64(&roundtrip_samples),
        sample_latency_micros_avg: sample_avg,
        effective_roundtrips_per_second: roundtrips as f64 / elapsed_secs,
        effective_payload_gbps: total_wire_bytes * 8.0 / elapsed_secs / 1_000_000_000.0,
        request_payload_matches,
        response_payload_matches,
        send_completions,
        recv_completions,
        poll_iterations,
        local_endpoint,
        peer_endpoint,
    }
}

fn connect_control_stream(peer: &str, port: u16) -> Result<TcpStream> {
    let addr = peer_control_addr(peer, port);
    let timeout = verbs_app_control_timeout();
    let mut addrs = addr
        .to_socket_addrs()
        .with_context(|| format!("resolving GLMRT verbs app peer {addr}"))?;
    let first = addrs
        .next()
        .with_context(|| format!("GLMRT verbs app peer {addr} resolved no addresses"))?;
    TcpStream::connect_timeout(&first, timeout)
        .with_context(|| format!("connecting GLMRT verbs app control plane to {addr}"))
}

fn peer_control_addr(peer: &str, port: u16) -> String {
    if peer
        .rsplit_once(':')
        .is_some_and(|(_, maybe_port)| maybe_port.parse::<u16>().is_ok())
    {
        peer.to_owned()
    } else {
        format!("{peer}:{port}")
    }
}

fn configure_control_stream(stream: &TcpStream) -> Result<()> {
    let timeout = verbs_app_control_timeout();
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    Ok(())
}

fn write_control<T: Serialize>(stream: &mut TcpStream, value: &T) -> Result<()> {
    serde_json::to_writer(&mut *stream, value)?;
    stream.write_all(b"\n")?;
    stream.flush()?;
    Ok(())
}

fn read_control<T: DeserializeOwned>(reader: &mut BufReader<TcpStream>) -> Result<T> {
    let mut line = String::new();
    let bytes = reader.read_line(&mut line)?;
    if bytes == 0 {
        anyhow::bail!("GLMRT verbs app control plane closed");
    }
    Ok(serde_json::from_str(line.trim_end())?)
}

fn validate_payload_matches_frames(
    payload: &GlmrtVerbsAppPayload,
    request_frame: &[u8],
    response_frame: &[u8],
) -> Result<()> {
    if payload.request_wire_bytes != request_frame.len() {
        anyhow::bail!(
            "GLMRT verbs app request frame bytes {} did not match payload {}",
            request_frame.len(),
            payload.request_wire_bytes
        );
    }
    if payload.response_wire_bytes != response_frame.len() {
        anyhow::bail!(
            "GLMRT verbs app response frame bytes {} did not match payload {}",
            response_frame.len(),
            payload.response_wire_bytes
        );
    }
    Ok(())
}

fn accumulate_completion_stats(
    send_total: &mut u64,
    recv_total: &mut u64,
    poll_total: &mut u64,
    stats: &GlmrtRdmaRcCompletionStats,
) {
    *send_total += stats.send_completions as u64;
    *recv_total += stats.recv_completions as u64;
    *poll_total += stats.poll_iterations as u64;
}

fn local_control_host(role: &str) -> String {
    env::var("HOSTNAME")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| format!("local-{role}"))
}

fn local_psn_for_run(role: &str, payload_index: usize, run_kind: &str) -> u32 {
    let role_base = match role {
        "client" => 0x110000,
        "server" => 0x220000,
        _ => 0x330000,
    };
    let kind_offset = if run_kind.starts_with("chain") {
        0x1000
    } else {
        0
    };
    role_base + kind_offset + payload_index as u32 + 1
}

fn verbs_app_ib_port_num() -> Result<u32> {
    parse_env_u32("GLMRT_VERBS_APP_IB_PORT_NUM", 1)
}

fn verbs_app_control_timeout() -> Duration {
    let secs = parse_env_u64("GLMRT_VERBS_APP_CONTROL_TIMEOUT_SECS", 60).unwrap_or(60);
    Duration::from_secs(secs)
}

fn verbs_app_roundtrip_iterations(
    payload: &GlmrtVerbsAppPayload,
    duration_secs: u64,
) -> Result<usize> {
    if let Some(value) = parse_optional_env_usize("GLMRT_VERBS_APP_ROUNDTRIP_ITERATIONS")? {
        return Ok(value.max(1));
    }
    let multiplier = duration_secs.max(1) as usize;
    let logical = payload.request_logical_payload_bytes;
    let base = if logical <= 12_288 {
        1_000
    } else if logical <= 196_608 {
        200
    } else if logical <= 786_432 {
        50
    } else {
        10
    };
    Ok(base * multiplier)
}

fn verbs_app_chain_iterations() -> Result<usize> {
    parse_optional_env_usize("GLMRT_VERBS_APP_CHAIN_ITERATIONS")
        .map(|value| value.unwrap_or(1).max(1))
}

fn verbs_app_chain_hops() -> Result<usize> {
    parse_optional_env_usize("GLMRT_VERBS_APP_CHAIN_HOPS")
        .map(|value| value.unwrap_or_else(glm52_sparse_moe_chain_hops).max(1))
}

fn parse_optional_env_usize(name: &str) -> Result<Option<usize>> {
    match env::var(name) {
        Ok(value) if !value.trim().is_empty() => value
            .parse::<usize>()
            .map(Some)
            .with_context(|| format!("invalid {name} value {value}")),
        _ => Ok(None),
    }
}

fn parse_env_u32(name: &str, default: u32) -> Result<u32> {
    match env::var(name) {
        Ok(value) if !value.trim().is_empty() => value
            .parse::<u32>()
            .with_context(|| format!("invalid {name} value {value}")),
        _ => Ok(default),
    }
}

fn parse_env_u64(name: &str, default: u64) -> Result<u64> {
    match env::var(name) {
        Ok(value) if !value.trim().is_empty() => value
            .parse::<u64>()
            .with_context(|| format!("invalid {name} value {value}")),
        _ => Ok(default),
    }
}

fn mean_f64(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(values.iter().sum::<f64>() / values.len() as f64)
}

fn min_f64(values: &[f64]) -> Option<f64> {
    values.iter().copied().reduce(f64::min)
}

fn max_f64(values: &[f64]) -> Option<f64> {
    values.iter().copied().reduce(f64::max)
}

fn protocol_v2_payloads_for_benchmark(
    payload_bytes: &[usize],
) -> Result<Vec<GlmrtVerbsAppPayload>> {
    let hidden_row_bytes = GLM52_HIDDEN_SIZE * ExpertV2Dtype::Bf16.bytes_per_element();
    payload_bytes
        .iter()
        .enumerate()
        .map(|(idx, payload)| {
            if payload % hidden_row_bytes != 0 {
                anyhow::bail!(
                    "GLMRT app payload {payload} is not an exact multiple of hidden row bytes {hidden_row_bytes}"
                );
            }
            let row_count = payload / hidden_row_bytes;
            if row_count == 0 {
                anyhow::bail!("GLMRT app payload must contain at least one hidden row");
            }
            protocol_v2_payload_for_row_count(idx as u64 + 1, row_count)
        })
        .collect()
}

fn protocol_v2_payload_for_row_count(
    request_id: u64,
    row_count: usize,
) -> Result<GlmrtVerbsAppPayload> {
    let (source_kind, request, response) =
        protocol_v2_request_response_for_row_count(request_id, row_count)?;
    let request_stats = request.wire_stats();
    let response_stats = response.wire_stats();
    let frame_arena = frame_arena_evidence(&request, &response)?;
    Ok(GlmrtVerbsAppPayload {
        row_count,
        source_kind: source_kind_label(source_kind).to_owned(),
        request_logical_payload_bytes: request_stats.logical_payload_bytes,
        request_wire_bytes: frame_arena.request_wire_bytes,
        response_logical_payload_bytes: response_stats.logical_payload_bytes,
        response_wire_bytes: frame_arena.response_wire_bytes,
        total_logical_payload_bytes: request_stats.logical_payload_bytes
            + response_stats.logical_payload_bytes,
        total_wire_bytes: frame_arena.request_wire_bytes + frame_arena.response_wire_bytes,
        request_frame_buffer_capacity_bytes: frame_arena.request_capacity_bytes,
        response_frame_buffer_capacity_bytes: frame_arena.response_capacity_bytes,
        request_frame_buffer_stable: frame_arena.request_stable_allocation,
        response_frame_buffer_stable: frame_arena.response_stable_allocation,
        request_frame_arena_capacity_bytes: frame_arena.request_capacity_bytes,
        response_frame_arena_capacity_bytes: frame_arena.response_capacity_bytes,
        request_frame_arena_stable: frame_arena.request_stable_allocation,
        response_frame_arena_stable: frame_arena.response_stable_allocation,
        frame_arena_registration_alignment_bytes: frame_arena.registration_alignment_bytes,
        request_registered_span_bytes: frame_arena.request_registered_span_bytes,
        response_registered_span_bytes: frame_arena.response_registered_span_bytes,
        total_registered_span_bytes: frame_arena
            .request_registered_span_bytes
            .checked_add(frame_arena.response_registered_span_bytes)
            .context("total registered span byte count overflow")?,
        request_registration_slack_bytes: frame_arena.request_registration_slack_bytes,
        response_registration_slack_bytes: frame_arena.response_registration_slack_bytes,
        request_registered_span_aligned: frame_arena.request_registered_span_aligned,
        response_registered_span_aligned: frame_arena.response_registered_span_aligned,
        request_hidden_row_view_count: frame_arena.request_row_views.rows,
        response_partial_output_row_view_count: frame_arena.response_row_views.rows,
        request_hidden_row_view_payload_bytes: frame_arena.request_row_views.payload_bytes,
        response_partial_output_row_view_payload_bytes: frame_arena
            .response_row_views
            .payload_bytes,
        request_row_views_cover_payload: frame_arena.request_row_views.cover_payload,
        response_row_views_cover_payload: frame_arena.response_row_views.cover_payload,
        response_generated_by_route_dependent_executor: true,
        response_differs_from_request_payload: request.hidden_payload
            != response.partial_output_payload,
        host_memory_registration_required: true,
    })
}

fn protocol_v2_request_response_for_row_count(
    request_id: u64,
    row_count: usize,
) -> Result<(
    ExpertV2SourceKind,
    ExpertProtocolV2Request,
    ExpertProtocolV2Response,
)> {
    let source_kind = source_kind_for_row_count(row_count);
    let rows = (0..row_count)
        .map(|idx| ExpertProtocolV2RowDescriptor {
            row_id: idx as u64,
            source_kind,
            source_request_id: request_id,
            token_position: idx as u64,
            route_offset: idx as u32,
            route_count: 1,
        })
        .collect::<Vec<_>>();
    let routes = (0..row_count)
        .map(|idx| ExpertProtocolV2RouteEntry {
            row_index: idx as u32,
            expert_id: (idx % 256) as u32,
            gate_weight: 1.0,
        })
        .collect::<Vec<_>>();
    let hidden_payload = deterministic_hidden_payload_bf16(row_count, GLM52_HIDDEN_SIZE)?;
    let request = ExpertProtocolV2Request::new(
        request_id,
        1,
        3,
        GLM52_HIDDEN_SIZE as u32,
        ExpertV2Dtype::Bf16,
        rows,
        routes,
        hidden_payload,
    )?;
    let response = protocol_v2_synthetic_response(&request)?;
    Ok((source_kind, request, response))
}

fn source_kind_for_row_count(row_count: usize) -> ExpertV2SourceKind {
    match row_count {
        1 => ExpertV2SourceKind::Decode,
        2 | 4 | 8 => ExpertV2SourceKind::MtpVerify,
        _ => ExpertV2SourceKind::Prefill,
    }
}

fn deterministic_hidden_payload_bf16(row_count: usize, hidden_dim: usize) -> Result<Vec<u8>> {
    let value_count = row_count
        .checked_mul(hidden_dim)
        .context("deterministic hidden value count overflow")?;
    let mut payload = Vec::with_capacity(
        value_count
            .checked_mul(ExpertV2Dtype::Bf16.bytes_per_element())
            .context("deterministic hidden payload byte count overflow")?,
    );
    for row in 0..row_count {
        for col in 0..hidden_dim {
            let lane = ((row as u32)
                .wrapping_mul(17)
                .wrapping_add((col as u32).wrapping_mul(31))
                .wrapping_add(13)
                % 257) as f32;
            let value = (lane - 128.0) / 256.0;
            payload.extend_from_slice(&f32_to_bf16_bytes(value));
        }
    }
    Ok(payload)
}

fn f32_to_bf16_bytes(value: f32) -> [u8; 2] {
    ((value.to_bits() >> 16) as u16).to_le_bytes()
}

fn protocol_v2_chain_plans_for_payloads(
    payloads: &[GlmrtVerbsAppPayload],
    hops: usize,
) -> Result<Vec<GlmrtVerbsAppChainPlan>> {
    payloads
        .iter()
        .map(|payload| protocol_v2_chain_plan_for_payload(payload, hops))
        .collect()
}

fn protocol_v2_chain_plan_for_payload(
    payload: &GlmrtVerbsAppPayload,
    hops: usize,
) -> Result<GlmrtVerbsAppChainPlan> {
    let chain_kind = match payload.source_kind.as_str() {
        "decode" => "decode_sparse_moe_chain",
        "mtp_verify" => "mtp_verify_sparse_moe_chain",
        _ => "prefill_sparse_moe_chain",
    };
    Ok(GlmrtVerbsAppChainPlan {
        chain_kind: chain_kind.to_owned(),
        row_count: payload.row_count,
        source_kind: payload.source_kind.clone(),
        hops,
        request_logical_payload_bytes_per_hop: payload.request_logical_payload_bytes,
        response_logical_payload_bytes_per_hop: payload.response_logical_payload_bytes,
        request_wire_bytes_per_hop: payload.request_wire_bytes,
        response_wire_bytes_per_hop: payload.response_wire_bytes,
        registered_span_bytes_per_hop: payload.total_registered_span_bytes,
        total_request_wire_bytes: checked_mul(
            payload.request_wire_bytes,
            hops,
            "chain request wire bytes",
        )?,
        total_response_wire_bytes: checked_mul(
            payload.response_wire_bytes,
            hops,
            "chain response wire bytes",
        )?,
        total_wire_bytes: checked_mul(payload.total_wire_bytes, hops, "chain total wire bytes")?,
        total_logical_payload_bytes: checked_mul(
            payload.total_logical_payload_bytes,
            hops,
            "chain total logical payload bytes",
        )?,
        total_registered_span_bytes: checked_mul(
            payload.total_registered_span_bytes,
            hops,
            "chain registered span bytes",
        )?,
        uses_registered_frame_spans: payload.request_registered_span_aligned
            && payload.response_registered_span_aligned
            && payload.host_memory_registration_required,
        uses_request_response_row_views: payload.request_row_views_cover_payload
            && payload.response_row_views_cover_payload,
    })
}

fn glm52_sparse_moe_chain_hops() -> usize {
    GLM52_NUM_HIDDEN_LAYERS - GLM52_FIRST_K_DENSE_REPLACE
}

struct ProtocolV2ArenaEvidence {
    request_wire_bytes: usize,
    response_wire_bytes: usize,
    request_capacity_bytes: usize,
    response_capacity_bytes: usize,
    request_stable_allocation: bool,
    response_stable_allocation: bool,
    registration_alignment_bytes: usize,
    request_registered_span_bytes: usize,
    response_registered_span_bytes: usize,
    request_registration_slack_bytes: usize,
    response_registration_slack_bytes: usize,
    request_registered_span_aligned: bool,
    response_registered_span_aligned: bool,
    request_row_views: ProtocolV2RowViewEvidence,
    response_row_views: ProtocolV2RowViewEvidence,
}

struct ProtocolV2RowViewEvidence {
    rows: usize,
    payload_bytes: usize,
    cover_payload: bool,
}

fn frame_arena_evidence(
    request: &ExpertProtocolV2Request,
    response: &ExpertProtocolV2Response,
) -> Result<ProtocolV2ArenaEvidence> {
    let expected_request_wire_bytes = request.wire_stats().wire_bytes;
    let expected_wire_bytes = response.wire_stats().wire_bytes;
    let registration_alignment = glmrt_transport::verbs_host_capabilities().preferred_alignment;
    let request_capacity = align_up(expected_request_wire_bytes, registration_alignment)?;
    let response_capacity = align_up(expected_wire_bytes, registration_alignment)?;
    let mut arena =
        ExpertProtocolV2FrameArena::with_capacities(request_capacity, response_capacity);
    let first_request_ptr = arena.request_ptr();
    let first_response_ptr = arena.response_ptr();
    let first_request_capacity = arena.request_capacity();
    let first_response_capacity = arena.response_capacity();
    let first_request_wire_bytes = {
        let view = arena.encode_request_view(request)?;
        view.wire_stats().wire_bytes
    };
    let first_response_wire_bytes = {
        let view = arena.encode_response_view(response)?;
        view.wire_stats().wire_bytes
    };
    let (second_request_wire_bytes, request_row_views) = {
        let view = arena.encode_request_view(request)?;
        let wire_bytes = view.wire_stats().wire_bytes;
        let row_views = request_row_view_evidence(&view)?;
        (wire_bytes, row_views)
    };
    let (second_response_wire_bytes, response_row_views) = {
        let view = arena.encode_response_view(response)?;
        let wire_bytes = view.wire_stats().wire_bytes;
        let row_views = response_row_view_evidence(&view)?;
        (wire_bytes, row_views)
    };
    if first_request_wire_bytes != expected_request_wire_bytes
        || second_request_wire_bytes != expected_request_wire_bytes
    {
        anyhow::bail!(
            "ProtocolV2 request frame bytes changed during reusable-arena encode: expected={expected_request_wire_bytes} first={first_request_wire_bytes} second={second_request_wire_bytes}"
        );
    }
    if first_response_wire_bytes != expected_wire_bytes
        || second_response_wire_bytes != expected_wire_bytes
    {
        anyhow::bail!(
            "ProtocolV2 response frame bytes changed during reusable-arena encode: expected={expected_wire_bytes} first={first_response_wire_bytes} second={second_response_wire_bytes}"
        );
    }
    let request_registered_span_bytes = align_up(arena.request_capacity(), registration_alignment)?;
    let response_registered_span_bytes =
        align_up(arena.response_capacity(), registration_alignment)?;
    Ok(ProtocolV2ArenaEvidence {
        request_wire_bytes: second_request_wire_bytes,
        response_wire_bytes: second_response_wire_bytes,
        request_capacity_bytes: arena.request_capacity(),
        response_capacity_bytes: arena.response_capacity(),
        request_stable_allocation: first_request_ptr == arena.request_ptr()
            && first_request_capacity == arena.request_capacity(),
        response_stable_allocation: first_response_ptr == arena.response_ptr()
            && first_response_capacity == arena.response_capacity(),
        registration_alignment_bytes: registration_alignment,
        request_registered_span_bytes,
        response_registered_span_bytes,
        request_registration_slack_bytes: request_registered_span_bytes
            .checked_sub(second_request_wire_bytes)
            .context("request registration span smaller than wire bytes")?,
        response_registration_slack_bytes: response_registered_span_bytes
            .checked_sub(second_response_wire_bytes)
            .context("response registration span smaller than wire bytes")?,
        request_registered_span_aligned: request_registered_span_bytes % registration_alignment
            == 0,
        response_registered_span_aligned: response_registered_span_bytes % registration_alignment
            == 0,
        request_row_views,
        response_row_views,
    })
}

fn request_row_view_evidence(
    view: &ExpertProtocolV2RequestView<'_>,
) -> Result<ProtocolV2RowViewEvidence> {
    let mut payload_bytes = 0usize;
    let rows = view.header.row_count as usize;
    for row_index in 0..rows {
        payload_bytes = payload_bytes
            .checked_add(view.hidden_row_payload(row_index)?.len())
            .context("request row-view payload byte count overflow")?;
    }
    Ok(ProtocolV2RowViewEvidence {
        rows,
        payload_bytes,
        cover_payload: payload_bytes == view.hidden_payload().len(),
    })
}

fn response_row_view_evidence(
    view: &ExpertProtocolV2ResponseView<'_>,
) -> Result<ProtocolV2RowViewEvidence> {
    let mut payload_bytes = 0usize;
    let rows = view.header.row_count as usize;
    for row_index in 0..rows {
        payload_bytes = payload_bytes
            .checked_add(view.partial_output_row_payload(row_index)?.len())
            .context("response row-view payload byte count overflow")?;
    }
    Ok(ProtocolV2RowViewEvidence {
        rows,
        payload_bytes,
        cover_payload: payload_bytes == view.partial_output_payload().len(),
    })
}

fn align_up(value: usize, alignment: usize) -> Result<usize> {
    if alignment == 0 {
        anyhow::bail!("registration alignment must be non-zero");
    }
    let remainder = value % alignment;
    if remainder == 0 {
        return Ok(value);
    }
    value
        .checked_add(alignment - remainder)
        .context("registration-aligned frame span overflow")
}

fn checked_mul(value: usize, rhs: usize, label: &str) -> Result<usize> {
    value
        .checked_mul(rhs)
        .with_context(|| format!("{label} overflow"))
}

fn source_kind_label(source_kind: ExpertV2SourceKind) -> &'static str {
    match source_kind {
        ExpertV2SourceKind::Decode => "decode",
        ExpertV2SourceKind::Prefill => "prefill",
        ExpertV2SourceKind::MtpVerify => "mtp_verify",
        ExpertV2SourceKind::Benchmark => "benchmark",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;

    fn parse_payload_bytes(value: &str) -> Result<Vec<usize>> {
        value
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(|part| {
                part.parse::<usize>()
                    .map_err(anyhow::Error::from)
                    .with_context(|| format!("invalid payload byte size: {part}"))
            })
            .collect()
    }

    #[test]
    fn app_protocol_v2_payloads_match_glm_hidden_rows() {
        let payloads =
            parse_payload_bytes(super::super::DEFAULT_APP_PROTOCOL_V2_PAYLOAD_BYTES).unwrap();
        let attempt =
            glmrt_verbs_app_benchmark_attempt("app-client", Some("emu".to_owned()), &payloads)
                .unwrap();

        assert!(attempt.attempted);
        assert_eq!(attempt.protocol, "ExpertProtocolV2");
        assert!(attempt.uses_reusable_protocol_v2_frame_buffers);
        assert!(attempt.uses_protocol_v2_frame_arenas);
        assert!(attempt.app_transport_implemented);
        assert!(attempt.app_transport_blocker.contains("rc-qp-send-recv"));
        assert!(attempt.app_transport_runs.is_empty());
        assert!(attempt.app_transport_error.is_none());
        assert!(attempt.native_rdma_probe.attempted);
        assert_eq!(
            attempt.native_rdma_probe.rc_send_recv_loopback_bytes,
            12_288
        );
        assert_eq!(
            attempt.native_rdma_probe.rc_protocol_v2_request_frame_bytes,
            attempt.payloads[0].request_wire_bytes
        );
        assert_eq!(
            attempt
                .native_rdma_probe
                .rc_protocol_v2_response_frame_bytes,
            attempt.payloads[0].response_wire_bytes
        );
        assert_eq!(
            attempt.protocol_v2_endpoint_plan.request_frame_bytes,
            attempt.payloads[0].request_wire_bytes
        );
        assert_eq!(
            attempt.protocol_v2_endpoint_plan.response_frame_bytes,
            attempt.payloads[0].response_wire_bytes
        );
        assert_eq!(attempt.protocol_v2_endpoint_plan.request_rows, 1);
        assert_eq!(attempt.protocol_v2_endpoint_plan.response_rows, 1);
        assert_eq!(
            attempt
                .protocol_v2_endpoint_plan
                .registration_alignment_bytes,
            4096
        );
        assert_eq!(
            attempt
                .protocol_v2_endpoint_plan
                .send_work_requests_per_roundtrip,
            2
        );
        assert_eq!(
            attempt
                .protocol_v2_endpoint_plan
                .recv_work_requests_per_roundtrip,
            2
        );
        assert!(attempt.protocol_v2_endpoint_plan.requires_peer_qp_num);
        assert!(attempt.protocol_v2_endpoint_plan.requires_peer_psn);
        assert!(attempt.protocol_v2_endpoint_plan.requires_peer_gid);
        assert!(
            attempt
                .protocol_v2_endpoint_plan
                .requires_registered_host_buffers
        );
        assert!(attempt.protocol_v2_endpoint_plan.app_transport_implemented);
        assert_eq!(
            attempt
                .protocol_v2_handshake_contract
                .client_send_frame_bytes,
            attempt.payloads[0].request_wire_bytes
        );
        assert_eq!(
            attempt
                .protocol_v2_handshake_contract
                .client_recv_frame_bytes,
            attempt.payloads[0].response_wire_bytes
        );
        assert_eq!(
            attempt
                .protocol_v2_handshake_contract
                .server_send_frame_bytes,
            attempt.payloads[0].response_wire_bytes
        );
        assert_eq!(
            attempt
                .protocol_v2_handshake_contract
                .server_recv_frame_bytes,
            attempt.payloads[0].request_wire_bytes
        );
        assert!(attempt
            .protocol_v2_handshake_contract
            .descriptor_fields
            .contains(&"qp_num".to_owned()));
        assert!(attempt
            .protocol_v2_handshake_contract
            .descriptor_fields
            .contains(&"psn".to_owned()));
        assert!(attempt
            .protocol_v2_handshake_contract
            .descriptor_fields
            .contains(&"gid_hex".to_owned()));
        assert!(
            attempt
                .protocol_v2_handshake_contract
                .descriptor_validation_available
        );
        assert_eq!(attempt.protocol_v2_control_plane_dry_run.app_role, "client");
        assert_eq!(
            attempt
                .protocol_v2_control_plane_dry_run
                .local_endpoint_role,
            "client"
        );
        assert!(
            attempt
                .protocol_v2_control_plane_dry_run
                .peer_endpoint_required
        );
        assert_eq!(
            attempt.protocol_v2_control_plane_dry_run.peer_endpoint_host,
            "emu"
        );
        assert!(
            attempt
                .protocol_v2_control_plane_dry_run
                .validation
                .validated
        );
        assert!(
            attempt
                .protocol_v2_control_plane_dry_run
                .validates_peer_qp_psn_gid
        );
        assert!(
            attempt
                .protocol_v2_control_plane_dry_run
                .validates_registered_frame_spans
        );
        assert!(
            !attempt
                .protocol_v2_control_plane_dry_run
                .data_plane_attempted
        );
        assert_eq!(
            attempt
                .protocol_v2_control_plane_dry_run
                .client_endpoint
                .send_frame_bytes,
            attempt.payloads[0].request_wire_bytes
        );
        assert_eq!(
            attempt
                .protocol_v2_control_plane_dry_run
                .server_endpoint
                .send_frame_bytes,
            attempt.payloads[0].response_wire_bytes
        );
        assert_eq!(
            attempt.protocol_v2_round_trip_plan.protocol,
            "ExpertProtocolV2"
        );
        assert_eq!(
            attempt.protocol_v2_round_trip_plan.data_plane,
            "rc-qp-send-recv"
        );
        assert_eq!(
            attempt.protocol_v2_round_trip_plan.control_plane,
            "tcp-qp-gid-psn-handshake"
        );
        assert_eq!(
            attempt.protocol_v2_round_trip_plan.client_host,
            "local-client"
        );
        assert_eq!(attempt.protocol_v2_round_trip_plan.server_host, "emu");
        assert_eq!(
            attempt.protocol_v2_round_trip_plan.request_frame_bytes,
            attempt.payloads[0].request_wire_bytes
        );
        assert_eq!(
            attempt.protocol_v2_round_trip_plan.response_frame_bytes,
            attempt.payloads[0].response_wire_bytes
        );
        assert_eq!(attempt.protocol_v2_round_trip_plan.request_rows, 1);
        assert_eq!(attempt.protocol_v2_round_trip_plan.response_rows, 1);
        assert_eq!(
            attempt
                .protocol_v2_round_trip_plan
                .client_send_work_requests,
            1
        );
        assert_eq!(
            attempt
                .protocol_v2_round_trip_plan
                .client_recv_work_requests,
            1
        );
        assert_eq!(
            attempt
                .protocol_v2_round_trip_plan
                .server_send_work_requests,
            1
        );
        assert_eq!(
            attempt
                .protocol_v2_round_trip_plan
                .server_recv_work_requests,
            1
        );
        assert_eq!(attempt.protocol_v2_round_trip_plan.total_work_requests, 4);
        assert!(
            attempt
                .protocol_v2_round_trip_plan
                .request_frame_fits_registered_span
        );
        assert!(
            attempt
                .protocol_v2_round_trip_plan
                .response_frame_fits_registered_span
        );
        assert!(
            attempt
                .protocol_v2_round_trip_plan
                .request_response_headers_match
        );
        assert!(attempt.protocol_v2_round_trip_plan.endpoints_validated);
        assert!(
            attempt
                .protocol_v2_round_trip_plan
                .registered_spans_match_endpoint_plan
        );
        assert!(
            attempt
                .protocol_v2_round_trip_plan
                .app_transport_implemented
        );
        assert!(attempt
            .protocol_v2_round_trip_plan
            .app_transport_blocker
            .contains("rc-qp-send-recv"));
        if attempt.native_rdma_probe.native_library_loaded {
            assert!(attempt.native_rdma_probe.rdma_enabled.is_some());
            assert!(attempt.native_rdma_probe.host_buffer_plan_checked);
            assert_eq!(
                attempt.native_rdma_probe.host_buffer_plan_span_aligned,
                Some(true)
            );
            assert!(
                attempt.native_rdma_probe.rc_send_recv_loopback_ok
                    || attempt.native_rdma_probe.rc_send_recv_loopback_unavailable
            );
            assert!(
                attempt.native_rdma_probe.rc_protocol_v2_loopback_ok
                    || attempt
                        .native_rdma_probe
                        .rc_protocol_v2_loopback_unavailable
            );
        } else {
            assert!(attempt.native_rdma_probe.native_library_error.is_some());
        }
        assert_eq!(attempt.chain_hops, 75);
        assert_eq!(
            attempt.protocol_v2_chain_plans.len(),
            attempt.payloads.len()
        );
        assert_eq!(
            attempt
                .payloads
                .iter()
                .map(|payload| payload.row_count)
                .collect::<Vec<_>>(),
            vec![1, 2, 4, 8, 16, 64, 256, 512]
        );
        assert_eq!(
            attempt
                .payloads
                .iter()
                .map(|payload| payload.source_kind.as_str())
                .collect::<Vec<_>>(),
            vec![
                "decode",
                "mtp_verify",
                "mtp_verify",
                "mtp_verify",
                "prefill",
                "prefill",
                "prefill",
                "prefill"
            ]
        );
        for (payload, chain_plan) in attempt
            .payloads
            .iter()
            .zip(attempt.protocol_v2_chain_plans.iter())
        {
            assert_eq!(
                payload.request_logical_payload_bytes,
                payload.response_logical_payload_bytes
            );
            assert!(payload.request_frame_buffer_stable);
            assert!(payload.response_frame_buffer_stable);
            assert!(payload.request_frame_arena_stable);
            assert!(payload.response_frame_arena_stable);
            assert!(payload.host_memory_registration_required);
            assert_eq!(payload.frame_arena_registration_alignment_bytes, 4096);
            assert!(payload.request_wire_bytes > payload.request_logical_payload_bytes);
            assert!(payload.response_wire_bytes > payload.response_logical_payload_bytes);
            assert!(payload.request_frame_buffer_capacity_bytes >= payload.request_wire_bytes);
            assert!(payload.response_frame_buffer_capacity_bytes >= payload.response_wire_bytes);
            assert_eq!(
                payload.request_frame_arena_capacity_bytes,
                payload.request_frame_buffer_capacity_bytes
            );
            assert_eq!(
                payload.response_frame_arena_capacity_bytes,
                payload.response_frame_buffer_capacity_bytes
            );
            assert_eq!(payload.request_hidden_row_view_count, payload.row_count);
            assert_eq!(
                payload.response_partial_output_row_view_count,
                payload.row_count
            );
            assert_eq!(
                payload.request_hidden_row_view_payload_bytes,
                payload.request_logical_payload_bytes
            );
            assert_eq!(
                payload.response_partial_output_row_view_payload_bytes,
                payload.response_logical_payload_bytes
            );
            assert!(payload.request_row_views_cover_payload);
            assert!(payload.response_row_views_cover_payload);
            assert!(payload.request_registered_span_aligned);
            assert!(payload.response_registered_span_aligned);
            assert!(payload.response_generated_by_route_dependent_executor);
            assert!(payload.response_differs_from_request_payload);
            assert_eq!(payload.request_registered_span_bytes % 4096, 0);
            assert_eq!(payload.response_registered_span_bytes % 4096, 0);
            assert!(
                payload.request_registered_span_bytes >= payload.request_frame_arena_capacity_bytes
            );
            assert!(
                payload.response_registered_span_bytes
                    >= payload.response_frame_arena_capacity_bytes
            );
            assert_eq!(
                payload.total_registered_span_bytes,
                payload.request_registered_span_bytes + payload.response_registered_span_bytes
            );
            assert_eq!(
                payload.request_registration_slack_bytes,
                payload.request_registered_span_bytes - payload.request_wire_bytes
            );
            assert_eq!(
                payload.response_registration_slack_bytes,
                payload.response_registered_span_bytes - payload.response_wire_bytes
            );
            assert_eq!(chain_plan.row_count, payload.row_count);
            assert_eq!(chain_plan.source_kind, payload.source_kind);
            assert_eq!(chain_plan.hops, 75);
            assert_eq!(
                chain_plan.request_wire_bytes_per_hop,
                payload.request_wire_bytes
            );
            assert_eq!(
                chain_plan.response_wire_bytes_per_hop,
                payload.response_wire_bytes
            );
            assert_eq!(
                chain_plan.registered_span_bytes_per_hop,
                payload.total_registered_span_bytes
            );
            assert_eq!(
                chain_plan.total_registered_span_bytes,
                payload.total_registered_span_bytes * 75
            );
            assert!(chain_plan.uses_registered_frame_spans);
            assert!(chain_plan.uses_request_response_row_views);
        }
        assert_eq!(
            attempt.protocol_v2_chain_plans[0].chain_kind,
            "decode_sparse_moe_chain"
        );
        assert_eq!(
            attempt.protocol_v2_chain_plans[1].chain_kind,
            "mtp_verify_sparse_moe_chain"
        );
        assert_eq!(
            attempt.protocol_v2_chain_plans[4].chain_kind,
            "prefill_sparse_moe_chain"
        );
        assert_eq!(
            attempt.protocol_v2_chain_plans[0].total_registered_span_bytes,
            2_457_600
        );
        assert_eq!(
            attempt.protocol_v2_chain_plans[7].total_registered_span_bytes,
            946_176_000
        );
    }

    #[test]
    fn app_protocol_v2_payloads_reject_non_hidden_row_sizes() {
        let err = protocol_v2_payloads_for_benchmark(&[4096])
            .unwrap_err()
            .to_string();
        assert!(err.contains("not an exact multiple of hidden row bytes 12288"));
    }

    #[test]
    fn app_protocol_v2_control_plane_dry_run_tracks_roles() {
        let payloads = parse_payload_bytes("12288").unwrap();
        let server =
            glmrt_verbs_app_benchmark_attempt("app-server", Some("emu".to_owned()), &payloads)
                .unwrap();
        let client =
            glmrt_verbs_app_benchmark_attempt("app-client", Some("emu".to_owned()), &payloads)
                .unwrap();
        let capability =
            glmrt_verbs_app_benchmark_attempt("app-capability", None, &payloads).unwrap();

        assert_eq!(server.protocol_v2_control_plane_dry_run.app_role, "server");
        assert_eq!(
            server.protocol_v2_control_plane_dry_run.local_endpoint_role,
            "server"
        );
        assert_eq!(
            server.protocol_v2_control_plane_dry_run.peer_endpoint_host,
            "emu"
        );
        assert_eq!(
            server
                .protocol_v2_control_plane_dry_run
                .validation
                .server_host,
            "local-server"
        );
        assert!(
            server
                .protocol_v2_control_plane_dry_run
                .validation
                .validated
        );
        assert_eq!(server.protocol_v2_round_trip_plan.client_host, "emu");
        assert_eq!(
            server.protocol_v2_round_trip_plan.server_host,
            "local-server"
        );

        assert_eq!(client.protocol_v2_control_plane_dry_run.app_role, "client");
        assert_eq!(
            client.protocol_v2_control_plane_dry_run.local_endpoint_role,
            "client"
        );
        assert_eq!(
            client.protocol_v2_control_plane_dry_run.peer_endpoint_host,
            "emu"
        );
        assert_eq!(
            client
                .protocol_v2_control_plane_dry_run
                .validation
                .client_host,
            "local-client"
        );
        assert!(
            client
                .protocol_v2_control_plane_dry_run
                .validation
                .validated
        );
        assert_eq!(
            client.protocol_v2_round_trip_plan.client_host,
            "local-client"
        );
        assert_eq!(client.protocol_v2_round_trip_plan.server_host, "emu");

        assert_eq!(
            capability.protocol_v2_control_plane_dry_run.app_role,
            "capability"
        );
        assert_eq!(
            capability
                .protocol_v2_control_plane_dry_run
                .local_endpoint_role,
            "none"
        );
        assert!(
            !capability
                .protocol_v2_control_plane_dry_run
                .peer_endpoint_required
        );
        assert!(
            capability
                .protocol_v2_control_plane_dry_run
                .validation
                .validated
        );
        assert!(
            capability
                .protocol_v2_control_plane_dry_run
                .validates_peer_qp_psn_gid
        );
        assert_eq!(
            capability.protocol_v2_round_trip_plan.total_work_requests,
            4
        );
        assert!(
            capability
                .protocol_v2_round_trip_plan
                .request_response_headers_match
        );
    }
}
