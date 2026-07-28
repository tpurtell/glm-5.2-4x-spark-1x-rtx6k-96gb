use crate::{
    protocol_v2_inproc_roundtrip_arena_response_view, protocol_v2_synthetic_response,
    tcp_protocol_v2_roundtrip_arena_response_view, verbs_host_app_transport_blocker,
    verbs_host_available, verbs_host_capabilities, verbs_host_preflight,
    verbs_host_protocol_v2_endpoint_plan, verbs_host_protocol_v2_handshake_contract,
    verbs_host_protocol_v2_round_trip_plan, verbs_host_validate_protocol_v2_handshake,
    ExpertProtocolV2FrameArena, ExpertProtocolV2Request, ExpertProtocolV2RouteEntry,
    ExpertProtocolV2RowDescriptor, ExpertV2Dtype, ExpertV2SourceKind, TcpTransportConfig,
    VerbsHostProtocolV2EndpointPlan, VerbsHostRcEndpointDescriptor, DEBUG_JSON_FRAME_PROTOCOL,
    EXPERT_PROTOCOL_V2_FRAME_PROTOCOL, VERBS_HOST_APP_TRANSPORT_STATUS,
};

use super::common::{protocol_v2_request, spawn_protocol_v2_server};

#[test]
fn verbs_host_capability_reports_environment_availability() {
    let caps = verbs_host_capabilities();
    assert!(caps.supports_rdma);
    assert!(caps.app_transport_implemented);
    assert_eq!(caps.app_transport_status, VERBS_HOST_APP_TRANSPORT_STATUS);
    let _available = verbs_host_available();
}

#[test]
fn verbs_host_preflight_reports_rdma_device_state() {
    match verbs_host_preflight() {
        Ok(preflight) => {
            assert_eq!(preflight.infiniband_path, "/dev/infiniband");
            assert_eq!(preflight.frame_protocol, EXPERT_PROTOCOL_V2_FRAME_PROTOCOL);
            assert!(preflight.requires_pinned_host_memory);
            assert!(verbs_host_available());
        }
        Err(err) => {
            let err = err.to_string();
            assert!(err.contains("/dev/infiniband"));
            assert!(!err.contains("not implemented"));
            assert!(!verbs_host_available());
        }
    }
}

#[test]
fn verbs_host_app_transport_status_uses_real_rc_qp() {
    let blocker = verbs_host_app_transport_blocker();
    assert!(blocker.contains("app transport is implemented"));
    assert!(VERBS_HOST_APP_TRANSPORT_STATUS.contains("rc-qp-send-recv"));
    assert!(!blocker.contains(DEBUG_JSON_FRAME_PROTOCOL));
}

#[test]
fn verbs_host_protocol_v2_endpoint_plan_uses_frame_views_and_registered_spans() {
    let request = endpoint_plan_request(16);
    let response = protocol_v2_synthetic_response(&request).unwrap();
    let request_frame = request.encode().unwrap();
    let response_frame = response.encode().unwrap();

    let plan = verbs_host_protocol_v2_endpoint_plan(&request_frame, &response_frame, 4096).unwrap();

    assert_eq!(plan.protocol, "ExpertProtocolV2");
    assert_eq!(plan.data_plane, "rc-qp-send-recv");
    assert_eq!(plan.control_plane, "tcp-qp-gid-psn-handshake");
    assert_eq!(plan.memory, "registered-host-frame-arenas");
    assert_eq!(plan.polling, "busy-poll-cq");
    assert_eq!(plan.request_frame_bytes, request_frame.len());
    assert_eq!(plan.response_frame_bytes, response_frame.len());
    assert_eq!(plan.request_logical_payload_bytes, 16 * 12_288);
    assert_eq!(plan.response_logical_payload_bytes, 16 * 12_288);
    assert_eq!(plan.request_rows, 16);
    assert_eq!(plan.response_rows, 16);
    assert_eq!(plan.registration_alignment_bytes, 4096);
    assert_eq!(plan.request_registered_span_bytes % 4096, 0);
    assert_eq!(plan.response_registered_span_bytes % 4096, 0);
    assert_eq!(
        plan.total_registered_span_bytes,
        plan.request_registered_span_bytes + plan.response_registered_span_bytes
    );
    assert_eq!(
        plan.request_registration_slack_bytes,
        plan.request_registered_span_bytes - plan.request_frame_bytes
    );
    assert_eq!(
        plan.response_registration_slack_bytes,
        plan.response_registered_span_bytes - plan.response_frame_bytes
    );
    assert!(plan.request_registered_span_aligned);
    assert!(plan.response_registered_span_aligned);
    assert_eq!(plan.queue_pairs_per_peer, 1);
    assert_eq!(plan.send_work_requests_per_roundtrip, 2);
    assert_eq!(plan.recv_work_requests_per_roundtrip, 2);
    assert_eq!(plan.scatter_gather_entries_per_message, 1);
    assert!(plan.requires_peer_qp_num);
    assert!(plan.requires_peer_psn);
    assert!(plan.requires_peer_gid);
    assert!(plan.requires_registered_host_buffers);
    assert!(plan.app_transport_implemented);
    assert_eq!(plan.app_transport_blocker, VERBS_HOST_APP_TRANSPORT_STATUS);
}

#[test]
fn verbs_host_protocol_v2_endpoint_plan_rejects_bad_frames() {
    let request = endpoint_plan_request(1);
    let response = protocol_v2_synthetic_response(&request).unwrap();
    let response_frame = response.encode().unwrap();
    let err = verbs_host_protocol_v2_endpoint_plan(b"not-protocol-v2", &response_frame, 4096)
        .unwrap_err()
        .to_string();
    assert!(err.contains("request frame is not valid ProtocolV2"));
}

#[tokio::test]
async fn verbs_host_protocol_v2_endpoint_plan_matches_tcp_hot_view_frames() {
    let (addr, shutdown) = spawn_protocol_v2_server().await.unwrap();
    for (request_id, row_count, source_kind) in [
        (6_000, 1, ExpertV2SourceKind::Decode),
        (6_001, 8, ExpertV2SourceKind::MtpVerify),
        (6_002, 16, ExpertV2SourceKind::Prefill),
    ] {
        let request = protocol_v2_request(request_id, row_count, source_kind).unwrap();
        let mut inproc_arena = ExpertProtocolV2FrameArena::with_capacities(
            request.wire_stats().wire_bytes,
            request.wire_stats().wire_bytes,
        );
        let mut tcp_arena = ExpertProtocolV2FrameArena::with_capacities(
            request.wire_stats().wire_bytes,
            request.wire_stats().wire_bytes,
        );

        let (inproc_header, inproc_payload, inproc_wire_stats) = {
            let view =
                protocol_v2_inproc_roundtrip_arena_response_view(&request, &mut inproc_arena)
                    .await
                    .unwrap();
            assert!(!view.debug_checksum_enabled());
            (
                view.header.clone(),
                view.partial_output_payload().to_vec(),
                view.wire_stats(),
            )
        };

        {
            let view = tcp_protocol_v2_roundtrip_arena_response_view(
                addr,
                &request,
                TcpTransportConfig::default(),
                &mut tcp_arena,
            )
            .await
            .unwrap();
            assert!(!view.debug_checksum_enabled());
            assert_eq!(view.header, inproc_header);
            assert_eq!(view.wire_stats(), inproc_wire_stats);
            assert_eq!(view.partial_output_payload(), inproc_payload.as_slice());
        }

        let mut tcp_request_frame = tcp_arena.request_frame().to_vec();
        tcp_request_frame.extend_from_slice(&request.hidden_payload);
        let plan = verbs_host_protocol_v2_endpoint_plan(
            &tcp_request_frame,
            tcp_arena.response_frame(),
            4096,
        )
        .unwrap();

        assert_eq!(plan.protocol, "ExpertProtocolV2");
        assert_ne!(plan.protocol, DEBUG_JSON_FRAME_PROTOCOL);
        assert_eq!(plan.data_plane, "rc-qp-send-recv");
        assert_eq!(plan.memory, "registered-host-frame-arenas");
        assert_eq!(plan.request_frame_bytes, request.wire_stats().wire_bytes);
        assert_eq!(plan.response_frame_bytes, inproc_wire_stats.wire_bytes);
        assert_eq!(
            plan.request_logical_payload_bytes,
            request.wire_stats().logical_payload_bytes
        );
        assert_eq!(
            plan.response_logical_payload_bytes,
            inproc_wire_stats.logical_payload_bytes
        );
        assert_eq!(plan.request_rows, row_count);
        assert_eq!(plan.response_rows, row_count);
        assert!(plan.request_registered_span_aligned);
        assert!(plan.response_registered_span_aligned);
        assert!(plan.requires_registered_host_buffers);
        assert!(plan.app_transport_implemented);
        assert_eq!(plan.app_transport_blocker, VERBS_HOST_APP_TRANSPORT_STATUS);
    }
    let _ = shutdown.send(());
}

#[test]
fn verbs_host_protocol_v2_handshake_contract_validates_endpoint_descriptors() {
    let plan = endpoint_plan(16);
    let contract = verbs_host_protocol_v2_handshake_contract(&plan);

    assert_eq!(contract.protocol, "ExpertProtocolV2");
    assert_eq!(contract.control_plane, "tcp-qp-gid-psn-handshake");
    assert!(contract.descriptor_fields.contains(&"qp_num".to_owned()));
    assert!(contract.descriptor_fields.contains(&"gid_hex".to_owned()));
    assert_eq!(contract.client_role, "client");
    assert_eq!(contract.server_role, "server");
    assert_eq!(contract.client_send_frame_bytes, plan.request_frame_bytes);
    assert_eq!(contract.client_recv_frame_bytes, plan.response_frame_bytes);
    assert_eq!(contract.server_send_frame_bytes, plan.response_frame_bytes);
    assert_eq!(contract.server_recv_frame_bytes, plan.request_frame_bytes);
    assert!(contract.requires_peer_qp_num);
    assert!(contract.requires_peer_psn);
    assert!(contract.requires_peer_gid);
    assert!(contract.requires_registered_host_buffers);
    assert!(contract.descriptor_validation_available);

    let client = endpoint_descriptor("client", "kiwi", &plan);
    let server = endpoint_descriptor("server", "emu", &plan);
    let validation = verbs_host_validate_protocol_v2_handshake(&plan, &client, &server).unwrap();

    assert_eq!(validation.protocol, "ExpertProtocolV2");
    assert_eq!(validation.control_plane, "tcp-qp-gid-psn-handshake");
    assert_eq!(validation.client_host, "kiwi");
    assert_eq!(validation.server_host, "emu");
    assert!(validation.client_sends_request);
    assert!(validation.server_sends_response);
    assert!(validation.peer_qp_num_present);
    assert!(validation.peer_psn_present);
    assert!(validation.peer_gid_present);
    assert!(validation.registered_spans_match_endpoint_plan);
    assert!(validation.validated);
}

#[test]
fn verbs_host_protocol_v2_handshake_rejects_invalid_endpoint_descriptors() {
    let plan = endpoint_plan(1);
    let client = endpoint_descriptor("client", "kiwi", &plan);
    let server = endpoint_descriptor("server", "emu", &plan);

    let mut bad_qp = client.clone();
    bad_qp.qp_num = 0;
    let err = verbs_host_validate_protocol_v2_handshake(&plan, &bad_qp, &server)
        .unwrap_err()
        .to_string();
    assert!(err.contains("qp_num must be non-zero"));

    let mut bad_gid = client.clone();
    bad_gid.gid_hex = "not-a-gid".to_owned();
    let err = verbs_host_validate_protocol_v2_handshake(&plan, &bad_gid, &server)
        .unwrap_err()
        .to_string();
    assert!(err.contains("gid_hex must be 32 hex characters"));

    let mut bad_direction = server.clone();
    bad_direction.recv_frame_bytes = plan.response_frame_bytes;
    let err = verbs_host_validate_protocol_v2_handshake(&plan, &client, &bad_direction)
        .unwrap_err()
        .to_string();
    assert!(err.contains("server recv_frame_bytes"));
}

#[test]
fn verbs_host_protocol_v2_round_trip_plan_binds_frames_to_validated_endpoints() {
    let request = endpoint_plan_request(8);
    let response = protocol_v2_synthetic_response(&request).unwrap();
    let request_frame = request.encode().unwrap();
    let response_frame = response.encode().unwrap();
    let plan = verbs_host_protocol_v2_endpoint_plan(&request_frame, &response_frame, 4096).unwrap();
    let client = endpoint_descriptor("client", "kiwi", &plan);
    let server = endpoint_descriptor("server", "emu", &plan);
    let validation = verbs_host_validate_protocol_v2_handshake(&plan, &client, &server).unwrap();

    let round_trip =
        verbs_host_protocol_v2_round_trip_plan(&plan, &validation, &request_frame, &response_frame)
            .unwrap();

    assert_eq!(round_trip.protocol, "ExpertProtocolV2");
    assert_eq!(round_trip.data_plane, "rc-qp-send-recv");
    assert_eq!(round_trip.control_plane, "tcp-qp-gid-psn-handshake");
    assert_eq!(round_trip.memory, "registered-host-frame-arenas");
    assert_eq!(round_trip.polling, "busy-poll-cq");
    assert_eq!(round_trip.client_host, "kiwi");
    assert_eq!(round_trip.server_host, "emu");
    assert_eq!(round_trip.request_id, request.header.request_id);
    assert_eq!(
        round_trip.placement_version,
        request.header.placement_version
    );
    assert_eq!(round_trip.layer_id, request.header.layer_id);
    assert_eq!(round_trip.request_frame_bytes, plan.request_frame_bytes);
    assert_eq!(round_trip.response_frame_bytes, plan.response_frame_bytes);
    assert_eq!(
        round_trip.request_logical_payload_bytes,
        plan.request_logical_payload_bytes
    );
    assert_eq!(
        round_trip.response_logical_payload_bytes,
        plan.response_logical_payload_bytes
    );
    assert_eq!(round_trip.request_rows, 8);
    assert_eq!(round_trip.response_rows, 8);
    assert_eq!(
        round_trip.request_registered_span_bytes,
        plan.request_registered_span_bytes
    );
    assert_eq!(
        round_trip.response_registered_span_bytes,
        plan.response_registered_span_bytes
    );
    assert_eq!(round_trip.client_send_frame_bytes, plan.request_frame_bytes);
    assert_eq!(
        round_trip.client_recv_frame_bytes,
        plan.response_frame_bytes
    );
    assert_eq!(
        round_trip.server_send_frame_bytes,
        plan.response_frame_bytes
    );
    assert_eq!(round_trip.server_recv_frame_bytes, plan.request_frame_bytes);
    assert_eq!(round_trip.client_send_work_requests, 1);
    assert_eq!(round_trip.client_recv_work_requests, 1);
    assert_eq!(round_trip.server_send_work_requests, 1);
    assert_eq!(round_trip.server_recv_work_requests, 1);
    assert_eq!(round_trip.total_work_requests, 4);
    assert_eq!(round_trip.scatter_gather_entries_per_message, 1);
    assert!(round_trip.request_frame_fits_registered_span);
    assert!(round_trip.response_frame_fits_registered_span);
    assert!(round_trip.request_response_headers_match);
    assert!(round_trip.endpoints_validated);
    assert!(round_trip.registered_spans_match_endpoint_plan);
    assert!(round_trip.app_transport_implemented);
    assert_eq!(
        round_trip.app_transport_blocker,
        VERBS_HOST_APP_TRANSPORT_STATUS
    );
}

#[test]
fn verbs_host_protocol_v2_round_trip_plan_rejects_mismatched_frames() {
    let request = endpoint_plan_request(1);
    let response = protocol_v2_synthetic_response(&request).unwrap();
    let request_frame = request.encode().unwrap();
    let response_frame = response.encode().unwrap();
    let plan = verbs_host_protocol_v2_endpoint_plan(&request_frame, &response_frame, 4096).unwrap();
    let client = endpoint_descriptor("client", "kiwi", &plan);
    let server = endpoint_descriptor("server", "emu", &plan);
    let validation = verbs_host_validate_protocol_v2_handshake(&plan, &client, &server).unwrap();

    let larger_request = endpoint_plan_request(2);
    let larger_response = protocol_v2_synthetic_response(&larger_request).unwrap();
    let err = verbs_host_protocol_v2_round_trip_plan(
        &plan,
        &validation,
        &larger_request.encode().unwrap(),
        &larger_response.encode().unwrap(),
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("request frame bytes"));
    assert!(err.contains("did not match endpoint plan"));
}

#[test]
fn verbs_host_protocol_v2_round_trip_plan_requires_validated_handshake() {
    let request = endpoint_plan_request(1);
    let response = protocol_v2_synthetic_response(&request).unwrap();
    let request_frame = request.encode().unwrap();
    let response_frame = response.encode().unwrap();
    let plan = verbs_host_protocol_v2_endpoint_plan(&request_frame, &response_frame, 4096).unwrap();
    let client = endpoint_descriptor("client", "kiwi", &plan);
    let server = endpoint_descriptor("server", "emu", &plan);
    let mut validation =
        verbs_host_validate_protocol_v2_handshake(&plan, &client, &server).unwrap();
    validation.validated = false;

    let err =
        verbs_host_protocol_v2_round_trip_plan(&plan, &validation, &request_frame, &response_frame)
            .unwrap_err()
            .to_string();
    assert!(err.contains("requires a validated handshake"));
}

fn endpoint_plan(row_count: usize) -> VerbsHostProtocolV2EndpointPlan {
    let request = endpoint_plan_request(row_count);
    let response = protocol_v2_synthetic_response(&request).unwrap();
    verbs_host_protocol_v2_endpoint_plan(
        &request.encode().unwrap(),
        &response.encode().unwrap(),
        4096,
    )
    .unwrap()
}

fn endpoint_descriptor(
    role: &str,
    host: &str,
    plan: &VerbsHostProtocolV2EndpointPlan,
) -> VerbsHostRcEndpointDescriptor {
    let client = role == "client";
    VerbsHostRcEndpointDescriptor {
        role: role.to_owned(),
        host: host.to_owned(),
        port_num: 1,
        qp_num: if client { 0x1234 } else { 0x5678 },
        psn: if client { 0x010203 } else { 0x040506 },
        gid_hex: if client {
            "00000000000000000000ffff0a000001".to_owned()
        } else {
            "00000000000000000000ffff0a000002".to_owned()
        },
        send_frame_bytes: if client {
            plan.request_frame_bytes
        } else {
            plan.response_frame_bytes
        },
        recv_frame_bytes: if client {
            plan.response_frame_bytes
        } else {
            plan.request_frame_bytes
        },
        send_registered_span_bytes: if client {
            plan.request_registered_span_bytes
        } else {
            plan.response_registered_span_bytes
        },
        recv_registered_span_bytes: if client {
            plan.response_registered_span_bytes
        } else {
            plan.request_registered_span_bytes
        },
        max_send_wr: 4,
        max_recv_wr: 4,
        max_sge: 1,
    }
}

fn endpoint_plan_request(row_count: usize) -> ExpertProtocolV2Request {
    let rows = (0..row_count)
        .map(|idx| ExpertProtocolV2RowDescriptor {
            row_id: idx as u64,
            source_kind: ExpertV2SourceKind::Prefill,
            source_request_id: 7,
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
    ExpertProtocolV2Request::new(
        7,
        1,
        3,
        6144,
        ExpertV2Dtype::Bf16,
        rows,
        routes,
        vec![0_u8; row_count * 12_288],
    )
    .unwrap()
}
