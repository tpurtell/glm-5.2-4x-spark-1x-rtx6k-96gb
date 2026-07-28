use anyhow::Result;
use glmrt_core::{
    ExpertRequest, ExpertRow, LayerWaveMode, RouteEntry, GLM52_HIDDEN_BF16_BYTES, GLM52_HIDDEN_SIZE,
};

use super::common::request_with_rows;
use crate::{
    protocol_v2_echo_loopback_response, protocol_v2_synthetic_response, ExpertProtocolV2Request,
    ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor, ExpertV2Dtype, ExpertV2SourceKind,
    PROTOCOL_V2_ECHO_EXECUTOR, PROTOCOL_V2_SYNTHETIC_ROUTE_EXECUTOR,
};
use crate::{synthetic_expert_response, ProtocolV2ExpertExecutor, SyntheticRouteExecutor};

#[test]
fn prefill_shaped_expert_service_accepts_c_16_64_256() -> Result<()> {
    for row_count in [16, 64, 256] {
        let request = request_with_rows(
            60 + row_count as u64,
            row_count,
            GLM52_HIDDEN_SIZE,
            LayerWaveMode::Prefill,
        );
        let response = synthetic_expert_response(&request)?;

        assert_eq!(request.header().wave_mode, Some(LayerWaveMode::Prefill));
        assert_eq!(
            request.header().logical_bf16_payload_bytes,
            Some(row_count * GLM52_HIDDEN_BF16_BYTES)
        );
        assert_eq!(response.partial_outputs.len(), row_count);
        assert_eq!(response.partial_outputs[0].len(), GLM52_HIDDEN_SIZE);
    }
    Ok(())
}

#[test]
fn synthetic_nvfp4_bf16_output_depends_on_route_ids() -> Result<()> {
    let hidden = (0..32)
        .map(|idx| idx as f32 / 16.0 - 1.0)
        .collect::<Vec<_>>();
    let request_for_expert = |request_id, expert_id| ExpertRequest {
        protocol_version: 1,
        request_id,
        placement_version: "synthetic-nvfp4-bf16-test".to_owned(),
        layer_id: 3,
        hidden_dim: hidden.len() as u32,
        wave: None,
        rows: vec![ExpertRow {
            row_id: 0,
            hidden: hidden.clone(),
            routes: vec![RouteEntry {
                expert_id,
                gate: 1.0,
            }],
        }],
    };

    let expert_7 = synthetic_expert_response(&request_for_expert(70, 7))?;
    let expert_8 = synthetic_expert_response(&request_for_expert(71, 8))?;

    assert_eq!(expert_7.partial_outputs[0].len(), hidden.len());
    assert!(expert_7.partial_outputs[0]
        .iter()
        .all(|value| value.is_finite()));
    assert_ne!(expert_7.partial_outputs[0], expert_8.partial_outputs[0]);
    Ok(())
}

#[test]
fn protocol_v2_echo_executor_is_transport_loopback_only() -> Result<()> {
    let request = protocol_v2_tiny_request(80, 7, 1.0, 0.0)?;
    let response = protocol_v2_echo_loopback_response(&request)?;

    assert_eq!(
        SyntheticRouteExecutor.name(),
        PROTOCOL_V2_SYNTHETIC_ROUTE_EXECUTOR
    );
    assert_eq!(PROTOCOL_V2_ECHO_EXECUTOR, "protocol-v2-echo-loopback");
    assert_eq!(response.header.output_dim, request.header.hidden_dim);
    assert_eq!(
        response.header.output_row_stride_bytes,
        request.header.hidden_row_stride_bytes
    );
    assert_eq!(response.partial_output_payload, request.hidden_payload);
    Ok(())
}

#[test]
fn protocol_v2_synthetic_route_executor_depends_on_routes_gates_and_hidden() -> Result<()> {
    let route_7 = protocol_v2_synthetic_response(&protocol_v2_tiny_request(81, 7, 1.0, 0.0)?)?;
    let route_8 = protocol_v2_synthetic_response(&protocol_v2_tiny_request(82, 8, 1.0, 0.0)?)?;
    let half_gate = protocol_v2_synthetic_response(&protocol_v2_tiny_request(83, 7, 0.5, 0.0)?)?;
    let shifted_hidden =
        protocol_v2_synthetic_response(&protocol_v2_tiny_request(84, 7, 1.0, 0.25)?)?;

    assert_eq!(route_7.header.output_dim, 16);
    assert_eq!(route_7.header.output_dtype, ExpertV2Dtype::Bf16);
    assert_eq!(route_7.partial_output_payload.len(), 32);
    assert_ne!(
        route_7.partial_output_payload,
        route_8.partial_output_payload
    );
    assert_ne!(
        route_7.partial_output_payload,
        half_gate.partial_output_payload
    );
    assert_ne!(
        route_7.partial_output_payload,
        shifted_hidden.partial_output_payload
    );
    assert_ne!(
        route_7.partial_output_payload,
        protocol_v2_tiny_request(85, 7, 1.0, 0.0)?.hidden_payload
    );
    Ok(())
}

#[test]
fn synthetic_partitioned_routes_sum_to_full_output() -> Result<()> {
    let rows = (0..3)
        .map(|row_id| ExpertRow {
            row_id,
            hidden: (0..32)
                .map(|idx| (row_id as f32 * 0.25) + (idx as f32 / 31.0) - 0.5)
                .collect(),
            routes: vec![
                RouteEntry {
                    expert_id: 3,
                    gate: 0.10,
                },
                RouteEntry {
                    expert_id: 17,
                    gate: 0.20,
                },
                RouteEntry {
                    expert_id: 42,
                    gate: 0.30,
                },
                RouteEntry {
                    expert_id: 99,
                    gate: 0.40,
                },
            ],
        })
        .collect::<Vec<_>>();
    let request_with_row_routes = |request_id, route_range: std::ops::Range<usize>| ExpertRequest {
        protocol_version: 1,
        request_id,
        placement_version: "synthetic-partition-test".to_owned(),
        layer_id: 12,
        hidden_dim: 32,
        wave: None,
        rows: rows
            .iter()
            .cloned()
            .map(|mut row| {
                row.routes = row.routes[route_range.clone()].to_vec();
                row
            })
            .collect(),
    };

    let full = synthetic_expert_response(&request_with_row_routes(72, 0..4))?;
    let left = synthetic_expert_response(&request_with_row_routes(73, 0..2))?;
    let right = synthetic_expert_response(&request_with_row_routes(74, 2..4))?;

    for row_idx in 0..rows.len() {
        for col_idx in 0..32 {
            let partitioned =
                left.partial_outputs[row_idx][col_idx] + right.partial_outputs[row_idx][col_idx];
            let full_value = full.partial_outputs[row_idx][col_idx];
            assert!(
                (partitioned - full_value).abs() < 1.0e-6,
                "row {row_idx} col {col_idx}: {partitioned} != {full_value}"
            );
        }
    }
    Ok(())
}

fn protocol_v2_tiny_request(
    request_id: u64,
    expert_id: u32,
    gate_weight: f32,
    hidden_offset: f32,
) -> Result<ExpertProtocolV2Request> {
    let rows = vec![ExpertProtocolV2RowDescriptor {
        row_id: 0,
        source_kind: ExpertV2SourceKind::Decode,
        source_request_id: request_id,
        token_position: 0,
        route_offset: 0,
        route_count: 1,
    }];
    let routes = vec![ExpertProtocolV2RouteEntry {
        row_index: 0,
        expert_id,
        gate_weight,
    }];
    let hidden_payload = (0..16)
        .flat_map(|idx| {
            let value = idx as f32 / 16.0 - 0.5 + hidden_offset;
            let bf16 = (value.to_bits() >> 16) as u16;
            bf16.to_le_bytes()
        })
        .collect::<Vec<_>>();
    ExpertProtocolV2Request::new(
        request_id,
        0x51CE,
        3,
        16,
        ExpertV2Dtype::Bf16,
        rows,
        routes,
        hidden_payload,
    )
}
