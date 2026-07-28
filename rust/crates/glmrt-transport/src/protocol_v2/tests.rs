use super::*;
use glmrt_core::{
    plan_completion_first_routes, CompletionRoutePlanEntry, DType, DecodeStep, ExpertBatch,
    ExpertBatchRoute, ExpertHostBatch, GraphBucket, LayerId, LayerWave, MtpVerifyBlock,
    PlacementPolicy, PositionId, PrefillChunk, Priority, EXPERT_HOSTS, GLM52_HIDDEN_SIZE,
};

const GLM52_HIDDEN_BF16_BYTES: usize = 12_288;

fn request(row_count: usize, source_kind: ExpertV2SourceKind) -> Result<ExpertProtocolV2Request> {
    let hidden_dim = GLM52_HIDDEN_SIZE as u32;
    let rows = (0..row_count)
        .map(|row| ExpertProtocolV2RowDescriptor {
            row_id: row as u64,
            source_kind,
            source_request_id: 1000 + row as u64,
            token_position: row as u64,
            route_offset: row as u32,
            route_count: 1,
        })
        .collect::<Vec<_>>();
    let routes = (0..row_count)
        .map(|row| ExpertProtocolV2RouteEntry {
            row_index: row as u32,
            expert_id: (row % 256) as u32,
            gate_weight: 1.0,
        })
        .collect::<Vec<_>>();
    let hidden_payload = (0..row_count * GLM52_HIDDEN_BF16_BYTES)
        .map(|idx| (idx % 251) as u8)
        .collect::<Vec<_>>();
    ExpertProtocolV2Request::new(
        42,
        7,
        13,
        hidden_dim,
        ExpertV2Dtype::Bf16,
        rows,
        routes,
        hidden_payload,
    )
}

#[test]
fn encode_decode_one_row_decode_request_has_12288_payload_bytes() -> Result<()> {
    let request = request(1, ExpertV2SourceKind::Decode)?;
    let encoded = request.encode()?;
    let decoded = ExpertProtocolV2Request::decode(&encoded)?;

    assert_eq!(decoded, request);
    assert_eq!(decoded.header.hidden_payload_bytes, 12_288);
    assert_eq!(
        decoded.header.hidden_row_stride_bytes,
        GLM52_HIDDEN_BF16_BYTES as u32
    );
    assert_eq!(decoded.wire_stats().logical_payload_bytes, 12_288);
    assert!(decoded.wire_stats().wire_bytes > decoded.wire_stats().logical_payload_bytes);
    Ok(())
}

#[test]
fn regular_decode_request_roundtrips_spark_reduction_flag() -> Result<()> {
    let request = request(1, ExpertV2SourceKind::Decode)?.with_spark_reduction();
    let encoded = request.encode()?;
    let decoded = ExpertProtocolV2Request::decode(&encoded)?;
    let view = ExpertProtocolV2RequestView::parse(&encoded)?;

    assert!(decoded.spark_reduction_enabled());
    assert!(view.spark_reduction_enabled());
    assert!(!decoded.stream_plan_enabled());
    assert!(!decoded.stream_data_enabled());
    Ok(())
}

#[test]
fn prefill_request_roundtrips_row_sharded_reduction_flag() -> Result<()> {
    let mut request = request(4, ExpertV2SourceKind::Prefill)?.with_spark_row_sharded_reduction();
    request.set_spark_collective_part_count(2)?;
    let encoded = request.encode()?;
    let decoded = ExpertProtocolV2Request::decode(&encoded)?;
    let view = ExpertProtocolV2RequestView::parse(&encoded)?;

    assert!(decoded.spark_reduction_enabled());
    assert!(decoded.spark_row_sharded_reduction_enabled());
    assert!(view.spark_reduction_enabled());
    assert!(view.spark_row_sharded_reduction_enabled());
    assert_eq!(decoded.spark_collective_part_count(), 2);
    assert_eq!(view.spark_collective_part_count(), 2);
    Ok(())
}

#[test]
fn striped_collective_part_count_requires_row_sharding_and_multiple_parts() -> Result<()> {
    let mut regular = request(4, ExpertV2SourceKind::Prefill)?;
    assert!(regular.set_spark_collective_part_count(2).is_err());

    let mut row_sharded = regular.with_spark_row_sharded_reduction();
    assert!(row_sharded.set_spark_collective_part_count(1).is_err());
    assert!(row_sharded
        .set_spark_collective_part_count(EXPERT_PROTOCOL_V2_MAX_SPARK_COLLECTIVE_PARTS + 1)
        .is_err());
    Ok(())
}

#[test]
fn forwarded_request_reuses_frame_and_strips_owner_flags() -> Result<()> {
    let mut request = request(4, ExpertV2SourceKind::Prefill)?
        .with_fp8_e4m3_row_scaled_response()
        .with_spark_row_sharded_reduction()
        .with_debug_checksum();
    request.set_spark_collective_part_count(2)?;
    let encoded = request.encode()?;
    let view = ExpertProtocolV2RequestView::parse(&encoded)?;
    let mut frame = ExpertProtocolV2FrameBuffer::with_capacity(encoded.len());

    let first_ptr = {
        let forwarded =
            frame.encode_regular_forwarded_request(&view, ExpertV2Dtype::Nvfp4E2m1Fp8E4m3)?;
        let decoded = ExpertProtocolV2Request::decode(forwarded)?;
        assert!(!decoded.spark_reduction_enabled());
        assert!(!decoded.spark_row_sharded_reduction_enabled());
        assert_eq!(decoded.spark_collective_part_count(), 0);
        assert!(!decoded.fp8_e4m3_row_scaled_response_enabled());
        assert!(decoded.nvfp4_e2m1_fp8_e4m3_response_enabled());
        assert!(decoded.debug_checksum_enabled());
        assert_eq!(decoded.rows, request.rows);
        assert_eq!(decoded.routes, request.routes);
        assert_eq!(decoded.hidden_payload, request.hidden_payload);
        ExpertProtocolV2RequestView::parse(forwarded)?.verify_checksum()?;
        forwarded.as_ptr()
    };
    let capacity = frame.capacity();
    let second_ptr = frame
        .encode_regular_forwarded_request(&view, ExpertV2Dtype::Bf16)?
        .as_ptr();

    assert_eq!(second_ptr, first_ptr);
    assert_eq!(frame.capacity(), capacity);
    Ok(())
}

#[test]
fn layer_block_request_roundtrips_without_precomputed_routes() -> Result<()> {
    let row = ExpertProtocolV2RowDescriptor {
        row_id: 0,
        source_kind: ExpertV2SourceKind::Decode,
        source_request_id: 991,
        token_position: 128,
        route_offset: 0,
        route_count: 0,
    };
    let request = ExpertProtocolV2Request::new(
        43,
        8,
        18,
        GLM52_HIDDEN_SIZE as u32,
        ExpertV2Dtype::Bf16,
        vec![row],
        Vec::new(),
        vec![0_u8; GLM52_HIDDEN_BF16_BYTES],
    )?
    .with_layer_block();
    let encoded = request.encode()?;
    let decoded = ExpertProtocolV2Request::decode(&encoded)?;
    let view = ExpertProtocolV2RequestView::parse(&encoded)?;

    assert!(decoded.layer_block_enabled());
    assert!(view.layer_block_enabled());
    assert_eq!(decoded.header.route_count, 0);
    assert_eq!(view.row(0)?.token_position, 128);

    let mut invalid = request.clone();
    invalid.rows[0].route_count = 1;
    assert!(invalid.validate().is_err());
    Ok(())
}

#[test]
fn row_scaled_fp8_response_negotiation_and_payload_roundtrip() -> Result<()> {
    let request = request(2, ExpertV2SourceKind::Prefill)?.with_fp8_e4m3_row_scaled_response();
    let request_wire = request.encode()?;
    let decoded_request = ExpertProtocolV2Request::decode(&request_wire)?;
    let request_view = ExpertProtocolV2RequestView::parse(&request_wire)?;
    assert!(decoded_request.fp8_e4m3_row_scaled_response_enabled());
    assert!(request_view.fp8_e4m3_row_scaled_response_enabled());

    let row_bytes = ExpertV2Dtype::Fp8E4m3RowScaled.row_bytes(GLM52_HIDDEN_SIZE)?;
    assert_eq!(row_bytes, 6_148);
    let payload = vec![0x35_u8; 2 * row_bytes];
    let response = ExpertProtocolV2Response::new(
        request.header.request_id,
        request.header.placement_version,
        request.header.layer_id,
        2,
        GLM52_HIDDEN_SIZE as u32,
        ExpertV2Dtype::Fp8E4m3RowScaled,
        ExpertProtocolV2Status::Ok,
        payload.clone(),
    )?;
    let response_wire = response.encode()?;
    let decoded_response = ExpertProtocolV2Response::decode(&response_wire)?;
    let response_view = ExpertProtocolV2ResponseView::parse(&response_wire)?;
    assert_eq!(decoded_response, response);
    assert_eq!(
        decoded_response.header.output_row_stride_bytes,
        row_bytes as u32
    );
    assert_eq!(
        response_view.partial_output_row_payload(1)?,
        &payload[row_bytes..]
    );
    Ok(())
}

#[test]
fn nvfp4_response_negotiation_and_payload_roundtrip() -> Result<()> {
    let request = request(2, ExpertV2SourceKind::Prefill)?.with_nvfp4_e2m1_fp8_e4m3_response();
    let request_wire = request.encode()?;
    let decoded_request = ExpertProtocolV2Request::decode(&request_wire)?;
    let request_view = ExpertProtocolV2RequestView::parse(&request_wire)?;
    assert!(decoded_request.nvfp4_e2m1_fp8_e4m3_response_enabled());
    assert!(request_view.nvfp4_e2m1_fp8_e4m3_response_enabled());

    let row_bytes = ExpertV2Dtype::Nvfp4E2m1Fp8E4m3.row_bytes(GLM52_HIDDEN_SIZE)?;
    assert_eq!(row_bytes, 3_456);
    let payload = vec![0x12_u8; 2 * row_bytes];
    let response = ExpertProtocolV2Response::new(
        request.header.request_id,
        request.header.placement_version,
        request.header.layer_id,
        2,
        GLM52_HIDDEN_SIZE as u32,
        ExpertV2Dtype::Nvfp4E2m1Fp8E4m3,
        ExpertProtocolV2Status::Ok,
        payload.clone(),
    )?;
    let response_wire = response.encode()?;
    let decoded_response = ExpertProtocolV2Response::decode(&response_wire)?;
    let response_view = ExpertProtocolV2ResponseView::parse(&response_wire)?;
    assert_eq!(decoded_response, response);
    assert_eq!(
        response_view.partial_output_row_payload(1)?,
        &payload[row_bytes..]
    );
    Ok(())
}

#[test]
fn encode_decode_nvfp4_hidden_exchange_uses_packed_values_plus_fp8_scales() -> Result<()> {
    let hidden_dim = GLM52_HIDDEN_SIZE as u32;
    let row_bytes = ExpertV2Dtype::Nvfp4E2m1Fp8E4m3.row_bytes(hidden_dim as usize)?;
    let payload = vec![7_u8; 2 * row_bytes];
    let request = ExpertProtocolV2Request::new(
        44,
        9,
        13,
        hidden_dim,
        ExpertV2Dtype::Nvfp4E2m1Fp8E4m3,
        (0..2)
            .map(|row| ExpertProtocolV2RowDescriptor {
                row_id: row,
                source_kind: ExpertV2SourceKind::Prefill,
                source_request_id: 1000,
                token_position: row,
                route_offset: 0,
                route_count: 0,
            })
            .collect(),
        Vec::new(),
        payload.clone(),
    )?;
    let encoded = request.encode()?;
    let decoded = ExpertProtocolV2Request::decode(&encoded)?;
    let view = ExpertProtocolV2RequestView::parse(&encoded)?;

    assert_eq!(row_bytes, 3_456);
    assert_eq!(decoded, request);
    assert_eq!(decoded.header.hidden_row_stride_bytes, row_bytes as u32);
    assert_eq!(view.hidden_row_payload(1)?, &payload[row_bytes..]);
    Ok(())
}

#[test]
fn streamed_ingress_plan_roundtrips_without_treating_plan_as_hidden_rows() -> Result<()> {
    let base = request(3, ExpertV2SourceKind::Prefill)?;
    let plan_payload = vec![0x50, 0x4c, 0x41, 0x4e, 1, 2, 3, 4];
    let plan = ExpertProtocolV2Request::new_stream_plan_with_hidden_stride(
        base.header.request_id,
        base.header.placement_version,
        base.header.layer_id,
        base.header.hidden_dim,
        base.header.hidden_dtype,
        base.header.hidden_row_stride_bytes,
        base.rows,
        base.routes,
        plan_payload.clone(),
    )?
    .with_fp8_e4m3_row_scaled_response()
    .with_spark_reduction()
    .with_debug_checksum();
    let encoded = plan.encode()?;
    let decoded = ExpertProtocolV2Request::decode(&encoded)?;
    let view = ExpertProtocolV2RequestView::parse(&encoded)?;

    assert_eq!(decoded, plan);
    assert!(decoded.stream_plan_enabled());
    assert!(!decoded.stream_data_enabled());
    assert!(view.stream_plan_enabled());
    assert!(decoded.spark_reduction_enabled());
    assert!(view.spark_reduction_enabled());
    assert_eq!(view.hidden_payload(), plan_payload);
    assert!(view.hidden_row_payload(0).is_err());
    view.verify_checksum()?;
    Ok(())
}

#[test]
fn streamed_ingress_data_roundtrips_offset_rows_and_final_marker() -> Result<()> {
    let stride = ExpertV2Dtype::Nvfp4E2m1Fp8E4m3.row_bytes(GLM52_HIDDEN_SIZE)?;
    let payload = vec![0x37_u8; 2 * stride];
    let data = ExpertProtocolV2Request::new_stream_data(
        91,
        7,
        13,
        GLM52_HIDDEN_SIZE as u32,
        ExpertV2Dtype::Nvfp4E2m1Fp8E4m3,
        stride as u32,
        64,
        2,
        payload.clone(),
        true,
    )?
    .with_nvfp4_e2m1_fp8_e4m3_response()
    .with_spark_reduction()
    .with_debug_checksum();
    let encoded = data.encode()?;
    let decoded = ExpertProtocolV2Request::decode(&encoded)?;
    let view = ExpertProtocolV2RequestView::parse(&encoded)?;

    assert_eq!(decoded, data);
    assert!(decoded.stream_data_enabled());
    assert!(decoded.stream_final_enabled());
    assert!(decoded.spark_reduction_enabled());
    assert!(view.spark_reduction_enabled());
    assert_eq!(decoded.stream_data_row_offset(), Some(64));
    assert!(decoded.rows.is_empty());
    assert!(decoded.routes.is_empty());
    assert_eq!(view.stream_data_row_offset(), Some(64));
    assert_eq!(view.hidden_row_payload(1)?, &payload[stride..]);
    assert!(view.row(0).is_err());
    assert!(view.route(0).is_err());
    view.verify_checksum()?;
    Ok(())
}

#[test]
fn streamed_ingress_rejects_conflicting_frame_flags() -> Result<()> {
    let mut request = request(1, ExpertV2SourceKind::Prefill)?;
    request.header.flags =
        EXPERT_PROTOCOL_V2_FLAG_STREAM_PLAN | EXPERT_PROTOCOL_V2_FLAG_STREAM_DATA;
    assert!(request.validate().is_err());

    request.header.flags = EXPERT_PROTOCOL_V2_FLAG_STREAM_FINAL;
    assert!(request.validate().is_err());
    Ok(())
}

#[test]
fn streamed_ingress_completion_plan_roundtrips_and_matches_request_routes() -> Result<()> {
    let rows = vec![
        ExpertProtocolV2RowDescriptor {
            row_id: 0,
            source_kind: ExpertV2SourceKind::Prefill,
            source_request_id: 10,
            token_position: 0,
            route_offset: 0,
            route_count: 2,
        },
        ExpertProtocolV2RowDescriptor {
            row_id: 1,
            source_kind: ExpertV2SourceKind::Prefill,
            source_request_id: 10,
            token_position: 1,
            route_offset: 2,
            route_count: 1,
        },
        ExpertProtocolV2RowDescriptor {
            row_id: 2,
            source_kind: ExpertV2SourceKind::Prefill,
            source_request_id: 10,
            token_position: 2,
            route_offset: 3,
            route_count: 1,
        },
    ];
    let routes = vec![
        ExpertProtocolV2RouteEntry {
            row_index: 0,
            expert_id: 5,
            gate_weight: 0.5,
        },
        ExpertProtocolV2RouteEntry {
            row_index: 0,
            expert_id: 9,
            gate_weight: 0.5,
        },
        ExpertProtocolV2RouteEntry {
            row_index: 1,
            expert_id: 5,
            gate_weight: 1.0,
        },
        ExpertProtocolV2RouteEntry {
            row_index: 2,
            expert_id: 9,
            gate_weight: 1.0,
        },
    ];
    let entries = routes
        .iter()
        .map(|route| CompletionRoutePlanEntry {
            row_index: route.row_index as usize,
            expert_id: route.expert_id as usize,
            intermediate_rows: 1_536,
        })
        .collect::<Vec<_>>();
    let completion = plan_completion_first_routes(&entries, rows.len(), 256)?;
    let plan =
        ExpertProtocolV2StreamPlan::from_completion_first(rows.len(), routes.len(), &completion)?;
    plan.validate_against_request(&rows, &routes)?;
    let decoded = ExpertProtocolV2StreamPlan::decode(&plan.encode()?)?;

    assert_eq!(decoded, plan);
    assert_eq!(decoded.activation_row_order, vec![1, 0, 2]);
    assert_eq!(decoded.groups[0].completed_rows, vec![1]);
    assert_eq!(decoded.groups[1].completed_rows, vec![0, 2]);

    let mut mixed_expert_plan = decoded;
    let first = mixed_expert_plan.groups[0].route_indices[1];
    let second = mixed_expert_plan.groups[1].route_indices[0];
    mixed_expert_plan.groups[0].route_indices[1] = second;
    mixed_expert_plan.groups[1].route_indices[0] = first;
    assert!(mixed_expert_plan
        .validate_against_request(&rows, &routes)
        .is_err());
    Ok(())
}

#[test]
fn route_entries_carry_gate_weights_as_bf16_on_wire() -> Result<()> {
    let hidden_dim = 2;
    let hidden_payload = vec![0_u8; hidden_dim * ExpertV2Dtype::Bf16.bytes_per_element()];
    let request = ExpertProtocolV2Request::new(
        43,
        8,
        13,
        hidden_dim as u32,
        ExpertV2Dtype::Bf16,
        vec![ExpertProtocolV2RowDescriptor {
            row_id: 0,
            source_kind: ExpertV2SourceKind::Decode,
            source_request_id: 1000,
            token_position: 0,
            route_offset: 0,
            route_count: 1,
        }],
        vec![ExpertProtocolV2RouteEntry {
            row_index: 0,
            expert_id: 17,
            gate_weight: 0.1,
        }],
        hidden_payload,
    )?;
    let expected_gate_bits = (0.1_f32.to_bits() >> 16) as u16;
    let expected_gate = f32::from_bits((expected_gate_bits as u32) << 16);
    let encoded = request.encode()?;
    let route_offset =
        EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN + EXPERT_PROTOCOL_V2_ROW_DESCRIPTOR_LEN;
    let decoded = ExpertProtocolV2Request::decode(&encoded)?;
    let view = ExpertProtocolV2RequestView::parse(&encoded)?;

    assert_eq!(EXPERT_PROTOCOL_V2_ROUTE_ENTRY_LEN, 10);
    assert_eq!(request.header.route_bytes, 10);
    assert_eq!(
        request.routes[0].gate_weight.to_bits(),
        expected_gate.to_bits()
    );
    assert_eq!(
        &encoded[route_offset + 8..route_offset + 10],
        &expected_gate_bits.to_le_bytes()
    );
    assert_eq!(
        decoded.routes[0].gate_weight.to_bits(),
        expected_gate.to_bits()
    );
    assert_eq!(
        view.route(0)?.gate_weight.to_bits(),
        expected_gate.to_bits()
    );
    assert_eq!(view.route_bytes().len(), EXPERT_PROTOCOL_V2_ROUTE_ENTRY_LEN);
    Ok(())
}

#[test]
fn request_header_carries_explicit_hidden_stride() -> Result<()> {
    let rows = (0..2)
        .map(|row| ExpertProtocolV2RowDescriptor {
            row_id: row,
            source_kind: ExpertV2SourceKind::Benchmark,
            source_request_id: 99,
            token_position: row,
            route_offset: 0,
            route_count: 0,
        })
        .collect::<Vec<_>>();
    let hidden_payload = vec![3_u8; 2 * 8];
    let request = ExpertProtocolV2Request::new_with_hidden_stride(
        91,
        92,
        3,
        3,
        ExpertV2Dtype::Bf16,
        8,
        rows,
        Vec::new(),
        hidden_payload.clone(),
    )?;
    let encoded = request.encode()?;
    let decoded = ExpertProtocolV2Request::decode(&encoded)?;
    let view = ExpertProtocolV2RequestView::parse(&encoded)?;

    assert_eq!(decoded, request);
    assert_eq!(decoded.header.hidden_dim, 3);
    assert_eq!(decoded.header.hidden_row_stride_bytes, 8);
    assert_eq!(
        decoded.header.hidden_payload_bytes,
        hidden_payload.len() as u64
    );
    assert_eq!(view.header.hidden_row_stride_bytes, 8);
    assert_eq!(view.hidden_payload(), hidden_payload.as_slice());
    Ok(())
}

#[test]
fn encode_decode_mtp_request_shapes() -> Result<()> {
    for row_count in [1, 2, 4, 8] {
        let request = request(row_count, ExpertV2SourceKind::MtpVerify)?;
        let decoded = ExpertProtocolV2Request::decode(&request.encode()?)?;

        assert_eq!(decoded.rows.len(), row_count);
        assert_eq!(
            decoded.wire_stats().logical_payload_bytes,
            row_count * GLM52_HIDDEN_BF16_BYTES
        );
        assert!(decoded
            .rows
            .iter()
            .all(|row| row.source_kind == ExpertV2SourceKind::MtpVerify));
    }
    Ok(())
}

#[test]
fn encode_decode_prefill_request_shapes() -> Result<()> {
    for row_count in [16, 64, 256, 512] {
        let request = request(row_count, ExpertV2SourceKind::Prefill)?;
        let decoded = ExpertProtocolV2Request::decode(&request.encode()?)?;

        assert_eq!(decoded.rows.len(), row_count);
        assert_eq!(
            decoded.header.hidden_payload_bytes as usize,
            row_count * GLM52_HIDDEN_BF16_BYTES
        );
        assert!(decoded
            .rows
            .iter()
            .all(|row| row.source_kind == ExpertV2SourceKind::Prefill));
    }
    Ok(())
}

#[test]
fn malformed_length_is_rejected() -> Result<()> {
    let request = request(1, ExpertV2SourceKind::Decode)?;
    let mut encoded = request.encode()?;
    encoded.truncate(encoded.len() - 1);

    let err = ExpertProtocolV2Request::decode(&encoded)
        .unwrap_err()
        .to_string();

    assert!(err.contains("wire bytes mismatch") || err.contains("length mismatch"));
    Ok(())
}

#[test]
fn hidden_row_stride_too_small_is_rejected() -> Result<()> {
    let request = request(1, ExpertV2SourceKind::Decode)?;
    let mut encoded = request.encode()?;
    encoded[88..92].copy_from_slice(&(1_u32).to_le_bytes());

    let err = ExpertProtocolV2Request::decode(&encoded)
        .unwrap_err()
        .to_string();

    assert!(err.contains("hidden row stride"));
    Ok(())
}

#[test]
fn row_descriptor_route_offsets_are_bounds_checked() {
    let mut request = request(2, ExpertV2SourceKind::Prefill).unwrap();
    request.rows[1].route_offset = 2;
    request.rows[1].route_count = 1;

    let err = request.validate().unwrap_err().to_string();

    assert!(err.contains("route range"));
}

#[test]
fn response_encode_decode_reports_wire_and_logical_bytes() -> Result<()> {
    let row_count = 2;
    let payload = vec![7_u8; row_count * GLM52_HIDDEN_BF16_BYTES];
    let response = ExpertProtocolV2Response::new(
        42,
        7,
        13,
        row_count as u32,
        GLM52_HIDDEN_SIZE as u32,
        ExpertV2Dtype::Bf16,
        ExpertProtocolV2Status::Ok,
        payload,
    )?;
    let decoded = ExpertProtocolV2Response::decode(&response.encode()?)?;

    assert_eq!(decoded, response);
    assert_eq!(decoded.header.executor_id, 0);
    assert_eq!(decoded.header.output_dim, GLM52_HIDDEN_SIZE as u32);
    assert_eq!(
        decoded.header.output_row_stride_bytes,
        GLM52_HIDDEN_BF16_BYTES as u32
    );
    assert_eq!(
        decoded.wire_stats().logical_payload_bytes,
        row_count * GLM52_HIDDEN_BF16_BYTES
    );
    assert!(decoded.wire_stats().wire_bytes > decoded.wire_stats().logical_payload_bytes);
    Ok(())
}

#[test]
fn response_header_carries_explicit_output_shape_and_stride() -> Result<()> {
    let row_count = 3;
    let output_dim = 5;
    let output_row_stride_bytes = 16;
    let payload = vec![11_u8; row_count * output_row_stride_bytes as usize];
    let executor_name = "protocol-v2-test-executor";
    let executor_id = expert_protocol_v2_compact_id(executor_name);
    let response = ExpertProtocolV2Response::new_with_output_stride(
        50,
        51,
        7,
        row_count as u32,
        output_dim,
        ExpertV2Dtype::Bf16,
        output_row_stride_bytes,
        ExpertProtocolV2Status::Ok,
        payload.clone(),
    )?
    .with_executor_name(executor_name);
    let encoded = response.encode()?;
    let decoded = ExpertProtocolV2Response::decode(&encoded)?;
    let view = ExpertProtocolV2ResponseView::parse(&encoded)?;

    assert_eq!(decoded, response);
    assert_eq!(decoded.header.output_dim, output_dim);
    assert_eq!(
        decoded.header.output_row_stride_bytes,
        output_row_stride_bytes
    );
    assert_eq!(decoded.header.output_payload_bytes, payload.len() as u64);
    assert_eq!(decoded.header.executor_id, executor_id);
    assert_eq!(view.header.output_dim, output_dim);
    assert_eq!(view.header.output_row_stride_bytes, output_row_stride_bytes);
    assert_eq!(view.header.executor_id, executor_id);
    assert_eq!(view.partial_output_payload(), payload.as_slice());
    Ok(())
}

#[test]
fn row_indexed_response_chunk_round_trips_without_changing_default_frames() -> Result<()> {
    let response = ExpertProtocolV2Response::new_with_output_stride(
        61,
        62,
        7,
        2,
        2,
        ExpertV2Dtype::Bf16,
        4,
        ExpertProtocolV2Status::Ok,
        vec![1, 2, 3, 4, 5, 6, 7, 8],
    )?
    .with_row_indices(vec![7, 2], true)?
    .with_debug_checksum();
    let encoded = response.encode()?;
    let decoded = ExpertProtocolV2Response::decode(&encoded)?;
    let view = ExpertProtocolV2ResponseView::parse(&encoded)?;

    assert_eq!(decoded, response);
    assert!(view.row_indexed());
    assert!(view.more_chunks());
    assert_eq!(view.request_row_index(0)?, 7);
    assert_eq!(view.request_row_index(1)?, 2);
    assert_eq!(view.partial_output_payload(), &[1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(
        encoded.len(),
        response.header_len() + 2 * std::mem::size_of::<u32>() + 8
    );
    view.verify_checksum()?;

    let duplicate = ExpertProtocolV2Response::new(
        61,
        62,
        7,
        2,
        2,
        ExpertV2Dtype::Bf16,
        ExpertProtocolV2Status::Ok,
        vec![0; 8],
    )?
    .with_row_indices(vec![3, 3], false);
    assert!(duplicate.is_err());
    Ok(())
}

#[test]
fn default_frames_omit_debug_checksum_from_hot_path() -> Result<()> {
    let request = request(1, ExpertV2SourceKind::Decode)?;
    let mut encoded_request = request.encode()?;
    assert!(!request.debug_checksum_enabled());
    assert!(!request.precompile_warmup_enabled());
    assert_eq!(request.header_len(), EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN);
    assert_eq!(encoded_request.len(), request.wire_stats().wire_bytes);
    assert_eq!(
        request
            .clone()
            .with_debug_checksum()
            .wire_stats()
            .wire_bytes,
        request.wire_stats().wire_bytes + CHECKSUM_LEN
    );

    let request_view = ExpertProtocolV2RequestView::parse(&encoded_request)?;
    assert!(!request_view.debug_checksum_enabled());
    assert!(!request_view.precompile_warmup_enabled());
    assert_eq!(
        request_view.header_len(),
        EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN
    );
    assert!(request_view
        .verify_checksum()
        .unwrap_err()
        .to_string()
        .contains("debug checksum flag is not set"));
    let payload_start = request_payload_offset(&request);
    encoded_request[payload_start] ^= 0x5a;
    let corrupted_request = ExpertProtocolV2Request::decode(&encoded_request)?;
    assert_ne!(corrupted_request.hidden_payload, request.hidden_payload);

    let response = ExpertProtocolV2Response::new(
        42,
        7,
        13,
        1,
        GLM52_HIDDEN_SIZE as u32,
        ExpertV2Dtype::Bf16,
        ExpertProtocolV2Status::Ok,
        vec![7_u8; GLM52_HIDDEN_BF16_BYTES],
    )?;
    let mut encoded_response = response.encode()?;
    assert!(!response.debug_checksum_enabled());
    assert_eq!(
        response.header_len(),
        EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN
    );
    assert_eq!(encoded_response.len(), response.wire_stats().wire_bytes);
    assert_eq!(
        response
            .clone()
            .with_debug_checksum()
            .wire_stats()
            .wire_bytes,
        response.wire_stats().wire_bytes + CHECKSUM_LEN
    );

    let response_view = ExpertProtocolV2ResponseView::parse(&encoded_response)?;
    assert!(!response_view.debug_checksum_enabled());
    assert_eq!(
        response_view.header_len(),
        EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN
    );
    assert!(response_view
        .verify_checksum()
        .unwrap_err()
        .to_string()
        .contains("debug checksum flag is not set"));
    encoded_response[response.header_len()] ^= 0x5a;
    let corrupted_response = ExpertProtocolV2Response::decode(&encoded_response)?;
    assert_ne!(
        corrupted_response.partial_output_payload,
        response.partial_output_payload
    );
    Ok(())
}

#[test]
fn precompile_warmup_request_flag_keeps_hot_header_shape() -> Result<()> {
    let hot_request = request(1, ExpertV2SourceKind::Decode)?;
    let request = hot_request.clone().with_precompile_warmup();
    let encoded = request.encode()?;
    let decoded = ExpertProtocolV2Request::decode(&encoded)?;
    let view = ExpertProtocolV2RequestView::parse(&encoded)?;

    assert!(request.precompile_warmup_enabled());
    assert!(decoded.precompile_warmup_enabled());
    assert!(view.precompile_warmup_enabled());
    assert!(!request.debug_checksum_enabled());
    assert!(!view.debug_checksum_enabled());
    assert_eq!(request.header_len(), EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN);
    assert_eq!(encoded.len(), hot_request.wire_stats().wire_bytes);
    assert_eq!(
        request.wire_stats().wire_bytes,
        hot_request.wire_stats().wire_bytes
    );
    assert_eq!(
        request.header.flags,
        EXPERT_PROTOCOL_V2_FLAG_PRECOMPILE_WARMUP
    );
    assert!(view
        .verify_checksum()
        .unwrap_err()
        .to_string()
        .contains("debug checksum flag is not set"));
    Ok(())
}

#[test]
fn response_rejects_request_only_precompile_warmup_flag() -> Result<()> {
    let mut response = ExpertProtocolV2Response::new(
        42,
        7,
        13,
        1,
        GLM52_HIDDEN_SIZE as u32,
        ExpertV2Dtype::Bf16,
        ExpertProtocolV2Status::Ok,
        vec![7_u8; GLM52_HIDDEN_BF16_BYTES],
    )?;
    response.header.flags |= EXPERT_PROTOCOL_V2_FLAG_PRECOMPILE_WARMUP;

    assert!(response
        .encode()
        .unwrap_err()
        .to_string()
        .contains("unknown flags"));
    Ok(())
}

#[test]
fn request_frame_view_borrows_sections_without_decoding_payload() -> Result<()> {
    let request = request(2, ExpertV2SourceKind::Prefill)?.with_debug_checksum();
    let encoded = request.encode()?;
    let view = ExpertProtocolV2RequestView::parse(&encoded)?;

    assert_eq!(view.header, request.header);
    assert_eq!(
        view.row_descriptor_bytes().len(),
        2 * EXPERT_PROTOCOL_V2_ROW_DESCRIPTOR_LEN
    );
    assert_eq!(
        view.route_bytes().len(),
        2 * EXPERT_PROTOCOL_V2_ROUTE_ENTRY_LEN
    );
    assert_eq!(
        view.hidden_payload().as_ptr(),
        request_payload_ptr(&encoded, &request)
    );
    assert_eq!(view.hidden_payload(), request.hidden_payload.as_slice());
    assert_eq!(view.hidden_row_payload(1)?.as_ptr(), unsafe {
        request_payload_ptr(&encoded, &request).add(GLM52_HIDDEN_BF16_BYTES)
    });
    assert_eq!(
        view.hidden_row_payload(1)?,
        &request.hidden_payload[GLM52_HIDDEN_BF16_BYTES..2 * GLM52_HIDDEN_BF16_BYTES]
    );
    assert!(view
        .hidden_row_payload(2)
        .unwrap_err()
        .to_string()
        .contains("row index"));
    assert_eq!(view.row(1)?.row_id, request.rows[1].row_id);
    assert_eq!(view.route(1)?.expert_id, request.routes[1].expert_id);
    assert_eq!(view.wire_stats(), request.wire_stats());
    assert!(view.debug_checksum_enabled());
    view.verify_checksum()?;
    Ok(())
}

#[test]
fn request_frame_view_checksum_is_explicit_debug_validation() -> Result<()> {
    let request = request(1, ExpertV2SourceKind::Decode)?.with_debug_checksum();
    let mut encoded = request.encode()?;
    let payload_start = request_payload_offset(&request);
    encoded[payload_start] ^= 0x5a;

    let view = ExpertProtocolV2RequestView::parse(&encoded)?;
    assert_ne!(view.hidden_payload(), request.hidden_payload.as_slice());
    assert!(view
        .verify_checksum()
        .unwrap_err()
        .to_string()
        .contains("checksum mismatch"));
    assert!(ExpertProtocolV2Request::decode(&encoded)
        .unwrap_err()
        .to_string()
        .contains("checksum mismatch"));
    Ok(())
}

#[test]
fn response_frame_view_checksum_is_explicit_debug_validation() -> Result<()> {
    let response = ExpertProtocolV2Response::new(
        42,
        7,
        13,
        1,
        GLM52_HIDDEN_SIZE as u32,
        ExpertV2Dtype::Bf16,
        ExpertProtocolV2Status::Ok,
        vec![7_u8; GLM52_HIDDEN_BF16_BYTES],
    )?
    .with_debug_checksum();
    let mut encoded = response.encode()?;
    encoded[response.header_len()] ^= 0x5a;

    let view = ExpertProtocolV2ResponseView::parse(&encoded)?;
    assert_ne!(
        view.partial_output_payload(),
        response.partial_output_payload.as_slice()
    );
    assert!(view.debug_checksum_enabled());
    assert!(view
        .verify_checksum()
        .unwrap_err()
        .to_string()
        .contains("checksum mismatch"));
    assert!(ExpertProtocolV2Response::decode(&encoded)
        .unwrap_err()
        .to_string()
        .contains("checksum mismatch"));
    Ok(())
}

#[test]
fn response_frame_view_borrows_payload_without_decoding_owned_vec() -> Result<()> {
    let row_count = 2;
    let payload = vec![7_u8; row_count * GLM52_HIDDEN_BF16_BYTES];
    let response = ExpertProtocolV2Response::new(
        42,
        7,
        13,
        row_count as u32,
        GLM52_HIDDEN_SIZE as u32,
        ExpertV2Dtype::Bf16,
        ExpertProtocolV2Status::Ok,
        payload,
    )?
    .with_debug_checksum();
    let encoded = response.encode()?;
    let view = ExpertProtocolV2ResponseView::parse(&encoded)?;

    assert_eq!(view.header, response.header);
    assert_eq!(
        view.partial_output_payload().as_ptr(),
        encoded[response.header_len()..].as_ptr()
    );
    assert_eq!(
        view.partial_output_payload(),
        response.partial_output_payload.as_slice()
    );
    assert_eq!(view.partial_output_row_payload(1)?.as_ptr(), unsafe {
        encoded[response.header_len()..]
            .as_ptr()
            .add(GLM52_HIDDEN_BF16_BYTES)
    });
    assert_eq!(
        view.partial_output_row_payload(1)?,
        &response.partial_output_payload[GLM52_HIDDEN_BF16_BYTES..2 * GLM52_HIDDEN_BF16_BYTES]
    );
    assert!(view
        .partial_output_row_payload(2)
        .unwrap_err()
        .to_string()
        .contains("row index"));
    assert_eq!(view.wire_stats(), response.wire_stats());
    assert!(view.debug_checksum_enabled());
    view.verify_checksum()?;
    Ok(())
}

#[test]
fn reusable_frame_buffer_preserves_request_allocation() -> Result<()> {
    let request = request(16, ExpertV2SourceKind::Prefill)?;
    let mut buffer = ExpertProtocolV2FrameBuffer::with_capacity(request.wire_stats().wire_bytes);
    let first_ptr = {
        let frame = buffer.encode_request(&request)?;
        assert_eq!(ExpertProtocolV2Request::decode(frame)?, request);
        frame.as_ptr()
    };
    let first_capacity = buffer.capacity();

    let second_ptr = {
        let frame = buffer.encode_request(&request)?;
        assert_eq!(ExpertProtocolV2Request::decode(frame)?, request);
        frame.as_ptr()
    };

    assert_eq!(second_ptr, first_ptr);
    assert_eq!(buffer.capacity(), first_capacity);
    Ok(())
}

#[test]
fn reusable_frame_buffer_preserves_response_allocation() -> Result<()> {
    let row_count = 4;
    let response = ExpertProtocolV2Response::new(
        42,
        7,
        13,
        row_count,
        GLM52_HIDDEN_SIZE as u32,
        ExpertV2Dtype::Bf16,
        ExpertProtocolV2Status::Ok,
        vec![9_u8; row_count as usize * GLM52_HIDDEN_BF16_BYTES],
    )?;
    let mut buffer = ExpertProtocolV2FrameBuffer::with_capacity(response.wire_stats().wire_bytes);
    let first_ptr = {
        let frame = buffer.encode_response(&response)?;
        assert_eq!(ExpertProtocolV2Response::decode(frame)?, response);
        frame.as_ptr()
    };
    let first_capacity = buffer.capacity();

    let second_ptr = {
        let frame = buffer.encode_response(&response)?;
        assert_eq!(ExpertProtocolV2Response::decode(frame)?, response);
        frame.as_ptr()
    };

    assert_eq!(second_ptr, first_ptr);
    assert_eq!(buffer.capacity(), first_capacity);
    Ok(())
}

#[test]
fn frame_prefix_encoders_match_full_wire_frames() -> Result<()> {
    let request = request(4, ExpertV2SourceKind::Prefill)?.with_debug_checksum();
    let mut full_request = ExpertProtocolV2FrameBuffer::new();
    let full_request = full_request.encode_request(&request)?.to_vec();
    let mut request_prefix = ExpertProtocolV2FrameBuffer::new();
    let mut split_request = request_prefix.encode_request_prefix(&request)?.to_vec();
    split_request.extend_from_slice(&request.hidden_payload);
    assert_eq!(split_request, full_request);
    assert_eq!(ExpertProtocolV2Request::decode(&split_request)?, request);

    let response = ExpertProtocolV2Response::new(
        42,
        7,
        13,
        4,
        GLM52_HIDDEN_SIZE as u32,
        ExpertV2Dtype::Bf16,
        ExpertProtocolV2Status::Ok,
        vec![9_u8; 4 * GLM52_HIDDEN_BF16_BYTES],
    )?
    .with_debug_checksum()
    .with_executor_name("test-executor");
    let mut full_response = ExpertProtocolV2FrameBuffer::new();
    let full_response = full_response.encode_response(&response)?.to_vec();
    let mut response_prefix = ExpertProtocolV2FrameBuffer::new();
    let mut split_response = response_prefix.encode_response_prefix(&response)?.to_vec();
    split_response.extend_from_slice(&response.partial_output_payload);
    assert_eq!(split_response, full_response);
    assert_eq!(ExpertProtocolV2Response::decode(&split_response)?, response);

    let borrowed_response = response.as_borrowed();
    let mut borrowed_prefix = ExpertProtocolV2FrameBuffer::new();
    let mut split_borrowed = borrowed_prefix
        .encode_borrowed_response_prefix(&borrowed_response)?
        .to_vec();
    split_borrowed.extend_from_slice(borrowed_response.partial_output_payload);
    assert_eq!(split_borrowed, full_response);
    assert_eq!(borrowed_response.to_owned()?, response);
    Ok(())
}

#[test]
fn device_response_prefix_matches_host_response_metadata() -> Result<()> {
    let row_indices = [7_u32, 3_u32];
    let payload_bytes = 2 * GLM52_HIDDEN_BF16_BYTES;
    let host = ExpertProtocolV2Response::new(
        42,
        7,
        13,
        2,
        GLM52_HIDDEN_SIZE as u32,
        ExpertV2Dtype::Bf16,
        ExpertProtocolV2Status::Ok,
        vec![0_u8; payload_bytes],
    )?
    .with_row_indices(row_indices.to_vec(), false)?
    .with_executor_name("test-executor");
    let device = ExpertProtocolV2DeviceResponseRef::new_with_output_stride(
        42,
        7,
        13,
        2,
        GLM52_HIDDEN_SIZE as u32,
        ExpertV2Dtype::Bf16,
        GLM52_HIDDEN_BF16_BYTES as u32,
        ExpertProtocolV2Status::Ok,
        glmrt_ffi::GlmrtDeviceBuffer {
            ptr: 0x10_0000_usize as *mut std::ffi::c_void,
            bytes: payload_bytes,
            device_id: 0,
            flags: 0,
        },
    )?
    .with_row_indices(&row_indices, false)?
    .with_executor_name("test-executor");

    let mut host_prefix = ExpertProtocolV2FrameBuffer::new();
    let host_prefix = host_prefix.encode_response_prefix(&host)?.to_vec();
    let mut device_prefix = ExpertProtocolV2FrameBuffer::new();
    let device_prefix = device_prefix
        .encode_device_response_prefix(&device)?
        .to_vec();

    assert_eq!(device_prefix, host_prefix);
    assert_eq!(device.wire_stats(), host.wire_stats());
    Ok(())
}

#[test]
fn frame_arena_preserves_request_response_allocations_and_returns_views() -> Result<()> {
    let request = request(8, ExpertV2SourceKind::MtpVerify)?;
    let response = ExpertProtocolV2Response::new(
        request.header.request_id,
        request.header.placement_version,
        request.header.layer_id,
        request.header.row_count,
        GLM52_HIDDEN_SIZE as u32,
        ExpertV2Dtype::Bf16,
        ExpertProtocolV2Status::Ok,
        vec![13_u8; 8 * GLM52_HIDDEN_BF16_BYTES],
    )?;
    let mut arena = ExpertProtocolV2FrameArena::with_capacities(
        request.wire_stats().wire_bytes,
        response.wire_stats().wire_bytes,
    );
    let request_ptr = arena.request_ptr();
    let request_capacity = arena.request_capacity();
    let response_ptr = arena.response_ptr();
    let response_capacity = arena.response_capacity();

    {
        let view = arena.encode_request_view(&request)?;
        assert_eq!(view.header, request.header);
        assert_eq!(view.hidden_payload(), request.hidden_payload.as_slice());
        assert_eq!(view.hidden_payload().as_ptr(), unsafe {
            request_ptr.add(request_payload_offset(&request))
        });
    }
    assert_eq!(arena.request_frame().len(), request.wire_stats().wire_bytes);
    assert_eq!(arena.request_ptr(), request_ptr);
    assert_eq!(arena.request_capacity(), request_capacity);

    {
        let view = arena.encode_response_view(&response)?;
        assert_eq!(view.header, response.header);
        assert_eq!(
            view.partial_output_payload(),
            response.partial_output_payload.as_slice()
        );
        assert_eq!(view.partial_output_payload().as_ptr(), unsafe {
            response_ptr.add(response.header_len())
        });
    }
    assert_eq!(
        arena.response_frame().len(),
        response.wire_stats().wire_bytes
    );
    assert_eq!(arena.response_ptr(), response_ptr);
    assert_eq!(arena.response_capacity(), response_capacity);

    {
        let view = arena.encode_request_view(&request)?;
        assert_eq!(view.wire_stats(), request.wire_stats());
    }
    {
        let view = arena.encode_response_view(&response)?;
        assert_eq!(view.wire_stats(), response.wire_stats());
    }
    assert_eq!(arena.request_ptr(), request_ptr);
    assert_eq!(arena.request_capacity(), request_capacity);
    assert_eq!(arena.response_ptr(), response_ptr);
    assert_eq!(arena.response_capacity(), response_capacity);
    Ok(())
}

#[test]
fn expert_batch_mixed_rows_build_protocol_v2_request() -> Result<()> {
    let batch = mixed_expert_batch()?;
    let routes = routes_for_batch(&batch);
    let hidden_payload = vec![3_u8; batch.num_rows() * batch.hidden_bytes_per_row];
    let request =
        ExpertProtocolV2Request::from_expert_batch(777, &batch, routes, hidden_payload.clone())?;
    let decoded = ExpertProtocolV2Request::decode(&request.encode()?)?;

    assert_eq!(decoded, request);
    assert_eq!(
        decoded.header.placement_version,
        expert_protocol_v2_compact_id("placement-mixed")
    );
    assert_eq!(decoded.header.layer_id, 3);
    assert_eq!(decoded.header.hidden_dim, GLM52_HIDDEN_SIZE as u32);
    assert_eq!(
        decoded.header.hidden_row_stride_bytes,
        GLM52_HIDDEN_BF16_BYTES as u32
    );
    assert_eq!(decoded.header.hidden_dtype, ExpertV2Dtype::Bf16);
    assert_eq!(decoded.hidden_payload, hidden_payload);
    assert_eq!(
        decoded
            .rows
            .iter()
            .map(|row| row.source_kind)
            .collect::<Vec<_>>(),
        vec![
            ExpertV2SourceKind::Prefill,
            ExpertV2SourceKind::Prefill,
            ExpertV2SourceKind::Prefill,
            ExpertV2SourceKind::Prefill,
            ExpertV2SourceKind::MtpVerify,
            ExpertV2SourceKind::MtpVerify,
            ExpertV2SourceKind::Decode,
        ]
    );
    assert_eq!(decoded.rows[0].route_offset, 0);
    assert_eq!(decoded.rows[4].route_offset, 32);
    assert_eq!(decoded.rows[6].route_offset, 48);
    assert_eq!(decoded.routes.len(), batch.route_count());
    Ok(())
}

#[test]
fn expert_batch_protocol_v2_builder_rejects_misaligned_routes() -> Result<()> {
    let batch = mixed_expert_batch()?;
    let mut routes = routes_for_batch(&batch);
    routes[batch.rows[1].route_offset].row_index = 0;
    let hidden_payload = vec![3_u8; batch.num_rows() * batch.hidden_bytes_per_row];

    let err = ExpertProtocolV2Request::from_expert_batch(777, &batch, routes, hidden_payload)
        .unwrap_err()
        .to_string();

    assert!(err.contains("does not match batch row index"));
    Ok(())
}

#[test]
fn expert_host_batch_builds_compact_protocol_v2_request() -> Result<()> {
    let batch = mixed_expert_batch()?;
    let hosts = expert_hosts();
    let routes = core_routes_for_batch(&batch);
    let host_batch = ExpertHostBatch::from_expert_batch(
        &batch,
        "spark-1",
        &routes,
        &hosts,
        PlacementPolicy::Modulo,
    )?;
    let global_hidden = hidden_payload_for_batch(&batch);
    let compact_hidden = host_batch.compact_hidden_payload(&global_hidden, batch.num_rows())?;
    let request =
        ExpertProtocolV2Request::from_expert_host_batch(778, &host_batch, compact_hidden.clone())?;
    let decoded = ExpertProtocolV2Request::decode(&request.encode()?)?;

    assert_eq!(decoded, request);
    assert_eq!(decoded.header.layer_id, 3);
    assert_eq!(decoded.header.hidden_dim, GLM52_HIDDEN_SIZE as u32);
    assert_eq!(
        decoded.header.hidden_row_stride_bytes,
        GLM52_HIDDEN_BF16_BYTES as u32
    );
    assert_eq!(decoded.rows.len(), host_batch.num_rows());
    assert_eq!(decoded.routes.len(), host_batch.route_count());
    assert_eq!(decoded.hidden_payload, compact_hidden);
    assert_eq!(
        decoded
            .rows
            .iter()
            .map(|row| row.row_id)
            .collect::<Vec<_>>(),
        host_batch
            .rows
            .iter()
            .map(|row| row.row_id)
            .collect::<Vec<_>>()
    );
    assert!(decoded
        .routes
        .iter()
        .all(|route| route.row_index < decoded.rows.len() as u32));
    assert!(host_batch.routes.iter().all(|route| {
        glmrt_core::owner_for_expert(
            host_batch.layer_id.0 as usize,
            route.expert_id,
            &hosts,
            PlacementPolicy::Modulo,
        )
        .as_deref()
            == Some("spark-1")
    }));
    Ok(())
}

#[test]
fn expert_host_batch_protocol_v2_response_scatters_to_global_rows() -> Result<()> {
    let batch = mixed_expert_batch()?;
    let hosts = expert_hosts();
    let routes = core_routes_for_batch(&batch);
    let host_batch = ExpertHostBatch::from_expert_batch(
        &batch,
        "spark-1",
        &routes,
        &hosts,
        PlacementPolicy::Modulo,
    )?;
    let global_hidden = hidden_payload_for_batch(&batch);
    let compact_hidden = host_batch.compact_hidden_payload(&global_hidden, batch.num_rows())?;
    let request =
        ExpertProtocolV2Request::from_expert_host_batch(779, &host_batch, compact_hidden)?;
    let response = crate::protocol_v2_synthetic_response(&request)?;
    let encoded_response = response.encode()?;
    let response_view = ExpertProtocolV2ResponseView::parse(&encoded_response)?;
    assert_eq!(
        response_view.partial_output_payload().as_ptr(),
        encoded_response[response_view.header_len()..].as_ptr()
    );
    let partials = (0..response_view.header.row_count as usize)
        .map(|row_index| {
            response_view
                .partial_output_row_payload(row_index)
                .map(|row| row.to_vec())
        })
        .collect::<Result<Vec<_>>>()?;

    let scattered = host_batch.scatter_partial_outputs(&partials, batch.num_rows())?;
    let host_global_rows = host_batch.global_row_indices().collect::<Vec<_>>();
    assert_eq!(partials.len(), host_batch.num_rows());
    for (host_row_index, global_row_index) in host_global_rows.iter().copied().enumerate() {
        assert_eq!(
            scattered[global_row_index].as_deref(),
            Some(partials[host_row_index].as_slice())
        );
        let start = global_row_index * batch.hidden_bytes_per_row;
        assert_ne!(
            partials[host_row_index].as_slice(),
            &global_hidden[start..start + batch.hidden_bytes_per_row]
        );
    }
    for global_row_index in 0..batch.num_rows() {
        if !host_global_rows.contains(&global_row_index) {
            assert!(scattered[global_row_index].is_none());
        }
    }
    Ok(())
}

#[test]
fn expert_host_batches_partition_all_hosts_and_reconstruct_route_counts() -> Result<()> {
    let batch = mixed_expert_batch()?;
    let hosts = expert_hosts();
    let routes = core_routes_for_batch(&batch);
    let global_hidden = hidden_payload_for_batch(&batch);
    let mut routes_by_global_row = vec![0_usize; batch.num_rows()];
    let mut host_touches_by_global_row = vec![0_usize; batch.num_rows()];
    let mut total_host_routes = 0_usize;

    for (host_index, host) in hosts.iter().enumerate() {
        let host_batch = ExpertHostBatch::from_expert_batch(
            &batch,
            host,
            &routes,
            &hosts,
            PlacementPolicy::Modulo,
        )?;
        let compact_hidden = host_batch.compact_hidden_payload(&global_hidden, batch.num_rows())?;
        assert_eq!(
            compact_hidden.len(),
            host_batch.num_rows() * batch.hidden_bytes_per_row
        );
        assert!(host_batch.routes.iter().all(|route| {
            glmrt_core::owner_for_expert(
                host_batch.layer_id.0 as usize,
                route.expert_id,
                &hosts,
                PlacementPolicy::Modulo,
            )
            .as_deref()
                == Some(host.as_str())
        }));

        total_host_routes += host_batch.route_count();
        for row in &host_batch.rows {
            routes_by_global_row[row.global_row_index] += row.route_count;
            host_touches_by_global_row[row.global_row_index] += 1;
        }

        let request = ExpertProtocolV2Request::from_expert_host_batch(
            790 + host_index as u64,
            &host_batch,
            compact_hidden.clone(),
        )?;
        let request_frame = request.encode()?;
        let request_view = ExpertProtocolV2RequestView::parse(&request_frame)?;
        assert_eq!(request_view.hidden_payload(), compact_hidden.as_slice());
        for (host_row_index, row) in host_batch.rows.iter().enumerate() {
            let global_start = row.global_row_index * batch.hidden_bytes_per_row;
            assert_eq!(
                request_view.hidden_row_payload(host_row_index)?,
                &global_hidden[global_start..global_start + batch.hidden_bytes_per_row]
            );
        }

        let response = crate::protocol_v2_synthetic_response(&request)?;
        let encoded_response = response.encode()?;
        let response_view = ExpertProtocolV2ResponseView::parse(&encoded_response)?;
        let partials = (0..host_batch.num_rows())
            .map(|host_row_index| {
                response_view
                    .partial_output_row_payload(host_row_index)
                    .map(|row| row.to_vec())
            })
            .collect::<Result<Vec<_>>>()?;
        let scattered = host_batch.scatter_partial_outputs(&partials, batch.num_rows())?;
        for (host_row_index, row) in host_batch.rows.iter().enumerate() {
            let start = row.global_row_index * batch.hidden_bytes_per_row;
            assert_eq!(
                scattered[row.global_row_index].as_deref(),
                Some(partials[host_row_index].as_slice())
            );
            assert_ne!(
                partials[host_row_index].as_slice(),
                &global_hidden[start..start + batch.hidden_bytes_per_row]
            );
        }
        for (global_row_index, maybe_output) in scattered.iter().enumerate() {
            if !host_batch
                .rows
                .iter()
                .any(|row| row.global_row_index == global_row_index)
            {
                assert!(maybe_output.is_none());
            }
        }
    }

    assert_eq!(total_host_routes, batch.route_count());
    for (row_index, row) in batch.rows.iter().enumerate() {
        assert_eq!(routes_by_global_row[row_index], row.route_count);
        assert!(host_touches_by_global_row[row_index] > 0);
    }
    Ok(())
}

fn mixed_expert_batch() -> Result<ExpertBatch> {
    let graph_bucket = GraphBucket::new(8);
    let prefill = LayerWave::prefill(PrefillChunk::new(
        "request-a",
        "sequence-a",
        LayerId(3),
        PositionId(16),
        4,
        1,
        Priority(1),
        graph_bucket,
        "placement-mixed",
    ));
    let mtp = LayerWave::mtp_verify(MtpVerifyBlock::new(
        "request-a",
        "sequence-a",
        LayerId(3),
        PositionId(20),
        2,
        Some(1),
        Priority(2),
        graph_bucket,
        "placement-mixed",
    ));
    let decode = LayerWave::decode(DecodeStep::new(
        "request-b",
        "sequence-b",
        LayerId(3),
        PositionId(22),
        Some(1),
        Priority(3),
        "placement-mixed",
    ));
    let mut batch = ExpertBatch::from_wave_with_envelope(
        &prefill,
        DType::Bf16,
        "glm52_nvfp4_lukealonso_v1",
        graph_bucket,
    )?;
    batch.try_append_wave(&mtp, DType::Bf16, "glm52_nvfp4_lukealonso_v1")?;
    batch.try_append_wave(&decode, DType::Bf16, "glm52_nvfp4_lukealonso_v1")?;
    Ok(batch)
}

fn routes_for_batch(batch: &ExpertBatch) -> Vec<ExpertProtocolV2RouteEntry> {
    batch
        .rows
        .iter()
        .enumerate()
        .flat_map(|(row_index, row)| {
            (0..row.route_count).map(move |route| ExpertProtocolV2RouteEntry {
                row_index: row_index as u32,
                expert_id: ((row_index * row.route_count + route) % 256) as u32,
                gate_weight: 1.0 / row.route_count as f32,
            })
        })
        .collect()
}

fn core_routes_for_batch(batch: &ExpertBatch) -> Vec<ExpertBatchRoute> {
    batch
        .rows
        .iter()
        .enumerate()
        .flat_map(|(row_index, row)| {
            (0..row.route_count).map(move |route| ExpertBatchRoute {
                row_index,
                expert_id: (row_index * row.route_count + route) % 256,
                gate_weight: 1.0 / row.route_count as f32,
            })
        })
        .collect()
}

fn hidden_payload_for_batch(batch: &ExpertBatch) -> Vec<u8> {
    let mut payload = vec![0_u8; batch.num_rows() * batch.hidden_bytes_per_row];
    for (row_index, row) in batch.rows.iter().enumerate() {
        let start = row_index * batch.hidden_bytes_per_row;
        payload[start..start + 8].copy_from_slice(&row.row_id.to_le_bytes());
        payload[start + 8..start + 16].copy_from_slice(&row.token_position.0.to_le_bytes());
    }
    payload
}

fn expert_hosts() -> Vec<String> {
    EXPERT_HOSTS.iter().map(|host| (*host).to_owned()).collect()
}

fn request_payload_offset(request: &ExpertProtocolV2Request) -> usize {
    request.header_len()
        + request.rows.len() * EXPERT_PROTOCOL_V2_ROW_DESCRIPTOR_LEN
        + request.routes.len() * EXPERT_PROTOCOL_V2_ROUTE_ENTRY_LEN
}

fn request_payload_ptr<'a>(encoded: &'a [u8], request: &ExpertProtocolV2Request) -> *const u8 {
    encoded[request_payload_offset(request)..].as_ptr()
}
