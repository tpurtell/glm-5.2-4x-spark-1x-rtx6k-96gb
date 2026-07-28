use super::*;

#[test]
fn layerwave_decode_wave_has_one_glm_row() {
    let wave = LayerWave::decode(DecodeStep::new(
        "req-a",
        "seq-a",
        7,
        42,
        Some(11),
        Priority(0),
        "placement-a",
    ));

    assert_eq!(wave.mode, LayerWaveMode::Decode);
    assert_eq!(wave.layer_id, LayerId(7));
    assert_eq!(wave.num_rows(), 1);
    assert_eq!(wave.payload_bytes_per_direction(), GLM52_HIDDEN_BF16_BYTES);
    assert_eq!(wave.roundtrip_bytes_per_host(), GLM52_HIDDEN_BF16_BYTES * 2);
    assert_eq!(wave.row_sources[0].kind, RowSourceKind::DecodeStep);
    assert_eq!(wave.kv_reads[0].token_count, 42);
    assert_eq!(wave.kv_writes[0].token_start, PositionId(42));
}

#[test]
fn layerwave_mtp_verify_wave_is_multirow() {
    let wave = LayerWave::mtp_verify(MtpVerifyBlock::new(
        "req-a",
        "seq-a",
        9,
        128,
        4,
        Some(22),
        Priority(1),
        GraphBucket::new(8),
        "placement-a",
    ));

    assert_eq!(wave.mode, LayerWaveMode::MtpVerify);
    assert_eq!(wave.num_rows(), 4);
    assert_eq!(
        wave.payload_bytes_per_direction(),
        4 * GLM52_HIDDEN_BF16_BYTES
    );
    assert_eq!(wave.row_sources[0].kind, RowSourceKind::MtpVerifyBlock);
    assert_eq!(wave.kv_reads[0].token_count, 128);
    assert!(wave.kv_writes.is_empty());
    assert_eq!(wave.tentative_kv_writes.len(), 4);
    assert_eq!(wave.tentative_kv_writes[0].token_start, PositionId(128));
    assert_eq!(wave.tentative_kv_writes[0].token_count, 1);
    assert_eq!(wave.tentative_kv_writes[3].token_start, PositionId(131));
}

#[test]
fn prefill_chunk_zero_has_no_prefix_read_and_writes_chunk_range() {
    let wave = LayerWave::prefill(PrefillChunk::new(
        "req-a",
        "seq-a",
        3,
        0,
        16,
        33,
        Priority(2),
        GraphBucket::new(16),
        "placement-a",
    ));

    assert!(wave.kv_reads.is_empty());
    assert_eq!(wave.kv_writes.len(), 1);
    assert_eq!(wave.kv_writes[0].token_start, PositionId(0));
    assert_eq!(wave.kv_writes[0].token_count, 16);
    assert!(wave.tentative_kv_writes.is_empty());
}

#[test]
fn later_prefill_chunk_reads_prefix_and_writes_chunk_range() {
    let chunk = PrefillChunk::new(
        "req-a",
        "seq-a",
        3,
        64,
        32,
        33,
        Priority(2),
        GraphBucket::new(64),
        "placement-a",
    );
    let wave = LayerWave::prefill(chunk);

    assert_eq!(wave.mode, LayerWaveMode::Prefill);
    assert_eq!(wave.num_rows(), 32);
    assert_eq!(
        wave.payload_bytes_per_direction(),
        32 * GLM52_HIDDEN_BF16_BYTES
    );
    assert_eq!(wave.routed_expert_assignments(), 32 * GLM52_TOP_K);
    assert_eq!(wave.average_rows_per_expert(), 1.0);
    assert_eq!(wave.kv_reads.len(), 1);
    assert_eq!(wave.kv_reads[0].reservation_id, 33);
    assert_eq!(wave.kv_reads[0].token_start, PositionId(0));
    assert_eq!(wave.kv_reads[0].token_count, 64);
    assert_eq!(wave.kv_writes[0].reservation_id, 33);
    assert_eq!(wave.kv_writes[0].token_start, PositionId(64));
    assert_eq!(wave.kv_writes[0].token_count, 32);
    assert!(wave.tentative_kv_writes.is_empty());
}

#[test]
fn prefill_policy_splits_prompt_by_runtime_chunk_size() {
    let policy = PrefillChunkPolicy::latency_smoke(32);
    let chunks = plan_prefill_chunks(
        "req-a",
        "seq-a",
        3,
        70,
        44,
        Priority(0),
        &policy,
        "placement-a",
    );

    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].token_start, PositionId(0));
    assert_eq!(chunks[0].token_count, 32);
    assert_eq!(chunks[1].token_start, PositionId(32));
    assert_eq!(chunks[1].token_count, 32);
    assert_eq!(chunks[2].token_start, PositionId(64));
    assert_eq!(chunks[2].token_count, 6);
    assert_eq!(chunks[0].graph_bucket, GraphBucket::new(32));
}

#[test]
fn prefill_policy_accepts_phase0_chunk_sizes() {
    for chunk_size in [16, 32, 64, 128] {
        let policy = PrefillChunkPolicy::latency_smoke(chunk_size);
        let chunks = plan_prefill_chunks(
            "req-a",
            "seq-a",
            3,
            chunk_size * 2 + 1,
            44,
            Priority(0),
            &policy,
            "placement-a",
        );

        assert_eq!(chunks[0].token_count, chunk_size);
        assert_eq!(chunks[1].token_count, chunk_size);
        assert_eq!(chunks[2].token_count, 1);
        assert_eq!(chunks[0].graph_bucket, GraphBucket::new(chunk_size));
    }
}

#[test]
fn layerwave_admission_prioritizes_decode_over_prefill() {
    let policy = PrefillChunkPolicy {
        chunk_tokens: 16,
        max_prefill_tokens_per_iteration: 16,
        max_active_prefill_chunks: 1,
        decode_priority: true,
    };
    let prefill = LayerWave::prefill(PrefillChunk::new(
        "prefill",
        "seq-a",
        3,
        0,
        16,
        55,
        Priority(0),
        GraphBucket::new(16),
        "placement-a",
    ));
    let decode = LayerWave::decode(DecodeStep::new(
        "decode",
        "seq-b",
        3,
        10,
        Some(66),
        Priority(10),
        "placement-a",
    ));
    let admission = admit_layerwaves_for_iteration(vec![prefill, decode], &policy);

    assert_eq!(admission.selected[0].mode, LayerWaveMode::Decode);
    assert_eq!(admission.selected[1].mode, LayerWaveMode::Prefill);
    assert_eq!(admission.selected_decode_rows, 1);
    assert_eq!(admission.selected_prefill_rows, 16);
    assert!(admission.deferred.is_empty());
}

#[test]
fn layerwave_admission_defers_prefill_beyond_token_and_chunk_budget() {
    let policy = PrefillChunkPolicy {
        chunk_tokens: 16,
        max_prefill_tokens_per_iteration: 32,
        max_active_prefill_chunks: 2,
        decode_priority: true,
    };
    let waves = (0..3)
        .map(|idx| {
            LayerWave::prefill(PrefillChunk::new(
                format!("prefill-{idx}"),
                format!("seq-{idx}"),
                3,
                idx as u64 * 16,
                16,
                55 + idx as u64,
                Priority(idx),
                GraphBucket::new(16),
                "placement-a",
            ))
        })
        .collect::<Vec<_>>();

    let admission = admit_layerwaves_for_iteration(waves, &policy);

    assert_eq!(admission.selected_prefill_chunks, 2);
    assert_eq!(admission.selected_prefill_rows, 32);
    assert_eq!(admission.deferred.len(), 1);
    assert_eq!(admission.deferred[0].mode, LayerWaveMode::Prefill);
}

#[test]
fn layerwaves_mix_only_with_same_layer_and_graph_bucket() {
    let left = LayerWave::prefill(PrefillChunk::new(
        "req-a",
        "seq-a",
        3,
        0,
        16,
        55,
        Priority(3),
        GraphBucket::new(64),
        "placement-a",
    ));
    let right = LayerWave::prefill(PrefillChunk::new(
        "req-b",
        "seq-b",
        3,
        0,
        16,
        66,
        Priority(1),
        GraphBucket::new(64),
        "placement-a",
    ));
    let merged = left.try_merge(&right).unwrap();

    assert_eq!(merged.num_rows(), 32);
    assert_eq!(merged.row_sources.len(), 2);
    assert_eq!(merged.kv_writes.len(), 2);
    assert_eq!(merged.priority, Priority(1));

    let different_layer = LayerWave::prefill(PrefillChunk::new(
        "req-c",
        "seq-c",
        4,
        0,
        16,
        77,
        Priority(1),
        GraphBucket::new(64),
        "placement-a",
    ));
    let err = merged.try_merge(&different_layer).unwrap_err();
    assert!(matches!(err, GlmrtError::LayerWaveMixRejected { .. }));
    assert!(err.to_string().contains("different layers"));
}

#[test]
fn prefill_wave_keeps_reservation_metadata_across_allocator_transitions() {
    let mut allocator = KvCacheAllocator::new(KvCacheConfig::glm52_phase0(128));
    let reservation_id = allocator.reserve("seq-a", 96).unwrap();
    let wave = LayerWave::prefill(PrefillChunk::new(
        "req-a",
        "seq-a",
        3,
        0,
        64,
        reservation_id,
        Priority(0),
        GraphBucket::new(64),
        "placement-a",
    ));

    allocator.pause(reservation_id).unwrap();
    assert_eq!(
        allocator.reservation(reservation_id).unwrap().state,
        KvReservationState::Paused
    );
    allocator.resume(reservation_id).unwrap();
    assert_eq!(
        allocator.reservation(reservation_id).unwrap().state,
        KvReservationState::Active
    );
    assert_eq!(wave.kv_writes[0].reservation_id, reservation_id);
    assert_eq!(wave.kv_writes[0].token_count, 64);
}
