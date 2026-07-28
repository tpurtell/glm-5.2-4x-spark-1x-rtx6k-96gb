use super::*;

#[test]
fn kv_backing_store_writes_and_reads_visible_layer_bytes() {
    let config = KvCacheConfig::glm52_phase0(16);
    let layer_bytes = config.layer_bytes_per_token(LayerId(3));
    let mut store = KvCacheBackingStore::new(config);
    let reservation_id = store.reserve("seq-a", 8).unwrap();
    let payload = vec![7_u8; layer_bytes * 2];
    let write_id = store
        .write_committed_block(
            KvBlockDescriptor {
                reservation_id,
                sequence_id: "seq-a".to_owned(),
                layer_id: LayerId(3),
                token_start: PositionId(0),
                token_count: 2,
            },
            payload.clone(),
        )
        .unwrap();

    assert_eq!(store.backed_write_count(), 1);
    assert_eq!(store.backed_write_bytes(), payload.len());
    assert!(store
        .read_visible_blocks_for_decode(reservation_id, LayerId(3), PositionId(1))
        .is_empty());
    let visible = store.read_visible_blocks_for_decode(reservation_id, LayerId(3), PositionId(2));
    assert_eq!(visible.len(), 1);
    assert_eq!(visible[0].write_id, write_id);
    assert_eq!(visible[0].state, KvWriteState::Written);
    assert_eq!(visible[0].bytes, payload);
}

#[test]
fn kv_backing_store_validates_payload_size() {
    let config = KvCacheConfig::glm52_phase0(16);
    let mut store = KvCacheBackingStore::new(config);
    let reservation_id = store.reserve("seq-a", 8).unwrap();
    let err = store
        .write_committed_block(
            KvBlockDescriptor {
                reservation_id,
                sequence_id: "seq-a".to_owned(),
                layer_id: LayerId(3),
                token_start: PositionId(0),
                token_count: 2,
            },
            vec![0_u8; 17],
        )
        .unwrap_err();

    assert!(matches!(
        err,
        GlmrtError::KvBackingPayloadSizeMismatch {
            expected_bytes: 2304,
            actual_bytes: 17
        }
    ));
}

#[test]
fn kv_backing_store_applies_layerwave_prefill_and_decode_io() {
    let config = KvCacheConfig::glm52_phase0(32);
    let layer_bytes = config.layer_bytes_per_token(LayerId(3));
    let mut store = KvCacheBackingStore::new(config);
    let reservation_id = store.reserve("seq-a", 16).unwrap();
    let first_prefill = LayerWave::prefill(PrefillChunk::new(
        "req-a",
        "seq-a",
        3,
        0,
        4,
        reservation_id,
        Priority(0),
        GraphBucket::new(4),
        "placement-a",
    ));
    assert!(store
        .read_visible_blocks_for_wave(&first_prefill)
        .is_empty());
    store
        .write_committed_blocks_for_wave(&first_prefill, vec![vec![1_u8; layer_bytes * 4]])
        .unwrap();

    let later_prefill = LayerWave::prefill(PrefillChunk::new(
        "req-a",
        "seq-a",
        3,
        4,
        4,
        reservation_id,
        Priority(0),
        GraphBucket::new(4),
        "placement-a",
    ));
    let later_reads = store.read_visible_blocks_for_wave(&later_prefill);
    assert_eq!(later_reads.len(), 1);
    assert_eq!(later_reads[0].descriptor.token_start, PositionId(0));
    assert_eq!(later_reads[0].bytes.len(), layer_bytes * 4);
    store
        .write_committed_blocks_for_wave(&later_prefill, vec![vec![2_u8; layer_bytes * 4]])
        .unwrap();

    let decode = LayerWave::decode(DecodeStep::new(
        "req-a",
        "seq-a",
        3,
        8,
        Some(reservation_id),
        Priority(0),
        "placement-a",
    ));
    let decode_reads = store.read_visible_blocks_for_wave(&decode);
    assert_eq!(decode_reads.len(), 2);
    assert_eq!(decode_reads[0].descriptor.token_start, PositionId(0));
    assert_eq!(decode_reads[1].descriptor.token_start, PositionId(4));
    assert_eq!(
        store.read_attention_blocks_for_wave(&decode),
        decode_reads,
        "payload-backed layers must retain the detailed read path"
    );
    let write_ids = store
        .write_committed_blocks_for_wave(&decode, vec![vec![3_u8; layer_bytes]])
        .unwrap();
    assert_eq!(write_ids.len(), 1);

    let visible_after_decode =
        store.read_visible_blocks_for_decode(reservation_id, LayerId(3), PositionId(9));
    assert_eq!(visible_after_decode.len(), 3);
    assert_eq!(
        visible_after_decode[2].descriptor.token_start,
        PositionId(8)
    );
}

#[test]
fn kv_backing_store_metadata_only_records_remain_visible_without_host_bytes() {
    let config = KvCacheConfig::glm52_phase0(32);
    let mut store = KvCacheBackingStore::new(config);
    let reservation_id = store.reserve("seq-a", 16).unwrap();
    let prefill = LayerWave::prefill(PrefillChunk::new(
        "req-a",
        "seq-a",
        3,
        0,
        4,
        reservation_id,
        Priority(0),
        GraphBucket::new(4),
        "placement-a",
    ));

    let write_ids = store
        .write_committed_block_metadata_for_wave(&prefill)
        .unwrap();

    assert_eq!(write_ids.len(), 1);
    assert_eq!(store.backed_write_count(), 1);
    assert_eq!(store.backed_write_bytes(), 0);

    let decode = LayerWave::decode(DecodeStep::new(
        "req-a",
        "seq-a",
        3,
        4,
        Some(reservation_id),
        Priority(0),
        "placement-a",
    ));
    let visible = store.read_visible_blocks_for_wave(&decode);
    assert_eq!(visible.len(), 1);
    assert_eq!(visible[0].write_id, write_ids[0]);
    assert_eq!(visible[0].descriptor.token_start, PositionId(0));
    assert_eq!(visible[0].descriptor.token_count, 4);
    assert!(visible[0].bytes.is_empty());
}

#[test]
fn kv_backing_store_coalesces_metadata_for_attention_without_changing_the_ledger() {
    let config = KvCacheConfig::glm52_phase0(32);
    let mut store = KvCacheBackingStore::new(config);
    let reservation_id = store.reserve("seq-a", 16).unwrap();
    for token_start in [0, 4] {
        let prefill = LayerWave::prefill(PrefillChunk::new(
            "req-a",
            "seq-a",
            3,
            token_start,
            4,
            reservation_id,
            Priority(0),
            GraphBucket::new(4),
            "placement-a",
        ));
        store
            .write_committed_block_metadata_for_wave(&prefill)
            .unwrap();
    }
    let decode = LayerWave::decode(DecodeStep::new(
        "req-a",
        "seq-a",
        3,
        8,
        Some(reservation_id),
        Priority(0),
        "placement-a",
    ));
    store
        .write_committed_block_metadata_for_wave(&decode)
        .unwrap();
    let mtp = LayerWave::mtp_verify(MtpVerifyBlock::new(
        "req-a",
        "seq-a",
        3,
        9,
        3,
        Some(reservation_id),
        Priority(0),
        GraphBucket::new(4),
        "placement-a",
    ));
    store.write_tentative_block_metadata_for_wave(&mtp).unwrap();
    store
        .resolve_mtp_tentative_writes(reservation_id, LayerId(3), PositionId(9), 3, 2)
        .unwrap();

    let later_decode = LayerWave::decode(DecodeStep::new(
        "req-a",
        "seq-a",
        3,
        11,
        Some(reservation_id),
        Priority(0),
        "placement-a",
    ));
    let ledger = store.read_visible_blocks_for_wave(&later_decode);
    assert_eq!(ledger.len(), 5);
    let attention = store.read_attention_blocks_for_wave(&later_decode);
    assert_eq!(attention.len(), 1);
    assert_eq!(attention[0].descriptor.token_start, PositionId(0));
    assert_eq!(attention[0].descriptor.token_count, 11);

    assert_eq!(
        store.discard_writes_from(reservation_id, LayerId(3), PositionId(10)),
        1
    );
    let rewound = store.read_attention_blocks_for_wave(&later_decode);
    assert_eq!(rewound.len(), 1);
    assert_eq!(rewound[0].descriptor.token_count, 10);
}

#[test]
fn kv_backing_store_rejects_layerwave_payload_count_mismatch() {
    let config = KvCacheConfig::glm52_phase0(32);
    let mut store = KvCacheBackingStore::new(config);
    let reservation_id = store.reserve("seq-a", 16).unwrap();
    let wave = LayerWave::prefill(PrefillChunk::new(
        "req-a",
        "seq-a",
        3,
        0,
        4,
        reservation_id,
        Priority(0),
        GraphBucket::new(4),
        "placement-a",
    ));
    let err = store
        .write_committed_blocks_for_wave(&wave, Vec::new())
        .unwrap_err();

    assert!(matches!(
        err,
        GlmrtError::KvBackingPayloadCountMismatch {
            expected_blocks: 1,
            actual_blocks: 0
        }
    ));
}

#[test]
fn kv_backing_store_applies_layerwave_mtp_tentative_io() {
    let config = KvCacheConfig::glm52_phase0(64);
    let layer_bytes = config.layer_bytes_per_token(LayerId(9));
    let mut store = KvCacheBackingStore::new(config);
    let reservation_id = store.reserve("seq-a", 32).unwrap();
    let mtp = LayerWave::mtp_verify(MtpVerifyBlock::new(
        "req-a",
        "seq-a",
        9,
        16,
        4,
        Some(reservation_id),
        Priority(0),
        GraphBucket::new(4),
        "placement-a",
    ));
    assert!(store.read_visible_blocks_for_wave(&mtp).is_empty());
    let write_ids = store
        .write_tentative_blocks_for_wave(
            &mtp,
            (0..4)
                .map(|offset| vec![offset as u8; layer_bytes])
                .collect(),
        )
        .unwrap();
    assert_eq!(write_ids.len(), 4);
    store
        .resolve_mtp_tentative_writes(reservation_id, LayerId(9), PositionId(16), 4, 2)
        .unwrap();

    let decode = LayerWave::decode(DecodeStep::new(
        "req-a",
        "seq-a",
        9,
        20,
        Some(reservation_id),
        Priority(0),
        "placement-a",
    ));
    let visible = store.read_visible_blocks_for_wave(&decode);
    assert_eq!(visible.len(), 2);
    assert_eq!(visible[0].state, KvWriteState::Committed);
    assert_eq!(visible[0].bytes, vec![0_u8; layer_bytes]);
    assert_eq!(visible[1].descriptor.token_start, PositionId(17));
    assert_eq!(store.backed_write_count(), 2);
}

#[test]
fn kv_backing_store_commits_mtp_prefix_and_discards_suffix_bytes() {
    let config = KvCacheConfig::glm52_phase0(256);
    let layer_bytes = config.layer_bytes_per_token(LayerId(9));
    let mut store = KvCacheBackingStore::new(config);
    let reservation_id = store.reserve("seq-a", 160).unwrap();
    for draft_offset in 0..4_u64 {
        store
            .write_tentative_block(
                KvBlockDescriptor {
                    reservation_id,
                    sequence_id: "seq-a".to_owned(),
                    layer_id: LayerId(9),
                    token_start: PositionId(128 + draft_offset),
                    token_count: 1,
                },
                vec![draft_offset as u8; layer_bytes],
            )
            .unwrap();
    }
    assert_eq!(store.backed_write_count(), 4);
    assert_eq!(store.backed_write_bytes(), layer_bytes * 4);

    store
        .resolve_mtp_tentative_writes(reservation_id, LayerId(9), PositionId(128), 4, 2)
        .unwrap();
    let visible = store.read_visible_blocks_for_decode(reservation_id, LayerId(9), PositionId(132));
    assert_eq!(visible.len(), 2);
    assert_eq!(visible[0].descriptor.token_start, PositionId(128));
    assert_eq!(visible[0].state, KvWriteState::Committed);
    assert_eq!(visible[0].bytes, vec![0_u8; layer_bytes]);
    assert_eq!(visible[1].descriptor.token_start, PositionId(129));
    assert_eq!(visible[1].bytes, vec![1_u8; layer_bytes]);
    assert_eq!(store.backed_write_count(), 2);
    assert_eq!(store.backed_write_bytes(), layer_bytes * 2);
    let snapshot = store.snapshot();
    assert_eq!(snapshot.committed_writes, 2);
    assert_eq!(snapshot.discarded_writes, 2);
}

#[test]
fn direct_mtp_transaction_matches_range_backing_store() {
    let config = KvCacheConfig::glm52_phase0(256);
    let layer_bytes = config.layer_bytes_per_token(LayerId(9));
    let mut baseline = KvCacheBackingStore::new(config);
    let reservation_id = baseline.reserve("seq-a", 160).unwrap();
    let write_ids = (0..6_u64)
        .map(|offset| {
            baseline
                .write_tentative_block(
                    KvBlockDescriptor {
                        reservation_id,
                        sequence_id: "seq-a".to_owned(),
                        layer_id: LayerId(9),
                        token_start: PositionId(128 + offset),
                        token_count: 1,
                    },
                    vec![offset as u8; layer_bytes],
                )
                .unwrap()
        })
        .collect::<Vec<_>>();
    let mut candidate = baseline.clone();

    baseline
        .resolve_mtp_tentative_writes(
            reservation_id,
            LayerId(9),
            PositionId(128),
            write_ids.len(),
            3,
        )
        .unwrap();
    candidate
        .resolve_mtp_tentative_write_ids(&write_ids, 3)
        .unwrap();

    assert_eq!(baseline.snapshot(), candidate.snapshot());
    assert_eq!(
        baseline.backed_write_count(),
        candidate.backed_write_count()
    );
    assert_eq!(
        baseline.backed_write_bytes(),
        candidate.backed_write_bytes()
    );
    assert_eq!(
        baseline.read_visible_blocks_for_decode(reservation_id, LayerId(9), PositionId(134),),
        candidate.read_visible_blocks_for_decode(reservation_id, LayerId(9), PositionId(134),)
    );
}

#[test]
fn kv_backing_store_rewind_removes_discarded_payloads() {
    let config = KvCacheConfig::glm52_phase0(32);
    let layer_bytes = config.layer_bytes_per_token(LayerId(9));
    let mut store = KvCacheBackingStore::new(config);
    let reservation_id = store.reserve("seq-a", 16).unwrap();
    for position in 4..8 {
        store
            .write_committed_block(
                KvBlockDescriptor {
                    reservation_id,
                    sequence_id: "seq-a".to_owned(),
                    layer_id: LayerId(9),
                    token_start: PositionId(position),
                    token_count: 1,
                },
                vec![position as u8; layer_bytes],
            )
            .unwrap();
    }

    assert_eq!(
        store.discard_writes_from(reservation_id, LayerId(9), PositionId(6)),
        2
    );
    let visible = store.read_visible_blocks_for_decode(reservation_id, LayerId(9), PositionId(8));
    assert_eq!(visible.len(), 2);
    assert_eq!(visible[0].descriptor.token_start, PositionId(4));
    assert_eq!(visible[1].descriptor.token_start, PositionId(5));
    assert_eq!(store.backed_write_count(), 2);
    assert_eq!(store.backed_write_bytes(), 2 * layer_bytes);
}
