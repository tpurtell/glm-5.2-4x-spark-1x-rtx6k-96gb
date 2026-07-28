use super::*;

#[test]
fn kv_reservation_capacity_uses_compressed_default() {
    let mut allocator = KvCacheAllocator::new(KvCacheConfig::glm52_phase0(4));
    let reservation_id = allocator.reserve("seq-a", 4).unwrap();
    let snapshot = allocator.snapshot();

    assert_eq!(snapshot.config.layout, KvLayout::Glm52CompressedBf16);
    assert_eq!(
        snapshot.bytes_per_token,
        GLM52_COMPRESSED_KV_BF16_BYTES_PER_TOKEN
    );
    assert_eq!(
        snapshot.capacity_bytes,
        4 * GLM52_COMPRESSED_KV_BF16_BYTES_PER_TOKEN
    );
    assert_eq!(
        allocator.reservation(reservation_id).unwrap().bytes,
        4 * GLM52_COMPRESSED_KV_BF16_BYTES_PER_TOKEN
    );
}

#[test]
fn kv_cache_reservations_pause_resume_and_release() {
    let mut allocator = KvCacheAllocator::new(KvCacheConfig::glm52_phase0(16));
    let id = allocator.reserve("seq-a", 4).unwrap();
    let snapshot = allocator.snapshot();
    assert_eq!(snapshot.resident_tokens, 4);
    assert_eq!(snapshot.active_reservations, 1);

    allocator.pause(id).unwrap();
    assert_eq!(
        allocator.reservation(id).unwrap().state,
        KvReservationState::Paused
    );
    let snapshot = allocator.snapshot();
    assert_eq!(snapshot.paused_reservations, 1);
    assert_eq!(snapshot.resident_tokens, 4);

    allocator.resume(id).unwrap();
    assert_eq!(
        allocator.reservation(id).unwrap().state,
        KvReservationState::Active
    );

    allocator.release(id).unwrap();
    let snapshot = allocator.snapshot();
    assert_eq!(snapshot.resident_tokens, 0);
    assert_eq!(snapshot.resident_bytes, 0);
}

#[test]
fn kv_cache_tracks_prefill_writes_by_layer_and_position() {
    let mut allocator = KvCacheAllocator::new(KvCacheConfig::glm52_phase0(64));
    let reservation_id = allocator.reserve("seq-a", 32).unwrap();
    let first_write = allocator
        .record_prefill_write(KvBlockDescriptor {
            reservation_id,
            sequence_id: "seq-a".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(0),
            token_count: 16,
        })
        .unwrap();
    let second_write = allocator
        .record_prefill_write(KvBlockDescriptor {
            reservation_id,
            sequence_id: "seq-a".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(16),
            token_count: 16,
        })
        .unwrap();

    assert_eq!(
        allocator.write(first_write).unwrap().state,
        KvWriteState::Pending
    );
    assert_eq!(
        allocator.write(second_write).unwrap().token_start,
        PositionId(16)
    );
    assert_eq!(allocator.writes_for_reservation(reservation_id).len(), 2);

    allocator.pause_write(first_write).unwrap();
    assert_eq!(
        allocator.write(first_write).unwrap().state,
        KvWriteState::Paused
    );
    allocator.resume_write(first_write).unwrap();
    assert_eq!(
        allocator.write(first_write).unwrap().state,
        KvWriteState::Pending
    );
    allocator.mark_write_written(first_write).unwrap();
    assert_eq!(
        allocator.write(first_write).unwrap().state,
        KvWriteState::Written
    );

    let snapshot = allocator.snapshot();
    assert_eq!(snapshot.pending_writes, 1);
    assert_eq!(snapshot.written_writes, 1);
    assert_eq!(snapshot.paused_writes, 0);
    assert_eq!(snapshot.tentative_writes, 0);
    assert_eq!(snapshot.committed_writes, 0);
    assert_eq!(snapshot.discarded_writes, 0);
}

#[test]
fn kv_cache_indexes_writes_by_reservation_and_layer() {
    let mut allocator = KvCacheAllocator::new(KvCacheConfig::glm52_phase0(64));
    let reservation_a = allocator.reserve("seq-a", 32).unwrap();
    let reservation_b = allocator.reserve("seq-b", 16).unwrap();
    let write = |reservation_id, sequence_id: &str, layer_id| KvBlockDescriptor {
        reservation_id,
        sequence_id: sequence_id.to_owned(),
        layer_id: LayerId(layer_id),
        token_start: PositionId(0),
        token_count: 8,
    };
    let a_layer_3 = allocator
        .record_prefill_write(write(reservation_a, "seq-a", 3))
        .unwrap();
    let a_layer_4 = allocator
        .record_prefill_write(write(reservation_a, "seq-a", 4))
        .unwrap();
    let b_layer_3 = allocator
        .record_prefill_write(write(reservation_b, "seq-b", 3))
        .unwrap();

    assert_eq!(
        allocator
            .writes_for_reservation_layer(reservation_a, LayerId(3))
            .iter()
            .map(|write| write.id)
            .collect::<Vec<_>>(),
        vec![a_layer_3]
    );
    assert_eq!(
        allocator
            .writes_for_reservation_layer(reservation_a, LayerId(4))
            .iter()
            .map(|write| write.id)
            .collect::<Vec<_>>(),
        vec![a_layer_4]
    );
    assert_eq!(
        allocator
            .writes_for_reservation_layer(reservation_b, LayerId(3))
            .iter()
            .map(|write| write.id)
            .collect::<Vec<_>>(),
        vec![b_layer_3]
    );
    assert!(allocator
        .writes_for_reservation_layer(reservation_b, LayerId(4))
        .is_empty());
}

#[test]
fn mtp_tentative_writes_commit_accepted_prefix_and_discard_suffix() {
    let mut allocator = KvCacheAllocator::new(KvCacheConfig::glm52_phase0(256));
    let reservation_id = allocator.reserve("seq-a", 160).unwrap();
    let wave = LayerWave::mtp_verify(MtpVerifyBlock::new(
        "req-a",
        "seq-a",
        9,
        128,
        4,
        Some(reservation_id),
        Priority(1),
        GraphBucket::new(8),
        "placement-a",
    ));
    let write_ids = wave
        .tentative_kv_writes
        .iter()
        .cloned()
        .map(|descriptor| allocator.record_tentative_write(descriptor))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(write_ids.len(), 4);
    assert!(write_ids
        .iter()
        .all(|id| allocator.write(*id).unwrap().state == KvWriteState::Tentative));

    allocator
        .resolve_mtp_tentative_writes(reservation_id, LayerId(9), PositionId(128), 4, 2)
        .unwrap();

    assert_eq!(
        allocator.write(write_ids[0]).unwrap().state,
        KvWriteState::Committed
    );
    assert_eq!(
        allocator.write(write_ids[1]).unwrap().state,
        KvWriteState::Committed
    );
    assert_eq!(
        allocator.write(write_ids[2]).unwrap().state,
        KvWriteState::Discarded
    );
    assert_eq!(
        allocator.write(write_ids[3]).unwrap().state,
        KvWriteState::Discarded
    );

    let snapshot = allocator.snapshot();
    assert_eq!(snapshot.tentative_writes, 0);
    assert_eq!(snapshot.committed_writes, 2);
    assert_eq!(snapshot.discarded_writes, 2);

    let visible = allocator.visible_writes_for_decode(reservation_id, LayerId(9), PositionId(132));
    assert_eq!(visible.len(), 2);
    assert_eq!(visible[0].token_start, PositionId(128));
    assert_eq!(visible[1].token_start, PositionId(129));
    assert!(visible
        .iter()
        .all(|write| write.token_start.0 < 130 && write.is_visible_to_attention()));
}

#[test]
fn mtp_tentative_resolution_rejects_accepted_count_beyond_draft() {
    let mut allocator = KvCacheAllocator::new(KvCacheConfig::glm52_phase0(16));
    let err = allocator
        .resolve_mtp_tentative_writes(1, LayerId(9), PositionId(4), 2, 3)
        .unwrap_err();

    assert!(matches!(
        err,
        GlmrtError::MtpAcceptedTokensExceedDraft {
            accepted_tokens: 3,
            draft_tokens: 2
        }
    ));
}

#[test]
fn direct_mtp_transaction_matches_range_resolution() {
    let mut baseline = KvCacheAllocator::new(KvCacheConfig::glm52_phase0(256));
    let reservation_id = baseline.reserve("seq-a", 192).unwrap();
    let write_ids = (0..6)
        .map(|offset| {
            baseline
                .record_tentative_write(KvBlockDescriptor {
                    reservation_id,
                    sequence_id: "seq-a".to_owned(),
                    layer_id: LayerId(9),
                    token_start: PositionId(128 + offset),
                    token_count: 1,
                })
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
    let discarded = candidate
        .resolve_mtp_tentative_write_ids(&write_ids, 3)
        .unwrap();

    assert_eq!(discarded, write_ids[3..]);
    for write_id in write_ids {
        assert_eq!(baseline.write(write_id), candidate.write(write_id));
    }
    assert_eq!(baseline.snapshot(), candidate.snapshot());
}

#[test]
fn direct_mtp_transaction_rejects_nonconsecutive_writes_without_mutation() {
    let mut allocator = KvCacheAllocator::new(KvCacheConfig::glm52_phase0(256));
    let reservation_id = allocator.reserve("seq-a", 192).unwrap();
    let write_ids = [128_u64, 130]
        .into_iter()
        .map(|position| {
            allocator
                .record_tentative_write(KvBlockDescriptor {
                    reservation_id,
                    sequence_id: "seq-a".to_owned(),
                    layer_id: LayerId(9),
                    token_start: PositionId(position),
                    token_count: 1,
                })
                .unwrap()
        })
        .collect::<Vec<_>>();

    let error = allocator
        .resolve_mtp_tentative_write_ids(&write_ids, 1)
        .unwrap_err();
    assert!(matches!(error, GlmrtError::InvalidMtpKvTransaction { .. }));
    assert!(write_ids
        .iter()
        .all(|write_id| { allocator.write(*write_id).unwrap().state == KvWriteState::Tentative }));
}

#[test]
fn kv_cache_discards_only_writes_at_or_after_rewind_position() {
    let mut allocator = KvCacheAllocator::new(KvCacheConfig::glm52_phase0(32));
    let reservation_id = allocator.reserve("seq-a", 16).unwrap();
    let write_ids = (4..8)
        .map(|position| {
            let write_id = allocator
                .record_prefill_write(KvBlockDescriptor {
                    reservation_id,
                    sequence_id: "seq-a".to_owned(),
                    layer_id: LayerId(9),
                    token_start: PositionId(position),
                    token_count: 1,
                })
                .unwrap();
            allocator.mark_write_written(write_id).unwrap();
            write_id
        })
        .collect::<Vec<_>>();

    let discarded = allocator.discard_writes_from(reservation_id, LayerId(9), PositionId(6));

    assert_eq!(discarded, write_ids[2..]);
    assert_eq!(
        allocator.write(write_ids[0]).unwrap().state,
        KvWriteState::Written
    );
    assert_eq!(
        allocator.write(write_ids[1]).unwrap().state,
        KvWriteState::Written
    );
    assert!(write_ids[2..]
        .iter()
        .all(|write_id| { allocator.write(*write_id).unwrap().state == KvWriteState::Discarded }));
}

#[test]
fn kv_cache_rejects_out_of_bounds_prefill_write() {
    let mut allocator = KvCacheAllocator::new(KvCacheConfig::glm52_phase0(16));
    let reservation_id = allocator.reserve("seq-a", 8).unwrap();
    let err = allocator
        .record_prefill_write(KvBlockDescriptor {
            reservation_id,
            sequence_id: "seq-a".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(4),
            token_count: 8,
        })
        .unwrap_err();

    assert!(matches!(
        err,
        GlmrtError::KvWriteOutOfBounds {
            token_start: 4,
            token_count: 8,
            reservation_tokens: 8
        }
    ));
}

#[test]
fn kv_cache_capacity_is_enforced() {
    let mut allocator = KvCacheAllocator::new(KvCacheConfig::glm52_phase0(4));
    allocator.reserve("seq-a", 3).unwrap();
    let err = allocator.reserve("seq-b", 2).unwrap_err();
    assert!(matches!(
        err,
        GlmrtError::KvCapacityExceeded {
            requested_tokens: 2,
            available_tokens: 1
        }
    ));
}
