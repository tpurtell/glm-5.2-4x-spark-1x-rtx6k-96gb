use super::*;

#[test]
fn expert_batch_mixes_decode_and_prefill_rows_with_fixed_envelope() {
    let recipe = ModelFacts::default().quantization_recipe;
    let prefill = LayerWave::prefill(PrefillChunk::new(
        "prefill",
        "seq-a",
        3,
        0,
        15,
        55,
        Priority(2),
        GraphBucket::new(16),
        "placement-a",
    ));
    let decode = LayerWave::decode(DecodeStep::new(
        "decode",
        "seq-a",
        3,
        15,
        Some(55),
        Priority(0),
        "placement-a",
    ));

    assert!(prefill.try_merge(&decode).is_err());

    let mut batch = ExpertBatch::from_wave_with_envelope(
        &prefill,
        DType::Bf16,
        recipe.clone(),
        GraphBucket::new(16),
    )
    .unwrap();
    batch
        .try_append_wave(&decode, DType::Bf16, recipe.clone())
        .unwrap();

    assert_eq!(batch.num_rows(), 16);
    assert_eq!(batch.route_count(), 16 * GLM52_TOP_K);
    assert_eq!(batch.rows[0].source_kind, RowSourceKind::PrefillChunk);
    assert_eq!(batch.rows[0].token_position, PositionId(0));
    assert_eq!(batch.rows[14].source_kind, RowSourceKind::PrefillChunk);
    assert_eq!(batch.rows[14].token_position, PositionId(14));
    assert_eq!(batch.rows[15].source_kind, RowSourceKind::DecodeStep);
    assert_eq!(batch.rows[15].token_position, PositionId(15));
    assert_eq!(batch.rows[15].route_offset, 15 * GLM52_TOP_K);
}

#[test]
fn expert_batch_mixes_mtp_and_decode_rows_with_fixed_envelope() {
    let recipe = ModelFacts::default().quantization_recipe;
    let mtp = LayerWave::mtp_verify(MtpVerifyBlock::new(
        "mtp",
        "seq-a",
        9,
        128,
        4,
        Some(22),
        Priority(1),
        GraphBucket::new(8),
        "placement-a",
    ));
    let decode = LayerWave::decode(DecodeStep::new(
        "decode",
        "seq-a",
        9,
        132,
        Some(22),
        Priority(0),
        "placement-a",
    ));

    let mut batch = ExpertBatch::from_wave_with_envelope(
        &mtp,
        DType::Bf16,
        recipe.clone(),
        GraphBucket::new(8),
    )
    .unwrap();
    batch
        .try_append_wave(&decode, DType::Bf16, recipe.clone())
        .unwrap();

    assert_eq!(batch.num_rows(), 5);
    assert_eq!(batch.rows[0].source_kind, RowSourceKind::MtpVerifyBlock);
    assert_eq!(batch.rows[0].token_position, PositionId(128));
    assert_eq!(batch.rows[3].token_position, PositionId(131));
    assert_eq!(batch.rows[4].source_kind, RowSourceKind::DecodeStep);
    assert_eq!(batch.rows[4].token_position, PositionId(132));
}

#[test]
fn expert_batch_rejects_incompatible_layer_placement_and_dtype() {
    let recipe = ModelFacts::default().quantization_recipe;
    let left_wave = LayerWave::decode(DecodeStep::new(
        "decode-a",
        "seq-a",
        3,
        0,
        None,
        Priority(0),
        "placement-a",
    ));
    let left = ExpertBatch::from_wave_with_envelope(
        &left_wave,
        DType::Bf16,
        recipe.clone(),
        GraphBucket::new(4),
    )
    .unwrap();

    let different_layer = ExpertBatch::from_wave_with_envelope(
        &LayerWave::decode(DecodeStep::new(
            "decode-b",
            "seq-b",
            4,
            0,
            None,
            Priority(0),
            "placement-a",
        )),
        DType::Bf16,
        recipe.clone(),
        GraphBucket::new(4),
    )
    .unwrap();
    assert!(left
        .try_merge(&different_layer)
        .unwrap_err()
        .to_string()
        .contains("different layers"));

    let different_placement = ExpertBatch::from_wave_with_envelope(
        &LayerWave::decode(DecodeStep::new(
            "decode-c",
            "seq-c",
            3,
            0,
            None,
            Priority(0),
            "placement-b",
        )),
        DType::Bf16,
        recipe.clone(),
        GraphBucket::new(4),
    )
    .unwrap();
    assert!(left
        .try_merge(&different_placement)
        .unwrap_err()
        .to_string()
        .contains("different placement versions"));

    let different_dtype =
        ExpertBatch::from_wave_with_envelope(&left_wave, DType::F16, recipe, GraphBucket::new(4))
            .unwrap();
    assert!(left
        .try_merge(&different_dtype)
        .unwrap_err()
        .to_string()
        .contains("different hidden dtypes"));
}

#[test]
fn expert_batch_reconstructs_partials_in_batch_row_order() {
    let recipe = ModelFacts::default().quantization_recipe;
    let mtp = LayerWave::mtp_verify(MtpVerifyBlock::new(
        "mtp",
        "seq-a",
        9,
        128,
        2,
        Some(22),
        Priority(1),
        GraphBucket::new(4),
        "placement-a",
    ));
    let decode = LayerWave::decode(DecodeStep::new(
        "decode",
        "seq-a",
        9,
        130,
        Some(22),
        Priority(0),
        "placement-a",
    ));
    let mut batch = ExpertBatch::from_wave_with_envelope(
        &mtp,
        DType::Bf16,
        recipe.clone(),
        GraphBucket::new(4),
    )
    .unwrap();
    batch.try_append_wave(&decode, DType::Bf16, recipe).unwrap();

    let reconstructed = batch
        .reconstruct_partial_outputs(&["mtp-0", "mtp-1", "decode-0"])
        .unwrap();

    assert_eq!(
        reconstructed[0].0.source_kind,
        RowSourceKind::MtpVerifyBlock
    );
    assert_eq!(reconstructed[0].0.token_position, PositionId(128));
    assert_eq!(reconstructed[0].1, "mtp-0");
    assert_eq!(reconstructed[1].0.token_position, PositionId(129));
    assert_eq!(reconstructed[1].1, "mtp-1");
    assert_eq!(reconstructed[2].0.source_kind, RowSourceKind::DecodeStep);
    assert_eq!(reconstructed[2].0.token_position, PositionId(130));
    assert_eq!(reconstructed[2].1, "decode-0");

    let err = batch
        .reconstruct_partial_outputs(&["missing-one"])
        .unwrap_err();
    assert!(matches!(
        err,
        GlmrtError::ExpertBatchPartialRowCountMismatch {
            expected: 3,
            actual: 1
        }
    ));
}

#[test]
fn expert_host_batch_filters_rows_routes_and_compacts_hidden_payload() {
    let batch = small_prefill_batch(3);
    let hosts = expert_hosts();
    let routes = routes_with_experts(
        &batch,
        &[
            &[0, 4, 8, 12, 16, 20, 24, 28],
            &[1, 5, 9, 13, 0, 4, 8, 12],
            &[2, 6, 10, 14, 18, 22, 26, 30],
        ],
    );

    let host_batch = ExpertHostBatch::from_expert_batch(
        &batch,
        "spark-1",
        &routes,
        &hosts,
        PlacementPolicy::Modulo,
    )
    .unwrap();

    assert_eq!(host_batch.host, "spark-1");
    assert_eq!(host_batch.num_rows(), 1);
    assert_eq!(host_batch.route_count(), 4);
    assert_eq!(host_batch.rows[0].global_row_index, 1);
    assert_eq!(host_batch.rows[0].row_id, batch.rows[1].row_id);
    assert_eq!(host_batch.rows[0].route_offset, 0);
    assert_eq!(host_batch.rows[0].route_count, 4);
    assert_eq!(host_batch.global_row_indices().collect::<Vec<_>>(), vec![1]);
    assert!(host_batch.routes.iter().all(|route| route.row_index == 0));
    assert_eq!(
        host_batch
            .routes
            .iter()
            .map(|route| route.expert_id)
            .collect::<Vec<_>>(),
        vec![1, 5, 9, 13]
    );

    let global_hidden = marked_hidden_payload(&batch);
    let compact = host_batch
        .compact_hidden_payload(&global_hidden, batch.num_rows())
        .unwrap();
    assert_eq!(compact.len(), batch.hidden_bytes_per_row);
    assert_eq!(&compact[..8], &1_u64.to_le_bytes());
    assert_eq!(
        &compact[8..16],
        &batch.rows[1].token_position.0.to_le_bytes()
    );
}

#[test]
fn expert_host_batch_filters_routes_with_explicit_owner_lookup() {
    let batch = small_prefill_batch(2);
    let hosts = vec!["spark-1".to_owned(), "spark-3".to_owned()];
    let routes = routes_with_experts(
        &batch,
        &[&[0, 1, 2, 3, 4, 5, 6, 7], &[8, 9, 10, 11, 12, 13, 14, 15]],
    );
    let owner_lookup = ExpertOwnerLookup::from_pairs((0..16).map(|expert_id| {
        let owner = if expert_id % 2 == 0 {
            "spark-1.cluster.local"
        } else {
            "spark-3.cluster.local"
        };
        ((3, expert_id), owner.to_owned())
    }));

    let host_batch = ExpertHostBatch::from_expert_batch_with_owner_lookup(
        &batch,
        "spark-1",
        &routes,
        &hosts,
        &owner_lookup,
    )
    .unwrap();

    assert_eq!(host_batch.host, "spark-1");
    assert_eq!(host_batch.num_rows(), batch.num_rows());
    assert_eq!(host_batch.route_count(), batch.route_count() / 2);
    assert_eq!(
        host_batch.global_row_indices().collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(
        host_batch
            .routes
            .iter()
            .map(|route| route.expert_id)
            .collect::<Vec<_>>(),
        vec![0, 2, 4, 6, 8, 10, 12, 14]
    );
    assert!(host_batch.routes.iter().all(|route| route.row_index < 2));

    let global_hidden = marked_hidden_payload(&batch);
    let compact = host_batch
        .compact_hidden_payload(&global_hidden, batch.num_rows())
        .unwrap();
    assert_eq!(compact.len(), batch.num_rows() * batch.hidden_bytes_per_row);
    assert_eq!(
        &compact[..batch.hidden_bytes_per_row],
        &global_hidden[..batch.hidden_bytes_per_row]
    );
}

#[test]
fn expert_host_batch_scatters_partials_back_to_global_rows() {
    let batch = small_prefill_batch(3);
    let hosts = expert_hosts();
    let routes = routes_with_experts(
        &batch,
        &[
            &[0, 4, 8, 12, 16, 20, 24, 28],
            &[1, 5, 9, 13, 0, 4, 8, 12],
            &[2, 6, 10, 14, 18, 22, 26, 30],
        ],
    );
    let host_batch = ExpertHostBatch::from_expert_batch(
        &batch,
        "spark-1",
        &routes,
        &hosts,
        PlacementPolicy::Modulo,
    )
    .unwrap();

    let scattered = host_batch
        .scatter_partial_outputs(&["dodo-row-1"], batch.num_rows())
        .unwrap();

    assert_eq!(scattered, vec![None, Some("dodo-row-1"), None]);
    assert!(matches!(
        host_batch.scatter_partial_outputs::<&str>(&[], batch.num_rows()),
        Err(GlmrtError::ExpertHostBatchPartialRowCountMismatch {
            expected: 1,
            actual: 0
        })
    ));
}

#[test]
fn expert_host_batch_set_partitions_with_explicit_owner_lookup() {
    let batch = small_prefill_batch(2);
    let hosts = vec!["spark-1".to_owned(), "spark-3".to_owned()];
    let routes = routes_with_experts(
        &batch,
        &[&[0, 1, 2, 3, 4, 5, 6, 7], &[8, 9, 10, 11, 12, 13, 14, 15]],
    );
    let owner_lookup = ExpertOwnerLookup::from_pairs((0..16).map(|expert_id| {
        let owner = if expert_id % 2 == 0 {
            "spark-1"
        } else {
            "spark-3"
        };
        ((3, expert_id), owner.to_owned())
    }));

    let set = ExpertHostBatchSet::from_expert_batch_with_owner_lookup(
        &batch,
        &routes,
        &hosts,
        &owner_lookup,
    )
    .unwrap();

    assert_eq!(set.num_hosts(), 2);
    assert_eq!(set.route_count(), batch.route_count());
    assert_eq!(set.host_row_count(), batch.num_rows() * 2);
    assert_eq!(
        set.touched_hosts().collect::<Vec<_>>(),
        vec!["spark-1", "spark-3"]
    );
    assert_eq!(
        set.reconstruction_plan.host_row_maps[0].global_row_indices,
        vec![0, 1]
    );
    assert_eq!(
        set.reconstruction_plan.host_row_maps[1].global_row_indices,
        vec![0, 1]
    );

    let missing_owner_lookup = ExpertOwnerLookup::from_pairs((0..15).map(|expert_id| {
        let owner = if expert_id % 2 == 0 {
            "spark-1"
        } else {
            "spark-3"
        };
        ((3, expert_id), owner.to_owned())
    }));
    assert!(matches!(
        ExpertHostBatchSet::from_expert_batch_with_owner_lookup(
            &batch,
            &routes,
            &hosts,
            &missing_owner_lookup,
        ),
        Err(GlmrtError::ExpertHostBatchSetRouteCountMismatch {
            expected: 16,
            actual: 15,
        })
    ));
}

#[test]
fn expert_host_batches_accumulate_all_host_partials_into_global_rows() {
    let batch = small_prefill_batch(3);
    let hosts = expert_hosts();
    let routes = routes_with_experts(
        &batch,
        &[
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[8, 9, 10, 11, 12, 13, 14, 15],
            &[0, 4, 8, 12, 16, 20, 24, 28],
        ],
    );
    let output_dim = 3;
    let mut global_accumulator = vec![0.0_f32; batch.num_rows() * output_dim];
    let mut contribution_counts = vec![0_usize; batch.num_rows()];
    let mut expected_route_sums = vec![0.0_f32; batch.num_rows()];
    let mut expected_host_sums = vec![0.0_f32; batch.num_rows()];

    for (host_index, host) in hosts.iter().enumerate() {
        let host_batch = ExpertHostBatch::from_expert_batch(
            &batch,
            host,
            &routes,
            &hosts,
            PlacementPolicy::Modulo,
        )
        .unwrap();
        let partial_outputs = host_batch
            .rows
            .iter()
            .map(|row| {
                expected_route_sums[row.global_row_index] += row.route_count as f32;
                expected_host_sums[row.global_row_index] += host_index as f32;
                vec![
                    row.route_count as f32,
                    host_index as f32,
                    row.global_row_index as f32 + 0.5,
                ]
            })
            .collect::<Vec<_>>();

        host_batch
            .accumulate_partial_outputs_f32(
                &partial_outputs,
                batch.num_rows(),
                output_dim,
                &mut global_accumulator,
                &mut contribution_counts,
            )
            .unwrap();
    }

    for (row_index, row) in batch.rows.iter().enumerate() {
        let start = row_index * output_dim;
        assert_eq!(global_accumulator[start], expected_route_sums[row_index]);
        assert_eq!(global_accumulator[start], row.route_count as f32);
        assert_eq!(global_accumulator[start + 1], expected_host_sums[row_index]);
        assert_eq!(
            global_accumulator[start + 2],
            contribution_counts[row_index] as f32 * (row_index as f32 + 0.5)
        );
        assert!(contribution_counts[row_index] > 0);
    }
}

#[test]
fn expert_host_batch_accumulator_validates_shapes() {
    let batch = small_prefill_batch(2);
    let hosts = expert_hosts();
    let routes = routes_with_experts(
        &batch,
        &[&[0, 4, 8, 12, 16, 20, 24, 28], &[1, 5, 9, 13, 0, 4, 8, 12]],
    );
    let host_batch = ExpertHostBatch::from_expert_batch(
        &batch,
        "spark-1",
        &routes,
        &hosts,
        PlacementPolicy::Modulo,
    )
    .unwrap();
    let partial_outputs = vec![vec![1.0_f32]];
    let mut accumulator = vec![0.0_f32; batch.num_rows() * 2];
    let mut contribution_counts = vec![0_usize; batch.num_rows()];

    assert!(matches!(
        host_batch.accumulate_partial_outputs_f32(
            &partial_outputs,
            batch.num_rows(),
            2,
            &mut accumulator,
            &mut contribution_counts,
        ),
        Err(GlmrtError::ExpertHostBatchPartialOutputWidthMismatch {
            row_index: 0,
            expected_width: 2,
            actual_width: 1,
        })
    ));

    assert!(matches!(
        host_batch.accumulate_partial_outputs_f32(
            &[vec![1.0_f32, 2.0]],
            batch.num_rows(),
            2,
            &mut accumulator[..3],
            &mut contribution_counts,
        ),
        Err(GlmrtError::ExpertHostBatchPartialAccumulatorSizeMismatch {
            expected_values: 4,
            actual_values: 3,
        })
    ));

    assert!(matches!(
        host_batch.accumulate_partial_outputs_f32(
            &[vec![1.0_f32, 2.0]],
            batch.num_rows(),
            2,
            &mut accumulator,
            &mut contribution_counts[..1],
        ),
        Err(GlmrtError::ExpertHostBatchContributionCountMismatch {
            expected: 2,
            actual: 1,
        })
    ));
}

#[test]
fn expert_host_batch_set_partitions_only_touched_hosts_and_compacts_hidden_payloads() {
    let batch = small_prefill_batch(2);
    let hosts = expert_hosts();
    let routes = routes_with_experts(
        &batch,
        &[&[0, 1, 2, 3, 4, 5, 6, 7], &[8, 9, 10, 11, 12, 13, 14, 15]],
    );

    let set =
        ExpertHostBatchSet::from_expert_batch(&batch, &routes, &hosts, PlacementPolicy::Modulo)
            .unwrap();

    assert_eq!(set.num_hosts(), 4);
    assert_eq!(set.route_count(), batch.route_count());
    assert_eq!(set.host_row_count(), batch.num_rows() * 4);
    assert_eq!(
        set.touched_hosts().collect::<Vec<_>>(),
        hosts.iter().map(String::as_str).collect::<Vec<_>>()
    );
    assert_eq!(set.reconstruction_plan.global_row_count, batch.num_rows());
    assert!(set
        .batches
        .iter()
        .all(|host_batch| host_batch.rows.iter().all(|row| row.route_count > 0)));
    assert!(set.batches.iter().all(|host_batch| {
        host_batch.route_count() < batch.route_count()
            && host_batch.routes.iter().all(|route| {
                owner_for_expert(
                    host_batch.layer_id.0 as usize,
                    route.expert_id,
                    &hosts,
                    PlacementPolicy::Modulo,
                )
                .as_deref()
                    == Some(host_batch.host.as_str())
            })
    }));

    let global_hidden = marked_hidden_payload(&batch);
    let compact_hidden = set.compact_hidden_payloads(&global_hidden).unwrap();
    assert_eq!(compact_hidden.len(), set.num_hosts());
    for (host_batch, compact) in set.batches.iter().zip(compact_hidden.iter()) {
        assert_eq!(
            compact.len(),
            host_batch.num_rows() * batch.hidden_bytes_per_row
        );
        for (host_row_index, row) in host_batch.rows.iter().enumerate() {
            let compact_start = host_row_index * batch.hidden_bytes_per_row;
            let global_start = row.global_row_index * batch.hidden_bytes_per_row;
            assert_eq!(
                &compact[compact_start..compact_start + batch.hidden_bytes_per_row],
                &global_hidden[global_start..global_start + batch.hidden_bytes_per_row]
            );
        }
    }
}

#[test]
fn expert_host_batch_set_replicates_routes_for_intermediate_shards() {
    let batch = small_prefill_batch(2);
    let hosts = expert_hosts();
    let routes = routes_with_experts(
        &batch,
        &[&[0, 1, 2, 3, 4, 5, 6, 7], &[8, 9, 10, 11, 12, 13, 14, 15]],
    );
    let set = ExpertHostBatchSet::replicated_from_expert_batch(&batch, &routes, &hosts).unwrap();

    assert_eq!(set.num_hosts(), hosts.len());
    assert_eq!(set.route_count(), batch.route_count() * hosts.len());
    assert_eq!(set.host_row_count(), batch.num_rows() * hosts.len());
    assert!(set.batches.iter().all(|host_batch| {
        host_batch.num_rows() == batch.num_rows()
            && host_batch.route_count() == batch.route_count()
            && host_batch
                .rows
                .iter()
                .zip(batch.rows.iter())
                .all(|(host_row, row)| host_row.route_count == row.route_count)
    }));

    let shard_partials = set
        .batches
        .iter()
        .enumerate()
        .map(|(shard, host_batch)| {
            host_batch
                .rows
                .iter()
                .map(|_| vec![(shard + 1) as f32; 2])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let accumulation = set
        .accumulate_partial_outputs_f32(&shard_partials, 2)
        .unwrap();
    assert_eq!(accumulation.values, vec![10.0; batch.num_rows() * 2]);
    assert_eq!(accumulation.contribution_counts, vec![4; batch.num_rows()]);
}

#[test]
fn expert_host_batch_set_collapses_single_host_routes() {
    let batch = small_prefill_batch(3);
    let hosts = expert_hosts();
    let routes = routes_with_experts(
        &batch,
        &[
            &[1, 5, 9, 13, 17, 21, 25, 29],
            &[1, 5, 9, 13, 17, 21, 25, 29],
            &[1, 5, 9, 13, 17, 21, 25, 29],
        ],
    );

    let set =
        ExpertHostBatchSet::from_expert_batch(&batch, &routes, &hosts, PlacementPolicy::Modulo)
            .unwrap();

    assert_eq!(set.num_hosts(), 1);
    assert_eq!(set.touched_hosts().collect::<Vec<_>>(), vec!["spark-1"]);
    assert_eq!(set.batches[0].num_rows(), batch.num_rows());
    assert_eq!(set.batches[0].route_count(), batch.route_count());
    assert!(set.batches[0].rows.iter().all(|row| row.route_count > 0));
    assert_eq!(
        set.reconstruction_plan.host_row_maps[0].global_row_indices,
        vec![0, 1, 2]
    );
}

#[test]
fn expert_host_batch_set_accumulates_partials_and_validates_host_count() {
    let batch = small_prefill_batch(2);
    let hosts = expert_hosts();
    let routes = routes_with_experts(
        &batch,
        &[&[0, 1, 2, 3, 4, 5, 6, 7], &[8, 9, 10, 11, 12, 13, 14, 15]],
    );
    let set =
        ExpertHostBatchSet::from_expert_batch(&batch, &routes, &hosts, PlacementPolicy::Modulo)
            .unwrap();
    let output_dim = 2;
    let mut expected_route_sums = vec![0.0_f32; batch.num_rows()];
    let mut expected_host_sums = vec![0.0_f32; batch.num_rows()];
    let partials_by_host = set
        .batches
        .iter()
        .enumerate()
        .map(|(host_index, host_batch)| {
            host_batch
                .rows
                .iter()
                .map(|row| {
                    expected_route_sums[row.global_row_index] += row.route_count as f32;
                    expected_host_sums[row.global_row_index] += (host_index + 1) as f32;
                    vec![row.route_count as f32, (host_index + 1) as f32]
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let accumulation = set
        .accumulate_partial_outputs_f32(&partials_by_host, output_dim)
        .unwrap();

    for (row_index, row) in batch.rows.iter().enumerate() {
        let start = row_index * output_dim;
        assert_eq!(accumulation.values[start], row.route_count as f32);
        assert_eq!(accumulation.values[start], expected_route_sums[row_index]);
        assert_eq!(
            accumulation.values[start + 1],
            expected_host_sums[row_index]
        );
        assert_eq!(accumulation.contribution_counts[row_index], set.num_hosts());
    }
    assert!(matches!(
        set.accumulate_partial_outputs_f32(&partials_by_host[..set.num_hosts() - 1], output_dim),
        Err(GlmrtError::ExpertHostBatchSetPartialHostCountMismatch {
            expected: 4,
            actual: 3,
        })
    ));
}

#[test]
fn expert_host_batch_set_validates_reconstruction_plan_before_accumulating() {
    let batch = small_prefill_batch(2);
    let hosts = expert_hosts();
    let routes = routes_with_experts(
        &batch,
        &[&[0, 1, 2, 3, 4, 5, 6, 7], &[8, 9, 10, 11, 12, 13, 14, 15]],
    );
    let set =
        ExpertHostBatchSet::from_expert_batch(&batch, &routes, &hosts, PlacementPolicy::Modulo)
            .unwrap();
    let partials_by_host = set
        .batches
        .iter()
        .map(|host_batch| {
            host_batch
                .rows
                .iter()
                .map(|row| vec![row.route_count as f32])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut bad = set.clone();
    bad.reconstruction_plan.global_row_count += 1;
    assert!(matches!(
        bad.accumulate_partial_outputs_f32(&partials_by_host, 1),
        Err(
            GlmrtError::ExpertHostBatchSetReconstructionPlanGlobalRowCountMismatch {
                expected: 2,
                actual: 3,
            }
        )
    ));

    let mut bad = set.clone();
    bad.reconstruction_plan.host_row_maps.pop();
    assert!(matches!(
        bad.accumulate_partial_outputs_f32(&partials_by_host, 1),
        Err(
            GlmrtError::ExpertHostBatchSetReconstructionPlanHostCountMismatch {
                expected: 4,
                actual: 3,
            }
        )
    ));

    let mut bad = set.clone();
    let duplicate_host = bad.reconstruction_plan.host_row_maps[0].host.clone();
    bad.reconstruction_plan.host_row_maps[1].host = duplicate_host.clone();
    let err = bad
        .accumulate_partial_outputs_f32(&partials_by_host, 1)
        .unwrap_err();
    assert!(matches!(
        err,
        GlmrtError::ExpertHostBatchSetDuplicateHost { host } if host == duplicate_host
    ));

    let mut bad = set.clone();
    bad.reconstruction_plan.host_row_maps[0]
        .global_row_indices
        .pop();
    assert!(matches!(
        bad.accumulate_partial_outputs_f32(&partials_by_host, 1),
        Err(
            GlmrtError::ExpertHostBatchSetReconstructionPlanRowCountMismatch {
                expected: 2,
                actual: 1,
                ..
            }
        )
    ));

    let mut bad = set.clone();
    bad.reconstruction_plan.host_row_maps[0].global_row_indices[0] = batch.num_rows();
    assert!(matches!(
        bad.accumulate_partial_outputs_f32(&partials_by_host, 1),
        Err(GlmrtError::ExpertHostBatchGlobalRowOutOfBounds {
            row_index: 2,
            row_count: 2,
        })
    ));

    let mut bad = set.clone();
    let expected = bad.batches[0].rows[0].global_row_index;
    bad.reconstruction_plan.host_row_maps[0].global_row_indices[0] =
        (expected + 1) % batch.num_rows();
    assert!(matches!(
        bad.accumulate_partial_outputs_f32(&partials_by_host, 1),
        Err(
            GlmrtError::ExpertHostBatchSetReconstructionPlanGlobalRowMismatch {
                expected: 0,
                actual: 1,
                ..
            }
        )
    ));

    let missing_row_plan = PartialReconstructionPlan {
        global_row_count: 2,
        host_row_maps: vec![HostRowToGlobalRowMap {
            host: "spark-1".to_owned(),
            global_row_indices: vec![1],
        }],
    };
    let partials_by_host = vec![vec![vec![1.0_f32]]];
    assert!(matches!(
        missing_row_plan.accumulate_partial_outputs_f32(&partials_by_host, 1),
        Err(GlmrtError::ExpertHostBatchSetReconstructionPlanMissingGlobalRow { row_index: 0 })
    ));
}

#[test]
fn expert_host_batch_rejects_bad_routes_and_unknown_hosts() {
    let batch = small_prefill_batch(2);
    let hosts = expert_hosts();
    let routes = routes_with_experts(
        &batch,
        &[&[0, 1, 2, 3, 4, 5, 6, 7], &[8, 9, 10, 11, 12, 13, 14, 15]],
    );

    assert!(matches!(
        ExpertHostBatch::from_expert_batch(
            &batch,
            "missing",
            &routes,
            &hosts,
            PlacementPolicy::Modulo
        ),
        Err(GlmrtError::ExpertHostBatchUnknownHost { .. })
    ));

    assert!(matches!(
        ExpertHostBatch::from_expert_batch(
            &batch,
            "spark-1",
            &routes[..routes.len() - 1],
            &hosts,
            PlacementPolicy::Modulo
        ),
        Err(GlmrtError::ExpertHostBatchRouteCountMismatch {
            expected: 16,
            actual: 15
        })
    ));

    let mut bad_routes = routes.clone();
    bad_routes[batch.rows[1].route_offset].row_index = 0;
    assert!(matches!(
        ExpertHostBatch::from_expert_batch(
            &batch,
            "spark-1",
            &bad_routes,
            &hosts,
            PlacementPolicy::Modulo
        ),
        Err(GlmrtError::ExpertHostBatchRouteRowMismatch {
            expected: 1,
            actual: 0
        })
    ));
}

fn small_prefill_batch(rows: usize) -> ExpertBatch {
    ExpertBatch::from_wave_with_envelope(
        &LayerWave::prefill(PrefillChunk::new(
            "prefill",
            "seq-a",
            3,
            0,
            rows,
            55,
            Priority(2),
            GraphBucket::new(8),
            "placement-a",
        )),
        DType::Bf16,
        ModelFacts::default().quantization_recipe,
        GraphBucket::new(8),
    )
    .unwrap()
}

fn routes_with_experts(batch: &ExpertBatch, experts_by_row: &[&[usize]]) -> Vec<ExpertBatchRoute> {
    assert_eq!(experts_by_row.len(), batch.num_rows());
    batch
        .rows
        .iter()
        .enumerate()
        .flat_map(|(row_index, row)| {
            assert_eq!(experts_by_row[row_index].len(), row.route_count);
            experts_by_row[row_index]
                .iter()
                .map(move |expert_id| ExpertBatchRoute {
                    row_index,
                    expert_id: *expert_id,
                    gate_weight: 1.0 / row.route_count as f32,
                })
        })
        .collect()
}

fn marked_hidden_payload(batch: &ExpertBatch) -> Vec<u8> {
    let mut payload = vec![0_u8; batch.num_rows() * batch.hidden_bytes_per_row];
    for (row_index, row) in batch.rows.iter().enumerate() {
        let start = row_index * batch.hidden_bytes_per_row;
        payload[start..start + 8].copy_from_slice(&(row_index as u64).to_le_bytes());
        payload[start + 8..start + 16].copy_from_slice(&row.token_position.0.to_le_bytes());
    }
    payload
}

fn expert_hosts() -> Vec<String> {
    EXPERT_HOSTS.iter().map(|host| (*host).to_owned()).collect()
}
