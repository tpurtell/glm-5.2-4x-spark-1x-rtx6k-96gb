use super::*;

#[test]
fn glm52_decode_graph_contract_matches_protocol_v2_row_shape() {
    let contract = ExpertGraphBufferContract::glm52_bf16(
        LayerId(3),
        LayerWaveMode::Decode,
        GraphBucket::decode(),
        ModelFacts::default().quantization_recipe,
    )
    .unwrap();

    assert_eq!(
        contract.key.execution_envelope,
        ExpertGraphExecutionEnvelope::SparseMoeMixedRows
    );
    assert_eq!(
        contract.key.transport_layout,
        EXPERT_GRAPH_PROTOCOL_V2_LAYOUT
    );
    assert_eq!(contract.hidden_rows.max_rows, 1);
    assert_eq!(contract.hidden_rows.hidden_dim, GLM52_HIDDEN_SIZE);
    assert_eq!(
        contract.hidden_rows.row_stride_bytes,
        GLM52_HIDDEN_BF16_BYTES
    );
    assert_eq!(contract.request_payload_bytes(), GLM52_HIDDEN_BF16_BYTES);
    assert_eq!(contract.response_payload_bytes(), GLM52_HIDDEN_BF16_BYTES);
    assert_eq!(
        contract.route_metadata.max_local_routes_per_row,
        GLM52_TOP_K
    );
    assert_eq!(
        contract.route_metadata.host_row_global_index_bytes,
        EXPERT_GRAPH_HOST_ROW_GLOBAL_INDEX_BYTES
    );
    assert_eq!(contract.route_metadata.route_capacity(), GLM52_TOP_K);
    assert_eq!(
        contract.route_metadata.bytes(),
        GLM52_TOP_K * EXPERT_GRAPH_ROUTE_ENTRY_BYTES
            + EXPERT_GRAPH_ROW_ROUTE_COUNT_BYTES
            + EXPERT_GRAPH_HOST_ROW_GLOBAL_INDEX_BYTES
            + EXPERT_GRAPH_ACTIVE_COUNTS_BYTES
    );
    assert_eq!(contract.workspace.max_expert_tiles, GLM52_TOP_K);
    contract
        .validate_active_counts(ExpertGraphActiveCounts {
            rows: 1,
            routes: GLM52_TOP_K,
            expert_tiles: GLM52_TOP_K,
        })
        .unwrap();
}

#[test]
fn graph_keys_do_not_split_compatible_rows_by_source_mode() {
    let recipe = ModelFacts::default().quantization_recipe;
    let decode = ExpertGraphBufferContract::glm52_bf16(
        LayerId(3),
        LayerWaveMode::Decode,
        GraphBucket::new(16),
        recipe.clone(),
    )
    .unwrap();
    let prefill = ExpertGraphBufferContract::glm52_bf16(
        LayerId(3),
        LayerWaveMode::Prefill,
        GraphBucket::new(16),
        recipe.clone(),
    )
    .unwrap();
    let mtp = ExpertGraphBufferContract::glm52_bf16(
        LayerId(3),
        LayerWaveMode::MtpVerify,
        GraphBucket::new(16),
        recipe,
    )
    .unwrap();

    assert_eq!(decode.key, prefill.key);
    assert_eq!(decode.key, mtp.key);
    assert_eq!(
        decode.key.execution_envelope,
        ExpertGraphExecutionEnvelope::SparseMoeMixedRows
    );
}

#[test]
fn glm52_prefill_graph_contract_keeps_fixed_bucket_for_smaller_active_counts() {
    let contract = ExpertGraphBufferContract::glm52_bf16(
        LayerId(9),
        LayerWaveMode::Prefill,
        GraphBucket::new(512),
        ModelFacts::default().quantization_recipe,
    )
    .unwrap();

    assert_eq!(contract.hidden_rows.max_rows, 512);
    assert_eq!(
        contract.request_payload_bytes(),
        512 * GLM52_HIDDEN_BF16_BYTES
    );
    assert_eq!(
        contract.response_payload_bytes(),
        512 * GLM52_HIDDEN_BF16_BYTES
    );
    assert_eq!(contract.route_metadata.route_capacity(), 512 * GLM52_TOP_K);
    assert_eq!(
        contract.route_metadata.bytes(),
        512 * GLM52_TOP_K * EXPERT_GRAPH_ROUTE_ENTRY_BYTES
            + 512 * EXPERT_GRAPH_ROW_ROUTE_COUNT_BYTES
            + 512 * EXPERT_GRAPH_HOST_ROW_GLOBAL_INDEX_BYTES
            + EXPERT_GRAPH_ACTIVE_COUNTS_BYTES
    );
    assert_eq!(contract.workspace.max_expert_tiles, GLM52_ROUTED_EXPERTS);
    assert_eq!(
        contract.workspace.partial_accumulator_bytes,
        512 * GLM52_HIDDEN_BF16_BYTES
    );
    assert!(contract.fixed_buffer_bytes() > contract.request_payload_bytes() * 2);
    contract
        .validate_active_counts(ExpertGraphActiveCounts {
            rows: 128,
            routes: 128 * GLM52_TOP_K,
            expert_tiles: 64,
        })
        .unwrap();
}

#[test]
fn graph_contract_derives_active_counts_from_expert_host_batch() {
    let batch = graph_prefill_batch(3);
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
    let contract = ExpertGraphBufferContract::glm52_bf16(
        LayerId(3),
        LayerWaveMode::Prefill,
        GraphBucket::new(8),
        ModelFacts::default().quantization_recipe,
    )
    .unwrap();

    let counts = contract.active_counts_for_host_batch(&host_batch).unwrap();

    assert_eq!(counts.rows, host_batch.num_rows());
    assert_eq!(counts.routes, host_batch.route_count());
    assert_eq!(counts.expert_tiles, 4);
}

#[test]
fn graph_contract_accepts_mixed_source_host_batch_under_one_envelope() {
    let batch = graph_mixed_source_batch();
    let hosts = expert_hosts();
    let routes = routes_with_experts(
        &batch,
        &[
            &[1, 5, 9, 13, 17, 21, 25, 29],
            &[1, 33, 65, 97, 129, 161, 193, 225],
            &[5, 37, 69, 101, 133, 165, 197, 229],
            &[9, 41, 73, 105, 137, 169, 201, 233],
            &[13, 45, 77, 109, 141, 173, 205, 237],
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
    let contract = ExpertGraphBufferContract::glm52_bf16(
        LayerId(3),
        LayerWaveMode::Decode,
        GraphBucket::new(8),
        ModelFacts::default().quantization_recipe,
    )
    .unwrap();

    let counts = contract.active_counts_for_host_batch(&host_batch).unwrap();

    assert_eq!(
        contract.key.execution_envelope,
        ExpertGraphExecutionEnvelope::SparseMoeMixedRows
    );
    assert!(host_batch
        .rows
        .iter()
        .any(|row| row.source_kind == RowSourceKind::PrefillChunk));
    assert!(host_batch
        .rows
        .iter()
        .any(|row| row.source_kind == RowSourceKind::MtpVerifyBlock));
    assert!(host_batch
        .rows
        .iter()
        .any(|row| row.source_kind == RowSourceKind::DecodeStep));
    assert_eq!(counts.rows, host_batch.num_rows());
    assert_eq!(counts.routes, host_batch.route_count());
    assert!(counts.expert_tiles <= contract.workspace.max_expert_tiles);
}

#[test]
fn graph_instance_pool_acquires_and_releases_mixed_source_bucket() {
    let batch = graph_mixed_source_batch();
    let hosts = expert_hosts();
    let routes = routes_with_experts(
        &batch,
        &[
            &[1, 5, 9, 13, 17, 21, 25, 29],
            &[1, 33, 65, 97, 129, 161, 193, 225],
            &[5, 37, 69, 101, 133, 165, 197, 229],
            &[9, 41, 73, 105, 137, 169, 201, 233],
            &[13, 45, 77, 109, 141, 173, 205, 237],
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
    let mut pool = ExpertGraphInstancePool::new();
    let key = pool
        .register_glm52_bf16(
            LayerId(3),
            LayerWaveMode::Decode,
            GraphBucket::new(8),
            ModelFacts::default().quantization_recipe,
            2,
        )
        .unwrap();

    let lease = pool.acquire_for_host_batch(&host_batch).unwrap();

    assert_eq!(lease.key, key);
    assert_eq!(
        lease.key.execution_envelope,
        ExpertGraphExecutionEnvelope::SparseMoeMixedRows
    );
    assert_eq!(lease.active_counts.rows, host_batch.num_rows());
    assert_eq!(lease.active_counts.routes, host_batch.route_count());
    assert!(lease.fixed_buffer_bytes > host_batch.num_rows() * GLM52_HIDDEN_BF16_BYTES);
    assert_eq!(
        pool.stats(),
        ExpertGraphPoolStats {
            graph_keys: 1,
            total_instances: 2,
            available_instances: 1,
            in_use_instances: 1,
            active_leases: 1,
            acquisitions: 1,
            reuses: 0,
        }
    );

    pool.release(lease).unwrap();

    assert_eq!(pool.stats().available_instances, 2);
    assert_eq!(pool.stats().active_leases, 0);
}

#[test]
fn graph_instance_pool_selects_smallest_compatible_bucket() {
    let batch = graph_prefill_batch(3);
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
    let recipe = ModelFacts::default().quantization_recipe;
    let mut pool = ExpertGraphInstancePool::new();
    pool.register_glm52_bf16(
        LayerId(3),
        LayerWaveMode::Prefill,
        GraphBucket::new(512),
        recipe.clone(),
        1,
    )
    .unwrap();
    pool.register_glm52_bf16(
        LayerId(3),
        LayerWaveMode::Decode,
        GraphBucket::new(8),
        recipe,
        1,
    )
    .unwrap();

    let lease = pool.acquire_for_host_batch(&host_batch).unwrap();

    assert_eq!(lease.key.row_bucket, GraphBucket::new(8));
    assert_eq!(lease.active_counts.rows, host_batch.num_rows());
    assert_eq!(pool.stats().in_use_instances, 1);
}

#[test]
fn graph_instance_pool_reuses_released_instance() {
    let batch = graph_prefill_batch(3);
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
    let mut pool = ExpertGraphInstancePool::new();
    pool.register_glm52_bf16(
        LayerId(3),
        LayerWaveMode::Prefill,
        GraphBucket::new(8),
        ModelFacts::default().quantization_recipe,
        1,
    )
    .unwrap();
    let first = pool.acquire_for_host_batch(&host_batch).unwrap();
    let instance_index = first.instance_index;
    pool.release(first).unwrap();

    let second = pool.acquire_for_host_batch(&host_batch).unwrap();

    assert_eq!(second.instance_index, instance_index);
    assert_eq!(pool.stats().acquisitions, 2);
    assert_eq!(pool.stats().reuses, 1);
}

#[test]
fn graph_instance_pool_reports_exhaustion_and_unknown_release() {
    let batch = graph_prefill_batch(3);
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
    let mut pool = ExpertGraphInstancePool::new();
    pool.register_glm52_bf16(
        LayerId(3),
        LayerWaveMode::Prefill,
        GraphBucket::new(8),
        ModelFacts::default().quantization_recipe,
        1,
    )
    .unwrap();
    let lease = pool.acquire_for_host_batch(&host_batch).unwrap();

    let exhausted = pool.acquire_for_host_batch(&host_batch).unwrap_err();
    assert!(exhausted.to_string().contains("graph pool exhausted"));
    pool.release(lease.clone()).unwrap();
    let unknown_release = pool.release(lease).unwrap_err();
    assert!(unknown_release
        .to_string()
        .contains("unknown graph pool lease"));
}

#[test]
fn graph_instance_pool_acquires_host_batch_set_atomically() {
    let batch = graph_prefill_batch(3);
    let hosts = expert_hosts();
    let routes = routes_with_experts(
        &batch,
        &[
            &[0, 4, 8, 12, 16, 20, 24, 28],
            &[1, 5, 9, 13, 17, 21, 25, 29],
            &[2, 6, 10, 14, 3, 7, 11, 15],
        ],
    );
    let set =
        ExpertHostBatchSet::from_expert_batch(&batch, &routes, &hosts, PlacementPolicy::Modulo)
            .unwrap();
    assert_eq!(set.num_hosts(), 4);
    let mut pool = ExpertGraphInstancePool::new();
    pool.register_glm52_bf16(
        LayerId(3),
        LayerWaveMode::Prefill,
        GraphBucket::new(8),
        ModelFacts::default().quantization_recipe,
        4,
    )
    .unwrap();

    let lease = pool.acquire_for_host_batch_set(&set).unwrap();

    assert_eq!(lease.num_hosts(), set.num_hosts());
    assert_eq!(lease.active_counts.rows, set.host_row_count());
    assert_eq!(lease.active_counts.routes, set.route_count());
    assert!(lease.active_counts.expert_tiles >= set.num_hosts());
    assert_eq!(lease.bucket_rows(), vec![8, 8, 8, 8]);
    assert_eq!(pool.stats().active_leases, 4);
    assert_eq!(pool.stats().available_instances, 0);

    pool.release_host_batch_set(lease).unwrap();

    assert_eq!(pool.stats().active_leases, 0);
    assert_eq!(pool.stats().available_instances, 4);
}

#[test]
fn graph_instance_pool_rolls_back_host_batch_set_on_exhaustion() {
    let batch = graph_prefill_batch(3);
    let hosts = expert_hosts();
    let routes = routes_with_experts(
        &batch,
        &[
            &[0, 4, 8, 12, 16, 20, 24, 28],
            &[1, 5, 9, 13, 17, 21, 25, 29],
            &[2, 6, 10, 14, 3, 7, 11, 15],
        ],
    );
    let set =
        ExpertHostBatchSet::from_expert_batch(&batch, &routes, &hosts, PlacementPolicy::Modulo)
            .unwrap();
    let mut pool = ExpertGraphInstancePool::new();
    pool.register_glm52_bf16(
        LayerId(3),
        LayerWaveMode::Prefill,
        GraphBucket::new(8),
        ModelFacts::default().quantization_recipe,
        set.num_hosts() - 1,
    )
    .unwrap();

    let err = pool.acquire_for_host_batch_set(&set).unwrap_err();

    assert!(err.to_string().contains("graph pool exhausted"));
    assert_eq!(pool.stats().active_leases, 0);
    assert_eq!(pool.stats().available_instances, set.num_hosts() - 1);
}

#[test]
fn graph_contract_rejects_incompatible_expert_host_batch() {
    let batch = graph_prefill_batch(3);
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
    let contract = ExpertGraphBufferContract::glm52_bf16(
        LayerId(4),
        LayerWaveMode::Prefill,
        GraphBucket::new(8),
        ModelFacts::default().quantization_recipe,
    )
    .unwrap();

    let err = contract
        .active_counts_for_host_batch(&host_batch)
        .unwrap_err();

    assert!(matches!(
        &err,
        GlmrtError::GraphBufferContractInvalid { .. }
    ));
    assert!(err.to_string().contains("layer"));
}

#[test]
fn glm52_mtp_graph_contract_rejects_active_counts_outside_bucket() {
    let contract = ExpertGraphBufferContract::glm52_bf16(
        LayerId(11),
        LayerWaveMode::MtpVerify,
        GraphBucket::new(8),
        ModelFacts::default().quantization_recipe,
    )
    .unwrap();

    assert!(matches!(
        contract.validate_active_counts(ExpertGraphActiveCounts {
            rows: 9,
            routes: 9,
            expert_tiles: 1,
        }),
        Err(GlmrtError::GraphBufferActiveCountOutOfBounds { field: "rows", .. })
    ));
    assert!(matches!(
        contract.validate_active_counts(ExpertGraphActiveCounts {
            rows: 2,
            routes: 2 * GLM52_TOP_K + 1,
            expert_tiles: 1,
        }),
        Err(GlmrtError::GraphBufferActiveCountOutOfBounds {
            field: "routes_per_active_rows",
            ..
        })
    ));
}

#[test]
fn graph_contract_rejects_non_exchange_hidden_dtype() {
    let err = ExpertGraphBufferContract::glm52(
        LayerId(3),
        LayerWaveMode::Benchmark,
        GraphBucket::new(1),
        DType::F4,
        ModelFacts::default().quantization_recipe,
    )
    .unwrap_err();

    assert!(matches!(
        &err,
        GlmrtError::GraphBufferContractInvalid { .. }
    ));
    assert!(err.to_string().contains("not graphable for phase0"));
}

fn graph_prefill_batch(rows: usize) -> ExpertBatch {
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

fn graph_mixed_source_batch() -> ExpertBatch {
    let recipe = ModelFacts::default().quantization_recipe;
    let bucket = GraphBucket::new(8);
    let prefill = LayerWave::prefill(PrefillChunk::new(
        "prefill",
        "seq-a",
        3,
        0,
        2,
        55,
        Priority(2),
        bucket,
        "placement-a",
    ));
    let mtp = LayerWave::mtp_verify(MtpVerifyBlock::new(
        "mtp",
        "seq-a",
        3,
        2,
        2,
        Some(55),
        Priority(1),
        bucket,
        "placement-a",
    ));
    let decode = LayerWave::decode(DecodeStep::new(
        "decode",
        "seq-a",
        3,
        4,
        Some(55),
        Priority(0),
        "placement-a",
    ));
    let mut batch =
        ExpertBatch::from_wave_with_envelope(&prefill, DType::Bf16, recipe.clone(), bucket)
            .unwrap();
    batch
        .try_append_wave(&mtp, DType::Bf16, recipe.clone())
        .unwrap();
    batch.try_append_wave(&decode, DType::Bf16, recipe).unwrap();
    batch
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

fn expert_hosts() -> Vec<String> {
    EXPERT_HOSTS.iter().map(|host| (*host).to_owned()).collect()
}
