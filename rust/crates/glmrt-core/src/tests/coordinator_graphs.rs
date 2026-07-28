use super::*;

#[test]
fn glm52_coordinator_graph_plan_declares_all_instances() {
    let plans = CoordinatorGraphInstancePlan::glm52_bf16_all();
    assert_eq!(plans.len(), COORDINATOR_GRAPH_INSTANCE_COUNT);
    assert_eq!(COORDINATOR_GRAPH_INSTANCE_COUNT, 70);

    for shape in COORDINATOR_GRAPH_SHAPES {
        let shape_plans: Vec<_> = plans
            .iter()
            .filter(|plan| plan.key.shape == shape)
            .collect();
        assert_eq!(
            shape_plans
                .iter()
                .map(|plan| plan.key.row_bucket.row_capacity)
                .collect::<Vec<_>>(),
            vec![1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
        );
        assert!(shape_plans.iter().all(|plan| plan.key.dtype == DType::Bf16));
        assert!(shape_plans
            .iter()
            .all(|plan| plan.op_count == shape.op_count()));
        assert!(shape_plans
            .iter()
            .all(|plan| plan.network_boundary == shape.network_boundary()));
    }
}

#[test]
fn coordinator_graph_keys_do_not_split_by_source_mode() {
    let decode = CoordinatorGraphKey::glm52_bf16(
        CoordinatorGraphShape::CoordSparseA,
        LayerWaveMode::Decode,
        1,
    )
    .unwrap();
    let mtp = CoordinatorGraphKey::glm52_bf16(
        CoordinatorGraphShape::CoordSparseA,
        LayerWaveMode::MtpVerify,
        1,
    )
    .unwrap();
    let prefill = CoordinatorGraphKey::glm52_bf16(
        CoordinatorGraphShape::CoordSparseA,
        LayerWaveMode::Prefill,
        13,
    )
    .unwrap();
    let benchmark = CoordinatorGraphKey::glm52_bf16(
        CoordinatorGraphShape::CoordSparseA,
        LayerWaveMode::Benchmark,
        13,
    )
    .unwrap();

    assert_eq!(decode, mtp);
    assert_eq!(prefill, benchmark);
    assert_eq!(prefill.row_bucket, GraphBucket::new(16));
}

#[test]
fn coordinator_graph_bucket_selection_uses_decode_and_prefill_ceilings() {
    assert_eq!(
        coordinator_graph_bucket_for_active_rows(1).unwrap(),
        GraphBucket::decode()
    );
    assert_eq!(
        coordinator_graph_bucket_for_active_rows(2).unwrap(),
        GraphBucket::new(16)
    );
    assert_eq!(
        coordinator_graph_bucket_for_active_rows(17).unwrap(),
        GraphBucket::new(32)
    );
    assert_eq!(
        coordinator_graph_bucket_for_active_rows(512).unwrap(),
        GraphBucket::new(512)
    );
    assert_eq!(
        coordinator_graph_bucket_for_active_rows(513).unwrap(),
        GraphBucket::new(1024)
    );
    assert_eq!(
        coordinator_graph_bucket_for_active_rows(1024).unwrap(),
        GraphBucket::new(1024)
    );
    assert_eq!(
        coordinator_graph_bucket_for_active_rows(1025).unwrap(),
        GraphBucket::new(2048)
    );
    assert_eq!(
        coordinator_graph_bucket_for_active_rows(2048).unwrap(),
        GraphBucket::new(2048)
    );
    assert_eq!(
        coordinator_graph_bucket_for_active_rows(2049).unwrap(),
        GraphBucket::new(4096)
    );
    assert_eq!(
        coordinator_graph_bucket_for_active_rows(16384).unwrap(),
        GraphBucket::new(16384)
    );
    assert_eq!(
        coordinator_graph_bucket_for_active_rows(16385).unwrap(),
        GraphBucket::new(32768)
    );
    assert_eq!(
        coordinator_graph_bucket_for_active_rows(32769).unwrap(),
        GraphBucket::new(65536)
    );
    assert!(matches!(
        coordinator_graph_bucket_for_active_rows(0),
        Err(GlmrtError::GraphBufferContractInvalid { .. })
    ));
    assert!(matches!(
        coordinator_graph_bucket_for_active_rows(65537),
        Err(GlmrtError::GraphBufferContractInvalid { .. })
    ));
}

#[test]
fn coordinator_graph_shapes_validate_dense_and_sparse_layer_families() {
    CoordinatorGraphShape::CoordAttention
        .validate_layer(LayerId(0))
        .unwrap();
    CoordinatorGraphShape::CoordAttention
        .validate_layer(LayerId(77))
        .unwrap();
    CoordinatorGraphShape::CoordCompressedAttention
        .validate_layer(LayerId(78))
        .unwrap();
    CoordinatorGraphShape::CoordDense
        .validate_layer(LayerId(0))
        .unwrap();
    CoordinatorGraphShape::CoordDense
        .validate_layer(LayerId(2))
        .unwrap();
    CoordinatorGraphShape::CoordSparseA
        .validate_layer(LayerId(3))
        .unwrap();
    CoordinatorGraphShape::CoordSparseB
        .validate_layer(LayerId(77))
        .unwrap();
    CoordinatorGraphShape::CoordAttention
        .validate_layer(LayerId(78))
        .unwrap();
    CoordinatorGraphShape::CoordSparseA
        .validate_layer(LayerId(78))
        .unwrap();
    CoordinatorGraphShape::CoordSparseB
        .validate_layer(LayerId(78))
        .unwrap();

    assert!(matches!(
        CoordinatorGraphShape::CoordDense.validate_layer(LayerId(3)),
        Err(GlmrtError::GraphBufferContractInvalid { .. })
    ));
    assert!(matches!(
        CoordinatorGraphShape::CoordSparseA.validate_layer(LayerId(2)),
        Err(GlmrtError::GraphBufferContractInvalid { .. })
    ));
    assert!(matches!(
        CoordinatorGraphShape::CoordSparseB.validate_layer(LayerId(79)),
        Err(GlmrtError::GraphBufferContractInvalid { .. })
    ));
}
