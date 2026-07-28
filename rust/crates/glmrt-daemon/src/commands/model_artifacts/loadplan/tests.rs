use super::*;

fn assignment(
    tensor_name: &str,
    owner: &str,
    role: TensorRole,
    layer_id: Option<u32>,
    expert_id: Option<u32>,
) -> TensorAssignment {
    TensorAssignment {
        tensor_name: tensor_name.to_owned(),
        owner: owner.to_owned(),
        role,
        layer_id,
        expert_id,
    }
}

#[test]
fn expert_loadplan_summary_filters_owner_and_layer() {
    let plan = ExpertNodeLoadPlan {
        model_id: glmrt_core::DEFAULT_MODEL_ID.to_owned(),
        placement_version: "placement-test".to_owned(),
        owner: Some("ostrich".to_owned()),
        assignments: vec![
            assignment(
                "model.layers.3.mlp.experts.0.down_proj.weight",
                "ostrich",
                TensorRole::RoutedExpert,
                Some(3),
                Some(0),
            ),
            assignment(
                "model.layers.3.mlp.experts.0.down_proj.weight_scale",
                "ostrich",
                TensorRole::RoutedExpert,
                Some(3),
                Some(0),
            ),
            assignment(
                "model.layers.3.mlp.experts.1.down_proj.weight",
                "dodo",
                TensorRole::RoutedExpert,
                Some(3),
                Some(1),
            ),
            assignment(
                "model.layers.4.mlp.experts.0.down_proj.weight",
                "ostrich",
                TensorRole::RoutedExpert,
                Some(4),
                Some(0),
            ),
            assignment(
                "model.layers.3.mlp.shared_experts.down_proj.weight",
                "ostrich",
                TensorRole::SharedExpert,
                Some(3),
                None,
            ),
        ],
    };

    let readiness = summarize_expert_loadplan(&plan, None, Some(3));

    assert_eq!(readiness.owner, "ostrich");
    assert_eq!(readiness.source_assignments, 5);
    assert_eq!(readiness.assigned_tensors, 2);
    assert_eq!(readiness.assigned_layers, 1);
    assert_eq!(readiness.assigned_experts, 1);
    assert_eq!(readiness.routed_weight_tensors, 1);
    assert_eq!(readiness.routed_metadata_tensors, 1);
    assert_eq!(
        readiness.first_tensors,
        vec![
            "model.layers.3.mlp.experts.0.down_proj.weight".to_owned(),
            "model.layers.3.mlp.experts.0.down_proj.weight_scale".to_owned()
        ]
    );
}

#[test]
fn full_loadplan_summary_uses_explicit_role_hostname() {
    let plan = ExpertNodeLoadPlan {
        model_id: glmrt_core::DEFAULT_MODEL_ID.to_owned(),
        placement_version: "placement-test".to_owned(),
        owner: None,
        assignments: vec![
            assignment(
                "model.layers.5.mlp.experts.8.gate_proj.weight",
                "ostrich.spark.local",
                TensorRole::RoutedExpert,
                Some(5),
                Some(8),
            ),
            assignment(
                "model.layers.5.mlp.experts.12.gate_proj.weight",
                "kiwi",
                TensorRole::RoutedExpert,
                Some(5),
                Some(12),
            ),
        ],
    };

    let readiness = summarize_expert_loadplan(&plan, Some("ostrich"), None);

    assert_eq!(readiness.owner, "ostrich");
    assert_eq!(readiness.assigned_tensors, 1);
    assert_eq!(readiness.assigned_layers, 1);
    assert_eq!(readiness.assigned_experts, 1);
    assert_eq!(readiness.routed_weight_tensors, 1);
    assert_eq!(readiness.routed_metadata_tensors, 0);
}

#[test]
fn expert_owner_lookup_uses_routed_expert_assignments() {
    let plan = ExpertNodeLoadPlan {
        model_id: glmrt_core::DEFAULT_MODEL_ID.to_owned(),
        placement_version: "placement-test".to_owned(),
        owner: None,
        assignments: vec![
            assignment(
                "model.layers.3.mlp.experts.0.gate_proj.weight",
                "ostrich",
                TensorRole::RoutedExpert,
                Some(3),
                Some(0),
            ),
            assignment(
                "model.layers.3.mlp.experts.0.gate_proj.weight_scale",
                "ostrich",
                TensorRole::RoutedExpert,
                Some(3),
                Some(0),
            ),
            assignment(
                "model.layers.3.mlp.experts.1.gate_proj.weight",
                "dodo",
                TensorRole::RoutedExpert,
                Some(3),
                Some(1),
            ),
            assignment(
                "model.layers.3.mlp.shared_experts.down_proj.weight",
                "ostrich",
                TensorRole::SharedExpert,
                Some(3),
                None,
            ),
        ],
    };

    let lookup = expert_owner_lookup(&plan);

    assert_eq!(lookup.len(), 2);
    assert_eq!(lookup.owner_for(3, 0), Some("ostrich"));
    assert_eq!(lookup.owner_for(3, 1), Some("dodo"));
    assert_eq!(lookup.owner_for(4, 0), None);
}

#[test]
fn serving_loadplan_reader_returns_readiness_and_owner_lookup() {
    let tempdir = tempfile::tempdir().unwrap();
    let path = tempdir.path().join("loadplan.json");
    let plan = serde_json::json!({
        "model_id": glmrt_core::DEFAULT_MODEL_ID,
        "placement_version": "placement-test",
        "owner": null,
        "assignments": [
            {
                "tensor_name": "model.layers.3.mlp.experts.0.gate_proj.weight",
                "owner": "ostrich",
                "role": "routed-expert",
                "layer_id": 3,
                "expert_id": 0
            },
            {
                "tensor_name": "model.layers.3.mlp.experts.1.gate_proj.weight",
                "owner": "dodo",
                "role": "routed-expert",
                "layer_id": 3,
                "expert_id": 1
            },
            {
                "tensor_name": "model.layers.3.mlp.shared_experts.down_proj.weight",
                "owner": "ostrich",
                "role": "shared-expert",
                "layer_id": 3,
                "expert_id": null
            }
        ]
    });
    std::fs::write(&path, serde_json::to_vec(&plan).unwrap()).unwrap();

    let serving = read_expert_serving_loadplan(&path, Some("ostrich"), Some(3)).unwrap();

    assert_eq!(serving.readiness.owner, "ostrich");
    assert_eq!(serving.readiness.source_assignments, 3);
    assert_eq!(serving.readiness.assigned_tensors, 1);
    assert_eq!(serving.readiness.assigned_layers, 1);
    assert_eq!(serving.readiness.assigned_experts, 1);
    assert_eq!(serving.owner_lookup.len(), 2);
    assert_eq!(serving.owner_lookup.owner_for(3, 0), Some("ostrich"));
    assert_eq!(serving.owner_lookup.owner_for(3, 1), Some("dodo"));
}
