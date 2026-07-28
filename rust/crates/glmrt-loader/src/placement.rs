use anyhow::{Context, Result};
use glmrt_core::{
    owner_for_expert, LoadPlan, PlacementPolicy, TensorAssignment, TensorCatalog, TensorRole,
    COORDINATOR_HOST,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub fn build_load_plan(
    catalog: &TensorCatalog,
    policy: PlacementPolicy,
    expert_hosts: Vec<String>,
) -> Result<LoadPlan> {
    if expert_hosts.is_empty() {
        anyhow::bail!("at least one expert host is required");
    }
    let mut assignments = Vec::with_capacity(catalog.tensors.len());
    for tensor in &catalog.tensors {
        let owner = if tensor.role == TensorRole::RoutedExpert {
            let layer_id = tensor
                .layer_id
                .with_context(|| format!("routed tensor missing layer id: {}", tensor.name))?;
            let expert_id = tensor
                .expert_id
                .with_context(|| format!("routed tensor missing expert id: {}", tensor.name))?;
            owner_for_expert(layer_id as usize, expert_id as usize, &expert_hosts, policy)
                .expect("expert hosts are non-empty")
        } else {
            COORDINATOR_HOST.to_owned()
        };
        assignments.push(TensorAssignment {
            tensor_name: tensor.name.clone(),
            owner,
            role: tensor.role.clone(),
            layer_id: tensor.layer_id,
            expert_id: tensor.expert_id,
        });
    }
    let mut plan = LoadPlan {
        model_id: catalog.model_id.clone(),
        placement_version: String::new(),
        policy,
        coordinator_host: COORDINATOR_HOST.to_owned(),
        expert_hosts,
        assignments,
    };
    plan.placement_version = placement_version(catalog, &plan);
    Ok(plan)
}

fn placement_version(catalog: &TensorCatalog, plan: &LoadPlan) -> String {
    let mut hasher = Sha256::new();
    hasher.update(catalog.content_hash().as_bytes());
    hasher.update(format!("{:?}", plan.policy).as_bytes());
    for host in &plan.expert_hosts {
        hasher.update(host.as_bytes());
    }
    let digest = hasher.finalize();
    format!("{digest:x}")
}

pub fn assignments_by_owner(plan: &LoadPlan) -> BTreeMap<String, Vec<TensorAssignment>> {
    let mut out: BTreeMap<String, Vec<TensorAssignment>> = BTreeMap::new();
    for assignment in &plan.assignments {
        out.entry(assignment.owner.clone())
            .or_default()
            .push(assignment.clone());
    }
    out
}
