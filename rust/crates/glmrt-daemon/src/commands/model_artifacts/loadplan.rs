use anyhow::{Context, Result};
use glmrt_core::{ExpertOwnerLookup, LoadPlan, TensorAssignment, TensorRole};
use glmrt_loader::assignments_by_owner;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs::File;
use std::path::{Path, PathBuf};

#[cfg(test)]
mod tests;

#[derive(Debug, Serialize)]
struct NodeLoadPlan<'a> {
    model_id: &'a str,
    placement_version: &'a str,
    owner: &'a str,
    assignments: &'a [TensorAssignment],
}

#[derive(Debug, Deserialize)]
struct ExpertNodeLoadPlan {
    model_id: String,
    placement_version: String,
    owner: Option<String>,
    assignments: Vec<TensorAssignment>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ExpertLoadPlanReadiness {
    model_id: String,
    placement_version: String,
    owner: String,
    real_layer: Option<u32>,
    source_assignments: usize,
    assigned_tensors: usize,
    assigned_layers: usize,
    assigned_experts: usize,
    routed_weight_tensors: usize,
    routed_metadata_tensors: usize,
    first_tensors: Vec<String>,
}

pub(crate) struct ExpertServingLoadPlan {
    pub(crate) readiness: ExpertLoadPlanReadiness,
    pub(crate) owner_lookup: ExpertOwnerLookup,
}

pub(super) fn write_node_load_plans(out: &PathBuf, plan: &LoadPlan) -> Result<()> {
    let by_owner = assignments_by_owner(plan);
    let parent = out.parent().unwrap_or_else(|| Path::new("."));
    let stem = out
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("loadplan");
    for (owner, assignments) in &by_owner {
        let node_path = parent.join(format!("{stem}.{owner}.json"));
        let node_plan = NodeLoadPlan {
            model_id: &plan.model_id,
            placement_version: &plan.placement_version,
            owner,
            assignments,
        };
        serde_json::to_writer(File::create(&node_path)?, &node_plan)
            .with_context(|| format!("writing {}", node_path.display()))?;
    }
    Ok(())
}

pub(crate) fn read_expert_loadplan(
    path: &Path,
    role_hostname: Option<&str>,
    real_layer: Option<u32>,
) -> Result<ExpertLoadPlanReadiness> {
    let plan = read_expert_node_loadplan(path)?;
    Ok(summarize_expert_loadplan(&plan, role_hostname, real_layer))
}

pub(crate) fn read_expert_owner_lookup(path: &Path) -> Result<ExpertOwnerLookup> {
    let plan = read_expert_node_loadplan(path)?;
    Ok(expert_owner_lookup(&plan))
}

pub(crate) fn read_expert_serving_loadplan(
    path: &Path,
    role_hostname: Option<&str>,
    real_layer: Option<u32>,
) -> Result<ExpertServingLoadPlan> {
    let plan = read_expert_node_loadplan(path)?;
    Ok(ExpertServingLoadPlan {
        readiness: summarize_expert_loadplan(&plan, role_hostname, real_layer),
        owner_lookup: expert_owner_lookup(&plan),
    })
}

fn read_expert_node_loadplan(path: &Path) -> Result<ExpertNodeLoadPlan> {
    serde_json::from_reader(
        File::open(path).with_context(|| format!("opening {}", path.display()))?,
    )
    .with_context(|| format!("parsing {}", path.display()))
}

fn summarize_expert_loadplan(
    plan: &ExpertNodeLoadPlan,
    role_hostname: Option<&str>,
    real_layer: Option<u32>,
) -> ExpertLoadPlanReadiness {
    let owner = role_hostname
        .map(str::to_owned)
        .or_else(|| plan.owner.clone())
        .unwrap_or_else(current_hostname);
    let assigned = plan
        .assignments
        .iter()
        .filter(|assignment| host_matches(&assignment.owner, &owner))
        .filter(|assignment| assignment.role == TensorRole::RoutedExpert)
        .filter(|assignment| {
            real_layer
                .map(|layer| assignment.layer_id == Some(layer))
                .unwrap_or(true)
        })
        .collect::<Vec<_>>();
    let assigned_layers = assigned
        .iter()
        .filter_map(|assignment| assignment.layer_id)
        .collect::<BTreeSet<_>>();
    let assigned_experts = assigned
        .iter()
        .filter_map(|assignment| Some((assignment.layer_id?, assignment.expert_id?)))
        .collect::<BTreeSet<_>>();
    let routed_weight_tensors = assigned
        .iter()
        .filter(|assignment| assignment.tensor_name.ends_with(".weight"))
        .count();
    let routed_metadata_tensors = assigned.len().saturating_sub(routed_weight_tensors);
    let first_tensors = assigned
        .iter()
        .take(5)
        .map(|assignment| assignment.tensor_name.clone())
        .collect();

    ExpertLoadPlanReadiness {
        model_id: plan.model_id.clone(),
        placement_version: plan.placement_version.clone(),
        owner,
        real_layer,
        source_assignments: plan.assignments.len(),
        assigned_tensors: assigned.len(),
        assigned_layers: assigned_layers.len(),
        assigned_experts: assigned_experts.len(),
        routed_weight_tensors,
        routed_metadata_tensors,
        first_tensors,
    }
}

fn expert_owner_lookup(plan: &ExpertNodeLoadPlan) -> ExpertOwnerLookup {
    ExpertOwnerLookup::from_assignments(plan.assignments.iter())
}

fn current_hostname() -> String {
    std::env::var("GLMRT_ROLE_HOSTNAME")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            std::env::var("HOSTNAME")
                .ok()
                .filter(|value| !value.trim().is_empty())
        })
        .unwrap_or_else(|| "unknown".to_owned())
}

fn host_matches(assignment_owner: &str, requested_owner: &str) -> bool {
    assignment_owner == requested_owner
        || assignment_owner.split('.').next() == Some(requested_owner)
        || requested_owner.split('.').next() == Some(assignment_owner)
}
