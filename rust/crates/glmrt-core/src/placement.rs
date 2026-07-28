use crate::{GlmrtError, TensorRole, GLM52_ROUTED_EXPERTS};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, str::FromStr};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PlacementPolicy {
    Modulo,
    Range,
}

impl FromStr for PlacementPolicy {
    type Err = GlmrtError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "modulo" => Ok(PlacementPolicy::Modulo),
            "range" => Ok(PlacementPolicy::Range),
            other => Err(GlmrtError::UnknownPlacementPolicy(other.to_owned())),
        }
    }
}

pub fn owner_for_expert(
    layer_id: usize,
    expert_id: usize,
    hosts: &[String],
    policy: PlacementPolicy,
) -> Option<String> {
    if hosts.is_empty() {
        return None;
    }
    let index = match policy {
        PlacementPolicy::Modulo => (layer_id * GLM52_ROUTED_EXPERTS + expert_id) % hosts.len(),
        PlacementPolicy::Range => {
            let chunk = GLM52_ROUTED_EXPERTS.div_ceil(hosts.len());
            (expert_id / chunk).min(hosts.len() - 1)
        }
    };
    hosts.get(index).cloned()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorAssignment {
    pub tensor_name: String,
    pub owner: String,
    pub role: TensorRole,
    pub layer_id: Option<u32>,
    pub expert_id: Option<u32>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExpertOwnerLookup {
    owners_by_expert: BTreeMap<(usize, usize), String>,
}

impl ExpertOwnerLookup {
    pub fn from_pairs(pairs: impl IntoIterator<Item = ((usize, usize), String)>) -> Self {
        Self {
            owners_by_expert: pairs.into_iter().collect(),
        }
    }

    pub fn from_assignments<'a>(
        assignments: impl IntoIterator<Item = &'a TensorAssignment>,
    ) -> Self {
        let mut owners_by_expert = BTreeMap::new();
        for assignment in assignments {
            if assignment.role != TensorRole::RoutedExpert {
                continue;
            }
            let (Some(layer_id), Some(expert_id)) = (assignment.layer_id, assignment.expert_id)
            else {
                continue;
            };
            owners_by_expert
                .entry((layer_id as usize, expert_id as usize))
                .or_insert_with(|| assignment.owner.clone());
        }
        Self { owners_by_expert }
    }

    pub fn from_load_plan(plan: &LoadPlan) -> Self {
        Self::from_assignments(plan.assignments.iter())
    }

    pub fn owner_for(&self, layer_id: usize, expert_id: usize) -> Option<&str> {
        self.owners_by_expert
            .get(&(layer_id, expert_id))
            .map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.owners_by_expert.len()
    }

    pub fn is_empty(&self) -> bool {
        self.owners_by_expert.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadPlan {
    pub model_id: String,
    pub placement_version: String,
    pub policy: PlacementPolicy,
    pub coordinator_host: String,
    pub expert_hosts: Vec<String>,
    pub assignments: Vec<TensorAssignment>,
}

impl LoadPlan {
    pub fn readiness_hash(&self) -> String {
        let encoded = serde_json::to_vec(self).expect("serializing load plan cannot fail");
        let digest = Sha256::digest(encoded);
        format!("{digest:x}")
    }
}
