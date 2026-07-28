use crate::GlmrtError;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NodeRole {
    Coordinator,
    Expert,
    Dev,
}

impl NodeRole {
    pub fn expected_cuda_arch(self) -> &'static str {
        match self {
            NodeRole::Coordinator | NodeRole::Dev => "sm_120",
            NodeRole::Expert => "sm_121",
        }
    }
}

impl fmt::Display for NodeRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeRole::Coordinator => write!(f, "coordinator"),
            NodeRole::Expert => write!(f, "expert"),
            NodeRole::Dev => write!(f, "dev"),
        }
    }
}

impl FromStr for NodeRole {
    type Err = GlmrtError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "coordinator" | "oliver" => Ok(NodeRole::Coordinator),
            "expert" | "spark" => Ok(NodeRole::Expert),
            "dev" => Ok(NodeRole::Dev),
            other => Err(GlmrtError::UnknownRole(other.to_owned())),
        }
    }
}
