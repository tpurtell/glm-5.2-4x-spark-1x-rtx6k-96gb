use glmrt_core::DEFAULT_MODEL_ID;
use std::sync::Arc;

use crate::{RealFullInfo, RealFullRequestExecutor, RealSliceInfo};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiBackend {
    Tiny,
    SyntheticGlmLayer,
    RealGlmSlice,
    RealGlmFull,
}

impl ApiBackend {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "tiny" => Some(Self::Tiny),
            "synthetic-glm-layer" => Some(Self::SyntheticGlmLayer),
            "real-glm-slice" => Some(Self::RealGlmSlice),
            "real-glm-full" => Some(Self::RealGlmFull),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiTransport {
    Inproc,
    Tcp,
    TcpDebugJson,
    VerbsHost,
}

impl ApiTransport {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "inproc" => Some(Self::Inproc),
            "tcp" => Some(Self::Tcp),
            "tcp-debug-json" | "debug-json" => Some(Self::TcpDebugJson),
            "verbs-host" => Some(Self::VerbsHost),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Inproc => "inproc",
            Self::Tcp => "tcp",
            Self::TcpDebugJson => "tcp-debug-json",
            Self::VerbsHost => "verbs-host",
        }
    }
}

#[derive(Clone)]
pub struct ApiConfig {
    pub backend: ApiBackend,
    pub transport: ApiTransport,
    pub model_id: String,
    pub expert_targets: Vec<String>,
    pub real_slice: Option<RealSliceInfo>,
    pub real_full: Option<RealFullInfo>,
    pub real_full_executor: Option<Arc<dyn RealFullRequestExecutor>>,
}

impl std::fmt::Debug for ApiConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApiConfig")
            .field("backend", &self.backend)
            .field("transport", &self.transport)
            .field("model_id", &self.model_id)
            .field("expert_targets", &self.expert_targets)
            .field("real_slice", &self.real_slice)
            .field("real_full", &self.real_full)
            .field("real_full_executor", &self.real_full_executor.is_some())
            .finish()
    }
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            backend: ApiBackend::Tiny,
            transport: ApiTransport::Inproc,
            model_id: DEFAULT_MODEL_ID.to_owned(),
            expert_targets: Vec::new(),
            real_slice: None,
            real_full: None,
            real_full_executor: None,
        }
    }
}
