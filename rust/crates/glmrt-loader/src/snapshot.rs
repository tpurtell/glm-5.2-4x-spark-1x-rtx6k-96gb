use anyhow::{Context, Result};
use glmrt_core::{ModelFacts, TensorCatalog};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotResolution {
    pub model_id: String,
    pub cache_root: PathBuf,
    pub model_cache: PathBuf,
    pub snapshot_path: Option<PathBuf>,
    pub snapshots: Vec<PathBuf>,
}

pub fn default_hf_home() -> PathBuf {
    env::var_os("HF_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache/huggingface")))
        .unwrap_or_else(|| PathBuf::from("/root/.cache/huggingface"))
}

pub fn model_cache_dir(hf_home: &Path, model_id: &str) -> PathBuf {
    hf_home
        .join("hub")
        .join(format!("models--{}", model_id.replace('/', "--")))
}

pub fn resolve_snapshot(model_id: &str, hf_home: Option<&Path>) -> Result<SnapshotResolution> {
    let cache_root = hf_home
        .map(Path::to_path_buf)
        .unwrap_or_else(default_hf_home);
    let model_cache = model_cache_dir(&cache_root, model_id);
    let snapshots_root = model_cache.join("snapshots");
    let mut snapshots = Vec::new();
    if snapshots_root.is_dir() {
        for entry in fs::read_dir(&snapshots_root)
            .with_context(|| format!("reading {}", snapshots_root.display()))?
        {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                snapshots.push(entry.path());
            }
        }
    }
    snapshots.sort();
    let snapshot_path = snapshots.last().cloned();
    Ok(SnapshotResolution {
        model_id: model_id.to_owned(),
        cache_root,
        model_cache,
        snapshot_path,
        snapshots,
    })
}

pub fn empty_catalog_for_snapshot(model_id: &str, snapshot_path: &Path) -> TensorCatalog {
    TensorCatalog {
        model_id: model_id.to_owned(),
        snapshot_path: snapshot_path.display().to_string(),
        facts: ModelFacts {
            model_id: model_id.to_owned(),
            ..ModelFacts::default()
        },
        tensors: Vec::new(),
    }
}
