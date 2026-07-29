use anyhow::{Context, Result};
use glmrt_core::{DType, TensorCatalog, TensorInfo, TensorRole};
use glmrt_loader::{load_tensor_bytes, read_tensor_bytes_into, LoadedTensor, LoadedTensorSummary};
use std::collections::BTreeMap;
use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use super::coordinator_kernels::{
    coordinator_w4a16_o_proj_decode_enabled, coordinator_w4a16_q_b_decode_enabled,
    coordinator_w8a16_o_proj_decode_enabled, coordinator_w8a16_q_a_decode_enabled,
    coordinator_w8a16_q_b_decode_enabled, preload_coordinator_w4a16_projection,
    preload_coordinator_w8a16_projection, preload_resident_weight_from_host_staging,
    release_preloaded_resident_weight_device_buffer,
};
use super::layer_blocks::{tensor_is_spark_layer_block_resident, SparkLayerBlock};
use super::sparse_mlp::cache_router_correction_bias_host_values;
use super::types::RealFullCoordinatorResidentPreloadPlan;

const COORDINATOR_RESIDENT_PRELOAD_SCOPE: &str =
    "select coordinator-owned immutable GLM-5.2 tensors for named startup GPU residency";
const COORDINATOR_RESIDENT_SAMPLE_LIMIT: usize = 12;
const COORDINATOR_INCLUDE_MTP_LAYER_ENV: &str = "GLMRT_COORDINATOR_INCLUDE_MTP_LAYER";
const REAL_FULL_MTP_ENV: &str = "GLMRT_REAL_FULL_MTP";
const REAL_FULL_MTP_PROBE_ENV: &str = "GLMRT_REAL_FULL_MTP_PROBE";
const REQUIRED_COORDINATOR_RESIDENT_ROLE_LABELS: [&str; 7] = [
    "Embedding",
    "LmHead",
    "Attention",
    "Router",
    "Norm",
    "DenseMlp",
    "SharedExpert",
];

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct SparkLayerBlockResidentPreloadStats {
    pub(crate) layers: usize,
    pub(crate) tensors: usize,
    pub(crate) bytes: u64,
}

pub(crate) fn preload_real_full_spark_layer_block_weights(
    catalog: &TensorCatalog,
    block: SparkLayerBlock,
) -> Result<SparkLayerBlockResidentPreloadStats> {
    let tensors = catalog
        .tensors
        .iter()
        .filter(|tensor| tensor_is_spark_layer_block_resident(tensor, block))
        .collect::<Vec<_>>();
    anyhow::ensure!(
        !tensors.is_empty(),
        "Spark layer block {}:{} selected no resident tensors",
        block.start_layer,
        block.end_layer
    );
    let mut loaded_bytes = 0_u64;
    for tensor in &tensors {
        let expected_bytes: usize = tensor.byte_length.try_into().with_context(|| {
            format!(
                "Spark layer-block tensor {} byte length {} does not fit in usize",
                tensor.name, tensor.byte_length
            )
        })?;
        preload_resident_weight_from_host_staging(
            &tensor.name,
            expected_bytes,
            "startup resident Spark layer-block weight pinned staging",
            |staging| {
                let summary = read_tensor_bytes_into(catalog, &tensor.name, staging)
                    .with_context(|| format!("reading Spark layer-block tensor {}", tensor.name))?;
                validate_coordinator_resident_tensor_summary(tensor, &summary)?;
                cache_router_correction_bias_host_values(
                    catalog,
                    tensor,
                    &staging[..expected_bytes],
                )?;
                Ok(())
            },
        )
        .with_context(|| format!("preloading Spark layer-block tensor {}", tensor.name))?;
        loaded_bytes = loaded_bytes
            .checked_add(tensor.byte_length)
            .context("Spark layer-block resident byte count overflow")?;
    }
    Ok(SparkLayerBlockResidentPreloadStats {
        layers: block.layer_count(),
        tensors: tensors.len(),
        bytes: loaded_bytes,
    })
}

pub(super) fn real_full_coordinator_resident_preload_plan(
    catalog: &TensorCatalog,
) -> RealFullCoordinatorResidentPreloadPlan {
    coordinator_resident_preload_plan_for_tensors(catalog, "planned", 0)
}

pub(super) fn preload_real_full_coordinator_resident_weights(
    catalog: &TensorCatalog,
) -> Result<RealFullCoordinatorResidentPreloadPlan> {
    let preload_started = Instant::now();
    let tensors = coordinator_resident_tensors(catalog);
    let source_started = Instant::now();
    let next_tensor = AtomicUsize::new(0);
    let source_workers = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(32)
        .min(tensors.len().max(1));
    let (source_sender, source_receiver) = mpsc::channel();
    let mut source_bytes = 0_u64;
    let mut source_ms = 0.0_f64;
    let mut loaded_bytes = 0_u64;
    let mut upload_pack_ms = 0.0_f64;
    std::thread::scope(|scope| -> Result<()> {
        for _ in 0..source_workers {
            let sender = source_sender.clone();
            let next_tensor = &next_tensor;
            let tensors = &tensors;
            scope.spawn(move || loop {
                let index = next_tensor.fetch_add(1, Ordering::Relaxed);
                let Some(tensor) = tensors.get(index) else {
                    break;
                };
                let loaded = load_tensor_bytes(catalog, &tensor.name).with_context(|| {
                    format!("reading coordinator resident tensor {}", tensor.name)
                });
                let completed_ms = source_started.elapsed().as_secs_f64() * 1_000.0;
                if sender.send((index, loaded, completed_ms)).is_err() {
                    break;
                }
            });
        }
        drop(source_sender);
        for _ in 0..tensors.len() {
            let (index, loaded, completed_ms) = source_receiver
                .recv()
                .context("coordinator resident source workers stopped before completion")?;
            let tensor = tensors
                .get(index)
                .context("coordinator resident source worker returned an invalid tensor index")?;
            let loaded = loaded?;
            source_ms = source_ms.max(completed_ms);
            source_bytes = source_bytes
                .checked_add(loaded.bytes.len() as u64)
                .context("coordinator resident source byte count overflow")?;
            let upload_started = Instant::now();
            let tensor_bytes =
                preload_real_full_coordinator_resident_tensor(catalog, tensor, loaded)?;
            upload_pack_ms += upload_started.elapsed().as_secs_f64() * 1_000.0;
            loaded_bytes = loaded_bytes
                .checked_add(tensor_bytes)
                .context("coordinator resident loaded byte count overflow")?;
        }
        Ok(())
    })?;
    let source_gbps = source_bytes as f64 / (source_ms * 1.0e6).max(1.0);
    eprintln!(
        "real_full_coordinator_resident_source_load tensors={} workers={} bytes={} elapsed_ms={source_ms:.3} source_gbps={source_gbps:.3}",
        tensors.len(),
        source_workers,
        source_bytes,
    );
    let total_ms = preload_started.elapsed().as_secs_f64() * 1_000.0;
    let overlap_ms = (source_ms + upload_pack_ms - total_ms).max(0.0);
    let total_gbps = loaded_bytes as f64 / (total_ms * 1.0e6).max(1.0);
    eprintln!(
        "real_full_coordinator_resident_preload tensors={} bytes={} source_ms={source_ms:.3} upload_pack_ms={upload_pack_ms:.3} overlap_ms={overlap_ms:.3} total_ms={total_ms:.3} effective_gbps={total_gbps:.3}",
        tensors.len(),
        loaded_bytes,
    );
    Ok(coordinator_resident_preload_plan_for_tensors(
        catalog,
        "loaded",
        loaded_bytes,
    ))
}

fn preload_real_full_coordinator_resident_tensor(
    catalog: &TensorCatalog,
    tensor: &TensorInfo,
    loaded: LoadedTensor,
) -> Result<u64> {
    let expected_bytes: usize = tensor.byte_length.try_into().with_context(|| {
        format!(
            "coordinator resident tensor {} byte length {} does not fit in usize",
            tensor.name, tensor.byte_length
        )
    })?;
    let summary = loaded.summary();
    validate_coordinator_resident_tensor_summary(tensor, &summary)?;
    cache_router_correction_bias_host_values(catalog, tensor, &loaded.bytes)?;
    preload_resident_weight_from_host_staging(
        &tensor.name,
        expected_bytes,
        "startup resident coordinator weight pinned staging",
        |staging| {
            staging[..expected_bytes].copy_from_slice(&loaded.bytes);
            Ok(())
        },
    )
    .with_context(|| format!("preloading coordinator resident tensor {}", tensor.name))?;
    let pack_w4a16_q_b = tensor.name.ends_with(".self_attn.q_b_proj.weight")
        && coordinator_w4a16_q_b_decode_enabled();
    let pack_w4a16_o_proj = tensor.name.ends_with(".self_attn.o_proj.weight")
        && coordinator_w4a16_o_proj_decode_enabled();
    let pack_w8a16_o_proj = tensor.name.ends_with(".self_attn.o_proj.weight")
        && coordinator_w8a16_o_proj_decode_enabled();
    let pack_w8a16_q_a = tensor.name.ends_with(".self_attn.q_a_proj.weight")
        && coordinator_w8a16_q_a_decode_enabled();
    let pack_w8a16_q_b = tensor.name.ends_with(".self_attn.q_b_proj.weight")
        && coordinator_w8a16_q_b_decode_enabled();
    anyhow::ensure!(
        !(pack_w4a16_o_proj && pack_w8a16_o_proj),
        "coordinator O projection cannot enable W4A16 and W8A16 simultaneously"
    );
    anyhow::ensure!(
        !(pack_w4a16_q_b && pack_w8a16_q_b),
        "coordinator Q-B projection cannot enable W4A16 and W8A16 simultaneously"
    );
    if pack_w4a16_q_b || pack_w4a16_o_proj {
        anyhow::ensure!(
            tensor.dtype == DType::Bf16 && tensor.shape.len() == 2,
            "coordinator W4A16 projection {} must be a BF16 matrix",
            tensor.name
        );
        let size_n: usize = tensor.shape[0].try_into().with_context(|| {
            format!("coordinator W4A16 projection {} rows overflow", tensor.name)
        })?;
        let size_k: usize = tensor.shape[1].try_into().with_context(|| {
            format!(
                "coordinator W4A16 projection {} columns overflow",
                tensor.name
            )
        })?;
        preload_coordinator_w4a16_projection(&tensor.name, size_k, size_n)
            .with_context(|| format!("packing coordinator W4A16 projection {}", tensor.name))?;
    }
    if pack_w8a16_q_a || pack_w8a16_q_b || pack_w8a16_o_proj {
        anyhow::ensure!(
            tensor.dtype == DType::Bf16 && tensor.shape.len() == 2,
            "coordinator W8A16 projection {} must be a BF16 matrix",
            tensor.name
        );
        let size_n: usize = tensor.shape[0].try_into().with_context(|| {
            format!("coordinator W8A16 projection {} rows overflow", tensor.name)
        })?;
        let size_k: usize = tensor.shape[1].try_into().with_context(|| {
            format!(
                "coordinator W8A16 projection {} columns overflow",
                tensor.name
            )
        })?;
        preload_coordinator_w8a16_projection(&tensor.name, size_k, size_n)
            .with_context(|| format!("packing coordinator W8A16 projection {}", tensor.name))?;
        release_preloaded_resident_weight_device_buffer(&tensor.name, expected_bytes)
            .with_context(|| {
                format!(
                    "releasing superseded BF16 coordinator projection {}",
                    tensor.name
                )
            })?;
    }
    Ok(summary.bytes_read)
}

fn coordinator_resident_preload_plan_for_tensors(
    catalog: &TensorCatalog,
    status: &'static str,
    loaded_bytes: u64,
) -> RealFullCoordinatorResidentPreloadPlan {
    let tensors = coordinator_resident_tensors(catalog);
    let mut role_counts = BTreeMap::new();
    let mut role_bytes = BTreeMap::new();
    let mut bf16_tensors = 0_usize;
    let mut non_bf16_tensors = 0_usize;
    let mut selected_bytes = 0_u64;
    let mut sample_resident_keys = Vec::new();
    for tensor in &tensors {
        let role = coordinator_resident_role_label(&tensor.role).to_owned();
        *role_counts.entry(role.clone()).or_insert(0) += 1;
        *role_bytes.entry(role).or_insert(0) += tensor.byte_length;
        selected_bytes += tensor.byte_length;
        if tensor.dtype == DType::Bf16 {
            bf16_tensors += 1;
        } else {
            non_bf16_tensors += 1;
        }
        if sample_resident_keys.len() < COORDINATOR_RESIDENT_SAMPLE_LIMIT {
            sample_resident_keys.push(tensor.name.clone());
        }
    }
    let selected_tensor_count_from_roles = role_counts.values().copied().sum::<usize>();
    let selected_tensor_bytes_from_roles = role_bytes.values().copied().sum::<u64>();
    let missing_required_roles = REQUIRED_COORDINATOR_RESIDENT_ROLE_LABELS
        .iter()
        .filter(|role| role_counts.get(**role).copied().unwrap_or_default() == 0)
        .map(|role| (*role).to_owned())
        .collect::<Vec<_>>();

    let skipped_routed_expert_tensors = catalog
        .tensors
        .iter()
        .filter(|tensor| tensor.role == TensorRole::RoutedExpert)
        .count();
    let skipped_routed_expert_bytes = catalog
        .tensors
        .iter()
        .filter(|tensor| tensor.role == TensorRole::RoutedExpert)
        .map(|tensor| tensor.byte_length)
        .sum();
    let skipped_quantization_tensors = catalog
        .tensors
        .iter()
        .filter(|tensor| tensor.role == TensorRole::Quantization || tensor.is_quantization_metadata)
        .count();
    let skipped_quantization_bytes = catalog
        .tensors
        .iter()
        .filter(|tensor| tensor.role == TensorRole::Quantization || tensor.is_quantization_metadata)
        .map(|tensor| tensor.byte_length)
        .sum();
    let skipped_mtp_tensors = catalog
        .tensors
        .iter()
        .filter(|tensor| tensor.role == TensorRole::Mtp && !coordinator_resident_tensor(tensor))
        .count();
    let skipped_mtp_bytes = catalog
        .tensors
        .iter()
        .filter(|tensor| tensor.role == TensorRole::Mtp && !coordinator_resident_tensor(tensor))
        .map(|tensor| tensor.byte_length)
        .sum();

    RealFullCoordinatorResidentPreloadPlan {
        status,
        scope: COORDINATOR_RESIDENT_PRELOAD_SCOPE,
        startup_required: true,
        selected_tensor_count: tensors.len(),
        selected_tensor_bytes: selected_bytes,
        loaded_tensor_bytes: loaded_bytes,
        bf16_tensor_count: bf16_tensors,
        non_bf16_tensor_count: non_bf16_tensors,
        role_counts,
        role_bytes,
        required_role_count: REQUIRED_COORDINATOR_RESIDENT_ROLE_LABELS.len(),
        required_roles_present: REQUIRED_COORDINATOR_RESIDENT_ROLE_LABELS.len()
            - missing_required_roles.len(),
        missing_required_roles,
        selected_tensor_count_matches_roles: selected_tensor_count_from_roles == tensors.len(),
        selected_tensor_bytes_matches_roles: selected_tensor_bytes_from_roles == selected_bytes,
        skipped_routed_expert_tensors,
        skipped_routed_expert_bytes,
        skipped_quantization_tensors,
        skipped_quantization_bytes,
        skipped_mtp_tensors,
        skipped_mtp_bytes,
        sample_resident_keys,
        uses_named_resident_buffers: true,
    }
}

fn validate_coordinator_resident_tensor_summary(
    tensor: &TensorInfo,
    summary: &LoadedTensorSummary,
) -> Result<()> {
    if summary.tensor_name != tensor.name {
        anyhow::bail!(
            "coordinator resident tensor {} staged the wrong tensor {}",
            tensor.name,
            summary.tensor_name
        );
    }
    if summary.dtype != tensor.dtype {
        anyhow::bail!(
            "coordinator resident tensor {} dtype mismatch while staging: read {:?}, catalog {:?}",
            tensor.name,
            summary.dtype,
            tensor.dtype
        );
    }
    if summary.shape != tensor.shape {
        anyhow::bail!(
            "coordinator resident tensor {} shape mismatch while staging: read {:?}, catalog {:?}",
            tensor.name,
            summary.shape,
            tensor.shape
        );
    }
    if summary.role != tensor.role {
        anyhow::bail!(
            "coordinator resident tensor {} role mismatch while staging: read {:?}, catalog {:?}",
            tensor.name,
            summary.role,
            tensor.role
        );
    }
    if summary.layer_id != tensor.layer_id || summary.expert_id != tensor.expert_id {
        anyhow::bail!(
            "coordinator resident tensor {} layer/expert mismatch while staging: read layer={:?} expert={:?}, catalog layer={:?} expert={:?}",
            tensor.name,
            summary.layer_id,
            summary.expert_id,
            tensor.layer_id,
            tensor.expert_id
        );
    }
    if summary.byte_offset != tensor.byte_offset {
        anyhow::bail!(
            "coordinator resident tensor {} byte offset mismatch while staging: read {}, catalog {}",
            tensor.name,
            summary.byte_offset,
            tensor.byte_offset
        );
    }
    if summary.bytes_requested != tensor.byte_length || summary.bytes_read != tensor.byte_length {
        anyhow::bail!(
            "coordinator resident tensor {} byte count mismatch while staging: requested {} read {}, catalog {}",
            tensor.name,
            summary.bytes_requested,
            summary.bytes_read,
            tensor.byte_length
        );
    }
    Ok(())
}

fn coordinator_resident_tensors(catalog: &TensorCatalog) -> Vec<&TensorInfo> {
    catalog
        .tensors
        .iter()
        .filter(|tensor| coordinator_resident_tensor(tensor))
        .collect()
}

fn coordinator_resident_tensor(tensor: &TensorInfo) -> bool {
    coordinator_resident_role(&tensor.role)
        && !tensor.is_quantization_metadata
        && (tensor.role != TensorRole::Mtp || coordinator_mtp_residency_enabled())
        && !(tensor.role == TensorRole::Mtp && tensor.name.contains(".mlp.experts."))
}

fn coordinator_mtp_residency_enabled() -> bool {
    if let Some(include) = env::var(COORDINATOR_INCLUDE_MTP_LAYER_ENV)
        .ok()
        .and_then(|value| parse_bool_env_value(&value))
    {
        return include;
    }
    let mtp = env::var(REAL_FULL_MTP_ENV)
        .ok()
        .and_then(|value| parse_bool_env_value(&value));
    let probe = env::var(REAL_FULL_MTP_PROBE_ENV)
        .ok()
        .and_then(|value| parse_bool_env_value(&value))
        .unwrap_or(false);
    mtp.unwrap_or(true) || probe
}

fn parse_bool_env_value(value: &str) -> Option<bool> {
    match value.trim() {
        "1" | "true" | "TRUE" | "yes" | "YES" => Some(true),
        "0" | "false" | "FALSE" | "no" | "NO" => Some(false),
        _ => None,
    }
}

fn coordinator_resident_role(role: &TensorRole) -> bool {
    matches!(
        role,
        TensorRole::Embedding
            | TensorRole::LmHead
            | TensorRole::Attention
            | TensorRole::Router
            | TensorRole::Norm
            | TensorRole::DenseMlp
            | TensorRole::SharedExpert
            | TensorRole::Mtp
    )
}

fn coordinator_resident_role_label(role: &TensorRole) -> &'static str {
    match role {
        TensorRole::Embedding => "Embedding",
        TensorRole::LmHead => "LmHead",
        TensorRole::Attention => "Attention",
        TensorRole::Router => "Router",
        TensorRole::Norm => "Norm",
        TensorRole::DenseMlp => "DenseMlp",
        TensorRole::SharedExpert => "SharedExpert",
        TensorRole::Mtp => "Mtp",
        _ => "Other",
    }
}

#[cfg(test)]
mod tests {
    use super::{
        preload_real_full_coordinator_resident_weights,
        real_full_coordinator_resident_preload_plan, validate_coordinator_resident_tensor_summary,
    };
    use crate::commands::real_full::coordinator_kernels::{
        coordinator_cuda_reference_kernels_enabled, resident_weight_is_preloaded,
    };
    use crate::commands::real_full::tests::fixture::full_catalog;
    use glmrt_core::{DType, ModelFacts, TensorCatalog, TensorInfo, TensorRole, DEFAULT_MODEL_ID};
    use glmrt_loader::LoadedTensorSummary;
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn coordinator_resident_preload_plan_selects_only_coordinator_weights() {
        let catalog = full_catalog();
        let plan = real_full_coordinator_resident_preload_plan(&catalog);

        assert_eq!(plan.status, "planned");
        assert!(plan.startup_required);
        assert!(plan.uses_named_resident_buffers);
        assert_eq!(plan.loaded_tensor_bytes, 0);
        assert!(plan.selected_tensor_count > 0);
        assert!(plan.bf16_tensor_count > 0);
        assert!(
            plan.non_bf16_tensor_count > 0,
            "F32 router correction bias vectors are coordinator-resident immutable tensors"
        );
        assert_eq!(
            plan.selected_tensor_count,
            plan.bf16_tensor_count + plan.non_bf16_tensor_count
        );
        assert_eq!(
            plan.role_counts.get("Embedding").copied(),
            Some(1),
            "embedding table is coordinator-resident"
        );
        assert_eq!(
            plan.role_counts.get("LmHead").copied(),
            Some(1),
            "LM head is coordinator-resident"
        );
        assert!(plan.role_counts.get("Attention").copied().unwrap_or(0) > 0);
        assert!(plan.role_counts.get("Router").copied().unwrap_or(0) > 0);
        assert!(plan.role_counts.get("Norm").copied().unwrap_or(0) > 0);
        assert!(plan.role_counts.get("DenseMlp").copied().unwrap_or(0) > 0);
        assert_eq!(
            plan.required_roles_present, plan.required_role_count,
            "all coordinator-resident roles should be present in the fixture"
        );
        assert!(plan.missing_required_roles.is_empty());
        assert!(plan.selected_tensor_count_matches_roles);
        assert!(plan.selected_tensor_bytes_matches_roles);
        assert_eq!(plan.role_counts.get("RoutedExpert"), None);
        assert!(plan.skipped_routed_expert_tensors > 0);
        assert!(plan.skipped_quantization_tensors > 0);
        assert!(plan
            .sample_resident_keys
            .iter()
            .any(|name| name == "model.embed_tokens.weight"));
    }

    #[test]
    fn coordinator_resident_preload_plan_reports_missing_required_roles() {
        let mut catalog = full_catalog();
        catalog
            .tensors
            .retain(|tensor| tensor.role != TensorRole::LmHead);
        let plan = real_full_coordinator_resident_preload_plan(&catalog);

        assert!(plan.required_roles_present < plan.required_role_count);
        assert_eq!(plan.missing_required_roles, vec!["LmHead".to_owned()]);
        assert!(plan.selected_tensor_count_matches_roles);
        assert!(plan.selected_tensor_bytes_matches_roles);
    }

    #[test]
    fn coordinator_residency_selects_mtp_envelope_but_not_experts_or_metadata() {
        let mtp_layer = glmrt_core::GLM52_MTP_LAYER_ID as u32;
        let catalog = TensorCatalog {
            model_id: DEFAULT_MODEL_ID.to_owned(),
            snapshot_path: "/tmp/glmrt-snapshot".to_owned(),
            facts: ModelFacts::default(),
            tensors: vec![
                TensorInfo {
                    name: "model.layers.78.eh_proj.weight".to_owned(),
                    file: "model.safetensors".to_owned(),
                    dtype: DType::Bf16,
                    shape: vec![1],
                    byte_offset: 0,
                    byte_length: 2,
                    role: TensorRole::Mtp,
                    layer_id: Some(mtp_layer),
                    expert_id: None,
                    is_quantization_metadata: false,
                },
                TensorInfo {
                    name: "model.layers.78.eh_proj.weight_scale".to_owned(),
                    file: "model.safetensors".to_owned(),
                    dtype: DType::F32,
                    shape: vec![1],
                    byte_offset: 2,
                    byte_length: 4,
                    role: TensorRole::Mtp,
                    layer_id: Some(mtp_layer),
                    expert_id: None,
                    is_quantization_metadata: true,
                },
                TensorInfo {
                    name: "model.layers.78.mlp.experts.0.gate_proj.weight".to_owned(),
                    file: "model.safetensors".to_owned(),
                    dtype: DType::U8,
                    shape: vec![1],
                    byte_offset: 6,
                    byte_length: 1,
                    role: TensorRole::RoutedExpert,
                    layer_id: Some(mtp_layer),
                    expert_id: Some(0),
                    is_quantization_metadata: false,
                },
                TensorInfo {
                    name: "model.layers.78.mlp.experts.1.gate_proj.weight".to_owned(),
                    file: "model.safetensors".to_owned(),
                    dtype: DType::U8,
                    shape: vec![1],
                    byte_offset: 7,
                    byte_length: 1,
                    role: TensorRole::Mtp,
                    layer_id: Some(mtp_layer),
                    expert_id: Some(1),
                    is_quantization_metadata: false,
                },
            ],
        };

        let plan = real_full_coordinator_resident_preload_plan(&catalog);

        assert_eq!(plan.selected_tensor_count, 1);
        assert_eq!(plan.selected_tensor_bytes, 2);
        assert_eq!(plan.role_counts.get("Mtp"), Some(&1));
        assert_eq!(plan.skipped_mtp_tensors, 2);
        assert_eq!(plan.skipped_mtp_bytes, 5);
        assert_eq!(plan.skipped_routed_expert_tensors, 1);
        assert_eq!(plan.skipped_routed_expert_bytes, 1);
    }

    #[test]
    fn coordinator_resident_summary_validation_accepts_matching_catalog_entry() {
        let tensor = sample_tensor_info();
        let summary = sample_tensor_summary();

        validate_coordinator_resident_tensor_summary(&tensor, &summary)
            .expect("matching resident preload summary should validate");
    }

    #[test]
    fn coordinator_resident_summary_validation_rejects_identity_mismatch() {
        let tensor = sample_tensor_info();
        let mut summary = sample_tensor_summary();
        summary.tensor_name = "model.layers.0.self_attn.q_a_proj.weight".to_owned();

        let err = validate_coordinator_resident_tensor_summary(&tensor, &summary)
            .expect_err("wrong tensor name should fail validation");
        assert!(err.to_string().contains("staged the wrong tensor"));
    }

    #[test]
    fn coordinator_resident_summary_validation_rejects_shape_dtype_and_byte_mismatch() {
        let tensor = sample_tensor_info();

        let mut dtype_summary = sample_tensor_summary();
        dtype_summary.dtype = DType::F32;
        let err = validate_coordinator_resident_tensor_summary(&tensor, &dtype_summary)
            .expect_err("wrong dtype should fail validation");
        assert!(err.to_string().contains("dtype mismatch"));

        let mut shape_summary = sample_tensor_summary();
        shape_summary.shape = vec![4, 2];
        let err = validate_coordinator_resident_tensor_summary(&tensor, &shape_summary)
            .expect_err("wrong shape should fail validation");
        assert!(err.to_string().contains("shape mismatch"));

        let mut byte_summary = sample_tensor_summary();
        byte_summary.bytes_read -= 2;
        let err = validate_coordinator_resident_tensor_summary(&tensor, &byte_summary)
            .expect_err("short read should fail validation");
        assert!(err.to_string().contains("byte count mismatch"));
    }

    #[test]
    fn coordinator_resident_startup_preload_uploads_named_cuda_buffers_when_available() {
        let tempdir = tempfile::tempdir().unwrap();
        let (catalog, tensors) = tiny_resident_catalog(tempdir.path());

        let result = preload_real_full_coordinator_resident_weights(&catalog);
        let plan = match result {
            Ok(plan) => plan,
            Err(error) if !coordinator_cuda_reference_kernels_enabled() => {
                eprintln!("skipped: CUDA resident preload unavailable: {error:#}");
                return;
            }
            Err(error) => panic!("CUDA-required resident preload failed: {error:#}"),
        };

        assert_eq!(plan.status, "loaded");
        assert!(plan.startup_required);
        assert!(plan.uses_named_resident_buffers);
        assert_eq!(plan.required_roles_present, plan.required_role_count);
        assert!(plan.missing_required_roles.is_empty());
        assert_eq!(plan.selected_tensor_count, tensors.len());
        assert_eq!(plan.loaded_tensor_bytes, plan.selected_tensor_bytes);
        for tensor in tensors {
            assert!(
                resident_weight_is_preloaded(&tensor.name, tensor.byte_length as usize),
                "resident CUDA buffer {} should be preloaded",
                tensor.name
            );
        }
    }

    fn sample_tensor_info() -> TensorInfo {
        TensorInfo {
            name: "model.layers.0.input_layernorm.weight".to_owned(),
            file: "model-00001-of-00001.safetensors".to_owned(),
            dtype: DType::Bf16,
            shape: vec![4],
            byte_offset: 128,
            byte_length: 8,
            role: TensorRole::Norm,
            layer_id: Some(0),
            expert_id: None,
            is_quantization_metadata: false,
        }
    }

    fn sample_tensor_summary() -> LoadedTensorSummary {
        LoadedTensorSummary {
            tensor_name: "model.layers.0.input_layernorm.weight".to_owned(),
            source_path: "/tmp/model-00001-of-00001.safetensors".to_owned(),
            dtype: DType::Bf16,
            shape: vec![4],
            role: TensorRole::Norm,
            layer_id: Some(0),
            expert_id: None,
            byte_offset: 128,
            bytes_requested: 8,
            bytes_read: 8,
            elapsed_micros: 10,
            read_gbps: 0.001,
            sha256: String::new(),
        }
    }

    fn tiny_resident_catalog(snapshot_path: &std::path::Path) -> (TensorCatalog, Vec<TensorInfo>) {
        let shard_name = "tiny-resident.safetensors";
        let shard_path = snapshot_path.join(shard_name);
        let mut shard = File::create(&shard_path).expect("create tiny resident shard");
        let specs = [
            (
                "test.resident.embed_tokens.weight",
                TensorRole::Embedding,
                None,
                DType::Bf16,
                8_u64,
            ),
            (
                "test.resident.lm_head.weight",
                TensorRole::LmHead,
                None,
                DType::Bf16,
                8_u64,
            ),
            (
                "test.resident.layers.0.self_attn.q_a_proj.weight",
                TensorRole::Attention,
                Some(0),
                DType::Bf16,
                8_u64,
            ),
            (
                "test.resident.layers.0.mlp.gate.weight",
                TensorRole::Router,
                Some(0),
                DType::Bf16,
                8_u64,
            ),
            (
                "test.resident.layers.0.input_layernorm.weight",
                TensorRole::Norm,
                Some(0),
                DType::Bf16,
                8_u64,
            ),
            (
                "test.resident.layers.0.mlp.gate_proj.weight",
                TensorRole::DenseMlp,
                Some(0),
                DType::Bf16,
                8_u64,
            ),
            (
                "test.resident.layers.3.mlp.shared_experts.gate_proj.weight",
                TensorRole::SharedExpert,
                Some(3),
                DType::Bf16,
                8_u64,
            ),
        ];
        let mut offset = 0_u64;
        let mut tensors = Vec::new();
        for (index, (name, role, layer_id, dtype, byte_length)) in specs.iter().enumerate() {
            let bytes = (0..*byte_length)
                .map(|byte| (index as u8).wrapping_mul(17).wrapping_add(byte as u8))
                .collect::<Vec<_>>();
            shard
                .write_all(&bytes)
                .expect("write tiny resident tensor bytes");
            tensors.push(TensorInfo {
                name: (*name).to_owned(),
                file: shard_name.to_owned(),
                dtype: dtype.clone(),
                shape: vec![(*byte_length / 2) as usize],
                byte_offset: offset,
                byte_length: *byte_length,
                role: role.clone(),
                layer_id: *layer_id,
                expert_id: None,
                is_quantization_metadata: false,
            });
            offset += *byte_length;
        }
        let catalog = TensorCatalog {
            model_id: DEFAULT_MODEL_ID.to_owned(),
            snapshot_path: snapshot_path.display().to_string(),
            facts: ModelFacts::default(),
            tensors: tensors.clone(),
        };
        (catalog, tensors)
    }
}
