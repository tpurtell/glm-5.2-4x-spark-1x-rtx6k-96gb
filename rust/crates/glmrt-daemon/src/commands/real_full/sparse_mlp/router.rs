use anyhow::{Context, Result};
use glmrt_core::{DType, TensorCatalog, TensorInfo};
use glmrt_loader::{load_tensor_bytes, read_tensor_bytes_into};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use super::math::f32_bytes_to_f32;
use crate::commands::real_full::coordinator_kernels::{
    coordinator_cuda_reference_kernels_enabled, preload_resident_weight_from_host_staging,
    resident_weight_is_preloaded, router_topk_bf16, router_topk_bf16_preloaded_resident_weight,
    router_topk_bf16_preloaded_resident_weight_bias,
    router_topk_bf16_preloaded_resident_weight_bias_device_input,
    router_topk_bf16_preloaded_resident_weight_device_input, router_topk_bf16_resident_weight,
    router_topk_bf16_resident_weight_device_input, DeviceBf16Output,
};
use crate::commands::real_full::dense::math::bf16_bytes_from_f32;

#[derive(Clone)]
pub(in crate::commands::real_full) struct ScoredRoute {
    pub(in crate::commands::real_full) expert_id: usize,
    pub(in crate::commands::real_full) score: f32,
    pub(in crate::commands::real_full) corrected_score: f32,
    pub(in crate::commands::real_full) normalized_weight: f32,
}

pub(in crate::commands::real_full) struct RouterScoring {
    pub(in crate::commands::real_full) routes: Vec<ScoredRoute>,
    pub(in crate::commands::real_full) router_weight_bytes_read: u64,
    pub(in crate::commands::real_full) router_bias_bytes_read: u64,
    pub(in crate::commands::real_full) router_backend: &'static str,
}

pub(in crate::commands::real_full) struct RouterBatchScoring {
    pub(in crate::commands::real_full) row_routes: Vec<Vec<ScoredRoute>>,
    pub(in crate::commands::real_full) router_weight_bytes_read: u64,
    pub(in crate::commands::real_full) router_bias_bytes_read: u64,
    pub(in crate::commands::real_full) router_backend: &'static str,
}

struct RouterRouteSelection {
    routes: Vec<ScoredRoute>,
    backend: &'static str,
}

struct RouterBatchRouteSelection {
    row_routes: Vec<Vec<ScoredRoute>>,
    backend: &'static str,
}

#[derive(Default)]
pub(in crate::commands::real_full) struct RouterTensorCache {
    tensors_by_layer: HashMap<usize, CachedRouterTensors>,
    tensor_loads: usize,
    cache_hits: usize,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct RouterCorrectionBiasCacheKey {
    snapshot_path: String,
    file: String,
    tensor_name: String,
    byte_offset: u64,
    byte_length: u64,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct RouterTensorMetadataCacheKey {
    model_id: String,
    snapshot_path: String,
    layer_id: usize,
    cuda_reference_enabled: bool,
}

#[derive(Clone)]
struct CachedRouterTensors {
    router_weight_name: String,
    router_bias_name: String,
    router_weight_bf16_bytes: Option<Vec<u8>>,
    correction_bias: Arc<[f32]>,
    correction_bias_preloaded: bool,
    expert_count: usize,
    hidden_width: usize,
    router_weight_bytes: u64,
    router_bias_bytes: u64,
    tensor_loads: usize,
}

static ROUTER_CORRECTION_BIAS_HOST_CACHE: OnceLock<
    Mutex<HashMap<RouterCorrectionBiasCacheKey, Arc<[f32]>>>,
> = OnceLock::new();
static ROUTER_TENSOR_METADATA_CACHE: OnceLock<
    Mutex<HashMap<RouterTensorMetadataCacheKey, CachedRouterTensors>>,
> = OnceLock::new();

#[derive(Clone, Copy, Debug, Default)]
pub(in crate::commands::real_full) struct RouterTensorCacheStats {
    pub(in crate::commands::real_full) entries: usize,
    pub(in crate::commands::real_full) tensor_loads: usize,
    pub(in crate::commands::real_full) cache_hits: usize,
}

impl RouterTensorCache {
    pub(in crate::commands::real_full) fn stats(&self) -> RouterTensorCacheStats {
        RouterTensorCacheStats {
            entries: self.tensors_by_layer.len(),
            tensor_loads: self.tensor_loads,
            cache_hits: self.cache_hits,
        }
    }
}

pub(in crate::commands::real_full) fn cache_router_correction_bias_host_values(
    catalog: &TensorCatalog,
    tensor: &TensorInfo,
    bytes: &[u8],
) -> Result<bool> {
    if !is_router_correction_bias_tensor(tensor) {
        return Ok(false);
    }
    let expected_bytes = f32_vector_byte_len(tensor)?;
    if bytes.len() != expected_bytes {
        anyhow::bail!(
            "real full router correction-bias host cache byte mismatch for {}: got {} expected {}",
            tensor.name,
            bytes.len(),
            expected_bytes
        );
    }
    let values = f32_bytes_to_f32(bytes)?;
    validate_router_correction_bias_values(tensor, &values)?;
    router_correction_bias_host_cache()
        .lock()
        .map_err(|_| anyhow::anyhow!("router correction-bias host cache mutex poisoned"))?
        .insert(
            router_correction_bias_cache_key(catalog, tensor),
            values.into(),
        );
    Ok(true)
}

pub(in crate::commands::real_full) fn score_real_router_routes_bf16(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden_bf16: &[u8],
    hidden_width: usize,
    top_k: usize,
) -> Result<RouterScoring> {
    let tensors = load_router_tensors(catalog, layer_id)?;
    score_real_router_routes_from_bf16_tensors(layer_id, hidden_bf16, hidden_width, top_k, &tensors)
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn score_real_router_routes_bf16_cached(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden_bf16: &[u8],
    hidden_width: usize,
    top_k: usize,
    cache: &mut RouterTensorCache,
) -> Result<RouterScoring> {
    if let Some(tensors) = cache.tensors_by_layer.get(&layer_id) {
        cache.cache_hits += 1;
        return score_real_router_routes_from_bf16_tensors(
            layer_id,
            hidden_bf16,
            hidden_width,
            top_k,
            tensors,
        );
    }
    let tensors = load_router_tensors(catalog, layer_id)?;
    cache.tensor_loads += tensors.tensor_loads;
    cache.tensors_by_layer.insert(layer_id, tensors);
    let tensors = cache
        .tensors_by_layer
        .get(&layer_id)
        .expect("router tensor cache entry inserted");
    score_real_router_routes_from_bf16_tensors(layer_id, hidden_bf16, hidden_width, top_k, tensors)
}

pub(in crate::commands::real_full) fn score_real_router_routes_bf16_cached_device_input(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden_device: &DeviceBf16Output,
    hidden_host_bf16: Option<&[u8]>,
    hidden_width: usize,
    top_k: usize,
    cache: &mut RouterTensorCache,
) -> Result<RouterBatchScoring> {
    if let Some(tensors) = cache.tensors_by_layer.get(&layer_id) {
        cache.cache_hits += 1;
        return score_real_router_route_batch_from_bf16_device_tensors(
            layer_id,
            hidden_device,
            hidden_host_bf16,
            hidden_width,
            top_k,
            tensors,
        );
    }
    let tensors = load_router_tensors(catalog, layer_id)?;
    cache.tensor_loads += tensors.tensor_loads;
    cache.tensors_by_layer.insert(layer_id, tensors);
    let tensors = cache
        .tensors_by_layer
        .get(&layer_id)
        .expect("router tensor cache entry inserted");
    score_real_router_route_batch_from_bf16_device_tensors(
        layer_id,
        hidden_device,
        hidden_host_bf16,
        hidden_width,
        top_k,
        tensors,
    )
}

pub(super) fn score_real_router_routes_cached(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden: &[f32],
    top_k: usize,
    cache: &mut RouterTensorCache,
) -> Result<RouterScoring> {
    if let Some(tensors) = cache.tensors_by_layer.get(&layer_id) {
        cache.cache_hits += 1;
        return score_real_router_routes_from_tensors(layer_id, hidden, top_k, tensors);
    }
    let tensors = load_router_tensors(catalog, layer_id)?;
    cache.tensor_loads += tensors.tensor_loads;
    cache.tensors_by_layer.insert(layer_id, tensors);
    let tensors = cache
        .tensors_by_layer
        .get(&layer_id)
        .expect("router tensor cache entry inserted");
    score_real_router_routes_from_tensors(layer_id, hidden, top_k, tensors)
}

fn load_router_tensors(catalog: &TensorCatalog, layer_id: usize) -> Result<CachedRouterTensors> {
    let cuda_reference_enabled = coordinator_cuda_reference_kernels_enabled();
    let cache_key = router_tensor_metadata_cache_key(catalog, layer_id, cuda_reference_enabled);
    if let Some(mut tensors) = router_tensor_metadata_cache()
        .lock()
        .map_err(|_| anyhow::anyhow!("router tensor metadata cache mutex poisoned"))?
        .get(&cache_key)
        .cloned()
    {
        tensors.tensor_loads = 0;
        return Ok(tensors);
    }

    let tensors = load_router_tensors_uncached(catalog, layer_id, cuda_reference_enabled)?;
    router_tensor_metadata_cache()
        .lock()
        .map_err(|_| anyhow::anyhow!("router tensor metadata cache mutex poisoned"))?
        .insert(cache_key, tensors.clone());
    Ok(tensors)
}

fn load_router_tensors_uncached(
    catalog: &TensorCatalog,
    layer_id: usize,
    cuda_reference_enabled: bool,
) -> Result<CachedRouterTensors> {
    let router_weight_name = format!("model.layers.{layer_id}.mlp.gate.weight");
    let router_bias_name = format!("model.layers.{layer_id}.mlp.gate.e_score_correction_bias");
    let router_weight_info = catalog_tensor(catalog, &router_weight_name)?;
    let router_bias_info = catalog_tensor(catalog, &router_bias_name)?;
    if router_weight_info.dtype != DType::Bf16 || router_bias_info.dtype != DType::F32 {
        anyhow::bail!(
            "real full NVFP4 routed probe expects BF16 router weight and F32 correction bias for layer {layer_id}, got {:?} and {:?}",
            router_weight_info.dtype,
            router_bias_info.dtype
        );
    }
    if router_weight_info.shape.len() != 2 {
        anyhow::bail!(
            "real full NVFP4 routed probe expected rank-2 router weight for layer {layer_id}, got {:?}",
            router_weight_info.shape
        );
    }
    let expert_count = router_weight_info.shape[0];
    let hidden_width = router_weight_info.shape[1];
    if router_bias_info.shape != vec![expert_count] {
        anyhow::bail!(
            "real full NVFP4 routed probe expected rank-1 router correction bias for layer {layer_id} with {expert_count} experts, got {:?}",
            router_bias_info.shape
        );
    }
    let router_weight_bytes = bf16_matrix_byte_len(router_weight_info)?;
    let router_bias_bytes = f32_vector_byte_len(router_bias_info)?;
    let mut tensor_loads = 0_usize;
    let router_weight_bf16_bytes = if cuda_reference_enabled {
        if !resident_weight_is_preloaded(&router_weight_name, router_weight_bytes) {
            preload_router_weight_resident_from_host_staging(
                catalog,
                router_weight_info,
                router_weight_bytes,
            )?;
            tensor_loads += 1;
        }
        None
    } else {
        let router_weight = load_tensor_bytes(catalog, &router_weight_name)?;
        if router_weight.bytes.len() != router_weight_bytes {
            anyhow::bail!(
                "real full NVFP4 routed probe router weight byte length mismatch for layer {layer_id}: loaded={} expected={router_weight_bytes}",
                router_weight.bytes.len()
            );
        }
        tensor_loads += 1;
        Some(router_weight.bytes)
    };
    let (correction_bias, loaded_bias_from_tensor) =
        load_router_correction_bias(catalog, layer_id, router_bias_info, router_bias_bytes)?;
    if loaded_bias_from_tensor {
        tensor_loads += 1;
    }
    let router_bias_preloaded = cuda_reference_enabled
        && resident_weight_is_preloaded(&router_bias_name, router_bias_bytes);
    Ok(CachedRouterTensors {
        router_weight_name,
        router_bias_name,
        router_weight_bf16_bytes,
        correction_bias,
        correction_bias_preloaded: router_bias_preloaded,
        expert_count,
        hidden_width,
        router_weight_bytes: router_weight_bytes as u64,
        router_bias_bytes: router_bias_bytes as u64,
        tensor_loads,
    })
}

fn router_tensor_metadata_cache_key(
    catalog: &TensorCatalog,
    layer_id: usize,
    cuda_reference_enabled: bool,
) -> RouterTensorMetadataCacheKey {
    RouterTensorMetadataCacheKey {
        model_id: catalog.model_id.clone(),
        snapshot_path: catalog.snapshot_path.clone(),
        layer_id,
        cuda_reference_enabled,
    }
}

fn router_tensor_metadata_cache(
) -> &'static Mutex<HashMap<RouterTensorMetadataCacheKey, CachedRouterTensors>> {
    ROUTER_TENSOR_METADATA_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn preload_router_weight_resident_from_host_staging(
    catalog: &TensorCatalog,
    router_weight_info: &TensorInfo,
    expected_bytes: usize,
) -> Result<()> {
    preload_resident_weight_from_host_staging(
        &router_weight_info.name,
        expected_bytes,
        "router weight pinned staging",
        |staging| {
            let summary = read_tensor_bytes_into(catalog, &router_weight_info.name, staging)
                .with_context(|| {
                    format!(
                        "reading router weight tensor {} into pinned staging",
                        router_weight_info.name
                    )
                })?;
            if summary.dtype != DType::Bf16 {
                anyhow::bail!(
                    "router weight tensor {} expects BF16, got {:?}",
                    router_weight_info.name,
                    summary.dtype
                );
            }
            if summary.shape != router_weight_info.shape {
                anyhow::bail!(
                    "router weight tensor {} shape mismatch: expected {:?} got {:?}",
                    router_weight_info.name,
                    router_weight_info.shape,
                    summary.shape
                );
            }
            if summary.bytes_read as usize != expected_bytes {
                anyhow::bail!(
                    "router weight tensor {} read {} bytes, expected {}",
                    router_weight_info.name,
                    summary.bytes_read,
                    expected_bytes
                );
            }
            Ok(())
        },
    )
    .with_context(|| {
        format!(
            "preloading router weight tensor {} from pinned staging",
            router_weight_info.name
        )
    })
}

fn load_router_correction_bias(
    catalog: &TensorCatalog,
    layer_id: usize,
    router_bias_info: &TensorInfo,
    router_bias_bytes: usize,
) -> Result<(Arc<[f32]>, bool)> {
    if let Some(correction_bias) = cached_router_correction_bias(catalog, router_bias_info)? {
        return Ok((correction_bias, false));
    }
    if coordinator_cuda_reference_kernels_enabled()
        && !resident_weight_is_preloaded(&router_bias_info.name, router_bias_bytes)
    {
        preload_router_correction_bias_resident_from_host_staging(
            catalog,
            router_bias_info,
            router_bias_bytes,
        )?;
        if let Some(correction_bias) = cached_router_correction_bias(catalog, router_bias_info)? {
            return Ok((correction_bias, true));
        }
    }
    let router_bias = load_tensor_bytes(catalog, &router_bias_info.name)?;
    if router_bias.bytes.len() != router_bias_bytes {
        anyhow::bail!(
            "real full NVFP4 routed probe router correction bias byte length mismatch for layer {layer_id}: loaded={} expected={router_bias_bytes}",
            router_bias.bytes.len()
        );
    }
    cache_router_correction_bias_host_values(catalog, router_bias_info, &router_bias.bytes)?;
    let correction_bias =
        cached_router_correction_bias(catalog, router_bias_info)?.with_context(|| {
            format!(
                "router correction-bias cache fill failed for {}",
                router_bias_info.name
            )
        })?;
    Ok((correction_bias, true))
}

fn preload_router_correction_bias_resident_from_host_staging(
    catalog: &TensorCatalog,
    router_bias_info: &TensorInfo,
    expected_bytes: usize,
) -> Result<()> {
    preload_resident_weight_from_host_staging(
        &router_bias_info.name,
        expected_bytes,
        "router correction-bias pinned staging",
        |staging| {
            let summary = read_tensor_bytes_into(catalog, &router_bias_info.name, staging)
                .with_context(|| {
                    format!(
                        "reading router correction-bias tensor {} into pinned staging",
                        router_bias_info.name
                    )
                })?;
            if summary.dtype != DType::F32 {
                anyhow::bail!(
                    "router correction-bias tensor {} expects F32, got {:?}",
                    router_bias_info.name,
                    summary.dtype
                );
            }
            if summary.shape != router_bias_info.shape {
                anyhow::bail!(
                    "router correction-bias tensor {} shape mismatch: expected {:?} got {:?}",
                    router_bias_info.name,
                    router_bias_info.shape,
                    summary.shape
                );
            }
            if summary.bytes_read as usize != expected_bytes {
                anyhow::bail!(
                    "router correction-bias tensor {} read {} bytes, expected {}",
                    router_bias_info.name,
                    summary.bytes_read,
                    expected_bytes
                );
            }
            cache_router_correction_bias_host_values(catalog, router_bias_info, staging)?;
            Ok(())
        },
    )
    .with_context(|| {
        format!(
            "preloading router correction-bias tensor {} from pinned staging",
            router_bias_info.name
        )
    })
}

fn cached_router_correction_bias(
    catalog: &TensorCatalog,
    tensor: &TensorInfo,
) -> Result<Option<Arc<[f32]>>> {
    if !is_router_correction_bias_tensor(tensor) {
        return Ok(None);
    }
    Ok(router_correction_bias_host_cache()
        .lock()
        .map_err(|_| anyhow::anyhow!("router correction-bias host cache mutex poisoned"))?
        .get(&router_correction_bias_cache_key(catalog, tensor))
        .cloned())
}

fn router_correction_bias_host_cache(
) -> &'static Mutex<HashMap<RouterCorrectionBiasCacheKey, Arc<[f32]>>> {
    ROUTER_CORRECTION_BIAS_HOST_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn router_correction_bias_cache_key(
    catalog: &TensorCatalog,
    tensor: &TensorInfo,
) -> RouterCorrectionBiasCacheKey {
    RouterCorrectionBiasCacheKey {
        snapshot_path: catalog.snapshot_path.clone(),
        file: tensor.file.clone(),
        tensor_name: tensor.name.clone(),
        byte_offset: tensor.byte_offset,
        byte_length: tensor.byte_length,
    }
}

fn is_router_correction_bias_tensor(tensor: &TensorInfo) -> bool {
    tensor.dtype == DType::F32 && tensor.name.ends_with(".mlp.gate.e_score_correction_bias")
}

fn validate_router_correction_bias_values(tensor: &TensorInfo, values: &[f32]) -> Result<()> {
    if tensor.shape.len() != 1 || tensor.shape[0] != values.len() {
        anyhow::bail!(
            "real full router correction-bias host cache shape mismatch for {}: shape {:?} values {}",
            tensor.name,
            tensor.shape,
            values.len()
        );
    }
    if !values.iter().all(|value| value.is_finite()) {
        anyhow::bail!(
            "real full router correction-bias host cache found non-finite value for {}",
            tensor.name
        );
    }
    Ok(())
}

fn score_real_router_routes_from_tensors(
    layer_id: usize,
    hidden: &[f32],
    top_k: usize,
    tensors: &CachedRouterTensors,
) -> Result<RouterScoring> {
    let selection = if let Some(router_weight_bf16_bytes) = &tensors.router_weight_bf16_bytes {
        score_router_routes_with_backend(
            layer_id,
            Some(tensors.router_weight_name.as_str()),
            hidden,
            top_k,
            router_weight_bf16_bytes,
            &tensors.correction_bias,
            tensors.expert_count,
            tensors.hidden_width,
        )?
    } else {
        score_router_routes_with_preloaded_backend(
            layer_id,
            tensors.router_weight_name.as_str(),
            tensors
                .correction_bias_preloaded
                .then_some(tensors.router_bias_name.as_str()),
            hidden,
            top_k,
            &tensors.correction_bias,
            tensors.expert_count,
            tensors.hidden_width,
        )?
    };
    Ok(RouterScoring {
        routes: selection.routes,
        router_weight_bytes_read: tensors.router_weight_bytes,
        router_bias_bytes_read: tensors.router_bias_bytes,
        router_backend: selection.backend,
    })
}

fn score_real_router_routes_from_bf16_tensors(
    layer_id: usize,
    hidden_bf16: &[u8],
    hidden_width: usize,
    top_k: usize,
    tensors: &CachedRouterTensors,
) -> Result<RouterScoring> {
    let selection = if let Some(router_weight_bf16_bytes) = &tensors.router_weight_bf16_bytes {
        score_router_routes_bf16_with_backend(
            layer_id,
            Some(tensors.router_weight_name.as_str()),
            hidden_bf16,
            hidden_width,
            top_k,
            router_weight_bf16_bytes,
            &tensors.correction_bias,
            tensors.expert_count,
            tensors.hidden_width,
        )?
    } else {
        score_router_routes_bf16_with_preloaded_backend(
            layer_id,
            tensors.router_weight_name.as_str(),
            tensors
                .correction_bias_preloaded
                .then_some(tensors.router_bias_name.as_str()),
            hidden_bf16,
            hidden_width,
            top_k,
            &tensors.correction_bias,
            tensors.expert_count,
            tensors.hidden_width,
        )?
    };
    Ok(RouterScoring {
        routes: selection.routes,
        router_weight_bytes_read: tensors.router_weight_bytes,
        router_bias_bytes_read: tensors.router_bias_bytes,
        router_backend: selection.backend,
    })
}

fn score_real_router_route_batch_from_bf16_device_tensors(
    layer_id: usize,
    hidden_device: &DeviceBf16Output,
    hidden_host_bf16: Option<&[u8]>,
    hidden_width: usize,
    top_k: usize,
    tensors: &CachedRouterTensors,
) -> Result<RouterBatchScoring> {
    if hidden_width != tensors.hidden_width || hidden_device.values_per_row != hidden_width {
        anyhow::bail!(
            "real full router device-input hidden width mismatch: requested={} device={} router_width={}",
            hidden_width,
            hidden_device.values_per_row,
            tensors.hidden_width
        );
    }
    let selection = if coordinator_cuda_reference_kernels_enabled() {
        if let Some(router_weight_bf16_bytes) = &tensors.router_weight_bf16_bytes {
            score_router_routes_bf16_device_with_backend(
                layer_id,
                tensors.router_weight_name.as_str(),
                hidden_device,
                top_k,
                router_weight_bf16_bytes,
                &tensors.correction_bias,
                tensors.expert_count,
            )?
        } else {
            score_router_routes_bf16_device_with_preloaded_backend(
                layer_id,
                tensors.router_weight_name.as_str(),
                tensors
                    .correction_bias_preloaded
                    .then_some(tensors.router_bias_name.as_str()),
                hidden_device,
                top_k,
                &tensors.correction_bias,
                tensors.expert_count,
            )?
        }
    } else {
        let hidden_host_bf16 = hidden_host_bf16.with_context(|| {
            format!(
                "real full router layer {layer_id} device-input fallback requires host BF16 hidden bytes when CUDA is disabled"
            )
        })?;
        score_router_routes_bf16_host_batch_with_backend(
            layer_id,
            hidden_host_bf16,
            hidden_device.rows,
            hidden_width,
            top_k,
            tensors,
        )?
    };
    Ok(RouterBatchScoring {
        row_routes: selection.row_routes,
        router_weight_bytes_read: tensors.router_weight_bytes,
        router_bias_bytes_read: tensors.router_bias_bytes,
        router_backend: selection.backend,
    })
}

#[cfg(test)]
fn score_router_routes(
    layer_id: usize,
    hidden: &[f32],
    top_k: usize,
    router_values: &[f32],
    correction_bias: &[f32],
    expert_count: usize,
    hidden_width: usize,
) -> Result<Vec<ScoredRoute>> {
    Ok(score_router_routes_with_backend(
        layer_id,
        None,
        hidden,
        top_k,
        &bf16_bytes_from_f32(router_values),
        correction_bias,
        expert_count,
        hidden_width,
    )?
    .routes)
}

fn score_router_routes_with_backend(
    layer_id: usize,
    resident_router_weight_name: Option<&str>,
    hidden: &[f32],
    top_k: usize,
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    expert_count: usize,
    hidden_width: usize,
) -> Result<RouterRouteSelection> {
    let hidden_bf16 = bf16_bytes_from_f32(hidden);
    score_router_routes_bf16_with_backend(
        layer_id,
        resident_router_weight_name,
        &hidden_bf16,
        hidden.len(),
        top_k,
        router_weight_bf16,
        correction_bias,
        expert_count,
        hidden_width,
    )
}

fn score_router_routes_with_preloaded_backend(
    layer_id: usize,
    resident_router_weight_name: &str,
    resident_correction_bias_name: Option<&str>,
    hidden: &[f32],
    top_k: usize,
    correction_bias: &[f32],
    expert_count: usize,
    hidden_width: usize,
) -> Result<RouterRouteSelection> {
    let hidden_bf16 = bf16_bytes_from_f32(hidden);
    score_router_routes_bf16_with_preloaded_backend(
        layer_id,
        resident_router_weight_name,
        resident_correction_bias_name,
        &hidden_bf16,
        hidden.len(),
        top_k,
        correction_bias,
        expert_count,
        hidden_width,
    )
}

fn score_router_routes_bf16_with_backend(
    layer_id: usize,
    resident_router_weight_name: Option<&str>,
    hidden_bf16: &[u8],
    hidden_width: usize,
    top_k: usize,
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    expert_count: usize,
    router_hidden_width: usize,
) -> Result<RouterRouteSelection> {
    if hidden_width != router_hidden_width {
        anyhow::bail!(
            "real full router score hidden width mismatch: hidden={} router_width={}",
            hidden_width,
            router_hidden_width
        );
    }
    let expected_hidden_bytes = hidden_width
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| anyhow::anyhow!("real full router hidden byte length overflow"))?;
    if hidden_bf16.len() != expected_hidden_bytes {
        anyhow::bail!(
            "real full BF16 router score hidden byte length mismatch: values={} expected={expected_hidden_bytes}",
            hidden_bf16.len()
        );
    }
    if top_k == 0 || top_k > expert_count {
        anyhow::bail!("real full router score invalid top_k={top_k} for experts={expert_count}");
    }
    let expected_weight_bytes = expert_count
        .checked_mul(router_hidden_width)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .ok_or_else(|| anyhow::anyhow!("real full router weight byte length overflow"))?;
    if router_weight_bf16.len() != expected_weight_bytes {
        anyhow::bail!(
            "real full BF16 router score weight byte length mismatch: values={} expected={}",
            router_weight_bf16.len(),
            expected_weight_bytes
        );
    }
    if correction_bias.len() != expert_count {
        anyhow::bail!(
            "real full router score correction bias length mismatch: bias={} experts={expert_count}",
            correction_bias.len()
        );
    }

    let topk = if let Some(weight_name) = resident_router_weight_name {
        router_topk_bf16_resident_weight(
            weight_name,
            hidden_bf16,
            router_weight_bf16,
            correction_bias,
            1,
            router_hidden_width,
            expert_count,
            top_k,
        )?
    } else {
        router_topk_bf16(
            hidden_bf16,
            router_weight_bf16,
            correction_bias,
            1,
            router_hidden_width,
            expert_count,
            top_k,
        )?
    };
    let routes = routes_from_topk(
        layer_id,
        top_k,
        correction_bias,
        topk.indices,
        topk.scores,
        topk.weights,
    )?;
    Ok(RouterRouteSelection {
        routes,
        backend: topk.backend,
    })
}

fn score_router_routes_bf16_with_preloaded_backend(
    layer_id: usize,
    resident_router_weight_name: &str,
    resident_correction_bias_name: Option<&str>,
    hidden_bf16: &[u8],
    hidden_width: usize,
    top_k: usize,
    correction_bias: &[f32],
    expert_count: usize,
    router_hidden_width: usize,
) -> Result<RouterRouteSelection> {
    if hidden_width != router_hidden_width {
        anyhow::bail!(
            "real full router score hidden width mismatch: hidden={} router_width={}",
            hidden_width,
            router_hidden_width
        );
    }
    let expected_hidden_bytes = hidden_width
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| anyhow::anyhow!("real full router hidden byte length overflow"))?;
    if hidden_bf16.len() != expected_hidden_bytes {
        anyhow::bail!(
            "real full BF16 router score hidden byte length mismatch: values={} expected={expected_hidden_bytes}",
            hidden_bf16.len()
        );
    }
    if top_k == 0 || top_k > expert_count {
        anyhow::bail!("real full router score invalid top_k={top_k} for experts={expert_count}");
    }
    if correction_bias.len() != expert_count {
        anyhow::bail!(
            "real full router score correction bias length mismatch: bias={} experts={expert_count}",
            correction_bias.len()
        );
    }

    let topk = if let Some(correction_bias_name) = resident_correction_bias_name {
        router_topk_bf16_preloaded_resident_weight_bias(
            resident_router_weight_name,
            correction_bias_name,
            hidden_bf16,
            correction_bias,
            1,
            router_hidden_width,
            expert_count,
            top_k,
        )?
    } else {
        router_topk_bf16_preloaded_resident_weight(
            resident_router_weight_name,
            hidden_bf16,
            correction_bias,
            1,
            router_hidden_width,
            expert_count,
            top_k,
        )?
    };
    let routes = routes_from_topk(
        layer_id,
        top_k,
        correction_bias,
        topk.indices,
        topk.scores,
        topk.weights,
    )?;
    Ok(RouterRouteSelection {
        routes,
        backend: topk.backend,
    })
}

fn score_router_routes_bf16_device_with_backend(
    layer_id: usize,
    resident_router_weight_name: &str,
    hidden_device: &DeviceBf16Output,
    top_k: usize,
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    expert_count: usize,
) -> Result<RouterBatchRouteSelection> {
    if hidden_device.values_per_row == 0 {
        anyhow::bail!("real full router device-input hidden width is zero");
    }
    if top_k == 0 || top_k > expert_count {
        anyhow::bail!("real full router score invalid top_k={top_k} for experts={expert_count}");
    }
    if correction_bias.len() != expert_count {
        anyhow::bail!(
            "real full router score correction bias length mismatch: bias={} experts={expert_count}",
            correction_bias.len()
        );
    }

    let topk = router_topk_bf16_resident_weight_device_input(
        resident_router_weight_name,
        hidden_device,
        router_weight_bf16,
        correction_bias,
        expert_count,
        top_k,
    )?;
    row_routes_from_topk_batch(layer_id, top_k, correction_bias, topk)
}

fn score_router_routes_bf16_device_with_preloaded_backend(
    layer_id: usize,
    resident_router_weight_name: &str,
    resident_correction_bias_name: Option<&str>,
    hidden_device: &DeviceBf16Output,
    top_k: usize,
    correction_bias: &[f32],
    expert_count: usize,
) -> Result<RouterBatchRouteSelection> {
    if hidden_device.values_per_row == 0 {
        anyhow::bail!("real full router device-input hidden width is zero");
    }
    if top_k == 0 || top_k > expert_count {
        anyhow::bail!("real full router score invalid top_k={top_k} for experts={expert_count}");
    }
    if correction_bias.len() != expert_count {
        anyhow::bail!(
            "real full router score correction bias length mismatch: bias={} experts={expert_count}",
            correction_bias.len()
        );
    }

    let topk = if let Some(correction_bias_name) = resident_correction_bias_name {
        router_topk_bf16_preloaded_resident_weight_bias_device_input(
            resident_router_weight_name,
            correction_bias_name,
            hidden_device,
            expert_count,
            top_k,
        )?
    } else {
        router_topk_bf16_preloaded_resident_weight_device_input(
            resident_router_weight_name,
            hidden_device,
            correction_bias,
            expert_count,
            top_k,
        )?
    };
    row_routes_from_topk_batch(layer_id, top_k, correction_bias, topk)
}

fn score_router_routes_bf16_host_batch_with_backend(
    layer_id: usize,
    hidden_bf16: &[u8],
    rows: usize,
    hidden_width: usize,
    top_k: usize,
    tensors: &CachedRouterTensors,
) -> Result<RouterBatchRouteSelection> {
    let row_bytes = hidden_width
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| anyhow::anyhow!("real full router host fallback row byte width overflow"))?;
    let expected_hidden_bytes = rows
        .checked_mul(row_bytes)
        .ok_or_else(|| anyhow::anyhow!("real full router host fallback hidden bytes overflow"))?;
    if hidden_bf16.len() != expected_hidden_bytes {
        anyhow::bail!(
            "real full router host fallback hidden byte mismatch: got {} expected {expected_hidden_bytes}",
            hidden_bf16.len()
        );
    }

    let mut row_routes = Vec::with_capacity(rows);
    let mut backend = None;
    for row_index in 0..rows {
        let row_start = row_index
            .checked_mul(row_bytes)
            .ok_or_else(|| anyhow::anyhow!("real full router row start overflow"))?;
        let row_end = row_start
            .checked_add(row_bytes)
            .ok_or_else(|| anyhow::anyhow!("real full router row end overflow"))?;
        let scoring = score_real_router_routes_from_bf16_tensors(
            layer_id,
            &hidden_bf16[row_start..row_end],
            hidden_width,
            top_k,
            tensors,
        )?;
        if let Some(previous) = backend {
            if previous != scoring.router_backend {
                anyhow::bail!(
                    "real full router host fallback backend mismatch: {previous} vs {}",
                    scoring.router_backend
                );
            }
        } else {
            backend = Some(scoring.router_backend);
        }
        row_routes.push(scoring.routes);
    }
    Ok(RouterBatchRouteSelection {
        row_routes,
        backend: backend.unwrap_or("not-run"),
    })
}

fn row_routes_from_topk_batch(
    layer_id: usize,
    top_k: usize,
    correction_bias: &[f32],
    topk: crate::commands::real_full::coordinator_kernels::RouterTopKOutput,
) -> Result<RouterBatchRouteSelection> {
    if top_k == 0 {
        anyhow::bail!("real full router top-k batch splitter requires top_k > 0");
    }
    if topk.indices.len() != topk.scores.len() || topk.indices.len() != topk.weights.len() {
        anyhow::bail!(
            "real full router top-k batch length mismatch: indices={} scores={} weights={}",
            topk.indices.len(),
            topk.scores.len(),
            topk.weights.len()
        );
    }
    if topk.indices.len() % top_k != 0 {
        anyhow::bail!(
            "real full router top-k batch length {} is not divisible by top_k={top_k}",
            topk.indices.len()
        );
    }
    let rows = topk.indices.len() / top_k;
    let mut row_routes = Vec::with_capacity(rows);
    for row_index in 0..rows {
        let start = row_index
            .checked_mul(top_k)
            .ok_or_else(|| anyhow::anyhow!("real full router top-k row start overflow"))?;
        let end = start
            .checked_add(top_k)
            .ok_or_else(|| anyhow::anyhow!("real full router top-k row end overflow"))?;
        row_routes.push(routes_from_topk(
            layer_id,
            top_k,
            correction_bias,
            topk.indices[start..end].to_vec(),
            topk.scores[start..end].to_vec(),
            topk.weights[start..end].to_vec(),
        )?);
    }
    Ok(RouterBatchRouteSelection {
        row_routes,
        backend: topk.backend,
    })
}

fn routes_from_topk(
    layer_id: usize,
    top_k: usize,
    correction_bias: &[f32],
    indices: Vec<usize>,
    scores: Vec<f32>,
    weights: Vec<f32>,
) -> Result<Vec<ScoredRoute>> {
    if indices.len() != top_k || scores.len() != top_k || weights.len() != top_k {
        anyhow::bail!(
            "real full router top-k entry mismatch for layer {layer_id}: top_k={top_k} indices={} scores={} weights={}",
            indices.len(),
            scores.len(),
            weights.len()
        );
    }
    let mut routes = Vec::with_capacity(top_k);
    for rank in 0..top_k {
        let expert_id = indices[rank];
        let correction = correction_bias.get(expert_id).copied().with_context(|| {
            format!(
                "real full router returned invalid expert id {expert_id} for layer {layer_id} rank {rank}; experts={}",
                correction_bias.len()
            )
        })?;
        let score = scores[rank];
        if !score.is_finite() {
            anyhow::bail!(
                "real full router returned non-finite score for layer {layer_id} rank {rank} expert {expert_id}: {score}"
            );
        }
        let normalized_weight = weights[rank];
        if !normalized_weight.is_finite() {
            anyhow::bail!(
                "real full router returned non-finite normalized weight for layer {layer_id} rank {rank} expert {expert_id}: {normalized_weight}"
            );
        }
        let corrected_score = score + correction;
        if !corrected_score.is_finite() {
            anyhow::bail!(
                "real full router produced non-finite corrected score for layer {layer_id} rank {rank} expert {expert_id}: score={score} correction={correction}"
            );
        }
        routes.push(ScoredRoute {
            expert_id,
            score,
            corrected_score,
            normalized_weight,
        });
    }
    Ok(routes)
}

fn catalog_tensor<'a>(catalog: &'a TensorCatalog, name: &str) -> Result<&'a TensorInfo> {
    catalog
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| anyhow::anyhow!("tensor {name} not found in real full router catalog"))
}

fn bf16_matrix_byte_len(info: &TensorInfo) -> Result<usize> {
    if info.shape.len() != 2 {
        anyhow::bail!(
            "real full router expected rank-2 BF16 matrix {}, got shape {:?}",
            info.name,
            info.shape
        );
    }
    let bytes = info.shape[0]
        .checked_mul(info.shape[1])
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "real full router tensor {} byte length overflows usize",
                info.name
            )
        })?;
    if info.byte_length != bytes as u64 {
        anyhow::bail!(
            "real full router tensor {} catalog byte length mismatch: catalog={} expected={bytes}",
            info.name,
            info.byte_length
        );
    }
    Ok(bytes)
}

fn f32_vector_byte_len(info: &TensorInfo) -> Result<usize> {
    if info.shape.len() != 1 {
        anyhow::bail!(
            "real full router expected rank-1 F32 vector {}, got shape {:?}",
            info.name,
            info.shape
        );
    }
    info.shape[0]
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| anyhow::anyhow!("real full router F32 vector byte length overflow"))
}

#[cfg(test)]
mod tests {
    use super::{
        cache_router_correction_bias_host_values, load_router_tensors, routes_from_topk,
        score_real_router_routes_cached, score_router_routes,
        score_router_routes_bf16_with_backend, RouterTensorCache,
    };
    use crate::commands::real_full::coordinator_kernels::{
        coordinator_cuda_reference_kernels_enabled, cuda_reference_kernels_test_override,
        resident_weight_is_preloaded, CPU_REFERENCE_ROUTER_TOPK_BF16_BACKEND,
    };
    use crate::commands::real_full::dense::math::bf16_bytes_from_f32;
    use glmrt_core::{
        DType, ModelFacts, TensorCatalog, TensorInfo, TensorRole, GLM52_ROUTED_SCALING_FACTOR,
    };
    use std::{fs::File, io::Write};

    #[test]
    fn router_scores_use_correction_bias_and_normalized_weights() {
        let hidden = [1.0_f32, 0.0];
        let router_values = [
            0.0_f32, 0.0, //
            1.0, 0.0, //
            0.5, 0.0, //
        ];
        let correction_bias = [0.9_f32, 0.0, 0.2];

        let routes =
            score_router_routes(3, &hidden, 2, &router_values, &correction_bias, 3, 2).unwrap();

        assert_eq!(routes.len(), 2);
        assert_eq!(routes[0].expert_id, 0);
        assert_eq!(routes[1].expert_id, 2);
        assert!(
            (routes[0].normalized_weight + routes[1].normalized_weight
                - GLM52_ROUTED_SCALING_FACTOR)
                .abs()
                < 1.0e-6
        );
    }

    #[test]
    fn router_routes_reject_out_of_range_expert_id() {
        let err = match routes_from_topk(3, 1, &[0.0_f32; 256], vec![256], vec![1.0], vec![1.0]) {
            Ok(_) => panic!("out-of-range router expert id was accepted"),
            Err(err) => err,
        };

        assert!(err
            .to_string()
            .contains("invalid expert id 256 for layer 3 rank 0"));
    }

    #[test]
    fn router_routes_reject_non_finite_weight() {
        let err = match routes_from_topk(4, 1, &[0.0_f32; 256], vec![7], vec![1.0], vec![f32::NAN])
        {
            Ok(_) => panic!("non-finite router weight was accepted"),
            Err(err) => err,
        };

        assert!(err
            .to_string()
            .contains("non-finite normalized weight for layer 4 rank 0 expert 7"));
    }

    #[test]
    fn router_routes_reject_non_finite_score() {
        let err = match routes_from_topk(4, 1, &[0.0_f32; 256], vec![7], vec![f32::NAN], vec![1.0])
        {
            Ok(_) => panic!("non-finite router score was accepted"),
            Err(err) => err,
        };

        assert!(err
            .to_string()
            .contains("non-finite score for layer 4 rank 0 expert 7"));
    }

    #[test]
    fn router_scores_accept_precomputed_bf16_hidden() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);
        let hidden_bf16 = bf16_bytes_from_f32(&[1.0_f32, 0.0]);
        let router_values_bf16 = bf16_bytes_from_f32(&[
            0.0_f32, 0.0, //
            1.0, 0.0, //
            0.5, 0.0, //
        ]);
        let correction_bias = [0.9_f32, 0.0, 0.2];

        let selection = score_router_routes_bf16_with_backend(
            3,
            None,
            &hidden_bf16,
            2,
            2,
            &router_values_bf16,
            &correction_bias,
            3,
            2,
        )
        .unwrap();

        assert_eq!(selection.routes.len(), 2);
        assert_eq!(selection.routes[0].expert_id, 0);
        assert_eq!(selection.routes[1].expert_id, 2);
        assert_eq!(selection.backend, CPU_REFERENCE_ROUTER_TOPK_BF16_BACKEND);
    }

    #[test]
    fn router_tensor_cache_reuses_loaded_layer_tensors() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);
        let tempdir = tempfile::tempdir().unwrap();
        let shard_path = tempdir.path().join("router.bin");
        let mut bytes = Vec::new();
        for value in [0.0_f32, 0.0, 1.0, 0.0, 0.5, 0.0] {
            bytes.extend_from_slice(&bf16_bytes(value));
        }
        let bias_offset = bytes.len() as u64;
        for value in [0.9_f32, 0.0, 0.2] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        File::create(&shard_path)
            .unwrap()
            .write_all(&bytes)
            .unwrap();
        let catalog = TensorCatalog {
            model_id: "test/model".to_owned(),
            snapshot_path: tempdir.path().display().to_string(),
            facts: ModelFacts::default(),
            tensors: vec![
                TensorInfo {
                    name: "model.layers.3.mlp.gate.weight".to_owned(),
                    file: "router.bin".to_owned(),
                    dtype: DType::Bf16,
                    shape: vec![3, 2],
                    byte_offset: 0,
                    byte_length: bias_offset,
                    role: TensorRole::Other,
                    layer_id: Some(3),
                    expert_id: None,
                    is_quantization_metadata: false,
                },
                TensorInfo {
                    name: "model.layers.3.mlp.gate.e_score_correction_bias".to_owned(),
                    file: "router.bin".to_owned(),
                    dtype: DType::F32,
                    shape: vec![3],
                    byte_offset: bias_offset,
                    byte_length: 12,
                    role: TensorRole::Other,
                    layer_id: Some(3),
                    expert_id: None,
                    is_quantization_metadata: false,
                },
            ],
        };
        let mut cache = RouterTensorCache::default();

        let first =
            score_real_router_routes_cached(&catalog, 3, &[1.0_f32, 0.0], 2, &mut cache).unwrap();
        let second =
            score_real_router_routes_cached(&catalog, 3, &[1.0_f32, 0.0], 2, &mut cache).unwrap();

        assert_eq!(first.routes[0].expert_id, 0);
        assert_eq!(second.routes[0].expert_id, 0);
        assert_eq!(first.router_backend, CPU_REFERENCE_ROUTER_TOPK_BF16_BACKEND);
        assert_eq!(
            second.router_backend,
            CPU_REFERENCE_ROUTER_TOPK_BF16_BACKEND
        );
        assert_eq!(
            first.router_weight_bytes_read,
            second.router_weight_bytes_read
        );
        assert_eq!(first.router_bias_bytes_read, second.router_bias_bytes_read);
        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.tensor_loads, 2);
        assert_eq!(stats.cache_hits, 1);
    }

    #[test]
    fn router_cuda_fallback_preloads_weight_and_bias_from_pinned_staging() {
        if !coordinator_cuda_reference_kernels_enabled() {
            return;
        }
        let layer_id = 4_242_usize;
        let tempdir = tempfile::tempdir().unwrap();
        let shard_path = tempdir.path().join("router-staged.bin");
        let mut bytes = Vec::new();
        for value in [0.0_f32, 0.0, 1.0, 0.0, 0.5, 0.0] {
            bytes.extend_from_slice(&bf16_bytes(value));
        }
        let bias_offset = bytes.len() as u64;
        for value in [0.9_f32, 0.0, 0.2] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        File::create(&shard_path)
            .unwrap()
            .write_all(&bytes)
            .unwrap();
        let router_weight_name = format!("model.layers.{layer_id}.mlp.gate.weight");
        let router_bias_name = format!("model.layers.{layer_id}.mlp.gate.e_score_correction_bias");
        let catalog = TensorCatalog {
            model_id: "test/model".to_owned(),
            snapshot_path: tempdir.path().display().to_string(),
            facts: ModelFacts::default(),
            tensors: vec![
                TensorInfo {
                    name: router_weight_name.clone(),
                    file: "router-staged.bin".to_owned(),
                    dtype: DType::Bf16,
                    shape: vec![3, 2],
                    byte_offset: 0,
                    byte_length: bias_offset,
                    role: TensorRole::Router,
                    layer_id: Some(layer_id as u32),
                    expert_id: None,
                    is_quantization_metadata: false,
                },
                TensorInfo {
                    name: router_bias_name.clone(),
                    file: "router-staged.bin".to_owned(),
                    dtype: DType::F32,
                    shape: vec![3],
                    byte_offset: bias_offset,
                    byte_length: 12,
                    role: TensorRole::Router,
                    layer_id: Some(layer_id as u32),
                    expert_id: None,
                    is_quantization_metadata: false,
                },
            ],
        };

        let tensors = load_router_tensors(&catalog, layer_id).unwrap();

        assert!(tensors.router_weight_bf16_bytes.is_none());
        assert!(tensors.correction_bias_preloaded);
        assert_eq!(tensors.correction_bias.as_ref(), &[0.9_f32, 0.0, 0.2]);
        assert_eq!(tensors.tensor_loads, 2);
        assert!(resident_weight_is_preloaded(
            &router_weight_name,
            bias_offset as usize
        ));
        assert!(resident_weight_is_preloaded(&router_bias_name, 12));
    }

    #[test]
    fn router_correction_bias_host_cache_avoids_bias_tensor_reload() {
        let tempdir = tempfile::tempdir().unwrap();
        let shard_path = tempdir.path().join("router-cache.bin");
        let mut bytes = Vec::new();
        for value in [0.0_f32, 0.0, 1.0, 0.0, 0.5, 0.0] {
            bytes.extend_from_slice(&bf16_bytes(value));
        }
        let bias_offset = bytes.len() as u64;
        let mut bias_bytes = Vec::new();
        for value in [0.9_f32, 0.0, 0.2] {
            bias_bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&bias_bytes);
        File::create(&shard_path)
            .unwrap()
            .write_all(&bytes)
            .unwrap();
        let catalog = TensorCatalog {
            model_id: "test/model".to_owned(),
            snapshot_path: tempdir.path().display().to_string(),
            facts: ModelFacts::default(),
            tensors: vec![
                TensorInfo {
                    name: "model.layers.3.mlp.gate.weight".to_owned(),
                    file: "router-cache.bin".to_owned(),
                    dtype: DType::Bf16,
                    shape: vec![3, 2],
                    byte_offset: 0,
                    byte_length: bias_offset,
                    role: TensorRole::Router,
                    layer_id: Some(3),
                    expert_id: None,
                    is_quantization_metadata: false,
                },
                TensorInfo {
                    name: "model.layers.3.mlp.gate.e_score_correction_bias".to_owned(),
                    file: "router-cache.bin".to_owned(),
                    dtype: DType::F32,
                    shape: vec![3],
                    byte_offset: bias_offset,
                    byte_length: bias_bytes.len() as u64,
                    role: TensorRole::Router,
                    layer_id: Some(3),
                    expert_id: None,
                    is_quantization_metadata: false,
                },
            ],
        };

        let cached =
            cache_router_correction_bias_host_values(&catalog, &catalog.tensors[1], &bias_bytes)
                .unwrap();
        let tensors = load_router_tensors(&catalog, 3).unwrap();

        assert!(cached);
        assert_eq!(
            tensors.tensor_loads, 1,
            "only the non-resident router weight should be read after startup cached the bias"
        );
        assert_eq!(tensors.correction_bias.as_ref(), &[0.9_f32, 0.0, 0.2]);
    }

    fn bf16_bytes(value: f32) -> [u8; 2] {
        ((value.to_bits() >> 16) as u16).to_le_bytes()
    }
}
