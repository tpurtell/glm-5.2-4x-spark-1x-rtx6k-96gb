use anyhow::{Context, Result};
use glmrt_core::{
    DType, TensorCatalog, TensorInfo, GLM52_FIRST_K_DENSE_REPLACE, GLM52_HIDDEN_SIZE,
    GLM52_NUM_HIDDEN_LAYERS, GLM52_TOP_K, GLM52_TOTAL_LAYERS_WITH_MTP,
};
use glmrt_loader::{load_tensor_bytes, read_tensor_bytes_into};

use crate::commands::real_full::types::{
    RealFullExpertSparseMlpSharedChainLayerProbe, RealFullSparseMoeRoute,
};

use super::prefix::{execute_shared_expert_prefix, execute_shared_expert_prefix_with_device_input};
use super::{
    REAL_FULL_SHARED_CHAIN_OUTPUT_ROWS, REAL_FULL_SHARED_CHAIN_ROUTED_INTERMEDIATE_ROWS,
    REAL_FULL_SHARED_CHAIN_SHARED_INTERMEDIATE_ROWS, REAL_FULL_SHARED_CHAIN_TOP_K,
};
use crate::commands::real_full::coordinator_kernels::{
    coordinator_cuda_reference_kernels_enabled, device_bf16_output_from_f32_values,
    preload_resident_weight_from_host_staging, resident_weight_is_preloaded,
    residual_add_bf16_device_inputs_device_output, residual_add_bf16_device_inputs_output,
    residual_add_prefix_bf16_bytes_into, rmsnorm_hidden_bf16_preloaded_resident_weight,
    rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output,
    rmsnorm_hidden_bf16_resident_weight, DeviceBf16Output,
};
use crate::commands::real_full::dense::math::{
    bf16_bytes_from_f32, bf16_bytes_to_f32, fill_bf16_bytes_from_f32,
};
use crate::commands::real_full::dense::REAL_FULL_DENSE_RMSNORM_EPS;
use crate::commands::real_full::sparse_mlp::math::checksum_f64;
use crate::commands::real_full::sparse_mlp::route::execute_nvfp4_route;
use crate::commands::real_full::sparse_mlp::router::score_real_router_routes_bf16;

pub(super) struct RealSparseMlpSharedLayerExecution {
    pub(super) hidden_after_layer: Vec<f32>,
    pub(super) device_hidden_after_layer: Option<DeviceBf16Output>,
    pub(super) expert_input_hidden_bf16_payload: Vec<u8>,
    pub(super) layer_summary: RealFullExpertSparseMlpSharedChainLayerProbe,
    pub(super) routes: Vec<RealFullSparseMoeRoute>,
    pub(super) routed_outputs: Vec<f32>,
    pub(super) shared_outputs: Vec<f32>,
    pub(super) layer_outputs: Vec<f32>,
    pub(super) routes_executed: usize,
    pub(super) shared_expert_executed: bool,
    pub(super) routed_intermediate_rows: usize,
    pub(super) shared_intermediate_rows: usize,
    pub(super) output_rows: usize,
    pub(super) expert_input_norm_backend: &'static str,
    pub(super) router_backend: &'static str,
    pub(super) shared_mlp_backend: &'static str,
    pub(super) final_residual_checksum: f64,
    pub(super) residual_add_backend: &'static str,
    pub(super) covers_full_top_k: bool,
    pub(super) passed: bool,
}

#[derive(Default)]
struct SparseResidualAddWorkspace {
    residual_bf16: Vec<u8>,
    delta_bf16: Vec<u8>,
    output_bf16: Vec<u8>,
}

struct SparseResidualAddResult {
    values: Vec<f32>,
    backend: &'static str,
    checksum: f64,
    first: f32,
    last: f32,
}

struct SparsePostAttentionNormOutput {
    values: Vec<f32>,
    bf16_payload: Vec<u8>,
    device_hidden: Option<DeviceBf16Output>,
    backend: &'static str,
}

#[derive(Clone, Copy)]
pub(super) struct SparseMlpSharedChainExecutionPlan {
    pub(super) routed_intermediate_rows: usize,
    pub(super) shared_intermediate_rows: usize,
    pub(super) output_rows: usize,
}

pub(super) fn bounded_sparse_mlp_shared_chain_execution_plan() -> SparseMlpSharedChainExecutionPlan
{
    SparseMlpSharedChainExecutionPlan {
        routed_intermediate_rows: REAL_FULL_SHARED_CHAIN_ROUTED_INTERMEDIATE_ROWS,
        shared_intermediate_rows: REAL_FULL_SHARED_CHAIN_SHARED_INTERMEDIATE_ROWS,
        output_rows: REAL_FULL_SHARED_CHAIN_OUTPUT_ROWS,
    }
}

pub(super) fn execute_real_sparse_mlp_shared_layer_from_hidden_with_plan(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden: Vec<f32>,
    plan: SparseMlpSharedChainExecutionPlan,
) -> Result<RealSparseMlpSharedLayerExecution> {
    let mut residual_workspace = SparseResidualAddWorkspace::default();
    execute_real_sparse_mlp_shared_layer_from_hidden_with_plan_and_workspace(
        catalog,
        layer_id,
        hidden,
        None,
        plan,
        &mut residual_workspace,
    )
}

pub(super) fn execute_real_sparse_mlp_shared_layer_from_hidden_with_plan_and_device_input(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden: Vec<f32>,
    device_hidden: &DeviceBf16Output,
    plan: SparseMlpSharedChainExecutionPlan,
) -> Result<RealSparseMlpSharedLayerExecution> {
    let mut residual_workspace = SparseResidualAddWorkspace::default();
    execute_real_sparse_mlp_shared_layer_from_hidden_with_plan_and_workspace(
        catalog,
        layer_id,
        hidden,
        Some(device_hidden),
        plan,
        &mut residual_workspace,
    )
}

fn execute_real_sparse_mlp_shared_layer_from_hidden_with_plan_and_workspace(
    catalog: &TensorCatalog,
    layer_id: usize,
    mut hidden: Vec<f32>,
    device_hidden: Option<&DeviceBf16Output>,
    plan: SparseMlpSharedChainExecutionPlan,
    residual_workspace: &mut SparseResidualAddWorkspace,
) -> Result<RealSparseMlpSharedLayerExecution> {
    validate_sparse_mlp_shared_inputs("sparse MLP shared-layer", &hidden, plan)?;
    if !(GLM52_FIRST_K_DENSE_REPLACE..GLM52_TOTAL_LAYERS_WITH_MTP).contains(&layer_id) {
        anyhow::bail!(
            "sparse MLP shared-layer probe expected sparse layer id in {}..{}, got {layer_id}",
            GLM52_FIRST_K_DENSE_REPLACE,
            GLM52_TOTAL_LAYERS_WITH_MTP
        );
    }
    if let Some(device_hidden) = device_hidden {
        if plan.output_rows != GLM52_HIDDEN_SIZE {
            anyhow::bail!(
                "sparse MLP shared-layer device-hidden residual execution requires full-output rows, got {}",
                plan.output_rows
            );
        }
        if device_hidden.rows != 1 || device_hidden.values_per_row != hidden.len() {
            anyhow::bail!(
                "sparse MLP shared-layer device-hidden shape mismatch: expected 1x{} got {}x{}",
                hidden.len(),
                device_hidden.rows,
                device_hidden.values_per_row
            );
        }
    }
    let expert_hidden = post_attention_norm_hidden(catalog, layer_id, &hidden, device_hidden)?;
    let expert_input_norm_backend = expert_hidden.backend;
    let expert_input_hidden_bf16_payload = expert_hidden.bf16_payload;
    let expert_hidden_device = expert_hidden.device_hidden;
    let expert_hidden = expert_hidden.values;

    let scoring = score_real_router_routes_bf16(
        catalog,
        layer_id,
        &expert_input_hidden_bf16_payload,
        expert_hidden.len(),
        REAL_FULL_SHARED_CHAIN_TOP_K,
    )?;
    let top_route = scoring.routes.first().cloned().with_context(|| {
        format!("sparse MLP shared-layer probe selected no route for layer {layer_id}")
    })?;
    let residual_before = hidden[..plan.output_rows].to_vec();
    let residual_before_checksum = checksum_f64(&residual_before);
    let mut routed_outputs = vec![0.0_f32; plan.output_rows];
    let mut routed_weight_bytes_read = 0_u64;
    let mut routed_quant_metadata_bytes_read = 0_u64;
    let mut routes_executed = 0_usize;

    for route in &scoring.routes {
        let execution = execute_nvfp4_route(
            catalog,
            layer_id,
            &expert_hidden,
            route,
            plan.routed_intermediate_rows,
            plan.output_rows,
        )?;
        for (reduced, output) in routed_outputs.iter_mut().zip(&execution.outputs) {
            *reduced += *output;
        }
        routed_weight_bytes_read += execution.weight_bytes_read;
        routed_quant_metadata_bytes_read += execution.quant_metadata_bytes_read;
        routes_executed += 1;
    }

    let shared = if expert_hidden_device.is_some() {
        execute_shared_expert_prefix_with_device_input(
            catalog,
            layer_id,
            &expert_hidden,
            expert_hidden_device.as_ref(),
            plan.shared_intermediate_rows,
            plan.output_rows,
        )?
    } else {
        execute_shared_expert_prefix(
            catalog,
            layer_id,
            &expert_hidden,
            plan.shared_intermediate_rows,
            plan.output_rows,
        )?
    };
    let shared_weight_bytes_read =
        shared.gate_proj_bytes_read + shared.up_proj_bytes_read + shared.down_proj_bytes_read;

    let mut layer_outputs = Vec::with_capacity(plan.output_rows);
    for (routed, shared) in routed_outputs.iter().zip(&shared.outputs) {
        layer_outputs.push(routed + shared);
    }

    let (
        residual_values,
        residual_add_backend,
        first_residual_after,
        last_residual_after,
        residual_after_checksum,
        device_hidden_after_layer,
    ) = if let Some(device_hidden) = device_hidden {
        let layer_output_device = if let Some(shared_output_device) = shared.output_device.as_ref()
        {
            let routed_output_device = device_bf16_output_from_f32_values(
                &routed_outputs,
                1,
                plan.output_rows,
                "sparse MLP routed delta device upload",
            )?;
            residual_add_bf16_device_inputs_device_output(
                shared_output_device,
                &routed_output_device,
            )?
        } else {
            device_bf16_output_from_f32_values(
                &layer_outputs,
                1,
                plan.output_rows,
                "sparse MLP routed plus shared delta device upload",
            )?
        };
        let residual_after =
            residual_add_bf16_device_inputs_output(device_hidden, &layer_output_device)?;
        let first = residual_after.values.first().copied().unwrap_or_default();
        let last = residual_after.values.last().copied().unwrap_or_default();
        let checksum = checksum_f64(&residual_after.values);
        (
            residual_after.values,
            residual_after.backend,
            first,
            last,
            checksum,
            Some(residual_after.device_output),
        )
    } else {
        let residual_after =
            sparse_residual_add_bf16(&residual_before, &layer_outputs, residual_workspace)?;
        (
            residual_after.values,
            residual_after.backend,
            residual_after.first,
            residual_after.last,
            residual_after.checksum,
            None,
        )
    };
    hidden[..plan.output_rows].copy_from_slice(&residual_values);
    let routed_output_checksum = checksum_f64(&routed_outputs);
    let layer_output_checksum = checksum_f64(&layer_outputs);
    let output_l2_sum = layer_outputs.iter().map(|value| value * value).sum::<f32>();
    let layer_output_l2_norm = output_l2_sum.sqrt();
    let covers_full_top_k = routes_executed == GLM52_TOP_K;
    let routes = scoring
        .routes
        .iter()
        .enumerate()
        .map(|(rank, route)| RealFullSparseMoeRoute {
            rank,
            expert_id: route.expert_id,
            owner: route.owner.clone(),
            score: route.score,
            corrected_score: route.corrected_score,
            normalized_weight: route.normalized_weight,
        })
        .collect::<Vec<_>>();
    let passed = covers_full_top_k
        && scoring.router_weight_bytes_read > 0
        && scoring.router_bias_bytes_read > 0
        && routed_weight_bytes_read > 0
        && routed_quant_metadata_bytes_read > 0
        && shared_weight_bytes_read > 0
        && layer_output_checksum.is_finite()
        && layer_output_l2_norm.is_finite()
        && residual_before_checksum.is_finite()
        && residual_after_checksum.is_finite();

    Ok(RealSparseMlpSharedLayerExecution {
        hidden_after_layer: hidden,
        device_hidden_after_layer,
        expert_input_hidden_bf16_payload,
        layer_summary: RealFullExpertSparseMlpSharedChainLayerProbe {
            layer_id,
            expert_id: top_route.expert_id,
            owner: top_route.owner,
            score: top_route.score,
            corrected_score: top_route.corrected_score,
            routed_output_checksum,
            shared_output_checksum: shared.output_checksum,
            output_checksum: layer_output_checksum,
            output_l2_norm: layer_output_l2_norm,
            residual_before_checksum,
            residual_delta_checksum: layer_output_checksum,
            residual_after_checksum,
            expert_input_norm_backend,
            router_backend: scoring.router_backend,
            shared_mlp_backend: shared.mlp_backend,
            residual_add_backend,
            first_residual_after,
            last_residual_after,
        },
        routes,
        routed_outputs,
        shared_outputs: shared.outputs,
        layer_outputs,
        routes_executed,
        shared_expert_executed: true,
        routed_intermediate_rows: plan.routed_intermediate_rows,
        shared_intermediate_rows: plan.shared_intermediate_rows,
        output_rows: plan.output_rows,
        expert_input_norm_backend,
        router_backend: scoring.router_backend,
        shared_mlp_backend: shared.mlp_backend,
        final_residual_checksum: residual_after_checksum,
        residual_add_backend,
        covers_full_top_k,
        passed,
    })
}

fn sparse_residual_add_bf16(
    residual: &[f32],
    delta: &[f32],
    workspace: &mut SparseResidualAddWorkspace,
) -> Result<SparseResidualAddResult> {
    if residual.len() != delta.len() {
        anyhow::bail!(
            "real full sparse residual-add length mismatch: residual={} delta={}",
            residual.len(),
            delta.len()
        );
    }
    fill_bf16_bytes_from_f32(residual, &mut workspace.residual_bf16);
    fill_bf16_bytes_from_f32(delta, &mut workspace.delta_bf16);
    workspace
        .output_bf16
        .resize(workspace.residual_bf16.len(), 0);
    let backend = residual_add_prefix_bf16_bytes_into(
        &workspace.residual_bf16,
        &workspace.delta_bf16,
        &mut workspace.output_bf16,
    )?;
    let values = bf16_bytes_to_f32(&workspace.output_bf16)?;
    let checksum = checksum_f64(&values);
    Ok(SparseResidualAddResult {
        first: values.first().copied().unwrap_or_default(),
        last: values.last().copied().unwrap_or_default(),
        values,
        backend,
        checksum,
    })
}

fn validate_sparse_mlp_shared_inputs(
    context: &str,
    hidden: &[f32],
    plan: SparseMlpSharedChainExecutionPlan,
) -> Result<()> {
    if hidden.len() != GLM52_HIDDEN_SIZE {
        anyhow::bail!(
            "{context} initial hidden width mismatch: expected {} got {}",
            GLM52_HIDDEN_SIZE,
            hidden.len()
        );
    }
    if plan.routed_intermediate_rows == 0 || plan.shared_intermediate_rows == 0 {
        anyhow::bail!("{context} expected nonzero routed/shared intermediate rows");
    }
    if plan.output_rows == 0 || plan.output_rows > GLM52_HIDDEN_SIZE {
        anyhow::bail!(
            "{context} invalid output_rows={} for hidden size {GLM52_HIDDEN_SIZE}",
            plan.output_rows
        );
    }
    Ok(())
}

fn post_attention_norm_hidden(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden: &[f32],
    device_hidden: Option<&DeviceBf16Output>,
) -> Result<SparsePostAttentionNormOutput> {
    let norm_name = format!("model.layers.{layer_id}.post_attention_layernorm.weight");
    let norm_info = catalog_tensor(catalog, &norm_name)?;
    validate_bf16_vector_shape(
        norm_info,
        "sparse MLP shared-layer post-attention layernorm",
    )?;
    if norm_info.shape != vec![GLM52_HIDDEN_SIZE] {
        anyhow::bail!(
            "sparse MLP shared-layer post-attention layernorm shape mismatch for layer {layer_id}: {:?}",
            norm_info.shape
        );
    }
    let hidden_bf16 = bf16_bytes_from_f32(hidden);
    let norm_bytes = GLM52_HIDDEN_SIZE * std::mem::size_of::<u16>();
    let cuda_norm_resident = if coordinator_cuda_reference_kernels_enabled() {
        ensure_sparse_post_attention_norm_resident_from_host_staging(
            catalog, &norm_name, norm_info, norm_bytes,
        )?;
        true
    } else {
        false
    };
    let normalized = if let (Some(device_hidden), true) = (device_hidden, cuda_norm_resident) {
        let normalized_device = rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output(
            &norm_name,
            device_hidden.buffer(),
            1,
            GLM52_HIDDEN_SIZE,
            REAL_FULL_DENSE_RMSNORM_EPS,
        )?;
        let bf16_payload = normalized_device.copy_to_host_bytes()?;
        let values = bf16_bytes_to_f32(&bf16_payload)?;
        SparsePostAttentionNormOutput {
            values,
            bf16_payload,
            backend: normalized_device.backend,
            device_hidden: Some(normalized_device),
        }
    } else if cuda_norm_resident {
        let normalized = rmsnorm_hidden_bf16_preloaded_resident_weight(
            &norm_name,
            &hidden_bf16,
            1,
            GLM52_HIDDEN_SIZE,
            REAL_FULL_DENSE_RMSNORM_EPS,
        )?;
        let bf16_payload = bf16_bytes_from_f32(&normalized.values);
        SparsePostAttentionNormOutput {
            values: normalized.values,
            bf16_payload,
            backend: normalized.backend,
            device_hidden: None,
        }
    } else {
        let norm = load_tensor_bytes(catalog, &norm_name)?;
        let normalized = rmsnorm_hidden_bf16_resident_weight(
            &norm_name,
            &hidden_bf16,
            &norm.bytes,
            1,
            GLM52_HIDDEN_SIZE,
            REAL_FULL_DENSE_RMSNORM_EPS,
        )?;
        let bf16_payload = bf16_bytes_from_f32(&normalized.values);
        SparsePostAttentionNormOutput {
            values: normalized.values,
            bf16_payload,
            backend: normalized.backend,
            device_hidden: None,
        }
    };
    if !normalized.values.iter().all(|value| value.is_finite()) {
        anyhow::bail!(
            "sparse MLP shared-layer post-attention RMSNorm produced non-finite hidden for layer {layer_id}"
        );
    }
    Ok(normalized)
}

fn ensure_sparse_post_attention_norm_resident_from_host_staging(
    catalog: &TensorCatalog,
    norm_name: &str,
    norm_info: &TensorInfo,
    expected_bytes: usize,
) -> Result<()> {
    if resident_weight_is_preloaded(norm_name, expected_bytes) {
        return Ok(());
    }
    preload_resident_weight_from_host_staging(
        norm_name,
        expected_bytes,
        "sparse shared post-attention norm pinned staging",
        |staging| {
            let summary =
                read_tensor_bytes_into(catalog, norm_name, staging).with_context(|| {
                    format!("reading sparse shared norm tensor {norm_name} into pinned staging")
                })?;
            if summary.dtype != DType::Bf16 {
                anyhow::bail!(
                    "sparse shared norm tensor {norm_name} expects BF16, got {:?}",
                    summary.dtype
                );
            }
            if summary.shape != norm_info.shape {
                anyhow::bail!(
                    "sparse shared norm tensor {norm_name} shape mismatch: expected {:?} got {:?}",
                    norm_info.shape,
                    summary.shape
                );
            }
            if summary.bytes_read as usize != expected_bytes {
                anyhow::bail!(
                    "sparse shared norm tensor {norm_name} read {} bytes, expected {}",
                    summary.bytes_read,
                    expected_bytes
                );
            }
            Ok(())
        },
    )
    .with_context(|| {
        format!("preloading sparse shared norm tensor {norm_name} from pinned staging")
    })
}

fn catalog_tensor<'a>(catalog: &'a TensorCatalog, name: &str) -> Result<&'a TensorInfo> {
    catalog
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| anyhow::anyhow!("tensor {name} not found in sparse MLP shared catalog"))
}

fn validate_bf16_vector_shape(info: &TensorInfo, context: &str) -> Result<()> {
    if info.dtype != DType::Bf16 {
        anyhow::bail!(
            "{context} expected BF16 tensor {}, got {:?}",
            info.name,
            info.dtype
        );
    }
    if info.shape.len() != 1 || info.shape[0] == 0 {
        anyhow::bail!(
            "{context} expected non-empty rank-1 tensor {}, got shape {:?}",
            info.name,
            info.shape
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use glmrt_core::{ModelFacts, TensorRole};
    use std::{fs::File, io::Write};

    #[test]
    fn sparse_post_attention_norm_cuda_fallback_preloads_from_pinned_staging() -> Result<()> {
        if !coordinator_cuda_reference_kernels_enabled() {
            return Ok(());
        }

        let tempdir = tempfile::tempdir()?;
        let shard_name = "sparse-shared-norm.bin";
        let norm_name = "test.sparse.shared.post_attention_layernorm.weight";
        let weights = (0..GLM52_HIDDEN_SIZE)
            .map(|index| 1.0_f32 + ((index % 17) as f32) * 0.001)
            .collect::<Vec<_>>();
        let bytes = bf16_bytes_from_f32(&weights);
        File::create(tempdir.path().join(shard_name))?.write_all(&bytes)?;
        let info = TensorInfo {
            name: norm_name.to_owned(),
            file: shard_name.to_owned(),
            dtype: DType::Bf16,
            shape: vec![GLM52_HIDDEN_SIZE],
            byte_offset: 0,
            byte_length: bytes.len() as u64,
            role: TensorRole::Norm,
            layer_id: Some(GLM52_FIRST_K_DENSE_REPLACE as u32),
            expert_id: None,
            is_quantization_metadata: false,
        };
        let catalog = TensorCatalog {
            model_id: "test/model".to_owned(),
            snapshot_path: tempdir.path().display().to_string(),
            facts: ModelFacts::default(),
            tensors: vec![info.clone()],
        };

        ensure_sparse_post_attention_norm_resident_from_host_staging(
            &catalog,
            norm_name,
            &info,
            bytes.len(),
        )?;

        assert!(resident_weight_is_preloaded(norm_name, bytes.len()));
        Ok(())
    }
}
