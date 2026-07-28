use super::*;
use crate::python_graph_capture::coordinator_python_capture_enabled;
use anyhow::{Context, Result};
use glmrt_core::{
    CoordinatorGraphInstancePlan, CoordinatorGraphKey, CoordinatorGraphShape, LayerId,
    LayerWaveMode, COORDINATOR_GRAPH_INSTANCE_COUNT, GLM52_FIRST_K_DENSE_REPLACE,
    GLM52_HIDDEN_BF16_BYTES, GLM52_HIDDEN_SIZE, GLM52_NUM_HIDDEN_LAYERS,
    GLM52_ROUTED_SCALING_FACTOR, GLM52_TOP_K,
};
use glmrt_ffi::{
    GlmrtCudaGraphCaptureInfo, GlmrtDeviceBuffer, GlmrtHostBuffer, NativeLibrary,
    GLMRT_CUDA_ROUTER_TOPK_MAX_K, GLMRT_CUDA_SAMPLE_TOPK_MAX_K,
};
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::slice;
use std::sync::Mutex;

pub(in crate::commands::real_full) const CPU_REFERENCE_ROUTER_TOPK_BACKEND: &str =
    "cpu-reference-router-topk";
#[allow(dead_code)]
pub(in crate::commands::real_full) const CUDA_REFERENCE_ROUTER_TOPK_BACKEND: &str =
    "cuda-reference-router-topk-f32";
pub(in crate::commands::real_full) const CPU_REFERENCE_ROUTER_TOPK_BF16_BACKEND: &str =
    "cpu-reference-router-topk-bf16";
pub(in crate::commands::real_full) const CUDA_REFERENCE_ROUTER_TOPK_BF16_BACKEND: &str =
    "cuda-reference-router-topk-bf16";
pub(in crate::commands::real_full) const CUDA_REFERENCE_ROUTER_TOPK_BF16_RESIDENT_WEIGHT_BACKEND:
    &str = "cuda-reference-router-topk-bf16-resident-weight";
pub(in crate::commands::real_full) const CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND:
    &str = "cuda-reference-router-topk-bf16-preloaded-resident-weight";
pub(in crate::commands::real_full) const CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_BACKEND:
    &str = "cuda-reference-router-topk-bf16-preloaded-resident-weight-bias";
pub(in crate::commands::real_full) const CUDA_REFERENCE_ROUTER_TOPK_BF16_RESIDENT_WEIGHT_DEVICE_INPUT_BACKEND:
    &str = "cuda-reference-router-topk-bf16-resident-weight-device-input";
pub(in crate::commands::real_full) const CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_DEVICE_INPUT_BACKEND:
    &str = "cuda-reference-router-topk-bf16-preloaded-resident-weight-device-input";
pub(in crate::commands::real_full) const CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_DEVICE_INPUT_BACKEND:
    &str = "cuda-reference-router-topk-bf16-preloaded-resident-weight-bias-device-input";
pub(in crate::commands::real_full) const TRITON_ROUTER_TOPK_BF16_RESIDENT_WEIGHT_BACKEND: &str =
    "triton-router-topk-bf16-resident-weight";
pub(in crate::commands::real_full) const TRITON_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND:
    &str = "triton-router-topk-bf16-preloaded-resident-weight";
pub(in crate::commands::real_full) const TRITON_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_BACKEND:
    &str = "triton-router-topk-bf16-preloaded-resident-weight-bias";
pub(in crate::commands::real_full) const TRITON_ROUTER_TOPK_BF16_RESIDENT_WEIGHT_DEVICE_INPUT_BACKEND:
    &str = "triton-router-topk-bf16-resident-weight-device-input";
pub(in crate::commands::real_full) const TRITON_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_DEVICE_INPUT_BACKEND:
    &str = "triton-router-topk-bf16-preloaded-resident-weight-device-input";
pub(in crate::commands::real_full) const TRITON_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_DEVICE_INPUT_BACKEND:
    &str = "triton-router-topk-bf16-preloaded-resident-weight-bias-device-input";

#[allow(dead_code)]
pub(in crate::commands::real_full) fn router_topk(
    hidden: &[f32],
    router_weight: &[f32],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    validate_router_topk_inputs(
        hidden,
        router_weight,
        correction_bias,
        rows,
        hidden_dim,
        experts,
        top_k,
    )?;
    if cuda_reference_kernels_enabled() {
        return cuda_router_topk(
            hidden,
            router_weight,
            correction_bias,
            rows,
            hidden_dim,
            experts,
            top_k,
        );
    }
    Ok(cpu_router_topk(
        hidden,
        router_weight,
        correction_bias,
        rows,
        hidden_dim,
        experts,
        top_k,
    ))
}

pub(in crate::commands::real_full) fn router_topk_bf16(
    hidden_bf16: &[u8],
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    validate_router_topk_bf16_inputs(
        hidden_bf16,
        router_weight_bf16,
        correction_bias,
        rows,
        hidden_dim,
        experts,
        top_k,
    )?;
    if cuda_reference_kernels_enabled() {
        return cuda_router_topk_bf16(
            hidden_bf16,
            router_weight_bf16,
            correction_bias,
            rows,
            hidden_dim,
            experts,
            top_k,
        );
    }
    Ok(cpu_router_topk_bf16(
        hidden_bf16,
        router_weight_bf16,
        correction_bias,
        rows,
        hidden_dim,
        experts,
        top_k,
    ))
}

pub(in crate::commands::real_full) fn router_topk_bf16_resident_weight(
    router_weight_name: &str,
    hidden_bf16: &[u8],
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    validate_resident_weight_name(router_weight_name)?;
    validate_router_topk_bf16_inputs(
        hidden_bf16,
        router_weight_bf16,
        correction_bias,
        rows,
        hidden_dim,
        experts,
        top_k,
    )?;
    if cuda_reference_kernels_enabled() {
        return cuda_router_topk_bf16_resident_weight(
            router_weight_name,
            hidden_bf16,
            router_weight_bf16,
            correction_bias,
            rows,
            hidden_dim,
            experts,
            top_k,
        );
    }
    Ok(cpu_router_topk_bf16(
        hidden_bf16,
        router_weight_bf16,
        correction_bias,
        rows,
        hidden_dim,
        experts,
        top_k,
    ))
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn router_topk_bf16_preloaded_resident_weight(
    router_weight_name: &str,
    hidden_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    validate_resident_weight_name(router_weight_name)?;
    let weight_bytes = validate_router_topk_bf16_preloaded_resident_inputs(
        hidden_bf16,
        correction_bias,
        rows,
        hidden_dim,
        experts,
        top_k,
    )?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!("preloaded resident BF16 router top-k requires CUDA reference kernels");
    }
    cuda_router_topk_bf16_preloaded_resident_weight(
        router_weight_name,
        hidden_bf16,
        correction_bias,
        rows,
        hidden_dim,
        experts,
        top_k,
        weight_bytes,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn router_topk_bf16_preloaded_resident_weight_bias(
    router_weight_name: &str,
    correction_bias_name: &str,
    hidden_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    validate_resident_weight_name(router_weight_name)?;
    validate_resident_weight_name(correction_bias_name)?;
    let weight_bytes = validate_router_topk_bf16_preloaded_resident_inputs(
        hidden_bf16,
        correction_bias,
        rows,
        hidden_dim,
        experts,
        top_k,
    )?;
    let bias_bytes = correction_bias
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .context(
            "real full preloaded BF16 router top-k correction bias byte count overflows usize",
        )?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "preloaded resident BF16 router top-k weight+bias requires CUDA reference kernels"
        );
    }
    cuda_router_topk_bf16_preloaded_resident_weight_bias(
        router_weight_name,
        correction_bias_name,
        hidden_bf16,
        rows,
        hidden_dim,
        experts,
        top_k,
        weight_bytes,
        bias_bytes,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn router_topk_bf16_resident_weight_device_input(
    router_weight_name: &str,
    hidden: &DeviceBf16Output,
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    validate_resident_weight_name(router_weight_name)?;
    validate_router_topk_bf16_resident_device_input(
        hidden,
        router_weight_bf16,
        correction_bias,
        experts,
        top_k,
    )?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!("resident BF16 router top-k device-input requires CUDA reference kernels");
    }
    cuda_router_topk_bf16_resident_weight_device_input(
        router_weight_name,
        hidden,
        router_weight_bf16,
        correction_bias,
        experts,
        top_k,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn router_topk_bf16_preloaded_resident_weight_device_input(
    router_weight_name: &str,
    hidden: &DeviceBf16Output,
    correction_bias: &[f32],
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    validate_resident_weight_name(router_weight_name)?;
    let weight_bytes = validate_router_topk_bf16_preloaded_resident_device_input(
        hidden,
        Some(correction_bias),
        experts,
        top_k,
    )?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "preloaded resident BF16 router top-k device-input requires CUDA reference kernels"
        );
    }
    cuda_router_topk_bf16_preloaded_resident_weight_device_input(
        router_weight_name,
        hidden,
        correction_bias,
        experts,
        top_k,
        weight_bytes,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn router_topk_bf16_preloaded_resident_weight_bias_device_input(
    router_weight_name: &str,
    correction_bias_name: &str,
    hidden: &DeviceBf16Output,
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    validate_resident_weight_name(router_weight_name)?;
    validate_resident_weight_name(correction_bias_name)?;
    let weight_bytes =
        validate_router_topk_bf16_preloaded_resident_device_input(hidden, None, experts, top_k)?;
    let bias_bytes = experts
        .checked_mul(std::mem::size_of::<f32>())
        .context(
            "real full preloaded BF16 router top-k device-input correction bias byte count overflows usize",
        )?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "preloaded resident BF16 router top-k weight+bias device-input requires CUDA reference kernels"
        );
    }
    cuda_router_topk_bf16_preloaded_resident_weight_bias_device_input(
        router_weight_name,
        correction_bias_name,
        hidden,
        experts,
        top_k,
        weight_bytes,
        bias_bytes,
    )
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn validate_router_topk_inputs(
    hidden: &[f32],
    router_weight: &[f32],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<()> {
    if rows == 0 || hidden_dim == 0 || experts == 0 {
        anyhow::bail!(
            "real full router top-k requires non-zero shape, got rows={rows} hidden_dim={hidden_dim} experts={experts}"
        );
    }
    if top_k == 0 || top_k > experts || top_k > GLMRT_CUDA_ROUTER_TOPK_MAX_K {
        anyhow::bail!(
            "real full router top-k invalid top_k={top_k} for experts={experts}; max supported top_k={GLMRT_CUDA_ROUTER_TOPK_MAX_K}"
        );
    }
    let expected_hidden = rows.checked_mul(hidden_dim).context(
        "real full router top-k hidden shape overflows usize while validating coordinator kernel input",
    )?;
    if hidden.len() != expected_hidden {
        anyhow::bail!(
            "real full router top-k hidden length mismatch: expected {} got {}",
            expected_hidden,
            hidden.len()
        );
    }
    let expected_weight = experts.checked_mul(hidden_dim).context(
        "real full router top-k weight shape overflows usize while validating coordinator kernel input",
    )?;
    if router_weight.len() != expected_weight {
        anyhow::bail!(
            "real full router top-k weight length mismatch: expected {} got {}",
            expected_weight,
            router_weight.len()
        );
    }
    if correction_bias.len() != experts {
        anyhow::bail!(
            "real full router top-k correction bias length mismatch: expected {} got {}",
            experts,
            correction_bias.len()
        );
    }
    Ok(())
}

pub(in crate::commands::real_full) fn validate_router_topk_bf16_inputs(
    hidden_bf16: &[u8],
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<()> {
    if rows == 0 || hidden_dim == 0 || experts == 0 {
        anyhow::bail!(
            "real full BF16 router top-k requires non-zero shape, got rows={rows} hidden_dim={hidden_dim} experts={experts}"
        );
    }
    if top_k == 0 || top_k > experts || top_k > GLMRT_CUDA_ROUTER_TOPK_MAX_K {
        anyhow::bail!(
            "real full BF16 router top-k invalid top_k={top_k} for experts={experts}; max supported top_k={GLMRT_CUDA_ROUTER_TOPK_MAX_K}"
        );
    }
    let expected_hidden = rows
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full BF16 router top-k hidden shape overflows usize while validating input",
        )?;
    if hidden_bf16.len() != expected_hidden {
        anyhow::bail!(
            "real full BF16 router top-k hidden byte length mismatch: expected {} got {}",
            expected_hidden,
            hidden_bf16.len()
        );
    }
    let expected_weight = experts
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full BF16 router top-k weight shape overflows usize while validating input",
        )?;
    if router_weight_bf16.len() != expected_weight {
        anyhow::bail!(
            "real full BF16 router top-k weight byte length mismatch: expected {} got {}",
            expected_weight,
            router_weight_bf16.len()
        );
    }
    if correction_bias.len() != experts {
        anyhow::bail!(
            "real full BF16 router top-k correction bias length mismatch: expected {} got {}",
            experts,
            correction_bias.len()
        );
    }
    Ok(())
}

pub(in crate::commands::real_full) fn validate_router_topk_bf16_preloaded_resident_inputs(
    hidden_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<usize> {
    if rows == 0 || hidden_dim == 0 || experts == 0 {
        anyhow::bail!(
            "real full preloaded BF16 router top-k requires non-zero shape, got rows={rows} hidden_dim={hidden_dim} experts={experts}"
        );
    }
    if top_k == 0 || top_k > experts || top_k > GLMRT_CUDA_ROUTER_TOPK_MAX_K {
        anyhow::bail!(
            "real full preloaded BF16 router top-k invalid top_k={top_k} for experts={experts}; max supported top_k={GLMRT_CUDA_ROUTER_TOPK_MAX_K}"
        );
    }
    let expected_hidden = rows
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full preloaded BF16 router top-k hidden shape overflows usize while validating input",
        )?;
    if hidden_bf16.len() != expected_hidden {
        anyhow::bail!(
            "real full preloaded BF16 router top-k hidden byte length mismatch: expected {} got {}",
            expected_hidden,
            hidden_bf16.len()
        );
    }
    let expected_weight = experts
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full preloaded BF16 router top-k weight shape overflows usize while validating input",
        )?;
    if correction_bias.len() != experts {
        anyhow::bail!(
            "real full preloaded BF16 router top-k correction bias length mismatch: expected {} got {}",
            experts,
            correction_bias.len()
        );
    }
    Ok(expected_weight)
}

pub(in crate::commands::real_full) fn validate_router_topk_bf16_resident_device_input(
    hidden: &DeviceBf16Output,
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    experts: usize,
    top_k: usize,
) -> Result<()> {
    let expected_weight = validate_router_topk_bf16_preloaded_resident_device_input(
        hidden,
        Some(correction_bias),
        experts,
        top_k,
    )?;
    if router_weight_bf16.len() != expected_weight {
        anyhow::bail!(
            "real full BF16 router top-k device-input weight byte length mismatch: expected {} got {}",
            expected_weight,
            router_weight_bf16.len()
        );
    }
    Ok(())
}

pub(in crate::commands::real_full) fn validate_router_topk_bf16_preloaded_resident_device_input(
    hidden: &DeviceBf16Output,
    correction_bias: Option<&[f32]>,
    experts: usize,
    top_k: usize,
) -> Result<usize> {
    if hidden.rows == 0 || hidden.values_per_row == 0 || experts == 0 {
        anyhow::bail!(
            "real full preloaded BF16 router top-k device-input requires non-zero shape, got rows={} hidden_dim={} experts={experts}",
            hidden.rows,
            hidden.values_per_row
        );
    }
    if top_k == 0 || top_k > experts || top_k > GLMRT_CUDA_ROUTER_TOPK_MAX_K {
        anyhow::bail!(
            "real full preloaded BF16 router top-k device-input invalid top_k={top_k} for experts={experts}; max supported top_k={GLMRT_CUDA_ROUTER_TOPK_MAX_K}"
        );
    }
    let expected_hidden = hidden
        .rows
        .checked_mul(hidden.values_per_row)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full preloaded BF16 router top-k device-input hidden shape overflows usize while validating input",
        )?;
    if hidden.bytes != expected_hidden {
        anyhow::bail!(
            "real full preloaded BF16 router top-k device-input hidden byte length mismatch: expected {} got {}",
            expected_hidden,
            hidden.bytes
        );
    }
    let buffer = hidden.buffer();
    if buffer.ptr.is_null() {
        anyhow::bail!("real full preloaded BF16 router top-k device-input hidden buffer is null");
    }
    if buffer.bytes < expected_hidden {
        anyhow::bail!(
            "real full preloaded BF16 router top-k device-input hidden buffer has {} bytes, needs {expected_hidden}",
            buffer.bytes
        );
    }
    let expected_weight = experts
        .checked_mul(hidden.values_per_row)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full preloaded BF16 router top-k device-input weight shape overflows usize while validating input",
        )?;
    if let Some(correction_bias) = correction_bias {
        if correction_bias.len() != experts {
            anyhow::bail!(
                "real full preloaded BF16 router top-k device-input correction bias length mismatch: expected {} got {}",
                experts,
                correction_bias.len()
            );
        }
    }
    Ok(expected_weight)
}

pub(in crate::commands::real_full) fn cpu_router_topk(
    hidden: &[f32],
    router_weight: &[f32],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> RouterTopKOutput {
    let mut indices = vec![0_usize; rows * top_k];
    let mut scores = vec![0.0_f32; rows * top_k];
    let mut weights = vec![0.0_f32; rows * top_k];
    for row in 0..rows {
        let mut best_scores = vec![0.0_f32; top_k];
        let mut best_corrected = vec![f32::NEG_INFINITY; top_k];
        let mut best_indices = vec![0_usize; top_k];
        let hidden_start = row * hidden_dim;
        for expert in 0..experts {
            let weight_start = expert * hidden_dim;
            let mut logit = 0.0_f32;
            for col in 0..hidden_dim {
                logit += hidden[hidden_start + col] * router_weight[weight_start + col];
            }
            let score = 1.0 / (1.0 + (-logit).exp());
            let corrected = score + correction_bias[expert];
            for rank in 0..top_k {
                if corrected > best_corrected[rank] {
                    for shift in (rank + 1..top_k).rev() {
                        best_corrected[shift] = best_corrected[shift - 1];
                        best_scores[shift] = best_scores[shift - 1];
                        best_indices[shift] = best_indices[shift - 1];
                    }
                    best_corrected[rank] = corrected;
                    best_scores[rank] = score;
                    best_indices[rank] = expert;
                    break;
                }
            }
        }
        let score_sum = best_scores.iter().sum::<f32>().max(1.0e-12);
        let out_start = row * top_k;
        for rank in 0..top_k {
            indices[out_start + rank] = best_indices[rank];
            scores[out_start + rank] = best_scores[rank];
            weights[out_start + rank] = best_scores[rank] / score_sum * GLM52_ROUTED_SCALING_FACTOR;
        }
    }
    RouterTopKOutput {
        indices,
        scores,
        weights,
        backend: CPU_REFERENCE_ROUTER_TOPK_BACKEND,
    }
}

pub(in crate::commands::real_full) fn cpu_router_topk_bf16(
    hidden_bf16: &[u8],
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> RouterTopKOutput {
    let mut output = cpu_router_topk(
        &bf16_values_to_f32(hidden_bf16),
        &bf16_values_to_f32(router_weight_bf16),
        correction_bias,
        rows,
        hidden_dim,
        experts,
        top_k,
    );
    output.backend = CPU_REFERENCE_ROUTER_TOPK_BF16_BACKEND;
    output
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_router_topk(
    hidden: &[f32],
    router_weight: &[f32],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    let library = cuda_native_library()?;
    let hidden_bytes = std::mem::size_of_val(hidden);
    let weight_bytes = std::mem::size_of_val(router_weight);
    let bias_bytes = std::mem::size_of_val(correction_bias);
    let output_values = rows
        .checked_mul(top_k)
        .context("CUDA router top-k output shape overflows usize")?;
    let index_bytes = output_values
        .checked_mul(std::mem::size_of::<u32>())
        .context("CUDA router top-k index bytes overflow usize")?;
    let score_bytes = output_values
        .checked_mul(std::mem::size_of::<f32>())
        .context("CUDA router top-k score bytes overflow usize")?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let hidden_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        hidden_bytes,
        "router hidden",
    )?;
    let weight_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::B,
        weight_bytes,
        "router weight",
    )?;
    let bias_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        bias_bytes,
        "router correction bias",
    )?;
    let index_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        index_bytes,
        "router top-k indices",
    )?;
    let score_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::E,
        score_bytes,
        "router top-k scores",
    )?;
    let weight_buffer_out = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::F,
        score_bytes,
        "router top-k weights",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            f32_bytes(hidden),
            "router hidden",
        )
        .context("copying router hidden to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::B,
            f32_bytes(router_weight),
            "router weight",
        )
        .context("copying router weight to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::C,
            f32_bytes(correction_bias),
            "router correction bias",
        )
        .context("copying router correction bias to device")?;
    library
        .cuda_router_topk_f32(
            hidden_buffer,
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            rows,
            hidden_dim,
            experts,
            top_k,
        )
        .context("executing CUDA router top-k")?;
    let mut index_out = vec![0_u8; index_bytes];
    let mut score_out = vec![0_u8; score_bytes];
    let mut weight_out = vec![0_u8; score_bytes];
    library
        .copy_d2h(&mut index_out, index_buffer)
        .context("copying router top-k indices to host")?;
    library
        .copy_d2h(&mut score_out, score_buffer)
        .context("copying router top-k scores to host")?;
    library
        .copy_d2h(&mut weight_out, weight_buffer_out)
        .context("copying router top-k weights to host")?;

    Ok(RouterTopKOutput {
        indices: u32_vec_from_bytes(&index_out)?
            .into_iter()
            .map(|value| value as usize)
            .collect(),
        scores: f32_vec_from_bytes(&score_out)?,
        weights: f32_vec_from_bytes(&weight_out)?,
        backend: CUDA_REFERENCE_ROUTER_TOPK_BACKEND,
    })
}

pub(in crate::commands::real_full) fn cuda_router_topk_bf16(
    hidden_bf16: &[u8],
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    if let Some(graph_key) = coord_sparse_a_graph_key_for_full_hidden_rows(rows, hidden_dim)? {
        return cuda_router_topk_bf16_graph_slot(
            &graph_key,
            hidden_bf16,
            router_weight_bf16,
            correction_bias,
            rows,
            hidden_dim,
            experts,
            top_k,
        );
    }
    cuda_router_topk_bf16_legacy(
        hidden_bf16,
        router_weight_bf16,
        correction_bias,
        rows,
        hidden_dim,
        experts,
        top_k,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_graph_slot(
    graph_key: &CoordinatorGraphKey,
    hidden_bf16: &[u8],
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    let weight_bytes = router_weight_bf16.len();
    let bias_bytes = std::mem::size_of_val(correction_bias);
    let (_, index_bytes, score_bytes) =
        router_topk_output_byte_counts(rows, top_k, "CUDA BF16 graph-slot router top-k")?;
    let hidden_graph_bytes =
        router_topk_graph_hidden_bytes(graph_key, hidden_dim, "CUDA BF16 graph-slot router top-k")?;
    let (_, graph_index_bytes, graph_score_bytes) = router_topk_graph_output_byte_counts(
        graph_key,
        top_k,
        "CUDA BF16 graph-slot router top-k",
    )?;
    let signature = router_topk_graph_signature(graph_key, hidden_dim, experts, top_k);
    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let hidden_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            hidden_graph_bytes,
            "BF16 router hidden",
        )?;
        let weight_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::B,
            weight_bytes,
            "BF16 router weight",
        )?;
        let bias_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            bias_bytes,
            "BF16 router correction bias",
        )?;
        let index_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            graph_index_bytes,
            "BF16 router top-k indices",
        )?;
        let score_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::E,
            graph_score_bytes,
            "BF16 router top-k scores",
        )?;
        let weight_buffer_out = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::F,
            graph_score_bytes,
            "BF16 router top-k weights",
        )?;

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::A,
                hidden_bf16,
                "BF16 router hidden",
                cuda_stream,
            )
            .context("async copying BF16 router hidden to device")?;
        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::B,
                router_weight_bf16,
                "BF16 router weight",
                cuda_stream,
            )
            .context("async copying BF16 router weight to device")?;
        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::C,
                f32_bytes(correction_bias),
                "BF16 router correction bias",
                cuda_stream,
            )
            .context("async copying BF16 router correction bias to device")?;
        capture_or_update_sparse_a_router_topk_bf16_graph(
            library,
            slot,
            signature,
            hidden_buffer,
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            rows,
            hidden_dim,
            experts,
            top_k,
            "BF16 router top-k",
        )?;
        let mut index_out = vec![0_u8; index_bytes];
        let mut score_out = vec![0_u8; score_bytes];
        let mut weight_out = vec![0_u8; score_bytes];
        unsafe {
            library
                .copy_d2h_async(&mut index_out, index_buffer, cuda_stream)
                .context("async copying BF16 router top-k indices to host")?;
            library
                .copy_d2h_async(&mut score_out, score_buffer, cuda_stream)
                .context("async copying BF16 router top-k scores to host")?;
            library
                .copy_d2h_async(&mut weight_out, weight_buffer_out, cuda_stream)
                .context("async copying BF16 router top-k weights to host")?;
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing BF16 router top-k graph slot stream")?;
        }

        Ok(RouterTopKOutput {
            indices: u32_vec_from_bytes(&index_out)?
                .into_iter()
                .map(|value| value as usize)
                .collect(),
            scores: f32_vec_from_bytes(&score_out)?,
            weights: f32_vec_from_bytes(&weight_out)?,
            backend: CUDA_REFERENCE_ROUTER_TOPK_BF16_BACKEND,
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn capture_or_update_sparse_a_router_topk_bf16_graph(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    signature: CoordinatorCudaGraphSignature,
    hidden_buffer: GlmrtDeviceBuffer,
    weight_buffer: GlmrtDeviceBuffer,
    bias_buffer: GlmrtDeviceBuffer,
    index_buffer: GlmrtDeviceBuffer,
    score_buffer: GlmrtDeviceBuffer,
    weight_buffer_out: GlmrtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
    label: &'static str,
) -> Result<()> {
    if !slot.has_captured_graph(
        CoordinatorCudaGraphProgram::SparseARouterTopKBf16,
        signature,
    ) {
        slot.stream_synchronize()
            .with_context(|| format!("synchronizing {label} inputs before graph capture"))?;
        slot.capture_graph(
            library,
            CoordinatorCudaGraphProgram::SparseARouterTopKBf16,
            signature,
            |library, cuda_stream, _workspace| unsafe {
                library
                    .cuda_router_topk_bf16_async(
                        hidden_buffer,
                        weight_buffer,
                        bias_buffer,
                        index_buffer,
                        score_buffer,
                        weight_buffer_out,
                        rows,
                        hidden_dim,
                        experts,
                        top_k,
                        cuda_stream,
                    )
                    .with_context(|| format!("capturing async CUDA {label}"))?;
                Ok(())
            },
        )?;
    } else {
        let (graph_raw, exec_raw) = slot
            .captured_graph_raw_handles(
                CoordinatorCudaGraphProgram::SparseARouterTopKBf16,
                signature,
            )
            .context(
                "coordinator CUDA graph slot lost captured router top-k graph before update",
            )?;
        unsafe {
            library
                .cuda_graph_update_router_topk_bf16_node(
                    graph_raw,
                    exec_raw,
                    0,
                    hidden_buffer,
                    weight_buffer,
                    bias_buffer,
                    index_buffer,
                    score_buffer,
                    weight_buffer_out,
                    rows,
                    hidden_dim,
                    experts,
                    top_k,
                )
                .with_context(|| format!("updating captured CUDA {label} graph node"))?;
        }
    }
    slot.launch_captured_graph(
        library,
        CoordinatorCudaGraphProgram::SparseARouterTopKBf16,
        signature,
    )
}

#[allow(clippy::too_many_arguments)]
fn capture_or_update_router_topk_bf16_graph_for_slot(
    graph_key: &CoordinatorGraphKey,
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    hidden_buffer: GlmrtDeviceBuffer,
    weight_buffer: GlmrtDeviceBuffer,
    bias_buffer: GlmrtDeviceBuffer,
    index_buffer: GlmrtDeviceBuffer,
    score_buffer: GlmrtDeviceBuffer,
    weight_buffer_out: GlmrtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
    native_backend: &'static str,
    triton_backend: &'static str,
    label: &'static str,
) -> Result<&'static str> {
    if triton_router_topk_bf16_supported(graph_key, rows, hidden_dim, experts, top_k) {
        let capture_rows = graph_key.row_bucket.row_capacity;
        let hidden_bytes = rows
            .checked_mul(hidden_dim)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .with_context(|| format!("{label} Triton hidden bytes overflow usize"))?;
        let capture_hidden_bytes = capture_rows
            .checked_mul(hidden_dim)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .with_context(|| format!("{label} Triton graph hidden bytes overflow usize"))?;
        let graph_hidden_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            capture_hidden_bytes,
            "Triton BF16 router graph hidden input",
        )?;
        let score_scratch_bytes =
            triton_router_topk_score_scratch_bytes(graph_key, experts, label)?;
        let score_scratch_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::B,
            score_scratch_bytes,
            "Triton BF16 router score scratch",
        )?;
        let cuda_stream = slot.stream_ptr();
        unsafe {
            if hidden_buffer.ptr != graph_hidden_buffer.ptr {
                library
                    .copy_d2d_async(
                        graph_hidden_buffer,
                        hidden_buffer,
                        hidden_bytes,
                        cuda_stream,
                    )
                    .with_context(|| format!("staging {label} Triton graph hidden input"))?;
            }
            if capture_hidden_bytes > hidden_bytes {
                let padding = device_buffer_byte_view(
                    graph_hidden_buffer,
                    hidden_bytes,
                    capture_hidden_bytes - hidden_bytes,
                    "Triton BF16 router padded hidden rows",
                )?;
                library
                    .cuda_zero_bytes_async(padding, padding.bytes, cuda_stream)
                    .with_context(|| format!("zeroing {label} Triton padded hidden rows"))?;
            }
        }
        let signature = triton_router_topk_graph_signature(
            capture_rows,
            hidden_dim,
            experts,
            top_k,
            graph_hidden_buffer,
            weight_buffer,
            bias_buffer,
        );
        capture_or_update_sparse_a_triton_router_topk_bf16_graph(
            library,
            slot,
            signature,
            graph_hidden_buffer,
            weight_buffer,
            bias_buffer,
            score_scratch_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            capture_rows,
            hidden_dim,
            experts,
            top_k,
            label,
        )?;
        Ok(triton_backend)
    } else {
        let signature = router_topk_graph_signature(graph_key, hidden_dim, experts, top_k);
        capture_or_update_sparse_a_router_topk_bf16_graph(
            library,
            slot,
            signature,
            hidden_buffer,
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            rows,
            hidden_dim,
            experts,
            top_k,
            label,
        )?;
        Ok(native_backend)
    }
}

fn triton_router_topk_bf16_supported(
    graph_key: &CoordinatorGraphKey,
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> bool {
    coordinator_python_capture_enabled()
        && graph_key.shape == CoordinatorGraphShape::CoordSparseA
        && rows > 0
        && rows <= graph_key.row_bucket.row_capacity
        && hidden_dim == GLM52_HIDDEN_SIZE
        && experts > 0
        && top_k > 0
        && top_k <= experts
}

fn triton_router_topk_score_scratch_bytes(
    graph_key: &CoordinatorGraphKey,
    experts: usize,
    label: &str,
) -> Result<usize> {
    graph_key
        .row_bucket
        .row_capacity
        .checked_mul(experts)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .with_context(|| format!("{label} Triton score scratch buffer bytes overflow usize"))
}

pub(in crate::commands::real_full) fn cuda_router_topk_bf16_legacy(
    hidden_bf16: &[u8],
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    let library = cuda_native_library()?;
    let hidden_bytes = hidden_bf16.len();
    let weight_bytes = router_weight_bf16.len();
    let bias_bytes = std::mem::size_of_val(correction_bias);
    let output_values = rows
        .checked_mul(top_k)
        .context("CUDA BF16 router top-k output shape overflows usize")?;
    let index_bytes = output_values
        .checked_mul(std::mem::size_of::<u32>())
        .context("CUDA BF16 router top-k index bytes overflow usize")?;
    let score_bytes = output_values
        .checked_mul(std::mem::size_of::<f32>())
        .context("CUDA BF16 router top-k score bytes overflow usize")?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let hidden_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        hidden_bytes,
        "BF16 router hidden",
    )?;
    let weight_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::B,
        weight_bytes,
        "BF16 router weight",
    )?;
    let bias_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        bias_bytes,
        "BF16 router correction bias",
    )?;
    let index_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        index_bytes,
        "BF16 router top-k indices",
    )?;
    let score_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::E,
        score_bytes,
        "BF16 router top-k scores",
    )?;
    let weight_buffer_out = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::F,
        score_bytes,
        "BF16 router top-k weights",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            hidden_bf16,
            "BF16 router hidden",
        )
        .context("copying BF16 router hidden to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::B,
            router_weight_bf16,
            "BF16 router weight",
        )
        .context("copying BF16 router weight to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::C,
            f32_bytes(correction_bias),
            "BF16 router correction bias",
        )
        .context("copying BF16 router correction bias to device")?;
    library
        .cuda_router_topk_bf16(
            hidden_buffer,
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            rows,
            hidden_dim,
            experts,
            top_k,
        )
        .context("executing CUDA BF16 router top-k")?;
    let mut index_out = vec![0_u8; index_bytes];
    let mut score_out = vec![0_u8; score_bytes];
    let mut weight_out = vec![0_u8; score_bytes];
    library
        .copy_d2h(&mut index_out, index_buffer)
        .context("copying BF16 router top-k indices to host")?;
    library
        .copy_d2h(&mut score_out, score_buffer)
        .context("copying BF16 router top-k scores to host")?;
    library
        .copy_d2h(&mut weight_out, weight_buffer_out)
        .context("copying BF16 router top-k weights to host")?;

    Ok(RouterTopKOutput {
        indices: u32_vec_from_bytes(&index_out)?
            .into_iter()
            .map(|value| value as usize)
            .collect(),
        scores: f32_vec_from_bytes(&score_out)?,
        weights: f32_vec_from_bytes(&weight_out)?,
        backend: CUDA_REFERENCE_ROUTER_TOPK_BF16_BACKEND,
    })
}

pub(in crate::commands::real_full) fn cuda_router_topk_bf16_resident_weight(
    router_weight_name: &str,
    hidden_bf16: &[u8],
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    if let Some(graph_key) = coord_sparse_a_graph_key_for_full_hidden_rows(rows, hidden_dim)? {
        return cuda_router_topk_bf16_resident_weight_graph_slot(
            &graph_key,
            router_weight_name,
            hidden_bf16,
            router_weight_bf16,
            correction_bias,
            rows,
            hidden_dim,
            experts,
            top_k,
        );
    }
    cuda_router_topk_bf16_resident_weight_legacy(
        router_weight_name,
        hidden_bf16,
        router_weight_bf16,
        correction_bias,
        rows,
        hidden_dim,
        experts,
        top_k,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_resident_weight_graph_slot(
    graph_key: &CoordinatorGraphKey,
    router_weight_name: &str,
    hidden_bf16: &[u8],
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    let bias_bytes = std::mem::size_of_val(correction_bias);
    let (_, index_bytes, score_bytes) =
        router_topk_output_byte_counts(rows, top_k, "CUDA BF16 resident router top-k graph-slot")?;
    let hidden_graph_bytes = router_topk_graph_hidden_bytes(
        graph_key,
        hidden_dim,
        "CUDA BF16 resident router top-k graph-slot",
    )?;
    let (_, graph_index_bytes, graph_score_bytes) = router_topk_graph_output_byte_counts(
        graph_key,
        top_k,
        "CUDA BF16 resident router top-k graph-slot",
    )?;
    let weight_buffer = resident_weight_buffer_from_registry(
        router_weight_name,
        router_weight_bf16,
        "BF16 resident router weight",
    )?;

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let hidden_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            hidden_graph_bytes,
            "BF16 resident router hidden",
        )?;
        let bias_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            bias_bytes,
            "BF16 resident router correction bias",
        )?;
        let index_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            graph_index_bytes,
            "BF16 resident router top-k indices",
        )?;
        let score_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::E,
            graph_score_bytes,
            "BF16 resident router top-k scores",
        )?;
        let weight_buffer_out = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::F,
            graph_score_bytes,
            "BF16 resident router top-k weights",
        )?;

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::A,
                hidden_bf16,
                "BF16 resident router hidden",
                cuda_stream,
            )
            .context("async copying BF16 resident router hidden to device")?;
        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::C,
                f32_bytes(correction_bias),
                "BF16 resident router correction bias",
                cuda_stream,
            )
            .context("async copying BF16 resident router correction bias to device")?;
        let backend = capture_or_update_router_topk_bf16_graph_for_slot(
            graph_key,
            library,
            slot,
            hidden_buffer,
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            rows,
            hidden_dim,
            experts,
            top_k,
            CUDA_REFERENCE_ROUTER_TOPK_BF16_RESIDENT_WEIGHT_BACKEND,
            TRITON_ROUTER_TOPK_BF16_RESIDENT_WEIGHT_BACKEND,
            "BF16 resident router top-k",
        )?;
        let mut index_out = vec![0_u8; index_bytes];
        let mut score_out = vec![0_u8; score_bytes];
        let mut weight_out = vec![0_u8; score_bytes];
        unsafe {
            library
                .copy_d2h_async(&mut index_out, index_buffer, cuda_stream)
                .context("async copying BF16 resident router top-k indices to host")?;
            library
                .copy_d2h_async(&mut score_out, score_buffer, cuda_stream)
                .context("async copying BF16 resident router top-k scores to host")?;
            library
                .copy_d2h_async(&mut weight_out, weight_buffer_out, cuda_stream)
                .context("async copying BF16 resident router top-k weights to host")?;
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing BF16 resident router top-k graph slot stream")?;
        }

        Ok(RouterTopKOutput {
            indices: u32_vec_from_bytes(&index_out)?
                .into_iter()
                .map(|value| value as usize)
                .collect(),
            scores: f32_vec_from_bytes(&score_out)?,
            weights: f32_vec_from_bytes(&weight_out)?,
            backend,
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_resident_weight_legacy(
    router_weight_name: &str,
    hidden_bf16: &[u8],
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    let library = cuda_native_library()?;
    let hidden_bytes = hidden_bf16.len();
    let bias_bytes = std::mem::size_of_val(correction_bias);
    let output_values = rows
        .checked_mul(top_k)
        .context("CUDA BF16 resident router top-k output shape overflows usize")?;
    let index_bytes = output_values
        .checked_mul(std::mem::size_of::<u32>())
        .context("CUDA BF16 resident router top-k index bytes overflow usize")?;
    let score_bytes = output_values
        .checked_mul(std::mem::size_of::<f32>())
        .context("CUDA BF16 resident router top-k score bytes overflow usize")?;
    let weight_buffer = resident_weight_buffer_from_registry(
        router_weight_name,
        router_weight_bf16,
        "BF16 resident router weight",
    )?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let hidden_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        hidden_bytes,
        "BF16 resident router hidden",
    )?;
    let bias_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        bias_bytes,
        "BF16 resident router correction bias",
    )?;
    let index_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        index_bytes,
        "BF16 resident router top-k indices",
    )?;
    let score_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::E,
        score_bytes,
        "BF16 resident router top-k scores",
    )?;
    let weight_buffer_out = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::F,
        score_bytes,
        "BF16 resident router top-k weights",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            hidden_bf16,
            "BF16 resident router hidden",
        )
        .context("copying BF16 resident router hidden to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::C,
            f32_bytes(correction_bias),
            "BF16 resident router correction bias",
        )
        .context("copying BF16 resident router correction bias to device")?;
    library
        .cuda_router_topk_bf16(
            hidden_buffer,
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            rows,
            hidden_dim,
            experts,
            top_k,
        )
        .context("executing CUDA BF16 resident router top-k")?;
    let mut index_out = vec![0_u8; index_bytes];
    let mut score_out = vec![0_u8; score_bytes];
    let mut weight_out = vec![0_u8; score_bytes];
    library
        .copy_d2h(&mut index_out, index_buffer)
        .context("copying BF16 resident router top-k indices to host")?;
    library
        .copy_d2h(&mut score_out, score_buffer)
        .context("copying BF16 resident router top-k scores to host")?;
    library
        .copy_d2h(&mut weight_out, weight_buffer_out)
        .context("copying BF16 resident router top-k weights to host")?;

    Ok(RouterTopKOutput {
        indices: u32_vec_from_bytes(&index_out)?
            .into_iter()
            .map(|value| value as usize)
            .collect(),
        scores: f32_vec_from_bytes(&score_out)?,
        weights: f32_vec_from_bytes(&weight_out)?,
        backend: CUDA_REFERENCE_ROUTER_TOPK_BF16_RESIDENT_WEIGHT_BACKEND,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_preloaded_resident_weight(
    router_weight_name: &str,
    hidden_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
    weight_bytes: usize,
) -> Result<RouterTopKOutput> {
    if let Some(graph_key) = coord_sparse_a_graph_key_for_full_hidden_rows(rows, hidden_dim)? {
        return cuda_router_topk_bf16_preloaded_resident_weight_graph_slot(
            &graph_key,
            router_weight_name,
            hidden_bf16,
            correction_bias,
            rows,
            hidden_dim,
            experts,
            top_k,
            weight_bytes,
        );
    }
    cuda_router_topk_bf16_preloaded_resident_weight_legacy(
        router_weight_name,
        hidden_bf16,
        correction_bias,
        rows,
        hidden_dim,
        experts,
        top_k,
        weight_bytes,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_preloaded_resident_weight_graph_slot(
    graph_key: &CoordinatorGraphKey,
    router_weight_name: &str,
    hidden_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
    weight_bytes: usize,
) -> Result<RouterTopKOutput> {
    let bias_bytes = std::mem::size_of_val(correction_bias);
    let (_, index_bytes, score_bytes) = router_topk_output_byte_counts(
        rows,
        top_k,
        "CUDA BF16 preloaded resident router top-k graph-slot",
    )?;
    let hidden_graph_bytes = router_topk_graph_hidden_bytes(
        graph_key,
        hidden_dim,
        "CUDA BF16 preloaded resident router top-k graph-slot",
    )?;
    let (_, graph_index_bytes, graph_score_bytes) = router_topk_graph_output_byte_counts(
        graph_key,
        top_k,
        "CUDA BF16 preloaded resident router top-k graph-slot",
    )?;
    let weight_buffer = preloaded_resident_weight_device_buffer(router_weight_name, weight_bytes)?;

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let hidden_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            hidden_graph_bytes,
            "BF16 preloaded resident router hidden",
        )?;
        let bias_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            bias_bytes,
            "BF16 preloaded resident router correction bias",
        )?;
        let index_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            graph_index_bytes,
            "BF16 preloaded resident router top-k indices",
        )?;
        let score_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::E,
            graph_score_bytes,
            "BF16 preloaded resident router top-k scores",
        )?;
        let weight_buffer_out = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::F,
            graph_score_bytes,
            "BF16 preloaded resident router top-k weights",
        )?;

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::A,
                hidden_bf16,
                "BF16 preloaded resident router hidden",
                cuda_stream,
            )
            .context("async copying BF16 preloaded resident router hidden to device")?;
        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::C,
                f32_bytes(correction_bias),
                "BF16 preloaded resident router correction bias",
                cuda_stream,
            )
            .context("async copying BF16 preloaded resident router correction bias to device")?;
        let backend = capture_or_update_router_topk_bf16_graph_for_slot(
            graph_key,
            library,
            slot,
            hidden_buffer,
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            rows,
            hidden_dim,
            experts,
            top_k,
            CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
            TRITON_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
            "BF16 preloaded resident router top-k",
        )?;
        let mut index_out = vec![0_u8; index_bytes];
        let mut score_out = vec![0_u8; score_bytes];
        let mut weight_out = vec![0_u8; score_bytes];
        unsafe {
            library
                .copy_d2h_async(&mut index_out, index_buffer, cuda_stream)
                .context("async copying BF16 preloaded resident router top-k indices to host")?;
            library
                .copy_d2h_async(&mut score_out, score_buffer, cuda_stream)
                .context("async copying BF16 preloaded resident router top-k scores to host")?;
            library
                .copy_d2h_async(&mut weight_out, weight_buffer_out, cuda_stream)
                .context("async copying BF16 preloaded resident router top-k weights to host")?;
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing BF16 preloaded resident router top-k graph slot stream")?;
        }

        Ok(RouterTopKOutput {
            indices: u32_vec_from_bytes(&index_out)?
                .into_iter()
                .map(|value| value as usize)
                .collect(),
            scores: f32_vec_from_bytes(&score_out)?,
            weights: f32_vec_from_bytes(&weight_out)?,
            backend,
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_preloaded_resident_weight_legacy(
    router_weight_name: &str,
    hidden_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
    weight_bytes: usize,
) -> Result<RouterTopKOutput> {
    let library = cuda_native_library()?;
    let hidden_bytes = hidden_bf16.len();
    let bias_bytes = std::mem::size_of_val(correction_bias);
    let output_values = rows
        .checked_mul(top_k)
        .context("CUDA BF16 preloaded resident router top-k output shape overflows usize")?;
    let index_bytes = output_values
        .checked_mul(std::mem::size_of::<u32>())
        .context("CUDA BF16 preloaded resident router top-k index bytes overflow usize")?;
    let score_bytes = output_values
        .checked_mul(std::mem::size_of::<f32>())
        .context("CUDA BF16 preloaded resident router top-k score bytes overflow usize")?;
    let weight_buffer = preloaded_resident_weight_device_buffer(router_weight_name, weight_bytes)?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let hidden_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        hidden_bytes,
        "BF16 preloaded resident router hidden",
    )?;
    let bias_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        bias_bytes,
        "BF16 preloaded resident router correction bias",
    )?;
    let index_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        index_bytes,
        "BF16 preloaded resident router top-k indices",
    )?;
    let score_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::E,
        score_bytes,
        "BF16 preloaded resident router top-k scores",
    )?;
    let weight_buffer_out = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::F,
        score_bytes,
        "BF16 preloaded resident router top-k weights",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            hidden_bf16,
            "BF16 preloaded resident router hidden",
        )
        .context("copying BF16 preloaded resident router hidden to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::C,
            f32_bytes(correction_bias),
            "BF16 preloaded resident router correction bias",
        )
        .context("copying BF16 preloaded resident router correction bias to device")?;
    library
        .cuda_router_topk_bf16(
            hidden_buffer,
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            rows,
            hidden_dim,
            experts,
            top_k,
        )
        .context("executing CUDA BF16 preloaded resident router top-k")?;
    let mut index_out = vec![0_u8; index_bytes];
    let mut score_out = vec![0_u8; score_bytes];
    let mut weight_out = vec![0_u8; score_bytes];
    library
        .copy_d2h(&mut index_out, index_buffer)
        .context("copying BF16 preloaded resident router top-k indices to host")?;
    library
        .copy_d2h(&mut score_out, score_buffer)
        .context("copying BF16 preloaded resident router top-k scores to host")?;
    library
        .copy_d2h(&mut weight_out, weight_buffer_out)
        .context("copying BF16 preloaded resident router top-k weights to host")?;

    Ok(RouterTopKOutput {
        indices: u32_vec_from_bytes(&index_out)?
            .into_iter()
            .map(|value| value as usize)
            .collect(),
        scores: f32_vec_from_bytes(&score_out)?,
        weights: f32_vec_from_bytes(&weight_out)?,
        backend: CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_preloaded_resident_weight_bias(
    router_weight_name: &str,
    correction_bias_name: &str,
    hidden_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
    weight_bytes: usize,
    bias_bytes: usize,
) -> Result<RouterTopKOutput> {
    if let Some(graph_key) = coord_sparse_a_graph_key_for_full_hidden_rows(rows, hidden_dim)? {
        return cuda_router_topk_bf16_preloaded_resident_weight_bias_graph_slot(
            &graph_key,
            router_weight_name,
            correction_bias_name,
            hidden_bf16,
            rows,
            hidden_dim,
            experts,
            top_k,
            weight_bytes,
            bias_bytes,
        );
    }
    cuda_router_topk_bf16_preloaded_resident_weight_bias_legacy(
        router_weight_name,
        correction_bias_name,
        hidden_bf16,
        rows,
        hidden_dim,
        experts,
        top_k,
        weight_bytes,
        bias_bytes,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_preloaded_resident_weight_bias_graph_slot(
    graph_key: &CoordinatorGraphKey,
    router_weight_name: &str,
    correction_bias_name: &str,
    hidden_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
    weight_bytes: usize,
    bias_bytes: usize,
) -> Result<RouterTopKOutput> {
    let (_, index_bytes, score_bytes) = router_topk_output_byte_counts(
        rows,
        top_k,
        "CUDA BF16 preloaded resident router top-k graph-slot weight+bias",
    )?;
    let hidden_graph_bytes = router_topk_graph_hidden_bytes(
        graph_key,
        hidden_dim,
        "CUDA BF16 preloaded resident router top-k graph-slot weight+bias",
    )?;
    let (_, graph_index_bytes, graph_score_bytes) = router_topk_graph_output_byte_counts(
        graph_key,
        top_k,
        "CUDA BF16 preloaded resident router top-k graph-slot weight+bias",
    )?;
    let weight_buffer = preloaded_resident_weight_device_buffer(router_weight_name, weight_bytes)?;
    let bias_buffer = preloaded_resident_weight_device_buffer(correction_bias_name, bias_bytes)?;

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let hidden_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            hidden_graph_bytes,
            "BF16 preloaded resident router weight+bias hidden",
        )?;
        let index_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            graph_index_bytes,
            "BF16 preloaded resident router weight+bias top-k indices",
        )?;
        let score_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::E,
            graph_score_bytes,
            "BF16 preloaded resident router weight+bias top-k scores",
        )?;
        let weight_buffer_out = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::F,
            graph_score_bytes,
            "BF16 preloaded resident router weight+bias top-k weights",
        )?;

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::A,
                hidden_bf16,
                "BF16 preloaded resident router weight+bias hidden",
                cuda_stream,
            )
            .context("async copying BF16 preloaded resident router weight+bias hidden to device")?;
        let backend = capture_or_update_router_topk_bf16_graph_for_slot(
            graph_key,
            library,
            slot,
            hidden_buffer,
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            rows,
            hidden_dim,
            experts,
            top_k,
            CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_BACKEND,
            TRITON_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_BACKEND,
            "BF16 preloaded resident router top-k weight+bias",
        )?;
        let mut index_out = vec![0_u8; index_bytes];
        let mut score_out = vec![0_u8; score_bytes];
        let mut weight_out = vec![0_u8; score_bytes];
        unsafe {
            library
                .copy_d2h_async(&mut index_out, index_buffer, cuda_stream)
                .context(
                    "async copying BF16 preloaded resident router top-k weight+bias indices to host",
                )?;
            library
                .copy_d2h_async(&mut score_out, score_buffer, cuda_stream)
                .context(
                    "async copying BF16 preloaded resident router top-k weight+bias scores to host",
                )?;
            library
                .copy_d2h_async(&mut weight_out, weight_buffer_out, cuda_stream)
                .context(
                    "async copying BF16 preloaded resident router top-k weight+bias weights to host",
                )?;
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing BF16 preloaded resident router top-k graph slot stream")?;
        }

        Ok(RouterTopKOutput {
            indices: u32_vec_from_bytes(&index_out)?
                .into_iter()
                .map(|value| value as usize)
                .collect(),
            scores: f32_vec_from_bytes(&score_out)?,
            weights: f32_vec_from_bytes(&weight_out)?,
            backend,
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_preloaded_resident_weight_bias_legacy(
    router_weight_name: &str,
    correction_bias_name: &str,
    hidden_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
    weight_bytes: usize,
    bias_bytes: usize,
) -> Result<RouterTopKOutput> {
    let library = cuda_native_library()?;
    let hidden_bytes = hidden_bf16.len();
    let output_values = rows.checked_mul(top_k).context(
        "CUDA BF16 preloaded resident router top-k weight+bias output shape overflows usize",
    )?;
    let index_bytes = output_values
        .checked_mul(std::mem::size_of::<u32>())
        .context(
            "CUDA BF16 preloaded resident router top-k weight+bias index bytes overflow usize",
        )?;
    let score_bytes = output_values
        .checked_mul(std::mem::size_of::<f32>())
        .context(
            "CUDA BF16 preloaded resident router top-k weight+bias score bytes overflow usize",
        )?;
    let weight_buffer = preloaded_resident_weight_device_buffer(router_weight_name, weight_bytes)?;
    let bias_buffer = preloaded_resident_weight_device_buffer(correction_bias_name, bias_bytes)?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let hidden_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        hidden_bytes,
        "BF16 preloaded resident router weight+bias hidden",
    )?;
    let index_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        index_bytes,
        "BF16 preloaded resident router weight+bias top-k indices",
    )?;
    let score_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::E,
        score_bytes,
        "BF16 preloaded resident router weight+bias top-k scores",
    )?;
    let weight_buffer_out = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::F,
        score_bytes,
        "BF16 preloaded resident router weight+bias top-k weights",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            hidden_bf16,
            "BF16 preloaded resident router weight+bias hidden",
        )
        .context("copying BF16 preloaded resident router weight+bias hidden to device")?;
    library
        .cuda_router_topk_bf16(
            hidden_buffer,
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            rows,
            hidden_dim,
            experts,
            top_k,
        )
        .context("executing CUDA BF16 preloaded resident router top-k weight+bias")?;
    let mut index_out = vec![0_u8; index_bytes];
    let mut score_out = vec![0_u8; score_bytes];
    let mut weight_out = vec![0_u8; score_bytes];
    library
        .copy_d2h(&mut index_out, index_buffer)
        .context("copying BF16 preloaded resident router top-k weight+bias indices to host")?;
    library
        .copy_d2h(&mut score_out, score_buffer)
        .context("copying BF16 preloaded resident router top-k weight+bias scores to host")?;
    library
        .copy_d2h(&mut weight_out, weight_buffer_out)
        .context("copying BF16 preloaded resident router top-k weight+bias weights to host")?;

    Ok(RouterTopKOutput {
        indices: u32_vec_from_bytes(&index_out)?
            .into_iter()
            .map(|value| value as usize)
            .collect(),
        scores: f32_vec_from_bytes(&score_out)?,
        weights: f32_vec_from_bytes(&weight_out)?,
        backend: CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_BACKEND,
    })
}

pub(in crate::commands::real_full) fn router_topk_output_byte_counts(
    rows: usize,
    top_k: usize,
    context: &str,
) -> Result<(usize, usize, usize)> {
    let output_values = rows
        .checked_mul(top_k)
        .with_context(|| format!("{context} output shape overflows usize"))?;
    let index_bytes = output_values
        .checked_mul(std::mem::size_of::<u32>())
        .with_context(|| format!("{context} index bytes overflow usize"))?;
    let score_bytes = output_values
        .checked_mul(std::mem::size_of::<f32>())
        .with_context(|| format!("{context} score bytes overflow usize"))?;
    Ok((output_values, index_bytes, score_bytes))
}

pub(in crate::commands::real_full) fn router_topk_graph_output_byte_counts(
    graph_key: &CoordinatorGraphKey,
    top_k: usize,
    context: &str,
) -> Result<(usize, usize, usize)> {
    router_topk_output_byte_counts(graph_key.row_bucket.row_capacity, top_k, context)
}

pub(in crate::commands::real_full) fn router_topk_graph_hidden_bytes(
    graph_key: &CoordinatorGraphKey,
    hidden_dim: usize,
    context: &str,
) -> Result<usize> {
    graph_key
        .row_bucket
        .row_capacity
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .with_context(|| format!("{context} hidden graph buffer bytes overflow usize"))
}

pub(in crate::commands::real_full) fn router_topk_graph_signature(
    graph_key: &CoordinatorGraphKey,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> CoordinatorCudaGraphSignature {
    CoordinatorCudaGraphSignature::router_topk_bf16(
        graph_key.row_bucket.row_capacity,
        hidden_dim,
        experts,
        top_k,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_resident_weight_device_input(
    router_weight_name: &str,
    hidden: &DeviceBf16Output,
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    if let Some(graph_key) =
        coord_sparse_a_graph_key_for_full_hidden_rows(hidden.rows, hidden.values_per_row)?
    {
        return cuda_router_topk_bf16_resident_weight_device_input_graph_slot(
            &graph_key,
            router_weight_name,
            hidden,
            router_weight_bf16,
            correction_bias,
            experts,
            top_k,
        );
    }
    cuda_router_topk_bf16_resident_weight_device_input_legacy(
        router_weight_name,
        hidden,
        router_weight_bf16,
        correction_bias,
        experts,
        top_k,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_resident_weight_device_input_graph_slot(
    graph_key: &CoordinatorGraphKey,
    router_weight_name: &str,
    hidden: &DeviceBf16Output,
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    let bias_bytes = std::mem::size_of_val(correction_bias);
    let (_, index_bytes, score_bytes) = router_topk_output_byte_counts(
        hidden.rows,
        top_k,
        "CUDA BF16 resident router top-k device-input graph-slot",
    )?;
    let (_, graph_index_bytes, graph_score_bytes) = router_topk_graph_output_byte_counts(
        graph_key,
        top_k,
        "CUDA BF16 resident router top-k device-input graph-slot",
    )?;
    let weight_buffer = resident_weight_buffer_from_registry(
        router_weight_name,
        router_weight_bf16,
        "BF16 resident router device-input weight",
    )?;

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        hidden
            .wait_ready_on_stream(cuda_stream)
            .context("waiting for BF16 resident router device input")?;
        let hidden_buffer = hidden.buffer();
        let bias_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            bias_bytes,
            "BF16 resident router device-input correction bias",
        )?;
        let index_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            graph_index_bytes,
            "BF16 resident router device-input top-k indices",
        )?;
        let score_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::E,
            graph_score_bytes,
            "BF16 resident router device-input top-k scores",
        )?;
        let weight_buffer_out = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::F,
            graph_score_bytes,
            "BF16 resident router device-input top-k weights",
        )?;

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::C,
                f32_bytes(correction_bias),
                "BF16 resident router device-input correction bias",
                cuda_stream,
            )
            .context("async copying BF16 resident router device-input correction bias to device")?;
        let backend = capture_or_update_router_topk_bf16_graph_for_slot(
            graph_key,
            library,
            slot,
            hidden_buffer,
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            hidden.rows,
            hidden.values_per_row,
            experts,
            top_k,
            CUDA_REFERENCE_ROUTER_TOPK_BF16_RESIDENT_WEIGHT_DEVICE_INPUT_BACKEND,
            TRITON_ROUTER_TOPK_BF16_RESIDENT_WEIGHT_DEVICE_INPUT_BACKEND,
            "BF16 resident router top-k device-input",
        )?;
        let mut index_out = vec![0_u8; index_bytes];
        let mut score_out = vec![0_u8; score_bytes];
        let mut weight_out = vec![0_u8; score_bytes];
        unsafe {
            library
                .copy_d2h_async(&mut index_out, index_buffer, cuda_stream)
                .context("async copying BF16 resident router device-input top-k indices to host")?;
            library
                .copy_d2h_async(&mut score_out, score_buffer, cuda_stream)
                .context("async copying BF16 resident router device-input top-k scores to host")?;
            library
                .copy_d2h_async(&mut weight_out, weight_buffer_out, cuda_stream)
                .context("async copying BF16 resident router device-input top-k weights to host")?;
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing BF16 resident router device-input graph slot stream")?;
        }

        Ok(RouterTopKOutput {
            indices: u32_vec_from_bytes(&index_out)?
                .into_iter()
                .map(|value| value as usize)
                .collect(),
            scores: f32_vec_from_bytes(&score_out)?,
            weights: f32_vec_from_bytes(&weight_out)?,
            backend,
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_resident_weight_device_input_legacy(
    router_weight_name: &str,
    hidden: &DeviceBf16Output,
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    let library = cuda_native_library()?;
    let bias_bytes = std::mem::size_of_val(correction_bias);
    let (_, index_bytes, score_bytes) = router_topk_output_byte_counts(
        hidden.rows,
        top_k,
        "CUDA BF16 resident router top-k device-input",
    )?;
    let weight_buffer = resident_weight_buffer_from_registry(
        router_weight_name,
        router_weight_bf16,
        "BF16 resident router device-input weight",
    )?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let bias_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        bias_bytes,
        "BF16 resident router device-input correction bias",
    )?;
    let index_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        index_bytes,
        "BF16 resident router device-input top-k indices",
    )?;
    let score_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::E,
        score_bytes,
        "BF16 resident router device-input top-k scores",
    )?;
    let weight_buffer_out = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::F,
        score_bytes,
        "BF16 resident router device-input top-k weights",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::C,
            f32_bytes(correction_bias),
            "BF16 resident router device-input correction bias",
        )
        .context("copying BF16 resident router device-input correction bias to device")?;
    library
        .cuda_router_topk_bf16(
            hidden.buffer(),
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            hidden.rows,
            hidden.values_per_row,
            experts,
            top_k,
        )
        .context("executing CUDA BF16 resident router top-k device-input")?;
    let mut index_out = vec![0_u8; index_bytes];
    let mut score_out = vec![0_u8; score_bytes];
    let mut weight_out = vec![0_u8; score_bytes];
    library
        .copy_d2h(&mut index_out, index_buffer)
        .context("copying BF16 resident router device-input top-k indices to host")?;
    library
        .copy_d2h(&mut score_out, score_buffer)
        .context("copying BF16 resident router device-input top-k scores to host")?;
    library
        .copy_d2h(&mut weight_out, weight_buffer_out)
        .context("copying BF16 resident router device-input top-k weights to host")?;

    Ok(RouterTopKOutput {
        indices: u32_vec_from_bytes(&index_out)?
            .into_iter()
            .map(|value| value as usize)
            .collect(),
        scores: f32_vec_from_bytes(&score_out)?,
        weights: f32_vec_from_bytes(&weight_out)?,
        backend: CUDA_REFERENCE_ROUTER_TOPK_BF16_RESIDENT_WEIGHT_DEVICE_INPUT_BACKEND,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_preloaded_resident_weight_device_input(
    router_weight_name: &str,
    hidden: &DeviceBf16Output,
    correction_bias: &[f32],
    experts: usize,
    top_k: usize,
    weight_bytes: usize,
) -> Result<RouterTopKOutput> {
    if let Some(graph_key) =
        coord_sparse_a_graph_key_for_full_hidden_rows(hidden.rows, hidden.values_per_row)?
    {
        return cuda_router_topk_bf16_preloaded_resident_weight_device_input_graph_slot(
            &graph_key,
            router_weight_name,
            hidden,
            correction_bias,
            experts,
            top_k,
            weight_bytes,
        );
    }
    cuda_router_topk_bf16_preloaded_resident_weight_device_input_legacy(
        router_weight_name,
        hidden,
        correction_bias,
        experts,
        top_k,
        weight_bytes,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_preloaded_resident_weight_device_input_graph_slot(
    graph_key: &CoordinatorGraphKey,
    router_weight_name: &str,
    hidden: &DeviceBf16Output,
    correction_bias: &[f32],
    experts: usize,
    top_k: usize,
    weight_bytes: usize,
) -> Result<RouterTopKOutput> {
    let bias_bytes = std::mem::size_of_val(correction_bias);
    let (_, index_bytes, score_bytes) = router_topk_output_byte_counts(
        hidden.rows,
        top_k,
        "CUDA BF16 preloaded resident router top-k device-input graph-slot",
    )?;
    let (_, graph_index_bytes, graph_score_bytes) = router_topk_graph_output_byte_counts(
        graph_key,
        top_k,
        "CUDA BF16 preloaded resident router top-k device-input graph-slot",
    )?;
    let weight_buffer = preloaded_resident_weight_device_buffer(router_weight_name, weight_bytes)?;

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        hidden
            .wait_ready_on_stream(cuda_stream)
            .context("waiting for BF16 preloaded resident router device input")?;
        let hidden_buffer = hidden.buffer();
        let bias_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            bias_bytes,
            "BF16 preloaded resident router device-input correction bias",
        )?;
        let index_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            graph_index_bytes,
            "BF16 preloaded resident router device-input top-k indices",
        )?;
        let score_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::E,
            graph_score_bytes,
            "BF16 preloaded resident router device-input top-k scores",
        )?;
        let weight_buffer_out = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::F,
            graph_score_bytes,
            "BF16 preloaded resident router device-input top-k weights",
        )?;

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::C,
                f32_bytes(correction_bias),
                "BF16 preloaded resident router device-input correction bias",
                cuda_stream,
            )
            .context(
                "async copying BF16 preloaded resident router device-input correction bias to device",
            )?;
        let backend = capture_or_update_router_topk_bf16_graph_for_slot(
            graph_key,
            library,
            slot,
            hidden_buffer,
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            hidden.rows,
            hidden.values_per_row,
            experts,
            top_k,
            CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_DEVICE_INPUT_BACKEND,
            TRITON_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_DEVICE_INPUT_BACKEND,
            "BF16 preloaded resident router top-k device-input",
        )?;
        let mut index_out = vec![0_u8; index_bytes];
        let mut score_out = vec![0_u8; score_bytes];
        let mut weight_out = vec![0_u8; score_bytes];
        unsafe {
            library
                .copy_d2h_async(&mut index_out, index_buffer, cuda_stream)
                .context(
                    "async copying BF16 preloaded resident router device-input top-k indices to host",
                )?;
            library
                .copy_d2h_async(&mut score_out, score_buffer, cuda_stream)
                .context(
                    "async copying BF16 preloaded resident router device-input top-k scores to host",
                )?;
            library
                .copy_d2h_async(&mut weight_out, weight_buffer_out, cuda_stream)
                .context(
                    "async copying BF16 preloaded resident router device-input top-k weights to host",
                )?;
            library.cuda_stream_synchronize(cuda_stream).context(
                "synchronizing BF16 preloaded resident router device-input graph slot stream",
            )?;
        }

        Ok(RouterTopKOutput {
            indices: u32_vec_from_bytes(&index_out)?
                .into_iter()
                .map(|value| value as usize)
                .collect(),
            scores: f32_vec_from_bytes(&score_out)?,
            weights: f32_vec_from_bytes(&weight_out)?,
            backend,
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_preloaded_resident_weight_device_input_legacy(
    router_weight_name: &str,
    hidden: &DeviceBf16Output,
    correction_bias: &[f32],
    experts: usize,
    top_k: usize,
    weight_bytes: usize,
) -> Result<RouterTopKOutput> {
    let library = cuda_native_library()?;
    let bias_bytes = std::mem::size_of_val(correction_bias);
    let (_, index_bytes, score_bytes) = router_topk_output_byte_counts(
        hidden.rows,
        top_k,
        "CUDA BF16 preloaded resident router top-k device-input",
    )?;
    let weight_buffer = preloaded_resident_weight_device_buffer(router_weight_name, weight_bytes)?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let bias_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        bias_bytes,
        "BF16 preloaded resident router device-input correction bias",
    )?;
    let index_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        index_bytes,
        "BF16 preloaded resident router device-input top-k indices",
    )?;
    let score_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::E,
        score_bytes,
        "BF16 preloaded resident router device-input top-k scores",
    )?;
    let weight_buffer_out = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::F,
        score_bytes,
        "BF16 preloaded resident router device-input top-k weights",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::C,
            f32_bytes(correction_bias),
            "BF16 preloaded resident router device-input correction bias",
        )
        .context("copying BF16 preloaded resident router device-input correction bias to device")?;
    library
        .cuda_router_topk_bf16(
            hidden.buffer(),
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            hidden.rows,
            hidden.values_per_row,
            experts,
            top_k,
        )
        .context("executing CUDA BF16 preloaded resident router top-k device-input")?;
    let mut index_out = vec![0_u8; index_bytes];
    let mut score_out = vec![0_u8; score_bytes];
    let mut weight_out = vec![0_u8; score_bytes];
    library
        .copy_d2h(&mut index_out, index_buffer)
        .context("copying BF16 preloaded resident router device-input top-k indices to host")?;
    library
        .copy_d2h(&mut score_out, score_buffer)
        .context("copying BF16 preloaded resident router device-input top-k scores to host")?;
    library
        .copy_d2h(&mut weight_out, weight_buffer_out)
        .context("copying BF16 preloaded resident router device-input top-k weights to host")?;

    Ok(RouterTopKOutput {
        indices: u32_vec_from_bytes(&index_out)?
            .into_iter()
            .map(|value| value as usize)
            .collect(),
        scores: f32_vec_from_bytes(&score_out)?,
        weights: f32_vec_from_bytes(&weight_out)?,
        backend: CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_DEVICE_INPUT_BACKEND,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_preloaded_resident_weight_bias_device_input(
    router_weight_name: &str,
    correction_bias_name: &str,
    hidden: &DeviceBf16Output,
    experts: usize,
    top_k: usize,
    weight_bytes: usize,
    bias_bytes: usize,
) -> Result<RouterTopKOutput> {
    if let Some(graph_key) =
        coord_sparse_a_graph_key_for_full_hidden_rows(hidden.rows, hidden.values_per_row)?
    {
        return cuda_router_topk_bf16_preloaded_resident_weight_bias_device_input_graph_slot(
            &graph_key,
            router_weight_name,
            correction_bias_name,
            hidden,
            experts,
            top_k,
            weight_bytes,
            bias_bytes,
        );
    }
    cuda_router_topk_bf16_preloaded_resident_weight_bias_device_input_legacy(
        router_weight_name,
        correction_bias_name,
        hidden,
        experts,
        top_k,
        weight_bytes,
        bias_bytes,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_preloaded_resident_weight_bias_device_input_graph_slot(
    graph_key: &CoordinatorGraphKey,
    router_weight_name: &str,
    correction_bias_name: &str,
    hidden: &DeviceBf16Output,
    experts: usize,
    top_k: usize,
    weight_bytes: usize,
    bias_bytes: usize,
) -> Result<RouterTopKOutput> {
    let (_, index_bytes, score_bytes) = router_topk_output_byte_counts(
        hidden.rows,
        top_k,
        "CUDA BF16 preloaded resident router top-k weight+bias device-input graph-slot",
    )?;
    let (_, graph_index_bytes, graph_score_bytes) = router_topk_graph_output_byte_counts(
        graph_key,
        top_k,
        "CUDA BF16 preloaded resident router top-k weight+bias device-input graph-slot",
    )?;
    let graph_output_bytes = graph_score_bytes
        .checked_mul(2)
        .and_then(|score_and_weight_bytes| graph_index_bytes.checked_add(score_and_weight_bytes))
        .context(
            "CUDA BF16 preloaded resident router top-k packed graph output bytes overflow usize",
        )?;
    let graph_score_offset = graph_index_bytes;
    let graph_weight_offset = graph_index_bytes
        .checked_add(graph_score_bytes)
        .context("CUDA BF16 preloaded resident router top-k packed weight offset overflow usize")?;
    let weight_buffer = preloaded_resident_weight_device_buffer(router_weight_name, weight_bytes)?;
    let bias_buffer = preloaded_resident_weight_device_buffer(correction_bias_name, bias_bytes)?;

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        hidden
            .wait_ready_on_stream(cuda_stream)
            .context("waiting for BF16 preloaded resident router weight+bias device input")?;
        let hidden_buffer = hidden.buffer();
        let packed_output_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            graph_output_bytes,
            "BF16 preloaded resident router weight+bias device-input packed top-k output",
        )?;
        let index_buffer = device_buffer_byte_view(
            packed_output_buffer,
            0,
            graph_index_bytes,
            "BF16 preloaded resident router weight+bias device-input top-k index view",
        )?;
        let score_buffer = device_buffer_byte_view(
            packed_output_buffer,
            graph_score_offset,
            graph_score_bytes,
            "BF16 preloaded resident router weight+bias device-input top-k score view",
        )?;
        let weight_buffer_out = device_buffer_byte_view(
            packed_output_buffer,
            graph_weight_offset,
            graph_score_bytes,
            "BF16 preloaded resident router weight+bias device-input top-k weight view",
        )?;

        let backend = capture_or_update_router_topk_bf16_graph_for_slot(
            graph_key,
            library,
            slot,
            hidden_buffer,
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            hidden.rows,
            hidden.values_per_row,
            experts,
            top_k,
            CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_DEVICE_INPUT_BACKEND,
            TRITON_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_DEVICE_INPUT_BACKEND,
            "BF16 preloaded resident router top-k weight+bias device-input",
        )?;
        let packed_output = slot.workspace.host_buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            graph_output_bytes,
            "BF16 preloaded resident router weight+bias device-input packed top-k readback",
        )?;
        unsafe {
            library
                .copy_d2h_host_buffer_async(
                    packed_output,
                    packed_output_buffer,
                    graph_output_bytes,
                    cuda_stream,
                )
                .context(
                    "async copying BF16 preloaded resident router weight+bias device-input packed top-k output to host",
                )?;
            library
                .cuda_stream_synchronize(cuda_stream)
                .context(
                    "synchronizing BF16 preloaded resident router weight+bias device-input graph slot stream",
                )?;
        }

        let packed_bytes = unsafe {
            std::slice::from_raw_parts(packed_output.ptr.cast::<u8>(), graph_output_bytes)
        };
        let index_host_bytes = &packed_bytes[..index_bytes];
        let score_host_end = graph_score_offset.checked_add(score_bytes).context(
            "CUDA BF16 preloaded resident router top-k packed score readback end overflow usize",
        )?;
        let score_host_bytes = &packed_bytes[graph_score_offset..score_host_end];
        let weight_host_end = graph_weight_offset.checked_add(score_bytes).context(
            "CUDA BF16 preloaded resident router top-k packed weight readback end overflow usize",
        )?;
        let weight_host_bytes = &packed_bytes[graph_weight_offset..weight_host_end];

        Ok(RouterTopKOutput {
            indices: u32_vec_from_bytes(index_host_bytes)?
                .into_iter()
                .map(|value| value as usize)
                .collect(),
            scores: f32_vec_from_bytes(score_host_bytes)?,
            weights: f32_vec_from_bytes(weight_host_bytes)?,
            backend,
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_preloaded_resident_weight_bias_device_input_legacy(
    router_weight_name: &str,
    correction_bias_name: &str,
    hidden: &DeviceBf16Output,
    experts: usize,
    top_k: usize,
    weight_bytes: usize,
    bias_bytes: usize,
) -> Result<RouterTopKOutput> {
    let library = cuda_native_library()?;
    let (_, index_bytes, score_bytes) = router_topk_output_byte_counts(
        hidden.rows,
        top_k,
        "CUDA BF16 preloaded resident router top-k weight+bias device-input",
    )?;
    let weight_buffer = preloaded_resident_weight_device_buffer(router_weight_name, weight_bytes)?;
    let bias_buffer = preloaded_resident_weight_device_buffer(correction_bias_name, bias_bytes)?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let index_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        index_bytes,
        "BF16 preloaded resident router weight+bias device-input top-k indices",
    )?;
    let score_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::E,
        score_bytes,
        "BF16 preloaded resident router weight+bias device-input top-k scores",
    )?;
    let weight_buffer_out = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::F,
        score_bytes,
        "BF16 preloaded resident router weight+bias device-input top-k weights",
    )?;

    library
        .cuda_router_topk_bf16(
            hidden.buffer(),
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            hidden.rows,
            hidden.values_per_row,
            experts,
            top_k,
        )
        .context("executing CUDA BF16 preloaded resident router top-k weight+bias device-input")?;
    let mut index_out = vec![0_u8; index_bytes];
    let mut score_out = vec![0_u8; score_bytes];
    let mut weight_out = vec![0_u8; score_bytes];
    library.copy_d2h(&mut index_out, index_buffer).context(
        "copying BF16 preloaded resident router weight+bias device-input top-k indices to host",
    )?;
    library.copy_d2h(&mut score_out, score_buffer).context(
        "copying BF16 preloaded resident router weight+bias device-input top-k scores to host",
    )?;
    library
        .copy_d2h(&mut weight_out, weight_buffer_out)
        .context(
            "copying BF16 preloaded resident router weight+bias device-input top-k weights to host",
        )?;

    Ok(RouterTopKOutput {
        indices: u32_vec_from_bytes(&index_out)?
            .into_iter()
            .map(|value| value as usize)
            .collect(),
        scores: f32_vec_from_bytes(&score_out)?,
        weights: f32_vec_from_bytes(&weight_out)?,
        backend:
            CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_DEVICE_INPUT_BACKEND,
    })
}
