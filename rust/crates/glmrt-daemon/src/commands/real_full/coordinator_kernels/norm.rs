use super::*;
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

#[allow(dead_code)]
pub(in crate::commands::real_full) const CPU_REFERENCE_RMSNORM_BACKEND: &str =
    "cpu-reference-rmsnorm";
#[allow(dead_code)]
pub(in crate::commands::real_full) const CUDA_REFERENCE_RMSNORM_BACKEND: &str =
    "cuda-reference-rmsnorm-f32";
pub(in crate::commands::real_full) const CPU_REFERENCE_RMSNORM_BF16_BACKEND: &str =
    "cpu-reference-rmsnorm-bf16";
pub(in crate::commands::real_full) const CUDA_REFERENCE_RMSNORM_BF16_BACKEND: &str =
    "cuda-reference-rmsnorm-bf16";
pub(in crate::commands::real_full) const CUDA_REFERENCE_RMSNORM_BF16_RESIDENT_WEIGHT_BACKEND: &str =
    "cuda-reference-rmsnorm-bf16-resident-weight";
pub(in crate::commands::real_full) const CUDA_REFERENCE_RMSNORM_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND:
    &str = "cuda-reference-rmsnorm-bf16-preloaded-resident-weight";
pub(in crate::commands::real_full) const CUDA_REFERENCE_LAYER_NORM_AFFINE_F32_BF16_PRELOADED_RESIDENT_BACKEND:
    &str = "cuda-reference-layernorm-affine-f32-bf16-preloaded-resident-weight-bias";
pub(in crate::commands::real_full) const CUDA_REFERENCE_LAYER_NORM_AFFINE_BF16_PRELOADED_RESIDENT_BACKEND:
    &str = "cuda-reference-layernorm-affine-bf16-preloaded-resident-weight-bias";

#[allow(dead_code)]
pub(in crate::commands::real_full) fn rmsnorm_hidden(
    hidden: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<RmsNormOutput> {
    validate_rmsnorm_inputs(hidden, weight)?;
    if cuda_reference_kernels_enabled() {
        return cuda_rmsnorm_hidden(hidden, weight, eps);
    }
    Ok(cpu_rmsnorm_hidden(hidden, weight, eps))
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn rmsnorm_hidden_bf16(
    hidden_bf16: &[u8],
    weight_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    eps: f32,
) -> Result<RmsNormOutput> {
    validate_rmsnorm_bf16_inputs(hidden_bf16, weight_bf16, rows, hidden_dim)?;
    if cuda_reference_kernels_enabled() {
        return cuda_rmsnorm_hidden_bf16(hidden_bf16, weight_bf16, rows, hidden_dim, eps);
    }
    Ok(cpu_rmsnorm_hidden_bf16(
        hidden_bf16,
        weight_bf16,
        rows,
        hidden_dim,
        eps,
    ))
}

pub(in crate::commands::real_full) fn rmsnorm_hidden_bf16_resident_weight(
    weight_name: &str,
    hidden_bf16: &[u8],
    weight_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    eps: f32,
) -> Result<RmsNormOutput> {
    validate_resident_weight_name(weight_name)?;
    validate_rmsnorm_bf16_inputs(hidden_bf16, weight_bf16, rows, hidden_dim)?;
    if cuda_reference_kernels_enabled() {
        return cuda_rmsnorm_hidden_bf16_resident_weight(
            weight_name,
            hidden_bf16,
            weight_bf16,
            rows,
            hidden_dim,
            eps,
        );
    }
    Ok(cpu_rmsnorm_hidden_bf16(
        hidden_bf16,
        weight_bf16,
        rows,
        hidden_dim,
        eps,
    ))
}

pub(in crate::commands::real_full) fn rmsnorm_hidden_bf16_preloaded_resident_weight(
    weight_name: &str,
    hidden_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    eps: f32,
) -> Result<RmsNormOutput> {
    validate_resident_weight_name(weight_name)?;
    let weight_bytes =
        validate_rmsnorm_bf16_preloaded_resident_inputs(hidden_bf16, rows, hidden_dim)?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "preloaded resident BF16 RMSNorm requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1"
        );
    }
    cuda_rmsnorm_hidden_bf16_preloaded_resident_weight(
        weight_name,
        hidden_bf16,
        rows,
        hidden_dim,
        eps,
        weight_bytes,
    )
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn rmsnorm_hidden_bf16_preloaded_resident_weight_device_output(
    weight_name: &str,
    hidden_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    eps: f32,
) -> Result<DeviceBf16Output> {
    validate_resident_weight_name(weight_name)?;
    let weight_bytes =
        validate_rmsnorm_bf16_preloaded_resident_inputs(hidden_bf16, rows, hidden_dim)?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "preloaded resident BF16 RMSNorm device output requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1"
        );
    }
    cuda_rmsnorm_hidden_bf16_preloaded_resident_weight_device_output(
        weight_name,
        hidden_bf16,
        rows,
        hidden_dim,
        eps,
        weight_bytes,
    )
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output(
    weight_name: &str,
    hidden_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    eps: f32,
) -> Result<DeviceBf16Output> {
    validate_resident_weight_name(weight_name)?;
    let weight_bytes =
        validate_rmsnorm_bf16_preloaded_resident_device_input(hidden_buffer, rows, hidden_dim)?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "preloaded resident BF16 RMSNorm device-input output requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1"
        );
    }
    // The terminal norm is immediately consumed by a separately captured
    // LM-head graph.  Keep this boundary on the synchronous direct launch
    // until the cross-program graph-update path is proven safe for recycled
    // owned input/output pointers.
    if weight_name == "model.norm.weight" {
        return cuda_rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output_legacy(
            weight_name,
            hidden_buffer,
            rows,
            hidden_dim,
            eps,
            weight_bytes,
        );
    }
    cuda_rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output(
        weight_name,
        hidden_buffer,
        rows,
        hidden_dim,
        eps,
        weight_bytes,
    )
}

pub(in crate::commands::real_full) fn rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output_async(
    weight_name: &str,
    hidden_buffer: GlmrtDeviceBuffer,
    hidden_ready_event: Option<&CoordinatorCudaEvent>,
    rows: usize,
    hidden_dim: usize,
    eps: f32,
) -> Result<DeviceBf16Output> {
    validate_resident_weight_name(weight_name)?;
    let weight_bytes =
        validate_rmsnorm_bf16_preloaded_resident_device_input(hidden_buffer, rows, hidden_dim)?;
    anyhow::ensure!(
        cuda_reference_kernels_enabled(),
        "asynchronous preloaded resident BF16 RMSNorm requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1"
    );
    let graph_key = coord_layer_graph_key_for_full_hidden_rows(weight_name, rows, hidden_dim)?
        .context("asynchronous preloaded resident BF16 RMSNorm requires a graph slot")?;
    cuda_rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output_graph_slot_impl(
        &graph_key,
        weight_name,
        hidden_buffer,
        rows,
        hidden_dim,
        eps,
        weight_bytes,
        false,
        hidden_ready_event,
    )
}

pub(in crate::commands::real_full) fn layer_norm_affine_f32_bf16_preloaded_resident_weight_bias(
    weight_name: &str,
    bias_name: &str,
    values: &[f32],
    rows: usize,
    hidden_dim: usize,
    eps: f32,
) -> Result<LayerNormAffineOutput> {
    validate_resident_weight_name(weight_name)?;
    validate_resident_weight_name(bias_name)?;
    let weight_bytes =
        validate_layer_norm_affine_f32_bf16_preloaded_inputs(values, rows, hidden_dim)?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "preloaded resident BF16 affine LayerNorm requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1"
        );
    }
    cuda_layer_norm_affine_f32_bf16_preloaded_resident_weight_bias(
        weight_name,
        bias_name,
        values,
        rows,
        hidden_dim,
        eps,
        weight_bytes,
    )
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn layer_norm_affine_bf16_preloaded_resident_weight_bias_device_input_output(
    weight_name: &str,
    bias_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    eps: f32,
) -> Result<DeviceBf16Output> {
    validate_resident_weight_name(weight_name)?;
    validate_resident_weight_name(bias_name)?;
    let vector_bytes = validate_layer_norm_affine_bf16_preloaded_resident_device_input(
        input_buffer,
        rows,
        hidden_dim,
    )?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "preloaded resident BF16 affine LayerNorm device-input output requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1"
        );
    }
    cuda_layer_norm_affine_bf16_preloaded_resident_weight_bias_device_input_output(
        weight_name,
        bias_name,
        input_buffer,
        rows,
        hidden_dim,
        eps,
        vector_bytes,
    )
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn validate_rmsnorm_inputs(
    hidden: &[f32],
    weight: &[f32],
) -> Result<()> {
    if hidden.len() != weight.len() {
        anyhow::bail!(
            "real full RMSNorm hidden/weight length mismatch: {} != {}",
            hidden.len(),
            weight.len()
        );
    }
    if hidden.is_empty() {
        anyhow::bail!("real full RMSNorm hidden vector must not be empty");
    }
    Ok(())
}

pub(in crate::commands::real_full) fn validate_rmsnorm_bf16_inputs(
    hidden_bf16: &[u8],
    weight_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
) -> Result<()> {
    if rows == 0 || hidden_dim == 0 {
        anyhow::bail!(
            "real full BF16 RMSNorm requires non-zero shape, got rows={rows} hidden_dim={hidden_dim}"
        );
    }
    let hidden_bytes = rows
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("real full BF16 RMSNorm hidden shape overflows usize while validating input")?;
    if hidden_bf16.len() != hidden_bytes {
        anyhow::bail!(
            "real full BF16 RMSNorm hidden byte length mismatch: expected {} got {}",
            hidden_bytes,
            hidden_bf16.len()
        );
    }
    let weight_bytes = hidden_dim
        .checked_mul(std::mem::size_of::<u16>())
        .context("real full BF16 RMSNorm weight shape overflows usize while validating input")?;
    if weight_bf16.len() != weight_bytes {
        anyhow::bail!(
            "real full BF16 RMSNorm weight byte length mismatch: expected {} got {}",
            weight_bytes,
            weight_bf16.len()
        );
    }
    Ok(())
}

pub(in crate::commands::real_full) fn validate_rmsnorm_bf16_preloaded_resident_inputs(
    hidden_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
) -> Result<usize> {
    if rows == 0 || hidden_dim == 0 {
        anyhow::bail!(
            "real full preloaded BF16 RMSNorm requires non-zero shape, got rows={rows} hidden_dim={hidden_dim}"
        );
    }
    let hidden_bytes = rows
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full preloaded BF16 RMSNorm hidden shape overflows usize while validating input",
        )?;
    if hidden_bf16.len() != hidden_bytes {
        anyhow::bail!(
            "real full preloaded BF16 RMSNorm hidden byte length mismatch: expected {} got {}",
            hidden_bytes,
            hidden_bf16.len()
        );
    }
    hidden_dim
        .checked_mul(std::mem::size_of::<u16>())
        .context("real full preloaded BF16 RMSNorm weight shape overflows usize")
}

pub(in crate::commands::real_full) fn validate_rmsnorm_bf16_preloaded_resident_device_input(
    hidden_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
) -> Result<usize> {
    if rows == 0 || hidden_dim == 0 {
        anyhow::bail!(
            "real full preloaded BF16 RMSNorm device input requires non-zero shape, got rows={rows} hidden_dim={hidden_dim}"
        );
    }
    let hidden_bytes = rows
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full preloaded BF16 RMSNorm device input shape overflows usize while validating input",
        )?;
    if hidden_buffer.ptr.is_null() {
        anyhow::bail!("real full preloaded BF16 RMSNorm device input buffer is null");
    }
    if hidden_buffer.bytes < hidden_bytes {
        anyhow::bail!(
            "real full preloaded BF16 RMSNorm device input byte length mismatch: expected at least {} got {}",
            hidden_bytes,
            hidden_buffer.bytes
        );
    }
    hidden_dim
        .checked_mul(std::mem::size_of::<u16>())
        .context("real full preloaded BF16 RMSNorm device input weight shape overflows usize")
}

pub(in crate::commands::real_full) fn validate_layer_norm_affine_f32_bf16_preloaded_inputs(
    values: &[f32],
    rows: usize,
    hidden_dim: usize,
) -> Result<usize> {
    if rows == 0 || hidden_dim == 0 {
        anyhow::bail!(
            "real full preloaded BF16 affine LayerNorm requires non-zero shape, got rows={rows} hidden_dim={hidden_dim}"
        );
    }
    let value_count = rows.checked_mul(hidden_dim).context(
        "real full preloaded BF16 affine LayerNorm value shape overflows usize while validating input",
    )?;
    if values.len() != value_count {
        anyhow::bail!(
            "real full preloaded BF16 affine LayerNorm value length mismatch: expected {} got {}",
            value_count,
            values.len()
        );
    }
    hidden_dim
        .checked_mul(std::mem::size_of::<u16>())
        .context("real full preloaded BF16 affine LayerNorm vector byte count overflows usize")
}

pub(in crate::commands::real_full) fn validate_layer_norm_affine_bf16_preloaded_resident_device_input(
    input_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
) -> Result<usize> {
    if rows == 0 || hidden_dim == 0 {
        anyhow::bail!(
            "real full preloaded BF16 affine LayerNorm device input requires non-zero shape, got rows={rows} hidden_dim={hidden_dim}"
        );
    }
    let hidden_bytes = rows
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full preloaded BF16 affine LayerNorm device input shape overflows usize while validating input",
        )?;
    if input_buffer.ptr.is_null() {
        anyhow::bail!("real full preloaded BF16 affine LayerNorm device input buffer is null");
    }
    if input_buffer.bytes < hidden_bytes {
        anyhow::bail!(
            "real full preloaded BF16 affine LayerNorm device input byte length mismatch: expected at least {} got {}",
            hidden_bytes,
            input_buffer.bytes
        );
    }
    hidden_dim.checked_mul(std::mem::size_of::<u16>()).context(
        "real full preloaded BF16 affine LayerNorm device input vector byte count overflows usize",
    )
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn cpu_rmsnorm_hidden(
    hidden: &[f32],
    weight: &[f32],
    eps: f32,
) -> RmsNormOutput {
    let variance = hidden.iter().map(|value| value * value).sum::<f32>() / hidden.len() as f32;
    let scale = (variance + eps).sqrt().recip();
    RmsNormOutput {
        values: hidden
            .iter()
            .zip(weight.iter())
            .map(|(hidden, weight)| hidden * scale * weight)
            .collect(),
        backend: CPU_REFERENCE_RMSNORM_BACKEND,
    }
}

pub(in crate::commands::real_full) fn cpu_rmsnorm_hidden_bf16(
    hidden_bf16: &[u8],
    weight_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    eps: f32,
) -> RmsNormOutput {
    let hidden = bf16_values_to_f32(hidden_bf16);
    let weight = bf16_values_to_f32(weight_bf16);
    let mut values = vec![0.0_f32; rows * hidden_dim];
    for row in 0..rows {
        let row_start = row * hidden_dim;
        let row_end = row_start + hidden_dim;
        let variance = hidden[row_start..row_end]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            / hidden_dim as f32;
        let scale = (variance + eps).sqrt().recip();
        for col in 0..hidden_dim {
            values[row_start + col] = hidden[row_start + col] * scale * weight[col];
        }
    }
    RmsNormOutput {
        values,
        backend: CPU_REFERENCE_RMSNORM_BF16_BACKEND,
    }
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_rmsnorm_hidden(
    hidden: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<RmsNormOutput> {
    let library = cuda_native_library()?;
    let bytes = std::mem::size_of_val(hidden);
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let hidden_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        bytes,
        "RMSNorm input",
    )?;
    let weight_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::B,
        bytes,
        "RMSNorm weight",
    )?;
    let output_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        bytes,
        "RMSNorm output",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            f32_bytes(hidden),
            "RMSNorm input",
        )
        .context("copying RMSNorm input to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::B,
            f32_bytes(weight),
            "RMSNorm weight",
        )
        .context("copying RMSNorm weight to device")?;
    library
        .cuda_rmsnorm_f32(
            hidden_buffer,
            weight_buffer,
            output_buffer,
            1,
            hidden.len() as i32,
            eps,
        )
        .context("executing CUDA RMSNorm")?;
    let mut out_bytes = vec![0_u8; bytes];
    library
        .copy_d2h(&mut out_bytes, output_buffer)
        .context("copying RMSNorm output to host")?;

    Ok(RmsNormOutput {
        values: f32_vec_from_bytes(&out_bytes)?,
        backend: CUDA_REFERENCE_RMSNORM_BACKEND,
    })
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_rmsnorm_hidden_bf16(
    hidden_bf16: &[u8],
    weight_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    eps: f32,
) -> Result<RmsNormOutput> {
    let library = cuda_native_library()?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let hidden_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        hidden_bf16.len(),
        "BF16 RMSNorm input",
    )?;
    let weight_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::B,
        weight_bf16.len(),
        "BF16 RMSNorm weight",
    )?;
    let output_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        hidden_bf16.len(),
        "BF16 RMSNorm output",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            hidden_bf16,
            "BF16 RMSNorm input",
        )
        .context("copying BF16 RMSNorm input to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::B,
            weight_bf16,
            "BF16 RMSNorm weight",
        )
        .context("copying BF16 RMSNorm weight to device")?;
    library
        .cuda_rmsnorm_bf16(
            hidden_buffer,
            weight_buffer,
            output_buffer,
            rows as i32,
            hidden_dim as i32,
            eps,
        )
        .context("executing CUDA BF16 RMSNorm")?;
    let mut out_bytes = vec![0_u8; hidden_bf16.len()];
    library
        .copy_d2h(&mut out_bytes, output_buffer)
        .context("copying BF16 RMSNorm output to host")?;

    Ok(RmsNormOutput {
        values: bf16_values_to_f32(&out_bytes),
        backend: CUDA_REFERENCE_RMSNORM_BF16_BACKEND,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn rmsnorm_bf16_device_buffers_for_layer(
    layer_id: usize,
    hidden_buffer: GlmrtDeviceBuffer,
    weight_buffer: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    eps: f32,
) -> Result<&'static str> {
    validate_rmsnorm_bf16_device_buffers(
        hidden_buffer,
        weight_buffer,
        output_buffer,
        rows,
        hidden_dim,
    )?;
    let graph_key = coord_attention_graph_key_for_layer_rows(layer_id, rows)?;
    let signature = rmsnorm_graph_signature(&graph_key, hidden_dim, eps);
    match with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        capture_or_update_layer_rmsnorm_bf16_graph(
            library,
            slot,
            signature,
            hidden_buffer,
            weight_buffer,
            output_buffer,
            rows,
            hidden_dim,
            eps,
            "BF16 layer RMSNorm device-buffer",
        )?;
        unsafe {
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing BF16 layer RMSNorm device-buffer graph slot stream")?;
        }
        Ok(CUDA_REFERENCE_RMSNORM_BF16_BACKEND)
    }) {
        Ok(backend) => Ok(backend),
        Err(_error) => cuda_rmsnorm_bf16_device_buffers_direct(
            hidden_buffer,
            weight_buffer,
            output_buffer,
            rows,
            hidden_dim,
            eps,
            "BF16 layer RMSNorm device-buffer",
        ),
    }
}

fn cuda_rmsnorm_bf16_device_buffers_direct(
    hidden_buffer: GlmrtDeviceBuffer,
    weight_buffer: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    eps: f32,
    label: &str,
) -> Result<&'static str> {
    let library = cuda_native_library()?;
    library
        .cuda_rmsnorm_bf16(
            hidden_buffer,
            weight_buffer,
            output_buffer,
            rows as i32,
            hidden_dim as i32,
            eps,
        )
        .with_context(|| format!("executing CUDA {label} direct fallback"))?;
    Ok(CUDA_REFERENCE_RMSNORM_BF16_BACKEND)
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn capture_or_update_layer_rmsnorm_bf16_graph(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    signature: CoordinatorCudaGraphSignature,
    hidden_buffer: GlmrtDeviceBuffer,
    weight_buffer: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    eps: f32,
    label: &'static str,
) -> Result<()> {
    if !slot.has_captured_graph(CoordinatorCudaGraphProgram::LayerRmsNormBf16, signature) {
        slot.stream_synchronize()
            .with_context(|| format!("synchronizing {label} inputs before graph capture"))?;
        slot.capture_graph(
            library,
            CoordinatorCudaGraphProgram::LayerRmsNormBf16,
            signature,
            |library, cuda_stream, _workspace| unsafe {
                library
                    .cuda_rmsnorm_bf16_async(
                        hidden_buffer,
                        weight_buffer,
                        output_buffer,
                        rows as i32,
                        hidden_dim as i32,
                        eps,
                        cuda_stream,
                    )
                    .with_context(|| format!("capturing async CUDA {label}"))?;
                Ok(())
            },
        )?;
    } else {
        let (graph_raw, exec_raw) = slot
            .captured_graph_raw_handles(CoordinatorCudaGraphProgram::LayerRmsNormBf16, signature)
            .context("coordinator CUDA graph slot lost captured RMSNorm graph before update")?;
        unsafe {
            library
                .cuda_graph_update_rmsnorm_bf16_node(
                    graph_raw,
                    exec_raw,
                    0,
                    hidden_buffer,
                    weight_buffer,
                    output_buffer,
                    rows as i32,
                    hidden_dim as i32,
                    eps,
                )
                .with_context(|| format!("updating captured CUDA {label} graph node"))?;
        }
    }
    slot.launch_captured_graph(
        library,
        CoordinatorCudaGraphProgram::LayerRmsNormBf16,
        signature,
    )
}

pub(in crate::commands::real_full) fn validate_rmsnorm_bf16_device_buffers(
    hidden_buffer: GlmrtDeviceBuffer,
    weight_buffer: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
) -> Result<()> {
    if rows == 0 || hidden_dim == 0 {
        anyhow::bail!(
            "CUDA BF16 layer RMSNorm device-buffer requires nonzero shape, got rows={rows} hidden_dim={hidden_dim}"
        );
    }
    let hidden_bytes = rows
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 layer RMSNorm device-buffer hidden bytes overflow usize")?;
    let weight_bytes = hidden_dim
        .checked_mul(std::mem::size_of::<u16>())
        .context("CUDA BF16 layer RMSNorm device-buffer weight bytes overflow usize")?;
    let buffers = [
        ("hidden", hidden_buffer, hidden_bytes),
        ("weight", weight_buffer, weight_bytes),
        ("output", output_buffer, hidden_bytes),
    ];
    for (label, buffer, required_bytes) in buffers {
        if buffer.ptr.is_null() {
            anyhow::bail!("CUDA BF16 layer RMSNorm device-buffer {label} is null");
        }
        if buffer.bytes < required_bytes {
            anyhow::bail!(
                "CUDA BF16 layer RMSNorm device-buffer {label} has {} bytes, expected at least {required_bytes}",
                buffer.bytes
            );
        }
        if buffer.device_id != hidden_buffer.device_id {
            anyhow::bail!(
                "CUDA BF16 layer RMSNorm device-buffer {label} is on CUDA device {}, expected {}",
                buffer.device_id,
                hidden_buffer.device_id
            );
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn capture_or_update_layernorm_affine_f32_bf16_graph(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    signature: CoordinatorCudaGraphSignature,
    input_buffer: GlmrtDeviceBuffer,
    weight_buffer: GlmrtDeviceBuffer,
    bias_buffer: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
    rows_i32: i32,
    hidden_i32: i32,
    eps: f32,
    label: &'static str,
) -> Result<()> {
    if !slot.has_captured_graph(
        CoordinatorCudaGraphProgram::LayerNormAffineF32Bf16,
        signature,
    ) {
        slot.stream_synchronize()
            .with_context(|| format!("synchronizing {label} inputs before graph capture"))?;
        slot.capture_graph(
            library,
            CoordinatorCudaGraphProgram::LayerNormAffineF32Bf16,
            signature,
            |library, cuda_stream, _workspace| unsafe {
                library
                    .cuda_layernorm_affine_f32_bf16_async(
                        input_buffer,
                        weight_buffer,
                        bias_buffer,
                        output_buffer,
                        rows_i32,
                        hidden_i32,
                        eps,
                        cuda_stream,
                    )
                    .with_context(|| format!("capturing async CUDA {label}"))?;
                Ok(())
            },
        )?;
    } else {
        let (graph_raw, exec_raw) = slot
            .captured_graph_raw_handles(
                CoordinatorCudaGraphProgram::LayerNormAffineF32Bf16,
                signature,
            )
            .context(
                "coordinator CUDA graph slot lost captured F32/BF16 affine LayerNorm graph before update",
            )?;
        unsafe {
            library
                .cuda_graph_update_layernorm_affine_f32_bf16_node(
                    graph_raw,
                    exec_raw,
                    0,
                    input_buffer,
                    weight_buffer,
                    bias_buffer,
                    output_buffer,
                    rows_i32,
                    hidden_i32,
                    eps,
                )
                .with_context(|| format!("updating captured CUDA {label} graph node"))?;
        }
    }
    slot.launch_captured_graph(
        library,
        CoordinatorCudaGraphProgram::LayerNormAffineF32Bf16,
        signature,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn capture_or_update_layernorm_affine_bf16_graph(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    signature: CoordinatorCudaGraphSignature,
    input_buffer: GlmrtDeviceBuffer,
    weight_buffer: GlmrtDeviceBuffer,
    bias_buffer: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
    rows_i32: i32,
    hidden_i32: i32,
    eps: f32,
    label: &'static str,
) -> Result<()> {
    if !slot.has_captured_graph(CoordinatorCudaGraphProgram::LayerNormAffineBf16, signature) {
        slot.stream_synchronize()
            .with_context(|| format!("synchronizing {label} inputs before graph capture"))?;
        slot.capture_graph(
            library,
            CoordinatorCudaGraphProgram::LayerNormAffineBf16,
            signature,
            |library, cuda_stream, _workspace| unsafe {
                library
                    .cuda_layernorm_affine_bf16_async(
                        input_buffer,
                        weight_buffer,
                        bias_buffer,
                        output_buffer,
                        rows_i32,
                        hidden_i32,
                        eps,
                        cuda_stream,
                    )
                    .with_context(|| format!("capturing async CUDA {label}"))?;
                Ok(())
            },
        )?;
    } else {
        let (graph_raw, exec_raw) = slot
            .captured_graph_raw_handles(CoordinatorCudaGraphProgram::LayerNormAffineBf16, signature)
            .context(
                "coordinator CUDA graph slot lost captured BF16 affine LayerNorm graph before update",
            )?;
        unsafe {
            library
                .cuda_graph_update_layernorm_affine_bf16_node(
                    graph_raw,
                    exec_raw,
                    0,
                    input_buffer,
                    weight_buffer,
                    bias_buffer,
                    output_buffer,
                    rows_i32,
                    hidden_i32,
                    eps,
                )
                .with_context(|| format!("updating captured CUDA {label} graph node"))?;
        }
    }
    slot.launch_captured_graph(
        library,
        CoordinatorCudaGraphProgram::LayerNormAffineBf16,
        signature,
    )
}

pub(in crate::commands::real_full) fn rmsnorm_graph_hidden_bytes(
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

pub(in crate::commands::real_full) fn rmsnorm_graph_signature(
    graph_key: &CoordinatorGraphKey,
    hidden_dim: usize,
    eps: f32,
) -> CoordinatorCudaGraphSignature {
    CoordinatorCudaGraphSignature::rmsnorm_bf16(graph_key.row_bucket.row_capacity, hidden_dim, eps)
}

pub(in crate::commands::real_full) fn cuda_rmsnorm_hidden_bf16_resident_weight(
    weight_name: &str,
    hidden_bf16: &[u8],
    weight_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    eps: f32,
) -> Result<RmsNormOutput> {
    if let Some(graph_key) =
        coord_layer_graph_key_for_full_hidden_rows(weight_name, rows, hidden_dim)?
    {
        match cuda_rmsnorm_hidden_bf16_resident_weight_graph_slot(
            &graph_key,
            weight_name,
            hidden_bf16,
            weight_bf16,
            rows,
            hidden_dim,
            eps,
        ) {
            Ok(output) => return Ok(output),
            Err(_error) => {
                return cuda_rmsnorm_hidden_bf16_resident_weight_legacy(
                    weight_name,
                    hidden_bf16,
                    weight_bf16,
                    rows,
                    hidden_dim,
                    eps,
                );
            }
        }
    }
    cuda_rmsnorm_hidden_bf16_resident_weight_legacy(
        weight_name,
        hidden_bf16,
        weight_bf16,
        rows,
        hidden_dim,
        eps,
    )
}

pub(in crate::commands::real_full) fn cuda_rmsnorm_hidden_bf16_resident_weight_graph_slot(
    graph_key: &CoordinatorGraphKey,
    weight_name: &str,
    hidden_bf16: &[u8],
    weight_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    eps: f32,
) -> Result<RmsNormOutput> {
    let weight_buffer = resident_weight_buffer_from_registry(
        weight_name,
        weight_bf16,
        "BF16 resident RMSNorm weight",
    )?;
    let graph_hidden_bytes = rmsnorm_graph_hidden_bytes(
        graph_key,
        hidden_dim,
        "CUDA BF16 resident RMSNorm graph-slot",
    )?;
    let signature = rmsnorm_graph_signature(graph_key, hidden_dim, eps);

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let hidden_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            graph_hidden_bytes,
            "BF16 resident RMSNorm input",
        )?;
        let output_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            graph_hidden_bytes,
            "BF16 resident RMSNorm output",
        )?;

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::A,
                hidden_bf16,
                "BF16 resident RMSNorm input",
                cuda_stream,
            )
            .context("async copying BF16 resident RMSNorm input to device")?;
        capture_or_update_layer_rmsnorm_bf16_graph(
            library,
            slot,
            signature,
            hidden_buffer,
            weight_buffer,
            output_buffer,
            rows,
            hidden_dim,
            eps,
            "BF16 resident RMSNorm",
        )?;
        let mut out_bytes = vec![0_u8; hidden_bf16.len()];
        unsafe {
            library
                .copy_d2h_async(&mut out_bytes, output_buffer, cuda_stream)
                .context("async copying BF16 resident RMSNorm output to host")?;
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing BF16 resident RMSNorm graph slot stream")?;
        }

        Ok(RmsNormOutput {
            values: bf16_values_to_f32(&out_bytes),
            backend: CUDA_REFERENCE_RMSNORM_BF16_RESIDENT_WEIGHT_BACKEND,
        })
    })
}

pub(in crate::commands::real_full) fn cuda_rmsnorm_hidden_bf16_resident_weight_legacy(
    weight_name: &str,
    hidden_bf16: &[u8],
    weight_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    eps: f32,
) -> Result<RmsNormOutput> {
    let library = cuda_native_library()?;
    let weight_buffer = resident_weight_buffer_from_registry(
        weight_name,
        weight_bf16,
        "BF16 resident RMSNorm weight",
    )?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let hidden_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        hidden_bf16.len(),
        "BF16 resident RMSNorm input",
    )?;
    let output_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        hidden_bf16.len(),
        "BF16 resident RMSNorm output",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            hidden_bf16,
            "BF16 resident RMSNorm input",
        )
        .context("copying BF16 resident RMSNorm input to device")?;
    library
        .cuda_rmsnorm_bf16(
            hidden_buffer,
            weight_buffer,
            output_buffer,
            rows as i32,
            hidden_dim as i32,
            eps,
        )
        .context("executing CUDA BF16 resident RMSNorm")?;
    let mut out_bytes = vec![0_u8; hidden_bf16.len()];
    library
        .copy_d2h(&mut out_bytes, output_buffer)
        .context("copying BF16 resident RMSNorm output to host")?;

    Ok(RmsNormOutput {
        values: bf16_values_to_f32(&out_bytes),
        backend: CUDA_REFERENCE_RMSNORM_BF16_RESIDENT_WEIGHT_BACKEND,
    })
}

pub(in crate::commands::real_full) fn cuda_rmsnorm_hidden_bf16_preloaded_resident_weight(
    weight_name: &str,
    hidden_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    eps: f32,
    weight_bytes: usize,
) -> Result<RmsNormOutput> {
    if let Some(graph_key) =
        coord_layer_graph_key_for_full_hidden_rows(weight_name, rows, hidden_dim)?
    {
        match cuda_rmsnorm_hidden_bf16_preloaded_resident_weight_graph_slot(
            &graph_key,
            weight_name,
            hidden_bf16,
            rows,
            hidden_dim,
            eps,
            weight_bytes,
        ) {
            Ok(output) => return Ok(output),
            Err(_error) => {
                return cuda_rmsnorm_hidden_bf16_preloaded_resident_weight_legacy(
                    weight_name,
                    hidden_bf16,
                    rows,
                    hidden_dim,
                    eps,
                    weight_bytes,
                );
            }
        }
    }
    cuda_rmsnorm_hidden_bf16_preloaded_resident_weight_legacy(
        weight_name,
        hidden_bf16,
        rows,
        hidden_dim,
        eps,
        weight_bytes,
    )
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_rmsnorm_hidden_bf16_preloaded_resident_weight_device_output(
    weight_name: &str,
    hidden_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    eps: f32,
    weight_bytes: usize,
) -> Result<DeviceBf16Output> {
    if let Some(graph_key) =
        coord_layer_graph_key_for_full_hidden_rows(weight_name, rows, hidden_dim)?
    {
        match cuda_rmsnorm_hidden_bf16_preloaded_resident_weight_device_output_graph_slot(
            &graph_key,
            weight_name,
            hidden_bf16,
            rows,
            hidden_dim,
            eps,
            weight_bytes,
        ) {
            Ok(output) => return Ok(output),
            Err(_error) => {
                return cuda_rmsnorm_hidden_bf16_preloaded_resident_weight_device_output_legacy(
                    weight_name,
                    hidden_bf16,
                    rows,
                    hidden_dim,
                    eps,
                    weight_bytes,
                );
            }
        }
    }
    cuda_rmsnorm_hidden_bf16_preloaded_resident_weight_device_output_legacy(
        weight_name,
        hidden_bf16,
        rows,
        hidden_dim,
        eps,
        weight_bytes,
    )
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output(
    weight_name: &str,
    hidden_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    eps: f32,
    weight_bytes: usize,
) -> Result<DeviceBf16Output> {
    if let Some(graph_key) =
        coord_layer_graph_key_for_full_hidden_rows(weight_name, rows, hidden_dim)?
    {
        match cuda_rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output_graph_slot(
            &graph_key,
            weight_name,
            hidden_buffer,
            rows,
            hidden_dim,
            eps,
            weight_bytes,
        ) {
            Ok(output) => return Ok(output),
            Err(_error) => {
                return cuda_rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output_legacy(
                    weight_name,
                    hidden_buffer,
                    rows,
                    hidden_dim,
                    eps,
                    weight_bytes,
                );
            }
        }
    }
    cuda_rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output_legacy(
        weight_name,
        hidden_buffer,
        rows,
        hidden_dim,
        eps,
        weight_bytes,
    )
}

pub(in crate::commands::real_full) fn cuda_rmsnorm_hidden_bf16_preloaded_resident_weight_graph_slot(
    graph_key: &CoordinatorGraphKey,
    weight_name: &str,
    hidden_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    eps: f32,
    weight_bytes: usize,
) -> Result<RmsNormOutput> {
    let weight_buffer = preloaded_resident_weight_device_buffer(weight_name, weight_bytes)?;
    let graph_hidden_bytes = rmsnorm_graph_hidden_bytes(
        graph_key,
        hidden_dim,
        "CUDA BF16 preloaded resident RMSNorm graph-slot",
    )?;
    let signature = rmsnorm_graph_signature(graph_key, hidden_dim, eps);

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let hidden_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            graph_hidden_bytes,
            "BF16 preloaded resident RMSNorm input",
        )?;
        let output_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            graph_hidden_bytes,
            "BF16 preloaded resident RMSNorm output",
        )?;

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::A,
                hidden_bf16,
                "BF16 preloaded resident RMSNorm input",
                cuda_stream,
            )
            .context("async copying BF16 preloaded resident RMSNorm input to device")?;
        capture_or_update_layer_rmsnorm_bf16_graph(
            library,
            slot,
            signature,
            hidden_buffer,
            weight_buffer,
            output_buffer,
            rows,
            hidden_dim,
            eps,
            "BF16 preloaded resident RMSNorm",
        )?;
        let mut out_bytes = vec![0_u8; hidden_bf16.len()];
        unsafe {
            library
                .copy_d2h_async(&mut out_bytes, output_buffer, cuda_stream)
                .context("async copying BF16 preloaded resident RMSNorm output to host")?;
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing BF16 preloaded resident RMSNorm graph slot stream")?;
        }

        Ok(RmsNormOutput {
            values: bf16_values_to_f32(&out_bytes),
            backend: CUDA_REFERENCE_RMSNORM_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
        })
    })
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_rmsnorm_hidden_bf16_preloaded_resident_weight_device_output_graph_slot(
    graph_key: &CoordinatorGraphKey,
    weight_name: &str,
    hidden_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    eps: f32,
    weight_bytes: usize,
) -> Result<DeviceBf16Output> {
    let weight_buffer = preloaded_resident_weight_device_buffer(weight_name, weight_bytes)?;
    let graph_hidden_bytes = rmsnorm_graph_hidden_bytes(
        graph_key,
        hidden_dim,
        "CUDA BF16 preloaded resident RMSNorm device-output graph-slot",
    )?;
    let signature = rmsnorm_graph_signature(graph_key, hidden_dim, eps);

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let hidden_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            graph_hidden_bytes,
            "BF16 preloaded resident RMSNorm device-output input",
        )?;
        let output_buffer = OwnedCoordinatorDeviceBuffer::new(
            library,
            hidden_bf16.len(),
            "BF16 preloaded resident RMSNorm device output",
        )?;

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::A,
                hidden_bf16,
                "BF16 preloaded resident RMSNorm device-output input",
                cuda_stream,
            )
            .context(
                "async copying BF16 preloaded resident RMSNorm device-output input to device",
            )?;
        capture_or_update_layer_rmsnorm_bf16_graph(
            library,
            slot,
            signature,
            hidden_buffer,
            weight_buffer,
            output_buffer.buffer,
            rows,
            hidden_dim,
            eps,
            "BF16 preloaded resident RMSNorm device-output",
        )?;
        unsafe {
            library.cuda_stream_synchronize(cuda_stream).context(
                "synchronizing BF16 preloaded resident RMSNorm device-output graph slot stream",
            )?;
        }

        Ok(DeviceBf16Output {
            buffer: output_buffer,
            bytes: hidden_bf16.len(),
            rows,
            values_per_row: hidden_dim,
            backend: CUDA_REFERENCE_RMSNORM_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
        })
    })
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output_graph_slot(
    graph_key: &CoordinatorGraphKey,
    weight_name: &str,
    hidden_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    eps: f32,
    weight_bytes: usize,
) -> Result<DeviceBf16Output> {
    cuda_rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output_graph_slot_impl(
        graph_key,
        weight_name,
        hidden_buffer,
        rows,
        hidden_dim,
        eps,
        weight_bytes,
        true,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn cuda_rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output_graph_slot_impl(
    graph_key: &CoordinatorGraphKey,
    weight_name: &str,
    hidden_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    eps: f32,
    weight_bytes: usize,
    synchronize: bool,
    hidden_ready_event: Option<&CoordinatorCudaEvent>,
) -> Result<DeviceBf16Output> {
    let hidden_bytes = rows
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "CUDA BF16 preloaded resident RMSNorm device-input output shape overflows usize",
        )?;
    let weight_buffer = preloaded_resident_weight_device_buffer(weight_name, weight_bytes)?;
    if hidden_buffer.device_id != weight_buffer.device_id {
        anyhow::bail!(
            "CUDA BF16 preloaded resident RMSNorm device-input buffers are on different devices: input={} weight={}",
            hidden_buffer.device_id,
            weight_buffer.device_id
        );
    }
    let signature = rmsnorm_graph_signature(graph_key, hidden_dim, eps);

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        if let Some(hidden_ready_event) = hidden_ready_event {
            hidden_ready_event
                .wait_on_stream(cuda_stream)
                .context("waiting for asynchronous RMSNorm device input")?;
        }
        let output_buffer = OwnedCoordinatorDeviceBuffer::new(
            library,
            hidden_bytes,
            "BF16 preloaded resident RMSNorm device-input output",
        )?;
        capture_or_update_layer_rmsnorm_bf16_graph(
            library,
            slot,
            signature,
            hidden_buffer,
            weight_buffer,
            output_buffer.buffer,
            rows,
            hidden_dim,
            eps,
            "BF16 preloaded resident RMSNorm device-input output",
        )?;
        let ready_event = if synchronize {
            unsafe {
                library.cuda_stream_synchronize(cuda_stream).context(
                    "synchronizing BF16 preloaded resident RMSNorm device-input output graph slot stream",
                )?;
            }
            None
        } else {
            Some(slot.record_output_ready_event(library)?)
        };

        let mut output = DeviceBf16Output {
            buffer: output_buffer,
            bytes: hidden_bytes,
            rows,
            values_per_row: hidden_dim,
            backend: CUDA_REFERENCE_RMSNORM_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
        };
        if let Some(ready_event) = ready_event {
            output.set_ready_event(ready_event);
        }
        Ok(output)
    })
}

pub(in crate::commands::real_full) fn cuda_rmsnorm_hidden_bf16_preloaded_resident_weight_legacy(
    weight_name: &str,
    hidden_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    eps: f32,
    weight_bytes: usize,
) -> Result<RmsNormOutput> {
    let library = cuda_native_library()?;
    let weight_buffer = preloaded_resident_weight_device_buffer(weight_name, weight_bytes)?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let hidden_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        hidden_bf16.len(),
        "BF16 preloaded resident RMSNorm input",
    )?;
    let output_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        hidden_bf16.len(),
        "BF16 preloaded resident RMSNorm output",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            hidden_bf16,
            "BF16 preloaded resident RMSNorm input",
        )
        .context("copying BF16 preloaded resident RMSNorm input to device")?;
    library
        .cuda_rmsnorm_bf16(
            hidden_buffer,
            weight_buffer,
            output_buffer,
            rows as i32,
            hidden_dim as i32,
            eps,
        )
        .context("executing CUDA BF16 preloaded resident RMSNorm")?;
    let mut out_bytes = vec![0_u8; hidden_bf16.len()];
    library
        .copy_d2h(&mut out_bytes, output_buffer)
        .context("copying BF16 preloaded resident RMSNorm output to host")?;

    Ok(RmsNormOutput {
        values: bf16_values_to_f32(&out_bytes),
        backend: CUDA_REFERENCE_RMSNORM_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
    })
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_rmsnorm_hidden_bf16_preloaded_resident_weight_device_output_legacy(
    weight_name: &str,
    hidden_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    eps: f32,
    weight_bytes: usize,
) -> Result<DeviceBf16Output> {
    let library = cuda_native_library()?;
    let weight_buffer = preloaded_resident_weight_device_buffer(weight_name, weight_bytes)?;
    let output_buffer = OwnedCoordinatorDeviceBuffer::new(
        library,
        hidden_bf16.len(),
        "BF16 preloaded resident RMSNorm device output",
    )?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let hidden_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        hidden_bf16.len(),
        "BF16 preloaded resident RMSNorm device-output input",
    )?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            hidden_bf16,
            "BF16 preloaded resident RMSNorm device-output input",
        )
        .context("copying BF16 preloaded resident RMSNorm device-output input to device")?;
    library
        .cuda_rmsnorm_bf16(
            hidden_buffer,
            weight_buffer,
            output_buffer.buffer,
            rows as i32,
            hidden_dim as i32,
            eps,
        )
        .context("executing CUDA BF16 preloaded resident RMSNorm device output")?;

    Ok(DeviceBf16Output {
        buffer: output_buffer,
        bytes: hidden_bf16.len(),
        rows,
        values_per_row: hidden_dim,
        backend: CUDA_REFERENCE_RMSNORM_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
    })
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output_legacy(
    weight_name: &str,
    hidden_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    eps: f32,
    weight_bytes: usize,
) -> Result<DeviceBf16Output> {
    let library = cuda_native_library()?;
    let hidden_bytes = rows
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "CUDA BF16 preloaded resident RMSNorm device-input output shape overflows usize",
        )?;
    let weight_buffer = preloaded_resident_weight_device_buffer(weight_name, weight_bytes)?;
    if hidden_buffer.device_id != weight_buffer.device_id {
        anyhow::bail!(
            "CUDA BF16 preloaded resident RMSNorm device-input buffers are on different devices: input={} weight={}",
            hidden_buffer.device_id,
            weight_buffer.device_id
        );
    }
    let output_buffer = OwnedCoordinatorDeviceBuffer::new(
        library,
        hidden_bytes,
        "BF16 preloaded resident RMSNorm device-input output",
    )?;
    library
        .cuda_rmsnorm_bf16(
            hidden_buffer,
            weight_buffer,
            output_buffer.buffer,
            rows as i32,
            hidden_dim as i32,
            eps,
        )
        .context("executing CUDA BF16 preloaded resident RMSNorm device-input output")?;

    Ok(DeviceBf16Output {
        buffer: output_buffer,
        bytes: hidden_bytes,
        rows,
        values_per_row: hidden_dim,
        backend: CUDA_REFERENCE_RMSNORM_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
    })
}

pub(in crate::commands::real_full) fn cuda_layer_norm_affine_f32_bf16_preloaded_resident_weight_bias(
    weight_name: &str,
    bias_name: &str,
    values: &[f32],
    rows: usize,
    hidden_dim: usize,
    eps: f32,
    vector_bytes: usize,
) -> Result<LayerNormAffineOutput> {
    if let Some(graph_key) =
        coord_layer_graph_key_for_dsa_k_norm_names(weight_name, bias_name, rows)?
    {
        return cuda_layer_norm_affine_f32_bf16_preloaded_resident_weight_bias_graph_slot(
            &graph_key,
            weight_name,
            bias_name,
            values,
            rows,
            hidden_dim,
            eps,
            vector_bytes,
        );
    }
    cuda_layer_norm_affine_f32_bf16_preloaded_resident_weight_bias_legacy(
        weight_name,
        bias_name,
        values,
        rows,
        hidden_dim,
        eps,
        vector_bytes,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_layer_norm_affine_f32_bf16_preloaded_resident_weight_bias_graph_slot(
    graph_key: &CoordinatorGraphKey,
    weight_name: &str,
    bias_name: &str,
    values: &[f32],
    rows: usize,
    hidden_dim: usize,
    eps: f32,
    vector_bytes: usize,
) -> Result<LayerNormAffineOutput> {
    let rows_i32 =
        i32::try_from(rows).context("CUDA affine LayerNorm row count does not fit i32")?;
    let hidden_i32 =
        i32::try_from(hidden_dim).context("CUDA affine LayerNorm hidden dim does not fit i32")?;
    let value_bytes = std::mem::size_of_val(values);
    let weight_buffer = preloaded_resident_weight_device_buffer(weight_name, vector_bytes)?;
    let bias_buffer = preloaded_resident_weight_device_buffer(bias_name, vector_bytes)?;
    let graph_value_bytes = layernorm_affine_graph_value_bytes(
        graph_key,
        hidden_dim,
        std::mem::size_of::<f32>(),
        "CUDA F32/BF16 preloaded resident affine LayerNorm graph-slot",
    )?;

    let signature = layernorm_affine_graph_signature(graph_key, hidden_dim, eps);
    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let input_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            graph_value_bytes,
            "F32 preloaded resident affine LayerNorm input",
        )?;
        let output_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            graph_value_bytes,
            "F32 preloaded resident affine LayerNorm output",
        )?;
        let cuda_stream = slot.stream_ptr();

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::A,
                f32_bytes(values),
                "F32 preloaded resident affine LayerNorm input",
                cuda_stream,
            )
            .context("async copying F32 preloaded resident affine LayerNorm input to device")?;
        capture_or_update_layernorm_affine_f32_bf16_graph(
            library,
            slot,
            signature,
            input_buffer,
            weight_buffer,
            bias_buffer,
            output_buffer,
            rows_i32,
            hidden_i32,
            eps,
            "F32/BF16 preloaded resident affine LayerNorm",
        )
        .context("executing captured CUDA F32/BF16 preloaded resident affine LayerNorm graph")?;
        let mut out_bytes = vec![0_u8; value_bytes];
        unsafe {
            library
                .copy_d2h_async(&mut out_bytes, output_buffer, cuda_stream)
                .context("async copying F32 preloaded resident affine LayerNorm output to host")?;
            library.cuda_stream_synchronize(cuda_stream).context(
                "synchronizing F32/BF16 preloaded resident affine LayerNorm graph slot stream",
            )?;
        }

        Ok(LayerNormAffineOutput {
            values: f32_vec_from_bytes(&out_bytes)?,
            backend: CUDA_REFERENCE_LAYER_NORM_AFFINE_F32_BF16_PRELOADED_RESIDENT_BACKEND,
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_layer_norm_affine_f32_bf16_preloaded_resident_weight_bias_legacy(
    weight_name: &str,
    bias_name: &str,
    values: &[f32],
    rows: usize,
    hidden_dim: usize,
    eps: f32,
    vector_bytes: usize,
) -> Result<LayerNormAffineOutput> {
    let library = cuda_native_library()?;
    let rows_i32 =
        i32::try_from(rows).context("CUDA affine LayerNorm row count does not fit i32")?;
    let hidden_i32 =
        i32::try_from(hidden_dim).context("CUDA affine LayerNorm hidden dim does not fit i32")?;
    let value_bytes = std::mem::size_of_val(values);
    let weight_buffer = preloaded_resident_weight_device_buffer(weight_name, vector_bytes)?;
    let bias_buffer = preloaded_resident_weight_device_buffer(bias_name, vector_bytes)?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let input_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        value_bytes,
        "F32 preloaded resident affine LayerNorm input",
    )?;
    let output_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        value_bytes,
        "F32 preloaded resident affine LayerNorm output",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            f32_bytes(values),
            "F32 preloaded resident affine LayerNorm input",
        )
        .context("copying F32 preloaded resident affine LayerNorm input to device")?;
    library
        .cuda_layernorm_affine_f32_bf16(
            input_buffer,
            weight_buffer,
            bias_buffer,
            output_buffer,
            rows_i32,
            hidden_i32,
            eps,
        )
        .context("executing CUDA F32/BF16 preloaded resident affine LayerNorm")?;
    let mut out_bytes = vec![0_u8; value_bytes];
    library
        .copy_d2h(&mut out_bytes, output_buffer)
        .context("copying F32 preloaded resident affine LayerNorm output to host")?;

    Ok(LayerNormAffineOutput {
        values: f32_vec_from_bytes(&out_bytes)?,
        backend: CUDA_REFERENCE_LAYER_NORM_AFFINE_F32_BF16_PRELOADED_RESIDENT_BACKEND,
    })
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_layer_norm_affine_bf16_preloaded_resident_weight_bias_device_input_output(
    weight_name: &str,
    bias_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    eps: f32,
    vector_bytes: usize,
) -> Result<DeviceBf16Output> {
    if let Some(graph_key) =
        coord_layer_graph_key_for_dsa_k_norm_names(weight_name, bias_name, rows)?
    {
        match cuda_layer_norm_affine_bf16_preloaded_resident_weight_bias_device_input_output_graph_slot(
            &graph_key,
            weight_name,
            bias_name,
            input_buffer,
            rows,
            hidden_dim,
            eps,
            vector_bytes,
        ) {
            Ok(output) => return Ok(output),
            Err(_error) => {
                return cuda_layer_norm_affine_bf16_preloaded_resident_weight_bias_device_input_output_legacy(
                    weight_name,
                    bias_name,
                    input_buffer,
                    rows,
                    hidden_dim,
                    eps,
                    vector_bytes,
                );
            }
        }
    }
    cuda_layer_norm_affine_bf16_preloaded_resident_weight_bias_device_input_output_legacy(
        weight_name,
        bias_name,
        input_buffer,
        rows,
        hidden_dim,
        eps,
        vector_bytes,
    )
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_layer_norm_affine_bf16_preloaded_resident_weight_bias_device_input_output_graph_slot(
    graph_key: &CoordinatorGraphKey,
    weight_name: &str,
    bias_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    eps: f32,
    vector_bytes: usize,
) -> Result<DeviceBf16Output> {
    let rows_i32 =
        i32::try_from(rows).context("CUDA BF16 affine LayerNorm row count does not fit i32")?;
    let hidden_i32 = i32::try_from(hidden_dim)
        .context("CUDA BF16 affine LayerNorm hidden dim does not fit i32")?;
    let output_bytes = rows
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "CUDA BF16 preloaded resident affine LayerNorm device-output shape overflows usize",
        )?;
    let weight_buffer = preloaded_resident_weight_device_buffer(weight_name, vector_bytes)?;
    let bias_buffer = preloaded_resident_weight_device_buffer(bias_name, vector_bytes)?;
    if input_buffer.device_id != weight_buffer.device_id
        || input_buffer.device_id != bias_buffer.device_id
    {
        anyhow::bail!(
            "CUDA BF16 preloaded resident affine LayerNorm device-input buffers are on different devices: input={} weight={} bias={}",
            input_buffer.device_id,
            weight_buffer.device_id,
            bias_buffer.device_id
        );
    }

    let signature = layernorm_affine_graph_signature(graph_key, hidden_dim, eps);
    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let output_buffer = OwnedCoordinatorDeviceBuffer::new(
            library,
            output_bytes,
            "BF16 preloaded resident affine LayerNorm device-input output",
        )?;
        capture_or_update_layernorm_affine_bf16_graph(
            library,
            slot,
            signature,
            input_buffer,
            weight_buffer,
            bias_buffer,
            output_buffer.buffer,
            rows_i32,
            hidden_i32,
            eps,
            "BF16 preloaded resident affine LayerNorm device-input output",
        )
        .context(
            "executing captured CUDA BF16 preloaded resident affine LayerNorm device-input output graph",
        )?;
        unsafe {
            library.cuda_stream_synchronize(cuda_stream).context(
                "synchronizing BF16 preloaded resident affine LayerNorm device-input output graph slot stream",
            )?;
        }

        Ok(DeviceBf16Output {
            buffer: output_buffer,
            bytes: output_bytes,
            rows,
            values_per_row: hidden_dim,
            backend: CUDA_REFERENCE_LAYER_NORM_AFFINE_BF16_PRELOADED_RESIDENT_BACKEND,
        })
    })
}

pub(in crate::commands::real_full) fn layernorm_affine_graph_value_bytes(
    graph_key: &CoordinatorGraphKey,
    hidden_dim: usize,
    element_size: usize,
    context: &str,
) -> Result<usize> {
    graph_key
        .row_bucket
        .row_capacity
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(element_size))
        .with_context(|| format!("{context} graph buffer bytes overflow usize"))
}

pub(in crate::commands::real_full) fn layernorm_affine_graph_signature(
    graph_key: &CoordinatorGraphKey,
    hidden_dim: usize,
    eps: f32,
) -> CoordinatorCudaGraphSignature {
    CoordinatorCudaGraphSignature::layernorm_affine(
        graph_key.row_bucket.row_capacity,
        hidden_dim,
        eps,
    )
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_layer_norm_affine_bf16_preloaded_resident_weight_bias_device_input_output_legacy(
    weight_name: &str,
    bias_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    eps: f32,
    vector_bytes: usize,
) -> Result<DeviceBf16Output> {
    let library = cuda_native_library()?;
    let rows_i32 =
        i32::try_from(rows).context("CUDA BF16 affine LayerNorm row count does not fit i32")?;
    let hidden_i32 = i32::try_from(hidden_dim)
        .context("CUDA BF16 affine LayerNorm hidden dim does not fit i32")?;
    let output_bytes = rows
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "CUDA BF16 preloaded resident affine LayerNorm device-output shape overflows usize",
        )?;
    let weight_buffer = preloaded_resident_weight_device_buffer(weight_name, vector_bytes)?;
    let bias_buffer = preloaded_resident_weight_device_buffer(bias_name, vector_bytes)?;
    if input_buffer.device_id != weight_buffer.device_id
        || input_buffer.device_id != bias_buffer.device_id
    {
        anyhow::bail!(
            "CUDA BF16 preloaded resident affine LayerNorm device-input buffers are on different devices: input={} weight={} bias={}",
            input_buffer.device_id,
            weight_buffer.device_id,
            bias_buffer.device_id
        );
    }
    let output_buffer = OwnedCoordinatorDeviceBuffer::new(
        library,
        output_bytes,
        "BF16 preloaded resident affine LayerNorm device-input output",
    )?;
    library
        .cuda_layernorm_affine_bf16(
            input_buffer,
            weight_buffer,
            bias_buffer,
            output_buffer.buffer,
            rows_i32,
            hidden_i32,
            eps,
        )
        .context("executing CUDA BF16 preloaded resident affine LayerNorm device-input output")?;

    Ok(DeviceBf16Output {
        buffer: output_buffer,
        bytes: output_bytes,
        rows,
        values_per_row: hidden_dim,
        backend: CUDA_REFERENCE_LAYER_NORM_AFFINE_BF16_PRELOADED_RESIDENT_BACKEND,
    })
}
