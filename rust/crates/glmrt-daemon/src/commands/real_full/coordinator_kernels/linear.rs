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

pub(in crate::commands::real_full) const CPU_REFERENCE_LINEAR_BACKEND: &str =
    "cpu-reference-linear";
pub(in crate::commands::real_full) const CUDA_REFERENCE_LINEAR_BACKEND: &str =
    "cuda-reference-linear-f32";
pub(in crate::commands::real_full) const CPU_REFERENCE_LINEAR_BF16_BACKEND: &str =
    "cpu-reference-linear-bf16";
pub(in crate::commands::real_full) const CUDA_REFERENCE_LINEAR_BF16_BACKEND: &str =
    "cuda-reference-linear-bf16";
pub(in crate::commands::real_full) const CUDA_REFERENCE_LINEAR_BF16_RESIDENT_WEIGHT_BACKEND: &str =
    "cuda-reference-linear-bf16-resident-weight";
pub(in crate::commands::real_full) const CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND:
    &str = "cuda-reference-linear-bf16-preloaded-resident-weight";
pub(in crate::commands::real_full) const CUDA_REFERENCE_LINEAR_BF16_M1_PARITY_BATCHED_BACKEND:
    &str = "cuda-reference-linear-bf16-m1-parity-batched";

pub(in crate::commands::real_full) fn linear_rows(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> Result<LinearOutput> {
    validate_linear_inputs(input, weight, bias, rows, input_dim, output_dim)?;
    if cuda_reference_kernels_enabled() {
        return cuda_linear_rows(input, weight, bias, rows, input_dim, output_dim);
    }
    Ok(cpu_linear_rows(
        input, weight, bias, rows, input_dim, output_dim,
    ))
}

pub(in crate::commands::real_full) fn linear_rows_bf16(
    input_bf16: &[u8],
    weight_bf16: &[u8],
    bias_bf16: Option<&[u8]>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> Result<LinearOutput> {
    validate_linear_bf16_inputs(
        input_bf16,
        weight_bf16,
        bias_bf16,
        rows,
        input_dim,
        output_dim,
    )?;
    if cuda_reference_kernels_enabled() {
        return cuda_linear_rows_bf16(
            input_bf16,
            weight_bf16,
            bias_bf16,
            rows,
            input_dim,
            output_dim,
        );
    }
    Ok(cpu_linear_rows_bf16(
        input_bf16,
        weight_bf16,
        bias_bf16,
        rows,
        input_dim,
        output_dim,
    ))
}

pub(in crate::commands::real_full) fn linear_rows_bf16_resident_weight(
    weight_name: &str,
    input_bf16: &[u8],
    weight_bf16: &[u8],
    bias_bf16: Option<&[u8]>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> Result<LinearOutput> {
    validate_resident_weight_name(weight_name)?;
    validate_linear_bf16_inputs(
        input_bf16,
        weight_bf16,
        bias_bf16,
        rows,
        input_dim,
        output_dim,
    )?;
    if cuda_reference_kernels_enabled() {
        return cuda_linear_rows_bf16_resident_weight(
            weight_name,
            input_bf16,
            weight_bf16,
            bias_bf16,
            rows,
            input_dim,
            output_dim,
        );
    }
    Ok(cpu_linear_rows_bf16(
        input_bf16,
        weight_bf16,
        bias_bf16,
        rows,
        input_dim,
        output_dim,
    ))
}

pub(in crate::commands::real_full) fn linear_rows_bf16_preloaded_resident_weight(
    weight_name: &str,
    input_bf16: &[u8],
    bias_bf16: Option<&[u8]>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    full_output_dim: usize,
) -> Result<LinearOutput> {
    validate_resident_weight_name(weight_name)?;
    let view = validate_linear_bf16_preloaded_resident_inputs(
        input_bf16,
        bias_bf16,
        rows,
        input_dim,
        output_dim,
        full_output_dim,
    )?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "preloaded resident BF16 linear requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1"
        );
    }
    cuda_linear_rows_bf16_preloaded_resident_weight(
        weight_name,
        input_bf16,
        bias_bf16,
        rows,
        input_dim,
        output_dim,
        view,
    )
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn linear_rows_bf16_preloaded_resident_weight_device_input(
    weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    bias_bf16: Option<&[u8]>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    full_output_dim: usize,
) -> Result<LinearOutput> {
    validate_resident_weight_name(weight_name)?;
    let view = validate_linear_bf16_preloaded_resident_device_input(
        input_buffer,
        bias_bf16,
        rows,
        input_dim,
        output_dim,
        full_output_dim,
    )?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "preloaded resident BF16 device-input linear requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1"
        );
    }
    cuda_linear_rows_bf16_preloaded_resident_weight_device_input(
        weight_name,
        input_buffer,
        bias_bf16,
        rows,
        input_dim,
        output_dim,
        view,
    )
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn linear_rows_bf16_preloaded_resident_weight_device_output(
    weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    bias_bf16: Option<&[u8]>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    full_output_dim: usize,
) -> Result<DeviceBf16Output> {
    validate_resident_weight_name(weight_name)?;
    let view = validate_linear_bf16_preloaded_resident_device_input(
        input_buffer,
        bias_bf16,
        rows,
        input_dim,
        output_dim,
        full_output_dim,
    )?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "preloaded resident BF16 device-output linear requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1"
        );
    }
    cuda_linear_rows_bf16_preloaded_resident_weight_device_output(
        weight_name,
        input_buffer,
        bias_bf16,
        rows,
        input_dim,
        output_dim,
        view,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn linear_rows_bf16_m1_parity_batched_preloaded_resident_weight_device_output(
    weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    full_output_dim: usize,
) -> Result<DeviceBf16Output> {
    anyhow::ensure!(
        (2..=16).contains(&rows),
        "M1-parity BF16 resident projection requires 2..=16 rows"
    );
    validate_resident_weight_name(weight_name)?;
    let view = validate_linear_bf16_preloaded_resident_device_input(
        input_buffer,
        None,
        rows,
        input_dim,
        output_dim,
        full_output_dim,
    )?;
    anyhow::ensure!(
        cuda_reference_kernels_enabled(),
        "M1-parity BF16 resident projection requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1"
    );
    let graph_key = coord_linear_graph_key_for_weight_name(weight_name, rows)?
        .with_context(|| format!("selecting M1-parity graph slot for {weight_name}"))?;
    let output_bytes = rows
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("M1-parity BF16 resident projection output bytes overflow")?;
    let weight_buffer = preloaded_resident_weight_device_buffer_view(
        weight_name,
        view.full_bytes,
        view.offset_bytes,
        view.view_bytes,
    )?;
    anyhow::ensure!(
        input_buffer.device_id == weight_buffer.device_id,
        "M1-parity BF16 resident projection buffers are on different devices: input={} weight={}",
        input_buffer.device_id,
        weight_buffer.device_id
    );

    with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
        let output = OwnedCoordinatorDeviceBuffer::new(
            library,
            output_bytes,
            "M1-parity BF16 resident projection output",
        )?;
        let stream = slot.stream_ptr();
        unsafe {
            library
                .cuda_linear_bf16_m1_parity_batched_cublaslt_async(
                    input_buffer,
                    weight_buffer,
                    output.buffer,
                    rows,
                    input_dim,
                    output_dim,
                    stream,
                )
                .with_context(|| format!("launching M1-parity BF16 projection {weight_name}"))?;
            library.cuda_stream_synchronize(stream).with_context(|| {
                format!("synchronizing M1-parity BF16 projection {weight_name}")
            })?;
        }
        Ok(DeviceBf16Output {
            buffer: output,
            bytes: output_bytes,
            rows,
            values_per_row: output_dim,
            backend: CUDA_REFERENCE_LINEAR_BF16_M1_PARITY_BATCHED_BACKEND,
        })
    })
}

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn linear_rows_bf16_preloaded_resident_weight_padded_device_input(
    weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    bias_bf16: Option<&[u8]>,
    rows: usize,
    active_input_dim: usize,
    full_input_dim: usize,
    output_dim: usize,
    full_output_dim: usize,
) -> Result<LinearOutput> {
    validate_resident_weight_name(weight_name)?;
    let view = validate_linear_bf16_preloaded_resident_padded_device_input(
        input_buffer,
        bias_bf16,
        rows,
        active_input_dim,
        full_input_dim,
        output_dim,
        full_output_dim,
    )?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "preloaded resident BF16 padded device-input linear requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1"
        );
    }
    cuda_linear_rows_bf16_preloaded_resident_weight_padded_device_input(
        weight_name,
        input_buffer,
        bias_bf16,
        rows,
        full_input_dim,
        output_dim,
        view,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn linear_residual_add_rows_bf16_preloaded_resident_weight_device_input(
    weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    bias_bf16: Option<&[u8]>,
    residual_bf16: &[u8],
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    full_output_dim: usize,
) -> Result<LinearResidualAddOutput> {
    validate_resident_weight_name(weight_name)?;
    let view = validate_linear_bf16_preloaded_resident_device_input(
        input_buffer,
        bias_bf16,
        rows,
        input_dim,
        output_dim,
        full_output_dim,
    )?;
    validate_linear_residual_add_bf16_residual(
        residual_bf16,
        rows,
        output_dim,
        "preloaded resident BF16 device-input linear residual-add",
    )?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "preloaded resident BF16 device-input linear residual-add requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1"
        );
    }
    cuda_linear_residual_add_rows_bf16_preloaded_resident_weight_device_input(
        weight_name,
        input_buffer,
        bias_bf16,
        residual_bf16,
        rows,
        input_dim,
        output_dim,
        view,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn linear_residual_add_rows_bf16_preloaded_resident_weight_device_input_device_output(
    weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    bias_bf16: Option<&[u8]>,
    residual_bf16: &[u8],
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    full_output_dim: usize,
) -> Result<LinearResidualAddDeviceOutput> {
    validate_resident_weight_name(weight_name)?;
    let view = validate_linear_bf16_preloaded_resident_device_input(
        input_buffer,
        bias_bf16,
        rows,
        input_dim,
        output_dim,
        full_output_dim,
    )?;
    validate_linear_residual_add_bf16_residual(
        residual_bf16,
        rows,
        output_dim,
        "preloaded resident BF16 device-input linear residual-add device-output",
    )?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "preloaded resident BF16 device-input linear residual-add device-output requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1"
        );
    }
    cuda_linear_residual_add_rows_bf16_preloaded_resident_weight_device_input_device_output(
        weight_name,
        input_buffer,
        bias_bf16,
        residual_bf16,
        rows,
        input_dim,
        output_dim,
        view,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn linear_residual_add_rows_bf16_preloaded_resident_weight_padded_device_input(
    weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    bias_bf16: Option<&[u8]>,
    residual_bf16: &[u8],
    rows: usize,
    active_input_dim: usize,
    full_input_dim: usize,
    output_dim: usize,
    full_output_dim: usize,
) -> Result<LinearResidualAddOutput> {
    validate_resident_weight_name(weight_name)?;
    let view = validate_linear_bf16_preloaded_resident_padded_device_input(
        input_buffer,
        bias_bf16,
        rows,
        active_input_dim,
        full_input_dim,
        output_dim,
        full_output_dim,
    )?;
    validate_linear_residual_add_bf16_residual(
        residual_bf16,
        rows,
        output_dim,
        "preloaded resident BF16 padded device-input linear residual-add",
    )?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "preloaded resident BF16 padded device-input linear residual-add requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1"
        );
    }
    cuda_linear_residual_add_rows_bf16_preloaded_resident_weight_padded_device_input(
        weight_name,
        input_buffer,
        bias_bf16,
        residual_bf16,
        rows,
        full_input_dim,
        output_dim,
        view,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn linear_residual_add_rows_bf16_preloaded_resident_weight_padded_device_input_device_output(
    weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    bias_bf16: Option<&[u8]>,
    residual_bf16: &[u8],
    rows: usize,
    active_input_dim: usize,
    full_input_dim: usize,
    output_dim: usize,
    full_output_dim: usize,
) -> Result<LinearResidualAddDeviceOutput> {
    validate_resident_weight_name(weight_name)?;
    let view = validate_linear_bf16_preloaded_resident_padded_device_input(
        input_buffer,
        bias_bf16,
        rows,
        active_input_dim,
        full_input_dim,
        output_dim,
        full_output_dim,
    )?;
    validate_linear_residual_add_bf16_residual(
        residual_bf16,
        rows,
        output_dim,
        "preloaded resident BF16 padded device-input linear residual-add device-output",
    )?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "preloaded resident BF16 padded device-input linear residual-add device-output requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1"
        );
    }
    cuda_linear_residual_add_rows_bf16_preloaded_resident_weight_padded_device_input_device_output(
        weight_name,
        input_buffer,
        bias_bf16,
        residual_bf16,
        rows,
        full_input_dim,
        output_dim,
        view,
    )
}

pub(in crate::commands::real_full) fn validate_linear_inputs(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> Result<()> {
    if rows == 0 || input_dim == 0 || output_dim == 0 {
        anyhow::bail!(
            "real full linear requires non-zero shape, got rows={rows} input_dim={input_dim} output_dim={output_dim}"
        );
    }
    let expected_input = rows.checked_mul(input_dim).context(
        "real full linear input shape overflows usize while validating coordinator kernel input",
    )?;
    if input.len() != expected_input {
        anyhow::bail!(
            "real full linear input length mismatch: expected {} got {}",
            expected_input,
            input.len()
        );
    }
    let expected_weight = output_dim.checked_mul(input_dim).context(
        "real full linear weight shape overflows usize while validating coordinator kernel input",
    )?;
    if weight.len() != expected_weight {
        anyhow::bail!(
            "real full linear weight length mismatch: expected {} got {}",
            expected_weight,
            weight.len()
        );
    }
    if let Some(bias) = bias {
        if bias.len() != output_dim {
            anyhow::bail!(
                "real full linear bias length mismatch: expected {} got {}",
                output_dim,
                bias.len()
            );
        }
    }
    Ok(())
}

pub(in crate::commands::real_full) fn validate_linear_bf16_inputs(
    input_bf16: &[u8],
    weight_bf16: &[u8],
    bias_bf16: Option<&[u8]>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> Result<()> {
    if rows == 0 || input_dim == 0 || output_dim == 0 {
        anyhow::bail!(
            "real full BF16 linear requires non-zero shape, got rows={rows} input_dim={input_dim} output_dim={output_dim}"
        );
    }
    let input_bytes = rows
        .checked_mul(input_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("real full BF16 linear input shape overflows usize while validating input")?;
    if input_bf16.len() != input_bytes {
        anyhow::bail!(
            "real full BF16 linear input byte length mismatch: expected {} got {}",
            input_bytes,
            input_bf16.len()
        );
    }
    let weight_bytes = output_dim
        .checked_mul(input_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("real full BF16 linear weight shape overflows usize while validating input")?;
    if weight_bf16.len() != weight_bytes {
        anyhow::bail!(
            "real full BF16 linear weight byte length mismatch: expected {} got {}",
            weight_bytes,
            weight_bf16.len()
        );
    }
    if let Some(bias_bf16) = bias_bf16 {
        let bias_bytes = output_dim
            .checked_mul(std::mem::size_of::<u16>())
            .context("real full BF16 linear bias shape overflows usize while validating input")?;
        if bias_bf16.len() != bias_bytes {
            anyhow::bail!(
                "real full BF16 linear bias byte length mismatch: expected {} got {}",
                bias_bytes,
                bias_bf16.len()
            );
        }
    }
    Ok(())
}

pub(in crate::commands::real_full) fn validate_linear_bf16_preloaded_resident_inputs(
    input_bf16: &[u8],
    bias_bf16: Option<&[u8]>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    full_output_dim: usize,
) -> Result<LinearResidentView> {
    if rows == 0 || input_dim == 0 || output_dim == 0 || full_output_dim == 0 {
        anyhow::bail!(
            "real full preloaded BF16 linear requires non-zero shape, got rows={rows} input_dim={input_dim} output_dim={output_dim} full_output_dim={full_output_dim}"
        );
    }
    if output_dim > full_output_dim {
        anyhow::bail!(
            "real full preloaded BF16 linear row prefix output_dim={output_dim} exceeds full_output_dim={full_output_dim}"
        );
    }
    let input_bytes = rows
        .checked_mul(input_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full preloaded BF16 linear input shape overflows usize while validating input",
        )?;
    if input_bf16.len() != input_bytes {
        anyhow::bail!(
            "real full preloaded BF16 linear input byte length mismatch: expected {} got {}",
            input_bytes,
            input_bf16.len()
        );
    }
    if let Some(bias_bf16) = bias_bf16 {
        let bias_bytes = output_dim.checked_mul(std::mem::size_of::<u16>()).context(
            "real full preloaded BF16 linear bias shape overflows usize while validating input",
        )?;
        if bias_bf16.len() != bias_bytes {
            anyhow::bail!(
                "real full preloaded BF16 linear bias byte length mismatch: expected {} got {}",
                bias_bytes,
                bias_bf16.len()
            );
        }
    }
    let row_bytes = input_dim
        .checked_mul(std::mem::size_of::<u16>())
        .context("real full preloaded BF16 linear row byte width overflows usize")?;
    let full_bytes = full_output_dim
        .checked_mul(row_bytes)
        .context("real full preloaded BF16 linear full tensor bytes overflow usize")?;
    let view_bytes = output_dim
        .checked_mul(row_bytes)
        .context("real full preloaded BF16 linear row-prefix byte count overflows usize")?;
    Ok(LinearResidentView {
        full_bytes,
        offset_bytes: 0,
        view_bytes,
    })
}

pub(in crate::commands::real_full) fn validate_linear_bf16_preloaded_resident_device_input(
    input_buffer: GlmrtDeviceBuffer,
    bias_bf16: Option<&[u8]>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    full_output_dim: usize,
) -> Result<LinearResidentView> {
    if rows == 0 || input_dim == 0 || output_dim == 0 || full_output_dim == 0 {
        anyhow::bail!(
            "real full preloaded BF16 device-input linear requires non-zero shape, got rows={rows} input_dim={input_dim} output_dim={output_dim} full_output_dim={full_output_dim}"
        );
    }
    if output_dim > full_output_dim {
        anyhow::bail!(
            "real full preloaded BF16 device-input linear row prefix output_dim={output_dim} exceeds full_output_dim={full_output_dim}"
        );
    }
    let input_bytes = rows
        .checked_mul(input_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full preloaded BF16 device-input linear input shape overflows usize while validating input",
        )?;
    if input_buffer.ptr.is_null() {
        anyhow::bail!("real full preloaded BF16 device-input linear input buffer is null");
    }
    if input_buffer.bytes < input_bytes {
        anyhow::bail!(
            "real full preloaded BF16 device-input linear input buffer byte length mismatch: expected at least {} got {}",
            input_bytes,
            input_buffer.bytes
        );
    }
    if let Some(bias_bf16) = bias_bf16 {
        let bias_bytes = output_dim.checked_mul(std::mem::size_of::<u16>()).context(
            "real full preloaded BF16 device-input linear bias shape overflows usize while validating input",
        )?;
        if bias_bf16.len() != bias_bytes {
            anyhow::bail!(
                "real full preloaded BF16 device-input linear bias byte length mismatch: expected {} got {}",
                bias_bytes,
                bias_bf16.len()
            );
        }
    }
    let row_bytes = input_dim
        .checked_mul(std::mem::size_of::<u16>())
        .context("real full preloaded BF16 device-input linear row byte width overflows usize")?;
    let full_bytes = full_output_dim
        .checked_mul(row_bytes)
        .context("real full preloaded BF16 device-input linear full tensor bytes overflow usize")?;
    let view_bytes = output_dim.checked_mul(row_bytes).context(
        "real full preloaded BF16 device-input linear row-prefix byte count overflows usize",
    )?;
    Ok(LinearResidentView {
        full_bytes,
        offset_bytes: 0,
        view_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn validate_linear_bf16_preloaded_resident_padded_device_input(
    input_buffer: GlmrtDeviceBuffer,
    bias_bf16: Option<&[u8]>,
    rows: usize,
    active_input_dim: usize,
    full_input_dim: usize,
    output_dim: usize,
    full_output_dim: usize,
) -> Result<LinearPaddedDeviceInputView> {
    if rows == 0
        || active_input_dim == 0
        || full_input_dim == 0
        || output_dim == 0
        || full_output_dim == 0
    {
        anyhow::bail!(
            "real full preloaded BF16 padded device-input linear requires non-zero shape, got rows={rows} active_input_dim={active_input_dim} full_input_dim={full_input_dim} output_dim={output_dim} full_output_dim={full_output_dim}"
        );
    }
    if active_input_dim > full_input_dim {
        anyhow::bail!(
            "real full preloaded BF16 padded device-input linear active_input_dim={active_input_dim} exceeds full_input_dim={full_input_dim}"
        );
    }
    if output_dim > full_output_dim {
        anyhow::bail!(
            "real full preloaded BF16 padded device-input linear row prefix output_dim={output_dim} exceeds full_output_dim={full_output_dim}"
        );
    }
    let active_row_bytes = active_input_dim
        .checked_mul(std::mem::size_of::<u16>())
        .context("real full preloaded BF16 padded device-input active row bytes overflow usize")?;
    let padded_row_bytes = full_input_dim
        .checked_mul(std::mem::size_of::<u16>())
        .context("real full preloaded BF16 padded device-input full row bytes overflow usize")?;
    let active_input_bytes = rows
        .checked_mul(active_row_bytes)
        .context("real full preloaded BF16 padded device-input active bytes overflow usize")?;
    let padded_input_bytes = rows
        .checked_mul(padded_row_bytes)
        .context("real full preloaded BF16 padded device-input padded bytes overflow usize")?;
    if input_buffer.ptr.is_null() {
        anyhow::bail!("real full preloaded BF16 padded device-input linear input buffer is null");
    }
    if input_buffer.bytes < active_input_bytes {
        anyhow::bail!(
            "real full preloaded BF16 padded device-input linear input buffer byte length mismatch: expected at least {} got {}",
            active_input_bytes,
            input_buffer.bytes
        );
    }
    if let Some(bias_bf16) = bias_bf16 {
        let bias_bytes = output_dim.checked_mul(std::mem::size_of::<u16>()).context(
            "real full preloaded BF16 padded device-input linear bias shape overflows usize while validating input",
        )?;
        if bias_bf16.len() != bias_bytes {
            anyhow::bail!(
                "real full preloaded BF16 padded device-input linear bias byte length mismatch: expected {} got {}",
                bias_bytes,
                bias_bf16.len()
            );
        }
    }
    let full_bytes = full_output_dim.checked_mul(padded_row_bytes).context(
        "real full preloaded BF16 padded device-input linear full tensor bytes overflow usize",
    )?;
    let view_bytes = output_dim.checked_mul(padded_row_bytes).context(
        "real full preloaded BF16 padded device-input linear row-prefix byte count overflows usize",
    )?;
    Ok(LinearPaddedDeviceInputView {
        weight: LinearResidentView {
            full_bytes,
            offset_bytes: 0,
            view_bytes,
        },
        padded_input_bytes,
        active_row_bytes,
        padded_row_bytes,
    })
}

pub(in crate::commands::real_full) fn validate_linear_residual_add_bf16_residual(
    residual_bf16: &[u8],
    rows: usize,
    output_dim: usize,
    context: &str,
) -> Result<usize> {
    let output_bytes = rows
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .with_context(|| format!("{context} output byte shape overflows usize"))?;
    if residual_bf16.len() != output_bytes {
        anyhow::bail!(
            "real full {context} residual byte length mismatch: expected {} got {}",
            output_bytes,
            residual_bf16.len()
        );
    }
    if residual_bf16.is_empty() || residual_bf16.len() % std::mem::size_of::<u16>() != 0 {
        anyhow::bail!(
            "real full {context} residual BF16 byte length must be nonzero and even, got {}",
            residual_bf16.len()
        );
    }
    Ok(output_bytes)
}

#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn host_linear_residual_add_bf16_output(
    linear: LinearOutput,
    residual_bf16: &[u8],
) -> Result<LinearResidualAddOutput> {
    let linear_bf16 = f32_values_to_bf16_bytes(&linear.values);
    let mut residual_out_bf16 = vec![0_u8; residual_bf16.len()];
    let residual_add_backend =
        residual_add_prefix_bf16_bytes_into(residual_bf16, &linear_bf16, &mut residual_out_bf16)?;
    Ok(LinearResidualAddOutput {
        linear_values: linear.values,
        residual_values: bf16_values_to_f32(&residual_out_bf16),
        linear_backend: linear.backend,
        residual_add_backend,
    })
}

pub(in crate::commands::real_full) fn cpu_linear_rows(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> LinearOutput {
    let mut values = vec![0.0_f32; rows * output_dim];
    for row in 0..rows {
        let input_start = row * input_dim;
        let output_start = row * output_dim;
        for output_index in 0..output_dim {
            let weight_start = output_index * input_dim;
            let mut value = bias.map(|bias| bias[output_index]).unwrap_or(0.0);
            for input_index in 0..input_dim {
                value += input[input_start + input_index] * weight[weight_start + input_index];
            }
            values[output_start + output_index] = value;
        }
    }
    LinearOutput {
        values,
        backend: CPU_REFERENCE_LINEAR_BACKEND,
    }
}

pub(in crate::commands::real_full) fn cpu_linear_rows_bf16(
    input_bf16: &[u8],
    weight_bf16: &[u8],
    bias_bf16: Option<&[u8]>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> LinearOutput {
    let input = bf16_values_to_f32(input_bf16);
    let weight = bf16_values_to_f32(weight_bf16);
    let bias = bias_bf16.map(bf16_values_to_f32);
    let mut output = cpu_linear_rows(
        &input,
        &weight,
        bias.as_deref(),
        rows,
        input_dim,
        output_dim,
    );
    output.backend = CPU_REFERENCE_LINEAR_BF16_BACKEND;
    output
}

pub(in crate::commands::real_full) fn cuda_linear_rows(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> Result<LinearOutput> {
    let library = cuda_native_library()?;
    let input_bytes = std::mem::size_of_val(input);
    let weight_bytes = std::mem::size_of_val(weight);
    let output_bytes = rows
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context("CUDA linear output shape overflows usize")?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let input_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        input_bytes,
        "linear input",
    )?;
    let weight_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::B,
        weight_bytes,
        "linear weight",
    )?;
    let output_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        output_bytes,
        "linear output",
    )?;
    let bias_buffer = if let Some(bias) = bias {
        let buffer = workspace.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            std::mem::size_of_val(bias),
            "linear bias",
        )?;
        workspace
            .copy_h2d_to_slot(
                library,
                CoordinatorCudaScratchSlot::D,
                f32_bytes(bias),
                "linear bias",
            )
            .context("copying linear bias to device")?;
        Some(buffer)
    } else {
        None
    };

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            f32_bytes(input),
            "linear input",
        )
        .context("copying linear input to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::B,
            f32_bytes(weight),
            "linear weight",
        )
        .context("copying linear weight to device")?;
    library
        .cuda_linear_f32(
            input_buffer,
            weight_buffer,
            bias_buffer,
            output_buffer,
            rows,
            input_dim,
            output_dim,
        )
        .context("executing CUDA linear")?;
    let mut out_bytes = vec![0_u8; output_bytes];
    library
        .copy_d2h(&mut out_bytes, output_buffer)
        .context("copying linear output to host")?;

    Ok(LinearOutput {
        values: f32_vec_from_bytes(&out_bytes)?,
        backend: CUDA_REFERENCE_LINEAR_BACKEND,
    })
}

pub(in crate::commands::real_full) fn cuda_linear_rows_bf16(
    input_bf16: &[u8],
    weight_bf16: &[u8],
    bias_bf16: Option<&[u8]>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> Result<LinearOutput> {
    let library = cuda_native_library()?;
    let output_bytes = rows
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 linear output shape overflows usize")?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let input_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        input_bf16.len(),
        "BF16 linear input",
    )?;
    let weight_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::B,
        weight_bf16.len(),
        "BF16 linear weight",
    )?;
    let output_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        output_bytes,
        "BF16 linear output",
    )?;
    let bias_buffer = if let Some(bias_bf16) = bias_bf16 {
        let buffer = workspace.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            bias_bf16.len(),
            "BF16 linear bias",
        )?;
        workspace
            .copy_h2d_to_slot(
                library,
                CoordinatorCudaScratchSlot::D,
                bias_bf16,
                "BF16 linear bias",
            )
            .context("copying BF16 linear bias to device")?;
        Some(buffer)
    } else {
        None
    };

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            input_bf16,
            "BF16 linear input",
        )
        .context("copying BF16 linear input to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::B,
            weight_bf16,
            "BF16 linear weight",
        )
        .context("copying BF16 linear weight to device")?;
    library
        .cuda_linear_bf16_cublas(
            input_buffer,
            weight_buffer,
            bias_buffer,
            output_buffer,
            rows,
            input_dim,
            output_dim,
        )
        .context("executing CUDA BF16 linear")?;
    let mut out_bytes = vec![0_u8; output_bytes];
    library
        .copy_d2h(&mut out_bytes, output_buffer)
        .context("copying BF16 linear output to host")?;

    Ok(LinearOutput {
        values: bf16_values_to_f32(&out_bytes),
        backend: CUDA_REFERENCE_LINEAR_BF16_BACKEND,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn linear_rows_bf16_device_buffers_for_layer(
    layer_id: usize,
    input_buffer: GlmrtDeviceBuffer,
    weight_buffer: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> Result<&'static str> {
    validate_linear_bf16_device_buffers(
        input_buffer,
        weight_buffer,
        output_buffer,
        rows,
        input_dim,
        output_dim,
    )?;
    let graph_key = coord_attention_graph_key_for_layer_rows(layer_id, rows)?;
    let signature = linear_graph_signature(&graph_key, input_dim, output_dim, false);
    with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        capture_or_update_layer_linear_bf16_graph(
            library,
            slot,
            signature,
            input_buffer,
            weight_buffer,
            None,
            output_buffer,
            rows,
            input_dim,
            output_dim,
            "BF16 layer linear device-buffer",
        )?;
        unsafe {
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing BF16 layer linear device-buffer graph slot stream")?;
        }
        Ok(CUDA_REFERENCE_LINEAR_BF16_BACKEND)
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn capture_or_update_layer_linear_bf16_graph(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    signature: CoordinatorCudaGraphSignature,
    input_buffer: GlmrtDeviceBuffer,
    weight_buffer: GlmrtDeviceBuffer,
    bias_buffer: Option<GlmrtDeviceBuffer>,
    output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    label: &'static str,
) -> Result<()> {
    let capture_rows = linear_bf16_cublas_graph_rows(
        signature,
        input_buffer,
        output_buffer,
        rows,
        input_dim,
        output_dim,
    )
    .with_context(|| format!("choosing CUDA cuBLAS {label} graph row count"))?;
    let capture_identity = linear_bf16_cublas_graph_capture_identity(
        input_buffer,
        weight_buffer,
        bias_buffer,
        output_buffer,
        capture_rows,
        input_dim,
        output_dim,
    );
    if !slot.has_captured_graph_identity(
        CoordinatorCudaGraphProgram::LayerLinearBf16,
        signature,
        capture_identity,
    ) {
        unsafe {
            library
                .cuda_linear_bf16_cublas_async(
                    input_buffer,
                    weight_buffer,
                    bias_buffer,
                    output_buffer,
                    capture_rows,
                    input_dim,
                    output_dim,
                    slot.stream_ptr(),
                )
                .with_context(|| format!("warming CUDA cuBLAS {label} before graph capture"))?;
        }
    }
    slot.capture_or_update_graph_exec(
        library,
        CoordinatorCudaGraphProgram::LayerLinearBf16,
        signature,
        capture_identity,
        |library, cuda_stream, _workspace| unsafe {
            library
                .cuda_linear_bf16_cublas_async(
                    input_buffer,
                    weight_buffer,
                    bias_buffer,
                    output_buffer,
                    capture_rows,
                    input_dim,
                    output_dim,
                    cuda_stream,
                )
                .with_context(|| format!("capturing async CUDA cuBLAS {label}"))?;
            Ok(())
        },
    )?;
    slot.launch_captured_graph_identity(
        library,
        CoordinatorCudaGraphProgram::LayerLinearBf16,
        signature,
        capture_identity,
    )
}

fn linear_bf16_cublas_graph_rows(
    signature: CoordinatorCudaGraphSignature,
    input_buffer: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> Result<usize> {
    let graph_rows = signature.rows;
    let graph_input_bytes = graph_rows
        .checked_mul(input_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 cuBLAS linear graph input bytes overflow usize")?;
    let graph_output_bytes = graph_rows
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 cuBLAS linear graph output bytes overflow usize")?;
    if graph_rows >= rows
        && input_buffer.bytes >= graph_input_bytes
        && output_buffer.bytes >= graph_output_bytes
    {
        Ok(graph_rows)
    } else {
        Ok(rows)
    }
}

fn linear_bf16_cublas_graph_capture_identity(
    input_buffer: GlmrtDeviceBuffer,
    weight_buffer: GlmrtDeviceBuffer,
    bias_buffer: Option<GlmrtDeviceBuffer>,
    output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> usize {
    let bias_ptr = bias_buffer.map(|buffer| buffer.ptr as usize).unwrap_or(0);
    graph_capture_identity(&[
        input_buffer.ptr as usize,
        weight_buffer.ptr as usize,
        bias_ptr,
        output_buffer.ptr as usize,
        rows,
        input_dim,
        output_dim,
    ])
}

fn graph_capture_identity(parts: &[usize]) -> usize {
    parts.iter().fold(0xcbf29ce484222325_usize, |hash, part| {
        hash.wrapping_mul(0x100000001b3).wrapping_add(*part)
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn validate_linear_bf16_device_buffers(
    input_buffer: GlmrtDeviceBuffer,
    weight_buffer: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> Result<()> {
    if rows == 0 || input_dim == 0 || output_dim == 0 {
        anyhow::bail!(
            "CUDA BF16 layer linear device-buffer requires nonzero shape, got rows={rows} input_dim={input_dim} output_dim={output_dim}"
        );
    }
    let input_bytes = rows
        .checked_mul(input_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 layer linear device-buffer input bytes overflow usize")?;
    let weight_bytes = output_dim
        .checked_mul(input_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 layer linear device-buffer weight bytes overflow usize")?;
    let output_bytes = rows
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 layer linear device-buffer output bytes overflow usize")?;
    let buffers = [
        ("input", input_buffer, input_bytes),
        ("weight", weight_buffer, weight_bytes),
        ("output", output_buffer, output_bytes),
    ];
    for (label, buffer, required_bytes) in buffers {
        if buffer.ptr.is_null() {
            anyhow::bail!("CUDA BF16 layer linear device-buffer {label} is null");
        }
        if buffer.bytes < required_bytes {
            anyhow::bail!(
                "CUDA BF16 layer linear device-buffer {label} has {} bytes, expected at least {required_bytes}",
                buffer.bytes
            );
        }
        if buffer.device_id != input_buffer.device_id {
            anyhow::bail!(
                "CUDA BF16 layer linear device-buffer {label} is on CUDA device {}, expected {}",
                buffer.device_id,
                input_buffer.device_id
            );
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn capture_or_update_layer_linear_residual_add_bf16_graph(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    signature: CoordinatorCudaGraphSignature,
    input_buffer: GlmrtDeviceBuffer,
    weight_buffer: GlmrtDeviceBuffer,
    bias_buffer: Option<GlmrtDeviceBuffer>,
    linear_output_buffer: GlmrtDeviceBuffer,
    residual_buffer: GlmrtDeviceBuffer,
    residual_output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    label: &'static str,
) -> Result<()> {
    let capture_rows = linear_bf16_cublas_residual_graph_rows(
        signature,
        input_buffer,
        linear_output_buffer,
        residual_buffer,
        residual_output_buffer,
        rows,
        input_dim,
        output_dim,
    )
    .with_context(|| format!("choosing CUDA cuBLAS {label} graph row count"))?;
    let count = capture_rows
        .checked_mul(output_dim)
        .with_context(|| format!("{label} residual value count overflows usize"))?;
    let capture_identity = linear_bf16_cublas_residual_graph_capture_identity(
        input_buffer,
        weight_buffer,
        bias_buffer,
        linear_output_buffer,
        residual_buffer,
        residual_output_buffer,
        capture_rows,
        input_dim,
        output_dim,
    );
    if !slot.has_captured_graph_identity(
        CoordinatorCudaGraphProgram::LayerLinearResidualAddBf16,
        signature,
        capture_identity,
    ) {
        unsafe {
            library
                .cuda_linear_bf16_cublas_async(
                    input_buffer,
                    weight_buffer,
                    bias_buffer,
                    linear_output_buffer,
                    capture_rows,
                    input_dim,
                    output_dim,
                    slot.stream_ptr(),
                )
                .with_context(|| format!("warming CUDA cuBLAS {label} before graph capture"))?;
        }
    }
    slot.capture_or_update_graph_exec(
        library,
        CoordinatorCudaGraphProgram::LayerLinearResidualAddBf16,
        signature,
        capture_identity,
        |library, cuda_stream, _workspace| unsafe {
            library
                .cuda_linear_bf16_cublas_async(
                    input_buffer,
                    weight_buffer,
                    bias_buffer,
                    linear_output_buffer,
                    capture_rows,
                    input_dim,
                    output_dim,
                    cuda_stream,
                )
                .with_context(|| format!("capturing async CUDA cuBLAS {label} linear"))?;
            library
                .cuda_residual_add_bf16_async(
                    residual_buffer,
                    linear_output_buffer,
                    residual_output_buffer,
                    count,
                    cuda_stream,
                )
                .with_context(|| format!("capturing async CUDA {label} residual add"))?;
            Ok(())
        },
    )?;
    slot.launch_captured_graph_identity(
        library,
        CoordinatorCudaGraphProgram::LayerLinearResidualAddBf16,
        signature,
        capture_identity,
    )
}

#[allow(clippy::too_many_arguments)]
fn linear_bf16_cublas_residual_graph_rows(
    signature: CoordinatorCudaGraphSignature,
    input_buffer: GlmrtDeviceBuffer,
    linear_output_buffer: GlmrtDeviceBuffer,
    residual_buffer: GlmrtDeviceBuffer,
    residual_output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> Result<usize> {
    let graph_rows = signature.rows;
    let graph_input_bytes = linear_bf16_graph_bytes(graph_rows, input_dim)
        .context("CUDA BF16 cuBLAS residual linear graph input bytes overflow usize")?;
    let graph_output_bytes = linear_bf16_graph_bytes(graph_rows, output_dim)
        .context("CUDA BF16 cuBLAS residual linear graph output bytes overflow usize")?;
    if graph_rows >= rows
        && input_buffer.bytes >= graph_input_bytes
        && linear_output_buffer.bytes >= graph_output_bytes
        && residual_buffer.bytes >= graph_output_bytes
        && residual_output_buffer.bytes >= graph_output_bytes
    {
        Ok(graph_rows)
    } else {
        Ok(rows)
    }
}

#[allow(clippy::too_many_arguments)]
fn linear_bf16_cublas_residual_graph_capture_identity(
    input_buffer: GlmrtDeviceBuffer,
    weight_buffer: GlmrtDeviceBuffer,
    bias_buffer: Option<GlmrtDeviceBuffer>,
    linear_output_buffer: GlmrtDeviceBuffer,
    residual_buffer: GlmrtDeviceBuffer,
    residual_output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> usize {
    let bias_ptr = bias_buffer.map(|buffer| buffer.ptr as usize).unwrap_or(0);
    graph_capture_identity(&[
        input_buffer.ptr as usize,
        weight_buffer.ptr as usize,
        bias_ptr,
        linear_output_buffer.ptr as usize,
        residual_buffer.ptr as usize,
        residual_output_buffer.ptr as usize,
        rows,
        input_dim,
        output_dim,
    ])
}

fn linear_bf16_graph_bytes(rows: usize, dim: usize) -> Result<usize> {
    rows.checked_mul(dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 graph bytes overflow usize")
}

#[allow(clippy::too_many_arguments)]
fn padded_linear_bf16_cublas_graph_rows(
    graph_rows: usize,
    padded_input_buffer: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    full_input_dim: usize,
    output_dim: usize,
) -> Result<usize> {
    let graph_input_bytes = linear_bf16_graph_bytes(graph_rows, full_input_dim)
        .context("CUDA BF16 cuBLAS padded linear graph input bytes overflow usize")?;
    let graph_output_bytes = linear_bf16_graph_bytes(graph_rows, output_dim)
        .context("CUDA BF16 cuBLAS padded linear graph output bytes overflow usize")?;
    if graph_rows >= rows
        && padded_input_buffer.bytes >= graph_input_bytes
        && output_buffer.bytes >= graph_output_bytes
    {
        Ok(graph_rows)
    } else {
        Ok(rows)
    }
}

#[allow(clippy::too_many_arguments)]
fn padded_linear_bf16_cublas_residual_graph_rows(
    graph_rows: usize,
    padded_input_buffer: GlmrtDeviceBuffer,
    linear_output_buffer: GlmrtDeviceBuffer,
    residual_buffer: GlmrtDeviceBuffer,
    residual_output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    full_input_dim: usize,
    output_dim: usize,
) -> Result<usize> {
    let graph_input_bytes = linear_bf16_graph_bytes(graph_rows, full_input_dim)
        .context("CUDA BF16 cuBLAS padded residual linear graph input bytes overflow usize")?;
    let graph_output_bytes = linear_bf16_graph_bytes(graph_rows, output_dim)
        .context("CUDA BF16 cuBLAS padded residual linear graph output bytes overflow usize")?;
    if graph_rows >= rows
        && padded_input_buffer.bytes >= graph_input_bytes
        && linear_output_buffer.bytes >= graph_output_bytes
        && residual_buffer.bytes >= graph_output_bytes
        && residual_output_buffer.bytes >= graph_output_bytes
    {
        Ok(graph_rows)
    } else {
        Ok(rows)
    }
}

#[allow(clippy::too_many_arguments)]
fn padded_linear_bf16_cublas_graph_capture_identity(
    padded_input_buffer: GlmrtDeviceBuffer,
    weight_buffer: GlmrtDeviceBuffer,
    bias_buffer: Option<GlmrtDeviceBuffer>,
    output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    full_input_dim: usize,
    output_dim: usize,
    view: LinearPaddedDeviceInputView,
) -> usize {
    let bias_ptr = bias_buffer.map(|buffer| buffer.ptr as usize).unwrap_or(0);
    graph_capture_identity(&[
        padded_input_buffer.ptr as usize,
        weight_buffer.ptr as usize,
        bias_ptr,
        output_buffer.ptr as usize,
        rows,
        full_input_dim,
        output_dim,
        view.active_row_bytes,
        view.padded_row_bytes,
    ])
}

#[allow(clippy::too_many_arguments)]
fn padded_linear_bf16_cublas_residual_graph_capture_identity(
    padded_input_buffer: GlmrtDeviceBuffer,
    weight_buffer: GlmrtDeviceBuffer,
    bias_buffer: Option<GlmrtDeviceBuffer>,
    linear_output_buffer: GlmrtDeviceBuffer,
    residual_buffer: GlmrtDeviceBuffer,
    residual_output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    full_input_dim: usize,
    output_dim: usize,
    view: LinearPaddedDeviceInputView,
) -> usize {
    let bias_ptr = bias_buffer.map(|buffer| buffer.ptr as usize).unwrap_or(0);
    graph_capture_identity(&[
        padded_input_buffer.ptr as usize,
        weight_buffer.ptr as usize,
        bias_ptr,
        linear_output_buffer.ptr as usize,
        residual_buffer.ptr as usize,
        residual_output_buffer.ptr as usize,
        rows,
        full_input_dim,
        output_dim,
        view.active_row_bytes,
        view.padded_row_bytes,
    ])
}

unsafe fn update_padded_linear_copy_nodes_if_captured(
    library: &'static NativeLibrary,
    slot: &CoordinatorCudaGraphWorkspaceSlot,
    program: CoordinatorCudaGraphProgram,
    signature: CoordinatorCudaGraphSignature,
    input_buffer: GlmrtDeviceBuffer,
    padded_input_buffer: GlmrtDeviceBuffer,
    rows: usize,
    graph_rows: usize,
    view: LinearPaddedDeviceInputView,
    label: &'static str,
) -> Result<()> {
    if let Some((graph_raw, exec_raw)) = slot.captured_graph_raw_handles(program, signature) {
        unsafe {
            update_padded_device_input_copy_graph_nodes(
                library,
                graph_raw,
                exec_raw,
                0,
                input_buffer,
                padded_input_buffer,
                rows,
                graph_rows,
                view,
                label,
            )?;
        }
    }
    Ok(())
}

pub(in crate::commands::real_full) fn linear_graph_input_bytes(
    graph_key: &CoordinatorGraphKey,
    input_dim: usize,
    context: &str,
) -> Result<usize> {
    graph_key
        .row_bucket
        .row_capacity
        .checked_mul(input_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .with_context(|| format!("{context} input graph buffer bytes overflow usize"))
}

pub(in crate::commands::real_full) fn linear_graph_output_bytes(
    graph_key: &CoordinatorGraphKey,
    output_dim: usize,
    context: &str,
) -> Result<usize> {
    graph_key
        .row_bucket
        .row_capacity
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .with_context(|| format!("{context} output graph buffer bytes overflow usize"))
}

pub(in crate::commands::real_full) fn linear_graph_signature(
    graph_key: &CoordinatorGraphKey,
    input_dim: usize,
    output_dim: usize,
    has_bias: bool,
) -> CoordinatorCudaGraphSignature {
    CoordinatorCudaGraphSignature::linear_bf16(
        graph_key.row_bucket.row_capacity,
        input_dim,
        output_dim,
        has_bias,
    )
}

pub(in crate::commands::real_full) fn padded_linear_graph_padded_input_bytes(
    graph_key: &CoordinatorGraphKey,
    view: LinearPaddedDeviceInputView,
    context: &str,
) -> Result<usize> {
    graph_key
        .row_bucket
        .row_capacity
        .checked_mul(view.padded_row_bytes)
        .with_context(|| format!("{context} padded input graph buffer bytes overflow usize"))
}

pub(in crate::commands::real_full) fn padded_linear_graph_output_bytes(
    graph_key: &CoordinatorGraphKey,
    output_dim: usize,
    context: &str,
) -> Result<usize> {
    graph_key
        .row_bucket
        .row_capacity
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .with_context(|| format!("{context} output graph buffer bytes overflow usize"))
}

pub(in crate::commands::real_full) fn padded_linear_graph_signature(
    graph_key: &CoordinatorGraphKey,
    active_input_dim: usize,
    full_input_dim: usize,
    output_dim: usize,
    has_bias: bool,
) -> CoordinatorCudaGraphSignature {
    CoordinatorCudaGraphSignature::padded_linear_bf16(
        graph_key.row_bucket.row_capacity,
        active_input_dim,
        full_input_dim,
        output_dim,
        has_bias,
    )
}

pub(in crate::commands::real_full) fn padded_linear_residual_graph_signature(
    graph_key: &CoordinatorGraphKey,
    active_input_dim: usize,
    full_input_dim: usize,
    output_dim: usize,
    has_bias: bool,
) -> CoordinatorCudaGraphSignature {
    CoordinatorCudaGraphSignature::padded_linear_residual_add_bf16(
        graph_key.row_bucket.row_capacity,
        active_input_dim,
        full_input_dim,
        output_dim,
        has_bias,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn capture_or_update_layer_padded_linear_bf16_graph(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    signature: CoordinatorCudaGraphSignature,
    input_buffer: GlmrtDeviceBuffer,
    padded_input_buffer: GlmrtDeviceBuffer,
    weight_buffer: GlmrtDeviceBuffer,
    bias_buffer: Option<GlmrtDeviceBuffer>,
    output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    graph_rows: usize,
    full_input_dim: usize,
    output_dim: usize,
    view: LinearPaddedDeviceInputView,
    label: &'static str,
) -> Result<()> {
    let capture_rows = padded_linear_bf16_cublas_graph_rows(
        graph_rows,
        padded_input_buffer,
        output_buffer,
        rows,
        full_input_dim,
        output_dim,
    )
    .with_context(|| format!("choosing CUDA cuBLAS {label} graph row count"))?;
    let capture_identity = padded_linear_bf16_cublas_graph_capture_identity(
        padded_input_buffer,
        weight_buffer,
        bias_buffer,
        output_buffer,
        capture_rows,
        full_input_dim,
        output_dim,
        view,
    );
    if !slot.has_captured_graph_identity(
        CoordinatorCudaGraphProgram::LayerPaddedLinearBf16,
        signature,
        capture_identity,
    ) {
        unsafe {
            library
                .cuda_linear_bf16_cublas_async(
                    padded_input_buffer,
                    weight_buffer,
                    bias_buffer,
                    output_buffer,
                    capture_rows,
                    full_input_dim,
                    output_dim,
                    slot.stream_ptr(),
                )
                .with_context(|| format!("warming CUDA cuBLAS {label} before graph capture"))?;
        }
    }
    slot.capture_or_update_graph_exec(
        library,
        CoordinatorCudaGraphProgram::LayerPaddedLinearBf16,
        signature,
        capture_identity,
        |library, cuda_stream, _workspace| unsafe {
            library
                .cuda_zero_bytes_async(padded_input_buffer, padded_input_buffer.bytes, cuda_stream)
                .with_context(|| format!("capturing async CUDA {label} padded input zero"))?;
            copy_rows_to_padded_device_input_async(
                library,
                input_buffer,
                padded_input_buffer,
                rows,
                graph_rows,
                view,
                cuda_stream,
            )
            .with_context(|| format!("capturing async CUDA {label} active row copies"))?;
            library
                .cuda_linear_bf16_cublas_async(
                    padded_input_buffer,
                    weight_buffer,
                    bias_buffer,
                    output_buffer,
                    capture_rows,
                    full_input_dim,
                    output_dim,
                    cuda_stream,
                )
                .with_context(|| format!("capturing async CUDA cuBLAS {label} linear"))?;
            Ok(())
        },
    )?;
    unsafe {
        update_padded_linear_copy_nodes_if_captured(
            library,
            slot,
            CoordinatorCudaGraphProgram::LayerPaddedLinearBf16,
            signature,
            input_buffer,
            padded_input_buffer,
            rows,
            graph_rows,
            view,
            label,
        )?;
    }
    slot.launch_captured_graph_identity(
        library,
        CoordinatorCudaGraphProgram::LayerPaddedLinearBf16,
        signature,
        capture_identity,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn capture_or_update_layer_padded_linear_residual_add_bf16_graph(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    signature: CoordinatorCudaGraphSignature,
    input_buffer: GlmrtDeviceBuffer,
    padded_input_buffer: GlmrtDeviceBuffer,
    weight_buffer: GlmrtDeviceBuffer,
    bias_buffer: Option<GlmrtDeviceBuffer>,
    linear_output_buffer: GlmrtDeviceBuffer,
    residual_buffer: GlmrtDeviceBuffer,
    residual_output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    graph_rows: usize,
    full_input_dim: usize,
    output_dim: usize,
    view: LinearPaddedDeviceInputView,
    label: &'static str,
) -> Result<()> {
    let capture_rows = padded_linear_bf16_cublas_residual_graph_rows(
        graph_rows,
        padded_input_buffer,
        linear_output_buffer,
        residual_buffer,
        residual_output_buffer,
        rows,
        full_input_dim,
        output_dim,
    )
    .with_context(|| format!("choosing CUDA cuBLAS {label} graph row count"))?;
    let count = capture_rows
        .checked_mul(output_dim)
        .with_context(|| format!("{label} residual value count overflows usize"))?;
    let capture_identity = padded_linear_bf16_cublas_residual_graph_capture_identity(
        padded_input_buffer,
        weight_buffer,
        bias_buffer,
        linear_output_buffer,
        residual_buffer,
        residual_output_buffer,
        capture_rows,
        full_input_dim,
        output_dim,
        view,
    );
    if !slot.has_captured_graph_identity(
        CoordinatorCudaGraphProgram::LayerPaddedLinearResidualAddBf16,
        signature,
        capture_identity,
    ) {
        unsafe {
            library
                .cuda_linear_bf16_cublas_async(
                    padded_input_buffer,
                    weight_buffer,
                    bias_buffer,
                    linear_output_buffer,
                    capture_rows,
                    full_input_dim,
                    output_dim,
                    slot.stream_ptr(),
                )
                .with_context(|| format!("warming CUDA cuBLAS {label} before graph capture"))?;
        }
    }
    slot.capture_or_update_graph_exec(
        library,
        CoordinatorCudaGraphProgram::LayerPaddedLinearResidualAddBf16,
        signature,
        capture_identity,
        |library, cuda_stream, _workspace| unsafe {
            library
                .cuda_zero_bytes_async(padded_input_buffer, padded_input_buffer.bytes, cuda_stream)
                .with_context(|| format!("capturing async CUDA {label} padded input zero"))?;
            copy_rows_to_padded_device_input_async(
                library,
                input_buffer,
                padded_input_buffer,
                rows,
                graph_rows,
                view,
                cuda_stream,
            )
            .with_context(|| format!("capturing async CUDA {label} active row copies"))?;
            library
                .cuda_linear_bf16_cublas_async(
                    padded_input_buffer,
                    weight_buffer,
                    bias_buffer,
                    linear_output_buffer,
                    capture_rows,
                    full_input_dim,
                    output_dim,
                    cuda_stream,
                )
                .with_context(|| format!("capturing async CUDA cuBLAS {label} linear"))?;
            library
                .cuda_residual_add_bf16_async(
                    residual_buffer,
                    linear_output_buffer,
                    residual_output_buffer,
                    count,
                    cuda_stream,
                )
                .with_context(|| format!("capturing async CUDA {label} residual add"))?;
            Ok(())
        },
    )?;
    unsafe {
        update_padded_linear_copy_nodes_if_captured(
            library,
            slot,
            CoordinatorCudaGraphProgram::LayerPaddedLinearResidualAddBf16,
            signature,
            input_buffer,
            padded_input_buffer,
            rows,
            graph_rows,
            view,
            label,
        )?;
    }
    slot.launch_captured_graph_identity(
        library,
        CoordinatorCudaGraphProgram::LayerPaddedLinearResidualAddBf16,
        signature,
        capture_identity,
    )
}

#[allow(clippy::too_many_arguments)]
unsafe fn update_padded_device_input_copy_graph_nodes(
    library: &NativeLibrary,
    graph_raw: *mut c_void,
    exec_raw: *mut c_void,
    first_kernel_node_index: usize,
    input_buffer: GlmrtDeviceBuffer,
    padded_input_buffer: GlmrtDeviceBuffer,
    rows: usize,
    graph_rows: usize,
    view: LinearPaddedDeviceInputView,
    label: &'static str,
) -> Result<()> {
    for row in 0..graph_rows {
        let dst_offset = row
            .checked_mul(view.padded_row_bytes)
            .with_context(|| format!("{label} padded destination row offset overflows usize"))?;
        let (src, bytes) = if row < rows {
            let src_offset = row
                .checked_mul(view.active_row_bytes)
                .with_context(|| format!("{label} padded source row offset overflows usize"))?;
            (
                device_buffer_byte_view(
                    input_buffer,
                    src_offset,
                    view.active_row_bytes,
                    "padded graph active row source",
                )?,
                view.active_row_bytes,
            )
        } else {
            (
                device_buffer_byte_view(
                    padded_input_buffer,
                    dst_offset,
                    1,
                    "padded graph inactive row self-copy source",
                )?,
                1,
            )
        };
        unsafe {
            library
                .cuda_graph_update_kv_cache_write_bytes_node(
                    graph_raw,
                    exec_raw,
                    first_kernel_node_index + row,
                    src,
                    padded_input_buffer,
                    dst_offset,
                    bytes,
                )
                .with_context(|| format!("updating captured CUDA {label} row-copy graph node"))?;
        }
    }
    Ok(())
}

pub(in crate::commands::real_full) fn cuda_linear_rows_bf16_resident_weight(
    weight_name: &str,
    input_bf16: &[u8],
    weight_bf16: &[u8],
    bias_bf16: Option<&[u8]>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> Result<LinearOutput> {
    if let Some(graph_key) = coord_linear_graph_key_for_weight_name(weight_name, rows)? {
        return cuda_linear_rows_bf16_resident_weight_graph_slot(
            &graph_key,
            weight_name,
            input_bf16,
            weight_bf16,
            bias_bf16,
            rows,
            input_dim,
            output_dim,
        );
    }
    cuda_linear_rows_bf16_resident_weight_legacy(
        weight_name,
        input_bf16,
        weight_bf16,
        bias_bf16,
        rows,
        input_dim,
        output_dim,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_linear_rows_bf16_resident_weight_graph_slot(
    graph_key: &CoordinatorGraphKey,
    weight_name: &str,
    input_bf16: &[u8],
    weight_bf16: &[u8],
    bias_bf16: Option<&[u8]>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> Result<LinearOutput> {
    let output_bytes = rows
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 resident graph-slot linear output shape overflows usize")?;
    let input_graph_bytes =
        linear_graph_input_bytes(graph_key, input_dim, "CUDA BF16 resident graph-slot linear")?;
    let output_graph_bytes = linear_graph_output_bytes(
        graph_key,
        output_dim,
        "CUDA BF16 resident graph-slot linear",
    )?;
    let weight_buffer = resident_weight_buffer_from_registry(
        weight_name,
        weight_bf16,
        "BF16 resident linear weight",
    )?;
    let signature = linear_graph_signature(graph_key, input_dim, output_dim, bias_bf16.is_some());

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let input_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            input_graph_bytes,
            "BF16 resident linear input",
        )?;
        let output_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            output_graph_bytes,
            "BF16 resident linear output",
        )?;
        let bias_buffer = if let Some(bias_bf16) = bias_bf16 {
            let buffer = slot.buffer(
                library,
                CoordinatorCudaScratchSlot::D,
                bias_bf16.len(),
                "BF16 resident linear bias",
            )?;
            slot.workspace
                .copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::D,
                    bias_bf16,
                    "BF16 resident linear bias",
                    cuda_stream,
                )
                .context("async copying BF16 resident linear bias to device")?;
            Some(buffer)
        } else {
            None
        };

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::A,
                input_bf16,
                "BF16 resident linear input",
                cuda_stream,
            )
            .context("async copying BF16 resident linear input to device")?;
        capture_or_update_layer_linear_bf16_graph(
            library,
            slot,
            signature,
            input_buffer,
            weight_buffer,
            bias_buffer,
            output_buffer,
            rows,
            input_dim,
            output_dim,
            "BF16 resident linear",
        )?;
        let mut out_bytes = vec![0_u8; output_bytes];
        unsafe {
            library
                .copy_d2h_async(&mut out_bytes, output_buffer, cuda_stream)
                .context("async copying BF16 resident linear output to host")?;
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing BF16 resident linear graph slot stream")?;
        }

        Ok(LinearOutput {
            values: bf16_values_to_f32(&out_bytes),
            backend: CUDA_REFERENCE_LINEAR_BF16_RESIDENT_WEIGHT_BACKEND,
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_linear_rows_bf16_resident_weight_legacy(
    weight_name: &str,
    input_bf16: &[u8],
    weight_bf16: &[u8],
    bias_bf16: Option<&[u8]>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> Result<LinearOutput> {
    let library = cuda_native_library()?;
    let output_bytes = rows
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 resident linear output shape overflows usize")?;
    let weight_buffer = resident_weight_buffer_from_registry(
        weight_name,
        weight_bf16,
        "BF16 resident linear weight",
    )?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let input_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        input_bf16.len(),
        "BF16 resident linear input",
    )?;
    let output_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        output_bytes,
        "BF16 resident linear output",
    )?;
    let bias_buffer = if let Some(bias_bf16) = bias_bf16 {
        let buffer = workspace.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            bias_bf16.len(),
            "BF16 resident linear bias",
        )?;
        workspace
            .copy_h2d_to_slot(
                library,
                CoordinatorCudaScratchSlot::D,
                bias_bf16,
                "BF16 resident linear bias",
            )
            .context("copying BF16 resident linear bias to device")?;
        Some(buffer)
    } else {
        None
    };

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            input_bf16,
            "BF16 resident linear input",
        )
        .context("copying BF16 resident linear input to device")?;
    library
        .cuda_linear_bf16_cublas(
            input_buffer,
            weight_buffer,
            bias_buffer,
            output_buffer,
            rows,
            input_dim,
            output_dim,
        )
        .context("executing CUDA BF16 resident linear")?;
    let mut out_bytes = vec![0_u8; output_bytes];
    library
        .copy_d2h(&mut out_bytes, output_buffer)
        .context("copying BF16 resident linear output to host")?;

    Ok(LinearOutput {
        values: bf16_values_to_f32(&out_bytes),
        backend: CUDA_REFERENCE_LINEAR_BF16_RESIDENT_WEIGHT_BACKEND,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_linear_rows_bf16_preloaded_resident_weight(
    weight_name: &str,
    input_bf16: &[u8],
    bias_bf16: Option<&[u8]>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    view: LinearResidentView,
) -> Result<LinearOutput> {
    if let Some(graph_key) = coord_linear_graph_key_for_weight_name(weight_name, rows)? {
        return cuda_linear_rows_bf16_preloaded_resident_weight_graph_slot(
            &graph_key,
            weight_name,
            input_bf16,
            bias_bf16,
            rows,
            input_dim,
            output_dim,
            view,
        );
    }
    cuda_linear_rows_bf16_preloaded_resident_weight_legacy(
        weight_name,
        input_bf16,
        bias_bf16,
        rows,
        input_dim,
        output_dim,
        view,
    )
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_linear_rows_bf16_preloaded_resident_weight_device_input(
    weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    bias_bf16: Option<&[u8]>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    view: LinearResidentView,
) -> Result<LinearOutput> {
    if let Some(graph_key) = coord_linear_graph_key_for_weight_name(weight_name, rows)? {
        return cuda_linear_rows_bf16_preloaded_resident_weight_device_input_graph_slot(
            &graph_key,
            weight_name,
            input_buffer,
            bias_bf16,
            rows,
            input_dim,
            output_dim,
            view,
        );
    }
    cuda_linear_rows_bf16_preloaded_resident_weight_device_input_legacy(
        weight_name,
        input_buffer,
        bias_bf16,
        rows,
        input_dim,
        output_dim,
        view,
    )
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_linear_rows_bf16_preloaded_resident_weight_device_output(
    weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    bias_bf16: Option<&[u8]>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    view: LinearResidentView,
) -> Result<DeviceBf16Output> {
    if let Some(graph_key) = coord_linear_graph_key_for_weight_name(weight_name, rows)? {
        return cuda_linear_rows_bf16_preloaded_resident_weight_device_output_graph_slot(
            &graph_key,
            weight_name,
            input_buffer,
            bias_bf16,
            rows,
            input_dim,
            output_dim,
            view,
        );
    }
    cuda_linear_rows_bf16_preloaded_resident_weight_device_output_legacy(
        weight_name,
        input_buffer,
        bias_bf16,
        rows,
        input_dim,
        output_dim,
        view,
    )
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_linear_rows_bf16_preloaded_resident_weight_padded_device_input(
    weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    bias_bf16: Option<&[u8]>,
    rows: usize,
    full_input_dim: usize,
    output_dim: usize,
    view: LinearPaddedDeviceInputView,
) -> Result<LinearOutput> {
    if let Some(graph_key) = coord_linear_graph_key_for_weight_name(weight_name, rows)? {
        return cuda_linear_rows_bf16_preloaded_resident_weight_padded_device_input_graph_slot(
            &graph_key,
            weight_name,
            input_buffer,
            bias_bf16,
            rows,
            full_input_dim,
            output_dim,
            view,
        );
    }
    cuda_linear_rows_bf16_preloaded_resident_weight_padded_device_input_legacy(
        weight_name,
        input_buffer,
        bias_bf16,
        rows,
        full_input_dim,
        output_dim,
        view,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn cuda_linear_residual_add_rows_bf16_preloaded_resident_weight_device_input(
    weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    bias_bf16: Option<&[u8]>,
    residual_bf16: &[u8],
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    view: LinearResidentView,
) -> Result<LinearResidualAddOutput> {
    if let Some(graph_key) = coord_linear_graph_key_for_weight_name(weight_name, rows)? {
        return cuda_linear_residual_add_rows_bf16_preloaded_resident_weight_device_input_graph_slot(
            &graph_key,
            weight_name,
            input_buffer,
            bias_bf16,
            residual_bf16,
            rows,
            input_dim,
            output_dim,
            view,
        );
    }
    let linear = cuda_linear_rows_bf16_preloaded_resident_weight_device_input_legacy(
        weight_name,
        input_buffer,
        bias_bf16,
        rows,
        input_dim,
        output_dim,
        view,
    )?;
    host_linear_residual_add_bf16_output(linear, residual_bf16)
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn cuda_linear_residual_add_rows_bf16_preloaded_resident_weight_padded_device_input(
    weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    bias_bf16: Option<&[u8]>,
    residual_bf16: &[u8],
    rows: usize,
    full_input_dim: usize,
    output_dim: usize,
    view: LinearPaddedDeviceInputView,
) -> Result<LinearResidualAddOutput> {
    if let Some(graph_key) = coord_linear_graph_key_for_weight_name(weight_name, rows)? {
        return cuda_linear_residual_add_rows_bf16_preloaded_resident_weight_padded_device_input_graph_slot(
            &graph_key,
            weight_name,
            input_buffer,
            bias_bf16,
            residual_bf16,
            rows,
            full_input_dim,
            output_dim,
            view,
        );
    }
    let linear = cuda_linear_rows_bf16_preloaded_resident_weight_padded_device_input_legacy(
        weight_name,
        input_buffer,
        bias_bf16,
        rows,
        full_input_dim,
        output_dim,
        view,
    )?;
    host_linear_residual_add_bf16_output(linear, residual_bf16)
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_linear_residual_add_rows_bf16_preloaded_resident_weight_device_input_device_output(
    weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    bias_bf16: Option<&[u8]>,
    residual_bf16: &[u8],
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    view: LinearResidentView,
) -> Result<LinearResidualAddDeviceOutput> {
    if let Some(graph_key) = coord_linear_graph_key_for_weight_name(weight_name, rows)? {
        return cuda_linear_residual_add_rows_bf16_preloaded_resident_weight_device_input_device_output_graph_slot(
            &graph_key,
            weight_name,
            input_buffer,
            bias_bf16,
            residual_bf16,
            rows,
            input_dim,
            output_dim,
            view,
        );
    }
    cuda_linear_residual_add_rows_bf16_preloaded_resident_weight_device_input_device_output_legacy(
        weight_name,
        input_buffer,
        bias_bf16,
        residual_bf16,
        rows,
        input_dim,
        output_dim,
        view,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_linear_residual_add_rows_bf16_preloaded_resident_weight_padded_device_input_device_output(
    weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    bias_bf16: Option<&[u8]>,
    residual_bf16: &[u8],
    rows: usize,
    full_input_dim: usize,
    output_dim: usize,
    view: LinearPaddedDeviceInputView,
) -> Result<LinearResidualAddDeviceOutput> {
    if let Some(graph_key) = coord_linear_graph_key_for_weight_name(weight_name, rows)? {
        return cuda_linear_residual_add_rows_bf16_preloaded_resident_weight_padded_device_input_device_output_graph_slot(
            &graph_key,
            weight_name,
            input_buffer,
            bias_bf16,
            residual_bf16,
            rows,
            full_input_dim,
            output_dim,
            view,
        );
    }
    cuda_linear_residual_add_rows_bf16_preloaded_resident_weight_padded_device_input_device_output_legacy(
        weight_name,
        input_buffer,
        bias_bf16,
        residual_bf16,
        rows,
        full_input_dim,
        output_dim,
        view,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_linear_rows_bf16_preloaded_resident_weight_graph_slot(
    graph_key: &CoordinatorGraphKey,
    weight_name: &str,
    input_bf16: &[u8],
    bias_bf16: Option<&[u8]>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    view: LinearResidentView,
) -> Result<LinearOutput> {
    let output_bytes = rows
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 preloaded resident graph-slot linear output shape overflows usize")?;
    let input_graph_bytes = linear_graph_input_bytes(
        graph_key,
        input_dim,
        "CUDA BF16 preloaded resident graph-slot linear",
    )?;
    let output_graph_bytes = linear_graph_output_bytes(
        graph_key,
        output_dim,
        "CUDA BF16 preloaded resident graph-slot linear",
    )?;
    let weight_buffer = preloaded_resident_weight_device_buffer_view(
        weight_name,
        view.full_bytes,
        view.offset_bytes,
        view.view_bytes,
    )?;
    let signature = linear_graph_signature(graph_key, input_dim, output_dim, bias_bf16.is_some());

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let input_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            input_graph_bytes,
            "BF16 preloaded resident linear input",
        )?;
        let output_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            output_graph_bytes,
            "BF16 preloaded resident linear output",
        )?;
        let bias_buffer = if let Some(bias_bf16) = bias_bf16 {
            let buffer = slot.buffer(
                library,
                CoordinatorCudaScratchSlot::D,
                bias_bf16.len(),
                "BF16 preloaded resident linear bias",
            )?;
            slot.workspace
                .copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::D,
                    bias_bf16,
                    "BF16 preloaded resident linear bias",
                    cuda_stream,
                )
                .context("async copying BF16 preloaded resident linear bias to device")?;
            Some(buffer)
        } else {
            None
        };

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::A,
                input_bf16,
                "BF16 preloaded resident linear input",
                cuda_stream,
            )
            .context("async copying BF16 preloaded resident linear input to device")?;
        capture_or_update_layer_linear_bf16_graph(
            library,
            slot,
            signature,
            input_buffer,
            weight_buffer,
            bias_buffer,
            output_buffer,
            rows,
            input_dim,
            output_dim,
            "BF16 preloaded resident linear",
        )?;
        let mut out_bytes = vec![0_u8; output_bytes];
        unsafe {
            library
                .copy_d2h_async(&mut out_bytes, output_buffer, cuda_stream)
                .context("async copying BF16 preloaded resident linear output to host")?;
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing BF16 preloaded resident linear graph slot stream")?;
        }

        Ok(LinearOutput {
            values: bf16_values_to_f32(&out_bytes),
            backend: CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
        })
    })
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_linear_rows_bf16_preloaded_resident_weight_padded_device_input_graph_slot(
    graph_key: &CoordinatorGraphKey,
    weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    bias_bf16: Option<&[u8]>,
    rows: usize,
    full_input_dim: usize,
    output_dim: usize,
    view: LinearPaddedDeviceInputView,
) -> Result<LinearOutput> {
    let output_bytes = rows
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "CUDA BF16 preloaded resident padded device-input graph-slot linear output shape overflows usize",
        )?;
    let weight_buffer = preloaded_resident_weight_device_buffer_view(
        weight_name,
        view.weight.full_bytes,
        view.weight.offset_bytes,
        view.weight.view_bytes,
    )?;
    if input_buffer.device_id != weight_buffer.device_id {
        anyhow::bail!(
            "CUDA BF16 preloaded resident padded device-input linear buffers are on different devices: input={} weight={}",
            input_buffer.device_id,
            weight_buffer.device_id
        );
    }
    let active_input_dim = view
        .active_row_bytes
        .checked_div(std::mem::size_of::<u16>())
        .context("CUDA BF16 preloaded resident padded device-input linear active row width overflows usize")?;
    let graph_padded_input_bytes = padded_linear_graph_padded_input_bytes(
        graph_key,
        view,
        "CUDA BF16 preloaded resident padded device-input graph-slot linear",
    )?;
    let graph_output_bytes = padded_linear_graph_output_bytes(
        graph_key,
        output_dim,
        "CUDA BF16 preloaded resident padded device-input graph-slot linear",
    )?;
    let signature = padded_linear_graph_signature(
        graph_key,
        active_input_dim,
        full_input_dim,
        output_dim,
        bias_bf16.is_some(),
    );

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let padded_input_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            graph_padded_input_bytes,
            "BF16 preloaded resident padded device-input linear padded input",
        )?;
        let output_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            graph_output_bytes,
            "BF16 preloaded resident padded device-input linear output",
        )?;
        let bias_buffer = if let Some(bias_bf16) = bias_bf16 {
            let buffer = slot.buffer(
                library,
                CoordinatorCudaScratchSlot::D,
                bias_bf16.len(),
                "BF16 preloaded resident padded device-input linear bias",
            )?;
            slot.workspace
                .copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::D,
                    bias_bf16,
                    "BF16 preloaded resident padded device-input linear bias",
                    cuda_stream,
                )
                .context(
                    "async copying BF16 preloaded resident padded device-input linear bias to device",
                )?;
            Some(buffer)
        } else {
            None
        };

        capture_or_update_layer_padded_linear_bf16_graph(
            library,
            slot,
            signature,
            input_buffer,
            padded_input_buffer,
            weight_buffer,
            bias_buffer,
            output_buffer,
            rows,
            graph_key.row_bucket.row_capacity,
            full_input_dim,
            output_dim,
            view,
            "BF16 preloaded resident padded device-input linear",
        )?;
        let mut out_bytes = vec![0_u8; output_bytes];
        unsafe {
            library
                .copy_d2h_async(&mut out_bytes, output_buffer, cuda_stream)
                .context(
                    "async copying BF16 preloaded resident padded device-input linear output to host",
                )?;
            library.cuda_stream_synchronize(cuda_stream).context(
                "synchronizing BF16 preloaded resident padded device-input linear graph slot stream",
            )?;
        }

        Ok(LinearOutput {
            values: bf16_values_to_f32(&out_bytes),
            backend: CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
        })
    })
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_linear_rows_bf16_preloaded_resident_weight_device_input_graph_slot(
    graph_key: &CoordinatorGraphKey,
    weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    bias_bf16: Option<&[u8]>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    view: LinearResidentView,
) -> Result<LinearOutput> {
    let output_bytes = rows
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "CUDA BF16 preloaded resident device-input graph-slot linear output shape overflows usize",
        )?;
    let output_graph_bytes = linear_graph_output_bytes(
        graph_key,
        output_dim,
        "CUDA BF16 preloaded resident device-input graph-slot linear",
    )?;
    let weight_buffer = preloaded_resident_weight_device_buffer_view(
        weight_name,
        view.full_bytes,
        view.offset_bytes,
        view.view_bytes,
    )?;
    if input_buffer.device_id != weight_buffer.device_id {
        anyhow::bail!(
            "CUDA BF16 preloaded resident device-input linear buffers are on different devices: input={} weight={}",
            input_buffer.device_id,
            weight_buffer.device_id
        );
    }
    let signature = linear_graph_signature(graph_key, input_dim, output_dim, bias_bf16.is_some());

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let output_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            output_graph_bytes,
            "BF16 preloaded resident device-input linear output",
        )?;
        let bias_buffer = if let Some(bias_bf16) = bias_bf16 {
            let buffer = slot.buffer(
                library,
                CoordinatorCudaScratchSlot::D,
                bias_bf16.len(),
                "BF16 preloaded resident device-input linear bias",
            )?;
            slot.workspace
                .copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::D,
                    bias_bf16,
                    "BF16 preloaded resident device-input linear bias",
                    cuda_stream,
                )
                .context(
                    "async copying BF16 preloaded resident device-input linear bias to device",
                )?;
            Some(buffer)
        } else {
            None
        };

        capture_or_update_layer_linear_bf16_graph(
            library,
            slot,
            signature,
            input_buffer,
            weight_buffer,
            bias_buffer,
            output_buffer,
            rows,
            input_dim,
            output_dim,
            "BF16 preloaded resident device-input linear",
        )?;
        let mut out_bytes = vec![0_u8; output_bytes];
        unsafe {
            library
                .copy_d2h_async(&mut out_bytes, output_buffer, cuda_stream)
                .context(
                    "async copying BF16 preloaded resident device-input linear output to host",
                )?;
            library.cuda_stream_synchronize(cuda_stream).context(
                "synchronizing BF16 preloaded resident device-input linear graph slot stream",
            )?;
        }

        Ok(LinearOutput {
            values: bf16_values_to_f32(&out_bytes),
            backend: CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
        })
    })
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_linear_rows_bf16_preloaded_resident_weight_device_output_graph_slot(
    graph_key: &CoordinatorGraphKey,
    weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    bias_bf16: Option<&[u8]>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    view: LinearResidentView,
) -> Result<DeviceBf16Output> {
    let output_bytes = rows
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "CUDA BF16 preloaded resident device-output graph-slot linear output shape overflows usize",
        )?;
    let weight_buffer = preloaded_resident_weight_device_buffer_view(
        weight_name,
        view.full_bytes,
        view.offset_bytes,
        view.view_bytes,
    )?;
    if input_buffer.device_id != weight_buffer.device_id {
        anyhow::bail!(
            "CUDA BF16 preloaded resident device-output linear buffers are on different devices: input={} weight={}",
            input_buffer.device_id,
            weight_buffer.device_id
        );
    }
    let signature = linear_graph_signature(graph_key, input_dim, output_dim, bias_bf16.is_some());

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let output_buffer = OwnedCoordinatorDeviceBuffer::new(
            library,
            output_bytes,
            "BF16 preloaded resident device-output linear output",
        )?;
        let bias_buffer = if let Some(bias_bf16) = bias_bf16 {
            let buffer = slot.buffer(
                library,
                CoordinatorCudaScratchSlot::D,
                bias_bf16.len(),
                "BF16 preloaded resident device-output linear bias",
            )?;
            slot.workspace
                .copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::D,
                    bias_bf16,
                    "BF16 preloaded resident device-output linear bias",
                    cuda_stream,
                )
                .context(
                    "async copying BF16 preloaded resident device-output linear bias to device",
                )?;
            Some(buffer)
        } else {
            None
        };

        capture_or_update_layer_linear_bf16_graph(
            library,
            slot,
            signature,
            input_buffer,
            weight_buffer,
            bias_buffer,
            output_buffer.buffer,
            rows,
            input_dim,
            output_dim,
            "BF16 preloaded resident device-output linear",
        )?;
        unsafe {
            library.cuda_stream_synchronize(cuda_stream).context(
                "synchronizing BF16 preloaded resident device-output linear graph slot stream",
            )?;
        }

        Ok(DeviceBf16Output {
            buffer: output_buffer,
            bytes: output_bytes,
            rows,
            values_per_row: output_dim,
            backend: CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
        })
    })
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn cuda_linear_residual_add_rows_bf16_preloaded_resident_weight_device_input_graph_slot(
    graph_key: &CoordinatorGraphKey,
    weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    bias_bf16: Option<&[u8]>,
    residual_bf16: &[u8],
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    view: LinearResidentView,
) -> Result<LinearResidualAddOutput> {
    let output_bytes = rows
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "CUDA BF16 preloaded resident device-input graph-slot linear residual-add output shape overflows usize",
        )?;
    let output_graph_bytes = linear_graph_output_bytes(
        graph_key,
        output_dim,
        "CUDA BF16 preloaded resident device-input graph-slot linear residual-add",
    )?;
    let weight_buffer = preloaded_resident_weight_device_buffer_view(
        weight_name,
        view.full_bytes,
        view.offset_bytes,
        view.view_bytes,
    )?;
    if input_buffer.device_id != weight_buffer.device_id {
        anyhow::bail!(
            "CUDA BF16 preloaded resident device-input linear residual-add buffers are on different devices: input={} weight={}",
            input_buffer.device_id,
            weight_buffer.device_id
        );
    }

    let signature = linear_graph_signature(graph_key, input_dim, output_dim, bias_bf16.is_some());

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let residual_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::B,
            output_graph_bytes,
            "BF16 preloaded resident device-input linear residual-add residual",
        )?;
        let linear_output_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            output_graph_bytes,
            "BF16 preloaded resident device-input linear residual-add delta",
        )?;
        let residual_output_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::E,
            output_graph_bytes,
            "BF16 preloaded resident device-input linear residual-add output",
        )?;
        let bias_buffer = if let Some(bias_bf16) = bias_bf16 {
            let buffer = slot.buffer(
                library,
                CoordinatorCudaScratchSlot::D,
                bias_bf16.len(),
                "BF16 preloaded resident device-input linear residual-add bias",
            )?;
            slot.workspace
                .copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::D,
                    bias_bf16,
                    "BF16 preloaded resident device-input linear residual-add bias",
                    cuda_stream,
                )
                .context(
                    "async copying BF16 preloaded resident device-input linear residual-add bias to device",
                )?;
            Some(buffer)
        } else {
            None
        };

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::B,
                residual_bf16,
                "BF16 preloaded resident device-input linear residual-add residual",
                cuda_stream,
            )
            .context(
                "async copying BF16 preloaded resident device-input linear residual-add residual to device",
            )?;
        capture_or_update_layer_linear_residual_add_bf16_graph(
            library,
            slot,
            signature,
            input_buffer,
            weight_buffer,
            bias_buffer,
            linear_output_buffer,
            residual_buffer,
            residual_output_buffer,
            rows,
            input_dim,
            output_dim,
            "BF16 preloaded resident device-input linear residual-add",
        )?;
        let mut linear_out_bytes = vec![0_u8; output_bytes];
        let mut residual_out_bytes = vec![0_u8; output_bytes];
        unsafe {
            library
                .copy_d2h_async(&mut linear_out_bytes, linear_output_buffer, cuda_stream)
                .context(
                    "async copying BF16 preloaded resident device-input linear residual-add delta to host",
                )?;
            library
                .copy_d2h_async(&mut residual_out_bytes, residual_output_buffer, cuda_stream)
                .context(
                    "async copying BF16 preloaded resident device-input linear residual-add output to host",
                )?;
            library.cuda_stream_synchronize(cuda_stream).context(
                "synchronizing BF16 preloaded resident device-input linear residual-add graph slot stream",
            )?;
        }

        Ok(LinearResidualAddOutput {
            linear_values: bf16_values_to_f32(&linear_out_bytes),
            residual_values: bf16_values_to_f32(&residual_out_bytes),
            linear_backend: CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
            residual_add_backend: CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_linear_residual_add_rows_bf16_preloaded_resident_weight_device_input_device_output_graph_slot(
    graph_key: &CoordinatorGraphKey,
    weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    bias_bf16: Option<&[u8]>,
    residual_bf16: &[u8],
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    view: LinearResidentView,
) -> Result<LinearResidualAddDeviceOutput> {
    let output_bytes = rows
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "CUDA BF16 preloaded resident device-input graph-slot linear residual-add device-output shape overflows usize",
        )?;
    let output_graph_bytes = linear_graph_output_bytes(
        graph_key,
        output_dim,
        "CUDA BF16 preloaded resident device-input graph-slot linear residual-add device-output",
    )?;
    let weight_buffer = preloaded_resident_weight_device_buffer_view(
        weight_name,
        view.full_bytes,
        view.offset_bytes,
        view.view_bytes,
    )?;
    if input_buffer.device_id != weight_buffer.device_id {
        anyhow::bail!(
            "CUDA BF16 preloaded resident device-input linear residual-add device-output buffers are on different devices: input={} weight={}",
            input_buffer.device_id,
            weight_buffer.device_id
        );
    }

    let signature = linear_graph_signature(graph_key, input_dim, output_dim, bias_bf16.is_some());

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let residual_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::B,
            output_graph_bytes,
            "BF16 preloaded resident device-input linear residual-add device-output residual",
        )?;
        let linear_output_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            output_graph_bytes,
            "BF16 preloaded resident device-input linear residual-add device-output delta",
        )?;
        let residual_output_buffer = OwnedCoordinatorDeviceBuffer::new(
            library,
            output_bytes,
            "BF16 preloaded resident device-input linear residual-add device output",
        )?;
        let bias_buffer = if let Some(bias_bf16) = bias_bf16 {
            let buffer = slot.buffer(
                library,
                CoordinatorCudaScratchSlot::D,
                bias_bf16.len(),
                "BF16 preloaded resident device-input linear residual-add device-output bias",
            )?;
            slot.workspace
                .copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::D,
                    bias_bf16,
                    "BF16 preloaded resident device-input linear residual-add device-output bias",
                    cuda_stream,
                )
                .context(
                    "async copying BF16 preloaded resident device-input linear residual-add device-output bias to device",
                )?;
            Some(buffer)
        } else {
            None
        };

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::B,
                residual_bf16,
                "BF16 preloaded resident device-input linear residual-add device-output residual",
                cuda_stream,
            )
            .context(
                "async copying BF16 preloaded resident device-input linear residual-add device-output residual to device",
            )?;
        capture_or_update_layer_linear_residual_add_bf16_graph(
            library,
            slot,
            signature,
            input_buffer,
            weight_buffer,
            bias_buffer,
            linear_output_buffer,
            residual_buffer,
            residual_output_buffer.buffer,
            rows,
            input_dim,
            output_dim,
            "BF16 preloaded resident device-input linear residual-add device-output",
        )?;
        let mut linear_out_bytes = vec![0_u8; output_bytes];
        let mut residual_out_bytes = vec![0_u8; output_bytes];
        unsafe {
            library
                .copy_d2h_async(&mut linear_out_bytes, linear_output_buffer, cuda_stream)
                .context(
                    "async copying BF16 preloaded resident device-input linear residual-add device-output delta to host",
                )?;
            library
                .copy_d2h_async(
                    &mut residual_out_bytes,
                    residual_output_buffer.buffer,
                    cuda_stream,
                )
                .context(
                    "async copying BF16 preloaded resident device-input linear residual-add device output to host",
                )?;
            library.cuda_stream_synchronize(cuda_stream).context(
                "synchronizing BF16 preloaded resident device-input linear residual-add device-output graph slot stream",
            )?;
        }

        Ok(LinearResidualAddDeviceOutput {
            linear_values: bf16_values_to_f32(&linear_out_bytes),
            residual_values: bf16_values_to_f32(&residual_out_bytes),
            residual_device: DeviceBf16Output {
                buffer: residual_output_buffer,
                bytes: output_bytes,
                rows,
                values_per_row: output_dim,
                backend: CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
            },
            linear_backend: CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
            residual_add_backend: CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
        })
    })
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn cuda_linear_residual_add_rows_bf16_preloaded_resident_weight_padded_device_input_graph_slot(
    graph_key: &CoordinatorGraphKey,
    weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    bias_bf16: Option<&[u8]>,
    residual_bf16: &[u8],
    rows: usize,
    full_input_dim: usize,
    output_dim: usize,
    view: LinearPaddedDeviceInputView,
) -> Result<LinearResidualAddOutput> {
    let output_bytes = rows
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "CUDA BF16 preloaded resident padded device-input graph-slot linear residual-add output shape overflows usize",
        )?;
    let weight_buffer = preloaded_resident_weight_device_buffer_view(
        weight_name,
        view.weight.full_bytes,
        view.weight.offset_bytes,
        view.weight.view_bytes,
    )?;
    if input_buffer.device_id != weight_buffer.device_id {
        anyhow::bail!(
            "CUDA BF16 preloaded resident padded device-input linear residual-add buffers are on different devices: input={} weight={}",
            input_buffer.device_id,
            weight_buffer.device_id
        );
    }
    let active_input_dim = view
        .active_row_bytes
        .checked_div(std::mem::size_of::<u16>())
        .context(
            "CUDA BF16 preloaded resident padded device-input active row width overflows usize",
        )?;
    let graph_padded_input_bytes = padded_linear_graph_padded_input_bytes(
        graph_key,
        view,
        "CUDA BF16 preloaded resident padded device-input graph-slot linear residual-add",
    )?;
    let graph_output_bytes = padded_linear_graph_output_bytes(
        graph_key,
        output_dim,
        "CUDA BF16 preloaded resident padded device-input graph-slot linear residual-add",
    )?;
    let signature = padded_linear_residual_graph_signature(
        graph_key,
        active_input_dim,
        full_input_dim,
        output_dim,
        bias_bf16.is_some(),
    );

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let padded_input_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            graph_padded_input_bytes,
            "BF16 preloaded resident padded device-input linear residual-add padded input",
        )?;
        let residual_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::B,
            graph_output_bytes,
            "BF16 preloaded resident padded device-input linear residual-add residual",
        )?;
        let linear_output_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            graph_output_bytes,
            "BF16 preloaded resident padded device-input linear residual-add delta",
        )?;
        let residual_output_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::E,
            graph_output_bytes,
            "BF16 preloaded resident padded device-input linear residual-add output",
        )?;
        let bias_buffer = if let Some(bias_bf16) = bias_bf16 {
            let buffer = slot.buffer(
                library,
                CoordinatorCudaScratchSlot::D,
                bias_bf16.len(),
                "BF16 preloaded resident padded device-input linear residual-add bias",
            )?;
            slot.workspace
                .copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::D,
                    bias_bf16,
                    "BF16 preloaded resident padded device-input linear residual-add bias",
                    cuda_stream,
                )
                .context(
                    "async copying BF16 preloaded resident padded device-input linear residual-add bias to device",
                )?;
            Some(buffer)
        } else {
            None
        };

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::B,
                residual_bf16,
                "BF16 preloaded resident padded device-input linear residual-add residual",
                cuda_stream,
            )
            .context(
                "async copying BF16 preloaded resident padded device-input linear residual-add residual to device",
            )?;
        capture_or_update_layer_padded_linear_residual_add_bf16_graph(
            library,
            slot,
            signature,
            input_buffer,
            padded_input_buffer,
            weight_buffer,
            bias_buffer,
            linear_output_buffer,
            residual_buffer,
            residual_output_buffer,
            rows,
            graph_key.row_bucket.row_capacity,
            full_input_dim,
            output_dim,
            view,
            "BF16 preloaded resident padded device-input linear residual-add",
        )?;
        let mut linear_out_bytes = vec![0_u8; output_bytes];
        let mut residual_out_bytes = vec![0_u8; output_bytes];
        unsafe {
            library
                .copy_d2h_async(&mut linear_out_bytes, linear_output_buffer, cuda_stream)
                .context(
                    "async copying BF16 preloaded resident padded device-input linear residual-add delta to host",
                )?;
            library
                .copy_d2h_async(&mut residual_out_bytes, residual_output_buffer, cuda_stream)
                .context(
                    "async copying BF16 preloaded resident padded device-input linear residual-add output to host",
                )?;
            library.cuda_stream_synchronize(cuda_stream).context(
                "synchronizing BF16 preloaded resident padded device-input linear residual-add graph slot stream",
            )?;
        }

        Ok(LinearResidualAddOutput {
            linear_values: bf16_values_to_f32(&linear_out_bytes),
            residual_values: bf16_values_to_f32(&residual_out_bytes),
            linear_backend: CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
            residual_add_backend: CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_linear_residual_add_rows_bf16_preloaded_resident_weight_padded_device_input_device_output_graph_slot(
    graph_key: &CoordinatorGraphKey,
    weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    bias_bf16: Option<&[u8]>,
    residual_bf16: &[u8],
    rows: usize,
    full_input_dim: usize,
    output_dim: usize,
    view: LinearPaddedDeviceInputView,
) -> Result<LinearResidualAddDeviceOutput> {
    let output_bytes = rows
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "CUDA BF16 preloaded resident padded device-input graph-slot linear residual-add device-output shape overflows usize",
        )?;
    let weight_buffer = preloaded_resident_weight_device_buffer_view(
        weight_name,
        view.weight.full_bytes,
        view.weight.offset_bytes,
        view.weight.view_bytes,
    )?;
    if input_buffer.device_id != weight_buffer.device_id {
        anyhow::bail!(
            "CUDA BF16 preloaded resident padded device-input linear residual-add device-output buffers are on different devices: input={} weight={}",
            input_buffer.device_id,
            weight_buffer.device_id
        );
    }
    let active_input_dim = view
        .active_row_bytes
        .checked_div(std::mem::size_of::<u16>())
        .context("CUDA BF16 preloaded resident padded device-input device-output active row width overflows usize")?;
    let graph_padded_input_bytes = padded_linear_graph_padded_input_bytes(
        graph_key,
        view,
        "CUDA BF16 preloaded resident padded device-input graph-slot linear residual-add device-output",
    )?;
    let graph_output_bytes = padded_linear_graph_output_bytes(
        graph_key,
        output_dim,
        "CUDA BF16 preloaded resident padded device-input graph-slot linear residual-add device-output",
    )?;
    let signature = padded_linear_residual_graph_signature(
        graph_key,
        active_input_dim,
        full_input_dim,
        output_dim,
        bias_bf16.is_some(),
    );

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let padded_input_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            graph_padded_input_bytes,
            "BF16 preloaded resident padded device-input linear residual-add device-output padded input",
        )?;
        let residual_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::B,
            graph_output_bytes,
            "BF16 preloaded resident padded device-input linear residual-add device-output residual",
        )?;
        let linear_output_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            graph_output_bytes,
            "BF16 preloaded resident padded device-input linear residual-add device-output delta",
        )?;
        let residual_output_buffer = OwnedCoordinatorDeviceBuffer::new(
            library,
            output_bytes,
            "BF16 preloaded resident padded device-input linear residual-add device output",
        )?;
        let bias_buffer = if let Some(bias_bf16) = bias_bf16 {
            let buffer = slot.buffer(
                library,
                CoordinatorCudaScratchSlot::D,
                bias_bf16.len(),
                "BF16 preloaded resident padded device-input linear residual-add device-output bias",
            )?;
            slot.workspace
                .copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::D,
                    bias_bf16,
                    "BF16 preloaded resident padded device-input linear residual-add device-output bias",
                    cuda_stream,
                )
                .context(
                    "async copying BF16 preloaded resident padded device-input linear residual-add device-output bias to device",
                )?;
            Some(buffer)
        } else {
            None
        };

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::B,
                residual_bf16,
                "BF16 preloaded resident padded device-input linear residual-add device-output residual",
                cuda_stream,
            )
            .context(
                "async copying BF16 preloaded resident padded device-input linear residual-add device-output residual to device",
            )?;
        capture_or_update_layer_padded_linear_residual_add_bf16_graph(
            library,
            slot,
            signature,
            input_buffer,
            padded_input_buffer,
            weight_buffer,
            bias_buffer,
            linear_output_buffer,
            residual_buffer,
            residual_output_buffer.buffer,
            rows,
            graph_key.row_bucket.row_capacity,
            full_input_dim,
            output_dim,
            view,
            "BF16 preloaded resident padded device-input linear residual-add device-output",
        )?;
        let mut linear_out_bytes = vec![0_u8; output_bytes];
        let mut residual_out_bytes = vec![0_u8; output_bytes];
        unsafe {
            library
                .copy_d2h_async(&mut linear_out_bytes, linear_output_buffer, cuda_stream)
                .context(
                    "async copying BF16 preloaded resident padded device-input linear residual-add device-output delta to host",
                )?;
            library
                .copy_d2h_async(
                    &mut residual_out_bytes,
                    residual_output_buffer.buffer,
                    cuda_stream,
                )
                .context(
                    "async copying BF16 preloaded resident padded device-input linear residual-add device output to host",
                )?;
            library.cuda_stream_synchronize(cuda_stream).context(
                "synchronizing BF16 preloaded resident padded device-input linear residual-add device-output graph slot stream",
            )?;
        }

        Ok(LinearResidualAddDeviceOutput {
            linear_values: bf16_values_to_f32(&linear_out_bytes),
            residual_values: bf16_values_to_f32(&residual_out_bytes),
            residual_device: DeviceBf16Output {
                buffer: residual_output_buffer,
                bytes: output_bytes,
                rows,
                values_per_row: output_dim,
                backend: CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
            },
            linear_backend: CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
            residual_add_backend: CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
        })
    })
}

unsafe fn copy_rows_to_padded_device_input_async(
    library: &NativeLibrary,
    input_buffer: GlmrtDeviceBuffer,
    padded_input_buffer: GlmrtDeviceBuffer,
    rows: usize,
    graph_rows: usize,
    view: LinearPaddedDeviceInputView,
    cuda_stream: *mut c_void,
) -> Result<()> {
    for row in 0..graph_rows {
        let dst_offset = row
            .checked_mul(view.padded_row_bytes)
            .context("padded device-input linear destination row offset overflow usize")?;
        let (src, bytes) = if row < rows {
            let src_offset = row
                .checked_mul(view.active_row_bytes)
                .context("padded device-input linear source row offset overflow usize")?;
            (
                device_buffer_byte_view(
                    input_buffer,
                    src_offset,
                    view.active_row_bytes,
                    "padded device-input linear active row",
                )?,
                view.active_row_bytes,
            )
        } else {
            (
                device_buffer_byte_view(
                    padded_input_buffer,
                    dst_offset,
                    1,
                    "padded device-input linear inactive row self-copy source",
                )?,
                1,
            )
        };
        unsafe {
            library
                .cuda_kv_cache_write_bytes_async(
                    src,
                    padded_input_buffer,
                    dst_offset,
                    bytes,
                    cuda_stream,
                )
                .context("async copying active BF16 row into padded device-input linear buffer")?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_linear_rows_bf16_preloaded_resident_weight_legacy(
    weight_name: &str,
    input_bf16: &[u8],
    bias_bf16: Option<&[u8]>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    view: LinearResidentView,
) -> Result<LinearOutput> {
    let library = cuda_native_library()?;
    let output_bytes = rows
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 preloaded resident linear output shape overflows usize")?;
    let weight_buffer = preloaded_resident_weight_device_buffer_view(
        weight_name,
        view.full_bytes,
        view.offset_bytes,
        view.view_bytes,
    )?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let input_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        input_bf16.len(),
        "BF16 preloaded resident linear input",
    )?;
    let output_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        output_bytes,
        "BF16 preloaded resident linear output",
    )?;
    let bias_buffer = if let Some(bias_bf16) = bias_bf16 {
        let buffer = workspace.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            bias_bf16.len(),
            "BF16 preloaded resident linear bias",
        )?;
        workspace
            .copy_h2d_to_slot(
                library,
                CoordinatorCudaScratchSlot::D,
                bias_bf16,
                "BF16 preloaded resident linear bias",
            )
            .context("copying BF16 preloaded resident linear bias to device")?;
        Some(buffer)
    } else {
        None
    };

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            input_bf16,
            "BF16 preloaded resident linear input",
        )
        .context("copying BF16 preloaded resident linear input to device")?;
    library
        .cuda_linear_bf16_cublas(
            input_buffer,
            weight_buffer,
            bias_buffer,
            output_buffer,
            rows,
            input_dim,
            output_dim,
        )
        .context("executing CUDA BF16 preloaded resident linear")?;
    let mut out_bytes = vec![0_u8; output_bytes];
    library
        .copy_d2h(&mut out_bytes, output_buffer)
        .context("copying BF16 preloaded resident linear output to host")?;

    Ok(LinearOutput {
        values: bf16_values_to_f32(&out_bytes),
        backend: CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
    })
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_linear_rows_bf16_preloaded_resident_weight_padded_device_input_legacy(
    weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    bias_bf16: Option<&[u8]>,
    rows: usize,
    full_input_dim: usize,
    output_dim: usize,
    view: LinearPaddedDeviceInputView,
) -> Result<LinearOutput> {
    let library = cuda_native_library()?;
    let output_bytes = rows
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "CUDA BF16 preloaded resident padded device-input linear output shape overflows usize",
        )?;
    let weight_buffer = preloaded_resident_weight_device_buffer_view(
        weight_name,
        view.weight.full_bytes,
        view.weight.offset_bytes,
        view.weight.view_bytes,
    )?;
    if input_buffer.device_id != weight_buffer.device_id {
        anyhow::bail!(
            "CUDA BF16 preloaded resident padded device-input linear buffers are on different devices: input={} weight={}",
            input_buffer.device_id,
            weight_buffer.device_id
        );
    }
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let padded_input_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        view.padded_input_bytes,
        "BF16 preloaded resident padded device-input linear padded input",
    )?;
    let output_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        output_bytes,
        "BF16 preloaded resident padded device-input linear output",
    )?;
    let bias_buffer = if let Some(bias_bf16) = bias_bf16 {
        let buffer = workspace.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            bias_bf16.len(),
            "BF16 preloaded resident padded device-input linear bias",
        )?;
        workspace
            .copy_h2d_to_slot(
                library,
                CoordinatorCudaScratchSlot::D,
                bias_bf16,
                "BF16 preloaded resident padded device-input linear bias",
            )
            .context("copying BF16 preloaded resident padded device-input linear bias to device")?;
        Some(buffer)
    } else {
        None
    };

    library
        .cuda_zero_bytes(padded_input_buffer, view.padded_input_bytes)
        .context("zeroing BF16 preloaded resident padded device-input linear input")?;
    for row in 0..rows {
        let src_offset = row
            .checked_mul(view.active_row_bytes)
            .context("padded device-input linear source row offset overflow usize")?;
        let dst_offset = row
            .checked_mul(view.padded_row_bytes)
            .context("padded device-input linear destination row offset overflow usize")?;
        let src = device_buffer_byte_view(
            input_buffer,
            src_offset,
            view.active_row_bytes,
            "padded device-input linear active row",
        )?;
        library
            .cuda_kv_cache_write_bytes(src, padded_input_buffer, dst_offset, view.active_row_bytes)
            .context("copying active BF16 row into padded device-input linear buffer")?;
    }
    library
        .cuda_linear_bf16_cublas(
            padded_input_buffer,
            weight_buffer,
            bias_buffer,
            output_buffer,
            rows,
            full_input_dim,
            output_dim,
        )
        .context("executing CUDA BF16 preloaded resident padded device-input linear")?;
    let mut out_bytes = vec![0_u8; output_bytes];
    library
        .copy_d2h(&mut out_bytes, output_buffer)
        .context("copying BF16 preloaded resident padded device-input linear output to host")?;

    Ok(LinearOutput {
        values: bf16_values_to_f32(&out_bytes),
        backend: CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
    })
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_linear_rows_bf16_preloaded_resident_weight_device_input_legacy(
    weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    bias_bf16: Option<&[u8]>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    view: LinearResidentView,
) -> Result<LinearOutput> {
    let library = cuda_native_library()?;
    let output_bytes = rows
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 preloaded resident device-input linear output shape overflows usize")?;
    let weight_buffer = preloaded_resident_weight_device_buffer_view(
        weight_name,
        view.full_bytes,
        view.offset_bytes,
        view.view_bytes,
    )?;
    if input_buffer.device_id != weight_buffer.device_id {
        anyhow::bail!(
            "CUDA BF16 preloaded resident device-input linear buffers are on different devices: input={} weight={}",
            input_buffer.device_id,
            weight_buffer.device_id
        );
    }
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let output_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        output_bytes,
        "BF16 preloaded resident device-input linear output",
    )?;
    let bias_buffer = if let Some(bias_bf16) = bias_bf16 {
        let buffer = workspace.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            bias_bf16.len(),
            "BF16 preloaded resident device-input linear bias",
        )?;
        workspace
            .copy_h2d_to_slot(
                library,
                CoordinatorCudaScratchSlot::D,
                bias_bf16,
                "BF16 preloaded resident device-input linear bias",
            )
            .context("copying BF16 preloaded resident device-input linear bias to device")?;
        Some(buffer)
    } else {
        None
    };

    library
        .cuda_linear_bf16_cublas(
            input_buffer,
            weight_buffer,
            bias_buffer,
            output_buffer,
            rows,
            input_dim,
            output_dim,
        )
        .context("executing CUDA BF16 preloaded resident device-input linear")?;
    let mut out_bytes = vec![0_u8; output_bytes];
    library
        .copy_d2h(&mut out_bytes, output_buffer)
        .context("copying BF16 preloaded resident device-input linear output to host")?;

    Ok(LinearOutput {
        values: bf16_values_to_f32(&out_bytes),
        backend: CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
    })
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_linear_rows_bf16_preloaded_resident_weight_device_output_legacy(
    weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    bias_bf16: Option<&[u8]>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    view: LinearResidentView,
) -> Result<DeviceBf16Output> {
    let library = cuda_native_library()?;
    let output_bytes = rows
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "CUDA BF16 preloaded resident device-output linear output shape overflows usize",
        )?;
    let weight_buffer = preloaded_resident_weight_device_buffer_view(
        weight_name,
        view.full_bytes,
        view.offset_bytes,
        view.view_bytes,
    )?;
    if input_buffer.device_id != weight_buffer.device_id {
        anyhow::bail!(
            "CUDA BF16 preloaded resident device-output linear buffers are on different devices: input={} weight={}",
            input_buffer.device_id,
            weight_buffer.device_id
        );
    }
    let output_buffer = OwnedCoordinatorDeviceBuffer::new(
        library,
        output_bytes,
        "BF16 preloaded resident device-output linear output",
    )?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let bias_buffer = if let Some(bias_bf16) = bias_bf16 {
        let buffer = workspace.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            bias_bf16.len(),
            "BF16 preloaded resident device-output linear bias",
        )?;
        workspace
            .copy_h2d_to_slot(
                library,
                CoordinatorCudaScratchSlot::D,
                bias_bf16,
                "BF16 preloaded resident device-output linear bias",
            )
            .context("copying BF16 preloaded resident device-output linear bias to device")?;
        Some(buffer)
    } else {
        None
    };

    library
        .cuda_linear_bf16_cublas(
            input_buffer,
            weight_buffer,
            bias_buffer,
            output_buffer.buffer,
            rows,
            input_dim,
            output_dim,
        )
        .context("executing CUDA BF16 preloaded resident device-output linear")?;

    Ok(DeviceBf16Output {
        buffer: output_buffer,
        bytes: output_bytes,
        rows,
        values_per_row: output_dim,
        backend: CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_linear_residual_add_rows_bf16_preloaded_resident_weight_device_input_device_output_legacy(
    weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    bias_bf16: Option<&[u8]>,
    residual_bf16: &[u8],
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    view: LinearResidentView,
) -> Result<LinearResidualAddDeviceOutput> {
    let library = cuda_native_library()?;
    let output_bytes = rows
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "CUDA BF16 preloaded resident device-input linear residual-add device-output shape overflows usize",
        )?;
    let weight_buffer = preloaded_resident_weight_device_buffer_view(
        weight_name,
        view.full_bytes,
        view.offset_bytes,
        view.view_bytes,
    )?;
    if input_buffer.device_id != weight_buffer.device_id {
        anyhow::bail!(
            "CUDA BF16 preloaded resident device-input linear residual-add device-output buffers are on different devices: input={} weight={}",
            input_buffer.device_id,
            weight_buffer.device_id
        );
    }
    let residual_output_buffer = OwnedCoordinatorDeviceBuffer::new(
        library,
        output_bytes,
        "BF16 preloaded resident device-input linear residual-add device output",
    )?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let residual_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::B,
        residual_bf16.len(),
        "BF16 preloaded resident device-input linear residual-add device-output residual",
    )?;
    let linear_output_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        output_bytes,
        "BF16 preloaded resident device-input linear residual-add device-output delta",
    )?;
    let bias_buffer = if let Some(bias_bf16) = bias_bf16 {
        let buffer = workspace.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            bias_bf16.len(),
            "BF16 preloaded resident device-input linear residual-add device-output bias",
        )?;
        workspace
            .copy_h2d_to_slot(
                library,
                CoordinatorCudaScratchSlot::D,
                bias_bf16,
                "BF16 preloaded resident device-input linear residual-add device-output bias",
            )
            .context(
                "copying BF16 preloaded resident device-input linear residual-add device-output bias to device",
            )?;
        Some(buffer)
    } else {
        None
    };

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::B,
            residual_bf16,
            "BF16 preloaded resident device-input linear residual-add device-output residual",
        )
        .context(
            "copying BF16 preloaded resident device-input linear residual-add device-output residual to device",
        )?;
    library
        .cuda_linear_bf16_cublas(
            input_buffer,
            weight_buffer,
            bias_buffer,
            linear_output_buffer,
            rows,
            input_dim,
            output_dim,
        )
        .context(
            "executing CUDA BF16 preloaded resident device-input linear residual-add device-output linear",
        )?;
    library
        .cuda_residual_add_bf16(
            residual_buffer,
            linear_output_buffer,
            residual_output_buffer.buffer,
            output_bytes / std::mem::size_of::<u16>(),
        )
        .context(
            "executing CUDA BF16 preloaded resident device-input linear residual-add device-output",
        )?;
    let mut linear_out_bytes = vec![0_u8; output_bytes];
    let mut residual_out_bytes = vec![0_u8; output_bytes];
    library
        .copy_d2h(&mut linear_out_bytes, linear_output_buffer)
        .context(
            "copying BF16 preloaded resident device-input linear residual-add device-output delta to host",
        )?;
    library
        .copy_d2h(&mut residual_out_bytes, residual_output_buffer.buffer)
        .context(
            "copying BF16 preloaded resident device-input linear residual-add device output to host",
        )?;

    Ok(LinearResidualAddDeviceOutput {
        linear_values: bf16_values_to_f32(&linear_out_bytes),
        residual_values: bf16_values_to_f32(&residual_out_bytes),
        residual_device: DeviceBf16Output {
            buffer: residual_output_buffer,
            bytes: output_bytes,
            rows,
            values_per_row: output_dim,
            backend: CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
        },
        linear_backend: CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
        residual_add_backend: CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_linear_residual_add_rows_bf16_preloaded_resident_weight_padded_device_input_device_output_legacy(
    weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    bias_bf16: Option<&[u8]>,
    residual_bf16: &[u8],
    rows: usize,
    full_input_dim: usize,
    output_dim: usize,
    view: LinearPaddedDeviceInputView,
) -> Result<LinearResidualAddDeviceOutput> {
    let library = cuda_native_library()?;
    let output_bytes = rows
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "CUDA BF16 preloaded resident padded device-input linear residual-add device-output shape overflows usize",
        )?;
    let weight_buffer = preloaded_resident_weight_device_buffer_view(
        weight_name,
        view.weight.full_bytes,
        view.weight.offset_bytes,
        view.weight.view_bytes,
    )?;
    if input_buffer.device_id != weight_buffer.device_id {
        anyhow::bail!(
            "CUDA BF16 preloaded resident padded device-input linear residual-add device-output buffers are on different devices: input={} weight={}",
            input_buffer.device_id,
            weight_buffer.device_id
        );
    }
    let residual_output_buffer = OwnedCoordinatorDeviceBuffer::new(
        library,
        output_bytes,
        "BF16 preloaded resident padded device-input linear residual-add device output",
    )?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let padded_input_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        view.padded_input_bytes,
        "BF16 preloaded resident padded device-input linear residual-add device-output padded input",
    )?;
    let residual_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::B,
        residual_bf16.len(),
        "BF16 preloaded resident padded device-input linear residual-add device-output residual",
    )?;
    let linear_output_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        output_bytes,
        "BF16 preloaded resident padded device-input linear residual-add device-output delta",
    )?;
    let bias_buffer = if let Some(bias_bf16) = bias_bf16 {
        let buffer = workspace.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            bias_bf16.len(),
            "BF16 preloaded resident padded device-input linear residual-add device-output bias",
        )?;
        workspace
            .copy_h2d_to_slot(
                library,
                CoordinatorCudaScratchSlot::D,
                bias_bf16,
                "BF16 preloaded resident padded device-input linear residual-add device-output bias",
            )
            .context(
                "copying BF16 preloaded resident padded device-input linear residual-add device-output bias to device",
            )?;
        Some(buffer)
    } else {
        None
    };

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::B,
            residual_bf16,
            "BF16 preloaded resident padded device-input linear residual-add device-output residual",
        )
        .context(
            "copying BF16 preloaded resident padded device-input linear residual-add device-output residual to device",
        )?;
    library
        .cuda_zero_bytes(padded_input_buffer, view.padded_input_bytes)
        .context(
            "zeroing BF16 preloaded resident padded device-input linear residual-add device-output input",
        )?;
    for row in 0..rows {
        let src_offset = row
            .checked_mul(view.active_row_bytes)
            .context("padded device-input linear source row offset overflow usize")?;
        let dst_offset = row
            .checked_mul(view.padded_row_bytes)
            .context("padded device-input linear destination row offset overflow usize")?;
        let src = device_buffer_byte_view(
            input_buffer,
            src_offset,
            view.active_row_bytes,
            "padded device-input linear active row",
        )?;
        library
            .cuda_kv_cache_write_bytes(src, padded_input_buffer, dst_offset, view.active_row_bytes)
            .context("copying active BF16 row into padded device-input linear buffer")?;
    }
    library
        .cuda_linear_bf16_cublas(
            padded_input_buffer,
            weight_buffer,
            bias_buffer,
            linear_output_buffer,
            rows,
            full_input_dim,
            output_dim,
        )
        .context(
            "executing CUDA BF16 preloaded resident padded device-input linear residual-add device-output linear",
        )?;
    library
        .cuda_residual_add_bf16(
            residual_buffer,
            linear_output_buffer,
            residual_output_buffer.buffer,
            output_bytes / std::mem::size_of::<u16>(),
        )
        .context(
            "executing CUDA BF16 preloaded resident padded device-input linear residual-add device-output",
        )?;
    let mut linear_out_bytes = vec![0_u8; output_bytes];
    let mut residual_out_bytes = vec![0_u8; output_bytes];
    library
        .copy_d2h(&mut linear_out_bytes, linear_output_buffer)
        .context(
            "copying BF16 preloaded resident padded device-input linear residual-add device-output delta to host",
        )?;
    library
        .copy_d2h(&mut residual_out_bytes, residual_output_buffer.buffer)
        .context(
            "copying BF16 preloaded resident padded device-input linear residual-add device output to host",
        )?;

    Ok(LinearResidualAddDeviceOutput {
        linear_values: bf16_values_to_f32(&linear_out_bytes),
        residual_values: bf16_values_to_f32(&residual_out_bytes),
        residual_device: DeviceBf16Output {
            buffer: residual_output_buffer,
            bytes: output_bytes,
            rows,
            values_per_row: output_dim,
            backend: CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
        },
        linear_backend: CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
        residual_add_backend: CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
    })
}

pub(in crate::commands::real_full) fn coord_linear_graph_key_for_weight_name(
    weight_name: &str,
    rows: usize,
) -> Result<Option<CoordinatorGraphKey>> {
    if rows == 0 {
        return Ok(None);
    }
    let Some((layer_id, subpath)) = glm52_layer_tensor_subpath(weight_name) else {
        return Ok(None);
    };
    let shape = if is_glm52_attention_linear_weight_subpath(subpath) {
        if layer_id < GLM52_FIRST_K_DENSE_REPLACE {
            CoordinatorGraphShape::CoordDense
        } else {
            CoordinatorGraphShape::CoordSparseA
        }
    } else if layer_id < GLM52_FIRST_K_DENSE_REPLACE
        && is_glm52_dense_mlp_linear_weight_subpath(subpath)
    {
        CoordinatorGraphShape::CoordDense
    } else {
        return Ok(None);
    };
    shape
        .validate_layer(LayerId(layer_id as u32))
        .context("validating coordinator linear graph layer family for GLM tensor")?;
    let mode = if rows == 1 {
        LayerWaveMode::Decode
    } else {
        LayerWaveMode::Prefill
    };
    CoordinatorGraphKey::glm52_bf16(shape, mode, rows)
        .map(Some)
        .context("selecting coordinator graph slot for BF16 linear layer tensor")
}

pub(in crate::commands::real_full) fn is_glm52_dense_mlp_linear_weight_subpath(
    subpath: &str,
) -> bool {
    [
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]
    .iter()
    .any(|prefix| subpath.starts_with(prefix))
}
