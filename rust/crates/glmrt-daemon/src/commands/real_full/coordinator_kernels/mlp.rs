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

pub(in crate::commands::real_full) const CPU_REFERENCE_SILU_GATED_MLP_BACKEND: &str =
    "cpu-reference-silu-gated-mlp";
#[allow(dead_code)]
pub(in crate::commands::real_full) const CUDA_REFERENCE_SILU_GATED_MLP_BACKEND: &str =
    "cuda-reference-silu-gated-mlp-f32";
#[allow(dead_code)]
pub(in crate::commands::real_full) const CPU_REFERENCE_SILU_GATED_MLP_BF16_BACKEND: &str =
    "cpu-reference-silu-gated-mlp-bf16";
#[allow(dead_code)]
pub(in crate::commands::real_full) const CUDA_REFERENCE_SILU_GATED_MLP_BF16_BACKEND: &str =
    "cuda-reference-silu-gated-mlp-bf16";
pub(in crate::commands::real_full) const CUDA_REFERENCE_SILU_GATED_MLP_BF16_RESIDENT_WEIGHT_BACKEND:
    &str = "cuda-reference-silu-gated-mlp-bf16-resident-weight";
pub(in crate::commands::real_full) const CUDA_REFERENCE_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND:
    &str = "cuda-reference-silu-gated-mlp-bf16-preloaded-gate-up-down-resident-weight";
pub(in crate::commands::real_full) const TRITON_SILU_GATED_MLP_BF16_RESIDENT_WEIGHT_BACKEND: &str =
    "triton-silu-gated-mlp-bf16-resident-weight";
pub(in crate::commands::real_full) const TRITON_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND:
    &str = "triton-silu-gated-mlp-bf16-preloaded-gate-up-down-resident-weight";

#[allow(dead_code)]
pub(in crate::commands::real_full) fn silu_gated_mlp_rows(
    input: &[f32],
    gate_weight: &[f32],
    up_weight: &[f32],
    down_weight: &[f32],
    rows: usize,
    input_dim: usize,
    intermediate_dim: usize,
    output_dim: usize,
) -> Result<SiluGatedMlpOutput> {
    validate_silu_gated_mlp_inputs(
        input,
        gate_weight,
        up_weight,
        down_weight,
        rows,
        input_dim,
        intermediate_dim,
        output_dim,
    )?;
    if cuda_reference_kernels_enabled() && output_dim == input_dim {
        return cuda_silu_gated_mlp_rows(
            input,
            gate_weight,
            up_weight,
            down_weight,
            rows,
            input_dim,
            intermediate_dim,
        );
    }
    Ok(cpu_silu_gated_mlp_rows(
        input,
        gate_weight,
        up_weight,
        down_weight,
        rows,
        input_dim,
        intermediate_dim,
        output_dim,
    ))
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn silu_gated_mlp_rows_bf16(
    input_bf16: &[u8],
    gate_weight_bf16: &[u8],
    up_weight_bf16: &[u8],
    down_weight_bf16: &[u8],
    rows: usize,
    input_dim: usize,
    intermediate_dim: usize,
    output_dim: usize,
) -> Result<SiluGatedMlpOutput> {
    validate_silu_gated_mlp_bf16_inputs(
        input_bf16,
        gate_weight_bf16,
        up_weight_bf16,
        down_weight_bf16,
        rows,
        input_dim,
        intermediate_dim,
        output_dim,
    )?;
    if cuda_reference_kernels_enabled() && output_dim == input_dim {
        return cuda_silu_gated_mlp_rows_bf16(
            input_bf16,
            gate_weight_bf16,
            up_weight_bf16,
            down_weight_bf16,
            rows,
            input_dim,
            intermediate_dim,
        );
    }
    Ok(cpu_silu_gated_mlp_rows_bf16(
        input_bf16,
        gate_weight_bf16,
        up_weight_bf16,
        down_weight_bf16,
        rows,
        input_dim,
        intermediate_dim,
        output_dim,
    ))
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn silu_gated_mlp_rows_bf16_resident_weight(
    gate_weight_name: &str,
    up_weight_name: &str,
    down_weight_name: &str,
    input_bf16: &[u8],
    gate_weight_bf16: &[u8],
    up_weight_bf16: &[u8],
    down_weight_bf16: &[u8],
    rows: usize,
    input_dim: usize,
    intermediate_dim: usize,
    output_dim: usize,
) -> Result<SiluGatedMlpOutput> {
    validate_resident_weight_name(gate_weight_name)?;
    validate_resident_weight_name(up_weight_name)?;
    validate_resident_weight_name(down_weight_name)?;
    validate_silu_gated_mlp_bf16_inputs(
        input_bf16,
        gate_weight_bf16,
        up_weight_bf16,
        down_weight_bf16,
        rows,
        input_dim,
        intermediate_dim,
        output_dim,
    )?;
    if cuda_reference_kernels_enabled() && output_dim == input_dim {
        return cuda_silu_gated_mlp_rows_bf16_resident_weight(
            gate_weight_name,
            up_weight_name,
            down_weight_name,
            input_bf16,
            gate_weight_bf16,
            up_weight_bf16,
            down_weight_bf16,
            rows,
            input_dim,
            intermediate_dim,
        );
    }
    Ok(cpu_silu_gated_mlp_rows_bf16(
        input_bf16,
        gate_weight_bf16,
        up_weight_bf16,
        down_weight_bf16,
        rows,
        input_dim,
        intermediate_dim,
        output_dim,
    ))
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn silu_gated_mlp_rows_bf16_preloaded_gate_up_resident_weight(
    gate_weight_name: &str,
    up_weight_name: &str,
    down_weight_name: &str,
    input_bf16: &[u8],
    down_weight_bf16: &[u8],
    rows: usize,
    input_dim: usize,
    intermediate_dim: usize,
    full_intermediate_dim: usize,
    output_dim: usize,
) -> Result<SiluGatedMlpOutput> {
    validate_resident_weight_name(gate_weight_name)?;
    validate_resident_weight_name(up_weight_name)?;
    validate_resident_weight_name(down_weight_name)?;
    let gate_up_view = validate_silu_gated_mlp_bf16_preloaded_gate_up_inputs(
        input_bf16,
        down_weight_bf16,
        rows,
        input_dim,
        intermediate_dim,
        full_intermediate_dim,
        output_dim,
    )?;
    if !cuda_reference_kernels_enabled() || output_dim != input_dim {
        anyhow::bail!(
            "preloaded resident BF16 SiLU-gated MLP gate/up requires CUDA reference kernels and full-output shape"
        );
    }
    cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_resident_weight(
        gate_weight_name,
        up_weight_name,
        down_weight_name,
        input_bf16,
        down_weight_bf16,
        rows,
        input_dim,
        intermediate_dim,
        output_dim,
        gate_up_view,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight(
    gate_weight_name: &str,
    up_weight_name: &str,
    down_weight_name: &str,
    input_bf16: &[u8],
    rows: usize,
    input_dim: usize,
    intermediate_dim: usize,
    full_intermediate_dim: usize,
    output_dim: usize,
) -> Result<SiluGatedMlpOutput> {
    validate_resident_weight_name(gate_weight_name)?;
    validate_resident_weight_name(up_weight_name)?;
    validate_resident_weight_name(down_weight_name)?;
    let gate_up_down_view = validate_silu_gated_mlp_bf16_preloaded_gate_up_down_inputs(
        input_bf16,
        rows,
        input_dim,
        intermediate_dim,
        full_intermediate_dim,
        output_dim,
    )?;
    if !cuda_reference_kernels_enabled() || output_dim != input_dim {
        anyhow::bail!(
            "preloaded resident BF16 SiLU-gated MLP gate/up/down requires CUDA reference kernels and full-output shape"
        );
    }
    cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight(
        gate_weight_name,
        up_weight_name,
        down_weight_name,
        input_bf16,
        rows,
        input_dim,
        intermediate_dim,
        output_dim,
        gate_up_down_view,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight_device_input_device_output(
    gate_weight_name: &str,
    up_weight_name: &str,
    down_weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    rows: usize,
    input_dim: usize,
    intermediate_dim: usize,
    full_intermediate_dim: usize,
    output_dim: usize,
) -> Result<SiluGatedMlpDeviceOutput> {
    validate_resident_weight_name(gate_weight_name)?;
    validate_resident_weight_name(up_weight_name)?;
    validate_resident_weight_name(down_weight_name)?;
    let gate_up_down_view = validate_silu_gated_mlp_bf16_preloaded_gate_up_down_device_input(
        input_buffer,
        rows,
        input_dim,
        intermediate_dim,
        full_intermediate_dim,
        output_dim,
    )?;
    if !cuda_reference_kernels_enabled() || output_dim != input_dim {
        anyhow::bail!(
            "preloaded resident BF16 SiLU-gated MLP gate/up/down device-input device-output requires CUDA reference kernels and full-output shape"
        );
    }
    cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight_device_input_device_output(
        gate_weight_name,
        up_weight_name,
        down_weight_name,
        input_buffer,
        rows,
        input_dim,
        intermediate_dim,
        output_dim,
        gate_up_down_view,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight_device_input_device_output_only(
    gate_weight_name: &str,
    up_weight_name: &str,
    down_weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    rows: usize,
    input_dim: usize,
    intermediate_dim: usize,
    full_intermediate_dim: usize,
    output_dim: usize,
) -> Result<DeviceBf16Output> {
    validate_resident_weight_name(gate_weight_name)?;
    validate_resident_weight_name(up_weight_name)?;
    validate_resident_weight_name(down_weight_name)?;
    let gate_up_down_view = validate_silu_gated_mlp_bf16_preloaded_gate_up_down_device_input(
        input_buffer,
        rows,
        input_dim,
        intermediate_dim,
        full_intermediate_dim,
        output_dim,
    )?;
    if !cuda_reference_kernels_enabled() || output_dim != input_dim {
        anyhow::bail!(
            "preloaded resident BF16 SiLU-gated MLP gate/up/down device-output-only requires CUDA reference kernels and full-output shape"
        );
    }
    cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight_device_input_device_output_only(
        gate_weight_name,
        up_weight_name,
        down_weight_name,
        input_buffer,
        rows,
        input_dim,
        intermediate_dim,
        output_dim,
        gate_up_down_view,
    )
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn validate_silu_gated_mlp_inputs(
    input: &[f32],
    gate_weight: &[f32],
    up_weight: &[f32],
    down_weight: &[f32],
    rows: usize,
    input_dim: usize,
    intermediate_dim: usize,
    output_dim: usize,
) -> Result<()> {
    if rows == 0 || input_dim == 0 || intermediate_dim == 0 || output_dim == 0 {
        anyhow::bail!(
            "real full SiLU-gated MLP requires non-zero shape, got rows={rows} input_dim={input_dim} intermediate_dim={intermediate_dim} output_dim={output_dim}"
        );
    }
    let expected_input = rows.checked_mul(input_dim).context(
        "real full SiLU-gated MLP input shape overflows usize while validating coordinator kernel input",
    )?;
    if input.len() != expected_input {
        anyhow::bail!(
            "real full SiLU-gated MLP input length mismatch: expected {} got {}",
            expected_input,
            input.len()
        );
    }
    let expected_gate = intermediate_dim.checked_mul(input_dim).context(
        "real full SiLU-gated MLP gate/up shape overflows usize while validating coordinator kernel input",
    )?;
    if gate_weight.len() != expected_gate {
        anyhow::bail!(
            "real full SiLU-gated MLP gate weight length mismatch: expected {} got {}",
            expected_gate,
            gate_weight.len()
        );
    }
    if up_weight.len() != expected_gate {
        anyhow::bail!(
            "real full SiLU-gated MLP up weight length mismatch: expected {} got {}",
            expected_gate,
            up_weight.len()
        );
    }
    let expected_down = output_dim.checked_mul(intermediate_dim).context(
        "real full SiLU-gated MLP down shape overflows usize while validating coordinator kernel input",
    )?;
    if down_weight.len() != expected_down {
        anyhow::bail!(
            "real full SiLU-gated MLP down weight length mismatch: expected {} got {}",
            expected_down,
            down_weight.len()
        );
    }
    Ok(())
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn validate_silu_gated_mlp_bf16_inputs(
    input_bf16: &[u8],
    gate_weight_bf16: &[u8],
    up_weight_bf16: &[u8],
    down_weight_bf16: &[u8],
    rows: usize,
    input_dim: usize,
    intermediate_dim: usize,
    output_dim: usize,
) -> Result<()> {
    if rows == 0 || input_dim == 0 || intermediate_dim == 0 || output_dim == 0 {
        anyhow::bail!(
            "real full BF16 SiLU-gated MLP requires non-zero shape, got rows={rows} input_dim={input_dim} intermediate_dim={intermediate_dim} output_dim={output_dim}"
        );
    }
    let input_bytes = rows
        .checked_mul(input_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full BF16 SiLU-gated MLP input shape overflows usize while validating input",
        )?;
    if input_bf16.len() != input_bytes {
        anyhow::bail!(
            "real full BF16 SiLU-gated MLP input byte length mismatch: expected {} got {}",
            input_bytes,
            input_bf16.len()
        );
    }
    let gate_bytes = intermediate_dim
        .checked_mul(input_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full BF16 SiLU-gated MLP gate/up shape overflows usize while validating input",
        )?;
    if gate_weight_bf16.len() != gate_bytes {
        anyhow::bail!(
            "real full BF16 SiLU-gated MLP gate weight byte length mismatch: expected {} got {}",
            gate_bytes,
            gate_weight_bf16.len()
        );
    }
    if up_weight_bf16.len() != gate_bytes {
        anyhow::bail!(
            "real full BF16 SiLU-gated MLP up weight byte length mismatch: expected {} got {}",
            gate_bytes,
            up_weight_bf16.len()
        );
    }
    let down_bytes = output_dim
        .checked_mul(intermediate_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full BF16 SiLU-gated MLP down shape overflows usize while validating input",
        )?;
    if down_weight_bf16.len() != down_bytes {
        anyhow::bail!(
            "real full BF16 SiLU-gated MLP down weight byte length mismatch: expected {} got {}",
            down_bytes,
            down_weight_bf16.len()
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn validate_silu_gated_mlp_bf16_preloaded_gate_up_inputs(
    input_bf16: &[u8],
    down_weight_bf16: &[u8],
    rows: usize,
    input_dim: usize,
    intermediate_dim: usize,
    full_intermediate_dim: usize,
    output_dim: usize,
) -> Result<MlpGateUpResidentView> {
    if rows == 0
        || input_dim == 0
        || intermediate_dim == 0
        || full_intermediate_dim == 0
        || output_dim == 0
    {
        anyhow::bail!(
            "real full preloaded BF16 SiLU-gated MLP requires non-zero shape, got rows={rows} input_dim={input_dim} intermediate_dim={intermediate_dim} full_intermediate_dim={full_intermediate_dim} output_dim={output_dim}"
        );
    }
    if intermediate_dim > full_intermediate_dim {
        anyhow::bail!(
            "real full preloaded BF16 SiLU-gated MLP intermediate prefix {intermediate_dim} exceeds full intermediate {full_intermediate_dim}"
        );
    }
    let input_bytes = rows
        .checked_mul(input_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full preloaded BF16 SiLU-gated MLP input shape overflows usize while validating input",
        )?;
    if input_bf16.len() != input_bytes {
        anyhow::bail!(
            "real full preloaded BF16 SiLU-gated MLP input byte length mismatch: expected {} got {}",
            input_bytes,
            input_bf16.len()
        );
    }
    let down_bytes = output_dim
        .checked_mul(intermediate_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full preloaded BF16 SiLU-gated MLP down shape overflows usize while validating input",
        )?;
    if down_weight_bf16.len() != down_bytes {
        anyhow::bail!(
            "real full preloaded BF16 SiLU-gated MLP down weight byte length mismatch: expected {} got {}",
            down_bytes,
            down_weight_bf16.len()
        );
    }
    let row_bytes = input_dim
        .checked_mul(std::mem::size_of::<u16>())
        .context("real full preloaded BF16 SiLU-gated MLP gate/up row bytes overflow usize")?;
    let full_bytes = full_intermediate_dim
        .checked_mul(row_bytes)
        .context("real full preloaded BF16 SiLU-gated MLP gate/up full bytes overflow usize")?;
    let view_bytes = intermediate_dim.checked_mul(row_bytes).context(
        "real full preloaded BF16 SiLU-gated MLP gate/up row-prefix bytes overflow usize",
    )?;
    Ok(MlpGateUpResidentView {
        full_bytes,
        offset_bytes: 0,
        view_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn validate_silu_gated_mlp_bf16_preloaded_gate_up_down_inputs(
    input_bf16: &[u8],
    rows: usize,
    input_dim: usize,
    intermediate_dim: usize,
    full_intermediate_dim: usize,
    output_dim: usize,
) -> Result<MlpGateUpDownResidentView> {
    if rows == 0
        || input_dim == 0
        || intermediate_dim == 0
        || full_intermediate_dim == 0
        || output_dim == 0
    {
        anyhow::bail!(
            "real full preloaded BF16 SiLU-gated MLP gate/up/down requires non-zero shape, got rows={rows} input_dim={input_dim} intermediate_dim={intermediate_dim} full_intermediate_dim={full_intermediate_dim} output_dim={output_dim}"
        );
    }
    if intermediate_dim > full_intermediate_dim {
        anyhow::bail!(
            "real full preloaded BF16 SiLU-gated MLP gate/up/down intermediate prefix {intermediate_dim} exceeds full intermediate {full_intermediate_dim}"
        );
    }
    let input_bytes = rows
        .checked_mul(input_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full preloaded BF16 SiLU-gated MLP gate/up/down input shape overflows usize while validating input",
        )?;
    if input_bf16.len() != input_bytes {
        anyhow::bail!(
            "real full preloaded BF16 SiLU-gated MLP gate/up/down input byte length mismatch: expected {} got {}",
            input_bytes,
            input_bf16.len()
        );
    }
    let gate_row_bytes = input_dim.checked_mul(std::mem::size_of::<u16>()).context(
        "real full preloaded BF16 SiLU-gated MLP gate/up/down gate/up row bytes overflow usize",
    )?;
    let gate_up_full_bytes = full_intermediate_dim.checked_mul(gate_row_bytes).context(
        "real full preloaded BF16 SiLU-gated MLP gate/up/down gate/up full bytes overflow usize",
    )?;
    let gate_up_view_bytes = intermediate_dim.checked_mul(gate_row_bytes).context(
        "real full preloaded BF16 SiLU-gated MLP gate/up/down gate/up row-prefix bytes overflow usize",
    )?;
    let down_full_bytes = output_dim
        .checked_mul(full_intermediate_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full preloaded BF16 SiLU-gated MLP gate/up/down down full bytes overflow usize",
        )?;
    Ok(MlpGateUpDownResidentView {
        gate_up: MlpGateUpResidentView {
            full_bytes: gate_up_full_bytes,
            offset_bytes: 0,
            view_bytes: gate_up_view_bytes,
        },
        down_full_bytes,
        down_stride: full_intermediate_dim,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn validate_silu_gated_mlp_bf16_preloaded_gate_up_down_device_input(
    input_buffer: GlmrtDeviceBuffer,
    rows: usize,
    input_dim: usize,
    intermediate_dim: usize,
    full_intermediate_dim: usize,
    output_dim: usize,
) -> Result<MlpGateUpDownResidentView> {
    if rows == 0
        || input_dim == 0
        || intermediate_dim == 0
        || full_intermediate_dim == 0
        || output_dim == 0
    {
        anyhow::bail!(
            "real full preloaded BF16 SiLU-gated MLP gate/up/down device-input requires non-zero shape, got rows={rows} input_dim={input_dim} intermediate_dim={intermediate_dim} full_intermediate_dim={full_intermediate_dim} output_dim={output_dim}"
        );
    }
    if intermediate_dim > full_intermediate_dim {
        anyhow::bail!(
            "real full preloaded BF16 SiLU-gated MLP gate/up/down device-input intermediate prefix {intermediate_dim} exceeds full intermediate {full_intermediate_dim}"
        );
    }
    if input_buffer.ptr.is_null() {
        anyhow::bail!(
            "real full preloaded BF16 SiLU-gated MLP gate/up/down device-input buffer is null"
        );
    }
    let input_bytes = rows
        .checked_mul(input_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full preloaded BF16 SiLU-gated MLP gate/up/down device-input shape overflows usize while validating input",
        )?;
    if input_buffer.bytes < input_bytes {
        anyhow::bail!(
            "real full preloaded BF16 SiLU-gated MLP gate/up/down device-input byte length mismatch: expected at least {} got {}",
            input_bytes,
            input_buffer.bytes
        );
    }
    let gate_row_bytes = input_dim.checked_mul(std::mem::size_of::<u16>()).context(
        "real full preloaded BF16 SiLU-gated MLP gate/up/down device-input gate/up row bytes overflow usize",
    )?;
    let gate_up_full_bytes = full_intermediate_dim.checked_mul(gate_row_bytes).context(
        "real full preloaded BF16 SiLU-gated MLP gate/up/down device-input gate/up full bytes overflow usize",
    )?;
    let gate_up_view_bytes = intermediate_dim.checked_mul(gate_row_bytes).context(
        "real full preloaded BF16 SiLU-gated MLP gate/up/down device-input gate/up row-prefix bytes overflow usize",
    )?;
    let down_full_bytes = output_dim
        .checked_mul(full_intermediate_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full preloaded BF16 SiLU-gated MLP gate/up/down device-input down full bytes overflow usize",
        )?;
    Ok(MlpGateUpDownResidentView {
        gate_up: MlpGateUpResidentView {
            full_bytes: gate_up_full_bytes,
            offset_bytes: 0,
            view_bytes: gate_up_view_bytes,
        },
        down_full_bytes,
        down_stride: full_intermediate_dim,
    })
}

pub(in crate::commands::real_full) fn cpu_silu_gated_mlp_rows(
    input: &[f32],
    gate_weight: &[f32],
    up_weight: &[f32],
    down_weight: &[f32],
    rows: usize,
    input_dim: usize,
    intermediate_dim: usize,
    output_dim: usize,
) -> SiluGatedMlpOutput {
    let mut values = vec![0.0_f32; rows * output_dim];
    let mut activations = vec![0.0_f32; rows * intermediate_dim];
    for row in 0..rows {
        let input_start = row * input_dim;
        let activation_start = row * intermediate_dim;
        for mid in 0..intermediate_dim {
            let weight_start = mid * input_dim;
            let mut gate = 0.0_f32;
            let mut up = 0.0_f32;
            for col in 0..input_dim {
                let value = input[input_start + col];
                gate += value * gate_weight[weight_start + col];
                up += value * up_weight[weight_start + col];
            }
            let silu = gate / (1.0 + (-gate).exp());
            activations[activation_start + mid] = silu * up;
        }
        let output_start = row * output_dim;
        for output_index in 0..output_dim {
            let down_start = output_index * intermediate_dim;
            let mut value = 0.0_f32;
            for mid in 0..intermediate_dim {
                value += activations[activation_start + mid] * down_weight[down_start + mid];
            }
            values[output_start + output_index] = value;
        }
    }
    SiluGatedMlpOutput {
        values,
        backend: CPU_REFERENCE_SILU_GATED_MLP_BACKEND,
    }
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn cpu_silu_gated_mlp_rows_bf16(
    input_bf16: &[u8],
    gate_weight_bf16: &[u8],
    up_weight_bf16: &[u8],
    down_weight_bf16: &[u8],
    rows: usize,
    input_dim: usize,
    intermediate_dim: usize,
    output_dim: usize,
) -> SiluGatedMlpOutput {
    let mut output = cpu_silu_gated_mlp_rows(
        &bf16_values_to_f32(input_bf16),
        &bf16_values_to_f32(gate_weight_bf16),
        &bf16_values_to_f32(up_weight_bf16),
        &bf16_values_to_f32(down_weight_bf16),
        rows,
        input_dim,
        intermediate_dim,
        output_dim,
    );
    output.backend = CPU_REFERENCE_SILU_GATED_MLP_BF16_BACKEND;
    output
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_silu_gated_mlp_rows(
    input: &[f32],
    gate_weight: &[f32],
    up_weight: &[f32],
    down_weight: &[f32],
    rows: usize,
    hidden: usize,
    intermediate: usize,
) -> Result<SiluGatedMlpOutput> {
    let library = cuda_native_library()?;
    let input_bytes = std::mem::size_of_val(input);
    let gate_bytes = std::mem::size_of_val(gate_weight);
    let up_bytes = std::mem::size_of_val(up_weight);
    let down_bytes = std::mem::size_of_val(down_weight);
    let output_bytes = rows
        .checked_mul(hidden)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context("CUDA SiLU-gated MLP output shape overflows usize")?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let input_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        input_bytes,
        "SiLU-gated MLP input",
    )?;
    let gate_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::B,
        gate_bytes,
        "SiLU-gated MLP gate weight",
    )?;
    let up_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        up_bytes,
        "SiLU-gated MLP up weight",
    )?;
    let down_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        down_bytes,
        "SiLU-gated MLP down weight",
    )?;
    let output_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::E,
        output_bytes,
        "SiLU-gated MLP output",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            f32_bytes(input),
            "SiLU-gated MLP input",
        )
        .context("copying SiLU-gated MLP input to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::B,
            f32_bytes(gate_weight),
            "SiLU-gated MLP gate weight",
        )
        .context("copying SiLU-gated MLP gate weight to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::C,
            f32_bytes(up_weight),
            "SiLU-gated MLP up weight",
        )
        .context("copying SiLU-gated MLP up weight to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::D,
            f32_bytes(down_weight),
            "SiLU-gated MLP down weight",
        )
        .context("copying SiLU-gated MLP down weight to device")?;
    library
        .cuda_silu_gated_mlp_rows_f32(
            input_buffer,
            gate_buffer,
            up_buffer,
            down_buffer,
            output_buffer,
            rows,
            hidden,
            intermediate,
        )
        .context("executing CUDA SiLU-gated MLP")?;
    let mut out_bytes = vec![0_u8; output_bytes];
    library
        .copy_d2h(&mut out_bytes, output_buffer)
        .context("copying SiLU-gated MLP output to host")?;

    Ok(SiluGatedMlpOutput {
        values: f32_vec_from_bytes(&out_bytes)?,
        backend: CUDA_REFERENCE_SILU_GATED_MLP_BACKEND,
    })
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_silu_gated_mlp_rows_bf16(
    input_bf16: &[u8],
    gate_weight_bf16: &[u8],
    up_weight_bf16: &[u8],
    down_weight_bf16: &[u8],
    rows: usize,
    hidden: usize,
    intermediate: usize,
) -> Result<SiluGatedMlpOutput> {
    let library = cuda_native_library()?;
    let output_bytes = rows
        .checked_mul(hidden)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 SiLU-gated MLP output shape overflows usize")?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let input_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        input_bf16.len(),
        "BF16 SiLU-gated MLP input",
    )?;
    let gate_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::B,
        gate_weight_bf16.len(),
        "BF16 SiLU-gated MLP gate weight",
    )?;
    let up_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        up_weight_bf16.len(),
        "BF16 SiLU-gated MLP up weight",
    )?;
    let down_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        down_weight_bf16.len(),
        "BF16 SiLU-gated MLP down weight",
    )?;
    let output_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::E,
        output_bytes,
        "BF16 SiLU-gated MLP output",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            input_bf16,
            "BF16 SiLU-gated MLP input",
        )
        .context("copying BF16 SiLU-gated MLP input to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::B,
            gate_weight_bf16,
            "BF16 SiLU-gated MLP gate weight",
        )
        .context("copying BF16 SiLU-gated MLP gate weight to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::C,
            up_weight_bf16,
            "BF16 SiLU-gated MLP up weight",
        )
        .context("copying BF16 SiLU-gated MLP up weight to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::D,
            down_weight_bf16,
            "BF16 SiLU-gated MLP down weight",
        )
        .context("copying BF16 SiLU-gated MLP down weight to device")?;
    library
        .cuda_silu_gated_mlp_rows_bf16(
            input_buffer,
            gate_buffer,
            up_buffer,
            down_buffer,
            output_buffer,
            rows,
            hidden,
            intermediate,
        )
        .context("executing CUDA BF16 SiLU-gated MLP")?;
    let mut out_bytes = vec![0_u8; output_bytes];
    library
        .copy_d2h(&mut out_bytes, output_buffer)
        .context("copying BF16 SiLU-gated MLP output to host")?;

    Ok(SiluGatedMlpOutput {
        values: bf16_values_to_f32(&out_bytes),
        backend: CUDA_REFERENCE_SILU_GATED_MLP_BF16_BACKEND,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_silu_gated_mlp_rows_bf16_resident_weight(
    gate_weight_name: &str,
    up_weight_name: &str,
    down_weight_name: &str,
    input_bf16: &[u8],
    gate_weight_bf16: &[u8],
    up_weight_bf16: &[u8],
    down_weight_bf16: &[u8],
    rows: usize,
    hidden: usize,
    intermediate: usize,
) -> Result<SiluGatedMlpOutput> {
    if hidden == GLM52_HIDDEN_SIZE {
        if let Some(graph_key) = coord_dense_mlp_graph_key_for_gate_up_down_names(
            gate_weight_name,
            up_weight_name,
            down_weight_name,
            rows,
        )? {
            return cuda_silu_gated_mlp_rows_bf16_resident_weight_graph_slot(
                &graph_key,
                gate_weight_name,
                up_weight_name,
                down_weight_name,
                input_bf16,
                gate_weight_bf16,
                up_weight_bf16,
                down_weight_bf16,
                rows,
                hidden,
                intermediate,
            );
        }
    }
    cuda_silu_gated_mlp_rows_bf16_resident_weight_legacy(
        gate_weight_name,
        up_weight_name,
        down_weight_name,
        input_bf16,
        gate_weight_bf16,
        up_weight_bf16,
        down_weight_bf16,
        rows,
        hidden,
        intermediate,
    )
}

#[allow(clippy::too_many_arguments)]
fn capture_or_update_dense_mlp_bf16_graph_for_slot(
    graph_key: &CoordinatorGraphKey,
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    input_buffer: GlmrtDeviceBuffer,
    gate_buffer: GlmrtDeviceBuffer,
    up_buffer: GlmrtDeviceBuffer,
    down_buffer: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden: usize,
    intermediate: usize,
    down_stride: usize,
    native_backend: &'static str,
    triton_backend: &'static str,
    label: &'static str,
) -> Result<&'static str> {
    if triton_dense_mlp_bf16_supported(graph_key, rows, hidden, intermediate, down_stride) {
        let capture_rows = graph_key.row_bucket.row_capacity;
        let input_bytes = rows
            .checked_mul(hidden)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .with_context(|| format!("{label} Triton input bytes overflow usize"))?;
        let capture_value_bytes = capture_rows
            .checked_mul(hidden)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .with_context(|| format!("{label} Triton graph value bytes overflow usize"))?;
        let graph_input_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            capture_value_bytes,
            "Triton BF16 dense MLP graph input",
        )?;
        let graph_output_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::E,
            capture_value_bytes,
            "Triton BF16 dense MLP graph output",
        )?;
        let intermediate_bytes =
            triton_dense_mlp_intermediate_bytes(graph_key, intermediate, label)?;
        let activation_bytes = graph_key
            .row_bucket
            .row_capacity
            .checked_mul(intermediate)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .with_context(|| format!("{label} Triton BF16 activation bytes overflow usize"))?;
        let gate_output_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::B,
            intermediate_bytes,
            "Triton BF16 dense MLP gate output",
        )?;
        let up_output_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            intermediate_bytes,
            "Triton BF16 dense MLP up output",
        )?;
        let activation_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            activation_bytes,
            "Triton BF16 dense MLP activation",
        )?;
        let cuda_stream = slot.stream_ptr();
        unsafe {
            if input_buffer.ptr != graph_input_buffer.ptr {
                library
                    .copy_d2d_async(graph_input_buffer, input_buffer, input_bytes, cuda_stream)
                    .with_context(|| format!("staging {label} Triton graph input"))?;
            }
            if capture_value_bytes > input_bytes {
                let padding = device_buffer_byte_view(
                    graph_input_buffer,
                    input_bytes,
                    capture_value_bytes - input_bytes,
                    "Triton BF16 dense MLP padded input rows",
                )?;
                library
                    .cuda_zero_bytes_async(padding, padding.bytes, cuda_stream)
                    .with_context(|| format!("zeroing {label} Triton padded input rows"))?;
            }
        }
        let signature = triton_dense_mlp_graph_signature(
            capture_rows,
            hidden,
            intermediate,
            down_stride,
            graph_input_buffer,
            gate_buffer,
            up_buffer,
            down_buffer,
            graph_output_buffer,
        );
        capture_or_update_layer_triton_dense_mlp_bf16_graph(
            library,
            slot,
            signature,
            graph_input_buffer,
            gate_buffer,
            up_buffer,
            down_buffer,
            gate_output_buffer,
            up_output_buffer,
            activation_buffer,
            graph_output_buffer,
            capture_rows,
            hidden,
            intermediate,
            down_stride,
            label,
        )?;
        if output_buffer.ptr != graph_output_buffer.ptr {
            unsafe {
                library
                    .copy_d2d_async(output_buffer, graph_output_buffer, input_bytes, cuda_stream)
                    .with_context(|| format!("copying {label} Triton graph output"))?;
            }
        }
        Ok(triton_backend)
    } else {
        let signature = dense_mlp_graph_signature(graph_key, hidden, intermediate, down_stride);
        capture_or_update_layer_dense_mlp_bf16_graph(
            library,
            slot,
            signature,
            input_buffer,
            gate_buffer,
            up_buffer,
            down_buffer,
            output_buffer,
            rows,
            hidden,
            intermediate,
            down_stride,
            label,
        )?;
        Ok(native_backend)
    }
}

fn triton_dense_mlp_bf16_supported(
    graph_key: &CoordinatorGraphKey,
    rows: usize,
    hidden: usize,
    intermediate: usize,
    down_stride: usize,
) -> bool {
    coordinator_python_capture_enabled()
        && matches!(
            graph_key.shape,
            CoordinatorGraphShape::CoordDense | CoordinatorGraphShape::CoordSparseA
        )
        && rows > 0
        && rows <= graph_key.row_bucket.row_capacity
        && hidden == GLM52_HIDDEN_SIZE
        && intermediate > 0
        && down_stride >= intermediate
}

fn triton_dense_mlp_intermediate_bytes(
    graph_key: &CoordinatorGraphKey,
    intermediate: usize,
    label: &str,
) -> Result<usize> {
    graph_key
        .row_bucket
        .row_capacity
        .checked_mul(intermediate)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .with_context(|| format!("{label} Triton intermediate buffer bytes overflow usize"))
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_silu_gated_mlp_rows_bf16_resident_weight_graph_slot(
    graph_key: &CoordinatorGraphKey,
    gate_weight_name: &str,
    up_weight_name: &str,
    down_weight_name: &str,
    input_bf16: &[u8],
    gate_weight_bf16: &[u8],
    up_weight_bf16: &[u8],
    down_weight_bf16: &[u8],
    rows: usize,
    hidden: usize,
    intermediate: usize,
) -> Result<SiluGatedMlpOutput> {
    let output_bytes = rows
        .checked_mul(hidden)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 resident graph-slot SiLU-gated MLP output shape overflows usize")?;
    let graph_input_bytes = dense_mlp_graph_value_bytes(
        graph_key,
        hidden,
        "CUDA BF16 resident graph-slot SiLU-gated MLP",
    )?;
    let graph_output_bytes = dense_mlp_graph_value_bytes(
        graph_key,
        hidden,
        "CUDA BF16 resident graph-slot SiLU-gated MLP",
    )?;
    let gate_buffer = resident_weight_buffer_from_registry(
        gate_weight_name,
        gate_weight_bf16,
        "BF16 resident SiLU-gated MLP gate weight",
    )?;
    let up_buffer = resident_weight_buffer_from_registry(
        up_weight_name,
        up_weight_bf16,
        "BF16 resident SiLU-gated MLP up weight",
    )?;
    let down_buffer = resident_weight_buffer_from_registry(
        down_weight_name,
        down_weight_bf16,
        "BF16 resident SiLU-gated MLP down weight",
    )?;

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let input_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            graph_input_bytes,
            "BF16 resident SiLU-gated MLP input",
        )?;
        let output_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::E,
            graph_output_bytes,
            "BF16 resident SiLU-gated MLP output",
        )?;

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::A,
                input_bf16,
                "BF16 resident SiLU-gated MLP input",
                cuda_stream,
            )
            .context("async copying BF16 resident SiLU-gated MLP input to device")?;
        let backend = capture_or_update_dense_mlp_bf16_graph_for_slot(
            graph_key,
            library,
            slot,
            input_buffer,
            gate_buffer,
            up_buffer,
            down_buffer,
            output_buffer,
            rows,
            hidden,
            intermediate,
            intermediate,
            CUDA_REFERENCE_SILU_GATED_MLP_BF16_RESIDENT_WEIGHT_BACKEND,
            TRITON_SILU_GATED_MLP_BF16_RESIDENT_WEIGHT_BACKEND,
            "BF16 resident SiLU-gated MLP",
        )?;
        let mut out_bytes = vec![0_u8; output_bytes];
        unsafe {
            library
                .copy_d2h_async(&mut out_bytes, output_buffer, cuda_stream)
                .context("async copying BF16 resident SiLU-gated MLP output to host")?;
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing BF16 resident SiLU-gated MLP graph slot stream")?;
        }

        Ok(SiluGatedMlpOutput {
            values: bf16_values_to_f32(&out_bytes),
            backend,
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_silu_gated_mlp_rows_bf16_resident_weight_legacy(
    gate_weight_name: &str,
    up_weight_name: &str,
    down_weight_name: &str,
    input_bf16: &[u8],
    gate_weight_bf16: &[u8],
    up_weight_bf16: &[u8],
    down_weight_bf16: &[u8],
    rows: usize,
    hidden: usize,
    intermediate: usize,
) -> Result<SiluGatedMlpOutput> {
    let library = cuda_native_library()?;
    let output_bytes = rows
        .checked_mul(hidden)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 resident SiLU-gated MLP output shape overflows usize")?;
    let gate_buffer = resident_weight_buffer_from_registry(
        gate_weight_name,
        gate_weight_bf16,
        "BF16 resident SiLU-gated MLP gate weight",
    )?;
    let up_buffer = resident_weight_buffer_from_registry(
        up_weight_name,
        up_weight_bf16,
        "BF16 resident SiLU-gated MLP up weight",
    )?;
    let down_buffer = resident_weight_buffer_from_registry(
        down_weight_name,
        down_weight_bf16,
        "BF16 resident SiLU-gated MLP down weight",
    )?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let input_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        input_bf16.len(),
        "BF16 resident SiLU-gated MLP input",
    )?;
    let output_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::E,
        output_bytes,
        "BF16 resident SiLU-gated MLP output",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            input_bf16,
            "BF16 resident SiLU-gated MLP input",
        )
        .context("copying BF16 resident SiLU-gated MLP input to device")?;
    library
        .cuda_silu_gated_mlp_rows_bf16(
            input_buffer,
            gate_buffer,
            up_buffer,
            down_buffer,
            output_buffer,
            rows,
            hidden,
            intermediate,
        )
        .context("executing CUDA BF16 resident SiLU-gated MLP")?;
    let mut out_bytes = vec![0_u8; output_bytes];
    library
        .copy_d2h(&mut out_bytes, output_buffer)
        .context("copying BF16 resident SiLU-gated MLP output to host")?;

    Ok(SiluGatedMlpOutput {
        values: bf16_values_to_f32(&out_bytes),
        backend: CUDA_REFERENCE_SILU_GATED_MLP_BF16_RESIDENT_WEIGHT_BACKEND,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_resident_weight(
    gate_weight_name: &str,
    up_weight_name: &str,
    down_weight_name: &str,
    input_bf16: &[u8],
    down_weight_bf16: &[u8],
    rows: usize,
    hidden: usize,
    intermediate: usize,
    output_dim: usize,
    gate_up_view: MlpGateUpResidentView,
) -> Result<SiluGatedMlpOutput> {
    if hidden == GLM52_HIDDEN_SIZE && output_dim == GLM52_HIDDEN_SIZE {
        if let Some(graph_key) = coord_dense_mlp_graph_key_for_gate_up_down_names(
            gate_weight_name,
            up_weight_name,
            down_weight_name,
            rows,
        )? {
            return cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_resident_weight_graph_slot(
                &graph_key,
                gate_weight_name,
                up_weight_name,
                down_weight_name,
                input_bf16,
                down_weight_bf16,
                rows,
                hidden,
                intermediate,
                output_dim,
                gate_up_view,
            );
        }
    }
    cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_resident_weight_legacy(
        gate_weight_name,
        up_weight_name,
        down_weight_name,
        input_bf16,
        down_weight_bf16,
        rows,
        hidden,
        intermediate,
        output_dim,
        gate_up_view,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_resident_weight_graph_slot(
    graph_key: &CoordinatorGraphKey,
    gate_weight_name: &str,
    up_weight_name: &str,
    down_weight_name: &str,
    input_bf16: &[u8],
    down_weight_bf16: &[u8],
    rows: usize,
    hidden: usize,
    intermediate: usize,
    output_dim: usize,
    gate_up_view: MlpGateUpResidentView,
) -> Result<SiluGatedMlpOutput> {
    let output_bytes = rows
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "CUDA BF16 preloaded gate/up resident graph-slot SiLU-gated MLP output shape overflows usize",
        )?;
    let graph_input_bytes = dense_mlp_graph_value_bytes(
        graph_key,
        hidden,
        "CUDA BF16 preloaded gate/up resident graph-slot SiLU-gated MLP",
    )?;
    let graph_output_bytes = dense_mlp_graph_value_bytes(
        graph_key,
        output_dim,
        "CUDA BF16 preloaded gate/up resident graph-slot SiLU-gated MLP",
    )?;
    let gate_buffer = preloaded_resident_weight_device_buffer_view(
        gate_weight_name,
        gate_up_view.full_bytes,
        gate_up_view.offset_bytes,
        gate_up_view.view_bytes,
    )?;
    let up_buffer = preloaded_resident_weight_device_buffer_view(
        up_weight_name,
        gate_up_view.full_bytes,
        gate_up_view.offset_bytes,
        gate_up_view.view_bytes,
    )?;
    let down_buffer = resident_weight_buffer_from_registry(
        down_weight_name,
        down_weight_bf16,
        "BF16 preloaded gate/up resident SiLU-gated MLP compact down weight",
    )?;

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let input_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            graph_input_bytes,
            "BF16 preloaded gate/up resident SiLU-gated MLP input",
        )?;
        let output_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::E,
            graph_output_bytes,
            "BF16 preloaded gate/up resident SiLU-gated MLP output",
        )?;

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::A,
                input_bf16,
                "BF16 preloaded gate/up resident SiLU-gated MLP input",
                cuda_stream,
            )
            .context(
                "async copying BF16 preloaded gate/up resident SiLU-gated MLP input to device",
            )?;
        let backend = capture_or_update_dense_mlp_bf16_graph_for_slot(
            graph_key,
            library,
            slot,
            input_buffer,
            gate_buffer,
            up_buffer,
            down_buffer,
            output_buffer,
            rows,
            hidden,
            intermediate,
            intermediate,
            CUDA_REFERENCE_SILU_GATED_MLP_BF16_RESIDENT_WEIGHT_BACKEND,
            TRITON_SILU_GATED_MLP_BF16_RESIDENT_WEIGHT_BACKEND,
            "BF16 preloaded gate/up resident SiLU-gated MLP",
        )?;
        let mut out_bytes = vec![0_u8; output_bytes];
        unsafe {
            library
                .copy_d2h_async(&mut out_bytes, output_buffer, cuda_stream)
                .context(
                    "async copying BF16 preloaded gate/up resident SiLU-gated MLP output to host",
                )?;
            library.cuda_stream_synchronize(cuda_stream).context(
                "synchronizing BF16 preloaded gate/up resident SiLU-gated MLP graph slot stream",
            )?;
        }

        Ok(SiluGatedMlpOutput {
            values: bf16_values_to_f32(&out_bytes),
            backend,
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_resident_weight_legacy(
    gate_weight_name: &str,
    up_weight_name: &str,
    down_weight_name: &str,
    input_bf16: &[u8],
    down_weight_bf16: &[u8],
    rows: usize,
    hidden: usize,
    intermediate: usize,
    output_dim: usize,
    gate_up_view: MlpGateUpResidentView,
) -> Result<SiluGatedMlpOutput> {
    let library = cuda_native_library()?;
    let output_bytes = rows
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 preloaded resident SiLU-gated MLP output shape overflows usize")?;
    let gate_buffer = preloaded_resident_weight_device_buffer_view(
        gate_weight_name,
        gate_up_view.full_bytes,
        gate_up_view.offset_bytes,
        gate_up_view.view_bytes,
    )?;
    let up_buffer = preloaded_resident_weight_device_buffer_view(
        up_weight_name,
        gate_up_view.full_bytes,
        gate_up_view.offset_bytes,
        gate_up_view.view_bytes,
    )?;
    let down_buffer = resident_weight_buffer_from_registry(
        down_weight_name,
        down_weight_bf16,
        "BF16 preloaded resident SiLU-gated MLP compact down weight",
    )?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let input_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        input_bf16.len(),
        "BF16 preloaded resident SiLU-gated MLP input",
    )?;
    let output_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::E,
        output_bytes,
        "BF16 preloaded resident SiLU-gated MLP output",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            input_bf16,
            "BF16 preloaded resident SiLU-gated MLP input",
        )
        .context("copying BF16 preloaded resident SiLU-gated MLP input to device")?;
    library
        .cuda_silu_gated_mlp_rows_bf16(
            input_buffer,
            gate_buffer,
            up_buffer,
            down_buffer,
            output_buffer,
            rows,
            hidden,
            intermediate,
        )
        .context("executing CUDA BF16 preloaded resident SiLU-gated MLP")?;
    let mut out_bytes = vec![0_u8; output_bytes];
    library
        .copy_d2h(&mut out_bytes, output_buffer)
        .context("copying BF16 preloaded resident SiLU-gated MLP output to host")?;

    Ok(SiluGatedMlpOutput {
        values: bf16_values_to_f32(&out_bytes),
        backend: CUDA_REFERENCE_SILU_GATED_MLP_BF16_RESIDENT_WEIGHT_BACKEND,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight(
    gate_weight_name: &str,
    up_weight_name: &str,
    down_weight_name: &str,
    input_bf16: &[u8],
    rows: usize,
    hidden: usize,
    intermediate: usize,
    output_dim: usize,
    view: MlpGateUpDownResidentView,
) -> Result<SiluGatedMlpOutput> {
    if hidden == GLM52_HIDDEN_SIZE && output_dim == GLM52_HIDDEN_SIZE {
        if let Some(graph_key) = coord_dense_mlp_graph_key_for_gate_up_down_names(
            gate_weight_name,
            up_weight_name,
            down_weight_name,
            rows,
        )? {
            return cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight_graph_slot(
                &graph_key,
                gate_weight_name,
                up_weight_name,
                down_weight_name,
                input_bf16,
                rows,
                hidden,
                intermediate,
                output_dim,
                view,
            );
        }
    }
    cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight_legacy(
        gate_weight_name,
        up_weight_name,
        down_weight_name,
        input_bf16,
        rows,
        hidden,
        intermediate,
        output_dim,
        view,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight_device_input_device_output(
    gate_weight_name: &str,
    up_weight_name: &str,
    down_weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden: usize,
    intermediate: usize,
    output_dim: usize,
    view: MlpGateUpDownResidentView,
) -> Result<SiluGatedMlpDeviceOutput> {
    if hidden == GLM52_HIDDEN_SIZE && output_dim == GLM52_HIDDEN_SIZE {
        if let Some(graph_key) = coord_dense_mlp_graph_key_for_gate_up_down_names(
            gate_weight_name,
            up_weight_name,
            down_weight_name,
            rows,
        )? {
            return cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight_device_input_device_output_graph_slot(
                &graph_key,
                gate_weight_name,
                up_weight_name,
                down_weight_name,
                input_buffer,
                rows,
                hidden,
                intermediate,
                output_dim,
                view,
            );
        }
    }
    cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight_device_input_device_output_legacy(
        gate_weight_name,
        up_weight_name,
        down_weight_name,
        input_buffer,
        rows,
        hidden,
        intermediate,
        output_dim,
        view,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight_device_input_device_output_only(
    gate_weight_name: &str,
    up_weight_name: &str,
    down_weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden: usize,
    intermediate: usize,
    output_dim: usize,
    view: MlpGateUpDownResidentView,
) -> Result<DeviceBf16Output> {
    if hidden == GLM52_HIDDEN_SIZE
        && output_dim == GLM52_HIDDEN_SIZE
        && coordinator_python_capture_enabled()
    {
        if let Some(graph_key) = coord_dense_mlp_graph_key_for_gate_up_down_names(
            gate_weight_name,
            up_weight_name,
            down_weight_name,
            rows,
        )? {
            return cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight_device_input_device_output_only_graph_slot(
                &graph_key,
                gate_weight_name,
                up_weight_name,
                down_weight_name,
                input_buffer,
                rows,
                hidden,
                intermediate,
                output_dim,
                view,
            );
        }
    }
    cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight_device_input_device_output_only_legacy(
        gate_weight_name,
        up_weight_name,
        down_weight_name,
        input_buffer,
        rows,
        hidden,
        intermediate,
        output_dim,
        view,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight_device_input_device_output_only_legacy(
    gate_weight_name: &str,
    up_weight_name: &str,
    down_weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden: usize,
    intermediate: usize,
    output_dim: usize,
    view: MlpGateUpDownResidentView,
) -> Result<DeviceBf16Output> {
    let library = cuda_native_library()?;
    let output_bytes = rows
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "CUDA BF16 preloaded resident gate/up/down staged MLP output shape overflows usize",
        )?;
    let activation_bytes = rows
        .checked_mul(intermediate)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context(
            "CUDA BF16 preloaded resident gate/up/down staged MLP activation shape overflows usize",
        )?;
    let gate_buffer = preloaded_resident_weight_device_buffer_view(
        gate_weight_name,
        view.gate_up.full_bytes,
        view.gate_up.offset_bytes,
        view.gate_up.view_bytes,
    )?;
    let up_buffer = preloaded_resident_weight_device_buffer_view(
        up_weight_name,
        view.gate_up.full_bytes,
        view.gate_up.offset_bytes,
        view.gate_up.view_bytes,
    )?;
    let down_buffer =
        preloaded_resident_weight_device_buffer(down_weight_name, view.down_full_bytes)?;
    if input_buffer.device_id != gate_buffer.device_id
        || input_buffer.device_id != up_buffer.device_id
        || input_buffer.device_id != down_buffer.device_id
    {
        anyhow::bail!(
            "CUDA BF16 staged preloaded resident gate/up/down device-input MLP buffers are on different devices: input={} gate={} up={} down={}",
            input_buffer.device_id,
            gate_buffer.device_id,
            up_buffer.device_id,
            down_buffer.device_id
        );
    }
    let activation_buffer = OwnedCoordinatorDeviceBuffer::new(
        library,
        activation_bytes,
        "BF16 preloaded resident gate/up/down SiLU-gated MLP staged activation",
    )?;
    let output_buffer = OwnedCoordinatorDeviceBuffer::new(
        library,
        output_bytes,
        "BF16 preloaded resident gate/up/down SiLU-gated MLP staged device output",
    )?;
    library
        .cuda_silu_gated_mlp_rows_bf16_down_stride_staged(
            input_buffer,
            gate_buffer,
            up_buffer,
            down_buffer,
            activation_buffer.buffer,
            output_buffer.buffer,
            rows,
            hidden,
            intermediate,
            view.down_stride,
        )
        .context(
            "executing staged CUDA BF16 preloaded resident gate/up/down device-input SiLU-gated MLP",
        )?;

    Ok(DeviceBf16Output {
        buffer: output_buffer,
        bytes: output_bytes,
        rows,
        values_per_row: output_dim,
        backend: CUDA_REFERENCE_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight_device_input_device_output_only_graph_slot(
    graph_key: &CoordinatorGraphKey,
    gate_weight_name: &str,
    up_weight_name: &str,
    down_weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden: usize,
    intermediate: usize,
    output_dim: usize,
    view: MlpGateUpDownResidentView,
) -> Result<DeviceBf16Output> {
    let output_bytes = rows
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "CUDA BF16 preloaded resident gate/up/down graph-slot SiLU-gated MLP output-only shape overflows usize",
        )?;
    let gate_buffer = preloaded_resident_weight_device_buffer_view(
        gate_weight_name,
        view.gate_up.full_bytes,
        view.gate_up.offset_bytes,
        view.gate_up.view_bytes,
    )?;
    let up_buffer = preloaded_resident_weight_device_buffer_view(
        up_weight_name,
        view.gate_up.full_bytes,
        view.gate_up.offset_bytes,
        view.gate_up.view_bytes,
    )?;
    let down_buffer =
        preloaded_resident_weight_device_buffer(down_weight_name, view.down_full_bytes)?;
    if input_buffer.device_id != gate_buffer.device_id
        || input_buffer.device_id != up_buffer.device_id
        || input_buffer.device_id != down_buffer.device_id
    {
        anyhow::bail!(
            "CUDA BF16 preloaded resident gate/up/down output-only MLP buffers are on different devices: input={} gate={} up={} down={}",
            input_buffer.device_id,
            gate_buffer.device_id,
            up_buffer.device_id,
            down_buffer.device_id
        );
    }

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let output_buffer = OwnedCoordinatorDeviceBuffer::new(
            library,
            output_bytes,
            "BF16 preloaded resident gate/up/down SiLU-gated MLP output-only device output",
        )?;

        let backend = capture_or_update_dense_mlp_bf16_graph_for_slot(
            graph_key,
            library,
            slot,
            input_buffer,
            gate_buffer,
            up_buffer,
            down_buffer,
            output_buffer.buffer,
            rows,
            hidden,
            intermediate,
            view.down_stride,
            CUDA_REFERENCE_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND,
            TRITON_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND,
            "BF16 preloaded resident gate/up/down output-only SiLU-gated MLP",
        )?;
        unsafe {
            library
                .cuda_stream_synchronize(cuda_stream)
                .context(
                    "synchronizing BF16 preloaded resident gate/up/down output-only SiLU-gated MLP graph slot stream",
                )?;
        }

        Ok(DeviceBf16Output {
            buffer: output_buffer,
            bytes: output_bytes,
            rows,
            values_per_row: output_dim,
            backend,
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight_graph_slot(
    graph_key: &CoordinatorGraphKey,
    gate_weight_name: &str,
    up_weight_name: &str,
    down_weight_name: &str,
    input_bf16: &[u8],
    rows: usize,
    hidden: usize,
    intermediate: usize,
    output_dim: usize,
    view: MlpGateUpDownResidentView,
) -> Result<SiluGatedMlpOutput> {
    let output_bytes = rows
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "CUDA BF16 preloaded resident gate/up/down graph-slot SiLU-gated MLP output shape overflows usize",
        )?;
    let graph_input_bytes = dense_mlp_graph_value_bytes(
        graph_key,
        hidden,
        "CUDA BF16 preloaded resident gate/up/down graph-slot SiLU-gated MLP",
    )?;
    let graph_output_bytes = dense_mlp_graph_value_bytes(
        graph_key,
        output_dim,
        "CUDA BF16 preloaded resident gate/up/down graph-slot SiLU-gated MLP",
    )?;
    let gate_buffer = preloaded_resident_weight_device_buffer_view(
        gate_weight_name,
        view.gate_up.full_bytes,
        view.gate_up.offset_bytes,
        view.gate_up.view_bytes,
    )?;
    let up_buffer = preloaded_resident_weight_device_buffer_view(
        up_weight_name,
        view.gate_up.full_bytes,
        view.gate_up.offset_bytes,
        view.gate_up.view_bytes,
    )?;
    let down_buffer =
        preloaded_resident_weight_device_buffer(down_weight_name, view.down_full_bytes)?;

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let input_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            graph_input_bytes,
            "BF16 preloaded resident gate/up/down SiLU-gated MLP input",
        )?;
        let output_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::E,
            graph_output_bytes,
            "BF16 preloaded resident gate/up/down SiLU-gated MLP output",
        )?;

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::A,
                input_bf16,
                "BF16 preloaded resident gate/up/down SiLU-gated MLP input",
                cuda_stream,
            )
            .context(
                "async copying BF16 preloaded resident gate/up/down SiLU-gated MLP input to device",
            )?;
        let backend = capture_or_update_dense_mlp_bf16_graph_for_slot(
            graph_key,
            library,
            slot,
            input_buffer,
            gate_buffer,
            up_buffer,
            down_buffer,
            output_buffer,
            rows,
            hidden,
            intermediate,
            view.down_stride,
            CUDA_REFERENCE_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND,
            TRITON_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND,
            "BF16 preloaded resident gate/up/down SiLU-gated MLP",
        )?;
        let mut out_bytes = vec![0_u8; output_bytes];
        unsafe {
            library
                .copy_d2h_async(&mut out_bytes, output_buffer, cuda_stream)
                .context(
                    "async copying BF16 preloaded resident gate/up/down SiLU-gated MLP output to host",
                )?;
            library
                .cuda_stream_synchronize(cuda_stream)
                .context(
                    "synchronizing BF16 preloaded resident gate/up/down SiLU-gated MLP graph slot stream",
                )?;
        }

        Ok(SiluGatedMlpOutput {
            values: bf16_values_to_f32(&out_bytes),
            backend,
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight_device_input_device_output_graph_slot(
    graph_key: &CoordinatorGraphKey,
    gate_weight_name: &str,
    up_weight_name: &str,
    down_weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden: usize,
    intermediate: usize,
    output_dim: usize,
    view: MlpGateUpDownResidentView,
) -> Result<SiluGatedMlpDeviceOutput> {
    let output_bytes = rows
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "CUDA BF16 preloaded resident gate/up/down graph-slot SiLU-gated MLP device-output shape overflows usize",
        )?;
    let gate_buffer = preloaded_resident_weight_device_buffer_view(
        gate_weight_name,
        view.gate_up.full_bytes,
        view.gate_up.offset_bytes,
        view.gate_up.view_bytes,
    )?;
    let up_buffer = preloaded_resident_weight_device_buffer_view(
        up_weight_name,
        view.gate_up.full_bytes,
        view.gate_up.offset_bytes,
        view.gate_up.view_bytes,
    )?;
    let down_buffer =
        preloaded_resident_weight_device_buffer(down_weight_name, view.down_full_bytes)?;
    if input_buffer.device_id != gate_buffer.device_id
        || input_buffer.device_id != up_buffer.device_id
        || input_buffer.device_id != down_buffer.device_id
    {
        anyhow::bail!(
            "CUDA BF16 preloaded resident gate/up/down device-input MLP buffers are on different devices: input={} gate={} up={} down={}",
            input_buffer.device_id,
            gate_buffer.device_id,
            up_buffer.device_id,
            down_buffer.device_id
        );
    }

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let output_buffer = OwnedCoordinatorDeviceBuffer::new(
            library,
            output_bytes,
            "BF16 preloaded resident gate/up/down SiLU-gated MLP device output",
        )?;

        let backend = capture_or_update_dense_mlp_bf16_graph_for_slot(
            graph_key,
            library,
            slot,
            input_buffer,
            gate_buffer,
            up_buffer,
            down_buffer,
            output_buffer.buffer,
            rows,
            hidden,
            intermediate,
            view.down_stride,
            CUDA_REFERENCE_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND,
            TRITON_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND,
            "BF16 preloaded resident gate/up/down device-input SiLU-gated MLP",
        )?;
        let mut out_bytes = vec![0_u8; output_bytes];
        unsafe {
            library
                .copy_d2h_async(&mut out_bytes, output_buffer.buffer, cuda_stream)
                .context(
                    "async copying BF16 preloaded resident gate/up/down device-input SiLU-gated MLP output to host",
                )?;
            library
                .cuda_stream_synchronize(cuda_stream)
                .context(
                    "synchronizing BF16 preloaded resident gate/up/down device-input SiLU-gated MLP graph slot stream",
                )?;
        }

        Ok(SiluGatedMlpDeviceOutput {
            values: bf16_values_to_f32(&out_bytes),
            device_output: DeviceBf16Output {
                buffer: output_buffer,
                bytes: output_bytes,
                rows,
                values_per_row: output_dim,
                backend,
            },
            backend,
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight_legacy(
    gate_weight_name: &str,
    up_weight_name: &str,
    down_weight_name: &str,
    input_bf16: &[u8],
    rows: usize,
    hidden: usize,
    intermediate: usize,
    output_dim: usize,
    view: MlpGateUpDownResidentView,
) -> Result<SiluGatedMlpOutput> {
    let library = cuda_native_library()?;
    let output_bytes = rows
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "CUDA BF16 preloaded resident gate/up/down SiLU-gated MLP output shape overflows usize",
        )?;
    let gate_buffer = preloaded_resident_weight_device_buffer_view(
        gate_weight_name,
        view.gate_up.full_bytes,
        view.gate_up.offset_bytes,
        view.gate_up.view_bytes,
    )?;
    let up_buffer = preloaded_resident_weight_device_buffer_view(
        up_weight_name,
        view.gate_up.full_bytes,
        view.gate_up.offset_bytes,
        view.gate_up.view_bytes,
    )?;
    let down_buffer =
        preloaded_resident_weight_device_buffer(down_weight_name, view.down_full_bytes)?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let input_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        input_bf16.len(),
        "BF16 preloaded resident gate/up/down SiLU-gated MLP input",
    )?;
    let output_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::E,
        output_bytes,
        "BF16 preloaded resident gate/up/down SiLU-gated MLP output",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            input_bf16,
            "BF16 preloaded resident gate/up/down SiLU-gated MLP input",
        )
        .context("copying BF16 preloaded resident gate/up/down SiLU-gated MLP input to device")?;
    library
        .cuda_silu_gated_mlp_rows_bf16_down_stride(
            input_buffer,
            gate_buffer,
            up_buffer,
            down_buffer,
            output_buffer,
            rows,
            hidden,
            intermediate,
            view.down_stride,
        )
        .context("executing CUDA BF16 preloaded resident gate/up/down SiLU-gated MLP")?;
    let mut out_bytes = vec![0_u8; output_bytes];
    library
        .copy_d2h(&mut out_bytes, output_buffer)
        .context("copying BF16 preloaded resident gate/up/down SiLU-gated MLP output to host")?;

    Ok(SiluGatedMlpOutput {
        values: bf16_values_to_f32(&out_bytes),
        backend: CUDA_REFERENCE_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight_device_input_device_output_legacy(
    gate_weight_name: &str,
    up_weight_name: &str,
    down_weight_name: &str,
    input_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden: usize,
    intermediate: usize,
    output_dim: usize,
    view: MlpGateUpDownResidentView,
) -> Result<SiluGatedMlpDeviceOutput> {
    let library = cuda_native_library()?;
    let output_bytes = rows
        .checked_mul(output_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "CUDA BF16 preloaded resident gate/up/down SiLU-gated MLP device-output shape overflows usize",
        )?;
    let gate_buffer = preloaded_resident_weight_device_buffer_view(
        gate_weight_name,
        view.gate_up.full_bytes,
        view.gate_up.offset_bytes,
        view.gate_up.view_bytes,
    )?;
    let up_buffer = preloaded_resident_weight_device_buffer_view(
        up_weight_name,
        view.gate_up.full_bytes,
        view.gate_up.offset_bytes,
        view.gate_up.view_bytes,
    )?;
    let down_buffer =
        preloaded_resident_weight_device_buffer(down_weight_name, view.down_full_bytes)?;
    if input_buffer.device_id != gate_buffer.device_id
        || input_buffer.device_id != up_buffer.device_id
        || input_buffer.device_id != down_buffer.device_id
    {
        anyhow::bail!(
            "CUDA BF16 preloaded resident gate/up/down device-input MLP buffers are on different devices: input={} gate={} up={} down={}",
            input_buffer.device_id,
            gate_buffer.device_id,
            up_buffer.device_id,
            down_buffer.device_id
        );
    }
    let output_buffer = OwnedCoordinatorDeviceBuffer::new(
        library,
        output_bytes,
        "BF16 preloaded resident gate/up/down SiLU-gated MLP device output",
    )?;

    library
        .cuda_silu_gated_mlp_rows_bf16_down_stride(
            input_buffer,
            gate_buffer,
            up_buffer,
            down_buffer,
            output_buffer.buffer,
            rows,
            hidden,
            intermediate,
            view.down_stride,
        )
        .context(
            "executing CUDA BF16 preloaded resident gate/up/down device-input SiLU-gated MLP",
        )?;
    let mut out_bytes = vec![0_u8; output_bytes];
    library
        .copy_d2h(&mut out_bytes, output_buffer.buffer)
        .context(
        "copying BF16 preloaded resident gate/up/down device-input SiLU-gated MLP output to host",
    )?;

    Ok(SiluGatedMlpDeviceOutput {
        values: bf16_values_to_f32(&out_bytes),
        device_output: DeviceBf16Output {
            buffer: output_buffer,
            bytes: output_bytes,
            rows,
            values_per_row: output_dim,
            backend:
                CUDA_REFERENCE_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND,
        },
        backend: CUDA_REFERENCE_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND,
    })
}
