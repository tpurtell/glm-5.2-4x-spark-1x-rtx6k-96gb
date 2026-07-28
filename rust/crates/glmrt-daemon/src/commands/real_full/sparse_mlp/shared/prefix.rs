use anyhow::Result;
use glmrt_core::{DType, TensorCatalog, TensorInfo};
use glmrt_loader::{load_tensor_rows, LoadedTensorRows};

use crate::commands::real_full::coordinator_kernels::{
    coordinator_cuda_reference_kernels_enabled, resident_weight_is_preloaded,
    silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight,
    silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight_device_input_device_output,
    silu_gated_mlp_rows_bf16_preloaded_gate_up_resident_weight,
    silu_gated_mlp_rows_bf16_resident_weight, DeviceBf16Output,
};
use crate::commands::real_full::dense::math::{bf16_bytes_from_f32, bf16_compact_row_prefix_bytes};

use crate::commands::real_full::sparse_mlp::math::checksum_f64;

pub(super) struct SharedExpertPrefixExecution {
    pub(super) outputs: Vec<f32>,
    pub(super) output_device: Option<DeviceBf16Output>,
    pub(super) gate_proj_bytes_read: u64,
    pub(super) up_proj_bytes_read: u64,
    pub(super) down_proj_bytes_read: u64,
    pub(super) output_checksum: f64,
    pub(super) mlp_backend: &'static str,
}

#[derive(Clone, Copy)]
struct Bf16MatrixShape {
    rows: usize,
    width: usize,
}

pub(super) fn execute_shared_expert_prefix(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden: &[f32],
    intermediate_rows: usize,
    output_rows: usize,
) -> Result<SharedExpertPrefixExecution> {
    execute_shared_expert_prefix_with_device_input(
        catalog,
        layer_id,
        hidden,
        None,
        intermediate_rows,
        output_rows,
    )
}

pub(super) fn execute_shared_expert_prefix_with_device_input(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden: &[f32],
    device_hidden: Option<&DeviceBf16Output>,
    intermediate_rows: usize,
    output_rows: usize,
) -> Result<SharedExpertPrefixExecution> {
    let gate_name = format!("model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight");
    let up_name = format!("model.layers.{layer_id}.mlp.shared_experts.up_proj.weight");
    let down_name = format!("model.layers.{layer_id}.mlp.shared_experts.down_proj.weight");
    let gate_shape =
        catalog_bf16_matrix_shape(catalog, &gate_name, "real full shared expert gate")?;
    let up_shape = catalog_bf16_matrix_shape(catalog, &up_name, "real full shared expert up")?;
    let down_shape =
        catalog_bf16_matrix_shape(catalog, &down_name, "real full shared expert down")?;
    let gate_up_full_resident = coordinator_cuda_reference_kernels_enabled()
        && output_rows == hidden.len()
        && up_shape.rows == gate_shape.rows
        && bf16_full_matrix_resident_available(&gate_name, gate_shape)
        && bf16_full_matrix_resident_available(&up_name, up_shape);
    let down_full_resident = gate_up_full_resident
        && down_shape.rows == output_rows
        && down_shape.width == gate_shape.rows
        && bf16_full_matrix_resident_available(&down_name, down_shape);
    let gate = if gate_up_full_resident {
        None
    } else {
        Some(load_tensor_rows(catalog, &gate_name, 0, intermediate_rows)?)
    };
    let up = if gate_up_full_resident {
        None
    } else {
        Some(load_tensor_rows(catalog, &up_name, 0, intermediate_rows)?)
    };
    let down = if down_full_resident {
        None
    } else {
        Some(load_tensor_rows(catalog, &down_name, 0, output_rows)?)
    };
    if gate_shape.rows < intermediate_rows || up_shape.rows < intermediate_rows {
        anyhow::bail!(
            "real full shared expert probe row count mismatch: gate={} up={} expected_prefix={intermediate_rows}",
            gate_shape.rows,
            up_shape.rows
        );
    }
    if gate_shape.width != hidden.len() || up_shape.width != hidden.len() {
        anyhow::bail!(
            "real full shared expert probe hidden width mismatch: hidden={} gate_width={} up_width={}",
            hidden.len(),
            gate_shape.width,
            up_shape.width
        );
    }
    if down_shape.rows < output_rows || down_shape.width < intermediate_rows {
        anyhow::bail!(
            "real full shared expert probe down shape mismatch: rows={} width={} expected_rows={output_rows} min_width={intermediate_rows}",
            down_shape.rows,
            down_shape.width
        );
    }
    if let Some(device_hidden) = device_hidden {
        if output_rows != hidden.len() {
            anyhow::bail!(
                "real full shared expert device-input path requires full-output rows, got output_rows={output_rows} hidden={}",
                hidden.len()
            );
        }
        if device_hidden.rows != 1 || device_hidden.values_per_row != hidden.len() {
            anyhow::bail!(
                "real full shared expert device-input shape mismatch: expected 1x{} got {}x{}",
                hidden.len(),
                device_hidden.rows,
                device_hidden.values_per_row
            );
        }
    }

    let hidden_bf16 = bf16_bytes_from_f32(hidden);
    let gate_weight_key = format!("{gate_name}[rows=0..{intermediate_rows}]");
    let up_weight_key = format!("{up_name}[rows=0..{intermediate_rows}]");
    let down_weight_key = format!("{down_name}[rows=0..{output_rows},cols=0..{intermediate_rows}]");
    let (outputs, mlp_backend, output_device) = if let (Some(device_hidden), true) =
        (device_hidden, gate_up_full_resident && down_full_resident)
    {
        let mlp_output =
                silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight_device_input_device_output(
                    &gate_name,
                    &up_name,
                    &down_name,
                    device_hidden.buffer(),
                    1,
                    hidden.len(),
                    intermediate_rows,
                    gate_shape.rows,
                    output_rows,
                )?;
        (
            mlp_output.values,
            mlp_output.backend,
            Some(mlp_output.device_output),
        )
    } else if gate_up_full_resident && down_full_resident {
        let mlp_output = silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight(
            &gate_name,
            &up_name,
            &down_name,
            &hidden_bf16,
            1,
            hidden.len(),
            intermediate_rows,
            gate_shape.rows,
            output_rows,
        )?;
        (mlp_output.values, mlp_output.backend, None)
    } else if gate_up_full_resident {
        let down = required_loaded_rows(down.as_ref(), &down_weight_key)?;
        let down_prefix_bytes = bf16_compact_row_prefix_bytes(
            &down.bytes,
            output_rows,
            down.row_width,
            intermediate_rows,
        )?;
        let mlp_output = silu_gated_mlp_rows_bf16_preloaded_gate_up_resident_weight(
            &gate_name,
            &up_name,
            &down_weight_key,
            &hidden_bf16,
            &down_prefix_bytes,
            1,
            hidden.len(),
            intermediate_rows,
            gate_shape.rows,
            output_rows,
        )?;
        (mlp_output.values, mlp_output.backend, None)
    } else {
        let gate = required_loaded_rows(gate.as_ref(), &gate_weight_key)?;
        let up = required_loaded_rows(up.as_ref(), &up_weight_key)?;
        let down = required_loaded_rows(down.as_ref(), &down_weight_key)?;
        let down_prefix_bytes = bf16_compact_row_prefix_bytes(
            &down.bytes,
            output_rows,
            down.row_width,
            intermediate_rows,
        )?;
        let mlp_output = silu_gated_mlp_rows_bf16_resident_weight(
            &gate_weight_key,
            &up_weight_key,
            &down_weight_key,
            &hidden_bf16,
            &gate.bytes,
            &up.bytes,
            &down_prefix_bytes,
            1,
            hidden.len(),
            intermediate_rows,
            output_rows,
        )?;
        (mlp_output.values, mlp_output.backend, None)
    };
    for (row_index, output) in outputs.iter().copied().enumerate() {
        if !output.is_finite() {
            anyhow::bail!(
                "real full shared expert probe produced non-finite output at row {row_index}"
            );
        }
    }

    let output_checksum = checksum_f64(&outputs);
    if !output_checksum.is_finite() {
        anyhow::bail!("real full shared expert probe produced a non-finite checksum");
    }

    Ok(SharedExpertPrefixExecution {
        outputs,
        output_device,
        gate_proj_bytes_read: gate
            .as_ref()
            .map(|rows| rows.bytes.len() as u64)
            .unwrap_or(0),
        up_proj_bytes_read: up.as_ref().map(|rows| rows.bytes.len() as u64).unwrap_or(0),
        down_proj_bytes_read: down
            .as_ref()
            .map(|rows| rows.bytes.len() as u64)
            .unwrap_or(0),
        output_checksum,
        mlp_backend,
    })
}

fn required_loaded_rows<'a>(
    rows: Option<&'a LoadedTensorRows>,
    weight_name: &str,
) -> Result<&'a LoadedTensorRows> {
    rows.ok_or_else(|| {
        anyhow::anyhow!("real full shared expert missing loaded rows for {weight_name}")
    })
}

fn catalog_bf16_matrix_shape(
    catalog: &TensorCatalog,
    name: &str,
    context: &str,
) -> Result<Bf16MatrixShape> {
    let info = catalog_tensor(catalog, name)?;
    validate_bf16_matrix_shape(info, context)
}

fn catalog_tensor<'a>(catalog: &'a TensorCatalog, name: &str) -> Result<&'a TensorInfo> {
    catalog
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| {
            anyhow::anyhow!("tensor {name} not found in real full shared expert catalog")
        })
}

fn validate_bf16_matrix_shape(info: &TensorInfo, context: &str) -> Result<Bf16MatrixShape> {
    if info.dtype != DType::Bf16 {
        anyhow::bail!(
            "{context} expected BF16 tensor {}, got {:?}",
            info.name,
            info.dtype
        );
    }
    if info.shape.len() != 2 || info.shape[0] == 0 || info.shape[1] == 0 {
        anyhow::bail!(
            "{context} expected non-empty rank-2 tensor {}, got shape {:?}",
            info.name,
            info.shape
        );
    }
    Ok(Bf16MatrixShape {
        rows: info.shape[0],
        width: info.shape[1],
    })
}

fn bf16_full_matrix_resident_available(name: &str, shape: Bf16MatrixShape) -> bool {
    let Some(bytes) = shape
        .rows
        .checked_mul(shape.width)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
    else {
        return false;
    };
    resident_weight_is_preloaded(name, bytes)
}
