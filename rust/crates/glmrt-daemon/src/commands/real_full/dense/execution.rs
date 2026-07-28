use anyhow::{Context, Result};
use glmrt_core::{
    DType, TensorCatalog, TensorInfo, GLM52_FIRST_K_DENSE_REPLACE, GLM52_HIDDEN_SIZE,
};
use glmrt_loader::{
    load_tensor_bytes, load_tensor_rows, read_tensor_bytes_into, read_tensor_row_prefix_into,
    read_tensor_rows_into, LoadedTensorRows,
};

use super::math::{
    bf16_bytes_from_f32, bf16_bytes_to_f32, bf16_compact_row_prefix_bytes, checksum_f64,
    deterministic_dense_hidden, fill_bf16_bytes_from_f32, silu,
};
use super::{
    REAL_FULL_DENSE_PREFIX_INTERMEDIATE_ROWS, REAL_FULL_DENSE_PREFIX_OUTPUT_ROWS,
    REAL_FULL_DENSE_RMSNORM_EPS,
};
use crate::commands::real_full::coordinator_kernels::{
    coordinator_cuda_reference_kernels_enabled, linear_rows_bf16_preloaded_resident_weight,
    linear_rows_bf16_resident_weight, preload_resident_weight_from_host_staging,
    resident_weight_is_preloaded, residual_add_bf16_device_inputs_output,
    residual_add_prefix_bf16_bytes_into, rmsnorm_hidden_bf16_preloaded_resident_weight,
    rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output,
    rmsnorm_hidden_bf16_resident_weight,
    silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight,
    silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight_device_input_device_output,
    silu_gated_mlp_rows_bf16_preloaded_gate_up_resident_weight,
    silu_gated_mlp_rows_bf16_resident_weight, DeviceBf16Output,
    CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
    CUDA_REFERENCE_LINEAR_BF16_RESIDENT_WEIGHT_BACKEND,
    CUDA_REFERENCE_RMSNORM_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
    CUDA_REFERENCE_RMSNORM_BF16_RESIDENT_WEIGHT_BACKEND,
    CUDA_REFERENCE_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND,
    CUDA_REFERENCE_SILU_GATED_MLP_BF16_RESIDENT_WEIGHT_BACKEND,
};
use crate::commands::real_full::types::RealFullDensePrefixLayerProbe;

#[derive(Clone, Copy)]
pub(super) struct DensePrefixExecutionPlan {
    pub(super) intermediate_rows: usize,
    pub(super) output_rows: usize,
}

pub(super) struct DensePrefixExecution {
    pub(super) layers_executed: usize,
    pub(super) intermediate_rows: usize,
    pub(super) output_rows: usize,
    pub(super) residual_adds: usize,
    pub(super) norm_bytes_read: u64,
    pub(super) weight_bytes_read: u64,
    pub(super) norm_checksum: f64,
    pub(super) activation_checksum: f64,
    pub(super) output_checksum: f64,
    pub(super) output_l2_norm: f32,
    pub(super) initial_residual_checksum: f64,
    pub(super) residual_delta_checksum: f64,
    pub(super) final_residual_checksum: f64,
    pub(super) first_layer_id: Option<usize>,
    pub(super) last_layer_id: Option<usize>,
    pub(super) layer_summaries: Vec<RealFullDensePrefixLayerProbe>,
    pub(super) covers_all_dense_layers: bool,
    pub(super) covers_full_output_rows: bool,
    pub(super) passed: bool,
}

pub(super) struct DenseLayerResidualExecution {
    pub(super) hidden_after_layer: Vec<f32>,
    pub(super) device_hidden_after_layer: Option<DeviceBf16Output>,
    pub(super) layer_id: usize,
    pub(super) intermediate_rows: usize,
    pub(super) output_rows: usize,
    pub(super) residual_adds: usize,
    pub(super) norm_bytes_read: u64,
    pub(super) weight_bytes_read: u64,
    pub(super) norm_backend: &'static str,
    pub(super) linear_backend: &'static str,
    pub(super) mlp_backend: &'static str,
    pub(super) norm_checksum: f64,
    pub(super) activation_checksum: f64,
    pub(super) output_checksum: f64,
    pub(super) output_l2_norm: f32,
    pub(super) initial_residual_checksum: f64,
    pub(super) residual_delta_checksum: f64,
    pub(super) final_residual_checksum: f64,
    pub(super) residual_add_backend: &'static str,
    pub(super) first_residual_after: f32,
    pub(super) last_residual_after: f32,
    pub(super) passed: bool,
}

struct DenseLayerExecution {
    outputs: Vec<f32>,
    output_device: Option<DeviceBf16Output>,
    norm_bytes_read: u64,
    weight_bytes_read: u64,
    norm_backend: &'static str,
    linear_backend: &'static str,
    mlp_backend: &'static str,
    norm_checksum: f64,
    norm_l2_norm: f32,
    activation_checksum: f64,
    output_checksum: f64,
    output_l2_norm: f32,
}

fn dense_norm_backend_uses_resident_weight(backend: &str) -> bool {
    matches!(
        backend,
        CUDA_REFERENCE_RMSNORM_BF16_RESIDENT_WEIGHT_BACKEND
            | CUDA_REFERENCE_RMSNORM_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
    )
}

fn dense_linear_backend_uses_resident_weight(backend: &str) -> bool {
    matches!(
        backend,
        CUDA_REFERENCE_LINEAR_BF16_RESIDENT_WEIGHT_BACKEND
            | CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
    )
}

fn dense_mlp_backend_uses_resident_weight(backend: &str) -> bool {
    matches!(
        backend,
        CUDA_REFERENCE_SILU_GATED_MLP_BF16_RESIDENT_WEIGHT_BACKEND
            | CUDA_REFERENCE_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND
    )
}

fn dense_layer_weight_evidence(layer: &DenseLayerExecution) -> bool {
    layer.weight_bytes_read > 0
        || (dense_linear_backend_uses_resident_weight(layer.linear_backend)
            && dense_mlp_backend_uses_resident_weight(layer.mlp_backend))
}

fn dense_prefix_norm_evidence(
    norm_bytes_read: u64,
    layer_summaries: &[RealFullDensePrefixLayerProbe],
) -> bool {
    norm_bytes_read > 0
        || (!layer_summaries.is_empty()
            && layer_summaries
                .iter()
                .all(|layer| dense_norm_backend_uses_resident_weight(layer.norm_backend)))
}

fn dense_prefix_weight_evidence(
    weight_bytes_read: u64,
    layer_summaries: &[RealFullDensePrefixLayerProbe],
) -> bool {
    weight_bytes_read > 0
        || (!layer_summaries.is_empty()
            && layer_summaries.iter().all(|layer| {
                dense_linear_backend_uses_resident_weight(layer.linear_backend)
                    && dense_mlp_backend_uses_resident_weight(layer.mlp_backend)
            }))
}

#[derive(Clone, Copy)]
struct Bf16MatrixShape {
    rows: usize,
    width: usize,
}

#[derive(Clone, Copy)]
struct Bf16VectorShape {
    values: usize,
}

#[derive(Default)]
struct DenseResidualAddWorkspace {
    residual_bf16: Vec<u8>,
    delta_bf16: Vec<u8>,
    output_bf16: Vec<u8>,
}

struct DenseResidualAddResult {
    values: Vec<f32>,
    backend: &'static str,
    checksum: f64,
    first: f32,
    last: f32,
}

pub(super) fn bounded_dense_prefix_execution_plan() -> DensePrefixExecutionPlan {
    DensePrefixExecutionPlan {
        intermediate_rows: REAL_FULL_DENSE_PREFIX_INTERMEDIATE_ROWS,
        output_rows: REAL_FULL_DENSE_PREFIX_OUTPUT_ROWS,
    }
}

pub(super) fn full_output_dense_prefix_execution_plan() -> DensePrefixExecutionPlan {
    DensePrefixExecutionPlan {
        intermediate_rows: REAL_FULL_DENSE_PREFIX_INTERMEDIATE_ROWS,
        output_rows: GLM52_HIDDEN_SIZE,
    }
}

pub(super) fn execute_dense_prefix_with_plan(
    catalog: &TensorCatalog,
    plan: DensePrefixExecutionPlan,
) -> Result<DensePrefixExecution> {
    execute_dense_prefix_from_hidden_with_plan(
        catalog,
        deterministic_dense_hidden(GLM52_HIDDEN_SIZE),
        plan,
    )
}

pub(super) fn execute_dense_layer_residual_from_hidden(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden: Vec<f32>,
) -> Result<DenseLayerResidualExecution> {
    execute_dense_layer_residual_from_hidden_with_plan(
        catalog,
        layer_id,
        hidden,
        bounded_dense_prefix_execution_plan(),
    )
}

pub(super) fn execute_dense_layer_residual_from_hidden_with_plan(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden: Vec<f32>,
    plan: DensePrefixExecutionPlan,
) -> Result<DenseLayerResidualExecution> {
    execute_dense_layer_residual_from_hidden_with_plan_and_device_input(
        catalog, layer_id, hidden, None, plan,
    )
}

pub(super) fn execute_dense_layer_residual_from_hidden_with_plan_and_device_input(
    catalog: &TensorCatalog,
    layer_id: usize,
    mut hidden: Vec<f32>,
    device_hidden: Option<&DeviceBf16Output>,
    plan: DensePrefixExecutionPlan,
) -> Result<DenseLayerResidualExecution> {
    if layer_id >= GLM52_FIRST_K_DENSE_REPLACE {
        anyhow::bail!(
            "real full dense layer residual execution only supports dense layers 0..{}, got {layer_id}",
            GLM52_FIRST_K_DENSE_REPLACE
        );
    }
    if hidden.len() != GLM52_HIDDEN_SIZE {
        anyhow::bail!(
            "real full dense layer residual initial hidden width mismatch: expected {} got {}",
            GLM52_HIDDEN_SIZE,
            hidden.len()
        );
    }
    if plan.intermediate_rows == 0 || plan.output_rows == 0 {
        anyhow::bail!("real full dense layer residual execution plan requires non-zero rows");
    }
    if plan.output_rows > GLM52_HIDDEN_SIZE {
        anyhow::bail!(
            "real full dense layer residual output rows {} exceeds hidden size {}",
            plan.output_rows,
            GLM52_HIDDEN_SIZE
        );
    }
    if let Some(device_hidden) = device_hidden {
        if plan.output_rows != GLM52_HIDDEN_SIZE {
            anyhow::bail!(
                "real full dense layer device-hidden residual execution requires full-output rows, got {}",
                plan.output_rows
            );
        }
        if device_hidden.rows != 1 || device_hidden.values_per_row != hidden.len() {
            anyhow::bail!(
                "real full dense layer device-hidden shape mismatch: expected 1x{} got {}x{}",
                hidden.len(),
                device_hidden.rows,
                device_hidden.values_per_row
            );
        }
    }

    let mut residual_workspace = DenseResidualAddWorkspace::default();
    let layer = execute_dense_layer(catalog, layer_id, &hidden, device_hidden, plan)?;
    let residual_before = hidden[..plan.output_rows].to_vec();
    let initial_residual_checksum = checksum_f64(&residual_before);
    let (
        residual_values,
        residual_add_backend,
        first_residual_after,
        last_residual_after,
        device_hidden_after_layer,
    ) = if let (Some(device_hidden), Some(output_device)) =
        (device_hidden, layer.output_device.as_ref())
    {
        let residual_after = residual_add_bf16_device_inputs_output(device_hidden, output_device)?;
        let first = residual_after.values.first().copied().unwrap_or_default();
        let last = residual_after.values.last().copied().unwrap_or_default();
        (
            residual_after.values,
            residual_after.backend,
            first,
            last,
            Some(residual_after.device_output),
        )
    } else {
        let residual_after =
            dense_residual_add_bf16(&residual_before, &layer.outputs, &mut residual_workspace)?;
        (
            residual_after.values,
            residual_after.backend,
            residual_after.first,
            residual_after.last,
            None,
        )
    };
    let final_residual_checksum = checksum_f64(&residual_values);
    hidden[..plan.output_rows].copy_from_slice(&residual_values);
    let passed = (layer.norm_bytes_read > 0
        || dense_norm_backend_uses_resident_weight(layer.norm_backend))
        && dense_layer_weight_evidence(&layer)
        && layer.norm_checksum.is_finite()
        && layer.activation_checksum.is_finite()
        && layer.output_checksum.is_finite()
        && layer.output_l2_norm.is_finite()
        && final_residual_checksum.is_finite();

    Ok(DenseLayerResidualExecution {
        hidden_after_layer: hidden,
        device_hidden_after_layer,
        layer_id,
        intermediate_rows: plan.intermediate_rows,
        output_rows: plan.output_rows,
        residual_adds: 1,
        norm_bytes_read: layer.norm_bytes_read,
        weight_bytes_read: layer.weight_bytes_read,
        norm_backend: layer.norm_backend,
        linear_backend: layer.linear_backend,
        mlp_backend: layer.mlp_backend,
        norm_checksum: layer.norm_checksum,
        activation_checksum: layer.activation_checksum,
        output_checksum: layer.output_checksum,
        output_l2_norm: layer.output_l2_norm,
        initial_residual_checksum,
        residual_delta_checksum: layer.output_checksum,
        final_residual_checksum,
        residual_add_backend,
        first_residual_after,
        last_residual_after,
        passed,
    })
}

fn execute_dense_prefix_from_hidden_with_plan(
    catalog: &TensorCatalog,
    mut hidden: Vec<f32>,
    plan: DensePrefixExecutionPlan,
) -> Result<DensePrefixExecution> {
    if hidden.len() != GLM52_HIDDEN_SIZE {
        anyhow::bail!(
            "real full dense-prefix initial hidden width mismatch: expected {} got {}",
            GLM52_HIDDEN_SIZE,
            hidden.len()
        );
    }
    if plan.intermediate_rows == 0 || plan.output_rows == 0 {
        anyhow::bail!("real full dense-prefix execution plan requires non-zero rows");
    }
    if plan.output_rows > GLM52_HIDDEN_SIZE {
        anyhow::bail!(
            "real full dense-prefix output rows {} exceeds hidden size {}",
            plan.output_rows,
            GLM52_HIDDEN_SIZE
        );
    }
    let initial_prefix = hidden[..plan.output_rows].to_vec();
    let initial_residual_checksum = checksum_f64(&initial_prefix);
    let mut layer_summaries = Vec::with_capacity(GLM52_FIRST_K_DENSE_REPLACE);
    let mut norm_bytes_read = 0_u64;
    let mut weight_bytes_read = 0_u64;
    let mut norm_checksum = 0.0_f64;
    let mut activation_checksum = 0.0_f64;
    let mut output_checksum = 0.0_f64;
    let mut output_l2_sum = 0.0_f32;
    let mut residual_delta_checksum = 0.0_f64;
    let mut residual_workspace = DenseResidualAddWorkspace::default();

    for layer_id in 0..GLM52_FIRST_K_DENSE_REPLACE {
        let layer = execute_dense_layer(catalog, layer_id, &hidden, None, plan)?;
        let residual_before = hidden[..plan.output_rows].to_vec();
        let residual_before_checksum = checksum_f64(&residual_before);
        let residual_after =
            dense_residual_add_bf16(&residual_before, &layer.outputs, &mut residual_workspace)?;
        hidden[..plan.output_rows].copy_from_slice(&residual_after.values);
        let residual_after_checksum = residual_after.checksum;

        norm_bytes_read += layer.norm_bytes_read;
        weight_bytes_read += layer.weight_bytes_read;
        norm_checksum += layer.norm_checksum;
        activation_checksum += layer.activation_checksum;
        output_checksum += layer.output_checksum;
        output_l2_sum += layer.output_l2_norm * layer.output_l2_norm;
        residual_delta_checksum += layer.output_checksum;
        layer_summaries.push(RealFullDensePrefixLayerProbe {
            layer_id,
            norm_backend: layer.norm_backend,
            linear_backend: layer.linear_backend,
            mlp_backend: layer.mlp_backend,
            norm_checksum: layer.norm_checksum,
            norm_l2_norm: layer.norm_l2_norm,
            activation_checksum: layer.activation_checksum,
            output_checksum: layer.output_checksum,
            output_l2_norm: layer.output_l2_norm,
            residual_before_checksum,
            residual_delta_checksum: layer.output_checksum,
            residual_after_checksum,
            residual_add_backend: residual_after.backend,
            first_residual_after: residual_after.first,
            last_residual_after: residual_after.last,
        });
    }

    let layers_executed = layer_summaries.len();
    let first_layer_id = layer_summaries.first().map(|layer| layer.layer_id);
    let last_layer_id = layer_summaries.last().map(|layer| layer.layer_id);
    let final_prefix = hidden[..plan.output_rows].to_vec();
    let final_residual_checksum = checksum_f64(&final_prefix);
    let output_l2_norm = output_l2_sum.sqrt();
    let covers_all_dense_layers = layers_executed == GLM52_FIRST_K_DENSE_REPLACE
        && first_layer_id == Some(0)
        && last_layer_id == Some(GLM52_FIRST_K_DENSE_REPLACE - 1);
    let covers_full_output_rows = plan.output_rows == GLM52_HIDDEN_SIZE;
    let passed = covers_all_dense_layers
        && dense_prefix_norm_evidence(norm_bytes_read, &layer_summaries)
        && dense_prefix_weight_evidence(weight_bytes_read, &layer_summaries)
        && norm_checksum.is_finite()
        && activation_checksum.is_finite()
        && output_checksum.is_finite()
        && output_l2_norm.is_finite()
        && residual_delta_checksum.is_finite()
        && final_residual_checksum.is_finite();

    Ok(DensePrefixExecution {
        layers_executed,
        intermediate_rows: plan.intermediate_rows,
        output_rows: plan.output_rows,
        residual_adds: layers_executed,
        norm_bytes_read,
        weight_bytes_read,
        norm_checksum,
        activation_checksum,
        output_checksum,
        output_l2_norm,
        initial_residual_checksum,
        residual_delta_checksum,
        final_residual_checksum,
        first_layer_id,
        last_layer_id,
        layer_summaries,
        covers_all_dense_layers,
        covers_full_output_rows,
        passed,
    })
}

fn dense_residual_add_bf16(
    residual: &[f32],
    delta: &[f32],
    workspace: &mut DenseResidualAddWorkspace,
) -> Result<DenseResidualAddResult> {
    if residual.len() != delta.len() {
        anyhow::bail!(
            "real full dense residual-add length mismatch: residual={} delta={}",
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
    Ok(DenseResidualAddResult {
        first: values.first().copied().unwrap_or_default(),
        last: values.last().copied().unwrap_or_default(),
        values,
        backend,
        checksum,
    })
}

fn execute_dense_layer(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden: &[f32],
    device_hidden: Option<&DeviceBf16Output>,
    plan: DensePrefixExecutionPlan,
) -> Result<DenseLayerExecution> {
    let norm_name = format!("model.layers.{layer_id}.post_attention_layernorm.weight");
    let gate_name = format!("model.layers.{layer_id}.mlp.gate_proj.weight");
    let up_name = format!("model.layers.{layer_id}.mlp.up_proj.weight");
    let down_name = format!("model.layers.{layer_id}.mlp.down_proj.weight");
    let norm_shape = catalog_bf16_vector_shape(
        catalog,
        &norm_name,
        "real full dense-prefix post-attention norm",
    )?;
    let gate_shape = catalog_bf16_matrix_shape(catalog, &gate_name, "real full dense-prefix gate")?;
    let up_shape = catalog_bf16_matrix_shape(catalog, &up_name, "real full dense-prefix up")?;
    let down_shape = catalog_bf16_matrix_shape(catalog, &down_name, "real full dense-prefix down")?;
    let cuda_reference_enabled = coordinator_cuda_reference_kernels_enabled();
    let norm_full_resident =
        cuda_reference_enabled && bf16_full_vector_resident_available(&norm_name, norm_shape);
    let mut norm_bytes_read = 0_u64;
    let norm = if norm_full_resident {
        None
    } else if cuda_reference_enabled {
        norm_bytes_read = preload_dense_norm_resident_from_host_staging(
            catalog,
            &norm_name,
            norm_shape,
            "BF16 dense post-attention norm pinned staging",
        )?;
        None
    } else {
        let norm = load_tensor_bytes(catalog, &norm_name)?;
        norm_bytes_read = norm.bytes.len() as u64;
        Some(norm)
    };
    let gate_up_full_resident = cuda_reference_enabled
        && plan.output_rows == hidden.len()
        && up_shape.rows == gate_shape.rows
        && bf16_full_matrix_resident_available(&gate_name, gate_shape)
        && bf16_full_matrix_resident_available(&up_name, up_shape);
    let down_full_resident = gate_up_full_resident
        && down_shape.rows == plan.output_rows
        && down_shape.width == gate_shape.rows
        && bf16_full_matrix_resident_available(&down_name, down_shape);
    let gate_weight_key = format!("{gate_name}[rows=0..{}]", plan.intermediate_rows);
    let up_weight_key = format!("{up_name}[rows=0..{}]", plan.intermediate_rows);
    let down_weight_key = format!(
        "{down_name}[rows=0..{},cols=0..{}]",
        plan.output_rows, plan.intermediate_rows
    );
    let stage_dense_mlp_row_windows =
        cuda_reference_enabled && plan.output_rows == hidden.len() && !gate_up_full_resident;
    let mut weight_bytes_read = 0_u64;
    let gate = if gate_up_full_resident {
        None
    } else if stage_dense_mlp_row_windows {
        weight_bytes_read += preload_dense_rows_resident_from_host_staging(
            catalog,
            &gate_name,
            &gate_weight_key,
            plan.intermediate_rows,
            gate_shape.width,
            "BF16 dense gate row-window pinned staging",
        )?;
        None
    } else {
        let rows = load_tensor_rows(catalog, &gate_name, 0, plan.intermediate_rows)?;
        weight_bytes_read += rows.bytes.len() as u64;
        Some(rows)
    };
    let up = if gate_up_full_resident {
        None
    } else if stage_dense_mlp_row_windows {
        weight_bytes_read += preload_dense_rows_resident_from_host_staging(
            catalog,
            &up_name,
            &up_weight_key,
            plan.intermediate_rows,
            up_shape.width,
            "BF16 dense up row-window pinned staging",
        )?;
        None
    } else {
        let rows = load_tensor_rows(catalog, &up_name, 0, plan.intermediate_rows)?;
        weight_bytes_read += rows.bytes.len() as u64;
        Some(rows)
    };
    let down = if down_full_resident {
        None
    } else if stage_dense_mlp_row_windows {
        weight_bytes_read += preload_dense_row_prefix_resident_from_host_staging(
            catalog,
            &down_name,
            &down_weight_key,
            plan.output_rows,
            plan.intermediate_rows,
            "BF16 dense down row-prefix pinned staging",
        )?;
        None
    } else {
        let rows = load_tensor_rows(catalog, &down_name, 0, plan.output_rows)?;
        weight_bytes_read += rows.bytes.len() as u64;
        Some(rows)
    };
    if norm_shape.values != hidden.len() {
        anyhow::bail!(
            "real full dense-prefix norm shape mismatch for layer {layer_id}: {:?} for hidden {}",
            vec![norm_shape.values],
            hidden.len()
        );
    }
    if gate_shape.width != hidden.len() || up_shape.width != hidden.len() {
        anyhow::bail!(
            "real full dense-prefix hidden width mismatch for layer {layer_id}: hidden={} gate_width={} up_width={}",
            hidden.len(),
            gate_shape.width,
            up_shape.width
        );
    }
    if gate_shape.rows < plan.intermediate_rows || up_shape.rows < plan.intermediate_rows {
        anyhow::bail!(
            "real full dense-prefix intermediate prefix exceeds gate/up rows for layer {layer_id}: intermediate={} gate_rows={} up_rows={}",
            plan.intermediate_rows,
            gate_shape.rows,
            up_shape.rows
        );
    }
    if down_shape.rows < plan.output_rows || down_shape.width < plan.intermediate_rows {
        anyhow::bail!(
            "real full dense-prefix down shape mismatch for layer {layer_id}: rows={} width={} expected_rows={} min_width={}",
            down_shape.rows,
            down_shape.width,
            plan.output_rows,
            plan.intermediate_rows
        );
    }

    let device_norm_path = device_hidden.is_some()
        && norm_full_resident
        && gate_up_full_resident
        && down_full_resident
        && plan.output_rows == hidden.len();
    let hidden_bf16 = bf16_bytes_from_f32(hidden);
    let normalized_device = if device_norm_path {
        let device_hidden =
            device_hidden.expect("device_norm_path requires a dense device hidden input");
        Some(
            rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output(
                &norm_name,
                device_hidden.buffer(),
                1,
                hidden.len(),
                REAL_FULL_DENSE_RMSNORM_EPS,
            )?,
        )
    } else {
        None
    };
    let normalized = if let Some(normalized_device) = normalized_device.as_ref() {
        let values = normalized_device.copy_to_host_values()?;
        if values.len() != hidden.len() {
            anyhow::bail!(
                "real full dense device RMSNorm readback length mismatch: expected {} got {}",
                hidden.len(),
                values.len()
            );
        }
        crate::commands::real_full::coordinator_kernels::RmsNormOutput {
            values,
            backend: normalized_device.backend,
        }
    } else if norm_full_resident || cuda_reference_enabled {
        rmsnorm_hidden_bf16_preloaded_resident_weight(
            &norm_name,
            &hidden_bf16,
            1,
            hidden.len(),
            REAL_FULL_DENSE_RMSNORM_EPS,
        )?
    } else {
        let norm = norm.as_ref().ok_or_else(|| {
            anyhow::anyhow!("real full dense-prefix missing loaded norm for {norm_name}")
        })?;
        rmsnorm_hidden_bf16_resident_weight(
            &norm_name,
            &hidden_bf16,
            &norm.bytes,
            1,
            hidden.len(),
            REAL_FULL_DENSE_RMSNORM_EPS,
        )?
    };
    let norm_backend = normalized.backend;
    let normalized = normalized.values;
    let normalized_bf16 = bf16_bytes_from_f32(&normalized);
    let gate_projection = if gate_up_full_resident {
        linear_rows_bf16_preloaded_resident_weight(
            &gate_name,
            &normalized_bf16,
            None,
            1,
            hidden.len(),
            plan.intermediate_rows,
            gate_shape.rows,
        )?
    } else if stage_dense_mlp_row_windows {
        linear_rows_bf16_preloaded_resident_weight(
            &gate_weight_key,
            &normalized_bf16,
            None,
            1,
            hidden.len(),
            plan.intermediate_rows,
            plan.intermediate_rows,
        )?
    } else {
        let gate = required_loaded_rows(gate.as_ref(), &gate_weight_key)?;
        linear_rows_bf16_resident_weight(
            &gate_weight_key,
            &normalized_bf16,
            &gate.bytes,
            None,
            1,
            hidden.len(),
            plan.intermediate_rows,
        )?
    };
    let linear_backend = gate_projection.backend;
    let up_projection = if gate_up_full_resident {
        linear_rows_bf16_preloaded_resident_weight(
            &up_name,
            &normalized_bf16,
            None,
            1,
            hidden.len(),
            plan.intermediate_rows,
            up_shape.rows,
        )?
    } else if stage_dense_mlp_row_windows {
        linear_rows_bf16_preloaded_resident_weight(
            &up_weight_key,
            &normalized_bf16,
            None,
            1,
            hidden.len(),
            plan.intermediate_rows,
            plan.intermediate_rows,
        )?
    } else {
        let up = required_loaded_rows(up.as_ref(), &up_weight_key)?;
        linear_rows_bf16_resident_weight(
            &up_weight_key,
            &normalized_bf16,
            &up.bytes,
            None,
            1,
            hidden.len(),
            plan.intermediate_rows,
        )?
    };
    if up_projection.backend != linear_backend {
        anyhow::bail!(
            "real full dense-prefix probe mixed linear backends at layer {layer_id}: gate={} up={}",
            linear_backend,
            up_projection.backend
        );
    }

    let mut activations = Vec::with_capacity(plan.intermediate_rows);
    for row_index in 0..gate_projection.values.len() {
        let gate_value = gate_projection.values[row_index];
        let up_value = up_projection.values[row_index];
        let activation = silu(gate_value) * up_value;
        if !activation.is_finite() {
            anyhow::bail!(
                "real full dense-prefix probe produced non-finite activation at layer {layer_id} row {row_index}"
            );
        }
        activations.push(activation);
    }

    let (outputs, mlp_backend, output_device) = if let Some(normalized_device) =
        normalized_device.as_ref()
    {
        let mlp_output =
                silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight_device_input_device_output(
                    &gate_name,
                    &up_name,
                    &down_name,
                    normalized_device.buffer(),
                    1,
                    hidden.len(),
                    plan.intermediate_rows,
                    gate_shape.rows,
                    plan.output_rows,
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
            &normalized_bf16,
            1,
            hidden.len(),
            plan.intermediate_rows,
            gate_shape.rows,
            plan.output_rows,
        )?;
        (mlp_output.values, mlp_output.backend, None)
    } else if stage_dense_mlp_row_windows {
        let mlp_output = silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight(
            &gate_weight_key,
            &up_weight_key,
            &down_weight_key,
            &normalized_bf16,
            1,
            hidden.len(),
            plan.intermediate_rows,
            plan.intermediate_rows,
            plan.output_rows,
        )?;
        (mlp_output.values, mlp_output.backend, None)
    } else if gate_up_full_resident {
        let down = required_loaded_rows(down.as_ref(), &down_weight_key)?;
        let down_prefix_bytes = bf16_compact_row_prefix_bytes(
            &down.bytes,
            plan.output_rows,
            down.row_width,
            plan.intermediate_rows,
        )?;
        let mlp_output = silu_gated_mlp_rows_bf16_preloaded_gate_up_resident_weight(
            &gate_name,
            &up_name,
            &down_weight_key,
            &normalized_bf16,
            &down_prefix_bytes,
            1,
            hidden.len(),
            plan.intermediate_rows,
            gate_shape.rows,
            plan.output_rows,
        )?;
        (mlp_output.values, mlp_output.backend, None)
    } else {
        let gate = required_loaded_rows(gate.as_ref(), &gate_weight_key)?;
        let up = required_loaded_rows(up.as_ref(), &up_weight_key)?;
        let down = required_loaded_rows(down.as_ref(), &down_weight_key)?;
        let down_prefix_bytes = bf16_compact_row_prefix_bytes(
            &down.bytes,
            plan.output_rows,
            down.row_width,
            plan.intermediate_rows,
        )?;
        let mlp_output = silu_gated_mlp_rows_bf16_resident_weight(
            &gate_weight_key,
            &up_weight_key,
            &down_weight_key,
            &normalized_bf16,
            &gate.bytes,
            &up.bytes,
            &down_prefix_bytes,
            1,
            hidden.len(),
            plan.intermediate_rows,
            plan.output_rows,
        )?;
        (mlp_output.values, mlp_output.backend, None)
    };
    for (row_index, output) in outputs.iter().copied().enumerate() {
        if !output.is_finite() {
            anyhow::bail!(
                "real full dense-prefix probe produced non-finite output at layer {layer_id} row {row_index}"
            );
        }
    }

    let norm_checksum = checksum_f64(&normalized);
    let norm_l2_norm = normalized
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    let activation_checksum = checksum_f64(&activations);
    let output_checksum = checksum_f64(&outputs);
    let output_l2_norm = outputs
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    Ok(DenseLayerExecution {
        outputs,
        output_device,
        norm_bytes_read,
        weight_bytes_read,
        norm_backend,
        linear_backend,
        mlp_backend,
        norm_checksum,
        norm_l2_norm,
        activation_checksum,
        output_checksum,
        output_l2_norm,
    })
}

fn preload_dense_norm_resident_from_host_staging(
    catalog: &TensorCatalog,
    norm_name: &str,
    norm_shape: Bf16VectorShape,
    label: &'static str,
) -> Result<u64> {
    let expected_bytes = norm_shape
        .values
        .checked_mul(std::mem::size_of::<u16>())
        .context("real full dense norm byte length overflows usize")?;
    let mut bytes_read = 0_u64;
    preload_resident_weight_from_host_staging(norm_name, expected_bytes, label, |staging| {
        let summary = read_tensor_bytes_into(catalog, norm_name, staging).with_context(|| {
            format!("reading dense norm tensor {norm_name} into pinned staging")
        })?;
        if summary.dtype != DType::Bf16 {
            anyhow::bail!(
                "real full dense norm tensor {norm_name} expects BF16, got {:?}",
                summary.dtype
            );
        }
        if summary.shape != vec![norm_shape.values] {
            anyhow::bail!(
                "real full dense norm tensor {norm_name} shape mismatch: expected {:?} got {:?}",
                vec![norm_shape.values],
                summary.shape
            );
        }
        if summary.bytes_read as usize != expected_bytes {
            anyhow::bail!(
                "real full dense norm tensor {norm_name} read {} bytes, expected {}",
                summary.bytes_read,
                expected_bytes
            );
        }
        bytes_read = summary.bytes_read;
        Ok(())
    })
    .with_context(|| format!("preloading dense norm tensor {norm_name} from pinned staging"))?;
    Ok(bytes_read)
}

fn preload_dense_rows_resident_from_host_staging(
    catalog: &TensorCatalog,
    tensor_name: &str,
    resident_name: &str,
    row_count: usize,
    row_width: usize,
    label: &'static str,
) -> Result<u64> {
    let expected_bytes = row_count
        .checked_mul(row_width)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("real full dense row-window byte length overflows usize")?;
    let mut bytes_read = 0_u64;
    preload_resident_weight_from_host_staging(resident_name, expected_bytes, label, |staging| {
        let summary = read_tensor_rows_into(catalog, tensor_name, 0, row_count, staging)
            .with_context(|| {
                format!(
                    "reading dense row window {tensor_name}[rows=0..{row_count}] into pinned staging"
                )
            })?;
        if summary.dtype != DType::Bf16 {
            anyhow::bail!(
                "real full dense row-window tensor {tensor_name} expects BF16, got {:?}",
                summary.dtype
            );
        }
        if summary.row_count != row_count || summary.row_width != row_width {
            anyhow::bail!(
                "real full dense row-window tensor {tensor_name} shape mismatch: expected rows={} width={} got rows={} width={}",
                row_count,
                row_width,
                summary.row_count,
                summary.row_width
            );
        }
        if summary.bytes_read as usize != expected_bytes {
            anyhow::bail!(
                "real full dense row-window tensor {tensor_name} read {} bytes, expected {}",
                summary.bytes_read,
                expected_bytes
            );
        }
        bytes_read = summary.bytes_read;
        Ok(())
    })
    .with_context(|| {
        format!("preloading dense row-window tensor {resident_name} from pinned staging")
    })?;
    Ok(bytes_read)
}

fn preload_dense_row_prefix_resident_from_host_staging(
    catalog: &TensorCatalog,
    tensor_name: &str,
    resident_name: &str,
    row_count: usize,
    prefix_width: usize,
    label: &'static str,
) -> Result<u64> {
    let expected_bytes = row_count
        .checked_mul(prefix_width)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("real full dense row-prefix byte length overflows usize")?;
    let mut bytes_read = 0_u64;
    preload_resident_weight_from_host_staging(resident_name, expected_bytes, label, |staging| {
        let summary =
            read_tensor_row_prefix_into(catalog, tensor_name, 0, row_count, prefix_width, staging)
                .with_context(|| {
                    format!(
                        "reading dense row prefix {tensor_name}[rows=0..{row_count}, cols=0..{prefix_width}] into pinned staging"
                    )
                })?;
        if summary.dtype != DType::Bf16 {
            anyhow::bail!(
                "real full dense row-prefix tensor {tensor_name} expects BF16, got {:?}",
                summary.dtype
            );
        }
        if summary.row_count != row_count || summary.row_width != prefix_width {
            anyhow::bail!(
                "real full dense row-prefix tensor {tensor_name} shape mismatch: expected rows={} prefix_width={} got rows={} width={}",
                row_count,
                prefix_width,
                summary.row_count,
                summary.row_width
            );
        }
        if summary.bytes_read as usize != expected_bytes {
            anyhow::bail!(
                "real full dense row-prefix tensor {tensor_name} read {} bytes, expected {}",
                summary.bytes_read,
                expected_bytes
            );
        }
        bytes_read = summary.bytes_read;
        Ok(())
    })
    .with_context(|| {
        format!("preloading dense row-prefix tensor {resident_name} from pinned staging")
    })?;
    Ok(bytes_read)
}

fn required_loaded_rows<'a>(
    rows: Option<&'a LoadedTensorRows>,
    weight_name: &str,
) -> Result<&'a LoadedTensorRows> {
    rows.ok_or_else(|| {
        anyhow::anyhow!("real full dense-prefix missing loaded rows for {weight_name}")
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

fn catalog_bf16_vector_shape(
    catalog: &TensorCatalog,
    name: &str,
    context: &str,
) -> Result<Bf16VectorShape> {
    let info = catalog_tensor(catalog, name)?;
    validate_bf16_vector_shape(info, context)
}

fn catalog_tensor<'a>(catalog: &'a TensorCatalog, name: &str) -> Result<&'a TensorInfo> {
    catalog
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| anyhow::anyhow!("tensor {name} not found in real full dense-prefix catalog"))
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

fn validate_bf16_vector_shape(info: &TensorInfo, context: &str) -> Result<Bf16VectorShape> {
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
    Ok(Bf16VectorShape {
        values: info.shape[0],
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

fn bf16_full_vector_resident_available(name: &str, shape: Bf16VectorShape) -> bool {
    let Some(bytes) = shape.values.checked_mul(std::mem::size_of::<u16>()) else {
        return false;
    };
    resident_weight_is_preloaded(name, bytes)
}
