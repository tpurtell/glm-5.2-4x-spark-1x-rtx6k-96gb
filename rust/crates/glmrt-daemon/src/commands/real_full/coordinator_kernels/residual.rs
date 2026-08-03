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
use glmrt_transport::ExpertV2Dtype;
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::slice;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

#[allow(dead_code)]
pub(in crate::commands::real_full) const CPU_REFERENCE_RESIDUAL_ADD_BACKEND: &str =
    "cpu-reference-residual-add";
#[allow(dead_code)]
pub(in crate::commands::real_full) const CUDA_REFERENCE_RESIDUAL_ADD_BACKEND: &str =
    "cuda-reference-residual-add-f32";
pub(in crate::commands::real_full) const CPU_REFERENCE_RESIDUAL_ADD_BF16_BACKEND: &str =
    "cpu-reference-residual-add-bf16";
pub(in crate::commands::real_full) const CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND: &str =
    "cuda-reference-residual-add-bf16";
pub(in crate::commands::real_full) const CPU_REFERENCE_GATHER_ROWS_BF16_BACKEND: &str =
    "cpu-reference-gather-rows-bf16";
pub(in crate::commands::real_full) const CUDA_REFERENCE_GATHER_ROWS_BF16_BACKEND: &str =
    "cuda-reference-gather-rows-bf16";

pub(in crate::commands::real_full) const CPU_REFERENCE_SCATTER_ADD_ROWS_BF16_TO_F32_BACKEND: &str =
    "cpu-reference-scatter-add-rows-bf16-to-f32";
pub(in crate::commands::real_full) const CUDA_REFERENCE_SCATTER_ADD_ROWS_BF16_TO_F32_BACKEND: &str =
    "cuda-reference-scatter-add-rows-bf16-to-f32";

static SPARSE_B_SCATTER_GRAPH_DISABLED: AtomicBool = AtomicBool::new(false);

pub(in crate::commands::real_full) fn residual_add_prefix_bf16_bytes_into(
    residual_bf16: &[u8],
    delta_bf16: &[u8],
    output_bf16: &mut [u8],
) -> Result<&'static str> {
    validate_residual_add_bf16_inputs(residual_bf16, delta_bf16)?;
    if output_bf16.len() != residual_bf16.len() {
        anyhow::bail!(
            "real full BF16 residual add output byte length mismatch: expected {} got {}",
            residual_bf16.len(),
            output_bf16.len()
        );
    }
    if cuda_reference_kernels_enabled() {
        return cuda_residual_add_prefix_bf16_bytes_into(residual_bf16, delta_bf16, output_bf16);
    }
    Ok(cpu_residual_add_prefix_bf16_bytes_into(
        residual_bf16,
        delta_bf16,
        output_bf16,
    ))
}

#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn residual_add_bf16_device_inputs_output(
    residual: &DeviceBf16Output,
    delta: &DeviceBf16Output,
) -> Result<ResidualAddDeviceOutput> {
    let bytes = validate_residual_add_bf16_device_inputs(residual, delta)?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!("BF16 device residual-add requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1");
    }
    cuda_residual_add_bf16_device_inputs_output(residual, delta, bytes)
}

#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn residual_add_bf16_device_inputs_device_output(
    residual: &DeviceBf16Output,
    delta: &DeviceBf16Output,
) -> Result<DeviceBf16Output> {
    let bytes = validate_residual_add_bf16_device_inputs(residual, delta)?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!("BF16 device residual-add requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1");
    }
    cuda_residual_add_bf16_device_inputs_device_output(residual, delta, bytes)
}

pub(in crate::commands::real_full) fn residual_add_bf16_device_input_delta_view_device_output(
    residual: &DeviceBf16Output,
    delta_source: &DeviceBf16Output,
    delta: GlmrtDeviceBuffer,
) -> Result<DeviceBf16Output> {
    let rows = residual.rows;
    let values_per_row = residual.values_per_row;
    let bytes = rows
        .checked_mul(values_per_row)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("BF16 device residual-add delta view byte shape overflows usize")?;
    let residual_buffer = residual.buffer();
    anyhow::ensure!(
        residual_buffer.device_id == delta.device_id,
        "BF16 device residual-add delta view device mismatch: residual={} delta={}",
        residual_buffer.device_id,
        delta.device_id
    );
    anyhow::ensure!(
        !delta.ptr.is_null() && delta.bytes >= bytes,
        "BF16 device residual-add delta view has {} bytes, expected at least {bytes}",
        delta.bytes
    );
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!("BF16 device residual-add requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1");
    }

    if let Some(graph_key) = coord_sparse_b_graph_key_for_bf16_full_rows(bytes)? {
        if let Ok(output) = with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
            let output_buffer = OwnedCoordinatorDeviceBuffer::new(
                library,
                bytes,
                "BF16 device-input residual-add delta-view output",
            )?;
            let signature = CoordinatorCudaGraphSignature::residual_add_bf16(bytes);
            let stream = slot.stream_ptr();
            residual
                .wait_ready_on_stream(stream)
                .context("waiting for BF16 residual input before delta-view residual add")?;
            delta_source
                .wait_ready_on_stream(stream)
                .context("waiting for BF16 delta view before residual add")?;
            capture_or_update_sparse_b_residual_add_bf16_graph(
                library,
                slot,
                signature,
                residual_buffer,
                delta,
                output_buffer.buffer,
                bytes / std::mem::size_of::<u16>(),
                "BF16 device-input residual add delta view",
            )?;
            unsafe {
                library
                    .cuda_stream_synchronize(stream)
                    .context("synchronizing BF16 residual-add delta-view graph")?;
            }
            Ok(DeviceBf16Output {
                buffer: output_buffer,
                bytes,
                rows,
                values_per_row,
                backend: CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
            })
        }) {
            return Ok(output);
        }
    }

    let library = cuda_native_library()?;
    let output_buffer = OwnedCoordinatorDeviceBuffer::new(
        library,
        bytes,
        "BF16 device-input residual-add delta-view output",
    )?;
    residual
        .synchronize_ready()
        .context("waiting for BF16 residual input before legacy delta-view residual add")?;
    delta_source
        .synchronize_ready()
        .context("waiting for BF16 delta view before legacy residual add")?;
    library
        .cuda_residual_add_bf16(
            residual_buffer,
            delta,
            output_buffer.buffer,
            bytes / std::mem::size_of::<u16>(),
        )
        .context("executing CUDA BF16 device-input residual add delta view")?;
    Ok(DeviceBf16Output {
        buffer: output_buffer,
        bytes,
        rows,
        values_per_row,
        backend: CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
    })
}

pub(in crate::commands::real_full) fn validate_residual_add_bf16_device_inputs(
    residual: &DeviceBf16Output,
    delta: &DeviceBf16Output,
) -> Result<usize> {
    if residual.rows != delta.rows || residual.values_per_row != delta.values_per_row {
        anyhow::bail!(
            "real full BF16 device residual-add shape mismatch: residual={}x{} delta={}x{}",
            residual.rows,
            residual.values_per_row,
            delta.rows,
            delta.values_per_row
        );
    }
    let residual_buffer = residual.buffer();
    let delta_buffer = delta.buffer();
    if residual_buffer.device_id != delta_buffer.device_id {
        anyhow::bail!(
            "real full BF16 device residual-add buffers are on different devices: residual={} delta={}",
            residual_buffer.device_id,
            delta_buffer.device_id
        );
    }
    let bytes = residual
        .rows
        .checked_mul(residual.values_per_row)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("real full BF16 device residual-add byte shape overflows usize")?;
    if residual_buffer.bytes < bytes || delta_buffer.bytes < bytes {
        anyhow::bail!(
            "real full BF16 device residual-add buffer byte length mismatch: expected at least {bytes}, residual={} delta={}",
            residual_buffer.bytes,
            delta_buffer.bytes
        );
    }
    Ok(bytes)
}

pub(in crate::commands::real_full) fn gather_rows_bf16(
    src_bf16: &[u8],
    row_indices: &[usize],
    src_rows: usize,
    row_width: usize,
) -> Result<GatherRowsBf16Output> {
    let row_indices = validate_gather_rows_bf16_inputs(src_bf16, row_indices, src_rows, row_width)?;
    if cuda_reference_kernels_enabled() {
        return cuda_gather_rows_bf16(src_bf16, &row_indices, src_rows, row_width);
    }
    Ok(cpu_gather_rows_bf16(
        src_bf16,
        &row_indices,
        src_rows,
        row_width,
    ))
}

pub(in crate::commands::real_full) fn scatter_add_rows_bf16_to_f32(
    src_bf16: &[u8],
    row_indices: &[usize],
    dst_rows: usize,
    row_width: usize,
    initial: Option<&[f32]>,
) -> Result<ScatterAddRowsBf16ToF32Output> {
    let row_indices = validate_scatter_add_rows_bf16_to_f32_inputs(
        src_bf16,
        row_indices,
        dst_rows,
        row_width,
        initial,
    )?;
    if cuda_reference_kernels_enabled() {
        return cuda_scatter_add_rows_bf16_to_f32(
            src_bf16,
            &row_indices,
            dst_rows,
            row_width,
            initial,
        );
    }
    Ok(cpu_scatter_add_rows_bf16_to_f32(
        src_bf16,
        &row_indices,
        dst_rows,
        row_width,
        initial,
    ))
}

pub(in crate::commands::real_full) fn sparse_b_scatter_residual_add_bf16(
    residual: &[f32],
    initial_delta: &[f32],
    partial_outputs_bf16_by_host: &[impl AsRef<[u8]>],
    global_row_indices_by_host: &[impl AsRef<[usize]>],
    dst_rows: usize,
    row_width: usize,
) -> Result<SparseBScatterResidualAddOutput> {
    let (src_bf16, row_indices) = flatten_sparse_b_bf16_partials(
        partial_outputs_bf16_by_host,
        global_row_indices_by_host,
        dst_rows,
        row_width,
    )?;
    let row_indices = validate_sparse_b_scatter_residual_add_inputs(
        residual,
        initial_delta,
        &src_bf16,
        &row_indices,
        dst_rows,
        row_width,
    )?;
    if cuda_reference_kernels_enabled() {
        return cuda_sparse_b_scatter_residual_add_bf16(
            residual,
            initial_delta,
            &src_bf16,
            &row_indices,
            dst_rows,
            row_width,
        );
    }
    cpu_sparse_b_scatter_residual_add_bf16(
        residual,
        initial_delta,
        &src_bf16,
        &row_indices,
        dst_rows,
        row_width,
    )
}

pub(in crate::commands::real_full) fn sparse_b_scatter_shared_residual_add_bf16_device_output(
    residual: &DeviceBf16Output,
    shared_delta: &DeviceBf16Output,
    partial_outputs_bf16_by_host: &[impl AsRef<[u8]>],
    global_row_indices_by_host: &[impl AsRef<[usize]>],
    dst_rows: usize,
    row_width: usize,
) -> Result<DeviceBf16Output> {
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "BF16 Sparse-B scatter shared residual-add requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1"
        );
    }
    validate_sparse_b_device_residual_inputs(residual, shared_delta, dst_rows, row_width)?;
    let partials = validate_sparse_b_bf16_partial_layout(
        partial_outputs_bf16_by_host,
        global_row_indices_by_host,
        dst_rows,
        row_width,
    )?;
    if partials.row_indices.is_empty() {
        return residual_add_bf16_device_inputs_device_output(residual, shared_delta);
    }
    cuda_sparse_b_scatter_shared_residual_add_bf16_device_output(
        residual,
        shared_delta,
        partial_outputs_bf16_by_host,
        &partials,
        dst_rows,
        row_width,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn sparse_b_scatter_shared_residual_add_low_precision_device_output(
    residual: &DeviceBf16Output,
    shared_delta: &DeviceBf16Output,
    partial_outputs: &[impl AsRef<[u8]>],
    global_row_indices: &[impl AsRef<[usize]>],
    output_dtype: ExpertV2Dtype,
    output_row_stride_bytes: usize,
    host_ordered: bool,
    dst_rows: usize,
    row_width: usize,
) -> Result<DeviceBf16Output> {
    anyhow::ensure!(
        matches!(
            output_dtype,
            ExpertV2Dtype::Fp8E4m3RowScaled | ExpertV2Dtype::Nvfp4E2m1Fp8E4m3
        ),
        "low-precision Sparse-B requires FP8 or NVFP4 partials, got {output_dtype:?}"
    );
    if host_ordered {
        anyhow::ensure!(
            partial_outputs.len() <= 4,
            "host-ordered low-precision Sparse-B supports at most four host payloads"
        );
        for rows in global_row_indices {
            let rows = rows.as_ref();
            for (row_offset, row) in rows.iter().enumerate() {
                anyhow::ensure!(
                    !rows[..row_offset].contains(row),
                    "host-ordered low-precision Sparse-B host payload repeats global row {row}"
                );
            }
        }
    }
    validate_sparse_b_device_residual_inputs(residual, shared_delta, dst_rows, row_width)?;
    let partials = validate_sparse_b_low_precision_partial_layout(
        partial_outputs,
        global_row_indices,
        output_dtype,
        output_row_stride_bytes,
        dst_rows,
        row_width,
    )?;
    if partials.row_indices.is_empty() {
        return residual_add_bf16_device_inputs_device_output(residual, shared_delta);
    }
    if output_dtype == ExpertV2Dtype::Fp8E4m3RowScaled
        && dst_rows == 1
        && (1..=4).contains(&partials.row_indices.len())
        && partials.row_indices.iter().all(|row| *row == 0)
    {
        return cuda_sparse_b_single_row_fp8_shared_residual_add_device_output(
            residual,
            shared_delta,
            partial_outputs,
            output_row_stride_bytes,
            partials.row_indices.len(),
            row_width,
        );
    }
    cuda_sparse_b_scatter_shared_residual_add_low_precision_device_output(
        residual,
        shared_delta,
        partial_outputs,
        &partials,
        output_dtype,
        output_row_stride_bytes,
        host_ordered,
        dst_rows,
        row_width,
    )
}

fn cuda_sparse_b_single_row_fp8_shared_residual_add_device_output(
    residual: &DeviceBf16Output,
    shared_delta: &DeviceBf16Output,
    partial_outputs: &[impl AsRef<[u8]>],
    output_row_stride_bytes: usize,
    partial_rows: usize,
    row_width: usize,
) -> Result<DeviceBf16Output> {
    let partial_bytes = partial_rows
        .checked_mul(output_row_stride_bytes)
        .context("single-row FP8 Sparse-B partial byte count overflows usize")?;
    let bf16_bytes = row_width
        .checked_mul(std::mem::size_of::<u16>())
        .context("single-row FP8 Sparse-B output byte count overflows usize")?;
    let graph_key = CoordinatorGraphKey::glm52_bf16(
        CoordinatorGraphShape::CoordSparseB,
        LayerWaveMode::Decode,
        1,
    )
    .context("selecting Coord-Sparse-B graph slot for single-row FP8 accumulation")?;
    let device_output_buffer = with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let src_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            partial_bytes,
            "single-row FP8 Sparse-B partials",
        )?;
        let device_output_buffer = OwnedCoordinatorDeviceBuffer::new(
            library,
            bf16_bytes,
            "single-row FP8 Sparse-B owned BF16 output",
        )?;

        slot.workspace
            .copy_h2d_segments_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::A,
                partial_outputs,
                partial_bytes,
                "single-row FP8 Sparse-B partials",
                cuda_stream,
            )
            .context("copying single-row FP8 Sparse-B partials to device")?;
        unsafe {
            library
                .cuda_fp8_decode_combine_residual_async(
                    residual.buffer(),
                    shared_delta.buffer(),
                    src_buffer,
                    output_row_stride_bytes,
                    device_output_buffer.buffer,
                    partial_rows,
                    row_width,
                    cuda_stream,
                )
                .context("executing single-row FP8 Sparse-B combine and residual add")?;
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing single-row FP8 Sparse-B residual add")?;
        }
        Ok(device_output_buffer)
    })?;

    Ok(DeviceBf16Output {
        buffer: device_output_buffer,
        bytes: bf16_bytes,
        rows: 1,
        values_per_row: row_width,
        backend: CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
    })
}

pub(in crate::commands::real_full) fn validate_residual_add_bf16_inputs(
    residual_bf16: &[u8],
    delta_bf16: &[u8],
) -> Result<()> {
    if residual_bf16.is_empty() {
        anyhow::bail!("real full BF16 residual add requires at least one value");
    }
    if residual_bf16.len() % std::mem::size_of::<u16>() != 0 {
        anyhow::bail!(
            "real full BF16 residual add residual byte length must be even, got {}",
            residual_bf16.len()
        );
    }
    if delta_bf16.len() % std::mem::size_of::<u16>() != 0 {
        anyhow::bail!(
            "real full BF16 residual add delta byte length must be even, got {}",
            delta_bf16.len()
        );
    }
    if residual_bf16.len() != delta_bf16.len() {
        anyhow::bail!(
            "real full BF16 residual add byte length mismatch: residual={} delta={}",
            residual_bf16.len(),
            delta_bf16.len()
        );
    }
    Ok(())
}

pub(in crate::commands::real_full) fn validate_gather_rows_bf16_inputs(
    src_bf16: &[u8],
    row_indices: &[usize],
    src_rows: usize,
    row_width: usize,
) -> Result<Vec<u32>> {
    if row_width == 0 {
        anyhow::bail!("real full BF16 row gather requires non-zero row_width");
    }
    if src_rows == 0 && !row_indices.is_empty() {
        anyhow::bail!("real full BF16 row gather cannot gather from zero source rows");
    }
    let expected_src_bytes = src_rows
        .checked_mul(row_width)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("real full BF16 row gather source shape overflows usize")?;
    if src_bf16.len() != expected_src_bytes {
        anyhow::bail!(
            "real full BF16 row gather source byte length mismatch: expected {} got {}",
            expected_src_bytes,
            src_bf16.len()
        );
    }
    row_indices
        .iter()
        .map(|row_index| {
            if *row_index >= src_rows {
                anyhow::bail!(
                    "real full BF16 row gather index {} out of bounds for {} source rows",
                    row_index,
                    src_rows
                );
            }
            u32::try_from(*row_index).with_context(|| {
                format!("real full BF16 row gather index {row_index} does not fit CUDA u32")
            })
        })
        .collect()
}

pub(in crate::commands::real_full) fn validate_scatter_add_rows_bf16_to_f32_inputs(
    src_bf16: &[u8],
    row_indices: &[usize],
    dst_rows: usize,
    row_width: usize,
    initial: Option<&[f32]>,
) -> Result<Vec<u32>> {
    if row_width == 0 {
        anyhow::bail!("real full BF16-to-f32 row scatter-add requires non-zero row_width");
    }
    if dst_rows == 0 && !row_indices.is_empty() {
        anyhow::bail!("real full BF16-to-f32 row scatter-add cannot target zero destination rows");
    }
    let expected_src_bytes = row_indices
        .len()
        .checked_mul(row_width)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("real full BF16-to-f32 row scatter-add source shape overflows usize")?;
    if src_bf16.len() != expected_src_bytes {
        anyhow::bail!(
            "real full BF16-to-f32 row scatter-add source byte length mismatch: expected {} got {}",
            expected_src_bytes,
            src_bf16.len()
        );
    }
    let expected_dst_values = dst_rows
        .checked_mul(row_width)
        .context("real full BF16-to-f32 row scatter-add destination shape overflows usize")?;
    if let Some(initial) = initial {
        if initial.len() != expected_dst_values {
            anyhow::bail!(
                "real full BF16-to-f32 row scatter-add initial value length mismatch: expected {} got {}",
                expected_dst_values,
                initial.len()
            );
        }
    }
    row_indices
        .iter()
        .map(|row_index| {
            if *row_index >= dst_rows {
                anyhow::bail!(
                    "real full BF16-to-f32 row scatter-add index {} out of bounds for {} destination rows",
                    row_index,
                    dst_rows
                );
            }
            u32::try_from(*row_index).with_context(|| {
                format!(
                    "real full BF16-to-f32 row scatter-add index {row_index} does not fit CUDA u32"
                )
            })
        })
        .collect()
}

pub(in crate::commands::real_full) fn flatten_sparse_b_bf16_partials(
    partial_outputs_bf16_by_host: &[impl AsRef<[u8]>],
    global_row_indices_by_host: &[impl AsRef<[usize]>],
    dst_rows: usize,
    row_width: usize,
) -> Result<(Vec<u8>, Vec<usize>)> {
    if partial_outputs_bf16_by_host.len() != global_row_indices_by_host.len() {
        anyhow::bail!(
            "Sparse-B BF16 partial host count mismatch: payloads={} row_maps={}",
            partial_outputs_bf16_by_host.len(),
            global_row_indices_by_host.len()
        );
    }
    let row_bytes = row_width
        .checked_mul(std::mem::size_of::<u16>())
        .context("Sparse-B BF16 partial row byte count overflows usize")?;
    let total_rows = global_row_indices_by_host
        .iter()
        .map(|row_indices| row_indices.as_ref().len())
        .sum::<usize>();
    let mut src_bf16 = Vec::with_capacity(total_rows * row_bytes);
    let mut row_indices = Vec::with_capacity(total_rows);
    for (host_index, (payload, row_indices_for_host)) in partial_outputs_bf16_by_host
        .iter()
        .zip(global_row_indices_by_host.iter())
        .enumerate()
    {
        let payload = payload.as_ref();
        let row_indices_for_host = row_indices_for_host.as_ref();
        let expected_bytes = row_indices_for_host
            .len()
            .checked_mul(row_bytes)
            .context("Sparse-B BF16 partial host byte count overflows usize")?;
        if payload.len() != expected_bytes {
            anyhow::bail!(
                "Sparse-B BF16 partial payload for host index {host_index} expected {expected_bytes} bytes, got {}",
                payload.len()
            );
        }
        for row_index in row_indices_for_host {
            if *row_index >= dst_rows {
                anyhow::bail!(
                    "Sparse-B BF16 partial row index {row_index} out of bounds for {dst_rows} destination rows"
                );
            }
        }
        src_bf16.extend_from_slice(payload);
        row_indices.extend_from_slice(row_indices_for_host);
    }
    Ok((src_bf16, row_indices))
}

pub(in crate::commands::real_full) fn validate_sparse_b_bf16_partial_layout(
    partial_outputs_bf16_by_host: &[impl AsRef<[u8]>],
    global_row_indices_by_host: &[impl AsRef<[usize]>],
    dst_rows: usize,
    row_width: usize,
) -> Result<SparseBBf16PartialLayout> {
    if partial_outputs_bf16_by_host.len() != global_row_indices_by_host.len() {
        anyhow::bail!(
            "Sparse-B BF16 partial host count mismatch: payloads={} row_maps={}",
            partial_outputs_bf16_by_host.len(),
            global_row_indices_by_host.len()
        );
    }
    let row_bytes = row_width
        .checked_mul(std::mem::size_of::<u16>())
        .context("Sparse-B BF16 partial row byte count overflows usize")?;
    let total_rows = global_row_indices_by_host
        .iter()
        .map(|row_indices| row_indices.as_ref().len())
        .sum::<usize>();
    let src_bytes = total_rows
        .checked_mul(row_bytes)
        .context("Sparse-B BF16 partial total byte count overflows usize")?;
    let mut row_indices = Vec::with_capacity(total_rows);
    for (host_index, (payload, row_indices_for_host)) in partial_outputs_bf16_by_host
        .iter()
        .zip(global_row_indices_by_host.iter())
        .enumerate()
    {
        let payload = payload.as_ref();
        let row_indices_for_host = row_indices_for_host.as_ref();
        let expected_bytes = row_indices_for_host
            .len()
            .checked_mul(row_bytes)
            .context("Sparse-B BF16 partial host byte count overflows usize")?;
        if payload.len() != expected_bytes {
            anyhow::bail!(
                "Sparse-B BF16 partial payload for host index {host_index} expected {expected_bytes} bytes, got {}",
                payload.len()
            );
        }
        for row_index in row_indices_for_host {
            if *row_index >= dst_rows {
                anyhow::bail!(
                    "Sparse-B BF16 partial row index {row_index} out of bounds for {dst_rows} destination rows"
                );
            }
            row_indices.push(u32::try_from(*row_index).with_context(|| {
                format!("Sparse-B BF16 partial row index {row_index} does not fit CUDA u32")
            })?);
        }
    }
    Ok(SparseBBf16PartialLayout {
        row_indices,
        src_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_sparse_b_low_precision_partial_layout(
    partial_outputs: &[impl AsRef<[u8]>],
    global_row_indices: &[impl AsRef<[usize]>],
    output_dtype: ExpertV2Dtype,
    output_row_stride_bytes: usize,
    dst_rows: usize,
    row_width: usize,
) -> Result<SparseBLowPrecisionPartialLayout> {
    anyhow::ensure!(
        partial_outputs.len() == global_row_indices.len(),
        "low-precision Sparse-B partial host count mismatch: payloads={} row_maps={}",
        partial_outputs.len(),
        global_row_indices.len()
    );
    let logical_row_bytes = output_dtype.row_bytes(row_width)?;
    anyhow::ensure!(
        output_row_stride_bytes == logical_row_bytes,
        "low-precision Sparse-B row stride {output_row_stride_bytes} did not match logical {logical_row_bytes} for {output_dtype:?}"
    );
    let total_rows = global_row_indices
        .iter()
        .map(|row_indices| row_indices.as_ref().len())
        .sum::<usize>();
    anyhow::ensure!(
        total_rows <= dst_rows.saturating_mul(GLM52_TOP_K),
        "low-precision Sparse-B partial rows {total_rows} exceed routed contribution capacity {}",
        dst_rows.saturating_mul(GLM52_TOP_K)
    );
    let src_bytes = total_rows
        .checked_mul(output_row_stride_bytes)
        .context("low-precision Sparse-B partial byte count overflows usize")?;
    let mut row_indices = Vec::with_capacity(total_rows);
    let mut row_counts = Vec::with_capacity(global_row_indices.len());
    for (host_index, (payload, rows)) in partial_outputs
        .iter()
        .zip(global_row_indices.iter())
        .enumerate()
    {
        let rows = rows.as_ref();
        row_counts.push(rows.len());
        let expected_bytes = rows
            .len()
            .checked_mul(output_row_stride_bytes)
            .context("low-precision Sparse-B host byte count overflows usize")?;
        anyhow::ensure!(
            payload.as_ref().len() == expected_bytes,
            "low-precision Sparse-B payload for host index {host_index} expected {expected_bytes} bytes, got {}",
            payload.as_ref().len()
        );
        for row_index in rows {
            anyhow::ensure!(
                *row_index < dst_rows,
                "low-precision Sparse-B row index {row_index} out of bounds for {dst_rows} destination rows"
            );
            row_indices.push(u32::try_from(*row_index).with_context(|| {
                format!("low-precision Sparse-B row index {row_index} does not fit CUDA u32")
            })?);
        }
    }
    Ok(SparseBLowPrecisionPartialLayout {
        row_indices,
        row_counts,
        src_bytes,
    })
}

pub(in crate::commands::real_full) fn validate_sparse_b_scatter_residual_add_inputs(
    residual: &[f32],
    initial_delta: &[f32],
    src_bf16: &[u8],
    row_indices: &[usize],
    dst_rows: usize,
    row_width: usize,
) -> Result<Vec<u32>> {
    if dst_rows == 0 || row_width == 0 {
        anyhow::bail!(
            "Sparse-B scatter residual-add requires non-zero shape, got rows={dst_rows} row_width={row_width}"
        );
    }
    if row_indices.is_empty() {
        anyhow::bail!("Sparse-B scatter residual-add requires at least one partial row");
    }
    let expected_values = dst_rows
        .checked_mul(row_width)
        .context("Sparse-B scatter residual-add value count overflows usize")?;
    if residual.len() != expected_values {
        anyhow::bail!(
            "Sparse-B scatter residual-add residual length mismatch: expected {expected_values} got {}",
            residual.len()
        );
    }
    if initial_delta.len() != expected_values {
        anyhow::bail!(
            "Sparse-B scatter residual-add initial delta length mismatch: expected {expected_values} got {}",
            initial_delta.len()
        );
    }
    validate_scatter_add_rows_bf16_to_f32_inputs(
        src_bf16,
        row_indices,
        dst_rows,
        row_width,
        Some(initial_delta),
    )
}

pub(in crate::commands::real_full) fn validate_sparse_b_device_residual_inputs(
    residual: &DeviceBf16Output,
    shared_delta: &DeviceBf16Output,
    dst_rows: usize,
    row_width: usize,
) -> Result<()> {
    if dst_rows == 0 || row_width == 0 {
        anyhow::bail!(
            "Sparse-B device scatter residual-add requires non-zero shape, got rows={dst_rows} row_width={row_width}"
        );
    }
    if residual.rows != dst_rows || residual.values_per_row != row_width {
        anyhow::bail!(
            "Sparse-B device scatter residual-add residual shape mismatch: expected {dst_rows}x{row_width}, got {}x{}",
            residual.rows,
            residual.values_per_row
        );
    }
    if shared_delta.rows != dst_rows || shared_delta.values_per_row != row_width {
        anyhow::bail!(
            "Sparse-B device scatter residual-add shared delta shape mismatch: expected {dst_rows}x{row_width}, got {}x{}",
            shared_delta.rows,
            shared_delta.values_per_row
        );
    }
    let expected_bytes = dst_rows
        .checked_mul(row_width)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("Sparse-B device scatter residual-add byte count overflows usize")?;
    if residual.bytes != expected_bytes {
        anyhow::bail!(
            "Sparse-B device scatter residual-add residual bytes mismatch: expected {expected_bytes} got {}",
            residual.bytes
        );
    }
    if shared_delta.bytes != expected_bytes {
        anyhow::bail!(
            "Sparse-B device scatter residual-add shared delta bytes mismatch: expected {expected_bytes} got {}",
            shared_delta.bytes
        );
    }
    let residual_buffer = residual.buffer();
    let shared_buffer = shared_delta.buffer();
    if residual_buffer.device_id != shared_buffer.device_id {
        anyhow::bail!(
            "Sparse-B device scatter residual-add device mismatch: residual={} shared={}",
            residual_buffer.device_id,
            shared_buffer.device_id
        );
    }
    Ok(())
}

pub(in crate::commands::real_full) fn cpu_residual_add_prefix_bf16_bytes_into(
    residual_bf16: &[u8],
    delta_bf16: &[u8],
    output_bf16: &mut [u8],
) -> &'static str {
    for idx in 0..residual_bf16.len() / std::mem::size_of::<u16>() {
        let value = bf16_value(residual_bf16, idx) + bf16_value(delta_bf16, idx);
        let bits = (value.to_bits() >> 16) as u16;
        let byte_index = idx * std::mem::size_of::<u16>();
        output_bf16[byte_index..byte_index + std::mem::size_of::<u16>()]
            .copy_from_slice(&bits.to_le_bytes());
    }
    CPU_REFERENCE_RESIDUAL_ADD_BF16_BACKEND
}

pub(in crate::commands::real_full) fn cpu_gather_rows_bf16(
    src_bf16: &[u8],
    row_indices: &[u32],
    _src_rows: usize,
    row_width: usize,
) -> GatherRowsBf16Output {
    let row_bytes = row_width * std::mem::size_of::<u16>();
    let mut bytes = Vec::with_capacity(row_indices.len() * row_bytes);
    for row_index in row_indices {
        let start = *row_index as usize * row_bytes;
        bytes.extend_from_slice(&src_bf16[start..start + row_bytes]);
    }
    GatherRowsBf16Output {
        bytes,
        backend: CPU_REFERENCE_GATHER_ROWS_BF16_BACKEND,
    }
}

pub(in crate::commands::real_full) fn cpu_scatter_add_rows_bf16_to_f32(
    src_bf16: &[u8],
    row_indices: &[u32],
    dst_rows: usize,
    row_width: usize,
    initial: Option<&[f32]>,
) -> ScatterAddRowsBf16ToF32Output {
    let mut values = initial
        .map(|values| values.to_vec())
        .unwrap_or_else(|| vec![0.0_f32; dst_rows * row_width]);
    for (src_row, row_index) in row_indices.iter().copied().enumerate() {
        let src_start = src_row * row_width;
        let dst_start = row_index as usize * row_width;
        for col in 0..row_width {
            values[dst_start + col] += bf16_value(src_bf16, src_start + col);
        }
    }
    ScatterAddRowsBf16ToF32Output {
        values,
        backend: CPU_REFERENCE_SCATTER_ADD_ROWS_BF16_TO_F32_BACKEND,
    }
}

pub(in crate::commands::real_full) fn cpu_sparse_b_scatter_residual_add_bf16(
    residual: &[f32],
    initial_delta: &[f32],
    src_bf16: &[u8],
    row_indices: &[u32],
    dst_rows: usize,
    row_width: usize,
) -> Result<SparseBScatterResidualAddOutput> {
    let scatter = cpu_scatter_add_rows_bf16_to_f32(
        src_bf16,
        row_indices,
        dst_rows,
        row_width,
        Some(initial_delta),
    );
    let residual_bf16 = f32_values_to_bf16_bytes(residual);
    let delta_bf16 = f32_values_to_bf16_bytes(&scatter.values);
    let mut output_bf16 = vec![0_u8; residual_bf16.len()];
    let backend =
        cpu_residual_add_prefix_bf16_bytes_into(&residual_bf16, &delta_bf16, &mut output_bf16);
    Ok(SparseBScatterResidualAddOutput {
        values: bf16_values_to_f32(&output_bf16),
        delta_values: scatter.values,
        output_bf16,
        device_output: None,
        backend,
    })
}

pub(in crate::commands::real_full) fn cuda_residual_add_bf16_device_inputs_output(
    residual: &DeviceBf16Output,
    delta: &DeviceBf16Output,
    bytes: usize,
) -> Result<ResidualAddDeviceOutput> {
    if let Some(graph_key) = coord_sparse_b_graph_key_for_bf16_full_rows(bytes)? {
        match cuda_residual_add_bf16_device_inputs_output_graph_slot(
            &graph_key, residual, delta, bytes,
        ) {
            Ok(output) => return Ok(output),
            Err(_error) => {}
        }
    }
    cuda_residual_add_bf16_device_inputs_output_legacy(residual, delta, bytes)
}

pub(in crate::commands::real_full) fn cuda_residual_add_bf16_device_inputs_device_output(
    residual: &DeviceBf16Output,
    delta: &DeviceBf16Output,
    bytes: usize,
) -> Result<DeviceBf16Output> {
    if let Some(graph_key) = coord_sparse_b_graph_key_for_bf16_full_rows(bytes)? {
        match cuda_residual_add_bf16_device_inputs_device_output_graph_slot(
            &graph_key, residual, delta, bytes,
        ) {
            Ok(output) => return Ok(output),
            Err(_error) => {}
        }
    }
    cuda_residual_add_bf16_device_inputs_device_output_legacy(residual, delta, bytes)
}

pub(in crate::commands::real_full) fn cuda_residual_add_bf16_device_inputs_output_graph_slot(
    graph_key: &CoordinatorGraphKey,
    residual: &DeviceBf16Output,
    delta: &DeviceBf16Output,
    bytes: usize,
) -> Result<ResidualAddDeviceOutput> {
    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let output_buffer = OwnedCoordinatorDeviceBuffer::new(
            library,
            bytes,
            "BF16 device-input residual-add output",
        )?;
        let signature = CoordinatorCudaGraphSignature::residual_add_bf16(bytes);
        let cuda_stream = slot.stream_ptr();
        residual
            .wait_ready_on_stream(cuda_stream)
            .context("waiting for BF16 residual input before residual add")?;
        delta
            .wait_ready_on_stream(cuda_stream)
            .context("waiting for BF16 delta input before residual add")?;
        capture_or_update_sparse_b_residual_add_bf16_graph(
            library,
            slot,
            signature,
            residual.buffer(),
            delta.buffer(),
            output_buffer.buffer,
            bytes / std::mem::size_of::<u16>(),
            "BF16 device-input residual add",
        )?;
        let mut out_bytes = vec![0_u8; bytes];
        unsafe {
            library
                .copy_d2h_async(&mut out_bytes, output_buffer.buffer, cuda_stream)
                .context("async copying BF16 device-input residual add output to host")?;
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing BF16 device-input residual add graph slot stream")?;
        }
        let values = bf16_values_to_f32(&out_bytes);
        Ok(ResidualAddDeviceOutput {
            values,
            output_bf16: out_bytes,
            device_output: DeviceBf16Output {
                buffer: output_buffer,
                bytes,
                rows: residual.rows,
                values_per_row: residual.values_per_row,
                backend: CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
            },
            backend: CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
        })
    })
}

pub(in crate::commands::real_full) fn cuda_residual_add_bf16_device_inputs_device_output_graph_slot(
    graph_key: &CoordinatorGraphKey,
    residual: &DeviceBf16Output,
    delta: &DeviceBf16Output,
    bytes: usize,
) -> Result<DeviceBf16Output> {
    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let output_buffer = OwnedCoordinatorDeviceBuffer::new(
            library,
            bytes,
            "BF16 device-input residual-add device output",
        )?;
        let signature = CoordinatorCudaGraphSignature::residual_add_bf16(bytes);
        let cuda_stream = slot.stream_ptr();
        residual
            .wait_ready_on_stream(cuda_stream)
            .context("waiting for BF16 residual input before device-output residual add")?;
        delta
            .wait_ready_on_stream(cuda_stream)
            .context("waiting for BF16 delta input before device-output residual add")?;
        capture_or_update_sparse_b_residual_add_bf16_graph(
            library,
            slot,
            signature,
            residual.buffer(),
            delta.buffer(),
            output_buffer.buffer,
            bytes / std::mem::size_of::<u16>(),
            "BF16 device-input residual add device output",
        )?;
        unsafe {
            library.cuda_stream_synchronize(cuda_stream).context(
                "synchronizing BF16 device-input residual add device-output graph slot stream",
            )?;
        }
        Ok(DeviceBf16Output {
            buffer: output_buffer,
            bytes,
            rows: residual.rows,
            values_per_row: residual.values_per_row,
            backend: CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
        })
    })
}

pub(in crate::commands::real_full) fn cuda_residual_add_bf16_device_inputs_output_legacy(
    residual: &DeviceBf16Output,
    delta: &DeviceBf16Output,
    bytes: usize,
) -> Result<ResidualAddDeviceOutput> {
    let library = cuda_native_library()?;
    residual
        .synchronize_ready()
        .context("waiting for BF16 residual input before legacy residual add")?;
    delta
        .synchronize_ready()
        .context("waiting for BF16 delta input before legacy residual add")?;
    let output_buffer =
        OwnedCoordinatorDeviceBuffer::new(library, bytes, "BF16 device-input residual-add output")?;
    library
        .cuda_residual_add_bf16(
            residual.buffer(),
            delta.buffer(),
            output_buffer.buffer,
            bytes / std::mem::size_of::<u16>(),
        )
        .context("executing CUDA BF16 device-input residual add")?;
    let mut out_bytes = vec![0_u8; bytes];
    library
        .copy_d2h(&mut out_bytes, output_buffer.buffer)
        .context("copying BF16 device-input residual add output to host")?;
    let values = bf16_values_to_f32(&out_bytes);
    Ok(ResidualAddDeviceOutput {
        values,
        output_bf16: out_bytes,
        device_output: DeviceBf16Output {
            buffer: output_buffer,
            bytes,
            rows: residual.rows,
            values_per_row: residual.values_per_row,
            backend: CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
        },
        backend: CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
    })
}

pub(in crate::commands::real_full) fn cuda_residual_add_bf16_device_inputs_device_output_legacy(
    residual: &DeviceBf16Output,
    delta: &DeviceBf16Output,
    bytes: usize,
) -> Result<DeviceBf16Output> {
    let library = cuda_native_library()?;
    residual
        .synchronize_ready()
        .context("waiting for BF16 residual input before legacy device-output residual add")?;
    delta
        .synchronize_ready()
        .context("waiting for BF16 delta input before legacy device-output residual add")?;
    let output_buffer = OwnedCoordinatorDeviceBuffer::new(
        library,
        bytes,
        "BF16 device-input residual-add device output",
    )?;
    library
        .cuda_residual_add_bf16(
            residual.buffer(),
            delta.buffer(),
            output_buffer.buffer,
            bytes / std::mem::size_of::<u16>(),
        )
        .context("executing CUDA BF16 device-input residual add device output")?;
    Ok(DeviceBf16Output {
        buffer: output_buffer,
        bytes,
        rows: residual.rows,
        values_per_row: residual.values_per_row,
        backend: CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
    })
}

pub(in crate::commands::real_full) fn cuda_residual_add_prefix_bf16_bytes_into(
    residual_bf16: &[u8],
    delta_bf16: &[u8],
    output_bf16: &mut [u8],
) -> Result<&'static str> {
    if let Some(graph_key) = coord_sparse_b_graph_key_for_bf16_full_rows(residual_bf16.len())? {
        match cuda_residual_add_prefix_bf16_bytes_into_graph_slot(
            &graph_key,
            residual_bf16,
            delta_bf16,
            output_bf16,
        ) {
            Ok(backend) => return Ok(backend),
            Err(_error) => {}
        }
    }
    cuda_residual_add_prefix_bf16_bytes_into_legacy(residual_bf16, delta_bf16, output_bf16)
}

pub(in crate::commands::real_full) fn cuda_residual_add_prefix_bf16_bytes_into_graph_slot(
    graph_key: &CoordinatorGraphKey,
    residual_bf16: &[u8],
    delta_bf16: &[u8],
    output_bf16: &mut [u8],
) -> Result<&'static str> {
    let bytes = residual_bf16.len();
    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let signature = CoordinatorCudaGraphSignature::residual_add_bf16(bytes);
        let cuda_stream = slot.stream_ptr();
        let residual_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            bytes,
            "BF16 residual",
        )?;
        let delta_buffer =
            slot.buffer(library, CoordinatorCudaScratchSlot::B, bytes, "BF16 delta")?;
        let output_buffer =
            slot.buffer(library, CoordinatorCudaScratchSlot::C, bytes, "BF16 output")?;

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::A,
                residual_bf16,
                "BF16 residual",
                cuda_stream,
            )
            .context("async copying BF16 residual to device")?;
        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::B,
                delta_bf16,
                "BF16 delta",
                cuda_stream,
            )
            .context("async copying BF16 delta to device")?;
        capture_or_update_sparse_b_residual_add_bf16_graph(
            library,
            slot,
            signature,
            residual_buffer,
            delta_buffer,
            output_buffer,
            bytes / std::mem::size_of::<u16>(),
            "BF16 residual add",
        )?;
        unsafe {
            library
                .copy_d2h_async(output_bf16, output_buffer, cuda_stream)
                .context("async copying BF16 residual add output to host")?;
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing BF16 residual add graph slot stream")?;
        }
        Ok(CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND)
    })
}

pub(in crate::commands::real_full) fn capture_or_update_sparse_b_residual_add_bf16_graph(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    signature: CoordinatorCudaGraphSignature,
    residual_buffer: GlmrtDeviceBuffer,
    delta_buffer: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
    count: usize,
    label: &'static str,
) -> Result<()> {
    if !sparse_b_residual_add_graph_replay_enabled(signature) {
        unsafe {
            library
                .cuda_residual_add_bf16_async(
                    residual_buffer,
                    delta_buffer,
                    output_buffer,
                    count,
                    slot.stream_ptr(),
                )
                .with_context(|| format!("executing async CUDA {label}"))?;
        }
        return Ok(());
    }
    if !slot.has_captured_graph(
        CoordinatorCudaGraphProgram::SparseBResidualAddBf16,
        signature,
    ) {
        slot.stream_synchronize()
            .with_context(|| format!("synchronizing {label} inputs before graph capture"))?;
        slot.capture_graph(
            library,
            CoordinatorCudaGraphProgram::SparseBResidualAddBf16,
            signature,
            |library, cuda_stream, _workspace| unsafe {
                library
                    .cuda_residual_add_bf16_async(
                        residual_buffer,
                        delta_buffer,
                        output_buffer,
                        count,
                        cuda_stream,
                    )
                    .with_context(|| format!("capturing async CUDA {label}"))?;
                Ok(())
            },
        )?;
    } else {
        let (graph_raw, exec_raw) = slot
            .captured_graph_raw_handles(
                CoordinatorCudaGraphProgram::SparseBResidualAddBf16,
                signature,
            )
            .context(
                "coordinator CUDA graph slot lost captured residual-add graph before update",
            )?;
        unsafe {
            library
                .cuda_graph_update_residual_add_bf16_node(
                    graph_raw,
                    exec_raw,
                    0,
                    residual_buffer,
                    delta_buffer,
                    output_buffer,
                    count,
                )
                .with_context(|| format!("updating captured CUDA {label} graph node"))?;
        }
    }
    slot.launch_captured_graph(
        library,
        CoordinatorCudaGraphProgram::SparseBResidualAddBf16,
        signature,
    )
}

fn sparse_b_residual_add_graph_replay_enabled(signature: CoordinatorCudaGraphSignature) -> bool {
    signature.rows <= 16
}

pub(in crate::commands::real_full) fn cuda_residual_add_prefix_bf16_bytes_into_legacy(
    residual_bf16: &[u8],
    delta_bf16: &[u8],
    output_bf16: &mut [u8],
) -> Result<&'static str> {
    let library = cuda_native_library()?;
    let bytes = residual_bf16.len();
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let residual_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        bytes,
        "BF16 residual",
    )?;
    let delta_buffer =
        workspace.buffer(library, CoordinatorCudaScratchSlot::B, bytes, "BF16 delta")?;
    let output_buffer =
        workspace.buffer(library, CoordinatorCudaScratchSlot::C, bytes, "BF16 output")?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            residual_bf16,
            "BF16 residual",
        )
        .context("copying BF16 residual to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::B,
            delta_bf16,
            "BF16 delta",
        )
        .context("copying BF16 delta to device")?;
    library
        .cuda_residual_add_bf16(
            residual_buffer,
            delta_buffer,
            output_buffer,
            bytes / std::mem::size_of::<u16>(),
        )
        .context("executing CUDA BF16 residual add")?;
    library
        .copy_d2h(output_bf16, output_buffer)
        .context("copying BF16 residual add output to host")?;

    Ok(CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND)
}

pub(in crate::commands::real_full) fn cuda_gather_rows_bf16(
    src_bf16: &[u8],
    row_indices: &[u32],
    src_rows: usize,
    row_width: usize,
) -> Result<GatherRowsBf16Output> {
    let library = cuda_native_library()?;
    let index_bytes = std::mem::size_of_val(row_indices);
    let output_bytes = row_indices
        .len()
        .checked_mul(row_width)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 row gather output shape overflows usize")?;
    if row_indices.is_empty() {
        return Ok(GatherRowsBf16Output {
            bytes: Vec::new(),
            backend: CUDA_REFERENCE_GATHER_ROWS_BF16_BACKEND,
        });
    }
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let src_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        src_bf16.len(),
        "BF16 row gather src",
    )?;
    let index_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::B,
        index_bytes,
        "BF16 row gather indices",
    )?;
    let output_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        output_bytes,
        "BF16 row gather output",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            src_bf16,
            "BF16 row gather src",
        )
        .context("copying BF16 row gather src to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::B,
            u32_bytes(row_indices),
            "BF16 row gather indices",
        )
        .context("copying BF16 row gather indices to device")?;
    library
        .cuda_gather_rows_bf16(
            src_buffer,
            src_rows,
            index_buffer,
            output_buffer,
            row_indices.len(),
            row_width,
        )
        .context("executing CUDA BF16 row gather")?;
    let mut out_bytes = vec![0_u8; output_bytes];
    library
        .copy_d2h(&mut out_bytes, output_buffer)
        .context("copying BF16 row gather output to host")?;

    Ok(GatherRowsBf16Output {
        bytes: out_bytes,
        backend: CUDA_REFERENCE_GATHER_ROWS_BF16_BACKEND,
    })
}

pub(in crate::commands::real_full) fn cuda_scatter_add_rows_bf16_to_f32(
    src_bf16: &[u8],
    row_indices: &[u32],
    dst_rows: usize,
    row_width: usize,
    initial: Option<&[f32]>,
) -> Result<ScatterAddRowsBf16ToF32Output> {
    let dst_values = dst_rows
        .checked_mul(row_width)
        .context("CUDA BF16-to-f32 row scatter-add destination shape overflows usize")?;
    let initial_values = initial
        .map(|values| values.to_vec())
        .unwrap_or_else(|| vec![0.0_f32; dst_values]);
    let index_bytes = std::mem::size_of_val(row_indices);
    let dst_bytes = std::mem::size_of_val(initial_values.as_slice());
    if row_indices.is_empty() {
        return Ok(ScatterAddRowsBf16ToF32Output {
            values: initial_values,
            backend: CUDA_REFERENCE_SCATTER_ADD_ROWS_BF16_TO_F32_BACKEND,
        });
    }
    let graph_mode = if dst_rows == 1 {
        LayerWaveMode::Decode
    } else {
        LayerWaveMode::Prefill
    };
    let graph_key =
        CoordinatorGraphKey::glm52_bf16(CoordinatorGraphShape::CoordSparseB, graph_mode, dst_rows)
            .context("selecting Coord-Sparse-B graph slot for BF16-to-f32 row scatter-add")?;
    let signature = CoordinatorCudaGraphSignature::scatter_add_bf16_to_f32(
        dst_rows,
        row_width,
        row_indices.len(),
    );
    let out_bytes = with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let src_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            src_bf16.len(),
            "BF16-to-f32 row scatter-add src",
        )?;
        let index_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::B,
            index_bytes,
            "BF16-to-f32 row scatter-add indices",
        )?;
        let dst_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            dst_bytes,
            "BF16-to-f32 row scatter-add dst",
        )?;

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::A,
                src_bf16,
                "BF16-to-f32 row scatter-add src",
                cuda_stream,
            )
            .context("async copying BF16-to-f32 row scatter-add src to device")?;
        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::B,
                u32_bytes(row_indices),
                "BF16-to-f32 row scatter-add indices",
                cuda_stream,
            )
            .context("async copying BF16-to-f32 row scatter-add indices to device")?;
        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::C,
                f32_bytes(&initial_values),
                "BF16-to-f32 row scatter-add dst",
                cuda_stream,
            )
            .context("async copying BF16-to-f32 row scatter-add initial dst to device")?;
        capture_or_update_sparse_b_scatter_add_bf16_to_f32_graph(
            library,
            slot,
            signature,
            src_buffer,
            index_buffer,
            dst_buffer,
            dst_rows,
            row_indices.len(),
            row_width,
            "BF16-to-f32 row scatter-add",
        )?;
        let mut out_bytes = vec![0_u8; dst_bytes];
        unsafe {
            library
                .copy_d2h_async(&mut out_bytes, dst_buffer, cuda_stream)
                .context("async copying BF16-to-f32 row scatter-add output to host")?;
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing BF16-to-f32 row scatter-add graph slot stream")?;
        }
        Ok(out_bytes)
    })?;

    Ok(ScatterAddRowsBf16ToF32Output {
        values: f32_vec_from_bytes(&out_bytes)?,
        backend: CUDA_REFERENCE_SCATTER_ADD_ROWS_BF16_TO_F32_BACKEND,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn capture_or_update_sparse_b_scatter_add_bf16_to_f32_graph(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    signature: CoordinatorCudaGraphSignature,
    src_buffer: GlmrtDeviceBuffer,
    index_buffer: GlmrtDeviceBuffer,
    dst_buffer: GlmrtDeviceBuffer,
    dst_rows: usize,
    rows: usize,
    row_width: usize,
    label: &'static str,
) -> Result<()> {
    if SPARSE_B_SCATTER_GRAPH_DISABLED.load(Ordering::Relaxed) {
        return launch_sparse_b_scatter_add_bf16_to_f32_eager(
            library,
            src_buffer,
            index_buffer,
            dst_buffer,
            dst_rows,
            rows,
            row_width,
            slot.stream_ptr(),
            label,
        );
    }
    let graph_result = (|| -> Result<()> {
        if !slot.has_captured_graph(
            CoordinatorCudaGraphProgram::SparseBScatterAddBf16ToF32,
            signature,
        ) {
            slot.stream_synchronize()
                .with_context(|| format!("synchronizing {label} inputs before graph capture"))?;
            slot.capture_graph(
                library,
                CoordinatorCudaGraphProgram::SparseBScatterAddBf16ToF32,
                signature,
                |library, cuda_stream, _workspace| unsafe {
                    library
                        .cuda_scatter_add_rows_bf16_to_f32_async(
                            src_buffer,
                            index_buffer,
                            dst_buffer,
                            dst_rows,
                            rows,
                            row_width,
                            cuda_stream,
                        )
                        .with_context(|| format!("capturing async CUDA {label}"))?;
                    Ok(())
                },
            )?;
        } else {
            let (graph_raw, exec_raw) = slot
                .captured_graph_raw_handles(
                    CoordinatorCudaGraphProgram::SparseBScatterAddBf16ToF32,
                    signature,
                )
                .context(
                    "coordinator CUDA graph slot lost captured Sparse-B scatter-add graph before update",
                )?;
            unsafe {
                library
                    .cuda_graph_update_scatter_add_rows_bf16_to_f32_node(
                        graph_raw,
                        exec_raw,
                        0,
                        src_buffer,
                        index_buffer,
                        dst_buffer,
                        dst_rows,
                        rows,
                        row_width,
                    )
                    .with_context(|| format!("updating captured CUDA {label} graph node"))?;
            }
        }
        slot.launch_captured_graph(
            library,
            CoordinatorCudaGraphProgram::SparseBScatterAddBf16ToF32,
            signature,
        )
    })();
    match graph_result {
        Ok(()) => Ok(()),
        Err(_error) => {
            SPARSE_B_SCATTER_GRAPH_DISABLED.store(true, Ordering::Relaxed);
            launch_sparse_b_scatter_add_bf16_to_f32_eager(
                library,
                src_buffer,
                index_buffer,
                dst_buffer,
                dst_rows,
                rows,
                row_width,
                slot.stream_ptr(),
                label,
            )
            .with_context(|| format!("falling back to eager CUDA {label} after graph failure"))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_sparse_b_scatter_add_bf16_to_f32_eager(
    library: &'static NativeLibrary,
    src_buffer: GlmrtDeviceBuffer,
    index_buffer: GlmrtDeviceBuffer,
    dst_buffer: GlmrtDeviceBuffer,
    dst_rows: usize,
    rows: usize,
    row_width: usize,
    cuda_stream: *mut c_void,
    label: &'static str,
) -> Result<()> {
    unsafe {
        library
            .cuda_scatter_add_rows_bf16_to_f32_async(
                src_buffer,
                index_buffer,
                dst_buffer,
                dst_rows,
                rows,
                row_width,
                cuda_stream,
            )
            .with_context(|| format!("executing eager CUDA {label}"))
    }
}

pub(in crate::commands::real_full) fn cuda_sparse_b_scatter_residual_add_bf16(
    residual: &[f32],
    initial_delta: &[f32],
    src_bf16: &[u8],
    row_indices: &[u32],
    dst_rows: usize,
    row_width: usize,
) -> Result<SparseBScatterResidualAddOutput> {
    let values = dst_rows
        .checked_mul(row_width)
        .context("CUDA Sparse-B scatter residual-add destination shape overflows usize")?;
    let dst_f32_bytes = std::mem::size_of_val(initial_delta);
    let bf16_bytes = values
        .checked_mul(std::mem::size_of::<u16>())
        .context("CUDA Sparse-B scatter residual-add BF16 byte count overflows usize")?;
    let residual_bf16 = f32_values_to_bf16_bytes(residual);
    let graph_mode = if dst_rows == 1 {
        LayerWaveMode::Decode
    } else {
        LayerWaveMode::Prefill
    };
    let graph_key =
        CoordinatorGraphKey::glm52_bf16(CoordinatorGraphShape::CoordSparseB, graph_mode, dst_rows)
            .context("selecting Coord-Sparse-B graph slot for Sparse-B scatter residual-add")?;
    let row_capacity = graph_key.row_bucket.row_capacity;
    let partial_row_capacity = row_capacity
        .checked_mul(GLM52_TOP_K)
        .context("CUDA Sparse-B scatter residual-add partial row capacity overflows usize")?;
    if row_indices.len() > partial_row_capacity {
        anyhow::bail!(
            "CUDA Sparse-B scatter residual-add partial rows {} exceed graph bucket capacity {}",
            row_indices.len(),
            partial_row_capacity
        );
    }
    let src_capacity_bytes = partial_row_capacity
        .checked_mul(row_width)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA Sparse-B scatter residual-add partial buffer capacity overflows usize")?;
    let index_capacity_bytes = partial_row_capacity
        .checked_mul(std::mem::size_of::<u32>())
        .context("CUDA Sparse-B scatter residual-add index buffer capacity overflows usize")?;
    let bucket_values = row_capacity
        .checked_mul(row_width)
        .context("CUDA Sparse-B scatter residual-add row bucket shape overflows usize")?;
    let bucket_f32_bytes = bucket_values
        .checked_mul(std::mem::size_of::<f32>())
        .context("CUDA Sparse-B scatter residual-add f32 bucket byte count overflows usize")?;
    let bucket_bf16_bytes = bucket_values
        .checked_mul(std::mem::size_of::<u16>())
        .context("CUDA Sparse-B scatter residual-add BF16 bucket byte count overflows usize")?;
    let signature =
        CoordinatorCudaGraphSignature::coord_sparse_b_envelope_bf16(row_capacity, row_width);
    let (delta_bytes, output_bf16, device_output_buffer) =
        with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
            let cuda_stream = slot.stream_ptr();
            let src_buffer = slot.buffer(
                library,
                CoordinatorCudaScratchSlot::A,
                src_capacity_bytes,
                "Sparse-B scatter residual-add BF16 partials",
            )?;
            let index_buffer = slot.buffer(
                library,
                CoordinatorCudaScratchSlot::B,
                index_capacity_bytes,
                "Sparse-B scatter residual-add row indices",
            )?;
            let dst_f32_buffer = slot.buffer(
                library,
                CoordinatorCudaScratchSlot::C,
                bucket_f32_bytes,
                "Sparse-B scatter residual-add f32 accumulator",
            )?;
            let residual_bf16_buffer = slot.buffer(
                library,
                CoordinatorCudaScratchSlot::E,
                bucket_bf16_bytes,
                "Sparse-B scatter residual-add BF16 residual",
            )?;
            let output_bf16_buffer = slot.buffer(
                library,
                CoordinatorCudaScratchSlot::F,
                bucket_bf16_bytes,
                "Sparse-B scatter residual-add BF16 output",
            )?;
            let device_output_buffer = OwnedCoordinatorDeviceBuffer::new(
                library,
                bf16_bytes,
                "Sparse-B scatter residual-add owned BF16 output",
            )?;

            slot.workspace
                .copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::A,
                    src_bf16,
                    "Sparse-B scatter residual-add BF16 partials",
                    cuda_stream,
                )
                .context("async copying Sparse-B BF16 partials to device")?;
            slot.workspace
                .copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::B,
                    u32_bytes(row_indices),
                    "Sparse-B scatter residual-add row indices",
                    cuda_stream,
                )
                .context("async copying Sparse-B row indices to device")?;
            slot.workspace
                .copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::C,
                    f32_bytes(initial_delta),
                    "Sparse-B scatter residual-add initial accumulator",
                    cuda_stream,
                )
                .context("async copying Sparse-B initial accumulator to device")?;
            slot.workspace
                .copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::E,
                    &residual_bf16,
                    "Sparse-B scatter residual-add residual",
                    cuda_stream,
                )
                .context("async copying Sparse-B BF16 residual to device")?;
            capture_or_update_sparse_b_scatter_residual_add_bf16_graph(
                library,
                slot,
                signature,
                src_buffer,
                index_buffer,
                dst_f32_buffer,
                residual_bf16_buffer,
                output_bf16_buffer,
                dst_rows,
                row_indices.len(),
                row_width,
                values,
            )?;
            let mut delta_bytes = vec![0_u8; dst_f32_bytes];
            let mut output_bf16 = vec![0_u8; bf16_bytes];
            unsafe {
                library
                    .copy_d2d_async(
                        device_output_buffer.buffer,
                        output_bf16_buffer,
                        bf16_bytes,
                        cuda_stream,
                    )
                    .context("async cloning Sparse-B residual output to owned device buffer")?;
                library
                    .copy_d2h_async(&mut delta_bytes, dst_f32_buffer, cuda_stream)
                    .context("async copying Sparse-B scatter residual-add delta to host")?;
                library
                    .copy_d2h_async(&mut output_bf16, output_bf16_buffer, cuda_stream)
                    .context("async copying Sparse-B scatter residual-add output to host")?;
                library
                    .cuda_stream_synchronize(cuda_stream)
                    .context("synchronizing Sparse-B scatter residual-add graph slot stream")?;
            }
            Ok((delta_bytes, output_bf16, device_output_buffer))
        })?;

    Ok(SparseBScatterResidualAddOutput {
        values: bf16_values_to_f32(&output_bf16),
        delta_values: f32_vec_from_bytes(&delta_bytes)?,
        output_bf16,
        device_output: Some(DeviceBf16Output {
            buffer: device_output_buffer,
            bytes: bf16_bytes,
            rows: dst_rows,
            values_per_row: row_width,
            backend: CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
        }),
        backend: CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
    })
}

pub(in crate::commands::real_full) fn cuda_sparse_b_scatter_shared_residual_add_bf16_device_output(
    residual: &DeviceBf16Output,
    shared_delta: &DeviceBf16Output,
    partial_outputs_bf16_by_host: &[impl AsRef<[u8]>],
    partials: &SparseBBf16PartialLayout,
    dst_rows: usize,
    row_width: usize,
) -> Result<DeviceBf16Output> {
    let values = dst_rows
        .checked_mul(row_width)
        .context("CUDA Sparse-B fused device residual-add destination shape overflows usize")?;
    let bf16_bytes = values
        .checked_mul(std::mem::size_of::<u16>())
        .context("CUDA Sparse-B fused device residual-add BF16 byte count overflows usize")?;
    let graph_mode = if dst_rows == 1 {
        LayerWaveMode::Decode
    } else {
        LayerWaveMode::Prefill
    };
    let graph_key =
        CoordinatorGraphKey::glm52_bf16(CoordinatorGraphShape::CoordSparseB, graph_mode, dst_rows)
            .context(
                "selecting Coord-Sparse-B graph slot for fused Sparse-B device residual-add",
            )?;
    let row_capacity = graph_key.row_bucket.row_capacity;
    let partial_row_capacity = row_capacity
        .checked_mul(GLM52_TOP_K)
        .context("CUDA Sparse-B fused device residual-add partial row capacity overflows usize")?;
    if partials.row_indices.len() > partial_row_capacity {
        anyhow::bail!(
            "CUDA Sparse-B fused device residual-add partial rows {} exceed graph bucket capacity {}",
            partials.row_indices.len(),
            partial_row_capacity
        );
    }
    let src_capacity_bytes = partial_row_capacity
        .checked_mul(row_width)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "CUDA Sparse-B fused device residual-add partial buffer capacity overflows usize",
        )?;
    if partials.src_bytes > src_capacity_bytes {
        anyhow::bail!(
            "CUDA Sparse-B fused device residual-add partial bytes {} exceed graph bucket capacity {}",
            partials.src_bytes,
            src_capacity_bytes
        );
    }
    let index_capacity_bytes = partial_row_capacity
        .checked_mul(std::mem::size_of::<u32>())
        .context("CUDA Sparse-B fused device residual-add index buffer capacity overflows usize")?;
    let bucket_values = row_capacity
        .checked_mul(row_width)
        .context("CUDA Sparse-B fused device residual-add row bucket shape overflows usize")?;
    let bucket_f32_bytes = bucket_values
        .checked_mul(std::mem::size_of::<f32>())
        .context("CUDA Sparse-B fused device residual-add f32 bucket byte count overflows usize")?;
    let signature =
        CoordinatorCudaGraphSignature::coord_sparse_b_envelope_bf16(row_capacity, row_width);
    let device_output_buffer = with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let src_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            src_capacity_bytes,
            "Sparse-B fused BF16 partials",
        )?;
        let index_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::B,
            index_capacity_bytes,
            "Sparse-B fused row indices",
        )?;
        let dst_f32_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            bucket_f32_bytes,
            "Sparse-B fused f32 accumulator",
        )?;
        let device_output_buffer = OwnedCoordinatorDeviceBuffer::new(
            library,
            bf16_bytes,
            "Sparse-B fused owned BF16 output",
        )?;

        slot.workspace
            .copy_h2d_segments_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::A,
                partial_outputs_bf16_by_host,
                partials.src_bytes,
                "Sparse-B fused BF16 partials",
                cuda_stream,
            )
            .context("async copying fused Sparse-B BF16 partials to device")?;
        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::B,
                u32_bytes(&partials.row_indices),
                "Sparse-B fused row indices",
                cuda_stream,
            )
            .context("async copying fused Sparse-B row indices to device")?;
        unsafe {
            library
                .cuda_zero_f32_async(dst_f32_buffer, values, cuda_stream)
                .context("async zeroing fused Sparse-B f32 accumulator")?;
        }
        capture_or_update_sparse_b_scatter_shared_residual_add_bf16_graph(
            library,
            slot,
            signature,
            src_buffer,
            index_buffer,
            dst_f32_buffer,
            shared_delta.buffer(),
            residual.buffer(),
            device_output_buffer.buffer,
            dst_rows,
            partials.row_indices.len(),
            row_width,
            values,
        )?;
        unsafe {
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing fused Sparse-B device residual-add graph slot stream")?;
        }
        Ok(device_output_buffer)
    })?;

    Ok(DeviceBf16Output {
        buffer: device_output_buffer,
        bytes: bf16_bytes,
        rows: dst_rows,
        values_per_row: row_width,
        backend: CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
    })
}

#[allow(clippy::too_many_arguments)]
fn cuda_sparse_b_scatter_shared_residual_add_low_precision_device_output(
    residual: &DeviceBf16Output,
    shared_delta: &DeviceBf16Output,
    partial_outputs: &[impl AsRef<[u8]>],
    partials: &SparseBLowPrecisionPartialLayout,
    output_dtype: ExpertV2Dtype,
    output_row_stride_bytes: usize,
    host_ordered: bool,
    dst_rows: usize,
    row_width: usize,
) -> Result<DeviceBf16Output> {
    let values = dst_rows
        .checked_mul(row_width)
        .context("low-precision Sparse-B destination shape overflows usize")?;
    let bf16_bytes = values
        .checked_mul(std::mem::size_of::<u16>())
        .context("low-precision Sparse-B output byte count overflows usize")?;
    let graph_mode = if dst_rows == 1 {
        LayerWaveMode::Decode
    } else {
        LayerWaveMode::Prefill
    };
    let graph_key =
        CoordinatorGraphKey::glm52_bf16(CoordinatorGraphShape::CoordSparseB, graph_mode, dst_rows)
            .context("selecting Coord-Sparse-B graph slot for low-precision accumulation")?;
    let row_capacity = graph_key.row_bucket.row_capacity;
    let partial_row_capacity = row_capacity
        .checked_mul(GLM52_TOP_K)
        .context("low-precision Sparse-B partial row capacity overflows usize")?;
    anyhow::ensure!(
        partials.row_indices.len() <= partial_row_capacity,
        "low-precision Sparse-B partial rows {} exceed graph bucket capacity {partial_row_capacity}",
        partials.row_indices.len()
    );
    let src_capacity_bytes = partial_row_capacity
        .checked_mul(output_row_stride_bytes)
        .context("low-precision Sparse-B source capacity overflows usize")?;
    anyhow::ensure!(
        partials.src_bytes <= src_capacity_bytes,
        "low-precision Sparse-B partial bytes {} exceed graph bucket capacity {src_capacity_bytes}",
        partials.src_bytes
    );
    let index_capacity_bytes = partial_row_capacity
        .checked_mul(std::mem::size_of::<u32>())
        .context("low-precision Sparse-B index capacity overflows usize")?;
    let bucket_values = row_capacity
        .checked_mul(row_width)
        .context("low-precision Sparse-B bucket shape overflows usize")?;
    let bucket_f32_bytes = bucket_values
        .checked_mul(std::mem::size_of::<f32>())
        .context("low-precision Sparse-B accumulator capacity overflows usize")?;
    let device_output_buffer = with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let dst_slot = if host_ordered {
            CoordinatorCudaScratchSlot::S
        } else {
            CoordinatorCudaScratchSlot::C
        };
        let dst_f32_buffer = slot.buffer(
            library,
            dst_slot,
            bucket_f32_bytes,
            "low-precision Sparse-B accumulator",
        )?;
        let device_output_buffer = OwnedCoordinatorDeviceBuffer::new(
            library,
            bf16_bytes,
            "low-precision Sparse-B owned BF16 output",
        )?;
        unsafe {
            library
                .cuda_zero_f32_async(dst_f32_buffer, values, cuda_stream)
                .context("zeroing collected low-precision Sparse-B accumulator")?;
        }
        let launch_scatter = |src_buffer: GlmrtDeviceBuffer,
                              index_buffer: GlmrtDeviceBuffer,
                              partial_rows: usize|
         -> Result<()> {
            unsafe {
                match output_dtype {
                    ExpertV2Dtype::Fp8E4m3RowScaled => library
                        .cuda_scatter_add_rows_fp8_e4m3_row_scaled_to_f32_async(
                            src_buffer,
                            output_row_stride_bytes,
                            index_buffer,
                            dst_f32_buffer,
                            dst_rows,
                            partial_rows,
                            row_width,
                            cuda_stream,
                        )
                        .context("executing collected Sparse-B FP8 scatter-add")?,
                    ExpertV2Dtype::Nvfp4E2m1Fp8E4m3 => library
                        .cuda_scatter_add_rows_nvfp4_e2m1_fp8_e4m3_to_f32_async(
                            src_buffer,
                            output_row_stride_bytes,
                            index_buffer,
                            dst_f32_buffer,
                            dst_rows,
                            partial_rows,
                            row_width,
                            cuda_stream,
                        )
                        .context("executing collected Sparse-B NVFP4 scatter-add")?,
                    _ => unreachable!("low-precision Sparse-B dtype was validated above"),
                }
            }
            Ok(())
        };
        if host_ordered {
            const PAYLOAD_SLOTS: [CoordinatorCudaScratchSlot; 4] = [
                CoordinatorCudaScratchSlot::A,
                CoordinatorCudaScratchSlot::C,
                CoordinatorCudaScratchSlot::E,
                CoordinatorCudaScratchSlot::G,
            ];
            const INDEX_SLOTS: [CoordinatorCudaScratchSlot; 4] = [
                CoordinatorCudaScratchSlot::B,
                CoordinatorCudaScratchSlot::D,
                CoordinatorCudaScratchSlot::F,
                CoordinatorCudaScratchSlot::H,
            ];
            let mut row_offset = 0_usize;
            for (host_index, (partial_output, partial_rows)) in partial_outputs
                .iter()
                .zip(partials.row_counts.iter().copied())
                .enumerate()
            {
                if partial_rows == 0 {
                    continue;
                }
                let row_end = row_offset
                    .checked_add(partial_rows)
                    .context("host-ordered low-precision Sparse-B row range overflow")?;
                let payload_slot = PAYLOAD_SLOTS[host_index];
                let index_slot = INDEX_SLOTS[host_index];
                let partial_output = partial_output.as_ref();
                let src_buffer = slot.buffer(
                    library,
                    payload_slot,
                    partial_output.len(),
                    "host-ordered low-precision Sparse-B partials",
                )?;
                let index_buffer = slot.buffer(
                    library,
                    index_slot,
                    partial_rows * std::mem::size_of::<u32>(),
                    "host-ordered low-precision Sparse-B row indices",
                )?;
                slot.workspace
                    .copy_h2d_to_slot_async(
                        library,
                        payload_slot,
                        partial_output,
                        "host-ordered low-precision Sparse-B partials",
                        cuda_stream,
                    )
                    .context("copying host-ordered low-precision Sparse-B partials")?;
                slot.workspace
                    .copy_h2d_to_slot_async(
                        library,
                        index_slot,
                        u32_bytes(&partials.row_indices[row_offset..row_end]),
                        "host-ordered low-precision Sparse-B row indices",
                        cuda_stream,
                    )
                    .context("copying host-ordered low-precision Sparse-B row indices")?;
                launch_scatter(src_buffer, index_buffer, partial_rows)?;
                row_offset = row_end;
            }
            anyhow::ensure!(
                row_offset == partials.row_indices.len(),
                "host-ordered low-precision Sparse-B staged {row_offset} rows, expected {}",
                partials.row_indices.len()
            );
        } else {
            let src_buffer = slot.buffer(
                library,
                CoordinatorCudaScratchSlot::A,
                src_capacity_bytes,
                "low-precision Sparse-B partials",
            )?;
            let index_buffer = slot.buffer(
                library,
                CoordinatorCudaScratchSlot::B,
                index_capacity_bytes,
                "low-precision Sparse-B row indices",
            )?;
            slot.workspace
                .copy_h2d_segments_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::A,
                    partial_outputs,
                    partials.src_bytes,
                    "low-precision Sparse-B partials",
                    cuda_stream,
                )
                .context("copying collected low-precision Sparse-B partials to device")?;
            slot.workspace
                .copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::B,
                    u32_bytes(&partials.row_indices),
                    "low-precision Sparse-B row indices",
                    cuda_stream,
                )
                .context("copying collected low-precision Sparse-B row indices to device")?;
            launch_scatter(src_buffer, index_buffer, partials.row_indices.len())?;
        }
        unsafe {
            library
                .cuda_residual_add_shared_f32_delta_bf16_async(
                    residual.buffer(),
                    shared_delta.buffer(),
                    dst_f32_buffer,
                    device_output_buffer.buffer,
                    values,
                    cuda_stream,
                )
                .context("executing collected low-precision Sparse-B residual add")?;
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing collected low-precision Sparse-B accumulation")?;
        }
        Ok(device_output_buffer)
    })?;

    Ok(DeviceBf16Output {
        buffer: device_output_buffer,
        bytes: bf16_bytes,
        rows: dst_rows,
        values_per_row: row_width,
        backend: CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
    })
}

pub(in crate::commands::real_full) struct StreamedSparseBResidualSegment<'a> {
    pub(in crate::commands::real_full) residual: &'a DeviceBf16Output,
    pub(in crate::commands::real_full) shared_delta: &'a DeviceBf16Output,
    pub(in crate::commands::real_full) row_start: usize,
    pub(in crate::commands::real_full) row_count: usize,
}

pub(in crate::commands::real_full) struct StreamedSparseBAccumulatorChunk<'a> {
    pub(in crate::commands::real_full) partial_output: &'a [u8],
    pub(in crate::commands::real_full) global_row_indices: &'a [usize],
    pub(in crate::commands::real_full) completed_global_rows: &'a [usize],
    pub(in crate::commands::real_full) output_dtype: ExpertV2Dtype,
    pub(in crate::commands::real_full) output_row_stride_bytes: usize,
}

pub(in crate::commands::real_full) struct CudaStreamedSparseBAccumulator {
    graph_key: CoordinatorGraphKey,
    routed_f32: OwnedCoordinatorDeviceBuffer,
    completed_rows: Vec<bool>,
    finalized_rows: Vec<bool>,
    dst_rows: usize,
    row_width: usize,
    row_capacity: usize,
    total_partial_rows: usize,
}

impl CudaStreamedSparseBAccumulator {
    pub(in crate::commands::real_full) fn new(dst_rows: usize, row_width: usize) -> Result<Self> {
        anyhow::ensure!(
            dst_rows > 0 && row_width > 0,
            "CUDA streamed Sparse-B accumulator requires a non-empty destination"
        );
        let graph_mode = if dst_rows == 1 {
            LayerWaveMode::Decode
        } else {
            LayerWaveMode::Prefill
        };
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseB,
            graph_mode,
            dst_rows,
        )
        .context("selecting Coord-Sparse-B graph slot for incremental Sparse-B")?;
        let row_capacity = graph_key.row_bucket.row_capacity;
        let accumulator_bytes = dst_rows
            .checked_mul(row_width)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .context("incremental Sparse-B accumulator capacity overflows usize")?;
        let routed_f32 = with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
            let routed_f32 = OwnedCoordinatorDeviceBuffer::new(
                library,
                accumulator_bytes,
                "incremental Sparse-B f32 accumulator",
            )?;
            unsafe {
                library
                    .cuda_zero_f32_async(routed_f32.buffer, dst_rows * row_width, slot.stream_ptr())
                    .context("zeroing incremental Sparse-B f32 accumulator")?;
                library
                    .cuda_stream_synchronize(slot.stream_ptr())
                    .context("synchronizing incremental Sparse-B initialization")?;
            }
            Ok(routed_f32)
        })?;
        Ok(Self {
            graph_key,
            routed_f32,
            completed_rows: vec![false; dst_rows],
            finalized_rows: vec![false; dst_rows],
            dst_rows,
            row_width,
            row_capacity,
            total_partial_rows: 0,
        })
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(in crate::commands::real_full) fn push_chunk<Payload: AsRef<[u8]>>(
        &mut self,
        partial_output: Payload,
        global_row_indices: &[usize],
        completed_global_rows: &[usize],
        output_dtype: ExpertV2Dtype,
        output_row_stride_bytes: usize,
    ) -> Result<()> {
        let partial_output = partial_output.as_ref();
        self.push_chunks(&[StreamedSparseBAccumulatorChunk {
            partial_output,
            global_row_indices,
            completed_global_rows,
            output_dtype,
            output_row_stride_bytes,
        }])
    }

    pub(in crate::commands::real_full) fn push_chunks(
        &mut self,
        chunks: &[StreamedSparseBAccumulatorChunk<'_>],
    ) -> Result<()> {
        self.push_chunks_inner(chunks, false)
    }

    pub(in crate::commands::real_full) fn push_host_ordered_chunks(
        &mut self,
        chunks: &[StreamedSparseBAccumulatorChunk<'_>],
    ) -> Result<()> {
        self.push_chunks_inner(chunks, true)
    }

    fn push_chunks_inner(
        &mut self,
        chunks: &[StreamedSparseBAccumulatorChunk<'_>],
        host_ordered: bool,
    ) -> Result<()> {
        let first = chunks
            .first()
            .context("incremental Sparse-B response batch is empty")?;
        anyhow::ensure!(
            matches!(
                first.output_dtype,
                ExpertV2Dtype::Bf16
                    | ExpertV2Dtype::Fp8E4m3RowScaled
                    | ExpertV2Dtype::Nvfp4E2m1Fp8E4m3
            ),
            "incremental Sparse-B output dtype {:?} is unsupported",
            first.output_dtype
        );
        let logical_row_bytes = first.output_dtype.row_bytes(self.row_width)?;
        anyhow::ensure!(
            first.output_row_stride_bytes == logical_row_bytes,
            "incremental Sparse-B compact row stride {} did not match logical {logical_row_bytes} for {:?}",
            first.output_row_stride_bytes,
            first.output_dtype
        );

        let partial_row_capacity = self
            .row_capacity
            .checked_mul(GLM52_TOP_K)
            .context("incremental Sparse-B response batch row capacity overflows usize")?;
        let mut batch_partial_rows = 0_usize;
        let mut batch_payload_bytes = 0_usize;
        let mut payload_segments = Vec::with_capacity(chunks.len());
        let mut row_indices_u32 = Vec::new();
        let mut completed_rows = Vec::new();
        for chunk in chunks {
            anyhow::ensure!(
                chunk.output_dtype == first.output_dtype
                    && chunk.output_row_stride_bytes == first.output_row_stride_bytes,
                "incremental Sparse-B response metadata changed within a coalesced batch"
            );
            let chunk_rows = chunk.global_row_indices.len();
            anyhow::ensure!(
                chunk_rows > 0 && chunk_rows <= self.row_capacity,
                "incremental Sparse-B chunk rows {chunk_rows} are outside 1..={}",
                self.row_capacity
            );
            if host_ordered {
                for (row_offset, global_row_index) in chunk.global_row_indices.iter().enumerate() {
                    anyhow::ensure!(
                        !chunk.global_row_indices[..row_offset].contains(global_row_index),
                        "host-ordered incremental Sparse-B chunk repeats global row {global_row_index}"
                    );
                }
            }
            let expected_bytes = chunk_rows
                .checked_mul(first.output_row_stride_bytes)
                .context("incremental Sparse-B chunk byte count overflows usize")?;
            anyhow::ensure!(
                chunk.partial_output.len() == expected_bytes,
                "incremental Sparse-B chunk bytes {} did not match expected {expected_bytes}",
                chunk.partial_output.len()
            );
            batch_partial_rows = batch_partial_rows
                .checked_add(chunk_rows)
                .context("incremental Sparse-B response batch row count overflow")?;
            batch_payload_bytes = batch_payload_bytes
                .checked_add(chunk.partial_output.len())
                .context("incremental Sparse-B response batch byte count overflow")?;
            payload_segments.push(chunk.partial_output);

            for global_row_index in chunk.global_row_indices {
                anyhow::ensure!(
                    *global_row_index < self.dst_rows,
                    "incremental Sparse-B global row {global_row_index} exceeds destination rows {}",
                    self.dst_rows
                );
                anyhow::ensure!(
                    !self.finalized_rows[*global_row_index],
                    "incremental Sparse-B received another contribution for finalized row {global_row_index}"
                );
                row_indices_u32.push(u32::try_from(*global_row_index).with_context(|| {
                    format!("incremental Sparse-B global row {global_row_index} exceeds u32")
                })?);
            }
            for completed_row in chunk.completed_global_rows {
                anyhow::ensure!(
                    *completed_row < self.dst_rows,
                    "incremental Sparse-B completed row {completed_row} exceeds destination rows {}",
                    self.dst_rows
                );
                anyhow::ensure!(
                    chunk.global_row_indices.contains(completed_row),
                    "incremental Sparse-B completion row {completed_row} was not present in its response chunk"
                );
                anyhow::ensure!(
                    !self.completed_rows[*completed_row] && !completed_rows.contains(completed_row),
                    "incremental Sparse-B row {completed_row} completed more than once"
                );
                completed_rows.push(*completed_row);
            }
        }
        anyhow::ensure!(
            batch_partial_rows <= partial_row_capacity,
            "incremental Sparse-B response batch rows {batch_partial_rows} exceed capacity {partial_row_capacity}"
        );
        let next_total_partial_rows = self
            .total_partial_rows
            .checked_add(batch_partial_rows)
            .context("incremental Sparse-B partial row count overflow")?;
        anyhow::ensure!(
            next_total_partial_rows <= self.dst_rows.saturating_mul(GLM52_TOP_K),
            "incremental Sparse-B partial rows {} exceed routed contribution capacity {}",
            next_total_partial_rows,
            self.dst_rows.saturating_mul(GLM52_TOP_K)
        );

        let index_bytes = batch_partial_rows
            .checked_mul(std::mem::size_of::<u32>())
            .context("incremental Sparse-B index byte count overflows usize")?;
        with_coordinator_cuda_graph_slot(&self.graph_key, |library, slot| {
            let cuda_stream = slot.stream_ptr();
            let launch_scatter = |src_buffer: GlmrtDeviceBuffer,
                                  index_buffer: GlmrtDeviceBuffer,
                                  partial_rows: usize|
             -> Result<()> {
                unsafe {
                    match first.output_dtype {
                        ExpertV2Dtype::Bf16 => library
                            .cuda_scatter_add_rows_bf16_to_f32_async(
                                src_buffer,
                                index_buffer,
                                self.routed_f32.buffer,
                                self.dst_rows,
                                partial_rows,
                                self.row_width,
                                cuda_stream,
                            )
                            .context("executing incremental Sparse-B BF16 scatter-add")?,
                        ExpertV2Dtype::Fp8E4m3RowScaled => library
                            .cuda_scatter_add_rows_fp8_e4m3_row_scaled_to_f32_async(
                                src_buffer,
                                first.output_row_stride_bytes,
                                index_buffer,
                                self.routed_f32.buffer,
                                self.dst_rows,
                                partial_rows,
                                self.row_width,
                                cuda_stream,
                            )
                            .context("executing incremental Sparse-B FP8 scatter-add")?,
                        ExpertV2Dtype::Nvfp4E2m1Fp8E4m3 => library
                            .cuda_scatter_add_rows_nvfp4_e2m1_fp8_e4m3_to_f32_async(
                                src_buffer,
                                first.output_row_stride_bytes,
                                index_buffer,
                                self.routed_f32.buffer,
                                self.dst_rows,
                                partial_rows,
                                self.row_width,
                                cuda_stream,
                            )
                            .context("executing incremental Sparse-B NVFP4 scatter-add")?,
                        _ => unreachable!("incremental Sparse-B dtype was validated above"),
                    }
                }
                Ok(())
            };
            if host_ordered {
                const PAYLOAD_SLOTS: [CoordinatorCudaScratchSlot; 4] = [
                    CoordinatorCudaScratchSlot::A,
                    CoordinatorCudaScratchSlot::C,
                    CoordinatorCudaScratchSlot::E,
                    CoordinatorCudaScratchSlot::G,
                ];
                const INDEX_SLOTS: [CoordinatorCudaScratchSlot; 4] = [
                    CoordinatorCudaScratchSlot::B,
                    CoordinatorCudaScratchSlot::D,
                    CoordinatorCudaScratchSlot::F,
                    CoordinatorCudaScratchSlot::H,
                ];
                let mut row_offset = 0_usize;
                for (chunk_index, chunk) in chunks.iter().enumerate() {
                    let chunk_rows = chunk.global_row_indices.len();
                    let row_end = row_offset + chunk_rows;
                    let scratch_index = chunk_index % PAYLOAD_SLOTS.len();
                    if chunk_index > 0 && scratch_index == 0 {
                        // A streaming response may split one logical host into
                        // multiple chunks. Reuse the four staging pairs only
                        // after their preceding H2D/scatter work has drained,
                        // while retaining the host-sorted accumulation order.
                        unsafe {
                            library.cuda_stream_synchronize(cuda_stream).context(
                                "synchronizing host-ordered incremental Sparse-B staging batch",
                            )?;
                        }
                    }
                    let payload_slot = PAYLOAD_SLOTS[scratch_index];
                    let index_slot = INDEX_SLOTS[scratch_index];
                    let src_buffer = slot.buffer(
                        library,
                        payload_slot,
                        chunk.partial_output.len(),
                        "host-ordered incremental Sparse-B response partials",
                    )?;
                    let index_buffer = slot.buffer(
                        library,
                        index_slot,
                        chunk_rows * std::mem::size_of::<u32>(),
                        "host-ordered incremental Sparse-B response indices",
                    )?;
                    slot.workspace
                        .copy_h2d_to_slot_async(
                            library,
                            payload_slot,
                            chunk.partial_output,
                            "host-ordered incremental Sparse-B response partials",
                            cuda_stream,
                        )
                        .context("copying host-ordered incremental Sparse-B response partials")?;
                    slot.workspace
                        .copy_h2d_to_slot_async(
                            library,
                            index_slot,
                            u32_bytes(&row_indices_u32[row_offset..row_end]),
                            "host-ordered incremental Sparse-B response indices",
                            cuda_stream,
                        )
                        .context("copying host-ordered incremental Sparse-B response indices")?;
                    launch_scatter(src_buffer, index_buffer, chunk_rows)?;
                    row_offset = row_end;
                }
            } else {
                let src_buffer = slot.buffer(
                    library,
                    CoordinatorCudaScratchSlot::A,
                    batch_payload_bytes,
                    "incremental Sparse-B response partials",
                )?;
                let index_buffer = slot.buffer(
                    library,
                    CoordinatorCudaScratchSlot::B,
                    index_bytes,
                    "incremental Sparse-B response indices",
                )?;
                slot.workspace
                    .copy_h2d_segments_to_slot_async(
                        library,
                        CoordinatorCudaScratchSlot::A,
                        &payload_segments,
                        batch_payload_bytes,
                        "incremental Sparse-B response partials",
                        cuda_stream,
                    )
                    .context("copying incremental Sparse-B response partials")?;
                slot.workspace
                    .copy_h2d_to_slot_async(
                        library,
                        CoordinatorCudaScratchSlot::B,
                        u32_bytes(&row_indices_u32),
                        "incremental Sparse-B response indices",
                        cuda_stream,
                    )
                    .context("copying incremental Sparse-B response indices")?;
                launch_scatter(src_buffer, index_buffer, batch_partial_rows)?;
            }
            unsafe {
                library
                    .cuda_stream_synchronize(cuda_stream)
                    .context("synchronizing incremental Sparse-B response chunk")?;
            }
            Ok(())
        })?;
        self.total_partial_rows = next_total_partial_rows;
        for completed_row in completed_rows {
            self.completed_rows[completed_row] = true;
        }
        Ok(())
    }

    pub(in crate::commands::real_full) fn segment_ready(
        &self,
        row_start: usize,
        row_count: usize,
    ) -> Result<bool> {
        let row_end = row_start
            .checked_add(row_count)
            .context("incremental Sparse-B segment row range overflows usize")?;
        let completed = self
            .completed_rows
            .get(row_start..row_end)
            .with_context(|| {
                format!(
                    "incremental Sparse-B segment rows {row_start}..{row_end} exceed {}",
                    self.dst_rows
                )
            })?;
        Ok(!completed.is_empty() && completed.iter().all(|completed| *completed))
    }

    pub(in crate::commands::real_full) fn finalize_segment(
        &mut self,
        segment: &StreamedSparseBResidualSegment<'_>,
    ) -> Result<DeviceBf16Output> {
        validate_sparse_b_device_residual_inputs(
            segment.residual,
            segment.shared_delta,
            segment.row_count,
            self.row_width,
        )?;
        anyhow::ensure!(
            self.segment_ready(segment.row_start, segment.row_count)?,
            "incremental Sparse-B segment {}+{} is not complete",
            segment.row_start,
            segment.row_count
        );
        let row_end = segment
            .row_start
            .checked_add(segment.row_count)
            .context("incremental Sparse-B finalized row range overflows usize")?;
        let finalized = self
            .finalized_rows
            .get(segment.row_start..row_end)
            .context("incremental Sparse-B finalized row range is invalid")?;
        anyhow::ensure!(
            finalized.iter().all(|finalized| !*finalized),
            "incremental Sparse-B segment {}+{} overlaps an already finalized row",
            segment.row_start,
            segment.row_count
        );
        let segment_values = segment
            .row_count
            .checked_mul(self.row_width)
            .context("incremental Sparse-B segment shape overflows usize")?;
        let segment_bf16_bytes = segment_values
            .checked_mul(std::mem::size_of::<u16>())
            .context("incremental Sparse-B segment BF16 bytes overflow usize")?;
        let routed_offset_bytes = segment
            .row_start
            .checked_mul(self.row_width)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .context("incremental Sparse-B routed offset overflows usize")?;
        let routed_bytes = segment_values
            .checked_mul(std::mem::size_of::<f32>())
            .context("incremental Sparse-B routed bytes overflow usize")?;
        let routed_delta = device_buffer_byte_view(
            self.routed_f32.buffer,
            routed_offset_bytes,
            routed_bytes,
            "incremental Sparse-B routed segment",
        )?;
        let output = with_coordinator_cuda_graph_slot(&self.graph_key, |library, slot| {
            let output = OwnedCoordinatorDeviceBuffer::new(
                library,
                segment_bf16_bytes,
                "incremental Sparse-B fused BF16 output",
            )?;
            unsafe {
                library
                    .cuda_residual_add_shared_f32_delta_bf16_async(
                        segment.residual.buffer(),
                        segment.shared_delta.buffer(),
                        routed_delta,
                        output.buffer,
                        segment_values,
                        slot.stream_ptr(),
                    )
                    .context("executing incremental Sparse-B fused residual add")?;
                library
                    .cuda_stream_synchronize(slot.stream_ptr())
                    .context("synchronizing incremental Sparse-B segment output")?;
            }
            Ok(DeviceBf16Output {
                buffer: output,
                bytes: segment_bf16_bytes,
                rows: segment.row_count,
                values_per_row: self.row_width,
                backend: CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
            })
        })?;
        self.finalized_rows[segment.row_start..row_end].fill(true);
        Ok(output)
    }

    pub(in crate::commands::real_full) fn validate_complete(&self) -> Result<()> {
        anyhow::ensure!(
            self.total_partial_rows > 0,
            "incremental Sparse-B dispatch produced no partial rows"
        );
        anyhow::ensure!(
            self.completed_rows.iter().all(|completed| *completed),
            "incremental Sparse-B dispatch left incomplete rows"
        );
        anyhow::ensure!(
            self.finalized_rows.iter().all(|finalized| *finalized),
            "incremental Sparse-B dispatch left unfinalized rows"
        );
        Ok(())
    }
}

pub(in crate::commands::real_full) fn cuda_stream_sparse_b_scatter_shared_residual_add_bf16_device_outputs<
    NextChunk,
    Payload,
>(
    segments: &[StreamedSparseBResidualSegment<'_>],
    dst_rows: usize,
    row_width: usize,
    mut next_chunk: NextChunk,
) -> Result<Vec<DeviceBf16Output>>
where
    NextChunk: FnMut() -> Result<Option<(Payload, Vec<usize>, ExpertV2Dtype, usize)>>,
    Payload: AsRef<[u8]>,
{
    anyhow::ensure!(
        !segments.is_empty(),
        "CUDA streamed Sparse-B requires at least one residual segment"
    );
    let mut expected_row_start = 0_usize;
    for segment in segments {
        anyhow::ensure!(
            segment.row_start == expected_row_start,
            "CUDA streamed Sparse-B segment row start {} did not match expected {expected_row_start}",
            segment.row_start
        );
        validate_sparse_b_device_residual_inputs(
            segment.residual,
            segment.shared_delta,
            segment.row_count,
            row_width,
        )?;
        expected_row_start = expected_row_start
            .checked_add(segment.row_count)
            .context("CUDA streamed Sparse-B segment row range overflows usize")?;
    }
    anyhow::ensure!(
        expected_row_start == dst_rows,
        "CUDA streamed Sparse-B segments cover {expected_row_start} rows instead of {dst_rows}"
    );
    let values = dst_rows
        .checked_mul(row_width)
        .context("CUDA streamed Sparse-B destination shape overflows usize")?;
    let graph_mode = if dst_rows == 1 {
        LayerWaveMode::Decode
    } else {
        LayerWaveMode::Prefill
    };
    let graph_key =
        CoordinatorGraphKey::glm52_bf16(CoordinatorGraphShape::CoordSparseB, graph_mode, dst_rows)
            .context("selecting Coord-Sparse-B graph slot for streamed Sparse-B")?;
    let row_capacity = graph_key.row_bucket.row_capacity;
    let src_capacity_bytes = row_capacity
        .checked_mul(row_width)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA streamed Sparse-B source capacity overflows usize")?;
    let index_capacity_bytes = row_capacity
        .checked_mul(std::mem::size_of::<u32>())
        .context("CUDA streamed Sparse-B index capacity overflows usize")?;
    let bucket_values = row_capacity
        .checked_mul(row_width)
        .context("CUDA streamed Sparse-B bucket shape overflows usize")?;
    let bucket_f32_bytes = bucket_values
        .checked_mul(std::mem::size_of::<f32>())
        .context("CUDA streamed Sparse-B accumulator capacity overflows usize")?;

    with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let src_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            src_capacity_bytes,
            "streamed Sparse-B BF16 partials",
        )?;
        let index_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::B,
            index_capacity_bytes,
            "streamed Sparse-B row indices",
        )?;
        let dst_f32_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            bucket_f32_bytes,
            "streamed Sparse-B f32 accumulator",
        )?;
        unsafe {
            library
                .cuda_zero_f32_async(dst_f32_buffer, values, cuda_stream)
                .context("async zeroing streamed Sparse-B f32 accumulator")?;
        }

        let mut total_partial_rows = 0_usize;
        let mut row_indices_u32 = Vec::new();
        while let Some((
            partial_output,
            global_row_indices,
            output_dtype,
            output_row_stride_bytes,
        )) = next_chunk()?
        {
            let partial_output = partial_output.as_ref();
            let chunk_rows = global_row_indices.len();
            anyhow::ensure!(chunk_rows > 0, "streamed Sparse-B chunk must contain rows");
            anyhow::ensure!(
                chunk_rows <= row_capacity,
                "streamed Sparse-B chunk rows {chunk_rows} exceed graph bucket capacity {row_capacity}"
            );
            anyhow::ensure!(
                matches!(
                    output_dtype,
                    ExpertV2Dtype::Bf16
                        | ExpertV2Dtype::Fp8E4m3RowScaled
                        | ExpertV2Dtype::Nvfp4E2m1Fp8E4m3
                ),
                "streamed Sparse-B output dtype {output_dtype:?} is unsupported"
            );
            let logical_row_bytes = output_dtype.row_bytes(row_width)?;
            anyhow::ensure!(
                output_row_stride_bytes == logical_row_bytes,
                "streamed Sparse-B compact row stride {output_row_stride_bytes} did not match logical {logical_row_bytes} for {output_dtype:?}"
            );
            let expected_bytes = chunk_rows
                .checked_mul(output_row_stride_bytes)
                .context("streamed Sparse-B chunk byte count overflows usize")?;
            anyhow::ensure!(
                partial_output.len() == expected_bytes,
                "streamed Sparse-B chunk bytes {} did not match expected {expected_bytes}",
                partial_output.len()
            );
            total_partial_rows = total_partial_rows
                .checked_add(chunk_rows)
                .context("streamed Sparse-B partial row count overflow")?;
            anyhow::ensure!(
                total_partial_rows <= dst_rows.saturating_mul(GLM52_TOP_K),
                "streamed Sparse-B partial rows {total_partial_rows} exceed routed contribution capacity {}",
                dst_rows.saturating_mul(GLM52_TOP_K)
            );

            row_indices_u32.clear();
            row_indices_u32.reserve(chunk_rows);
            for global_row_index in global_row_indices {
                anyhow::ensure!(
                    global_row_index < dst_rows,
                    "streamed Sparse-B global row {global_row_index} exceeds destination rows {dst_rows}"
                );
                row_indices_u32.push(u32::try_from(global_row_index).with_context(|| {
                    format!("streamed Sparse-B global row {global_row_index} exceeds u32")
                })?);
            }

            slot.workspace
                .copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::A,
                    partial_output,
                    "streamed Sparse-B partials",
                    cuda_stream,
                )
                .context("async copying streamed Sparse-B partials")?;
            slot.workspace
                .copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::B,
                    u32_bytes(&row_indices_u32),
                    "streamed Sparse-B row indices",
                    cuda_stream,
                )
                .context("async copying streamed Sparse-B row indices")?;
            unsafe {
                match output_dtype {
                    ExpertV2Dtype::Bf16 => library
                        .cuda_scatter_add_rows_bf16_to_f32_async(
                            src_buffer,
                            index_buffer,
                            dst_f32_buffer,
                            dst_rows,
                            chunk_rows,
                            row_width,
                            cuda_stream,
                        )
                        .context("executing streamed Sparse-B BF16 scatter-add")?,
                    ExpertV2Dtype::Fp8E4m3RowScaled => library
                        .cuda_scatter_add_rows_fp8_e4m3_row_scaled_to_f32_async(
                            src_buffer,
                            output_row_stride_bytes,
                            index_buffer,
                            dst_f32_buffer,
                            dst_rows,
                            chunk_rows,
                            row_width,
                            cuda_stream,
                        )
                        .context("executing streamed Sparse-B FP8 scatter-add")?,
                    ExpertV2Dtype::Nvfp4E2m1Fp8E4m3 => library
                        .cuda_scatter_add_rows_nvfp4_e2m1_fp8_e4m3_to_f32_async(
                            src_buffer,
                            output_row_stride_bytes,
                            index_buffer,
                            dst_f32_buffer,
                            dst_rows,
                            chunk_rows,
                            row_width,
                            cuda_stream,
                        )
                        .context("executing streamed Sparse-B NVFP4 scatter-add")?,
                    _ => unreachable!("streamed Sparse-B dtype was validated above"),
                }
                library
                    .cuda_stream_synchronize(cuda_stream)
                    .context("synchronizing streamed Sparse-B chunk")?;
            }
        }
        anyhow::ensure!(
            total_partial_rows > 0,
            "streamed Sparse-B dispatch produced no partial rows"
        );
        let mut outputs = Vec::with_capacity(segments.len());
        for segment in segments {
            let segment_values = segment
                .row_count
                .checked_mul(row_width)
                .context("streamed Sparse-B segment shape overflows usize")?;
            let segment_bf16_bytes = segment_values
                .checked_mul(std::mem::size_of::<u16>())
                .context("streamed Sparse-B segment BF16 bytes overflow usize")?;
            let routed_offset_bytes = segment
                .row_start
                .checked_mul(row_width)
                .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
                .context("streamed Sparse-B segment routed offset overflows usize")?;
            let routed_bytes = segment_values
                .checked_mul(std::mem::size_of::<f32>())
                .context("streamed Sparse-B segment routed bytes overflow usize")?;
            let routed_delta = device_buffer_byte_view(
                dst_f32_buffer,
                routed_offset_bytes,
                routed_bytes,
                "streamed Sparse-B routed segment",
            )?;
            let device_output_buffer = OwnedCoordinatorDeviceBuffer::new(
                library,
                segment_bf16_bytes,
                "Sparse-B fused owned BF16 output",
            )?;
            unsafe {
                library
                    .cuda_residual_add_shared_f32_delta_bf16_async(
                        segment.residual.buffer(),
                        segment.shared_delta.buffer(),
                        routed_delta,
                        device_output_buffer.buffer,
                        segment_values,
                        cuda_stream,
                    )
                    .context("executing streamed Sparse-B shared+routed residual segment add")?;
            }
            outputs.push(DeviceBf16Output {
                buffer: device_output_buffer,
                bytes: segment_bf16_bytes,
                rows: segment.row_count,
                values_per_row: row_width,
                backend: CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
            });
        }
        unsafe {
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing streamed Sparse-B residual outputs")?;
        }
        Ok(outputs)
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn capture_or_update_sparse_b_scatter_residual_add_bf16_graph(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    signature: CoordinatorCudaGraphSignature,
    src_buffer: GlmrtDeviceBuffer,
    index_buffer: GlmrtDeviceBuffer,
    dst_f32_buffer: GlmrtDeviceBuffer,
    residual_bf16_buffer: GlmrtDeviceBuffer,
    output_bf16_buffer: GlmrtDeviceBuffer,
    dst_rows: usize,
    rows: usize,
    row_width: usize,
    values: usize,
) -> Result<()> {
    if SPARSE_B_SCATTER_GRAPH_DISABLED.load(Ordering::Relaxed) {
        return launch_sparse_b_scatter_residual_add_bf16_eager(
            library,
            src_buffer,
            index_buffer,
            dst_f32_buffer,
            residual_bf16_buffer,
            output_bf16_buffer,
            dst_rows,
            rows,
            row_width,
            values,
            slot.stream_ptr(),
        );
    }
    let graph_result = (|| -> Result<()> {
        if !slot.has_captured_graph(
            CoordinatorCudaGraphProgram::CoordSparseBEnvelopeBf16,
            signature,
        ) {
            slot.stream_synchronize().context(
                "synchronizing Sparse-B scatter residual-add inputs before graph capture",
            )?;
            slot.capture_graph(
                library,
                CoordinatorCudaGraphProgram::CoordSparseBEnvelopeBf16,
                signature,
                |library, cuda_stream, _workspace| unsafe {
                    library
                        .cuda_scatter_add_rows_bf16_to_f32_async(
                            src_buffer,
                            index_buffer,
                            dst_f32_buffer,
                            dst_rows,
                            rows,
                            row_width,
                            cuda_stream,
                        )
                        .context("capturing async Sparse-B BF16-to-f32 row scatter-add")?;
                    library
                        .cuda_residual_add_f32_delta_bf16_async(
                            residual_bf16_buffer,
                            dst_f32_buffer,
                            output_bf16_buffer,
                            values,
                            cuda_stream,
                        )
                        .context("capturing async Sparse-B BF16 residual add from f32 delta")?;
                    Ok(())
                },
            )?;
        } else {
            let (graph_raw, exec_raw) = slot
                .captured_graph_raw_handles(
                    CoordinatorCudaGraphProgram::CoordSparseBEnvelopeBf16,
                    signature,
                )
                .context(
                    "coordinator CUDA graph slot lost captured Sparse-B scatter residual-add graph before update",
                )?;
            unsafe {
                library
                    .cuda_graph_update_scatter_add_rows_bf16_to_f32_node(
                        graph_raw,
                        exec_raw,
                        0,
                        src_buffer,
                        index_buffer,
                        dst_f32_buffer,
                        dst_rows,
                        rows,
                        row_width,
                    )
                    .context("updating captured Sparse-B scatter-add graph node")?;
                library
                    .cuda_graph_update_residual_add_f32_delta_bf16_node(
                        graph_raw,
                        exec_raw,
                        1,
                        residual_bf16_buffer,
                        dst_f32_buffer,
                        output_bf16_buffer,
                        values,
                    )
                    .context("updating captured Sparse-B f32-delta residual-add graph node")?;
            }
        }
        slot.launch_captured_graph(
            library,
            CoordinatorCudaGraphProgram::CoordSparseBEnvelopeBf16,
            signature,
        )
    })();
    match graph_result {
        Ok(()) => Ok(()),
        Err(_error) => {
            SPARSE_B_SCATTER_GRAPH_DISABLED.store(true, Ordering::Relaxed);
            launch_sparse_b_scatter_residual_add_bf16_eager(
                library,
                src_buffer,
                index_buffer,
                dst_f32_buffer,
                residual_bf16_buffer,
                output_bf16_buffer,
                dst_rows,
                rows,
                row_width,
                values,
                slot.stream_ptr(),
            )
            .context("falling back to eager CUDA Sparse-B scatter residual-add after graph failure")
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn capture_or_update_sparse_b_scatter_shared_residual_add_bf16_graph(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    signature: CoordinatorCudaGraphSignature,
    src_buffer: GlmrtDeviceBuffer,
    index_buffer: GlmrtDeviceBuffer,
    dst_f32_buffer: GlmrtDeviceBuffer,
    shared_delta_bf16_buffer: GlmrtDeviceBuffer,
    residual_bf16_buffer: GlmrtDeviceBuffer,
    output_bf16_buffer: GlmrtDeviceBuffer,
    dst_rows: usize,
    rows: usize,
    row_width: usize,
    values: usize,
) -> Result<()> {
    if SPARSE_B_SCATTER_GRAPH_DISABLED.load(Ordering::Relaxed) {
        return launch_sparse_b_scatter_shared_residual_add_bf16_eager(
            library,
            src_buffer,
            index_buffer,
            dst_f32_buffer,
            shared_delta_bf16_buffer,
            residual_bf16_buffer,
            output_bf16_buffer,
            dst_rows,
            rows,
            row_width,
            values,
            slot.stream_ptr(),
        );
    }
    let graph_result = (|| -> Result<()> {
        if !slot.has_captured_graph(
            CoordinatorCudaGraphProgram::CoordSparseBSharedResidualEnvelopeBf16,
            signature,
        ) {
            slot.stream_synchronize()
                .context("synchronizing fused Sparse-B residual-add inputs before graph capture")?;
            slot.capture_graph(
                library,
                CoordinatorCudaGraphProgram::CoordSparseBSharedResidualEnvelopeBf16,
                signature,
                |library, cuda_stream, _workspace| unsafe {
                    library
                        .cuda_scatter_add_rows_bf16_to_f32_async(
                            src_buffer,
                            index_buffer,
                            dst_f32_buffer,
                            dst_rows,
                            rows,
                            row_width,
                            cuda_stream,
                        )
                        .context("capturing async fused Sparse-B BF16-to-f32 row scatter-add")?;
                    library
                        .cuda_residual_add_shared_f32_delta_bf16_async(
                            residual_bf16_buffer,
                            shared_delta_bf16_buffer,
                            dst_f32_buffer,
                            output_bf16_buffer,
                            values,
                            cuda_stream,
                        )
                        .context(
                            "capturing async fused Sparse-B shared+routed hidden residual add",
                        )?;
                    Ok(())
                },
            )?;
        } else {
            let (graph_raw, exec_raw) = slot
                .captured_graph_raw_handles(
                    CoordinatorCudaGraphProgram::CoordSparseBSharedResidualEnvelopeBf16,
                    signature,
                )
                .context(
                    "coordinator CUDA graph slot lost captured fused Sparse-B residual-add graph before update",
                )?;
            unsafe {
                library
                    .cuda_graph_update_scatter_add_rows_bf16_to_f32_node(
                        graph_raw,
                        exec_raw,
                        0,
                        src_buffer,
                        index_buffer,
                        dst_f32_buffer,
                        dst_rows,
                        rows,
                        row_width,
                    )
                    .context("updating captured fused Sparse-B scatter-add graph node")?;
                library
                    .cuda_graph_update_residual_add_shared_f32_delta_bf16_node(
                        graph_raw,
                        exec_raw,
                        1,
                        residual_bf16_buffer,
                        shared_delta_bf16_buffer,
                        dst_f32_buffer,
                        output_bf16_buffer,
                        values,
                    )
                    .context(
                        "updating captured fused Sparse-B shared+routed residual graph node",
                    )?;
            }
        }
        slot.launch_captured_graph(
            library,
            CoordinatorCudaGraphProgram::CoordSparseBSharedResidualEnvelopeBf16,
            signature,
        )
    })();
    match graph_result {
        Ok(()) => Ok(()),
        Err(_error) => {
            SPARSE_B_SCATTER_GRAPH_DISABLED.store(true, Ordering::Relaxed);
            launch_sparse_b_scatter_shared_residual_add_bf16_eager(
                library,
                src_buffer,
                index_buffer,
                dst_f32_buffer,
                shared_delta_bf16_buffer,
                residual_bf16_buffer,
                output_bf16_buffer,
                dst_rows,
                rows,
                row_width,
                values,
                slot.stream_ptr(),
            )
            .context("falling back to eager CUDA fused Sparse-B residual-add after graph failure")
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_sparse_b_scatter_residual_add_bf16_eager(
    library: &'static NativeLibrary,
    src_buffer: GlmrtDeviceBuffer,
    index_buffer: GlmrtDeviceBuffer,
    dst_f32_buffer: GlmrtDeviceBuffer,
    residual_bf16_buffer: GlmrtDeviceBuffer,
    output_bf16_buffer: GlmrtDeviceBuffer,
    dst_rows: usize,
    rows: usize,
    row_width: usize,
    values: usize,
    cuda_stream: *mut c_void,
) -> Result<()> {
    unsafe {
        library
            .cuda_scatter_add_rows_bf16_to_f32_async(
                src_buffer,
                index_buffer,
                dst_f32_buffer,
                dst_rows,
                rows,
                row_width,
                cuda_stream,
            )
            .context("executing eager Sparse-B BF16-to-f32 row scatter-add")?;
        library
            .cuda_residual_add_f32_delta_bf16_async(
                residual_bf16_buffer,
                dst_f32_buffer,
                output_bf16_buffer,
                values,
                cuda_stream,
            )
            .context("executing eager Sparse-B BF16 residual add from f32 delta")?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_sparse_b_scatter_shared_residual_add_bf16_eager(
    library: &'static NativeLibrary,
    src_buffer: GlmrtDeviceBuffer,
    index_buffer: GlmrtDeviceBuffer,
    dst_f32_buffer: GlmrtDeviceBuffer,
    shared_delta_bf16_buffer: GlmrtDeviceBuffer,
    residual_bf16_buffer: GlmrtDeviceBuffer,
    output_bf16_buffer: GlmrtDeviceBuffer,
    dst_rows: usize,
    rows: usize,
    row_width: usize,
    values: usize,
    cuda_stream: *mut c_void,
) -> Result<()> {
    unsafe {
        library
            .cuda_scatter_add_rows_bf16_to_f32_async(
                src_buffer,
                index_buffer,
                dst_f32_buffer,
                dst_rows,
                rows,
                row_width,
                cuda_stream,
            )
            .context("executing eager fused Sparse-B BF16-to-f32 row scatter-add")?;
        library
            .cuda_residual_add_shared_f32_delta_bf16_async(
                residual_bf16_buffer,
                shared_delta_bf16_buffer,
                dst_f32_buffer,
                output_bf16_buffer,
                values,
                cuda_stream,
            )
            .context("executing eager fused Sparse-B shared+routed hidden residual add")?;
    }
    Ok(())
}

#[cfg(test)]
mod prefill_graph_policy_tests {
    use super::{sparse_b_residual_add_graph_replay_enabled, CoordinatorCudaGraphSignature};

    #[test]
    fn sparse_b_residual_graph_replay_stops_above_retained_decode_width() {
        assert!(sparse_b_residual_add_graph_replay_enabled(
            CoordinatorCudaGraphSignature::residual_add_bf16(16 * 6_144 * 2)
        ));
        assert!(!sparse_b_residual_add_graph_replay_enabled(
            CoordinatorCudaGraphSignature::residual_add_bf16(17 * 6_144 * 2)
        ));
        assert!(!sparse_b_residual_add_graph_replay_enabled(
            CoordinatorCudaGraphSignature::residual_add_bf16(2_048 * 6_144 * 2)
        ));
    }
}
