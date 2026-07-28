use super::*;
use anyhow::{Context, Result};
use glmrt_core::{
    CoordinatorGraphInstancePlan, CoordinatorGraphKey, CoordinatorGraphShape, LayerId,
    LayerWaveMode, COORDINATOR_GRAPH_INSTANCE_COUNT, GLM52_FIRST_K_DENSE_REPLACE,
    GLM52_HIDDEN_BF16_BYTES, GLM52_HIDDEN_SIZE, GLM52_NUM_HIDDEN_LAYERS,
    GLM52_ROUTED_SCALING_FACTOR, GLM52_TOP_K, GLM52_TOTAL_LAYERS_WITH_MTP,
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

pub(in crate::commands::real_full) const CUDA_REFERENCE_DEVICE_BF16_HOST_UPLOAD_BACKEND: &str =
    "cuda-reference-device-bf16-host-upload";
pub(in crate::commands::real_full) const CUDA_REFERENCE_DEVICE_BF16_HOST_UPLOAD_DEVICE_PREFIX_BACKEND:
    &str = "cuda-reference-device-bf16-host-upload-device-prefix";
pub(in crate::commands::real_full) const CUDA_REFERENCE_DEVICE_BF16_TEMPLATE_COPY_BACKEND: &str =
    "cuda-reference-device-bf16-template-copy";
pub(in crate::commands::real_full) const CUDA_REFERENCE_DEVICE_BF16_TEMPLATE_COPY_DEVICE_PREFIX_BACKEND:
    &str = "cuda-reference-device-bf16-template-copy-device-prefix";
pub(in crate::commands::real_full) const CUDA_REFERENCE_DEVICE_BF16_FEATURE_CONCAT_BACKEND: &str =
    "cuda-reference-device-bf16-feature-concat";

pub(in crate::commands::real_full) fn device_bf16_output_from_owned_device_buffer(
    library: &'static NativeLibrary,
    buffer: GlmrtDeviceBuffer,
    rows: usize,
    values_per_row: usize,
    backend: &'static str,
    label: &'static str,
) -> Result<DeviceBf16Output> {
    let value_count = validate_device_bf16_output_shape(
        buffer.bytes / std::mem::size_of::<u16>(),
        rows,
        values_per_row,
        label,
    )?;
    let expected_bytes = value_count
        .checked_mul(std::mem::size_of::<u16>())
        .context("owned BF16 device output adopted byte count overflows usize")?;
    let output_buffer =
        OwnedCoordinatorDeviceBuffer::from_existing(library, buffer, expected_bytes, label)?;
    Ok(DeviceBf16Output {
        buffer: output_buffer,
        bytes: expected_bytes,
        rows,
        values_per_row,
        backend,
    })
}

pub(in crate::commands::real_full) fn device_bf16_output_uninitialized(
    rows: usize,
    values_per_row: usize,
    backend: &'static str,
    label: &'static str,
) -> Result<DeviceBf16Output> {
    let value_count = rows
        .checked_mul(values_per_row)
        .with_context(|| format!("uninitialized BF16 device output shape for {label} overflows"))?;
    let bytes = value_count
        .checked_mul(std::mem::size_of::<u16>())
        .context("uninitialized BF16 device output byte count overflows usize")?;
    let library = cuda_native_library()?;
    let buffer = OwnedCoordinatorDeviceBuffer::new(library, bytes, label)?;
    Ok(DeviceBf16Output {
        buffer,
        bytes,
        rows,
        values_per_row,
        backend,
    })
}

pub(in crate::commands::real_full) fn concat_device_bf16_row_features(
    left: &DeviceBf16Output,
    right: &DeviceBf16Output,
    label: &'static str,
) -> Result<DeviceBf16Output> {
    anyhow::ensure!(
        left.rows == right.rows,
        "device BF16 feature concat for {label} row mismatch: left={} right={}",
        left.rows,
        right.rows
    );
    anyhow::ensure!(
        left.buffer().device_id == right.buffer().device_id,
        "device BF16 feature concat for {label} requires inputs on one device: left={} right={}",
        left.buffer().device_id,
        right.buffer().device_id
    );
    let values_per_row = left
        .values_per_row
        .checked_add(right.values_per_row)
        .with_context(|| format!("device BF16 feature concat width for {label} overflows"))?;
    let output = device_bf16_output_uninitialized(
        left.rows,
        values_per_row,
        CUDA_REFERENCE_DEVICE_BF16_FEATURE_CONCAT_BACKEND,
        label,
    )?;
    let left_row_bytes = left
        .values_per_row
        .checked_mul(std::mem::size_of::<u16>())
        .context("device BF16 feature concat left row bytes overflow")?;
    let right_row_bytes = right
        .values_per_row
        .checked_mul(std::mem::size_of::<u16>())
        .context("device BF16 feature concat right row bytes overflow")?;
    let output_row_bytes = values_per_row
        .checked_mul(std::mem::size_of::<u16>())
        .context("device BF16 feature concat output row bytes overflow")?;
    let output_bytes = left
        .rows
        .checked_mul(output_row_bytes)
        .context("device BF16 feature concat output bytes overflow")?;
    let output_right = device_buffer_byte_view(
        output.buffer(),
        left_row_bytes,
        output_bytes - left_row_bytes,
        label,
    )?;
    let library = cuda_native_library()?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let stream = workspace.stream_ptr(library)?;
    unsafe {
        left.wait_ready_on_stream(stream)
            .with_context(|| format!("waiting for left device BF16 features for {label}"))?;
        right
            .wait_ready_on_stream(stream)
            .with_context(|| format!("waiting for right device BF16 features for {label}"))?;
        library
            .copy_d2d_2d_async(
                output.buffer(),
                output_row_bytes,
                left.buffer(),
                left_row_bytes,
                left_row_bytes,
                left.rows,
                stream,
            )
            .with_context(|| format!("copying left device BF16 features for {label}"))?;
        library
            .copy_d2d_2d_async(
                output_right,
                output_row_bytes,
                right.buffer(),
                right_row_bytes,
                right_row_bytes,
                right.rows,
                stream,
            )
            .with_context(|| format!("copying right device BF16 features for {label}"))?;
        library
            .cuda_stream_synchronize(stream)
            .with_context(|| format!("synchronizing device BF16 feature concat for {label}"))?;
    }
    Ok(output)
}

pub(in crate::commands::real_full) fn concat_device_bf16_row_batches(
    batches: &[&DeviceBf16Output],
    label: &'static str,
) -> Result<DeviceBf16Output> {
    concat_device_bf16_row_batches_impl(batches, label, true)
}

pub(in crate::commands::real_full) fn concat_device_bf16_row_batches_async(
    batches: &[&DeviceBf16Output],
    label: &'static str,
) -> Result<DeviceBf16Output> {
    concat_device_bf16_row_batches_impl(batches, label, false)
}

pub(in crate::commands::real_full) fn concat_device_bf16_row_slices_async(
    slices: &[(&DeviceBf16Output, usize, usize)],
    label: &'static str,
) -> Result<DeviceBf16Output> {
    let (first, _, _) = slices
        .first()
        .context("device BF16 row-slice concat requires at least one slice")?;
    let values_per_row = first.values_per_row;
    let device_id = first.buffer().device_id;
    let rows = slices
        .iter()
        .try_fold(0_usize, |total, (source, row_start, rows)| {
            anyhow::ensure!(
                *rows > 0
                    && source.values_per_row == values_per_row
                    && source.buffer().device_id == device_id
                    && row_start
                        .checked_add(*rows)
                        .is_some_and(|row_end| row_end <= source.rows),
                "device BF16 row slice for {label} is invalid"
            );
            total
                .checked_add(*rows)
                .context("device BF16 row-slice concat row count overflows usize")
        })?;
    let row_bytes = values_per_row
        .checked_mul(std::mem::size_of::<u16>())
        .context("device BF16 row-slice concat row bytes overflow")?;
    let mut output = device_bf16_output_uninitialized(
        rows,
        values_per_row,
        CUDA_REFERENCE_DEVICE_BF16_TEMPLATE_COPY_BACKEND,
        label,
    )?;
    let library = cuda_native_library()?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let stream = workspace.stream_ptr(library)?;
    let mut destination_offset = 0_usize;
    unsafe {
        for (source, row_start, rows) in slices {
            let bytes = rows
                .checked_mul(row_bytes)
                .context("device BF16 row-slice byte count overflow")?;
            let source_offset = row_start
                .checked_mul(row_bytes)
                .context("device BF16 row-slice source offset overflow")?;
            let source_view =
                device_buffer_byte_view(source.buffer(), source_offset, bytes, label)?;
            let destination =
                device_buffer_byte_view(output.buffer(), destination_offset, bytes, label)?;
            source
                .wait_ready_on_stream(stream)
                .with_context(|| format!("waiting for device BF16 row slice for {label}"))?;
            library
                .copy_d2d_async(destination, source_view, bytes, stream)
                .with_context(|| format!("copying device BF16 row slice for {label}"))?;
            destination_offset = destination_offset
                .checked_add(bytes)
                .context("device BF16 row-slice destination offset overflow")?;
        }
        let ready_event = Arc::new(CoordinatorCudaEvent::create(library)?);
        ready_event
            .record(stream)
            .with_context(|| format!("recording device BF16 row-slice concat for {label}"))?;
        output.set_ready_event(ready_event);
    }
    Ok(output)
}

fn concat_device_bf16_row_batches_impl(
    batches: &[&DeviceBf16Output],
    label: &'static str,
    synchronize: bool,
) -> Result<DeviceBf16Output> {
    let first = batches
        .first()
        .context("device BF16 row concat requires at least one batch")?;
    anyhow::ensure!(
        first.rows > 0 && first.values_per_row > 0,
        "device BF16 row concat for {label} requires a non-empty first batch"
    );
    let values_per_row = first.values_per_row;
    let device_id = first.buffer().device_id;
    let rows = batches.iter().try_fold(0_usize, |rows, batch| {
        anyhow::ensure!(
            batch.rows > 0 && batch.values_per_row == values_per_row,
            "device BF16 row concat for {label} expected width {values_per_row}, got {}x{}",
            batch.rows,
            batch.values_per_row
        );
        anyhow::ensure!(
            batch.buffer().device_id == device_id,
            "device BF16 row concat for {label} requires one device: expected {device_id}, got {}",
            batch.buffer().device_id
        );
        rows.checked_add(batch.rows)
            .context("device BF16 row concat row count overflows usize")
    })?;
    let row_bytes = values_per_row
        .checked_mul(std::mem::size_of::<u16>())
        .context("device BF16 row concat row bytes overflow")?;
    let mut output = device_bf16_output_uninitialized(
        rows,
        values_per_row,
        CUDA_REFERENCE_DEVICE_BF16_TEMPLATE_COPY_BACKEND,
        label,
    )?;
    let library = cuda_native_library()?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let stream = workspace.stream_ptr(library)?;
    let mut destination_offset = 0_usize;
    unsafe {
        for batch in batches {
            let bytes = batch
                .rows
                .checked_mul(row_bytes)
                .context("device BF16 row concat batch bytes overflow")?;
            let destination =
                device_buffer_byte_view(output.buffer(), destination_offset, bytes, label)?;
            batch
                .wait_ready_on_stream(stream)
                .with_context(|| format!("waiting for device BF16 row batch for {label}"))?;
            library
                .copy_d2d_async(destination, batch.buffer(), bytes, stream)
                .with_context(|| format!("copying device BF16 row batch for {label}"))?;
            destination_offset = destination_offset
                .checked_add(bytes)
                .context("device BF16 row concat destination offset overflow")?;
        }
        if synchronize {
            library
                .cuda_stream_synchronize(stream)
                .with_context(|| format!("synchronizing device BF16 row concat for {label}"))?;
        } else {
            let ready_event = Arc::new(CoordinatorCudaEvent::create(library)?);
            ready_event
                .record(stream)
                .with_context(|| format!("recording device BF16 row concat for {label}"))?;
            output.set_ready_event(ready_event);
        }
    }
    Ok(output)
}

#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn device_bf16_output_from_f32_values(
    values: &[f32],
    rows: usize,
    values_per_row: usize,
    label: &'static str,
) -> Result<DeviceBf16Output> {
    let value_count = validate_device_bf16_output_shape(values.len(), rows, values_per_row, label)?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "owned BF16 device output upload for {label} requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1"
        );
    }
    let bytes = f32_values_to_bf16_bytes(values);
    let expected_bytes = value_count
        .checked_mul(std::mem::size_of::<u16>())
        .context("owned BF16 device output byte count overflows usize")?;
    cuda_device_bf16_output_from_host_bytes(&bytes, expected_bytes, rows, values_per_row, label)
}

pub(in crate::commands::real_full) fn device_bf16_output_from_bf16_bytes(
    bytes: &[u8],
    rows: usize,
    values_per_row: usize,
    label: &'static str,
) -> Result<DeviceBf16Output> {
    if bytes.len() % std::mem::size_of::<u16>() != 0 {
        anyhow::bail!(
            "owned BF16 device output upload for {label} byte length must be even, got {}",
            bytes.len()
        );
    }
    let value_count = validate_device_bf16_output_shape(
        bytes.len() / std::mem::size_of::<u16>(),
        rows,
        values_per_row,
        label,
    )?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "owned BF16 device output upload for {label} requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1"
        );
    }
    let expected_bytes = value_count
        .checked_mul(std::mem::size_of::<u16>())
        .context("owned BF16 device output byte count overflows usize")?;
    cuda_device_bf16_output_from_host_bytes(bytes, expected_bytes, rows, values_per_row, label)
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn device_bf16_output_from_bf16_bytes_with_device_row_prefix(
    bytes: &[u8],
    rows: usize,
    values_per_row: usize,
    prefix_source: &DeviceBf16Output,
    source_row_offset: usize,
    prefix_values_per_row: usize,
    label: &'static str,
) -> Result<DeviceBf16Output> {
    if prefix_values_per_row == 0 || prefix_values_per_row > values_per_row {
        anyhow::bail!(
            "owned BF16 device output prefix for {label} width {prefix_values_per_row} outside 1..={values_per_row}"
        );
    }
    if prefix_source.values_per_row < prefix_values_per_row {
        anyhow::bail!(
            "owned BF16 device output prefix for {label} source width {} is smaller than prefix width {prefix_values_per_row}",
            prefix_source.values_per_row
        );
    }
    let end_source_row = source_row_offset
        .checked_add(rows)
        .context("owned BF16 device output prefix source row range overflows usize")?;
    if end_source_row > prefix_source.rows {
        anyhow::bail!(
            "owned BF16 device output prefix for {label} source rows {source_row_offset}..{end_source_row} exceed {}",
            prefix_source.rows
        );
    }
    let mut output = device_bf16_output_from_bf16_bytes(bytes, rows, values_per_row, label)?;
    output.backend = CUDA_REFERENCE_DEVICE_BF16_HOST_UPLOAD_DEVICE_PREFIX_BACKEND;
    copy_device_bf16_row_prefix(
        cuda_native_library()?,
        prefix_source,
        output.buffer.buffer,
        rows,
        values_per_row,
        source_row_offset,
        prefix_values_per_row,
        label,
        "owned BF16 device output prefix",
    )?;
    Ok(output)
}

pub(in crate::commands::real_full) fn device_bf16_output_from_device_template_buffer(
    template: GlmrtDeviceBuffer,
    rows: usize,
    values_per_row: usize,
    label: &'static str,
) -> Result<DeviceBf16Output> {
    let expected_bytes =
        validate_device_bf16_template_buffer(template, rows, values_per_row, label)?;
    let library = cuda_native_library()?;
    let output_buffer = OwnedCoordinatorDeviceBuffer::new(library, expected_bytes, label)?;
    library
        .copy_d2d(output_buffer.buffer, template, expected_bytes)
        .with_context(|| format!("copying BF16 device template for {label}"))?;
    Ok(DeviceBf16Output {
        buffer: output_buffer,
        bytes: expected_bytes,
        rows,
        values_per_row,
        backend: CUDA_REFERENCE_DEVICE_BF16_TEMPLATE_COPY_BACKEND,
    })
}

pub(in crate::commands::real_full) fn validate_device_bf16_template_buffer(
    template: GlmrtDeviceBuffer,
    rows: usize,
    values_per_row: usize,
    label: &'static str,
) -> Result<usize> {
    if template.ptr.is_null() {
        anyhow::bail!("owned BF16 device template for {label} is null");
    }
    let value_count = validate_device_bf16_output_shape(
        rows.checked_mul(values_per_row)
            .context("owned BF16 device template value count overflows usize")?,
        rows,
        values_per_row,
        label,
    )?;
    let expected_bytes = value_count
        .checked_mul(std::mem::size_of::<u16>())
        .context("owned BF16 device template byte count overflows usize")?;
    if template.bytes < expected_bytes {
        anyhow::bail!(
            "owned BF16 device template for {label} has {} bytes, expected at least {expected_bytes}",
            template.bytes
        );
    }
    Ok(expected_bytes)
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn device_bf16_output_from_device_template_with_device_row_prefix(
    template: GlmrtDeviceBuffer,
    rows: usize,
    values_per_row: usize,
    prefix_source: &DeviceBf16Output,
    source_row_offset: usize,
    prefix_values_per_row: usize,
    label: &'static str,
) -> Result<DeviceBf16Output> {
    let mut output =
        device_bf16_output_from_device_template_buffer(template, rows, values_per_row, label)?;
    output.backend = CUDA_REFERENCE_DEVICE_BF16_TEMPLATE_COPY_DEVICE_PREFIX_BACKEND;
    if prefix_values_per_row == 0 || prefix_values_per_row > values_per_row {
        anyhow::bail!(
            "owned BF16 device template prefix for {label} width {prefix_values_per_row} outside 1..={values_per_row}"
        );
    }
    if prefix_source.values_per_row < prefix_values_per_row {
        anyhow::bail!(
            "owned BF16 device template prefix for {label} source width {} is smaller than prefix width {prefix_values_per_row}",
            prefix_source.values_per_row
        );
    }
    let end_source_row = source_row_offset
        .checked_add(rows)
        .context("owned BF16 device template prefix source row range overflows usize")?;
    if end_source_row > prefix_source.rows {
        anyhow::bail!(
            "owned BF16 device template prefix for {label} source rows {source_row_offset}..{end_source_row} exceed {}",
            prefix_source.rows
        );
    }
    copy_device_bf16_row_prefix(
        cuda_native_library()?,
        prefix_source,
        output.buffer.buffer,
        rows,
        values_per_row,
        source_row_offset,
        prefix_values_per_row,
        label,
        "owned BF16 device template prefix",
    )?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn copy_device_bf16_row_prefix(
    library: &NativeLibrary,
    prefix_source: &DeviceBf16Output,
    output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    values_per_row: usize,
    source_row_offset: usize,
    prefix_values_per_row: usize,
    label: &'static str,
    context: &str,
) -> Result<()> {
    if prefix_source.buffer.buffer.device_id != output_buffer.device_id {
        anyhow::bail!(
            "{context} for {label} buffers are on different devices: source={} destination={}",
            prefix_source.buffer.buffer.device_id,
            output_buffer.device_id
        );
    }
    prefix_source
        .synchronize_ready()
        .with_context(|| format!("waiting for device BF16 row prefix for {label}"))?;
    library
        .cuda_copy_row_prefix_bf16(
            prefix_source.buffer.buffer,
            prefix_source.rows,
            output_buffer,
            rows,
            prefix_source.values_per_row,
            values_per_row,
            prefix_values_per_row,
            source_row_offset,
        )
        .with_context(|| format!("copying device BF16 row prefix for {label}"))
}

pub(in crate::commands::real_full) fn validate_device_bf16_output_shape(
    actual_values: usize,
    rows: usize,
    values_per_row: usize,
    label: &'static str,
) -> Result<usize> {
    if rows == 0 || values_per_row == 0 {
        anyhow::bail!(
            "owned BF16 device output upload for {label} requires non-zero shape, got rows={rows} values_per_row={values_per_row}"
        );
    }
    let value_count = rows
        .checked_mul(values_per_row)
        .context("owned BF16 device output shape overflows usize")?;
    if actual_values != value_count {
        anyhow::bail!(
            "owned BF16 device output upload for {label} value count mismatch: expected {value_count} got {actual_values}"
        );
    }
    Ok(value_count)
}

pub(in crate::commands::real_full) fn bf16_value(bytes: &[u8], index: usize) -> f32 {
    let byte_index = index * std::mem::size_of::<u16>();
    let bits = u16::from_le_bytes([bytes[byte_index], bytes[byte_index + 1]]);
    f32::from_bits((bits as u32) << 16)
}

pub(in crate::commands::real_full) fn bf16_values_to_f32(bytes: &[u8]) -> Vec<f32> {
    (0..bytes.len() / std::mem::size_of::<u16>())
        .map(|index| bf16_value(bytes, index))
        .collect()
}

#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn f32_values_to_bf16_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<u16>());
    for value in values {
        let bits = (value.to_bits() >> 16) as u16;
        bytes.extend_from_slice(&bits.to_le_bytes());
    }
    bytes
}

pub(in crate::commands::real_full) fn cuda_device_bf16_output_from_host_bytes(
    bytes: &[u8],
    expected_bytes: usize,
    rows: usize,
    values_per_row: usize,
    label: &'static str,
) -> Result<DeviceBf16Output> {
    if bytes.len() != expected_bytes {
        anyhow::bail!(
            "owned BF16 device output upload for {label} byte count mismatch: expected {expected_bytes} got {}",
            bytes.len()
        );
    }
    let library = cuda_native_library()?;
    let output_buffer = OwnedCoordinatorDeviceBuffer::new(library, expected_bytes, label)?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let staging = workspace
        .host_buffer(
            library,
            CoordinatorCudaScratchSlot::F,
            expected_bytes,
            label,
        )
        .with_context(|| format!("allocating pinned staging for owned BF16 upload {label}"))?;
    if staging.ptr.is_null() {
        anyhow::bail!("owned BF16 device output upload for {label} pinned staging is null");
    }
    if expected_bytes > staging.bytes {
        anyhow::bail!(
            "owned BF16 device output upload for {label} byte count {expected_bytes} exceeds pinned staging bytes {}",
            staging.bytes
        );
    }
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), staging.ptr.cast::<u8>(), expected_bytes);
    }
    library
        .copy_host_buffer_h2d(output_buffer.buffer, staging, expected_bytes)
        .with_context(|| format!("copying owned BF16 device output upload for {label}"))?;
    Ok(DeviceBf16Output {
        buffer: output_buffer,
        bytes: expected_bytes,
        rows,
        values_per_row,
        backend: CUDA_REFERENCE_DEVICE_BF16_HOST_UPLOAD_BACKEND,
    })
}

pub(in crate::commands::real_full) fn glm52_layer_id_from_tensor_name(
    tensor_name: &str,
) -> Option<usize> {
    glm52_layer_tensor_subpath(tensor_name).map(|(layer_id, _)| layer_id)
}

pub(in crate::commands::real_full) fn glm52_layer_tensor_subpath(
    tensor_name: &str,
) -> Option<(usize, &str)> {
    let after_prefix = tensor_name.strip_prefix("model.layers.")?;
    let (layer_text, subpath) = after_prefix.split_once('.')?;
    let layer_id = layer_text.parse::<usize>().ok()?;
    (layer_id < GLM52_TOTAL_LAYERS_WITH_MTP).then_some((layer_id, subpath))
}

pub(in crate::commands::real_full) const COORDINATOR_CUDA_SCRATCH_SLOTS:
    [CoordinatorCudaScratchSlot; 22] = [
    CoordinatorCudaScratchSlot::A,
    CoordinatorCudaScratchSlot::B,
    CoordinatorCudaScratchSlot::C,
    CoordinatorCudaScratchSlot::D,
    CoordinatorCudaScratchSlot::E,
    CoordinatorCudaScratchSlot::F,
    CoordinatorCudaScratchSlot::G,
    CoordinatorCudaScratchSlot::H,
    CoordinatorCudaScratchSlot::I,
    CoordinatorCudaScratchSlot::J,
    CoordinatorCudaScratchSlot::K,
    CoordinatorCudaScratchSlot::L,
    CoordinatorCudaScratchSlot::M,
    CoordinatorCudaScratchSlot::N,
    CoordinatorCudaScratchSlot::O,
    CoordinatorCudaScratchSlot::P,
    CoordinatorCudaScratchSlot::Q,
    CoordinatorCudaScratchSlot::R,
    CoordinatorCudaScratchSlot::S,
    CoordinatorCudaScratchSlot::T,
    CoordinatorCudaScratchSlot::U,
    CoordinatorCudaScratchSlot::V,
];
pub(in crate::commands::real_full) const COORDINATOR_CUDA_SCRATCH_SLOT_COUNT: usize =
    COORDINATOR_CUDA_SCRATCH_SLOTS.len();

pub(in crate::commands::real_full) fn device_buffer_byte_view(
    buffer: GlmrtDeviceBuffer,
    offset_bytes: usize,
    view_bytes: usize,
    label: &str,
) -> Result<GlmrtDeviceBuffer> {
    if buffer.ptr.is_null() {
        anyhow::bail!("{label} device buffer is null");
    }
    let end = offset_bytes
        .checked_add(view_bytes)
        .with_context(|| format!("{label} device buffer view end overflows usize"))?;
    if end > buffer.bytes {
        anyhow::bail!(
            "{label} device buffer view offset={} bytes={} exceeds buffer bytes {}",
            offset_bytes,
            view_bytes,
            buffer.bytes
        );
    }
    Ok(GlmrtDeviceBuffer {
        ptr: buffer.ptr.cast::<u8>().wrapping_add(offset_bytes).cast(),
        bytes: view_bytes,
        device_id: buffer.device_id,
        flags: buffer.flags,
    })
}

pub(in crate::commands::real_full) fn cuda_native_library() -> Result<&'static NativeLibrary> {
    if let Some(library) = CUDA_NATIVE_LIBRARY.get() {
        return Ok(library);
    }
    let path = native_library_path().with_context(|| {
        format!(
            "{REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1 requires GLMRT_NATIVE_LIB or native/build-cuda/libglmrt_native.so"
        )
    })?;
    let library = unsafe { NativeLibrary::load(&path) }
        .with_context(|| format!("loading native CUDA reference library {}", path.display()))?;
    require_cuda_enabled_native_library(&library, &path, "real-full coordinator CUDA kernels")?;
    Ok(CUDA_NATIVE_LIBRARY.get_or_init(|| library))
}

pub(in crate::commands::real_full) fn require_cuda_enabled_native_library(
    library: &NativeLibrary,
    path: &Path,
    purpose: &str,
) -> Result<()> {
    let version = library
        .version()
        .with_context(|| format!("reading native library version from {}", path.display()))?;
    if native_library_version_has_cuda(&version) {
        return Ok(());
    }
    anyhow::bail!(
        "{purpose} require a CUDA-enabled native library, but {} reports `{version}`. Rebuild with `cmake -S native -B native/build-cuda -G Ninja -DGLMRT_ENABLE_CUDA=ON -DGLMRT_ENABLE_RDMA=OFF -DGLMRT_CUDA_ARCHITECTURES=120 && cmake --build native/build-cuda`, or set GLMRT_NATIVE_LIB to a CUDA-enabled libglmrt_native.so",
        path.display()
    );
}

pub(in crate::commands::real_full) fn native_library_version_has_cuda(version: &str) -> bool {
    version
        .split_ascii_whitespace()
        .any(|part| part.eq_ignore_ascii_case("cuda=on"))
}

pub(in crate::commands::real_full) fn native_library_path() -> Option<PathBuf> {
    if let Ok(path) = env::var("GLMRT_NATIVE_LIB") {
        return Some(PathBuf::from(path));
    }
    if env::var_os("GLMRT_DISABLE_NATIVE_AUTO_DISCOVERY").is_some() {
        return None;
    }
    native_library_path_candidates()
        .into_iter()
        .find(|path| path.exists())
}

pub(in crate::commands::real_full) fn native_library_path_candidates() -> Vec<PathBuf> {
    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .join("native");
    vec![
        manifest_path.join("build-cuda/libglmrt_native.so"),
        PathBuf::from("native/build-cuda/libglmrt_native.so"),
    ]
}

pub(in crate::commands::real_full) fn f32_bytes(values: &[f32]) -> &[u8] {
    unsafe { slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

pub(in crate::commands::real_full) fn u32_bytes(values: &[u32]) -> &[u8] {
    unsafe { slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

pub(in crate::commands::real_full) fn f32_vec_from_bytes(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        anyhow::bail!(
            "CUDA reference kernel returned non-f32-aligned byte count {}",
            bytes.len()
        );
    }
    Ok(bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("chunk width checked")))
        .collect())
}

pub(in crate::commands::real_full) fn u32_vec_from_bytes(bytes: &[u8]) -> Result<Vec<u32>> {
    if bytes.len() % std::mem::size_of::<u32>() != 0 {
        anyhow::bail!(
            "CUDA reference kernel returned non-u32-aligned byte count {}",
            bytes.len()
        );
    }
    Ok(bytes
        .chunks_exact(std::mem::size_of::<u32>())
        .map(|chunk| u32::from_ne_bytes(chunk.try_into().expect("chunk width checked")))
        .collect())
}
