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
pub(in crate::commands::real_full) const CPU_REFERENCE_EMBEDDING_LOOKUP_BACKEND: &str =
    "cpu-reference-embedding-lookup";
#[allow(dead_code)]
pub(in crate::commands::real_full) const CUDA_REFERENCE_EMBEDDING_LOOKUP_BACKEND: &str =
    "cuda-reference-embedding-lookup-f32";
pub(in crate::commands::real_full) const CPU_REFERENCE_EMBEDDING_LOOKUP_BF16_BACKEND: &str =
    "cpu-reference-embedding-lookup-bf16";
#[allow(dead_code)]
pub(in crate::commands::real_full) const CUDA_REFERENCE_EMBEDDING_LOOKUP_BF16_BACKEND: &str =
    "cuda-reference-embedding-lookup-bf16";
pub(in crate::commands::real_full) const CUDA_REFERENCE_EMBEDDING_LOOKUP_BF16_RESIDENT_WEIGHT_BACKEND:
    &str = "cuda-reference-embedding-lookup-bf16-resident-weight";
pub(in crate::commands::real_full) const CUDA_REFERENCE_EMBEDDING_LOOKUP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND:
    &str = "cuda-reference-embedding-lookup-bf16-preloaded-resident-weight";

#[allow(dead_code)]
pub(in crate::commands::real_full) fn embedding_lookup_rows(
    embedding_rows: &[f32],
    token_ids: &[usize],
    base_token_id: usize,
    row_count: usize,
    hidden_dim: usize,
) -> Result<EmbeddingLookupOutput> {
    let relative_token_ids = validate_embedding_lookup_inputs(
        embedding_rows,
        token_ids,
        base_token_id,
        row_count,
        hidden_dim,
    )?;
    if cuda_reference_kernels_enabled() {
        return cuda_embedding_lookup_rows(
            embedding_rows,
            &relative_token_ids,
            row_count,
            hidden_dim,
        );
    }
    Ok(cpu_embedding_lookup_rows(
        embedding_rows,
        &relative_token_ids,
        hidden_dim,
    ))
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn embedding_lookup_rows_bf16(
    embedding_rows_bf16: &[u8],
    token_ids: &[usize],
    base_token_id: usize,
    row_count: usize,
    hidden_dim: usize,
) -> Result<EmbeddingLookupOutput> {
    let relative_token_ids = validate_embedding_lookup_bf16_inputs(
        embedding_rows_bf16,
        token_ids,
        base_token_id,
        row_count,
        hidden_dim,
    )?;
    if cuda_reference_kernels_enabled() {
        return cuda_embedding_lookup_rows_bf16(
            embedding_rows_bf16,
            &relative_token_ids,
            row_count,
            hidden_dim,
        );
    }
    Ok(cpu_embedding_lookup_rows_bf16(
        embedding_rows_bf16,
        &relative_token_ids,
        hidden_dim,
    ))
}

pub(in crate::commands::real_full) fn embedding_lookup_rows_bf16_resident_weight(
    embedding_name: &str,
    embedding_rows_bf16: &[u8],
    token_ids: &[usize],
    base_token_id: usize,
    row_count: usize,
    hidden_dim: usize,
) -> Result<EmbeddingLookupOutput> {
    validate_resident_weight_name(embedding_name)?;
    let relative_token_ids = validate_embedding_lookup_bf16_inputs(
        embedding_rows_bf16,
        token_ids,
        base_token_id,
        row_count,
        hidden_dim,
    )?;
    if cuda_reference_kernels_enabled() {
        return cuda_embedding_lookup_rows_bf16_resident_weight(
            embedding_name,
            embedding_rows_bf16,
            &relative_token_ids,
            row_count,
            hidden_dim,
        );
    }
    Ok(cpu_embedding_lookup_rows_bf16(
        embedding_rows_bf16,
        &relative_token_ids,
        hidden_dim,
    ))
}

pub(in crate::commands::real_full) fn embedding_lookup_rows_bf16_staged_resident_weight(
    embedding_name: &str,
    token_ids: &[usize],
    base_token_id: usize,
    row_count: usize,
    hidden_dim: usize,
) -> Result<EmbeddingLookupOutput> {
    validate_resident_weight_name(embedding_name)?;
    let relative_token_ids =
        validate_embedding_lookup_relative_inputs(token_ids, base_token_id, row_count, hidden_dim)?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "staged resident BF16 embedding lookup requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1"
        );
    }
    cuda_embedding_lookup_rows_bf16_staged_resident_weight(
        embedding_name,
        &relative_token_ids,
        row_count,
        hidden_dim,
    )
}

pub(in crate::commands::real_full) fn embedding_lookup_bf16_preloaded_resident_weight_device_output(
    embedding_name: &str,
    token_ids: &[usize],
    vocab_size: usize,
    hidden_dim: usize,
) -> Result<DeviceBf16Output> {
    validate_resident_weight_name(embedding_name)?;
    let token_ids = validate_embedding_lookup_full_bf16_inputs(token_ids, vocab_size, hidden_dim)?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "preloaded resident BF16 embedding device-output lookup requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1"
        );
    }
    cuda_embedding_lookup_bf16_preloaded_resident_weight_device_output(
        embedding_name,
        &token_ids,
        vocab_size,
        hidden_dim,
    )
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn validate_embedding_lookup_inputs(
    embedding_rows: &[f32],
    token_ids: &[usize],
    base_token_id: usize,
    row_count: usize,
    hidden_dim: usize,
) -> Result<Vec<u32>> {
    if row_count == 0 || hidden_dim == 0 {
        anyhow::bail!(
            "real full embedding lookup requires non-zero shape, got row_count={row_count} hidden_dim={hidden_dim}"
        );
    }
    if token_ids.is_empty() {
        anyhow::bail!("real full embedding lookup requires at least one token id");
    }
    let expected_values = row_count.checked_mul(hidden_dim).context(
        "real full embedding lookup row-window shape overflows usize while validating coordinator kernel input",
    )?;
    if embedding_rows.len() != expected_values {
        anyhow::bail!(
            "real full embedding lookup row-window length mismatch: expected {} got {}",
            expected_values,
            embedding_rows.len()
        );
    }
    let window_end = base_token_id.checked_add(row_count).context(
        "real full embedding lookup row-window token range overflows usize while validating coordinator kernel input",
    )?;
    token_ids
        .iter()
        .map(|token_id| {
            if *token_id < base_token_id || *token_id >= window_end {
                anyhow::bail!(
                    "real full embedding lookup token_id={} outside loaded row window [{}, {})",
                    token_id,
                    base_token_id,
                    window_end
                );
            }
            let relative = token_id - base_token_id;
            u32::try_from(relative).with_context(|| {
                format!(
                    "real full embedding lookup relative token id {relative} does not fit CUDA u32 index"
                )
            })
        })
        .collect()
}

pub(in crate::commands::real_full) fn validate_embedding_lookup_bf16_inputs(
    embedding_rows_bf16: &[u8],
    token_ids: &[usize],
    base_token_id: usize,
    row_count: usize,
    hidden_dim: usize,
) -> Result<Vec<u32>> {
    if row_count == 0 || hidden_dim == 0 {
        anyhow::bail!(
            "real full BF16 embedding lookup requires non-zero shape, got row_count={row_count} hidden_dim={hidden_dim}"
        );
    }
    if token_ids.is_empty() {
        anyhow::bail!("real full BF16 embedding lookup requires at least one token id");
    }
    let expected_bytes = row_count
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full BF16 embedding lookup row-window shape overflows usize while validating input",
        )?;
    if embedding_rows_bf16.len() != expected_bytes {
        anyhow::bail!(
            "real full BF16 embedding lookup row-window byte length mismatch: expected {} got {}",
            expected_bytes,
            embedding_rows_bf16.len()
        );
    }
    let window_end = base_token_id.checked_add(row_count).context(
        "real full BF16 embedding lookup row-window token range overflows usize while validating input",
    )?;
    token_ids
        .iter()
        .map(|token_id| {
            if *token_id < base_token_id || *token_id >= window_end {
                anyhow::bail!(
                    "real full BF16 embedding lookup token_id={} outside loaded row window [{}, {})",
                    token_id,
                    base_token_id,
                    window_end
                );
            }
            let relative = token_id - base_token_id;
            u32::try_from(relative).with_context(|| {
                format!(
                    "real full BF16 embedding lookup relative token id {relative} does not fit CUDA u32 index"
                )
            })
        })
        .collect()
}

pub(in crate::commands::real_full) fn validate_embedding_lookup_relative_inputs(
    token_ids: &[usize],
    base_token_id: usize,
    row_count: usize,
    hidden_dim: usize,
) -> Result<Vec<u32>> {
    if row_count == 0 || hidden_dim == 0 {
        anyhow::bail!(
            "real full staged resident BF16 embedding lookup requires non-zero shape, got row_count={row_count} hidden_dim={hidden_dim}"
        );
    }
    if token_ids.is_empty() {
        anyhow::bail!(
            "real full staged resident BF16 embedding lookup requires at least one token id"
        );
    }
    let window_end = base_token_id.checked_add(row_count).context(
        "real full staged resident BF16 embedding row-window token range overflows usize while validating input",
    )?;
    token_ids
        .iter()
        .map(|token_id| {
            if *token_id < base_token_id || *token_id >= window_end {
                anyhow::bail!(
                    "real full staged resident BF16 embedding token_id={} outside loaded row window [{}, {})",
                    token_id,
                    base_token_id,
                    window_end
                );
            }
            let relative = token_id - base_token_id;
            u32::try_from(relative).with_context(|| {
                format!(
                    "real full staged resident BF16 embedding relative token id {relative} does not fit CUDA u32 index"
                )
            })
        })
        .collect()
}

pub(in crate::commands::real_full) fn validate_embedding_lookup_full_bf16_inputs(
    token_ids: &[usize],
    vocab_size: usize,
    hidden_dim: usize,
) -> Result<Vec<u32>> {
    if vocab_size == 0 || hidden_dim == 0 {
        anyhow::bail!(
            "real full preloaded BF16 embedding lookup requires non-zero shape, got vocab_size={vocab_size} hidden_dim={hidden_dim}"
        );
    }
    if token_ids.is_empty() {
        anyhow::bail!("real full preloaded BF16 embedding lookup requires at least one token id");
    }
    token_ids
        .iter()
        .map(|token_id| {
            if *token_id >= vocab_size {
                anyhow::bail!(
                    "real full preloaded BF16 embedding lookup token_id={} outside full embedding table [0, {})",
                    token_id,
                    vocab_size
                );
            }
            u32::try_from(*token_id).with_context(|| {
                format!(
                    "real full preloaded BF16 embedding lookup token id {token_id} does not fit CUDA u32 index"
                )
            })
        })
        .collect()
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn cpu_embedding_lookup_rows(
    embedding_rows: &[f32],
    relative_token_ids: &[u32],
    hidden_dim: usize,
) -> EmbeddingLookupOutput {
    let mut values = Vec::with_capacity(relative_token_ids.len() * hidden_dim);
    for relative_token_id in relative_token_ids {
        let row_start = *relative_token_id as usize * hidden_dim;
        values.extend_from_slice(&embedding_rows[row_start..row_start + hidden_dim]);
    }
    EmbeddingLookupOutput {
        values,
        backend: CPU_REFERENCE_EMBEDDING_LOOKUP_BACKEND,
    }
}

pub(in crate::commands::real_full) fn cpu_embedding_lookup_rows_bf16(
    embedding_rows_bf16: &[u8],
    relative_token_ids: &[u32],
    hidden_dim: usize,
) -> EmbeddingLookupOutput {
    let mut values = Vec::with_capacity(relative_token_ids.len() * hidden_dim);
    for relative_token_id in relative_token_ids {
        let row_start = *relative_token_id as usize * hidden_dim;
        for col in 0..hidden_dim {
            values.push(bf16_value(embedding_rows_bf16, row_start + col));
        }
    }
    EmbeddingLookupOutput {
        values,
        backend: CPU_REFERENCE_EMBEDDING_LOOKUP_BF16_BACKEND,
    }
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_embedding_lookup_rows(
    embedding_rows: &[f32],
    relative_token_ids: &[u32],
    row_count: usize,
    hidden_dim: usize,
) -> Result<EmbeddingLookupOutput> {
    let library = cuda_native_library()?;
    let embedding_bytes = std::mem::size_of_val(embedding_rows);
    let token_bytes = std::mem::size_of_val(relative_token_ids);
    let output_bytes = relative_token_ids
        .len()
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context("CUDA embedding lookup output shape overflows usize")?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let embedding_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        embedding_bytes,
        "embedding row window",
    )?;
    let token_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::B,
        token_bytes,
        "embedding token ids",
    )?;
    let output_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        output_bytes,
        "embedding output",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            f32_bytes(embedding_rows),
            "embedding row window",
        )
        .context("copying embedding row window to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::B,
            u32_bytes(relative_token_ids),
            "embedding token ids",
        )
        .context("copying embedding token ids to device")?;
    library
        .cuda_embedding_lookup_f32(
            embedding_buffer,
            token_buffer,
            output_buffer,
            relative_token_ids.len(),
            row_count,
            hidden_dim,
        )
        .context("executing CUDA embedding lookup")?;
    let mut out_bytes = vec![0_u8; output_bytes];
    library
        .copy_d2h(&mut out_bytes, output_buffer)
        .context("copying embedding lookup output to host")?;

    Ok(EmbeddingLookupOutput {
        values: f32_vec_from_bytes(&out_bytes)?,
        backend: CUDA_REFERENCE_EMBEDDING_LOOKUP_BACKEND,
    })
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_embedding_lookup_rows_bf16(
    embedding_rows_bf16: &[u8],
    relative_token_ids: &[u32],
    row_count: usize,
    hidden_dim: usize,
) -> Result<EmbeddingLookupOutput> {
    let library = cuda_native_library()?;
    let token_bytes = std::mem::size_of_val(relative_token_ids);
    let output_bytes = relative_token_ids
        .len()
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 embedding lookup output shape overflows usize")?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let embedding_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        embedding_rows_bf16.len(),
        "BF16 embedding row window",
    )?;
    let token_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::B,
        token_bytes,
        "BF16 embedding token ids",
    )?;
    let output_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        output_bytes,
        "BF16 embedding output",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            embedding_rows_bf16,
            "BF16 embedding row window",
        )
        .context("copying BF16 embedding row window to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::B,
            u32_bytes(relative_token_ids),
            "BF16 embedding token ids",
        )
        .context("copying BF16 embedding token ids to device")?;
    library
        .cuda_embedding_lookup_bf16(
            embedding_buffer,
            token_buffer,
            output_buffer,
            relative_token_ids.len(),
            row_count,
            hidden_dim,
        )
        .context("executing CUDA BF16 embedding lookup")?;
    let mut out_bytes = vec![0_u8; output_bytes];
    library
        .copy_d2h(&mut out_bytes, output_buffer)
        .context("copying BF16 embedding lookup output to host")?;

    Ok(EmbeddingLookupOutput {
        values: bf16_values_to_f32(&out_bytes),
        backend: CUDA_REFERENCE_EMBEDDING_LOOKUP_BF16_BACKEND,
    })
}

pub(in crate::commands::real_full) fn cuda_embedding_lookup_rows_bf16_resident_weight(
    embedding_name: &str,
    embedding_rows_bf16: &[u8],
    relative_token_ids: &[u32],
    row_count: usize,
    hidden_dim: usize,
) -> Result<EmbeddingLookupOutput> {
    let library = cuda_native_library()?;
    let token_bytes = std::mem::size_of_val(relative_token_ids);
    let output_bytes = relative_token_ids
        .len()
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 resident embedding lookup output shape overflows usize")?;
    let embedding_buffer = resident_weight_buffer_from_registry(
        embedding_name,
        embedding_rows_bf16,
        "BF16 resident embedding row window",
    )?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let token_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        token_bytes,
        "BF16 resident embedding token ids",
    )?;
    let output_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        output_bytes,
        "BF16 resident embedding output",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            u32_bytes(relative_token_ids),
            "BF16 resident embedding token ids",
        )
        .context("copying BF16 resident embedding token ids to device")?;
    library
        .cuda_embedding_lookup_bf16(
            embedding_buffer,
            token_buffer,
            output_buffer,
            relative_token_ids.len(),
            row_count,
            hidden_dim,
        )
        .context("executing CUDA BF16 resident embedding lookup")?;
    let mut out_bytes = vec![0_u8; output_bytes];
    library
        .copy_d2h(&mut out_bytes, output_buffer)
        .context("copying BF16 resident embedding lookup output to host")?;

    Ok(EmbeddingLookupOutput {
        values: bf16_values_to_f32(&out_bytes),
        backend: CUDA_REFERENCE_EMBEDDING_LOOKUP_BF16_RESIDENT_WEIGHT_BACKEND,
    })
}

pub(in crate::commands::real_full) fn cuda_embedding_lookup_rows_bf16_staged_resident_weight(
    embedding_name: &str,
    relative_token_ids: &[u32],
    row_count: usize,
    hidden_dim: usize,
) -> Result<EmbeddingLookupOutput> {
    let library = cuda_native_library()?;
    let token_bytes = std::mem::size_of_val(relative_token_ids);
    let embedding_bytes = row_count
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 staged resident embedding row window shape overflows usize")?;
    let output_bytes = relative_token_ids
        .len()
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 staged resident embedding lookup output shape overflows usize")?;
    let embedding_buffer =
        preloaded_resident_weight_device_buffer(embedding_name, embedding_bytes)?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let token_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        token_bytes,
        "BF16 staged resident embedding token ids",
    )?;
    let output_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        output_bytes,
        "BF16 staged resident embedding output",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            u32_bytes(relative_token_ids),
            "BF16 staged resident embedding token ids",
        )
        .context("copying BF16 staged resident embedding token ids to device")?;
    library
        .cuda_embedding_lookup_bf16(
            embedding_buffer,
            token_buffer,
            output_buffer,
            relative_token_ids.len(),
            row_count,
            hidden_dim,
        )
        .context("executing CUDA BF16 staged resident embedding lookup")?;
    let mut out_bytes = vec![0_u8; output_bytes];
    library
        .copy_d2h(&mut out_bytes, output_buffer)
        .context("copying BF16 staged resident embedding lookup output to host")?;

    Ok(EmbeddingLookupOutput {
        values: bf16_values_to_f32(&out_bytes),
        backend: CUDA_REFERENCE_EMBEDDING_LOOKUP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn capture_or_update_input_embedding_lookup_bf16_graph(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    signature: CoordinatorCudaGraphSignature,
    embedding_buffer: GlmrtDeviceBuffer,
    token_buffer: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    vocab_size: usize,
    hidden_dim: usize,
    label: &'static str,
) -> Result<()> {
    if !slot.has_captured_graph(
        CoordinatorCudaGraphProgram::InputEmbeddingLookupBf16,
        signature,
    ) {
        slot.stream_synchronize()
            .with_context(|| format!("synchronizing {label} inputs before graph capture"))?;
        slot.capture_graph(
            library,
            CoordinatorCudaGraphProgram::InputEmbeddingLookupBf16,
            signature,
            |library, cuda_stream, _workspace| unsafe {
                library
                    .cuda_embedding_lookup_bf16_async(
                        embedding_buffer,
                        token_buffer,
                        output_buffer,
                        rows,
                        vocab_size,
                        hidden_dim,
                        cuda_stream,
                    )
                    .with_context(|| format!("capturing async CUDA {label}"))?;
                Ok(())
            },
        )?;
    } else {
        let (graph_raw, exec_raw) = slot
            .captured_graph_raw_handles(
                CoordinatorCudaGraphProgram::InputEmbeddingLookupBf16,
                signature,
            )
            .context(
                "coordinator CUDA graph slot lost captured input embedding lookup graph before update",
            )?;
        unsafe {
            library
                .cuda_graph_update_embedding_lookup_bf16_node(
                    graph_raw,
                    exec_raw,
                    0,
                    embedding_buffer,
                    token_buffer,
                    output_buffer,
                    rows,
                    vocab_size,
                    hidden_dim,
                )
                .with_context(|| format!("updating captured CUDA {label} graph node"))?;
        }
    }
    slot.launch_captured_graph(
        library,
        CoordinatorCudaGraphProgram::InputEmbeddingLookupBf16,
        signature,
    )
}

pub(in crate::commands::real_full) fn cuda_embedding_lookup_bf16_preloaded_resident_weight_device_output_graph_slot(
    graph_key: &CoordinatorGraphKey,
    embedding_name: &str,
    token_ids: &[u32],
    vocab_size: usize,
    hidden_dim: usize,
) -> Result<DeviceBf16Output> {
    let token_bytes = std::mem::size_of_val(token_ids);
    let graph_token_bytes = graph_key
        .row_bucket
        .row_capacity
        .checked_mul(std::mem::size_of::<u32>())
        .context("CUDA BF16 preloaded resident embedding graph token ids shape overflows usize")?;
    if token_bytes > graph_token_bytes {
        anyhow::bail!(
            "CUDA BF16 preloaded resident embedding graph token ids need {token_bytes} bytes, graph slot only has {graph_token_bytes}"
        );
    }
    let embedding_bytes = vocab_size
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 preloaded resident embedding graph table shape overflows usize")?;
    let output_bytes = token_ids
        .len()
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "CUDA BF16 preloaded resident embedding graph device-output lookup shape overflows usize",
        )?;
    let embedding_buffer =
        preloaded_resident_weight_device_buffer(embedding_name, embedding_bytes)?;
    let signature = CoordinatorCudaGraphSignature::embedding_lookup_bf16(
        graph_key.row_bucket.row_capacity,
        vocab_size,
        hidden_dim,
    );

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let token_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            graph_token_bytes,
            "BF16 preloaded resident embedding graph token ids",
        )?;
        let output_buffer = OwnedCoordinatorDeviceBuffer::new(
            library,
            output_bytes,
            "BF16 preloaded resident embedding graph device output",
        )?;

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::A,
                u32_bytes(token_ids),
                "BF16 preloaded resident embedding graph token ids",
                cuda_stream,
            )
            .context("async copying BF16 preloaded resident embedding graph token ids to device")?;
        capture_or_update_input_embedding_lookup_bf16_graph(
            library,
            slot,
            signature,
            embedding_buffer,
            token_buffer,
            output_buffer.buffer,
            token_ids.len(),
            vocab_size,
            hidden_dim,
            "BF16 preloaded resident embedding device-output lookup",
        )?;
        unsafe {
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing CUDA BF16 preloaded resident embedding graph stream")?;
        }

        Ok(DeviceBf16Output {
            buffer: output_buffer,
            bytes: output_bytes,
            rows: token_ids.len(),
            values_per_row: hidden_dim,
            backend: CUDA_REFERENCE_EMBEDDING_LOOKUP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
        })
    })
}

pub(in crate::commands::real_full) fn cuda_embedding_lookup_bf16_preloaded_resident_weight_device_output(
    embedding_name: &str,
    token_ids: &[u32],
    vocab_size: usize,
    hidden_dim: usize,
) -> Result<DeviceBf16Output> {
    if let Some(graph_key) = coord_embedding_graph_key(token_ids.len())? {
        match cuda_embedding_lookup_bf16_preloaded_resident_weight_device_output_graph_slot(
            &graph_key,
            embedding_name,
            token_ids,
            vocab_size,
            hidden_dim,
        ) {
            Ok(output) => return Ok(output),
            Err(_error) => {}
        }
    }

    let library = cuda_native_library()?;
    let token_bytes = std::mem::size_of_val(token_ids);
    let embedding_bytes = vocab_size
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 preloaded resident embedding table shape overflows usize")?;
    let output_bytes = token_ids
        .len()
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "CUDA BF16 preloaded resident embedding device-output lookup shape overflows usize",
        )?;
    let embedding_buffer =
        preloaded_resident_weight_device_buffer(embedding_name, embedding_bytes)?;
    let output_buffer = OwnedCoordinatorDeviceBuffer::new(
        library,
        output_bytes,
        "BF16 preloaded resident embedding device output",
    )?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let token_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        token_bytes,
        "BF16 preloaded resident embedding device-output token ids",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            u32_bytes(token_ids),
            "BF16 preloaded resident embedding device-output token ids",
        )
        .context("copying BF16 preloaded resident embedding device-output token ids to device")?;
    library
        .cuda_embedding_lookup_bf16(
            embedding_buffer,
            token_buffer,
            output_buffer.buffer,
            token_ids.len(),
            vocab_size,
            hidden_dim,
        )
        .context("executing CUDA BF16 preloaded resident embedding device-output lookup")?;
    unsafe {
        library
            .cuda_stream_synchronize(std::ptr::null_mut())
            .context("synchronizing CUDA BF16 preloaded resident embedding device output")?;
    }

    Ok(DeviceBf16Output {
        buffer: output_buffer,
        bytes: output_bytes,
        rows: token_ids.len(),
        values_per_row: hidden_dim,
        backend: CUDA_REFERENCE_EMBEDDING_LOOKUP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
    })
}
