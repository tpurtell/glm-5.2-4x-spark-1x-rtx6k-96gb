use super::*;
use crate::python_graph_capture::{
    launch_python_graph_capture, PythonDeviceBufferArg, PythonGraphCaptureLaunch, PythonKernelArg,
};
use anyhow::{Context, Result};
use glmrt_core::{
    CoordinatorGraphInstancePlan, CoordinatorGraphKey, CoordinatorGraphShape, KvCacheDType,
    LayerId, LayerWaveMode, COORDINATOR_GRAPH_INSTANCE_COUNT, GLM52_DSA_INDEX_HEAD_DIM,
    GLM52_FIRST_K_DENSE_REPLACE, GLM52_HIDDEN_BF16_BYTES, GLM52_HIDDEN_SIZE,
    GLM52_MLA_FP8_DS_BYTES_PER_TOKEN, GLM52_MLA_KV_LORA_RANK, GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN,
    GLM52_MLA_QK_ROPE_HEAD_DIM, GLM52_NUM_HIDDEN_LAYERS, GLM52_ROUTED_SCALING_FACTOR, GLM52_TOP_K,
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

const CUDA_MLA_DECODE_QUERY_PROJECTION_BF16_BACKEND: &str =
    "cuda-mla-decode-query-projection-bf16-graph";
const CUDA_MLA_DECODE_KV_COMMIT_BF16_BACKEND: &str =
    "cuda-mla-decode-kv-project-prepare-pack-commit-bf16-graph";

#[derive(Clone, Copy)]
pub(in crate::commands::real_full) struct MlaDecodeKvDsaProjectionWeights {
    pub(in crate::commands::real_full) wk: GlmrtDeviceBuffer,
    pub(in crate::commands::real_full) norm_weight: GlmrtDeviceBuffer,
    pub(in crate::commands::real_full) norm_bias: GlmrtDeviceBuffer,
}

#[derive(Clone, Copy)]
struct MlaDecodeKvCommitBuffers {
    hidden: GlmrtDeviceBuffer,
    input_norm_weight: GlmrtDeviceBuffer,
    normalized_hidden: GlmrtDeviceBuffer,
    kv_a_weight: GlmrtDeviceBuffer,
    kv_projected: GlmrtDeviceBuffer,
    positions: GlmrtDeviceBuffer,
    physical_slots: GlmrtDeviceBuffer,
    kv_norm_weight: GlmrtDeviceBuffer,
    prepared: GlmrtDeviceBuffer,
    packed: Option<GlmrtDeviceBuffer>,
    dsa_weights: Option<MlaDecodeKvDsaProjectionWeights>,
    dsa_projected: Option<GlmrtDeviceBuffer>,
    dsa_normalized: Option<GlmrtDeviceBuffer>,
    dsa_index_k_cache: Option<GlmrtDeviceBuffer>,
    dsa_index_k_cache_tokens: usize,
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn mla_decode_kv_commit_bf16_device_output(
    layer_id: usize,
    hidden: GlmrtDeviceBuffer,
    input_norm_weight: GlmrtDeviceBuffer,
    kv_a_weight: GlmrtDeviceBuffer,
    kv_norm_weight: GlmrtDeviceBuffer,
    dsa_weights: Option<MlaDecodeKvDsaProjectionWeights>,
    cache_row: GlmrtDeviceBuffer,
    dsa_cache_row: Option<GlmrtDeviceBuffer>,
    dsa_index_k_cache: Option<GlmrtDeviceBuffer>,
    dsa_index_k_cache_tokens: usize,
    attention_ready_row: Option<GlmrtDeviceBuffer>,
    attention_ready_row_fp8: bool,
    cache_dtype: KvCacheDType,
    position: u32,
    physical_slot: u32,
    hidden_dim: usize,
    eps: f32,
    theta: f32,
) -> Result<DeviceBf16Output> {
    anyhow::ensure!(
        hidden_dim > 0,
        "decode KV commit hidden dimension must be nonzero"
    );
    anyhow::ensure!(
        eps.is_finite() && eps > 0.0 && theta.is_finite() && theta > 0.0,
        "decode KV commit epsilon and RoPE theta must be finite and positive"
    );
    let bf16_bytes = std::mem::size_of::<u16>();
    let kv_width = GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM;
    let hidden_bytes = hidden_dim
        .checked_mul(bf16_bytes)
        .context("decode KV commit hidden bytes overflow")?;
    let kv_bf16_bytes = kv_width
        .checked_mul(bf16_bytes)
        .context("decode KV commit projected bytes overflow")?;
    let dsa_bytes = dsa_weights
        .is_some()
        .then_some(
            GLM52_DSA_INDEX_HEAD_DIM
                .checked_mul(bf16_bytes)
                .context("decode KV commit DSA bytes overflow")?,
        )
        .unwrap_or(0);
    let packed_main_bytes = match cache_dtype {
        KvCacheDType::Bf16 => kv_bf16_bytes,
        KvCacheDType::Fp8 => GLM52_MLA_FP8_DS_BYTES_PER_TOKEN,
        KvCacheDType::Nvfp4 => GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN,
        other => anyhow::bail!(
            "decode KV commit requires BF16, FP8, or NVFP4 cache storage, got {}",
            other.label()
        ),
    };
    let expected_buffers = [
        ("hidden", hidden, hidden_bytes),
        ("input norm weight", input_norm_weight, hidden_bytes),
        (
            "kv_a weight",
            kv_a_weight,
            kv_width
                .checked_mul(hidden_dim)
                .and_then(|values| values.checked_mul(bf16_bytes))
                .context("decode KV commit kv_a weight bytes overflow")?,
        ),
        (
            "KV norm weight",
            kv_norm_weight,
            GLM52_MLA_KV_LORA_RANK * bf16_bytes,
        ),
        ("cache row", cache_row, packed_main_bytes),
    ];
    for (label, buffer, expected_bytes) in expected_buffers {
        anyhow::ensure!(
            !buffer.ptr.is_null() && buffer.bytes >= expected_bytes,
            "decode KV commit {label} buffer has {} bytes, expected at least {expected_bytes}",
            buffer.bytes
        );
        anyhow::ensure!(
            buffer.device_id == hidden.device_id,
            "decode KV commit {label} buffer is on CUDA device {}, expected {}",
            buffer.device_id,
            hidden.device_id
        );
    }
    anyhow::ensure!(
        dsa_cache_row.is_some() == dsa_weights.is_some(),
        "decode KV commit DSA cache row presence does not match DSA weights"
    );
    anyhow::ensure!(
        dsa_index_k_cache.is_some() == dsa_weights.is_some(),
        "decode KV commit direct DSA cache presence does not match DSA weights"
    );
    if let Some(dsa_cache_row) = dsa_cache_row {
        anyhow::ensure!(
            !dsa_cache_row.ptr.is_null() && dsa_cache_row.bytes >= dsa_bytes,
            "decode KV commit DSA cache row has {} bytes, expected at least {dsa_bytes}",
            dsa_cache_row.bytes
        );
        anyhow::ensure!(
            dsa_cache_row.device_id == hidden.device_id,
            "decode KV commit DSA cache row is on CUDA device {}, expected {}",
            dsa_cache_row.device_id,
            hidden.device_id
        );
    }
    if let Some(dsa_index_k_cache) = dsa_index_k_cache {
        anyhow::ensure!(
            dsa_index_k_cache_tokens > 0,
            "decode KV commit direct DSA cache requires positive token capacity"
        );
        let direct_cache_pages = dsa_index_k_cache_tokens
            .checked_add(glmrt_ffi::GLMRT_CUDA_GLM_DSA_PAGE_SIZE - 1)
            .context("decode KV commit direct DSA cache page rounding overflow")?
            / glmrt_ffi::GLMRT_CUDA_GLM_DSA_PAGE_SIZE;
        let direct_cache_bytes = direct_cache_pages
            .checked_mul(glmrt_ffi::GLMRT_CUDA_GLM_DSA_PACKED_PAGE_BYTES)
            .context("decode KV commit direct DSA cache bytes overflow")?;
        anyhow::ensure!(
            !dsa_index_k_cache.ptr.is_null() && dsa_index_k_cache.bytes >= direct_cache_bytes,
            "decode KV commit direct DSA cache has {} bytes, expected at least {direct_cache_bytes}",
            dsa_index_k_cache.bytes
        );
        anyhow::ensure!(
            dsa_index_k_cache.device_id == hidden.device_id,
            "decode KV commit direct DSA cache is on CUDA device {}, expected {}",
            dsa_index_k_cache.device_id,
            hidden.device_id
        );
    }
    anyhow::ensure!(
        !attention_ready_row_fp8 || attention_ready_row.is_some(),
        "decode KV FP8 attention-ready mode requires a destination row"
    );
    if let Some(attention_ready_row) = attention_ready_row {
        let attention_ready_bytes = if attention_ready_row_fp8 {
            GLM52_MLA_FP8_DS_BYTES_PER_TOKEN
        } else {
            kv_bf16_bytes
        };
        anyhow::ensure!(
            !attention_ready_row.ptr.is_null()
                && attention_ready_row.bytes >= attention_ready_bytes,
            "decode KV attention-ready row has {} bytes, expected at least {attention_ready_bytes}",
            attention_ready_row.bytes
        );
        anyhow::ensure!(
            attention_ready_row.device_id == hidden.device_id,
            "decode KV attention-ready row is on CUDA device {}, expected {}",
            attention_ready_row.device_id,
            hidden.device_id
        );
    }
    if let Some(dsa) = dsa_weights {
        for (label, buffer, expected_bytes) in [
            (
                "DSA wk weight",
                dsa.wk,
                GLM52_DSA_INDEX_HEAD_DIM
                    .checked_mul(hidden_dim)
                    .and_then(|values| values.checked_mul(bf16_bytes))
                    .context("decode KV commit DSA wk weight bytes overflow")?,
            ),
            ("DSA norm weight", dsa.norm_weight, dsa_bytes),
            ("DSA norm bias", dsa.norm_bias, dsa_bytes),
        ] {
            anyhow::ensure!(
                !buffer.ptr.is_null() && buffer.bytes >= expected_bytes,
                "decode KV commit {label} buffer has {} bytes, expected at least {expected_bytes}",
                buffer.bytes
            );
            anyhow::ensure!(
                buffer.device_id == hidden.device_id,
                "decode KV commit {label} buffer is on CUDA device {}, expected {}",
                buffer.device_id,
                hidden.device_id
            );
        }
    }

    let graph_key = coord_attention_graph_key_for_layer_rows(layer_id, 1)?;
    let cache_format = match cache_dtype {
        KvCacheDType::Bf16 => 0,
        KvCacheDType::Fp8 => 1,
        KvCacheDType::Nvfp4 => 2,
        _ => unreachable!("decode KV cache dtype validated above"),
    };
    let signature = CoordinatorCudaGraphSignature::mla_decode_kv_commit_bf16(
        hidden_dim,
        GLM52_MLA_KV_LORA_RANK,
        GLM52_MLA_QK_ROPE_HEAD_DIM,
        dsa_weights.map_or(0, |_| GLM52_DSA_INDEX_HEAD_DIM),
        cache_format,
        eps,
        theta,
    );
    with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
        // A/B are also used later by query projection. Allocate them at that
        // path's full width so the first query cannot invalidate this graph.
        let query_scratch_bytes = GLM52_Q_LORA_RANK * bf16_bytes;
        let kv_projected = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            kv_bf16_bytes.max(query_scratch_bytes),
            "decode KV projected scratch",
        )?;
        let dsa_projected = dsa_weights
            .map(|_| {
                slot.buffer(
                    library,
                    CoordinatorCudaScratchSlot::B,
                    dsa_bytes.max(query_scratch_bytes),
                    "decode KV DSA projected scratch",
                )
            })
            .transpose()?;
        let position_pair = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            2 * std::mem::size_of::<u32>(),
            "decode KV logical and physical positions",
        )?;
        let positions = device_buffer_byte_view(
            position_pair,
            0,
            std::mem::size_of::<u32>(),
            "decode KV RoPE position",
        )?;
        let physical_slots = device_buffer_byte_view(
            position_pair,
            std::mem::size_of::<u32>(),
            std::mem::size_of::<u32>(),
            "decode KV physical slot",
        )?;
        let prepared = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            kv_bf16_bytes,
            "decode KV prepared scratch",
        )?;
        let packed = (!matches!(cache_dtype, KvCacheDType::Bf16))
            .then(|| {
                slot.buffer(
                    library,
                    CoordinatorCudaScratchSlot::E,
                    packed_main_bytes,
                    "decode KV packed scratch",
                )
            })
            .transpose()?;
        let dsa_normalized = dsa_weights
            .map(|_| {
                slot.buffer(
                    library,
                    CoordinatorCudaScratchSlot::F,
                    dsa_bytes,
                    "decode KV DSA normalized scratch",
                )
            })
            .transpose()?;
        let stable_hidden_bytes = hidden_bytes
            .checked_mul(2)
            .context("decode KV stable hidden staging bytes overflow")?;
        let stable_hidden_rows = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::G,
            stable_hidden_bytes,
            "decode KV stable hidden rows",
        )?;
        let staged_hidden = device_buffer_byte_view(
            stable_hidden_rows,
            0,
            hidden_bytes,
            "decode KV stable hidden input",
        )?;
        let staged_normalized_hidden = device_buffer_byte_view(
            stable_hidden_rows,
            hidden_bytes,
            hidden_bytes,
            "decode KV stable normalized hidden",
        )?;
        let normalized_hidden = OwnedCoordinatorDeviceBuffer::new(
            library,
            hidden_bytes,
            "fused MLA decode KV normalized hidden",
        )?;
        let buffers = MlaDecodeKvCommitBuffers {
            hidden: staged_hidden,
            input_norm_weight,
            normalized_hidden: staged_normalized_hidden,
            kv_a_weight,
            kv_projected,
            positions,
            physical_slots,
            kv_norm_weight,
            prepared,
            packed,
            dsa_weights,
            dsa_projected,
            dsa_normalized,
            dsa_index_k_cache,
            dsa_index_k_cache_tokens,
        };
        let capture_identity =
            mla_decode_kv_commit_capture_identity(buffers, cache_format, hidden_dim, eps, theta);
        let stream = slot.stream_ptr();
        unsafe {
            library
                .copy_d2d_async(staged_hidden, hidden, hidden_bytes, stream)
                .context("staging decode KV hidden row at a stable graph address")?;
        }
        let mut position_bytes = [0_u8; 2 * std::mem::size_of::<u32>()];
        position_bytes[..std::mem::size_of::<u32>()].copy_from_slice(&position.to_ne_bytes());
        position_bytes[std::mem::size_of::<u32>()..].copy_from_slice(&physical_slot.to_ne_bytes());
        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::C,
                &position_bytes,
                "decode KV logical and physical positions",
                stream,
            )
            .context("staging decode KV logical and physical positions")?;
        let program = CoordinatorCudaGraphProgram::LayerMlaDecodeKvCommitBf16;
        if !slot.has_captured_graph_identity(program, signature, capture_identity) {
            unsafe {
                enqueue_mla_decode_kv_commit(
                    library,
                    stream,
                    buffers,
                    cache_dtype,
                    hidden_dim,
                    eps,
                    theta,
                )?;
            }
            slot.stream_synchronize()
                .context("warming fused MLA decode KV commit")?;
        }
        slot.capture_or_update_graph_exec(
            library,
            program,
            signature,
            capture_identity,
            |library, stream, _workspace| unsafe {
                enqueue_mla_decode_kv_commit(
                    library,
                    stream,
                    buffers,
                    cache_dtype,
                    hidden_dim,
                    eps,
                    theta,
                )
            },
        )?;
        slot.launch_captured_graph_identity(library, program, signature, capture_identity)?;

        let write_src = packed.unwrap_or(prepared);
        unsafe {
            library
                .copy_d2d_async(
                    normalized_hidden.buffer,
                    staged_normalized_hidden,
                    hidden_bytes,
                    stream,
                )
                .context("copying stable decode KV normalized hidden to handoff buffer")?;
            library
                .copy_d2d_async(
                    device_buffer_byte_view(
                        cache_row,
                        0,
                        packed_main_bytes,
                        "decode KV cache main destination",
                    )?,
                    write_src,
                    packed_main_bytes,
                    stream,
                )
                .context("committing decode KV main row asynchronously")?;
            if let Some(attention_ready_row) = attention_ready_row {
                if attention_ready_row_fp8 {
                    library
                        .cuda_mla_kv_pack_fp8_ds_mla_async(
                            prepared,
                            attention_ready_row,
                            1,
                            kv_bf16_bytes,
                            GLM52_MLA_FP8_DS_BYTES_PER_TOKEN,
                            stream,
                        )
                        .context("packing decode KV attention-ready FP8 row asynchronously")?;
                } else {
                    library
                        .copy_d2d_async(attention_ready_row, prepared, kv_bf16_bytes, stream)
                        .context("retaining decode KV attention-ready BF16 row asynchronously")?;
                }
            }
            if let Some(dsa_normalized) = dsa_normalized {
                library
                    .copy_d2d_async(
                        dsa_cache_row.context("decode KV cache DSA destination missing")?,
                        dsa_normalized,
                        dsa_bytes,
                        stream,
                    )
                    .context("committing decode KV DSA row asynchronously")?;
            }
        }
        Ok(DeviceBf16Output {
            buffer: normalized_hidden,
            bytes: hidden_bytes,
            rows: 1,
            values_per_row: hidden_dim,
            backend: CUDA_MLA_DECODE_KV_COMMIT_BF16_BACKEND,
        })
    })
}

#[allow(clippy::too_many_arguments)]
unsafe fn enqueue_mla_decode_kv_commit(
    library: &'static NativeLibrary,
    stream: *mut c_void,
    buffers: MlaDecodeKvCommitBuffers,
    cache_dtype: KvCacheDType,
    hidden_dim: usize,
    eps: f32,
    theta: f32,
) -> Result<()> {
    let kv_width = GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM;
    let kv_stride_bytes = kv_width * std::mem::size_of::<u16>();
    unsafe {
        library.cuda_rmsnorm_bf16_async(
            buffers.hidden,
            buffers.input_norm_weight,
            buffers.normalized_hidden,
            1,
            hidden_dim as i32,
            eps,
            stream,
        )?;
        library.cuda_linear_bf16_cublas_async(
            buffers.normalized_hidden,
            buffers.kv_a_weight,
            None,
            buffers.kv_projected,
            1,
            hidden_dim,
            kv_width,
            stream,
        )?;
        if let (Some(dsa), Some(projected), Some(normalized)) = (
            buffers.dsa_weights,
            buffers.dsa_projected,
            buffers.dsa_normalized,
        ) {
            library.cuda_linear_bf16_cublas_async(
                buffers.normalized_hidden,
                dsa.wk,
                None,
                projected,
                1,
                hidden_dim,
                GLM52_DSA_INDEX_HEAD_DIM,
                stream,
            )?;
            library.cuda_layernorm_affine_bf16_async(
                projected,
                dsa.norm_weight,
                dsa.norm_bias,
                normalized,
                1,
                GLM52_DSA_INDEX_HEAD_DIM as i32,
                eps,
                stream,
            )?;
            library.cuda_glm_dsa_index_k_pack_b12x_async(
                normalized,
                buffers.positions,
                buffers.physical_slots,
                buffers
                    .dsa_index_k_cache
                    .context("decode KV commit direct DSA cache missing")?,
                1,
                buffers.dsa_index_k_cache_tokens,
                GLM52_DSA_INDEX_HEAD_DIM * std::mem::size_of::<u16>(),
                theta,
                stream,
            )?;
        }
        library.cuda_mla_kv_prepare_bf16_async(
            buffers.kv_projected,
            buffers.positions,
            buffers.kv_norm_weight,
            buffers.prepared,
            1,
            kv_stride_bytes,
            kv_stride_bytes,
            eps,
            theta,
            stream,
        )?;
        match cache_dtype {
            KvCacheDType::Bf16 => {}
            KvCacheDType::Fp8 => library.cuda_mla_kv_pack_fp8_ds_mla_async(
                buffers.prepared,
                buffers
                    .packed
                    .context("FP8 decode KV commit packed buffer missing")?,
                1,
                kv_stride_bytes,
                GLM52_MLA_FP8_DS_BYTES_PER_TOKEN,
                stream,
            )?,
            KvCacheDType::Nvfp4 => library.cuda_mla_kv_pack_mxfp4_ds_mla_async(
                buffers.prepared,
                buffers
                    .packed
                    .context("NVFP4 decode KV commit packed buffer missing")?,
                1,
                kv_stride_bytes,
                GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN,
                stream,
            )?,
            _ => unreachable!("decode KV cache dtype validated before enqueue"),
        }
    }
    Ok(())
}

fn mla_decode_kv_commit_capture_identity(
    buffers: MlaDecodeKvCommitBuffers,
    cache_format: usize,
    hidden_dim: usize,
    eps: f32,
    theta: f32,
) -> usize {
    graph_capture_identity(&[
        buffers.hidden.ptr as usize,
        buffers.input_norm_weight.ptr as usize,
        buffers.normalized_hidden.ptr as usize,
        buffers.kv_a_weight.ptr as usize,
        buffers.kv_projected.ptr as usize,
        buffers.positions.ptr as usize,
        buffers.kv_norm_weight.ptr as usize,
        buffers.prepared.ptr as usize,
        buffers.packed.map_or(0, |buffer| buffer.ptr as usize),
        buffers
            .dsa_weights
            .map_or(0, |weights| weights.wk.ptr as usize),
        buffers
            .dsa_weights
            .map_or(0, |weights| weights.norm_weight.ptr as usize),
        buffers
            .dsa_weights
            .map_or(0, |weights| weights.norm_bias.ptr as usize),
        buffers
            .dsa_projected
            .map_or(0, |buffer| buffer.ptr as usize),
        buffers
            .dsa_normalized
            .map_or(0, |buffer| buffer.ptr as usize),
        buffers
            .dsa_index_k_cache
            .map_or(0, |buffer| buffer.ptr as usize),
        buffers.dsa_index_k_cache_tokens,
        cache_format,
        hidden_dim,
        eps.to_bits() as usize,
        theta.to_bits() as usize,
    ])
}

#[derive(Clone, Copy)]
struct MlaDecodeQueryProjectionBuffers {
    normalized_hidden: GlmrtDeviceBuffer,
    q_a_weight: Option<GlmrtDeviceBuffer>,
    q_a_w8a16: Option<CoordinatorW8a16ProjectionBuffers>,
    q_a_projected: GlmrtDeviceBuffer,
    q_a_norm_weight: GlmrtDeviceBuffer,
    q_a_normalized: GlmrtDeviceBuffer,
    q_b_weight: Option<GlmrtDeviceBuffer>,
    q_b_w4a16: Option<GlmrtB12xCoordinatorW4a16Buffers>,
    q_b_w8a16: Option<CoordinatorW8a16ProjectionBuffers>,
    q_projected: GlmrtDeviceBuffer,
    dsa: Option<MlaDecodeQueryDsaProjectionBuffers>,
}

#[derive(Clone, Copy)]
struct MlaDecodeQueryDsaProjectionBuffers {
    wq_b_weight: GlmrtDeviceBuffer,
    weights_proj_weight: GlmrtDeviceBuffer,
    query_projected: GlmrtDeviceBuffer,
    weights_projected: GlmrtDeviceBuffer,
    query_dim: usize,
    heads: usize,
}

#[derive(Clone, Copy)]
struct MlaDecodeQueryDsaProjectionConfig {
    wq_b_weight: GlmrtDeviceBuffer,
    weights_proj_weight: GlmrtDeviceBuffer,
    query_dim: usize,
    heads: usize,
}

struct MlaDecodeQueryProjectionOutputs {
    q_projected: DeviceBf16Output,
    dsa_query_projected: Option<DeviceBf16Output>,
    dsa_weights_projected: Option<DeviceBf16Output>,
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn mla_decode_query_projection_bf16_device_output(
    layer_id: usize,
    normalized_hidden: GlmrtDeviceBuffer,
    q_a_weight_name: &str,
    q_a_weight: Option<GlmrtDeviceBuffer>,
    q_a_norm_weight: GlmrtDeviceBuffer,
    q_b_weight_name: &str,
    q_b_weight: Option<GlmrtDeviceBuffer>,
    rows: usize,
    hidden_dim: usize,
    q_lora_rank: usize,
    q_output_dim: usize,
    eps: f32,
) -> Result<DeviceBf16Output> {
    let outputs = mla_decode_query_projection_bf16_device_outputs_impl(
        layer_id,
        normalized_hidden,
        q_a_weight_name,
        q_a_weight,
        q_a_norm_weight,
        q_b_weight_name,
        q_b_weight,
        None,
        rows,
        hidden_dim,
        q_lora_rank,
        q_output_dim,
        eps,
    )?;
    debug_assert!(outputs.dsa_query_projected.is_none());
    debug_assert!(outputs.dsa_weights_projected.is_none());
    Ok(outputs.q_projected)
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn mla_decode_query_dsa_projection_bf16_device_outputs(
    layer_id: usize,
    normalized_hidden: GlmrtDeviceBuffer,
    q_a_weight_name: &str,
    q_a_weight: Option<GlmrtDeviceBuffer>,
    q_a_norm_weight: GlmrtDeviceBuffer,
    q_b_weight_name: &str,
    q_b_weight: Option<GlmrtDeviceBuffer>,
    dsa_wq_b_weight: GlmrtDeviceBuffer,
    dsa_weights_proj_weight: GlmrtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    q_lora_rank: usize,
    q_output_dim: usize,
    dsa_query_dim: usize,
    dsa_heads: usize,
    eps: f32,
) -> Result<(DeviceBf16Output, DeviceBf16Output, DeviceBf16Output)> {
    let outputs = mla_decode_query_projection_bf16_device_outputs_impl(
        layer_id,
        normalized_hidden,
        q_a_weight_name,
        q_a_weight,
        q_a_norm_weight,
        q_b_weight_name,
        q_b_weight,
        Some(MlaDecodeQueryDsaProjectionConfig {
            wq_b_weight: dsa_wq_b_weight,
            weights_proj_weight: dsa_weights_proj_weight,
            query_dim: dsa_query_dim,
            heads: dsa_heads,
        }),
        rows,
        hidden_dim,
        q_lora_rank,
        q_output_dim,
        eps,
    )?;
    Ok((
        outputs.q_projected,
        outputs
            .dsa_query_projected
            .context("fused MLA decode DSA query output is missing")?,
        outputs
            .dsa_weights_projected
            .context("fused MLA decode DSA weights output is missing")?,
    ))
}

#[allow(clippy::too_many_arguments)]
fn mla_decode_query_projection_bf16_device_outputs_impl(
    layer_id: usize,
    normalized_hidden: GlmrtDeviceBuffer,
    q_a_weight_name: &str,
    q_a_weight: Option<GlmrtDeviceBuffer>,
    q_a_norm_weight: GlmrtDeviceBuffer,
    q_b_weight_name: &str,
    q_b_weight: Option<GlmrtDeviceBuffer>,
    dsa: Option<MlaDecodeQueryDsaProjectionConfig>,
    rows: usize,
    hidden_dim: usize,
    q_lora_rank: usize,
    q_output_dim: usize,
    eps: f32,
) -> Result<MlaDecodeQueryProjectionOutputs> {
    anyhow::ensure!(
        rows == 1,
        "fused MLA decode query projection requires exactly one row, got {rows}"
    );
    anyhow::ensure!(
        hidden_dim > 0 && q_lora_rank > 0 && q_output_dim > 0,
        "fused MLA decode query projection dimensions must be nonzero"
    );
    anyhow::ensure!(
        eps.is_finite() && eps > 0.0,
        "fused MLA decode query projection epsilon must be finite and positive"
    );
    let bf16_bytes = std::mem::size_of::<u16>();
    let normalized_bytes = rows
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("fused MLA decode normalized hidden bytes overflow")?;
    let q_a_bytes = rows
        .checked_mul(q_lora_rank)
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("fused MLA decode q_a bytes overflow")?;
    let q_output_bytes = rows
        .checked_mul(q_output_dim)
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("fused MLA decode q_b bytes overflow")?;
    let dsa_query_bytes = dsa
        .map(|dsa| {
            rows.checked_mul(dsa.query_dim)
                .and_then(|values| values.checked_mul(bf16_bytes))
                .context("fused MLA decode DSA query bytes overflow")
        })
        .transpose()?
        .unwrap_or(0);
    let dsa_weights_bytes = dsa
        .map(|dsa| {
            rows.checked_mul(dsa.heads)
                .and_then(|values| values.checked_mul(bf16_bytes))
                .context("fused MLA decode DSA weights bytes overflow")
        })
        .transpose()?
        .unwrap_or(0);
    let q_a_weight_bytes = q_lora_rank
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("fused MLA decode q_a weight bytes overflow")?;
    for (label, buffer, expected_bytes) in [
        ("normalized hidden", normalized_hidden, normalized_bytes),
        ("q_a norm weight", q_a_norm_weight, q_lora_rank * bf16_bytes),
    ] {
        anyhow::ensure!(
            !buffer.ptr.is_null() && buffer.bytes >= expected_bytes,
            "fused MLA decode {label} buffer has {} bytes, expected at least {expected_bytes}",
            buffer.bytes
        );
        anyhow::ensure!(
            buffer.device_id == normalized_hidden.device_id,
            "fused MLA decode {label} buffer is on CUDA device {}, expected {}",
            buffer.device_id,
            normalized_hidden.device_id
        );
    }
    if let Some(q_a_weight) = q_a_weight {
        anyhow::ensure!(
            !q_a_weight.ptr.is_null() && q_a_weight.bytes >= q_a_weight_bytes,
            "fused MLA decode q_a weight buffer has {} bytes, expected at least {q_a_weight_bytes}",
            q_a_weight.bytes
        );
        anyhow::ensure!(
            q_a_weight.device_id == normalized_hidden.device_id,
            "fused MLA decode q_a weight buffer is on CUDA device {}, expected {}",
            q_a_weight.device_id,
            normalized_hidden.device_id
        );
    }
    if let Some(q_b_weight) = q_b_weight {
        let q_b_weight_bytes = q_output_dim
            .checked_mul(q_lora_rank)
            .and_then(|values| values.checked_mul(bf16_bytes))
            .context("fused MLA decode q_b weight bytes overflow")?;
        anyhow::ensure!(
            !q_b_weight.ptr.is_null() && q_b_weight.bytes >= q_b_weight_bytes,
            "fused MLA decode q_b weight buffer has {} bytes, expected at least {q_b_weight_bytes}",
            q_b_weight.bytes
        );
        anyhow::ensure!(
            q_b_weight.device_id == normalized_hidden.device_id,
            "fused MLA decode q_b weight buffer is on CUDA device {}, expected {}",
            q_b_weight.device_id,
            normalized_hidden.device_id
        );
    }
    if let Some(dsa) = dsa {
        anyhow::ensure!(
            dsa.query_dim > 0 && dsa.heads > 0,
            "fused MLA decode DSA projection dimensions must be nonzero"
        );
        for (label, buffer, expected_bytes) in [
            (
                "DSA wq_b weight",
                dsa.wq_b_weight,
                dsa.query_dim
                    .checked_mul(q_lora_rank)
                    .and_then(|values| values.checked_mul(bf16_bytes))
                    .context("fused MLA decode DSA wq_b weight bytes overflow")?,
            ),
            (
                "DSA weights projection weight",
                dsa.weights_proj_weight,
                dsa.heads
                    .checked_mul(hidden_dim)
                    .and_then(|values| values.checked_mul(bf16_bytes))
                    .context("fused MLA decode DSA weights projection bytes overflow")?,
            ),
        ] {
            anyhow::ensure!(
                !buffer.ptr.is_null() && buffer.bytes >= expected_bytes,
                "fused MLA decode {label} buffer has {} bytes, expected at least {expected_bytes}",
                buffer.bytes
            );
            anyhow::ensure!(
                buffer.device_id == normalized_hidden.device_id,
                "fused MLA decode {label} buffer is on CUDA device {}, expected {}",
                buffer.device_id,
                normalized_hidden.device_id
            );
        }
    }

    let w8a16_q_a_enabled = coordinator_w8a16_q_a_decode_enabled();
    let q_a_w8a16 = w8a16_q_a_enabled
        .then(|| preloaded_coordinator_w8a16_projection(q_a_weight_name, hidden_dim, q_lora_rank))
        .transpose()?;
    anyhow::ensure!(
        q_a_w8a16.is_some() ^ q_a_weight.is_some(),
        "fused MLA decode q_a projection requires exactly one BF16 or W8A16 weight"
    );
    let w4a16_q_b_enabled = coordinator_w4a16_q_b_decode_enabled();
    let w8a16_q_b_enabled = coordinator_w8a16_q_b_decode_enabled();
    anyhow::ensure!(
        !(w4a16_q_b_enabled && w8a16_q_b_enabled),
        "coordinator Q-B projection cannot enable W4A16 and W8A16 simultaneously"
    );
    let q_b_w4a16 = w4a16_q_b_enabled
        .then(|| preloaded_coordinator_w4a16_projection(q_b_weight_name, q_lora_rank, q_output_dim))
        .transpose()?;
    let q_b_w8a16 = w8a16_q_b_enabled
        .then(|| preloaded_coordinator_w8a16_projection(q_b_weight_name, q_lora_rank, q_output_dim))
        .transpose()?;
    anyhow::ensure!(
        q_b_w4a16.is_some() || q_b_w8a16.is_some() || q_b_weight.is_some(),
        "fused MLA decode q_b projection has no BF16, W4A16, or W8A16 weight"
    );
    let graph_key = coord_attention_graph_key_for_layer_rows(layer_id, rows)?;
    let signature = CoordinatorCudaGraphSignature::mla_decode_query_projection_bf16(
        rows,
        hidden_dim,
        q_lora_rank,
        q_output_dim,
        eps,
        usize::from(q_a_w8a16.is_some()),
        if q_b_w4a16.is_some() {
            1
        } else if q_b_w8a16.is_some() {
            2
        } else {
            0
        },
        dsa.map_or(0, |dsa| dsa.query_dim),
        dsa.map_or(0, |dsa| dsa.heads),
    );
    with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
        let q_a_projected = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            q_a_bytes,
            "fused MLA decode q_a projection",
        )?;
        let q_a_normalized = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::B,
            q_a_bytes,
            "fused MLA decode q_a normalization",
        )?;
        let q_projected = OwnedCoordinatorDeviceBuffer::new(
            library,
            q_output_bytes,
            "fused MLA decode q_b projection",
        )?;
        let dsa_query_projected = dsa
            .map(|_| {
                OwnedCoordinatorDeviceBuffer::new(
                    library,
                    dsa_query_bytes,
                    "fused MLA decode DSA query projection",
                )
            })
            .transpose()?;
        let dsa_weights_projected = dsa
            .map(|_| {
                OwnedCoordinatorDeviceBuffer::new(
                    library,
                    dsa_weights_bytes,
                    "fused MLA decode DSA weights projection",
                )
            })
            .transpose()?;
        let q_b_w4a16 = q_b_w4a16
            .map(|projection| {
                coordinator_w4a16_launch_buffers(
                    library,
                    slot,
                    projection,
                    q_a_normalized,
                    q_projected.buffer,
                    CoordinatorCudaScratchSlot::R,
                )
            })
            .transpose()?;
        let buffers = MlaDecodeQueryProjectionBuffers {
            normalized_hidden,
            q_a_weight,
            q_a_w8a16,
            q_a_projected,
            q_a_norm_weight,
            q_a_normalized,
            q_b_weight,
            q_b_w4a16,
            q_b_w8a16,
            q_projected: q_projected.buffer,
            dsa: dsa.map(|dsa| MlaDecodeQueryDsaProjectionBuffers {
                wq_b_weight: dsa.wq_b_weight,
                weights_proj_weight: dsa.weights_proj_weight,
                query_projected: dsa_query_projected
                    .as_ref()
                    .expect("fused MLA decode DSA query buffer allocated")
                    .buffer,
                weights_projected: dsa_weights_projected
                    .as_ref()
                    .expect("fused MLA decode DSA weights buffer allocated")
                    .buffer,
                query_dim: dsa.query_dim,
                heads: dsa.heads,
            }),
        };
        let capture_identity = mla_decode_query_projection_capture_identity(
            buffers,
            rows,
            hidden_dim,
            q_lora_rank,
            q_output_dim,
            eps,
        );
        let program = CoordinatorCudaGraphProgram::LayerMlaDecodeQueryProjectionBf16;
        if !slot.has_captured_graph_identity(program, signature, capture_identity) {
            unsafe {
                enqueue_mla_decode_query_projection(
                    library,
                    slot.stream_ptr(),
                    buffers,
                    rows,
                    hidden_dim,
                    q_lora_rank,
                    q_output_dim,
                    eps,
                )?;
            }
            slot.stream_synchronize()
                .context("warming fused MLA decode query projection")?;
        }
        slot.capture_or_update_graph_exec(
            library,
            program,
            signature,
            capture_identity,
            |library, stream, _workspace| unsafe {
                enqueue_mla_decode_query_projection(
                    library,
                    stream,
                    buffers,
                    rows,
                    hidden_dim,
                    q_lora_rank,
                    q_output_dim,
                    eps,
                )
            },
        )?;
        slot.launch_captured_graph_identity(library, program, signature, capture_identity)?;
        // Query split/RoPE and compressed attention use this same decode slot.
        // Keep the chain queued and synchronize once after the final attention graph.
        Ok(MlaDecodeQueryProjectionOutputs {
            q_projected: DeviceBf16Output {
                buffer: q_projected,
                bytes: q_output_bytes,
                rows,
                values_per_row: q_output_dim,
                backend: CUDA_MLA_DECODE_QUERY_PROJECTION_BF16_BACKEND,
            },
            dsa_query_projected: dsa_query_projected.map(|buffer| DeviceBf16Output {
                buffer,
                bytes: dsa_query_bytes,
                rows,
                values_per_row: dsa.map_or(0, |dsa| dsa.query_dim),
                backend: CUDA_MLA_DECODE_QUERY_PROJECTION_BF16_BACKEND,
            }),
            dsa_weights_projected: dsa_weights_projected.map(|buffer| DeviceBf16Output {
                buffer,
                bytes: dsa_weights_bytes,
                rows,
                values_per_row: dsa.map_or(0, |dsa| dsa.heads),
                backend: CUDA_MLA_DECODE_QUERY_PROJECTION_BF16_BACKEND,
            }),
        })
    })
}

#[allow(clippy::too_many_arguments)]
unsafe fn enqueue_mla_decode_query_projection(
    library: &'static NativeLibrary,
    stream: *mut c_void,
    buffers: MlaDecodeQueryProjectionBuffers,
    rows: usize,
    hidden_dim: usize,
    q_lora_rank: usize,
    q_output_dim: usize,
    eps: f32,
) -> Result<()> {
    unsafe {
        if let Some(w8a16) = buffers.q_a_w8a16 {
            library.cuda_linear_w8a16_group256_m1_simt_async(
                buffers.normalized_hidden,
                w8a16.weight,
                w8a16.scales,
                buffers.q_a_projected,
                hidden_dim,
                q_lora_rank,
                3,
                stream,
            )?;
        } else {
            library.cuda_linear_bf16_cublas_async(
                buffers.normalized_hidden,
                buffers
                    .q_a_weight
                    .context("fused MLA decode BF16 q_a weight is missing")?,
                None,
                buffers.q_a_projected,
                rows,
                hidden_dim,
                q_lora_rank,
                stream,
            )?;
        }
        library.cuda_rmsnorm_bf16_async(
            buffers.q_a_projected,
            buffers.q_a_norm_weight,
            buffers.q_a_normalized,
            rows as i32,
            q_lora_rank as i32,
            eps,
            stream,
        )?;
        if let Some(w4a16) = buffers.q_b_w4a16 {
            library.cuda_b12x_coordinator_w4a16_q_b_m1_async(&w4a16, stream)?;
        } else if let Some(w8a16) = buffers.q_b_w8a16 {
            library.cuda_linear_w8a16_group256_m1_simt_async(
                buffers.q_a_normalized,
                w8a16.weight,
                w8a16.scales,
                buffers.q_projected,
                q_lora_rank,
                q_output_dim,
                3,
                stream,
            )?;
        } else {
            library.cuda_linear_bf16_cublas_async(
                buffers.q_a_normalized,
                buffers
                    .q_b_weight
                    .context("fused MLA decode BF16 q_b weight is missing")?,
                None,
                buffers.q_projected,
                rows,
                q_lora_rank,
                q_output_dim,
                stream,
            )?;
        }
        if let Some(dsa) = buffers.dsa {
            library.cuda_linear_bf16_cublas_async(
                buffers.q_a_normalized,
                dsa.wq_b_weight,
                None,
                dsa.query_projected,
                rows,
                q_lora_rank,
                dsa.query_dim,
                stream,
            )?;
            library.cuda_linear_bf16_cublas_async(
                buffers.normalized_hidden,
                dsa.weights_proj_weight,
                None,
                dsa.weights_projected,
                rows,
                hidden_dim,
                dsa.heads,
                stream,
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn mla_decode_query_projection_capture_identity(
    buffers: MlaDecodeQueryProjectionBuffers,
    rows: usize,
    hidden_dim: usize,
    q_lora_rank: usize,
    q_output_dim: usize,
    eps: f32,
) -> usize {
    graph_capture_identity(&[
        buffers.normalized_hidden.ptr as usize,
        buffers.q_a_weight.map_or(0, |weight| weight.ptr as usize),
        buffers
            .q_a_w8a16
            .map_or(0, |w8a16| w8a16.weight.ptr as usize),
        buffers
            .q_a_w8a16
            .map_or(0, |w8a16| w8a16.scales.ptr as usize),
        buffers.q_a_projected.ptr as usize,
        buffers.q_a_norm_weight.ptr as usize,
        buffers.q_a_normalized.ptr as usize,
        buffers.q_b_weight.map_or(0, |weight| weight.ptr as usize),
        buffers
            .q_b_w4a16
            .map_or(0, |w4a16| w4a16.weight.ptr as usize),
        buffers
            .q_b_w4a16
            .map_or(0, |w4a16| w4a16.scale.ptr as usize),
        buffers
            .q_b_w4a16
            .map_or(0, |w4a16| w4a16.global_scale.ptr as usize),
        buffers
            .q_b_w4a16
            .map_or(0, |w4a16| w4a16.c_tmp.ptr as usize),
        buffers
            .q_b_w4a16
            .map_or(0, |w4a16| w4a16.locks.ptr as usize),
        buffers
            .q_b_w8a16
            .map_or(0, |w8a16| w8a16.weight.ptr as usize),
        buffers
            .q_b_w8a16
            .map_or(0, |w8a16| w8a16.scales.ptr as usize),
        buffers.q_projected.ptr as usize,
        buffers.dsa.map_or(0, |dsa| dsa.wq_b_weight.ptr as usize),
        buffers
            .dsa
            .map_or(0, |dsa| dsa.weights_proj_weight.ptr as usize),
        buffers
            .dsa
            .map_or(0, |dsa| dsa.query_projected.ptr as usize),
        buffers
            .dsa
            .map_or(0, |dsa| dsa.weights_projected.ptr as usize),
        buffers.dsa.map_or(0, |dsa| dsa.query_dim),
        buffers.dsa.map_or(0, |dsa| dsa.heads),
        rows,
        hidden_dim,
        q_lora_rank,
        q_output_dim,
        eps.to_bits() as usize,
    ])
}

#[derive(Clone, Copy)]
struct MlaDecodeScalarQaQueryProjectionBuffers {
    hidden: GlmrtDeviceBuffer,
    input_norm_weight: GlmrtDeviceBuffer,
    normalized_hidden: GlmrtDeviceBuffer,
    q_a_weight: Option<GlmrtDeviceBuffer>,
    q_a_w8a16: Option<CoordinatorW8a16ProjectionBuffers>,
    q_a_projected: GlmrtDeviceBuffer,
    q_a_norm_weight: GlmrtDeviceBuffer,
    q_a_normalized: GlmrtDeviceBuffer,
    q_b_w4a16: Option<GlmrtB12xCoordinatorW4a16Buffers>,
    q_b_w8a16: Option<CoordinatorW8a16ProjectionBuffers>,
    q_projected: GlmrtDeviceBuffer,
    dsa: Option<MlaDecodeQueryDsaProjectionBuffers>,
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn mla_decode_scalar_q_a_batched_q_b_projection_bf16_device_outputs(
    layer_id: usize,
    hidden: GlmrtDeviceBuffer,
    input_norm_weight: GlmrtDeviceBuffer,
    q_a_weight_name: &str,
    q_a_weight: Option<GlmrtDeviceBuffer>,
    q_a_norm_weight: GlmrtDeviceBuffer,
    q_b_weight_name: &str,
    dsa: Option<(GlmrtDeviceBuffer, GlmrtDeviceBuffer, usize, usize)>,
    rows: usize,
    hidden_dim: usize,
    q_lora_rank: usize,
    q_output_dim: usize,
    eps: f32,
) -> Result<(
    DeviceBf16Output,
    Option<DeviceBf16Output>,
    Option<DeviceBf16Output>,
)> {
    let w4a16_enabled = coordinator_w4a16_q_b_decode_enabled();
    let w8a16_enabled = coordinator_w8a16_q_b_decode_enabled();
    anyhow::ensure!(
        w4a16_enabled ^ w8a16_enabled,
        "scalar q_a/q1-shaped q_b projection requires exactly one coordinator W4A16 or W8A16 path"
    );
    anyhow::ensure!(
        (2..=16).contains(&rows),
        "scalar q_a/batched q_b projection requires 2..=16 rows, got {rows}"
    );
    anyhow::ensure!(
        hidden_dim > 0 && q_lora_rank > 0 && q_output_dim > 0,
        "scalar q_a/batched q_b projection dimensions must be nonzero"
    );
    anyhow::ensure!(
        hidden_dim <= i32::MAX as usize && q_lora_rank <= i32::MAX as usize,
        "scalar q_a/batched q_b normalization dimensions exceed i32"
    );
    anyhow::ensure!(
        eps.is_finite() && eps > 0.0,
        "scalar q_a/batched q_b projection epsilon must be finite and positive"
    );
    let bf16_bytes = std::mem::size_of::<u16>();
    let hidden_bytes = rows
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("scalar q_a hidden bytes overflow")?;
    let q_a_bytes = rows
        .checked_mul(q_lora_rank)
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("scalar q_a output bytes overflow")?;
    let q_output_bytes = rows
        .checked_mul(q_output_dim)
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("batched q_b output bytes overflow")?;
    let dsa = dsa.map(|(wq_b_weight, weights_proj_weight, query_dim, heads)| {
        MlaDecodeQueryDsaProjectionConfig {
            wq_b_weight,
            weights_proj_weight,
            query_dim,
            heads,
        }
    });
    let dsa_query_bytes = dsa
        .map(|dsa| {
            rows.checked_mul(dsa.query_dim)
                .and_then(|values| values.checked_mul(bf16_bytes))
                .context("scalar q_a DSA query bytes overflow")
        })
        .transpose()?
        .unwrap_or(0);
    let dsa_weights_bytes = dsa
        .map(|dsa| {
            rows.checked_mul(dsa.heads)
                .and_then(|values| values.checked_mul(bf16_bytes))
                .context("scalar q_a DSA weights bytes overflow")
        })
        .transpose()?
        .unwrap_or(0);
    let q_a_weight_bytes = q_lora_rank
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("scalar q_a weight bytes overflow")?;
    for (label, buffer, expected_bytes) in [
        ("hidden", hidden, hidden_bytes),
        (
            "input norm weight",
            input_norm_weight,
            hidden_dim * bf16_bytes,
        ),
        ("q_a norm weight", q_a_norm_weight, q_lora_rank * bf16_bytes),
    ] {
        anyhow::ensure!(
            !buffer.ptr.is_null() && buffer.bytes >= expected_bytes,
            "scalar q_a/batched q_b {label} buffer has {} bytes, expected at least {expected_bytes}",
            buffer.bytes
        );
        anyhow::ensure!(
            buffer.device_id == hidden.device_id,
            "scalar q_a/batched q_b {label} buffer is on CUDA device {}, expected {}",
            buffer.device_id,
            hidden.device_id
        );
    }
    if let Some(q_a_weight) = q_a_weight {
        anyhow::ensure!(
            !q_a_weight.ptr.is_null() && q_a_weight.bytes >= q_a_weight_bytes,
            "scalar q_a/batched q_b q_a weight buffer has {} bytes, expected at least {q_a_weight_bytes}",
            q_a_weight.bytes
        );
        anyhow::ensure!(
            q_a_weight.device_id == hidden.device_id,
            "scalar q_a/batched q_b q_a weight buffer is on CUDA device {}, expected {}",
            q_a_weight.device_id,
            hidden.device_id
        );
    }
    if let Some(dsa) = dsa {
        anyhow::ensure!(
            dsa.query_dim > 0 && dsa.heads > 0,
            "scalar q_a DSA projection dimensions must be nonzero"
        );
        for (label, buffer, expected_bytes) in [
            (
                "DSA wq_b weight",
                dsa.wq_b_weight,
                dsa.query_dim
                    .checked_mul(q_lora_rank)
                    .and_then(|values| values.checked_mul(bf16_bytes))
                    .context("scalar q_a DSA wq_b weight bytes overflow")?,
            ),
            (
                "DSA weights projection weight",
                dsa.weights_proj_weight,
                dsa.heads
                    .checked_mul(hidden_dim)
                    .and_then(|values| values.checked_mul(bf16_bytes))
                    .context("scalar q_a DSA weights projection bytes overflow")?,
            ),
        ] {
            anyhow::ensure!(
                !buffer.ptr.is_null() && buffer.bytes >= expected_bytes,
                "scalar q_a {label} buffer has {} bytes, expected at least {expected_bytes}",
                buffer.bytes
            );
            anyhow::ensure!(
                buffer.device_id == hidden.device_id,
                "scalar q_a {label} buffer is on CUDA device {}, expected {}",
                buffer.device_id,
                hidden.device_id
            );
        }
    }

    let q_a_w8a16 = coordinator_w8a16_q_a_decode_enabled()
        .then(|| preloaded_coordinator_w8a16_projection(q_a_weight_name, hidden_dim, q_lora_rank))
        .transpose()?;
    anyhow::ensure!(
        q_a_w8a16.is_some() ^ q_a_weight.is_some(),
        "scalar q_a/batched q_b requires exactly one BF16 or W8A16 q_a weight"
    );
    let w4a16_projection = w4a16_enabled
        .then(|| preloaded_coordinator_w4a16_projection(q_b_weight_name, q_lora_rank, q_output_dim))
        .transpose()?;
    let q_b_w8a16 = w8a16_enabled
        .then(|| preloaded_coordinator_w8a16_projection(q_b_weight_name, q_lora_rank, q_output_dim))
        .transpose()?;
    let graph_key = coord_attention_graph_key_for_layer_rows(layer_id, rows)?;
    let signature = CoordinatorCudaGraphSignature::mla_decode_scalar_q_a_query_projection_bf16(
        rows,
        hidden_dim,
        q_lora_rank,
        q_output_dim,
        eps,
        usize::from(q_a_w8a16.is_some()),
        dsa.map_or(0, |dsa| dsa.query_dim),
        dsa.map_or(0, |dsa| dsa.heads),
    );
    with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
        let normalized_hidden = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            hidden_bytes,
            "scalar q_a normalized hidden",
        )?;
        let q_a_projected = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::B,
            q_a_bytes,
            "scalar q_a projection",
        )?;
        let q_a_normalized = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            q_a_bytes,
            "scalar q_a normalization",
        )?;
        let q_projected =
            OwnedCoordinatorDeviceBuffer::new(library, q_output_bytes, "batched q_b projection")?;
        let dsa_query_projected = dsa
            .map(|_| {
                OwnedCoordinatorDeviceBuffer::new(
                    library,
                    dsa_query_bytes,
                    "scalar q_a DSA query projection",
                )
            })
            .transpose()?;
        let dsa_weights_projected = dsa
            .map(|_| {
                OwnedCoordinatorDeviceBuffer::new(
                    library,
                    dsa_weights_bytes,
                    "scalar q_a DSA weights projection",
                )
            })
            .transpose()?;
        let q_b_w4a16 = w4a16_projection
            .map(|projection| {
                coordinator_w4a16_launch_buffers(
                    library,
                    slot,
                    projection,
                    q_a_normalized,
                    q_projected.buffer,
                    CoordinatorCudaScratchSlot::R,
                )
            })
            .transpose()?;
        let buffers = MlaDecodeScalarQaQueryProjectionBuffers {
            hidden,
            input_norm_weight,
            normalized_hidden,
            q_a_weight,
            q_a_w8a16,
            q_a_projected,
            q_a_norm_weight,
            q_a_normalized,
            q_b_w4a16,
            q_b_w8a16,
            q_projected: q_projected.buffer,
            dsa: dsa.map(|dsa| MlaDecodeQueryDsaProjectionBuffers {
                wq_b_weight: dsa.wq_b_weight,
                weights_proj_weight: dsa.weights_proj_weight,
                query_projected: dsa_query_projected
                    .as_ref()
                    .expect("scalar q_a DSA query buffer allocated")
                    .buffer,
                weights_projected: dsa_weights_projected
                    .as_ref()
                    .expect("scalar q_a DSA weights buffer allocated")
                    .buffer,
                query_dim: dsa.query_dim,
                heads: dsa.heads,
            }),
        };
        let capture_identity = mla_decode_scalar_q_a_query_projection_capture_identity(
            buffers,
            rows,
            hidden_dim,
            q_lora_rank,
            q_output_dim,
            eps,
        );
        let program = CoordinatorCudaGraphProgram::LayerMlaDecodeScalarQAQueryProjectionBf16;
        if !slot.has_captured_graph_identity(program, signature, capture_identity) {
            unsafe {
                enqueue_mla_decode_scalar_q_a_batched_q_b_projection(
                    library,
                    slot.stream_ptr(),
                    buffers,
                    rows,
                    hidden_dim,
                    q_lora_rank,
                    eps,
                )?;
            }
            slot.stream_synchronize()
                .context("warming scalar q_a/batched q_b projection")?;
        }
        slot.capture_or_update_graph_exec(
            library,
            program,
            signature,
            capture_identity,
            |library, stream, _workspace| unsafe {
                enqueue_mla_decode_scalar_q_a_batched_q_b_projection(
                    library,
                    stream,
                    buffers,
                    rows,
                    hidden_dim,
                    q_lora_rank,
                    eps,
                )
            },
        )?;
        slot.launch_captured_graph_identity(library, program, signature, capture_identity)?;
        let ready_event = slot.record_output_ready_event(library)?;
        let mut q_projected = DeviceBf16Output {
            buffer: q_projected,
            bytes: q_output_bytes,
            rows,
            values_per_row: q_output_dim,
            backend: CUDA_MLA_DECODE_QUERY_PROJECTION_BF16_BACKEND,
        };
        q_projected.set_ready_event(Arc::clone(&ready_event));
        let dsa_query_projected = dsa_query_projected.map(|buffer| {
            let mut output = DeviceBf16Output {
                buffer,
                bytes: dsa_query_bytes,
                rows,
                values_per_row: dsa.map_or(0, |dsa| dsa.query_dim),
                backend: CUDA_MLA_DECODE_QUERY_PROJECTION_BF16_BACKEND,
            };
            output.set_ready_event(Arc::clone(&ready_event));
            output
        });
        let dsa_weights_projected = dsa_weights_projected.map(|buffer| {
            let mut output = DeviceBf16Output {
                buffer,
                bytes: dsa_weights_bytes,
                rows,
                values_per_row: dsa.map_or(0, |dsa| dsa.heads),
                backend: CUDA_MLA_DECODE_QUERY_PROJECTION_BF16_BACKEND,
            };
            output.set_ready_event(Arc::clone(&ready_event));
            output
        });
        Ok((q_projected, dsa_query_projected, dsa_weights_projected))
    })
}

pub(in crate::commands::real_full) fn copy_mla_decode_query_row_to_attention_stream(
    layer_id: usize,
    source: &DeviceBf16Output,
    row: usize,
    label: &'static str,
) -> Result<DeviceBf16Output> {
    anyhow::ensure!(
        row < source.rows,
        "{label} row {row} exceeds source rows {}",
        source.rows
    );
    let row_bytes = source
        .values_per_row
        .checked_mul(std::mem::size_of::<u16>())
        .with_context(|| format!("{label} row bytes overflow"))?;
    let source_row = device_buffer_byte_view(
        source.buffer(),
        row.checked_mul(row_bytes)
            .with_context(|| format!("{label} row offset overflow"))?,
        row_bytes,
        label,
    )?;
    let graph_key = coord_attention_graph_key_for_layer_rows(layer_id, 1)?;
    with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
        let output = OwnedCoordinatorDeviceBuffer::new(library, row_bytes, label)?;
        let stream = slot.stream_ptr();
        source
            .wait_ready_on_stream(stream)
            .with_context(|| format!("waiting for {label} source projection"))?;
        unsafe {
            library
                .copy_d2d_async(output.buffer, source_row, row_bytes, stream)
                .with_context(|| format!("copying {label} onto decode attention stream"))?;
        }
        Ok(DeviceBf16Output {
            buffer: output,
            bytes: row_bytes,
            rows: 1,
            values_per_row: source.values_per_row,
            backend: CUDA_MLA_DECODE_QUERY_PROJECTION_BF16_BACKEND,
        })
    })
}

#[allow(clippy::too_many_arguments)]
unsafe fn enqueue_mla_decode_scalar_q_a_batched_q_b_projection(
    library: &'static NativeLibrary,
    stream: *mut c_void,
    buffers: MlaDecodeScalarQaQueryProjectionBuffers,
    rows: usize,
    hidden_dim: usize,
    q_lora_rank: usize,
    eps: f32,
) -> Result<()> {
    let bf16_bytes = std::mem::size_of::<u16>();
    let hidden_row_bytes = hidden_dim * bf16_bytes;
    let q_a_row_bytes = q_lora_rank * bf16_bytes;
    // q_a changes BF16 results with GEMM M; preserve M=1 as a strided batch
    // instead of turning the target rows into one M=rows GEMM.
    for row in 0..rows {
        let hidden_offset = row * hidden_row_bytes;
        let hidden_row = device_buffer_byte_view(
            buffers.hidden,
            hidden_offset,
            hidden_row_bytes,
            "scalar q_a hidden row",
        )?;
        let normalized_row = device_buffer_byte_view(
            buffers.normalized_hidden,
            hidden_offset,
            hidden_row_bytes,
            "scalar q_a normalized hidden row",
        )?;
        unsafe {
            library.cuda_rmsnorm_bf16_async(
                hidden_row,
                buffers.input_norm_weight,
                normalized_row,
                1,
                hidden_dim as i32,
                eps,
                stream,
            )?;
        }
    }
    if let Some(w8a16) = buffers.q_a_w8a16 {
        // Share one row-major W8 traversal while retaining the exact W8 M=1
        // FMA and warp-reduction order independently for every target row.
        unsafe {
            library.cuda_linear_w8a16_group256_m1_parity_batched_async(
                buffers.normalized_hidden,
                w8a16.weight,
                w8a16.scales,
                buffers.q_a_projected,
                rows,
                hidden_dim,
                q_lora_rank,
                stream,
            )?;
        }
    } else {
        // Reuse the recurrent M=1 cuBLASLt tensor-core plan across all target
        // rows. The native launcher qualifies its first live result bit-for-bit
        // against repeated M=1 calls before retaining the shared-weight path.
        unsafe {
            library.cuda_linear_bf16_m1_parity_batched_cublaslt_async(
                buffers.normalized_hidden,
                buffers
                    .q_a_weight
                    .context("scalar q_a BF16 weight is missing")?,
                buffers.q_a_projected,
                rows,
                hidden_dim,
                q_lora_rank,
                stream,
            )?;
        }
    }
    for row in 0..rows {
        let q_a_offset = row * q_a_row_bytes;
        let q_a_projected_row = device_buffer_byte_view(
            buffers.q_a_projected,
            q_a_offset,
            q_a_row_bytes,
            "scalar q_a projected row",
        )?;
        let q_a_normalized_row = device_buffer_byte_view(
            buffers.q_a_normalized,
            q_a_offset,
            q_a_row_bytes,
            "scalar q_a normalized row",
        )?;
        unsafe {
            library.cuda_rmsnorm_bf16_async(
                q_a_projected_row,
                buffers.q_a_norm_weight,
                q_a_normalized_row,
                1,
                q_lora_rank as i32,
                eps,
                stream,
            )?;
        }
    }
    if let Some(dsa) = buffers.dsa {
        // These DSA shapes do not retain recurrent cuBLAS parity through the
        // strided-batched API. Keep the exact recurrent M=1 launch shape per
        // row until a bitwise-equivalent shared-weight kernel replaces it.
        let dsa_query_row_bytes = dsa.query_dim * bf16_bytes;
        let dsa_weights_row_bytes = dsa.heads * bf16_bytes;
        for row in 0..rows {
            let hidden_offset = row * hidden_row_bytes;
            let q_a_offset = row * q_a_row_bytes;
            let dsa_query_offset = row * dsa_query_row_bytes;
            let dsa_weights_offset = row * dsa_weights_row_bytes;
            let q_a_normalized_row = device_buffer_byte_view(
                buffers.q_a_normalized,
                q_a_offset,
                q_a_row_bytes,
                "scalar q_a DSA normalized query row",
            )?;
            let dsa_query_row = device_buffer_byte_view(
                dsa.query_projected,
                dsa_query_offset,
                dsa_query_row_bytes,
                "scalar q_a DSA projected query row",
            )?;
            let normalized_hidden_row = device_buffer_byte_view(
                buffers.normalized_hidden,
                hidden_offset,
                hidden_row_bytes,
                "scalar q_a DSA normalized hidden row",
            )?;
            let dsa_weights_row = device_buffer_byte_view(
                dsa.weights_projected,
                dsa_weights_offset,
                dsa_weights_row_bytes,
                "scalar q_a DSA projected weights row",
            )?;
            unsafe {
                library.cuda_linear_bf16_cublas_async(
                    q_a_normalized_row,
                    dsa.wq_b_weight,
                    None,
                    dsa_query_row,
                    1,
                    q_lora_rank,
                    dsa.query_dim,
                    stream,
                )?;
                library.cuda_linear_bf16_cublas_async(
                    normalized_hidden_row,
                    dsa.weights_proj_weight,
                    None,
                    dsa_weights_row,
                    1,
                    hidden_dim,
                    dsa.heads,
                    stream,
                )?;
            }
        }
    }
    if let Some(w4a16) = buffers.q_b_w4a16 {
        unsafe {
            library.cuda_b12x_coordinator_w4a16_q_b_m8_async(&w4a16, rows, stream)?;
        }
    } else if let Some(w8a16) = buffers.q_b_w8a16 {
        let q_output_dim = buffers.q_projected.bytes / (rows * bf16_bytes);
        // Share one weight traversal across the target rows while retaining
        // the recurrent M=1 FMA and warp-reduction order for every row.
        unsafe {
            library.cuda_linear_w8a16_group256_m1_parity_batched_async(
                buffers.q_a_normalized,
                w8a16.weight,
                w8a16.scales,
                buffers.q_projected,
                rows,
                q_lora_rank,
                q_output_dim,
                stream,
            )?;
        }
    } else {
        anyhow::bail!("scalar q_a/q1-shaped q_b projection has no quantized Q-B weights");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn mla_decode_scalar_q_a_query_projection_capture_identity(
    buffers: MlaDecodeScalarQaQueryProjectionBuffers,
    rows: usize,
    hidden_dim: usize,
    q_lora_rank: usize,
    q_output_dim: usize,
    eps: f32,
) -> usize {
    graph_capture_identity(&[
        buffers.hidden.ptr as usize,
        buffers.input_norm_weight.ptr as usize,
        buffers.normalized_hidden.ptr as usize,
        buffers.q_a_weight.map_or(0, |weight| weight.ptr as usize),
        buffers
            .q_a_w8a16
            .map_or(0, |w8a16| w8a16.weight.ptr as usize),
        buffers
            .q_a_w8a16
            .map_or(0, |w8a16| w8a16.scales.ptr as usize),
        buffers.q_a_projected.ptr as usize,
        buffers.q_a_norm_weight.ptr as usize,
        buffers.q_a_normalized.ptr as usize,
        buffers
            .q_b_w4a16
            .map_or(0, |w4a16| w4a16.weight.ptr as usize),
        buffers
            .q_b_w4a16
            .map_or(0, |w4a16| w4a16.scale.ptr as usize),
        buffers
            .q_b_w4a16
            .map_or(0, |w4a16| w4a16.global_scale.ptr as usize),
        buffers
            .q_b_w4a16
            .map_or(0, |w4a16| w4a16.packed_route_indices.ptr as usize),
        buffers
            .q_b_w4a16
            .map_or(0, |w4a16| w4a16.block_expert_ids.ptr as usize),
        buffers
            .q_b_w4a16
            .map_or(0, |w4a16| w4a16.packed_route_count.ptr as usize),
        buffers
            .q_b_w4a16
            .map_or(0, |w4a16| w4a16.topk_weights.ptr as usize),
        buffers
            .q_b_w4a16
            .map_or(0, |w4a16| w4a16.c_tmp.ptr as usize),
        buffers
            .q_b_w4a16
            .map_or(0, |w4a16| w4a16.locks.ptr as usize),
        buffers
            .q_b_w8a16
            .map_or(0, |w8a16| w8a16.weight.ptr as usize),
        buffers
            .q_b_w8a16
            .map_or(0, |w8a16| w8a16.scales.ptr as usize),
        buffers.q_projected.ptr as usize,
        buffers.dsa.map_or(0, |dsa| dsa.wq_b_weight.ptr as usize),
        buffers
            .dsa
            .map_or(0, |dsa| dsa.weights_proj_weight.ptr as usize),
        buffers
            .dsa
            .map_or(0, |dsa| dsa.query_projected.ptr as usize),
        buffers
            .dsa
            .map_or(0, |dsa| dsa.weights_projected.ptr as usize),
        buffers.dsa.map_or(0, |dsa| dsa.query_dim),
        buffers.dsa.map_or(0, |dsa| dsa.heads),
        rows,
        hidden_dim,
        q_lora_rank,
        q_output_dim,
        eps.to_bits() as usize,
    ])
}

pub(in crate::commands::real_full) fn coordinator_cuda_graph_stats(
) -> Result<CoordinatorCudaGraphStats> {
    let registry = coordinator_cuda_graph_workspace_registry()?;
    let mut stats = CoordinatorCudaGraphStats {
        slots: registry.len(),
        ..CoordinatorCudaGraphStats::default()
    };
    for slot in &registry.slots {
        let slot = slot
            .lock()
            .map_err(|_| anyhow::anyhow!("coordinator CUDA graph stats slot borrowed"))?;
        stats.captured_graphs += slot.captured_graphs.len();
        stats.graph_captures += slot.graph_captures;
        stats.graph_launches += slot.graph_launches;
        stats.acquisitions += slot.acquisitions;
    }
    Ok(stats)
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn capture_or_update_layer_dense_mlp_bf16_graph(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    signature: CoordinatorCudaGraphSignature,
    input_buffer: GlmrtDeviceBuffer,
    gate_buffer: GlmrtDeviceBuffer,
    up_buffer: GlmrtDeviceBuffer,
    down_buffer: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden: usize,
    intermediate: usize,
    down_stride: usize,
    label: &'static str,
) -> Result<()> {
    if !slot.has_captured_graph(CoordinatorCudaGraphProgram::LayerDenseMlpBf16, signature) {
        slot.stream_synchronize()
            .with_context(|| format!("synchronizing {label} inputs before graph capture"))?;
        slot.capture_graph(
            library,
            CoordinatorCudaGraphProgram::LayerDenseMlpBf16,
            signature,
            |library, cuda_stream, _workspace| unsafe {
                library
                    .cuda_silu_gated_mlp_rows_bf16_down_stride_async(
                        input_buffer,
                        gate_buffer,
                        up_buffer,
                        down_buffer,
                        output_buffer,
                        rows,
                        hidden,
                        intermediate,
                        down_stride,
                        cuda_stream,
                    )
                    .with_context(|| format!("capturing async CUDA {label}"))?;
                Ok(())
            },
        )?;
    } else {
        let (graph_raw, exec_raw) = slot
            .captured_graph_raw_handles(CoordinatorCudaGraphProgram::LayerDenseMlpBf16, signature)
            .context("coordinator CUDA graph slot lost captured dense MLP graph before update")?;
        unsafe {
            library
                .cuda_graph_update_silu_gated_mlp_rows_bf16_down_stride_node(
                    graph_raw,
                    exec_raw,
                    0,
                    input_buffer,
                    gate_buffer,
                    up_buffer,
                    down_buffer,
                    output_buffer,
                    rows,
                    hidden,
                    intermediate,
                    down_stride,
                )
                .with_context(|| format!("updating captured CUDA {label} graph node"))?;
        }
    }
    slot.launch_captured_graph(
        library,
        CoordinatorCudaGraphProgram::LayerDenseMlpBf16,
        signature,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn capture_or_update_layer_triton_dense_mlp_bf16_graph(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    signature: CoordinatorCudaGraphSignature,
    input_buffer: GlmrtDeviceBuffer,
    gate_buffer: GlmrtDeviceBuffer,
    up_buffer: GlmrtDeviceBuffer,
    down_buffer: GlmrtDeviceBuffer,
    gate_output_buffer: GlmrtDeviceBuffer,
    up_output_buffer: GlmrtDeviceBuffer,
    activation_buffer: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden: usize,
    intermediate: usize,
    down_stride: usize,
    label: &'static str,
) -> Result<()> {
    if !slot.has_captured_graph(
        CoordinatorCudaGraphProgram::LayerTritonDenseMlpBf16,
        signature,
    ) {
        slot.stream_synchronize()
            .with_context(|| format!("synchronizing {label} inputs before Triton warmup"))?;
        launch_triton_dense_mlp_bf16_graph_capture(
            slot.stream_ptr(),
            input_buffer,
            gate_buffer,
            up_buffer,
            down_buffer,
            gate_output_buffer,
            up_output_buffer,
            activation_buffer,
            output_buffer,
            rows,
            hidden,
            intermediate,
            down_stride,
        )
        .with_context(|| format!("warming Triton {label} graph capture"))?;
        slot.stream_synchronize()
            .with_context(|| format!("synchronizing {label} Triton warmup"))?;
        slot.capture_graph(
            library,
            CoordinatorCudaGraphProgram::LayerTritonDenseMlpBf16,
            signature,
            |_library, cuda_stream, _workspace| {
                launch_triton_dense_mlp_bf16_graph_capture(
                    cuda_stream,
                    input_buffer,
                    gate_buffer,
                    up_buffer,
                    down_buffer,
                    gate_output_buffer,
                    up_output_buffer,
                    activation_buffer,
                    output_buffer,
                    rows,
                    hidden,
                    intermediate,
                    down_stride,
                )
                .with_context(|| format!("capturing Triton {label}"))?;
                Ok(())
            },
        )?;
    }
    slot.launch_captured_graph(
        library,
        CoordinatorCudaGraphProgram::LayerTritonDenseMlpBf16,
        signature,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_triton_dense_mlp_bf16_graph_capture(
    cuda_stream: *mut c_void,
    input_buffer: GlmrtDeviceBuffer,
    gate_buffer: GlmrtDeviceBuffer,
    up_buffer: GlmrtDeviceBuffer,
    down_buffer: GlmrtDeviceBuffer,
    gate_output_buffer: GlmrtDeviceBuffer,
    up_output_buffer: GlmrtDeviceBuffer,
    activation_buffer: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden: usize,
    intermediate: usize,
    down_stride: usize,
) -> Result<()> {
    let buffers = [
        PythonDeviceBufferArg {
            name: "input",
            ptr: input_buffer.ptr,
            bytes: input_buffer.bytes,
            device_id: input_buffer.device_id,
            flags: input_buffer.flags,
        },
        PythonDeviceBufferArg {
            name: "gate_weight",
            ptr: gate_buffer.ptr,
            bytes: gate_buffer.bytes,
            device_id: gate_buffer.device_id,
            flags: gate_buffer.flags,
        },
        PythonDeviceBufferArg {
            name: "up_weight",
            ptr: up_buffer.ptr,
            bytes: up_buffer.bytes,
            device_id: up_buffer.device_id,
            flags: up_buffer.flags,
        },
        PythonDeviceBufferArg {
            name: "down_weight",
            ptr: down_buffer.ptr,
            bytes: down_buffer.bytes,
            device_id: down_buffer.device_id,
            flags: down_buffer.flags,
        },
        PythonDeviceBufferArg {
            name: "gate_output",
            ptr: gate_output_buffer.ptr,
            bytes: gate_output_buffer.bytes,
            device_id: gate_output_buffer.device_id,
            flags: gate_output_buffer.flags,
        },
        PythonDeviceBufferArg {
            name: "up_output",
            ptr: up_output_buffer.ptr,
            bytes: up_output_buffer.bytes,
            device_id: up_output_buffer.device_id,
            flags: up_output_buffer.flags,
        },
        PythonDeviceBufferArg {
            name: "activation",
            ptr: activation_buffer.ptr,
            bytes: activation_buffer.bytes,
            device_id: activation_buffer.device_id,
            flags: activation_buffer.flags,
        },
        PythonDeviceBufferArg {
            name: "output",
            ptr: output_buffer.ptr,
            bytes: output_buffer.bytes,
            device_id: output_buffer.device_id,
            flags: output_buffer.flags,
        },
    ];
    let kwargs = [
        ("rows", PythonKernelArg::Usize(rows)),
        ("hidden", PythonKernelArg::Usize(hidden)),
        ("intermediate", PythonKernelArg::Usize(intermediate)),
        ("down_stride", PythonKernelArg::Usize(down_stride)),
    ];
    launch_python_graph_capture(PythonGraphCaptureLaunch {
        module: "triton_mlp_capture",
        function: "capture_dense_mlp",
        cuda_stream,
        buffers: &buffers,
        kwargs: &kwargs,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn capture_or_update_sparse_a_triton_router_topk_bf16_graph(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    signature: CoordinatorCudaGraphSignature,
    hidden_buffer: GlmrtDeviceBuffer,
    weight_buffer: GlmrtDeviceBuffer,
    bias_buffer: GlmrtDeviceBuffer,
    score_scratch_buffer: GlmrtDeviceBuffer,
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
        CoordinatorCudaGraphProgram::SparseATritonRouterTopKBf16,
        signature,
    ) {
        slot.stream_synchronize()
            .with_context(|| format!("synchronizing {label} inputs before Triton warmup"))?;
        launch_triton_router_topk_bf16_graph_capture(
            slot.stream_ptr(),
            hidden_buffer,
            weight_buffer,
            bias_buffer,
            score_scratch_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            rows,
            hidden_dim,
            experts,
            top_k,
        )
        .with_context(|| format!("warming Triton {label} graph capture"))?;
        slot.stream_synchronize()
            .with_context(|| format!("synchronizing {label} Triton warmup"))?;
        slot.capture_graph(
            library,
            CoordinatorCudaGraphProgram::SparseATritonRouterTopKBf16,
            signature,
            |_library, cuda_stream, _workspace| {
                launch_triton_router_topk_bf16_graph_capture(
                    cuda_stream,
                    hidden_buffer,
                    weight_buffer,
                    bias_buffer,
                    score_scratch_buffer,
                    index_buffer,
                    score_buffer,
                    weight_buffer_out,
                    rows,
                    hidden_dim,
                    experts,
                    top_k,
                )
                .with_context(|| format!("capturing Triton {label}"))?;
                Ok(())
            },
        )?;
    }
    slot.launch_captured_graph(
        library,
        CoordinatorCudaGraphProgram::SparseATritonRouterTopKBf16,
        signature,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_triton_router_topk_bf16_graph_capture(
    cuda_stream: *mut c_void,
    hidden_buffer: GlmrtDeviceBuffer,
    weight_buffer: GlmrtDeviceBuffer,
    bias_buffer: GlmrtDeviceBuffer,
    score_scratch_buffer: GlmrtDeviceBuffer,
    index_buffer: GlmrtDeviceBuffer,
    score_buffer: GlmrtDeviceBuffer,
    weight_buffer_out: GlmrtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<()> {
    let buffers = [
        PythonDeviceBufferArg {
            name: "hidden",
            ptr: hidden_buffer.ptr,
            bytes: hidden_buffer.bytes,
            device_id: hidden_buffer.device_id,
            flags: hidden_buffer.flags,
        },
        PythonDeviceBufferArg {
            name: "router_weight",
            ptr: weight_buffer.ptr,
            bytes: weight_buffer.bytes,
            device_id: weight_buffer.device_id,
            flags: weight_buffer.flags,
        },
        PythonDeviceBufferArg {
            name: "correction_bias",
            ptr: bias_buffer.ptr,
            bytes: bias_buffer.bytes,
            device_id: bias_buffer.device_id,
            flags: bias_buffer.flags,
        },
        PythonDeviceBufferArg {
            name: "score_scratch",
            ptr: score_scratch_buffer.ptr,
            bytes: score_scratch_buffer.bytes,
            device_id: score_scratch_buffer.device_id,
            flags: score_scratch_buffer.flags,
        },
        PythonDeviceBufferArg {
            name: "topk_indices",
            ptr: index_buffer.ptr,
            bytes: index_buffer.bytes,
            device_id: index_buffer.device_id,
            flags: index_buffer.flags,
        },
        PythonDeviceBufferArg {
            name: "topk_scores",
            ptr: score_buffer.ptr,
            bytes: score_buffer.bytes,
            device_id: score_buffer.device_id,
            flags: score_buffer.flags,
        },
        PythonDeviceBufferArg {
            name: "topk_weights",
            ptr: weight_buffer_out.ptr,
            bytes: weight_buffer_out.bytes,
            device_id: weight_buffer_out.device_id,
            flags: weight_buffer_out.flags,
        },
    ];
    let kwargs = [
        ("rows", PythonKernelArg::Usize(rows)),
        ("hidden_dim", PythonKernelArg::Usize(hidden_dim)),
        ("experts", PythonKernelArg::Usize(experts)),
        ("top_k", PythonKernelArg::Usize(top_k)),
        (
            "routed_scaling_factor",
            PythonKernelArg::F64(GLM52_ROUTED_SCALING_FACTOR as f64),
        ),
    ];
    launch_python_graph_capture(PythonGraphCaptureLaunch {
        module: "triton_router_capture",
        function: "capture_router_topk",
        cuda_stream,
        buffers: &buffers,
        kwargs: &kwargs,
    })
}

pub(in crate::commands::real_full) fn dense_mlp_graph_value_bytes(
    graph_key: &CoordinatorGraphKey,
    values_per_row: usize,
    context: &str,
) -> Result<usize> {
    graph_key
        .row_bucket
        .row_capacity
        .checked_mul(values_per_row)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .with_context(|| format!("{context} graph buffer bytes overflow usize"))
}

pub(in crate::commands::real_full) fn dense_mlp_graph_signature(
    graph_key: &CoordinatorGraphKey,
    hidden: usize,
    intermediate: usize,
    down_stride: usize,
) -> CoordinatorCudaGraphSignature {
    CoordinatorCudaGraphSignature::silu_gated_mlp_rows_bf16_down_stride(
        graph_key.row_bucket.row_capacity,
        hidden,
        intermediate,
        down_stride,
    )
}

pub(in crate::commands::real_full) fn triton_dense_mlp_graph_signature(
    rows: usize,
    hidden: usize,
    intermediate: usize,
    down_stride: usize,
    input_buffer: GlmrtDeviceBuffer,
    gate_buffer: GlmrtDeviceBuffer,
    up_buffer: GlmrtDeviceBuffer,
    down_buffer: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
) -> CoordinatorCudaGraphSignature {
    CoordinatorCudaGraphSignature::triton_silu_gated_mlp_rows_bf16_down_stride(
        rows,
        hidden,
        intermediate,
        down_stride,
        triton_dense_mlp_buffer_identity(
            input_buffer,
            gate_buffer,
            up_buffer,
            down_buffer,
            output_buffer,
        ),
    )
}

fn triton_dense_mlp_buffer_identity(
    input_buffer: GlmrtDeviceBuffer,
    gate_buffer: GlmrtDeviceBuffer,
    up_buffer: GlmrtDeviceBuffer,
    down_buffer: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
) -> usize {
    [
        input_buffer,
        gate_buffer,
        up_buffer,
        down_buffer,
        output_buffer,
    ]
    .iter()
    .fold(0x9e37_79b9_7f4a_7c15_usize, |acc, buffer| {
        acc.rotate_left(13)
            ^ (buffer.ptr as usize)
            ^ buffer.bytes.rotate_left(7)
            ^ ((buffer.device_id as usize) << 3)
    })
}

pub(in crate::commands::real_full) fn triton_router_topk_graph_signature(
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
    hidden_buffer: GlmrtDeviceBuffer,
    weight_buffer: GlmrtDeviceBuffer,
    bias_buffer: GlmrtDeviceBuffer,
) -> CoordinatorCudaGraphSignature {
    CoordinatorCudaGraphSignature::triton_router_topk_bf16(
        rows,
        hidden_dim,
        experts,
        top_k,
        triton_router_topk_buffer_identity(hidden_buffer, weight_buffer, bias_buffer),
    )
}

fn triton_router_topk_buffer_identity(
    hidden_buffer: GlmrtDeviceBuffer,
    weight_buffer: GlmrtDeviceBuffer,
    bias_buffer: GlmrtDeviceBuffer,
) -> usize {
    [hidden_buffer, weight_buffer, bias_buffer].iter().fold(
        0xd1b5_4a32_d192_ed03_usize,
        |acc, buffer| {
            acc.rotate_left(11)
                ^ (buffer.ptr as usize)
                ^ buffer.bytes.rotate_left(5)
                ^ ((buffer.device_id as usize) << 2)
        },
    )
}

pub(in crate::commands::real_full) fn coord_sparse_b_graph_key_for_bf16_full_rows(
    bytes: usize,
) -> Result<Option<CoordinatorGraphKey>> {
    if bytes < GLM52_HIDDEN_BF16_BYTES || bytes % GLM52_HIDDEN_BF16_BYTES != 0 {
        return Ok(None);
    }
    let rows = bytes / GLM52_HIDDEN_BF16_BYTES;
    let mode = if rows == 1 {
        LayerWaveMode::Decode
    } else {
        LayerWaveMode::Prefill
    };
    CoordinatorGraphKey::glm52_bf16(CoordinatorGraphShape::CoordSparseB, mode, rows)
        .map(Some)
        .context("selecting Coord-Sparse-B graph slot for BF16 residual add")
}

pub(in crate::commands::real_full) fn coord_sparse_a_graph_key_for_full_hidden_rows(
    rows: usize,
    hidden_dim: usize,
) -> Result<Option<CoordinatorGraphKey>> {
    if rows == 0 || hidden_dim != GLM52_HIDDEN_SIZE {
        return Ok(None);
    }
    let mode = if rows == 1 {
        LayerWaveMode::Decode
    } else {
        LayerWaveMode::Prefill
    };
    CoordinatorGraphKey::glm52_bf16(CoordinatorGraphShape::CoordSparseA, mode, rows)
        .map(Some)
        .context("selecting Coord-Sparse-A graph slot for full-width BF16 router top-k")
}

pub(in crate::commands::real_full) fn coord_layer_graph_key_for_full_hidden_rows(
    tensor_name: &str,
    rows: usize,
    hidden_dim: usize,
) -> Result<Option<CoordinatorGraphKey>> {
    if rows == 0 {
        return Ok(None);
    }
    if tensor_name == "model.norm.weight" || tensor_name.starts_with("model.norm.weight.") {
        if hidden_dim != GLM52_HIDDEN_SIZE {
            return Ok(None);
        }
        let mode = if rows == 1 {
            LayerWaveMode::Decode
        } else {
            LayerWaveMode::Prefill
        };
        return CoordinatorGraphKey::glm52_bf16(CoordinatorGraphShape::CoordDense, mode, rows)
            .map(Some)
            .context("selecting Coord-Dense graph slot for terminal BF16 final RMSNorm");
    }
    let Some(layer_id) = glm52_layer_id_from_tensor_name(tensor_name) else {
        return Ok(None);
    };
    let Some((_, subpath)) = glm52_layer_tensor_subpath(tensor_name) else {
        return Ok(None);
    };
    let supported_width = hidden_dim == GLM52_HIDDEN_SIZE
        || (hidden_dim == GLM52_Q_LORA_RANK
            && subpath.starts_with("self_attn.q_a_layernorm.weight"));
    if !supported_width {
        return Ok(None);
    }
    let shape = if layer_id < GLM52_FIRST_K_DENSE_REPLACE {
        CoordinatorGraphShape::CoordDense
    } else {
        CoordinatorGraphShape::CoordSparseA
    };
    shape
        .validate_layer(LayerId(layer_id as u32))
        .context("validating coordinator graph layer family for GLM tensor")?;
    let mode = if rows == 1 {
        LayerWaveMode::Decode
    } else {
        LayerWaveMode::Prefill
    };
    CoordinatorGraphKey::glm52_bf16(shape, mode, rows)
        .map(Some)
        .context("selecting coordinator graph slot for full-width BF16 layer tensor")
}

pub(in crate::commands::real_full) fn coord_attention_graph_key_for_layer_rows(
    layer_id: usize,
    rows: usize,
) -> Result<CoordinatorGraphKey> {
    let layer_id_u32 = u32::try_from(layer_id)
        .with_context(|| format!("GLM-5.2 attention layer id {layer_id} exceeds u32"))?;
    let shape = CoordinatorGraphShape::CoordAttention;
    shape
        .validate_layer(LayerId(layer_id_u32))
        .context("validating coordinator attention graph layer family for GLM layer")?;
    let mode = if rows == 1 {
        LayerWaveMode::Decode
    } else {
        LayerWaveMode::Prefill
    };
    CoordinatorGraphKey::glm52_bf16(shape, mode, rows)
        .context("selecting coordinator graph slot for BF16 attention layer operation")
}

pub(in crate::commands::real_full) fn coord_compressed_attention_decode_graph_key_for_layer(
    layer_id: usize,
) -> Result<CoordinatorGraphKey> {
    let layer_id_u32 = u32::try_from(layer_id)
        .with_context(|| format!("GLM-5.2 attention layer id {layer_id} exceeds u32"))?;
    let shape = CoordinatorGraphShape::CoordCompressedAttention;
    shape
        .validate_layer(LayerId(layer_id_u32))
        .context("validating coordinator compressed-attention graph layer family")?;
    CoordinatorGraphKey::glm52_bf16(shape, LayerWaveMode::Decode, 1)
        .context("selecting dedicated compressed MLA decode graph slot")
}

pub(in crate::commands::real_full) fn coord_dense_mlp_graph_key_for_gate_up_down_names(
    gate_weight_name: &str,
    up_weight_name: &str,
    down_weight_name: &str,
    rows: usize,
) -> Result<Option<CoordinatorGraphKey>> {
    if rows == 0 {
        return Ok(None);
    }
    let Some((gate_layer, gate_subpath)) = glm52_layer_tensor_subpath(gate_weight_name) else {
        return Ok(None);
    };
    let Some((up_layer, up_subpath)) = glm52_layer_tensor_subpath(up_weight_name) else {
        return Ok(None);
    };
    let Some((down_layer, down_subpath)) = glm52_layer_tensor_subpath(down_weight_name) else {
        return Ok(None);
    };
    if gate_layer != up_layer || gate_layer != down_layer {
        return Ok(None);
    }
    let shape = if gate_layer < GLM52_FIRST_K_DENSE_REPLACE
        && gate_subpath.starts_with("mlp.gate_proj.weight")
        && up_subpath.starts_with("mlp.up_proj.weight")
        && down_subpath.starts_with("mlp.down_proj.weight")
    {
        CoordinatorGraphShape::CoordDense
    } else if gate_layer >= GLM52_FIRST_K_DENSE_REPLACE
        && gate_subpath.starts_with("mlp.shared_experts.gate_proj.weight")
        && up_subpath.starts_with("mlp.shared_experts.up_proj.weight")
        && down_subpath.starts_with("mlp.shared_experts.down_proj.weight")
    {
        CoordinatorGraphShape::CoordSparseA
    } else {
        return Ok(None);
    };
    shape
        .validate_layer(LayerId(gate_layer as u32))
        .context("validating MLP coordinator graph layer family for GLM tensor")?;
    let mode = if rows == 1 {
        LayerWaveMode::Decode
    } else {
        LayerWaveMode::Prefill
    };
    CoordinatorGraphKey::glm52_bf16(shape, mode, rows)
        .map(Some)
        .context("selecting coordinator graph slot for BF16 MLP layer tensor")
}

pub(in crate::commands::real_full) fn coord_embedding_graph_key(
    rows: usize,
) -> Result<Option<CoordinatorGraphKey>> {
    if rows == 0 {
        return Ok(None);
    }
    let mode = if rows == 1 {
        LayerWaveMode::Decode
    } else {
        LayerWaveMode::Prefill
    };
    CoordinatorGraphKey::glm52_bf16(CoordinatorGraphShape::CoordDense, mode, rows)
        .map(Some)
        .context("selecting Coord-Dense graph slot for BF16 embedding lookup")
}

pub(in crate::commands::real_full) fn coord_layer_graph_key_for_dsa_k_norm_names(
    weight_name: &str,
    bias_name: &str,
    rows: usize,
) -> Result<Option<CoordinatorGraphKey>> {
    if rows == 0 {
        return Ok(None);
    }
    let Some((weight_layer, weight_subpath)) = glm52_layer_tensor_subpath(weight_name) else {
        return Ok(None);
    };
    let Some((bias_layer, bias_subpath)) = glm52_layer_tensor_subpath(bias_name) else {
        return Ok(None);
    };
    if weight_layer != bias_layer {
        return Ok(None);
    }
    if !weight_subpath.starts_with("self_attn.indexer.k_norm.weight")
        || !bias_subpath.starts_with("self_attn.indexer.k_norm.bias")
    {
        return Ok(None);
    }
    let shape = if weight_layer < GLM52_FIRST_K_DENSE_REPLACE {
        CoordinatorGraphShape::CoordDense
    } else {
        CoordinatorGraphShape::CoordSparseA
    };
    shape
        .validate_layer(LayerId(weight_layer as u32))
        .context("validating DSA k_norm coordinator graph layer family for GLM tensor")?;
    let mode = if rows == 1 {
        LayerWaveMode::Decode
    } else {
        LayerWaveMode::Prefill
    };
    CoordinatorGraphKey::glm52_bf16(shape, mode, rows)
        .map(Some)
        .context("selecting coordinator graph slot for BF16 DSA k_norm affine LayerNorm")
}

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn capture_or_update_coord_dense_envelope_bf16_graph(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    signature: CoordinatorCudaGraphSignature,
    buffers: CoordDenseEnvelopeBf16Buffers,
    rows: usize,
    hidden: usize,
    intermediate: usize,
    label: &'static str,
) -> Result<()> {
    let values = rows
        .checked_mul(hidden)
        .with_context(|| format!("{label} value count overflows usize"))?;
    let capture_identity =
        coord_dense_envelope_bf16_capture_identity(buffers, rows, hidden, intermediate);
    if !slot.has_captured_graph_identity(
        CoordinatorCudaGraphProgram::CoordDenseEnvelopeBf16,
        signature,
        capture_identity,
    ) {
        unsafe {
            library
                .cuda_linear_bf16_cublas_async(
                    buffers.input,
                    buffers.q_weight,
                    None,
                    buffers.q_out,
                    rows,
                    hidden,
                    hidden,
                    slot.stream_ptr(),
                )
                .with_context(|| format!("warming CUDA cuBLAS {label} before graph capture"))?;
        }
    }
    slot.capture_or_update_graph_exec(
        library,
        CoordinatorCudaGraphProgram::CoordDenseEnvelopeBf16,
        signature,
        capture_identity,
        |library, stream, _workspace| unsafe {
            capture_coord_dense_envelope_bf16_ops(
                library,
                buffers,
                rows,
                hidden,
                intermediate,
                values,
                stream,
            )
            .with_context(|| format!("capturing async CUDA cuBLAS {label}"))
        },
    )?;
    slot.launch_captured_graph_identity(
        library,
        CoordinatorCudaGraphProgram::CoordDenseEnvelopeBf16,
        signature,
        capture_identity,
    )
}

#[allow(clippy::too_many_arguments)]
unsafe fn capture_coord_dense_envelope_bf16_ops(
    library: &'static NativeLibrary,
    buffers: CoordDenseEnvelopeBf16Buffers,
    rows: usize,
    hidden: usize,
    intermediate: usize,
    values: usize,
    stream: *mut c_void,
) -> Result<()> {
    unsafe {
        library.cuda_rmsnorm_bf16_async(
            buffers.input,
            buffers.norm0_weight,
            buffers.norm0_out,
            rows as i32,
            hidden as i32,
            1.0e-5,
            stream,
        )?;
        library.cuda_linear_bf16_cublas_async(
            buffers.norm0_out,
            buffers.q_weight,
            None,
            buffers.q_out,
            rows,
            hidden,
            hidden,
            stream,
        )?;
        library.cuda_linear_bf16_cublas_async(
            buffers.norm0_out,
            buffers.k_weight,
            None,
            buffers.k_out,
            rows,
            hidden,
            hidden,
            stream,
        )?;
        library.cuda_linear_bf16_cublas_async(
            buffers.norm0_out,
            buffers.v_weight,
            None,
            buffers.v_out,
            rows,
            hidden,
            hidden,
            stream,
        )?;
        library.cuda_causal_attention_bf16_async(
            buffers.q_out,
            buffers.k_out,
            buffers.v_out,
            buffers.attention_out,
            rows,
            1,
            hidden,
            hidden,
            0.5,
            stream,
        )?;
        library.cuda_linear_bf16_cublas_async(
            buffers.attention_out,
            buffers.o_weight,
            None,
            buffers.attention_proj,
            rows,
            hidden,
            hidden,
            stream,
        )?;
        library.cuda_residual_add_bf16_async(
            buffers.input,
            buffers.attention_proj,
            buffers.attention_residual,
            values,
            stream,
        )?;
        library.cuda_rmsnorm_bf16_async(
            buffers.attention_residual,
            buffers.norm1_weight,
            buffers.mlp_norm,
            rows as i32,
            hidden as i32,
            1.0e-5,
            stream,
        )?;
        library.cuda_linear_bf16_cublas_async(
            buffers.mlp_norm,
            buffers.probe_a_weight,
            None,
            buffers.probe_a_out,
            rows,
            hidden,
            hidden,
            stream,
        )?;
        library.cuda_linear_bf16_cublas_async(
            buffers.mlp_norm,
            buffers.probe_b_weight,
            None,
            buffers.probe_b_out,
            rows,
            hidden,
            hidden,
            stream,
        )?;
        library.cuda_residual_add_bf16_async(
            buffers.probe_a_out,
            buffers.probe_b_out,
            buffers.probe_mix,
            values,
            stream,
        )?;
        library.cuda_silu_gated_mlp_rows_bf16_down_stride_async(
            buffers.mlp_norm,
            buffers.gate_weight,
            buffers.up_weight,
            buffers.down_weight,
            buffers.mlp_out,
            rows,
            hidden,
            intermediate,
            intermediate,
            stream,
        )?;
        library.cuda_residual_add_bf16_async(
            buffers.probe_mix,
            buffers.mlp_out,
            buffers.mlp_delta,
            values,
            stream,
        )?;
        library.cuda_residual_add_bf16_async(
            buffers.attention_residual,
            buffers.mlp_delta,
            buffers.output,
            values,
            stream,
        )?;
    }
    Ok(())
}

fn coord_dense_envelope_bf16_capture_identity(
    buffers: CoordDenseEnvelopeBf16Buffers,
    rows: usize,
    hidden: usize,
    intermediate: usize,
) -> usize {
    graph_capture_identity(&[
        buffers.input.ptr as usize,
        buffers.norm0_weight.ptr as usize,
        buffers.norm0_out.ptr as usize,
        buffers.q_weight.ptr as usize,
        buffers.q_out.ptr as usize,
        buffers.k_weight.ptr as usize,
        buffers.k_out.ptr as usize,
        buffers.v_weight.ptr as usize,
        buffers.v_out.ptr as usize,
        buffers.attention_out.ptr as usize,
        buffers.o_weight.ptr as usize,
        buffers.attention_proj.ptr as usize,
        buffers.attention_residual.ptr as usize,
        buffers.norm1_weight.ptr as usize,
        buffers.mlp_norm.ptr as usize,
        buffers.probe_a_weight.ptr as usize,
        buffers.probe_a_out.ptr as usize,
        buffers.probe_b_weight.ptr as usize,
        buffers.probe_b_out.ptr as usize,
        buffers.probe_mix.ptr as usize,
        buffers.gate_weight.ptr as usize,
        buffers.up_weight.ptr as usize,
        buffers.down_weight.ptr as usize,
        buffers.mlp_out.ptr as usize,
        buffers.mlp_delta.ptr as usize,
        buffers.output.ptr as usize,
        rows,
        hidden,
        intermediate,
    ])
}

fn graph_capture_identity(parts: &[usize]) -> usize {
    parts.iter().fold(0xcbf29ce484222325_usize, |hash, part| {
        hash.wrapping_mul(0x100000001b3).wrapping_add(*part)
    })
}

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn capture_or_update_coord_sparse_a_envelope_bf16_graph(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    signature: CoordinatorCudaGraphSignature,
    buffers: CoordSparseAEnvelopeBf16Buffers,
    rows: usize,
    hidden: usize,
    intermediate: usize,
    experts: usize,
    top_k: usize,
    label: &'static str,
) -> Result<()> {
    let values = rows
        .checked_mul(hidden)
        .with_context(|| format!("{label} value count overflows usize"))?;
    let heads = 1_usize;
    let nope_dim = hidden;
    let rope_dim = hidden;
    let v_dim = hidden;
    let attention_scale = 0.5_f32;
    let capture_identity = coord_sparse_a_envelope_bf16_capture_identity(
        buffers,
        rows,
        hidden,
        intermediate,
        experts,
        top_k,
    );
    if !slot.has_captured_graph_identity(
        CoordinatorCudaGraphProgram::CoordSparseAEnvelopeBf16,
        signature,
        capture_identity,
    ) {
        unsafe {
            library
                .cuda_linear_bf16_cublas_async(
                    buffers.input,
                    buffers.q_nope_weight,
                    None,
                    buffers.q_nope_out,
                    rows,
                    hidden,
                    hidden,
                    slot.stream_ptr(),
                )
                .with_context(|| format!("warming CUDA cuBLAS {label} before graph capture"))?;
        }
    }
    slot.capture_or_update_graph_exec(
        library,
        CoordinatorCudaGraphProgram::CoordSparseAEnvelopeBf16,
        signature,
        capture_identity,
        |library, stream, _workspace| unsafe {
            capture_coord_sparse_a_envelope_bf16_ops(
                library,
                buffers,
                rows,
                hidden,
                intermediate,
                experts,
                top_k,
                values,
                heads,
                nope_dim,
                rope_dim,
                v_dim,
                attention_scale,
                stream,
            )
            .with_context(|| format!("capturing async CUDA cuBLAS {label}"))
        },
    )?;
    slot.launch_captured_graph_identity(
        library,
        CoordinatorCudaGraphProgram::CoordSparseAEnvelopeBf16,
        signature,
        capture_identity,
    )
}

#[allow(clippy::too_many_arguments)]
unsafe fn capture_coord_sparse_a_envelope_bf16_ops(
    library: &'static NativeLibrary,
    buffers: CoordSparseAEnvelopeBf16Buffers,
    rows: usize,
    hidden: usize,
    intermediate: usize,
    experts: usize,
    top_k: usize,
    values: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    attention_scale: f32,
    stream: *mut c_void,
) -> Result<()> {
    unsafe {
        library.cuda_rmsnorm_bf16_async(
            buffers.input,
            buffers.norm0_weight,
            buffers.norm0_out,
            rows as i32,
            hidden as i32,
            1.0e-5,
            stream,
        )?;
        library.cuda_linear_bf16_cublas_async(
            buffers.norm0_out,
            buffers.q_nope_weight,
            None,
            buffers.q_nope_out,
            rows,
            hidden,
            hidden,
            stream,
        )?;
        library.cuda_linear_bf16_cublas_async(
            buffers.norm0_out,
            buffers.q_rope_weight,
            None,
            buffers.q_rope_out,
            rows,
            hidden,
            hidden,
            stream,
        )?;
        library.cuda_linear_bf16_cublas_async(
            buffers.norm0_out,
            buffers.k_nope_weight,
            None,
            buffers.k_nope_out,
            rows,
            hidden,
            hidden,
            stream,
        )?;
        library.cuda_linear_bf16_cublas_async(
            buffers.norm0_out,
            buffers.k_rope_weight,
            None,
            buffers.k_rope_out,
            rows,
            hidden,
            hidden,
            stream,
        )?;
        library.cuda_linear_bf16_cublas_async(
            buffers.norm0_out,
            buffers.value_weight,
            None,
            buffers.value_out,
            rows,
            hidden,
            hidden,
            stream,
        )?;
        library.cuda_mla_rope_attention_bf16_async(
            buffers.q_nope_out,
            buffers.q_rope_out,
            buffers.k_nope_out,
            buffers.k_rope_out,
            buffers.value_out,
            buffers.attention_out,
            rows,
            heads,
            nope_dim,
            rope_dim,
            v_dim,
            attention_scale,
            stream,
        )?;
        library.cuda_linear_bf16_cublas_async(
            buffers.attention_out,
            buffers.o_weight,
            None,
            buffers.attention_proj,
            rows,
            hidden,
            hidden,
            stream,
        )?;
        library.cuda_residual_add_bf16_async(
            buffers.input,
            buffers.attention_proj,
            buffers.attention_residual,
            values,
            stream,
        )?;
        library.cuda_rmsnorm_bf16_async(
            buffers.attention_residual,
            buffers.norm1_weight,
            buffers.moe_norm,
            rows as i32,
            hidden as i32,
            1.0e-5,
            stream,
        )?;
        library.cuda_silu_gated_mlp_rows_bf16_down_stride_async(
            buffers.moe_norm,
            buffers.gate_weight,
            buffers.up_weight,
            buffers.down_weight,
            buffers.shared_out,
            rows,
            hidden,
            intermediate,
            intermediate,
            stream,
        )?;
        library.cuda_router_topk_bf16_async(
            buffers.moe_norm,
            buffers.router_weight,
            buffers.correction_bias,
            buffers.topk_indices,
            buffers.topk_scores,
            buffers.topk_weights,
            rows,
            hidden,
            experts,
            top_k,
            stream,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn coord_sparse_a_envelope_bf16_capture_identity(
    buffers: CoordSparseAEnvelopeBf16Buffers,
    rows: usize,
    hidden: usize,
    intermediate: usize,
    experts: usize,
    top_k: usize,
) -> usize {
    graph_capture_identity(&[
        buffers.input.ptr as usize,
        buffers.norm0_weight.ptr as usize,
        buffers.norm0_out.ptr as usize,
        buffers.q_nope_weight.ptr as usize,
        buffers.q_nope_out.ptr as usize,
        buffers.q_rope_weight.ptr as usize,
        buffers.q_rope_out.ptr as usize,
        buffers.k_nope_weight.ptr as usize,
        buffers.k_nope_out.ptr as usize,
        buffers.k_rope_weight.ptr as usize,
        buffers.k_rope_out.ptr as usize,
        buffers.value_weight.ptr as usize,
        buffers.value_out.ptr as usize,
        buffers.attention_out.ptr as usize,
        buffers.o_weight.ptr as usize,
        buffers.attention_proj.ptr as usize,
        buffers.attention_residual.ptr as usize,
        buffers.norm1_weight.ptr as usize,
        buffers.moe_norm.ptr as usize,
        buffers.gate_weight.ptr as usize,
        buffers.up_weight.ptr as usize,
        buffers.down_weight.ptr as usize,
        buffers.shared_out.ptr as usize,
        buffers.router_weight.ptr as usize,
        buffers.correction_bias.ptr as usize,
        buffers.topk_indices.ptr as usize,
        buffers.topk_scores.ptr as usize,
        buffers.topk_weights.ptr as usize,
        rows,
        hidden,
        intermediate,
        experts,
        top_k,
    ])
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn coordinator_cuda_graph_workspace_registry(
) -> Result<&'static CoordinatorCudaGraphWorkspaceRegistry> {
    COORDINATOR_CUDA_GRAPH_WORKSPACES.with(|registry| {
        let existing = *registry.borrow();
        if let Some(registry) = existing {
            return Ok(registry);
        }
        let library = cuda_native_library()?;
        let initialized = CoordinatorCudaGraphWorkspaceRegistry::glm52_bf16(library)?;
        let initialized = Box::leak(Box::new(initialized));
        *registry.borrow_mut() = Some(initialized);
        Ok(initialized)
    })
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn with_coordinator_cuda_graph_workspace_slot<T>(
    key: &CoordinatorGraphKey,
    action: impl FnOnce(&'static NativeLibrary, *mut c_void, &mut CoordinatorCudaWorkspace) -> Result<T>,
) -> Result<T> {
    with_coordinator_cuda_graph_slot(key, |library, slot| {
        let before = slot.workspace.scratch_slot_states();
        let stream = slot.stream_ptr();
        let result = action(library, stream, &mut slot.workspace);
        // Legacy callers can still resize scratch buffers directly through the
        // raw workspace borrow. Preserve captured graph execs across stable
        // reuse, but drop them if an existing scratch pointer/capacity changed.
        slot.clear_captured_graphs_if_scratch_changed(before);
        result
    })
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn with_coordinator_cuda_graph_slot<T>(
    key: &CoordinatorGraphKey,
    action: impl FnOnce(&'static NativeLibrary, &mut CoordinatorCudaGraphWorkspaceSlot) -> Result<T>,
) -> Result<T> {
    let library = cuda_native_library()?;
    let registry = coordinator_cuda_graph_workspace_registry()?;
    let mut slot = registry.slot_guard_for_key(key)?;
    action(library, &mut slot)
}
