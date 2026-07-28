use super::*;
use crate::python_graph_capture::{
    attention_python_capture_enabled, coordinator_python_capture_enabled,
    coordinator_python_capture_startup_open, launch_python_graph_capture, PythonDeviceBufferArg,
    PythonGraphCaptureLaunch, PythonKernelArg,
};
use anyhow::{Context, Result};
use glmrt_core::{
    CoordinatorGraphInstancePlan, CoordinatorGraphKey, CoordinatorGraphShape, KvCacheDType,
    LayerId, LayerWaveMode, COORDINATOR_GRAPH_INSTANCE_COUNT,
    COORDINATOR_GRAPH_PREFILL_BUCKET_ROWS, GLM52_DSA_INDEXER_LAYER_IDS_WITH_MTP,
    GLM52_DSA_INDEX_HEAD_DIM, GLM52_FIRST_K_DENSE_REPLACE, GLM52_HIDDEN_BF16_BYTES,
    GLM52_HIDDEN_SIZE, GLM52_MLA_KV_LORA_RANK, GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN,
    GLM52_MLA_QK_ROPE_HEAD_DIM, GLM52_MTP_LAYER_ID, GLM52_NUM_HIDDEN_LAYERS,
    GLM52_ROUTED_SCALING_FACTOR, GLM52_TOP_K,
};
use glmrt_ffi::{
    GlmrtCudaGraphCaptureInfo, GlmrtDeviceBuffer, GlmrtHostBuffer, NativeLibrary,
    GLMRT_CUDA_GLM_DSA_INDEX_HEADS, GLMRT_CUDA_GLM_DSA_PACKED_PAGE_BYTES,
    GLMRT_CUDA_GLM_DSA_PAGE_SIZE, GLMRT_CUDA_ROUTER_TOPK_MAX_K, GLMRT_CUDA_SAMPLE_TOPK_MAX_K,
};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::env;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::slice;
use std::sync::{Mutex, OnceLock};

#[allow(dead_code)]
pub(in crate::commands::real_full) const CPU_REFERENCE_CAUSAL_ATTENTION_BACKEND: &str =
    "cpu-reference-causal-attention";
#[allow(dead_code)]
pub(in crate::commands::real_full) const CUDA_REFERENCE_CAUSAL_ATTENTION_BACKEND: &str =
    "cuda-reference-causal-attention-f32";
pub(in crate::commands::real_full) const CPU_REFERENCE_CAUSAL_ATTENTION_BF16_BACKEND: &str =
    "cpu-reference-causal-attention-bf16";
pub(in crate::commands::real_full) const CUDA_REFERENCE_CAUSAL_ATTENTION_BF16_BACKEND: &str =
    "cuda-reference-causal-attention-bf16";
#[allow(dead_code)]
pub(in crate::commands::real_full) const CPU_REFERENCE_ROPE_BACKEND: &str = "cpu-reference-rope";
#[allow(dead_code)]
pub(in crate::commands::real_full) const CUDA_REFERENCE_ROPE_BACKEND: &str =
    "cuda-reference-rope-f32";
pub(in crate::commands::real_full) const CPU_REFERENCE_ROPE_BF16_BACKEND: &str =
    "cpu-reference-rope-bf16";
pub(in crate::commands::real_full) const CUDA_REFERENCE_ROPE_BF16_BACKEND: &str =
    "cuda-reference-rope-bf16";
#[allow(dead_code)]
pub(in crate::commands::real_full) const CPU_REFERENCE_MLA_ROPE_ATTENTION_BACKEND: &str =
    "cpu-reference-mla-rope-attention";
pub(in crate::commands::real_full) const CPU_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND: &str =
    "cpu-reference-mla-rope-attention-bf16";
pub(in crate::commands::real_full) const CUDA_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND: &str =
    "cuda-reference-mla-rope-attention-bf16";
pub(in crate::commands::real_full) const B12X_MLA_ROPE_ATTENTION_BF16_BACKEND: &str =
    "b12x-mla-rope-attention-bf16";
pub(in crate::commands::real_full) const FLASHINFER_MLA_ROPE_ATTENTION_BF16_BACKEND: &str =
    "flashinfer-mla-rope-attention-bf16";
pub(in crate::commands::real_full) const FLASHINFER_CUDNN_MLA_ROPE_ATTENTION_BF16_BACKEND: &str =
    "flashinfer-cudnn-mla-rope-attention-bf16";
pub(in crate::commands::real_full) const FLASHINFER_COMPRESSED_MLA_DECODE_BF16_BACKEND: &str =
    "flashinfer-compressed-mla-decode-bf16";
pub(in crate::commands::real_full) const FLASHINFER_COMPRESSED_MLA_DECODE_FP8_BACKEND: &str =
    "flashinfer-compressed-mla-decode-fp8-unpacked";
pub(in crate::commands::real_full) const FLASHINFER_PACKED_FP8_MLA_DECODE_BACKEND: &str =
    "flashinfer-packed-fp8-mla-decode-sm120";
pub(in crate::commands::real_full) const FLASHINFER_GLM_DSA_SPARSE_MLA_PREFILL_BACKEND: &str =
    "b12x-glm-dsa-flashinfer-packed-fp8-sparse-mla-prefill-sm120";
pub(in crate::commands::real_full) const SPARKINFER_GLM_DSA_SPARSE_NVFP4_MLA_BACKEND: &str =
    "sparkinfer-glm-dsa-native-packed-nvfp4-sparse-mla-sm120";
pub(in crate::commands::real_full) const FLASHINFER_COMPRESSED_MLA_DECODE_NVFP4_BACKEND: &str =
    "flashinfer-compressed-mla-decode-nvfp4-unpacked";
pub(in crate::commands::real_full) const CUDA_REFERENCE_MLA_KV_CACHE_UNPACK_BF16_BACKEND: &str =
    "cuda-reference-mla-kv-cache-unpack-bf16";
pub(in crate::commands::real_full) const CUDA_REFERENCE_MLA_KV_PROJECTED_SPLIT_BF16_BACKEND: &str =
    "cuda-reference-mla-kv-projected-split-bf16";
pub(in crate::commands::real_full) const CUDA_REFERENCE_MLA_KV_PREPARE_BF16_BACKEND: &str =
    "cuda-reference-mla-kv-prepare-bf16";
const B12X_MLA_CAPTURE_MODULE: &str = "b12x_mla_capture";
const B12X_MLA_CAPTURE_FUNCTION: &str = "capture_mla_rope_attention";
const FLASHINFER_MLA_PREPARE_FUNCTION: &str = "prepare_flashinfer_mla_rope_attention";
const FLASHINFER_MLA_CAPTURE_FUNCTION: &str = "capture_flashinfer_mla_rope_attention";
const FLASHINFER_CUDNN_MLA_PREPARE_FUNCTION: &str = "prepare_flashinfer_cudnn_mla_rope_attention";
const FLASHINFER_CUDNN_MLA_CAPTURE_FUNCTION: &str = "capture_flashinfer_cudnn_mla_rope_attention";
const FLASHINFER_COMPRESSED_MLA_PREPARE_FUNCTION: &str =
    "prepare_flashinfer_compressed_mla_decode_chunk";
const FLASHINFER_COMPRESSED_MLA_CAPTURE_FUNCTION: &str =
    "capture_flashinfer_compressed_mla_decode_chunk";
const FLASHINFER_PACKED_FP8_MLA_PREPARE_FUNCTION: &str = "prepare_flashinfer_packed_fp8_mla_decode";
const FLASHINFER_PACKED_FP8_MLA_CAPTURE_FUNCTION: &str = "capture_flashinfer_packed_fp8_mla_decode";
const B12X_GLM_DSA_PREFILL_PREPARE_FUNCTION: &str = "prepare_b12x_glm_dsa_indexer_prefill";
const B12X_GLM_DSA_PREFILL_CAPTURE_FUNCTION: &str = "capture_b12x_glm_dsa_indexer_prefill";
const FLASHINFER_PACKED_FP8_MLA_PREFILL_PREPARE_FUNCTION: &str =
    "prepare_flashinfer_packed_fp8_mla_prefill";
const FLASHINFER_PACKED_FP8_MLA_PREFILL_CAPTURE_FUNCTION: &str =
    "capture_flashinfer_packed_fp8_mla_prefill";
const SPARKINFER_NVFP4_MLA_DECODE_PREPARE_FUNCTION: &str = "prepare_sparkinfer_nvfp4_mla_decode";
const SPARKINFER_NVFP4_MLA_DECODE_CAPTURE_FUNCTION: &str = "capture_sparkinfer_nvfp4_mla_decode";
const SPARKINFER_NVFP4_MLA_PREFILL_PREPARE_FUNCTION: &str = "prepare_sparkinfer_nvfp4_mla_prefill";
const SPARKINFER_NVFP4_MLA_PREFILL_CAPTURE_FUNCTION: &str = "capture_sparkinfer_nvfp4_mla_prefill";
const FLASHINFER_SINGLE_PREFILL_TMP_BYTES: usize = 32 * 1024 * 1024;
const FLASHINFER_CUDNN_PREFILL_TMP_BYTES: usize = 128 * 1024 * 1024;
const FLASHINFER_MLA_SUFFIX_QUERY_FLOOR_ROWS: usize = 512;
const DEFAULT_FLASHINFER_CUDNN_MLA_SUFFIX_QUERY_CAPACITY: usize = 2_048;
const MAX_FLASHINFER_CUDNN_MLA_SUFFIX_QUERY_CAPACITY: usize = 2_048;
const FLASHINFER_CUDNN_MLA_SUFFIX_QUERY_CAPACITY_ENV: &str =
    "GLMRT_REAL_FULL_CUDNN_MLA_SUFFIX_QUERY_CAPACITY";
const FLASHINFER_CUDNN_MLA_SUFFIX_MAX_ROW_CAPACITY: usize =
    COORDINATOR_GRAPH_PREFILL_BUCKET_ROWS[COORDINATOR_GRAPH_PREFILL_BUCKET_ROWS.len() - 1];
const FLASHINFER_COMPRESSED_MLA_MAX_CHUNK_ROWS: usize = 2_048;
const FLASHINFER_COMPRESSED_MLA_EXACT_TAIL_ROWS: usize = 32;
const FLASHINFER_PACKED_FP8_MLA_ROW_BYTES: usize = 656;
const FLASHINFER_PACKED_FP8_MLA_BUCKETS: [usize; 4] = [128, 512, 1_024, 2_048];
const FLASHINFER_PACKED_FP8_MLA_MAX_QUERY_ROWS: usize = 16;
const GLM_DSA_NVFP4_SHORT_TOPK_MAX_QUERY_ROWS: usize = 64;
const GLM_DSA_PREFILL_TOPK: usize = 2_048;
const GLM_DSA_PREFILL_SUPERTILE_K: usize = 32_768;
const GLM_DSA_PREFILL_QUERY_BUCKETS: [usize; 10] = [1, 8, 16, 32, 64, 128, 256, 512, 1_024, 2_048];
const GLM_DSA_NVFP4_QUERY_BUCKETS: [usize; 12] =
    [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1_024, 2_048];
pub(in crate::commands::real_full) const GLM_DSA_PREFILL_MAX_QUERY_ROWS: usize = 2_048;
// One million physical tokens leaves room above the planned 600K shared pool.
// The paged-tiled selector's scratch is bounded by its 32K K supertile rather
// than total cache pages; q=2048 still needs 369,116,160 bytes at 600K.
const GLM_DSA_PREFILL_MAX_CACHE_PAGES: usize = 16_384;
// This is one reusable coordinator workspace, not a per-layer allocation.
const GLM_DSA_PREFILL_SELECTOR_SCRATCH_BYTES: usize = 369_116_160;
const GLM_DSA_BF16_GEMM_QUERY_BATCH_ROWS: usize = 64;
const GLM_DSA_BF16_GEMM_K_BYTES: usize =
    GLM_DSA_BF16_GEMM_QUERY_BATCH_ROWS * GLM_DSA_PREFILL_TOPK * (512 + 64) * 2;
const GLM_DSA_BF16_GEMM_V_BYTES: usize =
    GLM_DSA_BF16_GEMM_QUERY_BATCH_ROWS * GLM_DSA_PREFILL_TOPK * 512 * 2;
const GLM_DSA_BF16_GEMM_SCORE_BYTES: usize =
    GLM_DSA_BF16_GEMM_QUERY_BATCH_ROWS * 64 * GLM_DSA_PREFILL_TOPK * 2;
const GLM_DSA_BF16_GEMM_SCRATCH_BYTES: usize =
    GLM_DSA_BF16_GEMM_K_BYTES + GLM_DSA_BF16_GEMM_V_BYTES + GLM_DSA_BF16_GEMM_SCORE_BYTES;
const GLM_DSA_PREFILL_SCORE_SCALE: f32 = 1.0 / 64.0;
const GLM_DSA_OUTPUT_VALIDATE_ENV: &str = "GLMRT_REAL_FULL_DSA_OUTPUT_VALIDATE";
const GLM_DSA_INPUT_VALIDATE_MIN_ROWS_ENV: &str = "GLMRT_REAL_FULL_DSA_INPUT_VALIDATE_MIN_ROWS";
const REAL_FULL_ATTENTION_CUDA_TIMING_ENV: &str = "GLMRT_REAL_FULL_ATTENTION_CUDA_TIMING";
const W8A16_ASYNC_ATTENTION_ENV: &str = "GLMRT_COORDINATOR_W8A16_ASYNC_ATTENTION";
const PACKED_FP8_MLA_DIRECT_HIDDEN_OUTPUT_ENV: &str =
    "GLMRT_REAL_FULL_PACKED_FP8_MLA_DIRECT_HIDDEN_OUTPUT";

fn glm_dsa_sparse_mla_attention_topk(
    kv_dtype: KvCacheDType,
    query_bucket_rows: usize,
    total_rows: usize,
) -> usize {
    if kv_dtype != KvCacheDType::Nvfp4
        || query_bucket_rows > GLM_DSA_NVFP4_SHORT_TOPK_MAX_QUERY_ROWS
    {
        return GLM_DSA_PREFILL_TOPK;
    }
    let live_topk = total_rows.clamp(1, GLM_DSA_PREFILL_TOPK);
    FLASHINFER_PACKED_FP8_MLA_BUCKETS
        .into_iter()
        .find(|&bucket| bucket >= live_topk)
        .unwrap_or(GLM_DSA_PREFILL_TOPK)
}

fn glm_dsa_sparse_mla_query_bucket(kv_dtype: KvCacheDType, query_rows: usize) -> Option<usize> {
    let buckets = if matches!(kv_dtype, KvCacheDType::Bf16 | KvCacheDType::Nvfp4) {
        GLM_DSA_NVFP4_QUERY_BUCKETS.as_slice()
    } else {
        GLM_DSA_PREFILL_QUERY_BUCKETS.as_slice()
    };
    buckets.iter().copied().find(|bucket| query_rows <= *bucket)
}
const GLMRT_B12X_MLA_MODULE_ENV: &str = "GLMRT_B12X_MLA_MODULE";
const GLMRT_B12X_MLA_FUNCTION_ENV: &str = "GLMRT_B12X_MLA_FUNCTION";

struct AttentionCudaEventTimeline {
    library: &'static NativeLibrary,
    events: Vec<*mut c_void>,
}

fn parse_flashinfer_cudnn_mla_suffix_query_capacity(value: &str) -> Option<usize> {
    value.trim().parse::<usize>().ok().filter(|capacity| {
        capacity.is_power_of_two()
            && (FLASHINFER_MLA_SUFFIX_QUERY_FLOOR_ROWS
                ..=MAX_FLASHINFER_CUDNN_MLA_SUFFIX_QUERY_CAPACITY)
                .contains(capacity)
    })
}

fn packed_fp8_mla_direct_hidden_output_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(PACKED_FP8_MLA_DIRECT_HIDDEN_OUTPUT_ENV)
            .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
            .unwrap_or(true)
    })
}

fn w8a16_async_attention_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(W8A16_ASYNC_ATTENTION_ENV)
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false)
    })
}

fn flashinfer_cudnn_mla_suffix_query_capacity() -> usize {
    static CAPACITY: OnceLock<usize> = OnceLock::new();
    *CAPACITY.get_or_init(|| {
        env::var(FLASHINFER_CUDNN_MLA_SUFFIX_QUERY_CAPACITY_ENV)
            .ok()
            .as_deref()
            .and_then(parse_flashinfer_cudnn_mla_suffix_query_capacity)
            .unwrap_or(DEFAULT_FLASHINFER_CUDNN_MLA_SUFFIX_QUERY_CAPACITY)
    })
}

impl AttentionCudaEventTimeline {
    fn enabled() -> bool {
        env::var(REAL_FULL_ATTENTION_CUDA_TIMING_ENV)
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false)
    }

    fn new(library: &'static NativeLibrary, count: usize) -> Result<Self> {
        let mut timeline = Self {
            library,
            events: Vec::with_capacity(count),
        };
        for _ in 0..count {
            timeline.events.push(
                library
                    .cuda_event_create()
                    .context("creating attention CUDA timing event")?,
            );
        }
        Ok(timeline)
    }

    unsafe fn record(&self, index: usize, stream: *mut c_void, label: &str) -> Result<()> {
        self.library
            .cuda_event_record(self.events[index], stream)
            .with_context(|| format!("recording attention CUDA timing event {label}"))
    }

    unsafe fn elapsed_ms(&self, start: usize, end: usize, label: &str) -> Result<f64> {
        self.library
            .cuda_event_elapsed_ms(self.events[start], self.events[end])
            .map(f64::from)
            .with_context(|| format!("reading attention CUDA timing interval {label}"))
    }
}

impl Drop for AttentionCudaEventTimeline {
    fn drop(&mut self) {
        for event in self.events.drain(..) {
            let _ = unsafe { self.library.cuda_event_destroy(event) };
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FlashinferMlaCaptureShape {
    rows: usize,
    query_row_offset: usize,
    query_rows: usize,
    kv_prefix_padding: usize,
    query_prefix_padding: usize,
}

#[derive(Debug, Clone, Copy)]
struct FlashinferCudnnMlaSuffixBuffers {
    q: GlmrtDeviceBuffer,
    k: GlmrtDeviceBuffer,
    workspace: GlmrtDeviceBuffer,
    output: GlmrtDeviceBuffer,
    q_nope: GlmrtDeviceBuffer,
    q_rope: GlmrtDeviceBuffer,
    k_nope: GlmrtDeviceBuffer,
    k_rope: GlmrtDeviceBuffer,
    values: GlmrtDeviceBuffer,
}

#[derive(Default)]
struct FlashinferCudnnMlaSuffixWorkspace {
    q: ReusableDeviceBuffer,
    k: ReusableDeviceBuffer,
    workspace: ReusableDeviceBuffer,
    output: ReusableDeviceBuffer,
    q_nope: ReusableDeviceBuffer,
    q_rope: ReusableDeviceBuffer,
    k_nope: ReusableDeviceBuffer,
    k_rope: ReusableDeviceBuffer,
    values: ReusableDeviceBuffer,
    initialized: bool,
}

thread_local! {
    static FLASHINFER_CUDNN_MLA_SUFFIX_WORKSPACE: RefCell<FlashinferCudnnMlaSuffixWorkspace> =
        RefCell::new(FlashinferCudnnMlaSuffixWorkspace::default());
    static FLASHINFER_CUDNN_MLA_SUFFIX_PREWARMED: Cell<bool> = const { Cell::new(false) };
}

impl FlashinferCudnnMlaSuffixWorkspace {
    fn ensure(
        &mut self,
        library: &'static NativeLibrary,
    ) -> Result<FlashinferCudnnMlaSuffixBuffers> {
        const MAX_HEADS: usize = 64;
        const NOPE_DIM: usize = 192;
        const ROPE_DIM: usize = 64;
        const V_DIM: usize = 256;
        const QK_DIM: usize = NOPE_DIM + ROPE_DIM;

        let tensor_bytes = |rows: usize, heads: usize, dim: usize, label: &str| {
            rows.checked_mul(heads)
                .and_then(|values| values.checked_mul(dim))
                .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
                .with_context(|| format!("{label} byte count overflow"))
        };
        let q_rows = flashinfer_cudnn_mla_suffix_query_capacity();
        let kv_rows = FLASHINFER_CUDNN_MLA_SUFFIX_MAX_ROW_CAPACITY;
        self.q.ensure_capacity(
            library,
            tensor_bytes(q_rows, MAX_HEADS, QK_DIM, "cuDNN MLA shared query")?,
            "FlashInfer/cuDNN MLA shared query",
        )?;
        self.k.ensure_capacity(
            library,
            tensor_bytes(kv_rows, MAX_HEADS, QK_DIM, "cuDNN MLA shared key")?,
            "FlashInfer/cuDNN MLA shared key",
        )?;
        self.workspace.ensure_capacity(
            library,
            FLASHINFER_CUDNN_PREFILL_TMP_BYTES,
            "FlashInfer/cuDNN MLA shared workspace",
        )?;
        self.output.ensure_capacity(
            library,
            tensor_bytes(q_rows, MAX_HEADS, V_DIM, "cuDNN MLA shared output")?,
            "FlashInfer/cuDNN MLA shared output",
        )?;
        self.q_nope.ensure_capacity(
            library,
            tensor_bytes(q_rows, MAX_HEADS, NOPE_DIM, "cuDNN MLA shared q_nope")?,
            "FlashInfer/cuDNN MLA shared q_nope",
        )?;
        self.q_rope.ensure_capacity(
            library,
            tensor_bytes(q_rows, MAX_HEADS, ROPE_DIM, "cuDNN MLA shared q_rope")?,
            "FlashInfer/cuDNN MLA shared q_rope",
        )?;
        self.k_nope.ensure_capacity(
            library,
            tensor_bytes(kv_rows, MAX_HEADS, NOPE_DIM, "cuDNN MLA shared k_nope")?,
            "FlashInfer/cuDNN MLA shared k_nope",
        )?;
        self.k_rope.ensure_capacity(
            library,
            tensor_bytes(kv_rows, 1, ROPE_DIM, "cuDNN MLA shared k_rope")?,
            "FlashInfer/cuDNN MLA shared k_rope",
        )?;
        self.values.ensure_capacity(
            library,
            tensor_bytes(kv_rows, MAX_HEADS, V_DIM, "cuDNN MLA shared values")?,
            "FlashInfer/cuDNN MLA shared values",
        )?;
        Ok(FlashinferCudnnMlaSuffixBuffers {
            q: self.q.buffer,
            k: self.k.buffer,
            workspace: self.workspace.buffer,
            output: self.output.buffer,
            q_nope: self.q_nope.buffer,
            q_rope: self.q_rope.buffer,
            k_nope: self.k_nope.buffer,
            k_rope: self.k_rope.buffer,
            values: self.values.buffer,
        })
    }

    fn initialize_for_capture(
        &mut self,
        library: &'static NativeLibrary,
    ) -> Result<FlashinferCudnnMlaSuffixBuffers> {
        let buffers = self.ensure(library)?;
        if !self.initialized {
            for buffer in [
                buffers.q_nope,
                buffers.q_rope,
                buffers.k_nope,
                buffers.k_rope,
                buffers.values,
            ] {
                library
                    .cuda_zero_bytes(buffer, buffer.bytes)
                    .context("zeroing shared cuDNN MLA suffix capture input")?;
            }
            self.initialized = true;
        }
        Ok(buffers)
    }
}

fn flashinfer_cudnn_mla_suffix_buffers(
    library: &'static NativeLibrary,
) -> Result<FlashinferCudnnMlaSuffixBuffers> {
    FLASHINFER_CUDNN_MLA_SUFFIX_WORKSPACE.with(|workspace| {
        workspace
            .try_borrow_mut()
            .map_err(|_| anyhow::anyhow!("cuDNN MLA suffix workspace is already borrowed"))?
            .ensure(library)
    })
}

#[derive(Debug, Clone, Copy)]
struct GlmDsaSparseMlaPrefillBuffers {
    selector_scratch: GlmrtDeviceBuffer,
    sparse_mid_out: GlmrtDeviceBuffer,
    sparse_mid_lse: GlmrtDeviceBuffer,
    page_table: GlmrtDeviceBuffer,
    cache_seqlens: GlmrtDeviceBuffer,
    active_width: GlmrtDeviceBuffer,
    selected_indices: GlmrtDeviceBuffer,
    topk_lengths: GlmrtDeviceBuffer,
    compacted_indices: GlmrtDeviceBuffer,
    dsa_query_raw: GlmrtDeviceBuffer,
    dsa_weights_raw: GlmrtDeviceBuffer,
    dsa_positions: GlmrtDeviceBuffer,
    dsa_query_fp8: GlmrtDeviceBuffer,
    dsa_weights: GlmrtDeviceBuffer,
    q_nope: GlmrtDeviceBuffer,
    q_rope: GlmrtDeviceBuffer,
    combined_query: GlmrtDeviceBuffer,
    head_major: GlmrtDeviceBuffer,
    sparse_latent: GlmrtDeviceBuffer,
    auxiliary: GlmrtDeviceBuffer,
    final_output: GlmrtDeviceBuffer,
    out_lse: GlmrtDeviceBuffer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GlmDsaSelectionKey {
    source_layer: usize,
    physical_token_base: usize,
    physical_page_table_ptr: usize,
    physical_page_table_key: u64,
    total_rows: usize,
    prefix_rows: usize,
    query_rows: usize,
}

#[derive(Default)]
struct GlmDsaSparseMlaPrefillWorkspace {
    selector_scratch: ReusableDeviceBuffer,
    page_table: ReusableDeviceBuffer,
    cache_seqlens: ReusableDeviceBuffer,
    active_width: ReusableDeviceBuffer,
    selected_indices: ReusableDeviceBuffer,
    small_prefill_selected_indices: ReusableDeviceBuffer,
    topk_lengths: ReusableDeviceBuffer,
    compacted_indices: ReusableDeviceBuffer,
    dsa_query_raw: ReusableDeviceBuffer,
    dsa_weights_raw: ReusableDeviceBuffer,
    dsa_positions: ReusableDeviceBuffer,
    dsa_query_fp8: ReusableDeviceBuffer,
    dsa_weights: ReusableDeviceBuffer,
    q_nope: ReusableDeviceBuffer,
    q_rope: ReusableDeviceBuffer,
    combined_query: ReusableDeviceBuffer,
    head_major: ReusableDeviceBuffer,
    sparse_latent: ReusableDeviceBuffer,
    auxiliary: ReusableDeviceBuffer,
    final_output: ReusableDeviceBuffer,
    out_lse: ReusableDeviceBuffer,
    max_tokens: usize,
    max_pages: usize,
    page_table_physical_page_base: Option<usize>,
    page_table_physical_mapping: Option<(usize, u64)>,
    initialized: bool,
    banked_selections: HashMap<GlmDsaSelectionKey, OwnedCoordinatorDeviceBuffer>,
}

thread_local! {
    static GLM_DSA_SPARSE_MLA_PREFILL_WORKSPACE: RefCell<GlmDsaSparseMlaPrefillWorkspace> =
        RefCell::new(GlmDsaSparseMlaPrefillWorkspace::default());
}

pub(in crate::commands::real_full) fn reset_glm_dsa_sparse_mla_transient_state() -> Result<()> {
    GLM_DSA_SPARSE_MLA_PREFILL_WORKSPACE.with(|workspace| {
        let mut workspace = workspace
            .try_borrow_mut()
            .map_err(|_| anyhow::anyhow!("GLM DSA sparse MLA workspace is already borrowed"))?;
        workspace.banked_selections.clear();
        Ok(())
    })
}

fn bank_glm_dsa_selection(
    workspace: &mut GlmDsaSparseMlaPrefillWorkspace,
    library: &'static NativeLibrary,
    selection_key: GlmDsaSelectionKey,
    selected_indices: GlmrtDeviceBuffer,
    stream: *mut c_void,
) -> Result<()> {
    let selection_row_bytes = GLM_DSA_PREFILL_TOPK
        .checked_mul(std::mem::size_of::<i32>())
        .context("GLM DSA banked-selection row bytes overflow")?;
    let selection_bytes = selection_key
        .query_rows
        .checked_mul(selection_row_bytes)
        .context("GLM DSA banked-selection bytes overflow")?;
    if !workspace.banked_selections.contains_key(&selection_key) {
        let allocation_rows = selection_key
            .query_rows
            .next_power_of_two()
            .min(GLM_DSA_PREFILL_MAX_QUERY_ROWS);
        let allocation_bytes = allocation_rows
            .checked_mul(selection_row_bytes)
            .context("GLM DSA banked-selection allocation bytes overflow")?;
        workspace.banked_selections.insert(
            selection_key,
            OwnedCoordinatorDeviceBuffer::new(
                library,
                allocation_bytes,
                "GLM DSA live selection bank",
            )?,
        );
    }
    let selection_bank = workspace
        .banked_selections
        .get(&selection_key)
        .expect("active GLM DSA selection bank was inserted")
        .buffer;
    unsafe {
        library
            .copy_d2d_async(selection_bank, selected_indices, selection_bytes, stream)
            .context("banking GLM DSA indices for shared layers")?;
    }
    Ok(())
}

impl GlmDsaSparseMlaPrefillWorkspace {
    fn ensure(
        &mut self,
        library: &'static NativeLibrary,
        max_tokens: usize,
        query_rows: usize,
    ) -> Result<GlmDsaSparseMlaPrefillBuffers> {
        const HEADS: usize = 64;
        const NOPE_DIM: usize = 192;
        const ROPE_DIM: usize = 64;
        const V_DIM: usize = 256;
        const RANK: usize = GLM52_MLA_KV_LORA_RANK;
        const DECODE_ROWS: usize = 64;
        const DECODE_SPLITS: usize = GLM_DSA_PREFILL_TOPK / GLMRT_CUDA_GLM_DSA_PAGE_SIZE;
        const DSA_QUERY_WIDTH: usize = GLMRT_CUDA_GLM_DSA_INDEX_HEADS * GLM52_DSA_INDEX_HEAD_DIM;

        anyhow::ensure!(
            max_tokens > 0 && max_tokens % GLMRT_CUDA_GLM_DSA_PAGE_SIZE == 0,
            "GLM DSA sparse MLA prefill requires max_tokens divisible by {}, got {max_tokens}",
            GLMRT_CUDA_GLM_DSA_PAGE_SIZE
        );
        let max_pages = max_tokens / GLMRT_CUDA_GLM_DSA_PAGE_SIZE;
        anyhow::ensure!(
            max_pages <= GLM_DSA_PREFILL_MAX_CACHE_PAGES,
            "GLM DSA sparse MLA prefill supports at most {} cache pages ({} tokens), got {max_pages}",
            GLM_DSA_PREFILL_MAX_CACHE_PAGES,
            GLM_DSA_PREFILL_MAX_CACHE_PAGES * GLMRT_CUDA_GLM_DSA_PAGE_SIZE
        );
        if self.initialized {
            anyhow::ensure!(
                self.max_tokens == max_tokens,
                "GLM DSA sparse MLA workspace was initialized for {} tokens and cannot be rebound to {max_tokens} after graph capture",
                self.max_tokens
            );
        }

        let bf16_bytes = std::mem::size_of::<u16>();
        let f32_bytes = std::mem::size_of::<f32>();
        let i32_bytes = std::mem::size_of::<i32>();
        let tensor_bytes = |rows: usize, heads: usize, width: usize, element_bytes: usize| {
            rows.checked_mul(heads)
                .and_then(|values| values.checked_mul(width))
                .and_then(|values| values.checked_mul(element_bytes))
                .context("GLM DSA sparse MLA workspace tensor bytes overflow")
        };

        self.selector_scratch.ensure_capacity(
            library,
            GLM_DSA_PREFILL_SELECTOR_SCRATCH_BYTES,
            "GLM DSA selector scratch",
        )?;
        let sparse_mid_out_width = DECODE_SPLITS
            .checked_mul(RANK)
            .context("GLM sparse MLA decode split-output width overflow")?;
        let sparse_mid_out_bytes =
            tensor_bytes(DECODE_ROWS, HEADS, sparse_mid_out_width, bf16_bytes)
                .context("GLM sparse MLA decode partial-output bytes overflow")?;
        let sparse_mid_lse_bytes = tensor_bytes(DECODE_ROWS, HEADS, DECODE_SPLITS, f32_bytes)
            .context("GLM sparse MLA decode partial-LSE bytes overflow")?;
        anyhow::ensure!(
            sparse_mid_out_bytes
                .checked_add(sparse_mid_lse_bytes)
                .is_some_and(|bytes| bytes <= self.selector_scratch.buffer.bytes),
            "GLM sparse MLA decode scratch does not fit in selector arena"
        );
        self.page_table.ensure_capacity(
            library,
            max_pages
                .checked_mul(i32_bytes)
                .context("GLM DSA page table bytes overflow")?,
            "GLM DSA shared page table row",
        )?;
        self.cache_seqlens.ensure_capacity(
            library,
            GLM_DSA_PREFILL_MAX_QUERY_ROWS * i32_bytes,
            "GLM DSA cache sequence lengths",
        )?;
        self.active_width
            .ensure_capacity(library, i32_bytes, "GLM DSA active page-table width")?;
        self.selected_indices.ensure_capacity(
            library,
            GLM_DSA_PREFILL_MAX_QUERY_ROWS
                .checked_mul(GLM_DSA_PREFILL_TOPK)
                .and_then(|values| values.checked_mul(i32_bytes))
                .context("GLM DSA selected-index bytes overflow")?,
            "GLM DSA selected physical indices",
        )?;
        self.small_prefill_selected_indices.ensure_capacity(
            library,
            FLASHINFER_PACKED_FP8_MLA_MAX_QUERY_ROWS
                .checked_mul(GLM_DSA_PREFILL_TOPK)
                .and_then(|values| values.checked_mul(i32_bytes))
                .context("GLM DSA small-prefill selected-index bytes overflow")?,
            "GLM DSA small-prefill selected physical indices",
        )?;
        self.topk_lengths.ensure_capacity(
            library,
            GLM_DSA_PREFILL_MAX_QUERY_ROWS * i32_bytes,
            "GLM DSA sparse top-k lengths",
        )?;
        let compacted_index_rows = DECODE_ROWS
            .checked_mul(GLM_DSA_PREFILL_TOPK)
            .context("GLM DSA compacted-index row count overflow")?;
        self.compacted_indices.ensure_capacity(
            library,
            compacted_index_rows
                .checked_mul(i32_bytes)
                .context("GLM DSA compacted-index bytes overflow")?,
            "GLM DSA native NVFP4 compacted top-k indices",
        )?;
        self.dsa_query_raw.ensure_capacity(
            library,
            tensor_bytes(
                GLM_DSA_PREFILL_MAX_QUERY_ROWS,
                1,
                DSA_QUERY_WIDTH,
                bf16_bytes,
            )?,
            "GLM DSA raw projected query staging",
        )?;
        self.dsa_weights_raw.ensure_capacity(
            library,
            tensor_bytes(
                GLM_DSA_PREFILL_MAX_QUERY_ROWS,
                GLMRT_CUDA_GLM_DSA_INDEX_HEADS,
                1,
                bf16_bytes,
            )?,
            "GLM DSA raw head-weight staging",
        )?;
        self.dsa_positions.ensure_capacity(
            library,
            GLM_DSA_PREFILL_MAX_QUERY_ROWS * std::mem::size_of::<u32>(),
            "GLM DSA query position staging",
        )?;
        self.dsa_query_fp8.ensure_capacity(
            library,
            tensor_bytes(
                GLM_DSA_PREFILL_MAX_QUERY_ROWS,
                GLMRT_CUDA_GLM_DSA_INDEX_HEADS,
                GLM52_DSA_INDEX_HEAD_DIM,
                1,
            )?,
            "GLM DSA prepared FP8 query",
        )?;
        self.dsa_weights.ensure_capacity(
            library,
            tensor_bytes(
                GLM_DSA_PREFILL_MAX_QUERY_ROWS,
                GLMRT_CUDA_GLM_DSA_INDEX_HEADS,
                1,
                f32_bytes,
            )?,
            "GLM DSA adjusted head weights",
        )?;
        self.q_nope.ensure_capacity(
            library,
            tensor_bytes(GLM_DSA_PREFILL_MAX_QUERY_ROWS, HEADS, NOPE_DIM, bf16_bytes)?,
            "GLM sparse MLA q-nope staging",
        )?;
        self.q_rope.ensure_capacity(
            library,
            tensor_bytes(GLM_DSA_PREFILL_MAX_QUERY_ROWS, HEADS, ROPE_DIM, bf16_bytes)?,
            "GLM sparse MLA q-rope staging",
        )?;
        self.combined_query.ensure_capacity(
            library,
            tensor_bytes(
                GLM_DSA_PREFILL_MAX_QUERY_ROWS,
                HEADS,
                RANK + ROPE_DIM,
                bf16_bytes,
            )?,
            "GLM sparse MLA absorbed query",
        )?;
        self.head_major.ensure_capacity(
            library,
            tensor_bytes(GLM_DSA_PREFILL_MAX_QUERY_ROWS, HEADS, RANK, bf16_bytes)?,
            "GLM sparse MLA head-major latent",
        )?;
        self.sparse_latent.ensure_capacity(
            library,
            tensor_bytes(GLM_DSA_PREFILL_MAX_QUERY_ROWS, HEADS, RANK, bf16_bytes)?,
            "GLM sparse MLA latent output",
        )?;
        self.auxiliary.ensure_capacity(
            library,
            tensor_bytes(GLM_DSA_PREFILL_MAX_QUERY_ROWS, HEADS, V_DIM, bf16_bytes)?,
            "GLM sparse MLA auxiliary head-major values",
        )?;
        self.final_output.ensure_capacity(
            library,
            tensor_bytes(GLM_DSA_PREFILL_MAX_QUERY_ROWS, HEADS, V_DIM, bf16_bytes)?,
            "GLM sparse MLA query-major output",
        )?;
        self.out_lse.ensure_capacity(
            library,
            tensor_bytes(GLM_DSA_PREFILL_MAX_QUERY_ROWS, HEADS, 1, f32_bytes)?,
            "GLM sparse MLA output LSE",
        )?;

        let buffers = GlmDsaSparseMlaPrefillBuffers {
            selector_scratch: self.selector_scratch.buffer,
            sparse_mid_out: device_buffer_byte_view(
                self.selector_scratch.buffer,
                0,
                sparse_mid_out_bytes,
                "GLM sparse MLA decode partial output",
            )?,
            sparse_mid_lse: device_buffer_byte_view(
                self.selector_scratch.buffer,
                sparse_mid_out_bytes,
                sparse_mid_lse_bytes,
                "GLM sparse MLA decode partial LSE",
            )?,
            page_table: self.page_table.buffer,
            cache_seqlens: self.cache_seqlens.buffer,
            active_width: self.active_width.buffer,
            selected_indices: if query_rows > 1
                && query_rows <= FLASHINFER_PACKED_FP8_MLA_MAX_QUERY_ROWS
            {
                self.small_prefill_selected_indices.buffer
            } else {
                self.selected_indices.buffer
            },
            topk_lengths: self.topk_lengths.buffer,
            compacted_indices: self.compacted_indices.buffer,
            dsa_query_raw: self.dsa_query_raw.buffer,
            dsa_weights_raw: self.dsa_weights_raw.buffer,
            dsa_positions: self.dsa_positions.buffer,
            dsa_query_fp8: self.dsa_query_fp8.buffer,
            dsa_weights: self.dsa_weights.buffer,
            q_nope: self.q_nope.buffer,
            q_rope: self.q_rope.buffer,
            combined_query: self.combined_query.buffer,
            head_major: self.head_major.buffer,
            sparse_latent: self.sparse_latent.buffer,
            auxiliary: self.auxiliary.buffer,
            final_output: self.final_output.buffer,
            out_lse: self.out_lse.buffer,
        };
        if !self.initialized {
            for buffer in [
                buffers.dsa_query_raw,
                buffers.dsa_weights_raw,
                buffers.dsa_positions,
                buffers.q_nope,
                buffers.q_rope,
            ] {
                library
                    .cuda_zero_bytes(buffer, buffer.bytes)
                    .context("zeroing GLM DSA sparse MLA stable input staging")?;
            }
            library
                .cuda_glm_dsa_page_table_init(buffers.page_table, 1, max_pages)
                .context("initializing shared sequential GLM DSA page-table row")?;
            self.max_tokens = max_tokens;
            self.max_pages = max_pages;
            self.page_table_physical_page_base = Some(0);
            self.page_table_physical_mapping = None;
            self.initialized = true;
        }
        Ok(buffers)
    }
}

pub(in crate::commands::real_full) fn glm_dsa_index_source_layer(
    layer_id: usize,
) -> Option<(usize, bool)> {
    if GLM52_DSA_INDEXER_LAYER_IDS_WITH_MTP.contains(&layer_id) {
        return Some((layer_id, true));
    }
    if layer_id >= GLM52_NUM_HIDDEN_LAYERS {
        return None;
    }
    if (3..=5).contains(&layer_id) {
        return Some((2, false));
    }
    if layer_id >= 7 {
        let source_layer = 6 + ((layer_id - 6) / 4) * 4;
        if source_layer <= 74 {
            return Some((source_layer, false));
        }
    }
    None
}

pub(in crate::commands::real_full) fn glm_dsa_sparse_mla_prefill_supported(
    query_rows: usize,
    max_tokens: usize,
) -> bool {
    attention_python_capture_enabled()
        && (1..=GLM_DSA_PREFILL_MAX_QUERY_ROWS).contains(&query_rows)
        && max_tokens > 0
        && max_tokens % GLMRT_CUDA_GLM_DSA_PAGE_SIZE == 0
        && max_tokens / GLMRT_CUDA_GLM_DSA_PAGE_SIZE <= GLM_DSA_PREFILL_MAX_CACHE_PAGES
}

#[allow(dead_code, clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn mla_rope_attention_rows(
    q_nope: &[f32],
    q_rope: &[f32],
    k_nope: &[f32],
    k_rope: &[f32],
    values: &[f32],
    rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    scale: f32,
) -> Result<MlaRopeAttentionOutput> {
    validate_mla_rope_attention_inputs(
        q_nope, q_rope, k_nope, k_rope, values, rows, heads, nope_dim, rope_dim, v_dim, scale,
    )?;
    Ok(cpu_mla_rope_attention_rows(
        q_nope, q_rope, k_nope, k_rope, values, rows, heads, nope_dim, rope_dim, v_dim, scale,
    ))
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(in crate::commands::real_full) fn mla_rope_attention_rows_bf16(
    q_nope_bf16: &[u8],
    q_rope_bf16: &[u8],
    k_nope_bf16: &[u8],
    k_rope_bf16: &[u8],
    values_bf16: &[u8],
    rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    scale: f32,
) -> Result<MlaRopeAttentionOutput> {
    validate_mla_rope_attention_bf16_inputs(
        q_nope_bf16,
        q_rope_bf16,
        k_nope_bf16,
        k_rope_bf16,
        values_bf16,
        rows,
        heads,
        nope_dim,
        rope_dim,
        v_dim,
        scale,
    )?;
    if cuda_reference_kernels_enabled() {
        return cuda_mla_rope_attention_rows_bf16(
            q_nope_bf16,
            q_rope_bf16,
            k_nope_bf16,
            k_rope_bf16,
            values_bf16,
            rows,
            heads,
            nope_dim,
            rope_dim,
            v_dim,
            scale,
        );
    }
    Ok(cpu_mla_rope_attention_rows_bf16(
        q_nope_bf16,
        q_rope_bf16,
        k_nope_bf16,
        k_rope_bf16,
        values_bf16,
        rows,
        heads,
        nope_dim,
        rope_dim,
        v_dim,
        scale,
    ))
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn mla_rope_attention_rows_bf16_for_layer(
    layer_id: usize,
    q_nope_bf16: &[u8],
    q_rope_bf16: &[u8],
    k_nope_bf16: &[u8],
    k_rope_bf16: &[u8],
    values_bf16: &[u8],
    rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    scale: f32,
) -> Result<MlaRopeAttentionOutput> {
    validate_mla_rope_attention_bf16_inputs(
        q_nope_bf16,
        q_rope_bf16,
        k_nope_bf16,
        k_rope_bf16,
        values_bf16,
        rows,
        heads,
        nope_dim,
        rope_dim,
        v_dim,
        scale,
    )?;
    if cuda_reference_kernels_enabled() {
        return cuda_mla_rope_attention_rows_bf16_for_layer(
            layer_id,
            q_nope_bf16,
            q_rope_bf16,
            k_nope_bf16,
            k_rope_bf16,
            values_bf16,
            rows,
            heads,
            nope_dim,
            rope_dim,
            v_dim,
            scale,
        );
    }
    Ok(cpu_mla_rope_attention_rows_bf16(
        q_nope_bf16,
        q_rope_bf16,
        k_nope_bf16,
        k_rope_bf16,
        values_bf16,
        rows,
        heads,
        nope_dim,
        rope_dim,
        v_dim,
        scale,
    ))
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn rope_rows(
    input: &[f32],
    positions: &[usize],
    rows: usize,
    heads: usize,
    rotary_dim: usize,
    theta: f32,
) -> Result<RopeOutput> {
    let positions = validate_rope_inputs(input, positions, rows, heads, rotary_dim, theta)?;
    if cuda_reference_kernels_enabled() {
        return cuda_rope_rows(input, &positions, rows, heads, rotary_dim, theta);
    }
    Ok(cpu_rope_rows(
        input, &positions, rows, heads, rotary_dim, theta,
    ))
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn rope_rows_bf16(
    input_bf16: &[u8],
    positions: &[usize],
    rows: usize,
    heads: usize,
    rotary_dim: usize,
    theta: f32,
) -> Result<RopeOutput> {
    let positions =
        validate_rope_bf16_inputs(input_bf16, positions, rows, heads, rotary_dim, theta)?;
    if cuda_reference_kernels_enabled() {
        return cuda_rope_rows_bf16(input_bf16, &positions, rows, heads, rotary_dim, theta);
    }
    Ok(cpu_rope_rows_bf16(
        input_bf16, &positions, rows, heads, rotary_dim, theta,
    ))
}

pub(in crate::commands::real_full) fn rope_rows_bf16_for_layer(
    layer_id: usize,
    input_bf16: &[u8],
    positions: &[usize],
    rows: usize,
    heads: usize,
    rotary_dim: usize,
    theta: f32,
) -> Result<RopeOutput> {
    let positions =
        validate_rope_bf16_inputs(input_bf16, positions, rows, heads, rotary_dim, theta)?;
    if cuda_reference_kernels_enabled() {
        return cuda_rope_rows_bf16_for_layer(
            layer_id, input_bf16, &positions, rows, heads, rotary_dim, theta,
        );
    }
    Ok(cpu_rope_rows_bf16(
        input_bf16, &positions, rows, heads, rotary_dim, theta,
    ))
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn causal_attention_rows(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    rows: usize,
    heads: usize,
    qk_dim: usize,
    v_dim: usize,
    scale: f32,
) -> Result<CausalAttentionOutput> {
    validate_causal_attention_inputs(queries, keys, values, rows, heads, qk_dim, v_dim, scale)?;
    if cuda_reference_kernels_enabled() {
        return cuda_causal_attention_rows(
            queries, keys, values, rows, heads, qk_dim, v_dim, scale,
        );
    }
    Ok(cpu_causal_attention_rows(
        queries, keys, values, rows, heads, qk_dim, v_dim, scale,
    ))
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn causal_attention_rows_bf16(
    queries_bf16: &[u8],
    keys_bf16: &[u8],
    values_bf16: &[u8],
    rows: usize,
    heads: usize,
    qk_dim: usize,
    v_dim: usize,
    scale: f32,
) -> Result<CausalAttentionOutput> {
    validate_causal_attention_bf16_inputs(
        queries_bf16,
        keys_bf16,
        values_bf16,
        rows,
        heads,
        qk_dim,
        v_dim,
        scale,
    )?;
    if cuda_reference_kernels_enabled() {
        return cuda_causal_attention_rows_bf16(
            queries_bf16,
            keys_bf16,
            values_bf16,
            rows,
            heads,
            qk_dim,
            v_dim,
            scale,
        );
    }
    Ok(cpu_causal_attention_rows_bf16(
        queries_bf16,
        keys_bf16,
        values_bf16,
        rows,
        heads,
        qk_dim,
        v_dim,
        scale,
    ))
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn causal_attention_rows_bf16_for_layer(
    layer_id: usize,
    queries_bf16: &[u8],
    keys_bf16: &[u8],
    values_bf16: &[u8],
    rows: usize,
    heads: usize,
    qk_dim: usize,
    v_dim: usize,
    scale: f32,
) -> Result<CausalAttentionOutput> {
    validate_causal_attention_bf16_inputs(
        queries_bf16,
        keys_bf16,
        values_bf16,
        rows,
        heads,
        qk_dim,
        v_dim,
        scale,
    )?;
    if cuda_reference_kernels_enabled() {
        return cuda_causal_attention_rows_bf16_for_layer(
            layer_id,
            queries_bf16,
            keys_bf16,
            values_bf16,
            rows,
            heads,
            qk_dim,
            v_dim,
            scale,
        );
    }
    Ok(cpu_causal_attention_rows_bf16(
        queries_bf16,
        keys_bf16,
        values_bf16,
        rows,
        heads,
        qk_dim,
        v_dim,
        scale,
    ))
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn validate_causal_attention_inputs(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    rows: usize,
    heads: usize,
    qk_dim: usize,
    v_dim: usize,
    scale: f32,
) -> Result<()> {
    if rows == 0 || heads == 0 || qk_dim == 0 || v_dim == 0 {
        anyhow::bail!(
            "real full causal attention requires non-zero shape, got rows={rows} heads={heads} qk_dim={qk_dim} v_dim={v_dim}"
        );
    }
    if !scale.is_finite() {
        anyhow::bail!("real full causal attention scale must be finite");
    }
    let qk_values = rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(qk_dim))
        .context("real full causal attention q/k shape overflows usize while validating coordinator kernel input")?;
    if queries.len() != qk_values {
        anyhow::bail!(
            "real full causal attention query length mismatch: expected {} got {}",
            qk_values,
            queries.len()
        );
    }
    if keys.len() != qk_values {
        anyhow::bail!(
            "real full causal attention key length mismatch: expected {} got {}",
            qk_values,
            keys.len()
        );
    }
    let value_count = rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(v_dim))
        .context("real full causal attention value shape overflows usize while validating coordinator kernel input")?;
    if values.len() != value_count {
        anyhow::bail!(
            "real full causal attention value length mismatch: expected {} got {}",
            value_count,
            values.len()
        );
    }
    Ok(())
}

pub(in crate::commands::real_full) fn validate_causal_attention_bf16_inputs(
    queries_bf16: &[u8],
    keys_bf16: &[u8],
    values_bf16: &[u8],
    rows: usize,
    heads: usize,
    qk_dim: usize,
    v_dim: usize,
    scale: f32,
) -> Result<()> {
    if rows == 0 || heads == 0 || qk_dim == 0 || v_dim == 0 {
        anyhow::bail!(
            "real full BF16 causal attention requires non-zero shape, got rows={rows} heads={heads} qk_dim={qk_dim} v_dim={v_dim}"
        );
    }
    if !scale.is_finite() {
        anyhow::bail!("real full BF16 causal attention scale must be finite");
    }
    let qk_bytes = rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(qk_dim))
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("real full BF16 causal attention q/k shape overflows usize while validating coordinator kernel input")?;
    if queries_bf16.len() != qk_bytes {
        anyhow::bail!(
            "real full BF16 causal attention query byte length mismatch: expected {} got {}",
            qk_bytes,
            queries_bf16.len()
        );
    }
    if keys_bf16.len() != qk_bytes {
        anyhow::bail!(
            "real full BF16 causal attention key byte length mismatch: expected {} got {}",
            qk_bytes,
            keys_bf16.len()
        );
    }
    let value_bytes = rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(v_dim))
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("real full BF16 causal attention value shape overflows usize while validating coordinator kernel input")?;
    if values_bf16.len() != value_bytes {
        anyhow::bail!(
            "real full BF16 causal attention value byte length mismatch: expected {} got {}",
            value_bytes,
            values_bf16.len()
        );
    }
    Ok(())
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn validate_rope_inputs(
    input: &[f32],
    positions: &[usize],
    rows: usize,
    heads: usize,
    rotary_dim: usize,
    theta: f32,
) -> Result<Vec<u32>> {
    if rows == 0 || heads == 0 || rotary_dim == 0 || rotary_dim % 2 != 0 {
        anyhow::bail!(
            "real full RoPE requires non-zero rows/heads and positive even rotary_dim, got rows={rows} heads={heads} rotary_dim={rotary_dim}"
        );
    }
    if !theta.is_finite() || theta <= 0.0 {
        anyhow::bail!("real full RoPE theta must be finite and positive");
    }
    if positions.len() != rows {
        anyhow::bail!(
            "real full RoPE positions length mismatch: expected {} got {}",
            rows,
            positions.len()
        );
    }
    let expected_values = rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(rotary_dim))
        .context(
            "real full RoPE shape overflows usize while validating coordinator kernel input",
        )?;
    if input.len() != expected_values {
        anyhow::bail!(
            "real full RoPE input length mismatch: expected {} got {}",
            expected_values,
            input.len()
        );
    }
    positions
        .iter()
        .map(|position| {
            u32::try_from(*position).with_context(|| {
                format!("real full RoPE position {position} does not fit CUDA u32 index")
            })
        })
        .collect()
}

pub(in crate::commands::real_full) fn validate_rope_bf16_inputs(
    input_bf16: &[u8],
    positions: &[usize],
    rows: usize,
    heads: usize,
    rotary_dim: usize,
    theta: f32,
) -> Result<Vec<u32>> {
    if rows == 0 || heads == 0 || rotary_dim == 0 || rotary_dim % 2 != 0 {
        anyhow::bail!(
            "real full BF16 RoPE requires non-zero rows/heads and positive even rotary_dim, got rows={rows} heads={heads} rotary_dim={rotary_dim}"
        );
    }
    if !theta.is_finite() || theta <= 0.0 {
        anyhow::bail!("real full BF16 RoPE theta must be finite and positive");
    }
    if positions.len() != rows {
        anyhow::bail!(
            "real full BF16 RoPE positions length mismatch: expected {} got {}",
            rows,
            positions.len()
        );
    }
    let expected_bytes = rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(rotary_dim))
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full BF16 RoPE shape overflows usize while validating coordinator kernel input",
        )?;
    if input_bf16.len() != expected_bytes {
        anyhow::bail!(
            "real full BF16 RoPE input byte length mismatch: expected {} got {}",
            expected_bytes,
            input_bf16.len()
        );
    }
    positions
        .iter()
        .map(|position| {
            u32::try_from(*position).with_context(|| {
                format!("real full BF16 RoPE position {position} does not fit CUDA u32 index")
            })
        })
        .collect()
}

#[allow(dead_code, clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn validate_mla_rope_attention_inputs(
    q_nope: &[f32],
    q_rope: &[f32],
    k_nope: &[f32],
    k_rope: &[f32],
    values: &[f32],
    rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    scale: f32,
) -> Result<()> {
    if rows == 0 || heads == 0 || nope_dim == 0 || rope_dim == 0 || v_dim == 0 {
        anyhow::bail!(
            "real full MLA/RoPE attention requires non-zero shape, got rows={rows} heads={heads} nope_dim={nope_dim} rope_dim={rope_dim} v_dim={v_dim}"
        );
    }
    if rope_dim % 2 != 0 {
        anyhow::bail!("real full MLA/RoPE attention rope_dim must be even, got {rope_dim}");
    }
    if !scale.is_finite() {
        anyhow::bail!("real full MLA/RoPE attention scale must be finite");
    }
    let row_heads = rows
        .checked_mul(heads)
        .context("real full MLA/RoPE attention row-head shape overflows usize")?;
    let nope_values = row_heads
        .checked_mul(nope_dim)
        .context("real full MLA/RoPE attention no-RPE shape overflows usize")?;
    let rope_values = row_heads
        .checked_mul(rope_dim)
        .context("real full MLA/RoPE attention q-RoPE shape overflows usize")?;
    let shared_rope_values = rows
        .checked_mul(rope_dim)
        .context("real full MLA/RoPE attention shared k-RoPE shape overflows usize")?;
    let value_count = row_heads
        .checked_mul(v_dim)
        .context("real full MLA/RoPE attention value shape overflows usize")?;
    if q_nope.len() != nope_values {
        anyhow::bail!(
            "real full MLA/RoPE attention q_nope length mismatch: expected {} got {}",
            nope_values,
            q_nope.len()
        );
    }
    if k_nope.len() != nope_values {
        anyhow::bail!(
            "real full MLA/RoPE attention k_nope length mismatch: expected {} got {}",
            nope_values,
            k_nope.len()
        );
    }
    if q_rope.len() != rope_values {
        anyhow::bail!(
            "real full MLA/RoPE attention q_rope length mismatch: expected {} got {}",
            rope_values,
            q_rope.len()
        );
    }
    if k_rope.len() != shared_rope_values {
        anyhow::bail!(
            "real full MLA/RoPE attention k_rope length mismatch: expected {} got {}",
            shared_rope_values,
            k_rope.len()
        );
    }
    if values.len() != value_count {
        anyhow::bail!(
            "real full MLA/RoPE attention value length mismatch: expected {} got {}",
            value_count,
            values.len()
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn validate_mla_rope_attention_bf16_inputs(
    q_nope_bf16: &[u8],
    q_rope_bf16: &[u8],
    k_nope_bf16: &[u8],
    k_rope_bf16: &[u8],
    values_bf16: &[u8],
    rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    scale: f32,
) -> Result<()> {
    if rows == 0 || heads == 0 || nope_dim == 0 || rope_dim == 0 || v_dim == 0 {
        anyhow::bail!(
            "real full BF16 MLA/RoPE attention requires non-zero shape, got rows={rows} heads={heads} nope_dim={nope_dim} rope_dim={rope_dim} v_dim={v_dim}"
        );
    }
    if rope_dim % 2 != 0 {
        anyhow::bail!("real full BF16 MLA/RoPE attention rope_dim must be even, got {rope_dim}");
    }
    if !scale.is_finite() {
        anyhow::bail!("real full BF16 MLA/RoPE attention scale must be finite");
    }
    let row_heads = rows.checked_mul(heads).context(
        "real full BF16 MLA/RoPE attention row/head shape overflows usize while validating input",
    )?;
    let nope_bytes = row_heads
        .checked_mul(nope_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full BF16 MLA/RoPE attention no-RPE shape overflows usize while validating input",
        )?;
    if q_nope_bf16.len() != nope_bytes {
        anyhow::bail!(
            "real full BF16 MLA/RoPE attention q_nope byte length mismatch: expected {} got {}",
            nope_bytes,
            q_nope_bf16.len()
        );
    }
    if k_nope_bf16.len() != nope_bytes {
        anyhow::bail!(
            "real full BF16 MLA/RoPE attention k_nope byte length mismatch: expected {} got {}",
            nope_bytes,
            k_nope_bf16.len()
        );
    }
    let q_rope_bytes = row_heads
        .checked_mul(rope_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full BF16 MLA/RoPE attention q_rope shape overflows usize while validating input",
        )?;
    if q_rope_bf16.len() != q_rope_bytes {
        anyhow::bail!(
            "real full BF16 MLA/RoPE attention q_rope byte length mismatch: expected {} got {}",
            q_rope_bytes,
            q_rope_bf16.len()
        );
    }
    let k_rope_bytes = rows
        .checked_mul(rope_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full BF16 MLA/RoPE attention shared k_rope shape overflows usize while validating input",
        )?;
    if k_rope_bf16.len() != k_rope_bytes {
        anyhow::bail!(
            "real full BF16 MLA/RoPE attention k_rope byte length mismatch: expected {} got {}",
            k_rope_bytes,
            k_rope_bf16.len()
        );
    }
    let value_bytes = row_heads
        .checked_mul(v_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full BF16 MLA/RoPE attention value shape overflows usize while validating input",
        )?;
    if values_bf16.len() != value_bytes {
        anyhow::bail!(
            "real full BF16 MLA/RoPE attention value byte length mismatch: expected {} got {}",
            value_bytes,
            values_bf16.len()
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cpu_mla_rope_attention_rows(
    q_nope: &[f32],
    q_rope: &[f32],
    k_nope: &[f32],
    k_rope: &[f32],
    values: &[f32],
    rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    scale: f32,
) -> MlaRopeAttentionOutput {
    let mut output = vec![0.0_f32; rows * heads * v_dim];
    for row in 0..rows {
        for head in 0..heads {
            let q_nope_start = (row * heads + head) * nope_dim;
            let q_rope_start = (row * heads + head) * rope_dim;
            let q_nope_vec = &q_nope[q_nope_start..q_nope_start + nope_dim];
            let q_rope_vec = &q_rope[q_rope_start..q_rope_start + rope_dim];
            let mut scores = Vec::with_capacity(row + 1);
            for key_row in 0..=row {
                let k_nope_start = (key_row * heads + head) * nope_dim;
                let k_rope_start = key_row * rope_dim;
                let nope_score = q_nope_vec
                    .iter()
                    .zip(k_nope[k_nope_start..k_nope_start + nope_dim].iter())
                    .map(|(left, right)| left * right)
                    .sum::<f32>();
                let rope_score = q_rope_vec
                    .iter()
                    .zip(k_rope[k_rope_start..k_rope_start + rope_dim].iter())
                    .map(|(left, right)| left * right)
                    .sum::<f32>();
                scores.push((nope_score + rope_score) * scale);
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut weights = scores
                .iter()
                .map(|score| (score - max_score).exp())
                .collect::<Vec<_>>();
            let weight_sum = weights.iter().sum::<f32>().max(1.0e-12);
            for weight in &mut weights {
                *weight /= weight_sum;
            }
            let out_start = (row * heads + head) * v_dim;
            for (weight, key_row) in weights.iter().zip(0..=row) {
                let value_start = (key_row * heads + head) * v_dim;
                for value_index in 0..v_dim {
                    output[out_start + value_index] += weight * values[value_start + value_index];
                }
            }
        }
    }
    MlaRopeAttentionOutput {
        values: output,
        backend: CPU_REFERENCE_MLA_ROPE_ATTENTION_BACKEND,
    }
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cpu_mla_rope_attention_rows_bf16(
    q_nope_bf16: &[u8],
    q_rope_bf16: &[u8],
    k_nope_bf16: &[u8],
    k_rope_bf16: &[u8],
    values_bf16: &[u8],
    rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    scale: f32,
) -> MlaRopeAttentionOutput {
    let mut output = cpu_mla_rope_attention_rows(
        &bf16_values_to_f32(q_nope_bf16),
        &bf16_values_to_f32(q_rope_bf16),
        &bf16_values_to_f32(k_nope_bf16),
        &bf16_values_to_f32(k_rope_bf16),
        &bf16_values_to_f32(values_bf16),
        rows,
        heads,
        nope_dim,
        rope_dim,
        v_dim,
        scale,
    );
    output.backend = CPU_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND;
    output
}

pub(in crate::commands::real_full) fn cpu_rope_rows(
    input: &[f32],
    positions: &[u32],
    rows: usize,
    heads: usize,
    rotary_dim: usize,
    theta: f32,
) -> RopeOutput {
    let mut output = vec![0.0_f32; input.len()];
    let pair_count = rotary_dim / 2;
    for (row, position) in positions.iter().copied().enumerate().take(rows) {
        for head in 0..heads {
            let row_head_start = (row * heads + head) * rotary_dim;
            for pair in 0..pair_count {
                let offset = row_head_start + pair * 2;
                let angle = position as f32 * theta.powf(-2.0 * pair as f32 / rotary_dim as f32);
                let cos = angle.cos();
                let sin = angle.sin();
                let even = input[offset];
                let odd = input[offset + 1];
                output[offset] = even * cos - odd * sin;
                output[offset + 1] = even * sin + odd * cos;
            }
        }
    }
    RopeOutput {
        values: output,
        backend: CPU_REFERENCE_ROPE_BACKEND,
    }
}

pub(in crate::commands::real_full) fn cpu_rope_rows_bf16(
    input_bf16: &[u8],
    positions: &[u32],
    rows: usize,
    heads: usize,
    rotary_dim: usize,
    theta: f32,
) -> RopeOutput {
    let mut output = cpu_rope_rows(
        &bf16_values_to_f32(input_bf16),
        positions,
        rows,
        heads,
        rotary_dim,
        theta,
    );
    output.backend = CPU_REFERENCE_ROPE_BF16_BACKEND;
    output
}

pub(in crate::commands::real_full) fn cpu_causal_attention_rows(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    rows: usize,
    heads: usize,
    qk_dim: usize,
    v_dim: usize,
    scale: f32,
) -> CausalAttentionOutput {
    let mut output = vec![0.0_f32; rows * heads * v_dim];
    for row in 0..rows {
        for head in 0..heads {
            let q_start = (row * heads + head) * qk_dim;
            let query = &queries[q_start..q_start + qk_dim];
            let mut scores = Vec::with_capacity(row + 1);
            for key_row in 0..=row {
                let k_start = (key_row * heads + head) * qk_dim;
                let key = &keys[k_start..k_start + qk_dim];
                let score = query
                    .iter()
                    .zip(key.iter())
                    .map(|(query, key)| query * key)
                    .sum::<f32>()
                    * scale;
                scores.push(score);
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut weights = scores
                .iter()
                .map(|score| (score - max_score).exp())
                .collect::<Vec<_>>();
            let weight_sum = weights.iter().sum::<f32>().max(1.0e-12);
            for weight in &mut weights {
                *weight /= weight_sum;
            }
            let out_start = (row * heads + head) * v_dim;
            for (weight, key_row) in weights.iter().zip(0..=row) {
                let value_start = (key_row * heads + head) * v_dim;
                let value = &values[value_start..value_start + v_dim];
                for value_index in 0..v_dim {
                    output[out_start + value_index] += weight * value[value_index];
                }
            }
        }
    }
    CausalAttentionOutput {
        values: output,
        backend: CPU_REFERENCE_CAUSAL_ATTENTION_BACKEND,
    }
}

pub(in crate::commands::real_full) fn cpu_causal_attention_rows_bf16(
    queries_bf16: &[u8],
    keys_bf16: &[u8],
    values_bf16: &[u8],
    rows: usize,
    heads: usize,
    qk_dim: usize,
    v_dim: usize,
    scale: f32,
) -> CausalAttentionOutput {
    let mut output = cpu_causal_attention_rows(
        &bf16_values_to_f32(queries_bf16),
        &bf16_values_to_f32(keys_bf16),
        &bf16_values_to_f32(values_bf16),
        rows,
        heads,
        qk_dim,
        v_dim,
        scale,
    );
    output.backend = CPU_REFERENCE_CAUSAL_ATTENTION_BF16_BACKEND;
    output
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_rope_rows(
    input: &[f32],
    positions: &[u32],
    rows: usize,
    heads: usize,
    rotary_dim: usize,
    theta: f32,
) -> Result<RopeOutput> {
    let library = cuda_native_library()?;
    let input_bytes = std::mem::size_of_val(input);
    let position_bytes = std::mem::size_of_val(positions);
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let input_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        input_bytes,
        "RoPE input",
    )?;
    let position_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::B,
        position_bytes,
        "RoPE positions",
    )?;
    let output_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        input_bytes,
        "RoPE output",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            f32_bytes(input),
            "RoPE input",
        )
        .context("copying RoPE input to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::B,
            u32_bytes(positions),
            "RoPE positions",
        )
        .context("copying RoPE positions to device")?;
    library
        .cuda_rope_f32(
            input_buffer,
            position_buffer,
            output_buffer,
            rows,
            heads,
            rotary_dim,
            theta,
        )
        .context("executing CUDA RoPE")?;
    let mut out_bytes = vec![0_u8; input_bytes];
    library
        .copy_d2h(&mut out_bytes, output_buffer)
        .context("copying RoPE output to host")?;

    Ok(RopeOutput {
        values: f32_vec_from_bytes(&out_bytes)?,
        backend: CUDA_REFERENCE_ROPE_BACKEND,
    })
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_rope_rows_bf16(
    input_bf16: &[u8],
    positions: &[u32],
    rows: usize,
    heads: usize,
    rotary_dim: usize,
    theta: f32,
) -> Result<RopeOutput> {
    let library = cuda_native_library()?;
    let input_bytes = input_bf16.len();
    let position_bytes = std::mem::size_of_val(positions);
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let input_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        input_bytes,
        "BF16 RoPE input",
    )?;
    let position_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::B,
        position_bytes,
        "BF16 RoPE positions",
    )?;
    let output_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        input_bytes,
        "BF16 RoPE output",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            input_bf16,
            "BF16 RoPE input",
        )
        .context("copying BF16 RoPE input to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::B,
            u32_bytes(positions),
            "BF16 RoPE positions",
        )
        .context("copying BF16 RoPE positions to device")?;
    library
        .cuda_rope_bf16(
            input_buffer,
            position_buffer,
            output_buffer,
            rows,
            heads,
            rotary_dim,
            theta,
        )
        .context("executing CUDA BF16 RoPE")?;
    let mut out_bytes = vec![0_u8; input_bytes];
    library
        .copy_d2h(&mut out_bytes, output_buffer)
        .context("copying BF16 RoPE output to host")?;

    Ok(RopeOutput {
        values: bf16_values_to_f32(&out_bytes),
        backend: CUDA_REFERENCE_ROPE_BF16_BACKEND,
    })
}

pub(in crate::commands::real_full) fn rope_graph_input_bytes(
    graph_key: &CoordinatorGraphKey,
    heads: usize,
    rotary_dim: usize,
    context: &str,
) -> Result<usize> {
    graph_key
        .row_bucket
        .row_capacity
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(rotary_dim))
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .with_context(|| format!("{context} input graph buffer bytes overflow usize"))
}

pub(in crate::commands::real_full) fn rope_graph_position_bytes(
    graph_key: &CoordinatorGraphKey,
    context: &str,
) -> Result<usize> {
    graph_key
        .row_bucket
        .row_capacity
        .checked_mul(std::mem::size_of::<u32>())
        .with_context(|| format!("{context} position graph buffer bytes overflow usize"))
}

pub(in crate::commands::real_full) fn rope_graph_signature(
    graph_key: &CoordinatorGraphKey,
    heads: usize,
    rotary_dim: usize,
    theta: f32,
) -> CoordinatorCudaGraphSignature {
    CoordinatorCudaGraphSignature::rope_bf16(
        graph_key.row_bucket.row_capacity * heads * rotary_dim * std::mem::size_of::<u16>(),
        graph_key.row_bucket.row_capacity,
        heads,
        rotary_dim,
        theta,
    )
}

pub(in crate::commands::real_full) fn cuda_rope_rows_bf16_for_layer(
    layer_id: usize,
    input_bf16: &[u8],
    positions: &[u32],
    rows: usize,
    heads: usize,
    rotary_dim: usize,
    theta: f32,
) -> Result<RopeOutput> {
    let graph_key = coord_attention_graph_key_for_layer_rows(layer_id, rows)?;
    let input_bytes = input_bf16.len();
    let graph_input_bytes = rope_graph_input_bytes(
        &graph_key,
        heads,
        rotary_dim,
        "CUDA BF16 layer RoPE graph-slot",
    )?;
    let graph_position_bytes =
        rope_graph_position_bytes(&graph_key, "CUDA BF16 layer RoPE graph-slot")?;
    let signature = rope_graph_signature(&graph_key, heads, rotary_dim, theta);
    with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let input_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            graph_input_bytes,
            "BF16 layer RoPE input",
        )?;
        let position_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::B,
            graph_position_bytes,
            "BF16 layer RoPE positions",
        )?;
        let output_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            graph_input_bytes,
            "BF16 layer RoPE output",
        )?;

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::A,
                input_bf16,
                "BF16 layer RoPE input",
                cuda_stream,
            )
            .context("async copying BF16 layer RoPE input to device")?;
        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::B,
                u32_bytes(positions),
                "BF16 layer RoPE positions",
                cuda_stream,
            )
            .context("async copying BF16 layer RoPE positions to device")?;
        capture_or_update_layer_rope_bf16_graph(
            library,
            slot,
            signature,
            input_buffer,
            position_buffer,
            output_buffer,
            rows,
            heads,
            rotary_dim,
            theta,
            "BF16 layer RoPE",
        )?;
        let mut out_bytes = vec![0_u8; input_bytes];
        unsafe {
            library
                .copy_d2h_async(&mut out_bytes, output_buffer, cuda_stream)
                .context("async copying BF16 layer RoPE output to host")?;
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing BF16 layer RoPE graph slot stream")?;
        }

        Ok(RopeOutput {
            values: bf16_values_to_f32(&out_bytes),
            backend: CUDA_REFERENCE_ROPE_BF16_BACKEND,
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn rope_bf16_device_buffers_for_layer(
    layer_id: usize,
    input_buffer: GlmrtDeviceBuffer,
    position_buffer: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    heads: usize,
    rotary_dim: usize,
    theta: f32,
) -> Result<&'static str> {
    validate_rope_bf16_device_buffers(
        input_buffer,
        position_buffer,
        output_buffer,
        rows,
        heads,
        rotary_dim,
        theta,
    )?;
    let graph_key = coord_attention_graph_key_for_layer_rows(layer_id, rows)?;
    let signature = rope_graph_signature(&graph_key, heads, rotary_dim, theta);
    with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        capture_or_update_layer_rope_bf16_graph(
            library,
            slot,
            signature,
            input_buffer,
            position_buffer,
            output_buffer,
            rows,
            heads,
            rotary_dim,
            theta,
            "BF16 layer RoPE device-buffer",
        )?;
        unsafe {
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing BF16 layer RoPE device-buffer graph slot stream")?;
        }
        Ok(CUDA_REFERENCE_ROPE_BF16_BACKEND)
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn validate_rope_bf16_device_buffers(
    input_buffer: GlmrtDeviceBuffer,
    position_buffer: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    heads: usize,
    rotary_dim: usize,
    theta: f32,
) -> Result<()> {
    if rows == 0 || heads == 0 || rotary_dim == 0 || rotary_dim % 2 != 0 {
        anyhow::bail!(
            "CUDA BF16 layer RoPE device-buffer requires nonzero rows/heads and positive even rotary_dim, got rows={rows} heads={heads} rotary_dim={rotary_dim}"
        );
    }
    if !theta.is_finite() || theta <= 0.0 {
        anyhow::bail!("CUDA BF16 layer RoPE device-buffer theta must be finite and positive");
    }
    let input_bytes = rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(rotary_dim))
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 layer RoPE device-buffer input bytes overflow usize")?;
    let position_bytes = rows
        .checked_mul(std::mem::size_of::<u32>())
        .context("CUDA BF16 layer RoPE device-buffer position bytes overflow usize")?;
    let buffers = [
        ("input", input_buffer, input_bytes),
        ("positions", position_buffer, position_bytes),
        ("output", output_buffer, input_bytes),
    ];
    for (label, buffer, required_bytes) in buffers {
        if buffer.ptr.is_null() {
            anyhow::bail!("CUDA BF16 layer RoPE device-buffer {label} is null");
        }
        if buffer.bytes < required_bytes {
            anyhow::bail!(
                "CUDA BF16 layer RoPE device-buffer {label} has {} bytes, expected at least {required_bytes}",
                buffer.bytes
            );
        }
        if buffer.device_id != input_buffer.device_id {
            anyhow::bail!(
                "CUDA BF16 layer RoPE device-buffer {label} is on CUDA device {}, expected {}",
                buffer.device_id,
                input_buffer.device_id
            );
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn capture_or_update_layer_rope_bf16_graph(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    signature: CoordinatorCudaGraphSignature,
    input_buffer: GlmrtDeviceBuffer,
    position_buffer: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    heads: usize,
    rotary_dim: usize,
    theta: f32,
    label: &'static str,
) -> Result<()> {
    if !slot.has_captured_graph(CoordinatorCudaGraphProgram::LayerRopeBf16, signature) {
        slot.stream_synchronize()
            .with_context(|| format!("synchronizing {label} inputs before graph capture"))?;
        slot.capture_graph(
            library,
            CoordinatorCudaGraphProgram::LayerRopeBf16,
            signature,
            |library, cuda_stream, _workspace| unsafe {
                library
                    .cuda_rope_bf16_async(
                        input_buffer,
                        position_buffer,
                        output_buffer,
                        rows,
                        heads,
                        rotary_dim,
                        theta,
                        cuda_stream,
                    )
                    .with_context(|| format!("capturing async CUDA {label}"))?;
                Ok(())
            },
        )?;
    } else {
        let (graph_raw, exec_raw) = slot
            .captured_graph_raw_handles(CoordinatorCudaGraphProgram::LayerRopeBf16, signature)
            .context("coordinator CUDA graph slot lost captured RoPE graph before update")?;
        unsafe {
            library
                .cuda_graph_update_rope_bf16_node(
                    graph_raw,
                    exec_raw,
                    0,
                    input_buffer,
                    position_buffer,
                    output_buffer,
                    rows,
                    heads,
                    rotary_dim,
                    theta,
                )
                .with_context(|| format!("updating captured CUDA {label} graph node"))?;
        }
    }
    slot.launch_captured_graph(
        library,
        CoordinatorCudaGraphProgram::LayerRopeBf16,
        signature,
    )
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_mla_rope_attention_rows_bf16(
    q_nope_bf16: &[u8],
    q_rope_bf16: &[u8],
    k_nope_bf16: &[u8],
    k_rope_bf16: &[u8],
    values_bf16: &[u8],
    rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    scale: f32,
) -> Result<MlaRopeAttentionOutput> {
    let library = cuda_native_library()?;
    let nope_bytes = q_nope_bf16.len();
    let q_rope_bytes = q_rope_bf16.len();
    let k_rope_bytes = k_rope_bf16.len();
    let value_bytes = values_bf16.len();
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let q_nope_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        nope_bytes,
        "BF16 MLA/RoPE q_nope",
    )?;
    let q_rope_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::B,
        q_rope_bytes,
        "BF16 MLA/RoPE q_rope",
    )?;
    let k_nope_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        nope_bytes,
        "BF16 MLA/RoPE k_nope",
    )?;
    let k_rope_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        k_rope_bytes,
        "BF16 MLA/RoPE k_rope",
    )?;
    let value_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::E,
        value_bytes,
        "BF16 MLA/RoPE values",
    )?;
    let output_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::F,
        value_bytes,
        "BF16 MLA/RoPE output",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            q_nope_bf16,
            "BF16 MLA/RoPE q_nope",
        )
        .context("copying BF16 MLA/RoPE q_nope to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::B,
            q_rope_bf16,
            "BF16 MLA/RoPE q_rope",
        )
        .context("copying BF16 MLA/RoPE q_rope to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::C,
            k_nope_bf16,
            "BF16 MLA/RoPE k_nope",
        )
        .context("copying BF16 MLA/RoPE k_nope to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::D,
            k_rope_bf16,
            "BF16 MLA/RoPE k_rope",
        )
        .context("copying BF16 MLA/RoPE k_rope to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::E,
            values_bf16,
            "BF16 MLA/RoPE values",
        )
        .context("copying BF16 MLA/RoPE values to device")?;
    library
        .cuda_mla_rope_attention_bf16(
            q_nope_buffer,
            q_rope_buffer,
            k_nope_buffer,
            k_rope_buffer,
            value_buffer,
            output_buffer,
            rows,
            heads,
            nope_dim,
            rope_dim,
            v_dim,
            scale,
        )
        .context("executing CUDA BF16 MLA/RoPE attention")?;
    let mut out_bytes = vec![0_u8; value_bytes];
    library
        .copy_d2h(&mut out_bytes, output_buffer)
        .context("copying BF16 MLA/RoPE attention output to host")?;

    Ok(MlaRopeAttentionOutput {
        values: bf16_values_to_f32(&out_bytes),
        backend: CUDA_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_mla_rope_attention_rows_bf16_for_layer(
    layer_id: usize,
    q_nope_bf16: &[u8],
    q_rope_bf16: &[u8],
    k_nope_bf16: &[u8],
    k_rope_bf16: &[u8],
    values_bf16: &[u8],
    rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    scale: f32,
) -> Result<MlaRopeAttentionOutput> {
    let graph_key = coord_attention_graph_key_for_layer_rows(layer_id, rows)?;
    let value_bytes = values_bf16.len();
    let nope_graph_bytes = mla_rope_attention_graph_nope_bytes(
        &graph_key,
        heads,
        nope_dim,
        "CUDA BF16 layer MLA/RoPE attention graph-slot",
    )?;
    let q_rope_graph_bytes = mla_rope_attention_graph_q_rope_bytes(
        &graph_key,
        heads,
        rope_dim,
        "CUDA BF16 layer MLA/RoPE attention graph-slot",
    )?;
    let k_rope_graph_bytes = mla_rope_attention_graph_k_rope_bytes(
        &graph_key,
        rope_dim,
        "CUDA BF16 layer MLA/RoPE attention graph-slot",
    )?;
    let value_graph_bytes = mla_rope_attention_graph_value_bytes(
        &graph_key,
        heads,
        v_dim,
        "CUDA BF16 layer MLA/RoPE attention graph-slot",
    )?;
    let signature =
        mla_rope_attention_graph_signature(&graph_key, heads, nope_dim, rope_dim, v_dim, scale);
    with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let q_nope_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            nope_graph_bytes,
            "BF16 layer MLA/RoPE q_nope",
        )?;
        let q_rope_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::B,
            q_rope_graph_bytes,
            "BF16 layer MLA/RoPE q_rope",
        )?;
        let k_nope_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            nope_graph_bytes,
            "BF16 layer MLA/RoPE k_nope",
        )?;
        let k_rope_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            k_rope_graph_bytes,
            "BF16 layer MLA/RoPE k_rope",
        )?;
        let value_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::E,
            value_graph_bytes,
            "BF16 layer MLA/RoPE values",
        )?;
        let output_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::F,
            value_graph_bytes,
            "BF16 layer MLA/RoPE output",
        )?;

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::A,
                q_nope_bf16,
                "BF16 layer MLA/RoPE q_nope",
                cuda_stream,
            )
            .context("async copying BF16 layer MLA/RoPE q_nope to device")?;
        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::B,
                q_rope_bf16,
                "BF16 layer MLA/RoPE q_rope",
                cuda_stream,
            )
            .context("async copying BF16 layer MLA/RoPE q_rope to device")?;
        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::C,
                k_nope_bf16,
                "BF16 layer MLA/RoPE k_nope",
                cuda_stream,
            )
            .context("async copying BF16 layer MLA/RoPE k_nope to device")?;
        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::D,
                k_rope_bf16,
                "BF16 layer MLA/RoPE k_rope",
                cuda_stream,
            )
            .context("async copying BF16 layer MLA/RoPE k_rope to device")?;
        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::E,
                values_bf16,
                "BF16 layer MLA/RoPE values",
                cuda_stream,
            )
            .context("async copying BF16 layer MLA/RoPE values to device")?;
        let backend =
            if b12x_mla_rope_attention_bf16_supported(rows, heads, nope_dim, rope_dim, v_dim) {
                capture_or_update_layer_b12x_mla_rope_attention_bf16_graph(
                    library,
                    slot,
                    signature,
                    q_nope_buffer,
                    q_rope_buffer,
                    k_nope_buffer,
                    k_rope_buffer,
                    value_buffer,
                    output_buffer,
                    rows,
                    heads,
                    nope_dim,
                    rope_dim,
                    v_dim,
                    scale,
                    "BF16 layer b12x MLA/RoPE attention",
                )?;
                B12X_MLA_ROPE_ATTENTION_BF16_BACKEND
            } else {
                capture_or_update_layer_mla_rope_attention_bf16_graph(
                    library,
                    slot,
                    signature,
                    q_nope_buffer,
                    q_rope_buffer,
                    k_nope_buffer,
                    k_rope_buffer,
                    value_buffer,
                    output_buffer,
                    rows,
                    heads,
                    nope_dim,
                    rope_dim,
                    v_dim,
                    scale,
                    "BF16 layer MLA/RoPE attention",
                )?;
                CUDA_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND
            };
        let mut out_bytes = vec![0_u8; value_bytes];
        unsafe {
            library
                .copy_d2h_async(&mut out_bytes, output_buffer, cuda_stream)
                .context("async copying BF16 layer MLA/RoPE attention output to host")?;
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing BF16 layer MLA/RoPE attention graph slot stream")?;
        }

        Ok(MlaRopeAttentionOutput {
            values: bf16_values_to_f32(&out_bytes),
            backend,
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn mla_rope_attention_device_buffers_bf16_for_layer(
    layer_id: usize,
    q_nope_buffer: GlmrtDeviceBuffer,
    q_rope_buffer: GlmrtDeviceBuffer,
    k_nope_buffer: GlmrtDeviceBuffer,
    k_rope_buffer: GlmrtDeviceBuffer,
    value_buffer: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    scale: f32,
) -> Result<&'static str> {
    let graph_key = coord_attention_graph_key_for_layer_rows(layer_id, rows)?;
    validate_mla_rope_attention_device_buffers(
        rows,
        q_nope_buffer,
        q_rope_buffer,
        k_nope_buffer,
        k_rope_buffer,
        value_buffer,
        output_buffer,
        heads,
        nope_dim,
        rope_dim,
        v_dim,
    )?;
    let signature =
        mla_rope_attention_graph_signature(&graph_key, heads, nope_dim, rope_dim, v_dim, scale);
    with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let backend =
            if flashinfer_mla_rope_attention_bf16_supported(rows, heads, nope_dim, rope_dim, v_dim)
            {
                let exact_signature = flashinfer_mla_graph_signature(
                    rows,
                    0,
                    rows,
                    graph_key.row_bucket.row_capacity,
                    heads,
                    nope_dim,
                    rope_dim,
                    v_dim,
                    scale,
                )?;
                if capture_or_update_layer_flashinfer_mla_rope_attention_bf16_graph(
                    library,
                    slot,
                    exact_signature,
                    q_nope_buffer,
                    q_rope_buffer,
                    k_nope_buffer,
                    k_rope_buffer,
                    value_buffer,
                    output_buffer,
                    rows,
                    0,
                    rows,
                    graph_key.row_bucket.row_capacity,
                    heads,
                    nope_dim,
                    rope_dim,
                    v_dim,
                    scale,
                    "BF16 layer FlashInfer MLA/RoPE attention device-buffer",
                    false,
                )? {
                    FLASHINFER_MLA_ROPE_ATTENTION_BF16_BACKEND
                } else {
                    capture_or_update_layer_mla_rope_attention_bf16_graph(
                        library,
                        slot,
                        signature,
                        q_nope_buffer,
                        q_rope_buffer,
                        k_nope_buffer,
                        k_rope_buffer,
                        value_buffer,
                        output_buffer,
                        rows,
                        heads,
                        nope_dim,
                        rope_dim,
                        v_dim,
                        scale,
                        "BF16 layer MLA/RoPE attention device-buffer",
                    )?;
                    CUDA_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND
                }
            } else if b12x_mla_rope_attention_bf16_supported(rows, heads, nope_dim, rope_dim, v_dim)
            {
                capture_or_update_layer_b12x_mla_rope_attention_bf16_graph(
                    library,
                    slot,
                    signature,
                    q_nope_buffer,
                    q_rope_buffer,
                    k_nope_buffer,
                    k_rope_buffer,
                    value_buffer,
                    output_buffer,
                    rows,
                    heads,
                    nope_dim,
                    rope_dim,
                    v_dim,
                    scale,
                    "BF16 layer b12x MLA/RoPE attention device-buffer",
                )?;
                B12X_MLA_ROPE_ATTENTION_BF16_BACKEND
            } else {
                capture_or_update_layer_mla_rope_attention_bf16_graph(
                    library,
                    slot,
                    signature,
                    q_nope_buffer,
                    q_rope_buffer,
                    k_nope_buffer,
                    k_rope_buffer,
                    value_buffer,
                    output_buffer,
                    rows,
                    heads,
                    nope_dim,
                    rope_dim,
                    v_dim,
                    scale,
                    "BF16 layer MLA/RoPE attention device-buffer",
                )?;
                CUDA_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND
            };
        unsafe {
            library.cuda_stream_synchronize(cuda_stream).context(
                "synchronizing BF16 layer MLA/RoPE attention device-buffer graph slot stream",
            )?;
        }
        Ok(backend)
    })
}

#[allow(clippy::too_many_arguments)]
fn capture_or_update_layer_mla_rope_attention_bf16_suffix_graph(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    signature: CoordinatorCudaGraphSignature,
    q_nope_buffer: GlmrtDeviceBuffer,
    q_rope_buffer: GlmrtDeviceBuffer,
    k_nope_buffer: GlmrtDeviceBuffer,
    k_rope_buffer: GlmrtDeviceBuffer,
    value_buffer: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    query_row_offset: usize,
    query_rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    scale: f32,
    label: &'static str,
) -> Result<()> {
    let program = CoordinatorCudaGraphProgram::LayerMlaRopeAttentionBf16Suffix;
    if !slot.has_captured_graph(program, signature) {
        slot.stream_synchronize()
            .with_context(|| format!("synchronizing {label} inputs before graph capture"))?;
        slot.capture_graph(
            library,
            program,
            signature,
            |library, cuda_stream, _workspace| unsafe {
                library
                    .cuda_mla_rope_attention_bf16_suffix_async(
                        q_nope_buffer,
                        q_rope_buffer,
                        k_nope_buffer,
                        k_rope_buffer,
                        value_buffer,
                        output_buffer,
                        rows,
                        query_row_offset,
                        query_rows,
                        heads,
                        nope_dim,
                        rope_dim,
                        v_dim,
                        scale,
                        cuda_stream,
                    )
                    .with_context(|| format!("capturing async CUDA {label}"))?;
                Ok(())
            },
        )?;
    } else {
        let (graph_raw, exec_raw) = slot
            .captured_graph_raw_handles(program, signature)
            .context(
                "coordinator CUDA graph slot lost captured MLA/RoPE suffix attention graph before update",
            )?;
        unsafe {
            library
                .cuda_graph_update_mla_rope_attention_bf16_suffix_node(
                    graph_raw,
                    exec_raw,
                    0,
                    q_nope_buffer,
                    q_rope_buffer,
                    k_nope_buffer,
                    k_rope_buffer,
                    value_buffer,
                    output_buffer,
                    rows,
                    query_row_offset,
                    query_rows,
                    heads,
                    nope_dim,
                    rope_dim,
                    v_dim,
                    scale,
                )
                .with_context(|| format!("updating captured CUDA {label} graph node"))?;
        }
    }
    slot.launch_captured_graph(library, program, signature)
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn mla_rope_attention_suffix_device_buffers_bf16_for_layer(
    layer_id: usize,
    q_nope_buffer: GlmrtDeviceBuffer,
    q_rope_buffer: GlmrtDeviceBuffer,
    k_nope_buffer: GlmrtDeviceBuffer,
    k_rope_buffer: GlmrtDeviceBuffer,
    value_buffer: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    query_row_offset: usize,
    query_rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    scale: f32,
) -> Result<&'static str> {
    if query_rows == 0 {
        anyhow::bail!("CUDA BF16 layer MLA/RoPE suffix attention requires query rows");
    }
    if query_row_offset > rows || query_rows > rows - query_row_offset {
        anyhow::bail!(
            "CUDA BF16 layer MLA/RoPE suffix attention query rows {}..{} exceed rows {rows}",
            query_row_offset,
            query_row_offset.saturating_add(query_rows)
        );
    }
    let graph_key = coord_attention_graph_key_for_layer_rows(layer_id, rows)?;
    validate_mla_rope_attention_device_buffers_with_output_rows(
        rows,
        query_rows,
        q_nope_buffer,
        q_rope_buffer,
        k_nope_buffer,
        k_rope_buffer,
        value_buffer,
        output_buffer,
        heads,
        nope_dim,
        rope_dim,
        v_dim,
    )?;
    let signature = mla_rope_attention_suffix_graph_signature(
        &graph_key, query_rows, heads, nope_dim, rope_dim, v_dim, scale,
    );
    with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let backend = if flashinfer_cudnn_mla_rope_attention_bf16_supported(
            rows, heads, nope_dim, rope_dim, v_dim,
        ) {
            let exact_signature = flashinfer_mla_graph_signature(
                rows,
                query_row_offset,
                query_rows,
                graph_key.row_bucket.row_capacity,
                heads,
                nope_dim,
                rope_dim,
                v_dim,
                scale,
            )?;
            let captured = capture_or_update_layer_flashinfer_mla_rope_attention_bf16_graph(
                library,
                slot,
                exact_signature,
                q_nope_buffer,
                q_rope_buffer,
                k_nope_buffer,
                k_rope_buffer,
                value_buffer,
                output_buffer,
                rows,
                query_row_offset,
                query_rows,
                graph_key.row_bucket.row_capacity,
                heads,
                nope_dim,
                rope_dim,
                v_dim,
                scale,
                "BF16 layer FlashInfer/cuDNN MLA/RoPE suffix attention device-buffer",
                false,
            )?;
            anyhow::ensure!(
                    captured,
                    "FlashInfer/cuDNN MLA suffix graph bucket={} query_bucket={} was not captured during startup",
                    graph_key.row_bucket.row_capacity,
                    flashinfer_mla_capture_shape(
                        rows,
                        query_row_offset,
                        query_rows,
                        graph_key.row_bucket.row_capacity,
                    )?
                    .query_rows,
                );
            FLASHINFER_CUDNN_MLA_ROPE_ATTENTION_BF16_BACKEND
        } else {
            capture_or_update_layer_mla_rope_attention_bf16_suffix_graph(
                library,
                slot,
                signature,
                q_nope_buffer,
                q_rope_buffer,
                k_nope_buffer,
                k_rope_buffer,
                value_buffer,
                output_buffer,
                rows,
                query_row_offset,
                query_rows,
                heads,
                nope_dim,
                rope_dim,
                v_dim,
                scale,
                "BF16 layer MLA/RoPE suffix attention device-buffer",
            )?;
            CUDA_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND
        };
        unsafe {
            library.cuda_stream_synchronize(cuda_stream).context(
                "synchronizing BF16 layer MLA/RoPE suffix attention device-buffer graph slot stream",
            )?;
        }
        Ok(backend)
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn mla_kv_cache_unpack_bf16_device_buffers_for_layer(
    layer_id: usize,
    payload_buffer: GlmrtDeviceBuffer,
    kv_latent_buffer: GlmrtDeviceBuffer,
    k_rope_buffer: GlmrtDeviceBuffer,
    dsa_key_buffer: Option<GlmrtDeviceBuffer>,
    rows: usize,
    kv_lora_rank: usize,
    rope_dim: usize,
    dsa_dim: usize,
    payload_stride_bytes: usize,
) -> Result<&'static str> {
    validate_mla_kv_cache_unpack_bf16_device_buffers(
        payload_buffer,
        kv_latent_buffer,
        k_rope_buffer,
        dsa_key_buffer,
        rows,
        kv_lora_rank,
        rope_dim,
        dsa_dim,
        payload_stride_bytes,
    )?;
    let graph_key = coord_attention_graph_key_for_layer_rows(layer_id, rows)?;
    let signature = CoordinatorCudaGraphSignature::mla_kv_cache_unpack_bf16(
        graph_key.row_bucket.row_capacity,
        payload_stride_bytes,
        kv_lora_rank,
        rope_dim,
        dsa_dim,
    );
    match with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        capture_or_update_layer_mla_kv_cache_unpack_bf16_graph(
            library,
            slot,
            signature,
            payload_buffer,
            kv_latent_buffer,
            k_rope_buffer,
            dsa_key_buffer,
            rows,
            kv_lora_rank,
            rope_dim,
            dsa_dim,
            payload_stride_bytes,
            "BF16 layer MLA KV cache unpack device-buffer",
        )?;
        unsafe {
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing BF16 layer MLA KV cache unpack graph slot stream")?;
        }
        Ok(CUDA_REFERENCE_MLA_KV_CACHE_UNPACK_BF16_BACKEND)
    }) {
        Ok(backend) => Ok(backend),
        Err(_error) => mla_kv_cache_unpack_bf16_device_buffers_direct(
            payload_buffer,
            kv_latent_buffer,
            k_rope_buffer,
            dsa_key_buffer,
            rows,
            kv_lora_rank,
            rope_dim,
            dsa_dim,
            payload_stride_bytes,
            "BF16 layer MLA KV cache unpack device-buffer",
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn mla_kv_cache_unpack_bf16_device_buffers_direct(
    payload_buffer: GlmrtDeviceBuffer,
    kv_latent_buffer: GlmrtDeviceBuffer,
    k_rope_buffer: GlmrtDeviceBuffer,
    dsa_key_buffer: Option<GlmrtDeviceBuffer>,
    rows: usize,
    kv_lora_rank: usize,
    rope_dim: usize,
    dsa_dim: usize,
    payload_stride_bytes: usize,
    label: &str,
) -> Result<&'static str> {
    let library = cuda_native_library()?;
    library
        .cuda_mla_kv_cache_unpack_bf16(
            payload_buffer,
            kv_latent_buffer,
            k_rope_buffer,
            dsa_key_buffer,
            rows,
            kv_lora_rank,
            rope_dim,
            dsa_dim,
            payload_stride_bytes,
        )
        .with_context(|| format!("executing CUDA {label} direct fallback"))?;
    Ok(CUDA_REFERENCE_MLA_KV_CACHE_UNPACK_BF16_BACKEND)
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn mla_kv_prepare_bf16_device_buffers_for_layer(
    layer_id: usize,
    projected_buffer: GlmrtDeviceBuffer,
    positions_buffer: GlmrtDeviceBuffer,
    norm_weight_buffer: GlmrtDeviceBuffer,
    prepared_buffer: GlmrtDeviceBuffer,
    rows: usize,
    projected_stride_bytes: usize,
    prepared_stride_bytes: usize,
    eps: f32,
    theta: f32,
) -> Result<&'static str> {
    let graph_key = coord_attention_graph_key_for_layer_rows(layer_id, rows)?;
    with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        unsafe {
            library
                .cuda_mla_kv_prepare_bf16_async(
                    projected_buffer,
                    positions_buffer,
                    norm_weight_buffer,
                    prepared_buffer,
                    rows,
                    projected_stride_bytes,
                    prepared_stride_bytes,
                    eps,
                    theta,
                    cuda_stream,
                )
                .context("launching async MLA KV cache preparation")?;
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing MLA KV cache preparation stream")?;
        }
        Ok(CUDA_REFERENCE_MLA_KV_PREPARE_BF16_BACKEND)
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn mla_kv_projected_split_bf16_device_buffers_for_layer(
    layer_id: usize,
    projected_buffer: GlmrtDeviceBuffer,
    k_nope_buffer: GlmrtDeviceBuffer,
    value_buffer: GlmrtDeviceBuffer,
    rows: usize,
    heads: usize,
    nope_dim: usize,
    v_dim: usize,
) -> Result<&'static str> {
    validate_mla_kv_projected_split_bf16_device_buffers(
        projected_buffer,
        k_nope_buffer,
        value_buffer,
        rows,
        heads,
        nope_dim,
        v_dim,
    )?;
    let graph_key = coord_attention_graph_key_for_layer_rows(layer_id, rows)?;
    let signature = CoordinatorCudaGraphSignature::mla_kv_projected_split_bf16(
        graph_key.row_bucket.row_capacity,
        heads,
        nope_dim,
        v_dim,
    );
    with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        capture_or_update_layer_mla_kv_projected_split_bf16_graph(
            library,
            slot,
            signature,
            projected_buffer,
            k_nope_buffer,
            value_buffer,
            rows,
            heads,
            nope_dim,
            v_dim,
            "BF16 layer MLA KV projected split device-buffer",
        )?;
        unsafe {
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing BF16 layer MLA KV projected split graph slot stream")?;
        }
        Ok(CUDA_REFERENCE_MLA_KV_PROJECTED_SPLIT_BF16_BACKEND)
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn mla_query_split_rope_bf16_device_buffers_for_layer(
    layer_id: usize,
    projected_buffer: GlmrtDeviceBuffer,
    q_nope_buffer: GlmrtDeviceBuffer,
    q_rope_unrotated_buffer: GlmrtDeviceBuffer,
    position: u32,
    q_rope_rotated_buffer: GlmrtDeviceBuffer,
    rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    theta: f32,
) -> Result<&'static str> {
    anyhow::ensure!(
        rows == 1,
        "fused MLA query split/RoPE requires exactly one row, got {rows}"
    );
    validate_mla_kv_projected_split_bf16_device_buffers(
        projected_buffer,
        q_nope_buffer,
        q_rope_unrotated_buffer,
        rows,
        heads,
        nope_dim,
        rope_dim,
    )?;
    let graph_key = coord_attention_graph_key_for_layer_rows(layer_id, rows)?;
    let signature = CoordinatorCudaGraphSignature::mla_query_split_rope_bf16(
        rows, heads, nope_dim, rope_dim, theta,
    );
    with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
        let positions_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            std::mem::size_of::<u32>(),
            "fused MLA decode query position",
        )?;
        validate_rope_bf16_device_buffers(
            q_rope_unrotated_buffer,
            positions_buffer,
            q_rope_rotated_buffer,
            rows,
            heads,
            rope_dim,
            theta,
        )?;
        let stream = slot.stream_ptr();
        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::C,
                &position.to_ne_bytes(),
                "fused MLA decode query position",
                stream,
            )
            .context("staging fused MLA decode query position")?;
        let capture_identity = mla_graph_capture_identity(&[
            projected_buffer.ptr as usize,
            q_nope_buffer.ptr as usize,
            q_rope_unrotated_buffer.ptr as usize,
            positions_buffer.ptr as usize,
            q_rope_rotated_buffer.ptr as usize,
            rows,
            heads,
            nope_dim,
            rope_dim,
            theta.to_bits() as usize,
        ]);
        let program = CoordinatorCudaGraphProgram::LayerMlaQuerySplitRopeBf16;
        slot.capture_or_update_graph_exec(
            library,
            program,
            signature,
            capture_identity,
            |library, stream, _workspace| unsafe {
                library
                    .cuda_mla_kv_projected_split_bf16_async(
                        projected_buffer,
                        q_nope_buffer,
                        q_rope_unrotated_buffer,
                        rows,
                        heads,
                        nope_dim,
                        rope_dim,
                        stream,
                    )
                    .context("capturing fused MLA decode query split")?;
                library
                    .cuda_rope_bf16_async(
                        q_rope_unrotated_buffer,
                        positions_buffer,
                        q_rope_rotated_buffer,
                        rows,
                        heads,
                        rope_dim,
                        theta,
                        stream,
                    )
                    .context("capturing fused MLA decode query RoPE")?;
                Ok(())
            },
        )?;
        slot.launch_captured_graph_identity(library, program, signature, capture_identity)?;
        // Packed/compressed attention consumes these buffers on the same stream
        // and performs the single synchronization for the complete decode chain.
        Ok("cuda-mla-query-split-rope-bf16-graph")
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn mla_query_split_rope_bf16_device_positions_for_layer(
    layer_id: usize,
    projected_buffer: GlmrtDeviceBuffer,
    q_nope_buffer: GlmrtDeviceBuffer,
    q_rope_unrotated_buffer: GlmrtDeviceBuffer,
    positions_buffer: GlmrtDeviceBuffer,
    q_rope_rotated_buffer: GlmrtDeviceBuffer,
    rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    theta: f32,
) -> Result<&'static str> {
    anyhow::ensure!(
        rows > 1,
        "batched fused MLA query split/RoPE requires at least two rows, got {rows}"
    );
    validate_mla_kv_projected_split_bf16_device_buffers(
        projected_buffer,
        q_nope_buffer,
        q_rope_unrotated_buffer,
        rows,
        heads,
        nope_dim,
        rope_dim,
    )?;
    validate_rope_bf16_device_buffers(
        q_rope_unrotated_buffer,
        positions_buffer,
        q_rope_rotated_buffer,
        rows,
        heads,
        rope_dim,
        theta,
    )?;
    let graph_key = coord_attention_graph_key_for_layer_rows(layer_id, rows)?;
    let signature = CoordinatorCudaGraphSignature::mla_query_split_rope_bf16(
        rows, heads, nope_dim, rope_dim, theta,
    );
    with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
        let program = CoordinatorCudaGraphProgram::LayerMlaQuerySplitRopeBf16;
        if !slot.has_captured_graph(program, signature) {
            slot.stream_synchronize()
                .context("synchronizing batched MLA query split/RoPE inputs before capture")?;
            slot.capture_graph(
                library,
                program,
                signature,
                |library, stream, _workspace| unsafe {
                    library
                        .cuda_mla_kv_projected_split_bf16_async(
                            projected_buffer,
                            q_nope_buffer,
                            q_rope_unrotated_buffer,
                            rows,
                            heads,
                            nope_dim,
                            rope_dim,
                            stream,
                        )
                        .context("capturing batched MLA query split")?;
                    library
                        .cuda_rope_bf16_async(
                            q_rope_unrotated_buffer,
                            positions_buffer,
                            q_rope_rotated_buffer,
                            rows,
                            heads,
                            rope_dim,
                            theta,
                            stream,
                        )
                        .context("capturing batched MLA query RoPE")?;
                    Ok(())
                },
            )?;
        } else {
            let (graph_raw, exec_raw) = slot
                .captured_graph_raw_handles(program, signature)
                .context("batched MLA query split/RoPE graph disappeared before update")?;
            unsafe {
                library
                    .cuda_graph_update_mla_kv_projected_split_bf16_node(
                        graph_raw,
                        exec_raw,
                        0,
                        projected_buffer,
                        q_nope_buffer,
                        q_rope_unrotated_buffer,
                        rows,
                        heads,
                        nope_dim,
                        rope_dim,
                    )
                    .context("updating batched MLA query split graph node")?;
                library
                    .cuda_graph_update_rope_bf16_node(
                        graph_raw,
                        exec_raw,
                        1,
                        q_rope_unrotated_buffer,
                        positions_buffer,
                        q_rope_rotated_buffer,
                        rows,
                        heads,
                        rope_dim,
                        theta,
                    )
                    .context("updating batched MLA query RoPE graph node")?;
            }
        }
        slot.launch_captured_graph(library, program, signature)?;
        let stream = slot.stream_ptr();
        unsafe {
            library
                .cuda_stream_synchronize(stream)
                .context("synchronizing batched MLA query split/RoPE graph stream")?;
        }
        Ok("cuda-mla-query-split-rope-bf16-batched-graph")
    })
}

pub(in crate::commands::real_full) fn b12x_mla_rope_attention_bf16_shape_supported(
    rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
) -> bool {
    rows > 0
        && rows <= 512
        && heads == 8
        && nope_dim == GLM52_MLA_KV_LORA_RANK
        && rope_dim == GLM52_MLA_QK_ROPE_HEAD_DIM
        && v_dim == GLM52_MLA_KV_LORA_RANK
}

fn flashinfer_mla_rope_attention_bf16_supported(
    rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
) -> bool {
    coordinator_python_capture_enabled()
        && rows > 0
        && rows <= 2_048
        && flashinfer_glm52_attention_heads_supported(heads)
        && nope_dim == 192
        && rope_dim == GLM52_MLA_QK_ROPE_HEAD_DIM
        && v_dim == 256
}

fn flashinfer_cudnn_mla_rope_attention_bf16_supported(
    rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
) -> bool {
    coordinator_python_capture_enabled()
        && rows > 1
        && flashinfer_glm52_attention_heads_supported(heads)
        && nope_dim == 192
        && rope_dim == GLM52_MLA_QK_ROPE_HEAD_DIM
        && v_dim == 256
}

pub(in crate::commands::real_full) fn prewarm_flashinfer_cudnn_mla_suffix_graphs_for_worker(
) -> Result<()> {
    if !coordinator_python_capture_enabled()
        || FLASHINFER_CUDNN_MLA_SUFFIX_PREWARMED.with(Cell::get)
    {
        return Ok(());
    }
    anyhow::ensure!(
        coordinator_python_capture_startup_open(),
        "FlashInfer/cuDNN MLA suffix graphs were not prewarmed before Python capture closed"
    );

    let library = cuda_native_library()?;
    FLASHINFER_CUDNN_MLA_SUFFIX_WORKSPACE.with(|workspace| {
        workspace
            .try_borrow_mut()
            .map_err(|_| anyhow::anyhow!("cuDNN MLA suffix workspace is already borrowed"))?
            .initialize_for_capture(library)
    })?;

    const HEADS: usize = 64;
    const NOPE_DIM: usize = 192;
    const ROPE_DIM: usize = 64;
    const V_DIM: usize = 256;
    const SCALE: f32 = 0.0625;
    let query_capacity = flashinfer_cudnn_mla_suffix_query_capacity();
    let prewarm_start = std::time::Instant::now();
    for row_capacity in COORDINATOR_GRAPH_PREFILL_BUCKET_ROWS {
        let query_rows = (row_capacity / 2).max(1).min(query_capacity);
        let query_row_offset = row_capacity - query_rows;
        let graph_key = coord_attention_graph_key_for_layer_rows(0, row_capacity)?;
        with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
            let signature = flashinfer_mla_graph_signature(
                row_capacity,
                query_row_offset,
                query_rows,
                row_capacity,
                HEADS,
                NOPE_DIM,
                ROPE_DIM,
                V_DIM,
                SCALE,
            )?;
            let captured = capture_or_update_layer_flashinfer_mla_rope_attention_bf16_graph(
                library,
                slot,
                signature,
                GlmrtDeviceBuffer::default(),
                GlmrtDeviceBuffer::default(),
                GlmrtDeviceBuffer::default(),
                GlmrtDeviceBuffer::default(),
                GlmrtDeviceBuffer::default(),
                GlmrtDeviceBuffer::default(),
                row_capacity,
                query_row_offset,
                query_rows,
                row_capacity,
                HEADS,
                NOPE_DIM,
                ROPE_DIM,
                V_DIM,
                SCALE,
                "startup FlashInfer/cuDNN MLA suffix attention",
                true,
            )?;
            anyhow::ensure!(
                captured,
                "failed to capture startup FlashInfer/cuDNN MLA suffix graph row_bucket={row_capacity}"
            );
            slot.stream_synchronize()
                .context("synchronizing startup FlashInfer/cuDNN MLA suffix graph")
        })?;
    }
    FLASHINFER_CUDNN_MLA_SUFFIX_PREWARMED.with(|prewarmed| prewarmed.set(true));
    eprintln!(
        "real_full_startup_attention_prewarm_done row_buckets={} query_capacity={} elapsed_ms={:.3}",
        COORDINATOR_GRAPH_PREFILL_BUCKET_ROWS.len(),
        query_capacity,
        prewarm_start.elapsed().as_secs_f64() * 1_000.0,
    );
    Ok(())
}

fn flashinfer_glm52_attention_heads_supported(heads: usize) -> bool {
    matches!(heads, 16 | 64)
}

#[derive(Debug, Clone, Copy)]
pub(in crate::commands::real_full) struct FlashinferTargetKvPageTable {
    pub(in crate::commands::real_full) physical_pages: GlmrtDeviceBuffer,
    pub(in crate::commands::real_full) mapping_key: u64,
}

#[derive(Debug, Clone, Copy)]
pub(in crate::commands::real_full) enum FlashinferCompressedMlaKvInput {
    SplitBf16 {
        latent: GlmrtDeviceBuffer,
        rope: GlmrtDeviceBuffer,
    },
    Interleaved {
        payload: GlmrtDeviceBuffer,
        dtype: KvCacheDType,
        row_stride_bytes: usize,
        row_offset: usize,
        physical_page_table: Option<FlashinferTargetKvPageTable>,
        force_staged_hidden_projection: bool,
    },
}

#[derive(Clone, Copy)]
struct FlashinferCompressedMlaDecodeBuffers {
    q_nope: GlmrtDeviceBuffer,
    q_rope: GlmrtDeviceBuffer,
    kv: GlmrtDeviceBuffer,
    partial: GlmrtDeviceBuffer,
    partial_lse: GlmrtDeviceBuffer,
    accumulator: GlmrtDeviceBuffer,
    accumulator_lse: GlmrtDeviceBuffer,
    workspace: GlmrtDeviceBuffer,
}

#[derive(Clone, Copy)]
struct FlashinferPackedFp8MlaDecodeBuffers {
    q: GlmrtDeviceBuffer,
    kv: GlmrtDeviceBuffer,
    indices: GlmrtDeviceBuffer,
    topk_length: GlmrtDeviceBuffer,
    index_base: GlmrtDeviceBuffer,
    output: GlmrtDeviceBuffer,
    out_lse: GlmrtDeviceBuffer,
    mid_out: GlmrtDeviceBuffer,
    mid_lse: GlmrtDeviceBuffer,
}

#[derive(Clone, Copy)]
struct FlashinferPackedFp8MlaFullGraphBuffers {
    flashinfer: FlashinferPackedFp8MlaDecodeBuffers,
    q_nope: GlmrtDeviceBuffer,
    q_absorbed: GlmrtDeviceBuffer,
    q_rope: GlmrtDeviceBuffer,
    q_rope_staging: GlmrtDeviceBuffer,
    physical_page_table: Option<GlmrtDeviceBuffer>,
    kv_b_weight: GlmrtDeviceBuffer,
    value_weight: GlmrtDeviceBuffer,
    final_output: GlmrtDeviceBuffer,
    hidden_projection: Option<FlashinferMlaHiddenProjection>,
    hidden_projection_w4a16: Option<GlmrtB12xCoordinatorW4a16Buffers>,
    hidden_projection_w8a16_packed_o: Option<GlmrtB12xCoordinatorW4a16Buffers>,
}

#[derive(Clone, Copy)]
struct FlashinferPackedFp8MlaFullGraphGeometry {
    query_rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    rank: usize,
    weight_head_stride: usize,
    combined_query_row_bytes: usize,
}

#[derive(Clone, Copy)]
pub(in crate::commands::real_full) struct FlashinferMlaHiddenProjection {
    pub(in crate::commands::real_full) weight: GlmrtDeviceBuffer,
    pub(in crate::commands::real_full) output: GlmrtDeviceBuffer,
    pub(in crate::commands::real_full) hidden_dim: usize,
    pub(in crate::commands::real_full) w4a16: Option<CoordinatorW4a16ProjectionBuffers>,
    pub(in crate::commands::real_full) w8a16: Option<CoordinatorW8a16ProjectionBuffers>,
}

pub(in crate::commands::real_full) struct FlashinferCompressedMlaDecodeLaunch {
    pub(in crate::commands::real_full) backend: &'static str,
    pub(in crate::commands::real_full) hidden_projection_fused: bool,
    pub(in crate::commands::real_full) ready_event: Option<Arc<CoordinatorCudaEvent>>,
}

#[derive(Clone, Copy)]
pub(in crate::commands::real_full) struct FlashinferGlmDsaSparseMlaPrefillInput {
    pub(in crate::commands::real_full) layer_id: usize,
    pub(in crate::commands::real_full) q_nope: GlmrtDeviceBuffer,
    pub(in crate::commands::real_full) q_rope: GlmrtDeviceBuffer,
    pub(in crate::commands::real_full) dsa_query: Option<GlmrtDeviceBuffer>,
    pub(in crate::commands::real_full) dsa_weights: Option<GlmrtDeviceBuffer>,
    pub(in crate::commands::real_full) positions: GlmrtDeviceBuffer,
    pub(in crate::commands::real_full) packed_kv: GlmrtDeviceBuffer,
    pub(in crate::commands::real_full) kv_dtype: KvCacheDType,
    pub(in crate::commands::real_full) kv_row_stride_bytes: usize,
    pub(in crate::commands::real_full) index_k_cache: Option<GlmrtDeviceBuffer>,
    pub(in crate::commands::real_full) kv_b_weight: GlmrtDeviceBuffer,
    pub(in crate::commands::real_full) hidden_projection: Option<FlashinferMlaHiddenProjection>,
    pub(in crate::commands::real_full) total_rows: usize,
    pub(in crate::commands::real_full) prefix_rows: usize,
    pub(in crate::commands::real_full) query_rows: usize,
    pub(in crate::commands::real_full) heads: usize,
    pub(in crate::commands::real_full) nope_dim: usize,
    pub(in crate::commands::real_full) rope_dim: usize,
    pub(in crate::commands::real_full) v_dim: usize,
    pub(in crate::commands::real_full) rank: usize,
    pub(in crate::commands::real_full) max_tokens: usize,
    pub(in crate::commands::real_full) physical_token_base: usize,
    pub(in crate::commands::real_full) physical_page_table: Option<FlashinferTargetKvPageTable>,
    pub(in crate::commands::real_full) theta: f32,
    pub(in crate::commands::real_full) scale: f32,
}

pub(in crate::commands::real_full) struct FlashinferGlmDsaSparseMlaPrefillLaunch {
    pub(in crate::commands::real_full) backend: &'static str,
    pub(in crate::commands::real_full) output: GlmrtDeviceBuffer,
    pub(in crate::commands::real_full) hidden_projection_fused: bool,
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn flashinfer_compressed_mla_decode_device_buffers(
    layer_id: usize,
    q_nope_buffer: GlmrtDeviceBuffer,
    q_rope_buffer: GlmrtDeviceBuffer,
    kv_input: FlashinferCompressedMlaKvInput,
    kv_b_weight: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    query_row_offset: usize,
    query_rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    scale: f32,
    hidden_projection: Option<FlashinferMlaHiddenProjection>,
) -> Result<FlashinferCompressedMlaDecodeLaunch> {
    anyhow::ensure!(
        attention_python_capture_enabled(),
        "FlashInfer compressed MLA decode requires attention Python graph capture"
    );
    anyhow::ensure!(
        (1..=FLASHINFER_PACKED_FP8_MLA_MAX_QUERY_ROWS).contains(&query_rows),
        "FlashInfer compressed MLA suffix requires 1..={} query rows, got {query_rows}",
        FLASHINFER_PACKED_FP8_MLA_MAX_QUERY_ROWS,
    );
    anyhow::ensure!(
        query_row_offset < rows && query_row_offset + query_rows == rows,
        "FlashInfer compressed MLA decode query must be the final KV row"
    );
    anyhow::ensure!(
        flashinfer_glm52_attention_heads_supported(heads)
            && rope_dim == GLM52_MLA_QK_ROPE_HEAD_DIM
            && GLM52_MLA_KV_LORA_RANK == 512,
        "FlashInfer compressed GLM-5.2 decode requires heads=16 or 64, rope_dim=64, and rank=512"
    );
    let bf16_bytes = std::mem::size_of::<u16>();
    let q_nope_row_bytes = heads
        .checked_mul(nope_dim)
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("FlashInfer compressed MLA decode q_nope row bytes overflow")?;
    let q_rope_row_bytes = heads
        .checked_mul(rope_dim)
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("FlashInfer compressed MLA decode q_rope row bytes overflow")?;
    let q_nope = device_buffer_byte_view(
        q_nope_buffer,
        0,
        query_rows
            .checked_mul(q_nope_row_bytes)
            .context("FlashInfer compressed MLA suffix q_nope bytes overflow")?,
        "FlashInfer compressed MLA decode compact q_nope suffix",
    )?;
    let q_rope = device_buffer_byte_view(
        q_rope_buffer,
        0,
        query_rows
            .checked_mul(q_rope_row_bytes)
            .context("FlashInfer compressed MLA suffix q_rope bytes overflow")?,
        "FlashInfer compressed MLA decode compact q_rope suffix",
    )?;
    let rank = GLM52_MLA_KV_LORA_RANK;
    let head_width = nope_dim
        .checked_add(v_dim)
        .context("FlashInfer compressed MLA decode KV-B head width overflow")?;
    let weight_head_stride = head_width
        .checked_mul(rank)
        .context("FlashInfer compressed MLA decode KV-B head stride overflow")?;
    let weight_bytes = heads
        .checked_mul(weight_head_stride)
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("FlashInfer compressed MLA decode KV-B weight bytes overflow")?;
    anyhow::ensure!(
        kv_b_weight.bytes >= weight_bytes,
        "FlashInfer compressed MLA decode KV-B weight has {} bytes, expected at least {weight_bytes}",
        kv_b_weight.bytes
    );
    let value_weight_offset = nope_dim
        .checked_mul(rank)
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("FlashInfer compressed MLA decode value-weight offset overflow")?;
    let value_weight = device_buffer_byte_view(
        kv_b_weight,
        value_weight_offset,
        kv_b_weight
            .bytes
            .checked_sub(value_weight_offset)
            .context("FlashInfer compressed MLA decode value-weight view exceeds KV-B weight")?,
        "FlashInfer compressed MLA decode value weights",
    )?;
    let latent_bytes = heads
        .checked_mul(rank)
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("FlashInfer compressed MLA decode latent scratch bytes overflow")?;
    let lse_bytes = heads
        .checked_mul(std::mem::size_of::<f32>())
        .context("FlashInfer compressed MLA decode LSE scratch bytes overflow")?;
    let kv_row_bytes = rank
        .checked_add(rope_dim)
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("FlashInfer compressed MLA decode KV row bytes overflow")?;
    let kv_staging_bytes = FLASHINFER_COMPRESSED_MLA_MAX_CHUNK_ROWS
        .checked_mul(kv_row_bytes)
        .context("FlashInfer compressed MLA decode KV staging bytes overflow")?;
    let output_bytes = heads
        .checked_mul(v_dim)
        .and_then(|values| values.checked_mul(query_rows))
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("FlashInfer compressed MLA decode output bytes overflow")?;
    let _ = device_buffer_byte_view(
        output_buffer,
        0,
        output_bytes,
        "FlashInfer compressed MLA decode output",
    )?;

    if let FlashinferCompressedMlaKvInput::Interleaved {
        payload,
        dtype: KvCacheDType::Fp8,
        row_stride_bytes,
        row_offset,
        physical_page_table,
        force_staged_hidden_projection,
    } = kv_input
    {
        if rows <= FLASHINFER_COMPRESSED_MLA_MAX_CHUNK_ROWS {
            // A speculative rewind can make the same long layer-78 prefix
            // non-contiguous after startup. That path must use the chunked
            // compressed fallback even though startup itself sees a packed
            // contiguous prefix. Seed its Python-backed graphs while capture
            // is open so the later adaptive retry never captures at runtime.
            if query_rows == 1 {
                prewarm_flashinfer_compressed_mla_decode_fallback_graphs(
                    layer_id, heads, nope_dim, rope_dim, v_dim, rank, scale,
                )?;
            }
            return flashinfer_packed_fp8_mla_decode_device_buffers(
                layer_id,
                q_nope,
                q_rope,
                payload,
                row_stride_bytes,
                row_offset,
                physical_page_table,
                kv_b_weight,
                value_weight,
                output_buffer,
                rows,
                query_rows,
                heads,
                nope_dim,
                rope_dim,
                v_dim,
                rank,
                weight_head_stride,
                scale,
                hidden_projection,
                force_staged_hidden_projection,
            );
        }
    }

    anyhow::ensure!(
        query_rows == 1,
        "non-packed FlashInfer compressed MLA decode requires one query row"
    );

    // Compressed decode graphs are universal across layers and context lengths.
    // Keep them in their own one-row slot: scalar target-attention operations
    // have different scratch envelopes and must not evict these startup graphs.
    let graph_key = coord_compressed_attention_decode_graph_key_for_layer(layer_id)?;
    with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
        let q_absorbed = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::H,
            latent_bytes.max(heads * (nope_dim + rope_dim) * bf16_bytes),
            "FlashInfer compressed MLA absorbed query",
        )?;
        let q_rope_staging = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::I,
            (heads * (nope_dim + rope_dim) * bf16_bytes).max(q_rope_row_bytes),
            "FlashInfer compressed MLA RoPE query",
        )?;
        let workspace = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::J,
            FLASHINFER_SINGLE_PREFILL_TMP_BYTES,
            "FlashInfer compressed MLA workspace",
        )?;
        let kv_staging = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::K,
            kv_staging_bytes.max(output_bytes),
            "FlashInfer compressed MLA KV staging",
        )?;
        let partial = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::L,
            latent_bytes.max(q_nope_row_bytes),
            "FlashInfer compressed MLA partial state",
        )?;
        let partial_lse = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::M,
            lse_bytes.max(q_rope_row_bytes),
            "FlashInfer compressed MLA partial LSE",
        )?;
        let accumulator = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::N,
            latent_bytes.max(q_nope_row_bytes),
            "FlashInfer compressed MLA accumulated state",
        )?;
        let accumulator_lse = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::O,
            lse_bytes.max(rope_dim * bf16_bytes),
            "FlashInfer compressed MLA accumulated LSE",
        )?;
        let _expanded_value_staging = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::P,
            output_bytes,
            "FlashInfer expanded MLA value staging",
        )?;

        let buffers = FlashinferCompressedMlaDecodeBuffers {
            q_nope: q_absorbed,
            q_rope: q_rope_staging,
            kv: kv_staging,
            partial,
            partial_lse,
            accumulator,
            accumulator_lse,
            workspace,
        };
        ensure_flashinfer_compressed_mla_decode_graphs(
            library, slot, buffers, heads, rank, rope_dim, scale,
        )?;

        let stream = slot.stream_ptr();
        unsafe {
            library
                .cuda_matmul_bf16_strided_batched_cublas_async(
                    q_nope,
                    kv_b_weight,
                    q_absorbed,
                    heads,
                    1,
                    nope_dim,
                    rank,
                    nope_dim,
                    weight_head_stride,
                    rank,
                    stream,
                )
                .context("absorbing FlashInfer compressed MLA decode queries")?;
            library
                .copy_d2d_async(q_rope_staging, q_rope, q_rope_row_bytes, stream)
                .context("staging FlashInfer compressed MLA RoPE query")?;
        }

        let mut row_offset = 0_usize;
        let mut first_chunk = true;
        while row_offset < rows {
            let chunk_rows = flashinfer_compressed_mla_decode_chunk_rows(rows - row_offset);
            stage_flashinfer_compressed_mla_kv_chunk(
                library, kv_input, kv_staging, row_offset, chunk_rows, rank, rope_dim, stream,
            )?;
            let signature = CoordinatorCudaGraphSignature::flashinfer_compressed_mla_decode_bf16(
                chunk_rows, heads, rank, rope_dim, scale,
            );
            let program = if first_chunk {
                CoordinatorCudaGraphProgram::LayerFlashinferCompressedMlaDecodeBf16Init
            } else {
                CoordinatorCudaGraphProgram::LayerFlashinferCompressedMlaDecodeBf16Merge
            };
            slot.launch_captured_graph(library, program, signature)
                .with_context(|| {
                    format!(
                        "launching FlashInfer compressed MLA decode chunk rows={chunk_rows} offset={row_offset}"
                    )
                })?;
            first_chunk = false;
            row_offset += chunk_rows;
        }

        unsafe {
            library
                .cuda_linear_bf16_strided_batched_cublas_async(
                    accumulator,
                    value_weight,
                    output_buffer,
                    heads,
                    1,
                    rank,
                    v_dim,
                    rank,
                    weight_head_stride,
                    v_dim,
                    stream,
                )
                .context("expanding FlashInfer compressed MLA decode values")?;
            library
                .cuda_stream_synchronize(stream)
                .context("synchronizing FlashInfer compressed MLA decode stream")?;
        }
        let backend = match kv_input {
            FlashinferCompressedMlaKvInput::SplitBf16 { .. }
            | FlashinferCompressedMlaKvInput::Interleaved {
                dtype: KvCacheDType::Bf16,
                ..
            } => FLASHINFER_COMPRESSED_MLA_DECODE_BF16_BACKEND,
            FlashinferCompressedMlaKvInput::Interleaved {
                dtype: KvCacheDType::Fp8,
                ..
            } => FLASHINFER_COMPRESSED_MLA_DECODE_FP8_BACKEND,
            FlashinferCompressedMlaKvInput::Interleaved {
                dtype: KvCacheDType::Nvfp4,
                ..
            } => FLASHINFER_COMPRESSED_MLA_DECODE_NVFP4_BACKEND,
            FlashinferCompressedMlaKvInput::Interleaved { dtype, .. } => {
                anyhow::bail!("unsupported compressed MLA cache dtype {}", dtype.label())
            }
        };
        Ok(FlashinferCompressedMlaDecodeLaunch {
            backend,
            hidden_projection_fused: false,
            ready_event: None,
        })
    })
}

fn flashinfer_direct_packed_fp8_mla_capacity_supported(bytes: usize) -> bool {
    bytes % FLASHINFER_PACKED_FP8_MLA_ROW_BYTES == 0
        && bytes / FLASHINFER_PACKED_FP8_MLA_ROW_BYTES % GLMRT_CUDA_GLM_DSA_PAGE_SIZE == 0
}

#[allow(clippy::too_many_arguments)]
fn flashinfer_packed_fp8_mla_decode_device_buffers(
    layer_id: usize,
    q_nope: GlmrtDeviceBuffer,
    q_rope: GlmrtDeviceBuffer,
    packed_kv: GlmrtDeviceBuffer,
    packed_kv_row_stride_bytes: usize,
    packed_kv_row_offset: usize,
    physical_page_table: Option<FlashinferTargetKvPageTable>,
    kv_b_weight: GlmrtDeviceBuffer,
    value_weight: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    query_rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    rank: usize,
    weight_head_stride: usize,
    scale: f32,
    hidden_projection: Option<FlashinferMlaHiddenProjection>,
    force_staged_hidden_projection: bool,
) -> Result<FlashinferCompressedMlaDecodeLaunch> {
    anyhow::ensure!(
        (1..=FLASHINFER_PACKED_FP8_MLA_MAX_QUERY_ROWS).contains(&query_rows),
        "packed FP8 MLA requires 1..={} query rows, got {query_rows}",
        FLASHINFER_PACKED_FP8_MLA_MAX_QUERY_ROWS,
    );
    anyhow::ensure!(
        packed_kv_row_stride_bytes >= FLASHINFER_PACKED_FP8_MLA_ROW_BYTES,
        "packed FP8 MLA cache row stride {packed_kv_row_stride_bytes} is smaller than the {}-byte FlashInfer GLM ABI",
        FLASHINFER_PACKED_FP8_MLA_ROW_BYTES
    );
    let packed_visible_kv = if let Some(page_table) = physical_page_table {
        let required_page_bytes = rows
            .checked_add(GLMRT_CUDA_GLM_DSA_PAGE_SIZE - 1)
            .map(|rows| rows / GLMRT_CUDA_GLM_DSA_PAGE_SIZE)
            .and_then(|pages| pages.checked_mul(std::mem::size_of::<u32>()))
            .context("packed FP8 MLA physical page-table bytes overflow")?;
        anyhow::ensure!(
            packed_kv_row_offset == 0
                && !page_table.physical_pages.ptr.is_null()
                && page_table.physical_pages.bytes >= required_page_bytes
                && page_table.physical_pages.device_id == packed_kv.device_id
                && page_table.mapping_key != 0,
            "paged packed FP8 MLA page-table contract is invalid"
        );
        None
    } else {
        let packed_end_row = packed_kv_row_offset
            .checked_add(rows)
            .context("packed FP8 MLA source row range overflow")?;
        let packed_source_bytes = packed_end_row
            .checked_mul(packed_kv_row_stride_bytes)
            .context("packed FP8 MLA source byte count overflow")?;
        anyhow::ensure!(
            packed_kv.bytes >= packed_source_bytes,
            "packed FP8 MLA cache has {} bytes, expected at least {packed_source_bytes}",
            packed_kv.bytes
        );
        let packed_visible_offset = packed_kv_row_offset
            .checked_mul(packed_kv_row_stride_bytes)
            .context("packed FP8 MLA visible source offset overflow")?;
        Some(device_buffer_byte_view(
            packed_kv,
            packed_visible_offset,
            rows.checked_mul(packed_kv_row_stride_bytes)
                .context("packed FP8 MLA visible source bytes overflow")?,
            "packed FP8 MLA visible cache rows",
        )?)
    };
    let bucket_rows = FLASHINFER_PACKED_FP8_MLA_BUCKETS
        .iter()
        .copied()
        .find(|bucket| rows <= *bucket)
        .with_context(|| format!("packed FP8 MLA has no graph bucket for {rows} rows"))?;
    let bf16_bytes = std::mem::size_of::<u16>();
    let max_query_rows = FLASHINFER_PACKED_FP8_MLA_MAX_QUERY_ROWS;
    let max_latent_bytes = max_query_rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(rank))
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("packed FP8 MLA maximum latent byte count overflow")?;
    let combined_query_row_bytes = rank
        .checked_add(rope_dim)
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("packed FP8 MLA combined query row byte count overflow")?;
    let max_combined_query_bytes = max_query_rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(combined_query_row_bytes))
        .context("packed FP8 MLA maximum combined query byte count overflow")?;
    let q_nope_input_bytes = query_rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(nope_dim))
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("packed FP8 MLA q-nope input byte count overflow")?;
    let max_q_nope_input_bytes = max_query_rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(nope_dim))
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("packed FP8 MLA maximum q-nope input byte count overflow")?;
    let q_rope_input_bytes = query_rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(rope_dim))
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("packed FP8 MLA q-rope input byte count overflow")?;
    let max_q_rope_input_bytes = max_query_rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(rope_dim))
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("packed FP8 MLA maximum q-rope input byte count overflow")?;
    let attention_output_bytes = query_rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(v_dim))
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("packed FP8 MLA attention output byte count overflow")?;
    let max_attention_output_bytes = max_query_rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(v_dim))
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("packed FP8 MLA maximum attention output byte count overflow")?;
    let max_hidden_projection_output_bytes = hidden_projection
        .map(|projection| {
            projection
                .hidden_dim
                .checked_mul(max_query_rows)
                .and_then(|values| values.checked_mul(bf16_bytes))
                .context("packed FP8 MLA maximum hidden projection output byte count overflow")
        })
        .transpose()?
        .unwrap_or(0);
    let max_lse_bytes = max_query_rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context("packed FP8 MLA maximum LSE byte count overflow")?;
    let max_bucket_rows = *FLASHINFER_PACKED_FP8_MLA_BUCKETS
        .last()
        .expect("packed FP8 MLA bucket table is nonempty");
    let max_splits = max_bucket_rows / 64;
    let max_indices_capacity_bytes = max_query_rows
        .checked_mul(max_bucket_rows)
        .and_then(|values| values.checked_mul(std::mem::size_of::<i32>()))
        .context("packed FP8 MLA maximum index byte count overflow")?;
    let max_mid_out_capacity_bytes = max_query_rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(max_splits))
        .and_then(|values| values.checked_mul(rank))
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("packed FP8 MLA maximum split output byte count overflow")?;
    let max_mid_lse_capacity_bytes = max_query_rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(max_splits))
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context("packed FP8 MLA maximum split LSE byte count overflow")?;
    let topk_length_capacity_bytes = max_query_rows
        .checked_mul(std::mem::size_of::<i32>())
        .context("packed FP8 MLA maximum top-k length byte count overflow")?;
    let index_base_offset = topk_length_capacity_bytes;
    let indices_offset = index_base_offset
        .checked_add(topk_length_capacity_bytes)
        .context("packed FP8 MLA index-base metadata offset overflow")?;
    let mid_lse_offset = indices_offset
        .checked_add(max_indices_capacity_bytes)
        .context("packed FP8 MLA metadata offset overflow")?;
    let metadata_capacity_bytes = topk_length_capacity_bytes
        .checked_add(topk_length_capacity_bytes)
        .and_then(|bytes| bytes.checked_add(max_indices_capacity_bytes))
        .and_then(|bytes| bytes.checked_add(max_mid_lse_capacity_bytes))
        .context("packed FP8 MLA maximum metadata byte count overflow")?;
    let max_packed_kv_bytes = max_bucket_rows
        .checked_mul(FLASHINFER_PACKED_FP8_MLA_ROW_BYTES)
        .context("packed FP8 MLA staging byte count overflow")?;
    if let Some(projection) = hidden_projection {
        let input_width = heads
            .checked_mul(v_dim)
            .context("packed FP8 MLA hidden projection input width overflow")?;
        let output_bytes = projection
            .hidden_dim
            .checked_mul(query_rows)
            .and_then(|values| values.checked_mul(bf16_bytes))
            .context("packed FP8 MLA hidden projection output bytes overflow")?;
        anyhow::ensure!(
            projection.hidden_dim > 0 && projection.output.bytes >= output_bytes,
            "packed FP8 MLA hidden projection output buffer has {} bytes, expected at least {output_bytes}",
            projection.output.bytes
        );
        if let Some(w8a16) = projection.w8a16 {
            let weight_bytes = projection
                .hidden_dim
                .checked_mul(input_width)
                .context("packed FP8 MLA W8A16 hidden projection weight bytes overflow")?;
            let scale_bytes = projection
                .hidden_dim
                .checked_mul(input_width / 256)
                .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
                .context("packed FP8 MLA W8A16 hidden projection scale bytes overflow")?;
            anyhow::ensure!(
                input_width % 256 == 0
                    && w8a16.weight.bytes >= weight_bytes
                    && w8a16.scales.bytes >= scale_bytes,
                "packed FP8 MLA W8A16 hidden projection buffers are too small: weight={}/{} scales={}/{}",
                w8a16.weight.bytes,
                weight_bytes,
                w8a16.scales.bytes,
                scale_bytes
            );
        } else {
            let weight_bytes = projection
                .hidden_dim
                .checked_mul(input_width)
                .and_then(|values| values.checked_mul(bf16_bytes))
                .context("packed FP8 MLA hidden projection weight bytes overflow")?;
            anyhow::ensure!(
                projection.weight.bytes >= weight_bytes,
                "packed FP8 MLA hidden projection weight buffer has {} bytes, expected at least {weight_bytes}",
                projection.weight.bytes
            );
        }
    }

    let graph_key = coord_attention_graph_key_for_layer_rows(layer_id, 1)?;
    with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
        let q_absorbed = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::H,
            max_latent_bytes,
            "FlashInfer packed FP8 MLA absorbed query",
        )?;
        let q_combined = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::I,
            max_combined_query_bytes,
            "FlashInfer packed FP8 MLA combined query",
        )?;
        let mid_out = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::J,
            max_mid_out_capacity_bytes,
            "FlashInfer packed FP8 MLA split output",
        )?;
        let kv_staging = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::K,
            max_packed_kv_bytes,
            "FlashInfer packed FP8 MLA cache staging",
        )?;
        let direct_capacity_bytes = packed_kv_row_offset
            .checked_add(max_bucket_rows)
            .and_then(|rows| rows.checked_mul(FLASHINFER_PACKED_FP8_MLA_ROW_BYTES))
            .context("direct packed FP8 MLA bucket capacity overflow")?;
        let direct_packed_kv = packed_kv_row_stride_bytes == FLASHINFER_PACKED_FP8_MLA_ROW_BYTES
            && packed_kv.bytes >= direct_capacity_bytes
            && flashinfer_direct_packed_fp8_mla_capacity_supported(packed_kv.bytes);
        anyhow::ensure!(
            physical_page_table.is_none() || direct_packed_kv,
            "paged packed FP8 MLA requires direct access to the full cache plane"
        );
        let attention_kv = if direct_packed_kv {
            packed_kv
        } else {
            kv_staging
        };
        let partial = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::L,
            max_latent_bytes,
            "FlashInfer packed FP8 MLA latent output",
        )?;
        let partial_lse = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::M,
            max_lse_bytes,
            "FlashInfer packed FP8 MLA output LSE",
        )?;
        let q_nope_staging = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::N,
            max_q_nope_input_bytes.max(max_attention_output_bytes),
            "FlashInfer packed FP8 MLA q-nope and output staging",
        )?;
        let q_rope_staging = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::O,
            max_q_rope_input_bytes.max(max_hidden_projection_output_bytes),
            "FlashInfer packed FP8 MLA q-rope and hidden output staging",
        )?;
        let metadata = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::P,
            metadata_capacity_bytes,
            "FlashInfer packed FP8 MLA indices and split LSE",
        )?;
        let topk_length = device_buffer_byte_view(
            metadata,
            0,
            topk_length_capacity_bytes,
            "FlashInfer packed FP8 MLA valid length",
        )?;
        let indices = device_buffer_byte_view(
            metadata,
            indices_offset,
            max_indices_capacity_bytes,
            "FlashInfer packed FP8 MLA indices",
        )?;
        let index_base = device_buffer_byte_view(
            metadata,
            index_base_offset,
            topk_length_capacity_bytes,
            "FlashInfer packed FP8 MLA physical index bases",
        )?;
        let mid_lse = device_buffer_byte_view(
            metadata,
            mid_lse_offset,
            max_mid_lse_capacity_bytes,
            "FlashInfer packed FP8 MLA split LSE",
        )?;
        let buffers = FlashinferPackedFp8MlaDecodeBuffers {
            q: q_combined,
            kv: attention_kv,
            indices,
            topk_length,
            index_base,
            output: partial,
            out_lse: partial_lse,
            mid_out,
            mid_lse,
        };
        let q_combined_rope = device_buffer_byte_view(
            q_combined,
            rank * bf16_bytes,
            q_combined
                .bytes
                .checked_sub(rank * bf16_bytes)
                .context("packed FP8 MLA RoPE query view exceeds combined query")?,
            "FlashInfer packed FP8 MLA RoPE query destination",
        )?;
        // A request owns its compact target page list, so its device pointer
        // changes when an execution lane is rebound. Capturing that pointer in
        // every query-width graph would require one graph identity per request
        // lane and leaves cached-prefix admission shapes uncovered. Stage the
        // small page list into graph-slot-owned storage instead. The D2D copy
        // remains outside the graph on the same stream; the captured physical
        // index expansion then consumes only this stable pointer.
        let stable_physical_page_table = if let Some(page_table) = physical_page_table {
            let physical_pool_pages = packed_kv
                .bytes
                .checked_div(FLASHINFER_PACKED_FP8_MLA_ROW_BYTES)
                .and_then(|rows| {
                    rows.checked_add(GLMRT_CUDA_GLM_DSA_PAGE_SIZE - 1)
                        .map(|rows| rows / GLMRT_CUDA_GLM_DSA_PAGE_SIZE)
                })
                .context("packed FP8 MLA physical pool page count overflow")?;
            let stable_page_table_bytes = physical_pool_pages
                .checked_mul(std::mem::size_of::<u32>())
                .context("packed FP8 MLA stable page-table bytes overflow")?;
            anyhow::ensure!(
                page_table.physical_pages.bytes >= stable_page_table_bytes,
                "packed FP8 MLA source page table has {} bytes, needs {stable_page_table_bytes}",
                page_table.physical_pages.bytes,
            );
            let stable_page_table = slot.buffer(
                library,
                CoordinatorCudaScratchSlot::V,
                stable_page_table_bytes,
                "FlashInfer packed FP8 MLA stable physical page table",
            )?;
            let mapping_identity = (
                page_table.physical_pages.ptr as usize,
                page_table.mapping_key,
            );
            if slot.stable_packed_fp8_mla_page_mapping != Some(mapping_identity) {
                let stream = slot.stream_ptr();
                unsafe {
                    library
                        .copy_d2d_async(
                            stable_page_table,
                            page_table.physical_pages,
                            stable_page_table_bytes,
                            stream,
                        )
                        .context("staging FlashInfer packed FP8 MLA physical page table")?;
                }
                slot.stable_packed_fp8_mla_page_mapping = Some(mapping_identity);
            }
            Some(stable_page_table)
        } else {
            None
        };
        // Query split/RoPE and packed attention are ordered on this same stream,
        // so BF16 or native W8A16 O projection may overwrite the now-consumed Q
        // allocation. The W4 launch contract still owns a staging destination;
        // keep it on the explicit staged/copy path until that contract is
        // redesigned.
        let direct_hidden_output = query_rows == 1
            && !force_staged_hidden_projection
            && packed_fp8_mla_direct_hidden_output_enabled()
            && hidden_projection.is_some_and(|projection| projection.w4a16.is_none());
        let requested_hidden_projection_output =
            hidden_projection.map(|projection| projection.output);
        let (hidden_projection_output, hidden_projection) = stage_flashinfer_hidden_projection(
            hidden_projection,
            q_rope_staging,
            direct_hidden_output,
        );
        let hidden_projection_w4a16 = (query_rows == 1)
            .then(|| hidden_projection.and_then(|projection| projection.w4a16))
            .flatten()
            .map(|projection| {
                coordinator_w4a16_launch_buffers(
                    library,
                    slot,
                    projection,
                    q_nope_staging,
                    q_rope_staging,
                    CoordinatorCudaScratchSlot::S,
                )
            })
            .transpose()?;
        let available_hidden_projection_w8a16_packed_o = hidden_projection
            .and_then(|projection| projection.w8a16)
            .filter(|projection| projection.packed_layout)
            // The grouped packed-O workspace is isolated in scratch T/U. It
            // must not resize shared attention slots after startup because
            // that would invalidate otherwise-unrelated DSA graph captures.
            .filter(|_| query_rows >= 9 || coordinator_python_capture_startup_open())
            .map(|projection| {
                coordinator_w8a16_packed_o_launch_buffers(
                    library,
                    slot,
                    projection,
                    q_nope_staging,
                    q_rope_staging,
                    CoordinatorCudaScratchSlot::U,
                )
            })
            .transpose()?;
        let hidden_projection_w8a16_packed_o = (query_rows >= 9)
            .then_some(available_hidden_projection_w8a16_packed_o)
            .flatten();
        let full_graph_buffers = FlashinferPackedFp8MlaFullGraphBuffers {
            flashinfer: buffers,
            q_nope: q_nope_staging,
            q_absorbed,
            q_rope: q_rope_staging,
            q_rope_staging: q_combined_rope,
            physical_page_table: stable_physical_page_table,
            kv_b_weight,
            value_weight,
            final_output: q_nope_staging,
            hidden_projection,
            hidden_projection_w4a16,
            hidden_projection_w8a16_packed_o,
        };
        // Direct one-row O projection removes a small D2D copy, but its CUDA
        // graph identity includes the request-owned output pointer. Startup
        // covers the recurrent pointer set; unusual prefill chunk shapes can
        // select another rotation later, after the Python graph-capture bridge
        // has closed. Pre-capture a stable scratch-output identity per layer so
        // those requests can fall back without turning a valid API request
        // into a 502.
        let stable_one_row_graph_buffers = direct_hidden_output.then(|| {
            let stable_hidden_projection =
                hidden_projection.map(|projection| FlashinferMlaHiddenProjection {
                    output: q_rope_staging,
                    ..projection
                });
            FlashinferPackedFp8MlaFullGraphBuffers {
                hidden_projection: stable_hidden_projection,
                hidden_projection_w4a16: None,
                ..full_graph_buffers
            }
        });
        let full_graph_geometry = FlashinferPackedFp8MlaFullGraphGeometry {
            query_rows,
            heads,
            nope_dim,
            rope_dim,
            v_dim,
            rank,
            weight_head_stride,
            combined_query_row_bytes,
        };
        let capture_identity = flashinfer_packed_fp8_mla_capture_identity(full_graph_buffers);
        if coordinator_python_capture_startup_open() {
            if query_rows == 1 {
                ensure_flashinfer_packed_fp8_mla_decode_graphs(
                    library,
                    slot,
                    full_graph_buffers,
                    full_graph_geometry,
                    scale,
                    rows,
                    !direct_packed_kv,
                    if direct_packed_kv {
                        packed_kv_row_offset
                    } else {
                        0
                    },
                    capture_identity,
                )?;
                if let Some(stable_graph_buffers) = stable_one_row_graph_buffers {
                    let stable_capture_identity =
                        flashinfer_packed_fp8_mla_capture_identity(stable_graph_buffers);
                    ensure_flashinfer_packed_fp8_mla_decode_graphs(
                        library,
                        slot,
                        stable_graph_buffers,
                        full_graph_geometry,
                        scale,
                        rows,
                        !direct_packed_kv,
                        if direct_packed_kv {
                            packed_kv_row_offset
                        } else {
                            0
                        },
                        stable_capture_identity,
                    )?;
                }
                if layer_id == GLM52_MTP_LAYER_ID && direct_packed_kv {
                    // MTP rewind can leave its logical layer-78 prefix split
                    // across physical blocks even when the target layers stay
                    // direct. Qualify the packed gather/staging identity now,
                    // rather than falling back to a full BF16 KV unpack after
                    // Python graph capture has closed.
                    let staged_hidden_projection =
                        full_graph_buffers.hidden_projection.map(|projection| {
                            FlashinferMlaHiddenProjection {
                                output: q_rope_staging,
                                ..projection
                            }
                        });
                    let staged_graph_buffers = FlashinferPackedFp8MlaFullGraphBuffers {
                        flashinfer: FlashinferPackedFp8MlaDecodeBuffers {
                            kv: kv_staging,
                            ..full_graph_buffers.flashinfer
                        },
                        physical_page_table: None,
                        hidden_projection: staged_hidden_projection,
                        ..full_graph_buffers
                    };
                    ensure_flashinfer_packed_fp8_mla_decode_graphs(
                        library,
                        slot,
                        staged_graph_buffers,
                        full_graph_geometry,
                        scale,
                        rows,
                        true,
                        0,
                        flashinfer_packed_fp8_mla_capture_identity(staged_graph_buffers),
                    )?;
                }
            } else {
                // Later batched startup passes can resize shared attention
                // scratch after the serial one-row sweep, which clears every
                // graph in this slot. Re-establish the stable one-row identity
                // from the final workspace addresses. Production one-row
                // requests may own a different hidden-output pointer, but
                // they already fall back to this staged destination and copy
                // the result on the same stream.
                let one_row_hidden_projection =
                    full_graph_buffers.hidden_projection.map(|projection| {
                        FlashinferMlaHiddenProjection {
                            output: q_rope_staging,
                            ..projection
                        }
                    });
                let one_row_graph_buffers = FlashinferPackedFp8MlaFullGraphBuffers {
                    hidden_projection: one_row_hidden_projection,
                    hidden_projection_w4a16: None,
                    hidden_projection_w8a16_packed_o: None,
                    ..full_graph_buffers
                };
                ensure_flashinfer_packed_fp8_mla_decode_graphs(
                    library,
                    slot,
                    one_row_graph_buffers,
                    FlashinferPackedFp8MlaFullGraphGeometry {
                        query_rows: 1,
                        ..full_graph_geometry
                    },
                    scale,
                    rows,
                    !direct_packed_kv,
                    if direct_packed_kv {
                        packed_kv_row_offset
                    } else {
                        0
                    },
                    flashinfer_packed_fp8_mla_capture_identity(one_row_graph_buffers),
                )?;
            }

            // The guaranteed recurrent startup request visits every layer with
            // one query row. Use that pass to capture every exact batched
            // suffix width too, before the Python capture bridge closes.
            // W8 target suffixes use a stable scratch destination so startup
            // can capture M=2--8 without binding those graphs to a one-row
            // request allocation. Production copies the projected result to
            // the request-owned output on this same stream before synchronizing
            // it. W4 and BF16 retain their prior external path.
            let batched_hidden_projection = full_graph_buffers
                .hidden_projection
                .filter(|projection| projection.w8a16.is_some())
                .map(|projection| FlashinferMlaHiddenProjection {
                    output: q_rope_staging,
                    ..projection
                });
            let batched_graph_buffers = FlashinferPackedFp8MlaFullGraphBuffers {
                hidden_projection: batched_hidden_projection,
                hidden_projection_w4a16: None,
                ..full_graph_buffers
            };
            for capture_query_rows in 2..=FLASHINFER_PACKED_FP8_MLA_MAX_QUERY_ROWS {
                let capture_graph_buffers = FlashinferPackedFp8MlaFullGraphBuffers {
                    hidden_projection_w8a16_packed_o: (capture_query_rows >= 9)
                        .then_some(available_hidden_projection_w8a16_packed_o)
                        .flatten(),
                    ..batched_graph_buffers
                };
                let batched_capture_identity =
                    flashinfer_packed_fp8_mla_capture_identity(capture_graph_buffers);
                ensure_flashinfer_packed_fp8_mla_decode_graphs(
                    library,
                    slot,
                    capture_graph_buffers,
                    FlashinferPackedFp8MlaFullGraphGeometry {
                        query_rows: capture_query_rows,
                        ..full_graph_geometry
                    },
                    scale,
                    rows,
                    !direct_packed_kv,
                    if direct_packed_kv {
                        packed_kv_row_offset
                    } else {
                        0
                    },
                    batched_capture_identity,
                )?;
            }
        }

        let stream = slot.stream_ptr();
        let cuda_timeline = AttentionCudaEventTimeline::enabled()
            .then(|| AttentionCudaEventTimeline::new(library, 3))
            .transpose()?;
        let async_w8a16_handoff = query_rows == 1
            && hidden_projection.is_some_and(|projection| projection.w8a16.is_some())
            && w8a16_async_attention_enabled()
            && cuda_timeline.is_none();
        unsafe {
            if let Some(timeline) = cuda_timeline.as_ref() {
                timeline.record(0, stream, "start")?;
            }
            library
                .copy_d2d_async(q_nope_staging, q_nope, q_nope_input_bytes, stream)
                .context("staging FlashInfer packed FP8 MLA q-nope input")?;
            library
                .copy_d2d_async(q_rope_staging, q_rope, q_rope_input_bytes, stream)
                .context("staging FlashInfer packed FP8 MLA q-rope input")?;
            if !direct_packed_kv {
                library
                    .copy_d2d_2d_async(
                        kv_staging,
                        FLASHINFER_PACKED_FP8_MLA_ROW_BYTES,
                        packed_visible_kv.expect("validated contiguous packed KV staging"),
                        packed_kv_row_stride_bytes,
                        FLASHINFER_PACKED_FP8_MLA_ROW_BYTES,
                        rows,
                        stream,
                    )
                    .context("compacting FlashInfer packed FP8 MLA cache rows")?;
            }
        }
        let prefix_rows = rows - query_rows;
        let runtime_index_base = i32::try_from(if direct_packed_kv {
            packed_kv_row_offset
        } else {
            0
        })
        .context("packed FP8 MLA physical index base does not fit i32")?
        .to_ne_bytes();
        let mut metadata_header =
            [0_u8; FLASHINFER_PACKED_FP8_MLA_MAX_QUERY_ROWS * std::mem::size_of::<i32>() * 2];
        for query_index in 0..query_rows {
            let valid_length = i32::try_from(prefix_rows + query_index + 1)
                .context("packed FP8 MLA valid length does not fit i32")?
                .to_ne_bytes();
            let byte_offset = query_index * std::mem::size_of::<i32>();
            metadata_header[byte_offset..byte_offset + valid_length.len()]
                .copy_from_slice(&valid_length);
            let base_offset = index_base_offset + byte_offset;
            metadata_header[base_offset..base_offset + runtime_index_base.len()]
                .copy_from_slice(&runtime_index_base);
        }
        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::P,
                &metadata_header,
                "FlashInfer packed FP8 MLA valid lengths and physical index bases",
                stream,
            )
            .context("staging FlashInfer packed FP8 MLA metadata header")?;
        unsafe {
            if let Some(timeline) = cuda_timeline.as_ref() {
                timeline.record(1, stream, "staged")?;
            }
        }

        let signature = CoordinatorCudaGraphSignature::flashinfer_packed_fp8_mla_decode(
            bucket_rows,
            query_rows,
            heads,
            rank,
            rope_dim,
            scale,
            if hidden_projection_w4a16.is_some() {
                2
            } else if hidden_projection.is_some_and(|projection| projection.w8a16.is_some()) {
                3
            } else {
                usize::from(hidden_projection.is_some())
            },
        );
        let program = CoordinatorCudaGraphProgram::LayerFlashinferPackedFp8MlaDecode;
        let (launch_capture_identity, launch_hidden_projection, launch_hidden_projection_output) =
            if slot.has_captured_graph_identity(program, signature, capture_identity) {
                (
                    capture_identity,
                    hidden_projection,
                    hidden_projection_output,
                )
            } else if let Some(stable_graph_buffers) = stable_one_row_graph_buffers {
                (
                    flashinfer_packed_fp8_mla_capture_identity(stable_graph_buffers),
                    stable_graph_buffers.hidden_projection,
                    requested_hidden_projection_output,
                )
            } else {
                (
                    capture_identity,
                    hidden_projection,
                    hidden_projection_output,
                )
            };
        slot.launch_captured_graph_identity(
            library,
            program,
            signature,
            launch_capture_identity,
        )
        .with_context(|| {
            format!(
                "launching FlashInfer packed FP8 MLA decode rows={rows} query_rows={query_rows} bucket={bucket_rows}"
            )
        })?;
        unsafe {
            if hidden_projection.is_none() {
                library
                    .copy_d2d_async(
                        output_buffer,
                        q_nope_staging,
                        attention_output_bytes,
                        stream,
                    )
                    .context("copying FlashInfer packed FP8 MLA attention output")?;
            } else if let (Some(projection), Some(copy_output)) =
                (launch_hidden_projection, launch_hidden_projection_output)
            {
                library
                    .copy_d2d_async(
                        copy_output,
                        projection.output,
                        query_rows * projection.hidden_dim * std::mem::size_of::<u16>(),
                        stream,
                    )
                    .context(
                        "copying FlashInfer packed FP8 MLA hidden projection output after graph",
                    )?;
            }
            if let Some(timeline) = cuda_timeline.as_ref() {
                timeline.record(2, stream, "full graph")?;
            }
        }
        let ready_event = if async_w8a16_handoff {
            Some(slot.record_output_ready_event(library)?)
        } else {
            unsafe {
                library
                    .cuda_stream_synchronize(stream)
                    .context("synchronizing FlashInfer packed FP8 MLA decode stream")?;
            }
            None
        };
        if let Some(timeline) = cuda_timeline.as_ref() {
            unsafe {
                eprintln!(
                    "real_full_packed_fp8_attention_cuda_timing layer_id={layer_id} rows={rows} bucket_rows={bucket_rows} direct_kv={direct_packed_kv} staging_ms={:.3} full_graph_ms={:.3} total_ms={:.3}",
                    timeline.elapsed_ms(0, 1, "staging")?,
                    timeline.elapsed_ms(1, 2, "full graph")?,
                    timeline.elapsed_ms(0, 2, "total")?,
                );
            }
        }
        Ok(FlashinferCompressedMlaDecodeLaunch {
            backend: FLASHINFER_PACKED_FP8_MLA_DECODE_BACKEND,
            hidden_projection_fused: hidden_projection.is_some(),
            ready_event,
        })
    })
}

pub(in crate::commands::real_full) fn flashinfer_glm_dsa_sparse_mla_prefill_device_buffers(
    input: FlashinferGlmDsaSparseMlaPrefillInput,
) -> Result<FlashinferGlmDsaSparseMlaPrefillLaunch> {
    anyhow::ensure!(
        attention_python_capture_enabled(),
        "GLM DSA sparse MLA prefill requires attention Python graph capture"
    );
    let (source_layer, full_indexer) = glm_dsa_index_source_layer(input.layer_id)
        .with_context(|| format!("layer {} has no GLM DSA index source", input.layer_id))?;
    anyhow::ensure!(
        (1..=GLM_DSA_PREFILL_MAX_QUERY_ROWS).contains(&input.query_rows),
        "GLM DSA sparse MLA requires 1..={} query rows, got {}",
        GLM_DSA_PREFILL_MAX_QUERY_ROWS,
        input.query_rows
    );
    anyhow::ensure!(
        input.prefix_rows.checked_add(input.query_rows) == Some(input.total_rows)
            && input.total_rows <= input.max_tokens,
        "GLM DSA sparse MLA geometry is invalid: prefix={} query={} total={} max_tokens={}",
        input.prefix_rows,
        input.query_rows,
        input.total_rows,
        input.max_tokens
    );
    anyhow::ensure!(
        input.heads == 64
            && input.nope_dim == 192
            && input.rope_dim == GLM52_MLA_QK_ROPE_HEAD_DIM
            && input.v_dim == 256
            && input.rank == GLM52_MLA_KV_LORA_RANK,
        "GLM DSA sparse MLA requires heads=64, q-nope=192, rope=64, value=256, rank=512"
    );
    anyhow::ensure!(
        input.theta.is_finite()
            && input.theta > 0.0
            && input.scale.is_finite()
            && input.scale > 0.0,
        "GLM DSA sparse MLA theta and scale must be finite and positive"
    );
    anyhow::ensure!(
        input.max_tokens > 0 && input.max_tokens % GLMRT_CUDA_GLM_DSA_PAGE_SIZE == 0,
        "GLM DSA sparse MLA max_tokens must be divisible by {}",
        GLMRT_CUDA_GLM_DSA_PAGE_SIZE
    );
    if let Some(page_table) = input.physical_page_table {
        let required_page_bytes = input
            .total_rows
            .checked_add(GLMRT_CUDA_GLM_DSA_PAGE_SIZE - 1)
            .map(|rows| rows / GLMRT_CUDA_GLM_DSA_PAGE_SIZE)
            .and_then(|pages| pages.checked_mul(std::mem::size_of::<u32>()))
            .context("GLM DSA sparse MLA physical page-table bytes overflow")?;
        anyhow::ensure!(
            !page_table.physical_pages.ptr.is_null()
                && page_table.physical_pages.bytes >= required_page_bytes
                && page_table.physical_pages.device_id == input.packed_kv.device_id
                && page_table.mapping_key != 0,
            "GLM DSA sparse MLA physical page-table contract is invalid"
        );
    } else {
        anyhow::ensure!(
            input.physical_token_base % GLMRT_CUDA_GLM_DSA_PAGE_SIZE == 0
                && input
                    .physical_token_base
                    .checked_add(input.total_rows)
                    .is_some_and(|end| end <= input.max_tokens),
            "GLM DSA sparse MLA physical extent base={} total={} exceeds {} tokens or is not {}-token aligned",
            input.physical_token_base,
            input.total_rows,
            input.max_tokens,
            GLMRT_CUDA_GLM_DSA_PAGE_SIZE,
        );
    }
    let max_pages = input.max_tokens / GLMRT_CUDA_GLM_DSA_PAGE_SIZE;
    anyhow::ensure!(
        max_pages <= GLM_DSA_PREFILL_MAX_CACHE_PAGES,
        "GLM DSA sparse MLA cache pages {max_pages} exceed supported maximum {}",
        GLM_DSA_PREFILL_MAX_CACHE_PAGES
    );
    let bucket_rows = glm_dsa_sparse_mla_query_bucket(input.kv_dtype, input.query_rows)
        .context("GLM DSA sparse MLA query has no graph bucket")?;
    let bf16_bytes = std::mem::size_of::<u16>();
    let q_nope_bytes = input
        .query_rows
        .checked_mul(input.heads)
        .and_then(|values| values.checked_mul(input.nope_dim))
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("GLM DSA sparse MLA q-nope bytes overflow")?;
    let q_rope_bytes = input
        .query_rows
        .checked_mul(input.heads)
        .and_then(|values| values.checked_mul(input.rope_dim))
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("GLM DSA sparse MLA q-rope bytes overflow")?;
    let positions_bytes = input
        .query_rows
        .checked_mul(std::mem::size_of::<u32>())
        .context("GLM DSA sparse MLA position bytes overflow")?;
    anyhow::ensure!(
        matches!(
            input.kv_dtype,
            KvCacheDType::Bf16 | KvCacheDType::Fp8 | KvCacheDType::Nvfp4
        ),
        "GLM DSA sparse MLA requires BF16, FP8, or NVFP4 compressed KV, got {}",
        input.kv_dtype.label()
    );
    let minimum_kv_row_bytes = match input.kv_dtype {
        KvCacheDType::Bf16 => {
            (GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM) * std::mem::size_of::<u16>()
        }
        KvCacheDType::Fp8 => FLASHINFER_PACKED_FP8_MLA_ROW_BYTES,
        KvCacheDType::Nvfp4 => GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN,
        _ => unreachable!("validated compressed GLM DSA KV dtype"),
    };
    anyhow::ensure!(
        input.kv_row_stride_bytes >= minimum_kv_row_bytes,
        "GLM DSA sparse MLA {} KV row stride {} is smaller than {minimum_kv_row_bytes}",
        input.kv_dtype.label(),
        input.kv_row_stride_bytes,
    );
    let packed_kv_bytes = input
        .max_tokens
        .checked_mul(input.kv_row_stride_bytes)
        .context("GLM DSA sparse MLA packed KV bytes overflow")?;
    let kv_b_head_width = input
        .nope_dim
        .checked_add(input.v_dim)
        .context("GLM DSA sparse MLA KV-B head width overflow")?;
    let kv_b_weight_bytes = input
        .heads
        .checked_mul(kv_b_head_width)
        .and_then(|values| values.checked_mul(input.rank))
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("GLM DSA sparse MLA KV-B weight bytes overflow")?;
    for (label, buffer, required_bytes) in [
        ("q-nope", input.q_nope, q_nope_bytes),
        ("q-rope", input.q_rope, q_rope_bytes),
        ("positions", input.positions, positions_bytes),
        ("packed KV", input.packed_kv, packed_kv_bytes),
        ("KV-B weight", input.kv_b_weight, kv_b_weight_bytes),
    ] {
        anyhow::ensure!(
            !buffer.ptr.is_null() && buffer.bytes >= required_bytes,
            "GLM DSA sparse MLA {label} buffer has {} bytes, expected at least {required_bytes}",
            buffer.bytes
        );
        anyhow::ensure!(
            buffer.device_id == input.packed_kv.device_id,
            "GLM DSA sparse MLA {label} is on device {}, expected device {}",
            buffer.device_id,
            input.packed_kv.device_id
        );
    }
    if let Some(projection) = input.hidden_projection {
        let input_dim = input
            .heads
            .checked_mul(input.v_dim)
            .context("GLM DSA hidden-projection input width overflow")?;
        let output_bytes = input
            .query_rows
            .checked_mul(projection.hidden_dim)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("GLM DSA hidden-projection output bytes overflow")?;
        anyhow::ensure!(
            projection.output.device_id == input.packed_kv.device_id
                && projection.output.bytes >= output_bytes,
            "GLM DSA hidden-projection output buffer contract is invalid: output_device={} kv_device={} output_bytes={} required_bytes={} bucket_rows={} hidden_dim={}",
            projection.output.device_id,
            input.packed_kv.device_id,
            projection.output.bytes,
            output_bytes,
            bucket_rows,
            projection.hidden_dim,
        );
        if let Some(w8a16) = projection.w8a16 {
            let weight_bytes = input_dim
                .checked_mul(projection.hidden_dim)
                .context("GLM DSA W8A16 O weight bytes overflow")?;
            let scale_bytes = projection
                .hidden_dim
                .checked_mul(input_dim.div_ceil(256))
                .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
                .context("GLM DSA W8A16 O scale bytes overflow")?;
            anyhow::ensure!(
                w8a16.weight.device_id == input.packed_kv.device_id
                    && w8a16.weight.bytes >= weight_bytes
                    && w8a16.scales.device_id == input.packed_kv.device_id
                    && w8a16.scales.bytes >= scale_bytes,
                "GLM DSA W8A16 hidden-projection weight contract is invalid"
            );
        }
    }

    let dsa_query_bytes = input
        .query_rows
        .checked_mul(GLMRT_CUDA_GLM_DSA_INDEX_HEADS)
        .and_then(|values| values.checked_mul(GLM52_DSA_INDEX_HEAD_DIM))
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("GLM DSA sparse MLA raw query bytes overflow")?;
    let dsa_weight_bytes = input
        .query_rows
        .checked_mul(GLMRT_CUDA_GLM_DSA_INDEX_HEADS)
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("GLM DSA sparse MLA raw weight bytes overflow")?;
    let run_selector = full_indexer && input.total_rows > GLM_DSA_PREFILL_TOPK;
    if run_selector {
        let dsa_query = input
            .dsa_query
            .context("full GLM DSA indexer layer is missing projected index query")?;
        let dsa_weights = input
            .dsa_weights
            .context("full GLM DSA indexer layer is missing projected head weights")?;
        let index_k_cache = input
            .index_k_cache
            .context("full GLM DSA indexer layer is missing packed index-K cache")?;
        let index_k_bytes = max_pages
            .checked_mul(GLMRT_CUDA_GLM_DSA_PACKED_PAGE_BYTES)
            .context("GLM DSA packed index-K cache bytes overflow")?;
        for (label, buffer, required_bytes) in [
            ("projected index query", dsa_query, dsa_query_bytes),
            ("projected head weights", dsa_weights, dsa_weight_bytes),
            ("packed index-K cache", index_k_cache, index_k_bytes),
        ] {
            anyhow::ensure!(
                !buffer.ptr.is_null()
                    && buffer.bytes >= required_bytes
                    && buffer.device_id == input.packed_kv.device_id,
                "GLM DSA sparse MLA {label} buffer contract is invalid"
            );
        }
    } else {
        anyhow::ensure!(
            input.dsa_query.is_none()
                && input.dsa_weights.is_none()
                && input.index_k_cache.is_none(),
            "GLM DSA layer {} must not provide index projections when reusing or directly enumerating layer {source_layer} indices",
            input.layer_id
        );
    }

    let selection_key = GlmDsaSelectionKey {
        source_layer,
        physical_token_base: input.physical_token_base,
        physical_page_table_ptr: input
            .physical_page_table
            .map_or(0, |page_table| page_table.physical_pages.ptr as usize),
        physical_page_table_key: input
            .physical_page_table
            .map_or(0, |page_table| page_table.mapping_key),
        total_rows: input.total_rows,
        prefix_rows: input.prefix_rows,
        query_rows: input.query_rows,
    };
    let source_has_shared_consumer = full_indexer
        && glm_dsa_index_source_layer(input.layer_id + 1)
            .is_some_and(|(next_source, next_full)| !next_full && next_source == source_layer);
    // Selection indices outlive the full indexer layer that produces them.
    // Bank every such result under its complete request geometry before the
    // shared scratch buffer can be rebound to another request. The previous
    // active-buffer optimization made correctness depend on request-major
    // layer execution: layer-major continuous batching could overwrite one
    // request's indices before its shared DSA layers consumed them.
    // When the causal history fits inside top-k, DSA necessarily selects every
    // visible token.  Populate the physical indices directly and avoid scoring
    // the auxiliary index cache; sparse MLA can still consume packed main KV.
    let graph_key = coord_attention_graph_key_for_layer_rows(input.layer_id, bucket_rows)?;
    with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
        GLM_DSA_SPARSE_MLA_PREFILL_WORKSPACE.with(|workspace| {
            let mut workspace = workspace
                .try_borrow_mut()
                .map_err(|_| anyhow::anyhow!("GLM DSA sparse MLA workspace is already borrowed"))?;
            let buffers = workspace.ensure(library, input.max_tokens, input.query_rows)?;
            let stream = slot.stream_ptr();
            let requested_hidden_projection_output =
                input.hidden_projection.map(|projection| projection.output);
            let graph_input = if let Some(projection) = input
                .hidden_projection
                .filter(|projection| projection.w4a16.is_none())
            {
                let stable_output_bytes = bucket_rows
                    .checked_mul(projection.hidden_dim)
                    .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
                    .context("GLM DSA stable hidden-projection output bytes overflow")?;
                FlashinferGlmDsaSparseMlaPrefillInput {
                    hidden_projection: Some(FlashinferMlaHiddenProjection {
                        output: device_buffer_byte_view(
                            buffers.auxiliary,
                            0,
                            stable_output_bytes,
                            "GLM DSA stable hidden-projection output",
                        )?,
                        ..projection
                    }),
                    ..input
                }
            } else {
                FlashinferGlmDsaSparseMlaPrefillInput {
                    hidden_projection: None,
                    ..input
                }
            };
            let cuda_timeline = AttentionCudaEventTimeline::enabled()
                .then(|| AttentionCudaEventTimeline::new(library, 3))
                .transpose()?;
            unsafe {
                if let Some(timeline) = cuda_timeline.as_ref() {
                    timeline.record(0, stream, "GLM DSA start")?;
                }
            }
            let shared_needs_banked_selection =
                !full_indexer && input.total_rows > GLM_DSA_PREFILL_TOPK;
            let banked_selection = if shared_needs_banked_selection {
                let selection_row_bytes = GLM_DSA_PREFILL_TOPK
                    .checked_mul(std::mem::size_of::<i32>())
                    .context("GLM DSA banked-selection row bytes overflow")?;
                let selection_bytes = input
                    .query_rows
                    .checked_mul(selection_row_bytes)
                    .context("GLM DSA banked-selection bytes overflow")?;
                let selection_buffer = workspace
                    .banked_selections
                    .get(&selection_key)
                    .with_context(|| {
                        format!(
                            "shared GLM DSA layer {} expected banked indices from layer {source_layer} for total={} prefix={} query={}",
                            input.layer_id,
                            input.total_rows,
                            input.prefix_rows,
                            input.query_rows,
                        )
                    })?
                    .buffer;
                Some((
                    device_buffer_byte_view(
                        selection_buffer,
                        0,
                        selection_bytes,
                        "GLM DSA live shared-selection bank",
                    )?,
                    selection_bytes,
                ))
            } else {
                None
            };
            if shared_needs_banked_selection {
                anyhow::ensure!(
                    workspace.banked_selections.contains_key(&selection_key),
                    "shared GLM DSA layer {} expected banked indices from layer {source_layer} for total={} prefix={} query={}",
                    input.layer_id,
                    input.total_rows,
                    input.prefix_rows,
                    input.query_rows,
                );
            }

            unsafe {
                if let Some(page_table) = input.physical_page_table {
                    let page_table_bytes = max_pages
                        .checked_mul(std::mem::size_of::<u32>())
                        .context("GLM DSA stable physical page-table bytes overflow")?;
                    anyhow::ensure!(
                        page_table.physical_pages.bytes >= page_table_bytes,
                        "GLM DSA physical page table has {} bytes, needs {page_table_bytes}",
                        page_table.physical_pages.bytes,
                    );
                    let mapping_identity = (
                        page_table.physical_pages.ptr as usize,
                        page_table.mapping_key,
                    );
                    if workspace.page_table_physical_mapping != Some(mapping_identity) {
                        library
                            .copy_d2d_async(
                                buffers.page_table,
                                page_table.physical_pages,
                                page_table_bytes,
                                stream,
                            )
                            .context("staging stable GLM DSA physical page table")?;
                        workspace.page_table_physical_mapping = Some(mapping_identity);
                        workspace.page_table_physical_page_base = None;
                    }
                } else {
                    let physical_page_base =
                        input.physical_token_base / GLMRT_CUDA_GLM_DSA_PAGE_SIZE;
                    if workspace.page_table_physical_page_base != Some(physical_page_base) {
                        library
                            .cuda_glm_dsa_page_table_init_base_async(
                                buffers.page_table,
                                1,
                                max_pages,
                                physical_page_base,
                                stream,
                            )
                            .context("rebasing repeated GLM DSA page table")?;
                        workspace.page_table_physical_page_base = Some(physical_page_base);
                        workspace.page_table_physical_mapping = None;
                    }
                }
                library
                    .copy_d2d_async(buffers.q_nope, input.q_nope, q_nope_bytes, stream)
                    .context("staging GLM DSA sparse MLA q-nope rows")?;
                library
                    .copy_d2d_async(buffers.q_rope, input.q_rope, q_rope_bytes, stream)
                    .context("staging GLM DSA sparse MLA q-rope rows")?;
                library
                    .copy_d2d_async(
                        buffers.dsa_positions,
                        input.positions,
                        positions_bytes,
                        stream,
                    )
                    .context("staging GLM DSA sparse MLA positions")?;
                if run_selector {
                    library
                        .copy_d2d_async(
                            buffers.dsa_query_raw,
                            input.dsa_query.expect("validated full DSA query"),
                            dsa_query_bytes,
                            stream,
                        )
                        .context("staging raw GLM DSA projected query")?;
                    library
                        .copy_d2d_async(
                            buffers.dsa_weights_raw,
                            input.dsa_weights.expect("validated full DSA weights"),
                            dsa_weight_bytes,
                            stream,
                        )
                        .context("staging raw GLM DSA projected head weights")?;
                }
                library
                    .cuda_glm_dsa_prefill_metadata_async(
                        buffers.cache_seqlens,
                        buffers.topk_lengths,
                        buffers.active_width,
                        bucket_rows,
                        input.query_rows,
                        input.prefix_rows,
                        input.total_rows,
                        GLM_DSA_PREFILL_TOPK,
                        stream,
                    )
                    .context("updating dynamic GLM DSA sparse MLA metadata")?;
                if shared_needs_banked_selection {
                    if let Some((selection_bank, selection_bytes)) = banked_selection {
                        library
                            .copy_d2d_async(
                                buffers.selected_indices,
                                selection_bank,
                                selection_bytes,
                                stream,
                            )
                            .context("restoring banked GLM DSA indices for shared layer")?;
                    }
                }
                if !run_selector
                    && (full_indexer || input.total_rows <= GLM_DSA_PREFILL_TOPK)
                {
                    if input.physical_page_table.is_some() {
                        library
                            .cuda_target_kv_page_table_expand_indices_async(
                                buffers.selected_indices,
                                buffers.page_table,
                                bucket_rows,
                                GLM_DSA_PREFILL_TOPK,
                                input.total_rows,
                                stream,
                            )
                            .context("expanding paged GLM DSA indices below top-k")?;
                    } else {
                        library
                            .cuda_glm_dsa_page_table_init_base_async(
                                buffers.selected_indices,
                                bucket_rows,
                                GLM_DSA_PREFILL_TOPK,
                                input.physical_token_base,
                                stream,
                            )
                        .context("populating sequential GLM DSA indices below top-k")?;
                    }
                }
            }
            if run_selector
                && glm_dsa_input_validate_min_rows()
                    .is_some_and(|min_rows| input.total_rows >= min_rows)
            {
                unsafe {
                    library
                        .cuda_stream_synchronize(stream)
                        .context("synchronizing GLM DSA staged inputs for validation")?;
                }
                validate_glm_dsa_debug_bf16(
                    library,
                    input.q_nope,
                    input.query_rows,
                    input.heads,
                    input.nope_dim,
                    "input q-nope",
                )?;
                validate_glm_dsa_debug_bf16(
                    library,
                    input.q_rope,
                    input.query_rows,
                    input.heads,
                    input.rope_dim,
                    "input q-rope",
                )?;
                validate_glm_dsa_debug_bf16(
                    library,
                    input.dsa_query.expect("validated full DSA query"),
                    input.query_rows,
                    GLMRT_CUDA_GLM_DSA_INDEX_HEADS,
                    GLM52_DSA_INDEX_HEAD_DIM,
                    "input projected index query",
                )?;
                validate_glm_dsa_debug_bf16(
                    library,
                    input.dsa_weights.expect("validated full DSA weights"),
                    input.query_rows,
                    GLMRT_CUDA_GLM_DSA_INDEX_HEADS,
                    1,
                    "input projected index weights",
                )?;
                if input.physical_page_table.is_none() {
                    validate_glm_dsa_debug_index_k(
                        library,
                        input.index_k_cache.expect("validated packed index-K cache"),
                        input.physical_token_base,
                        input.total_rows,
                    )?;
                    if input.kv_dtype == KvCacheDType::Fp8 {
                        validate_glm_dsa_debug_packed_kv(
                            library,
                            input.packed_kv,
                            input.physical_token_base,
                            input.total_rows,
                        )?;
                    }
                }
            }
            unsafe {
                if let Some(timeline) = cuda_timeline.as_ref() {
                    timeline.record(1, stream, "GLM DSA staged")?;
                }
            }

            let capture_identity = glm_dsa_sparse_mla_capture_identity(
                graph_input,
                buffers,
                full_indexer,
                run_selector,
            );
            ensure_glm_dsa_sparse_mla_prefill_graph(
                library,
                slot,
                graph_input,
                buffers,
                bucket_rows,
                max_pages,
                full_indexer,
                run_selector,
                capture_identity,
            )?;
            let sparse_topk =
                glm_dsa_sparse_mla_attention_topk(input.kv_dtype, bucket_rows, input.total_rows);
            let signature = CoordinatorCudaGraphSignature::glm_dsa_sparse_mla_prefill(
                bucket_rows,
                input.heads,
                input.rank,
                input.rope_dim,
                sparse_topk,
                max_pages,
                input.scale,
                run_selector,
            );
            let program = if full_indexer {
                CoordinatorCudaGraphProgram::LayerGlmDsaSparseMlaPrefillFull
            } else {
                CoordinatorCudaGraphProgram::LayerGlmDsaSparseMlaPrefillShared
            };
            slot.launch_captured_graph_identity(
                library,
                program,
                signature,
                capture_identity,
            )
            .with_context(|| {
                format!(
                    "launching GLM DSA sparse MLA layer={} source={} total={} query={} bucket={bucket_rows}",
                    input.layer_id, source_layer, input.total_rows, input.query_rows
                )
            })?;
            if source_has_shared_consumer && input.total_rows > GLM_DSA_PREFILL_TOPK {
                bank_glm_dsa_selection(
                    &mut workspace,
                    library,
                    selection_key,
                    buffers.selected_indices,
                    stream,
                )?;
            }
            unsafe {
                if let (Some(projection), Some(requested_output)) = (
                    graph_input.hidden_projection,
                    requested_hidden_projection_output,
                ) {
                    library
                        .copy_d2d_async(
                            requested_output,
                            projection.output,
                            input
                                .query_rows
                                .checked_mul(projection.hidden_dim)
                                .and_then(|values| {
                                    values.checked_mul(std::mem::size_of::<u16>())
                                })
                                .context(
                                    "GLM DSA active hidden-projection handoff bytes overflow",
                                )?,
                            stream,
                        )
                        .context("handing off fused GLM DSA hidden projection")?;
                }
                if let Some(timeline) = cuda_timeline.as_ref() {
                    timeline.record(2, stream, "GLM DSA graph")?;
                }
                library
                    .cuda_stream_synchronize(stream)
                    .with_context(|| {
                        format!(
                            "synchronizing GLM DSA sparse MLA prefill layer={} source={source_layer} total={} prefix={} query={}",
                            input.layer_id,
                            input.total_rows,
                            input.prefix_rows,
                            input.query_rows,
                        )
                    })?;
            }
            if let Some(timeline) = cuda_timeline.as_ref() {
                unsafe {
                    eprintln!(
                        "real_full_glm_dsa_sparse_mla_cuda_timing backend={} layer_id={} source_layer={} full_indexer={} selector={} total_rows={} query_rows={} bucket_rows={} topk={} staging_ms={:.3} graph_ms={:.3} total_ms={:.3}",
                        input.kv_dtype.label(),
                        input.layer_id,
                        source_layer,
                        full_indexer,
                        run_selector,
                        input.total_rows,
                        input.query_rows,
                        bucket_rows,
                        glm_dsa_sparse_mla_attention_topk(
                            input.kv_dtype,
                            bucket_rows,
                            input.total_rows,
                        ),
                        timeline.elapsed_ms(0, 1, "GLM DSA staging")?,
                        timeline.elapsed_ms(1, 2, "GLM DSA graph")?,
                        timeline.elapsed_ms(0, 2, "GLM DSA total")?,
                    );
                }
            }
            if glm_dsa_output_validate_enabled() {
                validate_glm_dsa_debug_bf16(
                    library,
                    buffers.combined_query,
                    input.query_rows,
                    input.heads,
                    input.rank + input.rope_dim,
                    "absorbed query",
                )?;
                if input.physical_page_table.is_none() {
                    validate_glm_dsa_debug_indices(
                        library,
                        buffers.selected_indices,
                        input.query_rows,
                        input.total_rows.min(GLM_DSA_PREFILL_TOPK),
                        input.physical_token_base,
                        input.total_rows,
                    )?;
                if input.kv_dtype == KvCacheDType::Fp8 {
                    validate_glm_dsa_debug_packed_kv(
                        library,
                        input.packed_kv,
                        input.physical_token_base,
                        input.total_rows,
                    )?;
                }
                }
                validate_glm_dsa_debug_bf16(
                    library,
                    buffers.sparse_latent,
                    input.query_rows,
                    input.heads,
                    input.rank,
                    "sparse latent output",
                )?;
                validate_glm_dsa_debug_bf16(
                    library,
                    buffers.final_output,
                    input.query_rows,
                    input.heads,
                    input.v_dim,
                    "expanded value output",
                )?;
            }
            if !source_has_shared_consumer
                && !full_indexer
                && glm_dsa_index_source_layer(input.layer_id + 1)
                    .is_none_or(|(next_source, _)| next_source != source_layer)
            {
                workspace.banked_selections.remove(&selection_key);
            }
            let output_bytes = input
                .query_rows
                .checked_mul(input.heads)
                .and_then(|values| values.checked_mul(input.v_dim))
                .and_then(|values| values.checked_mul(bf16_bytes))
                .context("GLM DSA sparse MLA output bytes overflow")?;
            Ok(FlashinferGlmDsaSparseMlaPrefillLaunch {
                backend: match input.kv_dtype {
                    KvCacheDType::Bf16 => "native-bf16-glm-dsa-sparse-mla",
                    KvCacheDType::Fp8 => FLASHINFER_GLM_DSA_SPARSE_MLA_PREFILL_BACKEND,
                    KvCacheDType::Nvfp4 => SPARKINFER_GLM_DSA_SPARSE_NVFP4_MLA_BACKEND,
                    _ => unreachable!("validated compressed GLM DSA KV dtype"),
                },
                output: device_buffer_byte_view(
                    buffers.final_output,
                    0,
                    output_bytes,
                    "GLM DSA sparse MLA active output",
                )?,
                hidden_projection_fused: input
                    .hidden_projection
                    .is_some_and(|projection| projection.w4a16.is_none()),
            })
        })
    })
}

fn glm_dsa_output_validate_enabled() -> bool {
    env::var(GLM_DSA_OUTPUT_VALIDATE_ENV)
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(false)
}

fn glm_dsa_input_validate_min_rows() -> Option<usize> {
    static MIN_ROWS: OnceLock<Option<usize>> = OnceLock::new();
    *MIN_ROWS.get_or_init(|| {
        env::var(GLM_DSA_INPUT_VALIDATE_MIN_ROWS_ENV)
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|rows| *rows > 0)
    })
}

fn validate_glm_dsa_debug_bf16(
    library: &NativeLibrary,
    buffer: GlmrtDeviceBuffer,
    rows: usize,
    heads: usize,
    width: usize,
    label: &str,
) -> Result<()> {
    let values = rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(width))
        .context("GLM DSA debug BF16 value count overflow")?;
    let bytes = values
        .checked_mul(std::mem::size_of::<u16>())
        .context("GLM DSA debug BF16 byte count overflow")?;
    let view = device_buffer_byte_view(buffer, 0, bytes, "GLM DSA debug BF16 view")?;
    let mut host = vec![0_u8; bytes];
    library
        .copy_d2h(&mut host, view)
        .with_context(|| format!("reading GLM DSA {label}"))?;
    let mut max_abs = 0.0_f32;
    for (index, chunk) in host.chunks_exact(2).enumerate() {
        let bits = u16::from_ne_bytes([chunk[0], chunk[1]]);
        if bits & 0x7f80 == 0x7f80 {
            let row_width = heads * width;
            anyhow::bail!(
                "GLM DSA {label} is non-finite at row={} head={} column={} bits=0x{bits:04x}",
                index / row_width,
                (index % row_width) / width,
                index % width,
            );
        }
        max_abs = max_abs.max(f32::from_bits((bits as u32) << 16).abs());
    }
    eprintln!(
        "real_full_glm_dsa_validation stage={} rows={} heads={} width={} max_abs={:.6e}",
        label, rows, heads, width, max_abs
    );
    Ok(())
}

fn validate_glm_dsa_debug_indices(
    library: &NativeLibrary,
    buffer: GlmrtDeviceBuffer,
    rows: usize,
    active_indices_per_row: usize,
    physical_token_base: usize,
    total_rows: usize,
) -> Result<()> {
    let values = rows
        .checked_mul(GLM_DSA_PREFILL_TOPK)
        .context("GLM DSA debug index count overflow")?;
    let bytes = values
        .checked_mul(std::mem::size_of::<i32>())
        .context("GLM DSA debug index bytes overflow")?;
    let view = device_buffer_byte_view(buffer, 0, bytes, "GLM DSA debug index view")?;
    let mut host = vec![0_u8; bytes];
    library
        .copy_d2h(&mut host, view)
        .context("reading GLM DSA selected indices")?;
    let physical_token_end = physical_token_base
        .checked_add(total_rows)
        .context("GLM DSA debug physical extent overflow")?;
    for row in 0..rows {
        for column in 0..active_indices_per_row {
            let index = row * GLM_DSA_PREFILL_TOPK + column;
            let byte = index * std::mem::size_of::<i32>();
            let slot = i32::from_ne_bytes(host[byte..byte + 4].try_into().expect("i32 width"));
            anyhow::ensure!(
                slot >= 0
                    && (slot as usize) >= physical_token_base
                    && (slot as usize) < physical_token_end,
                "GLM DSA selected slot is out of range at row={row} column={column}: slot={slot} physical_extent={physical_token_base}..{physical_token_end}"
            );
        }
    }
    Ok(())
}

fn validate_glm_dsa_debug_packed_kv(
    library: &NativeLibrary,
    buffer: GlmrtDeviceBuffer,
    physical_token_base: usize,
    rows: usize,
) -> Result<()> {
    let bytes = rows
        .checked_mul(FLASHINFER_PACKED_FP8_MLA_ROW_BYTES)
        .context("GLM DSA debug packed-KV bytes overflow")?;
    let offset = physical_token_base
        .checked_mul(FLASHINFER_PACKED_FP8_MLA_ROW_BYTES)
        .context("GLM DSA debug packed-KV offset overflow")?;
    let view = device_buffer_byte_view(buffer, offset, bytes, "GLM DSA debug packed-KV view")?;
    let mut host = vec![0_u8; bytes];
    library
        .copy_d2h(&mut host, view)
        .context("reading GLM DSA packed KV")?;
    let mut max_scale = 0.0_f32;
    for (row, payload) in host
        .chunks_exact(FLASHINFER_PACKED_FP8_MLA_ROW_BYTES)
        .enumerate()
    {
        for (column, code) in payload[..512].iter().copied().enumerate() {
            anyhow::ensure!(
                code & 0x7f != 0x7f,
                "GLM DSA packed KV contains FP8 NaN at row={row} column={column} code=0x{code:02x}"
            );
        }
        for group in 0..4 {
            let offset = 512 + group * std::mem::size_of::<f32>();
            let scale = f32::from_ne_bytes(
                payload[offset..offset + 4]
                    .try_into()
                    .expect("FP8 scale width"),
            );
            anyhow::ensure!(
                scale.is_finite() && scale > 0.0,
                "GLM DSA packed KV has invalid scale at row={row} group={group}: {scale}"
            );
            max_scale = max_scale.max(scale);
        }
        for (column, chunk) in payload[528..656].chunks_exact(2).enumerate() {
            let bits = u16::from_ne_bytes([chunk[0], chunk[1]]);
            anyhow::ensure!(
                bits & 0x7f80 != 0x7f80,
                "GLM DSA packed KV contains non-finite RoPE value at row={row} column={column} bits=0x{bits:04x}"
            );
        }
    }
    eprintln!(
        "real_full_glm_dsa_validation stage=packed_kv rows={} max_scale={:.6e}",
        rows, max_scale
    );
    Ok(())
}

fn validate_glm_dsa_debug_index_k(
    library: &NativeLibrary,
    buffer: GlmrtDeviceBuffer,
    physical_token_base: usize,
    rows: usize,
) -> Result<()> {
    anyhow::ensure!(
        physical_token_base % GLMRT_CUDA_GLM_DSA_PAGE_SIZE == 0,
        "GLM DSA debug index-K physical base must be page aligned"
    );
    let pages = rows
        .checked_add(GLMRT_CUDA_GLM_DSA_PAGE_SIZE - 1)
        .context("GLM DSA debug index-K page count overflow")?
        / GLMRT_CUDA_GLM_DSA_PAGE_SIZE;
    let bytes = pages
        .checked_mul(GLMRT_CUDA_GLM_DSA_PACKED_PAGE_BYTES)
        .context("GLM DSA debug index-K bytes overflow")?;
    let offset = (physical_token_base / GLMRT_CUDA_GLM_DSA_PAGE_SIZE)
        .checked_mul(GLMRT_CUDA_GLM_DSA_PACKED_PAGE_BYTES)
        .context("GLM DSA debug index-K offset overflow")?;
    let view = device_buffer_byte_view(buffer, offset, bytes, "GLM DSA debug index-K view")?;
    let mut host = vec![0_u8; bytes];
    library
        .copy_d2h(&mut host, view)
        .context("reading GLM DSA packed index-K cache")?;
    let mut max_scale = 0.0_f32;
    for physical_row in 0..rows {
        let page = physical_row / GLMRT_CUDA_GLM_DSA_PAGE_SIZE;
        let page_row = physical_row % GLMRT_CUDA_GLM_DSA_PAGE_SIZE;
        let page_offset = page * GLMRT_CUDA_GLM_DSA_PACKED_PAGE_BYTES;
        let quant_offset = page_offset + page_row * GLM52_DSA_INDEX_HEAD_DIM;
        for (column, code) in host[quant_offset..quant_offset + GLM52_DSA_INDEX_HEAD_DIM]
            .iter()
            .copied()
            .enumerate()
        {
            anyhow::ensure!(
                code & 0x7f != 0x7f,
                "GLM DSA packed index-K contains FP8 NaN at row={physical_row} column={column} code=0x{code:02x}"
            );
        }
        let scale_offset = page_offset
            + GLMRT_CUDA_GLM_DSA_PAGE_SIZE * GLM52_DSA_INDEX_HEAD_DIM
            + page_row * std::mem::size_of::<f32>();
        let scale = f32::from_ne_bytes(
            host[scale_offset..scale_offset + 4]
                .try_into()
                .expect("index-K scale width"),
        );
        anyhow::ensure!(
            scale.is_finite() && scale > 0.0,
            "GLM DSA packed index-K has invalid scale at row={physical_row}: {scale}"
        );
        max_scale = max_scale.max(scale);
    }
    eprintln!(
        "real_full_glm_dsa_validation stage=index_k rows={} max_scale={:.6e}",
        rows, max_scale
    );
    Ok(())
}

fn glm_dsa_sparse_mla_capture_identity(
    input: FlashinferGlmDsaSparseMlaPrefillInput,
    buffers: GlmDsaSparseMlaPrefillBuffers,
    full_indexer: bool,
    run_selector: bool,
) -> usize {
    mla_graph_capture_identity(&[
        input.packed_kv.ptr as usize,
        if run_selector {
            input.index_k_cache.map_or(0, |buffer| buffer.ptr as usize)
        } else {
            0
        },
        input.kv_b_weight.ptr as usize,
        input
            .hidden_projection
            .map_or(0, |projection| projection.weight.ptr as usize),
        input
            .hidden_projection
            .map_or(0, |projection| projection.output.ptr as usize),
        input
            .hidden_projection
            .and_then(|projection| projection.w8a16)
            .map_or(0, |projection| projection.weight.ptr as usize),
        input
            .hidden_projection
            .and_then(|projection| projection.w8a16)
            .map_or(0, |projection| projection.scales.ptr as usize),
        input
            .hidden_projection
            .is_some_and(|projection| projection.w4a16.is_none()) as usize,
        buffers.selector_scratch.ptr as usize,
        buffers.sparse_mid_out.ptr as usize,
        buffers.sparse_mid_lse.ptr as usize,
        buffers.page_table.ptr as usize,
        buffers.cache_seqlens.ptr as usize,
        buffers.active_width.ptr as usize,
        buffers.selected_indices.ptr as usize,
        buffers.topk_lengths.ptr as usize,
        buffers.compacted_indices.ptr as usize,
        buffers.dsa_query_raw.ptr as usize,
        buffers.dsa_weights_raw.ptr as usize,
        buffers.dsa_positions.ptr as usize,
        buffers.dsa_query_fp8.ptr as usize,
        buffers.dsa_weights.ptr as usize,
        buffers.q_nope.ptr as usize,
        buffers.q_rope.ptr as usize,
        buffers.combined_query.ptr as usize,
        buffers.head_major.ptr as usize,
        buffers.sparse_latent.ptr as usize,
        buffers.auxiliary.ptr as usize,
        buffers.final_output.ptr as usize,
        buffers.out_lse.ptr as usize,
        match input.kv_dtype {
            KvCacheDType::Fp8 => 1,
            KvCacheDType::Nvfp4 => 2,
            KvCacheDType::Bf16 => 3,
            _ => 0,
        },
        input.kv_row_stride_bytes,
        usize::from(full_indexer),
        usize::from(run_selector),
    ])
}

#[allow(clippy::too_many_arguments)]
fn ensure_glm_dsa_sparse_mla_prefill_graph(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    input: FlashinferGlmDsaSparseMlaPrefillInput,
    buffers: GlmDsaSparseMlaPrefillBuffers,
    bucket_rows: usize,
    max_pages: usize,
    full_indexer: bool,
    run_selector: bool,
    capture_identity: usize,
) -> Result<()> {
    let sparse_topk =
        glm_dsa_sparse_mla_attention_topk(input.kv_dtype, bucket_rows, input.total_rows);
    let signature = CoordinatorCudaGraphSignature::glm_dsa_sparse_mla_prefill(
        bucket_rows,
        input.heads,
        input.rank,
        input.rope_dim,
        sparse_topk,
        max_pages,
        input.scale,
        run_selector,
    );
    let program = if full_indexer {
        CoordinatorCudaGraphProgram::LayerGlmDsaSparseMlaPrefillFull
    } else {
        CoordinatorCudaGraphProgram::LayerGlmDsaSparseMlaPrefillShared
    };
    if slot.has_captured_graph_identity(program, signature, capture_identity) {
        return Ok(());
    }
    if !coordinator_python_capture_startup_open() {
        let retained_identity_count = slot
            .captured_graphs
            .iter()
            .filter(|entry| entry.program == program && entry.signature == signature)
            .count();
        anyhow::bail!(
            "GLM DSA sparse MLA graph shape={} row_bucket={} layer={} query_bucket={bucket_rows} identity={capture_identity} was not captured during startup; retained_identity_count={retained_identity_count} packed_kv_ptr={:#x} index_k_ptr={:#x} kv_b_ptr={:#x}",
            slot.plan.key.shape.label(),
            slot.plan.key.row_bucket.row_capacity,
            input.layer_id,
            input.packed_kv.ptr as usize,
            input.index_k_cache.map_or(0, |buffer| buffer.ptr as usize),
            input.kv_b_weight.ptr as usize,
        );
    }

    let selector_buffers = run_selector.then(|| {
        glm_dsa_selector_python_buffers(
            buffers,
            input
                .index_k_cache
                .expect("validated full DSA selector index-K cache"),
            buffers.page_table,
        )
    });
    let compact_nvfp4_indices =
        input.kv_dtype == KvCacheDType::Nvfp4 && sparse_topk < GLM_DSA_PREFILL_TOPK;
    let sparse_indices = if compact_nvfp4_indices {
        buffers.compacted_indices
    } else {
        buffers.selected_indices
    };
    let sparse_buffers =
        glm_dsa_sparse_mla_python_buffers(buffers, input.packed_kv, sparse_indices);
    let selector_kwargs = [
        ("query_rows", PythonKernelArg::Usize(bucket_rows)),
        ("page_table_width", PythonKernelArg::Usize(max_pages)),
        ("cache_pages", PythonKernelArg::Usize(max_pages)),
        ("topk", PythonKernelArg::Usize(GLM_DSA_PREFILL_TOPK)),
        (
            "supertile_k",
            PythonKernelArg::Usize(GLM_DSA_PREFILL_SUPERTILE_K),
        ),
        ("shared_page_table", PythonKernelArg::Bool(true)),
    ];
    let sparse_kwargs = [
        ("query_rows", PythonKernelArg::Usize(bucket_rows)),
        ("kv_pages", PythonKernelArg::Usize(max_pages)),
        ("topk", PythonKernelArg::Usize(sparse_topk)),
        ("heads", PythonKernelArg::Usize(input.heads)),
        ("nope_dim", PythonKernelArg::Usize(input.rank)),
        ("rope_dim", PythonKernelArg::Usize(input.rope_dim)),
        ("scale", PythonKernelArg::F64(input.scale as f64)),
    ];
    let stream = slot.stream_ptr();
    unsafe {
        enqueue_glm_dsa_sparse_mla_query(
            library,
            stream,
            input,
            buffers,
            bucket_rows,
            run_selector,
        )?;
    }
    slot.stream_synchronize()
        .context("synchronizing GLM DSA sparse MLA query before Python warmup")?;
    if let Some(selector_buffers) = selector_buffers.as_ref() {
        for function in [
            B12X_GLM_DSA_PREFILL_PREPARE_FUNCTION,
            B12X_GLM_DSA_PREFILL_CAPTURE_FUNCTION,
        ] {
            launch_python_graph_capture(PythonGraphCaptureLaunch {
                module: B12X_MLA_CAPTURE_MODULE,
                function,
                cuda_stream: stream,
                buffers: selector_buffers,
                kwargs: &selector_kwargs,
            })
            .with_context(|| {
                format!(
                    "warming GLM DSA selector layer={} query_bucket={bucket_rows}",
                    input.layer_id
                )
            })?;
        }
    }
    if compact_nvfp4_indices {
        unsafe {
            library
                .cuda_copy_row_prefix_bf16_async(
                    buffers.selected_indices,
                    bucket_rows,
                    buffers.compacted_indices,
                    bucket_rows,
                    GLM_DSA_PREFILL_TOPK * 2,
                    sparse_topk * 2,
                    sparse_topk * 2,
                    0,
                    stream,
                )
                .context("compacting native NVFP4 sparse-MLA indices for warmup")?;
        }
    }
    let sparse_python_functions = match input.kv_dtype {
        KvCacheDType::Bf16 => None,
        dtype => Some(match dtype {
            KvCacheDType::Fp8 => (
                FLASHINFER_PACKED_FP8_MLA_PREFILL_PREPARE_FUNCTION,
                FLASHINFER_PACKED_FP8_MLA_PREFILL_CAPTURE_FUNCTION,
                "packed-FP8",
            ),
            KvCacheDType::Nvfp4 if bucket_rows <= 16 => (
                SPARKINFER_NVFP4_MLA_DECODE_PREPARE_FUNCTION,
                SPARKINFER_NVFP4_MLA_DECODE_CAPTURE_FUNCTION,
                "native packed-NVFP4 decode",
            ),
            KvCacheDType::Nvfp4 => (
                SPARKINFER_NVFP4_MLA_PREFILL_PREPARE_FUNCTION,
                SPARKINFER_NVFP4_MLA_PREFILL_CAPTURE_FUNCTION,
                "native packed-NVFP4 prefill",
            ),
            _ => unreachable!("validated GLM DSA sparse MLA KV dtype"),
        }),
    };
    if input.kv_dtype == KvCacheDType::Bf16 {
        enqueue_glm_dsa_sparse_mla_bf16_attention(
            library,
            stream,
            input,
            buffers,
            sparse_indices,
            bucket_rows,
            sparse_topk,
        )
        .context("warming native BF16 sparse MLA attention")?;
    } else {
        let (sparse_prepare_function, sparse_capture_function, sparse_backend_label) =
            sparse_python_functions.expect("non-BF16 sparse MLA has Python functions");
        for function in [sparse_prepare_function, sparse_capture_function] {
            launch_python_graph_capture(PythonGraphCaptureLaunch {
                module: B12X_MLA_CAPTURE_MODULE,
                function,
                cuda_stream: stream,
                buffers: &sparse_buffers,
                kwargs: &sparse_kwargs,
            })
            .with_context(|| {
                format!(
                    "warming sparse {sparse_backend_label} MLA layer={} query_bucket={bucket_rows}",
                    input.layer_id
                )
            })?;
        }
    }
    unsafe {
        enqueue_glm_dsa_sparse_mla_output(library, stream, input, buffers, bucket_rows)?;
    }
    slot.stream_synchronize()
        .context("synchronizing warmed GLM DSA sparse MLA graph")?;

    slot.capture_or_update_graph_exec(
        library,
        program,
        signature,
        capture_identity,
        |library, cuda_stream, _workspace| {
            unsafe {
                enqueue_glm_dsa_sparse_mla_query(
                    library,
                    cuda_stream,
                    input,
                    buffers,
                    bucket_rows,
                    run_selector,
                )?;
            }
            if let Some(selector_buffers) = selector_buffers.as_ref() {
                launch_python_graph_capture(PythonGraphCaptureLaunch {
                    module: B12X_MLA_CAPTURE_MODULE,
                    function: B12X_GLM_DSA_PREFILL_CAPTURE_FUNCTION,
                    cuda_stream,
                    buffers: selector_buffers,
                    kwargs: &selector_kwargs,
                })
                .context("capturing direct packed GLM DSA selector")?;
            }
            if compact_nvfp4_indices {
                unsafe {
                    library
                        .cuda_copy_row_prefix_bf16_async(
                            buffers.selected_indices,
                            bucket_rows,
                            buffers.compacted_indices,
                            bucket_rows,
                            GLM_DSA_PREFILL_TOPK * 2,
                            sparse_topk * 2,
                            sparse_topk * 2,
                            0,
                            cuda_stream,
                        )
                        .context("capturing native NVFP4 sparse-MLA index compaction")?;
                }
            }
            if input.kv_dtype == KvCacheDType::Bf16 {
                enqueue_glm_dsa_sparse_mla_bf16_attention(
                    library,
                    cuda_stream,
                    input,
                    buffers,
                    sparse_indices,
                    bucket_rows,
                    sparse_topk,
                )
                .context("capturing native BF16 sparse MLA attention")?;
            } else {
                let (_, sparse_capture_function, sparse_backend_label) =
                    sparse_python_functions.expect("non-BF16 sparse MLA has Python functions");
                launch_python_graph_capture(PythonGraphCaptureLaunch {
                    module: B12X_MLA_CAPTURE_MODULE,
                    function: sparse_capture_function,
                    cuda_stream,
                    buffers: &sparse_buffers,
                    kwargs: &sparse_kwargs,
                })
                .with_context(|| format!("capturing direct sparse {sparse_backend_label} MLA"))?;
            }
            unsafe {
                enqueue_glm_dsa_sparse_mla_output(
                    library,
                    cuda_stream,
                    input,
                    buffers,
                    bucket_rows,
                )?;
            }
            Ok(())
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn enqueue_glm_dsa_sparse_mla_bf16_attention(
    library: &'static NativeLibrary,
    stream: *mut c_void,
    input: FlashinferGlmDsaSparseMlaPrefillInput,
    buffers: GlmDsaSparseMlaPrefillBuffers,
    sparse_indices: GlmrtDeviceBuffer,
    query_rows: usize,
    sparse_topk: usize,
) -> Result<()> {
    if query_rows <= 16 {
        unsafe {
            library.cuda_sparse_mla_bf16_async(
                buffers.combined_query,
                input.packed_kv,
                sparse_indices,
                buffers.topk_lengths,
                buffers.sparse_mid_out,
                buffers.sparse_mid_lse,
                buffers.sparse_latent,
                buffers.out_lse,
                query_rows,
                input.heads,
                sparse_topk,
                input.kv_row_stride_bytes,
                input.scale,
                stream,
            )
        }
        .context("launching low-row native BF16 sparse MLA attention")?;
        return Ok(());
    }

    anyhow::ensure!(
        input.heads == 64 && input.rank == 512 && input.rope_dim == 64 && sparse_topk == 2_048,
        "large-query BF16 sparse MLA requires heads=64 rank=512 rope_dim=64 topk=2048"
    );
    anyhow::ensure!(
        GLM_DSA_BF16_GEMM_SCRATCH_BYTES <= buffers.selector_scratch.bytes,
        "large-query BF16 sparse MLA scratch requires {} bytes, only {} are available",
        GLM_DSA_BF16_GEMM_SCRATCH_BYTES,
        buffers.selector_scratch.bytes
    );
    let bf16_bytes = std::mem::size_of::<u16>();
    let i32_bytes = std::mem::size_of::<i32>();
    let f32_bytes = std::mem::size_of::<f32>();
    let gathered_k = device_buffer_byte_view(
        buffers.selector_scratch,
        0,
        GLM_DSA_BF16_GEMM_K_BYTES,
        "large-query BF16 sparse MLA gathered K",
    )?;
    let gathered_v = device_buffer_byte_view(
        buffers.selector_scratch,
        GLM_DSA_BF16_GEMM_K_BYTES,
        GLM_DSA_BF16_GEMM_V_BYTES,
        "large-query BF16 sparse MLA gathered V",
    )?;
    let scores = device_buffer_byte_view(
        buffers.selector_scratch,
        GLM_DSA_BF16_GEMM_K_BYTES + GLM_DSA_BF16_GEMM_V_BYTES,
        GLM_DSA_BF16_GEMM_SCORE_BYTES,
        "large-query BF16 sparse MLA scores",
    )?;

    for query_start in (0..query_rows).step_by(GLM_DSA_BF16_GEMM_QUERY_BATCH_ROWS) {
        let batch_rows = (query_rows - query_start).min(GLM_DSA_BF16_GEMM_QUERY_BATCH_ROWS);
        let selected_offset = query_start
            .checked_mul(sparse_topk)
            .and_then(|values| values.checked_mul(i32_bytes))
            .context("large-query BF16 sparse MLA selected-index offset overflow")?;
        let selected_bytes = batch_rows
            .checked_mul(sparse_topk)
            .and_then(|values| values.checked_mul(i32_bytes))
            .context("large-query BF16 sparse MLA selected-index bytes overflow")?;
        let selected = device_buffer_byte_view(
            sparse_indices,
            selected_offset,
            selected_bytes,
            "large-query BF16 sparse MLA selected-index batch",
        )?;
        let lengths = device_buffer_byte_view(
            buffers.topk_lengths,
            query_start
                .checked_mul(i32_bytes)
                .context("large-query BF16 sparse MLA length offset overflow")?,
            batch_rows
                .checked_mul(i32_bytes)
                .context("large-query BF16 sparse MLA length bytes overflow")?,
            "large-query BF16 sparse MLA length batch",
        )?;
        let query = device_buffer_byte_view(
            buffers.combined_query,
            query_start
                .checked_mul(input.heads)
                .and_then(|values| values.checked_mul(input.rank + input.rope_dim))
                .and_then(|values| values.checked_mul(bf16_bytes))
                .context("large-query BF16 sparse MLA query offset overflow")?,
            batch_rows
                .checked_mul(input.heads)
                .and_then(|values| values.checked_mul(input.rank + input.rope_dim))
                .and_then(|values| values.checked_mul(bf16_bytes))
                .context("large-query BF16 sparse MLA query bytes overflow")?,
            "large-query BF16 sparse MLA query batch",
        )?;
        let output = device_buffer_byte_view(
            buffers.sparse_latent,
            query_start
                .checked_mul(input.heads)
                .and_then(|values| values.checked_mul(input.rank))
                .and_then(|values| values.checked_mul(bf16_bytes))
                .context("large-query BF16 sparse MLA output offset overflow")?,
            batch_rows
                .checked_mul(input.heads)
                .and_then(|values| values.checked_mul(input.rank))
                .and_then(|values| values.checked_mul(bf16_bytes))
                .context("large-query BF16 sparse MLA output bytes overflow")?,
            "large-query BF16 sparse MLA output batch",
        )?;
        let output_lse = device_buffer_byte_view(
            buffers.out_lse,
            query_start
                .checked_mul(input.heads)
                .and_then(|values| values.checked_mul(f32_bytes))
                .context("large-query BF16 sparse MLA LSE offset overflow")?,
            batch_rows
                .checked_mul(input.heads)
                .and_then(|values| values.checked_mul(f32_bytes))
                .context("large-query BF16 sparse MLA LSE bytes overflow")?,
            "large-query BF16 sparse MLA LSE batch",
        )?;
        unsafe {
            library
                .cuda_sparse_mla_bf16_gather_kv_async(
                    input.packed_kv,
                    selected,
                    lengths,
                    gathered_k,
                    gathered_v,
                    batch_rows,
                    sparse_topk,
                    input.kv_row_stride_bytes,
                    stream,
                )
                .context("gathering selected BF16 sparse MLA K/V rows")?;
            library
                .cuda_linear_bf16_strided_batched_cublas_async(
                    query,
                    gathered_k,
                    scores,
                    batch_rows,
                    input.heads,
                    input.rank + input.rope_dim,
                    sparse_topk,
                    input.heads * (input.rank + input.rope_dim),
                    sparse_topk * (input.rank + input.rope_dim),
                    input.heads * sparse_topk,
                    stream,
                )
                .context("multiplying large-query BF16 sparse MLA QK")?;
            library
                .cuda_sparse_mla_bf16_softmax_async(
                    scores,
                    lengths,
                    output_lse,
                    batch_rows,
                    input.heads,
                    sparse_topk,
                    input.scale,
                    stream,
                )
                .context("normalizing large-query BF16 sparse MLA scores")?;
            library
                .cuda_matmul_bf16_strided_batched_cublas_async(
                    scores,
                    gathered_v,
                    output,
                    batch_rows,
                    input.heads,
                    sparse_topk,
                    input.rank,
                    input.heads * sparse_topk,
                    sparse_topk * input.rank,
                    input.heads * input.rank,
                    stream,
                )
                .context("multiplying large-query BF16 sparse MLA probabilities by V")?;
        }
    }
    Ok(())
}

unsafe fn enqueue_glm_dsa_sparse_mla_query(
    library: &'static NativeLibrary,
    stream: *mut c_void,
    input: FlashinferGlmDsaSparseMlaPrefillInput,
    buffers: GlmDsaSparseMlaPrefillBuffers,
    bucket_rows: usize,
    prepare_dsa_query: bool,
) -> Result<()> {
    let bf16_bytes = std::mem::size_of::<u16>();
    let kv_b_head_width = input.nope_dim + input.v_dim;
    let weight_head_stride = kv_b_head_width * input.rank;
    if prepare_dsa_query {
        library
            .cuda_glm_dsa_query_prepare_b12x_async(
                buffers.dsa_query_raw,
                buffers.dsa_weights_raw,
                buffers.dsa_positions,
                buffers.dsa_query_fp8,
                buffers.dsa_weights,
                bucket_rows,
                GLMRT_CUDA_GLM_DSA_INDEX_HEADS * GLM52_DSA_INDEX_HEAD_DIM * bf16_bytes,
                GLMRT_CUDA_GLM_DSA_INDEX_HEADS * bf16_bytes,
                GLMRT_CUDA_GLM_DSA_INDEX_HEADS * GLM52_DSA_INDEX_HEAD_DIM,
                GLMRT_CUDA_GLM_DSA_INDEX_HEADS * std::mem::size_of::<f32>(),
                input.theta,
                GLM_DSA_PREFILL_SCORE_SCALE,
                stream,
            )
            .context("preparing direct packed GLM DSA queries and head weights")?;
    }
    library
        .cuda_transpose_rows_heads_bf16_async(
            buffers.q_nope,
            buffers.auxiliary,
            bucket_rows,
            input.heads,
            input.nope_dim,
            stream,
        )
        .context("transposing sparse MLA q-nope to head-major layout")?;
    library
        .cuda_matmul_bf16_strided_batched_cublas_async(
            buffers.auxiliary,
            input.kv_b_weight,
            buffers.head_major,
            input.heads,
            bucket_rows,
            input.nope_dim,
            input.rank,
            bucket_rows * input.nope_dim,
            weight_head_stride,
            bucket_rows * input.rank,
            stream,
        )
        .context("absorbing sparse MLA q-nope into latent rank")?;
    library
        .cuda_mla_compose_absorbed_query_bf16_async(
            buffers.head_major,
            buffers.q_rope,
            buffers.combined_query,
            bucket_rows,
            input.heads,
            input.rank,
            input.rope_dim,
            stream,
        )
        .context("composing absorbed sparse MLA query with RoPE suffix")
}

unsafe fn enqueue_glm_dsa_sparse_mla_output(
    library: &'static NativeLibrary,
    stream: *mut c_void,
    input: FlashinferGlmDsaSparseMlaPrefillInput,
    buffers: GlmDsaSparseMlaPrefillBuffers,
    bucket_rows: usize,
) -> Result<()> {
    let bf16_bytes = std::mem::size_of::<u16>();
    let kv_b_head_width = input.nope_dim + input.v_dim;
    let weight_head_stride = kv_b_head_width * input.rank;
    let value_weight_offset = input
        .nope_dim
        .checked_mul(input.rank)
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("sparse MLA value-weight offset overflow")?;
    let value_weight = device_buffer_byte_view(
        input.kv_b_weight,
        value_weight_offset,
        input
            .kv_b_weight
            .bytes
            .checked_sub(value_weight_offset)
            .context("sparse MLA value-weight view exceeds KV-B weight")?,
        "sparse MLA value weights",
    )?;
    library
        .cuda_transpose_rows_heads_bf16_async(
            buffers.sparse_latent,
            buffers.head_major,
            bucket_rows,
            input.heads,
            input.rank,
            stream,
        )
        .context("transposing sparse MLA latent output to head-major layout")?;
    library
        .cuda_linear_bf16_strided_batched_cublas_async(
            buffers.head_major,
            value_weight,
            buffers.auxiliary,
            input.heads,
            bucket_rows,
            input.rank,
            input.v_dim,
            bucket_rows * input.rank,
            weight_head_stride,
            bucket_rows * input.v_dim,
            stream,
        )
        .context("expanding sparse MLA latent output values")?;
    library
        .cuda_transpose_heads_rows_bf16_async(
            buffers.auxiliary,
            buffers.final_output,
            bucket_rows,
            input.heads,
            input.v_dim,
            stream,
        )
        .context("restoring sparse MLA output to query-major layout")?;
    enqueue_glm_dsa_sparse_mla_hidden_projection(
        library,
        stream,
        buffers.final_output,
        input.hidden_projection,
        bucket_rows,
        input.heads * input.v_dim,
    )
}

unsafe fn enqueue_glm_dsa_sparse_mla_hidden_projection(
    library: &'static NativeLibrary,
    stream: *mut c_void,
    attention_output: GlmrtDeviceBuffer,
    projection: Option<FlashinferMlaHiddenProjection>,
    rows: usize,
    input_dim: usize,
) -> Result<()> {
    let Some(projection) = projection else {
        return Ok(());
    };
    // The legacy W4 projection owns a distinct launch-buffer contract. Keep
    // that path outside this graph until it can consume the direct sparse
    // output without staging. Long mode uses the packed W8 path below.
    if projection.w4a16.is_some() {
        return Ok(());
    }
    if let Some(w8a16) = projection.w8a16 {
        if w8a16.packed_layout && rows == 1 {
            library
                .cuda_linear_w8a16_group256_m1_warp_packed_async(
                    attention_output,
                    w8a16.weight,
                    w8a16.scales,
                    projection.output,
                    input_dim,
                    projection.hidden_dim,
                    stream,
                )
                .context("fusing direct sparse MLA output with packed recurrent W8A16 O")?;
        } else if w8a16.packed_layout && rows >= 4 {
            library
                .cuda_linear_w8a16_group256_m1_warp_packed_parity_batched_async(
                    attention_output,
                    w8a16.weight,
                    w8a16.scales,
                    projection.output,
                    rows,
                    input_dim,
                    projection.hidden_dim,
                    stream,
                )
                .context("fusing direct sparse MLA output with packed parity W8A16 O")?;
        } else if w8a16.packed_layout {
            let input_row_bytes = input_dim * std::mem::size_of::<u16>();
            let output_row_bytes = projection.hidden_dim * std::mem::size_of::<u16>();
            for row in 0..rows {
                library
                    .cuda_linear_w8a16_group256_m1_warp_packed_async(
                        device_buffer_byte_view(
                            attention_output,
                            row * input_row_bytes,
                            input_row_bytes,
                            "direct sparse MLA packed W8A16 O input row",
                        )?,
                        w8a16.weight,
                        w8a16.scales,
                        device_buffer_byte_view(
                            projection.output,
                            row * output_row_bytes,
                            output_row_bytes,
                            "direct sparse MLA packed W8A16 O output row",
                        )?,
                        input_dim,
                        projection.hidden_dim,
                        stream,
                    )
                    .context("fusing direct sparse MLA output with packed recurrent W8A16 O")?;
            }
        } else if rows == 1 {
            library
                .cuda_linear_w8a16_group256_m1_simt_async(
                    attention_output,
                    w8a16.weight,
                    w8a16.scales,
                    projection.output,
                    input_dim,
                    projection.hidden_dim,
                    3,
                    stream,
                )
                .context("fusing direct sparse MLA output with recurrent W8A16 O")?;
        } else {
            library
                .cuda_linear_w8a16_group256_m1_parity_batched_async(
                    attention_output,
                    w8a16.weight,
                    w8a16.scales,
                    projection.output,
                    rows,
                    input_dim,
                    projection.hidden_dim,
                    stream,
                )
                .context("fusing direct sparse MLA output with parity W8A16 O")?;
        }
    } else {
        library
            .cuda_linear_bf16_cublas_async(
                attention_output,
                projection.weight,
                None,
                projection.output,
                rows,
                input_dim,
                projection.hidden_dim,
                stream,
            )
            .context("fusing direct sparse MLA output with BF16 O")?;
    }
    Ok(())
}

fn glm_dsa_selector_python_buffers(
    buffers: GlmDsaSparseMlaPrefillBuffers,
    index_k_cache: GlmrtDeviceBuffer,
    page_table: GlmrtDeviceBuffer,
) -> [PythonDeviceBufferArg<'static>; 8] {
    [
        python_device_buffer_arg("q_fp8", buffers.dsa_query_fp8),
        python_device_buffer_arg("weights", buffers.dsa_weights),
        python_device_buffer_arg("index_k_cache", index_k_cache),
        python_device_buffer_arg("page_table", page_table),
        python_device_buffer_arg("cache_seqlens", buffers.cache_seqlens),
        python_device_buffer_arg("active_width", buffers.active_width),
        python_device_buffer_arg("output_indices", buffers.selected_indices),
        python_device_buffer_arg("scratch", buffers.selector_scratch),
    ]
}

fn glm_dsa_sparse_mla_python_buffers(
    buffers: GlmDsaSparseMlaPrefillBuffers,
    packed_kv: GlmrtDeviceBuffer,
    selected_indices: GlmrtDeviceBuffer,
) -> [PythonDeviceBufferArg<'static>; 8] {
    [
        python_device_buffer_arg("q", buffers.combined_query),
        python_device_buffer_arg("kv", packed_kv),
        python_device_buffer_arg("indices", selected_indices),
        python_device_buffer_arg("topk_length", buffers.topk_lengths),
        python_device_buffer_arg("output", buffers.sparse_latent),
        python_device_buffer_arg("out_lse", buffers.out_lse),
        python_device_buffer_arg("mid_out", buffers.sparse_mid_out),
        python_device_buffer_arg("mid_lse", buffers.sparse_mid_lse),
    ]
}

fn stage_flashinfer_hidden_projection(
    projection: Option<FlashinferMlaHiddenProjection>,
    staging_output: GlmrtDeviceBuffer,
    direct_output: bool,
) -> (
    Option<GlmrtDeviceBuffer>,
    Option<FlashinferMlaHiddenProjection>,
) {
    if direct_output {
        return (None, projection);
    }
    let external_output = projection.map(|projection| projection.output);
    let staged_projection = projection.map(|projection| FlashinferMlaHiddenProjection {
        output: staging_output,
        ..projection
    });
    (external_output, staged_projection)
}

fn ensure_flashinfer_packed_fp8_mla_decode_graphs(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    buffers: FlashinferPackedFp8MlaFullGraphBuffers,
    geometry: FlashinferPackedFp8MlaFullGraphGeometry,
    scale: f32,
    capture_rows: usize,
    initialize_kv: bool,
    capture_index_base: usize,
    capture_identity: usize,
) -> Result<()> {
    let query_rows = geometry.query_rows;
    let heads = geometry.heads;
    let rank = geometry.rank;
    let rope_dim = geometry.rope_dim;
    for bucket_rows in FLASHINFER_PACKED_FP8_MLA_BUCKETS {
        let signature = CoordinatorCudaGraphSignature::flashinfer_packed_fp8_mla_decode(
            bucket_rows,
            query_rows,
            heads,
            rank,
            rope_dim,
            scale,
            if buffers.hidden_projection_w4a16.is_some() {
                2
            } else if buffers
                .hidden_projection
                .is_some_and(|projection| projection.w8a16.is_some())
            {
                3
            } else {
                usize::from(buffers.hidden_projection.is_some())
            },
        );
        let program = CoordinatorCudaGraphProgram::LayerFlashinferPackedFp8MlaDecode;
        if slot.has_captured_graph_identity(program, signature, capture_identity) {
            continue;
        }
        anyhow::ensure!(
            coordinator_python_capture_startup_open(),
            "FlashInfer packed FP8 MLA decode graph bucket={bucket_rows} was not captured during startup"
        );
        let kv_capacity_rows = if initialize_kv {
            bucket_rows
        } else {
            buffers.flashinfer.kv.bytes / FLASHINFER_PACKED_FP8_MLA_ROW_BYTES
        };
        let kwargs = [
            ("bucket_rows", PythonKernelArg::Usize(bucket_rows)),
            ("kv_capacity_rows", PythonKernelArg::Usize(kv_capacity_rows)),
            ("query_rows", PythonKernelArg::Usize(query_rows)),
            ("heads", PythonKernelArg::Usize(heads)),
            ("nope_dim", PythonKernelArg::Usize(rank)),
            ("rope_dim", PythonKernelArg::Usize(rope_dim)),
            ("scale", PythonKernelArg::F64(scale as f64)),
            ("initialize_kv", PythonKernelArg::Bool(initialize_kv)),
        ];
        let capture_length = i32::try_from(if initialize_kv {
            bucket_rows
        } else {
            capture_rows.min(bucket_rows)
        })
        .context("packed FP8 MLA capture bucket exceeds i32")?
        .to_ne_bytes();
        let capture_index_base = i32::try_from(capture_index_base)
            .context("packed FP8 MLA capture physical index base exceeds i32")?
            .to_ne_bytes();
        let capture_index_base_offset =
            FLASHINFER_PACKED_FP8_MLA_MAX_QUERY_ROWS * std::mem::size_of::<i32>();
        let mut capture_header =
            [0_u8; FLASHINFER_PACKED_FP8_MLA_MAX_QUERY_ROWS * std::mem::size_of::<i32>() * 2];
        for query_index in 0..query_rows {
            let byte_offset = query_index * std::mem::size_of::<i32>();
            capture_header[byte_offset..byte_offset + capture_length.len()]
                .copy_from_slice(&capture_length);
            let base_offset = capture_index_base_offset + byte_offset;
            capture_header[base_offset..base_offset + capture_index_base.len()]
                .copy_from_slice(&capture_index_base);
        }
        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::P,
                &capture_header,
                "FlashInfer packed FP8 MLA capture metadata header",
                slot.stream_ptr(),
            )
            .with_context(|| {
                format!(
                    "staging packed FP8 MLA capture metadata for bucket={bucket_rows} query_rows={query_rows}"
                )
            })?;
        slot.stream_synchronize().with_context(|| {
            format!("synchronizing packed FP8 MLA bucket={bucket_rows} before prepare")
        })?;
        let python_buffers = flashinfer_packed_fp8_mla_python_buffers(buffers.flashinfer);
        launch_python_graph_capture(PythonGraphCaptureLaunch {
            module: B12X_MLA_CAPTURE_MODULE,
            function: FLASHINFER_PACKED_FP8_MLA_PREPARE_FUNCTION,
            cuda_stream: slot.stream_ptr(),
            buffers: &python_buffers,
            kwargs: &kwargs,
        })
        .with_context(|| {
            format!("preparing packed FP8 MLA bucket={bucket_rows} query_rows={query_rows}")
        })?;
        unsafe {
            if let Some(physical_pages) = buffers.physical_page_table {
                library
                    .cuda_target_kv_page_table_expand_indices_async(
                        buffers.flashinfer.indices,
                        physical_pages,
                        query_rows,
                        bucket_rows,
                        bucket_rows,
                        slot.stream_ptr(),
                    )
                    .context("initializing paged packed FP8 MLA physical indices")?;
            } else if !initialize_kv {
                library
                    .cuda_glm_dsa_page_table_init_offsets_async(
                        buffers.flashinfer.indices,
                        buffers.flashinfer.index_base,
                        query_rows,
                        bucket_rows,
                        slot.stream_ptr(),
                    )
                    .context("initializing packed FP8 MLA physical indices")?;
            }
            enqueue_flashinfer_packed_fp8_mla_query(library, slot.stream_ptr(), buffers, geometry)?;
        }
        launch_python_graph_capture(PythonGraphCaptureLaunch {
            module: B12X_MLA_CAPTURE_MODULE,
            function: FLASHINFER_PACKED_FP8_MLA_CAPTURE_FUNCTION,
            cuda_stream: slot.stream_ptr(),
            buffers: &python_buffers,
            kwargs: &kwargs,
        })
        .with_context(|| {
            format!("warming packed FP8 MLA decode bucket={bucket_rows} query_rows={query_rows}")
        })?;
        unsafe {
            enqueue_flashinfer_packed_fp8_mla_output(
                library,
                slot.stream_ptr(),
                buffers,
                geometry,
            )?;
        }
        slot.stream_synchronize().with_context(|| {
            format!("synchronizing prepared packed FP8 MLA bucket={bucket_rows}")
        })?;
        slot.capture_or_update_graph_exec(
            library,
            program,
            signature,
            capture_identity,
            |library, cuda_stream, _workspace| {
                unsafe {
                    if let Some(physical_pages) = buffers.physical_page_table {
                        library
                            .cuda_target_kv_page_table_expand_indices_async(
                                buffers.flashinfer.indices,
                                physical_pages,
                                query_rows,
                                bucket_rows,
                                bucket_rows,
                                cuda_stream,
                            )
                            .context("capturing paged packed FP8 MLA physical-index init")?;
                    } else if !initialize_kv {
                        library
                            .cuda_glm_dsa_page_table_init_offsets_async(
                                buffers.flashinfer.indices,
                                buffers.flashinfer.index_base,
                                query_rows,
                                bucket_rows,
                                cuda_stream,
                            )
                            .context("capturing packed FP8 MLA physical-index init")?;
                    }
                    enqueue_flashinfer_packed_fp8_mla_query(
                        library,
                        cuda_stream,
                        buffers,
                        geometry,
                    )?;
                }
                launch_python_graph_capture(PythonGraphCaptureLaunch {
                    module: B12X_MLA_CAPTURE_MODULE,
                    function: FLASHINFER_PACKED_FP8_MLA_CAPTURE_FUNCTION,
                    cuda_stream,
                    buffers: &python_buffers,
                    kwargs: &kwargs,
                })
                .with_context(|| {
                    format!(
                        "capturing packed FP8 MLA decode bucket={bucket_rows} query_rows={query_rows}"
                    )
                })?;
                unsafe {
                    enqueue_flashinfer_packed_fp8_mla_output(
                        library,
                        cuda_stream,
                        buffers,
                        geometry,
                    )?;
                }
                Ok(())
            },
        )?;
    }
    Ok(())
}

fn flashinfer_packed_fp8_mla_capture_identity(
    buffers: FlashinferPackedFp8MlaFullGraphBuffers,
) -> usize {
    let (hidden_weight, hidden_output) = buffers
        .hidden_projection
        .map(|projection| {
            (
                projection.weight.ptr as usize,
                projection.output.ptr as usize,
            )
        })
        .unwrap_or((0, 0));
    let hidden_w4a16 = buffers.hidden_projection_w4a16;
    let hidden_w8a16_packed_o = buffers.hidden_projection_w8a16_packed_o;
    let hidden_w8a16 = buffers
        .hidden_projection
        .and_then(|projection| projection.w8a16);
    mla_graph_capture_identity(&[
        buffers.flashinfer.q.ptr as usize,
        buffers.flashinfer.kv.ptr as usize,
        buffers.flashinfer.indices.ptr as usize,
        buffers.flashinfer.topk_length.ptr as usize,
        buffers.flashinfer.index_base.ptr as usize,
        buffers.flashinfer.output.ptr as usize,
        buffers.flashinfer.out_lse.ptr as usize,
        buffers.flashinfer.mid_out.ptr as usize,
        buffers.flashinfer.mid_lse.ptr as usize,
        buffers.q_nope.ptr as usize,
        buffers.q_absorbed.ptr as usize,
        buffers.q_rope.ptr as usize,
        buffers.q_rope_staging.ptr as usize,
        buffers
            .physical_page_table
            .map_or(0, |page_table| page_table.ptr as usize),
        buffers.kv_b_weight.ptr as usize,
        buffers.value_weight.ptr as usize,
        buffers.final_output.ptr as usize,
        hidden_weight,
        hidden_output,
        hidden_w4a16.map_or(0, |w4a16| w4a16.weight.ptr as usize),
        hidden_w4a16.map_or(0, |w4a16| w4a16.scale.ptr as usize),
        hidden_w4a16.map_or(0, |w4a16| w4a16.global_scale.ptr as usize),
        hidden_w4a16.map_or(0, |w4a16| w4a16.c_tmp.ptr as usize),
        hidden_w4a16.map_or(0, |w4a16| w4a16.locks.ptr as usize),
        hidden_w8a16_packed_o.map_or(0, |w8a16| w8a16.c_tmp.ptr as usize),
        hidden_w8a16_packed_o.map_or(0, |w8a16| w8a16.packed_route_indices.ptr as usize),
        hidden_w8a16_packed_o.map_or(0, |w8a16| w8a16.locks.ptr as usize),
        hidden_w8a16.map_or(0, |w8a16| w8a16.weight.ptr as usize),
        hidden_w8a16.map_or(0, |w8a16| w8a16.scales.ptr as usize),
        hidden_w8a16.map_or(0, |w8a16| usize::from(w8a16.packed_layout)),
    ])
}

unsafe fn enqueue_flashinfer_packed_fp8_mla_query(
    library: &'static NativeLibrary,
    stream: *mut c_void,
    buffers: FlashinferPackedFp8MlaFullGraphBuffers,
    geometry: FlashinferPackedFp8MlaFullGraphGeometry,
) -> Result<()> {
    let bf16_bytes = std::mem::size_of::<u16>();
    let q_nope_row_bytes = geometry.heads * geometry.nope_dim * bf16_bytes;
    let q_absorbed_row_bytes = geometry.heads * geometry.rank * bf16_bytes;
    for query_index in 0..geometry.query_rows {
        let q_nope = device_buffer_byte_view(
            buffers.q_nope,
            query_index * q_nope_row_bytes,
            q_nope_row_bytes,
            "FlashInfer packed FP8 MLA q-nope row",
        )?;
        let q_absorbed = device_buffer_byte_view(
            buffers.q_absorbed,
            query_index * q_absorbed_row_bytes,
            q_absorbed_row_bytes,
            "FlashInfer packed FP8 MLA absorbed query row",
        )?;
        library
            .cuda_matmul_bf16_strided_batched_cublas_async(
                q_nope,
                buffers.kv_b_weight,
                q_absorbed,
                geometry.heads,
                1,
                geometry.nope_dim,
                geometry.rank,
                geometry.nope_dim,
                geometry.weight_head_stride,
                geometry.rank,
                stream,
            )
            .context("absorbing FlashInfer packed FP8 MLA suffix query")?;
    }
    library
        .copy_d2d_2d_async(
            buffers.flashinfer.q,
            geometry.combined_query_row_bytes,
            buffers.q_absorbed,
            geometry.rank * bf16_bytes,
            geometry.rank * bf16_bytes,
            geometry.query_rows * geometry.heads,
            stream,
        )
        .context("staging FlashInfer packed FP8 MLA latent query")?;
    library
        .copy_d2d_2d_async(
            buffers.q_rope_staging,
            geometry.combined_query_row_bytes,
            buffers.q_rope,
            geometry.rope_dim * bf16_bytes,
            geometry.rope_dim * bf16_bytes,
            geometry.query_rows * geometry.heads,
            stream,
        )
        .context("staging FlashInfer packed FP8 MLA RoPE query")
}

unsafe fn enqueue_flashinfer_packed_fp8_mla_output(
    library: &'static NativeLibrary,
    stream: *mut c_void,
    buffers: FlashinferPackedFp8MlaFullGraphBuffers,
    geometry: FlashinferPackedFp8MlaFullGraphGeometry,
) -> Result<()> {
    if geometry.query_rows == 1 {
        library
            .cuda_linear_bf16_strided_batched_cublas_async(
                buffers.flashinfer.output,
                buffers.value_weight,
                buffers.final_output,
                geometry.heads,
                1,
                geometry.rank,
                geometry.v_dim,
                geometry.rank,
                geometry.weight_head_stride,
                geometry.v_dim,
                stream,
            )
            .context("expanding FlashInfer packed FP8 MLA suffix values")?;
    } else {
        // Target verification produces query-major [Q,H,K] latent rows. One
        // head-major M=Q batched GEMM is bitwise equal to Q separate M=1
        // calls for this BF16 cublas contract while traversing each per-head
        // 512x512 value weight once. The two vectorized transposes preserve
        // the row-major [Q,H,V] ABI consumed by the recurrent-parity O kernel.
        library
            .cuda_transpose_rows_heads_bf16_async(
                buffers.flashinfer.output,
                buffers.q_absorbed,
                geometry.query_rows,
                geometry.heads,
                geometry.rank,
                stream,
            )
            .context("transposing packed FP8 MLA latent suffix to head-major layout")?;
        library
            .cuda_linear_bf16_strided_batched_cublas_async(
                buffers.q_absorbed,
                buffers.value_weight,
                buffers.flashinfer.q,
                geometry.heads,
                geometry.query_rows,
                geometry.rank,
                geometry.v_dim,
                geometry.query_rows * geometry.rank,
                geometry.weight_head_stride,
                geometry.query_rows * geometry.v_dim,
                stream,
            )
            .context("batching packed FP8 MLA suffix value expansion")?;
        library
            .cuda_transpose_heads_rows_bf16_async(
                buffers.flashinfer.q,
                buffers.final_output,
                geometry.query_rows,
                geometry.heads,
                geometry.v_dim,
                stream,
            )
            .context("restoring packed FP8 MLA suffix values to query-major layout")?;
    }
    if buffers
        .hidden_projection
        .is_some_and(|projection| projection.w8a16.is_some())
    {
        let projection = buffers
            .hidden_projection
            .expect("W8A16 hidden projection is present");
        let w8a16 = projection.w8a16.expect("W8A16 buffers are present");
        let input_dim = geometry.heads * geometry.v_dim;
        // Scalar decode and the qualified adaptive widths retain recurrent
        // arithmetic. The grouped packed-O kernel is flat at roughly 0.039 ms
        // through M=16, but differs in a handful of BF16 values and failed the
        // complete weighted trajectory when applied to M=2..8. Use it only to
        // replace the pathological M separate launches at the qualified
        // adaptive M=9..16 widths.
        if w8a16.packed_layout && geometry.query_rows >= 9 {
            let packed_o = buffers
                .hidden_projection_w8a16_packed_o
                .context("packed W8A16 O launch buffers are missing")?;
            library
                .cuda_w8a16_packed_o_initialize_launch_buffers_async(
                    &packed_o,
                    geometry.query_rows,
                    16,
                    stream,
                )
                .context("initializing packed FP8 MLA grouped W8A16 O metadata")?;
            library
                .cuda_w8a16_packed_o_async(&packed_o, geometry.query_rows, stream)
                .context("projecting packed FP8 MLA output with grouped W8A16 O")?;
        } else if w8a16.packed_layout && geometry.query_rows == 1 {
            library
                .cuda_linear_w8a16_group256_m1_warp_packed_async(
                    buffers.final_output,
                    w8a16.weight,
                    w8a16.scales,
                    projection.output,
                    input_dim,
                    projection.hidden_dim,
                    stream,
                )
                .context("projecting packed FP8 MLA output with packed recurrent W8A16")?;
        } else if w8a16.packed_layout && geometry.query_rows >= 4 {
            library
                .cuda_linear_w8a16_group256_m1_warp_packed_parity_batched_async(
                    buffers.final_output,
                    w8a16.weight,
                    w8a16.scales,
                    projection.output,
                    geometry.query_rows,
                    input_dim,
                    projection.hidden_dim,
                    stream,
                )
                .context("projecting packed FP8 MLA output with packed parity-batched W8A16")?;
        } else if w8a16.packed_layout {
            let input_row_bytes = input_dim * std::mem::size_of::<u16>();
            let output_row_bytes = projection.hidden_dim * std::mem::size_of::<u16>();
            for row in 0..geometry.query_rows {
                let input_row = device_buffer_byte_view(
                    buffers.final_output,
                    row * input_row_bytes,
                    input_row_bytes,
                    "packed W8A16 O recurrent input row",
                )?;
                let output_row = device_buffer_byte_view(
                    projection.output,
                    row * output_row_bytes,
                    output_row_bytes,
                    "packed W8A16 O recurrent output row",
                )?;
                library
                    .cuda_linear_w8a16_group256_m1_warp_packed_async(
                        input_row,
                        w8a16.weight,
                        w8a16.scales,
                        output_row,
                        input_dim,
                        projection.hidden_dim,
                        stream,
                    )
                    .context("projecting packed FP8 MLA output with packed recurrent W8A16")?;
            }
        } else if geometry.query_rows == 1 {
            library
                .cuda_linear_w8a16_group256_m1_simt_async(
                    buffers.final_output,
                    w8a16.weight,
                    w8a16.scales,
                    projection.output,
                    input_dim,
                    projection.hidden_dim,
                    3,
                    stream,
                )
                .context("projecting packed FP8 MLA output with recurrent W8A16")?;
        } else {
            library
                .cuda_linear_w8a16_group256_m1_parity_batched_async(
                    buffers.final_output,
                    w8a16.weight,
                    w8a16.scales,
                    projection.output,
                    geometry.query_rows,
                    input_dim,
                    projection.hidden_dim,
                    stream,
                )
                .context("projecting packed FP8 MLA output with parity-batched W8A16")?;
        }
    } else if let Some(w4a16) = buffers.hidden_projection_w4a16 {
        if coordinator_w4a16_o_proj_tn64_enabled() {
            library
                .cuda_b12x_coordinator_w4a16_o_proj_m1_tn64_async(&w4a16, stream)
                .context("projecting packed FP8 MLA decode output with W4A16 TN64")?;
        } else {
            library
                .cuda_b12x_coordinator_w4a16_o_proj_m1_async(&w4a16, stream)
                .context("projecting packed FP8 MLA decode output with W4A16")?;
        }
    } else if let Some(projection) = buffers.hidden_projection {
        library
            .cuda_linear_bf16_cublas_async(
                buffers.final_output,
                projection.weight,
                None,
                projection.output,
                geometry.query_rows,
                geometry.heads * geometry.v_dim,
                projection.hidden_dim,
                stream,
            )
            .context("projecting packed FP8 MLA decode output to hidden width")?;
    }
    Ok(())
}

fn coordinator_w4a16_o_proj_tn64_enabled() -> bool {
    std::env::var("GLMRT_B12X_COORDINATOR_W4A16_O_PROJ_TN64")
        .map(|value| value != "0")
        .unwrap_or(true)
}

fn flashinfer_packed_fp8_mla_python_buffers(
    buffers: FlashinferPackedFp8MlaDecodeBuffers,
) -> [PythonDeviceBufferArg<'static>; 8] {
    [
        python_device_buffer_arg("q", buffers.q),
        python_device_buffer_arg("kv", buffers.kv),
        python_device_buffer_arg("indices", buffers.indices),
        python_device_buffer_arg("topk_length", buffers.topk_length),
        python_device_buffer_arg("output", buffers.output),
        python_device_buffer_arg("out_lse", buffers.out_lse),
        python_device_buffer_arg("mid_out", buffers.mid_out),
        python_device_buffer_arg("mid_lse", buffers.mid_lse),
    ]
}

fn flashinfer_compressed_mla_decode_chunk_rows(remaining_rows: usize) -> usize {
    if remaining_rows <= FLASHINFER_COMPRESSED_MLA_EXACT_TAIL_ROWS {
        return remaining_rows;
    }
    let largest_power_of_two = 1_usize << (usize::BITS - 1 - remaining_rows.leading_zeros());
    largest_power_of_two.min(FLASHINFER_COMPRESSED_MLA_MAX_CHUNK_ROWS)
}

#[allow(clippy::too_many_arguments)]
fn ensure_flashinfer_compressed_mla_decode_graphs(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    buffers: FlashinferCompressedMlaDecodeBuffers,
    heads: usize,
    rank: usize,
    rope_dim: usize,
    scale: f32,
) -> Result<()> {
    for rows in (1..=FLASHINFER_COMPRESSED_MLA_EXACT_TAIL_ROWS).chain([
        64,
        128,
        256,
        512,
        1_024,
        FLASHINFER_COMPRESSED_MLA_MAX_CHUNK_ROWS,
    ]) {
        let signature = CoordinatorCudaGraphSignature::flashinfer_compressed_mla_decode_bf16(
            rows, heads, rank, rope_dim, scale,
        );
        let init_program = CoordinatorCudaGraphProgram::LayerFlashinferCompressedMlaDecodeBf16Init;
        let merge_program =
            CoordinatorCudaGraphProgram::LayerFlashinferCompressedMlaDecodeBf16Merge;
        let init_captured = slot.has_captured_graph(init_program, signature);
        let merge_captured = slot.has_captured_graph(merge_program, signature);
        if init_captured && merge_captured {
            continue;
        }
        anyhow::ensure!(
            coordinator_python_capture_startup_open(),
            "FlashInfer compressed MLA decode graph rows={rows} was not captured during startup"
        );

        let python_buffers = flashinfer_compressed_mla_python_buffers(buffers);
        let kwargs = [
            ("rows", PythonKernelArg::Usize(rows)),
            ("heads", PythonKernelArg::Usize(heads)),
            ("nope_dim", PythonKernelArg::Usize(rank)),
            ("rope_dim", PythonKernelArg::Usize(rope_dim)),
            ("scale", PythonKernelArg::F64(scale as f64)),
        ];
        slot.stream_synchronize().with_context(|| {
            format!("synchronizing FlashInfer compressed MLA rows={rows} before prepare")
        })?;
        launch_python_graph_capture(PythonGraphCaptureLaunch {
            module: B12X_MLA_CAPTURE_MODULE,
            function: FLASHINFER_COMPRESSED_MLA_PREPARE_FUNCTION,
            cuda_stream: slot.stream_ptr(),
            buffers: &python_buffers,
            kwargs: &kwargs,
        })
        .with_context(|| format!("preparing FlashInfer compressed MLA rows={rows}"))?;
        slot.stream_synchronize().with_context(|| {
            format!("synchronizing prepared FlashInfer compressed MLA rows={rows}")
        })?;

        if !init_captured {
            slot.capture_graph(
                library,
                init_program,
                signature,
                |_library, cuda_stream, _workspace| {
                    launch_python_graph_capture(PythonGraphCaptureLaunch {
                        module: B12X_MLA_CAPTURE_MODULE,
                        function: FLASHINFER_COMPRESSED_MLA_CAPTURE_FUNCTION,
                        cuda_stream,
                        buffers: &python_buffers,
                        kwargs: &kwargs,
                    })
                    .with_context(|| {
                        format!("capturing FlashInfer compressed MLA init rows={rows}")
                    })?;
                    unsafe {
                        library.copy_d2d_async(
                            buffers.accumulator,
                            buffers.partial,
                            heads * rank * std::mem::size_of::<u16>(),
                            cuda_stream,
                        )?;
                        library.copy_d2d_async(
                            buffers.accumulator_lse,
                            buffers.partial_lse,
                            heads * std::mem::size_of::<f32>(),
                            cuda_stream,
                        )?;
                    }
                    Ok(())
                },
            )?;
        }
        if !merge_captured {
            slot.capture_graph(
                library,
                merge_program,
                signature,
                |_library, cuda_stream, _workspace| {
                    launch_python_graph_capture(PythonGraphCaptureLaunch {
                        module: B12X_MLA_CAPTURE_MODULE,
                        function: FLASHINFER_COMPRESSED_MLA_CAPTURE_FUNCTION,
                        cuda_stream,
                        buffers: &python_buffers,
                        kwargs: &kwargs,
                    })
                    .with_context(|| {
                        format!("capturing FlashInfer compressed MLA merge rows={rows}")
                    })?;
                    unsafe {
                        library.cuda_mla_merge_state_bf16_async(
                            buffers.accumulator,
                            buffers.accumulator_lse,
                            buffers.partial,
                            buffers.partial_lse,
                            heads,
                            rank,
                            cuda_stream,
                        )?;
                    }
                    Ok(())
                },
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn prewarm_flashinfer_compressed_mla_decode_fallback_graphs(
    layer_id: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    rank: usize,
    scale: f32,
) -> Result<()> {
    if !coordinator_python_capture_startup_open() {
        return Ok(());
    }

    let bf16_bytes = std::mem::size_of::<u16>();
    let latent_bytes = heads
        .checked_mul(rank)
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("compressed MLA fallback latent bytes overflow")?;
    let lse_bytes = heads
        .checked_mul(std::mem::size_of::<f32>())
        .context("compressed MLA fallback LSE bytes overflow")?;
    let q_nope_row_bytes = heads
        .checked_mul(nope_dim)
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("compressed MLA fallback q_nope bytes overflow")?;
    let q_rope_row_bytes = heads
        .checked_mul(rope_dim)
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("compressed MLA fallback q_rope bytes overflow")?;
    let query_projection_bytes = heads
        .checked_mul(
            nope_dim
                .checked_add(rope_dim)
                .context("compressed MLA fallback query width overflow")?,
        )
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("compressed MLA fallback query projection bytes overflow")?;
    let kv_row_bytes = rank
        .checked_add(rope_dim)
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("compressed MLA fallback KV row bytes overflow")?;
    let output_bytes = heads
        .checked_mul(v_dim)
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("compressed MLA fallback output bytes overflow")?;
    let kv_staging_bytes = FLASHINFER_COMPRESSED_MLA_MAX_CHUNK_ROWS
        .checked_mul(kv_row_bytes)
        .context("compressed MLA fallback KV staging bytes overflow")?;

    let graph_key = coord_compressed_attention_decode_graph_key_for_layer(layer_id)?;
    with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
        let q_absorbed = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::H,
            latent_bytes.max(query_projection_bytes),
            "FlashInfer compressed MLA absorbed query",
        )?;
        let q_rope_staging = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::I,
            query_projection_bytes.max(q_rope_row_bytes),
            "FlashInfer compressed MLA RoPE query",
        )?;
        let workspace = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::J,
            FLASHINFER_SINGLE_PREFILL_TMP_BYTES,
            "FlashInfer compressed MLA workspace",
        )?;
        let kv_staging = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::K,
            kv_staging_bytes.max(output_bytes),
            "FlashInfer compressed MLA KV staging",
        )?;
        let partial = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::L,
            latent_bytes.max(q_nope_row_bytes),
            "FlashInfer compressed MLA partial state",
        )?;
        let partial_lse = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::M,
            lse_bytes.max(q_rope_row_bytes),
            "FlashInfer compressed MLA partial LSE",
        )?;
        let accumulator = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::N,
            latent_bytes.max(q_nope_row_bytes),
            "FlashInfer compressed MLA accumulated state",
        )?;
        let accumulator_lse = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::O,
            lse_bytes.max(rope_dim * bf16_bytes),
            "FlashInfer compressed MLA accumulated LSE",
        )?;
        let _expanded_value_staging = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::P,
            output_bytes,
            "FlashInfer expanded MLA value staging",
        )?;

        ensure_flashinfer_compressed_mla_decode_graphs(
            library,
            slot,
            FlashinferCompressedMlaDecodeBuffers {
                q_nope: q_absorbed,
                q_rope: q_rope_staging,
                kv: kv_staging,
                partial,
                partial_lse,
                accumulator,
                accumulator_lse,
                workspace,
            },
            heads,
            rank,
            rope_dim,
            scale,
        )
    })
}

fn flashinfer_compressed_mla_python_buffers(
    buffers: FlashinferCompressedMlaDecodeBuffers,
) -> [PythonDeviceBufferArg<'static>; 6] {
    [
        python_device_buffer_arg("q_nope", buffers.q_nope),
        python_device_buffer_arg("q_rope", buffers.q_rope),
        python_device_buffer_arg("kv", buffers.kv),
        python_device_buffer_arg("partial", buffers.partial),
        python_device_buffer_arg("partial_lse", buffers.partial_lse),
        python_device_buffer_arg("workspace", buffers.workspace),
    ]
}

fn python_device_buffer_arg(
    name: &'static str,
    buffer: GlmrtDeviceBuffer,
) -> PythonDeviceBufferArg<'static> {
    PythonDeviceBufferArg {
        name,
        ptr: buffer.ptr,
        bytes: buffer.bytes,
        device_id: buffer.device_id,
        flags: buffer.flags,
    }
}

#[allow(clippy::too_many_arguments)]
fn stage_flashinfer_compressed_mla_kv_chunk(
    library: &'static NativeLibrary,
    input: FlashinferCompressedMlaKvInput,
    staging: GlmrtDeviceBuffer,
    row_offset: usize,
    rows: usize,
    rank: usize,
    rope_dim: usize,
    cuda_stream: *mut c_void,
) -> Result<()> {
    let bf16_bytes = std::mem::size_of::<u16>();
    let latent_row_bytes = rank * bf16_bytes;
    let rope_row_bytes = rope_dim * bf16_bytes;
    let staging_row_bytes = latent_row_bytes + rope_row_bytes;
    match input {
        FlashinferCompressedMlaKvInput::SplitBf16 { latent, rope } => {
            let latent_offset = row_offset
                .checked_mul(latent_row_bytes)
                .context("FlashInfer compressed MLA latent chunk offset overflow")?;
            let rope_offset = row_offset
                .checked_mul(rope_row_bytes)
                .context("FlashInfer compressed MLA RoPE chunk offset overflow")?;
            let latent_source = device_buffer_byte_view(
                latent,
                latent_offset,
                latent
                    .bytes
                    .checked_sub(latent_offset)
                    .context("FlashInfer compressed MLA latent chunk exceeds source buffer")?,
                "FlashInfer compressed MLA latent chunk source",
            )?;
            let rope_source = device_buffer_byte_view(
                rope,
                rope_offset,
                rope.bytes
                    .checked_sub(rope_offset)
                    .context("FlashInfer compressed MLA RoPE chunk exceeds source buffer")?,
                "FlashInfer compressed MLA RoPE chunk source",
            )?;
            let rope_destination = device_buffer_byte_view(
                staging,
                latent_row_bytes,
                staging.bytes - latent_row_bytes,
                "FlashInfer compressed MLA interleaved RoPE destination",
            )?;
            unsafe {
                library.copy_d2d_2d_async(
                    staging,
                    staging_row_bytes,
                    latent_source,
                    latent_row_bytes,
                    latent_row_bytes,
                    rows,
                    cuda_stream,
                )?;
                library.copy_d2d_2d_async(
                    rope_destination,
                    staging_row_bytes,
                    rope_source,
                    rope_row_bytes,
                    rope_row_bytes,
                    rows,
                    cuda_stream,
                )?;
            }
        }
        FlashinferCompressedMlaKvInput::Interleaved {
            payload,
            dtype,
            row_stride_bytes,
            row_offset: payload_row_offset,
            physical_page_table,
            ..
        } => {
            anyhow::ensure!(
                physical_page_table.is_none(),
                "paged compressed MLA KV must use direct packed attention instead of contiguous chunk staging"
            );
            let source_offset = payload_row_offset
                .checked_add(row_offset)
                .context("FlashInfer compressed MLA packed source row offset overflow")?
                .checked_mul(row_stride_bytes)
                .context("FlashInfer compressed MLA packed chunk offset overflow")?;
            let source = device_buffer_byte_view(
                payload,
                source_offset,
                payload
                    .bytes
                    .checked_sub(source_offset)
                    .context("FlashInfer compressed MLA chunk exceeds packed source buffer")?,
                "FlashInfer compressed MLA packed chunk source",
            )?;
            unsafe {
                match dtype {
                    KvCacheDType::Bf16 => library.copy_d2d_2d_async(
                        staging,
                        staging_row_bytes,
                        source,
                        row_stride_bytes,
                        staging_row_bytes,
                        rows,
                        cuda_stream,
                    )?,
                    KvCacheDType::Fp8 => library.cuda_mla_kv_unpack_fp8_ds_mla_async(
                        source,
                        staging,
                        rows,
                        row_stride_bytes,
                        staging_row_bytes,
                        cuda_stream,
                    )?,
                    KvCacheDType::Nvfp4 => library.cuda_mla_kv_unpack_mxfp4_ds_mla_async(
                        source,
                        staging,
                        rows,
                        row_stride_bytes,
                        staging_row_bytes,
                        cuda_stream,
                    )?,
                    unsupported => anyhow::bail!(
                        "FlashInfer compressed MLA decode does not support {} cache rows",
                        unsupported.label()
                    ),
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn capture_or_update_layer_flashinfer_mla_rope_attention_bf16_graph(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    signature: CoordinatorCudaGraphSignature,
    q_nope_buffer: GlmrtDeviceBuffer,
    q_rope_buffer: GlmrtDeviceBuffer,
    k_nope_buffer: GlmrtDeviceBuffer,
    k_rope_buffer: GlmrtDeviceBuffer,
    value_buffer: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    query_row_offset: usize,
    query_rows: usize,
    row_capacity: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    scale: f32,
    label: &'static str,
    inputs_are_staged: bool,
) -> Result<bool> {
    if !inputs_are_staged {
        validate_mla_rope_attention_device_buffers_with_output_rows(
            rows,
            query_rows,
            q_nope_buffer,
            q_rope_buffer,
            k_nope_buffer,
            k_rope_buffer,
            value_buffer,
            output_buffer,
            heads,
            nope_dim,
            rope_dim,
            v_dim,
        )?;
    }
    if query_row_offset > rows || query_rows > rows - query_row_offset {
        anyhow::bail!(
            "FlashInfer MLA query rows {}..{} exceed KV rows {rows}",
            query_row_offset,
            query_row_offset.saturating_add(query_rows)
        );
    }
    if row_capacity < rows {
        anyhow::bail!(
            "FlashInfer MLA graph row capacity {row_capacity} is smaller than rows {rows}"
        );
    }
    let capture_shape =
        flashinfer_mla_capture_shape(rows, query_row_offset, query_rows, row_capacity)?;
    let capture_rows = capture_shape.rows;
    let capture_query_rows = capture_shape.query_rows;
    let full_query = query_row_offset == 0 && query_rows == rows;
    let dynamic_suffix = !full_query;
    anyhow::ensure!(
        !inputs_are_staged || dynamic_suffix,
        "pre-staged FlashInfer/cuDNN MLA inputs are only valid for suffix capture"
    );
    let qk_dim = nope_dim
        .checked_add(rope_dim)
        .context("FlashInfer MLA qk dimension overflow")?;
    let q_capacity_rows = if dynamic_suffix {
        capture_query_rows
    } else {
        row_capacity
    };
    let q_capacity_bytes = q_capacity_rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(qk_dim))
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("FlashInfer MLA query scratch capacity overflow")?;
    let k_bytes = row_capacity
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(qk_dim))
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("FlashInfer MLA key scratch bytes overflow")?;
    let output_bytes = query_rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(v_dim))
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("FlashInfer MLA output scratch bytes overflow")?;
    let output_capacity_rows = if dynamic_suffix {
        capture_query_rows
    } else {
        row_capacity
    };
    let output_capacity_bytes = output_capacity_rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(v_dim))
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("FlashInfer MLA output scratch capacity overflow")?;
    let capture_output_bytes = capture_query_rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(v_dim))
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("FlashInfer MLA captured output bytes overflow")?;
    let signature = if full_query {
        CoordinatorCudaGraphSignature::mla_rope_attention_bf16(
            capture_output_bytes,
            capture_rows,
            heads,
            nope_dim,
            rope_dim,
            v_dim,
            scale,
        )
    } else if capture_rows != rows || capture_query_rows != query_rows {
        CoordinatorCudaGraphSignature::mla_rope_attention_bf16_suffix(
            capture_output_bytes,
            capture_rows,
            capture_query_rows,
            heads,
            nope_dim,
            rope_dim,
            v_dim,
            scale,
        )
    } else {
        signature
    };
    let shared_suffix = if dynamic_suffix {
        let query_capacity = flashinfer_cudnn_mla_suffix_query_capacity();
        anyhow::ensure!(
            capture_query_rows <= query_capacity,
            "FlashInfer/cuDNN MLA suffix query bucket {capture_query_rows} exceeds rolling capacity {}",
            query_capacity
        );
        anyhow::ensure!(
            row_capacity <= FLASHINFER_CUDNN_MLA_SUFFIX_MAX_ROW_CAPACITY,
            "FlashInfer/cuDNN MLA suffix row capacity {row_capacity} exceeds shared capacity {}",
            FLASHINFER_CUDNN_MLA_SUFFIX_MAX_ROW_CAPACITY
        );
        Some(flashinfer_cudnn_mla_suffix_buffers(library)?)
    } else {
        None
    };
    let q_buffer = if let Some(buffers) = shared_suffix {
        buffers.q
    } else {
        slot.buffer(
            library,
            CoordinatorCudaScratchSlot::H,
            q_capacity_bytes,
            "FlashInfer MLA contiguous query",
        )?
    };
    let k_buffer = if let Some(buffers) = shared_suffix {
        buffers.k
    } else {
        slot.buffer(
            library,
            CoordinatorCudaScratchSlot::I,
            k_bytes,
            "FlashInfer MLA contiguous key",
        )?
    };
    let workspace_buffer = if let Some(buffers) = shared_suffix {
        buffers.workspace
    } else {
        slot.buffer(
            library,
            CoordinatorCudaScratchSlot::J,
            FLASHINFER_SINGLE_PREFILL_TMP_BYTES,
            "FlashInfer MLA single-prefill workspace",
        )?
    };
    let output_scratch = if let Some(buffers) = shared_suffix {
        buffers.output
    } else {
        slot.buffer(
            library,
            CoordinatorCudaScratchSlot::K,
            output_capacity_bytes,
            "FlashInfer MLA output",
        )?
    };
    let q_nope_row_bytes = heads
        .checked_mul(nope_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("FlashInfer MLA q_nope row bytes overflow")?;
    let q_rope_row_bytes = heads
        .checked_mul(rope_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("FlashInfer MLA q_rope row bytes overflow")?;
    let q_nope_bytes = query_rows
        .checked_mul(q_nope_row_bytes)
        .context("FlashInfer MLA q_nope bytes overflow")?;
    let q_rope_bytes = query_rows
        .checked_mul(q_rope_row_bytes)
        .context("FlashInfer MLA q_rope bytes overflow")?;
    let k_nope_bytes = rows
        .checked_mul(q_nope_row_bytes)
        .context("FlashInfer MLA k_nope bytes overflow")?;
    let k_rope_bytes = rows
        .checked_mul(rope_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("FlashInfer MLA k_rope bytes overflow")?;
    let value_bytes = rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(v_dim))
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("FlashInfer MLA value bytes overflow")?;
    let q_nope_capacity_bytes = q_capacity_rows
        .checked_mul(q_nope_row_bytes)
        .context("FlashInfer MLA q_nope staging capacity overflow")?;
    let q_rope_capacity_bytes = q_capacity_rows
        .checked_mul(q_rope_row_bytes)
        .context("FlashInfer MLA q_rope staging capacity overflow")?;
    let k_nope_capacity_bytes = row_capacity
        .checked_mul(q_nope_row_bytes)
        .context("FlashInfer MLA k_nope staging capacity overflow")?;
    let k_rope_capacity_bytes = row_capacity
        .checked_mul(rope_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("FlashInfer MLA k_rope staging capacity overflow")?;
    let q_nope_staging = if let Some(buffers) = shared_suffix {
        buffers.q_nope
    } else {
        slot.buffer(
            library,
            CoordinatorCudaScratchSlot::L,
            q_nope_capacity_bytes,
            "FlashInfer MLA q_nope staging",
        )?
    };
    let q_rope_staging = if let Some(buffers) = shared_suffix {
        buffers.q_rope
    } else {
        slot.buffer(
            library,
            CoordinatorCudaScratchSlot::M,
            q_rope_capacity_bytes,
            "FlashInfer MLA q_rope staging",
        )?
    };
    let k_nope_staging = if let Some(buffers) = shared_suffix {
        buffers.k_nope
    } else {
        slot.buffer(
            library,
            CoordinatorCudaScratchSlot::N,
            k_nope_capacity_bytes,
            "FlashInfer MLA k_nope staging",
        )?
    };
    let k_rope_staging = if let Some(buffers) = shared_suffix {
        buffers.k_rope
    } else {
        slot.buffer(
            library,
            CoordinatorCudaScratchSlot::O,
            k_rope_capacity_bytes,
            "FlashInfer MLA k_rope staging",
        )?
    };
    let value_staging = if let Some(buffers) = shared_suffix {
        buffers.values
    } else {
        slot.buffer(
            library,
            CoordinatorCudaScratchSlot::P,
            row_capacity
                .checked_mul(heads)
                .and_then(|values| values.checked_mul(v_dim))
                .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
                .context("FlashInfer MLA value staging capacity overflow")?,
            "FlashInfer MLA value staging",
        )?
    };
    let query_lengths = dynamic_suffix
        .then(|| {
            slot.buffer(
                library,
                CoordinatorCudaScratchSlot::Q,
                std::mem::size_of::<u32>(),
                "FlashInfer/cuDNN MLA query lengths",
            )
        })
        .transpose()?;
    let kv_lengths = dynamic_suffix
        .then(|| {
            slot.buffer(
                library,
                CoordinatorCudaScratchSlot::R,
                std::mem::size_of::<u32>(),
                "FlashInfer/cuDNN MLA KV lengths",
            )
        })
        .transpose()?;
    let q_nope_offset = query_row_offset
        .checked_mul(q_nope_row_bytes)
        .context("FlashInfer MLA q_nope suffix offset overflow")?;
    let q_rope_offset = query_row_offset
        .checked_mul(q_rope_row_bytes)
        .context("FlashInfer MLA q_rope suffix offset overflow")?;
    let q_nope_source = if inputs_are_staged {
        GlmrtDeviceBuffer::default()
    } else {
        device_buffer_byte_view(
            q_nope_buffer,
            q_nope_offset,
            q_nope_bytes,
            "FlashInfer MLA q_nope suffix source",
        )?
    };
    let q_rope_source = if inputs_are_staged {
        GlmrtDeviceBuffer::default()
    } else {
        device_buffer_byte_view(
            q_rope_buffer,
            q_rope_offset,
            q_rope_bytes,
            "FlashInfer MLA q_rope suffix source",
        )?
    };
    let query_staging_prefix_rows = if dynamic_suffix {
        0
    } else {
        capture_shape.query_prefix_padding
    };
    let q_nope_staging_offset = query_staging_prefix_rows
        .checked_mul(q_nope_row_bytes)
        .context("FlashInfer MLA q_nope staging offset overflow")?;
    let q_rope_staging_offset = query_staging_prefix_rows
        .checked_mul(q_rope_row_bytes)
        .context("FlashInfer MLA q_rope staging offset overflow")?;
    let q_nope_destination = device_buffer_byte_view(
        q_nope_staging,
        q_nope_staging_offset,
        q_nope_bytes,
        "FlashInfer MLA q_nope staging destination",
    )?;
    let q_rope_destination = device_buffer_byte_view(
        q_rope_staging,
        q_rope_staging_offset,
        q_rope_bytes,
        "FlashInfer MLA q_rope staging destination",
    )?;
    let cuda_stream = slot.stream_ptr();
    if dynamic_suffix {
        let query_rows =
            u32::try_from(query_rows).context("FlashInfer/cuDNN MLA query rows exceed u32")?;
        let rows = u32::try_from(rows).context("FlashInfer/cuDNN MLA KV rows exceed u32")?;
        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::Q,
                &query_rows.to_ne_bytes(),
                "FlashInfer/cuDNN MLA query lengths",
                cuda_stream,
            )
            .context("staging FlashInfer/cuDNN MLA query length")?;
        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::R,
                &rows.to_ne_bytes(),
                "FlashInfer/cuDNN MLA KV lengths",
                cuda_stream,
            )
            .context("staging FlashInfer/cuDNN MLA KV length")?;
    }
    unsafe {
        if !inputs_are_staged {
            if capture_query_rows != query_rows {
                library.cuda_zero_bytes_async(
                    q_nope_staging,
                    capture_query_rows * q_nope_row_bytes,
                    cuda_stream,
                )?;
                library.cuda_zero_bytes_async(
                    q_rope_staging,
                    capture_query_rows * q_rope_row_bytes,
                    cuda_stream,
                )?;
            }
            library.copy_d2d_async(q_nope_destination, q_nope_source, q_nope_bytes, cuda_stream)?;
            library.copy_d2d_async(q_rope_destination, q_rope_source, q_rope_bytes, cuda_stream)?;
            let k_nope_row_bytes = q_nope_row_bytes;
            let k_rope_row_bytes = rope_dim * std::mem::size_of::<u16>();
            let value_row_bytes = heads * v_dim * std::mem::size_of::<u16>();
            let kv_staging_prefix_rows = if dynamic_suffix {
                0
            } else {
                capture_shape.kv_prefix_padding
            };
            let k_nope_destination = device_buffer_byte_view(
                k_nope_staging,
                kv_staging_prefix_rows * k_nope_row_bytes,
                k_nope_bytes,
                "FlashInfer MLA k_nope staging destination",
            )?;
            let k_rope_destination = device_buffer_byte_view(
                k_rope_staging,
                kv_staging_prefix_rows * k_rope_row_bytes,
                k_rope_bytes,
                "FlashInfer MLA k_rope staging destination",
            )?;
            let value_destination = device_buffer_byte_view(
                value_staging,
                kv_staging_prefix_rows * value_row_bytes,
                value_bytes,
                "FlashInfer MLA value staging destination",
            )?;
            if kv_staging_prefix_rows > 0 {
                library.cuda_zero_bytes_async(
                    k_nope_staging,
                    kv_staging_prefix_rows * k_nope_row_bytes,
                    cuda_stream,
                )?;
                library.cuda_zero_bytes_async(
                    k_rope_staging,
                    kv_staging_prefix_rows * k_rope_row_bytes,
                    cuda_stream,
                )?;
                library.cuda_zero_bytes_async(
                    value_staging,
                    kv_staging_prefix_rows * value_row_bytes,
                    cuda_stream,
                )?;
            }
            library.copy_d2d_async(k_nope_destination, k_nope_buffer, k_nope_bytes, cuda_stream)?;
            library.copy_d2d_async(k_rope_destination, k_rope_buffer, k_rope_bytes, cuda_stream)?;
            library.copy_d2d_async(value_destination, value_buffer, value_bytes, cuda_stream)?;
            if capture_rows > rows {
                let padded_rows = capture_rows - rows;
                if kv_staging_prefix_rows == 0 {
                    let k_nope_padding = device_buffer_byte_view(
                        k_nope_staging,
                        k_nope_bytes,
                        padded_rows * k_nope_row_bytes,
                        "FlashInfer MLA k_nope padding",
                    )?;
                    let k_rope_padding = device_buffer_byte_view(
                        k_rope_staging,
                        k_rope_bytes,
                        padded_rows * k_rope_row_bytes,
                        "FlashInfer MLA k_rope padding",
                    )?;
                    let value_padding = device_buffer_byte_view(
                        value_staging,
                        value_bytes,
                        padded_rows * value_row_bytes,
                        "FlashInfer MLA value padding",
                    )?;
                    library.cuda_zero_bytes_async(
                        k_nope_padding,
                        k_nope_padding.bytes,
                        cuda_stream,
                    )?;
                    library.cuda_zero_bytes_async(
                        k_rope_padding,
                        k_rope_padding.bytes,
                        cuda_stream,
                    )?;
                    library.cuda_zero_bytes_async(
                        value_padding,
                        value_padding.bytes,
                        cuda_stream,
                    )?;
                }
            }
        }
    }
    let mut buffers = vec![
        PythonDeviceBufferArg {
            name: "q_nope",
            ptr: q_nope_staging.ptr,
            bytes: q_nope_staging.bytes,
            device_id: q_nope_staging.device_id,
            flags: q_nope_staging.flags,
        },
        PythonDeviceBufferArg {
            name: "q_rope",
            ptr: q_rope_staging.ptr,
            bytes: q_rope_staging.bytes,
            device_id: q_rope_staging.device_id,
            flags: q_rope_staging.flags,
        },
        PythonDeviceBufferArg {
            name: "k_nope",
            ptr: k_nope_staging.ptr,
            bytes: k_nope_staging.bytes,
            device_id: k_nope_staging.device_id,
            flags: k_nope_staging.flags,
        },
        PythonDeviceBufferArg {
            name: "k_rope",
            ptr: k_rope_staging.ptr,
            bytes: k_rope_staging.bytes,
            device_id: k_rope_staging.device_id,
            flags: k_rope_staging.flags,
        },
        PythonDeviceBufferArg {
            name: "values",
            ptr: value_staging.ptr,
            bytes: value_staging.bytes,
            device_id: value_staging.device_id,
            flags: value_staging.flags,
        },
        PythonDeviceBufferArg {
            name: "q",
            ptr: q_buffer.ptr,
            bytes: q_buffer.bytes,
            device_id: q_buffer.device_id,
            flags: q_buffer.flags,
        },
        PythonDeviceBufferArg {
            name: "k",
            ptr: k_buffer.ptr,
            bytes: k_buffer.bytes,
            device_id: k_buffer.device_id,
            flags: k_buffer.flags,
        },
        PythonDeviceBufferArg {
            name: "workspace",
            ptr: workspace_buffer.ptr,
            bytes: workspace_buffer.bytes,
            device_id: workspace_buffer.device_id,
            flags: workspace_buffer.flags,
        },
        PythonDeviceBufferArg {
            name: "output",
            ptr: output_scratch.ptr,
            bytes: output_scratch.bytes,
            device_id: output_scratch.device_id,
            flags: output_scratch.flags,
        },
    ];
    if let (Some(query_lengths), Some(kv_lengths)) = (query_lengths, kv_lengths) {
        buffers.extend([
            PythonDeviceBufferArg {
                name: "query_lengths",
                ptr: query_lengths.ptr,
                bytes: query_lengths.bytes,
                device_id: query_lengths.device_id,
                flags: query_lengths.flags,
            },
            PythonDeviceBufferArg {
                name: "kv_lengths",
                ptr: kv_lengths.ptr,
                bytes: kv_lengths.bytes,
                device_id: kv_lengths.device_id,
                flags: kv_lengths.flags,
            },
        ]);
    }
    let kwargs = if dynamic_suffix {
        vec![
            ("row_capacity", PythonKernelArg::Usize(capture_rows)),
            ("query_capacity", PythonKernelArg::Usize(capture_query_rows)),
            ("heads", PythonKernelArg::Usize(heads)),
            ("nope_dim", PythonKernelArg::Usize(nope_dim)),
            ("rope_dim", PythonKernelArg::Usize(rope_dim)),
            ("v_dim", PythonKernelArg::Usize(v_dim)),
            ("scale", PythonKernelArg::F64(scale as f64)),
        ]
    } else {
        vec![
            ("rows", PythonKernelArg::Usize(capture_rows)),
            (
                "query_row_offset",
                PythonKernelArg::Usize(capture_shape.query_row_offset),
            ),
            ("query_rows", PythonKernelArg::Usize(capture_query_rows)),
            ("heads", PythonKernelArg::Usize(heads)),
            ("nope_dim", PythonKernelArg::Usize(nope_dim)),
            ("rope_dim", PythonKernelArg::Usize(rope_dim)),
            ("v_dim", PythonKernelArg::Usize(v_dim)),
            ("scale", PythonKernelArg::F64(scale as f64)),
        ]
    };
    let program = if full_query {
        CoordinatorCudaGraphProgram::LayerFlashinferMlaRopeAttentionBf16
    } else {
        CoordinatorCudaGraphProgram::LayerFlashinferCudnnMlaRopeAttentionBf16Suffix
    };
    let mut capture_identity_parts = vec![
        q_nope_staging.ptr as usize,
        q_rope_staging.ptr as usize,
        k_nope_staging.ptr as usize,
        k_rope_staging.ptr as usize,
        value_staging.ptr as usize,
        q_buffer.ptr as usize,
        k_buffer.ptr as usize,
        workspace_buffer.ptr as usize,
        output_scratch.ptr as usize,
    ];
    if let Some(query_lengths) = query_lengths {
        capture_identity_parts.push(query_lengths.ptr as usize);
    }
    if let Some(kv_lengths) = kv_lengths {
        capture_identity_parts.push(kv_lengths.ptr as usize);
    }
    let capture_identity = mla_graph_capture_identity(&capture_identity_parts);
    let prepare_function = if dynamic_suffix {
        FLASHINFER_CUDNN_MLA_PREPARE_FUNCTION
    } else {
        FLASHINFER_MLA_PREPARE_FUNCTION
    };
    let capture_function = if dynamic_suffix {
        FLASHINFER_CUDNN_MLA_CAPTURE_FUNCTION
    } else {
        FLASHINFER_MLA_CAPTURE_FUNCTION
    };
    let graph_is_captured = slot.has_captured_graph_identity(program, signature, capture_identity);
    if !graph_is_captured && !coordinator_python_capture_startup_open() {
        return Ok(false);
    }
    if !graph_is_captured {
        slot.stream_synchronize()
            .with_context(|| format!("synchronizing {label} before prepare"))?;
        launch_python_graph_capture(PythonGraphCaptureLaunch {
            module: B12X_MLA_CAPTURE_MODULE,
            function: prepare_function,
            cuda_stream: slot.stream_ptr(),
            buffers: &buffers,
            kwargs: &kwargs,
        })
        .with_context(|| format!("preparing Python {label}"))?;
        slot.stream_synchronize()
            .with_context(|| format!("synchronizing prepared {label}"))?;
    }
    slot.capture_or_update_graph_exec(
        library,
        program,
        signature,
        capture_identity,
        |_library, cuda_stream, _workspace| {
            launch_python_graph_capture(PythonGraphCaptureLaunch {
                module: B12X_MLA_CAPTURE_MODULE,
                function: capture_function,
                cuda_stream,
                buffers: &buffers,
                kwargs: &kwargs,
            })
            .with_context(|| format!("capturing Python {label}"))?;
            Ok(())
        },
    )?;
    slot.launch_captured_graph_identity(library, program, signature, capture_identity)?;
    let output_row_bytes = heads
        .checked_mul(v_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("FlashInfer MLA output row bytes overflow")?;
    let output_source = device_buffer_byte_view(
        output_scratch,
        (if dynamic_suffix {
            0
        } else {
            capture_shape.query_prefix_padding
        })
        .checked_mul(output_row_bytes)
        .context("FlashInfer MLA output prefix offset overflow")?,
        output_bytes,
        "FlashInfer MLA real output rows",
    )?;
    if !inputs_are_staged {
        unsafe {
            library
                .copy_d2d_async(
                    output_buffer,
                    output_source,
                    output_bytes,
                    slot.stream_ptr(),
                )
                .with_context(|| format!("copying {label} output"))?;
        }
    }
    Ok(true)
}

fn mla_graph_capture_identity(parts: &[usize]) -> usize {
    parts.iter().fold(0xcbf29ce484222325_usize, |hash, part| {
        hash.wrapping_mul(0x100000001b3_usize) ^ part
    })
}

#[allow(clippy::too_many_arguments)]
fn flashinfer_mla_graph_signature(
    rows: usize,
    query_row_offset: usize,
    query_rows: usize,
    row_capacity: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    scale: f32,
) -> Result<CoordinatorCudaGraphSignature> {
    let capture_shape =
        flashinfer_mla_capture_shape(rows, query_row_offset, query_rows, row_capacity)?;
    let output_bytes = capture_shape
        .query_rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(v_dim))
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("FlashInfer MLA graph signature output byte count overflow")?;
    if query_row_offset == 0 && query_rows == rows {
        Ok(CoordinatorCudaGraphSignature::mla_rope_attention_bf16(
            output_bytes,
            capture_shape.rows,
            heads,
            nope_dim,
            rope_dim,
            v_dim,
            scale,
        ))
    } else {
        Ok(
            CoordinatorCudaGraphSignature::mla_rope_attention_bf16_suffix(
                output_bytes,
                capture_shape.rows,
                capture_shape.query_rows,
                heads,
                nope_dim,
                rope_dim,
                v_dim,
                scale,
            ),
        )
    }
}

fn flashinfer_mla_capture_shape(
    rows: usize,
    query_row_offset: usize,
    query_rows: usize,
    row_capacity: usize,
) -> Result<FlashinferMlaCaptureShape> {
    anyhow::ensure!(query_rows > 0, "FlashInfer MLA capture requires query rows");
    anyhow::ensure!(
        query_row_offset <= rows && query_rows == rows - query_row_offset,
        "FlashInfer MLA capture requires a terminal query suffix"
    );
    let full_query = query_row_offset == 0 && query_rows == rows;
    let capture_query_rows = if full_query {
        row_capacity
    } else {
        query_rows
            .max(FLASHINFER_MLA_SUFFIX_QUERY_FLOOR_ROWS.min(row_capacity))
            .checked_next_power_of_two()
            .context("FlashInfer MLA query row bucket overflow")?
    };
    let query_prefix_padding = capture_query_rows - query_rows;
    let (capture_rows, capture_query_row_offset, kv_prefix_padding, query_prefix_padding) =
        if full_query {
            (capture_query_rows, 0, 0, 0)
        } else {
            let kv_prefix_padding = row_capacity - rows;
            (
                row_capacity,
                row_capacity - capture_query_rows,
                kv_prefix_padding,
                query_prefix_padding,
            )
        };
    anyhow::ensure!(
        capture_rows <= row_capacity,
        "FlashInfer MLA padded rows {capture_rows} exceed graph capacity {row_capacity}"
    );
    Ok(FlashinferMlaCaptureShape {
        rows: capture_rows,
        query_row_offset: capture_query_row_offset,
        query_rows: capture_query_rows,
        kv_prefix_padding,
        query_prefix_padding,
    })
}

fn b12x_mla_rope_attention_bf16_supported(
    rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
) -> bool {
    coordinator_python_capture_enabled()
        && b12x_mla_rope_attention_bf16_shape_supported(rows, heads, nope_dim, rope_dim, v_dim)
}

fn b12x_mla_capture_module() -> String {
    env::var(GLMRT_B12X_MLA_MODULE_ENV).unwrap_or_else(|_| B12X_MLA_CAPTURE_MODULE.to_owned())
}

fn b12x_mla_capture_function() -> String {
    env::var(GLMRT_B12X_MLA_FUNCTION_ENV).unwrap_or_else(|_| B12X_MLA_CAPTURE_FUNCTION.to_owned())
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn capture_or_update_layer_b12x_mla_rope_attention_bf16_graph(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    signature: CoordinatorCudaGraphSignature,
    q_nope_buffer: GlmrtDeviceBuffer,
    q_rope_buffer: GlmrtDeviceBuffer,
    k_nope_buffer: GlmrtDeviceBuffer,
    k_rope_buffer: GlmrtDeviceBuffer,
    value_buffer: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    scale: f32,
    label: &'static str,
) -> Result<()> {
    validate_mla_rope_attention_device_buffers(
        rows,
        q_nope_buffer,
        q_rope_buffer,
        k_nope_buffer,
        k_rope_buffer,
        value_buffer,
        output_buffer,
        heads,
        nope_dim,
        rope_dim,
        v_dim,
    )?;
    let program = CoordinatorCudaGraphProgram::LayerB12xMlaRopeAttentionBf16;
    if !slot.has_captured_graph(program, signature) {
        let module = b12x_mla_capture_module();
        let function = b12x_mla_capture_function();
        slot.stream_synchronize()
            .with_context(|| format!("synchronizing {label} inputs before graph capture"))?;
        slot.capture_graph(
            library,
            program,
            signature,
            |_library, cuda_stream, _workspace| {
                let buffers = [
                    PythonDeviceBufferArg {
                        name: "q_nope",
                        ptr: q_nope_buffer.ptr,
                        bytes: q_nope_buffer.bytes,
                        device_id: q_nope_buffer.device_id,
                        flags: q_nope_buffer.flags,
                    },
                    PythonDeviceBufferArg {
                        name: "q_rope",
                        ptr: q_rope_buffer.ptr,
                        bytes: q_rope_buffer.bytes,
                        device_id: q_rope_buffer.device_id,
                        flags: q_rope_buffer.flags,
                    },
                    PythonDeviceBufferArg {
                        name: "k_nope",
                        ptr: k_nope_buffer.ptr,
                        bytes: k_nope_buffer.bytes,
                        device_id: k_nope_buffer.device_id,
                        flags: k_nope_buffer.flags,
                    },
                    PythonDeviceBufferArg {
                        name: "k_rope",
                        ptr: k_rope_buffer.ptr,
                        bytes: k_rope_buffer.bytes,
                        device_id: k_rope_buffer.device_id,
                        flags: k_rope_buffer.flags,
                    },
                    PythonDeviceBufferArg {
                        name: "values",
                        ptr: value_buffer.ptr,
                        bytes: value_buffer.bytes,
                        device_id: value_buffer.device_id,
                        flags: value_buffer.flags,
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
                    ("heads", PythonKernelArg::Usize(heads)),
                    ("nope_dim", PythonKernelArg::Usize(nope_dim)),
                    ("rope_dim", PythonKernelArg::Usize(rope_dim)),
                    ("v_dim", PythonKernelArg::Usize(v_dim)),
                    ("scale", PythonKernelArg::F64(scale as f64)),
                ];
                launch_python_graph_capture(PythonGraphCaptureLaunch {
                    module: module.as_str(),
                    function: function.as_str(),
                    cuda_stream,
                    buffers: &buffers,
                    kwargs: &kwargs,
                })
                .with_context(|| format!("capturing Python b12x {label}"))?;
                Ok(())
            },
        )?;
    }
    slot.launch_captured_graph(library, program, signature)
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn capture_or_update_layer_mla_rope_attention_bf16_graph(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    signature: CoordinatorCudaGraphSignature,
    q_nope_buffer: GlmrtDeviceBuffer,
    q_rope_buffer: GlmrtDeviceBuffer,
    k_nope_buffer: GlmrtDeviceBuffer,
    k_rope_buffer: GlmrtDeviceBuffer,
    value_buffer: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    scale: f32,
    label: &'static str,
) -> Result<()> {
    if !slot.has_captured_graph(
        CoordinatorCudaGraphProgram::LayerMlaRopeAttentionBf16,
        signature,
    ) {
        slot.stream_synchronize()
            .with_context(|| format!("synchronizing {label} inputs before graph capture"))?;
        slot.capture_graph(
            library,
            CoordinatorCudaGraphProgram::LayerMlaRopeAttentionBf16,
            signature,
            |library, cuda_stream, _workspace| unsafe {
                library
                    .cuda_mla_rope_attention_bf16_async(
                        q_nope_buffer,
                        q_rope_buffer,
                        k_nope_buffer,
                        k_rope_buffer,
                        value_buffer,
                        output_buffer,
                        rows,
                        heads,
                        nope_dim,
                        rope_dim,
                        v_dim,
                        scale,
                        cuda_stream,
                    )
                    .with_context(|| format!("capturing async CUDA {label}"))?;
                Ok(())
            },
        )?;
    } else {
        let (graph_raw, exec_raw) = slot
            .captured_graph_raw_handles(
                CoordinatorCudaGraphProgram::LayerMlaRopeAttentionBf16,
                signature,
            )
            .context(
                "coordinator CUDA graph slot lost captured MLA/RoPE attention graph before update",
            )?;
        unsafe {
            library
                .cuda_graph_update_mla_rope_attention_bf16_node(
                    graph_raw,
                    exec_raw,
                    0,
                    q_nope_buffer,
                    q_rope_buffer,
                    k_nope_buffer,
                    k_rope_buffer,
                    value_buffer,
                    output_buffer,
                    rows,
                    heads,
                    nope_dim,
                    rope_dim,
                    v_dim,
                    scale,
                )
                .with_context(|| format!("updating captured CUDA {label} graph node"))?;
        }
    }
    slot.launch_captured_graph(
        library,
        CoordinatorCudaGraphProgram::LayerMlaRopeAttentionBf16,
        signature,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn capture_or_update_layer_mla_kv_cache_unpack_bf16_graph(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    signature: CoordinatorCudaGraphSignature,
    payload_buffer: GlmrtDeviceBuffer,
    kv_latent_buffer: GlmrtDeviceBuffer,
    k_rope_buffer: GlmrtDeviceBuffer,
    dsa_key_buffer: Option<GlmrtDeviceBuffer>,
    rows: usize,
    kv_lora_rank: usize,
    rope_dim: usize,
    dsa_dim: usize,
    payload_stride_bytes: usize,
    label: &'static str,
) -> Result<()> {
    if !slot.has_captured_graph(
        CoordinatorCudaGraphProgram::LayerMlaKvCacheUnpackBf16,
        signature,
    ) {
        slot.stream_synchronize()
            .with_context(|| format!("synchronizing {label} inputs before graph capture"))?;
        slot.capture_graph(
            library,
            CoordinatorCudaGraphProgram::LayerMlaKvCacheUnpackBf16,
            signature,
            |library, cuda_stream, _workspace| unsafe {
                library
                    .cuda_mla_kv_cache_unpack_bf16_async(
                        payload_buffer,
                        kv_latent_buffer,
                        k_rope_buffer,
                        dsa_key_buffer,
                        rows,
                        kv_lora_rank,
                        rope_dim,
                        dsa_dim,
                        payload_stride_bytes,
                        cuda_stream,
                    )
                    .with_context(|| format!("capturing async CUDA {label}"))?;
                Ok(())
            },
        )?;
    } else {
        let (graph_raw, exec_raw) = slot
            .captured_graph_raw_handles(
                CoordinatorCudaGraphProgram::LayerMlaKvCacheUnpackBf16,
                signature,
            )
            .context(
                "coordinator CUDA graph slot lost captured MLA KV cache unpack graph before update",
            )?;
        unsafe {
            library
                .cuda_graph_update_mla_kv_cache_unpack_bf16_node(
                    graph_raw,
                    exec_raw,
                    0,
                    payload_buffer,
                    kv_latent_buffer,
                    k_rope_buffer,
                    dsa_key_buffer,
                    rows,
                    kv_lora_rank,
                    rope_dim,
                    dsa_dim,
                    payload_stride_bytes,
                )
                .with_context(|| format!("updating captured CUDA {label} graph node"))?;
        }
    }
    slot.launch_captured_graph(
        library,
        CoordinatorCudaGraphProgram::LayerMlaKvCacheUnpackBf16,
        signature,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn capture_or_update_layer_mla_kv_projected_split_bf16_graph(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    signature: CoordinatorCudaGraphSignature,
    projected_buffer: GlmrtDeviceBuffer,
    k_nope_buffer: GlmrtDeviceBuffer,
    value_buffer: GlmrtDeviceBuffer,
    rows: usize,
    heads: usize,
    nope_dim: usize,
    v_dim: usize,
    label: &'static str,
) -> Result<()> {
    if !slot.has_captured_graph(
        CoordinatorCudaGraphProgram::LayerMlaKvProjectedSplitBf16,
        signature,
    ) {
        slot.stream_synchronize()
            .with_context(|| format!("synchronizing {label} inputs before graph capture"))?;
        slot.capture_graph(
            library,
            CoordinatorCudaGraphProgram::LayerMlaKvProjectedSplitBf16,
            signature,
            |library, cuda_stream, _workspace| unsafe {
                library
                    .cuda_mla_kv_projected_split_bf16_async(
                        projected_buffer,
                        k_nope_buffer,
                        value_buffer,
                        rows,
                        heads,
                        nope_dim,
                        v_dim,
                        cuda_stream,
                    )
                    .with_context(|| format!("capturing async CUDA {label}"))?;
                Ok(())
            },
        )?;
    } else {
        let (graph_raw, exec_raw) = slot
            .captured_graph_raw_handles(
                CoordinatorCudaGraphProgram::LayerMlaKvProjectedSplitBf16,
                signature,
            )
            .context(
                "coordinator CUDA graph slot lost captured MLA KV projected split graph before update",
            )?;
        unsafe {
            library
                .cuda_graph_update_mla_kv_projected_split_bf16_node(
                    graph_raw,
                    exec_raw,
                    0,
                    projected_buffer,
                    k_nope_buffer,
                    value_buffer,
                    rows,
                    heads,
                    nope_dim,
                    v_dim,
                )
                .with_context(|| format!("updating captured CUDA {label} graph node"))?;
        }
    }
    slot.launch_captured_graph(
        library,
        CoordinatorCudaGraphProgram::LayerMlaKvProjectedSplitBf16,
        signature,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn validate_mla_rope_attention_device_buffers(
    rows: usize,
    q_nope_buffer: GlmrtDeviceBuffer,
    q_rope_buffer: GlmrtDeviceBuffer,
    k_nope_buffer: GlmrtDeviceBuffer,
    k_rope_buffer: GlmrtDeviceBuffer,
    value_buffer: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
) -> Result<()> {
    validate_mla_rope_attention_device_buffers_with_output_rows(
        rows,
        rows,
        q_nope_buffer,
        q_rope_buffer,
        k_nope_buffer,
        k_rope_buffer,
        value_buffer,
        output_buffer,
        heads,
        nope_dim,
        rope_dim,
        v_dim,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_mla_rope_attention_device_buffers_with_output_rows(
    rows: usize,
    output_rows: usize,
    q_nope_buffer: GlmrtDeviceBuffer,
    q_rope_buffer: GlmrtDeviceBuffer,
    k_nope_buffer: GlmrtDeviceBuffer,
    k_rope_buffer: GlmrtDeviceBuffer,
    value_buffer: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
) -> Result<()> {
    let q_nope_bytes = rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(nope_dim))
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 layer MLA/RoPE attention device-buffer q_nope bytes overflow usize")?;
    let q_rope_bytes = rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(rope_dim))
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 layer MLA/RoPE attention device-buffer q_rope bytes overflow usize")?;
    let k_rope_bytes = rows
        .checked_mul(rope_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 layer MLA/RoPE attention device-buffer k_rope bytes overflow usize")?;
    let value_bytes = rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(v_dim))
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 layer MLA/RoPE attention device-buffer value bytes overflow usize")?;
    let output_bytes = output_rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(v_dim))
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 layer MLA/RoPE attention device-buffer output bytes overflow usize")?;
    let buffers = [
        ("q_nope", q_nope_buffer, q_nope_bytes),
        ("q_rope", q_rope_buffer, q_rope_bytes),
        ("k_nope", k_nope_buffer, q_nope_bytes),
        ("k_rope", k_rope_buffer, k_rope_bytes),
        ("values", value_buffer, value_bytes),
        ("output", output_buffer, output_bytes),
    ];
    for (label, buffer, required_bytes) in buffers {
        if buffer.ptr.is_null() {
            anyhow::bail!("CUDA BF16 layer MLA/RoPE attention device-buffer {label} is null");
        }
        if buffer.bytes < required_bytes {
            anyhow::bail!(
                "CUDA BF16 layer MLA/RoPE attention device-buffer {label} has {} bytes, expected at least {required_bytes}",
                buffer.bytes
            );
        }
        if buffer.device_id != q_nope_buffer.device_id {
            anyhow::bail!(
                "CUDA BF16 layer MLA/RoPE attention device-buffer {label} is on CUDA device {}, expected {}",
                buffer.device_id,
                q_nope_buffer.device_id
            );
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn validate_mla_kv_cache_unpack_bf16_device_buffers(
    payload_buffer: GlmrtDeviceBuffer,
    kv_latent_buffer: GlmrtDeviceBuffer,
    k_rope_buffer: GlmrtDeviceBuffer,
    dsa_key_buffer: Option<GlmrtDeviceBuffer>,
    rows: usize,
    kv_lora_rank: usize,
    rope_dim: usize,
    dsa_dim: usize,
    payload_stride_bytes: usize,
) -> Result<()> {
    if rows == 0 || kv_lora_rank == 0 || rope_dim == 0 {
        anyhow::bail!(
            "CUDA BF16 layer MLA KV cache unpack requires nonzero rows/ranks, got rows={rows} kv_lora_rank={kv_lora_rank} rope_dim={rope_dim}"
        );
    }
    if payload_stride_bytes == 0 || payload_stride_bytes % std::mem::size_of::<u16>() != 0 {
        anyhow::bail!(
            "CUDA BF16 layer MLA KV cache unpack payload stride must be positive and BF16-aligned, got {payload_stride_bytes}"
        );
    }
    let packed_width = kv_lora_rank
        .checked_add(rope_dim)
        .and_then(|width| width.checked_add(dsa_dim))
        .context("CUDA BF16 layer MLA KV cache unpack packed width overflow usize")?;
    let payload_stride_values = payload_stride_bytes / std::mem::size_of::<u16>();
    if payload_stride_values < packed_width {
        anyhow::bail!(
            "CUDA BF16 layer MLA KV cache unpack payload stride is too small: stride_values={payload_stride_values} packed_width={packed_width}"
        );
    }
    let payload_bytes = rows
        .checked_mul(payload_stride_bytes)
        .context("CUDA BF16 layer MLA KV cache unpack payload bytes overflow usize")?;
    let kv_latent_bytes = rows
        .checked_mul(kv_lora_rank)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 layer MLA KV cache unpack kv_latent bytes overflow usize")?;
    let k_rope_bytes = rows
        .checked_mul(rope_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 layer MLA KV cache unpack k_rope bytes overflow usize")?;
    let buffers = [
        ("payload", payload_buffer, payload_bytes),
        ("kv_latent", kv_latent_buffer, kv_latent_bytes),
        ("k_rope", k_rope_buffer, k_rope_bytes),
    ];
    for (label, buffer, required_bytes) in buffers {
        if buffer.ptr.is_null() {
            anyhow::bail!("CUDA BF16 layer MLA KV cache unpack device-buffer {label} is null");
        }
        if buffer.bytes < required_bytes {
            anyhow::bail!(
                "CUDA BF16 layer MLA KV cache unpack device-buffer {label} has {} bytes, expected at least {required_bytes}",
                buffer.bytes
            );
        }
        if buffer.device_id != payload_buffer.device_id {
            anyhow::bail!(
                "CUDA BF16 layer MLA KV cache unpack device-buffer {label} is on CUDA device {}, expected {}",
                buffer.device_id,
                payload_buffer.device_id
            );
        }
    }
    if dsa_dim == 0 {
        return Ok(());
    }
    let dsa_key_buffer =
        dsa_key_buffer.context("CUDA BF16 layer MLA KV cache unpack dsa_key buffer is required")?;
    let dsa_key_bytes = rows
        .checked_mul(dsa_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 layer MLA KV cache unpack dsa_key bytes overflow usize")?;
    if dsa_key_buffer.ptr.is_null() {
        anyhow::bail!("CUDA BF16 layer MLA KV cache unpack device-buffer dsa_key is null");
    }
    if dsa_key_buffer.bytes < dsa_key_bytes {
        anyhow::bail!(
            "CUDA BF16 layer MLA KV cache unpack device-buffer dsa_key has {} bytes, expected at least {dsa_key_bytes}",
            dsa_key_buffer.bytes
        );
    }
    if dsa_key_buffer.device_id != payload_buffer.device_id {
        anyhow::bail!(
            "CUDA BF16 layer MLA KV cache unpack device-buffer dsa_key is on CUDA device {}, expected {}",
            dsa_key_buffer.device_id,
            payload_buffer.device_id
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn validate_mla_kv_projected_split_bf16_device_buffers(
    projected_buffer: GlmrtDeviceBuffer,
    k_nope_buffer: GlmrtDeviceBuffer,
    value_buffer: GlmrtDeviceBuffer,
    rows: usize,
    heads: usize,
    nope_dim: usize,
    v_dim: usize,
) -> Result<()> {
    if rows == 0 || heads == 0 || nope_dim == 0 || v_dim == 0 {
        anyhow::bail!(
            "CUDA BF16 layer MLA KV projected split requires nonzero shape, got rows={rows} heads={heads} nope_dim={nope_dim} v_dim={v_dim}"
        );
    }
    let row_heads = rows
        .checked_mul(heads)
        .context("CUDA BF16 layer MLA KV projected split row-head count overflow usize")?;
    let projected_width = nope_dim
        .checked_add(v_dim)
        .context("CUDA BF16 layer MLA KV projected split output width overflow usize")?;
    let projected_bytes = row_heads
        .checked_mul(projected_width)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 layer MLA KV projected split input bytes overflow usize")?;
    let k_nope_bytes = row_heads
        .checked_mul(nope_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 layer MLA KV projected split k_nope bytes overflow usize")?;
    let value_bytes = row_heads
        .checked_mul(v_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("CUDA BF16 layer MLA KV projected split value bytes overflow usize")?;
    let buffers = [
        ("projected", projected_buffer, projected_bytes),
        ("k_nope", k_nope_buffer, k_nope_bytes),
        ("values", value_buffer, value_bytes),
    ];
    for (label, buffer, required_bytes) in buffers {
        if buffer.ptr.is_null() {
            anyhow::bail!("CUDA BF16 layer MLA KV projected split device-buffer {label} is null");
        }
        if buffer.bytes < required_bytes {
            anyhow::bail!(
                "CUDA BF16 layer MLA KV projected split device-buffer {label} has {} bytes, expected at least {required_bytes}",
                buffer.bytes
            );
        }
        if buffer.device_id != projected_buffer.device_id {
            anyhow::bail!(
                "CUDA BF16 layer MLA KV projected split device-buffer {label} is on CUDA device {}, expected {}",
                buffer.device_id,
                projected_buffer.device_id
            );
        }
    }
    Ok(())
}

pub(in crate::commands::real_full) fn mla_rope_attention_graph_nope_bytes(
    graph_key: &CoordinatorGraphKey,
    heads: usize,
    nope_dim: usize,
    context: &str,
) -> Result<usize> {
    graph_key
        .row_bucket
        .row_capacity
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(nope_dim))
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .with_context(|| format!("{context} no-RPE graph buffer bytes overflow usize"))
}

pub(in crate::commands::real_full) fn mla_rope_attention_graph_q_rope_bytes(
    graph_key: &CoordinatorGraphKey,
    heads: usize,
    rope_dim: usize,
    context: &str,
) -> Result<usize> {
    graph_key
        .row_bucket
        .row_capacity
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(rope_dim))
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .with_context(|| format!("{context} q_rope graph buffer bytes overflow usize"))
}

pub(in crate::commands::real_full) fn mla_rope_attention_graph_k_rope_bytes(
    graph_key: &CoordinatorGraphKey,
    rope_dim: usize,
    context: &str,
) -> Result<usize> {
    graph_key
        .row_bucket
        .row_capacity
        .checked_mul(rope_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .with_context(|| format!("{context} k_rope graph buffer bytes overflow usize"))
}

pub(in crate::commands::real_full) fn mla_rope_attention_graph_value_bytes(
    graph_key: &CoordinatorGraphKey,
    heads: usize,
    v_dim: usize,
    context: &str,
) -> Result<usize> {
    graph_key
        .row_bucket
        .row_capacity
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(v_dim))
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .with_context(|| format!("{context} value graph buffer bytes overflow usize"))
}

pub(in crate::commands::real_full) fn mla_rope_attention_graph_signature(
    graph_key: &CoordinatorGraphKey,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    scale: f32,
) -> CoordinatorCudaGraphSignature {
    CoordinatorCudaGraphSignature::mla_rope_attention_bf16(
        graph_key.row_bucket.row_capacity * heads * v_dim * std::mem::size_of::<u16>(),
        graph_key.row_bucket.row_capacity,
        heads,
        nope_dim,
        rope_dim,
        v_dim,
        scale,
    )
}

pub(in crate::commands::real_full) fn mla_rope_attention_suffix_graph_signature(
    graph_key: &CoordinatorGraphKey,
    query_rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    scale: f32,
) -> CoordinatorCudaGraphSignature {
    CoordinatorCudaGraphSignature::mla_rope_attention_bf16_suffix(
        graph_key.row_bucket.row_capacity * heads * v_dim * std::mem::size_of::<u16>(),
        graph_key.row_bucket.row_capacity,
        query_rows,
        heads,
        nope_dim,
        rope_dim,
        v_dim,
        scale,
    )
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_causal_attention_rows(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    rows: usize,
    heads: usize,
    qk_dim: usize,
    v_dim: usize,
    scale: f32,
) -> Result<CausalAttentionOutput> {
    let library = cuda_native_library()?;
    let qk_bytes = std::mem::size_of_val(queries);
    let value_bytes = std::mem::size_of_val(values);
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let query_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        qk_bytes,
        "causal attention queries",
    )?;
    let key_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::B,
        qk_bytes,
        "causal attention keys",
    )?;
    let value_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        value_bytes,
        "causal attention values",
    )?;
    let output_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        value_bytes,
        "causal attention output",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            f32_bytes(queries),
            "causal attention queries",
        )
        .context("copying causal attention queries to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::B,
            f32_bytes(keys),
            "causal attention keys",
        )
        .context("copying causal attention keys to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::C,
            f32_bytes(values),
            "causal attention values",
        )
        .context("copying causal attention values to device")?;
    library
        .cuda_causal_attention_f32(
            query_buffer,
            key_buffer,
            value_buffer,
            output_buffer,
            rows,
            heads,
            qk_dim,
            v_dim,
            scale,
        )
        .context("executing CUDA causal attention")?;
    let mut out_bytes = vec![0_u8; value_bytes];
    library
        .copy_d2h(&mut out_bytes, output_buffer)
        .context("copying causal attention output to host")?;

    Ok(CausalAttentionOutput {
        values: f32_vec_from_bytes(&out_bytes)?,
        backend: CUDA_REFERENCE_CAUSAL_ATTENTION_BACKEND,
    })
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_causal_attention_rows_bf16(
    queries_bf16: &[u8],
    keys_bf16: &[u8],
    values_bf16: &[u8],
    rows: usize,
    heads: usize,
    qk_dim: usize,
    v_dim: usize,
    scale: f32,
) -> Result<CausalAttentionOutput> {
    let library = cuda_native_library()?;
    let qk_bytes = queries_bf16.len();
    let value_bytes = values_bf16.len();
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let query_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        qk_bytes,
        "BF16 causal attention queries",
    )?;
    let key_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::B,
        qk_bytes,
        "BF16 causal attention keys",
    )?;
    let value_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        value_bytes,
        "BF16 causal attention values",
    )?;
    let output_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        value_bytes,
        "BF16 causal attention output",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            queries_bf16,
            "BF16 causal attention queries",
        )
        .context("copying BF16 causal attention queries to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::B,
            keys_bf16,
            "BF16 causal attention keys",
        )
        .context("copying BF16 causal attention keys to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::C,
            values_bf16,
            "BF16 causal attention values",
        )
        .context("copying BF16 causal attention values to device")?;
    library
        .cuda_causal_attention_bf16(
            query_buffer,
            key_buffer,
            value_buffer,
            output_buffer,
            rows,
            heads,
            qk_dim,
            v_dim,
            scale,
        )
        .context("executing CUDA BF16 causal attention")?;
    let mut out_bytes = vec![0_u8; value_bytes];
    library
        .copy_d2h(&mut out_bytes, output_buffer)
        .context("copying BF16 causal attention output to host")?;

    Ok(CausalAttentionOutput {
        values: bf16_values_to_f32(&out_bytes),
        backend: CUDA_REFERENCE_CAUSAL_ATTENTION_BF16_BACKEND,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_causal_attention_rows_bf16_for_layer(
    layer_id: usize,
    queries_bf16: &[u8],
    keys_bf16: &[u8],
    values_bf16: &[u8],
    rows: usize,
    heads: usize,
    qk_dim: usize,
    v_dim: usize,
    scale: f32,
) -> Result<CausalAttentionOutput> {
    let graph_key = coord_attention_graph_key_for_layer_rows(layer_id, rows)?;
    let value_bytes = values_bf16.len();
    let qk_graph_bytes = causal_attention_graph_qk_bytes(
        &graph_key,
        heads,
        qk_dim,
        "CUDA BF16 layer causal attention graph-slot",
    )?;
    let value_graph_bytes = causal_attention_graph_value_bytes(
        &graph_key,
        heads,
        v_dim,
        "CUDA BF16 layer causal attention graph-slot",
    )?;
    let signature = causal_attention_graph_signature(&graph_key, heads, qk_dim, v_dim, scale);
    with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let query_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            qk_graph_bytes,
            "BF16 layer causal attention queries",
        )?;
        let key_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::B,
            qk_graph_bytes,
            "BF16 layer causal attention keys",
        )?;
        let value_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            value_graph_bytes,
            "BF16 layer causal attention values",
        )?;
        let output_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            value_graph_bytes,
            "BF16 layer causal attention output",
        )?;

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::A,
                queries_bf16,
                "BF16 layer causal attention queries",
                cuda_stream,
            )
            .context("async copying BF16 layer causal attention queries to device")?;
        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::B,
                keys_bf16,
                "BF16 layer causal attention keys",
                cuda_stream,
            )
            .context("async copying BF16 layer causal attention keys to device")?;
        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::C,
                values_bf16,
                "BF16 layer causal attention values",
                cuda_stream,
            )
            .context("async copying BF16 layer causal attention values to device")?;
        capture_or_update_layer_causal_attention_bf16_graph(
            library,
            slot,
            signature,
            query_buffer,
            key_buffer,
            value_buffer,
            output_buffer,
            rows,
            heads,
            qk_dim,
            v_dim,
            scale,
            "BF16 layer causal attention",
        )?;
        let mut out_bytes = vec![0_u8; value_bytes];
        unsafe {
            library
                .copy_d2h_async(&mut out_bytes, output_buffer, cuda_stream)
                .context("async copying BF16 layer causal attention output to host")?;
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing BF16 layer causal attention graph slot stream")?;
        }

        Ok(CausalAttentionOutput {
            values: bf16_values_to_f32(&out_bytes),
            backend: CUDA_REFERENCE_CAUSAL_ATTENTION_BF16_BACKEND,
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn capture_or_update_layer_causal_attention_bf16_graph(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    signature: CoordinatorCudaGraphSignature,
    query_buffer: GlmrtDeviceBuffer,
    key_buffer: GlmrtDeviceBuffer,
    value_buffer: GlmrtDeviceBuffer,
    output_buffer: GlmrtDeviceBuffer,
    rows: usize,
    heads: usize,
    qk_dim: usize,
    v_dim: usize,
    scale: f32,
    label: &'static str,
) -> Result<()> {
    if !slot.has_captured_graph(
        CoordinatorCudaGraphProgram::LayerCausalAttentionBf16,
        signature,
    ) {
        slot.stream_synchronize()
            .with_context(|| format!("synchronizing {label} inputs before graph capture"))?;
        slot.capture_graph(
            library,
            CoordinatorCudaGraphProgram::LayerCausalAttentionBf16,
            signature,
            |library, cuda_stream, _workspace| unsafe {
                library
                    .cuda_causal_attention_bf16_async(
                        query_buffer,
                        key_buffer,
                        value_buffer,
                        output_buffer,
                        rows,
                        heads,
                        qk_dim,
                        v_dim,
                        scale,
                        cuda_stream,
                    )
                    .with_context(|| format!("capturing async CUDA {label}"))?;
                Ok(())
            },
        )?;
    } else {
        let (graph_raw, exec_raw) = slot
            .captured_graph_raw_handles(
                CoordinatorCudaGraphProgram::LayerCausalAttentionBf16,
                signature,
            )
            .context(
                "coordinator CUDA graph slot lost captured causal attention graph before update",
            )?;
        unsafe {
            library
                .cuda_graph_update_causal_attention_bf16_node(
                    graph_raw,
                    exec_raw,
                    0,
                    query_buffer,
                    key_buffer,
                    value_buffer,
                    output_buffer,
                    rows,
                    heads,
                    qk_dim,
                    v_dim,
                    scale,
                )
                .with_context(|| format!("updating captured CUDA {label} graph node"))?;
        }
    }
    slot.launch_captured_graph(
        library,
        CoordinatorCudaGraphProgram::LayerCausalAttentionBf16,
        signature,
    )
}

pub(in crate::commands::real_full) fn causal_attention_graph_qk_bytes(
    graph_key: &CoordinatorGraphKey,
    heads: usize,
    qk_dim: usize,
    context: &str,
) -> Result<usize> {
    graph_key
        .row_bucket
        .row_capacity
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(qk_dim))
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .with_context(|| format!("{context} q/k graph buffer bytes overflow usize"))
}

pub(in crate::commands::real_full) fn causal_attention_graph_value_bytes(
    graph_key: &CoordinatorGraphKey,
    heads: usize,
    v_dim: usize,
    context: &str,
) -> Result<usize> {
    graph_key
        .row_bucket
        .row_capacity
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(v_dim))
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .with_context(|| format!("{context} value graph buffer bytes overflow usize"))
}

pub(in crate::commands::real_full) fn causal_attention_graph_signature(
    graph_key: &CoordinatorGraphKey,
    heads: usize,
    qk_dim: usize,
    v_dim: usize,
    scale: f32,
) -> CoordinatorCudaGraphSignature {
    CoordinatorCudaGraphSignature::causal_attention_bf16(
        graph_key.row_bucket.row_capacity * heads * v_dim * std::mem::size_of::<u16>(),
        graph_key.row_bucket.row_capacity,
        heads,
        qk_dim,
        v_dim,
        scale,
    )
}

pub(in crate::commands::real_full) fn is_glm52_attention_linear_weight_subpath(
    subpath: &str,
) -> bool {
    [
        "self_attn.q_a_proj.weight",
        "self_attn.q_b_proj.weight",
        "self_attn.kv_a_proj_with_mqa.weight",
        "self_attn.kv_b_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.indexer.weights_proj.weight",
        "self_attn.indexer.wk.weight",
        "self_attn.indexer.wq_b.weight",
    ]
    .iter()
    .any(|prefix| subpath.starts_with(prefix))
}

#[cfg(test)]
mod flashinfer_capture_shape_tests {
    use super::{
        coord_attention_graph_key_for_layer_rows,
        flashinfer_direct_packed_fp8_mla_capacity_supported,
        flashinfer_glm52_attention_heads_supported,
        flashinfer_glm_dsa_sparse_mla_prefill_device_buffers, flashinfer_mla_capture_shape,
        flashinfer_mla_graph_signature, glm_dsa_index_source_layer,
        glm_dsa_sparse_mla_attention_topk, glm_dsa_sparse_mla_query_bucket, native_library_path,
        native_library_version_has_cuda, parse_flashinfer_cudnn_mla_suffix_query_capacity,
        stage_flashinfer_hidden_projection, with_coordinator_cuda_graph_slot,
        CoordinatorCudaGraphProgram, FlashinferGlmDsaSparseMlaPrefillInput,
        FlashinferMlaCaptureShape, FlashinferMlaHiddenProjection, OwnedCoordinatorDeviceBuffer,
        FLASHINFER_PACKED_FP8_MLA_ROW_BYTES, SPARKINFER_GLM_DSA_SPARSE_NVFP4_MLA_BACKEND,
    };
    use crate::commands::real_full::coordinator_kernels::cuda_native_library;
    use crate::python_graph_capture::coordinator_python_capture_test_override;
    use anyhow::{Context, Result};
    use glmrt_core::{
        KvCacheDType, GLM52_MLA_KV_LORA_RANK, GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN,
        GLM52_MLA_ROPE_THETA,
    };
    use glmrt_ffi::GlmrtDeviceBuffer;

    #[test]
    fn glm52_flashinfer_accepts_full_and_tp4_head_counts() {
        assert!(flashinfer_glm52_attention_heads_supported(64));
        assert!(flashinfer_glm52_attention_heads_supported(16));
        assert!(!flashinfer_glm52_attention_heads_supported(8));
        assert!(!flashinfer_glm52_attention_heads_supported(32));
    }

    #[test]
    fn direct_packed_mla_capacity_requires_complete_physical_pages() {
        const ROW_BYTES: usize = 656;
        assert!(flashinfer_direct_packed_fp8_mla_capacity_supported(
            64 * ROW_BYTES
        ));
        assert!(!flashinfer_direct_packed_fp8_mla_capacity_supported(
            65 * ROW_BYTES
        ));
        assert!(!flashinfer_direct_packed_fp8_mla_capacity_supported(
            64 * ROW_BYTES + 1
        ));
    }

    #[test]
    fn nvfp4_staged_sparse_mla_uses_smallest_supported_live_topk_bucket() {
        assert_eq!(
            glm_dsa_sparse_mla_attention_topk(KvCacheDType::Nvfp4, 1, 1),
            128
        );
        assert_eq!(
            glm_dsa_sparse_mla_attention_topk(KvCacheDType::Nvfp4, 64, 128),
            128
        );
        assert_eq!(
            glm_dsa_sparse_mla_attention_topk(KvCacheDType::Nvfp4, 64, 129),
            512
        );
        assert_eq!(
            glm_dsa_sparse_mla_attention_topk(KvCacheDType::Nvfp4, 64, 513),
            1024
        );
        assert_eq!(
            glm_dsa_sparse_mla_attention_topk(KvCacheDType::Nvfp4, 64, 1025),
            2048
        );
        assert_eq!(
            glm_dsa_sparse_mla_attention_topk(KvCacheDType::Nvfp4, 128, 16),
            2048
        );
        assert_eq!(
            glm_dsa_sparse_mla_attention_topk(KvCacheDType::Fp8, 1, 16),
            2048
        );
    }

    #[test]
    fn nvfp4_sparse_mla_retains_exact_small_recurrent_query_buckets() {
        for rows in 1_usize..=16 {
            assert_eq!(
                glm_dsa_sparse_mla_query_bucket(KvCacheDType::Nvfp4, rows),
                Some(rows.next_power_of_two())
            );
        }
        assert_eq!(
            glm_dsa_sparse_mla_query_bucket(KvCacheDType::Fp8, 2),
            Some(8)
        );
        assert_eq!(
            glm_dsa_sparse_mla_query_bucket(KvCacheDType::Fp8, 4),
            Some(8)
        );
    }

    #[test]
    fn glm_dsa_shared_layers_reuse_the_immediately_preceding_full_indexer() {
        for layer in 0..78 {
            let (source, full) = glm_dsa_index_source_layer(layer).unwrap();
            if matches!(layer, 0 | 1 | 2) || (layer >= 6 && (layer - 6) % 4 == 0) {
                assert_eq!((source, full), (layer, true), "layer {layer}");
            } else {
                assert!(!full, "layer {layer}");
                assert!(source < layer, "layer {layer}");
                assert!(matches!(source, 2) || (source >= 6 && (source - 6) % 4 == 0));
                assert!(layer - source <= 3, "layer {layer}");
            }
        }
        assert_eq!(glm_dsa_index_source_layer(78), Some((78, true)));
        assert_eq!(glm_dsa_index_source_layer(79), None);
    }

    #[test]
    fn direct_glm_dsa_sparse_mla_captures_full_and_shared_graphs() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let library = cuda_native_library()?;
        if !native_library_version_has_cuda(&library.version()?) {
            return Ok(());
        }
        let _python_capture_guard = coordinator_python_capture_test_override(true);

        const QUERY_ROWS: usize = 512;
        const HEADS: usize = 64;
        const NOPE_DIM: usize = 192;
        const ROPE_DIM: usize = 64;
        const V_DIM: usize = 256;
        const DSA_HEADS: usize = 32;
        const DSA_DIM: usize = 128;
        const MAX_TOKENS: usize = 131_072;
        const MLA_ROW_BYTES: usize = 656;
        const DSA_PAGE_BYTES: usize = 8_448;

        let q_nope = upload_test_buffer(
            library,
            &vec![0_u8; QUERY_ROWS * HEADS * NOPE_DIM * 2],
            "direct DSA test q-nope",
        )?;
        let q_rope = upload_test_buffer(
            library,
            &vec![0_u8; QUERY_ROWS * HEADS * ROPE_DIM * 2],
            "direct DSA test q-rope",
        )?;
        let one_bf16 = ((1.0_f32.to_bits() >> 16) as u16).to_ne_bytes();
        let mut dsa_query_bytes = vec![0_u8; QUERY_ROWS * DSA_HEADS * DSA_DIM * 2];
        for value in dsa_query_bytes.chunks_exact_mut(2) {
            value.copy_from_slice(&one_bf16);
        }
        let dsa_query =
            upload_test_buffer(library, &dsa_query_bytes, "direct DSA test projected query")?;
        let mut dsa_weights_bytes = vec![0_u8; QUERY_ROWS * DSA_HEADS * 2];
        for value in dsa_weights_bytes.chunks_exact_mut(2) {
            value.copy_from_slice(&one_bf16);
        }
        let dsa_weights = upload_test_buffer(
            library,
            &dsa_weights_bytes,
            "direct DSA test projected head weights",
        )?;
        let positions = (0..QUERY_ROWS as u32).collect::<Vec<_>>();
        let positions_bytes = unsafe {
            std::slice::from_raw_parts(
                positions.as_ptr().cast::<u8>(),
                std::mem::size_of_val(positions.as_slice()),
            )
        };
        let positions = upload_test_buffer(library, positions_bytes, "direct DSA test positions")?;

        let mut packed_kv_bytes = vec![0_u8; MAX_TOKENS * MLA_ROW_BYTES];
        for row in packed_kv_bytes.chunks_exact_mut(MLA_ROW_BYTES) {
            for scale in 0..4 {
                let offset = 512 + scale * 4;
                row[offset..offset + 4].copy_from_slice(&1.0_f32.to_ne_bytes());
            }
        }
        let packed_kv = upload_test_buffer(
            library,
            &packed_kv_bytes,
            "direct DSA test packed MLA cache",
        )?;
        let pages = MAX_TOKENS / 64;
        let mut index_k_bytes = vec![0_u8; pages * DSA_PAGE_BYTES];
        for (page_index, page) in index_k_bytes.chunks_exact_mut(DSA_PAGE_BYTES).enumerate() {
            for row in 0..64 {
                let physical_row = page_index * 64 + row;
                let fp8_value = [0x30_u8, 0x38, 0x40, 0xb8][physical_row % 4];
                page[row * DSA_DIM..(row + 1) * DSA_DIM].fill(fp8_value);
                let offset = 64 * DSA_DIM + row * 4;
                page[offset..offset + 4].copy_from_slice(&1.0_f32.to_ne_bytes());
            }
        }
        let index_k_cache = upload_test_buffer(
            library,
            &index_k_bytes,
            "direct DSA test packed index-K cache",
        )?;
        let kv_b_weight = upload_test_buffer(
            library,
            &vec![0_u8; HEADS * (NOPE_DIM + V_DIM) * GLM52_MLA_KV_LORA_RANK * 2],
            "direct DSA test KV-B weight",
        )?;

        let common = FlashinferGlmDsaSparseMlaPrefillInput {
            layer_id: 2,
            q_nope: q_nope.buffer,
            q_rope: q_rope.buffer,
            dsa_query: None,
            dsa_weights: None,
            positions: positions.buffer,
            packed_kv: packed_kv.buffer,
            kv_dtype: KvCacheDType::Fp8,
            kv_row_stride_bytes: FLASHINFER_PACKED_FP8_MLA_ROW_BYTES,
            index_k_cache: None,
            kv_b_weight: kv_b_weight.buffer,
            hidden_projection: None,
            total_rows: QUERY_ROWS,
            prefix_rows: 0,
            query_rows: QUERY_ROWS,
            heads: HEADS,
            nope_dim: NOPE_DIM,
            rope_dim: ROPE_DIM,
            v_dim: V_DIM,
            rank: GLM52_MLA_KV_LORA_RANK,
            max_tokens: MAX_TOKENS,
            physical_token_base: 0,
            physical_page_table: None,
            theta: GLM52_MLA_ROPE_THETA,
            scale: 1.0 / ((NOPE_DIM + ROPE_DIM) as f32).sqrt(),
        };
        for query_rows in [1, 32, QUERY_ROWS] {
            let current = FlashinferGlmDsaSparseMlaPrefillInput {
                total_rows: query_rows,
                query_rows,
                ..common
            };
            let full = flashinfer_glm_dsa_sparse_mla_prefill_device_buffers(current)?;
            let shared = flashinfer_glm_dsa_sparse_mla_prefill_device_buffers(
                FlashinferGlmDsaSparseMlaPrefillInput {
                    layer_id: 3,
                    dsa_query: None,
                    dsa_weights: None,
                    index_k_cache: None,
                    ..current
                },
            )?;
            for output in [full.output, shared.output] {
                let mut bytes = vec![0_u8; output.bytes];
                library.copy_d2h(&mut bytes, output)?;
                for value in bytes.chunks_exact(2) {
                    let bits = u16::from_ne_bytes(value.try_into().expect("BF16 chunk width"));
                    let value = f32::from_bits((bits as u32) << 16);
                    assert!(value.is_finite());
                    assert!(value.abs() <= 1.0e-3, "expected zero output, got {value}");
                }
            }

            let graph_key = coord_attention_graph_key_for_layer_rows(2, query_rows)?;
            with_coordinator_cuda_graph_slot(&graph_key, |_library, slot| {
                assert!(slot.captured_graphs.iter().any(|entry| {
                    entry.program == CoordinatorCudaGraphProgram::LayerGlmDsaSparseMlaPrefillFull
                }));
                assert!(slot.captured_graphs.iter().any(|entry| {
                    entry.program == CoordinatorCudaGraphProgram::LayerGlmDsaSparseMlaPrefillShared
                }));
                slot.captured_graphs.retain(|entry| {
                    !matches!(
                        entry.program,
                        CoordinatorCudaGraphProgram::LayerGlmDsaSparseMlaPrefillFull
                            | CoordinatorCudaGraphProgram::LayerGlmDsaSparseMlaPrefillShared
                    )
                });
                Ok(())
            })?;
        }

        let mut packed_nvfp4_bytes = vec![0_u8; MAX_TOKENS * GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN];
        for row in packed_nvfp4_bytes.chunks_exact_mut(GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN) {
            row[256..288].fill(0x38);
        }
        let packed_nvfp4 = upload_test_buffer(
            library,
            &packed_nvfp4_bytes,
            "direct DSA test packed NVFP4 MLA cache",
        )?;
        let nvfp4_common = FlashinferGlmDsaSparseMlaPrefillInput {
            packed_kv: packed_nvfp4.buffer,
            kv_dtype: KvCacheDType::Nvfp4,
            kv_row_stride_bytes: GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN,
            ..common
        };
        for query_rows in [1, 32, 128] {
            let current = FlashinferGlmDsaSparseMlaPrefillInput {
                total_rows: query_rows,
                query_rows,
                ..nvfp4_common
            };
            let full = flashinfer_glm_dsa_sparse_mla_prefill_device_buffers(current)?;
            let shared = flashinfer_glm_dsa_sparse_mla_prefill_device_buffers(
                FlashinferGlmDsaSparseMlaPrefillInput {
                    layer_id: 3,
                    ..current
                },
            )?;
            assert_eq!(full.backend, SPARKINFER_GLM_DSA_SPARSE_NVFP4_MLA_BACKEND);
            assert_eq!(shared.backend, SPARKINFER_GLM_DSA_SPARSE_NVFP4_MLA_BACKEND);
            for output in [full.output, shared.output] {
                let mut bytes = vec![0_u8; output.bytes];
                library.copy_d2h(&mut bytes, output)?;
                for value in bytes.chunks_exact(2) {
                    let bits = u16::from_ne_bytes(value.try_into().expect("BF16 chunk width"));
                    let value = f32::from_bits((bits as u32) << 16);
                    assert!(value.is_finite());
                    assert!(value.abs() <= 1.0e-3, "expected zero output, got {value}");
                }
            }

            let graph_key = coord_attention_graph_key_for_layer_rows(2, query_rows)?;
            with_coordinator_cuda_graph_slot(&graph_key, |_library, slot| {
                slot.captured_graphs.retain(|entry| {
                    !matches!(
                        entry.program,
                        CoordinatorCudaGraphProgram::LayerGlmDsaSparseMlaPrefillFull
                            | CoordinatorCudaGraphProgram::LayerGlmDsaSparseMlaPrefillShared
                    )
                });
                Ok(())
            })?;
        }

        // A small prompt suffix and its decode row can be scheduled together.
        // They need distinct unbanked selector slots so the decode selection
        // cannot overwrite the suffix before its shared layers consume it.
        let small_prefill = FlashinferGlmDsaSparseMlaPrefillInput {
            total_rows: 8,
            query_rows: 8,
            ..common
        };
        let decode = FlashinferGlmDsaSparseMlaPrefillInput {
            total_rows: 1,
            query_rows: 1,
            ..common
        };
        flashinfer_glm_dsa_sparse_mla_prefill_device_buffers(small_prefill)?;
        flashinfer_glm_dsa_sparse_mla_prefill_device_buffers(decode)?;
        flashinfer_glm_dsa_sparse_mla_prefill_device_buffers(
            FlashinferGlmDsaSparseMlaPrefillInput {
                layer_id: 3,
                ..small_prefill
            },
        )?;
        flashinfer_glm_dsa_sparse_mla_prefill_device_buffers(
            FlashinferGlmDsaSparseMlaPrefillInput {
                layer_id: 3,
                ..decode
            },
        )?;

        // A layer-major prefill can publish several source-layer chunks before
        // the first shared-index layer consumes any of them. Verify that the
        // first selection remains addressable after the active selector buffer
        // has advanced to a later suffix.
        let delayed_positions = upload_test_buffer(
            library,
            &(QUERY_ROWS as u32..(2 * QUERY_ROWS) as u32)
                .flat_map(u32::to_ne_bytes)
                .collect::<Vec<_>>(),
            "direct DSA delayed shared-selection positions",
        )?;
        let first_chunk = FlashinferGlmDsaSparseMlaPrefillInput {
            total_rows: QUERY_ROWS,
            query_rows: QUERY_ROWS,
            ..common
        };
        let second_chunk = FlashinferGlmDsaSparseMlaPrefillInput {
            positions: delayed_positions.buffer,
            total_rows: 2 * QUERY_ROWS,
            prefix_rows: QUERY_ROWS,
            query_rows: QUERY_ROWS,
            ..common
        };
        flashinfer_glm_dsa_sparse_mla_prefill_device_buffers(first_chunk)?;
        flashinfer_glm_dsa_sparse_mla_prefill_device_buffers(second_chunk)?;
        flashinfer_glm_dsa_sparse_mla_prefill_device_buffers(
            FlashinferGlmDsaSparseMlaPrefillInput {
                layer_id: 3,
                ..first_chunk
            },
        )?;
        flashinfer_glm_dsa_sparse_mla_prefill_device_buffers(
            FlashinferGlmDsaSparseMlaPrefillInput {
                layer_id: 3,
                ..second_chunk
            },
        )?;

        // Continuous batching can run two request-local full indexers before
        // either request reaches the shared DSA layers. Both selections must
        // be banked immediately; treating the last one as an implicit active
        // selection makes the result depend on request-major execution.
        super::reset_glm_dsa_sparse_mla_transient_state()?;
        let interleaved_first = FlashinferGlmDsaSparseMlaPrefillInput {
            total_rows: 4_097,
            prefix_rows: 4_096,
            query_rows: 1,
            dsa_query: Some(dsa_query.buffer),
            dsa_weights: Some(dsa_weights.buffer),
            index_k_cache: Some(index_k_cache.buffer),
            ..common
        };
        let interleaved_second = FlashinferGlmDsaSparseMlaPrefillInput {
            physical_token_base: 65_536,
            ..interleaved_first
        };
        flashinfer_glm_dsa_sparse_mla_prefill_device_buffers(interleaved_first)?;
        flashinfer_glm_dsa_sparse_mla_prefill_device_buffers(interleaved_second)?;
        let banked_selection_count =
            super::GLM_DSA_SPARSE_MLA_PREFILL_WORKSPACE.with(|workspace| {
                Ok::<_, anyhow::Error>(
                    workspace
                        .try_borrow()
                        .map_err(|_| anyhow::anyhow!("GLM DSA test workspace is borrowed"))?
                        .banked_selections
                        .len(),
                )
            })?;
        assert_eq!(
            banked_selection_count, 2,
            "each interleaved request needs an independent DSA selection"
        );
        for request in [interleaved_first, interleaved_second] {
            flashinfer_glm_dsa_sparse_mla_prefill_device_buffers(
                FlashinferGlmDsaSparseMlaPrefillInput {
                    layer_id: 3,
                    dsa_query: None,
                    dsa_weights: None,
                    index_k_cache: None,
                    ..request
                },
            )?;
        }
        super::reset_glm_dsa_sparse_mla_transient_state()?;

        let selector_position = upload_test_buffer(
            library,
            &4_096_u32.to_ne_bytes(),
            "direct DSA selector test position",
        )?;
        let selector = FlashinferGlmDsaSparseMlaPrefillInput {
            dsa_query: Some(dsa_query.buffer),
            dsa_weights: Some(dsa_weights.buffer),
            positions: selector_position.buffer,
            index_k_cache: Some(index_k_cache.buffer),
            total_rows: 4_097,
            prefix_rows: 4_096,
            query_rows: 1,
            ..common
        };
        let full = flashinfer_glm_dsa_sparse_mla_prefill_device_buffers(selector)?;
        let shared = flashinfer_glm_dsa_sparse_mla_prefill_device_buffers(
            FlashinferGlmDsaSparseMlaPrefillInput {
                layer_id: 3,
                dsa_query: None,
                dsa_weights: None,
                index_k_cache: None,
                ..selector
            },
        )?;
        for output in [full.output, shared.output] {
            let mut bytes = vec![0_u8; output.bytes];
            library.copy_d2h(&mut bytes, output)?;
            for value in bytes.chunks_exact(2) {
                let bits = u16::from_ne_bytes(value.try_into().expect("BF16 chunk width"));
                let value = f32::from_bits((bits as u32) << 16);
                assert!(value.is_finite());
                assert!(value.abs() <= 1.0e-3, "expected zero output, got {value}");
            }
        }
        // The B12X automatic decode selector changes implementation above a
        // 32K live prefix.  Exercise the first row beyond that boundary so a
        // graph replay cannot silently reintroduce the persistent-grid hang.
        let long_selector_position = upload_test_buffer(
            library,
            &32_768_u32.to_ne_bytes(),
            "direct DSA long-context selector test position",
        )?;
        let long_selector = FlashinferGlmDsaSparseMlaPrefillInput {
            positions: long_selector_position.buffer,
            total_rows: 32_769,
            prefix_rows: 32_768,
            ..selector
        };
        let long_output =
            flashinfer_glm_dsa_sparse_mla_prefill_device_buffers(long_selector)?.output;
        let mut long_output_bytes = vec![0_u8; long_output.bytes];
        library.copy_d2h(&mut long_output_bytes, long_output)?;
        for value in long_output_bytes.chunks_exact(2) {
            let bits = u16::from_ne_bytes(value.try_into().expect("BF16 chunk width"));
            let value = f32::from_bits((bits as u32) << 16);
            assert!(value.is_finite());
            assert!(value.abs() <= 1.0e-3, "expected zero output, got {value}");
        }
        // Serving invokes the same q=1 fused selector at 21 independent full
        // indexer layers. Its cross-CTA merge counters must start clean on
        // every graph replay rather than relying on a previous kernel's
        // self-reset path.
        for _ in 0..32 {
            flashinfer_glm_dsa_sparse_mla_prefill_device_buffers(long_selector)?;
        }
        // The terminal full-size prefill chunk reaches the same 32K selector
        // boundary through the multi-row scorer. Keep that graph contract
        // covered independently from the single-row decode route.
        let long_prefill_positions = upload_test_buffer(
            library,
            &(32_256_u32..32_768_u32)
                .flat_map(u32::to_ne_bytes)
                .collect::<Vec<_>>(),
            "direct DSA long-context prefill test positions",
        )?;
        let long_prefill = FlashinferGlmDsaSparseMlaPrefillInput {
            dsa_query: Some(dsa_query.buffer),
            dsa_weights: Some(dsa_weights.buffer),
            positions: long_prefill_positions.buffer,
            index_k_cache: Some(index_k_cache.buffer),
            total_rows: 32_768,
            prefix_rows: 32_256,
            query_rows: QUERY_ROWS,
            ..common
        };
        let long_prefill_output =
            flashinfer_glm_dsa_sparse_mla_prefill_device_buffers(long_prefill)?.output;
        let mut long_prefill_output_bytes = vec![0_u8; long_prefill_output.bytes];
        library.copy_d2h(&mut long_prefill_output_bytes, long_prefill_output)?;
        assert!(long_prefill_output_bytes.chunks_exact(2).all(|value| {
            let bits = u16::from_ne_bytes(value.try_into().expect("BF16 chunk width"));
            let value = f32::from_bits((bits as u32) << 16);
            value.is_finite() && value.abs() <= 1.0e-3
        }));
        flashinfer_glm_dsa_sparse_mla_prefill_device_buffers(
            FlashinferGlmDsaSparseMlaPrefillInput {
                layer_id: 3,
                dsa_query: None,
                dsa_weights: None,
                index_k_cache: None,
                ..long_prefill
            },
        )?;
        // Serving reaches the same graph by replaying every 512-row chunk,
        // rather than jumping directly from 4K to 32K. Exercise that state
        // progression on layer 0, whose index selection has no sharing bank,
        // to distinguish persistent kernel scratch from cross-layer storage.
        let replay_positions = upload_test_buffer(
            library,
            &vec![0_u8; QUERY_ROWS * std::mem::size_of::<u32>()],
            "direct DSA repeated long-context positions",
        )?;
        let mut replay_output = None;
        for chunk_index in 5..=64 {
            let total_rows = chunk_index * QUERY_ROWS;
            let prefix_rows = total_rows - QUERY_ROWS;
            let position_bytes = (prefix_rows as u32..total_rows as u32)
                .flat_map(u32::to_ne_bytes)
                .collect::<Vec<_>>();
            library.copy_h2d(replay_positions.buffer, &position_bytes)?;
            replay_output = Some(
                flashinfer_glm_dsa_sparse_mla_prefill_device_buffers(
                    FlashinferGlmDsaSparseMlaPrefillInput {
                        layer_id: 0,
                        dsa_query: Some(dsa_query.buffer),
                        dsa_weights: Some(dsa_weights.buffer),
                        positions: replay_positions.buffer,
                        index_k_cache: Some(index_k_cache.buffer),
                        total_rows,
                        prefix_rows,
                        query_rows: QUERY_ROWS,
                        ..common
                    },
                )?
                .output,
            );
        }
        let replay_output = replay_output.expect("long-context replay loop produced output");
        let mut replay_output_bytes = vec![0_u8; replay_output.bytes];
        library.copy_d2h(&mut replay_output_bytes, replay_output)?;
        assert!(replay_output_bytes.chunks_exact(2).all(|value| {
            let bits = u16::from_ne_bytes(value.try_into().expect("BF16 chunk width"));
            let value = f32::from_bits((bits as u32) << 16);
            value.is_finite() && value.abs() <= 1.0e-3
        }));
        let selector_prefill_positions = upload_test_buffer(
            library,
            &(4_096_u32..4_104_u32)
                .flat_map(u32::to_ne_bytes)
                .collect::<Vec<_>>(),
            "direct DSA selector prefill test positions",
        )?;
        let selector_prefill = FlashinferGlmDsaSparseMlaPrefillInput {
            dsa_query: Some(dsa_query.buffer),
            dsa_weights: Some(dsa_weights.buffer),
            positions: selector_prefill_positions.buffer,
            index_k_cache: Some(index_k_cache.buffer),
            total_rows: 4_104,
            prefix_rows: 4_096,
            query_rows: 8,
            ..common
        };
        flashinfer_glm_dsa_sparse_mla_prefill_device_buffers(selector_prefill)?;
        flashinfer_glm_dsa_sparse_mla_prefill_device_buffers(
            FlashinferGlmDsaSparseMlaPrefillInput {
                layer_id: 3,
                dsa_query: None,
                dsa_weights: None,
                index_k_cache: None,
                ..selector_prefill
            },
        )?;
        Ok(())
    }

    fn upload_test_buffer(
        library: &'static super::NativeLibrary,
        bytes: &[u8],
        label: &'static str,
    ) -> Result<OwnedCoordinatorDeviceBuffer> {
        let buffer = OwnedCoordinatorDeviceBuffer::new(library, bytes.len(), label)?;
        library
            .copy_h2d(buffer.buffer, bytes)
            .with_context(|| format!("uploading {label}"))?;
        Ok(buffer)
    }

    #[test]
    fn cudnn_mla_suffix_query_capacity_accepts_supported_powers_of_two() {
        assert_eq!(
            parse_flashinfer_cudnn_mla_suffix_query_capacity("512"),
            Some(512)
        );
        assert_eq!(
            parse_flashinfer_cudnn_mla_suffix_query_capacity(" 1024 "),
            Some(1024)
        );
        assert_eq!(
            parse_flashinfer_cudnn_mla_suffix_query_capacity("2048"),
            Some(2048)
        );
        assert_eq!(
            parse_flashinfer_cudnn_mla_suffix_query_capacity("768"),
            None
        );
        assert_eq!(
            parse_flashinfer_cudnn_mla_suffix_query_capacity("4096"),
            None
        );
    }

    #[test]
    fn packed_projection_preserves_external_output_while_staging_graph_output() {
        let external_output = buffer_at(0x1000);
        let staging_output = buffer_at(0x2000);
        let projection = FlashinferMlaHiddenProjection {
            weight: buffer_at(0x3000),
            output: external_output,
            hidden_dim: 6_144,
            w4a16: None,
            w8a16: None,
        };

        let (copy_destination, staged) =
            stage_flashinfer_hidden_projection(Some(projection), staging_output, false);

        assert_eq!(copy_destination.unwrap().ptr, external_output.ptr);
        assert_eq!(staged.unwrap().output.ptr, staging_output.ptr);
    }

    #[test]
    fn packed_projection_can_write_directly_to_external_output() {
        let external_output = buffer_at(0x1000);
        let projection = FlashinferMlaHiddenProjection {
            weight: buffer_at(0x3000),
            output: external_output,
            hidden_dim: 6_144,
            w4a16: None,
            w8a16: None,
        };

        let (copy_destination, direct) =
            stage_flashinfer_hidden_projection(Some(projection), buffer_at(0x2000), true);

        assert!(copy_destination.is_none());
        assert_eq!(direct.unwrap().output.ptr, external_output.ptr);
    }

    fn buffer_at(address: usize) -> GlmrtDeviceBuffer {
        GlmrtDeviceBuffer {
            ptr: address as *mut std::ffi::c_void,
            bytes: 1,
            device_id: 0,
            flags: 0,
        }
    }

    #[test]
    fn pads_full_prefill_to_graph_bucket() {
        for query_rows in 2..=16 {
            assert_eq!(
                flashinfer_mla_capture_shape(query_rows, 0, query_rows, 16).unwrap(),
                FlashinferMlaCaptureShape {
                    rows: 16,
                    query_row_offset: 0,
                    query_rows: 16,
                    kv_prefix_padding: 0,
                    query_prefix_padding: 0,
                }
            );
        }
    }

    #[test]
    fn graph_signature_reuses_padded_bucket() {
        let first = flashinfer_mla_graph_signature(18, 0, 18, 32, 64, 192, 64, 256, 0.125).unwrap();
        let second =
            flashinfer_mla_graph_signature(24, 0, 24, 32, 64, 192, 64, 256, 0.125).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn legacy_capture_shape_left_pads_partial_suffix_to_reusable_bucket() {
        assert_eq!(
            flashinfer_mla_capture_shape(999, 768, 231, 1_024).unwrap(),
            FlashinferMlaCaptureShape {
                rows: 1_024,
                query_row_offset: 512,
                query_rows: 512,
                kv_prefix_padding: 25,
                query_prefix_padding: 281,
            }
        );
    }

    #[test]
    fn right_pads_full_prefill_to_power_of_two() {
        assert_eq!(
            flashinfer_mla_capture_shape(487, 0, 487, 512).unwrap(),
            FlashinferMlaCaptureShape {
                rows: 512,
                query_row_offset: 0,
                query_rows: 512,
                kv_prefix_padding: 0,
                query_prefix_padding: 0,
            }
        );
    }

    #[test]
    fn legacy_capture_shape_pads_small_power_of_two_suffix_to_reusable_bucket() {
        assert_eq!(
            flashinfer_mla_capture_shape(768, 512, 256, 1_024).unwrap(),
            FlashinferMlaCaptureShape {
                rows: 1_024,
                query_row_offset: 512,
                query_rows: 512,
                kv_prefix_padding: 256,
                query_prefix_padding: 256,
            }
        );
    }

    #[test]
    fn legacy_suffix_graph_signature_reuses_fixed_prefill_bucket() {
        let small =
            flashinfer_mla_graph_signature(1_088, 1_024, 64, 2_048, 64, 192, 64, 256, 0.125)
                .unwrap();
        let large =
            flashinfer_mla_graph_signature(1_280, 1_024, 256, 2_048, 64, 192, 64, 256, 0.125)
                .unwrap();
        assert_eq!(small, large);
    }
}
