use anyhow::{Context, Result};
use glmrt_core::{
    DType, KvBlockDescriptor, KvCacheConfig, LayerId, PositionId, TensorCatalog, TensorInfo,
    GLM52_DSA_INDEXER_LAYER_IDS_WITH_MTP, GLM52_DSA_INDEX_HEAD_DIM, GLM52_HIDDEN_SIZE,
    GLM52_MLA_KV_LORA_RANK, GLM52_MLA_QK_ROPE_HEAD_DIM, GLM52_MLA_ROPE_THETA,
};
use glmrt_loader::{
    load_tensor_bytes, load_tensor_rows, read_tensor_bytes_into, read_tensor_row_prefix_into,
    read_tensor_rows_into,
};

use super::super::coordinator_kernels::{
    causal_attention_rows_bf16_for_layer, coordinator_cuda_reference_kernels_enabled,
    layer_norm_affine_bf16_preloaded_resident_weight_bias_device_input_output,
    linear_residual_add_rows_bf16_preloaded_resident_weight_device_input_device_output,
    linear_residual_add_rows_bf16_preloaded_resident_weight_padded_device_input_device_output,
    linear_rows_bf16_preloaded_resident_weight_device_output,
    preload_resident_weight_from_host_staging, preloaded_resident_weight_device_buffer,
    preloaded_resident_weight_device_buffer_view, residual_add_prefix_bf16_bytes_into,
    rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output,
    rmsnorm_hidden_bf16_preloaded_resident_weight_device_output, DeviceBf16Output,
    CUDA_REFERENCE_LINEAR_BF16_BACKEND,
    CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
    CUDA_REFERENCE_LINEAR_BF16_RESIDENT_WEIGHT_BACKEND,
    CUDA_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND,
};
use super::super::dense::math::{
    bf16_bytes_from_f32, bf16_bytes_to_f32, checksum_f64, fill_bf16_bytes_from_f32,
};
use super::super::dense::REAL_FULL_DENSE_RMSNORM_EPS;
use super::super::kv::device::{RealFullDeviceKvExecutionMirror, RealFullDeviceMlaAttentionParts};
use super::super::types::{
    RealFullAttentionResidualPrefixProbe, RealFullDsaIndexerAttentionProbe,
    RealFullMlaRopeAttentionProbe,
};
use super::mla::{causal_mla_rope_attention_f32, MlaRopeAttentionF32Shape};
use dsa_indexer::real_full_dsa_indexer_attention_probe as run_real_full_dsa_indexer_attention_probe;
use math::{
    apply_rope_row_with_backend, bf16_full_row_prefix_resident_available, compact_row_prefix_bytes,
    deterministic_attention_hidden_rows, flatten_rows,
    project_rows_bf16_with_optional_padded_preloaded_prefix_weight,
    project_rows_bf16_with_optional_preloaded_full_weight,
    project_rows_bf16_with_optional_preloaded_prefix_weight,
    rmsnorm_bf16_with_optional_preloaded_resident_weight,
};

mod dsa_indexer;
mod math;

const REAL_FULL_ATTENTION_RESIDUAL_LAYER_ID: usize = 0;
const REAL_FULL_ATTENTION_RESIDUAL_PREFIX_VALUES: usize = 4;
const REAL_FULL_ATTENTION_RESIDUAL_ROWS: usize = 2;
const REAL_FULL_ATTENTION_RESIDUAL_PROBE_ENV: &str = "GLMRT_REAL_FULL_ATTENTION_RESIDUAL_PROBE";
const REAL_FULL_MLA_ROPE_PROBE_HEADS: usize = 2;
const REAL_FULL_MLA_NUM_ATTENTION_HEADS: usize = 64;
const REAL_FULL_MLA_QK_NOPE_HEAD_DIM: usize = 192;
const REAL_FULL_MLA_V_HEAD_DIM: usize = 256;
const REAL_FULL_MLA_OUTPUT_PREFIX_VALUES: usize = 16;
const REAL_FULL_MLA_ROPE_THETA: f64 = GLM52_MLA_ROPE_THETA as f64;

#[derive(Clone, Copy)]
struct AttentionResidualProbeMode {
    output_count: usize,
    hidden_source: &'static str,
    scope: &'static str,
}

pub(in crate::commands::real_full) struct RealFullAttentionResidualPrefixHidden {
    pub(in crate::commands::real_full) hidden: Vec<f32>,
    pub(in crate::commands::real_full) device_hidden: Option<DeviceBf16Output>,
    pub(in crate::commands::real_full) layer_id: usize,
    pub(in crate::commands::real_full) attention_rows: usize,
    pub(in crate::commands::real_full) prefix_context_rows: usize,
    pub(in crate::commands::real_full) total_context_rows: usize,
    pub(in crate::commands::real_full) uses_kv_cache_context: bool,
    pub(in crate::commands::real_full) kv_cache_context_bytes: usize,
    pub(in crate::commands::real_full) residual_adds: usize,
    pub(in crate::commands::real_full) residual_prefix_values: usize,
    pub(in crate::commands::real_full) input_norm_bytes_read: u64,
    pub(in crate::commands::real_full) projection_bytes_read: u64,
    pub(in crate::commands::real_full) o_proj_bytes_read: u64,
    pub(in crate::commands::real_full) projection_backend: &'static str,
    pub(in crate::commands::real_full) attention_backend: &'static str,
    pub(in crate::commands::real_full) residual_add_backend: &'static str,
    pub(in crate::commands::real_full) initial_residual_checksum: f64,
    pub(in crate::commands::real_full) residual_delta_checksum: f64,
    pub(in crate::commands::real_full) final_residual_checksum: f64,
    pub(in crate::commands::real_full) includes_causal_softmax: bool,
    pub(in crate::commands::real_full) includes_mla_softmax: bool,
    pub(in crate::commands::real_full) includes_dsa_candidate_selection: bool,
    pub(in crate::commands::real_full) includes_dsa_softmax: bool,
    pub(in crate::commands::real_full) dsa_candidate_rows: usize,
    pub(in crate::commands::real_full) dsa_selected_indices: Vec<usize>,
    pub(in crate::commands::real_full) dsa_attention_context_checksum: Option<f64>,
    pub(in crate::commands::real_full) dsa_projection_backend: Option<&'static str>,
}

impl Clone for RealFullAttentionResidualPrefixHidden {
    fn clone(&self) -> Self {
        Self {
            hidden: self.hidden.clone(),
            device_hidden: None,
            layer_id: self.layer_id,
            attention_rows: self.attention_rows,
            prefix_context_rows: self.prefix_context_rows,
            total_context_rows: self.total_context_rows,
            uses_kv_cache_context: self.uses_kv_cache_context,
            kv_cache_context_bytes: self.kv_cache_context_bytes,
            residual_adds: self.residual_adds,
            residual_prefix_values: self.residual_prefix_values,
            input_norm_bytes_read: self.input_norm_bytes_read,
            projection_bytes_read: self.projection_bytes_read,
            o_proj_bytes_read: self.o_proj_bytes_read,
            projection_backend: self.projection_backend,
            attention_backend: self.attention_backend,
            residual_add_backend: self.residual_add_backend,
            initial_residual_checksum: self.initial_residual_checksum,
            residual_delta_checksum: self.residual_delta_checksum,
            final_residual_checksum: self.final_residual_checksum,
            includes_causal_softmax: self.includes_causal_softmax,
            includes_mla_softmax: self.includes_mla_softmax,
            includes_dsa_candidate_selection: self.includes_dsa_candidate_selection,
            includes_dsa_softmax: self.includes_dsa_softmax,
            dsa_candidate_rows: self.dsa_candidate_rows,
            dsa_selected_indices: self.dsa_selected_indices.clone(),
            dsa_attention_context_checksum: self.dsa_attention_context_checksum,
            dsa_projection_backend: self.dsa_projection_backend,
        }
    }
}

#[derive(Clone)]
pub(in crate::commands::real_full) struct RealFullMlaRopeKvCacheBlock {
    pub(in crate::commands::real_full) token_start: usize,
    pub(in crate::commands::real_full) token_count: usize,
    pub(in crate::commands::real_full) bytes: Vec<u8>,
}

pub(in crate::commands::real_full) struct RealFullAttentionResidualPrefixRows {
    pub(in crate::commands::real_full) hidden_rows: Vec<Vec<f32>>,
    pub(in crate::commands::real_full) layer_id: usize,
    pub(in crate::commands::real_full) attention_rows: usize,
    pub(in crate::commands::real_full) residual_adds: usize,
    pub(in crate::commands::real_full) residual_prefix_values: usize,
    pub(in crate::commands::real_full) input_norm_bytes_read: u64,
    pub(in crate::commands::real_full) projection_bytes_read: u64,
    pub(in crate::commands::real_full) o_proj_bytes_read: u64,
    pub(in crate::commands::real_full) projection_backend: &'static str,
    pub(in crate::commands::real_full) attention_backend: &'static str,
    pub(in crate::commands::real_full) residual_add_backend: &'static str,
    pub(in crate::commands::real_full) initial_residual_checksum: f64,
    pub(in crate::commands::real_full) residual_delta_checksum: f64,
    pub(in crate::commands::real_full) final_residual_checksum: f64,
    pub(in crate::commands::real_full) includes_causal_softmax: bool,
    pub(in crate::commands::real_full) includes_mla_softmax: bool,
}

impl RealFullAttentionResidualPrefixRows {
    fn uses_cuda_projection_backend(&self) -> bool {
        matches!(
            self.projection_backend,
            CUDA_REFERENCE_LINEAR_BF16_BACKEND
                | CUDA_REFERENCE_LINEAR_BF16_RESIDENT_WEIGHT_BACKEND
                | CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        )
    }

    fn uses_cuda_attention_path(&self) -> bool {
        self.uses_cuda_projection_backend()
            || self.attention_backend.starts_with("cuda-reference-")
            || self.residual_add_backend.starts_with("cuda-reference-")
    }

    pub(in crate::commands::real_full) fn input_norm_weight_evidence(&self) -> bool {
        self.input_norm_bytes_read > 0 || self.uses_cuda_attention_path()
    }

    pub(in crate::commands::real_full) fn projection_weight_evidence(&self) -> bool {
        self.projection_bytes_read > 0 || self.uses_cuda_projection_backend()
    }

    pub(in crate::commands::real_full) fn o_proj_weight_evidence(&self) -> bool {
        self.o_proj_bytes_read > 0 || self.uses_cuda_projection_backend()
    }

    pub(in crate::commands::real_full) fn required_weight_evidence(&self) -> bool {
        self.input_norm_weight_evidence()
            && self.projection_weight_evidence()
            && self.o_proj_weight_evidence()
    }
}

struct AttentionResidualPrefixExecution {
    probe: RealFullAttentionResidualPrefixProbe,
    hidden_after_attention_prefix_rows: Vec<Vec<f32>>,
    first_row_residual_before_checksum: f64,
    first_row_residual_delta_checksum: f64,
    first_row_residual_after_checksum: f64,
}

struct MlaRopeAttentionPrefixExecution {
    probe: RealFullMlaRopeAttentionProbe,
    hidden_after_attention_prefix_rows: Vec<Vec<f32>>,
    device_hidden_after_attention: Option<DeviceBf16Output>,
    projection_backend: &'static str,
    residual_add_backend: &'static str,
    uses_kv_cache_context: bool,
    kv_cache_context_bytes: usize,
    first_row_residual_before_checksum: f64,
    first_row_residual_delta_checksum: f64,
    first_row_residual_after_checksum: f64,
}

struct MlaRopeDeviceCurrentRowProjectionOutputs {
    q_projected: DeviceBf16Output,
    kv_a_projected: DeviceBf16Output,
    dsa_key: Option<DeviceBf16Output>,
}

#[derive(Default)]
struct AttentionResidualAddWorkspace {
    residual_bf16: Vec<u8>,
    delta_bf16: Vec<u8>,
    output_bf16: Vec<u8>,
}

struct AttentionResidualAddResult {
    values: Vec<f32>,
    backend: &'static str,
}

pub(in crate::commands::real_full) fn real_full_attention_residual_prefix_probe(
    catalog: &TensorCatalog,
) -> RealFullAttentionResidualPrefixProbe {
    let mode = attention_residual_probe_mode(
        std::env::var(REAL_FULL_ATTENTION_RESIDUAL_PROBE_ENV)
            .ok()
            .as_deref(),
    );
    match run_real_full_attention_residual_prefix_probe_with_mode(catalog, mode) {
        Ok(probe) => probe,
        Err(error) => skipped_real_full_attention_residual_prefix_probe(
            "error",
            mode,
            Some(error.to_string()),
        ),
    }
}

pub(in crate::commands::real_full) fn real_full_mla_rope_attention_probe(
    catalog: &TensorCatalog,
) -> RealFullMlaRopeAttentionProbe {
    match execute_real_full_mla_rope_attention_probe(catalog) {
        Ok(probe) => probe,
        Err(error) => skipped_real_full_mla_rope_attention_probe("error", Some(error.to_string())),
    }
}

pub(in crate::commands::real_full) fn real_full_dsa_indexer_attention_probe(
    catalog: &TensorCatalog,
) -> RealFullDsaIndexerAttentionProbe {
    run_real_full_dsa_indexer_attention_probe(catalog)
}

fn attention_residual_probe_mode(env_setting: Option<&str>) -> AttentionResidualProbeMode {
    let normalized = env_setting.map(|value| value.trim().to_ascii_lowercase());
    match normalized.as_deref() {
        Some("full-output-rows" | "full-output" | "output-full") => AttentionResidualProbeMode {
            output_count: GLM52_HIDDEN_SIZE,
            hidden_source: "two-deterministic-hidden-shaped-f32-rows-full-output-attention-residual",
            scope: "opt-in real GLM-5.2 BF16 input RMSNorm plus q/kv projection prefixes expanded to hidden-width output rows, causal softmax over two rows, and o_proj output rows applied into full hidden residual rows for layer 0; full attention context, full MLA/RoPE attention, and full-model residuals are still omitted",
        },
        _ => AttentionResidualProbeMode {
            output_count: REAL_FULL_ATTENTION_RESIDUAL_PREFIX_VALUES,
            hidden_source: "two-deterministic-hidden-shaped-f32-rows",
            scope: "default bounded real GLM-5.2 BF16 input RMSNorm plus q/kv projection prefixes, causal softmax over two rows, and o_proj output prefix applied into residual prefixes for layer 0; full MLA/RoPE attention is still omitted",
        },
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn real_full_attention_residual_full_output_hidden(
    catalog: &TensorCatalog,
) -> Result<RealFullAttentionResidualPrefixHidden> {
    let execution = execute_real_full_attention_residual_prefix_with_rows(
        catalog,
        REAL_FULL_ATTENTION_RESIDUAL_LAYER_ID,
        GLM52_HIDDEN_SIZE,
        deterministic_attention_hidden_rows(),
        "two-deterministic-hidden-shaped-f32-rows-full-output-attention-residual",
        "execute hidden-width real GLM-5.2 BF16 attention output rows for layer 0 from deterministic hidden rows; full attention context, full MLA/RoPE attention, and full-model residuals remain omitted",
    )?;
    attention_hidden_from_execution(
        execution,
        "real full attention full-output hidden execution produced no rows",
    )
}

pub(in crate::commands::real_full) fn real_full_mla_rope_attention_prefix_hidden_for_layer_from_initial(
    catalog: &TensorCatalog,
    layer_id: usize,
    initial_hidden: Vec<f32>,
) -> Result<RealFullAttentionResidualPrefixHidden> {
    real_full_mla_rope_attention_hidden_for_layer_from_initial(
        catalog,
        layer_id,
        Vec::new(),
        initial_hidden,
        REAL_FULL_MLA_OUTPUT_PREFIX_VALUES,
        REAL_FULL_MLA_ROPE_PROBE_HEADS,
        "single-supplied-hidden-row-carried-from-previous-residual-stage-mla-rope-attention",
        "execute bounded real GLM-5.2 BF16 main MLA/RoPE attention residual prefix for one selected layer from a supplied residual hidden row, plus bounded DSA/indexer attention on configured DSA layers; full context and full-model residuals remain omitted",
        "single-row-main-mla-rope-causal-context",
        None,
    )
}

pub(in crate::commands::real_full) fn real_full_mla_rope_attention_full_output_hidden_for_layer_from_initial(
    catalog: &TensorCatalog,
    layer_id: usize,
    initial_hidden: Vec<f32>,
) -> Result<RealFullAttentionResidualPrefixHidden> {
    real_full_mla_rope_attention_hidden_for_layer_from_initial(
        catalog,
        layer_id,
        Vec::new(),
        initial_hidden,
        GLM52_HIDDEN_SIZE,
        REAL_FULL_MLA_NUM_ATTENTION_HEADS,
        "single-supplied-hidden-row-carried-from-previous-residual-stage-full-output-mla-rope-attention",
        "execute hidden-width real GLM-5.2 BF16 main MLA/RoPE attention residual output for one selected layer from a supplied residual hidden row, plus bounded DSA/indexer attention on configured DSA layers; full committed-KV context and full-model residuals remain omitted",
        "single-row-full-output-main-mla-rope-causal-context",
        None,
    )
}

pub(in crate::commands::real_full) fn real_full_mla_rope_attention_prefix_context_hidden_for_layer_from_initial(
    catalog: &TensorCatalog,
    layer_id: usize,
    prefix_context_rows: Vec<Vec<f32>>,
    initial_hidden: Vec<f32>,
) -> Result<RealFullAttentionResidualPrefixHidden> {
    real_full_mla_rope_attention_hidden_for_layer_from_initial(
        catalog,
        layer_id,
        prefix_context_rows,
        initial_hidden,
        REAL_FULL_MLA_OUTPUT_PREFIX_VALUES,
        REAL_FULL_MLA_ROPE_PROBE_HEADS,
        "single-supplied-hidden-row-with-supplied-prefix-context-mla-rope-attention",
        "execute bounded real GLM-5.2 BF16 main MLA/RoPE attention residual prefix for one selected layer from a supplied residual hidden row while attending over supplied committed-prefix context rows; KV-cache-backed context and full-model residuals remain omitted",
        "supplied-prefix-plus-single-row-main-mla-rope-causal-context",
        None,
    )
}

pub(in crate::commands::real_full) fn real_full_mla_rope_attention_full_output_prefix_context_hidden_for_layer_from_initial(
    catalog: &TensorCatalog,
    layer_id: usize,
    prefix_context_rows: Vec<Vec<f32>>,
    initial_hidden: Vec<f32>,
) -> Result<RealFullAttentionResidualPrefixHidden> {
    real_full_mla_rope_attention_hidden_for_layer_from_initial(
        catalog,
        layer_id,
        prefix_context_rows,
        initial_hidden,
        GLM52_HIDDEN_SIZE,
        REAL_FULL_MLA_NUM_ATTENTION_HEADS,
        "single-supplied-hidden-row-with-supplied-prefix-context-full-output-mla-rope-attention",
        "execute hidden-width real GLM-5.2 BF16 main MLA/RoPE attention residual row for one selected layer from a supplied residual hidden row while attending over supplied committed-prefix context rows; KV-cache-backed context and full-model residuals remain omitted",
        "supplied-prefix-plus-single-row-main-mla-rope-causal-context-full-output",
        None,
    )
}

pub(in crate::commands::real_full) fn real_full_mla_rope_kv_cache_block_for_layer_from_hidden(
    catalog: &TensorCatalog,
    layer_id: usize,
    token_start: usize,
    hidden: &[f32],
) -> Result<RealFullMlaRopeKvCacheBlock> {
    let hidden_rows = vec![hidden.to_vec()];
    real_full_mla_rope_kv_cache_block_for_layer_from_hidden_rows(
        catalog,
        layer_id,
        token_start,
        &hidden_rows,
    )
}

pub(in crate::commands::real_full) fn real_full_mla_rope_kv_cache_block_for_layer_from_hidden_rows(
    catalog: &TensorCatalog,
    layer_id: usize,
    token_start: usize,
    hidden_rows: &[Vec<f32>],
) -> Result<RealFullMlaRopeKvCacheBlock> {
    if hidden_rows.is_empty() {
        anyhow::bail!("real full MLA/RoPE KV payload requires at least one hidden row");
    }
    for (row_index, hidden) in hidden_rows.iter().enumerate() {
        if hidden.len() != GLM52_HIDDEN_SIZE {
            anyhow::bail!(
                "real full MLA/RoPE KV payload hidden row {row_index} width mismatch: expected {} got {}",
                GLM52_HIDDEN_SIZE,
                hidden.len()
            );
        }
    }

    let input_norm_name = format!("model.layers.{layer_id}.input_layernorm.weight");
    let kv_a_name = format!("model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.weight");
    let input_norm_info = catalog_tensor(catalog, &input_norm_name)?;
    let kv_a_info = catalog_tensor(catalog, &kv_a_name)?;
    if input_norm_info.dtype != DType::Bf16 || kv_a_info.dtype != DType::Bf16 {
        anyhow::bail!("real full MLA/RoPE KV payload expects BF16 tensors for layer {layer_id}");
    }
    let kv_width = GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM;
    if input_norm_info.shape != vec![GLM52_HIDDEN_SIZE]
        || kv_a_info.shape != vec![kv_width, GLM52_HIDDEN_SIZE]
    {
        anyhow::bail!(
            "real full MLA/RoPE KV payload tensor shape mismatch for layer {layer_id}: input_norm={:?} kv_a={:?}",
            input_norm_info.shape,
            kv_a_info.shape
        );
    }
    let cuda_reference_enabled = coordinator_cuda_reference_kernels_enabled();
    let input_norm_full_resident =
        math::bf16_full_vector_resident_available(&input_norm_name, GLM52_HIDDEN_SIZE);
    let input_norm = if input_norm_full_resident {
        None
    } else if cuda_reference_enabled {
        preload_bf16_tensor_resident_from_host_staging(
            catalog,
            &input_norm_name,
            &[GLM52_HIDDEN_SIZE],
            "BF16 MLA/RoPE KV input norm pinned staging",
        )?;
        None
    } else {
        Some(load_tensor_bytes(catalog, &input_norm_name)?)
    };
    let kv_a_full_resident =
        bf16_full_row_prefix_resident_available(&kv_a_name, kv_width, GLM52_HIDDEN_SIZE, kv_width);
    let kv_a = if kv_a_full_resident {
        None
    } else if cuda_reference_enabled {
        preload_bf16_tensor_resident_from_host_staging(
            catalog,
            &kv_a_name,
            &[kv_width, GLM52_HIDDEN_SIZE],
            "BF16 MLA/RoPE KV kv_a pinned staging",
        )?;
        None
    } else {
        Some(load_tensor_bytes(catalog, &kv_a_name)?)
    };

    let flattened_hidden = flatten_rows(hidden_rows);
    let normalized = rmsnorm_bf16_with_optional_preloaded_resident_weight(
        &input_norm_name,
        &bf16_bytes_from_f32(&flattened_hidden),
        input_norm.as_ref().map(|tensor| tensor.bytes.as_slice()),
        hidden_rows.len(),
        GLM52_HIDDEN_SIZE,
        REAL_FULL_DENSE_RMSNORM_EPS,
    )?
    .values;
    let dsa_key_rows = if GLM52_DSA_INDEXER_LAYER_IDS_WITH_MTP.contains(&layer_id) {
        let rows = dsa_indexer::real_full_dsa_indexer_kv_payload_rows_for_layer_from_hidden_rows(
            catalog,
            layer_id,
            hidden_rows,
        )?;
        if rows.len() != hidden_rows.len() {
            anyhow::bail!(
                "real DSA/indexer KV payload row count mismatch for layer {layer_id}: expected {} got {}",
                hidden_rows.len(),
                rows.len()
            );
        }
        Some(rows)
    } else {
        None
    };

    let mut bytes = Vec::with_capacity(
        hidden_rows.len()
            * (kv_width
                + dsa_key_rows
                    .as_ref()
                    .map(|_| GLM52_DSA_INDEX_HEAD_DIM)
                    .unwrap_or(0))
            * std::mem::size_of::<u16>(),
    );
    for (row_index, normalized_row) in normalized.chunks_exact(GLM52_HIDDEN_SIZE).enumerate() {
        let kv_a_projected = project_rows_bf16_with_optional_preloaded_full_weight(
            &kv_a_name,
            normalized_row,
            kv_a.as_ref().map(|tensor| tensor.bytes.as_slice()),
            kv_width,
            GLM52_HIDDEN_SIZE,
        )?;
        bytes.extend_from_slice(&bf16_bytes_from_f32(&kv_a_projected.values));
        if let Some(dsa_rows) = dsa_key_rows.as_ref() {
            let dsa_key = &dsa_rows[row_index];
            if dsa_key.len() != GLM52_DSA_INDEX_HEAD_DIM {
                anyhow::bail!(
                    "real DSA/indexer KV payload width mismatch for layer {layer_id} row {row_index}: expected {} got {}",
                    GLM52_DSA_INDEX_HEAD_DIM,
                    dsa_key.len()
                );
            }
            bytes.extend_from_slice(&bf16_bytes_from_f32(dsa_key));
        }
    }

    Ok(RealFullMlaRopeKvCacheBlock {
        token_start,
        token_count: hidden_rows.len(),
        bytes,
    })
}

pub(in crate::commands::real_full) fn real_full_mla_rope_attention_kv_cache_context_hidden_for_layer_from_initial(
    catalog: &TensorCatalog,
    layer_id: usize,
    prefix_kv_blocks: Vec<RealFullMlaRopeKvCacheBlock>,
    initial_hidden: Vec<f32>,
) -> Result<RealFullAttentionResidualPrefixHidden> {
    real_full_mla_rope_attention_hidden_for_layer_from_initial_with_kv_cache(
        catalog,
        layer_id,
        prefix_kv_blocks,
        initial_hidden,
        REAL_FULL_MLA_OUTPUT_PREFIX_VALUES,
        REAL_FULL_MLA_ROPE_PROBE_HEADS,
        "single-supplied-hidden-row-with-kv-cache-prefix-context-mla-rope-attention",
        "execute bounded real GLM-5.2 BF16 main MLA/RoPE attention residual prefix for one selected layer from a supplied residual hidden row while consuming committed BF16 compressed KV cache prefix blocks; full DSA KV-cache consumption and full-model residuals remain omitted",
        "kv-cache-prefix-plus-single-row-main-mla-rope-causal-context",
        None,
    )
}

pub(in crate::commands::real_full) fn real_full_mla_rope_attention_kv_cache_context_hidden_for_layer_from_initial_device_input(
    catalog: &TensorCatalog,
    layer_id: usize,
    prefix_kv_blocks: Vec<RealFullMlaRopeKvCacheBlock>,
    initial_hidden: Vec<f32>,
    initial_device_hidden: &DeviceBf16Output,
) -> Result<RealFullAttentionResidualPrefixHidden> {
    real_full_mla_rope_attention_hidden_for_layer_from_initial_with_kv_cache(
        catalog,
        layer_id,
        prefix_kv_blocks,
        initial_hidden,
        REAL_FULL_MLA_OUTPUT_PREFIX_VALUES,
        REAL_FULL_MLA_ROPE_PROBE_HEADS,
        "single-supplied-hidden-row-with-kv-cache-prefix-context-mla-rope-attention-device-input",
        "execute bounded real GLM-5.2 BF16 main MLA/RoPE attention residual prefix for one selected layer from a supplied residual hidden row while consuming committed BF16 compressed KV cache prefix blocks; the current row may enter the device query/KV path from a resident BF16 device buffer, while full DSA KV-cache consumption and full-model residuals remain omitted",
        "kv-cache-prefix-plus-single-row-main-mla-rope-causal-context-device-input",
        Some(initial_device_hidden),
    )
}

pub(in crate::commands::real_full) fn real_full_mla_rope_attention_full_output_kv_cache_context_hidden_for_layer_from_initial(
    catalog: &TensorCatalog,
    layer_id: usize,
    prefix_kv_blocks: Vec<RealFullMlaRopeKvCacheBlock>,
    initial_hidden: Vec<f32>,
) -> Result<RealFullAttentionResidualPrefixHidden> {
    real_full_mla_rope_attention_hidden_for_layer_from_initial_with_kv_cache(
        catalog,
        layer_id,
        prefix_kv_blocks,
        initial_hidden,
        GLM52_HIDDEN_SIZE,
        REAL_FULL_MLA_NUM_ATTENTION_HEADS,
        "single-supplied-hidden-row-with-kv-cache-prefix-context-full-output-mla-rope-attention",
        "execute hidden-width real GLM-5.2 BF16 main MLA/RoPE attention residual row for one selected layer from a supplied residual hidden row while consuming committed BF16 compressed KV cache prefix blocks; full DSA KV-cache consumption and full-model residuals remain omitted",
        "kv-cache-prefix-plus-single-row-main-mla-rope-causal-context-full-output",
        None,
    )
}

pub(in crate::commands::real_full) fn real_full_mla_rope_attention_full_output_kv_cache_context_hidden_for_layer_from_initial_device_input(
    catalog: &TensorCatalog,
    layer_id: usize,
    prefix_kv_blocks: Vec<RealFullMlaRopeKvCacheBlock>,
    initial_hidden: Vec<f32>,
    initial_device_hidden: &DeviceBf16Output,
) -> Result<RealFullAttentionResidualPrefixHidden> {
    real_full_mla_rope_attention_hidden_for_layer_from_initial_with_kv_cache(
        catalog,
        layer_id,
        prefix_kv_blocks,
        initial_hidden,
        GLM52_HIDDEN_SIZE,
        REAL_FULL_MLA_NUM_ATTENTION_HEADS,
        "single-supplied-hidden-row-with-kv-cache-prefix-context-full-output-mla-rope-attention-device-input",
        "execute hidden-width real GLM-5.2 BF16 main MLA/RoPE attention residual row for one selected layer from a supplied residual hidden row while consuming committed BF16 compressed KV cache prefix blocks; the current row may enter the device query/KV path from a resident BF16 device buffer, while full DSA KV-cache consumption and full-model residuals remain omitted",
        "kv-cache-prefix-plus-single-row-main-mla-rope-causal-context-full-output-device-input",
        Some(initial_device_hidden),
    )
}

#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn real_full_mla_rope_attention_kv_cache_context_rows_for_layer_from_initial_rows(
    catalog: &TensorCatalog,
    layer_id: usize,
    prefix_kv_blocks: Vec<RealFullMlaRopeKvCacheBlock>,
    hidden_rows: Vec<Vec<f32>>,
) -> Result<RealFullAttentionResidualPrefixRows> {
    real_full_mla_rope_attention_rows_for_layer_from_initial_with_kv_cache(
        catalog,
        layer_id,
        prefix_kv_blocks,
        hidden_rows,
        REAL_FULL_MLA_OUTPUT_PREFIX_VALUES,
        REAL_FULL_MLA_ROPE_PROBE_HEADS,
        "prefill-hidden-rows-with-kv-cache-prefix-context-mla-rope-attention",
        "execute bounded real GLM-5.2 BF16 main MLA/RoPE attention residual prefixes for multiple current rows while consuming committed BF16 compressed KV cache prefix blocks; full DSA value/context completion and full-model residuals remain omitted",
        "kv-cache-prefix-plus-prefill-rows-main-mla-rope-causal-context",
    )
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn real_full_mla_rope_attention_full_output_kv_cache_context_rows_for_layer_from_initial_rows(
    catalog: &TensorCatalog,
    layer_id: usize,
    prefix_kv_blocks: Vec<RealFullMlaRopeKvCacheBlock>,
    hidden_rows: Vec<Vec<f32>>,
) -> Result<RealFullAttentionResidualPrefixRows> {
    real_full_mla_rope_attention_rows_for_layer_from_initial_with_kv_cache(
        catalog,
        layer_id,
        prefix_kv_blocks,
        hidden_rows,
        GLM52_HIDDEN_SIZE,
        REAL_FULL_MLA_NUM_ATTENTION_HEADS,
        "prefill-hidden-rows-with-kv-cache-prefix-context-full-output-mla-rope-attention",
        "execute hidden-width real GLM-5.2 BF16 main MLA/RoPE attention residual rows for multiple current rows while consuming committed BF16 compressed KV cache prefix blocks; full DSA value/context completion and full-model residuals remain omitted",
        "kv-cache-prefix-plus-prefill-rows-main-mla-rope-causal-context-full-output",
    )
}

fn real_full_mla_rope_attention_hidden_for_layer_from_initial(
    catalog: &TensorCatalog,
    layer_id: usize,
    prefix_context_rows: Vec<Vec<f32>>,
    initial_hidden: Vec<f32>,
    output_count: usize,
    attention_heads: usize,
    hidden_source: &'static str,
    scope: &'static str,
    context_source: &'static str,
    initial_device_hidden: Option<&DeviceBf16Output>,
) -> Result<RealFullAttentionResidualPrefixHidden> {
    let dsa_hidden = initial_hidden.clone();
    let execution = execute_real_full_mla_rope_attention_prefix_with_context_rows(
        catalog,
        layer_id,
        output_count,
        attention_heads,
        prefix_context_rows,
        Vec::new(),
        vec![initial_hidden],
        hidden_source,
        scope,
        context_source,
        initial_device_hidden,
    )?;
    let mut attention = mla_rope_attention_hidden_from_execution(
        execution,
        "real full MLA/RoPE attention supplied-hidden execution produced no rows",
    )?;
    if GLM52_DSA_INDEXER_LAYER_IDS_WITH_MTP.contains(&layer_id) {
        let dsa = dsa_indexer::real_full_dsa_indexer_attention_for_layer_from_hidden_rows(
            catalog,
            layer_id,
            vec![dsa_hidden],
            "single-supplied-hidden-row-carried-from-previous-residual-stage-dsa-indexer",
            "single-row-dsa-indexer-rope-causal-context",
        )?;
        attention.includes_dsa_candidate_selection =
            dsa.passed && dsa.includes_dsa_candidate_selection;
        attention.includes_dsa_softmax = dsa.passed && dsa.includes_dsa_softmax;
        attention.dsa_candidate_rows = dsa.candidate_rows;
        attention.dsa_selected_indices = dsa.selected_indices;
        attention.dsa_attention_context_checksum = dsa.attention_context_checksum;
        attention.dsa_projection_backend = Some(dsa.projection_backend);
    }
    Ok(attention)
}

#[allow(clippy::too_many_arguments)]
fn real_full_mla_rope_attention_rows_for_layer_from_initial_with_kv_cache(
    catalog: &TensorCatalog,
    layer_id: usize,
    prefix_kv_blocks: Vec<RealFullMlaRopeKvCacheBlock>,
    hidden_rows: Vec<Vec<f32>>,
    output_count: usize,
    attention_heads: usize,
    hidden_source: &'static str,
    scope: &'static str,
    context_source: &'static str,
) -> Result<RealFullAttentionResidualPrefixRows> {
    let execution = execute_real_full_mla_rope_attention_prefix_with_context_rows(
        catalog,
        layer_id,
        output_count,
        attention_heads,
        Vec::new(),
        prefix_kv_blocks,
        hidden_rows,
        hidden_source,
        scope,
        context_source,
        None,
    )?;
    mla_rope_attention_rows_from_execution(
        execution,
        "real full MLA/RoPE attention KV-cache row execution produced no rows",
    )
}

fn dsa_kv_candidate_rows_from_mla_rope_blocks(
    layer_id: usize,
    prefix_kv_blocks: &[RealFullMlaRopeKvCacheBlock],
) -> Result<Vec<dsa_indexer::RealFullDsaIndexerKvCandidateRow>> {
    if !GLM52_DSA_INDEXER_LAYER_IDS_WITH_MTP.contains(&layer_id) {
        return Ok(Vec::new());
    }
    let main_kv_bytes = (GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM) * 2;
    let dsa_kv_bytes = GLM52_DSA_INDEX_HEAD_DIM * 2;
    let bytes_per_token = main_kv_bytes + dsa_kv_bytes;
    let mut rows = Vec::new();
    for (block_index, block) in prefix_kv_blocks.iter().enumerate() {
        if block.token_count == 0 {
            anyhow::bail!("DSA KV candidate block {block_index} has zero tokens");
        }
        if block.bytes.len() != block.token_count * bytes_per_token {
            anyhow::bail!(
                "DSA KV candidate block {block_index} bytes mismatch for layer {layer_id}: expected {} got {}",
                block.token_count * bytes_per_token,
                block.bytes.len()
            );
        }
        for token_offset in 0..block.token_count {
            let row_start = token_offset * bytes_per_token;
            let dsa_start = row_start + main_kv_bytes;
            let dsa_end = dsa_start + dsa_kv_bytes;
            rows.push(dsa_indexer::RealFullDsaIndexerKvCandidateRow {
                position: block.token_start + token_offset,
                key_norm: bf16_bytes_to_f32(&block.bytes[dsa_start..dsa_end])?,
                bytes: dsa_kv_bytes,
            });
        }
    }
    Ok(rows)
}

fn next_hidden_position_after_prefix_kv_blocks(
    prefix_kv_blocks: &[RealFullMlaRopeKvCacheBlock],
) -> usize {
    prefix_kv_blocks
        .iter()
        .map(|block| block.token_start.saturating_add(block.token_count))
        .max()
        .unwrap_or_default()
}

fn real_full_mla_rope_attention_hidden_for_layer_from_initial_with_kv_cache(
    catalog: &TensorCatalog,
    layer_id: usize,
    prefix_kv_blocks: Vec<RealFullMlaRopeKvCacheBlock>,
    initial_hidden: Vec<f32>,
    output_count: usize,
    attention_heads: usize,
    hidden_source: &'static str,
    scope: &'static str,
    context_source: &'static str,
    initial_device_hidden: Option<&DeviceBf16Output>,
) -> Result<RealFullAttentionResidualPrefixHidden> {
    let dsa_hidden = initial_hidden.clone();
    let dsa_kv_candidate_rows = if GLM52_DSA_INDEXER_LAYER_IDS_WITH_MTP.contains(&layer_id) {
        dsa_kv_candidate_rows_from_mla_rope_blocks(layer_id, &prefix_kv_blocks)?
    } else {
        Vec::new()
    };
    let execution = execute_real_full_mla_rope_attention_prefix_with_context_rows(
        catalog,
        layer_id,
        output_count,
        attention_heads,
        Vec::new(),
        prefix_kv_blocks,
        vec![initial_hidden],
        hidden_source,
        scope,
        context_source,
        initial_device_hidden,
    )?;
    let mut attention = mla_rope_attention_hidden_from_execution(
        execution,
        "real full MLA/RoPE attention KV-cache execution produced no rows",
    )?;
    if GLM52_DSA_INDEXER_LAYER_IDS_WITH_MTP.contains(&layer_id) {
        let dsa =
            dsa_indexer::real_full_dsa_indexer_attention_for_layer_from_hidden_rows_with_kv_cache_candidates(
            catalog,
            layer_id,
            dsa_kv_candidate_rows,
            vec![dsa_hidden],
            "single-supplied-hidden-row-carried-from-previous-residual-stage-dsa-indexer",
            "kv-cache-prefix-plus-single-row-dsa-indexer-candidate-context",
        )?;
        attention.includes_dsa_candidate_selection =
            dsa.passed && dsa.includes_dsa_candidate_selection;
        attention.includes_dsa_softmax = dsa.passed && dsa.includes_dsa_softmax;
        attention.dsa_candidate_rows = dsa.candidate_rows;
        attention.dsa_selected_indices = dsa.selected_indices;
        attention.dsa_attention_context_checksum = dsa.attention_context_checksum;
        attention.dsa_projection_backend = Some(dsa.projection_backend);
    }
    Ok(attention)
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::path::{Path, PathBuf};

    use glmrt_core::{
        ModelFacts, TensorCatalog, GLM52_DSA_INDEX_HEAD_DIM, GLM52_HIDDEN_SIZE,
        GLM52_MLA_KV_LORA_RANK, GLM52_MLA_QK_ROPE_HEAD_DIM,
    };

    use super::{
        attention_residual_probe_mode, execute_real_full_mla_rope_attention_probe,
        next_hidden_position_after_prefix_kv_blocks, real_full_dsa_indexer_attention_probe,
        run_real_full_attention_residual_prefix_probe_with_mode, RealFullMlaRopeKvCacheBlock,
        REAL_FULL_ATTENTION_RESIDUAL_PREFIX_VALUES, REAL_FULL_ATTENTION_RESIDUAL_ROWS,
        REAL_FULL_MLA_OUTPUT_PREFIX_VALUES, REAL_FULL_MLA_QK_NOPE_HEAD_DIM,
        REAL_FULL_MLA_ROPE_PROBE_HEADS, REAL_FULL_MLA_V_HEAD_DIM,
    };

    #[test]
    fn attention_residual_probe_mode_parses_bounded_and_full_output_rows() {
        let default_mode = attention_residual_probe_mode(None);
        assert_eq!(
            default_mode.output_count,
            REAL_FULL_ATTENTION_RESIDUAL_PREFIX_VALUES
        );
        assert!(default_mode.scope.contains("default bounded"));

        let bounded_mode = attention_residual_probe_mode(Some("bounded"));
        assert_eq!(
            bounded_mode.output_count,
            REAL_FULL_ATTENTION_RESIDUAL_PREFIX_VALUES
        );
        assert!(bounded_mode.scope.contains("default bounded"));

        for value in ["full-output-rows", "full-output", "output-full"] {
            let full_output_mode = attention_residual_probe_mode(Some(value));
            assert_eq!(full_output_mode.output_count, GLM52_HIDDEN_SIZE);
            assert!(full_output_mode.scope.contains("hidden-width output rows"));
        }
    }

    #[test]
    fn kv_cache_prefix_token_positions_advance_mla_rope_current_position() {
        let blocks = vec![
            RealFullMlaRopeKvCacheBlock {
                token_start: 7,
                token_count: 2,
                bytes: Vec::new(),
            },
            RealFullMlaRopeKvCacheBlock {
                token_start: 12,
                token_count: 1,
                bytes: Vec::new(),
            },
        ];

        assert_eq!(next_hidden_position_after_prefix_kv_blocks(&blocks), 13);
        assert_eq!(next_hidden_position_after_prefix_kv_blocks(&[]), 0);
    }

    #[test]
    fn mla_rope_kv_cache_block_rejects_empty_hidden_rows_before_loading_tensors() {
        let catalog = TensorCatalog {
            model_id: "empty".to_owned(),
            snapshot_path: String::new(),
            facts: ModelFacts::default(),
            tensors: Vec::new(),
        };

        let error = match super::real_full_mla_rope_kv_cache_block_for_layer_from_hidden_rows(
            &catalog,
            0,
            0,
            &[],
        ) {
            Ok(_) => panic!("empty hidden rows should fail before catalog loading"),
            Err(error) => error,
        };

        assert!(error
            .to_string()
            .contains("requires at least one hidden row"));
    }

    #[test]
    fn mla_rope_kv_cache_attention_rows_reject_empty_current_rows_before_loading_tensors() {
        let catalog = TensorCatalog {
            model_id: "empty".to_owned(),
            snapshot_path: String::new(),
            facts: ModelFacts::default(),
            tensors: Vec::new(),
        };

        let error = match super::real_full_mla_rope_attention_kv_cache_context_rows_for_layer_from_initial_rows(
            &catalog,
            0,
            Vec::new(),
            Vec::new(),
        ) {
            Ok(_) => panic!("empty current rows should fail before catalog loading"),
            Err(error) => error,
        };

        assert!(error
            .to_string()
            .contains("requires at least one hidden row"));
    }

    #[test]
    fn skipped_mla_rope_attention_probe_starts_without_prefix_context() {
        let probe = super::skipped_real_full_mla_rope_attention_probe("not-run", None);

        assert_eq!(probe.status, "not-run");
        assert_eq!(probe.attention_rows, REAL_FULL_ATTENTION_RESIDUAL_ROWS);
        assert_eq!(probe.prefix_context_rows, 0);
        assert_eq!(probe.total_context_rows, 0);
        assert_eq!(probe.compressed_kv_values, 0);
        assert_eq!(probe.causal_attention_scores, 0);
        assert_eq!(probe.mla_softmax_rows, 0);
        assert!(!probe.passed);
    }

    #[test]
    #[ignore = "loads hidden-width q/kv/o_proj attention rows from the real checkpoint"]
    fn real_checkpoint_attention_residual_full_output_rows_probe_when_available() {
        let Some(catalog) = load_real_catalog_or_skip() else {
            return;
        };

        let probe = run_real_full_attention_residual_prefix_probe_with_mode(
            &catalog,
            attention_residual_probe_mode(Some("full-output-rows")),
        )
        .expect("running real attention residual full-output-row probe");

        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "status": probe.status,
                "layer": probe.layer_id,
                "attention_rows": probe.attention_rows,
                "output_prefix_values": probe.output_prefix_values,
                "residual_prefix_values": probe.residual_prefix_values,
                "residual_adds": probe.residual_adds,
                "causal_attention_scores": probe.causal_attention_scores,
                "causal_softmax_rows": probe.causal_softmax_rows,
                "q_lora_rank": probe.q_lora_rank,
                "kv_lora_rank": probe.kv_lora_rank,
                "input_norm_bytes_read": probe.input_norm_bytes_read,
                "projection_bytes_read": probe.projection_bytes_read,
                "o_proj_bytes_read": probe.o_proj_bytes_read,
                "includes_causal_softmax": probe.includes_causal_softmax,
                "includes_mla_softmax": probe.includes_mla_softmax,
                "uses_full_model_residual": probe.uses_full_model_residual,
                "residual_after_checksum": probe.residual_after_checksum,
                "attention_output_l2_norm": probe.attention_output_l2_norm,
            }))
            .unwrap()
        );
        assert_eq!(
            probe.status,
            "numeric-real-bf16-causal-attention-full-output-rows"
        );
        assert_eq!(probe.layer_id, 0);
        assert_eq!(probe.attention_rows, REAL_FULL_ATTENTION_RESIDUAL_ROWS);
        assert_eq!(probe.output_prefix_values, GLM52_HIDDEN_SIZE);
        assert_eq!(
            probe.residual_prefix_values,
            GLM52_HIDDEN_SIZE * REAL_FULL_ATTENTION_RESIDUAL_ROWS
        );
        assert_eq!(probe.residual_adds, REAL_FULL_ATTENTION_RESIDUAL_ROWS);
        assert_eq!(probe.causal_attention_scores, 3);
        assert_eq!(probe.causal_softmax_rows, REAL_FULL_ATTENTION_RESIDUAL_ROWS);
        assert_eq!(probe.q_lora_rank, 2048);
        assert_eq!(probe.kv_lora_rank, 512);
        assert_eq!(probe.input_norm_tensors_read, 1);
        assert_eq!(probe.attention_tensors_read, 7);
        assert_eq!(probe.input_norm_bytes_read, 12_288);
        assert_eq!(probe.o_proj_bytes_read, 201_326_592);
        assert!(probe.projection_bytes_read > 260_000_000);
        assert!(probe.uses_real_attention_weights);
        assert!(probe.applies_attention_residual_prefix);
        assert!(probe.includes_causal_softmax);
        assert!(!probe.includes_mla_softmax);
        assert!(!probe.uses_full_model_residual);
        assert!(probe.q_output_checksum.unwrap().is_finite());
        assert!(probe.kv_output_checksum.unwrap().is_finite());
        assert!(probe.kv_rope_checksum.unwrap().is_finite());
        assert!(probe.attention_output_checksum.unwrap().is_finite());
        assert!(probe.attention_output_l2_norm.unwrap().is_finite());
        assert!(probe.residual_before_checksum.unwrap().is_finite());
        assert!(probe.residual_delta_checksum.unwrap().is_finite());
        assert!(probe.residual_after_checksum.unwrap().is_finite());
        assert!(probe.passed);
    }

    #[test]
    #[ignore = "loads bounded real q/kv/o_proj rows and executes main MLA/RoPE attention math"]
    fn real_checkpoint_mla_rope_attention_probe_when_available() {
        let Some(catalog) = load_real_catalog_or_skip() else {
            return;
        };

        let probe = execute_real_full_mla_rope_attention_probe(&catalog)
            .expect("running real bounded main MLA/RoPE attention probe");

        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "status": probe.status,
                "layer": probe.layer_id,
                "attention_rows": probe.attention_rows,
                "prefix_context_rows": probe.prefix_context_rows,
                "total_context_rows": probe.total_context_rows,
                "attention_heads": probe.attention_heads,
                "q_lora_rank": probe.q_lora_rank,
                "kv_lora_rank": probe.kv_lora_rank,
                "qk_nope_head_dim": probe.qk_nope_head_dim,
                "qk_rope_head_dim": probe.qk_rope_head_dim,
                "v_head_dim": probe.v_head_dim,
                "compressed_kv_width": probe.compressed_kv_width,
                "causal_attention_scores": probe.causal_attention_scores,
                "mla_softmax_rows": probe.mla_softmax_rows,
                "attention_context_values": probe.attention_context_values,
                "output_prefix_values": probe.output_prefix_values,
                "residual_prefix_values": probe.residual_prefix_values,
                "includes_rope": probe.includes_rope,
                "includes_mla_softmax": probe.includes_mla_softmax,
                "uses_full_model_residual": probe.uses_full_model_residual,
                "q_rope_rotated_checksum": probe.q_rope_rotated_checksum,
                "k_rope_rotated_checksum": probe.k_rope_rotated_checksum,
                "attention_scores_checksum": probe.attention_scores_checksum,
                "attention_weights_checksum": probe.attention_weights_checksum,
                "attention_context_checksum": probe.attention_context_checksum,
                "attention_output_checksum": probe.attention_output_checksum,
                "attention_output_l2_norm": probe.attention_output_l2_norm,
                "residual_after_checksum": probe.residual_after_checksum,
            }))
            .unwrap()
        );
        assert_eq!(probe.status, "numeric-real-bounded-main-mla-rope-attention");
        assert_eq!(probe.layer_id, 0);
        assert_eq!(probe.attention_rows, REAL_FULL_ATTENTION_RESIDUAL_ROWS);
        assert_eq!(probe.prefix_context_rows, 0);
        assert_eq!(probe.total_context_rows, REAL_FULL_ATTENTION_RESIDUAL_ROWS);
        assert_eq!(probe.attention_heads, REAL_FULL_MLA_ROPE_PROBE_HEADS);
        assert_eq!(probe.q_lora_rank, 2048);
        assert_eq!(probe.kv_lora_rank, 512);
        assert_eq!(probe.qk_nope_head_dim, REAL_FULL_MLA_QK_NOPE_HEAD_DIM);
        assert_eq!(probe.qk_rope_head_dim, 64);
        assert_eq!(probe.q_head_dim, REAL_FULL_MLA_QK_NOPE_HEAD_DIM + 64);
        assert_eq!(probe.v_head_dim, REAL_FULL_MLA_V_HEAD_DIM);
        assert_eq!(probe.compressed_kv_width, 576);
        assert_eq!(
            probe.compressed_kv_values,
            REAL_FULL_ATTENTION_RESIDUAL_ROWS * probe.compressed_kv_width
        );
        assert_eq!(
            probe.causal_attention_scores,
            REAL_FULL_MLA_ROPE_PROBE_HEADS
                * REAL_FULL_ATTENTION_RESIDUAL_ROWS
                * (REAL_FULL_ATTENTION_RESIDUAL_ROWS + 1)
                / 2
        );
        assert_eq!(
            probe.mla_softmax_rows,
            REAL_FULL_MLA_ROPE_PROBE_HEADS * REAL_FULL_ATTENTION_RESIDUAL_ROWS
        );
        assert_eq!(
            probe.attention_context_values,
            REAL_FULL_ATTENTION_RESIDUAL_ROWS
                * REAL_FULL_MLA_ROPE_PROBE_HEADS
                * REAL_FULL_MLA_V_HEAD_DIM
        );
        assert_eq!(
            probe.output_prefix_values,
            REAL_FULL_MLA_OUTPUT_PREFIX_VALUES
        );
        assert_eq!(
            probe.residual_prefix_values,
            REAL_FULL_ATTENTION_RESIDUAL_ROWS * REAL_FULL_MLA_OUTPUT_PREFIX_VALUES
        );
        assert!(probe.uses_real_attention_weights);
        assert!(probe.includes_rope);
        assert!(probe.includes_mla_softmax);
        assert!(probe.applies_attention_residual_prefix);
        assert!(!probe.uses_full_model_residual);
        assert!(probe.q_rope_rotated_checksum.unwrap().is_finite());
        assert!(probe.k_rope_rotated_checksum.unwrap().is_finite());
        assert!(probe.attention_scores_checksum.unwrap().is_finite());
        assert!(
            (probe.attention_weights_checksum.unwrap() - probe.mla_softmax_rows as f64).abs()
                < 1.0e-5
        );
        assert!(probe.attention_context_checksum.unwrap().is_finite());
        assert!(probe.attention_output_checksum.unwrap().is_finite());
        assert!(probe.attention_output_l2_norm.unwrap().is_finite());
        assert!(probe.residual_after_checksum.unwrap().is_finite());
        assert!(probe.passed);
    }

    #[test]
    #[ignore = "loads real MLA/RoPE and DSA tensors to build BF16 compressed KV cache payloads"]
    fn real_checkpoint_mla_rope_kv_cache_block_matches_glm52_payload_widths_when_available() {
        let Some(catalog) = load_real_catalog_or_skip() else {
            return;
        };
        let hidden = vec![0.03125_f32; GLM52_HIDDEN_SIZE];

        let main_block =
            super::real_full_mla_rope_kv_cache_block_for_layer_from_hidden(&catalog, 3, 0, &hidden)
                .expect("building layer-3 main MLA/RoPE KV cache block");
        assert_eq!(main_block.token_start, 0);
        assert_eq!(main_block.token_count, 1);
        assert_eq!(
            main_block.bytes.len(),
            (GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM) * 2
        );
        let chunk_block = super::real_full_mla_rope_kv_cache_block_for_layer_from_hidden_rows(
            &catalog,
            3,
            4,
            &[hidden.clone(), vec![0.0625_f32; GLM52_HIDDEN_SIZE]],
        )
        .expect("building layer-3 two-token main MLA/RoPE KV cache block");
        assert_eq!(chunk_block.token_start, 4);
        assert_eq!(chunk_block.token_count, 2);
        assert_eq!(
            chunk_block.bytes.len(),
            2 * (GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM) * 2
        );

        let dsa_block = super::real_full_mla_rope_kv_cache_block_for_layer_from_hidden(
            &catalog, 22, 0, &hidden,
        )
        .expect("building layer-22 main MLA/RoPE plus DSA KV cache block");
        assert_eq!(
            dsa_block.bytes.len(),
            (GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM + GLM52_DSA_INDEX_HEAD_DIM) * 2
        );
        let dsa_chunk_block = super::real_full_mla_rope_kv_cache_block_for_layer_from_hidden_rows(
            &catalog,
            22,
            4,
            &[hidden.clone(), vec![0.0625_f32; GLM52_HIDDEN_SIZE]],
        )
        .expect("building layer-22 two-token main MLA/RoPE plus DSA KV cache block");
        assert_eq!(dsa_chunk_block.token_start, 4);
        assert_eq!(dsa_chunk_block.token_count, 2);
        assert_eq!(
            dsa_chunk_block.bytes.len(),
            2 * (GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM + GLM52_DSA_INDEX_HEAD_DIM)
                * 2
        );
    }

    #[test]
    #[ignore = "loads real tensors and consumes a BF16 compressed KV cache prefix block in MLA/RoPE attention"]
    fn real_checkpoint_mla_rope_attention_consumes_kv_cache_prefix_context_when_available() {
        let Some(catalog) = load_real_catalog_or_skip() else {
            return;
        };
        let prefix_hidden = vec![0.03125_f32; GLM52_HIDDEN_SIZE];
        let current_hidden = vec![0.0625_f32; GLM52_HIDDEN_SIZE];
        let prefix_block = super::real_full_mla_rope_kv_cache_block_for_layer_from_hidden(
            &catalog,
            0,
            0,
            &prefix_hidden,
        )
        .expect("building layer-0 prefix KV cache block");
        let prefix_bytes = prefix_block.bytes.len();

        let attention =
            super::real_full_mla_rope_attention_kv_cache_context_hidden_for_layer_from_initial(
                &catalog,
                0,
                vec![prefix_block],
                current_hidden,
            )
            .expect("running MLA/RoPE attention with KV-cache prefix context");

        assert_eq!(attention.layer_id, 0);
        assert_eq!(attention.attention_rows, 1);
        assert_eq!(attention.prefix_context_rows, 1);
        assert_eq!(attention.total_context_rows, 2);
        assert!(attention.uses_kv_cache_context);
        assert_eq!(attention.kv_cache_context_bytes, prefix_bytes);
        assert!(attention.includes_mla_softmax);
        assert!(attention.final_residual_checksum.is_finite());
    }

    #[test]
    #[ignore = "loads real tensors and consumes a two-token BF16 compressed KV prefix block while executing two current rows"]
    fn real_checkpoint_mla_rope_attention_consumes_chunked_kv_prefix_for_current_rows_when_available(
    ) {
        let Some(catalog) = load_real_catalog_or_skip() else {
            return;
        };
        let prefix_rows = vec![
            vec![0.03125_f32; GLM52_HIDDEN_SIZE],
            vec![0.046875_f32; GLM52_HIDDEN_SIZE],
        ];
        let current_rows = vec![
            vec![0.0625_f32; GLM52_HIDDEN_SIZE],
            vec![0.078125_f32; GLM52_HIDDEN_SIZE],
        ];
        let prefix_block = super::real_full_mla_rope_kv_cache_block_for_layer_from_hidden_rows(
            &catalog,
            3,
            0,
            &prefix_rows,
        )
        .expect("building layer-3 two-token prefix KV cache block");
        assert_eq!(prefix_block.token_count, 2);

        let attention =
            super::real_full_mla_rope_attention_kv_cache_context_rows_for_layer_from_initial_rows(
                &catalog,
                3,
                vec![prefix_block],
                current_rows,
            )
            .expect("running MLA/RoPE attention with chunked KV prefix and current rows");

        assert_eq!(attention.layer_id, 3);
        assert_eq!(attention.attention_rows, 2);
        assert_eq!(attention.hidden_rows.len(), 2);
        assert_eq!(attention.residual_adds, 2);
        assert_eq!(
            attention.residual_prefix_values,
            2 * REAL_FULL_MLA_OUTPUT_PREFIX_VALUES
        );
        assert_eq!(attention.hidden_rows[0].len(), GLM52_HIDDEN_SIZE);
        assert!(attention.includes_causal_softmax);
        assert!(attention.includes_mla_softmax);
        if !super::coordinator_cuda_reference_kernels_enabled() {
            assert!(attention.input_norm_bytes_read > 0);
            assert!(attention.projection_bytes_read > 0);
            assert!(attention.o_proj_bytes_read > 0);
        }
        assert!(attention.final_residual_checksum.is_finite());
    }

    #[test]
    #[ignore = "loads real DSA tensors and uses the DSA KV-cache prefix key for candidate selection"]
    fn real_checkpoint_mla_rope_attention_consumes_dsa_kv_cache_candidate_when_available() {
        let Some(catalog) = load_real_catalog_or_skip() else {
            return;
        };
        let prefix_hidden = vec![0.03125_f32; GLM52_HIDDEN_SIZE];
        let current_hidden = vec![0.0625_f32; GLM52_HIDDEN_SIZE];
        let prefix_block = super::real_full_mla_rope_kv_cache_block_for_layer_from_hidden(
            &catalog,
            22,
            0,
            &prefix_hidden,
        )
        .expect("building layer-22 prefix KV cache block");

        let attention =
            super::real_full_mla_rope_attention_kv_cache_context_hidden_for_layer_from_initial(
                &catalog,
                22,
                vec![prefix_block],
                current_hidden,
            )
            .expect("running layer-22 MLA/RoPE attention with DSA KV-cache candidate context");

        assert_eq!(attention.layer_id, 22);
        assert_eq!(attention.attention_rows, 1);
        assert_eq!(attention.prefix_context_rows, 1);
        assert!(attention.uses_kv_cache_context);
        assert!(attention.includes_dsa_candidate_selection);
        assert!(attention.includes_dsa_softmax);
        assert_eq!(attention.dsa_candidate_rows, 2);
        assert_eq!(attention.dsa_selected_indices.len(), 2);
        assert!(attention.final_residual_checksum.is_finite());
    }

    #[test]
    #[ignore = "loads bounded real DSA/indexer tensors and executes candidate selection plus causal DSA attention"]
    fn real_checkpoint_dsa_indexer_attention_probe_when_available() {
        let Some(catalog) = load_real_catalog_or_skip() else {
            return;
        };

        let probe = real_full_dsa_indexer_attention_probe(&catalog);

        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "status": probe.status,
                "layer": probe.layer_id,
                "attention_rows": probe.attention_rows,
                "q_lora_rank": probe.q_lora_rank,
                "dsa_query_dim": probe.dsa_query_dim,
                "dsa_value_width": probe.dsa_value_width,
                "candidate_rows": probe.candidate_rows,
                "prefix_kv_candidate_rows": probe.prefix_kv_candidate_rows,
                "kv_cache_candidate_bytes": probe.kv_cache_candidate_bytes,
                "dsa_top_k": probe.dsa_top_k,
                "selected_indices": probe.selected_indices,
                "score_order": probe.score_order,
                "causal_attention_scores": probe.causal_attention_scores,
                "dsa_softmax_rows": probe.dsa_softmax_rows,
                "attention_context_values": probe.attention_context_values,
                "rope_theta": probe.rope_theta,
                "uses_real_indexer_weights": probe.uses_real_indexer_weights,
                "projection_backend": probe.projection_backend,
                "rope_backend": probe.rope_backend,
                "includes_rope": probe.includes_rope,
                "includes_dsa_candidate_selection": probe.includes_dsa_candidate_selection,
                "includes_dsa_softmax": probe.includes_dsa_softmax,
                "uses_full_model_residual": probe.uses_full_model_residual,
                "candidate_scores_checksum": probe.candidate_scores_checksum,
                "attention_weights_checksum": probe.attention_weights_checksum,
                "attention_context_checksum": probe.attention_context_checksum,
            }))
            .unwrap()
        );

        assert_eq!(probe.status, "numeric-real-bounded-dsa-indexer-attention");
        assert_eq!(probe.layer_id, 22);
        assert_eq!(probe.attention_rows, 3);
        assert_eq!(probe.q_lora_rank, 2048);
        assert_eq!(probe.dsa_query_dim, GLM52_DSA_INDEX_HEAD_DIM);
        assert_eq!(probe.dsa_value_width, 32);
        assert_eq!(probe.candidate_rows, 3);
        assert_eq!(probe.prefix_kv_candidate_rows, 0);
        assert_eq!(probe.kv_cache_candidate_bytes, 0);
        assert_eq!(probe.dsa_top_k, 3);
        assert_eq!(probe.selected_indices.len(), 3);
        assert_eq!(probe.score_order.len(), 3);
        assert_eq!(probe.causal_attention_scores, 6);
        assert_eq!(probe.dsa_softmax_rows, 3);
        assert_eq!(probe.attention_context_values, 96);
        assert_eq!(probe.rope_theta, 250_000.0);
        assert_eq!(probe.projection_backend, "cpu-reference-linear-bf16");
        assert_eq!(probe.rope_backend, "cpu-reference-rope-bf16");
        assert!(probe.uses_real_indexer_weights);
        assert!(probe.includes_rope);
        assert!(probe.includes_dsa_candidate_selection);
        assert!(probe.includes_dsa_softmax);
        assert!(!probe.uses_full_model_residual);
        assert!(probe.q_checksum.unwrap().is_finite());
        assert!(probe.q_rope_rotated_checksum.unwrap().is_finite());
        assert!(probe.k_norm_checksum.unwrap().is_finite());
        assert!(probe.k_rope_rotated_checksum.unwrap().is_finite());
        assert!(probe.value_checksum.unwrap().is_finite());
        assert!(probe.candidate_scores_checksum.unwrap().is_finite());
        assert!(
            (probe.attention_weights_checksum.unwrap() - probe.dsa_softmax_rows as f64).abs()
                < 1.0e-5
        );
        assert!(probe.attention_context_checksum.unwrap().is_finite());
        assert!(probe.passed);
    }

    fn load_real_catalog_or_skip() -> Option<TensorCatalog> {
        if !crate::commands::real_full::tests::fixture::real_checkpoint_tests_enabled() {
            crate::commands::real_full::tests::fixture::real_checkpoint_tests_skip_message(
                "attention residual",
            );
            return None;
        }
        let catalog_path =
            repo_root().join(".glmrt-cache/model-artifacts/diagnostic/model_catalog.json");
        let Ok(file) = File::open(&catalog_path) else {
            eprintln!("skipped: missing {}", catalog_path.display());
            return None;
        };
        let catalog: TensorCatalog =
            serde_json::from_reader(file).expect("parsing real GLM catalog fixture");
        if !Path::new(&catalog.snapshot_path).exists() {
            eprintln!("skipped: missing snapshot {}", catalog.snapshot_path);
            return None;
        }
        Some(catalog)
    }

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
    }
}

pub(in crate::commands::real_full) fn real_full_attention_residual_prefix_hidden_for_layer_from_initial(
    catalog: &TensorCatalog,
    layer_id: usize,
    initial_hidden: Vec<f32>,
) -> Result<RealFullAttentionResidualPrefixHidden> {
    let execution = execute_real_full_attention_residual_prefix_with_rows(
        catalog,
        layer_id,
        REAL_FULL_ATTENTION_RESIDUAL_PREFIX_VALUES,
        vec![initial_hidden],
        "single-supplied-hidden-row-carried-from-previous-residual-stage",
        "execute a bounded real GLM-5.2 BF16 attention residual prefix for one selected layer from a supplied residual hidden row; full MLA/RoPE attention remains omitted",
    )?;
    attention_hidden_from_execution(
        execution,
        "real full attention residual-prefix supplied-hidden execution produced no rows",
    )
}

pub(in crate::commands::real_full) fn real_full_attention_residual_full_output_hidden_for_layer_from_initial(
    catalog: &TensorCatalog,
    layer_id: usize,
    initial_hidden: Vec<f32>,
) -> Result<RealFullAttentionResidualPrefixHidden> {
    let execution = execute_real_full_attention_residual_prefix_with_rows(
        catalog,
        layer_id,
        GLM52_HIDDEN_SIZE,
        vec![initial_hidden],
        "single-supplied-hidden-row-carried-from-previous-residual-stage-full-output-attention",
        "execute hidden-width real GLM-5.2 BF16 attention residual rows for one selected layer from a supplied residual hidden row; full attention context, full MLA/RoPE attention, and full-model residuals remain omitted",
    )?;
    attention_hidden_from_execution(
        execution,
        "real full attention full-output supplied-hidden execution produced no rows",
    )
}

fn attention_hidden_from_execution(
    execution: AttentionResidualPrefixExecution,
    empty_rows_message: &'static str,
) -> Result<RealFullAttentionResidualPrefixHidden> {
    let Some(hidden) = execution
        .hidden_after_attention_prefix_rows
        .into_iter()
        .next()
    else {
        anyhow::bail!(empty_rows_message);
    };

    Ok(RealFullAttentionResidualPrefixHidden {
        hidden,
        device_hidden: None,
        layer_id: execution.probe.layer_id,
        attention_rows: execution.probe.attention_rows,
        prefix_context_rows: 0,
        total_context_rows: execution.probe.attention_rows,
        uses_kv_cache_context: false,
        kv_cache_context_bytes: 0,
        residual_adds: 1,
        residual_prefix_values: execution.probe.output_prefix_values,
        input_norm_bytes_read: execution.probe.input_norm_bytes_read,
        projection_bytes_read: execution.probe.projection_bytes_read,
        o_proj_bytes_read: execution.probe.o_proj_bytes_read,
        projection_backend: execution.probe.projection_backend,
        attention_backend: execution.probe.attention_backend,
        residual_add_backend: execution.probe.residual_add_backend,
        initial_residual_checksum: execution.first_row_residual_before_checksum,
        residual_delta_checksum: execution.first_row_residual_delta_checksum,
        final_residual_checksum: execution.first_row_residual_after_checksum,
        includes_causal_softmax: execution.probe.includes_causal_softmax,
        includes_mla_softmax: execution.probe.includes_mla_softmax,
        includes_dsa_candidate_selection: false,
        includes_dsa_softmax: false,
        dsa_candidate_rows: 0,
        dsa_selected_indices: Vec::new(),
        dsa_attention_context_checksum: None,
        dsa_projection_backend: None,
    })
}

fn mla_rope_attention_hidden_from_execution(
    execution: MlaRopeAttentionPrefixExecution,
    empty_rows_message: &'static str,
) -> Result<RealFullAttentionResidualPrefixHidden> {
    let Some(hidden) = execution
        .hidden_after_attention_prefix_rows
        .into_iter()
        .next()
    else {
        anyhow::bail!(empty_rows_message);
    };

    Ok(RealFullAttentionResidualPrefixHidden {
        hidden,
        device_hidden: execution.device_hidden_after_attention,
        layer_id: execution.probe.layer_id,
        attention_rows: execution.probe.attention_rows,
        prefix_context_rows: execution.probe.prefix_context_rows,
        total_context_rows: execution.probe.total_context_rows,
        uses_kv_cache_context: execution.uses_kv_cache_context,
        kv_cache_context_bytes: execution.kv_cache_context_bytes,
        residual_adds: 1,
        residual_prefix_values: execution.probe.output_prefix_values,
        input_norm_bytes_read: execution.probe.input_norm_bytes_read,
        projection_bytes_read: execution.probe.projection_bytes_read,
        o_proj_bytes_read: execution.probe.o_proj_bytes_read,
        projection_backend: execution.projection_backend,
        attention_backend: execution.probe.attention_backend,
        residual_add_backend: execution.residual_add_backend,
        initial_residual_checksum: execution.first_row_residual_before_checksum,
        residual_delta_checksum: execution.first_row_residual_delta_checksum,
        final_residual_checksum: execution.first_row_residual_after_checksum,
        includes_causal_softmax: true,
        includes_mla_softmax: execution.probe.includes_mla_softmax,
        includes_dsa_candidate_selection: false,
        includes_dsa_softmax: false,
        dsa_candidate_rows: 0,
        dsa_selected_indices: Vec::new(),
        dsa_attention_context_checksum: None,
        dsa_projection_backend: None,
    })
}

fn mla_rope_attention_rows_from_execution(
    execution: MlaRopeAttentionPrefixExecution,
    empty_rows_message: &'static str,
) -> Result<RealFullAttentionResidualPrefixRows> {
    if execution.hidden_after_attention_prefix_rows.is_empty() {
        anyhow::bail!(empty_rows_message);
    }

    Ok(RealFullAttentionResidualPrefixRows {
        hidden_rows: execution.hidden_after_attention_prefix_rows,
        layer_id: execution.probe.layer_id,
        attention_rows: execution.probe.attention_rows,
        residual_adds: execution.probe.residual_adds,
        residual_prefix_values: execution.probe.residual_prefix_values,
        input_norm_bytes_read: execution.probe.input_norm_bytes_read,
        projection_bytes_read: execution.probe.projection_bytes_read,
        o_proj_bytes_read: execution.probe.o_proj_bytes_read,
        projection_backend: execution.projection_backend,
        attention_backend: execution.probe.attention_backend,
        residual_add_backend: execution.residual_add_backend,
        initial_residual_checksum: execution.probe.residual_before_checksum.unwrap_or_default(),
        residual_delta_checksum: execution.probe.residual_delta_checksum.unwrap_or_default(),
        final_residual_checksum: execution.probe.residual_after_checksum.unwrap_or_default(),
        includes_causal_softmax: true,
        includes_mla_softmax: execution.probe.includes_mla_softmax,
    })
}

pub(in crate::commands::real_full) fn real_full_attention_residual_prefix_rows(
    catalog: &TensorCatalog,
) -> Result<RealFullAttentionResidualPrefixRows> {
    let execution = execute_real_full_attention_residual_prefix(catalog)?;
    if execution.hidden_after_attention_prefix_rows.is_empty() {
        anyhow::bail!("real full attention residual-prefix row execution produced no rows");
    }

    Ok(RealFullAttentionResidualPrefixRows {
        hidden_rows: execution.hidden_after_attention_prefix_rows,
        layer_id: execution.probe.layer_id,
        attention_rows: execution.probe.attention_rows,
        residual_adds: execution.probe.residual_adds,
        residual_prefix_values: execution.probe.residual_prefix_values,
        input_norm_bytes_read: execution.probe.input_norm_bytes_read,
        projection_bytes_read: execution.probe.projection_bytes_read,
        o_proj_bytes_read: execution.probe.o_proj_bytes_read,
        projection_backend: execution.probe.projection_backend,
        attention_backend: execution.probe.attention_backend,
        residual_add_backend: execution.probe.residual_add_backend,
        initial_residual_checksum: execution.probe.residual_before_checksum.unwrap_or_default(),
        residual_delta_checksum: execution.probe.residual_delta_checksum.unwrap_or_default(),
        final_residual_checksum: execution.probe.residual_after_checksum.unwrap_or_default(),
        includes_causal_softmax: execution.probe.includes_causal_softmax,
        includes_mla_softmax: execution.probe.includes_mla_softmax,
    })
}

fn skipped_real_full_attention_residual_prefix_probe(
    status: &'static str,
    mode: AttentionResidualProbeMode,
    skipped_reason: Option<String>,
) -> RealFullAttentionResidualPrefixProbe {
    RealFullAttentionResidualPrefixProbe {
        status,
        scope: mode.scope,
        layer_id: REAL_FULL_ATTENTION_RESIDUAL_LAYER_ID,
        hidden_source: "not-run",
        context_source: "not-run",
        q_lora_rank: 0,
        kv_lora_rank: 0,
        attention_rows: REAL_FULL_ATTENTION_RESIDUAL_ROWS,
        output_prefix_values: mode.output_count,
        residual_prefix_values: 0,
        residual_adds: 0,
        causal_attention_scores: 0,
        causal_softmax_rows: 0,
        input_norm_tensors_read: 0,
        attention_tensors_read: 0,
        input_norm_bytes_read: 0,
        projection_bytes_read: 0,
        o_proj_bytes_read: 0,
        projection_backend: "not-run",
        attention_backend: "not-run",
        residual_add_backend: "not-run",
        q_output_checksum: None,
        kv_output_checksum: None,
        kv_rope_checksum: None,
        attention_output_checksum: None,
        attention_output_l2_norm: None,
        residual_before_checksum: None,
        residual_delta_checksum: None,
        residual_after_checksum: None,
        first_residual_after: None,
        last_residual_after: None,
        uses_real_attention_weights: false,
        applies_attention_residual_prefix: false,
        includes_causal_softmax: false,
        includes_mla_softmax: false,
        uses_full_model_residual: false,
        passed: false,
        skipped_reason,
    }
}

fn skipped_real_full_mla_rope_attention_probe(
    status: &'static str,
    skipped_reason: Option<String>,
) -> RealFullMlaRopeAttentionProbe {
    RealFullMlaRopeAttentionProbe {
        status,
        scope: "execute bounded real GLM-5.2 main MLA attention math with separated q_nope/q_rope, compressed KV latent, RoPE-applied shared k_rope, per-head MLA softmax, and a bounded o_proj residual prefix; full context, DSA/indexer attention, and full-model residuals remain incomplete",
        layer_id: REAL_FULL_ATTENTION_RESIDUAL_LAYER_ID,
        hidden_source: "not-run",
        context_source: "not-run",
        attention_rows: REAL_FULL_ATTENTION_RESIDUAL_ROWS,
        prefix_context_rows: 0,
        total_context_rows: 0,
        attention_heads: REAL_FULL_MLA_ROPE_PROBE_HEADS,
        q_lora_rank: 0,
        kv_lora_rank: 0,
        qk_nope_head_dim: REAL_FULL_MLA_QK_NOPE_HEAD_DIM,
        qk_rope_head_dim: GLM52_MLA_QK_ROPE_HEAD_DIM,
        q_head_dim: REAL_FULL_MLA_QK_NOPE_HEAD_DIM + GLM52_MLA_QK_ROPE_HEAD_DIM,
        v_head_dim: REAL_FULL_MLA_V_HEAD_DIM,
        compressed_kv_width: GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM,
        compressed_kv_values: 0,
        causal_attention_scores: 0,
        mla_softmax_rows: 0,
        attention_context_values: 0,
        output_prefix_values: REAL_FULL_MLA_OUTPUT_PREFIX_VALUES,
        residual_prefix_values: 0,
        residual_adds: 0,
        rope_theta: REAL_FULL_MLA_ROPE_THETA,
        input_norm_bytes_read: 0,
        projection_bytes_read: 0,
        o_proj_bytes_read: 0,
        rope_backend: "not-run",
        attention_backend: "not-run",
        q_nope_checksum: None,
        q_rope_checksum: None,
        q_rope_rotated_checksum: None,
        k_nope_checksum: None,
        k_rope_checksum: None,
        k_rope_rotated_checksum: None,
        value_checksum: None,
        attention_scores_checksum: None,
        attention_weights_checksum: None,
        attention_context_checksum: None,
        attention_output_checksum: None,
        attention_output_l2_norm: None,
        residual_before_checksum: None,
        residual_delta_checksum: None,
        residual_after_checksum: None,
        uses_real_attention_weights: false,
        includes_rope: false,
        includes_mla_softmax: false,
        applies_attention_residual_prefix: false,
        uses_full_model_residual: false,
        passed: false,
        skipped_reason,
    }
}

fn run_real_full_attention_residual_prefix_probe_with_mode(
    catalog: &TensorCatalog,
    mode: AttentionResidualProbeMode,
) -> Result<RealFullAttentionResidualPrefixProbe> {
    Ok(execute_real_full_attention_residual_prefix_with_rows(
        catalog,
        REAL_FULL_ATTENTION_RESIDUAL_LAYER_ID,
        mode.output_count,
        deterministic_attention_hidden_rows(),
        mode.hidden_source,
        mode.scope,
    )?
    .probe)
}

fn record_stage_backend(
    current: &mut Option<&'static str>,
    observed: &'static str,
    stage_name: &str,
    op_name: &str,
) -> Result<()> {
    match current {
        Some(existing) => {
            if let Some(canonical) = compatible_stage_backend(*existing, observed) {
                *existing = canonical;
            } else {
                anyhow::bail!(
                    "real full {stage_name} mixed coordinator backends: first={} {op_name}={observed}",
                    *existing
                );
            }
        }
        None => *current = Some(observed),
    }
    Ok(())
}

fn compatible_stage_backend(
    existing: &'static str,
    observed: &'static str,
) -> Option<&'static str> {
    if existing == observed {
        return Some(existing);
    }
    if is_cuda_linear_bf16_resident_family(existing)
        && is_cuda_linear_bf16_resident_family(observed)
    {
        return Some(CUDA_REFERENCE_LINEAR_BF16_BACKEND);
    }
    None
}

fn is_cuda_linear_bf16_resident_family(backend: &'static str) -> bool {
    matches!(
        backend,
        CUDA_REFERENCE_LINEAR_BF16_BACKEND
            | CUDA_REFERENCE_LINEAR_BF16_RESIDENT_WEIGHT_BACKEND
            | CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
    )
}

fn required_stage_backend(
    backend: Option<&'static str>,
    stage_name: &str,
    backend_name: &str,
) -> Result<&'static str> {
    backend.ok_or_else(|| {
        anyhow::anyhow!("real full {stage_name} did not record a {backend_name} backend")
    })
}

fn catalog_tensor<'a>(catalog: &'a TensorCatalog, name: &str) -> Result<&'a TensorInfo> {
    catalog
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| anyhow::anyhow!("missing tensor {name} in real full catalog"))
}

fn preload_bf16_tensor_resident_from_host_staging(
    catalog: &TensorCatalog,
    tensor_name: &str,
    expected_shape: &[usize],
    label: &'static str,
) -> Result<u64> {
    let expected_bytes = expected_shape
        .iter()
        .try_fold(1_usize, |acc, dim| acc.checked_mul(*dim))
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("real full attention BF16 tensor byte length overflows usize")?;
    let mut bytes_read = 0_u64;
    preload_resident_weight_from_host_staging(tensor_name, expected_bytes, label, |staging| {
        let summary = read_tensor_bytes_into(catalog, tensor_name, staging).with_context(|| {
            format!("reading attention tensor {tensor_name} into pinned staging")
        })?;
        if summary.dtype != DType::Bf16 {
            anyhow::bail!(
                "real full attention tensor {tensor_name} expects BF16, got {:?}",
                summary.dtype
            );
        }
        if summary.shape != expected_shape {
            anyhow::bail!(
                "real full attention tensor {tensor_name} shape mismatch: expected {:?} got {:?}",
                expected_shape,
                summary.shape
            );
        }
        if summary.bytes_read as usize != expected_bytes {
            anyhow::bail!(
                "real full attention tensor {tensor_name} read {} bytes, expected {}",
                summary.bytes_read,
                expected_bytes
            );
        }
        bytes_read = summary.bytes_read;
        Ok(())
    })
    .with_context(|| format!("preloading attention tensor {tensor_name} from pinned staging"))?;
    Ok(bytes_read)
}

fn preload_bf16_rows_resident_from_host_staging(
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
        .context("real full attention BF16 row-window byte length overflows usize")?;
    let mut bytes_read = 0_u64;
    preload_resident_weight_from_host_staging(resident_name, expected_bytes, label, |staging| {
        let summary = read_tensor_rows_into(catalog, tensor_name, 0, row_count, staging)
            .with_context(|| {
                format!(
                    "reading attention row window {tensor_name}[rows=0..{row_count}] into pinned staging"
                )
            })?;
        if summary.dtype != DType::Bf16 {
            anyhow::bail!(
                "real full attention row-window tensor {tensor_name} expects BF16, got {:?}",
                summary.dtype
            );
        }
        if summary.row_count != row_count || summary.row_width != row_width {
            anyhow::bail!(
                "real full attention row-window tensor {tensor_name} shape mismatch: expected rows={} width={} got rows={} width={}",
                row_count,
                row_width,
                summary.row_count,
                summary.row_width
            );
        }
        if summary.bytes_read as usize != expected_bytes {
            anyhow::bail!(
                "real full attention row-window tensor {tensor_name} read {} bytes, expected {}",
                summary.bytes_read,
                expected_bytes
            );
        }
        bytes_read = summary.bytes_read;
        Ok(())
    })
    .with_context(|| {
        format!("preloading attention row-window tensor {resident_name} from pinned staging")
    })?;
    Ok(bytes_read)
}

fn preload_bf16_row_prefix_resident_from_host_staging(
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
        .context("real full attention BF16 row-prefix byte length overflows usize")?;
    let mut bytes_read = 0_u64;
    preload_resident_weight_from_host_staging(resident_name, expected_bytes, label, |staging| {
        let summary =
            read_tensor_row_prefix_into(catalog, tensor_name, 0, row_count, prefix_width, staging)
                .with_context(|| {
                    format!(
                        "reading attention row prefix {tensor_name}[rows=0..{row_count}, cols=0..{prefix_width}] into pinned staging"
                    )
                })?;
        if summary.dtype != DType::Bf16 {
            anyhow::bail!(
                "real full attention row-prefix tensor {tensor_name} expects BF16, got {:?}",
                summary.dtype
            );
        }
        if summary.row_count != row_count || summary.row_width != prefix_width {
            anyhow::bail!(
                "real full attention row-prefix tensor {tensor_name} shape mismatch: expected rows={} prefix_width={} got rows={} width={}",
                row_count,
                prefix_width,
                summary.row_count,
                summary.row_width
            );
        }
        if summary.bytes_read as usize != expected_bytes {
            anyhow::bail!(
                "real full attention row-prefix tensor {tensor_name} read {} bytes, expected {}",
                summary.bytes_read,
                expected_bytes
            );
        }
        bytes_read = summary.bytes_read;
        Ok(())
    })
    .with_context(|| {
        format!("preloading attention row-prefix tensor {resident_name} from pinned staging")
    })?;
    Ok(bytes_read)
}

fn validate_bf16_matrix_tensor(
    info: &TensorInfo,
    context: &str,
    layer_id: usize,
) -> Result<(usize, usize)> {
    if info.dtype != DType::Bf16 || info.shape.len() != 2 {
        anyhow::bail!(
            "{context} expected rank-2 BF16 tensor {} for layer {layer_id}, got dtype={:?} shape={:?}",
            info.name,
            info.dtype,
            info.shape
        );
    }
    Ok((info.shape[0], info.shape[1]))
}

fn attention_residual_add_bf16(
    residual: &[f32],
    delta: &[f32],
    workspace: &mut AttentionResidualAddWorkspace,
) -> Result<AttentionResidualAddResult> {
    if residual.len() != delta.len() {
        anyhow::bail!(
            "real full attention residual-add length mismatch: residual={} delta={}",
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
    Ok(AttentionResidualAddResult { values, backend })
}

fn execute_real_full_mla_rope_attention_probe(
    catalog: &TensorCatalog,
) -> Result<RealFullMlaRopeAttentionProbe> {
    Ok(execute_real_full_mla_rope_attention_prefix_with_rows(
        catalog,
        REAL_FULL_ATTENTION_RESIDUAL_LAYER_ID,
        REAL_FULL_MLA_OUTPUT_PREFIX_VALUES,
        REAL_FULL_MLA_ROPE_PROBE_HEADS,
        deterministic_attention_hidden_rows(),
        "two-deterministic-hidden-shaped-f32-rows",
        "execute bounded real GLM-5.2 main MLA attention math with separated q_nope/q_rope, compressed KV latent, RoPE-applied shared k_rope, per-head MLA softmax, and a bounded o_proj residual prefix; full context, DSA/indexer attention, and full-model residuals remain incomplete",
        "bounded-main-mla-rope-two-head-causal-context",
    )?
    .probe)
}

fn execute_real_full_mla_rope_attention_prefix_with_rows(
    catalog: &TensorCatalog,
    layer_id: usize,
    output_count: usize,
    attention_heads: usize,
    hidden_rows: Vec<Vec<f32>>,
    hidden_source: &'static str,
    scope: &'static str,
    context_source: &'static str,
) -> Result<MlaRopeAttentionPrefixExecution> {
    execute_real_full_mla_rope_attention_prefix_with_context_rows(
        catalog,
        layer_id,
        output_count,
        attention_heads,
        Vec::new(),
        Vec::new(),
        hidden_rows,
        hidden_source,
        scope,
        context_source,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_real_full_mla_rope_attention_prefix_with_context_rows(
    catalog: &TensorCatalog,
    layer_id: usize,
    output_count: usize,
    attention_heads: usize,
    prefix_context_rows: Vec<Vec<f32>>,
    prefix_kv_blocks: Vec<RealFullMlaRopeKvCacheBlock>,
    hidden_rows: Vec<Vec<f32>>,
    hidden_source: &'static str,
    scope: &'static str,
    context_source: &'static str,
    current_hidden_device_rows: Option<&DeviceBf16Output>,
) -> Result<MlaRopeAttentionPrefixExecution> {
    if hidden_rows.is_empty() {
        anyhow::bail!("real full MLA/RoPE attention requires at least one hidden row");
    }
    if output_count == 0 || output_count > GLM52_HIDDEN_SIZE {
        anyhow::bail!(
            "real full MLA/RoPE attention output_count={} is invalid for hidden {}",
            output_count,
            GLM52_HIDDEN_SIZE
        );
    }
    if attention_heads == 0 || attention_heads > REAL_FULL_MLA_NUM_ATTENTION_HEADS {
        anyhow::bail!(
            "real full MLA/RoPE attention head count {} is invalid for model heads {}",
            attention_heads,
            REAL_FULL_MLA_NUM_ATTENTION_HEADS
        );
    }
    for (row_index, hidden) in hidden_rows.iter().enumerate() {
        if hidden.len() != GLM52_HIDDEN_SIZE {
            anyhow::bail!(
                "real full MLA/RoPE attention hidden row {row_index} width mismatch: expected {} got {}",
                GLM52_HIDDEN_SIZE,
                hidden.len()
            );
        }
    }
    if let Some(device_rows) = current_hidden_device_rows {
        if device_rows.rows != hidden_rows.len() || device_rows.values_per_row != GLM52_HIDDEN_SIZE
        {
            anyhow::bail!(
                "real full MLA/RoPE attention device hidden row shape mismatch: expected rows={} width={} got rows={} width={}",
                hidden_rows.len(),
                GLM52_HIDDEN_SIZE,
                device_rows.rows,
                device_rows.values_per_row
            );
        }
    }
    for (row_index, hidden) in prefix_context_rows.iter().enumerate() {
        if hidden.len() != GLM52_HIDDEN_SIZE {
            anyhow::bail!(
                "real full MLA/RoPE attention prefix context row {row_index} width mismatch: expected {} got {}",
                GLM52_HIDDEN_SIZE,
                hidden.len()
            );
        }
    }
    if !prefix_context_rows.is_empty() && !prefix_kv_blocks.is_empty() {
        anyhow::bail!(
            "real full MLA/RoPE attention accepts supplied prefix hidden rows or KV-cache prefix blocks, not both"
        );
    }
    if current_hidden_device_rows.is_some() && !prefix_context_rows.is_empty() {
        anyhow::bail!(
            "real full MLA/RoPE attention device hidden rows only cover current rows; supplied prefix hidden rows require host execution"
        );
    }
    for (block_index, block) in prefix_kv_blocks.iter().enumerate() {
        if block.token_count == 0 {
            anyhow::bail!("real full MLA/RoPE KV-cache prefix block {block_index} has zero tokens");
        }
        if block.bytes.is_empty() {
            anyhow::bail!("real full MLA/RoPE KV-cache prefix block {block_index} has no bytes");
        }
        if block.bytes.len() % block.token_count != 0 {
            anyhow::bail!(
                "real full MLA/RoPE KV-cache prefix block {block_index} byte count {} is not divisible by token_count {}",
                block.bytes.len(),
                block.token_count
            );
        }
    }

    let prefix_kv_row_count = prefix_kv_blocks
        .iter()
        .map(|block| block.token_count)
        .sum::<usize>();
    let prefix_context_row_count = prefix_kv_row_count + prefix_context_rows.len();
    let total_context_rows = prefix_context_row_count + hidden_rows.len();
    let mut context_hidden_rows = Vec::with_capacity(prefix_context_rows.len() + hidden_rows.len());
    context_hidden_rows.extend(prefix_context_rows.iter().cloned());
    context_hidden_rows.extend(hidden_rows.iter().cloned());
    let hidden_position_start = next_hidden_position_after_prefix_kv_blocks(&prefix_kv_blocks);
    let hidden_positions = (hidden_position_start
        ..hidden_position_start + context_hidden_rows.len())
        .collect::<Vec<_>>();
    let q_head_dim = REAL_FULL_MLA_QK_NOPE_HEAD_DIM + GLM52_MLA_QK_ROPE_HEAD_DIM;
    let q_b_rows = attention_heads * q_head_dim;
    let kv_b_rows = attention_heads * (REAL_FULL_MLA_QK_NOPE_HEAD_DIM + REAL_FULL_MLA_V_HEAD_DIM);
    let context_width = attention_heads * REAL_FULL_MLA_V_HEAD_DIM;

    let input_norm_name = format!("model.layers.{layer_id}.input_layernorm.weight");
    let q_a_name = format!("model.layers.{layer_id}.self_attn.q_a_proj.weight");
    let q_a_norm_name = format!("model.layers.{layer_id}.self_attn.q_a_layernorm.weight");
    let q_b_name = format!("model.layers.{layer_id}.self_attn.q_b_proj.weight");
    let kv_a_name = format!("model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.weight");
    let kv_a_norm_name = format!("model.layers.{layer_id}.self_attn.kv_a_layernorm.weight");
    let kv_b_name = format!("model.layers.{layer_id}.self_attn.kv_b_proj.weight");
    let o_proj_name = format!("model.layers.{layer_id}.self_attn.o_proj.weight");

    let input_norm_info = catalog_tensor(catalog, &input_norm_name)?;
    let q_a_info = catalog_tensor(catalog, &q_a_name)?;
    let kv_a_info = catalog_tensor(catalog, &kv_a_name)?;
    let q_a_norm_info = catalog_tensor(catalog, &q_a_norm_name)?;
    let kv_a_norm_info = catalog_tensor(catalog, &kv_a_norm_name)?;
    let q_b_info = catalog_tensor(catalog, &q_b_name)?;
    let kv_b_info = catalog_tensor(catalog, &kv_b_name)?;
    let o_proj_info = catalog_tensor(catalog, &o_proj_name)?;
    let q_b_weight_key = format!("{q_b_name}[rows=0..{q_b_rows}]");
    let kv_b_weight_key = format!("{kv_b_name}[rows=0..{kv_b_rows}]");
    let o_proj_weight_key = format!("{o_proj_name}[rows=0..{output_count}]");

    if input_norm_info.dtype != DType::Bf16
        || q_a_info.dtype != DType::Bf16
        || q_a_norm_info.dtype != DType::Bf16
        || q_b_info.dtype != DType::Bf16
        || kv_a_info.dtype != DType::Bf16
        || kv_a_norm_info.dtype != DType::Bf16
        || kv_b_info.dtype != DType::Bf16
        || o_proj_info.dtype != DType::Bf16
    {
        anyhow::bail!(
            "real full MLA/RoPE probe expects BF16 attention tensors for layer {layer_id}"
        );
    }
    if input_norm_info.shape != vec![GLM52_HIDDEN_SIZE] {
        anyhow::bail!(
            "real full MLA/RoPE input norm shape mismatch for layer {layer_id}: {:?}",
            input_norm_info.shape
        );
    }
    if q_a_info.shape.len() != 2 || kv_a_info.shape.len() != 2 {
        anyhow::bail!(
            "real full MLA/RoPE probe expected rank-2 q_a/kv_a weights, got {:?} and {:?}",
            q_a_info.shape,
            kv_a_info.shape
        );
    }
    let q_lora_rank = q_a_info.shape[0];
    let kv_a_rows = kv_a_info.shape[0];
    let kv_lora_rank = kv_a_norm_info.shape.first().copied().unwrap_or_default();
    if q_a_info.shape[1] != GLM52_HIDDEN_SIZE || kv_a_info.shape[1] != GLM52_HIDDEN_SIZE {
        anyhow::bail!(
            "real full MLA/RoPE hidden width mismatch: hidden={} q_a_width={} kv_a_width={}",
            GLM52_HIDDEN_SIZE,
            q_a_info.shape[1],
            kv_a_info.shape[1]
        );
    }
    if q_lora_rank == 0 || q_a_norm_info.shape != vec![q_lora_rank] {
        anyhow::bail!(
            "real full MLA/RoPE q_a norm shape mismatch: {:?} for q rank {q_lora_rank}",
            q_a_norm_info.shape
        );
    }
    if kv_lora_rank != GLM52_MLA_KV_LORA_RANK
        || kv_a_rows != GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM
        || kv_a_norm_info.shape != vec![kv_lora_rank]
    {
        anyhow::bail!(
            "real full MLA/RoPE compressed KV shape mismatch: kv_a_rows={kv_a_rows} kv_norm_shape={:?}",
            kv_a_norm_info.shape
        );
    }
    let (q_b_full_rows, q_b_row_width) =
        validate_bf16_matrix_tensor(q_b_info, "real full MLA/RoPE q_b", layer_id)?;
    let (kv_b_full_rows, kv_b_row_width) =
        validate_bf16_matrix_tensor(kv_b_info, "real full MLA/RoPE kv_b", layer_id)?;
    let (o_proj_full_rows, o_proj_row_width) =
        validate_bf16_matrix_tensor(o_proj_info, "real full MLA/RoPE o_proj", layer_id)?;
    if q_b_row_width != q_lora_rank
        || q_b_full_rows != REAL_FULL_MLA_NUM_ATTENTION_HEADS * q_head_dim
    {
        anyhow::bail!(
            "real full MLA/RoPE q_b shape mismatch: rows={} row_width={} q_rank={q_lora_rank}",
            q_b_full_rows,
            q_b_row_width
        );
    }
    if kv_b_row_width != kv_lora_rank
        || kv_b_full_rows
            != REAL_FULL_MLA_NUM_ATTENTION_HEADS
                * (REAL_FULL_MLA_QK_NOPE_HEAD_DIM + REAL_FULL_MLA_V_HEAD_DIM)
    {
        anyhow::bail!(
            "real full MLA/RoPE kv_b shape mismatch: rows={} row_width={} kv_rank={kv_lora_rank}",
            kv_b_full_rows,
            kv_b_row_width
        );
    }
    if o_proj_row_width != REAL_FULL_MLA_NUM_ATTENTION_HEADS * REAL_FULL_MLA_V_HEAD_DIM {
        anyhow::bail!(
            "real full MLA/RoPE o_proj row width mismatch: {}",
            o_proj_row_width
        );
    }
    if output_count > o_proj_full_rows {
        anyhow::bail!(
            "real full MLA/RoPE o_proj row prefix exceeds full rows: output_count={output_count} full_rows={o_proj_full_rows}"
        );
    }
    let o_proj_row_window_weight_key = if context_width == o_proj_row_width {
        o_proj_weight_key.clone()
    } else {
        format!("{o_proj_name}[rows=0..{output_count},cols=0..{context_width}]")
    };
    let cuda_reference_enabled = coordinator_cuda_reference_kernels_enabled();
    let mut input_norm_bytes_read = 0_u64;
    let mut projection_bytes_read = 0_u64;
    let mut o_proj_bytes_read = 0_u64;
    let input_norm_full_resident =
        math::bf16_full_vector_resident_available(&input_norm_name, GLM52_HIDDEN_SIZE);
    let input_norm = if input_norm_full_resident {
        None
    } else if cuda_reference_enabled {
        input_norm_bytes_read = preload_bf16_tensor_resident_from_host_staging(
            catalog,
            &input_norm_name,
            &[GLM52_HIDDEN_SIZE],
            "BF16 MLA/RoPE attention input norm pinned staging",
        )?;
        None
    } else {
        let tensor = load_tensor_bytes(catalog, &input_norm_name)?;
        input_norm_bytes_read = tensor.bytes.len() as u64;
        Some(tensor)
    };
    let q_a_norm_full_resident =
        math::bf16_full_vector_resident_available(&q_a_norm_name, q_lora_rank);
    let q_a_norm = if q_a_norm_full_resident {
        None
    } else if cuda_reference_enabled {
        projection_bytes_read += preload_bf16_tensor_resident_from_host_staging(
            catalog,
            &q_a_norm_name,
            &[q_lora_rank],
            "BF16 MLA/RoPE attention q_a norm pinned staging",
        )?;
        None
    } else {
        let tensor = load_tensor_bytes(catalog, &q_a_norm_name)?;
        projection_bytes_read += tensor.bytes.len() as u64;
        Some(tensor)
    };
    let kv_a_norm_full_resident =
        math::bf16_full_vector_resident_available(&kv_a_norm_name, kv_lora_rank);
    let kv_a_norm = if kv_a_norm_full_resident {
        None
    } else if cuda_reference_enabled {
        projection_bytes_read += preload_bf16_tensor_resident_from_host_staging(
            catalog,
            &kv_a_norm_name,
            &[kv_lora_rank],
            "BF16 MLA/RoPE attention kv_a norm pinned staging",
        )?;
        None
    } else {
        let tensor = load_tensor_bytes(catalog, &kv_a_norm_name)?;
        projection_bytes_read += tensor.bytes.len() as u64;
        Some(tensor)
    };
    let q_a_full_resident = bf16_full_row_prefix_resident_available(
        &q_a_name,
        q_lora_rank,
        GLM52_HIDDEN_SIZE,
        q_lora_rank,
    );
    let kv_a_full_resident = bf16_full_row_prefix_resident_available(
        &kv_a_name,
        kv_a_rows,
        GLM52_HIDDEN_SIZE,
        kv_a_rows,
    );
    let q_b_full_resident =
        bf16_full_row_prefix_resident_available(&q_b_name, q_b_full_rows, q_lora_rank, q_b_rows);
    let kv_b_full_resident = bf16_full_row_prefix_resident_available(
        &kv_b_name,
        kv_b_full_rows,
        kv_lora_rank,
        kv_b_rows,
    );
    let o_proj_full_resident = bf16_full_row_prefix_resident_available(
        &o_proj_name,
        o_proj_full_rows,
        o_proj_row_width,
        output_count,
    );
    let q_b = if q_b_full_resident {
        None
    } else if cuda_reference_enabled {
        projection_bytes_read += preload_bf16_rows_resident_from_host_staging(
            catalog,
            &q_b_name,
            &q_b_weight_key,
            q_b_rows,
            q_lora_rank,
            "BF16 MLA/RoPE attention q_b row-window pinned staging",
        )?;
        None
    } else {
        let rows = load_tensor_rows(catalog, &q_b_name, 0, q_b_rows)?;
        projection_bytes_read += rows.bytes.len() as u64;
        Some(rows)
    };
    let kv_b = if kv_b_full_resident {
        None
    } else if cuda_reference_enabled {
        projection_bytes_read += preload_bf16_rows_resident_from_host_staging(
            catalog,
            &kv_b_name,
            &kv_b_weight_key,
            kv_b_rows,
            kv_lora_rank,
            "BF16 MLA/RoPE attention kv_b row-window pinned staging",
        )?;
        None
    } else {
        let rows = load_tensor_rows(catalog, &kv_b_name, 0, kv_b_rows)?;
        projection_bytes_read += rows.bytes.len() as u64;
        Some(rows)
    };
    let o_proj = if o_proj_full_resident {
        None
    } else if cuda_reference_enabled {
        o_proj_bytes_read = preload_bf16_row_prefix_resident_from_host_staging(
            catalog,
            &o_proj_name,
            &o_proj_row_window_weight_key,
            output_count,
            context_width,
            "BF16 MLA/RoPE attention o_proj row-prefix pinned staging",
        )?;
        projection_bytes_read += o_proj_bytes_read;
        None
    } else {
        let rows = load_tensor_rows(catalog, &o_proj_name, 0, output_count)?;
        o_proj_bytes_read = rows.bytes.len() as u64;
        projection_bytes_read += o_proj_bytes_read;
        Some(rows)
    };
    let o_proj_row_window_resident = !o_proj_full_resident && cuda_reference_enabled;
    let q_a = if q_a_full_resident {
        None
    } else if cuda_reference_enabled {
        projection_bytes_read += preload_bf16_tensor_resident_from_host_staging(
            catalog,
            &q_a_name,
            &[q_lora_rank, GLM52_HIDDEN_SIZE],
            "BF16 MLA/RoPE attention q_a pinned staging",
        )?;
        None
    } else {
        let tensor = load_tensor_bytes(catalog, &q_a_name)?;
        projection_bytes_read += tensor.bytes.len() as u64;
        Some(tensor)
    };
    let kv_a = if kv_a_full_resident {
        None
    } else if cuda_reference_enabled {
        projection_bytes_read += preload_bf16_tensor_resident_from_host_staging(
            catalog,
            &kv_a_name,
            &[kv_a_rows, GLM52_HIDDEN_SIZE],
            "BF16 MLA/RoPE attention kv_a pinned staging",
        )?;
        None
    } else {
        let tensor = load_tensor_bytes(catalog, &kv_a_name)?;
        projection_bytes_read += tensor.bytes.len() as u64;
        Some(tensor)
    };

    let mut q_nope_rows = Vec::with_capacity(total_context_rows);
    let mut q_rope_rows = Vec::with_capacity(total_context_rows);
    let mut q_rope_rotated_rows = Vec::with_capacity(total_context_rows);
    let mut q_projected_current_values = Vec::with_capacity(context_hidden_rows.len() * q_b_rows);
    let mut k_nope_rows = Vec::with_capacity(total_context_rows);
    let mut k_rope_rows = Vec::with_capacity(total_context_rows);
    let mut k_rope_rotated_rows = Vec::with_capacity(total_context_rows);
    let mut value_rows = Vec::with_capacity(total_context_rows);
    let mut rope_backend = None;
    let mut projection_backend = None;
    let main_kv_cache_bytes = (GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM) * 2;
    let dsa_kv_cache_bytes = if GLM52_DSA_INDEXER_LAYER_IDS_WITH_MTP.contains(&layer_id) {
        GLM52_DSA_INDEX_HEAD_DIM * 2
    } else {
        0
    };
    let expected_kv_cache_bytes_per_token = main_kv_cache_bytes + dsa_kv_cache_bytes;

    for (block_index, block) in prefix_kv_blocks.iter().enumerate() {
        let bytes_per_token = block.bytes.len() / block.token_count;
        if bytes_per_token != expected_kv_cache_bytes_per_token {
            anyhow::bail!(
                "real full MLA/RoPE KV-cache prefix block {block_index} bytes/token mismatch for layer {layer_id}: expected {} got {}",
                expected_kv_cache_bytes_per_token,
                bytes_per_token
            );
        }
        for token_offset in 0..block.token_count {
            let row_start = token_offset * bytes_per_token;
            let main_end = row_start + main_kv_cache_bytes;
            let kv_values = bf16_bytes_to_f32(&block.bytes[row_start..main_end])?;
            let kv_latent = &kv_values[..kv_lora_rank];
            let k_rope = kv_values[kv_lora_rank..].to_vec();
            let position = block.token_start + token_offset;
            let k_rope_rotated =
                apply_rope_row_with_backend(layer_id, &k_rope, position, REAL_FULL_MLA_ROPE_THETA)?;
            record_stage_backend(
                &mut rope_backend,
                k_rope_rotated.backend,
                "attention RoPE",
                "kv_cache_k_rope",
            )?;
            let kv_a_normalized = rmsnorm_bf16_with_optional_preloaded_resident_weight(
                &kv_a_norm_name,
                &bf16_bytes_from_f32(kv_latent),
                kv_a_norm.as_ref().map(|tensor| tensor.bytes.as_slice()),
                1,
                kv_latent.len(),
                REAL_FULL_DENSE_RMSNORM_EPS,
            )?
            .values;
            let kv_projected = project_rows_bf16_with_optional_preloaded_prefix_weight(
                &kv_b_name,
                &kv_b_weight_key,
                &kv_a_normalized,
                kv_b.as_ref().map(|rows| rows.bytes.as_slice()),
                kv_b_rows,
                kv_lora_rank,
                kv_b_full_rows,
            )?;
            record_stage_backend(
                &mut projection_backend,
                kv_projected.backend,
                "MLA/RoPE attention projection",
                "kv_cache_kv_b",
            )?;

            let mut k_nope_heads = Vec::with_capacity(attention_heads);
            let mut value_heads = Vec::with_capacity(attention_heads);
            for head in 0..attention_heads {
                let kv_start = head * (REAL_FULL_MLA_QK_NOPE_HEAD_DIM + REAL_FULL_MLA_V_HEAD_DIM);
                let kv_nope_end = kv_start + REAL_FULL_MLA_QK_NOPE_HEAD_DIM;
                let kv_end = kv_nope_end + REAL_FULL_MLA_V_HEAD_DIM;
                k_nope_heads.push(kv_projected.values[kv_start..kv_nope_end].to_vec());
                value_heads.push(kv_projected.values[kv_nope_end..kv_end].to_vec());
            }
            q_nope_rows.push(vec![
                vec![0.0; REAL_FULL_MLA_QK_NOPE_HEAD_DIM];
                attention_heads
            ]);
            q_rope_rows.push(vec![vec![0.0; GLM52_MLA_QK_ROPE_HEAD_DIM]; attention_heads]);
            q_rope_rotated_rows.push(vec![vec![0.0; GLM52_MLA_QK_ROPE_HEAD_DIM]; attention_heads]);
            k_nope_rows.push(k_nope_heads);
            k_rope_rows.push(k_rope);
            k_rope_rotated_rows.push(k_rope_rotated.values);
            value_rows.push(value_heads);
        }
    }

    for (row_index, hidden) in context_hidden_rows.iter().enumerate() {
        let position = hidden_positions[row_index];
        let normalized = rmsnorm_bf16_with_optional_preloaded_resident_weight(
            &input_norm_name,
            &bf16_bytes_from_f32(hidden),
            input_norm.as_ref().map(|tensor| tensor.bytes.as_slice()),
            1,
            hidden.len(),
            REAL_FULL_DENSE_RMSNORM_EPS,
        )?
        .values;
        let q_a_projected = project_rows_bf16_with_optional_preloaded_full_weight(
            &q_a_name,
            &normalized,
            q_a.as_ref().map(|tensor| tensor.bytes.as_slice()),
            q_lora_rank,
            hidden.len(),
        )?;
        record_stage_backend(
            &mut projection_backend,
            q_a_projected.backend,
            "MLA/RoPE attention projection",
            "q_a",
        )?;
        let q_a_normalized = rmsnorm_bf16_with_optional_preloaded_resident_weight(
            &q_a_norm_name,
            &bf16_bytes_from_f32(&q_a_projected.values),
            q_a_norm.as_ref().map(|tensor| tensor.bytes.as_slice()),
            1,
            q_a_projected.values.len(),
            REAL_FULL_DENSE_RMSNORM_EPS,
        )?
        .values;
        let q_projected = project_rows_bf16_with_optional_preloaded_prefix_weight(
            &q_b_name,
            &q_b_weight_key,
            &q_a_normalized,
            q_b.as_ref().map(|rows| rows.bytes.as_slice()),
            q_b_rows,
            q_lora_rank,
            q_b_full_rows,
        )?;
        record_stage_backend(
            &mut projection_backend,
            q_projected.backend,
            "MLA/RoPE attention projection",
            "q_b",
        )?;
        q_projected_current_values.extend_from_slice(&q_projected.values);

        let kv_a_projected = project_rows_bf16_with_optional_preloaded_full_weight(
            &kv_a_name,
            &normalized,
            kv_a.as_ref().map(|tensor| tensor.bytes.as_slice()),
            kv_a_rows,
            hidden.len(),
        )?;
        record_stage_backend(
            &mut projection_backend,
            kv_a_projected.backend,
            "MLA/RoPE attention projection",
            "kv_a",
        )?;
        let kv_latent = &kv_a_projected.values[..kv_lora_rank];
        let k_rope = kv_a_projected.values[kv_lora_rank..].to_vec();
        let k_rope_rotated =
            apply_rope_row_with_backend(layer_id, &k_rope, position, REAL_FULL_MLA_ROPE_THETA)?;
        record_stage_backend(
            &mut rope_backend,
            k_rope_rotated.backend,
            "attention RoPE",
            "k_rope",
        )?;
        let kv_a_normalized = rmsnorm_bf16_with_optional_preloaded_resident_weight(
            &kv_a_norm_name,
            &bf16_bytes_from_f32(kv_latent),
            kv_a_norm.as_ref().map(|tensor| tensor.bytes.as_slice()),
            1,
            kv_latent.len(),
            REAL_FULL_DENSE_RMSNORM_EPS,
        )?
        .values;
        let kv_projected = project_rows_bf16_with_optional_preloaded_prefix_weight(
            &kv_b_name,
            &kv_b_weight_key,
            &kv_a_normalized,
            kv_b.as_ref().map(|rows| rows.bytes.as_slice()),
            kv_b_rows,
            kv_lora_rank,
            kv_b_full_rows,
        )?;
        record_stage_backend(
            &mut projection_backend,
            kv_projected.backend,
            "MLA/RoPE attention projection",
            "kv_b",
        )?;

        let mut q_nope_heads = Vec::with_capacity(attention_heads);
        let mut q_rope_heads = Vec::with_capacity(attention_heads);
        let mut q_rope_rotated_heads = Vec::with_capacity(attention_heads);
        let mut k_nope_heads = Vec::with_capacity(attention_heads);
        let mut value_heads = Vec::with_capacity(attention_heads);
        for head in 0..attention_heads {
            let q_start = head * q_head_dim;
            let q_nope_end = q_start + REAL_FULL_MLA_QK_NOPE_HEAD_DIM;
            let q_end = q_start + q_head_dim;
            let q_rope = q_projected.values[q_nope_end..q_end].to_vec();
            q_nope_heads.push(q_projected.values[q_start..q_nope_end].to_vec());
            let q_rope_rotated =
                apply_rope_row_with_backend(layer_id, &q_rope, position, REAL_FULL_MLA_ROPE_THETA)?;
            record_stage_backend(
                &mut rope_backend,
                q_rope_rotated.backend,
                "attention RoPE",
                "q_rope",
            )?;
            q_rope_rotated_heads.push(q_rope_rotated.values);
            q_rope_heads.push(q_rope);

            let kv_start = head * (REAL_FULL_MLA_QK_NOPE_HEAD_DIM + REAL_FULL_MLA_V_HEAD_DIM);
            let kv_nope_end = kv_start + REAL_FULL_MLA_QK_NOPE_HEAD_DIM;
            let kv_end = kv_nope_end + REAL_FULL_MLA_V_HEAD_DIM;
            k_nope_heads.push(kv_projected.values[kv_start..kv_nope_end].to_vec());
            value_heads.push(kv_projected.values[kv_nope_end..kv_end].to_vec());
        }
        q_nope_rows.push(q_nope_heads);
        q_rope_rows.push(q_rope_heads);
        q_rope_rotated_rows.push(q_rope_rotated_heads);
        k_nope_rows.push(k_nope_heads);
        k_rope_rows.push(k_rope);
        k_rope_rotated_rows.push(k_rope_rotated.values);
        value_rows.push(value_heads);
    }
    let rope_backend = required_stage_backend(rope_backend, "attention RoPE", "RoPE")?;

    let scale = (q_head_dim as f32).sqrt().recip();
    let q_nope_values = flatten_head_rows(&q_nope_rows);
    let q_rope_values = flatten_head_rows(&q_rope_rows);
    let q_rope_rotated_values = flatten_head_rows(&q_rope_rotated_rows);
    let k_nope_values = flatten_head_rows(&k_nope_rows);
    let k_rope_values = flatten_rows(&k_rope_rows);
    let k_rope_rotated_values = flatten_rows(&k_rope_rotated_rows);
    let value_values = flatten_head_rows(&value_rows);
    let mut device_prefix_attention_parts: Option<RealFullDeviceMlaAttentionParts> = None;
    let mut device_prefix_context_rows = None;
    if cuda_reference_enabled && !prefix_kv_blocks.is_empty() {
        let sequence_id = format!("real-full-mla-rope-device-prefix-layer-{layer_id}");
        let max_token_end = prefix_kv_blocks
            .iter()
            .map(|block| block.token_start + block.token_count)
            .max()
            .unwrap_or_default();
        let current_token_end = hidden_positions
            .last()
            .copied()
            .map(|position| position + 1)
            .unwrap_or(max_token_end);
        let config = KvCacheConfig::glm52_phase0(max_token_end.max(current_token_end).max(1));
        let current_kv_device_write_has_dsa =
            config.layer_has_dsa_indexer(LayerId(layer_id as u32));
        let descriptors = prefix_kv_blocks
            .iter()
            .enumerate()
            .map(|(index, block)| {
                Ok(KvBlockDescriptor {
                    reservation_id: u64::try_from(index + 1)
                        .context("MLA/RoPE device prefix reservation id overflow")?,
                    sequence_id: sequence_id.clone(),
                    layer_id: LayerId(layer_id as u32),
                    token_start: PositionId(
                        u64::try_from(block.token_start)
                            .context("MLA/RoPE device prefix token_start overflow")?,
                    ),
                    token_count: block.token_count,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let current_descriptors = if context_hidden_rows.is_empty() {
            Vec::new()
        } else {
            vec![KvBlockDescriptor {
                reservation_id: u64::try_from(descriptors.len() + 1)
                    .context("MLA/RoPE device current reservation id overflow")?,
                sequence_id: sequence_id.clone(),
                layer_id: LayerId(layer_id as u32),
                token_start: PositionId(
                    u64::try_from(hidden_positions[0])
                        .context("MLA/RoPE device current token_start overflow")?,
                ),
                token_count: context_hidden_rows.len(),
            }]
        };
        let payloads = prefix_kv_blocks
            .iter()
            .map(|block| block.bytes.clone())
            .collect::<Vec<_>>();
        let prefix_positions = prefix_kv_blocks
            .iter()
            .flat_map(|block| block.token_start..block.token_start + block.token_count)
            .map(|position| {
                u32::try_from(position).context("MLA/RoPE device prefix position overflow")
            })
            .collect::<Result<Vec<_>>>()?;
        let prefix_k_nope_values = prefix_kv_row_count
            .checked_mul(attention_heads)
            .and_then(|values| values.checked_mul(REAL_FULL_MLA_QK_NOPE_HEAD_DIM))
            .context("MLA/RoPE device prefix k_nope value offset overflow")?;
        let prefix_k_rope_values = prefix_kv_row_count
            .checked_mul(GLM52_MLA_QK_ROPE_HEAD_DIM)
            .context("MLA/RoPE device prefix k_rope value offset overflow")?;
        let prefix_value_values = prefix_kv_row_count
            .checked_mul(attention_heads)
            .and_then(|values| values.checked_mul(REAL_FULL_MLA_V_HEAD_DIM))
            .context("MLA/RoPE device prefix value offset overflow")?;
        let suffix_k_nope_bf16 = bf16_bytes_from_f32(&k_nope_values[prefix_k_nope_values..]);
        let suffix_k_rope_rotated_bf16 =
            bf16_bytes_from_f32(&k_rope_rotated_values[prefix_k_rope_values..]);
        let suffix_values_bf16 = bf16_bytes_from_f32(&value_values[prefix_value_values..]);
        let kv_norm_weight_bytes = kv_lora_rank
            .checked_mul(std::mem::size_of::<u16>())
            .context("MLA/RoPE device prefix kv norm weight bytes overflow")?;
        let kv_b_weight_bytes = kv_b_rows
            .checked_mul(kv_lora_rank)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("MLA/RoPE device prefix kv_b weight bytes overflow")?;
        let kv_b_full_bytes = kv_b_full_rows
            .checked_mul(kv_lora_rank)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("MLA/RoPE device prefix kv_b full weight bytes overflow")?;
        let resident_device_weights = if kv_a_norm.is_none() && kv_b.is_none() {
            let kv_norm_weight =
                preloaded_resident_weight_device_buffer(&kv_a_norm_name, kv_norm_weight_bytes)
                    .with_context(|| {
                        format!("resolving resident MLA/RoPE device prefix {kv_a_norm_name}")
                    })?;
            let kv_b_weight = if kv_b_full_resident {
                preloaded_resident_weight_device_buffer_view(
                    &kv_b_name,
                    kv_b_full_bytes,
                    0,
                    kv_b_weight_bytes,
                )
                .with_context(|| {
                    format!("resolving resident MLA/RoPE device prefix {kv_b_name} row view")
                })?
            } else {
                preloaded_resident_weight_device_buffer(&kv_b_weight_key, kv_b_weight_bytes)
                    .with_context(|| {
                        format!("resolving resident MLA/RoPE device prefix {kv_b_weight_key}")
                    })?
            };
            Some((kv_norm_weight, kv_b_weight))
        } else {
            None
        };
        let dsa_device_key_names = if current_kv_device_write_has_dsa {
            let wk_name = format!("model.layers.{layer_id}.self_attn.indexer.wk.weight");
            let k_norm_weight_name =
                format!("model.layers.{layer_id}.self_attn.indexer.k_norm.weight");
            let k_norm_bias_name = format!("model.layers.{layer_id}.self_attn.indexer.k_norm.bias");
            let wk_info = catalog_tensor(catalog, &wk_name)?;
            let k_norm_weight_info = catalog_tensor(catalog, &k_norm_weight_name)?;
            let k_norm_bias_info = catalog_tensor(catalog, &k_norm_bias_name)?;
            for (name, dtype) in [
                (&wk_info.name, &wk_info.dtype),
                (&k_norm_weight_info.name, &k_norm_weight_info.dtype),
                (&k_norm_bias_info.name, &k_norm_bias_info.dtype),
            ] {
                if *dtype != DType::Bf16 {
                    anyhow::bail!(
                        "real full MLA/RoPE device DSA key path expects BF16 tensor {name}, got {dtype:?}"
                    );
                }
            }
            if wk_info.shape != vec![GLM52_DSA_INDEX_HEAD_DIM, GLM52_HIDDEN_SIZE]
                || k_norm_weight_info.shape != vec![GLM52_DSA_INDEX_HEAD_DIM]
                || k_norm_bias_info.shape != vec![GLM52_DSA_INDEX_HEAD_DIM]
            {
                anyhow::bail!(
                    "real full MLA/RoPE device DSA key tensor shape mismatch for layer {layer_id}: wk={:?} k_norm_w={:?} k_norm_b={:?}",
                    wk_info.shape,
                    k_norm_weight_info.shape,
                    k_norm_bias_info.shape
                );
            }
            let wk_full_resident = bf16_full_row_prefix_resident_available(
                &wk_name,
                GLM52_DSA_INDEX_HEAD_DIM,
                GLM52_HIDDEN_SIZE,
                GLM52_DSA_INDEX_HEAD_DIM,
            );
            if !wk_full_resident {
                projection_bytes_read += preload_bf16_tensor_resident_from_host_staging(
                    catalog,
                    &wk_name,
                    &[GLM52_DSA_INDEX_HEAD_DIM, GLM52_HIDDEN_SIZE],
                    "BF16 MLA/RoPE device DSA wk pinned staging",
                )?;
            }
            let k_norm_weight_resident = math::bf16_full_vector_resident_available(
                &k_norm_weight_name,
                GLM52_DSA_INDEX_HEAD_DIM,
            );
            if !k_norm_weight_resident {
                projection_bytes_read += preload_bf16_tensor_resident_from_host_staging(
                    catalog,
                    &k_norm_weight_name,
                    &[GLM52_DSA_INDEX_HEAD_DIM],
                    "BF16 MLA/RoPE device DSA k_norm weight pinned staging",
                )?;
            }
            let k_norm_bias_resident = math::bf16_full_vector_resident_available(
                &k_norm_bias_name,
                GLM52_DSA_INDEX_HEAD_DIM,
            );
            if !k_norm_bias_resident {
                projection_bytes_read += preload_bf16_tensor_resident_from_host_staging(
                    catalog,
                    &k_norm_bias_name,
                    &[GLM52_DSA_INDEX_HEAD_DIM],
                    "BF16 MLA/RoPE device DSA k_norm bias pinned staging",
                )?;
            }
            Some((wk_name, k_norm_weight_name, k_norm_bias_name))
        } else {
            None
        };
        let device_current_row_projection_outputs = if resident_device_weights.is_some()
            && input_norm.is_none()
            && q_a.is_none()
            && q_a_norm.is_none()
            && q_b.is_none()
            && kv_a.is_none()
        {
            let normalized_device = if let Some(device_rows) = current_hidden_device_rows {
                rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output(
                    &input_norm_name,
                    device_rows.buffer(),
                    context_hidden_rows.len(),
                    GLM52_HIDDEN_SIZE,
                    REAL_FULL_DENSE_RMSNORM_EPS,
                )
                .context("executing resident MLA/RoPE device query input RMSNorm from device hidden rows")?
            } else {
                let flattened_hidden = flatten_rows(&context_hidden_rows);
                rmsnorm_hidden_bf16_preloaded_resident_weight_device_output(
                    &input_norm_name,
                    &bf16_bytes_from_f32(&flattened_hidden),
                    context_hidden_rows.len(),
                    GLM52_HIDDEN_SIZE,
                    REAL_FULL_DENSE_RMSNORM_EPS,
                )
                .context("executing resident MLA/RoPE device query input RMSNorm")?
            };
            let q_a_device = linear_rows_bf16_preloaded_resident_weight_device_output(
                &q_a_name,
                normalized_device.buffer(),
                None,
                context_hidden_rows.len(),
                GLM52_HIDDEN_SIZE,
                q_lora_rank,
                q_lora_rank,
            )
            .context("executing resident MLA/RoPE device q_a projection")?;
            let q_a_normalized_device =
                rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output(
                    &q_a_norm_name,
                    q_a_device.buffer(),
                    context_hidden_rows.len(),
                    q_lora_rank,
                    REAL_FULL_DENSE_RMSNORM_EPS,
                )
                .context("executing resident MLA/RoPE device q_a RMSNorm")?;
            let q_b_resident_name = if q_b_full_resident {
                q_b_name.as_str()
            } else {
                q_b_weight_key.as_str()
            };
            let q_b_resident_full_rows = if q_b_full_resident {
                q_b_full_rows
            } else {
                q_b_rows
            };
            let q_projected = linear_rows_bf16_preloaded_resident_weight_device_output(
                q_b_resident_name,
                q_a_normalized_device.buffer(),
                None,
                context_hidden_rows.len(),
                q_lora_rank,
                q_b_rows,
                q_b_resident_full_rows,
            )
            .context("executing resident MLA/RoPE device q_b projection")?;
            let kv_a_projected = linear_rows_bf16_preloaded_resident_weight_device_output(
                &kv_a_name,
                normalized_device.buffer(),
                None,
                context_hidden_rows.len(),
                GLM52_HIDDEN_SIZE,
                kv_a_rows,
                kv_a_rows,
            )
            .context("executing resident MLA/RoPE device kv_a projection")?;
            let dsa_key = if let Some((wk_name, k_norm_weight_name, k_norm_bias_name)) =
                dsa_device_key_names.as_ref()
            {
                let wk_projected = linear_rows_bf16_preloaded_resident_weight_device_output(
                    wk_name,
                    normalized_device.buffer(),
                    None,
                    context_hidden_rows.len(),
                    GLM52_HIDDEN_SIZE,
                    GLM52_DSA_INDEX_HEAD_DIM,
                    GLM52_DSA_INDEX_HEAD_DIM,
                )
                .context("executing resident MLA/RoPE device DSA wk projection")?;
                Some(
                    layer_norm_affine_bf16_preloaded_resident_weight_bias_device_input_output(
                        k_norm_weight_name,
                        k_norm_bias_name,
                        wk_projected.buffer(),
                        context_hidden_rows.len(),
                        GLM52_DSA_INDEX_HEAD_DIM,
                        REAL_FULL_DENSE_RMSNORM_EPS,
                    )
                    .context("executing resident MLA/RoPE device DSA k_norm")?,
                )
            } else {
                None
            };
            Some(MlaRopeDeviceCurrentRowProjectionOutputs {
                q_projected,
                kv_a_projected,
                dsa_key,
            })
        } else {
            None
        };
        let mut device_kv = RealFullDeviceKvExecutionMirror::new(config)
            .context("creating real full MLA/RoPE device-prefix attention mirror")?;
        device_kv
            .write_host_blocks(&descriptors, &payloads)
            .context("writing real full MLA/RoPE prefix blocks to device cache")?;
        let mut current_device_kv_cache_write_available = false;
        let current_kv_norm_weight = resident_device_weights
            .as_ref()
            .map(|(kv_norm_weight, _)| *kv_norm_weight);
        if let Some(outputs) = device_current_row_projection_outputs.as_ref() {
            if current_kv_device_write_has_dsa {
                let dsa_key_output = outputs.dsa_key.as_ref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "real full MLA/RoPE DSA device current-row write missing DSA key output"
                    )
                })?;
                current_device_kv_cache_write_available = device_kv
                    .write_projected_mla_kv_a_and_dsa_key_device_blocks_bf16(
                        &current_descriptors,
                        outputs.kv_a_projected.buffer(),
                        dsa_key_output.buffer(),
                        current_kv_norm_weight,
                    )
                    .context(
                        "writing real full MLA/RoPE current projected kv_a plus DSA key to device cache",
                    )?
                    .is_some();
            } else {
                current_device_kv_cache_write_available = device_kv
                    .write_projected_mla_kv_a_device_blocks_bf16(
                        &current_descriptors,
                        outputs.kv_a_projected.buffer(),
                        current_kv_norm_weight,
                    )
                    .context("writing real full MLA/RoPE current projected kv_a to device cache")?
                    .is_some();
            }
        }
        let q_nope_bf16 = bf16_bytes_from_f32(&q_nope_values);
        let q_rope_rotated_bf16 = bf16_bytes_from_f32(&q_rope_rotated_values);
        if let Some((kv_norm_weight, kv_b_weight)) = resident_device_weights {
            let q_suffix_positions = hidden_positions
                .iter()
                .map(|position| {
                    u32::try_from(*position)
                        .context("MLA/RoPE device projected-query position overflow")
                })
                .collect::<Result<Vec<_>>>()?;
            device_prefix_attention_parts = if let Some(outputs) =
                device_current_row_projection_outputs.as_ref()
            {
                if current_device_kv_cache_write_available && !current_descriptors.is_empty() {
                    let mut attention_descriptors =
                        Vec::with_capacity(descriptors.len() + current_descriptors.len());
                    attention_descriptors.extend(descriptors.iter().cloned());
                    attention_descriptors.extend(current_descriptors.iter().cloned());
                    let mut attention_positions =
                        Vec::with_capacity(prefix_positions.len() + q_suffix_positions.len());
                    attention_positions.extend(prefix_positions.iter().copied());
                    attention_positions.extend(q_suffix_positions.iter().copied());
                    device_kv
                        .run_mla_rope_attention_parts_from_device_kv_with_device_weights_and_projected_query_device_bf16(
                            &attention_descriptors,
                            &attention_positions,
                            kv_norm_weight,
                            kv_b_weight,
                            outputs.q_projected.buffer(),
                            None,
                            None,
                            &q_suffix_positions,
                            attention_heads,
                            REAL_FULL_MLA_QK_NOPE_HEAD_DIM,
                            REAL_FULL_MLA_V_HEAD_DIM,
                            REAL_FULL_DENSE_RMSNORM_EPS,
                            REAL_FULL_MLA_ROPE_THETA as f32,
                            scale,
                        )
                        .context("executing real full MLA/RoPE device-cache attention")?
                } else {
                    device_kv
                        .run_mla_rope_attention_parts_from_device_prefix_with_device_weights_and_projected_query_device_kv_suffix_bf16(
                            &descriptors,
                            &prefix_positions,
                            kv_norm_weight,
                            kv_b_weight,
                            outputs.q_projected.buffer(),
                            outputs.kv_a_projected.buffer(),
                            &q_suffix_positions,
                            attention_heads,
                            REAL_FULL_MLA_QK_NOPE_HEAD_DIM,
                            REAL_FULL_MLA_V_HEAD_DIM,
                            REAL_FULL_DENSE_RMSNORM_EPS,
                            REAL_FULL_MLA_ROPE_THETA as f32,
                            scale,
                        )
                        .context("executing real full MLA/RoPE device-prefix attention")?
                }
            } else {
                let q_projected_bf16 = bf16_bytes_from_f32(&q_projected_current_values);
                device_kv
                    .run_mla_rope_attention_parts_from_device_prefix_with_device_weights_and_projected_query_host_suffix_bf16(
                    &descriptors,
                    &prefix_positions,
                    kv_norm_weight,
                    kv_b_weight,
                    &q_projected_bf16,
                    &q_suffix_positions,
                    &suffix_k_nope_bf16,
                    &suffix_k_rope_rotated_bf16,
                    &suffix_values_bf16,
                    attention_heads,
                    REAL_FULL_MLA_QK_NOPE_HEAD_DIM,
                    REAL_FULL_MLA_V_HEAD_DIM,
                    REAL_FULL_DENSE_RMSNORM_EPS,
                    REAL_FULL_MLA_ROPE_THETA as f32,
                    scale,
                )
                    .context("executing real full MLA/RoPE device-prefix attention")?
            };
        } else {
            let kv_norm_weight_bf16 = match kv_a_norm.as_ref() {
                Some(tensor) => tensor.bytes.clone(),
                None => load_tensor_bytes(catalog, &kv_a_norm_name)?.bytes,
            };
            let kv_b_weight_bf16 = match kv_b.as_ref() {
                Some(rows) => rows.bytes.clone(),
                None => load_tensor_rows(catalog, &kv_b_name, 0, kv_b_rows)?.bytes,
            };
            let readback = device_kv
                .run_mla_rope_attention_from_device_prefix_with_host_suffix_bf16(
                    &descriptors,
                    &prefix_positions,
                    &kv_norm_weight_bf16,
                    &kv_b_weight_bf16,
                    &q_nope_bf16,
                    &q_rope_rotated_bf16,
                    &suffix_k_nope_bf16,
                    &suffix_k_rope_rotated_bf16,
                    &suffix_values_bf16,
                    attention_heads,
                    REAL_FULL_MLA_QK_NOPE_HEAD_DIM,
                    REAL_FULL_MLA_V_HEAD_DIM,
                    REAL_FULL_DENSE_RMSNORM_EPS,
                    REAL_FULL_MLA_ROPE_THETA as f32,
                    scale,
                )
                .context("executing real full MLA/RoPE device-prefix attention")?;
            if let Some(readback) = readback {
                let context_values = bf16_bytes_to_f32(&readback.output_bf16)?;
                device_prefix_context_rows = Some(
                    context_values
                        .chunks_exact(context_width)
                        .map(|row| row.to_vec())
                        .collect::<Vec<_>>(),
                );
            }
        }
    }
    let attention_context = causal_mla_rope_attention_f32(
        layer_id,
        &q_nope_values,
        &q_rope_rotated_values,
        &k_nope_values,
        &k_rope_rotated_values,
        &value_values,
        MlaRopeAttentionF32Shape {
            rows: total_context_rows,
            heads: attention_heads,
            nope_dim: REAL_FULL_MLA_QK_NOPE_HEAD_DIM,
            rope_dim: GLM52_MLA_QK_ROPE_HEAD_DIM,
            value_dim: REAL_FULL_MLA_V_HEAD_DIM,
        },
        scale,
    )?;
    let attention_scores = attention_context.scores;
    let attention_weights = attention_context.weights;
    let mut attention_backend = attention_context.context_backend;
    let mut context_rows = attention_context.context_rows;
    if device_prefix_attention_parts.is_some() {
        attention_backend = CUDA_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND;
    }
    if let Some(device_context_rows) = device_prefix_context_rows {
        attention_backend = CUDA_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND;
        context_rows = device_context_rows;
    }
    let resident_device_o_proj = if device_prefix_attention_parts.is_some() {
        if o_proj_full_resident && context_width == o_proj_row_width {
            Some((o_proj_name.as_str(), o_proj_full_rows, None))
        } else if o_proj_row_window_resident {
            Some((o_proj_row_window_weight_key.as_str(), output_count, None))
        } else if o_proj_full_resident && context_width < o_proj_row_width {
            Some((
                o_proj_name.as_str(),
                o_proj_full_rows,
                Some(o_proj_row_width),
            ))
        } else {
            None
        }
    } else {
        None
    };
    if resident_device_o_proj.is_none() {
        if let Some(attention_parts) = device_prefix_attention_parts.as_ref() {
            let readback = attention_parts.copy_to_host()?;
            let context_values = bf16_bytes_to_f32(&readback.output_bf16)?;
            let mut chunks = context_values.chunks_exact(context_width);
            context_rows = chunks.by_ref().map(|row| row.to_vec()).collect();
            if !chunks.remainder().is_empty() {
                anyhow::bail!(
                    "real full MLA/RoPE device attention readback length {} is not divisible by context_width {context_width}",
                    context_values.len()
                );
            }
        }
    }

    let query_context_rows = context_rows
        .get(prefix_context_row_count..)
        .ok_or_else(|| anyhow::anyhow!("MLA/RoPE query context rows missing after prefix rows"))?;
    let o_proj_compact_prefix_bytes = if o_proj_full_resident || o_proj_row_window_resident {
        None
    } else {
        let o_proj = o_proj.as_ref().ok_or_else(|| {
            anyhow::anyhow!("MLA/RoPE compact o_proj prefix requires loaded row-window bytes")
        })?;
        Some(compact_row_prefix_bytes(
            &o_proj.bytes,
            output_count,
            o_proj.row_width,
            context_width,
        )?)
    };
    let mut attention_output_rows = Vec::with_capacity(query_context_rows.len());
    let mut device_residual_after_rows = None;
    let mut device_hidden_after_attention = None;
    let mut device_residual_add_backend = None;
    if let Some(attention_parts) = device_prefix_attention_parts.as_ref() {
        if let Some((resident_weight_name, resident_full_rows, padded_full_input_dim)) =
            resident_device_o_proj
        {
            let query_context_buffer =
                attention_parts.output_row_buffer(prefix_context_row_count, hidden_rows.len())?;
            let residual_before_values = hidden_rows
                .iter()
                .flat_map(|hidden| hidden[..output_count].iter().copied())
                .collect::<Vec<_>>();
            let residual_before_bf16 = bf16_bytes_from_f32(&residual_before_values);
            let o_projection = if let Some(full_input_dim) = padded_full_input_dim {
                linear_residual_add_rows_bf16_preloaded_resident_weight_padded_device_input_device_output(
                    resident_weight_name,
                    query_context_buffer,
                    None,
                    &residual_before_bf16,
                    hidden_rows.len(),
                    context_width,
                    full_input_dim,
                    output_count,
                    resident_full_rows,
                )?
            } else {
                linear_residual_add_rows_bf16_preloaded_resident_weight_device_input_device_output(
                    resident_weight_name,
                    query_context_buffer,
                    None,
                    &residual_before_bf16,
                    hidden_rows.len(),
                    context_width,
                    output_count,
                    resident_full_rows,
                )?
            };
            record_stage_backend(
                &mut projection_backend,
                o_projection.linear_backend,
                "MLA/RoPE attention projection",
                "o_proj",
            )?;
            record_stage_backend(
                &mut device_residual_add_backend,
                o_projection.residual_add_backend,
                "MLA/RoPE attention residual add",
                "residual_add",
            )?;
            if o_projection.residual_device.rows != hidden_rows.len()
                || o_projection.residual_device.values_per_row != output_count
            {
                anyhow::bail!(
                    "real full MLA/RoPE device residual output shape rows={} values_per_row={} does not match expected rows={} values_per_row={output_count}",
                    o_projection.residual_device.rows,
                    o_projection.residual_device.values_per_row,
                    hidden_rows.len()
                );
            }
            let linear_values = o_projection.linear_values;
            let residual_values = o_projection.residual_values;
            device_hidden_after_attention = Some(o_projection.residual_device);
            let mut chunks = linear_values.chunks_exact(output_count);
            for (row_index, row) in chunks.by_ref().enumerate() {
                for (output_idx, output) in row.iter().copied().enumerate() {
                    if !output.is_finite() {
                        anyhow::bail!(
                            "real full MLA/RoPE probe produced non-finite device o_proj output at row {row_index} index {output_idx}"
                        );
                    }
                }
                attention_output_rows.push(row.to_vec());
            }
            if !chunks.remainder().is_empty() {
                anyhow::bail!(
                    "real full MLA/RoPE device o_proj output length {} is not divisible by output_count {output_count}",
                    linear_values.len()
                );
            }
            let mut residual_chunks = residual_values.chunks_exact(output_count);
            let residual_rows = residual_chunks
                .by_ref()
                .map(|row| row.to_vec())
                .collect::<Vec<_>>();
            if !residual_chunks.remainder().is_empty() {
                anyhow::bail!(
                    "real full MLA/RoPE device o_proj residual output length {} is not divisible by output_count {output_count}",
                    residual_values.len()
                );
            }
            device_residual_after_rows = Some(residual_rows);
        }
    }
    if attention_output_rows.is_empty() {
        for context in query_context_rows {
            let o_projection = project_rows_bf16_with_optional_padded_preloaded_prefix_weight(
                &o_proj_name,
                &o_proj_row_window_weight_key,
                context,
                o_proj_compact_prefix_bytes
                    .as_ref()
                    .map(|bytes| bytes.as_slice()),
                output_count,
                context_width,
                o_proj_row_width,
                o_proj_full_rows,
            )?;
            record_stage_backend(
                &mut projection_backend,
                o_projection.backend,
                "MLA/RoPE attention projection",
                "o_proj",
            )?;
            let attention_outputs = o_projection.values;
            for (output_idx, output) in attention_outputs.iter().copied().enumerate() {
                if !output.is_finite() {
                    anyhow::bail!(
                        "real full MLA/RoPE probe produced non-finite output at index {output_idx}"
                    );
                }
            }
            attention_output_rows.push(attention_outputs);
        }
    }
    let projection_backend = required_stage_backend(
        projection_backend,
        "MLA/RoPE attention projection",
        "linear",
    )?;

    let (residual_after_rows, residual_add_backend) = if let Some(residual_after_rows) =
        device_residual_after_rows
    {
        let residual_add_backend = required_stage_backend(
            device_residual_add_backend,
            "MLA/RoPE attention residual add",
            "residual-add",
        )?;
        (residual_after_rows, residual_add_backend)
    } else {
        let mut residual_workspace = AttentionResidualAddWorkspace::default();
        let mut residual_after_rows = Vec::with_capacity(attention_output_rows.len());
        let mut residual_add_backend = None;
        for (hidden, attention_outputs) in hidden_rows.iter().zip(attention_output_rows.iter()) {
            let residual_before = hidden[..output_count].to_vec();
            let residual_after = attention_residual_add_bf16(
                &residual_before,
                attention_outputs,
                &mut residual_workspace,
            )?;
            record_stage_backend(
                &mut residual_add_backend,
                residual_after.backend,
                "MLA/RoPE attention residual add",
                "residual_add",
            )?;
            residual_after_rows.push(residual_after.values);
        }
        let residual_add_backend = required_stage_backend(
            residual_add_backend,
            "MLA/RoPE attention residual add",
            "residual-add",
        )?;
        (residual_after_rows, residual_add_backend)
    };

    let mut hidden_after_attention_prefix_rows = hidden_rows.clone();
    for (hidden, residual_after) in hidden_after_attention_prefix_rows
        .iter_mut()
        .zip(residual_after_rows.iter())
    {
        hidden[..output_count].copy_from_slice(residual_after);
    }

    let context_values = flatten_rows(query_context_rows);
    let attention_output_values = flatten_rows(&attention_output_rows);
    let residual_before_values = hidden_rows
        .iter()
        .flat_map(|hidden| hidden[..output_count].iter().copied())
        .collect::<Vec<_>>();
    let residual_after_values = flatten_rows(&residual_after_rows);
    let attention_output_checksum = checksum_f64(&attention_output_values);
    let attention_output_l2_norm = attention_output_values
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    let kv_cache_context_bytes = prefix_kv_blocks
        .iter()
        .map(|block| block.bytes.len())
        .sum::<usize>();
    let causal_attention_scores = attention_scores.len();
    let mla_softmax_rows = attention_heads * total_context_rows;
    let residual_after_checksum = checksum_f64(&residual_after_values);
    let attention_weights_checksum = checksum_f64(&attention_weights);
    let passed = causal_attention_scores == attention_scores.len()
        && (attention_weights_checksum - mla_softmax_rows as f64).abs() < 1.0e-5
        && attention_output_l2_norm.is_finite()
        && attention_output_checksum.is_finite()
        && residual_after_checksum.is_finite();

    let first_row_residual_before_checksum = checksum_f64(&hidden_rows[0][..output_count]);
    let first_row_residual_delta_checksum = checksum_f64(&attention_output_rows[0]);
    let first_row_residual_after_checksum = checksum_f64(&residual_after_rows[0]);

    let status = if output_count == GLM52_HIDDEN_SIZE
        && attention_heads == REAL_FULL_MLA_NUM_ATTENTION_HEADS
    {
        "numeric-real-full-output-main-mla-rope-attention"
    } else {
        "numeric-real-bounded-main-mla-rope-attention"
    };

    Ok(MlaRopeAttentionPrefixExecution {
        probe: RealFullMlaRopeAttentionProbe {
            status,
            scope,
            layer_id,
            hidden_source,
            context_source,
            attention_rows: hidden_rows.len(),
            prefix_context_rows: prefix_context_row_count,
            total_context_rows,
            attention_heads,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim: REAL_FULL_MLA_QK_NOPE_HEAD_DIM,
            qk_rope_head_dim: GLM52_MLA_QK_ROPE_HEAD_DIM,
            q_head_dim,
            v_head_dim: REAL_FULL_MLA_V_HEAD_DIM,
            compressed_kv_width: GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM,
            compressed_kv_values: total_context_rows
                * (GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM),
            causal_attention_scores,
            mla_softmax_rows,
            attention_context_values: context_values.len(),
            output_prefix_values: output_count,
            residual_prefix_values: residual_after_values.len(),
            residual_adds: hidden_rows.len(),
            rope_theta: REAL_FULL_MLA_ROPE_THETA,
            input_norm_bytes_read,
            projection_bytes_read,
            o_proj_bytes_read,
            rope_backend,
            attention_backend,
            q_nope_checksum: Some(checksum_f64(&q_nope_values)),
            q_rope_checksum: Some(checksum_f64(&q_rope_values)),
            q_rope_rotated_checksum: Some(checksum_f64(&q_rope_rotated_values)),
            k_nope_checksum: Some(checksum_f64(&k_nope_values)),
            k_rope_checksum: Some(checksum_f64(&k_rope_values)),
            k_rope_rotated_checksum: Some(checksum_f64(&k_rope_rotated_values)),
            value_checksum: Some(checksum_f64(&value_values)),
            attention_scores_checksum: Some(checksum_f64(&attention_scores)),
            attention_weights_checksum: Some(attention_weights_checksum),
            attention_context_checksum: Some(checksum_f64(&context_values)),
            attention_output_checksum: Some(attention_output_checksum),
            attention_output_l2_norm: Some(attention_output_l2_norm),
            residual_before_checksum: Some(checksum_f64(&residual_before_values)),
            residual_delta_checksum: Some(attention_output_checksum),
            residual_after_checksum: Some(residual_after_checksum),
            uses_real_attention_weights: true,
            includes_rope: true,
            includes_mla_softmax: true,
            applies_attention_residual_prefix: true,
            uses_full_model_residual: false,
            passed,
            skipped_reason: None,
        },
        hidden_after_attention_prefix_rows,
        device_hidden_after_attention,
        projection_backend,
        residual_add_backend,
        uses_kv_cache_context: kv_cache_context_bytes > 0,
        kv_cache_context_bytes,
        first_row_residual_before_checksum,
        first_row_residual_delta_checksum,
        first_row_residual_after_checksum,
    })
}

fn flatten_head_rows(rows: &[Vec<Vec<f32>>]) -> Vec<f32> {
    rows.iter()
        .flat_map(|row| row.iter().flat_map(|head| head.iter().copied()))
        .collect()
}

fn execute_real_full_attention_residual_prefix(
    catalog: &TensorCatalog,
) -> Result<AttentionResidualPrefixExecution> {
    let mode = attention_residual_probe_mode(None);
    execute_real_full_attention_residual_prefix_with_rows(
        catalog,
        REAL_FULL_ATTENTION_RESIDUAL_LAYER_ID,
        mode.output_count,
        deterministic_attention_hidden_rows(),
        mode.hidden_source,
        mode.scope,
    )
}

fn execute_real_full_attention_residual_prefix_with_rows(
    catalog: &TensorCatalog,
    layer_id: usize,
    output_count: usize,
    hidden_rows: Vec<Vec<f32>>,
    hidden_source: &'static str,
    scope: &'static str,
) -> Result<AttentionResidualPrefixExecution> {
    if hidden_rows.is_empty() {
        anyhow::bail!("real full attention residual-prefix probe requires at least one hidden row");
    }
    if output_count == 0 || output_count > GLM52_HIDDEN_SIZE {
        anyhow::bail!(
            "real full attention residual-prefix output_count={} is invalid for hidden {}",
            output_count,
            GLM52_HIDDEN_SIZE
        );
    }
    for (row_index, hidden) in hidden_rows.iter().enumerate() {
        if hidden.len() != GLM52_HIDDEN_SIZE {
            anyhow::bail!(
                "real full attention residual-prefix hidden row {row_index} width mismatch: expected {} got {}",
                GLM52_HIDDEN_SIZE,
                hidden.len()
            );
        }
    }

    let input_norm_name = format!("model.layers.{layer_id}.input_layernorm.weight");
    let q_a_name = format!("model.layers.{layer_id}.self_attn.q_a_proj.weight");
    let q_a_norm_name = format!("model.layers.{layer_id}.self_attn.q_a_layernorm.weight");
    let q_b_name = format!("model.layers.{layer_id}.self_attn.q_b_proj.weight");
    let kv_a_name = format!("model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.weight");
    let kv_a_norm_name = format!("model.layers.{layer_id}.self_attn.kv_a_layernorm.weight");
    let kv_b_name = format!("model.layers.{layer_id}.self_attn.kv_b_proj.weight");
    let o_proj_name = format!("model.layers.{layer_id}.self_attn.o_proj.weight");

    let input_norm_info = catalog_tensor(catalog, &input_norm_name)?;
    let q_a_info = catalog_tensor(catalog, &q_a_name)?;
    let kv_a_info = catalog_tensor(catalog, &kv_a_name)?;
    let q_a_norm_info = catalog_tensor(catalog, &q_a_norm_name)?;
    let kv_a_norm_info = catalog_tensor(catalog, &kv_a_norm_name)?;
    let q_b_info = catalog_tensor(catalog, &q_b_name)?;
    let kv_b_info = catalog_tensor(catalog, &kv_b_name)?;
    let o_proj_info = catalog_tensor(catalog, &o_proj_name)?;
    let q_b_weight_key = format!("{q_b_name}[rows=0..{output_count}]");
    let kv_b_weight_key = format!("{kv_b_name}[rows=0..{output_count}]");
    let o_proj_weight_key = format!("{o_proj_name}[rows=0..{output_count}]");

    if input_norm_info.dtype != DType::Bf16
        || q_a_info.dtype != DType::Bf16
        || q_a_norm_info.dtype != DType::Bf16
        || q_b_info.dtype != DType::Bf16
        || kv_a_info.dtype != DType::Bf16
        || kv_a_norm_info.dtype != DType::Bf16
        || kv_b_info.dtype != DType::Bf16
        || o_proj_info.dtype != DType::Bf16
    {
        anyhow::bail!(
            "real full attention residual-prefix probe expects BF16 input norm and attention weights for layer {layer_id}, got {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, and {:?}",
            input_norm_info.dtype,
            q_a_info.dtype,
            q_a_norm_info.dtype,
            q_b_info.dtype,
            kv_a_info.dtype,
            kv_a_norm_info.dtype,
            kv_b_info.dtype,
            o_proj_info.dtype
        );
    }
    if input_norm_info.shape != vec![GLM52_HIDDEN_SIZE] {
        anyhow::bail!(
            "real full attention residual-prefix input norm shape mismatch for layer {layer_id}: {:?} for hidden {}",
            input_norm_info.shape,
            GLM52_HIDDEN_SIZE
        );
    }
    if q_a_info.shape.len() != 2 || kv_a_info.shape.len() != 2 {
        anyhow::bail!(
            "real full attention residual-prefix probe expected rank-2 q_a/kv_a weights, got {:?} and {:?}",
            q_a_info.shape,
            kv_a_info.shape
        );
    }

    let q_lora_rank = q_a_info.shape[0];
    let kv_a_rows = kv_a_info.shape[0];
    let kv_lora_rank = kv_a_norm_info.shape.first().copied().unwrap_or_default();
    if q_a_info.shape[1] != GLM52_HIDDEN_SIZE || kv_a_info.shape[1] != GLM52_HIDDEN_SIZE {
        anyhow::bail!(
            "real full attention residual-prefix hidden width mismatch: hidden={} q_a_width={} kv_a_width={}",
            GLM52_HIDDEN_SIZE,
            q_a_info.shape[1],
            kv_a_info.shape[1]
        );
    }
    if q_a_norm_info.shape != vec![q_lora_rank] {
        anyhow::bail!(
            "real full attention residual-prefix q_a norm shape mismatch: {:?} for q rank {q_lora_rank}",
            q_a_norm_info.shape
        );
    }
    if kv_lora_rank == 0 || kv_lora_rank > kv_a_rows || kv_a_norm_info.shape != vec![kv_lora_rank] {
        anyhow::bail!(
            "real full attention residual-prefix kv norm shape mismatch: {:?} for kv_a rows {kv_a_rows}",
            kv_a_norm_info.shape
        );
    }
    let (q_b_full_rows, q_b_row_width) = validate_bf16_matrix_tensor(
        q_b_info,
        "real full attention residual-prefix q_b",
        layer_id,
    )?;
    let (kv_b_full_rows, kv_b_row_width) = validate_bf16_matrix_tensor(
        kv_b_info,
        "real full attention residual-prefix kv_b",
        layer_id,
    )?;
    let (o_proj_full_rows, o_proj_row_width) = validate_bf16_matrix_tensor(
        o_proj_info,
        "real full attention residual-prefix o_proj",
        layer_id,
    )?;
    if q_b_row_width != q_lora_rank || kv_b_row_width != kv_lora_rank {
        anyhow::bail!(
            "real full attention residual-prefix q_b/kv_b width mismatch: q_b={} q_rank={} kv_b={} kv_rank={}",
            q_b_row_width,
            q_lora_rank,
            kv_b_row_width,
            kv_lora_rank
        );
    }
    if output_count > q_b_full_rows || output_count > kv_b_full_rows {
        anyhow::bail!(
            "real full attention residual-prefix q_b/kv_b row prefix exceeds full rows: output_count={output_count} q_b_rows={q_b_full_rows} kv_b_rows={kv_b_full_rows}"
        );
    }
    if o_proj_row_width < output_count {
        anyhow::bail!(
            "real full attention residual-prefix o_proj width {} is smaller than context prefix {}",
            o_proj_row_width,
            output_count
        );
    }
    if output_count > o_proj_full_rows {
        anyhow::bail!(
            "real full attention residual-prefix o_proj row prefix exceeds full rows: output_count={output_count} full_rows={o_proj_full_rows}"
        );
    }
    let o_proj_row_window_weight_key = if output_count == o_proj_row_width {
        o_proj_weight_key.clone()
    } else {
        format!("{o_proj_name}[rows=0..{output_count},cols=0..{output_count}]")
    };
    let cuda_reference_enabled = coordinator_cuda_reference_kernels_enabled();
    let mut input_norm_bytes_read = 0_u64;
    let mut projection_bytes_read = 0_u64;
    let mut o_proj_bytes_read = 0_u64;
    let input_norm_full_resident =
        math::bf16_full_vector_resident_available(&input_norm_name, GLM52_HIDDEN_SIZE);
    let input_norm = if input_norm_full_resident {
        None
    } else if cuda_reference_enabled {
        input_norm_bytes_read = preload_bf16_tensor_resident_from_host_staging(
            catalog,
            &input_norm_name,
            &[GLM52_HIDDEN_SIZE],
            "BF16 bounded causal attention input norm pinned staging",
        )?;
        None
    } else {
        let tensor = load_tensor_bytes(catalog, &input_norm_name)?;
        input_norm_bytes_read = tensor.bytes.len() as u64;
        Some(tensor)
    };
    let q_a_norm_full_resident =
        math::bf16_full_vector_resident_available(&q_a_norm_name, q_lora_rank);
    let q_a_norm = if q_a_norm_full_resident {
        None
    } else if cuda_reference_enabled {
        projection_bytes_read += preload_bf16_tensor_resident_from_host_staging(
            catalog,
            &q_a_norm_name,
            &[q_lora_rank],
            "BF16 bounded causal attention q_a norm pinned staging",
        )?;
        None
    } else {
        let tensor = load_tensor_bytes(catalog, &q_a_norm_name)?;
        projection_bytes_read += tensor.bytes.len() as u64;
        Some(tensor)
    };
    let kv_a_norm_full_resident =
        math::bf16_full_vector_resident_available(&kv_a_norm_name, kv_lora_rank);
    let kv_a_norm = if kv_a_norm_full_resident {
        None
    } else if cuda_reference_enabled {
        projection_bytes_read += preload_bf16_tensor_resident_from_host_staging(
            catalog,
            &kv_a_norm_name,
            &[kv_lora_rank],
            "BF16 bounded causal attention kv_a norm pinned staging",
        )?;
        None
    } else {
        let tensor = load_tensor_bytes(catalog, &kv_a_norm_name)?;
        projection_bytes_read += tensor.bytes.len() as u64;
        Some(tensor)
    };
    let q_a_full_resident = bf16_full_row_prefix_resident_available(
        &q_a_name,
        q_lora_rank,
        GLM52_HIDDEN_SIZE,
        q_lora_rank,
    );
    let kv_a_full_resident = bf16_full_row_prefix_resident_available(
        &kv_a_name,
        kv_a_rows,
        GLM52_HIDDEN_SIZE,
        kv_a_rows,
    );
    let q_b_full_resident = bf16_full_row_prefix_resident_available(
        &q_b_name,
        q_b_full_rows,
        q_lora_rank,
        output_count,
    );
    let kv_b_full_resident = bf16_full_row_prefix_resident_available(
        &kv_b_name,
        kv_b_full_rows,
        kv_lora_rank,
        output_count,
    );
    let o_proj_full_resident = bf16_full_row_prefix_resident_available(
        &o_proj_name,
        o_proj_full_rows,
        o_proj_row_width,
        output_count,
    );
    let q_b = if q_b_full_resident {
        None
    } else if cuda_reference_enabled {
        projection_bytes_read += preload_bf16_rows_resident_from_host_staging(
            catalog,
            &q_b_name,
            &q_b_weight_key,
            output_count,
            q_lora_rank,
            "BF16 bounded causal attention q_b row-window pinned staging",
        )?;
        None
    } else {
        let rows = load_tensor_rows(catalog, &q_b_name, 0, output_count)?;
        projection_bytes_read += rows.bytes.len() as u64;
        Some(rows)
    };
    let kv_b = if kv_b_full_resident {
        None
    } else if cuda_reference_enabled {
        projection_bytes_read += preload_bf16_rows_resident_from_host_staging(
            catalog,
            &kv_b_name,
            &kv_b_weight_key,
            output_count,
            kv_lora_rank,
            "BF16 bounded causal attention kv_b row-window pinned staging",
        )?;
        None
    } else {
        let rows = load_tensor_rows(catalog, &kv_b_name, 0, output_count)?;
        projection_bytes_read += rows.bytes.len() as u64;
        Some(rows)
    };
    let o_proj = if o_proj_full_resident {
        None
    } else if cuda_reference_enabled {
        o_proj_bytes_read = preload_bf16_row_prefix_resident_from_host_staging(
            catalog,
            &o_proj_name,
            &o_proj_row_window_weight_key,
            output_count,
            output_count,
            "BF16 bounded causal attention o_proj row-prefix pinned staging",
        )?;
        projection_bytes_read += o_proj_bytes_read;
        None
    } else {
        let rows = load_tensor_rows(catalog, &o_proj_name, 0, output_count)?;
        o_proj_bytes_read = rows.bytes.len() as u64;
        projection_bytes_read += o_proj_bytes_read;
        Some(rows)
    };
    let o_proj_row_window_resident = !o_proj_full_resident && cuda_reference_enabled;
    let q_a = if q_a_full_resident {
        None
    } else if cuda_reference_enabled {
        projection_bytes_read += preload_bf16_tensor_resident_from_host_staging(
            catalog,
            &q_a_name,
            &[q_lora_rank, GLM52_HIDDEN_SIZE],
            "BF16 bounded causal attention q_a pinned staging",
        )?;
        None
    } else {
        let tensor = load_tensor_bytes(catalog, &q_a_name)?;
        projection_bytes_read += tensor.bytes.len() as u64;
        Some(tensor)
    };
    let kv_a = if kv_a_full_resident {
        None
    } else if cuda_reference_enabled {
        projection_bytes_read += preload_bf16_tensor_resident_from_host_staging(
            catalog,
            &kv_a_name,
            &[kv_a_rows, GLM52_HIDDEN_SIZE],
            "BF16 bounded causal attention kv_a pinned staging",
        )?;
        None
    } else {
        let tensor = load_tensor_bytes(catalog, &kv_a_name)?;
        projection_bytes_read += tensor.bytes.len() as u64;
        Some(tensor)
    };

    let mut q_output_rows = Vec::with_capacity(hidden_rows.len());
    let mut kv_output_rows = Vec::with_capacity(hidden_rows.len());
    let mut kv_rope_checksum = 0.0_f64;
    let mut projection_backend = None;
    for hidden in &hidden_rows {
        let normalized = rmsnorm_bf16_with_optional_preloaded_resident_weight(
            &input_norm_name,
            &bf16_bytes_from_f32(hidden),
            input_norm.as_ref().map(|tensor| tensor.bytes.as_slice()),
            1,
            hidden.len(),
            REAL_FULL_DENSE_RMSNORM_EPS,
        )?
        .values;
        let q_a_projected = project_rows_bf16_with_optional_preloaded_full_weight(
            &q_a_name,
            &normalized,
            q_a.as_ref().map(|tensor| tensor.bytes.as_slice()),
            q_lora_rank,
            hidden.len(),
        )?;
        record_stage_backend(
            &mut projection_backend,
            q_a_projected.backend,
            "attention projection",
            "q_a",
        )?;
        let q_a_normalized = rmsnorm_bf16_with_optional_preloaded_resident_weight(
            &q_a_norm_name,
            &bf16_bytes_from_f32(&q_a_projected.values),
            q_a_norm.as_ref().map(|tensor| tensor.bytes.as_slice()),
            1,
            q_a_projected.values.len(),
            REAL_FULL_DENSE_RMSNORM_EPS,
        )?
        .values;
        let q_output = project_rows_bf16_with_optional_preloaded_prefix_weight(
            &q_b_name,
            &q_b_weight_key,
            &q_a_normalized,
            q_b.as_ref().map(|rows| rows.bytes.as_slice()),
            output_count,
            q_lora_rank,
            q_b_full_rows,
        )?;
        record_stage_backend(
            &mut projection_backend,
            q_output.backend,
            "attention projection",
            "q_b",
        )?;
        q_output_rows.push(q_output.values);

        let kv_a_projected = project_rows_bf16_with_optional_preloaded_full_weight(
            &kv_a_name,
            &normalized,
            kv_a.as_ref().map(|tensor| tensor.bytes.as_slice()),
            kv_a_rows,
            hidden.len(),
        )?;
        record_stage_backend(
            &mut projection_backend,
            kv_a_projected.backend,
            "attention projection",
            "kv_a",
        )?;
        let kv_latent = &kv_a_projected.values[..kv_lora_rank];
        let kv_a_normalized = rmsnorm_bf16_with_optional_preloaded_resident_weight(
            &kv_a_norm_name,
            &bf16_bytes_from_f32(kv_latent),
            kv_a_norm.as_ref().map(|tensor| tensor.bytes.as_slice()),
            1,
            kv_latent.len(),
            REAL_FULL_DENSE_RMSNORM_EPS,
        )?
        .values;
        let kv_output = project_rows_bf16_with_optional_preloaded_prefix_weight(
            &kv_b_name,
            &kv_b_weight_key,
            &kv_a_normalized,
            kv_b.as_ref().map(|rows| rows.bytes.as_slice()),
            output_count,
            kv_lora_rank,
            kv_b_full_rows,
        )?;
        record_stage_backend(
            &mut projection_backend,
            kv_output.backend,
            "attention projection",
            "kv_b",
        )?;
        kv_output_rows.push(kv_output.values);
        kv_rope_checksum += kv_a_projected.values[kv_lora_rank..]
            .iter()
            .map(|value| *value as f64)
            .sum::<f64>();
    }

    let attention_scale = (output_count as f32).sqrt().recip();
    let q_output_values = flatten_rows(&q_output_rows);
    let kv_output_values = flatten_rows(&kv_output_rows);
    let causal_context_output = causal_attention_rows_bf16_for_layer(
        layer_id,
        &bf16_bytes_from_f32(&q_output_values),
        &bf16_bytes_from_f32(&kv_output_values),
        &bf16_bytes_from_f32(&kv_output_values),
        hidden_rows.len(),
        1,
        output_count,
        output_count,
        attention_scale,
    )?;
    let attention_backend = causal_context_output.backend;
    let causal_context = causal_context_output
        .values
        .chunks_exact(output_count)
        .map(|row| row.to_vec())
        .collect::<Vec<_>>();
    let causal_attention_scores = hidden_rows.len() * (hidden_rows.len() + 1) / 2;
    let o_proj_prefix_bytes = if o_proj_full_resident || o_proj_row_window_resident {
        None
    } else {
        let o_proj = o_proj.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "attention residual compact o_proj prefix requires loaded row-window bytes"
            )
        })?;
        Some(compact_row_prefix_bytes(
            &o_proj.bytes,
            output_count,
            o_proj.row_width,
            output_count,
        )?)
    };
    let mut attention_output_rows = Vec::with_capacity(causal_context.len());
    for context in &causal_context {
        let o_projection = project_rows_bf16_with_optional_padded_preloaded_prefix_weight(
            &o_proj_name,
            &o_proj_row_window_weight_key,
            context,
            o_proj_prefix_bytes.as_ref().map(|bytes| bytes.as_slice()),
            output_count,
            output_count,
            o_proj_row_width,
            o_proj_full_rows,
        )?;
        record_stage_backend(
            &mut projection_backend,
            o_projection.backend,
            "attention projection",
            "o_proj",
        )?;
        let attention_outputs = o_projection.values;
        for (output_idx, output) in attention_outputs.iter().copied().enumerate() {
            if !output.is_finite() {
                anyhow::bail!(
                    "real full attention residual-prefix probe produced non-finite output at index {output_idx}"
                );
            }
        }
        attention_output_rows.push(attention_outputs);
    }
    let projection_backend =
        required_stage_backend(projection_backend, "attention projection", "linear")?;

    let mut residual_workspace = AttentionResidualAddWorkspace::default();
    let mut residual_after_rows = Vec::with_capacity(attention_output_rows.len());
    let mut residual_add_backend = None;
    for (hidden, attention_outputs) in hidden_rows.iter().zip(attention_output_rows.iter()) {
        let residual_before = hidden[..output_count].to_vec();
        let residual_after = attention_residual_add_bf16(
            &residual_before,
            attention_outputs,
            &mut residual_workspace,
        )?;
        record_stage_backend(
            &mut residual_add_backend,
            residual_after.backend,
            "attention residual add",
            "residual_add",
        )?;
        residual_after_rows.push(residual_after.values);
    }
    let residual_add_backend = required_stage_backend(
        residual_add_backend,
        "attention residual add",
        "residual-add",
    )?;
    let mut hidden_after_attention_prefix_rows = hidden_rows.clone();
    for (hidden, residual_after) in hidden_after_attention_prefix_rows
        .iter_mut()
        .zip(residual_after_rows.iter())
    {
        hidden[..output_count].copy_from_slice(residual_after);
    }
    let attention_output_values = flatten_rows(&attention_output_rows);
    let residual_before_values = hidden_rows
        .iter()
        .flat_map(|hidden| hidden[..output_count].iter().copied())
        .collect::<Vec<_>>();
    let residual_after_values = flatten_rows(&residual_after_rows);

    let attention_output_checksum = checksum_f64(&attention_output_values);
    let attention_output_l2_norm = attention_output_values
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    let first_row_residual_before_checksum = checksum_f64(&hidden_rows[0][..output_count]);
    let first_row_residual_delta_checksum = checksum_f64(&attention_output_rows[0]);
    let first_row_residual_after_checksum = checksum_f64(&residual_after_rows[0]);
    let residual_before_checksum = checksum_f64(&residual_before_values);
    let residual_after_checksum = checksum_f64(&residual_after_values);
    let expected_causal_attention_scores = hidden_rows.len() * (hidden_rows.len() + 1) / 2;
    let passed = q_lora_rank > 0
        && kv_lora_rank > 0
        && causal_attention_scores == expected_causal_attention_scores
        && attention_output_l2_norm.is_finite()
        && attention_output_checksum.is_finite()
        && residual_after_checksum.is_finite();

    let status = if output_count == GLM52_HIDDEN_SIZE {
        "numeric-real-bf16-causal-attention-full-output-rows"
    } else {
        "numeric-real-bf16-causal-attention-residual-prefix"
    };

    Ok(AttentionResidualPrefixExecution {
        probe: RealFullAttentionResidualPrefixProbe {
            status,
            scope,
            layer_id,
            hidden_source,
            context_source: "causal-softmax-over-real-q-kv-prefixes",
            q_lora_rank,
            kv_lora_rank,
            attention_rows: hidden_rows.len(),
            output_prefix_values: output_count,
            residual_prefix_values: residual_after_values.len(),
            residual_adds: hidden_rows.len(),
            causal_attention_scores,
            causal_softmax_rows: hidden_rows.len(),
            input_norm_tensors_read: 1,
            attention_tensors_read: 7,
            input_norm_bytes_read,
            projection_bytes_read,
            o_proj_bytes_read,
            projection_backend,
            attention_backend,
            residual_add_backend,
            q_output_checksum: Some(checksum_f64(&q_output_values)),
            kv_output_checksum: Some(checksum_f64(&kv_output_values)),
            kv_rope_checksum: Some(kv_rope_checksum),
            attention_output_checksum: Some(attention_output_checksum),
            attention_output_l2_norm: Some(attention_output_l2_norm),
            residual_before_checksum: Some(residual_before_checksum),
            residual_delta_checksum: Some(attention_output_checksum),
            residual_after_checksum: Some(residual_after_checksum),
            first_residual_after: residual_after_values.first().copied(),
            last_residual_after: residual_after_values.last().copied(),
            uses_real_attention_weights: true,
            applies_attention_residual_prefix: true,
            includes_causal_softmax: true,
            includes_mla_softmax: false,
            uses_full_model_residual: false,
            passed,
            skipped_reason: None,
        },
        hidden_after_attention_prefix_rows,
        first_row_residual_before_checksum,
        first_row_residual_delta_checksum,
        first_row_residual_after_checksum,
    })
}
