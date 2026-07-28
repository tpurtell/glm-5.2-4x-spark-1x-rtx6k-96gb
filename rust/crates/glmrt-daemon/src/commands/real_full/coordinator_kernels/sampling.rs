use super::*;
use crate::python_graph_capture::{
    coordinator_python_capture_enabled, launch_python_kernel, PythonDeviceBufferArg,
    PythonGraphCaptureLaunch, PythonKernelArg,
};
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
pub(in crate::commands::real_full) const CPU_REFERENCE_LOGITS_ARGMAX_BACKEND: &str =
    "cpu-reference-logits-argmax";
#[allow(dead_code)]
pub(in crate::commands::real_full) const CUDA_REFERENCE_LOGITS_ARGMAX_BACKEND: &str =
    "cuda-reference-logits-argmax-f32";
pub(in crate::commands::real_full) const CPU_REFERENCE_LOGITS_SAMPLE_TOPK_TOPP_BACKEND: &str =
    "cpu-reference-logits-sample-topk-topp";
#[allow(dead_code)]
pub(in crate::commands::real_full) const CUDA_REFERENCE_LOGITS_SAMPLE_TOPK_TOPP_BACKEND: &str =
    "cuda-reference-logits-sample-topk-topp-f32";
pub(in crate::commands::real_full) const CPU_REFERENCE_LM_HEAD_ARGMAX_BF16_BACKEND: &str =
    "cpu-reference-lm-head-argmax-bf16";
#[allow(dead_code)]
pub(in crate::commands::real_full) const CUDA_REFERENCE_LM_HEAD_ARGMAX_BF16_BACKEND: &str =
    "cuda-reference-lm-head-argmax-bf16";
pub(in crate::commands::real_full) const CUDA_REFERENCE_LM_HEAD_ARGMAX_BF16_RESIDENT_WEIGHT_BACKEND:
    &str = "cuda-reference-lm-head-argmax-bf16-resident-weight";
pub(in crate::commands::real_full) const CUDA_REFERENCE_LM_HEAD_ARGMAX_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND:
    &str = "cuda-reference-lm-head-argmax-bf16-preloaded-resident-weight";
pub(in crate::commands::real_full) const CPU_REFERENCE_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_BACKEND: &str =
    "cpu-reference-lm-head-sample-topk-topp-bf16";
#[allow(dead_code)]
pub(in crate::commands::real_full) const CUDA_REFERENCE_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_BACKEND:
    &str = "cuda-reference-lm-head-sample-topk-topp-bf16";
pub(in crate::commands::real_full) const CUDA_REFERENCE_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_RESIDENT_WEIGHT_BACKEND:
    &str = "cuda-reference-lm-head-sample-topk-topp-bf16-resident-weight";
pub(in crate::commands::real_full) const CUDA_REFERENCE_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND:
    &str = "cuda-reference-lm-head-sample-topk-topp-bf16-preloaded-resident-weight";
pub(in crate::commands::real_full) const TRITON_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND:
    &str = "triton-lm-head-sample-topk-topp-bf16-preloaded-resident-weight";
const CUDA_LOGITS_SAMPLE_TOPK_TOPP_CUB_TEMP_STORAGE_BYTES: usize = 128 * 1024 * 1024;

#[derive(Debug)]
pub(in crate::commands::real_full) struct LogitsArgmaxSampleTopKToppOutput {
    pub(in crate::commands::real_full) argmax: LogitsArgmaxOutput,
    pub(in crate::commands::real_full) sampler: LogitsSampleTopKToppOutput,
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn logits_argmax(
    logits: &[f32],
    rows: usize,
    vocab: usize,
) -> Result<LogitsArgmaxOutput> {
    validate_logits_argmax_inputs(logits, rows, vocab)?;
    if cuda_reference_kernels_enabled() {
        return cuda_logits_argmax(logits, rows, vocab);
    }
    Ok(cpu_logits_argmax(logits, rows, vocab))
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn logits_sample_topk_topp(
    logits: &[f32],
    random_uniforms: &[f32],
    rows: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Result<LogitsSampleTopKToppOutput> {
    validate_logits_sample_topk_topp_inputs(
        logits,
        random_uniforms,
        rows,
        vocab,
        temperature,
        top_k,
        top_p,
    )?;
    if cuda_reference_kernels_enabled() {
        return cuda_logits_sample_topk_topp(
            logits,
            random_uniforms,
            rows,
            vocab,
            temperature,
            top_k,
            top_p,
        );
    }
    Ok(cpu_logits_sample_topk_topp(
        logits,
        random_uniforms,
        rows,
        vocab,
        temperature,
        top_k,
        top_p,
    ))
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn lm_head_argmax_bf16(
    hidden_bf16: &[u8],
    lm_head_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
) -> Result<LogitsArgmaxOutput> {
    validate_lm_head_bf16_inputs(hidden_bf16, lm_head_bf16, rows, hidden_dim, vocab)?;
    if cuda_reference_kernels_enabled() {
        return cuda_lm_head_argmax_bf16(hidden_bf16, lm_head_bf16, rows, hidden_dim, vocab);
    }
    Ok(cpu_lm_head_argmax_bf16(
        hidden_bf16,
        lm_head_bf16,
        rows,
        hidden_dim,
        vocab,
    ))
}

pub(in crate::commands::real_full) fn lm_head_argmax_bf16_resident_weight(
    lm_head_name: &str,
    hidden_bf16: &[u8],
    lm_head_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
) -> Result<LogitsArgmaxOutput> {
    validate_resident_weight_name(lm_head_name)?;
    validate_lm_head_bf16_inputs(hidden_bf16, lm_head_bf16, rows, hidden_dim, vocab)?;
    if cuda_reference_kernels_enabled() {
        return cuda_lm_head_argmax_bf16_resident_weight(
            lm_head_name,
            hidden_bf16,
            lm_head_bf16,
            rows,
            hidden_dim,
            vocab,
        );
    }
    Ok(cpu_lm_head_argmax_bf16(
        hidden_bf16,
        lm_head_bf16,
        rows,
        hidden_dim,
        vocab,
    ))
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn lm_head_argmax_bf16_preloaded_resident_weight(
    lm_head_name: &str,
    hidden_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    full_vocab: usize,
    start_token_id: usize,
    vocab: usize,
) -> Result<LogitsArgmaxOutput> {
    validate_resident_weight_name(lm_head_name)?;
    let view = validate_lm_head_preloaded_bf16_inputs(
        hidden_bf16,
        rows,
        hidden_dim,
        full_vocab,
        start_token_id,
        vocab,
    )?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "preloaded resident BF16 lm_head argmax requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1"
        );
    }
    cuda_lm_head_argmax_bf16_preloaded_resident_weight(
        lm_head_name,
        hidden_bf16,
        rows,
        hidden_dim,
        vocab,
        view,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn lm_head_argmax_bf16_preloaded_resident_weight_device_input(
    lm_head_name: &str,
    hidden: &DeviceBf16Output,
    full_vocab: usize,
    start_token_id: usize,
    vocab: usize,
) -> Result<LogitsArgmaxOutput> {
    validate_resident_weight_name(lm_head_name)?;
    let view =
        validate_lm_head_preloaded_bf16_device_input(hidden, full_vocab, start_token_id, vocab)?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "preloaded resident BF16 lm_head device-input argmax requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1"
        );
    }
    cuda_lm_head_argmax_bf16_preloaded_resident_weight_device_input(
        lm_head_name,
        hidden.buffer(),
        hidden.rows,
        hidden.values_per_row,
        vocab,
        view,
    )
}

pub(in crate::commands::real_full) fn lm_head_argmax_bf16_staged_resident_weight(
    lm_head_name: &str,
    hidden_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
) -> Result<LogitsArgmaxOutput> {
    validate_resident_weight_name(lm_head_name)?;
    let lm_head_bytes =
        validate_lm_head_bf16_resident_window_inputs(hidden_bf16, rows, hidden_dim, vocab)?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "staged resident BF16 lm_head argmax requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1"
        );
    }
    cuda_lm_head_argmax_bf16_preloaded_resident_weight(
        lm_head_name,
        hidden_bf16,
        rows,
        hidden_dim,
        vocab,
        LmHeadResidentView {
            full_bytes: lm_head_bytes,
            offset_bytes: 0,
            view_bytes: lm_head_bytes,
        },
    )
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(in crate::commands::real_full) fn lm_head_sample_topk_topp_bf16(
    hidden_bf16: &[u8],
    lm_head_bf16: &[u8],
    random_uniforms: &[f32],
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Result<LogitsSampleTopKToppOutput> {
    validate_lm_head_sample_topk_topp_bf16_inputs(
        hidden_bf16,
        lm_head_bf16,
        random_uniforms,
        rows,
        hidden_dim,
        vocab,
        temperature,
        top_k,
        top_p,
    )?;
    if cuda_reference_kernels_enabled() {
        return cuda_lm_head_sample_topk_topp_bf16(
            hidden_bf16,
            lm_head_bf16,
            random_uniforms,
            rows,
            hidden_dim,
            vocab,
            temperature,
            top_k,
            top_p,
        );
    }
    Ok(cpu_lm_head_sample_topk_topp_bf16(
        hidden_bf16,
        lm_head_bf16,
        random_uniforms,
        rows,
        hidden_dim,
        vocab,
        temperature,
        top_k,
        top_p,
    ))
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn lm_head_sample_topk_topp_bf16_preloaded_resident_weight(
    lm_head_name: &str,
    hidden_bf16: &[u8],
    random_uniforms: &[f32],
    rows: usize,
    hidden_dim: usize,
    full_vocab: usize,
    start_token_id: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Result<LogitsSampleTopKToppOutput> {
    validate_resident_weight_name(lm_head_name)?;
    let view = validate_lm_head_preloaded_bf16_inputs(
        hidden_bf16,
        rows,
        hidden_dim,
        full_vocab,
        start_token_id,
        vocab,
    )?;
    validate_lm_head_sampler_options(random_uniforms, rows, vocab, temperature, top_k, top_p)?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "preloaded resident BF16 lm_head sampler requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1"
        );
    }
    cuda_lm_head_sample_topk_topp_bf16_preloaded_resident_weight(
        lm_head_name,
        hidden_bf16,
        random_uniforms,
        rows,
        hidden_dim,
        vocab,
        temperature,
        top_k,
        top_p,
        view,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn lm_head_sample_topk_topp_bf16_preloaded_resident_weight_device_input(
    lm_head_name: &str,
    hidden: &DeviceBf16Output,
    random_uniforms: &[f32],
    full_vocab: usize,
    start_token_id: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Result<LogitsSampleTopKToppOutput> {
    validate_resident_weight_name(lm_head_name)?;
    let view =
        validate_lm_head_preloaded_bf16_device_input(hidden, full_vocab, start_token_id, vocab)?;
    validate_lm_head_sampler_options(
        random_uniforms,
        hidden.rows,
        vocab,
        temperature,
        top_k,
        top_p,
    )?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "preloaded resident BF16 lm_head device-input sampler requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1"
        );
    }
    cuda_lm_head_sample_topk_topp_bf16_preloaded_resident_weight_device_input(
        lm_head_name,
        hidden.buffer(),
        random_uniforms,
        hidden.rows,
        hidden.values_per_row,
        vocab,
        temperature,
        top_k,
        top_p,
        view,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn lm_head_argmax_sample_topk_topp_bf16_preloaded_resident_weight_device_input(
    lm_head_name: &str,
    hidden: &DeviceBf16Output,
    random_uniforms: &[f32],
    full_vocab: usize,
    start_token_id: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Result<LogitsArgmaxSampleTopKToppOutput> {
    validate_resident_weight_name(lm_head_name)?;
    let view =
        validate_lm_head_preloaded_bf16_device_input(hidden, full_vocab, start_token_id, vocab)?;
    validate_lm_head_sampler_options(
        random_uniforms,
        hidden.rows,
        vocab,
        temperature,
        top_k,
        top_p,
    )?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "preloaded resident BF16 lm_head device-input argmax+sampler requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1"
        );
    }
    cuda_lm_head_argmax_sample_topk_topp_bf16_preloaded_resident_weight_device_input(
        lm_head_name,
        hidden.buffer(),
        random_uniforms,
        hidden.rows,
        hidden.values_per_row,
        vocab,
        temperature,
        top_k,
        top_p,
        view,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn lm_head_argmax_sample_topk_topp_bf16_preloaded_resident_weight_device_input_without_graph(
    lm_head_name: &str,
    hidden: &DeviceBf16Output,
    random_uniforms: &[f32],
    full_vocab: usize,
    start_token_id: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Result<LogitsArgmaxSampleTopKToppOutput> {
    validate_resident_weight_name(lm_head_name)?;
    let view =
        validate_lm_head_preloaded_bf16_device_input(hidden, full_vocab, start_token_id, vocab)?;
    validate_lm_head_sampler_options(
        random_uniforms,
        hidden.rows,
        vocab,
        temperature,
        top_k,
        top_p,
    )?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "preloaded resident BF16 lm_head device-input argmax+sampler requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1"
        );
    }
    cuda_lm_head_argmax_sample_topk_topp_bf16_preloaded_resident_weight_device_input(
        lm_head_name,
        hidden.buffer(),
        random_uniforms,
        hidden.rows,
        hidden.values_per_row,
        vocab,
        temperature,
        top_k,
        top_p,
        view,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn lm_head_sample_topk_topp_bf16_staged_resident_weight(
    lm_head_name: &str,
    hidden_bf16: &[u8],
    random_uniforms: &[f32],
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Result<LogitsSampleTopKToppOutput> {
    validate_resident_weight_name(lm_head_name)?;
    let lm_head_bytes =
        validate_lm_head_bf16_resident_window_inputs(hidden_bf16, rows, hidden_dim, vocab)?;
    validate_lm_head_sampler_options(random_uniforms, rows, vocab, temperature, top_k, top_p)?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "staged resident BF16 lm_head sampler requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1"
        );
    }
    cuda_lm_head_sample_topk_topp_bf16_preloaded_resident_weight(
        lm_head_name,
        hidden_bf16,
        random_uniforms,
        rows,
        hidden_dim,
        vocab,
        temperature,
        top_k,
        top_p,
        LmHeadResidentView {
            full_bytes: lm_head_bytes,
            offset_bytes: 0,
            view_bytes: lm_head_bytes,
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn lm_head_sample_topk_topp_bf16_resident_weight(
    lm_head_name: &str,
    hidden_bf16: &[u8],
    lm_head_bf16: &[u8],
    random_uniforms: &[f32],
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Result<LogitsSampleTopKToppOutput> {
    validate_resident_weight_name(lm_head_name)?;
    validate_lm_head_sample_topk_topp_bf16_inputs(
        hidden_bf16,
        lm_head_bf16,
        random_uniforms,
        rows,
        hidden_dim,
        vocab,
        temperature,
        top_k,
        top_p,
    )?;
    if cuda_reference_kernels_enabled() {
        return cuda_lm_head_sample_topk_topp_bf16_resident_weight(
            lm_head_name,
            hidden_bf16,
            lm_head_bf16,
            random_uniforms,
            rows,
            hidden_dim,
            vocab,
            temperature,
            top_k,
            top_p,
        );
    }
    Ok(cpu_lm_head_sample_topk_topp_bf16(
        hidden_bf16,
        lm_head_bf16,
        random_uniforms,
        rows,
        hidden_dim,
        vocab,
        temperature,
        top_k,
        top_p,
    ))
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn validate_logits_argmax_inputs(
    logits: &[f32],
    rows: usize,
    vocab: usize,
) -> Result<()> {
    if rows == 0 || vocab == 0 {
        anyhow::bail!(
            "real full logits argmax requires non-zero shape, got rows={rows} vocab={vocab}"
        );
    }
    let expected_values = rows.checked_mul(vocab).context(
        "real full logits argmax shape overflows usize while validating coordinator kernel input",
    )?;
    if logits.len() != expected_values {
        anyhow::bail!(
            "real full logits argmax length mismatch: expected {} got {}",
            expected_values,
            logits.len()
        );
    }
    Ok(())
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn validate_logits_sample_topk_topp_inputs(
    logits: &[f32],
    random_uniforms: &[f32],
    rows: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Result<()> {
    if rows == 0 || vocab == 0 {
        anyhow::bail!(
            "real full logits top-k/top-p sampler requires non-zero shape, got rows={rows} vocab={vocab}"
        );
    }
    if top_k == 0 || top_k > vocab || top_k > GLMRT_CUDA_SAMPLE_TOPK_MAX_K {
        anyhow::bail!(
            "real full logits top-k/top-p sampler invalid top_k={top_k} for vocab={vocab}; max supported top_k={GLMRT_CUDA_SAMPLE_TOPK_MAX_K}"
        );
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        anyhow::bail!(
            "real full logits top-k/top-p sampler temperature must be finite and positive"
        );
    }
    if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
        anyhow::bail!("real full logits top-k/top-p sampler top_p must be finite and in (0, 1]");
    }
    let expected_values = rows.checked_mul(vocab).context(
        "real full logits top-k/top-p sampler shape overflows usize while validating coordinator kernel input",
    )?;
    if logits.len() != expected_values {
        anyhow::bail!(
            "real full logits top-k/top-p sampler length mismatch: expected {} got {}",
            expected_values,
            logits.len()
        );
    }
    if random_uniforms.len() != rows {
        anyhow::bail!(
            "real full logits top-k/top-p sampler random uniform length mismatch: expected {} got {}",
            rows,
            random_uniforms.len()
        );
    }
    if random_uniforms.iter().any(|value| !value.is_finite()) {
        anyhow::bail!("real full logits top-k/top-p sampler random uniforms must be finite");
    }
    Ok(())
}

pub(in crate::commands::real_full) fn validate_lm_head_bf16_inputs(
    hidden_bf16: &[u8],
    lm_head_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
) -> Result<()> {
    if rows == 0 || hidden_dim == 0 || vocab == 0 {
        anyhow::bail!(
            "real full BF16 lm_head scorer requires non-zero shape, got rows={rows} hidden_dim={hidden_dim} vocab={vocab}"
        );
    }
    let hidden_bytes = rows
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full BF16 lm_head scorer hidden shape overflows usize while validating input",
        )?;
    if hidden_bf16.len() != hidden_bytes {
        anyhow::bail!(
            "real full BF16 lm_head scorer hidden byte length mismatch: expected {} got {}",
            hidden_bytes,
            hidden_bf16.len()
        );
    }
    let lm_head_bytes = vocab
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full BF16 lm_head scorer weight shape overflows usize while validating input",
        )?;
    if lm_head_bf16.len() != lm_head_bytes {
        anyhow::bail!(
            "real full BF16 lm_head scorer weight byte length mismatch: expected {} got {}",
            lm_head_bytes,
            lm_head_bf16.len()
        );
    }
    Ok(())
}

pub(in crate::commands::real_full) fn validate_lm_head_bf16_resident_window_inputs(
    hidden_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
) -> Result<usize> {
    if rows == 0 || hidden_dim == 0 || vocab == 0 {
        anyhow::bail!(
            "real full staged resident BF16 lm_head scorer requires non-zero shape, got rows={rows} hidden_dim={hidden_dim} vocab={vocab}"
        );
    }
    let hidden_bytes = rows
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full staged resident BF16 lm_head scorer hidden shape overflows usize while validating input",
        )?;
    if hidden_bf16.len() != hidden_bytes {
        anyhow::bail!(
            "real full staged resident BF16 lm_head scorer hidden byte length mismatch: expected {} got {}",
            hidden_bytes,
            hidden_bf16.len()
        );
    }
    vocab
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full staged resident BF16 lm_head scorer weight shape overflows usize while validating input",
        )
}

pub(in crate::commands::real_full) fn validate_lm_head_preloaded_bf16_inputs(
    hidden_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    full_vocab: usize,
    start_token_id: usize,
    vocab: usize,
) -> Result<LmHeadResidentView> {
    if rows == 0 || hidden_dim == 0 || full_vocab == 0 || vocab == 0 {
        anyhow::bail!(
            "real full preloaded BF16 lm_head scorer requires non-zero shape, got rows={rows} hidden_dim={hidden_dim} full_vocab={full_vocab} vocab={vocab}"
        );
    }
    if vocab > u32::MAX as usize {
        anyhow::bail!("real full preloaded BF16 lm_head scorer vocab must fit CUDA u32 indices");
    }
    let chunk_end = start_token_id
        .checked_add(vocab)
        .context("real full preloaded BF16 lm_head scorer token range overflows usize")?;
    if chunk_end > full_vocab {
        anyhow::bail!(
            "real full preloaded BF16 lm_head scorer chunk [{}, {}) exceeds full vocab {}",
            start_token_id,
            chunk_end,
            full_vocab
        );
    }
    let hidden_bytes = rows
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full preloaded BF16 lm_head scorer hidden shape overflows usize while validating input",
        )?;
    if hidden_bf16.len() != hidden_bytes {
        anyhow::bail!(
            "real full preloaded BF16 lm_head scorer hidden byte length mismatch: expected {} got {}",
            hidden_bytes,
            hidden_bf16.len()
        );
    }
    let row_bytes = hidden_dim
        .checked_mul(std::mem::size_of::<u16>())
        .context("real full preloaded BF16 lm_head row byte width overflows usize")?;
    let full_bytes = full_vocab
        .checked_mul(row_bytes)
        .context("real full preloaded BF16 lm_head full tensor bytes overflow usize")?;
    let offset_bytes = start_token_id
        .checked_mul(row_bytes)
        .context("real full preloaded BF16 lm_head chunk offset overflows usize")?;
    let view_bytes = vocab
        .checked_mul(row_bytes)
        .context("real full preloaded BF16 lm_head chunk byte count overflows usize")?;
    Ok(LmHeadResidentView {
        full_bytes,
        offset_bytes,
        view_bytes,
    })
}

pub(in crate::commands::real_full) fn validate_lm_head_preloaded_bf16_device_input(
    hidden: &DeviceBf16Output,
    full_vocab: usize,
    start_token_id: usize,
    vocab: usize,
) -> Result<LmHeadResidentView> {
    if hidden.rows == 0 || hidden.values_per_row == 0 || full_vocab == 0 || vocab == 0 {
        anyhow::bail!(
            "real full preloaded BF16 lm_head device-input scorer requires non-zero shape, got rows={} hidden_dim={} full_vocab={full_vocab} vocab={vocab}",
            hidden.rows,
            hidden.values_per_row
        );
    }
    if vocab > u32::MAX as usize {
        anyhow::bail!(
            "real full preloaded BF16 lm_head device-input scorer vocab must fit CUDA u32 indices"
        );
    }
    let chunk_end = start_token_id.checked_add(vocab).context(
        "real full preloaded BF16 lm_head device-input scorer token range overflows usize",
    )?;
    if chunk_end > full_vocab {
        anyhow::bail!(
            "real full preloaded BF16 lm_head device-input scorer chunk [{}, {}) exceeds full vocab {}",
            start_token_id,
            chunk_end,
            full_vocab
        );
    }
    let hidden_bytes = hidden
        .rows
        .checked_mul(hidden.values_per_row)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full preloaded BF16 lm_head device-input scorer hidden shape overflows usize while validating input",
        )?;
    if hidden.bytes != hidden_bytes {
        anyhow::bail!(
            "real full preloaded BF16 lm_head device-input scorer hidden byte length mismatch: expected {} got {}",
            hidden_bytes,
            hidden.bytes
        );
    }
    let hidden_buffer = hidden.buffer();
    if hidden_buffer.ptr.is_null() || hidden_buffer.bytes < hidden_bytes {
        anyhow::bail!(
            "real full preloaded BF16 lm_head device-input scorer buffer byte length mismatch: expected at least {} got {}",
            hidden_bytes,
            hidden_buffer.bytes
        );
    }
    let row_bytes = hidden
        .values_per_row
        .checked_mul(std::mem::size_of::<u16>())
        .context("real full preloaded BF16 lm_head device-input row byte width overflows usize")?;
    let full_bytes = full_vocab.checked_mul(row_bytes).context(
        "real full preloaded BF16 lm_head device-input full tensor bytes overflow usize",
    )?;
    let offset_bytes = start_token_id
        .checked_mul(row_bytes)
        .context("real full preloaded BF16 lm_head device-input chunk offset overflows usize")?;
    let view_bytes = vocab.checked_mul(row_bytes).context(
        "real full preloaded BF16 lm_head device-input chunk byte count overflows usize",
    )?;
    Ok(LmHeadResidentView {
        full_bytes,
        offset_bytes,
        view_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn validate_lm_head_sample_topk_topp_bf16_inputs(
    hidden_bf16: &[u8],
    lm_head_bf16: &[u8],
    random_uniforms: &[f32],
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Result<()> {
    validate_lm_head_bf16_inputs(hidden_bf16, lm_head_bf16, rows, hidden_dim, vocab)?;
    validate_lm_head_sampler_options(random_uniforms, rows, vocab, temperature, top_k, top_p)
}

pub(in crate::commands::real_full) fn validate_lm_head_sampler_options(
    random_uniforms: &[f32],
    rows: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Result<()> {
    if top_k == 0 || top_k > vocab || top_k > GLMRT_CUDA_SAMPLE_TOPK_MAX_K {
        anyhow::bail!(
            "real full BF16 lm_head sampler invalid top_k={top_k} for vocab={vocab}; max supported top_k={GLMRT_CUDA_SAMPLE_TOPK_MAX_K}"
        );
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        anyhow::bail!("real full BF16 lm_head sampler temperature must be finite and positive");
    }
    if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
        anyhow::bail!("real full BF16 lm_head sampler top_p must be finite and in (0, 1]");
    }
    if random_uniforms.len() != rows {
        anyhow::bail!(
            "real full BF16 lm_head sampler random uniform length mismatch: expected {} got {}",
            rows,
            random_uniforms.len()
        );
    }
    if random_uniforms.iter().any(|value| !value.is_finite()) {
        anyhow::bail!("real full BF16 lm_head sampler random uniforms must be finite");
    }
    Ok(())
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn cpu_logits_argmax(
    logits: &[f32],
    rows: usize,
    vocab: usize,
) -> LogitsArgmaxOutput {
    let mut indices = vec![0_usize; rows];
    let mut scores = vec![f32::NEG_INFINITY; rows];
    for row in 0..rows {
        let row_start = row * vocab;
        let row_logits = &logits[row_start..row_start + vocab];
        for (token_id, logit) in row_logits.iter().copied().enumerate() {
            if logit > scores[row] {
                scores[row] = logit;
                indices[row] = token_id;
            }
        }
    }
    LogitsArgmaxOutput {
        indices,
        scores,
        backend: CPU_REFERENCE_LOGITS_ARGMAX_BACKEND,
    }
}

pub(in crate::commands::real_full) fn cpu_logits_sample_topk_topp(
    logits: &[f32],
    random_uniforms: &[f32],
    rows: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> LogitsSampleTopKToppOutput {
    let mut indices = vec![0_usize; rows];
    let mut scores = vec![0.0_f32; rows];
    for row in 0..rows {
        let mut best_logits = vec![f32::NEG_INFINITY; top_k];
        let mut best_indices = vec![0_usize; top_k];
        for col in 0..vocab {
            let logit = logits[row * vocab + col];
            for rank in 0..top_k {
                if logit > best_logits[rank]
                    || (logit == best_logits[rank] && col < best_indices[rank])
                {
                    for shift in (rank + 1..top_k).rev() {
                        best_logits[shift] = best_logits[shift - 1];
                        best_indices[shift] = best_indices[shift - 1];
                    }
                    best_logits[rank] = logit;
                    best_indices[rank] = col;
                    break;
                }
            }
        }

        let mut scaled = vec![0.0_f32; top_k];
        let mut max_scaled = f32::NEG_INFINITY;
        for rank in 0..top_k {
            scaled[rank] = best_logits[rank] / temperature;
            max_scaled = max_scaled.max(scaled[rank]);
        }
        let mut probs = vec![0.0_f32; top_k];
        let mut total = 0.0_f32;
        for rank in 0..top_k {
            probs[rank] = (scaled[rank] - max_scaled).exp();
            total += probs[rank];
        }
        total = total.max(1.0e-20);
        for prob in &mut probs {
            *prob /= total;
        }

        let top_p_clamped = top_p.clamp(1.0e-6, 1.0);
        let mut nucleus_mass = 0.0_f32;
        let mut nucleus_count = 0_usize;
        for (rank, prob) in probs.iter().enumerate() {
            nucleus_mass += *prob;
            nucleus_count = rank + 1;
            if nucleus_mass >= top_p_clamped {
                break;
            }
        }
        nucleus_mass = nucleus_mass.max(1.0e-20);

        let target = random_uniforms[row].clamp(0.0, 0.999_999_94) * nucleus_mass;
        let mut cumulative = 0.0_f32;
        let mut selected_rank = nucleus_count - 1;
        for (rank, prob) in probs.iter().enumerate().take(nucleus_count) {
            cumulative += *prob;
            if target <= cumulative {
                selected_rank = rank;
                break;
            }
        }
        indices[row] = best_indices[selected_rank];
        scores[row] = probs[selected_rank] / nucleus_mass;
    }
    LogitsSampleTopKToppOutput {
        indices,
        scores,
        backend: CPU_REFERENCE_LOGITS_SAMPLE_TOPK_TOPP_BACKEND,
    }
}

pub(in crate::commands::real_full) fn cpu_lm_head_argmax_bf16(
    hidden_bf16: &[u8],
    lm_head_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
) -> LogitsArgmaxOutput {
    let mut indices = vec![0_usize; rows];
    let mut scores = vec![f32::NEG_INFINITY; rows];
    for row in 0..rows {
        for token_id in 0..vocab {
            let score = bf16_lm_head_dot(hidden_bf16, lm_head_bf16, row, token_id, hidden_dim);
            if score > scores[row] || (score == scores[row] && token_id < indices[row]) {
                scores[row] = score;
                indices[row] = token_id;
            }
        }
    }
    LogitsArgmaxOutput {
        indices,
        scores,
        backend: CPU_REFERENCE_LM_HEAD_ARGMAX_BF16_BACKEND,
    }
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cpu_lm_head_sample_topk_topp_bf16(
    hidden_bf16: &[u8],
    lm_head_bf16: &[u8],
    random_uniforms: &[f32],
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> LogitsSampleTopKToppOutput {
    let mut logits = vec![0.0_f32; rows * vocab];
    for row in 0..rows {
        for token_id in 0..vocab {
            logits[row * vocab + token_id] =
                bf16_lm_head_dot(hidden_bf16, lm_head_bf16, row, token_id, hidden_dim);
        }
    }
    let mut output = cpu_logits_sample_topk_topp(
        &logits,
        random_uniforms,
        rows,
        vocab,
        temperature,
        top_k,
        top_p,
    );
    output.backend = CPU_REFERENCE_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_BACKEND;
    output
}

pub(in crate::commands::real_full) fn bf16_lm_head_dot(
    hidden_bf16: &[u8],
    lm_head_bf16: &[u8],
    row: usize,
    token_id: usize,
    hidden_dim: usize,
) -> f32 {
    let mut score = 0.0_f32;
    let hidden_start = row * hidden_dim;
    let weight_start = token_id * hidden_dim;
    for col in 0..hidden_dim {
        score += bf16_value(hidden_bf16, hidden_start + col)
            * bf16_value(lm_head_bf16, weight_start + col);
    }
    score
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_logits_argmax(
    logits: &[f32],
    rows: usize,
    vocab: usize,
) -> Result<LogitsArgmaxOutput> {
    let library = cuda_native_library()?;
    let logits_bytes = std::mem::size_of_val(logits);
    let index_bytes = rows
        .checked_mul(std::mem::size_of::<u32>())
        .context("CUDA logits argmax index bytes overflow usize")?;
    let score_bytes = rows
        .checked_mul(std::mem::size_of::<f32>())
        .context("CUDA logits argmax score bytes overflow usize")?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let logits_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        logits_bytes,
        "logits argmax logits",
    )?;
    let index_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::B,
        index_bytes,
        "logits argmax indices",
    )?;
    let score_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        score_bytes,
        "logits argmax scores",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            f32_bytes(logits),
            "logits argmax logits",
        )
        .context("copying logits to device")?;
    library
        .cuda_logits_argmax_f32(logits_buffer, index_buffer, score_buffer, rows, vocab)
        .context("executing CUDA logits argmax")?;
    let mut index_out = vec![0_u8; index_bytes];
    let mut score_out = vec![0_u8; score_bytes];
    library
        .copy_d2h(&mut index_out, index_buffer)
        .context("copying logits argmax indices to host")?;
    library
        .copy_d2h(&mut score_out, score_buffer)
        .context("copying logits argmax scores to host")?;

    Ok(LogitsArgmaxOutput {
        indices: u32_vec_from_bytes(&index_out)?
            .into_iter()
            .map(|value| value as usize)
            .collect(),
        scores: f32_vec_from_bytes(&score_out)?,
        backend: CUDA_REFERENCE_LOGITS_ARGMAX_BACKEND,
    })
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_logits_sample_topk_topp(
    logits: &[f32],
    random_uniforms: &[f32],
    rows: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Result<LogitsSampleTopKToppOutput> {
    let library = cuda_native_library()?;
    let logits_bytes = std::mem::size_of_val(logits);
    let random_bytes = std::mem::size_of_val(random_uniforms);
    let logits_index_bytes = logits
        .len()
        .checked_mul(std::mem::size_of::<u32>())
        .context("CUDA CUB logits top-k/top-p sampler index workspace bytes overflow usize")?;
    let segment_offset_bytes = rows
        .checked_add(1)
        .and_then(|values| values.checked_mul(std::mem::size_of::<i32>()))
        .context("CUDA CUB logits top-k/top-p sampler segment offset bytes overflow usize")?;
    let index_bytes = rows
        .checked_mul(std::mem::size_of::<u32>())
        .context("CUDA logits top-k/top-p sampler index bytes overflow usize")?;
    let score_bytes = rows
        .checked_mul(std::mem::size_of::<f32>())
        .context("CUDA logits top-k/top-p sampler score bytes overflow usize")?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let logits_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        logits_bytes,
        "logits sampler logits",
    )?;
    let random_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::B,
        random_bytes,
        "logits sampler random uniforms",
    )?;
    let sorted_logits_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        logits_bytes,
        "CUB logits sampler sorted logits",
    )?;
    let unsorted_index_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        logits_index_bytes,
        "CUB logits sampler unsorted indices",
    )?;
    let sorted_index_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::E,
        logits_index_bytes,
        "CUB logits sampler sorted indices",
    )?;
    let segment_offset_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::F,
        segment_offset_bytes,
        "CUB logits sampler segment offsets",
    )?;
    let temp_storage_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::G,
        CUDA_LOGITS_SAMPLE_TOPK_TOPP_CUB_TEMP_STORAGE_BYTES,
        "CUB logits sampler temp storage",
    )?;
    let index_buffer = sorted_index_buffer;
    let score_buffer = sorted_logits_buffer;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            f32_bytes(logits),
            "logits sampler logits",
        )
        .context("copying logits sampler logits to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::B,
            f32_bytes(random_uniforms),
            "logits sampler random uniforms",
        )
        .context("copying logits sampler random uniforms to device")?;
    library
        .cuda_logits_sample_topk_topp_f32_cub(
            logits_buffer,
            random_buffer,
            sorted_logits_buffer,
            unsorted_index_buffer,
            sorted_index_buffer,
            segment_offset_buffer,
            index_buffer,
            score_buffer,
            temp_storage_buffer,
            CUDA_LOGITS_SAMPLE_TOPK_TOPP_CUB_TEMP_STORAGE_BYTES,
            rows,
            vocab,
            temperature,
            top_k,
            top_p,
        )
        .context("executing CUDA CUB logits top-k/top-p sampler")?;
    let mut index_out = vec![0_u8; index_bytes];
    let mut score_out = vec![0_u8; score_bytes];
    library
        .copy_d2h(&mut index_out, index_buffer)
        .context("copying logits sampler indices to host")?;
    library
        .copy_d2h(&mut score_out, score_buffer)
        .context("copying logits sampler scores to host")?;

    Ok(LogitsSampleTopKToppOutput {
        indices: u32_vec_from_bytes(&index_out)?
            .into_iter()
            .map(|value| value as usize)
            .collect(),
        scores: f32_vec_from_bytes(&score_out)?,
        backend: CUDA_REFERENCE_LOGITS_SAMPLE_TOPK_TOPP_BACKEND,
    })
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_lm_head_argmax_bf16(
    hidden_bf16: &[u8],
    lm_head_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
) -> Result<LogitsArgmaxOutput> {
    let library = cuda_native_library()?;
    let index_bytes = rows
        .checked_mul(std::mem::size_of::<u32>())
        .context("CUDA BF16 lm_head argmax index bytes overflow usize")?;
    let score_bytes = rows
        .checked_mul(std::mem::size_of::<f32>())
        .context("CUDA BF16 lm_head argmax score bytes overflow usize")?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let hidden_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        hidden_bf16.len(),
        "BF16 lm_head argmax hidden",
    )?;
    let lm_head_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::B,
        lm_head_bf16.len(),
        "BF16 lm_head argmax weights",
    )?;
    let index_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        index_bytes,
        "BF16 lm_head argmax indices",
    )?;
    let score_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        score_bytes,
        "BF16 lm_head argmax scores",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            hidden_bf16,
            "BF16 lm_head argmax hidden",
        )
        .context("copying BF16 lm_head argmax hidden to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::B,
            lm_head_bf16,
            "BF16 lm_head argmax weights",
        )
        .context("copying BF16 lm_head argmax weights to device")?;
    library
        .cuda_lm_head_argmax_bf16(
            hidden_buffer,
            lm_head_buffer,
            index_buffer,
            score_buffer,
            rows,
            hidden_dim,
            vocab,
        )
        .context("executing CUDA BF16 lm_head argmax")?;
    let mut index_out = vec![0_u8; index_bytes];
    let mut score_out = vec![0_u8; score_bytes];
    library
        .copy_d2h(&mut index_out, index_buffer)
        .context("copying BF16 lm_head argmax indices to host")?;
    library
        .copy_d2h(&mut score_out, score_buffer)
        .context("copying BF16 lm_head argmax scores to host")?;

    Ok(LogitsArgmaxOutput {
        indices: u32_vec_from_bytes(&index_out)?
            .into_iter()
            .map(|value| value as usize)
            .collect(),
        scores: f32_vec_from_bytes(&score_out)?,
        backend: CUDA_REFERENCE_LM_HEAD_ARGMAX_BF16_BACKEND,
    })
}

pub(in crate::commands::real_full) fn cuda_lm_head_argmax_bf16_resident_weight(
    lm_head_name: &str,
    hidden_bf16: &[u8],
    lm_head_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
) -> Result<LogitsArgmaxOutput> {
    let library = cuda_native_library()?;
    let index_bytes = rows
        .checked_mul(std::mem::size_of::<u32>())
        .context("CUDA BF16 resident lm_head argmax index bytes overflow usize")?;
    let score_bytes = rows
        .checked_mul(std::mem::size_of::<f32>())
        .context("CUDA BF16 resident lm_head argmax score bytes overflow usize")?;
    let lm_head_buffer = resident_weight_buffer_from_registry(
        lm_head_name,
        lm_head_bf16,
        "BF16 resident lm_head argmax weights",
    )?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let hidden_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        hidden_bf16.len(),
        "BF16 resident lm_head argmax hidden",
    )?;
    let index_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        index_bytes,
        "BF16 resident lm_head argmax indices",
    )?;
    let score_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        score_bytes,
        "BF16 resident lm_head argmax scores",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            hidden_bf16,
            "BF16 resident lm_head argmax hidden",
        )
        .context("copying BF16 resident lm_head argmax hidden to device")?;
    library
        .cuda_lm_head_argmax_bf16(
            hidden_buffer,
            lm_head_buffer,
            index_buffer,
            score_buffer,
            rows,
            hidden_dim,
            vocab,
        )
        .context("executing CUDA BF16 resident lm_head argmax")?;
    let mut index_out = vec![0_u8; index_bytes];
    let mut score_out = vec![0_u8; score_bytes];
    library
        .copy_d2h(&mut index_out, index_buffer)
        .context("copying BF16 resident lm_head argmax indices to host")?;
    library
        .copy_d2h(&mut score_out, score_buffer)
        .context("copying BF16 resident lm_head argmax scores to host")?;

    Ok(LogitsArgmaxOutput {
        indices: u32_vec_from_bytes(&index_out)?
            .into_iter()
            .map(|value| value as usize)
            .collect(),
        scores: f32_vec_from_bytes(&score_out)?,
        backend: CUDA_REFERENCE_LM_HEAD_ARGMAX_BF16_RESIDENT_WEIGHT_BACKEND,
    })
}

pub(in crate::commands::real_full) fn cuda_lm_head_argmax_bf16_preloaded_resident_weight(
    lm_head_name: &str,
    hidden_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    view: LmHeadResidentView,
) -> Result<LogitsArgmaxOutput> {
    let library = cuda_native_library()?;
    let index_bytes = rows
        .checked_mul(std::mem::size_of::<u32>())
        .context("CUDA BF16 preloaded resident lm_head argmax index bytes overflow usize")?;
    let score_bytes = rows
        .checked_mul(std::mem::size_of::<f32>())
        .context("CUDA BF16 preloaded resident lm_head argmax score bytes overflow usize")?;
    let lm_head_buffer = preloaded_resident_weight_device_buffer_view(
        lm_head_name,
        view.full_bytes,
        view.offset_bytes,
        view.view_bytes,
    )?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let hidden_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        hidden_bf16.len(),
        "BF16 preloaded resident lm_head argmax hidden",
    )?;
    let index_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        index_bytes,
        "BF16 preloaded resident lm_head argmax indices",
    )?;
    let score_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        score_bytes,
        "BF16 preloaded resident lm_head argmax scores",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            hidden_bf16,
            "BF16 preloaded resident lm_head argmax hidden",
        )
        .context("copying BF16 preloaded resident lm_head argmax hidden to device")?;
    library
        .cuda_lm_head_argmax_bf16(
            hidden_buffer,
            lm_head_buffer,
            index_buffer,
            score_buffer,
            rows,
            hidden_dim,
            vocab,
        )
        .context("executing CUDA BF16 preloaded resident lm_head argmax")?;
    let mut index_out = vec![0_u8; index_bytes];
    let mut score_out = vec![0_u8; score_bytes];
    library
        .copy_d2h(&mut index_out, index_buffer)
        .context("copying BF16 preloaded resident lm_head argmax indices to host")?;
    library
        .copy_d2h(&mut score_out, score_buffer)
        .context("copying BF16 preloaded resident lm_head argmax scores to host")?;

    Ok(LogitsArgmaxOutput {
        indices: u32_vec_from_bytes(&index_out)?
            .into_iter()
            .map(|value| value as usize)
            .collect(),
        scores: f32_vec_from_bytes(&score_out)?,
        backend: CUDA_REFERENCE_LM_HEAD_ARGMAX_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn capture_or_update_terminal_lm_head_argmax_bf16_graph(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    signature: CoordinatorCudaGraphSignature,
    hidden_buffer: GlmrtDeviceBuffer,
    lm_head_buffer: GlmrtDeviceBuffer,
    index_buffer: GlmrtDeviceBuffer,
    score_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    label: &'static str,
) -> Result<()> {
    if !slot.has_captured_graph(
        CoordinatorCudaGraphProgram::TerminalLmHeadArgmaxBf16,
        signature,
    ) {
        slot.stream_synchronize()
            .with_context(|| format!("synchronizing {label} inputs before graph capture"))?;
        slot.capture_graph(
            library,
            CoordinatorCudaGraphProgram::TerminalLmHeadArgmaxBf16,
            signature,
            |library, cuda_stream, _workspace| unsafe {
                library
                    .cuda_lm_head_argmax_bf16_async(
                        hidden_buffer,
                        lm_head_buffer,
                        index_buffer,
                        score_buffer,
                        rows,
                        hidden_dim,
                        vocab,
                        cuda_stream,
                    )
                    .with_context(|| format!("capturing async CUDA {label}"))?;
                Ok(())
            },
        )?;
    } else {
        let (graph_raw, exec_raw) = slot
            .captured_graph_raw_handles(
                CoordinatorCudaGraphProgram::TerminalLmHeadArgmaxBf16,
                signature,
            )
            .context(
                "coordinator CUDA graph slot lost captured terminal lm_head argmax graph before update",
            )?;
        unsafe {
            library
                .cuda_graph_update_lm_head_argmax_bf16_node(
                    graph_raw,
                    exec_raw,
                    0,
                    hidden_buffer,
                    lm_head_buffer,
                    index_buffer,
                    score_buffer,
                    rows,
                    hidden_dim,
                    vocab,
                )
                .with_context(|| format!("updating captured CUDA {label} graph node"))?;
        }
    }
    slot.launch_captured_graph(
        library,
        CoordinatorCudaGraphProgram::TerminalLmHeadArgmaxBf16,
        signature,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn capture_or_update_terminal_lm_head_sample_topk_topp_bf16_graph(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    signature: CoordinatorCudaGraphSignature,
    hidden_buffer: GlmrtDeviceBuffer,
    lm_head_buffer: GlmrtDeviceBuffer,
    random_buffer: GlmrtDeviceBuffer,
    index_buffer: GlmrtDeviceBuffer,
    score_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    label: &'static str,
) -> Result<()> {
    if !slot.has_captured_graph(
        CoordinatorCudaGraphProgram::TerminalLmHeadSampleTopKToppBf16,
        signature,
    ) {
        slot.stream_synchronize()
            .with_context(|| format!("synchronizing {label} inputs before graph capture"))?;
        slot.capture_graph(
            library,
            CoordinatorCudaGraphProgram::TerminalLmHeadSampleTopKToppBf16,
            signature,
            |library, cuda_stream, _workspace| unsafe {
                library
                    .cuda_lm_head_sample_topk_topp_bf16_async(
                        hidden_buffer,
                        lm_head_buffer,
                        random_buffer,
                        index_buffer,
                        score_buffer,
                        rows,
                        hidden_dim,
                        vocab,
                        temperature,
                        top_k,
                        top_p,
                        cuda_stream,
                    )
                    .with_context(|| format!("capturing async CUDA {label}"))?;
                Ok(())
            },
        )?;
    } else {
        let (graph_raw, exec_raw) = slot
            .captured_graph_raw_handles(
                CoordinatorCudaGraphProgram::TerminalLmHeadSampleTopKToppBf16,
                signature,
            )
            .context(
                "coordinator CUDA graph slot lost captured terminal lm_head sampler graph before update",
            )?;
        unsafe {
            library
                .cuda_graph_update_lm_head_sample_topk_topp_bf16_node(
                    graph_raw,
                    exec_raw,
                    0,
                    hidden_buffer,
                    lm_head_buffer,
                    random_buffer,
                    index_buffer,
                    score_buffer,
                    rows,
                    hidden_dim,
                    vocab,
                    temperature,
                    top_k,
                    top_p,
                )
                .with_context(|| format!("updating captured CUDA {label} graph node"))?;
        }
    }
    slot.launch_captured_graph(
        library,
        CoordinatorCudaGraphProgram::TerminalLmHeadSampleTopKToppBf16,
        signature,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn capture_or_update_terminal_triton_lm_head_sample_topk_topp_bf16_graph(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    signature: CoordinatorCudaGraphSignature,
    hidden_buffer: GlmrtDeviceBuffer,
    lm_head_buffer: GlmrtDeviceBuffer,
    random_buffer: GlmrtDeviceBuffer,
    logits_buffer: GlmrtDeviceBuffer,
    candidate_score_buffer: GlmrtDeviceBuffer,
    candidate_index_buffer: GlmrtDeviceBuffer,
    argmax_index_buffer: GlmrtDeviceBuffer,
    argmax_score_buffer: GlmrtDeviceBuffer,
    index_buffer: GlmrtDeviceBuffer,
    score_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    label: &'static str,
) -> Result<()> {
    if !slot.has_captured_graph(
        CoordinatorCudaGraphProgram::TerminalTritonLmHeadSampleTopKToppBf16,
        signature,
    ) {
        slot.stream_synchronize()
            .with_context(|| format!("synchronizing {label} inputs before Triton warmup"))?;
        launch_triton_lm_head_sample_topk_topp_bf16_graph_capture(
            slot.stream_ptr(),
            hidden_buffer,
            lm_head_buffer,
            random_buffer,
            logits_buffer,
            candidate_score_buffer,
            candidate_index_buffer,
            argmax_index_buffer,
            argmax_score_buffer,
            index_buffer,
            score_buffer,
            rows,
            hidden_dim,
            vocab,
            temperature,
            top_k,
            top_p,
        )
        .with_context(|| format!("warming Triton {label} graph capture"))?;
        slot.stream_synchronize()
            .with_context(|| format!("synchronizing {label} Triton warmup"))?;
        slot.capture_graph(
            library,
            CoordinatorCudaGraphProgram::TerminalTritonLmHeadSampleTopKToppBf16,
            signature,
            |_library, cuda_stream, _workspace| {
                launch_triton_lm_head_sample_topk_topp_bf16_graph_capture(
                    cuda_stream,
                    hidden_buffer,
                    lm_head_buffer,
                    random_buffer,
                    logits_buffer,
                    candidate_score_buffer,
                    candidate_index_buffer,
                    argmax_index_buffer,
                    argmax_score_buffer,
                    index_buffer,
                    score_buffer,
                    rows,
                    hidden_dim,
                    vocab,
                    temperature,
                    top_k,
                    top_p,
                )
                .with_context(|| format!("capturing Triton {label}"))?;
                Ok(())
            },
        )?;
    }
    slot.launch_captured_graph(
        library,
        CoordinatorCudaGraphProgram::TerminalTritonLmHeadSampleTopKToppBf16,
        signature,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_triton_lm_head_sample_topk_topp_bf16_graph_capture(
    cuda_stream: *mut c_void,
    hidden_buffer: GlmrtDeviceBuffer,
    lm_head_buffer: GlmrtDeviceBuffer,
    random_buffer: GlmrtDeviceBuffer,
    logits_buffer: GlmrtDeviceBuffer,
    candidate_score_buffer: GlmrtDeviceBuffer,
    candidate_index_buffer: GlmrtDeviceBuffer,
    argmax_index_buffer: GlmrtDeviceBuffer,
    argmax_score_buffer: GlmrtDeviceBuffer,
    index_buffer: GlmrtDeviceBuffer,
    score_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
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
            name: "lm_head",
            ptr: lm_head_buffer.ptr,
            bytes: lm_head_buffer.bytes,
            device_id: lm_head_buffer.device_id,
            flags: lm_head_buffer.flags,
        },
        PythonDeviceBufferArg {
            name: "random_uniforms",
            ptr: random_buffer.ptr,
            bytes: random_buffer.bytes,
            device_id: random_buffer.device_id,
            flags: random_buffer.flags,
        },
        PythonDeviceBufferArg {
            name: "logits",
            ptr: logits_buffer.ptr,
            bytes: logits_buffer.bytes,
            device_id: logits_buffer.device_id,
            flags: logits_buffer.flags,
        },
        PythonDeviceBufferArg {
            name: "candidate_scores",
            ptr: candidate_score_buffer.ptr,
            bytes: candidate_score_buffer.bytes,
            device_id: candidate_score_buffer.device_id,
            flags: candidate_score_buffer.flags,
        },
        PythonDeviceBufferArg {
            name: "candidate_indices",
            ptr: candidate_index_buffer.ptr,
            bytes: candidate_index_buffer.bytes,
            device_id: candidate_index_buffer.device_id,
            flags: candidate_index_buffer.flags,
        },
        PythonDeviceBufferArg {
            name: "out_argmax_indices",
            ptr: argmax_index_buffer.ptr,
            bytes: argmax_index_buffer.bytes,
            device_id: argmax_index_buffer.device_id,
            flags: argmax_index_buffer.flags,
        },
        PythonDeviceBufferArg {
            name: "out_argmax_scores",
            ptr: argmax_score_buffer.ptr,
            bytes: argmax_score_buffer.bytes,
            device_id: argmax_score_buffer.device_id,
            flags: argmax_score_buffer.flags,
        },
        PythonDeviceBufferArg {
            name: "out_indices",
            ptr: index_buffer.ptr,
            bytes: index_buffer.bytes,
            device_id: index_buffer.device_id,
            flags: index_buffer.flags,
        },
        PythonDeviceBufferArg {
            name: "out_scores",
            ptr: score_buffer.ptr,
            bytes: score_buffer.bytes,
            device_id: score_buffer.device_id,
            flags: score_buffer.flags,
        },
    ];
    let kwargs = [
        ("rows", PythonKernelArg::Usize(rows)),
        ("hidden_dim", PythonKernelArg::Usize(hidden_dim)),
        ("vocab", PythonKernelArg::Usize(vocab)),
        ("temperature", PythonKernelArg::F64(temperature as f64)),
        ("top_k", PythonKernelArg::Usize(top_k)),
        ("top_p", PythonKernelArg::F64(top_p as f64)),
    ];
    launch_python_kernel(PythonGraphCaptureLaunch {
        module: "triton_sampling_capture",
        function: "capture_lm_head_sample_topk_topp",
        cuda_stream,
        buffers: &buffers,
        kwargs: &kwargs,
    })
}

pub(in crate::commands::real_full) fn lm_head_output_bytes(
    rows: usize,
    context: &str,
) -> Result<(usize, usize)> {
    let index_bytes = rows
        .checked_mul(std::mem::size_of::<u32>())
        .with_context(|| format!("{context} index bytes overflow usize"))?;
    let score_bytes = rows
        .checked_mul(std::mem::size_of::<f32>())
        .with_context(|| format!("{context} score bytes overflow usize"))?;
    Ok((index_bytes, score_bytes))
}

pub(in crate::commands::real_full) fn lm_head_graph_output_bytes(
    graph_key: &CoordinatorGraphKey,
    context: &str,
) -> Result<(usize, usize)> {
    lm_head_output_bytes(graph_key.row_bucket.row_capacity, context)
}

fn triton_lm_head_sampler_supported(
    graph_key: &CoordinatorGraphKey,
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    top_k: usize,
) -> bool {
    coordinator_python_capture_enabled()
        && graph_key.shape == CoordinatorGraphShape::CoordDense
        && rows > 0
        && rows <= graph_key.row_bucket.row_capacity
        && hidden_dim > 0
        && vocab > 0
        && top_k > 0
        && top_k <= vocab
        && top_k <= GLMRT_CUDA_SAMPLE_TOPK_MAX_K
}

// MTP exposes one authoritative decode row plus one to seven useful draft
// rows.  The Triton logits kernel uses a 16-row tile for every one of those
// shapes, so compiling a separate exact-row graph only adds a roughly 0.5 s
// capture/JIT boundary without reducing the launched work.  Keep M=1 decode
// scalar, but use one stable M=8 terminal graph for the complete M=2..8 MTP
// envelope and ignore its inactive suffix outputs.
const TRITON_LM_HEAD_MTP_EXECUTION_ROWS: usize = 8;

fn triton_lm_head_execution_rows(rows: usize) -> usize {
    if (2..=TRITON_LM_HEAD_MTP_EXECUTION_ROWS).contains(&rows) {
        TRITON_LM_HEAD_MTP_EXECUTION_ROWS
    } else {
        rows
    }
}

fn triton_lm_head_sampler_logits_bytes(
    graph_key: &CoordinatorGraphKey,
    vocab: usize,
    label: &str,
) -> Result<usize> {
    graph_key
        .row_bucket
        .row_capacity
        .checked_mul(vocab)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .with_context(|| format!("{label} Triton logits buffer bytes overflow usize"))
}

fn triton_lm_head_sampler_candidate_layout(
    graph_key: &CoordinatorGraphKey,
    vocab: usize,
    top_k: usize,
    label: &str,
) -> Result<(usize, usize, usize, usize)> {
    let vocab_blocks = vocab.div_ceil(1024);
    let candidate_values = graph_key
        .row_bucket
        .row_capacity
        .checked_mul(vocab_blocks)
        .and_then(|values| values.checked_mul(top_k))
        .with_context(|| format!("{label} Triton candidate value count overflow usize"))?;
    let score_bytes = candidate_values
        .checked_mul(std::mem::size_of::<f32>())
        .with_context(|| format!("{label} Triton candidate score bytes overflow usize"))?;
    let index_bytes = candidate_values
        .checked_mul(std::mem::size_of::<u32>())
        .with_context(|| format!("{label} Triton candidate index bytes overflow usize"))?;
    let total_bytes = score_bytes
        .checked_add(index_bytes)
        .with_context(|| format!("{label} Triton candidate buffer bytes overflow usize"))?;
    Ok((vocab_blocks, score_bytes, index_bytes, total_bytes))
}

#[allow(clippy::too_many_arguments)]
fn triton_lm_head_sampler_graph_signature(
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    hidden_buffer: GlmrtDeviceBuffer,
    lm_head_buffer: GlmrtDeviceBuffer,
    random_buffer: GlmrtDeviceBuffer,
) -> CoordinatorCudaGraphSignature {
    CoordinatorCudaGraphSignature::triton_lm_head_sample_topk_topp_bf16(
        rows,
        hidden_dim,
        vocab,
        temperature,
        top_k,
        top_p,
        triton_lm_head_sampler_buffer_identity(hidden_buffer, lm_head_buffer, random_buffer),
    )
}

fn triton_lm_head_sampler_buffer_identity(
    hidden_buffer: GlmrtDeviceBuffer,
    lm_head_buffer: GlmrtDeviceBuffer,
    random_buffer: GlmrtDeviceBuffer,
) -> usize {
    [hidden_buffer, lm_head_buffer, random_buffer].iter().fold(
        0xa076_1d64_78bd_642f_usize,
        |acc, buffer| {
            acc.rotate_left(9)
                ^ (buffer.ptr as usize)
                ^ buffer.bytes.rotate_left(6)
                ^ ((buffer.device_id as usize) << 1)
        },
    )
}

fn device_buffer_slice(
    buffer: GlmrtDeviceBuffer,
    offset_bytes: usize,
    bytes: usize,
    label: &str,
) -> Result<GlmrtDeviceBuffer> {
    let end = offset_bytes
        .checked_add(bytes)
        .with_context(|| format!("{label} device buffer slice end overflows usize"))?;
    if end > buffer.bytes {
        anyhow::bail!(
            "{label} device buffer slice [{offset_bytes}, {end}) exceeds buffer bytes {}",
            buffer.bytes
        );
    }
    let ptr = unsafe { (buffer.ptr as *mut u8).add(offset_bytes).cast::<c_void>() };
    Ok(GlmrtDeviceBuffer {
        ptr,
        bytes,
        device_id: buffer.device_id,
        flags: buffer.flags,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_lm_head_argmax_bf16_preloaded_resident_weight_device_input_graph_slot(
    graph_key: &CoordinatorGraphKey,
    lm_head_name: &str,
    hidden_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    view: LmHeadResidentView,
) -> Result<LogitsArgmaxOutput> {
    let (index_bytes, score_bytes) = lm_head_output_bytes(
        rows,
        "CUDA BF16 preloaded resident lm_head device-input graph-slot argmax",
    )?;
    let (graph_index_bytes, graph_score_bytes) = lm_head_graph_output_bytes(
        graph_key,
        "CUDA BF16 preloaded resident lm_head device-input graph-slot argmax",
    )?;
    let lm_head_buffer = preloaded_resident_weight_device_buffer_view(
        lm_head_name,
        view.full_bytes,
        view.offset_bytes,
        view.view_bytes,
    )?;
    if hidden_buffer.device_id != lm_head_buffer.device_id {
        anyhow::bail!(
            "CUDA BF16 preloaded resident lm_head device-input argmax buffers are on different devices: hidden={} lm_head={}",
            hidden_buffer.device_id,
            lm_head_buffer.device_id
        );
    }
    let signature = CoordinatorCudaGraphSignature::lm_head_argmax_bf16(
        graph_key.row_bucket.row_capacity,
        hidden_dim,
        vocab,
    );

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let index_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            graph_index_bytes,
            "BF16 preloaded resident lm_head device-input argmax indices",
        )?;
        let score_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            graph_score_bytes,
            "BF16 preloaded resident lm_head device-input argmax scores",
        )?;
        capture_or_update_terminal_lm_head_argmax_bf16_graph(
            library,
            slot,
            signature,
            hidden_buffer,
            lm_head_buffer,
            index_buffer,
            score_buffer,
            rows,
            hidden_dim,
            vocab,
            "BF16 preloaded resident lm_head device-input argmax",
        )?;
        COORDINATOR_LM_HEAD_READBACK_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            scratch.argmax_index.resize(index_bytes, 0);
            scratch.argmax_score.resize(score_bytes, 0);
            unsafe {
                library
                    .copy_d2h_async(&mut scratch.argmax_index, index_buffer, cuda_stream)
                    .context("async copying BF16 preloaded resident lm_head device-input graph argmax indices to host")?;
                library
                    .copy_d2h_async(&mut scratch.argmax_score, score_buffer, cuda_stream)
                    .context("async copying BF16 preloaded resident lm_head device-input graph argmax scores to host")?;
                library.cuda_stream_synchronize(cuda_stream).context(
                    "synchronizing BF16 preloaded resident lm_head device-input argmax graph slot stream",
                )?;
            }
            Ok(LogitsArgmaxOutput {
                indices: u32_vec_from_bytes(&scratch.argmax_index)?
                    .into_iter()
                    .map(|value| value as usize)
                    .collect(),
                scores: f32_vec_from_bytes(&scratch.argmax_score)?,
                backend: CUDA_REFERENCE_LM_HEAD_ARGMAX_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
            })
        })
    })
}

pub(in crate::commands::real_full) fn cuda_lm_head_argmax_bf16_preloaded_resident_weight_device_input(
    lm_head_name: &str,
    hidden_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    view: LmHeadResidentView,
) -> Result<LogitsArgmaxOutput> {
    if let Some(graph_key) = coord_terminal_lm_head_graph_key(rows)? {
        match cuda_lm_head_argmax_bf16_preloaded_resident_weight_device_input_graph_slot(
            &graph_key,
            lm_head_name,
            hidden_buffer,
            rows,
            hidden_dim,
            vocab,
            view,
        ) {
            Ok(output) => return Ok(output),
            Err(_error) => {
                return cuda_lm_head_argmax_bf16_preloaded_resident_weight_device_input_direct(
                    lm_head_name,
                    hidden_buffer,
                    rows,
                    hidden_dim,
                    vocab,
                    view,
                );
            }
        }
    }
    cuda_lm_head_argmax_bf16_preloaded_resident_weight_device_input_direct(
        lm_head_name,
        hidden_buffer,
        rows,
        hidden_dim,
        vocab,
        view,
    )
}

#[allow(clippy::too_many_arguments)]
fn cuda_lm_head_argmax_bf16_preloaded_resident_weight_device_input_direct(
    lm_head_name: &str,
    hidden_buffer: GlmrtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    view: LmHeadResidentView,
) -> Result<LogitsArgmaxOutput> {
    let library = cuda_native_library()?;
    let index_bytes = rows.checked_mul(std::mem::size_of::<u32>()).context(
        "CUDA BF16 preloaded resident lm_head device-input argmax index bytes overflow usize",
    )?;
    let score_bytes = rows.checked_mul(std::mem::size_of::<f32>()).context(
        "CUDA BF16 preloaded resident lm_head device-input argmax score bytes overflow usize",
    )?;
    let lm_head_buffer = preloaded_resident_weight_device_buffer_view(
        lm_head_name,
        view.full_bytes,
        view.offset_bytes,
        view.view_bytes,
    )?;
    if hidden_buffer.device_id != lm_head_buffer.device_id {
        anyhow::bail!(
            "CUDA BF16 preloaded resident lm_head device-input argmax buffers are on different devices: hidden={} lm_head={}",
            hidden_buffer.device_id,
            lm_head_buffer.device_id
        );
    }
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let index_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        index_bytes,
        "BF16 preloaded resident lm_head device-input argmax indices",
    )?;
    let score_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        score_bytes,
        "BF16 preloaded resident lm_head device-input argmax scores",
    )?;

    library
        .cuda_lm_head_argmax_bf16(
            hidden_buffer,
            lm_head_buffer,
            index_buffer,
            score_buffer,
            rows,
            hidden_dim,
            vocab,
        )
        .context("executing CUDA BF16 preloaded resident lm_head device-input argmax")?;
    read_lm_head_device_input_argmax_output(
        library,
        index_buffer,
        index_bytes,
        score_buffer,
        score_bytes,
    )
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_lm_head_sample_topk_topp_bf16(
    hidden_bf16: &[u8],
    lm_head_bf16: &[u8],
    random_uniforms: &[f32],
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Result<LogitsSampleTopKToppOutput> {
    let library = cuda_native_library()?;
    let random_bytes = std::mem::size_of_val(random_uniforms);
    let index_bytes = rows
        .checked_mul(std::mem::size_of::<u32>())
        .context("CUDA BF16 lm_head sampler index bytes overflow usize")?;
    let score_bytes = rows
        .checked_mul(std::mem::size_of::<f32>())
        .context("CUDA BF16 lm_head sampler score bytes overflow usize")?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let hidden_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        hidden_bf16.len(),
        "BF16 lm_head sampler hidden",
    )?;
    let lm_head_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::B,
        lm_head_bf16.len(),
        "BF16 lm_head sampler weights",
    )?;
    let random_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        random_bytes,
        "BF16 lm_head sampler random uniforms",
    )?;
    let index_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        index_bytes,
        "BF16 lm_head sampler indices",
    )?;
    let score_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::E,
        score_bytes,
        "BF16 lm_head sampler scores",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            hidden_bf16,
            "BF16 lm_head sampler hidden",
        )
        .context("copying BF16 lm_head sampler hidden to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::B,
            lm_head_bf16,
            "BF16 lm_head sampler weights",
        )
        .context("copying BF16 lm_head sampler weights to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::C,
            f32_bytes(random_uniforms),
            "BF16 lm_head sampler random uniforms",
        )
        .context("copying BF16 lm_head sampler random uniforms to device")?;
    library
        .cuda_lm_head_sample_topk_topp_bf16(
            hidden_buffer,
            lm_head_buffer,
            random_buffer,
            index_buffer,
            score_buffer,
            rows,
            hidden_dim,
            vocab,
            temperature,
            top_k,
            top_p,
        )
        .context("executing CUDA BF16 lm_head top-k/top-p sampler")?;
    let mut index_out = vec![0_u8; index_bytes];
    let mut score_out = vec![0_u8; score_bytes];
    library
        .copy_d2h(&mut index_out, index_buffer)
        .context("copying BF16 lm_head sampler indices to host")?;
    library
        .copy_d2h(&mut score_out, score_buffer)
        .context("copying BF16 lm_head sampler scores to host")?;

    Ok(LogitsSampleTopKToppOutput {
        indices: u32_vec_from_bytes(&index_out)?
            .into_iter()
            .map(|value| value as usize)
            .collect(),
        scores: f32_vec_from_bytes(&score_out)?,
        backend: CUDA_REFERENCE_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_BACKEND,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_lm_head_sample_topk_topp_bf16_resident_weight(
    lm_head_name: &str,
    hidden_bf16: &[u8],
    lm_head_bf16: &[u8],
    random_uniforms: &[f32],
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Result<LogitsSampleTopKToppOutput> {
    let library = cuda_native_library()?;
    let random_bytes = std::mem::size_of_val(random_uniforms);
    let index_bytes = rows
        .checked_mul(std::mem::size_of::<u32>())
        .context("CUDA BF16 resident lm_head sampler index bytes overflow usize")?;
    let score_bytes = rows
        .checked_mul(std::mem::size_of::<f32>())
        .context("CUDA BF16 resident lm_head sampler score bytes overflow usize")?;
    let lm_head_buffer = resident_weight_buffer_from_registry(
        lm_head_name,
        lm_head_bf16,
        "BF16 resident lm_head sampler weights",
    )?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let hidden_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        hidden_bf16.len(),
        "BF16 resident lm_head sampler hidden",
    )?;
    let random_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        random_bytes,
        "BF16 resident lm_head sampler random uniforms",
    )?;
    let index_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        index_bytes,
        "BF16 resident lm_head sampler indices",
    )?;
    let score_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::E,
        score_bytes,
        "BF16 resident lm_head sampler scores",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            hidden_bf16,
            "BF16 resident lm_head sampler hidden",
        )
        .context("copying BF16 resident lm_head sampler hidden to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::C,
            f32_bytes(random_uniforms),
            "BF16 resident lm_head sampler random uniforms",
        )
        .context("copying BF16 resident lm_head sampler random uniforms to device")?;
    library
        .cuda_lm_head_sample_topk_topp_bf16(
            hidden_buffer,
            lm_head_buffer,
            random_buffer,
            index_buffer,
            score_buffer,
            rows,
            hidden_dim,
            vocab,
            temperature,
            top_k,
            top_p,
        )
        .context("executing CUDA BF16 resident lm_head top-k/top-p sampler")?;
    let mut index_out = vec![0_u8; index_bytes];
    let mut score_out = vec![0_u8; score_bytes];
    library
        .copy_d2h(&mut index_out, index_buffer)
        .context("copying BF16 resident lm_head sampler indices to host")?;
    library
        .copy_d2h(&mut score_out, score_buffer)
        .context("copying BF16 resident lm_head sampler scores to host")?;

    Ok(LogitsSampleTopKToppOutput {
        indices: u32_vec_from_bytes(&index_out)?
            .into_iter()
            .map(|value| value as usize)
            .collect(),
        scores: f32_vec_from_bytes(&score_out)?,
        backend: CUDA_REFERENCE_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_RESIDENT_WEIGHT_BACKEND,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_lm_head_sample_topk_topp_bf16_preloaded_resident_weight(
    lm_head_name: &str,
    hidden_bf16: &[u8],
    random_uniforms: &[f32],
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    view: LmHeadResidentView,
) -> Result<LogitsSampleTopKToppOutput> {
    let library = cuda_native_library()?;
    let random_bytes = std::mem::size_of_val(random_uniforms);
    let index_bytes = rows
        .checked_mul(std::mem::size_of::<u32>())
        .context("CUDA BF16 preloaded resident lm_head sampler index bytes overflow usize")?;
    let score_bytes = rows
        .checked_mul(std::mem::size_of::<f32>())
        .context("CUDA BF16 preloaded resident lm_head sampler score bytes overflow usize")?;
    let lm_head_buffer = preloaded_resident_weight_device_buffer_view(
        lm_head_name,
        view.full_bytes,
        view.offset_bytes,
        view.view_bytes,
    )?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let hidden_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        hidden_bf16.len(),
        "BF16 preloaded resident lm_head sampler hidden",
    )?;
    let random_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        random_bytes,
        "BF16 preloaded resident lm_head sampler random uniforms",
    )?;
    let index_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        index_bytes,
        "BF16 preloaded resident lm_head sampler indices",
    )?;
    let score_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::E,
        score_bytes,
        "BF16 preloaded resident lm_head sampler scores",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            hidden_bf16,
            "BF16 preloaded resident lm_head sampler hidden",
        )
        .context("copying BF16 preloaded resident lm_head sampler hidden to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::C,
            f32_bytes(random_uniforms),
            "BF16 preloaded resident lm_head sampler random uniforms",
        )
        .context("copying BF16 preloaded resident lm_head sampler random uniforms to device")?;
    library
        .cuda_lm_head_sample_topk_topp_bf16(
            hidden_buffer,
            lm_head_buffer,
            random_buffer,
            index_buffer,
            score_buffer,
            rows,
            hidden_dim,
            vocab,
            temperature,
            top_k,
            top_p,
        )
        .context("executing CUDA BF16 preloaded resident lm_head top-k/top-p sampler")?;
    let mut index_out = vec![0_u8; index_bytes];
    let mut score_out = vec![0_u8; score_bytes];
    library
        .copy_d2h(&mut index_out, index_buffer)
        .context("copying BF16 preloaded resident lm_head sampler indices to host")?;
    library
        .copy_d2h(&mut score_out, score_buffer)
        .context("copying BF16 preloaded resident lm_head sampler scores to host")?;

    Ok(LogitsSampleTopKToppOutput {
        indices: u32_vec_from_bytes(&index_out)?
            .into_iter()
            .map(|value| value as usize)
            .collect(),
        scores: f32_vec_from_bytes(&score_out)?,
        backend: CUDA_REFERENCE_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_lm_head_sample_topk_topp_bf16_preloaded_resident_weight_device_input_graph_slot(
    graph_key: &CoordinatorGraphKey,
    lm_head_name: &str,
    hidden_buffer: GlmrtDeviceBuffer,
    random_uniforms: &[f32],
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    view: LmHeadResidentView,
) -> Result<LogitsSampleTopKToppOutput> {
    let random_bytes = std::mem::size_of_val(random_uniforms);
    let graph_random_bytes = graph_key
        .row_bucket
        .row_capacity
        .checked_mul(std::mem::size_of::<f32>())
        .context(
            "CUDA BF16 preloaded resident lm_head device-input graph-slot sampler random bytes overflow usize",
        )?;
    if random_bytes > graph_random_bytes {
        anyhow::bail!(
            "CUDA BF16 preloaded resident lm_head device-input graph-slot sampler random bytes {} exceed bucket capacity {}",
            random_bytes,
            graph_random_bytes
        );
    }
    let (index_bytes, score_bytes) = lm_head_output_bytes(
        rows,
        "CUDA BF16 preloaded resident lm_head device-input graph-slot sampler",
    )?;
    let (graph_index_bytes, graph_score_bytes) = lm_head_graph_output_bytes(
        graph_key,
        "CUDA BF16 preloaded resident lm_head device-input graph-slot sampler",
    )?;
    let lm_head_buffer = preloaded_resident_weight_device_buffer_view(
        lm_head_name,
        view.full_bytes,
        view.offset_bytes,
        view.view_bytes,
    )?;
    if hidden_buffer.device_id != lm_head_buffer.device_id {
        anyhow::bail!(
            "CUDA BF16 preloaded resident lm_head device-input sampler buffers are on different devices: hidden={} lm_head={}",
            hidden_buffer.device_id,
            lm_head_buffer.device_id
        );
    }
    if triton_lm_head_sampler_supported(graph_key, rows, hidden_dim, vocab, top_k) {
        return cuda_lm_head_argmax_sample_topk_topp_bf16_preloaded_resident_weight_device_input_triton_graph_slot(
            graph_key,
            lm_head_name,
            hidden_buffer,
            random_uniforms,
            rows,
            hidden_dim,
            vocab,
            temperature,
            top_k,
            top_p,
            view,
            true,
        )
        .map(|output| output.sampler);
    }

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let random_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            graph_random_bytes,
            "BF16 preloaded resident lm_head device-input sampler random uniforms",
        )?;
        let index_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            graph_index_bytes,
            "BF16 preloaded resident lm_head device-input sampler indices",
        )?;
        let score_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::E,
            graph_score_bytes,
            "BF16 preloaded resident lm_head device-input sampler scores",
        )?;

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::C,
                f32_bytes(random_uniforms),
                "BF16 preloaded resident lm_head device-input sampler random uniforms",
                cuda_stream,
            )
            .context(
                "async copying BF16 preloaded resident lm_head device-input sampler random uniforms to device",
            )?;
        let backend = if triton_lm_head_sampler_supported(graph_key, rows, hidden_dim, vocab, top_k)
        {
            let logits_bytes = triton_lm_head_sampler_logits_bytes(
                graph_key,
                vocab,
                "BF16 preloaded resident lm_head device-input top-k/top-p sampler",
            )?;
            let (_vocab_blocks, candidate_score_bytes, candidate_index_bytes, candidate_bytes) =
                triton_lm_head_sampler_candidate_layout(
                    graph_key,
                    vocab,
                    top_k,
                    "BF16 preloaded resident lm_head device-input top-k/top-p sampler",
                )?;
            let logits_buffer = slot.buffer(
                library,
                CoordinatorCudaScratchSlot::B,
                logits_bytes,
                "Triton BF16 lm_head sampler logits",
            )?;
            let candidate_buffer = slot.buffer(
                library,
                CoordinatorCudaScratchSlot::F,
                candidate_bytes,
                "Triton BF16 lm_head sampler candidates",
            )?;
            let candidate_score_buffer = device_buffer_slice(
                candidate_buffer,
                0,
                candidate_score_bytes,
                "Triton BF16 lm_head sampler candidate scores",
            )?;
            let candidate_index_buffer = device_buffer_slice(
                candidate_buffer,
                candidate_score_bytes,
                candidate_index_bytes,
                "Triton BF16 lm_head sampler candidate indices",
            )?;
            let signature = triton_lm_head_sampler_graph_signature(
                rows,
                hidden_dim,
                vocab,
                temperature,
                top_k,
                top_p,
                hidden_buffer,
                lm_head_buffer,
                random_buffer,
            );
            capture_or_update_terminal_triton_lm_head_sample_topk_topp_bf16_graph(
                library,
                slot,
                signature,
                hidden_buffer,
                lm_head_buffer,
                random_buffer,
                logits_buffer,
                candidate_score_buffer,
                candidate_index_buffer,
                index_buffer,
                score_buffer,
                index_buffer,
                score_buffer,
                rows,
                hidden_dim,
                vocab,
                temperature,
                top_k,
                top_p,
                "BF16 preloaded resident lm_head device-input Triton top-k/top-p sampler",
            )?;
            TRITON_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        } else {
            let signature = CoordinatorCudaGraphSignature::lm_head_sample_topk_topp_bf16(
                graph_key.row_bucket.row_capacity,
                hidden_dim,
                vocab,
                temperature,
                top_k,
                top_p,
            );
            capture_or_update_terminal_lm_head_sample_topk_topp_bf16_graph(
                library,
                slot,
                signature,
                hidden_buffer,
                lm_head_buffer,
                random_buffer,
                index_buffer,
                score_buffer,
                rows,
                hidden_dim,
                vocab,
                temperature,
                top_k,
                top_p,
                "BF16 preloaded resident lm_head device-input top-k/top-p sampler",
            )?;
            CUDA_REFERENCE_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        };
        COORDINATOR_LM_HEAD_READBACK_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            scratch.sample_index.resize(index_bytes, 0);
            scratch.sample_score.resize(score_bytes, 0);
            unsafe {
                library
                    .copy_d2h_async(&mut scratch.sample_index, index_buffer, cuda_stream)
                    .context("async copying BF16 preloaded resident lm_head device-input graph sampler indices to host")?;
                library
                    .copy_d2h_async(&mut scratch.sample_score, score_buffer, cuda_stream)
                    .context("async copying BF16 preloaded resident lm_head device-input graph sampler scores to host")?;
                library.cuda_stream_synchronize(cuda_stream).context(
                    "synchronizing BF16 preloaded resident lm_head device-input sampler graph slot stream",
                )?;
            }
            Ok(LogitsSampleTopKToppOutput {
                indices: u32_vec_from_bytes(&scratch.sample_index)?
                    .into_iter()
                    .map(|value| value as usize)
                    .collect(),
                scores: f32_vec_from_bytes(&scratch.sample_score)?,
                backend,
            })
        })
    })
}

#[allow(clippy::too_many_arguments)]
fn cuda_lm_head_argmax_sample_topk_topp_bf16_preloaded_resident_weight_device_input_triton_graph_slot(
    graph_key: &CoordinatorGraphKey,
    lm_head_name: &str,
    hidden_buffer: GlmrtDeviceBuffer,
    random_uniforms: &[f32],
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    view: LmHeadResidentView,
    launch_graph: bool,
) -> Result<LogitsArgmaxSampleTopKToppOutput> {
    anyhow::ensure!(
        triton_lm_head_sampler_supported(graph_key, rows, hidden_dim, vocab, top_k),
        "Triton combined lm_head argmax+sampler graph does not support the requested shape"
    );
    let execution_rows = triton_lm_head_execution_rows(rows);
    anyhow::ensure!(
        execution_rows <= graph_key.row_bucket.row_capacity,
        "Triton combined lm_head execution rows {execution_rows} exceed graph capacity {}",
        graph_key.row_bucket.row_capacity
    );
    let mut padded_random_uniforms = [0.0_f32; TRITON_LM_HEAD_MTP_EXECUTION_ROWS];
    let execution_random_uniforms = if execution_rows == rows {
        random_uniforms
    } else {
        padded_random_uniforms.fill(random_uniforms.last().copied().unwrap_or(0.5));
        padded_random_uniforms[..rows].copy_from_slice(random_uniforms);
        &padded_random_uniforms[..execution_rows]
    };
    let random_bytes = std::mem::size_of_val(execution_random_uniforms);
    let graph_random_bytes = graph_key
        .row_bucket
        .row_capacity
        .checked_mul(std::mem::size_of::<f32>())
        .context("Triton combined lm_head sampler random bytes overflow usize")?;
    anyhow::ensure!(
        random_bytes <= graph_random_bytes,
        "Triton combined lm_head sampler random bytes {random_bytes} exceed bucket capacity {graph_random_bytes}"
    );
    let (index_bytes, score_bytes) =
        lm_head_output_bytes(rows, "Triton combined lm_head argmax+sampler")?;
    let (graph_index_bytes, graph_score_bytes) =
        lm_head_graph_output_bytes(graph_key, "Triton combined lm_head argmax+sampler")?;
    let graph_argmax_bytes = graph_index_bytes
        .checked_add(graph_score_bytes)
        .context("Triton combined lm_head argmax output bytes overflow usize")?;
    let lm_head_buffer = preloaded_resident_weight_device_buffer_view(
        lm_head_name,
        view.full_bytes,
        view.offset_bytes,
        view.view_bytes,
    )?;
    anyhow::ensure!(
        hidden_buffer.device_id == lm_head_buffer.device_id,
        "Triton combined lm_head buffers are on different devices: hidden={} lm_head={}",
        hidden_buffer.device_id,
        lm_head_buffer.device_id
    );
    let active_hidden_bytes = rows
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("Triton combined lm_head active hidden bytes overflow usize")?;
    let graph_hidden_bytes = graph_key
        .row_bucket
        .row_capacity
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("Triton combined lm_head staged hidden bytes overflow usize")?;

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let staged_hidden_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            graph_hidden_bytes,
            "Triton combined lm_head staged hidden",
        )?;
        let random_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            graph_random_bytes,
            "Triton combined lm_head random uniforms",
        )?;
        let sample_index_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            graph_index_bytes,
            "Triton combined lm_head sample indices",
        )?;
        let sample_score_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::E,
            graph_score_bytes,
            "Triton combined lm_head sample scores",
        )?;
        let logits_bytes = triton_lm_head_sampler_logits_bytes(
            graph_key,
            vocab,
            "Triton combined lm_head argmax+sampler",
        )?;
        let (_vocab_blocks, candidate_score_bytes, candidate_index_bytes, candidate_bytes) =
            triton_lm_head_sampler_candidate_layout(
                graph_key,
                vocab,
                top_k,
                "Triton combined lm_head argmax+sampler",
            )?;
        let logits_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::B,
            logits_bytes,
            "Triton combined lm_head logits",
        )?;
        let candidate_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::F,
            candidate_bytes,
            "Triton combined lm_head candidates",
        )?;
        let candidate_score_buffer = device_buffer_slice(
            candidate_buffer,
            0,
            candidate_score_bytes,
            "Triton combined lm_head candidate scores",
        )?;
        let candidate_index_buffer = device_buffer_slice(
            candidate_buffer,
            candidate_score_bytes,
            candidate_index_bytes,
            "Triton combined lm_head candidate indices",
        )?;
        let argmax_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::G,
            graph_argmax_bytes,
            "Triton combined lm_head argmax outputs",
        )?;
        let argmax_index_buffer = device_buffer_slice(
            argmax_buffer,
            0,
            graph_index_bytes,
            "Triton combined lm_head argmax indices",
        )?;
        let argmax_score_buffer = device_buffer_slice(
            argmax_buffer,
            graph_index_bytes,
            graph_score_bytes,
            "Triton combined lm_head argmax scores",
        )?;

        unsafe {
            library
                .copy_d2d_async(
                    staged_hidden_buffer,
                    hidden_buffer,
                    active_hidden_bytes,
                    cuda_stream,
                )
                .context("staging Triton combined lm_head hidden rows")?;
        }
        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::C,
                f32_bytes(execution_random_uniforms),
                "Triton combined lm_head random uniforms",
                cuda_stream,
            )
            .context("copying Triton combined lm_head random uniforms to device")?;
        if launch_graph {
            let signature = triton_lm_head_sampler_graph_signature(
                execution_rows,
                hidden_dim,
                vocab,
                temperature,
                top_k,
                top_p,
                staged_hidden_buffer,
                lm_head_buffer,
                random_buffer,
            );
            capture_or_update_terminal_triton_lm_head_sample_topk_topp_bf16_graph(
                library,
                slot,
                signature,
                staged_hidden_buffer,
                lm_head_buffer,
                random_buffer,
                logits_buffer,
                candidate_score_buffer,
                candidate_index_buffer,
                argmax_index_buffer,
                argmax_score_buffer,
                sample_index_buffer,
                sample_score_buffer,
                execution_rows,
                hidden_dim,
                vocab,
                temperature,
                top_k,
                top_p,
                "Triton combined lm_head argmax+top-k/top-p sampler",
            )?;
        } else {
            launch_triton_lm_head_sample_topk_topp_bf16_graph_capture(
                cuda_stream,
                staged_hidden_buffer,
                lm_head_buffer,
                random_buffer,
                logits_buffer,
                candidate_score_buffer,
                candidate_index_buffer,
                argmax_index_buffer,
                argmax_score_buffer,
                sample_index_buffer,
                sample_score_buffer,
                execution_rows,
                hidden_dim,
                vocab,
                temperature,
                top_k,
                top_p,
            )
            .context("launching direct Triton combined lm_head sampler")?;
        }

        COORDINATOR_LM_HEAD_READBACK_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            scratch.argmax_index.resize(index_bytes, 0);
            scratch.argmax_score.resize(score_bytes, 0);
            scratch.sample_index.resize(index_bytes, 0);
            scratch.sample_score.resize(score_bytes, 0);
            unsafe {
                library.copy_d2h_async(
                    &mut scratch.argmax_index,
                    argmax_index_buffer,
                    cuda_stream,
                )?;
                library.copy_d2h_async(
                    &mut scratch.argmax_score,
                    argmax_score_buffer,
                    cuda_stream,
                )?;
                library.copy_d2h_async(
                    &mut scratch.sample_index,
                    sample_index_buffer,
                    cuda_stream,
                )?;
                library.copy_d2h_async(
                    &mut scratch.sample_score,
                    sample_score_buffer,
                    cuda_stream,
                )?;
                library
                    .cuda_stream_synchronize(cuda_stream)
                    .context("synchronizing Triton combined lm_head sampler")?;
            }
            Ok(LogitsArgmaxSampleTopKToppOutput {
                argmax: LogitsArgmaxOutput {
                    indices: u32_vec_from_bytes(&scratch.argmax_index)?
                        .into_iter()
                        .map(|value| value as usize)
                        .collect(),
                    scores: f32_vec_from_bytes(&scratch.argmax_score)?,
                    backend: CUDA_REFERENCE_LM_HEAD_ARGMAX_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
                },
                sampler: LogitsSampleTopKToppOutput {
                    indices: u32_vec_from_bytes(&scratch.sample_index)?
                        .into_iter()
                        .map(|value| value as usize)
                        .collect(),
                    scores: f32_vec_from_bytes(&scratch.sample_score)?,
                    backend: TRITON_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
                },
            })
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_lm_head_sample_topk_topp_bf16_preloaded_resident_weight_device_input(
    lm_head_name: &str,
    hidden_buffer: GlmrtDeviceBuffer,
    random_uniforms: &[f32],
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    view: LmHeadResidentView,
) -> Result<LogitsSampleTopKToppOutput> {
    if let Some(graph_key) = coord_terminal_lm_head_graph_key(rows)? {
        match cuda_lm_head_sample_topk_topp_bf16_preloaded_resident_weight_device_input_graph_slot(
            &graph_key,
            lm_head_name,
            hidden_buffer,
            random_uniforms,
            rows,
            hidden_dim,
            vocab,
            temperature,
            top_k,
            top_p,
            view,
        ) {
            Ok(output) => return Ok(output),
            Err(_error) => {}
        }
    }
    let library = cuda_native_library()?;
    let random_bytes = std::mem::size_of_val(random_uniforms);
    let index_bytes = rows.checked_mul(std::mem::size_of::<u32>()).context(
        "CUDA BF16 preloaded resident lm_head device-input sampler index bytes overflow usize",
    )?;
    let score_bytes = rows.checked_mul(std::mem::size_of::<f32>()).context(
        "CUDA BF16 preloaded resident lm_head device-input sampler score bytes overflow usize",
    )?;
    let lm_head_buffer = preloaded_resident_weight_device_buffer_view(
        lm_head_name,
        view.full_bytes,
        view.offset_bytes,
        view.view_bytes,
    )?;
    if hidden_buffer.device_id != lm_head_buffer.device_id {
        anyhow::bail!(
            "CUDA BF16 preloaded resident lm_head device-input sampler buffers are on different devices: hidden={} lm_head={}",
            hidden_buffer.device_id,
            lm_head_buffer.device_id
        );
    }
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let random_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        random_bytes,
        "BF16 preloaded resident lm_head device-input sampler random uniforms",
    )?;
    let index_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        index_bytes,
        "BF16 preloaded resident lm_head device-input sampler indices",
    )?;
    let score_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::E,
        score_bytes,
        "BF16 preloaded resident lm_head device-input sampler scores",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::C,
            f32_bytes(random_uniforms),
            "BF16 preloaded resident lm_head device-input sampler random uniforms",
        )
        .context(
            "copying BF16 preloaded resident lm_head device-input sampler random uniforms to device",
        )?;
    library
        .cuda_lm_head_sample_topk_topp_bf16(
            hidden_buffer,
            lm_head_buffer,
            random_buffer,
            index_buffer,
            score_buffer,
            rows,
            hidden_dim,
            vocab,
            temperature,
            top_k,
            top_p,
        )
        .context(
            "executing CUDA BF16 preloaded resident lm_head device-input top-k/top-p sampler",
        )?;
    read_lm_head_device_input_sampler_output(
        library,
        index_buffer,
        index_bytes,
        score_buffer,
        score_bytes,
    )
}

pub(in crate::commands::real_full) fn read_lm_head_device_input_argmax_output(
    library: &'static NativeLibrary,
    index_buffer: GlmrtDeviceBuffer,
    index_bytes: usize,
    score_buffer: GlmrtDeviceBuffer,
    score_bytes: usize,
) -> Result<LogitsArgmaxOutput> {
    COORDINATOR_LM_HEAD_READBACK_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        scratch.argmax_index.resize(index_bytes, 0);
        scratch.argmax_score.resize(score_bytes, 0);
        library
            .copy_d2h(&mut scratch.argmax_index, index_buffer)
            .context(
                "copying BF16 preloaded resident lm_head device-input argmax indices to host",
            )?;
        library
            .copy_d2h(&mut scratch.argmax_score, score_buffer)
            .context(
                "copying BF16 preloaded resident lm_head device-input argmax scores to host",
            )?;
        Ok(LogitsArgmaxOutput {
            indices: u32_vec_from_bytes(&scratch.argmax_index)?
                .into_iter()
                .map(|value| value as usize)
                .collect(),
            scores: f32_vec_from_bytes(&scratch.argmax_score)?,
            backend: CUDA_REFERENCE_LM_HEAD_ARGMAX_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
        })
    })
}

pub(in crate::commands::real_full) fn read_lm_head_device_input_sampler_output(
    library: &'static NativeLibrary,
    index_buffer: GlmrtDeviceBuffer,
    index_bytes: usize,
    score_buffer: GlmrtDeviceBuffer,
    score_bytes: usize,
) -> Result<LogitsSampleTopKToppOutput> {
    COORDINATOR_LM_HEAD_READBACK_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        scratch.sample_index.resize(index_bytes, 0);
        scratch.sample_score.resize(score_bytes, 0);
        library
            .copy_d2h(&mut scratch.sample_index, index_buffer)
            .context(
                "copying BF16 preloaded resident lm_head device-input sampler indices to host",
            )?;
        library
            .copy_d2h(&mut scratch.sample_score, score_buffer)
            .context(
                "copying BF16 preloaded resident lm_head device-input sampler scores to host",
            )?;
        Ok(LogitsSampleTopKToppOutput {
            indices: u32_vec_from_bytes(&scratch.sample_index)?
                .into_iter()
                .map(|value| value as usize)
                .collect(),
            scores: f32_vec_from_bytes(&scratch.sample_score)?,
            backend: CUDA_REFERENCE_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
        })
    })
}

#[allow(clippy::too_many_arguments)]
fn cuda_lm_head_argmax_sample_topk_topp_bf16_preloaded_resident_weight_device_input(
    lm_head_name: &str,
    hidden_buffer: GlmrtDeviceBuffer,
    random_uniforms: &[f32],
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    view: LmHeadResidentView,
    allow_graph: bool,
) -> Result<LogitsArgmaxSampleTopKToppOutput> {
    if let Some(graph_key) = coord_terminal_lm_head_graph_key(rows)? {
        if triton_lm_head_sampler_supported(&graph_key, rows, hidden_dim, vocab, top_k) {
            match cuda_lm_head_argmax_sample_topk_topp_bf16_preloaded_resident_weight_device_input_triton_graph_slot(
                &graph_key,
                lm_head_name,
                hidden_buffer,
                random_uniforms,
                rows,
                hidden_dim,
                vocab,
                temperature,
                top_k,
                top_p,
                view,
                allow_graph,
            ) {
                Ok(output) => return Ok(output),
                Err(error) => tracing::warn!(
                    error = %format!("{error:#}"),
                    graph = allow_graph,
                    "falling back from Triton combined lm_head sampler to scalar CUDA kernel"
                ),
            }
        }
    }
    let library = cuda_native_library()?;
    let random_bytes = std::mem::size_of_val(random_uniforms);
    let index_bytes = rows.checked_mul(std::mem::size_of::<u32>()).context(
        "CUDA BF16 preloaded resident lm_head device-input argmax+sampler index bytes overflow usize",
    )?;
    let score_bytes = rows.checked_mul(std::mem::size_of::<f32>()).context(
        "CUDA BF16 preloaded resident lm_head device-input argmax+sampler score bytes overflow usize",
    )?;
    let logits_bytes = rows
        .checked_mul(vocab)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context(
            "CUDA BF16 preloaded resident lm_head device-input argmax+sampler logits workspace bytes overflow usize",
        )?;
    let lm_head_buffer = preloaded_resident_weight_device_buffer_view(
        lm_head_name,
        view.full_bytes,
        view.offset_bytes,
        view.view_bytes,
    )?;
    if hidden_buffer.device_id != lm_head_buffer.device_id {
        anyhow::bail!(
            "CUDA BF16 preloaded resident lm_head device-input argmax+sampler buffers are on different devices: hidden={} lm_head={}",
            hidden_buffer.device_id,
            lm_head_buffer.device_id
        );
    }
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let logits_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::B,
        logits_bytes,
        "BF16 preloaded resident lm_head device-input argmax+sampler logits",
    )?;
    let random_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        random_bytes,
        "BF16 preloaded resident lm_head device-input argmax+sampler random uniforms",
    )?;
    let argmax_index_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        index_bytes,
        "BF16 preloaded resident lm_head device-input argmax indices",
    )?;
    let argmax_score_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::E,
        score_bytes,
        "BF16 preloaded resident lm_head device-input argmax scores",
    )?;
    let sample_index_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::F,
        index_bytes,
        "BF16 preloaded resident lm_head device-input sampler indices",
    )?;
    let sample_score_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::G,
        score_bytes,
        "BF16 preloaded resident lm_head device-input sampler scores",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::C,
            f32_bytes(random_uniforms),
            "BF16 preloaded resident lm_head device-input argmax+sampler random uniforms",
        )
        .context(
            "copying BF16 preloaded resident lm_head device-input argmax+sampler random uniforms to device",
        )?;
    library
        .cuda_lm_head_argmax_sample_topk_topp_bf16_staged(
            hidden_buffer,
            lm_head_buffer,
            random_buffer,
            argmax_index_buffer,
            argmax_score_buffer,
            sample_index_buffer,
            sample_score_buffer,
            logits_buffer,
            rows,
            hidden_dim,
            vocab,
            temperature,
            top_k,
            top_p,
        )
        .context(
            "executing staged CUDA BF16 preloaded resident lm_head device-input argmax+top-k/top-p sampler",
        )?;

    COORDINATOR_LM_HEAD_READBACK_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        scratch.argmax_index.resize(index_bytes, 0);
        scratch.argmax_score.resize(score_bytes, 0);
        scratch.sample_index.resize(index_bytes, 0);
        scratch.sample_score.resize(score_bytes, 0);
        library
            .copy_d2h(&mut scratch.argmax_index, argmax_index_buffer)
            .context(
                "copying BF16 preloaded resident lm_head device-input staged argmax indices to host",
            )?;
        library
            .copy_d2h(&mut scratch.argmax_score, argmax_score_buffer)
            .context(
                "copying BF16 preloaded resident lm_head device-input staged argmax scores to host",
            )?;
        library
            .copy_d2h(&mut scratch.sample_index, sample_index_buffer)
            .context(
                "copying BF16 preloaded resident lm_head device-input staged sampler indices to host",
            )?;
        library
            .copy_d2h(&mut scratch.sample_score, sample_score_buffer)
            .context(
                "copying BF16 preloaded resident lm_head device-input staged sampler scores to host",
            )?;
        Ok(LogitsArgmaxSampleTopKToppOutput {
            argmax: LogitsArgmaxOutput {
                indices: u32_vec_from_bytes(&scratch.argmax_index)?
                    .into_iter()
                    .map(|value| value as usize)
                    .collect(),
                scores: f32_vec_from_bytes(&scratch.argmax_score)?,
                backend: CUDA_REFERENCE_LM_HEAD_ARGMAX_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
            },
            sampler: LogitsSampleTopKToppOutput {
                indices: u32_vec_from_bytes(&scratch.sample_index)?
                    .into_iter()
                    .map(|value| value as usize)
                    .collect(),
                scores: f32_vec_from_bytes(&scratch.sample_score)?,
                backend: CUDA_REFERENCE_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
            },
        })
    })
}

pub(in crate::commands::real_full) fn coord_terminal_lm_head_graph_key(
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
        .context("selecting Coord-Dense graph slot for terminal BF16 lm_head sampling")
}

#[cfg(test)]
mod tests {
    use super::triton_lm_head_execution_rows;

    #[test]
    fn triton_lm_head_uses_one_physical_mtp_bucket() {
        assert_eq!(triton_lm_head_execution_rows(0), 0);
        assert_eq!(triton_lm_head_execution_rows(1), 1);
        for rows in 2..=8 {
            assert_eq!(triton_lm_head_execution_rows(rows), 8);
        }
        assert_eq!(triton_lm_head_execution_rows(9), 9);
    }
}
