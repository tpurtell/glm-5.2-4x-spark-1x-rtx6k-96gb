use anyhow::{Context, Result};
use glmrt_core::{DType, TensorCatalog, TensorRole, GLM52_HIDDEN_SIZE, GLM52_MTP_LAYER_ID};
use glmrt_ffi::GLMRT_CUDA_SAMPLE_TOPK_MAX_K;

use super::coordinator_kernels::{
    concat_device_bf16_row_batches_async, concat_device_bf16_row_features, cuda_native_library,
    device_bf16_output_from_device_template_buffer, device_buffer_byte_view,
    linear_rows_bf16_preloaded_resident_weight_device_output,
    rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output, DeviceBf16Output,
};
use super::embedding::real_full_embedding_device_hidden_for_tokens;
use super::sampling::{
    score_real_lm_head_full_vocab_for_device_hidden,
    score_real_lm_head_full_vocab_for_device_hidden_rows,
    score_real_lm_head_full_vocab_for_device_hidden_rows_with_options,
    RealFullLmHeadSamplingOptions, RealLmHeadBatchScoreForHidden,
};

const MTP_EH_PROJ_WEIGHT: &str = "model.layers.78.eh_proj.weight";
const MTP_ENORM_WEIGHT: &str = "model.layers.78.enorm.weight";
const MTP_HNORM_WEIGHT: &str = "model.layers.78.hnorm.weight";
const TARGET_FINAL_NORM_WEIGHT: &str = "model.norm.weight";
pub(in crate::commands::real_full) const MTP_SHARED_HEAD_NORM_WEIGHT: &str =
    "model.layers.78.shared_head.norm.weight";
const MTP_RMSNORM_EPS: f32 = 1.0e-5;
const MTP_ENVELOPE_WIDTH: usize = GLM52_HIDDEN_SIZE * 2;

pub(in crate::commands::real_full) struct RealFullMtpDraftToken {
    pub(in crate::commands::real_full) token_id: usize,
    pub(in crate::commands::real_full) top_logit: f32,
    pub(in crate::commands::real_full) logits_evaluated: usize,
    pub(in crate::commands::real_full) argmax_backend: &'static str,
}

pub(in crate::commands::real_full) fn real_full_target_token_samples(
    catalog: &TensorCatalog,
    target_hidden: &DeviceBf16Output,
    suffix_rows: usize,
) -> Result<RealLmHeadBatchScoreForHidden> {
    let normalized = real_full_normalized_target_suffix(target_hidden, suffix_rows)?;
    score_real_lm_head_full_vocab_for_device_hidden_rows(catalog, &normalized)
        .context("scoring real-full target verification rows")
}

pub(in crate::commands::real_full) fn real_full_target_token_samples_with_options(
    catalog: &TensorCatalog,
    target_hidden: &DeviceBf16Output,
    suffix_rows: usize,
    sampler_options: RealFullLmHeadSamplingOptions,
    random_uniforms: &[f32],
) -> Result<RealLmHeadBatchScoreForHidden> {
    anyhow::ensure!(
        random_uniforms.len() == suffix_rows,
        "real-full target sampler received {} random uniforms for {suffix_rows} suffix rows",
        random_uniforms.len()
    );
    let normalized = real_full_normalized_target_suffix(target_hidden, suffix_rows)?;
    score_real_lm_head_full_vocab_for_device_hidden_rows_with_options(
        catalog,
        &normalized,
        sampler_options,
        random_uniforms,
        false,
    )
    .context("scoring sampled real-full target verification rows")
}

fn real_full_normalized_target_suffix(
    target_hidden: &DeviceBf16Output,
    suffix_rows: usize,
) -> Result<DeviceBf16Output> {
    anyhow::ensure!(
        suffix_rows > 0
            && suffix_rows <= target_hidden.rows
            && target_hidden.values_per_row == GLM52_HIDDEN_SIZE,
        "real-full target sampling suffix {} is invalid for hidden {}x{}",
        suffix_rows,
        target_hidden.rows,
        target_hidden.values_per_row
    );
    let row_bytes = GLM52_HIDDEN_SIZE * std::mem::size_of::<u16>();
    let suffix_bytes = suffix_rows * row_bytes;
    let offset_bytes = (target_hidden.rows - suffix_rows) * row_bytes;
    let view = device_buffer_byte_view(
        target_hidden.buffer(),
        offset_bytes,
        suffix_bytes,
        "real-full target sample suffix",
    )?;
    let hidden = device_bf16_output_from_device_template_buffer(
        view,
        suffix_rows,
        GLM52_HIDDEN_SIZE,
        "real-full target sample suffix",
    )?;
    let normalized = rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output(
        TARGET_FINAL_NORM_WEIGHT,
        hidden.buffer(),
        suffix_rows,
        GLM52_HIDDEN_SIZE,
        MTP_RMSNORM_EPS,
    )
    .context("normalizing real-full target verification rows")?;
    Ok(normalized)
}

pub(in crate::commands::real_full) fn prewarm_real_full_paired_target_token_sample_rows(
    catalog: &TensorCatalog,
    min_rows: usize,
    max_rows: usize,
) -> Result<()> {
    anyhow::ensure!(
        min_rows > 0 && min_rows <= max_rows,
        "invalid target-sample prewarm row range {min_rows}..={max_rows}"
    );
    let token_ids = vec![0; max_rows];
    let hidden = real_full_embedding_device_hidden_for_tokens(catalog, &token_ids)
        .context("creating target-sample prewarm hidden rows")?
        .context("target-sample prewarm requires device-resident embeddings")?;
    for rows in (min_rows..=max_rows).rev() {
        let rows_a = 16.min(rows - 2);
        let rows_b = rows - rows_a;
        let hidden_a = real_full_device_hidden_rows(&hidden.device_hidden, 0, rows_a)
            .context("slicing paired target-sample prewarm rows A")?;
        let hidden_b = real_full_device_hidden_rows(&hidden.device_hidden, rows_a, rows_b)
            .context("slicing paired target-sample prewarm rows B")?;
        real_full_target_token_samples_pair(catalog, &hidden_a, rows_a, &hidden_b, rows_b)
            .with_context(|| format!("prewarming paired target-sample graph for {rows} rows"))?;
    }
    Ok(())
}

pub(in crate::commands::real_full) fn prewarm_real_full_target_sampler_capacity(
    catalog: &TensorCatalog,
    max_rows: usize,
) -> Result<()> {
    anyhow::ensure!(
        max_rows > 0,
        "target sampler capacity prewarm requires nonzero rows"
    );
    let token_ids = vec![0; max_rows];
    let hidden = real_full_embedding_device_hidden_for_tokens(catalog, &token_ids)
        .context("creating target sampler capacity prewarm hidden rows")?
        .context("target sampler capacity prewarm requires device-resident embeddings")?;
    let options = RealFullLmHeadSamplingOptions {
        random_uniform: 0.5,
        temperature: 1.0,
        top_k: GLMRT_CUDA_SAMPLE_TOPK_MAX_K,
        top_p: 0.95,
    };
    for rows in [32, 16, 8, 1].into_iter().filter(|rows| *rows <= max_rows) {
        let target_hidden = real_full_device_hidden_rows(&hidden.device_hidden, 0, rows)
            .with_context(|| format!("slicing target sampler capacity rows={rows}"))?;
        let normalized = real_full_normalized_target_suffix(&target_hidden, rows)
            .with_context(|| format!("normalizing target sampler capacity rows={rows}"))?;
        let random_uniforms = vec![options.random_uniform; rows];
        score_real_lm_head_full_vocab_for_device_hidden_rows_with_options(
            catalog,
            &normalized,
            options,
            &random_uniforms,
            true,
        )
        .with_context(|| {
            format!(
                "prewarming target sampler capacity rows={rows} top_k={}",
                options.top_k
            )
        })?;
    }
    Ok(())
}

pub(in crate::commands::real_full) fn real_full_target_token_samples_pair(
    catalog: &TensorCatalog,
    target_hidden_a: &DeviceBf16Output,
    suffix_rows_a: usize,
    target_hidden_b: &DeviceBf16Output,
    suffix_rows_b: usize,
) -> Result<(RealLmHeadBatchScoreForHidden, RealLmHeadBatchScoreForHidden)> {
    let suffix_a = real_full_device_hidden_rows(
        target_hidden_a,
        target_hidden_a
            .rows
            .checked_sub(suffix_rows_a)
            .context("paired target sample A suffix exceeds retained rows")?,
        suffix_rows_a,
    )
    .context("slicing paired target sample A suffix")?;
    let suffix_b = real_full_device_hidden_rows(
        target_hidden_b,
        target_hidden_b
            .rows
            .checked_sub(suffix_rows_b)
            .context("paired target sample B suffix exceeds retained rows")?,
        suffix_rows_b,
    )
    .context("slicing paired target sample B suffix")?;
    let combined = concat_device_bf16_row_batches_async(
        &[&suffix_a, &suffix_b],
        "paired target sample hidden rows",
    )
    .context("concatenating paired target sample rows")?;
    let normalized = rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output(
        TARGET_FINAL_NORM_WEIGHT,
        combined.buffer(),
        combined.rows,
        GLM52_HIDDEN_SIZE,
        MTP_RMSNORM_EPS,
    )
    .context("normalizing paired target verification rows")?;
    let mut samples = score_real_lm_head_full_vocab_for_device_hidden_rows(catalog, &normalized)
        .context("scoring paired target verification rows")?;
    anyhow::ensure!(
        samples.top_token_ids.len() == suffix_rows_a + suffix_rows_b
            && samples.sampled_token_ids.len() == suffix_rows_a + suffix_rows_b,
        "paired target scoring returned top/sample rows {}/{} for {} expected rows",
        samples.top_token_ids.len(),
        samples.sampled_token_ids.len(),
        suffix_rows_a + suffix_rows_b,
    );
    let top_token_ids_b = samples.top_token_ids.split_off(suffix_rows_a);
    let sampled_token_ids_b = samples.sampled_token_ids.split_off(suffix_rows_a);
    let samples_b = RealLmHeadBatchScoreForHidden {
        vocab_size: samples.vocab_size,
        top_token_ids: top_token_ids_b,
        sampled_token_ids: sampled_token_ids_b,
        sample_top_k: samples.sample_top_k,
        sample_top_p: samples.sample_top_p,
        argmax_kernel_backend: samples.argmax_kernel_backend,
        sampler_kernel_backend: samples.sampler_kernel_backend,
    };
    Ok((samples, samples_b))
}

pub(in crate::commands::real_full) fn real_full_device_hidden_row(
    hidden: &DeviceBf16Output,
    row_index: usize,
) -> Result<DeviceBf16Output> {
    real_full_device_hidden_rows(hidden, row_index, 1)
}

pub(in crate::commands::real_full) fn real_full_device_hidden_rows(
    hidden: &DeviceBf16Output,
    row_start: usize,
    rows: usize,
) -> Result<DeviceBf16Output> {
    anyhow::ensure!(
        rows > 0
            && row_start
                .checked_add(rows)
                .is_some_and(|row_end| row_end <= hidden.rows)
            && hidden.values_per_row == GLM52_HIDDEN_SIZE,
        "real-full hidden rows {}+{} require Nx{}, got {}x{}",
        row_start,
        rows,
        GLM52_HIDDEN_SIZE,
        hidden.rows,
        hidden.values_per_row
    );
    let row_bytes = GLM52_HIDDEN_SIZE * std::mem::size_of::<u16>();
    let view = device_buffer_byte_view(
        hidden.buffer(),
        row_start * row_bytes,
        rows * row_bytes,
        "real-full hidden rows",
    )?;
    device_bf16_output_from_device_template_buffer(
        view,
        rows,
        GLM52_HIDDEN_SIZE,
        "real-full hidden rows",
    )
}

pub(in crate::commands::real_full) fn real_full_last_device_hidden_row(
    hidden: &DeviceBf16Output,
) -> Result<DeviceBf16Output> {
    anyhow::ensure!(hidden.rows > 0, "real-full hidden tail requires a row");
    real_full_device_hidden_row(hidden, hidden.rows - 1)
}

pub(in crate::commands::real_full) fn real_full_target_hidden_for_mtp(
    target_hidden: &DeviceBf16Output,
) -> Result<DeviceBf16Output> {
    rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output(
        TARGET_FINAL_NORM_WEIGHT,
        target_hidden.buffer(),
        target_hidden.rows,
        GLM52_HIDDEN_SIZE,
        MTP_RMSNORM_EPS,
    )
    .context("applying the target final norm before real-full MTP")
}

pub(in crate::commands::real_full) fn real_full_mtp_envelope_device_hidden(
    catalog: &TensorCatalog,
    shifted_token_ids: &[usize],
    target_hidden: &DeviceBf16Output,
    position_start: usize,
) -> Result<DeviceBf16Output> {
    validate_mtp_envelope_catalog(catalog)?;
    anyhow::ensure!(
        !shifted_token_ids.is_empty(),
        "real-full MTP envelope requires at least one draft token"
    );
    anyhow::ensure!(
        target_hidden.rows == shifted_token_ids.len()
            && target_hidden.values_per_row == GLM52_HIDDEN_SIZE,
        "real-full MTP target hidden shape must be {}x{}, got {}x{}",
        shifted_token_ids.len(),
        GLM52_HIDDEN_SIZE,
        target_hidden.rows,
        target_hidden.values_per_row
    );
    let embedding = real_full_embedding_device_hidden_for_tokens(catalog, shifted_token_ids)?
        .context("real-full MTP envelope requires the startup-resident device embedding")?;
    anyhow::ensure!(
        embedding.token_count == shifted_token_ids.len()
            && embedding.device_hidden.rows == shifted_token_ids.len()
            && embedding.device_hidden.values_per_row == GLM52_HIDDEN_SIZE,
        "real-full MTP embedding shape mismatch for {} draft tokens: got {}x{}",
        shifted_token_ids.len(),
        embedding.device_hidden.rows,
        embedding.device_hidden.values_per_row
    );
    if position_start == 0 {
        cuda_native_library()?
            .cuda_zero_bytes(
                embedding.device_hidden.buffer(),
                GLM52_HIDDEN_SIZE * std::mem::size_of::<u16>(),
            )
            .context("masking the real-full MTP position-zero embedding")?;
    }

    let normalized_embedding = rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output(
        MTP_ENORM_WEIGHT,
        embedding.device_hidden.buffer(),
        shifted_token_ids.len(),
        GLM52_HIDDEN_SIZE,
        MTP_RMSNORM_EPS,
    )
    .context("normalizing real-full MTP token embeddings")?;
    let normalized_hidden = rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output(
        MTP_HNORM_WEIGHT,
        target_hidden.buffer(),
        shifted_token_ids.len(),
        GLM52_HIDDEN_SIZE,
        MTP_RMSNORM_EPS,
    )
    .context("normalizing real-full MTP target hidden rows")?;
    let envelope = concat_device_bf16_row_features(
        &normalized_embedding,
        &normalized_hidden,
        "real-full MTP normalized embedding-hidden envelope",
    )?;
    linear_rows_bf16_preloaded_resident_weight_device_output(
        MTP_EH_PROJ_WEIGHT,
        envelope.buffer(),
        None,
        shifted_token_ids.len(),
        MTP_ENVELOPE_WIDTH,
        GLM52_HIDDEN_SIZE,
        GLM52_HIDDEN_SIZE,
    )
    .context("projecting real-full MTP embedding-hidden envelope")
}

pub(in crate::commands::real_full) fn real_full_mtp_shifted_input_token_ids(
    target_input_token_ids: &[usize],
    next_target_token_id: usize,
) -> Result<Vec<usize>> {
    anyhow::ensure!(
        !target_input_token_ids.is_empty(),
        "real-full MTP token shift requires at least one target input token"
    );
    let mut shifted = Vec::with_capacity(target_input_token_ids.len());
    shifted.extend_from_slice(&target_input_token_ids[1..]);
    shifted.push(next_target_token_id);
    Ok(shifted)
}

pub(in crate::commands::real_full) fn real_full_mtp_draft_token(
    catalog: &TensorCatalog,
    layer_hidden: &DeviceBf16Output,
    greedy_sampling: bool,
) -> Result<(RealFullMtpDraftToken, DeviceBf16Output)> {
    anyhow::ensure!(
        layer_hidden.rows == 1 && layer_hidden.values_per_row == GLM52_HIDDEN_SIZE,
        "real-full MTP draft scoring expects 1x{} hidden, got {}x{}",
        GLM52_HIDDEN_SIZE,
        layer_hidden.rows,
        layer_hidden.values_per_row
    );
    let normalized = rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output(
        MTP_SHARED_HEAD_NORM_WEIGHT,
        layer_hidden.buffer(),
        1,
        GLM52_HIDDEN_SIZE,
        MTP_RMSNORM_EPS,
    )
    .context("normalizing real-full MTP shared-head hidden")?;
    let score = score_real_lm_head_full_vocab_for_device_hidden(catalog, &normalized, 1)
        .context("scoring real-full MTP shared-head logits")?;
    anyhow::ensure!(
        score.covers_full_vocabulary && score.logits_evaluated == score.vocab_size,
        "real-full MTP draft scoring covered {} of {} logits",
        score.logits_evaluated,
        score.vocab_size
    );
    Ok((
        RealFullMtpDraftToken {
            token_id: if greedy_sampling {
                score.top_token_id
            } else {
                score.sampled_token_id
            },
            top_logit: score.top_logit,
            logits_evaluated: score.logits_evaluated,
            argmax_backend: score.argmax_kernel_backend,
        },
        normalized,
    ))
}

fn validate_mtp_envelope_catalog(catalog: &TensorCatalog) -> Result<()> {
    validate_mtp_tensor(
        catalog,
        MTP_EH_PROJ_WEIGHT,
        &[GLM52_HIDDEN_SIZE, MTP_ENVELOPE_WIDTH],
    )?;
    for name in [
        MTP_ENORM_WEIGHT,
        MTP_HNORM_WEIGHT,
        MTP_SHARED_HEAD_NORM_WEIGHT,
    ] {
        validate_mtp_tensor(catalog, name, &[GLM52_HIDDEN_SIZE])?;
    }
    Ok(())
}

fn validate_mtp_tensor(catalog: &TensorCatalog, name: &str, shape: &[usize]) -> Result<()> {
    let tensor = catalog
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .with_context(|| format!("real-full MTP tensor {name} is missing from the catalog"))?;
    anyhow::ensure!(
        tensor.dtype == DType::Bf16
            && tensor.role == TensorRole::Mtp
            && tensor.layer_id == Some(GLM52_MTP_LAYER_ID as u32)
            && tensor.expert_id.is_none()
            && !tensor.is_quantization_metadata,
        "real-full MTP tensor {name} has invalid metadata: dtype={:?} role={:?} layer={:?} expert={:?} quantization_metadata={}",
        tensor.dtype,
        tensor.role,
        tensor.layer_id,
        tensor.expert_id,
        tensor.is_quantization_metadata
    );
    anyhow::ensure!(
        tensor.shape == shape,
        "real-full MTP tensor {name} shape mismatch: expected {shape:?}, got {:?}",
        tensor.shape
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use glmrt_core::{ModelFacts, TensorCatalog, TensorInfo, DEFAULT_MODEL_ID};

    fn tensor(name: &str, shape: Vec<usize>) -> TensorInfo {
        TensorInfo {
            name: name.to_owned(),
            file: "model-mtp.safetensors".to_owned(),
            dtype: DType::Bf16,
            shape,
            byte_offset: 0,
            byte_length: 2,
            role: TensorRole::Mtp,
            layer_id: Some(GLM52_MTP_LAYER_ID as u32),
            expert_id: None,
            is_quantization_metadata: false,
        }
    }

    fn catalog() -> TensorCatalog {
        TensorCatalog {
            model_id: DEFAULT_MODEL_ID.to_owned(),
            snapshot_path: "/tmp/model".to_owned(),
            facts: ModelFacts::default(),
            tensors: vec![
                tensor(
                    MTP_EH_PROJ_WEIGHT,
                    vec![GLM52_HIDDEN_SIZE, MTP_ENVELOPE_WIDTH],
                ),
                tensor(MTP_ENORM_WEIGHT, vec![GLM52_HIDDEN_SIZE]),
                tensor(MTP_HNORM_WEIGHT, vec![GLM52_HIDDEN_SIZE]),
                tensor(MTP_SHARED_HEAD_NORM_WEIGHT, vec![GLM52_HIDDEN_SIZE]),
            ],
        }
    }

    #[test]
    fn mtp_envelope_catalog_accepts_glm52_shapes() {
        validate_mtp_envelope_catalog(&catalog()).unwrap();
    }

    #[test]
    fn mtp_envelope_catalog_rejects_wrong_projection_shape() {
        let mut catalog = catalog();
        catalog.tensors[0].shape = vec![GLM52_HIDDEN_SIZE, GLM52_HIDDEN_SIZE];
        assert!(validate_mtp_envelope_catalog(&catalog).is_err());
    }

    #[test]
    fn mtp_inputs_shift_target_tokens_and_append_next_target_token() {
        assert_eq!(
            real_full_mtp_shifted_input_token_ids(&[10, 20, 30], 40).unwrap(),
            vec![20, 30, 40]
        );
        assert_eq!(
            real_full_mtp_shifted_input_token_ids(&[10], 20).unwrap(),
            vec![20]
        );
        assert!(real_full_mtp_shifted_input_token_ids(&[], 20).is_err());
    }
}
