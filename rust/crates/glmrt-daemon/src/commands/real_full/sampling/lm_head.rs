use anyhow::{Context, Result};
use glmrt_core::{DType, TensorCatalog, TensorInfo, TensorRole};
use glmrt_loader::{load_tensor_rows, read_tensor_rows_into};

use super::REAL_FULL_SCORE_LM_HEAD_ENV;
use crate::commands::real_full::coordinator_kernels::{
    coordinator_cuda_reference_kernels_enabled, preload_resident_weight_from_host_staging,
    DeviceBf16Output,
};
use crate::commands::real_full::probe_env;
use crate::commands::real_full::types::{
    RealFullSamplingLmHeadChunkProbe, RealFullSamplingRealLmHeadProbe,
};

mod execution;
mod scorer;

use execution::{run_real_lm_head_default_chunk_probe, run_real_lm_head_full_vocab_probe};
use scorer::{
    lm_head_full_resident_available, score_bf16_lm_head_full_resident_sample,
    score_bf16_lm_head_full_resident_sample_device_input_with_options,
    score_bf16_lm_head_full_resident_sample_device_inputs_with_uniforms, score_bf16_lm_head_rows,
    score_bf16_lm_head_staged_resident_rows, LmHeadChunkScore,
};

#[derive(Debug, Clone, Copy)]
pub(in crate::commands::real_full) struct RealFullLmHeadSamplingOptions {
    pub(in crate::commands::real_full) random_uniform: f32,
    pub(in crate::commands::real_full) temperature: f32,
    pub(in crate::commands::real_full) top_k: usize,
    pub(in crate::commands::real_full) top_p: f32,
}

impl RealFullLmHeadSamplingOptions {
    pub(in crate::commands::real_full) fn diagnostic() -> Self {
        Self {
            random_uniform: 0.5,
            temperature: 0.7,
            top_k: 8,
            top_p: 0.95,
        }
    }
}

#[derive(Debug, Clone)]
pub(in crate::commands::real_full) struct RealLmHeadChunkScoreForHidden {
    pub(in crate::commands::real_full) lm_head_tensor: String,
    pub(in crate::commands::real_full) hidden_dim: usize,
    pub(in crate::commands::real_full) vocab_size: usize,
    pub(in crate::commands::real_full) start_token_id: usize,
    pub(in crate::commands::real_full) chunk_rows: usize,
    pub(in crate::commands::real_full) rows_scored: usize,
    pub(in crate::commands::real_full) chunks_scored: usize,
    pub(in crate::commands::real_full) lm_head_bytes_read: u64,
    pub(in crate::commands::real_full) hidden_values: usize,
    pub(in crate::commands::real_full) logits_evaluated: usize,
    pub(in crate::commands::real_full) multiply_accumulate_ops: u64,
    pub(in crate::commands::real_full) covers_full_vocabulary: bool,
    pub(in crate::commands::real_full) logits_kernel_backend: &'static str,
    pub(in crate::commands::real_full) argmax_kernel_backend: &'static str,
    pub(in crate::commands::real_full) sampler_kernel_backend: &'static str,
    pub(in crate::commands::real_full) top_token_id: usize,
    pub(in crate::commands::real_full) top_logit: f32,
    pub(in crate::commands::real_full) sampled_token_id: usize,
    pub(in crate::commands::real_full) sampled_score: f32,
    pub(in crate::commands::real_full) sample_random_uniform: f32,
    pub(in crate::commands::real_full) sample_temperature: f32,
    pub(in crate::commands::real_full) sample_top_k: usize,
    pub(in crate::commands::real_full) sample_top_p: f32,
}

pub(in crate::commands::real_full) struct RealLmHeadBatchScoreForHidden {
    pub(in crate::commands::real_full) vocab_size: usize,
    pub(in crate::commands::real_full) top_token_ids: Vec<usize>,
    pub(in crate::commands::real_full) sampled_token_ids: Vec<usize>,
    pub(in crate::commands::real_full) sample_top_k: usize,
    pub(in crate::commands::real_full) sample_top_p: f32,
    pub(in crate::commands::real_full) argmax_kernel_backend: &'static str,
    pub(in crate::commands::real_full) sampler_kernel_backend: &'static str,
}

pub(in crate::commands::real_full) fn score_real_lm_head_chunk_for_hidden(
    catalog: &TensorCatalog,
    hidden: &[f32],
    chunk_rows: usize,
) -> Result<RealLmHeadChunkScoreForHidden> {
    if chunk_rows == 0 {
        anyhow::bail!("layer-ordered lm_head chunk scoring requires non-zero chunk_rows");
    }
    let lm_head = catalog
        .tensors
        .iter()
        .find(|tensor| tensor.role == TensorRole::LmHead)
        .context("layer-ordered lm_head chunk scoring requires lm_head.weight in the catalog")?;
    if lm_head.dtype != DType::Bf16 {
        anyhow::bail!(
            "layer-ordered lm_head chunk scoring expects BF16 lm_head, found {:?}",
            lm_head.dtype
        );
    }
    if lm_head.shape.len() != 2 {
        anyhow::bail!(
            "layer-ordered lm_head chunk scoring expects 2D lm_head, found shape {:?}",
            lm_head.shape
        );
    }
    let vocab_size = lm_head.shape[0];
    let hidden_dim = lm_head.shape[1];
    if hidden.len() != hidden_dim {
        anyhow::bail!(
            "layer-ordered lm_head chunk scoring hidden width mismatch: expected {} got {}",
            hidden_dim,
            hidden.len()
        );
    }
    let hidden_bf16 = bf16_bytes_from_f32(hidden);
    let rows_scored = vocab_size.min(chunk_rows);
    let (chunk_result, lm_head_bytes_read) = score_bf16_lm_head_rows_from_catalog(
        catalog,
        lm_head,
        &hidden_bf16,
        0,
        rows_scored,
        vocab_size,
    )?;
    Ok(RealLmHeadChunkScoreForHidden {
        lm_head_tensor: lm_head.name.clone(),
        hidden_dim,
        vocab_size,
        start_token_id: 0,
        chunk_rows,
        rows_scored,
        chunks_scored: usize::from(rows_scored > 0),
        lm_head_bytes_read,
        hidden_values: hidden.len(),
        logits_evaluated: chunk_result.logits_evaluated,
        multiply_accumulate_ops: chunk_result.logits_evaluated as u64 * hidden_dim as u64,
        covers_full_vocabulary: rows_scored == vocab_size,
        logits_kernel_backend: chunk_result.logits_kernel_backend,
        argmax_kernel_backend: chunk_result.argmax_kernel_backend,
        sampler_kernel_backend: chunk_result.sampler_kernel_backend,
        top_token_id: chunk_result.top_token_id,
        top_logit: chunk_result.top_logit,
        sampled_token_id: chunk_result.sampled_token_id,
        sampled_score: chunk_result.sampled_score,
        sample_random_uniform: chunk_result.sample_random_uniform,
        sample_temperature: chunk_result.sample_temperature,
        sample_top_k: chunk_result.sample_top_k,
        sample_top_p: chunk_result.sample_top_p,
    })
}

pub(in crate::commands::real_full) fn score_real_lm_head_full_vocab_for_hidden(
    catalog: &TensorCatalog,
    hidden: &[f32],
    chunk_rows: usize,
) -> Result<RealLmHeadChunkScoreForHidden> {
    if chunk_rows == 0 {
        anyhow::bail!("layer-ordered lm_head full-vocab scoring requires non-zero chunk_rows");
    }
    let lm_head = catalog
        .tensors
        .iter()
        .find(|tensor| tensor.role == TensorRole::LmHead)
        .context(
            "layer-ordered lm_head full-vocab scoring requires lm_head.weight in the catalog",
        )?;
    if lm_head.dtype != DType::Bf16 {
        anyhow::bail!(
            "layer-ordered lm_head full-vocab scoring expects BF16 lm_head, found {:?}",
            lm_head.dtype
        );
    }
    if lm_head.shape.len() != 2 {
        anyhow::bail!(
            "layer-ordered lm_head full-vocab scoring expects 2D lm_head, found shape {:?}",
            lm_head.shape
        );
    }
    let vocab_size = lm_head.shape[0];
    let hidden_dim = lm_head.shape[1];
    if vocab_size == 0 {
        anyhow::bail!("layer-ordered lm_head full-vocab scoring requires a non-empty vocabulary");
    }
    if hidden.len() != hidden_dim {
        anyhow::bail!(
            "layer-ordered lm_head full-vocab scoring hidden width mismatch: expected {} got {}",
            hidden_dim,
            hidden.len()
        );
    }
    let hidden_bf16 = bf16_bytes_from_f32(hidden);
    if lm_head_full_resident_available(&lm_head.name, vocab_size, hidden_dim) {
        let sample =
            score_bf16_lm_head_full_resident_sample(&lm_head.name, &hidden_bf16, vocab_size)?;
        return Ok(RealLmHeadChunkScoreForHidden {
            lm_head_tensor: lm_head.name.clone(),
            hidden_dim,
            vocab_size,
            start_token_id: 0,
            chunk_rows: vocab_size,
            rows_scored: sample.logits_evaluated,
            chunks_scored: 1,
            lm_head_bytes_read: 0,
            hidden_values: hidden.len(),
            logits_evaluated: sample.logits_evaluated,
            multiply_accumulate_ops: sample.logits_evaluated as u64 * hidden_dim as u64,
            covers_full_vocabulary: sample.logits_evaluated == vocab_size,
            logits_kernel_backend: sample.logits_kernel_backend,
            argmax_kernel_backend: sample.argmax_kernel_backend,
            sampler_kernel_backend: sample.sampler_kernel_backend,
            top_token_id: sample.top_token_id,
            top_logit: sample.top_logit,
            sampled_token_id: sample.sampled_token_id,
            sampled_score: sample.sampled_score,
            sample_random_uniform: sample.sample_random_uniform,
            sample_temperature: sample.sample_temperature,
            sample_top_k: sample.sample_top_k,
            sample_top_p: sample.sample_top_p,
        });
    }

    let mut chunks_scored = 0_usize;
    let mut lm_head_bytes_read = 0_u64;
    let mut logits_evaluated = 0_usize;
    let mut top_token_id = 0_usize;
    let mut top_logit = f32::NEG_INFINITY;
    let mut sampled_token_id = 0_usize;
    let mut sampled_score = f32::NEG_INFINITY;
    let mut sample_random_uniform = 0.0_f32;
    let mut sample_temperature = 0.0_f32;
    let mut sample_top_k = 0_usize;
    let mut sample_top_p = 0.0_f32;
    let mut logits_kernel_backend = None;
    let mut argmax_kernel_backend = None;
    let mut sampler_kernel_backend = None;
    for chunk_start in (0..vocab_size).step_by(chunk_rows) {
        let rows_scored = (vocab_size - chunk_start).min(chunk_rows);
        let (chunk_result, chunk_bytes_read) = score_bf16_lm_head_rows_from_catalog(
            catalog,
            lm_head,
            &hidden_bf16,
            chunk_start,
            rows_scored,
            vocab_size,
        )?;
        if chunk_result.top_logit > top_logit {
            top_logit = chunk_result.top_logit;
            top_token_id = chunk_result.top_token_id;
            sampled_token_id = chunk_result.sampled_token_id;
            sampled_score = chunk_result.sampled_score;
            sample_random_uniform = chunk_result.sample_random_uniform;
            sample_temperature = chunk_result.sample_temperature;
            sample_top_k = chunk_result.sample_top_k;
            sample_top_p = chunk_result.sample_top_p;
        }
        logits_kernel_backend.get_or_insert(chunk_result.logits_kernel_backend);
        argmax_kernel_backend.get_or_insert(chunk_result.argmax_kernel_backend);
        sampler_kernel_backend.get_or_insert(chunk_result.sampler_kernel_backend);
        chunks_scored += 1;
        lm_head_bytes_read += chunk_bytes_read;
        logits_evaluated += chunk_result.logits_evaluated;
    }

    Ok(RealLmHeadChunkScoreForHidden {
        lm_head_tensor: lm_head.name.clone(),
        hidden_dim,
        vocab_size,
        start_token_id: 0,
        chunk_rows,
        rows_scored: logits_evaluated,
        chunks_scored,
        lm_head_bytes_read,
        hidden_values: hidden.len(),
        logits_evaluated,
        multiply_accumulate_ops: logits_evaluated as u64 * hidden_dim as u64,
        covers_full_vocabulary: logits_evaluated == vocab_size,
        logits_kernel_backend: logits_kernel_backend
            .context("layer-ordered lm_head full-vocab scoring produced no logits chunks")?,
        argmax_kernel_backend: argmax_kernel_backend
            .context("layer-ordered lm_head full-vocab scoring produced no argmax chunks")?,
        sampler_kernel_backend: sampler_kernel_backend
            .context("layer-ordered lm_head full-vocab scoring produced no sampler chunks")?,
        top_token_id,
        top_logit,
        sampled_token_id,
        sampled_score,
        sample_random_uniform,
        sample_temperature,
        sample_top_k,
        sample_top_p,
    })
}

pub(in crate::commands::real_full) fn score_real_lm_head_full_vocab_for_device_hidden(
    catalog: &TensorCatalog,
    hidden: &DeviceBf16Output,
    chunk_rows: usize,
) -> Result<RealLmHeadChunkScoreForHidden> {
    score_real_lm_head_full_vocab_for_device_hidden_with_options(
        catalog,
        hidden,
        chunk_rows,
        RealFullLmHeadSamplingOptions::diagnostic(),
    )
}

pub(in crate::commands::real_full) fn score_real_lm_head_full_vocab_for_device_hidden_with_options(
    catalog: &TensorCatalog,
    hidden: &DeviceBf16Output,
    chunk_rows: usize,
    sampler_options: RealFullLmHeadSamplingOptions,
) -> Result<RealLmHeadChunkScoreForHidden> {
    if chunk_rows == 0 {
        anyhow::bail!(
            "layer-ordered lm_head full-vocab device-input scoring requires non-zero chunk_rows"
        );
    }
    let lm_head = catalog
        .tensors
        .iter()
        .find(|tensor| tensor.role == TensorRole::LmHead)
        .context(
            "layer-ordered lm_head full-vocab device-input scoring requires lm_head.weight in the catalog",
        )?;
    if lm_head.dtype != DType::Bf16 {
        anyhow::bail!(
            "layer-ordered lm_head full-vocab device-input scoring expects BF16 lm_head, found {:?}",
            lm_head.dtype
        );
    }
    if lm_head.shape.len() != 2 {
        anyhow::bail!(
            "layer-ordered lm_head full-vocab device-input scoring expects 2D lm_head, found shape {:?}",
            lm_head.shape
        );
    }
    let vocab_size = lm_head.shape[0];
    let hidden_dim = lm_head.shape[1];
    if vocab_size == 0 {
        anyhow::bail!(
            "layer-ordered lm_head full-vocab device-input scoring requires a non-empty vocabulary"
        );
    }
    if hidden.rows != 1 || hidden.values_per_row != hidden_dim {
        anyhow::bail!(
            "layer-ordered lm_head full-vocab device-input hidden width mismatch: expected 1x{} got {}x{}",
            hidden_dim,
            hidden.rows,
            hidden.values_per_row
        );
    }
    let sample = score_bf16_lm_head_full_resident_sample_device_input_with_options(
        &lm_head.name,
        hidden,
        vocab_size,
        sampler_options,
    )?;
    Ok(RealLmHeadChunkScoreForHidden {
        lm_head_tensor: lm_head.name.clone(),
        hidden_dim,
        vocab_size,
        start_token_id: 0,
        chunk_rows: vocab_size,
        rows_scored: sample.logits_evaluated,
        chunks_scored: 1,
        lm_head_bytes_read: 0,
        hidden_values: hidden.values_per_row,
        logits_evaluated: sample.logits_evaluated,
        multiply_accumulate_ops: sample.logits_evaluated as u64 * hidden_dim as u64,
        covers_full_vocabulary: sample.logits_evaluated == vocab_size,
        logits_kernel_backend: sample.logits_kernel_backend,
        argmax_kernel_backend: sample.argmax_kernel_backend,
        sampler_kernel_backend: sample.sampler_kernel_backend,
        top_token_id: sample.top_token_id,
        top_logit: sample.top_logit,
        sampled_token_id: sample.sampled_token_id,
        sampled_score: sample.sampled_score,
        sample_random_uniform: sample.sample_random_uniform,
        sample_temperature: sample.sample_temperature,
        sample_top_k: sample.sample_top_k,
        sample_top_p: sample.sample_top_p,
    })
}

pub(in crate::commands::real_full) fn score_real_lm_head_full_vocab_for_device_hidden_rows(
    catalog: &TensorCatalog,
    hidden: &DeviceBf16Output,
) -> Result<RealLmHeadBatchScoreForHidden> {
    let random_uniforms =
        vec![RealFullLmHeadSamplingOptions::diagnostic().random_uniform; hidden.rows];
    score_real_lm_head_full_vocab_for_device_hidden_rows_with_options(
        catalog,
        hidden,
        RealFullLmHeadSamplingOptions::diagnostic(),
        &random_uniforms,
        true,
    )
}

pub(in crate::commands::real_full) fn score_real_lm_head_full_vocab_for_device_hidden_rows_with_options(
    catalog: &TensorCatalog,
    hidden: &DeviceBf16Output,
    sampler_options: RealFullLmHeadSamplingOptions,
    random_uniforms: &[f32],
    allow_graph: bool,
) -> Result<RealLmHeadBatchScoreForHidden> {
    let lm_head = catalog
        .tensors
        .iter()
        .find(|tensor| tensor.role == TensorRole::LmHead)
        .context("batch lm_head scoring requires lm_head.weight in the catalog")?;
    anyhow::ensure!(
        lm_head.dtype == DType::Bf16 && lm_head.shape.len() == 2,
        "batch lm_head scoring expects a 2D BF16 lm_head, got dtype={:?} shape={:?}",
        lm_head.dtype,
        lm_head.shape
    );
    let vocab_size = lm_head.shape[0];
    let hidden_dim = lm_head.shape[1];
    anyhow::ensure!(
        hidden.rows > 0 && hidden.values_per_row == hidden_dim,
        "batch lm_head hidden shape must be Nx{}, got {}x{}",
        hidden_dim,
        hidden.rows,
        hidden.values_per_row
    );
    let score = score_bf16_lm_head_full_resident_sample_device_inputs_with_uniforms(
        &lm_head.name,
        hidden,
        vocab_size,
        sampler_options,
        random_uniforms,
        allow_graph,
    )?;
    Ok(RealLmHeadBatchScoreForHidden {
        vocab_size,
        top_token_ids: score.top_token_ids,
        sampled_token_ids: score.sampled_token_ids,
        sample_top_k: score.sample_top_k,
        sample_top_p: score.sample_top_p,
        argmax_kernel_backend: score.argmax_kernel_backend,
        sampler_kernel_backend: score.sampler_kernel_backend,
    })
}

fn score_bf16_lm_head_rows_from_catalog(
    catalog: &TensorCatalog,
    lm_head: &TensorInfo,
    hidden_bf16: &[u8],
    start_token_id: usize,
    row_count: usize,
    full_vocab_size: usize,
) -> Result<(LmHeadChunkScore, u64)> {
    if hidden_bf16.is_empty() || hidden_bf16.len() % 2 != 0 {
        anyhow::bail!(
            "lm_head row-window scoring hidden BF16 row must be non-empty and 2-byte aligned"
        );
    }
    if row_count == 0 {
        anyhow::bail!("lm_head row-window scoring requires non-zero row_count");
    }
    let hidden_dim = hidden_bf16.len() / 2;
    if lm_head_full_resident_available(&lm_head.name, full_vocab_size, hidden_dim) {
        let score = score_bf16_lm_head_rows(
            &lm_head.name,
            hidden_bf16,
            &[],
            start_token_id,
            row_count,
            full_vocab_size,
        )?;
        return Ok((score, 0));
    }

    if coordinator_cuda_reference_kernels_enabled() {
        let end_token_id = start_token_id
            .checked_add(row_count)
            .context("lm_head pinned row-window end overflows usize")?;
        let lm_head_key = format!("{}[rows={start_token_id}..{end_token_id}]", lm_head.name);
        let expected_bytes = row_count
            .checked_mul(hidden_bf16.len())
            .context("lm_head pinned row-window byte length overflows usize")?;
        let mut bytes_read = 0_u64;
        preload_resident_weight_from_host_staging(
            &lm_head_key,
            expected_bytes,
            "BF16 lm_head row-window pinned staging",
            |staging| {
                let summary = read_tensor_rows_into(
                    catalog,
                    &lm_head.name,
                    start_token_id,
                    row_count,
                    staging,
                )
                .with_context(|| {
                    format!(
                        "reading lm_head row window [{start_token_id}, {end_token_id}) into pinned staging"
                    )
                })?;
                if summary.dtype != DType::Bf16 {
                    anyhow::bail!(
                        "lm_head pinned row-window expects BF16 rows, got {:?}",
                        summary.dtype
                    );
                }
                if summary.row_width != hidden_dim {
                    anyhow::bail!(
                        "lm_head pinned row-window width mismatch: expected {} got {}",
                        hidden_dim,
                        summary.row_width
                    );
                }
                if summary.bytes_read as usize != expected_bytes {
                    anyhow::bail!(
                        "lm_head pinned row-window read {} bytes, expected {}",
                        summary.bytes_read,
                        expected_bytes
                    );
                }
                bytes_read = summary.bytes_read;
                Ok(())
            },
        )?;
        let score = score_bf16_lm_head_staged_resident_rows(
            &lm_head_key,
            hidden_bf16,
            start_token_id,
            row_count,
            full_vocab_size,
        )?;
        return Ok((score, bytes_read));
    }

    let rows = load_tensor_rows(catalog, &lm_head.name, start_token_id, row_count)?;
    if rows.info.dtype != DType::Bf16 {
        anyhow::bail!(
            "lm_head row-window scoring expects BF16 rows, got {:?}",
            rows.info.dtype
        );
    }
    if rows.row_width != hidden_dim {
        anyhow::bail!(
            "lm_head row-window scoring width mismatch: expected {} got {}",
            hidden_dim,
            rows.row_width
        );
    }
    let bytes_read = rows.bytes.len() as u64;
    let score = score_bf16_lm_head_rows(
        &lm_head.name,
        hidden_bf16,
        &rows.bytes,
        start_token_id,
        row_count,
        full_vocab_size,
    )?;
    Ok((score, bytes_read))
}

pub(super) fn real_lm_head_default_chunk_probe(
    catalog: &TensorCatalog,
    lm_head: &TensorInfo,
    chunk_rows: usize,
) -> RealFullSamplingLmHeadChunkProbe {
    match run_real_lm_head_default_chunk_probe(catalog, lm_head, chunk_rows) {
        Ok(probe) => probe,
        Err(error) => failed_default_chunk_probe(lm_head, chunk_rows, Some(error.to_string())),
    }
}

pub(super) fn real_lm_head_full_vocab_probe(
    catalog: &TensorCatalog,
    lm_head: &TensorInfo,
    chunk_rows: usize,
) -> RealFullSamplingRealLmHeadProbe {
    if probe_env::var(REAL_FULL_SCORE_LM_HEAD_ENV).as_deref() != Ok("1") {
        return skipped_real_lm_head_probe(
            "not-run",
            lm_head,
            chunk_rows,
            REAL_FULL_SCORE_LM_HEAD_ENV,
            Some(format!(
                "set {REAL_FULL_SCORE_LM_HEAD_ENV}=1 to stream-score real lm_head.weight against a deterministic diagnostic hidden row; live request sampling uses the scheduler terminal lm_head path"
            )),
        );
    }

    match run_real_lm_head_full_vocab_probe(catalog, lm_head, chunk_rows) {
        Ok(probe) => probe,
        Err(error) => skipped_real_lm_head_probe(
            "error",
            lm_head,
            chunk_rows,
            REAL_FULL_SCORE_LM_HEAD_ENV,
            Some(error.to_string()),
        ),
    }
}

fn skipped_real_lm_head_probe(
    status: &'static str,
    lm_head: &TensorInfo,
    chunk_rows: usize,
    opt_in_env: &'static str,
    skipped_reason: Option<String>,
) -> RealFullSamplingRealLmHeadProbe {
    let vocab_size = lm_head.shape.first().copied().unwrap_or_default();
    let hidden_dim = lm_head.shape.get(1).copied().unwrap_or_default();
    let chunk_count = if chunk_rows == 0 || vocab_size == 0 {
        0
    } else {
        vocab_size.div_ceil(chunk_rows)
    };
    let final_chunk_rows = if chunk_count == 0 {
        0
    } else {
        vocab_size - ((chunk_count - 1) * chunk_rows)
    };
    RealFullSamplingRealLmHeadProbe {
        status,
        scope: "stream-score real lm_head.weight rows against a terminal hidden row",
        opt_in_env,
        hidden_source: "not-run",
        uses_real_lm_head: false,
        uses_full_model_residual: false,
        hidden_dim,
        vocab_size,
        chunk_rows,
        chunk_count,
        final_chunk_rows,
        chunks_scored: 0,
        lm_head_bytes_read: 0,
        hidden_bytes: 0,
        logits_evaluated: 0,
        multiply_accumulate_ops: 0,
        logits_kernel_backend: None,
        argmax_kernel_backend: None,
        sampler_kernel_backend: None,
        top_token_id: None,
        top_logit: None,
        sampled_token_id: None,
        sampled_score: None,
        sample_random_uniform: None,
        sample_temperature: None,
        sample_top_k: None,
        sample_top_p: None,
        uses_real_attention_prefix: false,
        uses_real_nvfp4_residual_prefix: false,
        uses_real_nvfp4_residual_chain: false,
        uses_real_sparse_mlp_shared_chain: false,
        uses_real_dense_prefix: false,
        residual_source_layer_id: None,
        residual_source_attention_rows: 0,
        residual_source_attention_residual_adds: 0,
        residual_source_includes_causal_softmax: false,
        residual_source_includes_mla_softmax: false,
        residual_source_dense_layers: 0,
        residual_source_dense_residual_adds: 0,
        residual_source_dense_norm_bytes_read: 0,
        residual_source_dense_weight_bytes_read: 0,
        residual_source_covers_all_dense_layers: false,
        residual_source_sparse_layers: 0,
        residual_source_top_k: 0,
        residual_source_route_count: 0,
        residual_source_output_rows: 0,
        residual_source_residual_adds: 0,
        residual_source_total_residual_adds: 0,
        residual_source_router_weight_bytes_read: 0,
        residual_source_router_bias_bytes_read: 0,
        residual_source_weight_bytes_read: 0,
        residual_source_quant_metadata_bytes_read: 0,
        residual_source_shared_expert_layers: 0,
        residual_source_shared_weight_bytes_read: 0,
        residual_source_covers_all_sparse_layers: false,
        residual_source_covers_full_top_k: false,
        residual_prefix_values: 0,
        residual_before_checksum: None,
        residual_delta_checksum: None,
        residual_after_checksum: None,
        passed: false,
        skipped_reason,
    }
}

fn failed_default_chunk_probe(
    lm_head: &TensorInfo,
    chunk_rows: usize,
    error: Option<String>,
) -> RealFullSamplingLmHeadChunkProbe {
    let vocab_size = lm_head.shape.first().copied().unwrap_or_default();
    let hidden_dim = lm_head.shape.get(1).copied().unwrap_or_default();
    RealFullSamplingLmHeadChunkProbe {
        status: "error",
        scope: "score the first default-sized real lm_head.weight chunk against a deterministic terminal hidden row",
        hidden_source: "deterministic-terminal-residual-shaped-bf16-row",
        uses_real_lm_head: false,
        uses_full_model_residual: false,
        uses_real_dense_prefix: false,
        hidden_dim,
        vocab_size,
        start_token_id: 0,
        chunk_rows,
        rows_scored: 0,
        chunks_scored: 0,
        lm_head_bytes_read: 0,
        hidden_bytes: 0,
        logits_evaluated: 0,
        multiply_accumulate_ops: 0,
        logits_kernel_backend: None,
        argmax_kernel_backend: None,
        sampler_kernel_backend: None,
        top_token_id: None,
        top_logit: None,
        sampled_token_id: None,
        sampled_score: None,
        sample_random_uniform: None,
        sample_temperature: None,
        sample_top_k: None,
        sample_top_p: None,
        residual_source_dense_layers: 0,
        residual_source_dense_residual_adds: 0,
        residual_source_dense_weight_bytes_read: 0,
        residual_source_covers_all_dense_layers: false,
        residual_after_checksum: None,
        passed: false,
        error,
    }
}

fn bf16_bytes_from_f32(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::real_full::coordinator_kernels::cuda_reference_kernels_test_override;
    use glmrt_core::{ModelFacts, TensorCatalog, TensorInfo};
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn score_real_lm_head_full_vocab_for_hidden_streams_all_chunks() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);
        let tempdir = tempfile::tempdir().unwrap();
        let shard_path = tempdir.path().join("lm_head.bin");
        let lm_head_bytes = bf16_bytes(&[
            0.0, 0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0, //
            0.0, -1.0, 0.0, 0.0, //
            0.0, 0.0, 4.0, 0.0, //
            0.0, 0.0, 0.0, -12.0, //
        ]);
        File::create(&shard_path)
            .unwrap()
            .write_all(&lm_head_bytes)
            .unwrap();
        let lm_head = TensorInfo {
            name: "lm_head.weight".to_owned(),
            file: "lm_head.bin".to_owned(),
            dtype: DType::Bf16,
            shape: vec![5, 4],
            byte_offset: 0,
            byte_length: lm_head_bytes.len() as u64,
            role: TensorRole::LmHead,
            layer_id: None,
            expert_id: None,
            is_quantization_metadata: false,
        };
        let catalog = TensorCatalog {
            model_id: "test/model".to_owned(),
            snapshot_path: tempdir.path().display().to_string(),
            facts: ModelFacts::default(),
            tensors: vec![lm_head.clone()],
        };

        let score =
            score_real_lm_head_full_vocab_for_hidden(&catalog, &[0.25, -1.0, 0.5, -0.25], 2)
                .unwrap();

        assert_eq!(score.lm_head_tensor, "lm_head.weight");
        assert_eq!(score.hidden_dim, 4);
        assert_eq!(score.vocab_size, 5);
        assert_eq!(score.chunk_rows, 2);
        assert_eq!(score.rows_scored, 5);
        assert_eq!(score.chunks_scored, 3);
        assert_eq!(score.lm_head_bytes_read, lm_head.byte_length);
        assert_eq!(score.hidden_values, 4);
        assert_eq!(score.logits_evaluated, 5);
        assert_eq!(score.multiply_accumulate_ops, 20);
        assert!(score.covers_full_vocabulary);
        assert_eq!(
            score.logits_kernel_backend,
            "cpu-reference-lm-head-argmax-bf16"
        );
        assert_eq!(
            score.argmax_kernel_backend,
            "cpu-reference-lm-head-argmax-bf16"
        );
        assert_eq!(
            score.sampler_kernel_backend,
            "cpu-reference-lm-head-sample-topk-topp-bf16"
        );
        assert_eq!(score.top_token_id, 4);
        assert!((score.top_logit - 3.0).abs() < 1.0e-6);
        assert_eq!(score.sampled_token_id, 4);
        assert!((score.sampled_score - 1.0).abs() < 1.0e-6);
        assert_eq!(score.sample_top_k, 1);
        assert_eq!(score.sample_top_p, 1.0);
        assert_eq!(score.sample_temperature, 1.0);
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }
}
