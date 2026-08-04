use anyhow::{Context, Result};

use super::RealFullLmHeadSamplingOptions;
use crate::commands::real_full::coordinator_kernels::{
    coordinator_cuda_reference_kernels_enabled, lm_head_argmax_bf16_preloaded_resident_weight,
    lm_head_argmax_bf16_preloaded_resident_weight_device_input,
    lm_head_argmax_bf16_resident_weight, lm_head_argmax_bf16_staged_resident_weight,
    lm_head_argmax_sample_topk_topp_bf16_preloaded_resident_weight_device_input,
    lm_head_argmax_sample_topk_topp_bf16_preloaded_resident_weight_device_input_without_graph,
    lm_head_constrained_argmax_sample_topk_topp_bf16_preloaded_resident_weight_device_input,
    lm_head_sample_topk_topp_bf16_preloaded_resident_weight,
    lm_head_sample_topk_topp_bf16_preloaded_resident_weight_device_input,
    lm_head_sample_topk_topp_bf16_resident_weight,
    lm_head_sample_topk_topp_bf16_staged_resident_weight, resident_weight_is_preloaded,
    DeviceBf16Output,
};

const GREEDY_SAMPLER_OPTIONS: RealFullLmHeadSamplingOptions = RealFullLmHeadSamplingOptions {
    random_uniform: 0.0,
    temperature: 1.0,
    top_k: 1,
    top_p: 1.0,
};
const FULL_VOCAB_SAMPLER_OPTIONS: RealFullLmHeadSamplingOptions = RealFullLmHeadSamplingOptions {
    random_uniform: 0.5,
    temperature: 0.7,
    top_k: 8,
    top_p: 0.95,
};
#[derive(Debug, Clone, Copy)]
pub(super) struct LmHeadChunkScore {
    pub(super) logits_evaluated: usize,
    pub(super) top_token_id: usize,
    pub(super) top_logit: f32,
    pub(super) sampled_token_id: usize,
    pub(super) sampled_score: f32,
    pub(super) sample_random_uniform: f32,
    pub(super) sample_temperature: f32,
    pub(super) sample_top_k: usize,
    pub(super) sample_top_p: f32,
    pub(super) logits_kernel_backend: &'static str,
    pub(super) argmax_kernel_backend: &'static str,
    pub(super) sampler_kernel_backend: &'static str,
}

pub(super) fn score_bf16_lm_head_rows(
    lm_head_name: &str,
    hidden_bf16: &[u8],
    lm_head_bf16: &[u8],
    start_token_id: usize,
    row_count: usize,
    full_vocab_size: usize,
) -> Result<LmHeadChunkScore> {
    if hidden_bf16.is_empty() || hidden_bf16.len() % 2 != 0 {
        anyhow::bail!("real lm_head probe hidden BF16 row must be non-empty and 2-byte aligned");
    }
    let hidden_dim = hidden_bf16.len() / 2;
    let row_bytes = hidden_bf16.len();
    let expected_bytes = row_count
        .checked_mul(row_bytes)
        .context("real lm_head probe row byte length overflow")?;
    let end_token_id = start_token_id
        .checked_add(row_count)
        .context("real lm_head probe row-window token range overflows usize")?;
    if end_token_id > full_vocab_size {
        anyhow::bail!(
            "real lm_head probe row-window [{}, {}) exceeds full vocabulary {}",
            start_token_id,
            end_token_id,
            full_vocab_size
        );
    }
    let full_resident_available =
        lm_head_full_resident_available(lm_head_name, full_vocab_size, hidden_dim);
    if !full_resident_available && lm_head_bf16.len() != expected_bytes {
        anyhow::bail!(
            "real lm_head probe row bytes mismatch: expected {} got {}",
            expected_bytes,
            lm_head_bf16.len()
        );
    }
    let sampler_options = GREEDY_SAMPLER_OPTIONS;
    let (argmax, sampler) = if full_resident_available {
        (
            lm_head_argmax_bf16_preloaded_resident_weight(
                lm_head_name,
                hidden_bf16,
                1,
                hidden_dim,
                full_vocab_size,
                start_token_id,
                row_count,
            )?,
            lm_head_sample_topk_topp_bf16_preloaded_resident_weight(
                lm_head_name,
                hidden_bf16,
                &[sampler_options.random_uniform],
                1,
                hidden_dim,
                full_vocab_size,
                start_token_id,
                row_count,
                sampler_options.temperature,
                sampler_options.top_k,
                sampler_options.top_p,
            )?,
        )
    } else {
        let lm_head_key = format!("{lm_head_name}[rows={start_token_id}..{end_token_id}]");
        (
            lm_head_argmax_bf16_resident_weight(
                &lm_head_key,
                hidden_bf16,
                lm_head_bf16,
                1,
                hidden_dim,
                row_count,
            )?,
            lm_head_sample_topk_topp_bf16_resident_weight(
                &lm_head_key,
                hidden_bf16,
                lm_head_bf16,
                &[sampler_options.random_uniform],
                1,
                hidden_dim,
                row_count,
                sampler_options.temperature,
                sampler_options.top_k,
                sampler_options.top_p,
            )?,
        )
    };
    let top_index = sampler
        .indices
        .first()
        .copied()
        .context("real lm_head scorer expected one sampled index")?;
    let argmax_index = argmax
        .indices
        .first()
        .copied()
        .context("real lm_head scorer expected one argmax index")?;
    if top_index != argmax_index {
        anyhow::bail!(
            "real lm_head scorer sampled token index {} did not match greedy argmax {} for top_k=1",
            top_index,
            argmax_index
        );
    }
    let top_logit = argmax
        .scores
        .first()
        .copied()
        .context("real lm_head scorer expected one argmax score")?;
    let sampling_score = sampler
        .scores
        .first()
        .copied()
        .context("real lm_head scorer expected one sampled score")?;
    if !valid_sampled_score(sampling_score) {
        anyhow::bail!("real lm_head scorer sampler produced invalid sampled score");
    }
    Ok(LmHeadChunkScore {
        logits_evaluated: row_count,
        top_token_id: start_token_id + argmax_index,
        top_logit,
        sampled_token_id: start_token_id + top_index,
        sampled_score: sampling_score,
        sample_random_uniform: sampler_options.random_uniform,
        sample_temperature: sampler_options.temperature,
        sample_top_k: sampler_options.top_k,
        sample_top_p: sampler_options.top_p,
        logits_kernel_backend: argmax.backend,
        argmax_kernel_backend: argmax.backend,
        sampler_kernel_backend: sampler.backend,
    })
}

pub(super) fn score_bf16_lm_head_full_resident_sample(
    lm_head_name: &str,
    hidden_bf16: &[u8],
    full_vocab_size: usize,
) -> Result<LmHeadChunkScore> {
    if hidden_bf16.is_empty() || hidden_bf16.len() % 2 != 0 {
        anyhow::bail!(
            "real lm_head full-resident sampler hidden BF16 row must be non-empty and 2-byte aligned"
        );
    }
    if full_vocab_size == 0 {
        anyhow::bail!("real lm_head full-resident sampler requires non-empty vocabulary");
    }
    let hidden_dim = hidden_bf16.len() / 2;
    if !lm_head_full_resident_available(lm_head_name, full_vocab_size, hidden_dim) {
        anyhow::bail!("real lm_head full-resident sampler requires preloaded full lm_head.weight");
    }
    let sampler_options = RealFullLmHeadSamplingOptions {
        top_k: FULL_VOCAB_SAMPLER_OPTIONS.top_k.min(full_vocab_size),
        ..FULL_VOCAB_SAMPLER_OPTIONS
    };
    let argmax = lm_head_argmax_bf16_preloaded_resident_weight(
        lm_head_name,
        hidden_bf16,
        1,
        hidden_dim,
        full_vocab_size,
        0,
        full_vocab_size,
    )?;
    let sampler = lm_head_sample_topk_topp_bf16_preloaded_resident_weight(
        lm_head_name,
        hidden_bf16,
        &[sampler_options.random_uniform],
        1,
        hidden_dim,
        full_vocab_size,
        0,
        full_vocab_size,
        sampler_options.temperature,
        sampler_options.top_k,
        sampler_options.top_p,
    )?;
    let top_token_id = argmax
        .indices
        .first()
        .copied()
        .context("real lm_head full-resident sampler expected one argmax index")?;
    let top_logit = argmax
        .scores
        .first()
        .copied()
        .context("real lm_head full-resident sampler expected one argmax score")?;
    let sampled_token_id = sampler
        .indices
        .first()
        .copied()
        .context("real lm_head full-resident sampler expected one sampled index")?;
    let sampled_score = sampler
        .scores
        .first()
        .copied()
        .context("real lm_head full-resident sampler expected one sampled score")?;
    if !top_logit.is_finite() || !valid_sampled_score(sampled_score) {
        anyhow::bail!(
            "real lm_head full-resident sampler produced invalid score: top_logit={top_logit} sampled_score={sampled_score} argmax_backend={} sampler_backend={} top_token_id={top_token_id} sampled_token_id={sampled_token_id}",
            argmax.backend,
            sampler.backend
        );
    }

    Ok(LmHeadChunkScore {
        logits_evaluated: full_vocab_size,
        top_token_id,
        top_logit,
        sampled_token_id,
        sampled_score,
        sample_random_uniform: sampler_options.random_uniform,
        sample_temperature: sampler_options.temperature,
        sample_top_k: sampler_options.top_k,
        sample_top_p: sampler_options.top_p,
        logits_kernel_backend: argmax.backend,
        argmax_kernel_backend: argmax.backend,
        sampler_kernel_backend: sampler.backend,
    })
}

pub(super) struct LmHeadBatchScore {
    pub(super) top_token_ids: Vec<usize>,
    pub(super) top_logits: Vec<f32>,
    pub(super) sampled_token_ids: Vec<usize>,
    pub(super) sampled_scores: Vec<f32>,
    pub(super) sample_random_uniform: f32,
    pub(super) sample_temperature: f32,
    pub(super) sample_top_k: usize,
    pub(super) sample_top_p: f32,
    pub(super) argmax_kernel_backend: &'static str,
    pub(super) sampler_kernel_backend: &'static str,
}

pub(super) fn score_bf16_lm_head_full_resident_sample_device_inputs(
    lm_head_name: &str,
    hidden: &DeviceBf16Output,
    full_vocab_size: usize,
) -> Result<LmHeadBatchScore> {
    let sampler_options = RealFullLmHeadSamplingOptions {
        top_k: FULL_VOCAB_SAMPLER_OPTIONS.top_k.min(full_vocab_size),
        ..FULL_VOCAB_SAMPLER_OPTIONS
    };
    score_bf16_lm_head_full_resident_sample_device_inputs_with_options(
        lm_head_name,
        hidden,
        full_vocab_size,
        sampler_options,
        true,
    )
}

fn score_bf16_lm_head_full_resident_sample_device_inputs_with_options(
    lm_head_name: &str,
    hidden: &DeviceBf16Output,
    full_vocab_size: usize,
    sampler_options: RealFullLmHeadSamplingOptions,
    allow_graph: bool,
) -> Result<LmHeadBatchScore> {
    let random_uniforms = vec![sampler_options.random_uniform; hidden.rows];
    score_bf16_lm_head_full_resident_sample_device_inputs_with_uniforms(
        lm_head_name,
        hidden,
        full_vocab_size,
        sampler_options,
        &random_uniforms,
        allow_graph,
    )
}

pub(super) fn score_bf16_lm_head_full_resident_sample_device_inputs_with_uniforms(
    lm_head_name: &str,
    hidden: &DeviceBf16Output,
    full_vocab_size: usize,
    sampler_options: RealFullLmHeadSamplingOptions,
    random_uniforms: &[f32],
    allow_graph: bool,
) -> Result<LmHeadBatchScore> {
    if hidden.rows == 0 || hidden.values_per_row == 0 {
        anyhow::bail!(
            "real lm_head full-resident device-input sampler expected non-empty hidden rows, got {}x{}",
            hidden.rows,
            hidden.values_per_row
        );
    }
    if full_vocab_size == 0 {
        anyhow::bail!(
            "real lm_head full-resident device-input sampler requires non-empty vocabulary"
        );
    }
    let hidden_dim = hidden.values_per_row;
    if !lm_head_full_resident_available(lm_head_name, full_vocab_size, hidden_dim) {
        anyhow::bail!(
            "real lm_head full-resident device-input sampler requires preloaded full lm_head.weight"
        );
    }
    let sampler_options = RealFullLmHeadSamplingOptions {
        top_k: sampler_options.top_k.min(full_vocab_size),
        ..sampler_options
    };
    anyhow::ensure!(
        random_uniforms.len() == hidden.rows,
        "real lm_head full-resident sampler received {} random uniforms for {} rows",
        random_uniforms.len(),
        hidden.rows
    );
    let combined = if allow_graph {
        lm_head_argmax_sample_topk_topp_bf16_preloaded_resident_weight_device_input(
            lm_head_name,
            hidden,
            random_uniforms,
            full_vocab_size,
            0,
            full_vocab_size,
            sampler_options.temperature,
            sampler_options.top_k,
            sampler_options.top_p,
        )
    } else {
        lm_head_argmax_sample_topk_topp_bf16_preloaded_resident_weight_device_input_without_graph(
            lm_head_name,
            hidden,
            random_uniforms,
            full_vocab_size,
            0,
            full_vocab_size,
            sampler_options.temperature,
            sampler_options.top_k,
            sampler_options.top_p,
        )
    };
    let (top_token_ids, top_logits, argmax_backend, sampler) = match combined {
        Ok(combined) => (
            combined.argmax.indices,
            combined.argmax.scores,
            combined.argmax.backend,
            combined.sampler,
        ),
        Err(error) if !allow_graph => return Err(error),
        Err(_) => {
            let argmax = lm_head_argmax_bf16_preloaded_resident_weight_device_input(
                lm_head_name,
                hidden,
                full_vocab_size,
                0,
                full_vocab_size,
            )?;
            let sampler = lm_head_sample_topk_topp_bf16_preloaded_resident_weight_device_input(
                lm_head_name,
                hidden,
                random_uniforms,
                full_vocab_size,
                0,
                full_vocab_size,
                sampler_options.temperature,
                sampler_options.top_k,
                sampler_options.top_p,
            )?;
            (argmax.indices, argmax.scores, argmax.backend, sampler)
        }
    };
    anyhow::ensure!(
        top_token_ids.len() == hidden.rows
            && top_logits.len() == hidden.rows
            && sampler.indices.len() == hidden.rows
            && sampler.scores.len() == hidden.rows,
        "real lm_head batch sampler output lengths argmax_ids={} argmax_scores={} sampled_ids={} sampled_scores={} do not match rows {}",
        top_token_ids.len(),
        top_logits.len(),
        sampler.indices.len(),
        sampler.scores.len(),
        hidden.rows
    );
    if top_logits.iter().any(|score| !score.is_finite())
        || sampler
            .scores
            .iter()
            .copied()
            .any(|score| !valid_sampled_score(score))
    {
        anyhow::bail!(
            "real lm_head full-resident device-input batch sampler produced invalid scores for {} rows: argmax_backend={argmax_backend} sampler_backend={}",
            hidden.rows,
            sampler.backend
        );
    }

    Ok(LmHeadBatchScore {
        top_token_ids,
        top_logits,
        sampled_token_ids: sampler.indices,
        sampled_scores: sampler.scores,
        sample_random_uniform: random_uniforms[0],
        sample_temperature: sampler_options.temperature,
        sample_top_k: sampler_options.top_k,
        sample_top_p: sampler_options.top_p,
        argmax_kernel_backend: argmax_backend,
        sampler_kernel_backend: sampler.backend,
    })
}

pub(super) fn score_bf16_lm_head_full_resident_constrained_sample_device_inputs_with_uniforms(
    lm_head_name: &str,
    hidden: &DeviceBf16Output,
    full_vocab_size: usize,
    sampler_options: RealFullLmHeadSamplingOptions,
    random_uniforms: &[f32],
    token_bitmasks: &[u32],
) -> Result<LmHeadBatchScore> {
    anyhow::ensure!(
        hidden.rows > 0 && hidden.values_per_row > 0,
        "constrained real lm_head sampler expected non-empty hidden rows, got {}x{}",
        hidden.rows,
        hidden.values_per_row
    );
    anyhow::ensure!(
        full_vocab_size > 0,
        "constrained real lm_head sampler requires non-empty vocabulary"
    );
    anyhow::ensure!(
        lm_head_full_resident_available(lm_head_name, full_vocab_size, hidden.values_per_row),
        "constrained real lm_head sampler requires preloaded full lm_head.weight"
    );
    anyhow::ensure!(
        random_uniforms.len() == hidden.rows,
        "constrained real lm_head sampler received {} random uniforms for {} rows",
        random_uniforms.len(),
        hidden.rows
    );
    let mask_words = full_vocab_size.div_ceil(u32::BITS as usize);
    anyhow::ensure!(
        token_bitmasks.len() == hidden.rows * mask_words,
        "constrained real lm_head sampler received {} mask words for {}x{}",
        token_bitmasks.len(),
        hidden.rows,
        mask_words
    );
    let sampler_options = RealFullLmHeadSamplingOptions {
        top_k: sampler_options.top_k.min(full_vocab_size),
        ..sampler_options
    };
    let combined =
        lm_head_constrained_argmax_sample_topk_topp_bf16_preloaded_resident_weight_device_input(
            lm_head_name,
            hidden,
            random_uniforms,
            token_bitmasks,
            full_vocab_size,
            sampler_options.temperature,
            sampler_options.top_k,
            sampler_options.top_p,
        )?;
    anyhow::ensure!(
        combined.argmax.indices.len() == hidden.rows
            && combined.argmax.scores.len() == hidden.rows
            && combined.sampler.indices.len() == hidden.rows
            && combined.sampler.scores.len() == hidden.rows,
        "constrained real lm_head sampler output lengths do not match {} rows",
        hidden.rows
    );
    anyhow::ensure!(
        combined.argmax.scores.iter().all(|score| score.is_finite())
            && combined
                .sampler
                .scores
                .iter()
                .copied()
                .all(valid_sampled_score),
        "constrained real lm_head sampler produced invalid scores: argmax_backend={} sampler_backend={}",
        combined.argmax.backend,
        combined.sampler.backend
    );
    Ok(LmHeadBatchScore {
        top_token_ids: combined.argmax.indices,
        top_logits: combined.argmax.scores,
        sampled_token_ids: combined.sampler.indices,
        sampled_scores: combined.sampler.scores,
        sample_random_uniform: random_uniforms[0],
        sample_temperature: sampler_options.temperature,
        sample_top_k: sampler_options.top_k,
        sample_top_p: sampler_options.top_p,
        argmax_kernel_backend: combined.argmax.backend,
        sampler_kernel_backend: combined.sampler.backend,
    })
}

pub(super) fn score_bf16_lm_head_full_resident_sample_device_input(
    lm_head_name: &str,
    hidden: &DeviceBf16Output,
    full_vocab_size: usize,
) -> Result<LmHeadChunkScore> {
    anyhow::ensure!(
        hidden.rows == 1,
        "real lm_head scalar device-input sampler expected one hidden row, got {}",
        hidden.rows
    );
    let batch = score_bf16_lm_head_full_resident_sample_device_inputs(
        lm_head_name,
        hidden,
        full_vocab_size,
    )?;
    Ok(LmHeadChunkScore {
        logits_evaluated: full_vocab_size,
        top_token_id: batch.top_token_ids[0],
        top_logit: batch.top_logits[0],
        sampled_token_id: batch.sampled_token_ids[0],
        sampled_score: batch.sampled_scores[0],
        sample_random_uniform: batch.sample_random_uniform,
        sample_temperature: batch.sample_temperature,
        sample_top_k: batch.sample_top_k,
        sample_top_p: batch.sample_top_p,
        logits_kernel_backend: batch.argmax_kernel_backend,
        argmax_kernel_backend: batch.argmax_kernel_backend,
        sampler_kernel_backend: batch.sampler_kernel_backend,
    })
}

pub(super) fn score_bf16_lm_head_full_resident_sample_device_input_with_options(
    lm_head_name: &str,
    hidden: &DeviceBf16Output,
    full_vocab_size: usize,
    sampler_options: RealFullLmHeadSamplingOptions,
) -> Result<LmHeadChunkScore> {
    anyhow::ensure!(
        hidden.rows == 1,
        "real lm_head scalar device-input sampler expected one hidden row, got {}",
        hidden.rows
    );
    let batch = score_bf16_lm_head_full_resident_sample_device_inputs_with_options(
        lm_head_name,
        hidden,
        full_vocab_size,
        sampler_options,
        false,
    )?;
    Ok(LmHeadChunkScore {
        logits_evaluated: full_vocab_size,
        top_token_id: batch.top_token_ids[0],
        top_logit: batch.top_logits[0],
        sampled_token_id: batch.sampled_token_ids[0],
        sampled_score: batch.sampled_scores[0],
        sample_random_uniform: batch.sample_random_uniform,
        sample_temperature: batch.sample_temperature,
        sample_top_k: batch.sample_top_k,
        sample_top_p: batch.sample_top_p,
        logits_kernel_backend: batch.argmax_kernel_backend,
        argmax_kernel_backend: batch.argmax_kernel_backend,
        sampler_kernel_backend: batch.sampler_kernel_backend,
    })
}

pub(super) fn score_bf16_lm_head_staged_resident_rows(
    lm_head_window_name: &str,
    hidden_bf16: &[u8],
    start_token_id: usize,
    row_count: usize,
    full_vocab_size: usize,
) -> Result<LmHeadChunkScore> {
    if hidden_bf16.is_empty() || hidden_bf16.len() % 2 != 0 {
        anyhow::bail!(
            "real lm_head staged resident scorer hidden BF16 row must be non-empty and 2-byte aligned"
        );
    }
    let hidden_dim = hidden_bf16.len() / 2;
    let expected_bytes = row_count
        .checked_mul(hidden_bf16.len())
        .context("real lm_head staged resident scorer row byte length overflow")?;
    let end_token_id = start_token_id
        .checked_add(row_count)
        .context("real lm_head staged resident scorer row-window token range overflows usize")?;
    if end_token_id > full_vocab_size {
        anyhow::bail!(
            "real lm_head staged resident scorer row-window [{}, {}) exceeds full vocabulary {}",
            start_token_id,
            end_token_id,
            full_vocab_size
        );
    }
    if !resident_weight_is_preloaded(lm_head_window_name, expected_bytes) {
        anyhow::bail!(
            "real lm_head staged resident scorer requires preloaded row-window weight {lm_head_window_name} with {expected_bytes} bytes"
        );
    }
    let argmax = lm_head_argmax_bf16_staged_resident_weight(
        lm_head_window_name,
        hidden_bf16,
        1,
        hidden_dim,
        row_count,
    )?;
    let sampler_options = GREEDY_SAMPLER_OPTIONS;
    let sampler = lm_head_sample_topk_topp_bf16_staged_resident_weight(
        lm_head_window_name,
        hidden_bf16,
        &[sampler_options.random_uniform],
        1,
        hidden_dim,
        row_count,
        sampler_options.temperature,
        sampler_options.top_k,
        sampler_options.top_p,
    )?;
    let top_index = sampler
        .indices
        .first()
        .copied()
        .context("real lm_head staged resident scorer expected one sampled index")?;
    let argmax_index = argmax
        .indices
        .first()
        .copied()
        .context("real lm_head staged resident scorer expected one argmax index")?;
    if top_index != argmax_index {
        anyhow::bail!(
            "real lm_head staged resident scorer sampled token index {} did not match greedy argmax {} for top_k=1",
            top_index,
            argmax_index
        );
    }
    let top_logit = argmax
        .scores
        .first()
        .copied()
        .context("real lm_head staged resident scorer expected one argmax score")?;
    let sampling_score = sampler
        .scores
        .first()
        .copied()
        .context("real lm_head staged resident scorer expected one sampled score")?;
    if !valid_sampled_score(sampling_score) {
        anyhow::bail!("real lm_head staged resident scorer sampler produced invalid sampled score");
    }
    Ok(LmHeadChunkScore {
        logits_evaluated: row_count,
        top_token_id: start_token_id + argmax_index,
        top_logit,
        sampled_token_id: start_token_id + top_index,
        sampled_score: sampling_score,
        sample_random_uniform: sampler_options.random_uniform,
        sample_temperature: sampler_options.temperature,
        sample_top_k: sampler_options.top_k,
        sample_top_p: sampler_options.top_p,
        logits_kernel_backend: argmax.backend,
        argmax_kernel_backend: argmax.backend,
        sampler_kernel_backend: sampler.backend,
    })
}
pub(super) fn lm_head_full_resident_available(
    lm_head_name: &str,
    full_vocab_size: usize,
    hidden_dim: usize,
) -> bool {
    if !coordinator_cuda_reference_kernels_enabled() {
        return false;
    }
    lm_head_full_resident_bytes(full_vocab_size, hidden_dim)
        .map(|bytes| resident_weight_is_preloaded(lm_head_name, bytes))
        .unwrap_or(false)
}

fn lm_head_full_resident_bytes(full_vocab_size: usize, hidden_dim: usize) -> Result<usize> {
    full_vocab_size
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("real lm_head full resident byte size overflows usize")
}

fn valid_sampled_score(score: f32) -> bool {
    score.is_finite() && score > 0.0 && score <= 1.0
}

#[cfg(test)]
mod tests {
    use super::score_bf16_lm_head_full_resident_sample;
    use super::score_bf16_lm_head_full_resident_sample_device_input;
    use super::score_bf16_lm_head_rows;
    use crate::commands::real_full::coordinator_kernels::{
        coordinator_cuda_reference_kernels_enabled, cuda_reference_kernels_test_override,
        device_bf16_output_from_f32_values, preload_resident_weight_from_host_staging,
        CUDA_REFERENCE_LM_HEAD_ARGMAX_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
        CUDA_REFERENCE_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
        TRITON_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
    };
    use crate::commands::real_full::dense::math::bf16_bytes_from_f32;

    #[test]
    fn real_lm_head_row_scorer_reduces_chunk_without_decoding_full_tensor() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);
        let hidden = bf16_bytes_from_f32(&[1.0, -2.0, 0.5, 3.0]);
        let lm_head = bf16_bytes_from_f32(&[
            0.0, 0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0, //
            0.0, -1.0, 0.0, 0.0, //
            0.0, 0.0, 2.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, //
        ]);

        let result =
            score_bf16_lm_head_rows("lm_head.weight", &hidden, &lm_head, 32, 5, 64).unwrap();

        assert_eq!(result.logits_evaluated, 5);
        assert_eq!(result.top_token_id, 36);
        assert!((result.top_logit - 3.0).abs() < 1.0e-6);
        assert_eq!(
            result.logits_kernel_backend,
            "cpu-reference-lm-head-argmax-bf16"
        );
        assert_eq!(
            result.argmax_kernel_backend,
            "cpu-reference-lm-head-argmax-bf16"
        );
        assert_eq!(
            result.sampler_kernel_backend,
            "cpu-reference-lm-head-sample-topk-topp-bf16"
        );
        assert_eq!(result.sampled_token_id, 36);
        assert!((result.sampled_score - 1.0).abs() < 1.0e-6);
        assert_eq!(result.sample_top_k, 1);
        assert_eq!(result.sample_top_p, 1.0);
        assert_eq!(result.sample_temperature, 1.0);
    }

    #[test]
    fn real_lm_head_full_resident_sampler_scores_full_vocab_when_cuda_enabled() {
        if !coordinator_cuda_reference_kernels_enabled() {
            return;
        }
        let weight_name = format!(
            "test.lm_head.full-resident-sampler.{}.{}",
            std::process::id(),
            line!()
        );
        let hidden = bf16_bytes_from_f32(&[1.0, 0.0]);
        let lm_head = bf16_bytes_from_f32(&[
            1.0, 0.0, //
            0.0, 2.0, //
            3.0, 0.0, //
            2.0, 0.0, //
        ]);
        preload_resident_weight_from_host_staging(
            &weight_name,
            lm_head.len(),
            "test full-resident lm_head sampler",
            |staging| {
                staging.copy_from_slice(&lm_head);
                Ok(())
            },
        )
        .expect("preloading tiny lm_head weight into resident CUDA buffer");

        let result = score_bf16_lm_head_full_resident_sample(&weight_name, &hidden, 4).unwrap();

        assert_eq!(result.logits_evaluated, 4);
        assert_eq!(result.top_token_id, 2);
        assert!((result.top_logit - 3.0).abs() < 1.0e-6);
        assert!(result.sampled_token_id < 4);
        assert!(result.sampled_score.is_finite());
        assert!(result.sampled_score > 0.0 && result.sampled_score <= 1.0);
        assert_eq!(result.sample_top_k, 4);
        assert_eq!(result.sample_top_p, 0.95);
        assert_eq!(result.sample_temperature, 0.7);
        assert_eq!(
            result.argmax_kernel_backend,
            CUDA_REFERENCE_LM_HEAD_ARGMAX_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(
            result.sampler_kernel_backend,
            CUDA_REFERENCE_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
    }

    #[test]
    fn real_lm_head_full_resident_device_input_sampler_scores_full_vocab_when_cuda_enabled() {
        if !coordinator_cuda_reference_kernels_enabled() {
            return;
        }
        let weight_name = format!(
            "test.lm_head.full-resident-device-input-sampler.{}.{}",
            std::process::id(),
            line!()
        );
        let lm_head = bf16_bytes_from_f32(&[
            1.0, 0.0, //
            0.0, 2.0, //
            3.0, 0.0, //
            2.0, 0.0, //
        ]);
        preload_resident_weight_from_host_staging(
            &weight_name,
            lm_head.len(),
            "test full-resident device-input lm_head sampler",
            |staging| {
                staging.copy_from_slice(&lm_head);
                Ok(())
            },
        )
        .expect("preloading tiny lm_head weight into resident CUDA buffer");
        let hidden = device_bf16_output_from_f32_values(
            &[1.0, 0.0],
            1,
            2,
            "test full-resident device-input lm_head hidden",
        )
        .expect("uploading tiny device hidden row");

        let result =
            score_bf16_lm_head_full_resident_sample_device_input(&weight_name, &hidden, 4).unwrap();

        assert_eq!(result.logits_evaluated, 4);
        assert_eq!(result.top_token_id, 2);
        assert!((result.top_logit - 3.0).abs() < 1.0e-6);
        assert!(result.sampled_token_id < 4);
        assert!(result.sampled_score.is_finite());
        assert!(result.sampled_score > 0.0 && result.sampled_score <= 1.0);
        assert_eq!(result.sample_top_k, 4);
        assert_eq!(result.sample_top_p, 0.95);
        assert_eq!(result.sample_temperature, 0.7);
        assert_eq!(
            result.argmax_kernel_backend,
            CUDA_REFERENCE_LM_HEAD_ARGMAX_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        assert!(matches!(
            result.sampler_kernel_backend,
            CUDA_REFERENCE_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
                | TRITON_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        ));
    }
}
