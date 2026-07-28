use anyhow::{Context, Result};
use glmrt_core::{DType, TensorCatalog, TensorRole, GLM52_HIDDEN_BF16_BYTES, GLM52_HIDDEN_SIZE};

use super::types::{RealFullExecutionPlan, RealFullSamplingDryRun};

mod lm_head;

use lm_head::{real_lm_head_default_chunk_probe, real_lm_head_full_vocab_probe};
pub(in crate::commands::real_full) use lm_head::{
    score_real_lm_head_chunk_for_hidden, score_real_lm_head_full_vocab_for_device_hidden,
    score_real_lm_head_full_vocab_for_device_hidden_rows,
    score_real_lm_head_full_vocab_for_device_hidden_rows_with_options,
    score_real_lm_head_full_vocab_for_device_hidden_with_options,
    score_real_lm_head_full_vocab_for_hidden, RealFullLmHeadSamplingOptions,
    RealLmHeadBatchScoreForHidden, RealLmHeadChunkScoreForHidden,
};

const REAL_FULL_SAMPLING_CHUNK_ROWS: usize = 1024;
pub(super) const REAL_FULL_SCORE_LM_HEAD_ENV: &str = "GLMRT_REAL_FULL_SCORE_LM_HEAD";

pub(super) fn real_full_sampling_dry_run(
    catalog: &TensorCatalog,
    execution_plan: &RealFullExecutionPlan,
) -> Result<RealFullSamplingDryRun> {
    let lm_head = catalog
        .tensors
        .iter()
        .find(|tensor| tensor.role == TensorRole::LmHead)
        .context("full-vocab sampling dry-run requires lm_head.weight in the catalog")?;
    if lm_head.dtype != DType::Bf16 {
        anyhow::bail!(
            "full-vocab sampling dry-run expects BF16 lm_head, found {:?}",
            lm_head.dtype
        );
    }
    if lm_head.shape.len() != 2 {
        anyhow::bail!(
            "full-vocab sampling dry-run expects 2D lm_head, found shape {:?}",
            lm_head.shape
        );
    }
    let vocab_size = lm_head.shape[0];
    let hidden_dim = lm_head.shape[1];
    if vocab_size == 0 {
        anyhow::bail!("full-vocab sampling dry-run requires non-empty lm_head vocabulary");
    }
    if hidden_dim != GLM52_HIDDEN_SIZE {
        anyhow::bail!(
            "full-vocab sampling dry-run hidden dim mismatch: expected {} got {}",
            GLM52_HIDDEN_SIZE,
            hidden_dim
        );
    }
    let expected_lm_head_bytes = (vocab_size * GLM52_HIDDEN_BF16_BYTES) as u64;
    if lm_head.byte_length != expected_lm_head_bytes {
        anyhow::bail!(
            "full-vocab sampling dry-run lm_head bytes mismatch: expected {} got {}",
            expected_lm_head_bytes,
            lm_head.byte_length
        );
    }

    let sampled_rows = execution_plan.decode_rows.max(1);
    let chunk_rows = REAL_FULL_SAMPLING_CHUNK_ROWS;
    let chunk_count = vocab_size.div_ceil(chunk_rows);
    let final_chunk_rows = vocab_size - ((chunk_count - 1) * chunk_rows);
    let max_chunk_bytes = chunk_rows * GLM52_HIDDEN_BF16_BYTES;
    let logical_lm_head_read_bytes = lm_head.byte_length * sampled_rows as u64;
    let logical_hidden_read_bytes = sampled_rows * GLM52_HIDDEN_BF16_BYTES;
    let logical_logit_bytes = sampled_rows * vocab_size * std::mem::size_of::<f32>();
    let dot_products = sampled_rows * vocab_size;
    let multiply_accumulate_ops = dot_products as u64 * hidden_dim as u64;
    let real_lm_head_default_chunk_probe =
        real_lm_head_default_chunk_probe(catalog, lm_head, chunk_rows);
    let real_lm_head_full_vocab_probe = real_lm_head_full_vocab_probe(catalog, lm_head, chunk_rows);
    let status = if real_lm_head_full_vocab_probe.passed {
        "dry-run-plus-real-full-vocab-lm-head-probe"
    } else {
        "dry-run-only"
    };

    Ok(RealFullSamplingDryRun {
        status,
        scope: "plan real full-vocabulary lm_head scoring through BF16 coordinator kernel wrappers",
        lm_head_tensor: lm_head.name.clone(),
        hidden_dim,
        vocab_size,
        sampled_rows,
        hidden_dtype: "bf16",
        output_logit_dtype: "f32",
        lm_head_bytes: lm_head.byte_length,
        hidden_bytes_per_row: GLM52_HIDDEN_BF16_BYTES,
        chunk_rows,
        chunk_count,
        final_chunk_rows,
        max_chunk_bytes,
        logical_lm_head_read_bytes,
        logical_hidden_read_bytes,
        logical_logit_bytes,
        dot_products,
        multiply_accumulate_ops,
        greedy_reduce_chunks: chunk_count,
        real_lm_head_default_chunk_probe,
        real_lm_head_full_vocab_probe,
        covers_full_vocabulary: chunk_count * chunk_rows >= vocab_size && final_chunk_rows > 0,
        requires_numeric_logits: true,
    })
}
