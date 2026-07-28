use anyhow::Result;
use glmrt_core::{TensorCatalog, TensorInfo};

use super::score_bf16_lm_head_rows_from_catalog;
use crate::commands::real_full::dense::math::bf16_bytes_from_f32;
use crate::commands::real_full::sampling::REAL_FULL_SCORE_LM_HEAD_ENV;
use crate::commands::real_full::types::{
    RealFullSamplingLmHeadChunkProbe, RealFullSamplingRealLmHeadProbe,
};

const DIAGNOSTIC_HIDDEN_SOURCE: &str = "deterministic-terminal-residual-shaped-bf16-row";

pub(super) fn run_real_lm_head_default_chunk_probe(
    catalog: &TensorCatalog,
    lm_head: &TensorInfo,
    chunk_rows: usize,
) -> Result<RealFullSamplingLmHeadChunkProbe> {
    if chunk_rows == 0 {
        anyhow::bail!("real lm_head default chunk probe requires non-zero chunk_rows");
    }
    let vocab_size = lm_head.shape[0];
    let hidden_dim = lm_head.shape[1];
    let rows_in_chunk = vocab_size.min(chunk_rows);
    let hidden = deterministic_terminal_hidden_bf16(hidden_dim);
    let (chunk_result, lm_head_bytes_read) = score_bf16_lm_head_rows_from_catalog(
        catalog,
        lm_head,
        &hidden,
        0,
        rows_in_chunk,
        vocab_size,
    )?;
    let multiply_accumulate_ops = chunk_result.logits_evaluated as u64 * hidden_dim as u64;
    let expected_lm_head_bytes_read = (rows_in_chunk * hidden_dim * 2) as u64;
    let passed = rows_in_chunk > 0
        && chunk_result.logits_evaluated == rows_in_chunk
        && (lm_head_bytes_read == 0 || lm_head_bytes_read == expected_lm_head_bytes_read)
        && chunk_result.top_logit.is_finite();

    Ok(RealFullSamplingLmHeadChunkProbe {
        status: "numeric-real-lm-head-default-chunk",
        scope: "score the first default-sized real lm_head.weight chunk against a deterministic diagnostic hidden row",
        hidden_source: DIAGNOSTIC_HIDDEN_SOURCE,
        uses_real_lm_head: true,
        uses_full_model_residual: false,
        uses_real_dense_prefix: false,
        hidden_dim,
        vocab_size,
        start_token_id: 0,
        chunk_rows,
        rows_scored: rows_in_chunk,
        chunks_scored: 1,
        lm_head_bytes_read,
        hidden_bytes: hidden.len(),
        logits_evaluated: chunk_result.logits_evaluated,
        multiply_accumulate_ops,
        logits_kernel_backend: Some(chunk_result.logits_kernel_backend),
        argmax_kernel_backend: Some(chunk_result.argmax_kernel_backend),
        sampler_kernel_backend: Some(chunk_result.sampler_kernel_backend),
        top_token_id: Some(chunk_result.top_token_id),
        top_logit: Some(chunk_result.top_logit),
        sampled_token_id: Some(chunk_result.sampled_token_id),
        sampled_score: Some(chunk_result.sampled_score),
        sample_random_uniform: Some(chunk_result.sample_random_uniform),
        sample_temperature: Some(chunk_result.sample_temperature),
        sample_top_k: Some(chunk_result.sample_top_k),
        sample_top_p: Some(chunk_result.sample_top_p),
        residual_source_dense_layers: 0,
        residual_source_dense_residual_adds: 0,
        residual_source_dense_weight_bytes_read: 0,
        residual_source_covers_all_dense_layers: false,
        residual_after_checksum: None,
        passed,
        error: None,
    })
}

pub(super) fn run_real_lm_head_full_vocab_probe(
    catalog: &TensorCatalog,
    lm_head: &TensorInfo,
    chunk_rows: usize,
) -> Result<RealFullSamplingRealLmHeadProbe> {
    if chunk_rows == 0 {
        anyhow::bail!("real lm_head full-vocab probe requires non-zero chunk_rows");
    }
    let vocab_size = lm_head.shape[0];
    let hidden_dim = lm_head.shape[1];
    let hidden = deterministic_terminal_hidden_bf16(hidden_dim);
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
        let rows_in_chunk = (vocab_size - chunk_start).min(chunk_rows);
        let (chunk_result, chunk_bytes_read) = score_bf16_lm_head_rows_from_catalog(
            catalog,
            lm_head,
            &hidden,
            chunk_start,
            rows_in_chunk,
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

    let chunk_count = vocab_size.div_ceil(chunk_rows);
    let final_chunk_rows = vocab_size - ((chunk_count - 1) * chunk_rows);
    let multiply_accumulate_ops = logits_evaluated as u64 * hidden_dim as u64;
    let passed = chunks_scored == chunk_count
        && logits_evaluated == vocab_size
        && (lm_head_bytes_read == 0 || lm_head_bytes_read == lm_head.byte_length)
        && top_logit.is_finite();

    Ok(RealFullSamplingRealLmHeadProbe {
        status: "numeric-real-lm-head",
        scope: "stream-score real lm_head.weight rows against a terminal hidden row",
        opt_in_env: REAL_FULL_SCORE_LM_HEAD_ENV,
        hidden_source: DIAGNOSTIC_HIDDEN_SOURCE,
        uses_real_lm_head: true,
        uses_full_model_residual: false,
        hidden_dim,
        vocab_size,
        chunk_rows,
        chunk_count,
        final_chunk_rows,
        chunks_scored,
        lm_head_bytes_read,
        hidden_bytes: hidden.len(),
        logits_evaluated,
        multiply_accumulate_ops,
        logits_kernel_backend,
        argmax_kernel_backend,
        sampler_kernel_backend,
        top_token_id: Some(top_token_id),
        top_logit: Some(top_logit),
        sampled_token_id: Some(sampled_token_id),
        sampled_score: Some(sampled_score),
        sample_random_uniform: Some(sample_random_uniform),
        sample_temperature: Some(sample_temperature),
        sample_top_k: Some(sample_top_k),
        sample_top_p: Some(sample_top_p),
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
        passed,
        skipped_reason: None,
    })
}

fn deterministic_terminal_hidden_bf16(hidden_dim: usize) -> Vec<u8> {
    let values = (0..hidden_dim)
        .map(|idx| {
            let base = ((idx % 31) as f32 - 15.0) / 64.0;
            if idx % 127 == 0 {
                base + 0.5
            } else {
                base
            }
        })
        .collect::<Vec<_>>();
    bf16_bytes_from_f32(&values)
}

#[cfg(test)]
mod tests {
    use super::run_real_lm_head_full_vocab_probe;
    use crate::commands::real_full::dense::math::bf16_bytes_from_f32;
    use glmrt_core::{DType, ModelFacts, TensorCatalog, TensorInfo, TensorRole};
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn real_lm_head_full_vocab_probe_streams_tiny_catalog_chunks() {
        let tempdir = tempfile::tempdir().unwrap();
        let shard_path = tempdir.path().join("lm_head.safetensors");
        let lm_head_bytes = bf16_bytes_from_f32(&[
            0.0, 0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0, //
            0.0, -1.0, 0.0, 0.0, //
            0.0, 0.0, -4.0, 0.0, //
            0.0, 0.0, 0.0, -6.0, //
        ]);
        File::create(&shard_path)
            .unwrap()
            .write_all(&lm_head_bytes)
            .unwrap();
        let lm_head = TensorInfo {
            name: "lm_head.weight".to_owned(),
            file: "lm_head.safetensors".to_owned(),
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

        let probe = run_real_lm_head_full_vocab_probe(&catalog, &lm_head, 2).unwrap();

        assert_eq!(probe.status, "numeric-real-lm-head");
        assert_eq!(
            probe.hidden_source,
            "deterministic-terminal-residual-shaped-bf16-row"
        );
        assert!(probe.uses_real_lm_head);
        assert!(!probe.uses_full_model_residual);
        assert_eq!(probe.hidden_dim, 4);
        assert_eq!(probe.vocab_size, 5);
        assert_eq!(probe.chunk_rows, 2);
        assert_eq!(probe.chunk_count, 3);
        assert_eq!(probe.final_chunk_rows, 1);
        assert_eq!(probe.chunks_scored, 3);
        assert_eq!(probe.lm_head_bytes_read, lm_head_bytes.len() as u64);
        assert_eq!(probe.logits_evaluated, 5);
        assert_eq!(probe.multiply_accumulate_ops, 20);
        assert_eq!(probe.top_token_id, Some(4));
        assert!((probe.top_logit.unwrap() - 1.125).abs() < 1.0e-6);
        assert!(probe.passed);
    }
}
