use anyhow::{bail, Context, Result};

use crate::commands::real_full::coordinator_kernels::mla_rope_attention_rows_bf16_for_layer;
use crate::commands::real_full::dense::math::bf16_bytes_from_f32;

#[derive(Debug, Clone, Copy)]
pub(in crate::commands::real_full::attention) struct MlaRopeAttentionF32Shape {
    pub(in crate::commands::real_full::attention) rows: usize,
    pub(in crate::commands::real_full::attention) heads: usize,
    pub(in crate::commands::real_full::attention) nope_dim: usize,
    pub(in crate::commands::real_full::attention) rope_dim: usize,
    pub(in crate::commands::real_full::attention) value_dim: usize,
}

#[derive(Debug)]
pub(in crate::commands::real_full::attention) struct MlaRopeAttentionF32Output {
    pub(in crate::commands::real_full::attention) scores: Vec<f32>,
    pub(in crate::commands::real_full::attention) weights: Vec<f32>,
    pub(in crate::commands::real_full::attention) context_rows: Vec<Vec<f32>>,
    pub(in crate::commands::real_full::attention) context_backend: &'static str,
}

pub(in crate::commands::real_full::attention) fn causal_mla_rope_attention_f32(
    layer_id: usize,
    q_nope: &[f32],
    q_rope: &[f32],
    k_nope: &[f32],
    k_rope: &[f32],
    values: &[f32],
    shape: MlaRopeAttentionF32Shape,
    scale: f32,
) -> Result<MlaRopeAttentionF32Output> {
    validate_mla_rope_attention_inputs(q_nope, q_rope, k_nope, k_rope, values, shape, scale)?;

    let mut all_scores = Vec::with_capacity(shape.heads * shape.rows * (shape.rows + 1) / 2);
    let mut all_weights = Vec::with_capacity(all_scores.capacity());
    for row in 0..shape.rows {
        for head in 0..shape.heads {
            let q_nope_vec = row_head_slice(q_nope, row, head, shape.heads, shape.nope_dim);
            let q_rope_vec = row_head_slice(q_rope, row, head, shape.heads, shape.rope_dim);
            let mut scores = Vec::with_capacity(row + 1);
            for key_row in 0..=row {
                let k_nope_vec = row_head_slice(k_nope, key_row, head, shape.heads, shape.nope_dim);
                let k_rope_vec = row_slice(k_rope, key_row, shape.rope_dim);
                scores.push((dot(q_nope_vec, k_nope_vec)? + dot(q_rope_vec, k_rope_vec)?) * scale);
            }
            let weights = softmax(&scores)?;
            all_scores.extend(scores);
            all_weights.extend(weights);
        }
    }
    let context = mla_rope_attention_rows_bf16_for_layer(
        layer_id,
        &bf16_bytes_from_f32(q_nope),
        &bf16_bytes_from_f32(q_rope),
        &bf16_bytes_from_f32(k_nope),
        &bf16_bytes_from_f32(k_rope),
        &bf16_bytes_from_f32(values),
        shape.rows,
        shape.heads,
        shape.nope_dim,
        shape.rope_dim,
        shape.value_dim,
        scale,
    )?;
    let context_width = shape.heads * shape.value_dim;
    let context_rows = context
        .values
        .chunks_exact(context_width)
        .map(|row| row.to_vec())
        .collect::<Vec<_>>();

    Ok(MlaRopeAttentionF32Output {
        scores: all_scores,
        weights: all_weights,
        context_rows,
        context_backend: context.backend,
    })
}

fn validate_mla_rope_attention_inputs(
    q_nope: &[f32],
    q_rope: &[f32],
    k_nope: &[f32],
    k_rope: &[f32],
    values: &[f32],
    shape: MlaRopeAttentionF32Shape,
    scale: f32,
) -> Result<()> {
    if shape.rows == 0 {
        bail!("MLA/RoPE attention rows must be non-zero");
    }
    if shape.heads == 0 {
        bail!("MLA/RoPE attention heads must be non-zero");
    }
    if shape.nope_dim == 0 {
        bail!("MLA/RoPE attention no-RPE dimension must be non-zero");
    }
    if shape.rope_dim == 0 || shape.rope_dim % 2 != 0 {
        bail!("MLA/RoPE attention RoPE dimension must be positive and even");
    }
    if shape.value_dim == 0 {
        bail!("MLA/RoPE attention value dimension must be non-zero");
    }
    if !scale.is_finite() {
        bail!("MLA/RoPE attention scale must be finite");
    }
    let row_heads = shape
        .rows
        .checked_mul(shape.heads)
        .context("MLA/RoPE attention row-head count overflow")?;
    let nope_values = row_heads
        .checked_mul(shape.nope_dim)
        .context("MLA/RoPE attention no-RPE value count overflow")?;
    let rope_values = row_heads
        .checked_mul(shape.rope_dim)
        .context("MLA/RoPE attention RoPE value count overflow")?;
    let k_rope_values = shape
        .rows
        .checked_mul(shape.rope_dim)
        .context("MLA/RoPE attention shared k-RoPE value count overflow")?;
    let value_values = row_heads
        .checked_mul(shape.value_dim)
        .context("MLA/RoPE attention value count overflow")?;
    if q_nope.len() != nope_values {
        bail!(
            "MLA/RoPE attention q_nope length mismatch: expected {} got {}",
            nope_values,
            q_nope.len()
        );
    }
    if k_nope.len() != nope_values {
        bail!(
            "MLA/RoPE attention k_nope length mismatch: expected {} got {}",
            nope_values,
            k_nope.len()
        );
    }
    if q_rope.len() != rope_values {
        bail!(
            "MLA/RoPE attention q_rope length mismatch: expected {} got {}",
            rope_values,
            q_rope.len()
        );
    }
    if k_rope.len() != k_rope_values {
        bail!(
            "MLA/RoPE attention k_rope length mismatch: expected {} got {}",
            k_rope_values,
            k_rope.len()
        );
    }
    if values.len() != value_values {
        bail!(
            "MLA/RoPE attention value length mismatch: expected {} got {}",
            value_values,
            values.len()
        );
    }
    Ok(())
}

fn row_head_slice(values: &[f32], row: usize, head: usize, heads: usize, width: usize) -> &[f32] {
    let start = (row * heads + head) * width;
    &values[start..start + width]
}

fn row_slice(values: &[f32], row: usize, width: usize) -> &[f32] {
    let start = row * width;
    &values[start..start + width]
}

fn dot(lhs: &[f32], rhs: &[f32]) -> Result<f32> {
    if lhs.len() != rhs.len() {
        bail!(
            "attention MLA/RoPE dot length mismatch: lhs={} rhs={}",
            lhs.len(),
            rhs.len()
        );
    }
    Ok(lhs.iter().zip(rhs.iter()).map(|(lhs, rhs)| lhs * rhs).sum())
}

fn softmax(scores: &[f32]) -> Result<Vec<f32>> {
    if scores.is_empty() {
        bail!("attention MLA/RoPE softmax requires at least one score");
    }
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut weights = scores
        .iter()
        .map(|score| (score - max_score).exp())
        .collect::<Vec<_>>();
    let sum = weights.iter().sum::<f32>();
    if sum == 0.0 {
        bail!("attention MLA/RoPE softmax weight sum is zero");
    }
    for weight in &mut weights {
        *weight /= sum;
    }
    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::{causal_mla_rope_attention_f32, MlaRopeAttentionF32Shape};
    use crate::commands::real_full::coordinator_kernels::cuda_reference_kernels_test_override;

    #[test]
    fn mla_rope_attention_composes_nope_and_shared_rope_scores() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);
        let shape = MlaRopeAttentionF32Shape {
            rows: 3,
            heads: 2,
            nope_dim: 2,
            rope_dim: 2,
            value_dim: 2,
        };
        let scale = 0.5_f32;
        let q_nope = [
            0.10_f32, -0.20, 0.30, 0.05, -0.40, 0.25, 0.15, -0.35, 0.50, 0.10, -0.25, 0.45,
        ];
        let q_rope = [
            -0.20_f32, 0.45, 0.15, -0.35, 0.30, -0.15, -0.55, 0.20, 0.55, 0.20, -0.40, 0.30,
        ];
        let k_nope = [
            0.25_f32, 0.15, -0.10, 0.40, 0.35, -0.45, 0.60, 0.20, -0.30, 0.50, 0.45, -0.15,
        ];
        let k_rope = [0.10_f32, 0.50, 0.35, -0.25, -0.40, 0.30];
        let values = [
            0.10_f32, 0.20, -0.40, -0.50, 0.70, 0.80, 1.00, -1.10, -0.20, 0.40, 0.30, -0.70,
        ];

        let output = causal_mla_rope_attention_f32(
            3, &q_nope, &q_rope, &k_nope, &k_rope, &values, shape, scale,
        )
        .unwrap();

        assert_eq!(
            output.scores.len(),
            shape.heads * shape.rows * (shape.rows + 1) / 2
        );
        assert_eq!(output.weights.len(), output.scores.len());
        assert_eq!(output.context_rows.len(), shape.rows);
        assert!(output
            .context_rows
            .iter()
            .all(|row| row.len() == shape.heads * shape.value_dim));
        assert_eq!(
            output.context_backend,
            "cpu-reference-mla-rope-attention-bf16"
        );
        assert!((output.context_rows[0][0] - values[0]).abs() < 2.0e-3);
        assert!((output.context_rows[0][2] - values[2]).abs() < 2.0e-3);
        for weights in output.weights.chunks(1).take(shape.heads) {
            assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1.0e-6);
        }
    }

    #[test]
    fn mla_rope_attention_rejects_missing_shared_rope_rows() {
        let shape = MlaRopeAttentionF32Shape {
            rows: 2,
            heads: 1,
            nope_dim: 1,
            rope_dim: 2,
            value_dim: 1,
        };
        let err = causal_mla_rope_attention_f32(
            3,
            &[1.0, 2.0],
            &[0.1, 0.2, 0.3, 0.4],
            &[1.0, 2.0],
            &[0.1, 0.2],
            &[3.0, 4.0],
            shape,
            1.0,
        )
        .unwrap_err();

        assert!(err.to_string().contains("k_rope length mismatch"));
    }
}
