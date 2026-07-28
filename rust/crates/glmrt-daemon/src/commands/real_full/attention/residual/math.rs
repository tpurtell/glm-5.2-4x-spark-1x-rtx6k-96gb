use anyhow::Result;
use glmrt_core::GLM52_HIDDEN_SIZE;

use crate::commands::real_full::coordinator_kernels::{
    coordinator_cuda_reference_kernels_enabled, linear_rows, linear_rows_bf16,
    linear_rows_bf16_preloaded_resident_weight, linear_rows_bf16_resident_weight,
    resident_weight_is_preloaded, rmsnorm_hidden_bf16_preloaded_resident_weight,
    rmsnorm_hidden_bf16_resident_weight, rope_rows_bf16_for_layer, RmsNormOutput,
    CUDA_REFERENCE_LINEAR_BF16_RESIDENT_WEIGHT_BACKEND,
};
use crate::commands::real_full::dense::math::bf16_bytes_from_f32;

use super::super::super::dense::math::deterministic_dense_hidden;

#[derive(Debug)]
pub(super) struct ProjectedRows {
    pub(super) values: Vec<f32>,
    pub(super) backend: &'static str,
}

pub(super) struct RopeRow {
    pub(super) values: Vec<f32>,
    pub(super) backend: &'static str,
}

pub(super) fn deterministic_attention_hidden_rows() -> Vec<Vec<f32>> {
    let first = deterministic_dense_hidden(GLM52_HIDDEN_SIZE);
    let mut second = first.clone();
    for (idx, value) in second.iter_mut().enumerate() {
        *value += ((idx % 11) as f32 - 5.0) / 4096.0;
        if idx % 97 == 0 {
            *value -= 0.03125;
        }
    }
    vec![first, second]
}

pub(super) fn softmax_weights(scores: &[f32]) -> Result<Vec<f32>> {
    if scores.is_empty() {
        anyhow::bail!("real full attention residual-prefix softmax requires scores");
    }
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut weights = scores
        .iter()
        .map(|score| (score - max_score).exp())
        .collect::<Vec<_>>();
    let sum = weights.iter().sum::<f32>();
    if sum == 0.0 || !sum.is_finite() {
        anyhow::bail!("real full attention residual-prefix softmax produced invalid sum");
    }
    for weight in &mut weights {
        *weight /= sum;
    }
    Ok(weights)
}

pub(super) fn apply_rope_row_with_backend(
    layer_id: usize,
    row: &[f32],
    position: usize,
    theta: f64,
) -> Result<RopeRow> {
    if row.is_empty() || row.len() % 2 != 0 {
        anyhow::bail!(
            "real full attention RoPE row must have a positive even width, got {}",
            row.len()
        );
    }
    let output = rope_rows_bf16_for_layer(
        layer_id,
        &bf16_bytes_from_f32(row),
        &[position],
        1,
        1,
        row.len(),
        theta as f32,
    )?;
    Ok(RopeRow {
        values: output.values,
        backend: output.backend,
    })
}

pub(super) fn flatten_rows(rows: &[Vec<f32>]) -> Vec<f32> {
    rows.iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>()
}

#[allow(dead_code)]
pub(super) fn project_rows(
    hidden: &[f32],
    values: &[f32],
    row_count: usize,
    row_width: usize,
) -> Result<Vec<f32>> {
    Ok(project_rows_with_backend(hidden, values, row_count, row_width)?.values)
}

#[allow(dead_code)]
pub(super) fn project_rows_with_backend(
    hidden: &[f32],
    values: &[f32],
    row_count: usize,
    row_width: usize,
) -> Result<ProjectedRows> {
    if hidden.len() != row_width {
        anyhow::bail!(
            "real full attention residual-prefix projection hidden width mismatch: hidden={} row_width={row_width}",
            hidden.len()
        );
    }
    if values.len() != row_count * row_width {
        anyhow::bail!(
            "real full attention residual-prefix projection value length mismatch: values={} expected={}",
            values.len(),
            row_count * row_width
        );
    }
    let projected = linear_rows(hidden, values, None, 1, row_width, row_count)?;
    Ok(ProjectedRows {
        values: projected.values,
        backend: projected.backend,
    })
}

#[allow(dead_code)]
pub(super) fn project_rows_bf16(
    hidden: &[f32],
    values_bf16: &[u8],
    row_count: usize,
    row_width: usize,
) -> Result<Vec<f32>> {
    Ok(project_rows_bf16_with_backend(hidden, values_bf16, row_count, row_width)?.values)
}

#[allow(dead_code)]
pub(super) fn project_rows_bf16_with_backend(
    hidden: &[f32],
    values_bf16: &[u8],
    row_count: usize,
    row_width: usize,
) -> Result<ProjectedRows> {
    project_rows_bf16_with_backend_impl(None, hidden, values_bf16, row_count, row_width)
}

pub(super) fn project_rows_bf16_with_optional_preloaded_full_weight(
    full_weight_name: &str,
    hidden: &[f32],
    values_bf16: Option<&[u8]>,
    row_count: usize,
    row_width: usize,
) -> Result<ProjectedRows> {
    if hidden.len() != row_width {
        anyhow::bail!(
            "real full BF16 attention projection hidden width mismatch: hidden={} row_width={row_width}",
            hidden.len()
        );
    }
    let hidden_bf16 = bf16_bytes_from_f32(hidden);
    if bf16_full_row_prefix_resident_available(full_weight_name, row_count, row_width, row_count) {
        let projected = linear_rows_bf16_preloaded_resident_weight(
            full_weight_name,
            &hidden_bf16,
            None,
            1,
            row_width,
            row_count,
            row_count,
        )?;
        return Ok(ProjectedRows {
            values: projected.values,
            backend: CUDA_REFERENCE_LINEAR_BF16_RESIDENT_WEIGHT_BACKEND,
        });
    }
    let values_bf16 = values_bf16.ok_or_else(|| {
        anyhow::anyhow!(
            "real full BF16 attention projection full weight bytes missing for {full_weight_name}"
        )
    })?;
    project_rows_bf16_with_backend_impl(
        Some(full_weight_name),
        hidden,
        values_bf16,
        row_count,
        row_width,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn project_rows_bf16_with_optional_preloaded_prefix_weight(
    full_weight_name: &str,
    row_window_weight_name: &str,
    hidden: &[f32],
    values_bf16: Option<&[u8]>,
    row_count: usize,
    row_width: usize,
    full_row_count: usize,
) -> Result<ProjectedRows> {
    if hidden.len() != row_width {
        anyhow::bail!(
            "real full BF16 attention projection hidden width mismatch: hidden={} row_width={row_width}",
            hidden.len()
        );
    }
    let hidden_bf16 = bf16_bytes_from_f32(hidden);
    if bf16_full_row_prefix_resident_available(
        full_weight_name,
        full_row_count,
        row_width,
        row_count,
    ) {
        let projected = linear_rows_bf16_preloaded_resident_weight(
            full_weight_name,
            &hidden_bf16,
            None,
            1,
            row_width,
            row_count,
            full_row_count,
        )?;
        return Ok(ProjectedRows {
            values: projected.values,
            backend: CUDA_REFERENCE_LINEAR_BF16_RESIDENT_WEIGHT_BACKEND,
        });
    }
    if bf16_full_row_prefix_resident_available(
        row_window_weight_name,
        row_count,
        row_width,
        row_count,
    ) {
        let projected = linear_rows_bf16_preloaded_resident_weight(
            row_window_weight_name,
            &hidden_bf16,
            None,
            1,
            row_width,
            row_count,
            row_count,
        )?;
        return Ok(ProjectedRows {
            values: projected.values,
            backend: CUDA_REFERENCE_LINEAR_BF16_RESIDENT_WEIGHT_BACKEND,
        });
    }
    let values_bf16 = values_bf16.ok_or_else(|| {
        anyhow::anyhow!(
            "real full BF16 attention projection row-window bytes missing for {row_window_weight_name}"
        )
    })?;
    project_rows_bf16_with_backend_impl(
        Some(row_window_weight_name),
        hidden,
        values_bf16,
        row_count,
        row_width,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn project_rows_bf16_with_optional_padded_preloaded_prefix_weight(
    full_weight_name: &str,
    row_window_weight_name: &str,
    hidden: &[f32],
    values_bf16: Option<&[u8]>,
    row_count: usize,
    active_row_width: usize,
    full_row_width: usize,
    full_row_count: usize,
) -> Result<ProjectedRows> {
    if hidden.len() != active_row_width {
        anyhow::bail!(
            "real full BF16 attention projection hidden width mismatch: hidden={} active_row_width={active_row_width}",
            hidden.len()
        );
    }
    if active_row_width == 0 || full_row_width == 0 || active_row_width > full_row_width {
        anyhow::bail!(
            "real full BF16 attention projection invalid row widths: active={active_row_width} full={full_row_width}"
        );
    }
    if bf16_full_row_prefix_resident_available(
        full_weight_name,
        full_row_count,
        full_row_width,
        row_count,
    ) {
        let mut padded_hidden = vec![0.0_f32; full_row_width];
        padded_hidden[..active_row_width].copy_from_slice(hidden);
        let hidden_bf16 = bf16_bytes_from_f32(&padded_hidden);
        let projected = linear_rows_bf16_preloaded_resident_weight(
            full_weight_name,
            &hidden_bf16,
            None,
            1,
            full_row_width,
            row_count,
            full_row_count,
        )?;
        return Ok(ProjectedRows {
            values: projected.values,
            backend: CUDA_REFERENCE_LINEAR_BF16_RESIDENT_WEIGHT_BACKEND,
        });
    }
    if bf16_full_row_prefix_resident_available(
        row_window_weight_name,
        row_count,
        active_row_width,
        row_count,
    ) {
        let hidden_bf16 = bf16_bytes_from_f32(hidden);
        let projected = linear_rows_bf16_preloaded_resident_weight(
            row_window_weight_name,
            &hidden_bf16,
            None,
            1,
            active_row_width,
            row_count,
            row_count,
        )?;
        return Ok(ProjectedRows {
            values: projected.values,
            backend: CUDA_REFERENCE_LINEAR_BF16_RESIDENT_WEIGHT_BACKEND,
        });
    }
    let values_bf16 = values_bf16.ok_or_else(|| {
        anyhow::anyhow!(
            "real full BF16 attention projection row-window bytes missing for {row_window_weight_name}"
        )
    })?;
    project_rows_bf16_with_backend_impl(
        Some(row_window_weight_name),
        hidden,
        values_bf16,
        row_count,
        active_row_width,
    )
}

pub(super) fn bf16_full_row_prefix_resident_available(
    weight_name: &str,
    full_row_count: usize,
    row_width: usize,
    prefix_rows: usize,
) -> bool {
    if !coordinator_cuda_reference_kernels_enabled()
        || full_row_count == 0
        || row_width == 0
        || prefix_rows == 0
        || prefix_rows > full_row_count
    {
        return false;
    }
    let Some(full_bytes) = full_row_count
        .checked_mul(row_width)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
    else {
        return false;
    };
    resident_weight_is_preloaded(weight_name, full_bytes)
}

pub(super) fn rmsnorm_bf16_with_optional_preloaded_resident_weight(
    weight_name: &str,
    hidden_bf16: &[u8],
    weight_bf16: Option<&[u8]>,
    rows: usize,
    hidden_dim: usize,
    eps: f32,
) -> Result<RmsNormOutput> {
    if bf16_full_vector_resident_available(weight_name, hidden_dim) {
        return rmsnorm_hidden_bf16_preloaded_resident_weight(
            weight_name,
            hidden_bf16,
            rows,
            hidden_dim,
            eps,
        );
    }
    let weight_bf16 = weight_bf16.ok_or_else(|| {
        anyhow::anyhow!("real full BF16 RMSNorm weight bytes missing for {weight_name}")
    })?;
    rmsnorm_hidden_bf16_resident_weight(
        weight_name,
        hidden_bf16,
        weight_bf16,
        rows,
        hidden_dim,
        eps,
    )
}

pub(super) fn bf16_full_vector_resident_available(weight_name: &str, values: usize) -> bool {
    if !coordinator_cuda_reference_kernels_enabled() || values == 0 {
        return false;
    }
    let Some(bytes) = values.checked_mul(std::mem::size_of::<u16>()) else {
        return false;
    };
    resident_weight_is_preloaded(weight_name, bytes)
}

fn project_rows_bf16_with_backend_impl(
    weight_name: Option<&str>,
    hidden: &[f32],
    values_bf16: &[u8],
    row_count: usize,
    row_width: usize,
) -> Result<ProjectedRows> {
    if hidden.len() != row_width {
        anyhow::bail!(
            "real full BF16 attention projection hidden width mismatch: hidden={} row_width={row_width}",
            hidden.len()
        );
    }
    let expected_bytes = row_count
        .checked_mul(row_width)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "real full BF16 attention projection shape overflows usize while validating input"
            )
        })?;
    if values_bf16.len() != expected_bytes {
        anyhow::bail!(
            "real full BF16 attention projection byte length mismatch: values={} expected={}",
            values_bf16.len(),
            expected_bytes
        );
    }
    let hidden_bf16 = bf16_bytes_from_f32(hidden);
    let projected = if let Some(weight_name) = weight_name {
        linear_rows_bf16_resident_weight(
            weight_name,
            &hidden_bf16,
            values_bf16,
            None,
            1,
            row_width,
            row_count,
        )?
    } else {
        linear_rows_bf16(&hidden_bf16, values_bf16, None, 1, row_width, row_count)?
    };
    Ok(ProjectedRows {
        values: projected.values,
        backend: projected.backend,
    })
}

pub(super) fn compact_row_prefix_bytes(
    values: &[u8],
    row_count: usize,
    row_stride: usize,
    prefix_width: usize,
) -> Result<Vec<u8>> {
    if row_count == 0 || row_stride == 0 || prefix_width == 0 {
        anyhow::bail!(
            "real full BF16 attention projection compaction requires non-zero shape, got rows={row_count} row_stride={row_stride} prefix_width={prefix_width}"
        );
    }
    if prefix_width > row_stride {
        anyhow::bail!(
            "real full BF16 attention projection prefix width {prefix_width} exceeds row stride {row_stride}"
        );
    }
    let expected_bytes = row_count
        .checked_mul(row_stride)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "real full BF16 attention projection compaction shape overflows usize while validating input"
            )
        })?;
    if values.len() != expected_bytes {
        anyhow::bail!(
            "real full BF16 attention projection compaction byte length mismatch: values={} expected={}",
            values.len(),
            expected_bytes
        );
    }
    let mut compacted = Vec::with_capacity(row_count * prefix_width * std::mem::size_of::<u16>());
    for row_index in 0..row_count {
        let start = row_index * row_stride * std::mem::size_of::<u16>();
        let end = start + prefix_width * std::mem::size_of::<u16>();
        compacted.extend_from_slice(&values[start..end]);
    }
    Ok(compacted)
}

#[allow(dead_code)]
pub(super) fn compact_row_prefixes(
    values: &[f32],
    row_count: usize,
    row_stride: usize,
    prefix_width: usize,
) -> Result<Vec<f32>> {
    if row_count == 0 || row_stride == 0 || prefix_width == 0 {
        anyhow::bail!(
            "real full attention projection compaction requires non-zero shape, got rows={row_count} row_stride={row_stride} prefix_width={prefix_width}"
        );
    }
    if prefix_width > row_stride {
        anyhow::bail!(
            "real full attention projection prefix width {prefix_width} exceeds row stride {row_stride}"
        );
    }
    if values.len() != row_count * row_stride {
        anyhow::bail!(
            "real full attention projection compaction length mismatch: values={} expected={}",
            values.len(),
            row_count * row_stride
        );
    }
    let mut compacted = Vec::with_capacity(row_count * prefix_width);
    for row_index in 0..row_count {
        let start = row_index * row_stride;
        compacted.extend_from_slice(&values[start..start + prefix_width]);
    }
    Ok(compacted)
}

#[cfg(test)]
mod tests {
    use super::{
        compact_row_prefix_bytes, compact_row_prefixes, project_rows_bf16_with_backend,
        project_rows_bf16_with_optional_padded_preloaded_prefix_weight,
        project_rows_bf16_with_optional_preloaded_full_weight, project_rows_with_backend,
        rmsnorm_bf16_with_optional_preloaded_resident_weight,
    };
    use crate::commands::real_full::coordinator_kernels::cuda_reference_kernels_test_override;

    #[test]
    fn project_rows_uses_coordinator_linear_backend() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let projected =
            project_rows_with_backend(&[1.0, 2.0], &[0.5, 1.0, -1.0, 2.0], 2, 2).unwrap();

        assert_eq!(projected.values, vec![2.5, 3.0]);
        assert_eq!(projected.backend, "cpu-reference-linear");
    }

    #[test]
    fn compact_row_prefixes_keeps_row_order() {
        let compacted = compact_row_prefixes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3, 2).unwrap();

        assert_eq!(compacted, vec![1.0, 2.0, 4.0, 5.0]);
    }

    #[test]
    fn project_rows_bf16_uses_coordinator_linear_backend() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let projected =
            project_rows_bf16_with_backend(&[1.0, 2.0], &bf16_bytes(&[0.5, 1.0, -1.0, 2.0]), 2, 2)
                .unwrap();

        assert_eq!(projected.values, vec![2.5, 3.0]);
        assert_eq!(projected.backend, "cpu-reference-linear-bf16");
    }

    #[test]
    fn padded_preloaded_prefix_projection_falls_back_to_compact_row_window() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let projected = project_rows_bf16_with_optional_padded_preloaded_prefix_weight(
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.self_attn.o_proj.weight[rows=0..2,cols=0..2]",
            &[2.0, 3.0],
            Some(&bf16_bytes(&[1.0, 0.5, -1.0, 2.0])),
            2,
            2,
            4,
            8,
        )
        .unwrap();

        assert_eq!(projected.values, vec![3.5, 4.0]);
        assert_eq!(projected.backend, "cpu-reference-linear-bf16");
    }

    #[test]
    fn optional_preloaded_full_projection_falls_back_to_loaded_weight() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let projected = project_rows_bf16_with_optional_preloaded_full_weight(
            "model.layers.0.self_attn.q_a_proj.weight",
            &[1.0, 2.0],
            Some(&bf16_bytes(&[0.5, 1.0, -1.0, 2.0])),
            2,
            2,
        )
        .unwrap();

        assert_eq!(projected.values, vec![2.5, 3.0]);
        assert_eq!(projected.backend, "cpu-reference-linear-bf16");
    }

    #[test]
    fn optional_preloaded_full_projection_requires_bytes_without_resident_weight() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let err = project_rows_bf16_with_optional_preloaded_full_weight(
            "model.layers.0.self_attn.q_a_proj.weight",
            &[1.0, 2.0],
            None,
            2,
            2,
        )
        .unwrap_err();

        assert!(err.to_string().contains("full weight bytes missing"));
    }

    #[test]
    fn compact_row_prefix_bytes_keeps_row_order() {
        let compacted =
            compact_row_prefix_bytes(&bf16_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 2, 3, 2)
                .unwrap();

        assert_eq!(compacted, bf16_bytes(&[1.0, 2.0, 4.0, 5.0]));
    }

    #[test]
    fn optional_preloaded_rmsnorm_falls_back_to_weight_bytes() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let output = rmsnorm_bf16_with_optional_preloaded_resident_weight(
            "model.layers.0.input_layernorm.weight",
            &bf16_bytes(&[1.0, 2.0, 3.0]),
            Some(&bf16_bytes(&[1.0, 0.5, 2.0])),
            1,
            3,
            1.0e-6,
        )
        .unwrap();

        assert_eq!(output.values.len(), 3);
        assert_eq!(output.backend, "cpu-reference-rmsnorm-bf16");
    }

    #[test]
    fn optional_preloaded_rmsnorm_requires_bytes_without_resident_vector() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let err = rmsnorm_bf16_with_optional_preloaded_resident_weight(
            "model.layers.0.input_layernorm.weight",
            &bf16_bytes(&[1.0, 2.0, 3.0]),
            None,
            1,
            3,
            1.0e-6,
        )
        .unwrap_err();

        assert!(err.to_string().contains("RMSNorm weight bytes missing"));
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }
}
