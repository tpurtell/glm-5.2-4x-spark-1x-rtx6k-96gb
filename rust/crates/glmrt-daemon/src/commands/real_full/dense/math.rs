use anyhow::Result;

pub(in crate::commands::real_full) fn deterministic_dense_hidden(hidden_dim: usize) -> Vec<f32> {
    (0..hidden_dim)
        .map(|idx| {
            let base = ((idx % 37) as f32 - 18.0) / 256.0;
            if idx % 257 == 0 {
                base + 0.125
            } else {
                base
            }
        })
        .collect()
}

#[cfg(test)]
pub(in crate::commands::real_full) fn apply_residual_prefix(
    residual: &[f32],
    delta: &[f32],
) -> Result<Vec<f32>> {
    if residual.len() != delta.len() {
        anyhow::bail!(
            "real full dense-prefix residual length mismatch: residual={} delta={}",
            residual.len(),
            delta.len()
        );
    }
    Ok(residual
        .iter()
        .zip(delta.iter())
        .map(|(residual, delta)| residual + delta)
        .collect())
}

pub(in crate::commands::real_full) fn bf16_bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 2 != 0 {
        anyhow::bail!("BF16 byte slice length must be even, got {}", bytes.len());
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            f32::from_bits((bits as u32) << 16)
        })
        .collect())
}

pub(in crate::commands::real_full) fn bf16_bytes_from_f32(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    fill_bf16_bytes_from_f32(values, &mut bytes);
    bytes
}

pub(in crate::commands::real_full) fn fill_bf16_bytes_from_f32(
    values: &[f32],
    bytes: &mut Vec<u8>,
) {
    bytes.clear();
    bytes.reserve(values.len() * 2);
    for value in values {
        let bf16 = (value.to_bits() >> 16) as u16;
        bytes.extend_from_slice(&bf16.to_le_bytes());
    }
}

pub(in crate::commands::real_full) fn bf16_compact_row_prefix_bytes(
    bytes: &[u8],
    row_count: usize,
    row_stride: usize,
    prefix_width: usize,
) -> Result<Vec<u8>> {
    if row_count == 0 || row_stride == 0 || prefix_width == 0 {
        anyhow::bail!(
            "real full BF16 row-prefix compaction requires non-zero shape, got rows={row_count} row_stride={row_stride} prefix_width={prefix_width}"
        );
    }
    if prefix_width > row_stride {
        anyhow::bail!(
            "real full BF16 row-prefix compaction prefix width {prefix_width} exceeds row stride {row_stride}"
        );
    }
    let expected_bytes = row_count
        .checked_mul(row_stride)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "real full BF16 row-prefix compaction shape overflows usize while validating input"
            )
        })?;
    if bytes.len() != expected_bytes {
        anyhow::bail!(
            "real full BF16 row-prefix compaction byte length mismatch: expected {} got {}",
            expected_bytes,
            bytes.len()
        );
    }

    let mut compacted = Vec::with_capacity(row_count * prefix_width * std::mem::size_of::<u16>());
    for row_index in 0..row_count {
        let start = row_index * row_stride * std::mem::size_of::<u16>();
        let end = start + prefix_width * std::mem::size_of::<u16>();
        compacted.extend_from_slice(&bytes[start..end]);
    }
    Ok(compacted)
}

#[cfg(test)]
pub(in crate::commands::real_full) fn rmsnorm(
    hidden: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<Vec<f32>> {
    if hidden.len() != weight.len() {
        anyhow::bail!(
            "RMSNorm hidden/weight length mismatch: {} != {}",
            hidden.len(),
            weight.len()
        );
    }
    if hidden.is_empty() {
        anyhow::bail!("RMSNorm hidden vector must not be empty");
    }
    let variance = hidden.iter().map(|value| value * value).sum::<f32>() / hidden.len() as f32;
    let scale = (variance + eps).sqrt().recip();
    Ok(hidden
        .iter()
        .zip(weight.iter())
        .map(|(hidden, weight)| hidden * scale * weight)
        .collect())
}

pub(in crate::commands::real_full) fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| left * right)
        .sum()
}

pub(in crate::commands::real_full) fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

pub(in crate::commands::real_full) fn checksum_f64(values: &[f32]) -> f64 {
    values.iter().map(|value| *value as f64).sum()
}

#[cfg(test)]
mod tests {
    use super::{
        apply_residual_prefix, bf16_bytes_from_f32, bf16_compact_row_prefix_bytes,
        deterministic_dense_hidden, rmsnorm,
    };

    #[test]
    fn deterministic_dense_hidden_has_expected_width_and_nonzero_values() {
        let hidden = deterministic_dense_hidden(32);

        assert_eq!(hidden.len(), 32);
        assert!(hidden.iter().any(|value| *value != 0.0));
    }

    #[test]
    fn dense_prefix_rmsnorm_and_residual_helpers_are_numeric() {
        let hidden = [1.0_f32, 2.0, 3.0];
        let weight = [1.0_f32, 0.5, 2.0];
        let normalized = rmsnorm(&hidden, &weight, 1.0e-6).unwrap();

        assert_eq!(normalized.len(), hidden.len());
        assert!(normalized.iter().all(|value| value.is_finite()));
        let updated = apply_residual_prefix(&hidden[..2], &[0.25, -0.5]).unwrap();
        assert_eq!(updated, vec![1.25, 1.5]);
    }

    #[test]
    fn bf16_compact_row_prefix_bytes_keeps_row_order() {
        let bytes = bf16_bytes_from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let compacted = bf16_compact_row_prefix_bytes(&bytes, 2, 3, 2).unwrap();

        assert_eq!(compacted, bf16_bytes_from_f32(&[1.0, 2.0, 4.0, 5.0]));
    }

    #[test]
    fn bf16_compact_row_prefix_bytes_rejects_mismatched_lengths() {
        let err =
            bf16_compact_row_prefix_bytes(&bf16_bytes_from_f32(&[1.0, 2.0]), 2, 2, 1).unwrap_err();

        assert!(err.to_string().contains("byte length mismatch"));
    }
}
