use anyhow::{Context, Result};
use glmrt_loader::LoadedTensorRows;

pub(super) fn bf16_bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>> {
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

pub(super) fn f32_bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        anyhow::bail!(
            "F32 byte slice length must be divisible by 4, got {}",
            bytes.len()
        );
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

pub(in crate::commands::real_full) fn deterministic_probe_hidden(hidden_dim: usize) -> Vec<f32> {
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

pub(super) fn apply_residual_prefix(residual: &[f32], delta: &[f32]) -> Result<Vec<f32>> {
    if residual.len() != delta.len() {
        anyhow::bail!(
            "real full residual prefix length mismatch: residual={} delta={}",
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

pub(in crate::commands::real_full) fn checksum_f64(values: &[f32]) -> f64 {
    values.iter().map(|value| *value as f64).sum()
}

pub(super) fn first_f32_scalar(tensor_name: &str, bytes: &[u8]) -> Result<f32> {
    if bytes.len() != std::mem::size_of::<f32>() {
        anyhow::bail!(
            "expected scalar F32 tensor {tensor_name}, got {} bytes",
            bytes.len()
        );
    }
    Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

pub(super) fn tensor_row_bytes(rows: &LoadedTensorRows, row_index: usize) -> Result<&[u8]> {
    if row_index >= rows.row_count {
        anyhow::bail!(
            "row index {row_index} exceeds loaded row count {} for {}",
            rows.row_count,
            rows.info.name
        );
    }
    let row_bytes = rows
        .row_width
        .checked_mul(rows.bytes_per_scalar)
        .context("real full NVFP4 row byte width overflow")?;
    let start = row_index
        .checked_mul(row_bytes)
        .context("real full NVFP4 row byte offset overflow")?;
    let end = start
        .checked_add(row_bytes)
        .context("real full NVFP4 row byte end overflow")?;
    rows.bytes
        .get(start..end)
        .with_context(|| format!("row {row_index} out of bounds for {}", rows.info.name))
}

pub(super) fn dot_packed_nvfp4(
    input: &[f32],
    packed_row: &[u8],
    scale_row: &[u8],
    scale_2: f32,
) -> Result<f32> {
    let required_packed_bytes = input.len().div_ceil(2);
    let required_scale_bytes = input.len().div_ceil(16);
    if packed_row.len() < required_packed_bytes {
        anyhow::bail!(
            "packed NVFP4 row has {} bytes, needs {required_packed_bytes} for {} values",
            packed_row.len(),
            input.len()
        );
    }
    if scale_row.len() < required_scale_bytes {
        anyhow::bail!(
            "packed NVFP4 scale row has {} bytes, needs {required_scale_bytes} for {} values",
            scale_row.len(),
            input.len()
        );
    }
    if !scale_2.is_finite() {
        anyhow::bail!("packed NVFP4 second scale must be finite");
    }

    let mut sum = 0.0_f32;
    for (value_idx, input_value) in input.iter().enumerate() {
        let weight = packed_nvfp4_value(packed_row, scale_row, value_idx, scale_2);
        sum = input_value.mul_add(weight, sum);
    }
    Ok(sum)
}

fn packed_nvfp4_value(packed_row: &[u8], scale_row: &[u8], value_idx: usize, scale_2: f32) -> f32 {
    let packed = packed_row[value_idx / 2];
    let code = if value_idx % 2 == 0 {
        packed & 0x0f
    } else {
        packed >> 4
    };
    let scale = f8e4m3_byte_to_f32(scale_row[value_idx / 16]);
    nvfp4_e2m1_code_value(code) * scale * scale_2
}

fn nvfp4_e2m1_code_value(code: u8) -> f32 {
    const CODEBOOK: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    CODEBOOK[(code & 0x0f) as usize]
}

pub(super) fn f8e4m3_byte_to_f32(byte: u8) -> f32 {
    if byte == 0 || byte == 0x80 {
        return 0.0;
    }
    let sign = if byte & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = ((byte >> 3) & 0x0f) as i32;
    let mantissa = (byte & 0x07) as f32;
    let significand = if exponent == 0 {
        mantissa / 8.0
    } else {
        1.0 + mantissa / 8.0
    };
    let exponent_power = if exponent == 0 { -6 } else { exponent - 7 };
    sign * significand * 2.0_f32.powi(exponent_power)
}

pub(super) fn silu(value: f32) -> f32 {
    if value >= 0.0 {
        value / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        value * exp_value / (1.0 + exp_value)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        apply_residual_prefix, dot_packed_nvfp4, f8e4m3_byte_to_f32, nvfp4_e2m1_code_value,
        packed_nvfp4_value, silu,
    };
    use serde::Deserialize;
    use std::{fs, path::PathBuf};

    #[derive(Debug, Deserialize)]
    struct Nvfp4DecodeFixture {
        source: String,
        quant_recipe: String,
        packing_order: String,
        projection: String,
        value_count: usize,
        packed_bytes_hex: String,
        scale_bytes_hex: String,
        weight_scale_2: f32,
        decoded_values: Vec<f32>,
        decoded_checksum: f64,
        full_row: Nvfp4DecodeFixtureFullRow,
        tolerance_abs: f32,
        tensors: Nvfp4DecodeFixtureTensors,
    }

    #[derive(Debug, Deserialize)]
    struct Nvfp4DecodeFixtureFullRow {
        value_count: usize,
        packed_byte_count: usize,
        scale_byte_count: usize,
        packed_bytes_hex: String,
        scale_bytes_hex: String,
        decoded_checksum: f64,
        decoded_l2_norm: f64,
        first_decoded: f32,
        last_decoded: f32,
    }

    #[derive(Debug, Deserialize)]
    struct Nvfp4DecodeFixtureTensors {
        weight: Nvfp4DecodeFixtureTensor,
        weight_scale: Nvfp4DecodeFixtureTensor,
    }

    #[derive(Debug, Deserialize)]
    struct Nvfp4DecodeFixtureTensor {
        name: String,
        dtype: String,
    }

    #[test]
    fn packed_nvfp4_dot_uses_nibbles_and_scale_groups() {
        let input = vec![1.0_f32; 17];
        let mut packed = vec![0_u8; 9];
        packed[0] = 0xa9;
        packed[1..8].fill(0x88);
        packed[8] = 0x89;
        let scale = vec![0x38, 0x40];

        let dot = dot_packed_nvfp4(&input, &packed, &scale, 1.0).unwrap();

        assert!((dot + 2.5).abs() < 1.0e-6);
        assert!((f8e4m3_byte_to_f32(0x38) - 1.0).abs() < 1.0e-6);
        assert_eq!(nvfp4_e2m1_code_value(10), -1.0);
    }

    #[test]
    fn residual_prefix_applies_delta_elementwise() {
        let residual = [1.0_f32, -2.0, 3.5];
        let delta = [0.25_f32, 0.5, -1.0];

        let updated = apply_residual_prefix(&residual, &delta).unwrap();

        assert_eq!(updated, vec![1.25, -1.5, 2.5]);
        let err = apply_residual_prefix(&residual, &delta[..2]).unwrap_err();
        assert!(err.to_string().contains("residual prefix length mismatch"));
    }

    #[test]
    fn silu_uses_stable_negative_branch() {
        let activated = silu(-100.0);

        assert!(activated.is_finite());
        assert!(activated < 0.0);
        assert!(activated.abs() < 1.0e-38);
    }

    #[test]
    fn real_checkpoint_nvfp4_decode_matches_python_fixture() {
        let fixture_path = repo_root().join("tests/fixtures/nvfp4/real_tensor_decode.json");
        let fixture_bytes = fs::read(&fixture_path)
            .unwrap_or_else(|err| panic!("reading {}: {err}", fixture_path.display()));
        let fixture: Nvfp4DecodeFixture =
            serde_json::from_slice(&fixture_bytes).expect("parsing NVFP4 decode fixture");
        let packed = decode_hex(&fixture.packed_bytes_hex);
        let scales = decode_hex(&fixture.scale_bytes_hex);

        assert_eq!(fixture.source, "python-reference-raw-safetensors");
        assert_eq!(fixture.quant_recipe, "nvfp4-e2m1-f8e4m3");
        assert_eq!(fixture.packing_order, "low-nibble-first");
        assert_eq!(fixture.projection, "gate_proj");
        assert_eq!(
            fixture.tensors.weight.name,
            "model.layers.3.mlp.experts.0.gate_proj.weight"
        );
        assert_eq!(fixture.tensors.weight.dtype, "u8");
        assert_eq!(fixture.tensors.weight_scale.dtype, "f8e4m3");
        assert_eq!(fixture.decoded_values.len(), fixture.value_count);
        assert!(packed.len() >= fixture.value_count.div_ceil(2));
        assert!(scales.len() >= fixture.value_count.div_ceil(16));

        let decoded = (0..fixture.value_count)
            .map(|value_idx| {
                packed_nvfp4_value(&packed, &scales, value_idx, fixture.weight_scale_2)
            })
            .collect::<Vec<_>>();
        for (value_idx, (actual, expected)) in decoded
            .iter()
            .zip(fixture.decoded_values.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= fixture.tolerance_abs,
                "decoded value {value_idx} differs: actual={actual} expected={expected}"
            );
        }
        let checksum = decoded.iter().map(|value| *value as f64).sum::<f64>();
        assert!(
            (checksum - fixture.decoded_checksum).abs() <= fixture.tolerance_abs as f64,
            "decoded checksum differs: actual={checksum} expected={}",
            fixture.decoded_checksum
        );
        assert!((f8e4m3_byte_to_f32(scales[0]) - 1.25).abs() <= 1.0e-6);

        let full_packed = decode_hex(&fixture.full_row.packed_bytes_hex);
        let full_scales = decode_hex(&fixture.full_row.scale_bytes_hex);
        assert_eq!(fixture.full_row.value_count, 6144);
        assert_eq!(full_packed.len(), fixture.full_row.packed_byte_count);
        assert_eq!(full_scales.len(), fixture.full_row.scale_byte_count);
        assert_eq!(fixture.full_row.packed_byte_count, 3072);
        assert_eq!(fixture.full_row.scale_byte_count, 384);

        let full_decoded = (0..fixture.full_row.value_count)
            .map(|value_idx| {
                packed_nvfp4_value(
                    &full_packed,
                    &full_scales,
                    value_idx,
                    fixture.weight_scale_2,
                )
            })
            .collect::<Vec<_>>();
        let full_checksum = full_decoded.iter().map(|value| *value as f64).sum::<f64>();
        let full_l2_norm = full_decoded
            .iter()
            .map(|value| {
                let value = *value as f64;
                value * value
            })
            .sum::<f64>();
        assert!(
            (full_checksum - fixture.full_row.decoded_checksum).abs()
                <= fixture.tolerance_abs as f64,
            "full-row checksum differs: actual={full_checksum} expected={}",
            fixture.full_row.decoded_checksum
        );
        assert!(
            (full_l2_norm - fixture.full_row.decoded_l2_norm).abs() <= fixture.tolerance_abs as f64,
            "full-row l2 differs: actual={full_l2_norm} expected={}",
            fixture.full_row.decoded_l2_norm
        );
        assert!(
            (full_decoded[0] - fixture.full_row.first_decoded).abs() <= fixture.tolerance_abs,
            "full-row first decoded value differs"
        );
        assert!(
            (full_decoded[full_decoded.len() - 1] - fixture.full_row.last_decoded).abs()
                <= fixture.tolerance_abs,
            "full-row last decoded value differs"
        );
    }

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
    }

    fn decode_hex(value: &str) -> Vec<u8> {
        assert!(
            value.len() % 2 == 0,
            "hex fixture value must have even length"
        );
        (0..value.len())
            .step_by(2)
            .map(|idx| u8::from_str_radix(&value[idx..idx + 2], 16).expect("valid fixture hex"))
            .collect()
    }
}
