use anyhow::{Context, Result};
use glmrt_core::{DType, TensorCatalog, TensorRole, GLM52_HIDDEN_SIZE};
use glmrt_loader::{load_tensor_rows, read_tensor_rows_into};

use super::coordinator_kernels::{
    coordinator_cuda_reference_kernels_enabled,
    embedding_lookup_bf16_preloaded_resident_weight_device_output,
    embedding_lookup_rows_bf16_resident_weight, embedding_lookup_rows_bf16_staged_resident_weight,
    preload_resident_weight_from_host_staging, resident_weight_is_preloaded, DeviceBf16Output,
};
use super::dense::math::checksum_f64;

const REAL_FULL_EMBEDDING_TENSOR: &str = "model.embed_tokens.weight";

pub(in crate::commands::real_full) struct RealFullEmbeddingHidden {
    pub(in crate::commands::real_full) hidden: Vec<f32>,
    pub(in crate::commands::real_full) device_hidden: Option<DeviceBf16Output>,
    pub(in crate::commands::real_full) token_id: usize,
    pub(in crate::commands::real_full) bytes_read: u64,
    pub(in crate::commands::real_full) kernel_backend: &'static str,
    pub(in crate::commands::real_full) checksum: f64,
}

pub(in crate::commands::real_full) struct RealFullEmbeddingDeviceHidden {
    pub(in crate::commands::real_full) device_hidden: DeviceBf16Output,
    pub(in crate::commands::real_full) token_count: usize,
    pub(in crate::commands::real_full) bytes_read: u64,
}

pub(in crate::commands::real_full) fn real_full_embedding_device_hidden_for_tokens(
    catalog: &TensorCatalog,
    token_ids: &[usize],
) -> Result<Option<RealFullEmbeddingDeviceHidden>> {
    if token_ids.is_empty() {
        return Ok(None);
    }
    let info = catalog
        .tensors
        .iter()
        .find(|tensor| tensor.name == REAL_FULL_EMBEDDING_TENSOR)
        .context("real full embedding tensor model.embed_tokens.weight not found in catalog")?;
    if info.dtype != DType::Bf16 || info.role != TensorRole::Embedding {
        anyhow::bail!(
            "real full embedding lookup expects BF16 embedding tensor, got dtype={:?} role={:?}",
            info.dtype,
            info.role
        );
    }
    if info.shape.len() != 2 || info.shape[1] != GLM52_HIDDEN_SIZE {
        anyhow::bail!(
            "real full embedding lookup hidden width mismatch: expected {} got shape {:?}",
            GLM52_HIDDEN_SIZE,
            info.shape
        );
    }
    let vocab_size = info.shape[0];
    let embedding_bytes = info
        .shape
        .iter()
        .try_fold(1_usize, |acc, dim| acc.checked_mul(*dim))
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("real full embedding lookup tensor shape overflows usize")?;
    if !coordinator_cuda_reference_kernels_enabled()
        || !resident_weight_is_preloaded(REAL_FULL_EMBEDDING_TENSOR, embedding_bytes)
    {
        return Ok(None);
    }
    let lookup = embedding_lookup_bf16_preloaded_resident_weight_device_output(
        REAL_FULL_EMBEDDING_TENSOR,
        token_ids,
        vocab_size,
        GLM52_HIDDEN_SIZE,
    )?;
    Ok(Some(RealFullEmbeddingDeviceHidden {
        device_hidden: lookup,
        token_count: token_ids.len(),
        bytes_read: 0,
    }))
}

pub(in crate::commands::real_full) fn real_full_embedding_hidden_for_token(
    catalog: &TensorCatalog,
    token_id: usize,
) -> Result<RealFullEmbeddingHidden> {
    let info = catalog
        .tensors
        .iter()
        .find(|tensor| tensor.name == REAL_FULL_EMBEDDING_TENSOR)
        .context("real full embedding tensor model.embed_tokens.weight not found in catalog")?;
    if info.dtype != DType::Bf16 || info.role != TensorRole::Embedding {
        anyhow::bail!(
            "real full embedding lookup expects BF16 embedding tensor, got dtype={:?} role={:?}",
            info.dtype,
            info.role
        );
    }
    if info.shape.len() != 2 || info.shape[1] != GLM52_HIDDEN_SIZE {
        anyhow::bail!(
            "real full embedding lookup hidden width mismatch: expected {} got shape {:?}",
            GLM52_HIDDEN_SIZE,
            info.shape
        );
    }
    let vocab_size = info.shape[0];
    let embedding_bytes = info
        .shape
        .iter()
        .try_fold(1_usize, |acc, dim| acc.checked_mul(*dim))
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("real full embedding lookup tensor shape overflows usize")?;
    if coordinator_cuda_reference_kernels_enabled()
        && resident_weight_is_preloaded(REAL_FULL_EMBEDDING_TENSOR, embedding_bytes)
    {
        let lookup = embedding_lookup_bf16_preloaded_resident_weight_device_output(
            REAL_FULL_EMBEDDING_TENSOR,
            &[token_id],
            vocab_size,
            GLM52_HIDDEN_SIZE,
        )?;
        let hidden = lookup.copy_to_host_values()?;
        let kernel_backend = lookup.backend;
        let checksum = checksum_f64(&hidden);
        return Ok(RealFullEmbeddingHidden {
            hidden,
            device_hidden: Some(lookup),
            token_id,
            bytes_read: 0,
            kernel_backend,
            checksum,
        });
    }

    let window_end = token_id
        .checked_add(1)
        .context("real full embedding lookup token row window end overflows usize")?;
    let embedding_key = format!("{REAL_FULL_EMBEDDING_TENSOR}[rows={token_id}..{window_end}]");

    if coordinator_cuda_reference_kernels_enabled() {
        let expected_bytes = GLM52_HIDDEN_SIZE
            .checked_mul(std::mem::size_of::<u16>())
            .context("real full embedding row-window byte size overflows usize")?;
        let mut bytes_read = 0_u64;
        preload_resident_weight_from_host_staging(
            &embedding_key,
            expected_bytes,
            "BF16 embedding row-window pinned staging",
            |staging| {
                let summary =
                    read_tensor_rows_into(catalog, REAL_FULL_EMBEDDING_TENSOR, token_id, 1, staging)
                        .with_context(|| {
                            format!(
                                "reading embedding row window [{token_id}, {window_end}) into pinned staging"
                            )
                        })?;
                if summary.dtype != DType::Bf16 {
                    anyhow::bail!(
                        "real full embedding row-window expects BF16 rows, got {:?}",
                        summary.dtype
                    );
                }
                if summary.row_width != GLM52_HIDDEN_SIZE {
                    anyhow::bail!(
                        "real full embedding lookup hidden width mismatch: expected {} got {}",
                        GLM52_HIDDEN_SIZE,
                        summary.row_width
                    );
                }
                if summary.bytes_read as usize != expected_bytes {
                    anyhow::bail!(
                        "real full embedding row-window read {} bytes, expected {}",
                        summary.bytes_read,
                        expected_bytes
                    );
                }
                bytes_read = summary.bytes_read;
                Ok(())
            },
        )
        .with_context(|| format!("preloading embedding row-window tensor {embedding_key}"))?;
        let lookup = embedding_lookup_rows_bf16_staged_resident_weight(
            &embedding_key,
            &[token_id],
            token_id,
            1,
            GLM52_HIDDEN_SIZE,
        )?;
        let hidden = lookup.values;
        let checksum = checksum_f64(&hidden);
        return Ok(RealFullEmbeddingHidden {
            hidden,
            device_hidden: None,
            token_id,
            bytes_read,
            kernel_backend: lookup.backend,
            checksum,
        });
    }

    let row = load_tensor_rows(catalog, REAL_FULL_EMBEDDING_TENSOR, token_id, 1)?;
    if row.row_width != GLM52_HIDDEN_SIZE {
        anyhow::bail!(
            "real full embedding lookup hidden width mismatch: expected {} got {}",
            GLM52_HIDDEN_SIZE,
            row.row_width
        );
    }
    let lookup = embedding_lookup_rows_bf16_resident_weight(
        &embedding_key,
        &row.bytes,
        &[token_id],
        token_id,
        1,
        GLM52_HIDDEN_SIZE,
    )?;
    let hidden = lookup.values;
    let checksum = checksum_f64(&hidden);
    Ok(RealFullEmbeddingHidden {
        hidden,
        device_hidden: None,
        token_id,
        bytes_read: row.bytes.len() as u64,
        kernel_backend: lookup.backend,
        checksum,
    })
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use glmrt_core::{DType, ModelFacts, TensorCatalog, TensorInfo, TensorRole};

    use super::super::coordinator_kernels::{
        cuda_reference_kernels_test_override, CPU_REFERENCE_EMBEDDING_LOOKUP_BF16_BACKEND,
    };
    use super::real_full_embedding_hidden_for_token;
    #[test]
    fn embedding_lookup_loads_one_bf16_hidden_row() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);
        let snapshot_path = unique_temp_dir("glmrt-real-full-embedding-test");
        fs::create_dir_all(&snapshot_path).expect("creating temp snapshot dir");
        let file_path = snapshot_path.join("model.safetensors");
        let mut row0 = vec![0.0_f32; 6144];
        let mut row1 = vec![0.0_f32; 6144];
        row0[0] = 1.0;
        row0[6143] = -2.0;
        row1[0] = 3.0;
        row1[1] = -4.0;
        row1[6143] = 5.0;
        let mut bytes = bf16_bytes_from_f32(&row0);
        bytes.extend_from_slice(&bf16_bytes_from_f32(&row1));
        fs::write(&file_path, bytes).expect("writing temp embedding tensor bytes");

        let catalog = TensorCatalog {
            model_id: "test".to_owned(),
            snapshot_path: snapshot_path.display().to_string(),
            facts: ModelFacts::default(),
            tensors: vec![TensorInfo {
                name: "model.embed_tokens.weight".to_owned(),
                file: "model.safetensors".to_owned(),
                dtype: DType::Bf16,
                shape: vec![2, 6144],
                byte_offset: 0,
                byte_length: 2 * 6144 * 2,
                role: TensorRole::Embedding,
                layer_id: None,
                expert_id: None,
                is_quantization_metadata: false,
            }],
        };

        let hidden =
            real_full_embedding_hidden_for_token(&catalog, 1).expect("loading embedding row 1");
        assert_eq!(hidden.token_id, 1);
        assert_eq!(hidden.bytes_read, 12_288);
        assert_eq!(hidden.hidden.len(), 6144);
        assert_eq!(hidden.hidden[0], 3.0);
        assert_eq!(hidden.hidden[1], -4.0);
        assert_eq!(hidden.hidden[6143], 5.0);
        assert!(hidden.device_hidden.is_none());
        assert_eq!(
            hidden.kernel_backend,
            CPU_REFERENCE_EMBEDDING_LOOKUP_BF16_BACKEND
        );
        assert_eq!(hidden.checksum, 4.0);

        fs::remove_dir_all(snapshot_path).expect("removing temp snapshot dir");
    }

    fn unique_temp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("{name}-{}", std::process::id()))
    }

    fn bf16_bytes_from_f32(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for value in values {
            let bf16 = (value.to_bits() >> 16) as u16;
            bytes.extend_from_slice(&bf16.to_le_bytes());
        }
        bytes
    }
}
