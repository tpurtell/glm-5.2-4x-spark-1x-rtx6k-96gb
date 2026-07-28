use anyhow::Result;
use glmrt_core::{
    DType, TensorCatalog, TensorInfo, GLM52_DSA_INDEXER_LAYER_IDS_WITH_MTP,
    GLM52_DSA_INDEX_HEAD_DIM, GLM52_HIDDEN_SIZE,
};
use glmrt_loader::{load_tensor_bytes, load_tensor_rows};

use crate::commands::real_full::coordinator_kernels::{
    coordinator_cuda_reference_kernels_enabled,
    layer_norm_affine_f32_bf16_preloaded_resident_weight_bias,
};
use crate::commands::real_full::dense::math::{
    bf16_bytes_from_f32, bf16_bytes_to_f32, checksum_f64, dot,
};
use crate::commands::real_full::dense::REAL_FULL_DENSE_RMSNORM_EPS;
use crate::commands::real_full::types::RealFullDsaIndexerAttentionProbe;

use super::math::{
    apply_rope_row_with_backend, bf16_full_row_prefix_resident_available,
    deterministic_attention_hidden_rows, flatten_rows,
    project_rows_bf16_with_optional_preloaded_full_weight,
    project_rows_bf16_with_optional_preloaded_prefix_weight,
    rmsnorm_bf16_with_optional_preloaded_resident_weight, softmax_weights,
};
const REAL_FULL_DSA_INDEXER_LAYER_ID: usize = 22;
const REAL_FULL_DSA_ATTENTION_ROWS: usize = 3;
const REAL_FULL_DSA_QUERY_PROBE_HEADS: usize = 1;
const REAL_FULL_DSA_VALUE_WIDTH: usize = 32;
const REAL_FULL_DSA_TOP_K: usize = 3;
const REAL_FULL_DSA_ROPE_THETA: f64 = 250_000.0;

#[derive(Clone)]
struct CandidateScore {
    candidate_id: usize,
    score: f32,
}

#[derive(Clone)]
pub(super) struct RealFullDsaIndexerKvCandidateRow {
    pub(super) position: usize,
    pub(super) key_norm: Vec<f32>,
    pub(super) bytes: usize,
}

pub(super) fn real_full_dsa_indexer_attention_probe(
    catalog: &TensorCatalog,
) -> RealFullDsaIndexerAttentionProbe {
    match execute_real_full_dsa_indexer_attention_probe(catalog) {
        Ok(probe) => probe,
        Err(error) => {
            skipped_real_full_dsa_indexer_attention_probe("error", Some(error.to_string()))
        }
    }
}

pub(super) fn real_full_dsa_indexer_attention_for_layer_from_hidden_rows(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden_rows: Vec<Vec<f32>>,
    hidden_source: &'static str,
    context_source: &'static str,
) -> Result<RealFullDsaIndexerAttentionProbe> {
    let dsa_top_k = hidden_rows.len().min(REAL_FULL_DSA_TOP_K);
    execute_real_full_dsa_indexer_attention(
        catalog,
        layer_id,
        Vec::new(),
        hidden_rows,
        dsa_top_k,
        hidden_source,
        context_source,
        "execute bounded real GLM-5.2 DSA/indexer attention math from supplied residual hidden rows with real indexer q/k/value projections, RoPE, candidate scoring, and causal softmax; full context and full residual integration remain incomplete",
    )
}

pub(super) fn real_full_dsa_indexer_attention_for_layer_from_hidden_rows_with_kv_cache_candidates(
    catalog: &TensorCatalog,
    layer_id: usize,
    prefix_kv_candidate_rows: Vec<RealFullDsaIndexerKvCandidateRow>,
    hidden_rows: Vec<Vec<f32>>,
    hidden_source: &'static str,
    context_source: &'static str,
) -> Result<RealFullDsaIndexerAttentionProbe> {
    let candidate_rows = prefix_kv_candidate_rows.len() + hidden_rows.len();
    let dsa_top_k = candidate_rows.min(REAL_FULL_DSA_TOP_K);
    execute_real_full_dsa_indexer_attention(
        catalog,
        layer_id,
        prefix_kv_candidate_rows,
        hidden_rows,
        dsa_top_k,
        hidden_source,
        context_source,
        "execute bounded real GLM-5.2 DSA/indexer candidate selection using committed BF16 DSA KV-cache prefix keys plus current residual hidden rows; DSA value/context softmax remains bounded to current rows",
    )
}

pub(super) fn real_full_dsa_indexer_kv_payload_rows_for_layer_from_hidden_rows(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden_rows: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>> {
    if !GLM52_DSA_INDEXER_LAYER_IDS_WITH_MTP.contains(&layer_id) {
        anyhow::bail!("real DSA/indexer KV payload layer {layer_id} is not a configured DSA layer");
    }
    if hidden_rows.is_empty() {
        anyhow::bail!("real DSA/indexer KV payload requires at least one hidden row");
    }
    for (row_index, hidden) in hidden_rows.iter().enumerate() {
        if hidden.len() != GLM52_HIDDEN_SIZE {
            anyhow::bail!(
                "real DSA/indexer KV payload hidden row {row_index} width mismatch: expected {} got {}",
                GLM52_HIDDEN_SIZE,
                hidden.len()
            );
        }
    }

    let input_norm_name = format!("model.layers.{layer_id}.input_layernorm.weight");
    let wk_name = format!("model.layers.{layer_id}.self_attn.indexer.wk.weight");
    let k_norm_weight_name = format!("model.layers.{layer_id}.self_attn.indexer.k_norm.weight");
    let k_norm_bias_name = format!("model.layers.{layer_id}.self_attn.indexer.k_norm.bias");

    let input_norm_info = catalog_tensor(catalog, &input_norm_name)?;
    let wk_info = catalog_tensor(catalog, &wk_name)?;
    let cuda_reference_enabled = coordinator_cuda_reference_kernels_enabled();
    let input_norm_full_resident =
        super::math::bf16_full_vector_resident_available(&input_norm_name, GLM52_HIDDEN_SIZE);
    let input_norm = if input_norm_full_resident {
        None
    } else if cuda_reference_enabled {
        super::preload_bf16_tensor_resident_from_host_staging(
            catalog,
            &input_norm_name,
            &[GLM52_HIDDEN_SIZE],
            "BF16 DSA/indexer KV payload input norm pinned staging",
        )?;
        None
    } else {
        Some(load_tensor_bytes(catalog, &input_norm_name)?)
    };
    let k_norm_weight_info = catalog_tensor(catalog, &k_norm_weight_name)?;
    let k_norm_bias_info = catalog_tensor(catalog, &k_norm_bias_name)?;
    let k_norm_vectors_resident =
        k_norm_vectors_are_preloaded(&k_norm_weight_name, &k_norm_bias_name);
    let k_norm_weight = if k_norm_vectors_resident {
        None
    } else if cuda_reference_enabled {
        super::preload_bf16_tensor_resident_from_host_staging(
            catalog,
            &k_norm_weight_name,
            &[GLM52_DSA_INDEX_HEAD_DIM],
            "BF16 DSA/indexer KV payload k_norm weight pinned staging",
        )?;
        None
    } else {
        Some(load_tensor_bytes(catalog, &k_norm_weight_name)?)
    };
    let k_norm_bias = if k_norm_vectors_resident {
        None
    } else if cuda_reference_enabled {
        super::preload_bf16_tensor_resident_from_host_staging(
            catalog,
            &k_norm_bias_name,
            &[GLM52_DSA_INDEX_HEAD_DIM],
            "BF16 DSA/indexer KV payload k_norm bias pinned staging",
        )?;
        None
    } else {
        Some(load_tensor_bytes(catalog, &k_norm_bias_name)?)
    };

    for (name, dtype) in [
        (&input_norm_info.name, &input_norm_info.dtype),
        (&wk_info.name, &wk_info.dtype),
        (&k_norm_weight_info.name, &k_norm_weight_info.dtype),
        (&k_norm_bias_info.name, &k_norm_bias_info.dtype),
    ] {
        if *dtype != DType::Bf16 {
            anyhow::bail!("real DSA/indexer KV payload expects BF16 tensor {name}, got {dtype:?}");
        }
    }
    if input_norm_info.shape != vec![GLM52_HIDDEN_SIZE]
        || wk_info.shape != vec![GLM52_DSA_INDEX_HEAD_DIM, GLM52_HIDDEN_SIZE]
        || k_norm_weight_info.shape != vec![GLM52_DSA_INDEX_HEAD_DIM]
        || k_norm_bias_info.shape != vec![GLM52_DSA_INDEX_HEAD_DIM]
    {
        anyhow::bail!(
            "real DSA/indexer KV payload tensor shape mismatch for layer {layer_id}: input_norm={:?} wk={:?} k_norm_w={:?} k_norm_b={:?}",
            input_norm_info.shape,
            wk_info.shape,
            k_norm_weight_info.shape,
            k_norm_bias_info.shape
        );
    }
    let wk_full_resident = bf16_full_row_prefix_resident_available(
        &wk_name,
        GLM52_DSA_INDEX_HEAD_DIM,
        GLM52_HIDDEN_SIZE,
        GLM52_DSA_INDEX_HEAD_DIM,
    );
    let wk = if wk_full_resident {
        None
    } else if cuda_reference_enabled {
        super::preload_bf16_tensor_resident_from_host_staging(
            catalog,
            &wk_name,
            &[GLM52_DSA_INDEX_HEAD_DIM, GLM52_HIDDEN_SIZE],
            "BF16 DSA/indexer KV payload wk pinned staging",
        )?;
        None
    } else {
        Some(load_tensor_bytes(catalog, &wk_name)?)
    };

    let flattened_hidden = flatten_rows(hidden_rows);
    let normalized = rmsnorm_bf16_with_optional_preloaded_resident_weight(
        &input_norm_name,
        &bf16_bytes_from_f32(&flattened_hidden),
        input_norm.as_ref().map(|tensor| tensor.bytes.as_slice()),
        hidden_rows.len(),
        GLM52_HIDDEN_SIZE,
        REAL_FULL_DENSE_RMSNORM_EPS,
    )?
    .values;
    let k_norm_weight_values = k_norm_weight
        .as_ref()
        .map(|tensor| bf16_bytes_to_f32(&tensor.bytes))
        .transpose()?;
    let k_norm_bias_values = k_norm_bias
        .as_ref()
        .map(|tensor| bf16_bytes_to_f32(&tensor.bytes))
        .transpose()?;
    normalized
        .chunks_exact(GLM52_HIDDEN_SIZE)
        .map(|normalized_row| {
            let key = project_rows_bf16_with_optional_preloaded_full_weight(
                &wk_name,
                normalized_row,
                wk.as_ref().map(|tensor| tensor.bytes.as_slice()),
                GLM52_DSA_INDEX_HEAD_DIM,
                GLM52_HIDDEN_SIZE,
            )?;
            layer_norm_affine_with_optional_preloaded_resident_weight_bias(
                &k_norm_weight_name,
                &k_norm_bias_name,
                &key.values,
                k_norm_weight_values.as_deref(),
                k_norm_bias_values.as_deref(),
                GLM52_DSA_INDEX_HEAD_DIM,
                REAL_FULL_DENSE_RMSNORM_EPS,
            )
        })
        .collect()
}

fn skipped_real_full_dsa_indexer_attention_probe(
    status: &'static str,
    skipped_reason: Option<String>,
) -> RealFullDsaIndexerAttentionProbe {
    RealFullDsaIndexerAttentionProbe {
        status,
        scope: "execute bounded real GLM-5.2 DSA/indexer attention math with real indexer q/k/value projections, RoPE, candidate scoring, and causal softmax; full context and full residual integration remain incomplete",
        layer_id: REAL_FULL_DSA_INDEXER_LAYER_ID,
        hidden_source: "not-run",
        context_source: "not-run",
        attention_rows: REAL_FULL_DSA_ATTENTION_ROWS,
        q_lora_rank: 0,
        dsa_query_dim: GLM52_DSA_INDEX_HEAD_DIM,
        dsa_value_width: REAL_FULL_DSA_VALUE_WIDTH,
        candidate_rows: 0,
        prefix_kv_candidate_rows: 0,
        kv_cache_candidate_bytes: 0,
        dsa_top_k: REAL_FULL_DSA_TOP_K,
        selected_indices: Vec::new(),
        score_order: Vec::new(),
        causal_attention_scores: 0,
        dsa_softmax_rows: 0,
        attention_context_values: 0,
        rope_theta: REAL_FULL_DSA_ROPE_THETA,
        input_norm_bytes_read: 0,
        q_projection_bytes_read: 0,
        indexer_bytes_read: 0,
        projection_backend: "not-run",
        rope_backend: "not-run",
        q_checksum: None,
        q_rope_rotated_checksum: None,
        k_checksum: None,
        k_norm_checksum: None,
        k_rope_rotated_checksum: None,
        value_checksum: None,
        candidate_scores_checksum: None,
        attention_weights_checksum: None,
        attention_context_checksum: None,
        uses_real_indexer_weights: false,
        includes_rope: false,
        includes_dsa_candidate_selection: false,
        includes_dsa_softmax: false,
        uses_full_model_residual: false,
        passed: false,
        skipped_reason,
    }
}

fn execute_real_full_dsa_indexer_attention_probe(
    catalog: &TensorCatalog,
) -> Result<RealFullDsaIndexerAttentionProbe> {
    let layer_id = REAL_FULL_DSA_INDEXER_LAYER_ID;
    execute_real_full_dsa_indexer_attention(
        catalog,
        layer_id,
        Vec::new(),
        deterministic_dsa_indexer_hidden_rows(),
        REAL_FULL_DSA_TOP_K,
        "three-deterministic-hidden-shaped-f32-rows",
        "bounded-dsa-indexer-rope-causal-context",
        "execute bounded real GLM-5.2 DSA/indexer attention math with real indexer q/k/value projections, RoPE, candidate scoring, and causal softmax; full context and full residual integration remain incomplete",
    )
}

fn execute_real_full_dsa_indexer_attention(
    catalog: &TensorCatalog,
    layer_id: usize,
    prefix_kv_candidate_rows: Vec<RealFullDsaIndexerKvCandidateRow>,
    hidden_rows: Vec<Vec<f32>>,
    dsa_top_k: usize,
    hidden_source: &'static str,
    context_source: &'static str,
    scope: &'static str,
) -> Result<RealFullDsaIndexerAttentionProbe> {
    if !GLM52_DSA_INDEXER_LAYER_IDS_WITH_MTP.contains(&layer_id) {
        anyhow::bail!("real DSA/indexer probe layer {layer_id} is not a configured DSA layer");
    }
    if hidden_rows.is_empty() {
        anyhow::bail!("real DSA/indexer attention requires at least one hidden row");
    }
    let candidate_row_count = prefix_kv_candidate_rows.len() + hidden_rows.len();
    if dsa_top_k == 0 || dsa_top_k > candidate_row_count {
        anyhow::bail!(
            "real DSA/indexer top-k {dsa_top_k} is invalid for {candidate_row_count} candidate rows",
        );
    }
    for (row_index, candidate) in prefix_kv_candidate_rows.iter().enumerate() {
        if candidate.key_norm.len() != GLM52_DSA_INDEX_HEAD_DIM {
            anyhow::bail!(
                "real DSA/indexer KV candidate row {row_index} width mismatch: expected {} got {}",
                GLM52_DSA_INDEX_HEAD_DIM,
                candidate.key_norm.len()
            );
        }
    }
    for (row_index, hidden) in hidden_rows.iter().enumerate() {
        if hidden.len() != GLM52_HIDDEN_SIZE {
            anyhow::bail!(
                "real DSA/indexer hidden row {row_index} width mismatch: expected {} got {}",
                GLM52_HIDDEN_SIZE,
                hidden.len()
            );
        }
    }
    let hidden_position_start = next_hidden_position_after_kv_candidates(&prefix_kv_candidate_rows);
    let positions =
        (hidden_position_start..hidden_position_start + hidden_rows.len()).collect::<Vec<_>>();

    let input_norm_name = format!("model.layers.{layer_id}.input_layernorm.weight");
    let q_a_name = format!("model.layers.{layer_id}.self_attn.q_a_proj.weight");
    let q_a_norm_name = format!("model.layers.{layer_id}.self_attn.q_a_layernorm.weight");
    let wq_b_name = format!("model.layers.{layer_id}.self_attn.indexer.wq_b.weight");
    let wk_name = format!("model.layers.{layer_id}.self_attn.indexer.wk.weight");
    let weights_proj_name =
        format!("model.layers.{layer_id}.self_attn.indexer.weights_proj.weight");
    let k_norm_weight_name = format!("model.layers.{layer_id}.self_attn.indexer.k_norm.weight");
    let k_norm_bias_name = format!("model.layers.{layer_id}.self_attn.indexer.k_norm.bias");

    let input_norm_info = catalog_tensor(catalog, &input_norm_name)?;
    let q_a_info = catalog_tensor(catalog, &q_a_name)?;
    let q_a_norm_info = catalog_tensor(catalog, &q_a_norm_name)?;
    let wq_b_rows = REAL_FULL_DSA_QUERY_PROBE_HEADS * GLM52_DSA_INDEX_HEAD_DIM;
    let wq_b_info = catalog_tensor(catalog, &wq_b_name)?;
    let wk_info = catalog_tensor(catalog, &wk_name)?;
    let weights_proj_info = catalog_tensor(catalog, &weights_proj_name)?;
    let k_norm_weight_info = catalog_tensor(catalog, &k_norm_weight_name)?;
    let k_norm_bias_info = catalog_tensor(catalog, &k_norm_bias_name)?;
    let wq_b_weight_key = format!("{wq_b_name}[rows=0..{wq_b_rows}]");

    for (name, dtype) in [
        (&input_norm_info.name, &input_norm_info.dtype),
        (&q_a_info.name, &q_a_info.dtype),
        (&q_a_norm_info.name, &q_a_norm_info.dtype),
        (&wq_b_info.name, &wq_b_info.dtype),
        (&wk_info.name, &wk_info.dtype),
        (&weights_proj_info.name, &weights_proj_info.dtype),
        (&k_norm_weight_info.name, &k_norm_weight_info.dtype),
        (&k_norm_bias_info.name, &k_norm_bias_info.dtype),
    ] {
        if *dtype != DType::Bf16 {
            anyhow::bail!("real DSA/indexer probe expects BF16 tensor {name}, got {dtype:?}");
        }
    }
    if input_norm_info.shape != vec![GLM52_HIDDEN_SIZE]
        || q_a_info.shape.len() != 2
        || q_a_info.shape[1] != GLM52_HIDDEN_SIZE
        || wk_info.shape != vec![GLM52_DSA_INDEX_HEAD_DIM, GLM52_HIDDEN_SIZE]
        || weights_proj_info.shape != vec![REAL_FULL_DSA_VALUE_WIDTH, GLM52_HIDDEN_SIZE]
        || k_norm_weight_info.shape != vec![GLM52_DSA_INDEX_HEAD_DIM]
        || k_norm_bias_info.shape != vec![GLM52_DSA_INDEX_HEAD_DIM]
    {
        anyhow::bail!(
            "real DSA/indexer probe tensor shape mismatch for layer {layer_id}: input_norm={:?} q_a={:?} wk={:?} weights_proj={:?} k_norm_w={:?} k_norm_b={:?}",
            input_norm_info.shape,
            q_a_info.shape,
            wk_info.shape,
            weights_proj_info.shape,
            k_norm_weight_info.shape,
            k_norm_bias_info.shape
        );
    }
    let q_lora_rank = q_a_info.shape[0];
    let (wq_b_full_rows, wq_b_row_width) =
        validate_bf16_matrix_tensor(wq_b_info, "real DSA/indexer wq_b", layer_id)?;
    if q_a_norm_info.shape != vec![q_lora_rank] || wq_b_row_width != q_lora_rank {
        anyhow::bail!(
            "real DSA/indexer q projection shape mismatch: q_rank={q_lora_rank} q_norm={:?} wq_b_width={}",
            q_a_norm_info.shape,
            wq_b_row_width
        );
    }
    if wq_b_full_rows % GLM52_DSA_INDEX_HEAD_DIM != 0 || wq_b_rows > wq_b_full_rows {
        anyhow::bail!(
            "real DSA/indexer wq_b row prefix invalid: prefix_rows={wq_b_rows} full_rows={wq_b_full_rows} head_dim={}",
            GLM52_DSA_INDEX_HEAD_DIM,
        );
    }
    let q_a_full_resident = bf16_full_row_prefix_resident_available(
        &q_a_name,
        q_lora_rank,
        GLM52_HIDDEN_SIZE,
        q_lora_rank,
    );
    let wk_full_resident = bf16_full_row_prefix_resident_available(
        &wk_name,
        GLM52_DSA_INDEX_HEAD_DIM,
        GLM52_HIDDEN_SIZE,
        GLM52_DSA_INDEX_HEAD_DIM,
    );
    let weights_proj_full_resident = bf16_full_row_prefix_resident_available(
        &weights_proj_name,
        REAL_FULL_DSA_VALUE_WIDTH,
        GLM52_HIDDEN_SIZE,
        REAL_FULL_DSA_VALUE_WIDTH,
    );
    let wq_b_full_resident =
        bf16_full_row_prefix_resident_available(&wq_b_name, wq_b_full_rows, q_lora_rank, wq_b_rows);
    let cuda_reference_enabled = coordinator_cuda_reference_kernels_enabled();
    let mut input_norm_bytes_read = 0_u64;
    let mut q_projection_bytes_read = 0_u64;
    let mut indexer_bytes_read = 0_u64;
    let input_norm_full_resident =
        super::math::bf16_full_vector_resident_available(&input_norm_name, GLM52_HIDDEN_SIZE);
    let input_norm = if input_norm_full_resident {
        None
    } else if cuda_reference_enabled {
        input_norm_bytes_read = super::preload_bf16_tensor_resident_from_host_staging(
            catalog,
            &input_norm_name,
            &[GLM52_HIDDEN_SIZE],
            "BF16 DSA/indexer attention input norm pinned staging",
        )?;
        None
    } else {
        let tensor = load_tensor_bytes(catalog, &input_norm_name)?;
        input_norm_bytes_read = tensor.bytes.len() as u64;
        Some(tensor)
    };
    let q_a_norm_full_resident =
        super::math::bf16_full_vector_resident_available(&q_a_norm_name, q_lora_rank);
    let q_a_norm = if q_a_norm_full_resident {
        None
    } else if cuda_reference_enabled {
        q_projection_bytes_read += super::preload_bf16_tensor_resident_from_host_staging(
            catalog,
            &q_a_norm_name,
            &[q_lora_rank],
            "BF16 DSA/indexer attention q_a norm pinned staging",
        )?;
        None
    } else {
        let tensor = load_tensor_bytes(catalog, &q_a_norm_name)?;
        q_projection_bytes_read += tensor.bytes.len() as u64;
        Some(tensor)
    };
    let k_norm_vectors_resident =
        k_norm_vectors_are_preloaded(&k_norm_weight_name, &k_norm_bias_name);
    let k_norm_weight = if k_norm_vectors_resident {
        None
    } else if cuda_reference_enabled {
        indexer_bytes_read += super::preload_bf16_tensor_resident_from_host_staging(
            catalog,
            &k_norm_weight_name,
            &[GLM52_DSA_INDEX_HEAD_DIM],
            "BF16 DSA/indexer attention k_norm weight pinned staging",
        )?;
        None
    } else {
        let tensor = load_tensor_bytes(catalog, &k_norm_weight_name)?;
        indexer_bytes_read += tensor.bytes.len() as u64;
        Some(tensor)
    };
    let k_norm_bias = if k_norm_vectors_resident {
        None
    } else if cuda_reference_enabled {
        indexer_bytes_read += super::preload_bf16_tensor_resident_from_host_staging(
            catalog,
            &k_norm_bias_name,
            &[GLM52_DSA_INDEX_HEAD_DIM],
            "BF16 DSA/indexer attention k_norm bias pinned staging",
        )?;
        None
    } else {
        let tensor = load_tensor_bytes(catalog, &k_norm_bias_name)?;
        indexer_bytes_read += tensor.bytes.len() as u64;
        Some(tensor)
    };
    let wq_b = if wq_b_full_resident {
        None
    } else if cuda_reference_enabled {
        q_projection_bytes_read += super::preload_bf16_rows_resident_from_host_staging(
            catalog,
            &wq_b_name,
            &wq_b_weight_key,
            wq_b_rows,
            q_lora_rank,
            "BF16 DSA/indexer attention wq_b row-window pinned staging",
        )?;
        None
    } else {
        let rows = load_tensor_rows(catalog, &wq_b_name, 0, wq_b_rows)?;
        q_projection_bytes_read += rows.bytes.len() as u64;
        Some(rows)
    };
    let q_a = if q_a_full_resident {
        None
    } else if cuda_reference_enabled {
        q_projection_bytes_read += super::preload_bf16_tensor_resident_from_host_staging(
            catalog,
            &q_a_name,
            &[q_lora_rank, GLM52_HIDDEN_SIZE],
            "BF16 DSA/indexer attention q_a pinned staging",
        )?;
        None
    } else {
        let tensor = load_tensor_bytes(catalog, &q_a_name)?;
        q_projection_bytes_read += tensor.bytes.len() as u64;
        Some(tensor)
    };
    let wk = if wk_full_resident {
        None
    } else if cuda_reference_enabled {
        indexer_bytes_read += super::preload_bf16_tensor_resident_from_host_staging(
            catalog,
            &wk_name,
            &[GLM52_DSA_INDEX_HEAD_DIM, GLM52_HIDDEN_SIZE],
            "BF16 DSA/indexer attention wk pinned staging",
        )?;
        None
    } else {
        let tensor = load_tensor_bytes(catalog, &wk_name)?;
        indexer_bytes_read += tensor.bytes.len() as u64;
        Some(tensor)
    };
    let weights_proj = if weights_proj_full_resident {
        None
    } else if cuda_reference_enabled {
        indexer_bytes_read += super::preload_bf16_tensor_resident_from_host_staging(
            catalog,
            &weights_proj_name,
            &[REAL_FULL_DSA_VALUE_WIDTH, GLM52_HIDDEN_SIZE],
            "BF16 DSA/indexer attention weights_proj pinned staging",
        )?;
        None
    } else {
        let tensor = load_tensor_bytes(catalog, &weights_proj_name)?;
        indexer_bytes_read += tensor.bytes.len() as u64;
        Some(tensor)
    };

    let k_norm_weight_values = k_norm_weight
        .as_ref()
        .map(|tensor| bf16_bytes_to_f32(&tensor.bytes))
        .transpose()?;
    let k_norm_bias_values = k_norm_bias
        .as_ref()
        .map(|tensor| bf16_bytes_to_f32(&tensor.bytes))
        .transpose()?;

    let mut query_rows = Vec::with_capacity(hidden_rows.len());
    let mut query_rope_rows = Vec::with_capacity(hidden_rows.len());
    let mut key_rows = Vec::with_capacity(hidden_rows.len());
    let mut key_norm_rows = Vec::with_capacity(hidden_rows.len());
    let mut key_rope_rows = Vec::with_capacity(hidden_rows.len());
    let mut value_rows = Vec::with_capacity(hidden_rows.len());
    let mut candidate_key_rope_rows =
        Vec::with_capacity(prefix_kv_candidate_rows.len() + hidden_rows.len());
    let mut candidate_ids = Vec::with_capacity(prefix_kv_candidate_rows.len() + hidden_rows.len());
    let mut projection_backend = None;
    let mut rope_backend = None;

    for candidate in &prefix_kv_candidate_rows {
        let key_rope = apply_rope_row_with_backend(
            layer_id,
            &candidate.key_norm,
            candidate.position,
            REAL_FULL_DSA_ROPE_THETA,
        )?;
        record_backend(&mut rope_backend, key_rope.backend, "kv_cache_key_rope")?;
        candidate_key_rope_rows.push(key_rope.values);
        candidate_ids.push(candidate.position);
    }

    for (row_index, hidden) in hidden_rows.iter().enumerate() {
        let normalized = rmsnorm_bf16_with_optional_preloaded_resident_weight(
            &input_norm_name,
            &bf16_bytes_from_f32(hidden),
            input_norm.as_ref().map(|tensor| tensor.bytes.as_slice()),
            1,
            hidden.len(),
            REAL_FULL_DENSE_RMSNORM_EPS,
        )?
        .values;
        let q_a_projected = project_rows_bf16_with_optional_preloaded_full_weight(
            &q_a_name,
            &normalized,
            q_a.as_ref().map(|tensor| tensor.bytes.as_slice()),
            q_lora_rank,
            hidden.len(),
        )?;
        record_backend(
            &mut projection_backend,
            q_a_projected.backend,
            "q_a_projection",
        )?;
        let q_a_normalized = rmsnorm_bf16_with_optional_preloaded_resident_weight(
            &q_a_norm_name,
            &bf16_bytes_from_f32(&q_a_projected.values),
            q_a_norm.as_ref().map(|tensor| tensor.bytes.as_slice()),
            1,
            q_a_projected.values.len(),
            REAL_FULL_DENSE_RMSNORM_EPS,
        )?
        .values;
        let query = project_rows_bf16_with_optional_preloaded_prefix_weight(
            &wq_b_name,
            &wq_b_weight_key,
            &q_a_normalized,
            wq_b.as_ref().map(|rows| rows.bytes.as_slice()),
            GLM52_DSA_INDEX_HEAD_DIM,
            q_lora_rank,
            wq_b_full_rows,
        )?;
        record_backend(&mut projection_backend, query.backend, "query_projection")?;
        let key = project_rows_bf16_with_optional_preloaded_full_weight(
            &wk_name,
            &normalized,
            wk.as_ref().map(|tensor| tensor.bytes.as_slice()),
            GLM52_DSA_INDEX_HEAD_DIM,
            hidden.len(),
        )?;
        record_backend(&mut projection_backend, key.backend, "key_projection")?;
        let key_norm = layer_norm_affine_with_optional_preloaded_resident_weight_bias(
            &k_norm_weight_name,
            &k_norm_bias_name,
            &key.values,
            k_norm_weight_values.as_deref(),
            k_norm_bias_values.as_deref(),
            GLM52_DSA_INDEX_HEAD_DIM,
            REAL_FULL_DENSE_RMSNORM_EPS,
        )?;
        let value = project_rows_bf16_with_optional_preloaded_full_weight(
            &weights_proj_name,
            &normalized,
            weights_proj.as_ref().map(|tensor| tensor.bytes.as_slice()),
            REAL_FULL_DSA_VALUE_WIDTH,
            hidden.len(),
        )?;
        record_backend(&mut projection_backend, value.backend, "value_projection")?;

        let query_rope = apply_rope_row_with_backend(
            layer_id,
            &query.values,
            positions[row_index],
            REAL_FULL_DSA_ROPE_THETA,
        )?;
        record_backend(&mut rope_backend, query_rope.backend, "query_rope")?;
        let key_rope = apply_rope_row_with_backend(
            layer_id,
            &key_norm,
            positions[row_index],
            REAL_FULL_DSA_ROPE_THETA,
        )?;
        record_backend(&mut rope_backend, key_rope.backend, "key_rope")?;
        query_rope_rows.push(query_rope.values);
        candidate_key_rope_rows.push(key_rope.values.clone());
        candidate_ids.push(positions[row_index]);
        key_rope_rows.push(key_rope.values);
        query_rows.push(query.values);
        key_rows.push(key.values);
        key_norm_rows.push(key_norm);
        value_rows.push(value.values);
    }
    let projection_backend = projection_backend.ok_or_else(|| {
        anyhow::anyhow!("real full DSA/indexer probe did not record projection backend")
    })?;
    let rope_backend = rope_backend.ok_or_else(|| {
        anyhow::anyhow!("real full DSA/indexer probe did not record RoPE backend")
    })?;

    let candidate_scores = score_candidates(
        query_rope_rows
            .last()
            .ok_or_else(|| anyhow::anyhow!("DSA query rows unexpectedly empty"))?,
        &candidate_key_rope_rows,
        &candidate_ids,
    )?;
    let mut sorted_scores = candidate_scores.clone();
    sorted_scores.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });
    let selected_indices = sorted_scores
        .iter()
        .take(dsa_top_k)
        .map(|score| score.candidate_id)
        .collect::<Vec<_>>();
    let score_order = sorted_scores
        .iter()
        .map(|score| score.candidate_id)
        .collect::<Vec<_>>();

    let (attention_scores, attention_weights, context_rows) =
        causal_dsa_attention(&query_rope_rows, &key_rope_rows, &value_rows)?;
    let context_values = flatten_rows(&context_rows);
    let attention_weights_checksum = checksum_f64(&attention_weights);
    let candidate_scores_checksum = candidate_scores
        .iter()
        .map(|score| score.score as f64)
        .sum::<f64>();
    let causal_attention_scores = hidden_rows.len() * (hidden_rows.len() + 1) / 2;
    let dsa_softmax_rows = hidden_rows.len();
    let passed = attention_scores.len() == causal_attention_scores
        && selected_indices.len() == dsa_top_k
        && score_order.len() == candidate_key_rope_rows.len()
        && (attention_weights_checksum - dsa_softmax_rows as f64).abs() < 1.0e-5
        && context_values.iter().all(|value| value.is_finite())
        && candidate_scores.iter().all(|score| score.score.is_finite());

    Ok(RealFullDsaIndexerAttentionProbe {
        status: "numeric-real-bounded-dsa-indexer-attention",
        scope,
        layer_id,
        hidden_source,
        context_source,
        attention_rows: hidden_rows.len(),
        q_lora_rank,
        dsa_query_dim: GLM52_DSA_INDEX_HEAD_DIM,
        dsa_value_width: REAL_FULL_DSA_VALUE_WIDTH,
        candidate_rows: candidate_key_rope_rows.len(),
        prefix_kv_candidate_rows: prefix_kv_candidate_rows.len(),
        kv_cache_candidate_bytes: prefix_kv_candidate_rows
            .iter()
            .map(|candidate| candidate.bytes)
            .sum(),
        dsa_top_k,
        selected_indices,
        score_order,
        causal_attention_scores,
        dsa_softmax_rows,
        attention_context_values: context_values.len(),
        rope_theta: REAL_FULL_DSA_ROPE_THETA,
        input_norm_bytes_read,
        q_projection_bytes_read,
        indexer_bytes_read,
        projection_backend,
        rope_backend,
        q_checksum: Some(checksum_f64(&flatten_rows(&query_rows))),
        q_rope_rotated_checksum: Some(checksum_f64(&flatten_rows(&query_rope_rows))),
        k_checksum: Some(checksum_f64(&flatten_rows(&key_rows))),
        k_norm_checksum: Some(checksum_f64(&flatten_rows(&key_norm_rows))),
        k_rope_rotated_checksum: Some(checksum_f64(&flatten_rows(&key_rope_rows))),
        value_checksum: Some(checksum_f64(&flatten_rows(&value_rows))),
        candidate_scores_checksum: Some(candidate_scores_checksum),
        attention_weights_checksum: Some(attention_weights_checksum),
        attention_context_checksum: Some(checksum_f64(&context_values)),
        uses_real_indexer_weights: true,
        includes_rope: true,
        includes_dsa_candidate_selection: true,
        includes_dsa_softmax: true,
        uses_full_model_residual: false,
        passed,
        skipped_reason: None,
    })
}

fn k_norm_vectors_are_preloaded(weight_name: &str, bias_name: &str) -> bool {
    super::math::bf16_full_vector_resident_available(weight_name, GLM52_DSA_INDEX_HEAD_DIM)
        && super::math::bf16_full_vector_resident_available(bias_name, GLM52_DSA_INDEX_HEAD_DIM)
}

fn layer_norm_affine_with_optional_preloaded_resident_weight_bias(
    weight_name: &str,
    bias_name: &str,
    values: &[f32],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    hidden_dim: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    if hidden_dim == 0 || values.is_empty() || values.len() % hidden_dim != 0 {
        anyhow::bail!(
            "real DSA/indexer affine LayerNorm shape mismatch: values={} hidden_dim={hidden_dim}",
            values.len()
        );
    }
    if k_norm_vectors_are_preloaded(weight_name, bias_name) {
        return Ok(layer_norm_affine_f32_bf16_preloaded_resident_weight_bias(
            weight_name,
            bias_name,
            values,
            values.len() / hidden_dim,
            hidden_dim,
            eps,
        )?
        .values);
    }
    let weight = weight.ok_or_else(|| {
        anyhow::anyhow!("real DSA/indexer affine LayerNorm weight bytes missing for {weight_name}")
    })?;
    let bias = bias.ok_or_else(|| {
        anyhow::anyhow!("real DSA/indexer affine LayerNorm bias bytes missing for {bias_name}")
    })?;
    layer_norm_affine(values, weight, bias, eps)
}

fn layer_norm_affine(values: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Result<Vec<f32>> {
    if values.len() != weight.len() || values.len() != bias.len() {
        anyhow::bail!(
            "real DSA/indexer k_norm shape mismatch: values={} weight={} bias={}",
            values.len(),
            weight.len(),
            bias.len()
        );
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f32>()
        / values.len() as f32;
    let inv_std = (variance + eps).sqrt().recip();
    Ok(values
        .iter()
        .zip(weight)
        .zip(bias)
        .map(|((value, weight), bias)| (value - mean) * inv_std * weight + bias)
        .collect())
}

fn record_backend(
    current: &mut Option<&'static str>,
    observed: &'static str,
    source: &str,
) -> Result<()> {
    match current {
        Some(existing) if *existing != observed => {
            anyhow::bail!(
                "real full DSA/indexer mixed coordinator backends: first={} {source}={observed}",
                *existing
            );
        }
        Some(_) => {}
        None => *current = Some(observed),
    }
    Ok(())
}

fn catalog_tensor<'a>(catalog: &'a TensorCatalog, name: &str) -> Result<&'a TensorInfo> {
    catalog
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| anyhow::anyhow!("missing tensor {name} in real full catalog"))
}

fn validate_bf16_matrix_tensor(
    info: &TensorInfo,
    context: &str,
    layer_id: usize,
) -> Result<(usize, usize)> {
    if info.dtype != DType::Bf16 || info.shape.len() != 2 {
        anyhow::bail!(
            "{context} expected rank-2 BF16 tensor {} for layer {layer_id}, got dtype={:?} shape={:?}",
            info.name,
            info.dtype,
            info.shape
        );
    }
    Ok((info.shape[0], info.shape[1]))
}

fn deterministic_dsa_indexer_hidden_rows() -> Vec<Vec<f32>> {
    let mut rows = deterministic_attention_hidden_rows();
    let mut third = rows
        .first()
        .cloned()
        .unwrap_or_else(|| vec![0.0; GLM52_HIDDEN_SIZE]);
    for (idx, value) in third.iter_mut().enumerate() {
        *value += ((idx % 17) as f32 - 8.0) / 3072.0;
        if idx % 113 == 0 {
            *value += 0.0234375;
        }
        if idx % 251 == 0 {
            *value -= 0.015625;
        }
    }
    rows.push(third);
    rows
}

fn next_hidden_position_after_kv_candidates(
    prefix_kv_candidate_rows: &[RealFullDsaIndexerKvCandidateRow],
) -> usize {
    prefix_kv_candidate_rows
        .iter()
        .map(|candidate| candidate.position.saturating_add(1))
        .max()
        .unwrap_or_default()
}

fn score_candidates(
    query: &[f32],
    candidates: &[Vec<f32>],
    candidate_ids: &[usize],
) -> Result<Vec<CandidateScore>> {
    if candidates.is_empty() {
        anyhow::bail!("real DSA/indexer candidate scoring requires candidates");
    }
    if candidates.len() != candidate_ids.len() {
        anyhow::bail!(
            "real DSA/indexer candidate id count mismatch: candidates={} ids={}",
            candidates.len(),
            candidate_ids.len()
        );
    }
    candidates
        .iter()
        .zip(candidate_ids)
        .map(|(candidate, candidate_id)| {
            if candidate.len() != query.len() {
                anyhow::bail!(
                    "real DSA/indexer candidate width mismatch: query={} candidate={}",
                    query.len(),
                    candidate.len()
                );
            }
            Ok(CandidateScore {
                candidate_id: *candidate_id,
                score: dot(query, candidate),
            })
        })
        .collect()
}

fn causal_dsa_attention(
    queries: &[Vec<f32>],
    keys: &[Vec<f32>],
    values: &[Vec<f32>],
) -> Result<(Vec<f32>, Vec<f32>, Vec<Vec<f32>>)> {
    if queries.len() != keys.len() || queries.len() != values.len() {
        anyhow::bail!(
            "real DSA/indexer attention row mismatch: q={} k={} v={}",
            queries.len(),
            keys.len(),
            values.len()
        );
    }
    let mut all_scores = Vec::new();
    let mut all_weights = Vec::new();
    let mut contexts = Vec::with_capacity(queries.len());
    let scale = (queries
        .first()
        .ok_or_else(|| anyhow::anyhow!("real DSA/indexer attention requires rows"))?
        .len() as f32)
        .sqrt()
        .recip();

    for row_index in 0..queries.len() {
        let query = &queries[row_index];
        let scores = keys[..=row_index]
            .iter()
            .map(|key| dot(query, key) * scale)
            .collect::<Vec<_>>();
        let weights = softmax_weights(&scores)?;
        let mut context = vec![0.0_f32; values[0].len()];
        for (weight, value) in weights.iter().zip(&values[..=row_index]) {
            for (target, value) in context.iter_mut().zip(value) {
                *target += weight * value;
            }
        }
        all_scores.extend(scores);
        all_weights.extend(weights);
        contexts.push(context);
    }
    Ok((all_scores, all_weights, contexts))
}

#[cfg(test)]
mod tests {
    use super::{
        next_hidden_position_after_kv_candidates, score_candidates,
        RealFullDsaIndexerKvCandidateRow,
    };

    #[test]
    fn score_candidates_preserves_kv_cache_candidate_positions() {
        let scores = score_candidates(&[1.0, 2.0], &[vec![3.0, 4.0], vec![5.0, 6.0]], &[7, 11])
            .expect("scoring candidates");

        assert_eq!(scores.len(), 2);
        assert_eq!(scores[0].candidate_id, 7);
        assert_eq!(scores[0].score, 11.0);
        assert_eq!(scores[1].candidate_id, 11);
        assert_eq!(scores[1].score, 17.0);
    }

    #[test]
    fn kv_cache_candidate_positions_advance_current_dsa_position() {
        let candidates = vec![
            RealFullDsaIndexerKvCandidateRow {
                position: 7,
                key_norm: Vec::new(),
                bytes: 0,
            },
            RealFullDsaIndexerKvCandidateRow {
                position: 11,
                key_norm: Vec::new(),
                bytes: 0,
            },
        ];

        assert_eq!(next_hidden_position_after_kv_candidates(&candidates), 12);
        assert_eq!(next_hidden_position_after_kv_candidates(&[]), 0);
    }
}
