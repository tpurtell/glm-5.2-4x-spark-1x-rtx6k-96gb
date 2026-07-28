use glmrt_core::{
    DType, ModelFacts, TensorCatalog, TensorInfo, TensorRole, DEFAULT_MODEL_ID,
    GLM52_DSA_INDEXER_LAYER_IDS, GLM52_FIRST_K_DENSE_REPLACE, GLM52_HIDDEN_SIZE,
    GLM52_NUM_HIDDEN_LAYERS, GLM52_ROUTED_EXPERTS,
};
use std::env;

use crate::cli::CoordinatorArgs;

use super::super::probe_env;

const REAL_CHECKPOINT_TESTS_ENV: &str = "GLMRT_RUN_REAL_CHECKPOINT_TESTS";

pub(in crate::commands::real_full) fn real_checkpoint_tests_enabled() -> bool {
    env::var(REAL_CHECKPOINT_TESTS_ENV)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}

pub(in crate::commands::real_full) fn real_checkpoint_tests_skip_message(scope: &str) {
    eprintln!("skipped: set {REAL_CHECKPOINT_TESTS_ENV}=1 to run real checkpoint test {scope}");
}

pub(super) fn clear_real_full_probe_env() -> probe_env::ProbeEnvTestOverride {
    probe_env::mask_for_test(&[
        "GLMRT_REAL_FULL_SCORE_LM_HEAD",
        "GLMRT_REAL_FULL_PROBE_DENSE_PREFIX",
    ])
}

pub(in crate::commands::real_full) fn full_catalog() -> TensorCatalog {
    let mut tensors = vec![
        tensor_with_shape(
            "model.embed_tokens.weight",
            TensorRole::Embedding,
            None,
            vec![1, GLM52_HIDDEN_SIZE],
        ),
        tensor_with_shape(
            "lm_head.weight",
            TensorRole::LmHead,
            None,
            vec![154_880, GLM52_HIDDEN_SIZE],
        ),
    ];
    for layer_id in 0..GLM52_NUM_HIDDEN_LAYERS as u32 {
        tensors.push(tensor(
            &format!("model.layers.{layer_id}.input_layernorm.weight"),
            TensorRole::Norm,
            Some(layer_id),
        ));
        tensors.push(tensor_with_shape(
            &format!("model.layers.{layer_id}.post_attention_layernorm.weight"),
            TensorRole::Norm,
            Some(layer_id),
            vec![GLM52_HIDDEN_SIZE],
        ));
        push_attention_tensors(&mut tensors, layer_id);
        if layer_id < GLM52_FIRST_K_DENSE_REPLACE as u32 {
            push_dense_mlp_tensors(&mut tensors, layer_id);
        } else {
            tensors.push(tensor_with_shape(
                &format!("model.layers.{layer_id}.mlp.gate.weight"),
                TensorRole::Router,
                Some(layer_id),
                vec![GLM52_ROUTED_EXPERTS, GLM52_HIDDEN_SIZE],
            ));
            tensors.push(tensor_with_dtype_shape(
                &format!("model.layers.{layer_id}.mlp.gate.e_score_correction_bias"),
                TensorRole::Router,
                Some(layer_id),
                DType::F32,
                vec![GLM52_ROUTED_EXPERTS],
            ));
            push_shared_expert_tensors(&mut tensors, layer_id);
            for expert_id in 0..GLM52_ROUTED_EXPERTS as u32 {
                push_routed_expert_projection_tensors(&mut tensors, layer_id, expert_id);
            }
        }
    }

    TensorCatalog {
        model_id: DEFAULT_MODEL_ID.to_owned(),
        snapshot_path: "/tmp/glmrt-snapshot".to_owned(),
        facts: ModelFacts::default(),
        tensors,
    }
}

pub(super) fn attention_catalog() -> TensorCatalog {
    let mut tensors = Vec::new();
    for layer_id in 0..GLM52_NUM_HIDDEN_LAYERS as u32 {
        push_attention_tensors(&mut tensors, layer_id);
    }

    TensorCatalog {
        model_id: DEFAULT_MODEL_ID.to_owned(),
        snapshot_path: "/tmp/glmrt-snapshot".to_owned(),
        facts: ModelFacts::default(),
        tensors,
    }
}

pub(super) fn coordinator_args() -> CoordinatorArgs {
    CoordinatorArgs {
        backend: "real-glm-full".to_owned(),
        transport: "tcp".to_owned(),
        kv_cache_dtype: "bf16".to_owned(),
        max_context_tokens: crate::cli::DEFAULT_REAL_FULL_MAX_CONTEXT_TOKENS,
        listen: "127.0.0.1:8000".to_owned(),
        model_id: DEFAULT_MODEL_ID.to_owned(),
        expert_hosts: "spark-0,spark-1,spark-2,spark-3".to_owned(),
        catalog: None,
        loadplan: None,
        preflight_only: false,
    }
}

fn tensor(name: &str, role: TensorRole, layer_id: Option<u32>) -> TensorInfo {
    tensor_with_shape(name, role, layer_id, vec![1])
}

fn tensor_with_shape(
    name: &str,
    role: TensorRole,
    layer_id: Option<u32>,
    shape: Vec<usize>,
) -> TensorInfo {
    tensor_with_dtype_shape(name, role, layer_id, DType::Bf16, shape)
}

fn tensor_with_dtype_shape(
    name: &str,
    role: TensorRole,
    layer_id: Option<u32>,
    dtype: DType,
    shape: Vec<usize>,
) -> TensorInfo {
    let dtype_bytes = match dtype {
        DType::F32 => 4,
        _ => 2,
    };
    let byte_length = shape.iter().product::<usize>() as u64 * dtype_bytes;
    TensorInfo {
        name: name.to_owned(),
        file: "model.safetensors".to_owned(),
        dtype,
        shape,
        byte_offset: 0,
        byte_length,
        role,
        layer_id,
        expert_id: None,
        is_quantization_metadata: false,
    }
}

fn routed_expert_tensor(
    name: &str,
    layer_id: u32,
    expert_id: u32,
    is_quantization_metadata: bool,
    dtype: DType,
    shape: Vec<usize>,
) -> TensorInfo {
    let dtype_bytes = match dtype {
        DType::F32 => 4,
        _ => 1,
    };
    let byte_length = shape.iter().product::<usize>() as u64 * dtype_bytes;
    TensorInfo {
        name: name.to_owned(),
        file: "model.safetensors".to_owned(),
        dtype,
        shape,
        byte_offset: 0,
        byte_length,
        role: TensorRole::RoutedExpert,
        layer_id: Some(layer_id),
        expert_id: Some(expert_id),
        is_quantization_metadata,
    }
}

fn push_attention_tensors(tensors: &mut Vec<TensorInfo>, layer_id: u32) {
    for suffix in [
        "q_a_proj.weight",
        "q_a_layernorm.weight",
        "q_b_proj.weight",
        "kv_a_proj_with_mqa.weight",
        "kv_a_layernorm.weight",
        "kv_b_proj.weight",
        "o_proj.weight",
    ] {
        tensors.push(tensor(
            &format!("model.layers.{layer_id}.self_attn.{suffix}"),
            TensorRole::Attention,
            Some(layer_id),
        ));
    }
    if GLM52_DSA_INDEXER_LAYER_IDS.contains(&(layer_id as usize)) {
        for suffix in [
            "k_norm.bias",
            "k_norm.weight",
            "weights_proj.weight",
            "wk.weight",
            "wq_b.weight",
        ] {
            tensors.push(tensor(
                &format!("model.layers.{layer_id}.self_attn.indexer.{suffix}"),
                TensorRole::Attention,
                Some(layer_id),
            ));
        }
    }
}

fn push_dense_mlp_tensors(tensors: &mut Vec<TensorInfo>, layer_id: u32) {
    for (suffix, shape) in [
        ("gate_proj.weight", vec![4, GLM52_HIDDEN_SIZE]),
        ("up_proj.weight", vec![4, GLM52_HIDDEN_SIZE]),
        ("down_proj.weight", vec![GLM52_HIDDEN_SIZE, 4]),
    ] {
        tensors.push(tensor_with_shape(
            &format!("model.layers.{layer_id}.mlp.{suffix}"),
            TensorRole::DenseMlp,
            Some(layer_id),
            shape,
        ));
    }
}

fn push_shared_expert_tensors(tensors: &mut Vec<TensorInfo>, layer_id: u32) {
    for (suffix, shape) in [
        ("gate_proj.weight", vec![4, GLM52_HIDDEN_SIZE]),
        ("up_proj.weight", vec![4, GLM52_HIDDEN_SIZE]),
        ("down_proj.weight", vec![GLM52_HIDDEN_SIZE, 4]),
    ] {
        tensors.push(tensor_with_shape(
            &format!("model.layers.{layer_id}.mlp.shared_experts.{suffix}"),
            TensorRole::SharedExpert,
            Some(layer_id),
            shape,
        ));
    }
}

fn push_routed_expert_projection_tensors(
    tensors: &mut Vec<TensorInfo>,
    layer_id: u32,
    expert_id: u32,
) {
    for projection in ["gate_proj", "up_proj", "down_proj"] {
        let base = format!("model.layers.{layer_id}.mlp.experts.{expert_id}.{projection}");
        tensors.push(routed_expert_tensor(
            &format!("{base}.weight"),
            layer_id,
            expert_id,
            false,
            DType::U8,
            vec![4, GLM52_HIDDEN_SIZE / 2],
        ));
        tensors.push(routed_expert_tensor(
            &format!("{base}.weight_scale"),
            layer_id,
            expert_id,
            true,
            DType::F8E4M3,
            vec![4, GLM52_HIDDEN_SIZE / 16],
        ));
        tensors.push(routed_expert_tensor(
            &format!("{base}.input_scale"),
            layer_id,
            expert_id,
            true,
            DType::F32,
            vec![],
        ));
        tensors.push(routed_expert_tensor(
            &format!("{base}.weight_scale_2"),
            layer_id,
            expert_id,
            true,
            DType::F32,
            vec![],
        ));
    }
}
