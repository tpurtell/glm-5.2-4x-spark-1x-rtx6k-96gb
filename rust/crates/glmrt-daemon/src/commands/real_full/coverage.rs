use glmrt_core::{
    TensorCatalog, TensorRole, GLM52_FIRST_K_DENSE_REPLACE, GLM52_NUM_HIDDEN_LAYERS,
    GLM52_ROUTED_EXPERTS,
};
use std::collections::BTreeMap;

use super::types::FullModelTensorCoverage;

pub(super) fn tensor_coverage(catalog: &TensorCatalog) -> FullModelTensorCoverage {
    let mut layers_with_any_tensor = BTreeMap::new();
    let mut sparse_layers_with_routed_experts = BTreeMap::new();
    let mut dense_layers_with_dense_mlp = BTreeMap::new();
    let mut coverage = FullModelTensorCoverage {
        hidden_layers_with_any_tensor: 0,
        sparse_layers_with_routed_experts: 0,
        dense_layers_with_dense_mlp: 0,
        routed_expert_tensors: 0,
        routed_quant_metadata_tensors: 0,
        attention_tensors: 0,
        router_tensors: 0,
        shared_expert_tensors: 0,
        embedding_tensors: 0,
        lm_head_tensors: 0,
    };
    for tensor in &catalog.tensors {
        if let Some(layer_id) = tensor.layer_id {
            if (layer_id as usize) < GLM52_NUM_HIDDEN_LAYERS {
                layers_with_any_tensor.insert(layer_id, ());
            }
        }
        match tensor.role {
            TensorRole::RoutedExpert => {
                coverage.routed_expert_tensors += 1;
                if tensor.is_quantization_metadata {
                    coverage.routed_quant_metadata_tensors += 1;
                }
                if let Some(layer_id) = tensor
                    .layer_id
                    .filter(|layer_id| (*layer_id as usize) < GLM52_NUM_HIDDEN_LAYERS)
                {
                    sparse_layers_with_routed_experts.insert(layer_id, ());
                }
            }
            TensorRole::DenseMlp => {
                if let Some(layer_id) = tensor
                    .layer_id
                    .filter(|layer_id| (*layer_id as usize) < GLM52_NUM_HIDDEN_LAYERS)
                {
                    dense_layers_with_dense_mlp.insert(layer_id, ());
                }
            }
            TensorRole::Attention => coverage.attention_tensors += 1,
            TensorRole::Router => coverage.router_tensors += 1,
            TensorRole::SharedExpert => coverage.shared_expert_tensors += 1,
            TensorRole::Embedding => coverage.embedding_tensors += 1,
            TensorRole::LmHead => coverage.lm_head_tensors += 1,
            _ => {}
        }
    }
    coverage.hidden_layers_with_any_tensor = layers_with_any_tensor.len();
    coverage.sparse_layers_with_routed_experts = sparse_layers_with_routed_experts.len();
    coverage.dense_layers_with_dense_mlp = dense_layers_with_dense_mlp.len();
    coverage
}

pub(in crate::commands::real_full) fn catalog_supports_default_sparse_router_routed_nvfp4(
    catalog: &TensorCatalog,
) -> bool {
    let coverage = SparseExpertCatalogCoverage::from_catalog(catalog);
    coverage.supports_sparse_router_routed_nvfp4()
}

pub(in crate::commands::real_full) fn catalog_supports_default_sparse_mlp_shared_chain(
    catalog: &TensorCatalog,
) -> bool {
    let coverage = SparseExpertCatalogCoverage::from_catalog(catalog);
    coverage.supports_sparse_router_routed_nvfp4() && coverage.supports_sparse_shared_experts()
}

pub(in crate::commands::real_full) fn catalog_supports_default_dense_sparse_shared_lm_head(
    catalog: &TensorCatalog,
) -> bool {
    catalog
        .tensors
        .iter()
        .any(|tensor| tensor.role == TensorRole::LmHead)
        && DensePrefixCatalogCoverage::from_catalog(catalog).supports_dense_prefix()
        && catalog_supports_default_sparse_mlp_shared_chain(catalog)
}

struct DensePrefixCatalogCoverage {
    post_attention_norm_by_layer: BTreeMap<u32, usize>,
    dense_mlp_by_layer: BTreeMap<u32, usize>,
}

impl DensePrefixCatalogCoverage {
    fn from_catalog(catalog: &TensorCatalog) -> Self {
        let mut coverage = Self {
            post_attention_norm_by_layer: BTreeMap::new(),
            dense_mlp_by_layer: BTreeMap::new(),
        };

        for tensor in &catalog.tensors {
            match tensor.role {
                TensorRole::Norm if tensor.name.ends_with(".post_attention_layernorm.weight") => {
                    if let Some(layer_id) = tensor.layer_id {
                        *coverage
                            .post_attention_norm_by_layer
                            .entry(layer_id)
                            .or_default() += 1;
                    }
                }
                TensorRole::DenseMlp => {
                    if let Some(layer_id) = tensor.layer_id {
                        *coverage.dense_mlp_by_layer.entry(layer_id).or_default() += 1;
                    }
                }
                _ => {}
            }
        }

        coverage
    }

    fn supports_dense_prefix(&self) -> bool {
        (0..GLM52_FIRST_K_DENSE_REPLACE as u32).all(|layer_id| {
            self.post_attention_norm_by_layer
                .get(&layer_id)
                .copied()
                .unwrap_or(0)
                >= 1
                && self.dense_mlp_by_layer.get(&layer_id).copied().unwrap_or(0) >= 3
        })
    }
}

struct SparseExpertCatalogCoverage {
    shared_by_layer: BTreeMap<u32, usize>,
    router_by_layer: BTreeMap<u32, usize>,
    routed_weight_by_expert: BTreeMap<(u32, u32), usize>,
    routed_quant_metadata_by_expert: BTreeMap<(u32, u32), usize>,
}

impl SparseExpertCatalogCoverage {
    fn from_catalog(catalog: &TensorCatalog) -> Self {
        let mut coverage = Self {
            shared_by_layer: BTreeMap::new(),
            router_by_layer: BTreeMap::new(),
            routed_weight_by_expert: BTreeMap::new(),
            routed_quant_metadata_by_expert: BTreeMap::new(),
        };

        for tensor in &catalog.tensors {
            match tensor.role {
                TensorRole::SharedExpert => {
                    if let Some(layer_id) = tensor.layer_id {
                        *coverage.shared_by_layer.entry(layer_id).or_default() += 1;
                    }
                }
                TensorRole::Router => {
                    if let Some(layer_id) = tensor.layer_id {
                        *coverage.router_by_layer.entry(layer_id).or_default() += 1;
                    }
                }
                TensorRole::RoutedExpert if tensor.is_quantization_metadata => {
                    if let (Some(layer_id), Some(expert_id)) = (tensor.layer_id, tensor.expert_id) {
                        *coverage
                            .routed_quant_metadata_by_expert
                            .entry((layer_id, expert_id))
                            .or_default() += 1;
                    }
                }
                TensorRole::RoutedExpert => {
                    if let (Some(layer_id), Some(expert_id)) = (tensor.layer_id, tensor.expert_id) {
                        *coverage
                            .routed_weight_by_expert
                            .entry((layer_id, expert_id))
                            .or_default() += 1;
                    }
                }
                _ => {}
            }
        }

        coverage
    }

    fn supports_sparse_router_routed_nvfp4(&self) -> bool {
        for layer_id in sparse_layer_ids() {
            if self.router_by_layer.get(&layer_id).copied().unwrap_or(0) < 2 {
                return false;
            }
            for expert_id in 0..GLM52_ROUTED_EXPERTS as u32 {
                let expert_key = (layer_id, expert_id);
                if self
                    .routed_weight_by_expert
                    .get(&expert_key)
                    .copied()
                    .unwrap_or(0)
                    < 3
                {
                    return false;
                }
                if self
                    .routed_quant_metadata_by_expert
                    .get(&expert_key)
                    .copied()
                    .unwrap_or(0)
                    < 9
                {
                    return false;
                }
            }
        }
        true
    }

    fn supports_sparse_shared_experts(&self) -> bool {
        sparse_layer_ids()
            .all(|layer_id| self.shared_by_layer.get(&layer_id).copied().unwrap_or(0) >= 3)
    }
}

fn sparse_layer_ids() -> std::ops::Range<u32> {
    GLM52_FIRST_K_DENSE_REPLACE as u32..GLM52_NUM_HIDDEN_LAYERS as u32
}

#[cfg(test)]
mod tests {
    use super::{
        catalog_supports_default_dense_sparse_shared_lm_head,
        catalog_supports_default_sparse_mlp_shared_chain,
        catalog_supports_default_sparse_router_routed_nvfp4,
    };
    use glmrt_core::{
        DType, ModelFacts, TensorCatalog, TensorInfo, TensorRole, GLM52_FIRST_K_DENSE_REPLACE,
        GLM52_NUM_HIDDEN_LAYERS, GLM52_ROUTED_EXPERTS,
    };

    #[test]
    fn default_real_probe_coverage_gates_require_complete_tensor_families() {
        let mut catalog = full_default_real_probe_catalog();
        assert!(catalog_supports_default_dense_sparse_shared_lm_head(
            &catalog
        ));
        assert!(catalog_supports_default_sparse_router_routed_nvfp4(
            &catalog
        ));
        assert!(catalog_supports_default_sparse_mlp_shared_chain(&catalog));
        assert!(catalog_supports_default_dense_sparse_shared_lm_head(
            &catalog
        ));

        let sparse_layer = GLM52_FIRST_K_DENSE_REPLACE as u32;
        let missing_quant = remove_first_matching(&mut catalog, |tensor| {
            tensor.role == TensorRole::RoutedExpert
                && tensor.is_quantization_metadata
                && tensor.layer_id == Some(sparse_layer)
                && tensor.expert_id == Some(0)
        });
        assert!(!catalog_supports_default_sparse_router_routed_nvfp4(
            &catalog
        ));
        assert!(!catalog_supports_default_sparse_mlp_shared_chain(&catalog));
        assert!(!catalog_supports_default_dense_sparse_shared_lm_head(
            &catalog
        ));
        catalog.tensors.push(missing_quant);
        assert!(catalog_supports_default_sparse_router_routed_nvfp4(
            &catalog
        ));

        let missing_shared = remove_first_matching(&mut catalog, |tensor| {
            tensor.role == TensorRole::SharedExpert && tensor.layer_id == Some(sparse_layer)
        });
        assert!(catalog_supports_default_sparse_router_routed_nvfp4(
            &catalog
        ));
        assert!(!catalog_supports_default_sparse_mlp_shared_chain(&catalog));
        assert!(!catalog_supports_default_dense_sparse_shared_lm_head(
            &catalog
        ));
        catalog.tensors.push(missing_shared);
        assert!(catalog_supports_default_sparse_mlp_shared_chain(&catalog));

        let missing_dense = remove_first_matching(&mut catalog, |tensor| {
            tensor.role == TensorRole::DenseMlp && tensor.layer_id == Some(0)
        });
        assert!(catalog_supports_default_sparse_mlp_shared_chain(&catalog));
        assert!(!catalog_supports_default_dense_sparse_shared_lm_head(
            &catalog
        ));
        catalog.tensors.push(missing_dense);
        assert!(catalog_supports_default_dense_sparse_shared_lm_head(
            &catalog
        ));

        let missing_lm_head =
            remove_first_matching(&mut catalog, |tensor| tensor.role == TensorRole::LmHead);
        assert!(!catalog_supports_default_dense_sparse_shared_lm_head(
            &catalog
        ));
        catalog.tensors.push(missing_lm_head);
        assert!(catalog_supports_default_dense_sparse_shared_lm_head(
            &catalog
        ));
    }

    fn full_default_real_probe_catalog() -> TensorCatalog {
        let mut tensors = vec![tensor(
            "lm_head.weight",
            TensorRole::LmHead,
            None,
            None,
            false,
        )];
        for layer_id in 0..GLM52_FIRST_K_DENSE_REPLACE as u32 {
            tensors.push(tensor(
                &format!("model.layers.{layer_id}.post_attention_layernorm.weight"),
                TensorRole::Norm,
                Some(layer_id),
                None,
                false,
            ));
            for _ in 0..3 {
                tensors.push(tensor(
                    "dense.weight",
                    TensorRole::DenseMlp,
                    Some(layer_id),
                    None,
                    false,
                ));
            }
        }
        for layer_id in GLM52_FIRST_K_DENSE_REPLACE as u32..GLM52_NUM_HIDDEN_LAYERS as u32 {
            for _ in 0..2 {
                tensors.push(tensor(
                    "router.weight",
                    TensorRole::Router,
                    Some(layer_id),
                    None,
                    false,
                ));
            }
            for _ in 0..3 {
                tensors.push(tensor(
                    "shared.weight",
                    TensorRole::SharedExpert,
                    Some(layer_id),
                    None,
                    false,
                ));
            }
            for expert_id in 0..GLM52_ROUTED_EXPERTS as u32 {
                for _ in 0..3 {
                    tensors.push(tensor(
                        "routed.weight",
                        TensorRole::RoutedExpert,
                        Some(layer_id),
                        Some(expert_id),
                        false,
                    ));
                }
                for _ in 0..9 {
                    tensors.push(tensor(
                        "routed.weight_scale",
                        TensorRole::RoutedExpert,
                        Some(layer_id),
                        Some(expert_id),
                        true,
                    ));
                }
            }
        }
        TensorCatalog {
            model_id: "test/model".to_owned(),
            snapshot_path: "/tmp/glmrt-test".to_owned(),
            facts: ModelFacts::default(),
            tensors,
        }
    }

    fn tensor(
        name: &str,
        role: TensorRole,
        layer_id: Option<u32>,
        expert_id: Option<u32>,
        is_quantization_metadata: bool,
    ) -> TensorInfo {
        TensorInfo {
            name: name.to_owned(),
            file: "model.safetensors".to_owned(),
            dtype: DType::Bf16,
            shape: vec![1],
            byte_offset: 0,
            byte_length: 2,
            role,
            layer_id,
            expert_id,
            is_quantization_metadata,
        }
    }

    fn remove_first_matching(
        catalog: &mut TensorCatalog,
        mut predicate: impl FnMut(&TensorInfo) -> bool,
    ) -> TensorInfo {
        let index = catalog
            .tensors
            .iter()
            .position(|tensor| predicate(tensor))
            .expect("test catalog contains matching tensor");
        catalog.tensors.remove(index)
    }
}
