use std::collections::{BTreeMap, BTreeSet};

use glmrt_core::{
    DType, KvCacheConfig, LayerId, TensorCatalog, TensorRole, GLM52_DSA_INDEXER_LAYERS,
    GLM52_DSA_INDEXER_LAYER_IDS, GLM52_NUM_HIDDEN_LAYERS,
};

use super::types::{RealFullAttentionKvBindingDryRun, RealFullAttentionKvIoDryRun};
#[cfg(test)]
pub(in crate::commands::real_full) use residual::real_full_attention_residual_full_output_hidden;
pub(in crate::commands::real_full) use residual::{
    real_full_attention_residual_full_output_hidden_for_layer_from_initial,
    real_full_attention_residual_prefix_hidden_for_layer_from_initial,
    real_full_attention_residual_prefix_probe, real_full_attention_residual_prefix_rows,
    real_full_dsa_indexer_attention_probe,
    real_full_mla_rope_attention_full_output_hidden_for_layer_from_initial,
    real_full_mla_rope_attention_full_output_kv_cache_context_hidden_for_layer_from_initial,
    real_full_mla_rope_attention_full_output_kv_cache_context_hidden_for_layer_from_initial_device_input,
    real_full_mla_rope_attention_full_output_prefix_context_hidden_for_layer_from_initial,
    real_full_mla_rope_attention_kv_cache_context_hidden_for_layer_from_initial,
    real_full_mla_rope_attention_kv_cache_context_hidden_for_layer_from_initial_device_input,
    real_full_mla_rope_attention_prefix_context_hidden_for_layer_from_initial,
    real_full_mla_rope_attention_prefix_hidden_for_layer_from_initial,
    real_full_mla_rope_attention_probe, real_full_mla_rope_kv_cache_block_for_layer_from_hidden,
    RealFullAttentionResidualPrefixHidden, RealFullMlaRopeKvCacheBlock,
};

mod mla;
mod residual;

const COMMON_ATTENTION_SUFFIXES: [&str; 7] = [
    ".self_attn.q_a_proj.weight",
    ".self_attn.q_a_layernorm.weight",
    ".self_attn.q_b_proj.weight",
    ".self_attn.kv_a_proj_with_mqa.weight",
    ".self_attn.kv_a_layernorm.weight",
    ".self_attn.kv_b_proj.weight",
    ".self_attn.o_proj.weight",
];
const DSA_INDEXER_SUFFIXES: [&str; 5] = [
    ".self_attn.indexer.k_norm.bias",
    ".self_attn.indexer.k_norm.weight",
    ".self_attn.indexer.weights_proj.weight",
    ".self_attn.indexer.wk.weight",
    ".self_attn.indexer.wq_b.weight",
];

#[derive(Debug, Default)]
struct AttentionLayerStats {
    common_suffixes: BTreeSet<&'static str>,
    indexer_suffixes: BTreeSet<&'static str>,
    tensor_count: usize,
    bf16_tensor_count: usize,
    byte_count: u64,
}

pub(super) fn real_full_attention_kv_binding_dry_run(
    catalog: &TensorCatalog,
    kv_config: &KvCacheConfig,
    kv_io: &RealFullAttentionKvIoDryRun,
) -> RealFullAttentionKvBindingDryRun {
    let mut stats_by_layer = BTreeMap::<usize, AttentionLayerStats>::new();
    let mut attention_tensors = 0_usize;
    let mut bf16_attention_tensors = 0_usize;
    let mut common_attention_tensors = 0_usize;
    let mut indexer_attention_tensors = 0_usize;
    let mut attention_tensor_bytes = 0_u64;

    for tensor in &catalog.tensors {
        if tensor.role != TensorRole::Attention {
            continue;
        }
        let Some(layer_id) = tensor.layer_id.map(|layer_id| layer_id as usize) else {
            continue;
        };
        if layer_id >= GLM52_NUM_HIDDEN_LAYERS {
            continue;
        }
        let stats = stats_by_layer.entry(layer_id).or_default();
        stats.tensor_count += 1;
        stats.byte_count += tensor.byte_length;
        attention_tensors += 1;
        attention_tensor_bytes += tensor.byte_length;
        if tensor.dtype == DType::Bf16 {
            stats.bf16_tensor_count += 1;
            bf16_attention_tensors += 1;
        }
        if let Some(suffix) = matching_suffix(&tensor.name, &COMMON_ATTENTION_SUFFIXES) {
            stats.common_suffixes.insert(suffix);
            common_attention_tensors += 1;
        }
        if let Some(suffix) = matching_suffix(&tensor.name, &DSA_INDEXER_SUFFIXES) {
            stats.indexer_suffixes.insert(suffix);
            indexer_attention_tensors += 1;
        }
    }

    // This report audits the 78 target layers; layer 78's MTP-only indexer is
    // accounted for by the serving KV configuration and execution path.
    let config_dsa_layer_ids = kv_config
        .dsa_indexer_layer_ids()
        .iter()
        .copied()
        .filter(|layer_id| *layer_id < GLM52_NUM_HIDDEN_LAYERS)
        .collect::<Vec<_>>();
    let config_dsa_layer_set = config_dsa_layer_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let catalog_dsa_layer_ids = stats_by_layer
        .iter()
        .filter_map(|(layer_id, stats)| (!stats.indexer_suffixes.is_empty()).then_some(*layer_id))
        .collect::<Vec<_>>();
    let catalog_dsa_layer_set = catalog_dsa_layer_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();

    let mut common_layers_with_required_tensors = 0_usize;
    let mut dsa_indexer_layers_with_required_tensors = 0_usize;
    let mut non_dsa_layers_without_indexer_tensors = 0_usize;
    let mut attention_layers = 0_usize;

    for layer_id in 0..GLM52_NUM_HIDDEN_LAYERS {
        let Some(stats) = stats_by_layer.get(&layer_id) else {
            continue;
        };
        attention_layers += 1;
        common_layers_with_required_tensors +=
            usize::from(stats.common_suffixes.len() == COMMON_ATTENTION_SUFFIXES.len());
        if config_dsa_layer_set.contains(&layer_id) {
            dsa_indexer_layers_with_required_tensors +=
                usize::from(stats.indexer_suffixes.len() == DSA_INDEXER_SUFFIXES.len());
        } else {
            non_dsa_layers_without_indexer_tensors +=
                usize::from(stats.indexer_suffixes.is_empty());
        }
    }

    let kv_layer_bytes_sum = (0..GLM52_NUM_HIDDEN_LAYERS)
        .map(|layer_id| kv_config.layer_bytes_per_token(LayerId(layer_id as u32)))
        .sum::<usize>();
    let non_dsa_layer_id = (0..GLM52_NUM_HIDDEN_LAYERS)
        .find(|layer_id| !kv_config.layer_has_dsa_indexer(LayerId(*layer_id as u32)))
        .unwrap_or_default();
    RealFullAttentionKvBindingDryRun {
        status: "attention-kv-binding-dry-run",
        scope: "bind real full-model attention tensor coverage to compressed KV layer-byte accounting and LayerWave KV I/O",
        attention_layers,
        attention_tensors,
        bf16_attention_tensors,
        common_attention_tensors,
        indexer_attention_tensors,
        attention_tensor_bytes,
        common_layers_with_required_tensors,
        dsa_indexer_layers: catalog_dsa_layer_ids.len(),
        dsa_indexer_layers_with_required_tensors,
        non_dsa_layers: GLM52_NUM_HIDDEN_LAYERS - config_dsa_layer_ids.len(),
        non_dsa_layers_without_indexer_tensors,
        catalog_dsa_indexer_layer_ids: catalog_dsa_layer_ids,
        config_dsa_indexer_layer_ids: config_dsa_layer_ids,
        catalog_dsa_indexer_layers_match_kv_config: catalog_dsa_layer_set == config_dsa_layer_set,
        dsa_layer_bytes_per_token: kv_config.layer_bytes_per_token(LayerId(
            GLM52_DSA_INDEXER_LAYER_IDS[0] as u32,
        )),
        non_dsa_layer_bytes_per_token: kv_config
            .layer_bytes_per_token(LayerId(non_dsa_layer_id as u32)),
        kv_bytes_per_token: kv_config.bytes_per_token(),
        kv_layer_bytes_sum,
        kv_io_layer_count: kv_io.layer_count,
        kv_io_prefill_writes: kv_io.prefix_prefill_wave_writes + kv_io.later_prefill_wave_writes,
        kv_io_decode_writes: kv_io.decode_wave_writes,
        kv_io_tentative_mtp_writes: kv_io.mtp_tentative_wave_writes,
        kv_io_prefix_read_blocks: kv_io.later_prefill_prefix_read_blocks
            + kv_io.decode_prefix_read_blocks
            + kv_io.mtp_prefix_read_blocks,
        kv_io_backed_bytes_after_discard: kv_io.backed_bytes_after_discard,
        all_attention_layers_bound_to_kv: attention_layers == GLM52_NUM_HIDDEN_LAYERS
            && common_layers_with_required_tensors == GLM52_NUM_HIDDEN_LAYERS
            && dsa_indexer_layers_with_required_tensors == GLM52_DSA_INDEXER_LAYERS
            && non_dsa_layers_without_indexer_tensors
                == GLM52_NUM_HIDDEN_LAYERS - GLM52_DSA_INDEXER_LAYERS
            && bf16_attention_tensors == attention_tensors
            && catalog_dsa_layer_set == config_dsa_layer_set
            && kv_layer_bytes_sum == kv_config.bytes_per_token()
            && kv_io.layer_count == GLM52_NUM_HIDDEN_LAYERS,
    }
}

fn matching_suffix(name: &str, suffixes: &[&'static str]) -> Option<&'static str> {
    suffixes
        .iter()
        .copied()
        .find(|suffix| name.ends_with(suffix))
}
