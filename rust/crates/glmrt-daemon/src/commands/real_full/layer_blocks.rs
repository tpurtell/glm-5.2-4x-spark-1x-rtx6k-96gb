use anyhow::{Context, Result};
use glmrt_core::{
    KvCacheConfig, KvCacheDType, MlaKvCacheRepresentation, TensorInfo, TensorRole,
    GLM52_MTP_LAYER_ID, GLM52_NUM_HIDDEN_LAYERS, GLM52_TOTAL_LAYERS_WITH_MTP,
};
use std::env;
use std::net::{SocketAddr, ToSocketAddrs};

use super::intermediate_sharding::ExpertIntermediateShard;

pub(crate) const SPARK_LAYER_BLOCKS_ENV: &str = "GLMRT_SPARK_LAYER_BLOCKS";
pub(crate) const SPARK_LAYER_BLOCK_RANGE_ENV: &str = "GLMRT_SPARK_LAYER_BLOCK_RANGE";
pub(crate) const SPARK_LAYER_BLOCK_OWNER_ENDPOINT_ENV: &str =
    "GLMRT_SPARK_LAYER_BLOCK_OWNER_ENDPOINT";
pub(crate) const SPARK_LAYER_BLOCK_KV_DTYPE_ENV: &str = "GLMRT_SPARK_LAYER_BLOCK_KV_DTYPE";
const DEFAULT_SPARK_LAYER_BLOCK_OWNER_ENDPOINT: &str = "127.0.0.1:9100";
const DEFAULT_SPARK_LAYER_BLOCK_MAX_TOKENS: usize = 4096;

// The first three dense layers are almost twice the size of a sparse layer.
// These boundaries keep the four GLM-5.2 blocks within roughly 300 MiB.
const GLM52_TP4_LAYER_BLOCK_BOUNDARIES: [usize; 5] = [0, 18, 38, 58, 78];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SparkLayerBlock {
    pub(crate) start_layer: usize,
    pub(crate) end_layer: usize,
}

impl SparkLayerBlock {
    pub(crate) fn new(start_layer: usize, end_layer: usize) -> Result<Self> {
        anyhow::ensure!(
            start_layer < end_layer,
            "Spark layer block start {start_layer} must precede end {end_layer}"
        );
        anyhow::ensure!(
            end_layer <= GLM52_NUM_HIDDEN_LAYERS,
            "Spark layer block end {end_layer} exceeds {GLM52_NUM_HIDDEN_LAYERS} layers"
        );
        Ok(Self {
            start_layer,
            end_layer,
        })
    }

    pub(crate) fn mtp() -> Self {
        Self {
            start_layer: GLM52_MTP_LAYER_ID,
            end_layer: GLM52_TOTAL_LAYERS_WITH_MTP,
        }
    }

    fn for_tp4_rank(rank: usize) -> Result<Self> {
        anyhow::ensure!(rank < 4, "Spark layer block rank {rank} is outside 0..4");
        Self::new(
            GLM52_TP4_LAYER_BLOCK_BOUNDARIES[rank],
            GLM52_TP4_LAYER_BLOCK_BOUNDARIES[rank + 1],
        )
    }

    pub(crate) fn contains(self, layer_id: usize) -> bool {
        (self.start_layer..self.end_layer).contains(&layer_id)
    }

    pub(crate) fn layer_count(self) -> usize {
        self.end_layer - self.start_layer
    }
}

pub(crate) fn spark_layer_block_from_env(
    shard: Option<ExpertIntermediateShard>,
) -> Result<Option<SparkLayerBlock>> {
    let enabled = env::var(SPARK_LAYER_BLOCKS_ENV)
        .ok()
        .map(|value| parse_enabled(&value))
        .transpose()?
        .unwrap_or(false);
    let explicit = env::var(SPARK_LAYER_BLOCK_RANGE_ENV).ok();
    if !enabled && explicit.is_none() {
        return Ok(None);
    }
    if let Some(raw) = explicit {
        return parse_layer_block_range(&raw).map(Some);
    }
    let shard = shard.with_context(|| {
        format!("{SPARK_LAYER_BLOCKS_ENV}=1 requires four-way intermediate sharding")
    })?;
    anyhow::ensure!(
        shard.count == 4,
        "Spark layer blocks currently require four intermediate shards"
    );
    SparkLayerBlock::for_tp4_rank(shard.rank).map(Some)
}

pub(crate) fn tensor_is_spark_layer_block_resident(
    tensor: &TensorInfo,
    block: SparkLayerBlock,
) -> bool {
    tensor
        .layer_id
        .map(|layer_id| block.contains(layer_id as usize))
        .unwrap_or(false)
        && !tensor.is_quantization_metadata
        && matches!(
            tensor.role,
            TensorRole::Attention
                | TensorRole::Router
                | TensorRole::Norm
                | TensorRole::DenseMlp
                | TensorRole::SharedExpert
        )
}

pub(crate) fn spark_layer_block_owner_endpoint_from_env() -> Result<SocketAddr> {
    let endpoint = env::var(SPARK_LAYER_BLOCK_OWNER_ENDPOINT_ENV)
        .unwrap_or_else(|_| DEFAULT_SPARK_LAYER_BLOCK_OWNER_ENDPOINT.to_owned());
    endpoint
        .to_socket_addrs()
        .with_context(|| format!("resolving {SPARK_LAYER_BLOCK_OWNER_ENDPOINT_ENV}={endpoint}"))?
        .next()
        .with_context(|| {
            format!("{SPARK_LAYER_BLOCK_OWNER_ENDPOINT_ENV}={endpoint} resolved no addresses")
        })
}

pub(crate) fn spark_layer_block_kv_config_from_env() -> Result<KvCacheConfig> {
    let raw = env::var(SPARK_LAYER_BLOCK_KV_DTYPE_ENV).unwrap_or_else(|_| "fp8".to_owned());
    spark_layer_block_kv_config(&raw)
}

fn spark_layer_block_kv_config(raw: &str) -> Result<KvCacheConfig> {
    let dtype = KvCacheDType::parse_glm52_cache_dtype(&raw).with_context(|| {
        format!("{SPARK_LAYER_BLOCK_KV_DTYPE_ENV} must be bf16, fp8, or nvfp4, got {raw}")
    })?;
    KvCacheConfig::glm52_compressed(DEFAULT_SPARK_LAYER_BLOCK_MAX_TOKENS, dtype)
        .map(|config| config.with_mla_representation(MlaKvCacheRepresentation::NormalizedRotated))
        .with_context(|| format!("building Spark layer-block {raw} KV cache configuration"))
}

fn parse_enabled(raw: &str) -> Result<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" | "" => Ok(false),
        value => anyhow::bail!("{SPARK_LAYER_BLOCKS_ENV} must be boolean-like, got {value}"),
    }
}

fn parse_layer_block_range(raw: &str) -> Result<SparkLayerBlock> {
    let (start, end) = raw.trim().split_once(':').with_context(|| {
        format!("{SPARK_LAYER_BLOCK_RANGE_ENV} must use start:end (end exclusive), got {raw}")
    })?;
    let start_layer = start
        .parse::<usize>()
        .with_context(|| format!("parsing {SPARK_LAYER_BLOCK_RANGE_ENV} start"))?;
    let end_layer = end
        .parse::<usize>()
        .with_context(|| format!("parsing {SPARK_LAYER_BLOCK_RANGE_ENV} end"))?;
    SparkLayerBlock::new(start_layer, end_layer)
}

#[cfg(test)]
mod tests {
    use super::{
        parse_enabled, parse_layer_block_range, spark_layer_block_kv_config, SparkLayerBlock,
        GLM52_TP4_LAYER_BLOCK_BOUNDARIES,
    };
    use glmrt_core::{KvCacheDType, MlaKvCacheRepresentation};

    #[test]
    fn tp4_blocks_cover_every_layer_once() {
        assert_eq!(GLM52_TP4_LAYER_BLOCK_BOUNDARIES, [0, 18, 38, 58, 78]);
        let blocks = (0..4)
            .map(|rank| SparkLayerBlock::for_tp4_rank(rank).unwrap())
            .collect::<Vec<_>>();
        for layer in 0..78 {
            assert_eq!(
                blocks.iter().filter(|block| block.contains(layer)).count(),
                1,
                "layer {layer} must have exactly one owner"
            );
        }
    }

    #[test]
    fn explicit_range_is_end_exclusive_and_validated() {
        assert_eq!(
            parse_layer_block_range("18:38").unwrap(),
            SparkLayerBlock {
                start_layer: 18,
                end_layer: 38
            }
        );
        assert!(parse_layer_block_range("38:18").is_err());
        assert!(parse_layer_block_range("0:79").is_err());
    }

    #[test]
    fn mtp_block_selects_only_the_next_token_prediction_layer() {
        let block = SparkLayerBlock::mtp();
        assert_eq!(block.start_layer, 78);
        assert_eq!(block.end_layer, 79);
        assert_eq!(block.layer_count(), 1);
        assert!(block.contains(78));
        assert!(!block.contains(77));
        assert!(!block.contains(79));
    }

    #[test]
    fn enable_flag_rejects_unknown_values() {
        assert!(parse_enabled("on").unwrap());
        assert!(!parse_enabled("off").unwrap());
        assert!(parse_enabled("maybe").is_err());
    }

    #[test]
    fn layer_block_kv_uses_packed_attention_representation() {
        for (raw, dtype) in [
            ("bf16", KvCacheDType::Bf16),
            ("fp8", KvCacheDType::Fp8),
            ("nvfp4", KvCacheDType::Nvfp4),
        ] {
            let config = spark_layer_block_kv_config(raw).unwrap();
            assert_eq!(config.dtype, dtype);
            assert_eq!(
                config.mla_representation,
                MlaKvCacheRepresentation::NormalizedRotated
            );
        }
    }
}
