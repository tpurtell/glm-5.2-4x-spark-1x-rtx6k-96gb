use super::RealFullSchedulerExecutionState;
use crate::commands::real_full::coordinator_kernels::{
    coordinator_w4a16_o_proj_decode_enabled, coordinator_w4a16_q_b_decode_enabled,
    coordinator_w8a16_o_proj_decode_enabled, coordinator_w8a16_packed_o_enabled,
    coordinator_w8a16_q_a_decode_enabled, coordinator_w8a16_q_b_decode_enabled,
};
use anyhow::{Context, Result};
use glmrt_core::{KvCacheConfig, LayerId, TensorCatalog, GLM52_MTP_LAYER_ID};
use glmrt_ffi::{GLMRT_CUDA_GLM_DSA_PACKED_PAGE_BYTES, GLMRT_CUDA_GLM_DSA_PAGE_SIZE};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const REAL_FULL_KV_SNAPSHOT_FORMAT: &str = "glmrt-kv-dsa-v2";
const LEGACY_REAL_FULL_KV_SNAPSHOT_FORMAT: &str = "glmrt-kv-dsa-v1";
const REAL_FULL_KV_SNAPSHOT_SEMANTICS_REVISION: u32 = 1;
const TOKEN_IDS_FILE: &str = "tokens.u32le";
const METADATA_FILE: &str = "metadata.json";

#[derive(Debug, Serialize, Deserialize)]
struct RealFullKvSnapshotLayer {
    layer_id: u32,
    payload_file: String,
    payload_bytes: usize,
    payload_sha256: String,
    dsa_index_file: Option<String>,
    dsa_index_bytes: usize,
    dsa_index_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct RealFullKvSnapshotProducerProfile {
    cache_semantics_revision: u32,
    w4a16_q_b: bool,
    w4a16_o: bool,
    w8a16_q_a: bool,
    w8a16_q_b: bool,
    w8a16_o: bool,
    w8a16_packed_o: bool,
    w8a16_async_attention: bool,
    packed_fp8_mla_direct_hidden_output: bool,
    b12x_direct_route: bool,
    b12x_grouped_decode: bool,
    route_grouped_multirow: bool,
    fused_fp8_reduction: bool,
    nccl_bf16_reduce: bool,
    cuda_reference_kernels: String,
    moe_response_dtype: String,
    moe_owner_response_dtype: String,
    mtp_moe_response_dtype: String,
    intermediate_shards: usize,
    intermediate_reduction: String,
    intermediate_reduction_dtype: String,
    intermediate_owner_reduction_dtype: String,
    intermediate_reduction_min_rows: usize,
    intermediate_owner_max_rows: usize,
    intermediate_row_sharded_reduction: bool,
    target_prefill_chunk_tokens: usize,
    target_large_prefill_min_tokens: usize,
    target_long_prefix_prefill_chunk_tokens: usize,
    mtp_prefill_chunk_tokens: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct RealFullKvSnapshotMetadata {
    format: String,
    model_id: String,
    catalog_sha256: String,
    engine_commit: String,
    token_count: usize,
    token_ids_file: String,
    token_ids_bytes: usize,
    token_ids_sha256: String,
    kv_config: KvCacheConfig,
    dsa_page_tokens: usize,
    dsa_packed_page_bytes: usize,
    transient_attention_frontier: String,
    #[serde(default)]
    mtp_layer_token_count: usize,
    #[serde(default)]
    producer_profile: Option<RealFullKvSnapshotProducerProfile>,
    layers: Vec<RealFullKvSnapshotLayer>,
}

pub(in crate::commands::real_full) struct RealFullKvSnapshot {
    root: PathBuf,
    metadata: RealFullKvSnapshotMetadata,
    token_ids: Vec<usize>,
}

impl RealFullKvSnapshot {
    pub(in crate::commands::real_full) fn root(&self) -> &Path {
        &self.root
    }

    pub(in crate::commands::real_full) fn token_ids(&self) -> &[usize] {
        &self.token_ids
    }

    pub(in crate::commands::real_full) fn token_count(&self) -> usize {
        self.token_ids.len()
    }

    pub(in crate::commands::real_full) fn mtp_layer_token_count(&self) -> usize {
        self.metadata.mtp_layer_token_count
    }

    pub(in crate::commands::real_full) fn is_mtp_ready(&self) -> bool {
        self.metadata.mtp_layer_token_count == self.metadata.token_count
    }

    pub(in crate::commands::real_full) fn restore(
        &self,
        state: &mut RealFullSchedulerExecutionState,
    ) -> Result<()> {
        anyhow::ensure!(
            state.processed_token_ids().is_empty(),
            "KV snapshot restore requires an empty scheduler token frontier"
        );
        for layer in &self.metadata.layers {
            if layer.layer_id as usize == GLM52_MTP_LAYER_ID && !self.is_mtp_ready() {
                // Target-only snapshots predate an MTP prompt-cache build. Do
                // not publish their zero/uninitialized layer-78 payload as an
                // attention-visible committed prefix.
                continue;
            }
            let payload_path = self.root.join(&layer.payload_file);
            let payload =
                read_verified_file(&payload_path, layer.payload_bytes, &layer.payload_sha256)
                    .with_context(|| format!("reading KV snapshot layer {}", layer.layer_id))?;
            let dsa_index_payload = match (
                layer.dsa_index_file.as_deref(),
                layer.dsa_index_sha256.as_deref(),
            ) {
                (Some(file), Some(sha256)) => Some(
                    read_verified_file(&self.root.join(file), layer.dsa_index_bytes, sha256)
                        .with_context(|| {
                            format!("reading packed DSA snapshot layer {}", layer.layer_id)
                        })?,
                ),
                (None, None) => None,
                _ => anyhow::bail!(
                    "KV snapshot layer {} has incomplete DSA metadata",
                    layer.layer_id
                ),
            };
            state.restore_snapshot_layer(
                LayerId(layer.layer_id),
                self.metadata.token_count,
                &payload,
                dsa_index_payload.as_deref(),
            )?;
        }
        state.set_restored_token_ids(self.token_ids.clone())
    }
}

pub(in crate::commands::real_full) fn save_real_full_kv_snapshot(
    state: &mut RealFullSchedulerExecutionState,
    root: &Path,
    catalog: &TensorCatalog,
    engine_commit: &str,
    token_count: usize,
) -> Result<()> {
    let processed_token_ids = state.processed_token_ids();
    anyhow::ensure!(token_count > 0, "cannot save an empty KV snapshot");
    anyhow::ensure!(
        token_count <= processed_token_ids.len(),
        "cannot save {token_count} KV snapshot tokens from a {}-token committed frontier",
        processed_token_ids.len()
    );
    let token_ids = &processed_token_ids[..token_count];
    let mtp_layer_token_count = state.snapshot_mtp_layer_token_count(token_count);
    anyhow::ensure!(
        !root.exists(),
        "KV snapshot destination already exists: {}",
        root.display()
    );
    let parent = root
        .parent()
        .context("KV snapshot destination has no parent directory")?;
    fs::create_dir_all(parent)
        .with_context(|| format!("creating KV snapshot parent {}", parent.display()))?;
    let name = root
        .file_name()
        .and_then(|name| name.to_str())
        .context("KV snapshot destination has no UTF-8 file name")?;
    let staging = parent.join(format!(".{name}.tmp-{}", std::process::id()));
    anyhow::ensure!(
        !staging.exists(),
        "KV snapshot staging destination already exists: {}",
        staging.display()
    );
    fs::create_dir(&staging)
        .with_context(|| format!("creating KV snapshot staging {}", staging.display()))?;
    let mut staging_guard = SnapshotStagingDir::new(staging.clone());

    let token_bytes = token_ids_u32le(token_ids)?;
    let token_ids_sha256 = sha256_hex(&token_bytes);
    write_file(&staging.join(TOKEN_IDS_FILE), &token_bytes)?;

    let config = state.store.config().clone();
    let mut layers = Vec::with_capacity(config.layers);
    for layer_index in 0..config.layers {
        let layer_id =
            LayerId(u32::try_from(layer_index).context("snapshot layer index exceeds u32")?);
        let payload = state
            .snapshot_layer_payload(layer_id, token_count)
            .with_context(|| format!("reading compressed KV snapshot layer {layer_index}"))?;
        let expected_payload_bytes = token_count
            .checked_mul(config.layer_bytes_per_token(layer_id))
            .context("snapshot layer payload byte count overflow usize")?;
        anyhow::ensure!(
            payload.len() == expected_payload_bytes,
            "snapshot layer {layer_index} has {} bytes, expected {expected_payload_bytes}",
            payload.len()
        );
        let payload_file = format!("layer-{layer_index:03}.kv");
        let payload_sha256 = sha256_hex(&payload);
        write_file(&staging.join(&payload_file), &payload)?;

        let dsa_index_payload = state
            .snapshot_dsa_index_prefix(layer_id, token_count)
            .with_context(|| format!("reading packed DSA snapshot layer {layer_index}"))?;
        let (dsa_index_file, dsa_index_bytes, dsa_index_sha256) =
            if let Some(dsa_index_payload) = dsa_index_payload {
                let file = format!("layer-{layer_index:03}.dsa");
                let sha256 = sha256_hex(&dsa_index_payload);
                write_file(&staging.join(&file), &dsa_index_payload)?;
                (Some(file), dsa_index_payload.len(), Some(sha256))
            } else {
                (None, 0, None)
            };
        layers.push(RealFullKvSnapshotLayer {
            layer_id: layer_id.0,
            payload_file,
            payload_bytes: payload.len(),
            payload_sha256,
            dsa_index_file,
            dsa_index_bytes,
            dsa_index_sha256,
        });
    }
    let metadata = RealFullKvSnapshotMetadata {
        format: REAL_FULL_KV_SNAPSHOT_FORMAT.to_owned(),
        model_id: catalog.model_id.clone(),
        catalog_sha256: catalog.content_hash(),
        engine_commit: engine_commit.to_owned(),
        token_count,
        token_ids_file: TOKEN_IDS_FILE.to_owned(),
        token_ids_bytes: token_bytes.len(),
        token_ids_sha256,
        kv_config: config,
        dsa_page_tokens: GLMRT_CUDA_GLM_DSA_PAGE_SIZE,
        dsa_packed_page_bytes: GLMRT_CUDA_GLM_DSA_PACKED_PAGE_BYTES,
        transient_attention_frontier: "not-snapshotted-rebuild-from-packed-cache".to_owned(),
        mtp_layer_token_count,
        producer_profile: Some(current_snapshot_producer_profile()),
        layers,
    };
    let metadata_json =
        serde_json::to_vec_pretty(&metadata).context("serializing KV snapshot metadata")?;
    write_file(&staging.join(METADATA_FILE), &metadata_json)?;
    fs::rename(&staging, root).with_context(|| {
        format!(
            "publishing KV snapshot {} from {}",
            root.display(),
            staging.display()
        )
    })?;
    staging_guard.published = true;
    Ok(())
}

pub(in crate::commands::real_full) fn load_real_full_kv_snapshot(
    root: &Path,
    catalog: &TensorCatalog,
    config: &KvCacheConfig,
) -> Result<RealFullKvSnapshot> {
    let metadata_path = root.join(METADATA_FILE);
    let metadata: RealFullKvSnapshotMetadata = serde_json::from_reader(BufReader::new(
        File::open(&metadata_path)
            .with_context(|| format!("opening KV snapshot metadata {}", metadata_path.display()))?,
    ))
    .with_context(|| format!("parsing KV snapshot metadata {}", metadata_path.display()))?;
    anyhow::ensure!(
        metadata.format != LEGACY_REAL_FULL_KV_SNAPSHOT_FORMAT,
        "legacy KV snapshot format {} has no execution-profile fingerprint and cannot be safely reused; regenerate the snapshot with the current runtime",
        metadata.format
    );
    anyhow::ensure!(
        metadata.format == REAL_FULL_KV_SNAPSHOT_FORMAT,
        "unsupported KV snapshot format {}",
        metadata.format
    );
    anyhow::ensure!(
        metadata.token_ids_file == TOKEN_IDS_FILE,
        "KV snapshot token payload has noncanonical path {}",
        metadata.token_ids_file
    );
    anyhow::ensure!(
        metadata.model_id == catalog.model_id,
        "KV snapshot model {} does not match {}",
        metadata.model_id,
        catalog.model_id
    );
    anyhow::ensure!(
        metadata.catalog_sha256 == catalog.content_hash(),
        "KV snapshot tensor catalog fingerprint does not match the loaded model"
    );
    let current_producer_profile = current_snapshot_producer_profile();
    let saved_producer_profile = metadata.producer_profile.as_ref().context(
        "KV snapshot is missing its execution-profile fingerprint; regenerate it with the current runtime",
    )?;
    validate_compatible_producer_profile(saved_producer_profile, &current_producer_profile)?;
    validate_compatible_config(&metadata.kv_config, config)?;
    anyhow::ensure!(
        metadata.dsa_page_tokens == GLMRT_CUDA_GLM_DSA_PAGE_SIZE
            && metadata.dsa_packed_page_bytes == GLMRT_CUDA_GLM_DSA_PACKED_PAGE_BYTES,
        "KV snapshot packed DSA page geometry {}/{} does not match runtime {}/{}",
        metadata.dsa_page_tokens,
        metadata.dsa_packed_page_bytes,
        GLMRT_CUDA_GLM_DSA_PAGE_SIZE,
        GLMRT_CUDA_GLM_DSA_PACKED_PAGE_BYTES,
    );
    anyhow::ensure!(
        metadata.token_count > 0 && metadata.token_count <= config.max_tokens,
        "KV snapshot token count {} is invalid for capacity {}",
        metadata.token_count,
        config.max_tokens
    );
    anyhow::ensure!(
        metadata.mtp_layer_token_count == 0
            || metadata.mtp_layer_token_count == metadata.token_count,
        "KV snapshot MTP layer frontier {} must be either zero or the full {}-token snapshot frontier",
        metadata.mtp_layer_token_count,
        metadata.token_count,
    );
    anyhow::ensure!(
        metadata.layers.len() == config.layers,
        "KV snapshot has {} layers, expected {}",
        metadata.layers.len(),
        config.layers
    );
    for (layer_index, layer) in metadata.layers.iter().enumerate() {
        anyhow::ensure!(
            layer.layer_id as usize == layer_index,
            "KV snapshot layer metadata is not canonical at index {layer_index}: {}",
            layer.layer_id
        );
        let expected_payload_file = format!("layer-{layer_index:03}.kv");
        anyhow::ensure!(
            layer.payload_file == expected_payload_file,
            "KV snapshot layer {layer_index} payload path {} is not canonical",
            layer.payload_file
        );
        let expected_payload_bytes = metadata
            .token_count
            .checked_mul(config.layer_bytes_per_token(LayerId(layer.layer_id)))
            .context("KV snapshot expected layer bytes overflow usize")?;
        anyhow::ensure!(
            layer.payload_bytes == expected_payload_bytes,
            "KV snapshot layer {} declares {} bytes, expected {expected_payload_bytes}",
            layer.layer_id,
            layer.payload_bytes
        );
        if config.layer_has_dsa_indexer(LayerId(layer.layer_id)) {
            let expected_dsa_file = format!("layer-{layer_index:03}.dsa");
            let expected_dsa_bytes = metadata
                .token_count
                .checked_add(GLMRT_CUDA_GLM_DSA_PAGE_SIZE - 1)
                .context("KV snapshot DSA page rounding overflow usize")?
                / GLMRT_CUDA_GLM_DSA_PAGE_SIZE;
            let expected_dsa_bytes = expected_dsa_bytes
                .checked_mul(GLMRT_CUDA_GLM_DSA_PACKED_PAGE_BYTES)
                .context("KV snapshot DSA byte count overflow usize")?;
            anyhow::ensure!(
                layer.dsa_index_file.as_deref() == Some(expected_dsa_file.as_str())
                    && layer.dsa_index_sha256.is_some()
                    && layer.dsa_index_bytes == expected_dsa_bytes,
                "KV snapshot layer {layer_index} has invalid packed DSA payload metadata"
            );
        } else {
            anyhow::ensure!(
                layer.dsa_index_file.is_none()
                    && layer.dsa_index_sha256.is_none()
                    && layer.dsa_index_bytes == 0,
                "KV snapshot non-indexer layer {layer_index} unexpectedly has packed DSA payload metadata"
            );
        }
    }
    let token_bytes = read_verified_file(
        &root.join(&metadata.token_ids_file),
        metadata.token_ids_bytes,
        &metadata.token_ids_sha256,
    )
    .context("reading KV snapshot token IDs")?;
    let token_ids = token_ids_from_u32le(&token_bytes)?;
    anyhow::ensure!(
        token_ids.len() == metadata.token_count,
        "KV snapshot contains {} token IDs, expected {}",
        token_ids.len(),
        metadata.token_count
    );
    Ok(RealFullKvSnapshot {
        root: root.to_owned(),
        metadata,
        token_ids,
    })
}

struct SnapshotStagingDir {
    path: PathBuf,
    published: bool,
}

impl SnapshotStagingDir {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            published: false,
        }
    }
}

impl Drop for SnapshotStagingDir {
    fn drop(&mut self) {
        if !self.published {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

fn current_snapshot_producer_profile() -> RealFullKvSnapshotProducerProfile {
    let moe_response_dtype = env_or_default("GLMRT_REAL_FULL_MOE_RESPONSE_DTYPE", "bf16");
    let intermediate_reduction =
        env_or_default("GLMRT_EXPERT_INTERMEDIATE_REDUCTION", "coordinator");
    let intermediate_reduction_dtype =
        env_or_default("GLMRT_EXPERT_INTERMEDIATE_REDUCTION_DTYPE", "fp8");
    RealFullKvSnapshotProducerProfile {
        cache_semantics_revision: REAL_FULL_KV_SNAPSHOT_SEMANTICS_REVISION,
        w4a16_q_b: coordinator_w4a16_q_b_decode_enabled(),
        w4a16_o: coordinator_w4a16_o_proj_decode_enabled(),
        w8a16_q_a: coordinator_w8a16_q_a_decode_enabled(),
        w8a16_q_b: coordinator_w8a16_q_b_decode_enabled(),
        w8a16_o: coordinator_w8a16_o_proj_decode_enabled(),
        w8a16_packed_o: coordinator_w8a16_packed_o_enabled(),
        w8a16_async_attention: explicit_truthy_env(
            "GLMRT_COORDINATOR_W8A16_ASYNC_ATTENTION",
            false,
        ),
        packed_fp8_mla_direct_hidden_output: default_true_env(
            "GLMRT_REAL_FULL_PACKED_FP8_MLA_DIRECT_HIDDEN_OUTPUT",
        ),
        b12x_direct_route: default_true_env("GLMRT_B12X_SPARK_DIRECT_ROUTE"),
        b12x_grouped_decode: default_true_env("GLMRT_B12X_SPARK_GROUPED_DECODE"),
        route_grouped_multirow: default_true_env("GLMRT_REAL_FULL_NVFP4_ROUTE_GROUPED_MULTIROW"),
        fused_fp8_reduction: default_true_env("GLMRT_EXPERT_FUSED_FP8_REDUCTION"),
        nccl_bf16_reduce: explicit_truthy_env("GLMRT_EXPERT_NCCL_BF16_REDUCE", false),
        cuda_reference_kernels: env_or_default("GLMRT_REAL_FULL_CUDA_REFERENCE_KERNELS", "false"),
        moe_owner_response_dtype: env_or_default(
            "GLMRT_REAL_FULL_MOE_OWNER_RESPONSE_DTYPE",
            &moe_response_dtype,
        ),
        moe_response_dtype,
        mtp_moe_response_dtype: env_or_default("GLMRT_REAL_FULL_MTP_MOE_RESPONSE_DTYPE", "bf16"),
        intermediate_shards: positive_env_usize("GLMRT_EXPERT_INTERMEDIATE_SHARDS", 1),
        intermediate_owner_reduction_dtype: env_or_default(
            "GLMRT_EXPERT_INTERMEDIATE_OWNER_REDUCTION_DTYPE",
            &intermediate_reduction_dtype,
        ),
        intermediate_reduction_dtype,
        intermediate_reduction_min_rows: positive_env_usize(
            "GLMRT_EXPERT_INTERMEDIATE_REDUCTION_MIN_ROWS",
            if matches!(
                intermediate_reduction.as_str(),
                "spark-owner" | "owner" | "verbs-owner"
            ) {
                1
            } else {
                16
            },
        ),
        intermediate_reduction,
        intermediate_owner_max_rows: positive_env_usize(
            "GLMRT_EXPERT_INTERMEDIATE_OWNER_MAX_ROWS",
            8,
        ),
        intermediate_row_sharded_reduction: explicit_truthy_env(
            "GLMRT_EXPERT_INTERMEDIATE_ROW_SHARDED_REDUCTION",
            false,
        ),
        target_prefill_chunk_tokens: positive_env_usize(
            "GLMRT_REAL_FULL_REQUEST_PREFILL_CHUNK_TOKENS",
            2 * 1024,
        ),
        target_large_prefill_min_tokens: positive_env_usize(
            "GLMRT_REAL_FULL_REQUEST_LARGE_PREFILL_MIN_TOKENS",
            4 * 1024,
        ),
        target_long_prefix_prefill_chunk_tokens: positive_env_usize(
            "GLMRT_REAL_FULL_REQUEST_LONG_PREFIX_SMALL_PREFILL_CHUNK_TOKENS",
            512,
        ),
        mtp_prefill_chunk_tokens: positive_env_usize(
            "GLMRT_REAL_FULL_MTP_PREFILL_CHUNK_TOKENS",
            1024,
        ),
    }
}

fn validate_compatible_producer_profile(
    saved: &RealFullKvSnapshotProducerProfile,
    current: &RealFullKvSnapshotProducerProfile,
) -> Result<()> {
    anyhow::ensure!(
        saved == current,
        "KV snapshot execution profile does not match the current runtime; regenerate it (saved={saved:?}, current={current:?})"
    );
    Ok(())
}

fn positive_env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn env_or_default(name: &str, default: &str) -> String {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default.to_owned())
}

fn explicit_truthy_env(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on" | "enabled"
            )
        })
        .unwrap_or(default)
}

fn default_true_env(name: &str) -> bool {
    env::var(name)
        .ok()
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off" | "disabled"
            )
        })
        .unwrap_or(true)
}

fn validate_compatible_config(saved: &KvCacheConfig, current: &KvCacheConfig) -> Result<()> {
    anyhow::ensure!(
        saved.layout == current.layout
            && saved.layers == current.layers
            && saved.key_value_width == current.key_value_width
            && saved.dtype == current.dtype
            && saved.mla_representation == current.mla_representation
            && saved.dsa_indexer_layers == current.dsa_indexer_layers
            && saved.dsa_index_head_dim == current.dsa_index_head_dim
            && saved.fp8_scale_metadata_bytes_per_token
                == current.fp8_scale_metadata_bytes_per_token,
        "KV snapshot cache layout/configuration does not match the current runtime"
    );
    Ok(())
}

fn token_ids_u32le(token_ids: &[usize]) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(
        token_ids
            .len()
            .checked_mul(std::mem::size_of::<u32>())
            .context("snapshot token byte count overflow usize")?,
    );
    for token_id in token_ids {
        bytes.extend_from_slice(
            &u32::try_from(*token_id)
                .context("snapshot token ID exceeds u32")?
                .to_le_bytes(),
        );
    }
    Ok(bytes)
}

fn token_ids_from_u32le(bytes: &[u8]) -> Result<Vec<usize>> {
    anyhow::ensure!(
        bytes.len() % std::mem::size_of::<u32>() == 0,
        "snapshot token payload byte count {} is not divisible by four",
        bytes.len()
    );
    bytes
        .chunks_exact(std::mem::size_of::<u32>())
        .map(|chunk| {
            usize::try_from(u32::from_le_bytes(
                chunk.try_into().expect("four-byte chunk"),
            ))
            .context("snapshot token ID does not fit usize")
        })
        .collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut writer = BufWriter::new(
        File::create(path).with_context(|| format!("creating snapshot file {}", path.display()))?,
    );
    writer
        .write_all(bytes)
        .with_context(|| format!("writing snapshot file {}", path.display()))?;
    writer
        .flush()
        .with_context(|| format!("flushing snapshot file {}", path.display()))
}

fn read_verified_file(
    path: &Path,
    expected_bytes: usize,
    expected_sha256: &str,
) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(expected_bytes);
    BufReader::new(
        File::open(path).with_context(|| format!("opening snapshot file {}", path.display()))?,
    )
    .read_to_end(&mut bytes)
    .with_context(|| format!("reading snapshot file {}", path.display()))?;
    anyhow::ensure!(
        bytes.len() == expected_bytes,
        "snapshot file {} has {} bytes, expected {expected_bytes}",
        path.display(),
        bytes.len()
    );
    let actual_sha256 = sha256_hex(&bytes);
    anyhow::ensure!(
        actual_sha256 == expected_sha256,
        "snapshot file {} checksum mismatch: expected {expected_sha256}, got {actual_sha256}",
        path.display()
    );
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::{
        current_snapshot_producer_profile, token_ids_from_u32le, token_ids_u32le,
        validate_compatible_producer_profile,
    };

    #[test]
    fn snapshot_token_ids_roundtrip_u32le() {
        let token_ids = vec![0, 1, 154_879, u32::MAX as usize];
        let bytes = token_ids_u32le(&token_ids).unwrap();
        assert_eq!(token_ids_from_u32le(&bytes).unwrap(), token_ids);
    }

    #[test]
    fn snapshot_token_ids_reject_partial_word() {
        assert!(token_ids_from_u32le(&[1, 2, 3]).is_err());
    }

    #[test]
    fn snapshot_producer_profile_requires_exact_semantic_match() {
        let saved = current_snapshot_producer_profile();
        assert!(validate_compatible_producer_profile(&saved, &saved).is_ok());

        let mut changed = saved.clone();
        changed.w8a16_q_a = !changed.w8a16_q_a;
        let error = validate_compatible_producer_profile(&saved, &changed).unwrap_err();
        assert!(error
            .to_string()
            .contains("KV snapshot execution profile does not match"));

        let mut revised = saved.clone();
        revised.cache_semantics_revision += 1;
        assert!(validate_compatible_producer_profile(&saved, &revised).is_err());
    }
}
