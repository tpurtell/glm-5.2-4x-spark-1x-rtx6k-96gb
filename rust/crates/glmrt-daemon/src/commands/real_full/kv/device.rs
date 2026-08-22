use anyhow::{Context, Result};
use glmrt_core::{
    KvBackedBlock, KvBlockDescriptor, KvCacheConfig, KvCacheDType, KvLayout, LayerId,
    MlaKvCacheRepresentation, PositionId, GLM52_DSA_INDEX_HEAD_DIM, GLM52_HIDDEN_SIZE,
    GLM52_MLA_FP8_DS_BYTES_PER_TOKEN, GLM52_MLA_FP8_DS_SCALE_BYTES_PER_TOKEN,
    GLM52_MLA_KV_LORA_RANK, GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN, GLM52_MLA_QK_ROPE_HEAD_DIM,
    GLM52_MLA_ROPE_THETA,
};
use glmrt_ffi::{
    GlmrtDeviceBuffer, GlmrtHostBuffer, NativeLibrary, GLMRT_CUDA_GLM_DSA_PACKED_PAGE_BYTES,
    GLMRT_CUDA_GLM_DSA_PAGE_SIZE,
};
use std::{
    collections::BTreeMap,
    env, fs,
    os::raw::c_void,
    path::{Path, PathBuf},
    slice,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex, OnceLock,
    },
    time::Instant,
};

use crate::commands::real_full::coordinator_kernels::{
    coordinator_cuda_reference_kernels_enabled, coordinator_w4a16_o_proj_decode_enabled,
    coordinator_w8a16_o_proj_decode_enabled, cuda_native_library,
    device_bf16_output_from_owned_device_buffer, flashinfer_compressed_mla_decode_device_buffers,
    flashinfer_glm_dsa_sparse_mla_prefill_device_buffers, glm_dsa_index_source_layer,
    glm_dsa_sparse_mla_prefill_supported, linear_rows_bf16_device_buffers_for_layer,
    linear_rows_bf16_preloaded_resident_weight_device_output,
    linear_rows_w8a16_preloaded_resident_weight_device_output,
    mla_decode_kv_commit_bf16_device_output, mla_kv_cache_unpack_bf16_device_buffers_for_layer,
    mla_kv_prepare_bf16_device_buffers_for_layer,
    mla_kv_projected_split_bf16_device_buffers_for_layer,
    mla_query_split_rope_bf16_device_buffers_for_layer,
    mla_query_split_rope_bf16_device_positions_for_layer,
    mla_rope_attention_device_buffers_bf16_for_layer,
    mla_rope_attention_suffix_device_buffers_bf16_for_layer,
    preloaded_coordinator_w4a16_projection, preloaded_coordinator_w8a16_projection,
    preloaded_resident_weight_device_buffer, rmsnorm_bf16_device_buffers_for_layer,
    rope_bf16_device_buffers_for_layer, CoordinatorCudaEvent, DeviceBf16Output,
    FlashinferCompressedMlaKvInput, FlashinferGlmDsaSparseMlaPrefillInput,
    FlashinferMlaHiddenProjection, FlashinferTargetKvPageTable, MlaDecodeKvDsaProjectionWeights,
};
use crate::python_graph_capture::coordinator_python_capture_startup_open;

const REAL_FULL_SCHEDULER_DEVICE_ATTENTION_HEADS: usize = 2;
const REAL_FULL_SCHEDULER_DEVICE_ATTENTION_NOPE_DIM: usize = 3;
const REAL_FULL_SCHEDULER_DEVICE_ATTENTION_VALUE_DIM: usize = 2;
const REAL_FULL_SCHEDULER_DEVICE_ATTENTION_EPS: f32 = 1.0e-5;
const REAL_FULL_SCHEDULER_DEVICE_ATTENTION_SCALE: f32 = 0.25;
const REAL_FULL_COMPRESSED_MLA_MAX_QUERY_ROWS: usize = 16;
const REAL_FULL_PACKED_FP8_MLA_MAX_ROWS: usize = 2_048;
const REAL_FULL_DSA_TOP_K: usize = 2_048;
const DEFAULT_REAL_FULL_ATTENTION_READY_FRONTIER_MAX_TOKENS: usize = 16 * 1024;
const REAL_FULL_ATTENTION_READY_FRONTIER_MAX_TOKENS_ENV: &str =
    "GLMRT_REAL_FULL_ATTENTION_READY_FRONTIER_MAX_TOKENS";
const ADMISSION_STAGE_TIMING_ENV: &str = "GLMRT_REAL_FULL_ADMISSION_STAGE_TIMING";
const DSA_OUTPUT_VALIDATE_ENV: &str = "GLMRT_REAL_FULL_DSA_OUTPUT_VALIDATE";
const PACKED_FP8_MLA_BATCHED_SUFFIX_ENV: &str = "GLMRT_REAL_FULL_PACKED_FP8_MLA_BATCHED_SUFFIX";
const DSPARK_TRACE_ENV: &str = "GLMRT_REAL_FULL_DSPARK_TRACE";
const PACKED_MLA_TRACE_DIR_ENV: &str = "GLMRT_REAL_FULL_PACKED_MLA_TRACE_DIR";
const PACKED_MLA_TRACE_LAYERS_ENV: &str = "GLMRT_REAL_FULL_PACKED_MLA_TRACE_LAYERS";
const PACKED_MLA_TRACE_ROWS_ENV: &str = "GLMRT_REAL_FULL_PACKED_MLA_TRACE_ROWS";
const PACKED_MLA_TRACE_QUERY_ROWS_ENV: &str = "GLMRT_REAL_FULL_PACKED_MLA_TRACE_QUERY_ROWS";
const PACKED_MLA_TRACE_LIMIT_ENV: &str = "GLMRT_REAL_FULL_PACKED_MLA_TRACE_LIMIT";
static NEXT_PHYSICAL_PAGE_TABLE_KEY: AtomicU64 = AtomicU64::new(1);

fn next_physical_page_table_key() -> u64 {
    NEXT_PHYSICAL_PAGE_TABLE_KEY.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug)]
struct PackedMlaTraceConfig {
    directory: PathBuf,
    layers: Option<Vec<usize>>,
    rows: Option<Vec<usize>>,
    query_rows: Option<Vec<usize>>,
    limit: usize,
}

impl PackedMlaTraceConfig {
    fn matches(&self, layer: usize, rows: usize, query_rows: usize) -> bool {
        self.layers
            .as_ref()
            .is_none_or(|values| values.contains(&layer))
            && self
                .rows
                .as_ref()
                .is_none_or(|values| values.contains(&rows))
            && self
                .query_rows
                .as_ref()
                .is_none_or(|values| values.contains(&query_rows))
    }
}

fn parse_packed_mla_trace_filter(name: &str) -> Option<Vec<usize>> {
    let value = env::var(name).ok()?;
    let values = value
        .split(',')
        .filter_map(|part| part.trim().parse::<usize>().ok())
        .collect::<Vec<_>>();
    (!values.is_empty()).then_some(values)
}

fn packed_mla_trace_config() -> Option<&'static PackedMlaTraceConfig> {
    static CONFIG: OnceLock<Option<PackedMlaTraceConfig>> = OnceLock::new();
    CONFIG
        .get_or_init(|| {
            let directory = env::var_os(PACKED_MLA_TRACE_DIR_ENV).map(PathBuf::from)?;
            let limit = env::var(PACKED_MLA_TRACE_LIMIT_ENV)
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .filter(|value| *value > 0)
                .unwrap_or(2);
            Some(PackedMlaTraceConfig {
                directory,
                layers: parse_packed_mla_trace_filter(PACKED_MLA_TRACE_LAYERS_ENV),
                rows: parse_packed_mla_trace_filter(PACKED_MLA_TRACE_ROWS_ENV),
                query_rows: parse_packed_mla_trace_filter(PACKED_MLA_TRACE_QUERY_ROWS_ENV),
                limit,
            })
        })
        .as_ref()
}

fn copy_device_buffer_bytes(
    library: &NativeLibrary,
    buffer: GlmrtDeviceBuffer,
    offset_bytes: usize,
    bytes: usize,
    context: &str,
) -> Result<Vec<u8>> {
    let view = device_buffer_byte_view(buffer, offset_bytes, bytes, context)?;
    let mut host = vec![0_u8; bytes];
    library
        .copy_d2h(&mut host, view)
        .with_context(|| format!("copying {context} to trace host storage"))?;
    Ok(host)
}

fn write_trace_bytes(directory: &Path, name: &str, bytes: Vec<u8>) -> Result<()> {
    fs::write(directory.join(name), bytes)
        .with_context(|| format!("writing packed MLA trace {name}"))
}

fn use_compressed_mla_suffix_attention(prefix_rows: usize, query_rows: usize) -> bool {
    prefix_rows > 0 && (1..=REAL_FULL_COMPRESSED_MLA_MAX_QUERY_ROWS).contains(&query_rows)
}

fn attention_ready_frontier_max_tokens() -> usize {
    static LIMIT: OnceLock<usize> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        env::var(REAL_FULL_ATTENTION_READY_FRONTIER_MAX_TOKENS_ENV)
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_REAL_FULL_ATTENTION_READY_FRONTIER_MAX_TOKENS)
    })
}

fn attention_ready_frontier_capacity_tokens_with_limit(
    max_context_tokens: usize,
    frontier_max_tokens: usize,
) -> usize {
    max_context_tokens.min(frontier_max_tokens)
}

fn attention_ready_frontier_capacity_tokens(max_context_tokens: usize) -> usize {
    attention_ready_frontier_capacity_tokens_with_limit(
        max_context_tokens,
        attention_ready_frontier_max_tokens(),
    )
}

fn attention_ready_frontier_row_stride_bytes(dtype: KvCacheDType) -> Result<usize> {
    match dtype {
        KvCacheDType::Nvfp4 => Ok(GLM52_MLA_FP8_DS_BYTES_PER_TOKEN),
        KvCacheDType::Bf16 | KvCacheDType::Fp8 => (GLM52_MLA_KV_LORA_RANK
            + GLM52_MLA_QK_ROPE_HEAD_DIM)
            .checked_mul(std::mem::size_of::<u16>())
            .context("attention-ready MLA BF16 row stride overflows usize"),
        other => anyhow::bail!(
            "attention-ready MLA frontier does not support {} cache storage",
            other.label()
        ),
    }
}

fn device_attention_stage_timing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(ADMISSION_STAGE_TIMING_ENV)
            .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
            .unwrap_or(false)
    })
}

fn dsa_output_validate_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(DSA_OUTPUT_VALIDATE_ENV)
            .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
            .unwrap_or(false)
    })
}

fn packed_fp8_mla_batched_suffix_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(PACKED_FP8_MLA_BATCHED_SUFFIX_ENV)
            .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
            .unwrap_or(true)
    })
}

fn dspark_attention_route_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(DSPARK_TRACE_ENV)
            .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
            .unwrap_or(false)
    })
}

fn use_packed_fp8_mla_suffix(query_rows: usize, batched_suffix_enabled: bool) -> bool {
    query_rows == 1
        || (batched_suffix_enabled
            && (2..=REAL_FULL_COMPRESSED_MLA_MAX_QUERY_ROWS).contains(&query_rows))
}

fn use_direct_glm_dsa_sparse_mla_prefill(
    config: &KvCacheConfig,
    descriptors: &[KvBlockDescriptor],
    positions: &[u32],
    q_suffix_positions: &[u32],
    dsa_query: Option<(GlmrtDeviceBuffer, GlmrtDeviceBuffer)>,
) -> bool {
    let Some(first) = descriptors.first() else {
        return false;
    };
    let Some((_source_layer, full_indexer)) = glm_dsa_index_source_layer(first.layer_id.0 as usize)
    else {
        return false;
    };
    if !matches!(
        config.dtype,
        KvCacheDType::Bf16 | KvCacheDType::Fp8 | KvCacheDType::Nvfp4
    ) || config.mla_representation != MlaKvCacheRepresentation::NormalizedRotated
        || !glm_dsa_sparse_mla_prefill_supported(q_suffix_positions.len(), config.max_tokens)
        || first.token_start.0 != 0
    {
        return false;
    }
    let mut expected_position = 0_u64;
    for descriptor in descriptors {
        if descriptor.layer_id != first.layer_id
            || descriptor.reservation_id != first.reservation_id
            || descriptor.sequence_id != first.sequence_id
            || descriptor.token_start.0 != expected_position
        {
            return false;
        }
        let Some(next) = expected_position.checked_add(descriptor.token_count as u64) else {
            return false;
        };
        expected_position = next;
    }
    let Ok(rows) = usize::try_from(expected_position) else {
        return false;
    };
    if rows != positions.len() || q_suffix_positions.len() > rows {
        return false;
    }
    let prefix_rows = rows - q_suffix_positions.len();
    if (!full_indexer && dsa_query.is_some())
        || (full_indexer && rows > REAL_FULL_DSA_TOP_K && dsa_query.is_none())
        || (config.dtype == KvCacheDType::Fp8
            && prefix_rows > 0
            && rows <= REAL_FULL_PACKED_FP8_MLA_MAX_ROWS
            && q_suffix_positions.len() <= REAL_FULL_COMPRESSED_MLA_MAX_QUERY_ROWS)
    {
        return false;
    }
    positions
        .iter()
        .enumerate()
        .all(|(index, position)| *position as usize == index)
        && q_suffix_positions
            .iter()
            .enumerate()
            .all(|(index, position)| *position as usize == prefix_rows + index)
}

fn elapsed_ms_optional(start: Option<Instant>) -> f64 {
    start
        .map(|start| start.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::commands::real_full) struct RealFullDeviceKvBlockIo {
    pub(in crate::commands::real_full) offset_bytes: usize,
    pub(in crate::commands::real_full) payload_bytes: usize,
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::commands::real_full) struct RealFullDeviceKvRoundTrip {
    pub(in crate::commands::real_full) status: &'static str,
    pub(in crate::commands::real_full) writes: usize,
    pub(in crate::commands::real_full) reads: usize,
    pub(in crate::commands::real_full) bytes: usize,
    pub(in crate::commands::real_full) uses_device_kv_cache: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::commands::real_full) struct RealFullDeviceKvExecutionSummary {
    pub(in crate::commands::real_full) status: &'static str,
    pub(in crate::commands::real_full) writes: usize,
    pub(in crate::commands::real_full) reads: usize,
    pub(in crate::commands::real_full) bytes: usize,
    pub(in crate::commands::real_full) scheduler_attention_resident_uploads: usize,
    pub(in crate::commands::real_full) scheduler_attention_resident_buffer_uses: usize,
    pub(in crate::commands::real_full) scheduler_attention_resident_query_shapes: usize,
    pub(in crate::commands::real_full) uses_device_kv_cache: bool,
}

pub(in crate::commands::real_full) struct RealFullDeviceMlaDecodeKvCommit {
    pub(in crate::commands::real_full) writes: Vec<RealFullDeviceKvBlockIo>,
    pub(in crate::commands::real_full) normalized_hidden: DeviceBf16Output,
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::commands::real_full) struct RealFullDeviceMlaKvUnpackReadback {
    pub(in crate::commands::real_full) status: &'static str,
    pub(in crate::commands::real_full) rows: usize,
    pub(in crate::commands::real_full) payload_bytes: usize,
    pub(in crate::commands::real_full) kv_latent_bf16: Vec<u8>,
    pub(in crate::commands::real_full) k_rope_bf16: Vec<u8>,
    pub(in crate::commands::real_full) dsa_key_bf16: Option<Vec<u8>>,
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::commands::real_full) struct RealFullDeviceMlaKvProjectedReadback {
    pub(in crate::commands::real_full) status: &'static str,
    pub(in crate::commands::real_full) rows: usize,
    pub(in crate::commands::real_full) heads: usize,
    pub(in crate::commands::real_full) nope_dim: usize,
    pub(in crate::commands::real_full) v_dim: usize,
    pub(in crate::commands::real_full) normalized_bf16: Vec<u8>,
    pub(in crate::commands::real_full) projected_bf16: Vec<u8>,
    pub(in crate::commands::real_full) k_nope_bf16: Vec<u8>,
    pub(in crate::commands::real_full) values_bf16: Vec<u8>,
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::commands::real_full) struct RealFullDeviceMlaKvRopeReadback {
    pub(in crate::commands::real_full) status: &'static str,
    pub(in crate::commands::real_full) rows: usize,
    pub(in crate::commands::real_full) rotary_dim: usize,
    pub(in crate::commands::real_full) k_rope_rotated_bf16: Vec<u8>,
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::commands::real_full) struct RealFullDeviceMlaQueryReadback {
    pub(in crate::commands::real_full) status: &'static str,
    pub(in crate::commands::real_full) rows: usize,
    pub(in crate::commands::real_full) prefix_rows: usize,
    pub(in crate::commands::real_full) suffix_rows: usize,
    pub(in crate::commands::real_full) heads: usize,
    pub(in crate::commands::real_full) nope_dim: usize,
    pub(in crate::commands::real_full) rope_dim: usize,
    pub(in crate::commands::real_full) q_nope_bf16: Vec<u8>,
    pub(in crate::commands::real_full) q_rope_rotated_bf16: Vec<u8>,
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::commands::real_full) struct RealFullDeviceMlaAttentionReadback {
    pub(in crate::commands::real_full) status: &'static str,
    pub(in crate::commands::real_full) rows: usize,
    pub(in crate::commands::real_full) heads: usize,
    pub(in crate::commands::real_full) nope_dim: usize,
    pub(in crate::commands::real_full) rope_dim: usize,
    pub(in crate::commands::real_full) v_dim: usize,
    pub(in crate::commands::real_full) output_bf16: Vec<u8>,
}

#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) struct RealFullDeviceMlaKvDeviceParts {
    status: &'static str,
    layer_id: LayerId,
    rows: usize,
    payload_bytes: usize,
    payload_stride_bytes: usize,
    kv_latent_bytes: usize,
    k_rope_bytes: usize,
    dsa_key_bytes: usize,
    kv_latent: DeviceBufferGuard<'static>,
    k_rope: DeviceBufferGuard<'static>,
    dsa_key: Option<DeviceBufferGuard<'static>>,
}

#[cfg_attr(not(test), allow(dead_code))]
struct RealFullDeviceMlaKvReadShape {
    layer_id: LayerId,
    dtype: KvCacheDType,
    rows: usize,
    dsa_dim: usize,
    payload_stride_bytes: usize,
    payload_bytes: usize,
    kv_latent_bytes: usize,
    k_rope_bytes: usize,
    dsa_key_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
struct RealFullDeviceMlaKvDirectSpan {
    rows: usize,
    row_offset: usize,
    dtype: KvCacheDType,
    row_stride_bytes: usize,
    payload: GlmrtDeviceBuffer,
    physical_page_table: Option<FlashinferTargetKvPageTable>,
    force_staged_hidden_projection: bool,
}

#[allow(clippy::too_many_arguments)]
fn maybe_dump_packed_mla_trace(
    library: &NativeLibrary,
    descriptors: &[KvBlockDescriptor],
    positions: &[u32],
    query_positions: &[u32],
    span: RealFullDeviceMlaKvDirectSpan,
    q_projected: GlmrtDeviceBuffer,
    q_nope: GlmrtDeviceBuffer,
    q_rope_rotated: GlmrtDeviceBuffer,
    kv_b_weight: GlmrtDeviceBuffer,
    hidden_projection: FlashinferMlaHiddenProjection,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    eps: f32,
    theta: f32,
    scale: f32,
) -> Result<()> {
    static TRACE_COUNT: AtomicUsize = AtomicUsize::new(0);
    static WEIGHT_WRITE_LOCK: Mutex<()> = Mutex::new(());

    let Some(config) = packed_mla_trace_config() else {
        return Ok(());
    };
    let layer = descriptors
        .first()
        .context("packed MLA trace requires at least one descriptor")?
        .layer_id
        .0 as usize;
    let query_rows = query_positions.len();
    if !config.matches(layer, span.rows, query_rows) {
        return Ok(());
    }
    let Some(w8a16) = hidden_projection.w8a16 else {
        return Ok(());
    };
    if !w8a16.packed_layout {
        return Ok(());
    }
    let ordinal = TRACE_COUNT.fetch_add(1, Ordering::Relaxed);
    if ordinal >= config.limit {
        return Ok(());
    }

    let trace_directory = config
        .directory
        .join(format!("trace_{}_{ordinal:03}", std::process::id()));
    fs::create_dir_all(&trace_directory).with_context(|| {
        format!(
            "creating packed MLA trace directory {}",
            trace_directory.display()
        )
    })?;

    let bf16_bytes = std::mem::size_of::<u16>();
    let q_projected_bytes = query_rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(nope_dim + rope_dim))
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("packed MLA trace projected Q bytes overflow usize")?;
    let q_nope_bytes = query_rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(nope_dim))
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("packed MLA trace Q nope bytes overflow usize")?;
    let q_rope_bytes = query_rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(rope_dim))
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("packed MLA trace Q rope bytes overflow usize")?;
    let packed_kv_bytes = span
        .rows
        .checked_mul(span.row_stride_bytes)
        .context("packed MLA trace KV span bytes overflow usize")?;
    let packed_kv_offset = span
        .row_offset
        .checked_mul(span.row_stride_bytes)
        .context("packed MLA trace KV offset overflows usize")?;
    let kv_b_weight_bytes = heads
        .checked_mul(nope_dim + v_dim)
        .and_then(|values| values.checked_mul(GLM52_MLA_KV_LORA_RANK))
        .and_then(|values| values.checked_mul(bf16_bytes))
        .context("packed MLA trace KV-B weight bytes overflow usize")?;

    write_trace_bytes(
        &trace_directory,
        "q_projected.bf16",
        copy_device_buffer_bytes(
            library,
            q_projected,
            0,
            q_projected_bytes,
            "packed MLA projected Q",
        )?,
    )?;
    write_trace_bytes(
        &trace_directory,
        "q_nope.bf16",
        copy_device_buffer_bytes(library, q_nope, 0, q_nope_bytes, "packed MLA split Q nope")?,
    )?;
    write_trace_bytes(
        &trace_directory,
        "q_rope_rotated.bf16",
        copy_device_buffer_bytes(
            library,
            q_rope_rotated,
            0,
            q_rope_bytes,
            "packed MLA rotated Q rope",
        )?,
    )?;
    write_trace_bytes(
        &trace_directory,
        "packed_kv.fp8",
        copy_device_buffer_bytes(
            library,
            span.payload,
            packed_kv_offset,
            packed_kv_bytes,
            "packed MLA physical KV span",
        )?,
    )?;
    let weight_relative_directory = PathBuf::from("../weights").join(format!("layer_{layer:03}"));
    let weight_directory = trace_directory.join(&weight_relative_directory);
    let weight_manifest = weight_directory.join("meta.json");
    {
        let _weight_write_guard = WEIGHT_WRITE_LOCK
            .lock()
            .map_err(|_| anyhow::anyhow!("packed MLA trace weight-write lock is poisoned"))?;
        fs::create_dir_all(&weight_directory).with_context(|| {
            format!(
                "creating packed MLA trace weight directory {}",
                weight_directory.display()
            )
        })?;
        if !weight_manifest.is_file() {
            write_trace_bytes(
                &weight_directory,
                "kv_b_weight.bf16",
                copy_device_buffer_bytes(
                    library,
                    kv_b_weight,
                    0,
                    kv_b_weight_bytes,
                    "packed MLA KV-B weight",
                )?,
            )?;
            write_trace_bytes(
                &weight_directory,
                "o_weight_w8_packed.i8",
                copy_device_buffer_bytes(
                    library,
                    w8a16.weight,
                    0,
                    w8a16.weight.bytes,
                    "packed MLA W8 O weight",
                )?,
            )?;
            write_trace_bytes(
                &weight_directory,
                "o_weight_w8_scales.f32",
                copy_device_buffer_bytes(
                    library,
                    w8a16.scales,
                    0,
                    w8a16.scales.bytes,
                    "packed MLA W8 O scales",
                )?,
            )?;
            fs::write(
                &weight_manifest,
                serde_json::to_vec_pretty(&serde_json::json!({
                    "format": "glmrt-packed-mla-trace-weights-v1",
                    "layer": layer,
                    "heads": heads,
                    "rank": GLM52_MLA_KV_LORA_RANK,
                    "nope_dim": nope_dim,
                    "value_dim": v_dim,
                    "hidden_dim": hidden_projection.hidden_dim,
                    "o_weight_packed_layout": w8a16.packed_layout,
                }))
                .context("serializing packed MLA trace weight metadata")?,
            )
            .context("writing packed MLA trace weight metadata")?;
        }
    }
    let weight_file = |name: &str| {
        weight_relative_directory
            .join(name)
            .to_string_lossy()
            .into_owned()
    };

    let descriptor_rows = descriptors
        .iter()
        .map(|descriptor| {
            serde_json::json!({
                "token_start": descriptor.token_start.0,
                "token_count": descriptor.token_count,
            })
        })
        .collect::<Vec<_>>();
    let metadata = serde_json::json!({
        "format": "glmrt-packed-mla-trace-v1",
        "layer": layer,
        "rows": span.rows,
        "prefix_rows": span.rows - query_rows,
        "query_rows": query_rows,
        "heads": heads,
        "rank": GLM52_MLA_KV_LORA_RANK,
        "nope_dim": nope_dim,
        "rope_dim": rope_dim,
        "value_dim": v_dim,
        "hidden_dim": hidden_projection.hidden_dim,
        "kv_dtype": format!("{:?}", span.dtype),
        "kv_row_stride_bytes": span.row_stride_bytes,
        "source_physical_row_offset": span.row_offset,
        "eps": eps,
        "theta": theta,
        "scale": scale,
        "positions": positions,
        "query_positions": query_positions,
        "descriptors": descriptor_rows,
        "o_weight_packed_layout": w8a16.packed_layout,
        "files": {
            "q_projected": "q_projected.bf16",
            "q_nope": "q_nope.bf16",
            "q_rope_rotated": "q_rope_rotated.bf16",
            "packed_kv": "packed_kv.fp8",
            "kv_b_weight": weight_file("kv_b_weight.bf16"),
            "o_weight_w8_packed": weight_file("o_weight_w8_packed.i8"),
            "o_weight_w8_scales": weight_file("o_weight_w8_scales.f32"),
        },
    });
    fs::write(
        trace_directory.join("meta.json"),
        serde_json::to_vec_pretty(&metadata).context("serializing packed MLA trace metadata")?,
    )
    .context("writing packed MLA trace metadata")?;
    eprintln!(
        "real_full_packed_mla_trace directory={} layer={} rows={} query_rows={}",
        trace_directory.display(),
        layer,
        span.rows,
        query_rows
    );
    Ok(())
}

#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) struct RealFullDeviceMlaKvDeviceBufferView {
    rows: usize,
    kv_latent_bytes: usize,
    k_rope_bytes: usize,
    kv_latent: GlmrtDeviceBuffer,
    k_rope: GlmrtDeviceBuffer,
}

#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) struct RealFullDeviceMlaKvProjectedParts {
    status: &'static str,
    rows: usize,
    heads: usize,
    nope_dim: usize,
    v_dim: usize,
    normalized_bytes: usize,
    projected_bytes: usize,
    k_nope_bytes: usize,
    values_bytes: usize,
    normalized: DeviceBufferGuard<'static>,
    projected: DeviceBufferGuard<'static>,
    k_nope: DeviceBufferGuard<'static>,
    values: DeviceBufferGuard<'static>,
}

#[cfg_attr(not(test), allow(dead_code))]
struct RealFullDeviceMlaKvProjectedDeviceBuffers {
    rows: usize,
    heads: usize,
    nope_dim: usize,
    v_dim: usize,
    k_nope_bytes: usize,
    values_bytes: usize,
    k_nope: GlmrtDeviceBuffer,
    values: GlmrtDeviceBuffer,
}

#[cfg_attr(not(test), allow(dead_code))]
struct RealFullDeviceMlaKvRopeDeviceBuffers {
    rows: usize,
    rotary_dim: usize,
    k_rope_rotated_bytes: usize,
    k_rope_rotated: GlmrtDeviceBuffer,
}

#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) struct RealFullDeviceMlaKvRopeParts {
    status: &'static str,
    rows: usize,
    rotary_dim: usize,
    k_rope_rotated_bytes: usize,
    k_rope_rotated: DeviceBufferGuard<'static>,
}

#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) struct RealFullDeviceMlaQueryParts {
    status: &'static str,
    rows: usize,
    prefix_rows: usize,
    suffix_rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    q_nope_bytes: usize,
    q_rope_rotated_bytes: usize,
    q_nope: DeviceBufferGuard<'static>,
    q_rope_rotated: DeviceBufferGuard<'static>,
}

#[cfg_attr(not(test), allow(dead_code))]
struct RealFullDeviceMlaQueryDeviceBuffers {
    q_nope: GlmrtDeviceBuffer,
    q_rope_rotated: GlmrtDeviceBuffer,
}

#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) struct RealFullDeviceMlaAttentionParts {
    status: &'static str,
    rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    output_bytes: usize,
    output: DeviceBufferGuard<'static>,
    hidden_projection_fused: bool,
    ready_event: Option<Arc<CoordinatorCudaEvent>>,
}

pub(in crate::commands::real_full) struct RealFullDeviceKvExecutionMirror {
    cache: Option<RealFullDeviceKvCache<'static>>,
    status: &'static str,
    writes: usize,
    reads: usize,
    bytes: usize,
    scheduler_attention_weights: Option<RealFullSchedulerAttentionResidentWeights>,
    scheduler_attention_queries: BTreeMap<usize, DeviceBufferGuard<'static>>,
    scheduler_attention_resident_uploads: usize,
    scheduler_attention_resident_buffer_uses: usize,
    scheduler_attention_descriptors: Vec<KvBlockDescriptor>,
    scheduler_attention_positions: Vec<u32>,
    scheduler_attention_query_positions: Vec<u32>,
    scheduler_attention_weight_upload_bf16_scratch: Vec<u8>,
    scheduler_attention_projected_query_upload_bf16_scratch: Vec<u8>,
    host_readback_payload_scratch: Vec<u8>,
}

struct RealFullSchedulerAttentionResidentWeights {
    kv_norm_weight: DeviceBufferGuard<'static>,
    kv_b_weight: DeviceBufferGuard<'static>,
    query_projection_weight: Option<DeviceBufferGuard<'static>>,
    output_projection_weight: Option<DeviceBufferGuard<'static>>,
}

fn real_full_device_mla_kv_read_shape(
    config: &KvCacheConfig,
    descriptors: &[KvBlockDescriptor],
) -> Result<Option<RealFullDeviceMlaKvReadShape>> {
    if descriptors.is_empty() {
        return Ok(None);
    }
    if !matches!(
        config.dtype,
        KvCacheDType::Bf16 | KvCacheDType::Fp8 | KvCacheDType::Nvfp4
    ) {
        anyhow::bail!(
            "device MLA KV unpack currently requires BF16, FP8, or NVFP4 cache payloads, got {}",
            config.dtype_label()
        );
    }
    let layer_id = descriptors[0].layer_id;
    for descriptor in descriptors {
        if descriptor.layer_id != layer_id {
            anyhow::bail!(
                "device MLA KV unpack requires same-layer descriptors, got layer {} and {}",
                layer_id.0,
                descriptor.layer_id.0
            );
        }
    }
    let rows = descriptors
        .iter()
        .try_fold(0_usize, |acc, descriptor| {
            acc.checked_add(descriptor.token_count)
        })
        .context("device MLA KV unpack row count overflows usize")?;
    if rows == 0 {
        anyhow::bail!("device MLA KV unpack requires at least one token row");
    }
    let dsa_dim = if config.layer_has_dsa_indexer(layer_id) {
        GLM52_DSA_INDEX_HEAD_DIM
    } else {
        0
    };
    let payload_stride_bytes = config.layer_bytes_per_token(layer_id);
    let expected_stride_bytes = match config.dtype {
        KvCacheDType::Bf16 => {
            (GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM + dsa_dim)
                * std::mem::size_of::<u16>()
        }
        KvCacheDType::Fp8 => {
            GLM52_MLA_FP8_DS_BYTES_PER_TOKEN + dsa_dim * std::mem::size_of::<u16>()
        }
        KvCacheDType::Nvfp4 => {
            GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN + dsa_dim * std::mem::size_of::<u16>()
        }
        _ => unreachable!("validated cache dtype above"),
    };
    if payload_stride_bytes != expected_stride_bytes {
        anyhow::bail!(
            "device MLA KV unpack stride mismatch for layer {}: expected {} got {}",
            layer_id.0,
            expected_stride_bytes,
            payload_stride_bytes
        );
    }
    let payload_bytes = rows
        .checked_mul(payload_stride_bytes)
        .context("device MLA KV unpack payload bytes overflow usize")?;
    let kv_latent_bytes = rows
        .checked_mul(GLM52_MLA_KV_LORA_RANK)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("device MLA KV latent bytes overflow usize")?;
    let k_rope_bytes = rows
        .checked_mul(GLM52_MLA_QK_ROPE_HEAD_DIM)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("device MLA KV rope bytes overflow usize")?;
    let dsa_key_bytes = rows
        .checked_mul(dsa_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("device MLA KV DSA bytes overflow usize")?;
    Ok(Some(RealFullDeviceMlaKvReadShape {
        layer_id,
        dtype: config.dtype,
        rows,
        dsa_dim,
        payload_stride_bytes,
        payload_bytes,
        kv_latent_bytes,
        k_rope_bytes,
        dsa_key_bytes,
    }))
}

#[allow(clippy::too_many_arguments)]
fn unpack_mla_kv_payload_device_buffers_for_shape(
    library: &NativeLibrary,
    shape: &RealFullDeviceMlaKvReadShape,
    payload: GlmrtDeviceBuffer,
    kv_latent: GlmrtDeviceBuffer,
    k_rope: GlmrtDeviceBuffer,
    dsa_key: Option<GlmrtDeviceBuffer>,
    fp8_projected_scratch: &mut DeviceKvReusableDeviceBuffer<'_>,
) -> Result<&'static str> {
    match shape.dtype {
        KvCacheDType::Bf16 => {
            mla_kv_cache_unpack_bf16_device_buffers_for_layer(
                shape.layer_id.0 as usize,
                payload,
                kv_latent,
                k_rope,
                dsa_key,
                shape.rows,
                GLM52_MLA_KV_LORA_RANK,
                GLM52_MLA_QK_ROPE_HEAD_DIM,
                shape.dsa_dim,
                shape.payload_stride_bytes,
            )
            .context("unpacking BF16 device MLA KV compressed payload")?;
            Ok("cuda-kv-cache-mla-kv-unpack-device-buffers")
        }
        KvCacheDType::Fp8 => {
            let projected_stride_bytes = (GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM)
                .checked_mul(std::mem::size_of::<u16>())
                .context("device MLA FP8 unpack projected stride overflow usize")?;
            let projected_bytes = shape
                .rows
                .checked_mul(projected_stride_bytes)
                .context("device MLA FP8 unpack projected bytes overflow usize")?;
            let projected = fp8_projected_scratch
                .buffer(projected_bytes, "device MLA FP8 unpack projected scratch")
                .context("allocating device MLA FP8 unpack projected scratch")?;
            library
                .cuda_mla_kv_unpack_fp8_ds_mla(
                    payload,
                    projected,
                    shape.rows,
                    shape.payload_stride_bytes,
                    projected_stride_bytes,
                )
                .context("unpacking device MLA FP8 DS payload to BF16 projected KV")?;
            mla_kv_cache_unpack_bf16_device_buffers_for_layer(
                shape.layer_id.0 as usize,
                projected,
                kv_latent,
                k_rope,
                None,
                shape.rows,
                GLM52_MLA_KV_LORA_RANK,
                GLM52_MLA_QK_ROPE_HEAD_DIM,
                0,
                projected_stride_bytes,
            )
            .context("splitting unpacked device MLA FP8 DS projected KV")?;
            if shape.dsa_dim > 0 {
                let Some(dsa_key) = dsa_key else {
                    anyhow::bail!("device MLA FP8 unpack missing DSA output buffer");
                };
                let dsa_stride_bytes = shape
                    .dsa_dim
                    .checked_mul(std::mem::size_of::<u16>())
                    .context("device MLA FP8 unpack DSA stride overflow usize")?;
                for row in 0..shape.rows {
                    let src_offset = row
                        .checked_mul(shape.payload_stride_bytes)
                        .and_then(|offset| offset.checked_add(GLM52_MLA_FP8_DS_BYTES_PER_TOKEN))
                        .context("device MLA FP8 unpack DSA source offset overflow usize")?;
                    let dst_offset = row
                        .checked_mul(dsa_stride_bytes)
                        .context("device MLA FP8 unpack DSA destination offset overflow usize")?;
                    library
                        .cuda_kv_cache_write_bytes(
                            device_buffer_byte_view(
                                payload,
                                src_offset,
                                dsa_stride_bytes,
                                "device MLA FP8 packed DSA row",
                            )?,
                            dsa_key,
                            dst_offset,
                            dsa_stride_bytes,
                        )
                        .context("copying device MLA FP8 packed DSA row to unpack output")?;
                }
            }
            Ok("cuda-kv-cache-mla-kv-unpack-fp8-ds-device-buffers")
        }
        KvCacheDType::Nvfp4 => {
            let projected_stride_bytes = (GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM)
                .checked_mul(std::mem::size_of::<u16>())
                .context("device MLA MXFP4 unpack projected stride overflow usize")?;
            let projected_bytes = shape
                .rows
                .checked_mul(projected_stride_bytes)
                .context("device MLA MXFP4 unpack projected bytes overflow usize")?;
            let projected = fp8_projected_scratch
                .buffer(projected_bytes, "device MLA MXFP4 unpack projected scratch")
                .context("allocating device MLA MXFP4 unpack projected scratch")?;
            library
                .cuda_mla_kv_unpack_mxfp4_ds_mla(
                    payload,
                    projected,
                    shape.rows,
                    shape.payload_stride_bytes,
                    projected_stride_bytes,
                )
                .context("unpacking device MLA MXFP4 DS payload to BF16 projected KV")?;
            mla_kv_cache_unpack_bf16_device_buffers_for_layer(
                shape.layer_id.0 as usize,
                projected,
                kv_latent,
                k_rope,
                None,
                shape.rows,
                GLM52_MLA_KV_LORA_RANK,
                GLM52_MLA_QK_ROPE_HEAD_DIM,
                0,
                projected_stride_bytes,
            )
            .context("splitting unpacked device MLA MXFP4 DS projected KV")?;
            if shape.dsa_dim > 0 {
                let Some(dsa_key) = dsa_key else {
                    anyhow::bail!("device MLA MXFP4 unpack missing DSA output buffer");
                };
                let dsa_stride_bytes = shape
                    .dsa_dim
                    .checked_mul(std::mem::size_of::<u16>())
                    .context("device MLA MXFP4 unpack DSA stride overflow usize")?;
                for row in 0..shape.rows {
                    let src_offset = row
                        .checked_mul(shape.payload_stride_bytes)
                        .and_then(|offset| offset.checked_add(GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN))
                        .context("device MLA MXFP4 unpack DSA source offset overflow usize")?;
                    let dst_offset = row
                        .checked_mul(dsa_stride_bytes)
                        .context("device MLA MXFP4 unpack DSA destination offset overflow usize")?;
                    library
                        .cuda_kv_cache_write_bytes(
                            device_buffer_byte_view(
                                payload,
                                src_offset,
                                dsa_stride_bytes,
                                "device MLA MXFP4 packed DSA row",
                            )?,
                            dsa_key,
                            dst_offset,
                            dsa_stride_bytes,
                        )
                        .context("copying device MLA MXFP4 packed DSA row to unpack output")?;
                }
            }
            Ok("cuda-kv-cache-mla-kv-unpack-mxfp4-ds-device-buffers")
        }
        _ => unreachable!("device MLA KV read shape validates supported dtype"),
    }
}

#[allow(dead_code)]
pub(in crate::commands::real_full) struct RealFullDeviceSchedulerAttentionLaunch {
    pub(in crate::commands::real_full) status: &'static str,
    pub(in crate::commands::real_full) descriptors: usize,
    pub(in crate::commands::real_full) rows: usize,
    pub(in crate::commands::real_full) output_rows: usize,
    pub(in crate::commands::real_full) output_row_offset: usize,
    pub(in crate::commands::real_full) query_rows: usize,
    pub(in crate::commands::real_full) heads: usize,
    pub(in crate::commands::real_full) nope_dim: usize,
    pub(in crate::commands::real_full) rope_dim: usize,
    pub(in crate::commands::real_full) v_dim: usize,
    pub(in crate::commands::real_full) output_bytes: usize,
    pub(in crate::commands::real_full) output_values: usize,
    pub(in crate::commands::real_full) output_finite_values: usize,
    pub(in crate::commands::real_full) output_nonzero_values: usize,
    pub(in crate::commands::real_full) output_checksum: f64,
    pub(in crate::commands::real_full) output_bf16: Option<Vec<u8>>,
    pub(in crate::commands::real_full) output_device: DeviceBf16Output,
    pub(in crate::commands::real_full) output_projected_to_hidden: bool,
}

#[cfg_attr(not(test), allow(dead_code))]
impl RealFullDeviceMlaKvDeviceParts {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::commands::real_full) fn from_projected_kv_a_device_bf16(
        library: &'static NativeLibrary,
        projected_kv_a_buffer: GlmrtDeviceBuffer,
        layer_id: LayerId,
        rows: usize,
        kv_lora_rank: usize,
        rope_dim: usize,
    ) -> Result<Self> {
        if rows == 0 || kv_lora_rank == 0 || rope_dim == 0 {
            anyhow::bail!(
                "device MLA current-row KV split requires nonzero shape, got rows={rows} kv_lora_rank={kv_lora_rank} rope_dim={rope_dim}"
            );
        }
        if kv_lora_rank != GLM52_MLA_KV_LORA_RANK {
            anyhow::bail!(
                "device MLA current-row KV split kv_lora_rank mismatch: expected {} got {}",
                GLM52_MLA_KV_LORA_RANK,
                kv_lora_rank
            );
        }
        if rope_dim != GLM52_MLA_QK_ROPE_HEAD_DIM {
            anyhow::bail!(
                "device MLA current-row KV split rope_dim mismatch: expected {} got {}",
                GLM52_MLA_QK_ROPE_HEAD_DIM,
                rope_dim
            );
        }
        let payload_stride_bytes = kv_lora_rank
            .checked_add(rope_dim)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA current-row KV projected row stride overflow")?;
        let payload_bytes = rows
            .checked_mul(payload_stride_bytes)
            .context("device MLA current-row KV projected byte count overflow")?;
        validate_contiguous_payload_buffer(
            "device MLA current-row projected kv_a",
            projected_kv_a_buffer,
            payload_bytes,
        )?;

        let kv_latent_bytes = rows
            .checked_mul(kv_lora_rank)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA current-row KV latent bytes overflow")?;
        let k_rope_bytes = rows
            .checked_mul(rope_dim)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA current-row KV RoPE bytes overflow")?;
        let kv_latent = DeviceBufferGuard::new(library, kv_latent_bytes)
            .context("allocating device MLA current-row KV latent")?;
        let k_rope = DeviceBufferGuard::new(library, k_rope_bytes)
            .context("allocating device MLA current-row KV k_rope")?;
        if projected_kv_a_buffer.device_id != kv_latent.buffer.device_id {
            anyhow::bail!(
                "device MLA current-row projected kv_a buffer is on CUDA device {}, but split outputs are on device {}",
                projected_kv_a_buffer.device_id,
                kv_latent.buffer.device_id
            );
        }
        library
            .cuda_mla_kv_cache_unpack_bf16(
                projected_kv_a_buffer,
                kv_latent.buffer,
                k_rope.buffer,
                None,
                rows,
                kv_lora_rank,
                rope_dim,
                0,
                payload_stride_bytes,
            )
            .context("splitting device MLA current-row projected kv_a")?;

        Ok(Self {
            status: "cuda-kv-cache-mla-current-kv-split-device-buffers",
            layer_id,
            rows,
            payload_bytes,
            payload_stride_bytes,
            kv_latent_bytes,
            k_rope_bytes,
            dsa_key_bytes: 0,
            kv_latent,
            k_rope,
            dsa_key: None,
        })
    }

    pub(in crate::commands::real_full) fn status(&self) -> &'static str {
        self.status
    }

    pub(in crate::commands::real_full) fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    pub(in crate::commands::real_full) fn rows(&self) -> usize {
        self.rows
    }

    pub(in crate::commands::real_full) fn payload_bytes(&self) -> usize {
        self.payload_bytes
    }

    pub(in crate::commands::real_full) fn payload_stride_bytes(&self) -> usize {
        self.payload_stride_bytes
    }

    pub(in crate::commands::real_full) fn kv_latent_buffer(&self) -> GlmrtDeviceBuffer {
        self.kv_latent.buffer
    }

    pub(in crate::commands::real_full) fn k_rope_buffer(&self) -> GlmrtDeviceBuffer {
        self.k_rope.buffer
    }

    pub(in crate::commands::real_full) fn dsa_key_buffer(&self) -> Option<GlmrtDeviceBuffer> {
        self.dsa_key.as_ref().map(|guard| guard.buffer)
    }

    pub(in crate::commands::real_full) fn rotate_k_rope_bf16(
        &self,
        positions: &[u32],
        theta: f32,
    ) -> Result<RealFullDeviceMlaKvRopeParts> {
        if positions.len() != self.rows {
            anyhow::bail!(
                "device MLA KV RoPE positions length mismatch: expected {} got {}",
                self.rows,
                positions.len()
            );
        }
        if !theta.is_finite() || theta <= 0.0 {
            anyhow::bail!("device MLA KV RoPE theta must be finite and positive");
        }
        let library = self.k_rope.library;
        let position_bytes = std::mem::size_of_val(positions);
        let positions_device = DeviceBufferGuard::new(library, position_bytes)
            .context("allocating device MLA KV RoPE positions")?;
        library
            .copy_h2d(positions_device.buffer, u32_slice_bytes(positions))
            .context("copying device MLA KV RoPE positions")?;
        self.rotate_k_rope_bf16_with_position_buffer(positions, positions_device.buffer, theta)
    }

    pub(in crate::commands::real_full) fn rotate_k_rope_bf16_with_position_buffer(
        &self,
        positions: &[u32],
        positions_device: GlmrtDeviceBuffer,
        theta: f32,
    ) -> Result<RealFullDeviceMlaKvRopeParts> {
        if positions.len() != self.rows {
            anyhow::bail!(
                "device MLA KV RoPE positions length mismatch: expected {} got {}",
                self.rows,
                positions.len()
            );
        }
        if !theta.is_finite() || theta <= 0.0 {
            anyhow::bail!("device MLA KV RoPE theta must be finite and positive");
        }
        validate_contiguous_payload_buffer(
            "device MLA KV RoPE positions",
            positions_device,
            std::mem::size_of_val(positions),
        )?;
        if positions_device.device_id != self.k_rope.buffer.device_id {
            anyhow::bail!(
                "device MLA KV RoPE positions buffer is on CUDA device {}, but k_rope is on device {}",
                positions_device.device_id,
                self.k_rope.buffer.device_id
            );
        }
        let library = self.k_rope.library;
        let k_rope_rotated = DeviceBufferGuard::new(library, self.k_rope_bytes)
            .context("allocating device MLA KV rotated k_rope output")?;
        library
            .cuda_rope_bf16(
                self.k_rope.buffer,
                positions_device,
                k_rope_rotated.buffer,
                self.rows,
                1,
                GLM52_MLA_QK_ROPE_HEAD_DIM,
                theta,
            )
            .context("executing device MLA KV k_rope RoPE")?;
        Ok(RealFullDeviceMlaKvRopeParts {
            status: "cuda-kv-cache-mla-kv-k-rope-device-buffer",
            rows: self.rows,
            rotary_dim: GLM52_MLA_QK_ROPE_HEAD_DIM,
            k_rope_rotated_bytes: self.k_rope_bytes,
            k_rope_rotated,
        })
    }

    pub(in crate::commands::real_full) fn project_kv_latent_and_split_bf16(
        &self,
        kv_norm_weight: GlmrtDeviceBuffer,
        kv_b_weight: GlmrtDeviceBuffer,
        heads: usize,
        nope_dim: usize,
        v_dim: usize,
        eps: f32,
    ) -> Result<RealFullDeviceMlaKvProjectedParts> {
        if heads == 0 {
            anyhow::bail!("device MLA KV projected split requires at least one head");
        }
        if nope_dim == 0 {
            anyhow::bail!("device MLA KV projected split requires nonzero nope_dim");
        }
        if v_dim == 0 {
            anyhow::bail!("device MLA KV projected split requires nonzero v_dim");
        }
        if !eps.is_finite() {
            anyhow::bail!("device MLA KV projected split RMSNorm eps must be finite");
        }
        let library = self.kv_latent.library;
        let normalized_bytes = self.kv_latent_bytes;
        let projected_width = heads
            .checked_mul(
                nope_dim
                    .checked_add(v_dim)
                    .context("device MLA KV projected split head width overflow")?,
            )
            .context("device MLA KV projected width overflow")?;
        let projected_bytes = self
            .rows
            .checked_mul(projected_width)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA KV projected bytes overflow usize")?;
        let k_nope_bytes = self
            .rows
            .checked_mul(heads)
            .and_then(|values| values.checked_mul(nope_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA KV k_nope bytes overflow usize")?;
        let values_bytes = self
            .rows
            .checked_mul(heads)
            .and_then(|values| values.checked_mul(v_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA KV value bytes overflow usize")?;
        validate_contiguous_payload_buffer(
            "device MLA KV RMSNorm weight",
            kv_norm_weight,
            GLM52_MLA_KV_LORA_RANK * std::mem::size_of::<u16>(),
        )?;
        validate_contiguous_payload_buffer(
            "device MLA KV kv_b weight",
            kv_b_weight,
            projected_width
                .checked_mul(GLM52_MLA_KV_LORA_RANK)
                .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
                .context("device MLA KV kv_b weight bytes overflow usize")?,
        )?;
        let normalized = DeviceBufferGuard::new(library, normalized_bytes)
            .context("allocating device MLA KV normalized latent output")?;
        let projected = DeviceBufferGuard::new(library, projected_bytes)
            .context("allocating device MLA KV projected output")?;
        let k_nope = DeviceBufferGuard::new(library, k_nope_bytes)
            .context("allocating device MLA KV k_nope output")?;
        let values = DeviceBufferGuard::new(library, values_bytes)
            .context("allocating device MLA KV value output")?;
        library
            .cuda_rmsnorm_bf16(
                self.kv_latent.buffer,
                kv_norm_weight,
                normalized.buffer,
                usize_to_i32("device MLA KV RMSNorm rows", self.rows)?,
                usize_to_i32("device MLA KV RMSNorm hidden", GLM52_MLA_KV_LORA_RANK)?,
                eps,
            )
            .context("executing device MLA KV latent RMSNorm")?;
        library
            .cuda_linear_bf16(
                normalized.buffer,
                kv_b_weight,
                None,
                projected.buffer,
                self.rows,
                GLM52_MLA_KV_LORA_RANK,
                projected_width,
            )
            .context("executing device MLA KV kv_b projection")?;
        library
            .cuda_mla_kv_projected_split_bf16(
                projected.buffer,
                k_nope.buffer,
                values.buffer,
                self.rows,
                heads,
                nope_dim,
                v_dim,
            )
            .context("splitting device MLA KV projected buffer")?;
        Ok(RealFullDeviceMlaKvProjectedParts {
            status: "cuda-kv-cache-mla-kv-norm-linear-split-device-buffers",
            rows: self.rows,
            heads,
            nope_dim,
            v_dim,
            normalized_bytes,
            projected_bytes,
            k_nope_bytes,
            values_bytes,
            normalized,
            projected,
            k_nope,
            values,
        })
    }

    fn copy_to_host(&self) -> Result<RealFullDeviceMlaKvUnpackReadback> {
        let mut kv_latent_bf16 = vec![0_u8; self.kv_latent_bytes];
        let mut k_rope_bf16 = vec![0_u8; self.k_rope_bytes];
        self.kv_latent
            .library
            .copy_d2h(&mut kv_latent_bf16, self.kv_latent.buffer)?;
        self.k_rope
            .library
            .copy_d2h(&mut k_rope_bf16, self.k_rope.buffer)?;
        let dsa_key_bf16 = match &self.dsa_key {
            Some(guard) => {
                let mut bytes = vec![0_u8; self.dsa_key_bytes];
                guard.library.copy_d2h(&mut bytes, guard.buffer)?;
                Some(bytes)
            }
            None => None,
        };
        Ok(RealFullDeviceMlaKvUnpackReadback {
            status: "cuda-kv-cache-mla-kv-unpack-readback",
            rows: self.rows,
            payload_bytes: self.payload_bytes,
            kv_latent_bf16,
            k_rope_bf16,
            dsa_key_bf16,
        })
    }
}

#[cfg_attr(not(test), allow(dead_code))]
impl RealFullDeviceMlaKvProjectedParts {
    pub(in crate::commands::real_full) fn status(&self) -> &'static str {
        self.status
    }

    pub(in crate::commands::real_full) fn rows(&self) -> usize {
        self.rows
    }

    pub(in crate::commands::real_full) fn heads(&self) -> usize {
        self.heads
    }

    pub(in crate::commands::real_full) fn nope_dim(&self) -> usize {
        self.nope_dim
    }

    pub(in crate::commands::real_full) fn v_dim(&self) -> usize {
        self.v_dim
    }

    pub(in crate::commands::real_full) fn normalized_buffer(&self) -> GlmrtDeviceBuffer {
        self.normalized.buffer
    }

    pub(in crate::commands::real_full) fn projected_buffer(&self) -> GlmrtDeviceBuffer {
        self.projected.buffer
    }

    pub(in crate::commands::real_full) fn k_nope_buffer(&self) -> GlmrtDeviceBuffer {
        self.k_nope.buffer
    }

    pub(in crate::commands::real_full) fn values_buffer(&self) -> GlmrtDeviceBuffer {
        self.values.buffer
    }

    pub(in crate::commands::real_full) fn run_mla_rope_attention_bf16(
        &self,
        rotated_k_rope: &RealFullDeviceMlaKvRopeParts,
        q_nope: GlmrtDeviceBuffer,
        q_rope: GlmrtDeviceBuffer,
        scale: f32,
    ) -> Result<RealFullDeviceMlaAttentionParts> {
        if rotated_k_rope.rows != self.rows {
            anyhow::bail!(
                "device MLA attention row mismatch: projected rows={} rotated k_rope rows={}",
                self.rows,
                rotated_k_rope.rows
            );
        }
        if rotated_k_rope.rotary_dim != GLM52_MLA_QK_ROPE_HEAD_DIM {
            anyhow::bail!(
                "device MLA attention rotary dim mismatch: expected {} got {}",
                GLM52_MLA_QK_ROPE_HEAD_DIM,
                rotated_k_rope.rotary_dim
            );
        }
        if !scale.is_finite() {
            anyhow::bail!("device MLA attention scale must be finite");
        }
        if q_nope.device_id != self.k_nope.buffer.device_id
            || q_rope.device_id != self.k_nope.buffer.device_id
            || rotated_k_rope.k_rope_rotated.buffer.device_id != self.k_nope.buffer.device_id
            || self.values.buffer.device_id != self.k_nope.buffer.device_id
        {
            anyhow::bail!("device MLA attention buffers must be on the same CUDA device");
        }
        let q_nope_bytes = self
            .rows
            .checked_mul(self.heads)
            .and_then(|values| values.checked_mul(self.nope_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA attention q_nope bytes overflow usize")?;
        let q_rope_bytes = self
            .rows
            .checked_mul(self.heads)
            .and_then(|values| values.checked_mul(rotated_k_rope.rotary_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA attention q_rope bytes overflow usize")?;
        let output_bytes = self
            .rows
            .checked_mul(self.heads)
            .and_then(|values| values.checked_mul(self.v_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA attention output bytes overflow usize")?;
        validate_contiguous_payload_buffer("device MLA attention q_nope", q_nope, q_nope_bytes)?;
        validate_contiguous_payload_buffer("device MLA attention q_rope", q_rope, q_rope_bytes)?;
        validate_contiguous_payload_buffer(
            "device MLA attention k_nope",
            self.k_nope.buffer,
            self.k_nope_bytes,
        )?;
        validate_contiguous_payload_buffer(
            "device MLA attention k_rope",
            rotated_k_rope.k_rope_rotated.buffer,
            rotated_k_rope.k_rope_rotated_bytes,
        )?;
        validate_contiguous_payload_buffer(
            "device MLA attention values",
            self.values.buffer,
            self.values_bytes,
        )?;
        let library = self.k_nope.library;
        let output = DeviceBufferGuard::new(library, output_bytes)
            .context("allocating device MLA attention output")?;
        library
            .cuda_mla_rope_attention_bf16(
                q_nope,
                q_rope,
                self.k_nope.buffer,
                rotated_k_rope.k_rope_rotated.buffer,
                self.values.buffer,
                output.buffer,
                self.rows,
                self.heads,
                self.nope_dim,
                rotated_k_rope.rotary_dim,
                self.v_dim,
                scale,
            )
            .context("executing device MLA RoPE attention")?;
        Ok(RealFullDeviceMlaAttentionParts {
            status: "cuda-kv-cache-mla-rope-attention-device-buffer",
            rows: self.rows,
            heads: self.heads,
            nope_dim: self.nope_dim,
            rope_dim: rotated_k_rope.rotary_dim,
            v_dim: self.v_dim,
            output_bytes,
            output,
            hidden_projection_fused: false,
            ready_event: None,
        })
    }

    pub(in crate::commands::real_full) fn run_mla_rope_attention_with_host_suffix_bf16(
        &self,
        rotated_k_rope: &RealFullDeviceMlaKvRopeParts,
        q_nope: GlmrtDeviceBuffer,
        q_rope: GlmrtDeviceBuffer,
        suffix_k_nope_bf16: &[u8],
        suffix_k_rope_rotated_bf16: &[u8],
        suffix_values_bf16: &[u8],
        scale: f32,
    ) -> Result<RealFullDeviceMlaAttentionParts> {
        if rotated_k_rope.rows != self.rows {
            anyhow::bail!(
                "device MLA attention suffix row mismatch: projected rows={} rotated k_rope rows={}",
                self.rows,
                rotated_k_rope.rows
            );
        }
        if rotated_k_rope.rotary_dim != GLM52_MLA_QK_ROPE_HEAD_DIM {
            anyhow::bail!(
                "device MLA attention suffix rotary dim mismatch: expected {} got {}",
                GLM52_MLA_QK_ROPE_HEAD_DIM,
                rotated_k_rope.rotary_dim
            );
        }
        if !scale.is_finite() {
            anyhow::bail!("device MLA attention suffix scale must be finite");
        }
        let value_size = std::mem::size_of::<u16>();
        let suffix_k_rope_row_bytes = rotated_k_rope
            .rotary_dim
            .checked_mul(value_size)
            .context("device MLA attention suffix k_rope row bytes overflow usize")?;
        if suffix_k_rope_rotated_bf16.is_empty()
            || suffix_k_rope_rotated_bf16.len() % suffix_k_rope_row_bytes != 0
        {
            anyhow::bail!(
                "device MLA attention suffix rotated k_rope bytes {} are not a non-empty multiple of row bytes {}",
                suffix_k_rope_rotated_bf16.len(),
                suffix_k_rope_row_bytes
            );
        }
        let suffix_rows = suffix_k_rope_rotated_bf16.len() / suffix_k_rope_row_bytes;
        let suffix_k_nope_bytes = suffix_rows
            .checked_mul(self.heads)
            .and_then(|values| values.checked_mul(self.nope_dim))
            .and_then(|values| values.checked_mul(value_size))
            .context("device MLA attention suffix k_nope bytes overflow usize")?;
        let suffix_values_bytes = suffix_rows
            .checked_mul(self.heads)
            .and_then(|values| values.checked_mul(self.v_dim))
            .and_then(|values| values.checked_mul(value_size))
            .context("device MLA attention suffix values bytes overflow usize")?;
        if suffix_k_nope_bf16.len() != suffix_k_nope_bytes {
            anyhow::bail!(
                "device MLA attention suffix k_nope byte mismatch: expected {} got {}",
                suffix_k_nope_bytes,
                suffix_k_nope_bf16.len()
            );
        }
        if suffix_values_bf16.len() != suffix_values_bytes {
            anyhow::bail!(
                "device MLA attention suffix value byte mismatch: expected {} got {}",
                suffix_values_bytes,
                suffix_values_bf16.len()
            );
        }
        let total_rows = self
            .rows
            .checked_add(suffix_rows)
            .context("device MLA attention total rows overflow usize")?;
        let q_nope_bytes = total_rows
            .checked_mul(self.heads)
            .and_then(|values| values.checked_mul(self.nope_dim))
            .and_then(|values| values.checked_mul(value_size))
            .context("device MLA attention suffix q_nope bytes overflow usize")?;
        let q_rope_bytes = total_rows
            .checked_mul(self.heads)
            .and_then(|values| values.checked_mul(rotated_k_rope.rotary_dim))
            .and_then(|values| values.checked_mul(value_size))
            .context("device MLA attention suffix q_rope bytes overflow usize")?;
        let output_bytes = total_rows
            .checked_mul(self.heads)
            .and_then(|values| values.checked_mul(self.v_dim))
            .and_then(|values| values.checked_mul(value_size))
            .context("device MLA attention suffix output bytes overflow usize")?;
        validate_contiguous_payload_buffer(
            "device MLA attention suffix q_nope",
            q_nope,
            q_nope_bytes,
        )?;
        validate_contiguous_payload_buffer(
            "device MLA attention suffix q_rope",
            q_rope,
            q_rope_bytes,
        )?;
        if q_nope.device_id != self.k_nope.buffer.device_id
            || q_rope.device_id != self.k_nope.buffer.device_id
            || rotated_k_rope.k_rope_rotated.buffer.device_id != self.k_nope.buffer.device_id
            || self.values.buffer.device_id != self.k_nope.buffer.device_id
        {
            anyhow::bail!("device MLA attention suffix buffers must be on the same CUDA device");
        }

        let library = self.k_nope.library;
        let combined_k_nope_bytes = self
            .k_nope_bytes
            .checked_add(suffix_k_nope_bytes)
            .context("device MLA attention combined k_nope bytes overflow usize")?;
        let combined_k_rope_bytes = rotated_k_rope
            .k_rope_rotated_bytes
            .checked_add(suffix_k_rope_rotated_bf16.len())
            .context("device MLA attention combined k_rope bytes overflow usize")?;
        let combined_values_bytes = self
            .values_bytes
            .checked_add(suffix_values_bytes)
            .context("device MLA attention combined value bytes overflow usize")?;
        let combined_k_nope = DeviceBufferGuard::new(library, combined_k_nope_bytes)
            .context("allocating device MLA attention combined k_nope")?;
        let combined_k_rope = DeviceBufferGuard::new(library, combined_k_rope_bytes)
            .context("allocating device MLA attention combined k_rope")?;
        let combined_values = DeviceBufferGuard::new(library, combined_values_bytes)
            .context("allocating device MLA attention combined values")?;
        let suffix_k_nope = DeviceBufferGuard::new(library, suffix_k_nope_bytes)
            .context("allocating device MLA attention suffix k_nope")?;
        let suffix_k_rope = DeviceBufferGuard::new(library, suffix_k_rope_rotated_bf16.len())
            .context("allocating device MLA attention suffix k_rope")?;
        let suffix_values = DeviceBufferGuard::new(library, suffix_values_bytes)
            .context("allocating device MLA attention suffix values")?;
        library
            .copy_h2d(suffix_k_nope.buffer, suffix_k_nope_bf16)
            .context("copying device MLA attention suffix k_nope")?;
        library
            .copy_h2d(suffix_k_rope.buffer, suffix_k_rope_rotated_bf16)
            .context("copying device MLA attention suffix k_rope")?;
        library
            .copy_h2d(suffix_values.buffer, suffix_values_bf16)
            .context("copying device MLA attention suffix values")?;
        library
            .cuda_kv_cache_write_bytes(
                self.k_nope.buffer,
                combined_k_nope.buffer,
                0,
                self.k_nope_bytes,
            )
            .context("copying device MLA attention prefix k_nope")?;
        library
            .cuda_kv_cache_write_bytes(
                suffix_k_nope.buffer,
                combined_k_nope.buffer,
                self.k_nope_bytes,
                suffix_k_nope_bytes,
            )
            .context("copying device MLA attention suffix k_nope into combined buffer")?;
        library
            .cuda_kv_cache_write_bytes(
                rotated_k_rope.k_rope_rotated.buffer,
                combined_k_rope.buffer,
                0,
                rotated_k_rope.k_rope_rotated_bytes,
            )
            .context("copying device MLA attention prefix k_rope")?;
        library
            .cuda_kv_cache_write_bytes(
                suffix_k_rope.buffer,
                combined_k_rope.buffer,
                rotated_k_rope.k_rope_rotated_bytes,
                suffix_k_rope_rotated_bf16.len(),
            )
            .context("copying device MLA attention suffix k_rope into combined buffer")?;
        library
            .cuda_kv_cache_write_bytes(
                self.values.buffer,
                combined_values.buffer,
                0,
                self.values_bytes,
            )
            .context("copying device MLA attention prefix values")?;
        library
            .cuda_kv_cache_write_bytes(
                suffix_values.buffer,
                combined_values.buffer,
                self.values_bytes,
                suffix_values_bytes,
            )
            .context("copying device MLA attention suffix values into combined buffer")?;

        let output = DeviceBufferGuard::new(library, output_bytes)
            .context("allocating device MLA attention suffix output")?;
        library
            .cuda_mla_rope_attention_bf16(
                q_nope,
                q_rope,
                combined_k_nope.buffer,
                combined_k_rope.buffer,
                combined_values.buffer,
                output.buffer,
                total_rows,
                self.heads,
                self.nope_dim,
                rotated_k_rope.rotary_dim,
                self.v_dim,
                scale,
            )
            .context("executing device MLA RoPE attention with host suffix")?;
        Ok(RealFullDeviceMlaAttentionParts {
            status: "cuda-kv-cache-mla-rope-attention-device-buffer-with-host-suffix",
            rows: total_rows,
            heads: self.heads,
            nope_dim: self.nope_dim,
            rope_dim: rotated_k_rope.rotary_dim,
            v_dim: self.v_dim,
            output_bytes,
            output,
            hidden_projection_fused: false,
            ready_event: None,
        })
    }

    fn copy_to_host(&self) -> Result<RealFullDeviceMlaKvProjectedReadback> {
        let mut normalized_bf16 = vec![0_u8; self.normalized_bytes];
        let mut projected_bf16 = vec![0_u8; self.projected_bytes];
        let mut k_nope_bf16 = vec![0_u8; self.k_nope_bytes];
        let mut values_bf16 = vec![0_u8; self.values_bytes];
        self.normalized
            .library
            .copy_d2h(&mut normalized_bf16, self.normalized.buffer)?;
        self.projected
            .library
            .copy_d2h(&mut projected_bf16, self.projected.buffer)?;
        self.k_nope
            .library
            .copy_d2h(&mut k_nope_bf16, self.k_nope.buffer)?;
        self.values
            .library
            .copy_d2h(&mut values_bf16, self.values.buffer)?;
        Ok(RealFullDeviceMlaKvProjectedReadback {
            status: "cuda-kv-cache-mla-kv-norm-linear-split-readback",
            rows: self.rows,
            heads: self.heads,
            nope_dim: self.nope_dim,
            v_dim: self.v_dim,
            normalized_bf16,
            projected_bf16,
            k_nope_bf16,
            values_bf16,
        })
    }
}

#[cfg_attr(not(test), allow(dead_code))]
impl RealFullDeviceMlaKvRopeParts {
    pub(in crate::commands::real_full) fn status(&self) -> &'static str {
        self.status
    }

    pub(in crate::commands::real_full) fn rows(&self) -> usize {
        self.rows
    }

    pub(in crate::commands::real_full) fn rotary_dim(&self) -> usize {
        self.rotary_dim
    }

    pub(in crate::commands::real_full) fn k_rope_rotated_buffer(&self) -> GlmrtDeviceBuffer {
        self.k_rope_rotated.buffer
    }

    fn copy_to_host(&self) -> Result<RealFullDeviceMlaKvRopeReadback> {
        let mut k_rope_rotated_bf16 = vec![0_u8; self.k_rope_rotated_bytes];
        self.k_rope_rotated
            .library
            .copy_d2h(&mut k_rope_rotated_bf16, self.k_rope_rotated.buffer)?;
        Ok(RealFullDeviceMlaKvRopeReadback {
            status: "cuda-kv-cache-mla-kv-k-rope-readback",
            rows: self.rows,
            rotary_dim: self.rotary_dim,
            k_rope_rotated_bf16,
        })
    }
}

#[cfg_attr(not(test), allow(dead_code))]
impl RealFullDeviceMlaQueryParts {
    #[allow(clippy::too_many_arguments)]
    fn from_projected_suffix_bf16(
        library: &'static NativeLibrary,
        projected_query_bf16: &[u8],
        prefix_rows: usize,
        suffix_positions: &[u32],
        heads: usize,
        nope_dim: usize,
        rope_dim: usize,
        theta: f32,
    ) -> Result<Self> {
        if suffix_positions.is_empty() {
            anyhow::bail!("device MLA query projection requires at least one suffix row");
        }
        if heads == 0 || nope_dim == 0 || rope_dim == 0 {
            anyhow::bail!(
                "device MLA query projection requires nonzero shape, got heads={heads} nope_dim={nope_dim} rope_dim={rope_dim}"
            );
        }
        if !theta.is_finite() || theta <= 0.0 {
            anyhow::bail!("device MLA query projection RoPE theta must be finite and positive");
        }
        let suffix_rows = suffix_positions.len();
        let rows = prefix_rows
            .checked_add(suffix_rows)
            .context("device MLA query projection total row count overflow")?;
        let head_width = nope_dim
            .checked_add(rope_dim)
            .context("device MLA query projection head width overflow")?;
        let projected_bytes = suffix_rows
            .checked_mul(heads)
            .and_then(|values| values.checked_mul(head_width))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA query projection q_b byte count overflow")?;
        if projected_query_bf16.len() != projected_bytes {
            anyhow::bail!(
                "device MLA query projection q_b bytes mismatch: expected {} got {}",
                projected_bytes,
                projected_query_bf16.len()
            );
        }
        let suffix_q_nope_bytes = suffix_rows
            .checked_mul(heads)
            .and_then(|values| values.checked_mul(nope_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA query projection suffix q_nope bytes overflow")?;
        let suffix_q_rope_bytes = suffix_rows
            .checked_mul(heads)
            .and_then(|values| values.checked_mul(rope_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA query projection suffix q_rope bytes overflow")?;
        let q_nope_bytes = rows
            .checked_mul(heads)
            .and_then(|values| values.checked_mul(nope_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA query projection full q_nope bytes overflow")?;
        let q_rope_rotated_bytes = rows
            .checked_mul(heads)
            .and_then(|values| values.checked_mul(rope_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA query projection full q_rope bytes overflow")?;
        let prefix_q_nope_bytes = prefix_rows
            .checked_mul(heads)
            .and_then(|values| values.checked_mul(nope_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA query projection prefix q_nope bytes overflow")?;
        let prefix_q_rope_bytes = prefix_rows
            .checked_mul(heads)
            .and_then(|values| values.checked_mul(rope_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA query projection prefix q_rope bytes overflow")?;

        let projected = DeviceBufferGuard::new(library, projected_bytes)
            .context("allocating device MLA query projected q_b")?;
        let q_rope_unrotated = DeviceBufferGuard::new(library, suffix_q_rope_bytes)
            .context("allocating device MLA query unrotated q_rope suffix")?;
        let q_nope = DeviceBufferGuard::new(library, q_nope_bytes)
            .context("allocating device MLA query full q_nope")?;
        let q_rope_rotated = DeviceBufferGuard::new(library, q_rope_rotated_bytes)
            .context("allocating device MLA query full rotated q_rope")?;
        let positions = DeviceBufferGuard::new(library, std::mem::size_of_val(suffix_positions))
            .context("allocating device MLA query RoPE positions")?;

        library
            .copy_h2d(projected.buffer, projected_query_bf16)
            .context("copying device MLA query projected q_b")?;
        library
            .copy_h2d(positions.buffer, u32_slice_bytes(suffix_positions))
            .context("copying device MLA query RoPE positions")?;
        zero_device_buffer_bytes(
            library,
            q_nope.buffer,
            q_nope_bytes,
            "device MLA query q_nope",
        )?;
        zero_device_buffer_bytes(
            library,
            q_rope_rotated.buffer,
            q_rope_rotated_bytes,
            "device MLA query rotated q_rope",
        )?;

        let q_nope_suffix = device_buffer_byte_view(
            q_nope.buffer,
            prefix_q_nope_bytes,
            suffix_q_nope_bytes,
            "device MLA query q_nope suffix",
        )?;
        let q_rope_rotated_suffix = device_buffer_byte_view(
            q_rope_rotated.buffer,
            prefix_q_rope_bytes,
            suffix_q_rope_bytes,
            "device MLA query rotated q_rope suffix",
        )?;
        library
            .cuda_mla_kv_projected_split_bf16(
                projected.buffer,
                q_nope_suffix,
                q_rope_unrotated.buffer,
                suffix_rows,
                heads,
                nope_dim,
                rope_dim,
            )
            .context("splitting device MLA query projected q_b")?;
        library
            .cuda_rope_bf16(
                q_rope_unrotated.buffer,
                positions.buffer,
                q_rope_rotated_suffix,
                suffix_rows,
                heads,
                rope_dim,
                theta,
            )
            .context("rotating device MLA query q_rope suffix")?;

        Ok(Self {
            status: "cuda-kv-cache-mla-query-split-rope-device-buffers",
            rows,
            prefix_rows,
            suffix_rows,
            heads,
            nope_dim,
            rope_dim,
            q_nope_bytes,
            q_rope_rotated_bytes,
            q_nope,
            q_rope_rotated,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::commands::real_full) fn from_projected_suffix_device_bf16(
        library: &'static NativeLibrary,
        projected_query_buffer: GlmrtDeviceBuffer,
        prefix_rows: usize,
        suffix_positions: &[u32],
        heads: usize,
        nope_dim: usize,
        rope_dim: usize,
        theta: f32,
    ) -> Result<Self> {
        if suffix_positions.is_empty() {
            anyhow::bail!("device MLA query projection requires at least one suffix row");
        }
        let positions = DeviceBufferGuard::new(library, std::mem::size_of_val(suffix_positions))
            .context("allocating device MLA query RoPE positions")?;
        library
            .copy_h2d(positions.buffer, u32_slice_bytes(suffix_positions))
            .context("copying device MLA query RoPE positions")?;
        Self::from_projected_suffix_device_bf16_with_position_buffer(
            library,
            projected_query_buffer,
            prefix_rows,
            suffix_positions,
            positions.buffer,
            heads,
            nope_dim,
            rope_dim,
            theta,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::commands::real_full) fn from_projected_suffix_device_bf16_with_position_buffer(
        library: &'static NativeLibrary,
        projected_query_buffer: GlmrtDeviceBuffer,
        prefix_rows: usize,
        suffix_positions: &[u32],
        positions_device: GlmrtDeviceBuffer,
        heads: usize,
        nope_dim: usize,
        rope_dim: usize,
        theta: f32,
    ) -> Result<Self> {
        if suffix_positions.is_empty() {
            anyhow::bail!("device MLA query projection requires at least one suffix row");
        }
        if heads == 0 || nope_dim == 0 || rope_dim == 0 {
            anyhow::bail!(
                "device MLA query projection requires nonzero shape, got heads={heads} nope_dim={nope_dim} rope_dim={rope_dim}"
            );
        }
        if !theta.is_finite() || theta <= 0.0 {
            anyhow::bail!("device MLA query projection RoPE theta must be finite and positive");
        }
        let suffix_rows = suffix_positions.len();
        let rows = prefix_rows
            .checked_add(suffix_rows)
            .context("device MLA query projection total row count overflow")?;
        let head_width = nope_dim
            .checked_add(rope_dim)
            .context("device MLA query projection head width overflow")?;
        let projected_bytes = suffix_rows
            .checked_mul(heads)
            .and_then(|values| values.checked_mul(head_width))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA query projection q_b byte count overflow")?;
        validate_contiguous_payload_buffer(
            "device MLA query projected q_b device buffer",
            projected_query_buffer,
            projected_bytes,
        )?;
        validate_contiguous_payload_buffer(
            "device MLA query RoPE positions",
            positions_device,
            std::mem::size_of_val(suffix_positions),
        )?;
        let suffix_q_nope_bytes = suffix_rows
            .checked_mul(heads)
            .and_then(|values| values.checked_mul(nope_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA query projection suffix q_nope bytes overflow")?;
        let suffix_q_rope_bytes = suffix_rows
            .checked_mul(heads)
            .and_then(|values| values.checked_mul(rope_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA query projection suffix q_rope bytes overflow")?;
        let q_nope_bytes = rows
            .checked_mul(heads)
            .and_then(|values| values.checked_mul(nope_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA query projection full q_nope bytes overflow")?;
        let q_rope_rotated_bytes = rows
            .checked_mul(heads)
            .and_then(|values| values.checked_mul(rope_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA query projection full q_rope bytes overflow")?;
        let prefix_q_nope_bytes = prefix_rows
            .checked_mul(heads)
            .and_then(|values| values.checked_mul(nope_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA query projection prefix q_nope bytes overflow")?;
        let prefix_q_rope_bytes = prefix_rows
            .checked_mul(heads)
            .and_then(|values| values.checked_mul(rope_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA query projection prefix q_rope bytes overflow")?;

        let q_rope_unrotated = DeviceBufferGuard::new(library, suffix_q_rope_bytes)
            .context("allocating device MLA query unrotated q_rope suffix")?;
        let q_nope = DeviceBufferGuard::new(library, q_nope_bytes)
            .context("allocating device MLA query full q_nope")?;
        let q_rope_rotated = DeviceBufferGuard::new(library, q_rope_rotated_bytes)
            .context("allocating device MLA query full rotated q_rope")?;
        if projected_query_buffer.device_id != q_nope.buffer.device_id {
            anyhow::bail!(
                "device MLA query projected q_b buffer is on CUDA device {}, but query outputs are on device {}",
                projected_query_buffer.device_id,
                q_nope.buffer.device_id
            );
        }
        if positions_device.device_id != q_nope.buffer.device_id {
            anyhow::bail!(
                "device MLA query RoPE positions buffer is on CUDA device {}, but query outputs are on device {}",
                positions_device.device_id,
                q_nope.buffer.device_id
            );
        }
        zero_device_buffer_bytes(
            library,
            q_nope.buffer,
            q_nope_bytes,
            "device MLA query q_nope",
        )?;
        zero_device_buffer_bytes(
            library,
            q_rope_rotated.buffer,
            q_rope_rotated_bytes,
            "device MLA query rotated q_rope",
        )?;

        let q_nope_suffix = device_buffer_byte_view(
            q_nope.buffer,
            prefix_q_nope_bytes,
            suffix_q_nope_bytes,
            "device MLA query q_nope suffix",
        )?;
        let q_rope_rotated_suffix = device_buffer_byte_view(
            q_rope_rotated.buffer,
            prefix_q_rope_bytes,
            suffix_q_rope_bytes,
            "device MLA query rotated q_rope suffix",
        )?;
        library
            .cuda_mla_kv_projected_split_bf16(
                projected_query_buffer,
                q_nope_suffix,
                q_rope_unrotated.buffer,
                suffix_rows,
                heads,
                nope_dim,
                rope_dim,
            )
            .context("splitting device MLA query projected q_b")?;
        library
            .cuda_rope_bf16(
                q_rope_unrotated.buffer,
                positions_device,
                q_rope_rotated_suffix,
                suffix_rows,
                heads,
                rope_dim,
                theta,
            )
            .context("rotating device MLA query q_rope suffix")?;

        Ok(Self {
            status: "cuda-kv-cache-mla-query-split-rope-device-buffers",
            rows,
            prefix_rows,
            suffix_rows,
            heads,
            nope_dim,
            rope_dim,
            q_nope_bytes,
            q_rope_rotated_bytes,
            q_nope,
            q_rope_rotated,
        })
    }

    pub(in crate::commands::real_full) fn status(&self) -> &'static str {
        self.status
    }

    fn copy_to_host(&self) -> Result<RealFullDeviceMlaQueryReadback> {
        let mut q_nope_bf16 = vec![0_u8; self.q_nope_bytes];
        let mut q_rope_rotated_bf16 = vec![0_u8; self.q_rope_rotated_bytes];
        self.q_nope
            .library
            .copy_d2h(&mut q_nope_bf16, self.q_nope.buffer)?;
        self.q_rope_rotated
            .library
            .copy_d2h(&mut q_rope_rotated_bf16, self.q_rope_rotated.buffer)?;
        Ok(RealFullDeviceMlaQueryReadback {
            status: "cuda-kv-cache-mla-query-readback",
            rows: self.rows,
            prefix_rows: self.prefix_rows,
            suffix_rows: self.suffix_rows,
            heads: self.heads,
            nope_dim: self.nope_dim,
            rope_dim: self.rope_dim,
            q_nope_bf16,
            q_rope_rotated_bf16,
        })
    }
}

#[cfg_attr(not(test), allow(dead_code))]
impl RealFullDeviceMlaKvProjectedDeviceBuffers {
    fn run_mla_rope_attention_bf16(
        &self,
        library: &'static NativeLibrary,
        layer_id: LayerId,
        rotated_k_rope: &RealFullDeviceMlaKvRopeDeviceBuffers,
        q_nope: GlmrtDeviceBuffer,
        q_rope: GlmrtDeviceBuffer,
        output_buffer: GlmrtDeviceBuffer,
        scale: f32,
    ) -> Result<RealFullDeviceMlaAttentionParts> {
        if rotated_k_rope.rows != self.rows {
            anyhow::bail!(
                "device MLA attention row mismatch: projected rows={} rotated k_rope rows={}",
                self.rows,
                rotated_k_rope.rows
            );
        }
        if rotated_k_rope.rotary_dim != GLM52_MLA_QK_ROPE_HEAD_DIM {
            anyhow::bail!(
                "device MLA attention rotary dim mismatch: expected {} got {}",
                GLM52_MLA_QK_ROPE_HEAD_DIM,
                rotated_k_rope.rotary_dim
            );
        }
        if !scale.is_finite() {
            anyhow::bail!("device MLA attention scale must be finite");
        }
        if q_nope.device_id != self.k_nope.device_id
            || q_rope.device_id != self.k_nope.device_id
            || rotated_k_rope.k_rope_rotated.device_id != self.k_nope.device_id
            || self.values.device_id != self.k_nope.device_id
        {
            anyhow::bail!("device MLA attention buffers must be on the same CUDA device");
        }
        let q_nope_bytes = self
            .rows
            .checked_mul(self.heads)
            .and_then(|values| values.checked_mul(self.nope_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA attention q_nope bytes overflow usize")?;
        let q_rope_bytes = self
            .rows
            .checked_mul(self.heads)
            .and_then(|values| values.checked_mul(rotated_k_rope.rotary_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA attention q_rope bytes overflow usize")?;
        let output_bytes = self
            .rows
            .checked_mul(self.heads)
            .and_then(|values| values.checked_mul(self.v_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA attention output bytes overflow usize")?;
        validate_contiguous_payload_buffer("device MLA attention q_nope", q_nope, q_nope_bytes)?;
        validate_contiguous_payload_buffer("device MLA attention q_rope", q_rope, q_rope_bytes)?;
        validate_contiguous_payload_buffer(
            "device MLA attention k_nope",
            self.k_nope,
            self.k_nope_bytes,
        )?;
        validate_contiguous_payload_buffer(
            "device MLA attention k_rope",
            rotated_k_rope.k_rope_rotated,
            rotated_k_rope.k_rope_rotated_bytes,
        )?;
        validate_contiguous_payload_buffer(
            "device MLA attention values",
            self.values,
            self.values_bytes,
        )?;
        validate_contiguous_payload_buffer(
            "device MLA attention output",
            output_buffer,
            output_bytes,
        )?;
        if output_buffer.device_id != self.k_nope.device_id {
            anyhow::bail!(
                "device MLA attention output is on CUDA device {}, but inputs are on device {}",
                output_buffer.device_id,
                self.k_nope.device_id
            );
        }
        mla_rope_attention_device_buffers_bf16_for_layer(
            layer_id.0 as usize,
            q_nope,
            q_rope,
            self.k_nope,
            rotated_k_rope.k_rope_rotated,
            self.values,
            output_buffer,
            self.rows,
            self.heads,
            self.nope_dim,
            rotated_k_rope.rotary_dim,
            self.v_dim,
            scale,
        )
        .context("executing device MLA RoPE attention")?;
        Ok(RealFullDeviceMlaAttentionParts {
            status: "cuda-kv-cache-mla-rope-attention-device-buffer",
            rows: self.rows,
            heads: self.heads,
            nope_dim: self.nope_dim,
            rope_dim: rotated_k_rope.rotary_dim,
            v_dim: self.v_dim,
            output_bytes,
            output: DeviceBufferGuard::borrowed(library, output_buffer),
            hidden_projection_fused: false,
            ready_event: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run_mla_rope_attention_suffix_bf16(
        &self,
        library: &'static NativeLibrary,
        layer_id: LayerId,
        rotated_k_rope: &RealFullDeviceMlaKvRopeDeviceBuffers,
        q_nope: GlmrtDeviceBuffer,
        q_rope: GlmrtDeviceBuffer,
        query_row_offset: usize,
        query_rows: usize,
        output_buffer: GlmrtDeviceBuffer,
        scale: f32,
    ) -> Result<RealFullDeviceMlaAttentionParts> {
        if rotated_k_rope.rows != self.rows {
            anyhow::bail!(
                "device MLA suffix attention row mismatch: projected rows={} rotated k_rope rows={}",
                self.rows,
                rotated_k_rope.rows
            );
        }
        if query_rows == 0 {
            anyhow::bail!("device MLA suffix attention requires at least one query row");
        }
        if query_row_offset > self.rows || query_rows > self.rows - query_row_offset {
            anyhow::bail!(
                "device MLA suffix attention query rows {}..{} exceed rows {}",
                query_row_offset,
                query_row_offset.saturating_add(query_rows),
                self.rows
            );
        }
        if rotated_k_rope.rotary_dim != GLM52_MLA_QK_ROPE_HEAD_DIM {
            anyhow::bail!(
                "device MLA suffix attention rotary dim mismatch: expected {} got {}",
                GLM52_MLA_QK_ROPE_HEAD_DIM,
                rotated_k_rope.rotary_dim
            );
        }
        if !scale.is_finite() {
            anyhow::bail!("device MLA suffix attention scale must be finite");
        }
        if q_nope.device_id != self.k_nope.device_id
            || q_rope.device_id != self.k_nope.device_id
            || rotated_k_rope.k_rope_rotated.device_id != self.k_nope.device_id
            || self.values.device_id != self.k_nope.device_id
        {
            anyhow::bail!("device MLA suffix attention buffers must be on the same CUDA device");
        }
        let q_nope_bytes = self
            .rows
            .checked_mul(self.heads)
            .and_then(|values| values.checked_mul(self.nope_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA suffix attention q_nope bytes overflow usize")?;
        let q_rope_bytes = self
            .rows
            .checked_mul(self.heads)
            .and_then(|values| values.checked_mul(rotated_k_rope.rotary_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA suffix attention q_rope bytes overflow usize")?;
        let suffix_output_bytes = query_rows
            .checked_mul(self.heads)
            .and_then(|values| values.checked_mul(self.v_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA suffix attention output bytes overflow usize")?;
        validate_contiguous_payload_buffer(
            "device MLA suffix attention q_nope",
            q_nope,
            q_nope_bytes,
        )?;
        validate_contiguous_payload_buffer(
            "device MLA suffix attention q_rope",
            q_rope,
            q_rope_bytes,
        )?;
        validate_contiguous_payload_buffer(
            "device MLA suffix attention k_nope",
            self.k_nope,
            self.k_nope_bytes,
        )?;
        validate_contiguous_payload_buffer(
            "device MLA suffix attention k_rope",
            rotated_k_rope.k_rope_rotated,
            rotated_k_rope.k_rope_rotated_bytes,
        )?;
        validate_contiguous_payload_buffer(
            "device MLA suffix attention values",
            self.values,
            self.values_bytes,
        )?;
        let full_output_bytes = self
            .rows
            .checked_mul(self.heads)
            .and_then(|values| values.checked_mul(self.v_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA suffix attention full output bytes overflow usize")?;
        validate_contiguous_payload_buffer(
            "device MLA suffix attention output",
            output_buffer,
            full_output_bytes,
        )?;
        if output_buffer.device_id != self.k_nope.device_id {
            anyhow::bail!(
                "device MLA suffix attention output is on CUDA device {}, but inputs are on device {}",
                output_buffer.device_id,
                self.k_nope.device_id
            );
        }
        let suffix_status = mla_rope_attention_suffix_device_buffers_bf16_for_layer(
            layer_id.0 as usize,
            q_nope,
            q_rope,
            self.k_nope,
            rotated_k_rope.k_rope_rotated,
            self.values,
            output_buffer,
            self.rows,
            query_row_offset,
            query_rows,
            self.heads,
            self.nope_dim,
            rotated_k_rope.rotary_dim,
            self.v_dim,
            scale,
        )
        .with_context(|| {
            format!(
                "executing device MLA RoPE suffix attention for layer {} rows={} query_offset={} query_rows={} heads={} nope_dim={} rope_dim={} v_dim={}",
                layer_id.0,
                self.rows,
                query_row_offset,
                query_rows,
                self.heads,
                self.nope_dim,
                rotated_k_rope.rotary_dim,
                self.v_dim
            )
        });
        let (rows, output_bytes) = match suffix_status {
            Ok(_) => (query_rows, suffix_output_bytes),
            Err(error) => {
                if device_attention_stage_timing_enabled() {
                    eprintln!(
                        "real_full_mla_suffix_fallback layer_id={} rows={} query_offset={} query_rows={} error={error:#}",
                        layer_id.0, self.rows, query_row_offset, query_rows
                    );
                }
                library
                    .cuda_mla_rope_attention_bf16(
                        q_nope,
                        q_rope,
                        self.k_nope,
                        rotated_k_rope.k_rope_rotated,
                        self.values,
                        output_buffer,
                        self.rows,
                        self.heads,
                        self.nope_dim,
                        rotated_k_rope.rotary_dim,
                        self.v_dim,
                        scale,
                    )
                    .with_context(|| {
                        format!(
                            "falling back to full device MLA RoPE attention after suffix attention failed for layer {} rows={} query_offset={} query_rows={}: {error:#}",
                            layer_id.0, self.rows, query_row_offset, query_rows
                        )
                    })?;
                (self.rows, full_output_bytes)
            }
        };
        Ok(RealFullDeviceMlaAttentionParts {
            status: "cuda-kv-cache-mla-rope-attention-device-buffer",
            rows,
            heads: self.heads,
            nope_dim: self.nope_dim,
            rope_dim: rotated_k_rope.rotary_dim,
            v_dim: self.v_dim,
            output_bytes,
            output: DeviceBufferGuard::borrowed(library, output_buffer),
            hidden_projection_fused: false,
            ready_event: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run_mla_rope_attention_with_uploaded_suffix_bf16(
        &self,
        library: &'static NativeLibrary,
        cache: &mut RealFullDeviceKvCache<'_>,
        rotated_k_rope: &RealFullDeviceMlaKvRopeDeviceBuffers,
        q_nope: GlmrtDeviceBuffer,
        q_rope: GlmrtDeviceBuffer,
        suffix_k_nope: GlmrtDeviceBuffer,
        suffix_k_nope_bytes: usize,
        suffix_k_rope: GlmrtDeviceBuffer,
        suffix_k_rope_bytes: usize,
        suffix_values: GlmrtDeviceBuffer,
        suffix_values_bytes: usize,
        scale: f32,
    ) -> Result<RealFullDeviceMlaAttentionParts> {
        if rotated_k_rope.rows != self.rows {
            anyhow::bail!(
                "device MLA attention uploaded-suffix row mismatch: projected rows={} rotated k_rope rows={}",
                self.rows,
                rotated_k_rope.rows
            );
        }
        if rotated_k_rope.rotary_dim != GLM52_MLA_QK_ROPE_HEAD_DIM {
            anyhow::bail!(
                "device MLA attention uploaded-suffix rotary dim mismatch: expected {} got {}",
                GLM52_MLA_QK_ROPE_HEAD_DIM,
                rotated_k_rope.rotary_dim
            );
        }
        if !scale.is_finite() {
            anyhow::bail!("device MLA attention uploaded-suffix scale must be finite");
        }
        let value_size = std::mem::size_of::<u16>();
        let suffix_k_rope_row_bytes = rotated_k_rope
            .rotary_dim
            .checked_mul(value_size)
            .context("device MLA attention uploaded-suffix k_rope row bytes overflow usize")?;
        if suffix_k_rope_bytes == 0 || suffix_k_rope_bytes % suffix_k_rope_row_bytes != 0 {
            anyhow::bail!(
                "device MLA attention uploaded-suffix rotated k_rope bytes {suffix_k_rope_bytes} are not a non-empty multiple of row bytes {suffix_k_rope_row_bytes}"
            );
        }
        let suffix_rows = suffix_k_rope_bytes / suffix_k_rope_row_bytes;
        let expected_suffix_k_nope_bytes = suffix_rows
            .checked_mul(self.heads)
            .and_then(|values| values.checked_mul(self.nope_dim))
            .and_then(|values| values.checked_mul(value_size))
            .context("device MLA attention uploaded-suffix k_nope bytes overflow usize")?;
        let expected_suffix_values_bytes = suffix_rows
            .checked_mul(self.heads)
            .and_then(|values| values.checked_mul(self.v_dim))
            .and_then(|values| values.checked_mul(value_size))
            .context("device MLA attention uploaded-suffix values bytes overflow usize")?;
        if suffix_k_nope_bytes != expected_suffix_k_nope_bytes {
            anyhow::bail!(
                "device MLA attention uploaded-suffix k_nope byte mismatch: expected {expected_suffix_k_nope_bytes} got {suffix_k_nope_bytes}"
            );
        }
        if suffix_values_bytes != expected_suffix_values_bytes {
            anyhow::bail!(
                "device MLA attention uploaded-suffix value byte mismatch: expected {expected_suffix_values_bytes} got {suffix_values_bytes}"
            );
        }
        let total_rows = self
            .rows
            .checked_add(suffix_rows)
            .context("device MLA attention uploaded-suffix total rows overflow usize")?;
        let q_nope_bytes = total_rows
            .checked_mul(self.heads)
            .and_then(|values| values.checked_mul(self.nope_dim))
            .and_then(|values| values.checked_mul(value_size))
            .context("device MLA attention uploaded-suffix q_nope bytes overflow usize")?;
        let q_rope_bytes = total_rows
            .checked_mul(self.heads)
            .and_then(|values| values.checked_mul(rotated_k_rope.rotary_dim))
            .and_then(|values| values.checked_mul(value_size))
            .context("device MLA attention uploaded-suffix q_rope bytes overflow usize")?;
        let output_bytes = total_rows
            .checked_mul(self.heads)
            .and_then(|values| values.checked_mul(self.v_dim))
            .and_then(|values| values.checked_mul(value_size))
            .context("device MLA attention uploaded-suffix output bytes overflow usize")?;
        validate_contiguous_payload_buffer(
            "device MLA attention uploaded-suffix q_nope",
            q_nope,
            q_nope_bytes,
        )?;
        validate_contiguous_payload_buffer(
            "device MLA attention uploaded-suffix q_rope",
            q_rope,
            q_rope_bytes,
        )?;
        validate_contiguous_payload_buffer(
            "device MLA attention uploaded-suffix k_nope",
            suffix_k_nope,
            suffix_k_nope_bytes,
        )?;
        validate_contiguous_payload_buffer(
            "device MLA attention uploaded-suffix k_rope",
            suffix_k_rope,
            suffix_k_rope_bytes,
        )?;
        validate_contiguous_payload_buffer(
            "device MLA attention uploaded-suffix values",
            suffix_values,
            suffix_values_bytes,
        )?;
        if q_nope.device_id != self.k_nope.device_id
            || q_rope.device_id != self.k_nope.device_id
            || suffix_k_nope.device_id != self.k_nope.device_id
            || suffix_k_rope.device_id != self.k_nope.device_id
            || suffix_values.device_id != self.k_nope.device_id
            || rotated_k_rope.k_rope_rotated.device_id != self.k_nope.device_id
            || self.values.device_id != self.k_nope.device_id
        {
            anyhow::bail!(
                "device MLA attention uploaded-suffix buffers must be on the same CUDA device"
            );
        }

        if !std::ptr::eq(cache.library, library) {
            anyhow::bail!(
                "device MLA attention uploaded-suffix reusable cache belongs to a different native library"
            );
        }
        let combined_k_nope_bytes = self
            .k_nope_bytes
            .checked_add(suffix_k_nope_bytes)
            .context("device MLA attention uploaded-suffix combined k_nope bytes overflow usize")?;
        let combined_k_rope_bytes = rotated_k_rope
            .k_rope_rotated_bytes
            .checked_add(suffix_k_rope_bytes)
            .context("device MLA attention uploaded-suffix combined k_rope bytes overflow usize")?;
        let combined_values_bytes = self
            .values_bytes
            .checked_add(suffix_values_bytes)
            .context("device MLA attention uploaded-suffix combined value bytes overflow usize")?;
        let (combined_k_nope, combined_k_rope, combined_values) = cache
            .attention_combined_buffers(
                combined_k_nope_bytes,
                combined_k_rope_bytes,
                combined_values_bytes,
                "device MLA attention uploaded-suffix combined",
            )
            .context("allocating reusable device MLA attention uploaded-suffix combined buffers")?;
        library
            .cuda_kv_cache_write_bytes(self.k_nope, combined_k_nope, 0, self.k_nope_bytes)
            .context("copying device MLA attention uploaded-suffix prefix k_nope")?;
        library
            .cuda_kv_cache_write_bytes(
                suffix_k_nope,
                combined_k_nope,
                self.k_nope_bytes,
                suffix_k_nope_bytes,
            )
            .context("copying device MLA attention uploaded-suffix k_nope into combined buffer")?;
        library
            .cuda_kv_cache_write_bytes(
                rotated_k_rope.k_rope_rotated,
                combined_k_rope,
                0,
                rotated_k_rope.k_rope_rotated_bytes,
            )
            .context("copying device MLA attention uploaded-suffix prefix k_rope")?;
        library
            .cuda_kv_cache_write_bytes(
                suffix_k_rope,
                combined_k_rope,
                rotated_k_rope.k_rope_rotated_bytes,
                suffix_k_rope_bytes,
            )
            .context("copying device MLA attention uploaded-suffix k_rope into combined buffer")?;
        library
            .cuda_kv_cache_write_bytes(self.values, combined_values, 0, self.values_bytes)
            .context("copying device MLA attention uploaded-suffix prefix values")?;
        library
            .cuda_kv_cache_write_bytes(
                suffix_values,
                combined_values,
                self.values_bytes,
                suffix_values_bytes,
            )
            .context("copying device MLA attention uploaded-suffix values into combined buffer")?;

        let output = DeviceBufferGuard::new(library, output_bytes)
            .context("allocating device MLA attention uploaded-suffix output")?;
        library
            .cuda_mla_rope_attention_bf16(
                q_nope,
                q_rope,
                combined_k_nope,
                combined_k_rope,
                combined_values,
                output.buffer,
                total_rows,
                self.heads,
                self.nope_dim,
                rotated_k_rope.rotary_dim,
                self.v_dim,
                scale,
            )
            .context("executing device MLA RoPE attention with uploaded suffix")?;
        Ok(RealFullDeviceMlaAttentionParts {
            status: "cuda-kv-cache-mla-rope-attention-device-buffer-with-host-suffix",
            rows: total_rows,
            heads: self.heads,
            nope_dim: self.nope_dim,
            rope_dim: rotated_k_rope.rotary_dim,
            v_dim: self.v_dim,
            output_bytes,
            output,
            hidden_projection_fused: false,
            ready_event: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run_mla_rope_attention_with_device_suffix_bf16(
        &self,
        library: &'static NativeLibrary,
        cache: &mut RealFullDeviceKvCache<'_>,
        rotated_k_rope: &RealFullDeviceMlaKvRopeDeviceBuffers,
        q_nope: GlmrtDeviceBuffer,
        q_rope: GlmrtDeviceBuffer,
        suffix_projected: &RealFullDeviceMlaKvProjectedDeviceBuffers,
        suffix_rotated_k_rope: &RealFullDeviceMlaKvRopeDeviceBuffers,
        scale: f32,
    ) -> Result<RealFullDeviceMlaAttentionParts> {
        if rotated_k_rope.rows != self.rows {
            anyhow::bail!(
                "device MLA attention device suffix row mismatch: projected rows={} rotated k_rope rows={}",
                self.rows,
                rotated_k_rope.rows
            );
        }
        if suffix_rotated_k_rope.rows != suffix_projected.rows {
            anyhow::bail!(
                "device MLA attention device suffix row mismatch: suffix projected rows={} rotated rows={}",
                suffix_projected.rows,
                suffix_rotated_k_rope.rows
            );
        }
        if suffix_projected.heads != self.heads
            || suffix_projected.nope_dim != self.nope_dim
            || suffix_projected.v_dim != self.v_dim
        {
            anyhow::bail!(
                "device MLA attention device suffix shape mismatch: prefix heads={} nope={} v={} suffix heads={} nope={} v={}",
                self.heads,
                self.nope_dim,
                self.v_dim,
                suffix_projected.heads,
                suffix_projected.nope_dim,
                suffix_projected.v_dim
            );
        }
        if rotated_k_rope.rotary_dim != GLM52_MLA_QK_ROPE_HEAD_DIM
            || suffix_rotated_k_rope.rotary_dim != GLM52_MLA_QK_ROPE_HEAD_DIM
        {
            anyhow::bail!(
                "device MLA attention device suffix rotary dim mismatch: prefix={} suffix={} expected {}",
                rotated_k_rope.rotary_dim,
                suffix_rotated_k_rope.rotary_dim,
                GLM52_MLA_QK_ROPE_HEAD_DIM
            );
        }
        if !scale.is_finite() {
            anyhow::bail!("device MLA attention device suffix scale must be finite");
        }
        if q_nope.device_id != self.k_nope.device_id
            || q_rope.device_id != self.k_nope.device_id
            || rotated_k_rope.k_rope_rotated.device_id != self.k_nope.device_id
            || self.values.device_id != self.k_nope.device_id
            || suffix_projected.k_nope.device_id != self.k_nope.device_id
            || suffix_projected.values.device_id != self.k_nope.device_id
            || suffix_rotated_k_rope.k_rope_rotated.device_id != self.k_nope.device_id
        {
            anyhow::bail!(
                "device MLA attention device suffix buffers must be on the same CUDA device"
            );
        }
        if !std::ptr::eq(cache.library, library) {
            anyhow::bail!(
                "device MLA attention device-suffix reusable cache belongs to a different native library"
            );
        }
        let value_size = std::mem::size_of::<u16>();
        let total_rows = self
            .rows
            .checked_add(suffix_projected.rows)
            .context("device MLA attention device suffix total rows overflow usize")?;
        let q_nope_bytes = total_rows
            .checked_mul(self.heads)
            .and_then(|values| values.checked_mul(self.nope_dim))
            .and_then(|values| values.checked_mul(value_size))
            .context("device MLA attention device suffix q_nope bytes overflow usize")?;
        let q_rope_bytes = total_rows
            .checked_mul(self.heads)
            .and_then(|values| values.checked_mul(rotated_k_rope.rotary_dim))
            .and_then(|values| values.checked_mul(value_size))
            .context("device MLA attention device suffix q_rope bytes overflow usize")?;
        let output_bytes = total_rows
            .checked_mul(self.heads)
            .and_then(|values| values.checked_mul(self.v_dim))
            .and_then(|values| values.checked_mul(value_size))
            .context("device MLA attention device suffix output bytes overflow usize")?;
        validate_contiguous_payload_buffer(
            "device MLA attention device suffix q_nope",
            q_nope,
            q_nope_bytes,
        )?;
        validate_contiguous_payload_buffer(
            "device MLA attention device suffix q_rope",
            q_rope,
            q_rope_bytes,
        )?;

        let combined_k_nope_bytes = self
            .k_nope_bytes
            .checked_add(suffix_projected.k_nope_bytes)
            .context("device MLA attention device suffix combined k_nope bytes overflow usize")?;
        let combined_k_rope_bytes = rotated_k_rope
            .k_rope_rotated_bytes
            .checked_add(suffix_rotated_k_rope.k_rope_rotated_bytes)
            .context("device MLA attention device suffix combined k_rope bytes overflow usize")?;
        let combined_values_bytes = self
            .values_bytes
            .checked_add(suffix_projected.values_bytes)
            .context("device MLA attention device suffix combined value bytes overflow usize")?;
        let (combined_k_nope, combined_k_rope, combined_values) = cache
            .attention_combined_buffers(
                combined_k_nope_bytes,
                combined_k_rope_bytes,
                combined_values_bytes,
                "device MLA attention device-suffix combined",
            )
            .context("allocating reusable device MLA attention device-suffix combined buffers")?;
        library
            .cuda_kv_cache_write_bytes(self.k_nope, combined_k_nope, 0, self.k_nope_bytes)
            .context("copying device MLA attention device suffix prefix k_nope")?;
        library
            .cuda_kv_cache_write_bytes(
                suffix_projected.k_nope,
                combined_k_nope,
                self.k_nope_bytes,
                suffix_projected.k_nope_bytes,
            )
            .context("copying device MLA attention device suffix k_nope into combined buffer")?;
        library
            .cuda_kv_cache_write_bytes(
                rotated_k_rope.k_rope_rotated,
                combined_k_rope,
                0,
                rotated_k_rope.k_rope_rotated_bytes,
            )
            .context("copying device MLA attention device suffix prefix k_rope")?;
        library
            .cuda_kv_cache_write_bytes(
                suffix_rotated_k_rope.k_rope_rotated,
                combined_k_rope,
                rotated_k_rope.k_rope_rotated_bytes,
                suffix_rotated_k_rope.k_rope_rotated_bytes,
            )
            .context("copying device MLA attention device suffix k_rope into combined buffer")?;
        library
            .cuda_kv_cache_write_bytes(self.values, combined_values, 0, self.values_bytes)
            .context("copying device MLA attention device suffix prefix values")?;
        library
            .cuda_kv_cache_write_bytes(
                suffix_projected.values,
                combined_values,
                self.values_bytes,
                suffix_projected.values_bytes,
            )
            .context("copying device MLA attention device suffix values into combined buffer")?;

        let output = DeviceBufferGuard::new(library, output_bytes)
            .context("allocating device MLA attention device suffix output")?;
        library
            .cuda_mla_rope_attention_bf16(
                q_nope,
                q_rope,
                combined_k_nope,
                combined_k_rope,
                combined_values,
                output.buffer,
                total_rows,
                self.heads,
                self.nope_dim,
                rotated_k_rope.rotary_dim,
                self.v_dim,
                scale,
            )
            .context("executing device MLA RoPE attention with device suffix")?;
        Ok(RealFullDeviceMlaAttentionParts {
            status: "cuda-kv-cache-mla-rope-attention-device-buffer-with-device-suffix",
            rows: total_rows,
            heads: self.heads,
            nope_dim: self.nope_dim,
            rope_dim: rotated_k_rope.rotary_dim,
            v_dim: self.v_dim,
            output_bytes,
            output,
            hidden_projection_fused: false,
            ready_event: None,
        })
    }
}

#[cfg_attr(not(test), allow(dead_code))]
impl RealFullDeviceMlaQueryDeviceBuffers {
    fn q_nope_buffer(&self) -> GlmrtDeviceBuffer {
        self.q_nope
    }

    fn q_rope_rotated_buffer(&self) -> GlmrtDeviceBuffer {
        self.q_rope_rotated
    }
}

#[cfg_attr(not(test), allow(dead_code))]
impl RealFullDeviceMlaAttentionParts {
    pub(in crate::commands::real_full) fn status(&self) -> &'static str {
        self.status
    }

    pub(in crate::commands::real_full) fn rows(&self) -> usize {
        self.rows
    }

    pub(in crate::commands::real_full) fn heads(&self) -> usize {
        self.heads
    }

    pub(in crate::commands::real_full) fn nope_dim(&self) -> usize {
        self.nope_dim
    }

    pub(in crate::commands::real_full) fn rope_dim(&self) -> usize {
        self.rope_dim
    }

    pub(in crate::commands::real_full) fn v_dim(&self) -> usize {
        self.v_dim
    }

    fn hidden_projection_fused(&self) -> bool {
        self.hidden_projection_fused
    }

    fn take_ready_event(&mut self) -> Option<Arc<CoordinatorCudaEvent>> {
        self.ready_event.take()
    }

    pub(in crate::commands::real_full) fn output_buffer(&self) -> GlmrtDeviceBuffer {
        self.output.buffer
    }

    pub(in crate::commands::real_full) fn output_row_buffer(
        &self,
        row_start: usize,
        row_count: usize,
    ) -> Result<GlmrtDeviceBuffer> {
        let row_bytes = self
            .heads
            .checked_mul(self.v_dim)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA attention output row bytes overflow usize")?;
        let offset_bytes = row_start
            .checked_mul(row_bytes)
            .context("device MLA attention output row offset overflow usize")?;
        let view_bytes = row_count
            .checked_mul(row_bytes)
            .context("device MLA attention output row view bytes overflow usize")?;
        let end = offset_bytes
            .checked_add(view_bytes)
            .context("device MLA attention output row view end overflow usize")?;
        if row_start > self.rows || row_count > self.rows || end > self.output_bytes {
            anyhow::bail!(
                "device MLA attention output row view row_start={row_start} row_count={row_count} exceeds rows={} bytes={}",
                self.rows,
                self.output_bytes
            );
        }
        Ok(GlmrtDeviceBuffer {
            ptr: self
                .output
                .buffer
                .ptr
                .cast::<u8>()
                .wrapping_add(offset_bytes)
                .cast(),
            bytes: view_bytes,
            device_id: self.output.buffer.device_id,
            flags: self.output.buffer.flags,
        })
    }

    pub(in crate::commands::real_full) fn into_device_bf16_output(
        self,
        rows: usize,
        values_per_row: usize,
        label: &'static str,
    ) -> Result<DeviceBf16Output> {
        let status = self.status;
        let library = self.output.library;
        let output = self.output.into_buffer()?;
        device_bf16_output_from_owned_device_buffer(
            library,
            output,
            rows,
            values_per_row,
            status,
            label,
        )
    }

    pub(in crate::commands::real_full) fn copy_to_host(
        &self,
    ) -> Result<RealFullDeviceMlaAttentionReadback> {
        let mut output_bf16 = vec![0_u8; self.output_bytes];
        self.output
            .library
            .copy_d2h(&mut output_bf16, self.output.buffer)?;
        Ok(RealFullDeviceMlaAttentionReadback {
            status: "cuda-kv-cache-mla-rope-attention-readback",
            rows: self.rows,
            heads: self.heads,
            nope_dim: self.nope_dim,
            rope_dim: self.rope_dim,
            v_dim: self.v_dim,
            output_bf16,
        })
    }
}

pub(in crate::commands::real_full) fn real_full_device_kv_block_io(
    config: &KvCacheConfig,
    descriptor: &KvBlockDescriptor,
) -> Result<RealFullDeviceKvBlockIo> {
    let offset_bytes = config
        .descriptor_offset_bytes(descriptor)
        .with_context(|| {
            format!(
                concat!(
                    "KV descriptor layer={} token_start={} token_count={} ",
                    "is outside the {} cache capacity"
                ),
                descriptor.layer_id.0,
                descriptor.token_start.0,
                descriptor.token_count,
                config.layout_label()
            )
        })?;
    let payload_bytes = config
        .descriptor_payload_bytes(descriptor)
        .with_context(|| "KV descriptor payload size is invalid for cache config")?;
    Ok(RealFullDeviceKvBlockIo {
        offset_bytes,
        payload_bytes,
    })
}

pub(in crate::commands::real_full) fn real_full_device_kv_block_ios(
    config: &KvCacheConfig,
    descriptors: &[KvBlockDescriptor],
) -> Result<Vec<RealFullDeviceKvBlockIo>> {
    descriptors
        .iter()
        .map(|descriptor| real_full_device_kv_block_io(config, descriptor))
        .collect()
}

fn real_full_device_main_mla_row_bytes(config: &KvCacheConfig) -> Result<usize> {
    config
        .main_mla_row_bytes()
        .or_else(|| {
            (config.layout == KvLayout::ExpandedDebugOnly)
                .then(|| config.layer_bytes_per_token(LayerId(0)))
        })
        .context("device main MLA cache format has no row geometry")
}

fn real_full_device_main_kv_block_io(
    config: &KvCacheConfig,
    descriptor: &KvBlockDescriptor,
) -> Result<RealFullDeviceKvBlockIo> {
    // Validate the public/logical descriptor first. DSA layers retain their
    // existing total layer span, but the device cache stores all main MLA rows
    // first so the 656-byte FP8 rows are directly page-addressable.
    real_full_device_kv_block_io(config, descriptor)?;
    let layer_base = config
        .layer_base_offset_bytes(descriptor.layer_id)
        .context("device main MLA layer base is invalid")?;
    let row_bytes = real_full_device_main_mla_row_bytes(config)?;
    let token_start = usize::try_from(descriptor.token_start.0)
        .context("device main MLA token start does not fit usize")?;
    let offset_bytes = token_start
        .checked_mul(row_bytes)
        .and_then(|offset| layer_base.checked_add(offset))
        .context("device main MLA block offset overflow usize")?;
    let payload_bytes = descriptor
        .token_count
        .checked_mul(row_bytes)
        .context("device main MLA block bytes overflow usize")?;
    Ok(RealFullDeviceKvBlockIo {
        offset_bytes,
        payload_bytes,
    })
}

fn real_full_device_dsa_bf16_block_io(
    config: &KvCacheConfig,
    descriptor: &KvBlockDescriptor,
) -> Result<Option<RealFullDeviceKvBlockIo>> {
    if !config.layer_has_dsa_indexer(descriptor.layer_id) {
        return Ok(None);
    }
    real_full_device_kv_block_io(config, descriptor)?;
    let layer_base = config
        .layer_base_offset_bytes(descriptor.layer_id)
        .context("device DSA layer base is invalid")?;
    let main_row_bytes = real_full_device_main_mla_row_bytes(config)?;
    let dsa_row_bytes = config
        .dsa_index_head_dim
        .checked_mul(std::mem::size_of::<u16>())
        .context("device DSA BF16 row bytes overflow usize")?;
    let dsa_base = main_row_bytes
        .checked_mul(config.max_tokens)
        .and_then(|offset| layer_base.checked_add(offset))
        .context("device DSA plane base overflow usize")?;
    let token_start = usize::try_from(descriptor.token_start.0)
        .context("device DSA token start does not fit usize")?;
    let offset_bytes = token_start
        .checked_mul(dsa_row_bytes)
        .and_then(|offset| dsa_base.checked_add(offset))
        .context("device DSA block offset overflow usize")?;
    let payload_bytes = descriptor
        .token_count
        .checked_mul(dsa_row_bytes)
        .context("device DSA block bytes overflow usize")?;
    Ok(Some(RealFullDeviceKvBlockIo {
        offset_bytes,
        payload_bytes,
    }))
}

fn real_full_device_dsa_index_k_b12x_layer_bytes(config: &KvCacheConfig) -> Result<usize> {
    let pages = config
        .max_tokens
        .checked_add(GLMRT_CUDA_GLM_DSA_PAGE_SIZE - 1)
        .context("device B12X DSA cache page rounding overflow usize")?
        / GLMRT_CUDA_GLM_DSA_PAGE_SIZE;
    pages
        .checked_mul(GLMRT_CUDA_GLM_DSA_PACKED_PAGE_BYTES)
        .context("device B12X DSA cache layer bytes overflow usize")
}

fn real_full_device_dsa_index_k_b12x_capacity_bytes(config: &KvCacheConfig) -> Result<usize> {
    real_full_device_dsa_index_k_b12x_layer_bytes(config)?
        .checked_mul(config.dsa_indexer_layers)
        .context("device B12X DSA cache capacity bytes overflow usize")
}

fn contiguous_device_main_kv_block_span(
    config: &KvCacheConfig,
    descriptors: &[KvBlockDescriptor],
) -> Result<Option<(usize, usize, Vec<RealFullDeviceKvBlockIo>)>> {
    let ios = descriptors
        .iter()
        .map(|descriptor| real_full_device_main_kv_block_io(config, descriptor))
        .collect::<Result<Vec<_>>>()?;
    let Some(first) = ios.first() else {
        return Ok(None);
    };
    let start = first.offset_bytes;
    let mut end = start;
    for io in &ios {
        if io.offset_bytes != end {
            return Ok(None);
        }
        end = end
            .checked_add(io.payload_bytes)
            .context("contiguous device main MLA block span overflows usize")?;
    }
    Ok(Some((start, end - start, ios)))
}

fn contiguous_device_kv_block_span(
    config: &KvCacheConfig,
    descriptors: &[KvBlockDescriptor],
) -> Result<Option<(usize, usize, Vec<RealFullDeviceKvBlockIo>)>> {
    let ios = real_full_device_kv_block_ios(config, descriptors)?;
    let Some(first) = ios.first() else {
        return Ok(None);
    };
    let start = first.offset_bytes;
    let mut end = start;
    for io in &ios {
        if io.offset_bytes != end {
            return Ok(None);
        }
        end = end
            .checked_add(io.payload_bytes)
            .context("contiguous device KV block span overflows usize")?;
    }
    Ok(Some((start, end - start, ios)))
}

#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn real_full_device_kv_roundtrip(
    config: &KvCacheConfig,
    descriptors: &[KvBlockDescriptor],
    payloads: &[Vec<u8>],
) -> Result<RealFullDeviceKvRoundTrip> {
    if descriptors.is_empty() {
        return Ok(RealFullDeviceKvRoundTrip {
            status: "cuda-kv-cache-not-needed",
            writes: 0,
            reads: 0,
            bytes: 0,
            uses_device_kv_cache: false,
        });
    }
    if descriptors.len() != payloads.len() {
        anyhow::bail!(
            "device KV roundtrip descriptor/payload mismatch: descriptors={} payloads={}",
            descriptors.len(),
            payloads.len()
        );
    }
    validate_device_kv_payloads(config, descriptors, payloads, "device KV roundtrip")?;
    let cuda_required = coordinator_cuda_reference_kernels_enabled();
    let library = match cuda_native_library() {
        Ok(library) => library,
        Err(error) => return device_kv_library_unavailable(error, cuda_required),
    };
    match execute_real_full_device_kv_roundtrip(library, config, descriptors, payloads) {
        Ok(roundtrip) => Ok(roundtrip),
        Err(error) => device_kv_roundtrip_failed(error, cuda_required),
    }
}

impl RealFullDeviceKvExecutionMirror {
    pub(in crate::commands::real_full) fn new(config: KvCacheConfig) -> Result<Self> {
        let cuda_required = coordinator_cuda_reference_kernels_enabled();
        let library = match cuda_native_library() {
            Ok(library) => library,
            Err(error) => {
                if cuda_required {
                    return Err(error).context(
                        "real-full live scheduler device KV cache requires CUDA reference execution but no CUDA-enabled native library is available",
                    );
                }
                return Ok(Self::unavailable("cuda-kv-cache-unavailable"));
            }
        };
        match RealFullDeviceKvCache::new(library, config) {
            Ok(cache) => Ok(Self {
                cache: Some(cache),
                status: "cuda-kv-cache-live-scheduler",
                writes: 0,
                reads: 0,
                bytes: 0,
                scheduler_attention_weights: None,
                scheduler_attention_queries: BTreeMap::new(),
                scheduler_attention_resident_uploads: 0,
                scheduler_attention_resident_buffer_uses: 0,
                scheduler_attention_descriptors: Vec::new(),
                scheduler_attention_positions: Vec::new(),
                scheduler_attention_query_positions: Vec::new(),
                scheduler_attention_weight_upload_bf16_scratch: Vec::new(),
                scheduler_attention_projected_query_upload_bf16_scratch: Vec::new(),
                host_readback_payload_scratch: Vec::new(),
            }),
            Err(error) => {
                if cuda_required {
                    return Err(error)
                        .context("allocating live scheduler real-full device KV cache");
                }
                Ok(Self::unavailable(
                    real_full_device_kv_roundtrip_error_status(&error),
                ))
            }
        }
    }

    pub(in crate::commands::real_full) fn new_with_storage(
        storage: RealFullDeviceKvStorageHandle,
        physical_token_base: usize,
        logical_capacity_tokens: usize,
    ) -> Result<Self> {
        let library = storage.library;
        let config = storage.config.clone();
        let cache = RealFullDeviceKvCache::new_with_storage(
            library,
            config,
            storage,
            physical_token_base,
            logical_capacity_tokens,
        )
        .context("binding live scheduler to shared real-full device KV storage")?;
        Ok(Self {
            cache: Some(cache),
            status: "cuda-kv-cache-live-scheduler-shared-storage",
            writes: 0,
            reads: 0,
            bytes: 0,
            scheduler_attention_weights: None,
            scheduler_attention_queries: BTreeMap::new(),
            scheduler_attention_resident_uploads: 0,
            scheduler_attention_resident_buffer_uses: 0,
            scheduler_attention_descriptors: Vec::new(),
            scheduler_attention_positions: Vec::new(),
            scheduler_attention_query_positions: Vec::new(),
            scheduler_attention_weight_upload_bf16_scratch: Vec::new(),
            scheduler_attention_projected_query_upload_bf16_scratch: Vec::new(),
            host_readback_payload_scratch: Vec::new(),
        })
    }

    pub(in crate::commands::real_full) fn storage_handle(
        &self,
    ) -> Option<RealFullDeviceKvStorageHandle> {
        self.cache.as_ref().map(|cache| Arc::clone(&cache.storage))
    }

    pub(in crate::commands::real_full) fn resolve_mtp_tentative_frontiers(
        &mut self,
        reservation_id: u64,
        sequence_id: &str,
        token_start: usize,
        draft_tokens: usize,
        accepted_tokens: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            accepted_tokens <= draft_tokens,
            "accepted MTP tokens {accepted_tokens} exceed draft tokens {draft_tokens}"
        );
        if accepted_tokens == draft_tokens {
            return Ok(());
        }
        let draft_end = token_start
            .checked_add(draft_tokens)
            .context("MTP tentative frontier draft end overflows usize")?;
        let accepted_end = token_start
            .checked_add(accepted_tokens)
            .context("MTP tentative frontier accepted end overflows usize")?;
        let Some(cache) = self.cache.as_mut() else {
            return Ok(());
        };
        for slot in &mut cache.mla_attention_ready_frontiers {
            let Some(mut frontier) = slot.take() else {
                continue;
            };
            let frontier_end = frontier
                .token_start
                .checked_add(frontier.rows)
                .context("attention-ready MLA frontier end overflows usize")?;
            if frontier.reservation_id == reservation_id
                && frontier.sequence_id == sequence_id
                && frontier_end > token_start
                && frontier.token_start < draft_end
                && frontier_end > accepted_end
            {
                frontier.rows = accepted_end.saturating_sub(frontier.token_start);
            }
            if frontier.rows > 0 {
                *slot = Some(frontier);
            }
        }
        Ok(())
    }

    pub(in crate::commands::real_full) fn rewind_attention_ready_frontier(
        &mut self,
        reservation_id: u64,
        sequence_id: &str,
        layer_id: LayerId,
        token_start: usize,
    ) -> Result<()> {
        let Some(cache) = self.cache.as_mut() else {
            return Ok(());
        };
        let layer_index =
            usize::try_from(layer_id.0).context("attention-ready rewind layer exceeds usize")?;
        let Some(slot) = cache.mla_attention_ready_frontiers.get_mut(layer_index) else {
            return Ok(());
        };
        rewind_device_kv_attention_ready_frontier(
            slot,
            reservation_id,
            sequence_id,
            layer_id,
            token_start,
        )
    }

    pub(in crate::commands::real_full) fn reset_sequence_metadata(&mut self) {
        self.writes = 0;
        self.reads = 0;
        self.bytes = 0;
        self.scheduler_attention_resident_uploads = 0;
        self.scheduler_attention_resident_buffer_uses = 0;
        self.scheduler_attention_descriptors.clear();
        self.scheduler_attention_positions.clear();
        self.scheduler_attention_query_positions.clear();
        self.scheduler_attention_weight_upload_bf16_scratch.clear();
        self.scheduler_attention_projected_query_upload_bf16_scratch
            .clear();
        self.host_readback_payload_scratch.clear();
        if let Some(cache) = self.cache.as_mut() {
            cache.mla_write_positions.clear();
            // Rebinding the scheduler wrapper is part of every recurrent
            // cycle. Keep the bounded attention-ready payload resident across
            // those rebinds; its reservation/sequence/layer tuple prevents a
            // different request from reading or appending stale rows, and the
            // cached-prefix seed path replaces it on the first write.
        }
    }

    pub(in crate::commands::real_full) fn rebind_physical_extent(
        &mut self,
        physical_token_base: usize,
        logical_capacity_tokens: usize,
    ) -> Result<()> {
        let cache = self
            .cache
            .as_mut()
            .context("live scheduler device KV cache is unavailable for extent rebind")?;
        anyhow::ensure!(
            logical_capacity_tokens > 0
                && physical_token_base
                    .checked_add(logical_capacity_tokens)
                    .is_some_and(|end| end <= cache.config.max_tokens),
            "device KV extent [{physical_token_base}..+{logical_capacity_tokens}) exceeds {} physical tokens",
            cache.config.max_tokens
        );
        cache.physical_token_base = physical_token_base;
        cache.physical_pages = None;
        cache.physical_page_table_key = next_physical_page_table_key();
        cache.logical_capacity_tokens = logical_capacity_tokens;
        self.reset_sequence_metadata();
        Ok(())
    }

    pub(in crate::commands::real_full) fn rebind_physical_pages(
        &mut self,
        physical_pages: &[u32],
        logical_capacity_tokens: usize,
    ) -> Result<()> {
        let cache = self
            .cache
            .as_mut()
            .context("live scheduler device KV cache is unavailable for page-table rebind")?;
        cache
            .rebind_physical_pages(physical_pages, logical_capacity_tokens)
            .context("rebinding recycled scheduler device KV page table")?;
        self.reset_sequence_metadata();
        Ok(())
    }

    pub(in crate::commands::real_full) fn extend_physical_pages(
        &mut self,
        physical_pages: &[u32],
        logical_capacity_tokens: usize,
    ) -> Result<()> {
        let cache = self
            .cache
            .as_mut()
            .context("live scheduler device KV cache is unavailable for page-table extension")?;
        cache
            .rebind_physical_pages(physical_pages, logical_capacity_tokens)
            .context("uploading extended scheduler device KV page table")
    }

    pub(in crate::commands::real_full) fn copy_target_kv_boundary_page(
        &mut self,
        source_page: u32,
        destination_page: u32,
        valid_tokens: usize,
    ) -> Result<()> {
        let cache = self
            .cache
            .as_mut()
            .context("live scheduler device KV cache is unavailable for radix boundary copy")?;
        cache.copy_target_kv_boundary_page(source_page, destination_page, valid_tokens)
    }

    fn unavailable(status: &'static str) -> Self {
        Self {
            cache: None,
            status,
            writes: 0,
            reads: 0,
            bytes: 0,
            scheduler_attention_weights: None,
            scheduler_attention_queries: BTreeMap::new(),
            scheduler_attention_resident_uploads: 0,
            scheduler_attention_resident_buffer_uses: 0,
            scheduler_attention_descriptors: Vec::new(),
            scheduler_attention_positions: Vec::new(),
            scheduler_attention_query_positions: Vec::new(),
            scheduler_attention_weight_upload_bf16_scratch: Vec::new(),
            scheduler_attention_projected_query_upload_bf16_scratch: Vec::new(),
            host_readback_payload_scratch: Vec::new(),
        }
    }

    pub(in crate::commands::real_full) fn write_host_blocks(
        &mut self,
        descriptors: &[KvBlockDescriptor],
        payloads: &[Vec<u8>],
    ) -> Result<()> {
        if descriptors.is_empty() {
            return Ok(());
        }
        let Some(cache) = self.cache.as_mut() else {
            return Ok(());
        };
        validate_device_kv_payloads(
            cache.config(),
            descriptors,
            payloads,
            "live scheduler device KV write",
        )?;
        let writes = cache
            .write_host_blocks_from_pinned_staging(descriptors, payloads)
            .context("writing live scheduler device KV blocks")?;
        self.writes += writes.len();
        self.bytes += writes.iter().map(|io| io.payload_bytes).sum::<usize>();
        Ok(())
    }

    pub(in crate::commands::real_full) fn write_projected_mla_kv_a_device_blocks_bf16(
        &mut self,
        descriptors: &[KvBlockDescriptor],
        projected_kv_a_buffer: GlmrtDeviceBuffer,
        kv_norm_weight: Option<GlmrtDeviceBuffer>,
    ) -> Result<Option<Vec<RealFullDeviceKvBlockIo>>> {
        if descriptors.is_empty() {
            return Ok(Some(Vec::new()));
        }
        let Some(cache) = self.cache.as_mut() else {
            return Ok(None);
        };
        let layer_id = descriptors[0].layer_id;
        for descriptor in descriptors {
            if descriptor.layer_id != layer_id {
                anyhow::bail!(
                    "device MLA projected kv_a cache write requires same-layer descriptors, got layer {} and {}",
                    layer_id.0,
                    descriptor.layer_id.0
                );
            }
        }
        if cache.config().layer_has_dsa_indexer(layer_id) {
            anyhow::bail!(
                "device MLA projected kv_a cache write does not include DSA/indexer payload bytes for layer {}",
                layer_id.0
            );
        }
        let payload_stride_bytes = (GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM)
            .checked_mul(std::mem::size_of::<u16>())
            .context("device MLA projected kv_a cache write stride overflow usize")?;
        let cache_stride_bytes = cache.config().layer_bytes_per_token(layer_id);
        let rows = descriptors
            .iter()
            .try_fold(0_usize, |acc, descriptor| {
                acc.checked_add(descriptor.token_count)
            })
            .context("device MLA projected kv_a cache write row count overflow usize")?;
        if rows == 0 {
            anyhow::bail!("device MLA projected kv_a cache write requires at least one row");
        }
        let payload_bytes = rows
            .checked_mul(payload_stride_bytes)
            .context("device MLA projected kv_a cache write payload bytes overflow usize")?;
        validate_contiguous_payload_buffer(
            "device MLA projected kv_a cache write src",
            projected_kv_a_buffer,
            payload_bytes,
        )?;
        if projected_kv_a_buffer.device_id != cache.storage.cache.device_id {
            anyhow::bail!(
                "device MLA projected kv_a cache write src is on CUDA device {}, but cache is on device {}",
                projected_kv_a_buffer.device_id,
                cache.storage.cache.device_id
            );
        }
        let prepared_kv_a_buffer = if cache.config().mla_representation
            == MlaKvCacheRepresentation::NormalizedRotated
        {
            Some(
                cache
                    .prepare_projected_mla_kv_for_cache(
                        descriptors,
                        projected_kv_a_buffer,
                        kv_norm_weight.context(
                            "normalized-rotated MLA KV cache write requires a KV normalization weight",
                        )?,
                        payload_stride_bytes,
                    )
                    .context("preparing projected MLA KV rows for cache storage")?,
            )
        } else {
            None
        };
        let cache_projected_buffer = prepared_kv_a_buffer.unwrap_or(projected_kv_a_buffer);
        let write_src = match cache.config().dtype {
            KvCacheDType::Bf16 => {
                if cache_stride_bytes != payload_stride_bytes {
                    anyhow::bail!(
                        "device MLA projected kv_a cache write stride mismatch for layer {}: expected {} got {}",
                        layer_id.0,
                        payload_stride_bytes,
                        cache_stride_bytes
                    );
                }
                cache_projected_buffer
            }
            KvCacheDType::Fp8 => {
                if cache_stride_bytes != GLM52_MLA_FP8_DS_BYTES_PER_TOKEN {
                    anyhow::bail!(
                        "device MLA FP8 projected kv_a cache write stride mismatch for layer {}: expected {} got {}",
                        layer_id.0,
                        GLM52_MLA_FP8_DS_BYTES_PER_TOKEN,
                        cache_stride_bytes
                    );
                }
                let packed_bytes = rows
                    .checked_mul(GLM52_MLA_FP8_DS_BYTES_PER_TOKEN)
                    .context("device MLA FP8 projected kv_a packed bytes overflow usize")?;
                let packed = cache
                    .mla_fp8_packed_write_payload
                    .buffer(packed_bytes, "device MLA FP8 packed KV payload")?;
                cache
                    .pack_mla_kv_fp8_ds_mla(
                        cache_projected_buffer,
                        packed,
                        rows,
                        payload_stride_bytes,
                        GLM52_MLA_FP8_DS_BYTES_PER_TOKEN,
                    )
                    .context("packing device MLA projected kv_a as FP8 DS payload")?;
                packed
            }
            KvCacheDType::Nvfp4 => {
                if cache_stride_bytes != GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN {
                    anyhow::bail!(
                        "device MLA MXFP4 projected kv_a cache write stride mismatch for layer {}: expected {} got {}",
                        layer_id.0,
                        GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN,
                        cache_stride_bytes
                    );
                }
                let packed_bytes = rows
                    .checked_mul(GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN)
                    .context("device MLA MXFP4 projected kv_a packed bytes overflow usize")?;
                let packed = cache
                    .mla_mxfp4_packed_write_payload
                    .buffer(packed_bytes, "device MLA MXFP4 packed KV payload")?;
                cache
                    .pack_mla_kv_mxfp4_ds_mla(
                        cache_projected_buffer,
                        packed,
                        rows,
                        payload_stride_bytes,
                        GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN,
                    )
                    .context("packing device MLA projected kv_a as MXFP4 DS payload")?;
                packed
            }
            dtype => {
                anyhow::bail!(
                    "device MLA projected kv_a cache write requires BF16, FP8, or NVFP4 cache payloads, got {}",
                    dtype.label()
                );
            }
        };
        let writes = cache
            .write_blocks_from_contiguous_device(descriptors, write_src)
            .context("writing device MLA projected kv_a blocks to live KV cache")?;
        self.writes += writes.len();
        self.bytes += writes.iter().map(|io| io.payload_bytes).sum::<usize>();
        Ok(Some(writes))
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::commands::real_full) fn write_mla_decode_kv_device_block_bf16(
        &mut self,
        descriptors: &[KvBlockDescriptor],
        hidden: GlmrtDeviceBuffer,
        input_norm_weight: GlmrtDeviceBuffer,
        kv_a_weight: GlmrtDeviceBuffer,
        kv_norm_weight: GlmrtDeviceBuffer,
        dsa_weights: Option<MlaDecodeKvDsaProjectionWeights>,
        eps: f32,
        theta: f32,
    ) -> Result<Option<RealFullDeviceMlaDecodeKvCommit>> {
        let [descriptor] = descriptors else {
            return Ok(None);
        };
        if descriptor.token_count != 1 {
            return Ok(None);
        }
        let Some(cache) = self.cache.as_mut() else {
            return Ok(None);
        };
        if cache.config().mla_representation != MlaKvCacheRepresentation::NormalizedRotated {
            return Ok(None);
        }
        if cache.config().layer_has_dsa_indexer(descriptor.layer_id) != dsa_weights.is_some() {
            anyhow::bail!(
                "decode KV DSA projection mismatch for layer {}: cache_requires_dsa={} projection_has_dsa={}",
                descriptor.layer_id.0,
                cache.config().layer_has_dsa_indexer(descriptor.layer_id),
                dsa_weights.is_some()
            );
        }
        let io = real_full_device_kv_block_io(cache.config(), descriptor)
            .context("planning fused decode KV cache row write")?;
        let main_io = cache
            .physical_main_kv_block_io(descriptor)
            .context("planning fused decode main KV cache row write")?;
        let cache_row = device_buffer_byte_view(
            cache.storage.cache,
            main_io.offset_bytes,
            main_io.payload_bytes,
            "fused decode main KV cache row",
        )?;
        let dsa_cache_row = cache
            .physical_dsa_bf16_block_io(descriptor)?
            .map(|dsa_io| {
                device_buffer_byte_view(
                    cache.storage.cache,
                    dsa_io.offset_bytes,
                    dsa_io.payload_bytes,
                    "fused decode DSA cache row",
                )
            })
            .transpose()?;
        let dsa_index_k_cache = cache.dsa_index_k_cache_b12x_for_layer(descriptor.layer_id)?;
        let position = u32::try_from(descriptor.token_start.0)
            .context("fused decode KV position exceeds u32")?;
        let physical_position = u32::try_from(
            cache.physical_token_position(
                usize::try_from(descriptor.token_start.0)
                    .context("fused decode KV position does not fit usize")?,
            )?,
        )
        .context("fused decode physical KV position exceeds u32")?;
        // Native NVFP4 sparse MLA consumes the canonical 432-byte cache row
        // directly.  Only the BF16/FP8 cache families retain an auxiliary
        // attention-ready frontier.
        let frontier_target = if cache.config().dtype == KvCacheDType::Nvfp4 {
            None
        } else {
            let attention_ready_row_stride =
                attention_ready_frontier_row_stride_bytes(cache.config().dtype)?;
            cache.attention_ready_mla_frontier_append_target(
                descriptors,
                attention_ready_row_stride,
            )?
        };
        let (attention_ready_row, next_frontier) = match frontier_target {
            Some((row, frontier)) => (Some(row), Some(frontier)),
            None => (None, None),
        };
        let frontier_index = usize::try_from(descriptor.layer_id.0)
            .context("decode KV attention-ready layer exceeds usize")?;
        if frontier_index < cache.mla_attention_ready_frontiers.len() {
            cache.mla_attention_ready_frontiers[frontier_index] = None;
        }
        let normalized_hidden = mla_decode_kv_commit_bf16_device_output(
            descriptor.layer_id.0 as usize,
            hidden,
            input_norm_weight,
            kv_a_weight,
            kv_norm_weight,
            dsa_weights,
            cache_row,
            dsa_cache_row,
            dsa_index_k_cache,
            cache.config().max_tokens,
            attention_ready_row,
            false,
            cache.config().dtype,
            position,
            physical_position,
            GLM52_HIDDEN_SIZE,
            eps,
            theta,
        )
        .context("graphing fused decode KV projection and cache commit")?;
        if let Some(next_frontier) = next_frontier {
            cache.mla_attention_ready_frontiers[frontier_index] = Some(next_frontier);
        }
        self.writes += 1;
        self.bytes += io.payload_bytes;
        Ok(Some(RealFullDeviceMlaDecodeKvCommit {
            writes: vec![io],
            normalized_hidden,
        }))
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(in crate::commands::real_full) fn write_projected_mla_kv_a_and_dsa_key_device_blocks_bf16(
        &mut self,
        descriptors: &[KvBlockDescriptor],
        projected_kv_a_buffer: GlmrtDeviceBuffer,
        dsa_key_buffer: GlmrtDeviceBuffer,
        kv_norm_weight: Option<GlmrtDeviceBuffer>,
    ) -> Result<Option<Vec<RealFullDeviceKvBlockIo>>> {
        if descriptors.is_empty() {
            return Ok(Some(Vec::new()));
        }
        let Some(cache) = self.cache.as_mut() else {
            return Ok(None);
        };
        let layer_id = descriptors[0].layer_id;
        for descriptor in descriptors {
            if descriptor.layer_id != layer_id {
                anyhow::bail!(
                    "device MLA+DSA projected kv_a cache write requires same-layer descriptors, got layer {} and {}",
                    layer_id.0,
                    descriptor.layer_id.0
                );
            }
        }
        if !cache.config().layer_has_dsa_indexer(layer_id) {
            anyhow::bail!(
                "device MLA+DSA projected kv_a cache write requires a DSA/indexer layer, got layer {}",
                layer_id.0
            );
        }
        let main_stride_bytes = (GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM)
            .checked_mul(std::mem::size_of::<u16>())
            .context("device MLA+DSA projected kv_a cache write main stride overflow usize")?;
        let dsa_stride_bytes = GLM52_DSA_INDEX_HEAD_DIM
            .checked_mul(std::mem::size_of::<u16>())
            .context("device MLA+DSA projected kv_a cache write DSA stride overflow usize")?;
        let payload_stride_bytes = main_stride_bytes
            .checked_add(dsa_stride_bytes)
            .context("device MLA+DSA projected kv_a cache write stride overflow usize")?;
        let cache_stride_bytes = cache.config().layer_bytes_per_token(layer_id);
        let rows = descriptors
            .iter()
            .try_fold(0_usize, |acc, descriptor| {
                acc.checked_add(descriptor.token_count)
            })
            .context("device MLA+DSA projected kv_a cache write row count overflow usize")?;
        if rows == 0 {
            anyhow::bail!("device MLA+DSA projected kv_a cache write requires at least one row");
        }
        let main_bytes = rows
            .checked_mul(main_stride_bytes)
            .context("device MLA+DSA projected kv_a cache write main bytes overflow usize")?;
        let dsa_bytes = rows
            .checked_mul(dsa_stride_bytes)
            .context("device MLA+DSA projected kv_a cache write DSA bytes overflow usize")?;
        validate_contiguous_payload_buffer(
            "device MLA+DSA projected kv_a cache write main src",
            projected_kv_a_buffer,
            main_bytes,
        )?;
        validate_contiguous_payload_buffer(
            "device MLA+DSA projected kv_a cache write DSA src",
            dsa_key_buffer,
            dsa_bytes,
        )?;
        if projected_kv_a_buffer.device_id != cache.storage.cache.device_id
            || dsa_key_buffer.device_id != cache.storage.cache.device_id
        {
            anyhow::bail!(
                "device MLA+DSA projected kv_a cache write buffers must be on CUDA device {}",
                cache.storage.cache.device_id
            );
        }
        let prepared_kv_a_buffer = if cache.config().mla_representation
            == MlaKvCacheRepresentation::NormalizedRotated
        {
            Some(
                cache
                    .prepare_projected_mla_kv_for_cache(
                        descriptors,
                        projected_kv_a_buffer,
                        kv_norm_weight.context(
                            "normalized-rotated MLA+DSA KV cache write requires a KV normalization weight",
                        )?,
                        main_stride_bytes,
                    )
                    .context("preparing projected MLA+DSA KV rows for cache storage")?,
            )
        } else {
            None
        };
        let cache_projected_buffer = prepared_kv_a_buffer.unwrap_or(projected_kv_a_buffer);

        let (main_payload, main_row_bytes) = match cache.config().dtype {
            KvCacheDType::Bf16 => {
                if cache_stride_bytes != payload_stride_bytes {
                    anyhow::bail!(
                        "device MLA+DSA projected kv_a cache write stride mismatch for layer {}: expected {} got {}",
                        layer_id.0,
                        payload_stride_bytes,
                        cache_stride_bytes
                    );
                }
                (cache_projected_buffer, main_stride_bytes)
            }
            KvCacheDType::Fp8 => {
                let logical_row_bytes = GLM52_MLA_FP8_DS_BYTES_PER_TOKEN
                    .checked_add(dsa_stride_bytes)
                    .context("device MLA+DSA FP8 payload stride overflow usize")?;
                if cache_stride_bytes != logical_row_bytes {
                    anyhow::bail!(
                        "device MLA+DSA FP8 projected kv_a cache write stride mismatch for layer {}: expected {} got {}",
                        layer_id.0,
                        logical_row_bytes,
                        cache_stride_bytes
                    );
                }
                let packed_bytes = rows
                    .checked_mul(GLM52_MLA_FP8_DS_BYTES_PER_TOKEN)
                    .context("device MLA+DSA FP8 packed main bytes overflow usize")?;
                let packed = cache
                    .mla_fp8_packed_write_payload
                    .buffer(packed_bytes, "device MLA+DSA FP8 packed main payload")?;
                cache
                    .pack_mla_kv_fp8_ds_mla(
                        cache_projected_buffer,
                        packed,
                        rows,
                        main_stride_bytes,
                        GLM52_MLA_FP8_DS_BYTES_PER_TOKEN,
                    )
                    .context("packing device MLA+DSA projected kv_a as FP8 DS payload")?;
                (packed, GLM52_MLA_FP8_DS_BYTES_PER_TOKEN)
            }
            KvCacheDType::Nvfp4 => {
                let logical_row_bytes = GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN
                    .checked_add(dsa_stride_bytes)
                    .context("device MLA+DSA MXFP4 payload stride overflow usize")?;
                if cache_stride_bytes != logical_row_bytes {
                    anyhow::bail!(
                        "device MLA+DSA MXFP4 projected kv_a cache write stride mismatch for layer {}: expected {} got {}",
                        layer_id.0,
                        logical_row_bytes,
                        cache_stride_bytes
                    );
                }
                let packed_bytes = rows
                    .checked_mul(GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN)
                    .context("device MLA+DSA MXFP4 packed main bytes overflow usize")?;
                let packed = cache
                    .mla_mxfp4_packed_write_payload
                    .buffer(packed_bytes, "device MLA+DSA MXFP4 packed main payload")?;
                cache
                    .pack_mla_kv_mxfp4_ds_mla(
                        cache_projected_buffer,
                        packed,
                        rows,
                        main_stride_bytes,
                        GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN,
                    )
                    .context("packing device MLA+DSA projected kv_a as MXFP4 DS payload")?;
                (packed, GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN)
            }
            dtype => {
                anyhow::bail!(
                    "device MLA+DSA projected kv_a cache write requires BF16, FP8, or NVFP4 cache payloads, got {}",
                    dtype.label()
                );
            }
        };

        let writes = cache
            .write_mla_dsa_planes_from_contiguous_device(
                descriptors,
                main_payload,
                main_row_bytes,
                dsa_key_buffer,
            )
            .context("writing direct device MLA and DSA planes to live KV cache")?;
        cache
            .write_dsa_index_k_cache_b12x(descriptors, dsa_key_buffer, dsa_stride_bytes)
            .context("updating direct B12X DSA index-K cache")?;
        self.writes += writes.len();
        self.bytes += writes.iter().map(|io| io.payload_bytes).sum::<usize>();
        Ok(Some(writes))
    }

    pub(in crate::commands::real_full) fn read_visible_blocks(
        &mut self,
        blocks: &[KvBackedBlock],
    ) -> Result<()> {
        if blocks.is_empty() {
            return Ok(());
        }
        let Some(batch) = self.device_kv_batch_plan_for_visible_blocks(blocks)? else {
            return Ok(());
        };
        let Some(reads) = self.read_batch_into_host_readback_scratch(batch)? else {
            return Ok(());
        };
        if reads.len() != blocks.len() {
            anyhow::bail!(
                "live scheduler device KV readback block count mismatch: reads={} blocks={}",
                reads.len(),
                blocks.len()
            );
        }
        let scratch = &self.host_readback_payload_scratch;
        let expected_bytes = blocks
            .iter()
            .try_fold(0_usize, |acc, block| acc.checked_add(block.bytes.len()))
            .context("live scheduler device KV expected read bytes overflow usize")?;
        if scratch.len() != expected_bytes {
            anyhow::bail!(
                "live scheduler device KV readback byte mismatch: expected {expected_bytes} got {}",
                scratch.len()
            );
        }
        let mut offset = 0_usize;
        for (block, io) in blocks.iter().zip(reads.iter()) {
            if io.payload_bytes != block.bytes.len() {
                anyhow::bail!(
                    "live scheduler device KV readback payload size mismatch: descriptor bytes={} block bytes={}",
                    io.payload_bytes,
                    block.bytes.len()
                );
            }
            let end = offset
                .checked_add(io.payload_bytes)
                .context("live scheduler device KV visible read offset overflow")?;
            if scratch[offset..end] != block.bytes {
                anyhow::bail!(
                    "live scheduler device KV readback mismatch: sequence={} layer={} token_start={} token_count={} bytes={}",
                    block.descriptor.sequence_id,
                    block.descriptor.layer_id.0,
                    block.descriptor.token_start.0,
                    block.descriptor.token_count,
                    io.payload_bytes
                );
            }
            offset = end;
        }
        if offset != scratch.len() {
            anyhow::bail!(
                "live scheduler device KV readback split mismatch: consumed {} of {} bytes",
                offset,
                scratch.len()
            );
        }
        Ok(())
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(in crate::commands::real_full) fn read_descriptor_payloads_to_host(
        &mut self,
        descriptors: &[KvBlockDescriptor],
    ) -> Result<Option<Vec<Vec<u8>>>> {
        let Some(batch) = self.device_kv_batch_plan_for_descriptors(descriptors)? else {
            return Ok(None);
        };
        let Some(reads) = self.read_batch_into_host_readback_scratch(batch)? else {
            return Ok(None);
        };
        let actual = &self.host_readback_payload_scratch;
        let mut payloads = Vec::with_capacity(reads.len());
        let mut offset = 0_usize;
        for io in &reads {
            let end = offset
                .checked_add(io.payload_bytes)
                .context("live scheduler device KV read payload offset overflow")?;
            payloads.push(actual[offset..end].to_vec());
            offset = end;
        }
        if offset != actual.len() {
            anyhow::bail!(
                "live scheduler device KV read split mismatch: consumed {} of {} bytes",
                offset,
                actual.len()
            );
        }
        Ok(Some(payloads))
    }

    pub(in crate::commands::real_full) fn snapshot_layer_payload(
        &mut self,
        descriptor: &KvBlockDescriptor,
    ) -> Result<Vec<u8>> {
        let payloads = self
            .read_descriptor_payloads_to_host(slice::from_ref(descriptor))?
            .context("live scheduler device KV cache is unavailable for snapshot")?;
        let mut payloads = payloads.into_iter();
        let payload = payloads
            .next()
            .context("device KV snapshot returned no layer payload")?;
        anyhow::ensure!(
            payloads.next().is_none(),
            "device KV snapshot returned more than one payload for one layer descriptor"
        );
        Ok(payload)
    }

    pub(in crate::commands::real_full) fn restore_layer_payload(
        &mut self,
        descriptor: &KvBlockDescriptor,
        payload: &[u8],
    ) -> Result<()> {
        let cache = self
            .cache
            .as_mut()
            .context("live scheduler device KV cache is unavailable for restore")?;
        let writes = cache
            .write_host_blocks_from_pinned_staging(slice::from_ref(descriptor), &[payload.to_vec()])
            .context("restoring compressed device KV layer payload")?;
        anyhow::ensure!(
            writes.len() == 1,
            "device KV restore produced {} writes for one layer payload",
            writes.len()
        );
        self.writes += 1;
        self.bytes = self
            .bytes
            .checked_add(payload.len())
            .context("device KV restore byte counter overflow")?;
        Ok(())
    }

    pub(in crate::commands::real_full) fn snapshot_dsa_index_prefix(
        &self,
        layer_id: LayerId,
        token_count: usize,
    ) -> Result<Option<Vec<u8>>> {
        let cache = self
            .cache
            .as_ref()
            .context("live scheduler device KV cache is unavailable for DSA snapshot")?;
        let Some(layer) = cache.dsa_index_k_cache_b12x_for_layer(layer_id)? else {
            return Ok(None);
        };
        let pages = token_count
            .checked_add(GLMRT_CUDA_GLM_DSA_PAGE_SIZE - 1)
            .context("DSA snapshot page rounding overflow usize")?
            / GLMRT_CUDA_GLM_DSA_PAGE_SIZE;
        let bytes = pages
            .checked_mul(GLMRT_CUDA_GLM_DSA_PACKED_PAGE_BYTES)
            .context("DSA snapshot byte count overflow usize")?;
        anyhow::ensure!(
            bytes <= layer.bytes,
            "DSA snapshot prefix requires {bytes} bytes but layer has {}",
            layer.bytes
        );
        let mut payload = vec![0_u8; bytes];
        if bytes > 0 {
            cache.library.copy_d2h(&mut payload, layer)?;
        }
        Ok(Some(payload))
    }

    pub(in crate::commands::real_full) fn restore_dsa_index_prefix(
        &mut self,
        layer_id: LayerId,
        token_count: usize,
        payload: &[u8],
    ) -> Result<()> {
        let cache = self
            .cache
            .as_mut()
            .context("live scheduler device KV cache is unavailable for DSA restore")?;
        let layer = cache
            .dsa_index_k_cache_b12x_for_layer(layer_id)?
            .context("DSA restore layer is not an indexer layer")?;
        let pages = token_count
            .checked_add(GLMRT_CUDA_GLM_DSA_PAGE_SIZE - 1)
            .context("DSA restore page rounding overflow usize")?
            / GLMRT_CUDA_GLM_DSA_PAGE_SIZE;
        let expected_bytes = pages
            .checked_mul(GLMRT_CUDA_GLM_DSA_PACKED_PAGE_BYTES)
            .context("DSA restore byte count overflow usize")?;
        anyhow::ensure!(
            payload.len() == expected_bytes,
            "DSA restore payload has {} bytes, expected {expected_bytes}",
            payload.len()
        );
        anyhow::ensure!(
            expected_bytes <= layer.bytes,
            "DSA restore prefix requires {expected_bytes} bytes but layer has {}",
            layer.bytes
        );
        if !payload.is_empty() {
            cache
                .library
                .copy_h2d(layer, payload)
                .context("restoring packed DSA index prefix")?;
        }
        Ok(())
    }

    fn device_kv_batch_plan_for_visible_blocks(
        &self,
        blocks: &[KvBackedBlock],
    ) -> Result<Option<DeviceKvBatchPlan>> {
        if blocks.is_empty() {
            return Ok(Some(DeviceKvBatchPlan::empty()?));
        }
        let Some(cache) = self.cache.as_ref() else {
            return Ok(None);
        };
        DeviceKvBatchPlan::new_from_descriptors(
            cache.config(),
            blocks.iter().map(|block| &block.descriptor),
        )
        .map(Some)
    }

    fn device_kv_batch_plan_for_descriptors(
        &self,
        descriptors: &[KvBlockDescriptor],
    ) -> Result<Option<DeviceKvBatchPlan>> {
        if descriptors.is_empty() {
            return Ok(Some(DeviceKvBatchPlan::empty()?));
        }
        let Some(cache) = self.cache.as_ref() else {
            return Ok(None);
        };
        DeviceKvBatchPlan::new(cache.config(), descriptors).map(Some)
    }

    fn read_batch_into_host_readback_scratch(
        &mut self,
        batch: DeviceKvBatchPlan,
    ) -> Result<Option<Vec<RealFullDeviceKvBlockIo>>> {
        if batch.block_count() == 0 {
            self.host_readback_payload_scratch.clear();
            return Ok(Some(Vec::new()));
        }
        let cache = &mut self.cache;
        let host_readback_payload_scratch = &mut self.host_readback_payload_scratch;
        let mirror_reads = &mut self.reads;
        let mirror_bytes = &mut self.bytes;
        let Some(cache) = cache.as_mut() else {
            return Ok(None);
        };
        let expected_bytes = batch.total_bytes;
        let dst = cache.host_readback_payload.buffer(
            expected_bytes,
            "live scheduler device KV host readback payload",
        )?;
        let reads = cache
            .read_batch_to_contiguous_device(batch, dst)
            .context("reading live scheduler device KV blocks")?;
        host_readback_payload_scratch.resize(expected_bytes, 0);
        cache.library.copy_d2h(host_readback_payload_scratch, dst)?;
        *mirror_reads += reads.len();
        *mirror_bytes += reads.iter().map(|io| io.payload_bytes).sum::<usize>();
        Ok(Some(reads))
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(in crate::commands::real_full) fn read_mla_kv_payloads_to_device_buffers(
        &mut self,
        descriptors: &[KvBlockDescriptor],
    ) -> Result<Option<RealFullDeviceMlaKvDeviceParts>> {
        let Some(cache) = self.cache.as_mut() else {
            return Ok(None);
        };
        let Some(shape) = real_full_device_mla_kv_read_shape(cache.config(), descriptors)? else {
            return Ok(None);
        };
        let payload = cache
            .mla_read_payload
            .buffer(shape.payload_bytes, "device MLA KV compressed read payload")?;
        let reads = cache
            .read_blocks_to_contiguous_device(descriptors, payload)
            .context("reading device MLA KV compressed payload blocks")?;
        let kv_latent = DeviceBufferGuard::new(cache.library, shape.kv_latent_bytes)
            .context("allocating device MLA KV latent output")?;
        let k_rope = DeviceBufferGuard::new(cache.library, shape.k_rope_bytes)
            .context("allocating device MLA KV rope output")?;
        let dsa_key = if shape.dsa_key_bytes > 0 {
            Some(
                DeviceBufferGuard::new(cache.library, shape.dsa_key_bytes)
                    .context("allocating device MLA KV DSA output")?,
            )
        } else {
            None
        };
        let status = unpack_mla_kv_payload_device_buffers_for_shape(
            cache.library,
            &shape,
            payload,
            kv_latent.buffer,
            k_rope.buffer,
            dsa_key.as_ref().map(|guard| guard.buffer),
            &mut cache.mla_fp8_unpacked_projected,
        )
        .context("unpacking device MLA KV compressed payload")?;
        self.reads += reads.len();
        self.bytes += reads.iter().map(|io| io.payload_bytes).sum::<usize>();
        Ok(Some(RealFullDeviceMlaKvDeviceParts {
            status,
            layer_id: shape.layer_id,
            rows: shape.rows,
            payload_bytes: shape.payload_bytes,
            payload_stride_bytes: shape.payload_stride_bytes,
            kv_latent_bytes: shape.kv_latent_bytes,
            k_rope_bytes: shape.k_rope_bytes,
            dsa_key_bytes: shape.dsa_key_bytes,
            kv_latent,
            k_rope,
            dsa_key,
        }))
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(in crate::commands::real_full) fn read_mla_kv_payloads_to_reusable_device_buffers(
        &mut self,
        descriptors: &[KvBlockDescriptor],
    ) -> Result<Option<RealFullDeviceMlaKvDeviceBufferView>> {
        let Some(cache) = self.cache.as_mut() else {
            return Ok(None);
        };
        let Some(shape) = real_full_device_mla_kv_read_shape(cache.config(), descriptors)? else {
            return Ok(None);
        };
        let direct_span = if cache.config.layer_has_dsa_indexer(shape.layer_id) {
            None
        } else {
            cache.contiguous_physical_kv_block_span(descriptors)?
        };
        let (payload, reads) = if let Some((offset, bytes, reads)) = direct_span {
            if bytes != shape.payload_bytes {
                anyhow::bail!(
                    "contiguous device MLA KV span has {bytes} bytes, expected {}",
                    shape.payload_bytes
                );
            }
            (
                device_buffer_byte_view(
                    cache.storage.cache,
                    offset,
                    bytes,
                    "direct contiguous device MLA KV cache span",
                )?,
                reads,
            )
        } else {
            let payload = cache
                .mla_read_payload
                .buffer(shape.payload_bytes, "device MLA KV compressed read payload")?;
            let reads = cache
                .read_blocks_to_contiguous_device(descriptors, payload)
                .context("reading device MLA KV compressed payload blocks")?;
            (payload, reads)
        };
        let kv_latent = cache
            .mla_unpacked_kv_latent
            .buffer(
                shape.kv_latent_bytes,
                "device MLA KV reusable latent output",
            )
            .context("allocating reusable device MLA KV latent output")?;
        let k_rope = cache
            .mla_unpacked_k_rope
            .buffer(shape.k_rope_bytes, "device MLA KV reusable rope output")
            .context("allocating reusable device MLA KV rope output")?;
        let dsa_key = if shape.dsa_key_bytes > 0 {
            Some(
                cache
                    .mla_unpacked_dsa_key
                    .buffer(shape.dsa_key_bytes, "device MLA KV reusable DSA output")
                    .context("allocating reusable device MLA KV DSA output")?,
            )
        } else {
            None
        };
        if payload.device_id != kv_latent.device_id || k_rope.device_id != kv_latent.device_id {
            anyhow::bail!("device MLA KV reusable unpack buffers must be on the same CUDA device");
        }
        if let Some(dsa_key) = dsa_key {
            if dsa_key.device_id != kv_latent.device_id {
                anyhow::bail!(
                    "device MLA KV reusable DSA output is on CUDA device {}, but KV outputs are on device {}",
                    dsa_key.device_id,
                    kv_latent.device_id
                );
            }
        }
        unpack_mla_kv_payload_device_buffers_for_shape(
            cache.library,
            &shape,
            payload,
            kv_latent,
            k_rope,
            dsa_key,
            &mut cache.mla_fp8_unpacked_projected,
        )
        .context("unpacking device MLA KV compressed payload into reusable buffers")?;
        self.reads += reads.len();
        self.bytes += reads.iter().map(|io| io.payload_bytes).sum::<usize>();
        Ok(Some(RealFullDeviceMlaKvDeviceBufferView {
            rows: shape.rows,
            kv_latent_bytes: shape.kv_latent_bytes,
            k_rope_bytes: shape.k_rope_bytes,
            kv_latent,
            k_rope,
        }))
    }

    fn direct_attention_ready_mla_kv_span(
        &mut self,
        descriptors: &[KvBlockDescriptor],
    ) -> Result<Option<RealFullDeviceMlaKvDirectSpan>> {
        let Some(cache) = self.cache.as_mut() else {
            return Ok(None);
        };
        if cache.config().mla_representation != MlaKvCacheRepresentation::NormalizedRotated
            || !matches!(
                cache.config().dtype,
                KvCacheDType::Bf16 | KvCacheDType::Fp8 | KvCacheDType::Nvfp4
            )
        {
            return Ok(None);
        }
        // The packed FP8/NVFP4 sparse kernels consume a request page table
        // directly. The BF16 compressed-MLA kernel instead consumes a
        // contiguous BF16 latent+RoPE view. For a radix-owned paged request,
        // fall back to the reusable gather/split path below rather than
        // binding the full layer plane and trying to stage logical rows as if
        // they were physically contiguous.
        if cache.config().dtype == KvCacheDType::Bf16 && cache.physical_page_table().is_some() {
            return Ok(None);
        }
        let Some(shape) = real_full_device_mla_kv_read_shape(cache.config(), descriptors)? else {
            return Ok(None);
        };
        if cache.config().dtype == KvCacheDType::Nvfp4
            && shape.rows <= REAL_FULL_PACKED_FP8_MLA_MAX_ROWS
        {
            if let Some((payload, frontier_rows, _)) =
                cache.attention_ready_mla_frontier_payload_for_descriptors(descriptors)?
            {
                anyhow::ensure!(
                    frontier_rows == shape.rows,
                    "attention-ready FP8 frontier has {frontier_rows} rows, expected {}",
                    shape.rows
                );
                let logical_reads = real_full_device_kv_block_ios(cache.config(), descriptors)?;
                self.reads += logical_reads.len();
                self.bytes += logical_reads
                    .iter()
                    .map(|io| io.payload_bytes)
                    .sum::<usize>();
                return Ok(Some(RealFullDeviceMlaKvDirectSpan {
                    rows: shape.rows,
                    row_offset: 0,
                    dtype: KvCacheDType::Fp8,
                    row_stride_bytes: GLM52_MLA_FP8_DS_BYTES_PER_TOKEN,
                    payload,
                    physical_page_table: None,
                    // A compact frontier is request-local. Keeping the
                    // one-row O output in graph-slot storage avoids retaining
                    // a graph identity for every request-buffer rotation.
                    force_staged_hidden_projection: true,
                }));
            }
        }
        let main_row_bytes = real_full_device_main_mla_row_bytes(cache.config())?;
        let expected_main_bytes = shape
            .rows
            .checked_mul(main_row_bytes)
            .context("direct attention-ready MLA main span bytes overflow usize")?;
        let layer_base = cache
            .config()
            .layer_base_offset_bytes(shape.layer_id)
            .context("direct attention-ready MLA layer base is invalid")?;
        let layer_bytes = main_row_bytes
            .checked_mul(cache.config().max_tokens)
            .context("direct attention-ready MLA layer bytes overflow usize")?;
        let physical_page_table =
            cache
                .physical_page_table()
                .map(
                    |(physical_pages, mapping_key)| FlashinferTargetKvPageTable {
                        physical_pages,
                        mapping_key,
                    },
                );
        let row_offset = if physical_page_table.is_some() {
            let mut expected_start = 0_u64;
            for descriptor in descriptors {
                anyhow::ensure!(
                    descriptor.token_start.0 == expected_start,
                    "paged direct MLA attention requires a complete logical prefix; expected token {expected_start}, got {}",
                    descriptor.token_start.0
                );
                expected_start = expected_start
                    .checked_add(
                        u64::try_from(descriptor.token_count)
                            .context("paged direct MLA descriptor length does not fit u64")?,
                    )
                    .context("paged direct MLA logical prefix length overflow")?;
            }
            anyhow::ensure!(
                usize::try_from(expected_start).ok() == Some(shape.rows),
                "paged direct MLA logical prefix contains {expected_start} rows, expected {}",
                shape.rows
            );
            0
        } else {
            let contiguous = cache.contiguous_physical_main_kv_block_span(descriptors)?;
            let Some((offset, bytes, _main_reads)) = contiguous else {
                if cache.config().dtype == KvCacheDType::Fp8
                    && shape.rows <= REAL_FULL_PACKED_FP8_MLA_MAX_ROWS
                {
                    let payload = cache
                        .mla_read_payload
                        .buffer(expected_main_bytes, "gathered packed FP8 MLA main KV")?;
                    let logical_reads =
                        cache.read_main_blocks_to_contiguous_device(descriptors, payload)?;
                    self.reads += logical_reads.len();
                    self.bytes += logical_reads
                        .iter()
                        .map(|io| io.payload_bytes)
                        .sum::<usize>();
                    return Ok(Some(RealFullDeviceMlaKvDirectSpan {
                        rows: shape.rows,
                        row_offset: 0,
                        dtype: KvCacheDType::Fp8,
                        row_stride_bytes: main_row_bytes,
                        payload,
                        physical_page_table: None,
                        // This request-local gather may rotate. The packed
                        // kernel stages it into its graph-owned workspace.
                        force_staged_hidden_projection: true,
                    }));
                }
                return Ok(None);
            };
            anyhow::ensure!(
                bytes == expected_main_bytes,
                "direct attention-ready MLA main KV span has {bytes} bytes, expected {expected_main_bytes}",
            );
            let relative_offset = offset
                .checked_sub(layer_base)
                .context("direct attention-ready MLA span starts before its main cache plane")?;
            anyhow::ensure!(
                relative_offset % main_row_bytes == 0,
                "direct attention-ready MLA span offset {relative_offset} is not row aligned"
            );
            relative_offset / main_row_bytes
        };
        anyhow::ensure!(
            row_offset
                .checked_add(shape.rows)
                .is_some_and(|end| end <= cache.config().max_tokens),
            "direct attention-ready MLA visible rows exceed their main cache plane"
        );
        // Bind bucketed attention to the graph-stable full layer plane. The
        // reservation's physical row offset is carried separately and turned
        // into the kernel's selected slot IDs inside the captured graph.
        let payload = device_buffer_byte_view(
            cache.storage.cache,
            layer_base,
            layer_bytes,
            "direct attention-ready MLA KV cache layer plane",
        )?;
        let logical_reads = real_full_device_kv_block_ios(cache.config(), descriptors)?;
        self.reads += logical_reads.len();
        self.bytes += logical_reads
            .iter()
            .map(|io| io.payload_bytes)
            .sum::<usize>();
        Ok(Some(RealFullDeviceMlaKvDirectSpan {
            rows: shape.rows,
            row_offset,
            dtype: shape.dtype,
            row_stride_bytes: main_row_bytes,
            payload,
            physical_page_table,
            force_staged_hidden_projection: false,
        }))
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(in crate::commands::real_full) fn read_mla_kv_payloads_to_device_parts(
        &mut self,
        descriptors: &[KvBlockDescriptor],
    ) -> Result<Option<RealFullDeviceMlaKvUnpackReadback>> {
        if descriptors.is_empty() {
            return Ok(Some(RealFullDeviceMlaKvUnpackReadback {
                status: "cuda-kv-cache-mla-kv-unpack-empty",
                rows: 0,
                payload_bytes: 0,
                kv_latent_bf16: Vec::new(),
                k_rope_bf16: Vec::new(),
                dsa_key_bf16: None,
            }));
        }
        let Some(parts) = self.read_mla_kv_payloads_to_device_buffers(descriptors)? else {
            return Ok(None);
        };
        parts.copy_to_host().map(Some)
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(in crate::commands::real_full) fn run_mla_rope_attention_from_device_prefix_with_host_suffix_bf16(
        &mut self,
        descriptors: &[KvBlockDescriptor],
        prefix_positions: &[u32],
        kv_norm_weight_bf16: &[u8],
        kv_b_weight_bf16: &[u8],
        q_nope_bf16: &[u8],
        q_rope_bf16: &[u8],
        suffix_k_nope_bf16: &[u8],
        suffix_k_rope_rotated_bf16: &[u8],
        suffix_values_bf16: &[u8],
        heads: usize,
        nope_dim: usize,
        v_dim: usize,
        eps: f32,
        theta: f32,
        scale: f32,
    ) -> Result<Option<RealFullDeviceMlaAttentionReadback>> {
        let kv_norm_weight = self
            .cache
            .as_mut()
            .context("device MLA prefix attention cache unavailable for host weight upload")?
            .upload_device_bytes_from_pinned_staging(
                kv_norm_weight_bf16,
                "device MLA prefix attention kv norm weight",
            )?;
        let kv_b_weight = self
            .cache
            .as_mut()
            .context("device MLA prefix attention cache unavailable for host weight upload")?
            .upload_device_bytes_from_pinned_staging(
                kv_b_weight_bf16,
                "device MLA prefix attention kv_b weight",
            )?;
        self.run_mla_rope_attention_from_device_prefix_with_device_weights_and_host_suffix_bf16(
            descriptors,
            prefix_positions,
            kv_norm_weight.buffer,
            kv_b_weight.buffer,
            q_nope_bf16,
            q_rope_bf16,
            suffix_k_nope_bf16,
            suffix_k_rope_rotated_bf16,
            suffix_values_bf16,
            heads,
            nope_dim,
            v_dim,
            eps,
            theta,
            scale,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(in crate::commands::real_full) fn run_mla_rope_attention_from_device_prefix_with_device_weights_and_host_suffix_bf16(
        &mut self,
        descriptors: &[KvBlockDescriptor],
        prefix_positions: &[u32],
        kv_norm_weight: GlmrtDeviceBuffer,
        kv_b_weight: GlmrtDeviceBuffer,
        q_nope_bf16: &[u8],
        q_rope_bf16: &[u8],
        suffix_k_nope_bf16: &[u8],
        suffix_k_rope_rotated_bf16: &[u8],
        suffix_values_bf16: &[u8],
        heads: usize,
        nope_dim: usize,
        v_dim: usize,
        eps: f32,
        theta: f32,
        scale: f32,
    ) -> Result<Option<RealFullDeviceMlaAttentionReadback>> {
        let Some(attention) = self
            .run_mla_rope_attention_parts_from_device_prefix_with_device_weights_and_host_suffix_bf16(
                descriptors,
                prefix_positions,
                kv_norm_weight,
                kv_b_weight,
                q_nope_bf16,
                q_rope_bf16,
                suffix_k_nope_bf16,
                suffix_k_rope_rotated_bf16,
                suffix_values_bf16,
                heads,
                nope_dim,
                v_dim,
                eps,
                theta,
                scale,
            )?
        else {
            return Ok(None);
        };
        attention.copy_to_host().map(Some)
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(in crate::commands::real_full) fn run_mla_rope_attention_parts_from_device_prefix_with_device_weights_and_host_suffix_bf16(
        &mut self,
        descriptors: &[KvBlockDescriptor],
        prefix_positions: &[u32],
        kv_norm_weight: GlmrtDeviceBuffer,
        kv_b_weight: GlmrtDeviceBuffer,
        q_nope_bf16: &[u8],
        q_rope_bf16: &[u8],
        suffix_k_nope_bf16: &[u8],
        suffix_k_rope_rotated_bf16: &[u8],
        suffix_values_bf16: &[u8],
        heads: usize,
        nope_dim: usize,
        v_dim: usize,
        eps: f32,
        theta: f32,
        scale: f32,
    ) -> Result<Option<RealFullDeviceMlaAttentionParts>> {
        if descriptors.is_empty() {
            return Ok(None);
        }
        let Some(prefix_parts) =
            self.read_mla_kv_payloads_to_reusable_device_buffers(descriptors)?
        else {
            return Ok(None);
        };
        let library = self
            .cache
            .as_ref()
            .context("device MLA prefix attention cache unavailable for reusable prefix KV")?
            .library;
        let kv_norm_weight_bytes = GLM52_MLA_KV_LORA_RANK
            .checked_mul(std::mem::size_of::<u16>())
            .context("device MLA prefix attention kv norm weight bytes overflow usize")?;
        let kv_b_output_dim = nope_dim
            .checked_add(v_dim)
            .context("device MLA prefix attention kv_b output dim overflow usize")?;
        let kv_b_weight_bytes = heads
            .checked_mul(kv_b_output_dim)
            .and_then(|values| values.checked_mul(GLM52_MLA_KV_LORA_RANK))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA prefix attention kv_b weight bytes overflow usize")?;
        validate_contiguous_payload_buffer(
            "device MLA prefix attention kv norm resident weight",
            kv_norm_weight,
            kv_norm_weight_bytes,
        )?;
        validate_contiguous_payload_buffer(
            "device MLA prefix attention kv_b resident weight",
            kv_b_weight,
            kv_b_weight_bytes,
        )?;
        if kv_norm_weight.device_id != prefix_parts.kv_latent.device_id
            || kv_b_weight.device_id != prefix_parts.kv_latent.device_id
        {
            anyhow::bail!(
                "device MLA prefix attention resident weights must be on the same CUDA device as prefix KV"
            );
        }
        let (q_nope, q_rope, suffix_k_nope, suffix_k_rope, suffix_values, prefix_positions_device) = {
            let cache = self
                .cache
                .as_mut()
                .context("device MLA prefix attention cache unavailable for host uploads")?;
            let q_nope = cache.stage_attention_host_slice_bytes(
                DeviceKvAttentionHostUploadSlot::QueryNope,
                q_nope_bf16,
                "device MLA prefix attention q_nope",
            )?;
            let q_rope = cache.stage_attention_host_slice_bytes(
                DeviceKvAttentionHostUploadSlot::QueryRope,
                q_rope_bf16,
                "device MLA prefix attention q_rope",
            )?;
            let suffix_k_nope = cache.stage_attention_host_slice_bytes(
                DeviceKvAttentionHostUploadSlot::SuffixKNope,
                suffix_k_nope_bf16,
                "device MLA prefix attention suffix k_nope",
            )?;
            let suffix_k_rope = cache.stage_attention_host_slice_bytes(
                DeviceKvAttentionHostUploadSlot::SuffixKRope,
                suffix_k_rope_rotated_bf16,
                "device MLA prefix attention suffix k_rope",
            )?;
            let suffix_values = cache.stage_attention_host_slice_bytes(
                DeviceKvAttentionHostUploadSlot::SuffixValues,
                suffix_values_bf16,
                "device MLA prefix attention suffix values",
            )?;
            let prefix_positions_device = cache.stage_rope_positions_u32(
                prefix_positions,
                "device MLA prefix attention prefix rows",
            )?;
            (
                q_nope,
                q_rope,
                suffix_k_nope,
                suffix_k_rope,
                suffix_values,
                prefix_positions_device,
            )
        };

        let projected = {
            let cache = self.cache.as_mut().context(
                "device MLA prefix attention cache unavailable for projected KV buffers",
            )?;
            cache.project_mla_kv_latent_and_split_to_reusable_buffers(
                descriptors[0].layer_id,
                &prefix_parts,
                kv_norm_weight,
                kv_b_weight,
                heads,
                nope_dim,
                v_dim,
                false,
                eps,
            )?
        };
        let rotated = {
            let cache = self
                .cache
                .as_mut()
                .context("device MLA prefix attention cache unavailable for prefix RoPE buffer")?;
            cache.rotate_mla_k_rope_to_reusable_buffer(
                descriptors[0].layer_id,
                &prefix_parts,
                prefix_positions,
                prefix_positions_device,
                theta,
            )?
        };
        let attention = {
            let cache = self
                .cache
                .as_mut()
                .context("device MLA prefix attention cache unavailable for combined buffers")?;
            projected.run_mla_rope_attention_with_uploaded_suffix_bf16(
                library,
                cache,
                &rotated,
                q_nope,
                q_rope,
                suffix_k_nope,
                suffix_k_nope_bf16.len(),
                suffix_k_rope,
                suffix_k_rope_rotated_bf16.len(),
                suffix_values,
                suffix_values_bf16.len(),
                scale,
            )?
        };
        Ok(Some(attention))
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::commands::real_full) fn run_mla_rope_attention_parts_from_device_prefix_with_device_weights_and_projected_query_host_suffix_bf16(
        &mut self,
        descriptors: &[KvBlockDescriptor],
        prefix_positions: &[u32],
        kv_norm_weight: GlmrtDeviceBuffer,
        kv_b_weight: GlmrtDeviceBuffer,
        q_projected_bf16: &[u8],
        q_suffix_positions: &[u32],
        suffix_k_nope_bf16: &[u8],
        suffix_k_rope_rotated_bf16: &[u8],
        suffix_values_bf16: &[u8],
        heads: usize,
        nope_dim: usize,
        v_dim: usize,
        eps: f32,
        theta: f32,
        scale: f32,
    ) -> Result<Option<RealFullDeviceMlaAttentionParts>> {
        if descriptors.is_empty() {
            return Ok(None);
        }
        let prefix_rows = descriptors
            .iter()
            .map(|descriptor| descriptor.token_count)
            .sum::<usize>();
        if prefix_rows != prefix_positions.len() {
            anyhow::bail!(
                "device MLA prefix attention projected query prefix row mismatch: descriptors={} positions={}",
                prefix_rows,
                prefix_positions.len()
            );
        }
        let Some(prefix_parts) =
            self.read_mla_kv_payloads_to_reusable_device_buffers(descriptors)?
        else {
            return Ok(None);
        };
        let library = self
            .cache
            .as_ref()
            .context(
                "device MLA prefix projected-query attention cache unavailable for reusable prefix KV",
            )?
            .library;
        let kv_norm_weight_bytes = GLM52_MLA_KV_LORA_RANK
            .checked_mul(std::mem::size_of::<u16>())
            .context(
                "device MLA prefix projected-query attention kv norm weight bytes overflow usize",
            )?;
        let kv_b_output_dim = nope_dim.checked_add(v_dim).context(
            "device MLA prefix projected-query attention kv_b output dim overflow usize",
        )?;
        let kv_b_weight_bytes = heads
            .checked_mul(kv_b_output_dim)
            .and_then(|values| values.checked_mul(GLM52_MLA_KV_LORA_RANK))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context(
                "device MLA prefix projected-query attention kv_b weight bytes overflow usize",
            )?;
        validate_contiguous_payload_buffer(
            "device MLA prefix projected-query attention kv norm resident weight",
            kv_norm_weight,
            kv_norm_weight_bytes,
        )?;
        validate_contiguous_payload_buffer(
            "device MLA prefix projected-query attention kv_b resident weight",
            kv_b_weight,
            kv_b_weight_bytes,
        )?;
        if kv_norm_weight.device_id != prefix_parts.kv_latent.device_id
            || kv_b_weight.device_id != prefix_parts.kv_latent.device_id
        {
            anyhow::bail!(
                "device MLA prefix projected-query attention resident weights must be on the same CUDA device as prefix KV"
            );
        }

        let queries = {
            let cache = self
                .cache
                .as_mut()
                .context("device MLA prefix projected-query attention cache unavailable")?;
            let q_projected = cache.stage_attention_host_slice_bytes(
                DeviceKvAttentionHostUploadSlot::ProjectedQuery,
                q_projected_bf16,
                "device MLA prefix projected-query attention q_b",
            )?;
            let q_positions_device = cache.stage_rope_positions_u32(
                q_suffix_positions,
                "device MLA prefix projected-query attention query suffix",
            )?;
            cache.split_projected_query_suffix_to_reusable_buffers(
                descriptors[0].layer_id,
                q_projected,
                prefix_rows,
                q_suffix_positions,
                Some(q_positions_device),
                heads,
                nope_dim,
                GLM52_MLA_QK_ROPE_HEAD_DIM,
                theta,
                false,
            )?
        };
        let (suffix_k_nope, suffix_k_rope, suffix_values, prefix_positions_device) = {
            let cache = self
                .cache
                .as_mut()
                .context("device MLA prefix projected-query attention cache unavailable")?;
            let suffix_k_nope = cache.stage_attention_host_slice_bytes(
                DeviceKvAttentionHostUploadSlot::SuffixKNope,
                suffix_k_nope_bf16,
                "device MLA prefix projected-query attention suffix k_nope",
            )?;
            let suffix_k_rope = cache.stage_attention_host_slice_bytes(
                DeviceKvAttentionHostUploadSlot::SuffixKRope,
                suffix_k_rope_rotated_bf16,
                "device MLA prefix projected-query attention suffix k_rope",
            )?;
            let suffix_values = cache.stage_attention_host_slice_bytes(
                DeviceKvAttentionHostUploadSlot::SuffixValues,
                suffix_values_bf16,
                "device MLA prefix projected-query attention suffix values",
            )?;
            let prefix_positions_device = cache.stage_rope_positions_u32(
                prefix_positions,
                "device MLA prefix projected-query attention prefix rows",
            )?;
            (
                suffix_k_nope,
                suffix_k_rope,
                suffix_values,
                prefix_positions_device,
            )
        };
        let projected = {
            let cache = self.cache.as_mut().context(
                "device MLA prefix projected-query attention cache unavailable for projected KV buffers",
            )?;
            cache.project_mla_kv_latent_and_split_to_reusable_buffers(
                descriptors[0].layer_id,
                &prefix_parts,
                kv_norm_weight,
                kv_b_weight,
                heads,
                nope_dim,
                v_dim,
                false,
                eps,
            )?
        };
        let rotated = {
            let cache = self.cache.as_mut().context(
                "device MLA prefix projected-query attention cache unavailable for prefix RoPE buffer",
            )?;
            cache.rotate_mla_k_rope_to_reusable_buffer(
                descriptors[0].layer_id,
                &prefix_parts,
                prefix_positions,
                prefix_positions_device,
                theta,
            )?
        };
        let attention = {
            let cache = self.cache.as_mut().context(
                "device MLA prefix projected-query attention cache unavailable for combined buffers",
            )?;
            projected.run_mla_rope_attention_with_uploaded_suffix_bf16(
                library,
                cache,
                &rotated,
                queries.q_nope_buffer(),
                queries.q_rope_rotated_buffer(),
                suffix_k_nope,
                suffix_k_nope_bf16.len(),
                suffix_k_rope,
                suffix_k_rope_rotated_bf16.len(),
                suffix_values,
                suffix_values_bf16.len(),
                scale,
            )?
        };
        Ok(Some(attention))
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(in crate::commands::real_full) fn run_mla_rope_attention_parts_from_device_prefix_with_device_weights_and_projected_query_device_suffix_bf16(
        &mut self,
        descriptors: &[KvBlockDescriptor],
        prefix_positions: &[u32],
        kv_norm_weight: GlmrtDeviceBuffer,
        kv_b_weight: GlmrtDeviceBuffer,
        q_projected_buffer: GlmrtDeviceBuffer,
        q_suffix_positions: &[u32],
        suffix_k_nope_bf16: &[u8],
        suffix_k_rope_rotated_bf16: &[u8],
        suffix_values_bf16: &[u8],
        heads: usize,
        nope_dim: usize,
        v_dim: usize,
        eps: f32,
        theta: f32,
        scale: f32,
    ) -> Result<Option<RealFullDeviceMlaAttentionParts>> {
        if descriptors.is_empty() {
            return Ok(None);
        }
        let prefix_rows = descriptors
            .iter()
            .map(|descriptor| descriptor.token_count)
            .sum::<usize>();
        if prefix_rows != prefix_positions.len() {
            anyhow::bail!(
                "device MLA prefix attention projected query prefix row mismatch: descriptors={} positions={}",
                prefix_rows,
                prefix_positions.len()
            );
        }
        let Some(prefix_parts) =
            self.read_mla_kv_payloads_to_reusable_device_buffers(descriptors)?
        else {
            return Ok(None);
        };
        let library = self
            .cache
            .as_ref()
            .context(
                "device MLA prefix projected-query attention cache unavailable for reusable prefix KV",
            )?
            .library;
        let kv_norm_weight_bytes = GLM52_MLA_KV_LORA_RANK
            .checked_mul(std::mem::size_of::<u16>())
            .context(
                "device MLA prefix projected-query attention kv norm weight bytes overflow usize",
            )?;
        let kv_b_output_dim = nope_dim.checked_add(v_dim).context(
            "device MLA prefix projected-query attention kv_b output dim overflow usize",
        )?;
        let kv_b_weight_bytes = heads
            .checked_mul(kv_b_output_dim)
            .and_then(|values| values.checked_mul(GLM52_MLA_KV_LORA_RANK))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context(
                "device MLA prefix projected-query attention kv_b weight bytes overflow usize",
            )?;
        validate_contiguous_payload_buffer(
            "device MLA prefix projected-query attention kv norm resident weight",
            kv_norm_weight,
            kv_norm_weight_bytes,
        )?;
        validate_contiguous_payload_buffer(
            "device MLA prefix projected-query attention kv_b resident weight",
            kv_b_weight,
            kv_b_weight_bytes,
        )?;
        if kv_norm_weight.device_id != prefix_parts.kv_latent.device_id
            || kv_b_weight.device_id != prefix_parts.kv_latent.device_id
            || q_projected_buffer.device_id != prefix_parts.kv_latent.device_id
        {
            anyhow::bail!(
                "device MLA prefix projected-query attention resident weights and query buffer must be on the same CUDA device as prefix KV"
            );
        }

        let queries = {
            let cache = self
                .cache
                .as_mut()
                .context("device MLA prefix projected-query attention cache unavailable")?;
            let q_positions_device = cache.stage_rope_positions_u32(
                q_suffix_positions,
                "device MLA prefix projected-query attention query suffix",
            )?;
            cache.split_projected_query_suffix_to_reusable_buffers(
                descriptors[0].layer_id,
                q_projected_buffer,
                prefix_rows,
                q_suffix_positions,
                Some(q_positions_device),
                heads,
                nope_dim,
                GLM52_MLA_QK_ROPE_HEAD_DIM,
                theta,
                false,
            )?
        };
        let (suffix_k_nope, suffix_k_rope, suffix_values, prefix_positions_device) = {
            let cache = self
                .cache
                .as_mut()
                .context("device MLA prefix projected-query attention cache unavailable")?;
            let suffix_k_nope = cache.stage_attention_host_slice_bytes(
                DeviceKvAttentionHostUploadSlot::SuffixKNope,
                suffix_k_nope_bf16,
                "device MLA prefix projected-query attention suffix k_nope",
            )?;
            let suffix_k_rope = cache.stage_attention_host_slice_bytes(
                DeviceKvAttentionHostUploadSlot::SuffixKRope,
                suffix_k_rope_rotated_bf16,
                "device MLA prefix projected-query attention suffix k_rope",
            )?;
            let suffix_values = cache.stage_attention_host_slice_bytes(
                DeviceKvAttentionHostUploadSlot::SuffixValues,
                suffix_values_bf16,
                "device MLA prefix projected-query attention suffix values",
            )?;
            let prefix_positions_device = cache.stage_rope_positions_u32(
                prefix_positions,
                "device MLA prefix projected-query attention prefix rows",
            )?;
            (
                suffix_k_nope,
                suffix_k_rope,
                suffix_values,
                prefix_positions_device,
            )
        };
        let projected = {
            let cache = self.cache.as_mut().context(
                "device MLA prefix projected-query device-buffer attention cache unavailable for projected KV buffers",
            )?;
            cache.project_mla_kv_latent_and_split_to_reusable_buffers(
                descriptors[0].layer_id,
                &prefix_parts,
                kv_norm_weight,
                kv_b_weight,
                heads,
                nope_dim,
                v_dim,
                false,
                eps,
            )?
        };
        let rotated = {
            let cache = self.cache.as_mut().context(
                "device MLA prefix projected-query device-buffer attention cache unavailable for prefix RoPE buffer",
            )?;
            cache.rotate_mla_k_rope_to_reusable_buffer(
                descriptors[0].layer_id,
                &prefix_parts,
                prefix_positions,
                prefix_positions_device,
                theta,
            )?
        };
        let attention = {
            let cache = self.cache.as_mut().context(
                "device MLA prefix projected-query device-buffer attention cache unavailable for combined buffers",
            )?;
            projected.run_mla_rope_attention_with_uploaded_suffix_bf16(
                library,
                cache,
                &rotated,
                queries.q_nope_buffer(),
                queries.q_rope_rotated_buffer(),
                suffix_k_nope,
                suffix_k_nope_bf16.len(),
                suffix_k_rope,
                suffix_k_rope_rotated_bf16.len(),
                suffix_values,
                suffix_values_bf16.len(),
                scale,
            )?
        };
        Ok(Some(attention))
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::commands::real_full) fn run_mla_rope_attention_parts_from_device_prefix_with_device_weights_and_projected_query_device_kv_suffix_bf16(
        &mut self,
        descriptors: &[KvBlockDescriptor],
        prefix_positions: &[u32],
        kv_norm_weight: GlmrtDeviceBuffer,
        kv_b_weight: GlmrtDeviceBuffer,
        q_projected_buffer: GlmrtDeviceBuffer,
        kv_a_projected_buffer: GlmrtDeviceBuffer,
        suffix_positions: &[u32],
        heads: usize,
        nope_dim: usize,
        v_dim: usize,
        eps: f32,
        theta: f32,
        scale: f32,
    ) -> Result<Option<RealFullDeviceMlaAttentionParts>> {
        if descriptors.is_empty() {
            return Ok(None);
        }
        if suffix_positions.is_empty() {
            anyhow::bail!("device MLA prefix attention device suffix requires suffix positions");
        }
        let prefix_rows = descriptors
            .iter()
            .map(|descriptor| descriptor.token_count)
            .sum::<usize>();
        if prefix_rows != prefix_positions.len() {
            anyhow::bail!(
                "device MLA prefix attention device suffix prefix row mismatch: descriptors={} positions={}",
                prefix_rows,
                prefix_positions.len()
            );
        }
        let Some(prefix_parts) =
            self.read_mla_kv_payloads_to_reusable_device_buffers(descriptors)?
        else {
            return Ok(None);
        };
        let library = self
            .cache
            .as_ref()
            .context(
                "device MLA prefix projected-query device suffix cache unavailable for reusable prefix KV",
            )?
            .library;
        let kv_norm_weight_bytes = GLM52_MLA_KV_LORA_RANK
            .checked_mul(std::mem::size_of::<u16>())
            .context(
                "device MLA prefix projected-query device suffix kv norm weight bytes overflow usize",
            )?;
        let kv_b_output_dim = nope_dim.checked_add(v_dim).context(
            "device MLA prefix projected-query device suffix kv_b output dim overflow usize",
        )?;
        let kv_b_weight_bytes = heads
            .checked_mul(kv_b_output_dim)
            .and_then(|values| values.checked_mul(GLM52_MLA_KV_LORA_RANK))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context(
                "device MLA prefix projected-query device suffix kv_b weight bytes overflow usize",
            )?;
        validate_contiguous_payload_buffer(
            "device MLA prefix projected-query device suffix kv norm resident weight",
            kv_norm_weight,
            kv_norm_weight_bytes,
        )?;
        validate_contiguous_payload_buffer(
            "device MLA prefix projected-query device suffix kv_b resident weight",
            kv_b_weight,
            kv_b_weight_bytes,
        )?;
        if kv_norm_weight.device_id != prefix_parts.kv_latent.device_id
            || kv_b_weight.device_id != prefix_parts.kv_latent.device_id
            || q_projected_buffer.device_id != prefix_parts.kv_latent.device_id
            || kv_a_projected_buffer.device_id != prefix_parts.kv_latent.device_id
        {
            anyhow::bail!(
                "device MLA prefix projected-query device suffix resident weights, query buffer, and KV suffix must be on the same CUDA device as prefix KV"
            );
        }

        let queries = {
            let cache = self
                .cache
                .as_mut()
                .context("device MLA prefix projected-query device suffix cache unavailable")?;
            let query_positions_device = cache.stage_rope_positions_u32(
                suffix_positions,
                "device MLA prefix projected-query device suffix query rows",
            )?;
            cache.split_projected_query_suffix_to_reusable_buffers(
                descriptors[0].layer_id,
                q_projected_buffer,
                prefix_rows,
                suffix_positions,
                Some(query_positions_device),
                heads,
                nope_dim,
                GLM52_MLA_QK_ROPE_HEAD_DIM,
                theta,
                false,
            )?
        };
        let prefix_projected = {
            let cache = self.cache.as_mut().context(
                "device MLA prefix projected-query device suffix cache unavailable for prefix projected KV buffers",
            )?;
            cache.project_mla_kv_latent_and_split_to_reusable_buffers(
                descriptors[0].layer_id,
                &prefix_parts,
                kv_norm_weight,
                kv_b_weight,
                heads,
                nope_dim,
                v_dim,
                false,
                eps,
            )?
        };
        let prefix_positions_device = self
            .cache
            .as_mut()
            .context("device MLA prefix projected-query device suffix cache unavailable")?
            .stage_rope_positions_u32(
                prefix_positions,
                "device MLA prefix projected-query device suffix prefix rows",
            )?;
        let prefix_rotated = {
            let cache = self.cache.as_mut().context(
                "device MLA prefix projected-query device suffix cache unavailable for prefix RoPE buffer",
            )?;
            cache.rotate_mla_k_rope_to_reusable_buffer(
                descriptors[0].layer_id,
                &prefix_parts,
                prefix_positions,
                prefix_positions_device,
                theta,
            )?
        };
        let suffix_parts = self
            .cache
            .as_mut()
            .context(
                "device MLA prefix projected-query device suffix cache unavailable for current KV split",
            )?
            .split_current_projected_kv_a_to_reusable_buffers(
                descriptors[0].layer_id,
                kv_a_projected_buffer,
                suffix_positions.len(),
                GLM52_MLA_KV_LORA_RANK,
                GLM52_MLA_QK_ROPE_HEAD_DIM,
            )?;
        let suffix_projected = {
            let cache = self.cache.as_mut().context(
                "device MLA prefix projected-query device suffix cache unavailable for suffix projected KV buffers",
            )?;
            cache.project_mla_kv_latent_and_split_to_suffix_reusable_buffers(
                descriptors[0].layer_id,
                &suffix_parts,
                kv_norm_weight,
                kv_b_weight,
                heads,
                nope_dim,
                v_dim,
                eps,
            )?
        };
        let suffix_positions_device = self
            .cache
            .as_mut()
            .context("device MLA prefix projected-query device suffix cache unavailable")?
            .stage_rope_positions_u32(
                suffix_positions,
                "device MLA prefix projected-query device suffix KV rows",
            )?;
        let suffix_rotated = {
            let cache = self.cache.as_mut().context(
                "device MLA prefix projected-query device suffix cache unavailable for suffix RoPE buffer",
            )?;
            cache.rotate_mla_k_rope_to_suffix_reusable_buffer(
                descriptors[0].layer_id,
                &suffix_parts,
                suffix_positions,
                suffix_positions_device,
                theta,
            )?
        };
        let attention = {
            let cache = self.cache.as_mut().context(
                "device MLA prefix projected-query device suffix cache unavailable for combined buffers",
            )?;
            prefix_projected.run_mla_rope_attention_with_device_suffix_bf16(
                library,
                cache,
                &prefix_rotated,
                queries.q_nope_buffer(),
                queries.q_rope_rotated_buffer(),
                &suffix_projected,
                &suffix_rotated,
                scale,
            )?
        };
        Ok(Some(attention))
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::commands::real_full) fn run_mla_rope_attention_parts_from_device_kv_with_device_weights_and_projected_query_device_bf16(
        &mut self,
        descriptors: &[KvBlockDescriptor],
        positions: &[u32],
        kv_norm_weight: GlmrtDeviceBuffer,
        kv_b_weight: GlmrtDeviceBuffer,
        q_projected_buffer: GlmrtDeviceBuffer,
        dsa_query: Option<(GlmrtDeviceBuffer, GlmrtDeviceBuffer)>,
        hidden_projection: Option<FlashinferMlaHiddenProjection>,
        q_suffix_positions: &[u32],
        heads: usize,
        nope_dim: usize,
        v_dim: usize,
        eps: f32,
        theta: f32,
        scale: f32,
    ) -> Result<Option<RealFullDeviceMlaAttentionParts>> {
        if descriptors.is_empty() {
            return Ok(None);
        }
        if q_suffix_positions.is_empty() {
            anyhow::bail!("device MLA KV attention requires projected query suffix positions");
        }
        let rows = descriptors
            .iter()
            .try_fold(0_usize, |acc, descriptor| {
                acc.checked_add(descriptor.token_count)
            })
            .context("device MLA KV attention row count overflows usize")?;
        if rows != positions.len() {
            anyhow::bail!(
                "device MLA KV attention row/position mismatch: descriptors={} positions={}",
                rows,
                positions.len()
            );
        }
        if q_suffix_positions.len() > rows {
            anyhow::bail!(
                "device MLA KV attention projected query suffix rows {} exceed total rows {rows}",
                q_suffix_positions.len()
            );
        }
        let layer_id = descriptors[0].layer_id;
        for descriptor in descriptors.iter().skip(1) {
            if descriptor.layer_id != layer_id {
                anyhow::bail!(
                    "device MLA KV attention descriptors span multiple layers: first={} later={}",
                    layer_id.0,
                    descriptor.layer_id.0
                );
            }
        }
        let stage_timing = device_attention_stage_timing_enabled();
        let total_start = stage_timing.then(Instant::now);
        let prefix_rows = rows - q_suffix_positions.len();
        let compressed_suffix_attention =
            use_compressed_mla_suffix_attention(prefix_rows, q_suffix_positions.len());
        let cache_is_attention_ready = self
            .cache
            .as_ref()
            .context("device MLA KV attention cache unavailable")?
            .config()
            .mla_representation
            == MlaKvCacheRepresentation::NormalizedRotated;
        let direct_dsa_prefill = {
            let cache = self
                .cache
                .as_mut()
                .context("device MLA KV attention cache unavailable")?;
            use_direct_glm_dsa_sparse_mla_prefill(
                cache.config(),
                descriptors,
                positions,
                q_suffix_positions,
                dsa_query,
            )
        };
        if dspark_attention_route_trace_enabled()
            && layer_id.0 == 0
            && prefix_rows > 0
            && q_suffix_positions.len() <= REAL_FULL_COMPRESSED_MLA_MAX_QUERY_ROWS
        {
            let first_token_start = descriptors
                .first()
                .map(|descriptor| descriptor.token_start.0)
                .unwrap_or(u64::MAX);
            let last_token_end = descriptors
                .last()
                .and_then(|descriptor| {
                    descriptor
                        .token_start
                        .0
                        .checked_add(descriptor.token_count as u64)
                })
                .unwrap_or(0);
            eprintln!(
                "real_full_attention_route layer_id={} rows={} prefix_rows={} query_rows={} descriptors={} first_token_start={} last_token_end={} dsa_query={} direct_dsa={} compressed_suffix={} dense_packed_eligible={}",
                layer_id.0,
                rows,
                prefix_rows,
                q_suffix_positions.len(),
                descriptors.len(),
                first_token_start,
                last_token_end,
                dsa_query.is_some(),
                direct_dsa_prefill,
                compressed_suffix_attention,
                rows <= REAL_FULL_PACKED_FP8_MLA_MAX_ROWS,
            );
        }
        if direct_dsa_prefill {
            // The sparse DSA graph returns compact attention output.  Its
            // caller already projects that output when
            // `hidden_projection_fused` is false, so a requested fused
            // projection is not a reason to abandon the direct sparse path.
            let query_split_start = stage_timing.then(Instant::now);
            let (
                queries,
                q_positions_device,
                packed_kv,
                kv_dtype,
                kv_row_stride_bytes,
                index_k_cache,
                max_tokens,
                physical_token_base,
                physical_page_table,
                library,
            ) = {
                let cache = self
                    .cache
                    .as_mut()
                    .context("device MLA KV attention cache unavailable for direct DSA prefill")?;
                let q_positions_device = cache
                    .stage_rope_positions_u32(
                        q_suffix_positions,
                        "direct GLM DSA sparse MLA query suffix",
                    )
                    .context("staging direct GLM DSA query positions")?;
                let queries = cache
                    .split_projected_query_suffix_to_reusable_buffers(
                        layer_id,
                        q_projected_buffer,
                        prefix_rows,
                        q_suffix_positions,
                        Some(q_positions_device),
                        heads,
                        nope_dim,
                        GLM52_MLA_QK_ROPE_HEAD_DIM,
                        theta,
                        true,
                    )
                    .context("splitting direct GLM DSA sparse MLA query")?;
                let packed_kv = cache.main_mla_cache_for_layer(layer_id)?;
                let kv_dtype = cache.config().dtype;
                let kv_row_stride_bytes = real_full_device_main_mla_row_bytes(cache.config())?;
                let (_source_layer, full_indexer) = glm_dsa_index_source_layer(layer_id.0 as usize)
                    .context("direct GLM DSA layer has no index source")?;
                let index_k_cache = if full_indexer && rows > REAL_FULL_DSA_TOP_K {
                    cache.dsa_index_k_cache_b12x_for_layer(layer_id)?
                } else {
                    None
                };
                let physical_page_table =
                    cache
                        .physical_page_table()
                        .map(
                            |(physical_pages, mapping_key)| FlashinferTargetKvPageTable {
                                physical_pages,
                                mapping_key,
                            },
                        );
                (
                    queries,
                    q_positions_device,
                    packed_kv,
                    kv_dtype,
                    kv_row_stride_bytes,
                    index_k_cache,
                    cache.config().max_tokens,
                    cache.physical_token_base,
                    physical_page_table,
                    cache.library,
                )
            };
            let query_split_ms = elapsed_ms_optional(query_split_start);
            let attention_start = stage_timing.then(Instant::now);
            let (dsa_query, dsa_weights) = dsa_query
                .map(|(query, weights)| (Some(query), Some(weights)))
                .unwrap_or((None, None));
            let launch = flashinfer_glm_dsa_sparse_mla_prefill_device_buffers(
                FlashinferGlmDsaSparseMlaPrefillInput {
                    layer_id: layer_id.0 as usize,
                    q_nope: queries.q_nope_buffer(),
                    q_rope: queries.q_rope_rotated_buffer(),
                    dsa_query,
                    dsa_weights,
                    positions: q_positions_device,
                    packed_kv,
                    kv_dtype,
                    kv_row_stride_bytes,
                    index_k_cache,
                    kv_b_weight,
                    hidden_projection,
                    total_rows: rows,
                    prefix_rows,
                    query_rows: q_suffix_positions.len(),
                    heads,
                    nope_dim,
                    rope_dim: GLM52_MLA_QK_ROPE_HEAD_DIM,
                    v_dim,
                    rank: GLM52_MLA_KV_LORA_RANK,
                    max_tokens,
                    physical_token_base,
                    physical_page_table,
                    theta,
                    scale,
                },
            )
            .context("executing direct GLM DSA sparse compressed MLA prefill")?;
            let attention_ms = elapsed_ms_optional(attention_start);
            let output_bytes = q_suffix_positions
                .len()
                .checked_mul(heads)
                .and_then(|values| values.checked_mul(v_dim))
                .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
                .context("direct GLM DSA sparse MLA output bytes overflow usize")?;
            if dsa_output_validate_enabled() {
                summarize_scheduler_bf16_device_buffer(
                    library,
                    launch.output,
                    output_bytes,
                    "direct GLM DSA sparse MLA compact output",
                )
                .with_context(|| {
                    format!(
                        "validating direct GLM DSA output for layer {} total_rows={rows} query_rows={}",
                        layer_id.0,
                        q_suffix_positions.len()
                    )
                })?;
            }
            if stage_timing {
                eprintln!(
                    "real_full_device_attention_parts_timing layer_id={} rows={} query_rows={} backend={} kv_read_ms=0.000 query_split_ms={:.3} kv_project_ms=0.000 kv_rope_ms=0.000 attention_ms={:.3} total_ms={:.3}",
                    layer_id.0,
                    rows,
                    q_suffix_positions.len(),
                    launch.backend,
                    query_split_ms,
                    attention_ms,
                    elapsed_ms_optional(total_start)
                );
            }
            return Ok(Some(RealFullDeviceMlaAttentionParts {
                status: launch.backend,
                rows: q_suffix_positions.len(),
                heads,
                nope_dim,
                rope_dim: GLM52_MLA_QK_ROPE_HEAD_DIM,
                v_dim,
                output_bytes,
                output: DeviceBufferGuard::borrowed(library, launch.output),
                hidden_projection_fused: launch.hidden_projection_fused,
                ready_event: None,
            }));
        }
        let kv_read_start = stage_timing.then(Instant::now);
        let attention_ready_frontier = if !compressed_suffix_attention && cache_is_attention_ready {
            self.cache
                .as_mut()
                .context("device MLA KV attention cache unavailable for active frontier")?
                .attention_ready_mla_frontier_parts(descriptors)?
        } else {
            None
        };
        let attention_ready_frontier_hit = attention_ready_frontier.is_some();
        let direct_span = if attention_ready_frontier.is_none()
            && compressed_suffix_attention
            && cache_is_attention_ready
        {
            self.direct_attention_ready_mla_kv_span(descriptors)?
        } else {
            None
        };
        let kv_parts = if let Some(parts) = attention_ready_frontier {
            Some(parts)
        } else if direct_span.is_none() {
            let Some(parts) = self.read_mla_kv_payloads_to_reusable_device_buffers(descriptors)?
            else {
                return Ok(None);
            };
            Some(parts)
        } else {
            None
        };
        let kv_read_ms = elapsed_ms_optional(kv_read_start);
        let kv_norm_weight_bytes = GLM52_MLA_KV_LORA_RANK
            .checked_mul(std::mem::size_of::<u16>())
            .context("device MLA KV attention kv norm weight bytes overflow usize")?;
        let kv_b_output_dim = nope_dim
            .checked_add(v_dim)
            .context("device MLA KV attention kv_b output dim overflow usize")?;
        let kv_b_weight_bytes = heads
            .checked_mul(kv_b_output_dim)
            .and_then(|values| values.checked_mul(GLM52_MLA_KV_LORA_RANK))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA KV attention kv_b weight bytes overflow usize")?;
        validate_contiguous_payload_buffer(
            "device MLA KV attention kv norm resident weight",
            kv_norm_weight,
            kv_norm_weight_bytes,
        )?;
        validate_contiguous_payload_buffer(
            "device MLA KV attention kv_b resident weight",
            kv_b_weight,
            kv_b_weight_bytes,
        )?;
        let kv_device_id = direct_span
            .map(|span| span.payload.device_id)
            .or_else(|| kv_parts.as_ref().map(|parts| parts.kv_latent.device_id))
            .context("device MLA KV attention has no cache input buffer")?;
        if kv_norm_weight.device_id != kv_device_id
            || kv_b_weight.device_id != kv_device_id
            || q_projected_buffer.device_id != kv_device_id
        {
            anyhow::bail!(
                "device MLA KV attention resident weights and query buffer must be on the same CUDA device as KV cache reads"
            );
        }

        let query_split_start = stage_timing.then(Instant::now);
        let queries = {
            let cache = self
                .cache
                .as_mut()
                .context("device MLA KV attention cache unavailable for query RoPE positions")?;
            let q_positions_device = if compressed_suffix_attention && q_suffix_positions.len() == 1
            {
                None
            } else {
                Some(cache.stage_rope_positions_u32(
                    q_suffix_positions,
                    "device MLA KV attention query suffix",
                )?)
            };
            cache.split_projected_query_suffix_to_reusable_buffers(
                layer_id,
                q_projected_buffer,
                prefix_rows,
                q_suffix_positions,
                q_positions_device,
                heads,
                nope_dim,
                GLM52_MLA_QK_ROPE_HEAD_DIM,
                theta,
                compressed_suffix_attention,
            )?
        };
        let query_split_ms = elapsed_ms_optional(query_split_start);
        if coordinator_python_capture_startup_open()
            && direct_span.is_some_and(|span| span.force_staged_hidden_projection)
        {
            // Fresh compact-NVFP4 requests use the bounded FP8 frontier, but a
            // radix-prefix hit (and another concurrently active request) can
            // only rely on canonical NVFP4 KV. Exercise that exact below-topK
            // sparse fallback during every startup query-width sweep so both
            // paths remain graph-complete after Python capture closes.
            let (
                q_positions_device,
                packed_kv,
                kv_row_stride_bytes,
                max_tokens,
                physical_token_base,
                physical_page_table,
            ) = {
                let cache = self.cache.as_mut().context(
                    "device MLA KV attention cache unavailable for NVFP4 fallback prewarm",
                )?;
                let q_positions_device = cache.stage_rope_positions_u32(
                    q_suffix_positions,
                    "NVFP4 cached-prefix sparse fallback prewarm",
                )?;
                let packed_kv = cache.main_mla_cache_for_layer(layer_id)?;
                let physical_page_table =
                    cache
                        .physical_page_table()
                        .map(
                            |(physical_pages, mapping_key)| FlashinferTargetKvPageTable {
                                physical_pages,
                                mapping_key,
                            },
                        );
                (
                    q_positions_device,
                    packed_kv,
                    real_full_device_main_mla_row_bytes(cache.config())?,
                    cache.config().max_tokens,
                    cache.physical_token_base,
                    physical_page_table,
                )
            };
            let _ = flashinfer_glm_dsa_sparse_mla_prefill_device_buffers(
                FlashinferGlmDsaSparseMlaPrefillInput {
                    layer_id: layer_id.0 as usize,
                    q_nope: queries.q_nope_buffer(),
                    q_rope: queries.q_rope_rotated_buffer(),
                    dsa_query: None,
                    dsa_weights: None,
                    positions: q_positions_device,
                    packed_kv,
                    kv_dtype: KvCacheDType::Nvfp4,
                    kv_row_stride_bytes,
                    index_k_cache: None,
                    kv_b_weight,
                    hidden_projection,
                    total_rows: rows,
                    prefix_rows,
                    query_rows: q_suffix_positions.len(),
                    heads,
                    nope_dim,
                    rope_dim: GLM52_MLA_QK_ROPE_HEAD_DIM,
                    v_dim,
                    rank: GLM52_MLA_KV_LORA_RANK,
                    max_tokens,
                    physical_token_base,
                    physical_page_table,
                    theta,
                    scale,
                },
            )
            .context("prewarming cached-prefix NVFP4 sparse attention fallback")?;
        }
        if let (Some(span), Some(projection)) = (direct_span, hidden_projection) {
            let trace_library = self
                .cache
                .as_ref()
                .context("device MLA KV attention cache unavailable for trace readback")?
                .library;
            if let Err(error) = maybe_dump_packed_mla_trace(
                trace_library,
                descriptors,
                positions,
                q_suffix_positions,
                span,
                q_projected_buffer,
                queries.q_nope_buffer(),
                queries.q_rope_rotated_buffer(),
                kv_b_weight,
                projection,
                heads,
                nope_dim,
                GLM52_MLA_QK_ROPE_HEAD_DIM,
                v_dim,
                eps,
                theta,
                scale,
            ) {
                eprintln!("real_full_packed_mla_trace_failed error={error:#}");
            }
        }
        let kv_project_start = stage_timing.then(Instant::now);
        let (normalized_latent, projected) = {
            let cache = self
                .cache
                .as_mut()
                .context("device MLA KV attention cache unavailable for projected KV buffers")?;
            if compressed_suffix_attention {
                (
                    if direct_span.is_some() {
                        None
                    } else {
                        let parts = kv_parts
                            .as_ref()
                            .context("compressed MLA decode split cache input missing")?;
                        Some(if cache_is_attention_ready {
                            parts.kv_latent
                        } else {
                            cache.normalize_mla_kv_latent_to_reusable_buffer(
                                layer_id,
                                parts,
                                kv_norm_weight,
                                eps,
                            )?
                        })
                    },
                    None,
                )
            } else {
                let parts = kv_parts
                    .as_ref()
                    .context("expanded MLA attention split cache input missing")?;
                (
                    None,
                    Some(cache.project_mla_kv_latent_and_split_to_reusable_buffers(
                        layer_id,
                        parts,
                        kv_norm_weight,
                        kv_b_weight,
                        heads,
                        nope_dim,
                        v_dim,
                        cache_is_attention_ready,
                        eps,
                    )?),
                )
            }
        };
        let kv_project_ms = elapsed_ms_optional(kv_project_start);
        let kv_rope_start = stage_timing.then(Instant::now);
        let rotated = if direct_span.is_some() {
            None
        } else {
            let parts = kv_parts
                .as_ref()
                .context("MLA attention split RoPE cache input missing")?;
            Some(if cache_is_attention_ready {
                RealFullDeviceMlaKvRopeDeviceBuffers {
                    rows: parts.rows,
                    rotary_dim: GLM52_MLA_QK_ROPE_HEAD_DIM,
                    k_rope_rotated_bytes: parts.k_rope_bytes,
                    k_rope_rotated: parts.k_rope,
                }
            } else {
                let cache = self
                    .cache
                    .as_mut()
                    .context("device MLA KV attention cache unavailable for KV RoPE positions")?;
                let kv_positions_device = cache
                    .stage_rope_positions_u32(positions, "device MLA KV attention cache rows")?;
                cache.rotate_mla_k_rope_to_reusable_buffer(
                    layer_id,
                    parts,
                    positions,
                    kv_positions_device,
                    theta,
                )?
            })
        };
        let kv_rope_ms = elapsed_ms_optional(kv_rope_start);
        let library = self
            .cache
            .as_ref()
            .context("device MLA KV attention cache unavailable for attention launch")?
            .library;
        let attention_output_rows = if compressed_suffix_attention {
            q_suffix_positions.len()
        } else {
            rows
        };
        let attention_output_bytes = attention_output_rows
            .checked_mul(heads)
            .and_then(|values| values.checked_mul(v_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA KV attention reusable output bytes overflow usize")?;
        let attention_output = self
            .cache
            .as_mut()
            .context("device MLA KV attention cache unavailable for reusable output")?
            .attention_output
            .buffer(
                attention_output_bytes,
                "device MLA attention reusable output",
            )?;
        let attention_start = stage_timing.then(Instant::now);
        let attention = if compressed_suffix_attention {
            let kv_input = if let Some(span) = direct_span {
                anyhow::ensure!(
                    span.rows == rows,
                    "direct MLA KV span rows {} do not match attention rows {rows}",
                    span.rows
                );
                FlashinferCompressedMlaKvInput::Interleaved {
                    payload: span.payload,
                    dtype: span.dtype,
                    row_stride_bytes: span.row_stride_bytes,
                    row_offset: span.row_offset,
                    physical_page_table: span.physical_page_table,
                    force_staged_hidden_projection: span.force_staged_hidden_projection,
                }
            } else {
                FlashinferCompressedMlaKvInput::SplitBf16 {
                    latent: normalized_latent
                        .context("compressed MLA decode normalized latent missing")?,
                    rope: rotated
                        .as_ref()
                        .context("compressed MLA decode rotated RoPE missing")?
                        .k_rope_rotated,
                }
            };
            let bf16_bytes = std::mem::size_of::<u16>();
            let q_nope_row_bytes = heads
                .checked_mul(nope_dim)
                .and_then(|values| values.checked_mul(bf16_bytes))
                .context("compressed MLA suffix q_nope row bytes overflow usize")?;
            let q_rope_row_bytes = heads
                .checked_mul(GLM52_MLA_QK_ROPE_HEAD_DIM)
                .and_then(|values| values.checked_mul(bf16_bytes))
                .context("compressed MLA suffix q_rope row bytes overflow usize")?;
            let output_row_bytes = heads
                .checked_mul(v_dim)
                .and_then(|values| values.checked_mul(bf16_bytes))
                .context("compressed MLA suffix output row bytes overflow usize")?;
            let query_rows = q_suffix_positions.len();
            let mut backend = None;
            let mut hidden_projection_fused = false;
            let mut ready_event = None;
            let packed_fp8_suffix =
                use_packed_fp8_mla_suffix(query_rows, packed_fp8_mla_batched_suffix_enabled())
                    && rows <= REAL_FULL_PACKED_FP8_MLA_MAX_ROWS
                    && matches!(
                        kv_input,
                        FlashinferCompressedMlaKvInput::Interleaved {
                            dtype: KvCacheDType::Fp8,
                            ..
                        }
                    );
            if packed_fp8_suffix {
                // Recurrent decode and exact MTP target verification can both
                // close attention with the packed-W8 O projection. Other
                // multirow projection modes retain their existing outer path.
                let packed_hidden_projection = hidden_projection
                    .filter(|projection| query_rows == 1 || projection.w8a16.is_some());
                let launch = flashinfer_compressed_mla_decode_device_buffers(
                    layer_id.0 as usize,
                    queries.q_nope_buffer(),
                    queries.q_rope_rotated_buffer(),
                    kv_input,
                    kv_b_weight,
                    attention_output,
                    rows,
                    prefix_rows,
                    query_rows,
                    heads,
                    nope_dim,
                    GLM52_MLA_QK_ROPE_HEAD_DIM,
                    v_dim,
                    scale,
                    packed_hidden_projection,
                )
                .context("executing batched FlashInfer packed-FP8 MLA suffix")?;
                backend = Some(launch.backend);
                hidden_projection_fused |= launch.hidden_projection_fused;
                ready_event = launch.ready_event;
            } else {
                for query_index in 0..query_rows {
                    let q_nope = device_buffer_byte_view(
                        queries.q_nope_buffer(),
                        query_index * q_nope_row_bytes,
                        q_nope_row_bytes,
                        "compressed MLA suffix q_nope row",
                    )?;
                    let q_rope = device_buffer_byte_view(
                        queries.q_rope_rotated_buffer(),
                        query_index * q_rope_row_bytes,
                        q_rope_row_bytes,
                        "compressed MLA suffix q_rope row",
                    )?;
                    let output = device_buffer_byte_view(
                        attention_output,
                        query_index * output_row_bytes,
                        output_row_bytes,
                        "compressed MLA suffix attention output row",
                    )?;
                    let causal_rows = prefix_rows
                        .checked_add(query_index + 1)
                        .context("compressed MLA suffix causal row count overflow usize")?;
                    let row_hidden_projection = hidden_projection
                        .map(|projection| {
                            let output_row_bytes = projection
                                .hidden_dim
                                .checked_mul(std::mem::size_of::<u16>())
                                .context(
                                    "compressed MLA suffix hidden projection row bytes overflow",
                                )?;
                            let output = device_buffer_byte_view(
                                projection.output,
                                query_index * output_row_bytes,
                                output_row_bytes,
                                "compressed MLA suffix hidden projection output row",
                            )?;
                            Ok::<FlashinferMlaHiddenProjection, anyhow::Error>(
                                FlashinferMlaHiddenProjection {
                                    output,
                                    ..projection
                                },
                            )
                        })
                        .transpose()?;
                    let launch = flashinfer_compressed_mla_decode_device_buffers(
                        layer_id.0 as usize,
                        q_nope,
                        q_rope,
                        kv_input,
                        kv_b_weight,
                        output,
                        causal_rows,
                        causal_rows - 1,
                        1,
                        heads,
                        nope_dim,
                        GLM52_MLA_QK_ROPE_HEAD_DIM,
                        v_dim,
                        scale,
                        row_hidden_projection,
                    )
                    .with_context(|| {
                        format!(
                            "executing FlashInfer compressed-cache MLA suffix row {query_index}/{query_rows}"
                        )
                    })?;
                    backend = Some(launch.backend);
                    hidden_projection_fused |= launch.hidden_projection_fused;
                    if launch.ready_event.is_some() {
                        ready_event = launch.ready_event;
                    }
                }
            }
            RealFullDeviceMlaAttentionParts {
                status: backend.context("compressed MLA suffix produced no attention launch")?,
                rows: query_rows,
                heads,
                nope_dim,
                rope_dim: GLM52_MLA_QK_ROPE_HEAD_DIM,
                v_dim,
                output_bytes: query_rows * output_row_bytes,
                output: DeviceBufferGuard::borrowed(library, attention_output),
                hidden_projection_fused,
                ready_event,
            }
        } else if prefix_rows > 0 {
            projected
                .as_ref()
                .context("expanded MLA decode projection missing")?
                .run_mla_rope_attention_suffix_bf16(
                    library,
                    layer_id,
                    rotated
                        .as_ref()
                        .context("expanded MLA decode rotated RoPE missing")?,
                    queries.q_nope_buffer(),
                    queries.q_rope_rotated_buffer(),
                    prefix_rows,
                    q_suffix_positions.len(),
                    attention_output,
                    scale,
                )?
        } else {
            projected
                .as_ref()
                .context("expanded MLA prefill projection missing")?
                .run_mla_rope_attention_bf16(
                    library,
                    layer_id,
                    rotated
                        .as_ref()
                        .context("expanded MLA prefill rotated RoPE missing")?,
                    queries.q_nope_buffer(),
                    queries.q_rope_rotated_buffer(),
                    attention_output,
                    scale,
                )?
        };
        let attention_ms = elapsed_ms_optional(attention_start);
        if stage_timing {
            eprintln!(
                "real_full_device_attention_parts_timing layer_id={} rows={} query_rows={} attention_ready_frontier={} kv_read_ms={:.3} query_split_ms={:.3} kv_project_ms={:.3} kv_rope_ms={:.3} attention_ms={:.3} total_ms={:.3}",
                layer_id.0,
                rows,
                q_suffix_positions.len(),
                attention_ready_frontier_hit,
                kv_read_ms,
                query_split_ms,
                kv_project_ms,
                kv_rope_ms,
                attention_ms,
                elapsed_ms_optional(total_start)
            );
        }
        Ok(Some(attention))
    }

    #[cfg(test)]
    fn run_scheduler_mla_attention_from_device_kv_bf16(
        &mut self,
        descriptors: &[KvBlockDescriptor],
        q_suffix_positions: &[u32],
        query_hidden: Option<GlmrtDeviceBuffer>,
    ) -> Result<Option<RealFullDeviceSchedulerAttentionLaunch>> {
        let mut positions = std::mem::take(&mut self.scheduler_attention_positions);
        let result = (|| {
            descriptor_positions_u32_into(descriptors, &mut positions)
                .context("building scheduler device MLA attention positions")?;
            self.run_scheduler_mla_attention_from_device_kv_positions_bf16(
                descriptors,
                &positions,
                q_suffix_positions,
                query_hidden,
            )
        })();
        self.scheduler_attention_positions = positions;
        result
    }

    pub(in crate::commands::real_full) fn run_scheduler_mla_attention_from_device_kv_descriptor_sets_bf16(
        &mut self,
        visible_blocks: &[KvBackedBlock],
        current_descriptors: &[KvBlockDescriptor],
        query_hidden: Option<GlmrtDeviceBuffer>,
    ) -> Result<Option<RealFullDeviceSchedulerAttentionLaunch>> {
        let mut descriptors = std::mem::take(&mut self.scheduler_attention_descriptors);
        let mut positions = std::mem::take(&mut self.scheduler_attention_positions);
        let mut q_suffix_positions = std::mem::take(&mut self.scheduler_attention_query_positions);
        let result = (|| {
            descriptors.clear();
            descriptors.reserve(visible_blocks.len() + current_descriptors.len());
            descriptors.extend(visible_blocks.iter().map(|block| block.descriptor.clone()));
            descriptors.extend(current_descriptors.iter().cloned());
            descriptor_positions_u32_into(&descriptors, &mut positions)
                .context("building scheduler device MLA attention descriptor positions")?;
            descriptor_positions_u32_into(current_descriptors, &mut q_suffix_positions)
                .context("building scheduler device MLA attention query suffix positions")?;
            self.run_scheduler_mla_attention_from_device_kv_positions_bf16(
                &descriptors,
                &positions,
                &q_suffix_positions,
                query_hidden,
            )
        })();
        self.scheduler_attention_descriptors = descriptors;
        self.scheduler_attention_positions = positions;
        self.scheduler_attention_query_positions = q_suffix_positions;
        result
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::commands::real_full) fn run_scheduler_mla_attention_from_device_kv_descriptor_sets_with_projected_query_bf16(
        &mut self,
        visible_blocks: &[KvBackedBlock],
        current_descriptors: &[KvBlockDescriptor],
        query_descriptors: &[KvBlockDescriptor],
        q_projected: DeviceBf16Output,
        dsa_query: Option<(GlmrtDeviceBuffer, GlmrtDeviceBuffer)>,
        kv_norm_weight: GlmrtDeviceBuffer,
        kv_b_weight: GlmrtDeviceBuffer,
        output_projection_weight_name: &str,
        heads: usize,
        nope_dim: usize,
        v_dim: usize,
        eps: f32,
        theta: f32,
        scale: f32,
    ) -> Result<Option<RealFullDeviceSchedulerAttentionLaunch>> {
        let mut descriptors = std::mem::take(&mut self.scheduler_attention_descriptors);
        let mut positions = std::mem::take(&mut self.scheduler_attention_positions);
        let mut q_suffix_positions = std::mem::take(&mut self.scheduler_attention_query_positions);
        let result = (|| {
            descriptors.clear();
            descriptors.reserve(visible_blocks.len() + current_descriptors.len());
            descriptors.extend(visible_blocks.iter().map(|block| block.descriptor.clone()));
            descriptors.extend(current_descriptors.iter().cloned());
            descriptor_positions_u32_into(&descriptors, &mut positions)
                .context("building scheduler real device MLA attention descriptor positions")?;
            descriptor_positions_u32_into(query_descriptors, &mut q_suffix_positions)
                .context("building scheduler real device MLA attention query suffix positions")?;
            self.run_scheduler_mla_attention_from_device_kv_positions_with_projected_query_bf16(
                &descriptors,
                &positions,
                &q_suffix_positions,
                q_projected,
                dsa_query,
                kv_norm_weight,
                kv_b_weight,
                output_projection_weight_name,
                heads,
                nope_dim,
                v_dim,
                eps,
                theta,
                scale,
            )
        })();
        self.scheduler_attention_descriptors = descriptors;
        self.scheduler_attention_positions = positions;
        self.scheduler_attention_query_positions = q_suffix_positions;
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn run_scheduler_mla_attention_from_device_kv_positions_with_projected_query_bf16(
        &mut self,
        descriptors: &[KvBlockDescriptor],
        positions: &[u32],
        q_suffix_positions: &[u32],
        q_projected: DeviceBf16Output,
        dsa_query: Option<(GlmrtDeviceBuffer, GlmrtDeviceBuffer)>,
        kv_norm_weight: GlmrtDeviceBuffer,
        kv_b_weight: GlmrtDeviceBuffer,
        output_projection_weight_name: &str,
        heads: usize,
        nope_dim: usize,
        v_dim: usize,
        eps: f32,
        theta: f32,
        scale: f32,
    ) -> Result<Option<RealFullDeviceSchedulerAttentionLaunch>> {
        if descriptors.is_empty() || q_suffix_positions.is_empty() {
            return Ok(None);
        }
        let Some(cache) = self.cache.as_mut() else {
            return Ok(None);
        };
        if !matches!(
            cache.config().dtype,
            KvCacheDType::Bf16 | KvCacheDType::Fp8 | KvCacheDType::Nvfp4
        ) {
            anyhow::bail!(
                "scheduler real device MLA attention requires BF16, FP8, or NVFP4 cache payloads, got {}",
                cache.config().dtype_label()
            );
        }
        let rows = descriptors
            .iter()
            .try_fold(0_usize, |acc, descriptor| {
                acc.checked_add(descriptor.token_count)
            })
            .context("scheduler real device MLA attention descriptor row count overflow")?;
        if q_suffix_positions.len() > rows {
            anyhow::bail!(
                "scheduler real device MLA attention query rows {} exceed descriptor rows {rows}",
                q_suffix_positions.len()
            );
        }
        if positions.len() != rows {
            anyhow::bail!(
                "scheduler real device MLA attention positions length mismatch: expected {rows} got {}",
                positions.len()
            );
        }
        let stage_timing = device_attention_stage_timing_enabled();
        let total_start = stage_timing.then(Instant::now);
        let attention_parts_start = stage_timing.then(Instant::now);
        let q_projected_buffer = q_projected.buffer();
        let compact_output_values_per_row = heads
            .checked_mul(v_dim)
            .context("scheduler real device MLA attention output row width overflows usize")?;
        let hidden_projection = if q_suffix_positions.len()
            <= REAL_FULL_COMPRESSED_MLA_MAX_QUERY_ROWS
            && matches!(
                cache.config().dtype,
                KvCacheDType::Fp8 | KvCacheDType::Nvfp4
            ) {
            let projection_weight_bytes = GLM52_HIDDEN_SIZE
                .checked_mul(compact_output_values_per_row)
                .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
                .context(
                    "scheduler real device MLA attention output projection weight bytes overflow",
                )?;
            let w4a16_enabled = coordinator_w4a16_o_proj_decode_enabled();
            let w8a16_enabled = coordinator_w8a16_o_proj_decode_enabled();
            anyhow::ensure!(
                !(w4a16_enabled && w8a16_enabled),
                "coordinator O projection cannot enable W4A16 and W8A16 simultaneously"
            );
            let w8a16 = w8a16_enabled
                .then(|| {
                    preloaded_coordinator_w8a16_projection(
                        output_projection_weight_name,
                        compact_output_values_per_row,
                        GLM52_HIDDEN_SIZE,
                    )
                })
                .transpose()?;
            let projection_weight = if let Some(w8a16) = w8a16 {
                w8a16.weight
            } else {
                preloaded_resident_weight_device_buffer(
                    output_projection_weight_name,
                    projection_weight_bytes,
                )
                .context("resolving scheduler real device MLA attention output projection weight")?
            };
            Some(FlashinferMlaHiddenProjection {
                weight: projection_weight,
                output: q_projected_buffer,
                hidden_dim: GLM52_HIDDEN_SIZE,
                w4a16: w4a16_enabled
                    .then(|| {
                        preloaded_coordinator_w4a16_projection(
                            output_projection_weight_name,
                            compact_output_values_per_row,
                            GLM52_HIDDEN_SIZE,
                        )
                    })
                    .transpose()?,
                w8a16,
            })
        } else {
            None
        };
        let attention =
            self.run_mla_rope_attention_parts_from_device_kv_with_device_weights_and_projected_query_device_bf16(
                descriptors,
                positions,
                kv_norm_weight,
                kv_b_weight,
                q_projected_buffer,
                dsa_query,
                hidden_projection,
                q_suffix_positions,
                heads,
                nope_dim,
                v_dim,
                eps,
                theta,
                scale,
            )?;
        let attention_parts_ms = elapsed_ms_optional(attention_parts_start);
        let Some(mut attention) = attention else {
            return Ok(None);
        };
        let attention_rows = attention.rows();
        let output_projection_start = stage_timing.then(Instant::now);
        let projected_output = if attention.hidden_projection_fused() {
            let ready_event = attention.take_ready_event();
            let mut projected = q_projected.into_prefix_shape(
                q_suffix_positions.len(),
                GLM52_HIDDEN_SIZE,
                "cuda-kv-cache-mla-rope-attention-hidden-projection-packed-graph",
                "packed FP8 MLA fused hidden projection",
            )?;
            if let Some(ready_event) = ready_event {
                projected.set_ready_event(ready_event);
            }
            projected
        } else {
            let suffix_row_offset = attention_rows
                .checked_sub(q_suffix_positions.len())
                .context("scheduler real device MLA attention output rows fewer than query rows")?;
            let attention_suffix_output = attention
                .output_row_buffer(suffix_row_offset, q_suffix_positions.len())
                .context("building scheduler real device MLA attention suffix output view")?;
            scheduler_attention_project_output_to_hidden_bf16_preloaded_resident(
                attention_suffix_output,
                q_suffix_positions.len(),
                compact_output_values_per_row,
                output_projection_weight_name,
            )
            .context("projecting scheduler real MLA attention output to hidden width")?
        };
        let output_projection_ms = elapsed_ms_optional(output_projection_start);
        let status = "cuda-kv-cache-mla-rope-attention-hidden-projection-device-buffer";
        let output_bytes = q_suffix_positions
            .len()
            .checked_mul(GLM52_HIDDEN_SIZE)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("scheduler real device MLA attention hidden output bytes overflow")?;
        let output_device = projected_output;
        let output_account_start = stage_timing.then(Instant::now);
        let output_summary = account_scheduler_hidden_width_device_bf16_output(
            &output_device,
            "scheduler real device MLA attention hidden-width output",
        )?;
        let output_account_ms = elapsed_ms_optional(output_account_start);
        if stage_timing {
            eprintln!(
                "real_full_device_attention_outer_timing layer_id={} rows={} query_rows={} attention_parts_ms={:.3} output_projection_ms={:.3} output_account_ms={:.3} total_ms={:.3}",
                descriptors[0].layer_id.0,
                rows,
                q_suffix_positions.len(),
                attention_parts_ms,
                output_projection_ms,
                output_account_ms,
                elapsed_ms_optional(total_start),
            );
        }
        Ok(Some(RealFullDeviceSchedulerAttentionLaunch {
            status,
            descriptors: descriptors.len(),
            rows: attention_rows,
            output_rows: q_suffix_positions.len(),
            output_row_offset: 0,
            query_rows: q_suffix_positions.len(),
            heads,
            nope_dim,
            rope_dim: GLM52_MLA_QK_ROPE_HEAD_DIM,
            v_dim,
            output_bytes,
            output_values: output_summary.values,
            output_finite_values: output_summary.finite_values,
            output_nonzero_values: output_summary.nonzero_values,
            output_checksum: output_summary.checksum,
            output_bf16: None,
            output_device,
            output_projected_to_hidden: true,
        }))
    }

    fn run_scheduler_mla_attention_from_device_kv_positions_bf16(
        &mut self,
        descriptors: &[KvBlockDescriptor],
        positions: &[u32],
        q_suffix_positions: &[u32],
        query_hidden: Option<GlmrtDeviceBuffer>,
    ) -> Result<Option<RealFullDeviceSchedulerAttentionLaunch>> {
        if descriptors.is_empty() || q_suffix_positions.is_empty() {
            return Ok(None);
        }
        let Some(cache) = self.cache.as_mut() else {
            return Ok(None);
        };
        if cache.config().dtype != KvCacheDType::Bf16 {
            anyhow::bail!(
                "scheduler device MLA attention currently requires BF16 cache payloads, got {}",
                cache.config().dtype_label()
            );
        }
        let rows = descriptors
            .iter()
            .try_fold(0_usize, |acc, descriptor| {
                acc.checked_add(descriptor.token_count)
            })
            .context("scheduler device MLA attention descriptor row count overflow")?;
        if q_suffix_positions.len() > rows {
            anyhow::bail!(
                "scheduler device MLA attention query rows {} exceed descriptor rows {rows}",
                q_suffix_positions.len()
            );
        }
        if positions.len() != rows {
            anyhow::bail!(
                "scheduler device MLA attention positions length mismatch: expected {rows} got {}",
                positions.len()
            );
        }
        let layer_id = descriptors[0].layer_id;
        for descriptor in descriptors.iter().skip(1) {
            if descriptor.layer_id != layer_id {
                anyhow::bail!(
                    "scheduler device MLA attention descriptors span multiple layers: first={} later={}",
                    layer_id.0,
                    descriptor.layer_id.0
                );
            }
        }
        let library = cache.library;
        let project_attention_output_to_hidden = coordinator_cuda_reference_kernels_enabled();
        let SchedulerAttentionResidentBuffers {
            kv_norm_weight,
            kv_b_weight,
            query_projection_weight,
            output_projection_weight,
        } = self.scheduler_attention_resident_buffers(
            query_hidden.is_some(),
            project_attention_output_to_hidden,
        )?;
        let q_projected = if let Some(query_hidden) = query_hidden {
            let projected_query_bytes =
                scheduler_attention_projected_query_bytes(q_suffix_positions.len())?;
            let q_projected = self
                .cache
                .as_mut()
                .context(
                    "scheduler device MLA attention cache unavailable for projected query output",
                )?
                .scheduler_projected_query
                .buffer(
                    projected_query_bytes,
                    "scheduler device MLA attention projected query",
                )?;
            scheduler_attention_project_query_from_hidden_bf16_into(
                layer_id,
                query_hidden,
                q_suffix_positions.len(),
                query_projection_weight
                    .context("scheduler attention query projection weight missing")?,
                q_projected,
            )
            .context("projecting scheduler MLA attention query from resident hidden")?;
            q_projected
        } else {
            self.scheduler_attention_static_projected_query_buffer(q_suffix_positions.len())?
        };
        let attention =
            self.run_mla_rope_attention_parts_from_device_kv_with_device_weights_and_projected_query_device_bf16(
                descriptors,
                positions,
                kv_norm_weight,
                kv_b_weight,
                q_projected,
                None,
                None,
                q_suffix_positions,
                REAL_FULL_SCHEDULER_DEVICE_ATTENTION_HEADS,
                REAL_FULL_SCHEDULER_DEVICE_ATTENTION_NOPE_DIM,
                REAL_FULL_SCHEDULER_DEVICE_ATTENTION_VALUE_DIM,
                REAL_FULL_SCHEDULER_DEVICE_ATTENTION_EPS,
                GLM52_MLA_ROPE_THETA,
                REAL_FULL_SCHEDULER_DEVICE_ATTENTION_SCALE,
            )?;
        let Some(attention) = attention else {
            return Ok(None);
        };
        let attention_rows = attention.rows();
        let compact_output_values_per_row = attention
            .heads()
            .checked_mul(attention.v_dim())
            .context("scheduler device MLA attention output row width overflows usize")?;
        let (
            status,
            output_rows,
            output_row_offset,
            output_bytes,
            output_values,
            output_finite_values,
            output_nonzero_values,
            output_checksum,
            output_bf16,
            output_device,
            output_projected_to_hidden,
        ) = if project_attention_output_to_hidden {
            let suffix_row_offset = attention_rows
                .checked_sub(q_suffix_positions.len())
                .context("scheduler device MLA attention output rows fewer than query rows")?;
            let attention_suffix_output = attention
                .output_row_buffer(suffix_row_offset, q_suffix_positions.len())
                .context("building scheduler device MLA attention suffix output view")?;
            let output_projection_weight = output_projection_weight
                .context("scheduler attention output projection weight missing")?;
            let projected_output = scheduler_attention_project_output_to_hidden_bf16(
                layer_id,
                library,
                attention_suffix_output,
                q_suffix_positions.len(),
                compact_output_values_per_row,
                output_projection_weight,
            )
            .context("projecting scheduler MLA attention output to hidden width")?;
            let status = "cuda-kv-cache-mla-rope-attention-hidden-projection-device-buffer";
            let output_bytes = projected_output.buffer.bytes;
            let output_device = device_bf16_output_from_owned_device_buffer(
                library,
                projected_output.into_buffer()?,
                q_suffix_positions.len(),
                GLM52_HIDDEN_SIZE,
                status,
                "scheduler device MLA attention hidden-width output",
            )
            .context("adopting scheduler device MLA attention hidden-width output buffer")?;
            let output_summary = account_scheduler_hidden_width_device_bf16_output(
                &output_device,
                "scheduler device MLA attention hidden-width output",
            )?;
            let output_values = output_summary.values;
            let output_finite_values = output_summary.finite_values;
            let output_nonzero_values = output_summary.nonzero_values;
            let output_checksum = output_summary.checksum;
            (
                status,
                q_suffix_positions.len(),
                0,
                output_bytes,
                output_values,
                output_finite_values,
                output_nonzero_values,
                output_checksum,
                None,
                output_device,
                true,
            )
        } else {
            let output_summary = summarize_scheduler_attention_output(&attention)?;
            let output_bytes = attention.output_buffer().bytes;
            let output_device = attention
                .into_device_bf16_output(
                    attention_rows,
                    compact_output_values_per_row,
                    "scheduler device MLA attention output",
                )
                .context("adopting scheduler device MLA attention output buffer")?;
            (
                output_device.backend,
                attention_rows,
                attention_rows
                    .checked_sub(q_suffix_positions.len())
                    .context("scheduler device MLA attention output rows fewer than query rows")?,
                output_bytes,
                output_summary.values,
                output_summary.finite_values,
                output_summary.nonzero_values,
                output_summary.checksum,
                Some(output_summary.output_bf16),
                output_device,
                false,
            )
        };
        Ok(Some(RealFullDeviceSchedulerAttentionLaunch {
            status,
            descriptors: descriptors.len(),
            rows: attention_rows,
            output_rows,
            output_row_offset,
            query_rows: q_suffix_positions.len(),
            heads: REAL_FULL_SCHEDULER_DEVICE_ATTENTION_HEADS,
            nope_dim: REAL_FULL_SCHEDULER_DEVICE_ATTENTION_NOPE_DIM,
            rope_dim: GLM52_MLA_QK_ROPE_HEAD_DIM,
            v_dim: REAL_FULL_SCHEDULER_DEVICE_ATTENTION_VALUE_DIM,
            output_bytes,
            output_values,
            output_finite_values,
            output_nonzero_values,
            output_checksum,
            output_bf16,
            output_device,
            output_projected_to_hidden,
        }))
    }

    fn scheduler_attention_resident_buffers(
        &mut self,
        include_query_projection: bool,
        include_output_projection: bool,
    ) -> Result<SchedulerAttentionResidentBuffers> {
        if self.scheduler_attention_weights.is_none() {
            let kv_norm_weight = self.upload_scheduler_attention_resident_weight(
                "scheduler device MLA attention resident kv norm weight",
                fill_scheduler_attention_kv_norm_weight_bf16,
            )?;
            let kv_b_weight = self.upload_scheduler_attention_resident_weight(
                "scheduler device MLA attention resident kv_b weight",
                fill_scheduler_attention_kv_b_weight_bf16,
            )?;
            self.scheduler_attention_resident_uploads += 2;
            self.scheduler_attention_weights = Some(RealFullSchedulerAttentionResidentWeights {
                kv_norm_weight,
                kv_b_weight,
                query_projection_weight: None,
                output_projection_weight: None,
            });
        }
        if include_query_projection {
            let needs_query_projection = self
                .scheduler_attention_weights
                .as_ref()
                .context("scheduler attention resident weights missing after upload")?
                .query_projection_weight
                .is_none();
            if needs_query_projection {
                let query_projection_weight = self.upload_scheduler_attention_resident_weight(
                    "scheduler device MLA attention resident query projection weight",
                    fill_scheduler_attention_query_projection_weight_bf16,
                )?;
                self.scheduler_attention_resident_uploads += 1;
                let weights = self
                    .scheduler_attention_weights
                    .as_mut()
                    .context("scheduler attention resident weights missing after upload")?;
                weights.query_projection_weight = Some(query_projection_weight);
            }
        }
        if include_output_projection {
            let needs_output_projection = self
                .scheduler_attention_weights
                .as_ref()
                .context("scheduler attention resident weights missing after upload")?
                .output_projection_weight
                .is_none();
            if needs_output_projection {
                let output_projection_weight = self.upload_scheduler_attention_resident_weight(
                    "scheduler device MLA attention resident output projection weight",
                    fill_scheduler_attention_output_projection_weight_bf16,
                )?;
                self.scheduler_attention_resident_uploads += 1;
                let weights = self
                    .scheduler_attention_weights
                    .as_mut()
                    .context("scheduler attention resident weights missing after upload")?;
                weights.output_projection_weight = Some(output_projection_weight);
            }
        }
        let weights = self
            .scheduler_attention_weights
            .as_ref()
            .context("scheduler attention resident weights missing after upload")?;
        self.scheduler_attention_resident_buffer_uses += 3 + usize::from(include_output_projection);
        Ok(SchedulerAttentionResidentBuffers {
            kv_norm_weight: weights.kv_norm_weight.buffer,
            kv_b_weight: weights.kv_b_weight.buffer,
            query_projection_weight: weights
                .query_projection_weight
                .as_ref()
                .map(|weight| weight.buffer),
            output_projection_weight: weights
                .output_projection_weight
                .as_ref()
                .map(|weight| weight.buffer),
        })
    }

    fn upload_scheduler_attention_resident_weight<F>(
        &mut self,
        context: &str,
        fill: F,
    ) -> Result<DeviceBufferGuard<'static>>
    where
        F: FnOnce(&mut Vec<u8>) -> Result<()>,
    {
        fill(&mut self.scheduler_attention_weight_upload_bf16_scratch)
            .with_context(|| format!("filling {context} BF16 upload scratch"))?;
        self.cache
            .as_mut()
            .context("scheduler attention upload requires live device KV cache")?
            .upload_device_bytes_from_pinned_staging(
                &self.scheduler_attention_weight_upload_bf16_scratch,
                context,
            )
    }

    fn scheduler_attention_static_projected_query_buffer(
        &mut self,
        query_rows: usize,
    ) -> Result<GlmrtDeviceBuffer> {
        if !self.scheduler_attention_queries.contains_key(&query_rows) {
            fill_scheduler_attention_projected_query_bf16(
                &mut self.scheduler_attention_projected_query_upload_bf16_scratch,
                query_rows,
            )?;
            let q_projected = self
                .cache
                .as_mut()
                .context(
                    "scheduler attention projected query upload requires live device KV cache",
                )?
                .upload_device_bytes_from_pinned_staging(
                    &self.scheduler_attention_projected_query_upload_bf16_scratch,
                    "scheduler device MLA attention resident projected query",
                )?;
            self.scheduler_attention_resident_uploads += 1;
            self.scheduler_attention_queries
                .insert(query_rows, q_projected);
        }
        let q_projected = self
            .scheduler_attention_queries
            .get(&query_rows)
            .context("scheduler attention resident query missing after upload")?;
        Ok(q_projected.buffer)
    }

    pub(in crate::commands::real_full) fn summary(&self) -> RealFullDeviceKvExecutionSummary {
        RealFullDeviceKvExecutionSummary {
            status: self.status,
            writes: self.writes,
            reads: self.reads,
            bytes: self.bytes,
            scheduler_attention_resident_uploads: self.scheduler_attention_resident_uploads,
            scheduler_attention_resident_buffer_uses: self.scheduler_attention_resident_buffer_uses,
            scheduler_attention_resident_query_shapes: self.scheduler_attention_queries.len(),
            uses_device_kv_cache: self.cache.is_some(),
        }
    }
}

struct SchedulerAttentionOutputSummary {
    values: usize,
    finite_values: usize,
    nonzero_values: usize,
    checksum: f64,
    output_bf16: Vec<u8>,
}

struct SchedulerAttentionResidentBuffers {
    kv_norm_weight: GlmrtDeviceBuffer,
    kv_b_weight: GlmrtDeviceBuffer,
    query_projection_weight: Option<GlmrtDeviceBuffer>,
    output_projection_weight: Option<GlmrtDeviceBuffer>,
}

fn summarize_scheduler_attention_output(
    attention: &RealFullDeviceMlaAttentionParts,
) -> Result<SchedulerAttentionOutputSummary> {
    summarize_scheduler_bf16_device_buffer(
        attention.output.library,
        attention.output.buffer,
        attention.output_bytes,
        "scheduler device MLA attention output",
    )
}

fn account_scheduler_hidden_width_device_bf16_output(
    output: &DeviceBf16Output,
    context: &str,
) -> Result<SchedulerAttentionOutputSummary> {
    if output.values_per_row != GLM52_HIDDEN_SIZE {
        anyhow::bail!(
            "{context} expected hidden-width rows, got values_per_row={}",
            output.values_per_row
        );
    }
    let values = output
        .rows
        .checked_mul(output.values_per_row)
        .with_context(|| format!("{context} value count overflows usize"))?;
    let expected_bytes = values
        .checked_mul(std::mem::size_of::<u16>())
        .with_context(|| format!("{context} byte count overflows usize"))?;
    let buffer = output.buffer();
    if buffer.ptr.is_null() || buffer.bytes < expected_bytes {
        anyhow::bail!(
            "{context} device buffer is too small: bytes={} expected at least {expected_bytes}",
            buffer.bytes
        );
    }
    Ok(SchedulerAttentionOutputSummary {
        values,
        finite_values: 0,
        nonzero_values: 0,
        checksum: 0.0,
        output_bf16: Vec::new(),
    })
}

fn summarize_scheduler_bf16_device_buffer(
    library: &NativeLibrary,
    buffer: GlmrtDeviceBuffer,
    output_bytes: usize,
    context: &str,
) -> Result<SchedulerAttentionOutputSummary> {
    let mut output_bf16 = vec![0_u8; output_bytes];
    library
        .copy_d2h(&mut output_bf16, buffer)
        .with_context(|| format!("reading {context}"))?;
    let summary = summarize_scheduler_bf16_bytes(&output_bf16)?;
    validate_scheduler_attention_summary(
        summary.values,
        summary.finite_values,
        summary.nonzero_values,
        summary.checksum,
        context,
    )?;
    Ok(SchedulerAttentionOutputSummary {
        values: summary.values,
        finite_values: summary.finite_values,
        nonzero_values: summary.nonzero_values,
        checksum: summary.checksum,
        output_bf16,
    })
}

fn validate_scheduler_attention_summary(
    values: usize,
    finite_values: usize,
    nonzero_values: usize,
    checksum: f64,
    context: &str,
) -> Result<()> {
    if finite_values != values {
        anyhow::bail!(
            "{context} contains non-finite BF16 output values: finite={finite_values} total={values}"
        );
    }
    if nonzero_values == 0 {
        anyhow::bail!("{context} produced an all-zero BF16 output buffer");
    }
    if !checksum.is_finite() {
        anyhow::bail!("{context} checksum is non-finite");
    }
    Ok(())
}

fn summarize_scheduler_bf16_bytes(output_bf16: &[u8]) -> Result<SchedulerAttentionOutputSummary> {
    if output_bf16.len() % std::mem::size_of::<u16>() != 0 {
        anyhow::bail!(
            "scheduler device MLA attention output byte count {} is not BF16 aligned",
            output_bf16.len()
        );
    }
    let values = output_bf16.len() / std::mem::size_of::<u16>();
    let checksum_stride = values
        .checked_div(4096)
        .filter(|stride| *stride > 0)
        .unwrap_or(1);
    let mut finite_values = 0_usize;
    let mut nonzero_values = 0_usize;
    let mut checksum = 0.0_f64;
    for (index, chunk) in output_bf16
        .chunks_exact(std::mem::size_of::<u16>())
        .enumerate()
    {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        if bits & 0x7f80 != 0x7f80 {
            finite_values += 1;
        }
        if bits & 0x7fff != 0 {
            nonzero_values += 1;
        }
        if index % checksum_stride == 0 {
            checksum += f32::from_bits((bits as u32) << 16) as f64;
        }
    }
    Ok(SchedulerAttentionOutputSummary {
        values,
        finite_values,
        nonzero_values,
        checksum,
        output_bf16: Vec::new(),
    })
}

fn validate_device_kv_payloads(
    config: &KvCacheConfig,
    descriptors: &[KvBlockDescriptor],
    payloads: &[Vec<u8>],
    context: &str,
) -> Result<()> {
    if descriptors.len() != payloads.len() {
        anyhow::bail!(
            "{context} descriptor/payload mismatch: descriptors={} payloads={}",
            descriptors.len(),
            payloads.len()
        );
    }
    for (index, (descriptor, payload)) in descriptors.iter().zip(payloads).enumerate() {
        let expected_bytes = config
            .descriptor_payload_bytes(descriptor)
            .with_context(|| format!("{context} descriptor {index} has invalid payload size"))?;
        if payload.len() != expected_bytes {
            anyhow::bail!(
                "{context} payload {index} byte mismatch for layer={} token_start={} token_count={}: expected {} got {}",
                descriptor.layer_id.0,
                descriptor.token_start.0,
                descriptor.token_count,
                expected_bytes,
                payload.len()
            );
        }
    }
    Ok(())
}

#[cfg_attr(not(test), allow(dead_code))]
fn device_kv_library_unavailable(
    error: anyhow::Error,
    cuda_required: bool,
) -> Result<RealFullDeviceKvRoundTrip> {
    if cuda_required {
        return Err(error).context(
            "real-full device KV roundtrip requires CUDA reference execution but no CUDA-enabled native library is available",
        );
    }
    Ok(unavailable_device_kv_roundtrip())
}

#[cfg_attr(not(test), allow(dead_code))]
fn device_kv_roundtrip_failed(
    error: anyhow::Error,
    cuda_required: bool,
) -> Result<RealFullDeviceKvRoundTrip> {
    if cuda_required {
        return Err(error)
            .context("real-full device KV roundtrip failed with CUDA reference execution enabled");
    }
    Ok(RealFullDeviceKvRoundTrip {
        status: real_full_device_kv_roundtrip_error_status(&error),
        writes: 0,
        reads: 0,
        bytes: 0,
        uses_device_kv_cache: false,
    })
}

#[cfg_attr(not(test), allow(dead_code))]
fn unavailable_device_kv_roundtrip() -> RealFullDeviceKvRoundTrip {
    RealFullDeviceKvRoundTrip {
        status: "cuda-kv-cache-unavailable",
        writes: 0,
        reads: 0,
        bytes: 0,
        uses_device_kv_cache: false,
    }
}

pub(in crate::commands::real_full) fn real_full_device_kv_roundtrip_error_status(
    error: &anyhow::Error,
) -> &'static str {
    let message = format!("{error:#}").to_ascii_lowercase();
    if message.contains("returned status 3") || message.contains("cuda unavailable") {
        "cuda-kv-cache-unavailable"
    } else {
        "cuda-kv-cache-error"
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn execute_real_full_device_kv_roundtrip(
    library: &'static NativeLibrary,
    config: &KvCacheConfig,
    descriptors: &[KvBlockDescriptor],
    payloads: &[Vec<u8>],
) -> Result<RealFullDeviceKvRoundTrip> {
    let plan = DeviceKvRoundTripPlan::new(config, descriptors, payloads)?;
    let mut cache = RealFullDeviceKvCache::new(library, config.clone())
        .context("allocating real-full device KV roundtrip cache")?;
    let dst = DeviceBufferGuard::new(library, plan.expected_readback.len())
        .context("allocating real-full device KV roundtrip destination payload")?;
    let writes = cache
        .write_host_blocks_from_pinned_staging(&plan.write_descriptors, &plan.write_payloads)
        .context("writing real-full device KV roundtrip blocks")?;
    if writes.len() != plan.write_descriptors.len() {
        anyhow::bail!(
            "real-full device KV roundtrip canonical write count mismatch: expected {} got {}",
            plan.write_descriptors.len(),
            writes.len()
        );
    }
    let reads = cache
        .read_blocks_to_contiguous_device(descriptors, dst.buffer)
        .context("reading real-full device KV roundtrip blocks")?;
    let mut roundtrip = vec![0_u8; plan.expected_readback.len()];
    library.copy_d2h(&mut roundtrip, dst.buffer)?;
    if roundtrip != plan.expected_readback {
        anyhow::bail!(
            "real-full device KV roundtrip mismatch: bytes={}",
            plan.expected_readback.len()
        );
    }
    Ok(RealFullDeviceKvRoundTrip {
        status: "cuda-kv-cache-blocks-roundtrip",
        writes: descriptors.len(),
        reads: reads.len(),
        bytes: plan.expected_readback.len(),
        uses_device_kv_cache: true,
    })
}

#[cfg_attr(not(test), allow(dead_code))]
struct DeviceKvRoundTripPlan {
    write_descriptors: Vec<KvBlockDescriptor>,
    write_payloads: Vec<Vec<u8>>,
    write_bytes: usize,
    expected_readback: Vec<u8>,
}

#[cfg_attr(not(test), allow(dead_code))]
impl DeviceKvRoundTripPlan {
    fn new(
        config: &KvCacheConfig,
        read_descriptors: &[KvBlockDescriptor],
        payloads: &[Vec<u8>],
    ) -> Result<Self> {
        let mut touched_layers = BTreeMap::<u32, DeviceKvLayerRoundTripPlan>::new();
        for descriptor in read_descriptors {
            let io = real_full_device_kv_block_io(config, descriptor)?;
            io.offset_bytes
                .checked_add(io.payload_bytes)
                .context("device KV roundtrip descriptor range overflows usize")?;
            match touched_layers.get_mut(&descriptor.layer_id.0) {
                Some(layer) => layer.include(descriptor)?,
                None => {
                    touched_layers.insert(
                        descriptor.layer_id.0,
                        DeviceKvLayerRoundTripPlan::new(descriptor)?,
                    );
                }
            }
        }

        let mut write_descriptors = Vec::with_capacity(touched_layers.len());
        let mut layer_payloads = BTreeMap::<u32, (RealFullDeviceKvBlockIo, Vec<u8>)>::new();
        let mut write_bytes = 0_usize;
        for (layer_id, layer) in touched_layers {
            let descriptor = layer.descriptor()?;
            let io = real_full_device_kv_block_io(config, &descriptor)?;
            write_bytes = write_bytes
                .checked_add(io.payload_bytes)
                .context("device KV roundtrip canonical write bytes overflow usize")?;
            layer_payloads.insert(layer_id, (io, vec![0_u8; io.payload_bytes]));
            write_descriptors.push(descriptor);
        }

        for (descriptor, payload) in read_descriptors.iter().zip(payloads) {
            let io = real_full_device_kv_block_io(config, descriptor)?;
            let (layer_io, layer_payload) = layer_payloads
                .get_mut(&descriptor.layer_id.0)
                .context("device KV roundtrip missing touched layer payload")?;
            let offset = io
                .offset_bytes
                .checked_sub(layer_io.offset_bytes)
                .context("device KV roundtrip descriptor precedes canonical layer range")?;
            let end = offset
                .checked_add(io.payload_bytes)
                .context("device KV roundtrip layer payload range overflows usize")?;
            layer_payload[offset..end].copy_from_slice(payload);
        }

        let read_bytes = payloads
            .iter()
            .try_fold(0_usize, |acc, payload| acc.checked_add(payload.len()))
            .context("device KV roundtrip expected readback bytes overflow usize")?;
        let mut expected_readback = Vec::with_capacity(read_bytes);
        for descriptor in read_descriptors {
            let io = real_full_device_kv_block_io(config, descriptor)?;
            let (layer_io, layer_payload) = layer_payloads
                .get(&descriptor.layer_id.0)
                .context("device KV roundtrip missing expected readback layer payload")?;
            let offset = io
                .offset_bytes
                .checked_sub(layer_io.offset_bytes)
                .context("device KV roundtrip read precedes canonical layer range")?;
            let end = offset
                .checked_add(io.payload_bytes)
                .context("device KV roundtrip expected read range overflows usize")?;
            expected_readback.extend_from_slice(&layer_payload[offset..end]);
        }

        let mut write_payloads = Vec::with_capacity(write_descriptors.len());
        for descriptor in &write_descriptors {
            let (_, payload) = layer_payloads
                .remove(&descriptor.layer_id.0)
                .context("device KV roundtrip missing canonical write payload")?;
            write_payloads.push(payload);
        }

        Ok(Self {
            write_descriptors,
            write_payloads,
            write_bytes,
            expected_readback,
        })
    }
}

#[cfg_attr(not(test), allow(dead_code))]
struct DeviceKvLayerRoundTripPlan {
    reservation_id: u64,
    sequence_id: String,
    layer_id: glmrt_core::LayerId,
    token_start: u64,
    token_end: u64,
}

#[cfg_attr(not(test), allow(dead_code))]
impl DeviceKvLayerRoundTripPlan {
    fn new(descriptor: &KvBlockDescriptor) -> Result<Self> {
        let token_start = descriptor.token_start.0;
        let token_end = token_start
            .checked_add(descriptor.token_count as u64)
            .context("device KV roundtrip layer token range overflows u64")?;
        Ok(Self {
            reservation_id: descriptor.reservation_id,
            sequence_id: descriptor.sequence_id.clone(),
            layer_id: descriptor.layer_id,
            token_start,
            token_end,
        })
    }

    fn include(&mut self, descriptor: &KvBlockDescriptor) -> Result<()> {
        let token_start = descriptor.token_start.0;
        let token_end = token_start
            .checked_add(descriptor.token_count as u64)
            .context("device KV roundtrip layer token range overflows u64")?;
        self.token_start = self.token_start.min(token_start);
        self.token_end = self.token_end.max(token_end);
        Ok(())
    }

    fn descriptor(&self) -> Result<KvBlockDescriptor> {
        let token_count = self
            .token_end
            .checked_sub(self.token_start)
            .context("device KV roundtrip canonical token range is invalid")?;
        Ok(KvBlockDescriptor {
            reservation_id: self.reservation_id,
            sequence_id: self.sequence_id.clone(),
            layer_id: self.layer_id,
            token_start: glmrt_core::PositionId(self.token_start),
            token_count: usize::try_from(token_count)
                .context("device KV roundtrip canonical token count does not fit usize")?,
        })
    }
}

struct DeviceKvReusableHostBuffer<'a> {
    library: &'a NativeLibrary,
    buffer: GlmrtHostBuffer,
    capacity: usize,
    label: &'static str,
}

impl<'a> DeviceKvReusableHostBuffer<'a> {
    fn new(library: &'a NativeLibrary) -> Self {
        Self {
            library,
            buffer: GlmrtHostBuffer::default(),
            capacity: 0,
            label: "",
        }
    }

    fn buffer(&mut self, bytes: usize, label: &'static str) -> Result<GlmrtHostBuffer> {
        if bytes == 0 {
            anyhow::bail!("device KV pinned host staging buffer {label} requires nonzero bytes");
        }
        if !self.buffer.ptr.is_null() && self.capacity >= bytes {
            return Ok(self.buffer);
        }
        if !self.buffer.ptr.is_null() {
            let mut old = self.buffer;
            self.library.free_host_buffer(&mut old).with_context(|| {
                format!(
                    "freeing reusable device KV pinned host staging buffer {}",
                    self.label
                )
            })?;
            self.buffer = GlmrtHostBuffer::default();
            self.capacity = 0;
            self.label = "";
        }
        let mut buffer = self.library.alloc_host_buffer(bytes).with_context(|| {
            format!("allocating reusable device KV pinned host staging buffer {label}")
        })?;
        if buffer.ptr.is_null() {
            let _ = self.library.free_host_buffer(&mut buffer);
            anyhow::bail!("reusable device KV pinned host staging buffer {label} is null");
        }
        if buffer.bytes < bytes {
            let allocated = buffer.bytes;
            let _ = self.library.free_host_buffer(&mut buffer);
            anyhow::bail!(
                "reusable device KV pinned host staging buffer {label} allocated {} bytes, expected at least {bytes}",
                allocated
            );
        }
        self.capacity = buffer.bytes;
        self.buffer = buffer;
        self.label = label;
        Ok(self.buffer)
    }
}

impl Drop for DeviceKvReusableHostBuffer<'_> {
    fn drop(&mut self) {
        if !self.buffer.ptr.is_null() {
            let mut old = self.buffer;
            let _ = self.library.free_host_buffer(&mut old);
            self.buffer = GlmrtHostBuffer::default();
            self.capacity = 0;
        }
    }
}

struct DeviceKvReusableDeviceBuffer<'a> {
    library: &'a NativeLibrary,
    buffer: GlmrtDeviceBuffer,
    capacity: usize,
    label: &'static str,
}

impl<'a> DeviceKvReusableDeviceBuffer<'a> {
    fn new(library: &'a NativeLibrary) -> Self {
        Self {
            library,
            buffer: GlmrtDeviceBuffer::default(),
            capacity: 0,
            label: "",
        }
    }

    fn buffer(&mut self, bytes: usize, label: &'static str) -> Result<GlmrtDeviceBuffer> {
        if bytes == 0 {
            anyhow::bail!("device KV reusable device buffer {label} requires nonzero bytes");
        }
        if !self.buffer.ptr.is_null() && self.capacity >= bytes {
            return Ok(self.buffer);
        }
        if !self.buffer.ptr.is_null() {
            let mut old = self.buffer;
            self.library.free_device_buffer(&mut old).with_context(|| {
                format!("freeing reusable device KV device buffer {}", self.label)
            })?;
            self.buffer = GlmrtDeviceBuffer::default();
            self.capacity = 0;
            self.label = "";
        }
        let mut buffer = self
            .library
            .alloc_device_buffer(bytes)
            .with_context(|| format!("allocating reusable device KV device buffer {label}"))?;
        if buffer.ptr.is_null() {
            let _ = self.library.free_device_buffer(&mut buffer);
            anyhow::bail!("reusable device KV device buffer {label} is null");
        }
        if buffer.bytes < bytes {
            let allocated = buffer.bytes;
            let _ = self.library.free_device_buffer(&mut buffer);
            anyhow::bail!(
                "reusable device KV device buffer {label} allocated {} bytes, expected at least {bytes}",
                allocated
            );
        }
        self.capacity = buffer.bytes;
        self.buffer = buffer;
        self.label = label;
        Ok(self.buffer)
    }
}

impl Drop for DeviceKvReusableDeviceBuffer<'_> {
    fn drop(&mut self) {
        if !self.buffer.ptr.is_null() {
            let mut old = self.buffer;
            let _ = self.library.free_device_buffer(&mut old);
            self.buffer = GlmrtDeviceBuffer::default();
            self.capacity = 0;
        }
    }
}

enum DeviceKvBatchMetadataSlot {
    PayloadOffsets,
    CacheOffsets,
    BlockBytes,
}

enum DeviceKvAttentionHostUploadSlot {
    QueryNope,
    QueryRope,
    ProjectedQuery,
    SuffixKNope,
    SuffixKRope,
    SuffixValues,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DeviceKvAttentionReadyFrontier {
    reservation_id: u64,
    sequence_id: String,
    layer_id: LayerId,
    token_start: usize,
    rows: usize,
}

fn rewind_device_kv_attention_ready_frontier(
    slot: &mut Option<DeviceKvAttentionReadyFrontier>,
    reservation_id: u64,
    sequence_id: &str,
    layer_id: LayerId,
    token_start: usize,
) -> Result<()> {
    let Some(mut frontier) = slot.take() else {
        return Ok(());
    };
    if frontier.reservation_id == reservation_id
        && frontier.sequence_id == sequence_id
        && frontier.layer_id == layer_id
    {
        let frontier_end = frontier
            .token_start
            .checked_add(frontier.rows)
            .context("attention-ready rewind frontier end overflows usize")?;
        if frontier_end > token_start {
            frontier.rows = token_start.saturating_sub(frontier.token_start);
        }
    }
    if frontier.rows > 0 {
        *slot = Some(frontier);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TargetKvPhysicalSpan {
    logical_token_offset: usize,
    physical_token_start: usize,
    token_count: usize,
}

fn target_kv_physical_spans(
    physical_pages: Option<&[u32]>,
    physical_token_base: usize,
    logical_capacity_tokens: usize,
    logical_token_start: usize,
    token_count: usize,
) -> Result<Vec<TargetKvPhysicalSpan>> {
    anyhow::ensure!(token_count > 0, "target KV physical span is empty");
    let logical_token_end = logical_token_start
        .checked_add(token_count)
        .context("target KV logical span end overflow")?;
    anyhow::ensure!(
        logical_token_end <= logical_capacity_tokens,
        "target KV logical span [{logical_token_start}..{logical_token_end}) exceeds capacity {logical_capacity_tokens}"
    );
    let Some(physical_pages) = physical_pages else {
        return Ok(vec![TargetKvPhysicalSpan {
            logical_token_offset: 0,
            physical_token_start: physical_token_base
                .checked_add(logical_token_start)
                .context("target KV contiguous physical span start overflow")?,
            token_count,
        }]);
    };

    let mut spans = Vec::<TargetKvPhysicalSpan>::new();
    let mut logical_token = logical_token_start;
    while logical_token < logical_token_end {
        let logical_page = logical_token / GLMRT_CUDA_GLM_DSA_PAGE_SIZE;
        let page_token = logical_token % GLMRT_CUDA_GLM_DSA_PAGE_SIZE;
        let physical_page = *physical_pages.get(logical_page).with_context(|| {
            format!(
                "target KV logical page {logical_page} is not materialized; materialized={}",
                physical_pages.len()
            )
        })? as usize;
        let tokens_here =
            (GLMRT_CUDA_GLM_DSA_PAGE_SIZE - page_token).min(logical_token_end - logical_token);
        let physical_token_start = physical_page
            .checked_mul(GLMRT_CUDA_GLM_DSA_PAGE_SIZE)
            .and_then(|base| base.checked_add(page_token))
            .context("target KV paged physical span start overflow")?;
        let logical_token_offset = logical_token - logical_token_start;
        if let Some(previous) = spans.last_mut() {
            if previous
                .physical_token_start
                .checked_add(previous.token_count)
                == Some(physical_token_start)
                && previous
                    .logical_token_offset
                    .checked_add(previous.token_count)
                    == Some(logical_token_offset)
            {
                previous.token_count = previous
                    .token_count
                    .checked_add(tokens_here)
                    .context("target KV coalesced physical span length overflow")?;
                logical_token += tokens_here;
                continue;
            }
        }
        spans.push(TargetKvPhysicalSpan {
            logical_token_offset,
            physical_token_start,
            token_count: tokens_here,
        });
        logical_token += tokens_here;
    }
    Ok(spans)
}

pub(in crate::commands::real_full) struct RealFullDeviceKvCache<'a> {
    library: &'a NativeLibrary,
    config: KvCacheConfig,
    storage: Arc<RealFullDeviceKvStorage<'a>>,
    physical_token_base: usize,
    physical_pages: Option<Vec<u32>>,
    physical_page_table_key: u64,
    logical_capacity_tokens: usize,
    host_write_staging: DeviceKvReusableHostBuffer<'a>,
    device_write_source: DeviceKvReusableDeviceBuffer<'a>,
    batch_metadata_staging: DeviceKvReusableHostBuffer<'a>,
    rope_position_staging: DeviceKvReusableHostBuffer<'a>,
    scheduler_upload_staging: DeviceKvReusableHostBuffer<'a>,
    attention_host_upload_staging: DeviceKvReusableHostBuffer<'a>,
    mla_write_positions: Vec<u32>,
    rope_positions_device: DeviceKvReusableDeviceBuffer<'a>,
    physical_positions_device: DeviceKvReusableDeviceBuffer<'a>,
    physical_page_table_device: DeviceKvReusableDeviceBuffer<'a>,
    scheduler_projected_query: DeviceKvReusableDeviceBuffer<'a>,
    attention_query_nope: DeviceKvReusableDeviceBuffer<'a>,
    attention_query_rope: DeviceKvReusableDeviceBuffer<'a>,
    attention_projected_query: DeviceKvReusableDeviceBuffer<'a>,
    attention_query_split_unrotated_rope: DeviceKvReusableDeviceBuffer<'a>,
    attention_query_split_nope: DeviceKvReusableDeviceBuffer<'a>,
    attention_query_split_rope_rotated: DeviceKvReusableDeviceBuffer<'a>,
    attention_projected_kv_normalized: DeviceKvReusableDeviceBuffer<'a>,
    attention_projected_kv_projected: DeviceKvReusableDeviceBuffer<'a>,
    attention_projected_kv_k_nope: DeviceKvReusableDeviceBuffer<'a>,
    attention_projected_kv_values: DeviceKvReusableDeviceBuffer<'a>,
    attention_suffix_projected_kv_normalized: DeviceKvReusableDeviceBuffer<'a>,
    attention_suffix_projected_kv_projected: DeviceKvReusableDeviceBuffer<'a>,
    attention_suffix_k_nope: DeviceKvReusableDeviceBuffer<'a>,
    attention_suffix_k_rope: DeviceKvReusableDeviceBuffer<'a>,
    attention_suffix_values: DeviceKvReusableDeviceBuffer<'a>,
    attention_combined_k_nope: DeviceKvReusableDeviceBuffer<'a>,
    attention_combined_k_rope: DeviceKvReusableDeviceBuffer<'a>,
    attention_combined_values: DeviceKvReusableDeviceBuffer<'a>,
    payload_offsets_device: DeviceKvReusableDeviceBuffer<'a>,
    cache_offsets_device: DeviceKvReusableDeviceBuffer<'a>,
    block_bytes_device: DeviceKvReusableDeviceBuffer<'a>,
    mla_prepared_write_payload: DeviceKvReusableDeviceBuffer<'a>,
    mla_attention_ready_frontier_payload: DeviceKvReusableDeviceBuffer<'a>,
    mla_attention_ready_frontiers: Vec<Option<DeviceKvAttentionReadyFrontier>>,
    mla_fp8_packed_write_payload: DeviceKvReusableDeviceBuffer<'a>,
    mla_mxfp4_packed_write_payload: DeviceKvReusableDeviceBuffer<'a>,
    mla_fp8_unpacked_projected: DeviceKvReusableDeviceBuffer<'a>,
    mla_read_payload: DeviceKvReusableDeviceBuffer<'a>,
    mla_unpacked_kv_latent: DeviceKvReusableDeviceBuffer<'a>,
    mla_unpacked_k_rope: DeviceKvReusableDeviceBuffer<'a>,
    mla_unpacked_dsa_key: DeviceKvReusableDeviceBuffer<'a>,
    mla_current_kv_latent: DeviceKvReusableDeviceBuffer<'a>,
    mla_current_k_rope: DeviceKvReusableDeviceBuffer<'a>,
    attention_k_rope_rotated: DeviceKvReusableDeviceBuffer<'a>,
    attention_output: DeviceKvReusableDeviceBuffer<'a>,
    host_readback_payload: DeviceKvReusableDeviceBuffer<'a>,
}

pub(in crate::commands::real_full) type RealFullDeviceKvStorageHandle =
    Arc<RealFullDeviceKvStorage<'static>>;

pub(in crate::commands::real_full) struct RealFullDeviceKvStorage<'a> {
    library: &'a NativeLibrary,
    config: KvCacheConfig,
    cache: GlmrtDeviceBuffer,
    dsa_index_k_cache_b12x: GlmrtDeviceBuffer,
}

// Device pointers are immutable allocation handles. Execution remains pinned
// to the owning coordinator worker; sharing this object shares ownership of
// the allocation, not concurrent mutable CUDA access.
unsafe impl Send for RealFullDeviceKvStorage<'_> {}
unsafe impl Sync for RealFullDeviceKvStorage<'_> {}

impl<'a> RealFullDeviceKvStorage<'a> {
    fn new(library: &'a NativeLibrary, config: KvCacheConfig) -> Result<Self> {
        let capacity_bytes = config.capacity_bytes();
        if capacity_bytes == 0 {
            anyhow::bail!("device KV cache capacity must be nonzero");
        }
        let dsa_index_k_cache_bytes = real_full_device_dsa_index_k_b12x_capacity_bytes(&config)?;
        let cache = library
            .alloc_device_buffer(capacity_bytes)
            .with_context(|| format!("allocating {} byte device KV cache", capacity_bytes))?;
        let dsa_index_k_cache_b12x = if dsa_index_k_cache_bytes == 0 {
            GlmrtDeviceBuffer::default()
        } else {
            match library.alloc_device_buffer(dsa_index_k_cache_bytes) {
                Ok(buffer) => buffer,
                Err(error) => {
                    let mut cache = cache;
                    let _ = library.free_device_buffer(&mut cache);
                    return Err(error).with_context(|| {
                        format!(
                            "allocating {dsa_index_k_cache_bytes} byte direct B12X DSA index-K cache"
                        )
                    });
                }
            }
        };
        Ok(Self {
            library,
            config,
            cache,
            dsa_index_k_cache_b12x,
        })
    }
}

impl<'a> RealFullDeviceKvCache<'a> {
    pub(in crate::commands::real_full) fn new(
        library: &'a NativeLibrary,
        config: KvCacheConfig,
    ) -> Result<Self> {
        let storage = Arc::new(RealFullDeviceKvStorage::new(library, config.clone())?);
        let logical_capacity_tokens = config.max_tokens;
        Self::new_with_storage(library, config, storage, 0, logical_capacity_tokens)
    }

    fn new_with_storage(
        library: &'a NativeLibrary,
        config: KvCacheConfig,
        storage: Arc<RealFullDeviceKvStorage<'a>>,
        physical_token_base: usize,
        logical_capacity_tokens: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            storage.config == config,
            "shared device KV storage config does not match the execution cache config"
        );
        anyhow::ensure!(
            logical_capacity_tokens > 0
                && physical_token_base
                    .checked_add(logical_capacity_tokens)
                    .is_some_and(|end| end <= config.max_tokens),
            "shared device KV extent [{physical_token_base}..+{logical_capacity_tokens}) exceeds {} physical tokens",
            config.max_tokens
        );
        let mla_attention_ready_frontiers = vec![None; config.layers];
        let physical_page_table_bytes = config
            .max_tokens
            .checked_add(GLMRT_CUDA_GLM_DSA_PAGE_SIZE - 1)
            .context("device target KV page-table page count overflow")?
            / GLMRT_CUDA_GLM_DSA_PAGE_SIZE
            * std::mem::size_of::<u32>();
        let mut physical_page_table_device = DeviceKvReusableDeviceBuffer::new(library);
        let physical_page_table_buffer = physical_page_table_device
            .buffer(
                physical_page_table_bytes,
                "device target KV physical page table",
            )
            .context("allocating stable target KV physical page table")?;
        library
            .cuda_glm_dsa_page_table_init_base(
                physical_page_table_buffer,
                1,
                physical_page_table_bytes / std::mem::size_of::<u32>(),
                0,
            )
            .context("initializing safe target KV physical page-table suffix")?;
        Ok(Self {
            library,
            config,
            storage,
            physical_token_base,
            physical_pages: None,
            physical_page_table_key: 0,
            logical_capacity_tokens,
            host_write_staging: DeviceKvReusableHostBuffer::new(library),
            device_write_source: DeviceKvReusableDeviceBuffer::new(library),
            batch_metadata_staging: DeviceKvReusableHostBuffer::new(library),
            rope_position_staging: DeviceKvReusableHostBuffer::new(library),
            scheduler_upload_staging: DeviceKvReusableHostBuffer::new(library),
            attention_host_upload_staging: DeviceKvReusableHostBuffer::new(library),
            mla_write_positions: Vec::new(),
            rope_positions_device: DeviceKvReusableDeviceBuffer::new(library),
            physical_positions_device: DeviceKvReusableDeviceBuffer::new(library),
            physical_page_table_device,
            scheduler_projected_query: DeviceKvReusableDeviceBuffer::new(library),
            attention_query_nope: DeviceKvReusableDeviceBuffer::new(library),
            attention_query_rope: DeviceKvReusableDeviceBuffer::new(library),
            attention_projected_query: DeviceKvReusableDeviceBuffer::new(library),
            attention_query_split_unrotated_rope: DeviceKvReusableDeviceBuffer::new(library),
            attention_query_split_nope: DeviceKvReusableDeviceBuffer::new(library),
            attention_query_split_rope_rotated: DeviceKvReusableDeviceBuffer::new(library),
            attention_projected_kv_normalized: DeviceKvReusableDeviceBuffer::new(library),
            attention_projected_kv_projected: DeviceKvReusableDeviceBuffer::new(library),
            attention_projected_kv_k_nope: DeviceKvReusableDeviceBuffer::new(library),
            attention_projected_kv_values: DeviceKvReusableDeviceBuffer::new(library),
            attention_suffix_projected_kv_normalized: DeviceKvReusableDeviceBuffer::new(library),
            attention_suffix_projected_kv_projected: DeviceKvReusableDeviceBuffer::new(library),
            attention_suffix_k_nope: DeviceKvReusableDeviceBuffer::new(library),
            attention_suffix_k_rope: DeviceKvReusableDeviceBuffer::new(library),
            attention_suffix_values: DeviceKvReusableDeviceBuffer::new(library),
            attention_combined_k_nope: DeviceKvReusableDeviceBuffer::new(library),
            attention_combined_k_rope: DeviceKvReusableDeviceBuffer::new(library),
            attention_combined_values: DeviceKvReusableDeviceBuffer::new(library),
            payload_offsets_device: DeviceKvReusableDeviceBuffer::new(library),
            cache_offsets_device: DeviceKvReusableDeviceBuffer::new(library),
            block_bytes_device: DeviceKvReusableDeviceBuffer::new(library),
            mla_prepared_write_payload: DeviceKvReusableDeviceBuffer::new(library),
            mla_attention_ready_frontier_payload: DeviceKvReusableDeviceBuffer::new(library),
            mla_attention_ready_frontiers,
            mla_fp8_packed_write_payload: DeviceKvReusableDeviceBuffer::new(library),
            mla_mxfp4_packed_write_payload: DeviceKvReusableDeviceBuffer::new(library),
            mla_fp8_unpacked_projected: DeviceKvReusableDeviceBuffer::new(library),
            mla_read_payload: DeviceKvReusableDeviceBuffer::new(library),
            mla_unpacked_kv_latent: DeviceKvReusableDeviceBuffer::new(library),
            mla_unpacked_k_rope: DeviceKvReusableDeviceBuffer::new(library),
            mla_unpacked_dsa_key: DeviceKvReusableDeviceBuffer::new(library),
            mla_current_kv_latent: DeviceKvReusableDeviceBuffer::new(library),
            mla_current_k_rope: DeviceKvReusableDeviceBuffer::new(library),
            attention_k_rope_rotated: DeviceKvReusableDeviceBuffer::new(library),
            attention_output: DeviceKvReusableDeviceBuffer::new(library),
            host_readback_payload: DeviceKvReusableDeviceBuffer::new(library),
        })
    }

    pub(in crate::commands::real_full) fn config(&self) -> &KvCacheConfig {
        &self.config
    }

    fn rebind_physical_pages(
        &mut self,
        physical_pages: &[u32],
        logical_capacity_tokens: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            logical_capacity_tokens > 0 && logical_capacity_tokens <= self.config.max_tokens,
            "device KV paged logical capacity {logical_capacity_tokens} exceeds {} physical tokens",
            self.config.max_tokens
        );
        let maximum_page_count = self
            .config
            .max_tokens
            .checked_add(GLMRT_CUDA_GLM_DSA_PAGE_SIZE - 1)
            .context("device KV maximum physical page count overflow")?
            / GLMRT_CUDA_GLM_DSA_PAGE_SIZE;
        let logical_capacity_pages = logical_capacity_tokens
            .checked_add(GLMRT_CUDA_GLM_DSA_PAGE_SIZE - 1)
            .context("device KV logical capacity page count overflow")?
            / GLMRT_CUDA_GLM_DSA_PAGE_SIZE;
        anyhow::ensure!(
            !physical_pages.is_empty() && physical_pages.len() <= logical_capacity_pages,
            "device KV page table has {} materialized pages for logical capacity {logical_capacity_tokens}",
            physical_pages.len()
        );
        anyhow::ensure!(
            physical_pages
                .iter()
                .all(|page| (*page as usize) < maximum_page_count),
            "device KV page table contains a page outside the {maximum_page_count}-page pool"
        );
        let bytes = u32_slice_bytes(physical_pages);
        anyhow::ensure!(
            self.physical_page_table_device.capacity >= bytes.len(),
            "stable device KV page-table allocation has {} bytes, needs {}",
            self.physical_page_table_device.capacity,
            bytes.len()
        );
        self.library
            .copy_h2d(self.physical_page_table_device.buffer, bytes)
            .context("uploading target KV physical page table")?;
        self.physical_pages = Some(physical_pages.to_vec());
        self.logical_capacity_tokens = logical_capacity_tokens;
        self.physical_page_table_key = next_physical_page_table_key();
        Ok(())
    }

    fn physical_token_position(&self, logical_token: usize) -> Result<usize> {
        anyhow::ensure!(
            logical_token < self.logical_capacity_tokens,
            "logical KV token {logical_token} exceeds sequence extent {}",
            self.logical_capacity_tokens
        );
        if let Some(physical_pages) = self.physical_pages.as_ref() {
            let logical_page = logical_token / GLMRT_CUDA_GLM_DSA_PAGE_SIZE;
            let physical_page = physical_pages.get(logical_page).with_context(|| {
                format!(
                    "logical KV token {logical_token} uses unmaterialized page {logical_page}; materialized={}",
                    physical_pages.len()
                )
            })?;
            return (*physical_page as usize)
                .checked_mul(GLMRT_CUDA_GLM_DSA_PAGE_SIZE)
                .and_then(|base| base.checked_add(logical_token % GLMRT_CUDA_GLM_DSA_PAGE_SIZE))
                .context("physical KV token position overflow");
        }
        self.physical_token_base
            .checked_add(logical_token)
            .context("physical KV token position overflow")
    }

    fn physical_page_table(&self) -> Option<(GlmrtDeviceBuffer, u64)> {
        self.physical_pages.as_ref().map(|_| {
            (
                self.physical_page_table_device.buffer,
                self.physical_page_table_key,
            )
        })
    }

    fn copy_target_kv_boundary_page(
        &mut self,
        source_page: u32,
        destination_page: u32,
        valid_tokens: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            source_page != destination_page
                && (1..GLMRT_CUDA_GLM_DSA_PAGE_SIZE).contains(&valid_tokens),
            "target KV radix boundary copy is invalid: source={source_page} destination={destination_page} valid_tokens={valid_tokens}"
        );
        let maximum_pages = self
            .config
            .max_tokens
            .checked_add(GLMRT_CUDA_GLM_DSA_PAGE_SIZE - 1)
            .context("target KV radix boundary maximum page count overflow")?
            / GLMRT_CUDA_GLM_DSA_PAGE_SIZE;
        anyhow::ensure!(
            (source_page as usize) < maximum_pages
                && (destination_page as usize) < maximum_pages,
            "target KV radix boundary pages source={source_page} destination={destination_page} exceed {maximum_pages} pages"
        );
        let copy_plane = |library: &NativeLibrary,
                          storage: GlmrtDeviceBuffer,
                          source: RealFullDeviceKvBlockIo,
                          destination: RealFullDeviceKvBlockIo,
                          label: &str|
         -> Result<()> {
            anyhow::ensure!(
                source.payload_bytes == destination.payload_bytes,
                "{label} source/destination byte counts differ"
            );
            let source =
                device_buffer_byte_view(storage, source.offset_bytes, source.payload_bytes, label)?;
            let destination = device_buffer_byte_view(
                storage,
                destination.offset_bytes,
                destination.payload_bytes,
                label,
            )?;
            library
                .copy_d2d(destination, source, source.bytes)
                .with_context(|| format!("copying {label}"))
        };
        let page_token = |page: u32| -> Result<PositionId> {
            (page as u64)
                .checked_mul(GLMRT_CUDA_GLM_DSA_PAGE_SIZE as u64)
                .map(PositionId)
                .context("target KV radix boundary token offset overflow")
        };
        for layer in 0..self.config.layers {
            let source_descriptor = KvBlockDescriptor {
                reservation_id: 0,
                sequence_id: "target-kv-radix-boundary-source".to_owned(),
                layer_id: LayerId(layer as u32),
                token_start: page_token(source_page)?,
                token_count: valid_tokens,
            };
            let destination_descriptor = KvBlockDescriptor {
                token_start: page_token(destination_page)?,
                sequence_id: "target-kv-radix-boundary-destination".to_owned(),
                ..source_descriptor.clone()
            };
            copy_plane(
                self.library,
                self.storage.cache,
                real_full_device_main_kv_block_io(&self.config, &source_descriptor)?,
                real_full_device_main_kv_block_io(&self.config, &destination_descriptor)?,
                "target KV radix main page",
            )?;
            if let (Some(source), Some(destination)) = (
                real_full_device_dsa_bf16_block_io(&self.config, &source_descriptor)?,
                real_full_device_dsa_bf16_block_io(&self.config, &destination_descriptor)?,
            ) {
                copy_plane(
                    self.library,
                    self.storage.cache,
                    source,
                    destination,
                    "target KV radix compatibility DSA page",
                )?;
            }
        }
        let packed_page_bytes = GLMRT_CUDA_GLM_DSA_PACKED_PAGE_BYTES;
        let packed_layer_bytes = real_full_device_dsa_index_k_b12x_layer_bytes(&self.config)?;
        for layer_index in 0..self.config.dsa_indexer_layer_ids().len() {
            let layer_base = layer_index
                .checked_mul(packed_layer_bytes)
                .context("target KV radix packed DSA layer offset overflow")?;
            let source_offset = (source_page as usize)
                .checked_mul(packed_page_bytes)
                .and_then(|offset| layer_base.checked_add(offset))
                .context("target KV radix packed DSA source offset overflow")?;
            let destination_offset = (destination_page as usize)
                .checked_mul(packed_page_bytes)
                .and_then(|offset| layer_base.checked_add(offset))
                .context("target KV radix packed DSA destination offset overflow")?;
            copy_plane(
                self.library,
                self.storage.dsa_index_k_cache_b12x,
                RealFullDeviceKvBlockIo {
                    offset_bytes: source_offset,
                    payload_bytes: packed_page_bytes,
                },
                RealFullDeviceKvBlockIo {
                    offset_bytes: destination_offset,
                    payload_bytes: packed_page_bytes,
                },
                "target KV radix packed DSA page",
            )?;
        }
        Ok(())
    }

    fn physical_descriptor_spans(
        &self,
        descriptor: &KvBlockDescriptor,
    ) -> Result<Vec<(KvBlockDescriptor, usize)>> {
        let token_start = usize::try_from(descriptor.token_start.0)
            .context("logical KV token start does not fit usize")?;
        let spans = target_kv_physical_spans(
            self.physical_pages.as_deref(),
            self.physical_token_base,
            self.logical_capacity_tokens,
            token_start,
            descriptor.token_count,
        )?;
        spans
            .into_iter()
            .map(|span| {
                let mut physical = descriptor.clone();
                physical.token_start = PositionId(
                    u64::try_from(span.physical_token_start)
                        .context("physical KV span start does not fit u64")?,
                );
                physical.token_count = span.token_count;
                Ok((physical, span.logical_token_offset))
            })
            .collect()
    }

    fn physical_descriptor(&self, descriptor: &KvBlockDescriptor) -> Result<KvBlockDescriptor> {
        let token_start = usize::try_from(descriptor.token_start.0)
            .context("logical KV token start does not fit usize")?;
        let token_end = token_start
            .checked_add(descriptor.token_count)
            .context("logical KV token range overflows usize")?;
        anyhow::ensure!(
            token_end <= self.logical_capacity_tokens,
            "logical KV token range [{token_start}..{token_end}) exceeds sequence extent {}",
            self.logical_capacity_tokens
        );
        let physical_token_start = self.physical_token_position(token_start)?;
        if descriptor.token_count > 1 {
            let physical_token_end = self.physical_token_position(token_end - 1)?;
            anyhow::ensure!(
                physical_token_end.checked_add(1)
                    == physical_token_start.checked_add(descriptor.token_count),
                "logical KV block [{token_start}..{token_end}) is not physically contiguous"
            );
        }
        let mut physical = descriptor.clone();
        physical.token_start = PositionId(
            u64::try_from(physical_token_start)
                .context("physical KV token start does not fit u64")?,
        );
        Ok(physical)
    }

    fn physical_main_kv_block_io(
        &self,
        descriptor: &KvBlockDescriptor,
    ) -> Result<RealFullDeviceKvBlockIo> {
        real_full_device_main_kv_block_io(&self.config, &self.physical_descriptor(descriptor)?)
    }

    fn physical_dsa_bf16_block_io(
        &self,
        descriptor: &KvBlockDescriptor,
    ) -> Result<Option<RealFullDeviceKvBlockIo>> {
        real_full_device_dsa_bf16_block_io(&self.config, &self.physical_descriptor(descriptor)?)
    }

    fn logical_batch_plan(&self, descriptors: &[KvBlockDescriptor]) -> Result<DeviceKvBatchPlan> {
        let ios = descriptors
            .iter()
            .map(|descriptor| real_full_device_kv_block_io(&self.config, descriptor))
            .collect::<Result<Vec<_>>>()?;
        DeviceKvBatchPlan::from_descriptors_and_ios(descriptors.to_vec(), ios)
    }

    fn contiguous_physical_main_kv_block_span(
        &self,
        descriptors: &[KvBlockDescriptor],
    ) -> Result<Option<(usize, usize, Vec<RealFullDeviceKvBlockIo>)>> {
        if self.physical_pages.is_some() {
            return Ok(None);
        }
        let physical = descriptors
            .iter()
            .map(|descriptor| self.physical_descriptor(descriptor))
            .collect::<Result<Vec<_>>>()?;
        contiguous_device_main_kv_block_span(&self.config, &physical)
    }

    fn read_main_blocks_to_contiguous_device(
        &mut self,
        descriptors: &[KvBlockDescriptor],
        dst: GlmrtDeviceBuffer,
    ) -> Result<Vec<RealFullDeviceKvBlockIo>> {
        let main_row_bytes = real_full_device_main_mla_row_bytes(&self.config)?;
        let total_rows = descriptors.iter().try_fold(0_usize, |rows, descriptor| {
            rows.checked_add(descriptor.token_count)
                .context("device main KV gather row count overflow")
        })?;
        let total_bytes = total_rows
            .checked_mul(main_row_bytes)
            .context("device main KV gather byte count overflow")?;
        validate_contiguous_payload_buffer("device main KV gather destination", dst, total_bytes)?;

        let mut destination_row = 0_usize;
        let mut logical_reads = Vec::with_capacity(descriptors.len());
        for descriptor in descriptors {
            logical_reads.push(real_full_device_main_kv_block_io(&self.config, descriptor)?);
            for (physical, logical_token_offset) in self.physical_descriptor_spans(descriptor)? {
                let source = real_full_device_main_kv_block_io(&self.config, &physical)?;
                let span_destination_row = destination_row
                    .checked_add(logical_token_offset)
                    .context("device main KV gather destination row overflow")?;
                let destination_offset = span_destination_row
                    .checked_mul(main_row_bytes)
                    .context("device main KV gather destination offset overflow")?;
                self.library
                    .copy_d2d(
                        device_buffer_byte_view(
                            dst,
                            destination_offset,
                            source.payload_bytes,
                            "device main KV gather destination span",
                        )?,
                        device_buffer_byte_view(
                            self.storage.cache,
                            source.offset_bytes,
                            source.payload_bytes,
                            "device main KV gather source span",
                        )?,
                        source.payload_bytes,
                    )
                    .context("gathering paged device main KV span")?;
            }
            destination_row = destination_row
                .checked_add(descriptor.token_count)
                .context("device main KV gather row frontier overflow")?;
        }
        Ok(logical_reads)
    }

    fn contiguous_physical_kv_block_span(
        &self,
        descriptors: &[KvBlockDescriptor],
    ) -> Result<Option<(usize, usize, Vec<RealFullDeviceKvBlockIo>)>> {
        if self.physical_pages.is_some() {
            return Ok(None);
        }
        let physical = descriptors
            .iter()
            .map(|descriptor| self.physical_descriptor(descriptor))
            .collect::<Result<Vec<_>>>()?;
        contiguous_device_kv_block_span(&self.config, &physical)
    }

    fn main_mla_cache_for_layer(&self, layer_id: LayerId) -> Result<GlmrtDeviceBuffer> {
        let layer_base = self
            .config
            .layer_base_offset_bytes(layer_id)
            .context("device main MLA layer base is invalid")?;
        let layer_bytes = real_full_device_main_mla_row_bytes(&self.config)?
            .checked_mul(self.config.max_tokens)
            .context("device main MLA layer cache bytes overflow usize")?;
        device_buffer_byte_view(
            self.storage.cache,
            layer_base,
            layer_bytes,
            "device main MLA layer cache",
        )
    }

    pub(in crate::commands::real_full) fn dsa_index_k_cache_b12x_for_layer(
        &self,
        layer_id: LayerId,
    ) -> Result<Option<GlmrtDeviceBuffer>> {
        let Some(layer_index) = self
            .config
            .dsa_indexer_layer_ids()
            .iter()
            .position(|configured| *configured == layer_id.0 as usize)
        else {
            return Ok(None);
        };
        anyhow::ensure!(
            !self.storage.dsa_index_k_cache_b12x.ptr.is_null(),
            "device B12X DSA cache is missing for configured layer {}",
            layer_id.0
        );
        let layer_bytes = real_full_device_dsa_index_k_b12x_layer_bytes(&self.config)?;
        let offset_bytes = layer_index
            .checked_mul(layer_bytes)
            .context("device B12X DSA cache layer offset overflow usize")?;
        device_buffer_byte_view(
            self.storage.dsa_index_k_cache_b12x,
            offset_bytes,
            layer_bytes,
            "device B12X DSA index-K layer cache",
        )
        .map(Some)
    }

    fn write_dsa_index_k_cache_b12x(
        &mut self,
        descriptors: &[KvBlockDescriptor],
        normalized_k: GlmrtDeviceBuffer,
        normalized_stride_bytes: usize,
    ) -> Result<()> {
        let Some(first) = descriptors.first() else {
            anyhow::bail!("device B12X DSA cache write requires descriptors");
        };
        anyhow::ensure!(
            descriptors
                .iter()
                .all(|descriptor| descriptor.layer_id == first.layer_id),
            "device B12X DSA cache write requires one layer"
        );
        let index_k_cache = self
            .dsa_index_k_cache_b12x_for_layer(first.layer_id)?
            .context("device B12X DSA cache write requires an indexer layer")?;
        let mut positions = std::mem::take(&mut self.mla_write_positions);
        let result = (|| {
            descriptor_positions_u32_into(descriptors, &mut positions)?;
            let logical_positions_device = self
                .stage_rope_positions_u32(&positions, "B12X DSA index-K cache write")
                .context("staging B12X DSA index-K logical positions")?;
            let physical_positions = positions
                .iter()
                .map(|position| {
                    let logical = usize::try_from(*position)
                        .context("B12X DSA logical position exceeds usize")?;
                    u32::try_from(self.physical_token_position(logical)?)
                        .context("B12X DSA physical position exceeds u32")
                })
                .collect::<Result<Vec<_>>>()?;
            let physical_positions_device = self
                .stage_physical_positions_u32(&physical_positions, "B12X DSA index-K cache write")
                .context("staging B12X DSA index-K physical slots")?;
            self.library
                .cuda_glm_dsa_index_k_pack_b12x(
                    normalized_k,
                    logical_positions_device,
                    physical_positions_device,
                    index_k_cache,
                    positions.len(),
                    self.config.max_tokens,
                    normalized_stride_bytes,
                    GLM52_MLA_ROPE_THETA,
                )
                .context("packing normalized DSA keys directly into B12X cache")
        })();
        self.mla_write_positions = positions;
        result
    }

    fn prepare_projected_mla_kv_for_cache(
        &mut self,
        descriptors: &[KvBlockDescriptor],
        projected: GlmrtDeviceBuffer,
        norm_weight: GlmrtDeviceBuffer,
        row_stride_bytes: usize,
    ) -> Result<GlmrtDeviceBuffer> {
        let mut positions = std::mem::take(&mut self.mla_write_positions);
        let result = (|| {
            descriptor_positions_u32_into(descriptors, &mut positions)?;
            let positions_device = self
                .stage_rope_positions_u32(&positions, "MLA KV cache write")
                .context("staging MLA KV cache write positions")?;
            let prepared_bytes = positions
                .len()
                .checked_mul(row_stride_bytes)
                .context("prepared MLA KV cache write bytes overflow usize")?;
            let prepared = self
                .mla_prepared_write_payload
                .buffer(prepared_bytes, "prepared MLA KV cache write payload")?;
            let frontier_target = if self.config.dtype == KvCacheDType::Nvfp4 {
                None
            } else {
                let row_stride = attention_ready_frontier_row_stride_bytes(self.config.dtype)?;
                self.attention_ready_mla_frontier_append_target(descriptors, row_stride)?
            };
            let frontier_index = usize::try_from(descriptors[0].layer_id.0)
                .context("attention-ready MLA frontier layer exceeds usize")?;
            if frontier_index < self.mla_attention_ready_frontiers.len() {
                self.mla_attention_ready_frontiers[frontier_index] = None;
            }
            mla_kv_prepare_bf16_device_buffers_for_layer(
                descriptors[0].layer_id.0 as usize,
                projected,
                positions_device,
                norm_weight,
                prepared,
                positions.len(),
                row_stride_bytes,
                row_stride_bytes,
                REAL_FULL_SCHEDULER_DEVICE_ATTENTION_EPS,
                GLM52_MLA_ROPE_THETA,
            )
            .context("normalizing and rotating projected MLA KV cache rows")?;
            if let Some((frontier_payload, next_frontier)) = frontier_target {
                match self.config.dtype {
                    KvCacheDType::Bf16 | KvCacheDType::Fp8 => self
                        .library
                        .copy_d2d(frontier_payload, prepared, prepared_bytes)
                        .context("retaining attention-ready BF16 MLA frontier")?,
                    KvCacheDType::Nvfp4 => {
                        unreachable!("native NVFP4 MLA does not materialize a BF16/FP8 frontier")
                    }
                    _ => unreachable!("validated attention-ready MLA frontier dtype"),
                }
                self.mla_attention_ready_frontiers[frontier_index] = Some(next_frontier);
            }
            Ok(prepared)
        })();
        self.mla_write_positions = positions;
        result
    }

    fn attention_ready_mla_frontier_append_target(
        &mut self,
        descriptors: &[KvBlockDescriptor],
        row_stride_bytes: usize,
    ) -> Result<Option<(GlmrtDeviceBuffer, DeviceKvAttentionReadyFrontier)>> {
        let Some(first) = descriptors.first() else {
            return Ok(None);
        };
        let layer_index = usize::try_from(first.layer_id.0)
            .context("attention-ready MLA frontier layer exceeds usize")?;
        anyhow::ensure!(
            layer_index < self.mla_attention_ready_frontiers.len(),
            "attention-ready MLA frontier layer {layer_index} exceeds configured layers {}",
            self.mla_attention_ready_frontiers.len()
        );
        let token_start = usize::try_from(first.token_start.0)
            .context("attention-ready MLA frontier token start exceeds usize")?;
        let mut expected_token_start = token_start;
        let mut rows = 0_usize;
        for descriptor in descriptors {
            if descriptor.reservation_id != first.reservation_id
                || descriptor.sequence_id != first.sequence_id
                || descriptor.layer_id != first.layer_id
            {
                return Ok(None);
            }
            let descriptor_token_start = usize::try_from(descriptor.token_start.0)
                .context("attention-ready MLA frontier descriptor token start exceeds usize")?;
            if descriptor_token_start != expected_token_start {
                return Ok(None);
            }
            expected_token_start = expected_token_start
                .checked_add(descriptor.token_count)
                .context("attention-ready MLA frontier token range overflows usize")?;
            rows = rows
                .checked_add(descriptor.token_count)
                .context("attention-ready MLA frontier row count overflows usize")?;
        }
        if rows == 0 {
            return Ok(None);
        }
        let target_bytes = rows
            .checked_mul(row_stride_bytes)
            .context("attention-ready MLA frontier target bytes overflow usize")?;

        let append = self.mla_attention_ready_frontiers[layer_index]
            .as_ref()
            .is_some_and(|frontier| {
                frontier.reservation_id == first.reservation_id
                    && frontier.sequence_id == first.sequence_id
                    && frontier.layer_id == first.layer_id
                    && frontier.token_start.checked_add(frontier.rows) == Some(token_start)
            });
        let (frontier_token_start, row_offset) = if append {
            let frontier = self.mla_attention_ready_frontiers[layer_index]
                .as_ref()
                .expect("appendable attention-ready MLA frontier exists");
            (frontier.token_start, frontier.rows)
        } else {
            (token_start, 0)
        };
        let next_rows = row_offset
            .checked_add(rows)
            .context("attention-ready MLA frontier append row count overflows usize")?;
        let frontier_capacity_tokens =
            attention_ready_frontier_capacity_tokens(self.config.max_tokens);
        if next_rows > frontier_capacity_tokens {
            return Ok(None);
        }
        let layer_capacity_bytes = frontier_capacity_tokens
            .checked_mul(row_stride_bytes)
            .context("attention-ready MLA frontier layer capacity bytes overflow usize")?;
        let capacity_bytes = self
            .mla_attention_ready_frontiers
            .len()
            .checked_mul(layer_capacity_bytes)
            .context("attention-ready MLA frontier capacity bytes overflow usize")?;
        let frontier_buffer = self
            .mla_attention_ready_frontier_payload
            .buffer(capacity_bytes, "attention-ready MLA frontier payload")?;
        let dst_offset = layer_index
            .checked_mul(layer_capacity_bytes)
            .and_then(|offset| {
                row_offset
                    .checked_mul(row_stride_bytes)
                    .and_then(|row_offset| offset.checked_add(row_offset))
            })
            .context("attention-ready MLA frontier destination offset overflows usize")?;
        let dst = device_buffer_byte_view(
            frontier_buffer,
            dst_offset,
            target_bytes,
            "attention-ready MLA frontier append",
        )?;
        let next_frontier = DeviceKvAttentionReadyFrontier {
            reservation_id: first.reservation_id,
            sequence_id: first.sequence_id.clone(),
            layer_id: first.layer_id,
            token_start: frontier_token_start,
            rows: next_rows,
        };
        Ok(Some((dst, next_frontier)))
    }

    fn attention_ready_mla_frontier_payload_for_descriptors(
        &mut self,
        descriptors: &[KvBlockDescriptor],
    ) -> Result<Option<(GlmrtDeviceBuffer, usize, usize)>> {
        let Some(first) = descriptors.first() else {
            return Ok(None);
        };
        let layer_index = usize::try_from(first.layer_id.0)
            .context("attention-ready MLA read layer exceeds usize")?;
        if layer_index >= self.mla_attention_ready_frontiers.len() {
            return Ok(None);
        }
        let token_start = usize::try_from(first.token_start.0)
            .context("attention-ready MLA read token start exceeds usize")?;
        let mut expected_token_start = token_start;
        let mut rows = 0_usize;
        for descriptor in descriptors {
            if descriptor.reservation_id != first.reservation_id
                || descriptor.sequence_id != first.sequence_id
                || descriptor.layer_id != first.layer_id
            {
                return Ok(None);
            }
            let descriptor_token_start = usize::try_from(descriptor.token_start.0)
                .context("attention-ready MLA read descriptor token start exceeds usize")?;
            if descriptor_token_start != expected_token_start {
                return Ok(None);
            }
            expected_token_start = expected_token_start
                .checked_add(descriptor.token_count)
                .context("attention-ready MLA read token range overflows usize")?;
            rows = rows
                .checked_add(descriptor.token_count)
                .context("attention-ready MLA read row count overflows usize")?;
        }
        let Some(frontier) = self.mla_attention_ready_frontiers[layer_index].as_ref() else {
            return Ok(None);
        };
        if frontier.reservation_id != first.reservation_id
            || frontier.sequence_id != first.sequence_id
            || frontier.layer_id != first.layer_id
            || frontier.token_start != token_start
            || frontier.rows != rows
        {
            return Ok(None);
        }
        let row_stride_bytes = attention_ready_frontier_row_stride_bytes(self.config.dtype)?;
        let payload_bytes = rows
            .checked_mul(row_stride_bytes)
            .context("attention-ready MLA read payload bytes overflow usize")?;
        let layer_capacity_bytes = attention_ready_frontier_capacity_tokens(self.config.max_tokens)
            .checked_mul(row_stride_bytes)
            .context("attention-ready MLA read layer capacity bytes overflow usize")?;
        let payload_offset = layer_index
            .checked_mul(layer_capacity_bytes)
            .context("attention-ready MLA read layer offset overflows usize")?;
        let payload = device_buffer_byte_view(
            self.mla_attention_ready_frontier_payload.buffer,
            payload_offset,
            payload_bytes,
            "attention-ready MLA frontier read",
        )?;
        Ok(Some((payload, rows, layer_index)))
    }

    fn attention_ready_mla_frontier_parts(
        &mut self,
        descriptors: &[KvBlockDescriptor],
    ) -> Result<Option<RealFullDeviceMlaKvDeviceBufferView>> {
        if self.config.dtype == KvCacheDType::Nvfp4 {
            // The long profile retains its bounded frontier in packed FP8 and
            // consumes it directly in sparse attention. Dense fallback reads
            // the canonical compressed cache instead of expanding this copy.
            return Ok(None);
        }
        let Some((payload, rows, layer_index)) =
            self.attention_ready_mla_frontier_payload_for_descriptors(descriptors)?
        else {
            return Ok(None);
        };
        let row_stride_bytes = (GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM)
            .checked_mul(std::mem::size_of::<u16>())
            .context("attention-ready MLA read stride overflows usize")?;
        let kv_latent_bytes = rows
            .checked_mul(GLM52_MLA_KV_LORA_RANK)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("attention-ready MLA latent bytes overflow usize")?;
        let k_rope_bytes = rows
            .checked_mul(GLM52_MLA_QK_ROPE_HEAD_DIM)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("attention-ready MLA rope bytes overflow usize")?;
        let kv_latent = self
            .mla_unpacked_kv_latent
            .buffer(kv_latent_bytes, "attention-ready MLA latent")?;
        let k_rope = self
            .mla_unpacked_k_rope
            .buffer(k_rope_bytes, "attention-ready MLA rope")?;
        mla_kv_cache_unpack_bf16_device_buffers_for_layer(
            layer_index,
            payload,
            kv_latent,
            k_rope,
            None,
            rows,
            GLM52_MLA_KV_LORA_RANK,
            GLM52_MLA_QK_ROPE_HEAD_DIM,
            0,
            row_stride_bytes,
        )
        .context("splitting attention-ready MLA frontier")?;
        Ok(Some(RealFullDeviceMlaKvDeviceBufferView {
            rows,
            kv_latent_bytes,
            k_rope_bytes,
            kv_latent,
            k_rope,
        }))
    }

    fn pack_mla_kv_fp8_ds_mla(
        &mut self,
        projected: GlmrtDeviceBuffer,
        packed: GlmrtDeviceBuffer,
        rows: usize,
        projected_stride_bytes: usize,
        packed_stride_bytes: usize,
    ) -> Result<()> {
        self.library.cuda_mla_kv_pack_fp8_ds_mla(
            projected,
            packed,
            rows,
            projected_stride_bytes,
            packed_stride_bytes,
        )
    }

    fn pack_mla_kv_mxfp4_ds_mla(
        &mut self,
        projected: GlmrtDeviceBuffer,
        packed: GlmrtDeviceBuffer,
        rows: usize,
        projected_stride_bytes: usize,
        packed_stride_bytes: usize,
    ) -> Result<()> {
        self.library.cuda_mla_kv_pack_mxfp4_ds_mla(
            projected,
            packed,
            rows,
            projected_stride_bytes,
            packed_stride_bytes,
        )
    }

    #[allow(dead_code)]
    pub(in crate::commands::real_full) fn buffer(&self) -> GlmrtDeviceBuffer {
        self.storage.cache
    }

    fn write_logical_block_from_device(
        &self,
        descriptor: &KvBlockDescriptor,
        src: GlmrtDeviceBuffer,
    ) -> Result<RealFullDeviceKvBlockIo> {
        let logical_io = real_full_device_kv_block_io(&self.config, descriptor)?;
        validate_contiguous_payload_buffer(
            "device KV logical block write src",
            src,
            logical_io.payload_bytes,
        )?;
        let main_io = self.physical_main_kv_block_io(descriptor)?;
        let Some(dsa_io) = self.physical_dsa_bf16_block_io(descriptor)? else {
            self.library
                .copy_d2d(
                    device_buffer_byte_view(
                        self.storage.cache,
                        main_io.offset_bytes,
                        main_io.payload_bytes,
                        "device main KV write destination",
                    )?,
                    src,
                    main_io.payload_bytes,
                )
                .context("copying device main KV block")?;
            return Ok(logical_io);
        };

        let rows = descriptor.token_count;
        anyhow::ensure!(rows > 0, "device logical KV write requires nonempty block");
        let main_row_bytes = main_io.payload_bytes / rows;
        let dsa_row_bytes = dsa_io.payload_bytes / rows;
        let logical_row_bytes = main_row_bytes
            .checked_add(dsa_row_bytes)
            .context("device logical KV write row bytes overflow usize")?;
        anyhow::ensure!(
            logical_row_bytes.checked_mul(rows) == Some(logical_io.payload_bytes),
            "device logical KV write row shape mismatch"
        );
        for row in 0..rows {
            let logical_row_offset = row
                .checked_mul(logical_row_bytes)
                .context("device logical KV write source row offset overflow usize")?;
            let main_row_offset = main_io
                .offset_bytes
                .checked_add(
                    row.checked_mul(main_row_bytes)
                        .context("device main KV write row offset overflow usize")?,
                )
                .context("device main KV write cache offset overflow usize")?;
            let dsa_row_offset = dsa_io
                .offset_bytes
                .checked_add(
                    row.checked_mul(dsa_row_bytes)
                        .context("device DSA write row offset overflow usize")?,
                )
                .context("device DSA write cache offset overflow usize")?;
            self.library.copy_d2d(
                device_buffer_byte_view(
                    self.storage.cache,
                    main_row_offset,
                    main_row_bytes,
                    "device main KV write row destination",
                )?,
                device_buffer_byte_view(
                    src,
                    logical_row_offset,
                    main_row_bytes,
                    "device main KV write row source",
                )?,
                main_row_bytes,
            )?;
            self.library.copy_d2d(
                device_buffer_byte_view(
                    self.storage.cache,
                    dsa_row_offset,
                    dsa_row_bytes,
                    "device DSA write row destination",
                )?,
                device_buffer_byte_view(
                    src,
                    logical_row_offset + main_row_bytes,
                    dsa_row_bytes,
                    "device DSA write row source",
                )?,
                dsa_row_bytes,
            )?;
        }
        Ok(logical_io)
    }

    fn write_mla_dsa_planes_from_contiguous_device(
        &self,
        descriptors: &[KvBlockDescriptor],
        main_src: GlmrtDeviceBuffer,
        main_row_bytes: usize,
        dsa_src: GlmrtDeviceBuffer,
    ) -> Result<Vec<RealFullDeviceKvBlockIo>> {
        anyhow::ensure!(
            !descriptors.is_empty(),
            "direct MLA+DSA plane write requires descriptors"
        );
        let rows = descriptors.iter().try_fold(0_usize, |rows, descriptor| {
            anyhow::ensure!(
                descriptor.token_count > 0,
                "direct MLA+DSA plane write requires nonempty descriptors"
            );
            rows.checked_add(descriptor.token_count)
                .context("direct MLA+DSA plane write row count overflow usize")
        })?;
        let dsa_row_bytes = self
            .config
            .dsa_index_head_dim
            .checked_mul(std::mem::size_of::<u16>())
            .context("direct MLA+DSA plane write DSA row bytes overflow usize")?;
        validate_contiguous_payload_buffer(
            "direct MLA+DSA main plane write source",
            main_src,
            rows.checked_mul(main_row_bytes)
                .context("direct MLA+DSA main plane write bytes overflow usize")?,
        )?;
        validate_contiguous_payload_buffer(
            "direct MLA+DSA index plane write source",
            dsa_src,
            rows.checked_mul(dsa_row_bytes)
                .context("direct MLA+DSA index plane write bytes overflow usize")?,
        )?;
        anyhow::ensure!(
            main_src.device_id == self.storage.cache.device_id
                && dsa_src.device_id == self.storage.cache.device_id,
            "direct MLA+DSA plane write sources must be on CUDA device {}",
            self.storage.cache.device_id
        );

        let mut logical_ios = Vec::with_capacity(descriptors.len());
        let mut source_row = 0_usize;
        for descriptor in descriptors {
            let logical_io = real_full_device_kv_block_io(&self.config, descriptor)?;
            for (physical, logical_token_offset) in self.physical_descriptor_spans(descriptor)? {
                let main_io = real_full_device_main_kv_block_io(&self.config, &physical)?;
                let dsa_io = real_full_device_dsa_bf16_block_io(&self.config, &physical)?
                    .context("direct MLA+DSA plane write descriptor is not a DSA layer")?;
                let expected_main_bytes = physical
                    .token_count
                    .checked_mul(main_row_bytes)
                    .context("direct MLA+DSA main span bytes overflow usize")?;
                let expected_dsa_bytes = physical
                    .token_count
                    .checked_mul(dsa_row_bytes)
                    .context("direct MLA+DSA index span bytes overflow usize")?;
                anyhow::ensure!(
                    main_io.payload_bytes == expected_main_bytes,
                    "direct MLA+DSA main plane span shape mismatch"
                );
                anyhow::ensure!(
                    dsa_io.payload_bytes == expected_dsa_bytes,
                    "direct MLA+DSA index plane span shape mismatch"
                );
                let span_source_row = source_row
                    .checked_add(logical_token_offset)
                    .context("direct MLA+DSA span source row overflow usize")?;
                let main_src_offset = span_source_row
                    .checked_mul(main_row_bytes)
                    .context("direct MLA+DSA main source offset overflow usize")?;
                let dsa_src_offset = span_source_row
                    .checked_mul(dsa_row_bytes)
                    .context("direct MLA+DSA index source offset overflow usize")?;
                self.library.copy_d2d(
                    device_buffer_byte_view(
                        self.storage.cache,
                        main_io.offset_bytes,
                        main_io.payload_bytes,
                        "direct MLA main plane write destination",
                    )?,
                    device_buffer_byte_view(
                        main_src,
                        main_src_offset,
                        main_io.payload_bytes,
                        "direct MLA main plane write source",
                    )?,
                    main_io.payload_bytes,
                )?;
                self.library.copy_d2d(
                    device_buffer_byte_view(
                        self.storage.cache,
                        dsa_io.offset_bytes,
                        dsa_io.payload_bytes,
                        "direct DSA plane write destination",
                    )?,
                    device_buffer_byte_view(
                        dsa_src,
                        dsa_src_offset,
                        dsa_io.payload_bytes,
                        "direct DSA plane write source",
                    )?,
                    dsa_io.payload_bytes,
                )?;
            }
            source_row = source_row
                .checked_add(descriptor.token_count)
                .context("direct MLA+DSA source row offset overflow usize")?;
            logical_ios.push(logical_io);
        }
        Ok(logical_ios)
    }

    fn read_logical_block_to_device(
        &self,
        descriptor: &KvBlockDescriptor,
        dst: GlmrtDeviceBuffer,
    ) -> Result<RealFullDeviceKvBlockIo> {
        let logical_io = real_full_device_kv_block_io(&self.config, descriptor)?;
        validate_contiguous_payload_buffer(
            "device KV logical block read dst",
            dst,
            logical_io.payload_bytes,
        )?;
        let main_io = self.physical_main_kv_block_io(descriptor)?;
        let Some(dsa_io) = self.physical_dsa_bf16_block_io(descriptor)? else {
            self.library
                .copy_d2d(
                    dst,
                    device_buffer_byte_view(
                        self.storage.cache,
                        main_io.offset_bytes,
                        main_io.payload_bytes,
                        "device main KV read source",
                    )?,
                    main_io.payload_bytes,
                )
                .context("copying device main KV block")?;
            return Ok(logical_io);
        };

        let rows = descriptor.token_count;
        anyhow::ensure!(rows > 0, "device logical KV read requires nonempty block");
        let main_row_bytes = main_io.payload_bytes / rows;
        let dsa_row_bytes = dsa_io.payload_bytes / rows;
        let logical_row_bytes = main_row_bytes
            .checked_add(dsa_row_bytes)
            .context("device logical KV read row bytes overflow usize")?;
        anyhow::ensure!(
            logical_row_bytes.checked_mul(rows) == Some(logical_io.payload_bytes),
            "device logical KV read row shape mismatch"
        );
        for row in 0..rows {
            let logical_row_offset = row
                .checked_mul(logical_row_bytes)
                .context("device logical KV read destination row offset overflow usize")?;
            let main_row_offset = main_io
                .offset_bytes
                .checked_add(
                    row.checked_mul(main_row_bytes)
                        .context("device main KV read row offset overflow usize")?,
                )
                .context("device main KV read cache offset overflow usize")?;
            let dsa_row_offset = dsa_io
                .offset_bytes
                .checked_add(
                    row.checked_mul(dsa_row_bytes)
                        .context("device DSA read row offset overflow usize")?,
                )
                .context("device DSA read cache offset overflow usize")?;
            self.library.copy_d2d(
                device_buffer_byte_view(
                    dst,
                    logical_row_offset,
                    main_row_bytes,
                    "device main KV read row destination",
                )?,
                device_buffer_byte_view(
                    self.storage.cache,
                    main_row_offset,
                    main_row_bytes,
                    "device main KV read row source",
                )?,
                main_row_bytes,
            )?;
            self.library.copy_d2d(
                device_buffer_byte_view(
                    dst,
                    logical_row_offset + main_row_bytes,
                    dsa_row_bytes,
                    "device DSA read row destination",
                )?,
                device_buffer_byte_view(
                    self.storage.cache,
                    dsa_row_offset,
                    dsa_row_bytes,
                    "device DSA read row source",
                )?,
                dsa_row_bytes,
            )?;
        }
        Ok(logical_io)
    }

    fn upload_device_bytes_from_pinned_staging(
        &mut self,
        bytes: &[u8],
        context: &str,
    ) -> Result<DeviceBufferGuard<'a>> {
        if bytes.is_empty() {
            anyhow::bail!("{context} upload requires non-empty bytes");
        }
        let guard = DeviceBufferGuard::new(self.library, bytes.len())
            .with_context(|| format!("allocating {context}"))?;
        let staging = self
            .scheduler_upload_staging
            .buffer(bytes.len(), "device KV scheduler upload staging")
            .with_context(|| format!("allocating pinned staging for {context}"))?;
        if staging.ptr.is_null() {
            anyhow::bail!("{context} pinned staging buffer is null");
        }
        if bytes.len() > staging.bytes {
            anyhow::bail!(
                "{context} byte count {} exceeds pinned staging bytes {}",
                bytes.len(),
                staging.bytes
            );
        }
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), staging.ptr.cast::<u8>(), bytes.len());
        }
        self.library
            .copy_host_buffer_h2d(guard.buffer, staging, bytes.len())
            .with_context(|| format!("copying {context} from pinned staging"))?;
        Ok(guard)
    }

    fn stage_attention_host_slice_bytes(
        &mut self,
        slot: DeviceKvAttentionHostUploadSlot,
        bytes: &[u8],
        label: &'static str,
    ) -> Result<GlmrtDeviceBuffer> {
        if bytes.is_empty() {
            anyhow::bail!("device KV attention host upload {label} requires nonempty bytes");
        }
        let staging = self
            .attention_host_upload_staging
            .buffer(bytes.len(), "device KV attention host upload staging")
            .with_context(|| format!("allocating pinned staging for {label}"))?;
        if staging.ptr.is_null() {
            anyhow::bail!("device KV attention host upload {label} staging buffer is null");
        }
        if bytes.len() > staging.bytes {
            anyhow::bail!(
                "device KV attention host upload {label} byte count {} exceeds staging bytes {}",
                bytes.len(),
                staging.bytes
            );
        }
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), staging.ptr.cast::<u8>(), bytes.len());
        }
        let dst = match slot {
            DeviceKvAttentionHostUploadSlot::QueryNope => self
                .attention_query_nope
                .buffer(bytes.len(), "device KV attention q_nope")?,
            DeviceKvAttentionHostUploadSlot::QueryRope => self
                .attention_query_rope
                .buffer(bytes.len(), "device KV attention q_rope")?,
            DeviceKvAttentionHostUploadSlot::ProjectedQuery => self
                .attention_projected_query
                .buffer(bytes.len(), "device KV attention projected query")?,
            DeviceKvAttentionHostUploadSlot::SuffixKNope => self
                .attention_suffix_k_nope
                .buffer(bytes.len(), "device KV attention suffix k_nope")?,
            DeviceKvAttentionHostUploadSlot::SuffixKRope => self
                .attention_suffix_k_rope
                .buffer(bytes.len(), "device KV attention suffix k_rope")?,
            DeviceKvAttentionHostUploadSlot::SuffixValues => self
                .attention_suffix_values
                .buffer(bytes.len(), "device KV attention suffix values")?,
        };
        self.library
            .copy_host_buffer_h2d(dst, staging, bytes.len())
            .with_context(|| format!("copying {label} from pinned staging"))?;
        Ok(dst)
    }

    fn split_current_projected_kv_a_to_reusable_buffers(
        &mut self,
        layer_id: LayerId,
        projected_kv_a_buffer: GlmrtDeviceBuffer,
        rows: usize,
        kv_lora_rank: usize,
        rope_dim: usize,
    ) -> Result<RealFullDeviceMlaKvDeviceBufferView> {
        if rows == 0 || kv_lora_rank == 0 || rope_dim == 0 {
            anyhow::bail!(
                "device MLA current-row KV split requires nonzero shape, got rows={rows} kv_lora_rank={kv_lora_rank} rope_dim={rope_dim}"
            );
        }
        if kv_lora_rank != GLM52_MLA_KV_LORA_RANK {
            anyhow::bail!(
                "device MLA current-row KV split kv_lora_rank mismatch: expected {} got {}",
                GLM52_MLA_KV_LORA_RANK,
                kv_lora_rank
            );
        }
        if rope_dim != GLM52_MLA_QK_ROPE_HEAD_DIM {
            anyhow::bail!(
                "device MLA current-row KV split rope_dim mismatch: expected {} got {}",
                GLM52_MLA_QK_ROPE_HEAD_DIM,
                rope_dim
            );
        }
        let payload_stride_bytes = kv_lora_rank
            .checked_add(rope_dim)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA current-row KV projected row stride overflow")?;
        let payload_bytes = rows
            .checked_mul(payload_stride_bytes)
            .context("device MLA current-row KV projected byte count overflow")?;
        validate_contiguous_payload_buffer(
            "device MLA current-row projected kv_a",
            projected_kv_a_buffer,
            payload_bytes,
        )?;
        let kv_latent_bytes = rows
            .checked_mul(kv_lora_rank)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA current-row KV latent bytes overflow")?;
        let k_rope_bytes = rows
            .checked_mul(rope_dim)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA current-row KV RoPE bytes overflow")?;
        let kv_latent = self
            .mla_current_kv_latent
            .buffer(kv_latent_bytes, "device MLA current-row reusable KV latent")
            .context("allocating reusable device MLA current-row KV latent")?;
        let k_rope = self
            .mla_current_k_rope
            .buffer(k_rope_bytes, "device MLA current-row reusable KV k_rope")
            .context("allocating reusable device MLA current-row KV k_rope")?;
        if projected_kv_a_buffer.device_id != kv_latent.device_id
            || k_rope.device_id != kv_latent.device_id
        {
            anyhow::bail!(
                "device MLA current-row projected kv_a and reusable split outputs must be on the same CUDA device"
            );
        }
        mla_kv_cache_unpack_bf16_device_buffers_for_layer(
            layer_id.0 as usize,
            projected_kv_a_buffer,
            kv_latent,
            k_rope,
            None,
            rows,
            kv_lora_rank,
            rope_dim,
            0,
            payload_stride_bytes,
        )
        .context("splitting device MLA current-row projected kv_a into reusable buffers")?;
        Ok(RealFullDeviceMlaKvDeviceBufferView {
            rows,
            kv_latent_bytes,
            k_rope_bytes,
            kv_latent,
            k_rope,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn split_projected_query_suffix_to_reusable_buffers(
        &mut self,
        layer_id: LayerId,
        projected_query_buffer: GlmrtDeviceBuffer,
        prefix_rows: usize,
        suffix_positions: &[u32],
        positions_device: Option<GlmrtDeviceBuffer>,
        heads: usize,
        nope_dim: usize,
        rope_dim: usize,
        theta: f32,
        compact_output: bool,
    ) -> Result<RealFullDeviceMlaQueryDeviceBuffers> {
        if suffix_positions.is_empty() {
            anyhow::bail!("device MLA query projection requires at least one suffix row");
        }
        if heads == 0 || nope_dim == 0 || rope_dim == 0 {
            anyhow::bail!(
                "device MLA query projection requires nonzero shape, got heads={heads} nope_dim={nope_dim} rope_dim={rope_dim}"
            );
        }
        if !theta.is_finite() || theta <= 0.0 {
            anyhow::bail!("device MLA query projection RoPE theta must be finite and positive");
        }
        let suffix_rows = suffix_positions.len();
        let rows = prefix_rows
            .checked_add(suffix_rows)
            .context("device MLA query projection total row count overflow")?;
        let head_width = nope_dim
            .checked_add(rope_dim)
            .context("device MLA query projection head width overflow")?;
        let projected_bytes = suffix_rows
            .checked_mul(heads)
            .and_then(|values| values.checked_mul(head_width))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA query projection q_b byte count overflow")?;
        validate_contiguous_payload_buffer(
            "device MLA query projected q_b device buffer",
            projected_query_buffer,
            projected_bytes,
        )?;
        if let Some(positions_device) = positions_device {
            validate_contiguous_payload_buffer(
                "device MLA query RoPE positions",
                positions_device,
                std::mem::size_of_val(suffix_positions),
            )?;
        }
        let suffix_q_nope_bytes = suffix_rows
            .checked_mul(heads)
            .and_then(|values| values.checked_mul(nope_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA query projection suffix q_nope bytes overflow")?;
        let suffix_q_rope_bytes = suffix_rows
            .checked_mul(heads)
            .and_then(|values| values.checked_mul(rope_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA query projection suffix q_rope bytes overflow")?;
        let output_rows = if compact_output { suffix_rows } else { rows };
        let q_nope_bytes = output_rows
            .checked_mul(heads)
            .and_then(|values| values.checked_mul(nope_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA query projection full q_nope bytes overflow")?;
        let q_rope_rotated_bytes = output_rows
            .checked_mul(heads)
            .and_then(|values| values.checked_mul(rope_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA query projection full q_rope bytes overflow")?;
        let output_prefix_rows = if compact_output { 0 } else { prefix_rows };
        let prefix_q_nope_bytes = output_prefix_rows
            .checked_mul(heads)
            .and_then(|values| values.checked_mul(nope_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA query projection prefix q_nope bytes overflow")?;
        let prefix_q_rope_bytes = output_prefix_rows
            .checked_mul(heads)
            .and_then(|values| values.checked_mul(rope_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA query projection prefix q_rope bytes overflow")?;

        let q_rope_unrotated = self
            .attention_query_split_unrotated_rope
            .buffer(
                suffix_q_rope_bytes,
                "device KV attention query split unrotated q_rope",
            )
            .context("allocating reusable device MLA query unrotated q_rope suffix")?;
        let q_nope = self
            .attention_query_split_nope
            .buffer(q_nope_bytes, "device KV attention query split q_nope")
            .context("allocating reusable device MLA query full q_nope")?;
        let q_rope_rotated = self
            .attention_query_split_rope_rotated
            .buffer(
                q_rope_rotated_bytes,
                "device KV attention query split rotated q_rope",
            )
            .context("allocating reusable device MLA query full rotated q_rope")?;
        if projected_query_buffer.device_id != q_nope.device_id {
            anyhow::bail!(
                "device MLA query projected q_b buffer is on CUDA device {}, but reusable query outputs are on device {}",
                projected_query_buffer.device_id,
                q_nope.device_id
            );
        }
        if let Some(positions_device) = positions_device {
            if positions_device.device_id != q_nope.device_id {
                anyhow::bail!(
                    "device MLA query RoPE positions buffer is on CUDA device {}, but reusable query outputs are on device {}",
                    positions_device.device_id,
                    q_nope.device_id
                );
            }
        }
        if !compact_output {
            zero_device_buffer_bytes(
                self.library,
                q_nope,
                q_nope_bytes,
                "device MLA query q_nope",
            )?;
            zero_device_buffer_bytes(
                self.library,
                q_rope_rotated,
                q_rope_rotated_bytes,
                "device MLA query rotated q_rope",
            )?;
        }

        let q_nope_suffix = device_buffer_byte_view(
            q_nope,
            prefix_q_nope_bytes,
            suffix_q_nope_bytes,
            "device MLA query q_nope suffix",
        )?;
        let q_rope_rotated_suffix = device_buffer_byte_view(
            q_rope_rotated,
            prefix_q_rope_bytes,
            suffix_q_rope_bytes,
            "device MLA query rotated q_rope suffix",
        )?;
        if compact_output && suffix_rows == 1 {
            mla_query_split_rope_bf16_device_buffers_for_layer(
                layer_id.0 as usize,
                projected_query_buffer,
                q_nope_suffix,
                q_rope_unrotated,
                suffix_positions[0],
                q_rope_rotated_suffix,
                suffix_rows,
                heads,
                nope_dim,
                rope_dim,
                theta,
            )
            .context("splitting and rotating fused device MLA decode query")?;
        } else if compact_output && suffix_rows <= REAL_FULL_COMPRESSED_MLA_MAX_QUERY_ROWS {
            let positions_device =
                positions_device.context("batched fused MLA query RoPE positions missing")?;
            mla_query_split_rope_bf16_device_positions_for_layer(
                layer_id.0 as usize,
                projected_query_buffer,
                q_nope_suffix,
                q_rope_unrotated,
                positions_device,
                q_rope_rotated_suffix,
                suffix_rows,
                heads,
                nope_dim,
                rope_dim,
                theta,
            )
            .context("splitting and rotating batched fused device MLA query")?;
        } else {
            let positions_device =
                positions_device.context("device MLA query RoPE positions buffer missing")?;
            mla_kv_projected_split_bf16_device_buffers_for_layer(
                layer_id.0 as usize,
                projected_query_buffer,
                q_nope_suffix,
                q_rope_unrotated,
                suffix_rows,
                heads,
                nope_dim,
                rope_dim,
            )
            .context("splitting device MLA query projected q_b into reusable buffers")?;
            rope_bf16_device_buffers_for_layer(
                layer_id.0 as usize,
                q_rope_unrotated,
                positions_device,
                q_rope_rotated_suffix,
                suffix_rows,
                heads,
                rope_dim,
                theta,
            )
            .context("rotating device MLA query q_rope suffix into reusable buffer")?;
        }

        Ok(RealFullDeviceMlaQueryDeviceBuffers {
            q_nope,
            q_rope_rotated,
        })
    }

    fn rotate_mla_k_rope_to_reusable_buffer(
        &mut self,
        layer_id: LayerId,
        parts: &RealFullDeviceMlaKvDeviceBufferView,
        positions: &[u32],
        positions_device: GlmrtDeviceBuffer,
        theta: f32,
    ) -> Result<RealFullDeviceMlaKvRopeDeviceBuffers> {
        if positions.len() != parts.rows {
            anyhow::bail!(
                "device MLA KV RoPE positions length mismatch: expected {} got {}",
                parts.rows,
                positions.len()
            );
        }
        if !theta.is_finite() || theta <= 0.0 {
            anyhow::bail!("device MLA KV RoPE theta must be finite and positive");
        }
        validate_contiguous_payload_buffer(
            "device MLA KV RoPE positions",
            positions_device,
            std::mem::size_of_val(positions),
        )?;
        validate_contiguous_payload_buffer(
            "device MLA KV k_rope",
            parts.k_rope,
            parts.k_rope_bytes,
        )?;
        let k_rope_rotated = self
            .attention_k_rope_rotated
            .buffer(
                parts.k_rope_bytes,
                "device KV attention reusable rotated k_rope",
            )
            .context("allocating reusable device MLA KV rotated k_rope output")?;
        if positions_device.device_id != parts.k_rope.device_id
            || k_rope_rotated.device_id != parts.k_rope.device_id
        {
            anyhow::bail!(
                "device MLA KV RoPE input, positions, and reusable output buffers must be on the same CUDA device"
            );
        }
        rope_bf16_device_buffers_for_layer(
            layer_id.0 as usize,
            parts.k_rope,
            positions_device,
            k_rope_rotated,
            parts.rows,
            1,
            GLM52_MLA_QK_ROPE_HEAD_DIM,
            theta,
        )
        .context("executing device MLA KV k_rope RoPE into reusable buffer")?;
        Ok(RealFullDeviceMlaKvRopeDeviceBuffers {
            rows: parts.rows,
            rotary_dim: GLM52_MLA_QK_ROPE_HEAD_DIM,
            k_rope_rotated_bytes: parts.k_rope_bytes,
            k_rope_rotated,
        })
    }

    fn rotate_mla_k_rope_to_suffix_reusable_buffer(
        &mut self,
        layer_id: LayerId,
        parts: &RealFullDeviceMlaKvDeviceBufferView,
        positions: &[u32],
        positions_device: GlmrtDeviceBuffer,
        theta: f32,
    ) -> Result<RealFullDeviceMlaKvRopeDeviceBuffers> {
        if positions.len() != parts.rows {
            anyhow::bail!(
                "device MLA KV suffix RoPE positions length mismatch: expected {} got {}",
                parts.rows,
                positions.len()
            );
        }
        if !theta.is_finite() || theta <= 0.0 {
            anyhow::bail!("device MLA KV suffix RoPE theta must be finite and positive");
        }
        validate_contiguous_payload_buffer(
            "device MLA KV suffix RoPE positions",
            positions_device,
            std::mem::size_of_val(positions),
        )?;
        validate_contiguous_payload_buffer(
            "device MLA KV suffix k_rope",
            parts.k_rope,
            parts.k_rope_bytes,
        )?;
        let k_rope_rotated = self
            .attention_suffix_k_rope
            .buffer(
                parts.k_rope_bytes,
                "device KV attention reusable suffix rotated k_rope",
            )
            .context("allocating reusable device MLA suffix KV rotated k_rope output")?;
        if positions_device.device_id != parts.k_rope.device_id
            || k_rope_rotated.device_id != parts.k_rope.device_id
        {
            anyhow::bail!(
                "device MLA suffix KV RoPE input, positions, and reusable output buffers must be on the same CUDA device"
            );
        }
        rope_bf16_device_buffers_for_layer(
            layer_id.0 as usize,
            parts.k_rope,
            positions_device,
            k_rope_rotated,
            parts.rows,
            1,
            GLM52_MLA_QK_ROPE_HEAD_DIM,
            theta,
        )
        .context("executing device MLA suffix KV k_rope RoPE into reusable buffer")?;
        Ok(RealFullDeviceMlaKvRopeDeviceBuffers {
            rows: parts.rows,
            rotary_dim: GLM52_MLA_QK_ROPE_HEAD_DIM,
            k_rope_rotated_bytes: parts.k_rope_bytes,
            k_rope_rotated,
        })
    }

    fn project_mla_kv_latent_and_split_to_reusable_buffers(
        &mut self,
        layer_id: LayerId,
        parts: &RealFullDeviceMlaKvDeviceBufferView,
        kv_norm_weight: GlmrtDeviceBuffer,
        kv_b_weight: GlmrtDeviceBuffer,
        heads: usize,
        nope_dim: usize,
        v_dim: usize,
        latent_is_normalized: bool,
        eps: f32,
    ) -> Result<RealFullDeviceMlaKvProjectedDeviceBuffers> {
        if heads == 0 {
            anyhow::bail!("device MLA KV projected split requires at least one head");
        }
        if nope_dim == 0 {
            anyhow::bail!("device MLA KV projected split requires nonzero nope_dim");
        }
        if v_dim == 0 {
            anyhow::bail!("device MLA KV projected split requires nonzero v_dim");
        }
        if !eps.is_finite() {
            anyhow::bail!("device MLA KV projected split RMSNorm eps must be finite");
        }
        let normalized_bytes = parts.kv_latent_bytes;
        let projected_width = heads
            .checked_mul(
                nope_dim
                    .checked_add(v_dim)
                    .context("device MLA KV projected split head width overflow")?,
            )
            .context("device MLA KV projected width overflow")?;
        let projected_bytes = parts
            .rows
            .checked_mul(projected_width)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA KV projected bytes overflow usize")?;
        let k_nope_bytes = parts
            .rows
            .checked_mul(heads)
            .and_then(|values| values.checked_mul(nope_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA KV k_nope bytes overflow usize")?;
        let values_bytes = parts
            .rows
            .checked_mul(heads)
            .and_then(|values| values.checked_mul(v_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA KV value bytes overflow usize")?;
        validate_contiguous_payload_buffer(
            "device MLA KV RMSNorm weight",
            kv_norm_weight,
            GLM52_MLA_KV_LORA_RANK * std::mem::size_of::<u16>(),
        )?;
        validate_contiguous_payload_buffer(
            "device MLA KV kv_b weight",
            kv_b_weight,
            projected_width
                .checked_mul(GLM52_MLA_KV_LORA_RANK)
                .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
                .context("device MLA KV kv_b weight bytes overflow usize")?,
        )?;
        let normalized = if latent_is_normalized {
            parts.kv_latent
        } else {
            self.attention_projected_kv_normalized
                .buffer(
                    normalized_bytes,
                    "device KV attention projected KV normalized latent",
                )
                .context("allocating reusable device MLA KV normalized latent output")?
        };
        let projected = self
            .attention_projected_kv_projected
            .buffer(projected_bytes, "device KV attention projected KV output")
            .context("allocating reusable device MLA KV projected output")?;
        let k_nope = self
            .attention_projected_kv_k_nope
            .buffer(k_nope_bytes, "device KV attention projected KV k_nope")
            .context("allocating reusable device MLA KV k_nope output")?;
        let values = self
            .attention_projected_kv_values
            .buffer(values_bytes, "device KV attention projected KV values")
            .context("allocating reusable device MLA KV value output")?;
        if kv_norm_weight.device_id != normalized.device_id
            || kv_b_weight.device_id != normalized.device_id
            || parts.kv_latent.device_id != normalized.device_id
        {
            anyhow::bail!("device MLA KV projection inputs and reusable outputs must be on the same CUDA device");
        }
        if !latent_is_normalized {
            rmsnorm_bf16_device_buffers_for_layer(
                layer_id.0 as usize,
                parts.kv_latent,
                kv_norm_weight,
                normalized,
                parts.rows,
                GLM52_MLA_KV_LORA_RANK,
                eps,
            )
            .context("executing reusable device MLA KV latent RMSNorm")?;
        }
        linear_rows_bf16_device_buffers_for_layer(
            layer_id.0 as usize,
            normalized,
            kv_b_weight,
            projected,
            parts.rows,
            GLM52_MLA_KV_LORA_RANK,
            projected_width,
        )
        .context("executing reusable device MLA KV kv_b projection")?;
        mla_kv_projected_split_bf16_device_buffers_for_layer(
            layer_id.0 as usize,
            projected,
            k_nope,
            values,
            parts.rows,
            heads,
            nope_dim,
            v_dim,
        )
        .context("splitting reusable device MLA KV projected buffer")?;
        Ok(RealFullDeviceMlaKvProjectedDeviceBuffers {
            rows: parts.rows,
            heads,
            nope_dim,
            v_dim,
            k_nope_bytes,
            values_bytes,
            k_nope,
            values,
        })
    }

    fn normalize_mla_kv_latent_to_reusable_buffer(
        &mut self,
        layer_id: LayerId,
        parts: &RealFullDeviceMlaKvDeviceBufferView,
        kv_norm_weight: GlmrtDeviceBuffer,
        eps: f32,
    ) -> Result<GlmrtDeviceBuffer> {
        if !eps.is_finite() || eps <= 0.0 {
            anyhow::bail!("device compressed MLA KV RMSNorm eps must be positive and finite");
        }
        validate_contiguous_payload_buffer(
            "device compressed MLA KV RMSNorm weight",
            kv_norm_weight,
            GLM52_MLA_KV_LORA_RANK * std::mem::size_of::<u16>(),
        )?;
        let normalized = self
            .attention_projected_kv_normalized
            .buffer(
                parts.kv_latent_bytes,
                "device KV compressed attention normalized latent",
            )
            .context("allocating reusable compressed MLA normalized latent")?;
        if kv_norm_weight.device_id != normalized.device_id
            || parts.kv_latent.device_id != normalized.device_id
        {
            anyhow::bail!(
                "device compressed MLA KV normalization buffers must be on one CUDA device"
            );
        }
        rmsnorm_bf16_device_buffers_for_layer(
            layer_id.0 as usize,
            parts.kv_latent,
            kv_norm_weight,
            normalized,
            parts.rows,
            GLM52_MLA_KV_LORA_RANK,
            eps,
        )
        .context("normalizing reusable compressed MLA KV latent")?;
        Ok(normalized)
    }

    fn project_mla_kv_latent_and_split_to_suffix_reusable_buffers(
        &mut self,
        layer_id: LayerId,
        parts: &RealFullDeviceMlaKvDeviceBufferView,
        kv_norm_weight: GlmrtDeviceBuffer,
        kv_b_weight: GlmrtDeviceBuffer,
        heads: usize,
        nope_dim: usize,
        v_dim: usize,
        eps: f32,
    ) -> Result<RealFullDeviceMlaKvProjectedDeviceBuffers> {
        if heads == 0 {
            anyhow::bail!("device MLA suffix KV projected split requires at least one head");
        }
        if nope_dim == 0 {
            anyhow::bail!("device MLA suffix KV projected split requires nonzero nope_dim");
        }
        if v_dim == 0 {
            anyhow::bail!("device MLA suffix KV projected split requires nonzero v_dim");
        }
        if !eps.is_finite() {
            anyhow::bail!("device MLA suffix KV projected split RMSNorm eps must be finite");
        }
        let normalized_bytes = parts.kv_latent_bytes;
        let projected_width = heads
            .checked_mul(
                nope_dim
                    .checked_add(v_dim)
                    .context("device MLA suffix KV projected split head width overflow")?,
            )
            .context("device MLA suffix KV projected width overflow")?;
        let projected_bytes = parts
            .rows
            .checked_mul(projected_width)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA suffix KV projected bytes overflow usize")?;
        let k_nope_bytes = parts
            .rows
            .checked_mul(heads)
            .and_then(|values| values.checked_mul(nope_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA suffix KV k_nope bytes overflow usize")?;
        let values_bytes = parts
            .rows
            .checked_mul(heads)
            .and_then(|values| values.checked_mul(v_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("device MLA suffix KV value bytes overflow usize")?;
        validate_contiguous_payload_buffer(
            "device MLA suffix KV RMSNorm weight",
            kv_norm_weight,
            GLM52_MLA_KV_LORA_RANK * std::mem::size_of::<u16>(),
        )?;
        validate_contiguous_payload_buffer(
            "device MLA suffix KV kv_b weight",
            kv_b_weight,
            projected_width
                .checked_mul(GLM52_MLA_KV_LORA_RANK)
                .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
                .context("device MLA suffix KV kv_b weight bytes overflow usize")?,
        )?;
        let normalized = self
            .attention_suffix_projected_kv_normalized
            .buffer(
                normalized_bytes,
                "device KV attention suffix projected KV normalized latent",
            )
            .context("allocating reusable device MLA suffix KV normalized latent output")?;
        let projected = self
            .attention_suffix_projected_kv_projected
            .buffer(
                projected_bytes,
                "device KV attention suffix projected KV output",
            )
            .context("allocating reusable device MLA suffix KV projected output")?;
        let k_nope = self
            .attention_suffix_k_nope
            .buffer(
                k_nope_bytes,
                "device KV attention suffix projected KV k_nope",
            )
            .context("allocating reusable device MLA suffix KV k_nope output")?;
        let values = self
            .attention_suffix_values
            .buffer(
                values_bytes,
                "device KV attention suffix projected KV values",
            )
            .context("allocating reusable device MLA suffix KV value output")?;
        if kv_norm_weight.device_id != normalized.device_id
            || kv_b_weight.device_id != normalized.device_id
            || parts.kv_latent.device_id != normalized.device_id
        {
            anyhow::bail!(
                "device MLA suffix KV projection inputs and reusable outputs must be on the same CUDA device"
            );
        }
        rmsnorm_bf16_device_buffers_for_layer(
            layer_id.0 as usize,
            parts.kv_latent,
            kv_norm_weight,
            normalized,
            parts.rows,
            GLM52_MLA_KV_LORA_RANK,
            eps,
        )
        .context("executing reusable device MLA suffix KV latent RMSNorm")?;
        linear_rows_bf16_device_buffers_for_layer(
            layer_id.0 as usize,
            normalized,
            kv_b_weight,
            projected,
            parts.rows,
            GLM52_MLA_KV_LORA_RANK,
            projected_width,
        )
        .context("executing reusable device MLA suffix KV kv_b projection")?;
        mla_kv_projected_split_bf16_device_buffers_for_layer(
            layer_id.0 as usize,
            projected,
            k_nope,
            values,
            parts.rows,
            heads,
            nope_dim,
            v_dim,
        )
        .context("splitting reusable device MLA suffix KV projected buffer")?;
        Ok(RealFullDeviceMlaKvProjectedDeviceBuffers {
            rows: parts.rows,
            heads,
            nope_dim,
            v_dim,
            k_nope_bytes,
            values_bytes,
            k_nope,
            values,
        })
    }

    fn attention_combined_buffers(
        &mut self,
        k_nope_bytes: usize,
        k_rope_bytes: usize,
        values_bytes: usize,
        label: &'static str,
    ) -> Result<(GlmrtDeviceBuffer, GlmrtDeviceBuffer, GlmrtDeviceBuffer)> {
        let k_nope = self
            .attention_combined_k_nope
            .buffer(k_nope_bytes, "device KV attention combined k_nope")
            .with_context(|| format!("allocating {label} k_nope buffer"))?;
        let k_rope = self
            .attention_combined_k_rope
            .buffer(k_rope_bytes, "device KV attention combined k_rope")
            .with_context(|| format!("allocating {label} k_rope buffer"))?;
        let values = self
            .attention_combined_values
            .buffer(values_bytes, "device KV attention combined values")
            .with_context(|| format!("allocating {label} values buffer"))?;
        Ok((k_nope, k_rope, values))
    }

    #[allow(dead_code)]
    pub(in crate::commands::real_full) fn write_block_from_device(
        &self,
        descriptor: &KvBlockDescriptor,
        src: GlmrtDeviceBuffer,
    ) -> Result<RealFullDeviceKvBlockIo> {
        self.write_logical_block_from_device(descriptor, src)
            .with_context(|| {
                format!(
                    "writing device KV block layer={} token_start={} token_count={}",
                    descriptor.layer_id.0, descriptor.token_start.0, descriptor.token_count,
                )
            })
    }

    #[allow(dead_code)]
    pub(in crate::commands::real_full) fn read_block_to_device(
        &self,
        descriptor: &KvBlockDescriptor,
        dst: GlmrtDeviceBuffer,
    ) -> Result<RealFullDeviceKvBlockIo> {
        self.read_logical_block_to_device(descriptor, dst)
            .with_context(|| {
                format!(
                    "reading device KV block layer={} token_start={} token_count={}",
                    descriptor.layer_id.0, descriptor.token_start.0, descriptor.token_count,
                )
            })
    }

    pub(in crate::commands::real_full) fn write_blocks_from_contiguous_device(
        &mut self,
        descriptors: &[KvBlockDescriptor],
        src: GlmrtDeviceBuffer,
    ) -> Result<Vec<RealFullDeviceKvBlockIo>> {
        let batch = self.logical_batch_plan(descriptors)?;
        if batch.block_count() == 0 {
            return Ok(Vec::new());
        }
        validate_contiguous_payload_buffer("device KV batch write src", src, batch.total_bytes)?;
        for (block_index, (descriptor, io)) in descriptors.iter().zip(&batch.ios).enumerate() {
            let src_offset = usize::try_from(batch.payload_offsets[block_index])
                .context("device KV batch write source offset does not fit usize")?;
            let row_bytes = io
                .payload_bytes
                .checked_div(descriptor.token_count)
                .context("device KV batch write descriptor has zero rows")?;
            for (physical, logical_token_offset) in self.physical_descriptor_spans(descriptor)? {
                let mut logical = descriptor.clone();
                logical.token_start = PositionId(
                    descriptor
                        .token_start
                        .0
                        .checked_add(logical_token_offset as u64)
                        .context("device KV batch write logical span start overflow")?,
                );
                logical.token_count = physical.token_count;
                let span_src_offset = src_offset
                    .checked_add(
                        logical_token_offset
                            .checked_mul(row_bytes)
                            .context("device KV batch write span row offset overflow")?,
                    )
                    .context("device KV batch write span source offset overflow")?;
                self.write_logical_block_from_device(
                    &logical,
                    device_buffer_byte_view(
                        src,
                        span_src_offset,
                        physical
                            .token_count
                            .checked_mul(row_bytes)
                            .context("device KV batch write span byte count overflow")?,
                        "device KV batch write source span",
                    )?,
                )
                .with_context(|| {
                    format!(
                        "copying device KV batch write block {block_index}/{} logical_offset={} src_offset={} span_rows={}",
                        batch.block_count(),
                        io.offset_bytes,
                        span_src_offset,
                        physical.token_count,
                    )
                })?;
            }
        }
        Ok(batch.ios)
    }

    pub(in crate::commands::real_full) fn write_host_blocks_from_pinned_staging(
        &mut self,
        descriptors: &[KvBlockDescriptor],
        payloads: &[Vec<u8>],
    ) -> Result<Vec<RealFullDeviceKvBlockIo>> {
        validate_device_kv_payloads(
            &self.config,
            descriptors,
            payloads,
            "device KV pinned host write",
        )?;
        let batch = self.logical_batch_plan(descriptors)?;
        if batch.block_count() == 0 {
            return Ok(Vec::new());
        }
        let staged_bytes = payloads
            .iter()
            .try_fold(0_usize, |acc, payload| acc.checked_add(payload.len()))
            .context("device KV pinned host write byte count overflow usize")?;
        if staged_bytes != batch.total_bytes {
            anyhow::bail!(
                "device KV pinned host write byte mismatch: descriptors require {} bytes, payloads contain {staged_bytes}",
                batch.total_bytes
            );
        }
        let staging = self
            .host_write_staging
            .buffer(batch.total_bytes, "live scheduler device KV write payload")?;
        let staging_slice =
            unsafe { slice::from_raw_parts_mut(staging.ptr.cast::<u8>(), batch.total_bytes) };
        let mut offset = 0_usize;
        for payload in payloads {
            let end = offset
                .checked_add(payload.len())
                .context("device KV pinned host write staging offset overflow usize")?;
            staging_slice[offset..end].copy_from_slice(payload);
            offset = end;
        }
        if offset != batch.total_bytes {
            anyhow::bail!(
                "device KV pinned host write staging consumed {offset} bytes, expected {}",
                batch.total_bytes
            );
        }
        let src = self
            .device_write_source
            .buffer(batch.total_bytes, "live scheduler device KV write source")?;
        self.library
            .copy_host_buffer_h2d(src, staging, batch.total_bytes)
            .context("copying live scheduler device KV write payload from pinned staging")?;
        self.write_blocks_from_contiguous_device(descriptors, src)
    }

    pub(in crate::commands::real_full) fn read_blocks_to_contiguous_device(
        &mut self,
        descriptors: &[KvBlockDescriptor],
        dst: GlmrtDeviceBuffer,
    ) -> Result<Vec<RealFullDeviceKvBlockIo>> {
        let batch = self.logical_batch_plan(descriptors)?;
        self.read_batch_to_contiguous_device(batch, dst)
    }

    fn read_batch_to_contiguous_device(
        &mut self,
        batch: DeviceKvBatchPlan,
        dst: GlmrtDeviceBuffer,
    ) -> Result<Vec<RealFullDeviceKvBlockIo>> {
        if batch.block_count() == 0 {
            return Ok(Vec::new());
        }
        validate_contiguous_payload_buffer("device KV batch read dst", dst, batch.total_bytes)?;
        for (block_index, (descriptor, io)) in batch.descriptors.iter().zip(&batch.ios).enumerate()
        {
            let dst_offset = usize::try_from(batch.payload_offsets[block_index])
                .context("device KV batch read destination offset does not fit usize")?;
            let row_bytes = io
                .payload_bytes
                .checked_div(descriptor.token_count)
                .context("device KV batch read descriptor has zero rows")?;
            for (physical, logical_token_offset) in self.physical_descriptor_spans(descriptor)? {
                let mut logical = descriptor.clone();
                logical.token_start = PositionId(
                    descriptor
                        .token_start
                        .0
                        .checked_add(logical_token_offset as u64)
                        .context("device KV batch read logical span start overflow")?,
                );
                logical.token_count = physical.token_count;
                let span_dst_offset = dst_offset
                    .checked_add(
                        logical_token_offset
                            .checked_mul(row_bytes)
                            .context("device KV batch read span row offset overflow")?,
                    )
                    .context("device KV batch read span destination offset overflow")?;
                self.read_logical_block_to_device(
                    &logical,
                    device_buffer_byte_view(
                        dst,
                        span_dst_offset,
                        physical
                            .token_count
                            .checked_mul(row_bytes)
                            .context("device KV batch read span byte count overflow")?,
                        "device KV batch read destination span",
                    )?,
                )
                .with_context(|| {
                    format!(
                        "copying device KV batch read block {block_index}/{} logical_offset={} dst_offset={} span_rows={}",
                        batch.block_count(),
                        io.offset_bytes,
                        span_dst_offset,
                        physical.token_count,
                    )
                })?;
            }
        }
        Ok(batch.ios)
    }

    fn stage_rope_positions_u32(
        &mut self,
        positions: &[u32],
        label: &'static str,
    ) -> Result<GlmrtDeviceBuffer> {
        if positions.is_empty() {
            anyhow::bail!("device KV RoPE positions {label} require nonempty values");
        }
        let bytes = u32_slice_bytes(positions);
        let staging = self
            .rope_position_staging
            .buffer(bytes.len(), "device KV RoPE position staging")?;
        let staging_slice =
            unsafe { slice::from_raw_parts_mut(staging.ptr.cast::<u8>(), bytes.len()) };
        staging_slice.copy_from_slice(bytes);
        let dst = self
            .rope_positions_device
            .buffer(bytes.len(), "device KV RoPE positions")?;
        self.library
            .copy_host_buffer_h2d(dst, staging, bytes.len())
            .with_context(|| format!("copying device KV RoPE positions {label}"))?;
        Ok(dst)
    }

    fn stage_physical_positions_u32(
        &mut self,
        positions: &[u32],
        label: &'static str,
    ) -> Result<GlmrtDeviceBuffer> {
        if positions.is_empty() {
            anyhow::bail!("device KV physical positions {label} require nonempty values");
        }
        let bytes = u32_slice_bytes(positions);
        let staging = self
            .rope_position_staging
            .buffer(bytes.len(), "device KV physical position staging")?;
        let staging_slice =
            unsafe { slice::from_raw_parts_mut(staging.ptr.cast::<u8>(), bytes.len()) };
        staging_slice.copy_from_slice(bytes);
        let dst = self
            .physical_positions_device
            .buffer(bytes.len(), "device KV physical positions")?;
        self.library
            .copy_host_buffer_h2d(dst, staging, bytes.len())
            .with_context(|| format!("copying device KV physical positions {label}"))?;
        Ok(dst)
    }

    fn stage_batch_metadata_u64s(
        &mut self,
        values: &[u64],
        slot: DeviceKvBatchMetadataSlot,
        label: &'static str,
    ) -> Result<GlmrtDeviceBuffer> {
        if values.is_empty() {
            anyhow::bail!("device KV batch metadata {label} requires nonempty values");
        }
        let bytes = u64_slice_bytes(values);
        let staging = self
            .batch_metadata_staging
            .buffer(bytes.len(), "device KV batch metadata staging")?;
        let staging_slice =
            unsafe { slice::from_raw_parts_mut(staging.ptr.cast::<u8>(), bytes.len()) };
        staging_slice.copy_from_slice(bytes);
        let dst = match slot {
            DeviceKvBatchMetadataSlot::PayloadOffsets => self
                .payload_offsets_device
                .buffer(bytes.len(), "device KV batch payload offsets")?,
            DeviceKvBatchMetadataSlot::CacheOffsets => self
                .cache_offsets_device
                .buffer(bytes.len(), "device KV batch cache offsets")?,
            DeviceKvBatchMetadataSlot::BlockBytes => self
                .block_bytes_device
                .buffer(bytes.len(), "device KV batch block byte counts")?,
        };
        self.library
            .copy_host_buffer_h2d(dst, staging, bytes.len())
            .with_context(|| format!("copying {label} from pinned staging"))?;
        Ok(dst)
    }

    #[allow(dead_code)]
    pub(in crate::commands::real_full) unsafe fn write_block_from_device_async(
        &self,
        descriptor: &KvBlockDescriptor,
        src: GlmrtDeviceBuffer,
        cuda_stream: *mut c_void,
    ) -> Result<RealFullDeviceKvBlockIo> {
        let logical_io = real_full_device_kv_block_io(&self.config, descriptor)?;
        validate_contiguous_payload_buffer(
            "async device KV logical block write src",
            src,
            logical_io.payload_bytes,
        )?;
        let main_io = self.physical_main_kv_block_io(descriptor)?;
        if let Some(dsa_io) = self.physical_dsa_bf16_block_io(descriptor)? {
            let rows = descriptor.token_count;
            anyhow::ensure!(rows > 0, "async device KV write requires nonempty block");
            let main_row_bytes = main_io.payload_bytes / rows;
            let dsa_row_bytes = dsa_io.payload_bytes / rows;
            let logical_row_bytes = main_row_bytes
                .checked_add(dsa_row_bytes)
                .context("async device logical KV write row bytes overflow usize")?;
            let dsa_source_bytes = logical_io
                .payload_bytes
                .checked_sub(main_row_bytes)
                .context("async device DSA write source bytes underflow usize")?;
            unsafe {
                self.library.copy_d2d_2d_async(
                    device_buffer_byte_view(
                        self.storage.cache,
                        main_io.offset_bytes,
                        main_io.payload_bytes,
                        "async device main KV write destination",
                    )?,
                    main_row_bytes,
                    src,
                    logical_row_bytes,
                    main_row_bytes,
                    rows,
                    cuda_stream,
                )?;
                self.library.copy_d2d_2d_async(
                    device_buffer_byte_view(
                        self.storage.cache,
                        dsa_io.offset_bytes,
                        dsa_io.payload_bytes,
                        "async device DSA write destination",
                    )?,
                    dsa_row_bytes,
                    device_buffer_byte_view(
                        src,
                        main_row_bytes,
                        dsa_source_bytes,
                        "async device DSA write source",
                    )?,
                    logical_row_bytes,
                    dsa_row_bytes,
                    rows,
                    cuda_stream,
                )?;
            }
        } else {
            unsafe {
                self.library.copy_d2d_async(
                    device_buffer_byte_view(
                        self.storage.cache,
                        main_io.offset_bytes,
                        main_io.payload_bytes,
                        "async device main KV write destination",
                    )?,
                    src,
                    main_io.payload_bytes,
                    cuda_stream,
                )?;
            }
        }
        Ok(logical_io)
    }

    #[allow(dead_code)]
    pub(in crate::commands::real_full) unsafe fn read_block_to_device_async(
        &self,
        descriptor: &KvBlockDescriptor,
        dst: GlmrtDeviceBuffer,
        cuda_stream: *mut c_void,
    ) -> Result<RealFullDeviceKvBlockIo> {
        let logical_io = real_full_device_kv_block_io(&self.config, descriptor)?;
        validate_contiguous_payload_buffer(
            "async device KV logical block read dst",
            dst,
            logical_io.payload_bytes,
        )?;
        let main_io = self.physical_main_kv_block_io(descriptor)?;
        if let Some(dsa_io) = self.physical_dsa_bf16_block_io(descriptor)? {
            let rows = descriptor.token_count;
            anyhow::ensure!(rows > 0, "async device KV read requires nonempty block");
            let main_row_bytes = main_io.payload_bytes / rows;
            let dsa_row_bytes = dsa_io.payload_bytes / rows;
            let logical_row_bytes = main_row_bytes
                .checked_add(dsa_row_bytes)
                .context("async device logical KV read row bytes overflow usize")?;
            let dsa_destination_bytes = logical_io
                .payload_bytes
                .checked_sub(main_row_bytes)
                .context("async device DSA read destination bytes underflow usize")?;
            unsafe {
                self.library.copy_d2d_2d_async(
                    dst,
                    logical_row_bytes,
                    device_buffer_byte_view(
                        self.storage.cache,
                        main_io.offset_bytes,
                        main_io.payload_bytes,
                        "async device main KV read source",
                    )?,
                    main_row_bytes,
                    main_row_bytes,
                    rows,
                    cuda_stream,
                )?;
                self.library.copy_d2d_2d_async(
                    device_buffer_byte_view(
                        dst,
                        main_row_bytes,
                        dsa_destination_bytes,
                        "async device DSA read destination",
                    )?,
                    logical_row_bytes,
                    device_buffer_byte_view(
                        self.storage.cache,
                        dsa_io.offset_bytes,
                        dsa_io.payload_bytes,
                        "async device DSA read source",
                    )?,
                    dsa_row_bytes,
                    dsa_row_bytes,
                    rows,
                    cuda_stream,
                )?;
            }
        } else {
            unsafe {
                self.library.copy_d2d_async(
                    dst,
                    device_buffer_byte_view(
                        self.storage.cache,
                        main_io.offset_bytes,
                        main_io.payload_bytes,
                        "async device main KV read source",
                    )?,
                    main_io.payload_bytes,
                    cuda_stream,
                )?;
            }
        }
        Ok(logical_io)
    }
}

struct DeviceKvBatchPlan {
    descriptors: Vec<KvBlockDescriptor>,
    ios: Vec<RealFullDeviceKvBlockIo>,
    payload_offsets: Vec<u64>,
    cache_offsets: Vec<u64>,
    block_bytes: Vec<u64>,
    total_bytes: usize,
}

impl DeviceKvBatchPlan {
    fn new(config: &KvCacheConfig, descriptors: &[KvBlockDescriptor]) -> Result<Self> {
        Self::new_from_descriptors(config, descriptors.iter())
    }

    fn new_from_descriptors<'a>(
        config: &KvCacheConfig,
        descriptors: impl IntoIterator<Item = &'a KvBlockDescriptor>,
    ) -> Result<Self> {
        let descriptors = descriptors.into_iter().cloned().collect::<Vec<_>>();
        let ios = descriptors
            .iter()
            .map(|descriptor| real_full_device_kv_block_io(config, descriptor))
            .collect::<Result<Vec<_>>>()?;
        Self::from_descriptors_and_ios(descriptors, ios)
    }

    fn from_descriptors_and_ios(
        descriptors: Vec<KvBlockDescriptor>,
        ios: Vec<RealFullDeviceKvBlockIo>,
    ) -> Result<Self> {
        anyhow::ensure!(
            descriptors.len() == ios.len(),
            "device KV batch descriptor/IO count mismatch"
        );
        let mut payload_offsets = Vec::with_capacity(ios.len());
        let mut cache_offsets = Vec::with_capacity(ios.len());
        let mut block_bytes = Vec::with_capacity(ios.len());
        let mut total_bytes = 0_usize;
        for io in &ios {
            payload_offsets.push(usize_to_u64("device KV payload offset", total_bytes)?);
            cache_offsets.push(usize_to_u64("device KV cache offset", io.offset_bytes)?);
            block_bytes.push(usize_to_u64("device KV block bytes", io.payload_bytes)?);
            total_bytes = total_bytes
                .checked_add(io.payload_bytes)
                .with_context(|| "device KV batch payload byte count overflows usize")?;
        }
        Ok(Self {
            descriptors,
            ios,
            payload_offsets,
            cache_offsets,
            block_bytes,
            total_bytes,
        })
    }

    fn empty() -> Result<Self> {
        Self::from_descriptors_and_ios(Vec::new(), Vec::new())
    }

    fn block_count(&self) -> usize {
        self.ios.len()
    }
}

struct DeviceBufferGuard<'a> {
    library: &'a NativeLibrary,
    buffer: GlmrtDeviceBuffer,
    owned: bool,
}

impl<'a> DeviceBufferGuard<'a> {
    fn new(library: &'a NativeLibrary, bytes: usize) -> Result<Self> {
        if bytes == 0 {
            anyhow::bail!("device buffer guard requires a nonzero allocation");
        }
        let buffer = library.alloc_device_buffer(bytes)?;
        Ok(Self {
            library,
            buffer,
            owned: true,
        })
    }

    fn borrowed(library: &'a NativeLibrary, buffer: GlmrtDeviceBuffer) -> Self {
        Self {
            library,
            buffer,
            owned: false,
        }
    }

    fn into_buffer(mut self) -> Result<GlmrtDeviceBuffer> {
        if !self.owned {
            let copy = Self::new(self.library, self.buffer.bytes)?;
            self.library
                .copy_d2d(copy.buffer, self.buffer, self.buffer.bytes)
                .context("copying borrowed device buffer before ownership transfer")?;
            return copy.into_buffer();
        }
        let buffer = self.buffer;
        self.buffer = GlmrtDeviceBuffer::default();
        self.owned = false;
        Ok(buffer)
    }
}

impl Drop for DeviceBufferGuard<'_> {
    fn drop(&mut self) {
        if self.owned {
            let _ = self.library.free_device_buffer(&mut self.buffer);
        }
    }
}

fn validate_contiguous_payload_buffer(
    context: &str,
    buffer: GlmrtDeviceBuffer,
    required_bytes: usize,
) -> Result<()> {
    if buffer.ptr.is_null() {
        anyhow::bail!("{context} buffer pointer is null");
    }
    if buffer.bytes < required_bytes {
        anyhow::bail!(
            "{context} buffer is too small: has {} bytes, needs {required_bytes}",
            buffer.bytes
        );
    }
    Ok(())
}

fn device_buffer_byte_view(
    buffer: GlmrtDeviceBuffer,
    offset_bytes: usize,
    view_bytes: usize,
    context: &str,
) -> Result<GlmrtDeviceBuffer> {
    if buffer.ptr.is_null() {
        anyhow::bail!("{context} base buffer pointer is null");
    }
    let end = offset_bytes
        .checked_add(view_bytes)
        .with_context(|| format!("{context} byte view end overflows usize"))?;
    if end > buffer.bytes {
        anyhow::bail!(
            "{context} byte view [{}..{}) exceeds buffer bytes {}",
            offset_bytes,
            end,
            buffer.bytes
        );
    }
    Ok(GlmrtDeviceBuffer {
        ptr: buffer.ptr.cast::<u8>().wrapping_add(offset_bytes).cast(),
        bytes: view_bytes,
        device_id: buffer.device_id,
        flags: buffer.flags,
    })
}

fn zero_device_buffer_bytes(
    library: &NativeLibrary,
    buffer: GlmrtDeviceBuffer,
    bytes: usize,
    context: &str,
) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    validate_contiguous_payload_buffer(context, buffer, bytes)?;
    library
        .cuda_zero_bytes(buffer, bytes)
        .with_context(|| format!("zeroing {context} with CUDA byte memset"))
}

fn usize_to_u64(context: &str, value: usize) -> Result<u64> {
    u64::try_from(value).with_context(|| format!("{context} does not fit in u64"))
}

fn usize_to_i32(context: &str, value: usize) -> Result<i32> {
    i32::try_from(value).with_context(|| format!("{context} does not fit in i32"))
}

fn u64_slice_bytes(values: &[u64]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn u32_slice_bytes(values: &[u32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn descriptor_positions_u32_into(
    descriptors: &[KvBlockDescriptor],
    positions: &mut Vec<u32>,
) -> Result<()> {
    let rows = descriptors
        .iter()
        .map(|descriptor| descriptor.token_count)
        .sum();
    positions.clear();
    positions.reserve(rows);
    for descriptor in descriptors {
        for offset in 0..descriptor.token_count {
            let position = descriptor
                .token_start
                .0
                .checked_add(offset as u64)
                .context("KV descriptor position overflows u64")?;
            positions
                .push(u32::try_from(position).context("KV descriptor position does not fit u32")?);
        }
    }
    Ok(())
}

fn fill_repeated_bf16_value(
    bytes: &mut Vec<u8>,
    value_count: usize,
    value: f32,
    context: &str,
) -> Result<()> {
    let byte_count = value_count
        .checked_mul(std::mem::size_of::<u16>())
        .with_context(|| format!("{context} BF16 byte count overflow"))?;
    bytes.clear();
    bytes.reserve(byte_count);
    let bf16 = ((value.to_bits() >> 16) as u16).to_le_bytes();
    for _ in 0..value_count {
        bytes.extend_from_slice(&bf16);
    }
    Ok(())
}

fn fill_zeroed_bf16_values(bytes: &mut Vec<u8>, value_count: usize, context: &str) -> Result<()> {
    let byte_count = value_count
        .checked_mul(std::mem::size_of::<u16>())
        .with_context(|| format!("{context} BF16 byte count overflow"))?;
    bytes.clear();
    bytes.resize(byte_count, 0);
    Ok(())
}

fn write_bf16_value(bytes: &mut [u8], value_index: usize, value: f32) {
    let start = value_index * std::mem::size_of::<u16>();
    let bf16 = ((value.to_bits() >> 16) as u16).to_le_bytes();
    bytes[start..start + std::mem::size_of::<u16>()].copy_from_slice(&bf16);
}

fn fill_scheduler_attention_kv_norm_weight_bf16(bytes: &mut Vec<u8>) -> Result<()> {
    fill_repeated_bf16_value(
        bytes,
        GLM52_MLA_KV_LORA_RANK,
        1.0,
        "scheduler device MLA attention kv norm weight",
    )
}

fn fill_scheduler_attention_kv_b_weight_bf16(bytes: &mut Vec<u8>) -> Result<()> {
    let projected_width = REAL_FULL_SCHEDULER_DEVICE_ATTENTION_HEADS
        * (REAL_FULL_SCHEDULER_DEVICE_ATTENTION_NOPE_DIM
            + REAL_FULL_SCHEDULER_DEVICE_ATTENTION_VALUE_DIM);
    fill_zeroed_bf16_values(
        bytes,
        projected_width * GLM52_MLA_KV_LORA_RANK,
        "scheduler device MLA attention kv_b weight",
    )?;
    for out_col in 0..projected_width {
        let primary = (out_col * 23 + 7) % GLM52_MLA_KV_LORA_RANK;
        let secondary = (out_col * 31 + 11) % GLM52_MLA_KV_LORA_RANK;
        write_bf16_value(
            bytes,
            out_col * GLM52_MLA_KV_LORA_RANK + primary,
            if out_col % 2 == 0 { 0.125 } else { -0.09375 },
        );
        if secondary != primary {
            write_bf16_value(
                bytes,
                out_col * GLM52_MLA_KV_LORA_RANK + secondary,
                if out_col % 3 == 0 { 0.0625 } else { -0.046875 },
            );
        }
    }
    Ok(())
}

fn fill_scheduler_attention_query_projection_weight_bf16(bytes: &mut Vec<u8>) -> Result<()> {
    let projected_width = REAL_FULL_SCHEDULER_DEVICE_ATTENTION_HEADS
        * (REAL_FULL_SCHEDULER_DEVICE_ATTENTION_NOPE_DIM + GLM52_MLA_QK_ROPE_HEAD_DIM);
    fill_zeroed_bf16_values(
        bytes,
        projected_width * GLM52_HIDDEN_SIZE,
        "scheduler device MLA attention query projection weight",
    )?;
    for out_col in 0..projected_width {
        let primary = (out_col * 97 + 13) % GLM52_HIDDEN_SIZE;
        let secondary = (out_col * 193 + 29) % GLM52_HIDDEN_SIZE;
        write_bf16_value(
            bytes,
            out_col * GLM52_HIDDEN_SIZE + primary,
            if out_col % 2 == 0 {
                0.03125
            } else {
                -0.0234375
            },
        );
        if secondary != primary {
            write_bf16_value(
                bytes,
                out_col * GLM52_HIDDEN_SIZE + secondary,
                if out_col % 5 == 0 {
                    0.015625
                } else {
                    -0.01171875
                },
            );
        }
    }
    Ok(())
}

fn fill_scheduler_attention_output_projection_weight_bf16(bytes: &mut Vec<u8>) -> Result<()> {
    let input_width =
        REAL_FULL_SCHEDULER_DEVICE_ATTENTION_HEADS * REAL_FULL_SCHEDULER_DEVICE_ATTENTION_VALUE_DIM;
    fill_zeroed_bf16_values(
        bytes,
        GLM52_HIDDEN_SIZE * input_width,
        "scheduler device MLA attention output projection weight",
    )?;
    for out_row in 0..GLM52_HIDDEN_SIZE {
        let primary = (out_row * 17 + 3) % input_width;
        let secondary = (out_row * 29 + 1) % input_width;
        write_bf16_value(
            bytes,
            out_row * input_width + primary,
            if out_row % 2 == 0 {
                0.078125
            } else {
                -0.0546875
            },
        );
        if secondary != primary {
            write_bf16_value(
                bytes,
                out_row * input_width + secondary,
                if out_row % 7 == 0 {
                    0.0390625
                } else {
                    -0.02734375
                },
            );
        }
    }
    Ok(())
}

fn scheduler_attention_project_query_from_hidden_bf16_into(
    layer_id: LayerId,
    query_hidden: GlmrtDeviceBuffer,
    query_rows: usize,
    query_projection_weight: GlmrtDeviceBuffer,
    projected_query: GlmrtDeviceBuffer,
) -> Result<()> {
    if query_rows == 0 {
        anyhow::bail!("scheduler device MLA attention hidden query requires at least one row");
    }
    let projected_width = scheduler_attention_projected_query_width();
    let hidden_bytes = query_rows
        .checked_mul(GLM52_HIDDEN_SIZE)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("scheduler device MLA attention hidden query byte count overflow")?;
    let projection_weight_bytes = projected_width
        .checked_mul(GLM52_HIDDEN_SIZE)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("scheduler device MLA attention query projection weight byte count overflow")?;
    let projected_bytes = scheduler_attention_projected_query_bytes(query_rows)?;
    validate_contiguous_payload_buffer(
        "scheduler device MLA attention hidden query",
        query_hidden,
        hidden_bytes,
    )?;
    validate_contiguous_payload_buffer(
        "scheduler device MLA attention query projection weight",
        query_projection_weight,
        projection_weight_bytes,
    )?;
    validate_contiguous_payload_buffer(
        "scheduler device MLA attention projected query output",
        projected_query,
        projected_bytes,
    )?;
    if query_hidden.device_id != query_projection_weight.device_id {
        anyhow::bail!(
            "scheduler device MLA attention hidden query is on CUDA device {}, but query projection weight is on device {}",
            query_hidden.device_id,
            query_projection_weight.device_id
        );
    }
    if projected_query.device_id != query_hidden.device_id {
        anyhow::bail!(
            "scheduler device MLA attention projected query output is on CUDA device {}, but hidden query is on device {}",
            projected_query.device_id,
            query_hidden.device_id
        );
    }
    linear_rows_bf16_device_buffers_for_layer(
        layer_id.0 as usize,
        query_hidden,
        query_projection_weight,
        projected_query,
        query_rows,
        GLM52_HIDDEN_SIZE,
        projected_width,
    )
    .context("executing scheduler device MLA attention hidden query projection graph")?;
    Ok(())
}

fn scheduler_attention_project_output_to_hidden_bf16(
    layer_id: LayerId,
    library: &'static NativeLibrary,
    attention_output: GlmrtDeviceBuffer,
    rows: usize,
    input_width: usize,
    output_projection_weight: GlmrtDeviceBuffer,
) -> Result<DeviceBufferGuard<'static>> {
    if rows == 0 {
        anyhow::bail!("scheduler device MLA attention output projection requires at least one row");
    }
    if input_width == 0 {
        anyhow::bail!(
            "scheduler device MLA attention output projection requires nonzero input width"
        );
    }
    let input_bytes = rows
        .checked_mul(input_width)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("scheduler device MLA attention output projection input byte count overflow")?;
    let projection_weight_bytes = GLM52_HIDDEN_SIZE
        .checked_mul(input_width)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("scheduler device MLA attention output projection weight byte count overflow")?;
    let projected_bytes = rows
        .checked_mul(GLM52_HIDDEN_SIZE)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("scheduler device MLA attention output projection byte count overflow")?;
    validate_contiguous_payload_buffer(
        "scheduler device MLA attention compact output",
        attention_output,
        input_bytes,
    )?;
    validate_contiguous_payload_buffer(
        "scheduler device MLA attention output projection weight",
        output_projection_weight,
        projection_weight_bytes,
    )?;
    if attention_output.device_id != output_projection_weight.device_id {
        anyhow::bail!(
            "scheduler device MLA attention compact output is on CUDA device {}, but output projection weight is on device {}",
            attention_output.device_id,
            output_projection_weight.device_id
        );
    }
    let projected = DeviceBufferGuard::new(library, projected_bytes)
        .context("allocating scheduler device MLA attention hidden-width output")?;
    if projected.buffer.device_id != attention_output.device_id {
        anyhow::bail!(
            "scheduler device MLA attention hidden-width output is on CUDA device {}, but compact output is on device {}",
            projected.buffer.device_id,
            attention_output.device_id
        );
    }
    linear_rows_bf16_device_buffers_for_layer(
        layer_id.0 as usize,
        attention_output,
        output_projection_weight,
        projected.buffer,
        rows,
        input_width,
        GLM52_HIDDEN_SIZE,
    )
    .context("executing scheduler device MLA attention output projection graph")?;
    Ok(projected)
}

fn scheduler_attention_project_output_to_hidden_bf16_preloaded_resident(
    attention_output: GlmrtDeviceBuffer,
    rows: usize,
    input_width: usize,
    output_projection_weight_name: &str,
) -> Result<DeviceBf16Output> {
    if rows == 0 {
        anyhow::bail!("scheduler device MLA attention output projection requires at least one row");
    }
    if input_width == 0 {
        anyhow::bail!(
            "scheduler device MLA attention output projection requires nonzero input width"
        );
    }
    if coordinator_w8a16_o_proj_decode_enabled() {
        linear_rows_w8a16_preloaded_resident_weight_device_output(
            output_projection_weight_name,
            attention_output,
            rows,
            input_width,
            GLM52_HIDDEN_SIZE,
        )
    } else {
        linear_rows_bf16_preloaded_resident_weight_device_output(
            output_projection_weight_name,
            attention_output,
            None,
            rows,
            input_width,
            GLM52_HIDDEN_SIZE,
            GLM52_HIDDEN_SIZE,
        )
    }
    .context("executing scheduler device MLA attention graph-aware output projection")
}

fn fill_scheduler_attention_projected_query_bf16(
    bytes: &mut Vec<u8>,
    query_rows: usize,
) -> Result<()> {
    if query_rows == 0 {
        anyhow::bail!("scheduler device MLA attention requires at least one query row");
    }
    let value_count = query_rows
        .checked_mul(scheduler_attention_projected_query_width())
        .context("scheduler device MLA attention projected query value count overflow")?;
    let byte_count = scheduler_attention_projected_query_bytes(query_rows)?;
    bytes.clear();
    bytes.reserve(byte_count);
    for index in 0..value_count {
        let value = ((index % 29) as f32 - 14.0) / 128.0;
        bytes.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
    }
    Ok(())
}

fn scheduler_attention_projected_query_width() -> usize {
    REAL_FULL_SCHEDULER_DEVICE_ATTENTION_HEADS
        * (REAL_FULL_SCHEDULER_DEVICE_ATTENTION_NOPE_DIM + GLM52_MLA_QK_ROPE_HEAD_DIM)
}

fn scheduler_attention_projected_query_bytes(query_rows: usize) -> Result<usize> {
    if query_rows == 0 {
        anyhow::bail!("scheduler device MLA attention requires at least one query row");
    }
    query_rows
        .checked_mul(scheduler_attention_projected_query_width())
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("scheduler device MLA attention projected query byte count overflow")
}

impl Drop for RealFullDeviceKvStorage<'_> {
    fn drop(&mut self) {
        if !self.dsa_index_k_cache_b12x.ptr.is_null() {
            let _ = self
                .library
                .free_device_buffer(&mut self.dsa_index_k_cache_b12x);
        }
        let _ = self.library.free_device_buffer(&mut self.cache);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::real_full::dense::math::{bf16_bytes_from_f32, bf16_bytes_to_f32};
    use glmrt_core::{
        KvWriteState, LayerId, PositionId, GLM52_MLA_MXFP4_CODE_BYTES_PER_TOKEN,
        GLM52_MLA_MXFP4_PADDING_BYTES_PER_TOKEN, GLM52_MLA_MXFP4_SCALE_BYTES_PER_TOKEN,
        GLM52_MTP_LAYER_ID,
    };

    #[test]
    fn target_kv_physical_spans_split_and_coalesce_page_runs() -> Result<()> {
        assert_eq!(
            target_kv_physical_spans(None, 128, 256, 63, 66)?,
            vec![TargetKvPhysicalSpan {
                logical_token_offset: 0,
                physical_token_start: 191,
                token_count: 66,
            }]
        );
        assert_eq!(
            target_kv_physical_spans(Some(&[4, 7, 8, 2]), 0, 256, 63, 130)?,
            vec![
                TargetKvPhysicalSpan {
                    logical_token_offset: 0,
                    physical_token_start: 4 * 64 + 63,
                    token_count: 1,
                },
                TargetKvPhysicalSpan {
                    logical_token_offset: 1,
                    physical_token_start: 7 * 64,
                    token_count: 128,
                },
                TargetKvPhysicalSpan {
                    logical_token_offset: 129,
                    physical_token_start: 2 * 64,
                    token_count: 1,
                },
            ]
        );
        assert!(target_kv_physical_spans(Some(&[1]), 0, 256, 64, 1).is_err());
        Ok(())
    }

    #[test]
    fn compressed_mla_suffix_attention_is_limited_to_small_cached_queries() {
        assert!(!use_compressed_mla_suffix_attention(0, 1));
        assert!(!use_compressed_mla_suffix_attention(1_024, 0));
        for query_rows in 1..=REAL_FULL_COMPRESSED_MLA_MAX_QUERY_ROWS {
            assert!(use_compressed_mla_suffix_attention(1_024, query_rows));
        }
        assert!(!use_compressed_mla_suffix_attention(
            1_024,
            REAL_FULL_COMPRESSED_MLA_MAX_QUERY_ROWS + 1
        ));
    }

    #[test]
    fn packed_fp8_mla_suffix_batches_every_supported_mtp_width() {
        assert!(!use_packed_fp8_mla_suffix(0, true));
        assert!(use_packed_fp8_mla_suffix(1, false));
        for query_rows in 2..=REAL_FULL_COMPRESSED_MLA_MAX_QUERY_ROWS {
            assert!(use_packed_fp8_mla_suffix(query_rows, true));
            assert!(!use_packed_fp8_mla_suffix(query_rows, false));
        }
        assert!(!use_packed_fp8_mla_suffix(
            REAL_FULL_COMPRESSED_MLA_MAX_QUERY_ROWS + 1,
            true
        ));
    }

    #[test]
    fn attention_ready_frontier_capacity_is_bounded_for_long_contexts() {
        assert_eq!(
            attention_ready_frontier_capacity_tokens_with_limit(4_096, 16 * 1024),
            4_096
        );
        assert_eq!(
            attention_ready_frontier_capacity_tokens_with_limit(128 * 1024, 16 * 1024),
            16 * 1024
        );
        assert_eq!(
            attention_ready_frontier_capacity_tokens_with_limit(256 * 1024, 64 * 1024),
            64 * 1024
        );
    }

    #[test]
    fn attention_ready_frontier_rewind_truncates_matching_mtp_suffix() -> Result<()> {
        let mut slot = Some(DeviceKvAttentionReadyFrontier {
            reservation_id: 7,
            sequence_id: "sequence".to_owned(),
            layer_id: LayerId(GLM52_MTP_LAYER_ID as u32),
            token_start: 32,
            rows: 12,
        });
        rewind_device_kv_attention_ready_frontier(
            &mut slot,
            7,
            "sequence",
            LayerId(GLM52_MTP_LAYER_ID as u32),
            39,
        )?;
        assert_eq!(slot.expect("rewound frontier remains visible").rows, 7);
        Ok(())
    }

    #[test]
    fn attention_ready_frontier_rewind_preserves_unrelated_sequence() -> Result<()> {
        let expected = DeviceKvAttentionReadyFrontier {
            reservation_id: 7,
            sequence_id: "sequence".to_owned(),
            layer_id: LayerId(GLM52_MTP_LAYER_ID as u32),
            token_start: 32,
            rows: 12,
        };
        let mut slot = Some(expected.clone());
        rewind_device_kv_attention_ready_frontier(
            &mut slot,
            8,
            "other",
            LayerId(GLM52_MTP_LAYER_ID as u32),
            39,
        )?;
        assert_eq!(slot, Some(expected));
        Ok(())
    }

    fn upload_device_bytes(
        library: &'static NativeLibrary,
        bytes: &[u8],
    ) -> Result<DeviceBufferGuard<'static>> {
        let guard = DeviceBufferGuard::new(library, bytes.len())?;
        library.copy_h2d(guard.buffer, bytes)?;
        Ok(guard)
    }

    fn u16_values_to_bytes(values: &[u16]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect()
    }

    fn mla_kv_payload_rows(
        rows: usize,
        include_dsa: bool,
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>, Option<Vec<u8>>) {
        let dsa_dim = if include_dsa {
            GLM52_DSA_INDEX_HEAD_DIM
        } else {
            0
        };
        let mut payload_values = Vec::new();
        let mut latent_values = Vec::new();
        let mut rope_values = Vec::new();
        let mut dsa_values = Vec::new();
        for row in 0..rows {
            for col in 0..GLM52_MLA_KV_LORA_RANK {
                let value = (1000 + row * 2048 + col) as u16;
                payload_values.push(value);
                latent_values.push(value);
            }
            for col in 0..GLM52_MLA_QK_ROPE_HEAD_DIM {
                let value = (2000 + row * 2048 + col) as u16;
                payload_values.push(value);
                rope_values.push(value);
            }
            for col in 0..dsa_dim {
                let value = (3000 + row * 2048 + col) as u16;
                payload_values.push(value);
                dsa_values.push(value);
            }
        }
        (
            u16_values_to_bytes(&payload_values),
            u16_values_to_bytes(&latent_values),
            u16_values_to_bytes(&rope_values),
            include_dsa.then(|| u16_values_to_bytes(&dsa_values)),
        )
    }

    fn dsa_projected_kv_payload_rows(rows: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut projected_kv_a_bf16 = Vec::new();
        let mut dsa_key_bf16 = Vec::new();
        let mut expected_payload = Vec::new();
        for row in 0..rows {
            let latent = (0..GLM52_MLA_KV_LORA_RANK)
                .map(|index| ((row * 17 + index % 31) as f32 - 15.0) / 128.0)
                .collect::<Vec<_>>();
            let rope = (0..GLM52_MLA_QK_ROPE_HEAD_DIM)
                .map(|index| ((row * 11 + index % 23) as f32 - 11.0) / 96.0)
                .collect::<Vec<_>>();
            let dsa = (0..GLM52_DSA_INDEX_HEAD_DIM)
                .map(|index| ((row * 7 + index % 19) as f32 - 9.0) / 64.0)
                .collect::<Vec<_>>();
            let mut main_row = bf16_bytes_from_f32(&latent);
            main_row.extend_from_slice(&bf16_bytes_from_f32(&rope));
            let dsa_row = bf16_bytes_from_f32(&dsa);
            projected_kv_a_bf16.extend_from_slice(&main_row);
            dsa_key_bf16.extend_from_slice(&dsa_row);
            expected_payload.extend_from_slice(&main_row);
            expected_payload.extend_from_slice(&dsa_row);
        }
        (projected_kv_a_bf16, dsa_key_bf16, expected_payload)
    }

    fn mxfp4_representable_projected_kv_payload_rows(rows: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        const CODEBOOK: [f32; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        let mut projected_kv_a_bf16 = Vec::new();
        let mut expected_latent_bf16 = Vec::new();
        let mut expected_rope_bf16 = Vec::new();
        for row in 0..rows {
            let latent = (0..GLM52_MLA_KV_LORA_RANK)
                .map(|index| CODEBOOK[(row + index) % CODEBOOK.len()])
                .collect::<Vec<_>>();
            let rope = (0..GLM52_MLA_QK_ROPE_HEAD_DIM)
                .map(|index| ((row * 11 + index % 23) as f32 - 11.0) / 96.0)
                .collect::<Vec<_>>();
            let latent_bf16 = bf16_bytes_from_f32(&latent);
            let rope_bf16 = bf16_bytes_from_f32(&rope);
            projected_kv_a_bf16.extend_from_slice(&latent_bf16);
            projected_kv_a_bf16.extend_from_slice(&rope_bf16);
            expected_latent_bf16.extend_from_slice(&latent_bf16);
            expected_rope_bf16.extend_from_slice(&rope_bf16);
        }
        (
            projected_kv_a_bf16,
            expected_latent_bf16,
            expected_rope_bf16,
        )
    }

    fn expected_rope_bf16(
        input_bf16: &[u8],
        positions: &[u32],
        rotary_dim: usize,
        theta: f32,
    ) -> Vec<u8> {
        let input = bf16_bytes_to_f32(input_bf16).expect("BF16 test RoPE input");
        let mut output = Vec::with_capacity(input.len());
        for (row, position) in positions.iter().copied().enumerate() {
            let row_start = row * rotary_dim;
            for pair in 0..rotary_dim / 2 {
                let offset = row_start + pair * 2;
                let angle = position as f32 * theta.powf(-2.0 * pair as f32 / rotary_dim as f32);
                let cos_value = angle.cos();
                let sin_value = angle.sin();
                let even = input[offset];
                let odd = input[offset + 1];
                output.push(even * cos_value - odd * sin_value);
                output.push(even * sin_value + odd * cos_value);
            }
        }
        bf16_bytes_from_f32(&output)
    }

    fn expected_mla_rope_attention_bf16(
        q_nope_bf16: &[u8],
        q_rope_bf16: &[u8],
        k_nope_bf16: &[u8],
        k_rope_bf16: &[u8],
        values_bf16: &[u8],
        rows: usize,
        heads: usize,
        nope_dim: usize,
        rope_dim: usize,
        v_dim: usize,
        scale: f32,
    ) -> Result<Vec<u8>> {
        let q_nope = bf16_bytes_to_f32(q_nope_bf16)?;
        let q_rope = bf16_bytes_to_f32(q_rope_bf16)?;
        let k_nope = bf16_bytes_to_f32(k_nope_bf16)?;
        let k_rope = bf16_bytes_to_f32(k_rope_bf16)?;
        let values = bf16_bytes_to_f32(values_bf16)?;
        let mut output = vec![0.0_f32; rows * heads * v_dim];
        for row in 0..rows {
            for head in 0..heads {
                let q_nope_base = (row * heads + head) * nope_dim;
                let q_rope_base = (row * heads + head) * rope_dim;
                let q_nope_vec = &q_nope[q_nope_base..q_nope_base + nope_dim];
                let q_rope_vec = &q_rope[q_rope_base..q_rope_base + rope_dim];
                let mut max_score = f32::NEG_INFINITY;
                for key_row in 0..=row {
                    let k_nope_base = (key_row * heads + head) * nope_dim;
                    let k_rope_base = key_row * rope_dim;
                    let mut nope_dot = 0.0_f32;
                    for col in 0..nope_dim {
                        nope_dot += q_nope_vec[col] * k_nope[k_nope_base + col];
                    }
                    let mut rope_dot = 0.0_f32;
                    for col in 0..rope_dim {
                        rope_dot += q_rope_vec[col] * k_rope[k_rope_base + col];
                    }
                    max_score = max_score.max((nope_dot + rope_dot) * scale);
                }
                for v_col in 0..v_dim {
                    let mut denom = 0.0_f32;
                    let mut acc = 0.0_f32;
                    for key_row in 0..=row {
                        let k_nope_base = (key_row * heads + head) * nope_dim;
                        let k_rope_base = key_row * rope_dim;
                        let mut nope_dot = 0.0_f32;
                        for col in 0..nope_dim {
                            nope_dot += q_nope_vec[col] * k_nope[k_nope_base + col];
                        }
                        let mut rope_dot = 0.0_f32;
                        for col in 0..rope_dim {
                            rope_dot += q_rope_vec[col] * k_rope[k_rope_base + col];
                        }
                        let weight = ((nope_dot + rope_dot) * scale - max_score).exp();
                        denom += weight;
                        acc += weight * values[(key_row * heads + head) * v_dim + v_col];
                    }
                    output[(row * heads + head) * v_dim + v_col] = acc / denom.max(1.0e-12);
                }
            }
        }
        Ok(bf16_bytes_from_f32(&output))
    }

    #[test]
    fn device_kv_block_io_uses_bf16_layer_major_offsets() {
        let config = KvCacheConfig::glm52_phase0(8);
        let descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-a".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(2),
            token_count: 3,
        };

        let io = real_full_device_kv_block_io(&config, &descriptor).unwrap();

        assert_eq!(io.offset_bytes, 3 * 1_408 * 8 + 2 * 1_152);
        assert_eq!(io.payload_bytes, 3 * 1_152);
    }

    #[test]
    fn device_kv_block_io_covers_fp8_and_nvfp4_payloads() {
        let descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-a".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(2),
            token_count: 3,
        };

        let fp8 =
            real_full_device_kv_block_io(&KvCacheConfig::glm52_compressed_fp8(8), &descriptor)
                .unwrap();
        let nvfp4 =
            real_full_device_kv_block_io(&KvCacheConfig::glm52_compressed_nvfp4(8), &descriptor)
                .unwrap();

        assert_eq!(fp8.offset_bytes, 3 * (656 + 128 * 2) * 8 + 2 * 656);
        assert_eq!(fp8.payload_bytes, 3 * 656);
        assert_eq!(
            nvfp4.offset_bytes,
            3 * (GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN + 128 * 2) * 8
                + 2 * GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN
        );
        assert_eq!(nvfp4.payload_bytes, 3 * GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN);
    }

    #[test]
    fn device_kv_physical_fp8_dsa_layer_splits_main_and_index_planes() {
        let config = KvCacheConfig::glm52_compressed_fp8(8);
        let descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-a".to_owned(),
            layer_id: LayerId(0),
            token_start: PositionId(2),
            token_count: 3,
        };

        let logical = real_full_device_kv_block_io(&config, &descriptor).unwrap();
        let main = real_full_device_main_kv_block_io(&config, &descriptor).unwrap();
        let dsa = real_full_device_dsa_bf16_block_io(&config, &descriptor)
            .unwrap()
            .unwrap();
        let next_layer_base = config.layer_base_offset_bytes(LayerId(1)).unwrap();

        assert_eq!(logical.offset_bytes, 2 * (656 + 256));
        assert_eq!(logical.payload_bytes, 3 * (656 + 256));
        assert_eq!(main.offset_bytes, 2 * 656);
        assert_eq!(main.payload_bytes, 3 * 656);
        assert_eq!(dsa.offset_bytes, 8 * 656 + 2 * 256);
        assert_eq!(dsa.payload_bytes, 3 * 256);
        assert!(main.offset_bytes + main.payload_bytes <= 8 * 656);
        assert!(dsa.offset_bytes + dsa.payload_bytes <= next_layer_base);
    }

    #[test]
    fn device_kv_physical_non_dsa_layer_keeps_logical_layout() {
        let config = KvCacheConfig::glm52_compressed_fp8(8);
        let descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-a".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(2),
            token_count: 3,
        };

        let logical = real_full_device_kv_block_io(&config, &descriptor).unwrap();
        let main = real_full_device_main_kv_block_io(&config, &descriptor).unwrap();

        assert_eq!(main.offset_bytes, logical.offset_bytes);
        assert_eq!(main.payload_bytes, logical.payload_bytes);
        assert!(real_full_device_dsa_bf16_block_io(&config, &descriptor)
            .unwrap()
            .is_none());
    }

    #[test]
    fn device_kv_physical_main_span_is_contiguous_across_blocks() {
        let config = KvCacheConfig::glm52_compressed_fp8(8);
        let descriptors = vec![
            KvBlockDescriptor {
                reservation_id: 1,
                sequence_id: "seq-a".to_owned(),
                layer_id: LayerId(0),
                token_start: PositionId(2),
                token_count: 2,
            },
            KvBlockDescriptor {
                reservation_id: 1,
                sequence_id: "seq-a".to_owned(),
                layer_id: LayerId(0),
                token_start: PositionId(4),
                token_count: 1,
            },
        ];

        let (offset, bytes, ios) = contiguous_device_main_kv_block_span(&config, &descriptors)
            .unwrap()
            .unwrap();

        assert_eq!(offset, 2 * 656);
        assert_eq!(bytes, 3 * 656);
        assert_eq!(ios.len(), 2);
    }

    #[test]
    fn device_kv_block_io_rejects_out_of_bounds_ranges() {
        let config = KvCacheConfig::glm52_phase0(8);
        let descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-a".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(7),
            token_count: 2,
        };

        let err = real_full_device_kv_block_io(&config, &descriptor)
            .unwrap_err()
            .to_string();

        assert!(err.contains("outside the glm52-compressed-bf16 cache capacity"));
    }

    #[test]
    fn device_kv_batch_plan_builds_contiguous_payload_offsets() {
        let config = KvCacheConfig::glm52_phase0(8);
        let descriptors = vec![
            KvBlockDescriptor {
                reservation_id: 1,
                sequence_id: "seq-a".to_owned(),
                layer_id: LayerId(3),
                token_start: PositionId(0),
                token_count: 2,
            },
            KvBlockDescriptor {
                reservation_id: 1,
                sequence_id: "seq-a".to_owned(),
                layer_id: LayerId(6),
                token_start: PositionId(4),
                token_count: 1,
            },
        ];

        let batch = DeviceKvBatchPlan::new(&config, &descriptors).unwrap();

        assert_eq!(batch.block_count(), 2);
        assert_eq!(batch.payload_offsets, vec![0, 2 * 1_152]);
        assert_eq!(
            batch.cache_offsets,
            vec![3 * 1_408 * 8, 3 * 1_408 * 8 + 3 * 1_152 * 8 + 4 * 1_408]
        );
        assert_eq!(batch.block_bytes, vec![2 * 1_152, 1_408]);
        assert_eq!(batch.total_bytes, 2 * 1_152 + 1_408);
    }

    #[test]
    fn device_kv_batch_plan_accepts_borrowed_visible_block_descriptors() {
        let config = KvCacheConfig::glm52_phase0(8);
        let first = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-a".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(0),
            token_count: 2,
        };
        let second = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-a".to_owned(),
            layer_id: LayerId(6),
            token_start: PositionId(4),
            token_count: 1,
        };
        let blocks = vec![
            KvBackedBlock {
                write_id: 1,
                descriptor: first,
                state: KvWriteState::Written,
                bytes: Vec::new(),
            },
            KvBackedBlock {
                write_id: 2,
                descriptor: second,
                state: KvWriteState::Written,
                bytes: Vec::new(),
            },
        ];

        let batch = DeviceKvBatchPlan::new_from_descriptors(
            &config,
            blocks.iter().map(|block| &block.descriptor),
        )
        .unwrap();

        assert_eq!(batch.block_count(), 2);
        assert_eq!(batch.payload_offsets, vec![0, 2 * 1_152]);
        assert_eq!(
            batch.cache_offsets,
            vec![3 * 1_408 * 8, 3 * 1_408 * 8 + 3 * 1_152 * 8 + 4 * 1_408]
        );
        assert_eq!(batch.block_bytes, vec![2 * 1_152, 1_408]);
        assert_eq!(batch.total_bytes, 2 * 1_152 + 1_408);
    }

    #[test]
    fn device_kv_roundtrip_rejects_descriptor_payload_mismatch() {
        let config = KvCacheConfig::glm52_phase0(8);
        let descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-a".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(0),
            token_count: 1,
        };

        let err = real_full_device_kv_roundtrip(&config, &[descriptor], &[])
            .unwrap_err()
            .to_string();

        assert!(err.contains("descriptor/payload mismatch"));
    }

    #[test]
    fn device_kv_roundtrip_rejects_payload_byte_mismatch() {
        let config = KvCacheConfig::glm52_compressed_nvfp4(8);
        let descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-a".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(0),
            token_count: 2,
        };

        let err = real_full_device_kv_roundtrip(&config, &[descriptor], &[vec![0_u8; 287]])
            .unwrap_err()
            .to_string();

        assert!(err.contains("payload 0 byte mismatch"));
        assert!(err.contains(&format!(
            "expected {} got 287",
            2 * GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN
        )));
    }

    #[test]
    fn device_kv_roundtrip_plan_reads_final_overlapping_cache_image() {
        let config = KvCacheConfig::glm52_compressed_nvfp4(8);
        let prefix = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-a".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(0),
            token_count: 2,
        };
        let second_token = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-a".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(1),
            token_count: 1,
        };
        let row_bytes = GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN;
        let payloads = vec![vec![0x11_u8; 2 * row_bytes], vec![0x22_u8; row_bytes]];

        let plan = DeviceKvRoundTripPlan::new(&config, &[prefix, second_token.clone()], &payloads)
            .expect("overlapping roundtrip plan");

        assert_eq!(plan.write_descriptors.len(), 1);
        assert_eq!(plan.write_descriptors[0].layer_id, LayerId(3));
        assert_eq!(plan.write_descriptors[0].token_start, PositionId(0));
        assert_eq!(plan.write_descriptors[0].token_count, 2);
        assert_eq!(plan.write_payloads.len(), 1);
        assert_eq!(
            plan.write_payloads[0].len(),
            config.layer_payload_bytes(LayerId(3), 2)
        );
        assert_eq!(plan.write_bytes, plan.write_payloads[0].len());
        assert_eq!(plan.expected_readback.len(), 3 * row_bytes);
        assert!(plan.expected_readback[0..row_bytes]
            .iter()
            .all(|byte| *byte == 0x11));
        assert!(plan.expected_readback[row_bytes..2 * row_bytes]
            .iter()
            .all(|byte| *byte == 0x22));
        assert!(plan.expected_readback[2 * row_bytes..3 * row_bytes]
            .iter()
            .all(|byte| *byte == 0x22));
    }

    #[test]
    fn device_kv_roundtrip_uses_cuda_when_native_available() -> Result<()> {
        let config = KvCacheConfig::glm52_compressed_nvfp4(8);
        let descriptors = vec![
            KvBlockDescriptor {
                reservation_id: 1,
                sequence_id: "seq-a".to_owned(),
                layer_id: LayerId(3),
                token_start: PositionId(0),
                token_count: 1,
            },
            KvBlockDescriptor {
                reservation_id: 1,
                sequence_id: "seq-a".to_owned(),
                layer_id: LayerId(6),
                token_start: PositionId(4),
                token_count: 2,
            },
        ];
        let payloads = descriptors
            .iter()
            .enumerate()
            .map(|(descriptor_index, descriptor)| {
                let bytes = config
                    .descriptor_payload_bytes(descriptor)
                    .expect("test descriptor payload bytes");
                (0..bytes)
                    .map(|byte_index| ((descriptor_index * 31 + byte_index) & 0xff) as u8)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let roundtrip = real_full_device_kv_roundtrip(&config, &descriptors, &payloads)?;
        if roundtrip.status == "cuda-kv-cache-unavailable" {
            return Ok(());
        }

        assert_eq!(roundtrip.status, "cuda-kv-cache-blocks-roundtrip");
        assert_eq!(roundtrip.writes, descriptors.len());
        assert_eq!(roundtrip.reads, descriptors.len());
        assert_eq!(
            roundtrip.bytes,
            payloads.iter().map(Vec::len).sum::<usize>()
        );
        assert!(roundtrip.uses_device_kv_cache);
        Ok(())
    }

    #[test]
    fn device_kv_execution_mirror_reads_live_scheduler_blocks_when_available() -> Result<()> {
        let config = KvCacheConfig::glm52_compressed_nvfp4(8);
        let descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-a".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(2),
            token_count: 2,
        };
        let payload = (0..config.descriptor_payload_bytes(&descriptor).unwrap())
            .map(|index| (index & 0xff) as u8)
            .collect::<Vec<_>>();

        let mut mirror = RealFullDeviceKvExecutionMirror::new(config)?;
        if !mirror.summary().uses_device_kv_cache {
            return Ok(());
        }

        mirror.write_host_blocks(
            std::slice::from_ref(&descriptor),
            std::slice::from_ref(&payload),
        )?;
        mirror.read_visible_blocks(&[KvBackedBlock {
            write_id: 1,
            descriptor: descriptor.clone(),
            state: KvWriteState::Written,
            bytes: payload.clone(),
        }])?;
        assert_eq!(mirror.host_readback_payload_scratch.len(), payload.len());
        let first_host_scratch_ptr = mirror.host_readback_payload_scratch.as_ptr();
        let first_host_scratch_capacity = mirror.host_readback_payload_scratch.capacity();
        assert!(first_host_scratch_capacity >= payload.len());

        mirror.read_visible_blocks(&[KvBackedBlock {
            write_id: 1,
            descriptor,
            state: KvWriteState::Written,
            bytes: payload.clone(),
        }])?;
        assert_eq!(
            mirror.host_readback_payload_scratch.as_ptr(),
            first_host_scratch_ptr
        );
        assert_eq!(
            mirror.host_readback_payload_scratch.capacity(),
            first_host_scratch_capacity
        );

        let summary = mirror.summary();
        assert_eq!(summary.status, "cuda-kv-cache-live-scheduler");
        assert_eq!(summary.writes, 1);
        assert_eq!(summary.reads, 2);
        assert_eq!(summary.bytes, payload.len() * 3);
        assert!(summary.uses_device_kv_cache);
        Ok(())
    }

    #[test]
    fn shared_device_kv_storage_isolates_nonzero_physical_extents() -> Result<()> {
        let config = KvCacheConfig::glm52_compressed_nvfp4(8);
        let descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "logical-sequence".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(1),
            token_count: 2,
        };
        let payload_bytes = config
            .descriptor_payload_bytes(&descriptor)
            .context("shared-extent descriptor must fit the test cache")?;
        let first_payload = vec![0x35; payload_bytes];
        let second_payload = vec![0xca; payload_bytes];

        let mut first = RealFullDeviceKvExecutionMirror::new(config)?;
        if !first.summary().uses_device_kv_cache {
            return Ok(());
        }
        first.rebind_physical_extent(0, 4)?;
        let storage = first
            .storage_handle()
            .context("CUDA mirror should expose shared storage")?;
        let mut second = RealFullDeviceKvExecutionMirror::new_with_storage(storage, 4, 4)?;

        first.write_host_blocks(
            std::slice::from_ref(&descriptor),
            std::slice::from_ref(&first_payload),
        )?;
        second.write_host_blocks(
            std::slice::from_ref(&descriptor),
            std::slice::from_ref(&second_payload),
        )?;

        assert_eq!(
            first
                .read_descriptor_payloads_to_host(std::slice::from_ref(&descriptor))?
                .context("first shared extent should be readable")?,
            vec![first_payload],
        );
        assert_eq!(
            second
                .read_descriptor_payloads_to_host(std::slice::from_ref(&descriptor))?
                .context("second shared extent should be readable")?,
            vec![second_payload],
        );
        Ok(())
    }

    #[test]
    fn device_kv_execution_mirror_unavailable_readbacks_remain_noops() -> Result<()> {
        let config = KvCacheConfig::glm52_compressed_nvfp4(8);
        let descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-unavailable".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(2),
            token_count: 1,
        };
        let payload = vec![
            0x5a;
            config
                .descriptor_payload_bytes(&descriptor)
                .expect("test descriptor payload bytes")
        ];
        let mut mirror = RealFullDeviceKvExecutionMirror::unavailable("cuda-kv-cache-unavailable");

        mirror.read_visible_blocks(&[KvBackedBlock {
            write_id: 1,
            descriptor: descriptor.clone(),
            state: KvWriteState::Written,
            bytes: payload,
        }])?;
        assert!(mirror.host_readback_payload_scratch.is_empty());
        assert!(mirror
            .read_descriptor_payloads_to_host(std::slice::from_ref(&descriptor))?
            .is_none());

        mirror.host_readback_payload_scratch = vec![1, 2, 3];
        let empty_payloads = mirror
            .read_descriptor_payloads_to_host(&[])?
            .expect("empty descriptor reads remain available without CUDA cache");
        assert!(empty_payloads.is_empty());
        assert!(mirror.host_readback_payload_scratch.is_empty());
        Ok(())
    }

    #[test]
    fn device_kv_execution_mirror_returns_descriptor_payloads_when_available() -> Result<()> {
        let config = KvCacheConfig::glm52_phase0(4);
        let descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-prefix".to_owned(),
            layer_id: LayerId(7),
            token_start: PositionId(1),
            token_count: 1,
        };
        let payload = (0..config.descriptor_payload_bytes(&descriptor).unwrap())
            .map(|index| (index.wrapping_mul(11) & 0xff) as u8)
            .collect::<Vec<_>>();

        let mut mirror = RealFullDeviceKvExecutionMirror::new(config)?;
        if !mirror.summary().uses_device_kv_cache {
            return Ok(());
        }

        mirror.write_host_blocks(
            std::slice::from_ref(&descriptor),
            std::slice::from_ref(&payload),
        )?;
        let payloads = mirror
            .read_descriptor_payloads_to_host(std::slice::from_ref(&descriptor))?
            .expect("CUDA-enabled mirror should return descriptor payloads");
        assert_eq!(payloads, vec![payload.clone()]);
        assert!(!mirror.host_readback_payload_scratch.is_empty());

        let empty_payloads = mirror
            .read_descriptor_payloads_to_host(&[])?
            .expect("empty descriptor read should still return an empty payload list");
        assert!(empty_payloads.is_empty());
        assert!(mirror.host_readback_payload_scratch.is_empty());

        let summary = mirror.summary();
        assert_eq!(summary.status, "cuda-kv-cache-live-scheduler");
        assert_eq!(summary.writes, 1);
        assert_eq!(summary.reads, 1);
        assert_eq!(summary.bytes, payload.len() * 2);
        assert!(summary.uses_device_kv_cache);
        Ok(())
    }

    #[test]
    fn device_kv_execution_mirror_reuses_host_readback_payload_buffer_when_available() -> Result<()>
    {
        let config = KvCacheConfig::glm52_phase0(4);
        let first_descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-readback-reuse".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(0),
            token_count: 2,
        };
        let second_descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-readback-reuse".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(1),
            token_count: 1,
        };
        let (payload, _, _, _) = mla_kv_payload_rows(2, false);
        assert_eq!(
            payload.len(),
            config.descriptor_payload_bytes(&first_descriptor).unwrap()
        );

        let mut mirror = RealFullDeviceKvExecutionMirror::new(config.clone())?;
        if !mirror.summary().uses_device_kv_cache {
            return Ok(());
        }

        mirror.write_host_blocks(
            std::slice::from_ref(&first_descriptor),
            std::slice::from_ref(&payload),
        )?;
        let first_readback = mirror
            .read_descriptor_payloads_to_host(std::slice::from_ref(&first_descriptor))?
            .expect("CUDA-enabled mirror should read first descriptor payload");
        assert_eq!(first_readback, vec![payload.clone()]);
        let cache = mirror.cache.as_ref().expect("CUDA mirror should own cache");
        let first_readback_ptr = cache.host_readback_payload.buffer.ptr;
        let first_readback_capacity = cache.host_readback_payload.capacity;
        assert!(!first_readback_ptr.is_null());
        let first_host_scratch_ptr = mirror.host_readback_payload_scratch.as_ptr();
        let first_host_scratch_capacity = mirror.host_readback_payload_scratch.capacity();
        assert_eq!(mirror.host_readback_payload_scratch.len(), payload.len());

        let second_readback = mirror
            .read_descriptor_payloads_to_host(std::slice::from_ref(&second_descriptor))?
            .expect("CUDA-enabled mirror should read second descriptor payload");
        let second_payload_bytes = config.descriptor_payload_bytes(&second_descriptor).unwrap();
        assert_eq!(
            second_readback,
            vec![payload[second_payload_bytes..second_payload_bytes * 2].to_vec()]
        );
        let cache = mirror.cache.as_ref().expect("CUDA mirror should own cache");
        assert_eq!(cache.host_readback_payload.buffer.ptr, first_readback_ptr);
        assert_eq!(
            cache.host_readback_payload.capacity,
            first_readback_capacity
        );
        assert_eq!(
            mirror.host_readback_payload_scratch.as_ptr(),
            first_host_scratch_ptr
        );
        assert_eq!(
            mirror.host_readback_payload_scratch.capacity(),
            first_host_scratch_capacity
        );
        assert_eq!(
            mirror.host_readback_payload_scratch.len(),
            second_payload_bytes
        );

        let summary = mirror.summary();
        assert_eq!(summary.status, "cuda-kv-cache-live-scheduler");
        assert_eq!(summary.writes, 1);
        assert_eq!(summary.reads, 2);
        assert_eq!(summary.bytes, payload.len() * 2 + second_payload_bytes);
        assert!(summary.uses_device_kv_cache);
        Ok(())
    }

    #[test]
    fn device_kv_cache_reuses_pinned_host_write_staging_when_available() -> Result<()> {
        let Ok(library) = cuda_native_library() else {
            return Ok(());
        };
        let config = KvCacheConfig::glm52_compressed_nvfp4(8);
        let mut cache = match RealFullDeviceKvCache::new(library, config.clone()) {
            Ok(cache) => cache,
            Err(error)
                if real_full_device_kv_roundtrip_error_status(&error)
                    == "cuda-kv-cache-unavailable" =>
            {
                return Ok(());
            }
            Err(error) => return Err(error),
        };
        let first_descriptors = vec![
            KvBlockDescriptor {
                reservation_id: 1,
                sequence_id: "seq-pinned".to_owned(),
                layer_id: LayerId(3),
                token_start: PositionId(0),
                token_count: 2,
            },
            KvBlockDescriptor {
                reservation_id: 1,
                sequence_id: "seq-pinned".to_owned(),
                layer_id: LayerId(6),
                token_start: PositionId(4),
                token_count: 1,
            },
        ];
        let first_payloads = first_descriptors
            .iter()
            .enumerate()
            .map(|(descriptor_index, descriptor)| {
                let bytes = config
                    .descriptor_payload_bytes(descriptor)
                    .expect("test descriptor payload bytes");
                (0..bytes)
                    .map(|byte_index| ((descriptor_index * 13 + byte_index) & 0xff) as u8)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let writes =
            cache.write_host_blocks_from_pinned_staging(&first_descriptors, &first_payloads)?;
        assert_eq!(writes.len(), first_descriptors.len());
        let first_host_ptr = cache.host_write_staging.buffer.ptr;
        let first_host_capacity = cache.host_write_staging.capacity;
        let first_device_ptr = cache.device_write_source.buffer.ptr;
        let first_device_capacity = cache.device_write_source.capacity;
        assert!(!first_host_ptr.is_null());
        assert!(!first_device_ptr.is_null());

        let second_descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-pinned".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(1),
            token_count: 1,
        };
        let second_payload_bytes = config
            .descriptor_payload_bytes(&second_descriptor)
            .context("test second descriptor payload bytes")?;
        let second_payload = (0..second_payload_bytes)
            .map(|byte_index| (0xaa_u8).wrapping_add(byte_index as u8))
            .collect::<Vec<_>>();

        let writes = cache.write_host_blocks_from_pinned_staging(
            std::slice::from_ref(&second_descriptor),
            std::slice::from_ref(&second_payload),
        )?;
        assert_eq!(writes.len(), 1);
        assert_eq!(cache.host_write_staging.buffer.ptr, first_host_ptr);
        assert_eq!(cache.host_write_staging.capacity, first_host_capacity);
        assert_eq!(cache.device_write_source.buffer.ptr, first_device_ptr);
        assert_eq!(cache.device_write_source.capacity, first_device_capacity);

        let read_bytes = config
            .descriptor_payload_bytes(&second_descriptor)
            .context("test second descriptor readback bytes")?;
        let readback_device = DeviceBufferGuard::new(library, read_bytes)?;
        cache.read_blocks_to_contiguous_device(
            std::slice::from_ref(&second_descriptor),
            readback_device.buffer,
        )?;
        let mut readback = vec![0_u8; read_bytes];
        library.copy_d2h(&mut readback, readback_device.buffer)?;
        assert_eq!(readback, second_payload);
        Ok(())
    }

    #[test]
    fn device_kv_cache_reuses_host_write_buffers_when_available() -> Result<()> {
        let Ok(library) = cuda_native_library() else {
            return Ok(());
        };
        let config = KvCacheConfig::glm52_compressed_nvfp4(8);
        let mut cache = match RealFullDeviceKvCache::new(library, config.clone()) {
            Ok(cache) => cache,
            Err(error)
                if real_full_device_kv_roundtrip_error_status(&error)
                    == "cuda-kv-cache-unavailable" =>
            {
                return Ok(());
            }
            Err(error) => return Err(error),
        };
        let descriptors = vec![
            KvBlockDescriptor {
                reservation_id: 1,
                sequence_id: "seq-metadata".to_owned(),
                layer_id: LayerId(3),
                token_start: PositionId(0),
                token_count: 2,
            },
            KvBlockDescriptor {
                reservation_id: 1,
                sequence_id: "seq-metadata".to_owned(),
                layer_id: LayerId(6),
                token_start: PositionId(3),
                token_count: 1,
            },
        ];
        let payloads = descriptors
            .iter()
            .enumerate()
            .map(|(descriptor_index, descriptor)| {
                let bytes = config
                    .descriptor_payload_bytes(descriptor)
                    .expect("test descriptor payload bytes");
                (0..bytes)
                    .map(|byte_index| ((descriptor_index * 17 + byte_index) & 0xff) as u8)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let payload_bytes = payloads.iter().map(Vec::len).sum::<usize>();

        cache.write_host_blocks_from_pinned_staging(&descriptors, &payloads)?;
        let host_write_ptr = cache.host_write_staging.buffer.ptr;
        let host_write_capacity = cache.host_write_staging.capacity;
        let device_source_ptr = cache.device_write_source.buffer.ptr;
        let device_source_capacity = cache.device_write_source.capacity;
        assert!(!host_write_ptr.is_null());
        assert!(!device_source_ptr.is_null());

        let readback_device = DeviceBufferGuard::new(library, payload_bytes)?;
        cache.read_blocks_to_contiguous_device(&descriptors, readback_device.buffer)?;
        assert_eq!(cache.host_write_staging.buffer.ptr, host_write_ptr);
        assert_eq!(cache.host_write_staging.capacity, host_write_capacity);
        assert_eq!(cache.device_write_source.buffer.ptr, device_source_ptr);
        assert_eq!(cache.device_write_source.capacity, device_source_capacity);

        let smaller_descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-metadata".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(1),
            token_count: 1,
        };
        let smaller_payload = vec![
            0x5a_u8;
            config
                .descriptor_payload_bytes(&smaller_descriptor)
                .context("test smaller descriptor payload bytes")?
        ];
        cache.write_host_blocks_from_pinned_staging(
            std::slice::from_ref(&smaller_descriptor),
            std::slice::from_ref(&smaller_payload),
        )?;
        assert_eq!(cache.host_write_staging.buffer.ptr, host_write_ptr);
        assert_eq!(cache.host_write_staging.capacity, host_write_capacity);
        assert_eq!(cache.device_write_source.buffer.ptr, device_source_ptr);
        assert_eq!(cache.device_write_source.capacity, device_source_capacity);
        Ok(())
    }

    #[test]
    fn device_kv_execution_mirror_writes_projected_mla_kv_a_device_blocks_when_available(
    ) -> Result<()> {
        let config = KvCacheConfig::glm52_phase0(8);
        let descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-device-kv-a".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(2),
            token_count: 2,
        };
        assert!(!config.layer_has_dsa_indexer(descriptor.layer_id));
        let mut payload = Vec::new();
        for row in 0..descriptor.token_count {
            let latent = (0..GLM52_MLA_KV_LORA_RANK)
                .map(|index| ((row * 17 + index % 31) as f32 - 15.0) / 128.0)
                .collect::<Vec<_>>();
            let rope = (0..GLM52_MLA_QK_ROPE_HEAD_DIM)
                .map(|index| ((row * 11 + index % 23) as f32 - 11.0) / 96.0)
                .collect::<Vec<_>>();
            payload.extend_from_slice(&bf16_bytes_from_f32(&latent));
            payload.extend_from_slice(&bf16_bytes_from_f32(&rope));
        }
        assert_eq!(
            payload.len(),
            config.descriptor_payload_bytes(&descriptor).unwrap()
        );

        let mut mirror = RealFullDeviceKvExecutionMirror::new(config)?;
        if !mirror.summary().uses_device_kv_cache {
            return Ok(());
        }
        let library = cuda_native_library()?;
        let projected_kv_a = upload_device_bytes(library, &payload)?;

        let writes = mirror
            .write_projected_mla_kv_a_device_blocks_bf16(
                std::slice::from_ref(&descriptor),
                projected_kv_a.buffer,
                None,
            )?
            .expect("CUDA-enabled mirror should write projected kv_a device blocks");

        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].payload_bytes, payload.len());
        let readback = mirror
            .read_descriptor_payloads_to_host(std::slice::from_ref(&descriptor))?
            .expect("CUDA-enabled mirror should read projected kv_a device block");
        assert_eq!(readback, vec![payload.clone()]);

        let summary = mirror.summary();
        assert_eq!(summary.status, "cuda-kv-cache-live-scheduler");
        assert_eq!(summary.writes, 1);
        assert_eq!(summary.reads, 1);
        assert_eq!(summary.bytes, payload.len() * 2);
        assert!(summary.uses_device_kv_cache);
        Ok(())
    }

    #[test]
    fn device_kv_execution_mirror_stores_normalized_rotated_mla_rows() -> Result<()> {
        let config = KvCacheConfig::glm52_phase0(8)
            .with_mla_representation(MlaKvCacheRepresentation::NormalizedRotated);
        let descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-prepared-kv-a".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(2),
            token_count: 2,
        };
        let mut projected = Vec::new();
        let mut latent = Vec::new();
        let mut rope = Vec::new();
        for row in 0..descriptor.token_count {
            let latent_row = (0..GLM52_MLA_KV_LORA_RANK)
                .map(|index| ((row * 19 + index % 41) as f32 - 20.0) / 13.0)
                .collect::<Vec<_>>();
            let rope_row = (0..GLM52_MLA_QK_ROPE_HEAD_DIM)
                .map(|index| ((row * 11 + index % 23) as f32 - 11.0) / 17.0)
                .collect::<Vec<_>>();
            let latent_row = bf16_bytes_from_f32(&latent_row);
            let rope_row = bf16_bytes_from_f32(&rope_row);
            projected.extend_from_slice(&latent_row);
            projected.extend_from_slice(&rope_row);
            latent.extend_from_slice(&latent_row);
            rope.extend_from_slice(&rope_row);
        }
        let norm_weight = bf16_bytes_from_f32(
            &(0..GLM52_MLA_KV_LORA_RANK)
                .map(|index| 0.75 + (index % 17) as f32 / 32.0)
                .collect::<Vec<_>>(),
        );

        let mut mirror = RealFullDeviceKvExecutionMirror::new(config)?;
        if !mirror.summary().uses_device_kv_cache {
            return Ok(());
        }
        let library = cuda_native_library()?;
        let projected_device = upload_device_bytes(library, &projected)?;
        let norm_weight_device = upload_device_bytes(library, &norm_weight)?;
        mirror
            .write_projected_mla_kv_a_device_blocks_bf16(
                std::slice::from_ref(&descriptor),
                projected_device.buffer,
                Some(norm_weight_device.buffer),
            )?
            .expect("CUDA-enabled mirror should write prepared projected kv_a rows");
        let stored = mirror
            .read_descriptor_payloads_to_host(std::slice::from_ref(&descriptor))?
            .expect("CUDA-enabled mirror should read prepared projected kv_a rows")
            .remove(0);

        let latent_device = upload_device_bytes(library, &latent)?;
        let normalized_device = DeviceBufferGuard::new(library, latent.len())?;
        rmsnorm_bf16_device_buffers_for_layer(
            descriptor.layer_id.0 as usize,
            latent_device.buffer,
            norm_weight_device.buffer,
            normalized_device.buffer,
            descriptor.token_count,
            GLM52_MLA_KV_LORA_RANK,
            REAL_FULL_SCHEDULER_DEVICE_ATTENTION_EPS,
        )?;
        let positions = [2_u32, 3_u32];
        let positions_device = upload_device_bytes(library, u32_slice_bytes(&positions))?;
        let rope_device = upload_device_bytes(library, &rope)?;
        let rotated_device = DeviceBufferGuard::new(library, rope.len())?;
        rope_bf16_device_buffers_for_layer(
            descriptor.layer_id.0 as usize,
            rope_device.buffer,
            positions_device.buffer,
            rotated_device.buffer,
            descriptor.token_count,
            1,
            GLM52_MLA_QK_ROPE_HEAD_DIM,
            GLM52_MLA_ROPE_THETA,
        )?;
        let mut normalized = vec![0_u8; latent.len()];
        let mut rotated = vec![0_u8; rope.len()];
        library.copy_d2h(&mut normalized, normalized_device.buffer)?;
        library.copy_d2h(&mut rotated, rotated_device.buffer)?;
        let latent_row_bytes = GLM52_MLA_KV_LORA_RANK * std::mem::size_of::<u16>();
        let rope_row_bytes = GLM52_MLA_QK_ROPE_HEAD_DIM * std::mem::size_of::<u16>();
        let mut expected = Vec::with_capacity(stored.len());
        for row in 0..descriptor.token_count {
            expected.extend_from_slice(
                &normalized[row * latent_row_bytes..(row + 1) * latent_row_bytes],
            );
            expected.extend_from_slice(&rotated[row * rope_row_bytes..(row + 1) * rope_row_bytes]);
        }
        assert_eq!(stored, expected);

        let frontier = mirror
            .cache
            .as_mut()
            .expect("CUDA-enabled mirror has a cache")
            .attention_ready_mla_frontier_parts(std::slice::from_ref(&descriptor))?
            .expect("normalized/rotated write stages an attention-ready frontier");
        let mut frontier_latent = vec![0_u8; latent.len()];
        let mut frontier_rope = vec![0_u8; rope.len()];
        library.copy_d2h(&mut frontier_latent, frontier.kv_latent)?;
        library.copy_d2h(&mut frontier_rope, frontier.k_rope)?;
        assert_eq!(frontier_latent, normalized);
        assert_eq!(frontier_rope, rotated);

        let next_layer_descriptor = KvBlockDescriptor {
            layer_id: LayerId(4),
            ..descriptor.clone()
        };
        mirror
            .write_projected_mla_kv_a_device_blocks_bf16(
                std::slice::from_ref(&next_layer_descriptor),
                projected_device.buffer,
                Some(norm_weight_device.buffer),
            )?
            .expect("CUDA-enabled mirror should retain the next layer's projected kv_a rows");
        let retained_frontier = mirror
            .cache
            .as_mut()
            .expect("CUDA-enabled mirror has a cache")
            .attention_ready_mla_frontier_parts(std::slice::from_ref(&descriptor))?
            .expect("preparing the next layer must preserve this layer's attention frontier");
        library.copy_d2h(&mut frontier_latent, retained_frontier.kv_latent)?;
        library.copy_d2h(&mut frontier_rope, retained_frontier.k_rope)?;
        assert_eq!(frontier_latent, normalized);
        assert_eq!(frontier_rope, rotated);
        Ok(())
    }

    #[test]
    fn device_kv_execution_mirror_writes_fp8_projected_mla_kv_a_device_blocks_when_available(
    ) -> Result<()> {
        let config = KvCacheConfig::glm52_compressed_fp8(8);
        let descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-device-kv-a-fp8".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(2),
            token_count: 2,
        };
        assert!(!config.layer_has_dsa_indexer(descriptor.layer_id));
        let mut projected_kv_a_bf16 = Vec::new();
        let mut expected_latent_bf16 = Vec::new();
        let mut expected_rope_bf16 = Vec::new();
        for row in 0..descriptor.token_count {
            let latent = (0..GLM52_MLA_KV_LORA_RANK)
                .map(|index| ((row * 19 + index % 97) as f32 - 48.0) / 128.0)
                .collect::<Vec<_>>();
            let rope = (0..GLM52_MLA_QK_ROPE_HEAD_DIM)
                .map(|index| ((row * 13 + index % 29) as f32 - 14.0) / 64.0)
                .collect::<Vec<_>>();
            let latent_bf16 = bf16_bytes_from_f32(&latent);
            let rope_bf16 = bf16_bytes_from_f32(&rope);
            projected_kv_a_bf16.extend_from_slice(&latent_bf16);
            projected_kv_a_bf16.extend_from_slice(&rope_bf16);
            expected_latent_bf16.extend_from_slice(&latent_bf16);
            expected_rope_bf16.extend_from_slice(&rope_bf16);
        }
        let expected_payload_bytes = config.descriptor_payload_bytes(&descriptor).unwrap();
        assert_eq!(
            expected_payload_bytes,
            descriptor.token_count * GLM52_MLA_FP8_DS_BYTES_PER_TOKEN
        );
        let mut mirror = RealFullDeviceKvExecutionMirror::new(config)?;
        if !mirror.summary().uses_device_kv_cache {
            return Ok(());
        }
        let library = cuda_native_library()?;
        let projected_kv_a = upload_device_bytes(library, &projected_kv_a_bf16)?;

        let writes = mirror
            .write_projected_mla_kv_a_device_blocks_bf16(
                std::slice::from_ref(&descriptor),
                projected_kv_a.buffer,
                None,
            )?
            .expect("CUDA-enabled mirror should write FP8 projected kv_a device blocks");

        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].payload_bytes, expected_payload_bytes);
        let readback = mirror
            .read_descriptor_payloads_to_host(std::slice::from_ref(&descriptor))?
            .expect("CUDA-enabled mirror should read FP8 projected kv_a device block");
        let packed = &readback[0];
        assert_eq!(packed.len(), expected_payload_bytes);
        let packed_rope_offset = GLM52_MLA_KV_LORA_RANK + GLM52_MLA_FP8_DS_SCALE_BYTES_PER_TOKEN;
        for row in 0..descriptor.token_count {
            let packed_row = row * GLM52_MLA_FP8_DS_BYTES_PER_TOKEN;
            let projected_row = row * (GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM) * 2;
            assert!(
                packed[packed_row + GLM52_MLA_KV_LORA_RANK..packed_row + packed_rope_offset]
                    .iter()
                    .any(|byte| *byte != 0)
            );
            assert_eq!(
                &packed[packed_row + packed_rope_offset
                    ..packed_row + GLM52_MLA_FP8_DS_BYTES_PER_TOKEN],
                &projected_kv_a_bf16[projected_row + GLM52_MLA_KV_LORA_RANK * 2
                    ..projected_row + (GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM) * 2]
            );
        }

        let parts = mirror
            .read_mla_kv_payloads_to_device_buffers(std::slice::from_ref(&descriptor))?
            .expect("CUDA-enabled mirror should unpack FP8 MLA KV payload");
        let mut latent_bf16 = vec![0_u8; parts.kv_latent_bytes];
        let mut rope_bf16 = vec![0_u8; parts.k_rope_bytes];
        library.copy_d2h(&mut latent_bf16, parts.kv_latent.buffer)?;
        library.copy_d2h(&mut rope_bf16, parts.k_rope.buffer)?;
        assert_eq!(rope_bf16, expected_rope_bf16);
        let actual_latent = bf16_bytes_to_f32(&latent_bf16)?;
        let expected_latent = bf16_bytes_to_f32(&expected_latent_bf16)?;
        for (actual, expected) in actual_latent.iter().zip(expected_latent.iter()) {
            assert!((*actual - *expected).abs() <= 0.025);
        }

        let summary = mirror.summary();
        assert_eq!(summary.status, "cuda-kv-cache-live-scheduler");
        assert_eq!(summary.writes, 1);
        assert_eq!(summary.reads, 2);
        assert_eq!(summary.bytes, expected_payload_bytes * 3);
        assert!(summary.uses_device_kv_cache);
        Ok(())
    }

    #[test]
    fn device_kv_direct_attention_span_exposes_full_main_plane_and_row_offset() -> Result<()> {
        let config = KvCacheConfig::glm52_compressed_fp8(8)
            .with_mla_representation(MlaKvCacheRepresentation::NormalizedRotated);
        let descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-direct-fp8-span".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(2),
            token_count: 2,
        };
        let max_tokens = config.max_tokens;
        let expected_visible_bytes = config
            .descriptor_payload_bytes(&descriptor)
            .context("direct-span test descriptor payload bytes")?;
        let mut mirror = RealFullDeviceKvExecutionMirror::new(config)?;
        if !mirror.summary().uses_device_kv_cache {
            return Ok(());
        }

        let direct = mirror
            .direct_attention_ready_mla_kv_span(std::slice::from_ref(&descriptor))?
            .expect("FP8 attention-ready cache should expose a direct main-plane span");
        assert_eq!(direct.rows, descriptor.token_count);
        assert_eq!(direct.row_offset, descriptor.token_start.0 as usize);
        assert_eq!(direct.dtype, KvCacheDType::Fp8);
        assert_eq!(direct.row_stride_bytes, GLM52_MLA_FP8_DS_BYTES_PER_TOKEN);
        assert_eq!(
            direct.payload.bytes,
            max_tokens * GLM52_MLA_FP8_DS_BYTES_PER_TOKEN
        );

        let summary = mirror.summary();
        assert_eq!(summary.reads, 1);
        assert_eq!(summary.bytes, expected_visible_bytes);
        Ok(())
    }

    #[test]
    fn device_kv_execution_mirror_writes_nvfp4_projected_mla_kv_a_device_blocks_when_available(
    ) -> Result<()> {
        let config = KvCacheConfig::glm52_compressed_nvfp4(8);
        let descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-device-kv-a-nvfp4".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(2),
            token_count: 2,
        };
        assert!(!config.layer_has_dsa_indexer(descriptor.layer_id));
        let (projected_kv_a_bf16, expected_latent_bf16, expected_rope_bf16) =
            mxfp4_representable_projected_kv_payload_rows(descriptor.token_count);
        let expected_payload_bytes = config.descriptor_payload_bytes(&descriptor).unwrap();
        assert_eq!(
            expected_payload_bytes,
            descriptor.token_count * GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN
        );

        let mut mirror = RealFullDeviceKvExecutionMirror::new(config)?;
        if !mirror.summary().uses_device_kv_cache {
            return Ok(());
        }
        let library = cuda_native_library()?;
        let projected_kv_a = upload_device_bytes(library, &projected_kv_a_bf16)?;

        let writes = mirror
            .write_projected_mla_kv_a_device_blocks_bf16(
                std::slice::from_ref(&descriptor),
                projected_kv_a.buffer,
                None,
            )?
            .expect("CUDA-enabled mirror should write NVFP4 projected kv_a device blocks");

        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].payload_bytes, expected_payload_bytes);
        let readback = mirror
            .read_descriptor_payloads_to_host(std::slice::from_ref(&descriptor))?
            .expect("CUDA-enabled mirror should read NVFP4 projected kv_a device block");
        let packed = &readback[0];
        assert_eq!(packed.len(), expected_payload_bytes);
        let packed_scale_offset = GLM52_MLA_MXFP4_CODE_BYTES_PER_TOKEN;
        let packed_padding_offset =
            GLM52_MLA_MXFP4_CODE_BYTES_PER_TOKEN + GLM52_MLA_MXFP4_SCALE_BYTES_PER_TOKEN;
        let packed_rope_offset = packed_padding_offset + GLM52_MLA_MXFP4_PADDING_BYTES_PER_TOKEN;
        for row in 0..descriptor.token_count {
            let packed_row = row * GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN;
            let projected_row = row * (GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM) * 2;
            assert!(
                packed[packed_row..packed_row + GLM52_MLA_MXFP4_CODE_BYTES_PER_TOKEN]
                    .iter()
                    .any(|byte| *byte != 0)
            );
            assert!(
                packed[packed_row + packed_scale_offset..packed_row + packed_padding_offset]
                    .iter()
                    .all(|byte| *byte == 0x38)
            );
            assert!(
                packed[packed_row + packed_padding_offset..packed_row + packed_rope_offset]
                    .iter()
                    .all(|byte| *byte == 0)
            );
            assert_eq!(
                &packed[packed_row + packed_rope_offset
                    ..packed_row + GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN],
                &projected_kv_a_bf16[projected_row + GLM52_MLA_KV_LORA_RANK * 2
                    ..projected_row + (GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM) * 2]
            );
        }

        let parts = mirror
            .read_mla_kv_payloads_to_device_buffers(std::slice::from_ref(&descriptor))?
            .expect("CUDA-enabled mirror should unpack NVFP4 MLA KV payload");
        let mut latent_bf16 = vec![0_u8; parts.kv_latent_bytes];
        let mut rope_bf16 = vec![0_u8; parts.k_rope_bytes];
        library.copy_d2h(&mut latent_bf16, parts.kv_latent.buffer)?;
        library.copy_d2h(&mut rope_bf16, parts.k_rope.buffer)?;
        assert_eq!(latent_bf16, expected_latent_bf16);
        assert_eq!(rope_bf16, expected_rope_bf16);

        let summary = mirror.summary();
        assert_eq!(summary.status, "cuda-kv-cache-live-scheduler");
        assert_eq!(summary.writes, 1);
        assert_eq!(summary.reads, 2);
        assert_eq!(summary.bytes, expected_payload_bytes * 3);
        assert!(summary.uses_device_kv_cache);
        Ok(())
    }

    #[test]
    fn nvfp4_radix_boundary_copy_preserves_rows_past_valid_prefix_when_available() -> Result<()> {
        let config = KvCacheConfig::glm52_compressed_nvfp4(128);
        let source = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "radix-source".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(0),
            token_count: GLMRT_CUDA_GLM_DSA_PAGE_SIZE,
        };
        let destination = KvBlockDescriptor {
            reservation_id: 2,
            sequence_id: "radix-destination".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(GLMRT_CUDA_GLM_DSA_PAGE_SIZE as u64),
            token_count: GLMRT_CUDA_GLM_DSA_PAGE_SIZE,
        };
        let row_bytes = config
            .main_mla_row_bytes()
            .context("NVFP4 format must expose row geometry")?;
        let page_bytes = config
            .main_mla_page_bytes(GLMRT_CUDA_GLM_DSA_PAGE_SIZE)
            .context("NVFP4 format must expose page geometry")?;
        let source_payload = (0..page_bytes)
            .map(|index| (index.wrapping_mul(17) & 0xff) as u8)
            .collect::<Vec<_>>();
        let destination_payload = vec![0xa5_u8; page_bytes];

        let mut mirror = RealFullDeviceKvExecutionMirror::new(config)?;
        if !mirror.summary().uses_device_kv_cache {
            return Ok(());
        }
        mirror.write_host_blocks(
            &[source.clone(), destination.clone()],
            &[source_payload.clone(), destination_payload.clone()],
        )?;

        let valid_tokens = 7;
        mirror.copy_target_kv_boundary_page(0, 1, valid_tokens)?;
        let readback = mirror
            .read_descriptor_payloads_to_host(std::slice::from_ref(&destination))?
            .context("CUDA-enabled mirror should read the cloned boundary page")?
            .pop()
            .context("boundary page readback is missing")?;
        let valid_bytes = valid_tokens * row_bytes;
        assert_eq!(&readback[..valid_bytes], &source_payload[..valid_bytes]);
        assert_eq!(
            &readback[valid_bytes..],
            &destination_payload[valid_bytes..]
        );
        Ok(())
    }

    #[test]
    fn nvfp4_device_kv_roundtrip_addresses_records_past_int32_byte_range_when_available(
    ) -> Result<()> {
        let config = KvCacheConfig::glm52_compressed_nvfp4(65_536);
        let descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "nvfp4-high-address".to_owned(),
            layer_id: LayerId((config.layers - 1) as u32),
            token_start: PositionId((config.max_tokens - 1) as u64),
            token_count: 1,
        };
        let io = real_full_device_main_kv_block_io(&config, &descriptor)?;
        assert!(io.offset_bytes > i32::MAX as usize);
        let payload = (0..io.payload_bytes)
            .map(|index| (index.wrapping_mul(29).wrapping_add(7) & 0xff) as u8)
            .collect::<Vec<_>>();

        let mut mirror = RealFullDeviceKvExecutionMirror::new(config)?;
        if !mirror.summary().uses_device_kv_cache {
            return Ok(());
        }
        mirror.write_host_blocks(
            std::slice::from_ref(&descriptor),
            std::slice::from_ref(&payload),
        )?;
        let readback = mirror
            .read_descriptor_payloads_to_host(std::slice::from_ref(&descriptor))?
            .context("CUDA-enabled mirror should read the high-address NVFP4 record")?;
        assert_eq!(readback, vec![payload]);
        Ok(())
    }

    #[test]
    fn device_kv_execution_mirror_writes_projected_mla_kv_a_and_dsa_device_blocks_when_available(
    ) -> Result<()> {
        let config = KvCacheConfig::glm52_phase0(8);
        let descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-device-kv-a-dsa".to_owned(),
            layer_id: LayerId(0),
            token_start: PositionId(2),
            token_count: 2,
        };
        assert!(config.layer_has_dsa_indexer(descriptor.layer_id));
        let mut projected_kv_a_bf16 = Vec::new();
        let mut dsa_key_bf16 = Vec::new();
        let mut expected_payload = Vec::new();
        for row in 0..descriptor.token_count {
            let latent = (0..GLM52_MLA_KV_LORA_RANK)
                .map(|index| ((row * 17 + index % 31) as f32 - 15.0) / 128.0)
                .collect::<Vec<_>>();
            let rope = (0..GLM52_MLA_QK_ROPE_HEAD_DIM)
                .map(|index| ((row * 11 + index % 23) as f32 - 11.0) / 96.0)
                .collect::<Vec<_>>();
            let dsa = (0..GLM52_DSA_INDEX_HEAD_DIM)
                .map(|index| ((row * 7 + index % 19) as f32 - 9.0) / 64.0)
                .collect::<Vec<_>>();
            let mut main_row = bf16_bytes_from_f32(&latent);
            main_row.extend_from_slice(&bf16_bytes_from_f32(&rope));
            let dsa_row = bf16_bytes_from_f32(&dsa);
            projected_kv_a_bf16.extend_from_slice(&main_row);
            dsa_key_bf16.extend_from_slice(&dsa_row);
            expected_payload.extend_from_slice(&main_row);
            expected_payload.extend_from_slice(&dsa_row);
        }
        assert_eq!(
            expected_payload.len(),
            config.descriptor_payload_bytes(&descriptor).unwrap()
        );

        let mut mirror = RealFullDeviceKvExecutionMirror::new(config)?;
        if !mirror.summary().uses_device_kv_cache {
            return Ok(());
        }
        let library = cuda_native_library()?;
        let projected_kv_a = upload_device_bytes(library, &projected_kv_a_bf16)?;
        let dsa_key = upload_device_bytes(library, &dsa_key_bf16)?;

        let writes = mirror
            .write_projected_mla_kv_a_and_dsa_key_device_blocks_bf16(
                std::slice::from_ref(&descriptor),
                projected_kv_a.buffer,
                dsa_key.buffer,
                None,
            )?
            .expect("CUDA-enabled mirror should write projected kv_a plus DSA device blocks");

        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].payload_bytes, expected_payload.len());
        let direct_dsa_cache = mirror
            .cache
            .as_ref()
            .expect("CUDA-enabled mirror should own a cache")
            .dsa_index_k_cache_b12x_for_layer(descriptor.layer_id)?
            .expect("DSA layer should own a direct B12X index-K cache");
        let mut direct_dsa_page = vec![0_u8; GLMRT_CUDA_GLM_DSA_PACKED_PAGE_BYTES];
        library.copy_d2h(&mut direct_dsa_page, direct_dsa_cache)?;
        for slot in 2..4 {
            let quant_start = slot * GLM52_DSA_INDEX_HEAD_DIM;
            assert!(
                direct_dsa_page[quant_start..quant_start + GLM52_DSA_INDEX_HEAD_DIM]
                    .iter()
                    .any(|byte| *byte != 0)
            );
            let scale_start = GLMRT_CUDA_GLM_DSA_PAGE_SIZE * GLM52_DSA_INDEX_HEAD_DIM
                + slot * std::mem::size_of::<f32>();
            let scale = f32::from_ne_bytes(
                direct_dsa_page[scale_start..scale_start + std::mem::size_of::<f32>()]
                    .try_into()
                    .unwrap(),
            );
            assert!(scale.is_finite() && scale > 0.0);
        }
        let readback = mirror
            .read_descriptor_payloads_to_host(std::slice::from_ref(&descriptor))?
            .expect("CUDA-enabled mirror should read projected kv_a plus DSA device block");
        assert_eq!(readback, vec![expected_payload.clone()]);

        let summary = mirror.summary();
        assert_eq!(summary.status, "cuda-kv-cache-live-scheduler");
        assert_eq!(summary.writes, 1);
        assert_eq!(summary.reads, 1);
        assert_eq!(summary.bytes, expected_payload.len() * 2);
        assert!(summary.uses_device_kv_cache);
        Ok(())
    }

    #[test]
    fn device_kv_execution_mirror_writes_dsa_blocks_from_graph_linear_device_output() -> Result<()>
    {
        let rows = 9_usize;
        let config = KvCacheConfig::glm52_phase0(32);
        let descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-device-kv-a-dsa-graph-output".to_owned(),
            layer_id: LayerId(1),
            token_start: PositionId(2),
            token_count: rows,
        };
        assert!(config.layer_has_dsa_indexer(descriptor.layer_id));
        let main_values_per_row = GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM;
        let input_bytes = rows * GLM52_HIDDEN_SIZE * std::mem::size_of::<u16>();
        let weight_bytes = main_values_per_row * GLM52_HIDDEN_SIZE * std::mem::size_of::<u16>();
        let main_bytes = rows * main_values_per_row * std::mem::size_of::<u16>();
        let dsa_stride_bytes = GLM52_DSA_INDEX_HEAD_DIM * std::mem::size_of::<u16>();
        let mut dsa_key_bf16 = Vec::new();
        let mut expected_payload = Vec::new();
        for row in 0..rows {
            let dsa = (0..GLM52_DSA_INDEX_HEAD_DIM)
                .map(|index| ((row * 7 + index % 19) as f32 - 9.0) / 64.0)
                .collect::<Vec<_>>();
            let dsa_row = bf16_bytes_from_f32(&dsa);
            expected_payload.extend(std::iter::repeat(0_u8).take(main_values_per_row * 2));
            expected_payload.extend_from_slice(&dsa_row);
            dsa_key_bf16.extend_from_slice(&dsa_row);
        }
        assert_eq!(main_bytes + rows * dsa_stride_bytes, expected_payload.len());

        let mut mirror = RealFullDeviceKvExecutionMirror::new(config)?;
        if !mirror.summary().uses_device_kv_cache {
            return Ok(());
        }
        let library = cuda_native_library()?;
        let input = upload_device_bytes(library, &vec![0_u8; input_bytes])?;
        let weight = upload_device_bytes(library, &vec![0_u8; weight_bytes])?;
        let projected = DeviceBufferGuard::new(library, main_bytes)?;
        linear_rows_bf16_device_buffers_for_layer(
            descriptor.layer_id.0 as usize,
            input.buffer,
            weight.buffer,
            projected.buffer,
            rows,
            GLM52_HIDDEN_SIZE,
            main_values_per_row,
        )?;
        let dsa_key = upload_device_bytes(library, &dsa_key_bf16)?;

        let writes = mirror
            .write_projected_mla_kv_a_and_dsa_key_device_blocks_bf16(
                std::slice::from_ref(&descriptor),
                projected.buffer,
                dsa_key.buffer,
                None,
            )?
            .expect(
                "CUDA-enabled mirror should write graph-produced projected kv_a plus DSA blocks",
            );

        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].payload_bytes, expected_payload.len());
        let readback = mirror
            .read_descriptor_payloads_to_host(std::slice::from_ref(&descriptor))?
            .expect("CUDA-enabled mirror should read graph-produced projected kv_a plus DSA block");
        assert_eq!(readback, vec![expected_payload]);
        Ok(())
    }

    #[test]
    fn device_kv_execution_mirror_writes_fp8_projected_mla_kv_a_and_dsa_device_blocks_when_available(
    ) -> Result<()> {
        let config = KvCacheConfig::glm52_compressed_fp8(8);
        let descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-device-kv-a-dsa-fp8".to_owned(),
            layer_id: LayerId(0),
            token_start: PositionId(2),
            token_count: 2,
        };
        assert!(config.layer_has_dsa_indexer(descriptor.layer_id));
        let (projected_kv_a_bf16, dsa_key_bf16, _bf16_payload) =
            dsa_projected_kv_payload_rows(descriptor.token_count);
        let main_stride_bytes =
            (GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM) * std::mem::size_of::<u16>();
        let latent_stride_bytes = GLM52_MLA_KV_LORA_RANK * std::mem::size_of::<u16>();
        let rope_stride_bytes = GLM52_MLA_QK_ROPE_HEAD_DIM * std::mem::size_of::<u16>();
        let dsa_stride_bytes = GLM52_DSA_INDEX_HEAD_DIM * std::mem::size_of::<u16>();
        let mut expected_latent_bf16 = Vec::new();
        let mut expected_rope_bf16 = Vec::new();
        for row in 0..descriptor.token_count {
            let main_row = row * main_stride_bytes;
            expected_latent_bf16
                .extend_from_slice(&projected_kv_a_bf16[main_row..main_row + latent_stride_bytes]);
            expected_rope_bf16.extend_from_slice(
                &projected_kv_a_bf16[main_row + latent_stride_bytes
                    ..main_row + latent_stride_bytes + rope_stride_bytes],
            );
        }
        let expected_payload_bytes = config.descriptor_payload_bytes(&descriptor).unwrap();
        assert_eq!(
            expected_payload_bytes,
            descriptor.token_count * (GLM52_MLA_FP8_DS_BYTES_PER_TOKEN + dsa_stride_bytes)
        );

        let mut mirror = RealFullDeviceKvExecutionMirror::new(config)?;
        if !mirror.summary().uses_device_kv_cache {
            return Ok(());
        }
        let library = cuda_native_library()?;
        let projected_kv_a = upload_device_bytes(library, &projected_kv_a_bf16)?;
        let dsa_key = upload_device_bytes(library, &dsa_key_bf16)?;

        let writes = mirror
            .write_projected_mla_kv_a_and_dsa_key_device_blocks_bf16(
                std::slice::from_ref(&descriptor),
                projected_kv_a.buffer,
                dsa_key.buffer,
                None,
            )?
            .expect("CUDA-enabled mirror should write FP8 projected kv_a plus DSA device blocks");

        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].payload_bytes, expected_payload_bytes);
        let readback = mirror
            .read_descriptor_payloads_to_host(std::slice::from_ref(&descriptor))?
            .expect("CUDA-enabled mirror should read FP8 projected kv_a plus DSA device block");
        let packed = &readback[0];
        assert_eq!(packed.len(), expected_payload_bytes);
        let packed_rope_offset = GLM52_MLA_KV_LORA_RANK + GLM52_MLA_FP8_DS_SCALE_BYTES_PER_TOKEN;
        for row in 0..descriptor.token_count {
            let packed_row = row * (GLM52_MLA_FP8_DS_BYTES_PER_TOKEN + dsa_stride_bytes);
            let projected_row = row * main_stride_bytes;
            let dsa_row = row * dsa_stride_bytes;
            assert!(
                packed[packed_row + GLM52_MLA_KV_LORA_RANK..packed_row + packed_rope_offset]
                    .iter()
                    .any(|byte| *byte != 0)
            );
            assert_eq!(
                &packed[packed_row + packed_rope_offset
                    ..packed_row + GLM52_MLA_FP8_DS_BYTES_PER_TOKEN],
                &projected_kv_a_bf16
                    [projected_row + latent_stride_bytes..projected_row + main_stride_bytes]
            );
            assert_eq!(
                &packed[packed_row + GLM52_MLA_FP8_DS_BYTES_PER_TOKEN
                    ..packed_row + GLM52_MLA_FP8_DS_BYTES_PER_TOKEN + dsa_stride_bytes],
                &dsa_key_bf16[dsa_row..dsa_row + dsa_stride_bytes]
            );
        }

        let parts = mirror
            .read_mla_kv_payloads_to_device_buffers(std::slice::from_ref(&descriptor))?
            .expect("CUDA-enabled mirror should unpack FP8 MLA+DSA KV payload");
        let mut latent_bf16 = vec![0_u8; parts.kv_latent_bytes];
        let mut rope_bf16 = vec![0_u8; parts.k_rope_bytes];
        let mut dsa_bf16 = vec![0_u8; parts.dsa_key_bytes];
        library.copy_d2h(&mut latent_bf16, parts.kv_latent.buffer)?;
        library.copy_d2h(&mut rope_bf16, parts.k_rope.buffer)?;
        library.copy_d2h(
            &mut dsa_bf16,
            parts
                .dsa_key
                .as_ref()
                .expect("FP8 DSA unpack should return DSA output")
                .buffer,
        )?;
        assert_eq!(rope_bf16, expected_rope_bf16);
        assert_eq!(dsa_bf16, dsa_key_bf16);
        let actual_latent = bf16_bytes_to_f32(&latent_bf16)?;
        let expected_latent = bf16_bytes_to_f32(&expected_latent_bf16)?;
        for (actual, expected) in actual_latent.iter().zip(expected_latent.iter()) {
            assert!((*actual - *expected).abs() <= 0.025);
        }

        let summary = mirror.summary();
        assert_eq!(summary.status, "cuda-kv-cache-live-scheduler");
        assert_eq!(summary.writes, 1);
        assert_eq!(summary.reads, 2);
        assert_eq!(summary.bytes, expected_payload_bytes * 3);
        assert!(summary.uses_device_kv_cache);
        Ok(())
    }

    #[test]
    fn device_kv_execution_mirror_writes_nvfp4_projected_mla_kv_a_and_dsa_device_blocks_when_available(
    ) -> Result<()> {
        let config = KvCacheConfig::glm52_compressed_nvfp4(8);
        let descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-device-kv-a-dsa-nvfp4".to_owned(),
            layer_id: LayerId(0),
            token_start: PositionId(2),
            token_count: 2,
        };
        assert!(config.layer_has_dsa_indexer(descriptor.layer_id));
        let (projected_kv_a_bf16, expected_latent_bf16, expected_rope_bf16) =
            mxfp4_representable_projected_kv_payload_rows(descriptor.token_count);
        let dsa_stride_bytes = GLM52_DSA_INDEX_HEAD_DIM * std::mem::size_of::<u16>();
        let mut dsa_key_bf16 = Vec::new();
        for row in 0..descriptor.token_count {
            let dsa = (0..GLM52_DSA_INDEX_HEAD_DIM)
                .map(|index| ((row * 7 + index % 19) as f32 - 9.0) / 64.0)
                .collect::<Vec<_>>();
            dsa_key_bf16.extend_from_slice(&bf16_bytes_from_f32(&dsa));
        }
        let expected_payload_bytes = config.descriptor_payload_bytes(&descriptor).unwrap();
        assert_eq!(
            expected_payload_bytes,
            descriptor.token_count * (GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN + dsa_stride_bytes)
        );

        let mut mirror = RealFullDeviceKvExecutionMirror::new(config)?;
        if !mirror.summary().uses_device_kv_cache {
            return Ok(());
        }
        let library = cuda_native_library()?;
        let projected_kv_a = upload_device_bytes(library, &projected_kv_a_bf16)?;
        let dsa_key = upload_device_bytes(library, &dsa_key_bf16)?;

        let writes = mirror
            .write_projected_mla_kv_a_and_dsa_key_device_blocks_bf16(
                std::slice::from_ref(&descriptor),
                projected_kv_a.buffer,
                dsa_key.buffer,
                None,
            )?
            .expect("CUDA-enabled mirror should write NVFP4 projected kv_a plus DSA device blocks");

        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].payload_bytes, expected_payload_bytes);
        let readback = mirror
            .read_descriptor_payloads_to_host(std::slice::from_ref(&descriptor))?
            .expect("CUDA-enabled mirror should read NVFP4 projected kv_a plus DSA device block");
        let packed = &readback[0];
        assert_eq!(packed.len(), expected_payload_bytes);
        let packed_scale_offset = GLM52_MLA_MXFP4_CODE_BYTES_PER_TOKEN;
        let packed_padding_offset =
            GLM52_MLA_MXFP4_CODE_BYTES_PER_TOKEN + GLM52_MLA_MXFP4_SCALE_BYTES_PER_TOKEN;
        let packed_rope_offset = packed_padding_offset + GLM52_MLA_MXFP4_PADDING_BYTES_PER_TOKEN;
        for row in 0..descriptor.token_count {
            let packed_row = row * (GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN + dsa_stride_bytes);
            let projected_row = row * (GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM) * 2;
            let dsa_row = row * dsa_stride_bytes;
            assert!(
                packed[packed_row..packed_row + GLM52_MLA_MXFP4_CODE_BYTES_PER_TOKEN]
                    .iter()
                    .any(|byte| *byte != 0)
            );
            assert!(
                packed[packed_row + packed_scale_offset..packed_row + packed_padding_offset]
                    .iter()
                    .all(|byte| *byte == 0x38)
            );
            assert!(
                packed[packed_row + packed_padding_offset..packed_row + packed_rope_offset]
                    .iter()
                    .all(|byte| *byte == 0)
            );
            assert_eq!(
                &packed[packed_row + packed_rope_offset
                    ..packed_row + GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN],
                &projected_kv_a_bf16[projected_row + GLM52_MLA_KV_LORA_RANK * 2
                    ..projected_row + (GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM) * 2]
            );
            assert_eq!(
                &packed[packed_row + GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN
                    ..packed_row + GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN + dsa_stride_bytes],
                &dsa_key_bf16[dsa_row..dsa_row + dsa_stride_bytes]
            );
        }

        let parts = mirror
            .read_mla_kv_payloads_to_device_buffers(std::slice::from_ref(&descriptor))?
            .expect("CUDA-enabled mirror should unpack NVFP4 MLA+DSA KV payload");
        let mut latent_bf16 = vec![0_u8; parts.kv_latent_bytes];
        let mut rope_bf16 = vec![0_u8; parts.k_rope_bytes];
        let mut dsa_bf16 = vec![0_u8; parts.dsa_key_bytes];
        library.copy_d2h(&mut latent_bf16, parts.kv_latent.buffer)?;
        library.copy_d2h(&mut rope_bf16, parts.k_rope.buffer)?;
        library.copy_d2h(
            &mut dsa_bf16,
            parts
                .dsa_key
                .as_ref()
                .expect("NVFP4 DSA unpack should return DSA output")
                .buffer,
        )?;
        assert_eq!(latent_bf16, expected_latent_bf16);
        assert_eq!(rope_bf16, expected_rope_bf16);
        assert_eq!(dsa_bf16, dsa_key_bf16);

        let summary = mirror.summary();
        assert_eq!(summary.status, "cuda-kv-cache-live-scheduler");
        assert_eq!(summary.writes, 1);
        assert_eq!(summary.reads, 2);
        assert_eq!(summary.bytes, expected_payload_bytes * 3);
        assert!(summary.uses_device_kv_cache);
        Ok(())
    }

    #[test]
    fn device_kv_execution_mirror_direct_dsa_writes_preserve_overlapping_rows() -> Result<()> {
        let config = KvCacheConfig::glm52_phase0(8);
        let first_descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-device-kv-a-dsa-reuse".to_owned(),
            layer_id: LayerId(0),
            token_start: PositionId(2),
            token_count: 2,
        };
        let second_descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-device-kv-a-dsa-reuse".to_owned(),
            layer_id: LayerId(0),
            token_start: PositionId(3),
            token_count: 1,
        };
        assert!(config.layer_has_dsa_indexer(first_descriptor.layer_id));
        assert!(config.layer_has_dsa_indexer(second_descriptor.layer_id));

        let mut mirror = RealFullDeviceKvExecutionMirror::new(config.clone())?;
        if !mirror.summary().uses_device_kv_cache {
            return Ok(());
        }
        let library = cuda_native_library()?;

        let (first_projected, first_dsa, first_expected) =
            dsa_projected_kv_payload_rows(first_descriptor.token_count);
        let projected_kv_a = upload_device_bytes(library, &first_projected)?;
        let dsa_key = upload_device_bytes(library, &first_dsa)?;
        mirror
            .write_projected_mla_kv_a_and_dsa_key_device_blocks_bf16(
                std::slice::from_ref(&first_descriptor),
                projected_kv_a.buffer,
                dsa_key.buffer,
                None,
            )?
            .expect("CUDA-enabled mirror should write first DSA payload");
        let (second_projected, second_dsa, second_expected) =
            dsa_projected_kv_payload_rows(second_descriptor.token_count);
        let projected_kv_a = upload_device_bytes(library, &second_projected)?;
        let dsa_key = upload_device_bytes(library, &second_dsa)?;
        mirror
            .write_projected_mla_kv_a_and_dsa_key_device_blocks_bf16(
                std::slice::from_ref(&second_descriptor),
                projected_kv_a.buffer,
                dsa_key.buffer,
                None,
            )?
            .expect("CUDA-enabled mirror should write second DSA payload");

        let first_readback = mirror
            .read_descriptor_payloads_to_host(std::slice::from_ref(&first_descriptor))?
            .expect("CUDA-enabled mirror should read first DSA payload");
        let mut expected_first_after_second = first_expected.clone();
        let second_payload_bytes = config.descriptor_payload_bytes(&second_descriptor).unwrap();
        expected_first_after_second[second_payload_bytes..second_payload_bytes * 2]
            .copy_from_slice(&second_expected);
        assert_eq!(first_readback, vec![expected_first_after_second]);

        let second_readback = mirror
            .read_descriptor_payloads_to_host(std::slice::from_ref(&second_descriptor))?
            .expect("CUDA-enabled mirror should read second DSA payload");
        assert_eq!(second_readback, vec![second_expected]);
        Ok(())
    }

    #[test]
    fn device_kv_execution_mirror_unpacks_dsa_mla_kv_payloads_when_available() -> Result<()> {
        let config = KvCacheConfig::glm52_phase0(4);
        let descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-dsa".to_owned(),
            layer_id: LayerId(0),
            token_start: PositionId(1),
            token_count: 2,
        };
        assert!(config.layer_has_dsa_indexer(descriptor.layer_id));
        let (payload, expected_latent, expected_rope, expected_dsa) = mla_kv_payload_rows(2, true);
        assert_eq!(
            payload.len(),
            config.descriptor_payload_bytes(&descriptor).unwrap()
        );

        let mut mirror = RealFullDeviceKvExecutionMirror::new(config)?;
        if !mirror.summary().uses_device_kv_cache {
            return Ok(());
        }

        mirror.write_host_blocks(
            std::slice::from_ref(&descriptor),
            std::slice::from_ref(&payload),
        )?;
        let unpack = mirror
            .read_mla_kv_payloads_to_device_parts(std::slice::from_ref(&descriptor))?
            .expect("CUDA-enabled mirror should unpack descriptor payloads");

        assert_eq!(unpack.status, "cuda-kv-cache-mla-kv-unpack-readback");
        assert_eq!(unpack.rows, 2);
        assert_eq!(unpack.payload_bytes, payload.len());
        assert_eq!(unpack.kv_latent_bf16, expected_latent);
        assert_eq!(unpack.k_rope_bf16, expected_rope);
        assert_eq!(unpack.dsa_key_bf16, expected_dsa);

        let summary = mirror.summary();
        assert_eq!(summary.status, "cuda-kv-cache-live-scheduler");
        assert_eq!(summary.writes, 1);
        assert_eq!(summary.reads, 1);
        assert_eq!(summary.bytes, payload.len() * 2);
        assert!(summary.uses_device_kv_cache);
        Ok(())
    }

    #[test]
    fn device_kv_execution_mirror_unpacks_main_mla_kv_payloads_when_available() -> Result<()> {
        let config = KvCacheConfig::glm52_phase0(4);
        let descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-main".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(0),
            token_count: 1,
        };
        assert!(!config.layer_has_dsa_indexer(descriptor.layer_id));
        let (payload, expected_latent, expected_rope, _) = mla_kv_payload_rows(1, false);
        assert_eq!(
            payload.len(),
            config.descriptor_payload_bytes(&descriptor).unwrap()
        );
        let expected_payload_stride = config.layer_bytes_per_token(descriptor.layer_id);

        let mut mirror = RealFullDeviceKvExecutionMirror::new(config)?;
        if !mirror.summary().uses_device_kv_cache {
            return Ok(());
        }

        mirror.write_host_blocks(
            std::slice::from_ref(&descriptor),
            std::slice::from_ref(&payload),
        )?;
        let parts = mirror
            .read_mla_kv_payloads_to_device_buffers(std::slice::from_ref(&descriptor))?
            .expect("CUDA-enabled mirror should return unpacked device buffers");

        assert_eq!(parts.status(), "cuda-kv-cache-mla-kv-unpack-device-buffers");
        assert_eq!(parts.layer_id(), LayerId(3));
        assert_eq!(parts.rows(), 1);
        assert_eq!(parts.payload_bytes(), payload.len());
        assert_eq!(parts.payload_stride_bytes(), expected_payload_stride);
        assert_eq!(
            parts.kv_latent_buffer().bytes,
            GLM52_MLA_KV_LORA_RANK * std::mem::size_of::<u16>()
        );
        assert_eq!(
            parts.k_rope_buffer().bytes,
            GLM52_MLA_QK_ROPE_HEAD_DIM * std::mem::size_of::<u16>()
        );
        assert!(parts.dsa_key_buffer().is_none());
        let unpack = parts.copy_to_host()?;

        assert_eq!(unpack.status, "cuda-kv-cache-mla-kv-unpack-readback");
        assert_eq!(unpack.rows, 1);
        assert_eq!(unpack.payload_bytes, payload.len());
        assert_eq!(unpack.kv_latent_bf16, expected_latent);
        assert_eq!(unpack.k_rope_bf16, expected_rope);
        assert!(unpack.dsa_key_bf16.is_none());

        let summary = mirror.summary();
        assert_eq!(summary.status, "cuda-kv-cache-live-scheduler");
        assert_eq!(summary.writes, 1);
        assert_eq!(summary.reads, 1);
        assert_eq!(summary.bytes, payload.len() * 2);
        assert!(summary.uses_device_kv_cache);
        Ok(())
    }

    #[test]
    fn device_kv_execution_mirror_reuses_mla_read_payload_buffer_when_available() -> Result<()> {
        let config = KvCacheConfig::glm52_phase0(4);
        let first_descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-main-read-reuse".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(0),
            token_count: 2,
        };
        let second_descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-main-read-reuse".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(1),
            token_count: 1,
        };
        assert!(!config.layer_has_dsa_indexer(first_descriptor.layer_id));
        let (payload, expected_latent, expected_rope, _) = mla_kv_payload_rows(2, false);
        assert_eq!(
            payload.len(),
            config.descriptor_payload_bytes(&first_descriptor).unwrap()
        );

        let mut mirror = RealFullDeviceKvExecutionMirror::new(config)?;
        if !mirror.summary().uses_device_kv_cache {
            return Ok(());
        }

        mirror.write_host_blocks(
            std::slice::from_ref(&first_descriptor),
            std::slice::from_ref(&payload),
        )?;
        let first_parts = mirror
            .read_mla_kv_payloads_to_device_buffers(std::slice::from_ref(&first_descriptor))?
            .expect("CUDA-enabled mirror should return first unpacked device buffers");
        assert_eq!(first_parts.rows(), 2);
        let cache = mirror.cache.as_ref().expect("CUDA mirror should own cache");
        let first_payload_ptr = cache.mla_read_payload.buffer.ptr;
        let first_payload_capacity = cache.mla_read_payload.capacity;
        assert!(!first_payload_ptr.is_null());
        let first_readback = first_parts.copy_to_host()?;
        assert_eq!(first_readback.kv_latent_bf16, expected_latent);
        assert_eq!(first_readback.k_rope_bf16, expected_rope);

        let second_parts = mirror
            .read_mla_kv_payloads_to_device_buffers(std::slice::from_ref(&second_descriptor))?
            .expect("CUDA-enabled mirror should return second unpacked device buffers");
        assert_eq!(second_parts.rows(), 1);
        let cache = mirror.cache.as_ref().expect("CUDA mirror should own cache");
        assert_eq!(cache.mla_read_payload.buffer.ptr, first_payload_ptr);
        assert_eq!(cache.mla_read_payload.capacity, first_payload_capacity);

        let latent_row_bytes = GLM52_MLA_KV_LORA_RANK * std::mem::size_of::<u16>();
        let rope_row_bytes = GLM52_MLA_QK_ROPE_HEAD_DIM * std::mem::size_of::<u16>();
        let second_readback = second_parts.copy_to_host()?;
        assert_eq!(
            second_readback.kv_latent_bf16,
            expected_latent[latent_row_bytes..latent_row_bytes * 2]
        );
        assert_eq!(
            second_readback.k_rope_bf16,
            expected_rope[rope_row_bytes..rope_row_bytes * 2]
        );
        assert!(second_readback.dsa_key_bf16.is_none());
        Ok(())
    }

    #[test]
    fn device_kv_execution_mirror_rotates_unpacked_k_rope_on_device() -> Result<()> {
        let config = KvCacheConfig::glm52_phase0(8);
        let descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-rope".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(4),
            token_count: 2,
        };
        let mut payload = Vec::new();
        let mut expected_k_rope_input = Vec::new();
        for row in 0..descriptor.token_count {
            let latent = (0..GLM52_MLA_KV_LORA_RANK)
                .map(|index| ((row * 13 + index % 19) as f32 - 9.0) / 64.0)
                .collect::<Vec<_>>();
            let rope = (0..GLM52_MLA_QK_ROPE_HEAD_DIM)
                .map(|index| ((row * 7 + index % 17) as f32 - 8.0) / 32.0)
                .collect::<Vec<_>>();
            payload.extend_from_slice(&bf16_bytes_from_f32(&latent));
            let rope_bf16 = bf16_bytes_from_f32(&rope);
            expected_k_rope_input.extend_from_slice(&rope_bf16);
            payload.extend_from_slice(&rope_bf16);
        }
        assert_eq!(
            payload.len(),
            config.descriptor_payload_bytes(&descriptor).unwrap()
        );

        let mut mirror = RealFullDeviceKvExecutionMirror::new(config)?;
        if !mirror.summary().uses_device_kv_cache {
            return Ok(());
        }

        mirror.write_host_blocks(
            std::slice::from_ref(&descriptor),
            std::slice::from_ref(&payload),
        )?;
        let parts = mirror
            .read_mla_kv_payloads_to_device_buffers(std::slice::from_ref(&descriptor))?
            .expect("CUDA-enabled mirror should return unpacked device buffers");
        let positions = [4_u32, 5_u32];
        let theta = 10_000.0_f32;
        let rotated = parts.rotate_k_rope_bf16(&positions, theta)?;

        assert_eq!(
            rotated.status(),
            "cuda-kv-cache-mla-kv-k-rope-device-buffer"
        );
        assert_eq!(rotated.rows(), descriptor.token_count);
        assert_eq!(rotated.rotary_dim(), GLM52_MLA_QK_ROPE_HEAD_DIM);
        assert_eq!(
            rotated.k_rope_rotated_buffer().bytes,
            descriptor.token_count * GLM52_MLA_QK_ROPE_HEAD_DIM * std::mem::size_of::<u16>()
        );
        let readback = rotated.copy_to_host()?;
        assert_eq!(readback.status, "cuda-kv-cache-mla-kv-k-rope-readback");
        assert_eq!(readback.rows, descriptor.token_count);
        assert_eq!(readback.rotary_dim, GLM52_MLA_QK_ROPE_HEAD_DIM);
        assert_eq!(
            readback.k_rope_rotated_bf16,
            expected_rope_bf16(
                &expected_k_rope_input,
                &positions,
                GLM52_MLA_QK_ROPE_HEAD_DIM,
                theta
            )
        );

        let summary = mirror.summary();
        assert_eq!(summary.status, "cuda-kv-cache-live-scheduler");
        assert_eq!(summary.writes, 1);
        assert_eq!(summary.reads, 1);
        assert_eq!(summary.bytes, payload.len() * 2);
        assert!(summary.uses_device_kv_cache);
        Ok(())
    }

    #[test]
    fn device_kv_execution_mirror_projects_and_splits_unpacked_mla_kv_on_device() -> Result<()> {
        let config = KvCacheConfig::glm52_phase0(4);
        let descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-project".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(0),
            token_count: 1,
        };
        let latent_values = (0..GLM52_MLA_KV_LORA_RANK)
            .map(|index| ((index % 29) as f32 - 14.0) / 32.0)
            .collect::<Vec<_>>();
        let rope_values = (0..GLM52_MLA_QK_ROPE_HEAD_DIM)
            .map(|index| ((index % 11) as f32 - 5.0) / 16.0)
            .collect::<Vec<_>>();
        let mut payload = bf16_bytes_from_f32(&latent_values);
        payload.extend_from_slice(&bf16_bytes_from_f32(&rope_values));
        assert_eq!(
            payload.len(),
            config.descriptor_payload_bytes(&descriptor).unwrap()
        );

        let mut mirror = RealFullDeviceKvExecutionMirror::new(config)?;
        if !mirror.summary().uses_device_kv_cache {
            return Ok(());
        }

        mirror.write_host_blocks(
            std::slice::from_ref(&descriptor),
            std::slice::from_ref(&payload),
        )?;
        let parts = mirror
            .read_mla_kv_payloads_to_device_buffers(std::slice::from_ref(&descriptor))?
            .expect("CUDA-enabled mirror should return unpacked device buffers");
        let library = parts.kv_latent.library;
        let heads = 2;
        let nope_dim = 3;
        let v_dim = 2;
        let projected_width = heads * (nope_dim + v_dim);
        let norm_weight_bf16 = bf16_bytes_from_f32(&vec![1.0_f32; GLM52_MLA_KV_LORA_RANK]);
        let mut kv_b_weight_values = vec![0.0_f32; projected_width * GLM52_MLA_KV_LORA_RANK];
        let selected_columns = (0..projected_width)
            .map(|out_col| (out_col * 17 + 3) % GLM52_MLA_KV_LORA_RANK)
            .collect::<Vec<_>>();
        for (out_col, selected_col) in selected_columns.iter().copied().enumerate() {
            kv_b_weight_values[out_col * GLM52_MLA_KV_LORA_RANK + selected_col] =
                if out_col % 2 == 0 { 1.0 } else { -1.0 };
        }
        let kv_b_weight_bf16 = bf16_bytes_from_f32(&kv_b_weight_values);
        let norm_weight = upload_device_bytes(library, &norm_weight_bf16)?;
        let kv_b_weight = upload_device_bytes(library, &kv_b_weight_bf16)?;

        let projected = parts.project_kv_latent_and_split_bf16(
            norm_weight.buffer,
            kv_b_weight.buffer,
            heads,
            nope_dim,
            v_dim,
            1.0e-5,
        )?;

        assert_eq!(
            projected.status(),
            "cuda-kv-cache-mla-kv-norm-linear-split-device-buffers"
        );
        assert_eq!(projected.rows(), 1);
        assert_eq!(projected.heads(), heads);
        assert_eq!(projected.nope_dim(), nope_dim);
        assert_eq!(projected.v_dim(), v_dim);
        assert_eq!(
            projected.normalized_buffer().bytes,
            GLM52_MLA_KV_LORA_RANK * std::mem::size_of::<u16>()
        );
        assert_eq!(
            projected.projected_buffer().bytes,
            projected_width * std::mem::size_of::<u16>()
        );
        assert_eq!(
            projected.k_nope_buffer().bytes,
            heads * nope_dim * std::mem::size_of::<u16>()
        );
        assert_eq!(
            projected.values_buffer().bytes,
            heads * v_dim * std::mem::size_of::<u16>()
        );

        let readback = projected.copy_to_host()?;
        assert_eq!(
            readback.status,
            "cuda-kv-cache-mla-kv-norm-linear-split-readback"
        );
        assert_eq!(readback.rows, 1);
        assert_eq!(readback.heads, heads);
        assert_eq!(readback.nope_dim, nope_dim);
        assert_eq!(readback.v_dim, v_dim);
        let normalized = bf16_bytes_to_f32(&readback.normalized_bf16)?;
        let expected_projected_values = selected_columns
            .iter()
            .copied()
            .enumerate()
            .map(|(out_col, selected_col)| {
                let sign = if out_col % 2 == 0 { 1.0 } else { -1.0 };
                normalized[selected_col] * sign
            })
            .collect::<Vec<_>>();
        let expected_projected_bf16 = bf16_bytes_from_f32(&expected_projected_values);
        assert_eq!(readback.projected_bf16, expected_projected_bf16);

        let mut expected_k_nope = Vec::new();
        let mut expected_values = Vec::new();
        let value_size = std::mem::size_of::<u16>();
        for head in 0..heads {
            let head_start = head * (nope_dim + v_dim) * value_size;
            let nope_end = head_start + nope_dim * value_size;
            let value_end = nope_end + v_dim * value_size;
            expected_k_nope.extend_from_slice(&readback.projected_bf16[head_start..nope_end]);
            expected_values.extend_from_slice(&readback.projected_bf16[nope_end..value_end]);
        }
        assert_eq!(readback.k_nope_bf16, expected_k_nope);
        assert_eq!(readback.values_bf16, expected_values);

        let summary = mirror.summary();
        assert_eq!(summary.status, "cuda-kv-cache-live-scheduler");
        assert_eq!(summary.writes, 1);
        assert_eq!(summary.reads, 1);
        assert_eq!(summary.bytes, payload.len() * 2);
        assert!(summary.uses_device_kv_cache);
        Ok(())
    }

    #[test]
    fn device_kv_execution_mirror_runs_mla_attention_from_device_kv_parts() -> Result<()> {
        let config = KvCacheConfig::glm52_phase0(8);
        let descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-attention".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(0),
            token_count: 2,
        };
        let mut payload = Vec::new();
        for row in 0..descriptor.token_count {
            let latent = (0..GLM52_MLA_KV_LORA_RANK)
                .map(|index| ((row * 11 + index % 31) as f32 - 15.0) / 96.0)
                .collect::<Vec<_>>();
            let rope = (0..GLM52_MLA_QK_ROPE_HEAD_DIM)
                .map(|index| ((row * 5 + index % 23) as f32 - 11.0) / 128.0)
                .collect::<Vec<_>>();
            payload.extend_from_slice(&bf16_bytes_from_f32(&latent));
            payload.extend_from_slice(&bf16_bytes_from_f32(&rope));
        }
        assert_eq!(
            payload.len(),
            config.descriptor_payload_bytes(&descriptor).unwrap()
        );

        let mut mirror = RealFullDeviceKvExecutionMirror::new(config)?;
        if !mirror.summary().uses_device_kv_cache {
            return Ok(());
        }

        mirror.write_host_blocks(
            std::slice::from_ref(&descriptor),
            std::slice::from_ref(&payload),
        )?;
        let parts = mirror
            .read_mla_kv_payloads_to_device_buffers(std::slice::from_ref(&descriptor))?
            .expect("CUDA-enabled mirror should return unpacked device buffers");
        let library = parts.kv_latent.library;
        let heads = 2;
        let nope_dim = 3;
        let v_dim = 2;
        let projected_width = heads * (nope_dim + v_dim);
        let norm_weight_bf16 = bf16_bytes_from_f32(&vec![1.0_f32; GLM52_MLA_KV_LORA_RANK]);
        let mut kv_b_weight_values = vec![0.0_f32; projected_width * GLM52_MLA_KV_LORA_RANK];
        for out_col in 0..projected_width {
            let selected_col = (out_col * 19 + 5) % GLM52_MLA_KV_LORA_RANK;
            kv_b_weight_values[out_col * GLM52_MLA_KV_LORA_RANK + selected_col] =
                if out_col % 2 == 0 { 0.5 } else { -0.75 };
        }
        let norm_weight = upload_device_bytes(library, &norm_weight_bf16)?;
        let kv_b_weight = upload_device_bytes(library, &bf16_bytes_from_f32(&kv_b_weight_values))?;
        let projected = parts.project_kv_latent_and_split_bf16(
            norm_weight.buffer,
            kv_b_weight.buffer,
            heads,
            nope_dim,
            v_dim,
            1.0e-5,
        )?;
        let positions = [0_u32, 1_u32];
        let rotated = parts.rotate_k_rope_bf16(&positions, 10_000.0)?;

        let q_nope_values = (0..descriptor.token_count * heads * nope_dim)
            .map(|index| ((index % 13) as f32 - 6.0) / 64.0)
            .collect::<Vec<_>>();
        let q_rope_values = (0..descriptor.token_count * heads * GLM52_MLA_QK_ROPE_HEAD_DIM)
            .map(|index| ((index % 17) as f32 - 8.0) / 256.0)
            .collect::<Vec<_>>();
        let q_nope_bf16 = bf16_bytes_from_f32(&q_nope_values);
        let q_rope_bf16 = bf16_bytes_from_f32(&q_rope_values);
        let q_nope = upload_device_bytes(library, &q_nope_bf16)?;
        let q_rope = upload_device_bytes(library, &q_rope_bf16)?;
        let scale = 0.25_f32;

        let attention =
            projected.run_mla_rope_attention_bf16(&rotated, q_nope.buffer, q_rope.buffer, scale)?;

        assert_eq!(
            attention.status(),
            "cuda-kv-cache-mla-rope-attention-device-buffer"
        );
        assert_eq!(attention.rows(), descriptor.token_count);
        assert_eq!(attention.heads(), heads);
        assert_eq!(attention.nope_dim(), nope_dim);
        assert_eq!(attention.rope_dim(), GLM52_MLA_QK_ROPE_HEAD_DIM);
        assert_eq!(attention.v_dim(), v_dim);
        assert_eq!(
            attention.output_buffer().bytes,
            descriptor.token_count * heads * v_dim * std::mem::size_of::<u16>()
        );

        let projected_readback = projected.copy_to_host()?;
        let rotated_readback = rotated.copy_to_host()?;
        let expected_bf16 = expected_mla_rope_attention_bf16(
            &q_nope_bf16,
            &q_rope_bf16,
            &projected_readback.k_nope_bf16,
            &rotated_readback.k_rope_rotated_bf16,
            &projected_readback.values_bf16,
            descriptor.token_count,
            heads,
            nope_dim,
            GLM52_MLA_QK_ROPE_HEAD_DIM,
            v_dim,
            scale,
        )?;
        let readback = attention.copy_to_host()?;
        assert_eq!(readback.status, "cuda-kv-cache-mla-rope-attention-readback");
        assert_eq!(readback.rows, descriptor.token_count);
        assert_eq!(readback.heads, heads);
        assert_eq!(readback.nope_dim, nope_dim);
        assert_eq!(readback.rope_dim, GLM52_MLA_QK_ROPE_HEAD_DIM);
        assert_eq!(readback.v_dim, v_dim);
        let actual = bf16_bytes_to_f32(&readback.output_bf16)?;
        let expected = bf16_bytes_to_f32(&expected_bf16)?;
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= 8.0e-3,
                "attention output {index} mismatch: actual={actual} expected={expected}"
            );
        }

        let summary = mirror.summary();
        assert_eq!(summary.status, "cuda-kv-cache-live-scheduler");
        assert_eq!(summary.writes, 1);
        assert_eq!(summary.reads, 1);
        assert_eq!(summary.bytes, payload.len() * 2);
        assert!(summary.uses_device_kv_cache);
        Ok(())
    }

    #[test]
    fn device_kv_execution_mirror_runs_mla_attention_with_host_suffix() -> Result<()> {
        let config = KvCacheConfig::glm52_phase0(8);
        let descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-attention-suffix".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(0),
            token_count: 1,
        };
        let latent = (0..GLM52_MLA_KV_LORA_RANK)
            .map(|index| ((index % 31) as f32 - 15.0) / 96.0)
            .collect::<Vec<_>>();
        let rope = (0..GLM52_MLA_QK_ROPE_HEAD_DIM)
            .map(|index| ((index % 23) as f32 - 11.0) / 128.0)
            .collect::<Vec<_>>();
        let mut payload = bf16_bytes_from_f32(&latent);
        payload.extend_from_slice(&bf16_bytes_from_f32(&rope));
        assert_eq!(
            payload.len(),
            config.descriptor_payload_bytes(&descriptor).unwrap()
        );

        let mut mirror = RealFullDeviceKvExecutionMirror::new(config)?;
        if !mirror.summary().uses_device_kv_cache {
            return Ok(());
        }

        mirror.write_host_blocks(
            std::slice::from_ref(&descriptor),
            std::slice::from_ref(&payload),
        )?;
        let parts = mirror
            .read_mla_kv_payloads_to_device_buffers(std::slice::from_ref(&descriptor))?
            .expect("CUDA-enabled mirror should return unpacked device buffers");
        let library = parts.kv_latent.library;
        let heads = 2;
        let nope_dim = 3;
        let v_dim = 2;
        let projected_width = heads * (nope_dim + v_dim);
        let norm_weight_bf16 = bf16_bytes_from_f32(&vec![1.0_f32; GLM52_MLA_KV_LORA_RANK]);
        let mut kv_b_weight_values = vec![0.0_f32; projected_width * GLM52_MLA_KV_LORA_RANK];
        for out_col in 0..projected_width {
            let selected_col = (out_col * 23 + 7) % GLM52_MLA_KV_LORA_RANK;
            kv_b_weight_values[out_col * GLM52_MLA_KV_LORA_RANK + selected_col] =
                if out_col % 2 == 0 { 0.625 } else { -0.5 };
        }
        let norm_weight = upload_device_bytes(library, &norm_weight_bf16)?;
        let kv_b_weight = upload_device_bytes(library, &bf16_bytes_from_f32(&kv_b_weight_values))?;
        let projected = parts.project_kv_latent_and_split_bf16(
            norm_weight.buffer,
            kv_b_weight.buffer,
            heads,
            nope_dim,
            v_dim,
            1.0e-5,
        )?;
        let rotated = parts.rotate_k_rope_bf16(&[0_u32], 10_000.0)?;

        let suffix_k_nope_values = (0..heads * nope_dim)
            .map(|index| ((index % 7) as f32 - 3.0) / 32.0)
            .collect::<Vec<_>>();
        let suffix_k_rope_values = (0..GLM52_MLA_QK_ROPE_HEAD_DIM)
            .map(|index| ((index % 19) as f32 - 9.0) / 128.0)
            .collect::<Vec<_>>();
        let suffix_values_values = (0..heads * v_dim)
            .map(|index| ((index % 5) as f32 - 2.0) / 16.0)
            .collect::<Vec<_>>();
        let suffix_k_nope_bf16 = bf16_bytes_from_f32(&suffix_k_nope_values);
        let suffix_k_rope_bf16 = bf16_bytes_from_f32(&suffix_k_rope_values);
        let suffix_k_rope_rotated_bf16 = expected_rope_bf16(
            &suffix_k_rope_bf16,
            &[1_u32],
            GLM52_MLA_QK_ROPE_HEAD_DIM,
            10_000.0,
        );
        let suffix_values_bf16 = bf16_bytes_from_f32(&suffix_values_values);
        let total_rows = descriptor.token_count + 1;
        let q_nope_values = (0..total_rows * heads * nope_dim)
            .map(|index| ((index % 13) as f32 - 6.0) / 64.0)
            .collect::<Vec<_>>();
        let q_rope_values = (0..total_rows * heads * GLM52_MLA_QK_ROPE_HEAD_DIM)
            .map(|index| ((index % 17) as f32 - 8.0) / 256.0)
            .collect::<Vec<_>>();
        let q_nope_bf16 = bf16_bytes_from_f32(&q_nope_values);
        let q_rope_bf16 = bf16_bytes_from_f32(&q_rope_values);
        let q_nope = upload_device_bytes(library, &q_nope_bf16)?;
        let q_rope = upload_device_bytes(library, &q_rope_bf16)?;
        let scale = 0.25_f32;

        let attention = projected.run_mla_rope_attention_with_host_suffix_bf16(
            &rotated,
            q_nope.buffer,
            q_rope.buffer,
            &suffix_k_nope_bf16,
            &suffix_k_rope_rotated_bf16,
            &suffix_values_bf16,
            scale,
        )?;

        assert_eq!(
            attention.status(),
            "cuda-kv-cache-mla-rope-attention-device-buffer-with-host-suffix"
        );
        assert_eq!(attention.rows(), total_rows);
        assert_eq!(attention.heads(), heads);
        assert_eq!(attention.nope_dim(), nope_dim);
        assert_eq!(attention.rope_dim(), GLM52_MLA_QK_ROPE_HEAD_DIM);
        assert_eq!(attention.v_dim(), v_dim);
        assert_eq!(
            attention.output_buffer().bytes,
            total_rows * heads * v_dim * std::mem::size_of::<u16>()
        );

        let projected_readback = projected.copy_to_host()?;
        let rotated_readback = rotated.copy_to_host()?;
        let mut expected_k_nope_bf16 = projected_readback.k_nope_bf16;
        expected_k_nope_bf16.extend_from_slice(&suffix_k_nope_bf16);
        let mut expected_k_rope_bf16 = rotated_readback.k_rope_rotated_bf16;
        expected_k_rope_bf16.extend_from_slice(&suffix_k_rope_rotated_bf16);
        let mut expected_values_bf16 = projected_readback.values_bf16;
        expected_values_bf16.extend_from_slice(&suffix_values_bf16);
        let expected_bf16 = expected_mla_rope_attention_bf16(
            &q_nope_bf16,
            &q_rope_bf16,
            &expected_k_nope_bf16,
            &expected_k_rope_bf16,
            &expected_values_bf16,
            total_rows,
            heads,
            nope_dim,
            GLM52_MLA_QK_ROPE_HEAD_DIM,
            v_dim,
            scale,
        )?;
        let readback = attention.copy_to_host()?;
        let actual = bf16_bytes_to_f32(&readback.output_bf16)?;
        let expected = bf16_bytes_to_f32(&expected_bf16)?;
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= 8.0e-3,
                "attention suffix output {index} mismatch: actual={actual} expected={expected}"
            );
        }

        let summary = mirror.summary();
        assert_eq!(summary.status, "cuda-kv-cache-live-scheduler");
        assert_eq!(summary.writes, 1);
        assert_eq!(summary.reads, 1);
        assert_eq!(summary.bytes, payload.len() * 2);
        assert!(summary.uses_device_kv_cache);
        Ok(())
    }

    #[test]
    fn device_kv_execution_mirror_runs_mla_attention_with_projected_query_suffix() -> Result<()> {
        let config = KvCacheConfig::glm52_phase0(8);
        let descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-attention-projected-query".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(0),
            token_count: 1,
        };
        let latent = (0..GLM52_MLA_KV_LORA_RANK)
            .map(|index| ((index % 31) as f32 - 15.0) / 96.0)
            .collect::<Vec<_>>();
        let rope = (0..GLM52_MLA_QK_ROPE_HEAD_DIM)
            .map(|index| ((index % 23) as f32 - 11.0) / 128.0)
            .collect::<Vec<_>>();
        let mut payload = bf16_bytes_from_f32(&latent);
        payload.extend_from_slice(&bf16_bytes_from_f32(&rope));

        let mut mirror = RealFullDeviceKvExecutionMirror::new(config)?;
        if !mirror.summary().uses_device_kv_cache {
            return Ok(());
        }
        mirror.write_host_blocks(
            std::slice::from_ref(&descriptor),
            std::slice::from_ref(&payload),
        )?;

        let library = cuda_native_library()?;
        let heads = 2;
        let nope_dim = 3;
        let v_dim = 2;
        let projected_width = heads * (nope_dim + v_dim);
        let norm_weight_bf16 = bf16_bytes_from_f32(&vec![1.0_f32; GLM52_MLA_KV_LORA_RANK]);
        let mut kv_b_weight_values = vec![0.0_f32; projected_width * GLM52_MLA_KV_LORA_RANK];
        for out_col in 0..projected_width {
            let selected_col = (out_col * 23 + 7) % GLM52_MLA_KV_LORA_RANK;
            kv_b_weight_values[out_col * GLM52_MLA_KV_LORA_RANK + selected_col] =
                if out_col % 2 == 0 { 0.625 } else { -0.5 };
        }
        let kv_b_weight_bf16 = bf16_bytes_from_f32(&kv_b_weight_values);
        let norm_weight = upload_device_bytes(library, &norm_weight_bf16)?;
        let kv_b_weight = upload_device_bytes(library, &kv_b_weight_bf16)?;

        let suffix_k_nope_bf16 = bf16_bytes_from_f32(
            &(0..heads * nope_dim)
                .map(|index| ((index % 7) as f32 - 3.0) / 32.0)
                .collect::<Vec<_>>(),
        );
        let suffix_k_rope_bf16 = bf16_bytes_from_f32(
            &(0..GLM52_MLA_QK_ROPE_HEAD_DIM)
                .map(|index| ((index % 19) as f32 - 9.0) / 128.0)
                .collect::<Vec<_>>(),
        );
        let suffix_k_rope_rotated_bf16 = expected_rope_bf16(
            &suffix_k_rope_bf16,
            &[1_u32],
            GLM52_MLA_QK_ROPE_HEAD_DIM,
            10_000.0,
        );
        let suffix_values_bf16 = bf16_bytes_from_f32(
            &(0..heads * v_dim)
                .map(|index| ((index % 5) as f32 - 2.0) / 16.0)
                .collect::<Vec<_>>(),
        );

        let suffix_q_nope_values = (0..heads * nope_dim)
            .map(|index| ((index % 13) as f32 - 6.0) / 64.0)
            .collect::<Vec<_>>();
        let suffix_q_rope_values = (0..heads * GLM52_MLA_QK_ROPE_HEAD_DIM)
            .map(|index| ((index % 17) as f32 - 8.0) / 256.0)
            .collect::<Vec<_>>();
        let mut q_projected_values =
            Vec::with_capacity(heads * (nope_dim + GLM52_MLA_QK_ROPE_HEAD_DIM));
        for head in 0..heads {
            let nope_start = head * nope_dim;
            let rope_start = head * GLM52_MLA_QK_ROPE_HEAD_DIM;
            q_projected_values
                .extend_from_slice(&suffix_q_nope_values[nope_start..nope_start + nope_dim]);
            q_projected_values.extend_from_slice(
                &suffix_q_rope_values[rope_start..rope_start + GLM52_MLA_QK_ROPE_HEAD_DIM],
            );
        }
        let q_projected_bf16 = bf16_bytes_from_f32(&q_projected_values);
        let q_projected_device = upload_device_bytes(library, &q_projected_bf16)?;
        let query_parts = RealFullDeviceMlaQueryParts::from_projected_suffix_bf16(
            library,
            &q_projected_bf16,
            descriptor.token_count,
            &[1_u32],
            heads,
            nope_dim,
            GLM52_MLA_QK_ROPE_HEAD_DIM,
            10_000.0,
        )?;
        assert_eq!(
            query_parts.status(),
            "cuda-kv-cache-mla-query-split-rope-device-buffers"
        );
        let query_readback = query_parts.copy_to_host()?;
        assert_eq!(query_readback.status, "cuda-kv-cache-mla-query-readback");
        assert_eq!(query_readback.rows, descriptor.token_count + 1);
        assert_eq!(query_readback.prefix_rows, descriptor.token_count);
        assert_eq!(query_readback.suffix_rows, 1);
        assert_eq!(query_readback.heads, heads);
        assert_eq!(query_readback.nope_dim, nope_dim);
        assert_eq!(query_readback.rope_dim, GLM52_MLA_QK_ROPE_HEAD_DIM);
        let query_device_parts = RealFullDeviceMlaQueryParts::from_projected_suffix_device_bf16(
            library,
            q_projected_device.buffer,
            descriptor.token_count,
            &[1_u32],
            heads,
            nope_dim,
            GLM52_MLA_QK_ROPE_HEAD_DIM,
            10_000.0,
        )?;
        assert_eq!(
            query_device_parts.status(),
            "cuda-kv-cache-mla-query-split-rope-device-buffers"
        );
        assert_eq!(query_device_parts.copy_to_host()?, query_readback);
        let attention = mirror
            .run_mla_rope_attention_parts_from_device_prefix_with_device_weights_and_projected_query_host_suffix_bf16(
                std::slice::from_ref(&descriptor),
                &[0_u32],
                norm_weight.buffer,
                kv_b_weight.buffer,
                &q_projected_bf16,
                &[1_u32],
                &suffix_k_nope_bf16,
                &suffix_k_rope_rotated_bf16,
                &suffix_values_bf16,
                heads,
                nope_dim,
                v_dim,
                1.0e-5,
                10_000.0,
                0.25,
            )?
            .expect("CUDA-enabled mirror should return projected-query prefix attention parts");

        assert_eq!(
            attention.status(),
            "cuda-kv-cache-mla-rope-attention-device-buffer-with-host-suffix"
        );
        let readback = attention.copy_to_host()?;
        let cache = mirror.cache.as_ref().expect("CUDA mirror should own cache");
        let first_attention_staging_ptr = cache.attention_host_upload_staging.buffer.ptr;
        let first_attention_staging_capacity = cache.attention_host_upload_staging.capacity;
        let first_projected_query_ptr = cache.attention_projected_query.buffer.ptr;
        let first_suffix_k_nope_ptr = cache.attention_suffix_k_nope.buffer.ptr;
        let first_suffix_k_rope_ptr = cache.attention_suffix_k_rope.buffer.ptr;
        let first_suffix_values_ptr = cache.attention_suffix_values.buffer.ptr;
        let first_query_split_unrotated_rope_ptr =
            cache.attention_query_split_unrotated_rope.buffer.ptr;
        let first_query_split_nope_ptr = cache.attention_query_split_nope.buffer.ptr;
        let first_query_split_rope_rotated_ptr =
            cache.attention_query_split_rope_rotated.buffer.ptr;
        let first_query_split_unrotated_rope_capacity =
            cache.attention_query_split_unrotated_rope.capacity;
        let first_query_split_nope_capacity = cache.attention_query_split_nope.capacity;
        let first_query_split_rope_rotated_capacity =
            cache.attention_query_split_rope_rotated.capacity;
        let first_combined_k_nope_ptr = cache.attention_combined_k_nope.buffer.ptr;
        let first_combined_k_rope_ptr = cache.attention_combined_k_rope.buffer.ptr;
        let first_combined_values_ptr = cache.attention_combined_values.buffer.ptr;
        let first_combined_k_nope_capacity = cache.attention_combined_k_nope.capacity;
        let first_combined_k_rope_capacity = cache.attention_combined_k_rope.capacity;
        let first_combined_values_capacity = cache.attention_combined_values.capacity;
        let first_rope_positions_ptr = cache.rope_positions_device.buffer.ptr;
        let first_rope_positions_capacity = cache.rope_positions_device.capacity;
        let first_prefix_unpacked_kv_latent_ptr = cache.mla_unpacked_kv_latent.buffer.ptr;
        let first_prefix_unpacked_k_rope_ptr = cache.mla_unpacked_k_rope.buffer.ptr;
        let first_prefix_unpacked_kv_latent_capacity = cache.mla_unpacked_kv_latent.capacity;
        let first_prefix_unpacked_k_rope_capacity = cache.mla_unpacked_k_rope.capacity;
        let first_prefix_projected_kv_normalized_ptr =
            cache.attention_projected_kv_normalized.buffer.ptr;
        let first_prefix_projected_kv_projected_ptr =
            cache.attention_projected_kv_projected.buffer.ptr;
        let first_prefix_projected_kv_k_nope_ptr = cache.attention_projected_kv_k_nope.buffer.ptr;
        let first_prefix_projected_kv_values_ptr = cache.attention_projected_kv_values.buffer.ptr;
        let first_prefix_rotated_k_rope_ptr = cache.attention_k_rope_rotated.buffer.ptr;
        let first_prefix_projected_kv_normalized_capacity =
            cache.attention_projected_kv_normalized.capacity;
        let first_prefix_projected_kv_projected_capacity =
            cache.attention_projected_kv_projected.capacity;
        let first_prefix_projected_kv_k_nope_capacity =
            cache.attention_projected_kv_k_nope.capacity;
        let first_prefix_projected_kv_values_capacity =
            cache.attention_projected_kv_values.capacity;
        let first_prefix_rotated_k_rope_capacity = cache.attention_k_rope_rotated.capacity;
        assert!(!first_attention_staging_ptr.is_null());
        assert!(first_attention_staging_capacity >= q_projected_bf16.len());
        assert!(!first_projected_query_ptr.is_null());
        assert!(!first_suffix_k_nope_ptr.is_null());
        assert!(!first_suffix_k_rope_ptr.is_null());
        assert!(!first_suffix_values_ptr.is_null());
        assert!(!first_query_split_unrotated_rope_ptr.is_null());
        assert!(!first_query_split_nope_ptr.is_null());
        assert!(!first_query_split_rope_rotated_ptr.is_null());
        assert!(first_query_split_unrotated_rope_capacity >= suffix_k_rope_rotated_bf16.len());
        assert!(first_query_split_nope_capacity >= suffix_k_nope_bf16.len());
        assert!(first_query_split_rope_rotated_capacity >= suffix_k_rope_rotated_bf16.len());
        assert!(!first_combined_k_nope_ptr.is_null());
        assert!(!first_combined_k_rope_ptr.is_null());
        assert!(!first_combined_values_ptr.is_null());
        assert!(first_combined_k_nope_capacity >= suffix_k_nope_bf16.len());
        assert!(first_combined_k_rope_capacity >= suffix_k_rope_rotated_bf16.len());
        assert!(first_combined_values_capacity >= suffix_values_bf16.len());
        assert!(!first_rope_positions_ptr.is_null());
        assert!(first_rope_positions_capacity >= std::mem::size_of::<u32>());
        assert!(!first_prefix_unpacked_kv_latent_ptr.is_null());
        assert!(!first_prefix_unpacked_k_rope_ptr.is_null());
        assert!(!first_prefix_projected_kv_normalized_ptr.is_null());
        assert!(!first_prefix_projected_kv_projected_ptr.is_null());
        assert!(!first_prefix_projected_kv_k_nope_ptr.is_null());
        assert!(!first_prefix_projected_kv_values_ptr.is_null());
        assert!(!first_prefix_rotated_k_rope_ptr.is_null());
        let device_query_attention = mirror
            .run_mla_rope_attention_parts_from_device_prefix_with_device_weights_and_projected_query_device_suffix_bf16(
                std::slice::from_ref(&descriptor),
                &[0_u32],
                norm_weight.buffer,
                kv_b_weight.buffer,
                q_projected_device.buffer,
                &[1_u32],
                &suffix_k_nope_bf16,
                &suffix_k_rope_rotated_bf16,
                &suffix_values_bf16,
                heads,
                nope_dim,
                v_dim,
                1.0e-5,
                10_000.0,
                0.25,
            )?
            .expect("CUDA-enabled mirror should return device projected-query prefix attention parts");
        assert_eq!(
            device_query_attention.status(),
            "cuda-kv-cache-mla-rope-attention-device-buffer-with-host-suffix"
        );
        let device_query_readback = device_query_attention.copy_to_host()?;
        let cache = mirror.cache.as_ref().expect("CUDA mirror should own cache");
        assert_eq!(
            cache.attention_host_upload_staging.buffer.ptr,
            first_attention_staging_ptr
        );
        assert_eq!(
            cache.attention_host_upload_staging.capacity,
            first_attention_staging_capacity
        );
        assert_eq!(
            cache.attention_projected_query.buffer.ptr,
            first_projected_query_ptr
        );
        assert_eq!(
            cache.attention_suffix_k_nope.buffer.ptr,
            first_suffix_k_nope_ptr
        );
        assert_eq!(
            cache.attention_suffix_k_rope.buffer.ptr,
            first_suffix_k_rope_ptr
        );
        assert_eq!(
            cache.attention_suffix_values.buffer.ptr,
            first_suffix_values_ptr
        );
        assert_eq!(
            cache.attention_query_split_unrotated_rope.buffer.ptr,
            first_query_split_unrotated_rope_ptr
        );
        assert_eq!(
            cache.attention_query_split_nope.buffer.ptr,
            first_query_split_nope_ptr
        );
        assert_eq!(
            cache.attention_query_split_rope_rotated.buffer.ptr,
            first_query_split_rope_rotated_ptr
        );
        assert_eq!(
            cache.attention_query_split_unrotated_rope.capacity,
            first_query_split_unrotated_rope_capacity
        );
        assert_eq!(
            cache.attention_query_split_nope.capacity,
            first_query_split_nope_capacity
        );
        assert_eq!(
            cache.attention_query_split_rope_rotated.capacity,
            first_query_split_rope_rotated_capacity
        );
        assert_eq!(
            cache.attention_combined_k_nope.buffer.ptr,
            first_combined_k_nope_ptr
        );
        assert_eq!(
            cache.attention_combined_k_rope.buffer.ptr,
            first_combined_k_rope_ptr
        );
        assert_eq!(
            cache.attention_combined_values.buffer.ptr,
            first_combined_values_ptr
        );
        assert_eq!(
            cache.attention_combined_k_nope.capacity,
            first_combined_k_nope_capacity
        );
        assert_eq!(
            cache.attention_combined_k_rope.capacity,
            first_combined_k_rope_capacity
        );
        assert_eq!(
            cache.attention_combined_values.capacity,
            first_combined_values_capacity
        );
        assert_eq!(
            cache.rope_positions_device.buffer.ptr,
            first_rope_positions_ptr
        );
        assert_eq!(
            cache.rope_positions_device.capacity,
            first_rope_positions_capacity
        );
        assert_eq!(
            cache.mla_unpacked_kv_latent.buffer.ptr,
            first_prefix_unpacked_kv_latent_ptr
        );
        assert_eq!(
            cache.mla_unpacked_k_rope.buffer.ptr,
            first_prefix_unpacked_k_rope_ptr
        );
        assert_eq!(
            cache.mla_unpacked_kv_latent.capacity,
            first_prefix_unpacked_kv_latent_capacity
        );
        assert_eq!(
            cache.mla_unpacked_k_rope.capacity,
            first_prefix_unpacked_k_rope_capacity
        );
        assert_eq!(
            cache.attention_projected_kv_normalized.buffer.ptr,
            first_prefix_projected_kv_normalized_ptr
        );
        assert_eq!(
            cache.attention_projected_kv_projected.buffer.ptr,
            first_prefix_projected_kv_projected_ptr
        );
        assert_eq!(
            cache.attention_projected_kv_k_nope.buffer.ptr,
            first_prefix_projected_kv_k_nope_ptr
        );
        assert_eq!(
            cache.attention_projected_kv_values.buffer.ptr,
            first_prefix_projected_kv_values_ptr
        );
        assert_eq!(
            cache.attention_k_rope_rotated.buffer.ptr,
            first_prefix_rotated_k_rope_ptr
        );
        assert_eq!(
            cache.attention_projected_kv_normalized.capacity,
            first_prefix_projected_kv_normalized_capacity
        );
        assert_eq!(
            cache.attention_projected_kv_projected.capacity,
            first_prefix_projected_kv_projected_capacity
        );
        assert_eq!(
            cache.attention_projected_kv_k_nope.capacity,
            first_prefix_projected_kv_k_nope_capacity
        );
        assert_eq!(
            cache.attention_projected_kv_values.capacity,
            first_prefix_projected_kv_values_capacity
        );
        assert_eq!(
            cache.attention_k_rope_rotated.capacity,
            first_prefix_rotated_k_rope_capacity
        );

        let parts = mirror
            .read_mla_kv_payloads_to_device_buffers(std::slice::from_ref(&descriptor))?
            .expect("CUDA-enabled mirror should return unpacked device buffers");
        let projected = parts.project_kv_latent_and_split_bf16(
            norm_weight.buffer,
            kv_b_weight.buffer,
            heads,
            nope_dim,
            v_dim,
            1.0e-5,
        )?;
        let rotated = parts.rotate_k_rope_bf16(&[0_u32], 10_000.0)?;
        let projected_readback = projected.copy_to_host()?;
        let rotated_readback = rotated.copy_to_host()?;
        let suffix_kv_latent_values = (0..GLM52_MLA_KV_LORA_RANK)
            .map(|index| ((index % 37) as f32 - 18.0) / 128.0)
            .collect::<Vec<_>>();
        let suffix_device_k_rope_values = (0..GLM52_MLA_QK_ROPE_HEAD_DIM)
            .map(|index| ((index % 19) as f32 - 9.0) / 128.0)
            .collect::<Vec<_>>();
        let mut suffix_kv_a_projected_bf16 = bf16_bytes_from_f32(&suffix_kv_latent_values);
        suffix_kv_a_projected_bf16
            .extend_from_slice(&bf16_bytes_from_f32(&suffix_device_k_rope_values));
        let suffix_kv_a_projected = upload_device_bytes(library, &suffix_kv_a_projected_bf16)?;
        let suffix_parts = RealFullDeviceMlaKvDeviceParts::from_projected_kv_a_device_bf16(
            library,
            suffix_kv_a_projected.buffer,
            descriptor.layer_id,
            1,
            GLM52_MLA_KV_LORA_RANK,
            GLM52_MLA_QK_ROPE_HEAD_DIM,
        )?;
        assert_eq!(
            suffix_parts.status(),
            "cuda-kv-cache-mla-current-kv-split-device-buffers"
        );
        let suffix_projected = suffix_parts.project_kv_latent_and_split_bf16(
            norm_weight.buffer,
            kv_b_weight.buffer,
            heads,
            nope_dim,
            v_dim,
            1.0e-5,
        )?;
        let suffix_rotated = suffix_parts.rotate_k_rope_bf16(&[1_u32], 10_000.0)?;
        let suffix_projected_readback = suffix_projected.copy_to_host()?;
        let suffix_rotated_readback = suffix_rotated.copy_to_host()?;
        let device_kv_suffix_attention = mirror
            .run_mla_rope_attention_parts_from_device_prefix_with_device_weights_and_projected_query_device_kv_suffix_bf16(
                std::slice::from_ref(&descriptor),
                &[0_u32],
                norm_weight.buffer,
                kv_b_weight.buffer,
                q_projected_device.buffer,
                suffix_kv_a_projected.buffer,
                &[1_u32],
                heads,
                nope_dim,
                v_dim,
                1.0e-5,
                10_000.0,
                0.25,
            )?
            .expect("CUDA-enabled mirror should return device projected-query and device-KV-suffix prefix attention parts");
        assert_eq!(
            device_kv_suffix_attention.status(),
            "cuda-kv-cache-mla-rope-attention-device-buffer-with-device-suffix"
        );
        let device_kv_suffix_readback = device_kv_suffix_attention.copy_to_host()?;
        let cache = mirror.cache.as_ref().expect("CUDA mirror should own cache");
        let first_current_kv_latent_ptr = cache.mla_current_kv_latent.buffer.ptr;
        let first_current_k_rope_ptr = cache.mla_current_k_rope.buffer.ptr;
        let first_current_kv_latent_capacity = cache.mla_current_kv_latent.capacity;
        let first_current_k_rope_capacity = cache.mla_current_k_rope.capacity;
        let first_suffix_projected_kv_normalized_ptr =
            cache.attention_suffix_projected_kv_normalized.buffer.ptr;
        let first_suffix_projected_kv_projected_ptr =
            cache.attention_suffix_projected_kv_projected.buffer.ptr;
        let first_device_suffix_k_nope_ptr = cache.attention_suffix_k_nope.buffer.ptr;
        let first_device_suffix_k_rope_ptr = cache.attention_suffix_k_rope.buffer.ptr;
        let first_device_suffix_values_ptr = cache.attention_suffix_values.buffer.ptr;
        let first_suffix_projected_kv_normalized_capacity =
            cache.attention_suffix_projected_kv_normalized.capacity;
        let first_suffix_projected_kv_projected_capacity =
            cache.attention_suffix_projected_kv_projected.capacity;
        let first_device_suffix_k_nope_capacity = cache.attention_suffix_k_nope.capacity;
        let first_device_suffix_k_rope_capacity = cache.attention_suffix_k_rope.capacity;
        let first_device_suffix_values_capacity = cache.attention_suffix_values.capacity;
        assert!(!first_current_kv_latent_ptr.is_null());
        assert!(!first_current_k_rope_ptr.is_null());
        assert!(!first_suffix_projected_kv_normalized_ptr.is_null());
        assert!(!first_suffix_projected_kv_projected_ptr.is_null());
        assert!(!first_device_suffix_k_nope_ptr.is_null());
        assert!(!first_device_suffix_k_rope_ptr.is_null());
        assert!(!first_device_suffix_values_ptr.is_null());
        assert_eq!(
            cache.attention_combined_k_nope.buffer.ptr,
            first_combined_k_nope_ptr
        );
        assert_eq!(
            cache.attention_combined_k_rope.buffer.ptr,
            first_combined_k_rope_ptr
        );
        assert_eq!(
            cache.attention_combined_values.buffer.ptr,
            first_combined_values_ptr
        );
        assert_eq!(
            cache.attention_query_split_unrotated_rope.buffer.ptr,
            first_query_split_unrotated_rope_ptr
        );
        assert_eq!(
            cache.attention_query_split_nope.buffer.ptr,
            first_query_split_nope_ptr
        );
        assert_eq!(
            cache.attention_query_split_rope_rotated.buffer.ptr,
            first_query_split_rope_rotated_ptr
        );
        assert_eq!(
            cache.mla_unpacked_kv_latent.buffer.ptr,
            first_prefix_unpacked_kv_latent_ptr
        );
        assert_eq!(
            cache.mla_unpacked_k_rope.buffer.ptr,
            first_prefix_unpacked_k_rope_ptr
        );
        assert_eq!(
            cache.mla_unpacked_kv_latent.capacity,
            first_prefix_unpacked_kv_latent_capacity
        );
        assert_eq!(
            cache.mla_unpacked_k_rope.capacity,
            first_prefix_unpacked_k_rope_capacity
        );
        assert_eq!(
            cache.attention_projected_kv_normalized.buffer.ptr,
            first_prefix_projected_kv_normalized_ptr
        );
        assert_eq!(
            cache.attention_projected_kv_projected.buffer.ptr,
            first_prefix_projected_kv_projected_ptr
        );
        assert_eq!(
            cache.attention_projected_kv_k_nope.buffer.ptr,
            first_prefix_projected_kv_k_nope_ptr
        );
        assert_eq!(
            cache.attention_projected_kv_values.buffer.ptr,
            first_prefix_projected_kv_values_ptr
        );
        assert_eq!(
            cache.attention_k_rope_rotated.buffer.ptr,
            first_prefix_rotated_k_rope_ptr
        );
        assert_eq!(
            cache.attention_projected_kv_normalized.capacity,
            first_prefix_projected_kv_normalized_capacity
        );
        assert_eq!(
            cache.attention_projected_kv_projected.capacity,
            first_prefix_projected_kv_projected_capacity
        );
        assert_eq!(
            cache.attention_projected_kv_k_nope.capacity,
            first_prefix_projected_kv_k_nope_capacity
        );
        assert_eq!(
            cache.attention_projected_kv_values.capacity,
            first_prefix_projected_kv_values_capacity
        );
        assert_eq!(
            cache.attention_k_rope_rotated.capacity,
            first_prefix_rotated_k_rope_capacity
        );
        let second_device_kv_suffix_attention = mirror
            .run_mla_rope_attention_parts_from_device_prefix_with_device_weights_and_projected_query_device_kv_suffix_bf16(
                std::slice::from_ref(&descriptor),
                &[0_u32],
                norm_weight.buffer,
                kv_b_weight.buffer,
                q_projected_device.buffer,
                suffix_kv_a_projected.buffer,
                &[1_u32],
                heads,
                nope_dim,
                v_dim,
                1.0e-5,
                10_000.0,
                0.25,
            )?
            .expect("CUDA-enabled mirror should return second device projected-query and device-KV-suffix prefix attention parts");
        assert_eq!(
            second_device_kv_suffix_attention.status(),
            "cuda-kv-cache-mla-rope-attention-device-buffer-with-device-suffix"
        );
        let second_device_kv_suffix_readback = second_device_kv_suffix_attention.copy_to_host()?;
        assert_eq!(
            second_device_kv_suffix_readback.output_bf16,
            device_kv_suffix_readback.output_bf16
        );
        let cache = mirror.cache.as_ref().expect("CUDA mirror should own cache");
        assert_eq!(
            cache.mla_current_kv_latent.buffer.ptr,
            first_current_kv_latent_ptr
        );
        assert_eq!(
            cache.mla_current_k_rope.buffer.ptr,
            first_current_k_rope_ptr
        );
        assert_eq!(
            cache.mla_current_kv_latent.capacity,
            first_current_kv_latent_capacity
        );
        assert_eq!(
            cache.mla_current_k_rope.capacity,
            first_current_k_rope_capacity
        );
        assert_eq!(
            cache.attention_projected_kv_normalized.buffer.ptr,
            first_prefix_projected_kv_normalized_ptr
        );
        assert_eq!(
            cache.attention_projected_kv_projected.buffer.ptr,
            first_prefix_projected_kv_projected_ptr
        );
        assert_eq!(
            cache.attention_projected_kv_k_nope.buffer.ptr,
            first_prefix_projected_kv_k_nope_ptr
        );
        assert_eq!(
            cache.attention_projected_kv_values.buffer.ptr,
            first_prefix_projected_kv_values_ptr
        );
        assert_eq!(
            cache.attention_k_rope_rotated.buffer.ptr,
            first_prefix_rotated_k_rope_ptr
        );
        assert_eq!(
            cache.attention_projected_kv_normalized.capacity,
            first_prefix_projected_kv_normalized_capacity
        );
        assert_eq!(
            cache.attention_projected_kv_projected.capacity,
            first_prefix_projected_kv_projected_capacity
        );
        assert_eq!(
            cache.attention_projected_kv_k_nope.capacity,
            first_prefix_projected_kv_k_nope_capacity
        );
        assert_eq!(
            cache.attention_projected_kv_values.capacity,
            first_prefix_projected_kv_values_capacity
        );
        assert_eq!(
            cache.attention_k_rope_rotated.capacity,
            first_prefix_rotated_k_rope_capacity
        );
        assert_eq!(
            cache.attention_suffix_projected_kv_normalized.buffer.ptr,
            first_suffix_projected_kv_normalized_ptr
        );
        assert_eq!(
            cache.attention_suffix_projected_kv_projected.buffer.ptr,
            first_suffix_projected_kv_projected_ptr
        );
        assert_eq!(
            cache.attention_suffix_k_nope.buffer.ptr,
            first_device_suffix_k_nope_ptr
        );
        assert_eq!(
            cache.attention_suffix_k_rope.buffer.ptr,
            first_device_suffix_k_rope_ptr
        );
        assert_eq!(
            cache.attention_suffix_values.buffer.ptr,
            first_device_suffix_values_ptr
        );
        assert_eq!(
            cache.attention_suffix_projected_kv_normalized.capacity,
            first_suffix_projected_kv_normalized_capacity
        );
        assert_eq!(
            cache.attention_suffix_projected_kv_projected.capacity,
            first_suffix_projected_kv_projected_capacity
        );
        assert_eq!(
            cache.attention_suffix_k_nope.capacity,
            first_device_suffix_k_nope_capacity
        );
        assert_eq!(
            cache.attention_suffix_k_rope.capacity,
            first_device_suffix_k_rope_capacity
        );
        assert_eq!(
            cache.attention_suffix_values.capacity,
            first_device_suffix_values_capacity
        );
        let current_descriptor = KvBlockDescriptor {
            reservation_id: 2,
            sequence_id: descriptor.sequence_id.clone(),
            layer_id: descriptor.layer_id,
            token_start: PositionId(1),
            token_count: 1,
        };
        let current_writes = mirror
            .write_projected_mla_kv_a_device_blocks_bf16(
                std::slice::from_ref(&current_descriptor),
                suffix_kv_a_projected.buffer,
                None,
            )?
            .expect("CUDA-enabled mirror should write current projected kv_a to device cache");
        assert_eq!(current_writes.len(), 1);
        let cache_attention_descriptors = [descriptor.clone(), current_descriptor];
        let device_cache_attention = mirror
            .run_mla_rope_attention_parts_from_device_kv_with_device_weights_and_projected_query_device_bf16(
                &cache_attention_descriptors,
                &[0_u32, 1_u32],
                norm_weight.buffer,
                kv_b_weight.buffer,
                q_projected_device.buffer,
                None,
                None,
                &[1_u32],
                heads,
                nope_dim,
                v_dim,
                1.0e-5,
                10_000.0,
                0.25,
            )?
            .expect("CUDA-enabled mirror should return device-cache projected-query attention parts");
        assert_eq!(
            device_cache_attention.status(),
            "cuda-kv-cache-mla-rope-attention-device-buffer"
        );
        let device_cache_readback = device_cache_attention.copy_to_host()?;
        assert_eq!(device_cache_readback.rows, 1);
        assert_eq!(
            device_cache_readback.output_bf16.len(),
            heads * v_dim * std::mem::size_of::<u16>()
        );
        let cache = mirror.cache.as_ref().expect("CUDA mirror should own cache");
        assert_eq!(
            cache.attention_query_split_unrotated_rope.buffer.ptr,
            first_query_split_unrotated_rope_ptr
        );
        assert_eq!(
            cache.attention_query_split_nope.buffer.ptr,
            first_query_split_nope_ptr
        );
        assert_eq!(
            cache.attention_query_split_rope_rotated.buffer.ptr,
            first_query_split_rope_rotated_ptr
        );
        let first_projected_kv_normalized_ptr = cache.attention_projected_kv_normalized.buffer.ptr;
        let first_projected_kv_projected_ptr = cache.attention_projected_kv_projected.buffer.ptr;
        let first_projected_kv_k_nope_ptr = cache.attention_projected_kv_k_nope.buffer.ptr;
        let first_projected_kv_values_ptr = cache.attention_projected_kv_values.buffer.ptr;
        let first_unpacked_kv_latent_ptr = cache.mla_unpacked_kv_latent.buffer.ptr;
        let first_unpacked_k_rope_ptr = cache.mla_unpacked_k_rope.buffer.ptr;
        let first_rotated_k_rope_ptr = cache.attention_k_rope_rotated.buffer.ptr;
        let first_projected_kv_normalized_capacity =
            cache.attention_projected_kv_normalized.capacity;
        let first_projected_kv_projected_capacity = cache.attention_projected_kv_projected.capacity;
        let first_projected_kv_k_nope_capacity = cache.attention_projected_kv_k_nope.capacity;
        let first_projected_kv_values_capacity = cache.attention_projected_kv_values.capacity;
        let first_unpacked_kv_latent_capacity = cache.mla_unpacked_kv_latent.capacity;
        let first_unpacked_k_rope_capacity = cache.mla_unpacked_k_rope.capacity;
        let first_rotated_k_rope_capacity = cache.attention_k_rope_rotated.capacity;
        assert!(!first_projected_kv_normalized_ptr.is_null());
        assert!(!first_projected_kv_projected_ptr.is_null());
        assert!(!first_projected_kv_k_nope_ptr.is_null());
        assert!(!first_projected_kv_values_ptr.is_null());
        assert!(!first_unpacked_kv_latent_ptr.is_null());
        assert!(!first_unpacked_k_rope_ptr.is_null());
        assert!(!first_rotated_k_rope_ptr.is_null());
        let second_device_cache_attention = mirror
            .run_mla_rope_attention_parts_from_device_kv_with_device_weights_and_projected_query_device_bf16(
                &cache_attention_descriptors,
                &[0_u32, 1_u32],
                norm_weight.buffer,
                kv_b_weight.buffer,
                q_projected_device.buffer,
                None,
                None,
                &[1_u32],
                heads,
                nope_dim,
                v_dim,
                1.0e-5,
                10_000.0,
                0.25,
            )?
            .expect("CUDA-enabled mirror should return second device-cache projected-query attention parts");
        assert_eq!(
            second_device_cache_attention.status(),
            "cuda-kv-cache-mla-rope-attention-device-buffer"
        );
        let second_device_cache_readback = second_device_cache_attention.copy_to_host()?;
        assert_eq!(second_device_cache_readback.rows, 1);
        assert_eq!(
            second_device_cache_readback.output_bf16,
            device_cache_readback.output_bf16
        );
        let cache = mirror.cache.as_ref().expect("CUDA mirror should own cache");
        assert_eq!(
            cache.attention_projected_kv_normalized.buffer.ptr,
            first_projected_kv_normalized_ptr
        );
        assert_eq!(
            cache.attention_projected_kv_projected.buffer.ptr,
            first_projected_kv_projected_ptr
        );
        assert_eq!(
            cache.attention_projected_kv_k_nope.buffer.ptr,
            first_projected_kv_k_nope_ptr
        );
        assert_eq!(
            cache.attention_projected_kv_values.buffer.ptr,
            first_projected_kv_values_ptr
        );
        assert_eq!(
            cache.mla_unpacked_kv_latent.buffer.ptr,
            first_unpacked_kv_latent_ptr
        );
        assert_eq!(
            cache.mla_unpacked_k_rope.buffer.ptr,
            first_unpacked_k_rope_ptr
        );
        assert_eq!(
            cache.attention_k_rope_rotated.buffer.ptr,
            first_rotated_k_rope_ptr
        );
        assert_eq!(
            cache.attention_projected_kv_normalized.capacity,
            first_projected_kv_normalized_capacity
        );
        assert_eq!(
            cache.attention_projected_kv_projected.capacity,
            first_projected_kv_projected_capacity
        );
        assert_eq!(
            cache.attention_projected_kv_k_nope.capacity,
            first_projected_kv_k_nope_capacity
        );
        assert_eq!(
            cache.attention_projected_kv_values.capacity,
            first_projected_kv_values_capacity
        );
        assert_eq!(
            cache.mla_unpacked_kv_latent.capacity,
            first_unpacked_kv_latent_capacity
        );
        assert_eq!(
            cache.mla_unpacked_k_rope.capacity,
            first_unpacked_k_rope_capacity
        );
        assert_eq!(
            cache.attention_k_rope_rotated.capacity,
            first_rotated_k_rope_capacity
        );

        let mut expected_k_nope_bf16 = projected_readback.k_nope_bf16.clone();
        expected_k_nope_bf16.extend_from_slice(&suffix_k_nope_bf16);
        let mut expected_k_rope_bf16 = rotated_readback.k_rope_rotated_bf16.clone();
        expected_k_rope_bf16.extend_from_slice(&suffix_k_rope_rotated_bf16);
        let mut expected_values_bf16 = projected_readback.values_bf16.clone();
        expected_values_bf16.extend_from_slice(&suffix_values_bf16);

        let mut expected_q_nope_bf16 = vec![0_u8; heads * nope_dim * std::mem::size_of::<u16>()];
        expected_q_nope_bf16.extend_from_slice(&bf16_bytes_from_f32(&suffix_q_nope_values));
        let suffix_q_rope_bf16 = bf16_bytes_from_f32(&suffix_q_rope_values);
        let mut suffix_positions_by_head = Vec::with_capacity(heads);
        suffix_positions_by_head.resize(heads, 1_u32);
        let suffix_q_rope_rotated_bf16 = expected_rope_bf16(
            &suffix_q_rope_bf16,
            &suffix_positions_by_head,
            GLM52_MLA_QK_ROPE_HEAD_DIM,
            10_000.0,
        );
        assert_eq!(query_readback.q_nope_bf16, expected_q_nope_bf16);
        let mut expected_q_rope_bf16 =
            vec![0_u8; heads * GLM52_MLA_QK_ROPE_HEAD_DIM * std::mem::size_of::<u16>()];
        expected_q_rope_bf16.extend_from_slice(&suffix_q_rope_rotated_bf16);
        assert_eq!(query_readback.q_rope_rotated_bf16, expected_q_rope_bf16);

        let expected_bf16 = expected_mla_rope_attention_bf16(
            &expected_q_nope_bf16,
            &expected_q_rope_bf16,
            &expected_k_nope_bf16,
            &expected_k_rope_bf16,
            &expected_values_bf16,
            descriptor.token_count + 1,
            heads,
            nope_dim,
            GLM52_MLA_QK_ROPE_HEAD_DIM,
            v_dim,
            0.25,
        )?;
        let mut expected_device_suffix_k_nope_bf16 = projected_readback.k_nope_bf16.clone();
        expected_device_suffix_k_nope_bf16
            .extend_from_slice(&suffix_projected_readback.k_nope_bf16);
        let mut expected_device_suffix_k_rope_bf16 = rotated_readback.k_rope_rotated_bf16.clone();
        expected_device_suffix_k_rope_bf16
            .extend_from_slice(&suffix_rotated_readback.k_rope_rotated_bf16);
        let mut expected_device_suffix_values_bf16 = projected_readback.values_bf16.clone();
        expected_device_suffix_values_bf16
            .extend_from_slice(&suffix_projected_readback.values_bf16);
        let expected_device_suffix_bf16 = expected_mla_rope_attention_bf16(
            &expected_q_nope_bf16,
            &expected_q_rope_bf16,
            &expected_device_suffix_k_nope_bf16,
            &expected_device_suffix_k_rope_bf16,
            &expected_device_suffix_values_bf16,
            descriptor.token_count + 1,
            heads,
            nope_dim,
            GLM52_MLA_QK_ROPE_HEAD_DIM,
            v_dim,
            0.25,
        )?;
        let expected_device_cache_suffix_row_bytes = heads * v_dim * std::mem::size_of::<u16>();
        let expected_device_cache_suffix_bf16 = expected_device_suffix_bf16
            [expected_device_suffix_bf16.len() - expected_device_cache_suffix_row_bytes..]
            .to_vec();
        let actual = bf16_bytes_to_f32(&readback.output_bf16)?;
        let device_query_actual = bf16_bytes_to_f32(&device_query_readback.output_bf16)?;
        let device_kv_suffix_actual = bf16_bytes_to_f32(&device_kv_suffix_readback.output_bf16)?;
        let device_cache_actual = bf16_bytes_to_f32(&device_cache_readback.output_bf16)?;
        let expected = bf16_bytes_to_f32(&expected_bf16)?;
        let expected_device_suffix = bf16_bytes_to_f32(&expected_device_suffix_bf16)?;
        let expected_device_cache_suffix = bf16_bytes_to_f32(&expected_device_cache_suffix_bf16)?;
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= 8.0e-3,
                "projected-query attention output {index} mismatch: actual={actual} expected={expected}"
            );
        }
        for (index, (actual, expected)) in
            device_query_actual.iter().zip(expected.iter()).enumerate()
        {
            assert!(
                (actual - expected).abs() <= 8.0e-3,
                "device projected-query attention output {index} mismatch: actual={actual} expected={expected}"
            );
        }
        for (index, (actual, expected)) in device_kv_suffix_actual
            .iter()
            .zip(expected_device_suffix.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 8.0e-3,
                "device projected-query and device-KV-suffix attention output {index} mismatch: actual={actual} expected={expected}"
            );
        }
        assert_eq!(
            device_cache_actual.len(),
            expected_device_cache_suffix.len()
        );
        for (index, (actual, expected)) in device_cache_actual
            .iter()
            .zip(expected_device_cache_suffix.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 8.0e-3,
                "device-cache projected-query attention output {index} mismatch: actual={actual} expected={expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn device_kv_execution_mirror_runs_prefix_attention_public_bridge() -> Result<()> {
        let config = KvCacheConfig::glm52_phase0(8);
        let descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-attention-public-bridge".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(0),
            token_count: 1,
        };
        let latent = (0..GLM52_MLA_KV_LORA_RANK)
            .map(|index| ((index % 29) as f32 - 14.0) / 80.0)
            .collect::<Vec<_>>();
        let rope = (0..GLM52_MLA_QK_ROPE_HEAD_DIM)
            .map(|index| ((index % 17) as f32 - 8.0) / 112.0)
            .collect::<Vec<_>>();
        let mut payload = bf16_bytes_from_f32(&latent);
        payload.extend_from_slice(&bf16_bytes_from_f32(&rope));
        assert_eq!(
            payload.len(),
            config.descriptor_payload_bytes(&descriptor).unwrap()
        );

        let mut mirror = RealFullDeviceKvExecutionMirror::new(config.clone())?;
        if !mirror.summary().uses_device_kv_cache {
            return Ok(());
        }

        mirror.write_host_blocks(
            std::slice::from_ref(&descriptor),
            std::slice::from_ref(&payload),
        )?;
        let heads = 2;
        let nope_dim = 3;
        let v_dim = 2;
        let projected_width = heads * (nope_dim + v_dim);
        let norm_weight_bf16 = bf16_bytes_from_f32(&vec![1.0_f32; GLM52_MLA_KV_LORA_RANK]);
        let mut kv_b_weight_values = vec![0.0_f32; projected_width * GLM52_MLA_KV_LORA_RANK];
        for out_col in 0..projected_width {
            let selected_col = (out_col * 13 + 11) % GLM52_MLA_KV_LORA_RANK;
            kv_b_weight_values[out_col * GLM52_MLA_KV_LORA_RANK + selected_col] =
                if out_col % 2 == 0 { 0.75 } else { -0.375 };
        }
        let kv_b_weight_bf16 = bf16_bytes_from_f32(&kv_b_weight_values);
        let suffix_k_nope_bf16 = bf16_bytes_from_f32(
            &(0..heads * nope_dim)
                .map(|index| ((index % 7) as f32 - 3.0) / 48.0)
                .collect::<Vec<_>>(),
        );
        let suffix_k_rope_bf16 = bf16_bytes_from_f32(
            &(0..GLM52_MLA_QK_ROPE_HEAD_DIM)
                .map(|index| ((index % 13) as f32 - 6.0) / 96.0)
                .collect::<Vec<_>>(),
        );
        let suffix_k_rope_rotated_bf16 = expected_rope_bf16(
            &suffix_k_rope_bf16,
            &[1_u32],
            GLM52_MLA_QK_ROPE_HEAD_DIM,
            10_000.0,
        );
        let suffix_values_bf16 = bf16_bytes_from_f32(
            &(0..heads * v_dim)
                .map(|index| ((index % 5) as f32 - 2.0) / 20.0)
                .collect::<Vec<_>>(),
        );
        let total_rows = descriptor.token_count + 1;
        let q_nope_bf16 = bf16_bytes_from_f32(
            &(0..total_rows * heads * nope_dim)
                .map(|index| ((index % 11) as f32 - 5.0) / 64.0)
                .collect::<Vec<_>>(),
        );
        let q_rope_bf16 = bf16_bytes_from_f32(
            &(0..total_rows * heads * GLM52_MLA_QK_ROPE_HEAD_DIM)
                .map(|index| ((index % 19) as f32 - 9.0) / 256.0)
                .collect::<Vec<_>>(),
        );

        let readback = mirror
            .run_mla_rope_attention_from_device_prefix_with_host_suffix_bf16(
                std::slice::from_ref(&descriptor),
                &[0_u32],
                &norm_weight_bf16,
                &kv_b_weight_bf16,
                &q_nope_bf16,
                &q_rope_bf16,
                &suffix_k_nope_bf16,
                &suffix_k_rope_rotated_bf16,
                &suffix_values_bf16,
                heads,
                nope_dim,
                v_dim,
                1.0e-5,
                10_000.0,
                0.25,
            )?
            .expect("CUDA-enabled mirror should return prefix attention readback");

        assert_eq!(readback.status, "cuda-kv-cache-mla-rope-attention-readback");
        assert_eq!(readback.rows, total_rows);
        assert_eq!(readback.heads, heads);
        assert_eq!(readback.nope_dim, nope_dim);
        assert_eq!(readback.rope_dim, GLM52_MLA_QK_ROPE_HEAD_DIM);
        assert_eq!(readback.v_dim, v_dim);
        assert_eq!(
            readback.output_bf16.len(),
            total_rows * heads * v_dim * std::mem::size_of::<u16>()
        );

        let summary = mirror.summary();
        assert_eq!(summary.status, "cuda-kv-cache-live-scheduler");
        assert_eq!(summary.writes, 1);
        assert_eq!(summary.reads, 1);
        assert_eq!(summary.bytes, payload.len() * 2);
        assert!(summary.uses_device_kv_cache);

        let library = cuda_native_library()?;
        let norm_weight = upload_device_bytes(library, &norm_weight_bf16)?;
        let kv_b_weight = upload_device_bytes(library, &kv_b_weight_bf16)?;
        let mut device_weight_mirror = RealFullDeviceKvExecutionMirror::new(config)?;
        device_weight_mirror.write_host_blocks(
            std::slice::from_ref(&descriptor),
            std::slice::from_ref(&payload),
        )?;
        let device_weight_readback = device_weight_mirror
            .run_mla_rope_attention_from_device_prefix_with_device_weights_and_host_suffix_bf16(
                std::slice::from_ref(&descriptor),
                &[0_u32],
                norm_weight.buffer,
                kv_b_weight.buffer,
                &q_nope_bf16,
                &q_rope_bf16,
                &suffix_k_nope_bf16,
                &suffix_k_rope_rotated_bf16,
                &suffix_values_bf16,
                heads,
                nope_dim,
                v_dim,
                1.0e-5,
                10_000.0,
                0.25,
            )?
            .expect("CUDA-enabled mirror should return resident-weight prefix attention readback");
        assert_eq!(
            device_weight_readback.status,
            "cuda-kv-cache-mla-rope-attention-readback"
        );
        assert_eq!(device_weight_readback.rows, total_rows);
        assert_eq!(
            device_weight_readback.output_bf16.len(),
            total_rows * heads * v_dim * std::mem::size_of::<u16>()
        );
        let device_weight_summary = device_weight_mirror.summary();
        assert_eq!(device_weight_summary.status, "cuda-kv-cache-live-scheduler");
        assert_eq!(device_weight_summary.writes, 1);
        assert_eq!(device_weight_summary.reads, 1);
        assert_eq!(device_weight_summary.bytes, payload.len() * 2);
        assert!(device_weight_summary.uses_device_kv_cache);
        let cache = device_weight_mirror
            .cache
            .as_ref()
            .expect("CUDA mirror should own cache");
        let first_attention_staging_ptr = cache.attention_host_upload_staging.buffer.ptr;
        let first_attention_staging_capacity = cache.attention_host_upload_staging.capacity;
        let first_query_nope_ptr = cache.attention_query_nope.buffer.ptr;
        let first_query_rope_ptr = cache.attention_query_rope.buffer.ptr;
        let first_suffix_k_nope_ptr = cache.attention_suffix_k_nope.buffer.ptr;
        let first_suffix_k_rope_ptr = cache.attention_suffix_k_rope.buffer.ptr;
        let first_suffix_values_ptr = cache.attention_suffix_values.buffer.ptr;
        let first_combined_k_nope_ptr = cache.attention_combined_k_nope.buffer.ptr;
        let first_combined_k_rope_ptr = cache.attention_combined_k_rope.buffer.ptr;
        let first_combined_values_ptr = cache.attention_combined_values.buffer.ptr;
        let first_combined_k_nope_capacity = cache.attention_combined_k_nope.capacity;
        let first_combined_k_rope_capacity = cache.attention_combined_k_rope.capacity;
        let first_combined_values_capacity = cache.attention_combined_values.capacity;
        let first_rope_positions_ptr = cache.rope_positions_device.buffer.ptr;
        let first_rope_positions_capacity = cache.rope_positions_device.capacity;
        assert!(!first_attention_staging_ptr.is_null());
        assert!(first_attention_staging_capacity >= q_rope_bf16.len());
        assert!(!first_query_nope_ptr.is_null());
        assert!(!first_query_rope_ptr.is_null());
        assert!(!first_suffix_k_nope_ptr.is_null());
        assert!(!first_suffix_k_rope_ptr.is_null());
        assert!(!first_suffix_values_ptr.is_null());
        assert!(!first_combined_k_nope_ptr.is_null());
        assert!(!first_combined_k_rope_ptr.is_null());
        assert!(!first_combined_values_ptr.is_null());
        assert!(first_combined_k_nope_capacity >= suffix_k_nope_bf16.len());
        assert!(first_combined_k_rope_capacity >= suffix_k_rope_rotated_bf16.len());
        assert!(first_combined_values_capacity >= suffix_values_bf16.len());
        assert!(!first_rope_positions_ptr.is_null());
        assert!(first_rope_positions_capacity >= std::mem::size_of::<u32>());

        let second_device_weight_readback = device_weight_mirror
            .run_mla_rope_attention_from_device_prefix_with_device_weights_and_host_suffix_bf16(
                std::slice::from_ref(&descriptor),
                &[0_u32],
                norm_weight.buffer,
                kv_b_weight.buffer,
                &q_nope_bf16,
                &q_rope_bf16,
                &suffix_k_nope_bf16,
                &suffix_k_rope_rotated_bf16,
                &suffix_values_bf16,
                heads,
                nope_dim,
                v_dim,
                1.0e-5,
                10_000.0,
                0.25,
            )?
            .expect(
                "CUDA-enabled mirror should return second resident-weight prefix attention readback",
            );
        assert_eq!(
            second_device_weight_readback.output_bf16.len(),
            total_rows * heads * v_dim * std::mem::size_of::<u16>()
        );
        let cache = device_weight_mirror
            .cache
            .as_ref()
            .expect("CUDA mirror should own cache");
        assert_eq!(
            cache.attention_host_upload_staging.buffer.ptr,
            first_attention_staging_ptr
        );
        assert_eq!(
            cache.attention_host_upload_staging.capacity,
            first_attention_staging_capacity
        );
        assert_eq!(cache.attention_query_nope.buffer.ptr, first_query_nope_ptr);
        assert_eq!(cache.attention_query_rope.buffer.ptr, first_query_rope_ptr);
        assert_eq!(
            cache.attention_suffix_k_nope.buffer.ptr,
            first_suffix_k_nope_ptr
        );
        assert_eq!(
            cache.attention_suffix_k_rope.buffer.ptr,
            first_suffix_k_rope_ptr
        );
        assert_eq!(
            cache.attention_suffix_values.buffer.ptr,
            first_suffix_values_ptr
        );
        assert_eq!(
            cache.attention_combined_k_nope.buffer.ptr,
            first_combined_k_nope_ptr
        );
        assert_eq!(
            cache.attention_combined_k_rope.buffer.ptr,
            first_combined_k_rope_ptr
        );
        assert_eq!(
            cache.attention_combined_values.buffer.ptr,
            first_combined_values_ptr
        );
        assert_eq!(
            cache.attention_combined_k_nope.capacity,
            first_combined_k_nope_capacity
        );
        assert_eq!(
            cache.attention_combined_k_rope.capacity,
            first_combined_k_rope_capacity
        );
        assert_eq!(
            cache.attention_combined_values.capacity,
            first_combined_values_capacity
        );
        assert_eq!(
            cache.rope_positions_device.buffer.ptr,
            first_rope_positions_ptr
        );
        assert_eq!(
            cache.rope_positions_device.capacity,
            first_rope_positions_capacity
        );
        Ok(())
    }

    #[test]
    fn scheduler_mla_attention_hidden_projection_accounts_without_host_payload() -> Result<()> {
        if !coordinator_cuda_reference_kernels_enabled() {
            return Ok(());
        }
        let config = KvCacheConfig::glm52_phase0(8);
        let descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-scheduler-attention-summary".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(0),
            token_count: 1,
        };
        let latent = (0..GLM52_MLA_KV_LORA_RANK)
            .map(|index| ((index % 31) as f32 - 15.0) / 64.0)
            .collect::<Vec<_>>();
        let rope = (0..GLM52_MLA_QK_ROPE_HEAD_DIM)
            .map(|index| ((index % 23) as f32 - 11.0) / 96.0)
            .collect::<Vec<_>>();
        let mut payload = bf16_bytes_from_f32(&latent);
        payload.extend_from_slice(&bf16_bytes_from_f32(&rope));
        assert_eq!(
            payload.len(),
            config.descriptor_payload_bytes(&descriptor).unwrap()
        );

        let library = cuda_native_library()?;
        let hidden = (0..GLM52_HIDDEN_SIZE)
            .map(|index| ((index % 37) as f32 - 18.0) / 128.0)
            .collect::<Vec<_>>();
        let hidden_bf16 = bf16_bytes_from_f32(&hidden);
        let hidden_device = upload_device_bytes(library, &hidden_bf16)?;
        let mut mirror = RealFullDeviceKvExecutionMirror::new(config)?;
        if !mirror.summary().uses_device_kv_cache {
            return Ok(());
        }

        mirror.write_host_blocks(
            std::slice::from_ref(&descriptor),
            std::slice::from_ref(&payload),
        )?;
        let launch = mirror
            .run_scheduler_mla_attention_from_device_kv_bf16(
                std::slice::from_ref(&descriptor),
                &[0_u32],
                Some(hidden_device.buffer),
            )?
            .context("CUDA-enabled mirror should launch scheduler attention")?;

        assert_eq!(
            launch.status,
            "cuda-kv-cache-mla-rope-attention-hidden-projection-device-buffer"
        );
        assert_eq!(launch.output_rows, 1);
        assert_eq!(launch.query_rows, 1);
        assert_eq!(launch.output_values, GLM52_HIDDEN_SIZE);
        assert_eq!(launch.output_finite_values, 0);
        assert_eq!(launch.output_nonzero_values, 0);
        assert_eq!(launch.output_checksum, 0.0);
        assert!(launch.output_bf16.is_none());
        assert_eq!(launch.output_device.rows, 1);
        assert_eq!(launch.output_device.values_per_row, GLM52_HIDDEN_SIZE);
        assert_eq!(
            launch.output_device.copy_to_host_bytes()?.len(),
            GLM52_HIDDEN_SIZE * std::mem::size_of::<u16>()
        );
        Ok(())
    }

    #[test]
    fn scheduler_mla_attention_reuses_projected_query_buffer_when_hidden_is_resident() -> Result<()>
    {
        if !coordinator_cuda_reference_kernels_enabled() {
            return Ok(());
        }
        let config = KvCacheConfig::glm52_phase0(8);
        let descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-scheduler-query-reuse".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(0),
            token_count: 1,
        };
        let latent = (0..GLM52_MLA_KV_LORA_RANK)
            .map(|index| ((index % 31) as f32 - 15.0) / 80.0)
            .collect::<Vec<_>>();
        let rope = (0..GLM52_MLA_QK_ROPE_HEAD_DIM)
            .map(|index| ((index % 23) as f32 - 11.0) / 112.0)
            .collect::<Vec<_>>();
        let mut payload = bf16_bytes_from_f32(&latent);
        payload.extend_from_slice(&bf16_bytes_from_f32(&rope));
        assert_eq!(
            payload.len(),
            config.descriptor_payload_bytes(&descriptor).unwrap()
        );

        let library = cuda_native_library()?;
        let hidden = (0..GLM52_HIDDEN_SIZE)
            .map(|index| ((index % 41) as f32 - 20.0) / 160.0)
            .collect::<Vec<_>>();
        let hidden_bf16 = bf16_bytes_from_f32(&hidden);
        let hidden_device = upload_device_bytes(library, &hidden_bf16)?;
        let mut mirror = RealFullDeviceKvExecutionMirror::new(config)?;
        if !mirror.summary().uses_device_kv_cache {
            return Ok(());
        }

        mirror.write_host_blocks(
            std::slice::from_ref(&descriptor),
            std::slice::from_ref(&payload),
        )?;
        let first_launch = mirror
            .run_scheduler_mla_attention_from_device_kv_bf16(
                std::slice::from_ref(&descriptor),
                &[0_u32],
                Some(hidden_device.buffer),
            )?
            .context(
                "CUDA-enabled mirror should launch first resident-hidden scheduler attention",
            )?;
        assert!(first_launch.output_bf16.is_none());
        let cache = mirror.cache.as_ref().expect("CUDA mirror should own cache");
        let first_query_ptr = cache.scheduler_projected_query.buffer.ptr;
        let first_query_capacity = cache.scheduler_projected_query.capacity;
        assert!(!first_query_ptr.is_null());
        assert!(first_query_capacity >= scheduler_attention_projected_query_bytes(1)?);

        let second_launch = mirror
            .run_scheduler_mla_attention_from_device_kv_bf16(
                std::slice::from_ref(&descriptor),
                &[0_u32],
                Some(hidden_device.buffer),
            )?
            .context(
                "CUDA-enabled mirror should launch second resident-hidden scheduler attention",
            )?;
        assert!(second_launch.output_bf16.is_none());
        let cache = mirror.cache.as_ref().expect("CUDA mirror should own cache");
        assert_eq!(cache.scheduler_projected_query.buffer.ptr, first_query_ptr);
        assert_eq!(
            cache.scheduler_projected_query.capacity,
            first_query_capacity
        );
        Ok(())
    }

    #[test]
    fn scheduler_attention_static_projected_query_upload_reuses_bf16_scratch() -> Result<()> {
        let config = KvCacheConfig::glm52_phase0(8);
        let mut mirror = RealFullDeviceKvExecutionMirror::new(config)?;
        if !mirror.summary().uses_device_kv_cache {
            return Ok(());
        }
        let two_row_query = mirror.scheduler_attention_static_projected_query_buffer(2)?;
        assert!(!two_row_query.ptr.is_null());
        assert_eq!(mirror.scheduler_attention_resident_uploads, 1);
        let two_row_bytes = scheduler_attention_projected_query_bytes(2)?;
        assert_eq!(
            mirror
                .scheduler_attention_projected_query_upload_bf16_scratch
                .len(),
            two_row_bytes
        );
        let first_scratch_ptr = mirror
            .scheduler_attention_projected_query_upload_bf16_scratch
            .as_ptr();
        let first_scratch_capacity = mirror
            .scheduler_attention_projected_query_upload_bf16_scratch
            .capacity();
        assert!(first_scratch_capacity >= two_row_bytes);
        let cache = mirror.cache.as_ref().expect("CUDA mirror should own cache");
        let first_staging_ptr = cache.scheduler_upload_staging.buffer.ptr;
        let first_staging_capacity = cache.scheduler_upload_staging.capacity;
        assert!(!first_staging_ptr.is_null());
        assert!(first_staging_capacity >= two_row_bytes);
        let generated = bf16_bytes_to_f32(
            &mirror.scheduler_attention_projected_query_upload_bf16_scratch
                [..scheduler_attention_projected_query_width()],
        )?;
        assert_eq!(generated[0], -14.0 / 128.0);
        assert_eq!(generated[14], 0.0);

        let one_row_query = mirror.scheduler_attention_static_projected_query_buffer(1)?;
        assert!(!one_row_query.ptr.is_null());
        assert_eq!(mirror.scheduler_attention_resident_uploads, 2);
        assert_eq!(
            mirror
                .scheduler_attention_projected_query_upload_bf16_scratch
                .as_ptr(),
            first_scratch_ptr
        );
        assert_eq!(
            mirror
                .scheduler_attention_projected_query_upload_bf16_scratch
                .capacity(),
            first_scratch_capacity
        );
        assert_eq!(
            mirror
                .scheduler_attention_projected_query_upload_bf16_scratch
                .len(),
            scheduler_attention_projected_query_bytes(1)?
        );
        let cache = mirror.cache.as_ref().expect("CUDA mirror should own cache");
        assert_eq!(cache.scheduler_upload_staging.buffer.ptr, first_staging_ptr);
        assert_eq!(
            cache.scheduler_upload_staging.capacity,
            first_staging_capacity
        );

        let two_row_query_again = mirror.scheduler_attention_static_projected_query_buffer(2)?;
        assert_eq!(two_row_query_again.ptr, two_row_query.ptr);
        assert_eq!(mirror.scheduler_attention_resident_uploads, 2);
        let cache = mirror.cache.as_ref().expect("CUDA mirror should own cache");
        assert_eq!(cache.scheduler_upload_staging.buffer.ptr, first_staging_ptr);
        assert_eq!(
            cache.scheduler_upload_staging.capacity,
            first_staging_capacity
        );
        Ok(())
    }

    #[test]
    fn scheduler_attention_resident_weight_upload_reuses_bf16_scratch() -> Result<()> {
        let config = KvCacheConfig::glm52_phase0(8);
        let mut mirror = RealFullDeviceKvExecutionMirror::new(config)?;
        if !mirror.summary().uses_device_kv_cache {
            return Ok(());
        }
        let buffers = mirror.scheduler_attention_resident_buffers(true, true)?;
        assert!(!buffers.kv_norm_weight.ptr.is_null());
        assert!(!buffers.kv_b_weight.ptr.is_null());
        assert!(!buffers
            .query_projection_weight
            .expect("query projection weight should be uploaded")
            .ptr
            .is_null());
        assert!(!buffers
            .output_projection_weight
            .expect("output projection weight should be uploaded")
            .ptr
            .is_null());
        assert_eq!(mirror.scheduler_attention_resident_uploads, 4);

        let first_scratch_ptr = mirror
            .scheduler_attention_weight_upload_bf16_scratch
            .as_ptr();
        let first_scratch_capacity = mirror
            .scheduler_attention_weight_upload_bf16_scratch
            .capacity();
        let query_projection_bytes = scheduler_attention_projected_query_width()
            * GLM52_HIDDEN_SIZE
            * std::mem::size_of::<u16>();
        assert!(first_scratch_capacity >= query_projection_bytes);
        let cache = mirror.cache.as_ref().expect("CUDA mirror should own cache");
        let first_staging_ptr = cache.scheduler_upload_staging.buffer.ptr;
        let first_staging_capacity = cache.scheduler_upload_staging.capacity;
        assert!(!first_staging_ptr.is_null());
        assert!(first_staging_capacity >= query_projection_bytes);

        let _ = mirror.scheduler_attention_resident_buffers(true, true)?;
        assert_eq!(mirror.scheduler_attention_resident_uploads, 4);
        assert_eq!(
            mirror
                .scheduler_attention_weight_upload_bf16_scratch
                .as_ptr(),
            first_scratch_ptr
        );
        assert_eq!(
            mirror
                .scheduler_attention_weight_upload_bf16_scratch
                .capacity(),
            first_scratch_capacity
        );
        let cache = mirror.cache.as_ref().expect("CUDA mirror should own cache");
        assert_eq!(cache.scheduler_upload_staging.buffer.ptr, first_staging_ptr);
        assert_eq!(
            cache.scheduler_upload_staging.capacity,
            first_staging_capacity
        );
        Ok(())
    }

    #[test]
    fn scheduler_mla_attention_reuses_descriptor_position_scratch_when_available() -> Result<()> {
        let config = KvCacheConfig::glm52_phase0(8);
        let prefix_descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-scheduler-descriptor-scratch".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(0),
            token_count: 1,
        };
        let current_descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-scheduler-descriptor-scratch".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(1),
            token_count: 1,
        };
        let (prefix_payload, _, _, _) = mla_kv_payload_rows(1, false);
        let (current_payload, _, _, _) = mla_kv_payload_rows(1, false);
        let mut mirror = RealFullDeviceKvExecutionMirror::new(config)?;
        if !mirror.summary().uses_device_kv_cache {
            return Ok(());
        }

        let write_descriptors = vec![prefix_descriptor.clone(), current_descriptor.clone()];
        let write_payloads = vec![prefix_payload.clone(), current_payload.clone()];
        mirror.write_host_blocks(&write_descriptors, &write_payloads)?;
        let visible_blocks = vec![KvBackedBlock {
            write_id: 1,
            descriptor: prefix_descriptor,
            state: KvWriteState::Committed,
            bytes: prefix_payload,
        }];
        let current_descriptors = [current_descriptor];

        let first_launch = mirror
            .run_scheduler_mla_attention_from_device_kv_descriptor_sets_bf16(
                &visible_blocks,
                &current_descriptors,
                None,
            )?
            .context("CUDA-enabled mirror should launch first descriptor-set attention")?;
        assert_eq!(first_launch.descriptors, 2);
        assert_eq!(first_launch.query_rows, 1);
        let first_descriptor_ptr = mirror.scheduler_attention_descriptors.as_ptr();
        let first_descriptor_capacity = mirror.scheduler_attention_descriptors.capacity();
        let first_positions_ptr = mirror.scheduler_attention_positions.as_ptr();
        let first_positions_capacity = mirror.scheduler_attention_positions.capacity();
        let first_query_positions_ptr = mirror.scheduler_attention_query_positions.as_ptr();
        let first_query_positions_capacity = mirror.scheduler_attention_query_positions.capacity();
        assert!(first_descriptor_capacity >= 2);
        assert!(first_positions_capacity >= 2);
        assert!(first_query_positions_capacity >= 1);

        let second_launch = mirror
            .run_scheduler_mla_attention_from_device_kv_descriptor_sets_bf16(
                &visible_blocks,
                &current_descriptors,
                None,
            )?
            .context("CUDA-enabled mirror should launch second descriptor-set attention")?;
        assert_eq!(second_launch.descriptors, 2);
        assert_eq!(second_launch.query_rows, 1);
        assert_eq!(
            mirror.scheduler_attention_descriptors.as_ptr(),
            first_descriptor_ptr
        );
        assert_eq!(
            mirror.scheduler_attention_descriptors.capacity(),
            first_descriptor_capacity
        );
        assert_eq!(
            mirror.scheduler_attention_positions.as_ptr(),
            first_positions_ptr
        );
        assert_eq!(
            mirror.scheduler_attention_positions.capacity(),
            first_positions_capacity
        );
        assert_eq!(
            mirror.scheduler_attention_query_positions.as_ptr(),
            first_query_positions_ptr
        );
        assert_eq!(
            mirror.scheduler_attention_query_positions.capacity(),
            first_query_positions_capacity
        );
        Ok(())
    }

    #[test]
    fn scheduler_mla_attention_reuses_rope_position_buffer_when_available() -> Result<()> {
        let config = KvCacheConfig::glm52_phase0(8);
        let first_descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-scheduler-position-reuse".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(0),
            token_count: 2,
        };
        let second_descriptor = KvBlockDescriptor {
            reservation_id: 1,
            sequence_id: "seq-scheduler-position-reuse".to_owned(),
            layer_id: LayerId(3),
            token_start: PositionId(1),
            token_count: 1,
        };
        let mut payload = Vec::new();
        for row in 0..first_descriptor.token_count {
            let latent = (0..GLM52_MLA_KV_LORA_RANK)
                .map(|index| ((row * 13 + index % 31) as f32 - 15.0) / 96.0)
                .collect::<Vec<_>>();
            let rope = (0..GLM52_MLA_QK_ROPE_HEAD_DIM)
                .map(|index| ((row * 7 + index % 23) as f32 - 11.0) / 128.0)
                .collect::<Vec<_>>();
            payload.extend_from_slice(&bf16_bytes_from_f32(&latent));
            payload.extend_from_slice(&bf16_bytes_from_f32(&rope));
        }
        assert_eq!(
            payload.len(),
            config.descriptor_payload_bytes(&first_descriptor).unwrap()
        );

        let mut mirror = RealFullDeviceKvExecutionMirror::new(config)?;
        if !mirror.summary().uses_device_kv_cache {
            return Ok(());
        }

        mirror.write_host_blocks(
            std::slice::from_ref(&first_descriptor),
            std::slice::from_ref(&payload),
        )?;
        let first_launch = mirror
            .run_scheduler_mla_attention_from_device_kv_bf16(
                std::slice::from_ref(&first_descriptor),
                &[1_u32],
                None,
            )?
            .context("CUDA-enabled mirror should launch first scheduler attention")?;
        assert_eq!(first_launch.query_rows, 1);
        assert_eq!(first_launch.rows, 2);
        let cache = mirror.cache.as_ref().expect("CUDA mirror should own cache");
        let first_position_ptr = cache.rope_positions_device.buffer.ptr;
        let first_position_capacity = cache.rope_positions_device.capacity;
        assert!(!first_position_ptr.is_null());
        assert!(first_position_capacity >= std::mem::size_of::<u32>() * 2);

        let second_launch = mirror
            .run_scheduler_mla_attention_from_device_kv_bf16(
                std::slice::from_ref(&second_descriptor),
                &[1_u32],
                None,
            )?
            .context("CUDA-enabled mirror should launch second scheduler attention")?;
        assert_eq!(second_launch.query_rows, 1);
        assert_eq!(second_launch.rows, 1);
        let cache = mirror.cache.as_ref().expect("CUDA mirror should own cache");
        assert_eq!(cache.rope_positions_device.buffer.ptr, first_position_ptr);
        assert_eq!(
            cache.rope_positions_device.capacity,
            first_position_capacity
        );
        Ok(())
    }

    #[test]
    fn device_kv_roundtrip_status_labels_cuda_unavailable_errors() {
        let error = anyhow::anyhow!(
            "allocating real-full device KV roundtrip cache: glmrt_alloc_device_buffer returned status 3: CUDA unavailable"
        );

        assert_eq!(
            real_full_device_kv_roundtrip_error_status(&error),
            "cuda-kv-cache-unavailable"
        );
    }

    #[test]
    fn device_kv_roundtrip_cuda_errors_are_soft_only_when_cuda_is_not_required() {
        let soft = device_kv_roundtrip_failed(
            anyhow::anyhow!("glmrt_cuda_kv_cache_write_blocks returned status 3: CUDA unavailable"),
            false,
        )
        .expect("soft CUDA miss should report status when CUDA is optional");

        assert_eq!(soft.status, "cuda-kv-cache-unavailable");
        assert!(!soft.uses_device_kv_cache);

        let hard = device_kv_roundtrip_failed(
            anyhow::anyhow!("glmrt_cuda_kv_cache_write_blocks returned status 7"),
            true,
        )
        .unwrap_err()
        .to_string();

        assert!(hard.contains("CUDA reference execution enabled"));
    }
}
