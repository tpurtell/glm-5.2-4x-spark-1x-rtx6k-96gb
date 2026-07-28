use anyhow::{Context, Result};
use glmrt_core::{DType, TensorCatalog, TensorInfo, TensorRole, GLM52_NUM_HIDDEN_LAYERS};
use glmrt_ffi::{GlmrtDeviceBuffer, GlmrtNcclComm, NativeLibrary};
use glmrt_loader::{read_tensor_bytes_into, read_tensor_row_window_into, read_tensor_rows_into};
use std::{env, ffi::c_void, path::PathBuf, sync::Arc, time::Instant};

use super::coordinator_kernels::{
    preload_resident_weight_from_host_staging, require_cuda_enabled_native_library,
};
use super::intermediate_sharding::{initialize_spark_nccl_communicator, ExpertIntermediateShard};
use super::sparse_mlp::cache_router_correction_bias_host_values;

pub(crate) const SPARK_TRANSFORMER_TP_ENV: &str = "GLMRT_SPARK_TRANSFORMER_TP";
pub(crate) const SPARK_TRANSFORMER_TP_RANGE_ENV: &str = "GLMRT_SPARK_TRANSFORMER_TP_RANGE";
pub(crate) const SPARK_TRANSFORMER_TP_ROOT_ENV: &str = "GLMRT_SPARK_TRANSFORMER_TP_ROOT";
pub(crate) const SPARK_TRANSFORMER_TP_PORT_ENV: &str = "GLMRT_SPARK_TRANSFORMER_TP_PORT";
pub(crate) const SPARK_TRANSFORMER_TP_COLLECTIVE_PROBE_ITERS_ENV: &str =
    "GLMRT_SPARK_TRANSFORMER_TP_COLLECTIVE_PROBE_ITERS";
const DEFAULT_SPARK_TRANSFORMER_TP_ROOT: &str = "ostrich.200gb";
const DEFAULT_SPARK_TRANSFORMER_TP_PORT: u16 = 9300;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SparkTransformerTp {
    pub(crate) shard: ExpertIntermediateShard,
    pub(crate) start_layer: usize,
    pub(crate) end_layer: usize,
}

impl SparkTransformerTp {
    pub(crate) fn new(
        shard: ExpertIntermediateShard,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            start_layer < end_layer && end_layer <= GLM52_NUM_HIDDEN_LAYERS,
            "Spark transformer TP layer range {start_layer}:{end_layer} must be within 0:{GLM52_NUM_HIDDEN_LAYERS}"
        );
        Ok(Self {
            shard,
            start_layer,
            end_layer,
        })
    }

    pub(crate) fn contains_layer(self, layer_id: usize) -> bool {
        (self.start_layer..self.end_layer).contains(&layer_id)
    }

    pub(crate) fn layer_count(self) -> usize {
        self.end_layer - self.start_layer
    }

    pub(crate) fn local_attention_heads(self) -> usize {
        64 / self.shard.count
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SparkTransformerTpTensorLayout {
    Replicated,
    Rows { start: usize, count: usize },
    Columns { start: usize, count: usize },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SparkTransformerTpTensorSpec {
    pub(crate) source_name: String,
    pub(crate) resident_name: String,
    pub(crate) layout: SparkTransformerTpTensorLayout,
    pub(crate) rows: usize,
    pub(crate) columns: usize,
    pub(crate) bytes: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct SparkTransformerTpResidentPreloadStats {
    pub(crate) layers: usize,
    pub(crate) tensors: usize,
    pub(crate) replicated_tensors: usize,
    pub(crate) row_shards: usize,
    pub(crate) column_shards: usize,
    pub(crate) bytes: u64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SparkTransformerTpCollectiveProbe {
    pub(crate) iterations: usize,
    pub(crate) values: usize,
    pub(crate) bytes: usize,
    pub(crate) total_ms: f64,
    pub(crate) mean_ms: f64,
}

pub(crate) struct SparkTransformerTpCollective {
    library: Arc<NativeLibrary>,
    communicator: GlmrtNcclComm,
    stream: *mut c_void,
}

impl SparkTransformerTpCollective {
    pub(crate) fn from_env(config: SparkTransformerTp) -> Result<Self> {
        let path = transformer_tp_native_library_path()?;
        let library = unsafe { NativeLibrary::load(&path) }
            .with_context(|| format!("loading transformer TP native library {}", path.display()))?;
        require_cuda_enabled_native_library(&library, &path, "Spark transformer TP collective")?;
        let library = Arc::new(library);
        let root = env::var(SPARK_TRANSFORMER_TP_ROOT_ENV)
            .unwrap_or_else(|_| DEFAULT_SPARK_TRANSFORMER_TP_ROOT.to_owned());
        let port = parse_port(
            SPARK_TRANSFORMER_TP_PORT_ENV,
            env::var(SPARK_TRANSFORMER_TP_PORT_ENV).ok().as_deref(),
            DEFAULT_SPARK_TRANSFORMER_TP_PORT,
        )?;
        let communicator = initialize_spark_nccl_communicator(
            &library,
            config.shard,
            &root,
            port,
            "transformer TP",
        )?;
        let stream = library
            .cuda_stream_create()
            .context("creating Spark transformer TP collective stream")?;
        Ok(Self {
            library,
            communicator,
            stream,
        })
    }

    pub(crate) fn all_reduce_bf16_in_place(
        &self,
        buffer: GlmrtDeviceBuffer,
        values: usize,
    ) -> Result<()> {
        unsafe {
            self.communicator
                .all_reduce_bf16_async(buffer, buffer, values, self.stream)
                .context("launching Spark transformer TP BF16 all-reduce")?;
        }
        Ok(())
    }

    pub(crate) fn synchronize(&self) -> Result<()> {
        unsafe {
            self.library
                .cuda_stream_synchronize(self.stream)
                .context("synchronizing Spark transformer TP collective stream")
        }
    }

    pub(crate) fn rank(&self) -> usize {
        self.communicator.rank()
    }
}

impl Drop for SparkTransformerTpCollective {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            let _ = unsafe { self.library.cuda_stream_destroy(self.stream) };
            self.stream = std::ptr::null_mut();
        }
    }
}

pub(crate) fn spark_transformer_tp_from_env(
    shard: Option<ExpertIntermediateShard>,
) -> Result<Option<SparkTransformerTp>> {
    let enabled = env::var(SPARK_TRANSFORMER_TP_ENV)
        .ok()
        .map(|raw| parse_enabled(SPARK_TRANSFORMER_TP_ENV, &raw))
        .transpose()?
        .unwrap_or(false);
    let explicit_range = env::var(SPARK_TRANSFORMER_TP_RANGE_ENV).ok();
    spark_transformer_tp_from_config(enabled, explicit_range.as_deref(), shard)
}

fn spark_transformer_tp_from_config(
    enabled: bool,
    explicit_range: Option<&str>,
    shard: Option<ExpertIntermediateShard>,
) -> Result<Option<SparkTransformerTp>> {
    if !enabled {
        return Ok(None);
    }
    let shard = shard.with_context(|| {
        format!("{SPARK_TRANSFORMER_TP_ENV}=1 requires four-way intermediate sharding")
    })?;
    let (start_layer, end_layer) = explicit_range
        .map(parse_layer_range)
        .transpose()?
        .unwrap_or((0, GLM52_NUM_HIDDEN_LAYERS));
    SparkTransformerTp::new(shard, start_layer, end_layer).map(Some)
}

pub(crate) fn tensor_is_spark_transformer_tp_resident(
    tensor: &TensorInfo,
    config: SparkTransformerTp,
) -> bool {
    tensor
        .layer_id
        .map(|layer_id| config.contains_layer(layer_id as usize))
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

pub(crate) fn spark_transformer_tp_resident_name(
    source_name: &str,
    shard: ExpertIntermediateShard,
) -> String {
    format!("{source_name}[transformer-tp4-rank={}]", shard.rank)
}

pub(crate) fn spark_transformer_tp_tensor_specs(
    catalog: &TensorCatalog,
    config: SparkTransformerTp,
) -> Result<Vec<SparkTransformerTpTensorSpec>> {
    catalog
        .tensors
        .iter()
        .filter(|tensor| tensor_is_spark_transformer_tp_resident(tensor, config))
        .map(|tensor| spark_transformer_tp_tensor_spec(tensor, config))
        .collect()
}

pub(crate) fn preload_real_full_spark_transformer_tp_weights(
    catalog: &TensorCatalog,
    config: SparkTransformerTp,
) -> Result<SparkTransformerTpResidentPreloadStats> {
    let specs = spark_transformer_tp_tensor_specs(catalog, config)?;
    anyhow::ensure!(
        !specs.is_empty(),
        "Spark transformer TP range {}:{} selected no resident tensors",
        config.start_layer,
        config.end_layer
    );
    let mut stats = SparkTransformerTpResidentPreloadStats {
        layers: config.layer_count(),
        ..Default::default()
    };
    for spec in &specs {
        let tensor = catalog
            .tensors
            .iter()
            .find(|tensor| tensor.name == spec.source_name)
            .with_context(|| format!("TP source tensor {} disappeared", spec.source_name))?;
        preload_resident_weight_from_host_staging(
            &spec.resident_name,
            spec.bytes,
            "startup resident Spark transformer TP weight pinned staging",
            |staging| {
                let summary = match spec.layout {
                    SparkTransformerTpTensorLayout::Replicated => {
                        let summary = read_tensor_bytes_into(catalog, &spec.source_name, staging)
                            .with_context(|| {
                            format!("reading replicated TP tensor {}", spec.source_name)
                        })?;
                        anyhow::ensure!(
                            summary.tensor_name == spec.source_name
                                && summary.dtype == tensor.dtype
                                && summary.bytes_read == spec.bytes as u64,
                            "replicated TP tensor {} staged unexpected source metadata",
                            spec.source_name
                        );
                        cache_router_correction_bias_host_values(
                            catalog,
                            tensor,
                            &staging[..spec.bytes],
                        )?;
                        return Ok(());
                    }
                    SparkTransformerTpTensorLayout::Rows { start, count } => {
                        read_tensor_rows_into(catalog, &spec.source_name, start, count, staging)
                            .with_context(|| {
                                format!("reading row-sharded TP tensor {}", spec.source_name)
                            })?
                    }
                    SparkTransformerTpTensorLayout::Columns { start, count } => {
                        read_tensor_row_window_into(
                            catalog,
                            &spec.source_name,
                            0,
                            spec.rows,
                            start,
                            count,
                            staging,
                        )
                        .with_context(|| {
                            format!("reading column-sharded TP tensor {}", spec.source_name)
                        })?
                    }
                };
                anyhow::ensure!(
                    summary.tensor_name == spec.source_name
                        && summary.dtype == DType::Bf16
                        && summary.bytes_read == spec.bytes as u64,
                    "sharded TP tensor {} staged unexpected source metadata",
                    spec.source_name
                );
                Ok(())
            },
        )
        .with_context(|| {
            format!(
                "preloading Spark transformer TP tensor {}",
                spec.source_name
            )
        })?;
        stats.tensors += 1;
        stats.bytes = stats
            .bytes
            .checked_add(spec.bytes as u64)
            .context("Spark transformer TP resident byte count overflow")?;
        match spec.layout {
            SparkTransformerTpTensorLayout::Replicated => stats.replicated_tensors += 1,
            SparkTransformerTpTensorLayout::Rows { .. } => stats.row_shards += 1,
            SparkTransformerTpTensorLayout::Columns { .. } => stats.column_shards += 1,
        }
    }
    Ok(stats)
}

pub(crate) fn probe_spark_transformer_tp_collective_from_env(
    config: SparkTransformerTp,
) -> Result<Option<SparkTransformerTpCollectiveProbe>> {
    let iterations = parse_nonnegative(
        SPARK_TRANSFORMER_TP_COLLECTIVE_PROBE_ITERS_ENV,
        env::var(SPARK_TRANSFORMER_TP_COLLECTIVE_PROBE_ITERS_ENV)
            .ok()
            .as_deref(),
        0,
    )?;
    if iterations == 0 {
        return Ok(None);
    }
    let collective = SparkTransformerTpCollective::from_env(config)?;
    let values = glmrt_core::GLM52_HIDDEN_SIZE;
    let bytes = values
        .checked_mul(std::mem::size_of::<u16>())
        .context("transformer TP collective probe byte count overflow")?;
    let mut buffer = collective
        .library
        .alloc_device_buffer(bytes)
        .context("allocating transformer TP collective probe buffer")?;
    let result = (|| -> Result<SparkTransformerTpCollectiveProbe> {
        unsafe {
            collective
                .library
                .cuda_zero_bytes_async(buffer, bytes, collective.stream)
                .context("zeroing transformer TP collective probe buffer")?;
        }
        for _ in 0..10 {
            collective.all_reduce_bf16_in_place(buffer, values)?;
        }
        collective.synchronize()?;
        let started = Instant::now();
        for _ in 0..iterations {
            collective.all_reduce_bf16_in_place(buffer, values)?;
        }
        collective.synchronize()?;
        let total_ms = started.elapsed().as_secs_f64() * 1_000.0;
        Ok(SparkTransformerTpCollectiveProbe {
            iterations,
            values,
            bytes,
            total_ms,
            mean_ms: total_ms / iterations as f64,
        })
    })();
    collective
        .library
        .free_device_buffer(&mut buffer)
        .context("freeing transformer TP collective probe buffer")?;
    result.map(Some)
}

fn spark_transformer_tp_tensor_spec(
    tensor: &TensorInfo,
    config: SparkTransformerTp,
) -> Result<SparkTransformerTpTensorSpec> {
    let layout = tensor_layout(tensor, config.shard)?;
    anyhow::ensure!(
        matches!(layout, SparkTransformerTpTensorLayout::Replicated) || tensor.dtype == DType::Bf16,
        "sharded Spark transformer TP tensor {} must be BF16, got {:?}",
        tensor.name,
        tensor.dtype
    );
    let (rows, columns) = local_shape(tensor, layout)?;
    let source_bytes: usize = tensor
        .byte_length
        .try_into()
        .with_context(|| format!("tensor {} byte length exceeds usize", tensor.name))?;
    let source_values = tensor
        .shape
        .iter()
        .try_fold(1_usize, |acc, dim| acc.checked_mul(*dim))
        .context("Spark transformer TP source shape value count overflow")?;
    let bytes_per_scalar = source_bytes
        .checked_div(source_values)
        .context("Spark transformer TP source tensor has zero values")?;
    anyhow::ensure!(
        source_values * bytes_per_scalar == source_bytes,
        "Spark transformer TP tensor {} shape {:?} does not match {} bytes",
        tensor.name,
        tensor.shape,
        source_bytes
    );
    let bytes = rows
        .checked_mul(columns)
        .and_then(|values| values.checked_mul(bytes_per_scalar))
        .context("Spark transformer TP tensor byte count overflow")?;
    let resident_name = if matches!(layout, SparkTransformerTpTensorLayout::Replicated) {
        tensor.name.clone()
    } else {
        spark_transformer_tp_resident_name(&tensor.name, config.shard)
    };
    Ok(SparkTransformerTpTensorSpec {
        source_name: tensor.name.clone(),
        resident_name,
        layout,
        rows,
        columns,
        bytes,
    })
}

fn tensor_layout(
    tensor: &TensorInfo,
    shard: ExpertIntermediateShard,
) -> Result<SparkTransformerTpTensorLayout> {
    let row_sharded = tensor.name.ends_with("self_attn.q_b_proj.weight")
        || tensor.name.ends_with("self_attn.kv_b_proj.weight")
        || tensor.name.ends_with("mlp.gate_proj.weight")
        || tensor.name.ends_with("mlp.up_proj.weight")
        || tensor.name.ends_with("mlp.shared_experts.gate_proj.weight")
        || tensor.name.ends_with("mlp.shared_experts.up_proj.weight");
    let column_sharded = tensor.name.ends_with("self_attn.o_proj.weight")
        || tensor.name.ends_with("mlp.down_proj.weight")
        || tensor.name.ends_with("mlp.shared_experts.down_proj.weight");
    if !row_sharded && !column_sharded {
        return Ok(SparkTransformerTpTensorLayout::Replicated);
    }
    anyhow::ensure!(
        tensor.shape.len() == 2,
        "sharded Spark transformer TP tensor {} must be a matrix, got {:?}",
        tensor.name,
        tensor.shape
    );
    if row_sharded {
        let count = shard.local_rows(tensor.shape[0])?;
        return Ok(SparkTransformerTpTensorLayout::Rows {
            start: shard.row_start(tensor.shape[0])?,
            count,
        });
    }
    let full_columns = tensor.shape[1];
    let count = shard.local_rows(full_columns)?;
    Ok(SparkTransformerTpTensorLayout::Columns {
        start: shard.row_start(full_columns)?,
        count,
    })
}

fn local_shape(
    tensor: &TensorInfo,
    layout: SparkTransformerTpTensorLayout,
) -> Result<(usize, usize)> {
    match layout {
        SparkTransformerTpTensorLayout::Replicated => match tensor.shape.as_slice() {
            [values] => Ok((1, *values)),
            [rows, columns] => Ok((*rows, *columns)),
            shape => anyhow::bail!(
                "replicated Spark transformer TP tensor {} has unsupported shape {:?}",
                tensor.name,
                shape
            ),
        },
        SparkTransformerTpTensorLayout::Rows { count, .. } => Ok((count, tensor.shape[1])),
        SparkTransformerTpTensorLayout::Columns { count, .. } => Ok((tensor.shape[0], count)),
    }
}

fn parse_enabled(name: &str, raw: &str) -> Result<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" | "" => Ok(false),
        value => anyhow::bail!("{name} must be boolean-like, got {value}"),
    }
}

fn parse_layer_range(raw: &str) -> Result<(usize, usize)> {
    let (start, end) = raw.trim().split_once(':').with_context(|| {
        format!("{SPARK_TRANSFORMER_TP_RANGE_ENV} must use start:end, got {raw}")
    })?;
    Ok((
        start
            .parse()
            .with_context(|| format!("parsing {SPARK_TRANSFORMER_TP_RANGE_ENV} start"))?,
        end.parse()
            .with_context(|| format!("parsing {SPARK_TRANSFORMER_TP_RANGE_ENV} end"))?,
    ))
}

fn parse_port(name: &str, raw: Option<&str>, default: u16) -> Result<u16> {
    let value = parse_nonnegative(name, raw, default as usize)?;
    anyhow::ensure!(value > 0, "{name} must be positive");
    value
        .try_into()
        .with_context(|| format!("{name} exceeds u16"))
}

fn parse_nonnegative(name: &str, raw: Option<&str>, default: usize) -> Result<usize> {
    raw.filter(|value| !value.trim().is_empty())
        .map(str::parse)
        .transpose()
        .with_context(|| format!("parsing {name}"))
        .map(|value| value.unwrap_or(default))
}

fn transformer_tp_native_library_path() -> Result<PathBuf> {
    if let Some(path) = env::var_os("GLMRT_NATIVE_LIB").map(PathBuf::from) {
        anyhow::ensure!(
            path.is_file(),
            "GLMRT_NATIVE_LIB {} is not a file",
            path.display()
        );
        return Ok(path);
    }
    [
        PathBuf::from("native/build-cuda-rdma/libglmrt_native.so"),
        PathBuf::from("../native/build-cuda-rdma/libglmrt_native.so"),
        PathBuf::from("native/build-cuda/libglmrt_native.so"),
        PathBuf::from("../native/build-cuda/libglmrt_native.so"),
    ]
    .into_iter()
    .find(|path| path.is_file())
    .context("Spark transformer TP requires GLMRT_NATIVE_LIB or a native CUDA build")
}

#[cfg(test)]
mod tests {
    use super::{
        spark_transformer_tp_from_config, spark_transformer_tp_tensor_specs, SparkTransformerTp,
        SparkTransformerTpTensorLayout,
    };
    use crate::commands::real_full::intermediate_sharding::ExpertIntermediateShard;
    use glmrt_core::{DType, ModelFacts, TensorCatalog, TensorInfo, TensorRole};

    #[test]
    fn disabled_transformer_tp_ignores_stale_range_configuration() {
        assert_eq!(
            spark_transformer_tp_from_config(false, Some("0:78"), None).unwrap(),
            None
        );
    }

    #[test]
    fn tp4_plan_replicates_shared_inputs_and_shards_projection_axes() {
        let shard = ExpertIntermediateShard::new(4, 1).unwrap();
        let config = SparkTransformerTp::new(shard, 0, 1).unwrap();
        let catalog = TensorCatalog {
            model_id: "test/tp".to_owned(),
            snapshot_path: "/tmp".to_owned(),
            facts: ModelFacts::default(),
            tensors: vec![
                tensor(
                    "model.layers.0.input_layernorm.weight",
                    &[8],
                    TensorRole::Norm,
                ),
                tensor(
                    "model.layers.0.self_attn.q_a_proj.weight",
                    &[8, 8],
                    TensorRole::Attention,
                ),
                tensor(
                    "model.layers.0.self_attn.q_b_proj.weight",
                    &[16, 8],
                    TensorRole::Attention,
                ),
                tensor(
                    "model.layers.0.self_attn.kv_b_proj.weight",
                    &[28, 8],
                    TensorRole::Attention,
                ),
                tensor(
                    "model.layers.0.self_attn.o_proj.weight",
                    &[8, 16],
                    TensorRole::Attention,
                ),
                tensor(
                    "model.layers.0.mlp.gate_proj.weight",
                    &[12, 8],
                    TensorRole::DenseMlp,
                ),
                tensor(
                    "model.layers.0.mlp.down_proj.weight",
                    &[8, 12],
                    TensorRole::DenseMlp,
                ),
                tensor(
                    "model.layers.0.mlp.experts.0.gate_proj.weight",
                    &[12, 8],
                    TensorRole::RoutedExpert,
                ),
            ],
        };

        let specs = spark_transformer_tp_tensor_specs(&catalog, config).unwrap();
        assert_eq!(specs.len(), 7);
        let find = |suffix: &str| {
            specs
                .iter()
                .find(|spec| spec.source_name.ends_with(suffix))
                .unwrap()
        };
        assert_eq!(
            find("q_a_proj.weight").layout,
            SparkTransformerTpTensorLayout::Replicated
        );
        assert_eq!(
            find("q_b_proj.weight").layout,
            SparkTransformerTpTensorLayout::Rows { start: 4, count: 4 }
        );
        assert_eq!(
            find("kv_b_proj.weight").layout,
            SparkTransformerTpTensorLayout::Rows { start: 7, count: 7 }
        );
        assert_eq!(
            find("o_proj.weight").layout,
            SparkTransformerTpTensorLayout::Columns { start: 4, count: 4 }
        );
        assert_eq!(
            find("mlp.gate_proj.weight").layout,
            SparkTransformerTpTensorLayout::Rows { start: 3, count: 3 }
        );
        assert_eq!(
            find("mlp.down_proj.weight").layout,
            SparkTransformerTpTensorLayout::Columns { start: 3, count: 3 }
        );
        assert!(find("q_b_proj.weight")
            .resident_name
            .ends_with("[transformer-tp4-rank=1]"));
        assert_eq!(config.local_attention_heads(), 16);
    }

    fn tensor(name: &str, shape: &[usize], role: TensorRole) -> TensorInfo {
        TensorInfo {
            name: name.to_owned(),
            file: "model.safetensors".to_owned(),
            dtype: DType::Bf16,
            shape: shape.to_vec(),
            byte_offset: 0,
            byte_length: (shape.iter().product::<usize>() * std::mem::size_of::<u16>()) as u64,
            role,
            layer_id: Some(0),
            expert_id: None,
            is_quantization_metadata: false,
        }
    }
}
