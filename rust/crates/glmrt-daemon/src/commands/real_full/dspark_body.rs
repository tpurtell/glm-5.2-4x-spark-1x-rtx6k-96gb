use std::ffi::c_void;
use std::time::Instant;

use anyhow::{Context, Result};
use glmrt_ffi::{GlmrtDeviceBuffer, NativeLibrary};
use serde::Serialize;

use super::coordinator_kernels::cuda_native_library;
use super::dspark_attention::{
    timing_summary, DsparkCudaEvent, DsparkCudaGraph, DsparkCudaStream, DsparkDeviceBuffer,
    DsparkPagedAttentionTiming,
};
use super::dspark_kv::{
    i32_buffer_bytes, DsparkKvStorage, DsparkPagedKvMetadata, DsparkPagedKvMetadataBuffers,
};
use crate::python_graph_capture::{
    launch_python_graph_capture, PythonDeviceBufferArg, PythonGraphCaptureLaunch, PythonKernelArg,
};

pub(super) const DSPARK_BODY_LAYERS: usize = 5;
pub(super) const DSPARK_BODY_HIDDEN: usize = 6_144;
pub(super) const DSPARK_BODY_INTERMEDIATE: usize = 12_288;
pub(super) const DSPARK_BODY_HEADS: usize = 64;
pub(super) const DSPARK_BODY_HEAD_DIM: usize = 64;
pub(super) const DSPARK_BODY_ATTENTION_WIDTH: usize = DSPARK_BODY_HEADS * DSPARK_BODY_HEAD_DIM;
pub(super) const DSPARK_BODY_WORKSPACE_BYTES: usize = 128 * 1024 * 1024;

const LAYER_WEIGHT_NAMES: [[&str; 8]; DSPARK_BODY_LAYERS] = [
    [
        "layer_0_input_norm",
        "layer_0_post_norm",
        "layer_0_q_norm",
        "layer_0_k_norm",
        "layer_0_qkv",
        "layer_0_output",
        "layer_0_gate_up",
        "layer_0_down",
    ],
    [
        "layer_1_input_norm",
        "layer_1_post_norm",
        "layer_1_q_norm",
        "layer_1_k_norm",
        "layer_1_qkv",
        "layer_1_output",
        "layer_1_gate_up",
        "layer_1_down",
    ],
    [
        "layer_2_input_norm",
        "layer_2_post_norm",
        "layer_2_q_norm",
        "layer_2_k_norm",
        "layer_2_qkv",
        "layer_2_output",
        "layer_2_gate_up",
        "layer_2_down",
    ],
    [
        "layer_3_input_norm",
        "layer_3_post_norm",
        "layer_3_q_norm",
        "layer_3_k_norm",
        "layer_3_qkv",
        "layer_3_output",
        "layer_3_gate_up",
        "layer_3_down",
    ],
    [
        "layer_4_input_norm",
        "layer_4_post_norm",
        "layer_4_q_norm",
        "layer_4_k_norm",
        "layer_4_qkv",
        "layer_4_output",
        "layer_4_gate_up",
        "layer_4_down",
    ],
];

#[derive(Clone, Copy, Debug)]
pub(super) struct DsparkBodyLayerResidentWeights {
    pub(super) input_norm: GlmrtDeviceBuffer,
    pub(super) post_norm: GlmrtDeviceBuffer,
    pub(super) q_norm: GlmrtDeviceBuffer,
    pub(super) k_norm: GlmrtDeviceBuffer,
    pub(super) qkv: GlmrtDeviceBuffer,
    pub(super) output: GlmrtDeviceBuffer,
    pub(super) gate_up: GlmrtDeviceBuffer,
    pub(super) down: GlmrtDeviceBuffer,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct DsparkBodyResidentWeights {
    pub(super) final_norm: GlmrtDeviceBuffer,
    pub(super) layers: [DsparkBodyLayerResidentWeights; DSPARK_BODY_LAYERS],
    pub(super) active_layers: usize,
    pub(super) resident_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct DsparkBodyBenchConfig {
    pub(super) layers: usize,
    pub(super) active_requests: usize,
    pub(super) query_rows: usize,
    pub(super) context_tokens: usize,
    pub(super) kv_capacity_tokens: usize,
    pub(super) page_size: usize,
    pub(super) kv_storage: DsparkKvStorage,
    pub(super) warmup: usize,
    pub(super) iterations: usize,
    pub(super) repeats: usize,
    pub(super) seed: i64,
    pub(super) initialize_input: bool,
    pub(super) initialize_kv: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct DsparkBodyGraphReport {
    backend: &'static str,
    kv_storage: DsparkKvStorage,
    kv_element_bytes: usize,
    layers: usize,
    active_requests: usize,
    query_rows_per_request: usize,
    context_tokens: usize,
    initial_kv_lengths: Vec<i32>,
    dynamic_replay_kv_lengths: Vec<i32>,
    page_size: usize,
    total_physical_pages: usize,
    max_pages_per_request: usize,
    max_sequence_kv: usize,
    physical_kv_bytes: u64,
    flashinfer_metadata_bytes: u64,
    rust_owned_mutable_bytes: u64,
    resident_weight_bytes: u64,
    graph_nodes: usize,
    graph_kernel_nodes: usize,
    graph_memcpy_nodes: usize,
    graph_memset_nodes: usize,
    eager_replay_exact: bool,
    eager_replay_mismatch_bytes: usize,
    eager_replay_max_abs: f64,
    eager_replay_relative_l2: f64,
    restored_replay_max_abs: f64,
    restored_replay_relative_l2: f64,
    absolute_query_positions_change_output: bool,
    absolute_query_position_changed_bytes: usize,
    dynamic_lengths_change_output: bool,
    dynamic_output_changed_bytes: usize,
    warmup: usize,
    iterations: usize,
    repeats: usize,
    gpu_ms_per_five_layer_replay: DsparkPagedAttentionTiming,
    host_ms_per_five_layer_replay: DsparkPagedAttentionTiming,
    fused_qkv_residents: bool,
    fused_gate_up_residents: bool,
    rust_owned_scratch: bool,
    dynamic_device_lengths: bool,
    paged_kv: bool,
    cold_capture_python_calls: usize,
    hot_replay_python_calls: usize,
    serving_dispatch_enabled: bool,
}

pub(super) fn benchmark_dspark_body_graph(
    weights: DsparkBodyResidentWeights,
    config: DsparkBodyBenchConfig,
) -> Result<DsparkBodyGraphReport> {
    let mut graph = DsparkBodyGraph::capture(weights, config)?;
    graph.replay()?;
    graph.stream.synchronize()?;
    let reference = graph.read_output(graph.reference_output)?;
    let replayed = graph.read_output(graph.output)?;
    let eager_replay_mismatch_bytes = byte_mismatch_count(&reference, &replayed);
    let eager_difference = bf16_difference(&reference, &replayed)?;
    anyhow::ensure!(
        eager_difference.max_abs <= 0.125 && eager_difference.relative_l2 <= 0.01,
        "dSpark body eager replay exceeds its numerical gate: max_abs={} relative_l2={}",
        eager_difference.max_abs,
        eager_difference.relative_l2
    );

    let shifted_query_positions = graph
        .initial_query_positions
        .iter()
        .map(|position| {
            position
                .checked_add(config.page_size as i32)
                .context("dSpark body shifted absolute query position overflow")
        })
        .collect::<Result<Vec<_>>>()?;
    graph.set_query_positions(&shifted_query_positions)?;
    graph.replay()?;
    graph.stream.synchronize()?;
    let shifted_positions = graph.read_output(graph.output)?;
    let absolute_query_position_changed_bytes = byte_mismatch_count(&reference, &shifted_positions);
    anyhow::ensure!(
        absolute_query_position_changed_bytes > 0,
        "dSpark body graph ignored changed absolute query positions"
    );
    graph.set_query_positions(&graph.initial_query_positions.clone())?;

    graph.set_kv_lengths(&graph.dynamic_replay_kv_lengths.clone())?;
    graph.replay()?;
    graph.stream.synchronize()?;
    let dynamic = graph.read_output(graph.output)?;
    let dynamic_output_changed_bytes = byte_mismatch_count(&reference, &dynamic);
    anyhow::ensure!(
        dynamic_output_changed_bytes > 0,
        "dSpark body graph ignored changed device-side KV lengths"
    );

    graph.set_kv_lengths(&graph.initial_kv_lengths.clone())?;
    graph.replay()?;
    graph.stream.synchronize()?;
    let restored = graph.read_output(graph.output)?;
    let restored_difference = bf16_difference(&reference, &restored)?;
    anyhow::ensure!(
        restored_difference.max_abs <= 0.125 && restored_difference.relative_l2 <= 0.01,
        "dSpark body restored replay exceeds its numerical gate: max_abs={} relative_l2={}",
        restored_difference.max_abs,
        restored_difference.relative_l2
    );

    for _ in 0..config.warmup {
        graph.replay()?;
    }
    graph.stream.synchronize()?;

    let mut gpu_samples = Vec::with_capacity(config.repeats);
    let mut host_samples = Vec::with_capacity(config.repeats);
    for _ in 0..config.repeats {
        let start_event = DsparkCudaEvent::create(graph.library)?;
        let end_event = DsparkCudaEvent::create(graph.library)?;
        unsafe {
            graph
                .library
                .cuda_event_record(start_event.raw, graph.stream.raw)
                .context("recording dSpark body benchmark start event")?;
        }
        let host_started = Instant::now();
        for _ in 0..config.iterations {
            graph.replay()?;
        }
        unsafe {
            graph
                .library
                .cuda_event_record(end_event.raw, graph.stream.raw)
                .context("recording dSpark body benchmark end event")?;
            graph
                .library
                .cuda_event_synchronize(end_event.raw)
                .context("waiting for dSpark body benchmark end event")?;
        }
        host_samples
            .push(host_started.elapsed().as_secs_f64() * 1_000.0 / config.iterations as f64);
        gpu_samples.push(
            unsafe {
                graph
                    .library
                    .cuda_event_elapsed_ms(start_event.raw, end_event.raw)
                    .context("measuring dSpark body CUDA graph replay")?
            } as f64
                / config.iterations as f64,
        );
    }

    Ok(DsparkBodyGraphReport {
        backend: match config.kv_storage {
            DsparkKvStorage::Bf16 => "fixed-address-bf16-cublas-triton-cudnn-paged",
            DsparkKvStorage::Fp8 => "fixed-address-bf16-cublas-triton-flashinfer-fa2-fp8-paged",
        },
        kv_storage: config.kv_storage,
        kv_element_bytes: config.kv_storage.element_bytes(),
        layers: config.layers,
        active_requests: config.active_requests,
        query_rows_per_request: config.query_rows,
        context_tokens: config.context_tokens,
        initial_kv_lengths: graph.initial_kv_lengths.clone(),
        dynamic_replay_kv_lengths: graph.dynamic_replay_kv_lengths.clone(),
        page_size: config.page_size,
        total_physical_pages: graph.total_physical_pages,
        max_pages_per_request: graph.max_pages_per_request,
        max_sequence_kv: graph.max_pages_per_request * config.page_size,
        physical_kv_bytes: graph.physical_kv_bytes,
        flashinfer_metadata_bytes: graph.flashinfer_metadata_bytes,
        rust_owned_mutable_bytes: graph.rust_owned_mutable_bytes,
        resident_weight_bytes: weights.resident_bytes,
        graph_nodes: graph.graph.node_count,
        graph_kernel_nodes: graph.graph.kernel_node_count,
        graph_memcpy_nodes: graph.graph.memcpy_node_count,
        graph_memset_nodes: graph.graph.memset_node_count,
        eager_replay_exact: eager_replay_mismatch_bytes == 0,
        eager_replay_mismatch_bytes,
        eager_replay_max_abs: eager_difference.max_abs,
        eager_replay_relative_l2: eager_difference.relative_l2,
        restored_replay_max_abs: restored_difference.max_abs,
        restored_replay_relative_l2: restored_difference.relative_l2,
        absolute_query_positions_change_output: true,
        absolute_query_position_changed_bytes,
        dynamic_lengths_change_output: true,
        dynamic_output_changed_bytes,
        warmup: config.warmup,
        iterations: config.iterations,
        repeats: config.repeats,
        gpu_ms_per_five_layer_replay: timing_summary(gpu_samples)?,
        host_ms_per_five_layer_replay: timing_summary(host_samples)?,
        fused_qkv_residents: true,
        fused_gate_up_residents: true,
        rust_owned_scratch: true,
        dynamic_device_lengths: true,
        paged_kv: true,
        cold_capture_python_calls: 2,
        hot_replay_python_calls: 0,
        serving_dispatch_enabled: false,
    })
}

struct DsparkBodyGraph {
    library: &'static NativeLibrary,
    graph: DsparkCudaGraph,
    stream: DsparkCudaStream,
    _owned_buffers: Vec<DsparkDeviceBuffer>,
    output: GlmrtDeviceBuffer,
    reference_output: GlmrtDeviceBuffer,
    kv_lengths: GlmrtDeviceBuffer,
    query_positions: GlmrtDeviceBuffer,
    paged_kv_metadata: DsparkPagedKvMetadataBuffers,
    config: DsparkBodyBenchConfig,
    total_physical_pages: usize,
    max_pages_per_request: usize,
    physical_pages_per_request: usize,
    physical_kv_bytes: u64,
    flashinfer_metadata_bytes: u64,
    rust_owned_mutable_bytes: u64,
    initial_kv_lengths: Vec<i32>,
    initial_query_positions: Vec<i32>,
    dynamic_replay_kv_lengths: Vec<i32>,
}

impl DsparkBodyGraph {
    fn capture(weights: DsparkBodyResidentWeights, config: DsparkBodyBenchConfig) -> Result<Self> {
        validate_config(config)?;
        validate_weights(weights)?;
        anyhow::ensure!(
            weights.active_layers == config.layers,
            "dSpark body resident/config layer mismatch: {} versus {}",
            weights.active_layers,
            config.layers,
        );
        let library = cuda_native_library()?;
        let stream = DsparkCudaStream::create(library)?;
        let total_rows = checked_mul(
            config.active_requests,
            config.query_rows,
            "dSpark body total rows",
        )?;
        let actual_kv_tokens = config
            .context_tokens
            .checked_add(config.query_rows)
            .context("dSpark body actual KV length overflow")?;
        anyhow::ensure!(
            actual_kv_tokens <= config.kv_capacity_tokens,
            "dSpark body context plus query rows ({actual_kv_tokens}) exceeds KV capacity {}",
            config.kv_capacity_tokens
        );
        let dynamic_kv_tokens = actual_kv_tokens
            .checked_add(3)
            .context("dSpark body dynamic KV probe length overflow")?;
        anyhow::ensure!(
            dynamic_kv_tokens <= config.kv_capacity_tokens,
            "dSpark body dynamic KV probe exceeds configured capacity"
        );
        let physical_pages_per_request = dynamic_kv_tokens.div_ceil(config.page_size);
        let total_physical_pages = checked_mul(
            config.active_requests,
            physical_pages_per_request,
            "dSpark body physical pages",
        )?;
        let max_pages_per_request = config.kv_capacity_tokens.div_ceil(config.page_size);

        let hidden_bytes = tensor_bytes(total_rows, DSPARK_BODY_HIDDEN, 2, "body hidden")?;
        let qkv_bytes = tensor_bytes(total_rows, 3 * DSPARK_BODY_ATTENTION_WIDTH, 2, "body QKV")?;
        let attention_bytes =
            tensor_bytes(total_rows, DSPARK_BODY_ATTENTION_WIDTH, 2, "body attention")?;
        let gate_up_bytes =
            tensor_bytes(total_rows, 2 * DSPARK_BODY_INTERMEDIATE, 2, "body gate/up")?;
        let activation_bytes =
            tensor_bytes(total_rows, DSPARK_BODY_INTERMEDIATE, 2, "body activation")?;
        let one_kv_bytes = checked_mul(
            checked_mul(
                checked_mul(
                    checked_mul(config.layers, total_physical_pages, "body KV layers/pages")?,
                    DSPARK_BODY_HEADS,
                    "body KV heads",
                )?,
                config.page_size,
                "body KV page rows",
            )?,
            DSPARK_BODY_HEAD_DIM * config.kv_storage.element_bytes(),
            "body KV head bytes",
        )?;
        let physical_kv_bytes = one_kv_bytes
            .checked_mul(2)
            .context("dSpark body K/V byte count overflow")? as u64;
        let length_bytes = checked_mul(
            config.active_requests,
            std::mem::size_of::<i32>(),
            "body length bytes",
        )?;
        let query_position_bytes = checked_mul(
            total_rows,
            std::mem::size_of::<i32>(),
            "body query position bytes",
        )?;
        let table_entries = checked_mul(
            config.active_requests,
            max_pages_per_request,
            "body block table entries",
        )?;
        let table_bytes = checked_mul(
            table_entries,
            std::mem::size_of::<i32>(),
            "body block table bytes",
        )?;
        let offset_entries = config
            .active_requests
            .checked_add(1)
            .context("dSpark body offset entry count overflow")?;
        let offset_bytes = checked_mul(
            offset_entries,
            std::mem::size_of::<i64>(),
            "body offset bytes",
        )?;

        let mut owned = Vec::new();
        let mut allocate = |bytes, label| -> Result<GlmrtDeviceBuffer> {
            let buffer = DsparkDeviceBuffer::new(library, bytes, label)?;
            let raw = buffer.raw;
            owned.push(buffer);
            Ok(raw)
        };
        let buffers = DsparkPythonBodyBuffers {
            input: allocate(hidden_bytes, "dSpark body input")?,
            output: allocate(hidden_bytes, "dSpark body output")?,
            reference_output: allocate(hidden_bytes, "dSpark body reference output")?,
            hidden_attention: allocate(hidden_bytes, "dSpark body attention residual")?,
            hidden_mlp: allocate(hidden_bytes, "dSpark body MLP residual")?,
            normalized: allocate(hidden_bytes, "dSpark body normalized hidden")?,
            qkv: allocate(qkv_bytes, "dSpark body QKV")?,
            q: allocate(attention_bytes, "dSpark body query")?,
            attention: allocate(attention_bytes, "dSpark body attention output")?,
            delta: allocate(hidden_bytes, "dSpark body projection delta")?,
            gate_up: allocate(gate_up_bytes, "dSpark body gate/up")?,
            activation: allocate(activation_bytes, "dSpark body activation")?,
            k_cache: allocate(one_kv_bytes, "dSpark body paged K")?,
            v_cache: allocate(one_kv_bytes, "dSpark body paged V")?,
            workspace: allocate(DSPARK_BODY_WORKSPACE_BYTES, "dSpark body cuDNN workspace")?,
            query_lengths: allocate(length_bytes, "dSpark body query lengths")?,
            kv_lengths: allocate(length_bytes, "dSpark body KV lengths")?,
            query_positions: allocate(
                query_position_bytes,
                "dSpark body absolute query positions",
            )?,
            block_tables: allocate(table_bytes, "dSpark body block tables")?,
            query_offsets: allocate(offset_bytes, "dSpark body query offsets")?,
            output_offsets: allocate(offset_bytes, "dSpark body output offsets")?,
            query_indptr: allocate(
                i32_buffer_bytes(config.active_requests + 1, "body query indptr")?,
                "dSpark body query indptr",
            )?,
            kv_indptr: allocate(
                i32_buffer_bytes(config.active_requests + 1, "body KV indptr")?,
                "dSpark body KV indptr",
            )?,
            page_indices: allocate(
                i32_buffer_bytes(total_physical_pages, "body page indices")?,
                "dSpark body page indices",
            )?,
            last_page_len: allocate(
                i32_buffer_bytes(config.active_requests, "body last-page lengths")?,
                "dSpark body last-page lengths",
            )?,
        };
        let paged_kv_metadata = DsparkPagedKvMetadataBuffers {
            query_indptr: buffers.query_indptr,
            kv_indptr: buffers.kv_indptr,
            page_indices: buffers.page_indices,
            last_page_len: buffers.last_page_len,
        };
        let flashinfer_metadata_bytes = [
            buffers.query_indptr.bytes,
            buffers.kv_indptr.bytes,
            buffers.page_indices.bytes,
            buffers.last_page_len.bytes,
        ]
        .into_iter()
        .try_fold(0_u64, |bytes, next| {
            bytes
                .checked_add(next as u64)
                .context("dSpark body FlashInfer metadata byte count overflow")
        })?;
        let rust_owned_mutable_bytes = owned.iter().try_fold(0_u64, |bytes, buffer| {
            bytes
                .checked_add(buffer.raw.bytes as u64)
                .context("dSpark body mutable byte count overflow")
        })?;

        let initial_kv_lengths = (0..config.active_requests)
            .map(|request| {
                actual_kv_tokens
                    .saturating_sub(request * 7)
                    .try_into()
                    .context("dSpark body initial KV length does not fit i32")
            })
            .collect::<Result<Vec<i32>>>()?;
        let dynamic_replay_kv_lengths = initial_kv_lengths
            .iter()
            .map(|length| {
                length
                    .checked_add(3)
                    .context("dSpark body replay KV length overflow")
            })
            .collect::<Result<Vec<i32>>>()?;
        library
            .copy_h2d(buffers.kv_lengths, as_bytes(&initial_kv_lengths))
            .context("uploading dSpark body initial KV lengths")?;
        let initial_query_positions =
            query_positions_for_lengths(&initial_kv_lengths, config.query_rows)?;
        library
            .copy_h2d(buffers.query_positions, as_bytes(&initial_query_positions))
            .context("uploading dSpark body initial query positions")?;
        DsparkPagedKvMetadata::for_lengths(
            &initial_kv_lengths,
            config.query_rows,
            config.page_size,
            physical_pages_per_request,
        )?
        .upload(library, paged_kv_metadata)?;

        let mut table = vec![0_i32; table_entries];
        for request in 0..config.active_requests {
            for page in 0..physical_pages_per_request {
                table[request * max_pages_per_request + page] =
                    (request * physical_pages_per_request + page)
                        .try_into()
                        .context("dSpark body physical page ID does not fit i32")?;
            }
        }
        library
            .copy_h2d(buffers.block_tables, as_bytes(&table))
            .context("uploading dSpark body block tables")?;

        launch_python_body(
            stream.raw,
            &buffers,
            &weights,
            config,
            total_physical_pages,
            max_pages_per_request,
            "prepare_dspark_cudnn_paged_body",
        )?;
        stream.synchronize()?;

        unsafe {
            library
                .cuda_graph_begin_capture(stream.raw)
                .context("beginning dSpark body CUDA graph capture")?;
        }
        if let Err(error) = launch_python_body(
            stream.raw,
            &buffers,
            &weights,
            config,
            total_physical_pages,
            max_pages_per_request,
            "capture_dspark_cudnn_paged_body",
        ) {
            unsafe {
                let _ = library.cuda_graph_end_capture_retained(stream.raw);
            }
            return Err(error).context("capturing dSpark transformer body");
        }
        let capture = unsafe {
            library
                .cuda_graph_end_capture_retained(stream.raw)
                .context("ending dSpark body CUDA graph capture")?
        };
        let graph = DsparkCudaGraph::new(library, capture)?;
        graph.validate()?;

        Ok(Self {
            library,
            graph,
            stream,
            _owned_buffers: owned,
            output: buffers.output,
            reference_output: buffers.reference_output,
            kv_lengths: buffers.kv_lengths,
            query_positions: buffers.query_positions,
            paged_kv_metadata,
            config,
            total_physical_pages,
            max_pages_per_request,
            physical_pages_per_request,
            physical_kv_bytes,
            flashinfer_metadata_bytes,
            rust_owned_mutable_bytes,
            initial_kv_lengths,
            initial_query_positions,
            dynamic_replay_kv_lengths,
        })
    }

    fn set_query_positions(&mut self, positions: &[i32]) -> Result<()> {
        let expected = self
            .config
            .active_requests
            .checked_mul(self.config.query_rows)
            .context("dSpark body expected query position count overflow")?;
        anyhow::ensure!(
            positions.len() == expected && positions.iter().all(|position| *position >= 0),
            "dSpark body absolute query positions are invalid: {positions:?}"
        );
        self.library
            .copy_h2d(self.query_positions, as_bytes(positions))
            .context("uploading dynamic dSpark body absolute query positions")
    }

    fn set_kv_lengths(&mut self, lengths: &[i32]) -> Result<()> {
        anyhow::ensure!(
            lengths.len() == self.config.active_requests,
            "dSpark body KV update has {} requests, expected {}",
            lengths.len(),
            self.config.active_requests
        );
        let max_sequence_kv = self.physical_pages_per_request * self.config.page_size;
        anyhow::ensure!(
            lengths.iter().all(|length| {
                (*length as usize) >= self.config.query_rows
                    && (*length as usize) <= max_sequence_kv
            }),
            "dSpark body KV update exceeds captured bounds: {lengths:?}"
        );
        self.library
            .copy_h2d(self.kv_lengths, as_bytes(lengths))
            .context("uploading dynamic dSpark body KV lengths")?;
        let query_positions = query_positions_for_lengths(lengths, self.config.query_rows)?;
        self.set_query_positions(&query_positions)?;
        DsparkPagedKvMetadata::for_lengths(
            lengths,
            self.config.query_rows,
            self.config.page_size,
            self.physical_pages_per_request,
        )?
        .upload(self.library, self.paged_kv_metadata)
    }

    fn replay(&self) -> Result<()> {
        self.graph.validate()?;
        unsafe {
            self.library
                .cuda_graph_launch(self.graph.exec_raw, self.stream.raw)
                .context("launching dSpark body CUDA graph")
        }
    }

    fn read_output(&self, buffer: GlmrtDeviceBuffer) -> Result<Vec<u8>> {
        let bytes = tensor_bytes(
            self.config.active_requests * self.config.query_rows,
            DSPARK_BODY_HIDDEN,
            2,
            "dSpark body readback",
        )?;
        let mut output = vec![0_u8; bytes];
        self.library
            .copy_d2h(&mut output, buffer)
            .context("reading dSpark body output")?;
        Ok(output)
    }
}

#[derive(Clone, Copy)]
pub(super) struct DsparkPythonBodyBuffers {
    pub(super) input: GlmrtDeviceBuffer,
    pub(super) output: GlmrtDeviceBuffer,
    pub(super) reference_output: GlmrtDeviceBuffer,
    pub(super) hidden_attention: GlmrtDeviceBuffer,
    pub(super) hidden_mlp: GlmrtDeviceBuffer,
    pub(super) normalized: GlmrtDeviceBuffer,
    pub(super) qkv: GlmrtDeviceBuffer,
    pub(super) q: GlmrtDeviceBuffer,
    pub(super) attention: GlmrtDeviceBuffer,
    pub(super) delta: GlmrtDeviceBuffer,
    pub(super) gate_up: GlmrtDeviceBuffer,
    pub(super) activation: GlmrtDeviceBuffer,
    pub(super) k_cache: GlmrtDeviceBuffer,
    pub(super) v_cache: GlmrtDeviceBuffer,
    pub(super) workspace: GlmrtDeviceBuffer,
    pub(super) query_lengths: GlmrtDeviceBuffer,
    pub(super) kv_lengths: GlmrtDeviceBuffer,
    pub(super) query_positions: GlmrtDeviceBuffer,
    pub(super) block_tables: GlmrtDeviceBuffer,
    pub(super) query_offsets: GlmrtDeviceBuffer,
    pub(super) output_offsets: GlmrtDeviceBuffer,
    pub(super) query_indptr: GlmrtDeviceBuffer,
    pub(super) kv_indptr: GlmrtDeviceBuffer,
    pub(super) page_indices: GlmrtDeviceBuffer,
    pub(super) last_page_len: GlmrtDeviceBuffer,
}

pub(super) fn launch_python_body(
    cuda_stream: *mut c_void,
    buffers: &DsparkPythonBodyBuffers,
    weights: &DsparkBodyResidentWeights,
    config: DsparkBodyBenchConfig,
    total_pages: usize,
    max_pages_per_request: usize,
    function: &str,
) -> Result<()> {
    let mut device_buffers = vec![
        python_buffer("input", buffers.input),
        python_buffer("output", buffers.output),
        python_buffer("reference_output", buffers.reference_output),
        python_buffer("hidden_attention", buffers.hidden_attention),
        python_buffer("hidden_mlp", buffers.hidden_mlp),
        python_buffer("normalized", buffers.normalized),
        python_buffer("qkv", buffers.qkv),
        python_buffer("q", buffers.q),
        python_buffer("attention", buffers.attention),
        python_buffer("delta", buffers.delta),
        python_buffer("gate_up", buffers.gate_up),
        python_buffer("activation", buffers.activation),
        python_buffer("k_cache", buffers.k_cache),
        python_buffer("v_cache", buffers.v_cache),
        python_buffer("workspace", buffers.workspace),
        python_buffer("query_lengths", buffers.query_lengths),
        python_buffer("kv_lengths", buffers.kv_lengths),
        python_buffer("query_positions", buffers.query_positions),
        python_buffer("block_tables", buffers.block_tables),
        python_buffer("query_offsets", buffers.query_offsets),
        python_buffer("output_offsets", buffers.output_offsets),
        python_buffer("query_indptr", buffers.query_indptr),
        python_buffer("kv_indptr", buffers.kv_indptr),
        python_buffer("page_indices", buffers.page_indices),
        python_buffer("last_page_len", buffers.last_page_len),
        python_buffer("final_norm", weights.final_norm),
    ];
    for (names, layer) in LAYER_WEIGHT_NAMES
        .iter()
        .zip(weights.layers.iter())
        .take(config.layers)
    {
        for (name, buffer) in names.iter().copied().zip([
            layer.input_norm,
            layer.post_norm,
            layer.q_norm,
            layer.k_norm,
            layer.qkv,
            layer.output,
            layer.gate_up,
            layer.down,
        ]) {
            device_buffers.push(python_buffer(name, buffer));
        }
    }
    let kwargs = [
        ("layers", PythonKernelArg::Usize(config.layers)),
        (
            "active_requests",
            PythonKernelArg::Usize(config.active_requests),
        ),
        ("query_rows", PythonKernelArg::Usize(config.query_rows)),
        ("total_pages", PythonKernelArg::Usize(total_pages)),
        ("page_size", PythonKernelArg::Usize(config.page_size)),
        (
            "max_pages_per_request",
            PythonKernelArg::Usize(max_pages_per_request),
        ),
        ("hidden_size", PythonKernelArg::Usize(DSPARK_BODY_HIDDEN)),
        (
            "intermediate_size",
            PythonKernelArg::Usize(DSPARK_BODY_INTERMEDIATE),
        ),
        ("heads", PythonKernelArg::Usize(DSPARK_BODY_HEADS)),
        ("head_dim", PythonKernelArg::Usize(DSPARK_BODY_HEAD_DIM)),
        ("seed", PythonKernelArg::I64(config.seed)),
        (
            "initialize_input",
            PythonKernelArg::Bool(config.initialize_input),
        ),
        ("initialize_kv", PythonKernelArg::Bool(config.initialize_kv)),
        (
            "cache_dtype",
            PythonKernelArg::Str(config.kv_storage.label()),
        ),
    ];
    launch_python_graph_capture(PythonGraphCaptureLaunch {
        module: "dspark_body_capture",
        function,
        cuda_stream,
        buffers: &device_buffers,
        kwargs: &kwargs,
    })
}

fn python_buffer(name: &str, buffer: GlmrtDeviceBuffer) -> PythonDeviceBufferArg<'_> {
    PythonDeviceBufferArg {
        name,
        ptr: buffer.ptr,
        bytes: buffer.bytes,
        device_id: buffer.device_id,
        flags: buffer.flags,
    }
}

fn query_positions_for_lengths(lengths: &[i32], query_rows: usize) -> Result<Vec<i32>> {
    anyhow::ensure!(query_rows > 0, "dSpark body query rows are zero");
    let mut positions = Vec::with_capacity(
        lengths
            .len()
            .checked_mul(query_rows)
            .context("dSpark body query position count overflow")?,
    );
    let query_rows_i32 =
        i32::try_from(query_rows).context("dSpark body query rows do not fit i32")?;
    for length in lengths.iter().copied() {
        let start = length
            .checked_sub(query_rows_i32)
            .context("dSpark body KV length is shorter than its query suffix")?;
        anyhow::ensure!(
            start >= 0,
            "dSpark body KV length is shorter than its query suffix"
        );
        for row in 0..query_rows_i32 {
            positions.push(
                start
                    .checked_add(row)
                    .context("dSpark body query position overflow")?,
            );
        }
    }
    Ok(positions)
}

fn validate_config(config: DsparkBodyBenchConfig) -> Result<()> {
    anyhow::ensure!(
        (1..=DSPARK_BODY_LAYERS).contains(&config.layers),
        "dSpark body layer count must be between 1 and {DSPARK_BODY_LAYERS}"
    );
    anyhow::ensure!(
        matches!(config.active_requests, 1 | 2 | 4),
        "dSpark body active request bucket must be 1, 2, or 4"
    );
    anyhow::ensure!(
        matches!(config.query_rows, 8 | 16),
        "dSpark body query rows must be 8 or 16"
    );
    anyhow::ensure!(
        config.context_tokens > 0,
        "dSpark body context must be positive"
    );
    let required_kv_tokens = config
        .context_tokens
        .checked_add(config.query_rows)
        .context("dSpark body context plus query rows overflow")?;
    anyhow::ensure!(
        config.kv_capacity_tokens >= required_kv_tokens,
        "dSpark body KV capacity is smaller than context plus query rows"
    );
    anyhow::ensure!(
        matches!(config.page_size, 16 | 32 | 64 | 128),
        "dSpark body page size must be 16, 32, 64, or 128"
    );
    anyhow::ensure!(
        config.iterations > 0 && config.repeats > 0,
        "dSpark body benchmark iterations and repeats must be positive"
    );
    Ok(())
}

fn validate_weights(weights: DsparkBodyResidentWeights) -> Result<()> {
    anyhow::ensure!(
        (1..=DSPARK_BODY_LAYERS).contains(&weights.active_layers),
        "dSpark body resident layer count is invalid"
    );
    validate_buffer("final norm", weights.final_norm, DSPARK_BODY_HIDDEN * 2)?;
    for (index, layer) in weights
        .layers
        .iter()
        .take(weights.active_layers)
        .enumerate()
    {
        validate_buffer(
            &format!("layer {index} input norm"),
            layer.input_norm,
            DSPARK_BODY_HIDDEN * 2,
        )?;
        validate_buffer(
            &format!("layer {index} post norm"),
            layer.post_norm,
            DSPARK_BODY_HIDDEN * 2,
        )?;
        validate_buffer(
            &format!("layer {index} Q norm"),
            layer.q_norm,
            DSPARK_BODY_HEAD_DIM * 2,
        )?;
        validate_buffer(
            &format!("layer {index} K norm"),
            layer.k_norm,
            DSPARK_BODY_HEAD_DIM * 2,
        )?;
        validate_buffer(
            &format!("layer {index} QKV"),
            layer.qkv,
            3 * DSPARK_BODY_ATTENTION_WIDTH * DSPARK_BODY_HIDDEN * 2,
        )?;
        validate_buffer(
            &format!("layer {index} output"),
            layer.output,
            DSPARK_BODY_HIDDEN * DSPARK_BODY_ATTENTION_WIDTH * 2,
        )?;
        validate_buffer(
            &format!("layer {index} gate/up"),
            layer.gate_up,
            2 * DSPARK_BODY_INTERMEDIATE * DSPARK_BODY_HIDDEN * 2,
        )?;
        validate_buffer(
            &format!("layer {index} down"),
            layer.down,
            DSPARK_BODY_HIDDEN * DSPARK_BODY_INTERMEDIATE * 2,
        )?;
    }
    Ok(())
}

fn validate_buffer(label: &str, buffer: GlmrtDeviceBuffer, expected_bytes: usize) -> Result<()> {
    anyhow::ensure!(!buffer.ptr.is_null(), "dSpark body {label} buffer is null");
    anyhow::ensure!(
        buffer.bytes == expected_bytes,
        "dSpark body {label} has {} bytes, expected {expected_bytes}",
        buffer.bytes
    );
    Ok(())
}

fn tensor_bytes(rows: usize, width: usize, element_bytes: usize, label: &str) -> Result<usize> {
    checked_mul(checked_mul(rows, width, label)?, element_bytes, label)
}

fn checked_mul(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .with_context(|| format!("dSpark {label} overflow"))
}

fn byte_mismatch_count(left: &[u8], right: &[u8]) -> usize {
    left.iter()
        .zip(right)
        .filter(|(left, right)| left != right)
        .count()
}

#[derive(Clone, Copy, Debug)]
struct Bf16Difference {
    max_abs: f64,
    relative_l2: f64,
}

fn bf16_difference(reference: &[u8], candidate: &[u8]) -> Result<Bf16Difference> {
    anyhow::ensure!(
        reference.len() == candidate.len() && reference.len() % 2 == 0,
        "dSpark BF16 comparison byte lengths are invalid"
    );
    let mut max_abs = 0.0_f64;
    let mut squared_delta = 0.0_f64;
    let mut squared_reference = 0.0_f64;
    for (reference, candidate) in reference.chunks_exact(2).zip(candidate.chunks_exact(2)) {
        let reference =
            f32::from_bits((u16::from_le_bytes([reference[0], reference[1]]) as u32) << 16) as f64;
        let candidate =
            f32::from_bits((u16::from_le_bytes([candidate[0], candidate[1]]) as u32) << 16) as f64;
        anyhow::ensure!(
            reference.is_finite() && candidate.is_finite(),
            "dSpark body output contains a non-finite BF16 value"
        );
        let delta = candidate - reference;
        max_abs = max_abs.max(delta.abs());
        squared_delta += delta * delta;
        squared_reference += reference * reference;
    }
    Ok(Bf16Difference {
        max_abs,
        relative_l2: (squared_delta / squared_reference.max(f64::MIN_POSITIVE)).sqrt(),
    })
}

fn as_bytes<T>(values: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

#[cfg(test)]
mod tests {
    use super::{query_positions_for_lengths, validate_config, DsparkBodyBenchConfig};
    use crate::commands::real_full::dspark_kv::DsparkKvStorage;

    fn config() -> DsparkBodyBenchConfig {
        DsparkBodyBenchConfig {
            layers: 5,
            active_requests: 4,
            query_rows: 16,
            context_tokens: 1_024,
            kv_capacity_tokens: 256 * 1_024,
            page_size: 64,
            kv_storage: DsparkKvStorage::Bf16,
            warmup: 2,
            iterations: 10,
            repeats: 3,
            seed: 17,
            initialize_input: true,
            initialize_kv: true,
        }
    }

    #[test]
    fn accepts_production_body_buckets() {
        for active_requests in [1, 2, 4] {
            for query_rows in [8, 16] {
                validate_config(DsparkBodyBenchConfig {
                    active_requests,
                    query_rows,
                    ..config()
                })
                .unwrap();
            }
        }
    }

    #[test]
    fn rejects_body_capacity_and_shape_mismatches() {
        let mut invalid = config();
        invalid.active_requests = 3;
        assert!(validate_config(invalid).is_err());
        invalid = config();
        invalid.kv_capacity_tokens = invalid.context_tokens;
        assert!(validate_config(invalid).is_err());
        invalid = config();
        invalid.page_size = 63;
        assert!(validate_config(invalid).is_err());
    }

    #[test]
    fn derives_absolute_query_positions_from_legacy_kv_lengths() {
        assert_eq!(
            query_positions_for_lengths(&[72, 136], 8).unwrap(),
            [64, 65, 66, 67, 68, 69, 70, 71, 128, 129, 130, 131, 132, 133, 134, 135,]
        );
        assert!(query_positions_for_lengths(&[7], 8).is_err());
    }
}
