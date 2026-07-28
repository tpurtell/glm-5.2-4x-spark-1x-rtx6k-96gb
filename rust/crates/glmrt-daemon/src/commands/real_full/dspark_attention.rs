use std::ffi::c_void;
use std::time::Instant;

use anyhow::{Context, Result};
use glmrt_ffi::{GlmrtCudaGraphCaptureInfo, GlmrtDeviceBuffer, NativeLibrary};
use serde::Serialize;

use super::coordinator_kernels::cuda_native_library;
use super::dspark_kv::{
    i32_buffer_bytes, DsparkKvStorage, DsparkPagedKvMetadata, DsparkPagedKvMetadataBuffers,
};
use crate::python_graph_capture::{
    launch_python_graph_capture, PythonDeviceBufferArg, PythonGraphCaptureLaunch, PythonKernelArg,
};

const DSPARK_ATTENTION_LAYERS: usize = 5;
const DSPARK_ATTENTION_HEADS: usize = 64;
const DSPARK_ATTENTION_HEAD_DIM: usize = 64;
const DSPARK_ATTENTION_WORKSPACE_BYTES: usize = 128 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub(super) struct DsparkPagedAttentionBenchConfig {
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
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct DsparkPagedAttentionTiming {
    min: f64,
    median: f64,
    p90: f64,
    max: f64,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct DsparkPagedAttentionGraphReport {
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
    page_table_bytes: u64,
    flashinfer_metadata_bytes: u64,
    graph_nodes: usize,
    graph_kernel_nodes: usize,
    graph_memcpy_nodes: usize,
    graph_memset_nodes: usize,
    warmup: usize,
    iterations: usize,
    repeats: usize,
    gpu_ms_per_five_layer_replay: DsparkPagedAttentionTiming,
    host_ms_per_five_layer_replay: DsparkPagedAttentionTiming,
    dynamic_device_lengths: bool,
    paged_kv: bool,
    cold_capture_python_calls: usize,
    hot_replay_python_calls: usize,
    serving_dispatch_enabled: bool,
}

pub(super) fn benchmark_dspark_paged_attention_graph(
    config: DsparkPagedAttentionBenchConfig,
) -> Result<DsparkPagedAttentionGraphReport> {
    let mut graph = DsparkPagedAttentionGraph::capture(config)?;
    graph.set_kv_lengths(&graph.dynamic_replay_kv_lengths.clone())?;
    graph.replay()?;
    graph.stream.synchronize()?;

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
                .context("recording dSpark attention benchmark start event")?;
        }
        let host_started = Instant::now();
        for _ in 0..config.iterations {
            graph.replay()?;
        }
        unsafe {
            graph
                .library
                .cuda_event_record(end_event.raw, graph.stream.raw)
                .context("recording dSpark attention benchmark end event")?;
            graph
                .library
                .cuda_event_synchronize(end_event.raw)
                .context("waiting for dSpark attention benchmark end event")?;
        }
        let host_ms = host_started.elapsed().as_secs_f64() * 1_000.0 / config.iterations as f64;
        let gpu_ms = unsafe {
            graph
                .library
                .cuda_event_elapsed_ms(start_event.raw, end_event.raw)
                .context("measuring dSpark attention CUDA graph replay")?
        } as f64
            / config.iterations as f64;
        gpu_samples.push(gpu_ms);
        host_samples.push(host_ms);
    }

    Ok(DsparkPagedAttentionGraphReport {
        backend: match config.kv_storage {
            DsparkKvStorage::Bf16 => "flashinfer-cudnn-paged-bf16",
            DsparkKvStorage::Fp8 => "flashinfer-fa2-paged-fp8-e4m3",
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
        page_table_bytes: graph.block_tables.raw.bytes as u64,
        flashinfer_metadata_bytes: graph.flashinfer_metadata_bytes,
        graph_nodes: graph.graph.node_count,
        graph_kernel_nodes: graph.graph.kernel_node_count,
        graph_memcpy_nodes: graph.graph.memcpy_node_count,
        graph_memset_nodes: graph.graph.memset_node_count,
        warmup: config.warmup,
        iterations: config.iterations,
        repeats: config.repeats,
        gpu_ms_per_five_layer_replay: timing_summary(gpu_samples)?,
        host_ms_per_five_layer_replay: timing_summary(host_samples)?,
        dynamic_device_lengths: true,
        paged_kv: true,
        cold_capture_python_calls: 2,
        hot_replay_python_calls: 0,
        serving_dispatch_enabled: false,
    })
}

// These buffers are intentionally retained even when their handles are only
// passed during capture: every graph node keeps the underlying CUDA address.
#[allow(dead_code)]
struct DsparkPagedAttentionGraph {
    library: &'static NativeLibrary,
    graph: DsparkCudaGraph,
    stream: DsparkCudaStream,
    q: DsparkDeviceBuffer,
    k_cache: DsparkDeviceBuffer,
    v_cache: DsparkDeviceBuffer,
    output: DsparkDeviceBuffer,
    workspace: DsparkDeviceBuffer,
    query_lengths: DsparkDeviceBuffer,
    kv_lengths: DsparkDeviceBuffer,
    block_tables: DsparkDeviceBuffer,
    query_offsets: DsparkDeviceBuffer,
    output_offsets: DsparkDeviceBuffer,
    query_indptr: DsparkDeviceBuffer,
    kv_indptr: DsparkDeviceBuffer,
    page_indices: DsparkDeviceBuffer,
    last_page_len: DsparkDeviceBuffer,
    config: DsparkPagedAttentionBenchConfig,
    total_physical_pages: usize,
    max_pages_per_request: usize,
    physical_pages_per_request: usize,
    physical_kv_bytes: u64,
    flashinfer_metadata_bytes: u64,
    initial_kv_lengths: Vec<i32>,
    dynamic_replay_kv_lengths: Vec<i32>,
}

impl DsparkPagedAttentionGraph {
    fn capture(config: DsparkPagedAttentionBenchConfig) -> Result<Self> {
        validate_config(config)?;
        let library = cuda_native_library()?;
        let stream = DsparkCudaStream::create(library)?;
        let total_query_rows = checked_mul(
            config.active_requests,
            config.query_rows,
            "dSpark total query rows",
        )?;
        let actual_kv_tokens = config
            .context_tokens
            .checked_add(config.query_rows)
            .context("dSpark attention actual KV length overflow")?;
        anyhow::ensure!(
            actual_kv_tokens <= config.kv_capacity_tokens,
            "dSpark attention context plus query rows ({actual_kv_tokens}) exceeds KV capacity {}",
            config.kv_capacity_tokens
        );
        let physical_pages_per_request = div_ceil(actual_kv_tokens, config.page_size);
        let total_physical_pages = checked_mul(
            config.active_requests,
            physical_pages_per_request,
            "dSpark physical KV pages",
        )?;
        let max_pages_per_request = div_ceil(config.kv_capacity_tokens, config.page_size);

        let q_values = checked_mul(
            checked_mul(config.layers, total_query_rows, "dSpark Q layer rows")?,
            DSPARK_ATTENTION_HEADS * DSPARK_ATTENTION_HEAD_DIM,
            "dSpark Q values",
        )?;
        let q_bytes = checked_mul(q_values, 2, "dSpark Q bytes")?;
        let one_kv_values = checked_mul(
            checked_mul(
                checked_mul(config.layers, total_physical_pages, "dSpark KV layer pages")?,
                DSPARK_ATTENTION_HEADS * config.page_size,
                "dSpark KV page head rows",
            )?,
            DSPARK_ATTENTION_HEAD_DIM,
            "dSpark KV values",
        )?;
        let one_kv_bytes = checked_mul(
            one_kv_values,
            config.kv_storage.element_bytes(),
            "dSpark one-sided KV bytes",
        )?;
        let physical_kv_bytes = one_kv_bytes
            .checked_mul(2)
            .context("dSpark physical K/V byte count overflow")?
            as u64;
        let lengths_bytes = checked_mul(
            config.active_requests,
            std::mem::size_of::<i32>(),
            "dSpark length bytes",
        )?;
        let table_entries = checked_mul(
            config.active_requests,
            max_pages_per_request,
            "dSpark page table entries",
        )?;
        let table_bytes = checked_mul(
            table_entries,
            std::mem::size_of::<i32>(),
            "dSpark page table bytes",
        )?;
        let offsets_bytes = checked_mul(
            config.active_requests + 1,
            std::mem::size_of::<i64>(),
            "dSpark offset bytes",
        )?;

        let q = DsparkDeviceBuffer::new(library, q_bytes, "dSpark attention Q")?;
        let k_cache = DsparkDeviceBuffer::new(library, one_kv_bytes, "dSpark attention paged K")?;
        let v_cache = DsparkDeviceBuffer::new(library, one_kv_bytes, "dSpark attention paged V")?;
        let output = DsparkDeviceBuffer::new(library, q_bytes, "dSpark attention output")?;
        let workspace = DsparkDeviceBuffer::new(
            library,
            DSPARK_ATTENTION_WORKSPACE_BYTES,
            "dSpark attention cuDNN workspace",
        )?;
        let query_lengths =
            DsparkDeviceBuffer::new(library, lengths_bytes, "dSpark query lengths")?;
        let kv_lengths = DsparkDeviceBuffer::new(library, lengths_bytes, "dSpark KV lengths")?;
        let block_tables = DsparkDeviceBuffer::new(library, table_bytes, "dSpark KV block tables")?;
        let query_offsets =
            DsparkDeviceBuffer::new(library, offsets_bytes, "dSpark query offsets")?;
        let output_offsets =
            DsparkDeviceBuffer::new(library, offsets_bytes, "dSpark output offsets")?;
        let query_indptr = DsparkDeviceBuffer::new(
            library,
            i32_buffer_bytes(config.active_requests + 1, "query indptr")?,
            "dSpark query indptr",
        )?;
        let kv_indptr = DsparkDeviceBuffer::new(
            library,
            i32_buffer_bytes(config.active_requests + 1, "KV indptr")?,
            "dSpark KV indptr",
        )?;
        let page_indices = DsparkDeviceBuffer::new(
            library,
            i32_buffer_bytes(total_physical_pages, "page indices")?,
            "dSpark page indices",
        )?;
        let last_page_len = DsparkDeviceBuffer::new(
            library,
            i32_buffer_bytes(config.active_requests, "last-page lengths")?,
            "dSpark last-page lengths",
        )?;
        let flashinfer_metadata_bytes = [
            query_indptr.raw.bytes,
            kv_indptr.raw.bytes,
            page_indices.raw.bytes,
            last_page_len.raw.bytes,
        ]
        .into_iter()
        .try_fold(0_u64, |bytes, next| {
            bytes
                .checked_add(next as u64)
                .context("dSpark FlashInfer metadata byte count overflow")
        })?;

        let initial_kv_lengths = (0..config.active_requests)
            .map(|request| {
                actual_kv_tokens
                    .saturating_sub(request * 7)
                    .try_into()
                    .context("dSpark initial KV length does not fit i32")
            })
            .collect::<Result<Vec<i32>>>()?;
        let dynamic_replay_kv_lengths = (0..config.active_requests)
            .map(|request| {
                actual_kv_tokens
                    .saturating_sub((request + 1) * 3)
                    .max(1)
                    .try_into()
                    .context("dSpark replay KV length does not fit i32")
            })
            .collect::<Result<Vec<i32>>>()?;
        library
            .copy_h2d(kv_lengths.raw, as_bytes(&initial_kv_lengths))
            .context("uploading dSpark initial KV lengths")?;
        DsparkPagedKvMetadata::for_lengths(
            &initial_kv_lengths,
            config.query_rows,
            config.page_size,
            physical_pages_per_request,
        )?
        .upload(
            library,
            DsparkPagedKvMetadataBuffers {
                query_indptr: query_indptr.raw,
                kv_indptr: kv_indptr.raw,
                page_indices: page_indices.raw,
                last_page_len: last_page_len.raw,
            },
        )?;

        let mut table = vec![0_i32; table_entries];
        for request in 0..config.active_requests {
            for page in 0..physical_pages_per_request {
                table[request * max_pages_per_request + page] =
                    (request * physical_pages_per_request + page)
                        .try_into()
                        .context("dSpark physical page ID does not fit i32")?;
            }
        }
        library
            .copy_h2d(block_tables.raw, as_bytes(&table))
            .context("uploading dSpark attention block tables")?;

        let buffers = DsparkPythonAttentionBuffers {
            q: q.raw,
            k_cache: k_cache.raw,
            v_cache: v_cache.raw,
            output: output.raw,
            workspace: workspace.raw,
            query_lengths: query_lengths.raw,
            kv_lengths: kv_lengths.raw,
            block_tables: block_tables.raw,
            query_offsets: query_offsets.raw,
            output_offsets: output_offsets.raw,
            query_indptr: query_indptr.raw,
            kv_indptr: kv_indptr.raw,
            page_indices: page_indices.raw,
            last_page_len: last_page_len.raw,
        };
        launch_python_attention(
            stream.raw,
            &buffers,
            config,
            total_physical_pages,
            max_pages_per_request,
            "prepare_dspark_cudnn_paged_attention",
        )?;
        stream.synchronize()?;

        unsafe {
            library
                .cuda_graph_begin_capture(stream.raw)
                .context("beginning dSpark attention CUDA graph capture")?;
        }
        if let Err(error) = launch_python_attention(
            stream.raw,
            &buffers,
            config,
            total_physical_pages,
            max_pages_per_request,
            "capture_dspark_cudnn_paged_attention",
        ) {
            unsafe {
                let _ = library.cuda_graph_end_capture_retained(stream.raw);
            }
            return Err(error).context("capturing dSpark paged attention");
        }
        let capture = unsafe {
            library
                .cuda_graph_end_capture_retained(stream.raw)
                .context("ending dSpark attention CUDA graph capture")?
        };
        let graph = DsparkCudaGraph::new(library, capture)?;
        graph.validate_min_kernel_nodes(config.layers)?;

        Ok(Self {
            library,
            graph,
            stream,
            q,
            k_cache,
            v_cache,
            output,
            workspace,
            query_lengths,
            kv_lengths,
            block_tables,
            query_offsets,
            output_offsets,
            query_indptr,
            kv_indptr,
            page_indices,
            last_page_len,
            config,
            total_physical_pages,
            max_pages_per_request,
            physical_pages_per_request,
            physical_kv_bytes,
            flashinfer_metadata_bytes,
            initial_kv_lengths,
            dynamic_replay_kv_lengths,
        })
    }

    fn set_kv_lengths(&mut self, lengths: &[i32]) -> Result<()> {
        anyhow::ensure!(
            lengths.len() == self.config.active_requests,
            "dSpark KV length update has {} requests, expected {}",
            lengths.len(),
            self.config.active_requests
        );
        let max_sequence_kv = self.physical_pages_per_request * self.config.page_size;
        anyhow::ensure!(
            lengths
                .iter()
                .all(|length| *length > 0 && (*length as usize) <= max_sequence_kv),
            "dSpark KV length update exceeds captured capacity {max_sequence_kv}: {lengths:?}"
        );
        self.library
            .copy_h2d(self.kv_lengths.raw, as_bytes(lengths))
            .context("uploading dynamic dSpark KV lengths")?;
        DsparkPagedKvMetadata::for_lengths(
            lengths,
            self.config.query_rows,
            self.config.page_size,
            self.physical_pages_per_request,
        )?
        .upload(
            self.library,
            DsparkPagedKvMetadataBuffers {
                query_indptr: self.query_indptr.raw,
                kv_indptr: self.kv_indptr.raw,
                page_indices: self.page_indices.raw,
                last_page_len: self.last_page_len.raw,
            },
        )
    }

    fn replay(&self) -> Result<()> {
        self.graph.validate()?;
        unsafe {
            self.library
                .cuda_graph_launch(self.graph.exec_raw, self.stream.raw)
                .context("launching dSpark attention CUDA graph")
        }
    }
}

#[derive(Clone, Copy)]
struct DsparkPythonAttentionBuffers {
    q: GlmrtDeviceBuffer,
    k_cache: GlmrtDeviceBuffer,
    v_cache: GlmrtDeviceBuffer,
    output: GlmrtDeviceBuffer,
    workspace: GlmrtDeviceBuffer,
    query_lengths: GlmrtDeviceBuffer,
    kv_lengths: GlmrtDeviceBuffer,
    block_tables: GlmrtDeviceBuffer,
    query_offsets: GlmrtDeviceBuffer,
    output_offsets: GlmrtDeviceBuffer,
    query_indptr: GlmrtDeviceBuffer,
    kv_indptr: GlmrtDeviceBuffer,
    page_indices: GlmrtDeviceBuffer,
    last_page_len: GlmrtDeviceBuffer,
}

fn launch_python_attention(
    cuda_stream: *mut c_void,
    buffers: &DsparkPythonAttentionBuffers,
    config: DsparkPagedAttentionBenchConfig,
    total_physical_pages: usize,
    max_pages_per_request: usize,
    function: &str,
) -> Result<()> {
    let device_buffers = [
        python_buffer("q", buffers.q),
        python_buffer("k_cache", buffers.k_cache),
        python_buffer("v_cache", buffers.v_cache),
        python_buffer("output", buffers.output),
        python_buffer("workspace", buffers.workspace),
        python_buffer("query_lengths", buffers.query_lengths),
        python_buffer("kv_lengths", buffers.kv_lengths),
        python_buffer("block_tables", buffers.block_tables),
        python_buffer("query_offsets", buffers.query_offsets),
        python_buffer("output_offsets", buffers.output_offsets),
        python_buffer("query_indptr", buffers.query_indptr),
        python_buffer("kv_indptr", buffers.kv_indptr),
        python_buffer("page_indices", buffers.page_indices),
        python_buffer("last_page_len", buffers.last_page_len),
    ];
    let kwargs = [
        ("layers", PythonKernelArg::Usize(config.layers)),
        (
            "active_requests",
            PythonKernelArg::Usize(config.active_requests),
        ),
        ("query_rows", PythonKernelArg::Usize(config.query_rows)),
        ("total_pages", PythonKernelArg::Usize(total_physical_pages)),
        ("page_size", PythonKernelArg::Usize(config.page_size)),
        (
            "max_pages_per_request",
            PythonKernelArg::Usize(max_pages_per_request),
        ),
        ("heads", PythonKernelArg::Usize(DSPARK_ATTENTION_HEADS)),
        (
            "head_dim",
            PythonKernelArg::Usize(DSPARK_ATTENTION_HEAD_DIM),
        ),
        (
            "cache_dtype",
            PythonKernelArg::Str(config.kv_storage.label()),
        ),
    ];
    launch_python_graph_capture(PythonGraphCaptureLaunch {
        module: "dspark_capture",
        function,
        cuda_stream,
        buffers: &device_buffers,
        kwargs: &kwargs,
    })
}

fn python_buffer(name: &'static str, buffer: GlmrtDeviceBuffer) -> PythonDeviceBufferArg<'static> {
    PythonDeviceBufferArg {
        name,
        ptr: buffer.ptr,
        bytes: buffer.bytes,
        device_id: buffer.device_id,
        flags: buffer.flags,
    }
}

pub(super) struct DsparkDeviceBuffer {
    pub(super) library: &'static NativeLibrary,
    pub(super) raw: GlmrtDeviceBuffer,
}

impl DsparkDeviceBuffer {
    pub(super) fn new(library: &'static NativeLibrary, bytes: usize, label: &str) -> Result<Self> {
        anyhow::ensure!(bytes > 0, "{label} requires non-zero bytes");
        let mut raw = library
            .alloc_device_buffer(bytes)
            .with_context(|| format!("allocating {label}"))?;
        if raw.ptr.is_null() || raw.bytes < bytes {
            let allocated = raw.bytes;
            let _ = library.free_device_buffer(&mut raw);
            anyhow::bail!("{label} allocated {allocated} bytes, expected at least {bytes}");
        }
        Ok(Self { library, raw })
    }
}

impl Drop for DsparkDeviceBuffer {
    fn drop(&mut self) {
        if !self.raw.ptr.is_null() {
            let _ = self.library.free_device_buffer(&mut self.raw);
        }
    }
}

pub(super) struct DsparkCudaStream {
    pub(super) library: &'static NativeLibrary,
    pub(super) raw: *mut c_void,
}

impl DsparkCudaStream {
    pub(super) fn create(library: &'static NativeLibrary) -> Result<Self> {
        let raw = library
            .cuda_stream_create()
            .context("creating dSpark attention CUDA stream")?;
        anyhow::ensure!(!raw.is_null(), "dSpark CUDA stream is null");
        Ok(Self { library, raw })
    }

    pub(super) fn synchronize(&self) -> Result<()> {
        unsafe {
            self.library
                .cuda_stream_synchronize(self.raw)
                .context("synchronizing dSpark attention CUDA stream")
        }
    }
}

impl Drop for DsparkCudaStream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = self.library.cuda_stream_destroy(self.raw);
            }
            self.raw = std::ptr::null_mut();
        }
    }
}

pub(super) struct DsparkCudaEvent {
    pub(super) library: &'static NativeLibrary,
    pub(super) raw: *mut c_void,
}

impl DsparkCudaEvent {
    pub(super) fn create(library: &'static NativeLibrary) -> Result<Self> {
        let raw = library
            .cuda_event_create()
            .context("creating dSpark attention CUDA event")?;
        anyhow::ensure!(!raw.is_null(), "dSpark CUDA event is null");
        Ok(Self { library, raw })
    }
}

impl Drop for DsparkCudaEvent {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = self.library.cuda_event_destroy(self.raw);
            }
            self.raw = std::ptr::null_mut();
        }
    }
}

pub(super) struct DsparkCudaGraph {
    pub(super) library: &'static NativeLibrary,
    pub(super) graph_raw: *mut c_void,
    pub(super) exec_raw: *mut c_void,
    pub(super) node_count: usize,
    pub(super) kernel_node_count: usize,
    pub(super) memcpy_node_count: usize,
    pub(super) memset_node_count: usize,
}

impl DsparkCudaGraph {
    pub(super) fn new(
        library: &'static NativeLibrary,
        capture: GlmrtCudaGraphCaptureInfo,
    ) -> Result<Self> {
        anyhow::ensure!(
            !capture.graph.is_null(),
            "dSpark captured CUDA graph is null"
        );
        anyhow::ensure!(
            !capture.graph_exec.is_null(),
            "dSpark captured CUDA graph exec is null"
        );
        Ok(Self {
            library,
            graph_raw: capture.graph,
            exec_raw: capture.graph_exec,
            node_count: capture.node_count,
            kernel_node_count: capture.kernel_node_count,
            memcpy_node_count: capture.memcpy_node_count,
            memset_node_count: capture.memset_node_count,
        })
    }

    pub(super) fn validate(&self) -> Result<()> {
        self.validate_min_kernel_nodes(1)
    }

    pub(super) fn validate_min_kernel_nodes(&self, minimum: usize) -> Result<()> {
        anyhow::ensure!(
            self.node_count > 0,
            "dSpark captured CUDA graph has no nodes"
        );
        anyhow::ensure!(
            self.kernel_node_count >= minimum,
            "dSpark captured CUDA graph has {} kernel nodes, expected at least {minimum}",
            self.kernel_node_count,
        );
        let classified = self
            .kernel_node_count
            .checked_add(self.memcpy_node_count)
            .and_then(|nodes| nodes.checked_add(self.memset_node_count))
            .context("dSpark captured graph classified node count overflow")?;
        anyhow::ensure!(
            classified <= self.node_count,
            "dSpark captured graph classifies {classified} nodes but has {} total",
            self.node_count
        );
        Ok(())
    }
}

impl Drop for DsparkCudaGraph {
    fn drop(&mut self) {
        unsafe {
            if !self.exec_raw.is_null() {
                let _ = self.library.cuda_graph_exec_destroy(self.exec_raw);
                self.exec_raw = std::ptr::null_mut();
            }
            if !self.graph_raw.is_null() {
                let _ = self.library.cuda_graph_destroy(self.graph_raw);
                self.graph_raw = std::ptr::null_mut();
            }
        }
    }
}

fn validate_config(config: DsparkPagedAttentionBenchConfig) -> Result<()> {
    anyhow::ensure!(
        (1..=DSPARK_ATTENTION_LAYERS).contains(&config.layers),
        "dSpark attention layer count must be between 1 and {DSPARK_ATTENTION_LAYERS}"
    );
    anyhow::ensure!(
        matches!(config.active_requests, 1 | 2 | 4),
        "dSpark attention active request bucket must be 1, 2, or 4"
    );
    anyhow::ensure!(
        matches!(config.query_rows, 8 | 16),
        "dSpark attention query rows must be 8 or 16"
    );
    anyhow::ensure!(config.context_tokens > 0, "dSpark context must be positive");
    anyhow::ensure!(
        config.kv_capacity_tokens > 0,
        "dSpark KV capacity must be positive"
    );
    anyhow::ensure!(
        matches!(config.page_size, 16 | 32 | 64 | 128),
        "dSpark attention page size must be 16, 32, 64, or 128"
    );
    anyhow::ensure!(
        config.iterations > 0 && config.repeats > 0,
        "dSpark attention benchmark iterations and repeats must be positive"
    );
    Ok(())
}

pub(super) fn timing_summary(mut values: Vec<f64>) -> Result<DsparkPagedAttentionTiming> {
    anyhow::ensure!(!values.is_empty(), "dSpark timing sample is empty");
    values.sort_by(f64::total_cmp);
    let p90_index = ((values.len() * 9).div_ceil(10)).saturating_sub(1);
    Ok(DsparkPagedAttentionTiming {
        min: values[0],
        median: values[values.len() / 2],
        p90: values[p90_index],
        max: values[values.len() - 1],
    })
}

fn checked_mul(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .with_context(|| format!("{label} overflow"))
}

fn div_ceil(value: usize, divisor: usize) -> usize {
    value.div_ceil(divisor)
}

fn as_bytes<T>(values: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

#[cfg(test)]
mod tests {
    use super::{timing_summary, validate_config, DsparkPagedAttentionBenchConfig};
    use crate::commands::real_full::dspark_kv::DsparkKvStorage;

    fn config() -> DsparkPagedAttentionBenchConfig {
        DsparkPagedAttentionBenchConfig {
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
        }
    }

    #[test]
    fn accepts_only_captured_dspark_shapes() {
        validate_config(config()).unwrap();

        let mut unsupported = config();
        unsupported.active_requests = 3;
        assert!(validate_config(unsupported).is_err());
        unsupported = config();
        unsupported.query_rows = 7;
        assert!(validate_config(unsupported).is_err());
        unsupported = config();
        unsupported.page_size = 8;
        assert!(validate_config(unsupported).is_err());
    }

    #[test]
    fn reports_sorted_replay_timing_quantiles() {
        let timing = timing_summary(vec![4.0, 1.0, 3.0, 2.0, 5.0]).unwrap();
        assert_eq!(timing.min, 1.0);
        assert_eq!(timing.median, 3.0);
        assert_eq!(timing.p90, 5.0);
        assert_eq!(timing.max, 5.0);
    }
}
