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

pub(super) const DSPARK_QUERY_HIDDEN: usize = 6_144;
pub(super) const DSPARK_QUERY_VOCAB: usize = 154_880;

#[derive(Clone, Copy, Debug)]
pub(super) struct DsparkQueryResidentWeights {
    pub(super) embedding: GlmrtDeviceBuffer,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct DsparkQueryBenchConfig {
    pub(super) active_requests: usize,
    pub(super) query_rows: usize,
    /// Synthetic mask-token rows after the anchor row. This is always
    /// `query_rows - 1`; it is not necessarily the number of predictions
    /// harvested from the trained block.
    pub(super) mask_tokens: usize,
    pub(super) mask_token_id: usize,
    pub(super) warmup: usize,
    pub(super) iterations: usize,
    pub(super) repeats: usize,
    pub(super) seed: i64,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct DsparkQueryGraphReport {
    backend: &'static str,
    active_requests: usize,
    query_rows_per_request: usize,
    mask_tokens_per_request: usize,
    mask_token_id: usize,
    initial_anchor_tokens: Vec<u32>,
    dynamic_anchor_tokens: Vec<u32>,
    query_output_bytes: u64,
    rust_owned_mutable_bytes: u64,
    referenced_embedding_bytes: u64,
    graph_nodes: usize,
    graph_kernel_nodes: usize,
    graph_memcpy_nodes: usize,
    graph_memset_nodes: usize,
    eager_reference_exact: bool,
    replay_reference_exact: bool,
    eager_replay_exact: bool,
    dynamic_reference_exact: bool,
    dynamic_anchor_changes_output: bool,
    dynamic_changed_bytes: usize,
    restored_replay_exact: bool,
    warmup: usize,
    iterations: usize,
    repeats: usize,
    gpu_ms_per_query_replay: DsparkPagedAttentionTiming,
    host_ms_per_query_replay: DsparkPagedAttentionTiming,
    request_major_anchor_mask_layout: bool,
    body_input_compatible: bool,
    cold_capture_python_calls: usize,
    hot_replay_python_calls: usize,
    serving_dispatch_enabled: bool,
}

pub(super) fn benchmark_dspark_query_graph(
    weights: DsparkQueryResidentWeights,
    config: DsparkQueryBenchConfig,
) -> Result<DsparkQueryGraphReport> {
    let graph = DsparkQueryGraph::capture(weights, config)?;
    let initial_reference = graph.reference_output(&graph.initial_anchor_tokens)?;
    let eager_output = graph.read_output()?;
    let eager_reference_exact = eager_output == initial_reference;
    anyhow::ensure!(
        eager_reference_exact,
        "dSpark query eager output differs from direct resident rows"
    );

    graph.replay()?;
    graph.stream.synchronize()?;
    let replay_output = graph.read_output()?;
    let replay_reference_exact = replay_output == initial_reference;
    let eager_replay_exact = replay_output == eager_output;
    anyhow::ensure!(
        replay_reference_exact && eager_replay_exact,
        "dSpark query graph replay differs from its eager/reference output"
    );

    graph.set_anchor_tokens(&graph.dynamic_anchor_tokens)?;
    graph.replay()?;
    graph.stream.synchronize()?;
    let dynamic_output = graph.read_output()?;
    let dynamic_reference = graph.reference_output(&graph.dynamic_anchor_tokens)?;
    let dynamic_reference_exact = dynamic_output == dynamic_reference;
    let dynamic_changed_bytes = byte_mismatch_count(&eager_output, &dynamic_output);
    let dynamic_anchor_changes_output = dynamic_changed_bytes > 0;
    anyhow::ensure!(
        dynamic_reference_exact && dynamic_anchor_changes_output,
        "dSpark query graph ignored or misapplied changed anchors"
    );

    graph.set_anchor_tokens(&graph.initial_anchor_tokens)?;
    graph.replay()?;
    graph.stream.synchronize()?;
    let restored_replay_exact = graph.read_output()? == eager_output;
    anyhow::ensure!(
        restored_replay_exact,
        "dSpark query graph did not restore its initial output"
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
                .context("recording dSpark query benchmark start event")?;
        }
        let host_started = Instant::now();
        for _ in 0..config.iterations {
            graph.replay()?;
        }
        unsafe {
            graph
                .library
                .cuda_event_record(end_event.raw, graph.stream.raw)
                .context("recording dSpark query benchmark end event")?;
            graph
                .library
                .cuda_event_synchronize(end_event.raw)
                .context("waiting for dSpark query benchmark end event")?;
        }
        host_samples
            .push(host_started.elapsed().as_secs_f64() * 1_000.0 / config.iterations as f64);
        gpu_samples.push(
            unsafe {
                graph
                    .library
                    .cuda_event_elapsed_ms(start_event.raw, end_event.raw)
                    .context("measuring dSpark query CUDA graph replay")?
            } as f64
                / config.iterations as f64,
        );
    }

    Ok(DsparkQueryGraphReport {
        backend: "native-bf16-resident-embedding-graph",
        active_requests: config.active_requests,
        query_rows_per_request: config.query_rows,
        mask_tokens_per_request: config.mask_tokens,
        mask_token_id: config.mask_token_id,
        initial_anchor_tokens: graph.initial_anchor_tokens.clone(),
        dynamic_anchor_tokens: graph.dynamic_anchor_tokens.clone(),
        query_output_bytes: graph.output.raw.bytes as u64,
        rust_owned_mutable_bytes: graph.rust_owned_mutable_bytes,
        referenced_embedding_bytes: weights.embedding.bytes as u64,
        graph_nodes: graph.graph.node_count,
        graph_kernel_nodes: graph.graph.kernel_node_count,
        graph_memcpy_nodes: graph.graph.memcpy_node_count,
        graph_memset_nodes: graph.graph.memset_node_count,
        eager_reference_exact,
        replay_reference_exact,
        eager_replay_exact,
        dynamic_reference_exact,
        dynamic_anchor_changes_output,
        dynamic_changed_bytes,
        restored_replay_exact,
        warmup: config.warmup,
        iterations: config.iterations,
        repeats: config.repeats,
        gpu_ms_per_query_replay: timing_summary(gpu_samples)?,
        host_ms_per_query_replay: timing_summary(host_samples)?,
        request_major_anchor_mask_layout: true,
        body_input_compatible: true,
        cold_capture_python_calls: 0,
        hot_replay_python_calls: 0,
        serving_dispatch_enabled: false,
    })
}

struct DsparkQueryGraph {
    library: &'static NativeLibrary,
    graph: DsparkCudaGraph,
    stream: DsparkCudaStream,
    token_ids: DsparkDeviceBuffer,
    output: DsparkDeviceBuffer,
    embedding: GlmrtDeviceBuffer,
    config: DsparkQueryBenchConfig,
    rust_owned_mutable_bytes: u64,
    initial_anchor_tokens: Vec<u32>,
    dynamic_anchor_tokens: Vec<u32>,
}

impl DsparkQueryGraph {
    fn capture(
        weights: DsparkQueryResidentWeights,
        config: DsparkQueryBenchConfig,
    ) -> Result<Self> {
        validate_config(config)?;
        validate_weights(weights)?;
        let library = cuda_native_library()?;
        let stream = DsparkCudaStream::create(library)?;
        let total_rows = checked_mul(
            config.active_requests,
            config.query_rows,
            "query total rows",
        )?;
        let token_bytes = checked_mul(total_rows, std::mem::size_of::<u32>(), "query tokens")?;
        let output_bytes = tensor_bytes(total_rows, DSPARK_QUERY_HIDDEN, 2, "query output")?;
        let token_ids = DsparkDeviceBuffer::new(library, token_bytes, "dSpark query token IDs")?;
        let output = DsparkDeviceBuffer::new(library, output_bytes, "dSpark query output")?;
        let rust_owned_mutable_bytes = token_ids
            .raw
            .bytes
            .checked_add(output.raw.bytes)
            .context("dSpark query mutable byte count overflow")?
            as u64;

        let initial_anchor_tokens = (0..config.active_requests)
            .map(|request| {
                normalized_anchor(config.seed + request as i64 * 104_729, config.mask_token_id)
            })
            .collect::<Vec<_>>();
        let dynamic_anchor_tokens = initial_anchor_tokens
            .iter()
            .map(|token| normalized_anchor(*token as i64 + 17, config.mask_token_id))
            .collect::<Vec<_>>();
        let initial_token_ids = query_token_ids(&initial_anchor_tokens, config)?;
        library
            .copy_h2d(token_ids.raw, as_bytes(&initial_token_ids))
            .context("uploading dSpark query token IDs")?;
        launch_embedding(
            library,
            stream.raw,
            weights.embedding,
            token_ids.raw,
            output.raw,
            total_rows,
        )?;
        stream.synchronize()?;

        unsafe {
            library
                .cuda_graph_begin_capture(stream.raw)
                .context("beginning dSpark query CUDA graph capture")?;
        }
        if let Err(error) = launch_embedding(
            library,
            stream.raw,
            weights.embedding,
            token_ids.raw,
            output.raw,
            total_rows,
        ) {
            unsafe {
                let _ = library.cuda_graph_end_capture_retained(stream.raw);
            }
            return Err(error).context("capturing dSpark query embedding lookup");
        }
        let capture = unsafe {
            library
                .cuda_graph_end_capture_retained(stream.raw)
                .context("ending dSpark query CUDA graph capture")?
        };
        let graph = DsparkCudaGraph::new(library, capture)?;
        graph.validate_min_kernel_nodes(1)?;

        Ok(Self {
            library,
            graph,
            stream,
            token_ids,
            output,
            embedding: weights.embedding,
            config,
            rust_owned_mutable_bytes,
            initial_anchor_tokens,
            dynamic_anchor_tokens,
        })
    }

    fn set_anchor_tokens(&self, anchors: &[u32]) -> Result<()> {
        let token_ids = query_token_ids(anchors, self.config)?;
        self.library
            .copy_h2d(self.token_ids.raw, as_bytes(&token_ids))
            .context("uploading dynamic dSpark query anchors")
    }

    fn replay(&self) -> Result<()> {
        self.graph.validate_min_kernel_nodes(1)?;
        unsafe {
            self.library
                .cuda_graph_launch(self.graph.exec_raw, self.stream.raw)
                .context("launching dSpark query CUDA graph")
        }
    }

    fn read_output(&self) -> Result<Vec<u8>> {
        let mut output = vec![0_u8; self.output.raw.bytes];
        self.library
            .copy_d2h(&mut output, self.output.raw)
            .context("reading dSpark query output")?;
        Ok(output)
    }

    fn reference_output(&self, anchors: &[u32]) -> Result<Vec<u8>> {
        anyhow::ensure!(
            anchors.len() == self.config.active_requests,
            "dSpark query reference has {} anchors, expected {}",
            anchors.len(),
            self.config.active_requests
        );
        let mask = read_embedding_row(self.library, self.embedding, self.config.mask_token_id)?;
        let mut output = Vec::with_capacity(self.output.raw.bytes);
        for anchor in anchors {
            output.extend_from_slice(&read_embedding_row(
                self.library,
                self.embedding,
                *anchor as usize,
            )?);
            for _ in 0..self.config.mask_tokens {
                output.extend_from_slice(&mask);
            }
        }
        anyhow::ensure!(
            output.len() == self.output.raw.bytes,
            "dSpark query reference produced {} bytes, expected {}",
            output.len(),
            self.output.raw.bytes
        );
        Ok(output)
    }
}

pub(super) fn launch_embedding(
    library: &'static NativeLibrary,
    stream: *mut c_void,
    embedding: GlmrtDeviceBuffer,
    token_ids: GlmrtDeviceBuffer,
    output: GlmrtDeviceBuffer,
    rows: usize,
) -> Result<()> {
    unsafe {
        library
            .cuda_embedding_lookup_bf16_async(
                embedding,
                token_ids,
                output,
                rows,
                DSPARK_QUERY_VOCAB,
                DSPARK_QUERY_HIDDEN,
                stream,
            )
            .context("launching dSpark query embedding lookup")
    }
}

fn read_embedding_row(
    library: &'static NativeLibrary,
    embedding: GlmrtDeviceBuffer,
    token: usize,
) -> Result<Vec<u8>> {
    anyhow::ensure!(
        token < DSPARK_QUERY_VOCAB,
        "dSpark query token {token} exceeds the embedding vocabulary"
    );
    let row_bytes = DSPARK_QUERY_HIDDEN * 2;
    let offset = token
        .checked_mul(row_bytes)
        .context("dSpark query embedding row offset overflow")?;
    let end = offset
        .checked_add(row_bytes)
        .context("dSpark query embedding row end overflow")?;
    anyhow::ensure!(
        end <= embedding.bytes,
        "dSpark query embedding row exceeds the resident buffer"
    );
    let view = GlmrtDeviceBuffer {
        ptr: unsafe { embedding.ptr.cast::<u8>().add(offset).cast::<c_void>() },
        bytes: row_bytes,
        device_id: embedding.device_id,
        flags: embedding.flags,
    };
    let mut row = vec![0_u8; row_bytes];
    library
        .copy_d2h(&mut row, view)
        .context("reading a dSpark embedding reference row")?;
    Ok(row)
}

pub(super) fn query_token_ids(anchors: &[u32], config: DsparkQueryBenchConfig) -> Result<Vec<u32>> {
    anyhow::ensure!(
        anchors.len() == config.active_requests,
        "dSpark query has {} anchors, expected {}",
        anchors.len(),
        config.active_requests
    );
    anyhow::ensure!(
        anchors
            .iter()
            .all(|token| (*token as usize) < DSPARK_QUERY_VOCAB),
        "dSpark query anchor token is outside the vocabulary: {anchors:?}"
    );
    let mut token_ids = Vec::with_capacity(config.active_requests * config.query_rows);
    for anchor in anchors {
        token_ids.push(*anchor);
        token_ids.extend(std::iter::repeat_n(
            config.mask_token_id as u32,
            config.mask_tokens,
        ));
    }
    anyhow::ensure!(
        token_ids.len() == config.active_requests * config.query_rows,
        "dSpark query layout produced the wrong row count"
    );
    Ok(token_ids)
}

fn normalized_anchor(candidate: i64, mask_token_id: usize) -> u32 {
    let mut token = candidate.rem_euclid(DSPARK_QUERY_VOCAB as i64) as usize;
    if token == mask_token_id {
        token = (token + 1) % DSPARK_QUERY_VOCAB;
    }
    token as u32
}

fn validate_config(config: DsparkQueryBenchConfig) -> Result<()> {
    anyhow::ensure!(
        matches!(config.active_requests, 1 | 2 | 4),
        "dSpark query active requests must be 1, 2, or 4"
    );
    anyhow::ensure!(
        matches!((config.query_rows, config.mask_tokens), (8, 7) | (16, 15)),
        "dSpark query rows/masks must be 8/7 or 16/15"
    );
    anyhow::ensure!(
        config.mask_token_id < DSPARK_QUERY_VOCAB,
        "dSpark query mask token is outside the vocabulary"
    );
    anyhow::ensure!(
        config.iterations > 0 && config.repeats > 0,
        "dSpark query benchmark iterations and repeats must be positive"
    );
    Ok(())
}

fn validate_weights(weights: DsparkQueryResidentWeights) -> Result<()> {
    let expected = DSPARK_QUERY_VOCAB * DSPARK_QUERY_HIDDEN * 2;
    anyhow::ensure!(
        !weights.embedding.ptr.is_null(),
        "dSpark query embedding resident is null"
    );
    anyhow::ensure!(
        weights.embedding.bytes == expected,
        "dSpark query embedding has {} bytes, expected {expected}",
        weights.embedding.bytes
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

fn as_bytes<T>(values: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

#[cfg(test)]
mod tests {
    use super::{query_token_ids, validate_config, DsparkQueryBenchConfig};

    fn config() -> DsparkQueryBenchConfig {
        DsparkQueryBenchConfig {
            active_requests: 2,
            query_rows: 16,
            mask_tokens: 15,
            mask_token_id: 154_856,
            warmup: 2,
            iterations: 10,
            repeats: 3,
            seed: 17,
        }
    }

    #[test]
    fn builds_request_major_bonus_anchor_layout() {
        let tokens = query_token_ids(&[3, 5], config()).unwrap();
        assert_eq!(tokens.len(), 32);
        assert_eq!(tokens[0], 3);
        assert!(tokens[1..16].iter().all(|token| *token == 154_856));
        assert_eq!(tokens[16], 5);
        assert!(tokens[17..32].iter().all(|token| *token == 154_856));
    }

    #[test]
    fn validates_both_checkpoint_query_layouts() {
        validate_config(config()).unwrap();
        validate_config(DsparkQueryBenchConfig {
            query_rows: 8,
            mask_tokens: 7,
            ..config()
        })
        .unwrap();
        assert!(validate_config(DsparkQueryBenchConfig {
            query_rows: 15,
            mask_tokens: 15,
            ..config()
        })
        .is_err());
    }
}
