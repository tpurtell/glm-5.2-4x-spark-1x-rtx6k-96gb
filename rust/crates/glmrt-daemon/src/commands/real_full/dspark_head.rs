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
use crate::python_graph_capture::{
    launch_python_graph_capture, PythonDeviceBufferArg, PythonGraphCaptureLaunch, PythonKernelArg,
};

pub(super) const DSPARK_HEAD_HIDDEN: usize = 6_144;
pub(super) const DSPARK_HEAD_MARKOV_RANK: usize = 256;
pub(super) const DSPARK_HEAD_VOCAB: usize = 154_880;
pub(super) const DSPARK_HEAD_ARGMAX_BLOCK: usize = 512;
pub(super) const DSPARK_HEAD_ARGMAX_BLOCKS: usize =
    DSPARK_HEAD_VOCAB.div_ceil(DSPARK_HEAD_ARGMAX_BLOCK);

#[derive(Clone, Copy, Debug)]
pub(super) struct DsparkHeadResidentWeights {
    pub(super) lm_head: GlmrtDeviceBuffer,
    pub(super) markov_w1: GlmrtDeviceBuffer,
    pub(super) markov_w2: GlmrtDeviceBuffer,
    pub(super) confidence_weight: GlmrtDeviceBuffer,
    pub(super) confidence_bias: GlmrtDeviceBuffer,
    pub(super) resident_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct DsparkHeadBenchConfig {
    pub(super) active_requests: usize,
    pub(super) proposal_tokens: usize,
    pub(super) hidden_rows_per_request: usize,
    pub(super) hidden_start_row: usize,
    pub(super) warmup: usize,
    pub(super) iterations: usize,
    pub(super) repeats: usize,
    pub(super) seed: i64,
    pub(super) initialize_hidden: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct DsparkHeadGraphReport {
    backend: &'static str,
    active_requests: usize,
    proposal_tokens_per_request: usize,
    hidden_rows_per_request: usize,
    hidden_start_row: usize,
    target_verification_rows: usize,
    initial_anchor_tokens: Vec<i64>,
    dynamic_anchor_tokens: Vec<i64>,
    resident_weight_bytes: u64,
    rust_owned_mutable_bytes: u64,
    graph_nodes: usize,
    graph_kernel_nodes: usize,
    graph_memcpy_nodes: usize,
    graph_memset_nodes: usize,
    reference_token_exact: bool,
    reference_confidence_max_abs: f32,
    eager_replay_exact: bool,
    dynamic_anchor_changes_output: bool,
    dynamic_token_changed_bytes: usize,
    dynamic_confidence_changed_bytes: usize,
    restored_replay_exact: bool,
    warmup: usize,
    iterations: usize,
    repeats: usize,
    gpu_ms_per_head_replay: DsparkPagedAttentionTiming,
    host_ms_per_head_replay: DsparkPagedAttentionTiming,
    batched_lm_head: bool,
    sequential_markov: bool,
    deferred_batched_confidence: bool,
    target_lm_head_alias: bool,
    rust_owned_scratch: bool,
    cold_capture_python_calls: usize,
    hot_replay_python_calls: usize,
    serving_dispatch_enabled: bool,
}

pub(super) fn benchmark_dspark_head_graph(
    weights: DsparkHeadResidentWeights,
    config: DsparkHeadBenchConfig,
) -> Result<DsparkHeadGraphReport> {
    let mut graph = DsparkHeadGraph::capture(weights, config)?;
    graph.replay()?;
    graph.stream.synchronize()?;

    let reference_tokens = graph.read_tokens(graph.buffers.reference_tokens)?;
    let reference_confidence = graph.read_confidence(graph.buffers.reference_confidence)?;
    let eager_tokens = graph.read_tokens(graph.buffers.eager_tokens)?;
    let eager_confidence = graph.read_confidence(graph.buffers.eager_confidence)?;
    let replay_tokens = graph.read_tokens(graph.buffers.output_tokens)?;
    let replay_confidence = graph.read_confidence(graph.buffers.output_confidence)?;
    let reference_token_exact = reference_tokens == replay_tokens;
    let reference_confidence_max_abs =
        f32_max_abs_difference(&reference_confidence, &replay_confidence)?;
    anyhow::ensure!(
        reference_token_exact,
        "dSpark retained head changed the official sequential reference tokens"
    );
    anyhow::ensure!(
        reference_confidence_max_abs <= 0.0078125,
        "dSpark retained head changed reference confidence by {reference_confidence_max_abs}"
    );
    let eager_replay_exact = eager_tokens == replay_tokens && eager_confidence == replay_confidence;
    anyhow::ensure!(
        eager_replay_exact,
        "dSpark retained head replay differs from its eager output"
    );

    graph.set_anchor_tokens(&graph.dynamic_anchor_tokens.clone())?;
    graph.replay()?;
    graph.stream.synchronize()?;
    let dynamic_tokens = graph.read_tokens(graph.buffers.output_tokens)?;
    let dynamic_confidence = graph.read_confidence(graph.buffers.output_confidence)?;
    let dynamic_token_changed_bytes = byte_mismatch_count(&eager_tokens, &dynamic_tokens);
    let dynamic_confidence_changed_bytes =
        byte_mismatch_count(&eager_confidence, &dynamic_confidence);
    let dynamic_anchor_changes_output =
        dynamic_token_changed_bytes > 0 || dynamic_confidence_changed_bytes > 0;
    anyhow::ensure!(
        dynamic_anchor_changes_output,
        "dSpark retained head ignored changed anchor tokens"
    );

    graph.set_anchor_tokens(&graph.initial_anchor_tokens.clone())?;
    graph.replay()?;
    graph.stream.synchronize()?;
    let restored_tokens = graph.read_tokens(graph.buffers.output_tokens)?;
    let restored_confidence = graph.read_confidence(graph.buffers.output_confidence)?;
    let restored_replay_exact =
        restored_tokens == eager_tokens && restored_confidence == eager_confidence;
    anyhow::ensure!(
        restored_replay_exact,
        "dSpark retained head did not restore exact output after anchor restoration"
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
                .context("recording dSpark head benchmark start event")?;
        }
        let host_started = Instant::now();
        for _ in 0..config.iterations {
            graph.replay()?;
        }
        unsafe {
            graph
                .library
                .cuda_event_record(end_event.raw, graph.stream.raw)
                .context("recording dSpark head benchmark end event")?;
            graph
                .library
                .cuda_event_synchronize(end_event.raw)
                .context("waiting for dSpark head benchmark end event")?;
        }
        host_samples
            .push(host_started.elapsed().as_secs_f64() * 1_000.0 / config.iterations as f64);
        gpu_samples.push(
            unsafe {
                graph
                    .library
                    .cuda_event_elapsed_ms(start_event.raw, end_event.raw)
                    .context("measuring dSpark head CUDA graph replay")?
            } as f64
                / config.iterations as f64,
        );
    }

    Ok(DsparkHeadGraphReport {
        backend: "fixed-address-bf16-cublas-sequential-markov-fused-add-argmax",
        active_requests: config.active_requests,
        proposal_tokens_per_request: config.proposal_tokens,
        hidden_rows_per_request: config.hidden_rows_per_request,
        hidden_start_row: config.hidden_start_row,
        target_verification_rows: config.active_requests * (config.proposal_tokens + 1),
        initial_anchor_tokens: graph.initial_anchor_tokens.clone(),
        dynamic_anchor_tokens: graph.dynamic_anchor_tokens.clone(),
        resident_weight_bytes: weights.resident_bytes,
        rust_owned_mutable_bytes: graph.rust_owned_mutable_bytes,
        graph_nodes: graph.graph.node_count,
        graph_kernel_nodes: graph.graph.kernel_node_count,
        graph_memcpy_nodes: graph.graph.memcpy_node_count,
        graph_memset_nodes: graph.graph.memset_node_count,
        reference_token_exact,
        reference_confidence_max_abs,
        eager_replay_exact,
        dynamic_anchor_changes_output,
        dynamic_token_changed_bytes,
        dynamic_confidence_changed_bytes,
        restored_replay_exact,
        warmup: config.warmup,
        iterations: config.iterations,
        repeats: config.repeats,
        gpu_ms_per_head_replay: timing_summary(gpu_samples)?,
        host_ms_per_head_replay: timing_summary(host_samples)?,
        batched_lm_head: true,
        sequential_markov: true,
        deferred_batched_confidence: true,
        target_lm_head_alias: true,
        rust_owned_scratch: true,
        cold_capture_python_calls: 2,
        hot_replay_python_calls: 0,
        serving_dispatch_enabled: false,
    })
}

struct DsparkHeadGraph {
    library: &'static NativeLibrary,
    graph: DsparkCudaGraph,
    stream: DsparkCudaStream,
    _owned_buffers: Vec<DsparkDeviceBuffer>,
    buffers: DsparkPythonHeadBuffers,
    config: DsparkHeadBenchConfig,
    rust_owned_mutable_bytes: u64,
    initial_anchor_tokens: Vec<i64>,
    dynamic_anchor_tokens: Vec<i64>,
}

impl DsparkHeadGraph {
    fn capture(weights: DsparkHeadResidentWeights, config: DsparkHeadBenchConfig) -> Result<Self> {
        validate_config(config)?;
        validate_weights(weights)?;
        let library = cuda_native_library()?;
        let stream = DsparkCudaStream::create(library)?;
        let proposal_rows = checked_mul(
            config.active_requests,
            config.proposal_tokens,
            "head proposal rows",
        )?;
        let hidden_rows = checked_mul(
            config.active_requests,
            config.hidden_rows_per_request,
            "head hidden source rows",
        )?;
        let feature_width = DSPARK_HEAD_HIDDEN + DSPARK_HEAD_MARKOV_RANK;
        let hidden_bytes = tensor_bytes(hidden_rows, DSPARK_HEAD_HIDDEN, 2, "head hidden")?;
        let proposal_hidden_bytes =
            tensor_bytes(proposal_rows, DSPARK_HEAD_HIDDEN, 2, "head proposal hidden")?;
        let base_logits_bytes =
            tensor_bytes(proposal_rows, DSPARK_HEAD_VOCAB, 2, "head base logits")?;
        let step_logits_bytes = tensor_bytes(
            config.active_requests,
            DSPARK_HEAD_VOCAB,
            2,
            "head step logits",
        )?;
        let argmax_candidate_bytes = tensor_bytes(
            config.active_requests,
            DSPARK_HEAD_ARGMAX_BLOCKS,
            4,
            "head argmax candidates",
        )?;
        let embedding_steps_bytes = tensor_bytes(
            proposal_rows,
            DSPARK_HEAD_MARKOV_RANK,
            2,
            "head embedding steps",
        )?;
        let token_bytes = checked_mul(
            proposal_rows,
            std::mem::size_of::<i64>(),
            "head token bytes",
        )?;
        let anchor_bytes = checked_mul(
            config.active_requests,
            std::mem::size_of::<i64>(),
            "head anchor bytes",
        )?;
        let confidence_features_bytes =
            tensor_bytes(proposal_rows, feature_width, 2, "head confidence features")?;
        let confidence_bf16_bytes = checked_mul(proposal_rows, 2, "head BF16 confidence bytes")?;
        let confidence_f32_bytes = checked_mul(proposal_rows, 4, "head FP32 confidence bytes")?;

        let mut owned = Vec::new();
        let mut allocate = |bytes, label| -> Result<GlmrtDeviceBuffer> {
            let buffer = DsparkDeviceBuffer::new(library, bytes, label)?;
            let raw = buffer.raw;
            owned.push(buffer);
            Ok(raw)
        };
        let buffers = DsparkPythonHeadBuffers {
            hidden: allocate(hidden_bytes, "dSpark head hidden")?,
            hidden_position_major: allocate(
                proposal_hidden_bytes,
                "dSpark head position-major hidden",
            )?,
            base_logits: allocate(base_logits_bytes, "dSpark head base logits")?,
            markov_logits: allocate(step_logits_bytes, "dSpark head Markov logits")?,
            argmax_candidate_scores: allocate(
                argmax_candidate_bytes,
                "dSpark head argmax candidate scores",
            )?,
            argmax_candidate_tokens: allocate(
                argmax_candidate_bytes,
                "dSpark head argmax candidate tokens",
            )?,
            embedding_steps: allocate(embedding_steps_bytes, "dSpark head embedding steps")?,
            token_steps: allocate(token_bytes, "dSpark head token steps")?,
            confidence_features: allocate(
                confidence_features_bytes,
                "dSpark head confidence features",
            )?,
            confidence_logits: allocate(confidence_bf16_bytes, "dSpark head confidence logits")?,
            confidence_probabilities: allocate(
                confidence_bf16_bytes,
                "dSpark head confidence probabilities",
            )?,
            anchor_tokens: allocate(anchor_bytes, "dSpark head anchor tokens")?,
            output_tokens: allocate(token_bytes, "dSpark head output tokens")?,
            output_confidence: allocate(confidence_f32_bytes, "dSpark head output confidence")?,
            reference_tokens: allocate(token_bytes, "dSpark head reference tokens")?,
            reference_confidence: allocate(
                confidence_f32_bytes,
                "dSpark head reference confidence",
            )?,
            eager_tokens: allocate(token_bytes, "dSpark head eager tokens")?,
            eager_confidence: allocate(confidence_f32_bytes, "dSpark head eager confidence")?,
        };
        let rust_owned_mutable_bytes = owned.iter().try_fold(0_u64, |bytes, buffer| {
            bytes
                .checked_add(buffer.raw.bytes as u64)
                .context("dSpark head mutable byte count overflow")
        })?;

        let initial_anchor_tokens = (0..config.active_requests)
            .map(|request| {
                (config.seed + request as i64 * 104_729).rem_euclid(DSPARK_HEAD_VOCAB as i64)
            })
            .collect::<Vec<_>>();
        let dynamic_anchor_tokens = initial_anchor_tokens
            .iter()
            // Use a deliberately distant, known vocabulary row. Nearby token
            // IDs can have nearly identical Markov embeddings in real
            // checkpoints, which makes a fixed +17 mutation a weak graph
            // liveness probe.
            .map(|token| if *token == 1 { 0 } else { 1 })
            .collect::<Vec<_>>();
        library
            .copy_h2d(buffers.anchor_tokens, as_bytes(&initial_anchor_tokens))
            .context("uploading dSpark head anchor tokens")?;

        launch_python_head(stream.raw, &buffers, weights, config, "prepare_dspark_head")?;
        stream.synchronize()?;

        unsafe {
            library
                .cuda_graph_begin_capture(stream.raw)
                .context("beginning dSpark head CUDA graph capture")?;
        }
        if let Err(error) =
            launch_python_head(stream.raw, &buffers, weights, config, "capture_dspark_head")
        {
            unsafe {
                let _ = library.cuda_graph_end_capture_retained(stream.raw);
            }
            return Err(error).context("capturing dSpark head");
        }
        let capture = unsafe {
            library
                .cuda_graph_end_capture_retained(stream.raw)
                .context("ending dSpark head CUDA graph capture")?
        };
        let graph = DsparkCudaGraph::new(library, capture)?;
        graph.validate()?;

        Ok(Self {
            library,
            graph,
            stream,
            _owned_buffers: owned,
            buffers,
            config,
            rust_owned_mutable_bytes,
            initial_anchor_tokens,
            dynamic_anchor_tokens,
        })
    }

    fn set_anchor_tokens(&mut self, tokens: &[i64]) -> Result<()> {
        anyhow::ensure!(
            tokens.len() == self.config.active_requests
                && tokens
                    .iter()
                    .all(|token| *token >= 0 && (*token as usize) < DSPARK_HEAD_VOCAB),
            "dSpark head anchor update is invalid: {tokens:?}"
        );
        self.library
            .copy_h2d(self.buffers.anchor_tokens, as_bytes(tokens))
            .context("uploading dynamic dSpark head anchor tokens")
    }

    fn replay(&self) -> Result<()> {
        self.graph.validate()?;
        unsafe {
            self.library
                .cuda_graph_launch(self.graph.exec_raw, self.stream.raw)
                .context("launching dSpark head CUDA graph")
        }
    }

    fn read_tokens(&self, buffer: GlmrtDeviceBuffer) -> Result<Vec<u8>> {
        let bytes = checked_mul(
            self.config.active_requests * self.config.proposal_tokens,
            std::mem::size_of::<i64>(),
            "head token readback",
        )?;
        self.read_buffer(buffer, bytes, "reading dSpark head tokens")
    }

    fn read_confidence(&self, buffer: GlmrtDeviceBuffer) -> Result<Vec<u8>> {
        let bytes = checked_mul(
            self.config.active_requests * self.config.proposal_tokens,
            std::mem::size_of::<f32>(),
            "head confidence readback",
        )?;
        self.read_buffer(buffer, bytes, "reading dSpark head confidence")
    }

    fn read_buffer(
        &self,
        buffer: GlmrtDeviceBuffer,
        bytes: usize,
        label: &'static str,
    ) -> Result<Vec<u8>> {
        let mut output = vec![0_u8; bytes];
        self.library.copy_d2h(&mut output, buffer).context(label)?;
        Ok(output)
    }
}

#[derive(Clone, Copy)]
pub(super) struct DsparkPythonHeadBuffers {
    pub(super) hidden: GlmrtDeviceBuffer,
    pub(super) hidden_position_major: GlmrtDeviceBuffer,
    pub(super) base_logits: GlmrtDeviceBuffer,
    pub(super) markov_logits: GlmrtDeviceBuffer,
    pub(super) argmax_candidate_scores: GlmrtDeviceBuffer,
    pub(super) argmax_candidate_tokens: GlmrtDeviceBuffer,
    pub(super) embedding_steps: GlmrtDeviceBuffer,
    pub(super) token_steps: GlmrtDeviceBuffer,
    pub(super) confidence_features: GlmrtDeviceBuffer,
    pub(super) confidence_logits: GlmrtDeviceBuffer,
    pub(super) confidence_probabilities: GlmrtDeviceBuffer,
    pub(super) anchor_tokens: GlmrtDeviceBuffer,
    pub(super) output_tokens: GlmrtDeviceBuffer,
    pub(super) output_confidence: GlmrtDeviceBuffer,
    pub(super) reference_tokens: GlmrtDeviceBuffer,
    pub(super) reference_confidence: GlmrtDeviceBuffer,
    pub(super) eager_tokens: GlmrtDeviceBuffer,
    pub(super) eager_confidence: GlmrtDeviceBuffer,
}

pub(super) fn launch_python_head(
    cuda_stream: *mut c_void,
    buffers: &DsparkPythonHeadBuffers,
    weights: DsparkHeadResidentWeights,
    config: DsparkHeadBenchConfig,
    function: &str,
) -> Result<()> {
    let device_buffers = [
        python_buffer("hidden", buffers.hidden),
        python_buffer("hidden_position_major", buffers.hidden_position_major),
        python_buffer("base_logits", buffers.base_logits),
        python_buffer("markov_logits", buffers.markov_logits),
        python_buffer("argmax_candidate_scores", buffers.argmax_candidate_scores),
        python_buffer("argmax_candidate_tokens", buffers.argmax_candidate_tokens),
        python_buffer("embedding_steps", buffers.embedding_steps),
        python_buffer("token_steps", buffers.token_steps),
        python_buffer("confidence_features", buffers.confidence_features),
        python_buffer("confidence_logits", buffers.confidence_logits),
        python_buffer("confidence_probabilities", buffers.confidence_probabilities),
        python_buffer("anchor_tokens", buffers.anchor_tokens),
        python_buffer("output_tokens", buffers.output_tokens),
        python_buffer("output_confidence", buffers.output_confidence),
        python_buffer("reference_tokens", buffers.reference_tokens),
        python_buffer("reference_confidence", buffers.reference_confidence),
        python_buffer("eager_tokens", buffers.eager_tokens),
        python_buffer("eager_confidence", buffers.eager_confidence),
        python_buffer("lm_head", weights.lm_head),
        python_buffer("markov_w1", weights.markov_w1),
        python_buffer("markov_w2", weights.markov_w2),
        python_buffer("confidence_weight", weights.confidence_weight),
        python_buffer("confidence_bias", weights.confidence_bias),
    ];
    let kwargs = [
        (
            "active_requests",
            PythonKernelArg::Usize(config.active_requests),
        ),
        (
            "proposal_tokens",
            PythonKernelArg::Usize(config.proposal_tokens),
        ),
        (
            "hidden_rows_per_request",
            PythonKernelArg::Usize(config.hidden_rows_per_request),
        ),
        (
            "hidden_start_row",
            PythonKernelArg::Usize(config.hidden_start_row),
        ),
        ("hidden_size", PythonKernelArg::Usize(DSPARK_HEAD_HIDDEN)),
        (
            "markov_rank",
            PythonKernelArg::Usize(DSPARK_HEAD_MARKOV_RANK),
        ),
        ("vocab_size", PythonKernelArg::Usize(DSPARK_HEAD_VOCAB)),
        ("seed", PythonKernelArg::I64(config.seed)),
        (
            "initialize_hidden",
            PythonKernelArg::Bool(config.initialize_hidden),
        ),
    ];
    launch_python_graph_capture(PythonGraphCaptureLaunch {
        module: "dspark_head_capture",
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

fn validate_config(config: DsparkHeadBenchConfig) -> Result<()> {
    anyhow::ensure!(
        matches!(config.active_requests, 1 | 2 | 4),
        "dSpark head active request bucket must be 1, 2, or 4"
    );
    anyhow::ensure!(
        matches!(config.proposal_tokens, 7 | 8 | 15),
        "dSpark head native prediction count must be 7, 8, or 15"
    );
    let hidden_end_row = config
        .hidden_start_row
        .checked_add(config.proposal_tokens)
        .context("dSpark head hidden source range overflow")?;
    anyhow::ensure!(
        hidden_end_row <= config.hidden_rows_per_request,
        "dSpark head proposal rows exceed the hidden source"
    );
    anyhow::ensure!(
        config.iterations > 0 && config.repeats > 0,
        "dSpark head benchmark iterations and repeats must be positive"
    );
    Ok(())
}

fn validate_weights(weights: DsparkHeadResidentWeights) -> Result<()> {
    validate_buffer(
        "LM head",
        weights.lm_head,
        DSPARK_HEAD_VOCAB * DSPARK_HEAD_HIDDEN * 2,
    )?;
    validate_buffer(
        "Markov W1",
        weights.markov_w1,
        DSPARK_HEAD_VOCAB * DSPARK_HEAD_MARKOV_RANK * 2,
    )?;
    validate_buffer(
        "Markov W2",
        weights.markov_w2,
        DSPARK_HEAD_VOCAB * DSPARK_HEAD_MARKOV_RANK * 2,
    )?;
    validate_buffer(
        "confidence weight",
        weights.confidence_weight,
        (DSPARK_HEAD_HIDDEN + DSPARK_HEAD_MARKOV_RANK) * 2,
    )?;
    validate_buffer("confidence bias", weights.confidence_bias, 2)
}

fn validate_buffer(label: &str, buffer: GlmrtDeviceBuffer, expected_bytes: usize) -> Result<()> {
    anyhow::ensure!(!buffer.ptr.is_null(), "dSpark head {label} buffer is null");
    anyhow::ensure!(
        buffer.bytes == expected_bytes,
        "dSpark head {label} has {} bytes, expected {expected_bytes}",
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

fn f32_max_abs_difference(left: &[u8], right: &[u8]) -> Result<f32> {
    anyhow::ensure!(
        left.len() == right.len() && left.len() % std::mem::size_of::<f32>() == 0,
        "dSpark confidence byte lengths are invalid"
    );
    let mut max_abs = 0.0_f32;
    for (left, right) in left.chunks_exact(4).zip(right.chunks_exact(4)) {
        let left = f32::from_le_bytes(left.try_into().expect("four-byte confidence chunk"));
        let right = f32::from_le_bytes(right.try_into().expect("four-byte confidence chunk"));
        anyhow::ensure!(
            left.is_finite() && right.is_finite(),
            "dSpark head confidence contains a non-finite value"
        );
        max_abs = max_abs.max((left - right).abs());
    }
    Ok(max_abs)
}

fn as_bytes<T>(values: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

#[cfg(test)]
mod tests {
    use super::{validate_config, DsparkHeadBenchConfig};

    fn config() -> DsparkHeadBenchConfig {
        DsparkHeadBenchConfig {
            active_requests: 4,
            proposal_tokens: 15,
            hidden_rows_per_request: 15,
            hidden_start_row: 0,
            warmup: 2,
            iterations: 10,
            repeats: 3,
            seed: 17,
            initialize_hidden: true,
        }
    }

    #[test]
    fn accepts_production_head_buckets() {
        for active_requests in [1, 2, 4] {
            for proposal_tokens in [7, 8, 15] {
                validate_config(DsparkHeadBenchConfig {
                    active_requests,
                    proposal_tokens,
                    ..config()
                })
                .unwrap();
            }
        }
        validate_config(DsparkHeadBenchConfig {
            hidden_rows_per_request: 16,
            hidden_start_row: 1,
            ..config()
        })
        .unwrap();
    }

    #[test]
    fn rejects_head_shape_and_benchmark_mismatches() {
        let mut invalid = config();
        invalid.active_requests = 3;
        assert!(validate_config(invalid).is_err());
        invalid = config();
        invalid.proposal_tokens = 14;
        assert!(validate_config(invalid).is_err());
        invalid = config();
        invalid.iterations = 0;
        assert!(validate_config(invalid).is_err());
        invalid = config();
        invalid.hidden_rows_per_request = 15;
        invalid.hidden_start_row = 1;
        assert!(validate_config(invalid).is_err());
    }
}
