use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::{Context, Result};
use glmrt_ffi::{GlmrtDeviceBuffer, NativeLibrary};
use serde::Serialize;

use super::coordinator_kernels::{cuda_native_library, device_buffer_byte_view, DeviceBf16Output};
use super::dspark_attention::{
    timing_summary, DsparkCudaEvent, DsparkCudaGraph, DsparkCudaStream, DsparkDeviceBuffer,
    DsparkPagedAttentionTiming,
};
use super::dspark_body::{
    launch_python_body, DsparkBodyBenchConfig, DsparkBodyResidentWeights, DsparkPythonBodyBuffers,
    DSPARK_BODY_ATTENTION_WIDTH, DSPARK_BODY_HEADS, DSPARK_BODY_HEAD_DIM, DSPARK_BODY_HIDDEN,
    DSPARK_BODY_INTERMEDIATE, DSPARK_BODY_LAYERS, DSPARK_BODY_WORKSPACE_BYTES,
};
use super::dspark_head::{
    launch_python_head, DsparkHeadBenchConfig, DsparkHeadResidentWeights, DsparkPythonHeadBuffers,
    DSPARK_HEAD_ARGMAX_BLOCKS, DSPARK_HEAD_HIDDEN, DSPARK_HEAD_MARKOV_RANK, DSPARK_HEAD_VOCAB,
};
use super::dspark_kv::{
    i32_buffer_bytes, DsparkKvStorage, DsparkPagedKvMetadata, DsparkPagedKvMetadataBuffers,
};
use super::dspark_query::{
    launch_embedding, query_token_ids, DsparkQueryBenchConfig, DsparkQueryResidentWeights,
    DSPARK_QUERY_HIDDEN, DSPARK_QUERY_VOCAB,
};
use super::dspark_update::{
    launch_python_update, DsparkPythonUpdateBuffers, DsparkUpdateBenchConfig,
    DsparkUpdateResidentWeights, DSPARK_UPDATE_ATTENTION_WIDTH, DSPARK_UPDATE_HEADS,
    DSPARK_UPDATE_HEAD_DIM, DSPARK_UPDATE_HIDDEN, DSPARK_UPDATE_LAYERS,
    DSPARK_UPDATE_TARGET_FEATURES,
};

#[derive(Clone, Copy, Debug)]
pub(super) struct DsparkStaticResidentWeights {
    pub(super) query: DsparkQueryResidentWeights,
    pub(super) update: DsparkUpdateResidentWeights,
    pub(super) body: DsparkBodyResidentWeights,
    pub(super) head: DsparkHeadResidentWeights,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct DsparkStaticBenchConfig {
    pub(super) draft_layers: usize,
    pub(super) active_requests: usize,
    pub(super) query_rows: usize,
    pub(super) proposal_tokens: usize,
    pub(super) proposal_start_row: usize,
    pub(super) accepted_rows_per_request: usize,
    pub(super) context_tokens: usize,
    pub(super) kv_capacity_tokens: usize,
    pub(super) allocate_full_kv_capacity: bool,
    pub(super) page_size: usize,
    pub(super) kv_storage: DsparkKvStorage,
    pub(super) mask_token_id: usize,
    pub(super) warmup: usize,
    pub(super) iterations: usize,
    pub(super) repeats: usize,
    pub(super) seed: i64,
}

pub(super) struct DsparkDraftStep {
    pub(super) context_tokens: usize,
    pub(super) committed_rows: usize,
    pub(super) anchor_token: usize,
    pub(super) proposal_token_ids: Vec<usize>,
    pub(super) conditional_confidence: Vec<f32>,
    pub(super) update_ms: f64,
    pub(super) suffix_ms: f64,
    pub(super) readback_ms: f64,
    pub(super) total_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct DsparkStaticGraphReport {
    backend: &'static str,
    kv_storage: DsparkKvStorage,
    kv_element_bytes: usize,
    active_requests: usize,
    query_rows_per_request: usize,
    proposal_tokens_per_request: usize,
    accepted_rows_per_request: usize,
    packed_update_rows: usize,
    context_tokens_before_update: usize,
    context_tokens_after_update: usize,
    body_kv_length: usize,
    page_size: usize,
    total_physical_pages: usize,
    max_pages_per_request: usize,
    shared_kv_bytes: u64,
    flashinfer_metadata_bytes: u64,
    rust_owned_mutable_bytes: u64,
    initial_anchor_tokens: Vec<u32>,
    dynamic_anchor_tokens: Vec<u32>,
    update_graph_nodes: usize,
    update_graph_kernel_nodes: usize,
    update_graph_memcpy_nodes: usize,
    update_graph_memset_nodes: usize,
    suffix_graph_nodes: usize,
    suffix_graph_kernel_nodes: usize,
    suffix_graph_memcpy_nodes: usize,
    suffix_graph_memset_nodes: usize,
    eager_token_exact: bool,
    eager_confidence_max_abs: f32,
    dynamic_anchor_changes_output: bool,
    dynamic_token_changed_bytes: usize,
    dynamic_confidence_changed_bytes: usize,
    restored_token_exact: bool,
    restored_confidence_max_abs: f32,
    warmup: usize,
    iterations: usize,
    repeats: usize,
    gpu_ms_per_update_replay: DsparkPagedAttentionTiming,
    host_ms_per_update_replay: DsparkPagedAttentionTiming,
    gpu_ms_per_suffix_replay: DsparkPagedAttentionTiming,
    host_ms_per_suffix_replay: DsparkPagedAttentionTiming,
    gpu_ms_per_full_step: DsparkPagedAttentionTiming,
    host_ms_per_full_step: DsparkPagedAttentionTiming,
    exact_row_update_graph: bool,
    fixed_query_body_head_suffix_graph: bool,
    shared_update_body_kv: bool,
    query_output_aliases_body_input: bool,
    body_output_is_head_source: bool,
    strided_proposal_view: bool,
    cold_capture_python_calls: usize,
    hot_replay_python_calls: usize,
    serving_dispatch_enabled: bool,
}

pub(super) fn benchmark_dspark_static_graph(
    weights: DsparkStaticResidentWeights,
    config: DsparkStaticBenchConfig,
) -> Result<DsparkStaticGraphReport> {
    let mut executor = DsparkStaticExecutor::capture(weights, config)?;

    executor.replay_full()?;
    executor.stream.synchronize()?;
    let eager_tokens = executor.read_tokens(executor.head_buffers.eager_tokens)?;
    let eager_confidence = executor.read_confidence(executor.head_buffers.eager_confidence)?;
    let replay_tokens = executor.read_tokens(executor.head_buffers.output_tokens)?;
    let replay_confidence = executor.read_confidence(executor.head_buffers.output_confidence)?;
    let eager_token_exact = eager_tokens == replay_tokens;
    let eager_confidence_max_abs = f32_max_abs_difference(&eager_confidence, &replay_confidence)?;
    anyhow::ensure!(
        eager_token_exact && eager_confidence_max_abs <= 0.0078125,
        "dSpark static replay changed eager output: token_exact={eager_token_exact} confidence_max_abs={eager_confidence_max_abs}"
    );

    executor.set_anchor_tokens(&executor.dynamic_anchor_tokens.clone())?;
    executor.replay_full()?;
    executor.stream.synchronize()?;
    let dynamic_tokens = executor.read_tokens(executor.head_buffers.output_tokens)?;
    let dynamic_confidence = executor.read_confidence(executor.head_buffers.output_confidence)?;
    let dynamic_token_changed_bytes = byte_mismatch_count(&replay_tokens, &dynamic_tokens);
    let dynamic_confidence_changed_bytes =
        byte_mismatch_count(&replay_confidence, &dynamic_confidence);
    let dynamic_anchor_changes_output =
        dynamic_token_changed_bytes > 0 || dynamic_confidence_changed_bytes > 0;
    anyhow::ensure!(
        dynamic_anchor_changes_output,
        "dSpark static graph ignored changed anchor tokens"
    );

    executor.set_anchor_tokens(&executor.initial_anchor_tokens.clone())?;
    executor.replay_full()?;
    executor.stream.synchronize()?;
    let restored_tokens = executor.read_tokens(executor.head_buffers.output_tokens)?;
    let restored_confidence = executor.read_confidence(executor.head_buffers.output_confidence)?;
    let restored_token_exact = restored_tokens == replay_tokens;
    let restored_confidence_max_abs =
        f32_max_abs_difference(&replay_confidence, &restored_confidence)?;
    anyhow::ensure!(
        restored_token_exact && restored_confidence_max_abs <= 0.0078125,
        "dSpark static graph did not restore output: token_exact={restored_token_exact} confidence_max_abs={restored_confidence_max_abs}"
    );

    for _ in 0..config.warmup {
        executor.replay_full()?;
    }
    executor.stream.synchronize()?;
    let (gpu_update, host_update) = executor.measure(ReplayPart::Update)?;
    let (gpu_suffix, host_suffix) = executor.measure(ReplayPart::Suffix)?;
    let (gpu_full, host_full) = executor.measure(ReplayPart::Full)?;

    Ok(DsparkStaticGraphReport {
        backend: match config.kv_storage {
            DsparkKvStorage::Bf16 => "fixed-address-bf16-update-plus-query-body-head",
            DsparkKvStorage::Fp8 => {
                "fixed-address-fp8-kv-update-plus-query-body-flashinfer-fa2-head"
            }
        },
        kv_storage: config.kv_storage,
        kv_element_bytes: config.kv_storage.element_bytes(),
        active_requests: config.active_requests,
        query_rows_per_request: config.query_rows,
        proposal_tokens_per_request: config.proposal_tokens,
        accepted_rows_per_request: config.accepted_rows_per_request,
        packed_update_rows: executor.update_rows,
        context_tokens_before_update: config.context_tokens,
        context_tokens_after_update: executor.context_after_update,
        body_kv_length: executor.body_kv_length,
        page_size: config.page_size,
        total_physical_pages: executor.total_physical_pages,
        max_pages_per_request: executor.max_pages_per_request,
        shared_kv_bytes: executor.shared_kv_bytes,
        flashinfer_metadata_bytes: executor.flashinfer_metadata_bytes,
        rust_owned_mutable_bytes: executor.arena.bytes,
        initial_anchor_tokens: executor.initial_anchor_tokens.clone(),
        dynamic_anchor_tokens: executor.dynamic_anchor_tokens.clone(),
        update_graph_nodes: executor.update_graph.node_count,
        update_graph_kernel_nodes: executor.update_graph.kernel_node_count,
        update_graph_memcpy_nodes: executor.update_graph.memcpy_node_count,
        update_graph_memset_nodes: executor.update_graph.memset_node_count,
        suffix_graph_nodes: executor.suffix_graph.node_count,
        suffix_graph_kernel_nodes: executor.suffix_graph.kernel_node_count,
        suffix_graph_memcpy_nodes: executor.suffix_graph.memcpy_node_count,
        suffix_graph_memset_nodes: executor.suffix_graph.memset_node_count,
        eager_token_exact,
        eager_confidence_max_abs,
        dynamic_anchor_changes_output,
        dynamic_token_changed_bytes,
        dynamic_confidence_changed_bytes,
        restored_token_exact,
        restored_confidence_max_abs,
        warmup: config.warmup,
        iterations: config.iterations,
        repeats: config.repeats,
        gpu_ms_per_update_replay: gpu_update,
        host_ms_per_update_replay: host_update,
        gpu_ms_per_suffix_replay: gpu_suffix,
        host_ms_per_suffix_replay: host_suffix,
        gpu_ms_per_full_step: gpu_full,
        host_ms_per_full_step: host_full,
        exact_row_update_graph: true,
        fixed_query_body_head_suffix_graph: true,
        shared_update_body_kv: true,
        query_output_aliases_body_input: true,
        body_output_is_head_source: true,
        strided_proposal_view: true,
        cold_capture_python_calls: 6,
        hot_replay_python_calls: 0,
        serving_dispatch_enabled: false,
    })
}

pub(super) struct DsparkStaticExecutor {
    library: &'static NativeLibrary,
    update_graph: DsparkCudaGraph,
    suffix_graph: DsparkCudaGraph,
    stream: DsparkCudaStream,
    arena: DsparkStaticArena,
    query_token_ids: GlmrtDeviceBuffer,
    update_buffers: DsparkPythonUpdateBuffers,
    body_buffers: DsparkPythonBodyBuffers,
    head_buffers: DsparkPythonHeadBuffers,
    paged_kv_metadata: DsparkPagedKvMetadataBuffers,
    query_config: DsparkQueryBenchConfig,
    config: DsparkStaticBenchConfig,
    update_rows: usize,
    context_after_update: usize,
    body_kv_length: usize,
    total_physical_pages: usize,
    max_pages_per_request: usize,
    request_page_tables: Vec<Vec<i32>>,
    shared_kv_bytes: u64,
    flashinfer_metadata_bytes: u64,
    initial_anchor_tokens: Vec<u32>,
    dynamic_anchor_tokens: Vec<u32>,
    batched_update_graphs: Option<DsparkBatchedUpdateGraphs>,
}

struct DsparkBatchedUpdateGraphs {
    buffers: DsparkPythonUpdateBuffers,
    graphs: BTreeMap<usize, DsparkCudaGraph>,
    max_rows: usize,
}

impl DsparkStaticExecutor {
    pub(super) fn capture(
        weights: DsparkStaticResidentWeights,
        config: DsparkStaticBenchConfig,
    ) -> Result<Self> {
        Self::capture_with_physical_pages(weights, config, None)
    }

    pub(super) fn capture_with_physical_pages(
        weights: DsparkStaticResidentWeights,
        config: DsparkStaticBenchConfig,
        physical_kv_pages: Option<usize>,
    ) -> Result<Self> {
        validate_config(config)?;
        anyhow::ensure!(
            weights.body.active_layers == config.draft_layers
                && weights.update.active_layers == config.draft_layers,
            "dSpark resident/config layer mismatch: body={} update={} config={}",
            weights.body.active_layers,
            weights.update.active_layers,
            config.draft_layers,
        );
        let library = cuda_native_library()?;
        let stream = DsparkCudaStream::create(library)?;
        let update_rows = checked_mul(
            config.active_requests,
            config.accepted_rows_per_request,
            "static update rows",
        )?;
        let total_query_rows = checked_mul(
            config.active_requests,
            config.query_rows,
            "static query rows",
        )?;
        let total_proposal_rows = checked_mul(
            config.active_requests,
            config.proposal_tokens,
            "static proposal rows",
        )?;
        let context_after_update = config
            .context_tokens
            .checked_add(config.accepted_rows_per_request)
            .context("dSpark static updated context overflow")?;
        let body_kv_length = context_after_update
            .checked_add(config.query_rows)
            .context("dSpark static body KV length overflow")?;
        let physical_pages_per_request = if config.allocate_full_kv_capacity {
            config.kv_capacity_tokens.div_ceil(config.page_size)
        } else {
            body_kv_length.div_ceil(config.page_size)
        };
        let minimum_physical_pages = checked_mul(
            config.active_requests,
            physical_pages_per_request,
            "static physical pages",
        )?;
        let total_physical_pages = physical_kv_pages.unwrap_or(minimum_physical_pages);
        anyhow::ensure!(
            total_physical_pages >= minimum_physical_pages,
            "dSpark physical KV pool has {total_physical_pages} pages but the executor requires at least {minimum_physical_pages}"
        );
        let max_pages_per_request = config.kv_capacity_tokens.div_ceil(config.page_size);

        let mut arena = DsparkStaticArena::default();
        let one_cache_bytes = tensor_bytes(
            checked_mul(
                checked_mul(
                    config.draft_layers,
                    total_physical_pages,
                    "static KV layers/pages",
                )?,
                DSPARK_BODY_HEADS,
                "static KV heads",
            )?,
            checked_mul(
                config.page_size,
                DSPARK_BODY_HEAD_DIM,
                "static KV page width",
            )?,
            config.kv_storage.element_bytes(),
            "static KV cache",
        )?;
        let shared_kv_bytes = u64::try_from(checked_mul(one_cache_bytes, 2, "static K/V bytes")?)
            .context("dSpark static K/V bytes do not fit u64")?;
        let k_cache = arena.allocate(library, one_cache_bytes, "dSpark static paged K")?;
        let v_cache = arena.allocate(library, one_cache_bytes, "dSpark static paged V")?;
        let block_table_entries = checked_mul(
            config.active_requests,
            max_pages_per_request,
            "static block table entries",
        )?;
        let block_tables = arena.allocate(
            library,
            checked_mul(block_table_entries, 4, "static block table bytes")?,
            "dSpark static block tables",
        )?;
        let request_i32_bytes = checked_mul(config.active_requests, 4, "static request metadata")?;
        let kv_lengths = arena.allocate(library, request_i32_bytes, "dSpark static KV lengths")?;
        let query_indptr = arena.allocate(
            library,
            i32_buffer_bytes(config.active_requests + 1, "static query indptr")?,
            "dSpark static query indptr",
        )?;
        let kv_indptr = arena.allocate(
            library,
            i32_buffer_bytes(config.active_requests + 1, "static KV indptr")?,
            "dSpark static KV indptr",
        )?;
        let page_indices = arena.allocate(
            library,
            i32_buffer_bytes(total_physical_pages, "static page indices")?,
            "dSpark static page indices",
        )?;
        let last_page_len = arena.allocate(
            library,
            i32_buffer_bytes(config.active_requests, "static last-page lengths")?,
            "dSpark static last-page lengths",
        )?;
        let paged_kv_metadata = DsparkPagedKvMetadataBuffers {
            query_indptr,
            kv_indptr,
            page_indices,
            last_page_len,
        };
        let flashinfer_metadata_bytes = [
            query_indptr.bytes,
            kv_indptr.bytes,
            page_indices.bytes,
            last_page_len.bytes,
        ]
        .into_iter()
        .try_fold(0_u64, |bytes, next| {
            bytes
                .checked_add(next as u64)
                .context("dSpark static FlashInfer metadata byte count overflow")
        })?;

        let update_hidden_bytes =
            tensor_bytes(update_rows, DSPARK_UPDATE_HIDDEN, 2, "static update hidden")?;
        let update_output_bytes = tensor_bytes(
            checked_mul(
                config.draft_layers,
                update_rows,
                "static update output rows",
            )?,
            DSPARK_UPDATE_ATTENTION_WIDTH,
            2,
            "static update output",
        )?;
        let update_row_metadata_bytes = checked_mul(update_rows, 4, "static update metadata")?;
        let update_buffers = DsparkPythonUpdateBuffers {
            target_hidden: arena.allocate(
                library,
                tensor_bytes(
                    update_rows,
                    DSPARK_UPDATE_TARGET_FEATURES,
                    2,
                    "static target hidden",
                )?,
                "dSpark static target hidden",
            )?,
            fusion_output: arena.allocate(
                library,
                update_hidden_bytes,
                "dSpark static fusion output",
            )?,
            fused_hidden: arena.allocate(
                library,
                update_hidden_bytes,
                "dSpark static fused hidden",
            )?,
            projected_kv: arena.allocate(
                library,
                tensor_bytes(
                    update_rows,
                    2 * DSPARK_UPDATE_ATTENTION_WIDTH,
                    2,
                    "static projected KV",
                )?,
                "dSpark static projected KV",
            )?,
            key_output: arena.allocate(library, update_output_bytes, "dSpark static keys")?,
            value_output: arena.allocate(library, update_output_bytes, "dSpark static values")?,
            reference_fused_hidden: arena.allocate(
                library,
                update_hidden_bytes,
                "dSpark static reference fused hidden",
            )?,
            reference_key_output: arena.allocate(
                library,
                update_output_bytes,
                "dSpark static reference keys",
            )?,
            reference_value_output: arena.allocate(
                library,
                update_output_bytes,
                "dSpark static reference values",
            )?,
            eager_fused_hidden: arena.allocate(
                library,
                update_hidden_bytes,
                "dSpark static eager fused hidden",
            )?,
            eager_key_output: arena.allocate(
                library,
                update_output_bytes,
                "dSpark static eager keys",
            )?,
            eager_value_output: arena.allocate(
                library,
                update_output_bytes,
                "dSpark static eager values",
            )?,
            k_cache,
            v_cache,
            row_request_ids: arena.allocate(
                library,
                update_row_metadata_bytes,
                "dSpark static update request IDs",
            )?,
            row_positions: arena.allocate(
                library,
                update_row_metadata_bytes,
                "dSpark static update positions",
            )?,
            row_cache_positions: arena.allocate(
                library,
                update_row_metadata_bytes,
                "dSpark static update cache positions",
            )?,
            block_tables,
        };

        let query_hidden_bytes = tensor_bytes(
            total_query_rows,
            DSPARK_BODY_HIDDEN,
            2,
            "static query hidden",
        )?;
        let attention_bytes = tensor_bytes(
            total_query_rows,
            DSPARK_BODY_ATTENTION_WIDTH,
            2,
            "static attention",
        )?;
        let query_hidden = arena.allocate(
            library,
            query_hidden_bytes,
            "dSpark static query/body input",
        )?;
        let body_output = arena.allocate(
            library,
            query_hidden_bytes,
            "dSpark static body/head output",
        )?;
        let offset_bytes = checked_mul(
            config.active_requests + 1,
            std::mem::size_of::<i64>(),
            "static offset bytes",
        )?;
        let body_buffers = DsparkPythonBodyBuffers {
            input: query_hidden,
            output: body_output,
            reference_output: arena.allocate(
                library,
                query_hidden_bytes,
                "dSpark static body reference",
            )?,
            hidden_attention: arena.allocate(
                library,
                query_hidden_bytes,
                "dSpark static attention residual",
            )?,
            hidden_mlp: arena.allocate(
                library,
                query_hidden_bytes,
                "dSpark static MLP residual",
            )?,
            normalized: arena.allocate(
                library,
                query_hidden_bytes,
                "dSpark static normalized hidden",
            )?,
            qkv: arena.allocate(
                library,
                tensor_bytes(
                    total_query_rows,
                    3 * DSPARK_BODY_ATTENTION_WIDTH,
                    2,
                    "static QKV",
                )?,
                "dSpark static QKV",
            )?,
            q: arena.allocate(library, attention_bytes, "dSpark static query")?,
            attention: arena.allocate(library, attention_bytes, "dSpark static attention")?,
            delta: arena.allocate(library, query_hidden_bytes, "dSpark static delta")?,
            gate_up: arena.allocate(
                library,
                tensor_bytes(
                    total_query_rows,
                    2 * DSPARK_BODY_INTERMEDIATE,
                    2,
                    "static gate/up",
                )?,
                "dSpark static gate/up",
            )?,
            activation: arena.allocate(
                library,
                tensor_bytes(
                    total_query_rows,
                    DSPARK_BODY_INTERMEDIATE,
                    2,
                    "static activation",
                )?,
                "dSpark static activation",
            )?,
            k_cache,
            v_cache,
            workspace: arena.allocate(
                library,
                DSPARK_BODY_WORKSPACE_BYTES,
                "dSpark static cuDNN workspace",
            )?,
            query_lengths: arena.allocate(
                library,
                request_i32_bytes,
                "dSpark static query lengths",
            )?,
            kv_lengths,
            query_positions: arena.allocate(
                library,
                checked_mul(total_query_rows, 4, "static query position bytes")?,
                "dSpark static absolute query positions",
            )?,
            block_tables,
            query_offsets: arena.allocate(library, offset_bytes, "dSpark static query offsets")?,
            output_offsets: arena.allocate(
                library,
                offset_bytes,
                "dSpark static output offsets",
            )?,
            query_indptr,
            kv_indptr,
            page_indices,
            last_page_len,
        };

        let feature_width = DSPARK_HEAD_HIDDEN + DSPARK_HEAD_MARKOV_RANK;
        let proposal_hidden_bytes = tensor_bytes(
            total_proposal_rows,
            DSPARK_HEAD_HIDDEN,
            2,
            "static proposal hidden",
        )?;
        let proposal_token_bytes = checked_mul(
            total_proposal_rows,
            std::mem::size_of::<i64>(),
            "static proposal tokens",
        )?;
        let proposal_confidence_bytes =
            checked_mul(total_proposal_rows, 4, "static proposal confidence")?;
        let head_buffers = DsparkPythonHeadBuffers {
            hidden: body_output,
            hidden_position_major: arena.allocate(
                library,
                proposal_hidden_bytes,
                "dSpark static position-major hidden",
            )?,
            base_logits: arena.allocate(
                library,
                tensor_bytes(
                    total_proposal_rows,
                    DSPARK_HEAD_VOCAB,
                    2,
                    "static base logits",
                )?,
                "dSpark static base logits",
            )?,
            markov_logits: arena.allocate(
                library,
                tensor_bytes(
                    config.active_requests,
                    DSPARK_HEAD_VOCAB,
                    2,
                    "static Markov logits",
                )?,
                "dSpark static Markov logits",
            )?,
            argmax_candidate_scores: arena.allocate(
                library,
                tensor_bytes(
                    config.active_requests,
                    DSPARK_HEAD_ARGMAX_BLOCKS,
                    4,
                    "static argmax candidate scores",
                )?,
                "dSpark static argmax candidate scores",
            )?,
            argmax_candidate_tokens: arena.allocate(
                library,
                tensor_bytes(
                    config.active_requests,
                    DSPARK_HEAD_ARGMAX_BLOCKS,
                    4,
                    "static argmax candidate tokens",
                )?,
                "dSpark static argmax candidate tokens",
            )?,
            embedding_steps: arena.allocate(
                library,
                tensor_bytes(
                    total_proposal_rows,
                    DSPARK_HEAD_MARKOV_RANK,
                    2,
                    "static Markov embeddings",
                )?,
                "dSpark static Markov embeddings",
            )?,
            token_steps: arena.allocate(
                library,
                proposal_token_bytes,
                "dSpark static token steps",
            )?,
            confidence_features: arena.allocate(
                library,
                tensor_bytes(
                    total_proposal_rows,
                    feature_width,
                    2,
                    "static confidence features",
                )?,
                "dSpark static confidence features",
            )?,
            confidence_logits: arena.allocate(
                library,
                checked_mul(total_proposal_rows, 2, "static confidence logits")?,
                "dSpark static confidence logits",
            )?,
            confidence_probabilities: arena.allocate(
                library,
                checked_mul(total_proposal_rows, 2, "static confidence probabilities")?,
                "dSpark static confidence probabilities",
            )?,
            anchor_tokens: arena.allocate(
                library,
                checked_mul(
                    config.active_requests,
                    std::mem::size_of::<i64>(),
                    "static anchor tokens",
                )?,
                "dSpark static anchor tokens",
            )?,
            output_tokens: arena.allocate(
                library,
                proposal_token_bytes,
                "dSpark static output tokens",
            )?,
            output_confidence: arena.allocate(
                library,
                proposal_confidence_bytes,
                "dSpark static output confidence",
            )?,
            reference_tokens: arena.allocate(
                library,
                proposal_token_bytes,
                "dSpark static reference tokens",
            )?,
            reference_confidence: arena.allocate(
                library,
                proposal_confidence_bytes,
                "dSpark static reference confidence",
            )?,
            eager_tokens: arena.allocate(
                library,
                proposal_token_bytes,
                "dSpark static eager tokens",
            )?,
            eager_confidence: arena.allocate(
                library,
                proposal_confidence_bytes,
                "dSpark static eager confidence",
            )?,
        };

        let query_token_ids = arena.allocate(
            library,
            checked_mul(total_query_rows, 4, "static query token IDs")?,
            "dSpark static query token IDs",
        )?;

        let mut block_table = vec![0_i32; block_table_entries];
        for request in 0..config.active_requests {
            for page in 0..physical_pages_per_request {
                block_table[request * max_pages_per_request + page] =
                    (request * physical_pages_per_request + page)
                        .try_into()
                        .context("dSpark static physical page does not fit i32")?;
            }
        }
        library
            .copy_h2d(block_tables, as_bytes(&block_table))
            .context("uploading dSpark static block table")?;
        let request_page_tables = block_table
            .chunks_exact(max_pages_per_request)
            .map(<[i32]>::to_vec)
            .collect::<Vec<_>>();
        let body_kv_length_i32 = i32::try_from(body_kv_length)
            .context("dSpark static body KV length does not fit i32")?;
        let body_lengths = vec![body_kv_length_i32; config.active_requests];
        library
            .copy_h2d(kv_lengths, as_bytes(&body_lengths))
            .context("uploading dSpark static KV lengths")?;
        DsparkPagedKvMetadata::for_lengths(
            &body_lengths,
            config.query_rows,
            config.page_size,
            physical_pages_per_request,
        )?
        .upload(library, paged_kv_metadata)?;
        let mut row_request_ids = Vec::<i32>::with_capacity(update_rows);
        let mut row_positions = Vec::<i32>::with_capacity(update_rows);
        for request in 0..config.active_requests {
            for row in 0..config.accepted_rows_per_request {
                row_request_ids.push(
                    request
                        .try_into()
                        .context("dSpark static request ID does not fit i32")?,
                );
                row_positions.push(
                    config
                        .context_tokens
                        .checked_add(row)
                        .context("dSpark static update position overflow")?
                        .try_into()
                        .context("dSpark static update position does not fit i32")?,
                );
            }
        }
        library
            .copy_h2d(update_buffers.row_request_ids, as_bytes(&row_request_ids))
            .context("uploading dSpark static update request IDs")?;
        library
            .copy_h2d(update_buffers.row_positions, as_bytes(&row_positions))
            .context("uploading dSpark static update positions")?;
        library
            .copy_h2d(update_buffers.row_cache_positions, as_bytes(&row_positions))
            .context("uploading dSpark static update cache positions")?;
        let mut query_positions = Vec::<i32>::with_capacity(total_query_rows);
        for _request in 0..config.active_requests {
            for row in 0..config.query_rows {
                query_positions.push(
                    context_after_update
                        .checked_add(row)
                        .context("dSpark static query position overflow")?
                        .try_into()
                        .context("dSpark static query position does not fit i32")?,
                );
            }
        }
        library
            .copy_h2d(body_buffers.query_positions, as_bytes(&query_positions))
            .context("uploading dSpark static absolute query positions")?;

        let query_config = DsparkQueryBenchConfig {
            active_requests: config.active_requests,
            query_rows: config.query_rows,
            mask_tokens: config.query_rows - 1,
            mask_token_id: config.mask_token_id,
            warmup: config.warmup,
            iterations: config.iterations,
            repeats: config.repeats,
            seed: config.seed,
        };
        let initial_anchor_tokens = (0..config.active_requests)
            .map(|request| {
                normalized_anchor(config.seed + request as i64 * 104_729, config.mask_token_id)
            })
            .collect::<Vec<_>>();
        let dynamic_anchor_tokens = initial_anchor_tokens
            .iter()
            // Keep graph liveness independent of local tokenizer row
            // similarity. Red Hat's Markov table contains neighborhoods whose
            // adjacent embeddings intentionally collapse after BF16 rounding.
            .map(|token| {
                let candidate = if *token == 1 { 0 } else { 1 };
                normalized_anchor(candidate, config.mask_token_id)
            })
            .collect::<Vec<_>>();
        upload_anchor_tokens(
            library,
            query_token_ids,
            head_buffers.anchor_tokens,
            &initial_anchor_tokens,
            query_config,
        )?;

        let update_config = DsparkUpdateBenchConfig {
            layers: config.draft_layers,
            rows: update_rows,
            active_requests: config.active_requests,
            context_tokens: config.context_tokens,
            kv_capacity_tokens: config.kv_capacity_tokens,
            page_size: config.page_size,
            kv_storage: config.kv_storage,
            warmup: config.warmup,
            iterations: config.iterations,
            repeats: config.repeats,
            seed: config.seed,
            initialize_target_hidden: true,
            initialize_kv: true,
        };
        launch_python_update(
            stream.raw,
            &update_buffers,
            weights.update,
            update_config,
            total_physical_pages,
            max_pages_per_request,
            "prepare_dspark_context_update",
        )?;
        stream.synchronize()?;
        unsafe {
            library
                .cuda_graph_begin_capture(stream.raw)
                .context("beginning dSpark static update capture")?;
        }
        if let Err(error) = launch_python_update(
            stream.raw,
            &update_buffers,
            weights.update,
            update_config,
            total_physical_pages,
            max_pages_per_request,
            "capture_dspark_context_update",
        ) {
            unsafe {
                let _ = library.cuda_graph_end_capture_retained(stream.raw);
            }
            return Err(error).context("capturing dSpark static update graph");
        }
        let update_capture = unsafe {
            library
                .cuda_graph_end_capture_retained(stream.raw)
                .context("ending dSpark static update capture")?
        };
        let update_graph = DsparkCudaGraph::new(library, update_capture)?;
        update_graph.validate()?;

        launch_embedding(
            library,
            stream.raw,
            weights.query.embedding,
            query_token_ids,
            query_hidden,
            total_query_rows,
        )?;
        let body_config = DsparkBodyBenchConfig {
            layers: config.draft_layers,
            active_requests: config.active_requests,
            query_rows: config.query_rows,
            context_tokens: context_after_update,
            kv_capacity_tokens: config.kv_capacity_tokens,
            page_size: config.page_size,
            kv_storage: config.kv_storage,
            warmup: config.warmup,
            iterations: config.iterations,
            repeats: config.repeats,
            seed: config.seed,
            initialize_input: false,
            initialize_kv: false,
        };
        launch_python_body(
            stream.raw,
            &body_buffers,
            &weights.body,
            body_config,
            total_physical_pages,
            max_pages_per_request,
            "prepare_dspark_cudnn_paged_body",
        )?;
        let head_config = DsparkHeadBenchConfig {
            active_requests: config.active_requests,
            proposal_tokens: config.proposal_tokens,
            hidden_rows_per_request: config.query_rows,
            hidden_start_row: config.proposal_start_row,
            warmup: config.warmup,
            iterations: config.iterations,
            repeats: config.repeats,
            seed: config.seed,
            initialize_hidden: false,
        };
        launch_python_head(
            stream.raw,
            &head_buffers,
            weights.head,
            head_config,
            "prepare_dspark_head",
        )?;
        stream.synchronize()?;

        unsafe {
            library
                .cuda_graph_begin_capture(stream.raw)
                .context("beginning dSpark static suffix capture")?;
        }
        let suffix_result = (|| {
            launch_embedding(
                library,
                stream.raw,
                weights.query.embedding,
                query_token_ids,
                query_hidden,
                total_query_rows,
            )?;
            launch_python_body(
                stream.raw,
                &body_buffers,
                &weights.body,
                body_config,
                total_physical_pages,
                max_pages_per_request,
                "capture_dspark_cudnn_paged_body",
            )?;
            launch_python_head(
                stream.raw,
                &head_buffers,
                weights.head,
                head_config,
                "capture_dspark_head",
            )
        })();
        if let Err(error) = suffix_result {
            unsafe {
                let _ = library.cuda_graph_end_capture_retained(stream.raw);
            }
            return Err(error).context("capturing dSpark static suffix graph");
        }
        let suffix_capture = unsafe {
            library
                .cuda_graph_end_capture_retained(stream.raw)
                .context("ending dSpark static suffix capture")?
        };
        let suffix_graph = DsparkCudaGraph::new(library, suffix_capture)?;
        suffix_graph.validate_min_kernel_nodes(config.draft_layers + 1)?;

        Ok(Self {
            library,
            update_graph,
            suffix_graph,
            stream,
            arena,
            query_token_ids,
            update_buffers,
            body_buffers,
            head_buffers,
            paged_kv_metadata,
            query_config,
            config,
            update_rows,
            context_after_update,
            body_kv_length,
            total_physical_pages,
            max_pages_per_request,
            request_page_tables,
            shared_kv_bytes,
            flashinfer_metadata_bytes,
            initial_anchor_tokens,
            dynamic_anchor_tokens,
            batched_update_graphs: None,
        })
    }

    pub(super) fn capture_batched_update_graphs(
        &mut self,
        weights: DsparkUpdateResidentWeights,
        row_buckets: &[usize],
    ) -> Result<()> {
        anyhow::ensure!(
            self.config.active_requests == 1 && self.config.accepted_rows_per_request == 1,
            "batched dSpark serving updates require the C=1 one-row base executor"
        );
        anyhow::ensure!(
            self.batched_update_graphs.is_none(),
            "batched dSpark update graphs were already captured"
        );
        anyhow::ensure!(
            !row_buckets.is_empty()
                && row_buckets.windows(2).all(|pair| pair[0] < pair[1])
                && row_buckets
                    .iter()
                    .all(|rows| rows.is_power_of_two() && *rows >= 2 && *rows <= 1_024),
            "dSpark batched update rows must be unique ascending powers of two in 2..=1024"
        );
        let max_rows = *row_buckets
            .last()
            .expect("non-empty dSpark batched update rows were checked above");
        let buffers = allocate_batched_update_buffers(
            &mut self.arena,
            self.library,
            self.config.draft_layers,
            max_rows,
            self.update_buffers.k_cache,
            self.update_buffers.v_cache,
            self.update_buffers.block_tables,
        )?;
        let mut graphs = BTreeMap::new();
        for &rows in row_buckets {
            let request_ids = vec![0_i32; rows];
            let positions = (0..rows)
                .map(|row| i32::try_from(row).context("dSpark batched update row does not fit i32"))
                .collect::<Result<Vec<_>>>()?;
            self.library
                .copy_h2d(buffers.row_request_ids, as_bytes(&request_ids))
                .context("uploading dSpark batched update request IDs")?;
            self.library
                .copy_h2d(buffers.row_positions, as_bytes(&positions))
                .context("uploading dSpark batched update positions")?;
            self.library
                .copy_h2d(buffers.row_cache_positions, as_bytes(&positions))
                .context("uploading dSpark batched update cache positions")?;
            let config = DsparkUpdateBenchConfig {
                layers: self.config.draft_layers,
                rows,
                active_requests: 1,
                context_tokens: 1,
                kv_capacity_tokens: self.config.kv_capacity_tokens,
                page_size: self.config.page_size,
                kv_storage: self.config.kv_storage,
                warmup: 0,
                iterations: 1,
                repeats: 1,
                seed: self.config.seed + rows as i64,
                initialize_target_hidden: true,
                initialize_kv: false,
            };
            launch_python_update(
                self.stream.raw,
                &buffers,
                weights,
                config,
                self.total_physical_pages,
                self.max_pages_per_request,
                "prepare_dspark_context_update",
            )
            .with_context(|| format!("preparing {rows}-row dSpark serving update"))?;
            self.stream.synchronize()?;
            unsafe {
                self.library
                    .cuda_graph_begin_capture(self.stream.raw)
                    .with_context(|| format!("beginning {rows}-row dSpark update capture"))?;
            }
            if let Err(error) = launch_python_update(
                self.stream.raw,
                &buffers,
                weights,
                config,
                self.total_physical_pages,
                self.max_pages_per_request,
                "capture_dspark_context_update",
            ) {
                unsafe {
                    let _ = self
                        .library
                        .cuda_graph_end_capture_retained(self.stream.raw);
                }
                return Err(error)
                    .with_context(|| format!("capturing {rows}-row dSpark serving update graph"));
            }
            let capture = unsafe {
                self.library
                    .cuda_graph_end_capture_retained(self.stream.raw)
                    .with_context(|| format!("ending {rows}-row dSpark update capture"))?
            };
            let graph = DsparkCudaGraph::new(self.library, capture)?;
            graph.validate()?;
            graphs.insert(rows, graph);
        }
        self.batched_update_graphs = Some(DsparkBatchedUpdateGraphs {
            buffers,
            graphs,
            max_rows,
        });
        Ok(())
    }

    fn set_anchor_tokens(&mut self, anchors: &[u32]) -> Result<()> {
        upload_anchor_tokens(
            self.library,
            self.query_token_ids,
            self.head_buffers.anchor_tokens,
            anchors,
            self.query_config,
        )
    }

    pub(super) fn set_request_page_table(&mut self, page_table: &[i32]) -> Result<()> {
        anyhow::ensure!(
            self.config.active_requests == 1,
            "single-request dSpark page-table update requires C=1"
        );
        anyhow::ensure!(
            page_table.len() == self.max_pages_per_request,
            "dSpark request page table has {} entries, expected {}",
            page_table.len(),
            self.max_pages_per_request
        );
        anyhow::ensure!(
            page_table.iter().all(|page| {
                *page >= 0
                    && usize::try_from(*page).is_ok_and(|page| page < self.total_physical_pages)
            }),
            "dSpark request page table contains an invalid physical page: {page_table:?}"
        );
        self.library
            .copy_h2d(self.body_buffers.block_tables, as_bytes(page_table))
            .context("uploading dSpark request block table")?;
        self.request_page_tables[0].copy_from_slice(page_table);
        Ok(())
    }

    pub(super) fn read_request_cache_snapshot(
        &self,
        page_table: &[i32],
        cache_context_tokens: usize,
    ) -> Result<Vec<u8>> {
        let logical_pages = cache_context_tokens.div_ceil(self.config.page_size);
        let page_bytes = self.request_cache_page_bytes()?;
        let snapshot_bytes = checked_mul(
            checked_mul(
                2 * self.config.draft_layers,
                logical_pages,
                "dSpark request snapshot plane/layer pages",
            )?,
            page_bytes,
            "dSpark request snapshot bytes",
        )?;
        self.validate_request_cache_snapshot_layout(page_table, logical_pages)?;
        self.stream
            .synchronize()
            .context("synchronizing before dSpark request cache snapshot")?;
        let mut snapshot = vec![0_u8; snapshot_bytes];
        for (plane, cache) in [self.body_buffers.k_cache, self.body_buffers.v_cache]
            .into_iter()
            .enumerate()
        {
            for layer in 0..self.config.draft_layers {
                let host_layer_base = (plane * self.config.draft_layers + layer)
                    .checked_mul(logical_pages)
                    .and_then(|pages| pages.checked_mul(page_bytes))
                    .context("dSpark request snapshot host offset overflow")?;
                let mut logical_page = 0;
                while logical_page < logical_pages {
                    let physical_page = usize::try_from(page_table[logical_page])
                        .context("dSpark request snapshot physical page is negative")?;
                    let mut run_pages = 1;
                    while logical_page + run_pages < logical_pages {
                        let next = usize::try_from(page_table[logical_page + run_pages])
                            .context("dSpark request snapshot physical page is negative")?;
                        if next != physical_page + run_pages {
                            break;
                        }
                        run_pages += 1;
                    }
                    let run_bytes = run_pages
                        .checked_mul(page_bytes)
                        .context("dSpark request snapshot run byte count overflow")?;
                    let device_offset = layer
                        .checked_mul(self.total_physical_pages)
                        .and_then(|pages| pages.checked_add(physical_page))
                        .and_then(|pages| pages.checked_mul(page_bytes))
                        .context("dSpark request snapshot device offset overflow")?;
                    let source = device_buffer_byte_view(
                        cache,
                        device_offset,
                        run_bytes,
                        "dSpark request snapshot device run",
                    )?;
                    let host_offset = host_layer_base
                        .checked_add(
                            logical_page
                                .checked_mul(page_bytes)
                                .context("dSpark request snapshot logical offset overflow")?,
                        )
                        .context("dSpark request snapshot host run offset overflow")?;
                    self.library
                        .copy_d2h(&mut snapshot[host_offset..host_offset + run_bytes], source)
                        .context("reading dSpark request cache run")?;
                    logical_page += run_pages;
                }
            }
        }
        self.zero_uncommitted_snapshot_tail(&mut snapshot, cache_context_tokens, logical_pages)?;
        Ok(snapshot)
    }

    pub(super) fn restore_request_cache_snapshot(
        &self,
        page_table: &[i32],
        cache_context_tokens: usize,
        snapshot: &[u8],
    ) -> Result<()> {
        let logical_pages = cache_context_tokens.div_ceil(self.config.page_size);
        let page_bytes = self.request_cache_page_bytes()?;
        let expected_bytes = checked_mul(
            checked_mul(
                2 * self.config.draft_layers,
                logical_pages,
                "dSpark request restore plane/layer pages",
            )?,
            page_bytes,
            "dSpark request restore bytes",
        )?;
        anyhow::ensure!(
            snapshot.len() == expected_bytes,
            "dSpark request cache snapshot has {} bytes, expected {expected_bytes}",
            snapshot.len()
        );
        self.validate_request_cache_snapshot_layout(page_table, logical_pages)?;
        for (plane, cache) in [self.body_buffers.k_cache, self.body_buffers.v_cache]
            .into_iter()
            .enumerate()
        {
            for layer in 0..self.config.draft_layers {
                let host_layer_base = (plane * self.config.draft_layers + layer)
                    .checked_mul(logical_pages)
                    .and_then(|pages| pages.checked_mul(page_bytes))
                    .context("dSpark request restore host offset overflow")?;
                let mut logical_page = 0;
                while logical_page < logical_pages {
                    let physical_page = usize::try_from(page_table[logical_page])
                        .context("dSpark request restore physical page is negative")?;
                    let mut run_pages = 1;
                    while logical_page + run_pages < logical_pages {
                        let next = usize::try_from(page_table[logical_page + run_pages])
                            .context("dSpark request restore physical page is negative")?;
                        if next != physical_page + run_pages {
                            break;
                        }
                        run_pages += 1;
                    }
                    let run_bytes = run_pages
                        .checked_mul(page_bytes)
                        .context("dSpark request restore run byte count overflow")?;
                    let device_offset = layer
                        .checked_mul(self.total_physical_pages)
                        .and_then(|pages| pages.checked_add(physical_page))
                        .and_then(|pages| pages.checked_mul(page_bytes))
                        .context("dSpark request restore device offset overflow")?;
                    let destination = device_buffer_byte_view(
                        cache,
                        device_offset,
                        run_bytes,
                        "dSpark request restore device run",
                    )?;
                    let host_offset = host_layer_base
                        .checked_add(
                            logical_page
                                .checked_mul(page_bytes)
                                .context("dSpark request restore logical offset overflow")?,
                        )
                        .context("dSpark request restore host run offset overflow")?;
                    self.library
                        .copy_h2d(destination, &snapshot[host_offset..host_offset + run_bytes])
                        .context("restoring dSpark request cache run")?;
                    logical_page += run_pages;
                }
            }
        }
        Ok(())
    }

    fn request_cache_page_bytes(&self) -> Result<usize> {
        tensor_bytes(
            checked_mul(
                DSPARK_BODY_HEADS,
                self.config.page_size,
                "dSpark request cache page heads/tokens",
            )?,
            DSPARK_BODY_HEAD_DIM,
            self.config.kv_storage.element_bytes(),
            "dSpark request cache page",
        )
    }

    fn validate_request_cache_snapshot_layout(
        &self,
        page_table: &[i32],
        logical_pages: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            logical_pages <= page_table.len(),
            "dSpark request cache snapshot needs {logical_pages} logical pages but the request table has {}",
            page_table.len()
        );
        anyhow::ensure!(
            page_table.iter().take(logical_pages).all(|page| {
                *page >= 0
                    && usize::try_from(*page)
                        .is_ok_and(|physical| physical < self.total_physical_pages)
            }),
            "dSpark request cache snapshot page table contains an invalid physical page"
        );
        Ok(())
    }

    fn zero_uncommitted_snapshot_tail(
        &self,
        snapshot: &mut [u8],
        cache_context_tokens: usize,
        logical_pages: usize,
    ) -> Result<()> {
        let valid_tokens = cache_context_tokens % self.config.page_size;
        if valid_tokens == 0 || logical_pages == 0 {
            return Ok(());
        }
        let element_bytes = self.config.kv_storage.element_bytes();
        let head_token_bytes = DSPARK_BODY_HEAD_DIM
            .checked_mul(element_bytes)
            .context("dSpark request snapshot head-token bytes overflow")?;
        let page_bytes = self.request_cache_page_bytes()?;
        let layer_bytes = logical_pages
            .checked_mul(page_bytes)
            .context("dSpark request snapshot layer bytes overflow")?;
        let final_page_offset = (logical_pages - 1)
            .checked_mul(page_bytes)
            .context("dSpark request snapshot final-page offset overflow")?;
        for plane_layer in 0..2 * self.config.draft_layers {
            let layer_base = plane_layer
                .checked_mul(layer_bytes)
                .and_then(|offset| offset.checked_add(final_page_offset))
                .context("dSpark request snapshot plane/layer offset overflow")?;
            for head in 0..DSPARK_BODY_HEADS {
                let invalid_start = layer_base
                    .checked_add(
                        head.checked_mul(self.config.page_size)
                            .and_then(|tokens| tokens.checked_add(valid_tokens))
                            .and_then(|tokens| tokens.checked_mul(head_token_bytes))
                            .context("dSpark request snapshot invalid-tail offset overflow")?,
                    )
                    .context("dSpark request snapshot invalid-tail start overflow")?;
                let invalid_end = layer_base
                    .checked_add(
                        (head + 1)
                            .checked_mul(self.config.page_size)
                            .and_then(|tokens| tokens.checked_mul(head_token_bytes))
                            .context("dSpark request snapshot invalid-tail end overflow")?,
                    )
                    .context("dSpark request snapshot invalid-tail end overflow")?;
                snapshot[invalid_start..invalid_end].fill(0);
            }
        }
        Ok(())
    }

    pub(super) fn replay_request_step_with_cache_context(
        &mut self,
        target_hidden_taps: [&DeviceBf16Output; DSPARK_UPDATE_LAYERS],
        target_row_start: usize,
        committed_rows: usize,
        context_tokens: usize,
        cache_context_tokens: usize,
        anchor_token: usize,
    ) -> Result<DsparkDraftStep> {
        let total_start = Instant::now();
        anyhow::ensure!(
            self.config.active_requests == 1 && self.config.accepted_rows_per_request == 1,
            "dSpark request replay requires the C=1 one-row update graph"
        );
        anyhow::ensure!(
            committed_rows > 0,
            "dSpark request replay requires a committed target row"
        );
        anyhow::ensure!(
            anchor_token < DSPARK_QUERY_VOCAB && anchor_token != self.config.mask_token_id,
            "dSpark request anchor token {anchor_token} is invalid"
        );
        anyhow::ensure!(
            target_hidden_taps.iter().all(|tap| {
                target_row_start
                    .checked_add(committed_rows)
                    .is_some_and(|row_end| row_end <= tap.rows)
                    && tap.values_per_row == DSPARK_UPDATE_HIDDEN
                    && tap.buffer().device_id == self.update_buffers.target_hidden.device_id
            }),
            "dSpark request target hidden taps do not contain rows {target_row_start}..{} with width {DSPARK_UPDATE_HIDDEN} on the executor device",
            target_row_start.saturating_add(committed_rows),
        );
        let context_after_update = context_tokens
            .checked_add(committed_rows)
            .context("dSpark request context update overflow")?;
        let cache_context_after_update = cache_context_tokens
            .checked_add(committed_rows)
            .context("dSpark request cache-context update overflow")?;
        let body_kv_length = cache_context_after_update
            .checked_add(self.config.query_rows)
            .context("dSpark request body KV length overflow")?;
        anyhow::ensure!(
            body_kv_length <= self.config.kv_capacity_tokens,
            "dSpark request body KV length {body_kv_length} exceeds capacity {}",
            self.config.kv_capacity_tokens
        );

        let update_start = Instant::now();
        let feature_bytes = DSPARK_UPDATE_HIDDEN
            .checked_mul(std::mem::size_of::<u16>())
            .context("dSpark request feature byte count overflow")?;
        for (tap_index, tap) in target_hidden_taps.iter().enumerate() {
            tap.wait_ready_on_stream(self.stream.raw)
                .with_context(|| format!("waiting for dSpark target hidden tap {tap_index}"))?;
        }
        let mut row_offset = 0_usize;
        while row_offset < committed_rows {
            let remaining = committed_rows - row_offset;
            let chunk_rows = self
                .batched_update_graphs
                .as_ref()
                .and_then(|set| {
                    set.graphs
                        .range(..=remaining.min(set.max_rows))
                        .next_back()
                        .map(|(rows, _)| *rows)
                })
                .unwrap_or(1);
            let (buffers, graph) = if chunk_rows == 1 {
                (&self.update_buffers, &self.update_graph)
            } else {
                let set = self
                    .batched_update_graphs
                    .as_ref()
                    .expect("a non-unit dSpark update bucket requires the batched registry");
                (
                    &set.buffers,
                    set.graphs
                        .get(&chunk_rows)
                        .expect("the dSpark update bucket came from this registry"),
                )
            };
            let row_positions = (0..chunk_rows)
                .map(|chunk_row| {
                    context_tokens
                        .checked_add(row_offset)
                        .and_then(|position| position.checked_add(chunk_row))
                        .context("dSpark request context position overflow")?
                        .try_into()
                        .context("dSpark request context position does not fit i32")
                })
                .collect::<Result<Vec<i32>>>()?;
            let row_cache_positions = (0..chunk_rows)
                .map(|chunk_row| {
                    cache_context_tokens
                        .checked_add(row_offset)
                        .and_then(|position| position.checked_add(chunk_row))
                        .context("dSpark request cache position overflow")?
                        .try_into()
                        .context("dSpark request cache position does not fit i32")
                })
                .collect::<Result<Vec<i32>>>()?;
            self.library
                .copy_h2d(buffers.row_positions, as_bytes(&row_positions))
                .context("uploading dSpark request update positions")?;
            self.library
                .copy_h2d(buffers.row_cache_positions, as_bytes(&row_cache_positions))
                .context("uploading dSpark request update cache positions")?;
            unsafe {
                let target_row = target_row_start
                    .checked_add(row_offset)
                    .context("dSpark request target row overflow")?;
                let source_bytes = chunk_rows
                    .checked_mul(feature_bytes)
                    .context("dSpark request source tap byte count overflow")?;
                let destination_pitch = DSPARK_UPDATE_LAYERS
                    .checked_mul(feature_bytes)
                    .context("dSpark request target feature pitch overflow")?;
                for (tap_index, tap) in target_hidden_taps.iter().enumerate() {
                    let source = device_buffer_byte_view(
                        tap.buffer(),
                        target_row
                            .checked_mul(feature_bytes)
                            .context("dSpark request source tap offset overflow")?,
                        source_bytes,
                        "dSpark request target hidden source rows",
                    )?;
                    let destination_span = chunk_rows
                        .saturating_sub(1)
                        .checked_mul(destination_pitch)
                        .and_then(|bytes| bytes.checked_add(feature_bytes))
                        .context("dSpark request destination tap span overflow")?;
                    let destination = device_buffer_byte_view(
                        buffers.target_hidden,
                        tap_index
                            .checked_mul(feature_bytes)
                            .context("dSpark request destination tap offset overflow")?,
                        destination_span,
                        "dSpark request target hidden destination feature",
                    )?;
                    self.library
                        .copy_d2d_2d_async(
                            destination,
                            destination_pitch,
                            source,
                            feature_bytes,
                            feature_bytes,
                            chunk_rows,
                            self.stream.raw,
                        )
                        .with_context(|| format!("copying dSpark target hidden tap {tap_index}"))?;
                }
            }
            graph.validate()?;
            unsafe {
                self.library
                    .cuda_graph_launch(graph.exec_raw, self.stream.raw)
                    .with_context(|| {
                        format!("launching {chunk_rows}-row dSpark request update graph")
                    })?;
            }
            // A later chunk reuses the same metadata and target-hidden staging
            // buffers. Keep one synchronization per large bucket, not per row.
            self.stream.synchronize()?;
            row_offset += chunk_rows;
        }
        let update_ms = update_start.elapsed().as_secs_f64() * 1_000.0;
        let body_kv_length_i32 = i32::try_from(body_kv_length)
            .context("dSpark request body KV length does not fit i32")?;
        self.library
            .copy_h2d(
                self.body_buffers.kv_lengths,
                as_bytes(std::slice::from_ref(&body_kv_length_i32)),
            )
            .context("uploading dSpark request body KV length")?;
        let query_positions = (0..self.config.query_rows)
            .map(|row| {
                context_after_update
                    .checked_add(row)
                    .context("dSpark request absolute query position overflow")?
                    .try_into()
                    .context("dSpark request absolute query position does not fit i32")
            })
            .collect::<Result<Vec<i32>>>()?;
        self.library
            .copy_h2d(
                self.body_buffers.query_positions,
                as_bytes(&query_positions),
            )
            .context("uploading dSpark request absolute query positions")?;
        DsparkPagedKvMetadata::for_page_tables(
            &[body_kv_length_i32],
            self.config.query_rows,
            self.config.page_size,
            &self.request_page_tables,
            self.total_physical_pages,
        )?
        .upload(self.library, self.paged_kv_metadata)?;
        self.set_anchor_tokens(&[anchor_token as u32])?;

        let suffix_start = Instant::now();
        self.replay_suffix()?;
        self.stream.synchronize()?;
        let suffix_ms = suffix_start.elapsed().as_secs_f64() * 1_000.0;
        let readback_start = Instant::now();
        let token_bytes = self.read_tokens(self.head_buffers.output_tokens)?;
        let confidence_bytes = self.read_confidence(self.head_buffers.output_confidence)?;
        let proposal_token_ids = token_bytes
            .chunks_exact(std::mem::size_of::<i64>())
            .map(|bytes| {
                let token = i64::from_ne_bytes(
                    bytes
                        .try_into()
                        .expect("dSpark request token chunk has i64 width"),
                );
                usize::try_from(token)
                    .context("dSpark request proposal token is negative or does not fit usize")
            })
            .collect::<Result<Vec<_>>>()?;
        let conditional_confidence = confidence_bytes
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|bytes| {
                f32::from_ne_bytes(
                    bytes
                        .try_into()
                        .expect("dSpark request confidence chunk has f32 width"),
                )
            })
            .collect::<Vec<_>>();
        anyhow::ensure!(
            proposal_token_ids.len() == self.config.proposal_tokens
                && proposal_token_ids
                    .iter()
                    .all(|token| *token < DSPARK_HEAD_VOCAB)
                && conditional_confidence.len() == self.config.proposal_tokens
                && conditional_confidence
                    .iter()
                    .all(|confidence| confidence.is_finite() && (0.0..=1.0).contains(confidence)),
            "dSpark request output geometry or values are invalid"
        );
        let readback_ms = readback_start.elapsed().as_secs_f64() * 1_000.0;
        self.context_after_update = context_after_update;
        self.body_kv_length = body_kv_length;
        Ok(DsparkDraftStep {
            context_tokens,
            committed_rows,
            anchor_token,
            proposal_token_ids,
            conditional_confidence,
            update_ms,
            suffix_ms,
            readback_ms,
            total_ms: total_start.elapsed().as_secs_f64() * 1_000.0,
        })
    }

    fn replay_update(&self) -> Result<()> {
        self.update_graph.validate()?;
        unsafe {
            self.library
                .cuda_graph_launch(self.update_graph.exec_raw, self.stream.raw)
                .context("launching dSpark static update graph")
        }
    }

    fn replay_suffix(&self) -> Result<()> {
        self.suffix_graph
            .validate_min_kernel_nodes(self.config.draft_layers + 1)?;
        unsafe {
            self.library
                .cuda_graph_launch(self.suffix_graph.exec_raw, self.stream.raw)
                .context("launching dSpark static suffix graph")
        }
    }

    fn replay_full(&self) -> Result<()> {
        self.replay_update()?;
        self.replay_suffix()
    }

    fn measure(
        &self,
        part: ReplayPart,
    ) -> Result<(DsparkPagedAttentionTiming, DsparkPagedAttentionTiming)> {
        let mut gpu_samples = Vec::with_capacity(self.config.repeats);
        let mut host_samples = Vec::with_capacity(self.config.repeats);
        for _ in 0..self.config.repeats {
            let start_event = DsparkCudaEvent::create(self.library)?;
            let end_event = DsparkCudaEvent::create(self.library)?;
            unsafe {
                self.library
                    .cuda_event_record(start_event.raw, self.stream.raw)
                    .context("recording dSpark static benchmark start event")?;
            }
            let host_started = Instant::now();
            for _ in 0..self.config.iterations {
                match part {
                    ReplayPart::Update => self.replay_update()?,
                    ReplayPart::Suffix => self.replay_suffix()?,
                    ReplayPart::Full => self.replay_full()?,
                }
            }
            unsafe {
                self.library
                    .cuda_event_record(end_event.raw, self.stream.raw)
                    .context("recording dSpark static benchmark end event")?;
                self.library
                    .cuda_event_synchronize(end_event.raw)
                    .context("waiting for dSpark static benchmark end event")?;
            }
            host_samples.push(
                host_started.elapsed().as_secs_f64() * 1_000.0 / self.config.iterations as f64,
            );
            gpu_samples.push(
                unsafe {
                    self.library
                        .cuda_event_elapsed_ms(start_event.raw, end_event.raw)
                        .context("measuring dSpark static CUDA graph replay")?
                } as f64
                    / self.config.iterations as f64,
            );
        }
        Ok((timing_summary(gpu_samples)?, timing_summary(host_samples)?))
    }

    fn read_tokens(&self, buffer: GlmrtDeviceBuffer) -> Result<Vec<u8>> {
        let bytes = checked_mul(
            checked_mul(
                self.config.active_requests,
                self.config.proposal_tokens,
                "static token rows",
            )?,
            std::mem::size_of::<i64>(),
            "static token bytes",
        )?;
        self.read_buffer(buffer, bytes, "reading dSpark static tokens")
    }

    fn read_confidence(&self, buffer: GlmrtDeviceBuffer) -> Result<Vec<u8>> {
        let bytes = checked_mul(
            checked_mul(
                self.config.active_requests,
                self.config.proposal_tokens,
                "static confidence rows",
            )?,
            std::mem::size_of::<f32>(),
            "static confidence bytes",
        )?;
        self.read_buffer(buffer, bytes, "reading dSpark static confidence")
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
enum ReplayPart {
    Update,
    Suffix,
    Full,
}

#[derive(Default)]
struct DsparkStaticArena {
    buffers: Vec<DsparkDeviceBuffer>,
    bytes: u64,
}

impl DsparkStaticArena {
    fn allocate(
        &mut self,
        library: &'static NativeLibrary,
        bytes: usize,
        label: &str,
    ) -> Result<GlmrtDeviceBuffer> {
        let buffer = DsparkDeviceBuffer::new(library, bytes, label)?;
        let raw = buffer.raw;
        self.bytes = self
            .bytes
            .checked_add(
                raw.bytes
                    .try_into()
                    .context("dSpark static buffer bytes do not fit u64")?,
            )
            .context("dSpark static mutable byte count overflow")?;
        self.buffers.push(buffer);
        Ok(raw)
    }
}

fn allocate_batched_update_buffers(
    arena: &mut DsparkStaticArena,
    library: &'static NativeLibrary,
    layers: usize,
    rows: usize,
    k_cache: GlmrtDeviceBuffer,
    v_cache: GlmrtDeviceBuffer,
    block_tables: GlmrtDeviceBuffer,
) -> Result<DsparkPythonUpdateBuffers> {
    let hidden_bytes = tensor_bytes(rows, DSPARK_UPDATE_HIDDEN, 2, "batched update hidden")?;
    let output_bytes = tensor_bytes(
        checked_mul(layers, rows, "batched update output rows")?,
        DSPARK_UPDATE_ATTENTION_WIDTH,
        2,
        "batched update output",
    )?;
    let allocate =
        |arena: &mut DsparkStaticArena, bytes, label| arena.allocate(library, bytes, label);
    Ok(DsparkPythonUpdateBuffers {
        target_hidden: allocate(
            arena,
            tensor_bytes(
                rows,
                DSPARK_UPDATE_TARGET_FEATURES,
                2,
                "batched update target hidden",
            )?,
            "dSpark batched update target hidden",
        )?,
        fusion_output: allocate(arena, hidden_bytes, "dSpark batched update fusion output")?,
        fused_hidden: allocate(arena, hidden_bytes, "dSpark batched update fused hidden")?,
        projected_kv: allocate(
            arena,
            tensor_bytes(
                rows,
                2 * DSPARK_UPDATE_ATTENTION_WIDTH,
                2,
                "batched update projected KV",
            )?,
            "dSpark batched update projected KV",
        )?,
        key_output: allocate(arena, output_bytes, "dSpark batched update keys")?,
        value_output: allocate(arena, output_bytes, "dSpark batched update values")?,
        reference_fused_hidden: allocate(
            arena,
            hidden_bytes,
            "dSpark batched update reference fused hidden",
        )?,
        reference_key_output: allocate(
            arena,
            output_bytes,
            "dSpark batched update reference keys",
        )?,
        reference_value_output: allocate(
            arena,
            output_bytes,
            "dSpark batched update reference values",
        )?,
        eager_fused_hidden: allocate(
            arena,
            hidden_bytes,
            "dSpark batched update eager fused hidden",
        )?,
        eager_key_output: allocate(arena, output_bytes, "dSpark batched update eager keys")?,
        eager_value_output: allocate(arena, output_bytes, "dSpark batched update eager values")?,
        k_cache,
        v_cache,
        row_request_ids: allocate(
            arena,
            checked_mul(rows, 4, "batched update request IDs")?,
            "dSpark batched update request IDs",
        )?,
        row_positions: allocate(
            arena,
            checked_mul(rows, 4, "batched update positions")?,
            "dSpark batched update positions",
        )?,
        row_cache_positions: allocate(
            arena,
            checked_mul(rows, 4, "batched update cache positions")?,
            "dSpark batched update cache positions",
        )?,
        block_tables,
    })
}

fn upload_anchor_tokens(
    library: &'static NativeLibrary,
    query_token_buffer: GlmrtDeviceBuffer,
    head_anchor_buffer: GlmrtDeviceBuffer,
    anchors: &[u32],
    query_config: DsparkQueryBenchConfig,
) -> Result<()> {
    let query_tokens = query_token_ids(anchors, query_config)?;
    library
        .copy_h2d(query_token_buffer, as_bytes(&query_tokens))
        .context("uploading dSpark static query tokens")?;
    let head_anchors = anchors
        .iter()
        .map(|token| *token as i64)
        .collect::<Vec<_>>();
    library
        .copy_h2d(head_anchor_buffer, as_bytes(&head_anchors))
        .context("uploading dSpark static head anchors")
}

fn normalized_anchor(candidate: i64, mask_token_id: usize) -> u32 {
    let mut token = candidate.rem_euclid(DSPARK_QUERY_VOCAB as i64) as usize;
    if token == mask_token_id {
        token = (token + 1) % DSPARK_QUERY_VOCAB;
    }
    token as u32
}

fn validate_config(config: DsparkStaticBenchConfig) -> Result<()> {
    anyhow::ensure!(
        (1..=DSPARK_BODY_LAYERS).contains(&config.draft_layers),
        "dSpark static draft layer count must be between 1 and {DSPARK_BODY_LAYERS}"
    );
    anyhow::ensure!(
        matches!(config.active_requests, 1 | 2 | 4),
        "dSpark static active requests must be 1, 2, or 4"
    );
    anyhow::ensure!(
        matches!(
            (
                config.query_rows,
                config.proposal_tokens,
                config.proposal_start_row
            ),
            (8, 8, 0) | (16, 15, 1)
        ),
        "dSpark static rows/proposals/start must be 8/8/0 or 16/15/1"
    );
    anyhow::ensure!(
        config.accepted_rows_per_request > 0,
        "dSpark static accepted rows must be positive"
    );
    let update_rows = config
        .active_requests
        .checked_mul(config.accepted_rows_per_request)
        .context("dSpark static update row count overflow")?;
    anyhow::ensure!(
        matches!(update_rows, 1 | 4 | 8 | 16),
        "dSpark static packed update rows must fit the 1/4/8/16 graph registry"
    );
    let body_kv = config
        .context_tokens
        .checked_add(config.accepted_rows_per_request)
        .and_then(|tokens| tokens.checked_add(config.query_rows))
        .context("dSpark static body KV length overflow")?;
    anyhow::ensure!(
        body_kv <= config.kv_capacity_tokens,
        "dSpark static body KV length exceeds capacity"
    );
    anyhow::ensure!(
        i32::try_from(body_kv).is_ok(),
        "dSpark static body KV length does not fit i32"
    );
    anyhow::ensure!(
        matches!(config.page_size, 16 | 32 | 64 | 128),
        "dSpark static page size must be 16, 32, 64, or 128"
    );
    anyhow::ensure!(
        config.mask_token_id < DSPARK_QUERY_VOCAB,
        "dSpark static mask token is outside the vocabulary"
    );
    anyhow::ensure!(
        DSPARK_QUERY_HIDDEN == DSPARK_BODY_HIDDEN
            && DSPARK_BODY_HIDDEN == DSPARK_HEAD_HIDDEN
            && config.draft_layers <= DSPARK_UPDATE_LAYERS
            && DSPARK_BODY_HEADS == DSPARK_UPDATE_HEADS
            && DSPARK_BODY_HEAD_DIM == DSPARK_UPDATE_HEAD_DIM,
        "dSpark static inter-stage geometry changed"
    );
    anyhow::ensure!(
        config.iterations > 0 && config.repeats > 0,
        "dSpark static benchmark iterations and repeats must be positive"
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
        "dSpark static confidence byte lengths are invalid"
    );
    let mut max_abs = 0.0_f32;
    for (left, right) in left.chunks_exact(4).zip(right.chunks_exact(4)) {
        let left = f32::from_le_bytes(left.try_into().expect("four-byte confidence chunk"));
        let right = f32::from_le_bytes(right.try_into().expect("four-byte confidence chunk"));
        anyhow::ensure!(
            left.is_finite() && right.is_finite(),
            "dSpark static confidence contains a non-finite value"
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
    use super::{validate_config, DsparkStaticBenchConfig};
    use crate::commands::real_full::dspark_kv::DsparkKvStorage;

    fn config() -> DsparkStaticBenchConfig {
        DsparkStaticBenchConfig {
            draft_layers: 5,
            active_requests: 4,
            query_rows: 16,
            proposal_tokens: 15,
            proposal_start_row: 1,
            accepted_rows_per_request: 4,
            context_tokens: 1_024,
            kv_capacity_tokens: 256 * 1_024,
            allocate_full_kv_capacity: false,
            page_size: 64,
            kv_storage: DsparkKvStorage::Bf16,
            mask_token_id: 154_856,
            warmup: 2,
            iterations: 10,
            repeats: 3,
            seed: 17,
        }
    }

    #[test]
    fn accepts_static_concurrency_and_checkpoint_shapes() {
        for active_requests in [1, 2, 4] {
            validate_config(DsparkStaticBenchConfig {
                active_requests,
                ..config()
            })
            .unwrap();
        }
        validate_config(DsparkStaticBenchConfig {
            draft_layers: 3,
            query_rows: 8,
            proposal_tokens: 8,
            proposal_start_row: 0,
            ..config()
        })
        .unwrap();
    }

    #[test]
    fn rejects_padded_or_mismatched_static_shapes() {
        let mut invalid = config();
        invalid.accepted_rows_per_request = 3;
        assert!(validate_config(invalid).is_err());
        invalid = config();
        invalid.query_rows = 15;
        assert!(validate_config(invalid).is_err());
        invalid = config();
        invalid.kv_capacity_tokens = invalid.context_tokens;
        assert!(validate_config(invalid).is_err());
        invalid = config();
        invalid.context_tokens = i32::MAX as usize;
        invalid.kv_capacity_tokens = invalid.context_tokens + 64;
        assert!(validate_config(invalid).is_err());
    }
}
