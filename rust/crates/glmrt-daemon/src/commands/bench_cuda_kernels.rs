use crate::cli::BenchCudaKernelsArgs;
use crate::python_graph_capture::{
    coordinator_python_capture_enabled, launch_python_graph_capture, PythonDeviceBufferArg,
    PythonGraphCaptureLaunch, PythonKernelArg,
};
use anyhow::{Context, Result};
use glmrt_core::GLM52_ROUTED_SCALING_FACTOR;
use glmrt_ffi::{
    GlmrtCudaGraphCaptureInfo, GlmrtDeviceBuffer, GlmrtHostBuffer, NativeLibrary,
    GLMRT_CUDA_ROUTER_TOPK_MAX_K, GLMRT_CUDA_SAMPLE_TOPK_MAX_K,
};
use serde_json::json;
use std::{
    env,
    ffi::c_void,
    path::{Path, PathBuf},
    slice,
    time::Instant,
};

const MLA_KV_FP8_NOPE_VALUES: usize = 512;
const MLA_KV_FP8_ROPE_VALUES: usize = 64;
const MLA_KV_FP8_PROJECTED_VALUES: usize = MLA_KV_FP8_NOPE_VALUES + MLA_KV_FP8_ROPE_VALUES;
const MLA_KV_FP8_PROJECTED_STRIDE_BYTES: usize = MLA_KV_FP8_PROJECTED_VALUES * 2;
const MLA_KV_FP8_PACKED_BYTES: usize = 656;
const MLA_KV_MXFP4_NOPE_VALUES: usize = 512;
const MLA_KV_MXFP4_ROPE_VALUES: usize = 64;
const MLA_KV_MXFP4_PROJECTED_VALUES: usize = MLA_KV_MXFP4_NOPE_VALUES + MLA_KV_MXFP4_ROPE_VALUES;
const MLA_KV_MXFP4_PROJECTED_STRIDE_BYTES: usize = MLA_KV_MXFP4_PROJECTED_VALUES * 2;
const MLA_KV_MXFP4_PACKED_BYTES: usize = MLA_KV_MXFP4_NOPE_VALUES / 2
    + MLA_KV_MXFP4_NOPE_VALUES / 16
    + 16
    + MLA_KV_MXFP4_ROPE_VALUES * 2;
const GLM52_MLA_ATTENTION_HEADS: usize = 64;
const GLM52_MLA_QK_NOPE_HEAD_DIM: usize = 192;
const GLM52_MLA_QK_ROPE_HEAD_DIM: usize = 64;
const GLM52_MLA_V_HEAD_DIM: usize = 256;
const B12X_MLA_ATTENTION_HEADS: usize = 8;
const B12X_MLA_KV_LORA_RANK: usize = 512;
const B12X_MLA_QK_ROPE_HEAD_DIM: usize = 64;
const B12X_MLA_CAPTURE_MODULE: &str = "b12x_mla_capture";
const B12X_MLA_CAPTURE_FUNCTION: &str = "capture_mla_rope_attention";
const KERNEL_FILTERS: &[&str] = &[
    "all",
    "triton",
    "triton-swaps",
    "cublas",
    "cub",
    "rmsnorm",
    "rmsnorm_bf16",
    "residual-add",
    "residual_add_bf16",
    "moe-response-fp8-tail",
    "moe-response-low-precision-tail",
    "moe_response_fp8_e4m3_row_scaled_tail",
    "moe_response_low_precision_tail",
    "linear",
    "linear_bf16",
    "linear_bf16_cublas",
    "mlp",
    "dense-mlp",
    "silu_gated_mlp_rows_bf16",
    "triton_silu_gated_mlp_rows_bf16_graph",
    "router",
    "router_topk_bf16",
    "router_topk_bf16_cub",
    "triton_router_topk_bf16_graph",
    "sampling",
    "sampler",
    "lm-head",
    "logits-sampling",
    "logits_sample_topk_topp_f32",
    "logits_sample_topk_topp_f32_cub",
    "lm_head_sample_topk_topp_bf16",
    "lm_head_sample_topk_topp_bf16_cub",
    "triton_lm_head_sample_topk_topp_bf16_graph",
    "attention",
    "mla-rope-attention",
    "mla_rope_attention_bf16",
    "mla_rope_attention_bf16_suffix",
    "b12x_mla_rope_attention_bf16_graph",
    "kv-pack",
    "mla-kv-pack",
    "mla_kv_pack_fp8_ds_mla",
    "mla_kv_pack_mxfp4_ds_mla",
    "triton_mla_kv_pack_fp8_ds_mla_graph",
    "embedding",
    "embedding_lookup_bf16",
    "nvfp4",
    "nvfp4-route",
    "nvfp4_route_bf16_staged_accumulate_pack",
    "layer-sweep-replay",
    "phase0-layer-sweep-replay",
    "phase0_layer_sweep_replay",
];

struct DeviceAllocation<'a> {
    library: &'a NativeLibrary,
    buffer: GlmrtDeviceBuffer,
}

impl<'a> DeviceAllocation<'a> {
    fn new(library: &'a NativeLibrary, bytes: usize, label: &str) -> Result<Self> {
        let buffer = library
            .alloc_device_buffer(bytes)
            .with_context(|| format!("allocating CUDA device buffer for {label}"))?;
        Ok(Self { library, buffer })
    }

    fn buffer(&self) -> GlmrtDeviceBuffer {
        self.buffer
    }
}

impl Drop for DeviceAllocation<'_> {
    fn drop(&mut self) {
        let _ = self.library.free_device_buffer(&mut self.buffer);
    }
}

struct HostAllocation<'a> {
    library: &'a NativeLibrary,
    buffer: GlmrtHostBuffer,
}

impl<'a> HostAllocation<'a> {
    fn new(library: &'a NativeLibrary, bytes: usize, label: &str) -> Result<Self> {
        let buffer = library
            .alloc_host_buffer(bytes)
            .with_context(|| format!("allocating pinned host buffer for {label}"))?;
        Ok(Self { library, buffer })
    }

    fn buffer(&self) -> GlmrtHostBuffer {
        self.buffer
    }
}

impl Drop for HostAllocation<'_> {
    fn drop(&mut self) {
        let _ = self.library.free_host_buffer(&mut self.buffer);
    }
}

struct CudaStream<'a> {
    library: &'a NativeLibrary,
    stream: *mut c_void,
}

impl<'a> CudaStream<'a> {
    fn new(library: &'a NativeLibrary) -> Result<Self> {
        let stream = library
            .cuda_stream_create()
            .context("creating CUDA microbenchmark stream")?;
        Ok(Self { library, stream })
    }

    fn raw(&self) -> *mut c_void {
        self.stream
    }

    unsafe fn synchronize(&self) -> Result<()> {
        unsafe { self.library.cuda_stream_synchronize(self.stream) }
            .context("synchronizing CUDA microbenchmark stream")
    }
}

impl Drop for CudaStream<'_> {
    fn drop(&mut self) {
        let _ = unsafe { self.library.cuda_stream_destroy(self.stream) };
    }
}

struct CudaEvent<'a> {
    library: &'a NativeLibrary,
    event: *mut c_void,
}

impl<'a> CudaEvent<'a> {
    fn new(library: &'a NativeLibrary) -> Result<Self> {
        let event = library
            .cuda_event_create()
            .context("creating CUDA microbenchmark event")?;
        Ok(Self { library, event })
    }

    fn raw(&self) -> *mut c_void {
        self.event
    }
}

impl Drop for CudaEvent<'_> {
    fn drop(&mut self) {
        if !self.event.is_null() {
            let _ = unsafe { self.library.cuda_event_destroy(self.event) };
            self.event = std::ptr::null_mut();
        }
    }
}

#[derive(Clone, Copy)]
struct Timing {
    avg_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

pub(crate) fn run_bench_cuda_kernels(args: BenchCudaKernelsArgs) -> Result<()> {
    validate_args(&args)?;
    let native_lib = native_library_path(&args);
    if !native_lib.exists() {
        if args.require_cuda {
            anyhow::bail!("CUDA native library not found: {}", native_lib.display());
        }
        emit_status(
            &native_lib,
            "native_lib_missing",
            "CUDA native library not found",
        )?;
        return Ok(());
    }

    let library = unsafe { NativeLibrary::load(&native_lib) }
        .with_context(|| format!("loading CUDA native library {}", native_lib.display()))?;
    let info = library
        .cuda_device_info(0)
        .context("querying CUDA device for kernel microbenchmarks")?;
    if info.cuda_available != 1 {
        if args.require_cuda {
            anyhow::bail!("CUDA device is unavailable for kernel microbenchmarks");
        }
        emit_status(
            &native_lib,
            "cuda_unavailable",
            "CUDA device is unavailable",
        )?;
        return Ok(());
    }

    let stream = CudaStream::new(&library)?;
    if kernel_selected(&args, "rmsnorm_bf16", &["rmsnorm"]) {
        bench_rmsnorm_bf16(&library, &stream, &native_lib, &args)?;
    }
    if kernel_selected(&args, "residual_add_bf16", &["residual-add"]) {
        bench_residual_add_bf16(&library, &stream, &native_lib, &args)?;
    }
    if kernel_selected_explicit(
        &args,
        "moe_response_low_precision_tail",
        &[
            "moe-response-low-precision-tail",
            "moe-response-fp8-tail",
            "moe_response_fp8_e4m3_row_scaled_tail",
        ],
    ) {
        bench_moe_response_fp8_tail(&library, &stream, &native_lib, &args)?;
    }
    if kernel_selected(&args, "linear_bf16", &["linear"]) {
        bench_linear_bf16(&library, &stream, &native_lib, &args)?;
    }
    if kernel_selected(&args, "linear_bf16_cublas", &["linear", "cublas"]) {
        bench_linear_bf16_cublas(&library, &stream, &native_lib, &args)?;
    }
    if kernel_selected(
        &args,
        "silu_gated_mlp_rows_bf16",
        &["mlp", "dense-mlp", "triton-swaps"],
    ) {
        bench_dense_mlp_bf16(&library, &stream, &native_lib, &args)?;
    }
    if kernel_selected(&args, "router_topk_bf16", &["router", "triton-swaps"]) {
        bench_router_topk_bf16(&library, &stream, &native_lib, &args)?;
    }
    if kernel_selected(&args, "router_topk_bf16_cub", &["router", "cub"]) {
        bench_router_topk_bf16_cub(&library, &stream, &native_lib, &args)?;
    }
    if kernel_selected(
        &args,
        "lm_head_sample_topk_topp_bf16",
        &["sampling", "sampler", "lm-head", "triton-swaps"],
    ) {
        bench_lm_head_sample_topk_topp_bf16(&library, &stream, &native_lib, &args)?;
    }
    if kernel_selected(
        &args,
        "lm_head_sample_topk_topp_bf16_cub",
        &["sampling", "sampler", "lm-head", "cub"],
    ) {
        bench_lm_head_sample_topk_topp_bf16_cub(&library, &stream, &native_lib, &args)?;
    }
    if kernel_selected_explicit(&args, "logits_sample_topk_topp_f32", &["logits-sampling"]) {
        bench_logits_sample_topk_topp_f32(&library, &stream, &native_lib, &args)?;
    }
    if kernel_selected_explicit(
        &args,
        "logits_sample_topk_topp_f32_cub",
        &["logits-sampling", "cub"],
    ) {
        bench_logits_sample_topk_topp_f32_cub(&library, &stream, &native_lib, &args)?;
    }
    if kernel_selected(
        &args,
        "mla_kv_pack_fp8_ds_mla",
        &["kv-pack", "mla-kv-pack", "triton-swaps"],
    ) {
        bench_mla_kv_pack_fp8_ds_mla(&library, &stream, &native_lib, &args)?;
    }
    if kernel_selected(
        &args,
        "mla_rope_attention_bf16",
        &["attention", "mla-rope-attention"],
    ) {
        bench_mla_rope_attention_bf16(&library, &stream, &native_lib, &args)?;
    }
    if kernel_selected_explicit(
        &args,
        "mla_rope_attention_bf16_suffix",
        &["mla-rope-attention-suffix"],
    ) {
        bench_mla_rope_attention_bf16_suffix(&library, &stream, &native_lib, &args)?;
    }
    if kernel_selected(
        &args,
        "mla_kv_pack_mxfp4_ds_mla",
        &["kv-pack", "mla-kv-pack"],
    ) {
        bench_mla_kv_pack_mxfp4_ds_mla(&library, &stream, &native_lib, &args)?;
    }
    if coordinator_python_capture_enabled() {
        if kernel_selected_no_all(
            &args,
            "b12x_mla_rope_attention_bf16_graph",
            &["b12x-mla-rope-attention"],
        ) {
            bench_b12x_mla_rope_attention_bf16_graph(&library, &stream, &native_lib, &args)?;
        }
        if kernel_selected(
            &args,
            "triton_silu_gated_mlp_rows_bf16_graph",
            &["triton", "mlp", "dense-mlp", "triton-swaps"],
        ) {
            bench_triton_dense_mlp_bf16_graph(&library, &stream, &native_lib, &args)?;
        }
        if kernel_selected(
            &args,
            "triton_router_topk_bf16_graph",
            &["triton", "router", "triton-swaps"],
        ) {
            bench_triton_router_topk_bf16_graph(&library, &stream, &native_lib, &args)?;
        }
        if kernel_selected(
            &args,
            "triton_lm_head_sample_topk_topp_bf16_graph",
            &["triton", "sampling", "sampler", "lm-head", "triton-swaps"],
        ) {
            bench_triton_lm_head_sample_topk_topp_bf16_graph(
                &library,
                &stream,
                &native_lib,
                &args,
            )?;
        }
        if kernel_selected(
            &args,
            "triton_mla_kv_pack_fp8_ds_mla_graph",
            &["triton", "kv-pack", "mla-kv-pack", "triton-swaps"],
        ) {
            bench_triton_mla_kv_pack_fp8_ds_mla_graph(&library, &stream, &native_lib, &args)?;
        }
    }
    if kernel_selected(&args, "embedding_lookup_bf16", &["embedding"]) {
        bench_embedding_lookup_bf16(&library, &stream, &native_lib, &args)?;
    }
    if kernel_selected(
        &args,
        "nvfp4_route_bf16_staged_accumulate_pack",
        &["nvfp4", "nvfp4-route"],
    ) {
        bench_nvfp4_route_staged(&library, &stream, &native_lib, &args)?;
    }
    if kernel_selected_explicit(
        &args,
        "phase0_layer_sweep_replay",
        &["layer-sweep-replay", "phase0-layer-sweep-replay"],
    ) {
        bench_phase0_layer_sweep_replay(&library, &stream, &native_lib, &args)?;
    }
    Ok(())
}

fn validate_args(args: &BenchCudaKernelsArgs) -> Result<()> {
    if args.rows == 0
        || args.hidden_dim == 0
        || args.intermediate_dim == 0
        || args.output_dim == 0
        || args.vocab == 0
        || args.routes == 0
        || args.iterations == 0
    {
        anyhow::bail!("bench-cuda-kernels dimensions, routes, and iterations must all be nonzero");
    }
    if args.top_k == 0 {
        anyhow::bail!("bench-cuda-kernels top_k must be nonzero");
    }
    for requested in &args.kernels {
        let requested = requested.trim();
        if requested.is_empty() {
            anyhow::bail!("bench-cuda-kernels kernel filters must be non-empty");
        }
        if !KERNEL_FILTERS
            .iter()
            .any(|known| filter_matches(requested, known))
        {
            anyhow::bail!(
                "unknown bench-cuda-kernels kernel filter '{requested}'; known filters: {}",
                KERNEL_FILTERS.join(", ")
            );
        }
    }
    Ok(())
}

fn kernel_selected(args: &BenchCudaKernelsArgs, kernel: &str, groups: &[&str]) -> bool {
    args.kernels.is_empty()
        || args.kernels.iter().any(|requested| {
            let requested = requested.trim();
            filter_matches(requested, "all")
                || filter_matches(requested, kernel)
                || groups.iter().any(|group| filter_matches(requested, group))
        })
}

fn kernel_selected_explicit(args: &BenchCudaKernelsArgs, kernel: &str, groups: &[&str]) -> bool {
    !args.kernels.is_empty()
        && args.kernels.iter().any(|requested| {
            let requested = requested.trim();
            filter_matches(requested, "all")
                || filter_matches(requested, kernel)
                || groups.iter().any(|group| filter_matches(requested, group))
        })
}

fn kernel_selected_no_all(args: &BenchCudaKernelsArgs, kernel: &str, groups: &[&str]) -> bool {
    !args.kernels.is_empty()
        && args.kernels.iter().any(|requested| {
            let requested = requested.trim();
            filter_matches(requested, kernel)
                || groups.iter().any(|group| filter_matches(requested, group))
        })
}

fn filter_matches(requested: &str, candidate: &str) -> bool {
    requested.eq_ignore_ascii_case(candidate)
        || requested
            .replace('-', "_")
            .eq_ignore_ascii_case(&candidate.replace('-', "_"))
}

fn router_top_k(args: &BenchCudaKernelsArgs, experts: usize, label: &str) -> Result<usize> {
    if args.top_k > experts || args.top_k > GLMRT_CUDA_ROUTER_TOPK_MAX_K {
        anyhow::bail!(
            "{label} invalid top_k={} for experts={experts}; max supported top_k={GLMRT_CUDA_ROUTER_TOPK_MAX_K}",
            args.top_k
        );
    }
    Ok(args.top_k)
}

fn sample_top_k(args: &BenchCudaKernelsArgs, label: &str) -> Result<usize> {
    if args.top_k > args.vocab || args.top_k > GLMRT_CUDA_SAMPLE_TOPK_MAX_K {
        anyhow::bail!(
            "{label} invalid top_k={} for vocab={}; max supported top_k={GLMRT_CUDA_SAMPLE_TOPK_MAX_K}",
            args.top_k,
            args.vocab
        );
    }
    Ok(args.top_k)
}

fn native_library_path(args: &BenchCudaKernelsArgs) -> PathBuf {
    args.native_lib
        .clone()
        .or_else(|| env::var_os("GLMRT_NATIVE_LIB").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("native/build-cuda/libglmrt_native.so"))
}

fn emit_status(native_lib: &Path, status: &str, reason: &str) -> Result<()> {
    println!(
        "{}",
        serde_json::to_string(&json!({
            "benchmark": "cuda_kernel_microbench",
            "status": status,
            "reason": reason,
            "native_lib": native_lib.display().to_string(),
        }))?
    );
    Ok(())
}

fn bench_rmsnorm_bf16(
    library: &NativeLibrary,
    stream: &CudaStream<'_>,
    native_lib: &Path,
    args: &BenchCudaKernelsArgs,
) -> Result<()> {
    let values = args
        .rows
        .checked_mul(args.hidden_dim)
        .context("rmsnorm value count overflow")?;
    let input = bf16_pattern(values);
    let weight = bf16_pattern(args.hidden_dim);
    let x_buffer = upload(library, u16_bytes(&input), "rmsnorm input")?;
    let weight_buffer = upload(library, u16_bytes(&weight), "rmsnorm weight")?;
    let out_buffer = DeviceAllocation::new(library, values * 2, "rmsnorm output")?;
    let rows = i32::try_from(args.rows).context("rmsnorm rows exceed i32")?;
    let hidden = i32::try_from(args.hidden_dim).context("rmsnorm hidden_dim exceeds i32")?;
    let timing = time_kernel(args, || unsafe {
        library.cuda_rmsnorm_bf16_async(
            x_buffer.buffer(),
            weight_buffer.buffer(),
            out_buffer.buffer(),
            rows,
            hidden,
            1.0e-5,
            stream.raw(),
        )?;
        stream.synchronize()
    })?;
    emit_timing(
        native_lib,
        "rmsnorm_bf16",
        &json!({
            "rows": args.rows,
            "hidden_dim": args.hidden_dim,
            "logical_payload_bytes": values * 2 + args.hidden_dim * 2 + values * 2,
        }),
        args,
        timing,
    )
}

fn bench_residual_add_bf16(
    library: &NativeLibrary,
    stream: &CudaStream<'_>,
    native_lib: &Path,
    args: &BenchCudaKernelsArgs,
) -> Result<()> {
    let values = args
        .rows
        .checked_mul(args.hidden_dim)
        .context("residual-add value count overflow")?;
    let residual = bf16_pattern(values);
    let delta = bf16_pattern(values);
    let residual_buffer = upload(library, u16_bytes(&residual), "residual-add residual")?;
    let delta_buffer = upload(library, u16_bytes(&delta), "residual-add delta")?;
    let out_buffer = DeviceAllocation::new(library, values * 2, "residual-add output")?;
    let timing = time_kernel(args, || unsafe {
        library.cuda_residual_add_bf16_async(
            residual_buffer.buffer(),
            delta_buffer.buffer(),
            out_buffer.buffer(),
            values,
            stream.raw(),
        )?;
        stream.synchronize()
    })?;
    emit_timing(
        native_lib,
        "residual_add_bf16",
        &json!({
            "rows": args.rows,
            "hidden_dim": args.hidden_dim,
            "logical_payload_bytes": values * 2 * 3,
        }),
        args,
        timing,
    )
}

fn bench_moe_response_fp8_tail(
    library: &NativeLibrary,
    stream: &CudaStream<'_>,
    native_lib: &Path,
    args: &BenchCudaKernelsArgs,
) -> Result<()> {
    anyhow::ensure!(
        args.hidden_dim > 0 && args.hidden_dim % 16 == 0,
        "MoE low-precision response benchmark hidden_dim must be a positive multiple of 16"
    );
    let values = args
        .rows
        .checked_mul(args.hidden_dim)
        .context("MoE FP8 response benchmark value count overflow")?;
    let row_stride_bytes = args
        .hidden_dim
        .checked_add(std::mem::size_of::<f32>())
        .context("MoE FP8 response benchmark row stride overflow")?;
    let output_bytes = args
        .rows
        .checked_mul(row_stride_bytes)
        .context("MoE FP8 response benchmark output byte count overflow")?;
    let source = (0..values)
        .map(|index| -4.0_f32 + (index % 257) as f32 * 0.03125)
        .collect::<Vec<_>>();
    let row_indices = (0..args.rows)
        .map(|row| u32::try_from(row).context("MoE FP8 response row index exceeds u32"))
        .collect::<Result<Vec<_>>>()?;
    let source_buffer = upload(library, f32_bytes(&source), "MoE FP8 response source")?;
    let row_indices_buffer = upload(
        library,
        u32_bytes(&row_indices),
        "MoE FP8 response row indices",
    )?;
    let output_buffer = DeviceAllocation::new(library, output_bytes, "MoE FP8 response output")?;
    let host_output = HostAllocation::new(library, output_bytes, "MoE FP8 response D2H")?;
    let mapped_host_output =
        HostAllocation::new(library, output_bytes, "MoE FP8 mapped response output")?;
    let mapped_output_buffer =
        library.cuda_host_buffer_device_alias(mapped_host_output.buffer())?;
    let unpacked_output = DeviceAllocation::new(
        library,
        values * std::mem::size_of::<f32>(),
        "MoE FP8 unpacked response",
    )?;

    let pack_timing = time_kernel(args, || unsafe {
        library.cuda_gather_rows_f32_to_fp8_e4m3_row_scaled_async(
            source_buffer.buffer(),
            args.rows,
            row_indices_buffer.buffer(),
            output_buffer.buffer(),
            args.rows,
            args.hidden_dim,
            row_stride_bytes,
            stream.raw(),
        )?;
        stream.synchronize()
    })?;
    emit_timing(
        native_lib,
        "moe_response_fp8_e4m3_row_scaled_pack",
        &json!({
            "rows": args.rows,
            "hidden_dim": args.hidden_dim,
            "row_stride_bytes": row_stride_bytes,
            "response_bytes": output_bytes,
            "destination": "device",
        }),
        args,
        pack_timing,
    )?;

    let eager_timing = time_kernel(args, || unsafe {
        library.cuda_gather_rows_f32_to_fp8_e4m3_row_scaled_async(
            source_buffer.buffer(),
            args.rows,
            row_indices_buffer.buffer(),
            output_buffer.buffer(),
            args.rows,
            args.hidden_dim,
            row_stride_bytes,
            stream.raw(),
        )?;
        library.copy_d2h_host_buffer_async(
            host_output.buffer(),
            output_buffer.buffer(),
            output_bytes,
            stream.raw(),
        )?;
        stream.synchronize()
    })?;
    emit_timing(
        native_lib,
        "moe_response_fp8_e4m3_row_scaled_tail_eager",
        &json!({
            "rows": args.rows,
            "hidden_dim": args.hidden_dim,
            "row_stride_bytes": row_stride_bytes,
            "response_bytes": output_bytes,
        }),
        args,
        eager_timing,
    )?;

    let mapped_timing = time_kernel(args, || unsafe {
        library.cuda_gather_rows_f32_to_fp8_e4m3_row_scaled_async(
            source_buffer.buffer(),
            args.rows,
            row_indices_buffer.buffer(),
            mapped_output_buffer,
            args.rows,
            args.hidden_dim,
            row_stride_bytes,
            stream.raw(),
        )?;
        stream.synchronize()
    })?;
    emit_timing(
        native_lib,
        "moe_response_fp8_e4m3_row_scaled_pack_mapped_host",
        &json!({
            "rows": args.rows,
            "hidden_dim": args.hidden_dim,
            "row_stride_bytes": row_stride_bytes,
            "response_bytes": output_bytes,
            "destination": "cuda_mapped_host",
            "speedup_vs_device_pack_d2h": eager_timing.avg_ms / mapped_timing.avg_ms,
        }),
        args,
        mapped_timing,
    )?;

    library.cuda_zero_f32(unpacked_output.buffer(), values)?;
    let unpack_timing = time_kernel(args, || unsafe {
        library.cuda_scatter_add_rows_fp8_e4m3_row_scaled_to_f32_async(
            output_buffer.buffer(),
            row_stride_bytes,
            row_indices_buffer.buffer(),
            unpacked_output.buffer(),
            args.rows,
            args.rows,
            args.hidden_dim,
            stream.raw(),
        )?;
        stream.synchronize()
    })?;
    emit_timing(
        native_lib,
        "moe_response_fp8_e4m3_row_scaled_unpack_scatter",
        &json!({
            "rows": args.rows,
            "hidden_dim": args.hidden_dim,
            "row_stride_bytes": row_stride_bytes,
            "response_bytes": output_bytes,
            "destination_dtype": "f32",
        }),
        args,
        unpack_timing,
    )?;

    let graph = unsafe {
        stream.synchronize()?;
        library
            .cuda_graph_begin_capture(stream.raw())
            .context("beginning MoE FP8 response tail graph capture")?;
        library.cuda_gather_rows_f32_to_fp8_e4m3_row_scaled_async(
            source_buffer.buffer(),
            args.rows,
            row_indices_buffer.buffer(),
            output_buffer.buffer(),
            args.rows,
            args.hidden_dim,
            row_stride_bytes,
            stream.raw(),
        )?;
        library.copy_d2h_host_buffer_async(
            host_output.buffer(),
            output_buffer.buffer(),
            output_bytes,
            stream.raw(),
        )?;
        BenchCudaGraph::new(
            library,
            library
                .cuda_graph_end_capture_retained(stream.raw())
                .context("ending MoE FP8 response tail graph capture")?,
        )?
    };
    let graph_timing = time_kernel(args, || unsafe {
        library
            .cuda_graph_launch(graph.graph_exec(), stream.raw())
            .context("launching MoE FP8 response tail graph")?;
        stream.synchronize()
    })?;
    emit_timing(
        native_lib,
        "moe_response_fp8_e4m3_row_scaled_tail_graph",
        &json!({
            "rows": args.rows,
            "hidden_dim": args.hidden_dim,
            "row_stride_bytes": row_stride_bytes,
            "response_bytes": output_bytes,
            "graph_nodes": graph.node_count(),
            "graph_kernel_nodes": graph.kernel_node_count(),
            "graph_memcpy_nodes": graph.memcpy_node_count(),
            "speedup_vs_eager": eager_timing.avg_ms / graph_timing.avg_ms,
        }),
        args,
        graph_timing,
    )?;

    let nvfp4_row_stride_bytes = args
        .hidden_dim
        .checked_div(2)
        .and_then(|packed| packed.checked_add(args.hidden_dim / 16))
        .context("MoE NVFP4 response row stride overflow")?;
    let nvfp4_output_bytes = args
        .rows
        .checked_mul(nvfp4_row_stride_bytes)
        .context("MoE NVFP4 response byte count overflow")?;
    let nvfp4_output =
        DeviceAllocation::new(library, nvfp4_output_bytes, "MoE NVFP4 response output")?;
    let nvfp4_host = HostAllocation::new(library, nvfp4_output_bytes, "MoE NVFP4 response D2H")?;
    let nvfp4_mapped_host = HostAllocation::new(
        library,
        nvfp4_output_bytes,
        "MoE NVFP4 mapped response output",
    )?;
    let nvfp4_mapped_output = library.cuda_host_buffer_device_alias(nvfp4_mapped_host.buffer())?;
    let nvfp4_unpacked = DeviceAllocation::new(
        library,
        values * std::mem::size_of::<f32>(),
        "MoE NVFP4 unpacked response",
    )?;

    let nvfp4_pack_timing = time_kernel(args, || unsafe {
        library.cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3_async(
            source_buffer.buffer(),
            args.rows,
            row_indices_buffer.buffer(),
            nvfp4_output.buffer(),
            args.rows,
            args.hidden_dim,
            nvfp4_row_stride_bytes,
            stream.raw(),
        )?;
        stream.synchronize()
    })?;
    emit_timing(
        native_lib,
        "moe_response_nvfp4_e2m1_fp8_e4m3_pack",
        &json!({
            "rows": args.rows,
            "hidden_dim": args.hidden_dim,
            "row_stride_bytes": nvfp4_row_stride_bytes,
            "response_bytes": nvfp4_output_bytes,
            "destination": "device",
            "slowdown_vs_fp8_pack": nvfp4_pack_timing.avg_ms / pack_timing.avg_ms,
        }),
        args,
        nvfp4_pack_timing,
    )?;

    let nvfp4_eager_timing = time_kernel(args, || unsafe {
        library.cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3_async(
            source_buffer.buffer(),
            args.rows,
            row_indices_buffer.buffer(),
            nvfp4_output.buffer(),
            args.rows,
            args.hidden_dim,
            nvfp4_row_stride_bytes,
            stream.raw(),
        )?;
        library.copy_d2h_host_buffer_async(
            nvfp4_host.buffer(),
            nvfp4_output.buffer(),
            nvfp4_output_bytes,
            stream.raw(),
        )?;
        stream.synchronize()
    })?;
    emit_timing(
        native_lib,
        "moe_response_nvfp4_e2m1_fp8_e4m3_tail_eager",
        &json!({
            "rows": args.rows,
            "hidden_dim": args.hidden_dim,
            "row_stride_bytes": nvfp4_row_stride_bytes,
            "response_bytes": nvfp4_output_bytes,
            "slowdown_vs_fp8_device_pack_d2h": nvfp4_eager_timing.avg_ms / eager_timing.avg_ms,
        }),
        args,
        nvfp4_eager_timing,
    )?;

    let nvfp4_mapped_timing = time_kernel(args, || unsafe {
        library.cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3_async(
            source_buffer.buffer(),
            args.rows,
            row_indices_buffer.buffer(),
            nvfp4_mapped_output,
            args.rows,
            args.hidden_dim,
            nvfp4_row_stride_bytes,
            stream.raw(),
        )?;
        stream.synchronize()
    })?;
    emit_timing(
        native_lib,
        "moe_response_nvfp4_e2m1_fp8_e4m3_pack_mapped_host",
        &json!({
            "rows": args.rows,
            "hidden_dim": args.hidden_dim,
            "row_stride_bytes": nvfp4_row_stride_bytes,
            "response_bytes": nvfp4_output_bytes,
            "destination": "cuda_mapped_host",
            "speedup_vs_device_pack_d2h": nvfp4_eager_timing.avg_ms / nvfp4_mapped_timing.avg_ms,
        }),
        args,
        nvfp4_mapped_timing,
    )?;

    library.cuda_zero_f32(nvfp4_unpacked.buffer(), values)?;
    let nvfp4_unpack_timing = time_kernel(args, || unsafe {
        library.cuda_scatter_add_rows_nvfp4_e2m1_fp8_e4m3_to_f32_async(
            nvfp4_output.buffer(),
            nvfp4_row_stride_bytes,
            row_indices_buffer.buffer(),
            nvfp4_unpacked.buffer(),
            args.rows,
            args.rows,
            args.hidden_dim,
            stream.raw(),
        )?;
        stream.synchronize()
    })?;
    emit_timing(
        native_lib,
        "moe_response_nvfp4_e2m1_fp8_e4m3_unpack_scatter",
        &json!({
            "rows": args.rows,
            "hidden_dim": args.hidden_dim,
            "row_stride_bytes": nvfp4_row_stride_bytes,
            "response_bytes": nvfp4_output_bytes,
            "destination_dtype": "f32",
            "slowdown_vs_fp8_unpack": nvfp4_unpack_timing.avg_ms / unpack_timing.avg_ms,
        }),
        args,
        nvfp4_unpack_timing,
    )?;
    Ok(())
}

fn bench_linear_bf16(
    library: &NativeLibrary,
    stream: &CudaStream<'_>,
    native_lib: &Path,
    args: &BenchCudaKernelsArgs,
) -> Result<()> {
    let input_values = args
        .rows
        .checked_mul(args.hidden_dim)
        .context("linear input value count overflow")?;
    let weight_values = args
        .output_dim
        .checked_mul(args.hidden_dim)
        .context("linear weight value count overflow")?;
    let output_values = args
        .rows
        .checked_mul(args.output_dim)
        .context("linear output value count overflow")?;
    let input = bf16_pattern(input_values);
    let weight = bf16_pattern(weight_values);
    let input_buffer = upload(library, u16_bytes(&input), "linear input")?;
    let weight_buffer = upload(library, u16_bytes(&weight), "linear weight")?;
    let out_buffer = DeviceAllocation::new(library, output_values * 2, "linear output")?;
    let timing = time_kernel(args, || unsafe {
        library.cuda_linear_bf16_async(
            input_buffer.buffer(),
            weight_buffer.buffer(),
            None,
            out_buffer.buffer(),
            args.rows,
            args.hidden_dim,
            args.output_dim,
            stream.raw(),
        )?;
        stream.synchronize()
    })?;
    emit_timing(
        native_lib,
        "linear_bf16",
        &json!({
            "rows": args.rows,
            "input_dim": args.hidden_dim,
            "output_dim": args.output_dim,
            "logical_payload_bytes": (input_values + weight_values + output_values) * 2,
        }),
        args,
        timing,
    )
}

fn bench_linear_bf16_cublas(
    library: &NativeLibrary,
    stream: &CudaStream<'_>,
    native_lib: &Path,
    args: &BenchCudaKernelsArgs,
) -> Result<()> {
    let input_values = args
        .rows
        .checked_mul(args.hidden_dim)
        .context("cuBLAS linear input value count overflow")?;
    let weight_values = args
        .output_dim
        .checked_mul(args.hidden_dim)
        .context("cuBLAS linear weight value count overflow")?;
    let output_values = args
        .rows
        .checked_mul(args.output_dim)
        .context("cuBLAS linear output value count overflow")?;
    let input = bf16_pattern(input_values);
    let weight = bf16_pattern(weight_values);
    let input_buffer = upload(library, u16_bytes(&input), "cuBLAS linear input")?;
    let weight_buffer = upload(library, u16_bytes(&weight), "cuBLAS linear weight")?;
    let out_buffer = DeviceAllocation::new(library, output_values * 2, "cuBLAS linear output")?;
    let timing = time_kernel(args, || unsafe {
        library.cuda_linear_bf16_cublas_async(
            input_buffer.buffer(),
            weight_buffer.buffer(),
            None,
            out_buffer.buffer(),
            args.rows,
            args.hidden_dim,
            args.output_dim,
            stream.raw(),
        )?;
        stream.synchronize()
    })?;
    emit_timing(
        native_lib,
        "linear_bf16_cublas",
        &json!({
            "rows": args.rows,
            "input_dim": args.hidden_dim,
            "output_dim": args.output_dim,
            "logical_payload_bytes": (input_values + weight_values + output_values) * 2,
        }),
        args,
        timing,
    )
}

fn bench_dense_mlp_bf16(
    library: &NativeLibrary,
    stream: &CudaStream<'_>,
    native_lib: &Path,
    args: &BenchCudaKernelsArgs,
) -> Result<()> {
    let input_values = args
        .rows
        .checked_mul(args.hidden_dim)
        .context("dense MLP input value count overflow")?;
    let gate_values = args
        .intermediate_dim
        .checked_mul(args.hidden_dim)
        .context("dense MLP gate value count overflow")?;
    let down_values = args
        .hidden_dim
        .checked_mul(args.intermediate_dim)
        .context("dense MLP down value count overflow")?;
    let input = bf16_pattern(input_values);
    let gate = bf16_pattern(gate_values);
    let up = bf16_pattern(gate_values);
    let down = bf16_pattern(down_values);
    let input_buffer = upload(library, u16_bytes(&input), "dense MLP input")?;
    let gate_buffer = upload(library, u16_bytes(&gate), "dense MLP gate weight")?;
    let up_buffer = upload(library, u16_bytes(&up), "dense MLP up weight")?;
    let down_buffer = upload(library, u16_bytes(&down), "dense MLP down weight")?;
    let out_buffer = DeviceAllocation::new(library, input_values * 2, "dense MLP output")?;
    let timing = time_kernel(args, || unsafe {
        library.cuda_silu_gated_mlp_rows_bf16_async(
            input_buffer.buffer(),
            gate_buffer.buffer(),
            up_buffer.buffer(),
            down_buffer.buffer(),
            out_buffer.buffer(),
            args.rows,
            args.hidden_dim,
            args.intermediate_dim,
            stream.raw(),
        )?;
        stream.synchronize()
    })?;
    emit_timing(
        native_lib,
        "silu_gated_mlp_rows_bf16",
        &json!({
            "rows": args.rows,
            "hidden_dim": args.hidden_dim,
            "intermediate_dim": args.intermediate_dim,
            "logical_payload_bytes": (input_values + gate_values * 2 + down_values + input_values) * 2,
        }),
        args,
        timing,
    )
}

fn bench_triton_dense_mlp_bf16_graph(
    library: &NativeLibrary,
    stream: &CudaStream<'_>,
    native_lib: &Path,
    args: &BenchCudaKernelsArgs,
) -> Result<()> {
    let input_values = args
        .rows
        .checked_mul(args.hidden_dim)
        .context("Triton dense MLP input value count overflow")?;
    let gate_values = args
        .intermediate_dim
        .checked_mul(args.hidden_dim)
        .context("Triton dense MLP gate value count overflow")?;
    let down_values = args
        .hidden_dim
        .checked_mul(args.intermediate_dim)
        .context("Triton dense MLP down value count overflow")?;
    let intermediate_values = args
        .rows
        .checked_mul(args.intermediate_dim)
        .context("Triton dense MLP intermediate value count overflow")?;
    let input = bf16_pattern(input_values);
    let gate = bf16_pattern(gate_values);
    let up = bf16_pattern(gate_values);
    let down = bf16_pattern(down_values);
    let input_buffer = upload(library, u16_bytes(&input), "Triton dense MLP input")?;
    let gate_buffer = upload(library, u16_bytes(&gate), "Triton dense MLP gate weight")?;
    let up_buffer = upload(library, u16_bytes(&up), "Triton dense MLP up weight")?;
    let down_buffer = upload(library, u16_bytes(&down), "Triton dense MLP down weight")?;
    let gate_output_buffer = DeviceAllocation::new(
        library,
        intermediate_values * 4,
        "Triton dense MLP gate output",
    )?;
    let up_output_buffer = DeviceAllocation::new(
        library,
        intermediate_values * 4,
        "Triton dense MLP up output",
    )?;
    let activation_buffer = DeviceAllocation::new(
        library,
        intermediate_values * 2,
        "Triton dense MLP activation",
    )?;
    let out_buffer = DeviceAllocation::new(library, input_values * 2, "Triton dense MLP output")?;
    let buffers = [
        PythonDeviceBufferArg {
            name: "input",
            ptr: input_buffer.buffer().ptr,
            bytes: input_buffer.buffer().bytes,
            device_id: input_buffer.buffer().device_id,
            flags: input_buffer.buffer().flags,
        },
        PythonDeviceBufferArg {
            name: "gate_weight",
            ptr: gate_buffer.buffer().ptr,
            bytes: gate_buffer.buffer().bytes,
            device_id: gate_buffer.buffer().device_id,
            flags: gate_buffer.buffer().flags,
        },
        PythonDeviceBufferArg {
            name: "up_weight",
            ptr: up_buffer.buffer().ptr,
            bytes: up_buffer.buffer().bytes,
            device_id: up_buffer.buffer().device_id,
            flags: up_buffer.buffer().flags,
        },
        PythonDeviceBufferArg {
            name: "down_weight",
            ptr: down_buffer.buffer().ptr,
            bytes: down_buffer.buffer().bytes,
            device_id: down_buffer.buffer().device_id,
            flags: down_buffer.buffer().flags,
        },
        PythonDeviceBufferArg {
            name: "gate_output",
            ptr: gate_output_buffer.buffer().ptr,
            bytes: gate_output_buffer.buffer().bytes,
            device_id: gate_output_buffer.buffer().device_id,
            flags: gate_output_buffer.buffer().flags,
        },
        PythonDeviceBufferArg {
            name: "up_output",
            ptr: up_output_buffer.buffer().ptr,
            bytes: up_output_buffer.buffer().bytes,
            device_id: up_output_buffer.buffer().device_id,
            flags: up_output_buffer.buffer().flags,
        },
        PythonDeviceBufferArg {
            name: "activation",
            ptr: activation_buffer.buffer().ptr,
            bytes: activation_buffer.buffer().bytes,
            device_id: activation_buffer.buffer().device_id,
            flags: activation_buffer.buffer().flags,
        },
        PythonDeviceBufferArg {
            name: "output",
            ptr: out_buffer.buffer().ptr,
            bytes: out_buffer.buffer().bytes,
            device_id: out_buffer.buffer().device_id,
            flags: out_buffer.buffer().flags,
        },
    ];
    let kwargs = [
        ("rows", PythonKernelArg::Usize(args.rows)),
        ("hidden", PythonKernelArg::Usize(args.hidden_dim)),
        (
            "intermediate",
            PythonKernelArg::Usize(args.intermediate_dim),
        ),
        ("down_stride", PythonKernelArg::Usize(args.intermediate_dim)),
    ];
    launch_python_graph_capture(PythonGraphCaptureLaunch {
        module: "triton_mlp_capture",
        function: "capture_dense_mlp",
        cuda_stream: stream.raw(),
        buffers: &buffers,
        kwargs: &kwargs,
    })
    .context("warming Triton dense MLP graph benchmark")?;
    unsafe {
        stream
            .synchronize()
            .context("synchronizing Triton dense MLP graph benchmark warmup")?;
        library
            .cuda_graph_begin_capture(stream.raw())
            .context("beginning Triton dense MLP CUDA graph capture")?;
        launch_python_graph_capture(PythonGraphCaptureLaunch {
            module: "triton_mlp_capture",
            function: "capture_dense_mlp",
            cuda_stream: stream.raw(),
            buffers: &buffers,
            kwargs: &kwargs,
        })
        .context("capturing Triton dense MLP graph benchmark")?;
        let graph = BenchCudaGraph::new(
            library,
            library
                .cuda_graph_end_capture_retained(stream.raw())
                .context("ending Triton dense MLP CUDA graph capture")?,
        )?;
        let timing = time_kernel(args, || {
            library
                .cuda_graph_launch(graph.graph_exec(), stream.raw())
                .context("launching Triton dense MLP CUDA graph")?;
            stream.synchronize()
        })?;
        emit_timing(
            native_lib,
            "triton_silu_gated_mlp_rows_bf16_graph",
            &json!({
                "rows": args.rows,
                "hidden_dim": args.hidden_dim,
                "intermediate_dim": args.intermediate_dim,
                "logical_payload_bytes": (input_values + gate_values * 2 + down_values + input_values) * 2
                    + intermediate_values * 10,
                "captured_nodes": graph.node_count(),
                "captured_kernel_nodes": graph.kernel_node_count(),
            }),
            args,
            timing,
        )
    }
}

fn bench_router_topk_bf16(
    library: &NativeLibrary,
    stream: &CudaStream<'_>,
    native_lib: &Path,
    args: &BenchCudaKernelsArgs,
) -> Result<()> {
    let experts = args.routes;
    let top_k = router_top_k(args, experts, "router top-k benchmark")?;
    let hidden_values = args
        .rows
        .checked_mul(args.hidden_dim)
        .context("router top-k hidden value count overflow")?;
    let weight_values = experts
        .checked_mul(args.hidden_dim)
        .context("router top-k weight value count overflow")?;
    let output_values = args
        .rows
        .checked_mul(top_k)
        .context("router top-k output value count overflow")?;
    let hidden = bf16_pattern(hidden_values);
    let router_weight = bf16_pattern(weight_values);
    let correction_bias = (0..experts)
        .map(|idx| ((idx % 7) as f32 - 3.0) * 0.03125)
        .collect::<Vec<_>>();
    let hidden_buffer = upload(library, u16_bytes(&hidden), "router top-k hidden")?;
    let weight_buffer = upload(library, u16_bytes(&router_weight), "router top-k weight")?;
    let bias_buffer = upload(
        library,
        f32_bytes(&correction_bias),
        "router top-k correction bias",
    )?;
    let index_buffer = DeviceAllocation::new(library, output_values * 4, "router top-k indices")?;
    let score_buffer = DeviceAllocation::new(library, output_values * 4, "router top-k scores")?;
    let weight_out_buffer =
        DeviceAllocation::new(library, output_values * 4, "router top-k weights")?;

    let timing = time_kernel(args, || unsafe {
        library.cuda_router_topk_bf16_async(
            hidden_buffer.buffer(),
            weight_buffer.buffer(),
            bias_buffer.buffer(),
            index_buffer.buffer(),
            score_buffer.buffer(),
            weight_out_buffer.buffer(),
            args.rows,
            args.hidden_dim,
            experts,
            top_k,
            stream.raw(),
        )?;
        stream.synchronize()
    })?;
    emit_timing(
        native_lib,
        "router_topk_bf16",
        &json!({
            "rows": args.rows,
            "hidden_dim": args.hidden_dim,
            "experts": experts,
            "top_k": top_k,
            "logical_payload_bytes": hidden_values * 2
                + weight_values * 2
                + experts * 4
                + output_values * 12,
        }),
        args,
        timing,
    )
}

fn bench_router_topk_bf16_cub(
    library: &NativeLibrary,
    stream: &CudaStream<'_>,
    native_lib: &Path,
    args: &BenchCudaKernelsArgs,
) -> Result<()> {
    const CUB_TEMP_STORAGE_BYTES: usize = 8 * 1024 * 1024;
    let experts = args.routes;
    let top_k = router_top_k(args, experts, "CUB router top-k benchmark")?;
    let hidden_values = args
        .rows
        .checked_mul(args.hidden_dim)
        .context("CUB router top-k hidden value count overflow")?;
    let weight_values = experts
        .checked_mul(args.hidden_dim)
        .context("CUB router top-k weight value count overflow")?;
    let score_values = args
        .rows
        .checked_mul(experts)
        .context("CUB router score value count overflow")?;
    let output_values = args
        .rows
        .checked_mul(top_k)
        .context("CUB router top-k output value count overflow")?;
    let hidden = bf16_pattern(hidden_values);
    let router_weight = bf16_pattern(weight_values);
    let correction_bias = (0..experts)
        .map(|idx| ((idx % 7) as f32 - 3.0) * 0.03125)
        .collect::<Vec<_>>();
    let hidden_buffer = upload(library, u16_bytes(&hidden), "CUB router top-k hidden")?;
    let weight_buffer = upload(
        library,
        u16_bytes(&router_weight),
        "CUB router top-k weight",
    )?;
    let bias_buffer = upload(
        library,
        f32_bytes(&correction_bias),
        "CUB router top-k correction bias",
    )?;
    let corrected_score_buffer =
        DeviceAllocation::new(library, score_values * 4, "CUB router corrected scores")?;
    let sorted_score_buffer =
        DeviceAllocation::new(library, score_values * 4, "CUB router sorted scores")?;
    let unsorted_index_buffer =
        DeviceAllocation::new(library, score_values * 4, "CUB router unsorted indices")?;
    let sorted_index_buffer =
        DeviceAllocation::new(library, score_values * 4, "CUB router sorted indices")?;
    let segment_offset_buffer =
        DeviceAllocation::new(library, (args.rows + 1) * 4, "CUB router segment offsets")?;
    let temp_storage_buffer =
        DeviceAllocation::new(library, CUB_TEMP_STORAGE_BYTES, "CUB router temp storage")?;
    let index_buffer =
        DeviceAllocation::new(library, output_values * 4, "CUB router top-k indices")?;
    let score_buffer =
        DeviceAllocation::new(library, output_values * 4, "CUB router top-k scores")?;
    let weight_out_buffer =
        DeviceAllocation::new(library, output_values * 4, "CUB router top-k weights")?;

    let timing = time_kernel(args, || unsafe {
        library.cuda_router_topk_bf16_cub_async(
            hidden_buffer.buffer(),
            weight_buffer.buffer(),
            bias_buffer.buffer(),
            corrected_score_buffer.buffer(),
            sorted_score_buffer.buffer(),
            unsorted_index_buffer.buffer(),
            sorted_index_buffer.buffer(),
            segment_offset_buffer.buffer(),
            index_buffer.buffer(),
            score_buffer.buffer(),
            weight_out_buffer.buffer(),
            temp_storage_buffer.buffer(),
            CUB_TEMP_STORAGE_BYTES,
            args.rows,
            args.hidden_dim,
            experts,
            top_k,
            stream.raw(),
        )?;
        stream.synchronize()
    })?;
    emit_timing(
        native_lib,
        "router_topk_bf16_cub",
        &json!({
            "rows": args.rows,
            "hidden_dim": args.hidden_dim,
            "experts": experts,
            "top_k": top_k,
            "cub_temp_storage_bytes": CUB_TEMP_STORAGE_BYTES,
            "logical_payload_bytes": hidden_values * 2
                + weight_values * 2
                + experts * 4
                + score_values * 16
                + (args.rows + 1) * 4
                + output_values * 12,
        }),
        args,
        timing,
    )
}

fn bench_triton_router_topk_bf16_graph(
    library: &NativeLibrary,
    stream: &CudaStream<'_>,
    native_lib: &Path,
    args: &BenchCudaKernelsArgs,
) -> Result<()> {
    let experts = args.routes;
    let top_k = router_top_k(args, experts, "Triton router top-k graph benchmark")?;
    let hidden_values = args
        .rows
        .checked_mul(args.hidden_dim)
        .context("Triton router top-k hidden value count overflow")?;
    let weight_values = experts
        .checked_mul(args.hidden_dim)
        .context("Triton router top-k weight value count overflow")?;
    let output_values = args
        .rows
        .checked_mul(top_k)
        .context("Triton router top-k output value count overflow")?;
    let score_scratch_values = args
        .rows
        .checked_mul(experts)
        .context("Triton router score scratch value count overflow")?;
    let hidden = bf16_pattern(hidden_values);
    let router_weight = bf16_pattern(weight_values);
    let correction_bias = (0..experts)
        .map(|idx| ((idx % 7) as f32 - 3.0) * 0.03125)
        .collect::<Vec<_>>();
    let hidden_buffer = upload(library, u16_bytes(&hidden), "Triton router top-k hidden")?;
    let weight_buffer = upload(
        library,
        u16_bytes(&router_weight),
        "Triton router top-k weight",
    )?;
    let bias_buffer = upload(
        library,
        f32_bytes(&correction_bias),
        "Triton router top-k correction bias",
    )?;
    let score_scratch_buffer = DeviceAllocation::new(
        library,
        score_scratch_values * 4,
        "Triton router score scratch",
    )?;
    let index_buffer =
        DeviceAllocation::new(library, output_values * 4, "Triton router top-k indices")?;
    let score_buffer =
        DeviceAllocation::new(library, output_values * 4, "Triton router top-k scores")?;
    let weight_out_buffer =
        DeviceAllocation::new(library, output_values * 4, "Triton router top-k weights")?;
    let buffers = [
        PythonDeviceBufferArg {
            name: "hidden",
            ptr: hidden_buffer.buffer().ptr,
            bytes: hidden_buffer.buffer().bytes,
            device_id: hidden_buffer.buffer().device_id,
            flags: hidden_buffer.buffer().flags,
        },
        PythonDeviceBufferArg {
            name: "router_weight",
            ptr: weight_buffer.buffer().ptr,
            bytes: weight_buffer.buffer().bytes,
            device_id: weight_buffer.buffer().device_id,
            flags: weight_buffer.buffer().flags,
        },
        PythonDeviceBufferArg {
            name: "correction_bias",
            ptr: bias_buffer.buffer().ptr,
            bytes: bias_buffer.buffer().bytes,
            device_id: bias_buffer.buffer().device_id,
            flags: bias_buffer.buffer().flags,
        },
        PythonDeviceBufferArg {
            name: "score_scratch",
            ptr: score_scratch_buffer.buffer().ptr,
            bytes: score_scratch_buffer.buffer().bytes,
            device_id: score_scratch_buffer.buffer().device_id,
            flags: score_scratch_buffer.buffer().flags,
        },
        PythonDeviceBufferArg {
            name: "topk_indices",
            ptr: index_buffer.buffer().ptr,
            bytes: index_buffer.buffer().bytes,
            device_id: index_buffer.buffer().device_id,
            flags: index_buffer.buffer().flags,
        },
        PythonDeviceBufferArg {
            name: "topk_scores",
            ptr: score_buffer.buffer().ptr,
            bytes: score_buffer.buffer().bytes,
            device_id: score_buffer.buffer().device_id,
            flags: score_buffer.buffer().flags,
        },
        PythonDeviceBufferArg {
            name: "topk_weights",
            ptr: weight_out_buffer.buffer().ptr,
            bytes: weight_out_buffer.buffer().bytes,
            device_id: weight_out_buffer.buffer().device_id,
            flags: weight_out_buffer.buffer().flags,
        },
    ];
    let kwargs = [
        ("rows", PythonKernelArg::Usize(args.rows)),
        ("hidden_dim", PythonKernelArg::Usize(args.hidden_dim)),
        ("experts", PythonKernelArg::Usize(experts)),
        ("top_k", PythonKernelArg::Usize(top_k)),
        (
            "routed_scaling_factor",
            PythonKernelArg::F64(GLM52_ROUTED_SCALING_FACTOR as f64),
        ),
    ];
    launch_python_graph_capture(PythonGraphCaptureLaunch {
        module: "triton_router_capture",
        function: "capture_router_topk",
        cuda_stream: stream.raw(),
        buffers: &buffers,
        kwargs: &kwargs,
    })
    .context("warming Triton router top-k graph benchmark")?;
    unsafe {
        stream
            .synchronize()
            .context("synchronizing Triton router top-k graph benchmark warmup")?;
        library
            .cuda_graph_begin_capture(stream.raw())
            .context("beginning Triton router top-k CUDA graph capture")?;
        launch_python_graph_capture(PythonGraphCaptureLaunch {
            module: "triton_router_capture",
            function: "capture_router_topk",
            cuda_stream: stream.raw(),
            buffers: &buffers,
            kwargs: &kwargs,
        })
        .context("capturing Triton router top-k graph benchmark")?;
        let graph = BenchCudaGraph::new(
            library,
            library
                .cuda_graph_end_capture_retained(stream.raw())
                .context("ending Triton router top-k CUDA graph capture")?,
        )?;
        let timing = time_kernel(args, || {
            library
                .cuda_graph_launch(graph.graph_exec(), stream.raw())
                .context("launching Triton router top-k CUDA graph")?;
            stream.synchronize()
        })?;
        emit_timing(
            native_lib,
            "triton_router_topk_bf16_graph",
            &json!({
                "rows": args.rows,
                "hidden_dim": args.hidden_dim,
                "experts": experts,
                "top_k": top_k,
                "logical_payload_bytes": hidden_values * 2
                    + weight_values * 2
                    + experts * 4
                    + score_scratch_values * 4
                    + output_values * 12,
                "captured_nodes": graph.node_count(),
                "captured_kernel_nodes": graph.kernel_node_count(),
            }),
            args,
            timing,
        )
    }
}

fn bench_lm_head_sample_topk_topp_bf16(
    library: &NativeLibrary,
    stream: &CudaStream<'_>,
    native_lib: &Path,
    args: &BenchCudaKernelsArgs,
) -> Result<()> {
    let top_k = sample_top_k(args, "LM-head sampler benchmark")?;
    let hidden_values = args
        .rows
        .checked_mul(args.hidden_dim)
        .context("LM-head sampler hidden value count overflow")?;
    let weight_values = args
        .vocab
        .checked_mul(args.hidden_dim)
        .context("LM-head sampler weight value count overflow")?;
    let hidden = bf16_pattern(hidden_values);
    let lm_head = bf16_pattern(weight_values);
    let random_uniforms = (0..args.rows)
        .map(|idx| ((idx % 997) as f32 + 0.5) / 997.0)
        .collect::<Vec<_>>();
    let hidden_buffer = upload(library, u16_bytes(&hidden), "LM-head sampler hidden")?;
    let lm_head_buffer = upload(library, u16_bytes(&lm_head), "LM-head sampler weight")?;
    let random_buffer = upload(
        library,
        f32_bytes(&random_uniforms),
        "LM-head sampler random uniforms",
    )?;
    let index_buffer = DeviceAllocation::new(library, args.rows * 4, "LM-head sampler indices")?;
    let score_buffer = DeviceAllocation::new(library, args.rows * 4, "LM-head sampler scores")?;
    let timing = time_kernel(args, || unsafe {
        library.cuda_lm_head_sample_topk_topp_bf16_async(
            hidden_buffer.buffer(),
            lm_head_buffer.buffer(),
            random_buffer.buffer(),
            index_buffer.buffer(),
            score_buffer.buffer(),
            args.rows,
            args.hidden_dim,
            args.vocab,
            0.7,
            top_k,
            0.95,
            stream.raw(),
        )?;
        stream.synchronize()
    })?;
    emit_timing(
        native_lib,
        "lm_head_sample_topk_topp_bf16",
        &json!({
            "rows": args.rows,
            "hidden_dim": args.hidden_dim,
            "vocab": args.vocab,
            "top_k": top_k,
            "temperature": 0.7,
            "top_p": 0.95,
            "logical_payload_bytes": hidden_values * 2
                + weight_values * 2
                + args.rows * 12,
        }),
        args,
        timing,
    )
}

fn bench_lm_head_sample_topk_topp_bf16_cub(
    library: &NativeLibrary,
    stream: &CudaStream<'_>,
    native_lib: &Path,
    args: &BenchCudaKernelsArgs,
) -> Result<()> {
    const CUB_TEMP_STORAGE_BYTES: usize = 128 * 1024 * 1024;
    let top_k = sample_top_k(args, "CUB LM-head sampler benchmark")?;
    let hidden_values = args
        .rows
        .checked_mul(args.hidden_dim)
        .context("CUB LM-head sampler hidden value count overflow")?;
    let weight_values = args
        .vocab
        .checked_mul(args.hidden_dim)
        .context("CUB LM-head sampler weight value count overflow")?;
    let logits_values = args
        .rows
        .checked_mul(args.vocab)
        .context("CUB LM-head sampler logits value count overflow")?;
    let hidden = bf16_pattern(hidden_values);
    let lm_head = bf16_pattern(weight_values);
    let random_uniforms = (0..args.rows)
        .map(|idx| ((idx % 997) as f32 + 0.5) / 997.0)
        .collect::<Vec<_>>();
    let hidden_buffer = upload(library, u16_bytes(&hidden), "CUB LM-head sampler hidden")?;
    let lm_head_buffer = upload(library, u16_bytes(&lm_head), "CUB LM-head sampler weight")?;
    let random_buffer = upload(
        library,
        f32_bytes(&random_uniforms),
        "CUB LM-head sampler random uniforms",
    )?;
    let logits_buffer =
        DeviceAllocation::new(library, logits_values * 4, "CUB LM-head logits workspace")?;
    let sorted_logits_buffer =
        DeviceAllocation::new(library, logits_values * 4, "CUB LM-head sorted logits")?;
    let unsorted_index_buffer = DeviceAllocation::new(
        library,
        logits_values * 4,
        "CUB LM-head unsorted logits indices",
    )?;
    let sorted_index_buffer = DeviceAllocation::new(
        library,
        logits_values * 4,
        "CUB LM-head sorted logits indices",
    )?;
    let segment_offset_buffer =
        DeviceAllocation::new(library, (args.rows + 1) * 4, "CUB LM-head segment offsets")?;
    let index_buffer =
        DeviceAllocation::new(library, args.rows * 4, "CUB LM-head sampler indices")?;
    let score_buffer = DeviceAllocation::new(library, args.rows * 4, "CUB LM-head sampler scores")?;
    let temp_storage_buffer =
        DeviceAllocation::new(library, CUB_TEMP_STORAGE_BYTES, "CUB LM-head temp storage")?;
    let timing = time_kernel(args, || unsafe {
        library.cuda_lm_head_sample_topk_topp_bf16_cub_async(
            hidden_buffer.buffer(),
            lm_head_buffer.buffer(),
            random_buffer.buffer(),
            logits_buffer.buffer(),
            sorted_logits_buffer.buffer(),
            unsorted_index_buffer.buffer(),
            sorted_index_buffer.buffer(),
            segment_offset_buffer.buffer(),
            index_buffer.buffer(),
            score_buffer.buffer(),
            temp_storage_buffer.buffer(),
            CUB_TEMP_STORAGE_BYTES,
            args.rows,
            args.hidden_dim,
            args.vocab,
            0.7,
            top_k,
            0.95,
            stream.raw(),
        )?;
        stream.synchronize()
    })?;
    emit_timing(
        native_lib,
        "lm_head_sample_topk_topp_bf16_cub",
        &json!({
            "rows": args.rows,
            "hidden_dim": args.hidden_dim,
            "vocab": args.vocab,
            "top_k": top_k,
            "temperature": 0.7,
            "top_p": 0.95,
            "cub_temp_storage_bytes": CUB_TEMP_STORAGE_BYTES,
            "logical_payload_bytes": hidden_values * 2
                + weight_values * 2
                + logits_values * 12
                + (args.rows + 1) * 4
                + CUB_TEMP_STORAGE_BYTES
                + args.rows * 12,
        }),
        args,
        timing,
    )
}

fn bench_logits_sample_topk_topp_f32(
    library: &NativeLibrary,
    stream: &CudaStream<'_>,
    native_lib: &Path,
    args: &BenchCudaKernelsArgs,
) -> Result<()> {
    let top_k = sample_top_k(args, "logits sampler benchmark")?;
    let logits_values = args
        .rows
        .checked_mul(args.vocab)
        .context("logits sampler value count overflow")?;
    let logits = logits_f32_pattern(args.rows, args.vocab);
    let random_uniforms = sample_random_uniforms(args.rows);
    let logits_buffer = upload(library, f32_bytes(&logits), "logits sampler logits")?;
    let random_buffer = upload(
        library,
        f32_bytes(&random_uniforms),
        "logits sampler random uniforms",
    )?;
    let index_buffer = DeviceAllocation::new(library, args.rows * 4, "logits sampler indices")?;
    let score_buffer = DeviceAllocation::new(library, args.rows * 4, "logits sampler scores")?;
    let timing = time_kernel(args, || unsafe {
        library.cuda_logits_sample_topk_topp_f32_async(
            logits_buffer.buffer(),
            random_buffer.buffer(),
            index_buffer.buffer(),
            score_buffer.buffer(),
            args.rows,
            args.vocab,
            0.7,
            top_k,
            0.95,
            stream.raw(),
        )?;
        stream.synchronize()
    })?;
    emit_timing(
        native_lib,
        "logits_sample_topk_topp_f32",
        &json!({
            "rows": args.rows,
            "vocab": args.vocab,
            "top_k": top_k,
            "temperature": 0.7,
            "top_p": 0.95,
            "logical_payload_bytes": logits_values * 4 + args.rows * 12,
        }),
        args,
        timing,
    )
}

fn bench_logits_sample_topk_topp_f32_cub(
    library: &NativeLibrary,
    stream: &CudaStream<'_>,
    native_lib: &Path,
    args: &BenchCudaKernelsArgs,
) -> Result<()> {
    const CUB_TEMP_STORAGE_BYTES: usize = 128 * 1024 * 1024;
    let top_k = sample_top_k(args, "CUB logits sampler benchmark")?;
    let logits_values = args
        .rows
        .checked_mul(args.vocab)
        .context("CUB logits sampler value count overflow")?;
    let logits = logits_f32_pattern(args.rows, args.vocab);
    let random_uniforms = sample_random_uniforms(args.rows);
    let logits_buffer = upload(library, f32_bytes(&logits), "CUB logits sampler logits")?;
    let random_buffer = upload(
        library,
        f32_bytes(&random_uniforms),
        "CUB logits sampler random uniforms",
    )?;
    let sorted_logits_buffer = DeviceAllocation::new(
        library,
        logits_values * 4,
        "CUB logits sampler sorted logits",
    )?;
    let unsorted_index_buffer = DeviceAllocation::new(
        library,
        logits_values * 4,
        "CUB logits sampler unsorted indices",
    )?;
    let sorted_index_buffer = DeviceAllocation::new(
        library,
        logits_values * 4,
        "CUB logits sampler sorted indices",
    )?;
    let segment_offset_buffer =
        DeviceAllocation::new(library, (args.rows + 1) * 4, "CUB logits sampler offsets")?;
    let temp_storage_buffer = DeviceAllocation::new(
        library,
        CUB_TEMP_STORAGE_BYTES,
        "CUB logits sampler temp storage",
    )?;
    let index_buffer = DeviceAllocation::new(library, args.rows * 4, "CUB logits sampler indices")?;
    let score_buffer = DeviceAllocation::new(library, args.rows * 4, "CUB logits sampler scores")?;
    let timing = time_kernel(args, || unsafe {
        library.cuda_logits_sample_topk_topp_f32_cub_async(
            logits_buffer.buffer(),
            random_buffer.buffer(),
            sorted_logits_buffer.buffer(),
            unsorted_index_buffer.buffer(),
            sorted_index_buffer.buffer(),
            segment_offset_buffer.buffer(),
            index_buffer.buffer(),
            score_buffer.buffer(),
            temp_storage_buffer.buffer(),
            CUB_TEMP_STORAGE_BYTES,
            args.rows,
            args.vocab,
            0.7,
            top_k,
            0.95,
            stream.raw(),
        )?;
        stream.synchronize()
    })?;
    emit_timing(
        native_lib,
        "logits_sample_topk_topp_f32_cub",
        &json!({
            "rows": args.rows,
            "vocab": args.vocab,
            "top_k": top_k,
            "temperature": 0.7,
            "top_p": 0.95,
            "cub_temp_storage_bytes": CUB_TEMP_STORAGE_BYTES,
            "logical_payload_bytes": logits_values * 16
                + (args.rows + 1) * 4
                + CUB_TEMP_STORAGE_BYTES
                + args.rows * 12,
        }),
        args,
        timing,
    )
}

fn bench_mla_kv_pack_fp8_ds_mla(
    library: &NativeLibrary,
    _stream: &CudaStream<'_>,
    native_lib: &Path,
    args: &BenchCudaKernelsArgs,
) -> Result<()> {
    let projected_values = args
        .rows
        .checked_mul(MLA_KV_FP8_PROJECTED_VALUES)
        .context("MLA FP8 KV pack projected value count overflow")?;
    let packed_bytes = args
        .rows
        .checked_mul(MLA_KV_FP8_PACKED_BYTES)
        .context("MLA FP8 KV pack packed byte count overflow")?;
    let projected = bf16_pattern(projected_values);
    let projected_buffer = upload(
        library,
        u16_bytes(&projected),
        "MLA FP8 DS KV pack projected input",
    )?;
    let packed_buffer =
        DeviceAllocation::new(library, packed_bytes, "MLA FP8 DS KV packed output")?;
    let timing = time_kernel(args, || {
        library.cuda_mla_kv_pack_fp8_ds_mla(
            projected_buffer.buffer(),
            packed_buffer.buffer(),
            args.rows,
            MLA_KV_FP8_PROJECTED_STRIDE_BYTES,
            MLA_KV_FP8_PACKED_BYTES,
        )
    })?;
    emit_timing(
        native_lib,
        "mla_kv_pack_fp8_ds_mla",
        &json!({
            "rows": args.rows,
            "projected_values_per_row": MLA_KV_FP8_PROJECTED_VALUES,
            "projected_stride_bytes": MLA_KV_FP8_PROJECTED_STRIDE_BYTES,
            "packed_stride_bytes": MLA_KV_FP8_PACKED_BYTES,
            "logical_payload_bytes": projected_values * 2 + packed_bytes,
        }),
        args,
        timing,
    )
}

fn bench_mla_kv_pack_mxfp4_ds_mla(
    library: &NativeLibrary,
    _stream: &CudaStream<'_>,
    native_lib: &Path,
    args: &BenchCudaKernelsArgs,
) -> Result<()> {
    let projected_values = args
        .rows
        .checked_mul(MLA_KV_MXFP4_PROJECTED_VALUES)
        .context("MLA MXFP4 KV pack projected value count overflow")?;
    let packed_bytes = args
        .rows
        .checked_mul(MLA_KV_MXFP4_PACKED_BYTES)
        .context("MLA MXFP4 KV pack packed byte count overflow")?;
    let projected = bf16_pattern(projected_values);
    let projected_buffer = upload(
        library,
        u16_bytes(&projected),
        "MLA MXFP4 DS KV pack projected input",
    )?;
    let packed_buffer =
        DeviceAllocation::new(library, packed_bytes, "MLA MXFP4 DS KV packed output")?;
    let timing = time_kernel(args, || {
        library.cuda_mla_kv_pack_mxfp4_ds_mla(
            projected_buffer.buffer(),
            packed_buffer.buffer(),
            args.rows,
            MLA_KV_MXFP4_PROJECTED_STRIDE_BYTES,
            MLA_KV_MXFP4_PACKED_BYTES,
        )
    })?;
    emit_timing(
        native_lib,
        "mla_kv_pack_mxfp4_ds_mla",
        &json!({
            "rows": args.rows,
            "projected_values_per_row": MLA_KV_MXFP4_PROJECTED_VALUES,
            "projected_stride_bytes": MLA_KV_MXFP4_PROJECTED_STRIDE_BYTES,
            "packed_stride_bytes": MLA_KV_MXFP4_PACKED_BYTES,
            "fp8_packed_stride_bytes": MLA_KV_FP8_PACKED_BYTES,
            "packed_stride_savings_bytes": MLA_KV_FP8_PACKED_BYTES - MLA_KV_MXFP4_PACKED_BYTES,
            "logical_payload_bytes": projected_values * 2 + packed_bytes,
        }),
        args,
        timing,
    )
}

fn bench_mla_rope_attention_bf16(
    library: &NativeLibrary,
    stream: &CudaStream<'_>,
    native_lib: &Path,
    args: &BenchCudaKernelsArgs,
) -> Result<()> {
    let q_nope_values = args
        .rows
        .checked_mul(GLM52_MLA_ATTENTION_HEADS)
        .and_then(|values| values.checked_mul(GLM52_MLA_QK_NOPE_HEAD_DIM))
        .context("MLA/RoPE q_nope value count overflow")?;
    let q_rope_values = args
        .rows
        .checked_mul(GLM52_MLA_ATTENTION_HEADS)
        .and_then(|values| values.checked_mul(GLM52_MLA_QK_ROPE_HEAD_DIM))
        .context("MLA/RoPE q_rope value count overflow")?;
    let k_rope_values = args
        .rows
        .checked_mul(GLM52_MLA_QK_ROPE_HEAD_DIM)
        .context("MLA/RoPE k_rope value count overflow")?;
    let v_values = args
        .rows
        .checked_mul(GLM52_MLA_ATTENTION_HEADS)
        .and_then(|values| values.checked_mul(GLM52_MLA_V_HEAD_DIM))
        .context("MLA/RoPE value count overflow")?;

    let q_nope = bf16_pattern(q_nope_values);
    let q_rope = bf16_pattern(q_rope_values);
    let k_nope = bf16_replay_pattern(q_nope_values);
    let k_rope = bf16_replay_pattern(k_rope_values);
    let values = bf16_pattern(v_values);
    let q_nope_buffer = upload(library, u16_bytes(&q_nope), "MLA/RoPE q_nope")?;
    let q_rope_buffer = upload(library, u16_bytes(&q_rope), "MLA/RoPE q_rope")?;
    let k_nope_buffer = upload(library, u16_bytes(&k_nope), "MLA/RoPE k_nope")?;
    let k_rope_buffer = upload(library, u16_bytes(&k_rope), "MLA/RoPE k_rope")?;
    let value_buffer = upload(library, u16_bytes(&values), "MLA/RoPE values")?;
    let out_buffer = DeviceAllocation::new(library, v_values * 2, "MLA/RoPE output")?;
    let scale = 1.0_f32 / ((GLM52_MLA_QK_NOPE_HEAD_DIM + GLM52_MLA_QK_ROPE_HEAD_DIM) as f32).sqrt();

    let timing = time_kernel(args, || unsafe {
        library.cuda_mla_rope_attention_bf16_async(
            q_nope_buffer.buffer(),
            q_rope_buffer.buffer(),
            k_nope_buffer.buffer(),
            k_rope_buffer.buffer(),
            value_buffer.buffer(),
            out_buffer.buffer(),
            args.rows,
            GLM52_MLA_ATTENTION_HEADS,
            GLM52_MLA_QK_NOPE_HEAD_DIM,
            GLM52_MLA_QK_ROPE_HEAD_DIM,
            GLM52_MLA_V_HEAD_DIM,
            scale,
            stream.raw(),
        )?;
        stream.synchronize()
    })?;
    emit_timing(
        native_lib,
        "mla_rope_attention_bf16",
        &json!({
            "rows": args.rows,
            "heads": GLM52_MLA_ATTENTION_HEADS,
            "qk_nope_head_dim": GLM52_MLA_QK_NOPE_HEAD_DIM,
            "qk_rope_head_dim": GLM52_MLA_QK_ROPE_HEAD_DIM,
            "v_head_dim": GLM52_MLA_V_HEAD_DIM,
            "scale": scale,
            "logical_payload_bytes": (q_nope_values * 2)
                + (q_rope_values * 2)
                + (q_nope_values * 2)
                + (k_rope_values * 2)
                + (v_values * 2)
                + (v_values * 2),
        }),
        args,
        timing,
    )
}

fn bench_mla_rope_attention_bf16_suffix(
    library: &NativeLibrary,
    stream: &CudaStream<'_>,
    native_lib: &Path,
    args: &BenchCudaKernelsArgs,
) -> Result<()> {
    let q_nope_values = args
        .rows
        .checked_mul(GLM52_MLA_ATTENTION_HEADS)
        .and_then(|values| values.checked_mul(GLM52_MLA_QK_NOPE_HEAD_DIM))
        .context("MLA/RoPE suffix q_nope value count overflow")?;
    let q_rope_values = args
        .rows
        .checked_mul(GLM52_MLA_ATTENTION_HEADS)
        .and_then(|values| values.checked_mul(GLM52_MLA_QK_ROPE_HEAD_DIM))
        .context("MLA/RoPE suffix q_rope value count overflow")?;
    let k_rope_values = args
        .rows
        .checked_mul(GLM52_MLA_QK_ROPE_HEAD_DIM)
        .context("MLA/RoPE suffix k_rope value count overflow")?;
    let v_values = args
        .rows
        .checked_mul(GLM52_MLA_ATTENTION_HEADS)
        .and_then(|values| values.checked_mul(GLM52_MLA_V_HEAD_DIM))
        .context("MLA/RoPE suffix value count overflow")?;

    let query_rows = 1_usize;
    let query_row_offset = args
        .rows
        .checked_sub(query_rows)
        .context("MLA/RoPE suffix benchmark requires rows >= query rows")?;
    let output_values = query_rows
        .checked_mul(GLM52_MLA_ATTENTION_HEADS)
        .and_then(|values| values.checked_mul(GLM52_MLA_V_HEAD_DIM))
        .context("MLA/RoPE suffix output value count overflow")?;
    let q_nope = bf16_pattern(q_nope_values);
    let q_rope = bf16_pattern(q_rope_values);
    let k_nope = bf16_replay_pattern(q_nope_values);
    let k_rope = bf16_replay_pattern(k_rope_values);
    let values = bf16_pattern(v_values);
    let q_nope_buffer = upload(library, u16_bytes(&q_nope), "MLA/RoPE suffix q_nope")?;
    let q_rope_buffer = upload(library, u16_bytes(&q_rope), "MLA/RoPE suffix q_rope")?;
    let k_nope_buffer = upload(library, u16_bytes(&k_nope), "MLA/RoPE suffix k_nope")?;
    let k_rope_buffer = upload(library, u16_bytes(&k_rope), "MLA/RoPE suffix k_rope")?;
    let value_buffer = upload(library, u16_bytes(&values), "MLA/RoPE suffix values")?;
    let out_buffer = DeviceAllocation::new(library, output_values * 2, "MLA/RoPE suffix output")?;
    let scale = 1.0_f32 / ((GLM52_MLA_QK_NOPE_HEAD_DIM + GLM52_MLA_QK_ROPE_HEAD_DIM) as f32).sqrt();

    let timing = time_kernel(args, || unsafe {
        library.cuda_mla_rope_attention_bf16_suffix_async(
            q_nope_buffer.buffer(),
            q_rope_buffer.buffer(),
            k_nope_buffer.buffer(),
            k_rope_buffer.buffer(),
            value_buffer.buffer(),
            out_buffer.buffer(),
            args.rows,
            query_row_offset,
            query_rows,
            GLM52_MLA_ATTENTION_HEADS,
            GLM52_MLA_QK_NOPE_HEAD_DIM,
            GLM52_MLA_QK_ROPE_HEAD_DIM,
            GLM52_MLA_V_HEAD_DIM,
            scale,
            stream.raw(),
        )?;
        stream.synchronize()
    })?;
    emit_timing(
        native_lib,
        "mla_rope_attention_bf16_suffix",
        &json!({
            "rows": args.rows,
            "query_row_offset": query_row_offset,
            "query_rows": query_rows,
            "heads": GLM52_MLA_ATTENTION_HEADS,
            "qk_nope_head_dim": GLM52_MLA_QK_NOPE_HEAD_DIM,
            "qk_rope_head_dim": GLM52_MLA_QK_ROPE_HEAD_DIM,
            "v_head_dim": GLM52_MLA_V_HEAD_DIM,
            "scale": scale,
            "logical_payload_bytes": (q_nope_values * 2)
                + (q_rope_values * 2)
                + (q_nope_values * 2)
                + (k_rope_values * 2)
                + (v_values * 2)
                + (output_values * 2),
        }),
        args,
        timing,
    )
}

fn bench_b12x_mla_rope_attention_bf16_graph(
    library: &NativeLibrary,
    stream: &CudaStream<'_>,
    native_lib: &Path,
    args: &BenchCudaKernelsArgs,
) -> Result<()> {
    if args.rows > 512 {
        anyhow::bail!(
            "b12x MLA/RoPE attention graph requires rows <= 512, got {}",
            args.rows
        );
    }
    let q_nope_values = args
        .rows
        .checked_mul(B12X_MLA_ATTENTION_HEADS)
        .and_then(|values| values.checked_mul(B12X_MLA_KV_LORA_RANK))
        .context("b12x MLA/RoPE q_nope value count overflow")?;
    let q_rope_values = args
        .rows
        .checked_mul(B12X_MLA_ATTENTION_HEADS)
        .and_then(|values| values.checked_mul(B12X_MLA_QK_ROPE_HEAD_DIM))
        .context("b12x MLA/RoPE q_rope value count overflow")?;
    let k_rope_values = args
        .rows
        .checked_mul(B12X_MLA_QK_ROPE_HEAD_DIM)
        .context("b12x MLA/RoPE k_rope value count overflow")?;
    let value_values = args
        .rows
        .checked_mul(B12X_MLA_ATTENTION_HEADS)
        .and_then(|values| values.checked_mul(B12X_MLA_KV_LORA_RANK))
        .context("b12x MLA/RoPE value count overflow")?;

    let q_nope = bf16_pattern(q_nope_values);
    let q_rope = bf16_replay_pattern(q_rope_values);
    let k_nope = bf16_replay_pattern(q_nope_values);
    let k_rope = bf16_pattern(k_rope_values);
    let values = bf16_pattern(value_values);
    let q_nope_buffer = upload(library, u16_bytes(&q_nope), "b12x MLA/RoPE q_nope")?;
    let q_rope_buffer = upload(library, u16_bytes(&q_rope), "b12x MLA/RoPE q_rope")?;
    let k_nope_buffer = upload(library, u16_bytes(&k_nope), "b12x MLA/RoPE k_nope")?;
    let k_rope_buffer = upload(library, u16_bytes(&k_rope), "b12x MLA/RoPE k_rope")?;
    let value_buffer = upload(library, u16_bytes(&values), "b12x MLA/RoPE values")?;
    let out_buffer = DeviceAllocation::new(library, value_values * 2, "b12x MLA/RoPE output")?;
    let scale = 1.0_f32 / ((B12X_MLA_KV_LORA_RANK + B12X_MLA_QK_ROPE_HEAD_DIM) as f32).sqrt();
    let buffers = [
        python_buffer("q_nope", q_nope_buffer.buffer()),
        python_buffer("q_rope", q_rope_buffer.buffer()),
        python_buffer("k_nope", k_nope_buffer.buffer()),
        python_buffer("k_rope", k_rope_buffer.buffer()),
        python_buffer("values", value_buffer.buffer()),
        python_buffer("output", out_buffer.buffer()),
    ];
    let kwargs = [
        ("rows", PythonKernelArg::Usize(args.rows)),
        ("heads", PythonKernelArg::Usize(B12X_MLA_ATTENTION_HEADS)),
        ("nope_dim", PythonKernelArg::Usize(B12X_MLA_KV_LORA_RANK)),
        (
            "rope_dim",
            PythonKernelArg::Usize(B12X_MLA_QK_ROPE_HEAD_DIM),
        ),
        ("v_dim", PythonKernelArg::Usize(B12X_MLA_KV_LORA_RANK)),
        ("scale", PythonKernelArg::F64(scale as f64)),
    ];
    let graph = capture_python_graph(
        library,
        stream,
        B12X_MLA_CAPTURE_MODULE,
        B12X_MLA_CAPTURE_FUNCTION,
        &buffers,
        &kwargs,
        "b12x MLA/RoPE attention graph",
    )?;
    let timing = time_kernel(args, || unsafe {
        library
            .cuda_graph_launch(graph.graph_exec(), stream.raw())
            .context("launching b12x MLA/RoPE attention graph")?;
        stream.synchronize()
    })?;
    emit_timing(
        native_lib,
        "b12x_mla_rope_attention_bf16_graph",
        &json!({
            "rows": args.rows,
            "heads": B12X_MLA_ATTENTION_HEADS,
            "qk_nope_head_dim": B12X_MLA_KV_LORA_RANK,
            "qk_rope_head_dim": B12X_MLA_QK_ROPE_HEAD_DIM,
            "v_head_dim": B12X_MLA_KV_LORA_RANK,
            "scale": scale,
            "graph_nodes": graph.node_count(),
            "graph_kernel_nodes": graph.kernel_node_count(),
            "capture_module": B12X_MLA_CAPTURE_MODULE,
            "capture_function": B12X_MLA_CAPTURE_FUNCTION,
            "logical_payload_bytes": (q_nope_values * 2)
                + (q_rope_values * 2)
                + (q_nope_values * 2)
                + (k_rope_values * 2)
                + (value_values * 2)
                + (value_values * 2),
        }),
        args,
        timing,
    )
}

fn bench_triton_mla_kv_pack_fp8_ds_mla_graph(
    library: &NativeLibrary,
    stream: &CudaStream<'_>,
    native_lib: &Path,
    args: &BenchCudaKernelsArgs,
) -> Result<()> {
    let projected_values = args
        .rows
        .checked_mul(MLA_KV_FP8_PROJECTED_VALUES)
        .context("Triton MLA FP8 KV pack projected value count overflow")?;
    let packed_bytes = args
        .rows
        .checked_mul(MLA_KV_FP8_PACKED_BYTES)
        .context("Triton MLA FP8 KV pack packed byte count overflow")?;
    let projected = bf16_pattern(projected_values);
    let projected_buffer = upload(
        library,
        u16_bytes(&projected),
        "Triton MLA FP8 DS KV pack projected input",
    )?;
    let packed_buffer =
        DeviceAllocation::new(library, packed_bytes, "Triton MLA FP8 DS KV packed output")?;
    let buffers = [
        PythonDeviceBufferArg {
            name: "projected",
            ptr: projected_buffer.buffer().ptr,
            bytes: projected_buffer.buffer().bytes,
            device_id: projected_buffer.buffer().device_id,
            flags: projected_buffer.buffer().flags,
        },
        PythonDeviceBufferArg {
            name: "packed",
            ptr: packed_buffer.buffer().ptr,
            bytes: packed_buffer.buffer().bytes,
            device_id: packed_buffer.buffer().device_id,
            flags: packed_buffer.buffer().flags,
        },
    ];
    let kwargs = [
        ("rows", PythonKernelArg::Usize(args.rows)),
        (
            "projected_stride_bytes",
            PythonKernelArg::Usize(MLA_KV_FP8_PROJECTED_STRIDE_BYTES),
        ),
        (
            "packed_stride_bytes",
            PythonKernelArg::Usize(MLA_KV_FP8_PACKED_BYTES),
        ),
    ];
    launch_python_graph_capture(PythonGraphCaptureLaunch {
        module: "triton_kv_pack_capture",
        function: "capture_mla_kv_pack_fp8_ds_mla",
        cuda_stream: stream.raw(),
        buffers: &buffers,
        kwargs: &kwargs,
    })
    .context("warming Triton MLA FP8 DS KV pack graph benchmark")?;
    unsafe {
        stream
            .synchronize()
            .context("synchronizing Triton MLA FP8 DS KV pack warmup")?;
        library
            .cuda_graph_begin_capture(stream.raw())
            .context("beginning Triton MLA FP8 DS KV pack graph capture")?;
        launch_python_graph_capture(PythonGraphCaptureLaunch {
            module: "triton_kv_pack_capture",
            function: "capture_mla_kv_pack_fp8_ds_mla",
            cuda_stream: stream.raw(),
            buffers: &buffers,
            kwargs: &kwargs,
        })
        .context("capturing Triton MLA FP8 DS KV pack graph benchmark")?;
        let graph = BenchCudaGraph::new(
            library,
            library
                .cuda_graph_end_capture_retained(stream.raw())
                .context("ending Triton MLA FP8 DS KV pack graph capture")?,
        )?;
        let timing = time_kernel(args, || unsafe {
            library
                .cuda_graph_launch(graph.graph_exec(), stream.raw())
                .context("launching Triton MLA FP8 DS KV pack graph")?;
            stream.synchronize()
        })?;
        emit_timing(
            native_lib,
            "triton_mla_kv_pack_fp8_ds_mla_graph",
            &json!({
                "rows": args.rows,
                "projected_values_per_row": MLA_KV_FP8_PROJECTED_VALUES,
                "projected_stride_bytes": MLA_KV_FP8_PROJECTED_STRIDE_BYTES,
                "packed_stride_bytes": MLA_KV_FP8_PACKED_BYTES,
                "graph_nodes": graph.node_count(),
                "kernel_nodes": graph.kernel_node_count(),
                "logical_payload_bytes": projected_values * 2 + packed_bytes,
            }),
            args,
            timing,
        )?;
    }
    Ok(())
}

fn bench_triton_lm_head_sample_topk_topp_bf16_graph(
    library: &NativeLibrary,
    stream: &CudaStream<'_>,
    native_lib: &Path,
    args: &BenchCudaKernelsArgs,
) -> Result<()> {
    let top_k = sample_top_k(args, "Triton LM-head sampler graph benchmark")?;
    let hidden_values = args
        .rows
        .checked_mul(args.hidden_dim)
        .context("Triton LM-head sampler hidden value count overflow")?;
    let weight_values = args
        .vocab
        .checked_mul(args.hidden_dim)
        .context("Triton LM-head sampler weight value count overflow")?;
    let logits_values = args
        .rows
        .checked_mul(args.vocab)
        .context("Triton LM-head sampler logits value count overflow")?;
    let vocab_blocks = args.vocab.div_ceil(1024);
    let candidate_values = args
        .rows
        .checked_mul(vocab_blocks)
        .and_then(|values| values.checked_mul(top_k))
        .context("Triton LM-head sampler candidate value count overflow")?;
    let hidden = bf16_pattern(hidden_values);
    let lm_head = bf16_pattern(weight_values);
    let random_uniforms = (0..args.rows)
        .map(|idx| ((idx % 997) as f32 + 0.5) / 997.0)
        .collect::<Vec<_>>();
    let hidden_buffer = upload(library, u16_bytes(&hidden), "Triton LM-head sampler hidden")?;
    let lm_head_buffer = upload(
        library,
        u16_bytes(&lm_head),
        "Triton LM-head sampler weight",
    )?;
    let random_buffer = upload(
        library,
        f32_bytes(&random_uniforms),
        "Triton LM-head sampler random uniforms",
    )?;
    let logits_buffer =
        DeviceAllocation::new(library, logits_values * 4, "Triton LM-head sampler logits")?;
    let candidate_score_buffer = DeviceAllocation::new(
        library,
        candidate_values * 4,
        "Triton LM-head sampler candidate scores",
    )?;
    let candidate_index_buffer = DeviceAllocation::new(
        library,
        candidate_values * 4,
        "Triton LM-head sampler candidate indices",
    )?;
    let index_buffer =
        DeviceAllocation::new(library, args.rows * 4, "Triton LM-head sampler indices")?;
    let score_buffer =
        DeviceAllocation::new(library, args.rows * 4, "Triton LM-head sampler scores")?;
    let buffers = [
        PythonDeviceBufferArg {
            name: "hidden",
            ptr: hidden_buffer.buffer().ptr,
            bytes: hidden_buffer.buffer().bytes,
            device_id: hidden_buffer.buffer().device_id,
            flags: hidden_buffer.buffer().flags,
        },
        PythonDeviceBufferArg {
            name: "lm_head",
            ptr: lm_head_buffer.buffer().ptr,
            bytes: lm_head_buffer.buffer().bytes,
            device_id: lm_head_buffer.buffer().device_id,
            flags: lm_head_buffer.buffer().flags,
        },
        PythonDeviceBufferArg {
            name: "random_uniforms",
            ptr: random_buffer.buffer().ptr,
            bytes: random_buffer.buffer().bytes,
            device_id: random_buffer.buffer().device_id,
            flags: random_buffer.buffer().flags,
        },
        PythonDeviceBufferArg {
            name: "logits",
            ptr: logits_buffer.buffer().ptr,
            bytes: logits_buffer.buffer().bytes,
            device_id: logits_buffer.buffer().device_id,
            flags: logits_buffer.buffer().flags,
        },
        PythonDeviceBufferArg {
            name: "candidate_scores",
            ptr: candidate_score_buffer.buffer().ptr,
            bytes: candidate_score_buffer.buffer().bytes,
            device_id: candidate_score_buffer.buffer().device_id,
            flags: candidate_score_buffer.buffer().flags,
        },
        PythonDeviceBufferArg {
            name: "candidate_indices",
            ptr: candidate_index_buffer.buffer().ptr,
            bytes: candidate_index_buffer.buffer().bytes,
            device_id: candidate_index_buffer.buffer().device_id,
            flags: candidate_index_buffer.buffer().flags,
        },
        PythonDeviceBufferArg {
            name: "out_indices",
            ptr: index_buffer.buffer().ptr,
            bytes: index_buffer.buffer().bytes,
            device_id: index_buffer.buffer().device_id,
            flags: index_buffer.buffer().flags,
        },
        PythonDeviceBufferArg {
            name: "out_scores",
            ptr: score_buffer.buffer().ptr,
            bytes: score_buffer.buffer().bytes,
            device_id: score_buffer.buffer().device_id,
            flags: score_buffer.buffer().flags,
        },
    ];
    let kwargs = [
        ("rows", PythonKernelArg::Usize(args.rows)),
        ("hidden_dim", PythonKernelArg::Usize(args.hidden_dim)),
        ("vocab", PythonKernelArg::Usize(args.vocab)),
        ("temperature", PythonKernelArg::F64(0.7)),
        ("top_k", PythonKernelArg::Usize(top_k)),
        ("top_p", PythonKernelArg::F64(0.95)),
    ];
    launch_python_graph_capture(PythonGraphCaptureLaunch {
        module: "triton_sampling_capture",
        function: "capture_lm_head_sample_topk_topp",
        cuda_stream: stream.raw(),
        buffers: &buffers,
        kwargs: &kwargs,
    })
    .context("warming Triton LM-head sampler graph benchmark")?;
    unsafe {
        stream
            .synchronize()
            .context("synchronizing Triton LM-head sampler graph benchmark warmup")?;
        library
            .cuda_graph_begin_capture(stream.raw())
            .context("beginning Triton LM-head sampler CUDA graph capture")?;
        launch_python_graph_capture(PythonGraphCaptureLaunch {
            module: "triton_sampling_capture",
            function: "capture_lm_head_sample_topk_topp",
            cuda_stream: stream.raw(),
            buffers: &buffers,
            kwargs: &kwargs,
        })
        .context("capturing Triton LM-head sampler graph benchmark")?;
        let graph = BenchCudaGraph::new(
            library,
            library
                .cuda_graph_end_capture_retained(stream.raw())
                .context("ending Triton LM-head sampler CUDA graph capture")?,
        )?;
        let timing = time_kernel(args, || {
            library
                .cuda_graph_launch(graph.graph_exec(), stream.raw())
                .context("launching Triton LM-head sampler CUDA graph")?;
            stream.synchronize()
        })?;
        emit_timing(
            native_lib,
            "triton_lm_head_sample_topk_topp_bf16_graph",
            &json!({
                "rows": args.rows,
                "hidden_dim": args.hidden_dim,
                "vocab": args.vocab,
                "top_k": top_k,
                "temperature": 0.7,
                "top_p": 0.95,
                "logical_payload_bytes": hidden_values * 2
                    + weight_values * 2
                    + logits_values * 4
                    + candidate_values * 8
                    + args.rows * 12,
                "captured_nodes": graph.node_count(),
                "captured_kernel_nodes": graph.kernel_node_count(),
            }),
            args,
            timing,
        )
    }
}

fn bench_embedding_lookup_bf16(
    library: &NativeLibrary,
    stream: &CudaStream<'_>,
    native_lib: &Path,
    args: &BenchCudaKernelsArgs,
) -> Result<()> {
    let embedding_values = args
        .vocab
        .checked_mul(args.hidden_dim)
        .context("embedding table value count overflow")?;
    let output_values = args
        .rows
        .checked_mul(args.hidden_dim)
        .context("embedding output value count overflow")?;
    let embedding = bf16_pattern(embedding_values);
    let token_ids = (0..args.rows)
        .map(|idx| (idx % args.vocab) as u32)
        .collect::<Vec<_>>();
    let embedding_buffer = upload(library, u16_bytes(&embedding), "embedding table")?;
    let token_buffer = upload(library, u32_bytes(&token_ids), "embedding token ids")?;
    let out_buffer = DeviceAllocation::new(library, output_values * 2, "embedding output")?;
    let timing = time_kernel(args, || unsafe {
        library.cuda_embedding_lookup_bf16_async(
            embedding_buffer.buffer(),
            token_buffer.buffer(),
            out_buffer.buffer(),
            args.rows,
            args.vocab,
            args.hidden_dim,
            stream.raw(),
        )?;
        stream.synchronize()
    })?;
    emit_timing(
        native_lib,
        "embedding_lookup_bf16",
        &json!({
            "rows": args.rows,
            "vocab": args.vocab,
            "hidden_dim": args.hidden_dim,
            "logical_payload_bytes": args.rows * std::mem::size_of::<u32>() + output_values * 2 * 2,
        }),
        args,
        timing,
    )
}

fn bench_nvfp4_route_staged(
    library: &NativeLibrary,
    stream: &CudaStream<'_>,
    native_lib: &Path,
    args: &BenchCudaKernelsArgs,
) -> Result<()> {
    let hidden_values = args
        .rows
        .checked_mul(args.hidden_dim)
        .context("NVFP4 route hidden value count overflow")?;
    let packed_hidden_bytes = args.hidden_dim.div_ceil(2);
    let hidden_scale_bytes = args.hidden_dim.div_ceil(16);
    let packed_intermediate_bytes = args.intermediate_dim.div_ceil(2);
    let intermediate_scale_bytes = args.intermediate_dim.div_ceil(16);
    let gate_weight_bytes = args
        .intermediate_dim
        .checked_mul(packed_hidden_bytes)
        .context("NVFP4 gate weight byte count overflow")?;
    let gate_scale_bytes = args
        .intermediate_dim
        .checked_mul(hidden_scale_bytes)
        .context("NVFP4 gate scale byte count overflow")?;
    let down_weight_bytes = args
        .output_dim
        .checked_mul(packed_intermediate_bytes)
        .context("NVFP4 down weight byte count overflow")?;
    let down_scale_bytes = args
        .output_dim
        .checked_mul(intermediate_scale_bytes)
        .context("NVFP4 down scale byte count overflow")?;
    let output_values = args
        .rows
        .checked_mul(args.output_dim)
        .context("NVFP4 output value count overflow")?;
    let activation_values = args
        .routes
        .checked_mul(args.intermediate_dim)
        .context("NVFP4 activation workspace value count overflow")?;

    let hidden = bf16_pattern(hidden_values);
    let row_indices = (0..args.routes)
        .map(|idx| (idx % args.rows) as u32)
        .collect::<Vec<_>>();
    let route_weights = vec![1.0_f32 / args.routes as f32; args.routes];
    let gate_weight = nvfp4_packed_pattern(gate_weight_bytes);
    let gate_scale = vec![0x38_u8; gate_scale_bytes];
    let up_weight = nvfp4_packed_pattern(gate_weight_bytes);
    let up_scale = vec![0x38_u8; gate_scale_bytes];
    let down_weight = nvfp4_packed_pattern(down_weight_bytes);
    let down_scale = vec![0x38_u8; down_scale_bytes];

    let hidden_buffer = upload(library, u16_bytes(&hidden), "NVFP4 route hidden")?;
    let row_indices_buffer = upload(library, u32_bytes(&row_indices), "NVFP4 row indices")?;
    let route_weights_buffer = upload(library, f32_bytes(&route_weights), "NVFP4 route weights")?;
    let gate_weight_buffer = upload(library, &gate_weight, "NVFP4 gate weight")?;
    let gate_scale_buffer = upload(library, &gate_scale, "NVFP4 gate scale")?;
    let up_weight_buffer = upload(library, &up_weight, "NVFP4 up weight")?;
    let up_scale_buffer = upload(library, &up_scale, "NVFP4 up scale")?;
    let down_weight_buffer = upload(library, &down_weight, "NVFP4 down weight")?;
    let down_scale_buffer = upload(library, &down_scale, "NVFP4 down scale")?;
    let activation_buffer =
        DeviceAllocation::new(library, activation_values * 4, "NVFP4 activation workspace")?;
    let accumulator_buffer =
        DeviceAllocation::new(library, output_values * 4, "NVFP4 accumulator")?;
    let out_buffer = DeviceAllocation::new(library, output_values * 2, "NVFP4 BF16 output")?;

    let timing = time_kernel(args, || unsafe {
        library.cuda_zero_f32_async(accumulator_buffer.buffer(), output_values, stream.raw())?;
        library.cuda_nvfp4_silu_gated_mlp_route_bf16_grouped_staged_accumulate_f32_async(
            hidden_buffer.buffer(),
            row_indices_buffer.buffer(),
            route_weights_buffer.buffer(),
            gate_weight_buffer.buffer(),
            gate_scale_buffer.buffer(),
            up_weight_buffer.buffer(),
            up_scale_buffer.buffer(),
            down_weight_buffer.buffer(),
            down_scale_buffer.buffer(),
            activation_buffer.buffer(),
            accumulator_buffer.buffer(),
            args.rows,
            args.routes,
            args.hidden_dim,
            args.hidden_dim,
            args.intermediate_dim,
            args.output_dim,
            packed_intermediate_bytes,
            intermediate_scale_bytes,
            1.0,
            1.0,
            1.0,
            stream.raw(),
        )?;
        library.cuda_f32_to_bf16_async(
            accumulator_buffer.buffer(),
            out_buffer.buffer(),
            output_values,
            stream.raw(),
        )?;
        stream.synchronize()
    })?;
    emit_timing(
        native_lib,
        "nvfp4_route_bf16_staged_accumulate_pack",
        &json!({
            "rows": args.rows,
            "routes": args.routes,
            "hidden_dim": args.hidden_dim,
            "intermediate_dim": args.intermediate_dim,
            "output_dim": args.output_dim,
            "includes_accumulator_zero": true,
            "includes_bf16_pack": true,
            "logical_payload_bytes": args.routes * args.hidden_dim * 2
                + gate_weight_bytes * 2
                + gate_scale_bytes * 2
                + down_weight_bytes
                + down_scale_bytes
                + output_values * 2,
        }),
        args,
        timing,
    )
}

fn bench_phase0_layer_sweep_replay(
    library: &NativeLibrary,
    stream: &CudaStream<'_>,
    native_lib: &Path,
    args: &BenchCudaKernelsArgs,
) -> Result<()> {
    if !coordinator_python_capture_enabled() {
        anyhow::bail!("phase0 layer-sweep replay requires coordinator Python/Triton graph capture");
    }
    if args.hidden_dim != args.output_dim {
        anyhow::bail!(
            "phase0 layer-sweep replay requires hidden_dim == output_dim, got hidden_dim={} output_dim={}",
            args.hidden_dim,
            args.output_dim
        );
    }
    let experts = args.routes;
    let top_k = router_top_k(args, experts, "phase0 layer-sweep replay router")?;
    let route_count = args
        .rows
        .checked_mul(top_k)
        .context("phase0 layer-sweep route count overflow")?;
    let rows_i32 = i32::try_from(args.rows).context("phase0 layer-sweep rows exceed i32")?;
    let hidden_i32 =
        i32::try_from(args.hidden_dim).context("phase0 layer-sweep hidden_dim exceeds i32")?;
    let hidden_values = args
        .rows
        .checked_mul(args.hidden_dim)
        .context("phase0 layer-sweep hidden value count overflow")?;
    let intermediate_values = args
        .rows
        .checked_mul(args.intermediate_dim)
        .context("phase0 layer-sweep intermediate value count overflow")?;
    let linear_weight_values = args
        .hidden_dim
        .checked_mul(args.hidden_dim)
        .context("phase0 layer-sweep linear weight value count overflow")?;
    let router_weight_values = experts
        .checked_mul(args.hidden_dim)
        .context("phase0 layer-sweep router weight value count overflow")?;
    let router_output_values = args
        .rows
        .checked_mul(top_k)
        .context("phase0 layer-sweep router output value count overflow")?;
    let router_score_scratch_values = args
        .rows
        .checked_mul(experts)
        .context("phase0 layer-sweep router score scratch overflow")?;
    let projected_values = args
        .rows
        .checked_mul(MLA_KV_MXFP4_PROJECTED_VALUES)
        .context("phase0 layer-sweep KV projected value count overflow")?;
    let packed_kv_bytes = args
        .rows
        .checked_mul(MLA_KV_MXFP4_PACKED_BYTES)
        .context("phase0 layer-sweep KV packed byte count overflow")?;

    let norm_input = upload(
        library,
        u16_bytes(&bf16_pattern(hidden_values)),
        "phase0 replay RMSNorm input",
    )?;
    let norm_weight = upload(
        library,
        u16_bytes(&bf16_pattern(args.hidden_dim)),
        "phase0 replay RMSNorm weight",
    )?;
    let norm_out = DeviceAllocation::new(library, hidden_values * 2, "phase0 replay RMSNorm out")?;

    let residual = upload(
        library,
        u16_bytes(&bf16_pattern(hidden_values)),
        "phase0 replay residual",
    )?;
    let residual_delta = upload(
        library,
        u16_bytes(&bf16_replay_pattern(hidden_values)),
        "phase0 replay residual delta",
    )?;
    let residual_out =
        DeviceAllocation::new(library, hidden_values * 2, "phase0 replay residual out")?;

    let linear_input = upload(
        library,
        u16_bytes(&bf16_pattern(hidden_values)),
        "phase0 replay cuBLAS linear input",
    )?;
    let linear_weight = upload(
        library,
        u16_bytes(&bf16_pattern(linear_weight_values)),
        "phase0 replay cuBLAS linear weight",
    )?;
    let linear_out = DeviceAllocation::new(
        library,
        hidden_values * 2,
        "phase0 replay cuBLAS linear out",
    )?;

    let q_nope_values = args
        .rows
        .checked_mul(GLM52_MLA_ATTENTION_HEADS)
        .and_then(|values| values.checked_mul(GLM52_MLA_QK_NOPE_HEAD_DIM))
        .context("phase0 replay q_nope value count overflow")?;
    let q_rope_values = args
        .rows
        .checked_mul(GLM52_MLA_ATTENTION_HEADS)
        .and_then(|values| values.checked_mul(GLM52_MLA_QK_ROPE_HEAD_DIM))
        .context("phase0 replay q_rope value count overflow")?;
    let k_rope_values = args
        .rows
        .checked_mul(GLM52_MLA_QK_ROPE_HEAD_DIM)
        .context("phase0 replay k_rope value count overflow")?;
    let v_values = args
        .rows
        .checked_mul(GLM52_MLA_ATTENTION_HEADS)
        .and_then(|values| values.checked_mul(GLM52_MLA_V_HEAD_DIM))
        .context("phase0 replay value count overflow")?;
    let q_nope = upload(
        library,
        u16_bytes(&bf16_pattern(q_nope_values)),
        "phase0 replay q_nope",
    )?;
    let q_rope = upload(
        library,
        u16_bytes(&bf16_pattern(q_rope_values)),
        "phase0 replay q_rope",
    )?;
    let k_nope = upload(
        library,
        u16_bytes(&bf16_replay_pattern(q_nope_values)),
        "phase0 replay k_nope",
    )?;
    let k_rope = upload(
        library,
        u16_bytes(&bf16_replay_pattern(k_rope_values)),
        "phase0 replay k_rope",
    )?;
    let value = upload(
        library,
        u16_bytes(&bf16_pattern(v_values)),
        "phase0 replay attention value",
    )?;
    let attention_out =
        DeviceAllocation::new(library, v_values * 2, "phase0 replay attention out")?;
    let attention_scale =
        1.0_f32 / ((GLM52_MLA_QK_NOPE_HEAD_DIM + GLM52_MLA_QK_ROPE_HEAD_DIM) as f32).sqrt();

    let kv_projected = upload(
        library,
        u16_bytes(&bf16_pattern(projected_values)),
        "phase0 replay MXFP4 KV projected",
    )?;
    let kv_packed = DeviceAllocation::new(library, packed_kv_bytes, "phase0 replay MXFP4 KV out")?;

    let dense_gate_values = args
        .intermediate_dim
        .checked_mul(args.hidden_dim)
        .context("phase0 replay dense MLP gate value count overflow")?;
    let dense_down_values = args
        .hidden_dim
        .checked_mul(args.intermediate_dim)
        .context("phase0 replay dense MLP down value count overflow")?;
    let dense_input = upload(
        library,
        u16_bytes(&bf16_pattern(hidden_values)),
        "phase0 replay Triton dense MLP input",
    )?;
    let dense_gate = upload(
        library,
        u16_bytes(&bf16_pattern(dense_gate_values)),
        "phase0 replay Triton dense MLP gate",
    )?;
    let dense_up = upload(
        library,
        u16_bytes(&bf16_pattern(dense_gate_values)),
        "phase0 replay Triton dense MLP up",
    )?;
    let dense_down = upload(
        library,
        u16_bytes(&bf16_pattern(dense_down_values)),
        "phase0 replay Triton dense MLP down",
    )?;
    let dense_gate_out = DeviceAllocation::new(
        library,
        intermediate_values * 4,
        "phase0 replay Triton dense MLP gate out",
    )?;
    let dense_up_out = DeviceAllocation::new(
        library,
        intermediate_values * 4,
        "phase0 replay Triton dense MLP up out",
    )?;
    let dense_activation = DeviceAllocation::new(
        library,
        intermediate_values * 4,
        "phase0 replay Triton dense MLP activation",
    )?;
    let dense_out = DeviceAllocation::new(
        library,
        hidden_values * 2,
        "phase0 replay Triton dense MLP out",
    )?;
    let dense_buffers = [
        python_buffer("input", dense_input.buffer()),
        python_buffer("gate_weight", dense_gate.buffer()),
        python_buffer("up_weight", dense_up.buffer()),
        python_buffer("down_weight", dense_down.buffer()),
        python_buffer("gate_output", dense_gate_out.buffer()),
        python_buffer("up_output", dense_up_out.buffer()),
        python_buffer("activation", dense_activation.buffer()),
        python_buffer("output", dense_out.buffer()),
    ];
    let dense_kwargs = [
        ("rows", PythonKernelArg::Usize(args.rows)),
        ("hidden", PythonKernelArg::Usize(args.hidden_dim)),
        (
            "intermediate",
            PythonKernelArg::Usize(args.intermediate_dim),
        ),
        ("down_stride", PythonKernelArg::Usize(args.intermediate_dim)),
    ];
    let dense_graph = capture_python_graph(
        library,
        stream,
        "triton_mlp_capture",
        "capture_dense_mlp",
        &dense_buffers,
        &dense_kwargs,
        "phase0 replay Triton dense MLP",
    )?;

    let router_hidden = upload(
        library,
        u16_bytes(&bf16_pattern(hidden_values)),
        "phase0 replay Triton router hidden",
    )?;
    let router_weight = upload(
        library,
        u16_bytes(&bf16_pattern(router_weight_values)),
        "phase0 replay Triton router weight",
    )?;
    let router_bias = upload(
        library,
        f32_bytes(
            &(0..experts)
                .map(|idx| ((idx % 7) as f32 - 3.0) * 0.03125)
                .collect::<Vec<_>>(),
        ),
        "phase0 replay Triton router correction bias",
    )?;
    let router_score_scratch = DeviceAllocation::new(
        library,
        router_score_scratch_values * 4,
        "phase0 replay Triton router score scratch",
    )?;
    let router_indices = DeviceAllocation::new(
        library,
        router_output_values * 4,
        "phase0 replay router indices",
    )?;
    let router_scores = DeviceAllocation::new(
        library,
        router_output_values * 4,
        "phase0 replay router scores",
    )?;
    let router_weights = DeviceAllocation::new(
        library,
        router_output_values * 4,
        "phase0 replay router weights",
    )?;
    let router_buffers = [
        python_buffer("hidden", router_hidden.buffer()),
        python_buffer("router_weight", router_weight.buffer()),
        python_buffer("correction_bias", router_bias.buffer()),
        python_buffer("score_scratch", router_score_scratch.buffer()),
        python_buffer("topk_indices", router_indices.buffer()),
        python_buffer("topk_scores", router_scores.buffer()),
        python_buffer("topk_weights", router_weights.buffer()),
    ];
    let router_kwargs = [
        ("rows", PythonKernelArg::Usize(args.rows)),
        ("hidden_dim", PythonKernelArg::Usize(args.hidden_dim)),
        ("experts", PythonKernelArg::Usize(experts)),
        ("top_k", PythonKernelArg::Usize(top_k)),
        (
            "routed_scaling_factor",
            PythonKernelArg::F64(GLM52_ROUTED_SCALING_FACTOR as f64),
        ),
    ];
    let router_graph = capture_python_graph(
        library,
        stream,
        "triton_router_capture",
        "capture_router_topk",
        &router_buffers,
        &router_kwargs,
        "phase0 replay Triton router",
    )?;

    let packed_hidden_bytes = args.hidden_dim.div_ceil(2);
    let hidden_scale_bytes = args.hidden_dim.div_ceil(16);
    let packed_intermediate_bytes = args.intermediate_dim.div_ceil(2);
    let intermediate_scale_bytes = args.intermediate_dim.div_ceil(16);
    let gate_weight_bytes = args
        .intermediate_dim
        .checked_mul(packed_hidden_bytes)
        .context("phase0 replay NVFP4 gate weight byte count overflow")?;
    let gate_scale_bytes = args
        .intermediate_dim
        .checked_mul(hidden_scale_bytes)
        .context("phase0 replay NVFP4 gate scale byte count overflow")?;
    let down_weight_bytes = args
        .output_dim
        .checked_mul(packed_intermediate_bytes)
        .context("phase0 replay NVFP4 down weight byte count overflow")?;
    let down_scale_bytes = args
        .output_dim
        .checked_mul(intermediate_scale_bytes)
        .context("phase0 replay NVFP4 down scale byte count overflow")?;
    let nvfp4_hidden = upload(
        library,
        u16_bytes(&bf16_pattern(hidden_values)),
        "phase0 replay NVFP4 hidden",
    )?;
    let row_indices = (0..route_count)
        .map(|idx| (idx % args.rows) as u32)
        .collect::<Vec<_>>();
    let nvfp4_row_indices = upload(
        library,
        u32_bytes(&row_indices),
        "phase0 replay NVFP4 row indices",
    )?;
    let route_weights = vec![1.0_f32 / top_k as f32; route_count];
    let nvfp4_route_weights = upload(
        library,
        f32_bytes(&route_weights),
        "phase0 replay NVFP4 route weights",
    )?;
    let nvfp4_gate_weight = upload(
        library,
        &nvfp4_packed_pattern(gate_weight_bytes),
        "phase0 replay NVFP4 gate weight",
    )?;
    let nvfp4_gate_scale = upload(
        library,
        &vec![0x38_u8; gate_scale_bytes],
        "phase0 replay NVFP4 gate scale",
    )?;
    let nvfp4_up_weight = upload(
        library,
        &nvfp4_packed_pattern(gate_weight_bytes),
        "phase0 replay NVFP4 up weight",
    )?;
    let nvfp4_up_scale = upload(
        library,
        &vec![0x38_u8; gate_scale_bytes],
        "phase0 replay NVFP4 up scale",
    )?;
    let nvfp4_down_weight = upload(
        library,
        &nvfp4_packed_pattern(down_weight_bytes),
        "phase0 replay NVFP4 down weight",
    )?;
    let nvfp4_down_scale = upload(
        library,
        &vec![0x38_u8; down_scale_bytes],
        "phase0 replay NVFP4 down scale",
    )?;
    let nvfp4_activation = DeviceAllocation::new(
        library,
        route_count * args.intermediate_dim * 4,
        "phase0 replay NVFP4 activation",
    )?;
    let nvfp4_accumulator = DeviceAllocation::new(
        library,
        hidden_values * 4,
        "phase0 replay NVFP4 accumulator",
    )?;
    let nvfp4_out = DeviceAllocation::new(library, hidden_values * 2, "phase0 replay NVFP4 out")?;

    let lm_hidden = upload(
        library,
        u16_bytes(&bf16_pattern(hidden_values)),
        "phase0 replay Triton LM-head hidden",
    )?;
    let lm_head = upload(
        library,
        u16_bytes(&bf16_pattern(args.vocab * args.hidden_dim)),
        "phase0 replay Triton LM-head weight",
    )?;
    let lm_random = upload(
        library,
        f32_bytes(&sample_random_uniforms(args.rows)),
        "phase0 replay Triton LM-head random uniforms",
    )?;
    let lm_logits = DeviceAllocation::new(
        library,
        args.rows * args.vocab * 4,
        "phase0 replay LM logits",
    )?;
    let vocab_blocks = args.vocab.div_ceil(1024);
    let lm_candidate_values = args
        .rows
        .checked_mul(vocab_blocks)
        .and_then(|values| values.checked_mul(top_k))
        .context("phase0 replay LM-head candidate count overflow")?;
    let lm_candidate_scores = DeviceAllocation::new(
        library,
        lm_candidate_values * 4,
        "phase0 replay LM candidate scores",
    )?;
    let lm_candidate_indices = DeviceAllocation::new(
        library,
        lm_candidate_values * 4,
        "phase0 replay LM candidate indices",
    )?;
    let lm_indices =
        DeviceAllocation::new(library, args.rows * 4, "phase0 replay LM output indices")?;
    let lm_scores =
        DeviceAllocation::new(library, args.rows * 4, "phase0 replay LM output scores")?;
    let lm_buffers = [
        python_buffer("hidden", lm_hidden.buffer()),
        python_buffer("lm_head", lm_head.buffer()),
        python_buffer("random_uniforms", lm_random.buffer()),
        python_buffer("logits", lm_logits.buffer()),
        python_buffer("candidate_scores", lm_candidate_scores.buffer()),
        python_buffer("candidate_indices", lm_candidate_indices.buffer()),
        python_buffer("out_indices", lm_indices.buffer()),
        python_buffer("out_scores", lm_scores.buffer()),
    ];
    let lm_kwargs = [
        ("rows", PythonKernelArg::Usize(args.rows)),
        ("hidden_dim", PythonKernelArg::Usize(args.hidden_dim)),
        ("vocab", PythonKernelArg::Usize(args.vocab)),
        ("temperature", PythonKernelArg::F64(0.7)),
        ("top_k", PythonKernelArg::Usize(top_k)),
        ("top_p", PythonKernelArg::F64(0.95)),
    ];
    let lm_graph = capture_python_graph(
        library,
        stream,
        "triton_sampling_capture",
        "capture_lm_head_sample_topk_topp",
        &lm_buffers,
        &lm_kwargs,
        "phase0 replay Triton LM-head",
    )?;

    let launch_dense_layer = || -> Result<()> {
        unsafe {
            library.cuda_rmsnorm_bf16_async(
                norm_input.buffer(),
                norm_weight.buffer(),
                norm_out.buffer(),
                rows_i32,
                hidden_i32,
                1.0e-5,
                stream.raw(),
            )?;
            for _ in 0..4 {
                library.cuda_linear_bf16_cublas_async(
                    linear_input.buffer(),
                    linear_weight.buffer(),
                    None,
                    linear_out.buffer(),
                    args.rows,
                    args.hidden_dim,
                    args.hidden_dim,
                    stream.raw(),
                )?;
            }
            library.cuda_mla_rope_attention_bf16_async(
                q_nope.buffer(),
                q_rope.buffer(),
                k_nope.buffer(),
                k_rope.buffer(),
                value.buffer(),
                attention_out.buffer(),
                args.rows,
                GLM52_MLA_ATTENTION_HEADS,
                GLM52_MLA_QK_NOPE_HEAD_DIM,
                GLM52_MLA_QK_ROPE_HEAD_DIM,
                GLM52_MLA_V_HEAD_DIM,
                attention_scale,
                stream.raw(),
            )?;
            library.cuda_mla_kv_pack_mxfp4_ds_mla(
                kv_projected.buffer(),
                kv_packed.buffer(),
                args.rows,
                MLA_KV_MXFP4_PROJECTED_STRIDE_BYTES,
                MLA_KV_MXFP4_PACKED_BYTES,
            )?;
            library
                .cuda_graph_launch(dense_graph.graph_exec(), stream.raw())
                .context("launching phase0 replay Triton dense MLP graph")?;
            library.cuda_residual_add_bf16_async(
                residual.buffer(),
                residual_delta.buffer(),
                residual_out.buffer(),
                hidden_values,
                stream.raw(),
            )?;
            library.cuda_residual_add_bf16_async(
                residual_out.buffer(),
                residual_delta.buffer(),
                residual.buffer(),
                hidden_values,
                stream.raw(),
            )?;
        }
        Ok(())
    };
    let launch_sparse_layer = || -> Result<()> {
        unsafe {
            library.cuda_rmsnorm_bf16_async(
                norm_input.buffer(),
                norm_weight.buffer(),
                norm_out.buffer(),
                rows_i32,
                hidden_i32,
                1.0e-5,
                stream.raw(),
            )?;
            library.cuda_mla_rope_attention_bf16_async(
                q_nope.buffer(),
                q_rope.buffer(),
                k_nope.buffer(),
                k_rope.buffer(),
                value.buffer(),
                attention_out.buffer(),
                args.rows,
                GLM52_MLA_ATTENTION_HEADS,
                GLM52_MLA_QK_NOPE_HEAD_DIM,
                GLM52_MLA_QK_ROPE_HEAD_DIM,
                GLM52_MLA_V_HEAD_DIM,
                attention_scale,
                stream.raw(),
            )?;
            library
                .cuda_graph_launch(router_graph.graph_exec(), stream.raw())
                .context("launching phase0 replay Triton router graph")?;
            library.cuda_mla_kv_pack_mxfp4_ds_mla(
                kv_projected.buffer(),
                kv_packed.buffer(),
                args.rows,
                MLA_KV_MXFP4_PROJECTED_STRIDE_BYTES,
                MLA_KV_MXFP4_PACKED_BYTES,
            )?;
            library.cuda_zero_f32_async(nvfp4_accumulator.buffer(), hidden_values, stream.raw())?;
            library.cuda_nvfp4_silu_gated_mlp_route_bf16_grouped_staged_accumulate_f32_async(
                nvfp4_hidden.buffer(),
                nvfp4_row_indices.buffer(),
                nvfp4_route_weights.buffer(),
                nvfp4_gate_weight.buffer(),
                nvfp4_gate_scale.buffer(),
                nvfp4_up_weight.buffer(),
                nvfp4_up_scale.buffer(),
                nvfp4_down_weight.buffer(),
                nvfp4_down_scale.buffer(),
                nvfp4_activation.buffer(),
                nvfp4_accumulator.buffer(),
                args.rows,
                route_count,
                args.hidden_dim,
                args.hidden_dim,
                args.intermediate_dim,
                args.output_dim,
                packed_intermediate_bytes,
                intermediate_scale_bytes,
                1.0,
                1.0,
                1.0,
                stream.raw(),
            )?;
            library.cuda_f32_to_bf16_async(
                nvfp4_accumulator.buffer(),
                nvfp4_out.buffer(),
                hidden_values,
                stream.raw(),
            )?;
            library.cuda_residual_add_bf16_async(
                residual.buffer(),
                residual_delta.buffer(),
                residual_out.buffer(),
                hidden_values,
                stream.raw(),
            )?;
            library.cuda_residual_add_bf16_async(
                residual_out.buffer(),
                nvfp4_out.buffer(),
                residual.buffer(),
                hidden_values,
                stream.raw(),
            )?;
        }
        Ok(())
    };

    let dense_timing = time_kernel(args, || {
        launch_dense_layer()?;
        unsafe { stream.synchronize() }
    })?;
    emit_layer_replay_timing(
        native_lib,
        "phase0_dense_layer_replay",
        &json!({
            "rows": args.rows,
            "scope": "single dense layer 0..2",
            "dense_layers": 1,
            "sparse_layers": 0,
            "hidden_dim": args.hidden_dim,
            "intermediate_dim": args.intermediate_dim,
            "components": {
                "rmsnorm": 1,
                "cublas_linear_6144x6144": 4,
                "mla_rope_attention": 1,
                "mxfp4_kv_pack": 1,
                "triton_dense_mlp_graph": 1,
                "residual_add": 2
            },
        }),
        args,
        dense_timing,
    )?;

    let sparse_timing = time_kernel(args, || {
        launch_sparse_layer()?;
        unsafe { stream.synchronize() }
    })?;
    emit_layer_replay_timing(
        native_lib,
        "phase0_sparse_layer_replay",
        &json!({
            "rows": args.rows,
            "scope": "single sparse layer 3..77",
            "dense_layers": 0,
            "sparse_layers": 1,
            "hidden_dim": args.hidden_dim,
            "intermediate_dim": args.intermediate_dim,
            "experts": experts,
            "top_k": top_k,
            "route_count": route_count,
            "components": {
                "rmsnorm": 1,
                "mla_rope_attention": 1,
                "triton_router_graph": 1,
                "mxfp4_kv_pack": 1,
                "local_nvfp4_sparse_expert_mlp": 1,
                "residual_add": 2
            },
        }),
        args,
        sparse_timing,
    )?;

    let full_timing = time_kernel(args, || {
        for _ in 0..3 {
            launch_dense_layer()?;
        }
        for _ in 0..75 {
            launch_sparse_layer()?;
        }
        unsafe {
            library
                .cuda_graph_launch(lm_graph.graph_exec(), stream.raw())
                .context("launching phase0 replay Triton LM-head graph")?;
        }
        unsafe { stream.synchronize() }
    })?;
    let baseline_decode_ms = 1000.0 / 1.66_f64;
    emit_layer_replay_timing(
        native_lib,
        "phase0_full_78_layer_coordinator_local_replay",
        &json!({
            "rows": args.rows,
            "scope": "full 78-layer coordinator-local synthetic replay",
            "dense_layers": 3,
            "sparse_layers": 75,
            "hidden_dim": args.hidden_dim,
            "intermediate_dim": args.intermediate_dim,
            "experts": experts,
            "top_k": top_k,
            "route_count_per_sparse_layer": route_count,
            "terminal_lm_head_graph": true,
            "baseline_decode_tps": 1.66,
            "baseline_decode_ms": baseline_decode_ms,
            "components": {
                "dense_layer_replay": 3,
                "sparse_layer_replay": 75,
                "triton_lm_head_graph": 1
            },
        }),
        args,
        full_timing,
    )?;
    Ok(())
}

struct BenchCudaGraph<'a> {
    library: &'a NativeLibrary,
    graph: *mut c_void,
    graph_exec: *mut c_void,
    node_count: usize,
    kernel_node_count: usize,
    memcpy_node_count: usize,
}

impl<'a> BenchCudaGraph<'a> {
    fn new(library: &'a NativeLibrary, capture: GlmrtCudaGraphCaptureInfo) -> Result<Self> {
        if capture.graph.is_null() || capture.graph_exec.is_null() {
            anyhow::bail!("CUDA graph capture returned a null graph handle");
        }
        if capture.kernel_node_count == 0 {
            anyhow::bail!("CUDA graph capture did not include a kernel node");
        }
        Ok(Self {
            library,
            graph: capture.graph,
            graph_exec: capture.graph_exec,
            node_count: capture.node_count,
            kernel_node_count: capture.kernel_node_count,
            memcpy_node_count: capture.memcpy_node_count,
        })
    }

    fn graph_exec(&self) -> *mut c_void {
        self.graph_exec
    }

    fn node_count(&self) -> usize {
        self.node_count
    }

    fn kernel_node_count(&self) -> usize {
        self.kernel_node_count
    }

    fn memcpy_node_count(&self) -> usize {
        self.memcpy_node_count
    }
}

impl Drop for BenchCudaGraph<'_> {
    fn drop(&mut self) {
        if !self.graph_exec.is_null() {
            let _ = unsafe { self.library.cuda_graph_exec_destroy(self.graph_exec) };
            self.graph_exec = std::ptr::null_mut();
        }
        if !self.graph.is_null() {
            let _ = unsafe { self.library.cuda_graph_destroy(self.graph) };
            self.graph = std::ptr::null_mut();
        }
    }
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

fn capture_python_graph<'a>(
    library: &'a NativeLibrary,
    stream: &CudaStream<'_>,
    module: &'static str,
    function: &'static str,
    buffers: &[PythonDeviceBufferArg],
    kwargs: &[(&'static str, PythonKernelArg)],
    label: &str,
) -> Result<BenchCudaGraph<'a>> {
    launch_python_graph_capture(PythonGraphCaptureLaunch {
        module,
        function,
        cuda_stream: stream.raw(),
        buffers,
        kwargs,
    })
    .with_context(|| format!("warming {label}"))?;
    unsafe {
        stream
            .synchronize()
            .with_context(|| format!("synchronizing {label} warmup"))?;
        library
            .cuda_graph_begin_capture(stream.raw())
            .with_context(|| format!("beginning {label} CUDA graph capture"))?;
        launch_python_graph_capture(PythonGraphCaptureLaunch {
            module,
            function,
            cuda_stream: stream.raw(),
            buffers,
            kwargs,
        })
        .with_context(|| format!("capturing {label}"))?;
        BenchCudaGraph::new(
            library,
            library
                .cuda_graph_end_capture_retained(stream.raw())
                .with_context(|| format!("ending {label} CUDA graph capture"))?,
        )
    }
}

fn upload<'a>(
    library: &'a NativeLibrary,
    bytes: &[u8],
    label: &str,
) -> Result<DeviceAllocation<'a>> {
    let allocation = DeviceAllocation::new(library, bytes.len(), label)?;
    library
        .copy_h2d(allocation.buffer(), bytes)
        .with_context(|| format!("uploading CUDA microbenchmark buffer for {label}"))?;
    Ok(allocation)
}

fn time_kernel<F>(args: &BenchCudaKernelsArgs, mut run: F) -> Result<Timing>
where
    F: FnMut() -> Result<()>,
{
    for _ in 0..args.warmup_iterations {
        run()?;
    }
    let mut samples = Vec::with_capacity(args.iterations);
    for _ in 0..args.iterations {
        let start = Instant::now();
        run()?;
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let total: f64 = samples.iter().sum();
    let min_ms = samples.iter().copied().fold(f64::INFINITY, f64::min);
    let max_ms = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Ok(Timing {
        avg_ms: total / samples.len() as f64,
        min_ms,
        max_ms,
    })
}

fn emit_timing(
    native_lib: &Path,
    kernel: &str,
    dims: &serde_json::Value,
    args: &BenchCudaKernelsArgs,
    timing: Timing,
) -> Result<()> {
    let mut row = json!({
        "benchmark": "cuda_kernel_microbench",
        "kernel": kernel,
        "status": "ok",
        "native_lib": native_lib.display().to_string(),
        "warmup_iterations": args.warmup_iterations,
        "iterations": args.iterations,
        "avg_ms": timing.avg_ms,
        "min_ms": timing.min_ms,
        "max_ms": timing.max_ms,
    });
    let row_object = row.as_object_mut().expect("microbenchmark row is object");
    for (key, value) in dims.as_object().expect("microbenchmark dims are object") {
        row_object.insert(key.clone(), value.clone());
    }
    println!("{}", serde_json::to_string(&row)?);
    Ok(())
}

fn emit_layer_replay_timing(
    native_lib: &Path,
    kernel: &str,
    dims: &serde_json::Value,
    args: &BenchCudaKernelsArgs,
    timing: Timing,
) -> Result<()> {
    let mut row = json!({
        "benchmark": "phase0_layer_sweep_replay",
        "kernel": kernel,
        "status": "ok",
        "native_lib": native_lib.display().to_string(),
        "warmup_iterations": args.warmup_iterations,
        "iterations": args.iterations,
        "avg_ms": timing.avg_ms,
        "min_ms": timing.min_ms,
        "max_ms": timing.max_ms,
        "synthetic_weights": true,
        "coordinator_local": true,
        "uses_spark": false,
        "model_load_attempted": false,
    });
    let row_object = row.as_object_mut().expect("layer replay row is object");
    for (key, value) in dims.as_object().expect("layer replay dims are object") {
        row_object.insert(key.clone(), value.clone());
    }
    if kernel == "phase0_full_78_layer_coordinator_local_replay" {
        let baseline_decode_ms = row_object
            .get("baseline_decode_ms")
            .and_then(|value| value.as_f64())
            .unwrap_or(1000.0 / 1.66_f64);
        row_object.insert(
            "speedup_vs_phase0_baseline".to_owned(),
            json!(baseline_decode_ms / timing.avg_ms),
        );
        row_object.insert(
            "decode_tokens_per_second_equivalent".to_owned(),
            json!(1000.0 / timing.avg_ms),
        );
    }
    println!("{}", serde_json::to_string(&row)?);
    Ok(())
}

fn bf16_pattern(values: usize) -> Vec<u16> {
    (0..values)
        .map(|idx| {
            let value = 0.25_f32 + ((idx % 17) as f32) * 0.03125;
            f32_to_bf16(value)
        })
        .collect()
}

fn bf16_replay_pattern(values: usize) -> Vec<u16> {
    (0..values)
        .map(|idx| {
            let value = -0.375_f32 + ((idx % 23) as f32) * 0.046875;
            f32_to_bf16(value)
        })
        .collect()
}

fn logits_f32_pattern(rows: usize, vocab: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(rows * vocab);
    for row in 0..rows {
        for col in 0..vocab {
            let local = col as f32 / vocab as f32;
            values.push(local + row as f32 * 0.000_001);
        }
    }
    values
}

fn sample_random_uniforms(rows: usize) -> Vec<f32> {
    (0..rows)
        .map(|idx| ((idx % 997) as f32 + 0.5) / 997.0)
        .collect()
}

fn nvfp4_packed_pattern(bytes: usize) -> Vec<u8> {
    const PATTERN: [u8; 8] = [0x9a, 0x8b, 0xa9, 0xb8, 0x6a, 0xa6, 0x59, 0x95];
    (0..bytes).map(|idx| PATTERN[idx % PATTERN.len()]).collect()
}

fn f32_to_bf16(value: f32) -> u16 {
    ((value.to_bits() + 0x8000) >> 16) as u16
}

fn u16_bytes(values: &[u16]) -> &[u8] {
    unsafe { slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) }
}

fn u32_bytes(values: &[u32]) -> &[u8] {
    unsafe { slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) }
}

fn f32_bytes(values: &[f32]) -> &[u8] {
    unsafe { slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validation_args() -> BenchCudaKernelsArgs {
        BenchCudaKernelsArgs {
            native_lib: None,
            kernels: Vec::new(),
            rows: 1,
            hidden_dim: 16,
            intermediate_dim: 8,
            output_dim: 16,
            vocab: 32,
            routes: 8,
            top_k: 4,
            warmup_iterations: 1,
            iterations: 1,
            require_cuda: false,
        }
    }

    fn load_cuda_validation_library() -> Result<Option<NativeLibrary>> {
        let native_lib = native_library_path(&validation_args());
        if !native_lib.exists() {
            return Ok(None);
        }
        let library = unsafe { NativeLibrary::load(&native_lib) }.with_context(|| {
            format!("loading validation native library {}", native_lib.display())
        })?;
        let info = library
            .cuda_device_info(0)
            .context("querying CUDA device for Triton swap validation")?;
        if info.cuda_available != 1 {
            return Ok(None);
        }
        Ok(Some(library))
    }

    fn load_validation_library() -> Result<Option<NativeLibrary>> {
        if !coordinator_python_capture_enabled() {
            return Ok(None);
        }
        load_cuda_validation_library()
    }

    fn read_device_bytes(
        library: &NativeLibrary,
        buffer: GlmrtDeviceBuffer,
        bytes: usize,
    ) -> Result<Vec<u8>> {
        let mut out = vec![0_u8; bytes];
        library.copy_d2h(&mut out, buffer)?;
        Ok(out)
    }

    fn u32s_from_le_bytes(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(std::mem::size_of::<u32>())
            .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("u32 chunk")))
            .collect()
    }

    fn f32s_from_le_bytes(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 chunk")))
            .collect()
    }

    fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(std::mem::size_of::<u16>())
            .map(|chunk| {
                let bits = u16::from_le_bytes(chunk.try_into().expect("bf16 chunk"));
                f32::from_bits((bits as u32) << 16)
            })
            .collect()
    }

    fn assert_f32_close(actual: &[f32], expected: &[f32], rtol: f32, atol: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            let allowed = atol + rtol * expected.abs();
            assert!(
                (actual - expected).abs() <= allowed,
                "value {index} mismatch: actual={actual} expected={expected} allowed={allowed}"
            );
        }
    }

    #[test]
    fn cublas_linear_bf16_matches_native_synthetic_output() -> Result<()> {
        let Some(library) = load_cuda_validation_library()? else {
            return Ok(());
        };
        let stream = CudaStream::new(&library)?;
        let rows = 3;
        let input_dim = 16;
        let output_dim = 10;
        let input_values = rows * input_dim;
        let weight_values = output_dim * input_dim;
        let output_values = rows * output_dim;

        let input = (0..input_values)
            .map(|idx| f32_to_bf16(((idx % 7) as f32 - 3.0) * 0.03125))
            .collect::<Vec<_>>();
        let weight = (0..weight_values)
            .map(|idx| f32_to_bf16(((idx % 11) as f32 - 5.0) * 0.015625))
            .collect::<Vec<_>>();
        let bias = (0..output_dim)
            .map(|idx| f32_to_bf16((idx as f32 - 4.0) * 0.0078125))
            .collect::<Vec<_>>();
        let input_buffer = upload(
            &library,
            u16_bytes(&input),
            "validation cuBLAS linear input",
        )?;
        let weight_buffer = upload(
            &library,
            u16_bytes(&weight),
            "validation cuBLAS linear weight",
        )?;
        let bias_buffer = upload(&library, u16_bytes(&bias), "validation cuBLAS linear bias")?;
        let native_out =
            DeviceAllocation::new(&library, output_values * 2, "native linear output")?;
        let cublas_out =
            DeviceAllocation::new(&library, output_values * 2, "cuBLAS linear output")?;

        unsafe {
            library.cuda_linear_bf16_async(
                input_buffer.buffer(),
                weight_buffer.buffer(),
                Some(bias_buffer.buffer()),
                native_out.buffer(),
                rows,
                input_dim,
                output_dim,
                stream.raw(),
            )?;
            library.cuda_linear_bf16_cublas_async(
                input_buffer.buffer(),
                weight_buffer.buffer(),
                Some(bias_buffer.buffer()),
                cublas_out.buffer(),
                rows,
                input_dim,
                output_dim,
                stream.raw(),
            )?;
            stream.synchronize()?;
        }

        let native = bf16_bytes_to_f32(&read_device_bytes(
            &library,
            native_out.buffer(),
            output_values * 2,
        )?);
        let cublas = bf16_bytes_to_f32(&read_device_bytes(
            &library,
            cublas_out.buffer(),
            output_values * 2,
        )?);
        assert_f32_close(&cublas, &native, 1.0e-3, 1.0e-2);
        Ok(())
    }

    #[test]
    fn cub_router_topk_matches_native_synthetic_indices() -> Result<()> {
        let Some(library) = load_cuda_validation_library()? else {
            return Ok(());
        };
        let stream = CudaStream::new(&library)?;
        let rows = 3;
        let hidden = 16;
        let experts = 12;
        let top_k = 4;
        let hidden_values = rows * hidden;
        let weight_values = experts * hidden;
        let score_values = rows * experts;
        let output_values = rows * top_k;
        let hidden_bf16 = bf16_pattern(hidden_values);
        let router_weight = bf16_replay_pattern(weight_values);
        let correction_bias = (0..experts)
            .map(|idx| ((idx % 5) as f32 - 2.0) * 0.0078125 + idx as f32 * 1.0e-5)
            .collect::<Vec<_>>();
        let hidden_buffer = upload(&library, u16_bytes(&hidden_bf16), "CUB validation hidden")?;
        let weight_buffer = upload(
            &library,
            u16_bytes(&router_weight),
            "CUB validation router weight",
        )?;
        let bias_buffer = upload(
            &library,
            f32_bytes(&correction_bias),
            "CUB validation correction bias",
        )?;
        let native_indices =
            DeviceAllocation::new(&library, output_values * 4, "native router indices")?;
        let native_scores =
            DeviceAllocation::new(&library, output_values * 4, "native router scores")?;
        let native_weights =
            DeviceAllocation::new(&library, output_values * 4, "native router weights")?;
        let cub_indices = DeviceAllocation::new(&library, output_values * 4, "CUB router indices")?;
        let cub_scores = DeviceAllocation::new(&library, output_values * 4, "CUB router scores")?;
        let cub_weights = DeviceAllocation::new(&library, output_values * 4, "CUB router weights")?;
        let corrected_scores =
            DeviceAllocation::new(&library, score_values * 4, "CUB corrected router scores")?;
        let sorted_scores =
            DeviceAllocation::new(&library, score_values * 4, "CUB sorted router scores")?;
        let unsorted_indices =
            DeviceAllocation::new(&library, score_values * 4, "CUB unsorted router indices")?;
        let sorted_indices =
            DeviceAllocation::new(&library, score_values * 4, "CUB sorted router indices")?;
        let segment_offsets =
            DeviceAllocation::new(&library, (rows + 1) * 4, "CUB router segment offsets")?;
        let temp_storage =
            DeviceAllocation::new(&library, 8 * 1024 * 1024, "CUB router temp storage")?;

        unsafe {
            library.cuda_router_topk_bf16_async(
                hidden_buffer.buffer(),
                weight_buffer.buffer(),
                bias_buffer.buffer(),
                native_indices.buffer(),
                native_scores.buffer(),
                native_weights.buffer(),
                rows,
                hidden,
                experts,
                top_k,
                stream.raw(),
            )?;
            library.cuda_router_topk_bf16_cub_async(
                hidden_buffer.buffer(),
                weight_buffer.buffer(),
                bias_buffer.buffer(),
                corrected_scores.buffer(),
                sorted_scores.buffer(),
                unsorted_indices.buffer(),
                sorted_indices.buffer(),
                segment_offsets.buffer(),
                cub_indices.buffer(),
                cub_scores.buffer(),
                cub_weights.buffer(),
                temp_storage.buffer(),
                8 * 1024 * 1024,
                rows,
                hidden,
                experts,
                top_k,
                stream.raw(),
            )?;
            stream.synchronize()?;
        }

        let native_indices = u32s_from_le_bytes(&read_device_bytes(
            &library,
            native_indices.buffer(),
            output_values * 4,
        )?);
        let cub_indices = u32s_from_le_bytes(&read_device_bytes(
            &library,
            cub_indices.buffer(),
            output_values * 4,
        )?);
        assert_eq!(cub_indices, native_indices);
        let native_scores = f32s_from_le_bytes(&read_device_bytes(
            &library,
            native_scores.buffer(),
            output_values * 4,
        )?);
        let cub_scores = f32s_from_le_bytes(&read_device_bytes(
            &library,
            cub_scores.buffer(),
            output_values * 4,
        )?);
        assert_f32_close(&cub_scores, &native_scores, 1.0e-6, 1.0e-6);
        let native_weights = f32s_from_le_bytes(&read_device_bytes(
            &library,
            native_weights.buffer(),
            output_values * 4,
        )?);
        let cub_weights = f32s_from_le_bytes(&read_device_bytes(
            &library,
            cub_weights.buffer(),
            output_values * 4,
        )?);
        assert_f32_close(&cub_weights, &native_weights, 1.0e-6, 1.0e-6);
        Ok(())
    }

    #[test]
    fn cub_logits_sample_topk_topp_matches_native_synthetic_output() -> Result<()> {
        let Some(library) = load_cuda_validation_library()? else {
            return Ok(());
        };
        let stream = CudaStream::new(&library)?;
        let rows = 3;
        let vocab = 37;
        let top_k = 8;
        let logits_values = rows * vocab;
        let logits = logits_f32_pattern(rows, vocab);
        let random_uniforms = vec![0.05_f32, 0.44, 0.91];
        let logits_buffer = upload(&library, f32_bytes(&logits), "CUB logits validation logits")?;
        let random_buffer = upload(
            &library,
            f32_bytes(&random_uniforms),
            "CUB logits validation random uniforms",
        )?;
        let native_indices =
            DeviceAllocation::new(&library, rows * 4, "native logits sampler indices")?;
        let native_scores =
            DeviceAllocation::new(&library, rows * 4, "native logits sampler scores")?;
        let cub_indices = DeviceAllocation::new(&library, rows * 4, "CUB logits sampler indices")?;
        let cub_scores = DeviceAllocation::new(&library, rows * 4, "CUB logits sampler scores")?;
        let sorted_logits =
            DeviceAllocation::new(&library, logits_values * 4, "CUB sorted logits")?;
        let unsorted_indices =
            DeviceAllocation::new(&library, logits_values * 4, "CUB unsorted logits indices")?;
        let sorted_indices =
            DeviceAllocation::new(&library, logits_values * 4, "CUB sorted logits indices")?;
        let segment_offsets =
            DeviceAllocation::new(&library, (rows + 1) * 4, "CUB logits segment offsets")?;
        let temp_storage =
            DeviceAllocation::new(&library, 4 * 1024 * 1024, "CUB logits temp storage")?;

        unsafe {
            library.cuda_logits_sample_topk_topp_f32_async(
                logits_buffer.buffer(),
                random_buffer.buffer(),
                native_indices.buffer(),
                native_scores.buffer(),
                rows,
                vocab,
                0.7,
                top_k,
                0.95,
                stream.raw(),
            )?;
            library.cuda_logits_sample_topk_topp_f32_cub_async(
                logits_buffer.buffer(),
                random_buffer.buffer(),
                sorted_logits.buffer(),
                unsorted_indices.buffer(),
                sorted_indices.buffer(),
                segment_offsets.buffer(),
                cub_indices.buffer(),
                cub_scores.buffer(),
                temp_storage.buffer(),
                4 * 1024 * 1024,
                rows,
                vocab,
                0.7,
                top_k,
                0.95,
                stream.raw(),
            )?;
            stream.synchronize()?;
        }

        let native_indices = u32s_from_le_bytes(&read_device_bytes(
            &library,
            native_indices.buffer(),
            rows * 4,
        )?);
        let cub_indices = u32s_from_le_bytes(&read_device_bytes(
            &library,
            cub_indices.buffer(),
            rows * 4,
        )?);
        assert_eq!(cub_indices, native_indices);
        let native_scores = f32s_from_le_bytes(&read_device_bytes(
            &library,
            native_scores.buffer(),
            rows * 4,
        )?);
        let cub_scores =
            f32s_from_le_bytes(&read_device_bytes(&library, cub_scores.buffer(), rows * 4)?);
        assert_f32_close(&cub_scores, &native_scores, 1.0e-6, 1.0e-6);
        Ok(())
    }

    #[test]
    fn cub_lm_head_sample_topk_topp_matches_native_synthetic_output() -> Result<()> {
        let Some(library) = load_cuda_validation_library()? else {
            return Ok(());
        };
        let stream = CudaStream::new(&library)?;
        let rows = 2;
        let hidden = 16;
        let vocab = 37;
        let top_k = 8;
        let logits_values = rows * vocab;

        let mut hidden_values = vec![0_u16; rows * hidden];
        hidden_values[0] = f32_to_bf16(1.0);
        hidden_values[hidden + 1] = f32_to_bf16(1.0);
        let mut lm_head = vec![0_u16; vocab * hidden];
        for token in 0..vocab {
            lm_head[token * hidden] = f32_to_bf16(0.01 * token as f32);
            lm_head[token * hidden + 1] = f32_to_bf16(0.01 * (vocab - token) as f32);
        }
        let random_uniforms = vec![0.25_f32, 0.75];

        let hidden_buffer = upload(&library, u16_bytes(&hidden_values), "CUB LM-head hidden")?;
        let lm_head_buffer = upload(&library, u16_bytes(&lm_head), "CUB LM-head weight")?;
        let random_buffer = upload(
            &library,
            f32_bytes(&random_uniforms),
            "CUB LM-head random uniforms",
        )?;
        let native_indices =
            DeviceAllocation::new(&library, rows * 4, "native LM-head sampler indices")?;
        let native_scores =
            DeviceAllocation::new(&library, rows * 4, "native LM-head sampler scores")?;
        let cub_indices = DeviceAllocation::new(&library, rows * 4, "CUB LM-head sampler indices")?;
        let cub_scores = DeviceAllocation::new(&library, rows * 4, "CUB LM-head sampler scores")?;
        let logits_workspace =
            DeviceAllocation::new(&library, logits_values * 4, "CUB LM-head logits workspace")?;
        let sorted_logits =
            DeviceAllocation::new(&library, logits_values * 4, "CUB LM-head sorted logits")?;
        let unsorted_indices =
            DeviceAllocation::new(&library, logits_values * 4, "CUB LM-head unsorted indices")?;
        let sorted_indices =
            DeviceAllocation::new(&library, logits_values * 4, "CUB LM-head sorted indices")?;
        let segment_offsets =
            DeviceAllocation::new(&library, (rows + 1) * 4, "CUB LM-head segment offsets")?;
        let temp_storage =
            DeviceAllocation::new(&library, 4 * 1024 * 1024, "CUB LM-head temp storage")?;

        unsafe {
            library.cuda_lm_head_sample_topk_topp_bf16_async(
                hidden_buffer.buffer(),
                lm_head_buffer.buffer(),
                random_buffer.buffer(),
                native_indices.buffer(),
                native_scores.buffer(),
                rows,
                hidden,
                vocab,
                0.7,
                top_k,
                0.95,
                stream.raw(),
            )?;
            library.cuda_lm_head_sample_topk_topp_bf16_cub_async(
                hidden_buffer.buffer(),
                lm_head_buffer.buffer(),
                random_buffer.buffer(),
                logits_workspace.buffer(),
                sorted_logits.buffer(),
                unsorted_indices.buffer(),
                sorted_indices.buffer(),
                segment_offsets.buffer(),
                cub_indices.buffer(),
                cub_scores.buffer(),
                temp_storage.buffer(),
                4 * 1024 * 1024,
                rows,
                hidden,
                vocab,
                0.7,
                top_k,
                0.95,
                stream.raw(),
            )?;
            stream.synchronize()?;
        }

        let native_indices = u32s_from_le_bytes(&read_device_bytes(
            &library,
            native_indices.buffer(),
            rows * 4,
        )?);
        let cub_indices = u32s_from_le_bytes(&read_device_bytes(
            &library,
            cub_indices.buffer(),
            rows * 4,
        )?);
        assert_eq!(cub_indices, native_indices);
        let native_scores = f32s_from_le_bytes(&read_device_bytes(
            &library,
            native_scores.buffer(),
            rows * 4,
        )?);
        let cub_scores =
            f32s_from_le_bytes(&read_device_bytes(&library, cub_scores.buffer(), rows * 4)?);
        assert_f32_close(&cub_scores, &native_scores, 1.0e-6, 1.0e-6);
        Ok(())
    }

    #[test]
    fn triton_dense_mlp_matches_native_synthetic_output() -> Result<()> {
        let Some(library) = load_validation_library()? else {
            return Ok(());
        };
        let stream = CudaStream::new(&library)?;
        let rows = 2;
        let hidden = 16;
        let intermediate = 8;
        let input_values = rows * hidden;
        let gate_values = intermediate * hidden;
        let intermediate_values = rows * intermediate;

        let mut input_values_f32 = vec![0.0_f32; input_values];
        input_values_f32[0] = 1.0;
        input_values_f32[hidden + 1] = -1.0;
        let input = input_values_f32
            .iter()
            .map(|value| f32_to_bf16(*value))
            .collect::<Vec<_>>();
        let mut gate_values_f32 = vec![0.0_f32; gate_values];
        gate_values_f32[0] = 0.5;
        gate_values_f32[hidden + 1] = -0.25;
        let gate = gate_values_f32
            .iter()
            .map(|value| f32_to_bf16(*value))
            .collect::<Vec<_>>();
        let mut up_values_f32 = vec![0.0_f32; gate_values];
        up_values_f32[0] = 1.0;
        up_values_f32[hidden + 1] = 0.75;
        let up = up_values_f32
            .iter()
            .map(|value| f32_to_bf16(*value))
            .collect::<Vec<_>>();
        let mut down_values_f32 = vec![0.0_f32; hidden * intermediate];
        down_values_f32[0] = 1.0;
        down_values_f32[intermediate + 1] = -1.0;
        let down = down_values_f32
            .iter()
            .map(|value| f32_to_bf16(*value))
            .collect::<Vec<_>>();
        let input_buffer = upload(&library, u16_bytes(&input), "validation dense MLP input")?;
        let gate_buffer = upload(&library, u16_bytes(&gate), "validation dense MLP gate")?;
        let up_buffer = upload(&library, u16_bytes(&up), "validation dense MLP up")?;
        let down_buffer = upload(&library, u16_bytes(&down), "validation dense MLP down")?;
        let native_out = DeviceAllocation::new(&library, input_values * 2, "native MLP output")?;
        let triton_out = DeviceAllocation::new(&library, input_values * 2, "Triton MLP output")?;
        let gate_output =
            DeviceAllocation::new(&library, intermediate_values * 4, "Triton MLP gate output")?;
        let up_output =
            DeviceAllocation::new(&library, intermediate_values * 4, "Triton MLP up output")?;
        let activation =
            DeviceAllocation::new(&library, intermediate_values * 2, "Triton MLP activation")?;

        unsafe {
            library.cuda_silu_gated_mlp_rows_bf16_async(
                input_buffer.buffer(),
                gate_buffer.buffer(),
                up_buffer.buffer(),
                down_buffer.buffer(),
                native_out.buffer(),
                rows,
                hidden,
                intermediate,
                stream.raw(),
            )?;
            stream.synchronize()?;
        }

        let buffers = [
            PythonDeviceBufferArg {
                name: "input",
                ptr: input_buffer.buffer().ptr,
                bytes: input_buffer.buffer().bytes,
                device_id: input_buffer.buffer().device_id,
                flags: input_buffer.buffer().flags,
            },
            PythonDeviceBufferArg {
                name: "gate_weight",
                ptr: gate_buffer.buffer().ptr,
                bytes: gate_buffer.buffer().bytes,
                device_id: gate_buffer.buffer().device_id,
                flags: gate_buffer.buffer().flags,
            },
            PythonDeviceBufferArg {
                name: "up_weight",
                ptr: up_buffer.buffer().ptr,
                bytes: up_buffer.buffer().bytes,
                device_id: up_buffer.buffer().device_id,
                flags: up_buffer.buffer().flags,
            },
            PythonDeviceBufferArg {
                name: "down_weight",
                ptr: down_buffer.buffer().ptr,
                bytes: down_buffer.buffer().bytes,
                device_id: down_buffer.buffer().device_id,
                flags: down_buffer.buffer().flags,
            },
            PythonDeviceBufferArg {
                name: "gate_output",
                ptr: gate_output.buffer().ptr,
                bytes: gate_output.buffer().bytes,
                device_id: gate_output.buffer().device_id,
                flags: gate_output.buffer().flags,
            },
            PythonDeviceBufferArg {
                name: "up_output",
                ptr: up_output.buffer().ptr,
                bytes: up_output.buffer().bytes,
                device_id: up_output.buffer().device_id,
                flags: up_output.buffer().flags,
            },
            PythonDeviceBufferArg {
                name: "activation",
                ptr: activation.buffer().ptr,
                bytes: activation.buffer().bytes,
                device_id: activation.buffer().device_id,
                flags: activation.buffer().flags,
            },
            PythonDeviceBufferArg {
                name: "output",
                ptr: triton_out.buffer().ptr,
                bytes: triton_out.buffer().bytes,
                device_id: triton_out.buffer().device_id,
                flags: triton_out.buffer().flags,
            },
        ];
        let kwargs = [
            ("rows", PythonKernelArg::Usize(rows)),
            ("hidden", PythonKernelArg::Usize(hidden)),
            ("intermediate", PythonKernelArg::Usize(intermediate)),
            ("down_stride", PythonKernelArg::Usize(intermediate)),
        ];
        launch_python_graph_capture(PythonGraphCaptureLaunch {
            module: "triton_mlp_capture",
            function: "capture_dense_mlp",
            cuda_stream: stream.raw(),
            buffers: &buffers,
            kwargs: &kwargs,
        })?;
        unsafe {
            stream.synchronize()?;
        }

        let native = bf16_bytes_to_f32(&read_device_bytes(
            &library,
            native_out.buffer(),
            input_values * 2,
        )?);
        let triton = bf16_bytes_to_f32(&read_device_bytes(
            &library,
            triton_out.buffer(),
            input_values * 2,
        )?);
        assert_f32_close(&triton, &native, 1.0e-3, 1.0e-3);
        Ok(())
    }

    #[test]
    fn triton_router_topk_matches_native_synthetic_indices() -> Result<()> {
        let Some(library) = load_validation_library()? else {
            return Ok(());
        };
        let stream = CudaStream::new(&library)?;
        let rows = 3;
        let hidden = 16;
        let experts = 8;
        let top_k = 4;
        let topk_values = rows * top_k;
        let hidden_bf16 = bf16_pattern(rows * hidden);
        let router_weight = bf16_replay_pattern(experts * hidden);
        let correction_bias = (0..experts)
            .map(|idx| ((idx % 5) as f32 - 2.0) * 0.125)
            .collect::<Vec<_>>();
        let hidden_buffer = upload(&library, u16_bytes(&hidden_bf16), "router hidden")?;
        let weight_buffer = upload(&library, u16_bytes(&router_weight), "router weight")?;
        let bias_buffer = upload(&library, f32_bytes(&correction_bias), "router bias")?;
        let native_indices =
            DeviceAllocation::new(&library, topk_values * 4, "native router indices")?;
        let native_scores =
            DeviceAllocation::new(&library, topk_values * 4, "native router scores")?;
        let native_weights =
            DeviceAllocation::new(&library, topk_values * 4, "native router weights")?;
        let triton_indices =
            DeviceAllocation::new(&library, topk_values * 4, "Triton router indices")?;
        let triton_scores =
            DeviceAllocation::new(&library, topk_values * 4, "Triton router scores")?;
        let triton_weights =
            DeviceAllocation::new(&library, topk_values * 4, "Triton router weights")?;
        let score_scratch =
            DeviceAllocation::new(&library, rows * experts * 4, "Triton router score scratch")?;

        unsafe {
            library.cuda_router_topk_bf16_async(
                hidden_buffer.buffer(),
                weight_buffer.buffer(),
                bias_buffer.buffer(),
                native_indices.buffer(),
                native_scores.buffer(),
                native_weights.buffer(),
                rows,
                hidden,
                experts,
                top_k,
                stream.raw(),
            )?;
            stream.synchronize()?;
        }

        let buffers = [
            PythonDeviceBufferArg {
                name: "hidden",
                ptr: hidden_buffer.buffer().ptr,
                bytes: hidden_buffer.buffer().bytes,
                device_id: hidden_buffer.buffer().device_id,
                flags: hidden_buffer.buffer().flags,
            },
            PythonDeviceBufferArg {
                name: "router_weight",
                ptr: weight_buffer.buffer().ptr,
                bytes: weight_buffer.buffer().bytes,
                device_id: weight_buffer.buffer().device_id,
                flags: weight_buffer.buffer().flags,
            },
            PythonDeviceBufferArg {
                name: "correction_bias",
                ptr: bias_buffer.buffer().ptr,
                bytes: bias_buffer.buffer().bytes,
                device_id: bias_buffer.buffer().device_id,
                flags: bias_buffer.buffer().flags,
            },
            PythonDeviceBufferArg {
                name: "score_scratch",
                ptr: score_scratch.buffer().ptr,
                bytes: score_scratch.buffer().bytes,
                device_id: score_scratch.buffer().device_id,
                flags: score_scratch.buffer().flags,
            },
            PythonDeviceBufferArg {
                name: "topk_indices",
                ptr: triton_indices.buffer().ptr,
                bytes: triton_indices.buffer().bytes,
                device_id: triton_indices.buffer().device_id,
                flags: triton_indices.buffer().flags,
            },
            PythonDeviceBufferArg {
                name: "topk_scores",
                ptr: triton_scores.buffer().ptr,
                bytes: triton_scores.buffer().bytes,
                device_id: triton_scores.buffer().device_id,
                flags: triton_scores.buffer().flags,
            },
            PythonDeviceBufferArg {
                name: "topk_weights",
                ptr: triton_weights.buffer().ptr,
                bytes: triton_weights.buffer().bytes,
                device_id: triton_weights.buffer().device_id,
                flags: triton_weights.buffer().flags,
            },
        ];
        let kwargs = [
            ("rows", PythonKernelArg::Usize(rows)),
            ("hidden_dim", PythonKernelArg::Usize(hidden)),
            ("experts", PythonKernelArg::Usize(experts)),
            ("top_k", PythonKernelArg::Usize(top_k)),
            (
                "routed_scaling_factor",
                PythonKernelArg::F64(GLM52_ROUTED_SCALING_FACTOR as f64),
            ),
        ];
        launch_python_graph_capture(PythonGraphCaptureLaunch {
            module: "triton_router_capture",
            function: "capture_router_topk",
            cuda_stream: stream.raw(),
            buffers: &buffers,
            kwargs: &kwargs,
        })?;
        unsafe {
            stream.synchronize()?;
        }

        let native_indices = u32s_from_le_bytes(&read_device_bytes(
            &library,
            native_indices.buffer(),
            topk_values * 4,
        )?);
        let triton_indices = u32s_from_le_bytes(&read_device_bytes(
            &library,
            triton_indices.buffer(),
            topk_values * 4,
        )?);
        assert_eq!(triton_indices, native_indices);
        let native_scores = f32s_from_le_bytes(&read_device_bytes(
            &library,
            native_scores.buffer(),
            topk_values * 4,
        )?);
        let triton_scores = f32s_from_le_bytes(&read_device_bytes(
            &library,
            triton_scores.buffer(),
            topk_values * 4,
        )?);
        assert_f32_close(&triton_scores, &native_scores, 1.0e-3, 2.0e-3);
        Ok(())
    }

    #[test]
    fn triton_lm_head_sampler_matches_native_synthetic_indices() -> Result<()> {
        let Some(library) = load_validation_library()? else {
            return Ok(());
        };
        let stream = CudaStream::new(&library)?;
        let rows = 2;
        let hidden = 16;
        let vocab = 32;
        let top_k = 8;
        let hidden_bf16 = bf16_pattern(rows * hidden);
        let lm_head = bf16_replay_pattern(vocab * hidden);
        let random_uniforms = [0.125_f32, 0.875_f32];
        let hidden_buffer = upload(&library, u16_bytes(&hidden_bf16), "sampler hidden")?;
        let lm_head_buffer = upload(&library, u16_bytes(&lm_head), "sampler lm_head")?;
        let random_buffer = upload(&library, f32_bytes(&random_uniforms), "sampler random")?;
        let native_indices = DeviceAllocation::new(&library, rows * 4, "native sampler indices")?;
        let native_scores = DeviceAllocation::new(&library, rows * 4, "native sampler scores")?;
        let triton_indices = DeviceAllocation::new(&library, rows * 4, "Triton sampler indices")?;
        let triton_scores = DeviceAllocation::new(&library, rows * 4, "Triton sampler scores")?;
        let logits = DeviceAllocation::new(&library, rows * vocab * 4, "Triton sampler logits")?;
        let candidate_values = rows * vocab.div_ceil(1024) * top_k;
        let candidate_scores =
            DeviceAllocation::new(&library, candidate_values * 4, "Triton candidate scores")?;
        let candidate_indices =
            DeviceAllocation::new(&library, candidate_values * 4, "Triton candidate indices")?;

        unsafe {
            library.cuda_lm_head_sample_topk_topp_bf16_async(
                hidden_buffer.buffer(),
                lm_head_buffer.buffer(),
                random_buffer.buffer(),
                native_indices.buffer(),
                native_scores.buffer(),
                rows,
                hidden,
                vocab,
                0.7,
                top_k,
                0.95,
                stream.raw(),
            )?;
            stream.synchronize()?;
        }

        let buffers = [
            PythonDeviceBufferArg {
                name: "hidden",
                ptr: hidden_buffer.buffer().ptr,
                bytes: hidden_buffer.buffer().bytes,
                device_id: hidden_buffer.buffer().device_id,
                flags: hidden_buffer.buffer().flags,
            },
            PythonDeviceBufferArg {
                name: "lm_head",
                ptr: lm_head_buffer.buffer().ptr,
                bytes: lm_head_buffer.buffer().bytes,
                device_id: lm_head_buffer.buffer().device_id,
                flags: lm_head_buffer.buffer().flags,
            },
            PythonDeviceBufferArg {
                name: "random_uniforms",
                ptr: random_buffer.buffer().ptr,
                bytes: random_buffer.buffer().bytes,
                device_id: random_buffer.buffer().device_id,
                flags: random_buffer.buffer().flags,
            },
            PythonDeviceBufferArg {
                name: "logits",
                ptr: logits.buffer().ptr,
                bytes: logits.buffer().bytes,
                device_id: logits.buffer().device_id,
                flags: logits.buffer().flags,
            },
            PythonDeviceBufferArg {
                name: "candidate_scores",
                ptr: candidate_scores.buffer().ptr,
                bytes: candidate_scores.buffer().bytes,
                device_id: candidate_scores.buffer().device_id,
                flags: candidate_scores.buffer().flags,
            },
            PythonDeviceBufferArg {
                name: "candidate_indices",
                ptr: candidate_indices.buffer().ptr,
                bytes: candidate_indices.buffer().bytes,
                device_id: candidate_indices.buffer().device_id,
                flags: candidate_indices.buffer().flags,
            },
            PythonDeviceBufferArg {
                name: "out_indices",
                ptr: triton_indices.buffer().ptr,
                bytes: triton_indices.buffer().bytes,
                device_id: triton_indices.buffer().device_id,
                flags: triton_indices.buffer().flags,
            },
            PythonDeviceBufferArg {
                name: "out_scores",
                ptr: triton_scores.buffer().ptr,
                bytes: triton_scores.buffer().bytes,
                device_id: triton_scores.buffer().device_id,
                flags: triton_scores.buffer().flags,
            },
        ];
        let kwargs = [
            ("rows", PythonKernelArg::Usize(rows)),
            ("hidden_dim", PythonKernelArg::Usize(hidden)),
            ("vocab", PythonKernelArg::Usize(vocab)),
            ("temperature", PythonKernelArg::F64(0.7)),
            ("top_k", PythonKernelArg::Usize(top_k)),
            ("top_p", PythonKernelArg::F64(0.95)),
        ];
        launch_python_graph_capture(PythonGraphCaptureLaunch {
            module: "triton_sampling_capture",
            function: "capture_lm_head_sample_topk_topp",
            cuda_stream: stream.raw(),
            buffers: &buffers,
            kwargs: &kwargs,
        })?;
        unsafe {
            stream.synchronize()?;
        }

        let native_indices = u32s_from_le_bytes(&read_device_bytes(
            &library,
            native_indices.buffer(),
            rows * 4,
        )?);
        let triton_indices = u32s_from_le_bytes(&read_device_bytes(
            &library,
            triton_indices.buffer(),
            rows * 4,
        )?);
        assert_eq!(triton_indices, native_indices);
        let native_scores = f32s_from_le_bytes(&read_device_bytes(
            &library,
            native_scores.buffer(),
            rows * 4,
        )?);
        let triton_scores = f32s_from_le_bytes(&read_device_bytes(
            &library,
            triton_scores.buffer(),
            rows * 4,
        )?);
        assert_f32_close(&triton_scores, &native_scores, 1.0e-3, 2.0e-3);
        Ok(())
    }

    #[test]
    fn triton_mla_kv_pack_matches_native_synthetic_values() -> Result<()> {
        let Some(library) = load_validation_library()? else {
            return Ok(());
        };
        let stream = CudaStream::new(&library)?;
        let rows = 3;
        let projected_values = rows * MLA_KV_FP8_PROJECTED_VALUES;
        let packed_bytes = rows * MLA_KV_FP8_PACKED_BYTES;
        let projected = bf16_pattern(projected_values);
        let projected_buffer = upload(&library, u16_bytes(&projected), "KV pack projected")?;
        let native_packed = DeviceAllocation::new(&library, packed_bytes, "native KV packed")?;
        let triton_packed = DeviceAllocation::new(&library, packed_bytes, "Triton KV packed")?;
        let native_unpacked =
            DeviceAllocation::new(&library, projected_values * 2, "native KV unpacked")?;
        let triton_unpacked =
            DeviceAllocation::new(&library, projected_values * 2, "Triton KV unpacked")?;

        unsafe {
            library.cuda_mla_kv_pack_fp8_ds_mla_async(
                projected_buffer.buffer(),
                native_packed.buffer(),
                rows,
                MLA_KV_FP8_PROJECTED_STRIDE_BYTES,
                MLA_KV_FP8_PACKED_BYTES,
                stream.raw(),
            )?;
            stream.synchronize()?;
        }

        let buffers = [
            PythonDeviceBufferArg {
                name: "projected",
                ptr: projected_buffer.buffer().ptr,
                bytes: projected_buffer.buffer().bytes,
                device_id: projected_buffer.buffer().device_id,
                flags: projected_buffer.buffer().flags,
            },
            PythonDeviceBufferArg {
                name: "packed",
                ptr: triton_packed.buffer().ptr,
                bytes: triton_packed.buffer().bytes,
                device_id: triton_packed.buffer().device_id,
                flags: triton_packed.buffer().flags,
            },
        ];
        let kwargs = [
            ("rows", PythonKernelArg::Usize(rows)),
            (
                "projected_stride_bytes",
                PythonKernelArg::Usize(MLA_KV_FP8_PROJECTED_STRIDE_BYTES),
            ),
            (
                "packed_stride_bytes",
                PythonKernelArg::Usize(MLA_KV_FP8_PACKED_BYTES),
            ),
        ];
        launch_python_graph_capture(PythonGraphCaptureLaunch {
            module: "triton_kv_pack_capture",
            function: "capture_mla_kv_pack_fp8_ds_mla",
            cuda_stream: stream.raw(),
            buffers: &buffers,
            kwargs: &kwargs,
        })?;
        unsafe {
            library.cuda_mla_kv_unpack_fp8_ds_mla_async(
                native_packed.buffer(),
                native_unpacked.buffer(),
                rows,
                MLA_KV_FP8_PACKED_BYTES,
                MLA_KV_FP8_PROJECTED_STRIDE_BYTES,
                stream.raw(),
            )?;
            library.cuda_mla_kv_unpack_fp8_ds_mla_async(
                triton_packed.buffer(),
                triton_unpacked.buffer(),
                rows,
                MLA_KV_FP8_PACKED_BYTES,
                MLA_KV_FP8_PROJECTED_STRIDE_BYTES,
                stream.raw(),
            )?;
            stream.synchronize()?;
        }

        let native = bf16_bytes_to_f32(&read_device_bytes(
            &library,
            native_unpacked.buffer(),
            projected_values * 2,
        )?);
        let triton = bf16_bytes_to_f32(&read_device_bytes(
            &library,
            triton_unpacked.buffer(),
            projected_values * 2,
        )?);
        assert_f32_close(&triton, &native, 1.0e-2, 3.0e-2);
        Ok(())
    }
}
