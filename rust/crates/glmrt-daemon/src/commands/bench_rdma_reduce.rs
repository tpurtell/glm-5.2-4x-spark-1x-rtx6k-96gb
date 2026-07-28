use crate::cli::BenchRdmaRingArgs;
use anyhow::{Context, Result};
use glmrt_ffi::{
    GlmrtDeviceBuffer, GlmrtRouteShardReductionBuffers, NativeLibrary,
    GLMRT_ROUTE_SHARD_LOCAL_BF16, GLMRT_ROUTE_SHARD_WIRE_BF16,
    GLMRT_ROUTE_SHARD_WIRE_FP8_E4M3_ROW_SCALED, GLMRT_ROUTE_SHARD_WIRE_NVFP4_E2M1_FP8_E4M3,
};
use glmrt_transport::{
    TcpTransportConfig, VerbsHostMappedRdmaRing, VerbsHostMappedRdmaRingConfig,
    VerbsHostMappedRdmaSlot,
};
use serde::Serialize;
use std::env;
use std::ffi::c_void;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const REDUCTION_HEADER_BYTES: usize = 64;
const REDUCTION_MAGIC: &[u8; 8] = b"GLMRRED1";
const REDUCTION_VERSION: u16 = 1;
const REDUCTION_TRIGGER_KIND: u16 = 1;
const REDUCTION_PARTIAL_KIND: u16 = 2;
const REDUCTION_ROOT_RANK: usize = 0;
const MAX_REDUCTION_PEERS: usize = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ReductionFrameHeader {
    kind: u16,
    sequence: u64,
    generation: u64,
    source_rank: usize,
    destination_rank: usize,
    rows: usize,
    row_width: usize,
    row_stride_bytes: usize,
    wire_dtype: u32,
    frame_bytes: usize,
    world_size: usize,
}

#[derive(Clone, Copy, Debug)]
struct ReductionCodec {
    wire_dtype: u32,
    row_stride_bytes: usize,
}

#[derive(Debug, Clone, Serialize)]
struct ReductionTimingSummary {
    samples: usize,
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
    max_us: f64,
}

#[derive(Debug, Clone, Serialize)]
struct ReductionQualityReport {
    exact: bool,
    relative_l2: f64,
    max_abs: f64,
    output_checksum: f64,
    expected_checksum: f64,
}

#[derive(Debug, Serialize)]
struct ReductionRootReport {
    benchmark: String,
    role: String,
    peers: Vec<String>,
    network_label: String,
    pre_firmware: bool,
    rank: usize,
    world_size: usize,
    wire_codec: String,
    rows: usize,
    row_width: usize,
    row_stride_bytes: usize,
    peer_payload_bytes: usize,
    response_frame_bytes: usize,
    slot_bytes: usize,
    depth: usize,
    validation_iterations: usize,
    warmup_iterations: usize,
    iterations: usize,
    elapsed_ms: f64,
    reductions_per_second: f64,
    peer_payload_gbps: f64,
    thread_cpu_ms: Option<f64>,
    thread_cpu_fraction: Option<f64>,
    fanout_post: ReductionTimingSummary,
    peer_wait: ReductionTimingSummary,
    reduce_kernel: ReductionTimingSummary,
    total: ReductionTimingSummary,
    kernel_only_us: f64,
    quality: ReductionQualityReport,
}

#[derive(Debug, Serialize)]
struct ReductionFollowerReport {
    benchmark: String,
    role: String,
    address: String,
    network_label: String,
    pre_firmware: bool,
    rank: usize,
    world_size: usize,
    wire_codec: String,
    rows: usize,
    row_width: usize,
    row_stride_bytes: usize,
    payload_bytes: usize,
    response_frame_bytes: usize,
    slot_bytes: usize,
    depth: usize,
    validation_iterations: usize,
    warmup_iterations: usize,
    iterations: usize,
    elapsed_ms: f64,
    responses_per_second: f64,
    thread_cpu_ms: Option<f64>,
    thread_cpu_fraction: Option<f64>,
    request_wait: ReductionTimingSummary,
    pack_gpu: ReductionTimingSummary,
    pack_and_sync: ReductionTimingSummary,
    response_post: ReductionTimingSummary,
}

#[derive(Default)]
struct RootMeasurements {
    fanout_post_us: Vec<f64>,
    peer_wait_us: Vec<f64>,
    reduce_kernel_us: Vec<f64>,
    total_us: Vec<f64>,
}

#[derive(Default)]
struct FollowerMeasurements {
    request_wait_us: Vec<f64>,
    pack_gpu_us: Vec<f64>,
    pack_and_sync_us: Vec<f64>,
    response_post_us: Vec<f64>,
}

struct ReductionEvents {
    start: *mut c_void,
    end: *mut c_void,
}

pub(super) fn run_reduction_root(
    args: &BenchRdmaRingArgs,
    transport: &TcpTransportConfig,
    config: VerbsHostMappedRdmaRingConfig,
) -> Result<()> {
    validate_reduction_args(args, true)?;
    let peers = parse_reduction_peers(args)?;
    let codec = reduction_codec(&args.wire_codec, args.row_width)?;
    let payload_bytes = reduction_payload_bytes(args.rows, codec.row_stride_bytes)?;
    let response_frame_bytes = REDUCTION_HEADER_BYTES
        .checked_add(payload_bytes)
        .context("reduction response frame byte count overflow")?;
    anyhow::ensure!(
        response_frame_bytes <= args.slot_bytes,
        "reduction response needs {response_frame_bytes} bytes, slot capacity is {}",
        args.slot_bytes
    );

    let mut rings = peers
        .iter()
        .map(|peer| VerbsHostMappedRdmaRing::connect(peer, transport, config))
        .collect::<Result<Vec<_>>>()?;
    let native = load_native_library(args.native_lib.as_deref())?;
    let values = args
        .rows
        .checked_mul(args.row_width)
        .context("reduction value count overflow")?;
    let local_values = reduction_fixture(REDUCTION_ROOT_RANK, args.rows, args.row_width);
    let expected = reduction_expected(args.reduction_world_size, args.rows, args.row_width);
    let mut local_f32 = native.alloc_device_buffer(
        values
            .checked_mul(std::mem::size_of::<f32>())
            .context("reduction local F32 byte count overflow")?,
    )?;
    let mut local_bf16 = native.alloc_device_buffer(
        values
            .checked_mul(std::mem::size_of::<u16>())
            .context("reduction local BF16 byte count overflow")?,
    )?;
    let mut output_f32 = native.alloc_device_buffer(
        values
            .checked_mul(std::mem::size_of::<f32>())
            .context("reduction output byte count overflow")?,
    )?;
    native.copy_h2d(local_f32, f32_bytes(&local_values))?;
    native.cuda_f32_to_bf16(local_f32, local_bf16, values)?;
    let events = ReductionEvents {
        start: native.cuda_event_create()?,
        end: native.cuda_event_create()?,
    };

    let operation = (|| -> Result<(RootMeasurements, f64, Duration, Option<f64>)> {
        let validation_iterations = args.depth;
        let mut kernel_only_us = None;
        run_root_phase(
            &native,
            &events,
            &mut rings,
            args,
            codec,
            local_bf16,
            output_f32,
            0,
            validation_iterations,
            true,
            false,
            &expected,
            &mut kernel_only_us,
            &mut RootMeasurements::default(),
        )?;
        run_root_phase(
            &native,
            &events,
            &mut rings,
            args,
            codec,
            local_bf16,
            output_f32,
            validation_iterations as u64,
            args.warmup_iterations,
            false,
            false,
            &expected,
            &mut kernel_only_us,
            &mut RootMeasurements::default(),
        )?;
        for ring in &mut rings {
            ring.flush_sends()?;
        }
        let measured_base = validation_iterations
            .checked_add(args.warmup_iterations)
            .context("reduction measured sequence overflow")? as u64;
        let mut measurements = RootMeasurements::default();
        let cpu_started = thread_cpu_time();
        let started = Instant::now();
        run_root_phase(
            &native,
            &events,
            &mut rings,
            args,
            codec,
            local_bf16,
            output_f32,
            measured_base,
            args.iterations,
            false,
            true,
            &expected,
            &mut kernel_only_us,
            &mut measurements,
        )?;
        for ring in &mut rings {
            ring.flush_sends()?;
        }
        let elapsed = started.elapsed();
        let cpu_ms = elapsed_thread_cpu_ms(cpu_started);
        Ok((
            measurements,
            kernel_only_us.context("reduction kernel-only timing was not captured")?,
            elapsed,
            cpu_ms,
        ))
    })();

    let quality = operation
        .as_ref()
        .ok()
        .map(|_| validate_reduction_output(&native, output_f32, &expected, &args.wire_codec))
        .transpose();
    unsafe {
        let _ = native.cuda_event_destroy(events.end);
        let _ = native.cuda_event_destroy(events.start);
    }
    let _ = native.free_device_buffer(&mut output_f32);
    let _ = native.free_device_buffer(&mut local_bf16);
    let _ = native.free_device_buffer(&mut local_f32);
    let (measurements, kernel_only_us, elapsed, thread_cpu_ms) = operation?;
    let quality = quality?
        .context("reduction quality validation was skipped after a successful operation")?;
    let elapsed_seconds = elapsed.as_secs_f64();
    let elapsed_ms = elapsed_seconds * 1e3;
    let report = ReductionRootReport {
        benchmark: "mapped-rdma-root-reduction".to_owned(),
        role: "reduce-root".to_owned(),
        peers,
        network_label: args.network_label.clone(),
        pre_firmware: args.network_label == "pre-firmware",
        rank: args.reduction_rank,
        world_size: args.reduction_world_size,
        wire_codec: args.wire_codec.clone(),
        rows: args.rows,
        row_width: args.row_width,
        row_stride_bytes: codec.row_stride_bytes,
        peer_payload_bytes: payload_bytes,
        response_frame_bytes,
        slot_bytes: args.slot_bytes,
        depth: args.depth,
        validation_iterations: args.depth,
        warmup_iterations: args.warmup_iterations,
        iterations: args.iterations,
        elapsed_ms,
        reductions_per_second: args.iterations as f64 / elapsed_seconds,
        peer_payload_gbps: args.iterations as f64 * payload_bytes as f64 * rings.len() as f64 * 8.0
            / elapsed_seconds
            / 1e9,
        thread_cpu_ms,
        thread_cpu_fraction: thread_cpu_ms.map(|cpu_ms| cpu_ms / elapsed_ms),
        fanout_post: summarize_us(measurements.fanout_post_us)?,
        peer_wait: summarize_us(measurements.peer_wait_us)?,
        reduce_kernel: summarize_us(measurements.reduce_kernel_us)?,
        total: summarize_us(measurements.total_us)?,
        kernel_only_us,
        quality,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

pub(super) fn run_reduction_follower(
    args: &BenchRdmaRingArgs,
    transport: &TcpTransportConfig,
) -> Result<()> {
    validate_reduction_args(args, false)?;
    let codec = reduction_codec(&args.wire_codec, args.row_width)?;
    let payload_bytes = reduction_payload_bytes(args.rows, codec.row_stride_bytes)?;
    let response_frame_bytes = REDUCTION_HEADER_BYTES
        .checked_add(payload_bytes)
        .context("reduction response frame byte count overflow")?;
    anyhow::ensure!(
        response_frame_bytes <= args.slot_bytes,
        "reduction response needs {response_frame_bytes} bytes, slot capacity is {}",
        args.slot_bytes
    );
    let listener = TcpListener::bind(&args.listen)
        .with_context(|| format!("binding reduction follower to {}", args.listen))?;
    let (stream, peer) = listener.accept().context("accepting reduction root")?;
    let mut ring = VerbsHostMappedRdmaRing::accept(stream, transport)?;
    let native = load_native_library(args.native_lib.as_deref())?;
    let values = args
        .rows
        .checked_mul(args.row_width)
        .context("reduction value count overflow")?;
    let source_values = reduction_fixture(args.reduction_rank, args.rows, args.row_width);
    let row_indices = (0..args.rows)
        .map(|row| u32::try_from(row).context("reduction row index exceeds u32"))
        .collect::<Result<Vec<_>>>()?;
    let mut source = native.alloc_device_buffer(
        values
            .checked_mul(std::mem::size_of::<f32>())
            .context("reduction source byte count overflow")?,
    )?;
    let mut indices = native.alloc_device_buffer(std::mem::size_of_val(row_indices.as_slice()))?;
    native.copy_h2d(source, f32_bytes(&source_values))?;
    native.copy_h2d(indices, u32_bytes(&row_indices))?;
    let events = ReductionEvents {
        start: native.cuda_event_create()?,
        end: native.cuda_event_create()?,
    };

    let operation = (|| -> Result<(FollowerMeasurements, Duration, Option<f64>)> {
        let validation_iterations = args.depth;
        let measured_start = validation_iterations
            .checked_add(args.warmup_iterations)
            .context("reduction follower measured sequence overflow")?;
        let total = measured_start
            .checked_add(args.iterations)
            .context("reduction follower total sequence overflow")?;
        let mut measurements = FollowerMeasurements::default();
        let mut measured_started = None;
        let mut cpu_started = None;
        for sequence in 0..total {
            let measured = sequence >= measured_start;
            if sequence == measured_start {
                measured_started = Some(Instant::now());
                cpu_started = thread_cpu_time();
            }
            let wait_started = Instant::now();
            let request = ring.wait_recv_slot()?;
            if measured {
                measurements
                    .request_wait_us
                    .push(wait_started.elapsed().as_secs_f64() * 1e6);
            }
            let request_frame = unsafe {
                std::slice::from_raw_parts(request.host_ptr.cast_const(), REDUCTION_HEADER_BYTES)
            };
            let header = decode_reduction_header(request_frame)?;
            validate_reduction_header(
                header,
                REDUCTION_TRIGGER_KIND,
                sequence as u64,
                args.depth,
                REDUCTION_ROOT_RANK,
                args.reduction_rank,
                args,
                codec,
                REDUCTION_HEADER_BYTES,
            )?;

            let send = ring.reserve_send_slot()?;
            anyhow::ensure!(
                send.sequence == sequence as u64,
                "reduction follower send sequence {} did not match {sequence}",
                send.sequence
            );
            encode_reduction_header(
                unsafe { std::slice::from_raw_parts_mut(send.host_ptr, REDUCTION_HEADER_BYTES) },
                ReductionFrameHeader {
                    kind: REDUCTION_PARTIAL_KIND,
                    sequence: send.sequence,
                    generation: send.sequence / args.depth as u64,
                    source_rank: args.reduction_rank,
                    destination_rank: REDUCTION_ROOT_RANK,
                    rows: args.rows,
                    row_width: args.row_width,
                    row_stride_bytes: codec.row_stride_bytes,
                    wire_dtype: codec.wire_dtype,
                    frame_bytes: response_frame_bytes,
                    world_size: args.reduction_world_size,
                },
            )?;
            let payload =
                device_buffer_slice(send.device_buffer, REDUCTION_HEADER_BYTES, payload_bytes)?;
            let pack_started = Instant::now();
            let pack_gpu_us = pack_reduction_payload(
                &native,
                &events,
                codec,
                source,
                indices,
                payload,
                args.rows,
                args.row_width,
            )?;
            if measured {
                measurements.pack_gpu_us.push(pack_gpu_us);
                measurements
                    .pack_and_sync_us
                    .push(pack_started.elapsed().as_secs_f64() * 1e6);
            }
            let post_started = Instant::now();
            ring.post_reserved_send(response_frame_bytes)?;
            ring.release_recv_slot(request.sequence)?;
            if measured {
                measurements
                    .response_post_us
                    .push(post_started.elapsed().as_secs_f64() * 1e6);
            }
            if sequence + 1 == validation_iterations || sequence + 1 == measured_start {
                ring.flush_sends()?;
            }
        }
        ring.flush_sends()?;
        Ok((
            measurements,
            measured_started
                .context("reduction follower never entered measured phase")?
                .elapsed(),
            elapsed_thread_cpu_ms(cpu_started),
        ))
    })();

    unsafe {
        let _ = native.cuda_event_destroy(events.end);
        let _ = native.cuda_event_destroy(events.start);
    }
    let _ = native.free_device_buffer(&mut indices);
    let _ = native.free_device_buffer(&mut source);
    let (measurements, elapsed, thread_cpu_ms) = operation?;
    let elapsed_seconds = elapsed.as_secs_f64();
    let elapsed_ms = elapsed_seconds * 1e3;
    let report = ReductionFollowerReport {
        benchmark: "mapped-rdma-root-reduction".to_owned(),
        role: "reduce-follower".to_owned(),
        address: peer.to_string(),
        network_label: args.network_label.clone(),
        pre_firmware: args.network_label == "pre-firmware",
        rank: args.reduction_rank,
        world_size: args.reduction_world_size,
        wire_codec: args.wire_codec.clone(),
        rows: args.rows,
        row_width: args.row_width,
        row_stride_bytes: codec.row_stride_bytes,
        payload_bytes,
        response_frame_bytes,
        slot_bytes: args.slot_bytes,
        depth: args.depth,
        validation_iterations: args.depth,
        warmup_iterations: args.warmup_iterations,
        iterations: args.iterations,
        elapsed_ms,
        responses_per_second: args.iterations as f64 / elapsed_seconds,
        thread_cpu_ms,
        thread_cpu_fraction: thread_cpu_ms.map(|cpu_ms| cpu_ms / elapsed_ms),
        request_wait: summarize_us(measurements.request_wait_us)?,
        pack_gpu: summarize_us(measurements.pack_gpu_us)?,
        pack_and_sync: summarize_us(measurements.pack_and_sync_us)?,
        response_post: summarize_us(measurements.response_post_us)?,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_root_phase(
    native: &NativeLibrary,
    events: &ReductionEvents,
    rings: &mut [VerbsHostMappedRdmaRing],
    args: &BenchRdmaRingArgs,
    codec: ReductionCodec,
    local_bf16: GlmrtDeviceBuffer,
    output_f32: GlmrtDeviceBuffer,
    sequence_base: u64,
    iterations: usize,
    validate_output: bool,
    measured: bool,
    expected: &[f32],
    kernel_only_us: &mut Option<f64>,
    measurements: &mut RootMeasurements,
) -> Result<()> {
    let payload_bytes = reduction_payload_bytes(args.rows, codec.row_stride_bytes)?;
    let response_frame_bytes = REDUCTION_HEADER_BYTES + payload_bytes;
    for iteration in 0..iterations {
        let total_started = Instant::now();
        let sequence = sequence_base.wrapping_add(iteration as u64);
        let fanout_started = Instant::now();
        for (peer_index, ring) in rings.iter_mut().enumerate() {
            let destination_rank = peer_index + 1;
            let slot = ring.reserve_send_slot()?;
            anyhow::ensure!(
                slot.sequence == sequence,
                "reduction root send sequence {} did not match {sequence}",
                slot.sequence
            );
            encode_reduction_header(
                unsafe { std::slice::from_raw_parts_mut(slot.host_ptr, REDUCTION_HEADER_BYTES) },
                ReductionFrameHeader {
                    kind: REDUCTION_TRIGGER_KIND,
                    sequence,
                    generation: sequence / args.depth as u64,
                    source_rank: REDUCTION_ROOT_RANK,
                    destination_rank,
                    rows: args.rows,
                    row_width: args.row_width,
                    row_stride_bytes: codec.row_stride_bytes,
                    wire_dtype: codec.wire_dtype,
                    frame_bytes: REDUCTION_HEADER_BYTES,
                    world_size: args.reduction_world_size,
                },
            )?;
            ring.post_reserved_send(REDUCTION_HEADER_BYTES)?;
        }
        if measured {
            measurements
                .fanout_post_us
                .push(fanout_started.elapsed().as_secs_f64() * 1e6);
        }

        let wait_started = Instant::now();
        let mut received = Vec::with_capacity(rings.len());
        for (peer_index, ring) in rings.iter_mut().enumerate() {
            let source_rank = peer_index + 1;
            let slot = ring.wait_recv_slot()?;
            let frame = unsafe {
                std::slice::from_raw_parts(slot.host_ptr.cast_const(), REDUCTION_HEADER_BYTES)
            };
            let header = decode_reduction_header(frame)?;
            validate_reduction_header(
                header,
                REDUCTION_PARTIAL_KIND,
                sequence,
                args.depth,
                source_rank,
                REDUCTION_ROOT_RANK,
                args,
                codec,
                response_frame_bytes,
            )?;
            received.push(slot);
        }
        if measured {
            measurements
                .peer_wait_us
                .push(wait_started.elapsed().as_secs_f64() * 1e6);
        }

        let buffers = reduction_buffers(local_bf16, output_f32, &received, payload_bytes)?;
        if kernel_only_us.is_none() {
            *kernel_only_us = Some(time_reduction_kernel(
                native,
                events,
                &buffers,
                args,
                codec,
                rings.len(),
            )?);
        }
        let kernel_us =
            launch_reduction_kernel(native, events, &buffers, args, codec, rings.len())?;
        if measured {
            measurements.reduce_kernel_us.push(kernel_us);
        }
        if validate_output {
            validate_reduction_output(native, output_f32, expected, &args.wire_codec)?;
        }
        for (ring, slot) in rings.iter_mut().zip(received) {
            ring.release_recv_slot(slot.sequence)?;
        }
        if measured {
            measurements
                .total_us
                .push(total_started.elapsed().as_secs_f64() * 1e6);
        }
    }
    Ok(())
}

fn reduction_buffers(
    local_bf16: GlmrtDeviceBuffer,
    output_f32: GlmrtDeviceBuffer,
    received: &[VerbsHostMappedRdmaSlot],
    payload_bytes: usize,
) -> Result<GlmrtRouteShardReductionBuffers> {
    anyhow::ensure!(
        !received.is_empty() && received.len() <= MAX_REDUCTION_PEERS,
        "reduction requires one to {MAX_REDUCTION_PEERS} peer slots"
    );
    let mut peers = [GlmrtDeviceBuffer::default(); MAX_REDUCTION_PEERS];
    for (peer, slot) in peers.iter_mut().zip(received) {
        *peer = device_buffer_slice(slot.device_buffer, REDUCTION_HEADER_BYTES, payload_bytes)?;
    }
    Ok(GlmrtRouteShardReductionBuffers {
        local: local_bf16,
        peers,
        output_f32,
    })
}

fn launch_reduction_kernel(
    native: &NativeLibrary,
    events: &ReductionEvents,
    buffers: &GlmrtRouteShardReductionBuffers,
    args: &BenchRdmaRingArgs,
    codec: ReductionCodec,
    peer_count: usize,
) -> Result<f64> {
    unsafe {
        native.cuda_event_record(events.start, std::ptr::null_mut())?;
        native.cuda_reduce_route_shards_to_f32_async(
            buffers,
            args.rows,
            args.row_width,
            codec.row_stride_bytes,
            GLMRT_ROUTE_SHARD_LOCAL_BF16,
            codec.wire_dtype,
            peer_count,
            std::ptr::null_mut(),
        )?;
        native.cuda_event_record(events.end, std::ptr::null_mut())?;
        native.cuda_event_synchronize(events.end)?;
        Ok(native.cuda_event_elapsed_ms(events.start, events.end)? as f64 * 1e3)
    }
}

fn time_reduction_kernel(
    native: &NativeLibrary,
    events: &ReductionEvents,
    buffers: &GlmrtRouteShardReductionBuffers,
    args: &BenchRdmaRingArgs,
    codec: ReductionCodec,
    peer_count: usize,
) -> Result<f64> {
    unsafe {
        native.cuda_event_record(events.start, std::ptr::null_mut())?;
        for _ in 0..args.kernel_iterations {
            native.cuda_reduce_route_shards_to_f32_async(
                buffers,
                args.rows,
                args.row_width,
                codec.row_stride_bytes,
                GLMRT_ROUTE_SHARD_LOCAL_BF16,
                codec.wire_dtype,
                peer_count,
                std::ptr::null_mut(),
            )?;
        }
        native.cuda_event_record(events.end, std::ptr::null_mut())?;
        native.cuda_event_synchronize(events.end)?;
        Ok(
            native.cuda_event_elapsed_ms(events.start, events.end)? as f64 * 1e3
                / args.kernel_iterations as f64,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn pack_reduction_payload(
    native: &NativeLibrary,
    events: &ReductionEvents,
    codec: ReductionCodec,
    source: GlmrtDeviceBuffer,
    indices: GlmrtDeviceBuffer,
    payload: GlmrtDeviceBuffer,
    rows: usize,
    row_width: usize,
) -> Result<f64> {
    let values = rows
        .checked_mul(row_width)
        .context("reduction pack value count overflow")?;
    unsafe {
        native.cuda_event_record(events.start, std::ptr::null_mut())?;
        match codec.wire_dtype {
            GLMRT_ROUTE_SHARD_WIRE_BF16 => {
                native.cuda_f32_to_bf16_async(source, payload, values, std::ptr::null_mut())?
            }
            GLMRT_ROUTE_SHARD_WIRE_FP8_E4M3_ROW_SCALED => native
                .cuda_gather_rows_f32_to_fp8_e4m3_row_scaled_async(
                    source,
                    rows,
                    indices,
                    payload,
                    rows,
                    row_width,
                    codec.row_stride_bytes,
                    std::ptr::null_mut(),
                )?,
            GLMRT_ROUTE_SHARD_WIRE_NVFP4_E2M1_FP8_E4M3 => native
                .cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3_async(
                    source,
                    rows,
                    indices,
                    payload,
                    rows,
                    row_width,
                    codec.row_stride_bytes,
                    std::ptr::null_mut(),
                )?,
            other => anyhow::bail!("unsupported reduction wire dtype {other}"),
        }
        native.cuda_event_record(events.end, std::ptr::null_mut())?;
        native.cuda_event_synchronize(events.end)?;
        Ok(native.cuda_event_elapsed_ms(events.start, events.end)? as f64 * 1e3)
    }
}

fn validate_reduction_args(args: &BenchRdmaRingArgs, root: bool) -> Result<()> {
    anyhow::ensure!(
        (2..=MAX_REDUCTION_PEERS + 1).contains(&args.reduction_world_size),
        "reduction world size must be in 2..={} ",
        MAX_REDUCTION_PEERS + 1
    );
    anyhow::ensure!(args.rows > 0, "reduction rows must be non-zero");
    anyhow::ensure!(args.row_width > 0, "reduction row width must be non-zero");
    anyhow::ensure!(
        args.kernel_iterations > 0,
        "reduction kernel iterations must be non-zero"
    );
    anyhow::ensure!(
        args.iterations >= args.depth.saturating_mul(2),
        "reduction iterations must cover at least two ring generations"
    );
    if root {
        anyhow::ensure!(
            args.reduction_rank == REDUCTION_ROOT_RANK,
            "reduction root rank must be {REDUCTION_ROOT_RANK}"
        );
    } else {
        anyhow::ensure!(
            args.reduction_rank > REDUCTION_ROOT_RANK
                && args.reduction_rank < args.reduction_world_size,
            "reduction follower rank must be in 1..{}",
            args.reduction_world_size
        );
    }
    Ok(())
}

fn parse_reduction_peers(args: &BenchRdmaRingArgs) -> Result<Vec<String>> {
    let peers = args
        .peers
        .as_deref()
        .context("reduction root requires --peers HOST:PORT,...")?
        .split(',')
        .map(str::trim)
        .filter(|peer| !peer.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    anyhow::ensure!(
        peers.len() + 1 == args.reduction_world_size,
        "reduction root has {} peers for world size {}",
        peers.len(),
        args.reduction_world_size
    );
    anyhow::ensure!(
        peers.len() <= MAX_REDUCTION_PEERS,
        "reduction root supports at most {MAX_REDUCTION_PEERS} peers"
    );
    Ok(peers)
}

fn reduction_codec(codec: &str, row_width: usize) -> Result<ReductionCodec> {
    match codec {
        "bf16" => Ok(ReductionCodec {
            wire_dtype: GLMRT_ROUTE_SHARD_WIRE_BF16,
            row_stride_bytes: row_width
                .checked_mul(std::mem::size_of::<u16>())
                .context("reduction BF16 row stride overflow")?,
        }),
        "fp8" => {
            anyhow::ensure!(
                row_width % std::mem::align_of::<f32>() == 0,
                "reduction FP8 row width must be FP32-aligned"
            );
            Ok(ReductionCodec {
                wire_dtype: GLMRT_ROUTE_SHARD_WIRE_FP8_E4M3_ROW_SCALED,
                row_stride_bytes: row_width
                    .checked_add(std::mem::size_of::<f32>())
                    .context("reduction FP8 row stride overflow")?,
            })
        }
        "nvfp4" => {
            anyhow::ensure!(
                row_width % 16 == 0,
                "reduction NVFP4 row width must be a multiple of 16"
            );
            Ok(ReductionCodec {
                wire_dtype: GLMRT_ROUTE_SHARD_WIRE_NVFP4_E2M1_FP8_E4M3,
                row_stride_bytes: row_width / 2 + row_width / 16,
            })
        }
        other => anyhow::bail!("unsupported reduction wire codec {other}"),
    }
}

fn reduction_payload_bytes(rows: usize, row_stride_bytes: usize) -> Result<usize> {
    rows.checked_mul(row_stride_bytes)
        .context("reduction payload byte count overflow")
}

fn reduction_fixture(rank: usize, rows: usize, row_width: usize) -> Vec<f32> {
    (0..rows)
        .flat_map(|row| {
            (0..row_width).map(move |column| {
                let code = (rank * 37 + row * 17 + column * 13) % 127;
                (code as i32 - 63) as f32 / 16.0
            })
        })
        .collect()
}

fn reduction_expected(world_size: usize, rows: usize, row_width: usize) -> Vec<f32> {
    let mut expected = vec![0.0_f32; rows * row_width];
    for rank in 0..world_size {
        for (destination, value) in expected
            .iter_mut()
            .zip(reduction_fixture(rank, rows, row_width))
        {
            *destination += value;
        }
    }
    expected
}

fn validate_reduction_output(
    native: &NativeLibrary,
    output: GlmrtDeviceBuffer,
    expected: &[f32],
    codec: &str,
) -> Result<ReductionQualityReport> {
    let mut bytes = vec![0_u8; std::mem::size_of_val(expected)];
    native.copy_d2h(&mut bytes, output)?;
    let actual = bytes_to_f32(&bytes).collect::<Vec<_>>();
    let mut error_squared = 0.0_f64;
    let mut expected_squared = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut output_checksum = 0.0_f64;
    let mut expected_checksum = 0.0_f64;
    for (actual, expected) in actual.iter().zip(expected) {
        let error = f64::from(*actual) - f64::from(*expected);
        error_squared += error * error;
        expected_squared += f64::from(*expected) * f64::from(*expected);
        max_abs = max_abs.max(error.abs());
        output_checksum += f64::from(*actual);
        expected_checksum += f64::from(*expected);
    }
    let relative_l2 = (error_squared / expected_squared.max(f64::MIN_POSITIVE)).sqrt();
    let exact = actual == expected;
    match codec {
        "bf16" => anyhow::ensure!(
            exact,
            "BF16 root reduction was not exact: relative_l2={relative_l2:.6e} max_abs={max_abs:.6e}"
        ),
        "fp8" => anyhow::ensure!(
            relative_l2 <= 5.0e-2,
            "FP8 root reduction relative L2 {relative_l2:.6e} exceeded 5e-2"
        ),
        "nvfp4" => anyhow::ensure!(
            relative_l2 <= 3.0e-1,
            "NVFP4 root reduction relative L2 {relative_l2:.6e} exceeded 3e-1"
        ),
        _ => unreachable!("codec validated before reduction"),
    }
    Ok(ReductionQualityReport {
        exact,
        relative_l2,
        max_abs,
        output_checksum,
        expected_checksum,
    })
}

fn encode_reduction_header(frame: &mut [u8], header: ReductionFrameHeader) -> Result<()> {
    anyhow::ensure!(
        frame.len() >= REDUCTION_HEADER_BYTES,
        "reduction frame is smaller than its header"
    );
    frame[..REDUCTION_HEADER_BYTES].fill(0);
    frame[0..8].copy_from_slice(REDUCTION_MAGIC);
    write_u16(frame, 8, REDUCTION_VERSION);
    write_u16(frame, 10, header.kind);
    write_u32(frame, 12, REDUCTION_HEADER_BYTES as u32);
    write_u64(frame, 16, header.sequence);
    write_u64(frame, 24, header.generation);
    write_u32(frame, 32, u32_value(header.source_rank, "source rank")?);
    write_u32(
        frame,
        36,
        u32_value(header.destination_rank, "destination rank")?,
    );
    write_u32(frame, 40, u32_value(header.rows, "rows")?);
    write_u32(frame, 44, u32_value(header.row_width, "row width")?);
    write_u32(frame, 48, u32_value(header.row_stride_bytes, "row stride")?);
    write_u32(frame, 52, header.wire_dtype);
    write_u32(frame, 56, u32_value(header.frame_bytes, "frame bytes")?);
    write_u32(frame, 60, u32_value(header.world_size, "world size")?);
    Ok(())
}

fn decode_reduction_header(frame: &[u8]) -> Result<ReductionFrameHeader> {
    anyhow::ensure!(
        frame.len() >= REDUCTION_HEADER_BYTES,
        "reduction frame is smaller than its header"
    );
    anyhow::ensure!(&frame[0..8] == REDUCTION_MAGIC, "reduction magic mismatch");
    anyhow::ensure!(
        read_u16(frame, 8) == REDUCTION_VERSION,
        "reduction version mismatch"
    );
    anyhow::ensure!(
        read_u32(frame, 12) as usize == REDUCTION_HEADER_BYTES,
        "reduction header size mismatch"
    );
    Ok(ReductionFrameHeader {
        kind: read_u16(frame, 10),
        sequence: read_u64(frame, 16),
        generation: read_u64(frame, 24),
        source_rank: read_u32(frame, 32) as usize,
        destination_rank: read_u32(frame, 36) as usize,
        rows: read_u32(frame, 40) as usize,
        row_width: read_u32(frame, 44) as usize,
        row_stride_bytes: read_u32(frame, 48) as usize,
        wire_dtype: read_u32(frame, 52),
        frame_bytes: read_u32(frame, 56) as usize,
        world_size: read_u32(frame, 60) as usize,
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_reduction_header(
    header: ReductionFrameHeader,
    expected_kind: u16,
    expected_sequence: u64,
    depth: usize,
    expected_source_rank: usize,
    expected_destination_rank: usize,
    args: &BenchRdmaRingArgs,
    codec: ReductionCodec,
    expected_frame_bytes: usize,
) -> Result<()> {
    validate_reduction_identity(
        header,
        expected_kind,
        expected_sequence,
        depth,
        expected_source_rank,
        expected_destination_rank,
    )?;
    anyhow::ensure!(
        header.rows == args.rows
            && header.row_width == args.row_width
            && header.row_stride_bytes == codec.row_stride_bytes,
        "reduction frame shape mismatch"
    );
    anyhow::ensure!(
        header.wire_dtype == codec.wire_dtype,
        "reduction frame dtype mismatch"
    );
    anyhow::ensure!(
        header.frame_bytes == expected_frame_bytes,
        "reduction frame bytes {} did not match {expected_frame_bytes}",
        header.frame_bytes
    );
    anyhow::ensure!(
        header.world_size == args.reduction_world_size,
        "reduction world size {} did not match {}",
        header.world_size,
        args.reduction_world_size
    );
    Ok(())
}

fn validate_reduction_identity(
    header: ReductionFrameHeader,
    expected_kind: u16,
    expected_sequence: u64,
    depth: usize,
    expected_source_rank: usize,
    expected_destination_rank: usize,
) -> Result<()> {
    anyhow::ensure!(
        header.kind == expected_kind,
        "reduction frame kind mismatch"
    );
    anyhow::ensure!(
        header.sequence == expected_sequence,
        "reduction sequence {} did not match {expected_sequence}",
        header.sequence
    );
    anyhow::ensure!(
        header.generation == expected_sequence / depth as u64,
        "reduction generation {} was stale for sequence {expected_sequence}",
        header.generation
    );
    anyhow::ensure!(
        header.source_rank == expected_source_rank,
        "reduction source rank {} did not match {expected_source_rank}",
        header.source_rank
    );
    anyhow::ensure!(
        header.destination_rank == expected_destination_rank,
        "reduction destination rank {} did not match {expected_destination_rank}",
        header.destination_rank
    );
    Ok(())
}

fn device_buffer_slice(
    buffer: GlmrtDeviceBuffer,
    offset_bytes: usize,
    bytes: usize,
) -> Result<GlmrtDeviceBuffer> {
    let end = offset_bytes
        .checked_add(bytes)
        .context("reduction device buffer slice overflow")?;
    anyhow::ensure!(
        !buffer.ptr.is_null() && end <= buffer.bytes,
        "reduction device buffer slice [{offset_bytes}, {end}) exceeds {} bytes",
        buffer.bytes
    );
    Ok(GlmrtDeviceBuffer {
        ptr: unsafe { buffer.ptr.cast::<u8>().add(offset_bytes).cast() },
        bytes,
        device_id: buffer.device_id,
        flags: buffer.flags,
    })
}

fn load_native_library(path: Option<&Path>) -> Result<NativeLibrary> {
    let path = path
        .map(Path::to_path_buf)
        .or_else(|| env::var_os("GLMRT_NATIVE_LIB").map(PathBuf::from))
        .context("RDMA reduction requires --native-lib or GLMRT_NATIVE_LIB")?;
    unsafe { NativeLibrary::load(&path) }
        .with_context(|| format!("loading native library {}", path.display()))
}

fn summarize_us(mut values: Vec<f64>) -> Result<ReductionTimingSummary> {
    anyhow::ensure!(!values.is_empty(), "reduction timing sample set is empty");
    values.sort_by(f64::total_cmp);
    let percentile = |fraction: f64| {
        let index = ((values.len() - 1) as f64 * fraction).round() as usize;
        values[index]
    };
    Ok(ReductionTimingSummary {
        samples: values.len(),
        mean_us: values.iter().sum::<f64>() / values.len() as f64,
        p50_us: percentile(0.50),
        p95_us: percentile(0.95),
        p99_us: percentile(0.99),
        max_us: *values.last().expect("non-empty reduction samples"),
    })
}

fn u32_value(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).with_context(|| format!("reduction {label} exceeds u32"))
}

fn write_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().expect("u16 field"))
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("u32 field"))
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("u64 field"))
}

fn f32_bytes(values: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

fn u32_bytes(values: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

fn bytes_to_f32(bytes: &[u8]) -> impl Iterator<Item = f32> + '_ {
    bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("four-byte chunk")))
}

#[cfg(target_os = "linux")]
fn thread_cpu_time() -> Option<Duration> {
    #[repr(C)]
    struct Timespec {
        tv_sec: i64,
        tv_nsec: i64,
    }
    unsafe extern "C" {
        fn clock_gettime(clock_id: i32, timespec: *mut Timespec) -> i32;
    }
    const CLOCK_THREAD_CPUTIME_ID: i32 = 3;
    let mut value = Timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    if unsafe { clock_gettime(CLOCK_THREAD_CPUTIME_ID, &mut value) } != 0
        || value.tv_sec < 0
        || value.tv_nsec < 0
    {
        return None;
    }
    Some(Duration::new(value.tv_sec as u64, value.tv_nsec as u32))
}

#[cfg(not(target_os = "linux"))]
fn thread_cpu_time() -> Option<Duration> {
    None
}

fn elapsed_thread_cpu_ms(started: Option<Duration>) -> Option<f64> {
    let started = started?;
    Some(thread_cpu_time()?.checked_sub(started)?.as_secs_f64() * 1e3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduction_header_roundtrips_across_generations() -> Result<()> {
        let header = ReductionFrameHeader {
            kind: REDUCTION_PARTIAL_KIND,
            sequence: 19,
            generation: 4,
            source_rank: 2,
            destination_rank: 0,
            rows: 16,
            row_width: 6144,
            row_stride_bytes: 6148,
            wire_dtype: GLMRT_ROUTE_SHARD_WIRE_FP8_E4M3_ROW_SCALED,
            frame_bytes: REDUCTION_HEADER_BYTES + 16 * 6148,
            world_size: 3,
        };
        let mut bytes = [0_u8; REDUCTION_HEADER_BYTES];
        encode_reduction_header(&mut bytes, header)?;
        assert_eq!(decode_reduction_header(&bytes)?, header);
        Ok(())
    }

    #[test]
    fn reduction_generation_and_rank_ownership_are_explicit() -> Result<()> {
        let mut header = ReductionFrameHeader {
            kind: REDUCTION_PARTIAL_KIND,
            sequence: 9,
            generation: 1,
            source_rank: 1,
            destination_rank: 0,
            rows: 1,
            row_width: 16,
            row_stride_bytes: 32,
            wire_dtype: GLMRT_ROUTE_SHARD_WIRE_BF16,
            frame_bytes: REDUCTION_HEADER_BYTES + 32,
            world_size: 3,
        };
        let error = validate_reduction_identity(header, REDUCTION_PARTIAL_KIND, 9, 4, 1, 0)
            .expect_err("stale generation must be rejected");
        assert!(error.to_string().contains("generation"));

        header.generation = header.sequence / 4;
        let error = validate_reduction_identity(header, REDUCTION_PARTIAL_KIND, 9, 4, 2, 0)
            .expect_err("wrong source rank must be rejected");
        assert!(error.to_string().contains("source rank"));
        Ok(())
    }

    #[test]
    fn reduction_fixture_is_bf16_exact_and_rank_distinct() {
        let left = reduction_fixture(0, 2, 64);
        let right = reduction_fixture(1, 2, 64);
        assert_ne!(left, right);
        assert!(left
            .iter()
            .all(|value| value * 16.0 == (value * 16.0).round()));
        assert_eq!(reduction_expected(3, 2, 64).len(), 128);
    }

    #[test]
    fn reduction_index_storage_scales_with_rows() {
        let indices = (0_u32..256).collect::<Vec<_>>();
        assert_eq!(std::mem::size_of_val(indices.as_slice()), 256 * 4);
        assert_eq!(
            std::mem::size_of_val(&indices),
            3 * std::mem::size_of::<usize>()
        );
    }
}
