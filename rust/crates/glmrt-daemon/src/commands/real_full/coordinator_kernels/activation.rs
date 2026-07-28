use super::*;
use std::{cell::RefCell, collections::HashMap};

pub(in crate::commands::real_full) fn nvfp4_e2m1_fp8_e4m3_row_bytes(
    hidden_dim: usize,
) -> Result<usize> {
    anyhow::ensure!(
        hidden_dim > 0 && hidden_dim % 16 == 0,
        "NVFP4 hidden width must be a nonzero multiple of 16, got {hidden_dim}"
    );
    hidden_dim
        .checked_div(2)
        .and_then(|packed| packed.checked_add(hidden_dim / 16))
        .context("NVFP4 hidden exchange row byte count overflow")
}

pub(in crate::commands::real_full) struct PendingNvfp4RowPayload {
    staging: GlmrtHostBuffer,
    payload_bytes: usize,
    ready_event: Arc<CoordinatorCudaEvent>,
}

#[derive(Default)]
struct Nvfp4RowPayloadWorkspace {
    output: ReusableDeviceBuffer,
    staging: ReusableHostBuffer,
}

thread_local! {
    static NVFP4_ROW_PAYLOAD_WORKSPACES: RefCell<HashMap<usize, Nvfp4RowPayloadWorkspace>> =
        RefCell::new(HashMap::new());
}

pub(in crate::commands::real_full) fn begin_quantize_device_bf16_to_nvfp4_row_payload(
    input: &DeviceBf16Output,
) -> Result<PendingNvfp4RowPayload> {
    anyhow::ensure!(
        input.rows > 0 && input.values_per_row > 0,
        "NVFP4 hidden exchange requires a nonempty BF16 device input"
    );
    let row_bytes = nvfp4_e2m1_fp8_e4m3_row_bytes(input.values_per_row)?;
    let payload_bytes = input
        .rows
        .checked_mul(row_bytes)
        .context("NVFP4 hidden exchange payload byte count overflow")?;
    let graph_key =
        coord_sparse_a_graph_key_for_full_hidden_rows(input.rows, input.values_per_row)?
            .context("asynchronous NVFP4 hidden exchange requires a Coord-Sparse-A graph slot")?;
    let library = cuda_native_library()?;
    let (output, staging) = NVFP4_ROW_PAYLOAD_WORKSPACES.with(|workspaces| {
        let mut workspaces = workspaces.borrow_mut();
        let workspace = workspaces
            .entry(graph_key.row_bucket.row_capacity)
            .or_default();
        workspace.output.ensure_capacity(
            library,
            payload_bytes,
            "NVFP4 hidden exchange payload",
        )?;
        workspace.staging.ensure_capacity(
            library,
            payload_bytes,
            "NVFP4 hidden exchange payload",
        )?;
        Ok::<_, anyhow::Error>((workspace.output.buffer, workspace.staging.buffer))
    })?;
    with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        unsafe {
            input
                .wait_ready_on_stream(cuda_stream)
                .context("waiting for BF16 hidden rows before NVFP4 exchange quantization")?;
            library
                .cuda_b12x_quantize_bf16_nvfp4_row_payload_async(
                    input.buffer(),
                    output,
                    input.rows,
                    input.values_per_row,
                    cuda_stream,
                )
                .context("quantizing coordinator BF16 hidden rows for NVFP4 exchange")?;
            library
                .copy_d2h_host_buffer_async(staging, output, payload_bytes, cuda_stream)
                .context("reading coordinator NVFP4 hidden exchange payload")?;
        }
        let ready_event = slot.record_output_ready_event(library)?;
        Ok(PendingNvfp4RowPayload {
            staging,
            payload_bytes,
            ready_event,
        })
    })
}

pub(in crate::commands::real_full) fn finish_quantize_device_bf16_to_nvfp4_row_payload(
    pending: PendingNvfp4RowPayload,
    payload: &mut Vec<u8>,
) -> Result<()> {
    pending
        .ready_event
        .synchronize()
        .context("waiting for coordinator NVFP4 hidden exchange payload")?;
    payload.resize(pending.payload_bytes, 0);
    unsafe {
        std::ptr::copy_nonoverlapping(
            pending.staging.ptr.cast::<u8>(),
            payload.as_mut_ptr(),
            pending.payload_bytes,
        );
    }
    Ok(())
}
