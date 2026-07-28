use anyhow::{Context, Result};
use glmrt_core::{
    CoordinatorGraphInstancePlan, CoordinatorGraphKey, CoordinatorGraphShape, LayerId,
    LayerWaveMode, COORDINATOR_GRAPH_INSTANCE_COUNT, GLM52_FIRST_K_DENSE_REPLACE,
    GLM52_HIDDEN_BF16_BYTES, GLM52_HIDDEN_SIZE, GLM52_NUM_HIDDEN_LAYERS,
    GLM52_ROUTED_SCALING_FACTOR, GLM52_TOP_K,
};
use glmrt_ffi::{
    GlmrtB12xCoordinatorW4a16Buffers, GlmrtCudaGraphCaptureInfo, GlmrtDeviceBuffer,
    GlmrtHostBuffer, NativeLibrary, GLMRT_CUDA_ROUTER_TOPK_MAX_K, GLMRT_CUDA_SAMPLE_TOPK_MAX_K,
};
use std::cell::{BorrowMutError, Cell, RefCell, RefMut};
use std::collections::{HashMap, HashSet};
use std::env;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::slice;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::thread::ThreadId;

mod attention;
pub(in crate::commands::real_full) use attention::*;
mod activation;
pub(in crate::commands::real_full) use activation::*;
mod embedding;
pub(in crate::commands::real_full) use embedding::*;
mod graphs;
pub(in crate::commands::real_full) use graphs::*;
mod linear;
pub(in crate::commands::real_full) use linear::*;
mod mlp;
pub(in crate::commands::real_full) use mlp::*;
mod norm;
pub(in crate::commands::real_full) use norm::*;
mod resident;
pub(in crate::commands::real_full) use resident::*;
mod residual;
pub(in crate::commands::real_full) use residual::*;
mod router;
pub(in crate::commands::real_full) use router::*;
mod sampling;
pub(in crate::commands::real_full) use sampling::*;
mod workspace;
pub(in crate::commands::real_full) use workspace::*;

const GLM52_Q_LORA_RANK: usize = 2048;

const REAL_FULL_CUDA_REFERENCE_KERNELS_ENV: &str = "GLMRT_REAL_FULL_CUDA_REFERENCE_KERNELS";
const REAL_FULL_GRAPH_CAPTURE_TRACE_ENV: &str = "GLMRT_REAL_FULL_GRAPH_CAPTURE_TRACE";
const REAL_FULL_RETAIN_MTP_QUERY_PROJECTION_GRAPHS_ENV: &str =
    "GLMRT_REAL_FULL_RETAIN_MTP_QUERY_PROJECTION_GRAPHS";

fn graph_capture_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(REAL_FULL_GRAPH_CAPTURE_TRACE_ENV)
            .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
            .unwrap_or(false)
    })
}

fn retain_mtp_query_projection_graphs_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(REAL_FULL_RETAIN_MTP_QUERY_PROJECTION_GRAPHS_ENV)
            .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
            .unwrap_or(true)
    })
}

fn graph_program_retains_capture_identity(
    program: CoordinatorCudaGraphProgram,
    row_capacity: usize,
    retain_mtp_query_projection_graphs: bool,
) -> bool {
    row_capacity == 1
        || matches!(
            program,
            CoordinatorCudaGraphProgram::LayerGlmDsaSparseMlaPrefillFull
                | CoordinatorCudaGraphProgram::LayerGlmDsaSparseMlaPrefillShared
        )
        || (row_capacity == 16 && program == CoordinatorCudaGraphProgram::LayerLinearBf16)
        || (program == CoordinatorCudaGraphProgram::LayerMlaDecodeScalarQAQueryProjectionBf16
            && retain_mtp_query_projection_graphs)
}

fn graph_program_capture_identity_capacity(
    program: CoordinatorCudaGraphProgram,
    row_capacity: usize,
) -> usize {
    if matches!(
        program,
        CoordinatorCudaGraphProgram::LayerGlmDsaSparseMlaPrefillFull
            | CoordinatorCudaGraphProgram::LayerGlmDsaSparseMlaPrefillShared
    ) {
        GLM52_NUM_HIDDEN_LAYERS
    } else if row_capacity == 16
        && matches!(
            program,
            CoordinatorCudaGraphProgram::LayerLinearBf16
                | CoordinatorCudaGraphProgram::LayerMlaDecodeScalarQAQueryProjectionBf16
        )
    {
        MAX_MTP_LINEAR_CAPTURE_IDENTITIES_PER_PROGRAM_SIGNATURE
    } else {
        MAX_DECODE_CAPTURE_IDENTITIES_PER_PROGRAM_SIGNATURE
    }
}

#[derive(Debug)]
#[cfg(test)]
pub(in crate::commands::real_full) struct ResidualAddOutput {
    pub(in crate::commands::real_full) values: Vec<f32>,
    pub(in crate::commands::real_full) backend: &'static str,
}

#[derive(Debug)]
#[cfg(test)]
pub(in crate::commands::real_full) struct ResidualAddBf16BytesOutput {
    pub(in crate::commands::real_full) bytes: Vec<u8>,
    pub(in crate::commands::real_full) backend: &'static str,
}

#[derive(Debug)]
pub(in crate::commands::real_full) struct GatherRowsBf16Output {
    pub(in crate::commands::real_full) bytes: Vec<u8>,
    pub(in crate::commands::real_full) backend: &'static str,
}

#[derive(Debug)]
pub(in crate::commands::real_full) struct ScatterAddRowsBf16ToF32Output {
    pub(in crate::commands::real_full) values: Vec<f32>,
    pub(in crate::commands::real_full) backend: &'static str,
}

#[derive(Debug)]
pub(in crate::commands::real_full) struct RmsNormOutput {
    pub(in crate::commands::real_full) values: Vec<f32>,
    pub(in crate::commands::real_full) backend: &'static str,
}

#[derive(Debug)]
#[allow(dead_code)]
pub(in crate::commands::real_full) struct LayerNormAffineOutput {
    pub(in crate::commands::real_full) values: Vec<f32>,
    pub(in crate::commands::real_full) backend: &'static str,
}

#[derive(Debug)]
pub(in crate::commands::real_full) struct LinearOutput {
    pub(in crate::commands::real_full) values: Vec<f32>,
    pub(in crate::commands::real_full) backend: &'static str,
}

#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) struct DeviceBf16Output {
    buffer: OwnedCoordinatorDeviceBuffer,
    bytes: usize,
    pub(in crate::commands::real_full) rows: usize,
    pub(in crate::commands::real_full) values_per_row: usize,
    pub(in crate::commands::real_full) backend: &'static str,
}

#[cfg_attr(not(test), allow(dead_code))]
impl DeviceBf16Output {
    pub(in crate::commands::real_full) fn buffer(&self) -> GlmrtDeviceBuffer {
        self.buffer.buffer
    }

    pub(in crate::commands::real_full) fn set_ready_event(
        &mut self,
        event: Arc<CoordinatorCudaEvent>,
    ) {
        self.buffer.ready_event = Some(event);
    }

    pub(in crate::commands::real_full) fn ready_event(&self) -> Option<Arc<CoordinatorCudaEvent>> {
        self.buffer.ready_event.clone()
    }

    pub(in crate::commands::real_full) fn wait_ready_on_stream(
        &self,
        stream: *mut c_void,
    ) -> Result<()> {
        self.buffer.wait_ready_on_stream(stream)
    }

    fn synchronize_ready(&self) -> Result<()> {
        self.buffer.synchronize_ready()
    }

    pub(in crate::commands::real_full) fn copy_to_host_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        self.copy_to_host_bytes_into(&mut bytes)?;
        Ok(bytes)
    }

    pub(in crate::commands::real_full) fn copy_to_host_bytes_into(
        &self,
        bytes: &mut Vec<u8>,
    ) -> Result<()> {
        self.synchronize_ready()
            .context("waiting for coordinator BF16 device output before host copy")?;
        bytes.resize(self.bytes, 0);
        self.buffer
            .library
            .copy_d2h(bytes, self.buffer.buffer)
            .context("copying owned coordinator BF16 device output to host")?;
        Ok(())
    }

    pub(in crate::commands::real_full) fn copy_to_host_values(&self) -> Result<Vec<f32>> {
        Ok(bf16_values_to_f32(&self.copy_to_host_bytes()?))
    }

    pub(in crate::commands::real_full) fn overwrite_from_host_bytes(
        &mut self,
        byte_offset: usize,
        bytes: &[u8],
        label: &str,
    ) -> Result<()> {
        self.synchronize_ready()
            .with_context(|| format!("waiting for {label} before BF16 row overwrite"))?;
        let view = device_buffer_byte_view(self.buffer.buffer, byte_offset, bytes.len(), label)?;
        self.buffer
            .library
            .copy_h2d(view, bytes)
            .with_context(|| format!("copying BF16 row overwrite for {label}"))
    }

    pub(in crate::commands::real_full) fn into_prefix_shape(
        mut self,
        rows: usize,
        values_per_row: usize,
        backend: &'static str,
        label: &'static str,
    ) -> Result<Self> {
        let bytes = rows
            .checked_mul(values_per_row)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .with_context(|| format!("BF16 device output prefix shape for {label} overflows"))?;
        anyhow::ensure!(
            bytes > 0 && bytes <= self.buffer.buffer.bytes,
            "BF16 device output prefix shape for {label} needs {bytes} bytes from a {} byte allocation",
            self.buffer.buffer.bytes
        );
        self.bytes = bytes;
        self.rows = rows;
        self.values_per_row = values_per_row;
        self.backend = backend;
        Ok(self)
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) struct ResidualAddDeviceOutput {
    pub(in crate::commands::real_full) values: Vec<f32>,
    #[allow(dead_code)]
    pub(in crate::commands::real_full) output_bf16: Vec<u8>,
    pub(in crate::commands::real_full) device_output: DeviceBf16Output,
    pub(in crate::commands::real_full) backend: &'static str,
}

#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) struct SparseBScatterResidualAddOutput {
    pub(in crate::commands::real_full) values: Vec<f32>,
    pub(in crate::commands::real_full) delta_values: Vec<f32>,
    pub(in crate::commands::real_full) output_bf16: Vec<u8>,
    pub(in crate::commands::real_full) device_output: Option<DeviceBf16Output>,
    pub(in crate::commands::real_full) backend: &'static str,
}

#[derive(Debug)]
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) struct LinearResidualAddOutput {
    pub(in crate::commands::real_full) linear_values: Vec<f32>,
    pub(in crate::commands::real_full) residual_values: Vec<f32>,
    pub(in crate::commands::real_full) linear_backend: &'static str,
    pub(in crate::commands::real_full) residual_add_backend: &'static str,
}

#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) struct LinearResidualAddDeviceOutput {
    pub(in crate::commands::real_full) linear_values: Vec<f32>,
    pub(in crate::commands::real_full) residual_values: Vec<f32>,
    pub(in crate::commands::real_full) residual_device: DeviceBf16Output,
    pub(in crate::commands::real_full) linear_backend: &'static str,
    pub(in crate::commands::real_full) residual_add_backend: &'static str,
}

#[derive(Debug)]
pub(in crate::commands::real_full) struct SiluGatedMlpOutput {
    pub(in crate::commands::real_full) values: Vec<f32>,
    pub(in crate::commands::real_full) backend: &'static str,
}

#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) struct SiluGatedMlpDeviceOutput {
    pub(in crate::commands::real_full) values: Vec<f32>,
    pub(in crate::commands::real_full) device_output: DeviceBf16Output,
    pub(in crate::commands::real_full) backend: &'static str,
}

#[derive(Debug)]
pub(in crate::commands::real_full) struct RouterTopKOutput {
    pub(in crate::commands::real_full) indices: Vec<usize>,
    pub(in crate::commands::real_full) scores: Vec<f32>,
    pub(in crate::commands::real_full) weights: Vec<f32>,
    pub(in crate::commands::real_full) backend: &'static str,
}

#[derive(Debug)]
pub(in crate::commands::real_full) struct EmbeddingLookupOutput {
    pub(in crate::commands::real_full) values: Vec<f32>,
    pub(in crate::commands::real_full) backend: &'static str,
}

#[allow(dead_code)]
#[derive(Debug)]
pub(in crate::commands::real_full) struct LogitsArgmaxOutput {
    pub(in crate::commands::real_full) indices: Vec<usize>,
    pub(in crate::commands::real_full) scores: Vec<f32>,
    pub(in crate::commands::real_full) backend: &'static str,
}

#[derive(Debug)]
pub(in crate::commands::real_full) struct LogitsSampleTopKToppOutput {
    pub(in crate::commands::real_full) indices: Vec<usize>,
    pub(in crate::commands::real_full) scores: Vec<f32>,
    pub(in crate::commands::real_full) backend: &'static str,
}

#[derive(Debug)]
pub(in crate::commands::real_full) struct CausalAttentionOutput {
    pub(in crate::commands::real_full) values: Vec<f32>,
    pub(in crate::commands::real_full) backend: &'static str,
}

#[derive(Debug)]
pub(in crate::commands::real_full) struct RopeOutput {
    pub(in crate::commands::real_full) values: Vec<f32>,
    pub(in crate::commands::real_full) backend: &'static str,
}

#[derive(Debug)]
pub(in crate::commands::real_full) struct MlaRopeAttentionOutput {
    pub(in crate::commands::real_full) values: Vec<f32>,
    pub(in crate::commands::real_full) backend: &'static str,
}

#[allow(dead_code)]
#[cfg(test)]
fn residual_add_prefix(residual: &[f32], delta: &[f32]) -> Result<ResidualAddOutput> {
    if residual.len() != delta.len() {
        anyhow::bail!(
            "real full residual length mismatch: residual={} delta={}",
            residual.len(),
            delta.len()
        );
    }
    if cuda_reference_kernels_enabled() {
        return cuda_residual_add_prefix(residual, delta);
    }
    Ok(cpu_residual_add_prefix(residual, delta))
}

#[cfg(test)]
fn residual_add_prefix_bf16(residual_bf16: &[u8], delta_bf16: &[u8]) -> Result<ResidualAddOutput> {
    let output = residual_add_prefix_bf16_bytes(residual_bf16, delta_bf16)?;
    Ok(ResidualAddOutput {
        values: bf16_values_to_f32(&output.bytes),
        backend: output.backend,
    })
}

#[cfg(test)]
fn residual_add_prefix_bf16_bytes(
    residual_bf16: &[u8],
    delta_bf16: &[u8],
) -> Result<ResidualAddBf16BytesOutput> {
    let mut bytes = vec![0_u8; residual_bf16.len()];
    let backend = residual_add_prefix_bf16_bytes_into(residual_bf16, delta_bf16, &mut bytes)?;
    Ok(ResidualAddBf16BytesOutput { bytes, backend })
}

fn cuda_reference_kernels_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = CUDA_REFERENCE_KERNELS_TEST_OVERRIDE.with(|value| value.get()) {
        return enabled;
    }

    env::var(REAL_FULL_CUDA_REFERENCE_KERNELS_ENV)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "cuda" | "reference"
            )
        })
        .unwrap_or(false)
}

#[cfg(test)]
pub(in crate::commands::real_full) fn set_cuda_reference_kernels_test_override(
    enabled: Option<bool>,
) -> Option<bool> {
    CUDA_REFERENCE_KERNELS_TEST_OVERRIDE.with(|value| {
        let previous = value.get();
        value.set(enabled);
        previous
    })
}

#[cfg(test)]
pub(in crate::commands::real_full) struct CudaReferenceKernelsTestOverride {
    previous: Option<bool>,
}

#[cfg(test)]
impl Drop for CudaReferenceKernelsTestOverride {
    fn drop(&mut self) {
        set_cuda_reference_kernels_test_override(self.previous);
    }
}

#[cfg(test)]
pub(in crate::commands::real_full) fn cuda_reference_kernels_test_override(
    enabled: bool,
) -> CudaReferenceKernelsTestOverride {
    CudaReferenceKernelsTestOverride {
        previous: set_cuda_reference_kernels_test_override(Some(enabled)),
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::commands::real_full) struct CoordinatorCudaGraphStats {
    pub(in crate::commands::real_full) slots: usize,
    pub(in crate::commands::real_full) captured_graphs: usize,
    pub(in crate::commands::real_full) graph_captures: usize,
    pub(in crate::commands::real_full) graph_launches: usize,
    pub(in crate::commands::real_full) acquisitions: usize,
}

#[cfg(test)]
pub(in crate::commands::real_full) fn coordinator_cuda_graph_test_stats(
) -> Result<CoordinatorCudaGraphStats> {
    coordinator_cuda_graph_stats()
}

pub(in crate::commands::real_full) fn coordinator_cuda_reference_kernels_enabled() -> bool {
    cuda_reference_kernels_enabled()
}

struct SparseBBf16PartialLayout {
    row_indices: Vec<u32>,
    src_bytes: usize,
}

struct SparseBLowPrecisionPartialLayout {
    row_indices: Vec<u32>,
    row_counts: Vec<usize>,
    src_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
struct LinearResidentView {
    full_bytes: usize,
    offset_bytes: usize,
    view_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
struct LinearPaddedDeviceInputView {
    weight: LinearResidentView,
    padded_input_bytes: usize,
    active_row_bytes: usize,
    padded_row_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
struct MlpGateUpResidentView {
    full_bytes: usize,
    offset_bytes: usize,
    view_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
struct MlpGateUpDownResidentView {
    gate_up: MlpGateUpResidentView,
    down_full_bytes: usize,
    down_stride: usize,
}

#[derive(Debug, Clone, Copy)]
struct LmHeadResidentView {
    full_bytes: usize,
    offset_bytes: usize,
    view_bytes: usize,
}

#[allow(dead_code)]
#[cfg(test)]
fn cpu_residual_add_prefix(residual: &[f32], delta: &[f32]) -> ResidualAddOutput {
    ResidualAddOutput {
        values: residual
            .iter()
            .zip(delta.iter())
            .map(|(residual, delta)| residual + delta)
            .collect(),
        backend: CPU_REFERENCE_RESIDUAL_ADD_BACKEND,
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LmHeadDeviceInputReadbackScratchState {
    argmax_index_ptr: usize,
    argmax_index_capacity: usize,
    argmax_score_ptr: usize,
    argmax_score_capacity: usize,
    sample_index_ptr: usize,
    sample_index_capacity: usize,
    sample_score_ptr: usize,
    sample_score_capacity: usize,
}

#[cfg(test)]
fn lm_head_device_input_readback_scratch_state() -> LmHeadDeviceInputReadbackScratchState {
    COORDINATOR_LM_HEAD_READBACK_SCRATCH.with(|scratch| {
        let scratch = scratch.borrow();
        LmHeadDeviceInputReadbackScratchState {
            argmax_index_ptr: scratch.argmax_index.as_ptr() as usize,
            argmax_index_capacity: scratch.argmax_index.capacity(),
            argmax_score_ptr: scratch.argmax_score.as_ptr() as usize,
            argmax_score_capacity: scratch.argmax_score.capacity(),
            sample_index_ptr: scratch.sample_index.as_ptr() as usize,
            sample_index_capacity: scratch.sample_index.capacity(),
            sample_score_ptr: scratch.sample_score.as_ptr() as usize,
            sample_score_capacity: scratch.sample_score.capacity(),
        }
    })
}

#[allow(dead_code)]
#[cfg(test)]
fn cuda_residual_add_prefix(residual: &[f32], delta: &[f32]) -> Result<ResidualAddOutput> {
    let library = cuda_native_library()?;
    let bytes = std::mem::size_of_val(residual);
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let residual_buffer =
        workspace.buffer(library, CoordinatorCudaScratchSlot::A, bytes, "residual")?;
    let delta_buffer = workspace.buffer(library, CoordinatorCudaScratchSlot::B, bytes, "delta")?;
    let output_buffer =
        workspace.buffer(library, CoordinatorCudaScratchSlot::C, bytes, "output")?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            f32_bytes(residual),
            "residual",
        )
        .context("copying residual to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::B,
            f32_bytes(delta),
            "delta",
        )
        .context("copying delta to device")?;
    library
        .cuda_residual_add_f32(residual_buffer, delta_buffer, output_buffer, residual.len())
        .context("executing CUDA residual add")?;
    let mut out_bytes = vec![0_u8; bytes];
    library
        .copy_d2h(&mut out_bytes, output_buffer)
        .context("copying residual add output to host")?;

    Ok(ResidualAddOutput {
        values: f32_vec_from_bytes(&out_bytes)?,
        backend: CUDA_REFERENCE_RESIDUAL_ADD_BACKEND,
    })
}

#[derive(Clone, Copy)]
enum CoordinatorCudaScratchSlot {
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P,
    Q,
    R,
    S,
    T,
    U,
    V,
}

type CoordinatorCudaScratchSlotState = Option<(*mut c_void, usize)>;

impl CoordinatorCudaScratchSlot {
    fn index(self) -> usize {
        match self {
            Self::A => 0,
            Self::B => 1,
            Self::C => 2,
            Self::D => 3,
            Self::E => 4,
            Self::F => 5,
            Self::G => 6,
            Self::H => 7,
            Self::I => 8,
            Self::J => 9,
            Self::K => 10,
            Self::L => 11,
            Self::M => 12,
            Self::N => 13,
            Self::O => 14,
            Self::P => 15,
            Self::Q => 16,
            Self::R => 17,
            Self::S => 18,
            Self::T => 19,
            Self::U => 20,
            Self::V => 21,
        }
    }
}

#[derive(Default)]
struct CoordinatorCudaWorkspace {
    scratch: Vec<ReusableDeviceBuffer>,
    host_staging: Vec<ReusableHostBuffer>,
    stream: Option<CoordinatorCudaStream>,
}

impl CoordinatorCudaWorkspace {
    fn stream_ptr(&mut self, library: &'static NativeLibrary) -> Result<*mut c_void> {
        if self.stream.is_none() {
            self.stream = Some(CoordinatorCudaStream::create(library)?);
        }
        Ok(self
            .stream
            .as_ref()
            .expect("coordinator CUDA workspace stream initialized")
            .as_ptr())
    }

    fn scratch_slot_state(
        &self,
        slot: CoordinatorCudaScratchSlot,
    ) -> CoordinatorCudaScratchSlotState {
        self.scratch
            .get(slot.index())
            .filter(|buffer| !buffer.buffer.ptr.is_null())
            .map(|buffer| (buffer.buffer.ptr, buffer.capacity))
    }

    fn scratch_slot_states(
        &self,
    ) -> [CoordinatorCudaScratchSlotState; COORDINATOR_CUDA_SCRATCH_SLOT_COUNT] {
        COORDINATOR_CUDA_SCRATCH_SLOTS.map(|slot| self.scratch_slot_state(slot))
    }

    fn buffer(
        &mut self,
        library: &'static NativeLibrary,
        slot: CoordinatorCudaScratchSlot,
        bytes: usize,
        label: &'static str,
    ) -> Result<GlmrtDeviceBuffer> {
        let index = slot.index();
        if self.scratch.len() <= index {
            self.scratch
                .resize_with(index + 1, ReusableDeviceBuffer::default);
        }
        self.scratch[index].ensure_capacity(library, bytes, label)?;
        Ok(self.scratch[index].buffer)
    }

    fn copy_h2d_to_slot(
        &mut self,
        library: &'static NativeLibrary,
        slot: CoordinatorCudaScratchSlot,
        src: &[u8],
        label: &'static str,
    ) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        let index = slot.index();
        let dst = self
            .scratch
            .get(index)
            .map(|buffer| buffer.buffer)
            .with_context(|| format!("coordinator CUDA device buffer {label} is not allocated"))?;
        if dst.ptr.is_null() {
            anyhow::bail!("coordinator CUDA device buffer {label} is null");
        }
        if src.len() > dst.bytes {
            anyhow::bail!(
                "coordinator CUDA staged H2D byte count {} exceeds device buffer {label} bytes {}",
                src.len(),
                dst.bytes
            );
        }
        let staging = self.host_buffer(library, slot, src.len(), label)?;
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), staging.ptr.cast::<u8>(), src.len());
        }
        library.copy_host_buffer_h2d(dst, staging, src.len())
    }

    fn copy_h2d_to_slot_async(
        &mut self,
        library: &'static NativeLibrary,
        slot: CoordinatorCudaScratchSlot,
        src: &[u8],
        label: &'static str,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        let index = slot.index();
        let dst = self
            .scratch
            .get(index)
            .map(|buffer| buffer.buffer)
            .with_context(|| format!("coordinator CUDA device buffer {label} is not allocated"))?;
        if dst.ptr.is_null() {
            anyhow::bail!("coordinator CUDA device buffer {label} is null");
        }
        if src.len() > dst.bytes {
            anyhow::bail!(
                "coordinator CUDA async staged H2D byte count {} exceeds device buffer {label} bytes {}",
                src.len(),
                dst.bytes
            );
        }
        let staging = self.host_buffer(library, slot, src.len(), label)?;
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), staging.ptr.cast::<u8>(), src.len());
            library.copy_host_buffer_h2d_async(dst, staging, src.len(), cuda_stream)
        }
    }

    fn copy_h2d_segments_to_slot_async(
        &mut self,
        library: &'static NativeLibrary,
        slot: CoordinatorCudaScratchSlot,
        segments: &[impl AsRef<[u8]>],
        bytes: usize,
        label: &'static str,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let mut copied = 0_usize;
        if bytes == 0 {
            for segment in segments {
                copied = copied
                    .checked_add(segment.as_ref().len())
                    .with_context(|| {
                        format!("coordinator CUDA segmented H2D {label} byte count overflows usize")
                    })?;
            }
            if copied != 0 {
                anyhow::bail!(
                    "coordinator CUDA segmented H2D expected 0 bytes for {label}, got {copied}"
                );
            }
            return Ok(());
        }
        let index = slot.index();
        let dst = self
            .scratch
            .get(index)
            .map(|buffer| buffer.buffer)
            .with_context(|| format!("coordinator CUDA device buffer {label} is not allocated"))?;
        if dst.ptr.is_null() {
            anyhow::bail!("coordinator CUDA device buffer {label} is null");
        }
        if bytes > dst.bytes {
            anyhow::bail!(
                "coordinator CUDA segmented H2D byte count {bytes} exceeds device buffer {label} bytes {}",
                dst.bytes
            );
        }
        let staging = self.host_buffer(library, slot, bytes, label)?;
        unsafe {
            for segment in segments {
                let segment = segment.as_ref();
                let next = copied.checked_add(segment.len()).with_context(|| {
                    format!("coordinator CUDA segmented H2D {label} byte count overflows usize")
                })?;
                if next > bytes {
                    anyhow::bail!(
                        "coordinator CUDA segmented H2D copied {next} bytes for {label}, expected {bytes}"
                    );
                }
                std::ptr::copy_nonoverlapping(
                    segment.as_ptr(),
                    staging.ptr.cast::<u8>().add(copied),
                    segment.len(),
                );
                copied = next;
            }
            if copied != bytes {
                anyhow::bail!(
                    "coordinator CUDA segmented H2D copied {copied} bytes for {label}, expected {bytes}"
                );
            }
            library.copy_host_buffer_h2d_async(dst, staging, bytes, cuda_stream)
        }
    }

    fn host_buffer(
        &mut self,
        library: &'static NativeLibrary,
        slot: CoordinatorCudaScratchSlot,
        bytes: usize,
        label: &'static str,
    ) -> Result<GlmrtHostBuffer> {
        let index = slot.index();
        if self.host_staging.len() <= index {
            self.host_staging
                .resize_with(index + 1, ReusableHostBuffer::default);
        }
        self.host_staging[index].ensure_capacity(library, bytes, label)?;
        Ok(self.host_staging[index].buffer)
    }
}

#[derive(Default)]
struct CoordinatorCudaResidentWeights {
    resident_weights: HashMap<String, ResidentDeviceBuffer>,
    host_staging: ReusableHostBuffer,
    w4a16_quant_scratch: ReusableDeviceBuffer,
}

impl CoordinatorCudaResidentWeights {
    fn resident_weight_buffer(
        &mut self,
        library: &'static NativeLibrary,
        name: &str,
        src: &[u8],
        label: &'static str,
    ) -> Result<GlmrtDeviceBuffer> {
        let key = resident_weight_registry_key(name, src.len());
        let buffer = {
            let resident = self
                .resident_weights
                .entry(key.clone())
                .or_insert_with(ResidentDeviceBuffer::default);
            resident.ensure_capacity(library, name, src.len(), label)?
        };
        let needs_upload = self
            .resident_weights
            .get(&key)
            .map(|resident| !resident.uploaded)
            .unwrap_or(true);
        if needs_upload {
            let staging = self.host_buffer(library, src.len(), label)?;
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), staging.ptr.cast::<u8>(), src.len());
            }
            library
                .copy_host_buffer_h2d(buffer, staging, src.len())
                .with_context(|| format!("uploading resident CUDA weight {name}"))?;
            let resident = self
                .resident_weights
                .get_mut(&key)
                .with_context(|| format!("resident CUDA weight {name} disappeared after upload"))?;
            resident.uploaded = true;
            resident.upload_count += 1;
        }
        Ok(buffer)
    }

    fn resident_weight_buffer_from_host_staging(
        &mut self,
        library: &'static NativeLibrary,
        name: &str,
        bytes: usize,
        label: &'static str,
        fill_staging: impl FnOnce(&mut [u8]) -> Result<()>,
    ) -> Result<GlmrtDeviceBuffer> {
        let key = resident_weight_registry_key(name, bytes);
        let buffer = {
            let resident = self
                .resident_weights
                .entry(key.clone())
                .or_insert_with(ResidentDeviceBuffer::default);
            resident.ensure_capacity(library, name, bytes, label)?
        };
        let needs_upload = self
            .resident_weights
            .get(&key)
            .map(|resident| !resident.uploaded)
            .unwrap_or(true);
        if needs_upload {
            let staging = self.host_buffer(library, bytes, label)?;
            if staging.ptr.is_null() {
                anyhow::bail!("resident CUDA weight {name} pinned staging buffer is null");
            }
            if bytes > staging.bytes {
                anyhow::bail!(
                    "resident CUDA weight {name} pinned staging byte count {bytes} exceeds host buffer bytes {}",
                    staging.bytes
                );
            }
            let staging_slice =
                unsafe { slice::from_raw_parts_mut(staging.ptr.cast::<u8>(), bytes) };
            fill_staging(staging_slice)?;
            library
                .copy_host_buffer_h2d(buffer, staging, bytes)
                .with_context(|| format!("uploading resident CUDA weight {name}"))?;
            let resident = self
                .resident_weights
                .get_mut(&key)
                .with_context(|| format!("resident CUDA weight {name} disappeared after upload"))?;
            resident.uploaded = true;
            resident.upload_count += 1;
        }
        Ok(buffer)
    }

    fn preloaded_resident_weight_buffer(
        &self,
        name: &str,
        expected_bytes: usize,
    ) -> Result<GlmrtDeviceBuffer> {
        if expected_bytes == 0 {
            anyhow::bail!(
                "preloaded resident coordinator CUDA weight {name} requires non-zero bytes"
            );
        }
        let key = resident_weight_registry_key(name, expected_bytes);
        let resident = self
            .resident_weights
            .get(&key)
            .with_context(|| format!("resident coordinator CUDA weight {name} is not preloaded"))?;
        if !resident.uploaded {
            anyhow::bail!("resident coordinator CUDA weight {name} exists but is not uploaded");
        }
        if resident.bytes != expected_bytes {
            anyhow::bail!(
                "resident coordinator CUDA weight {name} byte length mismatch: resident={} expected={expected_bytes}",
                resident.bytes
            );
        }
        if resident.buffer.ptr.is_null() {
            anyhow::bail!("resident coordinator CUDA weight {name} has a null device pointer");
        }
        Ok(resident.buffer)
    }

    fn resident_weight_is_preloaded(&self, name: &str, expected_bytes: usize) -> bool {
        let key = resident_weight_registry_key(name, expected_bytes);
        self.resident_weights
            .get(&key)
            .map(|resident| {
                resident.uploaded
                    && resident.bytes == expected_bytes
                    && !resident.buffer.ptr.is_null()
            })
            .unwrap_or(false)
    }

    fn release_resident_weight(
        &mut self,
        library: &'static NativeLibrary,
        name: &str,
        expected_bytes: usize,
    ) -> Result<()> {
        let key = resident_weight_registry_key(name, expected_bytes);
        let mut resident = self
            .resident_weights
            .remove(&key)
            .with_context(|| format!("resident coordinator CUDA weight {name} is not allocated"))?;
        anyhow::ensure!(
            resident.bytes == expected_bytes,
            "resident coordinator CUDA weight {name} has {} bytes, expected {expected_bytes}",
            resident.bytes
        );
        if !resident.buffer.ptr.is_null() {
            library
                .free_device_buffer(&mut resident.buffer)
                .with_context(|| format!("freeing resident coordinator CUDA weight {name}"))?;
        }
        Ok(())
    }

    fn host_buffer(
        &mut self,
        library: &'static NativeLibrary,
        bytes: usize,
        label: &'static str,
    ) -> Result<GlmrtHostBuffer> {
        self.host_staging.ensure_capacity(library, bytes, label)?;
        Ok(self.host_staging.buffer)
    }
}

#[cfg(test)]
fn resident_weight_registry_key(name: &str, bytes: usize) -> String {
    format!("{name}#test-bytes={bytes}")
}

#[allow(dead_code)]
struct CoordinatorCudaStream {
    library: &'static NativeLibrary,
    raw: *mut c_void,
}

#[allow(dead_code)]
pub(in crate::commands::real_full) struct CoordinatorCudaEvent {
    library: &'static NativeLibrary,
    raw: *mut c_void,
}

// The event is recorded and waited only through CUDA. The owning graph slot
// serializes re-recording, while Arc keeps the native handle alive for a
// downstream consumer that has not enqueued its wait yet.
unsafe impl Send for CoordinatorCudaEvent {}
unsafe impl Sync for CoordinatorCudaEvent {}

impl CoordinatorCudaEvent {
    fn create(library: &'static NativeLibrary) -> Result<Self> {
        let raw = library
            .cuda_event_create()
            .context("creating coordinator CUDA ready event")?;
        if raw.is_null() {
            anyhow::bail!("coordinator CUDA ready event create returned null");
        }
        Ok(Self { library, raw })
    }

    fn record(&self, stream: *mut c_void) -> Result<()> {
        unsafe {
            self.library
                .cuda_event_record(self.raw, stream)
                .context("recording coordinator CUDA ready event")
        }
    }

    pub(in crate::commands::real_full) fn wait_on_stream(&self, stream: *mut c_void) -> Result<()> {
        unsafe {
            self.library
                .cuda_stream_wait_event(stream, self.raw)
                .context("waiting for coordinator CUDA ready event on consumer stream")
        }
    }

    fn synchronize(&self) -> Result<()> {
        unsafe {
            self.library
                .cuda_event_synchronize(self.raw)
                .context("synchronizing coordinator CUDA ready event")
        }
    }
}

impl Drop for CoordinatorCudaEvent {
    fn drop(&mut self) {
        if self.raw.is_null() {
            return;
        }
        let raw = self.raw;
        self.raw = std::ptr::null_mut();
        unsafe {
            let _ = self.library.cuda_event_destroy(raw);
        }
    }
}

// Per-thread workspace ownership and its mutex serialize access to this opaque handle.
unsafe impl Send for CoordinatorCudaStream {}

#[allow(dead_code)]
impl CoordinatorCudaStream {
    fn create(library: &'static NativeLibrary) -> Result<Self> {
        let raw = library
            .cuda_stream_create()
            .context("creating coordinator CUDA graph stream")?;
        if raw.is_null() {
            anyhow::bail!("coordinator CUDA graph stream create returned a null stream");
        }
        Ok(Self { library, raw })
    }

    fn as_ptr(&self) -> *mut c_void {
        self.raw
    }

    fn synchronize(&self) -> Result<()> {
        unsafe {
            self.library
                .cuda_stream_synchronize(self.raw)
                .context("synchronizing coordinator CUDA graph stream")
        }
    }
}

impl Drop for CoordinatorCudaStream {
    fn drop(&mut self) {
        if self.raw.is_null() {
            return;
        }
        let raw = self.raw;
        self.raw = std::ptr::null_mut();
        unsafe {
            let _ = self.library.cuda_stream_destroy(raw);
        }
    }
}

struct CoordinatorCudaCapturedGraph {
    library: &'static NativeLibrary,
    graph_raw: *mut c_void,
    exec_raw: *mut c_void,
    node_count: usize,
    kernel_node_count: usize,
    memcpy_node_count: usize,
    memset_node_count: usize,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum CoordinatorCudaGraphProgram {
    #[cfg(test)]
    AdHocTest,
    CoordDenseEnvelopeBf16,
    CoordSparseAEnvelopeBf16,
    CoordSparseBEnvelopeBf16,
    CoordSparseBSharedResidualEnvelopeBf16,
    LayerCausalAttentionBf16,
    LayerDenseMlpBf16,
    LayerTritonDenseMlpBf16,
    LayerLinearBf16,
    LayerNormAffineBf16,
    LayerNormAffineF32Bf16,
    LayerPaddedLinearBf16,
    LayerPaddedLinearResidualAddBf16,
    LayerLinearResidualAddBf16,
    InputEmbeddingLookupBf16,
    TerminalLmHeadArgmaxBf16,
    TerminalLmHeadSampleTopKToppBf16,
    TerminalTritonLmHeadSampleTopKToppBf16,
    LayerB12xMlaRopeAttentionBf16,
    LayerFlashinferCompressedMlaDecodeBf16Init,
    LayerFlashinferCompressedMlaDecodeBf16Merge,
    LayerGlmDsaSparseMlaPrefillFull,
    LayerGlmDsaSparseMlaPrefillShared,
    LayerFlashinferPackedFp8MlaDecode,
    LayerFlashinferMlaRopeAttentionBf16,
    LayerFlashinferMlaRopeAttentionBf16Suffix,
    LayerFlashinferCudnnMlaRopeAttentionBf16Suffix,
    LayerMlaDecodeKvCommitBf16,
    LayerMlaDecodeQueryProjectionBf16,
    LayerMlaDecodeScalarQAQueryProjectionBf16,
    LayerMlaQuerySplitRopeBf16,
    LayerMlaKvCacheUnpackBf16,
    LayerMlaKvProjectedSplitBf16,
    LayerMlaRopeAttentionBf16,
    LayerMlaRopeAttentionBf16Suffix,
    LayerRmsNormBf16,
    LayerRopeBf16,
    SparseARouterTopKBf16,
    SparseATritonRouterTopKBf16,
    SparseBResidualAddBf16,
    SparseBScatterAddBf16ToF32,
}

#[cfg(test)]
mod capture_identity_policy_tests {
    use super::{
        graph_program_capture_identity_capacity, graph_program_retains_capture_identity,
        CoordinatorCudaGraphProgram, GLM52_NUM_HIDDEN_LAYERS,
        MAX_DECODE_CAPTURE_IDENTITIES_PER_PROGRAM_SIGNATURE,
        MAX_MTP_LINEAR_CAPTURE_IDENTITIES_PER_PROGRAM_SIGNATURE,
    };

    #[test]
    fn multirow_mtp_query_graphs_retain_bounded_layer_identities() {
        assert!(graph_program_retains_capture_identity(
            CoordinatorCudaGraphProgram::LayerMlaDecodeScalarQAQueryProjectionBf16,
            16,
            true,
        ));
        assert!(!graph_program_retains_capture_identity(
            CoordinatorCudaGraphProgram::LayerMlaDecodeScalarQAQueryProjectionBf16,
            16,
            false,
        ));
        assert!(graph_program_retains_capture_identity(
            CoordinatorCudaGraphProgram::LayerMlaDecodeQueryProjectionBf16,
            1,
            false,
        ));
        assert!(!graph_program_retains_capture_identity(
            CoordinatorCudaGraphProgram::LayerMlaDecodeQueryProjectionBf16,
            16,
            true,
        ));
        assert!(graph_program_retains_capture_identity(
            CoordinatorCudaGraphProgram::LayerLinearBf16,
            16,
            false,
        ));
        assert!(!graph_program_retains_capture_identity(
            CoordinatorCudaGraphProgram::LayerLinearBf16,
            32,
            false,
        ));
    }

    #[test]
    fn mtp_linear_graphs_retain_the_complete_prewarmed_identity_set() {
        assert_eq!(
            graph_program_capture_identity_capacity(
                CoordinatorCudaGraphProgram::LayerLinearBf16,
                16,
            ),
            MAX_MTP_LINEAR_CAPTURE_IDENTITIES_PER_PROGRAM_SIGNATURE,
        );
        assert_eq!(
            graph_program_capture_identity_capacity(
                CoordinatorCudaGraphProgram::LayerMlaDecodeScalarQAQueryProjectionBf16,
                16,
            ),
            MAX_MTP_LINEAR_CAPTURE_IDENTITIES_PER_PROGRAM_SIGNATURE,
        );
        assert_eq!(
            graph_program_capture_identity_capacity(
                CoordinatorCudaGraphProgram::LayerLinearBf16,
                32,
            ),
            MAX_DECODE_CAPTURE_IDENTITIES_PER_PROGRAM_SIGNATURE,
        );
        assert_eq!(
            graph_program_capture_identity_capacity(
                CoordinatorCudaGraphProgram::LayerGlmDsaSparseMlaPrefillFull,
                16,
            ),
            GLM52_NUM_HIDDEN_LAYERS,
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct CoordinatorCudaGraphSignature {
    values: usize,
    rows: usize,
    row_width: usize,
    index_count: usize,
    aux_count: usize,
    aux_width: usize,
    scalar_bits: u32,
}

impl CoordinatorCudaGraphSignature {
    fn residual_add_bf16(bytes: usize) -> Self {
        let values = bytes / std::mem::size_of::<u16>();
        Self {
            values,
            rows: values / GLM52_HIDDEN_SIZE,
            row_width: GLM52_HIDDEN_SIZE,
            index_count: 0,
            aux_count: 0,
            aux_width: 0,
            scalar_bits: 0,
        }
    }

    fn router_topk_bf16(rows: usize, hidden_dim: usize, experts: usize, top_k: usize) -> Self {
        Self {
            values: rows * top_k,
            rows,
            row_width: hidden_dim,
            index_count: experts,
            aux_count: 0,
            aux_width: 0,
            scalar_bits: 0,
        }
    }

    fn triton_router_topk_bf16(
        rows: usize,
        hidden_dim: usize,
        experts: usize,
        top_k: usize,
        buffer_identity: usize,
    ) -> Self {
        Self {
            values: rows * top_k,
            rows,
            row_width: hidden_dim,
            index_count: experts,
            aux_count: top_k,
            aux_width: buffer_identity,
            scalar_bits: 0,
        }
    }

    fn linear_bf16(rows: usize, input_dim: usize, output_dim: usize, has_bias: bool) -> Self {
        Self {
            values: rows * output_dim,
            rows,
            row_width: input_dim,
            index_count: output_dim,
            aux_count: usize::from(has_bias),
            aux_width: 0,
            scalar_bits: 0,
        }
    }

    fn lm_head_argmax_bf16(rows: usize, hidden_dim: usize, vocab: usize) -> Self {
        Self {
            values: rows,
            rows,
            row_width: hidden_dim,
            index_count: vocab,
            aux_count: 0,
            aux_width: 0,
            scalar_bits: 0,
        }
    }

    fn embedding_lookup_bf16(rows: usize, vocab: usize, hidden_dim: usize) -> Self {
        Self {
            values: rows * hidden_dim,
            rows,
            row_width: hidden_dim,
            index_count: vocab,
            aux_count: 0,
            aux_width: 0,
            scalar_bits: 0,
        }
    }

    fn lm_head_sample_topk_topp_bf16(
        rows: usize,
        hidden_dim: usize,
        vocab: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> Self {
        Self {
            values: rows,
            rows,
            row_width: hidden_dim,
            index_count: vocab,
            aux_count: top_k,
            aux_width: top_p.to_bits() as usize,
            scalar_bits: temperature.to_bits(),
        }
    }

    fn triton_lm_head_sample_topk_topp_bf16(
        rows: usize,
        hidden_dim: usize,
        vocab: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        buffer_identity: usize,
    ) -> Self {
        Self {
            values: rows,
            rows,
            row_width: hidden_dim,
            index_count: vocab,
            aux_count: top_k,
            aux_width: buffer_identity,
            scalar_bits: temperature.to_bits() ^ top_p.to_bits().rotate_left(16),
        }
    }

    fn padded_linear_bf16(
        rows: usize,
        active_input_dim: usize,
        full_input_dim: usize,
        output_dim: usize,
        has_bias: bool,
    ) -> Self {
        Self {
            values: rows * output_dim,
            rows,
            row_width: full_input_dim,
            index_count: output_dim,
            aux_count: active_input_dim,
            aux_width: usize::from(has_bias),
            scalar_bits: 0,
        }
    }

    fn padded_linear_residual_add_bf16(
        rows: usize,
        active_input_dim: usize,
        full_input_dim: usize,
        output_dim: usize,
        has_bias: bool,
    ) -> Self {
        Self {
            values: rows * output_dim,
            rows,
            row_width: full_input_dim,
            index_count: output_dim,
            aux_count: active_input_dim,
            aux_width: usize::from(has_bias),
            scalar_bits: 0,
        }
    }

    fn silu_gated_mlp_rows_bf16_down_stride(
        rows: usize,
        hidden: usize,
        intermediate: usize,
        down_stride: usize,
    ) -> Self {
        Self {
            values: rows * hidden,
            rows,
            row_width: hidden,
            index_count: intermediate,
            aux_count: down_stride,
            aux_width: 0,
            scalar_bits: 0,
        }
    }

    fn triton_silu_gated_mlp_rows_bf16_down_stride(
        rows: usize,
        hidden: usize,
        intermediate: usize,
        down_stride: usize,
        buffer_identity: usize,
    ) -> Self {
        Self {
            values: rows * hidden,
            rows,
            row_width: hidden,
            index_count: intermediate,
            aux_count: down_stride,
            aux_width: buffer_identity,
            scalar_bits: 0,
        }
    }

    fn rmsnorm_bf16(rows: usize, hidden_dim: usize, eps: f32) -> Self {
        Self {
            values: rows * hidden_dim,
            rows,
            row_width: hidden_dim,
            index_count: 0,
            aux_count: 0,
            aux_width: 0,
            scalar_bits: eps.to_bits(),
        }
    }

    fn layernorm_affine(rows: usize, hidden_dim: usize, eps: f32) -> Self {
        Self {
            values: rows * hidden_dim,
            rows,
            row_width: hidden_dim,
            index_count: 0,
            aux_count: 0,
            aux_width: 0,
            scalar_bits: eps.to_bits(),
        }
    }

    fn rope_bf16(
        input_bytes: usize,
        rows: usize,
        heads: usize,
        rotary_dim: usize,
        theta: f32,
    ) -> Self {
        Self {
            values: input_bytes / std::mem::size_of::<u16>(),
            rows,
            row_width: heads,
            index_count: rotary_dim,
            aux_count: 0,
            aux_width: 0,
            scalar_bits: theta.to_bits(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn mla_rope_attention_bf16(
        value_bytes: usize,
        rows: usize,
        heads: usize,
        nope_dim: usize,
        rope_dim: usize,
        v_dim: usize,
        scale: f32,
    ) -> Self {
        Self {
            values: value_bytes / std::mem::size_of::<u16>(),
            rows,
            row_width: heads,
            index_count: nope_dim,
            aux_count: rope_dim,
            aux_width: v_dim,
            scalar_bits: scale.to_bits(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn mla_rope_attention_bf16_suffix(
        value_bytes: usize,
        rows: usize,
        query_rows: usize,
        heads: usize,
        nope_dim: usize,
        rope_dim: usize,
        v_dim: usize,
        scale: f32,
    ) -> Self {
        Self {
            values: value_bytes / std::mem::size_of::<u16>(),
            rows,
            row_width: heads,
            index_count: nope_dim,
            aux_count: rope_dim,
            aux_width: v_dim.wrapping_mul(1_048_583).wrapping_add(query_rows),
            scalar_bits: scale.to_bits(),
        }
    }

    fn mla_kv_cache_unpack_bf16(
        rows: usize,
        payload_stride_bytes: usize,
        kv_lora_rank: usize,
        rope_dim: usize,
        dsa_dim: usize,
    ) -> Self {
        Self {
            values: rows * (kv_lora_rank + rope_dim + dsa_dim),
            rows,
            row_width: payload_stride_bytes,
            index_count: kv_lora_rank,
            aux_count: rope_dim,
            aux_width: dsa_dim,
            scalar_bits: 0,
        }
    }

    fn flashinfer_compressed_mla_decode_bf16(
        rows: usize,
        heads: usize,
        kv_lora_rank: usize,
        rope_dim: usize,
        scale: f32,
    ) -> Self {
        Self {
            values: heads * kv_lora_rank,
            rows,
            row_width: heads,
            index_count: kv_lora_rank,
            aux_count: rope_dim,
            aux_width: 0,
            scalar_bits: scale.to_bits(),
        }
    }

    fn flashinfer_packed_fp8_mla_decode(
        bucket_rows: usize,
        query_rows: usize,
        heads: usize,
        kv_lora_rank: usize,
        rope_dim: usize,
        scale: f32,
        hidden_projection_mode: usize,
    ) -> Self {
        Self {
            values: query_rows * heads * kv_lora_rank,
            rows: bucket_rows,
            row_width: heads,
            index_count: kv_lora_rank,
            aux_count: rope_dim,
            aux_width: query_rows * 8 + hidden_projection_mode,
            scalar_bits: scale.to_bits(),
        }
    }

    fn glm_dsa_sparse_mla_prefill(
        query_rows: usize,
        heads: usize,
        kv_lora_rank: usize,
        rope_dim: usize,
        topk: usize,
        max_pages: usize,
        scale: f32,
        run_selector: bool,
    ) -> Self {
        Self {
            values: query_rows * heads * kv_lora_rank,
            rows: query_rows,
            row_width: heads,
            index_count: kv_lora_rank,
            aux_count: rope_dim,
            aux_width: topk
                .wrapping_mul(1_048_583)
                .wrapping_add(max_pages)
                .wrapping_mul(1_048_583)
                .wrapping_add(usize::from(run_selector)),
            scalar_bits: scale.to_bits(),
        }
    }

    fn mla_kv_projected_split_bf16(
        rows: usize,
        heads: usize,
        nope_dim: usize,
        v_dim: usize,
    ) -> Self {
        Self {
            values: rows * heads * (nope_dim + v_dim),
            rows,
            row_width: heads,
            index_count: nope_dim,
            aux_count: v_dim,
            aux_width: 0,
            scalar_bits: 0,
        }
    }

    fn mla_decode_query_projection_bf16(
        rows: usize,
        hidden_dim: usize,
        q_lora_rank: usize,
        q_output_dim: usize,
        eps: f32,
        q_a_backend: usize,
        q_b_backend: usize,
        dsa_query_dim: usize,
        dsa_heads: usize,
    ) -> Self {
        Self {
            values: rows * hidden_dim,
            rows,
            row_width: hidden_dim,
            index_count: q_lora_rank,
            aux_count: q_output_dim,
            aux_width: q_a_backend
                .wrapping_mul(1_048_583)
                .wrapping_add(3 + q_b_backend)
                .wrapping_mul(1_048_583)
                .wrapping_add(dsa_query_dim)
                .wrapping_mul(1_048_583)
                .wrapping_add(dsa_heads),
            scalar_bits: eps.to_bits(),
        }
    }

    fn mla_decode_scalar_q_a_query_projection_bf16(
        rows: usize,
        hidden_dim: usize,
        q_lora_rank: usize,
        q_output_dim: usize,
        eps: f32,
        q_a_backend: usize,
        dsa_query_dim: usize,
        dsa_heads: usize,
    ) -> Self {
        Self {
            values: rows * hidden_dim,
            rows,
            row_width: hidden_dim,
            index_count: q_lora_rank,
            aux_count: q_output_dim,
            aux_width: q_a_backend
                .wrapping_mul(1_048_583)
                .wrapping_add(4)
                .wrapping_mul(1_048_583)
                .wrapping_add(dsa_query_dim)
                .wrapping_mul(1_048_583)
                .wrapping_add(dsa_heads),
            scalar_bits: eps.to_bits(),
        }
    }

    fn mla_decode_kv_commit_bf16(
        hidden_dim: usize,
        kv_lora_rank: usize,
        rope_dim: usize,
        dsa_dim: usize,
        cache_format: usize,
        eps: f32,
        theta: f32,
    ) -> Self {
        Self {
            values: hidden_dim,
            rows: 1,
            row_width: hidden_dim,
            index_count: kv_lora_rank,
            aux_count: rope_dim,
            aux_width: cache_format.wrapping_mul(257).wrapping_add(dsa_dim),
            scalar_bits: eps.to_bits() ^ theta.to_bits().rotate_left(13),
        }
    }

    fn mla_query_split_rope_bf16(
        rows: usize,
        heads: usize,
        nope_dim: usize,
        rope_dim: usize,
        theta: f32,
    ) -> Self {
        Self {
            values: rows * heads * (nope_dim + rope_dim),
            rows,
            row_width: heads,
            index_count: nope_dim,
            aux_count: rope_dim,
            aux_width: 2,
            scalar_bits: theta.to_bits(),
        }
    }

    fn causal_attention_bf16(
        value_bytes: usize,
        rows: usize,
        heads: usize,
        qk_dim: usize,
        v_dim: usize,
        scale: f32,
    ) -> Self {
        Self {
            values: value_bytes / std::mem::size_of::<u16>(),
            rows,
            row_width: heads,
            index_count: qk_dim,
            aux_count: v_dim,
            aux_width: 0,
            scalar_bits: scale.to_bits(),
        }
    }

    fn scatter_add_bf16_to_f32(dst_rows: usize, row_width: usize, index_count: usize) -> Self {
        Self {
            values: dst_rows * row_width,
            rows: dst_rows,
            row_width,
            index_count,
            aux_count: 0,
            aux_width: 0,
            scalar_bits: 0,
        }
    }

    fn coord_sparse_b_envelope_bf16(row_capacity: usize, row_width: usize) -> Self {
        Self {
            values: row_capacity * row_width,
            rows: row_capacity,
            row_width,
            index_count: 0,
            aux_count: GLM52_TOP_K,
            aux_width: 0,
            scalar_bits: 0,
        }
    }

    #[allow(dead_code)]
    fn coord_dense_envelope_bf16(rows: usize, hidden: usize, intermediate: usize) -> Self {
        Self {
            values: rows * hidden,
            rows,
            row_width: hidden,
            index_count: intermediate,
            aux_count: 14,
            aux_width: 0,
            scalar_bits: 0,
        }
    }

    #[allow(dead_code)]
    fn coord_sparse_a_envelope_bf16(
        rows: usize,
        hidden: usize,
        intermediate: usize,
        experts: usize,
        top_k: usize,
    ) -> Self {
        Self {
            values: rows * hidden,
            rows,
            row_width: hidden,
            index_count: intermediate,
            aux_count: experts,
            aux_width: top_k,
            scalar_bits: 0,
        }
    }

    #[cfg(test)]
    fn ad_hoc(values: usize, rows: usize, row_width: usize) -> Self {
        Self {
            values,
            rows,
            row_width,
            index_count: 0,
            aux_count: 0,
            aux_width: 0,
            scalar_bits: 0,
        }
    }
}

struct CoordinatorCudaCapturedGraphEntry {
    program: CoordinatorCudaGraphProgram,
    signature: CoordinatorCudaGraphSignature,
    capture_identity: usize,
    last_used: u64,
    graph: CoordinatorCudaCapturedGraph,
}

fn coordinator_cuda_captured_graph_entry_key(
    entry: &CoordinatorCudaCapturedGraphEntry,
) -> (
    CoordinatorCudaGraphProgram,
    CoordinatorCudaGraphSignature,
    usize,
) {
    (entry.program, entry.signature, entry.capture_identity)
}

fn least_recently_used_graph_index(
    entries: impl IntoIterator<Item = (usize, u64)>,
) -> Option<usize> {
    entries
        .into_iter()
        .min_by_key(|(_, last_used)| *last_used)
        .map(|(index, _)| index)
}

// Experimental C=8 can carry one pooled M=1 identity per lane across 78
// layers: 8 * 78 = 624. Retain that working set with a small margin.
const MAX_DECODE_CAPTURE_IDENTITIES_PER_PROGRAM_SIGNATURE: usize = 640;
// Experimental C=8 target verification uses two stable buffer rotations per
// lane across 78 layers: 2 * 8 * 78 = 1,248 retained M=2..16 Q/linear
// identities per signature. Retain the complete working set with a margin.
const MAX_MTP_LINEAR_CAPTURE_IDENTITIES_PER_PROGRAM_SIGNATURE: usize = 1280;

#[allow(dead_code)]
#[derive(Clone, Copy)]
struct CoordDenseEnvelopeBf16Buffers {
    input: GlmrtDeviceBuffer,
    norm0_weight: GlmrtDeviceBuffer,
    norm0_out: GlmrtDeviceBuffer,
    q_weight: GlmrtDeviceBuffer,
    q_out: GlmrtDeviceBuffer,
    k_weight: GlmrtDeviceBuffer,
    k_out: GlmrtDeviceBuffer,
    v_weight: GlmrtDeviceBuffer,
    v_out: GlmrtDeviceBuffer,
    attention_out: GlmrtDeviceBuffer,
    o_weight: GlmrtDeviceBuffer,
    attention_proj: GlmrtDeviceBuffer,
    attention_residual: GlmrtDeviceBuffer,
    norm1_weight: GlmrtDeviceBuffer,
    mlp_norm: GlmrtDeviceBuffer,
    probe_a_weight: GlmrtDeviceBuffer,
    probe_a_out: GlmrtDeviceBuffer,
    probe_b_weight: GlmrtDeviceBuffer,
    probe_b_out: GlmrtDeviceBuffer,
    probe_mix: GlmrtDeviceBuffer,
    gate_weight: GlmrtDeviceBuffer,
    up_weight: GlmrtDeviceBuffer,
    down_weight: GlmrtDeviceBuffer,
    mlp_out: GlmrtDeviceBuffer,
    mlp_delta: GlmrtDeviceBuffer,
    output: GlmrtDeviceBuffer,
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
struct CoordSparseAEnvelopeBf16Buffers {
    input: GlmrtDeviceBuffer,
    norm0_weight: GlmrtDeviceBuffer,
    norm0_out: GlmrtDeviceBuffer,
    q_nope_weight: GlmrtDeviceBuffer,
    q_nope_out: GlmrtDeviceBuffer,
    q_rope_weight: GlmrtDeviceBuffer,
    q_rope_out: GlmrtDeviceBuffer,
    k_nope_weight: GlmrtDeviceBuffer,
    k_nope_out: GlmrtDeviceBuffer,
    k_rope_weight: GlmrtDeviceBuffer,
    k_rope_out: GlmrtDeviceBuffer,
    value_weight: GlmrtDeviceBuffer,
    value_out: GlmrtDeviceBuffer,
    attention_out: GlmrtDeviceBuffer,
    o_weight: GlmrtDeviceBuffer,
    attention_proj: GlmrtDeviceBuffer,
    attention_residual: GlmrtDeviceBuffer,
    norm1_weight: GlmrtDeviceBuffer,
    moe_norm: GlmrtDeviceBuffer,
    gate_weight: GlmrtDeviceBuffer,
    up_weight: GlmrtDeviceBuffer,
    down_weight: GlmrtDeviceBuffer,
    shared_out: GlmrtDeviceBuffer,
    router_weight: GlmrtDeviceBuffer,
    correction_bias: GlmrtDeviceBuffer,
    topk_indices: GlmrtDeviceBuffer,
    topk_scores: GlmrtDeviceBuffer,
    topk_weights: GlmrtDeviceBuffer,
}

#[derive(Default)]
struct CoordinatorLmHeadReadbackScratch {
    argmax_index: Vec<u8>,
    argmax_score: Vec<u8>,
    sample_index: Vec<u8>,
    sample_score: Vec<u8>,
}

impl CoordinatorCudaCapturedGraph {
    fn new(library: &'static NativeLibrary, capture: GlmrtCudaGraphCaptureInfo) -> Result<Self> {
        if capture.graph.is_null() {
            anyhow::bail!("coordinator CUDA graph capture returned a null graph");
        }
        if capture.graph_exec.is_null() {
            anyhow::bail!("coordinator CUDA graph capture returned a null graph exec");
        }
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

    fn as_ptr(&self) -> *mut c_void {
        self.exec_raw
    }

    fn validate_before_launch(&self) -> Result<()> {
        if self.node_count == 0 {
            anyhow::bail!("coordinator CUDA captured graph has no nodes");
        }
        let classified_nodes = self
            .kernel_node_count
            .checked_add(self.memcpy_node_count)
            .and_then(|count| count.checked_add(self.memset_node_count))
            .context("coordinator CUDA captured graph node count overflows usize")?;
        if classified_nodes > self.node_count {
            anyhow::bail!(
                "coordinator CUDA captured graph classified node count {} exceeds total node count {}",
                classified_nodes,
                self.node_count
            );
        }
        Ok(())
    }

    fn update_exec_from_capture(&mut self, capture: GlmrtCudaGraphCaptureInfo) -> Result<()> {
        let mut next = CoordinatorCudaCapturedGraph::new(self.library, capture)?;

        let old_graph_raw = self.graph_raw;
        match unsafe {
            self.library
                .cuda_graph_exec_update(self.exec_raw, next.graph_raw)
        } {
            Ok(()) => {
                let temp_exec_raw = next.exec_raw;
                self.graph_raw = next.graph_raw;
                self.node_count = next.node_count;
                self.kernel_node_count = next.kernel_node_count;
                self.memcpy_node_count = next.memcpy_node_count;
                self.memset_node_count = next.memset_node_count;
                next.graph_raw = std::ptr::null_mut();
                next.exec_raw = std::ptr::null_mut();

                unsafe {
                    let _ = self.library.cuda_graph_exec_destroy(temp_exec_raw);
                    let _ = self.library.cuda_graph_destroy(old_graph_raw);
                }
            }
            Err(_) => {
                let old_exec_raw = self.exec_raw;
                self.graph_raw = next.graph_raw;
                self.exec_raw = next.exec_raw;
                self.node_count = next.node_count;
                self.kernel_node_count = next.kernel_node_count;
                self.memcpy_node_count = next.memcpy_node_count;
                self.memset_node_count = next.memset_node_count;
                next.graph_raw = std::ptr::null_mut();
                next.exec_raw = std::ptr::null_mut();

                unsafe {
                    let _ = self.library.cuda_graph_exec_destroy(old_exec_raw);
                    let _ = self.library.cuda_graph_destroy(old_graph_raw);
                }
            }
        }
        Ok(())
    }
}

impl Drop for CoordinatorCudaCapturedGraph {
    fn drop(&mut self) {
        let exec_raw = self.exec_raw;
        let graph_raw = self.graph_raw;
        self.exec_raw = std::ptr::null_mut();
        self.graph_raw = std::ptr::null_mut();
        unsafe {
            if !exec_raw.is_null() {
                let _ = self.library.cuda_graph_exec_destroy(exec_raw);
            }
            if !graph_raw.is_null() {
                let _ = self.library.cuda_graph_destroy(graph_raw);
            }
        }
    }
}

// CUDA streams and captured graph handles are opaque native resources owned by
// per-thread graph workspace slots. The runtime registry never shares a mutable
// stream/workspace/graph-exec tuple across host threads.

#[allow(dead_code)]
struct CoordinatorCudaGraphWorkspaceSlot {
    plan: CoordinatorGraphInstancePlan,
    stream: CoordinatorCudaStream,
    workspace: CoordinatorCudaWorkspace,
    captured_graphs: Vec<CoordinatorCudaCapturedGraphEntry>,
    acquisitions: usize,
    graph_captures: usize,
    graph_launches: usize,
    graph_use_clock: u64,
    output_ready_events: Vec<Arc<CoordinatorCudaEvent>>,
    stable_packed_fp8_mla_page_mapping: Option<(usize, u64)>,
}

#[allow(dead_code)]
impl CoordinatorCudaGraphWorkspaceSlot {
    fn next_graph_use(&mut self) -> u64 {
        self.graph_use_clock = self.graph_use_clock.saturating_add(1);
        self.graph_use_clock
    }

    fn insert_captured_graph(&mut self, entry: CoordinatorCudaCapturedGraphEntry) {
        let key = coordinator_cuda_captured_graph_entry_key(&entry);
        match self
            .captured_graphs
            .binary_search_by_key(&key, coordinator_cuda_captured_graph_entry_key)
        {
            Ok(index) => self.captured_graphs[index] = entry,
            Err(index) => self.captured_graphs.insert(index, entry),
        }
    }

    fn stream_ptr(&self) -> *mut c_void {
        self.stream.as_ptr()
    }

    fn stream_synchronize(&self) -> Result<()> {
        self.stream.synchronize()
    }

    fn record_output_ready_event(
        &mut self,
        library: &'static NativeLibrary,
    ) -> Result<Arc<CoordinatorCudaEvent>> {
        // A captured launch may hand several outputs to another stream before
        // any of them are consumed (notably sequential MTP target rows).  An
        // event cannot be re-recorded while an earlier output still uses that
        // event as its readiness fence, so reuse only slot-owned idle events.
        let event = if let Some(event) = self
            .output_ready_events
            .iter()
            .find(|event| Arc::strong_count(event) == 1)
        {
            Arc::clone(event)
        } else {
            let event = Arc::new(CoordinatorCudaEvent::create(library)?);
            self.output_ready_events.push(Arc::clone(&event));
            event
        };
        event.record(self.stream_ptr())?;
        Ok(event)
    }

    fn clear_captured_graphs_if_scratch_changed(
        &mut self,
        before: [CoordinatorCudaScratchSlotState; COORDINATOR_CUDA_SCRATCH_SLOT_COUNT],
    ) {
        if self.captured_graphs.is_empty() {
            return;
        }
        let after = self.workspace.scratch_slot_states();
        let stale_pointer = before
            .iter()
            .zip(after.iter())
            .any(|(before, after)| before.is_some() && before != after);
        if stale_pointer {
            if graph_capture_trace_enabled() {
                eprintln!(
                    "real_full_graph_capture_clear reason=scratch-change shape={} bucket={} entries={}",
                    self.plan.key.shape.label(),
                    self.plan.key.row_bucket.row_capacity,
                    self.captured_graphs.len(),
                );
            }
            self.captured_graphs.clear();
        }
    }

    fn capture_graph(
        &mut self,
        library: &'static NativeLibrary,
        program: CoordinatorCudaGraphProgram,
        signature: CoordinatorCudaGraphSignature,
        action: impl FnOnce(
            &'static NativeLibrary,
            *mut c_void,
            &mut CoordinatorCudaWorkspace,
        ) -> Result<()>,
    ) -> Result<()> {
        let stream = self.stream_ptr();
        unsafe {
            library
                .cuda_graph_begin_capture(stream)
                .context("beginning coordinator CUDA graph capture")?;
        }
        if let Err(error) = action(library, stream, &mut self.workspace) {
            if let Ok(graph_exec) = unsafe { library.cuda_graph_end_capture(stream) } {
                let _ = unsafe { library.cuda_graph_exec_destroy(graph_exec) };
            }
            return Err(error.context("capturing coordinator CUDA graph slot operations"));
        }
        let capture = unsafe {
            library
                .cuda_graph_end_capture_retained(stream)
                .context("ending coordinator CUDA graph capture")?
        };
        let graph = CoordinatorCudaCapturedGraph::new(library, capture)?;
        self.captured_graphs
            .retain(|entry| entry.program != program || entry.signature != signature);
        let last_used = self.next_graph_use();
        self.insert_captured_graph(CoordinatorCudaCapturedGraphEntry {
            program,
            signature,
            capture_identity: 0,
            last_used,
            graph,
        });
        self.graph_captures += 1;
        Ok(())
    }

    fn capture_or_update_graph_exec(
        &mut self,
        library: &'static NativeLibrary,
        program: CoordinatorCudaGraphProgram,
        signature: CoordinatorCudaGraphSignature,
        capture_identity: usize,
        action: impl FnOnce(
            &'static NativeLibrary,
            *mut c_void,
            &mut CoordinatorCudaWorkspace,
        ) -> Result<()>,
    ) -> Result<()> {
        let key = (program, signature, capture_identity);
        if self
            .captured_graphs
            .binary_search_by_key(&key, coordinator_cuda_captured_graph_entry_key)
            .is_ok()
        {
            return Ok(());
        }

        let existing_index = self
            .captured_graphs
            .iter()
            .position(|entry| entry.program == program && entry.signature == signature);
        // MTP verification walks 78 layer-specific Q-A/Q-B graphs at M=2..8.
        // Replacing one graph exec for every layer turns graph update into a
        // roughly 27 ms target-pass host boundary. Keep the bounded set just
        // as decode already does for its layer-specific M=1 identities.
        let retain_capture_identity = graph_program_retains_capture_identity(
            program,
            self.plan.key.row_bucket.row_capacity,
            retain_mtp_query_projection_graphs_enabled(),
        );

        if graph_capture_trace_enabled() {
            let matching_entries = self
                .captured_graphs
                .iter()
                .filter(|entry| entry.program == program && entry.signature == signature)
                .count();
            eprintln!(
                "real_full_graph_capture_miss shape={} bucket={} program={program:?} signature={signature:?} identity={capture_identity} entries={} matching_entries={} retain_identity={retain_capture_identity}",
                self.plan.key.shape.label(),
                self.plan.key.row_bucket.row_capacity,
                self.captured_graphs.len(),
                matching_entries,
            );
        }

        self.stream_synchronize()
            .context("synchronizing coordinator CUDA graph inputs before recapture")?;
        let stream = self.stream_ptr();
        unsafe {
            library
                .cuda_graph_begin_capture(stream)
                .context("beginning coordinator CUDA graph recapture")?;
        }
        if let Err(error) = action(library, stream, &mut self.workspace) {
            if let Ok(graph_exec) = unsafe { library.cuda_graph_end_capture(stream) } {
                let _ = unsafe { library.cuda_graph_exec_destroy(graph_exec) };
            }
            return Err(error.context("recapturing coordinator CUDA graph slot operations"));
        }
        let capture = unsafe {
            library
                .cuda_graph_end_capture_retained(stream)
                .context("ending coordinator CUDA graph recapture")?
        };

        if let Some(index) = existing_index.filter(|_| !retain_capture_identity) {
            let last_used = self.next_graph_use();
            let entry = self
                .captured_graphs
                .get_mut(index)
                .context("coordinator CUDA graph disappeared during recapture update")?;
            entry.graph.update_exec_from_capture(capture)?;
            entry.capture_identity = capture_identity;
            entry.last_used = last_used;
            self.captured_graphs
                .sort_unstable_by_key(coordinator_cuda_captured_graph_entry_key);
        } else {
            if retain_capture_identity {
                let matching_entries = self
                    .captured_graphs
                    .iter()
                    .filter(|entry| entry.program == program && entry.signature == signature)
                    .count();
                let capture_identity_capacity = graph_program_capture_identity_capacity(
                    program,
                    self.plan.key.row_bucket.row_capacity,
                );
                if matching_entries >= capture_identity_capacity {
                    let matching_uses =
                        self.captured_graphs
                            .iter()
                            .enumerate()
                            .filter_map(|(index, entry)| {
                                (entry.program == program && entry.signature == signature)
                                    .then_some((index, entry.last_used))
                            });
                    if let Some(index) = least_recently_used_graph_index(matching_uses) {
                        self.captured_graphs.remove(index);
                    }
                }
            }
            let graph = CoordinatorCudaCapturedGraph::new(library, capture)?;
            let last_used = self.next_graph_use();
            self.insert_captured_graph(CoordinatorCudaCapturedGraphEntry {
                program,
                signature,
                capture_identity,
                last_used,
                graph,
            });
        }
        self.graph_captures += 1;
        Ok(())
    }

    fn has_captured_graph(
        &self,
        program: CoordinatorCudaGraphProgram,
        signature: CoordinatorCudaGraphSignature,
    ) -> bool {
        self.captured_graphs
            .iter()
            .any(|entry| entry.program == program && entry.signature == signature)
    }

    fn has_captured_graph_identity(
        &self,
        program: CoordinatorCudaGraphProgram,
        signature: CoordinatorCudaGraphSignature,
        capture_identity: usize,
    ) -> bool {
        let key = (program, signature, capture_identity);
        self.captured_graphs
            .binary_search_by_key(&key, coordinator_cuda_captured_graph_entry_key)
            .is_ok()
    }

    fn captured_graph_raw_handles(
        &self,
        program: CoordinatorCudaGraphProgram,
        signature: CoordinatorCudaGraphSignature,
    ) -> Option<(*mut c_void, *mut c_void)> {
        self.captured_graphs
            .iter()
            .find(|entry| entry.program == program && entry.signature == signature)
            .map(|entry| (entry.graph.graph_raw, entry.graph.exec_raw))
    }

    #[cfg(test)]
    fn captured_graph_node_counts(
        &self,
        program: CoordinatorCudaGraphProgram,
        signature: CoordinatorCudaGraphSignature,
    ) -> Option<(usize, usize, usize, usize)> {
        self.captured_graphs
            .iter()
            .find(|entry| entry.program == program && entry.signature == signature)
            .map(|entry| {
                (
                    entry.graph.node_count,
                    entry.graph.kernel_node_count,
                    entry.graph.memcpy_node_count,
                    entry.graph.memset_node_count,
                )
            })
    }

    fn launch_captured_graph(
        &mut self,
        library: &'static NativeLibrary,
        program: CoordinatorCudaGraphProgram,
        signature: CoordinatorCudaGraphSignature,
    ) -> Result<()> {
        let stream = self.stream_ptr();
        let graph_index = self
            .captured_graphs
            .iter()
            .position(|entry| entry.program == program && entry.signature == signature)
            .with_context(|| {
                format!(
                    "coordinator CUDA graph slot for shape {} bucket {} has no captured graph exec for program {:?} signature {:?}",
                    self.plan.key.shape.label(),
                    self.plan.key.row_bucket.row_capacity,
                    program,
                    signature
                )
            })?;
        let graph_exec = &self.captured_graphs[graph_index];
        graph_exec.graph.validate_before_launch()?;
        unsafe {
            library
                .cuda_graph_launch(graph_exec.graph.as_ptr(), stream)
                .context("launching coordinator CUDA captured graph")?;
        }
        let last_used = self.next_graph_use();
        self.captured_graphs[graph_index].last_used = last_used;
        self.graph_launches += 1;
        Ok(())
    }

    fn launch_captured_graph_identity(
        &mut self,
        library: &'static NativeLibrary,
        program: CoordinatorCudaGraphProgram,
        signature: CoordinatorCudaGraphSignature,
        capture_identity: usize,
    ) -> Result<()> {
        let stream = self.stream_ptr();
        let key = (program, signature, capture_identity);
        let graph_index = self
            .captured_graphs
            .binary_search_by_key(&key, coordinator_cuda_captured_graph_entry_key)
            .ok()
            .with_context(|| {
                format!(
                    "coordinator CUDA graph slot for shape {} bucket {} has no captured graph exec for program {:?} signature {:?} identity {capture_identity}",
                    self.plan.key.shape.label(),
                    self.plan.key.row_bucket.row_capacity,
                    program,
                    signature
                )
            })?;
        let graph_exec = &self.captured_graphs[graph_index];
        graph_exec.graph.validate_before_launch()?;
        unsafe {
            library
                .cuda_graph_launch(graph_exec.graph.as_ptr(), stream)
                .context("launching coordinator CUDA captured graph identity")?;
        }
        let last_used = self.next_graph_use();
        self.captured_graphs[graph_index].last_used = last_used;
        self.graph_launches += 1;
        Ok(())
    }

    fn buffer(
        &mut self,
        library: &'static NativeLibrary,
        slot: CoordinatorCudaScratchSlot,
        bytes: usize,
        label: &'static str,
    ) -> Result<GlmrtDeviceBuffer> {
        let before = self.workspace.scratch_slot_state(slot);
        let buffer = self.workspace.buffer(library, slot, bytes, label)?;
        let after = self.workspace.scratch_slot_state(slot);
        if !self.captured_graphs.is_empty() && before.is_some() && before != after {
            if graph_capture_trace_enabled() {
                eprintln!(
                    "real_full_graph_capture_clear reason=buffer-resize shape={} bucket={} scratch_slot={} before_bytes={} after_bytes={} entries={}",
                    self.plan.key.shape.label(),
                    self.plan.key.row_bucket.row_capacity,
                    slot.index(),
                    before.map_or(0, |(_, bytes)| bytes),
                    after.map_or(0, |(_, bytes)| bytes),
                    self.captured_graphs.len(),
                );
            }
            self.captured_graphs.clear();
        }
        Ok(buffer)
    }
}

#[allow(dead_code)]
struct CoordinatorCudaGraphWorkspacePool {
    slots: Vec<CoordinatorCudaGraphWorkspaceSlot>,
}

#[allow(dead_code)]
impl CoordinatorCudaGraphWorkspacePool {
    fn glm52_bf16(library: &'static NativeLibrary) -> Result<Self> {
        let plans = CoordinatorGraphInstancePlan::glm52_bf16_all();
        let mut slots = Vec::with_capacity(COORDINATOR_GRAPH_INSTANCE_COUNT);
        for plan in plans {
            let stream = CoordinatorCudaStream::create(library).with_context(|| {
                format!(
                    "creating stream for coordinator CUDA graph shape {} bucket {}",
                    plan.key.shape.label(),
                    plan.key.row_bucket.row_capacity
                )
            })?;
            slots.push(CoordinatorCudaGraphWorkspaceSlot {
                plan,
                stream,
                workspace: CoordinatorCudaWorkspace::default(),
                captured_graphs: Vec::new(),
                acquisitions: 0,
                graph_captures: 0,
                graph_launches: 0,
                graph_use_clock: 0,
                output_ready_events: Vec::new(),
                stable_packed_fp8_mla_page_mapping: None,
            });
        }
        Ok(Self { slots })
    }

    fn len(&self) -> usize {
        self.slots.len()
    }

    fn keys(&self) -> Vec<CoordinatorGraphKey> {
        self.slots
            .iter()
            .map(|slot| slot.plan.key.clone())
            .collect()
    }

    fn stream_ptrs(&self) -> Vec<*mut c_void> {
        self.slots.iter().map(|slot| slot.stream_ptr()).collect()
    }

    fn slot_for_key_mut(
        &mut self,
        key: &CoordinatorGraphKey,
    ) -> Result<&mut CoordinatorCudaGraphWorkspaceSlot> {
        let Some(index) = self.slots.iter().position(|slot| slot.plan.key == *key) else {
            anyhow::bail!(
                "no coordinator CUDA graph workspace slot for shape {} bucket {} dtype {:?}",
                key.shape.label(),
                key.row_bucket.row_capacity,
                key.dtype
            );
        };
        self.slots[index].acquisitions += 1;
        Ok(&mut self.slots[index])
    }
}

#[allow(dead_code)]
struct CoordinatorCudaGraphWorkspaceRegistry {
    keys: Vec<CoordinatorGraphKey>,
    slots: Vec<CoordinatorCudaGraphWorkspaceSlotCell>,
}

#[allow(dead_code)]
struct CoordinatorCudaGraphWorkspaceSlotCell {
    slot: RefCell<CoordinatorCudaGraphWorkspaceSlot>,
}

type CoordinatorCudaGraphWorkspaceSlotGuard<'a> = RefMut<'a, CoordinatorCudaGraphWorkspaceSlot>;
type CoordinatorCudaGraphWorkspaceSlotBorrowResult<'a> =
    std::result::Result<CoordinatorCudaGraphWorkspaceSlotGuard<'a>, BorrowMutError>;

#[allow(dead_code)]
impl CoordinatorCudaGraphWorkspaceSlotCell {
    fn new(slot: CoordinatorCudaGraphWorkspaceSlot) -> Self {
        Self {
            slot: RefCell::new(slot),
        }
    }

    // Compatibility shim for test/helper call sites. This is a same-thread
    // mutable borrow check, not a blocking host mutex.
    fn lock(&self) -> CoordinatorCudaGraphWorkspaceSlotBorrowResult<'_> {
        self.slot.try_borrow_mut()
    }
}

#[allow(dead_code)]
impl CoordinatorCudaGraphWorkspaceRegistry {
    fn glm52_bf16(library: &'static NativeLibrary) -> Result<Self> {
        let pool = CoordinatorCudaGraphWorkspacePool::glm52_bf16(library)?;
        let keys = pool.keys();
        let slots = pool
            .slots
            .into_iter()
            .map(CoordinatorCudaGraphWorkspaceSlotCell::new)
            .collect();
        Ok(Self { keys, slots })
    }

    fn len(&self) -> usize {
        self.slots.len()
    }

    fn keys(&self) -> &[CoordinatorGraphKey] {
        &self.keys
    }

    fn slot_guard_for_key(
        &self,
        key: &CoordinatorGraphKey,
    ) -> Result<CoordinatorCudaGraphWorkspaceSlotGuard<'_>> {
        let Some(index) = self.keys.iter().position(|candidate| candidate == key) else {
            anyhow::bail!(
                "no coordinator CUDA graph workspace registry slot for shape {} bucket {} dtype {:?}",
                key.shape.label(),
                key.row_bucket.row_capacity,
                key.dtype
            );
        };
        let mut guard = self.slots[index].lock().map_err(|_| {
            anyhow::anyhow!(
                "coordinator CUDA graph workspace slot is already borrowed by this host thread"
            )
        })?;
        guard.acquisitions += 1;
        Ok(guard)
    }
}

#[derive(Default)]
struct ResidentDeviceBuffer {
    buffer: GlmrtDeviceBuffer,
    bytes: usize,
    uploaded: bool,
    upload_count: usize,
    label: &'static str,
}

impl ResidentDeviceBuffer {
    fn ensure_capacity(
        &mut self,
        library: &'static NativeLibrary,
        name: &str,
        bytes: usize,
        label: &'static str,
    ) -> Result<GlmrtDeviceBuffer> {
        if bytes == 0 {
            anyhow::bail!("resident coordinator CUDA weight {name} requires non-zero bytes");
        }
        if !self.buffer.ptr.is_null() {
            if self.bytes != bytes {
                anyhow::bail!(
                    "resident coordinator CUDA weight {name} byte length changed from {} to {bytes}; immutable weights must keep stable named buffers",
                    self.bytes
                );
            }
            return Ok(self.buffer);
        }
        let mut buffer = library
            .alloc_device_buffer(bytes)
            .with_context(|| format!("allocating resident coordinator CUDA weight {name}"))?;
        if buffer.ptr.is_null() {
            let _ = library.free_device_buffer(&mut buffer);
            anyhow::bail!("resident coordinator CUDA weight {name} allocated a null device buffer");
        }
        if buffer.bytes < bytes {
            let allocated = buffer.bytes;
            let _ = library.free_device_buffer(&mut buffer);
            anyhow::bail!(
                "resident coordinator CUDA weight {name} allocated {allocated} bytes, expected at least {bytes}"
            );
        }
        self.buffer = buffer;
        self.bytes = bytes;
        self.uploaded = false;
        self.label = label;
        Ok(self.buffer)
    }
}

// Device pointers are opaque native CUDA allocations. Access to resident
// weights is serialized by COORDINATOR_CUDA_RESIDENT_WEIGHTS.
unsafe impl Send for ResidentDeviceBuffer {}

struct OwnedCoordinatorDeviceBuffer {
    library: &'static NativeLibrary,
    buffer: GlmrtDeviceBuffer,
    pool_key: CoordinatorOwnedDeviceBufferPoolKey,
    ready_event: Option<Arc<CoordinatorCudaEvent>>,
}

type CoordinatorOwnedDeviceBufferPoolKey = (&'static str, usize, usize);

thread_local! {
    static COORDINATOR_OWNED_DEVICE_BUFFER_POOL: RefCell<
        HashMap<CoordinatorOwnedDeviceBufferPoolKey, Vec<PooledCoordinatorDeviceBuffer>>
    > = RefCell::new(HashMap::new());
    static COORDINATOR_OWNED_DEVICE_BUFFER_PERMANENT_KEYS: RefCell<
        HashSet<CoordinatorOwnedDeviceBufferPoolKey>
    > = RefCell::new(HashSet::new());
    static COORDINATOR_OWNED_DEVICE_BUFFER_POOL_SEALED: Cell<bool> = const { Cell::new(false) };
    static COORDINATOR_OWNED_DEVICE_BUFFER_BANK: Cell<usize> = const { Cell::new(0) };
}

// Startup exercises every supported decode/speculative width and the canonical
// prefill buckets, so those exact shapes are the permanent serving working set.
// Runtime prompt tails produce many other `(label, bytes)` pairs. Keeping every
// one forever made a single growing coding-agent session retain tens of GiB.

struct PooledCoordinatorDeviceBuffer {
    library: &'static NativeLibrary,
    buffer: GlmrtDeviceBuffer,
}

impl PooledCoordinatorDeviceBuffer {
    fn take(mut self) -> GlmrtDeviceBuffer {
        std::mem::take(&mut self.buffer)
    }
}

impl Drop for PooledCoordinatorDeviceBuffer {
    fn drop(&mut self) {
        if !self.buffer.ptr.is_null() {
            let _ = self.library.free_device_buffer(&mut self.buffer);
        }
    }
}

pub(in crate::commands::real_full) fn seal_coordinator_owned_device_buffer_pool() -> Result<()> {
    let permanent_keys = COORDINATOR_OWNED_DEVICE_BUFFER_POOL.with(|pool| {
        let pool = pool.try_borrow().map_err(|_| {
            anyhow::anyhow!("coordinator owned device-buffer pool is already borrowed")
        })?;
        Ok::<_, anyhow::Error>(pool.keys().copied().collect::<HashSet<_>>())
    })?;
    COORDINATOR_OWNED_DEVICE_BUFFER_PERMANENT_KEYS.with(|permanent| {
        *permanent.try_borrow_mut().map_err(|_| {
            anyhow::anyhow!("coordinator permanent device-buffer key set is already borrowed")
        })? = permanent_keys;
        Ok::<_, anyhow::Error>(())
    })?;
    COORDINATOR_OWNED_DEVICE_BUFFER_POOL_SEALED.with(|sealed| sealed.set(true));
    Ok(())
}

pub(in crate::commands::real_full) fn clear_transient_coordinator_owned_device_buffers(
) -> Result<()> {
    let sealed = COORDINATOR_OWNED_DEVICE_BUFFER_POOL_SEALED.with(Cell::get);
    if !sealed {
        return Ok(());
    }
    let permanent = COORDINATOR_OWNED_DEVICE_BUFFER_PERMANENT_KEYS.with(|permanent| {
        permanent
            .try_borrow()
            .map(|permanent| permanent.clone())
            .map_err(|_| {
                anyhow::anyhow!("coordinator permanent device-buffer key set is already borrowed")
            })
    })?;
    COORDINATOR_OWNED_DEVICE_BUFFER_POOL.with(|pool| {
        let mut pool = pool.try_borrow_mut().map_err(|_| {
            anyhow::anyhow!("coordinator owned device-buffer pool is already borrowed")
        })?;
        pool.retain(|key, _| permanent.contains(key));
        Ok(())
    })
}

impl OwnedCoordinatorDeviceBuffer {
    fn new(library: &'static NativeLibrary, bytes: usize, label: &'static str) -> Result<Self> {
        if bytes == 0 {
            anyhow::bail!("owned coordinator CUDA buffer {label} requires non-zero bytes");
        }
        let bank = COORDINATOR_OWNED_DEVICE_BUFFER_BANK.with(Cell::get);
        let pool_key = (label, bytes, bank);
        if bank > 0 && COORDINATOR_OWNED_DEVICE_BUFFER_POOL_SEALED.with(Cell::get) {
            COORDINATOR_OWNED_DEVICE_BUFFER_PERMANENT_KEYS.with(|permanent| {
                if let Ok(mut permanent) = permanent.try_borrow_mut() {
                    // Nonzero banks are used only by the bounded paired
                    // recurrent scheduler. Keep their exact working sets
                    // stable across requests so captured graphs do not churn
                    // on new device-pointer identities. Bank zero remains
                    // governed by startup's shape allow-list.
                    permanent.insert(pool_key);
                }
            });
        }
        let pooled = COORDINATOR_OWNED_DEVICE_BUFFER_POOL.with(|pool| {
            pool.try_borrow_mut()
                .ok()
                .and_then(|mut pool| pool.get_mut(&pool_key).and_then(Vec::pop))
                .map(PooledCoordinatorDeviceBuffer::take)
        });
        if let Some(buffer) = pooled {
            return Ok(Self {
                library,
                buffer,
                pool_key,
                ready_event: None,
            });
        }
        let mut buffer = library
            .alloc_device_buffer(bytes)
            .with_context(|| format!("allocating owned coordinator CUDA buffer {label}"))?;
        if buffer.ptr.is_null() {
            let _ = library.free_device_buffer(&mut buffer);
            anyhow::bail!("owned coordinator CUDA buffer {label} is null");
        }
        if buffer.bytes < bytes {
            let allocated = buffer.bytes;
            let _ = library.free_device_buffer(&mut buffer);
            anyhow::bail!(
                "owned coordinator CUDA buffer {label} allocated {allocated} bytes, expected at least {bytes}"
            );
        }
        Ok(Self {
            library,
            buffer,
            pool_key,
            ready_event: None,
        })
    }

    fn from_existing(
        library: &'static NativeLibrary,
        buffer: GlmrtDeviceBuffer,
        bytes: usize,
        label: &'static str,
    ) -> Result<Self> {
        if bytes == 0 {
            anyhow::bail!("owned coordinator CUDA buffer {label} requires non-zero bytes");
        }
        if buffer.ptr.is_null() {
            anyhow::bail!("owned coordinator CUDA buffer {label} is null");
        }
        if buffer.bytes < bytes {
            anyhow::bail!(
                "owned coordinator CUDA buffer {label} has {} bytes, expected at least {bytes}",
                buffer.bytes
            );
        }
        Ok(Self {
            library,
            buffer,
            pool_key: (
                label,
                bytes,
                COORDINATOR_OWNED_DEVICE_BUFFER_BANK.with(Cell::get),
            ),
            ready_event: None,
        })
    }

    fn wait_ready_on_stream(&self, stream: *mut c_void) -> Result<()> {
        if let Some(event) = self.ready_event.as_ref() {
            event.wait_on_stream(stream)?;
        }
        Ok(())
    }

    fn synchronize_ready(&self) -> Result<()> {
        if let Some(event) = self.ready_event.as_ref() {
            event.synchronize()?;
        }
        Ok(())
    }
}

pub(in crate::commands::real_full) fn with_coordinator_owned_device_buffer_bank<T, E>(
    bank: usize,
    action: impl FnOnce() -> std::result::Result<T, E>,
) -> std::result::Result<T, E> {
    struct RestoreBank(usize);

    impl Drop for RestoreBank {
        fn drop(&mut self) {
            COORDINATOR_OWNED_DEVICE_BUFFER_BANK.with(|current| current.set(self.0));
        }
    }

    let previous = COORDINATOR_OWNED_DEVICE_BUFFER_BANK.with(|current| current.replace(bank));
    let _restore = RestoreBank(previous);
    action()
}

impl Drop for OwnedCoordinatorDeviceBuffer {
    fn drop(&mut self) {
        if self.buffer.ptr.is_null() {
            return;
        }
        if self.ready_event.is_some() {
            let _ = self.synchronize_ready();
        }
        let mut buffer = self.buffer;
        self.buffer = GlmrtDeviceBuffer::default();
        let mut pooled = false;
        COORDINATOR_OWNED_DEVICE_BUFFER_POOL.with(|pool| {
            if let Ok(mut pool) = pool.try_borrow_mut() {
                pool.entry(self.pool_key)
                    .or_default()
                    .push(PooledCoordinatorDeviceBuffer {
                        library: self.library,
                        buffer,
                    });
                pooled = true;
            }
        });
        if !pooled {
            let _ = self.library.free_device_buffer(&mut buffer);
        }
    }
}

#[derive(Default)]
struct ReusableDeviceBuffer {
    buffer: GlmrtDeviceBuffer,
    capacity: usize,
    label: &'static str,
}

impl ReusableDeviceBuffer {
    fn ensure_capacity(
        &mut self,
        library: &'static NativeLibrary,
        bytes: usize,
        label: &'static str,
    ) -> Result<()> {
        if bytes == 0 {
            anyhow::bail!("coordinator CUDA workspace buffer {label} requires non-zero bytes");
        }
        if !self.buffer.ptr.is_null() && self.capacity >= bytes {
            return Ok(());
        }
        if !self.buffer.ptr.is_null() {
            let mut old = self.buffer;
            library.free_device_buffer(&mut old).with_context(|| {
                format!("freeing reusable coordinator CUDA buffer {}", self.label)
            })?;
            self.buffer = GlmrtDeviceBuffer::default();
            self.capacity = 0;
            self.label = "";
        }
        let mut buffer = library
            .alloc_device_buffer(bytes)
            .with_context(|| format!("allocating reusable coordinator CUDA buffer {label}"))?;
        if buffer.ptr.is_null() {
            let _ = library.free_device_buffer(&mut buffer);
            anyhow::bail!("reusable coordinator CUDA buffer {label} is null");
        }
        if buffer.bytes < bytes {
            let allocated = buffer.bytes;
            let _ = library.free_device_buffer(&mut buffer);
            anyhow::bail!(
                "reusable coordinator CUDA buffer {label} allocated {} bytes, expected at least {bytes}",
                allocated
            );
        }
        self.buffer = buffer;
        self.capacity = buffer.bytes;
        self.label = label;
        Ok(())
    }
}

// Device pointers are opaque native CUDA allocations. Access to reusable slots
// is serialized by mutable access to the owning workspace.
unsafe impl Send for ReusableDeviceBuffer {}

#[derive(Default)]
struct ReusableHostBuffer {
    buffer: GlmrtHostBuffer,
    capacity: usize,
    label: &'static str,
}

impl ReusableHostBuffer {
    fn ensure_capacity(
        &mut self,
        library: &'static NativeLibrary,
        bytes: usize,
        label: &'static str,
    ) -> Result<()> {
        if bytes == 0 {
            anyhow::bail!("coordinator CUDA pinned staging buffer {label} requires non-zero bytes");
        }
        if !self.buffer.ptr.is_null() && self.capacity >= bytes {
            return Ok(());
        }
        if !self.buffer.ptr.is_null() {
            let mut old = self.buffer;
            library.free_host_buffer(&mut old).with_context(|| {
                format!(
                    "freeing reusable coordinator CUDA pinned staging buffer {}",
                    self.label
                )
            })?;
            self.buffer = GlmrtHostBuffer::default();
            self.capacity = 0;
            self.label = "";
        }
        let mut buffer = library.alloc_host_buffer(bytes).with_context(|| {
            format!("allocating reusable coordinator CUDA pinned staging buffer {label}")
        })?;
        if buffer.ptr.is_null() {
            let _ = library.free_host_buffer(&mut buffer);
            anyhow::bail!("reusable coordinator CUDA pinned staging buffer {label} is null");
        }
        if buffer.bytes < bytes {
            let allocated = buffer.bytes;
            let _ = library.free_host_buffer(&mut buffer);
            anyhow::bail!(
                "reusable coordinator CUDA pinned staging buffer {label} allocated {} bytes, expected at least {bytes}",
                allocated
            );
        }
        self.buffer = buffer;
        self.capacity = buffer.bytes;
        self.label = label;
        Ok(())
    }
}

impl Drop for ReusableHostBuffer {
    fn drop(&mut self) {
        let Some(library) = CUDA_NATIVE_LIBRARY.get() else {
            return;
        };
        if !self.buffer.ptr.is_null() {
            let mut old = self.buffer;
            let _ = library.free_host_buffer(&mut old);
            self.buffer = GlmrtHostBuffer::default();
            self.capacity = 0;
        }
    }
}

// Pinned host pointers are opaque native allocations. Access to reusable
// staging slots is serialized by mutable access to the owning workspace or
// registry.
unsafe impl Send for ReusableHostBuffer {}

static COORDINATOR_CUDA_THREAD_WORKSPACES: OnceLock<
    Mutex<HashMap<ThreadId, &'static Mutex<CoordinatorCudaWorkspace>>>,
> = OnceLock::new();
static COORDINATOR_CUDA_RESIDENT_WEIGHTS: OnceLock<Mutex<CoordinatorCudaResidentWeights>> =
    OnceLock::new();
static CUDA_NATIVE_LIBRARY: OnceLock<NativeLibrary> = OnceLock::new();

thread_local! {
    #[allow(dead_code)]
    static COORDINATOR_CUDA_GRAPH_WORKSPACES: RefCell<Option<&'static CoordinatorCudaGraphWorkspaceRegistry>> =
        const { RefCell::new(None) };
    static COORDINATOR_LM_HEAD_READBACK_SCRATCH: RefCell<CoordinatorLmHeadReadbackScratch> =
        RefCell::new(CoordinatorLmHeadReadbackScratch::default());
}

#[cfg(test)]
thread_local! {
    static CUDA_REFERENCE_KERNELS_TEST_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

fn lock_coordinator_cuda_workspace() -> Result<MutexGuard<'static, CoordinatorCudaWorkspace>> {
    let thread_id = std::thread::current().id();
    let workspace_mutex = {
        let registry =
            COORDINATOR_CUDA_THREAD_WORKSPACES.get_or_init(|| Mutex::new(HashMap::new()));
        let mut registry = registry
            .lock()
            .map_err(|_| anyhow::anyhow!("coordinator CUDA thread workspace registry poisoned"))?;
        *registry.entry(thread_id).or_insert_with(|| {
            // Scratch buffers are process-lifetime CUDA allocations; using a
            // stable per-thread mutex keeps call sites simple without
            // reintroducing a process-wide GPU-work lock.
            Box::leak(Box::new(Mutex::new(CoordinatorCudaWorkspace::default())))
        })
    };
    workspace_mutex
        .lock()
        .map_err(|_| anyhow::anyhow!("coordinator CUDA thread workspace mutex poisoned"))
}

#[cfg(test)]
mod tests {
    use super::{
        b12x_mla_rope_attention_bf16_shape_supported, bf16_value, bf16_values_to_f32,
        capture_or_update_coord_dense_envelope_bf16_graph,
        capture_or_update_coord_sparse_a_envelope_bf16_graph,
        capture_or_update_layer_b12x_mla_rope_attention_bf16_graph,
        causal_attention_graph_signature, causal_attention_rows, causal_attention_rows_bf16,
        coord_attention_graph_key_for_layer_rows,
        coord_compressed_attention_decode_graph_key_for_layer,
        coord_dense_mlp_graph_key_for_gate_up_down_names,
        coord_layer_graph_key_for_dsa_k_norm_names, coord_layer_graph_key_for_full_hidden_rows,
        coord_linear_graph_key_for_weight_name, coordinator_cuda_graph_test_stats,
        coordinator_cuda_graph_workspace_registry, coordinator_cuda_reference_kernels_enabled,
        cpu_causal_attention_rows_bf16, cpu_linear_rows_bf16, cpu_lm_head_sample_topk_topp_bf16,
        cpu_mla_rope_attention_rows_bf16, cpu_rmsnorm_hidden_bf16, cpu_rope_rows_bf16,
        cpu_router_topk_bf16, cpu_silu_gated_mlp_rows_bf16,
        cuda_causal_attention_rows_bf16_for_layer,
        cuda_embedding_lookup_bf16_preloaded_resident_weight_device_output,
        cuda_layer_norm_affine_bf16_preloaded_resident_weight_bias_device_input_output,
        cuda_layer_norm_affine_f32_bf16_preloaded_resident_weight_bias,
        cuda_linear_rows_bf16_preloaded_resident_weight,
        cuda_linear_rows_bf16_preloaded_resident_weight_device_input,
        cuda_linear_rows_bf16_preloaded_resident_weight_device_output,
        cuda_linear_rows_bf16_resident_weight, cuda_mla_rope_attention_rows_bf16_for_layer,
        cuda_native_library, cuda_reference_kernels_test_override,
        cuda_residual_add_prefix_bf16_bytes_into,
        cuda_rmsnorm_hidden_bf16_preloaded_resident_weight,
        cuda_rmsnorm_hidden_bf16_resident_weight, cuda_rope_rows_bf16_for_layer,
        cuda_router_topk_bf16, cuda_router_topk_bf16_preloaded_resident_weight,
        cuda_router_topk_bf16_preloaded_resident_weight_bias,
        cuda_router_topk_bf16_resident_weight, cuda_scatter_add_rows_bf16_to_f32,
        cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight,
        cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight_device_input_device_output,
        cuda_silu_gated_mlp_rows_bf16_resident_weight,
        cuda_stream_sparse_b_scatter_shared_residual_add_bf16_device_outputs,
        dense_mlp_graph_signature, device_bf16_output_from_bf16_bytes,
        device_bf16_output_from_f32_values, device_buffer_byte_view,
        embedding_lookup_bf16_preloaded_resident_weight_device_output, embedding_lookup_rows,
        embedding_lookup_rows_bf16, embedding_lookup_rows_bf16_resident_weight,
        embedding_lookup_rows_bf16_staged_resident_weight, f32_bytes, f32_values_to_bf16_bytes,
        f32_vec_from_bytes, gather_rows_bf16, glm52_layer_id_from_tensor_name,
        layer_norm_affine_bf16_preloaded_resident_weight_bias_device_input_output,
        layer_norm_affine_f32_bf16_preloaded_resident_weight_bias, least_recently_used_graph_index,
        linear_residual_add_rows_bf16_preloaded_resident_weight_device_input,
        linear_residual_add_rows_bf16_preloaded_resident_weight_device_input_device_output,
        linear_residual_add_rows_bf16_preloaded_resident_weight_padded_device_input,
        linear_residual_add_rows_bf16_preloaded_resident_weight_padded_device_input_device_output,
        linear_rows, linear_rows_bf16, linear_rows_bf16_preloaded_resident_weight,
        linear_rows_bf16_preloaded_resident_weight_device_output,
        linear_rows_bf16_preloaded_resident_weight_padded_device_input,
        linear_rows_bf16_resident_weight, lm_head_argmax_bf16,
        lm_head_argmax_bf16_preloaded_resident_weight,
        lm_head_argmax_bf16_preloaded_resident_weight_device_input,
        lm_head_argmax_bf16_resident_weight,
        lm_head_argmax_sample_topk_topp_bf16_preloaded_resident_weight_device_input,
        lm_head_device_input_readback_scratch_state, lm_head_sample_topk_topp_bf16,
        lm_head_sample_topk_topp_bf16_preloaded_resident_weight,
        lm_head_sample_topk_topp_bf16_preloaded_resident_weight_device_input,
        lm_head_sample_topk_topp_bf16_resident_weight, lock_coordinator_cuda_resident_weights,
        lock_coordinator_cuda_workspace, logits_argmax, logits_sample_topk_topp,
        mla_decode_kv_commit_bf16_device_output,
        mla_decode_query_dsa_projection_bf16_device_outputs,
        mla_decode_query_projection_bf16_device_output,
        mla_query_split_rope_bf16_device_buffers_for_layer,
        mla_rope_attention_device_buffers_bf16_for_layer, mla_rope_attention_graph_signature,
        mla_rope_attention_rows, mla_rope_attention_rows_bf16, native_library_path,
        native_library_path_candidates, native_library_version_has_cuda,
        padded_linear_graph_signature, padded_linear_residual_graph_signature,
        preload_resident_weight_from_host_staging, require_cuda_enabled_native_library,
        resident_weight_registry_key, residual_add_bf16_device_inputs_device_output,
        residual_add_bf16_device_inputs_output, residual_add_prefix, residual_add_prefix_bf16,
        residual_add_prefix_bf16_bytes, residual_add_prefix_bf16_bytes_into, rmsnorm_hidden,
        rmsnorm_hidden_bf16, rmsnorm_hidden_bf16_preloaded_resident_weight,
        rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output,
        rmsnorm_hidden_bf16_preloaded_resident_weight_device_output,
        rmsnorm_hidden_bf16_resident_weight, rope_graph_signature, rope_rows, rope_rows_bf16,
        router_topk, router_topk_bf16, router_topk_bf16_preloaded_resident_weight,
        router_topk_bf16_preloaded_resident_weight_bias,
        router_topk_bf16_preloaded_resident_weight_bias_device_input,
        router_topk_bf16_preloaded_resident_weight_device_input, router_topk_bf16_resident_weight,
        router_topk_bf16_resident_weight_device_input, scatter_add_rows_bf16_to_f32,
        silu_gated_mlp_rows, silu_gated_mlp_rows_bf16,
        silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight,
        silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight_device_input_device_output_only,
        silu_gated_mlp_rows_bf16_preloaded_gate_up_resident_weight,
        silu_gated_mlp_rows_bf16_resident_weight, sparse_b_scatter_residual_add_bf16,
        sparse_b_scatter_shared_residual_add_bf16_device_output, u32_vec_from_bytes,
        validate_linear_bf16_preloaded_resident_padded_device_input,
        with_coordinator_cuda_graph_slot, with_coordinator_cuda_graph_workspace_slot,
        CoordDenseEnvelopeBf16Buffers, CoordSparseAEnvelopeBf16Buffers,
        CoordinatorCudaGraphProgram, CoordinatorCudaGraphSignature,
        CoordinatorCudaGraphWorkspacePool, CoordinatorCudaScratchSlot,
        CudaStreamedSparseBAccumulator, DeviceBf16Output, GlmrtDeviceBuffer, LayerNormAffineOutput,
        LinearOutput, LinearResidentView, MlaDecodeKvDsaProjectionWeights,
        MlpGateUpDownResidentView, MlpGateUpResidentView, NativeLibrary,
        OwnedCoordinatorDeviceBuffer, StreamedSparseBAccumulatorChunk,
        StreamedSparseBResidualSegment, B12X_MLA_ROPE_ATTENTION_BF16_BACKEND,
        CPU_REFERENCE_CAUSAL_ATTENTION_BACKEND, CPU_REFERENCE_CAUSAL_ATTENTION_BF16_BACKEND,
        CPU_REFERENCE_EMBEDDING_LOOKUP_BACKEND, CPU_REFERENCE_EMBEDDING_LOOKUP_BF16_BACKEND,
        CPU_REFERENCE_GATHER_ROWS_BF16_BACKEND, CPU_REFERENCE_LINEAR_BACKEND,
        CPU_REFERENCE_LINEAR_BF16_BACKEND, CPU_REFERENCE_LM_HEAD_ARGMAX_BF16_BACKEND,
        CPU_REFERENCE_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_BACKEND, CPU_REFERENCE_LOGITS_ARGMAX_BACKEND,
        CPU_REFERENCE_LOGITS_SAMPLE_TOPK_TOPP_BACKEND, CPU_REFERENCE_MLA_ROPE_ATTENTION_BACKEND,
        CPU_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND, CPU_REFERENCE_RESIDUAL_ADD_BACKEND,
        CPU_REFERENCE_RESIDUAL_ADD_BF16_BACKEND, CPU_REFERENCE_RMSNORM_BACKEND,
        CPU_REFERENCE_RMSNORM_BF16_BACKEND, CPU_REFERENCE_ROPE_BACKEND,
        CPU_REFERENCE_ROPE_BF16_BACKEND, CPU_REFERENCE_ROUTER_TOPK_BACKEND,
        CPU_REFERENCE_ROUTER_TOPK_BF16_BACKEND, CPU_REFERENCE_SCATTER_ADD_ROWS_BF16_TO_F32_BACKEND,
        CPU_REFERENCE_SILU_GATED_MLP_BACKEND, CPU_REFERENCE_SILU_GATED_MLP_BF16_BACKEND,
        CUDA_REFERENCE_CAUSAL_ATTENTION_BF16_BACKEND,
        CUDA_REFERENCE_DEVICE_BF16_HOST_UPLOAD_BACKEND,
        CUDA_REFERENCE_EMBEDDING_LOOKUP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
        CUDA_REFERENCE_LAYER_NORM_AFFINE_BF16_PRELOADED_RESIDENT_BACKEND,
        CUDA_REFERENCE_LAYER_NORM_AFFINE_F32_BF16_PRELOADED_RESIDENT_BACKEND,
        CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
        CUDA_REFERENCE_LINEAR_BF16_RESIDENT_WEIGHT_BACKEND,
        CUDA_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND, CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND,
        CUDA_REFERENCE_RMSNORM_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
        CUDA_REFERENCE_RMSNORM_BF16_RESIDENT_WEIGHT_BACKEND, CUDA_REFERENCE_ROPE_BF16_BACKEND,
        CUDA_REFERENCE_ROUTER_TOPK_BF16_BACKEND,
        CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
        CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_BACKEND,
        CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_DEVICE_INPUT_BACKEND,
        CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_DEVICE_INPUT_BACKEND,
        CUDA_REFERENCE_ROUTER_TOPK_BF16_RESIDENT_WEIGHT_BACKEND,
        CUDA_REFERENCE_ROUTER_TOPK_BF16_RESIDENT_WEIGHT_DEVICE_INPUT_BACKEND,
        CUDA_REFERENCE_SCATTER_ADD_ROWS_BF16_TO_F32_BACKEND,
        CUDA_REFERENCE_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND,
        CUDA_REFERENCE_SILU_GATED_MLP_BF16_RESIDENT_WEIGHT_BACKEND, GLM52_Q_LORA_RANK,
        TRITON_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
        TRITON_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
        TRITON_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_BACKEND,
        TRITON_ROUTER_TOPK_BF16_RESIDENT_WEIGHT_BACKEND,
        TRITON_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND,
        TRITON_SILU_GATED_MLP_BF16_RESIDENT_WEIGHT_BACKEND,
    };
    use crate::python_graph_capture::coordinator_python_capture_test_override;
    use anyhow::{Context, Result};
    use glmrt_core::{
        CoordinatorGraphInstancePlan, CoordinatorGraphKey, CoordinatorGraphShape, KvCacheDType,
        LayerWaveMode, COORDINATOR_GRAPH_INSTANCE_COUNT, GLM52_DSA_INDEX_HEAD_DIM,
        GLM52_HIDDEN_BF16_BYTES, GLM52_HIDDEN_SIZE, GLM52_MLA_FP8_DS_BYTES_PER_TOKEN,
        GLM52_MLA_KV_LORA_RANK, GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN, GLM52_MLA_QK_ROPE_HEAD_DIM,
        GLM52_ROUTED_SCALING_FACTOR,
    };
    use pyo3::exceptions::PyRuntimeError;
    use pyo3::prelude::*;
    use pyo3::types::PyModule;
    use std::collections::HashSet;
    use std::env;
    use std::ffi::{c_void, OsString};
    use std::path::PathBuf;
    use std::sync::Mutex;

    static B12X_MLA_TEST_ENV_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn owned_device_buffer_pool_keeps_only_startup_shapes_after_seal() {
        super::COORDINATOR_OWNED_DEVICE_BUFFER_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            pool.clear();
            pool.insert(("test permanent small shape", 16, 0), Vec::new());
            pool.insert(("test permanent large shape", 16 << 20, 0), Vec::new());
        });
        super::COORDINATOR_OWNED_DEVICE_BUFFER_PERMANENT_KEYS
            .with(|permanent| permanent.borrow_mut().clear());
        super::COORDINATOR_OWNED_DEVICE_BUFFER_POOL_SEALED.with(|sealed| sealed.set(false));

        super::seal_coordinator_owned_device_buffer_pool().unwrap();
        super::COORDINATOR_OWNED_DEVICE_BUFFER_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            pool.insert(("test transient small shape", 32, 0), Vec::new());
            pool.insert(("test transient large shape", 32 << 20, 0), Vec::new());
        });
        super::clear_transient_coordinator_owned_device_buffers().unwrap();

        super::COORDINATOR_OWNED_DEVICE_BUFFER_POOL.with(|pool| {
            let pool = pool.borrow();
            assert!(pool.contains_key(&("test permanent small shape", 16, 0)));
            assert!(pool.contains_key(&("test permanent large shape", 16 << 20, 0)));
            assert!(!pool.contains_key(&("test transient small shape", 32, 0)));
            assert!(!pool.contains_key(&("test transient large shape", 32 << 20, 0)));
        });
    }

    #[test]
    fn retained_cuda_graph_eviction_selects_least_recently_used_identity() {
        assert_eq!(
            least_recently_used_graph_index([(2, 41), (5, 7), (9, 23)]),
            Some(5)
        );
        assert_eq!(least_recently_used_graph_index([]), None);
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    fn patterned_bf16_values(len: usize, step: f32, salt: f32) -> Vec<f32> {
        (0..len)
            .map(|index| (((index as f32 + salt) % 29.0) - 14.0) * step)
            .collect()
    }

    fn cuda_reference_kernels_test_enabled() -> bool {
        native_library_path().is_some() && coordinator_cuda_reference_kernels_enabled()
    }

    struct EnvVarGuard {
        name: &'static str,
        previous: Option<OsString>,
    }

    impl EnvVarGuard {
        fn set(name: &'static str, value: impl AsRef<std::ffi::OsStr>) -> Self {
            let previous = env::var_os(name);
            env::set_var(name, value);
            Self { name, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(previous) = &self.previous {
                env::set_var(self.name, previous);
            } else {
                env::remove_var(self.name);
            }
        }
    }

    #[pyfunction(name = "capture", signature = (ctx, *, rows, heads, nope_dim, rope_dim, v_dim, scale))]
    fn test_b12x_mla_cuda_reference_capture(
        ctx: &Bound<'_, PyAny>,
        rows: usize,
        heads: usize,
        nope_dim: usize,
        rope_dim: usize,
        v_dim: usize,
        scale: f64,
    ) -> PyResult<()> {
        let buffers = ctx.get_item("buffers")?;
        let q_nope = py_device_buffer_arg(&buffers, "q_nope")?;
        let q_rope = py_device_buffer_arg(&buffers, "q_rope")?;
        let k_nope = py_device_buffer_arg(&buffers, "k_nope")?;
        let k_rope = py_device_buffer_arg(&buffers, "k_rope")?;
        let values = py_device_buffer_arg(&buffers, "values")?;
        let output = py_device_buffer_arg(&buffers, "output")?;
        let cuda_stream = ctx.get_item("cuda_stream")?.extract::<usize>()? as *mut c_void;
        let library =
            cuda_native_library().map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        unsafe {
            library
                .cuda_mla_rope_attention_bf16_async(
                    q_nope,
                    q_rope,
                    k_nope,
                    k_rope,
                    values,
                    output,
                    rows,
                    heads,
                    nope_dim,
                    rope_dim,
                    v_dim,
                    scale as f32,
                    cuda_stream,
                )
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        }
        Ok(())
    }

    fn py_device_buffer_arg(buffers: &Bound<'_, PyAny>, name: &str) -> PyResult<GlmrtDeviceBuffer> {
        let buffer = buffers.get_item(name)?;
        Ok(GlmrtDeviceBuffer {
            ptr: buffer.get_item("ptr")?.extract::<usize>()? as *mut c_void,
            bytes: buffer.get_item("bytes")?.extract::<usize>()?,
            device_id: buffer.get_item("device_id")?.extract::<i32>()?,
            flags: buffer.get_item("flags")?.extract::<u64>()?,
        })
    }

    fn install_b12x_mla_cuda_reference_capture_module() -> Result<()> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| -> PyResult<()> {
            let module = PyModule::new_bound(py, "glmrt_test_b12x_mla_cuda_capture")?;
            module.add_function(pyo3::wrap_pyfunction!(
                test_b12x_mla_cuda_reference_capture,
                &module
            )?)?;
            let sys = PyModule::import_bound(py, "sys")?;
            sys.getattr("modules")?
                .set_item("glmrt_test_b12x_mla_cuda_capture", module)?;
            Ok(())
        })
        .map_err(|err| anyhow::anyhow!(err.to_string()))
    }

    fn uploaded_test_device_buffer(
        library: &'static NativeLibrary,
        bytes: &[u8],
        label: &'static str,
    ) -> Result<OwnedCoordinatorDeviceBuffer> {
        let buffer = OwnedCoordinatorDeviceBuffer::new(library, bytes.len(), label)?;
        library
            .copy_h2d(buffer.buffer, bytes)
            .with_context(|| format!("uploading {label}"))?;
        Ok(buffer)
    }

    fn empty_test_device_buffer(
        library: &'static NativeLibrary,
        bytes: usize,
        label: &'static str,
    ) -> Result<OwnedCoordinatorDeviceBuffer> {
        OwnedCoordinatorDeviceBuffer::new(library, bytes, label)
    }

    fn patterned_bf16_bytes(values: usize, scale: f32, bias: f32) -> Vec<u8> {
        let values = (0..values)
            .map(|idx| {
                let centered = (idx % 11) as f32 - 5.0;
                bias + centered * scale
            })
            .collect::<Vec<_>>();
        bf16_bytes(&values)
    }

    fn constant_bf16_bytes(values: usize, value: f32) -> Vec<u8> {
        bf16_bytes(&vec![value; values])
    }

    fn output_has_finite_nonzero_bf16_values(bytes: &[u8]) -> bool {
        let values = bf16_values_to_f32(bytes);
        values.iter().all(|value| value.is_finite()) && values.iter().any(|value| *value != 0.0)
    }

    struct RouterGraphUpdateFixture {
        rows: usize,
        experts: usize,
        top_k: usize,
        hidden: Vec<u8>,
        weight_a: Vec<u8>,
        weight_b: Vec<u8>,
        bias_a: Vec<f32>,
        bias_b: Vec<f32>,
        expected_indices_a: Vec<usize>,
        expected_indices_b: Vec<usize>,
    }

    fn full_width_router_graph_update_fixture() -> RouterGraphUpdateFixture {
        let rows = 2_usize;
        let experts = 5_usize;
        let top_k = 3_usize;
        let mut hidden_values = vec![0.0_f32; rows * GLM52_HIDDEN_SIZE];
        hidden_values[0] = 1.0;
        hidden_values[GLM52_HIDDEN_SIZE + 1] = 1.0;
        let hidden = bf16_bytes(&hidden_values);

        let mut weight_values_a = vec![0.0_f32; experts * GLM52_HIDDEN_SIZE];
        weight_values_a[0] = 3.0;
        weight_values_a[GLM52_HIDDEN_SIZE + 1] = 3.0;
        weight_values_a[2 * GLM52_HIDDEN_SIZE] = 1.0;
        weight_values_a[3 * GLM52_HIDDEN_SIZE + 1] = 1.0;

        let mut weight_values_b = vec![0.0_f32; experts * GLM52_HIDDEN_SIZE];
        weight_values_b[2 * GLM52_HIDDEN_SIZE] = 3.0;
        weight_values_b[3 * GLM52_HIDDEN_SIZE + 1] = 3.0;
        weight_values_b[4 * GLM52_HIDDEN_SIZE] = 1.0;
        weight_values_b[4 * GLM52_HIDDEN_SIZE + 1] = 1.0;

        RouterGraphUpdateFixture {
            rows,
            experts,
            top_k,
            hidden,
            weight_a: bf16_bytes(&weight_values_a),
            weight_b: bf16_bytes(&weight_values_b),
            bias_a: vec![-0.1, -0.1, 0.0, 0.0, 0.1],
            bias_b: vec![0.1, -0.3, 0.0, -0.1, 0.1],
            expected_indices_a: vec![0, 2, 4, 1, 3, 4],
            expected_indices_b: vec![2, 4, 0, 3, 4, 0],
        }
    }

    struct LinearGraphUpdateFixture {
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        input: Vec<u8>,
        weight_a: Vec<u8>,
        weight_b: Vec<u8>,
        bias_a: Vec<u8>,
        bias_b: Vec<u8>,
        expected_a: Vec<f32>,
        expected_b: Vec<f32>,
    }

    fn full_width_linear_graph_update_fixture() -> LinearGraphUpdateFixture {
        let rows = 1_usize;
        let input_dim = GLM52_HIDDEN_SIZE;
        let output_dim = 3_usize;
        let mut input_values = vec![0.0_f32; input_dim];
        input_values[0] = 1.0;

        let mut weight_values_a = vec![0.0_f32; output_dim * input_dim];
        weight_values_a[0] = 0.5;
        weight_values_a[input_dim] = -2.0;
        weight_values_a[2 * input_dim] = 3.0;

        let mut weight_values_b = vec![0.0_f32; output_dim * input_dim];
        weight_values_b[0] = 2.0;
        weight_values_b[input_dim] = 1.0;
        weight_values_b[2 * input_dim] = -1.0;

        LinearGraphUpdateFixture {
            rows,
            input_dim,
            output_dim,
            input: bf16_bytes(&input_values),
            weight_a: bf16_bytes(&weight_values_a),
            weight_b: bf16_bytes(&weight_values_b),
            bias_a: bf16_bytes(&[0.25, -0.25, 1.0]),
            bias_b: bf16_bytes(&[0.5, 0.25, -0.75]),
            expected_a: vec![0.75, -2.25, 4.0],
            expected_b: vec![2.5, 1.25, -1.75],
        }
    }

    struct RmsNormGraphUpdateFixture {
        rows: usize,
        hidden_dim: usize,
        eps: f32,
        hidden: Vec<u8>,
        weight_a: Vec<u8>,
        weight_b: Vec<u8>,
    }

    fn full_width_rmsnorm_graph_update_fixture() -> RmsNormGraphUpdateFixture {
        let rows = 1_usize;
        let hidden_dim = GLM52_HIDDEN_SIZE;
        let eps = 1.0e-6_f32;
        let mut hidden_values = vec![0.0_f32; hidden_dim];
        hidden_values[0] = 0.25;
        hidden_values[1] = 2.0;
        let mut weight_values_a = vec![1.0_f32; hidden_dim];
        weight_values_a[0] = 1.0;
        weight_values_a[1] = 1.0;
        let mut weight_values_b = vec![1.0_f32; hidden_dim];
        weight_values_b[0] = 2.0;
        weight_values_b[1] = 0.5;

        RmsNormGraphUpdateFixture {
            rows,
            hidden_dim,
            eps,
            hidden: bf16_bytes(&hidden_values),
            weight_a: bf16_bytes(&weight_values_a),
            weight_b: bf16_bytes(&weight_values_b),
        }
    }

    fn assert_bf16_values_close(actual: &[u8], expected: &[u8], tolerance: f32) {
        let actual = bf16_values_to_f32(actual);
        let expected = bf16_values_to_f32(expected);
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "BF16 value {index} mismatch: actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn native_library_candidates_only_include_cuda_builds() {
        let candidates = native_library_path_candidates();

        assert!(!candidates.is_empty());
        assert!(candidates
            .iter()
            .all(|path| path.to_string_lossy().contains("native/build-cuda/")));
    }

    #[test]
    fn native_library_version_cuda_flag_is_explicit() {
        assert!(native_library_version_has_cuda(
            "glmrt_native 0.1.0 cuda=on"
        ));
        assert!(!native_library_version_has_cuda(
            "glmrt_native 0.1.0 cuda=off"
        ));
        assert!(!native_library_version_has_cuda("glmrt_native 0.1.0"));
    }

    #[test]
    fn cuda_native_library_rejects_cuda_off_build_when_present() -> Result<()> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .join("native/build/libglmrt_native.so");
        if !path.exists() {
            return Ok(());
        }
        let library = unsafe { NativeLibrary::load(&path)? };
        let version = library.version()?;
        if native_library_version_has_cuda(&version) {
            return Ok(());
        }

        let err =
            require_cuda_enabled_native_library(&library, &path, "test CUDA kernels").unwrap_err();

        assert!(err
            .to_string()
            .contains("require a CUDA-enabled native library"));
        assert!(err.to_string().contains("cuda=off"));
        Ok(())
    }

    #[test]
    fn residual_add_uses_cpu_reference_when_cuda_reference_is_not_enabled() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let output = residual_add_prefix(&[1.0, -2.0, 0.5], &[0.25, 3.0, -0.75]).unwrap();

        assert_eq!(output.values, vec![1.25, 1.0, -0.25]);
        assert_eq!(output.backend, CPU_REFERENCE_RESIDUAL_ADD_BACKEND);
    }

    #[test]
    fn residual_add_rejects_mismatched_prefixes_before_backend_selection() {
        let err = residual_add_prefix(&[1.0, 2.0], &[1.0]).unwrap_err();

        assert!(err.to_string().contains("residual length mismatch"));
    }

    #[test]
    fn residual_add_bf16_uses_cpu_reference_when_cuda_reference_is_not_enabled() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let residual = bf16_bytes(&[1.0, -2.0, 0.5]);
        let delta = bf16_bytes(&[0.25, 3.0, -0.75]);
        let output = residual_add_prefix_bf16(&residual, &delta).unwrap();

        assert_eq!(output.values, vec![1.25, 1.0, -0.25]);
        assert_eq!(output.backend, CPU_REFERENCE_RESIDUAL_ADD_BF16_BACKEND);
    }

    #[test]
    fn residual_add_bf16_bytes_preserves_bf16_output() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let residual = bf16_bytes(&[1.0, -2.0, 0.5]);
        let delta = bf16_bytes(&[0.25, 3.0, -0.75]);
        let output = residual_add_prefix_bf16_bytes(&residual, &delta).unwrap();

        assert_eq!(output.bytes, bf16_bytes(&[1.25, 1.0, -0.25]));
        assert_eq!(output.backend, CPU_REFERENCE_RESIDUAL_ADD_BF16_BACKEND);
    }

    #[test]
    fn residual_add_bf16_bytes_into_reuses_output_buffer() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let residual = bf16_bytes(&[1.0, -2.0, 0.5]);
        let delta = bf16_bytes(&[0.25, 3.0, -0.75]);
        let mut output = vec![0_u8; residual.len()];
        let backend = residual_add_prefix_bf16_bytes_into(&residual, &delta, &mut output).unwrap();

        assert_eq!(output, bf16_bytes(&[1.25, 1.0, -0.25]));
        assert_eq!(backend, CPU_REFERENCE_RESIDUAL_ADD_BF16_BACKEND);
    }

    #[test]
    fn residual_add_bf16_full_width_uses_coord_sparse_b_graph_slot_when_cuda_reference_is_enabled(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let values = GLM52_HIDDEN_BF16_BYTES / std::mem::size_of::<u16>();
        let residual = bf16_bytes(&vec![1.0_f32; values]);
        let delta = bf16_bytes(&vec![0.5_f32; values]);
        let mut output = vec![0_u8; GLM52_HIDDEN_BF16_BYTES];
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseB,
            LayerWaveMode::Decode,
            1,
        )?;
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Sparse-B decode graph key is registered");
        let (acquisitions_before, graph_captures_before, graph_launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_captures, slot.graph_launches)
        };

        let backend = match cuda_residual_add_prefix_bf16_bytes_into(&residual, &delta, &mut output)
        {
            Ok(backend) => backend,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let (acquisitions_after, graph_captures_after, graph_launches_after) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_captures, slot.graph_launches)
        };

        assert_eq!(backend, CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND);
        assert!(acquisitions_after > acquisitions_before);
        assert!(graph_captures_after >= graph_captures_before);
        assert!(graph_launches_after > graph_launches_before);
        assert_eq!(bf16_value(&output, 0), 1.5);
        assert_eq!(bf16_value(&output, values - 1), 1.5);
        Ok(())
    }

    #[test]
    fn residual_add_bf16_full_width_captures_coord_sparse_b_graph_exec() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);

        let result = (|| -> Result<()> {
            let rows = 32;
            let values = rows * GLM52_HIDDEN_SIZE;
            let bytes = values * std::mem::size_of::<u16>();
            let graph_key = CoordinatorGraphKey::glm52_bf16(
                CoordinatorGraphShape::CoordSparseB,
                LayerWaveMode::Prefill,
                rows,
            )?;
            let registry = match coordinator_cuda_graph_workspace_registry() {
                Ok(registry) => registry,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            let graph_index = registry
                .keys()
                .iter()
                .position(|key| key == &graph_key)
                .expect("Coord-Sparse-B prefill graph key is registered");
            let (graph_captures_before, graph_launches_before) = {
                let slot = registry.slots[graph_index]
                    .lock()
                    .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
                (slot.graph_captures, slot.graph_launches)
            };

            let residual0 = bf16_bytes(&vec![1.0_f32; values]);
            let delta0 = bf16_bytes(&vec![0.5_f32; values]);
            let mut output0 = vec![0_u8; bytes];
            let backend0 =
                cuda_residual_add_prefix_bf16_bytes_into(&residual0, &delta0, &mut output0)?;
            assert_eq!(backend0, CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND);
            assert_eq!(bf16_value(&output0, 0), 1.5);
            assert_eq!(bf16_value(&output0, values - 1), 1.5);

            let residual1 = bf16_bytes(&vec![2.0_f32; values]);
            let delta1 = bf16_bytes(&vec![0.25_f32; values]);
            let mut output1 = vec![0_u8; bytes];
            let backend1 =
                cuda_residual_add_prefix_bf16_bytes_into(&residual1, &delta1, &mut output1)?;
            assert_eq!(backend1, CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND);
            assert_eq!(bf16_value(&output1, 0), 2.25);
            assert_eq!(bf16_value(&output1, values - 1), 2.25);

            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            assert!(slot.has_captured_graph(
                CoordinatorCudaGraphProgram::SparseBResidualAddBf16,
                CoordinatorCudaGraphSignature::residual_add_bf16(bytes)
            ));
            assert!(slot.graph_captures >= graph_captures_before);
            assert!(slot.graph_launches >= graph_launches_before + 2);
            Ok(())
        })();

        result
    }

    #[test]
    fn residual_add_bf16_full_width_device_inputs_update_coord_sparse_b_graph_exec() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);

        let result = (|| -> Result<()> {
            let rows = 1;
            let values = rows * GLM52_HIDDEN_SIZE;
            let bytes = values * std::mem::size_of::<u16>();
            let graph_key = CoordinatorGraphKey::glm52_bf16(
                CoordinatorGraphShape::CoordSparseB,
                LayerWaveMode::Decode,
                rows,
            )?;
            let registry = match coordinator_cuda_graph_workspace_registry() {
                Ok(registry) => registry,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            let graph_index = registry
                .keys()
                .iter()
                .position(|key| key == &graph_key)
                .expect("Coord-Sparse-B decode graph key is registered");
            let (graph_captures_before, graph_launches_before) = {
                let slot = registry.slots[graph_index]
                    .lock()
                    .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
                (slot.graph_captures, slot.graph_launches)
            };

            let residual0 = match device_bf16_output_from_f32_values(
                &vec![1.0_f32; values],
                rows,
                GLM52_HIDDEN_SIZE,
                "test graph residual0 device upload",
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            let delta0 = match device_bf16_output_from_f32_values(
                &vec![0.5_f32; values],
                rows,
                GLM52_HIDDEN_SIZE,
                "test graph delta0 device upload",
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            let residual1 = match device_bf16_output_from_f32_values(
                &vec![2.0_f32; values],
                rows,
                GLM52_HIDDEN_SIZE,
                "test graph residual1 device upload",
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            let delta1 = match device_bf16_output_from_f32_values(
                &vec![0.25_f32; values],
                rows,
                GLM52_HIDDEN_SIZE,
                "test graph delta1 device upload",
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };

            let host_output0 = residual_add_bf16_device_inputs_output(&residual0, &delta0)?;
            assert_eq!(
                host_output0.backend,
                CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND
            );
            assert_eq!(host_output0.values[0], 1.5);
            assert_eq!(host_output0.values[values - 1], 1.5);

            let host_output1 = residual_add_bf16_device_inputs_output(&residual1, &delta1)?;
            assert_eq!(
                host_output1.backend,
                CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND
            );
            assert_eq!(host_output1.values[0], 2.25);
            assert_eq!(host_output1.values[values - 1], 2.25);

            let device_output0 =
                residual_add_bf16_device_inputs_device_output(&residual0, &delta1)?;
            let device_values0 = device_output0.copy_to_host_values()?;
            assert_eq!(
                device_output0.backend,
                CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND
            );
            assert_eq!(device_values0[0], 1.25);
            assert_eq!(device_values0[values - 1], 1.25);

            let device_output1 =
                residual_add_bf16_device_inputs_device_output(&residual1, &delta0)?;
            let device_values1 = device_output1.copy_to_host_values()?;
            assert_eq!(
                device_output1.backend,
                CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND
            );
            assert_eq!(device_values1[0], 2.5);
            assert_eq!(device_values1[values - 1], 2.5);

            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            assert!(slot.has_captured_graph(
                CoordinatorCudaGraphProgram::SparseBResidualAddBf16,
                CoordinatorCudaGraphSignature::residual_add_bf16(bytes)
            ));
            assert!(slot.graph_captures >= graph_captures_before);
            assert!(slot.graph_launches >= graph_launches_before + 4);
            Ok(())
        })();

        result
    }

    #[test]
    fn residual_add_bf16_rejects_mismatched_prefixes_before_backend_selection() {
        let residual = bf16_bytes(&[1.0, 2.0]);
        let delta = bf16_bytes(&[1.0]);
        let err = residual_add_prefix_bf16(&residual, &delta).unwrap_err();

        assert!(err
            .to_string()
            .contains("BF16 residual add byte length mismatch"));
    }

    #[test]
    fn device_bf16_output_from_f32_values_keeps_uploaded_delta_on_device() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);

        let result = (|| -> Result<()> {
            let residual = match device_bf16_output_from_f32_values(
                &[1.0, -2.0, 0.5, 4.0],
                1,
                4,
                "test residual device upload",
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            let delta = match device_bf16_output_from_f32_values(
                &[0.25, 3.0, -0.75, 1.0],
                1,
                4,
                "test delta device upload",
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            assert_eq!(
                residual.backend,
                CUDA_REFERENCE_DEVICE_BF16_HOST_UPLOAD_BACKEND
            );
            assert_eq!(
                delta.backend,
                CUDA_REFERENCE_DEVICE_BF16_HOST_UPLOAD_BACKEND
            );
            assert_eq!(residual.copy_to_host_values()?, vec![1.0, -2.0, 0.5, 4.0]);
            assert_eq!(delta.copy_to_host_values()?, vec![0.25, 3.0, -0.75, 1.0]);

            let output = residual_add_bf16_device_inputs_output(&residual, &delta)?;
            assert_eq!(output.backend, CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND);
            assert_eq!(output.device_output.rows, 1);
            assert_eq!(output.device_output.values_per_row, 4);
            assert_eq!(output.values, vec![1.25, 1.0, -0.25, 5.0]);
            assert_eq!(output.device_output.copy_to_host_values()?, output.values);
            Ok(())
        })();

        result
    }

    #[test]
    fn gather_rows_bf16_uses_cpu_reference_when_cuda_reference_is_not_enabled() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let src = bf16_bytes(&[
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0, //
        ]);
        let output = gather_rows_bf16(&src, &[2, 0], 3, 3).unwrap();

        assert_eq!(output.bytes, bf16_bytes(&[7.0, 8.0, 9.0, 1.0, 2.0, 3.0]));
        assert_eq!(output.backend, CPU_REFERENCE_GATHER_ROWS_BF16_BACKEND);
    }

    #[test]
    fn gather_rows_bf16_rejects_out_of_bounds_row_before_backend_selection() {
        let src = bf16_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let err = gather_rows_bf16(&src, &[2], 2, 2).unwrap_err();

        assert!(err.to_string().contains("row gather index 2 out of bounds"));
    }

    #[test]
    fn scatter_add_rows_bf16_to_f32_uses_cpu_reference_when_cuda_reference_is_not_enabled() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let src = bf16_bytes(&[
            1.0, 2.0, //
            3.0, 4.0, //
            -0.5, 0.25, //
        ]);
        let initial = [10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0];
        let output = scatter_add_rows_bf16_to_f32(&src, &[2, 0, 2], 3, 2, Some(&initial)).unwrap();

        assert_eq!(output.values, vec![13.0, 24.0, 30.0, 40.0, 50.5, 62.25]);
        assert_eq!(
            output.backend,
            CPU_REFERENCE_SCATTER_ADD_ROWS_BF16_TO_F32_BACKEND
        );
    }

    #[test]
    fn scatter_add_rows_bf16_to_f32_uses_coord_sparse_b_graph_slot_when_cuda_reference_is_enabled(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let src = bf16_bytes(&[
            1.0, 2.0, //
            3.0, 4.0, //
            -0.5, 0.25, //
        ]);
        let initial = [10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0];
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseB,
            LayerWaveMode::Prefill,
            3,
        )?;
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Sparse-B prefill graph key is registered");
        let (graph_captures_before, graph_launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.graph_captures, slot.graph_launches)
        };
        let output = match cuda_scatter_add_rows_bf16_to_f32(&src, &[2, 0, 2], 3, 2, Some(&initial))
        {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        assert_eq!(output.values, vec![13.0, 24.0, 30.0, 40.0, 50.5, 62.25]);
        assert_eq!(
            output.backend,
            CUDA_REFERENCE_SCATTER_ADD_ROWS_BF16_TO_F32_BACKEND
        );
        let slot = registry.slots[graph_index]
            .lock()
            .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
        assert!(slot.has_captured_graph(
            CoordinatorCudaGraphProgram::SparseBScatterAddBf16ToF32,
            CoordinatorCudaGraphSignature::scatter_add_bf16_to_f32(3, 2, 3)
        ));
        assert!(slot.graph_captures >= graph_captures_before);
        assert!(slot.graph_launches > graph_launches_before);
        Ok(())
    }

    #[test]
    fn scatter_add_rows_bf16_to_f32_graph_replays_with_updated_node_params() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseB,
            LayerWaveMode::Prefill,
            3,
        )?;
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Sparse-B prefill graph key is registered");
        let (graph_captures_before, graph_launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.graph_captures, slot.graph_launches)
        };

        let src0 = bf16_bytes(&[
            1.0, 2.0, //
            3.0, 4.0, //
            -0.5, 0.25, //
        ]);
        let initial0 = [10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0];
        let output0 =
            match cuda_scatter_add_rows_bf16_to_f32(&src0, &[2, 0, 2], 3, 2, Some(&initial0)) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
        assert_eq!(output0.values, vec![13.0, 24.0, 30.0, 40.0, 50.5, 62.25]);

        let src1 = bf16_bytes(&[
            0.5, 1.0, //
            1.5, -0.5, //
            2.0, 0.25, //
        ]);
        let initial1 = [0.25_f32; 6];
        let output1 =
            match cuda_scatter_add_rows_bf16_to_f32(&src1, &[0, 1, 0], 3, 2, Some(&initial1)) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
        assert_eq!(output1.values, vec![2.75, 1.5, 1.75, -0.25, 0.25, 0.25]);
        assert_ne!(output0.values, output1.values);

        let slot = registry.slots[graph_index]
            .lock()
            .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
        assert!(slot.has_captured_graph(
            CoordinatorCudaGraphProgram::SparseBScatterAddBf16ToF32,
            CoordinatorCudaGraphSignature::scatter_add_bf16_to_f32(3, 2, 3)
        ));
        assert!(slot.graph_captures >= graph_captures_before);
        assert!(slot.graph_launches >= graph_launches_before + 2);
        Ok(())
    }

    #[test]
    fn scatter_add_rows_bf16_to_f32_rejects_initial_shape_mismatch_before_backend_selection() {
        let src = bf16_bytes(&[1.0, 2.0]);
        let err = scatter_add_rows_bf16_to_f32(&src, &[0], 2, 2, Some(&[0.0, 0.0])).unwrap_err();

        assert!(err
            .to_string()
            .contains("row scatter-add initial value length mismatch"));
    }

    #[test]
    fn sparse_b_scatter_residual_add_bf16_uses_cpu_reference_when_cuda_reference_is_not_enabled() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let residual = [10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0];
        let initial_delta = [0.5_f32; 6];
        let partials = vec![
            bf16_bytes(&[
                1.0, 2.0, //
                3.0, 4.0, //
            ]),
            bf16_bytes(&[-0.5, 0.25]),
        ];
        let row_maps = vec![vec![2, 0], vec![2]];
        let output = sparse_b_scatter_residual_add_bf16(
            &residual,
            &initial_delta,
            &partials,
            &row_maps,
            3,
            2,
        )
        .unwrap();

        let expected = bf16_bytes(&[13.5, 24.5, 30.5, 40.5, 51.0, 62.75]);
        assert_eq!(output.output_bf16, expected);
        assert_eq!(output.values, bf16_values_to_f32(&expected));
        assert_eq!(output.delta_values, vec![3.5, 4.5, 0.5, 0.5, 1.0, 2.75]);
        assert!(output.device_output.is_none());
        assert_eq!(output.backend, CPU_REFERENCE_RESIDUAL_ADD_BF16_BACKEND);
    }

    #[test]
    fn sparse_b_scatter_residual_add_bf16_accepts_borrowed_row_maps() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let residual = [1.0_f32, 2.0, 3.0, 4.0];
        let initial_delta = [0.0_f32; 4];
        let partials = vec![bf16_bytes(&[0.5, 1.0]), bf16_bytes(&[2.0, 3.0])];
        let first_rows = [1_usize];
        let second_rows = [0_usize];
        let row_maps: [&[usize]; 2] = [&first_rows, &second_rows];
        let output = sparse_b_scatter_residual_add_bf16(
            &residual,
            &initial_delta,
            &partials,
            &row_maps,
            2,
            2,
        )
        .unwrap();

        assert_eq!(output.delta_values, vec![2.0, 3.0, 0.5, 1.0]);
        assert_eq!(output.values, vec![3.0, 5.0, 3.5, 5.0]);
        assert!(output.device_output.is_none());
    }

    #[test]
    fn sparse_b_scatter_residual_add_bf16_captures_coord_sparse_b_graph_exec() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);

        let result = (|| -> Result<()> {
            let graph_key = CoordinatorGraphKey::glm52_bf16(
                CoordinatorGraphShape::CoordSparseB,
                LayerWaveMode::Prefill,
                3,
            )?;
            let registry = match coordinator_cuda_graph_workspace_registry() {
                Ok(registry) => registry,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            let graph_index = registry
                .keys()
                .iter()
                .position(|key| key == &graph_key)
                .expect("Coord-Sparse-B prefill graph key is registered");
            let (graph_captures_before, graph_launches_before) = {
                let slot = registry.slots[graph_index]
                    .lock()
                    .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
                (slot.graph_captures, slot.graph_launches)
            };

            let residual = [10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0];
            let initial_delta = [0.5_f32; 6];
            let partials = vec![
                bf16_bytes(&[
                    1.0, 2.0, //
                    3.0, 4.0, //
                ]),
                bf16_bytes(&[-0.5, 0.25]),
            ];
            let row_maps = vec![vec![2, 0], vec![2]];
            let output = match sparse_b_scatter_residual_add_bf16(
                &residual,
                &initial_delta,
                &partials,
                &row_maps,
                3,
                2,
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };

            let expected = bf16_bytes(&[13.5, 24.5, 30.5, 40.5, 51.0, 62.75]);
            assert_eq!(output.output_bf16, expected);
            assert_eq!(output.values, bf16_values_to_f32(&expected));
            assert_eq!(output.delta_values, vec![3.5, 4.5, 0.5, 0.5, 1.0, 2.75]);
            assert_eq!(output.backend, CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND);
            let device_output = output
                .device_output
                .as_ref()
                .expect("CUDA Sparse-B output should preserve an owned device buffer");
            assert_eq!(device_output.rows, 3);
            assert_eq!(device_output.values_per_row, 2);
            assert_eq!(device_output.copy_to_host_values()?, output.values);
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            assert!(slot.has_captured_graph(
                CoordinatorCudaGraphProgram::CoordSparseBEnvelopeBf16,
                CoordinatorCudaGraphSignature::coord_sparse_b_envelope_bf16(16, 2)
            ));
            assert_eq!(
                slot.captured_graph_node_counts(
                    CoordinatorCudaGraphProgram::CoordSparseBEnvelopeBf16,
                    CoordinatorCudaGraphSignature::coord_sparse_b_envelope_bf16(16, 2)
                ),
                Some((2, 2, 0, 0))
            );
            assert!(slot.graph_captures >= graph_captures_before);
            assert!(slot.graph_launches > graph_launches_before);
            Ok(())
        })();

        result
    }

    #[test]
    fn sparse_b_scatter_residual_add_bf16_graph_replays_with_updated_node_params() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);

        let result = (|| -> Result<()> {
            let graph_key = CoordinatorGraphKey::glm52_bf16(
                CoordinatorGraphShape::CoordSparseB,
                LayerWaveMode::Prefill,
                3,
            )?;
            let registry = match coordinator_cuda_graph_workspace_registry() {
                Ok(registry) => registry,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            let graph_index = registry
                .keys()
                .iter()
                .position(|key| key == &graph_key)
                .expect("Coord-Sparse-B prefill graph key is registered");
            let (graph_captures_before, graph_launches_before) = {
                let slot = registry.slots[graph_index]
                    .lock()
                    .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
                (slot.graph_captures, slot.graph_launches)
            };

            let residual0 = [10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0];
            let initial_delta0 = [0.5_f32; 6];
            let partials0 = vec![
                bf16_bytes(&[
                    1.0, 2.0, //
                    3.0, 4.0, //
                ]),
                bf16_bytes(&[-0.5, 0.25]),
            ];
            let row_maps0 = vec![vec![2, 0], vec![2]];
            let output0 = match sparse_b_scatter_residual_add_bf16(
                &residual0,
                &initial_delta0,
                &partials0,
                &row_maps0,
                3,
                2,
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            let expected0 = bf16_bytes(&[13.5, 24.5, 30.5, 40.5, 51.0, 62.75]);
            assert_eq!(output0.output_bf16, expected0);
            assert_eq!(output0.delta_values, vec![3.5, 4.5, 0.5, 0.5, 1.0, 2.75]);

            let residual1 = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
            let initial_delta1 = [0.25_f32; 6];
            let partials1 = vec![
                bf16_bytes(&[
                    0.5, 1.0, //
                    1.5, -0.5, //
                ]),
                bf16_bytes(&[2.0, 0.25]),
            ];
            let row_maps1 = vec![vec![0, 1], vec![0]];
            let output1 = match sparse_b_scatter_residual_add_bf16(
                &residual1,
                &initial_delta1,
                &partials1,
                &row_maps1,
                3,
                2,
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            let expected1 = bf16_bytes(&[3.75, 3.5, 4.75, 3.75, 5.25, 6.25]);
            assert_eq!(output1.output_bf16, expected1);
            assert_eq!(
                output1.delta_values,
                vec![2.75, 1.5, 1.75, -0.25, 0.25, 0.25]
            );
            assert_ne!(output0.output_bf16, output1.output_bf16);

            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            assert!(slot.has_captured_graph(
                CoordinatorCudaGraphProgram::CoordSparseBEnvelopeBf16,
                CoordinatorCudaGraphSignature::coord_sparse_b_envelope_bf16(16, 2)
            ));
            assert!(slot.graph_captures >= graph_captures_before);
            assert!(slot.graph_launches >= graph_launches_before + 2);
            Ok(())
        })();

        result
    }

    #[test]
    fn sparse_b_coord_sparse_b_envelope_replays_same_bucket_when_active_counts_change() -> Result<()>
    {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);

        let result = (|| -> Result<()> {
            let row_width = 5;
            let graph_key = CoordinatorGraphKey::glm52_bf16(
                CoordinatorGraphShape::CoordSparseB,
                LayerWaveMode::Prefill,
                4,
            )?;
            let signature =
                CoordinatorCudaGraphSignature::coord_sparse_b_envelope_bf16(16, row_width);
            let registry = match coordinator_cuda_graph_workspace_registry() {
                Ok(registry) => registry,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            let graph_index = registry
                .keys()
                .iter()
                .position(|key| key == &graph_key)
                .expect("Coord-Sparse-B prefill graph key is registered");

            let residual0 = vec![10.0_f32; 2 * row_width];
            let initial_delta0 = vec![0.25_f32; 2 * row_width];
            let partials0 = vec![
                bf16_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0]),
                bf16_bytes(&[0.5, 1.0, 1.5, 2.0, 2.5]),
            ];
            let row_maps0 = vec![vec![1], vec![0]];
            let expected0 = super::cpu_sparse_b_scatter_residual_add_bf16(
                &residual0,
                &initial_delta0,
                &partials0.concat(),
                &[1_u32, 0],
                2,
                row_width,
            )?;
            let output0 = match sparse_b_scatter_residual_add_bf16(
                &residual0,
                &initial_delta0,
                &partials0,
                &row_maps0,
                2,
                row_width,
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            assert_eq!(output0.output_bf16, expected0.output_bf16);
            assert_eq!(output0.delta_values, expected0.delta_values);

            let (graph_captures_after_first, graph_launches_after_first) = {
                let slot = registry.slots[graph_index]
                    .lock()
                    .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
                assert!(slot.has_captured_graph(
                    CoordinatorCudaGraphProgram::CoordSparseBEnvelopeBf16,
                    signature
                ));
                assert_eq!(
                    slot.captured_graph_node_counts(
                        CoordinatorCudaGraphProgram::CoordSparseBEnvelopeBf16,
                        signature
                    ),
                    Some((2, 2, 0, 0))
                );
                (slot.graph_captures, slot.graph_launches)
            };

            let residual1 = vec![2.0_f32; 4 * row_width];
            let initial_delta1 = vec![0.0_f32; 4 * row_width];
            let partials1 = vec![
                bf16_bytes(&[
                    0.5, 0.5, 0.5, 0.5, 0.5, //
                    1.0, 1.0, 1.0, 1.0, 1.0, //
                    1.5, 1.5, 1.5, 1.5, 1.5, //
                ]),
                bf16_bytes(&[
                    2.0, 2.0, 2.0, 2.0, 2.0, //
                    -0.5, -0.5, -0.5, -0.5, -0.5, //
                ]),
            ];
            let row_maps1 = vec![vec![3, 0, 3], vec![1, 0]];
            let expected1 = super::cpu_sparse_b_scatter_residual_add_bf16(
                &residual1,
                &initial_delta1,
                &partials1.concat(),
                &[3_u32, 0, 3, 1, 0],
                4,
                row_width,
            )?;
            let output1 = match sparse_b_scatter_residual_add_bf16(
                &residual1,
                &initial_delta1,
                &partials1,
                &row_maps1,
                4,
                row_width,
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            assert_eq!(output1.output_bf16, expected1.output_bf16);
            assert_eq!(output1.delta_values, expected1.delta_values);
            assert_ne!(output0.output_bf16, output1.output_bf16);

            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            assert!(slot.has_captured_graph(
                CoordinatorCudaGraphProgram::CoordSparseBEnvelopeBf16,
                signature
            ));
            assert_eq!(slot.graph_captures, graph_captures_after_first);
            assert_eq!(slot.graph_launches, graph_launches_after_first + 1);
            Ok(())
        })();

        result
    }

    #[test]
    fn sparse_b_scatter_shared_residual_add_bf16_device_output_captures_two_node_graph(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);

        let result = (|| -> Result<()> {
            let graph_key = CoordinatorGraphKey::glm52_bf16(
                CoordinatorGraphShape::CoordSparseB,
                LayerWaveMode::Prefill,
                3,
            )?;
            let registry = match coordinator_cuda_graph_workspace_registry() {
                Ok(registry) => registry,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            let graph_index = registry
                .keys()
                .iter()
                .position(|key| key == &graph_key)
                .expect("Coord-Sparse-B prefill graph key is registered");

            let residual_bytes = bf16_bytes(&[10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0]);
            let shared_bytes = bf16_bytes(&[0.5_f32, 1.0, 1.5, 2.0, -0.25, 0.25]);
            let residual = match device_bf16_output_from_bf16_bytes(
                &residual_bytes,
                3,
                2,
                "test Sparse-B shared residual input",
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            let shared_delta = device_bf16_output_from_bf16_bytes(
                &shared_bytes,
                3,
                2,
                "test Sparse-B shared delta input",
            )?;
            let partials = vec![bf16_bytes(&[
                1.0_f32, 2.0, //
                -0.5, 0.25, //
                3.0, 4.0, //
            ])];
            let row_maps = vec![vec![2_usize, 0, 2]];

            let output = sparse_b_scatter_shared_residual_add_bf16_device_output(
                &residual,
                &shared_delta,
                &partials,
                &row_maps,
                3,
                2,
            )?;
            let expected = bf16_bytes(&[10.0_f32, 21.25, 31.5, 42.0, 53.75, 66.25]);
            assert_eq!(output.copy_to_host_bytes()?, expected);
            assert_eq!(output.rows, 3);
            assert_eq!(output.values_per_row, 2);

            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            let signature = CoordinatorCudaGraphSignature::coord_sparse_b_envelope_bf16(16, 2);
            assert!(slot.has_captured_graph(
                CoordinatorCudaGraphProgram::CoordSparseBSharedResidualEnvelopeBf16,
                signature
            ));
            assert_eq!(
                slot.captured_graph_node_counts(
                    CoordinatorCudaGraphProgram::CoordSparseBSharedResidualEnvelopeBf16,
                    signature
                ),
                Some((2, 2, 0, 0))
            );
            Ok(())
        })();

        result
    }

    #[test]
    fn streamed_sparse_b_scatter_shared_residual_add_accumulates_chunks() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);

        let residual0 = match device_bf16_output_from_bf16_bytes(
            &bf16_bytes(&[10.0_f32, 20.0, 30.0, 40.0]),
            2,
            2,
            "test streamed Sparse-B residual input 0",
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let residual1 = device_bf16_output_from_bf16_bytes(
            &bf16_bytes(&[50.0_f32, 60.0]),
            1,
            2,
            "test streamed Sparse-B residual input 1",
        )?;
        let shared0 = device_bf16_output_from_bf16_bytes(
            &bf16_bytes(&[0.5_f32, 1.0, 1.5, 2.0]),
            2,
            2,
            "test streamed Sparse-B shared delta input 0",
        )?;
        let shared1 = device_bf16_output_from_bf16_bytes(
            &bf16_bytes(&[-0.25_f32, 0.25]),
            1,
            2,
            "test streamed Sparse-B shared delta input 1",
        )?;
        let mut chunks = vec![
            (
                bf16_bytes(&[1.0_f32, 2.0, -0.5, 0.25]),
                vec![2, 0],
                glmrt_transport::ExpertV2Dtype::Bf16,
                4,
            ),
            (
                bf16_bytes(&[3.0_f32, 4.0]),
                vec![2],
                glmrt_transport::ExpertV2Dtype::Bf16,
                4,
            ),
        ]
        .into_iter();

        let segments = [
            StreamedSparseBResidualSegment {
                residual: &residual0,
                shared_delta: &shared0,
                row_start: 0,
                row_count: 2,
            },
            StreamedSparseBResidualSegment {
                residual: &residual1,
                shared_delta: &shared1,
                row_start: 2,
                row_count: 1,
            },
        ];
        let outputs = cuda_stream_sparse_b_scatter_shared_residual_add_bf16_device_outputs(
            &segments,
            3,
            2,
            || Ok(chunks.next()),
        )?;
        let expected = bf16_bytes(&[10.0_f32, 21.25, 31.5, 42.0, 53.75, 66.25]);
        let mut actual = Vec::new();
        for output in &outputs {
            actual.extend_from_slice(&output.copy_to_host_bytes()?);
            assert_eq!(output.values_per_row, 2);
        }
        assert_eq!(actual, expected);
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].rows, 2);
        assert_eq!(outputs[1].rows, 1);
        Ok(())
    }

    #[test]
    fn incremental_streamed_sparse_b_finalizes_ready_segments() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);

        let residual0 = match device_bf16_output_from_bf16_bytes(
            &bf16_bytes(&[10.0_f32, 20.0, 30.0, 40.0]),
            2,
            2,
            "test incremental Sparse-B residual input 0",
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let residual1 = device_bf16_output_from_bf16_bytes(
            &bf16_bytes(&[50.0_f32, 60.0]),
            1,
            2,
            "test incremental Sparse-B residual input 1",
        )?;
        let shared0 = device_bf16_output_from_bf16_bytes(
            &bf16_bytes(&[0.5_f32, 1.0, 1.5, 2.0]),
            2,
            2,
            "test incremental Sparse-B shared delta input 0",
        )?;
        let shared1 = device_bf16_output_from_bf16_bytes(
            &bf16_bytes(&[-0.25_f32, 0.25]),
            1,
            2,
            "test incremental Sparse-B shared delta input 1",
        )?;
        let segments = [
            StreamedSparseBResidualSegment {
                residual: &residual0,
                shared_delta: &shared0,
                row_start: 0,
                row_count: 2,
            },
            StreamedSparseBResidualSegment {
                residual: &residual1,
                shared_delta: &shared1,
                row_start: 2,
                row_count: 1,
            },
        ];

        let mut accumulator = CudaStreamedSparseBAccumulator::new(3, 2)?;
        accumulator.push_chunk(
            bf16_bytes(&[1.0_f32, 2.0, -0.5, 0.25]),
            &[2, 0],
            &[0],
            glmrt_transport::ExpertV2Dtype::Bf16,
            4,
        )?;
        assert!(!accumulator.segment_ready(0, 2)?);
        assert!(!accumulator.segment_ready(2, 1)?);

        let second_a = bf16_bytes(&[3.0_f32, 4.0]);
        let second_b = bf16_bytes(&[0.0_f32, 0.0]);
        accumulator.push_host_ordered_chunks(&[
            StreamedSparseBAccumulatorChunk {
                partial_output: &second_a,
                global_row_indices: &[2],
                completed_global_rows: &[2],
                output_dtype: glmrt_transport::ExpertV2Dtype::Bf16,
                output_row_stride_bytes: 4,
            },
            StreamedSparseBAccumulatorChunk {
                partial_output: &second_b,
                global_row_indices: &[1],
                completed_global_rows: &[1],
                output_dtype: glmrt_transport::ExpertV2Dtype::Bf16,
                output_row_stride_bytes: 4,
            },
        ])?;
        assert!(accumulator.segment_ready(0, 2)?);
        assert!(accumulator.segment_ready(2, 1)?);

        let output0 = accumulator.finalize_segment(&segments[0])?;
        let output1 = accumulator.finalize_segment(&segments[1])?;
        accumulator.validate_complete()?;
        let mut actual = output0.copy_to_host_bytes()?;
        actual.extend_from_slice(&output1.copy_to_host_bytes()?);
        assert_eq!(
            actual,
            bf16_bytes(&[10.0_f32, 21.25, 31.5, 42.0, 53.75, 66.25])
        );
        Ok(())
    }

    #[test]
    fn incremental_streamed_sparse_b_reuses_host_ordered_staging_slots() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);

        let residual = match device_bf16_output_from_bf16_bytes(
            &bf16_bytes(&[0.0_f32, 0.0]),
            1,
            2,
            "test incremental Sparse-B staging residual",
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let shared = device_bf16_output_from_bf16_bytes(
            &bf16_bytes(&[0.0_f32, 0.0]),
            1,
            2,
            "test incremental Sparse-B staging shared delta",
        )?;
        let payloads = (1..=5)
            .map(|value| bf16_bytes(&[value as f32, value as f32]))
            .collect::<Vec<_>>();
        let row_indices = [[0_usize]; 5];
        let completed_rows = [
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![0_usize],
        ];
        let chunks = payloads
            .iter()
            .zip(row_indices.iter())
            .zip(completed_rows.iter())
            .map(
                |((partial_output, global_row_indices), completed_global_rows)| {
                    StreamedSparseBAccumulatorChunk {
                        partial_output,
                        global_row_indices,
                        completed_global_rows,
                        output_dtype: glmrt_transport::ExpertV2Dtype::Bf16,
                        output_row_stride_bytes: 4,
                    }
                },
            )
            .collect::<Vec<_>>();

        let mut accumulator = CudaStreamedSparseBAccumulator::new(1, 2)?;
        accumulator.push_host_ordered_chunks(&chunks)?;
        let output = accumulator.finalize_segment(&StreamedSparseBResidualSegment {
            residual: &residual,
            shared_delta: &shared,
            row_start: 0,
            row_count: 1,
        })?;
        accumulator.validate_complete()?;
        assert_eq!(
            output.copy_to_host_bytes()?,
            bf16_bytes(&[15.0_f32, 15.0_f32])
        );
        Ok(())
    }

    #[test]
    fn rmsnorm_uses_cpu_reference_when_cuda_reference_is_not_enabled() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let output = rmsnorm_hidden(&[1.0, 2.0, 3.0], &[1.0, 0.5, 2.0], 1.0e-6).unwrap();

        assert_eq!(output.values.len(), 3);
        assert!(output.values.iter().all(|value| value.is_finite()));
        assert_eq!(output.backend, CPU_REFERENCE_RMSNORM_BACKEND);
    }

    #[test]
    fn rmsnorm_rejects_mismatched_inputs_before_backend_selection() {
        let err = rmsnorm_hidden(&[1.0, 2.0], &[1.0], 1.0e-6).unwrap_err();

        assert!(err
            .to_string()
            .contains("RMSNorm hidden/weight length mismatch"));
    }

    #[test]
    fn rmsnorm_hidden_bf16_uses_cpu_reference_when_cuda_reference_is_not_enabled() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let hidden = bf16_bytes(&[1.0, 2.0, 3.0, 4.0, -1.0, 0.5]);
        let weight = bf16_bytes(&[1.0, 0.5, 2.0]);
        let output = rmsnorm_hidden_bf16(&hidden, &weight, 2, 3, 1.0e-6).unwrap();

        assert_eq!(output.values.len(), 6);
        assert!(output.values.iter().all(|value| value.is_finite()));
        assert_eq!(output.backend, CPU_REFERENCE_RMSNORM_BF16_BACKEND);
        assert_ne!(output.values[0], output.values[3]);
    }

    #[test]
    fn rmsnorm_hidden_bf16_resident_weight_uses_cpu_reference_when_cuda_reference_is_not_enabled() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let hidden = bf16_bytes(&[1.0, 2.0, 3.0, 4.0, -1.0, 0.5]);
        let weight = bf16_bytes(&[1.0, 0.5, 2.0]);
        let output = rmsnorm_hidden_bf16_resident_weight(
            "model.layers.3.input_layernorm.weight",
            &hidden,
            &weight,
            2,
            3,
            1.0e-6,
        )
        .unwrap();

        assert_eq!(output.values.len(), 6);
        assert!(output.values.iter().all(|value| value.is_finite()));
        assert_eq!(output.backend, CPU_REFERENCE_RMSNORM_BF16_BACKEND);
        assert_ne!(output.values[0], output.values[3]);
    }

    #[test]
    fn rmsnorm_hidden_bf16_resident_weight_rejects_empty_weight_name() {
        let hidden = bf16_bytes(&[1.0, 2.0, 3.0]);
        let weight = bf16_bytes(&[1.0, 0.5, 2.0]);
        let err =
            rmsnorm_hidden_bf16_resident_weight("", &hidden, &weight, 1, 3, 1.0e-6).unwrap_err();

        assert!(err.to_string().contains("weight name must not be empty"));
    }

    #[test]
    fn rmsnorm_hidden_bf16_resident_weight_sparse_layer_uses_coord_sparse_a_graph_slot(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let fixture = full_width_rmsnorm_graph_update_fixture();
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseA,
            LayerWaveMode::Decode,
            fixture.rows,
        )?;
        let signature = CoordinatorCudaGraphSignature::rmsnorm_bf16(
            graph_key.row_bucket.row_capacity,
            fixture.hidden_dim,
            fixture.eps,
        );
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        let weight_name_a = format!(
            "model.layers.3.input_layernorm.weight.resident.test-a.{}.{}",
            std::process::id(),
            line!()
        );
        let weight_name_b = format!(
            "model.layers.3.input_layernorm.weight.resident.test-b.{}.{}",
            std::process::id(),
            line!()
        );

        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Sparse-A decode graph key is registered");
        let (acquisitions_before, graph_launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_launches)
        };

        let output_a = match cuda_rmsnorm_hidden_bf16_resident_weight(
            &weight_name_a,
            &fixture.hidden,
            &fixture.weight_a,
            fixture.rows,
            fixture.hidden_dim,
            fixture.eps,
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let output_b = match cuda_rmsnorm_hidden_bf16_resident_weight(
            &weight_name_b,
            &fixture.hidden,
            &fixture.weight_b,
            fixture.rows,
            fixture.hidden_dim,
            fixture.eps,
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let (acquisitions_after, graph_launches_after, has_graph) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (
                slot.acquisitions,
                slot.graph_launches,
                slot.has_captured_graph(CoordinatorCudaGraphProgram::LayerRmsNormBf16, signature),
            )
        };

        assert_eq!(
            output_a.backend,
            CUDA_REFERENCE_RMSNORM_BF16_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(
            output_b.backend,
            CUDA_REFERENCE_RMSNORM_BF16_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(output_a.values.len(), fixture.hidden_dim);
        assert_eq!(output_b.values.len(), fixture.hidden_dim);
        assert!(output_a.values.iter().all(|value| value.is_finite()));
        assert!(output_b.values.iter().all(|value| value.is_finite()));
        assert!((output_b.values[0] - (2.0 * output_a.values[0])).abs() < 2.0e-2);
        assert!((output_b.values[1] - (0.5 * output_a.values[1])).abs() < 2.0e-2);
        assert!(acquisitions_after > acquisitions_before);
        assert!(graph_launches_after >= graph_launches_before + 2);
        assert!(has_graph);
        Ok(())
    }

    #[test]
    fn rmsnorm_hidden_bf16_preloaded_resident_weight_requires_cuda_reference() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let hidden = bf16_bytes(&[1.0, 2.0, 3.0]);
        let err = rmsnorm_hidden_bf16_preloaded_resident_weight(
            "model.layers.3.input_layernorm.weight",
            &hidden,
            1,
            3,
            1.0e-6,
        )
        .unwrap_err();

        assert!(err
            .to_string()
            .contains("preloaded resident BF16 RMSNorm requires"));
    }

    #[test]
    fn rmsnorm_hidden_bf16_preloaded_resident_weight_rejects_shape_mismatch() {
        let hidden = bf16_bytes(&[1.0, 2.0]);
        let err = rmsnorm_hidden_bf16_preloaded_resident_weight(
            "model.layers.3.input_layernorm.weight",
            &hidden,
            1,
            3,
            1.0e-6,
        )
        .unwrap_err();

        assert!(err
            .to_string()
            .contains("preloaded BF16 RMSNorm hidden byte length mismatch"));
    }

    #[test]
    fn full_hidden_layer_tensor_graph_key_selects_dense_and_sparse_a() -> Result<()> {
        assert_eq!(
            glm52_layer_id_from_tensor_name("model.layers.0.input_layernorm.weight"),
            Some(0)
        );
        assert_eq!(
            glm52_layer_id_from_tensor_name("model.layers.3.post_attention_layernorm.weight"),
            Some(3)
        );
        assert_eq!(
            glm52_layer_id_from_tensor_name("model.embed_tokens.weight"),
            None
        );

        let dense = coord_layer_graph_key_for_full_hidden_rows(
            "model.layers.0.input_layernorm.weight",
            1,
            GLM52_HIDDEN_SIZE,
        )?
        .expect("dense layer graph key");
        assert_eq!(dense.shape, CoordinatorGraphShape::CoordDense);
        assert_eq!(dense.row_bucket.row_capacity, 1);

        let sparse = coord_layer_graph_key_for_full_hidden_rows(
            "model.layers.3.post_attention_layernorm.weight",
            17,
            GLM52_HIDDEN_SIZE,
        )?
        .expect("sparse layer graph key");
        assert_eq!(sparse.shape, CoordinatorGraphShape::CoordSparseA);
        assert_eq!(sparse.row_bucket.row_capacity, 32);

        let q_lora_norm = coord_layer_graph_key_for_full_hidden_rows(
            "model.layers.3.self_attn.q_a_layernorm.weight",
            1,
            GLM52_Q_LORA_RANK,
        )?
        .expect("q_a LayerNorm graph key");
        assert_eq!(q_lora_norm.shape, CoordinatorGraphShape::CoordSparseA);
        assert_eq!(q_lora_norm.row_bucket.row_capacity, 1);

        let terminal_norm =
            coord_layer_graph_key_for_full_hidden_rows("model.norm.weight", 1, GLM52_HIDDEN_SIZE)?
                .expect("terminal final RMSNorm graph key");
        assert_eq!(terminal_norm.shape, CoordinatorGraphShape::CoordDense);
        assert_eq!(terminal_norm.row_bucket.row_capacity, 1);

        let terminal_norm_test = coord_layer_graph_key_for_full_hidden_rows(
            "model.norm.weight.graph-test",
            17,
            GLM52_HIDDEN_SIZE,
        )?
        .expect("terminal final RMSNorm test graph key");
        assert_eq!(terminal_norm_test.shape, CoordinatorGraphShape::CoordDense);
        assert_eq!(terminal_norm_test.row_bucket.row_capacity, 32);

        assert!(coord_layer_graph_key_for_full_hidden_rows("model.norm.weight", 1, 3)?.is_none());
        assert!(coord_layer_graph_key_for_full_hidden_rows(
            "model.layers.3.input_layernorm.weight",
            1,
            3,
        )?
        .is_none());
        assert!(coord_layer_graph_key_for_full_hidden_rows(
            "model.embed_tokens.weight",
            1,
            GLM52_HIDDEN_SIZE,
        )?
        .is_none());
        Ok(())
    }

    #[test]
    fn attention_graph_key_is_dedicated_across_layer_families() -> Result<()> {
        let dense = coord_attention_graph_key_for_layer_rows(0, 231)?;
        let sparse = coord_attention_graph_key_for_layer_rows(3, 231)?;
        let compressed = coord_compressed_attention_decode_graph_key_for_layer(3)?;

        assert_eq!(dense.shape, CoordinatorGraphShape::CoordAttention);
        assert_eq!(sparse.shape, CoordinatorGraphShape::CoordAttention);
        assert_eq!(dense, sparse);
        assert_eq!(dense.row_bucket.row_capacity, 256);
        assert_eq!(
            compressed.shape,
            CoordinatorGraphShape::CoordCompressedAttention
        );
        assert_ne!(compressed, coord_attention_graph_key_for_layer_rows(3, 1)?);
        Ok(())
    }

    #[test]
    fn dsa_k_norm_graph_key_selects_layer_graph_shape() -> Result<()> {
        let dense = coord_layer_graph_key_for_dsa_k_norm_names(
            "model.layers.1.self_attn.indexer.k_norm.weight",
            "model.layers.1.self_attn.indexer.k_norm.bias",
            1,
        )?
        .expect("DSA k_norm dense decode graph key");
        assert_eq!(dense.shape, CoordinatorGraphShape::CoordDense);
        assert_eq!(dense.row_bucket.row_capacity, 1);

        let decode = coord_layer_graph_key_for_dsa_k_norm_names(
            "model.layers.22.self_attn.indexer.k_norm.weight",
            "model.layers.22.self_attn.indexer.k_norm.bias",
            1,
        )?
        .expect("DSA k_norm sparse decode graph key");
        assert_eq!(decode.shape, CoordinatorGraphShape::CoordSparseA);
        assert_eq!(decode.row_bucket.row_capacity, 1);

        let prefill = coord_layer_graph_key_for_dsa_k_norm_names(
            "model.layers.22.self_attn.indexer.k_norm.weight",
            "model.layers.22.self_attn.indexer.k_norm.bias",
            17,
        )?
        .expect("DSA k_norm sparse prefill graph key");
        assert_eq!(prefill.shape, CoordinatorGraphShape::CoordSparseA);
        assert_eq!(prefill.row_bucket.row_capacity, 32);

        assert!(coord_layer_graph_key_for_dsa_k_norm_names(
            "model.layers.22.self_attn.indexer.k_norm.weight",
            "model.layers.23.self_attn.indexer.k_norm.bias",
            1,
        )?
        .is_none());
        assert!(coord_layer_graph_key_for_dsa_k_norm_names(
            "model.layers.22.self_attn.q_a_layernorm.weight",
            "model.layers.22.self_attn.indexer.k_norm.bias",
            1,
        )?
        .is_none());
        Ok(())
    }

    #[test]
    fn rmsnorm_hidden_bf16_preloaded_resident_weight_dense_layer_uses_coord_dense_graph_slot(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordDense,
            LayerWaveMode::Decode,
            1,
        )?;
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        let mut hidden_values = vec![0.0_f32; GLM52_HIDDEN_SIZE];
        hidden_values[0] = 1.0;
        hidden_values[1] = -0.5;
        let hidden = bf16_bytes(&hidden_values);
        let weight = bf16_bytes(&vec![1.0_f32; GLM52_HIDDEN_SIZE]);
        let weight_name = format!(
            "model.layers.0.input_layernorm.weight.test.{}.{}",
            std::process::id(),
            line!()
        );
        match preload_resident_weight_from_host_staging(
            &weight_name,
            weight.len(),
            "test preloaded dense RMSNorm graph-slot weight",
            |staging| {
                staging.copy_from_slice(&weight);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        }

        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Dense decode graph key is registered");
        let acquisitions_before = registry.slots[graph_index]
            .lock()
            .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?
            .acquisitions;

        let output = match cuda_rmsnorm_hidden_bf16_preloaded_resident_weight(
            &weight_name,
            &hidden,
            1,
            GLM52_HIDDEN_SIZE,
            1.0e-6,
            weight.len(),
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let acquisitions_after = registry.slots[graph_index]
            .lock()
            .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?
            .acquisitions;

        assert_eq!(
            output.backend,
            CUDA_REFERENCE_RMSNORM_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(output.values.len(), GLM52_HIDDEN_SIZE);
        assert!(output.values.iter().all(|value| value.is_finite()));
        assert!(acquisitions_after > acquisitions_before);
        Ok(())
    }

    #[test]
    fn rmsnorm_hidden_bf16_preloaded_resident_weight_sparse_layer_uses_coord_sparse_a_graph_slot(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let fixture = full_width_rmsnorm_graph_update_fixture();
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseA,
            LayerWaveMode::Decode,
            fixture.rows,
        )?;
        let signature = CoordinatorCudaGraphSignature::rmsnorm_bf16(
            graph_key.row_bucket.row_capacity,
            fixture.hidden_dim,
            fixture.eps,
        );
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        let weight_name_a = format!(
            "model.layers.3.input_layernorm.weight.test-a.{}.{}",
            std::process::id(),
            line!()
        );
        let weight_name_b = format!(
            "model.layers.3.input_layernorm.weight.test-b.{}.{}",
            std::process::id(),
            line!()
        );
        match preload_resident_weight_from_host_staging(
            &weight_name_a,
            fixture.weight_a.len(),
            "test preloaded sparse RMSNorm graph-slot weight a",
            |staging| {
                staging.copy_from_slice(&fixture.weight_a);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        }
        match preload_resident_weight_from_host_staging(
            &weight_name_b,
            fixture.weight_b.len(),
            "test preloaded sparse RMSNorm graph-slot weight b",
            |staging| {
                staging.copy_from_slice(&fixture.weight_b);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        }

        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Sparse-A decode graph key is registered");
        let (acquisitions_before, graph_launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_launches)
        };

        let output_a = match cuda_rmsnorm_hidden_bf16_preloaded_resident_weight(
            &weight_name_a,
            &fixture.hidden,
            fixture.rows,
            fixture.hidden_dim,
            fixture.eps,
            fixture.weight_a.len(),
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let output_b = match cuda_rmsnorm_hidden_bf16_preloaded_resident_weight(
            &weight_name_b,
            &fixture.hidden,
            fixture.rows,
            fixture.hidden_dim,
            fixture.eps,
            fixture.weight_b.len(),
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let (acquisitions_after, graph_launches_after, has_graph) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (
                slot.acquisitions,
                slot.graph_launches,
                slot.has_captured_graph(CoordinatorCudaGraphProgram::LayerRmsNormBf16, signature),
            )
        };

        assert_eq!(
            output_a.backend,
            CUDA_REFERENCE_RMSNORM_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(
            output_b.backend,
            CUDA_REFERENCE_RMSNORM_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(output_a.values.len(), fixture.hidden_dim);
        assert_eq!(output_b.values.len(), fixture.hidden_dim);
        assert!(output_a.values.iter().all(|value| value.is_finite()));
        assert!(output_b.values.iter().all(|value| value.is_finite()));
        assert!((output_b.values[0] - (2.0 * output_a.values[0])).abs() < 2.0e-2);
        assert!((output_b.values[1] - (0.5 * output_a.values[1])).abs() < 2.0e-2);
        assert!(acquisitions_after > acquisitions_before);
        assert!(graph_launches_after >= graph_launches_before + 2);
        assert!(has_graph);
        Ok(())
    }

    #[test]
    fn rmsnorm_hidden_bf16_coord_sparse_a_graph_replays_same_bucket_when_rows_change() -> Result<()>
    {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let rows_first = 2_usize;
        let rows_second = 4_usize;
        let hidden_dim = GLM52_HIDDEN_SIZE;
        let eps = 1.0e-6_f32;
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseA,
            LayerWaveMode::Prefill,
            rows_first,
        )?;
        let signature = CoordinatorCudaGraphSignature::rmsnorm_bf16(
            graph_key.row_bucket.row_capacity,
            hidden_dim,
            eps,
        );
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Sparse-A prefill graph key is registered");

        let mut weight_values = vec![1.0_f32; hidden_dim];
        weight_values[0] = 1.5;
        weight_values[1] = 0.75;
        weight_values[2] = 1.25;
        weight_values[3] = 0.5;
        let weight = bf16_bytes(&weight_values);
        let weight_name = format!(
            "model.layers.3.input_layernorm.weight.prefill-replay.test.{}.{}",
            std::process::id(),
            line!()
        );
        match preload_resident_weight_from_host_staging(
            &weight_name,
            weight.len(),
            "test preloaded sparse RMSNorm row-bucket replay weight",
            |staging| {
                staging.copy_from_slice(&weight);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        }

        let assert_values_close = |actual: &[f32], expected: &[f32]| {
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected.iter()) {
                assert!(
                    (actual - expected).abs() <= 1.0e-2_f32.max(expected.abs() * 1.0e-2),
                    "actual={actual} expected={expected}"
                );
            }
        };

        let (captures_before, launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.graph_captures, slot.graph_launches)
        };

        let mut hidden_first_values = vec![0.0_f32; rows_first * hidden_dim];
        hidden_first_values[0] = 1.0;
        hidden_first_values[1] = 0.25;
        hidden_first_values[hidden_dim + 1] = -1.5;
        hidden_first_values[hidden_dim + 2] = 0.5;
        let hidden_first = bf16_bytes(&hidden_first_values);
        let expected_first =
            cpu_rmsnorm_hidden_bf16(&hidden_first, &weight, rows_first, hidden_dim, eps);
        let output_first = match cuda_rmsnorm_hidden_bf16_preloaded_resident_weight(
            &weight_name,
            &hidden_first,
            rows_first,
            hidden_dim,
            eps,
            weight.len(),
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        assert_eq!(
            output_first.backend,
            CUDA_REFERENCE_RMSNORM_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        assert_values_close(&output_first.values, &expected_first.values);

        let (captures_after_first, launches_after_first, has_graph) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (
                slot.graph_captures,
                slot.graph_launches,
                slot.has_captured_graph(CoordinatorCudaGraphProgram::LayerRmsNormBf16, signature),
            )
        };
        assert!(captures_after_first > captures_before);
        assert!(launches_after_first > launches_before);
        assert!(has_graph);

        let mut hidden_second_values = vec![0.0_f32; rows_second * hidden_dim];
        hidden_second_values[2] = 1.0;
        hidden_second_values[hidden_dim + 3] = -2.0;
        hidden_second_values[2 * hidden_dim] = 0.5;
        hidden_second_values[3 * hidden_dim + 1] = 1.25;
        let hidden_second = bf16_bytes(&hidden_second_values);
        let expected_second =
            cpu_rmsnorm_hidden_bf16(&hidden_second, &weight, rows_second, hidden_dim, eps);
        let output_second = match cuda_rmsnorm_hidden_bf16_preloaded_resident_weight(
            &weight_name,
            &hidden_second,
            rows_second,
            hidden_dim,
            eps,
            weight.len(),
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        assert_eq!(
            output_second.backend,
            CUDA_REFERENCE_RMSNORM_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        assert_values_close(&output_second.values, &expected_second.values);

        let (captures_after_second, launches_after_second) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.graph_captures, slot.graph_launches)
        };
        assert_eq!(captures_after_second, captures_after_first);
        assert!(launches_after_second > launches_after_first);
        Ok(())
    }

    #[test]
    fn rmsnorm_hidden_bf16_preloaded_resident_weight_device_output_replays_coord_sparse_a_graph(
    ) -> Result<()> {
        if !cuda_reference_kernels_test_enabled() {
            return Ok(());
        }
        let fixture = full_width_rmsnorm_graph_update_fixture();
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseA,
            LayerWaveMode::Decode,
            fixture.rows,
        )?;
        let signature = CoordinatorCudaGraphSignature::rmsnorm_bf16(
            graph_key.row_bucket.row_capacity,
            fixture.hidden_dim,
            fixture.eps,
        );
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        let weight_name_a = format!(
            "model.layers.3.input_layernorm.weight.device-output.test-a.{}.{}",
            std::process::id(),
            line!()
        );
        let weight_name_b = format!(
            "model.layers.3.input_layernorm.weight.device-output.test-b.{}.{}",
            std::process::id(),
            line!()
        );
        for (name, bytes, label) in [
            (
                weight_name_a.as_str(),
                fixture.weight_a.as_slice(),
                "test preloaded sparse RMSNorm device-output graph weight a",
            ),
            (
                weight_name_b.as_str(),
                fixture.weight_b.as_slice(),
                "test preloaded sparse RMSNorm device-output graph weight b",
            ),
        ] {
            match preload_resident_weight_from_host_staging(name, bytes.len(), label, |staging| {
                staging.copy_from_slice(bytes);
                Ok(())
            }) {
                Ok(()) => {}
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            }
        }

        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Sparse-A decode graph key is registered");
        let (acquisitions_before, graph_launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_launches)
        };

        let output_a = match rmsnorm_hidden_bf16_preloaded_resident_weight_device_output(
            &weight_name_a,
            &fixture.hidden,
            fixture.rows,
            fixture.hidden_dim,
            fixture.eps,
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let output_b = match rmsnorm_hidden_bf16_preloaded_resident_weight_device_output(
            &weight_name_b,
            &fixture.hidden,
            fixture.rows,
            fixture.hidden_dim,
            fixture.eps,
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let (acquisitions_after, graph_launches_after, has_graph) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (
                slot.acquisitions,
                slot.graph_launches,
                slot.has_captured_graph(CoordinatorCudaGraphProgram::LayerRmsNormBf16, signature),
            )
        };

        assert_eq!(
            output_a.backend,
            CUDA_REFERENCE_RMSNORM_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(
            output_b.backend,
            CUDA_REFERENCE_RMSNORM_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(output_a.rows, fixture.rows);
        assert_eq!(output_b.rows, fixture.rows);
        assert_eq!(output_a.values_per_row, fixture.hidden_dim);
        assert_eq!(output_b.values_per_row, fixture.hidden_dim);
        let values_a = output_a.copy_to_host_values()?;
        let values_b = output_b.copy_to_host_values()?;
        assert!(values_a.iter().all(|value| value.is_finite()));
        assert!(values_b.iter().all(|value| value.is_finite()));
        assert!((values_b[0] - (2.0 * values_a[0])).abs() < 2.0e-2);
        assert!((values_b[1] - (0.5 * values_a[1])).abs() < 2.0e-2);
        assert!(acquisitions_after > acquisitions_before);
        assert!(graph_launches_after >= graph_launches_before + 2);
        assert!(has_graph);
        Ok(())
    }

    #[test]
    fn rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output_replays_coord_sparse_a_graph(
    ) -> Result<()> {
        if !cuda_reference_kernels_test_enabled() {
            return Ok(());
        }
        let library = match cuda_native_library() {
            Ok(library) => library,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let fixture = full_width_rmsnorm_graph_update_fixture();
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseA,
            LayerWaveMode::Decode,
            fixture.rows,
        )?;
        let signature = CoordinatorCudaGraphSignature::rmsnorm_bf16(
            graph_key.row_bucket.row_capacity,
            fixture.hidden_dim,
            fixture.eps,
        );
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        let mut input_buffer = match library.alloc_device_buffer(fixture.hidden.len()) {
            Ok(buffer) => buffer,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        if let Err(error) = library.copy_h2d(input_buffer, &fixture.hidden) {
            let _ = library.free_device_buffer(&mut input_buffer);
            if cuda_allocation_unavailable(&error) {
                return Ok(());
            }
            return Err(error);
        }

        let weight_name_a = format!(
            "model.layers.3.input_layernorm.weight.device-input-output.test-a.{}.{}",
            std::process::id(),
            line!()
        );
        let weight_name_b = format!(
            "model.layers.3.input_layernorm.weight.device-input-output.test-b.{}.{}",
            std::process::id(),
            line!()
        );
        for (name, bytes, label) in [
            (
                weight_name_a.as_str(),
                fixture.weight_a.as_slice(),
                "test preloaded sparse RMSNorm device-input-output graph weight a",
            ),
            (
                weight_name_b.as_str(),
                fixture.weight_b.as_slice(),
                "test preloaded sparse RMSNorm device-input-output graph weight b",
            ),
        ] {
            match preload_resident_weight_from_host_staging(name, bytes.len(), label, |staging| {
                staging.copy_from_slice(bytes);
                Ok(())
            }) {
                Ok(()) => {}
                Err(error) => {
                    let _ = library.free_device_buffer(&mut input_buffer);
                    if cuda_allocation_unavailable(&error) {
                        return Ok(());
                    }
                    return Err(error);
                }
            }
        }

        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Sparse-A decode graph key is registered");
        let (acquisitions_before, graph_launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_launches)
        };

        let output_result = (|| -> Result<(DeviceBf16Output, DeviceBf16Output)> {
            let output_a = rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output(
                &weight_name_a,
                input_buffer,
                fixture.rows,
                fixture.hidden_dim,
                fixture.eps,
            )?;
            let output_b = rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output(
                &weight_name_b,
                input_buffer,
                fixture.rows,
                fixture.hidden_dim,
                fixture.eps,
            )?;
            Ok((output_a, output_b))
        })();
        let free_result = library.free_device_buffer(&mut input_buffer);
        let (output_a, output_b) = match output_result {
            Ok(outputs) => outputs,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        free_result?;
        let (acquisitions_after, graph_launches_after, has_graph) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (
                slot.acquisitions,
                slot.graph_launches,
                slot.has_captured_graph(CoordinatorCudaGraphProgram::LayerRmsNormBf16, signature),
            )
        };

        assert_eq!(
            output_a.backend,
            CUDA_REFERENCE_RMSNORM_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(
            output_b.backend,
            CUDA_REFERENCE_RMSNORM_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(output_a.rows, fixture.rows);
        assert_eq!(output_b.rows, fixture.rows);
        assert_eq!(output_a.values_per_row, fixture.hidden_dim);
        assert_eq!(output_b.values_per_row, fixture.hidden_dim);
        let values_a = output_a.copy_to_host_values()?;
        let values_b = output_b.copy_to_host_values()?;
        assert!(values_a.iter().all(|value| value.is_finite()));
        assert!(values_b.iter().all(|value| value.is_finite()));
        assert!((values_b[0] - (2.0 * values_a[0])).abs() < 2.0e-2);
        assert!((values_b[1] - (0.5 * values_a[1])).abs() < 2.0e-2);
        assert!(acquisitions_after > acquisitions_before);
        assert!(graph_launches_after >= graph_launches_before + 2);
        assert!(has_graph);
        Ok(())
    }

    #[test]
    fn terminal_final_rmsnorm_device_input_output_replays_coord_dense_graph() -> Result<()> {
        if !cuda_reference_kernels_test_enabled() {
            return Ok(());
        }
        let library = match cuda_native_library() {
            Ok(library) => library,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let fixture = full_width_rmsnorm_graph_update_fixture();
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordDense,
            LayerWaveMode::Decode,
            fixture.rows,
        )?;
        let signature = CoordinatorCudaGraphSignature::rmsnorm_bf16(
            graph_key.row_bucket.row_capacity,
            fixture.hidden_dim,
            fixture.eps,
        );
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        let mut input_buffer = match library.alloc_device_buffer(fixture.hidden.len()) {
            Ok(buffer) => buffer,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        if let Err(error) = library.copy_h2d(input_buffer, &fixture.hidden) {
            let _ = library.free_device_buffer(&mut input_buffer);
            if cuda_allocation_unavailable(&error) {
                return Ok(());
            }
            return Err(error);
        }

        let weight_name = format!(
            "model.norm.weight.device-input-output.test.{}.{}",
            std::process::id(),
            line!()
        );
        match preload_resident_weight_from_host_staging(
            &weight_name,
            fixture.weight_a.len(),
            "test preloaded terminal final RMSNorm graph weight",
            |staging| {
                staging.copy_from_slice(&fixture.weight_a);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) => {
                let _ = library.free_device_buffer(&mut input_buffer);
                if cuda_allocation_unavailable(&error) {
                    return Ok(());
                }
                return Err(error);
            }
        }

        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Dense decode graph key is registered");
        let (acquisitions_before, graph_launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_launches)
        };

        let output_result = rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output(
            &weight_name,
            input_buffer,
            fixture.rows,
            fixture.hidden_dim,
            fixture.eps,
        );
        let free_result = library.free_device_buffer(&mut input_buffer);
        let output = match output_result {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        free_result?;
        let (acquisitions_after, graph_launches_after, has_graph) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (
                slot.acquisitions,
                slot.graph_launches,
                slot.has_captured_graph(CoordinatorCudaGraphProgram::LayerRmsNormBf16, signature),
            )
        };

        assert_eq!(
            output.backend,
            CUDA_REFERENCE_RMSNORM_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(output.rows, fixture.rows);
        assert_eq!(output.values_per_row, fixture.hidden_dim);
        let values = output.copy_to_host_values()?;
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values[1] > values[0]);
        assert!(acquisitions_after > acquisitions_before);
        assert!(graph_launches_after > graph_launches_before);
        assert!(has_graph);
        Ok(())
    }

    #[test]
    fn layer_norm_affine_f32_bf16_preloaded_resident_requires_cuda_reference() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let err = layer_norm_affine_f32_bf16_preloaded_resident_weight_bias(
            "model.layers.22.self_attn.indexer.k_norm.weight",
            "model.layers.22.self_attn.indexer.k_norm.bias",
            &[1.0, 2.0, 3.0],
            1,
            3,
            1.0e-6,
        )
        .unwrap_err();

        assert!(err
            .to_string()
            .contains("preloaded resident BF16 affine LayerNorm requires"));
    }

    #[test]
    fn layer_norm_affine_f32_bf16_preloaded_resident_rejects_shape_mismatch() {
        let err = layer_norm_affine_f32_bf16_preloaded_resident_weight_bias(
            "model.layers.22.self_attn.indexer.k_norm.weight",
            "model.layers.22.self_attn.indexer.k_norm.bias",
            &[1.0, 2.0],
            1,
            3,
            1.0e-6,
        )
        .unwrap_err();

        assert!(err
            .to_string()
            .contains("preloaded BF16 affine LayerNorm value length mismatch"));
    }

    #[test]
    fn layer_norm_affine_bf16_preloaded_resident_device_input_rejects_shape_mismatch() {
        let input_buffer = GlmrtDeviceBuffer {
            ptr: std::ptr::NonNull::<c_void>::dangling().as_ptr(),
            bytes: 2,
            device_id: 0,
            flags: 0,
        };
        let err = match layer_norm_affine_bf16_preloaded_resident_weight_bias_device_input_output(
            "model.layers.22.self_attn.indexer.k_norm.weight",
            "model.layers.22.self_attn.indexer.k_norm.bias",
            input_buffer,
            1,
            3,
            1.0e-6,
        ) {
            Ok(_) => panic!("BF16 affine LayerNorm device input shape mismatch passed"),
            Err(error) => error,
        };

        assert!(err
            .to_string()
            .contains("preloaded BF16 affine LayerNorm device input byte length mismatch"));
    }

    #[test]
    fn layer_norm_affine_f32_bf16_preloaded_dsa_k_norm_uses_coord_sparse_a_graph_slot() -> Result<()>
    {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseA,
            LayerWaveMode::Decode,
            1,
        )?;
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        let values = [1.0_f32, -1.0, 0.5, 2.0];
        let weight_a = bf16_bytes(&[1.0_f32, 0.5, -1.0, 2.0]);
        let bias_a = bf16_bytes(&[0.0_f32, 0.25, 0.5, -0.5]);
        let weight_b = bf16_bytes(&[2.0_f32, 0.25, 0.5, -1.0]);
        let bias_b = bf16_bytes(&[-0.25_f32, 0.5, -0.5, 0.75]);
        let weight_name_a = format!(
            "model.layers.22.self_attn.indexer.k_norm.weight.test.a.{}.{}",
            std::process::id(),
            line!()
        );
        let bias_name_a = format!(
            "model.layers.22.self_attn.indexer.k_norm.bias.test.a.{}.{}",
            std::process::id(),
            line!()
        );
        let weight_name_b = format!(
            "model.layers.22.self_attn.indexer.k_norm.weight.test.b.{}.{}",
            std::process::id(),
            line!()
        );
        let bias_name_b = format!(
            "model.layers.22.self_attn.indexer.k_norm.bias.test.b.{}.{}",
            std::process::id(),
            line!()
        );
        match preload_resident_weight_from_host_staging(
            &weight_name_a,
            weight_a.len(),
            "test preloaded DSA k_norm graph-slot weight A",
            |staging| {
                staging.copy_from_slice(&weight_a);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        }
        match preload_resident_weight_from_host_staging(
            &bias_name_a,
            bias_a.len(),
            "test preloaded DSA k_norm graph-slot bias A",
            |staging| {
                staging.copy_from_slice(&bias_a);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        }
        match preload_resident_weight_from_host_staging(
            &weight_name_b,
            weight_b.len(),
            "test preloaded DSA k_norm graph-slot weight B",
            |staging| {
                staging.copy_from_slice(&weight_b);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        }
        match preload_resident_weight_from_host_staging(
            &bias_name_b,
            bias_b.len(),
            "test preloaded DSA k_norm graph-slot bias B",
            |staging| {
                staging.copy_from_slice(&bias_b);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        }

        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Sparse-A decode graph key is registered");
        let signature = CoordinatorCudaGraphSignature::layernorm_affine(
            graph_key.row_bucket.row_capacity,
            4,
            1.0e-6,
        );
        let (acquisitions_before, graph_launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_launches)
        };

        let output_result = (|| -> Result<(LayerNormAffineOutput, LayerNormAffineOutput)> {
            let output_a = cuda_layer_norm_affine_f32_bf16_preloaded_resident_weight_bias(
                &weight_name_a,
                &bias_name_a,
                &values,
                1,
                4,
                1.0e-6,
                weight_a.len(),
            )?;
            let output_b = cuda_layer_norm_affine_f32_bf16_preloaded_resident_weight_bias(
                &weight_name_b,
                &bias_name_b,
                &values,
                1,
                4,
                1.0e-6,
                weight_b.len(),
            )?;
            Ok((output_a, output_b))
        })();
        let (output_a, output_b) = match output_result {
            Ok(outputs) => outputs,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let (acquisitions_after, graph_launches_after, has_graph) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (
                slot.acquisitions,
                slot.graph_launches,
                slot.has_captured_graph(
                    CoordinatorCudaGraphProgram::LayerNormAffineF32Bf16,
                    signature,
                ),
            )
        };

        assert_eq!(
            output_a.backend,
            CUDA_REFERENCE_LAYER_NORM_AFFINE_F32_BF16_PRELOADED_RESIDENT_BACKEND
        );
        assert_eq!(
            output_b.backend,
            CUDA_REFERENCE_LAYER_NORM_AFFINE_F32_BF16_PRELOADED_RESIDENT_BACKEND
        );
        assert_eq!(output_a.values.len(), values.len());
        assert_eq!(output_b.values.len(), values.len());
        assert!(output_a.values.iter().all(|value| value.is_finite()));
        assert!(output_b.values.iter().all(|value| value.is_finite()));
        assert_ne!(output_a.values, output_b.values);
        assert!(acquisitions_after > acquisitions_before);
        assert!(graph_launches_after >= graph_launches_before + 2);
        assert!(has_graph);
        Ok(())
    }

    #[test]
    fn layer_norm_affine_f32_bf16_coord_sparse_a_graph_replays_same_bucket_when_rows_change(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let rows_first = 2_usize;
        let rows_second = 4_usize;
        let hidden_dim = 4_usize;
        let eps = 1.0e-6_f32;
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseA,
            LayerWaveMode::Prefill,
            rows_first,
        )?;
        let signature = CoordinatorCudaGraphSignature::layernorm_affine(
            graph_key.row_bucket.row_capacity,
            hidden_dim,
            eps,
        );
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Sparse-A prefill graph key is registered");

        let weight_values = [1.0_f32, 0.5, -1.0, 2.0];
        let bias_values = [0.0_f32, 0.25, 0.5, -0.5];
        let weight = bf16_bytes(&weight_values);
        let bias = bf16_bytes(&bias_values);
        let weight_name = format!(
            "model.layers.22.self_attn.indexer.k_norm.weight.prefill-replay.test.{}.{}",
            std::process::id(),
            line!()
        );
        let bias_name = format!(
            "model.layers.22.self_attn.indexer.k_norm.bias.prefill-replay.test.{}.{}",
            std::process::id(),
            line!()
        );
        for (name, bytes, label) in [
            (
                weight_name.as_str(),
                weight.as_slice(),
                "test preloaded DSA k_norm row-bucket replay weight",
            ),
            (
                bias_name.as_str(),
                bias.as_slice(),
                "test preloaded DSA k_norm row-bucket replay bias",
            ),
        ] {
            match preload_resident_weight_from_host_staging(name, bytes.len(), label, |staging| {
                staging.copy_from_slice(bytes);
                Ok(())
            }) {
                Ok(()) => {}
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            }
        }

        let expected = |values: &[f32], rows: usize| -> Vec<f32> {
            let mut output = vec![0.0_f32; rows * hidden_dim];
            for row in 0..rows {
                let row_start = row * hidden_dim;
                let row_values = &values[row_start..row_start + hidden_dim];
                let mean = row_values.iter().sum::<f32>() / hidden_dim as f32;
                let variance = row_values
                    .iter()
                    .map(|value| {
                        let centered = value - mean;
                        centered * centered
                    })
                    .sum::<f32>()
                    / hidden_dim as f32;
                let scale = (variance + eps).sqrt().recip();
                for col in 0..hidden_dim {
                    output[row_start + col] =
                        (values[row_start + col] - mean) * scale * weight_values[col]
                            + bias_values[col];
                }
            }
            output
        };
        let assert_values_close = |actual: &[f32], expected: &[f32]| {
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected.iter()) {
                assert!(
                    (actual - expected).abs() <= 1.0e-3_f32.max(expected.abs() * 1.0e-3),
                    "actual={actual} expected={expected}"
                );
            }
        };

        let (captures_before, launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.graph_captures, slot.graph_launches)
        };

        let values_first = [1.0_f32, -1.0, 0.5, 2.0, 0.25, 1.5, -0.75, 0.0];
        let expected_first = expected(&values_first, rows_first);
        let output_first = match cuda_layer_norm_affine_f32_bf16_preloaded_resident_weight_bias(
            &weight_name,
            &bias_name,
            &values_first,
            rows_first,
            hidden_dim,
            eps,
            weight.len(),
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        assert_eq!(
            output_first.backend,
            CUDA_REFERENCE_LAYER_NORM_AFFINE_F32_BF16_PRELOADED_RESIDENT_BACKEND
        );
        assert_values_close(&output_first.values, &expected_first);

        let (captures_after_first, launches_after_first, has_graph) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (
                slot.graph_captures,
                slot.graph_launches,
                slot.has_captured_graph(
                    CoordinatorCudaGraphProgram::LayerNormAffineF32Bf16,
                    signature,
                ),
            )
        };
        assert!(captures_after_first > captures_before);
        assert!(launches_after_first > launches_before);
        assert!(has_graph);

        let values_second = [
            -0.5_f32, 1.25, 0.75, -1.5, 2.0, 0.0, -1.0, 1.0, 0.5, -0.25, 1.5, -2.0, 3.0, 1.0, -0.5,
            0.25,
        ];
        let expected_second = expected(&values_second, rows_second);
        let output_second = match cuda_layer_norm_affine_f32_bf16_preloaded_resident_weight_bias(
            &weight_name,
            &bias_name,
            &values_second,
            rows_second,
            hidden_dim,
            eps,
            weight.len(),
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        assert_eq!(
            output_second.backend,
            CUDA_REFERENCE_LAYER_NORM_AFFINE_F32_BF16_PRELOADED_RESIDENT_BACKEND
        );
        assert_values_close(&output_second.values, &expected_second);

        let (captures_after_second, launches_after_second) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.graph_captures, slot.graph_launches)
        };
        assert_eq!(captures_after_second, captures_after_first);
        assert!(launches_after_second > launches_after_first);
        Ok(())
    }

    #[test]
    fn layer_norm_affine_bf16_preloaded_dsa_k_norm_device_output_uses_coord_sparse_a_graph_slot(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let library = match cuda_native_library() {
            Ok(library) => library,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseA,
            LayerWaveMode::Decode,
            1,
        )?;
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        let input_values = [1.0_f32, -1.0, 0.5, 2.0];
        let input = bf16_bytes(&input_values);
        let weight_a = bf16_bytes(&[1.0_f32, 0.5, -1.0, 2.0]);
        let bias_a = bf16_bytes(&[0.0_f32, 0.25, 0.5, -0.5]);
        let weight_b = bf16_bytes(&[2.0_f32, 0.25, 0.5, -1.0]);
        let bias_b = bf16_bytes(&[-0.25_f32, 0.5, -0.5, 0.75]);
        let weight_name_a = format!(
            "model.layers.22.self_attn.indexer.k_norm.weight.device-output.test.a.{}.{}",
            std::process::id(),
            line!()
        );
        let bias_name_a = format!(
            "model.layers.22.self_attn.indexer.k_norm.bias.device-output.test.a.{}.{}",
            std::process::id(),
            line!()
        );
        let weight_name_b = format!(
            "model.layers.22.self_attn.indexer.k_norm.weight.device-output.test.b.{}.{}",
            std::process::id(),
            line!()
        );
        let bias_name_b = format!(
            "model.layers.22.self_attn.indexer.k_norm.bias.device-output.test.b.{}.{}",
            std::process::id(),
            line!()
        );
        match preload_resident_weight_from_host_staging(
            &weight_name_a,
            weight_a.len(),
            "test preloaded BF16 DSA k_norm device-output graph-slot weight A",
            |staging| {
                staging.copy_from_slice(&weight_a);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        }
        match preload_resident_weight_from_host_staging(
            &bias_name_a,
            bias_a.len(),
            "test preloaded BF16 DSA k_norm device-output graph-slot bias A",
            |staging| {
                staging.copy_from_slice(&bias_a);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        }
        match preload_resident_weight_from_host_staging(
            &weight_name_b,
            weight_b.len(),
            "test preloaded BF16 DSA k_norm device-output graph-slot weight B",
            |staging| {
                staging.copy_from_slice(&weight_b);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        }
        match preload_resident_weight_from_host_staging(
            &bias_name_b,
            bias_b.len(),
            "test preloaded BF16 DSA k_norm device-output graph-slot bias B",
            |staging| {
                staging.copy_from_slice(&bias_b);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        }

        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Sparse-A decode graph key is registered");
        let signature = CoordinatorCudaGraphSignature::layernorm_affine(
            graph_key.row_bucket.row_capacity,
            4,
            1.0e-6,
        );
        let (acquisitions_before, graph_launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_launches)
        };

        let mut input_buffer = match library.alloc_device_buffer(input.len()) {
            Ok(buffer) => buffer,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        if let Err(error) = library.copy_h2d(input_buffer, &input) {
            let _ = library.free_device_buffer(&mut input_buffer);
            if cuda_allocation_unavailable(&error) {
                return Ok(());
            }
            return Err(error);
        }

        let output_result = (|| -> Result<(DeviceBf16Output, DeviceBf16Output)> {
            let output_a =
                cuda_layer_norm_affine_bf16_preloaded_resident_weight_bias_device_input_output(
                    &weight_name_a,
                    &bias_name_a,
                    input_buffer,
                    1,
                    4,
                    1.0e-6,
                    weight_a.len(),
                )?;
            let output_b =
                cuda_layer_norm_affine_bf16_preloaded_resident_weight_bias_device_input_output(
                    &weight_name_b,
                    &bias_name_b,
                    input_buffer,
                    1,
                    4,
                    1.0e-6,
                    weight_b.len(),
                )?;
            Ok((output_a, output_b))
        })();
        library.free_device_buffer(&mut input_buffer)?;
        let (output_a, output_b) = match output_result {
            Ok(outputs) => outputs,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let (acquisitions_after, graph_launches_after, has_graph) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (
                slot.acquisitions,
                slot.graph_launches,
                slot.has_captured_graph(
                    CoordinatorCudaGraphProgram::LayerNormAffineBf16,
                    signature,
                ),
            )
        };

        assert_eq!(
            output_a.backend,
            CUDA_REFERENCE_LAYER_NORM_AFFINE_BF16_PRELOADED_RESIDENT_BACKEND
        );
        assert_eq!(
            output_b.backend,
            CUDA_REFERENCE_LAYER_NORM_AFFINE_BF16_PRELOADED_RESIDENT_BACKEND
        );
        assert_eq!(output_a.rows, 1);
        assert_eq!(output_b.rows, 1);
        assert_eq!(output_a.values_per_row, 4);
        assert_eq!(output_b.values_per_row, 4);
        assert!(acquisitions_after > acquisitions_before);
        assert!(graph_launches_after >= graph_launches_before + 2);
        assert!(has_graph);

        let expected_a = cuda_layer_norm_affine_f32_bf16_preloaded_resident_weight_bias(
            &weight_name_a,
            &bias_name_a,
            &bf16_values_to_f32(&input),
            1,
            4,
            1.0e-6,
            weight_a.len(),
        )?;
        let expected_b = cuda_layer_norm_affine_f32_bf16_preloaded_resident_weight_bias(
            &weight_name_b,
            &bias_name_b,
            &bf16_values_to_f32(&input),
            1,
            4,
            1.0e-6,
            weight_b.len(),
        )?;
        let expected_a_bf16 = bf16_bytes(&expected_a.values);
        let expected_b_bf16 = bf16_bytes(&expected_b.values);
        let actual_a = output_a.copy_to_host_values()?;
        let actual_b = output_b.copy_to_host_values()?;
        assert_eq!(actual_a.len(), expected_a.values.len());
        assert_eq!(actual_b.len(), expected_b.values.len());
        assert_ne!(actual_a, actual_b);
        for (index, actual_value) in actual_a.iter().enumerate() {
            let expected_value = bf16_value(&expected_a_bf16, index);
            assert!(
                (actual_value - expected_value).abs() <= 1.0e-5,
                "BF16 DSA k_norm device output A {index} mismatch: actual={actual_value} expected={expected_value}"
            );
        }
        for (index, actual_value) in actual_b.iter().enumerate() {
            let expected_value = bf16_value(&expected_b_bf16, index);
            assert!(
                (actual_value - expected_value).abs() <= 1.0e-5,
                "BF16 DSA k_norm device output B {index} mismatch: actual={actual_value} expected={expected_value}"
            );
        }
        Ok(())
    }

    #[test]
    fn layer_norm_affine_bf16_preloaded_dense_dsa_k_norm_uses_coord_dense_graph_slot() -> Result<()>
    {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let library = match cuda_native_library() {
            Ok(library) => library,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let rows = 9_usize;
        let hidden_dim = GLM52_DSA_INDEX_HEAD_DIM;
        let eps = 1.0e-6_f32;
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordDense,
            LayerWaveMode::Prefill,
            rows,
        )?;
        let signature = CoordinatorCudaGraphSignature::layernorm_affine(
            graph_key.row_bucket.row_capacity,
            hidden_dim,
            eps,
        );
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Dense prefill graph key is registered");

        let input_values = patterned_bf16_values(rows * hidden_dim, 0.002, 3.0);
        let weight_values = patterned_bf16_values(hidden_dim, 0.001, 5.0)
            .into_iter()
            .map(|value| 1.0 + value)
            .collect::<Vec<_>>();
        let bias_values = patterned_bf16_values(hidden_dim, 0.0005, 7.0);
        let input = bf16_bytes(&input_values);
        let weight = bf16_bytes(&weight_values);
        let bias = bf16_bytes(&bias_values);
        let weight_name = format!(
            "model.layers.1.self_attn.indexer.k_norm.weight.dense-device-output.test.{}.{}",
            std::process::id(),
            line!()
        );
        let bias_name = format!(
            "model.layers.1.self_attn.indexer.k_norm.bias.dense-device-output.test.{}.{}",
            std::process::id(),
            line!()
        );
        for (name, bytes, label) in [
            (
                weight_name.as_str(),
                weight.as_slice(),
                "test preloaded dense BF16 DSA k_norm weight",
            ),
            (
                bias_name.as_str(),
                bias.as_slice(),
                "test preloaded dense BF16 DSA k_norm bias",
            ),
        ] {
            match preload_resident_weight_from_host_staging(name, bytes.len(), label, |staging| {
                staging.copy_from_slice(bytes);
                Ok(())
            }) {
                Ok(()) => {}
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            }
        }

        let (acquisitions_before, graph_launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_launches)
        };
        let mut input_buffer = match library.alloc_device_buffer(input.len()) {
            Ok(buffer) => buffer,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        if let Err(error) = library.copy_h2d(input_buffer, &input) {
            let _ = library.free_device_buffer(&mut input_buffer);
            if cuda_allocation_unavailable(&error) {
                return Ok(());
            }
            return Err(error);
        }

        let output_result =
            cuda_layer_norm_affine_bf16_preloaded_resident_weight_bias_device_input_output(
                &weight_name,
                &bias_name,
                input_buffer,
                rows,
                hidden_dim,
                eps,
                weight.len(),
            );
        library.free_device_buffer(&mut input_buffer)?;
        let output = match output_result {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let (acquisitions_after, graph_launches_after, has_graph) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (
                slot.acquisitions,
                slot.graph_launches,
                slot.has_captured_graph(
                    CoordinatorCudaGraphProgram::LayerNormAffineBf16,
                    signature,
                ),
            )
        };
        assert_eq!(
            output.backend,
            CUDA_REFERENCE_LAYER_NORM_AFFINE_BF16_PRELOADED_RESIDENT_BACKEND
        );
        assert_eq!(output.rows, rows);
        assert_eq!(output.values_per_row, hidden_dim);
        assert!(acquisitions_after > acquisitions_before);
        assert!(graph_launches_after > graph_launches_before);
        assert!(has_graph);

        let expected = cuda_layer_norm_affine_f32_bf16_preloaded_resident_weight_bias(
            &weight_name,
            &bias_name,
            &bf16_values_to_f32(&input),
            rows,
            hidden_dim,
            eps,
            weight.len(),
        )?;
        assert_bf16_values_close(
            &output.copy_to_host_bytes()?,
            &bf16_bytes(&expected.values),
            1.0e-5,
        );
        Ok(())
    }

    #[test]
    fn rmsnorm_hidden_bf16_rejects_shape_mismatch_before_backend_selection() {
        let hidden = bf16_bytes(&[1.0, 2.0, 3.0]);
        let weight = bf16_bytes(&[1.0, 0.5]);
        let err = rmsnorm_hidden_bf16(&hidden, &weight, 1, 3, 1.0e-6).unwrap_err();

        assert!(err
            .to_string()
            .contains("BF16 RMSNorm weight byte length mismatch"));
    }

    #[test]
    fn linear_rows_uses_cpu_reference_when_cuda_reference_is_not_enabled() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let output = linear_rows(
            &[1.0, 2.0, 3.0, 4.0],
            &[0.5, 1.0, -1.0, 2.0, 1.0, -0.5],
            Some(&[0.25, -0.25, 1.0]),
            2,
            2,
            3,
        )
        .unwrap();

        assert_eq!(output.values, vec![2.75, 2.75, 1.0, 5.75, 4.75, 2.0]);
        assert_eq!(output.backend, CPU_REFERENCE_LINEAR_BACKEND);
    }

    #[test]
    fn linear_rows_rejects_shape_mismatch_before_backend_selection() {
        let err = linear_rows(&[1.0, 2.0], &[1.0, 2.0, 3.0], None, 1, 2, 2).unwrap_err();

        assert!(err.to_string().contains("linear weight length mismatch"));
    }

    #[test]
    fn linear_rows_bf16_uses_cpu_reference_when_cuda_reference_is_not_enabled() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let input = bf16_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let weight = bf16_bytes(&[0.5, 1.0, -1.0, 2.0, 1.0, -0.5]);
        let bias = bf16_bytes(&[0.25, -0.25, 1.0]);
        let output = linear_rows_bf16(&input, &weight, Some(&bias), 2, 2, 3).unwrap();

        assert_eq!(output.values, vec![2.75, 2.75, 1.0, 5.75, 4.75, 2.0]);
        assert_eq!(output.backend, CPU_REFERENCE_LINEAR_BF16_BACKEND);
    }

    #[test]
    fn linear_rows_bf16_resident_weight_uses_cpu_reference_when_cuda_reference_is_not_enabled() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let input = bf16_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let weight = bf16_bytes(&[0.5, 1.0, -1.0, 2.0, 1.0, -0.5]);
        let bias = bf16_bytes(&[0.25, -0.25, 1.0]);
        let output = linear_rows_bf16_resident_weight(
            "model.layers.0.mlp.gate_proj.weight[rows=0..3]",
            &input,
            &weight,
            Some(&bias),
            2,
            2,
            3,
        )
        .unwrap();

        assert_eq!(output.values, vec![2.75, 2.75, 1.0, 5.75, 4.75, 2.0]);
        assert_eq!(output.backend, CPU_REFERENCE_LINEAR_BF16_BACKEND);
    }

    #[test]
    fn linear_rows_bf16_resident_weight_rejects_empty_weight_name() {
        let input = bf16_bytes(&[1.0, 2.0]);
        let weight = bf16_bytes(&[1.0, 2.0]);
        let err = linear_rows_bf16_resident_weight("", &input, &weight, None, 1, 2, 1).unwrap_err();

        assert!(err.to_string().contains("weight name must not be empty"));
    }

    #[test]
    fn linear_rows_bf16_resident_attention_weight_sparse_layer_uses_coord_sparse_a_graph_slot(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let fixture = full_width_linear_graph_update_fixture();
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseA,
            LayerWaveMode::Decode,
            fixture.rows,
        )?;
        let signature = CoordinatorCudaGraphSignature::linear_bf16(
            graph_key.row_bucket.row_capacity,
            fixture.input_dim,
            fixture.output_dim,
            true,
        );
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        let weight_name_a = format!(
            "model.layers.3.self_attn.q_a_proj.weight.resident.test-a.{}.{}",
            std::process::id(),
            line!()
        );
        let weight_name_b = format!(
            "model.layers.3.self_attn.q_a_proj.weight.resident.test-b.{}.{}",
            std::process::id(),
            line!()
        );

        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Sparse-A decode graph key is registered");
        let (acquisitions_before, graph_launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_launches)
        };

        let output_a = match cuda_linear_rows_bf16_resident_weight(
            &weight_name_a,
            &fixture.input,
            &fixture.weight_a,
            Some(&fixture.bias_a),
            fixture.rows,
            fixture.input_dim,
            fixture.output_dim,
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let output_b = match cuda_linear_rows_bf16_resident_weight(
            &weight_name_b,
            &fixture.input,
            &fixture.weight_b,
            Some(&fixture.bias_b),
            fixture.rows,
            fixture.input_dim,
            fixture.output_dim,
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let (acquisitions_after, graph_launches_after, has_graph) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (
                slot.acquisitions,
                slot.graph_launches,
                slot.has_captured_graph(CoordinatorCudaGraphProgram::LayerLinearBf16, signature),
            )
        };

        assert_eq!(
            output_a.backend,
            CUDA_REFERENCE_LINEAR_BF16_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(
            output_b.backend,
            CUDA_REFERENCE_LINEAR_BF16_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(output_a.values.len(), fixture.output_dim);
        assert_eq!(output_b.values.len(), fixture.output_dim);
        for (actual, expected) in output_a.values.iter().zip(fixture.expected_a.iter()) {
            assert!((actual - expected).abs() < 1.0e-3);
        }
        for (actual, expected) in output_b.values.iter().zip(fixture.expected_b.iter()) {
            assert!((actual - expected).abs() < 1.0e-3);
        }
        assert!(acquisitions_after > acquisitions_before);
        assert!(graph_launches_after >= graph_launches_before + 2);
        assert!(has_graph);
        Ok(())
    }

    #[test]
    fn linear_rows_bf16_preloaded_resident_weight_requires_cuda_reference() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let input = bf16_bytes(&[1.0, 2.0]);
        let err = linear_rows_bf16_preloaded_resident_weight(
            "model.layers.0.mlp.gate_proj.weight",
            &input,
            None,
            1,
            2,
            1,
            3,
        )
        .unwrap_err();

        assert!(err
            .to_string()
            .contains("preloaded resident BF16 linear requires"));
    }

    #[test]
    fn linear_rows_bf16_preloaded_resident_weight_rejects_prefix_past_full_rows() {
        let input = bf16_bytes(&[1.0, 2.0]);
        let err = linear_rows_bf16_preloaded_resident_weight(
            "model.layers.0.mlp.gate_proj.weight",
            &input,
            None,
            1,
            2,
            4,
            3,
        )
        .unwrap_err();

        assert!(err.to_string().contains("exceeds full_output_dim"));
    }

    #[test]
    fn linear_weight_graph_key_selects_only_coordinator_envelope_weights() -> Result<()> {
        let dense_attention =
            coord_linear_graph_key_for_weight_name("model.layers.0.self_attn.q_a_proj.weight", 1)?
                .expect("dense attention projection graph key");
        assert_eq!(dense_attention.shape, CoordinatorGraphShape::CoordDense);
        assert_eq!(dense_attention.row_bucket.row_capacity, 1);

        let sparse_attention =
            coord_linear_graph_key_for_weight_name("model.layers.3.self_attn.o_proj.weight", 17)?
                .expect("sparse attention projection graph key");
        assert_eq!(sparse_attention.shape, CoordinatorGraphShape::CoordSparseA);
        assert_eq!(sparse_attention.row_bucket.row_capacity, 32);

        let dense_mlp =
            coord_linear_graph_key_for_weight_name("model.layers.0.mlp.gate_proj.weight", 1)?
                .expect("dense MLP projection graph key");
        assert_eq!(dense_mlp.shape, CoordinatorGraphShape::CoordDense);

        assert!(coord_linear_graph_key_for_weight_name(
            "model.layers.3.mlp.shared_experts.gate_proj.weight",
            1,
        )?
        .is_none());
        assert!(
            coord_linear_graph_key_for_weight_name("model.layers.3.mlp.gate.weight", 1)?.is_none()
        );
        assert!(coord_linear_graph_key_for_weight_name("lm_head.weight", 1)?.is_none());
        Ok(())
    }

    #[test]
    fn linear_rows_bf16_preloaded_attention_weight_dense_layer_uses_coord_dense_graph_slot(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordDense,
            LayerWaveMode::Decode,
            1,
        )?;
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        let mut input_values = vec![0.0_f32; GLM52_HIDDEN_SIZE];
        input_values[0] = 1.0;
        let input = bf16_bytes(&input_values);
        let mut weight_values = vec![0.0_f32; 3 * GLM52_HIDDEN_SIZE];
        weight_values[0] = 1.0;
        weight_values[GLM52_HIDDEN_SIZE] = 2.0;
        weight_values[2 * GLM52_HIDDEN_SIZE] = -1.0;
        let weight = bf16_bytes(&weight_values);
        let weight_name = format!(
            "model.layers.0.self_attn.q_a_proj.weight.test.{}.{}",
            std::process::id(),
            line!()
        );
        match preload_resident_weight_from_host_staging(
            &weight_name,
            weight.len(),
            "test preloaded dense linear graph-slot weight",
            |staging| {
                staging.copy_from_slice(&weight);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        }

        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Dense decode graph key is registered");
        let acquisitions_before = registry.slots[graph_index]
            .lock()
            .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?
            .acquisitions;

        let output = match cuda_linear_rows_bf16_preloaded_resident_weight(
            &weight_name,
            &input,
            None,
            1,
            GLM52_HIDDEN_SIZE,
            3,
            LinearResidentView {
                full_bytes: weight.len(),
                offset_bytes: 0,
                view_bytes: weight.len(),
            },
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let acquisitions_after = registry.slots[graph_index]
            .lock()
            .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?
            .acquisitions;

        assert_eq!(
            output.backend,
            CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(output.values.len(), 3);
        assert!((output.values[0] - 1.0).abs() < 1.0e-3);
        assert!((output.values[1] - 2.0).abs() < 1.0e-3);
        assert!((output.values[2] + 1.0).abs() < 1.0e-3);
        assert!(acquisitions_after > acquisitions_before);
        Ok(())
    }

    #[test]
    fn linear_rows_bf16_preloaded_attention_weight_sparse_layer_uses_coord_sparse_a_graph_slot(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let fixture = full_width_linear_graph_update_fixture();
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseA,
            LayerWaveMode::Decode,
            fixture.rows,
        )?;
        let signature = CoordinatorCudaGraphSignature::linear_bf16(
            graph_key.row_bucket.row_capacity,
            fixture.input_dim,
            fixture.output_dim,
            true,
        );
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        let weight_name_a = format!(
            "model.layers.3.self_attn.q_a_proj.weight.test-a.{}.{}",
            std::process::id(),
            line!()
        );
        let weight_name_b = format!(
            "model.layers.3.self_attn.q_a_proj.weight.test-b.{}.{}",
            std::process::id(),
            line!()
        );

        match preload_resident_weight_from_host_staging(
            &weight_name_a,
            fixture.weight_a.len(),
            "test preloaded sparse linear graph-slot weight a",
            |staging| {
                staging.copy_from_slice(&fixture.weight_a);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        }
        match preload_resident_weight_from_host_staging(
            &weight_name_b,
            fixture.weight_b.len(),
            "test preloaded sparse linear graph-slot weight b",
            |staging| {
                staging.copy_from_slice(&fixture.weight_b);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        }

        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Sparse-A decode graph key is registered");
        let (acquisitions_before, graph_launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_launches)
        };

        let output_a = match cuda_linear_rows_bf16_preloaded_resident_weight(
            &weight_name_a,
            &fixture.input,
            Some(&fixture.bias_a),
            fixture.rows,
            fixture.input_dim,
            fixture.output_dim,
            LinearResidentView {
                full_bytes: fixture.weight_a.len(),
                offset_bytes: 0,
                view_bytes: fixture.weight_a.len(),
            },
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let output_b = match cuda_linear_rows_bf16_preloaded_resident_weight(
            &weight_name_b,
            &fixture.input,
            Some(&fixture.bias_b),
            fixture.rows,
            fixture.input_dim,
            fixture.output_dim,
            LinearResidentView {
                full_bytes: fixture.weight_b.len(),
                offset_bytes: 0,
                view_bytes: fixture.weight_b.len(),
            },
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let (acquisitions_after, graph_launches_after, has_graph) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (
                slot.acquisitions,
                slot.graph_launches,
                slot.has_captured_graph(CoordinatorCudaGraphProgram::LayerLinearBf16, signature),
            )
        };

        assert_eq!(
            output_a.backend,
            CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(
            output_b.backend,
            CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(output_a.values.len(), fixture.output_dim);
        assert_eq!(output_b.values.len(), fixture.output_dim);
        for (actual, expected) in output_a.values.iter().zip(fixture.expected_a.iter()) {
            assert!((actual - expected).abs() < 1.0e-3);
        }
        for (actual, expected) in output_b.values.iter().zip(fixture.expected_b.iter()) {
            assert!((actual - expected).abs() < 1.0e-3);
        }
        assert!(acquisitions_after > acquisitions_before);
        assert!(graph_launches_after >= graph_launches_before + 2);
        assert!(has_graph);
        Ok(())
    }

    #[test]
    fn linear_rows_bf16_coord_sparse_a_graph_replays_same_bucket_when_rows_change() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let rows_first = 2_usize;
        let rows_second = 4_usize;
        let input_dim = GLM52_HIDDEN_SIZE;
        let output_dim = 3_usize;
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseA,
            LayerWaveMode::Prefill,
            rows_first,
        )?;
        let signature = CoordinatorCudaGraphSignature::linear_bf16(
            graph_key.row_bucket.row_capacity,
            input_dim,
            output_dim,
            false,
        );
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Sparse-A prefill graph key is registered");

        let mut weight_values = vec![0.0_f32; output_dim * input_dim];
        weight_values[0] = 0.5;
        weight_values[input_dim + 1] = -1.5;
        weight_values[2 * input_dim] = 2.0;
        weight_values[2 * input_dim + 1] = 0.25;
        let weight = bf16_bytes(&weight_values);
        let weight_name = format!(
            "model.layers.3.self_attn.q_a_proj.weight.prefill-replay.test.{}.{}",
            std::process::id(),
            line!()
        );
        match preload_resident_weight_from_host_staging(
            &weight_name,
            weight.len(),
            "test preloaded sparse linear row-bucket replay weight",
            |staging| {
                staging.copy_from_slice(&weight);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        }

        let assert_values_close = |actual: &[f32], expected: &[f32]| {
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected.iter()) {
                assert!(
                    (actual - expected).abs() <= 1.0e-3_f32.max(expected.abs() * 1.0e-3),
                    "actual={actual} expected={expected}"
                );
            }
        };

        let (captures_before, launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.graph_captures, slot.graph_launches)
        };

        let mut input_first_values = vec![0.0_f32; rows_first * input_dim];
        input_first_values[0] = 1.0;
        input_first_values[1] = -2.0;
        input_first_values[input_dim] = 0.5;
        input_first_values[input_dim + 1] = 1.0;
        let input_first = bf16_bytes(&input_first_values);
        let expected_first = cpu_linear_rows_bf16(
            &input_first,
            &weight,
            None,
            rows_first,
            input_dim,
            output_dim,
        );
        let output_first = match cuda_linear_rows_bf16_preloaded_resident_weight(
            &weight_name,
            &input_first,
            None,
            rows_first,
            input_dim,
            output_dim,
            LinearResidentView {
                full_bytes: weight.len(),
                offset_bytes: 0,
                view_bytes: weight.len(),
            },
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        assert_eq!(
            output_first.backend,
            CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        assert_values_close(&output_first.values, &expected_first.values);

        let (captures_after_first, launches_after_first, has_graph) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (
                slot.graph_captures,
                slot.graph_launches,
                slot.has_captured_graph(CoordinatorCudaGraphProgram::LayerLinearBf16, signature),
            )
        };
        assert!(captures_after_first > captures_before);
        assert!(launches_after_first > launches_before);
        assert!(has_graph);

        let mut input_second_values = vec![0.0_f32; rows_second * input_dim];
        input_second_values[0] = -1.0;
        input_second_values[1] = 0.5;
        input_second_values[input_dim] = 2.0;
        input_second_values[input_dim + 1] = -0.25;
        input_second_values[2 * input_dim] = 0.75;
        input_second_values[2 * input_dim + 1] = 1.5;
        input_second_values[3 * input_dim] = -0.5;
        input_second_values[3 * input_dim + 1] = -1.0;
        let input_second = bf16_bytes(&input_second_values);
        let expected_second = cpu_linear_rows_bf16(
            &input_second,
            &weight,
            None,
            rows_second,
            input_dim,
            output_dim,
        );
        let output_second = match cuda_linear_rows_bf16_preloaded_resident_weight(
            &weight_name,
            &input_second,
            None,
            rows_second,
            input_dim,
            output_dim,
            LinearResidentView {
                full_bytes: weight.len(),
                offset_bytes: 0,
                view_bytes: weight.len(),
            },
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        assert_eq!(
            output_second.backend,
            CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        assert_values_close(&output_second.values, &expected_second.values);

        let (captures_after_second, launches_after_second) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.graph_captures, slot.graph_launches)
        };
        assert_eq!(captures_after_second, captures_after_first);
        assert!(launches_after_second > launches_after_first);
        Ok(())
    }

    #[test]
    fn linear_rows_bf16_preloaded_attention_weight_device_input_uses_coord_sparse_a_graph_slot(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let library = match cuda_native_library() {
            Ok(library) => library,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let fixture = full_width_linear_graph_update_fixture();
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseA,
            LayerWaveMode::Decode,
            fixture.rows,
        )?;
        let signature = CoordinatorCudaGraphSignature::linear_bf16(
            graph_key.row_bucket.row_capacity,
            fixture.input_dim,
            fixture.output_dim,
            true,
        );
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        let mut input_buffer = match library.alloc_device_buffer(fixture.input.len()) {
            Ok(buffer) => buffer,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        if let Err(error) = library.copy_h2d(input_buffer, &fixture.input) {
            let _ = library.free_device_buffer(&mut input_buffer);
            if cuda_allocation_unavailable(&error) {
                return Ok(());
            }
            return Err(error);
        }

        let weight_name_a = format!(
            "model.layers.3.self_attn.o_proj.weight.device-input.test-a.{}.{}",
            std::process::id(),
            line!()
        );
        let weight_name_b = format!(
            "model.layers.3.self_attn.o_proj.weight.device-input.test-b.{}.{}",
            std::process::id(),
            line!()
        );
        for (name, bytes, label) in [
            (
                weight_name_a.as_str(),
                fixture.weight_a.as_slice(),
                "test preloaded sparse device-input linear graph-slot weight a",
            ),
            (
                weight_name_b.as_str(),
                fixture.weight_b.as_slice(),
                "test preloaded sparse device-input linear graph-slot weight b",
            ),
        ] {
            match preload_resident_weight_from_host_staging(name, bytes.len(), label, |staging| {
                staging.copy_from_slice(bytes);
                Ok(())
            }) {
                Ok(()) => {}
                Err(error) => {
                    let _ = library.free_device_buffer(&mut input_buffer);
                    if cuda_allocation_unavailable(&error) {
                        return Ok(());
                    }
                    return Err(error);
                }
            }
        }

        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Sparse-A decode graph key is registered");
        let (acquisitions_before, graph_launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_launches)
        };

        let output_result = (|| -> Result<(LinearOutput, LinearOutput)> {
            let output_a = cuda_linear_rows_bf16_preloaded_resident_weight_device_input(
                &weight_name_a,
                input_buffer,
                Some(&fixture.bias_a),
                fixture.rows,
                fixture.input_dim,
                fixture.output_dim,
                LinearResidentView {
                    full_bytes: fixture.weight_a.len(),
                    offset_bytes: 0,
                    view_bytes: fixture.weight_a.len(),
                },
            )?;
            let output_b = cuda_linear_rows_bf16_preloaded_resident_weight_device_input(
                &weight_name_b,
                input_buffer,
                Some(&fixture.bias_b),
                fixture.rows,
                fixture.input_dim,
                fixture.output_dim,
                LinearResidentView {
                    full_bytes: fixture.weight_b.len(),
                    offset_bytes: 0,
                    view_bytes: fixture.weight_b.len(),
                },
            )?;
            Ok((output_a, output_b))
        })();
        let free_result = library.free_device_buffer(&mut input_buffer);
        let (output_a, output_b) = match output_result {
            Ok(outputs) => outputs,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        free_result?;
        let (acquisitions_after, graph_launches_after, has_graph) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (
                slot.acquisitions,
                slot.graph_launches,
                slot.has_captured_graph(CoordinatorCudaGraphProgram::LayerLinearBf16, signature),
            )
        };

        assert_eq!(
            output_a.backend,
            CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(
            output_b.backend,
            CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(output_a.values.len(), fixture.output_dim);
        assert_eq!(output_b.values.len(), fixture.output_dim);
        for (actual, expected) in output_a.values.iter().zip(fixture.expected_a.iter()) {
            assert!((actual - expected).abs() < 1.0e-3);
        }
        for (actual, expected) in output_b.values.iter().zip(fixture.expected_b.iter()) {
            assert!((actual - expected).abs() < 1.0e-3);
        }
        assert!(acquisitions_after > acquisitions_before);
        assert!(graph_launches_after >= graph_launches_before + 2);
        assert!(has_graph);
        Ok(())
    }

    #[test]
    fn linear_rows_bf16_preloaded_attention_weight_device_output_replays_coord_sparse_a_graph(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let library = match cuda_native_library() {
            Ok(library) => library,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let fixture = full_width_linear_graph_update_fixture();
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseA,
            LayerWaveMode::Decode,
            fixture.rows,
        )?;
        let signature = CoordinatorCudaGraphSignature::linear_bf16(
            graph_key.row_bucket.row_capacity,
            fixture.input_dim,
            fixture.output_dim,
            true,
        );
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        let mut input_buffer = match library.alloc_device_buffer(fixture.input.len()) {
            Ok(buffer) => buffer,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        if let Err(error) = library.copy_h2d(input_buffer, &fixture.input) {
            let _ = library.free_device_buffer(&mut input_buffer);
            if cuda_allocation_unavailable(&error) {
                return Ok(());
            }
            return Err(error);
        }

        let weight_name_a = format!(
            "model.layers.3.self_attn.o_proj.weight.device-output.test-a.{}.{}",
            std::process::id(),
            line!()
        );
        let weight_name_b = format!(
            "model.layers.3.self_attn.o_proj.weight.device-output.test-b.{}.{}",
            std::process::id(),
            line!()
        );
        for (name, bytes, label) in [
            (
                weight_name_a.as_str(),
                fixture.weight_a.as_slice(),
                "test preloaded sparse device-output linear graph-slot weight a",
            ),
            (
                weight_name_b.as_str(),
                fixture.weight_b.as_slice(),
                "test preloaded sparse device-output linear graph-slot weight b",
            ),
        ] {
            match preload_resident_weight_from_host_staging(name, bytes.len(), label, |staging| {
                staging.copy_from_slice(bytes);
                Ok(())
            }) {
                Ok(()) => {}
                Err(error) => {
                    let _ = library.free_device_buffer(&mut input_buffer);
                    if cuda_allocation_unavailable(&error) {
                        return Ok(());
                    }
                    return Err(error);
                }
            }
        }

        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Sparse-A decode graph key is registered");
        let (acquisitions_before, graph_launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_launches)
        };

        let output_result = (|| -> Result<(DeviceBf16Output, DeviceBf16Output)> {
            let output_a = cuda_linear_rows_bf16_preloaded_resident_weight_device_output(
                &weight_name_a,
                input_buffer,
                Some(&fixture.bias_a),
                fixture.rows,
                fixture.input_dim,
                fixture.output_dim,
                LinearResidentView {
                    full_bytes: fixture.weight_a.len(),
                    offset_bytes: 0,
                    view_bytes: fixture.weight_a.len(),
                },
            )?;
            let output_b = cuda_linear_rows_bf16_preloaded_resident_weight_device_output(
                &weight_name_b,
                input_buffer,
                Some(&fixture.bias_b),
                fixture.rows,
                fixture.input_dim,
                fixture.output_dim,
                LinearResidentView {
                    full_bytes: fixture.weight_b.len(),
                    offset_bytes: 0,
                    view_bytes: fixture.weight_b.len(),
                },
            )?;
            Ok((output_a, output_b))
        })();
        let free_result = library.free_device_buffer(&mut input_buffer);
        let (output_a, output_b) = match output_result {
            Ok(outputs) => outputs,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        free_result?;
        let (acquisitions_after, graph_launches_after, has_graph) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (
                slot.acquisitions,
                slot.graph_launches,
                slot.has_captured_graph(CoordinatorCudaGraphProgram::LayerLinearBf16, signature),
            )
        };

        assert_eq!(
            output_a.backend,
            CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(
            output_b.backend,
            CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(output_a.rows, fixture.rows);
        assert_eq!(output_b.rows, fixture.rows);
        assert_eq!(output_a.values_per_row, fixture.output_dim);
        assert_eq!(output_b.values_per_row, fixture.output_dim);
        let values_a = output_a.copy_to_host_values()?;
        let values_b = output_b.copy_to_host_values()?;
        for (actual, expected) in values_a.iter().zip(fixture.expected_a.iter()) {
            assert!((actual - expected).abs() < 1.0e-3);
        }
        for (actual, expected) in values_b.iter().zip(fixture.expected_b.iter()) {
            assert!((actual - expected).abs() < 1.0e-3);
        }
        assert!(acquisitions_after > acquisitions_before);
        assert!(graph_launches_after >= graph_launches_before + 2);
        assert!(has_graph);
        Ok(())
    }

    #[test]
    fn preloaded_resident_device_output_query_chain_matches_host_readbacks() -> Result<()> {
        if !cuda_reference_kernels_test_enabled() {
            return Ok(());
        }

        let mut hidden_values = vec![0.0_f32; GLM52_HIDDEN_SIZE];
        hidden_values[0] = 1.0;
        hidden_values[1] = -2.0;
        hidden_values[2] = 0.5;
        let hidden_bf16 = bf16_bytes(&hidden_values);
        let input_norm_weight = bf16_bytes(&vec![1.0_f32; GLM52_HIDDEN_SIZE]);
        let q_rank = 4;
        let q_rows = 3;
        let mut q_a_weight_values = vec![0.0_f32; q_rank * GLM52_HIDDEN_SIZE];
        q_a_weight_values[0] = 0.5;
        q_a_weight_values[GLM52_HIDDEN_SIZE + 1] = -1.0;
        q_a_weight_values[2 * GLM52_HIDDEN_SIZE + 2] = 2.0;
        q_a_weight_values[3 * GLM52_HIDDEN_SIZE] = -0.25;
        let q_a_weight = bf16_bytes(&q_a_weight_values);
        let q_norm_weight = bf16_bytes(&vec![1.0_f32; q_rank]);
        let q_b_weight = bf16_bytes(&[
            1.0_f32, 0.0, 0.0, 0.0, //
            0.0, 0.5, 0.0, 0.0, //
            0.0, 0.0, -1.0, 0.25,
        ]);

        let unique = format!("{}.{}", std::process::id(), line!());
        let input_norm_name =
            format!("model.layers.3.input_layernorm.weight.device-output.{unique}");
        let q_a_name = format!("model.layers.3.self_attn.q_a_proj.weight.device-output.{unique}");
        let q_norm_name =
            format!("model.layers.3.self_attn.q_a_layernorm.weight.device-output.{unique}");
        let q_b_name = format!("model.layers.3.self_attn.q_b_proj.weight.device-output.{unique}");

        for (name, bytes, label) in [
            (
                input_norm_name.as_str(),
                input_norm_weight.as_slice(),
                "test preloaded device-output input RMSNorm weight",
            ),
            (
                q_a_name.as_str(),
                q_a_weight.as_slice(),
                "test preloaded device-output q_a weight",
            ),
            (
                q_norm_name.as_str(),
                q_norm_weight.as_slice(),
                "test preloaded device-output q_a RMSNorm weight",
            ),
            (
                q_b_name.as_str(),
                q_b_weight.as_slice(),
                "test preloaded device-output q_b weight",
            ),
        ] {
            match preload_resident_weight_from_host_staging(name, bytes.len(), label, |staging| {
                staging.copy_from_slice(bytes);
                Ok(())
            }) {
                Ok(()) => {}
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            }
        }

        let normalized_device = match rmsnorm_hidden_bf16_preloaded_resident_weight_device_output(
            &input_norm_name,
            &hidden_bf16,
            1,
            GLM52_HIDDEN_SIZE,
            1.0e-5,
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        assert_eq!(
            normalized_device.backend,
            CUDA_REFERENCE_RMSNORM_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(normalized_device.rows, 1);
        assert_eq!(normalized_device.values_per_row, GLM52_HIDDEN_SIZE);
        let normalized_host = rmsnorm_hidden_bf16_preloaded_resident_weight(
            &input_norm_name,
            &hidden_bf16,
            1,
            GLM52_HIDDEN_SIZE,
            1.0e-5,
        )?;
        let normalized_readback = normalized_device.copy_to_host_values()?;
        for (index, (actual, expected)) in normalized_readback
            .iter()
            .zip(normalized_host.values.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 1.0e-3,
                "normalized device output {index} mismatch: actual={actual} expected={expected}"
            );
        }

        let q_a_device = linear_rows_bf16_preloaded_resident_weight_device_output(
            &q_a_name,
            normalized_device.buffer(),
            None,
            1,
            GLM52_HIDDEN_SIZE,
            q_rank,
            q_rank,
        )?;
        let q_a_host = cuda_linear_rows_bf16_preloaded_resident_weight_device_input(
            &q_a_name,
            normalized_device.buffer(),
            None,
            1,
            GLM52_HIDDEN_SIZE,
            q_rank,
            LinearResidentView {
                full_bytes: q_a_weight.len(),
                offset_bytes: 0,
                view_bytes: q_a_weight.len(),
            },
        )?;
        assert_eq!(
            q_a_device.backend,
            CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        let q_a_readback = q_a_device.copy_to_host_values()?;
        for (index, (actual, expected)) in
            q_a_readback.iter().zip(q_a_host.values.iter()).enumerate()
        {
            assert!(
                (actual - expected).abs() <= 1.0e-3,
                "q_a device output {index} mismatch: actual={actual} expected={expected}"
            );
        }

        let q_a_normalized_device =
            rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output(
                &q_norm_name,
                q_a_device.buffer(),
                1,
                q_rank,
                1.0e-5,
            )?;
        assert_eq!(q_a_normalized_device.rows, 1);
        assert_eq!(q_a_normalized_device.values_per_row, q_rank);
        let q_a_normalized_readback = q_a_normalized_device.copy_to_host_values()?;
        let q_a_normalized_host = rmsnorm_hidden_bf16_preloaded_resident_weight(
            &q_norm_name,
            &bf16_bytes(&q_a_readback),
            1,
            q_rank,
            1.0e-5,
        )?;
        for (index, (actual, expected)) in q_a_normalized_readback
            .iter()
            .zip(q_a_normalized_host.values.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 1.0e-3,
                "q_a norm device output {index} mismatch: actual={actual} expected={expected}"
            );
        }

        let q_b_device = linear_rows_bf16_preloaded_resident_weight_device_output(
            &q_b_name,
            q_a_normalized_device.buffer(),
            None,
            1,
            q_rank,
            q_rows,
            q_rows,
        )?;
        let q_b_host = cuda_linear_rows_bf16_preloaded_resident_weight_device_input(
            &q_b_name,
            q_a_normalized_device.buffer(),
            None,
            1,
            q_rank,
            q_rows,
            LinearResidentView {
                full_bytes: q_b_weight.len(),
                offset_bytes: 0,
                view_bytes: q_b_weight.len(),
            },
        )?;
        let q_b_readback = q_b_device.copy_to_host_values()?;
        for (index, (actual, expected)) in
            q_b_readback.iter().zip(q_b_host.values.iter()).enumerate()
        {
            assert!(
                (actual - expected).abs() <= 1.0e-3,
                "q_b device output {index} mismatch: actual={actual} expected={expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn linear_rows_bf16_preloaded_attention_weight_padded_device_input_uses_coord_sparse_a_graph_slot(
    ) -> Result<()> {
        if !cuda_reference_kernels_test_enabled() {
            return Ok(());
        }
        let library = match cuda_native_library() {
            Ok(library) => library,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseA,
            LayerWaveMode::Decode,
            1,
        )?;
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        let input = bf16_bytes(&[1.0, -2.0]);
        let mut input_buffer = match library.alloc_device_buffer(input.len()) {
            Ok(buffer) => buffer,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        if let Err(error) = library.copy_h2d(input_buffer, &input) {
            let _ = library.free_device_buffer(&mut input_buffer);
            if cuda_allocation_unavailable(&error) {
                return Ok(());
            }
            return Err(error);
        }

        let mut weight_values = vec![0.0_f32; 3 * GLM52_HIDDEN_SIZE];
        weight_values[0] = 0.5;
        weight_values[GLM52_HIDDEN_SIZE + 1] = -1.5;
        weight_values[2 * GLM52_HIDDEN_SIZE] = 2.0;
        weight_values[2 * GLM52_HIDDEN_SIZE + 2] = 9.0;
        let weight = bf16_bytes(&weight_values);
        let weight_name = format!(
            "model.layers.3.self_attn.o_proj.weight.padded-device-input.test.{}.{}",
            std::process::id(),
            line!()
        );
        match preload_resident_weight_from_host_staging(
            &weight_name,
            weight.len(),
            "test preloaded sparse padded device-input linear graph-slot weight",
            |staging| {
                staging.copy_from_slice(&weight);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) => {
                let _ = library.free_device_buffer(&mut input_buffer);
                if cuda_allocation_unavailable(&error) {
                    return Ok(());
                }
                return Err(error);
            }
        }

        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Sparse-A decode graph key is registered");
        let acquisitions_before = registry.slots[graph_index]
            .lock()
            .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?
            .acquisitions;

        let output_result = linear_rows_bf16_preloaded_resident_weight_padded_device_input(
            &weight_name,
            input_buffer,
            None,
            1,
            2,
            GLM52_HIDDEN_SIZE,
            3,
            3,
        );
        let free_result = library.free_device_buffer(&mut input_buffer);
        let output = match output_result {
            Ok(output) => output,
            Err(error) => {
                let _ = library.free_device_buffer(&mut input_buffer);
                if cuda_allocation_unavailable(&error) {
                    return Ok(());
                }
                return Err(error);
            }
        };
        free_result?;
        let acquisitions_after = registry.slots[graph_index]
            .lock()
            .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?
            .acquisitions;

        assert_eq!(
            output.backend,
            CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(output.values.len(), 3);
        assert!((output.values[0] - 0.5).abs() < 1.0e-3);
        assert!((output.values[1] - 3.0).abs() < 1.0e-3);
        assert!((output.values[2] - 2.0).abs() < 1.0e-3);
        assert!(acquisitions_after > acquisitions_before);
        Ok(())
    }

    #[test]
    fn linear_rows_bf16_preloaded_attention_padded_graph_replays_with_updated_copy_nodes(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);

        let result = (|| -> Result<()> {
            let library = match cuda_native_library() {
                Ok(library) => library,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            let rows = 2;
            let active_input_dim = 2;
            let full_input_dim = GLM52_HIDDEN_SIZE;
            let output_dim = 3;
            let graph_key = CoordinatorGraphKey::glm52_bf16(
                CoordinatorGraphShape::CoordSparseA,
                LayerWaveMode::Prefill,
                rows,
            )?;
            let registry = match coordinator_cuda_graph_workspace_registry() {
                Ok(registry) => registry,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            let graph_index = registry
                .keys()
                .iter()
                .position(|key| key == &graph_key)
                .expect("Coord-Sparse-A prefill graph key is registered");
            let (graph_captures_before, graph_launches_before) = {
                let slot = registry.slots[graph_index]
                    .lock()
                    .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
                (slot.graph_captures, slot.graph_launches)
            };

            let input_a = bf16_bytes(&[1.0, -2.0, 0.5, 1.0]);
            let mut input_buffer_a = match library.alloc_device_buffer(input_a.len()) {
                Ok(buffer) => buffer,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            if let Err(error) = library.copy_h2d(input_buffer_a, &input_a) {
                let _ = library.free_device_buffer(&mut input_buffer_a);
                if cuda_allocation_unavailable(&error) {
                    return Ok(());
                }
                return Err(error);
            }

            let input_b = bf16_bytes(&[-1.0, 0.5, 2.0, -0.25]);
            let mut input_buffer_b = match library.alloc_device_buffer(input_b.len()) {
                Ok(buffer) => buffer,
                Err(error) => {
                    let _ = library.free_device_buffer(&mut input_buffer_a);
                    if cuda_allocation_unavailable(&error) {
                        return Ok(());
                    }
                    return Err(error);
                }
            };
            if let Err(error) = library.copy_h2d(input_buffer_b, &input_b) {
                let _ = library.free_device_buffer(&mut input_buffer_b);
                let _ = library.free_device_buffer(&mut input_buffer_a);
                if cuda_allocation_unavailable(&error) {
                    return Ok(());
                }
                return Err(error);
            }

            let mut weight_values_a = vec![0.0_f32; output_dim * full_input_dim];
            weight_values_a[0] = 0.5;
            weight_values_a[full_input_dim + 1] = -1.5;
            weight_values_a[2 * full_input_dim] = 2.0;
            weight_values_a[2 * full_input_dim + 2] = 9.0;
            let weight_a = bf16_bytes(&weight_values_a);
            let weight_name_a = format!(
                "model.layers.3.self_attn.o_proj.weight.padded-linear-replay-a.{}.{}",
                std::process::id(),
                line!()
            );
            match preload_resident_weight_from_host_staging(
                &weight_name_a,
                weight_a.len(),
                "test preloaded sparse padded linear graph replay weight a",
                |staging| {
                    staging.copy_from_slice(&weight_a);
                    Ok(())
                },
            ) {
                Ok(()) => {}
                Err(error) => {
                    let _ = library.free_device_buffer(&mut input_buffer_b);
                    let _ = library.free_device_buffer(&mut input_buffer_a);
                    if cuda_allocation_unavailable(&error) {
                        return Ok(());
                    }
                    return Err(error);
                }
            }

            let mut weight_values_b = vec![0.0_f32; output_dim * full_input_dim];
            weight_values_b[0] = -2.0;
            weight_values_b[full_input_dim + 1] = 1.0;
            weight_values_b[2 * full_input_dim] = 0.25;
            weight_values_b[2 * full_input_dim + 1] = 0.5;
            weight_values_b[2 * full_input_dim + 2] = 11.0;
            let weight_b = bf16_bytes(&weight_values_b);
            let weight_name_b = format!(
                "model.layers.3.self_attn.o_proj.weight.padded-linear-replay-b.{}.{}",
                std::process::id(),
                line!()
            );
            match preload_resident_weight_from_host_staging(
                &weight_name_b,
                weight_b.len(),
                "test preloaded sparse padded linear graph replay weight b",
                |staging| {
                    staging.copy_from_slice(&weight_b);
                    Ok(())
                },
            ) {
                Ok(()) => {}
                Err(error) => {
                    let _ = library.free_device_buffer(&mut input_buffer_b);
                    let _ = library.free_device_buffer(&mut input_buffer_a);
                    if cuda_allocation_unavailable(&error) {
                        return Ok(());
                    }
                    return Err(error);
                }
            }

            let output_a = match linear_rows_bf16_preloaded_resident_weight_padded_device_input(
                &weight_name_a,
                input_buffer_a,
                None,
                rows,
                active_input_dim,
                full_input_dim,
                output_dim,
                output_dim,
            ) {
                Ok(output) => output,
                Err(error) => {
                    let _ = library.free_device_buffer(&mut input_buffer_b);
                    let _ = library.free_device_buffer(&mut input_buffer_a);
                    if cuda_allocation_unavailable(&error) {
                        return Ok(());
                    }
                    return Err(error);
                }
            };

            let output_b = match linear_rows_bf16_preloaded_resident_weight_padded_device_input(
                &weight_name_b,
                input_buffer_b,
                None,
                rows,
                active_input_dim,
                full_input_dim,
                output_dim,
                output_dim,
            ) {
                Ok(output) => output,
                Err(error) => {
                    let _ = library.free_device_buffer(&mut input_buffer_b);
                    let _ = library.free_device_buffer(&mut input_buffer_a);
                    if cuda_allocation_unavailable(&error) {
                        return Ok(());
                    }
                    return Err(error);
                }
            };
            library.free_device_buffer(&mut input_buffer_b)?;
            library.free_device_buffer(&mut input_buffer_a)?;

            let expected_a = [0.5_f32, 3.0, 2.0, 0.25, -1.5, 1.0];
            let expected_b = [2.0_f32, 0.5, 0.0, -4.0, -0.25, 0.375];
            for (actual, expected) in output_a.values.iter().zip(expected_a.iter()) {
                assert!((actual - expected).abs() < 1.0e-3);
            }
            for (actual, expected) in output_b.values.iter().zip(expected_b.iter()) {
                assert!((actual - expected).abs() < 1.0e-3);
            }
            assert_ne!(output_a.values, output_b.values);

            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            assert!(slot.has_captured_graph(
                CoordinatorCudaGraphProgram::LayerPaddedLinearBf16,
                padded_linear_graph_signature(
                    &graph_key,
                    active_input_dim,
                    full_input_dim,
                    output_dim,
                    false
                )
            ));
            assert!(slot.graph_captures >= graph_captures_before);
            assert!(slot.graph_launches >= graph_launches_before + 2);
            Ok(())
        })();

        result
    }

    #[test]
    fn linear_rows_bf16_preloaded_attention_padded_graph_replays_same_bucket_when_rows_change(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);

        let result = (|| -> Result<()> {
            let library = match cuda_native_library() {
                Ok(library) => library,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            let rows_first = 2;
            let rows_second = 4;
            let active_input_dim = 2;
            let full_input_dim = GLM52_HIDDEN_SIZE;
            let output_dim = 3;
            let graph_key = CoordinatorGraphKey::glm52_bf16(
                CoordinatorGraphShape::CoordSparseA,
                LayerWaveMode::Prefill,
                rows_first,
            )?;
            assert_eq!(
                graph_key,
                CoordinatorGraphKey::glm52_bf16(
                    CoordinatorGraphShape::CoordSparseA,
                    LayerWaveMode::Prefill,
                    rows_second,
                )?
            );
            let registry = match coordinator_cuda_graph_workspace_registry() {
                Ok(registry) => registry,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            let graph_index = registry
                .keys()
                .iter()
                .position(|key| key == &graph_key)
                .expect("Coord-Sparse-A prefill graph key is registered");
            let (graph_captures_before, graph_launches_before) = {
                let slot = registry.slots[graph_index]
                    .lock()
                    .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
                (slot.graph_captures, slot.graph_launches)
            };

            let input0 = bf16_bytes(&[1.0, -2.0, 0.5, 1.0]);
            let mut input_buffer0 = match library.alloc_device_buffer(input0.len()) {
                Ok(buffer) => buffer,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            if let Err(error) = library.copy_h2d(input_buffer0, &input0) {
                let _ = library.free_device_buffer(&mut input_buffer0);
                if cuda_allocation_unavailable(&error) {
                    return Ok(());
                }
                return Err(error);
            }

            let input1 = bf16_bytes(&[-1.0, 0.5, 2.0, -0.25, 0.0, 1.5, -0.75, -1.25]);
            let mut input_buffer1 = match library.alloc_device_buffer(input1.len()) {
                Ok(buffer) => buffer,
                Err(error) => {
                    let _ = library.free_device_buffer(&mut input_buffer0);
                    if cuda_allocation_unavailable(&error) {
                        return Ok(());
                    }
                    return Err(error);
                }
            };
            if let Err(error) = library.copy_h2d(input_buffer1, &input1) {
                let _ = library.free_device_buffer(&mut input_buffer1);
                let _ = library.free_device_buffer(&mut input_buffer0);
                if cuda_allocation_unavailable(&error) {
                    return Ok(());
                }
                return Err(error);
            }

            let mut weight_values = vec![0.0_f32; output_dim * full_input_dim];
            weight_values[0] = 0.5;
            weight_values[full_input_dim + 1] = -1.5;
            weight_values[2 * full_input_dim] = 2.0;
            weight_values[2 * full_input_dim + 2] = 9.0;
            let weight = bf16_bytes(&weight_values);
            let weight_name = format!(
                "model.layers.3.self_attn.o_proj.weight.padded-linear-row-bucket.{}.{}",
                std::process::id(),
                line!()
            );
            match preload_resident_weight_from_host_staging(
                &weight_name,
                weight.len(),
                "test preloaded sparse padded linear row-bucket replay weight",
                |staging| {
                    staging.copy_from_slice(&weight);
                    Ok(())
                },
            ) {
                Ok(()) => {}
                Err(error) => {
                    let _ = library.free_device_buffer(&mut input_buffer1);
                    let _ = library.free_device_buffer(&mut input_buffer0);
                    if cuda_allocation_unavailable(&error) {
                        return Ok(());
                    }
                    return Err(error);
                }
            }

            let output0 = match linear_rows_bf16_preloaded_resident_weight_padded_device_input(
                &weight_name,
                input_buffer0,
                None,
                rows_first,
                active_input_dim,
                full_input_dim,
                output_dim,
                output_dim,
            ) {
                Ok(output) => output,
                Err(error) => {
                    let _ = library.free_device_buffer(&mut input_buffer1);
                    let _ = library.free_device_buffer(&mut input_buffer0);
                    if cuda_allocation_unavailable(&error) {
                        return Ok(());
                    }
                    return Err(error);
                }
            };
            let expected0 = [0.5_f32, 3.0, 2.0, 0.25, -1.5, 1.0];
            for (actual, expected) in output0.values.iter().zip(expected0.iter()) {
                assert!((actual - expected).abs() < 1.0e-3);
            }

            let (graph_captures_after_first, graph_launches_after_first) = {
                let slot = registry.slots[graph_index]
                    .lock()
                    .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
                assert!(slot.has_captured_graph(
                    CoordinatorCudaGraphProgram::LayerPaddedLinearBf16,
                    padded_linear_graph_signature(
                        &graph_key,
                        active_input_dim,
                        full_input_dim,
                        output_dim,
                        false
                    )
                ));
                (slot.graph_captures, slot.graph_launches)
            };

            let output1 = match linear_rows_bf16_preloaded_resident_weight_padded_device_input(
                &weight_name,
                input_buffer1,
                None,
                rows_second,
                active_input_dim,
                full_input_dim,
                output_dim,
                output_dim,
            ) {
                Ok(output) => output,
                Err(error) => {
                    let _ = library.free_device_buffer(&mut input_buffer1);
                    let _ = library.free_device_buffer(&mut input_buffer0);
                    if cuda_allocation_unavailable(&error) {
                        return Ok(());
                    }
                    return Err(error);
                }
            };
            library.free_device_buffer(&mut input_buffer1)?;
            library.free_device_buffer(&mut input_buffer0)?;

            let expected1 = [
                -0.5_f32, -0.75, -2.0, //
                1.0, 0.375, 4.0, //
                0.0, -2.25, 0.0, //
                -0.375, 1.875, -1.5, //
            ];
            for (actual, expected) in output1.values.iter().zip(expected1.iter()) {
                assert!((actual - expected).abs() < 1.0e-3);
            }

            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            assert!(slot.has_captured_graph(
                CoordinatorCudaGraphProgram::LayerPaddedLinearBf16,
                padded_linear_graph_signature(
                    &graph_key,
                    active_input_dim,
                    full_input_dim,
                    output_dim,
                    false
                )
            ));
            assert!(slot.graph_captures >= graph_captures_before);
            assert_eq!(slot.graph_captures, graph_captures_after_first);
            assert!(slot.graph_launches >= graph_launches_before + 2);
            assert!(slot.graph_launches >= graph_launches_after_first + 1);
            Ok(())
        })();

        result
    }

    #[test]
    fn linear_residual_add_bf16_preloaded_attention_weight_device_input_uses_coord_sparse_a_graph_slot(
    ) -> Result<()> {
        if !cuda_reference_kernels_test_enabled() {
            return Ok(());
        }
        let library = match cuda_native_library() {
            Ok(library) => library,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseA,
            LayerWaveMode::Decode,
            1,
        )?;
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        let mut input_values = vec![0.0_f32; GLM52_HIDDEN_SIZE];
        input_values[0] = 1.0;
        input_values[1] = -2.0;
        let input = bf16_bytes(&input_values);
        let mut input_buffer = match library.alloc_device_buffer(input.len()) {
            Ok(buffer) => buffer,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        if let Err(error) = library.copy_h2d(input_buffer, &input) {
            let _ = library.free_device_buffer(&mut input_buffer);
            if cuda_allocation_unavailable(&error) {
                return Ok(());
            }
            return Err(error);
        }

        let mut weight_values = vec![0.0_f32; 3 * GLM52_HIDDEN_SIZE];
        weight_values[0] = 0.5;
        weight_values[GLM52_HIDDEN_SIZE + 1] = -1.5;
        weight_values[2 * GLM52_HIDDEN_SIZE] = 2.0;
        let weight = bf16_bytes(&weight_values);
        let residual = bf16_bytes(&[1.0, 10.0, -1.0]);
        let weight_name = format!(
            "model.layers.3.self_attn.o_proj.weight.device-input-residual.test.{}.{}",
            std::process::id(),
            line!()
        );
        match preload_resident_weight_from_host_staging(
            &weight_name,
            weight.len(),
            "test preloaded sparse device-input linear residual-add graph-slot weight",
            |staging| {
                staging.copy_from_slice(&weight);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) => {
                let _ = library.free_device_buffer(&mut input_buffer);
                if cuda_allocation_unavailable(&error) {
                    return Ok(());
                }
                return Err(error);
            }
        }

        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Sparse-A decode graph key is registered");
        let acquisitions_before = registry.slots[graph_index]
            .lock()
            .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?
            .acquisitions;

        let output_result = linear_residual_add_rows_bf16_preloaded_resident_weight_device_input(
            &weight_name,
            input_buffer,
            None,
            &residual,
            1,
            GLM52_HIDDEN_SIZE,
            3,
            3,
        );
        let free_result = library.free_device_buffer(&mut input_buffer);
        let output = match output_result {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        free_result?;
        let acquisitions_after = registry.slots[graph_index]
            .lock()
            .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?
            .acquisitions;

        assert_eq!(
            output.linear_backend,
            CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(
            output.residual_add_backend,
            CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND
        );
        assert_eq!(output.linear_values.len(), 3);
        assert_eq!(output.residual_values.len(), 3);
        assert!((output.linear_values[0] - 0.5).abs() < 1.0e-3);
        assert!((output.linear_values[1] - 3.0).abs() < 1.0e-3);
        assert!((output.linear_values[2] - 2.0).abs() < 1.0e-3);
        assert!((output.residual_values[0] - 1.5).abs() < 1.0e-3);
        assert!((output.residual_values[1] - 13.0).abs() < 1.0e-3);
        assert!((output.residual_values[2] - 1.0).abs() < 1.0e-3);
        assert!(acquisitions_after > acquisitions_before);
        Ok(())
    }

    #[test]
    fn linear_residual_add_bf16_preloaded_attention_weight_padded_device_input_uses_coord_sparse_a_graph_slot(
    ) -> Result<()> {
        if !cuda_reference_kernels_test_enabled() {
            return Ok(());
        }
        let library = match cuda_native_library() {
            Ok(library) => library,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseA,
            LayerWaveMode::Decode,
            1,
        )?;
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        let input = bf16_bytes(&[1.0, -2.0]);
        let mut input_buffer = match library.alloc_device_buffer(input.len()) {
            Ok(buffer) => buffer,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        if let Err(error) = library.copy_h2d(input_buffer, &input) {
            let _ = library.free_device_buffer(&mut input_buffer);
            if cuda_allocation_unavailable(&error) {
                return Ok(());
            }
            return Err(error);
        }

        let mut weight_values = vec![0.0_f32; 3 * GLM52_HIDDEN_SIZE];
        weight_values[0] = 0.5;
        weight_values[GLM52_HIDDEN_SIZE + 1] = -1.5;
        weight_values[2 * GLM52_HIDDEN_SIZE] = 2.0;
        weight_values[2 * GLM52_HIDDEN_SIZE + 2] = 9.0;
        let weight = bf16_bytes(&weight_values);
        let residual = bf16_bytes(&[1.0, 10.0, -1.0]);
        let weight_name = format!(
            "model.layers.3.self_attn.o_proj.weight.padded-device-input-residual.test.{}.{}",
            std::process::id(),
            line!()
        );
        match preload_resident_weight_from_host_staging(
            &weight_name,
            weight.len(),
            "test preloaded sparse padded device-input linear residual-add graph-slot weight",
            |staging| {
                staging.copy_from_slice(&weight);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) => {
                let _ = library.free_device_buffer(&mut input_buffer);
                if cuda_allocation_unavailable(&error) {
                    return Ok(());
                }
                return Err(error);
            }
        }

        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Sparse-A decode graph key is registered");
        let acquisitions_before = registry.slots[graph_index]
            .lock()
            .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?
            .acquisitions;

        let output_result =
            linear_residual_add_rows_bf16_preloaded_resident_weight_padded_device_input(
                &weight_name,
                input_buffer,
                None,
                &residual,
                1,
                2,
                GLM52_HIDDEN_SIZE,
                3,
                3,
            );
        let free_result = library.free_device_buffer(&mut input_buffer);
        let output = match output_result {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        free_result?;
        let acquisitions_after = registry.slots[graph_index]
            .lock()
            .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?
            .acquisitions;

        assert_eq!(
            output.linear_backend,
            CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(
            output.residual_add_backend,
            CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND
        );
        assert_eq!(output.linear_values.len(), 3);
        assert_eq!(output.residual_values.len(), 3);
        assert!((output.linear_values[0] - 0.5).abs() < 1.0e-3);
        assert!((output.linear_values[1] - 3.0).abs() < 1.0e-3);
        assert!((output.linear_values[2] - 2.0).abs() < 1.0e-3);
        assert!((output.residual_values[0] - 1.5).abs() < 1.0e-3);
        assert!((output.residual_values[1] - 13.0).abs() < 1.0e-3);
        assert!((output.residual_values[2] - 1.0).abs() < 1.0e-3);
        assert!(acquisitions_after > acquisitions_before);
        Ok(())
    }

    #[test]
    fn linear_residual_add_bf16_preloaded_attention_weight_device_input_device_output_keeps_residual_on_device(
    ) -> Result<()> {
        if !cuda_reference_kernels_test_enabled() {
            return Ok(());
        }
        let library = match cuda_native_library() {
            Ok(library) => library,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseA,
            LayerWaveMode::Decode,
            1,
        )?;
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        let mut input_values = vec![0.0_f32; GLM52_HIDDEN_SIZE];
        input_values[0] = 1.0;
        input_values[1] = -2.0;
        let input = bf16_bytes(&input_values);
        let mut input_buffer = match library.alloc_device_buffer(input.len()) {
            Ok(buffer) => buffer,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        if let Err(error) = library.copy_h2d(input_buffer, &input) {
            let _ = library.free_device_buffer(&mut input_buffer);
            if cuda_allocation_unavailable(&error) {
                return Ok(());
            }
            return Err(error);
        }

        let mut weight_values = vec![0.0_f32; 3 * GLM52_HIDDEN_SIZE];
        weight_values[0] = 0.5;
        weight_values[GLM52_HIDDEN_SIZE + 1] = -1.5;
        weight_values[2 * GLM52_HIDDEN_SIZE] = 2.0;
        let weight = bf16_bytes(&weight_values);
        let residual = bf16_bytes(&[1.0, 10.0, -1.0]);
        let weight_name = format!(
            "model.layers.3.self_attn.o_proj.weight.device-output-residual.test.{}.{}",
            std::process::id(),
            line!()
        );
        match preload_resident_weight_from_host_staging(
            &weight_name,
            weight.len(),
            "test preloaded sparse device-input linear residual-add device-output graph-slot weight",
            |staging| {
                staging.copy_from_slice(&weight);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) => {
                let _ = library.free_device_buffer(&mut input_buffer);
                if cuda_allocation_unavailable(&error) {
                    return Ok(());
                }
                return Err(error);
            }
        }

        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Sparse-A decode graph key is registered");
        let acquisitions_before = registry.slots[graph_index]
            .lock()
            .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?
            .acquisitions;

        let output_result =
            linear_residual_add_rows_bf16_preloaded_resident_weight_device_input_device_output(
                &weight_name,
                input_buffer,
                None,
                &residual,
                1,
                GLM52_HIDDEN_SIZE,
                3,
                3,
            );
        let free_result = library.free_device_buffer(&mut input_buffer);
        let output = match output_result {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        free_result?;
        let acquisitions_after = registry.slots[graph_index]
            .lock()
            .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?
            .acquisitions;

        assert_eq!(
            output.linear_backend,
            CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(
            output.residual_add_backend,
            CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND
        );
        assert_eq!(output.linear_values.len(), 3);
        assert_eq!(output.residual_values.len(), 3);
        assert!((output.linear_values[0] - 0.5).abs() < 1.0e-3);
        assert!((output.linear_values[1] - 3.0).abs() < 1.0e-3);
        assert!((output.linear_values[2] - 2.0).abs() < 1.0e-3);
        assert!((output.residual_values[0] - 1.5).abs() < 1.0e-3);
        assert!((output.residual_values[1] - 13.0).abs() < 1.0e-3);
        assert!((output.residual_values[2] - 1.0).abs() < 1.0e-3);
        assert_eq!(output.residual_device.rows, 1);
        assert_eq!(output.residual_device.values_per_row, 3);
        assert_eq!(
            output.residual_device.backend,
            CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND
        );
        let device_values = output.residual_device.copy_to_host_values()?;
        assert_eq!(device_values.len(), output.residual_values.len());
        for (actual, expected) in device_values.iter().zip(output.residual_values.iter()) {
            assert!((actual - expected).abs() < 1.0e-3);
        }
        assert!(acquisitions_after > acquisitions_before);
        Ok(())
    }

    #[test]
    fn linear_residual_add_bf16_preloaded_attention_graph_replays_with_updated_nodes() -> Result<()>
    {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);

        let result = (|| -> Result<()> {
            let library = match cuda_native_library() {
                Ok(library) => library,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            let rows = 2;
            let input_dim = GLM52_HIDDEN_SIZE;
            let output_dim = 3;
            let graph_key = CoordinatorGraphKey::glm52_bf16(
                CoordinatorGraphShape::CoordSparseA,
                LayerWaveMode::Prefill,
                rows,
            )?;
            let registry = match coordinator_cuda_graph_workspace_registry() {
                Ok(registry) => registry,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            let graph_index = registry
                .keys()
                .iter()
                .position(|key| key == &graph_key)
                .expect("Coord-Sparse-A prefill graph key is registered");
            let (graph_captures_before, graph_launches_before) = {
                let slot = registry.slots[graph_index]
                    .lock()
                    .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
                (slot.graph_captures, slot.graph_launches)
            };

            let mut input_values_a = vec![0.0_f32; rows * input_dim];
            input_values_a[0] = 1.0;
            input_values_a[1] = -2.0;
            input_values_a[input_dim] = 0.5;
            input_values_a[input_dim + 1] = 1.0;
            let input_a = bf16_bytes(&input_values_a);
            let mut input_buffer_a = match library.alloc_device_buffer(input_a.len()) {
                Ok(buffer) => buffer,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            if let Err(error) = library.copy_h2d(input_buffer_a, &input_a) {
                let _ = library.free_device_buffer(&mut input_buffer_a);
                if cuda_allocation_unavailable(&error) {
                    return Ok(());
                }
                return Err(error);
            }

            let mut input_values_b = vec![0.0_f32; rows * input_dim];
            input_values_b[0] = -1.0;
            input_values_b[1] = 0.5;
            input_values_b[input_dim] = 2.0;
            input_values_b[input_dim + 1] = -0.25;
            let input_b = bf16_bytes(&input_values_b);
            let mut input_buffer_b = match library.alloc_device_buffer(input_b.len()) {
                Ok(buffer) => buffer,
                Err(error) => {
                    let _ = library.free_device_buffer(&mut input_buffer_a);
                    if cuda_allocation_unavailable(&error) {
                        return Ok(());
                    }
                    return Err(error);
                }
            };
            if let Err(error) = library.copy_h2d(input_buffer_b, &input_b) {
                let _ = library.free_device_buffer(&mut input_buffer_b);
                let _ = library.free_device_buffer(&mut input_buffer_a);
                if cuda_allocation_unavailable(&error) {
                    return Ok(());
                }
                return Err(error);
            }

            let mut weight_values_a = vec![0.0_f32; output_dim * input_dim];
            weight_values_a[0] = 0.5;
            weight_values_a[input_dim + 1] = -1.5;
            weight_values_a[2 * input_dim] = 2.0;
            let weight_a = bf16_bytes(&weight_values_a);
            let weight_name_a = format!(
                "model.layers.3.self_attn.o_proj.weight.linear-residual-replay-a.{}.{}",
                std::process::id(),
                line!()
            );
            match preload_resident_weight_from_host_staging(
                &weight_name_a,
                weight_a.len(),
                "test preloaded sparse linear residual-add graph replay weight a",
                |staging| {
                    staging.copy_from_slice(&weight_a);
                    Ok(())
                },
            ) {
                Ok(()) => {}
                Err(error) => {
                    let _ = library.free_device_buffer(&mut input_buffer_b);
                    let _ = library.free_device_buffer(&mut input_buffer_a);
                    if cuda_allocation_unavailable(&error) {
                        return Ok(());
                    }
                    return Err(error);
                }
            }

            let mut weight_values_b = vec![0.0_f32; output_dim * input_dim];
            weight_values_b[0] = -2.0;
            weight_values_b[input_dim + 1] = 1.0;
            weight_values_b[2 * input_dim] = 0.25;
            weight_values_b[2 * input_dim + 1] = 0.5;
            let weight_b = bf16_bytes(&weight_values_b);
            let weight_name_b = format!(
                "model.layers.3.self_attn.o_proj.weight.linear-residual-replay-b.{}.{}",
                std::process::id(),
                line!()
            );
            match preload_resident_weight_from_host_staging(
                &weight_name_b,
                weight_b.len(),
                "test preloaded sparse linear residual-add graph replay weight b",
                |staging| {
                    staging.copy_from_slice(&weight_b);
                    Ok(())
                },
            ) {
                Ok(()) => {}
                Err(error) => {
                    let _ = library.free_device_buffer(&mut input_buffer_b);
                    let _ = library.free_device_buffer(&mut input_buffer_a);
                    if cuda_allocation_unavailable(&error) {
                        return Ok(());
                    }
                    return Err(error);
                }
            }

            let residual_a = bf16_bytes(&[1.0, 10.0, -1.0, 0.0, 0.5, 2.0]);
            let output_a =
                match linear_residual_add_rows_bf16_preloaded_resident_weight_device_input_device_output(
                    &weight_name_a,
                    input_buffer_a,
                    None,
                    &residual_a,
                    rows,
                    input_dim,
                    output_dim,
                    output_dim,
                ) {
                    Ok(output) => output,
                    Err(error) => {
                        let _ = library.free_device_buffer(&mut input_buffer_b);
                        let _ = library.free_device_buffer(&mut input_buffer_a);
                        if cuda_allocation_unavailable(&error) {
                            return Ok(());
                        }
                        return Err(error);
                    }
                };

            let residual_b = bf16_bytes(&[0.0, 1.0, 2.0, -1.0, 0.0, 0.5]);
            let output_b =
                match linear_residual_add_rows_bf16_preloaded_resident_weight_device_input_device_output(
                    &weight_name_b,
                    input_buffer_b,
                    None,
                    &residual_b,
                    rows,
                    input_dim,
                    output_dim,
                    output_dim,
                ) {
                    Ok(output) => output,
                    Err(error) => {
                        let _ = library.free_device_buffer(&mut input_buffer_b);
                        let _ = library.free_device_buffer(&mut input_buffer_a);
                        if cuda_allocation_unavailable(&error) {
                            return Ok(());
                        }
                        return Err(error);
                    }
                };
            library.free_device_buffer(&mut input_buffer_b)?;
            library.free_device_buffer(&mut input_buffer_a)?;

            let expected_linear_a = [0.5_f32, 3.0, 2.0, 0.25, -1.5, 1.0];
            let expected_residual_a = [1.5_f32, 13.0, 1.0, 0.25, -1.0, 3.0];
            let expected_linear_b = [2.0_f32, 0.5, 0.0, -4.0, -0.25, 0.375];
            let expected_residual_b = [2.0_f32, 1.5, 2.0, -5.0, -0.25, 0.875];
            for (actual, expected) in output_a.linear_values.iter().zip(expected_linear_a.iter()) {
                assert!((actual - expected).abs() < 1.0e-3);
            }
            for (actual, expected) in output_a
                .residual_values
                .iter()
                .zip(expected_residual_a.iter())
            {
                assert!((actual - expected).abs() < 1.0e-3);
            }
            for (actual, expected) in output_b.linear_values.iter().zip(expected_linear_b.iter()) {
                assert!((actual - expected).abs() < 1.0e-3);
            }
            for (actual, expected) in output_b
                .residual_values
                .iter()
                .zip(expected_residual_b.iter())
            {
                assert!((actual - expected).abs() < 1.0e-3);
            }
            assert_ne!(output_a.residual_values, output_b.residual_values);

            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            assert!(slot.has_captured_graph(
                CoordinatorCudaGraphProgram::LayerLinearResidualAddBf16,
                CoordinatorCudaGraphSignature::linear_bf16(
                    graph_key.row_bucket.row_capacity,
                    input_dim,
                    output_dim,
                    false
                )
            ));
            assert!(slot.graph_captures >= graph_captures_before);
            assert!(slot.graph_launches >= graph_launches_before + 2);
            Ok(())
        })();

        result
    }

    #[test]
    fn linear_residual_add_bf16_preloaded_attention_padded_graph_replays_with_updated_copy_nodes(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);

        let result = (|| -> Result<()> {
            let library = match cuda_native_library() {
                Ok(library) => library,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            let rows = 2;
            let active_input_dim = 2;
            let full_input_dim = GLM52_HIDDEN_SIZE;
            let output_dim = 3;
            let graph_key = CoordinatorGraphKey::glm52_bf16(
                CoordinatorGraphShape::CoordSparseA,
                LayerWaveMode::Prefill,
                rows,
            )?;
            let registry = match coordinator_cuda_graph_workspace_registry() {
                Ok(registry) => registry,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            let graph_index = registry
                .keys()
                .iter()
                .position(|key| key == &graph_key)
                .expect("Coord-Sparse-A prefill graph key is registered");
            let (graph_captures_before, graph_launches_before) = {
                let slot = registry.slots[graph_index]
                    .lock()
                    .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
                (slot.graph_captures, slot.graph_launches)
            };

            let input_a = bf16_bytes(&[1.0, -2.0, 0.5, 1.0]);
            let mut input_buffer_a = match library.alloc_device_buffer(input_a.len()) {
                Ok(buffer) => buffer,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            if let Err(error) = library.copy_h2d(input_buffer_a, &input_a) {
                let _ = library.free_device_buffer(&mut input_buffer_a);
                if cuda_allocation_unavailable(&error) {
                    return Ok(());
                }
                return Err(error);
            }

            let input_b = bf16_bytes(&[-1.0, 0.5, 2.0, -0.25]);
            let mut input_buffer_b = match library.alloc_device_buffer(input_b.len()) {
                Ok(buffer) => buffer,
                Err(error) => {
                    let _ = library.free_device_buffer(&mut input_buffer_a);
                    if cuda_allocation_unavailable(&error) {
                        return Ok(());
                    }
                    return Err(error);
                }
            };
            if let Err(error) = library.copy_h2d(input_buffer_b, &input_b) {
                let _ = library.free_device_buffer(&mut input_buffer_b);
                let _ = library.free_device_buffer(&mut input_buffer_a);
                if cuda_allocation_unavailable(&error) {
                    return Ok(());
                }
                return Err(error);
            }

            let mut weight_values_a = vec![0.0_f32; output_dim * full_input_dim];
            weight_values_a[0] = 0.5;
            weight_values_a[full_input_dim + 1] = -1.5;
            weight_values_a[2 * full_input_dim] = 2.0;
            weight_values_a[2 * full_input_dim + 2] = 9.0;
            let weight_a = bf16_bytes(&weight_values_a);
            let weight_name_a = format!(
                "model.layers.3.self_attn.o_proj.weight.padded-linear-residual-replay-a.{}.{}",
                std::process::id(),
                line!()
            );
            match preload_resident_weight_from_host_staging(
                &weight_name_a,
                weight_a.len(),
                "test preloaded sparse padded linear residual-add graph replay weight a",
                |staging| {
                    staging.copy_from_slice(&weight_a);
                    Ok(())
                },
            ) {
                Ok(()) => {}
                Err(error) => {
                    let _ = library.free_device_buffer(&mut input_buffer_b);
                    let _ = library.free_device_buffer(&mut input_buffer_a);
                    if cuda_allocation_unavailable(&error) {
                        return Ok(());
                    }
                    return Err(error);
                }
            }

            let mut weight_values_b = vec![0.0_f32; output_dim * full_input_dim];
            weight_values_b[0] = -2.0;
            weight_values_b[full_input_dim + 1] = 1.0;
            weight_values_b[2 * full_input_dim] = 0.25;
            weight_values_b[2 * full_input_dim + 1] = 0.5;
            weight_values_b[2 * full_input_dim + 2] = 11.0;
            let weight_b = bf16_bytes(&weight_values_b);
            let weight_name_b = format!(
                "model.layers.3.self_attn.o_proj.weight.padded-linear-residual-replay-b.{}.{}",
                std::process::id(),
                line!()
            );
            match preload_resident_weight_from_host_staging(
                &weight_name_b,
                weight_b.len(),
                "test preloaded sparse padded linear residual-add graph replay weight b",
                |staging| {
                    staging.copy_from_slice(&weight_b);
                    Ok(())
                },
            ) {
                Ok(()) => {}
                Err(error) => {
                    let _ = library.free_device_buffer(&mut input_buffer_b);
                    let _ = library.free_device_buffer(&mut input_buffer_a);
                    if cuda_allocation_unavailable(&error) {
                        return Ok(());
                    }
                    return Err(error);
                }
            }

            let residual_a = bf16_bytes(&[1.0, 10.0, -1.0, 0.0, 0.5, 2.0]);
            let output_a =
                match linear_residual_add_rows_bf16_preloaded_resident_weight_padded_device_input_device_output(
                    &weight_name_a,
                    input_buffer_a,
                    None,
                    &residual_a,
                    rows,
                    active_input_dim,
                    full_input_dim,
                    output_dim,
                    output_dim,
                ) {
                    Ok(output) => output,
                    Err(error) => {
                        let _ = library.free_device_buffer(&mut input_buffer_b);
                        let _ = library.free_device_buffer(&mut input_buffer_a);
                        if cuda_allocation_unavailable(&error) {
                            return Ok(());
                        }
                        return Err(error);
                    }
                };

            let residual_b = bf16_bytes(&[0.0, 1.0, 2.0, -1.0, 0.0, 0.5]);
            let output_b =
                match linear_residual_add_rows_bf16_preloaded_resident_weight_padded_device_input_device_output(
                    &weight_name_b,
                    input_buffer_b,
                    None,
                    &residual_b,
                    rows,
                    active_input_dim,
                    full_input_dim,
                    output_dim,
                    output_dim,
                ) {
                    Ok(output) => output,
                    Err(error) => {
                        let _ = library.free_device_buffer(&mut input_buffer_b);
                        let _ = library.free_device_buffer(&mut input_buffer_a);
                        if cuda_allocation_unavailable(&error) {
                            return Ok(());
                        }
                        return Err(error);
                    }
                };
            library.free_device_buffer(&mut input_buffer_b)?;
            library.free_device_buffer(&mut input_buffer_a)?;

            let expected_linear_a = [0.5_f32, 3.0, 2.0, 0.25, -1.5, 1.0];
            let expected_residual_a = [1.5_f32, 13.0, 1.0, 0.25, -1.0, 3.0];
            let expected_linear_b = [2.0_f32, 0.5, 0.0, -4.0, -0.25, 0.375];
            let expected_residual_b = [2.0_f32, 1.5, 2.0, -5.0, -0.25, 0.875];
            for (actual, expected) in output_a.linear_values.iter().zip(expected_linear_a.iter()) {
                assert!((actual - expected).abs() < 1.0e-3);
            }
            for (actual, expected) in output_a
                .residual_values
                .iter()
                .zip(expected_residual_a.iter())
            {
                assert!((actual - expected).abs() < 1.0e-3);
            }
            for (actual, expected) in output_b.linear_values.iter().zip(expected_linear_b.iter()) {
                assert!((actual - expected).abs() < 1.0e-3);
            }
            for (actual, expected) in output_b
                .residual_values
                .iter()
                .zip(expected_residual_b.iter())
            {
                assert!((actual - expected).abs() < 1.0e-3);
            }
            assert_ne!(output_a.residual_values, output_b.residual_values);

            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            assert!(slot.has_captured_graph(
                CoordinatorCudaGraphProgram::LayerPaddedLinearResidualAddBf16,
                padded_linear_residual_graph_signature(
                    &graph_key,
                    active_input_dim,
                    full_input_dim,
                    output_dim,
                    false
                )
            ));
            assert!(slot.graph_captures >= graph_captures_before);
            assert!(slot.graph_launches >= graph_launches_before + 2);
            Ok(())
        })();

        result
    }

    #[test]
    fn linear_residual_add_bf16_preloaded_attention_weight_padded_device_input_device_output_keeps_residual_on_device(
    ) -> Result<()> {
        if !cuda_reference_kernels_test_enabled() {
            return Ok(());
        }
        let library = match cuda_native_library() {
            Ok(library) => library,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseA,
            LayerWaveMode::Decode,
            1,
        )?;
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        let input = bf16_bytes(&[1.0, -2.0]);
        let mut input_buffer = match library.alloc_device_buffer(input.len()) {
            Ok(buffer) => buffer,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        if let Err(error) = library.copy_h2d(input_buffer, &input) {
            let _ = library.free_device_buffer(&mut input_buffer);
            if cuda_allocation_unavailable(&error) {
                return Ok(());
            }
            return Err(error);
        }

        let mut weight_values = vec![0.0_f32; 3 * GLM52_HIDDEN_SIZE];
        weight_values[0] = 0.5;
        weight_values[GLM52_HIDDEN_SIZE + 1] = -1.5;
        weight_values[2 * GLM52_HIDDEN_SIZE] = 2.0;
        weight_values[2 * GLM52_HIDDEN_SIZE + 2] = 9.0;
        let weight = bf16_bytes(&weight_values);
        let residual = bf16_bytes(&[1.0, 10.0, -1.0]);
        let weight_name = format!(
            "model.layers.3.self_attn.o_proj.weight.padded-device-output-residual.test.{}.{}",
            std::process::id(),
            line!()
        );
        match preload_resident_weight_from_host_staging(
            &weight_name,
            weight.len(),
            "test preloaded sparse padded device-input linear residual-add device-output graph-slot weight",
            |staging| {
                staging.copy_from_slice(&weight);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) => {
                let _ = library.free_device_buffer(&mut input_buffer);
                if cuda_allocation_unavailable(&error) {
                    return Ok(());
                }
                return Err(error);
            }
        }

        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Sparse-A decode graph key is registered");
        let acquisitions_before = registry.slots[graph_index]
            .lock()
            .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?
            .acquisitions;

        let output_result =
            linear_residual_add_rows_bf16_preloaded_resident_weight_padded_device_input_device_output(
                &weight_name,
                input_buffer,
                None,
                &residual,
                1,
                2,
                GLM52_HIDDEN_SIZE,
                3,
                3,
            );
        let free_result = library.free_device_buffer(&mut input_buffer);
        let output = match output_result {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        free_result?;
        let acquisitions_after = registry.slots[graph_index]
            .lock()
            .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?
            .acquisitions;

        assert_eq!(
            output.linear_backend,
            CUDA_REFERENCE_LINEAR_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(
            output.residual_add_backend,
            CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND
        );
        assert_eq!(output.linear_values.len(), 3);
        assert_eq!(output.residual_values.len(), 3);
        assert!((output.linear_values[0] - 0.5).abs() < 1.0e-3);
        assert!((output.linear_values[1] - 3.0).abs() < 1.0e-3);
        assert!((output.linear_values[2] - 2.0).abs() < 1.0e-3);
        assert!((output.residual_values[0] - 1.5).abs() < 1.0e-3);
        assert!((output.residual_values[1] - 13.0).abs() < 1.0e-3);
        assert!((output.residual_values[2] - 1.0).abs() < 1.0e-3);
        assert_eq!(output.residual_device.rows, 1);
        assert_eq!(output.residual_device.values_per_row, 3);
        assert_eq!(
            output.residual_device.backend,
            CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND
        );
        let device_values = output.residual_device.copy_to_host_values()?;
        assert_eq!(device_values.len(), output.residual_values.len());
        for (actual, expected) in device_values.iter().zip(output.residual_values.iter()) {
            assert!((actual - expected).abs() < 1.0e-3);
        }
        assert!(acquisitions_after > acquisitions_before);
        Ok(())
    }

    #[test]
    fn linear_rows_bf16_rejects_shape_mismatch_before_backend_selection() {
        let input = bf16_bytes(&[1.0, 2.0]);
        let weight = bf16_bytes(&[1.0, 2.0, 3.0]);
        let err = linear_rows_bf16(&input, &weight, None, 1, 2, 2).unwrap_err();

        assert!(err
            .to_string()
            .contains("BF16 linear weight byte length mismatch"));
    }

    #[test]
    fn padded_device_input_linear_allows_odd_bf16_zero_fill_width() -> Result<()> {
        let input_buffer = GlmrtDeviceBuffer {
            ptr: std::ptr::NonNull::<u8>::dangling().as_ptr().cast(),
            bytes: 6,
            device_id: 0,
            flags: 0,
        };
        let view = validate_linear_bf16_preloaded_resident_padded_device_input(
            input_buffer,
            None,
            1,
            3,
            3,
            1,
            1,
        )?;

        assert_eq!(view.active_row_bytes, 6);
        assert_eq!(view.padded_row_bytes, 6);
        assert_eq!(view.padded_input_bytes, 6);
        Ok(())
    }

    #[test]
    fn silu_gated_mlp_rows_uses_cpu_reference_when_cuda_reference_is_not_enabled() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let output = silu_gated_mlp_rows(
            &[1.0, -0.5],
            &[0.25, 0.5, -1.0, 0.25],
            &[0.5, -0.25, 0.75, 1.0],
            &[1.0, -0.5, 0.25, 2.0, -1.0, 0.5],
            1,
            2,
            2,
            3,
        )
        .unwrap();

        let gate0 = 1.0_f32 * 0.25 + -0.5 * 0.5;
        let gate1 = 1.0_f32 * -1.0 + -0.5 * 0.25;
        let up0 = 1.0_f32 * 0.5 + -0.5 * -0.25;
        let up1 = 1.0_f32 * 0.75 + -0.5 * 1.0;
        let act0 = gate0 / (1.0 + (-gate0).exp()) * up0;
        let act1 = gate1 / (1.0 + (-gate1).exp()) * up1;
        let expected = vec![
            act0 * 1.0 + act1 * -0.5,
            act0 * 0.25 + act1 * 2.0,
            act0 * -1.0 + act1 * 0.5,
        ];

        assert_eq!(output.values.len(), 3);
        for (actual, expected) in output.values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
        assert_eq!(output.backend, CPU_REFERENCE_SILU_GATED_MLP_BACKEND);
    }

    #[test]
    fn silu_gated_mlp_rows_rejects_shape_mismatch_before_backend_selection() {
        let err = silu_gated_mlp_rows(&[1.0, 2.0], &[1.0, 2.0], &[1.0, 2.0], &[1.0], 1, 2, 2, 1)
            .unwrap_err();

        assert!(err
            .to_string()
            .contains("SiLU-gated MLP gate weight length mismatch"));
    }

    #[test]
    fn silu_gated_mlp_rows_bf16_uses_cpu_reference_when_cuda_reference_is_not_enabled() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let input = bf16_bytes(&[1.0, -0.5]);
        let gate_weight = bf16_bytes(&[0.25, 0.5, -1.0, 0.25]);
        let up_weight = bf16_bytes(&[0.5, -0.25, 0.75, 1.0]);
        let down_weight = bf16_bytes(&[1.0, -0.5, 0.25, 2.0, -1.0, 0.5]);
        let output =
            silu_gated_mlp_rows_bf16(&input, &gate_weight, &up_weight, &down_weight, 1, 2, 2, 3)
                .unwrap();

        assert_eq!(output.values.len(), 3);
        assert!(output.values.iter().all(|value| value.is_finite()));
        assert_eq!(output.backend, CPU_REFERENCE_SILU_GATED_MLP_BF16_BACKEND);
    }

    #[test]
    fn silu_gated_mlp_rows_bf16_rejects_shape_mismatch_before_backend_selection() {
        let input = bf16_bytes(&[1.0, 2.0]);
        let gate_weight = bf16_bytes(&[1.0, 2.0]);
        let up_weight = bf16_bytes(&[1.0, 2.0]);
        let down_weight = bf16_bytes(&[1.0]);
        let err =
            silu_gated_mlp_rows_bf16(&input, &gate_weight, &up_weight, &down_weight, 1, 2, 2, 1)
                .unwrap_err();

        assert!(err
            .to_string()
            .contains("BF16 SiLU-gated MLP gate weight byte length mismatch"));
    }

    #[test]
    fn silu_gated_mlp_rows_bf16_resident_weight_uses_cpu_reference_when_cuda_reference_is_not_enabled(
    ) {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let input = bf16_bytes(&[1.0, -0.5]);
        let gate_weight = bf16_bytes(&[0.25, 0.5, -1.0, 0.25]);
        let up_weight = bf16_bytes(&[0.5, -0.25, 0.75, 1.0]);
        let down_weight = bf16_bytes(&[1.0, -0.5, 0.25, 2.0, -1.0, 0.5]);
        let output = silu_gated_mlp_rows_bf16_resident_weight(
            "model.layers.0.mlp.gate_proj.weight[rows=0..2]",
            "model.layers.0.mlp.up_proj.weight[rows=0..2]",
            "model.layers.0.mlp.down_proj.weight[rows=0..3,cols=0..2]",
            &input,
            &gate_weight,
            &up_weight,
            &down_weight,
            1,
            2,
            2,
            3,
        )
        .unwrap();

        assert_eq!(output.values.len(), 3);
        assert!(output.values.iter().all(|value| value.is_finite()));
        assert_eq!(output.backend, CPU_REFERENCE_SILU_GATED_MLP_BF16_BACKEND);
    }

    #[test]
    fn silu_gated_mlp_rows_bf16_preloaded_gate_up_requires_cuda_reference() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let input = bf16_bytes(&[1.0, -0.5]);
        let down_weight = bf16_bytes(&[1.0, -0.5]);
        let err = silu_gated_mlp_rows_bf16_preloaded_gate_up_resident_weight(
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight[rows=0..2,cols=0..1]",
            &input,
            &down_weight,
            1,
            2,
            1,
            2,
            2,
        )
        .unwrap_err();

        assert!(err
            .to_string()
            .contains("requires CUDA reference kernels and full-output shape"));
    }

    #[test]
    fn silu_gated_mlp_rows_bf16_preloaded_gate_up_rejects_prefix_past_full_rows() {
        let input = bf16_bytes(&[1.0, -0.5]);
        let down_weight = bf16_bytes(&[1.0, -0.5, 0.25, 2.0, -1.0, 0.5]);
        let err = silu_gated_mlp_rows_bf16_preloaded_gate_up_resident_weight(
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight[rows=0..2,cols=0..3]",
            &input,
            &down_weight,
            1,
            2,
            3,
            2,
            2,
        )
        .unwrap_err();

        assert!(err
            .to_string()
            .contains("intermediate prefix 3 exceeds full intermediate 2"));
    }

    #[test]
    fn silu_gated_mlp_rows_bf16_preloaded_gate_up_down_requires_cuda_reference() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let input = bf16_bytes(&[1.0, -0.5]);
        let err = silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight(
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            &input,
            1,
            2,
            1,
            2,
            2,
        )
        .unwrap_err();

        assert!(err
            .to_string()
            .contains("requires CUDA reference kernels and full-output shape"));
    }

    #[test]
    fn mlp_gate_up_down_graph_key_selects_dense_and_sparse_shared_layers() -> Result<()> {
        let dense = coord_dense_mlp_graph_key_for_gate_up_down_names(
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            1,
        )?
        .expect("dense MLP graph key");
        assert_eq!(dense.shape, CoordinatorGraphShape::CoordDense);
        assert_eq!(dense.row_bucket.row_capacity, 1);

        let dense_prefill = coord_dense_mlp_graph_key_for_gate_up_down_names(
            "model.layers.1.mlp.gate_proj.weight",
            "model.layers.1.mlp.up_proj.weight",
            "model.layers.1.mlp.down_proj.weight",
            17,
        )?
        .expect("dense MLP prefill graph key");
        assert_eq!(dense_prefill.shape, CoordinatorGraphShape::CoordDense);
        assert_eq!(dense_prefill.row_bucket.row_capacity, 32);

        assert!(coord_dense_mlp_graph_key_for_gate_up_down_names(
            "model.layers.3.mlp.gate_proj.weight",
            "model.layers.3.mlp.up_proj.weight",
            "model.layers.3.mlp.down_proj.weight",
            1,
        )?
        .is_none());
        let sparse_shared = coord_dense_mlp_graph_key_for_gate_up_down_names(
            "model.layers.3.mlp.shared_experts.gate_proj.weight",
            "model.layers.3.mlp.shared_experts.up_proj.weight",
            "model.layers.3.mlp.shared_experts.down_proj.weight",
            1,
        )?
        .expect("sparse shared MLP graph key");
        assert_eq!(sparse_shared.shape, CoordinatorGraphShape::CoordSparseA);
        assert_eq!(sparse_shared.row_bucket.row_capacity, 1);
        assert!(coord_dense_mlp_graph_key_for_gate_up_down_names(
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.1.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            1,
        )?
        .is_none());
        Ok(())
    }

    #[test]
    fn silu_gated_mlp_rows_bf16_preloaded_gate_up_dense_layer_uses_coord_dense_graph_slot(
    ) -> Result<()> {
        if !cuda_reference_kernels_test_enabled() {
            return Ok(());
        }
        let _python_capture_guard = coordinator_python_capture_test_override(true);
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordDense,
            LayerWaveMode::Decode,
            1,
        )?;
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        let mut input_values = vec![0.0_f32; GLM52_HIDDEN_SIZE];
        input_values[0] = 1.0;
        let input = bf16_bytes(&input_values);
        let mut gate_values = vec![0.0_f32; 2 * GLM52_HIDDEN_SIZE];
        gate_values[0] = 1.0;
        gate_values[GLM52_HIDDEN_SIZE] = -1.0;
        let gate = bf16_bytes(&gate_values);
        let mut up_values = vec![0.0_f32; 2 * GLM52_HIDDEN_SIZE];
        up_values[0] = 0.5;
        up_values[GLM52_HIDDEN_SIZE] = 2.0;
        let up = bf16_bytes(&up_values);
        let mut down_values = vec![0.0_f32; GLM52_HIDDEN_SIZE * 2];
        down_values[0] = 1.0;
        down_values[2 + 1] = 1.0;
        let down = bf16_bytes(&down_values);
        let gate_name = format!(
            "model.layers.0.mlp.gate_proj.weight.gateup.test.{}.{}",
            std::process::id(),
            line!()
        );
        let up_name = format!(
            "model.layers.0.mlp.up_proj.weight.gateup.test.{}.{}",
            std::process::id(),
            line!()
        );
        let down_name = format!(
            "model.layers.0.mlp.down_proj.weight.gateup.test.{}.{}",
            std::process::id(),
            line!()
        );
        for (name, bytes, label) in [
            (
                gate_name.as_str(),
                gate.as_slice(),
                "test preloaded gate/up dense MLP graph-slot gate weight",
            ),
            (
                up_name.as_str(),
                up.as_slice(),
                "test preloaded gate/up dense MLP graph-slot up weight",
            ),
        ] {
            match preload_resident_weight_from_host_staging(name, bytes.len(), label, |staging| {
                staging.copy_from_slice(bytes);
                Ok(())
            }) {
                Ok(()) => {}
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            }
        }

        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Dense decode graph key is registered");
        let signature = CoordinatorCudaGraphSignature::silu_gated_mlp_rows_bf16_down_stride(
            1,
            GLM52_HIDDEN_SIZE,
            2,
            2,
        );
        let (acquisitions_before, graph_launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_launches)
        };

        let output = match silu_gated_mlp_rows_bf16_preloaded_gate_up_resident_weight(
            &gate_name,
            &up_name,
            &down_name,
            &input,
            &down,
            1,
            GLM52_HIDDEN_SIZE,
            2,
            2,
            GLM52_HIDDEN_SIZE,
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let (acquisitions_after, graph_launches_after, has_graph) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (
                slot.acquisitions,
                slot.graph_launches,
                slot.captured_graphs.iter().any(|entry| {
                    entry.program == CoordinatorCudaGraphProgram::LayerTritonDenseMlpBf16
                }),
            )
        };

        let expected0 = (1.0_f32 / (1.0 + (-1.0_f32).exp())) * 0.5;
        let expected1 = (-1.0_f32 / (1.0 + 1.0_f32.exp())) * 2.0;
        assert_eq!(
            output.backend,
            TRITON_SILU_GATED_MLP_BF16_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(output.values.len(), GLM52_HIDDEN_SIZE);
        assert!((output.values[0] - expected0).abs() < 2.0e-2);
        assert!((output.values[1] - expected1).abs() < 2.0e-2);
        assert!(output.values[2..].iter().all(|value| value.abs() < 1.0e-6));
        assert!(acquisitions_after > acquisitions_before);
        assert!(graph_launches_after > graph_launches_before);
        assert!(has_graph);
        Ok(())
    }

    #[test]
    fn silu_gated_mlp_rows_bf16_preloaded_gate_up_down_dense_layer_uses_coord_dense_graph_slot(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _python_capture_guard = coordinator_python_capture_test_override(true);
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordDense,
            LayerWaveMode::Decode,
            1,
        )?;
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        let mut input_values = vec![0.0_f32; GLM52_HIDDEN_SIZE];
        input_values[0] = 1.0;
        let input = bf16_bytes(&input_values);
        let mut gate_values_a = vec![0.0_f32; 2 * GLM52_HIDDEN_SIZE];
        gate_values_a[0] = 1.0;
        gate_values_a[GLM52_HIDDEN_SIZE] = -1.0;
        let gate_a = bf16_bytes(&gate_values_a);
        let mut up_values_a = vec![0.0_f32; 2 * GLM52_HIDDEN_SIZE];
        up_values_a[0] = 0.5;
        up_values_a[GLM52_HIDDEN_SIZE] = 2.0;
        let up_a = bf16_bytes(&up_values_a);
        let mut down_values_a = vec![0.0_f32; GLM52_HIDDEN_SIZE * 2];
        down_values_a[0] = 1.0;
        down_values_a[2 + 1] = 1.0;
        let down_a = bf16_bytes(&down_values_a);
        let mut gate_values_b = vec![0.0_f32; 2 * GLM52_HIDDEN_SIZE];
        gate_values_b[0] = 0.5;
        gate_values_b[GLM52_HIDDEN_SIZE] = 1.5;
        let gate_b = bf16_bytes(&gate_values_b);
        let mut up_values_b = vec![0.0_f32; 2 * GLM52_HIDDEN_SIZE];
        up_values_b[0] = 1.0;
        up_values_b[GLM52_HIDDEN_SIZE] = -1.0;
        let up_b = bf16_bytes(&up_values_b);
        let mut down_values_b = vec![0.0_f32; GLM52_HIDDEN_SIZE * 2];
        down_values_b[0] = 2.0;
        down_values_b[2 + 1] = -0.5;
        let down_b = bf16_bytes(&down_values_b);
        let gate_name_a = format!(
            "model.layers.0.mlp.gate_proj.weight.test-a.{}.{}",
            std::process::id(),
            line!()
        );
        let up_name_a = format!(
            "model.layers.0.mlp.up_proj.weight.test-a.{}.{}",
            std::process::id(),
            line!()
        );
        let down_name_a = format!(
            "model.layers.0.mlp.down_proj.weight.test-a.{}.{}",
            std::process::id(),
            line!()
        );
        let gate_name_b = format!(
            "model.layers.0.mlp.gate_proj.weight.test-b.{}.{}",
            std::process::id(),
            line!()
        );
        let up_name_b = format!(
            "model.layers.0.mlp.up_proj.weight.test-b.{}.{}",
            std::process::id(),
            line!()
        );
        let down_name_b = format!(
            "model.layers.0.mlp.down_proj.weight.test-b.{}.{}",
            std::process::id(),
            line!()
        );
        for (name, bytes, label) in [
            (
                gate_name_a.as_str(),
                gate_a.as_slice(),
                "test preloaded dense MLP graph-slot gate weight a",
            ),
            (
                up_name_a.as_str(),
                up_a.as_slice(),
                "test preloaded dense MLP graph-slot up weight a",
            ),
            (
                down_name_a.as_str(),
                down_a.as_slice(),
                "test preloaded dense MLP graph-slot down weight a",
            ),
            (
                gate_name_b.as_str(),
                gate_b.as_slice(),
                "test preloaded dense MLP graph-slot gate weight b",
            ),
            (
                up_name_b.as_str(),
                up_b.as_slice(),
                "test preloaded dense MLP graph-slot up weight b",
            ),
            (
                down_name_b.as_str(),
                down_b.as_slice(),
                "test preloaded dense MLP graph-slot down weight b",
            ),
        ] {
            match preload_resident_weight_from_host_staging(name, bytes.len(), label, |staging| {
                staging.copy_from_slice(bytes);
                Ok(())
            }) {
                Ok(()) => {}
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            }
        }

        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Dense decode graph key is registered");
        let signature = CoordinatorCudaGraphSignature::silu_gated_mlp_rows_bf16_down_stride(
            1,
            GLM52_HIDDEN_SIZE,
            2,
            2,
        );
        let (acquisitions_before, graph_launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_launches)
        };

        let view_a = MlpGateUpDownResidentView {
            gate_up: MlpGateUpResidentView {
                full_bytes: gate_a.len(),
                offset_bytes: 0,
                view_bytes: gate_a.len(),
            },
            down_full_bytes: down_a.len(),
            down_stride: 2,
        };
        let view_b = MlpGateUpDownResidentView {
            gate_up: MlpGateUpResidentView {
                full_bytes: gate_b.len(),
                offset_bytes: 0,
                view_bytes: gate_b.len(),
            },
            down_full_bytes: down_b.len(),
            down_stride: 2,
        };
        let output_a = match cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight(
            &gate_name_a,
            &up_name_a,
            &down_name_a,
            &input,
            1,
            GLM52_HIDDEN_SIZE,
            2,
            GLM52_HIDDEN_SIZE,
            view_a,
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let output_b = match cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight(
            &gate_name_b,
            &up_name_b,
            &down_name_b,
            &input,
            1,
            GLM52_HIDDEN_SIZE,
            2,
            GLM52_HIDDEN_SIZE,
            view_b,
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let (acquisitions_after, graph_launches_after, has_graph) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (
                slot.acquisitions,
                slot.graph_launches,
                slot.captured_graphs.iter().any(|entry| {
                    entry.program == CoordinatorCudaGraphProgram::LayerTritonDenseMlpBf16
                }),
            )
        };

        let expected_a0 = (1.0_f32 / (1.0 + (-1.0_f32).exp())) * 0.5;
        let expected_a1 = (-1.0_f32 / (1.0 + 1.0_f32.exp())) * 2.0;
        let expected_b0 = (0.5_f32 / (1.0 + (-0.5_f32).exp())) * 1.0 * 2.0;
        let expected_b1 = (1.5_f32 / (1.0 + (-1.5_f32).exp())) * -0.5 * -1.0;
        assert_eq!(
            output_a.backend,
            TRITON_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(
            output_b.backend,
            TRITON_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(output_a.values.len(), GLM52_HIDDEN_SIZE);
        assert_eq!(output_b.values.len(), GLM52_HIDDEN_SIZE);
        assert!((output_a.values[0] - expected_a0).abs() < 2.0e-2);
        assert!((output_a.values[1] - expected_a1).abs() < 2.0e-2);
        assert!((output_b.values[0] - expected_b0).abs() < 2.0e-2);
        assert!((output_b.values[1] - expected_b1).abs() < 2.0e-2);
        assert!((output_b.values[0] - output_a.values[0]).abs() > 1.0e-1);
        assert!(output_a.values[2..]
            .iter()
            .all(|value| value.abs() < 1.0e-6));
        assert!(output_b.values[2..]
            .iter()
            .all(|value| value.abs() < 1.0e-6));
        assert!(acquisitions_after > acquisitions_before);
        assert!(graph_launches_after >= graph_launches_before + 2);
        assert!(has_graph);
        Ok(())
    }

    #[test]
    fn silu_gated_mlp_rows_bf16_sparse_shared_layer_uses_triton_graph_when_python_enabled(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _python_capture_guard = coordinator_python_capture_test_override(true);
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseA,
            LayerWaveMode::Decode,
            1,
        )?;
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        let mut input_values = vec![0.0_f32; GLM52_HIDDEN_SIZE];
        input_values[0] = 1.0;
        let input = bf16_bytes(&input_values);
        let mut gate_values = vec![0.0_f32; 2 * GLM52_HIDDEN_SIZE];
        gate_values[0] = 1.0;
        gate_values[GLM52_HIDDEN_SIZE] = -1.0;
        let gate = bf16_bytes(&gate_values);
        let mut up_values = vec![0.0_f32; 2 * GLM52_HIDDEN_SIZE];
        up_values[0] = 0.5;
        up_values[GLM52_HIDDEN_SIZE] = 2.0;
        let up = bf16_bytes(&up_values);
        let mut down_values = vec![0.0_f32; GLM52_HIDDEN_SIZE * 2];
        down_values[0] = 1.0;
        down_values[2 + 1] = 1.0;
        let down = bf16_bytes(&down_values);
        let gate_name = format!(
            "model.layers.3.mlp.shared_experts.gate_proj.weight.triton-sparse.test.{}.{}",
            std::process::id(),
            line!()
        );
        let up_name = format!(
            "model.layers.3.mlp.shared_experts.up_proj.weight.triton-sparse.test.{}.{}",
            std::process::id(),
            line!()
        );
        let down_name = format!(
            "model.layers.3.mlp.shared_experts.down_proj.weight.triton-sparse.test.{}.{}",
            std::process::id(),
            line!()
        );
        for (name, bytes, label) in [
            (
                gate_name.as_str(),
                gate.as_slice(),
                "test Triton sparse shared MLP graph-slot gate weight",
            ),
            (
                up_name.as_str(),
                up.as_slice(),
                "test Triton sparse shared MLP graph-slot up weight",
            ),
            (
                down_name.as_str(),
                down.as_slice(),
                "test Triton sparse shared MLP graph-slot down weight",
            ),
        ] {
            match preload_resident_weight_from_host_staging(name, bytes.len(), label, |staging| {
                staging.copy_from_slice(bytes);
                Ok(())
            }) {
                Ok(()) => {}
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            }
        }

        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Sparse-A decode graph key is registered");
        let (acquisitions_before, graph_launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_launches)
        };

        let view = MlpGateUpDownResidentView {
            gate_up: MlpGateUpResidentView {
                full_bytes: gate.len(),
                offset_bytes: 0,
                view_bytes: gate.len(),
            },
            down_full_bytes: down.len(),
            down_stride: 2,
        };
        let output = match cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight(
            &gate_name,
            &up_name,
            &down_name,
            &input,
            1,
            GLM52_HIDDEN_SIZE,
            2,
            GLM52_HIDDEN_SIZE,
            view,
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let (acquisitions_after, graph_launches_after, has_triton_graph) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (
                slot.acquisitions,
                slot.graph_launches,
                slot.captured_graphs.iter().any(|entry| {
                    entry.program == CoordinatorCudaGraphProgram::LayerTritonDenseMlpBf16
                }),
            )
        };

        let expected0 = (1.0_f32 / (1.0 + (-1.0_f32).exp())) * 0.5;
        let expected1 = (-1.0_f32 / (1.0 + 1.0_f32.exp())) * 2.0;
        assert_eq!(
            output.backend,
            TRITON_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(output.values.len(), GLM52_HIDDEN_SIZE);
        assert!((output.values[0] - expected0).abs() < 2.0e-2);
        assert!((output.values[1] - expected1).abs() < 2.0e-2);
        assert!(output.values[2..].iter().all(|value| value.abs() < 1.0e-6));
        assert!(acquisitions_after > acquisitions_before);
        assert!(graph_launches_after > graph_launches_before);
        assert!(has_triton_graph);
        Ok(())
    }

    #[test]
    fn silu_gated_mlp_rows_bf16_preloaded_gate_up_down_dense_layer_uses_triton_graph_when_python_enabled(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _python_capture_guard = coordinator_python_capture_test_override(true);
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordDense,
            LayerWaveMode::Decode,
            1,
        )?;
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        let mut input_values = vec![0.0_f32; GLM52_HIDDEN_SIZE];
        input_values[0] = 1.0;
        let input = bf16_bytes(&input_values);
        let mut gate_values = vec![0.0_f32; 2 * GLM52_HIDDEN_SIZE];
        gate_values[0] = 1.0;
        gate_values[GLM52_HIDDEN_SIZE] = -1.0;
        let gate = bf16_bytes(&gate_values);
        let mut up_values = vec![0.0_f32; 2 * GLM52_HIDDEN_SIZE];
        up_values[0] = 0.5;
        up_values[GLM52_HIDDEN_SIZE] = 2.0;
        let up = bf16_bytes(&up_values);
        let mut down_values = vec![0.0_f32; GLM52_HIDDEN_SIZE * 2];
        down_values[0] = 1.0;
        down_values[2 + 1] = 1.0;
        let down = bf16_bytes(&down_values);
        let gate_name = format!(
            "model.layers.0.mlp.gate_proj.weight.triton.test.{}.{}",
            std::process::id(),
            line!()
        );
        let up_name = format!(
            "model.layers.0.mlp.up_proj.weight.triton.test.{}.{}",
            std::process::id(),
            line!()
        );
        let down_name = format!(
            "model.layers.0.mlp.down_proj.weight.triton.test.{}.{}",
            std::process::id(),
            line!()
        );
        for (name, bytes, label) in [
            (
                gate_name.as_str(),
                gate.as_slice(),
                "test Triton dense MLP graph-slot gate weight",
            ),
            (
                up_name.as_str(),
                up.as_slice(),
                "test Triton dense MLP graph-slot up weight",
            ),
            (
                down_name.as_str(),
                down.as_slice(),
                "test Triton dense MLP graph-slot down weight",
            ),
        ] {
            match preload_resident_weight_from_host_staging(name, bytes.len(), label, |staging| {
                staging.copy_from_slice(bytes);
                Ok(())
            }) {
                Ok(()) => {}
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            }
        }

        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Dense decode graph key is registered");
        let (acquisitions_before, graph_launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_launches)
        };

        let view = MlpGateUpDownResidentView {
            gate_up: MlpGateUpResidentView {
                full_bytes: gate.len(),
                offset_bytes: 0,
                view_bytes: gate.len(),
            },
            down_full_bytes: down.len(),
            down_stride: 2,
        };
        let output = match cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight(
            &gate_name,
            &up_name,
            &down_name,
            &input,
            1,
            GLM52_HIDDEN_SIZE,
            2,
            GLM52_HIDDEN_SIZE,
            view,
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let (acquisitions_after, graph_launches_after, has_triton_graph) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (
                slot.acquisitions,
                slot.graph_launches,
                slot.captured_graphs.iter().any(|entry| {
                    entry.program == CoordinatorCudaGraphProgram::LayerTritonDenseMlpBf16
                }),
            )
        };

        let expected0 = (1.0_f32 / (1.0 + (-1.0_f32).exp())) * 0.5;
        let expected1 = (-1.0_f32 / (1.0 + 1.0_f32.exp())) * 2.0;
        assert_eq!(
            output.backend,
            TRITON_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(output.values.len(), GLM52_HIDDEN_SIZE);
        assert!((output.values[0] - expected0).abs() < 2.0e-2);
        assert!((output.values[1] - expected1).abs() < 2.0e-2);
        assert!(output.values[2..].iter().all(|value| value.abs() < 1.0e-6));
        assert!(acquisitions_after > acquisitions_before);
        assert!(graph_launches_after > graph_launches_before);
        assert!(has_triton_graph);
        Ok(())
    }

    #[test]
    fn silu_gated_mlp_rows_bf16_coord_dense_graph_replays_same_bucket_when_rows_change(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _python_capture_guard = coordinator_python_capture_test_override(true);
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);

        let result = (|| -> Result<()> {
            let rows_first = 2;
            let rows_second = 4;
            let intermediate = 2;
            let graph_key = CoordinatorGraphKey::glm52_bf16(
                CoordinatorGraphShape::CoordDense,
                LayerWaveMode::Prefill,
                rows_first,
            )?;
            assert_eq!(
                graph_key,
                CoordinatorGraphKey::glm52_bf16(
                    CoordinatorGraphShape::CoordDense,
                    LayerWaveMode::Prefill,
                    rows_second,
                )?
            );
            let registry = match coordinator_cuda_graph_workspace_registry() {
                Ok(registry) => registry,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };

            let mut gate_values = vec![0.0_f32; intermediate * GLM52_HIDDEN_SIZE];
            gate_values[0] = 1.0;
            gate_values[GLM52_HIDDEN_SIZE] = -1.0;
            let gate = bf16_bytes(&gate_values);
            let mut up_values = vec![0.0_f32; intermediate * GLM52_HIDDEN_SIZE];
            up_values[0] = 0.5;
            up_values[GLM52_HIDDEN_SIZE] = 2.0;
            let up = bf16_bytes(&up_values);
            let mut down_values = vec![0.0_f32; GLM52_HIDDEN_SIZE * intermediate];
            down_values[0] = 1.0;
            down_values[intermediate + 1] = 1.0;
            let down = bf16_bytes(&down_values);
            let gate_name = format!(
                "model.layers.0.mlp.gate_proj.weight.row-bucket.test.{}.{}",
                std::process::id(),
                line!()
            );
            let up_name = format!(
                "model.layers.0.mlp.up_proj.weight.row-bucket.test.{}.{}",
                std::process::id(),
                line!()
            );
            let down_name = format!(
                "model.layers.0.mlp.down_proj.weight.row-bucket.test.{}.{}",
                std::process::id(),
                line!()
            );
            for (name, bytes, label) in [
                (
                    gate_name.as_str(),
                    gate.as_slice(),
                    "test preloaded dense MLP row-bucket gate weight",
                ),
                (
                    up_name.as_str(),
                    up.as_slice(),
                    "test preloaded dense MLP row-bucket up weight",
                ),
                (
                    down_name.as_str(),
                    down.as_slice(),
                    "test preloaded dense MLP row-bucket down weight",
                ),
            ] {
                match preload_resident_weight_from_host_staging(
                    name,
                    bytes.len(),
                    label,
                    |staging| {
                        staging.copy_from_slice(bytes);
                        Ok(())
                    },
                ) {
                    Ok(()) => {}
                    Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                    Err(error) => return Err(error),
                }
            }

            let graph_index = registry
                .keys()
                .iter()
                .position(|key| key == &graph_key)
                .expect("Coord-Dense prefill graph key is registered");
            let signature = dense_mlp_graph_signature(
                &graph_key,
                GLM52_HIDDEN_SIZE,
                intermediate,
                intermediate,
            );
            let (graph_captures_before, graph_launches_before) = {
                let slot = registry.slots[graph_index]
                    .lock()
                    .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
                (slot.graph_captures, slot.graph_launches)
            };

            let input_for_rows = |row_values: &[f32]| -> Vec<u8> {
                let mut input_values = vec![0.0_f32; row_values.len() * GLM52_HIDDEN_SIZE];
                for (row, value) in row_values.iter().copied().enumerate() {
                    input_values[row * GLM52_HIDDEN_SIZE] = value;
                }
                bf16_bytes(&input_values)
            };
            let assert_row_outputs =
                |actual: &[f32], row_values: &[f32], tolerance: f32| -> Result<()> {
                    assert_eq!(actual.len(), row_values.len() * GLM52_HIDDEN_SIZE);
                    for (row, value) in row_values.iter().copied().enumerate() {
                        let base = row * GLM52_HIDDEN_SIZE;
                        let expected0 = (value / (1.0 + (-value).exp())) * (value * 0.5);
                        let expected1 = ((-value) / (1.0 + value.exp())) * (value * 2.0);
                        assert!((actual[base] - expected0).abs() < tolerance);
                        assert!((actual[base + 1] - expected1).abs() < tolerance);
                        assert!(actual[base + 2..base + GLM52_HIDDEN_SIZE]
                            .iter()
                            .all(|value| value.abs() < 1.0e-6));
                    }
                    Ok(())
                };

            let view = MlpGateUpDownResidentView {
                gate_up: MlpGateUpResidentView {
                    full_bytes: gate.len(),
                    offset_bytes: 0,
                    view_bytes: gate.len(),
                },
                down_full_bytes: down.len(),
                down_stride: intermediate,
            };
            let row_values0 = [1.0_f32, -0.5];
            let input0 = input_for_rows(&row_values0);
            let output0 = match cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight(
                &gate_name,
                &up_name,
                &down_name,
                &input0,
                rows_first,
                GLM52_HIDDEN_SIZE,
                intermediate,
                GLM52_HIDDEN_SIZE,
                view,
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            assert_eq!(
                output0.backend,
                TRITON_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND
            );
            assert_row_outputs(&output0.values, &row_values0, 2.0e-2)?;

            let (graph_captures_after_first, graph_launches_after_first) = {
                let slot = registry.slots[graph_index]
                    .lock()
                    .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
                assert!(slot.captured_graphs.iter().any(|entry| {
                    entry.program == CoordinatorCudaGraphProgram::LayerTritonDenseMlpBf16
                }));
                (slot.graph_captures, slot.graph_launches)
            };

            let row_values1 = [-0.75_f32, 0.25, 1.5, -1.0];
            let input1 = input_for_rows(&row_values1);
            let output1 = match cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight(
                &gate_name,
                &up_name,
                &down_name,
                &input1,
                rows_second,
                GLM52_HIDDEN_SIZE,
                intermediate,
                GLM52_HIDDEN_SIZE,
                view,
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            assert_eq!(
                output1.backend,
                TRITON_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND
            );
            assert_row_outputs(&output1.values, &row_values1, 2.0e-2)?;

            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            assert!(slot.captured_graphs.iter().any(|entry| {
                entry.program == CoordinatorCudaGraphProgram::LayerTritonDenseMlpBf16
            }));
            assert!(slot.graph_captures >= graph_captures_before);
            assert_eq!(slot.graph_captures, graph_captures_after_first);
            assert!(slot.graph_launches >= graph_launches_before + 2);
            assert!(slot.graph_launches >= graph_launches_after_first + 1);
            Ok(())
        })();

        result
    }

    #[test]
    fn silu_gated_mlp_rows_bf16_preloaded_gate_up_down_device_input_keeps_output_on_device(
    ) -> Result<()> {
        if !cuda_reference_kernels_test_enabled() {
            return Ok(());
        }
        let _python_capture_guard = coordinator_python_capture_test_override(true);
        let library = match cuda_native_library() {
            Ok(library) => library,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let dense_graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordDense,
            LayerWaveMode::Decode,
            1,
        )?;
        let sparse_a_graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseA,
            LayerWaveMode::Decode,
            1,
        )?;
        let sparse_b_graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseB,
            LayerWaveMode::Decode,
            1,
        )?;
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        let mut input_values = vec![0.0_f32; GLM52_HIDDEN_SIZE];
        input_values[0] = 1.0;
        let input = bf16_bytes(&input_values);
        let mut input_buffer = match library.alloc_device_buffer(input.len()) {
            Ok(buffer) => buffer,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        if let Err(error) = library.copy_h2d(input_buffer, &input) {
            let _ = library.free_device_buffer(&mut input_buffer);
            if cuda_allocation_unavailable(&error) {
                return Ok(());
            }
            return Err(error);
        }

        let mut gate_values = vec![0.0_f32; 2 * GLM52_HIDDEN_SIZE];
        gate_values[0] = 1.0;
        gate_values[GLM52_HIDDEN_SIZE] = -1.0;
        let gate = bf16_bytes(&gate_values);
        let mut up_values = vec![0.0_f32; 2 * GLM52_HIDDEN_SIZE];
        up_values[0] = 0.5;
        up_values[GLM52_HIDDEN_SIZE] = 2.0;
        let up = bf16_bytes(&up_values);
        let mut down_values = vec![0.0_f32; GLM52_HIDDEN_SIZE * 2];
        down_values[0] = 1.0;
        down_values[2 + 1] = 1.0;
        let down = bf16_bytes(&down_values);
        let gate_name = format!(
            "model.layers.0.mlp.gate_proj.weight.device-input.test.{}.{}",
            std::process::id(),
            line!()
        );
        let up_name = format!(
            "model.layers.0.mlp.up_proj.weight.device-input.test.{}.{}",
            std::process::id(),
            line!()
        );
        let down_name = format!(
            "model.layers.0.mlp.down_proj.weight.device-input.test.{}.{}",
            std::process::id(),
            line!()
        );
        let sparse_gate_name = format!(
            "model.layers.3.mlp.shared_experts.gate_proj.weight.device-output-only.test.{}.{}",
            std::process::id(),
            line!()
        );
        let sparse_up_name = format!(
            "model.layers.3.mlp.shared_experts.up_proj.weight.device-output-only.test.{}.{}",
            std::process::id(),
            line!()
        );
        let sparse_down_name = format!(
            "model.layers.3.mlp.shared_experts.down_proj.weight.device-output-only.test.{}.{}",
            std::process::id(),
            line!()
        );
        for (name, bytes, label) in [
            (
                gate_name.as_str(),
                gate.as_slice(),
                "test preloaded dense MLP device-input gate weight",
            ),
            (
                up_name.as_str(),
                up.as_slice(),
                "test preloaded dense MLP device-input up weight",
            ),
            (
                down_name.as_str(),
                down.as_slice(),
                "test preloaded dense MLP device-input down weight",
            ),
            (
                sparse_gate_name.as_str(),
                gate.as_slice(),
                "test preloaded sparse shared MLP device-output-only gate weight",
            ),
            (
                sparse_up_name.as_str(),
                up.as_slice(),
                "test preloaded sparse shared MLP device-output-only up weight",
            ),
            (
                sparse_down_name.as_str(),
                down.as_slice(),
                "test preloaded sparse shared MLP device-output-only down weight",
            ),
        ] {
            match preload_resident_weight_from_host_staging(name, bytes.len(), label, |staging| {
                staging.copy_from_slice(bytes);
                Ok(())
            }) {
                Ok(()) => {}
                Err(error) => {
                    let _ = library.free_device_buffer(&mut input_buffer);
                    if cuda_allocation_unavailable(&error) {
                        return Ok(());
                    }
                    return Err(error);
                }
            }
        }

        let dense_graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &dense_graph_key)
            .expect("Coord-Dense decode graph key is registered");
        let sparse_a_graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &sparse_a_graph_key)
            .expect("Coord-Sparse-A decode graph key is registered");
        let sparse_b_graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &sparse_b_graph_key)
            .expect("Coord-Sparse-B decode graph key is registered");
        let (dense_acquisitions_before, dense_graph_launches_before) = {
            let slot = registry.slots[dense_graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test dense graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_launches)
        };
        let sparse_b_acquisitions_before = registry.slots[sparse_b_graph_index]
            .lock()
            .map_err(|_| anyhow::anyhow!("test sparse-b graph slot already borrowed"))?
            .acquisitions;

        let output_result =
            cuda_silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight_device_input_device_output(
                &gate_name,
                &up_name,
                &down_name,
                input_buffer,
                1,
                GLM52_HIDDEN_SIZE,
                2,
                GLM52_HIDDEN_SIZE,
                MlpGateUpDownResidentView {
                    gate_up: MlpGateUpResidentView {
                        full_bytes: gate.len(),
                        offset_bytes: 0,
                        view_bytes: gate.len(),
                    },
                    down_full_bytes: down.len(),
                    down_stride: 2,
                },
            );
        let output = match output_result {
            Ok(output) => output,
            Err(error) => {
                let _ = library.free_device_buffer(&mut input_buffer);
                if cuda_allocation_unavailable(&error) {
                    return Ok(());
                }
                return Err(error);
            }
        };
        let device_only_residual = residual_add_bf16_device_inputs_device_output(
            &output.device_output,
            &output.device_output,
        )?;
        let residual =
            residual_add_bf16_device_inputs_output(&output.device_output, &output.device_output)?;
        let dense_output_only_result =
            silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight_device_input_device_output_only(
                &gate_name,
                &up_name,
                &down_name,
                input_buffer,
                1,
                GLM52_HIDDEN_SIZE,
                2,
                2,
                GLM52_HIDDEN_SIZE,
            );
        let dense_output_only = match dense_output_only_result {
            Ok(output) => output,
            Err(error) => {
                let _ = library.free_device_buffer(&mut input_buffer);
                if cuda_allocation_unavailable(&error) {
                    return Ok(());
                }
                return Err(error);
            }
        };
        let (dense_acquisitions_after, dense_graph_launches_after, dense_has_mlp_graph) = {
            let slot = registry.slots[dense_graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test dense graph slot already borrowed"))?;
            (
                slot.acquisitions,
                slot.graph_launches,
                slot.captured_graphs.iter().any(|entry| {
                    entry.program == CoordinatorCudaGraphProgram::LayerTritonDenseMlpBf16
                }),
            )
        };
        let (sparse_a_acquisitions_before, sparse_a_graph_launches_before) = {
            let slot = registry.slots[sparse_a_graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test sparse-a graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_launches)
        };
        let sparse_b_acquisitions_after = registry.slots[sparse_b_graph_index]
            .lock()
            .map_err(|_| anyhow::anyhow!("test sparse-b graph slot already borrowed"))?
            .acquisitions;

        let sparse_output_only_result =
            silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight_device_input_device_output_only(
                &sparse_gate_name,
                &sparse_up_name,
                &sparse_down_name,
                input_buffer,
                1,
                GLM52_HIDDEN_SIZE,
                2,
                2,
                GLM52_HIDDEN_SIZE,
            );
        let free_result = library.free_device_buffer(&mut input_buffer);
        let sparse_output_only = match sparse_output_only_result {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        free_result?;
        let (sparse_a_acquisitions_after, sparse_a_graph_launches_after) = {
            let slot = registry.slots[sparse_a_graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test sparse-a graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_launches)
        };
        let expected0 = (1.0_f32 / (1.0 + (-1.0_f32).exp())) * 0.5;
        let expected1 = (-1.0_f32 / (1.0 + 1.0_f32.exp())) * 2.0;
        assert_eq!(
            output.backend,
            TRITON_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(output.values.len(), GLM52_HIDDEN_SIZE);
        assert_eq!(output.device_output.rows, 1);
        assert_eq!(output.device_output.values_per_row, GLM52_HIDDEN_SIZE);
        assert!((output.values[0] - expected0).abs() < 2.0e-2);
        assert!((output.values[1] - expected1).abs() < 2.0e-2);
        assert!(output.values[2..].iter().all(|value| value.abs() < 1.0e-6));
        let device_values = output.device_output.copy_to_host_values()?;
        assert_eq!(device_values.len(), output.values.len());
        for (actual, expected) in device_values.iter().zip(output.values.iter()) {
            assert!((actual - expected).abs() < 1.0e-3);
        }
        let dense_output_only_values = dense_output_only.copy_to_host_values()?;
        assert_eq!(dense_output_only.rows, 1);
        assert_eq!(dense_output_only.values_per_row, GLM52_HIDDEN_SIZE);
        assert!((dense_output_only_values[0] - expected0).abs() < 2.0e-2);
        assert!((dense_output_only_values[1] - expected1).abs() < 2.0e-2);
        assert!(dense_output_only_values[2..]
            .iter()
            .all(|value| value.abs() < 1.0e-6));
        let sparse_values = sparse_output_only.copy_to_host_values()?;
        assert_eq!(sparse_output_only.rows, 1);
        assert_eq!(sparse_output_only.values_per_row, GLM52_HIDDEN_SIZE);
        assert!((sparse_values[0] - expected0).abs() < 2.0e-2);
        assert!((sparse_values[1] - expected1).abs() < 2.0e-2);
        assert!(sparse_values[2..].iter().all(|value| value.abs() < 1.0e-6));
        assert_eq!(
            device_only_residual.backend,
            CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND
        );
        assert_eq!(device_only_residual.rows, 1);
        assert_eq!(device_only_residual.values_per_row, GLM52_HIDDEN_SIZE);
        let device_only_values = device_only_residual.copy_to_host_values()?;
        assert!((device_only_values[0] - (expected0 + expected0)).abs() < 4.0e-2);
        assert!((device_only_values[1] - (expected1 + expected1)).abs() < 4.0e-2);
        assert_eq!(residual.backend, CUDA_REFERENCE_RESIDUAL_ADD_BF16_BACKEND);
        assert_eq!(residual.device_output.rows, 1);
        assert_eq!(residual.device_output.values_per_row, GLM52_HIDDEN_SIZE);
        assert!((residual.values[0] - (expected0 + expected0)).abs() < 4.0e-2);
        assert!((residual.values[1] - (expected1 + expected1)).abs() < 4.0e-2);
        assert!(dense_acquisitions_after > dense_acquisitions_before);
        assert!(dense_graph_launches_after > dense_graph_launches_before);
        assert!(dense_has_mlp_graph);
        assert!(sparse_a_acquisitions_after > sparse_a_acquisitions_before);
        assert!(sparse_a_graph_launches_after > sparse_a_graph_launches_before);
        assert!(sparse_b_acquisitions_after > sparse_b_acquisitions_before);
        Ok(())
    }

    #[test]
    fn silu_gated_mlp_rows_bf16_resident_dense_layer_uses_coord_dense_graph_slot() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _python_capture_guard = coordinator_python_capture_test_override(true);
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordDense,
            LayerWaveMode::Decode,
            1,
        )?;
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        let mut input_values = vec![0.0_f32; GLM52_HIDDEN_SIZE];
        input_values[0] = 1.0;
        let input = bf16_bytes(&input_values);
        let mut gate_values = vec![0.0_f32; 2 * GLM52_HIDDEN_SIZE];
        gate_values[0] = 1.0;
        gate_values[GLM52_HIDDEN_SIZE] = -1.0;
        let gate = bf16_bytes(&gate_values);
        let mut up_values = vec![0.0_f32; 2 * GLM52_HIDDEN_SIZE];
        up_values[0] = 0.5;
        up_values[GLM52_HIDDEN_SIZE] = 2.0;
        let up = bf16_bytes(&up_values);
        let mut down_values = vec![0.0_f32; GLM52_HIDDEN_SIZE * 2];
        down_values[0] = 1.0;
        down_values[2 + 1] = 1.0;
        let down = bf16_bytes(&down_values);
        let gate_name = format!(
            "model.layers.0.mlp.gate_proj.weight.resident.test.{}.{}",
            std::process::id(),
            line!()
        );
        let up_name = format!(
            "model.layers.0.mlp.up_proj.weight.resident.test.{}.{}",
            std::process::id(),
            line!()
        );
        let down_name = format!(
            "model.layers.0.mlp.down_proj.weight.resident.test.{}.{}",
            std::process::id(),
            line!()
        );

        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Dense decode graph key is registered");
        let signature = CoordinatorCudaGraphSignature::silu_gated_mlp_rows_bf16_down_stride(
            1,
            GLM52_HIDDEN_SIZE,
            2,
            2,
        );
        let (acquisitions_before, graph_launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_launches)
        };

        let output = match cuda_silu_gated_mlp_rows_bf16_resident_weight(
            &gate_name,
            &up_name,
            &down_name,
            &input,
            &gate,
            &up,
            &down,
            1,
            GLM52_HIDDEN_SIZE,
            2,
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let (acquisitions_after, graph_launches_after, has_graph) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (
                slot.acquisitions,
                slot.graph_launches,
                slot.captured_graphs.iter().any(|entry| {
                    entry.program == CoordinatorCudaGraphProgram::LayerTritonDenseMlpBf16
                }),
            )
        };

        let expected0 = (1.0_f32 / (1.0 + (-1.0_f32).exp())) * 0.5;
        let expected1 = (-1.0_f32 / (1.0 + 1.0_f32.exp())) * 2.0;
        assert_eq!(
            output.backend,
            TRITON_SILU_GATED_MLP_BF16_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(output.values.len(), GLM52_HIDDEN_SIZE);
        assert!((output.values[0] - expected0).abs() < 2.0e-2);
        assert!((output.values[1] - expected1).abs() < 2.0e-2);
        assert!(output.values[2..].iter().all(|value| value.abs() < 1.0e-6));
        assert!(acquisitions_after > acquisitions_before);
        assert!(graph_launches_after > graph_launches_before);
        assert!(has_graph);
        Ok(())
    }

    #[test]
    fn silu_gated_mlp_rows_bf16_preloaded_gate_up_down_rejects_prefix_past_full_rows() {
        let input = bf16_bytes(&[1.0, -0.5]);
        let err = silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight(
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            &input,
            1,
            2,
            3,
            2,
            2,
        )
        .unwrap_err();

        assert!(err
            .to_string()
            .contains("intermediate prefix 3 exceeds full intermediate 2"));
    }

    #[test]
    fn silu_gated_mlp_rows_bf16_resident_weight_rejects_empty_weight_name() {
        let input = bf16_bytes(&[1.0, 2.0]);
        let gate_weight = bf16_bytes(&[1.0, 2.0]);
        let up_weight = bf16_bytes(&[1.0, 2.0]);
        let down_weight = bf16_bytes(&[1.0]);
        let err = silu_gated_mlp_rows_bf16_resident_weight(
            "",
            "model.layers.0.mlp.up_proj.weight[rows=0..1]",
            "model.layers.0.mlp.down_proj.weight[rows=0..1,cols=0..1]",
            &input,
            &gate_weight,
            &up_weight,
            &down_weight,
            1,
            2,
            1,
            1,
        )
        .unwrap_err();

        assert!(err.to_string().contains("weight name must not be empty"));
    }

    #[test]
    fn router_topk_uses_cpu_reference_when_cuda_reference_is_not_enabled() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let output = router_topk(
            &[1.0, 0.0],
            &[
                0.0_f32, 0.0, //
                1.0, 0.0, //
                0.5, 0.0, //
            ],
            &[0.9, 0.0, 0.2],
            1,
            2,
            3,
            2,
        )
        .unwrap();

        assert_eq!(output.indices, vec![0, 2]);
        assert_eq!(output.scores.len(), 2);
        assert_eq!(output.weights.len(), 2);
        assert!((output.weights.iter().sum::<f32>() - GLM52_ROUTED_SCALING_FACTOR).abs() < 1.0e-6);
        assert_eq!(output.backend, CPU_REFERENCE_ROUTER_TOPK_BACKEND);
    }

    #[test]
    fn router_topk_rejects_shape_mismatch_before_backend_selection() {
        let err = router_topk(&[1.0, 0.0], &[0.0, 0.0, 1.0], &[0.0, 0.0], 1, 2, 2, 1).unwrap_err();

        assert!(err
            .to_string()
            .contains("router top-k weight length mismatch"));
    }

    #[test]
    fn router_topk_bf16_uses_cpu_reference_when_cuda_reference_is_not_enabled() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let hidden = bf16_bytes(&[1.0, 0.0]);
        let router_weight = bf16_bytes(&[
            0.0_f32, 0.0, //
            1.0, 0.0, //
            0.5, 0.0, //
        ]);
        let output =
            router_topk_bf16(&hidden, &router_weight, &[0.9, 0.0, 0.2], 1, 2, 3, 2).unwrap();

        assert_eq!(output.indices, vec![0, 2]);
        assert_eq!(output.scores.len(), 2);
        assert_eq!(output.weights.len(), 2);
        assert!((output.weights.iter().sum::<f32>() - GLM52_ROUTED_SCALING_FACTOR).abs() < 1.0e-6);
        assert_eq!(output.backend, CPU_REFERENCE_ROUTER_TOPK_BF16_BACKEND);
    }

    #[test]
    fn router_topk_bf16_full_width_uses_coord_sparse_a_graph_slot_when_cuda_reference_is_enabled(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let mut hidden = vec![0.0_f32; GLM52_HIDDEN_SIZE];
        hidden[0] = 1.0;
        let hidden = bf16_bytes(&hidden);
        let mut router_weight_values = vec![0.0_f32; 3 * GLM52_HIDDEN_SIZE];
        router_weight_values[GLM52_HIDDEN_SIZE] = 1.0;
        router_weight_values[2 * GLM52_HIDDEN_SIZE] = 0.5;
        let router_weight = bf16_bytes(&router_weight_values);
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseA,
            LayerWaveMode::Decode,
            1,
        )?;
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Sparse-A decode graph key is registered");
        let (acquisitions_before, graph_captures_before, graph_launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_captures, slot.graph_launches)
        };

        let output = match cuda_router_topk_bf16(
            &hidden,
            &router_weight,
            &[0.9, 0.0, 0.2],
            1,
            GLM52_HIDDEN_SIZE,
            3,
            2,
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let slot = registry.slots[graph_index]
            .lock()
            .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;

        assert_eq!(output.indices, vec![0, 2]);
        assert_eq!(output.backend, CUDA_REFERENCE_ROUTER_TOPK_BF16_BACKEND);
        assert_eq!(output.scores.len(), 2);
        assert_eq!(output.weights.len(), 2);
        assert!((output.weights.iter().sum::<f32>() - GLM52_ROUTED_SCALING_FACTOR).abs() < 1.0e-6);
        assert!(slot.acquisitions > acquisitions_before);
        assert!(slot.graph_captures >= graph_captures_before);
        assert!(slot.graph_launches > graph_launches_before);
        assert!(slot.has_captured_graph(
            CoordinatorCudaGraphProgram::SparseARouterTopKBf16,
            CoordinatorCudaGraphSignature::router_topk_bf16(1, GLM52_HIDDEN_SIZE, 3, 2)
        ));
        Ok(())
    }

    #[test]
    fn router_topk_bf16_coord_sparse_a_graph_replays_same_bucket_when_rows_change() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let rows_first = 2_usize;
        let rows_second = 4_usize;
        let experts = 6_usize;
        let top_k = 2_usize;
        let correction_bias = [0.0_f32, 0.1, 0.2, 0.3, 0.4, -0.2];
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseA,
            LayerWaveMode::Prefill,
            rows_first,
        )?;
        let signature = CoordinatorCudaGraphSignature::router_topk_bf16(
            graph_key.row_bucket.row_capacity,
            GLM52_HIDDEN_SIZE,
            experts,
            top_k,
        );
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Sparse-A prefill graph key is registered");

        let mut router_weight_values = vec![0.0_f32; experts * GLM52_HIDDEN_SIZE];
        router_weight_values[0] = 3.0;
        router_weight_values[GLM52_HIDDEN_SIZE + 1] = 3.0;
        router_weight_values[2 * GLM52_HIDDEN_SIZE + 2] = 3.0;
        router_weight_values[3 * GLM52_HIDDEN_SIZE + 3] = 3.0;
        router_weight_values[4 * GLM52_HIDDEN_SIZE] = 1.0;
        router_weight_values[4 * GLM52_HIDDEN_SIZE + 1] = 1.0;
        router_weight_values[4 * GLM52_HIDDEN_SIZE + 2] = 1.0;
        router_weight_values[4 * GLM52_HIDDEN_SIZE + 3] = 1.0;
        router_weight_values[5 * GLM52_HIDDEN_SIZE] = -1.0;
        router_weight_values[5 * GLM52_HIDDEN_SIZE + 1] = -1.0;
        let router_weight = bf16_bytes(&router_weight_values);

        let assert_router_matches =
            |actual: &super::RouterTopKOutput, expected: &super::RouterTopKOutput| {
                assert_eq!(actual.indices, expected.indices);
                assert_eq!(actual.scores.len(), expected.scores.len());
                assert_eq!(actual.weights.len(), expected.weights.len());
                for (actual, expected) in actual.scores.iter().zip(expected.scores.iter()) {
                    assert!((actual - expected).abs() < 1.0e-5);
                }
                for (actual, expected) in actual.weights.iter().zip(expected.weights.iter()) {
                    assert!((actual - expected).abs() < 1.0e-5);
                }
            };

        let (captures_before, launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.graph_captures, slot.graph_launches)
        };

        let mut hidden_first_values = vec![0.0_f32; rows_first * GLM52_HIDDEN_SIZE];
        hidden_first_values[0] = 1.0;
        hidden_first_values[GLM52_HIDDEN_SIZE + 1] = 1.0;
        let hidden_first = bf16_bytes(&hidden_first_values);
        let expected_first = cpu_router_topk_bf16(
            &hidden_first,
            &router_weight,
            &correction_bias,
            rows_first,
            GLM52_HIDDEN_SIZE,
            experts,
            top_k,
        );
        let output_first = match cuda_router_topk_bf16(
            &hidden_first,
            &router_weight,
            &correction_bias,
            rows_first,
            GLM52_HIDDEN_SIZE,
            experts,
            top_k,
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        assert_router_matches(&output_first, &expected_first);

        let (captures_after_first, launches_after_first, has_graph) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (
                slot.graph_captures,
                slot.graph_launches,
                slot.has_captured_graph(
                    CoordinatorCudaGraphProgram::SparseARouterTopKBf16,
                    signature,
                ),
            )
        };
        assert!(captures_after_first > captures_before);
        assert!(launches_after_first > launches_before);
        assert!(has_graph);

        let mut hidden_second_values = vec![0.0_f32; rows_second * GLM52_HIDDEN_SIZE];
        hidden_second_values[2] = 1.0;
        hidden_second_values[GLM52_HIDDEN_SIZE + 3] = 1.0;
        hidden_second_values[2 * GLM52_HIDDEN_SIZE] = 0.5;
        hidden_second_values[3 * GLM52_HIDDEN_SIZE + 1] = 0.5;
        let hidden_second = bf16_bytes(&hidden_second_values);
        let expected_second = cpu_router_topk_bf16(
            &hidden_second,
            &router_weight,
            &correction_bias,
            rows_second,
            GLM52_HIDDEN_SIZE,
            experts,
            top_k,
        );
        let output_second = match cuda_router_topk_bf16(
            &hidden_second,
            &router_weight,
            &correction_bias,
            rows_second,
            GLM52_HIDDEN_SIZE,
            experts,
            top_k,
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        assert_router_matches(&output_second, &expected_second);

        let (captures_after_second, launches_after_second) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.graph_captures, slot.graph_launches)
        };
        assert_eq!(captures_after_second, captures_after_first);
        assert!(launches_after_second > launches_after_first);
        Ok(())
    }

    #[test]
    fn router_topk_bf16_resident_weight_uses_cpu_reference_when_cuda_reference_is_not_enabled() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let hidden = bf16_bytes(&[1.0, 0.0]);
        let router_weight = bf16_bytes(&[
            0.0_f32, 0.0, //
            1.0, 0.0, //
            0.5, 0.0, //
        ]);
        let output = router_topk_bf16_resident_weight(
            "model.layers.3.mlp.gate.weight",
            &hidden,
            &router_weight,
            &[0.9, 0.0, 0.2],
            1,
            2,
            3,
            2,
        )
        .unwrap();

        assert_eq!(output.indices, vec![0, 2]);
        assert_eq!(output.scores.len(), 2);
        assert_eq!(output.weights.len(), 2);
        assert!((output.weights.iter().sum::<f32>() - GLM52_ROUTED_SCALING_FACTOR).abs() < 1.0e-6);
        assert_eq!(output.backend, CPU_REFERENCE_ROUTER_TOPK_BF16_BACKEND);
    }

    #[test]
    fn router_topk_bf16_resident_weight_full_width_uses_coord_sparse_a_graph_slot() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _python_capture_guard = coordinator_python_capture_test_override(true);
        let fixture = full_width_router_graph_update_fixture();
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseA,
            LayerWaveMode::Decode,
            fixture.rows,
        )?;
        let signature = CoordinatorCudaGraphSignature::router_topk_bf16(
            graph_key.row_bucket.row_capacity,
            GLM52_HIDDEN_SIZE,
            fixture.experts,
            fixture.top_k,
        );
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        let router_weight_name_a = format!(
            "model.layers.3.mlp.gate.weight.resident.graph-slot.test-a.{}.{}",
            std::process::id(),
            line!()
        );
        let router_weight_name_b = format!(
            "model.layers.3.mlp.gate.weight.resident.graph-slot.test-b.{}.{}",
            std::process::id(),
            line!()
        );

        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Sparse-A decode graph key is registered");
        let (acquisitions_before, graph_launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_launches)
        };

        let output_a = match cuda_router_topk_bf16_resident_weight(
            &router_weight_name_a,
            &fixture.hidden,
            &fixture.weight_a,
            &fixture.bias_a,
            fixture.rows,
            GLM52_HIDDEN_SIZE,
            fixture.experts,
            fixture.top_k,
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let output_b = match cuda_router_topk_bf16_resident_weight(
            &router_weight_name_b,
            &fixture.hidden,
            &fixture.weight_b,
            &fixture.bias_b,
            fixture.rows,
            GLM52_HIDDEN_SIZE,
            fixture.experts,
            fixture.top_k,
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let (acquisitions_after, graph_launches_after, has_graph) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (
                slot.acquisitions,
                slot.graph_launches,
                slot.captured_graphs.iter().any(|entry| {
                    entry.program == CoordinatorCudaGraphProgram::SparseATritonRouterTopKBf16
                }),
            )
        };

        assert_eq!(output_a.indices, fixture.expected_indices_a);
        assert_eq!(output_b.indices, fixture.expected_indices_b);
        assert_eq!(
            output_a.backend,
            TRITON_ROUTER_TOPK_BF16_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(
            output_b.backend,
            TRITON_ROUTER_TOPK_BF16_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(output_a.scores.len(), fixture.rows * fixture.top_k);
        assert_eq!(output_b.scores.len(), fixture.rows * fixture.top_k);
        for weights in output_a.weights.chunks_exact(fixture.top_k) {
            assert!((weights.iter().sum::<f32>() - GLM52_ROUTED_SCALING_FACTOR).abs() < 1.0e-6);
        }
        for weights in output_b.weights.chunks_exact(fixture.top_k) {
            assert!((weights.iter().sum::<f32>() - GLM52_ROUTED_SCALING_FACTOR).abs() < 1.0e-6);
        }
        assert!(acquisitions_after > acquisitions_before);
        assert!(graph_launches_after >= graph_launches_before + 2);
        assert!(has_graph);
        Ok(())
    }

    #[test]
    fn router_topk_bf16_device_input_variants_match_host_input_when_cuda_enabled() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);

        let result = (|| -> Result<()> {
            let rows = 2_usize;
            let hidden_dim = 2_usize;
            let experts = 4_usize;
            let top_k = 2_usize;
            let hidden = bf16_bytes(&[1.0_f32, 0.0, 0.0, 1.0]);
            let router_weight = bf16_bytes(&[
                0.0_f32, 0.0, //
                1.0, 0.0, //
                0.5, 0.0, //
                0.0, 1.0, //
            ]);
            let correction_bias = [0.9_f32, 0.0, 0.2, 0.1];
            let hidden_device = match device_bf16_output_from_bf16_bytes(
                &hidden,
                rows,
                hidden_dim,
                "test router device-input hidden",
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };

            let resident_name = format!(
                "test.router.device-input.resident.weight.{}.{}",
                std::process::id(),
                line!()
            );
            let resident_host = match router_topk_bf16_resident_weight(
                &resident_name,
                &hidden,
                &router_weight,
                &correction_bias,
                rows,
                hidden_dim,
                experts,
                top_k,
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            let resident_device = match router_topk_bf16_resident_weight_device_input(
                &resident_name,
                &hidden_device,
                &router_weight,
                &correction_bias,
                experts,
                top_k,
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };

            let preloaded_weight_name = format!(
                "test.router.device-input.preloaded.weight.{}.{}",
                std::process::id(),
                line!()
            );
            match preload_resident_weight_from_host_staging(
                &preloaded_weight_name,
                router_weight.len(),
                "test router device-input preloaded weight",
                |staging| {
                    staging.copy_from_slice(&router_weight);
                    Ok(())
                },
            ) {
                Ok(()) => {}
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            }
            let preloaded_host = match router_topk_bf16_preloaded_resident_weight(
                &preloaded_weight_name,
                &hidden,
                &correction_bias,
                rows,
                hidden_dim,
                experts,
                top_k,
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            let preloaded_device = match router_topk_bf16_preloaded_resident_weight_device_input(
                &preloaded_weight_name,
                &hidden_device,
                &correction_bias,
                experts,
                top_k,
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };

            let preloaded_bias_weight_name = format!(
                "test.router.device-input.preloaded-bias.weight.{}.{}",
                std::process::id(),
                line!()
            );
            let preloaded_bias_name = format!(
                "test.router.device-input.preloaded-bias.bias.{}.{}",
                std::process::id(),
                line!()
            );
            match preload_resident_weight_from_host_staging(
                &preloaded_bias_weight_name,
                router_weight.len(),
                "test router device-input preloaded-bias weight",
                |staging| {
                    staging.copy_from_slice(&router_weight);
                    Ok(())
                },
            ) {
                Ok(()) => {}
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            }
            let correction_bias_bytes = f32_bytes(&correction_bias).to_vec();
            match preload_resident_weight_from_host_staging(
                &preloaded_bias_name,
                correction_bias_bytes.len(),
                "test router device-input preloaded correction bias",
                |staging| {
                    staging.copy_from_slice(&correction_bias_bytes);
                    Ok(())
                },
            ) {
                Ok(()) => {}
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            }
            let preloaded_bias_host = match router_topk_bf16_preloaded_resident_weight_bias(
                &preloaded_bias_weight_name,
                &preloaded_bias_name,
                &hidden,
                &correction_bias,
                rows,
                hidden_dim,
                experts,
                top_k,
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            let preloaded_bias_device =
                match router_topk_bf16_preloaded_resident_weight_bias_device_input(
                    &preloaded_bias_weight_name,
                    &preloaded_bias_name,
                    &hidden_device,
                    experts,
                    top_k,
                ) {
                    Ok(output) => output,
                    Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                    Err(error) => return Err(error),
                };

            assert_eq!(resident_host.indices, resident_device.indices);
            assert_eq!(resident_host.scores, resident_device.scores);
            assert_eq!(resident_host.weights, resident_device.weights);
            assert_eq!(
                resident_device.backend,
                CUDA_REFERENCE_ROUTER_TOPK_BF16_RESIDENT_WEIGHT_DEVICE_INPUT_BACKEND
            );

            assert_eq!(preloaded_host.indices, preloaded_device.indices);
            assert_eq!(preloaded_host.scores, preloaded_device.scores);
            assert_eq!(preloaded_host.weights, preloaded_device.weights);
            assert_eq!(
                preloaded_device.backend,
                CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_DEVICE_INPUT_BACKEND
            );

            assert_eq!(preloaded_bias_host.indices, preloaded_bias_device.indices);
            assert_eq!(preloaded_bias_host.scores, preloaded_bias_device.scores);
            assert_eq!(preloaded_bias_host.weights, preloaded_bias_device.weights);
            assert_eq!(
                preloaded_bias_device.backend,
                CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_DEVICE_INPUT_BACKEND
            );
            Ok(())
        })();

        result
    }

    #[test]
    fn router_topk_bf16_preloaded_resident_weight_requires_cuda_reference() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let hidden = bf16_bytes(&[1.0, 0.0]);
        let err = router_topk_bf16_preloaded_resident_weight(
            "model.layers.3.mlp.gate.weight",
            &hidden,
            &[0.9, 0.0, 0.2],
            1,
            2,
            3,
            2,
        )
        .unwrap_err();

        assert!(err
            .to_string()
            .contains("preloaded resident BF16 router top-k requires CUDA reference kernels"));
    }

    #[test]
    fn router_topk_bf16_preloaded_resident_weight_full_width_uses_coord_sparse_a_graph_slot(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _python_capture_guard = coordinator_python_capture_test_override(true);
        let fixture = full_width_router_graph_update_fixture();
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseA,
            LayerWaveMode::Decode,
            fixture.rows,
        )?;
        let signature = CoordinatorCudaGraphSignature::router_topk_bf16(
            graph_key.row_bucket.row_capacity,
            GLM52_HIDDEN_SIZE,
            fixture.experts,
            fixture.top_k,
        );
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        let router_weight_name_a = format!(
            "test.router.preloaded-host-bias.graph-slot.weight-a.{}.{}",
            std::process::id(),
            line!()
        );
        let router_weight_name_b = format!(
            "test.router.preloaded-host-bias.graph-slot.weight-b.{}.{}",
            std::process::id(),
            line!()
        );

        match preload_resident_weight_from_host_staging(
            &router_weight_name_a,
            fixture.weight_a.len(),
            "test preloaded host-bias router graph-slot weight a",
            |staging| {
                staging.copy_from_slice(&fixture.weight_a);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        }
        match preload_resident_weight_from_host_staging(
            &router_weight_name_b,
            fixture.weight_b.len(),
            "test preloaded host-bias router graph-slot weight b",
            |staging| {
                staging.copy_from_slice(&fixture.weight_b);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        }

        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Sparse-A decode graph key is registered");
        let (acquisitions_before, graph_launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_launches)
        };

        let output_a = match cuda_router_topk_bf16_preloaded_resident_weight(
            &router_weight_name_a,
            &fixture.hidden,
            &fixture.bias_a,
            fixture.rows,
            GLM52_HIDDEN_SIZE,
            fixture.experts,
            fixture.top_k,
            fixture.weight_a.len(),
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let output_b = match cuda_router_topk_bf16_preloaded_resident_weight(
            &router_weight_name_b,
            &fixture.hidden,
            &fixture.bias_b,
            fixture.rows,
            GLM52_HIDDEN_SIZE,
            fixture.experts,
            fixture.top_k,
            fixture.weight_b.len(),
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let (acquisitions_after, graph_launches_after, has_graph) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (
                slot.acquisitions,
                slot.graph_launches,
                slot.captured_graphs.iter().any(|entry| {
                    entry.program == CoordinatorCudaGraphProgram::SparseATritonRouterTopKBf16
                }),
            )
        };

        assert_eq!(output_a.indices, fixture.expected_indices_a);
        assert_eq!(output_b.indices, fixture.expected_indices_b);
        assert_eq!(
            output_a.backend,
            TRITON_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(
            output_b.backend,
            TRITON_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(output_a.scores.len(), fixture.rows * fixture.top_k);
        assert_eq!(output_b.scores.len(), fixture.rows * fixture.top_k);
        for weights in output_a.weights.chunks_exact(fixture.top_k) {
            assert!((weights.iter().sum::<f32>() - GLM52_ROUTED_SCALING_FACTOR).abs() < 1.0e-6);
        }
        for weights in output_b.weights.chunks_exact(fixture.top_k) {
            assert!((weights.iter().sum::<f32>() - GLM52_ROUTED_SCALING_FACTOR).abs() < 1.0e-6);
        }
        assert!(acquisitions_after > acquisitions_before);
        assert!(graph_launches_after >= graph_launches_before + 2);
        assert!(has_graph);
        Ok(())
    }

    #[test]
    fn router_topk_bf16_preloaded_resident_weight_bias_requires_cuda_reference() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let hidden = bf16_bytes(&[1.0, 0.0]);
        let err = router_topk_bf16_preloaded_resident_weight_bias(
            "model.layers.3.mlp.gate.weight",
            "model.layers.3.mlp.gate.e_score_correction_bias",
            &hidden,
            &[0.9, 0.0, 0.2],
            1,
            2,
            3,
            2,
        )
        .unwrap_err();

        assert!(err.to_string().contains(
            "preloaded resident BF16 router top-k weight+bias requires CUDA reference kernels"
        ));
    }

    #[test]
    fn router_topk_bf16_preloaded_resident_weight_bias_full_width_uses_coord_sparse_a_graph_slot(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _python_capture_guard = coordinator_python_capture_test_override(true);
        let rows = 2_usize;
        let experts = 5_usize;
        let top_k = 3_usize;
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseA,
            LayerWaveMode::Decode,
            rows,
        )?;
        let signature = CoordinatorCudaGraphSignature::router_topk_bf16(
            graph_key.row_bucket.row_capacity,
            GLM52_HIDDEN_SIZE,
            experts,
            top_k,
        );
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        let mut hidden_values = vec![0.0_f32; rows * GLM52_HIDDEN_SIZE];
        hidden_values[0] = 1.0;
        hidden_values[GLM52_HIDDEN_SIZE + 1] = 1.0;
        let hidden = bf16_bytes(&hidden_values);
        let mut router_weight_values_a = vec![0.0_f32; experts * GLM52_HIDDEN_SIZE];
        router_weight_values_a[0] = 3.0;
        router_weight_values_a[GLM52_HIDDEN_SIZE + 1] = 3.0;
        router_weight_values_a[2 * GLM52_HIDDEN_SIZE] = 1.0;
        router_weight_values_a[3 * GLM52_HIDDEN_SIZE + 1] = 1.0;
        let router_weight_a = bf16_bytes(&router_weight_values_a);
        let correction_bias_a = [-0.1_f32, -0.1, 0.0, 0.0, 0.1];
        let mut router_weight_values_b = vec![0.0_f32; experts * GLM52_HIDDEN_SIZE];
        router_weight_values_b[2 * GLM52_HIDDEN_SIZE] = 3.0;
        router_weight_values_b[3 * GLM52_HIDDEN_SIZE + 1] = 3.0;
        router_weight_values_b[4 * GLM52_HIDDEN_SIZE] = 1.0;
        router_weight_values_b[4 * GLM52_HIDDEN_SIZE + 1] = 1.0;
        let router_weight_b = bf16_bytes(&router_weight_values_b);
        let correction_bias_b = [0.1_f32, -0.3, 0.0, -0.1, 0.1];
        let router_weight_name_a = format!(
            "test.router.preloaded.graph-slot.weight-a.{}.{}",
            std::process::id(),
            line!()
        );
        let correction_bias_name_a = format!(
            "test.router.preloaded.graph-slot.bias-a.{}.{}",
            std::process::id(),
            line!()
        );
        let router_weight_name_b = format!(
            "test.router.preloaded.graph-slot.weight-b.{}.{}",
            std::process::id(),
            line!()
        );
        let correction_bias_name_b = format!(
            "test.router.preloaded.graph-slot.bias-b.{}.{}",
            std::process::id(),
            line!()
        );

        match preload_resident_weight_from_host_staging(
            &router_weight_name_a,
            router_weight_a.len(),
            "test preloaded router graph-slot weight a",
            |staging| {
                staging.copy_from_slice(&router_weight_a);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        }
        let bias_bytes_a = f32_bytes(&correction_bias_a).to_vec();
        match preload_resident_weight_from_host_staging(
            &correction_bias_name_a,
            bias_bytes_a.len(),
            "test preloaded router graph-slot correction bias a",
            |staging| {
                staging.copy_from_slice(&bias_bytes_a);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        }
        match preload_resident_weight_from_host_staging(
            &router_weight_name_b,
            router_weight_b.len(),
            "test preloaded router graph-slot weight b",
            |staging| {
                staging.copy_from_slice(&router_weight_b);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        }
        let bias_bytes_b = f32_bytes(&correction_bias_b).to_vec();
        match preload_resident_weight_from_host_staging(
            &correction_bias_name_b,
            bias_bytes_b.len(),
            "test preloaded router graph-slot correction bias b",
            |staging| {
                staging.copy_from_slice(&bias_bytes_b);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        }

        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Sparse-A decode graph key is registered");
        let (acquisitions_before, graph_launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_launches)
        };

        let output_a = match cuda_router_topk_bf16_preloaded_resident_weight_bias(
            &router_weight_name_a,
            &correction_bias_name_a,
            &hidden,
            rows,
            GLM52_HIDDEN_SIZE,
            experts,
            top_k,
            router_weight_a.len(),
            bias_bytes_a.len(),
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let output_b = match cuda_router_topk_bf16_preloaded_resident_weight_bias(
            &router_weight_name_b,
            &correction_bias_name_b,
            &hidden,
            rows,
            GLM52_HIDDEN_SIZE,
            experts,
            top_k,
            router_weight_b.len(),
            bias_bytes_b.len(),
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let (acquisitions_after, graph_launches_after, has_graph) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (
                slot.acquisitions,
                slot.graph_launches,
                slot.captured_graphs.iter().any(|entry| {
                    entry.program == CoordinatorCudaGraphProgram::SparseATritonRouterTopKBf16
                }),
            )
        };

        assert_eq!(output_a.indices, vec![0, 2, 4, 1, 3, 4]);
        assert_eq!(output_b.indices, vec![2, 4, 0, 3, 4, 0]);
        assert_eq!(
            output_a.backend,
            TRITON_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_BACKEND
        );
        assert_eq!(
            output_b.backend,
            TRITON_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_BACKEND
        );
        assert_eq!(output_a.scores.len(), rows * top_k);
        assert_eq!(output_b.scores.len(), rows * top_k);
        assert_eq!(output_a.weights.len(), rows * top_k);
        assert_eq!(output_b.weights.len(), rows * top_k);
        for weights in output_a.weights.chunks_exact(top_k) {
            assert!((weights.iter().sum::<f32>() - GLM52_ROUTED_SCALING_FACTOR).abs() < 1.0e-6);
        }
        for weights in output_b.weights.chunks_exact(top_k) {
            assert!((weights.iter().sum::<f32>() - GLM52_ROUTED_SCALING_FACTOR).abs() < 1.0e-6);
        }
        assert!(acquisitions_after > acquisitions_before);
        assert!(graph_launches_after >= graph_launches_before + 2);
        assert!(has_graph);
        Ok(())
    }

    #[test]
    fn router_topk_bf16_preloaded_resident_weight_bias_full_width_uses_triton_graph_when_python_enabled(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _python_capture_guard = coordinator_python_capture_test_override(true);
        let fixture = full_width_router_graph_update_fixture();
        let expected = cpu_router_topk_bf16(
            &fixture.hidden,
            &fixture.weight_a,
            &fixture.bias_a,
            fixture.rows,
            GLM52_HIDDEN_SIZE,
            fixture.experts,
            fixture.top_k,
        );
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseA,
            LayerWaveMode::Decode,
            fixture.rows,
        )?;
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        let router_weight_name = format!(
            "test.router.triton.preloaded.graph-slot.weight.{}.{}",
            std::process::id(),
            line!()
        );
        let correction_bias_name = format!(
            "test.router.triton.preloaded.graph-slot.bias.{}.{}",
            std::process::id(),
            line!()
        );
        match preload_resident_weight_from_host_staging(
            &router_weight_name,
            fixture.weight_a.len(),
            "test Triton preloaded router graph-slot weight",
            |staging| {
                staging.copy_from_slice(&fixture.weight_a);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        }
        let bias_bytes = f32_bytes(&fixture.bias_a).to_vec();
        match preload_resident_weight_from_host_staging(
            &correction_bias_name,
            bias_bytes.len(),
            "test Triton preloaded router graph-slot correction bias",
            |staging| {
                staging.copy_from_slice(&bias_bytes);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        }

        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Sparse-A decode graph key is registered");
        let (acquisitions_before, graph_launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_launches)
        };

        let output = match cuda_router_topk_bf16_preloaded_resident_weight_bias(
            &router_weight_name,
            &correction_bias_name,
            &fixture.hidden,
            fixture.rows,
            GLM52_HIDDEN_SIZE,
            fixture.experts,
            fixture.top_k,
            fixture.weight_a.len(),
            bias_bytes.len(),
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let (acquisitions_after, graph_launches_after, has_triton_graph) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (
                slot.acquisitions,
                slot.graph_launches,
                slot.captured_graphs.iter().any(|entry| {
                    entry.program == CoordinatorCudaGraphProgram::SparseATritonRouterTopKBf16
                }),
            )
        };

        assert_eq!(output.indices, fixture.expected_indices_a);
        assert_eq!(
            output.backend,
            TRITON_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_BACKEND
        );
        assert_eq!(output.scores.len(), expected.scores.len());
        assert_eq!(output.weights.len(), expected.weights.len());
        for (actual, expected) in output.scores.iter().zip(expected.scores.iter()) {
            assert!((actual - expected).abs() < 2.0e-3);
        }
        for weights in output.weights.chunks_exact(fixture.top_k) {
            assert!((weights.iter().sum::<f32>() - GLM52_ROUTED_SCALING_FACTOR).abs() < 1.0e-5);
        }
        assert!(acquisitions_after > acquisitions_before);
        assert!(graph_launches_after > graph_launches_before);
        assert!(has_triton_graph);
        Ok(())
    }

    #[test]
    fn router_topk_bf16_preloaded_resident_weight_rejects_bias_mismatch() {
        let hidden = bf16_bytes(&[1.0, 0.0]);
        let err = router_topk_bf16_preloaded_resident_weight(
            "model.layers.3.mlp.gate.weight",
            &hidden,
            &[0.9, 0.0],
            1,
            2,
            3,
            2,
        )
        .unwrap_err();

        assert!(err.to_string().contains("correction bias length mismatch"));
    }

    #[test]
    fn router_topk_bf16_preloaded_resident_weight_bias_rejects_bias_mismatch() {
        let hidden = bf16_bytes(&[1.0, 0.0]);
        let err = router_topk_bf16_preloaded_resident_weight_bias(
            "model.layers.3.mlp.gate.weight",
            "model.layers.3.mlp.gate.e_score_correction_bias",
            &hidden,
            &[0.9, 0.0],
            1,
            2,
            3,
            2,
        )
        .unwrap_err();

        assert!(err.to_string().contains("correction bias length mismatch"));
    }

    #[test]
    fn router_topk_bf16_resident_weight_rejects_empty_weight_name() {
        let hidden = bf16_bytes(&[1.0, 0.0]);
        let router_weight = bf16_bytes(&[
            0.0_f32, 0.0, //
            1.0, 0.0, //
        ]);
        let err =
            router_topk_bf16_resident_weight("", &hidden, &router_weight, &[0.0, 0.0], 1, 2, 2, 1)
                .unwrap_err();

        assert!(err.to_string().contains("weight name must not be empty"));
    }

    #[test]
    fn router_topk_bf16_rejects_shape_mismatch_before_backend_selection() {
        let hidden = bf16_bytes(&[1.0, 0.0]);
        let router_weight = bf16_bytes(&[0.0, 0.0, 1.0]);
        let err = router_topk_bf16(&hidden, &router_weight, &[0.0, 0.0], 1, 2, 2, 1).unwrap_err();

        assert!(err
            .to_string()
            .contains("BF16 router top-k weight byte length mismatch"));
    }

    #[test]
    fn cuda_native_library_handle_is_cached_when_available() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let first = cuda_native_library()?;
        let second = cuda_native_library()?;

        assert!(std::ptr::eq(first, second));
        Ok(())
    }

    #[test]
    fn coordinator_cuda_workspace_reuses_scratch_slot_when_available() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let library = cuda_native_library()?;
        let mut workspace = lock_coordinator_cuda_workspace()?;
        let first = match workspace.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            4,
            "test coordinator scratch",
        ) {
            Ok(buffer) => buffer,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let second = workspace.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            4,
            "test coordinator scratch",
        )?;

        assert_eq!(first.ptr, second.ptr);
        assert!(second.bytes >= 4);
        let bias_first = workspace.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            4,
            "test coordinator scratch bias",
        )?;
        let bias_second = workspace.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            4,
            "test coordinator scratch bias",
        )?;

        assert_eq!(bias_first.ptr, bias_second.ptr);
        assert!(bias_second.bytes >= 4);
        let router_weight_first = workspace.buffer(
            library,
            CoordinatorCudaScratchSlot::F,
            4,
            "test coordinator scratch router weights",
        )?;
        let router_weight_second = workspace.buffer(
            library,
            CoordinatorCudaScratchSlot::F,
            4,
            "test coordinator scratch router weights",
        )?;

        assert_eq!(router_weight_first.ptr, router_weight_second.ptr);
        assert!(router_weight_second.bytes >= 4);
        let host_first = workspace.host_buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            4,
            "test coordinator pinned staging",
        )?;
        let host_second = workspace.host_buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            4,
            "test coordinator pinned staging",
        )?;

        assert_eq!(host_first.ptr, host_second.ptr);
        assert!(host_second.bytes >= 4);
        Ok(())
    }

    #[test]
    fn mla_decode_query_chain_orders_async_position_updates() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);
        let library = cuda_native_library()?;
        let upload = |bytes: &[u8], label: &'static str| -> Result<OwnedCoordinatorDeviceBuffer> {
            let buffer = OwnedCoordinatorDeviceBuffer::new(library, bytes.len(), label)?;
            library.copy_h2d(buffer.buffer, bytes)?;
            Ok(buffer)
        };
        let normalized_hidden = upload(&bf16_bytes(&[1.0, 2.0]), "query chain hidden")?;
        let q_a_weight = upload(&bf16_bytes(&[1.0, 0.0, 0.0, 1.0]), "query chain q_a weight")?;
        let q_a_norm_weight = upload(&bf16_bytes(&[1.0, 1.0]), "query chain q_a norm")?;
        let q_b_weight = upload(
            &bf16_bytes(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0]),
            "query chain q_b weight",
        )?;
        let dsa_wq_b_weight = upload(
            &bf16_bytes(&[1.0, 0.0, 0.0, 1.0]),
            "query chain DSA wq_b weight",
        )?;
        let dsa_weights_proj_weight = upload(
            &bf16_bytes(&[1.0, 1.0]),
            "query chain DSA weights projection weight",
        )?;
        let q_nope = OwnedCoordinatorDeviceBuffer::new(library, 4, "query chain q_nope")?;
        let q_rope_unrotated =
            OwnedCoordinatorDeviceBuffer::new(library, 4, "query chain q_rope unrotated")?;
        let q_rope_rotated =
            OwnedCoordinatorDeviceBuffer::new(library, 4, "query chain q_rope rotated")?;

        let run = |position: u32| -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
            let (projected, dsa_query, dsa_weights) =
                mla_decode_query_dsa_projection_bf16_device_outputs(
                    3,
                    normalized_hidden.buffer,
                    "query-chain-q-a",
                    Some(q_a_weight.buffer),
                    q_a_norm_weight.buffer,
                    "query-chain-q-b",
                    Some(q_b_weight.buffer),
                    dsa_wq_b_weight.buffer,
                    dsa_weights_proj_weight.buffer,
                    1,
                    2,
                    2,
                    4,
                    2,
                    1,
                    1.0e-5,
                )?;
            mla_query_split_rope_bf16_device_buffers_for_layer(
                3,
                projected.buffer(),
                q_nope.buffer,
                q_rope_unrotated.buffer,
                position,
                q_rope_rotated.buffer,
                1,
                1,
                2,
                2,
                10_000.0,
            )?;
            let key = coord_attention_graph_key_for_layer_rows(3, 1)?;
            with_coordinator_cuda_graph_slot(&key, |_library, slot| slot.stream_synchronize())?;
            let read = |buffer: GlmrtDeviceBuffer| -> Result<Vec<f32>> {
                let mut bytes = vec![0_u8; 4];
                library.copy_d2h(&mut bytes, buffer)?;
                Ok(bf16_values_to_f32(&bytes))
            };
            Ok((
                read(q_nope.buffer)?,
                read(q_rope_unrotated.buffer)?,
                read(q_rope_rotated.buffer)?,
                bf16_values_to_f32(&dsa_query.copy_to_host_bytes()?),
                bf16_values_to_f32(&dsa_weights.copy_to_host_bytes()?),
            ))
        };

        let (nope_zero, rope_unrotated_zero, rope_zero, dsa_query_zero, dsa_weights_zero) = run(0)?;
        let (nope_three, rope_unrotated_three, rope_three, dsa_query_three, dsa_weights_three) =
            run(3)?;
        assert_eq!(nope_zero, nope_three);
        assert_eq!(rope_unrotated_zero, rope_unrotated_three);
        assert_eq!(rope_unrotated_zero, rope_zero);
        assert_ne!(rope_zero, rope_three);
        assert!(rope_three.iter().all(|value| value.is_finite()));
        assert_eq!(dsa_query_zero, dsa_query_three);
        assert_eq!(dsa_weights_zero, dsa_weights_three);
        assert!(dsa_query_zero.iter().all(|value| value.is_finite()));
        assert!(dsa_weights_zero.iter().all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    fn mla_decode_kv_commit_orders_all_cache_format_writes_and_positions() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);
        let library = cuda_native_library()?;
        let upload = |bytes: &[u8], label: &'static str| -> Result<OwnedCoordinatorDeviceBuffer> {
            let buffer = OwnedCoordinatorDeviceBuffer::new(library, bytes.len(), label)?;
            library.copy_h2d(buffer.buffer, bytes)?;
            Ok(buffer)
        };
        let hidden_dim = GLM52_HIDDEN_SIZE;
        let hidden_values = (0..hidden_dim)
            .map(|index| ((index % 17) as f32 - 8.0) * 0.125)
            .collect::<Vec<_>>();
        let hidden = upload(&bf16_bytes(&hidden_values), "decode KV commit test hidden")?;
        let hidden_second = upload(
            &bf16_bytes(&hidden_values),
            "decode KV commit test second hidden",
        )?;
        let input_norm_weight = upload(
            &bf16_bytes(&vec![1.0; hidden_dim]),
            "decode KV commit test input norm",
        )?;
        let kv_width = GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM;
        let mut kv_weight = vec![0.0_f32; kv_width * hidden_dim];
        for output in 0..kv_width {
            kv_weight[output * hidden_dim + output % hidden_dim] = 1.0;
        }
        let kv_a_weight = upload(&bf16_bytes(&kv_weight), "decode KV commit test kv_a weight")?;
        let kv_norm_weight = upload(
            &bf16_bytes(&vec![1.0; GLM52_MLA_KV_LORA_RANK]),
            "decode KV commit test kv norm",
        )?;
        let mut dsa_wk = vec![0.0_f32; GLM52_DSA_INDEX_HEAD_DIM * hidden_dim];
        for output in 0..GLM52_DSA_INDEX_HEAD_DIM {
            dsa_wk[output * hidden_dim + output] = 1.0;
        }
        let dsa_wk = upload(&bf16_bytes(&dsa_wk), "decode KV commit test DSA wk")?;
        let dsa_norm_weight = upload(
            &bf16_bytes(&vec![1.0; GLM52_DSA_INDEX_HEAD_DIM]),
            "decode KV commit test DSA norm weight",
        )?;
        let dsa_norm_bias = upload(
            &bf16_bytes(&vec![0.0; GLM52_DSA_INDEX_HEAD_DIM]),
            "decode KV commit test DSA norm bias",
        )?;
        let dsa_weights = MlaDecodeKvDsaProjectionWeights {
            wk: dsa_wk.buffer,
            norm_weight: dsa_norm_weight.buffer,
            norm_bias: dsa_norm_bias.buffer,
        };
        for (cache_dtype, packed_main_bytes) in [
            (KvCacheDType::Bf16, kv_width * std::mem::size_of::<u16>()),
            (KvCacheDType::Fp8, GLM52_MLA_FP8_DS_BYTES_PER_TOKEN),
            (KvCacheDType::Nvfp4, GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN),
        ] {
            let dsa_row_bytes = GLM52_DSA_INDEX_HEAD_DIM * std::mem::size_of::<u16>();
            let dsa_base = 2 * packed_main_bytes;
            let cache_bytes = dsa_base + 2 * dsa_row_bytes;
            let cache = OwnedCoordinatorDeviceBuffer::new(
                library,
                cache_bytes,
                "decode KV commit test cache",
            )?;
            let direct_dsa_cache = OwnedCoordinatorDeviceBuffer::new(
                library,
                glmrt_ffi::GLMRT_CUDA_GLM_DSA_PACKED_PAGE_BYTES,
                "decode KV commit test direct DSA cache",
            )?;

            let run = |row: usize, position: u32| -> Result<DeviceBf16Output> {
                mla_decode_kv_commit_bf16_device_output(
                    0,
                    if row == 0 {
                        hidden.buffer
                    } else {
                        hidden_second.buffer
                    },
                    input_norm_weight.buffer,
                    kv_a_weight.buffer,
                    kv_norm_weight.buffer,
                    Some(dsa_weights),
                    device_buffer_byte_view(
                        cache.buffer,
                        row * packed_main_bytes,
                        packed_main_bytes,
                        "decode KV commit test main cache row",
                    )?,
                    Some(device_buffer_byte_view(
                        cache.buffer,
                        dsa_base + row * dsa_row_bytes,
                        dsa_row_bytes,
                        "decode KV commit test DSA cache row",
                    )?),
                    Some(direct_dsa_cache.buffer),
                    glmrt_ffi::GLMRT_CUDA_GLM_DSA_PAGE_SIZE,
                    None,
                    false,
                    cache_dtype,
                    position,
                    position,
                    hidden_dim,
                    1.0e-5,
                    10_000.0,
                )
            };
            let normalized_zero = run(0, 0)?;
            let normalized_three = run(1, 3)?;
            let key = coord_attention_graph_key_for_layer_rows(0, 1)?;
            with_coordinator_cuda_graph_slot(&key, |_library, slot| slot.stream_synchronize())?;

            assert_eq!(
                normalized_zero.copy_to_host_bytes()?,
                normalized_three.copy_to_host_bytes()?
            );
            let mut cache_host = vec![0_u8; cache_bytes];
            library.copy_d2h(&mut cache_host, cache.buffer)?;
            assert_ne!(
                &cache_host[..packed_main_bytes],
                &cache_host[packed_main_bytes..dsa_base]
            );
            // The compatibility DSA plane retains normalized, unrotated
            // BF16. The direct index cache has its own RoPE+FP8 pack path.
            assert_eq!(
                &cache_host[dsa_base..dsa_base + dsa_row_bytes],
                &cache_host[dsa_base + dsa_row_bytes..]
            );
            let mut direct_dsa_host = vec![0_u8; glmrt_ffi::GLMRT_CUDA_GLM_DSA_PACKED_PAGE_BYTES];
            library.copy_d2h(&mut direct_dsa_host, direct_dsa_cache.buffer)?;
            assert_ne!(
                &direct_dsa_host[..GLM52_DSA_INDEX_HEAD_DIM],
                &direct_dsa_host[3 * GLM52_DSA_INDEX_HEAD_DIM..4 * GLM52_DSA_INDEX_HEAD_DIM]
            );
        }
        Ok(())
    }

    #[test]
    fn coordinator_f32_reference_uploads_reuse_pinned_host_staging() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);

        let result = (|| -> Result<()> {
            let run_reference_wrappers = || -> Result<()> {
                let rope_output = rope_rows(&[1.0, 0.0, 0.0, 1.0], &[1], 1, 1, 4, 10_000.0)?;
                assert!(rope_output.values.iter().all(|value| value.is_finite()));

                let mla_output = mla_rope_attention_rows(
                    &[1.0, 0.0],
                    &[0.0, 1.0],
                    &[1.0, 0.0],
                    &[0.0, 1.0],
                    &[2.0, -1.0],
                    1,
                    1,
                    2,
                    2,
                    2,
                    1.0,
                )?;
                assert_eq!(mla_output.values, vec![2.0, -1.0]);

                let causal_output =
                    causal_attention_rows(&[1.0, 0.0], &[1.0, 0.0], &[2.0, -1.0], 1, 1, 2, 2, 1.0)?;
                assert_eq!(causal_output.values, vec![2.0, -1.0]);

                let argmax_output = logits_argmax(&[0.25, 3.0, -1.0], 1, 3)?;
                assert_eq!(argmax_output.indices, vec![1]);
                assert_eq!(argmax_output.scores, vec![3.0]);

                let sample_output =
                    logits_sample_topk_topp(&[3.0, 2.0, 1.0, 0.0], &[0.0], 1, 4, 1.0, 3, 0.8)?;
                assert_eq!(sample_output.indices, vec![0]);
                assert!(sample_output.scores.iter().all(|score| score.is_finite()));
                assert_eq!(
                    sample_output.backend,
                    "cuda-reference-logits-sample-topk-topp-f32"
                );

                let embedding_output =
                    embedding_lookup_rows(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2], 1, 3, 2)?;
                assert_eq!(embedding_output.values, vec![2.0, 3.0]);

                let router_output = router_topk(
                    &[1.0, 0.0, 0.0, 1.0],
                    &[3.0, 0.0, 0.0, 3.0, 1.0, 1.0],
                    &[0.0, 0.0, 0.0],
                    2,
                    2,
                    3,
                    2,
                )?;
                assert_eq!(router_output.indices.len(), 4);
                assert_eq!(router_output.weights.len(), 4);

                let linear_output = linear_rows(
                    &[1.0, -1.0],
                    &[2.0, 0.0, 0.0, 3.0],
                    Some(&[0.5, -0.5]),
                    1,
                    2,
                    2,
                )?;
                assert_eq!(linear_output.values, vec![2.5, -3.5]);

                let mlp_output = silu_gated_mlp_rows(
                    &[1.0, -0.5],
                    &[1.0, 0.0, 0.0, 1.0],
                    &[0.5, 0.0, 0.0, 0.25],
                    &[1.0, 0.0, 0.0, 1.0],
                    1,
                    2,
                    2,
                    2,
                )?;
                assert_eq!(mlp_output.values.len(), 2);
                assert!(mlp_output.values.iter().all(|value| value.is_finite()));

                let rmsnorm_output = rmsnorm_hidden(&[1.0, 2.0], &[1.0, 0.5], 1.0e-5)?;
                assert_eq!(rmsnorm_output.values.len(), 2);
                assert!(rmsnorm_output.values.iter().all(|value| value.is_finite()));

                let residual_output = residual_add_prefix(&[1.0, 2.0], &[0.5, -0.25])?;
                assert_eq!(residual_output.values, vec![1.5, 1.75]);
                Ok(())
            };

            let host_staging_states = || -> Result<Vec<(*mut c_void, usize)>> {
                let workspace = lock_coordinator_cuda_workspace()?;
                [
                    CoordinatorCudaScratchSlot::A,
                    CoordinatorCudaScratchSlot::B,
                    CoordinatorCudaScratchSlot::C,
                    CoordinatorCudaScratchSlot::D,
                ]
                .iter()
                .map(|slot| {
                    let staging = workspace.host_staging.get(slot.index()).with_context(|| {
                        format!("host staging slot {} was not initialized", slot.index())
                    })?;
                    if staging.buffer.ptr.is_null() {
                        anyhow::bail!("host staging slot {} pointer is null", slot.index());
                    }
                    if staging.capacity == 0 {
                        anyhow::bail!("host staging slot {} capacity is zero", slot.index());
                    }
                    Ok((staging.buffer.ptr, staging.capacity))
                })
                .collect()
            };

            match run_reference_wrappers() {
                Ok(()) => {}
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            }
            let first_states = host_staging_states()?;

            run_reference_wrappers()?;
            let second_states = host_staging_states()?;
            assert_eq!(second_states, first_states);
            Ok(())
        })();

        result
    }

    #[test]
    fn device_bf16_output_upload_reuses_pinned_host_staging() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);

        let result = (|| -> Result<()> {
            let first_bytes = bf16_bytes(&[1.0_f32, 2.0, 3.0, 4.0]);
            let first_output = match device_bf16_output_from_bf16_bytes(
                &first_bytes,
                1,
                4,
                "test owned BF16 upload pinned staging first",
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            assert_eq!(first_output.copy_to_host_bytes()?, first_bytes);
            let (first_staging_ptr, first_staging_capacity) = {
                let workspace = lock_coordinator_cuda_workspace()?;
                let staging = workspace
                    .host_staging
                    .get(CoordinatorCudaScratchSlot::F.index())
                    .context("owned BF16 upload pinned staging slot was not initialized")?;
                assert!(!staging.buffer.ptr.is_null());
                assert!(staging.capacity >= first_bytes.len());
                (staging.buffer.ptr, staging.capacity)
            };

            let second_bytes = bf16_bytes(&[5.0_f32, 6.0]);
            let second_output = device_bf16_output_from_bf16_bytes(
                &second_bytes,
                1,
                2,
                "test owned BF16 upload pinned staging second",
            )?;
            assert_eq!(second_output.copy_to_host_bytes()?, second_bytes);
            let (second_staging_ptr, second_staging_capacity) = {
                let workspace = lock_coordinator_cuda_workspace()?;
                let staging = workspace
                    .host_staging
                    .get(CoordinatorCudaScratchSlot::F.index())
                    .context("owned BF16 upload pinned staging slot disappeared")?;
                assert!(!staging.buffer.ptr.is_null());
                assert!(staging.capacity >= second_bytes.len());
                (staging.buffer.ptr, staging.capacity)
            };

            assert_eq!(second_staging_ptr, first_staging_ptr);
            assert_eq!(second_staging_capacity, first_staging_capacity);
            Ok(())
        })();

        result
    }

    #[test]
    fn coordinator_cuda_thread_workspace_borrow_is_per_thread() -> Result<()> {
        let mut workspace = lock_coordinator_cuda_workspace()?;
        let main_workspace = (&mut *workspace as *mut _) as usize;

        let thread = std::thread::spawn(|| -> Result<usize> {
            let mut workspace = lock_coordinator_cuda_workspace()?;
            Ok((&mut *workspace as *mut _) as usize)
        });
        let thread_workspace = thread
            .join()
            .map_err(|_| anyhow::anyhow!("coordinator CUDA thread workspace test panicked"))??;

        assert_ne!(main_workspace, thread_workspace);
        Ok(())
    }

    #[test]
    fn coordinator_cuda_graph_workspace_pool_has_one_stream_per_graph_plan() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let library = cuda_native_library()?;
        let mut pool = match CoordinatorCudaGraphWorkspacePool::glm52_bf16(library) {
            Ok(pool) => pool,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let expected_keys: Vec<_> = CoordinatorGraphInstancePlan::glm52_bf16_all()
            .into_iter()
            .map(|plan| plan.key)
            .collect();

        assert_eq!(pool.len(), COORDINATOR_GRAPH_INSTANCE_COUNT);
        assert_eq!(pool.keys(), expected_keys);

        let stream_ptrs = pool.stream_ptrs();
        assert!(stream_ptrs.iter().all(|stream| !stream.is_null()));
        let unique_streams = stream_ptrs
            .iter()
            .map(|stream| *stream as usize)
            .collect::<HashSet<_>>();
        assert_eq!(unique_streams.len(), COORDINATOR_GRAPH_INSTANCE_COUNT);

        let first_key = expected_keys[0].clone();
        let second_key = expected_keys[1].clone();
        let first_buffer = {
            let slot = pool.slot_for_key_mut(&first_key)?;
            let buffer = slot.buffer(
                library,
                CoordinatorCudaScratchSlot::A,
                4,
                "test coordinator graph slot scratch",
            )?;
            slot.stream_synchronize()?;
            buffer
        };
        let first_buffer_reused = {
            let slot = pool.slot_for_key_mut(&first_key)?;
            slot.buffer(
                library,
                CoordinatorCudaScratchSlot::A,
                4,
                "test coordinator graph slot scratch",
            )?
        };
        let second_buffer = {
            let slot = pool.slot_for_key_mut(&second_key)?;
            slot.buffer(
                library,
                CoordinatorCudaScratchSlot::A,
                4,
                "test coordinator graph slot scratch",
            )?
        };

        assert_eq!(first_buffer.ptr, first_buffer_reused.ptr);
        assert_ne!(first_buffer.ptr, second_buffer.ptr);
        assert_eq!(
            pool.slots
                .iter()
                .find(|slot| slot.plan.key == first_key)
                .map(|slot| slot.acquisitions),
            Some(2)
        );
        assert_eq!(
            pool.slots
                .iter()
                .find(|slot| slot.plan.key == second_key)
                .map(|slot| slot.acquisitions),
            Some(1)
        );
        Ok(())
    }

    #[test]
    fn coordinator_cuda_graph_workspace_registry_borrows_independent_thread_local_slots(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let library = cuda_native_library()?;
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        assert_eq!(registry.len(), COORDINATOR_GRAPH_INSTANCE_COUNT);
        let first_key = registry.keys()[0].clone();
        let second_key = registry.keys()[1].clone();
        let mut first_slot = registry.slot_guard_for_key(&first_key)?;
        let mut second_slot = registry.slot_guard_for_key(&second_key)?;

        assert_ne!(first_slot.stream_ptr(), second_slot.stream_ptr());
        let first_buffer = first_slot.buffer(
            library,
            CoordinatorCudaScratchSlot::B,
            4,
            "test coordinator graph registry scratch",
        )?;
        let second_buffer = second_slot.buffer(
            library,
            CoordinatorCudaScratchSlot::B,
            4,
            "test coordinator graph registry scratch",
        )?;

        assert_ne!(first_buffer.ptr, second_buffer.ptr);
        assert!(first_slot.acquisitions >= 1);
        assert!(second_slot.acquisitions >= 1);
        Ok(())
    }

    #[test]
    fn coordinator_cuda_graph_workspace_slot_access_provides_stream_and_workspace() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let key = registry.keys()[0].clone();

        let (stream, buffer) =
            with_coordinator_cuda_graph_workspace_slot(&key, |library, stream, workspace| {
                let buffer = workspace.buffer(
                    library,
                    CoordinatorCudaScratchSlot::C,
                    4,
                    "test coordinator graph slot access scratch",
                )?;
                Ok((stream, buffer))
            })?;

        assert!(!stream.is_null());
        assert!(!buffer.ptr.is_null());
        assert!(buffer.bytes >= 4);
        Ok(())
    }

    #[test]
    fn coordinator_cuda_graph_workspace_slot_preserves_capture_on_stable_scratch() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let key = registry.keys()[0].clone();
        let rows = 1_i32;
        let hidden = 4_i32;
        let values = rows as usize * hidden as usize;
        let bytes = values * std::mem::size_of::<u16>();
        let signature =
            CoordinatorCudaGraphSignature::ad_hoc(values, rows as usize, hidden as usize);

        let graph_captures_before =
            capture_ad_hoc_rmsnorm_graph_for_workspace_test(&key, rows, hidden, signature)?;
        with_coordinator_cuda_graph_workspace_slot(&key, |library, _stream, workspace| {
            let first = workspace.buffer(
                library,
                CoordinatorCudaScratchSlot::A,
                bytes,
                "test coordinator legacy stable scratch",
            )?;
            let second = workspace.buffer(
                library,
                CoordinatorCudaScratchSlot::A,
                bytes,
                "test coordinator legacy stable scratch",
            )?;
            assert_eq!(first.ptr, second.ptr);
            Ok(())
        })?;
        let (has_graph, graph_captures_after) =
            with_coordinator_cuda_graph_slot(&key, |_library, slot| {
                Ok((
                    slot.has_captured_graph(CoordinatorCudaGraphProgram::AdHocTest, signature),
                    slot.graph_captures,
                ))
            })?;

        assert!(has_graph);
        assert_eq!(graph_captures_after, graph_captures_before);
        Ok(())
    }

    #[test]
    fn coordinator_cuda_graph_workspace_slot_clears_capture_on_scratch_resize() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let key = registry.keys()[0].clone();
        let rows = 1_i32;
        let hidden = 6_i32;
        let values = rows as usize * hidden as usize;
        let signature =
            CoordinatorCudaGraphSignature::ad_hoc(values, rows as usize, hidden as usize);

        capture_ad_hoc_rmsnorm_graph_for_workspace_test(&key, rows, hidden, signature)?;
        with_coordinator_cuda_graph_workspace_slot(&key, |library, _stream, workspace| {
            let (_, capacity) = workspace
                .scratch_slot_state(CoordinatorCudaScratchSlot::A)
                .ok_or_else(|| {
                    anyhow::anyhow!("expected test scratch slot A to be allocated before resize")
                })?;
            workspace.buffer(
                library,
                CoordinatorCudaScratchSlot::A,
                capacity + 1,
                "test coordinator legacy resized scratch",
            )?;
            Ok(())
        })?;
        let has_graph = with_coordinator_cuda_graph_slot(&key, |_library, slot| {
            Ok(slot.has_captured_graph(CoordinatorCudaGraphProgram::AdHocTest, signature))
        })?;

        assert!(!has_graph);
        Ok(())
    }

    #[test]
    fn coordinator_cuda_graph_slot_captures_and_replays_bf16_kernels() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let key = registry.keys()[0].clone();
        let rows = 1_i32;
        let hidden = 8_i32;
        let values = rows as usize * hidden as usize;
        let bytes = values * std::mem::size_of::<u16>();
        let eps = 1.0e-5_f32;
        let x0 = bf16_bytes(&[0.1, 0.2, 0.3, 0.4, -0.5, -0.4, -0.3, -0.2]);
        let x1 = bf16_bytes(&[0.8, 0.7, -0.6, -0.5, 0.4, 0.3, -0.2, 0.1]);
        let weight = bf16_bytes(&vec![1.25_f32; hidden as usize]);
        let weight_alt = bf16_bytes(&vec![0.75_f32; hidden as usize]);
        let delta = bf16_bytes(&[0.05, -0.10, 0.15, -0.20, 0.25, -0.30, 0.35, -0.40]);
        let signature =
            CoordinatorCudaGraphSignature::ad_hoc(values, rows as usize, hidden as usize);

        let (graph_captures, graph_launches, graph_output, updated_graph_output) =
            with_coordinator_cuda_graph_slot(&key, |library, slot| {
                let stream = slot.stream_ptr();
                let x_buffer = slot.buffer(
                    library,
                    CoordinatorCudaScratchSlot::A,
                    bytes,
                    "test coordinator graph capture x",
                )?;
                let weight_buffer = slot.buffer(
                    library,
                    CoordinatorCudaScratchSlot::B,
                    bytes,
                    "test coordinator graph capture weight",
                )?;
                let weight_alt_buffer = slot.buffer(
                    library,
                    CoordinatorCudaScratchSlot::E,
                    bytes,
                    "test coordinator graph capture alternate weight",
                )?;
                let delta_buffer = slot.buffer(
                    library,
                    CoordinatorCudaScratchSlot::C,
                    bytes,
                    "test coordinator graph capture delta",
                )?;
                let out_buffer = slot.buffer(
                    library,
                    CoordinatorCudaScratchSlot::D,
                    bytes,
                    "test coordinator graph capture output",
                )?;
                slot.workspace.copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::A,
                    &x0,
                    "test coordinator graph capture x",
                    stream,
                )?;
                slot.workspace.copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::B,
                    &weight,
                    "test coordinator graph capture weight",
                    stream,
                )?;
                slot.workspace.copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::E,
                    &weight_alt,
                    "test coordinator graph capture alternate weight",
                    stream,
                )?;
                slot.workspace.copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::C,
                    &delta,
                    "test coordinator graph capture delta",
                    stream,
                )?;
                slot.stream_synchronize()?;
                slot.capture_graph(
                    library,
                    CoordinatorCudaGraphProgram::AdHocTest,
                    signature,
                    |library, stream, _workspace| unsafe {
                        library.cuda_rmsnorm_bf16_async(
                            x_buffer,
                            weight_buffer,
                            out_buffer,
                            rows,
                            hidden,
                            eps,
                            stream,
                        )?;
                        library.cuda_residual_add_bf16_async(
                            out_buffer,
                            delta_buffer,
                            out_buffer,
                            values,
                            stream,
                        )?;
                        Ok(())
                    },
                )?;
                slot.launch_captured_graph(
                    library,
                    CoordinatorCudaGraphProgram::AdHocTest,
                    signature,
                )?;
                slot.stream_synchronize()?;
                let captured = slot
                    .captured_graphs
                    .iter()
                    .find(|entry| {
                        entry.program == CoordinatorCudaGraphProgram::AdHocTest
                            && entry.signature == signature
                    })
                    .expect("test graph capture entry exists");
                assert_eq!(captured.graph.node_count, 2);
                assert_eq!(captured.graph.kernel_node_count, 2);
                assert_eq!(captured.graph.memcpy_node_count, 0);
                assert_eq!(captured.graph.memset_node_count, 0);
                let mut out0 = vec![0_u8; bytes];
                library.copy_d2h(&mut out0, out_buffer)?;
                let expected0_norm =
                    rmsnorm_hidden_bf16(&x0, &weight, rows as usize, hidden as usize, eps)?;
                let expected0 =
                    residual_add_prefix_bf16_bytes(&bf16_bytes(&expected0_norm.values), &delta)?;
                assert_bf16_values_close(&out0, &expected0.bytes, 1.0e-2);

                slot.workspace.copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::A,
                    &x1,
                    "test coordinator graph replay x",
                    stream,
                )?;
                slot.stream_synchronize()?;
                slot.launch_captured_graph(
                    library,
                    CoordinatorCudaGraphProgram::AdHocTest,
                    signature,
                )?;
                slot.stream_synchronize()?;
                let mut out1 = vec![0_u8; bytes];
                library.copy_d2h(&mut out1, out_buffer)?;
                let captured = slot
                    .captured_graphs
                    .iter()
                    .find(|entry| {
                        entry.program == CoordinatorCudaGraphProgram::AdHocTest
                            && entry.signature == signature
                    })
                    .expect("test graph capture entry exists after replay");
                unsafe {
                    library.cuda_graph_update_rmsnorm_bf16_node(
                        captured.graph.graph_raw,
                        captured.graph.exec_raw,
                        0,
                        x_buffer,
                        weight_alt_buffer,
                        out_buffer,
                        rows,
                        hidden,
                        eps,
                    )?;
                }
                slot.launch_captured_graph(
                    library,
                    CoordinatorCudaGraphProgram::AdHocTest,
                    signature,
                )?;
                slot.stream_synchronize()?;
                let mut out2 = vec![0_u8; bytes];
                library.copy_d2h(&mut out2, out_buffer)?;
                Ok((slot.graph_captures, slot.graph_launches, out1, out2))
            })?;

        let expected_norm = rmsnorm_hidden_bf16(&x1, &weight, rows as usize, hidden as usize, eps)?;
        let expected = residual_add_prefix_bf16_bytes(&bf16_bytes(&expected_norm.values), &delta)?;
        let expected_updated_norm =
            rmsnorm_hidden_bf16(&x1, &weight_alt, rows as usize, hidden as usize, eps)?;
        let expected_updated =
            residual_add_prefix_bf16_bytes(&bf16_bytes(&expected_updated_norm.values), &delta)?;
        assert!(graph_captures >= 1);
        assert!(graph_launches >= 3);
        assert_bf16_values_close(&graph_output, &expected.bytes, 1.0e-2);
        assert_bf16_values_close(&updated_graph_output, &expected_updated.bytes, 1.0e-2);
        Ok(())
    }

    #[test]
    fn coordinator_cuda_graph_coord_dense_envelope_bf16_captures_fourteen_kernel_graph_and_replays_with_updates(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let rows = 1_usize;
        let hidden = 4_usize;
        let intermediate = 4_usize;
        let values = rows * hidden;
        let value_bytes = values * std::mem::size_of::<u16>();
        let linear_weight_bytes = hidden * hidden * std::mem::size_of::<u16>();
        let mlp_weight_bytes = hidden * intermediate * std::mem::size_of::<u16>();
        let key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordDense,
            LayerWaveMode::Decode,
            rows,
        )?;
        assert!(registry.keys().iter().any(|candidate| candidate == &key));
        let signature =
            CoordinatorCudaGraphSignature::coord_dense_envelope_bf16(rows, hidden, intermediate);

        let (capture_delta, launch_delta, node_counts, out_a, out_b) =
            with_coordinator_cuda_graph_slot(&key, |library, slot| {
                let input_a = patterned_bf16_bytes(values, 0.03125, 0.25);
                let input_b = patterned_bf16_bytes(values, 0.0234375, -0.125);
                let norm_weight = constant_bf16_bytes(hidden, 1.0);
                let norm_weight_alt = constant_bf16_bytes(hidden, 0.75);
                let q_weight = patterned_bf16_bytes(hidden * hidden, 0.0078125, 0.0625);
                let k_weight = patterned_bf16_bytes(hidden * hidden, 0.0068359375, -0.03125);
                let v_weight = patterned_bf16_bytes(hidden * hidden, 0.005859375, 0.046875);
                let o_weight = patterned_bf16_bytes(hidden * hidden, 0.0048828125, 0.015625);
                let probe_a_weight = patterned_bf16_bytes(hidden * hidden, 0.00390625, 0.03125);
                let probe_b_weight = patterned_bf16_bytes(hidden * hidden, 0.0029296875, -0.015625);
                let gate_weight = patterned_bf16_bytes(hidden * intermediate, 0.00390625, 0.125);
                let up_weight = patterned_bf16_bytes(hidden * intermediate, 0.0029296875, 0.09375);
                let down_weight =
                    patterned_bf16_bytes(hidden * intermediate, 0.001953125, 0.078125);

                let input = uploaded_test_device_buffer(library, &input_a, "dense envelope input")?;
                let norm0_weight =
                    uploaded_test_device_buffer(library, &norm_weight, "dense envelope norm0")?;
                let norm1_weight =
                    uploaded_test_device_buffer(library, &norm_weight_alt, "dense envelope norm1")?;
                let q_weight =
                    uploaded_test_device_buffer(library, &q_weight, "dense envelope q weight")?;
                let k_weight =
                    uploaded_test_device_buffer(library, &k_weight, "dense envelope k weight")?;
                let v_weight =
                    uploaded_test_device_buffer(library, &v_weight, "dense envelope v weight")?;
                let o_weight =
                    uploaded_test_device_buffer(library, &o_weight, "dense envelope o weight")?;
                let probe_a_weight = uploaded_test_device_buffer(
                    library,
                    &probe_a_weight,
                    "dense envelope probe a weight",
                )?;
                let probe_b_weight = uploaded_test_device_buffer(
                    library,
                    &probe_b_weight,
                    "dense envelope probe b weight",
                )?;
                let gate_weight =
                    uploaded_test_device_buffer(library, &gate_weight, "dense envelope gate")?;
                let up_weight =
                    uploaded_test_device_buffer(library, &up_weight, "dense envelope up")?;
                let down_weight =
                    uploaded_test_device_buffer(library, &down_weight, "dense envelope down")?;
                assert!(q_weight.buffer.bytes >= linear_weight_bytes);
                assert!(gate_weight.buffer.bytes >= mlp_weight_bytes);

                let norm0 =
                    empty_test_device_buffer(library, value_bytes, "dense envelope norm0 out")?;
                let q = empty_test_device_buffer(library, value_bytes, "dense envelope q")?;
                let k = empty_test_device_buffer(library, value_bytes, "dense envelope k")?;
                let v = empty_test_device_buffer(library, value_bytes, "dense envelope v")?;
                let attention =
                    empty_test_device_buffer(library, value_bytes, "dense envelope attention")?;
                let attention_proj = empty_test_device_buffer(
                    library,
                    value_bytes,
                    "dense envelope attention proj",
                )?;
                let attention_residual = empty_test_device_buffer(
                    library,
                    value_bytes,
                    "dense envelope attention residual",
                )?;
                let mlp_norm =
                    empty_test_device_buffer(library, value_bytes, "dense envelope mlp norm")?;
                let probe_a =
                    empty_test_device_buffer(library, value_bytes, "dense envelope probe a")?;
                let probe_b =
                    empty_test_device_buffer(library, value_bytes, "dense envelope probe b")?;
                let probe_mix =
                    empty_test_device_buffer(library, value_bytes, "dense envelope probe mix")?;
                let mlp_out =
                    empty_test_device_buffer(library, value_bytes, "dense envelope mlp out")?;
                let mlp_delta =
                    empty_test_device_buffer(library, value_bytes, "dense envelope mlp delta")?;
                let output =
                    empty_test_device_buffer(library, value_bytes, "dense envelope output")?;

                let buffers = CoordDenseEnvelopeBf16Buffers {
                    input: input.buffer,
                    norm0_weight: norm0_weight.buffer,
                    norm0_out: norm0.buffer,
                    q_weight: q_weight.buffer,
                    q_out: q.buffer,
                    k_weight: k_weight.buffer,
                    k_out: k.buffer,
                    v_weight: v_weight.buffer,
                    v_out: v.buffer,
                    attention_out: attention.buffer,
                    o_weight: o_weight.buffer,
                    attention_proj: attention_proj.buffer,
                    attention_residual: attention_residual.buffer,
                    norm1_weight: norm1_weight.buffer,
                    mlp_norm: mlp_norm.buffer,
                    probe_a_weight: probe_a_weight.buffer,
                    probe_a_out: probe_a.buffer,
                    probe_b_weight: probe_b_weight.buffer,
                    probe_b_out: probe_b.buffer,
                    probe_mix: probe_mix.buffer,
                    gate_weight: gate_weight.buffer,
                    up_weight: up_weight.buffer,
                    down_weight: down_weight.buffer,
                    mlp_out: mlp_out.buffer,
                    mlp_delta: mlp_delta.buffer,
                    output: output.buffer,
                };
                let captures_before = slot.graph_captures;
                let launches_before = slot.graph_launches;
                capture_or_update_coord_dense_envelope_bf16_graph(
                    library,
                    slot,
                    signature,
                    buffers,
                    rows,
                    hidden,
                    intermediate,
                    "dense envelope test",
                )?;
                slot.stream_synchronize()?;
                let mut out_a = vec![0_u8; value_bytes];
                library.copy_d2h(&mut out_a, output.buffer)?;

                library.copy_h2d(input.buffer, &input_b)?;
                capture_or_update_coord_dense_envelope_bf16_graph(
                    library,
                    slot,
                    signature,
                    buffers,
                    rows,
                    hidden,
                    intermediate,
                    "dense envelope test update",
                )?;
                slot.stream_synchronize()?;
                let mut out_b = vec![0_u8; value_bytes];
                library.copy_d2h(&mut out_b, output.buffer)?;
                let node_counts = slot
                    .captured_graph_node_counts(
                        CoordinatorCudaGraphProgram::CoordDenseEnvelopeBf16,
                        signature,
                    )
                    .context("dense envelope graph should report node counts")?;
                Ok((
                    slot.graph_captures - captures_before,
                    slot.graph_launches - launches_before,
                    node_counts,
                    out_a,
                    out_b,
                ))
            })?;

        assert_eq!(capture_delta, 1);
        assert_eq!(launch_delta, 2);
        assert_eq!(node_counts, (14, 14, 0, 0));
        assert!(output_has_finite_nonzero_bf16_values(&out_a));
        assert!(output_has_finite_nonzero_bf16_values(&out_b));
        assert_ne!(out_a, out_b);
        Ok(())
    }

    #[test]
    fn coordinator_cuda_graph_coord_sparse_a_envelope_bf16_captures_twelve_op_graph_and_replays_with_updates(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let rows = 1_usize;
        let hidden = 4_usize;
        let intermediate = 4_usize;
        let experts = 4_usize;
        let top_k = 2_usize;
        let values = rows * hidden;
        let value_bytes = values * std::mem::size_of::<u16>();
        let topk_values = rows * top_k;
        let key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseA,
            LayerWaveMode::Decode,
            rows,
        )?;
        assert!(registry.keys().iter().any(|candidate| candidate == &key));
        let signature = CoordinatorCudaGraphSignature::coord_sparse_a_envelope_bf16(
            rows,
            hidden,
            intermediate,
            experts,
            top_k,
        );

        let (capture_delta, launch_delta, node_counts, shared_a, shared_b, scores_a, scores_b) =
            with_coordinator_cuda_graph_slot(&key, |library, slot| {
                let input_a = patterned_bf16_bytes(values, 0.02734375, 0.1875);
                let input_b = patterned_bf16_bytes(values, 0.01953125, -0.0625);
                let norm_weight = constant_bf16_bytes(hidden, 1.0);
                let post_norm_weight = constant_bf16_bytes(hidden, 0.875);
                let q_nope_weight = patterned_bf16_bytes(hidden * hidden, 0.0068359375, 0.046875);
                let q_rope_weight = patterned_bf16_bytes(hidden * hidden, 0.005859375, -0.0234375);
                let k_nope_weight = patterned_bf16_bytes(hidden * hidden, 0.0048828125, 0.0390625);
                let k_rope_weight = patterned_bf16_bytes(hidden * hidden, 0.00439453125, -0.015625);
                let value_weight = patterned_bf16_bytes(hidden * hidden, 0.00341796875, 0.0546875);
                let o_weight = patterned_bf16_bytes(hidden * hidden, 0.00390625, 0.03125);
                let gate_weight = patterned_bf16_bytes(hidden * intermediate, 0.0029296875, 0.125);
                let up_weight = patterned_bf16_bytes(hidden * intermediate, 0.001953125, 0.09375);
                let down_weight =
                    patterned_bf16_bytes(hidden * intermediate, 0.00146484375, 0.078125);
                let router_weight = patterned_bf16_bytes(hidden * experts, 0.0078125, 0.0625);
                let correction_bias = vec![0.0_f32, 0.01, -0.015, 0.02];

                let input =
                    uploaded_test_device_buffer(library, &input_a, "sparse-a envelope input")?;
                let norm0_weight =
                    uploaded_test_device_buffer(library, &norm_weight, "sparse-a norm0")?;
                let norm1_weight =
                    uploaded_test_device_buffer(library, &post_norm_weight, "sparse-a norm1")?;
                let q_nope_weight =
                    uploaded_test_device_buffer(library, &q_nope_weight, "sparse-a q_nope weight")?;
                let q_rope_weight =
                    uploaded_test_device_buffer(library, &q_rope_weight, "sparse-a q_rope weight")?;
                let k_nope_weight =
                    uploaded_test_device_buffer(library, &k_nope_weight, "sparse-a k_nope weight")?;
                let k_rope_weight =
                    uploaded_test_device_buffer(library, &k_rope_weight, "sparse-a k_rope weight")?;
                let value_weight =
                    uploaded_test_device_buffer(library, &value_weight, "sparse-a value weight")?;
                let o_weight =
                    uploaded_test_device_buffer(library, &o_weight, "sparse-a o weight")?;
                let gate_weight =
                    uploaded_test_device_buffer(library, &gate_weight, "sparse-a gate")?;
                let up_weight = uploaded_test_device_buffer(library, &up_weight, "sparse-a up")?;
                let down_weight =
                    uploaded_test_device_buffer(library, &down_weight, "sparse-a down")?;
                let router_weight =
                    uploaded_test_device_buffer(library, &router_weight, "sparse-a router weight")?;
                let correction_bias = uploaded_test_device_buffer(
                    library,
                    f32_bytes(&correction_bias),
                    "sparse-a router correction bias",
                )?;

                let norm0 = empty_test_device_buffer(library, value_bytes, "sparse-a norm0 out")?;
                let q_nope = empty_test_device_buffer(library, value_bytes, "sparse-a q_nope")?;
                let q_rope = empty_test_device_buffer(library, value_bytes, "sparse-a q_rope")?;
                let k_nope = empty_test_device_buffer(library, value_bytes, "sparse-a k_nope")?;
                let k_rope = empty_test_device_buffer(library, value_bytes, "sparse-a k_rope")?;
                let value = empty_test_device_buffer(library, value_bytes, "sparse-a value")?;
                let attention =
                    empty_test_device_buffer(library, value_bytes, "sparse-a attention")?;
                let attention_proj =
                    empty_test_device_buffer(library, value_bytes, "sparse-a attention proj")?;
                let attention_residual =
                    empty_test_device_buffer(library, value_bytes, "sparse-a residual")?;
                let moe_norm = empty_test_device_buffer(library, value_bytes, "sparse-a moe norm")?;
                let shared_out =
                    empty_test_device_buffer(library, value_bytes, "sparse-a shared out")?;
                let topk_indices = empty_test_device_buffer(
                    library,
                    topk_values * std::mem::size_of::<u32>(),
                    "sparse-a topk indices",
                )?;
                let topk_scores = empty_test_device_buffer(
                    library,
                    topk_values * std::mem::size_of::<f32>(),
                    "sparse-a topk scores",
                )?;
                let topk_weights = empty_test_device_buffer(
                    library,
                    topk_values * std::mem::size_of::<f32>(),
                    "sparse-a topk weights",
                )?;

                let buffers = CoordSparseAEnvelopeBf16Buffers {
                    input: input.buffer,
                    norm0_weight: norm0_weight.buffer,
                    norm0_out: norm0.buffer,
                    q_nope_weight: q_nope_weight.buffer,
                    q_nope_out: q_nope.buffer,
                    q_rope_weight: q_rope_weight.buffer,
                    q_rope_out: q_rope.buffer,
                    k_nope_weight: k_nope_weight.buffer,
                    k_nope_out: k_nope.buffer,
                    k_rope_weight: k_rope_weight.buffer,
                    k_rope_out: k_rope.buffer,
                    value_weight: value_weight.buffer,
                    value_out: value.buffer,
                    attention_out: attention.buffer,
                    o_weight: o_weight.buffer,
                    attention_proj: attention_proj.buffer,
                    attention_residual: attention_residual.buffer,
                    norm1_weight: norm1_weight.buffer,
                    moe_norm: moe_norm.buffer,
                    gate_weight: gate_weight.buffer,
                    up_weight: up_weight.buffer,
                    down_weight: down_weight.buffer,
                    shared_out: shared_out.buffer,
                    router_weight: router_weight.buffer,
                    correction_bias: correction_bias.buffer,
                    topk_indices: topk_indices.buffer,
                    topk_scores: topk_scores.buffer,
                    topk_weights: topk_weights.buffer,
                };
                let captures_before = slot.graph_captures;
                let launches_before = slot.graph_launches;
                capture_or_update_coord_sparse_a_envelope_bf16_graph(
                    library,
                    slot,
                    signature,
                    buffers,
                    rows,
                    hidden,
                    intermediate,
                    experts,
                    top_k,
                    "Sparse-A envelope test",
                )?;
                slot.stream_synchronize()?;
                let mut shared_a = vec![0_u8; value_bytes];
                let mut scores_a = vec![0_u8; topk_values * std::mem::size_of::<f32>()];
                library.copy_d2h(&mut shared_a, shared_out.buffer)?;
                library.copy_d2h(&mut scores_a, topk_scores.buffer)?;

                library.copy_h2d(input.buffer, &input_b)?;
                capture_or_update_coord_sparse_a_envelope_bf16_graph(
                    library,
                    slot,
                    signature,
                    buffers,
                    rows,
                    hidden,
                    intermediate,
                    experts,
                    top_k,
                    "Sparse-A envelope test update",
                )?;
                slot.stream_synchronize()?;
                let mut shared_b = vec![0_u8; value_bytes];
                let mut scores_b = vec![0_u8; topk_values * std::mem::size_of::<f32>()];
                library.copy_d2h(&mut shared_b, shared_out.buffer)?;
                library.copy_d2h(&mut scores_b, topk_scores.buffer)?;
                let node_counts = slot
                    .captured_graph_node_counts(
                        CoordinatorCudaGraphProgram::CoordSparseAEnvelopeBf16,
                        signature,
                    )
                    .context("Sparse-A envelope graph should report node counts")?;
                Ok((
                    slot.graph_captures - captures_before,
                    slot.graph_launches - launches_before,
                    node_counts,
                    shared_a,
                    shared_b,
                    scores_a,
                    scores_b,
                ))
            })?;

        assert_eq!(capture_delta, 1);
        assert_eq!(launch_delta, 2);
        // Sparse-A is a 12-op envelope; router top-k is one envelope op backed
        // by init/score/finalize kernels, so the physical CUDA graph has 14 nodes.
        assert_eq!(node_counts, (14, 14, 0, 0));
        assert!(output_has_finite_nonzero_bf16_values(&shared_a));
        assert!(output_has_finite_nonzero_bf16_values(&shared_b));
        assert_ne!(shared_a, shared_b);
        assert_ne!(scores_a, scores_b);
        Ok(())
    }

    #[test]
    fn coordinator_cuda_graph_slot_updates_linear_bf16_node() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let key = registry.keys()[0].clone();
        let rows = 2_usize;
        let input_dim = 2_usize;
        let output_dim = 2_usize;
        let input = bf16_bytes(&[1.0, 2.0, -1.0, 0.5]);
        let weight_a = bf16_bytes(&[1.0, 0.0, 0.0, 1.0]);
        let bias_a = bf16_bytes(&[0.0, 0.0]);
        let weight_b = bf16_bytes(&[2.0, 0.0, 0.0, -1.0]);
        let bias_b = bf16_bytes(&[0.5, 1.0]);
        let output_bytes = rows * output_dim * std::mem::size_of::<u16>();
        let signature = CoordinatorCudaGraphSignature::ad_hoc(output_dim * rows, rows, output_dim);

        let (out_a, out_b, graph_launches) =
            with_coordinator_cuda_graph_slot(&key, |library, slot| {
                let stream = slot.stream_ptr();
                let input_buffer = slot.buffer(
                    library,
                    CoordinatorCudaScratchSlot::A,
                    input.len(),
                    "test coordinator graph linear input",
                )?;
                let weight_a_buffer = slot.buffer(
                    library,
                    CoordinatorCudaScratchSlot::B,
                    weight_a.len(),
                    "test coordinator graph linear weight a",
                )?;
                let bias_a_buffer = slot.buffer(
                    library,
                    CoordinatorCudaScratchSlot::C,
                    bias_a.len(),
                    "test coordinator graph linear bias a",
                )?;
                let output_buffer = slot.buffer(
                    library,
                    CoordinatorCudaScratchSlot::D,
                    output_bytes,
                    "test coordinator graph linear output",
                )?;
                let weight_b_buffer = slot.buffer(
                    library,
                    CoordinatorCudaScratchSlot::E,
                    weight_b.len(),
                    "test coordinator graph linear weight b",
                )?;
                let bias_b_buffer = slot.buffer(
                    library,
                    CoordinatorCudaScratchSlot::F,
                    bias_b.len(),
                    "test coordinator graph linear bias b",
                )?;
                slot.workspace.copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::A,
                    &input,
                    "test coordinator graph linear input",
                    stream,
                )?;
                slot.workspace.copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::B,
                    &weight_a,
                    "test coordinator graph linear weight a",
                    stream,
                )?;
                slot.workspace.copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::C,
                    &bias_a,
                    "test coordinator graph linear bias a",
                    stream,
                )?;
                slot.workspace.copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::E,
                    &weight_b,
                    "test coordinator graph linear weight b",
                    stream,
                )?;
                slot.workspace.copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::F,
                    &bias_b,
                    "test coordinator graph linear bias b",
                    stream,
                )?;
                slot.stream_synchronize()?;
                slot.capture_graph(
                    library,
                    CoordinatorCudaGraphProgram::AdHocTest,
                    signature,
                    |library, stream, _workspace| unsafe {
                        library.cuda_linear_bf16_async(
                            input_buffer,
                            weight_a_buffer,
                            Some(bias_a_buffer),
                            output_buffer,
                            rows,
                            input_dim,
                            output_dim,
                            stream,
                        )?;
                        Ok(())
                    },
                )?;
                slot.launch_captured_graph(
                    library,
                    CoordinatorCudaGraphProgram::AdHocTest,
                    signature,
                )?;
                slot.stream_synchronize()?;
                let mut out_a = vec![0_u8; output_bytes];
                library.copy_d2h(&mut out_a, output_buffer)?;
                let captured = slot
                    .captured_graphs
                    .iter()
                    .find(|entry| {
                        entry.program == CoordinatorCudaGraphProgram::AdHocTest
                            && entry.signature == signature
                    })
                    .expect("test linear graph capture entry exists");
                unsafe {
                    library.cuda_graph_update_linear_bf16_node(
                        captured.graph.graph_raw,
                        captured.graph.exec_raw,
                        0,
                        input_buffer,
                        weight_b_buffer,
                        Some(bias_b_buffer),
                        output_buffer,
                        rows,
                        input_dim,
                        output_dim,
                    )?;
                }
                slot.launch_captured_graph(
                    library,
                    CoordinatorCudaGraphProgram::AdHocTest,
                    signature,
                )?;
                slot.stream_synchronize()?;
                let mut out_b = vec![0_u8; output_bytes];
                library.copy_d2h(&mut out_b, output_buffer)?;
                Ok((out_a, out_b, slot.graph_launches))
            })?;

        let out_a = bf16_values_to_f32(&out_a);
        assert!((out_a[0] - 1.0).abs() < 1.0e-3);
        assert!((out_a[1] - 2.0).abs() < 1.0e-3);
        assert!((out_a[2] + 1.0).abs() < 1.0e-3);
        assert!((out_a[3] - 0.5).abs() < 1.0e-3);
        let out_b = bf16_values_to_f32(&out_b);
        assert!((out_b[0] - 2.5).abs() < 1.0e-3);
        assert!((out_b[1] + 1.0).abs() < 1.0e-3);
        assert!((out_b[2] + 1.5).abs() < 1.0e-3);
        assert!((out_b[3] - 0.5).abs() < 1.0e-3);
        assert!(graph_launches >= 2);
        Ok(())
    }

    #[test]
    fn coordinator_cuda_graph_slot_updates_router_topk_bf16_node() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let key = registry.keys()[0].clone();
        let rows = 2_usize;
        let hidden_dim = 3_usize;
        let experts = 4_usize;
        let top_k = 2_usize;
        let hidden = bf16_bytes(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let router_weight_a =
            bf16_bytes(&[3.0, 0.0, 0.0, 0.0, 3.0, 0.0, -3.0, 0.0, 0.0, 0.0, -3.0, 0.0]);
        let router_weight_b =
            bf16_bytes(&[-3.0, 0.0, 0.0, 0.0, -3.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0, 0.0]);
        let correction_bias_a = [0.0_f32, 0.0, 0.0, 0.0];
        let correction_bias_b = [0.0_f32, 0.0, 0.25, 0.25];
        let output_values = rows * top_k;
        let index_bytes = output_values * std::mem::size_of::<u32>();
        let score_bytes = output_values * std::mem::size_of::<f32>();
        let signature =
            CoordinatorCudaGraphSignature::router_topk_bf16(rows, hidden_dim, experts, top_k);

        let (indices_a, scores_a, weights_a, indices_b, scores_b, weights_b, graph_launches) =
            with_coordinator_cuda_graph_slot(&key, |library, slot| {
                let stream = slot.stream_ptr();
                let hidden_buffer = slot.buffer(
                    library,
                    CoordinatorCudaScratchSlot::A,
                    hidden.len(),
                    "test coordinator graph router hidden",
                )?;
                let weight_a_buffer = slot.buffer(
                    library,
                    CoordinatorCudaScratchSlot::B,
                    router_weight_a.len(),
                    "test coordinator graph router weight a",
                )?;
                let bias_a_buffer = slot.buffer(
                    library,
                    CoordinatorCudaScratchSlot::C,
                    std::mem::size_of_val(&correction_bias_a),
                    "test coordinator graph router bias a",
                )?;
                let index_buffer = slot.buffer(
                    library,
                    CoordinatorCudaScratchSlot::D,
                    index_bytes,
                    "test coordinator graph router indices",
                )?;
                let score_buffer = slot.buffer(
                    library,
                    CoordinatorCudaScratchSlot::E,
                    score_bytes,
                    "test coordinator graph router scores",
                )?;
                let topk_weight_buffer = slot.buffer(
                    library,
                    CoordinatorCudaScratchSlot::F,
                    score_bytes,
                    "test coordinator graph router weights",
                )?;
                let mut weight_b_buffer = library.alloc_device_buffer(router_weight_b.len())?;
                let mut bias_b_buffer =
                    library.alloc_device_buffer(std::mem::size_of_val(&correction_bias_b))?;

                slot.workspace.copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::A,
                    &hidden,
                    "test coordinator graph router hidden",
                    stream,
                )?;
                slot.workspace.copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::B,
                    &router_weight_a,
                    "test coordinator graph router weight a",
                    stream,
                )?;
                slot.workspace.copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::C,
                    f32_bytes(&correction_bias_a),
                    "test coordinator graph router bias a",
                    stream,
                )?;
                unsafe {
                    library.copy_h2d_async(weight_b_buffer, &router_weight_b, stream)?;
                    library.copy_h2d_async(bias_b_buffer, f32_bytes(&correction_bias_b), stream)?;
                }
                slot.stream_synchronize()?;
                slot.capture_graph(
                    library,
                    CoordinatorCudaGraphProgram::AdHocTest,
                    signature,
                    |library, stream, _workspace| unsafe {
                        library.cuda_router_topk_bf16_async(
                            hidden_buffer,
                            weight_a_buffer,
                            bias_a_buffer,
                            index_buffer,
                            score_buffer,
                            topk_weight_buffer,
                            rows,
                            hidden_dim,
                            experts,
                            top_k,
                            stream,
                        )?;
                        Ok(())
                    },
                )?;
                slot.launch_captured_graph(
                    library,
                    CoordinatorCudaGraphProgram::AdHocTest,
                    signature,
                )?;
                slot.stream_synchronize()?;
                let mut index_out_a = vec![0_u8; index_bytes];
                let mut score_out_a = vec![0_u8; score_bytes];
                let mut weight_out_a = vec![0_u8; score_bytes];
                library.copy_d2h(&mut index_out_a, index_buffer)?;
                library.copy_d2h(&mut score_out_a, score_buffer)?;
                library.copy_d2h(&mut weight_out_a, topk_weight_buffer)?;
                let (graph_raw, exec_raw) = {
                    let captured = slot
                        .captured_graphs
                        .iter()
                        .find(|entry| {
                            entry.program == CoordinatorCudaGraphProgram::AdHocTest
                                && entry.signature == signature
                        })
                        .expect("test router graph capture entry exists");
                    (captured.graph.graph_raw, captured.graph.exec_raw)
                };
                unsafe {
                    library.cuda_graph_update_router_topk_bf16_node(
                        graph_raw,
                        exec_raw,
                        0,
                        hidden_buffer,
                        weight_b_buffer,
                        bias_b_buffer,
                        index_buffer,
                        score_buffer,
                        topk_weight_buffer,
                        rows,
                        hidden_dim,
                        experts,
                        top_k,
                    )?;
                }
                slot.launch_captured_graph(
                    library,
                    CoordinatorCudaGraphProgram::AdHocTest,
                    signature,
                )?;
                slot.stream_synchronize()?;
                let mut index_out_b = vec![0_u8; index_bytes];
                let mut score_out_b = vec![0_u8; score_bytes];
                let mut weight_out_b = vec![0_u8; score_bytes];
                library.copy_d2h(&mut index_out_b, index_buffer)?;
                library.copy_d2h(&mut score_out_b, score_buffer)?;
                library.copy_d2h(&mut weight_out_b, topk_weight_buffer)?;
                let graph_launches = slot.graph_launches;

                unsafe {
                    library.cuda_graph_update_router_topk_bf16_node(
                        graph_raw,
                        exec_raw,
                        0,
                        hidden_buffer,
                        weight_a_buffer,
                        bias_a_buffer,
                        index_buffer,
                        score_buffer,
                        topk_weight_buffer,
                        rows,
                        hidden_dim,
                        experts,
                        top_k,
                    )?;
                }
                library.free_device_buffer(&mut bias_b_buffer)?;
                library.free_device_buffer(&mut weight_b_buffer)?;
                Ok((
                    u32_vec_from_bytes(&index_out_a)?,
                    f32_vec_from_bytes(&score_out_a)?,
                    f32_vec_from_bytes(&weight_out_a)?,
                    u32_vec_from_bytes(&index_out_b)?,
                    f32_vec_from_bytes(&score_out_b)?,
                    f32_vec_from_bytes(&weight_out_b)?,
                    graph_launches,
                ))
            })?;

        let sigmoid_3 = 1.0_f32 / (1.0 + (-3.0_f32).exp());
        let expected_scores = [sigmoid_3, 0.5, sigmoid_3, 0.5];
        let expected_weights = [
            GLM52_ROUTED_SCALING_FACTOR * sigmoid_3 / (sigmoid_3 + 0.5),
            GLM52_ROUTED_SCALING_FACTOR * 0.5 / (sigmoid_3 + 0.5),
            GLM52_ROUTED_SCALING_FACTOR * sigmoid_3 / (sigmoid_3 + 0.5),
            GLM52_ROUTED_SCALING_FACTOR * 0.5 / (sigmoid_3 + 0.5),
        ];
        assert_eq!(indices_a, vec![0, 1, 1, 0]);
        assert_eq!(indices_b, vec![2, 3, 3, 2]);
        for (actual, expected) in scores_a.iter().zip(expected_scores.iter()) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
        for (actual, expected) in scores_b.iter().zip(expected_scores.iter()) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
        for (actual, expected) in weights_a.iter().zip(expected_weights.iter()) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
        for (actual, expected) in weights_b.iter().zip(expected_weights.iter()) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
        assert!(graph_launches >= 2);
        Ok(())
    }

    #[test]
    fn coordinator_cuda_graph_slot_updates_strided_mlp_bf16_node() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let key = registry.keys()[0].clone();
        let rows = 2_usize;
        let hidden = 3_usize;
        let intermediate = 2_usize;
        let down_stride = 2_usize;
        let input = bf16_bytes(&[0.25, -0.5, 1.0, 0.75, 0.5, -0.25]);
        let gate_a = bf16_bytes(&[0.2, -0.1, 0.4, 0.3, 0.5, -0.2]);
        let up_a = bf16_bytes(&[0.5, 0.25, -0.3, 0.2, -0.4, 0.6]);
        let down_a = bf16_bytes(&[0.3, -0.2, 0.1, 0.4, -0.25, 0.5]);
        let gate_b = bf16_bytes(&[-0.3, 0.2, 0.1, -0.5, 0.4, 0.25]);
        let up_b = bf16_bytes(&[0.2, -0.4, 0.6, 0.1, 0.3, -0.2]);
        let down_b = bf16_bytes(&[-0.5, 0.3, 0.25, -0.15, 0.4, 0.2]);
        let expected_a = cpu_silu_gated_mlp_rows_bf16(
            &input,
            &gate_a,
            &up_a,
            &down_a,
            rows,
            hidden,
            intermediate,
            hidden,
        );
        let expected_b = cpu_silu_gated_mlp_rows_bf16(
            &input,
            &gate_b,
            &up_b,
            &down_b,
            rows,
            hidden,
            intermediate,
            hidden,
        );
        let output_bytes = rows * hidden * std::mem::size_of::<u16>();
        let signature = CoordinatorCudaGraphSignature::ad_hoc(rows * hidden, rows, hidden);

        let (out_a, out_b, graph_launches) =
            with_coordinator_cuda_graph_slot(&key, |library, slot| {
                let stream = slot.stream_ptr();
                let input_buffer = slot.buffer(
                    library,
                    CoordinatorCudaScratchSlot::A,
                    input.len(),
                    "test coordinator graph MLP input",
                )?;
                let gate_a_buffer = slot.buffer(
                    library,
                    CoordinatorCudaScratchSlot::B,
                    gate_a.len(),
                    "test coordinator graph MLP gate a",
                )?;
                let up_a_buffer = slot.buffer(
                    library,
                    CoordinatorCudaScratchSlot::C,
                    up_a.len(),
                    "test coordinator graph MLP up a",
                )?;
                let down_a_buffer = slot.buffer(
                    library,
                    CoordinatorCudaScratchSlot::D,
                    down_a.len(),
                    "test coordinator graph MLP down a",
                )?;
                let output_buffer = slot.buffer(
                    library,
                    CoordinatorCudaScratchSlot::E,
                    output_bytes,
                    "test coordinator graph MLP output",
                )?;
                let mut gate_b_buffer = library.alloc_device_buffer(gate_b.len())?;
                let mut up_b_buffer = library.alloc_device_buffer(up_b.len())?;
                let mut down_b_buffer = library.alloc_device_buffer(down_b.len())?;

                slot.workspace.copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::A,
                    &input,
                    "test coordinator graph MLP input",
                    stream,
                )?;
                slot.workspace.copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::B,
                    &gate_a,
                    "test coordinator graph MLP gate a",
                    stream,
                )?;
                slot.workspace.copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::C,
                    &up_a,
                    "test coordinator graph MLP up a",
                    stream,
                )?;
                slot.workspace.copy_h2d_to_slot_async(
                    library,
                    CoordinatorCudaScratchSlot::D,
                    &down_a,
                    "test coordinator graph MLP down a",
                    stream,
                )?;
                unsafe {
                    library.copy_h2d_async(gate_b_buffer, &gate_b, stream)?;
                    library.copy_h2d_async(up_b_buffer, &up_b, stream)?;
                    library.copy_h2d_async(down_b_buffer, &down_b, stream)?;
                }
                slot.stream_synchronize()?;
                slot.capture_graph(
                    library,
                    CoordinatorCudaGraphProgram::AdHocTest,
                    signature,
                    |library, stream, _workspace| unsafe {
                        library.cuda_silu_gated_mlp_rows_bf16_down_stride_async(
                            input_buffer,
                            gate_a_buffer,
                            up_a_buffer,
                            down_a_buffer,
                            output_buffer,
                            rows,
                            hidden,
                            intermediate,
                            down_stride,
                            stream,
                        )?;
                        Ok(())
                    },
                )?;
                slot.launch_captured_graph(
                    library,
                    CoordinatorCudaGraphProgram::AdHocTest,
                    signature,
                )?;
                slot.stream_synchronize()?;
                let mut out_a = vec![0_u8; output_bytes];
                library.copy_d2h(&mut out_a, output_buffer)?;
                let (graph_raw, exec_raw) = {
                    let captured = slot
                        .captured_graphs
                        .iter()
                        .find(|entry| {
                            entry.program == CoordinatorCudaGraphProgram::AdHocTest
                                && entry.signature == signature
                        })
                        .expect("test strided MLP graph capture entry exists");
                    (captured.graph.graph_raw, captured.graph.exec_raw)
                };
                unsafe {
                    library.cuda_graph_update_silu_gated_mlp_rows_bf16_down_stride_node(
                        graph_raw,
                        exec_raw,
                        0,
                        input_buffer,
                        gate_b_buffer,
                        up_b_buffer,
                        down_b_buffer,
                        output_buffer,
                        rows,
                        hidden,
                        intermediate,
                        down_stride,
                    )?;
                }
                slot.launch_captured_graph(
                    library,
                    CoordinatorCudaGraphProgram::AdHocTest,
                    signature,
                )?;
                slot.stream_synchronize()?;
                let mut out_b = vec![0_u8; output_bytes];
                library.copy_d2h(&mut out_b, output_buffer)?;
                let graph_launches = slot.graph_launches;

                unsafe {
                    library.cuda_graph_update_silu_gated_mlp_rows_bf16_down_stride_node(
                        graph_raw,
                        exec_raw,
                        0,
                        input_buffer,
                        gate_a_buffer,
                        up_a_buffer,
                        down_a_buffer,
                        output_buffer,
                        rows,
                        hidden,
                        intermediate,
                        down_stride,
                    )?;
                }
                library.free_device_buffer(&mut down_b_buffer)?;
                library.free_device_buffer(&mut up_b_buffer)?;
                library.free_device_buffer(&mut gate_b_buffer)?;
                Ok((out_a, out_b, graph_launches))
            })?;

        assert_bf16_values_close(&out_a, &bf16_bytes(&expected_a.values), 1.0e-5);
        assert_bf16_values_close(&out_b, &bf16_bytes(&expected_b.values), 1.0e-5);
        assert_ne!(out_a, out_b);
        assert!(graph_launches >= 2);
        Ok(())
    }

    fn capture_ad_hoc_rmsnorm_graph_for_workspace_test(
        key: &CoordinatorGraphKey,
        rows: i32,
        hidden: i32,
        signature: CoordinatorCudaGraphSignature,
    ) -> Result<usize> {
        let values = rows as usize * hidden as usize;
        let bytes = values * std::mem::size_of::<u16>();
        let x = bf16_bytes(&vec![0.25_f32; values]);
        let weight = bf16_bytes(&vec![1.0_f32; values]);
        with_coordinator_cuda_graph_slot(key, |library, slot| {
            let stream = slot.stream_ptr();
            let x_buffer = slot.buffer(
                library,
                CoordinatorCudaScratchSlot::A,
                bytes,
                "test coordinator graph sentinel x",
            )?;
            let weight_buffer = slot.buffer(
                library,
                CoordinatorCudaScratchSlot::B,
                bytes,
                "test coordinator graph sentinel weight",
            )?;
            let out_buffer = slot.buffer(
                library,
                CoordinatorCudaScratchSlot::C,
                bytes,
                "test coordinator graph sentinel output",
            )?;
            slot.workspace.copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::A,
                &x,
                "test coordinator graph sentinel x",
                stream,
            )?;
            slot.workspace.copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::B,
                &weight,
                "test coordinator graph sentinel weight",
                stream,
            )?;
            slot.stream_synchronize()?;
            slot.capture_graph(
                library,
                CoordinatorCudaGraphProgram::AdHocTest,
                signature,
                |library, stream, _workspace| unsafe {
                    library.cuda_rmsnorm_bf16_async(
                        x_buffer,
                        weight_buffer,
                        out_buffer,
                        rows,
                        hidden,
                        1.0e-5_f32,
                        stream,
                    )?;
                    Ok(())
                },
            )?;
            Ok(slot.graph_captures)
        })
    }

    #[test]
    fn resident_weight_preload_can_fill_reusable_pinned_staging_when_available() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let library = cuda_native_library()?;
        let mut resident_weights = lock_coordinator_cuda_resident_weights()?;
        let weight_name = format!(
            "test.resident.pinned-staging.{}.{}",
            std::process::id(),
            line!()
        );
        let payload = [3_u8, 5, 8, 13, 21, 34, 55, 89];
        let mut filled = false;
        let first = match resident_weights.resident_weight_buffer_from_host_staging(
            library,
            &weight_name,
            payload.len(),
            "test resident pinned staging",
            |staging| {
                assert_eq!(staging.len(), payload.len());
                staging.copy_from_slice(&payload);
                filled = true;
                Ok(())
            },
        ) {
            Ok(buffer) => buffer,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        assert!(filled);
        assert!(resident_weights.resident_weight_is_preloaded(&weight_name, payload.len()));

        let mut filled_again = false;
        let second = resident_weights.resident_weight_buffer_from_host_staging(
            library,
            &weight_name,
            payload.len(),
            "test resident pinned staging",
            |_| {
                filled_again = true;
                Ok(())
            },
        )?;

        assert_eq!(first.ptr, second.ptr);
        assert!(!filled_again);
        assert_eq!(
            resident_weights
                .resident_weights
                .get(&resident_weight_registry_key(&weight_name, payload.len()))
                .map(|resident| resident.upload_count),
            Some(1)
        );
        Ok(())
    }

    #[test]
    fn resident_weight_registry_allows_test_fixture_shape_aliases_when_available() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let library = cuda_native_library()?;
        let mut resident_weights = lock_coordinator_cuda_resident_weights()?;
        let weight_name = format!(
            "test.resident.same-name-different-shape.{}.{}",
            std::process::id(),
            line!()
        );
        let small = [1_u8, 2, 3, 4];
        let large = [5_u8, 6, 7, 8, 9, 10, 11, 12];

        let small_buffer = match resident_weights.resident_weight_buffer_from_host_staging(
            library,
            &weight_name,
            small.len(),
            "test resident small shape",
            |staging| {
                staging.copy_from_slice(&small);
                Ok(())
            },
        ) {
            Ok(buffer) => buffer,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let large_buffer = resident_weights.resident_weight_buffer_from_host_staging(
            library,
            &weight_name,
            large.len(),
            "test resident large shape",
            |staging| {
                staging.copy_from_slice(&large);
                Ok(())
            },
        )?;

        assert_ne!(small_buffer.ptr, large_buffer.ptr);
        assert!(resident_weights.resident_weight_is_preloaded(&weight_name, small.len()));
        assert!(resident_weights.resident_weight_is_preloaded(&weight_name, large.len()));
        assert_eq!(
            resident_weights
                .preloaded_resident_weight_buffer(&weight_name, small.len())?
                .bytes,
            small.len()
        );
        assert_eq!(
            resident_weights
                .preloaded_resident_weight_buffer(&weight_name, large.len())?
                .bytes,
            large.len()
        );
        Ok(())
    }

    fn cuda_allocation_unavailable(error: &anyhow::Error) -> bool {
        let message = error.to_string().to_ascii_lowercase();
        message.contains("returned status 3") || message.contains("cuda unavailable")
    }

    #[test]
    fn embedding_lookup_rows_uses_cpu_reference_when_cuda_reference_is_not_enabled() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let output = embedding_lookup_rows(
            &[
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, //
            ],
            &[11, 10],
            10,
            2,
            3,
        )
        .unwrap();

        assert_eq!(output.values, vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
        assert_eq!(output.backend, CPU_REFERENCE_EMBEDDING_LOOKUP_BACKEND);
    }

    #[test]
    fn embedding_lookup_rows_rejects_token_outside_loaded_window() {
        let err = embedding_lookup_rows(&[1.0, 2.0, 3.0], &[12], 10, 1, 3).unwrap_err();

        assert!(err.to_string().contains("outside loaded row window"));
    }

    #[test]
    fn embedding_lookup_rows_bf16_uses_cpu_reference_when_cuda_reference_is_not_enabled() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let embedding = bf16_bytes(&[
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
        ]);
        let output = embedding_lookup_rows_bf16(&embedding, &[11, 10], 10, 2, 3).unwrap();

        assert_eq!(output.values, vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
        assert_eq!(output.backend, CPU_REFERENCE_EMBEDDING_LOOKUP_BF16_BACKEND);
    }

    #[test]
    fn embedding_lookup_rows_bf16_rejects_shape_mismatch_before_backend_selection() {
        let embedding = bf16_bytes(&[1.0, 2.0, 3.0]);
        let err = embedding_lookup_rows_bf16(&embedding, &[10], 10, 2, 3).unwrap_err();

        assert!(err
            .to_string()
            .contains("BF16 embedding lookup row-window byte length mismatch"));
    }

    #[test]
    fn embedding_lookup_rows_bf16_resident_weight_uses_cpu_reference_when_cuda_reference_is_not_enabled(
    ) {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let embedding = bf16_bytes(&[
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
        ]);
        let output = embedding_lookup_rows_bf16_resident_weight(
            "model.embed_tokens.weight[rows=10..12]",
            &embedding,
            &[11, 10],
            10,
            2,
            3,
        )
        .unwrap();

        assert_eq!(output.values, vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
        assert_eq!(output.backend, CPU_REFERENCE_EMBEDDING_LOOKUP_BF16_BACKEND);
    }

    #[test]
    fn embedding_lookup_rows_bf16_resident_weight_rejects_empty_weight_name() {
        let embedding = bf16_bytes(&[1.0, 2.0, 3.0]);
        let err = embedding_lookup_rows_bf16_resident_weight("", &embedding, &[10], 10, 1, 3)
            .unwrap_err();

        assert!(err.to_string().contains("weight name must not be empty"));
    }

    #[test]
    fn embedding_lookup_rows_bf16_staged_resident_weight_requires_cuda_reference() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let err = embedding_lookup_rows_bf16_staged_resident_weight(
            "model.embed_tokens.weight[rows=10..11]",
            &[10],
            10,
            1,
            3,
        )
        .unwrap_err();

        assert!(err
            .to_string()
            .contains("staged resident BF16 embedding lookup requires"));
    }

    #[test]
    fn embedding_lookup_rows_bf16_staged_resident_weight_rejects_out_of_window_token() {
        let err = embedding_lookup_rows_bf16_staged_resident_weight(
            "model.embed_tokens.weight[rows=10..11]",
            &[11],
            10,
            1,
            3,
        )
        .unwrap_err();

        assert!(err.to_string().contains("outside loaded row window"));
    }

    #[test]
    fn embedding_lookup_bf16_preloaded_resident_weight_device_output_requires_cuda_reference() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let err = match embedding_lookup_bf16_preloaded_resident_weight_device_output(
            "model.embed_tokens.weight",
            &[0],
            2,
            3,
        ) {
            Ok(_) => panic!("expected device-output embedding lookup to require CUDA reference"),
            Err(error) => error,
        };

        assert!(err
            .to_string()
            .contains("preloaded resident BF16 embedding device-output lookup requires"));
    }

    #[test]
    fn embedding_lookup_bf16_preloaded_resident_weight_device_output_matches_host_readback(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };

        let embedding_name = format!(
            "model.embed_tokens.weight.device-output.test.{}.{}",
            std::process::id(),
            line!()
        );
        let embedding = bf16_bytes(&[
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0, //
        ]);
        match preload_resident_weight_from_host_staging(
            &embedding_name,
            embedding.len(),
            "test preloaded embedding device-output table",
            |staging| {
                staging.copy_from_slice(&embedding);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        }

        let graph_stats_before = match coordinator_cuda_graph_test_stats() {
            Ok(stats) => stats,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let output = match cuda_embedding_lookup_bf16_preloaded_resident_weight_device_output(
            &embedding_name,
            &[2_u32, 0],
            3,
            4,
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };

        assert_eq!(output.rows, 2);
        assert_eq!(output.values_per_row, 4);
        assert_eq!(
            output.backend,
            CUDA_REFERENCE_EMBEDDING_LOOKUP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(
            output.copy_to_host_values()?,
            vec![9.0, 10.0, 11.0, 12.0, 1.0, 2.0, 3.0, 4.0]
        );

        let graph_stats_after_first = coordinator_cuda_graph_test_stats()?;
        assert!(
            graph_stats_after_first.graph_captures >= graph_stats_before.graph_captures + 1,
            "expected first embedding lookup to capture a CUDA graph: before={graph_stats_before:?} after={graph_stats_after_first:?}"
        );
        assert!(
            graph_stats_after_first.graph_launches >= graph_stats_before.graph_launches + 1,
            "expected first embedding lookup to launch a CUDA graph: before={graph_stats_before:?} after={graph_stats_after_first:?}"
        );

        let second = cuda_embedding_lookup_bf16_preloaded_resident_weight_device_output(
            &embedding_name,
            &[1_u32, 2],
            3,
            4,
        )?;
        assert_eq!(second.rows, 2);
        assert_eq!(second.values_per_row, 4);
        assert_eq!(
            second.copy_to_host_values()?,
            vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        );
        let graph_stats_after_second = coordinator_cuda_graph_test_stats()?;
        assert_eq!(
            graph_stats_after_second.graph_captures, graph_stats_after_first.graph_captures,
            "expected second embedding lookup to replay the captured CUDA graph without recapturing: first={graph_stats_after_first:?} second={graph_stats_after_second:?}"
        );
        assert!(
            graph_stats_after_second.graph_launches >= graph_stats_after_first.graph_launches + 1,
            "expected second embedding lookup to launch the captured CUDA graph: first={graph_stats_after_first:?} second={graph_stats_after_second:?}"
        );
        Ok(())
    }

    #[test]
    fn embedding_lookup_bf16_preloaded_resident_weight_device_output_rejects_out_of_range_token() {
        let err = match embedding_lookup_bf16_preloaded_resident_weight_device_output(
            "model.embed_tokens.weight",
            &[2],
            2,
            3,
        ) {
            Ok(_) => panic!("expected device-output embedding lookup to reject out-of-range token"),
            Err(error) => error,
        };

        assert!(err
            .to_string()
            .contains("outside full embedding table [0, 2)"));
    }

    #[test]
    fn logits_argmax_uses_cpu_reference_when_cuda_reference_is_not_enabled() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let output = logits_argmax(
            &[
                0.25, 3.0, -1.0, //
                5.0, 5.5, -8.0, //
            ],
            2,
            3,
        )
        .unwrap();

        assert_eq!(output.indices, vec![1, 1]);
        assert_eq!(output.scores, vec![3.0, 5.5]);
        assert_eq!(output.backend, CPU_REFERENCE_LOGITS_ARGMAX_BACKEND);
    }

    #[test]
    fn logits_argmax_rejects_shape_mismatch_before_backend_selection() {
        let err = logits_argmax(&[1.0, 2.0], 1, 3).unwrap_err();

        assert!(err.to_string().contains("logits argmax length mismatch"));
    }

    #[test]
    fn logits_sample_topk_topp_uses_cpu_reference_when_cuda_reference_is_not_enabled() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let output = logits_sample_topk_topp(
            &[
                3.0, 2.0, 1.0, 0.0, //
                0.0, 1.0, 3.0, 2.0, //
            ],
            &[0.95, 0.0],
            2,
            4,
            1.0,
            3,
            0.8,
        )
        .unwrap();

        assert_eq!(output.indices, vec![1, 2]);
        assert_eq!(output.scores.len(), 2);
        assert!(output.scores.iter().all(|score| score.is_finite()));
        assert_eq!(
            output.backend,
            CPU_REFERENCE_LOGITS_SAMPLE_TOPK_TOPP_BACKEND
        );
    }

    #[test]
    fn logits_sample_topk_topp_rejects_shape_mismatch_before_backend_selection() {
        let err = logits_sample_topk_topp(&[1.0, 2.0], &[0.0], 1, 3, 1.0, 1, 1.0).unwrap_err();

        assert!(err
            .to_string()
            .contains("logits top-k/top-p sampler length mismatch"));
    }

    #[test]
    fn lm_head_argmax_bf16_uses_cpu_reference_when_cuda_reference_is_not_enabled() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let hidden = bf16_bytes(&[
            1.0, -2.0, //
            -1.0, 0.5, //
        ]);
        let lm_head = bf16_bytes(&[
            1.0, 0.0, //
            0.0, -1.0, //
            -1.0, 1.0, //
        ]);
        let output = lm_head_argmax_bf16(&hidden, &lm_head, 2, 2, 3).unwrap();

        assert_eq!(output.indices, vec![1, 2]);
        assert_eq!(output.scores, vec![2.0, 1.5]);
        assert_eq!(output.backend, CPU_REFERENCE_LM_HEAD_ARGMAX_BF16_BACKEND);
    }

    #[test]
    fn lm_head_sample_topk_topp_bf16_uses_cpu_reference_when_cuda_reference_is_not_enabled() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let hidden = bf16_bytes(&[
            1.0, -2.0, //
            -1.0, 0.5, //
        ]);
        let lm_head = bf16_bytes(&[
            1.0, 0.0, //
            0.0, -1.0, //
            -1.0, 1.0, //
            0.5, 0.5, //
        ]);
        let output =
            lm_head_sample_topk_topp_bf16(&hidden, &lm_head, &[0.0, 0.95], 2, 2, 4, 1.0, 3, 0.8)
                .unwrap();

        assert_eq!(output.indices.len(), 2);
        assert!(output.scores.iter().all(|score| score.is_finite()));
        assert_eq!(
            output.backend,
            CPU_REFERENCE_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_BACKEND
        );
    }

    #[test]
    fn lm_head_sample_topk_topp_bf16_rejects_shape_mismatch_before_backend_selection() {
        let hidden = bf16_bytes(&[1.0, -2.0]);
        let lm_head = bf16_bytes(&[1.0, 0.0]);
        let err = lm_head_sample_topk_topp_bf16(&hidden, &lm_head, &[0.0], 1, 2, 2, 1.0, 1, 1.0)
            .unwrap_err();

        assert!(err
            .to_string()
            .contains("BF16 lm_head scorer weight byte length mismatch"));
    }

    #[test]
    fn lm_head_argmax_bf16_resident_weight_uses_cpu_reference_when_cuda_reference_is_not_enabled() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let hidden = bf16_bytes(&[
            1.0, -2.0, //
            -1.0, 0.5, //
        ]);
        let lm_head = bf16_bytes(&[
            1.0, 0.0, //
            0.0, -1.0, //
            -1.0, 1.0, //
        ]);
        let output = lm_head_argmax_bf16_resident_weight(
            "lm_head.weight[rows=0..3]",
            &hidden,
            &lm_head,
            2,
            2,
            3,
        )
        .unwrap();

        assert_eq!(output.indices, vec![1, 2]);
        assert_eq!(output.scores, vec![2.0, 1.5]);
        assert_eq!(output.backend, CPU_REFERENCE_LM_HEAD_ARGMAX_BF16_BACKEND);
    }

    #[test]
    fn lm_head_sample_topk_topp_bf16_resident_weight_uses_cpu_reference_when_cuda_reference_is_not_enabled(
    ) {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let hidden = bf16_bytes(&[
            1.0, -2.0, //
            -1.0, 0.5, //
        ]);
        let lm_head = bf16_bytes(&[
            1.0, 0.0, //
            0.0, -1.0, //
            -1.0, 1.0, //
            0.5, 0.5, //
        ]);
        let output = lm_head_sample_topk_topp_bf16_resident_weight(
            "lm_head.weight[rows=0..4]",
            &hidden,
            &lm_head,
            &[0.0, 0.95],
            2,
            2,
            4,
            1.0,
            3,
            0.8,
        )
        .unwrap();

        assert_eq!(output.indices.len(), 2);
        assert!(output.scores.iter().all(|score| score.is_finite()));
        assert_eq!(
            output.backend,
            CPU_REFERENCE_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_BACKEND
        );
    }

    #[test]
    fn lm_head_argmax_bf16_preloaded_resident_weight_requires_cuda_reference() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let hidden = bf16_bytes(&[1.0, -2.0]);
        let err =
            lm_head_argmax_bf16_preloaded_resident_weight("lm_head.weight", &hidden, 1, 2, 8, 2, 3)
                .unwrap_err();

        assert!(err
            .to_string()
            .contains("preloaded resident BF16 lm_head argmax requires"));
    }

    #[test]
    fn lm_head_preloaded_resident_weight_rejects_chunk_past_full_vocab() {
        let hidden = bf16_bytes(&[1.0, -2.0]);
        let err =
            lm_head_argmax_bf16_preloaded_resident_weight("lm_head.weight", &hidden, 1, 2, 4, 2, 3)
                .unwrap_err();

        assert!(err
            .to_string()
            .contains("chunk [2, 5) exceeds full vocab 4"));
    }

    #[test]
    fn lm_head_sample_topk_topp_bf16_preloaded_resident_weight_requires_cuda_reference() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let hidden = bf16_bytes(&[1.0, -2.0]);
        let err = lm_head_sample_topk_topp_bf16_preloaded_resident_weight(
            "lm_head.weight",
            &hidden,
            &[0.0],
            1,
            2,
            8,
            2,
            3,
            1.0,
            1,
            1.0,
        )
        .unwrap_err();

        assert!(err
            .to_string()
            .contains("preloaded resident BF16 lm_head sampler requires"));
    }

    #[test]
    fn lm_head_device_input_readbacks_reuse_thread_scratch_when_cuda_available() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);

        let result = (|| -> Result<()> {
            let weight_name = format!(
                "test.lm_head.device-input-readback-scratch.weight.{}.{}",
                std::process::id(),
                line!()
            );
            let lm_head = bf16_bytes(&[
                1.0_f32, 0.0, //
                0.0, 2.0, //
                3.0, 0.0, //
                2.0, 0.0, //
            ]);
            match preload_resident_weight_from_host_staging(
                &weight_name,
                lm_head.len(),
                "test lm_head device-input readback scratch weight",
                |staging| {
                    staging.copy_from_slice(&lm_head);
                    Ok(())
                },
            ) {
                Ok(()) => {}
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            }

            let hidden = match device_bf16_output_from_f32_values(
                &[1.0_f32, 0.0],
                1,
                2,
                "test lm_head device-input readback scratch hidden",
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            let graph_stats_before = coordinator_cuda_graph_test_stats()?;

            let argmax = lm_head_argmax_bf16_preloaded_resident_weight_device_input(
                &weight_name,
                &hidden,
                4,
                0,
                4,
            )?;
            let sampler = lm_head_sample_topk_topp_bf16_preloaded_resident_weight_device_input(
                &weight_name,
                &hidden,
                &[0.5],
                4,
                0,
                4,
                0.7,
                4,
                0.95,
            )?;

            assert_eq!(argmax.indices, vec![2]);
            assert_eq!(argmax.scores, vec![3.0]);
            assert_eq!(sampler.indices.len(), 1);
            assert!(sampler.indices[0] < 4);
            assert!(sampler.scores[0].is_finite());

            let first = lm_head_device_input_readback_scratch_state();
            assert!(first.argmax_index_capacity >= std::mem::size_of::<u32>());
            assert!(first.argmax_score_capacity >= std::mem::size_of::<f32>());
            assert!(first.sample_index_capacity >= std::mem::size_of::<u32>());
            assert!(first.sample_score_capacity >= std::mem::size_of::<f32>());
            let graph_stats_after_first = coordinator_cuda_graph_test_stats()?;
            assert!(
                graph_stats_after_first.graph_captures >= graph_stats_before.graph_captures + 2,
                "terminal lm_head argmax and sampler should capture retained graphs: before={graph_stats_before:?} after={graph_stats_after_first:?}"
            );
            assert!(
                graph_stats_after_first.graph_launches >= graph_stats_before.graph_launches + 2,
                "terminal lm_head argmax and sampler should launch retained graphs: before={graph_stats_before:?} after={graph_stats_after_first:?}"
            );

            let argmax_again = lm_head_argmax_bf16_preloaded_resident_weight_device_input(
                &weight_name,
                &hidden,
                4,
                0,
                4,
            )?;
            let sampler_again =
                lm_head_sample_topk_topp_bf16_preloaded_resident_weight_device_input(
                    &weight_name,
                    &hidden,
                    &[0.5],
                    4,
                    0,
                    4,
                    0.7,
                    4,
                    0.95,
                )?;
            let second = lm_head_device_input_readback_scratch_state();
            let graph_stats_after_second = coordinator_cuda_graph_test_stats()?;

            assert_eq!(argmax_again.indices, argmax.indices);
            assert_eq!(argmax_again.scores, argmax.scores);
            assert_eq!(sampler_again.indices, sampler.indices);
            assert_eq!(sampler_again.scores, sampler.scores);
            assert_eq!(second, first);
            assert_eq!(
                graph_stats_after_second.graph_captures, graph_stats_after_first.graph_captures,
                "second terminal lm_head pass should reuse captured graphs"
            );
            assert!(
                graph_stats_after_second.graph_launches
                    >= graph_stats_after_first.graph_launches + 2,
                "second terminal lm_head pass should replay retained graphs: first={graph_stats_after_first:?} second={graph_stats_after_second:?}"
            );
            Ok(())
        })();

        result
    }

    #[test]
    fn lm_head_sample_topk_topp_bf16_preloaded_device_input_uses_triton_graph_when_python_enabled(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _python_capture_guard = coordinator_python_capture_test_override(true);
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);

        let weight_name = format!(
            "test.lm_head.triton-sampler.weight.{}.{}",
            std::process::id(),
            line!()
        );
        let lm_head = bf16_bytes(&[
            1.0_f32, 0.0, //
            0.0, 2.0, //
            3.0, 0.0, //
            2.0, 0.0, //
        ]);
        match preload_resident_weight_from_host_staging(
            &weight_name,
            lm_head.len(),
            "test Triton lm_head sampler weight",
            |staging| {
                staging.copy_from_slice(&lm_head);
                Ok(())
            },
        ) {
            Ok(()) => {}
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        }

        let hidden_values = [1.0_f32, 0.0];
        let hidden_bf16 = bf16_bytes(&hidden_values);
        let hidden = match device_bf16_output_from_f32_values(
            &hidden_values,
            1,
            2,
            "test Triton lm_head sampler hidden",
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let random_uniforms = [0.5_f32];
        let expected = cpu_lm_head_sample_topk_topp_bf16(
            &hidden_bf16,
            &lm_head,
            &random_uniforms,
            1,
            2,
            4,
            0.7,
            4,
            0.95,
        );

        let combined =
            match lm_head_argmax_sample_topk_topp_bf16_preloaded_resident_weight_device_input(
                &weight_name,
                &hidden,
                &random_uniforms,
                4,
                0,
                4,
                0.7,
                4,
                0.95,
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
        let output = combined.sampler;
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordDense,
            LayerWaveMode::Decode,
            1,
        )?;
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Dense decode graph key is registered");
        let has_triton_graph = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            slot.captured_graphs.iter().any(|entry| {
                entry.program == CoordinatorCudaGraphProgram::TerminalTritonLmHeadSampleTopKToppBf16
            })
        };

        assert_eq!(
            output.backend,
            TRITON_LM_HEAD_SAMPLE_TOPK_TOPP_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND
        );
        assert_eq!(combined.argmax.indices, vec![2]);
        assert_eq!(combined.argmax.scores, vec![3.0]);
        assert_eq!(output.indices, expected.indices);
        assert_eq!(output.scores.len(), expected.scores.len());
        for (actual, expected) in output.scores.iter().zip(expected.scores.iter()) {
            assert!((actual - expected).abs() < 2.0e-3);
        }
        assert!(has_triton_graph);
        Ok(())
    }

    #[test]
    fn lm_head_resident_weight_wrappers_reject_empty_weight_name() {
        let hidden = bf16_bytes(&[1.0, -2.0]);
        let lm_head = bf16_bytes(&[1.0, 0.0]);
        let argmax_err =
            lm_head_argmax_bf16_resident_weight("", &hidden, &lm_head, 1, 2, 1).unwrap_err();
        let sampler_err = lm_head_sample_topk_topp_bf16_resident_weight(
            "",
            &hidden,
            &lm_head,
            &[0.0],
            1,
            2,
            1,
            1.0,
            1,
            1.0,
        )
        .unwrap_err();

        assert!(argmax_err
            .to_string()
            .contains("weight name must not be empty"));
        assert!(sampler_err
            .to_string()
            .contains("weight name must not be empty"));
    }

    #[test]
    fn causal_attention_rows_uses_cpu_reference_when_cuda_reference_is_not_enabled() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let output = causal_attention_rows(
            &[
                1.0, 0.0, //
                0.0, 1.0, //
            ],
            &[
                1.0, 0.0, //
                0.0, 1.0, //
            ],
            &[
                2.0, 0.0, //
                0.0, 4.0, //
            ],
            2,
            1,
            2,
            2,
            1.0,
        )
        .unwrap();

        assert_eq!(output.values.len(), 4);
        assert_eq!(output.values[0], 2.0);
        assert!(output.values[1].abs() < 1.0e-6);
        assert!(output.values[2] > 0.0);
        assert!(output.values[3] > 2.0);
        assert_eq!(output.backend, CPU_REFERENCE_CAUSAL_ATTENTION_BACKEND);
    }

    #[test]
    fn causal_attention_rows_rejects_shape_mismatch_before_backend_selection() {
        let err =
            causal_attention_rows(&[1.0, 0.0], &[1.0, 0.0], &[1.0], 1, 1, 2, 2, 1.0).unwrap_err();

        assert!(err
            .to_string()
            .contains("causal attention value length mismatch"));
    }

    #[test]
    fn causal_attention_rows_bf16_uses_cpu_reference_when_cuda_reference_is_not_enabled() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let q = bf16_bytes(&[1.0, 0.0, 0.0, 1.0]);
        let k = bf16_bytes(&[1.0, 0.0, 0.0, 1.0]);
        let v = bf16_bytes(&[2.0, 0.0, 0.0, 4.0]);
        let output = causal_attention_rows_bf16(&q, &k, &v, 2, 1, 2, 2, 1.0).unwrap();

        assert_eq!(output.values.len(), 4);
        assert_eq!(output.values[0], 2.0);
        assert!(output.values[1].abs() < 1.0e-6);
        assert!(output.values.iter().all(|value| value.is_finite()));
        assert_eq!(output.backend, CPU_REFERENCE_CAUSAL_ATTENTION_BF16_BACKEND);
    }

    #[test]
    fn causal_attention_rows_bf16_dense_layer_uses_coord_dense_graph_slot() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordDense,
            LayerWaveMode::Decode,
            1,
        )?;
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Dense decode graph key is registered");
        let (acquisitions_before, graph_captures_before, graph_launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_captures, slot.graph_launches)
        };

        let q = bf16_bytes(&[1.0, 0.0]);
        let k = bf16_bytes(&[1.0, 0.0]);
        let v = bf16_bytes(&[2.0, -1.0]);
        let output = match cuda_causal_attention_rows_bf16_for_layer(0, &q, &k, &v, 1, 1, 2, 2, 1.0)
        {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let slot = registry.slots[graph_index]
            .lock()
            .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;

        assert_eq!(output.backend, CUDA_REFERENCE_CAUSAL_ATTENTION_BF16_BACKEND);
        assert_eq!(output.values, vec![2.0, -1.0]);
        assert!(slot.acquisitions > acquisitions_before);
        assert!(slot.graph_captures >= graph_captures_before);
        assert!(slot.graph_launches > graph_launches_before);
        assert!(slot.has_captured_graph(
            CoordinatorCudaGraphProgram::LayerCausalAttentionBf16,
            CoordinatorCudaGraphSignature::causal_attention_bf16(v.len(), 1, 1, 2, 2, 1.0)
        ));
        Ok(())
    }

    #[test]
    fn causal_attention_rows_bf16_sparse_layer_uses_coord_sparse_a_graph_slot() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseA,
            LayerWaveMode::Decode,
            1,
        )?;
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Sparse-A decode graph key is registered");
        let (acquisitions_before, graph_captures_before, graph_launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_captures, slot.graph_launches)
        };

        let q = bf16_bytes(&[1.0, 0.0]);
        let k = bf16_bytes(&[1.0, 0.0]);
        let v = bf16_bytes(&[2.0, -1.0]);
        let output = match cuda_causal_attention_rows_bf16_for_layer(3, &q, &k, &v, 1, 1, 2, 2, 1.0)
        {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let slot = registry.slots[graph_index]
            .lock()
            .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;

        assert_eq!(output.backend, CUDA_REFERENCE_CAUSAL_ATTENTION_BF16_BACKEND);
        assert_eq!(output.values, vec![2.0, -1.0]);
        assert!(slot.acquisitions > acquisitions_before);
        assert!(slot.graph_captures >= graph_captures_before);
        assert!(slot.graph_launches > graph_launches_before);
        assert!(slot.has_captured_graph(
            CoordinatorCudaGraphProgram::LayerCausalAttentionBf16,
            CoordinatorCudaGraphSignature::causal_attention_bf16(v.len(), 1, 1, 2, 2, 1.0)
        ));
        Ok(())
    }

    #[test]
    fn causal_attention_rows_bf16_layer_graph_replays_with_updated_node_params() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);

        let result = (|| -> Result<()> {
            let rows = 2;
            let heads = 1;
            let qk_dim = 2;
            let v_dim = 2;
            let scale = 0.5_f32;
            let graph_key = CoordinatorGraphKey::glm52_bf16(
                CoordinatorGraphShape::CoordSparseA,
                LayerWaveMode::Prefill,
                rows,
            )?;
            let registry = match coordinator_cuda_graph_workspace_registry() {
                Ok(registry) => registry,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            let graph_index = registry
                .keys()
                .iter()
                .position(|key| key == &graph_key)
                .expect("Coord-Sparse-A prefill graph key is registered");
            let (graph_captures_before, graph_launches_before) = {
                let slot = registry.slots[graph_index]
                    .lock()
                    .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
                (slot.graph_captures, slot.graph_launches)
            };

            let q0 = bf16_bytes(&[1.0, 0.0, 0.25, -0.5]);
            let k0 = bf16_bytes(&[1.0, 0.0, 0.5, 0.25]);
            let v0 = bf16_bytes(&[2.0, -1.0, 0.5, 1.5]);
            let expected0 = bf16_values_to_f32(&f32_values_to_bf16_bytes(
                &cpu_causal_attention_rows_bf16(&q0, &k0, &v0, rows, heads, qk_dim, v_dim, scale)
                    .values,
            ));
            let output0 = match cuda_causal_attention_rows_bf16_for_layer(
                3, &q0, &k0, &v0, rows, heads, qk_dim, v_dim, scale,
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            assert_eq!(
                output0.backend,
                CUDA_REFERENCE_CAUSAL_ATTENTION_BF16_BACKEND
            );
            for (actual, expected) in output0.values.iter().zip(expected0.iter()) {
                assert!((actual - expected).abs() < 1.0e-5);
            }

            let q1 = bf16_bytes(&[-0.75, 0.5, 0.25, 1.0]);
            let k1 = bf16_bytes(&[-0.5, 0.25, 0.75, -0.25]);
            let v1 = bf16_bytes(&[-1.5, 0.25, 2.0, -0.5]);
            let expected1 = bf16_values_to_f32(&f32_values_to_bf16_bytes(
                &cpu_causal_attention_rows_bf16(&q1, &k1, &v1, rows, heads, qk_dim, v_dim, scale)
                    .values,
            ));
            let output1 = match cuda_causal_attention_rows_bf16_for_layer(
                3, &q1, &k1, &v1, rows, heads, qk_dim, v_dim, scale,
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            assert_eq!(
                output1.backend,
                CUDA_REFERENCE_CAUSAL_ATTENTION_BF16_BACKEND
            );
            for (actual, expected) in output1.values.iter().zip(expected1.iter()) {
                assert!((actual - expected).abs() < 1.0e-5);
            }
            assert_ne!(output0.values, output1.values);

            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            assert!(slot.has_captured_graph(
                CoordinatorCudaGraphProgram::LayerCausalAttentionBf16,
                causal_attention_graph_signature(&graph_key, heads, qk_dim, v_dim, scale)
            ));
            assert!(slot.graph_captures >= graph_captures_before);
            assert!(slot.graph_launches >= graph_launches_before + 2);
            Ok(())
        })();

        result
    }

    #[test]
    fn causal_attention_rows_bf16_coord_sparse_a_graph_replays_same_bucket_when_rows_change(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);

        let result = (|| -> Result<()> {
            let rows_first = 2;
            let rows_second = 4;
            let heads = 1;
            let qk_dim = 2;
            let v_dim = 2;
            let scale = 0.5_f32;
            let graph_key = CoordinatorGraphKey::glm52_bf16(
                CoordinatorGraphShape::CoordSparseA,
                LayerWaveMode::Prefill,
                rows_first,
            )?;
            assert_eq!(
                graph_key,
                CoordinatorGraphKey::glm52_bf16(
                    CoordinatorGraphShape::CoordSparseA,
                    LayerWaveMode::Prefill,
                    rows_second,
                )?
            );
            let registry = match coordinator_cuda_graph_workspace_registry() {
                Ok(registry) => registry,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            let graph_index = registry
                .keys()
                .iter()
                .position(|key| key == &graph_key)
                .expect("Coord-Sparse-A prefill graph key is registered");
            let (graph_captures_before, graph_launches_before) = {
                let slot = registry.slots[graph_index]
                    .lock()
                    .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
                (slot.graph_captures, slot.graph_launches)
            };

            let q0 = bf16_bytes(&[1.0, 0.0, 0.25, -0.5]);
            let k0 = bf16_bytes(&[1.0, 0.0, 0.5, 0.25]);
            let v0 = bf16_bytes(&[2.0, -1.0, 0.5, 1.5]);
            let expected0 = bf16_values_to_f32(&f32_values_to_bf16_bytes(
                &cpu_causal_attention_rows_bf16(
                    &q0, &k0, &v0, rows_first, heads, qk_dim, v_dim, scale,
                )
                .values,
            ));
            let output0 = match cuda_causal_attention_rows_bf16_for_layer(
                3, &q0, &k0, &v0, rows_first, heads, qk_dim, v_dim, scale,
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            assert_eq!(
                output0.backend,
                CUDA_REFERENCE_CAUSAL_ATTENTION_BF16_BACKEND
            );
            for (actual, expected) in output0.values.iter().zip(expected0.iter()) {
                assert!((actual - expected).abs() < 1.0e-5);
            }

            let (graph_captures_after_first, graph_launches_after_first) = {
                let slot = registry.slots[graph_index]
                    .lock()
                    .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
                assert!(slot.has_captured_graph(
                    CoordinatorCudaGraphProgram::LayerCausalAttentionBf16,
                    causal_attention_graph_signature(&graph_key, heads, qk_dim, v_dim, scale)
                ));
                (slot.graph_captures, slot.graph_launches)
            };

            let q1 = bf16_bytes(&[
                -0.75, 0.5, 0.25, 1.0, //
                1.5, -0.25, -0.5, 0.75, //
            ]);
            let k1 = bf16_bytes(&[
                -0.5, 0.25, 0.75, -0.25, //
                1.0, 0.5, -0.75, 0.25, //
            ]);
            let v1 = bf16_bytes(&[
                -1.5, 0.25, 2.0, -0.5, //
                0.75, 1.25, -0.25, -1.0, //
            ]);
            let expected1 = bf16_values_to_f32(&f32_values_to_bf16_bytes(
                &cpu_causal_attention_rows_bf16(
                    &q1,
                    &k1,
                    &v1,
                    rows_second,
                    heads,
                    qk_dim,
                    v_dim,
                    scale,
                )
                .values,
            ));
            let output1 = match cuda_causal_attention_rows_bf16_for_layer(
                3,
                &q1,
                &k1,
                &v1,
                rows_second,
                heads,
                qk_dim,
                v_dim,
                scale,
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            assert_eq!(
                output1.backend,
                CUDA_REFERENCE_CAUSAL_ATTENTION_BF16_BACKEND
            );
            for (actual, expected) in output1.values.iter().zip(expected1.iter()) {
                assert!((actual - expected).abs() < 1.0e-5);
            }

            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            assert!(slot.has_captured_graph(
                CoordinatorCudaGraphProgram::LayerCausalAttentionBf16,
                causal_attention_graph_signature(&graph_key, heads, qk_dim, v_dim, scale)
            ));
            assert!(slot.graph_captures >= graph_captures_before);
            assert_eq!(slot.graph_captures, graph_captures_after_first);
            assert!(slot.graph_launches >= graph_launches_before + 2);
            assert!(slot.graph_launches >= graph_launches_after_first + 1);
            Ok(())
        })();

        result
    }

    #[test]
    fn causal_attention_rows_bf16_rejects_shape_mismatch_before_backend_selection() {
        let q = bf16_bytes(&[1.0, 0.0]);
        let k = bf16_bytes(&[1.0, 0.0]);
        let v = bf16_bytes(&[1.0]);
        let err = causal_attention_rows_bf16(&q, &k, &v, 1, 1, 2, 2, 1.0).unwrap_err();

        assert!(err
            .to_string()
            .contains("BF16 causal attention value byte length mismatch"));
    }

    #[test]
    fn rope_rows_uses_cpu_reference_when_cuda_reference_is_not_enabled() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let output = rope_rows(
            &[
                1.0, 0.0, 0.0, 1.0, //
                0.5, -0.5, 2.0, 0.0, //
            ],
            &[0, 1],
            2,
            1,
            4,
            10_000.0,
        )
        .unwrap();

        assert_eq!(output.values[0], 1.0);
        assert_eq!(output.values[1], 0.0);
        assert_ne!(output.values[4], 0.5);
        assert_eq!(output.backend, CPU_REFERENCE_ROPE_BACKEND);
    }

    #[test]
    fn rope_rows_rejects_odd_rotary_dim_before_backend_selection() {
        let err = rope_rows(&[1.0, 0.0, 0.5], &[0], 1, 1, 3, 10_000.0).unwrap_err();

        assert!(err.to_string().contains("positive even rotary_dim"));
    }

    #[test]
    fn rope_rows_bf16_uses_cpu_reference_when_cuda_reference_is_not_enabled() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let input = bf16_bytes(&[
            1.0, 0.0, 0.0, 1.0, //
            0.5, -0.5, 2.0, 0.0, //
        ]);
        let output = rope_rows_bf16(&input, &[0, 1], 2, 1, 4, 10_000.0).unwrap();

        assert_eq!(output.values[0], 1.0);
        assert_eq!(output.values[1], 0.0);
        assert_ne!(output.values[4], 0.5);
        assert_eq!(output.backend, CPU_REFERENCE_ROPE_BF16_BACKEND);
    }

    #[test]
    fn rope_rows_bf16_rejects_shape_mismatch_before_backend_selection() {
        let input = bf16_bytes(&[1.0, 0.0, 0.5]);
        let err = rope_rows_bf16(&input, &[0], 1, 1, 4, 10_000.0).unwrap_err();

        assert!(err
            .to_string()
            .contains("BF16 RoPE input byte length mismatch"));
    }

    #[test]
    fn rope_rows_bf16_dense_layer_uses_coord_dense_graph_slot() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordDense,
            LayerWaveMode::Decode,
            1,
        )?;
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Dense decode graph key is registered");
        let (acquisitions_before, graph_captures_before, graph_launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_captures, slot.graph_launches)
        };

        let input = bf16_bytes(&[1.0, 0.0, 0.0, 1.0]);
        let output = match cuda_rope_rows_bf16_for_layer(0, &input, &[1], 1, 1, 4, 10_000.0) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let slot = registry.slots[graph_index]
            .lock()
            .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;

        assert_eq!(output.backend, CUDA_REFERENCE_ROPE_BF16_BACKEND);
        assert_eq!(output.values.len(), 4);
        assert!(output.values.iter().all(|value| value.is_finite()));
        assert!(slot.acquisitions > acquisitions_before);
        assert!(slot.graph_captures >= graph_captures_before);
        assert!(slot.graph_launches > graph_launches_before);
        assert!(slot.has_captured_graph(
            CoordinatorCudaGraphProgram::LayerRopeBf16,
            CoordinatorCudaGraphSignature::rope_bf16(input.len(), 1, 1, 4, 10_000.0)
        ));
        Ok(())
    }

    #[test]
    fn rope_rows_bf16_sparse_layer_uses_coord_sparse_a_graph_slot() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseA,
            LayerWaveMode::Decode,
            1,
        )?;
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Sparse-A decode graph key is registered");
        let (acquisitions_before, graph_captures_before, graph_launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_captures, slot.graph_launches)
        };

        let input = bf16_bytes(&[1.0, 0.0, 0.0, 1.0]);
        let output = match cuda_rope_rows_bf16_for_layer(3, &input, &[1], 1, 1, 4, 10_000.0) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let slot = registry.slots[graph_index]
            .lock()
            .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;

        assert_eq!(output.backend, CUDA_REFERENCE_ROPE_BF16_BACKEND);
        assert_eq!(output.values.len(), 4);
        assert!(output.values.iter().all(|value| value.is_finite()));
        assert!(slot.acquisitions > acquisitions_before);
        assert!(slot.graph_captures >= graph_captures_before);
        assert!(slot.graph_launches > graph_launches_before);
        assert!(slot.has_captured_graph(
            CoordinatorCudaGraphProgram::LayerRopeBf16,
            CoordinatorCudaGraphSignature::rope_bf16(input.len(), 1, 1, 4, 10_000.0)
        ));
        Ok(())
    }

    #[test]
    fn rope_rows_bf16_layer_graph_replays_with_updated_node_params() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);

        let result = (|| -> Result<()> {
            let rows = 2;
            let heads = 1;
            let rotary_dim = 4;
            let theta = 10_000.0_f32;
            let graph_key = CoordinatorGraphKey::glm52_bf16(
                CoordinatorGraphShape::CoordSparseA,
                LayerWaveMode::Prefill,
                rows,
            )?;
            let registry = match coordinator_cuda_graph_workspace_registry() {
                Ok(registry) => registry,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            let graph_index = registry
                .keys()
                .iter()
                .position(|key| key == &graph_key)
                .expect("Coord-Sparse-A prefill graph key is registered");
            let (graph_captures_before, graph_launches_before) = {
                let slot = registry.slots[graph_index]
                    .lock()
                    .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
                (slot.graph_captures, slot.graph_launches)
            };

            let input0 = bf16_bytes(&[1.0, 0.0, 0.0, 1.0, 0.5, -0.5, 2.0, 0.0]);
            let positions0 = [0_u32, 1];
            let expected0 = bf16_values_to_f32(&f32_values_to_bf16_bytes(
                &cpu_rope_rows_bf16(&input0, &positions0, rows, heads, rotary_dim, theta).values,
            ));
            let output0 = match cuda_rope_rows_bf16_for_layer(
                3,
                &input0,
                &positions0,
                rows,
                heads,
                rotary_dim,
                theta,
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            assert_eq!(output0.backend, CUDA_REFERENCE_ROPE_BF16_BACKEND);
            for (actual, expected) in output0.values.iter().zip(expected0.iter()) {
                assert!((actual - expected).abs() < 1.0e-5);
            }

            let input1 = bf16_bytes(&[-0.25, 0.75, 1.5, -0.5, 0.0, 1.0, -1.0, 0.5]);
            let positions1 = [2_u32, 3];
            let expected1 = bf16_values_to_f32(&f32_values_to_bf16_bytes(
                &cpu_rope_rows_bf16(&input1, &positions1, rows, heads, rotary_dim, theta).values,
            ));
            let output1 = match cuda_rope_rows_bf16_for_layer(
                3,
                &input1,
                &positions1,
                rows,
                heads,
                rotary_dim,
                theta,
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            assert_eq!(output1.backend, CUDA_REFERENCE_ROPE_BF16_BACKEND);
            for (actual, expected) in output1.values.iter().zip(expected1.iter()) {
                assert!((actual - expected).abs() < 1.0e-5);
            }
            assert_ne!(output0.values, output1.values);

            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            assert!(slot.has_captured_graph(
                CoordinatorCudaGraphProgram::LayerRopeBf16,
                rope_graph_signature(&graph_key, heads, rotary_dim, theta)
            ));
            assert!(slot.graph_captures >= graph_captures_before);
            assert!(slot.graph_launches >= graph_launches_before + 2);
            Ok(())
        })();

        result
    }

    #[test]
    fn rope_rows_bf16_coord_sparse_a_graph_replays_same_bucket_when_rows_change() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);

        let result = (|| -> Result<()> {
            let rows_first = 2;
            let rows_second = 4;
            let heads = 1;
            let rotary_dim = 4;
            let theta = 10_000.0_f32;
            let graph_key = CoordinatorGraphKey::glm52_bf16(
                CoordinatorGraphShape::CoordSparseA,
                LayerWaveMode::Prefill,
                rows_first,
            )?;
            assert_eq!(
                graph_key,
                CoordinatorGraphKey::glm52_bf16(
                    CoordinatorGraphShape::CoordSparseA,
                    LayerWaveMode::Prefill,
                    rows_second,
                )?
            );
            let registry = match coordinator_cuda_graph_workspace_registry() {
                Ok(registry) => registry,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            let graph_index = registry
                .keys()
                .iter()
                .position(|key| key == &graph_key)
                .expect("Coord-Sparse-A prefill graph key is registered");
            let (graph_captures_before, graph_launches_before) = {
                let slot = registry.slots[graph_index]
                    .lock()
                    .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
                (slot.graph_captures, slot.graph_launches)
            };

            let input0 = bf16_bytes(&[1.0, 0.0, 0.0, 1.0, 0.5, -0.5, 2.0, 0.0]);
            let positions0 = [0_u32, 1];
            let expected0 = bf16_values_to_f32(&f32_values_to_bf16_bytes(
                &cpu_rope_rows_bf16(&input0, &positions0, rows_first, heads, rotary_dim, theta)
                    .values,
            ));
            let output0 = match cuda_rope_rows_bf16_for_layer(
                3,
                &input0,
                &positions0,
                rows_first,
                heads,
                rotary_dim,
                theta,
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            assert_eq!(output0.backend, CUDA_REFERENCE_ROPE_BF16_BACKEND);
            for (actual, expected) in output0.values.iter().zip(expected0.iter()) {
                assert!((actual - expected).abs() < 1.0e-5);
            }

            let (graph_captures_after_first, graph_launches_after_first) = {
                let slot = registry.slots[graph_index]
                    .lock()
                    .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
                assert!(slot.has_captured_graph(
                    CoordinatorCudaGraphProgram::LayerRopeBf16,
                    rope_graph_signature(&graph_key, heads, rotary_dim, theta)
                ));
                (slot.graph_captures, slot.graph_launches)
            };

            let input1 = bf16_bytes(&[
                -0.25, 0.75, 1.5, -0.5, //
                0.0, 1.0, -1.0, 0.5, //
                2.0, -2.0, 0.25, 0.75, //
                -1.5, 0.25, 0.5, -0.75, //
            ]);
            let positions1 = [2_u32, 3, 4, 5];
            let expected1 = bf16_values_to_f32(&f32_values_to_bf16_bytes(
                &cpu_rope_rows_bf16(&input1, &positions1, rows_second, heads, rotary_dim, theta)
                    .values,
            ));
            let output1 = match cuda_rope_rows_bf16_for_layer(
                3,
                &input1,
                &positions1,
                rows_second,
                heads,
                rotary_dim,
                theta,
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            assert_eq!(output1.backend, CUDA_REFERENCE_ROPE_BF16_BACKEND);
            for (actual, expected) in output1.values.iter().zip(expected1.iter()) {
                assert!((actual - expected).abs() < 1.0e-5);
            }

            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            assert!(slot.has_captured_graph(
                CoordinatorCudaGraphProgram::LayerRopeBf16,
                rope_graph_signature(&graph_key, heads, rotary_dim, theta)
            ));
            assert!(slot.graph_captures >= graph_captures_before);
            assert_eq!(slot.graph_captures, graph_captures_after_first);
            assert!(slot.graph_launches >= graph_launches_before + 2);
            assert!(slot.graph_launches >= graph_launches_after_first + 1);
            Ok(())
        })();

        result
    }

    #[test]
    fn mla_rope_attention_rows_uses_cpu_reference_when_cuda_reference_is_not_enabled() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let output = mla_rope_attention_rows(
            &[0.10_f32, -0.20, 0.30, 0.05, -0.40, 0.25, 0.15, -0.35],
            &[-0.20_f32, 0.45, 0.15, -0.35, 0.30, -0.15, -0.55, 0.20],
            &[0.25_f32, 0.15, -0.10, 0.40, 0.35, -0.45, 0.60, 0.20],
            &[0.10_f32, 0.50, 0.35, -0.25],
            &[0.10_f32, 0.20, -0.40, -0.50, 0.70, 0.80, 1.00, -1.10],
            2,
            2,
            2,
            2,
            2,
            0.5,
        )
        .unwrap();

        assert_eq!(output.values.len(), 8);
        assert!((output.values[0] - 0.10).abs() < 1.0e-6);
        assert!((output.values[2] - -0.40).abs() < 1.0e-6);
        assert_eq!(output.backend, CPU_REFERENCE_MLA_ROPE_ATTENTION_BACKEND);
    }

    #[test]
    fn mla_rope_attention_rows_rejects_shape_mismatch_before_backend_selection() {
        let err = mla_rope_attention_rows(
            &[0.0, 1.0],
            &[0.0, 1.0],
            &[0.0, 1.0],
            &[0.0, 1.0],
            &[0.0],
            1,
            1,
            2,
            2,
            2,
            1.0,
        )
        .unwrap_err();

        assert!(err
            .to_string()
            .contains("MLA/RoPE attention value length mismatch"));
    }

    #[test]
    fn mla_rope_attention_rows_bf16_uses_cpu_reference_when_cuda_reference_is_not_enabled() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);

        let q_nope = bf16_bytes(&[1.0, 0.0]);
        let q_rope = bf16_bytes(&[0.0, 1.0]);
        let k_nope = bf16_bytes(&[1.0, 0.0]);
        let k_rope = bf16_bytes(&[0.0, 1.0]);
        let values = bf16_bytes(&[2.0, -1.0]);
        let output = mla_rope_attention_rows_bf16(
            &q_nope, &q_rope, &k_nope, &k_rope, &values, 1, 1, 2, 2, 2, 1.0,
        )
        .unwrap();

        assert_eq!(output.values, vec![2.0, -1.0]);
        assert_eq!(
            output.backend,
            CPU_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND
        );
    }

    #[test]
    fn mla_rope_attention_rows_bf16_dense_layer_uses_coord_dense_graph_slot() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordDense,
            LayerWaveMode::Decode,
            1,
        )?;
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Dense decode graph key is registered");
        let (acquisitions_before, graph_captures_before, graph_launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_captures, slot.graph_launches)
        };

        let q_nope = bf16_bytes(&[1.0, 0.0]);
        let q_rope = bf16_bytes(&[0.0, 1.0]);
        let k_nope = bf16_bytes(&[1.0, 0.0]);
        let k_rope = bf16_bytes(&[0.0, 1.0]);
        let values = bf16_bytes(&[2.0, -1.0]);
        let output = match cuda_mla_rope_attention_rows_bf16_for_layer(
            0, &q_nope, &q_rope, &k_nope, &k_rope, &values, 1, 1, 2, 2, 2, 1.0,
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let slot = registry.slots[graph_index]
            .lock()
            .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;

        assert_eq!(
            output.backend,
            CUDA_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND
        );
        assert_eq!(output.values, vec![2.0, -1.0]);
        assert!(slot.acquisitions > acquisitions_before);
        assert!(slot.graph_captures >= graph_captures_before);
        assert!(slot.graph_launches > graph_launches_before);
        assert!(slot.has_captured_graph(
            CoordinatorCudaGraphProgram::LayerMlaRopeAttentionBf16,
            CoordinatorCudaGraphSignature::mla_rope_attention_bf16(
                values.len(),
                1,
                1,
                2,
                2,
                2,
                1.0
            )
        ));
        Ok(())
    }

    #[test]
    fn mla_rope_attention_rows_bf16_sparse_layer_uses_coord_sparse_a_graph_slot() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let graph_key = CoordinatorGraphKey::glm52_bf16(
            CoordinatorGraphShape::CoordSparseA,
            LayerWaveMode::Decode,
            1,
        )?;
        let registry = match coordinator_cuda_graph_workspace_registry() {
            Ok(registry) => registry,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let graph_index = registry
            .keys()
            .iter()
            .position(|key| key == &graph_key)
            .expect("Coord-Sparse-A decode graph key is registered");
        let (acquisitions_before, graph_captures_before, graph_launches_before) = {
            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            (slot.acquisitions, slot.graph_captures, slot.graph_launches)
        };

        let q_nope = bf16_bytes(&[1.0, 0.0]);
        let q_rope = bf16_bytes(&[0.0, 1.0]);
        let k_nope = bf16_bytes(&[1.0, 0.0]);
        let k_rope = bf16_bytes(&[0.0, 1.0]);
        let values = bf16_bytes(&[2.0, -1.0]);
        let output = match cuda_mla_rope_attention_rows_bf16_for_layer(
            3, &q_nope, &q_rope, &k_nope, &k_rope, &values, 1, 1, 2, 2, 2, 1.0,
        ) {
            Ok(output) => output,
            Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
        let slot = registry.slots[graph_index]
            .lock()
            .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;

        assert_eq!(
            output.backend,
            CUDA_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND
        );
        assert_eq!(output.values, vec![2.0, -1.0]);
        assert!(slot.acquisitions > acquisitions_before);
        assert!(slot.graph_captures >= graph_captures_before);
        assert!(slot.graph_launches > graph_launches_before);
        assert!(slot.has_captured_graph(
            CoordinatorCudaGraphProgram::LayerMlaRopeAttentionBf16,
            CoordinatorCudaGraphSignature::mla_rope_attention_bf16(
                values.len(),
                1,
                1,
                2,
                2,
                2,
                1.0
            )
        ));
        Ok(())
    }

    #[test]
    fn mla_rope_attention_rows_bf16_layer_graph_replays_with_updated_node_params() -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);

        let result = (|| -> Result<()> {
            let rows = 2;
            let heads = 1;
            let nope_dim = 2;
            let rope_dim = 2;
            let v_dim = 2;
            let scale = 0.5_f32;
            let graph_key = CoordinatorGraphKey::glm52_bf16(
                CoordinatorGraphShape::CoordSparseA,
                LayerWaveMode::Prefill,
                rows,
            )?;
            let registry = match coordinator_cuda_graph_workspace_registry() {
                Ok(registry) => registry,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            let graph_index = registry
                .keys()
                .iter()
                .position(|key| key == &graph_key)
                .expect("Coord-Sparse-A prefill graph key is registered");
            let (graph_captures_before, graph_launches_before) = {
                let slot = registry.slots[graph_index]
                    .lock()
                    .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
                (slot.graph_captures, slot.graph_launches)
            };

            let q_nope0 = bf16_bytes(&[1.0, 0.0, 0.25, -0.5]);
            let q_rope0 = bf16_bytes(&[0.0, 1.0, -0.25, 0.5]);
            let k_nope0 = bf16_bytes(&[1.0, 0.0, 0.5, 0.25]);
            let k_rope0 = bf16_bytes(&[0.0, 1.0, 0.25, -0.25]);
            let values0 = bf16_bytes(&[2.0, -1.0, 0.5, 1.5]);
            let expected0 = bf16_values_to_f32(&f32_values_to_bf16_bytes(
                &cpu_mla_rope_attention_rows_bf16(
                    &q_nope0, &q_rope0, &k_nope0, &k_rope0, &values0, rows, heads, nope_dim,
                    rope_dim, v_dim, scale,
                )
                .values,
            ));
            let output0 = match cuda_mla_rope_attention_rows_bf16_for_layer(
                3, &q_nope0, &q_rope0, &k_nope0, &k_rope0, &values0, rows, heads, nope_dim,
                rope_dim, v_dim, scale,
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            assert_eq!(
                output0.backend,
                CUDA_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND
            );
            for (actual, expected) in output0.values.iter().zip(expected0.iter()) {
                assert!((actual - expected).abs() < 1.0e-5);
            }

            let q_nope1 = bf16_bytes(&[-0.75, 0.5, 0.25, 1.0]);
            let q_rope1 = bf16_bytes(&[0.5, -1.0, 1.25, 0.0]);
            let k_nope1 = bf16_bytes(&[-0.5, 0.25, 0.75, -0.25]);
            let k_rope1 = bf16_bytes(&[0.25, -0.5, 0.0, 1.0]);
            let values1 = bf16_bytes(&[-1.5, 0.25, 2.0, -0.5]);
            let expected1 = bf16_values_to_f32(&f32_values_to_bf16_bytes(
                &cpu_mla_rope_attention_rows_bf16(
                    &q_nope1, &q_rope1, &k_nope1, &k_rope1, &values1, rows, heads, nope_dim,
                    rope_dim, v_dim, scale,
                )
                .values,
            ));
            let output1 = match cuda_mla_rope_attention_rows_bf16_for_layer(
                3, &q_nope1, &q_rope1, &k_nope1, &k_rope1, &values1, rows, heads, nope_dim,
                rope_dim, v_dim, scale,
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            assert_eq!(
                output1.backend,
                CUDA_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND
            );
            for (actual, expected) in output1.values.iter().zip(expected1.iter()) {
                assert!((actual - expected).abs() < 1.0e-5);
            }
            assert_ne!(output0.values, output1.values);

            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            assert!(slot.has_captured_graph(
                CoordinatorCudaGraphProgram::LayerMlaRopeAttentionBf16,
                mla_rope_attention_graph_signature(
                    &graph_key, heads, nope_dim, rope_dim, v_dim, scale
                )
            ));
            assert!(slot.graph_captures >= graph_captures_before);
            assert!(slot.graph_launches >= graph_launches_before + 2);
            Ok(())
        })();

        result
    }

    #[test]
    fn mla_rope_attention_rows_bf16_coord_sparse_a_graph_replays_same_bucket_when_rows_change(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);

        let result = (|| -> Result<()> {
            let rows_first = 2;
            let rows_second = 4;
            let heads = 1;
            let nope_dim = 2;
            let rope_dim = 2;
            let v_dim = 2;
            let scale = 0.5_f32;
            let graph_key = CoordinatorGraphKey::glm52_bf16(
                CoordinatorGraphShape::CoordSparseA,
                LayerWaveMode::Prefill,
                rows_first,
            )?;
            assert_eq!(
                graph_key,
                CoordinatorGraphKey::glm52_bf16(
                    CoordinatorGraphShape::CoordSparseA,
                    LayerWaveMode::Prefill,
                    rows_second,
                )?
            );
            let registry = match coordinator_cuda_graph_workspace_registry() {
                Ok(registry) => registry,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            let graph_index = registry
                .keys()
                .iter()
                .position(|key| key == &graph_key)
                .expect("Coord-Sparse-A prefill graph key is registered");
            let (graph_captures_before, graph_launches_before) = {
                let slot = registry.slots[graph_index]
                    .lock()
                    .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
                (slot.graph_captures, slot.graph_launches)
            };

            let q_nope0 = bf16_bytes(&[1.0, 0.0, 0.25, -0.5]);
            let q_rope0 = bf16_bytes(&[0.0, 1.0, -0.25, 0.5]);
            let k_nope0 = bf16_bytes(&[1.0, 0.0, 0.5, 0.25]);
            let k_rope0 = bf16_bytes(&[0.0, 1.0, 0.25, -0.25]);
            let values0 = bf16_bytes(&[2.0, -1.0, 0.5, 1.5]);
            let expected0 = bf16_values_to_f32(&f32_values_to_bf16_bytes(
                &cpu_mla_rope_attention_rows_bf16(
                    &q_nope0, &q_rope0, &k_nope0, &k_rope0, &values0, rows_first, heads, nope_dim,
                    rope_dim, v_dim, scale,
                )
                .values,
            ));
            let output0 = match cuda_mla_rope_attention_rows_bf16_for_layer(
                3, &q_nope0, &q_rope0, &k_nope0, &k_rope0, &values0, rows_first, heads, nope_dim,
                rope_dim, v_dim, scale,
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            assert_eq!(
                output0.backend,
                CUDA_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND
            );
            for (actual, expected) in output0.values.iter().zip(expected0.iter()) {
                assert!((actual - expected).abs() < 1.0e-5);
            }

            let (graph_captures_after_first, graph_launches_after_first) = {
                let slot = registry.slots[graph_index]
                    .lock()
                    .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
                assert!(slot.has_captured_graph(
                    CoordinatorCudaGraphProgram::LayerMlaRopeAttentionBf16,
                    mla_rope_attention_graph_signature(
                        &graph_key, heads, nope_dim, rope_dim, v_dim, scale
                    )
                ));
                (slot.graph_captures, slot.graph_launches)
            };

            let q_nope1 = bf16_bytes(&[
                -0.75, 0.5, 0.25, 1.0, //
                1.5, -0.25, -0.5, 0.75, //
            ]);
            let q_rope1 = bf16_bytes(&[
                0.5, -1.0, 1.25, 0.0, //
                -0.25, 0.75, 0.5, -0.5, //
            ]);
            let k_nope1 = bf16_bytes(&[
                -0.5, 0.25, 0.75, -0.25, //
                1.0, 0.5, -0.75, 0.25, //
            ]);
            let k_rope1 = bf16_bytes(&[
                0.25, -0.5, 0.0, 1.0, //
                -1.25, 0.5, 0.75, -0.25, //
            ]);
            let values1 = bf16_bytes(&[
                -1.5, 0.25, 2.0, -0.5, //
                0.75, 1.25, -0.25, -1.0, //
            ]);
            let expected1 = bf16_values_to_f32(&f32_values_to_bf16_bytes(
                &cpu_mla_rope_attention_rows_bf16(
                    &q_nope1,
                    &q_rope1,
                    &k_nope1,
                    &k_rope1,
                    &values1,
                    rows_second,
                    heads,
                    nope_dim,
                    rope_dim,
                    v_dim,
                    scale,
                )
                .values,
            ));
            let output1 = match cuda_mla_rope_attention_rows_bf16_for_layer(
                3,
                &q_nope1,
                &q_rope1,
                &k_nope1,
                &k_rope1,
                &values1,
                rows_second,
                heads,
                nope_dim,
                rope_dim,
                v_dim,
                scale,
            ) {
                Ok(output) => output,
                Err(error) if cuda_allocation_unavailable(&error) => return Ok(()),
                Err(error) => return Err(error),
            };
            assert_eq!(
                output1.backend,
                CUDA_REFERENCE_MLA_ROPE_ATTENTION_BF16_BACKEND
            );
            for (actual, expected) in output1.values.iter().zip(expected1.iter()) {
                assert!((actual - expected).abs() < 1.0e-5);
            }

            let slot = registry.slots[graph_index]
                .lock()
                .map_err(|_| anyhow::anyhow!("test graph slot already borrowed"))?;
            assert!(slot.has_captured_graph(
                CoordinatorCudaGraphProgram::LayerMlaRopeAttentionBf16,
                mla_rope_attention_graph_signature(
                    &graph_key, heads, nope_dim, rope_dim, v_dim, scale
                )
            ));
            assert!(slot.graph_captures >= graph_captures_before);
            assert_eq!(slot.graph_captures, graph_captures_after_first);
            assert!(slot.graph_launches >= graph_launches_before + 2);
            assert!(slot.graph_launches >= graph_launches_after_first + 1);
            Ok(())
        })();

        result
    }

    #[test]
    fn b12x_mla_rope_attention_shape_gate_accepts_glm_nsa_tp8_contract() {
        assert!(b12x_mla_rope_attention_bf16_shape_supported(
            1,
            8,
            GLM52_MLA_KV_LORA_RANK,
            GLM52_MLA_QK_ROPE_HEAD_DIM,
            GLM52_MLA_KV_LORA_RANK,
        ));
        assert!(b12x_mla_rope_attention_bf16_shape_supported(
            512,
            8,
            GLM52_MLA_KV_LORA_RANK,
            GLM52_MLA_QK_ROPE_HEAD_DIM,
            GLM52_MLA_KV_LORA_RANK,
        ));
    }

    #[test]
    fn b12x_mla_rope_attention_shape_gate_rejects_phase0_split_debug_contract() {
        assert!(!b12x_mla_rope_attention_bf16_shape_supported(
            16, 64, 192, 64, 256,
        ));
        assert!(!b12x_mla_rope_attention_bf16_shape_supported(
            513,
            8,
            GLM52_MLA_KV_LORA_RANK,
            GLM52_MLA_QK_ROPE_HEAD_DIM,
            GLM52_MLA_KV_LORA_RANK,
        ));
    }

    #[test]
    fn b12x_mla_rope_attention_python_graph_dispatch_matches_cuda_reference_glm_nsa_shape(
    ) -> Result<()> {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _env_guard = B12X_MLA_TEST_ENV_MUTEX
            .lock()
            .expect("b12x MLA test env mutex poisoned");
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);
        install_b12x_mla_cuda_reference_capture_module()?;
        let _module_guard =
            EnvVarGuard::set("GLMRT_B12X_MLA_MODULE", "glmrt_test_b12x_mla_cuda_capture");
        let _function_guard = EnvVarGuard::set("GLMRT_B12X_MLA_FUNCTION", "capture");

        let result = (|| -> Result<()> {
            for rows in [1_usize, 16] {
                let heads = 8;
                let nope_dim = GLM52_MLA_KV_LORA_RANK;
                let rope_dim = GLM52_MLA_QK_ROPE_HEAD_DIM;
                let v_dim = GLM52_MLA_KV_LORA_RANK;
                let scale = 1.0 / ((nope_dim + rope_dim) as f32).sqrt();
                let q_nope = patterned_bf16_values(rows * heads * nope_dim, 0.001, 3.0);
                let q_rope = patterned_bf16_values(rows * heads * rope_dim, 0.001, 5.0);
                let k_nope = patterned_bf16_values(rows * heads * nope_dim, 0.001, 7.0);
                let k_rope = patterned_bf16_values(rows * rope_dim, 0.001, 11.0);
                let values = patterned_bf16_values(rows * heads * v_dim, 0.001, 13.0);
                let q_nope_bf16 = bf16_bytes(&q_nope);
                let q_rope_bf16 = bf16_bytes(&q_rope);
                let k_nope_bf16 = bf16_bytes(&k_nope);
                let k_rope_bf16 = bf16_bytes(&k_rope);
                let values_bf16 = bf16_bytes(&values);
                let expected = bf16_values_to_f32(&f32_values_to_bf16_bytes(
                    &cpu_mla_rope_attention_rows_bf16(
                        &q_nope_bf16,
                        &q_rope_bf16,
                        &k_nope_bf16,
                        &k_rope_bf16,
                        &values_bf16,
                        rows,
                        heads,
                        nope_dim,
                        rope_dim,
                        v_dim,
                        scale,
                    )
                    .values,
                ));
                let graph_key = coord_attention_graph_key_for_layer_rows(3, rows)?;
                let signature = mla_rope_attention_graph_signature(
                    &graph_key, heads, nope_dim, rope_dim, v_dim, scale,
                );
                let actual = with_coordinator_cuda_graph_slot(&graph_key, |library, slot| {
                    let cuda_stream = slot.stream_ptr();
                    let q_nope_buffer = slot.buffer(
                        library,
                        CoordinatorCudaScratchSlot::A,
                        q_nope_bf16.len(),
                        "test b12x q_nope",
                    )?;
                    let q_rope_buffer = slot.buffer(
                        library,
                        CoordinatorCudaScratchSlot::B,
                        q_rope_bf16.len(),
                        "test b12x q_rope",
                    )?;
                    let k_nope_buffer = slot.buffer(
                        library,
                        CoordinatorCudaScratchSlot::C,
                        k_nope_bf16.len(),
                        "test b12x k_nope",
                    )?;
                    let k_rope_buffer = slot.buffer(
                        library,
                        CoordinatorCudaScratchSlot::D,
                        k_rope_bf16.len(),
                        "test b12x k_rope",
                    )?;
                    let value_buffer = slot.buffer(
                        library,
                        CoordinatorCudaScratchSlot::E,
                        values_bf16.len(),
                        "test b12x values",
                    )?;
                    let output_buffer = slot.buffer(
                        library,
                        CoordinatorCudaScratchSlot::F,
                        values_bf16.len(),
                        "test b12x output",
                    )?;
                    slot.workspace
                        .copy_h2d_to_slot_async(
                            library,
                            CoordinatorCudaScratchSlot::A,
                            &q_nope_bf16,
                            "test b12x q_nope",
                            cuda_stream,
                        )
                        .context("copying test b12x q_nope to device")?;
                    slot.workspace
                        .copy_h2d_to_slot_async(
                            library,
                            CoordinatorCudaScratchSlot::B,
                            &q_rope_bf16,
                            "test b12x q_rope",
                            cuda_stream,
                        )
                        .context("copying test b12x q_rope to device")?;
                    slot.workspace
                        .copy_h2d_to_slot_async(
                            library,
                            CoordinatorCudaScratchSlot::C,
                            &k_nope_bf16,
                            "test b12x k_nope",
                            cuda_stream,
                        )
                        .context("copying test b12x k_nope to device")?;
                    slot.workspace
                        .copy_h2d_to_slot_async(
                            library,
                            CoordinatorCudaScratchSlot::D,
                            &k_rope_bf16,
                            "test b12x k_rope",
                            cuda_stream,
                        )
                        .context("copying test b12x k_rope to device")?;
                    slot.workspace
                        .copy_h2d_to_slot_async(
                            library,
                            CoordinatorCudaScratchSlot::E,
                            &values_bf16,
                            "test b12x values",
                            cuda_stream,
                        )
                        .context("copying test b12x values to device")?;
                    capture_or_update_layer_b12x_mla_rope_attention_bf16_graph(
                        library,
                        slot,
                        signature,
                        q_nope_buffer,
                        q_rope_buffer,
                        k_nope_buffer,
                        k_rope_buffer,
                        value_buffer,
                        output_buffer,
                        rows,
                        heads,
                        nope_dim,
                        rope_dim,
                        v_dim,
                        scale,
                        "test b12x MLA/RoPE attention",
                    )?;
                    let mut out_bytes = vec![0_u8; values_bf16.len()];
                    unsafe {
                        library
                            .copy_d2h_async(&mut out_bytes, output_buffer, cuda_stream)
                            .context("copying test b12x MLA/RoPE output to host")?;
                        library
                            .cuda_stream_synchronize(cuda_stream)
                            .context("synchronizing test b12x MLA/RoPE graph slot stream")?;
                    }
                    slot.captured_graphs.retain(|entry| {
                        entry.program != CoordinatorCudaGraphProgram::LayerB12xMlaRopeAttentionBf16
                    });
                    Ok(bf16_values_to_f32(&out_bytes))
                })?;
                for (actual, expected) in actual.iter().zip(expected.iter()) {
                    assert!(
                        (actual - expected).abs() <= 1.0e-3,
                        "rows={rows} actual={actual} expected={expected}"
                    );
                }
            }
            Ok(())
        })();

        match result {
            Ok(()) => Ok(()),
            Err(error) if cuda_allocation_unavailable(&error) => Ok(()),
            Err(error) => Err(error),
        }
    }

    #[test]
    fn b12x_mla_rope_attention_device_buffer_dispatch_uses_python_graph_when_enabled() -> Result<()>
    {
        let Some(_) = native_library_path() else {
            return Ok(());
        };
        let _env_guard = B12X_MLA_TEST_ENV_MUTEX
            .lock()
            .expect("b12x MLA test env mutex poisoned");
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);
        install_b12x_mla_cuda_reference_capture_module()?;
        let _b12x_guard = coordinator_python_capture_test_override(true);
        let _module_guard =
            EnvVarGuard::set("GLMRT_B12X_MLA_MODULE", "glmrt_test_b12x_mla_cuda_capture");
        let _function_guard = EnvVarGuard::set("GLMRT_B12X_MLA_FUNCTION", "capture");

        let result = (|| -> Result<()> {
            let rows = 1;
            let heads = 8;
            let nope_dim = GLM52_MLA_KV_LORA_RANK;
            let rope_dim = GLM52_MLA_QK_ROPE_HEAD_DIM;
            let v_dim = GLM52_MLA_KV_LORA_RANK;
            let scale = 1.0 / ((nope_dim + rope_dim) as f32).sqrt();
            let q_nope = patterned_bf16_values(rows * heads * nope_dim, 0.001, 3.0);
            let q_rope = patterned_bf16_values(rows * heads * rope_dim, 0.001, 5.0);
            let k_nope = patterned_bf16_values(rows * heads * nope_dim, 0.001, 7.0);
            let k_rope = patterned_bf16_values(rows * rope_dim, 0.001, 11.0);
            let values = patterned_bf16_values(rows * heads * v_dim, 0.001, 13.0);
            let q_nope_bf16 = bf16_bytes(&q_nope);
            let q_rope_bf16 = bf16_bytes(&q_rope);
            let k_nope_bf16 = bf16_bytes(&k_nope);
            let k_rope_bf16 = bf16_bytes(&k_rope);
            let values_bf16 = bf16_bytes(&values);
            let expected = bf16_values_to_f32(&f32_values_to_bf16_bytes(
                &cpu_mla_rope_attention_rows_bf16(
                    &q_nope_bf16,
                    &q_rope_bf16,
                    &k_nope_bf16,
                    &k_rope_bf16,
                    &values_bf16,
                    rows,
                    heads,
                    nope_dim,
                    rope_dim,
                    v_dim,
                    scale,
                )
                .values,
            ));

            let library = cuda_native_library()?;
            let q_nope_device =
                uploaded_test_device_buffer(library, &q_nope_bf16, "test b12x device q_nope")?;
            let q_rope_device =
                uploaded_test_device_buffer(library, &q_rope_bf16, "test b12x device q_rope")?;
            let k_nope_device =
                uploaded_test_device_buffer(library, &k_nope_bf16, "test b12x device k_nope")?;
            let k_rope_device =
                uploaded_test_device_buffer(library, &k_rope_bf16, "test b12x device k_rope")?;
            let values_device =
                uploaded_test_device_buffer(library, &values_bf16, "test b12x device values")?;
            let output_device = OwnedCoordinatorDeviceBuffer::new(
                library,
                values_bf16.len(),
                "test b12x device output",
            )?;

            let backend = mla_rope_attention_device_buffers_bf16_for_layer(
                0,
                q_nope_device.buffer,
                q_rope_device.buffer,
                k_nope_device.buffer,
                k_rope_device.buffer,
                values_device.buffer,
                output_device.buffer,
                rows,
                heads,
                nope_dim,
                rope_dim,
                v_dim,
                scale,
            )?;
            assert_eq!(backend, B12X_MLA_ROPE_ATTENTION_BF16_BACKEND);

            let mut out_bytes = vec![0_u8; values_bf16.len()];
            library.copy_d2h(&mut out_bytes, output_device.buffer)?;
            let actual = bf16_values_to_f32(&out_bytes);
            for (actual, expected) in actual.iter().zip(expected.iter()) {
                assert!(
                    (actual - expected).abs() <= 1.0e-3,
                    "actual={actual} expected={expected}"
                );
            }

            let graph_key = coord_attention_graph_key_for_layer_rows(0, rows)?;
            with_coordinator_cuda_graph_slot(&graph_key, |_library, slot| {
                slot.captured_graphs.retain(|entry| {
                    entry.program != CoordinatorCudaGraphProgram::LayerB12xMlaRopeAttentionBf16
                });
                Ok(())
            })?;
            Ok(())
        })();

        match result {
            Ok(()) => Ok(()),
            Err(error) if cuda_allocation_unavailable(&error) => Ok(()),
            Err(error) => Err(error),
        }
    }

    #[test]
    fn mla_rope_attention_rows_bf16_rejects_shape_mismatch_before_backend_selection() {
        let err = mla_rope_attention_rows_bf16(
            &bf16_bytes(&[1.0]),
            &bf16_bytes(&[1.0, 0.0]),
            &bf16_bytes(&[1.0]),
            &bf16_bytes(&[1.0, 0.0]),
            &bf16_bytes(&[1.0]),
            1,
            1,
            1,
            2,
            2,
            1.0,
        )
        .unwrap_err();

        assert!(err
            .to_string()
            .contains("BF16 MLA/RoPE attention value byte length mismatch"));
    }
}
