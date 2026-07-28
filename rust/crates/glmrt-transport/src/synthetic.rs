use anyhow::{bail, Context, Result};
use glmrt_core::{ExpertRequest, ExpertResponse, LayerWaveMode};
use glmrt_ffi::GlmrtDeviceBuffer;

use crate::{
    expert_protocol_v2_compact_id, ExpertProtocolV2DeviceResponseRef, ExpertProtocolV2Request,
    ExpertProtocolV2RequestView, ExpertProtocolV2Response, ExpertProtocolV2ResponseRef,
    ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor, ExpertProtocolV2Status,
    ExpertV2Dtype, ExpertV2SourceKind,
};

pub const SYNTHETIC_EXPERT_KERNEL: &str = "synthetic-nvfp4-bf16-diagonal-mlp";
pub const PROTOCOL_V2_ECHO_EXECUTOR: &str = "protocol-v2-echo-loopback";
pub const PROTOCOL_V2_SYNTHETIC_ROUTE_EXECUTOR: &str =
    "protocol-v2-synthetic-route-dependent-executor";

#[derive(Debug, Clone, Copy)]
pub struct ProtocolV2RequestDevicePayload {
    pub hidden_payload: GlmrtDeviceBuffer,
    pub response_slot: Option<GlmrtDeviceBuffer>,
    pub execution_lane: u32,
}

#[derive(Debug, Clone)]
pub enum ProtocolV2ExecutorResponseRef<'a> {
    Host(ExpertProtocolV2ResponseRef<'a>),
    Device(ExpertProtocolV2DeviceResponseRef<'a>),
}

impl<'a> ProtocolV2ExecutorResponseRef<'a> {
    pub fn more_chunks(&self) -> bool {
        match self {
            Self::Host(response) => response.more_chunks(),
            Self::Device(response) => response.more_chunks(),
        }
    }

    pub fn with_executor_name(self, executor_name: &str) -> Self {
        match self {
            Self::Host(response) => Self::Host(response.with_executor_name(executor_name)),
            Self::Device(response) => Self::Device(response.with_executor_name(executor_name)),
        }
    }
}

const SYNTHETIC_NVFP4_CODEBOOK: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];
const SYNTHETIC_WEIGHT_SCALE: f32 = 1.0 / 32.0;

pub trait ProtocolV2ExpertExecutor: Send + Sync {
    fn name(&self) -> &'static str;
    fn execute(
        &self,
        request: &ExpertProtocolV2RequestView<'_>,
    ) -> Result<ExpertProtocolV2Response>;

    fn execute_with_identity(
        &self,
        request: &ExpertProtocolV2RequestView<'_>,
    ) -> Result<ExpertProtocolV2Response> {
        Ok(self.execute(request)?.with_executor_name(self.name()))
    }

    fn execute_streaming(
        &self,
        request: &ExpertProtocolV2RequestView<'_>,
        emit: &mut dyn FnMut(ExpertProtocolV2ResponseRef<'_>) -> Result<()>,
    ) -> Result<()> {
        let response = self.execute(request)?;
        emit(response.as_borrowed())
    }

    fn execute_streaming_device_payload(
        &self,
        request: &ExpertProtocolV2RequestView<'_>,
        _device_payload: ProtocolV2RequestDevicePayload,
        emit: &mut dyn FnMut(ProtocolV2ExecutorResponseRef<'_>) -> Result<()>,
    ) -> Result<()> {
        self.execute_streaming(request, &mut |response| {
            emit(ProtocolV2ExecutorResponseRef::Host(response))
        })
    }

    fn execute_streaming_with_identity(
        &self,
        request: &ExpertProtocolV2RequestView<'_>,
        emit: &mut dyn FnMut(ExpertProtocolV2ResponseRef<'_>) -> Result<()>,
    ) -> Result<()> {
        let executor_name = self.name();
        self.execute_streaming(request, &mut |response| {
            emit(response.with_executor_name(executor_name))
        })
    }

    fn execute_streaming_device_payload_with_identity(
        &self,
        request: &ExpertProtocolV2RequestView<'_>,
        device_payload: ProtocolV2RequestDevicePayload,
        emit: &mut dyn FnMut(ProtocolV2ExecutorResponseRef<'_>) -> Result<()>,
    ) -> Result<()> {
        let executor_name = self.name();
        self.execute_streaming_device_payload(request, device_payload, &mut |response| {
            emit(response.with_executor_name(executor_name))
        })
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct EchoExecutor;

#[derive(Debug, Default, Clone, Copy)]
pub struct SyntheticRouteExecutor;

impl ProtocolV2ExpertExecutor for EchoExecutor {
    fn name(&self) -> &'static str {
        PROTOCOL_V2_ECHO_EXECUTOR
    }

    fn execute(
        &self,
        request: &ExpertProtocolV2RequestView<'_>,
    ) -> Result<ExpertProtocolV2Response> {
        verify_request_checksum_if_enabled(request)?;
        let response = ExpertProtocolV2Response::new_with_output_stride(
            request.header.request_id,
            request.header.placement_version,
            request.header.layer_id,
            request.header.row_count,
            request.header.hidden_dim,
            request.header.hidden_dtype,
            request.header.hidden_row_stride_bytes,
            ExpertProtocolV2Status::Ok,
            request.hidden_payload().to_vec(),
        )?;
        Ok(with_matching_debug_checksum(request, response))
    }
}

impl ProtocolV2ExpertExecutor for SyntheticRouteExecutor {
    fn name(&self) -> &'static str {
        PROTOCOL_V2_SYNTHETIC_ROUTE_EXECUTOR
    }

    fn execute(
        &self,
        request: &ExpertProtocolV2RequestView<'_>,
    ) -> Result<ExpertProtocolV2Response> {
        verify_request_checksum_if_enabled(request)?;
        if request.header.hidden_dtype != ExpertV2Dtype::Bf16 {
            bail!(
                "ProtocolV2 SyntheticRouteExecutor requires BF16 hidden dtype, got {:?}",
                request.header.hidden_dtype
            );
        }

        let row_count = request.header.row_count as usize;
        let hidden_dim = request.header.hidden_dim as usize;
        let output_stride = request.header.hidden_row_stride_bytes as usize;
        let logical_row_bytes = hidden_dim
            .checked_mul(ExpertV2Dtype::Bf16.bytes_per_element())
            .context("ProtocolV2 synthetic output row byte count overflow")?;
        let mut output_payload = vec![0_u8; row_count * output_stride];
        for row_index in 0..row_count {
            let row = request.row(row_index)?;
            let route_start = row.route_offset as usize;
            let route_end = route_start
                .checked_add(row.route_count as usize)
                .context("ProtocolV2 synthetic row route range overflow")?;
            let routes = (route_start..route_end)
                .map(|route_index| {
                    let route = request.route(route_index)?;
                    if route.row_index as usize != row_index {
                        bail!(
                            "ProtocolV2 synthetic route row_index {} did not match row {row_index}",
                            route.row_index
                        );
                    }
                    Ok(route)
                })
                .collect::<Result<Vec<_>>>()?;
            let hidden_row = request.hidden_row_payload(row_index)?;
            let hidden = bf16_row_to_f32(hidden_row, hidden_dim)?;
            let output = synthetic_nvfp4_bf16_expert_output_for_routes(
                request.header.layer_id,
                hidden_dim,
                &hidden,
                routes.iter(),
            );
            let start = row_index * output_stride;
            f32_values_to_bf16_bytes(
                &output,
                &mut output_payload[start..start + logical_row_bytes],
            );
        }

        let response = ExpertProtocolV2Response::new_with_output_stride(
            request.header.request_id,
            request.header.placement_version,
            request.header.layer_id,
            request.header.row_count,
            request.header.hidden_dim,
            ExpertV2Dtype::Bf16,
            request.header.hidden_row_stride_bytes,
            ExpertProtocolV2Status::Ok,
            output_payload,
        )?;
        Ok(with_matching_debug_checksum(request, response))
    }
}

pub fn synthetic_expert_response(request: &ExpertRequest) -> Result<ExpertResponse> {
    if request.hidden_dim == 0 {
        bail!("hidden_dim must be non-zero");
    }
    if let Some(wave) = &request.wave {
        if wave.graph_bucket_rows < request.rows.len() as u32 {
            bail!(
                "wave graph bucket rows {} smaller than request row count {}",
                wave.graph_bucket_rows,
                request.rows.len()
            );
        }
        let expected_logical_bytes = request.rows.len() * request.hidden_dim as usize * 2;
        if wave.logical_bf16_payload_bytes != expected_logical_bytes {
            bail!(
                "wave logical BF16 payload bytes {} did not match expected {}",
                wave.logical_bf16_payload_bytes,
                expected_logical_bytes
            );
        }
    }
    let mut partial_outputs = Vec::with_capacity(request.rows.len());
    for row in &request.rows {
        if row.hidden.len() != request.hidden_dim as usize {
            bail!(
                "row {} hidden length {} did not match hidden_dim {}",
                row.row_id,
                row.hidden.len(),
                request.hidden_dim
            );
        }
        partial_outputs.push(synthetic_nvfp4_bf16_expert_output(
            request.layer_id,
            request.hidden_dim as usize,
            row,
        ));
    }
    Ok(ExpertResponse {
        request_id: request.request_id,
        placement_version: request.placement_version.clone(),
        layer_id: request.layer_id,
        status: "ok".to_owned(),
        partial_outputs,
    })
}

pub fn protocol_v2_synthetic_response(
    request: &ExpertProtocolV2Request,
) -> Result<ExpertProtocolV2Response> {
    protocol_v2_route_dependent_synthetic_response(request)
}

pub fn protocol_v2_route_dependent_synthetic_response(
    request: &ExpertProtocolV2Request,
) -> Result<ExpertProtocolV2Response> {
    execute_owned_protocol_v2_request(request, SyntheticRouteExecutor)
}

pub fn protocol_v2_echo_loopback_response(
    request: &ExpertProtocolV2Request,
) -> Result<ExpertProtocolV2Response> {
    execute_owned_protocol_v2_request(request, EchoExecutor)
}

pub fn protocol_v2_request_from_expert_request(
    request: &ExpertRequest,
) -> Result<ExpertProtocolV2Request> {
    if request.hidden_dim == 0 {
        bail!("ProtocolV2 bridge hidden_dim must be non-zero");
    }
    if let Some(wave) = &request.wave {
        if wave.graph_bucket_rows < request.rows.len() as u32 {
            bail!(
                "ProtocolV2 bridge wave graph bucket rows {} smaller than request row count {}",
                wave.graph_bucket_rows,
                request.rows.len()
            );
        }
        let expected_logical_bytes = request.rows.len() * request.hidden_dim as usize * 2;
        if wave.logical_bf16_payload_bytes != expected_logical_bytes {
            bail!(
                "ProtocolV2 bridge wave logical BF16 payload bytes {} did not match expected {}",
                wave.logical_bf16_payload_bytes,
                expected_logical_bytes
            );
        }
    }

    let hidden_dim = request.hidden_dim as usize;
    let source_kind = request
        .wave
        .as_ref()
        .map(|wave| protocol_v2_source_kind(wave.mode))
        .unwrap_or(ExpertV2SourceKind::Decode);
    let mut rows = Vec::with_capacity(request.rows.len());
    let mut routes = Vec::new();
    let mut hidden_payload = Vec::with_capacity(request.rows.len() * hidden_dim * 2);

    for (row_index, row) in request.rows.iter().enumerate() {
        if row.hidden.len() != hidden_dim {
            bail!(
                "ProtocolV2 bridge row {} hidden length {} did not match hidden_dim {}",
                row.row_id,
                row.hidden.len(),
                hidden_dim
            );
        }
        rows.push(ExpertProtocolV2RowDescriptor {
            row_id: row.row_id,
            source_kind,
            source_request_id: request.request_id,
            token_position: row.row_id,
            route_offset: routes.len() as u32,
            route_count: row.routes.len() as u32,
        });
        for route in &row.routes {
            if !route.gate.is_finite() {
                bail!("ProtocolV2 bridge route gate must be finite");
            }
            routes.push(ExpertProtocolV2RouteEntry {
                row_index: row_index as u32,
                expert_id: route.expert_id,
                gate_weight: route.gate,
            });
        }
        let start = hidden_payload.len();
        hidden_payload.resize(start + hidden_dim * 2, 0);
        f32_values_to_bf16_bytes(&row.hidden, &mut hidden_payload[start..]);
    }

    ExpertProtocolV2Request::new(
        request.request_id,
        expert_protocol_v2_compact_id(&request.placement_version),
        request.layer_id,
        request.hidden_dim,
        ExpertV2Dtype::Bf16,
        rows,
        routes,
        hidden_payload,
    )
}

pub fn expert_response_from_protocol_v2_response(
    request: &ExpertRequest,
    response: &ExpertProtocolV2Response,
) -> Result<ExpertResponse> {
    let expected_placement = expert_protocol_v2_compact_id(&request.placement_version);
    if response.header.request_id != request.request_id {
        bail!(
            "ProtocolV2 bridge response request_id {} did not match request_id {}",
            response.header.request_id,
            request.request_id
        );
    }
    if response.header.placement_version != expected_placement {
        bail!(
            "ProtocolV2 bridge response placement_version {} did not match compact request placement {}",
            response.header.placement_version,
            expected_placement
        );
    }
    if response.header.layer_id != request.layer_id {
        bail!(
            "ProtocolV2 bridge response layer_id {} did not match request layer_id {}",
            response.header.layer_id,
            request.layer_id
        );
    }
    if response.header.row_count as usize != request.rows.len() {
        bail!(
            "ProtocolV2 bridge response row_count {} did not match request rows {}",
            response.header.row_count,
            request.rows.len()
        );
    }
    if response.header.output_dtype != ExpertV2Dtype::Bf16 {
        bail!(
            "ProtocolV2 bridge response output dtype {:?} is not supported",
            response.header.output_dtype
        );
    }

    let output_dim = response.header.output_dim as usize;
    let output_stride = response.header.output_row_stride_bytes as usize;
    let mut partial_outputs = Vec::with_capacity(request.rows.len());
    for row_index in 0..request.rows.len() {
        let start = row_index
            .checked_mul(output_stride)
            .context("ProtocolV2 bridge response row offset overflow")?;
        let end = start
            .checked_add(output_stride)
            .context("ProtocolV2 bridge response row end overflow")?;
        let row_bytes = response
            .partial_output_payload
            .get(start..end)
            .with_context(|| {
                format!(
                    "ProtocolV2 bridge response row {row_index} range {start}..{end} exceeds payload {}",
                    response.partial_output_payload.len()
                )
            })?;
        partial_outputs.push(bf16_row_to_f32(row_bytes, output_dim)?);
    }

    Ok(ExpertResponse {
        request_id: request.request_id,
        placement_version: request.placement_version.clone(),
        layer_id: request.layer_id,
        status: match response.header.status {
            ExpertProtocolV2Status::Ok => "ok",
            ExpertProtocolV2Status::Error => "error",
        }
        .to_owned(),
        partial_outputs,
    })
}

pub(crate) fn protocol_v2_synthetic_response_from_view(
    request: &ExpertProtocolV2RequestView<'_>,
) -> Result<ExpertProtocolV2Response> {
    SyntheticRouteExecutor.execute_with_identity(request)
}

fn execute_owned_protocol_v2_request(
    request: &ExpertProtocolV2Request,
    executor: impl ProtocolV2ExpertExecutor,
) -> Result<ExpertProtocolV2Response> {
    let frame = request.encode()?;
    let view = ExpertProtocolV2RequestView::parse(&frame)?;
    executor.execute_with_identity(&view)
}

fn protocol_v2_source_kind(mode: LayerWaveMode) -> ExpertV2SourceKind {
    match mode {
        LayerWaveMode::Decode => ExpertV2SourceKind::Decode,
        LayerWaveMode::Prefill => ExpertV2SourceKind::Prefill,
        LayerWaveMode::MtpVerify => ExpertV2SourceKind::MtpVerify,
        LayerWaveMode::Benchmark => ExpertV2SourceKind::Benchmark,
    }
}

fn verify_request_checksum_if_enabled(request: &ExpertProtocolV2RequestView<'_>) -> Result<()> {
    if request.debug_checksum_enabled() {
        request.verify_checksum()?;
    }
    Ok(())
}

fn with_matching_debug_checksum(
    request: &ExpertProtocolV2RequestView<'_>,
    response: ExpertProtocolV2Response,
) -> ExpertProtocolV2Response {
    if request.debug_checksum_enabled() {
        response.with_debug_checksum()
    } else {
        response
    }
}

fn synthetic_nvfp4_bf16_expert_output(
    layer_id: u32,
    hidden_dim: usize,
    row: &glmrt_core::ExpertRow,
) -> Vec<f32> {
    let hidden_bf16 = row
        .hidden
        .iter()
        .map(|value| bf16_truncate(*value))
        .collect::<Vec<_>>();
    synthetic_nvfp4_bf16_expert_output_for_routes(
        layer_id,
        hidden_dim,
        &hidden_bf16,
        row.routes.iter().map(|route| RouteRef {
            expert_id: route.expert_id,
            gate_weight: route.gate,
        }),
    )
}

fn synthetic_nvfp4_bf16_expert_output_for_routes<R>(
    layer_id: u32,
    hidden_dim: usize,
    hidden_bf16: &[f32],
    routes: R,
) -> Vec<f32>
where
    R: IntoIterator,
    R::Item: SyntheticRouteLike,
{
    let mut output = vec![0.0_f32; hidden_dim];
    for route in routes {
        let expert_id = route.expert_id();
        let gate_weight = route.gate_weight();
        for out_idx in 0..hidden_dim {
            let gate_src = synthetic_source_index(hidden_dim, out_idx, expert_id, 0);
            let up_src = synthetic_source_index(hidden_dim, out_idx, expert_id, 1);
            let gate = hidden_bf16[out_idx]
                * synthetic_nvfp4_weight(layer_id, expert_id, out_idx, 0)
                + hidden_bf16[gate_src] * synthetic_nvfp4_weight(layer_id, expert_id, out_idx, 1);
            let up = hidden_bf16[up_src] * synthetic_nvfp4_weight(layer_id, expert_id, out_idx, 2)
                + hidden_bf16[out_idx] * synthetic_nvfp4_weight(layer_id, expert_id, out_idx, 3);
            let down = synthetic_nvfp4_weight(layer_id, expert_id, out_idx, 4);
            output[out_idx] += gate_weight * silu(gate) * up * down;
        }
    }
    output
}

trait SyntheticRouteLike {
    fn expert_id(&self) -> u32;
    fn gate_weight(&self) -> f32;
}

struct RouteRef {
    expert_id: u32,
    gate_weight: f32,
}

impl SyntheticRouteLike for RouteRef {
    fn expert_id(&self) -> u32 {
        self.expert_id
    }

    fn gate_weight(&self) -> f32 {
        self.gate_weight
    }
}

impl SyntheticRouteLike for &ExpertProtocolV2RouteEntry {
    fn expert_id(&self) -> u32 {
        self.expert_id
    }

    fn gate_weight(&self) -> f32 {
        self.gate_weight
    }
}

fn bf16_row_to_f32(row_bytes: &[u8], hidden_dim: usize) -> Result<Vec<f32>> {
    let logical_len = hidden_dim
        .checked_mul(ExpertV2Dtype::Bf16.bytes_per_element())
        .context("BF16 row byte count overflow")?;
    if row_bytes.len() < logical_len {
        bail!(
            "BF16 row bytes {} smaller than hidden_dim {} logical bytes {}",
            row_bytes.len(),
            hidden_dim,
            logical_len
        );
    }
    Ok(row_bytes[..logical_len]
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            f32::from_bits((bits as u32) << 16)
        })
        .collect())
}

fn f32_values_to_bf16_bytes(values: &[f32], out: &mut [u8]) {
    for (value, dst) in values.iter().zip(out.chunks_exact_mut(2)) {
        let bf16 = (bf16_truncate(*value).to_bits() >> 16) as u16;
        dst.copy_from_slice(&bf16.to_le_bytes());
    }
}

fn bf16_truncate(value: f32) -> f32 {
    f32::from_bits(value.to_bits() & 0xFFFF_0000)
}

fn synthetic_source_index(
    hidden_dim: usize,
    output_index: usize,
    expert_id: u32,
    projection: u32,
) -> usize {
    let offset = (expert_id as usize)
        .wrapping_mul(37)
        .wrapping_add(projection as usize * 97)
        .wrapping_add(13);
    (output_index + offset) % hidden_dim
}

fn synthetic_nvfp4_weight(
    layer_id: u32,
    expert_id: u32,
    output_index: usize,
    projection: u32,
) -> f32 {
    let code = synthetic_nvfp4_code(layer_id, expert_id, output_index, projection);
    SYNTHETIC_NVFP4_CODEBOOK[code as usize] * SYNTHETIC_WEIGHT_SCALE
}

fn synthetic_nvfp4_code(layer_id: u32, expert_id: u32, output_index: usize, projection: u32) -> u8 {
    let mut value = (layer_id as u64).wrapping_mul(0x9E37_79B1_85EB_CA87);
    value ^= (expert_id as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    value ^= (output_index as u64).wrapping_mul(0x1656_67B1_9E37_79F9);
    value ^= (projection as u64).wrapping_mul(0x85EB_CA77_C2B2_AE63);
    value ^= value >> 33;
    value = value.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    value ^= value >> 33;
    (value & 0x0F) as u8
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}
