use anyhow::{bail, Context, Result};
use glmrt_core::{ExpertRequest, ExpertResponse};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::net::SocketAddr;
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::time::timeout;

use crate::{
    is_connection_closed, synthetic_expert_response, TcpTransportConfig, DEFAULT_MAX_FRAME_BYTES,
    DEFAULT_TIMEOUT, SYNTHETIC_EXPERT_KERNEL,
};

mod frame;

pub(crate) use frame::{encode_frame, read_frame, FrameKind};

/// Debug/integration expert protocol that serializes `ExpertRequest` as JSON
/// with host-side `Vec<f32>` hidden rows. The hot expert path is ProtocolV2.
pub const DEBUG_JSON_EXPERT_PROTOCOL_LABEL: &str = "glmrt-debug-json-f32-expert-protocol-v1";
pub const DEBUG_JSON_EXPERT_PROTOCOL_VERSION: u32 = 1;
/// Debug/integration TCP frame envelope used by the JSON/f32 expert protocol.
pub const DEBUG_JSON_FRAME_PROTOCOL: &str = "glmrt-debug-json-f32-frame-v1";

/// Compatibility alias for legacy callers. This uses the debug-only JSON/f32
/// expert protocol; new hot-path transport work should use ProtocolV2.
pub async fn tcp_roundtrip(
    addr: SocketAddr,
    request: &ExpertRequest,
    config: TcpTransportConfig,
) -> Result<ExpertResponse> {
    debug_json_tcp_roundtrip(addr, request, config).await
}

pub async fn debug_json_tcp_roundtrip(
    addr: SocketAddr,
    request: &ExpertRequest,
    config: TcpTransportConfig,
) -> Result<ExpertResponse> {
    let mut stream = timeout(config.timeout, TcpStream::connect(addr))
        .await
        .context("timed out connecting TCP transport")?
        .with_context(|| format!("connecting TCP transport to {addr}"))?;
    stream
        .set_nodelay(true)
        .context("setting TCP_NODELAY for TCP transport client")?;

    write_json_frame_with_timeout(
        &mut stream,
        FrameKind::Request,
        request.request_id,
        request,
        config.timeout,
        config.max_frame_bytes,
    )
    .await?;

    let response: ExpertResponse = read_json_frame_with_timeout(
        &mut stream,
        FrameKind::Response,
        config.timeout,
        config.max_frame_bytes,
    )
    .await?;
    if response.request_id != request.request_id {
        bail!(
            "response request_id {} did not match request_id {}",
            response.request_id,
            request.request_id
        );
    }
    Ok(response)
}

/// Compatibility alias for the synthetic debug JSON/f32 TCP expert service.
pub async fn serve_synthetic_tcp(addr: &str) -> Result<()> {
    serve_synthetic_debug_json_tcp(addr).await
}

pub async fn serve_synthetic_debug_json_tcp(addr: &str) -> Result<()> {
    serve_synthetic_debug_json_framed(addr, "tcp-debug-json").await
}

async fn serve_synthetic_debug_json_framed(
    addr: &str,
    transport_label: &'static str,
) -> Result<()> {
    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding synthetic {transport_label} expert service to {addr}"))?;
    tracing::info!(
        transport = transport_label,
        protocol = DEBUG_JSON_EXPERT_PROTOCOL_LABEL,
        frame_protocol = DEBUG_JSON_FRAME_PROTOCOL,
        addr = %listener.local_addr()?,
        "synthetic expert service listening"
    );
    loop {
        let (stream, peer) = listener
            .accept()
            .await
            .with_context(|| format!("accepting {transport_label} expert connection"))?;
        tokio::spawn(async move {
            if let Err(err) = handle_synthetic_connection(stream).await {
                tracing::warn!(
                    transport = transport_label,
                    peer = %peer,
                    error = %err,
                    "expert connection closed with error"
                );
            }
        });
    }
}

pub(crate) async fn handle_synthetic_connection(mut stream: TcpStream) -> Result<()> {
    stream
        .set_nodelay(true)
        .context("setting TCP_NODELAY for TCP expert connection")?;
    loop {
        let request: ExpertRequest = match read_json_frame_with_timeout(
            &mut stream,
            FrameKind::Request,
            DEFAULT_TIMEOUT,
            DEFAULT_MAX_FRAME_BYTES,
        )
        .await
        {
            Ok(request) => request,
            Err(err) if is_connection_closed(&err) => return Ok(()),
            Err(err) => return Err(err),
        };
        let header = request.header();
        let route_count = request
            .rows
            .iter()
            .map(|row| row.routes.len())
            .sum::<usize>();
        tracing::info!(
            request_id = header.request_id,
            layer_id = header.layer_id,
            hidden_dim = header.hidden_dim,
            row_count = header.row_count,
            route_count,
            wave_mode = ?header.wave_mode,
            graph_bucket_rows = ?header.graph_bucket_rows,
            logical_bf16_payload_bytes = ?header.logical_bf16_payload_bytes,
            protocol = DEBUG_JSON_EXPERT_PROTOCOL_LABEL,
            frame_protocol = DEBUG_JSON_FRAME_PROTOCOL,
            synthetic_kernel = SYNTHETIC_EXPERT_KERNEL,
            "synthetic expert request received"
        );
        let response = synthetic_expert_response(&request)?;
        write_json_frame_with_timeout(
            &mut stream,
            FrameKind::Response,
            response.request_id,
            &response,
            DEFAULT_TIMEOUT,
            DEFAULT_MAX_FRAME_BYTES,
        )
        .await?;
    }
}

async fn write_json_frame_with_timeout<W, T>(
    writer: &mut W,
    kind: FrameKind,
    request_id: u64,
    value: &T,
    timeout_duration: Duration,
    max_frame_bytes: usize,
) -> Result<()>
where
    W: AsyncWrite + Unpin,
    T: Serialize,
{
    let payload = serde_json::to_vec(value).context("serializing transport frame payload")?;
    let frame = encode_frame(kind, request_id, payload, max_frame_bytes)?;
    timeout(timeout_duration, writer.write_all(&frame))
        .await
        .context("timed out writing transport frame")?
        .context("writing transport frame")?;
    timeout(timeout_duration, writer.flush())
        .await
        .context("timed out flushing transport frame")?
        .context("flushing transport frame")?;
    Ok(())
}

async fn read_json_frame_with_timeout<R, T>(
    reader: &mut R,
    expected_kind: FrameKind,
    timeout_duration: Duration,
    max_frame_bytes: usize,
) -> Result<T>
where
    R: AsyncRead + Unpin,
    T: DeserializeOwned,
{
    let frame = timeout(timeout_duration, read_frame(reader, max_frame_bytes))
        .await
        .context("timed out reading transport frame")??;
    if frame.kind != expected_kind {
        bail!(
            "expected frame kind {:?}, got {:?}",
            expected_kind,
            frame.kind
        );
    }
    tracing::trace!(request_id = frame.request_id, "read transport frame");
    serde_json::from_slice(&frame.payload).context("deserializing transport frame payload")
}
