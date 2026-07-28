use anyhow::{bail, Context, Result};
use glmrt_core::{ExpertRequest, ExpertResponse};
use std::env;
use std::io::IoSlice;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::time::timeout;

use crate::synthetic::{
    expert_response_from_protocol_v2_response, protocol_v2_request_from_expert_request,
    ProtocolV2ExpertExecutor, SyntheticRouteExecutor,
};
use crate::{
    is_connection_closed, ExpertProtocolV2FrameArena, ExpertProtocolV2FrameBuffer,
    ExpertProtocolV2Request, ExpertProtocolV2RequestView, ExpertProtocolV2Response,
    ExpertProtocolV2ResponseHeader, ExpertProtocolV2ResponseView, TcpTransportConfig,
    DEFAULT_MAX_FRAME_BYTES, DEFAULT_TIMEOUT, EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN,
    EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN,
};

const PROTOCOL_V2_TCP_TIMING_ENV: &str = "GLMRT_PROTOCOL_V2_TCP_TIMING";

pub async fn tcp_protocol_v2_roundtrip(
    addr: SocketAddr,
    request: &ExpertProtocolV2Request,
    config: TcpTransportConfig,
) -> Result<ExpertProtocolV2Response> {
    let mut read_buffer = ExpertProtocolV2FrameBuffer::new();
    {
        let _view =
            tcp_protocol_v2_roundtrip_response_view(addr, request, config, &mut read_buffer)
                .await?;
    }
    ExpertProtocolV2Response::decode(read_buffer.as_slice())
}

pub async fn tcp_protocol_v2_expert_request_roundtrip(
    addr: SocketAddr,
    request: &ExpertRequest,
    config: TcpTransportConfig,
) -> Result<ExpertResponse> {
    let protocol_v2_request = protocol_v2_request_from_expert_request(request)?;
    let protocol_v2_response =
        tcp_protocol_v2_roundtrip(addr, &protocol_v2_request, config).await?;
    expert_response_from_protocol_v2_response(request, &protocol_v2_response)
}

pub async fn tcp_protocol_v2_roundtrip_response_view<'a>(
    addr: SocketAddr,
    request: &ExpertProtocolV2Request,
    config: TcpTransportConfig,
    response_buffer: &'a mut ExpertProtocolV2FrameBuffer,
) -> Result<ExpertProtocolV2ResponseView<'a>> {
    let mut stream = timeout(config.timeout, TcpStream::connect(addr))
        .await
        .context("timed out connecting TCP ProtocolV2 transport")?
        .with_context(|| format!("connecting TCP ProtocolV2 transport to {addr}"))?;
    stream
        .set_nodelay(true)
        .context("setting TCP_NODELAY for TCP ProtocolV2 transport client")?;
    let mut write_buffer =
        ExpertProtocolV2FrameBuffer::with_capacity(request.wire_stats().wire_bytes);

    write_protocol_v2_request_with_timeout(
        &mut stream,
        request,
        config.timeout,
        config.max_frame_bytes,
        &mut write_buffer,
    )
    .await?;

    read_protocol_v2_response_frame_with_timeout(
        &mut stream,
        config.timeout,
        config.max_frame_bytes,
        response_buffer,
    )
    .await?;
    let view = ExpertProtocolV2ResponseView::parse(response_buffer.as_slice())
        .context("validating TCP ProtocolV2 response frame view")?;
    validate_response_matches_request(&view.header, request)?;
    Ok(view)
}

pub async fn tcp_protocol_v2_roundtrip_arena_response_view<'a>(
    addr: SocketAddr,
    request: &ExpertProtocolV2Request,
    config: TcpTransportConfig,
    arena: &'a mut ExpertProtocolV2FrameArena,
) -> Result<ExpertProtocolV2ResponseView<'a>> {
    let mut stream = timeout(config.timeout, TcpStream::connect(addr))
        .await
        .context("timed out connecting TCP ProtocolV2 transport")?
        .with_context(|| format!("connecting TCP ProtocolV2 transport to {addr}"))?;
    stream
        .set_nodelay(true)
        .context("setting TCP_NODELAY for TCP ProtocolV2 transport client")?;

    write_protocol_v2_request_with_timeout(
        &mut stream,
        request,
        config.timeout,
        config.max_frame_bytes,
        arena.request_buffer_mut(),
    )
    .await?;

    read_protocol_v2_response_frame_with_timeout(
        &mut stream,
        config.timeout,
        config.max_frame_bytes,
        arena.response_buffer_mut(),
    )
    .await?;
    let view = ExpertProtocolV2ResponseView::parse(arena.response_frame())
        .context("validating TCP ProtocolV2 arena response frame view")?;
    validate_response_matches_request(&view.header, request)?;
    Ok(view)
}

pub struct TcpProtocolV2PersistentClient {
    addr: SocketAddr,
    config: TcpTransportConfig,
    stream: Option<TcpStream>,
    request_buffer: ExpertProtocolV2FrameBuffer,
    response_buffer: ExpertProtocolV2FrameBuffer,
}

impl TcpProtocolV2PersistentClient {
    pub fn new(addr: SocketAddr, config: TcpTransportConfig) -> Self {
        Self {
            addr,
            config,
            stream: None,
            request_buffer: ExpertProtocolV2FrameBuffer::new(),
            response_buffer: ExpertProtocolV2FrameBuffer::new(),
        }
    }

    pub async fn roundtrip(
        &mut self,
        request: &ExpertProtocolV2Request,
    ) -> Result<ExpertProtocolV2Response> {
        match self.roundtrip_once(request).await {
            Ok(response) => Ok(response),
            Err(err)
                if is_connection_closed(&err)
                    || is_protocol_v2_response_request_id_mismatch(&err) =>
            {
                self.stream = None;
                self.roundtrip_once(request).await
            }
            Err(err) => {
                self.stream = None;
                Err(err)
            }
        }
    }

    pub async fn roundtrip_response_view(
        &mut self,
        request: &ExpertProtocolV2Request,
    ) -> Result<ExpertProtocolV2ResponseView<'_>> {
        match self.roundtrip_frame_once(request).await {
            Ok(()) => {}
            Err(err)
                if is_connection_closed(&err)
                    || is_protocol_v2_response_request_id_mismatch(&err) =>
            {
                self.stream = None;
                self.roundtrip_frame_once(request).await?;
            }
            Err(err) => {
                self.stream = None;
                return Err(err);
            }
        }
        let view = ExpertProtocolV2ResponseView::parse(self.response_buffer.as_slice())
            .context("validating persistent TCP ProtocolV2 response frame view")?;
        validate_response_matches_request(&view.header, request)?;
        Ok(view)
    }

    async fn roundtrip_once(
        &mut self,
        request: &ExpertProtocolV2Request,
    ) -> Result<ExpertProtocolV2Response> {
        self.roundtrip_frame_once(request).await?;
        let decode_started = Instant::now();
        let response = ExpertProtocolV2Response::decode(self.response_buffer.as_slice())?;
        let decode_ms = elapsed_ms(decode_started);
        if protocol_v2_tcp_timing_enabled() {
            tracing::info!(
                addr = %self.addr,
                request_id = request.header.request_id,
                layer_id = request.header.layer_id,
                rows = request.header.row_count,
                routes = request.header.route_count,
                response_wire_bytes = response.wire_stats().wire_bytes,
                decode_ms = decode_ms,
                "protocol_v2_persistent_client_response_decode_timing"
            );
        }
        Ok(response)
    }

    async fn roundtrip_frame_once(&mut self, request: &ExpertProtocolV2Request) -> Result<()> {
        let timing_enabled = protocol_v2_tcp_timing_enabled();
        let mut connect_ms = 0.0_f64;
        if self.stream.is_none() {
            let started = Instant::now();
            self.stream = Some(connect_protocol_v2_stream(self.addr, self.config.clone()).await?);
            connect_ms = elapsed_ms(started);
        }
        let stream = self
            .stream
            .as_mut()
            .expect("stream is connected when present");
        let write_started = Instant::now();
        write_protocol_v2_request_with_timeout(
            stream,
            request,
            self.config.timeout,
            self.config.max_frame_bytes,
            &mut self.request_buffer,
        )
        .await?;
        let write_ms = elapsed_ms(write_started);

        let read_started = Instant::now();
        read_protocol_v2_response_frame_with_timeout(
            stream,
            self.config.timeout,
            self.config.max_frame_bytes,
            &mut self.response_buffer,
        )
        .await?;
        let read_ms = elapsed_ms(read_started);
        let parse_started = Instant::now();
        {
            let view = ExpertProtocolV2ResponseView::parse(self.response_buffer.as_slice())
                .context("validating persistent TCP ProtocolV2 response frame view")?;
            validate_response_matches_request(&view.header, request)?;
        }
        let parse_ms = elapsed_ms(parse_started);
        if timing_enabled {
            tracing::info!(
                addr = %self.addr,
                request_id = request.header.request_id,
                layer_id = request.header.layer_id,
                rows = request.header.row_count,
                routes = request.header.route_count,
                request_wire_bytes = request.wire_stats().wire_bytes,
                response_wire_bytes = self.response_buffer.len(),
                connect_ms = connect_ms,
                write_ms = write_ms,
                read_ms = read_ms,
                parse_ms = parse_ms,
                "protocol_v2_persistent_client_roundtrip_timing"
            );
        }
        Ok(())
    }

    pub fn reset(&mut self) {
        self.stream = None;
    }
}

async fn connect_protocol_v2_stream(
    addr: SocketAddr,
    config: TcpTransportConfig,
) -> Result<TcpStream> {
    let stream = timeout(config.timeout, TcpStream::connect(addr))
        .await
        .context("timed out connecting TCP ProtocolV2 transport")?
        .with_context(|| format!("connecting TCP ProtocolV2 transport to {addr}"))?;
    stream
        .set_nodelay(true)
        .context("setting TCP_NODELAY for TCP ProtocolV2 transport client")?;
    Ok(stream)
}

fn validate_response_matches_request(
    response: &ExpertProtocolV2ResponseHeader,
    request: &ExpertProtocolV2Request,
) -> Result<()> {
    if response.request_id != request.header.request_id {
        bail!(
            "ProtocolV2 response request_id {} did not match request_id {}",
            response.request_id,
            request.header.request_id
        );
    }
    if response.placement_version != request.header.placement_version {
        bail!(
            "ProtocolV2 response placement_version {} did not match request placement_version {}",
            response.placement_version,
            request.header.placement_version
        );
    }
    if response.layer_id != request.header.layer_id {
        bail!(
            "ProtocolV2 response layer_id {} did not match request layer_id {}",
            response.layer_id,
            request.header.layer_id
        );
    }
    Ok(())
}

fn is_protocol_v2_response_request_id_mismatch(error: &anyhow::Error) -> bool {
    let message = error.to_string();
    message.starts_with("ProtocolV2 response request_id ")
        && message.contains(" did not match request_id ")
}

pub async fn serve_synthetic_protocol_v2_tcp(addr: &str) -> Result<()> {
    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding synthetic ProtocolV2 TCP expert service to {addr}"))?;
    serve_synthetic_protocol_v2_tcp_listener(listener).await
}

pub async fn serve_synthetic_protocol_v2_tcp_listener(listener: TcpListener) -> Result<()> {
    serve_protocol_v2_tcp_listener_with_executor(listener, Arc::new(SyntheticRouteExecutor)).await
}

pub async fn serve_protocol_v2_tcp_with_executor(
    addr: &str,
    executor: Arc<dyn ProtocolV2ExpertExecutor>,
) -> Result<()> {
    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding ProtocolV2 TCP expert service to {addr}"))?;
    serve_protocol_v2_tcp_listener_with_executor(listener, executor).await
}

pub async fn serve_protocol_v2_tcp_listener_with_executor(
    listener: TcpListener,
    executor: Arc<dyn ProtocolV2ExpertExecutor>,
) -> Result<()> {
    tracing::info!(
        addr = %listener.local_addr()?,
        executor = executor.name(),
        "ProtocolV2 expert service listening"
    );
    loop {
        let (stream, peer) = listener
            .accept()
            .await
            .context("accepting ProtocolV2 expert connection")?;
        let executor = Arc::clone(&executor);
        tokio::spawn(async move {
            if let Err(err) = handle_protocol_v2_connection_with_executor(stream, executor).await {
                tracing::warn!(
                    peer = %peer,
                    error = %err,
                    "ProtocolV2 expert connection closed with error"
                );
            }
        });
    }
}

#[cfg(test)]
pub(crate) async fn handle_protocol_v2_synthetic_connection(stream: TcpStream) -> Result<()> {
    handle_protocol_v2_connection_with_executor(stream, Arc::new(SyntheticRouteExecutor)).await
}

pub(crate) async fn handle_protocol_v2_connection_with_executor(
    mut stream: TcpStream,
    executor: Arc<dyn ProtocolV2ExpertExecutor>,
) -> Result<()> {
    stream
        .set_nodelay(true)
        .context("setting TCP_NODELAY for TCP ProtocolV2 expert connection")?;
    let mut read_buffer = ExpertProtocolV2FrameBuffer::new();
    let mut write_buffer = ExpertProtocolV2FrameBuffer::new();
    loop {
        let request = match read_protocol_v2_request_view_with_timeout(
            &mut stream,
            DEFAULT_TIMEOUT,
            DEFAULT_MAX_FRAME_BYTES,
            &mut read_buffer,
        )
        .await
        {
            Ok(request) => request,
            Err(err) if is_connection_closed(&err) => return Ok(()),
            Err(err) => return Err(err),
        };
        tracing::info!(
            request_id = request.header.request_id,
            layer_id = request.header.layer_id,
            hidden_dim = request.header.hidden_dim,
            row_count = request.header.row_count,
            route_count = request.header.route_count,
            logical_payload_bytes = request.wire_stats().logical_payload_bytes,
            wire_bytes = request.wire_stats().wire_bytes,
            executor = executor.name(),
            "ProtocolV2 expert request received"
        );
        let timing_enabled = protocol_v2_tcp_timing_enabled();
        let execute_started = Instant::now();
        let response = executor.execute_with_identity(&request)?;
        let execute_ms = elapsed_ms(execute_started);
        let write_started = Instant::now();
        write_protocol_v2_response_with_timeout(
            &mut stream,
            &response,
            DEFAULT_TIMEOUT,
            DEFAULT_MAX_FRAME_BYTES,
            &mut write_buffer,
        )
        .await?;
        let write_ms = elapsed_ms(write_started);
        if timing_enabled {
            eprintln!(
                "protocol_v2_expert_server_roundtrip_timing request_id={} layer_id={} rows={} routes={} request_wire_bytes={} response_wire_bytes={} executor={} execute_ms={:.3} write_ms={:.3}",
                request.header.request_id,
                request.header.layer_id,
                request.header.row_count,
                request.header.route_count,
                request.wire_stats().wire_bytes,
                response.wire_stats().wire_bytes,
                executor.name(),
                execute_ms,
                write_ms
            );
            tracing::info!(
                request_id = request.header.request_id,
                layer_id = request.header.layer_id,
                rows = request.header.row_count,
                routes = request.header.route_count,
                request_wire_bytes = request.wire_stats().wire_bytes,
                response_wire_bytes = response.wire_stats().wire_bytes,
                executor = executor.name(),
                execute_ms = execute_ms,
                write_ms = write_ms,
                "protocol_v2_expert_server_roundtrip_timing"
            );
        }
    }
}

fn protocol_v2_tcp_timing_enabled() -> bool {
    env::var(PROTOCOL_V2_TCP_TIMING_ENV)
        .map(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

async fn write_protocol_v2_request_with_timeout<W>(
    writer: &mut W,
    request: &ExpertProtocolV2Request,
    timeout_duration: Duration,
    max_frame_bytes: usize,
    frame_buffer: &mut ExpertProtocolV2FrameBuffer,
) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    let prefix = frame_buffer.encode_request_prefix(request)?;
    write_protocol_v2_frame_parts_with_timeout(
        writer,
        &[prefix, request.hidden_payload.as_slice()],
        timeout_duration,
        max_frame_bytes,
    )
    .await
}

async fn write_protocol_v2_response_with_timeout<W>(
    writer: &mut W,
    response: &ExpertProtocolV2Response,
    timeout_duration: Duration,
    max_frame_bytes: usize,
    frame_buffer: &mut ExpertProtocolV2FrameBuffer,
) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    let prefix = frame_buffer.encode_response_prefix(response)?;
    write_protocol_v2_frame_parts_with_timeout(
        writer,
        &[prefix, response.partial_output_payload.as_slice()],
        timeout_duration,
        max_frame_bytes,
    )
    .await
}

async fn write_protocol_v2_frame_parts_with_timeout<W>(
    writer: &mut W,
    frame_parts: &[&[u8]],
    timeout_duration: Duration,
    max_frame_bytes: usize,
) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    let frame_len = frame_parts.iter().try_fold(0_usize, |acc, part| {
        acc.checked_add(part.len())
            .context("ProtocolV2 frame length overflows usize")
    })?;
    if frame_len > max_frame_bytes {
        bail!(
            "ProtocolV2 frame length {} exceeds max frame bytes {}",
            frame_len,
            max_frame_bytes
        );
    }
    let mut remaining_parts = frame_parts
        .iter()
        .filter(|part| !part.is_empty())
        .copied()
        .collect::<Vec<_>>();
    while !remaining_parts.is_empty() {
        let slices = remaining_parts
            .iter()
            .map(|part| IoSlice::new(part))
            .collect::<Vec<_>>();
        let bytes_written = timeout(timeout_duration, writer.write_vectored(&slices))
            .await
            .context("timed out writing ProtocolV2 frame")?
            .context("writing ProtocolV2 frame")?;
        if bytes_written == 0 {
            bail!("ProtocolV2 frame write returned zero bytes");
        }
        consume_written_parts(&mut remaining_parts, bytes_written);
    }
    timeout(timeout_duration, writer.flush())
        .await
        .context("timed out flushing ProtocolV2 frame")?
        .context("flushing ProtocolV2 frame")?;
    Ok(())
}

fn consume_written_parts<'a>(parts: &mut Vec<&'a [u8]>, mut bytes_written: usize) {
    while !parts.is_empty() && bytes_written >= parts[0].len() {
        bytes_written -= parts[0].len();
        parts.remove(0);
    }
    if bytes_written > 0 && !parts.is_empty() {
        parts[0] = &parts[0][bytes_written..];
    }
}

async fn read_protocol_v2_request_view_with_timeout<'a, R>(
    reader: &mut R,
    timeout_duration: Duration,
    max_frame_bytes: usize,
    frame_buffer: &'a mut ExpertProtocolV2FrameBuffer,
) -> Result<ExpertProtocolV2RequestView<'a>>
where
    R: AsyncRead + Unpin,
{
    read_protocol_v2_frame_with_timeout(
        reader,
        EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN,
        ExpertProtocolV2Request::wire_bytes_from_header,
        timeout_duration,
        max_frame_bytes,
        "request",
        frame_buffer,
    )
    .await?;
    ExpertProtocolV2RequestView::parse(frame_buffer.as_slice())
}

async fn read_protocol_v2_response_frame_with_timeout<R>(
    reader: &mut R,
    timeout_duration: Duration,
    max_frame_bytes: usize,
    frame_buffer: &mut ExpertProtocolV2FrameBuffer,
) -> Result<()>
where
    R: AsyncRead + Unpin,
{
    read_protocol_v2_frame_with_timeout(
        reader,
        EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN,
        ExpertProtocolV2Response::wire_bytes_from_header,
        timeout_duration,
        max_frame_bytes,
        "response",
        frame_buffer,
    )
    .await
}

async fn read_protocol_v2_frame_with_timeout<R, F>(
    reader: &mut R,
    header_len: usize,
    wire_bytes_from_header: F,
    timeout_duration: Duration,
    max_frame_bytes: usize,
    label: &'static str,
    frame_buffer: &mut ExpertProtocolV2FrameBuffer,
) -> Result<()>
where
    R: AsyncRead + Unpin,
    F: Fn(&[u8]) -> Result<usize>,
{
    let frame = frame_buffer.bytes_mut();
    frame.clear();
    frame.resize(header_len, 0);
    timeout(timeout_duration, reader.read_exact(frame.as_mut_slice()))
        .await
        .with_context(|| format!("timed out reading ProtocolV2 {label} header"))?
        .with_context(|| format!("reading ProtocolV2 {label} header"))?;
    let wire_bytes = wire_bytes_from_header(frame.as_slice())?;
    if wire_bytes < header_len {
        bail!("ProtocolV2 {label} wire bytes {wire_bytes} smaller than header length {header_len}");
    }
    if wire_bytes > max_frame_bytes {
        bail!(
            "ProtocolV2 {label} wire bytes {} exceeds max frame bytes {}",
            wire_bytes,
            max_frame_bytes
        );
    }
    frame.resize(wire_bytes, 0);
    timeout(
        timeout_duration,
        reader.read_exact(&mut frame[header_len..]),
    )
    .await
    .with_context(|| format!("timed out reading ProtocolV2 {label} payload"))?
    .with_context(|| format!("reading ProtocolV2 {label} payload"))?;
    Ok(())
}
