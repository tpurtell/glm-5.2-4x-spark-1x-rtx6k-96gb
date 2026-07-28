use anyhow::Result;
use glmrt_core::{LayerWaveMode, GLM52_HIDDEN_BF16_BYTES, GLM52_HIDDEN_SIZE};
use tokio::io::AsyncWriteExt;

use super::common::{request_with_hidden_len, request_with_rows, spawn_server};
use crate::{
    debug_json_tcp_roundtrip, encode_frame, inproc_roundtrip, read_frame,
    synthetic_expert_response, tcp_roundtrip, FrameKind, TcpTransportConfig,
    DEBUG_JSON_EXPERT_PROTOCOL_LABEL, DEBUG_JSON_EXPERT_PROTOCOL_VERSION,
    DEBUG_JSON_FRAME_PROTOCOL, DEFAULT_MAX_FRAME_BYTES,
};

#[test]
fn debug_json_expert_protocol_is_explicitly_debug_only() {
    assert_eq!(DEBUG_JSON_EXPERT_PROTOCOL_VERSION, 1);
    assert!(DEBUG_JSON_EXPERT_PROTOCOL_LABEL.contains("debug"));
    assert!(DEBUG_JSON_EXPERT_PROTOCOL_LABEL.contains("json"));
    assert!(DEBUG_JSON_EXPERT_PROTOCOL_LABEL.contains("f32"));
    assert!(DEBUG_JSON_FRAME_PROTOCOL.contains("debug"));
    assert!(DEBUG_JSON_FRAME_PROTOCOL.contains("json"));
    assert!(DEBUG_JSON_FRAME_PROTOCOL.contains("f32"));
}

#[tokio::test]
async fn debug_json_tcp_roundtrip_is_legacy_tcp_compatible() -> Result<()> {
    let (addr, shutdown) = spawn_server().await?;
    let request = request_with_hidden_len(41, 16);

    let debug = debug_json_tcp_roundtrip(addr, &request, TcpTransportConfig::default()).await?;
    let legacy = tcp_roundtrip(addr, &request, TcpTransportConfig::default()).await?;

    assert_eq!(serde_json::to_vec(&debug)?, serde_json::to_vec(&legacy)?);
    let _ = shutdown.send(());
    Ok(())
}

#[tokio::test]
async fn same_request_via_inproc_and_tcp_returns_byte_identical_response() -> Result<()> {
    let (addr, shutdown) = spawn_server().await?;
    let request = request_with_hidden_len(42, 16);
    let inproc = inproc_roundtrip(&request).await?;
    let tcp = tcp_roundtrip(addr, &request, TcpTransportConfig::default()).await?;
    assert_eq!(serde_json::to_vec(&tcp)?, serde_json::to_vec(&inproc)?);
    let _ = shutdown.send(());
    Ok(())
}

#[tokio::test]
async fn decode_shaped_glm_hidden_row_roundtrips() -> Result<()> {
    let (addr, shutdown) = spawn_server().await?;
    let request = request_with_rows(50, 1, GLM52_HIDDEN_SIZE, LayerWaveMode::Decode);
    assert_eq!(request.header().row_count, 1);
    assert_eq!(request.header().wave_mode, Some(LayerWaveMode::Decode));
    assert_eq!(GLM52_HIDDEN_BF16_BYTES, 12_288);

    let response = tcp_roundtrip(addr, &request, TcpTransportConfig::default()).await?;

    assert_eq!(response.partial_outputs.len(), 1);
    assert_eq!(response.partial_outputs[0].len(), GLM52_HIDDEN_SIZE);
    let _ = shutdown.send(());
    Ok(())
}

#[tokio::test]
async fn mtp_shaped_layerwave_rows_roundtrip() -> Result<()> {
    let (addr, shutdown) = spawn_server().await?;
    let draft_rows = 4;
    let request = request_with_rows(51, draft_rows, GLM52_HIDDEN_SIZE, LayerWaveMode::MtpVerify);
    assert_eq!(
        request.rows.len() * GLM52_HIDDEN_BF16_BYTES,
        draft_rows * 12_288
    );
    assert_eq!(request.header().wave_mode, Some(LayerWaveMode::MtpVerify));

    let inproc = inproc_roundtrip(&request).await?;
    let tcp = tcp_roundtrip(addr, &request, TcpTransportConfig::default()).await?;

    assert_eq!(tcp.partial_outputs.len(), draft_rows);
    assert_eq!(serde_json::to_vec(&tcp)?, serde_json::to_vec(&inproc)?);
    let _ = shutdown.send(());
    Ok(())
}

#[tokio::test]
async fn prefill_shaped_layerwave_rows_roundtrip() -> Result<()> {
    let (addr, shutdown) = spawn_server().await?;
    let prefill_rows = 16;
    let request = request_with_rows(52, prefill_rows, GLM52_HIDDEN_SIZE, LayerWaveMode::Prefill);
    assert_eq!(
        request.rows.len() * GLM52_HIDDEN_BF16_BYTES,
        prefill_rows * 12_288
    );
    assert_eq!(request.header().wave_mode, Some(LayerWaveMode::Prefill));
    assert_eq!(
        request.header().graph_bucket_rows,
        Some(prefill_rows as u32)
    );
    assert_eq!(
        request.header().logical_bf16_payload_bytes,
        Some(prefill_rows * GLM52_HIDDEN_BF16_BYTES)
    );

    let inproc = inproc_roundtrip(&request).await?;
    let tcp = tcp_roundtrip(addr, &request, TcpTransportConfig::default()).await?;

    assert_eq!(tcp.partial_outputs.len(), prefill_rows);
    assert_eq!(tcp.partial_outputs[0].len(), GLM52_HIDDEN_SIZE);
    assert_eq!(serde_json::to_vec(&tcp)?, serde_json::to_vec(&inproc)?);
    let _ = shutdown.send(());
    Ok(())
}

#[test]
fn malformed_wave_metadata_is_rejected() {
    let mut request = request_with_rows(53, 4, GLM52_HIDDEN_SIZE, LayerWaveMode::Prefill);
    request
        .wave
        .as_mut()
        .expect("test request has wave metadata")
        .logical_bf16_payload_bytes = GLM52_HIDDEN_BF16_BYTES;

    let err = synthetic_expert_response(&request).unwrap_err().to_string();

    assert!(err.contains("logical BF16 payload bytes"));
}

#[tokio::test]
async fn checksum_mismatch_is_rejected() -> Result<()> {
    let payload = serde_json::to_vec(&request_with_hidden_len(99, 4))?;
    let mut frame = encode_frame(FrameKind::Request, 99, payload, DEFAULT_MAX_FRAME_BYTES)?;
    let last = frame
        .last_mut()
        .ok_or_else(|| anyhow::anyhow!("empty frame"))?;
    *last ^= 0x01;
    let (mut client, mut server) = tokio::io::duplex(frame.len());
    client.write_all(&frame).await?;
    drop(client);
    let err = read_frame(&mut server, DEFAULT_MAX_FRAME_BYTES)
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("checksum mismatch"));
    Ok(())
}
