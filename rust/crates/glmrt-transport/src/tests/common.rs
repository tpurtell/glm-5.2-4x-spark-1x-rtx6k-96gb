use anyhow::Result;
use glmrt_core::{
    ExpertRequest, ExpertRow, ExpertWaveMetadata, LayerWaveMode, RouteEntry,
    GLM52_HIDDEN_BF16_BYTES, GLM52_HIDDEN_SIZE,
};
use std::net::SocketAddr;
use tokio::net::TcpListener;
use tokio::sync::oneshot;

use crate::{
    handle_protocol_v2_synthetic_connection, handle_synthetic_connection, ExpertProtocolV2Request,
    ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor, ExpertV2Dtype, ExpertV2SourceKind,
};

pub(super) fn request_with_hidden_len(request_id: u64, hidden_len: usize) -> ExpertRequest {
    ExpertRequest {
        protocol_version: 1,
        request_id,
        placement_version: "test-placement".to_owned(),
        layer_id: 7,
        hidden_dim: hidden_len as u32,
        wave: None,
        rows: vec![
            ExpertRow {
                row_id: 0,
                hidden: (0..hidden_len).map(|idx| idx as f32 * 0.25).collect(),
                routes: vec![
                    RouteEntry {
                        expert_id: 3,
                        gate: 0.25,
                    },
                    RouteEntry {
                        expert_id: 9,
                        gate: 0.75,
                    },
                ],
            },
            ExpertRow {
                row_id: 1,
                hidden: (0..hidden_len).map(|idx| 1.0 + idx as f32 * 0.5).collect(),
                routes: vec![RouteEntry {
                    expert_id: 11,
                    gate: 1.0,
                }],
            },
        ],
    }
}

pub(super) fn request_with_rows(
    request_id: u64,
    row_count: usize,
    hidden_len: usize,
    mode: LayerWaveMode,
) -> ExpertRequest {
    ExpertRequest {
        protocol_version: 1,
        request_id,
        placement_version: "test-layerwave-placement".to_owned(),
        layer_id: 3,
        hidden_dim: hidden_len as u32,
        wave: Some(ExpertWaveMetadata {
            mode,
            graph_bucket_rows: row_count as u32,
            logical_bf16_payload_bytes: row_count * GLM52_HIDDEN_BF16_BYTES,
        }),
        rows: (0..row_count)
            .map(|row_id| ExpertRow {
                row_id: row_id as u64,
                hidden: (0..hidden_len)
                    .map(|idx| ((row_id + idx) % 257) as f32 / 257.0)
                    .collect(),
                routes: vec![
                    RouteEntry {
                        expert_id: (row_id % 256) as u32,
                        gate: 0.625,
                    },
                    RouteEntry {
                        expert_id: ((row_id + 17) % 256) as u32,
                        gate: 0.375,
                    },
                ],
            })
            .collect(),
    }
}

pub(super) async fn spawn_server() -> Result<(SocketAddr, oneshot::Sender<()>)> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let (shutdown_tx, mut shutdown_rx) = oneshot::channel::<()>();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = &mut shutdown_rx => break,
                accepted = listener.accept() => {
                    let Ok((stream, _peer)) = accepted else {
                        break;
                    };
                    tokio::spawn(async move {
                        let _ = handle_synthetic_connection(stream).await;
                    });
                }
            }
        }
    });
    Ok((addr, shutdown_tx))
}

pub(super) fn protocol_v2_request(
    request_id: u64,
    row_count: usize,
    source_kind: ExpertV2SourceKind,
) -> Result<ExpertProtocolV2Request> {
    let routes_per_row = 2;
    let mut rows = Vec::with_capacity(row_count);
    let mut routes = Vec::with_capacity(row_count * routes_per_row);
    for row in 0..row_count {
        rows.push(ExpertProtocolV2RowDescriptor {
            row_id: row as u64,
            source_kind,
            source_request_id: request_id + row as u64,
            token_position: row as u64,
            route_offset: (row * routes_per_row) as u32,
            route_count: routes_per_row as u32,
        });
        routes.push(ExpertProtocolV2RouteEntry {
            row_index: row as u32,
            expert_id: (row % 256) as u32,
            gate_weight: 0.625,
        });
        routes.push(ExpertProtocolV2RouteEntry {
            row_index: row as u32,
            expert_id: ((row + 17) % 256) as u32,
            gate_weight: 0.375,
        });
    }
    let mut hidden_payload = Vec::with_capacity(row_count * GLM52_HIDDEN_BF16_BYTES);
    for row in 0..row_count {
        for col in 0..GLM52_HIDDEN_SIZE {
            let value = ((row * 17 + col) % 257) as f32 / 257.0 - 0.5;
            let bf16 = (value.to_bits() >> 16) as u16;
            hidden_payload.extend_from_slice(&bf16.to_le_bytes());
        }
    }
    ExpertProtocolV2Request::new(
        request_id,
        0x51CE,
        13,
        GLM52_HIDDEN_SIZE as u32,
        ExpertV2Dtype::Bf16,
        rows,
        routes,
        hidden_payload,
    )
}

pub(super) async fn spawn_protocol_v2_server() -> Result<(SocketAddr, oneshot::Sender<()>)> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let (shutdown_tx, mut shutdown_rx) = oneshot::channel::<()>();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = &mut shutdown_rx => break,
                accepted = listener.accept() => {
                    let Ok((stream, _peer)) = accepted else {
                        break;
                    };
                    tokio::spawn(async move {
                        let _ = handle_protocol_v2_synthetic_connection(stream).await;
                    });
                }
            }
        }
    });
    Ok((addr, shutdown_tx))
}
