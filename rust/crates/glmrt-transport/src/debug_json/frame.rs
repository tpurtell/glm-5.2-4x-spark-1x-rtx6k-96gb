use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncRead, AsyncReadExt};

const FRAME_MAGIC: &[u8; 8] = b"GLMRTF01";
const FRAME_VERSION: u16 = 1;
const HEADER_LEN: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FrameKind {
    Request = 1,
    Response = 2,
}

impl FrameKind {
    fn from_u16(value: u16) -> Result<Self> {
        match value {
            1 => Ok(Self::Request),
            2 => Ok(Self::Response),
            other => bail!("unknown frame kind {other}"),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TransportFrame {
    pub(crate) kind: FrameKind,
    pub(crate) request_id: u64,
    pub(crate) payload: Vec<u8>,
}

pub(crate) fn encode_frame(
    kind: FrameKind,
    request_id: u64,
    payload: Vec<u8>,
    max_frame_bytes: usize,
) -> Result<Vec<u8>> {
    if payload.len() > max_frame_bytes {
        bail!(
            "payload length {} exceeds max frame bytes {}",
            payload.len(),
            max_frame_bytes
        );
    }
    let mut frame = vec![0_u8; HEADER_LEN + payload.len()];
    frame[..8].copy_from_slice(FRAME_MAGIC);
    frame[8..10].copy_from_slice(&FRAME_VERSION.to_be_bytes());
    frame[10..12].copy_from_slice(&(kind as u16).to_be_bytes());
    frame[12..16].copy_from_slice(&0_u32.to_be_bytes());
    frame[16..24].copy_from_slice(&request_id.to_be_bytes());
    frame[24..32].copy_from_slice(&(payload.len() as u64).to_be_bytes());
    frame[32..64].copy_from_slice(&Sha256::digest(&payload));
    frame[HEADER_LEN..].copy_from_slice(&payload);
    Ok(frame)
}

pub(crate) async fn read_frame<R>(reader: &mut R, max_frame_bytes: usize) -> Result<TransportFrame>
where
    R: AsyncRead + Unpin,
{
    let mut header = [0_u8; HEADER_LEN];
    reader
        .read_exact(&mut header)
        .await
        .context("reading transport frame header")?;
    if &header[..8] != FRAME_MAGIC {
        bail!("invalid transport frame magic");
    }
    let version = u16::from_be_bytes(header[8..10].try_into().unwrap());
    if version != FRAME_VERSION {
        bail!("unsupported transport frame version {version}");
    }
    let kind = FrameKind::from_u16(u16::from_be_bytes(header[10..12].try_into().unwrap()))?;
    let request_id = u64::from_be_bytes(header[16..24].try_into().unwrap());
    let payload_len = u64::from_be_bytes(header[24..32].try_into().unwrap()) as usize;
    if payload_len > max_frame_bytes {
        bail!("payload length {payload_len} exceeds max frame bytes {max_frame_bytes}");
    }
    let expected_hash = &header[32..64];
    let mut payload = vec![0_u8; payload_len];
    reader
        .read_exact(&mut payload)
        .await
        .context("reading transport frame payload")?;
    let actual_hash = Sha256::digest(&payload);
    if actual_hash.as_slice() != expected_hash {
        bail!("transport frame checksum mismatch");
    }
    Ok(TransportFrame {
        kind,
        request_id,
        payload,
    })
}
