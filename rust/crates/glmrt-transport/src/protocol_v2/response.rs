use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};

use super::{
    checked_u32, checked_usize, debug_checksum_enabled, expert_protocol_v2_compact_id, push_u16,
    push_u32, push_u64, read_u16, read_u32, read_u64, response_header_len_from_flags,
    response_more_chunks_enabled, response_row_indices_enabled, validate_common_header,
    validate_flags, validate_header_len, ExpertProtocolV2DeviceResponseRef,
    ExpertProtocolV2Response, ExpertProtocolV2ResponseHeader, ExpertProtocolV2ResponseRef,
    ExpertProtocolV2Status, ExpertProtocolV2WireStats, ExpertV2Dtype, CHECKSUM_LEN,
    EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM, EXPERT_PROTOCOL_V2_FLAG_RESPONSE_MORE_CHUNKS,
    EXPERT_PROTOCOL_V2_FLAG_RESPONSE_ROW_INDICES, EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN, MAGIC,
    RESPONSE_CHECKSUM_OFFSET, RESPONSE_EXECUTOR_ID_OFFSET, RESPONSE_KIND, VERSION,
};

impl ExpertProtocolV2Response {
    pub fn new(
        request_id: u64,
        placement_version: u64,
        layer_id: u32,
        row_count: u32,
        output_dim: u32,
        output_dtype: ExpertV2Dtype,
        status: ExpertProtocolV2Status,
        partial_output_payload: Vec<u8>,
    ) -> Result<Self> {
        let output_row_stride_bytes = default_output_row_stride_bytes(output_dim, output_dtype)?;
        Self::new_with_output_stride(
            request_id,
            placement_version,
            layer_id,
            row_count,
            output_dim,
            output_dtype,
            output_row_stride_bytes,
            status,
            partial_output_payload,
        )
    }

    pub fn new_with_output_stride(
        request_id: u64,
        placement_version: u64,
        layer_id: u32,
        row_count: u32,
        output_dim: u32,
        output_dtype: ExpertV2Dtype,
        output_row_stride_bytes: u32,
        status: ExpertProtocolV2Status,
        partial_output_payload: Vec<u8>,
    ) -> Result<Self> {
        let header = ExpertProtocolV2ResponseHeader {
            request_id,
            placement_version,
            layer_id,
            row_count,
            output_dim,
            output_dtype,
            output_row_stride_bytes,
            output_payload_bytes: partial_output_payload.len() as u64,
            status,
            flags: 0,
            executor_id: 0,
        };
        let response = Self {
            header,
            row_indices: None,
            partial_output_payload,
        };
        response.validate()?;
        Ok(response)
    }

    pub fn with_debug_checksum(mut self) -> Self {
        self.header.flags |= EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM;
        self
    }

    pub fn with_executor_id(mut self, executor_id: u64) -> Self {
        self.header.executor_id = executor_id;
        self
    }

    pub fn with_executor_name(self, executor_name: &str) -> Self {
        self.with_executor_id(expert_protocol_v2_compact_id(executor_name))
    }

    pub fn with_row_indices(mut self, row_indices: Vec<u32>, more_chunks: bool) -> Result<Self> {
        self.header.flags |= EXPERT_PROTOCOL_V2_FLAG_RESPONSE_ROW_INDICES;
        if more_chunks {
            self.header.flags |= EXPERT_PROTOCOL_V2_FLAG_RESPONSE_MORE_CHUNKS;
        } else {
            self.header.flags &= !EXPERT_PROTOCOL_V2_FLAG_RESPONSE_MORE_CHUNKS;
        }
        self.row_indices = Some(row_indices);
        self.validate()?;
        Ok(self)
    }

    pub fn row_indexed(&self) -> bool {
        response_row_indices_enabled(self.header.flags)
    }

    pub fn more_chunks(&self) -> bool {
        response_more_chunks_enabled(self.header.flags)
    }

    pub fn debug_checksum_enabled(&self) -> bool {
        debug_checksum_enabled(self.header.flags)
    }

    pub fn header_len(&self) -> usize {
        response_header_len_from_flags(self.header.flags)
    }

    pub fn as_borrowed(&self) -> ExpertProtocolV2ResponseRef<'_> {
        ExpertProtocolV2ResponseRef {
            header: self.header.clone(),
            row_indices: self.row_indices.as_deref(),
            partial_output_payload: &self.partial_output_payload,
        }
    }

    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        self.encode_into(&mut out)?;
        Ok(out)
    }

    pub fn encode_into(&self, out: &mut Vec<u8>) -> Result<()> {
        self.encode_prefix_into(out)?;
        out.extend_from_slice(&self.partial_output_payload);
        Ok(())
    }

    pub(crate) fn encode_prefix_into(&self, out: &mut Vec<u8>) -> Result<()> {
        self.validate()?;
        let header_len = self.header_len();
        let wire_bytes = self.wire_stats().wire_bytes;
        let prefix_bytes = wire_bytes
            .checked_sub(self.partial_output_payload.len())
            .context("ExpertProtocolV2 response prefix byte count underflows")?;
        out.clear();
        if out.capacity() < prefix_bytes {
            out.reserve(prefix_bytes - out.capacity());
        }
        out.extend_from_slice(MAGIC);
        push_u16(out, VERSION);
        push_u16(out, RESPONSE_KIND);
        push_u32(out, header_len as u32);
        push_u64(out, self.header.request_id);
        push_u64(out, self.header.placement_version);
        push_u32(out, self.header.layer_id);
        push_u32(out, self.header.row_count);
        push_u16(out, self.header.output_dtype as u16);
        push_u16(out, self.header.status as u16);
        push_u64(out, self.header.output_payload_bytes);
        push_u64(out, self.wire_stats().logical_payload_bytes as u64);
        push_u64(out, wire_bytes as u64);
        push_u32(out, self.header.flags);
        push_u32(out, self.header.output_dim);
        push_u32(out, self.header.output_row_stride_bytes);
        push_u64(out, self.header.executor_id);
        if self.debug_checksum_enabled() {
            out.extend_from_slice(&self.frame_content_checksum());
        }
        out.resize(header_len, 0);
        if let Some(row_indices) = &self.row_indices {
            for row_index in row_indices {
                push_u32(out, *row_index);
            }
        }
        Ok(())
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN {
            bail!("ExpertProtocolV2 response frame too short: {}", bytes.len());
        }
        let header_len = validate_common_header(bytes, RESPONSE_KIND)?;
        let request_id = read_u64(bytes, 16, "request_id")?;
        let placement_version = read_u64(bytes, 24, "placement_version")?;
        let layer_id = read_u32(bytes, 32, "layer_id")?;
        let row_count = read_u32(bytes, 36, "row_count")?;
        let output_dtype = ExpertV2Dtype::from_u16(read_u16(bytes, 40, "output_dtype")?)?;
        let status = ExpertProtocolV2Status::from_u16(read_u16(bytes, 42, "status")?)?;
        let output_payload_bytes = read_u64(bytes, 44, "output_payload_bytes")?;
        let output_dim = read_u32(bytes, 72, "output_dim")?;
        let output_row_stride_bytes = read_u32(bytes, 76, "output_row_stride_bytes")?;
        let executor_id = read_u64(bytes, RESPONSE_EXECUTOR_ID_OFFSET, "executor_id")?;
        let wire_bytes = read_u64(bytes, 60, "wire_bytes")? as usize;
        let flags = read_u32(bytes, 68, "flags")?;
        validate_flags(flags, "response")?;
        validate_header_len(header_len, response_header_len_from_flags(flags))?;
        if bytes.len() < header_len {
            bail!(
                "ExpertProtocolV2 response frame shorter than declared header: frame={} header={header_len}",
                bytes.len()
            );
        }
        if wire_bytes != bytes.len() {
            bail!(
                "ExpertProtocolV2 response wire bytes mismatch: header={} actual={}",
                wire_bytes,
                bytes.len()
            );
        }
        let payload_len = checked_usize(output_payload_bytes, "output payload")?;
        let row_index_bytes = if response_row_indices_enabled(flags) {
            (row_count as usize)
                .checked_mul(std::mem::size_of::<u32>())
                .context("ExpertProtocolV2 response row index byte count overflow")?
        } else {
            0
        };
        let payload_start = header_len
            .checked_add(row_index_bytes)
            .context("ExpertProtocolV2 response payload offset overflow")?;
        let payload_end = payload_start + payload_len;
        if payload_end != bytes.len() {
            bail!(
                "ExpertProtocolV2 response length mismatch: sections end at {payload_end}, frame has {}",
                bytes.len()
            );
        }
        let row_indices = response_row_indices_enabled(flags).then(|| {
            bytes[header_len..payload_start]
                .chunks_exact(std::mem::size_of::<u32>())
                .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("u32 response row index")))
                .collect::<Vec<_>>()
        });
        let partial_output_payload = bytes[payload_start..payload_end].to_vec();
        if debug_checksum_enabled(flags) {
            let expected_checksum = &bytes[RESPONSE_CHECKSUM_OFFSET
                ..RESPONSE_CHECKSUM_OFFSET
                    .checked_add(CHECKSUM_LEN)
                    .expect("checksum range overflow")];
            let actual_checksum =
                frame_content_checksum(row_indices.as_deref(), &partial_output_payload);
            if actual_checksum.as_slice() != expected_checksum {
                bail!("ExpertProtocolV2 response checksum mismatch");
            }
        }

        let response = Self {
            header: ExpertProtocolV2ResponseHeader {
                request_id,
                placement_version,
                layer_id,
                row_count,
                output_dim,
                output_dtype,
                output_row_stride_bytes,
                output_payload_bytes,
                status,
                flags,
                executor_id,
            },
            row_indices,
            partial_output_payload,
        };
        response.validate()?;
        Ok(response)
    }

    pub fn wire_stats(&self) -> ExpertProtocolV2WireStats {
        ExpertProtocolV2WireStats {
            logical_payload_bytes: self.partial_output_payload.len(),
            wire_bytes: self.header_len()
                + self
                    .row_indices
                    .as_ref()
                    .map(|indices| indices.len() * 4)
                    .unwrap_or(0)
                + self.partial_output_payload.len(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        validate_response_parts(
            &self.header,
            self.row_indices.as_deref(),
            &self.partial_output_payload,
        )
    }

    pub fn wire_bytes_from_header(header: &[u8]) -> Result<usize> {
        if header.len() < EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN {
            bail!(
                "ExpertProtocolV2 response header too short: {}",
                header.len()
            );
        }
        let header_len = validate_common_header(header, RESPONSE_KIND)?;
        let flags = read_u32(header, 68, "flags")?;
        validate_flags(flags, "response")?;
        validate_header_len(header_len, response_header_len_from_flags(flags))?;
        checked_usize(read_u64(header, 60, "wire_bytes")?, "response wire bytes")
    }

    fn frame_content_checksum(&self) -> [u8; CHECKSUM_LEN] {
        frame_content_checksum(self.row_indices.as_deref(), &self.partial_output_payload).into()
    }
}

impl<'a> ExpertProtocolV2ResponseRef<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_output_stride(
        request_id: u64,
        placement_version: u64,
        layer_id: u32,
        row_count: u32,
        output_dim: u32,
        output_dtype: ExpertV2Dtype,
        output_row_stride_bytes: u32,
        status: ExpertProtocolV2Status,
        partial_output_payload: &'a [u8],
    ) -> Result<Self> {
        let response = Self {
            header: ExpertProtocolV2ResponseHeader {
                request_id,
                placement_version,
                layer_id,
                row_count,
                output_dim,
                output_dtype,
                output_row_stride_bytes,
                output_payload_bytes: partial_output_payload.len() as u64,
                status,
                flags: 0,
                executor_id: 0,
            },
            row_indices: None,
            partial_output_payload,
        };
        response.validate()?;
        Ok(response)
    }

    pub fn with_debug_checksum(mut self) -> Self {
        self.header.flags |= EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM;
        self
    }

    pub fn with_executor_id(mut self, executor_id: u64) -> Self {
        self.header.executor_id = executor_id;
        self
    }

    pub fn with_executor_name(self, executor_name: &str) -> Self {
        self.with_executor_id(expert_protocol_v2_compact_id(executor_name))
    }

    pub fn with_row_indices(mut self, row_indices: &'a [u32], more_chunks: bool) -> Result<Self> {
        self.header.flags |= EXPERT_PROTOCOL_V2_FLAG_RESPONSE_ROW_INDICES;
        if more_chunks {
            self.header.flags |= EXPERT_PROTOCOL_V2_FLAG_RESPONSE_MORE_CHUNKS;
        } else {
            self.header.flags &= !EXPERT_PROTOCOL_V2_FLAG_RESPONSE_MORE_CHUNKS;
        }
        self.row_indices = Some(row_indices);
        self.validate()?;
        Ok(self)
    }

    pub fn row_indexed(&self) -> bool {
        response_row_indices_enabled(self.header.flags)
    }

    pub fn more_chunks(&self) -> bool {
        response_more_chunks_enabled(self.header.flags)
    }

    pub fn debug_checksum_enabled(&self) -> bool {
        debug_checksum_enabled(self.header.flags)
    }

    pub fn header_len(&self) -> usize {
        response_header_len_from_flags(self.header.flags)
    }

    pub fn wire_stats(&self) -> ExpertProtocolV2WireStats {
        ExpertProtocolV2WireStats {
            logical_payload_bytes: self.partial_output_payload.len(),
            wire_bytes: self.header_len()
                + self
                    .row_indices
                    .map(|indices| indices.len() * std::mem::size_of::<u32>())
                    .unwrap_or(0)
                + self.partial_output_payload.len(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        validate_response_parts(&self.header, self.row_indices, self.partial_output_payload)
    }

    pub(crate) fn encode_prefix_into(&self, out: &mut Vec<u8>) -> Result<()> {
        self.validate()?;
        let header_len = self.header_len();
        let wire_bytes = self.wire_stats().wire_bytes;
        let prefix_bytes = wire_bytes
            .checked_sub(self.partial_output_payload.len())
            .context("ExpertProtocolV2 borrowed response prefix byte count underflows")?;
        out.clear();
        if out.capacity() < prefix_bytes {
            out.reserve(prefix_bytes - out.capacity());
        }
        out.extend_from_slice(MAGIC);
        push_u16(out, VERSION);
        push_u16(out, RESPONSE_KIND);
        push_u32(out, header_len as u32);
        push_u64(out, self.header.request_id);
        push_u64(out, self.header.placement_version);
        push_u32(out, self.header.layer_id);
        push_u32(out, self.header.row_count);
        push_u16(out, self.header.output_dtype as u16);
        push_u16(out, self.header.status as u16);
        push_u64(out, self.header.output_payload_bytes);
        push_u64(out, self.wire_stats().logical_payload_bytes as u64);
        push_u64(out, wire_bytes as u64);
        push_u32(out, self.header.flags);
        push_u32(out, self.header.output_dim);
        push_u32(out, self.header.output_row_stride_bytes);
        push_u64(out, self.header.executor_id);
        if self.debug_checksum_enabled() {
            out.extend_from_slice(
                frame_content_checksum(self.row_indices, self.partial_output_payload).as_slice(),
            );
        }
        out.resize(header_len, 0);
        if let Some(row_indices) = self.row_indices {
            for row_index in row_indices {
                push_u32(out, *row_index);
            }
        }
        Ok(())
    }

    pub fn to_owned(&self) -> Result<ExpertProtocolV2Response> {
        self.validate()?;
        Ok(ExpertProtocolV2Response {
            header: self.header.clone(),
            row_indices: self.row_indices.map(<[u32]>::to_vec),
            partial_output_payload: self.partial_output_payload.to_vec(),
        })
    }
}

impl<'a> ExpertProtocolV2DeviceResponseRef<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_output_stride(
        request_id: u64,
        placement_version: u64,
        layer_id: u32,
        row_count: u32,
        output_dim: u32,
        output_dtype: ExpertV2Dtype,
        output_row_stride_bytes: u32,
        status: ExpertProtocolV2Status,
        partial_output_payload: glmrt_ffi::GlmrtDeviceBuffer,
    ) -> Result<Self> {
        let response = Self {
            header: ExpertProtocolV2ResponseHeader {
                request_id,
                placement_version,
                layer_id,
                row_count,
                output_dim,
                output_dtype,
                output_row_stride_bytes,
                output_payload_bytes: partial_output_payload.bytes as u64,
                status,
                flags: 0,
                executor_id: 0,
            },
            row_indices: None,
            partial_output_payload,
        };
        response.validate()?;
        Ok(response)
    }

    pub fn with_executor_id(mut self, executor_id: u64) -> Self {
        self.header.executor_id = executor_id;
        self
    }

    pub fn with_executor_name(self, executor_name: &str) -> Self {
        self.with_executor_id(expert_protocol_v2_compact_id(executor_name))
    }

    pub fn with_row_indices(mut self, row_indices: &'a [u32], more_chunks: bool) -> Result<Self> {
        self.header.flags |= EXPERT_PROTOCOL_V2_FLAG_RESPONSE_ROW_INDICES;
        if more_chunks {
            self.header.flags |= EXPERT_PROTOCOL_V2_FLAG_RESPONSE_MORE_CHUNKS;
        } else {
            self.header.flags &= !EXPERT_PROTOCOL_V2_FLAG_RESPONSE_MORE_CHUNKS;
        }
        self.row_indices = Some(row_indices);
        self.validate()?;
        Ok(self)
    }

    pub fn more_chunks(&self) -> bool {
        response_more_chunks_enabled(self.header.flags)
    }

    pub fn header_len(&self) -> usize {
        response_header_len_from_flags(self.header.flags)
    }

    pub fn wire_stats(&self) -> ExpertProtocolV2WireStats {
        ExpertProtocolV2WireStats {
            logical_payload_bytes: self.partial_output_payload.bytes,
            wire_bytes: self.header_len()
                + self
                    .row_indices
                    .map(|indices| indices.len() * std::mem::size_of::<u32>())
                    .unwrap_or(0)
                + self.partial_output_payload.bytes,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.header.flags & EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM != 0 {
            bail!("device-backed ProtocolV2 responses do not support debug checksums");
        }
        if self.partial_output_payload.bytes > 0 && self.partial_output_payload.ptr.is_null() {
            bail!("device-backed ProtocolV2 response payload is null");
        }
        validate_response_shape(
            &self.header,
            self.row_indices,
            self.partial_output_payload.bytes,
        )
    }

    pub(crate) fn encode_prefix_into(&self, out: &mut Vec<u8>) -> Result<()> {
        self.validate()?;
        let header_len = self.header_len();
        let wire_bytes = self.wire_stats().wire_bytes;
        let prefix_bytes = wire_bytes
            .checked_sub(self.partial_output_payload.bytes)
            .context("ExpertProtocolV2 device response prefix byte count underflows")?;
        out.clear();
        if out.capacity() < prefix_bytes {
            out.reserve(prefix_bytes - out.capacity());
        }
        out.extend_from_slice(MAGIC);
        push_u16(out, VERSION);
        push_u16(out, RESPONSE_KIND);
        push_u32(out, header_len as u32);
        push_u64(out, self.header.request_id);
        push_u64(out, self.header.placement_version);
        push_u32(out, self.header.layer_id);
        push_u32(out, self.header.row_count);
        push_u16(out, self.header.output_dtype as u16);
        push_u16(out, self.header.status as u16);
        push_u64(out, self.header.output_payload_bytes);
        push_u64(out, self.wire_stats().logical_payload_bytes as u64);
        push_u64(out, wire_bytes as u64);
        push_u32(out, self.header.flags);
        push_u32(out, self.header.output_dim);
        push_u32(out, self.header.output_row_stride_bytes);
        push_u64(out, self.header.executor_id);
        out.resize(header_len, 0);
        if let Some(row_indices) = self.row_indices {
            for row_index in row_indices {
                push_u32(out, *row_index);
            }
        }
        Ok(())
    }
}

fn validate_response_parts(
    header: &ExpertProtocolV2ResponseHeader,
    row_indices: Option<&[u32]>,
    partial_output_payload: &[u8],
) -> Result<()> {
    validate_response_shape(header, row_indices, partial_output_payload.len())
}

fn validate_response_shape(
    header: &ExpertProtocolV2ResponseHeader,
    row_indices: Option<&[u32]>,
    payload_bytes: usize,
) -> Result<()> {
    validate_flags(header.flags, "response")?;
    let row_indexed = response_row_indices_enabled(header.flags);
    if row_indexed != row_indices.is_some() {
        bail!("ExpertProtocolV2 response row-index flag and row index table disagree");
    }
    if response_more_chunks_enabled(header.flags) && !row_indexed {
        bail!("ExpertProtocolV2 response more-chunks flag requires row indices");
    }
    if let Some(row_indices) = row_indices {
        if row_indices.len() != header.row_count as usize {
            bail!(
                "ExpertProtocolV2 response row index count mismatch: expected={} actual={}",
                header.row_count,
                row_indices.len()
            );
        }
        let mut sorted = row_indices.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != row_indices.len() {
            bail!("ExpertProtocolV2 response row indices must be unique within a chunk");
        }
    }
    if header.output_dim == 0 {
        bail!("ExpertProtocolV2 response output_dim must be non-zero");
    }
    let logical_row_bytes = header
        .output_dtype
        .row_bytes(header.output_dim as usize)
        .context("ExpertProtocolV2 response output row byte count")?;
    let output_row_stride_bytes = header.output_row_stride_bytes as usize;
    if output_row_stride_bytes < logical_row_bytes {
        bail!(
            "ExpertProtocolV2 response output row stride bytes {} smaller than logical row bytes {}",
            header.output_row_stride_bytes,
            logical_row_bytes
        );
    }
    let expected_payload = (header.row_count as usize)
        .checked_mul(output_row_stride_bytes)
        .context("ExpertProtocolV2 response output payload byte count overflow")?;
    if expected_payload != payload_bytes {
        bail!(
            "ExpertProtocolV2 response output payload bytes mismatch: expected={} actual={}",
            expected_payload,
            payload_bytes
        );
    }
    if header.output_payload_bytes as usize != payload_bytes {
        bail!("ExpertProtocolV2 response output_payload_bytes mismatch");
    }
    Ok(())
}

fn frame_content_checksum(
    row_indices: Option<&[u32]>,
    payload: &[u8],
) -> sha2::digest::Output<Sha256> {
    let mut hasher = Sha256::new();
    if let Some(row_indices) = row_indices {
        for row_index in row_indices {
            hasher.update(row_index.to_le_bytes());
        }
    }
    hasher.update(payload);
    hasher.finalize()
}

fn default_output_row_stride_bytes(output_dim: u32, output_dtype: ExpertV2Dtype) -> Result<u32> {
    checked_u32(
        output_dtype
            .row_bytes(output_dim as usize)
            .context("output row stride byte count")?,
        "output row stride bytes",
    )
}
