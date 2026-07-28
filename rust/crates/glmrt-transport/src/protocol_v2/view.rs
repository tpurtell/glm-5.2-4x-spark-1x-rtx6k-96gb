use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};

use super::{
    checked_usize, debug_checksum_enabled, decode_route_entry, decode_row_descriptor,
    layer_block_enabled, payload_checksum, precompile_warmup_enabled, read_u16, read_u32, read_u64,
    request_header_len_from_flags, response_fp8_e4m3_row_scaled_enabled,
    response_header_len_from_flags, response_more_chunks_enabled,
    response_nvfp4_e2m1_fp8_e4m3_enabled, response_row_indices_enabled,
    spark_collective_part_count, spark_reduction_enabled, spark_row_sharded_reduction_enabled,
    stream_data_enabled, stream_final_enabled, stream_plan_enabled, validate_common_header,
    validate_flags, validate_header_len, ExpertProtocolV2RequestHeader,
    ExpertProtocolV2ResponseHeader, ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor,
    ExpertProtocolV2Status, ExpertProtocolV2WireStats, ExpertV2Dtype, CHECKSUM_LEN,
    EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM, EXPERT_PROTOCOL_V2_FLAG_RESPONSE_FP8_E4M3_ROW_SCALED,
    EXPERT_PROTOCOL_V2_FLAG_RESPONSE_NVFP4_E2M1_FP8_E4M3, EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN,
    EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN, EXPERT_PROTOCOL_V2_ROUTE_ENTRY_LEN,
    EXPERT_PROTOCOL_V2_ROW_DESCRIPTOR_LEN, REQUEST_CHECKSUM_OFFSET, REQUEST_KIND,
    RESPONSE_CHECKSUM_OFFSET, RESPONSE_EXECUTOR_ID_OFFSET, RESPONSE_KIND,
};

#[derive(Debug, Clone)]
pub struct ExpertProtocolV2RequestView<'a> {
    pub header: ExpertProtocolV2RequestHeader,
    frame_bytes: &'a [u8],
    row_descriptor_bytes: &'a [u8],
    route_bytes: &'a [u8],
    hidden_payload: &'a [u8],
    checksum: Option<&'a [u8]>,
}

impl<'a> ExpertProtocolV2RequestView<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() < EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN {
            bail!("ExpertProtocolV2 request frame too short: {}", bytes.len());
        }
        let header_len = validate_common_header(bytes, REQUEST_KIND)?;
        let row_count = read_u32(bytes, 36, "row_count")?;
        let route_count = read_u32(bytes, 48, "route_count")?;
        let row_descriptor_bytes = read_u32(bytes, 52, "row_descriptor_bytes")?;
        let route_bytes = read_u32(bytes, 56, "route_bytes")?;
        let hidden_payload_bytes = read_u64(bytes, 60, "hidden_payload_bytes")?;
        let wire_bytes = read_u64(bytes, 76, "wire_bytes")? as usize;
        let flags = read_u32(bytes, 84, "flags")?;
        validate_flags(flags, "request")?;
        validate_header_len(header_len, request_header_len_from_flags(flags))?;
        if bytes.len() < header_len {
            bail!(
                "ExpertProtocolV2 request frame shorter than declared header: frame={} header={header_len}",
                bytes.len()
            );
        }
        if wire_bytes != bytes.len() {
            bail!(
                "ExpertProtocolV2 request wire bytes mismatch: header={} actual={}",
                wire_bytes,
                bytes.len()
            );
        }

        let rows_len = if stream_data_enabled(flags) {
            0
        } else {
            row_count as usize
        };
        let routes_len = if stream_data_enabled(flags) {
            0
        } else {
            route_count as usize
        };
        let row_descriptor_bytes_usize = row_descriptor_bytes as usize;
        let route_bytes_usize = route_bytes as usize;
        let hidden_payload_bytes_usize = checked_usize(hidden_payload_bytes, "hidden payload")?;
        let expected_rows_bytes = rows_len
            .checked_mul(EXPERT_PROTOCOL_V2_ROW_DESCRIPTOR_LEN)
            .context("row descriptor byte count overflow")?;
        let expected_route_bytes = routes_len
            .checked_mul(EXPERT_PROTOCOL_V2_ROUTE_ENTRY_LEN)
            .context("route byte count overflow")?;
        if row_descriptor_bytes_usize != expected_rows_bytes {
            bail!(
                "ExpertProtocolV2 request row descriptor bytes mismatch: header={} expected={expected_rows_bytes}",
                row_descriptor_bytes_usize
            );
        }
        if route_bytes_usize != expected_route_bytes {
            bail!(
                "ExpertProtocolV2 request route bytes mismatch: header={} expected={expected_route_bytes}",
                route_bytes_usize
            );
        }

        let rows_start = header_len;
        let routes_start = rows_start + row_descriptor_bytes_usize;
        let payload_start = routes_start + route_bytes_usize;
        let frame_end = payload_start + hidden_payload_bytes_usize;
        if frame_end != bytes.len() {
            bail!(
                "ExpertProtocolV2 request length mismatch: sections end at {frame_end}, frame has {}",
                bytes.len()
            );
        }

        let view = Self {
            header: ExpertProtocolV2RequestHeader {
                request_id: read_u64(bytes, 16, "request_id")?,
                placement_version: read_u64(bytes, 24, "placement_version")?,
                layer_id: read_u32(bytes, 32, "layer_id")?,
                row_count,
                hidden_dim: read_u32(bytes, 40, "hidden_dim")?,
                hidden_dtype: ExpertV2Dtype::from_u16(read_u16(bytes, 44, "hidden_dtype")?)?,
                hidden_row_stride_bytes: read_u32(bytes, 88, "hidden_row_stride_bytes")?,
                route_count,
                row_descriptor_bytes,
                route_bytes,
                hidden_payload_bytes,
                flags,
            },
            frame_bytes: bytes,
            row_descriptor_bytes: &bytes[rows_start..routes_start],
            route_bytes: &bytes[routes_start..payload_start],
            hidden_payload: &bytes[payload_start..frame_end],
            checksum: if debug_checksum_enabled(flags) {
                Some(
                    &bytes[REQUEST_CHECKSUM_OFFSET
                        ..REQUEST_CHECKSUM_OFFSET
                            .checked_add(CHECKSUM_LEN)
                            .expect("checksum range overflow")],
                )
            } else {
                None
            },
        };
        view.validate()?;
        Ok(view)
    }

    pub fn header_len(&self) -> usize {
        request_header_len_from_flags(self.header.flags)
    }

    pub fn row_descriptor_bytes(&self) -> &'a [u8] {
        self.row_descriptor_bytes
    }

    pub fn route_bytes(&self) -> &'a [u8] {
        self.route_bytes
    }

    pub fn hidden_payload(&self) -> &'a [u8] {
        self.hidden_payload
    }

    pub(crate) fn encode_regular_forwarded_into(
        &self,
        response_dtype: ExpertV2Dtype,
        out: &mut Vec<u8>,
    ) -> Result<()> {
        if self.stream_plan_enabled()
            || self.stream_data_enabled()
            || self.precompile_warmup_enabled()
            || self.layer_block_enabled()
        {
            bail!("only regular ExpertProtocolV2 requests can be forwarded");
        }
        let response_flags = match response_dtype {
            ExpertV2Dtype::Bf16 => 0,
            ExpertV2Dtype::Fp8E4m3RowScaled => EXPERT_PROTOCOL_V2_FLAG_RESPONSE_FP8_E4M3_ROW_SCALED,
            ExpertV2Dtype::Nvfp4E2m1Fp8E4m3 => EXPERT_PROTOCOL_V2_FLAG_RESPONSE_NVFP4_E2M1_FP8_E4M3,
            other => bail!("unsupported forwarded response dtype {other:?}"),
        };
        let flags = response_flags
            | if self.debug_checksum_enabled() {
                EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM
            } else {
                0
            };
        validate_flags(flags, "request")?;

        out.clear();
        if out.capacity() < self.frame_bytes.len() {
            out.reserve(self.frame_bytes.len() - out.capacity());
        }
        out.extend_from_slice(self.frame_bytes);
        out[84..88].copy_from_slice(&flags.to_le_bytes());
        Ok(())
    }

    pub fn hidden_row_payload(&self, index: usize) -> Result<&'a [u8]> {
        if self.stream_plan_enabled() {
            bail!("ExpertProtocolV2 stream plan payload does not contain hidden rows");
        }
        if index >= self.header.row_count as usize {
            bail!(
                "ExpertProtocolV2 request row index {index} exceeds row count {}",
                self.header.row_count
            );
        }
        let stride = self.header.hidden_row_stride_bytes as usize;
        let start = index
            .checked_mul(stride)
            .context("ExpertProtocolV2 request hidden row offset overflow")?;
        let end = start
            .checked_add(stride)
            .context("ExpertProtocolV2 request hidden row range overflow")?;
        self.hidden_payload
            .get(start..end)
            .with_context(|| format!("ExpertProtocolV2 request hidden row {index} is out of range"))
    }

    pub fn row(&self, index: usize) -> Result<ExpertProtocolV2RowDescriptor> {
        if self.stream_data_enabled() {
            bail!("ExpertProtocolV2 stream data frame has no row descriptors");
        }
        if index >= self.header.row_count as usize {
            bail!(
                "ExpertProtocolV2 request row index {index} exceeds row count {}",
                self.header.row_count
            );
        }
        decode_row_descriptor(
            self.row_descriptor_bytes,
            index * EXPERT_PROTOCOL_V2_ROW_DESCRIPTOR_LEN,
        )
    }

    pub fn route(&self, index: usize) -> Result<ExpertProtocolV2RouteEntry> {
        if self.stream_data_enabled() {
            bail!("ExpertProtocolV2 stream data frame has no route entries");
        }
        if index >= self.header.route_count as usize {
            bail!(
                "ExpertProtocolV2 request route index {index} exceeds route count {}",
                self.header.route_count
            );
        }
        decode_route_entry(self.route_bytes, index * EXPERT_PROTOCOL_V2_ROUTE_ENTRY_LEN)
    }

    pub fn wire_stats(&self) -> ExpertProtocolV2WireStats {
        ExpertProtocolV2WireStats {
            logical_payload_bytes: self.hidden_payload.len(),
            wire_bytes: self.header_len()
                + self.row_descriptor_bytes.len()
                + self.route_bytes.len()
                + self.hidden_payload.len(),
        }
    }

    pub fn debug_checksum_enabled(&self) -> bool {
        debug_checksum_enabled(self.header.flags)
    }

    pub fn precompile_warmup_enabled(&self) -> bool {
        precompile_warmup_enabled(self.header.flags)
    }

    pub fn fp8_e4m3_row_scaled_response_enabled(&self) -> bool {
        response_fp8_e4m3_row_scaled_enabled(self.header.flags)
    }

    pub fn nvfp4_e2m1_fp8_e4m3_response_enabled(&self) -> bool {
        response_nvfp4_e2m1_fp8_e4m3_enabled(self.header.flags)
    }

    pub fn stream_plan_enabled(&self) -> bool {
        stream_plan_enabled(self.header.flags)
    }

    pub fn stream_data_enabled(&self) -> bool {
        stream_data_enabled(self.header.flags)
    }

    pub fn stream_final_enabled(&self) -> bool {
        stream_final_enabled(self.header.flags)
    }

    pub fn spark_reduction_enabled(&self) -> bool {
        spark_reduction_enabled(self.header.flags)
    }

    pub fn spark_row_sharded_reduction_enabled(&self) -> bool {
        spark_row_sharded_reduction_enabled(self.header.flags)
    }

    pub fn spark_collective_part_count(&self) -> usize {
        spark_collective_part_count(self.header.flags)
    }

    pub fn layer_block_enabled(&self) -> bool {
        layer_block_enabled(self.header.flags)
    }

    pub fn stream_data_row_offset(&self) -> Option<usize> {
        self.stream_data_enabled()
            .then_some(self.header.route_count as usize)
    }

    pub fn verify_checksum(&self) -> Result<()> {
        if !self.debug_checksum_enabled() {
            bail!("ExpertProtocolV2 request debug checksum flag is not set");
        }
        let descriptor_rows = if self.stream_data_enabled() {
            0
        } else {
            self.header.row_count as usize
        };
        let route_entries = if self.stream_data_enabled() {
            0
        } else {
            self.header.route_count as usize
        };
        let rows = (0..descriptor_rows)
            .map(|index| self.row(index))
            .collect::<Result<Vec<_>>>()?;
        let routes = (0..route_entries)
            .map(|index| self.route(index))
            .collect::<Result<Vec<_>>>()?;
        let actual = payload_checksum(&rows, &routes, self.hidden_payload);
        let Some(checksum) = self.checksum else {
            bail!("ExpertProtocolV2 request checksum header is malformed");
        };
        if actual.as_slice() != checksum {
            bail!("ExpertProtocolV2 request checksum mismatch");
        }
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        let stream_plan = self.stream_plan_enabled();
        let stream_data = self.stream_data_enabled();
        let layer_block = self.layer_block_enabled();
        if stream_data {
            if !self.row_descriptor_bytes.is_empty() || !self.route_bytes.is_empty() {
                bail!("ExpertProtocolV2 stream data frame cannot contain row or route descriptors");
            }
            if self.header.row_count == 0 {
                bail!("ExpertProtocolV2 stream data frame must contain at least one row");
            }
            self.header
                .route_count
                .checked_add(self.header.row_count)
                .context("ExpertProtocolV2 stream data row range overflow")?;
        } else if stream_plan && self.header.row_count == 0 {
            bail!("ExpertProtocolV2 stream plan must contain at least one row");
        }
        if self.header.hidden_dim == 0 {
            bail!("ExpertProtocolV2 request hidden_dim must be non-zero");
        }
        let logical_row_bytes = self
            .header
            .hidden_dtype
            .row_bytes(self.header.hidden_dim as usize)
            .context("ExpertProtocolV2 request hidden row byte count")?;
        let hidden_row_stride_bytes = self.header.hidden_row_stride_bytes as usize;
        if hidden_row_stride_bytes < logical_row_bytes {
            bail!(
                "ExpertProtocolV2 request hidden row stride bytes {} smaller than logical row bytes {}",
                self.header.hidden_row_stride_bytes,
                logical_row_bytes
            );
        }
        if stream_plan {
            if self.hidden_payload.is_empty() {
                bail!("ExpertProtocolV2 stream plan payload must be non-empty");
            }
        } else {
            let expected_payload = (self.header.row_count as usize)
                .checked_mul(hidden_row_stride_bytes)
                .context("ExpertProtocolV2 request hidden payload byte count overflow")?;
            if expected_payload != self.hidden_payload.len() {
                bail!(
                    "ExpertProtocolV2 request hidden payload bytes mismatch: expected={} actual={}",
                    expected_payload,
                    self.hidden_payload.len()
                );
            }
        }
        let descriptor_rows = if stream_data {
            0
        } else {
            self.header.row_count as usize
        };
        let route_entries = if stream_data {
            0
        } else {
            self.header.route_count as usize
        };
        for row_index in 0..descriptor_rows {
            let row = self.row(row_index)?;
            if layer_block && (row.route_offset != 0 || row.route_count != 0) {
                bail!(
                    "ExpertProtocolV2 layer-block row {row_index} must not contain precomputed routes"
                );
            }
            let end =
                row.route_offset
                    .checked_add(row.route_count)
                    .context("ExpertProtocolV2 row route range overflow")? as usize;
            if end > self.header.route_count as usize {
                bail!(
                    "ExpertProtocolV2 row {row_index} route range {}..{} exceeds route count {}",
                    row.route_offset,
                    end,
                    self.header.route_count
                );
            }
        }
        for route_index in 0..route_entries {
            let route = self.route(route_index)?;
            if route.row_index >= self.header.row_count {
                bail!(
                    "ExpertProtocolV2 route row_index {} exceeds row count {}",
                    route.row_index,
                    self.header.row_count
                );
            }
            if !route.gate_weight.is_finite() {
                bail!("ExpertProtocolV2 route gate_weight must be finite");
            }
        }
        if layer_block && route_entries != 0 {
            bail!("ExpertProtocolV2 layer-block request cannot contain precomputed routes");
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ExpertProtocolV2ResponseView<'a> {
    pub header: ExpertProtocolV2ResponseHeader,
    row_index_bytes: Option<&'a [u8]>,
    partial_output_payload: &'a [u8],
    checksum: Option<&'a [u8]>,
}

impl<'a> ExpertProtocolV2ResponseView<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() < EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN {
            bail!("ExpertProtocolV2 response frame too short: {}", bytes.len());
        }
        let header_len = validate_common_header(bytes, RESPONSE_KIND)?;
        let row_count = read_u32(bytes, 36, "row_count")?;
        let output_payload_bytes = read_u64(bytes, 44, "output_payload_bytes")?;
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
        let row_index_bytes_len = if response_row_indices_enabled(flags) {
            (row_count as usize)
                .checked_mul(std::mem::size_of::<u32>())
                .context("ExpertProtocolV2 response row index byte count overflow")?
        } else {
            0
        };
        let payload_start = header_len
            .checked_add(row_index_bytes_len)
            .context("ExpertProtocolV2 response payload offset overflow")?;
        let payload_end = payload_start + payload_len;
        if payload_end != bytes.len() {
            bail!(
                "ExpertProtocolV2 response length mismatch: sections end at {payload_end}, frame has {}",
                bytes.len()
            );
        }

        let view = Self {
            header: ExpertProtocolV2ResponseHeader {
                request_id: read_u64(bytes, 16, "request_id")?,
                placement_version: read_u64(bytes, 24, "placement_version")?,
                layer_id: read_u32(bytes, 32, "layer_id")?,
                row_count,
                output_dim: read_u32(bytes, 72, "output_dim")?,
                output_dtype: ExpertV2Dtype::from_u16(read_u16(bytes, 40, "output_dtype")?)?,
                output_row_stride_bytes: read_u32(bytes, 76, "output_row_stride_bytes")?,
                status: ExpertProtocolV2Status::from_u16(read_u16(bytes, 42, "status")?)?,
                output_payload_bytes,
                flags,
                executor_id: read_u64(bytes, RESPONSE_EXECUTOR_ID_OFFSET, "executor_id")?,
            },
            row_index_bytes: response_row_indices_enabled(flags)
                .then_some(&bytes[header_len..payload_start]),
            partial_output_payload: &bytes[payload_start..payload_end],
            checksum: if debug_checksum_enabled(flags) {
                Some(
                    &bytes[RESPONSE_CHECKSUM_OFFSET
                        ..RESPONSE_CHECKSUM_OFFSET
                            .checked_add(CHECKSUM_LEN)
                            .expect("checksum range overflow")],
                )
            } else {
                None
            },
        };
        view.validate()?;
        Ok(view)
    }

    pub fn header_len(&self) -> usize {
        response_header_len_from_flags(self.header.flags)
    }

    pub fn partial_output_payload(&self) -> &'a [u8] {
        self.partial_output_payload
    }

    pub fn row_indexed(&self) -> bool {
        response_row_indices_enabled(self.header.flags)
    }

    pub fn more_chunks(&self) -> bool {
        response_more_chunks_enabled(self.header.flags)
    }

    pub fn request_row_index(&self, index: usize) -> Result<u32> {
        if index >= self.header.row_count as usize {
            bail!(
                "ExpertProtocolV2 response row index {index} exceeds row count {}",
                self.header.row_count
            );
        }
        let Some(row_index_bytes) = self.row_index_bytes else {
            return Ok(index as u32);
        };
        let start = index * std::mem::size_of::<u32>();
        Ok(u32::from_le_bytes(
            row_index_bytes[start..start + std::mem::size_of::<u32>()]
                .try_into()
                .expect("u32 response row index"),
        ))
    }

    pub fn partial_output_row_payload(&self, index: usize) -> Result<&'a [u8]> {
        if index >= self.header.row_count as usize {
            bail!(
                "ExpertProtocolV2 response row index {index} exceeds row count {}",
                self.header.row_count
            );
        }
        let stride = self.header.output_row_stride_bytes as usize;
        let start = index
            .checked_mul(stride)
            .context("ExpertProtocolV2 response output row offset overflow")?;
        let end = start
            .checked_add(stride)
            .context("ExpertProtocolV2 response output row range overflow")?;
        self.partial_output_payload
            .get(start..end)
            .with_context(|| {
                format!("ExpertProtocolV2 response output row {index} is out of range")
            })
    }

    pub fn wire_stats(&self) -> ExpertProtocolV2WireStats {
        ExpertProtocolV2WireStats {
            logical_payload_bytes: self.partial_output_payload.len(),
            wire_bytes: self.header_len()
                + self.row_index_bytes.map(<[u8]>::len).unwrap_or(0)
                + self.partial_output_payload.len(),
        }
    }

    pub fn debug_checksum_enabled(&self) -> bool {
        debug_checksum_enabled(self.header.flags)
    }

    pub fn verify_checksum(&self) -> Result<()> {
        if !self.debug_checksum_enabled() {
            bail!("ExpertProtocolV2 response debug checksum flag is not set");
        }
        let mut hasher = Sha256::new();
        if let Some(row_index_bytes) = self.row_index_bytes {
            hasher.update(row_index_bytes);
        }
        hasher.update(self.partial_output_payload);
        let actual = hasher.finalize();
        let Some(checksum) = self.checksum else {
            bail!("ExpertProtocolV2 response checksum header is malformed");
        };
        if actual.as_slice() != checksum {
            bail!("ExpertProtocolV2 response checksum mismatch");
        }
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        if self.row_indexed() != self.row_index_bytes.is_some() {
            bail!("ExpertProtocolV2 response row-index flag and row index table disagree");
        }
        if self.more_chunks() && !self.row_indexed() {
            bail!("ExpertProtocolV2 response more-chunks flag requires row indices");
        }
        if self.row_indexed() {
            let mut indices = (0..self.header.row_count as usize)
                .map(|index| self.request_row_index(index))
                .collect::<Result<Vec<_>>>()?;
            let original_len = indices.len();
            indices.sort_unstable();
            indices.dedup();
            if indices.len() != original_len {
                bail!("ExpertProtocolV2 response row indices must be unique within a chunk");
            }
        }
        if self.header.output_dim == 0 {
            bail!("ExpertProtocolV2 response output_dim must be non-zero");
        }
        let logical_row_bytes = self
            .header
            .output_dtype
            .row_bytes(self.header.output_dim as usize)
            .context("ExpertProtocolV2 response output row byte count")?;
        let output_row_stride_bytes = self.header.output_row_stride_bytes as usize;
        if output_row_stride_bytes < logical_row_bytes {
            bail!(
                "ExpertProtocolV2 response output row stride bytes {} smaller than logical row bytes {}",
                self.header.output_row_stride_bytes,
                logical_row_bytes
            );
        }
        let expected_payload = (self.header.row_count as usize)
            .checked_mul(output_row_stride_bytes)
            .context("ExpertProtocolV2 response output payload byte count overflow")?;
        if expected_payload != self.partial_output_payload.len() {
            bail!(
                "ExpertProtocolV2 response output payload bytes mismatch: expected={} actual={}",
                expected_payload,
                self.partial_output_payload.len()
            );
        }
        Ok(())
    }
}
