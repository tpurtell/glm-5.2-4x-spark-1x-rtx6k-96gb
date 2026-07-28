use anyhow::{bail, Context, Result};

use super::{
    canonical_route_entry, checked_u32, checked_usize, debug_checksum_enabled, decode_route_entry,
    decode_row_descriptor, encode_route_entry, encode_row_descriptor, layer_block_enabled,
    payload_checksum, precompile_warmup_enabled, push_u16, push_u32, push_u64, read_u16, read_u32,
    read_u64, request_header_len_from_flags, response_fp8_e4m3_row_scaled_enabled,
    response_nvfp4_e2m1_fp8_e4m3_enabled, spark_collective_part_count, spark_reduction_enabled,
    spark_row_sharded_reduction_enabled, stream_data_enabled, stream_final_enabled,
    stream_plan_enabled, validate_common_header, validate_flags, validate_header_len,
    ExpertProtocolV2Request, ExpertProtocolV2RequestHeader, ExpertProtocolV2RouteEntry,
    ExpertProtocolV2RowDescriptor, ExpertProtocolV2WireStats, ExpertV2Dtype, CHECKSUM_LEN,
    EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM, EXPERT_PROTOCOL_V2_FLAG_LAYER_BLOCK,
    EXPERT_PROTOCOL_V2_FLAG_PRECOMPILE_WARMUP,
    EXPERT_PROTOCOL_V2_FLAG_RESPONSE_FP8_E4M3_ROW_SCALED,
    EXPERT_PROTOCOL_V2_FLAG_RESPONSE_NVFP4_E2M1_FP8_E4M3, EXPERT_PROTOCOL_V2_FLAG_SPARK_REDUCTION,
    EXPERT_PROTOCOL_V2_FLAG_SPARK_ROW_SHARDED_REDUCTION, EXPERT_PROTOCOL_V2_FLAG_STREAM_DATA,
    EXPERT_PROTOCOL_V2_FLAG_STREAM_FINAL, EXPERT_PROTOCOL_V2_FLAG_STREAM_PLAN,
    EXPERT_PROTOCOL_V2_MAX_SPARK_COLLECTIVE_PARTS, EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN,
    EXPERT_PROTOCOL_V2_ROUTE_ENTRY_LEN, EXPERT_PROTOCOL_V2_ROW_DESCRIPTOR_LEN,
    EXPERT_PROTOCOL_V2_SPARK_COLLECTIVE_PART_COUNT_MASK,
    EXPERT_PROTOCOL_V2_SPARK_COLLECTIVE_PART_COUNT_SHIFT, MAGIC, REQUEST_CHECKSUM_OFFSET,
    REQUEST_KIND, VERSION,
};

impl ExpertProtocolV2Request {
    pub fn new(
        request_id: u64,
        placement_version: u64,
        layer_id: u32,
        hidden_dim: u32,
        hidden_dtype: ExpertV2Dtype,
        rows: Vec<ExpertProtocolV2RowDescriptor>,
        routes: Vec<ExpertProtocolV2RouteEntry>,
        hidden_payload: Vec<u8>,
    ) -> Result<Self> {
        let hidden_row_stride_bytes = default_hidden_row_stride_bytes(hidden_dim, hidden_dtype)?;
        Self::new_with_hidden_stride(
            request_id,
            placement_version,
            layer_id,
            hidden_dim,
            hidden_dtype,
            hidden_row_stride_bytes,
            rows,
            routes,
            hidden_payload,
        )
    }

    pub fn new_with_hidden_stride(
        request_id: u64,
        placement_version: u64,
        layer_id: u32,
        hidden_dim: u32,
        hidden_dtype: ExpertV2Dtype,
        hidden_row_stride_bytes: u32,
        rows: Vec<ExpertProtocolV2RowDescriptor>,
        routes: Vec<ExpertProtocolV2RouteEntry>,
        hidden_payload: Vec<u8>,
    ) -> Result<Self> {
        let routes = routes
            .into_iter()
            .map(canonical_route_entry)
            .collect::<Vec<_>>();
        let header = ExpertProtocolV2RequestHeader {
            request_id,
            placement_version,
            layer_id,
            row_count: rows.len() as u32,
            hidden_dim,
            hidden_dtype,
            hidden_row_stride_bytes,
            route_count: routes.len() as u32,
            row_descriptor_bytes: checked_u32(
                rows.len() * EXPERT_PROTOCOL_V2_ROW_DESCRIPTOR_LEN,
                "row descriptor bytes",
            )?,
            route_bytes: checked_u32(
                routes.len() * EXPERT_PROTOCOL_V2_ROUTE_ENTRY_LEN,
                "route bytes",
            )?,
            hidden_payload_bytes: hidden_payload.len() as u64,
            flags: 0,
        };
        let request = Self {
            header,
            rows,
            routes,
            hidden_payload,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn new_stream_plan(
        request_id: u64,
        placement_version: u64,
        layer_id: u32,
        hidden_dim: u32,
        hidden_dtype: ExpertV2Dtype,
        rows: Vec<ExpertProtocolV2RowDescriptor>,
        routes: Vec<ExpertProtocolV2RouteEntry>,
        plan_payload: Vec<u8>,
    ) -> Result<Self> {
        let hidden_row_stride_bytes = default_hidden_row_stride_bytes(hidden_dim, hidden_dtype)?;
        Self::new_stream_plan_with_hidden_stride(
            request_id,
            placement_version,
            layer_id,
            hidden_dim,
            hidden_dtype,
            hidden_row_stride_bytes,
            rows,
            routes,
            plan_payload,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_stream_plan_with_hidden_stride(
        request_id: u64,
        placement_version: u64,
        layer_id: u32,
        hidden_dim: u32,
        hidden_dtype: ExpertV2Dtype,
        hidden_row_stride_bytes: u32,
        rows: Vec<ExpertProtocolV2RowDescriptor>,
        routes: Vec<ExpertProtocolV2RouteEntry>,
        plan_payload: Vec<u8>,
    ) -> Result<Self> {
        let routes = routes
            .into_iter()
            .map(canonical_route_entry)
            .collect::<Vec<_>>();
        let header = ExpertProtocolV2RequestHeader {
            request_id,
            placement_version,
            layer_id,
            row_count: checked_u32(rows.len(), "stream plan row count")?,
            hidden_dim,
            hidden_dtype,
            hidden_row_stride_bytes,
            route_count: checked_u32(routes.len(), "stream plan route count")?,
            row_descriptor_bytes: checked_u32(
                rows.len() * EXPERT_PROTOCOL_V2_ROW_DESCRIPTOR_LEN,
                "stream plan row descriptor bytes",
            )?,
            route_bytes: checked_u32(
                routes.len() * EXPERT_PROTOCOL_V2_ROUTE_ENTRY_LEN,
                "stream plan route bytes",
            )?,
            hidden_payload_bytes: plan_payload.len() as u64,
            flags: EXPERT_PROTOCOL_V2_FLAG_STREAM_PLAN,
        };
        let request = Self {
            header,
            rows,
            routes,
            hidden_payload: plan_payload,
        };
        request.validate()?;
        Ok(request)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_stream_data(
        request_id: u64,
        placement_version: u64,
        layer_id: u32,
        hidden_dim: u32,
        hidden_dtype: ExpertV2Dtype,
        hidden_row_stride_bytes: u32,
        stream_row_offset: u32,
        row_count: u32,
        hidden_payload: Vec<u8>,
        final_frame: bool,
    ) -> Result<Self> {
        let request = Self {
            header: ExpertProtocolV2RequestHeader {
                request_id,
                placement_version,
                layer_id,
                row_count,
                hidden_dim,
                hidden_dtype,
                hidden_row_stride_bytes,
                // Stream-data frames have no route entries. This field is the
                // offset into the plan's activation-row order.
                route_count: stream_row_offset,
                row_descriptor_bytes: 0,
                route_bytes: 0,
                hidden_payload_bytes: hidden_payload.len() as u64,
                flags: EXPERT_PROTOCOL_V2_FLAG_STREAM_DATA
                    | if final_frame {
                        EXPERT_PROTOCOL_V2_FLAG_STREAM_FINAL
                    } else {
                        0
                    },
            },
            rows: Vec::new(),
            routes: Vec::new(),
            hidden_payload,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn with_debug_checksum(mut self) -> Self {
        self.header.flags |= EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM;
        self
    }

    pub fn with_precompile_warmup(mut self) -> Self {
        self.header.flags |= EXPERT_PROTOCOL_V2_FLAG_PRECOMPILE_WARMUP;
        self
    }

    pub fn with_fp8_e4m3_row_scaled_response(mut self) -> Self {
        self.header.flags |= EXPERT_PROTOCOL_V2_FLAG_RESPONSE_FP8_E4M3_ROW_SCALED;
        self
    }

    pub fn with_nvfp4_e2m1_fp8_e4m3_response(mut self) -> Self {
        self.header.flags |= EXPERT_PROTOCOL_V2_FLAG_RESPONSE_NVFP4_E2M1_FP8_E4M3;
        self
    }

    pub fn with_spark_reduction(mut self) -> Self {
        self.header.flags |= EXPERT_PROTOCOL_V2_FLAG_SPARK_REDUCTION;
        self
    }

    pub fn with_spark_row_sharded_reduction(mut self) -> Self {
        self.header.flags |= EXPERT_PROTOCOL_V2_FLAG_SPARK_REDUCTION
            | EXPERT_PROTOCOL_V2_FLAG_SPARK_ROW_SHARDED_REDUCTION;
        self
    }

    pub fn set_spark_collective_part_count(&mut self, part_count: usize) -> Result<()> {
        anyhow::ensure!(
            (2..=EXPERT_PROTOCOL_V2_MAX_SPARK_COLLECTIVE_PARTS).contains(&part_count),
            "Spark collective part count {part_count} must be in 2..={EXPERT_PROTOCOL_V2_MAX_SPARK_COLLECTIVE_PARTS}"
        );
        anyhow::ensure!(
            self.spark_row_sharded_reduction_enabled(),
            "striped Spark collective requires row-sharded reduction"
        );
        let encoded =
            u32::try_from(part_count).context("Spark collective part count exceeds u32")?;
        self.header.flags = (self.header.flags
            & !EXPERT_PROTOCOL_V2_SPARK_COLLECTIVE_PART_COUNT_MASK)
            | (encoded << EXPERT_PROTOCOL_V2_SPARK_COLLECTIVE_PART_COUNT_SHIFT);
        self.validate()
    }

    pub fn with_layer_block(mut self) -> Self {
        self.header.flags |= EXPERT_PROTOCOL_V2_FLAG_LAYER_BLOCK;
        self
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

    pub fn header_len(&self) -> usize {
        request_header_len_from_flags(self.header.flags)
    }

    pub fn payload_offset(&self) -> usize {
        self.header_len()
            + self.rows.len() * EXPERT_PROTOCOL_V2_ROW_DESCRIPTOR_LEN
            + self.routes.len() * EXPERT_PROTOCOL_V2_ROUTE_ENTRY_LEN
    }

    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        self.encode_into(&mut out)?;
        Ok(out)
    }

    pub fn encode_into(&self, out: &mut Vec<u8>) -> Result<()> {
        self.encode_prefix_into(out)?;
        out.extend_from_slice(&self.hidden_payload);
        Ok(())
    }

    pub(crate) fn encode_prefix_into(&self, out: &mut Vec<u8>) -> Result<()> {
        self.validate()?;
        let header_len = self.header_len();
        let wire_bytes = self.wire_stats().wire_bytes;
        let prefix_bytes = wire_bytes
            .checked_sub(self.hidden_payload.len())
            .context("ExpertProtocolV2 request prefix byte count underflows")?;
        out.clear();
        if out.capacity() < prefix_bytes {
            out.reserve(prefix_bytes - out.capacity());
        }
        out.extend_from_slice(MAGIC);
        push_u16(out, VERSION);
        push_u16(out, REQUEST_KIND);
        push_u32(out, header_len as u32);
        push_u64(out, self.header.request_id);
        push_u64(out, self.header.placement_version);
        push_u32(out, self.header.layer_id);
        push_u32(out, self.header.row_count);
        push_u32(out, self.header.hidden_dim);
        push_u16(out, self.header.hidden_dtype as u16);
        push_u16(out, 0);
        push_u32(out, self.header.route_count);
        push_u32(out, self.header.row_descriptor_bytes);
        push_u32(out, self.header.route_bytes);
        push_u64(out, self.header.hidden_payload_bytes);
        push_u64(out, self.wire_stats().logical_payload_bytes as u64);
        push_u64(out, wire_bytes as u64);
        push_u32(out, self.header.flags);
        push_u32(out, self.header.hidden_row_stride_bytes);
        if self.debug_checksum_enabled() {
            out.extend_from_slice(&payload_checksum(
                &self.rows,
                &self.routes,
                &self.hidden_payload,
            ));
        }
        out.resize(header_len, 0);

        for row in &self.rows {
            encode_row_descriptor(out, row);
        }
        for route in &self.routes {
            encode_route_entry(out, route);
        }
        Ok(())
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN {
            bail!("ExpertProtocolV2 request frame too short: {}", bytes.len());
        }
        let header_len = validate_common_header(bytes, REQUEST_KIND)?;
        let request_id = read_u64(bytes, 16, "request_id")?;
        let placement_version = read_u64(bytes, 24, "placement_version")?;
        let layer_id = read_u32(bytes, 32, "layer_id")?;
        let row_count = read_u32(bytes, 36, "row_count")?;
        let hidden_dim = read_u32(bytes, 40, "hidden_dim")?;
        let hidden_dtype = ExpertV2Dtype::from_u16(read_u16(bytes, 44, "hidden_dtype")?)?;
        let route_count = read_u32(bytes, 48, "route_count")?;
        let row_descriptor_bytes = read_u32(bytes, 52, "row_descriptor_bytes")?;
        let route_bytes = read_u32(bytes, 56, "route_bytes")?;
        let hidden_payload_bytes = read_u64(bytes, 60, "hidden_payload_bytes")?;
        let wire_bytes = read_u64(bytes, 76, "wire_bytes")? as usize;
        let flags = read_u32(bytes, 84, "flags")?;
        let hidden_row_stride_bytes = read_u32(bytes, 88, "hidden_row_stride_bytes")?;
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

        let rows = (0..rows_len)
            .map(|idx| {
                let offset = rows_start + idx * EXPERT_PROTOCOL_V2_ROW_DESCRIPTOR_LEN;
                decode_row_descriptor(bytes, offset)
            })
            .collect::<Result<Vec<_>>>()?;
        let routes = (0..routes_len)
            .map(|idx| {
                let offset = routes_start + idx * EXPERT_PROTOCOL_V2_ROUTE_ENTRY_LEN;
                decode_route_entry(bytes, offset)
            })
            .collect::<Result<Vec<_>>>()?;
        let hidden_payload = bytes[payload_start..frame_end].to_vec();

        if debug_checksum_enabled(flags) {
            let expected_checksum = &bytes[REQUEST_CHECKSUM_OFFSET
                ..REQUEST_CHECKSUM_OFFSET
                    .checked_add(CHECKSUM_LEN)
                    .expect("checksum range overflow")];
            let actual_checksum = payload_checksum(&rows, &routes, &hidden_payload);
            if actual_checksum.as_slice() != expected_checksum {
                bail!("ExpertProtocolV2 request checksum mismatch");
            }
        }

        let request = Self {
            header: ExpertProtocolV2RequestHeader {
                request_id,
                placement_version,
                layer_id,
                row_count,
                hidden_dim,
                hidden_dtype,
                hidden_row_stride_bytes,
                route_count,
                row_descriptor_bytes,
                route_bytes,
                hidden_payload_bytes,
                flags,
            },
            rows,
            routes,
            hidden_payload,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn wire_stats(&self) -> ExpertProtocolV2WireStats {
        ExpertProtocolV2WireStats {
            logical_payload_bytes: self.hidden_payload.len(),
            wire_bytes: self.header_len()
                + self.rows.len() * EXPERT_PROTOCOL_V2_ROW_DESCRIPTOR_LEN
                + self.routes.len() * EXPERT_PROTOCOL_V2_ROUTE_ENTRY_LEN
                + self.hidden_payload.len(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        validate_flags(self.header.flags, "request")?;
        let stream_plan = self.stream_plan_enabled();
        let stream_data = self.stream_data_enabled();
        let layer_block = self.layer_block_enabled();
        if stream_data {
            if !self.rows.is_empty() || !self.routes.is_empty() {
                bail!("ExpertProtocolV2 stream data frame cannot contain row or route descriptors");
            }
            if self.header.row_count == 0 {
                bail!("ExpertProtocolV2 stream data frame must contain at least one row");
            }
            self.header
                .route_count
                .checked_add(self.header.row_count)
                .context("ExpertProtocolV2 stream data row range overflow")?;
        } else {
            if self.header.row_count as usize != self.rows.len() {
                bail!("ExpertProtocolV2 request row_count does not match rows");
            }
            if self.header.route_count as usize != self.routes.len() {
                bail!("ExpertProtocolV2 request route_count does not match routes");
            }
            if stream_plan && self.rows.is_empty() {
                bail!("ExpertProtocolV2 stream plan must contain at least one row");
            }
            if layer_block && !self.routes.is_empty() {
                bail!("ExpertProtocolV2 layer-block request cannot contain precomputed routes");
            }
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
            let payload_rows = if stream_data {
                self.header.row_count as usize
            } else {
                self.rows.len()
            };
            let expected_payload = payload_rows
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
        if self.header.row_descriptor_bytes as usize
            != self.rows.len() * EXPERT_PROTOCOL_V2_ROW_DESCRIPTOR_LEN
        {
            bail!("ExpertProtocolV2 request row_descriptor_bytes mismatch");
        }
        if self.header.route_bytes as usize
            != self.routes.len() * EXPERT_PROTOCOL_V2_ROUTE_ENTRY_LEN
        {
            bail!("ExpertProtocolV2 request route_bytes mismatch");
        }
        if self.header.hidden_payload_bytes as usize != self.hidden_payload.len() {
            bail!("ExpertProtocolV2 request hidden_payload_bytes mismatch");
        }
        for (row_index, row) in self.rows.iter().enumerate() {
            if layer_block && (row.route_offset != 0 || row.route_count != 0) {
                bail!(
                    "ExpertProtocolV2 layer-block row {row_index} must not contain precomputed routes"
                );
            }
            let end =
                row.route_offset
                    .checked_add(row.route_count)
                    .context("ExpertProtocolV2 row route range overflow")? as usize;
            if end > self.routes.len() {
                bail!(
                    "ExpertProtocolV2 row {row_index} route range {}..{} exceeds route count {}",
                    row.route_offset,
                    end,
                    self.routes.len()
                );
            }
        }
        for route in &self.routes {
            if route.row_index as usize >= self.rows.len() {
                bail!(
                    "ExpertProtocolV2 route row_index {} exceeds row count {}",
                    route.row_index,
                    self.rows.len()
                );
            }
            if !route.gate_weight.is_finite() {
                bail!("ExpertProtocolV2 route gate_weight must be finite");
            }
        }
        Ok(())
    }

    pub fn wire_bytes_from_header(header: &[u8]) -> Result<usize> {
        if header.len() < EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN {
            bail!(
                "ExpertProtocolV2 request header too short: {}",
                header.len()
            );
        }
        let header_len = validate_common_header(header, REQUEST_KIND)?;
        let flags = read_u32(header, 84, "flags")?;
        validate_flags(flags, "request")?;
        validate_header_len(header_len, request_header_len_from_flags(flags))?;
        checked_usize(read_u64(header, 76, "wire_bytes")?, "request wire bytes")
    }
}

fn default_hidden_row_stride_bytes(hidden_dim: u32, hidden_dtype: ExpertV2Dtype) -> Result<u32> {
    checked_u32(
        hidden_dtype
            .row_bytes(hidden_dim as usize)
            .context("hidden row stride byte count")?,
        "hidden row stride bytes",
    )
}
