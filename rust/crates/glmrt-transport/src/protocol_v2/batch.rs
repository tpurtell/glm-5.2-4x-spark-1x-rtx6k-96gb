use anyhow::{bail, Context, Result};
use glmrt_core::{DType, ExpertBatch, ExpertHostBatch, RowSourceKind};

use super::{
    checked_u32, expert_protocol_v2_compact_id, ExpertProtocolV2Request,
    ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor, ExpertV2Dtype, ExpertV2SourceKind,
};

impl ExpertProtocolV2Request {
    pub fn from_expert_batch(
        request_id: u64,
        batch: &ExpertBatch,
        routes: Vec<ExpertProtocolV2RouteEntry>,
        hidden_payload: Vec<u8>,
    ) -> Result<Self> {
        if routes.len() != batch.route_count() {
            bail!(
                "ExpertProtocolV2 routes length {} does not match ExpertBatch route count {}",
                routes.len(),
                batch.route_count()
            );
        }
        validate_batch_route_rows(batch, &routes)?;

        let rows = batch
            .rows
            .iter()
            .map(|row| {
                Ok(ExpertProtocolV2RowDescriptor {
                    row_id: row.row_id,
                    source_kind: source_kind(row.source_kind),
                    source_request_id: expert_protocol_v2_compact_id(&row.request_id.0),
                    token_position: row.token_position.0,
                    route_offset: checked_u32(row.route_offset, "ExpertBatch route offset")?,
                    route_count: checked_u32(row.route_count, "ExpertBatch route count")?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Self::new(
            request_id,
            expert_protocol_v2_compact_id(&batch.placement_version.0),
            batch.layer_id.0,
            checked_u32(batch.hidden_dim, "ExpertBatch hidden dim")?,
            dtype(batch.hidden_dtype.clone())?,
            rows,
            routes,
            hidden_payload,
        )
    }

    pub fn from_expert_host_batch(
        request_id: u64,
        batch: &ExpertHostBatch,
        hidden_payload: Vec<u8>,
    ) -> Result<Self> {
        validate_host_batch_route_rows(batch)?;
        let rows = batch
            .rows
            .iter()
            .map(|row| {
                Ok(ExpertProtocolV2RowDescriptor {
                    row_id: row.row_id,
                    source_kind: source_kind(row.source_kind),
                    source_request_id: expert_protocol_v2_compact_id(&row.request_id.0),
                    token_position: row.token_position.0,
                    route_offset: checked_u32(row.route_offset, "ExpertHostBatch route offset")?,
                    route_count: checked_u32(row.route_count, "ExpertHostBatch route count")?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let routes = batch
            .routes
            .iter()
            .map(|route| {
                Ok(ExpertProtocolV2RouteEntry {
                    row_index: checked_u32(route.row_index, "ExpertHostBatch route row index")?,
                    expert_id: checked_u32(route.expert_id, "ExpertHostBatch route expert id")?,
                    gate_weight: route.gate_weight,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Self::new(
            request_id,
            expert_protocol_v2_compact_id(&batch.placement_version.0),
            batch.layer_id.0,
            checked_u32(batch.hidden_dim, "ExpertHostBatch hidden dim")?,
            dtype(batch.hidden_dtype.clone())?,
            rows,
            routes,
            hidden_payload,
        )
    }
}

fn validate_batch_route_rows(
    batch: &ExpertBatch,
    routes: &[ExpertProtocolV2RouteEntry],
) -> Result<()> {
    for (row_index, row) in batch.rows.iter().enumerate() {
        let end = row
            .route_offset
            .checked_add(row.route_count)
            .context("ExpertBatch route range overflow")?;
        let row_routes = routes.get(row.route_offset..end).with_context(|| {
            format!(
                "ExpertBatch row {row_index} route range {}..{} exceeds route count {}",
                row.route_offset,
                end,
                routes.len()
            )
        })?;
        for route in row_routes {
            if route.row_index as usize != row_index {
                bail!(
                    "ExpertBatch route row_index {} does not match batch row index {row_index}",
                    route.row_index
                );
            }
        }
    }
    Ok(())
}

fn validate_host_batch_route_rows(batch: &ExpertHostBatch) -> Result<()> {
    for (row_index, row) in batch.rows.iter().enumerate() {
        let end = row
            .route_offset
            .checked_add(row.route_count)
            .context("ExpertHostBatch route range overflow")?;
        let row_routes = batch.routes.get(row.route_offset..end).with_context(|| {
            format!(
                "ExpertHostBatch row {row_index} route range {}..{} exceeds route count {}",
                row.route_offset,
                end,
                batch.routes.len()
            )
        })?;
        for route in row_routes {
            if route.row_index != row_index {
                bail!(
                    "ExpertHostBatch route row_index {} does not match host row index {row_index}",
                    route.row_index
                );
            }
        }
    }
    Ok(())
}

fn dtype(dtype: DType) -> Result<ExpertV2Dtype> {
    match dtype {
        DType::Bf16 => Ok(ExpertV2Dtype::Bf16),
        DType::F16 => Ok(ExpertV2Dtype::F16),
        DType::F4 => Ok(ExpertV2Dtype::Nvfp4E2m1Fp8E4m3),
        other => bail!("ExpertBatch hidden dtype {other:?} is not supported by ExpertProtocolV2"),
    }
}

fn source_kind(source_kind: RowSourceKind) -> ExpertV2SourceKind {
    match source_kind {
        RowSourceKind::DecodeStep => ExpertV2SourceKind::Decode,
        RowSourceKind::PrefillChunk => ExpertV2SourceKind::Prefill,
        RowSourceKind::MtpVerifyBlock => ExpertV2SourceKind::MtpVerify,
        RowSourceKind::Benchmark => ExpertV2SourceKind::Benchmark,
    }
}
