use glmrt_core::{ExpertRequest, ExpertRow, ExpertWaveMetadata, LayerWaveMode};
use std::sync::atomic::Ordering;

use crate::{
    parse_expert_target, runtime_error, sum_partials, ApiError, ApiState, ApiTransport,
    RealSlicePrefillProbe, RealSlicePrefillRouteRow,
};

use super::routes::{partition_real_prefill_routes, target_for_real_router_owner};

pub(in crate::backends::real_slice) struct RealPrefillDispatchSummary {
    pub(in crate::backends::real_slice) transport: &'static str,
    pub(in crate::backends::real_slice) owners: String,
    pub(in crate::backends::real_slice) request_count: usize,
    pub(in crate::backends::real_slice) row_count: usize,
    pub(in crate::backends::real_slice) route_count: usize,
    pub(in crate::backends::real_slice) partial_count: usize,
    pub(in crate::backends::real_slice) checksum: f64,
    pub(in crate::backends::real_slice) first: f32,
    pub(in crate::backends::real_slice) last: f32,
}

pub(in crate::backends::real_slice) async fn dispatch_real_prefill_probe(
    state: &ApiState,
    prefill: &RealSlicePrefillProbe,
) -> Result<Option<RealPrefillDispatchSummary>, ApiError> {
    dispatch_real_prefill_rows(
        state,
        "real prefill",
        "phase0-real-prefill-dispatch",
        prefill.router_layer_id,
        prefill.chunk_token_count,
        &prefill.rmsnorm_rows,
        &prefill.route_rows,
    )
    .await
}

pub(in crate::backends::real_slice) async fn dispatch_real_prefill_mlp_input_probe(
    state: &ApiState,
    prefill: &RealSlicePrefillProbe,
) -> Result<Option<RealPrefillDispatchSummary>, ApiError> {
    dispatch_real_prefill_rows(
        state,
        "real prefill MLP input",
        "phase0-real-prefill-mlp-input-dispatch",
        prefill.router_layer_id,
        prefill.chunk_token_count,
        &prefill.mlp_input_rows,
        &prefill.mlp_input_route_rows,
    )
    .await
}

pub(super) async fn dispatch_real_prefill_rows(
    state: &ApiState,
    label: &str,
    placement_version: &str,
    layer_id: u32,
    chunk_token_count: usize,
    hidden_rows: &[Vec<f32>],
    route_rows: &[RealSlicePrefillRouteRow],
) -> Result<Option<RealPrefillDispatchSummary>, ApiError> {
    if hidden_rows.is_empty() || route_rows.is_empty() {
        return Ok(None);
    }
    if hidden_rows.len() != route_rows.len() {
        return Err(runtime_error(format!(
            "{label} probe row mismatch: hidden_rows={} route_rows={}",
            hidden_rows.len(),
            route_rows.len()
        )));
    }
    let hidden_dim = hidden_rows[0].len();
    if hidden_dim == 0 {
        return Ok(None);
    }
    let hidden_dim_u32 = u32::try_from(hidden_dim).map_err(|_| {
        runtime_error(format!(
            "{label} dispatch hidden size {hidden_dim} exceeds protocol u32 limit"
        ))
    })?;
    for row in hidden_rows {
        if row.len() != hidden_dim {
            return Err(runtime_error(format!(
                "{label} hidden rows have inconsistent widths"
            )));
        }
    }

    let route_groups = partition_real_prefill_routes(route_rows);
    if route_groups.is_empty() {
        return Ok(None);
    }
    let request_id = state
        .next_request_id
        .fetch_add(route_groups.len() as u64, Ordering::Relaxed);
    let owners = route_groups
        .iter()
        .map(|group| {
            let route_count = group.rows.iter().map(|row| row.routes.len()).sum::<usize>();
            format!("{}:{}", group.owner, route_count)
        })
        .collect::<Vec<_>>()
        .join(",");
    let route_count = route_groups
        .iter()
        .flat_map(|group| group.rows.iter())
        .map(|row| row.routes.len())
        .sum::<usize>();
    let mut per_row_partials = vec![Vec::<Vec<f32>>::new(); hidden_rows.len()];
    let mut partial_count = 0_usize;

    for (target_index, group) in route_groups.iter().enumerate() {
        let rows = group
            .rows
            .iter()
            .map(|row| ExpertRow {
                row_id: row.row_id,
                hidden: hidden_rows[row.row_index].clone(),
                routes: row.routes.clone(),
            })
            .collect::<Vec<_>>();
        let request = ExpertRequest {
            protocol_version: 1,
            request_id: request_id + target_index as u64,
            placement_version: placement_version.to_owned(),
            layer_id,
            hidden_dim: hidden_dim_u32,
            wave: Some(ExpertWaveMetadata {
                mode: LayerWaveMode::Prefill,
                graph_bucket_rows: chunk_token_count as u32,
                logical_bf16_payload_bytes: group.rows.len() * hidden_dim * 2,
            }),
            rows,
        };
        let response = match state.config.transport {
            ApiTransport::Inproc => glmrt_transport::inproc_roundtrip(&request)
                .await
                .map_err(runtime_error)?,
            ApiTransport::Tcp => {
                let target =
                    target_for_real_router_owner(&state.config.expert_targets, &group.owner)?;
                let addr = parse_expert_target(&target)?;
                glmrt_transport::tcp_protocol_v2_expert_request_roundtrip(
                    addr,
                    &request,
                    Default::default(),
                )
                .await
                .map_err(runtime_error)?
            }
            ApiTransport::TcpDebugJson => {
                let target =
                    target_for_real_router_owner(&state.config.expert_targets, &group.owner)?;
                let addr = parse_expert_target(&target)?;
                glmrt_transport::debug_json_tcp_roundtrip(addr, &request, Default::default())
                    .await
                    .map_err(runtime_error)?
            }
            ApiTransport::VerbsHost => {
                let target =
                    target_for_real_router_owner(&state.config.expert_targets, &group.owner)?;
                let addr = parse_expert_target(&target)?;
                glmrt_transport::verbs_host_protocol_v2_expert_request_roundtrip(
                    addr,
                    &request,
                    Default::default(),
                )
                .await
                .map_err(runtime_error)?
            }
        };
        if response.partial_outputs.len() != group.rows.len() {
            return Err(runtime_error(format!(
                "{label} expert response had {} rows, expected {}",
                response.partial_outputs.len(),
                group.rows.len()
            )));
        }
        for (grouped_row, partial) in group.rows.iter().zip(response.partial_outputs) {
            per_row_partials[grouped_row.row_index].push(partial);
            partial_count += 1;
        }
    }

    let mut checksum = 0.0_f64;
    let mut first = None;
    let mut last = 0.0_f32;
    for partials in &per_row_partials {
        if partials.is_empty() {
            return Err(runtime_error(format!(
                "{label} dispatch produced no partials for a row"
            )));
        }
        let summed = sum_partials(partials, hidden_dim)?;
        checksum += summed.iter().map(|value| *value as f64).sum::<f64>();
        first.get_or_insert(summed.first().copied().unwrap_or_default());
        last = summed.last().copied().unwrap_or_default();
    }

    Ok(Some(RealPrefillDispatchSummary {
        transport: state.config.transport.label(),
        owners,
        request_count: route_groups.len(),
        row_count: hidden_rows.len(),
        route_count,
        partial_count,
        checksum,
        first: first.unwrap_or_default(),
        last,
    }))
}
