use glmrt_core::{ExpertRequest, ExpertRow};
use std::sync::atomic::Ordering;

use crate::{
    parse_expert_target, runtime_error, sum_partials, ApiError, ApiState, ApiTransport,
    RealSliceLogitsProbe, RealSliceRouterProbe,
};

mod prefill;
mod prefill_attention;
mod routes;

pub(super) use prefill::{dispatch_real_prefill_mlp_input_probe, dispatch_real_prefill_probe};
pub(super) use prefill_attention::{
    dispatch_real_prefill_attention_mlp_input_probe,
    dispatch_real_prefill_deeper_layer_attention_mlp_input_probe,
    dispatch_real_prefill_extended_layer_attention_mlp_input_probe,
    dispatch_real_prefill_following_layer_attention_mlp_input_probe,
    dispatch_real_prefill_further_layer_attention_mlp_input_probe,
    dispatch_real_prefill_layer10_attention_mlp_input_probe,
    dispatch_real_prefill_layer11_attention_mlp_input_probe,
    dispatch_real_prefill_layer12_attention_mlp_input_probe,
    dispatch_real_prefill_layer13_attention_mlp_input_probe,
    dispatch_real_prefill_next_layer_attention_mlp_input_probe,
    dispatch_real_prefill_subsequent_layer_attention_mlp_input_probe,
};
use routes::{partition_real_router_routes, target_for_real_router_owner};

pub(super) struct RealRouterDispatchSummary {
    pub(super) transport: &'static str,
    pub(super) owners: String,
    pub(super) request_count: usize,
    pub(super) route_count: usize,
    pub(super) partial_count: usize,
    pub(super) checksum: f64,
    pub(super) first: f32,
    pub(super) last: f32,
}

pub(super) async fn dispatch_real_router_probe(
    state: &ApiState,
    probe: &RealSliceLogitsProbe,
    router: &RealSliceRouterProbe,
) -> Result<Option<RealRouterDispatchSummary>, ApiError> {
    if probe.rmsnorm_hidden.is_empty() || router.routes.is_empty() {
        return Ok(None);
    }
    let hidden_dim = probe.rmsnorm_hidden.len();
    let hidden_dim_u32 = u32::try_from(hidden_dim).map_err(|_| {
        runtime_error(format!(
            "real router dispatch hidden size {hidden_dim} exceeds protocol u32 limit"
        ))
    })?;
    let route_groups = partition_real_router_routes(&router.routes);
    let request_id = state
        .next_request_id
        .fetch_add(route_groups.len() as u64, Ordering::Relaxed);
    let mut partials = Vec::with_capacity(route_groups.len());
    let owners = route_groups
        .iter()
        .map(|group| format!("{}:{}", group.owner, group.routes.len()))
        .collect::<Vec<_>>()
        .join(",");
    let route_count = route_groups
        .iter()
        .map(|group| group.routes.len())
        .sum::<usize>();

    for (target_index, group) in route_groups.iter().enumerate() {
        let request = ExpertRequest {
            protocol_version: 1,
            request_id: request_id + target_index as u64,
            placement_version: "phase0-real-router-dispatch".to_owned(),
            layer_id: router.layer_id,
            hidden_dim: hidden_dim_u32,
            wave: None,
            rows: vec![ExpertRow {
                row_id: 0,
                hidden: probe.rmsnorm_hidden.clone(),
                routes: group.routes.clone(),
            }],
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
        if response.partial_outputs.len() != 1 {
            return Err(runtime_error(format!(
                "real router expert response had {} rows, expected 1",
                response.partial_outputs.len()
            )));
        }
        partials.push(response.partial_outputs[0].clone());
    }

    let summed = sum_partials(&partials, hidden_dim)?;
    let checksum = summed.iter().map(|value| *value as f64).sum::<f64>();
    let first = summed.first().copied().unwrap_or_default();
    let last = summed.last().copied().unwrap_or_default();
    Ok(Some(RealRouterDispatchSummary {
        transport: state.config.transport.label(),
        owners,
        request_count: partials.len(),
        route_count,
        partial_count: partials.len(),
        checksum,
        first,
        last,
    }))
}
