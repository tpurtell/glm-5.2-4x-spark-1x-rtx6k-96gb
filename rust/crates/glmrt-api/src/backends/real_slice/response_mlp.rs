use std::time::Instant;

use crate::metrics::BackendMetrics;
use crate::{duration_ms, ApiError, ApiState, RealSliceLogitsProbe};

use super::dispatch::dispatch_real_router_probe;

pub(super) async fn append_mlp_probe_summaries(
    response: &mut String,
    metrics: &mut BackendMetrics,
    state: &ApiState,
    probe: &RealSliceLogitsProbe,
) -> Result<(), ApiError> {
    if let Some(sampling) = &probe.sampling_probe {
        response.push_str(&format!(
            " sampling_probe strategy={} token={} decoded={:?} logit={:.6}",
            sampling.strategy,
            sampling.selected_token_id,
            sampling.decoded_text,
            sampling.selected_logit
        ));
    }
    if let Some(mlp_input) = &probe.mlp_input_norm_probe {
        response.push_str(&format!(
            " mlp_input_norm_probe layer={} source={} hidden_width={} checksum={:.6} norm_l2={:.6} first={:.6} last={:.6}",
            mlp_input.layer_id,
            mlp_input.residual_source,
            mlp_input.hidden_width,
            mlp_input.normalized_checksum,
            mlp_input.normalized_l2_norm,
            mlp_input.first_normalized,
            mlp_input.last_normalized
        ));
    }
    if let Some(router) = &probe.mlp_input_router_probe {
        let expert_ids = router
            .routes
            .iter()
            .map(|route| format!("{}@{}", route.expert_id, route.owner))
            .collect::<Vec<_>>()
            .join(",");
        response.push_str(&format!(
            " mlp_input_router_probe layer={} top_k={} experts=[{}]",
            router.layer_id, router.top_k, expert_ids
        ));
    }
    if let Some(routed) = &probe.mlp_input_routed_expert_probe {
        response.push_str(&format!(
            " mlp_input_routed_expert_probe layer={} expert={} owner={} intermediate={} outputs={} quant={} checksum={:.6} first={:.6} last={:.6} reduction_routes={} reduction_checksum={:.6} reduction_first={:.6} reduction_last={:.6}",
            routed.layer_id,
            routed.expert_id,
            routed.owner,
            routed.intermediate_count,
            routed.output_count,
            routed.quant_recipe,
            routed.output_checksum,
            routed.first_output,
            routed.last_output,
            routed.reduction_route_count,
            routed.reduction_output_checksum,
            routed.reduction_first_output,
            routed.reduction_last_output
        ));
    }
    if let Some(shared) = &probe.mlp_input_shared_expert_probe {
        response.push_str(&format!(
            " mlp_input_shared_expert_probe layer={} intermediate={} outputs={} checksum={:.6} first={:.6} last={:.6}",
            shared.layer_id,
            shared.intermediate_count,
            shared.output_count,
            shared.output_checksum,
            shared.first_output,
            shared.last_output
        ));
    }
    if let Some(moe) = &probe.mlp_input_moe_branch_probe {
        response.push_str(&format!(
            " mlp_input_moe_branch_probe layer={} outputs={} routes={} routed_checksum={:.6} shared_checksum={:.6} checksum={:.6} first={:.6} last={:.6}",
            moe.layer_id,
            moe.output_count,
            moe.routed_route_count,
            moe.routed_output_checksum,
            moe.shared_output_checksum,
            moe.output_checksum,
            moe.first_output,
            moe.last_output
        ));
    }
    if let Some(residual) = &probe.mlp_input_residual_probe {
        response.push_str(&format!(
            " mlp_input_residual_probe layer={} outputs={} source={} residual_checksum={:.6} branch_checksum={:.6} checksum={:.6} first={:.6} last={:.6}",
            residual.layer_id,
            residual.output_count,
            residual.residual_source,
            residual.residual_checksum,
            residual.branch_checksum,
            residual.output_checksum,
            residual.first_output,
            residual.last_output
        ));
    }
    if let Some(prefill_moe) = &probe.prefill_mlp_input_moe_probe {
        response.push_str(&format!(
            " prefill_mlp_input_moe_probe layer={} rows={} outputs={} routes={} source={} routed_checksum={:.6} shared_checksum={:.6} branch_checksum={:.6} residual_checksum={:.6} checksum={:.6} first={:.6} last={:.6}",
            prefill_moe.layer_id,
            prefill_moe.row_count,
            prefill_moe.output_count,
            prefill_moe.route_count,
            prefill_moe.residual_source,
            prefill_moe.routed_output_checksum,
            prefill_moe.shared_output_checksum,
            prefill_moe.branch_checksum,
            prefill_moe.residual_checksum,
            prefill_moe.output_checksum,
            prefill_moe.first_output,
            prefill_moe.last_output
        ));
    }
    if let Some(router) = &probe.router_probe {
        let expert_ids = router
            .routes
            .iter()
            .map(|route| format!("{}@{}", route.expert_id, route.owner))
            .collect::<Vec<_>>()
            .join(",");
        response.push_str(&format!(
            " router_probe layer={} top_k={} experts=[{}]",
            router.layer_id, router.top_k, expert_ids
        ));
        let decode_start = Instant::now();
        if let Some(dispatch) = dispatch_real_router_probe(state, probe, router).await? {
            metrics.decode_ms += duration_ms(decode_start.elapsed());
            metrics.layerwave_decode_rows = 1;
            response.push_str(&format!(
                " dispatch_probe transport={} owners=[{}] requests={} routes={} partials={} checksum={:.6} first={:.6} last={:.6}",
                dispatch.transport,
                dispatch.owners,
                dispatch.request_count,
                dispatch.route_count,
                dispatch.partial_count,
                dispatch.checksum,
                dispatch.first,
                dispatch.last
            ));
        } else {
            metrics.decode_ms += duration_ms(decode_start.elapsed());
        }
    }
    if let Some(routed) = &probe.routed_expert_probe {
        response.push_str(&format!(
            " routed_expert_probe layer={} expert={} owner={} intermediate={} outputs={} quant={} checksum={:.6} first={:.6} last={:.6} reduction_routes={} reduction_checksum={:.6} reduction_first={:.6} reduction_last={:.6}",
            routed.layer_id,
            routed.expert_id,
            routed.owner,
            routed.intermediate_count,
            routed.output_count,
            routed.quant_recipe,
            routed.output_checksum,
            routed.first_output,
            routed.last_output,
            routed.reduction_route_count,
            routed.reduction_output_checksum,
            routed.reduction_first_output,
            routed.reduction_last_output
        ));
    }
    if let Some(shared) = &probe.shared_expert_probe {
        response.push_str(&format!(
            " shared_expert_probe layer={} intermediate={} outputs={} checksum={:.6} first={:.6} last={:.6}",
            shared.layer_id,
            shared.intermediate_count,
            shared.output_count,
            shared.output_checksum,
            shared.first_output,
            shared.last_output
        ));
    }
    if let Some(moe) = &probe.moe_branch_probe {
        response.push_str(&format!(
            " moe_branch_probe layer={} outputs={} routes={} routed_checksum={:.6} shared_checksum={:.6} checksum={:.6} first={:.6} last={:.6}",
            moe.layer_id,
            moe.output_count,
            moe.routed_route_count,
            moe.routed_output_checksum,
            moe.shared_output_checksum,
            moe.output_checksum,
            moe.first_output,
            moe.last_output
        ));
    }
    if let Some(residual) = &probe.mlp_residual_probe {
        response.push_str(&format!(
            " mlp_residual_probe layer={} outputs={} source={} residual_checksum={:.6} branch_checksum={:.6} checksum={:.6} first={:.6} last={:.6}",
            residual.layer_id,
            residual.output_count,
            residual.residual_source,
            residual.residual_checksum,
            residual.branch_checksum,
            residual.output_checksum,
            residual.first_output,
            residual.last_output
        ));
    }
    Ok(())
}

pub(super) fn append_dense_probe_summaries(response: &mut String, probe: &RealSliceLogitsProbe) {
    if let Some(dense) = &probe.dense_mlp_probe {
        response.push_str(&format!(
            " dense_mlp_probe layer={} intermediate={} outputs={} checksum={:.6} first={:.6} last={:.6}",
            dense.layer_id,
            dense.intermediate_count,
            dense.output_count,
            dense.output_checksum,
            dense.first_output,
            dense.last_output
        ));
    }
    if let Some(prefill_dense) = &probe.prefill_dense_mlp_probe {
        response.push_str(&format!(
            " prefill_dense_mlp_probe layer={} rows={} intermediate={} outputs={} source={} norm_checksum={:.6} activation_checksum={:.6} checksum={:.6} first={:.6} last={:.6}",
            prefill_dense.layer_id,
            prefill_dense.row_count,
            prefill_dense.intermediate_count,
            prefill_dense.output_count,
            prefill_dense.residual_source,
            prefill_dense.norm_checksum,
            prefill_dense.activation_checksum,
            prefill_dense.output_checksum,
            prefill_dense.first_output,
            prefill_dense.last_output
        ));
    }
    if let Some(residual) = &probe.prefill_dense_mlp_residual_probe {
        response.push_str(&format!(
            " prefill_dense_mlp_residual_probe layer={} rows={} outputs={} source={} residual_checksum={:.6} branch_checksum={:.6} checksum={:.6} first={:.6} last={:.6}",
            residual.layer_id,
            residual.row_count,
            residual.output_count,
            residual.residual_source,
            residual.residual_checksum,
            residual.branch_checksum,
            residual.output_checksum,
            residual.first_output,
            residual.last_output
        ));
    }
}
