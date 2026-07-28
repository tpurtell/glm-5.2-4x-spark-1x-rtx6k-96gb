use std::time::Instant;

use crate::metrics::BackendMetrics;
use crate::{
    duration_ms, ApiError, ApiState, RealSliceLogitsProbe, RealSlicePrefillAttentionMlpInputProbe,
    RealSlicePrefillAttentionOutputProbe, RealSlicePrefillAttentionResidualProbe,
    RealSlicePrefillMlpInputMoeProbe, RealSlicePrefillNextLayerAttentionKvProbe,
};

use super::dispatch::{
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

macro_rules! append_attention_layer_probe_summary {
    (
        $response:expr,
        $metrics:expr,
        $state:expr,
        $prefix:expr,
        $kv:expr,
        $output:expr,
        $residual:expr,
        $mlp_input:expr,
        $mlp_moe:expr,
        $dispatch:ident
    ) => {{
        append_attention_kv_probe_summary($response, $prefix, $kv);
        append_attention_output_probe_summary($response, $prefix, $output);
        append_attention_residual_probe_summary($response, $prefix, $residual);
        if let Some(mlp_input) = $mlp_input {
            append_attention_mlp_input_probe_summary($response, $prefix, mlp_input);
            let start = Instant::now();
            if let Some(dispatch) = $dispatch($state, mlp_input).await? {
                $metrics.prefill_ms += duration_ms(start.elapsed());
                $response.push_str(&format!(
                    " {}_mlp_input_dispatch transport={} owners=[{}] requests={} rows={} routes={} partials={} checksum={:.6} first={:.6} last={:.6}",
                    $prefix,
                    dispatch.transport,
                    dispatch.owners,
                    dispatch.request_count,
                    dispatch.row_count,
                    dispatch.route_count,
                    dispatch.partial_count,
                    dispatch.checksum,
                    dispatch.first,
                    dispatch.last
                ));
            } else {
                $metrics.prefill_ms += duration_ms(start.elapsed());
            }
        }
        append_attention_mlp_input_moe_probe_summary($response, $prefix, $mlp_moe);
    }};
}

pub(super) async fn append_extra_attention_probe_summaries(
    response: &mut String,
    metrics: &mut BackendMetrics,
    state: &ApiState,
    probe: &RealSliceLogitsProbe,
) -> Result<(), ApiError> {
    append_attention_layer_probe_summary!(
        response,
        metrics,
        state,
        "prefill_next_layer_attention",
        &probe.prefill_next_layer_attention_kv_probe,
        &probe.prefill_next_layer_attention_output_probe,
        &probe.prefill_next_layer_attention_residual_probe,
        &probe.prefill_next_layer_attention_mlp_input_probe,
        &probe.prefill_next_layer_attention_mlp_input_moe_probe,
        dispatch_real_prefill_next_layer_attention_mlp_input_probe
    );
    append_attention_layer_probe_summary!(
        response,
        metrics,
        state,
        "prefill_following_layer_attention",
        &probe.prefill_following_layer_attention_kv_probe,
        &probe.prefill_following_layer_attention_output_probe,
        &probe.prefill_following_layer_attention_residual_probe,
        &probe.prefill_following_layer_attention_mlp_input_probe,
        &probe.prefill_following_layer_attention_mlp_input_moe_probe,
        dispatch_real_prefill_following_layer_attention_mlp_input_probe
    );
    append_attention_layer_probe_summary!(
        response,
        metrics,
        state,
        "prefill_subsequent_layer_attention",
        &probe.prefill_subsequent_layer_attention_kv_probe,
        &probe.prefill_subsequent_layer_attention_output_probe,
        &probe.prefill_subsequent_layer_attention_residual_probe,
        &probe.prefill_subsequent_layer_attention_mlp_input_probe,
        &probe.prefill_subsequent_layer_attention_mlp_input_moe_probe,
        dispatch_real_prefill_subsequent_layer_attention_mlp_input_probe
    );
    append_attention_layer_probe_summary!(
        response,
        metrics,
        state,
        "prefill_deeper_layer_attention",
        &probe.prefill_deeper_layer_attention_kv_probe,
        &probe.prefill_deeper_layer_attention_output_probe,
        &probe.prefill_deeper_layer_attention_residual_probe,
        &probe.prefill_deeper_layer_attention_mlp_input_probe,
        &probe.prefill_deeper_layer_attention_mlp_input_moe_probe,
        dispatch_real_prefill_deeper_layer_attention_mlp_input_probe
    );
    append_attention_layer_probe_summary!(
        response,
        metrics,
        state,
        "prefill_further_layer_attention",
        &probe.prefill_further_layer_attention_kv_probe,
        &probe.prefill_further_layer_attention_output_probe,
        &probe.prefill_further_layer_attention_residual_probe,
        &probe.prefill_further_layer_attention_mlp_input_probe,
        &probe.prefill_further_layer_attention_mlp_input_moe_probe,
        dispatch_real_prefill_further_layer_attention_mlp_input_probe
    );
    append_attention_layer_probe_summary!(
        response,
        metrics,
        state,
        "prefill_extended_layer_attention",
        &probe.prefill_extended_layer_attention_kv_probe,
        &probe.prefill_extended_layer_attention_output_probe,
        &probe.prefill_extended_layer_attention_residual_probe,
        &probe.prefill_extended_layer_attention_mlp_input_probe,
        &probe.prefill_extended_layer_attention_mlp_input_moe_probe,
        dispatch_real_prefill_extended_layer_attention_mlp_input_probe
    );
    append_attention_layer_probe_summary!(
        response,
        metrics,
        state,
        "prefill_layer10_attention",
        &probe.prefill_layer10_attention_kv_probe,
        &probe.prefill_layer10_attention_output_probe,
        &probe.prefill_layer10_attention_residual_probe,
        &probe.prefill_layer10_attention_mlp_input_probe,
        &probe.prefill_layer10_attention_mlp_input_moe_probe,
        dispatch_real_prefill_layer10_attention_mlp_input_probe
    );
    append_attention_layer_probe_summary!(
        response,
        metrics,
        state,
        "prefill_layer11_attention",
        &probe.prefill_layer11_attention_kv_probe,
        &probe.prefill_layer11_attention_output_probe,
        &probe.prefill_layer11_attention_residual_probe,
        &probe.prefill_layer11_attention_mlp_input_probe,
        &probe.prefill_layer11_attention_mlp_input_moe_probe,
        dispatch_real_prefill_layer11_attention_mlp_input_probe
    );
    append_attention_layer_probe_summary!(
        response,
        metrics,
        state,
        "prefill_layer12_attention",
        &probe.prefill_layer12_attention_kv_probe,
        &probe.prefill_layer12_attention_output_probe,
        &probe.prefill_layer12_attention_residual_probe,
        &probe.prefill_layer12_attention_mlp_input_probe,
        &probe.prefill_layer12_attention_mlp_input_moe_probe,
        dispatch_real_prefill_layer12_attention_mlp_input_probe
    );
    append_attention_layer_probe_summary!(
        response,
        metrics,
        state,
        "prefill_layer13_attention",
        &probe.prefill_layer13_attention_kv_probe,
        &probe.prefill_layer13_attention_output_probe,
        &probe.prefill_layer13_attention_residual_probe,
        &probe.prefill_layer13_attention_mlp_input_probe,
        &probe.prefill_layer13_attention_mlp_input_moe_probe,
        dispatch_real_prefill_layer13_attention_mlp_input_probe
    );
    Ok(())
}

fn append_attention_kv_probe_summary(
    response: &mut String,
    prefix: &str,
    probe: &Option<RealSlicePrefillNextLayerAttentionKvProbe>,
) {
    if let Some(probe) = probe {
        response.push_str(&format!(
            " {prefix}_kv_probe source_layer={} layer={} rows={} tokens={}..{} hidden_width={} prefix={} source={} norm_checksum={:.6} norm_l2={:.6} q_rank={} kv_rank={} q_outputs={} kv_outputs={} q_checksum={:.6} kv_checksum={:.6} kv_rope_checksum={:.6} q_first={:.6} kv_first={:.6} kv_last={:.6}",
            probe.source_layer_id,
            probe.layer_id,
            probe.row_count,
            probe.token_start,
            probe.token_start + probe.token_count as u32,
            probe.hidden_width,
            probe.input_prefix_count,
            probe.residual_source,
            probe.normalized_checksum,
            probe.normalized_l2_norm,
            probe.q_lora_rank,
            probe.kv_lora_rank,
            probe.q_output_count,
            probe.kv_output_count,
            probe.q_output_checksum,
            probe.kv_output_checksum,
            probe.kv_rope_checksum,
            probe.q_first_output,
            probe.kv_first_output,
            probe.kv_last_output
        ));
    }
}

fn append_attention_output_probe_summary(
    response: &mut String,
    prefix: &str,
    probe: &Option<RealSlicePrefillAttentionOutputProbe>,
) {
    if let Some(probe) = probe {
        response.push_str(&format!(
            " {prefix}_output_probe layer={} rows={} inputs={} outputs={} source={} input_checksum={:.6} checksum={:.6} first={:.6} last={:.6}",
            probe.layer_id,
            probe.row_count,
            probe.input_count,
            probe.output_count,
            probe.context_source,
            probe.input_checksum,
            probe.output_checksum,
            probe.first_output,
            probe.last_output
        ));
    }
}

fn append_attention_residual_probe_summary(
    response: &mut String,
    prefix: &str,
    probe: &Option<RealSlicePrefillAttentionResidualProbe>,
) {
    if let Some(probe) = probe {
        response.push_str(&format!(
            " {prefix}_residual_probe layer={} rows={} outputs={} residual_source={} attention_source={} residual_checksum={:.6} branch_checksum={:.6} checksum={:.6} first={:.6} last={:.6}",
            probe.layer_id,
            probe.row_count,
            probe.output_count,
            probe.residual_source,
            probe.attention_source,
            probe.residual_checksum,
            probe.branch_checksum,
            probe.output_checksum,
            probe.first_output,
            probe.last_output
        ));
    }
}

fn append_attention_mlp_input_probe_summary(
    response: &mut String,
    prefix: &str,
    probe: &RealSlicePrefillAttentionMlpInputProbe,
) {
    let first_route = probe
        .route_rows
        .first()
        .and_then(|row| row.routes.first())
        .map(|route| format!("{}@{}", route.expert_id, route.owner))
        .unwrap_or_else(|| "none".to_owned());
    response.push_str(&format!(
        " {prefix}_mlp_input_probe layer={} rows={} hidden_width={} prefix={} source={} route_rows={} first_route={} checksum={:.6} norm_l2={:.6} first={:.6} last={:.6}",
        probe.layer_id,
        probe.row_count,
        probe.hidden_width,
        probe.attention_prefix_count,
        probe.residual_source,
        probe.route_rows.len(),
        first_route,
        probe.normalized_checksum,
        probe.normalized_l2_norm,
        probe.first_normalized,
        probe.last_normalized
    ));
}

fn append_attention_mlp_input_moe_probe_summary(
    response: &mut String,
    prefix: &str,
    probe: &Option<RealSlicePrefillMlpInputMoeProbe>,
) {
    if let Some(probe) = probe {
        response.push_str(&format!(
            " {prefix}_mlp_input_moe_probe layer={} rows={} outputs={} routes={} source={} routed_checksum={:.6} shared_checksum={:.6} branch_checksum={:.6} residual_checksum={:.6} checksum={:.6} first={:.6} last={:.6}",
            probe.layer_id,
            probe.row_count,
            probe.output_count,
            probe.route_count,
            probe.residual_source,
            probe.routed_output_checksum,
            probe.shared_output_checksum,
            probe.branch_checksum,
            probe.residual_checksum,
            probe.output_checksum,
            probe.first_output,
            probe.last_output
        ));
    }
}
