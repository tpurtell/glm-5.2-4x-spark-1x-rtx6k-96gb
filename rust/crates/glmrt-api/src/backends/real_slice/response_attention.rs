use std::time::Instant;

use crate::metrics::BackendMetrics;
use crate::{duration_ms, ApiError, ApiState, RealSliceLogitsProbe};

use super::dispatch::dispatch_real_prefill_attention_mlp_input_probe;

pub(super) async fn append_attention_probe_summaries(
    response: &mut String,
    metrics: &mut BackendMetrics,
    state: &ApiState,
    probe: &RealSliceLogitsProbe,
) -> Result<(), ApiError> {
    if let Some(attn) = &probe.attention_probe {
        response.push_str(&format!(
            " attention_probe layer={} q_rank={} kv_rank={} q_outputs={} kv_outputs={} q_checksum={:.6} kv_checksum={:.6} q_first={:.6} kv_first={:.6}",
            attn.layer_id,
            attn.q_lora_rank,
            attn.kv_lora_rank,
            attn.q_output_count,
            attn.kv_output_count,
            attn.q_output_checksum,
            attn.kv_output_checksum,
            attn.q_first_output,
            attn.kv_first_output
        ));
    }
    if let Some(prefill_attn) = &probe.prefill_attention_kv_probe {
        response.push_str(&format!(
            " prefill_attention_kv_probe layer={} rows={} tokens={}..{} q_rank={} kv_rank={} q_outputs={} kv_outputs={} kv_writes={}/{} q_checksum={:.6} kv_checksum={:.6} kv_rope_checksum={:.6} q_first={:.6} kv_first={:.6} kv_last={:.6}",
            prefill_attn.layer_id,
            prefill_attn.row_count,
            prefill_attn.token_start,
            prefill_attn.token_start + prefill_attn.token_count as u32,
            prefill_attn.q_lora_rank,
            prefill_attn.kv_lora_rank,
            prefill_attn.q_output_count,
            prefill_attn.kv_output_count,
            prefill_attn.kv_written_count,
            prefill_attn.kv_write_count,
            prefill_attn.q_output_checksum,
            prefill_attn.kv_output_checksum,
            prefill_attn.kv_rope_checksum,
            prefill_attn.q_first_output,
            prefill_attn.kv_first_output,
            prefill_attn.kv_last_output
        ));
    }
    if let Some(prefill_attn_output) = &probe.prefill_attention_output_probe {
        response.push_str(&format!(
            " prefill_attention_output_probe layer={} rows={} inputs={} outputs={} source={} input_checksum={:.6} checksum={:.6} first={:.6} last={:.6}",
            prefill_attn_output.layer_id,
            prefill_attn_output.row_count,
            prefill_attn_output.input_count,
            prefill_attn_output.output_count,
            prefill_attn_output.context_source,
            prefill_attn_output.input_checksum,
            prefill_attn_output.output_checksum,
            prefill_attn_output.first_output,
            prefill_attn_output.last_output
        ));
    }
    if let Some(prefill_attn_residual) = &probe.prefill_attention_residual_probe {
        response.push_str(&format!(
            " prefill_attention_residual_probe layer={} rows={} outputs={} residual_source={} attention_source={} residual_checksum={:.6} branch_checksum={:.6} checksum={:.6} first={:.6} last={:.6}",
            prefill_attn_residual.layer_id,
            prefill_attn_residual.row_count,
            prefill_attn_residual.output_count,
            prefill_attn_residual.residual_source,
            prefill_attn_residual.attention_source,
            prefill_attn_residual.residual_checksum,
            prefill_attn_residual.branch_checksum,
            prefill_attn_residual.output_checksum,
            prefill_attn_residual.first_output,
            prefill_attn_residual.last_output
        ));
    }
    if let Some(prefill_attn_mlp_input) = &probe.prefill_attention_mlp_input_probe {
        let first_route = prefill_attn_mlp_input
            .route_rows
            .first()
            .and_then(|row| row.routes.first())
            .map(|route| format!("{}@{}", route.expert_id, route.owner))
            .unwrap_or_else(|| "none".to_owned());
        response.push_str(&format!(
            " prefill_attention_mlp_input_probe layer={} rows={} hidden_width={} prefix={} source={} route_rows={} first_route={} checksum={:.6} norm_l2={:.6} first={:.6} last={:.6}",
            prefill_attn_mlp_input.layer_id,
            prefill_attn_mlp_input.row_count,
            prefill_attn_mlp_input.hidden_width,
            prefill_attn_mlp_input.attention_prefix_count,
            prefill_attn_mlp_input.residual_source,
            prefill_attn_mlp_input.route_rows.len(),
            first_route,
            prefill_attn_mlp_input.normalized_checksum,
            prefill_attn_mlp_input.normalized_l2_norm,
            prefill_attn_mlp_input.first_normalized,
            prefill_attn_mlp_input.last_normalized
        ));
        let prefill_attn_mlp_start = Instant::now();
        if let Some(dispatch) =
            dispatch_real_prefill_attention_mlp_input_probe(state, prefill_attn_mlp_input).await?
        {
            metrics.prefill_ms += duration_ms(prefill_attn_mlp_start.elapsed());
            response.push_str(&format!(
                " prefill_attention_mlp_input_dispatch transport={} owners=[{}] requests={} rows={} routes={} partials={} checksum={:.6} first={:.6} last={:.6}",
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
            metrics.prefill_ms += duration_ms(prefill_attn_mlp_start.elapsed());
        }
    }
    if let Some(prefill_attn_mlp_moe) = &probe.prefill_attention_mlp_input_moe_probe {
        response.push_str(&format!(
            " prefill_attention_mlp_input_moe_probe layer={} rows={} outputs={} routes={} source={} routed_checksum={:.6} shared_checksum={:.6} branch_checksum={:.6} residual_checksum={:.6} checksum={:.6} first={:.6} last={:.6}",
            prefill_attn_mlp_moe.layer_id,
            prefill_attn_mlp_moe.row_count,
            prefill_attn_mlp_moe.output_count,
            prefill_attn_mlp_moe.route_count,
            prefill_attn_mlp_moe.residual_source,
            prefill_attn_mlp_moe.routed_output_checksum,
            prefill_attn_mlp_moe.shared_output_checksum,
            prefill_attn_mlp_moe.branch_checksum,
            prefill_attn_mlp_moe.residual_checksum,
            prefill_attn_mlp_moe.output_checksum,
            prefill_attn_mlp_moe.first_output,
            prefill_attn_mlp_moe.last_output
        ));
    }
    Ok(())
}
