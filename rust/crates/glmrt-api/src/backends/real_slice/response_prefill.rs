use std::time::Instant;

use crate::metrics::BackendMetrics;
use crate::{duration_ms, ApiError, ApiState, RealSliceLogitsProbe};

use super::dispatch::{dispatch_real_prefill_mlp_input_probe, dispatch_real_prefill_probe};

pub(super) async fn append_prefill_probe_summaries(
    response: &mut String,
    metrics: &mut BackendMetrics,
    state: &ApiState,
    probe: &RealSliceLogitsProbe,
) -> Result<(), ApiError> {
    if let Some(prefill) = &probe.prefill_probe {
        let first_route = prefill
            .route_rows
            .first()
            .and_then(|row| row.routes.first())
            .map(|route| format!("{}@{}", route.expert_id, route.owner))
            .unwrap_or_else(|| "none".to_owned());
        response.push_str(&format!(
            " prefill_probe tokens={} chunk={} hidden_width={} rows_bytes={} router_layer={} route_rows={} first_route={} rmsnorm_checksum={:.6} first={:.6} last={:.6}",
            prefill.prompt_token_count,
            prefill.chunk_token_count,
            prefill.hidden_width,
            prefill.embedding_rows_bytes,
            prefill.router_layer_id,
            prefill.route_rows.len(),
            first_route,
            prefill.rmsnorm_checksum,
            prefill.first_rmsnorm_value,
            prefill.last_rmsnorm_value
        ));
        metrics.prefill_tokens = prefill.prompt_token_count;
        metrics.prefill_chunk_count = if prefill.chunk_token_count == 0 {
            0
        } else {
            prefill
                .prompt_token_count
                .div_ceil(prefill.chunk_token_count)
        };
        metrics.layerwave_prefill_rows = prefill.route_rows.len();
        let prefill_start = Instant::now();
        if let Some(dispatch) = dispatch_real_prefill_probe(state, prefill).await? {
            metrics.prefill_ms += duration_ms(prefill_start.elapsed());
            metrics.layerwave_prefill_rows = dispatch.row_count;
            response.push_str(&format!(
                " prefill_dispatch transport={} owners=[{}] requests={} rows={} routes={} partials={} checksum={:.6} first={:.6} last={:.6}",
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
            metrics.prefill_ms += duration_ms(prefill_start.elapsed());
        }
        let first_mlp_route = prefill
            .mlp_input_route_rows
            .first()
            .and_then(|row| row.routes.first())
            .map(|route| format!("{}@{}", route.expert_id, route.owner))
            .unwrap_or_else(|| "none".to_owned());
        response.push_str(&format!(
            " prefill_mlp_input_probe tokens={} chunk={} hidden_width={} layer={} source={} route_rows={} first_route={} checksum={:.6} first={:.6} last={:.6}",
            prefill.prompt_token_count,
            prefill.chunk_token_count,
            prefill.hidden_width,
            prefill.router_layer_id,
            prefill.mlp_input_residual_source,
            prefill.mlp_input_route_rows.len(),
            first_mlp_route,
            prefill.mlp_input_checksum,
            prefill.first_mlp_input_value,
            prefill.last_mlp_input_value
        ));
        let prefill_mlp_start = Instant::now();
        if let Some(dispatch) = dispatch_real_prefill_mlp_input_probe(state, prefill).await? {
            metrics.prefill_ms += duration_ms(prefill_mlp_start.elapsed());
            response.push_str(&format!(
                " prefill_mlp_input_dispatch transport={} owners=[{}] requests={} rows={} routes={} partials={} checksum={:.6} first={:.6} last={:.6}",
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
            metrics.prefill_ms += duration_ms(prefill_mlp_start.elapsed());
        }
    }
    Ok(())
}
