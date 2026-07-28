use crate::{ApiError, ApiState, RealSlicePrefillAttentionMlpInputProbe};

use super::prefill::{dispatch_real_prefill_rows, RealPrefillDispatchSummary};

pub(in crate::backends::real_slice) async fn dispatch_real_prefill_attention_mlp_input_probe(
    state: &ApiState,
    probe: &RealSlicePrefillAttentionMlpInputProbe,
) -> Result<Option<RealPrefillDispatchSummary>, ApiError> {
    dispatch_real_prefill_rows(
        state,
        "real prefill attention MLP input",
        "phase0-real-prefill-attention-mlp-input-dispatch",
        probe.layer_id,
        probe.row_count,
        &probe.normalized_rows,
        &probe.route_rows,
    )
    .await
}

pub(in crate::backends::real_slice) async fn dispatch_real_prefill_next_layer_attention_mlp_input_probe(
    state: &ApiState,
    probe: &RealSlicePrefillAttentionMlpInputProbe,
) -> Result<Option<RealPrefillDispatchSummary>, ApiError> {
    dispatch_real_prefill_rows(
        state,
        "real prefill next-layer attention MLP input",
        "phase0-real-prefill-next-layer-attention-mlp-input-dispatch",
        probe.layer_id,
        probe.row_count,
        &probe.normalized_rows,
        &probe.route_rows,
    )
    .await
}

pub(in crate::backends::real_slice) async fn dispatch_real_prefill_following_layer_attention_mlp_input_probe(
    state: &ApiState,
    probe: &RealSlicePrefillAttentionMlpInputProbe,
) -> Result<Option<RealPrefillDispatchSummary>, ApiError> {
    dispatch_real_prefill_rows(
        state,
        "real prefill following-layer attention MLP input",
        "phase0-real-prefill-following-layer-attention-mlp-input-dispatch",
        probe.layer_id,
        probe.row_count,
        &probe.normalized_rows,
        &probe.route_rows,
    )
    .await
}

pub(in crate::backends::real_slice) async fn dispatch_real_prefill_subsequent_layer_attention_mlp_input_probe(
    state: &ApiState,
    probe: &RealSlicePrefillAttentionMlpInputProbe,
) -> Result<Option<RealPrefillDispatchSummary>, ApiError> {
    dispatch_real_prefill_rows(
        state,
        "real prefill subsequent-layer attention MLP input",
        "phase0-real-prefill-subsequent-layer-attention-mlp-input-dispatch",
        probe.layer_id,
        probe.row_count,
        &probe.normalized_rows,
        &probe.route_rows,
    )
    .await
}

pub(in crate::backends::real_slice) async fn dispatch_real_prefill_deeper_layer_attention_mlp_input_probe(
    state: &ApiState,
    probe: &RealSlicePrefillAttentionMlpInputProbe,
) -> Result<Option<RealPrefillDispatchSummary>, ApiError> {
    dispatch_real_prefill_rows(
        state,
        "real prefill deeper-layer attention MLP input",
        "phase0-real-prefill-deeper-layer-attention-mlp-input-dispatch",
        probe.layer_id,
        probe.row_count,
        &probe.normalized_rows,
        &probe.route_rows,
    )
    .await
}

pub(in crate::backends::real_slice) async fn dispatch_real_prefill_further_layer_attention_mlp_input_probe(
    state: &ApiState,
    probe: &RealSlicePrefillAttentionMlpInputProbe,
) -> Result<Option<RealPrefillDispatchSummary>, ApiError> {
    dispatch_real_prefill_rows(
        state,
        "real prefill further-layer attention MLP input",
        "phase0-real-prefill-further-layer-attention-mlp-input-dispatch",
        probe.layer_id,
        probe.row_count,
        &probe.normalized_rows,
        &probe.route_rows,
    )
    .await
}

pub(in crate::backends::real_slice) async fn dispatch_real_prefill_extended_layer_attention_mlp_input_probe(
    state: &ApiState,
    probe: &RealSlicePrefillAttentionMlpInputProbe,
) -> Result<Option<RealPrefillDispatchSummary>, ApiError> {
    dispatch_real_prefill_rows(
        state,
        "real prefill extended-layer attention MLP input",
        "phase0-real-prefill-extended-layer-attention-mlp-input-dispatch",
        probe.layer_id,
        probe.row_count,
        &probe.normalized_rows,
        &probe.route_rows,
    )
    .await
}

pub(in crate::backends::real_slice) async fn dispatch_real_prefill_layer10_attention_mlp_input_probe(
    state: &ApiState,
    probe: &RealSlicePrefillAttentionMlpInputProbe,
) -> Result<Option<RealPrefillDispatchSummary>, ApiError> {
    dispatch_real_prefill_rows(
        state,
        "real prefill layer10 attention MLP input",
        "phase0-real-prefill-layer10-attention-mlp-input-dispatch",
        probe.layer_id,
        probe.row_count,
        &probe.normalized_rows,
        &probe.route_rows,
    )
    .await
}

pub(in crate::backends::real_slice) async fn dispatch_real_prefill_layer11_attention_mlp_input_probe(
    state: &ApiState,
    probe: &RealSlicePrefillAttentionMlpInputProbe,
) -> Result<Option<RealPrefillDispatchSummary>, ApiError> {
    dispatch_real_prefill_rows(
        state,
        "real prefill layer11 attention MLP input",
        "phase0-real-prefill-layer11-attention-mlp-input-dispatch",
        probe.layer_id,
        probe.row_count,
        &probe.normalized_rows,
        &probe.route_rows,
    )
    .await
}

pub(in crate::backends::real_slice) async fn dispatch_real_prefill_layer12_attention_mlp_input_probe(
    state: &ApiState,
    probe: &RealSlicePrefillAttentionMlpInputProbe,
) -> Result<Option<RealPrefillDispatchSummary>, ApiError> {
    dispatch_real_prefill_rows(
        state,
        "real prefill layer12 attention MLP input",
        "phase0-real-prefill-layer12-attention-mlp-input-dispatch",
        probe.layer_id,
        probe.row_count,
        &probe.normalized_rows,
        &probe.route_rows,
    )
    .await
}

pub(in crate::backends::real_slice) async fn dispatch_real_prefill_layer13_attention_mlp_input_probe(
    state: &ApiState,
    probe: &RealSlicePrefillAttentionMlpInputProbe,
) -> Result<Option<RealPrefillDispatchSummary>, ApiError> {
    dispatch_real_prefill_rows(
        state,
        "real prefill layer13 attention MLP input",
        "phase0-real-prefill-layer13-attention-mlp-input-dispatch",
        probe.layer_id,
        probe.row_count,
        &probe.normalized_rows,
        &probe.route_rows,
    )
    .await
}
