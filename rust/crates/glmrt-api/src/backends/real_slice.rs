use crate::metrics::BackendMetrics;
use crate::{runtime_error, ApiError, ApiState, BackendCompletion};

mod dispatch;
mod response_attention;
mod response_layers;
mod response_mlp;
mod response_prefill;

use response_attention::append_attention_probe_summaries;
use response_layers::append_extra_attention_probe_summaries;
use response_mlp::{append_dense_probe_summaries, append_mlp_probe_summaries};
use response_prefill::append_prefill_probe_summaries;

pub(crate) async fn real_glm_slice_completion(
    state: &ApiState,
) -> Result<BackendCompletion, ApiError> {
    let slice = state
        .config
        .real_slice
        .as_ref()
        .ok_or_else(|| runtime_error("real-glm-slice backend has no loaded tensor summary"))?;
    let first = slice
        .tensors
        .first()
        .map(|tensor| {
            format!(
                "{}:{}",
                tensor.name,
                &tensor.sha256[..16.min(tensor.sha256.len())]
            )
        })
        .unwrap_or_else(|| "none".to_owned());
    let mut response = format!(
        "real glm slice loaded tensors={} bytes={} first={}",
        slice.tensor_count, slice.total_bytes, first
    );
    let mut metrics = BackendMetrics::default();
    if let Some(probe) = &slice.logits_probe {
        let logits_summary = format!(
            " logits_probe prompt_tokens={} hidden_token={} candidates={}..{} embedding_top_token={} embedding_top_logit={:.6} rmsnorm_top_token={} rmsnorm_top_logit={:.6}",
            probe.probe_prompt_token_ids.len(),
            probe.hidden_token_id,
            probe.candidate_start_token_id,
            probe.candidate_start_token_id + probe.candidate_count as u32,
            probe.embedding_top_token_id,
            probe.embedding_top_logit,
            probe.top_token_id,
            probe.top_logit
        );
        response.push_str(&logits_summary);
        append_prefill_probe_summaries(&mut response, &mut metrics, state, probe).await?;
        append_mlp_probe_summaries(&mut response, &mut metrics, state, probe).await?;
        append_attention_probe_summaries(&mut response, &mut metrics, state, probe).await?;
        append_extra_attention_probe_summaries(&mut response, &mut metrics, state, probe).await?;
        append_dense_probe_summaries(&mut response, probe);
    }
    Ok(BackendCompletion {
        content: response,
        reasoning_content: None,
        completion_tokens: None,
        stream_chunks: None,
        metrics,
    })
}
