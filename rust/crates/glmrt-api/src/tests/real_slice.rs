use super::*;

mod assertions;
mod fixture;

use assertions::assert_real_slice_content;
use fixture::real_slice_info_fixture;

#[tokio::test]
async fn real_glm_slice_returns_loaded_tensor_summary() {
    let mut state = test_state(ApiBackend::RealGlmSlice, ApiTransport::Inproc);
    state.config.real_slice = Some(real_slice_info_fixture());
    let mut request = base_request("Use real slice.");
    request.model = format!("{}-slice", DEFAULT_MODEL_ID);
    let output = build_completion(&state, request).await.unwrap();
    let content = output.content.unwrap();
    assert_real_slice_content(&content);
    assert_eq!(output.metrics.backend_mode, "real-glm-slice");
    assert_eq!(output.metrics.transport_backend, "inproc");
    assert_eq!(output.metrics.prefill_chunk_count, 1);
    assert_eq!(output.metrics.layerwave_prefill_rows, 2);
    assert_eq!(output.metrics.layerwave_decode_rows, 1);
    assert_eq!(output.metrics.prompt_tokens, output.usage.prompt_tokens);
}
