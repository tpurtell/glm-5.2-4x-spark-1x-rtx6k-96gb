use uuid::Uuid;

use std::path::Path;
use std::sync::Arc;

use crate::backends::{
    real_glm_full_completion, real_glm_slice_completion, synthetic_glm_layer_completion,
    tiny_backend_completion,
};
use crate::constrained::request_constraint;
use crate::metrics::CompletionMetrics;
use crate::request::{
    prompt_text, real_glm_full_request_prompt_text, request_image_sources, request_max_tokens,
    request_sampling_params, rough_token_count, stop_strings, tool_calls_enabled, unix_timestamp,
    validate_request,
};
use crate::tooling::parse_glm_tool_calls;
use crate::{
    ApiBackend, ApiError, ApiState, ApiTransport, AssistantMessage, ChatChoice,
    ChatCompletionRequest, ChatCompletionResponse, RealFullVisionEmbedding, ToolCall, Usage,
};

#[derive(Debug)]
pub(crate) struct CompletionOutput {
    pub(crate) id: String,
    pub(crate) created: u64,
    pub(crate) model: String,
    pub(crate) content: Option<String>,
    pub(crate) reasoning_content: Option<String>,
    pub(crate) stream_chunks: Option<Vec<String>>,
    pub(crate) tool_calls: Option<Vec<ToolCall>>,
    pub(crate) finish_reason: String,
    pub(crate) usage: Usage,
    pub(crate) metrics: CompletionMetrics,
}

impl CompletionOutput {
    pub(crate) fn into_response_body(self) -> ChatCompletionResponse {
        ChatCompletionResponse {
            id: self.id,
            object: "chat.completion",
            created: self.created,
            model: self.model,
            choices: vec![ChatChoice {
                index: 0,
                message: AssistantMessage {
                    role: "assistant",
                    content: self.content,
                    reasoning_content: self.reasoning_content,
                    tool_calls: self.tool_calls,
                },
                finish_reason: self.finish_reason,
            }],
            usage: self.usage,
            metrics: self.metrics,
        }
    }
}

pub(crate) async fn build_completion(
    state: &ApiState,
    request: ChatCompletionRequest,
) -> Result<CompletionOutput, ApiError> {
    validate_request(&request)?;
    let plain_prompt = prompt_text(&request.messages);
    let max_tokens = request_max_tokens(&request);
    let tools_enabled = tool_calls_enabled(&request);
    let created = unix_timestamp();
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let backend = selected_backend(state, &request);
    let constraint = request_constraint(&request)?;
    if constraint.is_some() && backend != ApiBackend::RealGlmFull {
        return Err(crate::invalid_request(
            "constrained response formats and strict tools require the real GLM full backend",
            Some("model"),
        ));
    }
    let prompt = if backend == ApiBackend::RealGlmFull {
        real_glm_full_request_prompt_text(&request)
    } else {
        plain_prompt
    };
    let image_sources = request_image_sources(&request)?;
    let (prompt_tokens, prepared_prompt_token_ids, vision_embeddings): (
        usize,
        Option<Arc<Vec<usize>>>,
        Option<Arc<Vec<RealFullVisionEmbedding>>>,
    ) = if image_sources.is_empty() {
        (prompt_token_count(state, backend, &prompt), None, None)
    } else {
        if backend != ApiBackend::RealGlmFull {
            return Err(crate::invalid_request(
                "image input is supported only by the real GLM full backend",
                Some("model"),
            ));
        }
        let initial_prompt_token_ids =
            prompt_token_ids(state, backend, &prompt).ok_or_else(|| {
                crate::runtime_error("vision input requires the loaded GLM tokenizer")
            })?;
        let prepared =
            crate::vision::prepare_vision_prompt(initial_prompt_token_ids, &image_sources)?;
        (
            prepared.prompt_token_ids.len(),
            Some(prepared.prompt_token_ids),
            Some(prepared.embeddings),
        )
    };
    let transport_backend = transport_name(state.config.transport);

    let backend_completion = match backend {
        ApiBackend::Tiny => tiny_backend_completion(&prompt, prompt_tokens, max_tokens),
        ApiBackend::SyntheticGlmLayer => {
            synthetic_glm_layer_completion(state, &prompt, prompt_tokens).await?
        }
        ApiBackend::RealGlmSlice => real_glm_slice_completion(state).await?,
        ApiBackend::RealGlmFull => {
            real_glm_full_completion(
                state,
                &prompt,
                prompt_tokens,
                prepared_prompt_token_ids,
                vision_embeddings,
                max_tokens,
                request.min_tokens.unwrap_or(0),
                request.ignore_eos.unwrap_or(false),
                request_sampling_params(&request),
                tools_enabled,
                constraint,
            )
            .await?
        }
    };
    let mut content = backend_completion.content;
    let reasoning_content = backend_completion.reasoning_content;
    let mut stream_chunks = backend_completion.stream_chunks;
    let mut completion_tokens = backend_completion.completion_tokens;
    let mut finish_reason = if completion_token_count(&content, completion_tokens) >= max_tokens {
        "length"
    } else {
        "stop"
    }
    .to_owned();
    if let Some(stop) = stop_strings(request.stop.as_ref())
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| content.find(stop).map(|idx| (idx, stop)))
        .min_by_key(|(idx, _)| *idx)
    {
        content.truncate(stop.0);
        completion_tokens = None;
        stream_chunks = None;
        finish_reason = "stop".to_owned();
    }
    let completion_tokens = completion_token_count(&content, completion_tokens);
    let parsed_tools = tools_enabled
        .then(|| parse_glm_tool_calls(&content, request.tools.as_deref().unwrap_or_default()));
    let (content, stream_chunks, tool_calls, finish_reason) = match parsed_tools {
        Some(parsed) if !parsed.tool_calls.is_empty() => (
            parsed.content,
            None,
            Some(parsed.tool_calls),
            "tool_calls".to_owned(),
        ),
        _ => (Some(content), stream_chunks, None, finish_reason),
    };
    let metrics = CompletionMetrics::from_backend(
        prompt_tokens,
        completion_tokens,
        backend_name(backend),
        transport_backend,
        backend_completion.metrics,
    );
    let usage = Usage::from_metrics(&metrics);
    Ok(CompletionOutput {
        id,
        created,
        model: request.model,
        content,
        reasoning_content,
        stream_chunks,
        tool_calls,
        finish_reason,
        usage,
        metrics,
    })
}

pub(crate) fn completion_token_count(
    content: &str,
    backend_completion_tokens: Option<usize>,
) -> usize {
    backend_completion_tokens.unwrap_or_else(|| rough_token_count(content))
}

pub(crate) fn selected_backend(state: &ApiState, request: &ChatCompletionRequest) -> ApiBackend {
    if request.model == "glmrt-synthetic-glm-layer" {
        ApiBackend::SyntheticGlmLayer
    } else if request.model.ends_with("-slice") {
        ApiBackend::RealGlmSlice
    } else if request.model.ends_with("-full") {
        ApiBackend::RealGlmFull
    } else {
        state.config.backend
    }
}

pub(crate) fn prompt_token_count(state: &ApiState, backend: ApiBackend, prompt: &str) -> usize {
    if backend != ApiBackend::RealGlmFull {
        return rough_token_count(prompt);
    }
    prompt_token_ids(state, backend, prompt)
        .map(|token_ids| token_ids.len())
        .unwrap_or_else(|| rough_token_count(prompt))
}

pub(crate) fn prompt_token_ids(
    state: &ApiState,
    backend: ApiBackend,
    prompt: &str,
) -> Option<Vec<usize>> {
    (backend == ApiBackend::RealGlmFull)
        .then_some(())
        .and_then(|_| state.config.real_full.as_ref())
        .and_then(|full| full.snapshot_path.as_deref())
        .and_then(|snapshot_path| {
            glmrt_loader::encode_tokenizer_text(Path::new(snapshot_path), prompt, false).ok()
        })
        .map(|summary| {
            summary
                .token_ids
                .into_iter()
                .map(|token_id| token_id as usize)
                .collect()
        })
}

pub(crate) fn backend_name(backend: ApiBackend) -> &'static str {
    match backend {
        ApiBackend::Tiny => "tiny",
        ApiBackend::SyntheticGlmLayer => "synthetic-glm-layer",
        ApiBackend::RealGlmSlice => "real-glm-slice",
        ApiBackend::RealGlmFull => "real-glm-full",
    }
}

pub(crate) fn transport_name(transport: ApiTransport) -> &'static str {
    transport.label()
}
