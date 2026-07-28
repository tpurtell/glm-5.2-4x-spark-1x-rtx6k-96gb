use serde_json::Value;
use std::collections::HashSet;
use std::env;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{invalid_request, ApiError};
use crate::tooling::{glm_tool_schema_json, render_glm_tool_call};
use crate::{
    ChatCompletionRequest, ChatMessage, ChatTool, RealFullSamplingParams, StopSpec, ToolChoice,
};

const REAL_FULL_ENABLE_THINKING_ENV: &str = "GLMRT_REAL_FULL_ENABLE_THINKING";

pub(crate) fn validate_request(request: &ChatCompletionRequest) -> Result<(), ApiError> {
    if request.model.trim().is_empty() {
        return Err(invalid_request("model must not be empty", Some("model")));
    }
    if request.messages.is_empty() {
        return Err(invalid_request(
            "messages must contain at least one item",
            Some("messages"),
        ));
    }
    let mut prior_tool_call_ids = HashSet::new();
    for (idx, message) in request.messages.iter().enumerate() {
        match message.role.as_str() {
            "system" | "user" | "assistant" | "tool" => {}
            _ => {
                return Err(invalid_request(
                    format!("unsupported message role {}", message.role),
                    Some(format!("messages[{idx}].role")),
                ));
            }
        }
        if message.tool_calls.is_some() && message.role != "assistant" {
            return Err(invalid_request(
                "tool_calls are only valid on assistant messages",
                Some(format!("messages[{idx}].tool_calls")),
            ));
        }
        if message.tool_call_id.is_some() && message.role != "tool" {
            return Err(invalid_request(
                "tool_call_id is only valid on tool messages",
                Some(format!("messages[{idx}].tool_call_id")),
            ));
        }
        if let Some(tool_calls) = &message.tool_calls {
            for (call_idx, tool_call) in tool_calls.iter().enumerate() {
                let param = format!("messages[{idx}].tool_calls[{call_idx}]");
                if tool_call.id.trim().is_empty() {
                    return Err(invalid_request(
                        "tool call id must not be empty",
                        Some(format!("{param}.id")),
                    ));
                }
                if !prior_tool_call_ids.insert(tool_call.id.clone()) {
                    return Err(invalid_request(
                        format!("duplicate tool call id {}", tool_call.id),
                        Some(format!("{param}.id")),
                    ));
                }
                if tool_call.tool_type != "function" {
                    return Err(invalid_request(
                        "only function tool calls are supported",
                        Some(format!("{param}.type")),
                    ));
                }
                if tool_call.function.name.trim().is_empty() {
                    return Err(invalid_request(
                        "tool call function name must not be empty",
                        Some(format!("{param}.function.name")),
                    ));
                }
                if !matches!(
                    serde_json::from_str::<Value>(&tool_call.function.arguments),
                    Ok(Value::Object(_))
                ) {
                    return Err(invalid_request(
                        "tool call arguments must be a JSON object string",
                        Some(format!("{param}.function.arguments")),
                    ));
                }
            }
        }
        if message.role == "tool" {
            let Some(tool_call_id) = message
                .tool_call_id
                .as_deref()
                .filter(|tool_call_id| !tool_call_id.trim().is_empty())
            else {
                return Err(invalid_request(
                    "tool messages require tool_call_id",
                    Some(format!("messages[{idx}].tool_call_id")),
                ));
            };
            if !prior_tool_call_ids.contains(tool_call_id) {
                return Err(invalid_request(
                    format!("tool_call_id {tool_call_id} has no preceding assistant tool call"),
                    Some(format!("messages[{idx}].tool_call_id")),
                ));
            }
        }
    }
    if request.max_tokens == Some(0) {
        return Err(invalid_request(
            "max_tokens must be greater than zero",
            Some("max_tokens"),
        ));
    }
    if request.max_completion_tokens == Some(0) {
        return Err(invalid_request(
            "max_completion_tokens must be greater than zero",
            Some("max_completion_tokens"),
        ));
    }
    if request
        .temperature
        .is_some_and(|value| !value.is_finite() || !(0.0..=2.0).contains(&value))
    {
        return Err(invalid_request(
            "temperature must be a finite number between 0 and 2",
            Some("temperature"),
        ));
    }
    if request
        .top_p
        .is_some_and(|value| !value.is_finite() || !(0.0 < value && value <= 1.0))
    {
        return Err(invalid_request(
            "top_p must be greater than 0 and at most 1",
            Some("top_p"),
        ));
    }
    if request
        .top_k
        .is_some_and(|value| !(1..=64).contains(&value))
    {
        return Err(invalid_request(
            "top_k must be between 1 and 64",
            Some("top_k"),
        ));
    }
    if let Some(tools) = &request.tools {
        let mut tool_names = HashSet::new();
        for (idx, tool) in tools.iter().enumerate() {
            if tool.tool_type != "function" {
                return Err(invalid_request(
                    "only function tools are supported",
                    Some(format!("tools[{idx}].type")),
                ));
            }
            if tool.function.name.trim().is_empty() {
                return Err(invalid_request(
                    "function tool name must not be empty",
                    Some(format!("tools[{idx}].function.name")),
                ));
            }
            if !tool_names.insert(tool.function.name.as_str()) {
                return Err(invalid_request(
                    format!("duplicate function tool name {}", tool.function.name),
                    Some(format!("tools[{idx}].function.name")),
                ));
            }
            if tool
                .function
                .parameters
                .as_ref()
                .is_some_and(|parameters| !parameters.is_object())
            {
                return Err(invalid_request(
                    "function tool parameters must be a JSON Schema object",
                    Some(format!("tools[{idx}].function.parameters")),
                ));
            }
        }
    }
    let tools = request.tools.as_deref().unwrap_or(&[]);
    match &request.tool_choice {
        Some(ToolChoice::Mode(mode)) if mode == "none" || mode == "auto" => {}
        Some(ToolChoice::Mode(mode)) if mode == "required" => {
            if tools.is_empty() {
                return Err(invalid_request(
                    "tool_choice required but no tools were provided",
                    Some("tool_choice"),
                ));
            }
        }
        Some(ToolChoice::Mode(mode)) => {
            return Err(invalid_request(
                format!("unsupported tool_choice mode {mode}"),
                Some("tool_choice"),
            ));
        }
        Some(ToolChoice::Specific {
            tool_type,
            function,
        }) => {
            if tool_type != "function" {
                return Err(invalid_request(
                    "specific tool_choice must have type function",
                    Some("tool_choice.type"),
                ));
            }
            if !tools.iter().any(|tool| tool.function.name == function.name) {
                return Err(invalid_request(
                    format!("requested tool {} was not provided", function.name),
                    Some("tool_choice.function.name"),
                ));
            }
        }
        None => {}
    }
    Ok(())
}

pub(crate) fn request_uses_greedy_sampling(request: &ChatCompletionRequest) -> bool {
    request.temperature.unwrap_or(0.0) == 0.0 || request.top_k == Some(1)
}

pub(crate) fn request_sampling_params(request: &ChatCompletionRequest) -> RealFullSamplingParams {
    static NEXT_SEED: AtomicU64 = AtomicU64::new(0);
    if request_uses_greedy_sampling(request) {
        return RealFullSamplingParams::greedy();
    }
    let generated_seed = || {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        now ^ NEXT_SEED.fetch_add(0x9e37_79b9_7f4a_7c15, Ordering::Relaxed)
    };
    RealFullSamplingParams::new(
        request.temperature.unwrap_or(1.0),
        request.top_p.unwrap_or(0.95),
        request.top_k.unwrap_or(50),
        request.seed.map_or_else(generated_seed, |seed| seed as u64),
    )
}

pub(crate) fn request_max_tokens(request: &ChatCompletionRequest) -> usize {
    request
        .max_completion_tokens
        .or(request.max_tokens)
        .unwrap_or(16)
}

pub(crate) fn tool_calls_enabled(request: &ChatCompletionRequest) -> bool {
    !matches!(request.tool_choice, Some(ToolChoice::Mode(ref mode)) if mode == "none")
        && request
            .tools
            .as_ref()
            .is_some_and(|tools| !tools.is_empty())
}

pub(crate) fn prompt_text(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|message| {
            let mut text = String::new();
            text.push_str(&message.role);
            text.push_str(": ");
            text.push_str(&message_content_text(&message.content));
            if let Some(name) = &message.name {
                text.push_str(" name=");
                text.push_str(name);
            }
            if let Some(tool_call_id) = &message.tool_call_id {
                text.push_str(" tool_call_id=");
                text.push_str(tool_call_id);
            }
            if let Some(tool_calls) = &message.tool_calls {
                text.push_str(" tool_calls=");
                text.push_str(&tool_calls.len().to_string());
            }
            text
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
pub(crate) fn real_glm_full_prompt_text(messages: &[ChatMessage]) -> String {
    render_real_glm_full_prompt(messages, &[], None, None)
}

pub(crate) fn real_glm_full_request_prompt_text(request: &ChatCompletionRequest) -> String {
    let tools = prompt_tools(request);
    let instruction = match &request.tool_choice {
        Some(ToolChoice::Mode(mode)) if mode == "required" => {
            Some("You must call at least one provided function.".to_owned())
        }
        Some(ToolChoice::Specific { function, .. }) => {
            Some(format!("You must call the function {}.", function.name))
        }
        _ => None,
    };
    render_real_glm_full_prompt(
        &request.messages,
        &tools,
        instruction.as_deref(),
        request_reasoning_effort_label(request),
    )
}

fn prompt_tools(request: &ChatCompletionRequest) -> Vec<&ChatTool> {
    if !tool_calls_enabled(request) {
        return Vec::new();
    }
    match &request.tool_choice {
        Some(ToolChoice::Specific { function, .. }) => request
            .tools
            .as_deref()
            .unwrap_or_default()
            .iter()
            .filter(|tool| tool.function.name == function.name)
            .collect(),
        _ => request
            .tools
            .as_deref()
            .unwrap_or_default()
            .iter()
            .collect(),
    }
}

fn render_real_glm_full_prompt(
    messages: &[ChatMessage],
    tools: &[&ChatTool],
    tool_choice_instruction: Option<&str>,
    reasoning_effort: Option<&str>,
) -> String {
    let mut prompt = String::new();
    prompt.push_str("[gMASK]<sop>");
    if let Some(reasoning_effort) = reasoning_effort {
        prompt.push_str("<|system|>Reasoning Effort: ");
        prompt.push_str(reasoning_effort);
    }
    if !tools.is_empty() {
        prompt.push_str(
            "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
        );
        for tool in tools {
            prompt.push_str(&glm_tool_schema_json(tool));
            prompt.push('\n');
        }
        prompt.push_str(
            "</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>",
        );
        if let Some(instruction) = tool_choice_instruction {
            prompt.push_str("\n\n");
            prompt.push_str(instruction);
        }
    }
    let mut previous_was_tool = false;
    for message in messages {
        match message.role.as_str() {
            "system" => {
                prompt.push_str("<|system|>");
                prompt.push_str(&message_content_text(&message.content));
                previous_was_tool = false;
            }
            "user" => {
                prompt.push_str("<|user|>");
                prompt.push_str(&message_content_text(&message.content));
                previous_was_tool = false;
            }
            "assistant" => {
                prompt.push_str("<|assistant|><think>");
                if let Some(reasoning) = message.reasoning_content.as_deref() {
                    prompt.push_str(reasoning);
                }
                prompt.push_str("</think>");
                prompt.push_str(&assistant_visible_content(&message_content_text(
                    &message.content,
                )));
                if let Some(tool_calls) = &message.tool_calls {
                    for tool_call in tool_calls {
                        prompt.push_str(&render_glm_tool_call(tool_call));
                    }
                }
                previous_was_tool = false;
            }
            "tool" => {
                if !previous_was_tool {
                    prompt.push_str("<|observation|>");
                }
                prompt.push_str("<tool_response>");
                prompt.push_str(&message_content_text(&message.content));
                prompt.push_str("</tool_response>");
                previous_was_tool = true;
            }
            _ => {
                prompt.push_str(&message.role);
                prompt.push_str(&message_content_text(&message.content));
                previous_was_tool = false;
            }
        }
    }
    if reasoning_effort.is_some() {
        prompt.push_str("<|assistant|><think>");
    } else {
        prompt.push_str("<|assistant|><think></think>");
    }
    prompt
}

fn assistant_visible_content(content: &str) -> String {
    let visible = content
        .split_once("</think>")
        .map_or(content, |(_, suffix)| suffix);
    visible.to_owned()
}

pub(crate) fn request_thinking_enabled(request: &ChatCompletionRequest) -> bool {
    if let Some(thinking) = request.thinking.as_ref() {
        return matches!(
            thinking.thinking_type.trim().to_ascii_lowercase().as_str(),
            "enabled" | "enable" | "on" | "true"
        );
    }
    if let Some(enable_thinking) = request.enable_thinking {
        return enable_thinking;
    }
    if let Some(reasoning_effort) = request.reasoning_effort.as_ref() {
        return !matches!(
            reasoning_effort
                .as_str()
                .map(str::to_ascii_lowercase)
                .as_deref(),
            Some("none" | "off" | "disabled")
        );
    }
    env::var(REAL_FULL_ENABLE_THINKING_ENV)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn request_reasoning_effort_label(request: &ChatCompletionRequest) -> Option<&'static str> {
    request_thinking_enabled(request).then(|| {
        // The checkpoint chat template distinguishes only `high` from its
        // default `max`; other client levels intentionally map to Max.
        if request
            .reasoning_effort
            .as_ref()
            .and_then(Value::as_str)
            .is_some_and(|effort| effort.eq_ignore_ascii_case("high"))
        {
            "High"
        } else {
            "Max"
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{
        assistant_visible_content, real_glm_full_request_prompt_text, request_image_sources,
        ChatCompletionRequest,
    };

    #[test]
    fn assistant_visible_content_discards_reasoning_prefix() {
        assert_eq!(
            assistant_visible_content("<think>hidden</think> answer "),
            " answer "
        );
    }

    #[test]
    fn reasoning_effort_uses_the_checkpoint_high_or_max_contract() {
        let high: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high"
        }))
        .unwrap();
        assert!(real_glm_full_request_prompt_text(&high)
            .starts_with("[gMASK]<sop><|system|>Reasoning Effort: High"));

        let medium: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "thinking": {"type": "enabled"},
            "reasoning_effort": "medium"
        }))
        .unwrap();
        assert!(real_glm_full_request_prompt_text(&medium)
            .starts_with("[gMASK]<sop><|system|>Reasoning Effort: Max"));
    }

    #[test]
    fn multipart_image_content_preserves_order_and_extracts_urls() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                    {"type": "text", "text": "after"}
                ]
            }]
        }))
        .unwrap();
        let prompt = real_glm_full_request_prompt_text(&request);
        assert!(prompt.contains("<|user|>before<|begin_of_image|><|image|><|end_of_image|>after"));
        assert_eq!(
            request_image_sources(&request).unwrap(),
            vec!["data:image/png;base64,AA=="]
        );
    }
}

fn message_content_text(content: &Option<Value>) -> String {
    match content {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    return Some(text.to_owned());
                }
                image_part_source(part)
                    .is_some()
                    .then(|| "<|begin_of_image|><|image|><|end_of_image|>".to_owned())
            })
            .collect::<String>(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

pub(crate) fn request_image_sources(
    request: &ChatCompletionRequest,
) -> Result<Vec<String>, ApiError> {
    let mut sources = Vec::new();
    for (message_index, message) in request.messages.iter().enumerate() {
        let Some(Value::Array(parts)) = message.content.as_ref() else {
            continue;
        };
        for (part_index, part) in parts.iter().enumerate() {
            let is_image = part
                .get("type")
                .and_then(Value::as_str)
                .is_some_and(|kind| matches!(kind, "image" | "image_url"));
            if !is_image {
                continue;
            }
            let source = image_part_source(part).ok_or_else(|| {
                invalid_request(
                    "image content requires image_url.url (or a string image_url)",
                    Some(format!(
                        "messages[{message_index}].content[{part_index}].image_url"
                    )),
                )
            })?;
            if source.trim().is_empty() {
                return Err(invalid_request(
                    "image URL must not be empty",
                    Some(format!(
                        "messages[{message_index}].content[{part_index}].image_url"
                    )),
                ));
            }
            sources.push(source.to_owned());
        }
    }
    Ok(sources)
}

fn image_part_source(part: &Value) -> Option<&str> {
    let kind = part.get("type")?.as_str()?;
    if !matches!(kind, "image" | "image_url") {
        return None;
    }
    for key in ["image_url", "image"] {
        let Some(value) = part.get(key) else {
            continue;
        };
        if let Some(source) = value.as_str() {
            return Some(source);
        }
        if let Some(source) = value.get("url").and_then(Value::as_str) {
            return Some(source);
        }
    }
    None
}

pub(crate) fn stop_strings(stop: Option<&StopSpec>) -> Vec<String> {
    match stop {
        Some(StopSpec::One(value)) => vec![value.clone()],
        Some(StopSpec::Many(values)) => values.clone(),
        None => Vec::new(),
    }
}

pub(crate) fn rough_token_count(text: &str) -> usize {
    text.split_whitespace().count()
}

pub(crate) fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}
