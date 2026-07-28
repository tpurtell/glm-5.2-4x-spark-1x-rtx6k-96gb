use async_stream::stream;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use std::convert::Infallible;

use crate::{CompletionMetrics, CompletionOutput, ToolCall, Usage};

#[derive(Debug, Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatStreamChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metrics: Option<CompletionMetrics>,
}

#[derive(Debug, Serialize)]
struct ChatStreamChoice {
    index: usize,
    delta: ChatStreamDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct ChatStreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ChatStreamToolCall>>,
}

#[derive(Debug, Serialize)]
struct ChatStreamToolCall {
    index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    tool_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function: Option<ChatStreamToolCallFunction>,
}

#[derive(Debug, Serialize)]
struct ChatStreamToolCallFunction {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<String>,
}

pub(crate) fn chat_stream_response(output: CompletionOutput, include_usage: bool) -> Response {
    let id = output.id;
    let created = output.created;
    let model = output.model;
    let content = output.content.unwrap_or_default();
    let content_chunks = output
        .stream_chunks
        .unwrap_or_else(|| whitespace_stream_chunks(&content));
    let reasoning_content = output.reasoning_content;
    let tool_calls = output.tool_calls.unwrap_or_default();
    let finish_reason = output.finish_reason;
    let usage = output.usage;
    let metrics = output.metrics;
    let stream = stream! {
        let first = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model.clone(),
            choices: vec![ChatStreamChoice {
                index: 0,
                delta: ChatStreamDelta {
                    role: Some("assistant"),
                    content: None,
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
            metrics: None,
        };
        yield Ok::<Event, Infallible>(Event::default().data(serde_json::to_string(&first).unwrap()));

        if let Some(reasoning) = reasoning_content {
            for token in whitespace_stream_chunks(&reasoning) {
                let chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model.clone(),
                    choices: vec![ChatStreamChoice {
                        index: 0,
                        delta: ChatStreamDelta {
                            role: None,
                            content: None,
                            reasoning_content: Some(token),
                            tool_calls: None,
                        },
                        finish_reason: None,
                    }],
                    usage: None,
                    metrics: None,
                };
                yield Ok::<Event, Infallible>(
                    Event::default().data(serde_json::to_string(&chunk).unwrap())
                );
            }
        }

        for token in content_chunks {
            let chunk = ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model.clone(),
                choices: vec![ChatStreamChoice {
                    index: 0,
                    delta: ChatStreamDelta {
                        role: None,
                        content: Some(token),
                        reasoning_content: None,
                        tool_calls: None,
                    },
                    finish_reason: None,
                }],
                usage: None,
                metrics: None,
            };
            yield Ok::<Event, Infallible>(Event::default().data(serde_json::to_string(&chunk).unwrap()));
        }

        for (tool_index, tool_call) in tool_calls.into_iter().enumerate() {
            let chunk = ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model.clone(),
                choices: vec![ChatStreamChoice {
                    index: 0,
                    delta: ChatStreamDelta {
                        role: None,
                        content: None,
                        reasoning_content: None,
                        tool_calls: Some(vec![stream_tool_call(tool_index, tool_call)]),
                    },
                    finish_reason: None,
                }],
                usage: None,
                metrics: None,
            };
            yield Ok::<Event, Infallible>(Event::default().data(serde_json::to_string(&chunk).unwrap()));
        }

        let done = ChatCompletionChunk {
            id,
            object: "chat.completion.chunk",
            created,
            model,
            choices: vec![ChatStreamChoice {
                index: 0,
                delta: ChatStreamDelta {
                    role: None,
                    content: None,
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: Some(finish_reason),
            }],
            usage: None,
            metrics: Some(metrics),
        };
        yield Ok::<Event, Infallible>(Event::default().data(serde_json::to_string(&done).unwrap()));
        if include_usage {
            let usage_chunk = ChatCompletionChunk {
                id: done.id,
                object: "chat.completion.chunk",
                created,
                model: done.model,
                choices: Vec::new(),
                usage: Some(usage),
                metrics: None,
            };
            yield Ok::<Event, Infallible>(
                Event::default().data(serde_json::to_string(&usage_chunk).unwrap())
            );
        }
        yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
    };
    Sse::new(stream).into_response()
}

pub(crate) fn chat_stream_role_event(id: &str, created: u64, model: &str) -> Event {
    let chunk = ChatCompletionChunk {
        id: id.to_owned(),
        object: "chat.completion.chunk",
        created,
        model: model.to_owned(),
        choices: vec![ChatStreamChoice {
            index: 0,
            delta: ChatStreamDelta {
                role: Some("assistant"),
                content: None,
                reasoning_content: None,
                tool_calls: None,
            },
            finish_reason: None,
        }],
        usage: None,
        metrics: None,
    };
    Event::default().data(serde_json::to_string(&chunk).unwrap())
}

pub(crate) fn chat_stream_content_event(
    id: &str,
    created: u64,
    model: &str,
    content: String,
) -> Event {
    let chunk = ChatCompletionChunk {
        id: id.to_owned(),
        object: "chat.completion.chunk",
        created,
        model: model.to_owned(),
        choices: vec![ChatStreamChoice {
            index: 0,
            delta: ChatStreamDelta {
                role: None,
                content: Some(content),
                reasoning_content: None,
                tool_calls: None,
            },
            finish_reason: None,
        }],
        usage: None,
        metrics: None,
    };
    Event::default().data(serde_json::to_string(&chunk).unwrap())
}

pub(crate) fn chat_stream_reasoning_event(
    id: &str,
    created: u64,
    model: &str,
    reasoning_content: String,
) -> Event {
    let chunk = ChatCompletionChunk {
        id: id.to_owned(),
        object: "chat.completion.chunk",
        created,
        model: model.to_owned(),
        choices: vec![ChatStreamChoice {
            index: 0,
            delta: ChatStreamDelta {
                role: None,
                content: None,
                reasoning_content: Some(reasoning_content),
                tool_calls: None,
            },
            finish_reason: None,
        }],
        usage: None,
        metrics: None,
    };
    Event::default().data(serde_json::to_string(&chunk).unwrap())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn chat_stream_tool_call_event(
    id: &str,
    created: u64,
    model: &str,
    index: usize,
    call_id: Option<String>,
    name: Option<String>,
    arguments: Option<String>,
) -> Event {
    let tool_type = call_id.as_ref().map(|_| "function".to_owned());
    let chunk = ChatCompletionChunk {
        id: id.to_owned(),
        object: "chat.completion.chunk",
        created,
        model: model.to_owned(),
        choices: vec![ChatStreamChoice {
            index: 0,
            delta: ChatStreamDelta {
                role: None,
                content: None,
                reasoning_content: None,
                tool_calls: Some(vec![ChatStreamToolCall {
                    index,
                    id: call_id,
                    tool_type,
                    function: Some(ChatStreamToolCallFunction { name, arguments }),
                }]),
            },
            finish_reason: None,
        }],
        usage: None,
        metrics: None,
    };
    Event::default().data(serde_json::to_string(&chunk).unwrap())
}

pub(crate) fn chat_stream_finish_event(
    id: String,
    created: u64,
    model: String,
    finish_reason: String,
    metrics: CompletionMetrics,
) -> Event {
    let chunk = ChatCompletionChunk {
        id,
        object: "chat.completion.chunk",
        created,
        model,
        choices: vec![ChatStreamChoice {
            index: 0,
            delta: ChatStreamDelta {
                role: None,
                content: None,
                reasoning_content: None,
                tool_calls: None,
            },
            finish_reason: Some(finish_reason),
        }],
        usage: None,
        metrics: Some(metrics),
    };
    Event::default().data(serde_json::to_string(&chunk).unwrap())
}

pub(crate) fn chat_stream_usage_event(
    id: String,
    created: u64,
    model: String,
    metrics: &CompletionMetrics,
) -> Event {
    let usage = Usage::from_metrics(metrics);
    let chunk = ChatCompletionChunk {
        id,
        object: "chat.completion.chunk",
        created,
        model,
        choices: Vec::new(),
        usage: Some(usage),
        metrics: None,
    };
    Event::default().data(serde_json::to_string(&chunk).unwrap())
}

pub(crate) fn chat_stream_done_event() -> Event {
    Event::default().data("[DONE]")
}

fn whitespace_stream_chunks(content: &str) -> Vec<String> {
    content
        .split_whitespace()
        .enumerate()
        .map(|(idx, token)| {
            if idx == 0 {
                token.to_owned()
            } else {
                format!(" {token}")
            }
        })
        .collect()
}

fn stream_tool_call(index: usize, tool_call: ToolCall) -> ChatStreamToolCall {
    ChatStreamToolCall {
        index,
        id: Some(tool_call.id),
        tool_type: Some(tool_call.tool_type),
        function: Some(ChatStreamToolCallFunction {
            name: Some(tool_call.function.name),
            arguments: Some(tool_call.function.arguments),
        }),
    }
}
