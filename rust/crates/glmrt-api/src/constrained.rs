use std::sync::Arc;

use serde_json::{json, Value};

use crate::error::{invalid_request, ApiError};
use crate::request::{request_thinking_enabled, tool_calls_enabled};
use crate::{
    ChatCompletionRequest, ChatTool, RealFullConstraint, RealFullConstraintGrammar, ResponseFormat,
    ToolChoice,
};

const GLM_THINK_CLOSE_TOKEN_ID: usize = 154_842;

pub(crate) fn request_constraint(
    request: &ChatCompletionRequest,
) -> Result<Option<Arc<RealFullConstraint>>, ApiError> {
    if let Some(response_format) = request.response_format.as_ref() {
        if !matches!(response_format, ResponseFormat::Text) {
            let grammar = match response_format {
                ResponseFormat::Text => unreachable!("text response format handled above"),
                ResponseFormat::JsonObject => RealFullConstraintGrammar::Json,
                ResponseFormat::JsonSchema { json_schema } => {
                    RealFullConstraintGrammar::JsonSchema {
                        schema_json: serde_json::to_string(&json_schema.schema).map_err(
                            |error| {
                                invalid_request(
                                    format!(
                                        "response_format JSON Schema cannot be serialized: {error}"
                                    ),
                                    Some("response_format.json_schema.schema"),
                                )
                            },
                        )?,
                        strict: json_schema.strict.unwrap_or(false),
                    }
                }
            };
            return Ok(Some(Arc::new(RealFullConstraint {
                grammar: wrap_reasoning_prefix(request, grammar)?,
            })));
        }
    }

    let selected_tools = selected_tools(request);
    if selected_tools.is_empty()
        || !selected_tools
            .iter()
            .any(|tool| tool.function.strict.unwrap_or(false))
    {
        return Ok(None);
    }
    let structural_tag = strict_tool_structural_tag(request, &selected_tools)?;
    Ok(Some(Arc::new(RealFullConstraint {
        grammar: RealFullConstraintGrammar::StructuralTag {
            structural_tag_json: serde_json::to_string(&structural_tag).map_err(|error| {
                invalid_request(
                    format!("strict tool grammar cannot be serialized: {error}"),
                    Some("tools"),
                )
            })?,
        },
    })))
}

fn wrap_reasoning_prefix(
    request: &ChatCompletionRequest,
    grammar: RealFullConstraintGrammar,
) -> Result<RealFullConstraintGrammar, ApiError> {
    if !request_thinking_enabled(request) {
        return Ok(grammar);
    }
    let content = grammar_format_json(&grammar)?;
    let structural_tag = json!({
        "type": "structural_tag",
        "format": {
            "type": "sequence",
            "elements": [
                {
                    "type": "any_tokens",
                    "exclude_tokens": [GLM_THINK_CLOSE_TOKEN_ID]
                },
                {"type": "token", "token": GLM_THINK_CLOSE_TOKEN_ID},
                content
            ]
        }
    });
    Ok(RealFullConstraintGrammar::StructuralTag {
        structural_tag_json: serde_json::to_string(&structural_tag).map_err(|error| {
            invalid_request(
                format!("reasoning-prefixed response grammar cannot be serialized: {error}"),
                Some("response_format"),
            )
        })?,
    })
}

fn grammar_format_json(grammar: &RealFullConstraintGrammar) -> Result<Value, ApiError> {
    match grammar {
        RealFullConstraintGrammar::Json => Ok(json!({
            "type": "json_schema",
            "json_schema": {"type": "object"}
        })),
        RealFullConstraintGrammar::JsonSchema {
            schema_json,
            strict: _,
        } => Ok(json!({
            "type": "json_schema",
            "json_schema": serde_json::from_str::<Value>(schema_json).map_err(|error| {
                invalid_request(
                    format!("response_format JSON Schema is invalid: {error}"),
                    Some("response_format.json_schema.schema"),
                )
            })?
        })),
        RealFullConstraintGrammar::StructuralTag { .. } => Err(invalid_request(
            "nested structural response constraints are unsupported",
            Some("response_format"),
        )),
    }
}

fn strict_tool_structural_tag(
    request: &ChatCompletionRequest,
    tools: &[&ChatTool],
) -> Result<Value, ApiError> {
    let tags = tools
        .iter()
        .map(|tool| {
            let parameters = if tool.function.strict.unwrap_or(false) {
                tool.function
                    .parameters
                    .clone()
                    .unwrap_or_else(empty_parameters)
            } else {
                json!({"type": "object", "additionalProperties": true})
            };
            json!({
                "type": "tag",
                "begin": format!("<tool_call>{}", tool.function.name),
                "content": {
                    "type": "json_schema",
                    "json_schema": parameters,
                    "style": "glm_xml",
                    "any_order": false
                },
                "end": "</tool_call>"
            })
        })
        .collect::<Vec<_>>();
    let stop_after_first = !request.parallel_tool_calls.unwrap_or(true);
    let format = match &request.tool_choice {
        Some(ToolChoice::Mode(mode)) if mode == "required" => json!({
            "type": "tags_with_separator",
            "tags": tags,
            "separator": "",
            "at_least_one": true,
            "stop_after_first": stop_after_first
        }),
        Some(ToolChoice::Specific { .. }) => json!({
            "type": "tags_with_separator",
            "tags": tags,
            "separator": "",
            "at_least_one": true,
            "stop_after_first": stop_after_first
        }),
        _ => json!({
            "type": "triggered_tags",
            "triggers": ["<tool_call>"],
            "tags": tags,
            "at_least_one": false,
            "stop_after_first": stop_after_first
        }),
    };
    let format = if request_thinking_enabled(request) {
        json!({
            "type": "sequence",
            "elements": [
                {
                    "type": "any_tokens",
                    "exclude_tokens": [GLM_THINK_CLOSE_TOKEN_ID]
                },
                {"type": "token", "token": GLM_THINK_CLOSE_TOKEN_ID},
                format
            ]
        })
    } else {
        format
    };
    Ok(json!({"type": "structural_tag", "format": format}))
}

fn selected_tools(request: &ChatCompletionRequest) -> Vec<&ChatTool> {
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

pub(crate) fn empty_parameters() -> Value {
    json!({
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": false
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn request(value: Value) -> ChatCompletionRequest {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn unconstrained_requests_have_no_constraint_object() {
        let request = request(json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}]
        }));
        assert!(request_constraint(&request).unwrap().is_none());
    }

    #[test]
    fn json_schema_preserves_schema_and_strict_mode() {
        let request = request(json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": false
                    }
                }
            }
        }));
        let constraint = request_constraint(&request).unwrap().unwrap();
        let RealFullConstraintGrammar::JsonSchema {
            schema_json,
            strict,
        } = &constraint.grammar
        else {
            panic!("expected JSON Schema grammar")
        };
        assert!(*strict);
        let schema: Value = serde_json::from_str(schema_json).unwrap();
        assert_eq!(schema["required"], json!(["answer"]));
    }

    #[test]
    fn strict_tools_use_glm_xml_and_openai_choice_controls() {
        let request = request(json!({
            "model": "test",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "strict": true,
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": false
                    }
                }
            }],
            "tool_choice": "required",
            "parallel_tool_calls": false
        }));
        let constraint = request_constraint(&request).unwrap().unwrap();
        let RealFullConstraintGrammar::StructuralTag {
            structural_tag_json,
        } = &constraint.grammar
        else {
            panic!("expected structural tag grammar")
        };
        let grammar: Value = serde_json::from_str(structural_tag_json).unwrap();
        assert_eq!(grammar["format"]["type"], "tags_with_separator");
        assert_eq!(grammar["format"]["at_least_one"], true);
        assert_eq!(grammar["format"]["stop_after_first"], true);
        assert_eq!(grammar["format"]["tags"][0]["begin"], "<tool_call>lookup");
        assert_eq!(grammar["format"]["tags"][0]["content"]["style"], "glm_xml");
    }

    #[test]
    fn explicit_text_response_format_preserves_strict_tool_constraint() {
        let request = request(json!({
            "model": "test",
            "messages": [{"role": "user", "content": "ping"}],
            "response_format": {"type": "text"},
            "tools": [{
                "type": "function",
                "function": {"name": "ping", "strict": true}
            }],
            "tool_choice": "required"
        }));
        assert!(matches!(
            request_constraint(&request).unwrap().unwrap().grammar,
            RealFullConstraintGrammar::StructuralTag { .. }
        ));
    }

    #[test]
    fn thinking_prefix_is_inside_the_same_constraint() {
        let request = request(json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "thinking": {"type": "enabled"},
            "response_format": {"type": "json_object"}
        }));
        let constraint = request_constraint(&request).unwrap().unwrap();
        let RealFullConstraintGrammar::StructuralTag {
            structural_tag_json,
        } = &constraint.grammar
        else {
            panic!("expected reasoning-prefixed structural tag")
        };
        let grammar: Value = serde_json::from_str(structural_tag_json).unwrap();
        assert_eq!(grammar["format"]["elements"][1]["token"], 154_842);
        assert_eq!(grammar["format"]["elements"][2]["type"], "json_schema");
    }
}
