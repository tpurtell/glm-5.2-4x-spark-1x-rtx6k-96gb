use serde::Serialize;
use serde_json::{Map, Value};
use uuid::Uuid;

use crate::{ChatTool, ToolCall, ToolCallFunction};

pub(crate) const GLM_TOOL_CALL_START: &str = "<tool_call>";
pub(crate) const GLM_TOOL_CALL_END: &str = "</tool_call>";
const GLM_ARG_KEY_START: &str = "<arg_key>";
const GLM_ARG_KEY_END: &str = "</arg_key>";
const GLM_ARG_VALUE_START: &str = "<arg_value>";
const GLM_ARG_VALUE_END: &str = "</arg_value>";

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ParsedToolOutput {
    pub(crate) content: Option<String>,
    pub(crate) tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GlmToolStreamDelta {
    Content(String),
    ToolCall {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GlmToolStreamState {
    Outside,
    Name,
    ArgumentKey,
    ArgumentValueStart,
    ArgumentValue,
    ArgumentOrEnd,
    DiscardCall,
}

#[derive(Debug)]
struct ActiveStreamToolCall {
    index: usize,
    id: String,
    name: Option<String>,
    argument_key: Option<String>,
    argument_count: usize,
    streaming_string_value: bool,
    streamed_arguments: String,
    emitted: bool,
}

#[derive(Debug)]
pub(crate) struct GlmToolCallStreamParser {
    tools: Vec<ChatTool>,
    state: GlmToolStreamState,
    pending: String,
    active: Option<ActiveStreamToolCall>,
    next_tool_index: usize,
    completed_tool_calls: usize,
    completed_tool_call_ids: Vec<String>,
    completed_tool_call_values: Vec<ToolCall>,
    saw_tool_syntax: bool,
}

impl GlmToolCallStreamParser {
    pub(crate) fn new(tools: Vec<ChatTool>) -> Self {
        Self {
            tools,
            state: GlmToolStreamState::Outside,
            pending: String::new(),
            active: None,
            next_tool_index: 0,
            completed_tool_calls: 0,
            completed_tool_call_ids: Vec::new(),
            completed_tool_call_values: Vec::new(),
            saw_tool_syntax: false,
        }
    }

    pub(crate) fn push(&mut self, chunk: &str) -> Vec<GlmToolStreamDelta> {
        self.pending.push_str(chunk);
        self.advance(false)
    }

    pub(crate) fn finish(&mut self) -> Vec<GlmToolStreamDelta> {
        self.advance(true)
    }

    pub(crate) fn completed_tool_calls(&self) -> usize {
        self.completed_tool_calls
    }

    pub(crate) fn completed_tool_call_ids(&self) -> &[String] {
        &self.completed_tool_call_ids
    }

    pub(crate) fn completed_tool_call_values(&self) -> &[ToolCall] {
        &self.completed_tool_call_values
    }

    fn advance(&mut self, finishing: bool) -> Vec<GlmToolStreamDelta> {
        let mut deltas = Vec::new();
        loop {
            let progressed = match self.state {
                GlmToolStreamState::Outside => self.advance_outside(finishing, &mut deltas),
                GlmToolStreamState::Name => self.advance_name(&mut deltas),
                GlmToolStreamState::ArgumentKey => self.advance_argument_key(),
                GlmToolStreamState::ArgumentValueStart => {
                    self.advance_argument_value_start(&mut deltas)
                }
                GlmToolStreamState::ArgumentValue => self.advance_argument_value(&mut deltas),
                GlmToolStreamState::ArgumentOrEnd => self.advance_argument_or_end(&mut deltas),
                GlmToolStreamState::DiscardCall => self.advance_discard_call(),
            };
            if !progressed {
                break;
            }
        }
        deltas
    }

    fn advance_outside(&mut self, finishing: bool, deltas: &mut Vec<GlmToolStreamDelta>) -> bool {
        if let Some(start) = self.pending.find(GLM_TOOL_CALL_START) {
            let prefix = self.pending[..start].to_owned();
            self.pending.drain(..start + GLM_TOOL_CALL_START.len());
            if !self.saw_tool_syntax && !prefix.is_empty() {
                deltas.push(GlmToolStreamDelta::Content(prefix));
            }
            self.saw_tool_syntax = true;
            self.active = Some(ActiveStreamToolCall {
                index: self.next_tool_index,
                id: format!("call_{}", Uuid::new_v4().simple()),
                name: None,
                argument_key: None,
                argument_count: 0,
                streaming_string_value: false,
                streamed_arguments: String::new(),
                emitted: false,
            });
            self.next_tool_index += 1;
            self.state = GlmToolStreamState::Name;
            return true;
        }

        if self.pending.is_empty() {
            return false;
        }
        let retained = if finishing {
            0
        } else {
            longest_suffix_matching_prefix(&self.pending, GLM_TOOL_CALL_START)
        };
        let emit_len = self.pending.len() - retained;
        if emit_len == 0 {
            return false;
        }
        let prefix = self.pending[..emit_len].to_owned();
        self.pending.drain(..emit_len);
        if !self.saw_tool_syntax {
            deltas.push(GlmToolStreamDelta::Content(prefix));
        }
        true
    }

    fn advance_name(&mut self, deltas: &mut Vec<GlmToolStreamDelta>) -> bool {
        let next_argument = self.pending.find(GLM_ARG_KEY_START);
        let call_end = self.pending.find(GLM_TOOL_CALL_END);
        let Some((offset, has_arguments)) = earliest_marker(next_argument, call_end) else {
            return false;
        };
        let name = self.pending[..offset].trim().to_owned();
        if name.is_empty() || name.contains('<') {
            self.state = GlmToolStreamState::DiscardCall;
            return true;
        }

        let marker = if has_arguments {
            GLM_ARG_KEY_START
        } else {
            GLM_TOOL_CALL_END
        };
        self.pending.drain(..offset + marker.len());
        let active = self
            .active
            .as_mut()
            .expect("tool stream name requires an active call");
        active.name = Some(name.clone());
        active.emitted = true;
        let arguments = if has_arguments { "{" } else { "{}" };
        active.streamed_arguments.push_str(arguments);
        deltas.push(GlmToolStreamDelta::ToolCall {
            index: active.index,
            id: Some(active.id.clone()),
            name: Some(name),
            arguments: Some(arguments.to_owned()),
        });
        if has_arguments {
            self.state = GlmToolStreamState::ArgumentKey;
        } else {
            self.complete_active_call();
        }
        true
    }

    fn advance_argument_key(&mut self) -> bool {
        let Some(end) = self.pending.find(GLM_ARG_KEY_END) else {
            return false;
        };
        let key = self.pending[..end].trim().to_owned();
        self.pending.drain(..end + GLM_ARG_KEY_END.len());
        if key.is_empty() || key.contains('<') {
            self.state = GlmToolStreamState::DiscardCall;
            return true;
        }
        self.active
            .as_mut()
            .expect("tool stream argument key requires an active call")
            .argument_key = Some(key);
        self.state = GlmToolStreamState::ArgumentValueStart;
        true
    }

    fn advance_argument_value_start(&mut self, deltas: &mut Vec<GlmToolStreamDelta>) -> bool {
        trim_pending_start(&mut self.pending);
        if self.pending.starts_with(GLM_ARG_VALUE_START) {
            self.pending.drain(..GLM_ARG_VALUE_START.len());
            let active = self
                .active
                .as_mut()
                .expect("tool stream argument value requires an active call");
            let name = active
                .name
                .as_deref()
                .expect("emitted tool stream call requires a name");
            let key = active
                .argument_key
                .as_deref()
                .expect("tool stream argument value requires a key");
            if tool_argument_accepts_string(name, key, &self.tools) {
                let separator = if active.argument_count == 0 { "" } else { "," };
                active.argument_count += 1;
                active.streaming_string_value = true;
                let key = serde_json::to_string(key)
                    .expect("serializing a streamed tool argument key should not fail");
                let arguments = format!("{separator}{key}:\"");
                active.streamed_arguments.push_str(&arguments);
                deltas.push(GlmToolStreamDelta::ToolCall {
                    index: active.index,
                    id: None,
                    name: None,
                    arguments: Some(arguments),
                });
            }
            self.state = GlmToolStreamState::ArgumentValue;
            return true;
        }
        if GLM_ARG_VALUE_START.starts_with(&self.pending) {
            return false;
        }
        self.state = GlmToolStreamState::DiscardCall;
        true
    }

    fn advance_argument_value(&mut self, deltas: &mut Vec<GlmToolStreamDelta>) -> bool {
        let streaming_string_value = self
            .active
            .as_ref()
            .expect("tool stream argument value requires an active call")
            .streaming_string_value;
        let end = self.pending.find(GLM_ARG_VALUE_END);
        if streaming_string_value && end.is_none() {
            let retained = longest_suffix_matching_prefix(&self.pending, GLM_ARG_VALUE_END);
            let emit_len = self.pending.len() - retained;
            if emit_len == 0 {
                return false;
            }
            let raw_fragment = self.pending[..emit_len].to_owned();
            self.pending.drain(..emit_len);
            let active = self
                .active
                .as_mut()
                .expect("tool stream argument value requires an active call");
            let arguments = json_string_contents(&raw_fragment);
            active.streamed_arguments.push_str(&arguments);
            deltas.push(GlmToolStreamDelta::ToolCall {
                index: active.index,
                id: None,
                name: None,
                arguments: Some(arguments),
            });
            return true;
        }
        let Some(end) = end else {
            return false;
        };
        let raw_value = self.pending[..end].to_owned();
        self.pending.drain(..end + GLM_ARG_VALUE_END.len());
        let active = self
            .active
            .as_mut()
            .expect("tool stream argument value requires an active call");
        let name = active
            .name
            .as_deref()
            .expect("emitted tool stream call requires a name");
        let key = active
            .argument_key
            .take()
            .expect("tool stream argument value requires a key");
        let arguments = if active.streaming_string_value {
            active.streaming_string_value = false;
            format!("{}\"", json_string_contents(&raw_value))
        } else {
            let value = parse_argument_value(name, &key, &raw_value, &self.tools);
            let separator = if active.argument_count == 0 { "" } else { "," };
            active.argument_count += 1;
            let key = serde_json::to_string(&key)
                .expect("serializing a streamed tool argument key should not fail");
            let value = serde_json::to_string(&value)
                .expect("serializing a streamed tool argument value should not fail");
            format!("{separator}{key}:{value}")
        };
        active.streamed_arguments.push_str(&arguments);
        deltas.push(GlmToolStreamDelta::ToolCall {
            index: active.index,
            id: None,
            name: None,
            arguments: Some(arguments),
        });
        self.state = GlmToolStreamState::ArgumentOrEnd;
        true
    }

    fn advance_argument_or_end(&mut self, deltas: &mut Vec<GlmToolStreamDelta>) -> bool {
        trim_pending_start(&mut self.pending);
        if self.pending.starts_with(GLM_ARG_KEY_START) {
            self.pending.drain(..GLM_ARG_KEY_START.len());
            self.state = GlmToolStreamState::ArgumentKey;
            return true;
        }
        if self.pending.starts_with(GLM_TOOL_CALL_END) {
            self.pending.drain(..GLM_TOOL_CALL_END.len());
            let active = self
                .active
                .as_mut()
                .expect("tool stream end requires an active call");
            active.streamed_arguments.push('}');
            deltas.push(GlmToolStreamDelta::ToolCall {
                index: active.index,
                id: None,
                name: None,
                arguments: Some("}".to_owned()),
            });
            self.complete_active_call();
            return true;
        }
        if GLM_ARG_KEY_START.starts_with(&self.pending)
            || GLM_TOOL_CALL_END.starts_with(&self.pending)
        {
            return false;
        }
        self.state = GlmToolStreamState::DiscardCall;
        true
    }

    fn advance_discard_call(&mut self) -> bool {
        let Some(end) = self.pending.find(GLM_TOOL_CALL_END) else {
            return false;
        };
        self.pending.drain(..end + GLM_TOOL_CALL_END.len());
        self.active = None;
        self.state = GlmToolStreamState::Outside;
        true
    }

    fn complete_active_call(&mut self) {
        let active = self
            .active
            .take()
            .expect("completing a tool stream requires an active call");
        if active.emitted {
            self.completed_tool_calls += 1;
            self.completed_tool_call_ids.push(active.id.clone());
            self.completed_tool_call_values.push(ToolCall {
                id: active.id,
                tool_type: "function".to_owned(),
                function: ToolCallFunction {
                    name: active
                        .name
                        .expect("completed streamed tool call requires a name"),
                    arguments: active.streamed_arguments,
                },
            });
        }
        self.state = GlmToolStreamState::Outside;
    }
}

fn earliest_marker(first: Option<usize>, second: Option<usize>) -> Option<(usize, bool)> {
    match (first, second) {
        (Some(first), Some(second)) if first <= second => Some((first, true)),
        (Some(_), Some(second)) => Some((second, false)),
        (Some(first), None) => Some((first, true)),
        (None, Some(second)) => Some((second, false)),
        (None, None) => None,
    }
}

fn trim_pending_start(pending: &mut String) {
    let whitespace = pending.len() - pending.trim_start().len();
    pending.drain(..whitespace);
}

fn longest_suffix_matching_prefix(text: &str, marker: &str) -> usize {
    let max_len = text.len().min(marker.len().saturating_sub(1));
    (1..=max_len)
        .rev()
        .find(|length| {
            let start = text.len() - length;
            text.is_char_boundary(start) && marker.starts_with(&text[start..])
        })
        .unwrap_or(0)
}

fn json_string_contents(value: &str) -> String {
    let quoted = serde_json::to_string(value)
        .expect("serializing a streamed string fragment should not fail");
    quoted[1..quoted.len() - 1].to_owned()
}

#[derive(Serialize)]
struct GlmToolSchema<'a> {
    name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<&'a Value>,
}

pub(crate) fn glm_tool_schema_json(tool: &ChatTool) -> String {
    serde_json::to_string(&GlmToolSchema {
        name: &tool.function.name,
        description: tool.function.description.as_deref(),
        parameters: tool.function.parameters.as_ref(),
    })
    .expect("serializing a function tool schema should not fail")
}

pub(crate) fn render_glm_tool_call(tool_call: &ToolCall) -> String {
    let mut rendered = format!("{GLM_TOOL_CALL_START}{}", tool_call.function.name);
    if let Ok(Value::Object(arguments)) =
        serde_json::from_str::<Value>(&tool_call.function.arguments)
    {
        for (key, value) in arguments {
            rendered.push_str(GLM_ARG_KEY_START);
            rendered.push_str(&key);
            rendered.push_str(GLM_ARG_KEY_END);
            rendered.push_str(GLM_ARG_VALUE_START);
            match value {
                Value::String(value) => rendered.push_str(&value),
                value => rendered.push_str(
                    &serde_json::to_string(&value)
                        .expect("serializing a prior tool-call argument should not fail"),
                ),
            }
            rendered.push_str(GLM_ARG_VALUE_END);
        }
    }
    rendered.push_str(GLM_TOOL_CALL_END);
    rendered
}

pub(crate) fn parse_glm_tool_calls(output: &str, tools: &[ChatTool]) -> ParsedToolOutput {
    let mut tool_calls = Vec::new();
    let mut first_valid_call = None;
    let mut search_offset = 0;

    while let Some(relative_start) = output[search_offset..].find(GLM_TOOL_CALL_START) {
        let call_start = search_offset + relative_start;
        let body_start = call_start + GLM_TOOL_CALL_START.len();
        let Some(relative_end) = output[body_start..].find(GLM_TOOL_CALL_END) else {
            break;
        };
        let body_end = body_start + relative_end;
        if let Some((name, arguments)) = parse_call_body(&output[body_start..body_end], tools) {
            first_valid_call.get_or_insert(call_start);
            tool_calls.push(ToolCall {
                id: format!("call_{}", Uuid::new_v4().simple()),
                tool_type: "function".to_owned(),
                function: ToolCallFunction { name, arguments },
            });
        }
        search_offset = body_end + GLM_TOOL_CALL_END.len();
    }

    let content = match first_valid_call {
        Some(call_start) => nonempty_content(&output[..call_start]),
        None => nonempty_content(output),
    };
    ParsedToolOutput {
        content,
        tool_calls,
    }
}

fn parse_call_body(body: &str, tools: &[ChatTool]) -> Option<(String, String)> {
    let first_argument = body.find(GLM_ARG_KEY_START);
    let name = body[..first_argument.unwrap_or(body.len())].trim();
    if name.is_empty() || name.contains('<') {
        return None;
    }

    let mut arguments = Map::new();
    let mut remainder = match first_argument {
        Some(offset) => &body[offset..],
        None if body.trim() == name => "",
        None => return None,
    };
    while !remainder.trim().is_empty() {
        remainder = remainder.trim_start();
        remainder = remainder.strip_prefix(GLM_ARG_KEY_START)?;
        let key_end = remainder.find(GLM_ARG_KEY_END)?;
        let key = remainder[..key_end].trim();
        if key.is_empty() {
            return None;
        }
        remainder = &remainder[key_end + GLM_ARG_KEY_END.len()..];
        remainder = remainder.trim_start();
        remainder = remainder.strip_prefix(GLM_ARG_VALUE_START)?;
        let value_end = remainder.find(GLM_ARG_VALUE_END)?;
        let raw_value = &remainder[..value_end];
        arguments.insert(
            key.to_owned(),
            parse_argument_value(name, key, raw_value, tools),
        );
        remainder = &remainder[value_end + GLM_ARG_VALUE_END.len()..];
    }

    let arguments = serde_json::to_string(&arguments).ok()?;
    Some((name.to_owned(), arguments))
}

fn parse_argument_value(
    tool_name: &str,
    argument_name: &str,
    raw_value: &str,
    tools: &[ChatTool],
) -> Value {
    if tool_argument_accepts_string(tool_name, argument_name, tools) {
        return Value::String(raw_value.to_owned());
    }
    let value = raw_value.trim();
    serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_owned()))
}

fn tool_argument_accepts_string(tool_name: &str, argument_name: &str, tools: &[ChatTool]) -> bool {
    tools
        .iter()
        .find(|tool| tool.function.name == tool_name)
        .and_then(|tool| tool.function.parameters.as_ref())
        .and_then(|parameters| parameters.get("properties"))
        .and_then(|properties| properties.get(argument_name))
        .is_some_and(schema_accepts_string)
}

fn schema_accepts_string(schema: &Value) -> bool {
    let direct = match schema.get("type") {
        Some(Value::String(kind)) => kind == "string",
        Some(Value::Array(kinds)) => kinds.iter().any(|kind| kind.as_str() == Some("string")),
        _ => false,
    };
    direct
        || ["anyOf", "oneOf"]
            .into_iter()
            .filter_map(|key| schema.get(key).and_then(Value::as_array))
            .flatten()
            .any(schema_accepts_string)
        || schema
            .get("enum")
            .and_then(Value::as_array)
            .is_some_and(|values| !values.is_empty() && values.iter().all(Value::is_string))
}

fn nonempty_content(content: &str) -> Option<String> {
    let content = content.trim();
    (!content.is_empty()).then(|| content.to_owned())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::json;

    use super::*;
    use crate::{ChatFunction, ChatTool};

    fn lookup_tool() -> ChatTool {
        ChatTool {
            tool_type: "function".to_owned(),
            function: ChatFunction {
                name: "lookup".to_owned(),
                description: Some("Look up a record.".to_owned()),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                        "exact": {"type": "boolean"}
                    }
                })),
            },
        }
    }

    #[test]
    fn parses_schema_typed_and_zero_argument_calls() {
        let parsed = parse_glm_tool_calls(
            "I'll check.<tool_call>lookup<arg_key>query</arg_key><arg_value>42</arg_value><arg_key>limit</arg_key><arg_value>3</arg_value><arg_key>exact</arg_key><arg_value>true</arg_value></tool_call><tool_call>refresh</tool_call>",
            &[lookup_tool()],
        );

        assert_eq!(parsed.content.as_deref(), Some("I'll check."));
        assert_eq!(parsed.tool_calls.len(), 2);
        assert_eq!(parsed.tool_calls[0].function.name, "lookup");
        assert_eq!(
            serde_json::from_str::<Value>(&parsed.tool_calls[0].function.arguments).unwrap(),
            json!({"query": "42", "limit": 3, "exact": true})
        );
        assert_eq!(parsed.tool_calls[1].function.arguments, "{}");
    }

    #[test]
    fn incomplete_or_malformed_call_remains_content() {
        let output = "before <tool_call>lookup<arg_key>query</arg_key>";
        let parsed = parse_glm_tool_calls(output, &[lookup_tool()]);

        assert!(parsed.tool_calls.is_empty());
        assert_eq!(parsed.content.as_deref(), Some(output));
    }

    #[test]
    fn incremental_parser_is_independent_of_chunk_boundaries() {
        let output = "I'll check.<tool_call>lookup<arg_key>query</arg_key><arg_value>Taipei</arg_value><arg_key>limit</arg_key><arg_value>3</arg_value></tool_call><tool_call>refresh</tool_call>ignored";
        let expected = (
            "I'll check.".to_owned(),
            vec![
                ("lookup".to_owned(), json!({"query": "Taipei", "limit": 3})),
                ("refresh".to_owned(), json!({})),
            ],
        );

        let mut boundaries = output
            .char_indices()
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        boundaries.push(output.len());
        for split in boundaries.iter().copied() {
            let mut parser = GlmToolCallStreamParser::new(vec![lookup_tool()]);
            let mut deltas = parser.push(&output[..split]);
            deltas.extend(parser.push(&output[split..]));
            deltas.extend(parser.finish());
            assert_eq!(stream_projection(&deltas), expected, "split={split}");
            assert_eq!(parser.completed_tool_calls(), 2, "split={split}");
        }

        let mut parser = GlmToolCallStreamParser::new(vec![lookup_tool()]);
        let mut deltas = Vec::new();
        for pair in boundaries.windows(2) {
            deltas.extend(parser.push(&output[pair[0]..pair[1]]));
        }
        deltas.extend(parser.finish());
        assert_eq!(stream_projection(&deltas), expected);
        assert_eq!(parser.completed_tool_calls(), 2);
    }

    #[test]
    fn incremental_parser_withholds_incomplete_control_syntax() {
        let mut parser = GlmToolCallStreamParser::new(vec![lookup_tool()]);
        let mut deltas = parser.push("visible <tool_");
        deltas.extend(parser.push("call>lookup<arg_key>query</arg_key><arg_value>partial"));
        deltas.extend(parser.finish());

        let content = deltas
            .iter()
            .filter_map(|delta| match delta {
                GlmToolStreamDelta::Content(content) => Some(content.as_str()),
                GlmToolStreamDelta::ToolCall { .. } => None,
            })
            .collect::<String>();
        assert_eq!(content, "visible ");
        assert!(deltas.iter().any(|delta| matches!(
            delta,
            GlmToolStreamDelta::ToolCall {
                name: Some(name),
                arguments: Some(arguments),
                ..
            } if name == "lookup" && arguments == "{"
        )));
        assert_eq!(parser.completed_tool_calls(), 0);
    }

    #[test]
    fn incremental_parser_streams_string_arguments_before_the_value_closes() {
        let mut parser = GlmToolCallStreamParser::new(vec![lookup_tool()]);
        let mut deltas = parser
            .push("<tool_call>lookup<arg_key>query</arg_key><arg_value>first \"quoted\" line");
        let early_arguments = deltas
            .iter()
            .filter_map(|delta| match delta {
                GlmToolStreamDelta::ToolCall {
                    arguments: Some(arguments),
                    ..
                } => Some(arguments.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(early_arguments, r#"{"query":"first \"quoted\" line"#);
        assert_eq!(parser.completed_tool_calls(), 0);

        deltas.extend(parser.push("\\tail\nnext</arg_value></tool_call>"));
        deltas.extend(parser.finish());
        assert_eq!(
            stream_projection(&deltas),
            (
                String::new(),
                vec![(
                    "lookup".to_owned(),
                    json!({"query": "first \"quoted\" line\\tail\nnext"})
                )],
            )
        );
        assert_eq!(parser.completed_tool_calls(), 1);
        assert_eq!(parser.completed_tool_call_ids().len(), 1);
        assert_eq!(
            parser.completed_tool_call_values()[0].function.arguments,
            r#"{"query":"first \"quoted\" line\\tail\nnext"}"#
        );
    }

    fn stream_projection(deltas: &[GlmToolStreamDelta]) -> (String, Vec<(String, Value)>) {
        let mut content = String::new();
        let mut calls = BTreeMap::<usize, (String, String)>::new();
        for delta in deltas {
            match delta {
                GlmToolStreamDelta::Content(chunk) => content.push_str(chunk),
                GlmToolStreamDelta::ToolCall {
                    index,
                    name,
                    arguments,
                    ..
                } => {
                    let call = calls.entry(*index).or_default();
                    if let Some(name) = name {
                        call.0 = name.clone();
                    }
                    if let Some(arguments) = arguments {
                        call.1.push_str(arguments);
                    }
                }
            }
        }
        let calls = calls
            .into_values()
            .map(|(name, arguments)| {
                let arguments = serde_json::from_str(&arguments)
                    .expect("streamed tool arguments should form JSON");
                (name, arguments)
            })
            .collect();
        (content, calls)
    }
}
