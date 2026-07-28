use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use crate::ChatCompletionRequest;

const MAX_TOOL_CONTINUATIONS: usize = 64;
const MAX_TOOL_CONTINUATION_TOKENS: usize = 8_000_000;

#[derive(Debug)]
struct ToolContinuation {
    call_ids: Vec<String>,
    prefix_text: String,
    token_ids: Arc<Vec<usize>>,
}

#[derive(Debug, Default)]
pub(crate) struct ToolContinuationCache {
    entries: VecDeque<Arc<ToolContinuation>>,
    by_call_id: HashMap<String, Arc<ToolContinuation>>,
    stored_tokens: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct ToolContinuationMatch {
    pub(crate) call_id: String,
    pub(crate) prefix_text_len: usize,
    pub(crate) token_ids: Arc<Vec<usize>>,
}

impl ToolContinuationCache {
    pub(crate) fn matching_prefix(
        &self,
        request: &ChatCompletionRequest,
        prompt: &str,
    ) -> Option<ToolContinuationMatch> {
        request
            .messages
            .iter()
            .rev()
            .filter_map(|message| message.tool_calls.as_deref())
            .flat_map(|tool_calls| tool_calls.iter().rev())
            .find_map(|tool_call| {
                let continuation = self.by_call_id.get(&tool_call.id)?;
                prompt
                    .starts_with(&continuation.prefix_text)
                    .then(|| ToolContinuationMatch {
                        call_id: tool_call.id.clone(),
                        prefix_text_len: continuation.prefix_text.len(),
                        token_ids: Arc::clone(&continuation.token_ids),
                    })
            })
    }

    pub(crate) fn known_call_without_text_match(
        &self,
        request: &ChatCompletionRequest,
        prompt: &str,
    ) -> Option<(String, usize)> {
        request
            .messages
            .iter()
            .rev()
            .filter_map(|message| message.tool_calls.as_deref())
            .flatten()
            .find_map(|tool_call| {
                let continuation = self.by_call_id.get(&tool_call.id)?;
                let matching_bytes = continuation
                    .prefix_text
                    .bytes()
                    .zip(prompt.bytes())
                    .take_while(|(expected, actual)| expected == actual)
                    .count();
                (!prompt.starts_with(&continuation.prefix_text))
                    .then(|| (tool_call.id.clone(), matching_bytes))
            })
    }

    pub(crate) fn insert(
        &mut self,
        call_ids: Vec<String>,
        prefix_text: String,
        token_ids: Arc<Vec<usize>>,
    ) {
        if call_ids.is_empty()
            || token_ids.is_empty()
            || token_ids.len() > MAX_TOOL_CONTINUATION_TOKENS
        {
            return;
        }
        let continuation = Arc::new(ToolContinuation {
            call_ids,
            prefix_text,
            token_ids,
        });
        self.stored_tokens = self
            .stored_tokens
            .saturating_add(continuation.token_ids.len());
        for call_id in &continuation.call_ids {
            self.by_call_id
                .insert(call_id.clone(), Arc::clone(&continuation));
        }
        self.entries.push_back(continuation);
        self.evict_to_limits();
    }

    fn evict_to_limits(&mut self) {
        while self.entries.len() > MAX_TOOL_CONTINUATIONS
            || self.stored_tokens > MAX_TOOL_CONTINUATION_TOKENS
        {
            let Some(continuation) = self.entries.pop_front() else {
                break;
            };
            self.stored_tokens = self
                .stored_tokens
                .saturating_sub(continuation.token_ids.len());
            for call_id in &continuation.call_ids {
                if self
                    .by_call_id
                    .get(call_id)
                    .is_some_and(|current| Arc::ptr_eq(current, &continuation))
                {
                    self.by_call_id.remove(call_id);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn continuation_request(arguments: &str) -> ChatCompletionRequest {
        serde_json::from_value(json!({
            "model": "test",
            "messages": [
                {"role": "user", "content": "write it"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "write",
                            "arguments": arguments
                        }
                    }]
                },
                {"role": "tool", "tool_call_id": "call_abc", "content": "ok"}
            ]
        }))
        .unwrap()
    }

    #[test]
    fn matching_requires_both_the_returned_call_and_canonical_turn_prefix() {
        let mut cache = ToolContinuationCache::default();
        cache.insert(
            vec!["call_abc".to_owned()],
            "canonical assistant turn".to_owned(),
            Arc::new(vec![1, 7, 9]),
        );

        let request = continuation_request(r#"{"content":"x","path":"a"}"#);
        let matched = cache
            .matching_prefix(&request, "canonical assistant turn<|observation|>ok")
            .unwrap();
        assert_eq!(matched.call_id, "call_abc");
        assert_eq!(matched.prefix_text_len, "canonical assistant turn".len());
        assert_eq!(&*matched.token_ids, &[1, 7, 9]);

        assert!(cache
            .matching_prefix(&request, "mutated assistant turn<|observation|>ok")
            .is_none());
        assert_eq!(
            cache.known_call_without_text_match(&request, "canonical assistant X<|observation|>ok"),
            Some(("call_abc".to_owned(), "canonical assistant ".len()))
        );
    }
}
