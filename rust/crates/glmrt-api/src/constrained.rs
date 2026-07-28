use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{ChatCompletionRequest, ToolChoice};

static NEXT_CONSTRAINT_LIFECYCLE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ToolConstraintMode {
    Disabled,
    Auto,
    Required,
    Named(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ToolConstraintPlan {
    pub(crate) mode: ToolConstraintMode,
    pub(crate) allowed_tool_names: Vec<String>,
}

impl ToolConstraintPlan {
    pub(crate) fn from_request(
        request: &ChatCompletionRequest,
    ) -> Result<Self, ConstraintLifecycleError> {
        let tool_names = request
            .tools
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(|tool| tool.function.name.clone())
            .collect::<Vec<_>>();
        match &request.tool_choice {
            Some(ToolChoice::Mode(mode)) if mode == "none" => Ok(Self {
                mode: ToolConstraintMode::Disabled,
                allowed_tool_names: Vec::new(),
            }),
            Some(ToolChoice::Mode(mode)) if mode == "auto" => Ok(Self {
                mode: if tool_names.is_empty() {
                    ToolConstraintMode::Disabled
                } else {
                    ToolConstraintMode::Auto
                },
                allowed_tool_names: tool_names,
            }),
            Some(ToolChoice::Mode(mode)) if mode == "required" => {
                ensure_constraint(
                    !tool_names.is_empty(),
                    "required tool constraint has no tools",
                )?;
                Ok(Self {
                    mode: ToolConstraintMode::Required,
                    allowed_tool_names: tool_names,
                })
            }
            Some(ToolChoice::Mode(mode)) => Err(ConstraintLifecycleError::new(format!(
                "unsupported tool constraint mode {mode}"
            ))),
            Some(ToolChoice::Specific { function, .. }) => {
                ensure_constraint(
                    tool_names.iter().any(|name| name == &function.name),
                    format!(
                        "named tool constraint {} is not present in tools",
                        function.name
                    ),
                )?;
                Ok(Self {
                    mode: ToolConstraintMode::Named(function.name.clone()),
                    allowed_tool_names: vec![function.name.clone()],
                })
            }
            None if tool_names.is_empty() => Ok(Self {
                mode: ToolConstraintMode::Disabled,
                allowed_tool_names: Vec::new(),
            }),
            None => Ok(Self {
                mode: ToolConstraintMode::Auto,
                allowed_tool_names: tool_names,
            }),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ConstraintMask {
    allowed_token_ids: Vec<u32>,
}

impl ConstraintMask {
    pub(crate) fn new(
        mut allowed_token_ids: Vec<u32>,
        vocab_size: usize,
    ) -> Result<Self, ConstraintLifecycleError> {
        ensure_constraint(vocab_size > 0, "constraint vocabulary must be non-zero")?;
        allowed_token_ids.sort_unstable();
        ensure_constraint(
            allowed_token_ids
                .iter()
                .all(|token| (*token as usize) < vocab_size),
            "constraint mask contains a token outside the vocabulary",
        )?;
        ensure_constraint(
            allowed_token_ids.windows(2).all(|pair| pair[0] != pair[1]),
            "constraint mask contains duplicate token IDs",
        )?;
        Ok(Self { allowed_token_ids })
    }

    pub(crate) fn allowed_token_ids(&self) -> &[u32] {
        &self.allowed_token_ids
    }

    pub(crate) fn allowed_count(&self) -> usize {
        self.allowed_token_ids.len()
    }

    pub(crate) fn allows(&self, token_id: u32) -> bool {
        self.allowed_token_ids.binary_search(&token_id).is_ok()
    }

    pub(crate) fn proven_identical(&self, other: &Self) -> bool {
        self.allowed_token_ids == other.allowed_token_ids
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ConstraintSamplingPath {
    SparseAllowedRows,
    DenseMaskedLogits,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ConstraintSamplingPolicy {
    vocab_size: usize,
    sparse_allowed_token_limit: usize,
}

impl ConstraintSamplingPolicy {
    pub(crate) fn new(
        vocab_size: usize,
        sparse_allowed_token_limit: usize,
    ) -> Result<Self, ConstraintLifecycleError> {
        ensure_constraint(vocab_size > 0, "constraint vocabulary must be non-zero")?;
        ensure_constraint(
            sparse_allowed_token_limit <= vocab_size,
            "constraint sparse threshold exceeds vocabulary size",
        )?;
        Ok(Self {
            vocab_size,
            sparse_allowed_token_limit,
        })
    }

    pub(crate) fn select(
        &self,
        mask: &ConstraintMask,
    ) -> Result<ConstraintSamplingPath, ConstraintLifecycleError> {
        ensure_constraint(
            mask.allowed_token_ids
                .iter()
                .all(|token| (*token as usize) < self.vocab_size),
            "constraint mask does not match sampling-policy vocabulary",
        )?;
        Ok(if mask.allowed_count() <= self.sparse_allowed_token_limit {
            ConstraintSamplingPath::SparseAllowedRows
        } else {
            ConstraintSamplingPath::DenseMaskedLogits
        })
    }
}

pub(crate) trait ConstraintMatcher: Clone {
    fn fill_allowed_token_ids(&self, output: &mut Vec<u32>) -> Result<(), String>;
    fn accept_token(&mut self, token_id: u32) -> Result<(), String>;
    fn is_complete(&self) -> bool;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ConstraintStep {
    pub(crate) position: usize,
    pub(crate) token_id: u32,
    pub(crate) mask: ConstraintMask,
    pub(crate) sampling_path: ConstraintSamplingPath,
    pub(crate) complete_after: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct ConstraintDraft<M> {
    lifecycle_id: u64,
    epoch: u64,
    pub(crate) tokens: Vec<u32>,
    pub(crate) steps: Vec<ConstraintStep>,
    states_after: Vec<M>,
}

pub(crate) struct ConstrainedDecodeLifecycle<M> {
    lifecycle_id: u64,
    epoch: u64,
    matcher: M,
    policy: ConstraintSamplingPolicy,
    committed_tokens: Vec<u32>,
}

impl<M: ConstraintMatcher> ConstrainedDecodeLifecycle<M> {
    pub(crate) fn new(matcher: M, policy: ConstraintSamplingPolicy) -> Self {
        Self {
            lifecycle_id: NEXT_CONSTRAINT_LIFECYCLE_ID.fetch_add(1, Ordering::Relaxed),
            epoch: 0,
            matcher,
            policy,
            committed_tokens: Vec::new(),
        }
    }

    pub(crate) fn prepare_draft(
        &self,
        tokens: &[u32],
    ) -> Result<ConstraintDraft<M>, ConstraintLifecycleError> {
        ensure_constraint(!tokens.is_empty(), "constraint draft must contain tokens")?;
        let mut matcher = self.matcher.clone();
        let mut states_after = Vec::with_capacity(tokens.len());
        let mut steps = Vec::with_capacity(tokens.len());
        let mut allowed = Vec::new();
        for (position, token_id) in tokens.iter().copied().enumerate() {
            allowed.clear();
            matcher
                .fill_allowed_token_ids(&mut allowed)
                .map_err(ConstraintLifecycleError::new)?;
            let mask = ConstraintMask::new(allowed.clone(), self.policy.vocab_size)?;
            ensure_constraint(
                mask.allows(token_id),
                format!("constraint rejected token {token_id} at speculative position {position}"),
            )?;
            let sampling_path = self.policy.select(&mask)?;
            matcher
                .accept_token(token_id)
                .map_err(ConstraintLifecycleError::new)?;
            let complete_after = matcher.is_complete();
            states_after.push(matcher.clone());
            steps.push(ConstraintStep {
                position,
                token_id,
                mask,
                sampling_path,
                complete_after,
            });
        }
        Ok(ConstraintDraft {
            lifecycle_id: self.lifecycle_id,
            epoch: self.epoch,
            tokens: tokens.to_vec(),
            steps,
            states_after,
        })
    }

    pub(crate) fn commit_prefix(
        &mut self,
        draft: &ConstraintDraft<M>,
        accepted_tokens: usize,
    ) -> Result<(), ConstraintLifecycleError> {
        ensure_constraint(
            draft.lifecycle_id == self.lifecycle_id,
            "constraint draft belongs to another request",
        )?;
        ensure_constraint(draft.epoch == self.epoch, "constraint draft is stale")?;
        ensure_constraint(
            accepted_tokens <= draft.tokens.len(),
            "constraint accepted prefix exceeds draft width",
        )?;
        if accepted_tokens > 0 {
            self.matcher = draft.states_after[accepted_tokens - 1].clone();
            self.committed_tokens
                .extend_from_slice(&draft.tokens[..accepted_tokens]);
        }
        self.epoch = self
            .epoch
            .checked_add(1)
            .ok_or_else(|| ConstraintLifecycleError::new("constraint epoch overflow"))?;
        Ok(())
    }

    pub(crate) fn committed_tokens(&self) -> &[u32] {
        &self.committed_tokens
    }

    pub(crate) fn is_complete(&self) -> bool {
        self.matcher.is_complete()
    }
}

pub(crate) fn proven_identical_mask_groups(masks: &[ConstraintMask]) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (mask_index, mask) in masks.iter().enumerate() {
        if let Some(group) = groups
            .iter_mut()
            .find(|group| mask.proven_identical(&masks[group[0]]))
        {
            group.push(mask_index);
        } else {
            groups.push(vec![mask_index]);
        }
    }
    groups
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ConstraintLifecycleError {
    message: String,
}

impl ConstraintLifecycleError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ConstraintLifecycleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ConstraintLifecycleError {}

fn ensure_constraint(
    condition: bool,
    message: impl Into<String>,
) -> Result<(), ConstraintLifecycleError> {
    if condition {
        Ok(())
    } else {
        Err(ConstraintLifecycleError::new(message))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Arc;

    use serde_json::{json, Value};

    use super::*;
    use crate::tooling::{parse_glm_tool_calls, GlmToolCallStreamParser, GlmToolStreamDelta};
    use crate::{ChatFunction, ChatTool};

    const VOCAB_SIZE: usize = 154_880;

    #[derive(Clone, Debug)]
    struct ScriptedMatcher {
        sequences: Arc<Vec<Vec<u32>>>,
        prefix: Vec<u32>,
    }

    impl ScriptedMatcher {
        fn new(sequences: Vec<Vec<u32>>) -> Self {
            Self {
                sequences: Arc::new(sequences),
                prefix: Vec::new(),
            }
        }
    }

    impl ConstraintMatcher for ScriptedMatcher {
        fn fill_allowed_token_ids(&self, output: &mut Vec<u32>) -> Result<(), String> {
            let next = self
                .sequences
                .iter()
                .filter(|sequence| sequence.starts_with(&self.prefix))
                .filter_map(|sequence| sequence.get(self.prefix.len()).copied())
                .collect::<BTreeSet<_>>();
            output.extend(next);
            Ok(())
        }

        fn accept_token(&mut self, token_id: u32) -> Result<(), String> {
            let mut allowed = Vec::new();
            self.fill_allowed_token_ids(&mut allowed)?;
            if !allowed.contains(&token_id) {
                return Err(format!("script rejects token {token_id}"));
            }
            self.prefix.push(token_id);
            Ok(())
        }

        fn is_complete(&self) -> bool {
            self.sequences
                .iter()
                .any(|sequence| sequence == &self.prefix)
        }
    }

    #[test]
    fn request_tool_scope_covers_disabled_auto_required_and_named() {
        let disabled = request_with_choice(Some(json!("none")), true);
        let auto = request_with_choice(Some(json!("auto")), true);
        let required = request_with_choice(Some(json!("required")), true);
        let named = request_with_choice(
            Some(json!({"type": "function", "function": {"name": "lookup"}})),
            true,
        );

        assert_eq!(
            first_mask_for_request(&disabled),
            ConstraintMask::new(vec![1], VOCAB_SIZE).unwrap()
        );
        assert_eq!(
            first_mask_for_request(&auto),
            ConstraintMask::new(vec![1, 10, 11], VOCAB_SIZE).unwrap()
        );
        assert_eq!(
            first_mask_for_request(&required),
            ConstraintMask::new(vec![10, 11], VOCAB_SIZE).unwrap()
        );
        assert_eq!(
            first_mask_for_request(&named),
            ConstraintMask::new(vec![10], VOCAB_SIZE).unwrap()
        );

        let missing_tools = request_with_choice(Some(json!("required")), false);
        assert!(ToolConstraintPlan::from_request(&missing_tools)
            .unwrap_err()
            .to_string()
            .contains("no tools"));
    }

    #[test]
    fn mtp_drafts_commit_prefixes_and_requests_keep_independent_state() {
        let policy = ConstraintSamplingPolicy::new(VOCAB_SIZE, 24_576).unwrap();
        let matcher = ScriptedMatcher::new(vec![vec![1, 2, 3, 9], vec![1, 2, 4, 9]]);
        let mut left = ConstrainedDecodeLifecycle::new(matcher.clone(), policy);
        let right = ConstrainedDecodeLifecycle::new(matcher, policy);

        let left_draft = left.prepare_draft(&[1, 2, 3]).unwrap();
        assert_eq!(left_draft.steps[2].mask.allowed_token_ids(), &[3, 4]);
        assert!(left
            .commit_prefix(&right.prepare_draft(&[1]).unwrap(), 1)
            .unwrap_err()
            .to_string()
            .contains("another request"));
        left.commit_prefix(&left_draft, 2).unwrap();
        assert_eq!(left.committed_tokens(), &[1, 2]);
        assert!(right.committed_tokens().is_empty());
        assert!(!left.is_complete());

        let tail = left.prepare_draft(&[4, 9]).unwrap();
        left.commit_prefix(&tail, 2).unwrap();
        assert!(left.is_complete());
        assert_eq!(left.committed_tokens(), &[1, 2, 4, 9]);
        assert!(left
            .commit_prefix(&tail, 2)
            .unwrap_err()
            .to_string()
            .contains("stale"));
    }

    #[test]
    fn exact_mask_grouping_and_sparse_threshold_are_explicit() {
        let masks = vec![
            ConstraintMask::new(vec![1, 2], VOCAB_SIZE).unwrap(),
            ConstraintMask::new(vec![2, 1], VOCAB_SIZE).unwrap(),
            ConstraintMask::new(vec![2], VOCAB_SIZE).unwrap(),
        ];
        assert_eq!(
            proven_identical_mask_groups(&masks),
            vec![vec![0, 1], vec![2]]
        );

        let policy = ConstraintSamplingPolicy::new(VOCAB_SIZE, 24_576).unwrap();
        let sparse = ConstraintMask::new((0..24_576).collect(), VOCAB_SIZE).unwrap();
        let dense = ConstraintMask::new((0..24_577).collect(), VOCAB_SIZE).unwrap();
        assert_eq!(
            policy.select(&sparse).unwrap(),
            ConstraintSamplingPath::SparseAllowedRows
        );
        assert_eq!(
            policy.select(&dense).unwrap(),
            ConstraintSamplingPath::DenseMaskedLogits
        );
    }

    #[test]
    fn mocked_streaming_and_non_streaming_tool_lifecycles_match() {
        let fragments = [
            "<tool_call>",
            "lookup<arg_key>query</arg_key>",
            "<arg_value>Taipei</arg_value>",
            "</tool_call>",
        ];
        let mut lifecycle = ConstrainedDecodeLifecycle::new(
            ScriptedMatcher::new(vec![vec![10, 20, 30, 99]]),
            ConstraintSamplingPolicy::new(VOCAB_SIZE, 24_576).unwrap(),
        );
        let draft = lifecycle.prepare_draft(&[10, 20, 30, 99]).unwrap();
        lifecycle.commit_prefix(&draft, 4).unwrap();
        assert!(lifecycle.is_complete());

        let tool = lookup_tool();
        let complete_text = fragments.concat();
        let non_streaming = parse_glm_tool_calls(&complete_text, std::slice::from_ref(&tool));
        assert_eq!(non_streaming.tool_calls.len(), 1);

        let mut parser = GlmToolCallStreamParser::new(vec![tool]);
        let mut deltas = Vec::new();
        for fragment in fragments {
            deltas.extend(parser.push(fragment));
        }
        deltas.extend(parser.finish());
        let streamed = stream_projection(&deltas);
        assert_eq!(parser.completed_tool_calls(), 1);
        assert_eq!(streamed.len(), 1);
        assert_eq!(streamed[0].0, non_streaming.tool_calls[0].function.name);
        assert_eq!(
            streamed[0].1,
            serde_json::from_str::<Value>(&non_streaming.tool_calls[0].function.arguments).unwrap()
        );
    }

    #[test]
    fn malformed_tokens_are_rejected_and_truncation_stays_incomplete() {
        let policy = ConstraintSamplingPolicy::new(VOCAB_SIZE, 24_576).unwrap();
        let matcher = ScriptedMatcher::new(vec![vec![10, 20, 30, 99]]);
        let mut lifecycle = ConstrainedDecodeLifecycle::new(matcher, policy);
        assert!(lifecycle
            .prepare_draft(&[10, 42])
            .unwrap_err()
            .to_string()
            .contains("rejected token"));
        assert!(lifecycle.committed_tokens().is_empty());

        let prefix = lifecycle.prepare_draft(&[10, 20, 30]).unwrap();
        lifecycle.commit_prefix(&prefix, 3).unwrap();
        assert!(!lifecycle.is_complete());

        let truncated = "<tool_call>lookup<arg_key>query</arg_key><arg_value>Taipei</arg_value>";
        let parsed = parse_glm_tool_calls(truncated, &[lookup_tool()]);
        assert!(parsed.tool_calls.is_empty());
        assert_eq!(parsed.content.as_deref(), Some(truncated));
        let mut parser = GlmToolCallStreamParser::new(vec![lookup_tool()]);
        parser.push(truncated);
        parser.finish();
        assert_eq!(parser.completed_tool_calls(), 0);
    }

    fn request_with_choice(choice: Option<Value>, include_tools: bool) -> ChatCompletionRequest {
        let mut value = json!({
            "model": "mock-full",
            "messages": [{"role": "user", "content": "use a tool"}]
        });
        if include_tools {
            value["tools"] = json!([
                {"type": "function", "function": {"name": "lookup"}},
                {"type": "function", "function": {"name": "other"}}
            ]);
        }
        if let Some(choice) = choice {
            value["tool_choice"] = choice;
        }
        serde_json::from_value(value).unwrap()
    }

    fn first_mask_for_request(request: &ChatCompletionRequest) -> ConstraintMask {
        let plan = ToolConstraintPlan::from_request(request).unwrap();
        let matcher = matcher_for_plan(&plan);
        let mut allowed = Vec::new();
        matcher.fill_allowed_token_ids(&mut allowed).unwrap();
        ConstraintMask::new(allowed, VOCAB_SIZE).unwrap()
    }

    fn matcher_for_plan(plan: &ToolConstraintPlan) -> ScriptedMatcher {
        let mut sequences = Vec::new();
        if matches!(
            plan.mode,
            ToolConstraintMode::Disabled | ToolConstraintMode::Auto
        ) {
            sequences.push(vec![1, 99]);
        }
        for name in &plan.allowed_tool_names {
            let sequence = match name.as_str() {
                "lookup" => vec![10, 20, 99],
                "other" => vec![11, 21, 99],
                _ => continue,
            };
            sequences.push(sequence);
        }
        ScriptedMatcher::new(sequences)
    }

    fn lookup_tool() -> ChatTool {
        ChatTool {
            tool_type: "function".to_owned(),
            function: ChatFunction {
                name: "lookup".to_owned(),
                description: None,
                parameters: Some(json!({
                    "type": "object",
                    "properties": {"query": {"type": "string"}}
                })),
            },
        }
    }

    fn stream_projection(deltas: &[GlmToolStreamDelta]) -> Vec<(String, Value)> {
        let mut calls = BTreeMap::<usize, (String, String)>::new();
        for delta in deltas {
            if let GlmToolStreamDelta::ToolCall {
                index,
                name,
                arguments,
                ..
            } = delta
            {
                let call = calls.entry(*index).or_default();
                if let Some(name) = name {
                    call.0.push_str(name);
                }
                if let Some(arguments) = arguments {
                    call.1.push_str(arguments);
                }
            }
        }
        calls
            .into_values()
            .map(|(name, arguments)| (name, serde_json::from_str(&arguments).unwrap()))
            .collect()
    }
}
