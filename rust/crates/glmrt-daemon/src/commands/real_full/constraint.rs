use anyhow::{Context, Result};
use glmrt_api::{RealFullConstraint, RealFullConstraintGrammar};
use glmrt_ffi::{
    GlmrtXGrammarCompiler, GlmrtXGrammarGrammar, GlmrtXGrammarMatcher, GLMRT_XGRAMMAR_JSON_OBJECT,
    GLMRT_XGRAMMAR_JSON_SCHEMA, GLMRT_XGRAMMAR_STRUCTURAL_TAG,
};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use super::coordinator_kernels::cuda_native_library;

const GLM_CHAT_GRAMMAR_STOP_TOKEN_IDS: &[i32] = &[154_820, 154_827, 154_829];
const MAX_CACHED_CONSTRAINT_GRAMMARS: usize = 64;

struct ConstraintCompilerInner {
    compiler: Option<GlmrtXGrammarCompiler<'static>>,
    grammars: HashMap<RealFullConstraint, Arc<GlmrtXGrammarGrammar<'static>>>,
    grammar_order: VecDeque<RealFullConstraint>,
}

pub(super) struct RealFullConstraintCompiler {
    tokenizer_json_path: PathBuf,
    vocab_size: usize,
    inner: Mutex<ConstraintCompilerInner>,
}

impl RealFullConstraintCompiler {
    pub(super) fn new(tokenizer_json_path: PathBuf, vocab_size: usize) -> Result<Self> {
        anyhow::ensure!(
            tokenizer_json_path.is_file(),
            "real-full constrained decoding tokenizer is missing: {}",
            tokenizer_json_path.display()
        );
        anyhow::ensure!(vocab_size > 0, "real-full constraint vocabulary is empty");
        Ok(Self {
            tokenizer_json_path,
            vocab_size,
            inner: Mutex::new(ConstraintCompilerInner {
                compiler: None,
                grammars: HashMap::new(),
                grammar_order: VecDeque::new(),
            }),
        })
    }

    pub(super) fn matcher(&self, spec: Arc<RealFullConstraint>) -> Result<RealFullConstraintState> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|error| anyhow::anyhow!("locking constraint compiler failed: {error}"))?;
        if inner.compiler.is_none() {
            let library = cuda_native_library()
                .context("loading native library for real-full constrained decoding")?;
            inner.compiler = Some(
                library
                    .xgrammar_compiler(
                        &self.tokenizer_json_path,
                        self.vocab_size,
                        GLM_CHAT_GRAMMAR_STOP_TOKEN_IDS,
                    )
                    .context("creating cached real-full XGrammar compiler")?,
            );
        }
        let grammar = if let Some(grammar) = inner.grammars.get(spec.as_ref()).cloned() {
            grammar
        } else {
            let compiler = inner
                .compiler
                .as_ref()
                .expect("constraint compiler initialized above");
            let grammar = match &spec.grammar {
                RealFullConstraintGrammar::Json => compiler
                    .compile(GLMRT_XGRAMMAR_JSON_OBJECT, None, true)
                    .context("compiling JSON-object response grammar")?,
                RealFullConstraintGrammar::JsonSchema {
                    schema_json,
                    strict,
                } => compiler
                    .compile(GLMRT_XGRAMMAR_JSON_SCHEMA, Some(schema_json), *strict)
                    .context("compiling JSON Schema response grammar")?,
                RealFullConstraintGrammar::StructuralTag {
                    structural_tag_json,
                } => compiler
                    .compile(
                        GLMRT_XGRAMMAR_STRUCTURAL_TAG,
                        Some(structural_tag_json),
                        true,
                    )
                    .context("compiling strict tool structural-tag grammar")?,
            };
            let grammar = Arc::new(grammar);
            if inner.grammars.len() >= MAX_CACHED_CONSTRAINT_GRAMMARS {
                while let Some(evicted) = inner.grammar_order.pop_front() {
                    if inner.grammars.remove(&evicted).is_some() {
                        break;
                    }
                }
            }
            inner
                .grammars
                .insert(spec.as_ref().clone(), Arc::clone(&grammar));
            inner.grammar_order.push_back(spec.as_ref().clone());
            grammar
        };
        let matcher = grammar
            .matcher()
            .context("creating real-full constraint matcher")?;
        Ok(RealFullConstraintState {
            spec,
            matcher,
            vocab_size: self.vocab_size,
            bitmask_words: self.vocab_size.div_ceil(32),
        })
    }
}

pub(super) struct RealFullConstraintMasks {
    pub(super) rows: usize,
    pub(super) words_per_row: usize,
    pub(super) words: Vec<u32>,
}

impl RealFullConstraintMasks {
    pub(super) fn allows(&self, row: usize, token_id: usize) -> bool {
        if row >= self.rows {
            return false;
        }
        let word = token_id / 32;
        if word >= self.words_per_row {
            return false;
        }
        let value = self.words[row * self.words_per_row + word];
        value & (1_u32 << (token_id % 32)) != 0
    }
}

pub(super) struct RealFullConstraintState {
    spec: Arc<RealFullConstraint>,
    matcher: GlmrtXGrammarMatcher<'static>,
    vocab_size: usize,
    bitmask_words: usize,
}

pub(super) struct RealFullConstraintBranch {
    matcher: GlmrtXGrammarMatcher<'static>,
    vocab_size: usize,
    bitmask_words: usize,
}

impl RealFullConstraintState {
    pub(super) fn matches_spec(&self, spec: &Arc<RealFullConstraint>) -> bool {
        self.spec.as_ref() == spec.as_ref()
    }

    pub(super) fn valid_draft_prefix(&self, draft_token_ids: &[usize]) -> Result<usize> {
        let mut branch = self
            .matcher
            .fork()
            .context("forking constraint matcher for speculative proposal validation")?;
        if branch
            .is_completed()
            .context("checking speculative proposal grammar frontier")?
        {
            return Ok(0);
        }
        for (index, token_id) in draft_token_ids.iter().copied().enumerate() {
            let token_id = u32::try_from(token_id)
                .context("speculative proposal token exceeds XGrammar token range")?;
            if !branch
                .accept_token(token_id)
                .context("validating speculative proposal against grammar")?
            {
                return Ok(index);
            }
            // Keep the token that completes the JSON/tag payload, but never
            // accept a speculative stop token or anything after it. The
            // target's final verification row will select the stop token.
            if branch
                .is_completed()
                .context("checking completed speculative proposal grammar")?
            {
                return Ok(index + 1);
            }
        }
        Ok(draft_token_ids.len())
    }

    pub(super) fn draft_branch(
        &self,
        committed_prefix: &[usize],
    ) -> Result<RealFullConstraintBranch> {
        let mut branch = RealFullConstraintBranch {
            matcher: self
                .matcher
                .fork()
                .context("forking constraint matcher for native MTP drafts")?,
            vocab_size: self.vocab_size,
            bitmask_words: self.bitmask_words,
        };
        branch
            .accept(committed_prefix)
            .context("advancing native MTP grammar branch through emitted target tokens")?;
        Ok(branch)
    }

    pub(super) fn masks_for_draft(
        &self,
        draft_token_ids: &[usize],
    ) -> Result<RealFullConstraintMasks> {
        let rows = draft_token_ids
            .len()
            .checked_add(1)
            .context("constraint speculative row count overflow")?;
        let values = rows
            .checked_mul(self.bitmask_words)
            .context("constraint bitmask size overflow")?;
        let mut words = vec![0_u32; values];
        let mut branch = self
            .matcher
            .fork()
            .context("forking constraint matcher for target verification masks")?;
        for row in 0..rows {
            let row_mask = &mut words[row * self.bitmask_words..(row + 1) * self.bitmask_words];
            let needs_mask = branch
                .fill_bitmask(row_mask)
                .context("filling target constraint bitmask")?;
            if !needs_mask {
                row_mask.fill(u32::MAX);
                if self.vocab_size % 32 != 0 {
                    *row_mask
                        .last_mut()
                        .expect("nonempty constraint bitmask row") &=
                        (1_u32 << (self.vocab_size % 32)) - 1;
                }
            }
            if let Some(token_id) = draft_token_ids.get(row).copied() {
                let word = token_id / 32;
                let allowed =
                    word < self.bitmask_words && row_mask[word] & (1_u32 << (token_id % 32)) != 0;
                anyhow::ensure!(
                    allowed,
                    "speculative proposal token {token_id} is not allowed by row {row} grammar state"
                );
                let token_id = u32::try_from(token_id)
                    .context("speculative proposal token exceeds XGrammar token range")?;
                anyhow::ensure!(
                    branch
                        .accept_token(token_id)
                        .context("advancing target verification grammar branch")?,
                    "XGrammar rejected a proposal after its bitmask allowed it"
                );
            }
        }
        Ok(RealFullConstraintMasks {
            rows,
            words_per_row: self.bitmask_words,
            words,
        })
    }

    pub(super) fn commit(&mut self, token_ids: &[usize]) -> Result<()> {
        for token_id in token_ids.iter().copied() {
            let token_id = u32::try_from(token_id)
                .context("committed constrained token exceeds XGrammar token range")?;
            anyhow::ensure!(
                self.matcher
                    .accept_token(token_id)
                    .context("committing generated token to XGrammar matcher")?,
                "generated token {token_id} was not accepted by the authoritative grammar state"
            );
        }
        Ok(())
    }
}

impl RealFullConstraintBranch {
    pub(super) fn is_completed(&self) -> Result<bool> {
        self.matcher
            .is_completed()
            .context("checking native MTP grammar branch completion")
    }

    pub(super) fn next_mask(&mut self) -> Result<RealFullConstraintMasks> {
        let mut words = vec![0_u32; self.bitmask_words];
        let needs_mask = self
            .matcher
            .fill_bitmask(&mut words)
            .context("filling native MTP draft constraint bitmask")?;
        if !needs_mask {
            words.fill(u32::MAX);
            if self.vocab_size % 32 != 0 {
                *words
                    .last_mut()
                    .expect("nonempty native MTP constraint bitmask") &=
                    (1_u32 << (self.vocab_size % 32)) - 1;
            }
        }
        Ok(RealFullConstraintMasks {
            rows: 1,
            words_per_row: self.bitmask_words,
            words,
        })
    }

    pub(super) fn accept(&mut self, token_ids: &[usize]) -> Result<()> {
        for token_id in token_ids.iter().copied() {
            let grammar_token = u32::try_from(token_id)
                .context("native MTP draft token exceeds XGrammar token range")?;
            anyhow::ensure!(
                self.matcher
                    .accept_token(grammar_token)
                    .context("advancing native MTP draft grammar branch")?,
                "native MTP draft token {token_id} was rejected by its grammar branch"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::RealFullConstraintMasks;

    #[test]
    fn packed_constraint_masks_are_row_scoped() {
        let masks = RealFullConstraintMasks {
            rows: 2,
            words_per_row: 2,
            words: vec![1 << 3, 0, 0, 1 << 1],
        };
        assert!(masks.allows(0, 3));
        assert!(!masks.allows(0, 33));
        assert!(masks.allows(1, 33));
        assert!(!masks.allows(2, 3));
    }
}
