use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};

static TOKENIZER_CACHE: OnceLock<RwLock<HashMap<PathBuf, Arc<LoadedTokenizer>>>> = OnceLock::new();

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerEncodingSummary {
    pub tokenizer_path: String,
    pub text: String,
    pub add_special_tokens: bool,
    pub token_count: usize,
    pub first_token_id: Option<u32>,
    pub last_token_id: Option<u32>,
    pub token_ids: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerDecodeSummary {
    pub tokenizer_path: String,
    pub token_ids: Vec<u32>,
    pub skip_special_tokens: bool,
    pub text: String,
}

pub struct LoadedTokenizer {
    tokenizer_path: String,
    tokenizer: tokenizers::Tokenizer,
}

/// Stateful decoder for token-by-token generation.
///
/// Byte-level BPE tokenizers can split one UTF-8 scalar across multiple token
/// IDs. Decoding each ID independently turns every partial scalar into U+FFFD.
/// This keeps the bounded prefix state used by `tokenizers::DecodeStream`
/// without borrowing the tokenizer, so it can live for an API request.
pub struct StreamingTokenDecoder {
    tokenizer: Arc<LoadedTokenizer>,
    skip_special_tokens: bool,
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}

impl StreamingTokenDecoder {
    pub fn step(&mut self, token_id: u32) -> Result<Option<String>> {
        tokenizers::tokenizer::step_decode_stream(
            &self.tokenizer.tokenizer,
            vec![token_id],
            self.skip_special_tokens,
            &mut self.ids,
            &mut self.prefix,
            &mut self.prefix_index,
        )
        .map_err(|err| anyhow::anyhow!("stream-decoding tokenizer token {token_id}: {err}"))
    }
}

impl LoadedTokenizer {
    pub fn from_snapshot(snapshot_path: &Path) -> Result<Self> {
        let tokenizer_path = snapshot_path.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|err| {
            anyhow::anyhow!("loading tokenizer {}: {err}", tokenizer_path.display())
        })?;
        Ok(Self {
            tokenizer_path: tokenizer_path.display().to_string(),
            tokenizer,
        })
    }

    pub fn encode_text(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<TokenizerEncodingSummary> {
        let encoding = self
            .tokenizer
            .encode(text.to_owned(), add_special_tokens)
            .map_err(|err| anyhow::anyhow!("encoding tokenizer text: {err}"))?;
        let token_ids = encoding.get_ids().to_vec();
        Ok(TokenizerEncodingSummary {
            tokenizer_path: self.tokenizer_path.clone(),
            text: text.to_owned(),
            add_special_tokens,
            token_count: token_ids.len(),
            first_token_id: token_ids.first().copied(),
            last_token_id: token_ids.last().copied(),
            token_ids,
        })
    }

    pub fn decode_ids(
        &self,
        token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<TokenizerDecodeSummary> {
        let text = self
            .tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|err| anyhow::anyhow!("decoding tokenizer ids: {err}"))?;
        Ok(TokenizerDecodeSummary {
            tokenizer_path: self.tokenizer_path.clone(),
            token_ids: token_ids.to_vec(),
            skip_special_tokens,
            text,
        })
    }
}

fn cached_tokenizer(snapshot_path: &Path) -> Result<Arc<LoadedTokenizer>> {
    let cache = TOKENIZER_CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    let key = snapshot_path.to_path_buf();
    if let Some(tokenizer) = cache
        .read()
        .map_err(|_| anyhow::anyhow!("tokenizer cache read lock is poisoned"))?
        .get(&key)
        .cloned()
    {
        return Ok(tokenizer);
    }

    // Hold the write lock while constructing the first tokenizer for this snapshot.
    // Otherwise a simultaneous C4 admission can redundantly parse tokenizer.json four
    // times before any caller populates the cache.
    let mut cache = cache
        .write()
        .map_err(|_| anyhow::anyhow!("tokenizer cache write lock is poisoned"))?;
    if let Some(tokenizer) = cache.get(&key).cloned() {
        return Ok(tokenizer);
    }
    let tokenizer = Arc::new(LoadedTokenizer::from_snapshot(snapshot_path)?);
    cache.insert(key, Arc::clone(&tokenizer));
    Ok(tokenizer)
}

pub fn encode_tokenizer_text(
    snapshot_path: &Path,
    text: &str,
    add_special_tokens: bool,
) -> Result<TokenizerEncodingSummary> {
    cached_tokenizer(snapshot_path)?.encode_text(text, add_special_tokens)
}

pub fn decode_tokenizer_ids(
    snapshot_path: &Path,
    token_ids: &[u32],
    skip_special_tokens: bool,
) -> Result<TokenizerDecodeSummary> {
    cached_tokenizer(snapshot_path)?.decode_ids(token_ids, skip_special_tokens)
}

pub fn streaming_token_decoder(
    snapshot_path: &Path,
    skip_special_tokens: bool,
) -> Result<StreamingTokenDecoder> {
    Ok(StreamingTokenDecoder {
        tokenizer: cached_tokenizer(snapshot_path)?,
        skip_special_tokens,
        ids: Vec::new(),
        prefix: String::new(),
        prefix_index: 0,
    })
}
