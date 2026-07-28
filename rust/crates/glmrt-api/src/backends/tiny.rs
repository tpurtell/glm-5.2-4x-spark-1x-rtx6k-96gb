use glmrt_core::{
    deterministic_tiny_completion, plan_prefill_chunks, DecodeStep, KvCacheAllocator,
    KvCacheConfig, LayerWave, PrefillChunkPolicy, Priority,
};
use std::time::Instant;

use crate::metrics::BackendMetrics;
use crate::{duration_ms, BackendCompletion};

pub(crate) fn tiny_backend_completion(
    prompt: &str,
    prompt_tokens: usize,
    max_tokens: usize,
) -> BackendCompletion {
    let prefill_tokens = prompt_tokens.max(1);
    let mut kv_allocator = KvCacheAllocator::new(KvCacheConfig::glm52_phase0(prefill_tokens + 1));
    let kv_reservation_id = kv_allocator
        .reserve("tiny", prefill_tokens + 1)
        .expect("tiny prefill reservation fits in tiny KV allocator");
    let policy = PrefillChunkPolicy::latency_smoke(16);

    let prefill_start = Instant::now();
    let prefill_rows = plan_prefill_chunks(
        "tiny-prefill",
        "tiny",
        0,
        prefill_tokens,
        kv_reservation_id,
        Priority(10),
        &policy,
        "phase0-tiny",
    )
    .into_iter()
    .map(LayerWave::prefill)
    .map(|wave| wave.num_rows())
    .sum::<usize>();
    let prefill_ms = duration_ms(prefill_start.elapsed());

    let decode_start = Instant::now();
    let decode_wave = LayerWave::decode(DecodeStep::new(
        "tiny-decode",
        "tiny",
        0,
        prefill_tokens as u64,
        Some(kv_reservation_id),
        Priority(0),
        "phase0-tiny",
    ));
    let content = deterministic_tiny_completion(prompt, max_tokens);
    let decode_ms = duration_ms(decode_start.elapsed());

    BackendCompletion {
        content,
        reasoning_content: None,
        completion_tokens: None,
        stream_chunks: None,
        metrics: BackendMetrics {
            cache_load_ms: 0.0,
            prefill_ms,
            time_to_first_token_ms: None,
            decode_ms,
            reasoning_tokens: 0,
            cached_prompt_tokens: 0,
            prefill_tokens,
            prefill_chunk_count: prefill_rows.div_ceil(policy.chunk_tokens),
            layerwave_prefill_rows: prefill_rows,
            layerwave_decode_rows: decode_wave.num_rows(),
            real_full: None,
        },
    }
}
