#!/usr/bin/env bash
set -euo pipefail

url="${1:-http://127.0.0.1:8000}"
model="${2:-glmrt-synthetic-glm-layer}"
prompt_tokens="${3:-16}"

if ! [[ "$prompt_tokens" =~ ^[0-9]+$ ]] || [ "$prompt_tokens" -lt 1 ]; then
  echo "PROMPT_TOKENS must be a positive integer" >&2
  exit 2
fi

content_words=$((prompt_tokens - 1))
prompt=""
for idx in $(seq 1 "$content_words"); do
  if [ -n "$prompt" ]; then
    prompt+=" "
  fi
  prompt+="tok${idx}"
done

response="$(
  curl -fsS "$url/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -cn \
      --arg model "$model" \
      --arg content "$prompt" \
      '{model:$model,messages:[{role:"user",content:$content}],max_tokens:128}')"
)"

echo "$response" | jq .

actual_prefill_rows="$(echo "$response" | jq -r '.metrics.layerwave_prefill_rows')"
actual_chunks="$(echo "$response" | jq -r '.metrics.prefill_chunk_count')"
backend_mode="$(echo "$response" | jq -r '.metrics.backend_mode')"
ttft_ms="$(echo "$response" | jq -r '.metrics.time_to_first_token_ms')"
prefill_ms="$(echo "$response" | jq -r '.metrics.prefill_ms')"

if [ "$actual_prefill_rows" -ne "$prompt_tokens" ]; then
  echo "expected layerwave_prefill_rows=$prompt_tokens, got $actual_prefill_rows" >&2
  exit 1
fi
if [ "$actual_chunks" -lt 1 ]; then
  echo "expected at least one prefill chunk, got $actual_chunks" >&2
  exit 1
fi
if [ "$backend_mode" != "synthetic-glm-layer" ] && [ "$backend_mode" != "tiny" ]; then
  echo "unexpected backend_mode=$backend_mode" >&2
  exit 1
fi
awk -v value="$ttft_ms" 'BEGIN { exit !(value >= 0.0) }'
awk -v value="$prefill_ms" 'BEGIN { exit !(value >= 0.0) }'
