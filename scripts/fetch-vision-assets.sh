#!/usr/bin/env bash
set -euo pipefail

model_id=baseten/GLM-5.2-Vision-NVFP4
revision=f6eab6117386a0c69152fdf272dc65bfd0254f9f
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"

files=(
  README.md
  chat_template.jinja
  config.json
  configuration_glm5v.py
  generation_config.json
  kimi_k25_processor.py
  kimi_k25_vision_processing.py
  media_utils.py
  mm_projector.safetensors
  model.safetensors.index.json
  preprocessor_config.json
  tokenizer.json
  tokenizer_config.json
  vision_tower.safetensors
  "plugins/**"
)

exec uvx --from huggingface-hub hf download \
  "$model_id" \
  "${files[@]}" \
  --revision "$revision" \
  --cache-dir "$hf_home/hub"
