#!/bin/bash

# Target directory
TARGET_DIR="/root/.cache/flexflow/weights/m4-ai/tinymistral-6x248m/full-precision"

# Loop through files containing "layer_"
for file in "$TARGET_DIR"/*layer_*; do
  if [[ -f "$file" ]]; then
    mv "$file" "${file//layer_/layer.}"
  fi
done

# Loop through files containing "_input"
for file in "$TARGET_DIR"/*_input*; do
  if [[ -f "$file" ]]; then
    mv "$file" "${file//_input/.input}"
  fi
done

# Loop through files containing ".mlp."
for file in "$TARGET_DIR"/*_mlp_*; do
  if [[ -f "$file" ]]; then
    mv "$file" "${file//.mlp./.mlp.}"
  fi
done

for file in "$TARGET_DIR"/*_post*; do
  if [[ -f "$file" ]]; then
    mv "$file" "${file//_post/.post}"
  fi
done


for file in "$TARGET_DIR"/*_block*; do
  if [[ -f "$file" ]]; then
    mv "$file" "${file//_block/.block}"
  fi
done

for file in "$TARGET_DIR"/*layers_*; do
  if [[ -f "$file" ]]; then
    mv "$file" "${file//layers_/layers.}"
  fi
done

for file in "$TARGET_DIR"/*proj_*; do
  if [[ -f "$file" ]]; then
    mv "$file" "${file//proj_/proj.}"
  fi
done

for file in "$TARGET_DIR"/*_self*; do
  if [[ -f "$file" ]]; then
    mv "$file" "${file//_self/.self}"
  fi
done

for file in "$TARGET_DIR"/*attn_*; do
  if [[ -f "$file" ]]; then
    mv "$file" "${file//attn_/attn.}"
  fi
done

for file in "$TARGET_DIR"/*embed_tokens_weight*; do
  if [[ -f "$file" ]]; then
    mv "$file" "${file//embed_tokens_weight/embed_tokens.weight}"
  fi
done

for file in "$TARGET_DIR"/*layernorm_weight*; do
  if [[ -f "$file" ]]; then
    mv "$file" "${file//layernorm_weight/layernorm.weight}"
  fi
done


for file in "$TARGET_DIR"/*_weight*; do
  if [[ -f "$file" ]]; then
    mv "$file" "${file//_weight/.weight}"
  fi
done