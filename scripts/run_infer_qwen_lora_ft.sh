#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

PY=/home/Hu_xuanwei/.conda/envs/qwen_vl/bin/python
ROOT=/home/Hu_xuanwei/aesthetic_eval_framework
BASE_CONFIG="${1:-$ROOT/configs/base.yaml}"

"$PY" "$ROOT/run.py" infer \
  --base-config "$BASE_CONFIG" \
  --model-config "$ROOT/configs/models/qwen3_vl_lora_ft.yaml"
