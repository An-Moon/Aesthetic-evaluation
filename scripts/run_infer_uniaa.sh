#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

PY=/home/Hu_xuanwei/.conda/envs/uniaa/bin/python
ROOT=/home/Hu_xuanwei/aesthetic_eval_framework

"$PY" "$ROOT/run.py" infer \
  --base-config "$ROOT/configs/base.yaml" \
  --model-config "$ROOT/configs/models/uniaa.yaml"
