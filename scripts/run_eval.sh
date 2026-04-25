#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <predictions.jsonl> <metrics_output.json>"
  exit 1
fi

PY=/home/Hu_xuanwei/.conda/envs/qwen_vl/bin/python
ROOT=/home/Hu_xuanwei/aesthetic_eval_framework

"$PY" "$ROOT/run.py" eval \
  --pred-file "$1" \
  --output-file "$2"
