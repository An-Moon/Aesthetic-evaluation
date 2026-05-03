#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python run.py infer-score \
  --base-config configs/base_score.yaml \
  --model-config configs/models/artimuse.yaml
