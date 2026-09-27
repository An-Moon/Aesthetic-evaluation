#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${AESTHETIC_PYTHON:-python}
MODEL_CONFIG=${1:?usage: run_uniaa_perception.sh MODEL_CONFIG [RESUME_DIR]}
RESUME_DIR=${2:-}

: "${AESTHETIC_DATA_ROOT:?set AESTHETIC_DATA_ROOT}"
: "${AESTHETIC_MODEL_ROOT:?set AESTHETIC_MODEL_ROOT}"
: "${AESTHETIC_OUTPUT_ROOT:?set AESTHETIC_OUTPUT_ROOT}"

ARGS=(infer --base-config "$ROOT_DIR/configs/protocols/uniaa_perception_official_v1.yaml" --model-config "$MODEL_CONFIG")
if [[ -n "$RESUME_DIR" ]]; then ARGS+=(--resume-dir "$RESUME_DIR"); fi
exec "$PYTHON_BIN" "$ROOT_DIR/run.py" "${ARGS[@]}"
