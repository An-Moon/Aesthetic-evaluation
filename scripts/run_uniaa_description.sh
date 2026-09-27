#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${AESTHETIC_PYTHON:-python}
MODEL_CONFIG=${1:?usage: run_uniaa_description.sh MODEL_CONFIG [RESUME_DIR]}
RESUME_DIR=${2:-}

: "${AESTHETIC_DATA_ROOT:?set AESTHETIC_DATA_ROOT}"
: "${AESTHETIC_MODEL_ROOT:?set AESTHETIC_MODEL_ROOT}"
: "${AESTHETIC_OUTPUT_ROOT:?set AESTHETIC_OUTPUT_ROOT}"
: "${UNIAA_1SHOT_CONTEXT_FILE:?set UNIAA_1SHOT_CONTEXT_FILE to the locally rebuilt context JSON}"

ARGS=(infer --base-config "$ROOT_DIR/configs/protocols/uniaa_description_1shot_greedy_v2.yaml" --model-config "$MODEL_CONFIG")
if [[ -n "$RESUME_DIR" ]]; then ARGS+=(--resume-dir "$RESUME_DIR"); fi
exec "$PYTHON_BIN" "$ROOT_DIR/run.py" "${ARGS[@]}"
