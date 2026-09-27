#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${AESTHETIC_PYTHON:-python}
MODEL_CONFIG=${1:?usage: run_artimuse_description.sh MODEL_CONFIG 0shot|16shot [RESUME_DIR]}
SHOT_MODE=${2:?usage: run_artimuse_description.sh MODEL_CONFIG 0shot|16shot [RESUME_DIR]}
RESUME_DIR=${3:-}

: "${AESTHETIC_DATA_ROOT:?set AESTHETIC_DATA_ROOT}"
: "${AESTHETIC_MODEL_ROOT:?set AESTHETIC_MODEL_ROOT}"
: "${AESTHETIC_OUTPUT_ROOT:?set AESTHETIC_OUTPUT_ROOT}"

case "$SHOT_MODE" in
  0shot) PROTOCOL="$ROOT_DIR/configs/protocols/artimuse_description_0shot.yaml" ;;
  16shot)
    : "${ARTIMUSE_CONTEXT_FILE:?set ARTIMUSE_CONTEXT_FILE to the locally rebuilt context JSON}"
    PROTOCOL="$ROOT_DIR/configs/protocols/artimuse_description_16shot.yaml"
    ;;
  *) echo "SHOT_MODE must be 0shot or 16shot" >&2; exit 2 ;;
esac

ARGS=(infer --base-config "$PROTOCOL" --model-config "$MODEL_CONFIG")
if [[ -n "$RESUME_DIR" ]]; then ARGS+=(--resume-dir "$RESUME_DIR"); fi
exec "$PYTHON_BIN" "$ROOT_DIR/run.py" "${ARGS[@]}"
