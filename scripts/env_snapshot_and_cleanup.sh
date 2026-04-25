#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN=${CONDA_BIN:-/opt/miniconda3/bin/conda}
ROOT_DIR=${ROOT_DIR:-/home/Hu_xuanwei/aesthetic_eval_framework}
SNAP_ROOT="$ROOT_DIR/env_snapshots"
TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$SNAP_ROOT/$TS"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <env1> [env2 ...]"
  echo "Example: $0 qwen_vl llava"
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "[INFO] snapshot dir: $OUT_DIR"

for ENV_NAME in "$@"; do
  echo "[INFO] export env: $ENV_NAME"
  "$CONDA_BIN" env export -n "$ENV_NAME" --no-builds > "$OUT_DIR/${ENV_NAME}.environment.yml"
  "$CONDA_BIN" list -n "$ENV_NAME" --explicit > "$OUT_DIR/${ENV_NAME}.explicit.txt"
  "$CONDA_BIN" run -n "$ENV_NAME" pip freeze > "$OUT_DIR/${ENV_NAME}.pip-freeze.txt"
done

echo "[INFO] exported files:"
ls -lh "$OUT_DIR"

for ENV_NAME in "$@"; do
  echo "[INFO] remove env: $ENV_NAME"
  "$CONDA_BIN" remove -n "$ENV_NAME" --all -y
done

echo "[DONE] cleanup finished"
echo "[INFO] remaining envs:"
"$CONDA_BIN" env list
