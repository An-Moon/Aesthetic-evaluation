# Aesthetic Eval Score Framework

Unified score-focused evaluation toolbox for MLLM aesthetic assessment.

This repository is separated from long-text VQA evaluation and focuses on
single-image aesthetic score regression. The design goal is simple: one CLI,
one auditable prediction format, one metric/report pipeline, while preserving
each model's most official scoring path when available.

## Features

- `infer-score`: run model inference and write unified `predictions.jsonl`.
- `eval-score`: compute PLCC, SRCC, KROCC, MAE, MSE, RMSE, plus error/parse stats.
- `validate-score`: smoke or AVA-8000 inference plus metrics in one command.
- `report`: aggregate metric files into a leaderboard JSON.
- Official-first adapters for ArtiMuse, ArtQuant, UniPercept, and OneAlign/Q-Align.
- Prompt or fallback scoring adapters for AesExpert, Qwen3VL, InternVL3.5, Q-SiT, and LLaVA-OneVision.

## Quick Start

```bash
git clone <repo-url>
cd aesthetic_eval_score_framework

python -m venv .venv
source .venv/bin/activate
pip install -r requirements-common.txt

cp .env.example .env
# Edit .env to point to your dataset, model checkpoints, and optional official repos.
set -a
source .env
set +a
```

Smoke test with an adapter available in your environment:

```bash
python run.py validate-score \
  --base-config configs/base_score_smoke_gpu3.yaml \
  --model-config configs/models/qwen3vl_4b_gpu3_smoke.yaml \
  --sample-size 4
```

Full AVA-8000 inference:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py infer-score \
  --base-config configs/base_score.yaml \
  --model-config configs/models/qwen3vl_4b.yaml
```

Evaluate a finished run:

```bash
python run.py eval-score \
  --pred-file outputs/<run_name>/predictions.jsonl \
  --output-file outputs/<run_name>/metrics.json
```

Build a leaderboard:

```bash
python run.py report \
  --metrics-glob "outputs/*/metrics.json" \
  --output-file outputs/leaderboard.json
```

If Hugging Face access is slow or blocked:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Data Format

The score dataset is a JSONL file. Each row should contain:

```json
{"sample_id": "0", "image": "000001.jpg", "gt_score": 5.42, "dataset": "AVA", "split": "test"}
```

`image` can be an absolute path or a path relative to `data.image_root`.

The default full config uses environment variables:

- `AVA_JSONL`: path to the full score JSONL.
- `AVA_IMAGE_ROOT`: image directory.
- `AES_OUTPUT_ROOT`: output directory, default `outputs`.

The smoke config defaults to `data/ava_smoke_64.jsonl`, which is only a format
example. Set `AVA_SMOKE_JSONL` and `AVA_IMAGE_ROOT` to a real local subset before
running smoke inference.

## Output Protocol

Each prediction row includes:

- `sample_id`, `image`, `image_resolved`, `gt_score`
- `raw_score`, `score_0_10`, `raw_response`
- `parse_status`, `error`
- `score_source`, `score_method`, `official_alignment`, `adapter_version`
- `normalization`, `model`, `task`, `timestamp_utc`

Parsing failures are not converted to midpoint scores. They are saved as
`parse_status=parse_failed` with `score_0_10=null`; `eval-score` skips them in
correlation/error metrics and reports `parse_failed_count`.

## Score Methods

- `official_score`: direct model score method, e.g. ArtiMuse or UniPercept `score()`.
- `official_logits_wa5`: official five-level logits/probability mapping, e.g. ArtQuant, OneAlign/Q-Align, Q-SiT.
- `prompt_numeric`: generic MLLM prompt asking for one numeric score.
- `fallback_logits_wa5`: auditable five-level logits mapping when no official regression scorer was found.

All comparable scores are mapped to `0-10`, while raw scores and mapping metadata
are kept in every row.

## Model Status

| Model | Config | Method | External repo |
| --- | --- | --- | --- |
| ArtiMuse | `configs/models/artimuse.yaml` | `official_score` | `ARTIMUSE_REPO` |
| ArtQuant | `configs/models/artquant.yaml` | `official_logits_wa5` | `ARTQUANT_REPO` |
| AesExpert | `configs/models/aesexpert.yaml` | `fallback_logits_wa5` | `LLAVA_REPO` |
| UniPercept | `configs/models/unipercept.yaml` | `official_score` | `UNIPERCEPT_REPO` |
| OneAlign / Q-Align | `configs/models/qalign.yaml` | `official_logits_wa5` | `QALIGN_REPO` |
| Q-SiT | `configs/models/qsit.yaml` | `official_logits_wa5` | none |
| Qwen3VL | `configs/models/qwen3vl_4b.yaml` | `prompt_numeric` | none |
| InternVL3.5 | `configs/models/internvl35_8b.yaml` | `prompt_numeric` | none |
| LLaVA-OneVision 1.5 | `configs/models/llava_onevision_8b.yaml` | `prompt_numeric` | none |

See `docs/EXTERNAL_DEPENDENCIES.md` for details. We intentionally do not vendor
full official repositories or model checkpoints.

## Environment Policy

The toolbox keeps one CLI and one output format, but not every upstream MLLM can
realistically share one Python environment. Older official LLaVA/Q-Align code and
newer LLaVA-OneVision/Qwen3 code may require incompatible Transformers versions.

Recommended dependency files:

- `requirements-common.txt`: CLI, data, metrics, report only.
- `requirements-llava37.txt`: ArtiMuse, ArtQuant, AesExpert, UniPercept, OneAlign/Q-Align.
- `requirements-qwen-internvl.txt`: Qwen3VL and InternVL3.5.
- `requirements-onevision.txt`: Q-SiT and LLaVA-OneVision 1.5.

Use the same `run.py` command in each environment; only the Python executable
changes.

## Local Path Variables

All model and official-repo paths can be configured through environment variables
or by editing the YAML files. Common variables:

- `ARTIMUSE_REPO`, `ARTIMUSE_MODEL`
- `ARTQUANT_REPO`, `ARTQUANT_WEIGHTS`, `MPLUG_OWL2_MODEL`
- `LLAVA_REPO`, `AESEXPERT_MODEL`
- `UNIPERCEPT_REPO`, `UNIPERCEPT_MODEL`
- `QALIGN_REPO`, `ONEALIGN_MODEL`
- `QSIT_MODEL`
- `QWEN3VL_MODEL`, `INTERNVL35_MODEL`, `LLAVA_ONEVISION_MODEL`

The YAML loader supports `${VAR}` and `${VAR:-default}` forms.

## Validation Policy

For each adapter:

1. Run 4 to 64 smoke samples.
2. Confirm `error_count=0` and parse failure rate is reasonable.
3. Run the fixed AVA-8000 manifest.
4. Compare against official or prior baseline where available.

Use `sample_manifest_out` once to freeze a subset, then `sample_manifest_in` for
future model runs.
